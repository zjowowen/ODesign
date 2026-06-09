# ODesign Module-Level Profiling 合同

本文定义 E1/E2 之后的下一轮 profiling：把已经定位到的 `forward/backward` 主瓶颈继续拆到模型模块级别。本文是执行合同，不是结果报告；它不支持 PBP 质量结论，也不直接证明任何配置能更快达到 PBP target。

## 背景

已完成证据：

- E1 2GPU baseline 50-update：`returncode=0`，`250 microbatches/rank`，checkpoint `49.pt`。
- E1 stage-level timing：单 microbatch 约 `52.16s`，其中 `forward ~= 18.73s`，`backward ~= 30.35s`，二者合计约 `94.1%`。
- E2：`empty_cache` 只带来约 `1%` 级别收益；`diffusion_lddt_chunk_size=2` 不加速且显著增显存。
- stage memory profiling：峰值主要发生在大 padded atom 样本的 `loss/backward`，不是 optimizer/checkpoint/eval。

因此下一步不应继续盲目扩大 E1，也不应优先调 dataloader 或 `empty_cache`。需要回答的是：`forward/backward` 内部到底由哪些模块和 checkpoint/recompute 路径主导。

## 目标问题

本轮必须回答：

1. `forward` 的约 `18.7s` 里，Pairformer / Pairwise head / Diffusion denoising / Permutation 分别耗时多少。
2. `backward` 的约 `30.3s` 里，哪些 forward 模块在 autograd 中贡献最大的 backward 时间。
3. `exp.model.blocks_per_ckpt=1` 是否可能过度保守，是否值得进入 `1 -> 2 -> 4 -> None` 的配置 sweep。

本轮不回答：

- 不证明任何 checkpoint PBP 达标。
- 不比较不同 GPU 卡数或节点数下的主指标。
- 不把短跑 step time 改善等同于 `time_to_target_checkpoint` 改善。

## 资源和数据合同

默认复用 E1/E2 的 2GPU H200 执行面：

| 项目 | 值 |
| --- | --- |
| pod | `zjow-odesign-pbp-efficiency-2gpu-4607978-8rw86` |
| GPU | `2 x NVIDIA H200` |
| runtime root | `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign` |
| data root | `/mnt/shared-storage-user/ai4sreason/scireason_2/data/odesign` |
| base checkpoint | `/mnt/shared-storage-user/ai4sreason/scireason_2/data/odesign/ckpt/protenix_base_default_v0.5.0.pt` |
| train config | `train_odesign_base_prot_flex_weighted_before20210930_reso_below4_16gpu_tokenbalanced` |
| train batch size | `2` |
| gradient accumulation | `5` |
| crop size | `640` |
| diffusion batch size | `48` |
| diffusion lDDT chunk size | `1` |
| dataloader workers | `4` |

如果正式 profiling runtime 仍在跑其它任务，必须复制到独立 runtime 后再做代码同步，避免污染正式/历史记录目录。

## Instrumentation 设计

### 训练 loop 已有字段

继续保留 E1 字段：

- `data_wait_sec`
- `to_device_sec`
- `forward_sec`
- `loss_sec`
- `backward_sec`
- `optimizer_sec`
- `empty_cache_sec`
- `microbatch_total_sec`
- GPU allocator fields
- loss 和 batch shape/padding fields

### 新增 forward 子阶段字段

在 profiling 打开且环境变量启用时，记录以下 `ODesign.forward(mode="train")` 子阶段：

| 字段 | 对应代码路径 | 含义 |
| --- | --- | --- |
| `module_profile_prepare_inputs_sec` | `ODesign.forward -> prepare_training_inputs` | Feature/label wrapper 构造 |
| `module_profile_pairformer_sec` | `ODesign.main_train_loop -> get_pairformer_output` | input embedding、MSA、PairformerStack、多 cycle |
| `module_profile_pairwise_head_sec` | `ODesign.main_train_loop -> pairwise_head` | distogram/bond head |
| `module_profile_diffusion_sec` | `ODesign.main_train_loop -> sample_diffusion_training` | diffusion training denoising |
| `module_profile_permutation_sec` | `ODesign.main_train_loop -> symmetric_permutation` | diffusion sample symmetry matching |

这些字段回答 forward 内部主耗时。

### 新增 backward 近似字段

第一版使用 PyTorch module `full_backward_hook` 记录模块级 backward wall time：

| 字段 | 模块 |
| --- | --- |
| `module_profile_backward_pairformer_sec` | `model.pairformer_stack` |
| `module_profile_backward_msa_sec` | `model.msa_module` |
| `module_profile_backward_pairwise_head_sec` | `model.pairwise_head` |
| `module_profile_backward_diffusion_module_sec` | `model.diffusion_module` |

边界：

- hook 时间是 autograd 调用模块 backward hook 的 wall time 近似，不等同于 CUDA kernel 精准归因。
- 若 activation checkpoint 导致 forward recompute，recompute 可能表现为 backward 阶段里的 forward 子模块执行时间。第一版要在报告中明确这一点。
- 若 hook 归因不稳定，下一步升级为 `torch.profiler` 或更细粒度 NVTX/PyTorch profiler trace。

## 环境变量

新增开关默认关闭：

```text
ODESIGN_PROFILE_MODULES=1
```

默认行为必须保持不变。只有同时满足以下条件才记录 module profile：

- `ODESIGN_PROFILE_JSONL` 已设置。
- 当前 rank 会写 profile。
- `ODESIGN_PROFILE_MODULES=1`。

可选：

```text
ODESIGN_PROFILE_MODULE_BACKWARD=1
```

默认开启或关闭可由实现决定，但必须在 profile record 中写出 `profile_module_backward`，避免事后误读。

## 第一轮实验

建议 run：

```text
effprof_2gpu_module_profile_20260609_r1
```

长度：

- smoke：`2 optimizer updates / 10 microbatches`，只验证字段和无崩溃。
- 正式短跑：`10 optimizer updates / 50 microbatches`，用于第一张 module-level 图。

为了先回答“模块耗时在哪”，建议第一轮保持：

- `ODESIGN_PROFILE_SYNC_CUDA=1`
- `ODESIGN_PROFILE_STAGE_CUDA_PEAKS=1`
- `ODESIGN_PROFILE_MODULES=1`
- `ODESIGN_PROFILE_ALL_RANKS=1`
- 不改 `blocks_per_ckpt`。

## 成功标准

smoke 成功：

- `returncode=0`
- 两个 rank 都写出 profile JSONL。
- 至少一条 row 含 `module_profile_pairformer_sec`、`module_profile_diffusion_sec`、`module_profile_pairwise_head_sec`、`module_profile_permutation_sec`。
- loss finite。
- 训练代码 hash 和文档记录一致。

正式短跑成功：

- `returncode=0`
- `50 rows/rank`
- 生成 module-level summary JSON/Markdown。
- 生成 module-level timing 图。
- 报告中区分 verified evidence 和 unverified boundary。

## 后续决策

若 Pairformer forward/backward 或 MSA 占比最高：

- 进入 `exp.model.blocks_per_ckpt` sweep：`1 -> 2 -> 4 -> None`。
- 每个候选必须记录 speed + memory + loss finite。

若 Diffusion module 占比最高：

- 检查 diffusion transformer checkpoint granularity。
- 评估 `diffusion_batch_size` 只作为非等价探索，不能直接 claim 质量不变。

若 Pairwise head 或 loss/backward 显存峰值主导：

- 继续 dense/sparse/hybrid lDDT 或 atom-pair chunk 方向，但必须先做数值一致性测试。

若 module hook 不能解释 backward：

- 升级到 PyTorch profiler/NVTX trace，而不是继续猜配置。
