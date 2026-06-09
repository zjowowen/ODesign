# ODesign Operator-Level Profiling 结果

本文记录 2026-06-09 在用户提供的 2GPU H200 长时容器中完成的 operator-level profiling。它承接 `docs/odesign_operator_level_acceleration_plan_20260609.zh.md`，目标是在不改变训练 setting 的前提下，把 Pairformer/MSA 主瓶颈进一步拆到 `record_function` range、PyTorch op 和 CUDA kernel 粒度。

本文是归因报告，不是加速结论，不支持 checkpoint 质量结论。

## 结论

本次真实 2GPU trace run `operator_profile_2gpu_1upd_20260609_r1` 成功完成：

| item | value |
| --- | --- |
| returncode | `0` |
| wall time | `567s` |
| resource | `2 x H200`，pod `zjow-odesign-pbp-efficiency-2gpu-4607978-8rw86` |
| profile rows | `5 rows/rank`, `10 total` |
| torch trace | `1` 个 rank0 trace，约 `2.07GB` |
| trace parser | `scripts/summarize_torch_trace.py`，29 秒完成标准库流式解析 |
| output checkpoint | `outputs/operator_profile_2gpu_1upd/operator_profile_2gpu_1upd_20260609_r1/checkpoints/0.pt` |

主要观察：

- 真实生产配置下 Pairformer triangle attention 确认进入 DeepSpeed Evo branch：trace 中 `odesign.openfold_attention.deepspeed_evo` 出现 `3024` 次，`odesign.openfold_attention.stock` 为 `0`。
- Evo attention 的 GPU kernel 也被捕获：`attention_kernel_batched_impl<AttentionKernel<...>>` 出现 `1512` 次，GPU kernel total 约 `6687ms`；attention backward kernel 出现 `216` 次，约 `3433ms`。
- `record_function` 的 `user_annotation` range 显示最重的子路径集中在 MSA stack、triangle multiplication、attention pair bias、triangle attention 和 transition。
- 全局 top event 里有大量 `aten::copy_`、`aten::to`、`aten::_to_copy`、`aten::item`、layer norm kernel、NCCL all-reduce 和 `cudaStreamSynchronize`。这些是下一步排查 runtime/layout/sync 的候选，但还不能直接 claim 某个优化一定有效。

直接决策：

- 不应把“切到 DeepSpeed Evo attention”作为优化候选，因为本 run 已证明生产路径已经走 Evo attention。
- 第一批候选应优先围绕 triangle multiplication、attention bias/mask/layout、transition/gate/layernorm/copy 路径做等价优化和同步审计。
- 任何候选优化进入 speed gate 前，必须先通过固定 batch 的 forward/loss/grad/参数更新 allclose gate。

## 运行合同

| item | value |
| --- | --- |
| run id | `operator_profile_2gpu_1upd_20260609_r1` |
| record dir | `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/module-profile-runtime-0609/ODesign/.cluster_operator/operator_profile_2gpu_1upd_20260609_r1` |
| train config | `train_odesign_base_prot_flex_weighted_before20210930_reso_below4_16gpu_tokenbalanced` |
| data root | `/mnt/shared-storage-user/ai4sreason/scireason_2/data/odesign` |
| base checkpoint | `/mnt/shared-storage-user/ai4sreason/scireason_2/data/odesign/ckpt/protenix_base_default_v0.5.0.pt` |
| launch | `torchrun --nnodes=1 --nproc_per_node=2` |
| max steps | `1 optimizer update` |
| train batch size | `2` |
| gradient accumulation | `5` |
| crop size | `640` |
| diffusion batch size | `48` |
| diffusion lDDT chunk size | `1` |
| train set limit | `256` |
| dataloader workers | `4` |
| DeepSpeed Evo attention | `true` |
| TF32 override | `NVIDIA_TF32_OVERRIDE=1` |
| empty cache | `ODESIGN_EMPTY_CACHE_POLICY=never` |

Profiler env：

| env | value |
| --- | --- |
| `ODESIGN_PROFILE_MODULES` | `1` |
| `ODESIGN_PROFILE_MODULE_BACKWARD` | `0` |
| `ODESIGN_PROFILE_PAIRFORMER_DETAIL` | `1` |
| `ODESIGN_PROFILE_SYNC_CUDA` | `0` |
| `ODESIGN_TORCH_PROFILER_ALL_RANKS` | `0` |
| `ODESIGN_TORCH_PROFILER_WAIT/WARMUP/ACTIVE/REPEAT` | `1/1/2/1` |
| `ODESIGN_TORCH_PROFILER_RECORD_SHAPES` | `1` |
| `ODESIGN_TORCH_PROFILER_PROFILE_MEMORY` | `1` |
| `ODESIGN_TORCH_PROFILER_WITH_STACK` | `0` |
| `ODESIGN_TORCH_PROFILER_WITH_MODULES` | `1` |

完整命令记录在：

```text
/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/module-profile-runtime-0609/ODesign/.cluster_operator/operator_profile_2gpu_1upd_20260609_r1/command.txt
```

## 代码版本

本地可审阅分支：

| item | value |
| --- | --- |
| branch | `feature/odesign-padding-batch` |
| fork commit | `fe1862c420a458660e243c06e9b77fe6bdb9baa6` |
| commit title | `feat: add operator-level profiling ranges` |

远端运行目录位于 `.cluster_operator` 复制树里，父目录 Git 元数据不代表该复制树代码版本。因此本 run 以关键文件 sha256 锚定代码内容；这些 hash 与本地 `fe1862c` 工作树一致：

| file | sha256 |
| --- | --- |
| `src/utils/train/train_runner.py` | `e271c752fe25edca031772d6baf3a70a7a9a35138d656d31cefa6c611f34f37b` |
| `src/utils/model/profiling.py` | `ff2dccf1431554529e48cc4d0cb0a9232eb886f5c4b7725499f48467ad170790` |
| `src/model/modules/pairformer.py` | `cdc28dd65a61265d0c54206ec593d7120f3bfd9b947782c02366ca4d0e94ef7d` |
| `src/utils/openfold_local/model/primitives.py` | `13fc349715ea8fa06a9d60dee937ff442eb4f053fbd72633fd05704934303a87` |
| `src/utils/openfold_local/model/triangular_attention.py` | `37cd58c811c838eee5060048b6fd2c10cd37cc213d208c0a8e6e90d73a062ccf` |
| `src/utils/openfold_local/model/triangular_multiplicative_update.py` | `bdbc9c4016d15004be17aa7efd5541f9e4bf1f3b6af054f4835e292ad33d66f4` |

## Artifact

| artifact | path |
| --- | --- |
| command | `${RECORD_DIR}/command.txt` |
| env | `${RECORD_DIR}/env.txt` |
| stdout/stderr | `${RECORD_DIR}/stdout_stderr.log` |
| JSONL profile rows | `${RECORD_DIR}/profile_steps_rank00.jsonl`, `${RECORD_DIR}/profile_steps_rank01.jsonl` |
| module summary | `${RECORD_DIR}/module_profile_summary.json`, `${RECORD_DIR}/module_profile_summary.svg` |
| torch trace | `${RECORD_DIR}/torch_profiler/rank00/*.pt.trace.json` |
| light trace counts | `${RECORD_DIR}/light_trace_counts.txt` |
| parsed trace summary | `${RECORD_DIR}/torch_trace_summary.json`, `${RECORD_DIR}/torch_trace_summary.md` |

## Module-Level 同步观察

本 run 只有 `1` 个 optimizer update，且 profiler active window 会带来显著诊断开销。因此下面数据只用于解释本 trace，不用于和无 profiler 训练吞吐直接比较。

统计口径：每个 rank 丢弃第一条 cold-start row 后合并，得到 `8` 条 rows。

| field | mean | p50 | p90 | max |
| --- | ---: | ---: | ---: | ---: |
| `forward_sec` | `16.533s` | `19.929s` | `21.267s` | `21.354s` |
| `backward_sec` | `27.355s` | `26.285s` | `31.267s` | `34.967s` |
| `loss_sec` | `0.067s` | `0.063s` | `0.086s` | `0.103s` |
| `microbatch_total_sec` | `72.454s` | `46.537s` | `113.708s` | `252.377s` |
| `module_profile_pairformer_sec` | `13.293s` | `16.176s` | `17.286s` | `17.331s` |
| `module_profile_diffusion_sec` | `2.958s` | `3.399s` | `3.689s` | `3.736s` |

解读：

- Pairformer 仍是 forward 主成本：`13.293s / 16.533s = 80.4%`。
- `microbatch_total_sec` 的 max 很高，主要因为 torch profiler active/export 对个别 microbatch 造成额外开销；不要把它当作真实训练 step time。

## ODesign Range 表

下面表来自 `scripts/summarize_torch_trace.py`，只统计 `category=user_annotation` 的 `odesign.*` ranges，避免把 `user_annotation` 和 `gpu_user_annotation` 双计。

| range | count | total | mean | max |
| --- | ---: | ---: | ---: | ---: |
| `odesign.msa_block.msa_stack` | `42` | `7164.781ms` | `170.590ms` | `383.888ms` |
| `odesign.msa_module.blocks` | `12` | `5563.573ms` | `463.631ms` | `774.440ms` |
| `odesign.pairformer_block.tri_mul_in` | `756` | `4992.818ms` | `6.604ms` | `28.306ms` |
| `odesign.triangle_multiplication.projections` | `1512` | `4939.581ms` | `3.267ms` | `27.437ms` |
| `odesign.pairformer_block.tri_mul_out` | `756` | `4765.570ms` | `6.304ms` | `19.467ms` |
| `odesign.triangle_multiplication.output` | `1512` | `3628.668ms` | `2.400ms` | `10.405ms` |
| `odesign.pairformer_block.attention_pair_bias` | `672` | `3618.846ms` | `5.385ms` | `21.056ms` |
| `odesign.pairformer_block.tri_att_start` | `756` | `3000.397ms` | `3.969ms` | `27.838ms` |
| `odesign.triangle_attention.mha` | `1512` | `2228.876ms` | `1.474ms` | `17.666ms` |
| `odesign.pairformer_block.pair_transition` | `756` | `2080.693ms` | `2.752ms` | `26.973ms` |
| `odesign.transition.projections` | `2004` | `1757.836ms` | `0.877ms` | `26.422ms` |
| `odesign.pairformer_block.tri_att_end` | `756` | `1422.225ms` | `1.881ms` | `27.726ms` |
| `odesign.triangle_attention.triangle_bias` | `1512` | `1359.276ms` | `0.899ms` | `26.345ms` |
| `odesign.transition.gate_output` | `2004` | `1353.543ms` | `0.675ms` | `7.475ms` |
| `odesign.openfold_attention.prep_qkv` | `1512` | `1322.795ms` | `0.875ms` | `12.888ms` |
| `odesign.pairformer_block.single_transition` | `672` | `1304.859ms` | `1.942ms` | `7.590ms` |

注意：`odesign.openfold_attention.deepspeed_evo` 的 `user_annotation` CPU span 只有 `368.220ms`，但对应 `gpu_user_annotation` 和 kernel 显示 GPU 侧约 `6.7s`。因此 attention kernel 成本不能只看 CPU range。

## Top Kernel / Op 观察

下面是 `torch_trace_summary.json` 中按 total 排序后的主要 kernel/op。它们包含 forward、backward、DDP 通信和 profiler active window 内的所有事件。

### CUDA Kernel

| event | count | total | mean | note |
| --- | ---: | ---: | ---: | --- |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL` | `110` | `27327.690ms` | `248.434ms` | DDP all-reduce，包含通信/同步等待 |
| `vectorized_layer_norm_kernel<bf16,float>` | `8564` | `7110.687ms` | `0.830ms` | layer norm forward |
| `GammaBetaBackwardCUDAKernel<float,float>` | `1392` | `6941.902ms` | `4.987ms` | layer norm gamma/beta backward |
| `attention_kernel_batched_impl<AttentionKernel<...>>` | `1512` | `6687.042ms` | `4.423ms` | DeepSpeed Evo attention forward kernel |
| `vectorized_layer_norm_kernel<float,float>` | `9306` | `6450.064ms` | `0.693ms` | layer norm forward |
| `elementwise_kernel<...direct_copy...>` | `13314` | `4750.472ms` | `0.357ms` | copy/cast/materialization |
| `GammaBetaBackwardCUDAKernel_32x32<float,float>` | `1200` | `3873.282ms` | `3.228ms` | layer norm backward |
| `attention_kernel_backward_batched_impl<...>` | `216` | `3433.336ms` | `15.895ms` | attention backward kernel |

### CPU / Runtime Op

| event | count | total | mean | note |
| --- | ---: | ---: | ---: | --- |
| `aten::copy_` | `110840` | `30409.530ms` | `0.274ms` | 可能对应 dtype/layout/materialize |
| `cudaLaunchKernel` | `384147` | `28377.335ms` | `0.074ms` | kernel launch overhead aggregate |
| `cudaStreamSynchronize` | `15019` | `26649.504ms` | `1.774ms` | 同步/等待，需排查是否由 `.item()` 或 profiler 引入 |
| `aten::to` | `78647` | `23505.810ms` | `0.299ms` | dtype/device/layout 转换 |
| `aten::_to_copy` | `66247` | `23455.924ms` | `0.354ms` | copy path |
| `MmBackward0` | `12310` | `22708.095ms` | `1.845ms` | matmul backward |
| `aten::mm` | `78840` | `22356.230ms` | `0.284ms` | matmul |
| `aten::item` / `aten::_local_scalar_dense` | `17548` each | about `15.9s` | `0.905ms` | 可能触发同步，优先审计 |
| `aten::linear` | `66994` | `15725.801ms` | `0.235ms` | linear projection |
| `aten::matmul` | `55276` | `10588.822ms` | `0.192ms` | matmul wrapper |

## 怎么理解这些结果

1. `Pairformer` 仍然是主前向成本，但 Pairformer 内部不是单一 attention 问题。
   `tri_mul_in/out`、triangle multiplication projection/output、attention pair bias 和 triangle attention 都在前排。

2. DeepSpeed Evo attention 已经生效。
   之前的疑问是 Pairformer openfold-local attention 是否真的走 Evo branch。本 trace 同时看到 `odesign.openfold_attention.deepspeed_evo`、Evo attention forward kernel 和 attention backward kernel，且 stock branch 为 0。

3. 大量 copy/to/item/synchronize 需要单独审计。
   这些 op 可能来自正常 dtype/layout 转换、mask/bias 构造、日志取 scalar、profiler record_shapes/profile_memory，也可能暴露实际同步热点。下一步不能直接删除它们，而要先定位来源并做 allclose gate。

4. 本 trace 文件很大，record_shapes/profile_memory 也有开销。
   因此它适合做首次归因，不适合做精确 speed benchmark。正式比较优化收益时应关掉 profiler，保留同一训练 setting，并另跑无 profiler speed run。

## 已补工具

新增 `scripts/summarize_torch_trace.py`：

- 标准库实现，不依赖 `ijson`、pandas 或 tensorboard。
- 按行流式扫描 PyTorch Chrome trace，不把 2GB JSON 全量载入内存。
- 默认只把 `category=user_annotation` 的 `odesign.*` range 汇总到 range 表，避免和 `gpu_user_annotation` 双计。
- 同时输出全局 top event 表，保留 kernel、cpu_op、cuda_runtime、gpu_user_annotation 视角。

验证：

| gate | evidence |
| --- | --- |
| local unit test | `python3 tests/test_torch_trace_summary.py`: `Ran 2 tests OK` |
| local py_compile | `python3 -m py_compile scripts/summarize_torch_trace.py tests/test_torch_trace_summary.py` |
| remote unit test | H200 runtime `python3 tests/test_torch_trace_summary.py`: `Ran 2 tests OK` |
| real trace parse | `2.07GB` trace，`4226492` X events，`26706` matched ODesign user ranges，29 秒完成 |

## 边界

- 本文没有 claim 任何训练速度提升。
- 本文没有 claim checkpoint 质量、PBP 成绩或 time-to-target 改善。
- `record_function` ranges 是 CPU/user annotation span，不等价于 self CUDA kernel time；attention 的 GPU 成本需要看 `gpu_user_annotation` 和 kernel 表。
- `torch profiler` 的 `record_shapes=1`、`profile_memory=1` 会放大 trace 体积和运行开销；这次结果用于 attribution，不用于最终吞吐对比。
- `NCCL all_reduce` 出现在 top kernel 中，但本 run 只有 2GPU/短 active window，不能直接外推到 16GPU 长训瓶颈。

## 下一步

建议按以下顺序继续：

1. 补一个轻量 r2 trace：`ACTIVE=1`，`RECORD_SHAPES=0`，`PROFILE_MEMORY=0`，训练 setting 不变。目标是生成更小 trace，核对 r1 的热点排序是否稳定。
2. 审计 `aten::item` / `_local_scalar_dense` 来源，判断是否来自训练日志/跳坏样本/条件分支/profiler，而不是核心数学路径。
3. 审计 `aten::to` / `_to_copy` / `copy_` 来源，优先检查 Pairformer triangle attention 的 mask/bias、transpose、contiguous、dtype 转换。
4. 从一个低风险局部候选开始，例如 attention bias/mask materialization 或 transition gate 局部 fusion；先做 G1 单模块 allclose，再做 G2 真实 batch 单 step allclose。
5. 数值通过后再做无 profiler speed run，比较同一训练 setting 下的 mean/p50/p90 和 peak memory。
