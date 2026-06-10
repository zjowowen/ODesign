# ODesign 训练显存峰值 Stage Profiling 记录

本文记录 2026-06-09 对 ODesign 正式 padded-batch 训练“平均显存较低但瞬时峰值较高”的诊断。本文只支持训练显存峰值来源和短跑控制实验结论，不支持 PBP 质量结论，也不证明 16GPU 长训的最终收敛质量。

## 结论摘要

一句话判断：当前观察到的显存峰值是“符合当前模型实现和训练配置预期的瞬时峰值”，不是显存泄漏，也不是 checkpoint 保存、eval 或 optimizer 阶段导致的异常峰值。

具体含义：

- 它是正常现象：大 padded atom 样本进入 dense lDDT loss 和 backward 时，会短时间申请很大的 atom-pair 临时张量。
- 它不是常驻显存：最大样本中 `backward` peak 约 `108.8GB`，但 backward 结束后 allocated 回落到约 `7.8GB`。
- 它不是 optimizer/checkpoint/eval 峰值：optimizer stage 最大只有约 `9.1GB`，与 `loss/backward` 的百 GB 级峰值不在同一量级。
- 它仍然需要监控：正常不等于没有风险。如果长训采到更极端的 padded atom 组合，或者 allocator reserved/cache 被外部采样计入，仍可能接近 H200 上限甚至 OOM。

本次控制实验给出的最重要证据：

| 配置 | 最大 PyTorch allocated peak | 解释 |
| --- | ---: | --- |
| `diffusion_batch_size=48` + dense lDDT | `108.8GB` | 当前 baseline，超大 atom-pair 样本峰值最高 |
| `diffusion_batch_size=24` + dense lDDT | `72.7GB` | 同样样本序列下，降低 diffusion sample 轴显著降峰值 |
| `diffusion_batch_size=48` + sparse lDDT | `68.1GB` | 保持 sample 轴不变，移除 dense atom-pair lDDT 后大样本峰值显著下降 |

因此，当前峰值的主因是：

```text
大 padded atom 数
  x diffusion_batch_size=48 的 N_sample 轴
  x dense lDDT 中的 N_atom x N_atom atom-pair 计算
  x backward/recompute
```

## 问题

正式 16GPU 训练中，外部 `nvidia-smi` 采样显示：

- p50 显存常在几十 GB。
- p95 明显更高。
- 个别采样能冲到 H200 接近满卡区间。

需要回答：

1. 峰值发生在训练 step 的哪个阶段。
2. 峰值是否来自 checkpoint/eval/optimizer，还是来自 forward/loss/backward 的临时张量。
3. 哪些配置变量对峰值最敏感。

## 诊断代码

在 `src/utils/train/train_runner.py` 增加默认关闭的 stage CUDA peak profiling：

```text
ODESIGN_PROFILE_STAGE_CUDA_PEAKS=1
```

生效条件：

- `ODESIGN_PROFILE_JSONL` 已设置。
- 当前 rank 会写 profile。
- CUDA 可用且 device 是 CUDA。

记录阶段：

- `to_device`
- `forward`
- `loss`
- `backward`
- `optimizer`
- `empty_cache`

每个阶段记录：

- start/end allocated MiB
- start/end reserved MiB
- peak allocated MiB
- peak reserved MiB

注意：stage profiling 会在每个阶段边界执行 `torch.cuda.synchronize()` 和 `torch.cuda.reset_peak_memory_stats()`，因此它是诊断模式，不应用作正式训练吞吐配置。

## 测试验证

H200 `odesign` conda 环境中执行：

```text
python -m unittest discover -s tests -p test_train_runner_efficiency_controls.py
```

结果：

```text
Ran 5 tests in 0.003s
OK
```

覆盖内容：

- 默认 `empty_cache` policy 行为。
- 非法 `ODESIGN_EMPTY_CACHE_POLICY` 报错。
- 未传 `profile_record` 时 stage profiling 不启用。
- fake CUDA monkeypatch 验证 stage begin/end 会 reset/sync 并写 peak 字段。

## 实验合同

所有实验均使用同一个诊断 runtime，避免影响正式训练目录：

```text
/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/memstage-runtime-0609/ODesign
```

统一资源：

| 项目 | 值 |
| --- | --- |
| GPU | `1 node x 2 H200` |
| CPU / memory | `32 CPU / 256GiB` |
| hostNetwork | `false` |
| train batch size | `2` |
| gradient accumulation | `5` |
| max steps | `5 optimizer steps / 25 microbatches` |
| train crop size | `640` |
| base checkpoint | `/mnt/shared-storage-user/ai4sreason/scireason_2/data/odesign/ckpt/protenix_base_default_v0.5.0.pt` |
| data root | `/mnt/shared-storage-user/ai4sreason/scireason_2/data/odesign` |

## 实验结果

### E3A：baseline，dbsz48 + dense lDDT

| 项目 | 值 |
| --- | --- |
| H200 job | `zjow-odesign-memstage-2gpu-0609-r2` |
| run id | `memstage_2gpu_bsz2_gacc5_0609_r2` |
| `diffusion_batch_size` | `48` |
| `diffusion_lddt_chunk_size` | `1` |
| `diffusion_lddt_loss_dense` | 默认 true |
| returncode | `0` |
| profile rows | `25 x 2 ranks` |

Stage peak 统计：

| stage | p50 allocated | p95 allocated | max allocated |
| --- | ---: | ---: | ---: |
| forward | `37995.6 MiB` | `46174.8 MiB` | `53043.9 MiB` |
| loss | `40115.8 MiB` | `86982.6 MiB` | `104477.7 MiB` |
| backward | `46516.9 MiB` | `90318.6 MiB` | `108784.9 MiB` |
| optimizer | `8630.6 MiB` | `9143.2 MiB` | `9143.2 MiB` |
| empty_cache | `7123.6 MiB` | `7397.9 MiB` | `7740.8 MiB` |

50 个 microbatch 中，top stage 全部是 `backward`。

最大样本：

```text
rank1 global_step=24 pdb_id=["3ccs", "3hlj"]
num_atoms_padded=23916
forward peak=48363.1 MiB
loss peak=104477.7 MiB
backward peak=108784.9 MiB
```

该样本的 start/end/peak 显示，`loss` 结束时 allocated 已到约 `99370 MiB`，`backward` 瞬时冲到约 `108785 MiB`，随后结束后回落到约 `7787 MiB`。这解释了为什么外部采样会看到平均显存不高但存在短时高峰。

### E3B：dbsz24 + dense lDDT

| 项目 | 值 |
| --- | --- |
| H200 job | `zjow-odesign-memstage-bsz24-2gpu-0609-r1-75958557` |
| run id | `memstage_2gpu_bsz2_gacc5_dbsz24_0609_r1` |
| changed variable | `diffusion_batch_size=24` |
| returncode | `0` |
| profile rows | `25 x 2 ranks` |

Stage peak 统计：

| stage | p50 allocated | p95 allocated | max allocated |
| --- | ---: | ---: | ---: |
| forward | `31630.7 MiB` | `38515.0 MiB` | `43817.2 MiB` |
| loss | `31361.8 MiB` | `57752.8 MiB` | `68440.1 MiB` |
| backward | `40159.2 MiB` | `61083.1 MiB` | `72740.7 MiB` |

与 baseline 按相同 rank/profile 行/pdb 匹配：

| 样本 | padded atoms | baseline max | dbsz24 max | ratio |
| --- | ---: | ---: | ---: | ---: |
| `["3ccs", "3hlj"]` | `23916` | `108784.9 MiB` | `72740.7 MiB` | `0.669` |
| `["5xy3", "6x3t"]` | `20716` | `97544.1 MiB` | `68660.9 MiB` | `0.704` |
| `["4ip7", "6zlw"]` | `20880` | `90318.6 MiB` | `61083.1 MiB` | `0.676` |

结论：`diffusion_batch_size` 对峰值有强影响，尤其是大 atom-pair 样本。降到 24 后，最坏样本峰值约下降 30% 以上。

### E3C：dbsz48 + sparse lDDT

| 项目 | 值 |
| --- | --- |
| H200 job | `zjow-odesign-memstage-sparse-2gpu-0609-r1-56996017` |
| run id | `memstage_2gpu_bsz2_gacc5_sparselddt_0609_r1` |
| changed variable | `exp.loss.diffusion_lddt_loss_dense=false` |
| `diffusion_batch_size` | `48` |
| returncode | `0` |
| profile rows | `25 x 2 ranks` |

Stage peak 统计：

| stage | p50 allocated | p95 allocated | max allocated |
| --- | ---: | ---: | ---: |
| forward | `37995.6 MiB` | `46174.8 MiB` | `53043.9 MiB` |
| loss | `36675.0 MiB` | `60599.9 MiB` | `68100.9 MiB` |
| backward | `46516.9 MiB` | `59287.6 MiB` | `66354.0 MiB` |

与 baseline 按相同 rank/profile 行/pdb 匹配：

| 样本 | padded atoms | baseline max | sparse max | ratio |
| --- | ---: | ---: | ---: | ---: |
| `["3ccs", "3hlj"]` | `23916` | `108784.9 MiB` | `68100.9 MiB` | `0.626` |
| `["5xy3", "6x3t"]` | `20716` | `97544.1 MiB` | `64414.5 MiB` | `0.660` |
| `["4ip7", "6zlw"]` | `20880` | `90318.6 MiB` | `60599.9 MiB` | `0.671` |
| `["4g17", "6fu2"]` | `7984` | `54717.9 MiB` | `54717.9 MiB` | `1.000` |

结论：sparse lDDT 对超大 atom-pair 样本显著降峰值，但对小/中样本不一定改变总体 peak，因为这些样本的峰值仍可能由 forward/backward transformer 路径主导。

## 根因判断

当前证据支持以下结论：

1. 这是当前实现下的正常瞬时峰值，不是泄漏。最大样本在 `backward` 结束后 allocated 从约 `108.8GB` 峰值回落到约 `7.8GB`。
2. 峰值主要不是 checkpoint 保存、eval 或 optimizer 造成的。`optimizer` stage max 只有约 `9.1 GiB`。
3. 峰值主要来自大 padded atom shape 下的 `loss` 和 `backward`。
4. dense smooth lDDT 会在 `SmoothLDDTLoss.dense_forward()` 中构造 `torch.cdist(pred_coordinate, pred_coordinate)`，形状含 `N_sample x N_atom x N_atom`。
5. `diffusion_lddt_chunk_size=1` 已经沿 `N_sample` 切 chunk，但每个 chunk 仍然保留 `N_atom x N_atom` 二次项，因此不能消除大 atom-pair 峰值。
6. `diffusion_batch_size=48` 还会放大 diffusion denoising 的样本轴，使 forward/backward 基线显存维持在较高水平。
7. `torch.cuda.empty_cache()` 只能影响 reserved/cache 行为，对 PyTorch allocated peak 的根因帮助有限。

如何理解“正常”：

- 如果显存峰值只在大样本的 `loss/backward` 中短时出现，之后 allocated 回落，并且没有随 step 单调增长，这是当前配置下可解释、可复现的行为。
- 如果显存持续不回落、每个 step 单调增大、出现在 optimizer/checkpoint/eval 阶段，或伴随 OOM/NaN/非零退出，那才应按异常处理。
- 当前 2GPU 诊断和控制实验支持第一种解释，不支持泄漏或保存异常解释。

相关代码路径：

- `src/model/modules/loss.py`：`SmoothLDDTLoss.dense_forward()` 中 dense `torch.cdist`。
- `src/model/modules/loss.py`：`ODesignLoss.update_label()` 中构造 batched atom-pair `lddt_mask/distance_mask`。
- `src/model/modules/generator.py`：训练 diffusion 生成 `[*batch_prefix, N_sample, N_atom, 3]`。
- `src/model/modules/diffusion.py`：denoise net 按 `N_sample` 扩展 token/pair embedding 并参与 forward/backward。

## 目前建议

短期建议：

1. 正式 16GPU 训练如果没有 OOM，不建议立刻中断；当前现象更像短生命周期大张量峰值，不像泄漏。
2. 若需要提高安全余量，最直接的开关是把 `diffusion_batch_size` 从 48 降到 24，但这改变训练 loss 的 diffusion sample 数，应做 PBP/训练质量确认。
3. 更有针对性的优化是实现“dense/sparse hybrid lDDT”：小样本走 dense，大 padded atom 样本走 sparse，并先用单元测试验证 dense 与 sparse 在同一 `lddt_mask` 上的数值一致性。
4. 另一个方向是对 dense lDDT 做 atom-pair 维度 chunk，而不是只沿 `N_sample` chunk；这能保持 dense 语义，但实现风险比直接 sparse/hybrid 更高。

不建议优先投入：

- 单纯调整 `empty_cache`。它对 wall time 有小幅影响，但不是 allocated peak 根因。
- 只调 `diffusion_lddt_chunk_size`。当前已经是 1，不能继续降低样本轴 chunk。

## 未验证边界

- 本文的 stage peak 结论来自 2GPU、5 optimizer step 的短诊断，不等价于完整 16GPU 长训统计。
- sparse lDDT 是否可作为正式训练替代，需要额外数值等价测试和 PBP 质量验证。
- 降低 `diffusion_batch_size` 的训练质量影响未在本文验证。
- 16GPU 正式任务未启用 `ODESIGN_PROFILE_STAGE_CUDA_PEAKS`，因此本文把 2GPU 诊断用于解释机制，而不是宣称已逐步观测 16GPU 的每个 stage。
