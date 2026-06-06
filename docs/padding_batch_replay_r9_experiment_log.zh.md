# ODesign Padding Batch Replay r9 实验日志

本文记录 `bsz2` vs `bsz1` replay integration test 的 r9 实验细节、控制变量、运行脚本、结果解释和重复实验设计。该日志用于审计实验真实性和复现实验条件。

## 结论边界

r9 证明的是：

- 在 `SMOKE_MODE=1`、单节点 4GPU DDP、真实 ODesign 数据、真实 `35999.pt` checkpoint、固定 sample trace、固定/禁用主要随机源、no-TF32/fp32 路径下，`batch_size_per_device=2, grad_accum=5` 与 `batch_size_per_device=1, grad_accum=10` 在 3 个 optimizer update 内训练集成层面等价。
- 通过口径是主阈值 `ATOL/RTOL=5e-4` 和 diagnostic 阈值 `DIAGNOSTIC_ATOL/RTOL=1e-4`。
- r9 不证明 bitwise 等价，不证明 `1e-6` 级 strict replay 等价，不证明 16GPU 跨节点正式 world size 已通过，也不证明正式长训收敛和 PBP/ODesignBench 指标已通过。

## 实验身份

| 项目 | 值 |
| --- | --- |
| run id | `replay_bsz2_bsz1_4gpu_notf32_pod_0606_r9_diag1e4` |
| 运行容器 | `zjow-sci2-bs-pbp-4gpu-clone20260606162737-59667537-2g4sk` |
| run dir | `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign/.cluster_operator/replay_bsz2_bsz1_4gpu_notf32_pod_0606_r9_diag1e4` |
| root checkout | `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign` |
| runner | `.cluster_operator/run_replay_padding_batch_integration_probe_0605.sh` |
| probe | `.cluster_operator/replay_padding_batch_integration_probe_0605.py` |
| summary | `${RUN_DIR}/outputs/summary.json` and `${RUN_DIR}/summary.json` |
| stdout log | `${RUN_DIR}/stdout_stderr_node0.log` |
| env snapshot | `${RUN_DIR}/preflight_env.txt`, `${RUN_DIR}/env_node0.txt` |
| GPU memory log | `${RUN_DIR}/gpu_memory_node0.csv` |
| return code | `${RUN_DIR}/returncode_node0.txt = 0` |

此前主验证文档中 r9 摘要的 commit：

- local commit: `816a4473bb675e2e020deae65458ac9a7e6090a1`
- fork API commit: `d05ea43acb906ada317a71db958e617882fc4098`

## 数据和 checkpoint

| 资产 | 路径 |
| --- | --- |
| data root | `/mnt/shared-storage-user/ai4sreason/scireason_2/data/odesign` |
| ckpt root | `/mnt/shared-storage-user/ai4sreason/scireason_2/data/odesign/ckpt` |
| resume checkpoint | `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_3/SciReasoner-2/reference/ODesign/outputs/train_odesign_base_prot_flex_weighted_before20210930_reso_below4_16gpu_tokenbalanced/2026-04-26_14-44-00/checkpoints/35999.pt` |
| exp config | `train_odesign_base_prot_flex_weighted_before20210930_reso_below4_16gpu_tokenbalanced` |

## 运行配置

`launch_env.sh` 中固定的关键环境变量：

```bash
export SMOKE_MODE=1
export NNODES=1
export NPROC_PER_NODE=4
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29509
export UPDATES=3
export TRACE_SEED=20260605
export ATOL=5e-4
export RTOL=5e-4
export DIAGNOSTIC_ATOL=1e-4
export DIAGNOSTIC_RTOL=1e-4
export SAVE_GRAD_TENSORS=false
export DISABLE_PERMUTATION=true
export DETERMINISTIC_MSA_ROWS=1
export DENSE_LDDT_MAX_ATOMS=0
export USE_DEEPSPEED_EVO_ATTENTION=false
export MODEL_DTYPE=fp32
export NVIDIA_TF32_OVERRIDE=0
export DISABLE_TF32=true
export LOG_GPU_MEMORY=true
```

实际执行命令由 runner 组装为：

```bash
torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=29509 \
  ./.cluster_operator/replay_padding_batch_integration_probe_0605.py \
  --output-dir /mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign/.cluster_operator/replay_bsz2_bsz1_4gpu_notf32_pod_0606_r9_diag1e4/outputs \
  --data-root-dir /mnt/shared-storage-user/ai4sreason/scireason_2/data/odesign \
  --ckpt-root-dir /mnt/shared-storage-user/ai4sreason/scireason_2/data/odesign/ckpt \
  --resume-ckpt /mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_3/SciReasoner-2/reference/ODesign/outputs/train_odesign_base_prot_flex_weighted_before20210930_reso_below4_16gpu_tokenbalanced/2026-04-26_14-44-00/checkpoints/35999.pt \
  --exp-config train_odesign_base_prot_flex_weighted_before20210930_reso_below4_16gpu_tokenbalanced \
  --updates 3 \
  --trace-seed 20260605 \
  --atol 5e-4 \
  --rtol 5e-4 \
  --diagnostic-atol 1e-4 \
  --diagnostic-rtol 1e-4 \
  --deterministic-msa-rows 1 \
  --dense-lddt-max-atoms 0 \
  --forward-probe-sample-pos -1 \
  --forward-probe-sampled-elements 4096 \
  --disable-permutation
```

## 代码快照

`env_node0.txt` 记录的运行时关键文件 sha256：

| 文件 | sha256 |
| --- | --- |
| `.cluster_operator/replay_padding_batch_integration_probe_0605.py` | `0d31948543de4817bd5e339f8aa4a7b7080b17437047ff056e6133b39f75a36b` |
| `src/utils/train/train_runner.py` | `708eb737a27dad3cea33ae5b7eb586d46f98bd0be67a9e586d15e31dcfdc78d1` |
| `src/data/dataloader.py` | `f63e53b1f300f798f9d2674987e884b61a4ae884170f3b8784949a5f45734e35` |
| `src/utils/model/padded_collate.py` | `94d17bd79ba68e1a3da6bead53a8302c12b820ff6984d4d4d69e93ce76d7e0e3` |
| `src/model/modules/generator.py` | `aace72f5eefc5d1e9c67830f7a51db1d4332f4c8b4b62c22fd2c074f2fc2fe8c` |
| `src/model/modules/loss.py` | `bb92c481857f913cf3520684cec82868c0e3d3f005112782585c5ec79cc0881f` |
| `src/model/odesign.py` | `cfade2496e05068d435a3b253160a992d36efbb760f5e0d6a2635b401be1586c` |

## 控制变量

| 随机性/变量 | 控制方式 | 目的 |
| --- | --- | --- |
| DDP topology | `NNODES=1`, `NPROC_PER_NODE=4`, `world_size=4` | 固定单节点 4 rank DDP allreduce 语义 |
| rank sample trace | `TRACE_SEED=20260605`, `per_rank_samples=10`, `UPDATES=3` | 同一 rank 的 bsz2 和 bsz1 使用同一批样本、同一顺序 |
| bad sample retry | `disable_random_resample(train_dataset)` | 避免某一路径遇到坏样本后随机替换导致样本序列分叉 |
| dataset sample seed | `fetch_sample` 前调用 `seed_all(sample_seed)` | 固定 crop/featurizer 中依赖 Python、NumPy、Torch 的随机行为 |
| dropout/drop path | replay config 中将 dropout/drop_path 置 0 | 避免训练 dropout 造成 bsz1/bsz2 分叉 |
| condition dropout | `condition_embedding_drop_rate=0.0` | 固定 conditioning 分支 |
| MSA row sampling | `DETERMINISTIC_MSA_ROWS=1`，patch MSA sampling helper | 固定 MSA rows，避免随机 row selection |
| SE(3) augmentation | `centre_random_augmentation` 替换为 deterministic center/identity rotation/zero translation | 固定坐标增强 |
| diffusion sigma | `sample_noise_level` 固定为 `1.0` | 固定 diffusion loss scale |
| diffusion noise | `add_noise_with_condition` 替换为 deterministic transform | 固定 noisy coordinate 输入 |
| permutation | `DISABLE_PERMUTATION=true` | 隔离 chain/symmetry permutation 随机分支 |
| TF32 | `NVIDIA_TF32_OVERRIDE=0`, `DISABLE_TF32=true`, `torch.set_float32_matmul_precision("highest")` | 排除 TF32/batch-shape kernel 数值差异 |
| dtype | `MODEL_DTYPE=fp32` | 使用 fp32 诊断路径 |
| DeepSpeed EVO attention | `USE_DEEPSPEED_EVO_ATTENTION=false` | 避免额外 attention kernel 差异 |
| dense LDDT diagnostic | `DENSE_LDDT_MAX_ATOMS=0` | 跳过大 atom dense LDDT 重算，避免 OOM；使用 sparse LDDT diagnostic |
| forward probe | `FORWARD_PROBE_SAMPLE_POS=-1` | r9 未启用 forward hook 细粒度 probe；`forward_probe_failure_count=0` 只能表示没有 forward probe failure，不能表示 forward probe 已比较通过 |

## 对比设计

每个 rank 每个 update 采样 10 个样本：

- `bsz2_gacc5`: micro batch size = 2，gradient accumulation = 5。
- `bsz1_gacc10`: micro batch size = 1，gradient accumulation = 10。

对比口径：

- `bsz2_gacc5` 先跑，保存 sample tensors、records 和 rank0 state。
- `bsz1_gacc10` 后跑，使用同一 trace，并把 `bsz2_gacc5` 作为 reference。
- 比较 sample tensors、loss/metrics summary、pre-clip grad summary、post-clip grad summary、optimizer step 后 state、all-rank grad/state hashes。
- r9 设置了 `FORWARD_PROBE_SAMPLE_POS=-1`，因此没有启用 forward probe 细粒度 hook；forward probe 相关 failure count 为 0 不是 forward hook 通过证据。

## r9 结果

`summary.json` 的关键字段：

```json
{
  "status": "pass",
  "world_size": 4,
  "updates": 3,
  "trace_seed": 20260605,
  "atol": 0.0005,
  "rtol": 0.0005,
  "diagnostic_atol": 0.0001,
  "diagnostic_rtol": 0.0001,
  "disable_permutation": true,
  "deterministic_msa_rows": 1,
  "save_grad_tensors": false,
  "failure_count": 0,
  "record_failure_count": 0,
  "state_failure_count": 0,
  "state_sync_failure_count": 0,
  "diagnostic_failure_count": 0
}
```

每 rank 每 update records 摘要：

| variant | rank | records | last update | record allclose | state allclose | diagnostic failures | state sync |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bsz2_gacc5` | 0 | 3 | 2 | N/A | N/A | 0 | true |
| `bsz2_gacc5` | 1 | 3 | 2 | N/A | N/A | 0 | true |
| `bsz2_gacc5` | 2 | 3 | 2 | N/A | N/A | 0 | true |
| `bsz2_gacc5` | 3 | 3 | 2 | N/A | N/A | 0 | true |
| `bsz1_gacc10` | 0 | 3 | 2 | true | true | 0 | true |
| `bsz1_gacc10` | 1 | 3 | 2 | true | N/A | 0 | true |
| `bsz1_gacc10` | 2 | 3 | 2 | true | N/A | 0 | true |
| `bsz1_gacc10` | 3 | 3 | 2 | true | N/A | 0 | true |

说明：

- `bsz2_gacc5` 是 reference variant，因此 `record_compare` 和 `state_compare` 为空是预期行为。
- `bsz1_gacc10` 的 `record_compare.allclose=true` 覆盖 4 个 rank、3 个 update；最后一条 record 是 `update_idx=2`。
- rank0 保存并比较完整 state，因此 rank0 的 `state_compare.allclose=true` 覆盖 3 个 update；非 rank0 不重复保存完整 state compare，但 all-rank `state_hashes_synced=true`。
- 4 个 rank 的 `pre_clip_grad_hashes_synced=true`、`post_clip_grad_hashes_synced=true`、`state_hashes_synced=true`，说明 DDP 同步没有 rank 分叉。
- r9 未启用 forward probe；如果需要补强“中间层 forward tensor”证据，应另跑 `FORWARD_PROBE_SAMPLE_POS=0` 或其他 sample position 的 repeat。

## 重复实验设计

### Repeat A: exact repeat

目的：证明 r9 不是偶发运行结果。

保持不变：

- 同一 root checkout。
- 同一 runner/probe。
- 同一 `35999.pt` checkpoint。
- 同一 `TRACE_SEED=20260605`。
- 同一 `UPDATES=3`。
- 同一 world size：`NNODES=1`, `NPROC_PER_NODE=4`。
- 同一随机控制：no-TF32/fp32、disable permutation、deterministic MSA rows、fixed diffusion sigma/noise、zero dropout。
- 同一通过阈值：主 `5e-4`，diagnostic `1e-4`。

只改变：

- `RUN_ID`。
- `RECORD_DIR/OUTPUT_DIR`。
- `MASTER_PORT`，避免端口复用风险。

已启动的 exact repeat：

| 项目 | 值 |
| --- | --- |
| run id | `replay_bsz2_bsz1_4gpu_notf32_pod_0606_r10_repeat` |
| tmux session | `odesign_replay4g_0606_r10_repeat` |
| run dir | `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign/.cluster_operator/replay_bsz2_bsz1_4gpu_notf32_pod_0606_r10_repeat` |
| master port | `29510` |

通过标准：

- `returncode_node0.txt = 0`。
- `summary.json.status = pass`。
- `failure_count = 0`。
- `record_failure_count = 0`。
- `state_failure_count = 0`。
- `state_sync_failure_count = 0`。
- `diagnostic_failure_count = 0`。
- `bsz1_gacc10` 4 个 rank 均有 3 条 records，最后 update 的 `record_compare.allclose=true`、`diagnostic_failure_count=0`、`state_hashes_synced=true`。

### Repeat B: shifted trace seed

目的：扩大真实样本覆盖，检查结论是否依赖单一 sample trace。

建议配置：

- `RUN_ID=replay_bsz2_bsz1_4gpu_notf32_pod_0606_r11_seed20260606`
- `TRACE_SEED=20260606`
- `MASTER_PORT=29511`
- 其他变量与 r9/r10 完全一致。

通过标准同 Repeat A。

执行顺序：

1. 等 Repeat A 终态。
2. 若 Repeat A 通过，再启动 Repeat B。
3. 若 Repeat A 失败，先诊断差异来源，不应直接启动 Repeat B 掩盖问题。

### Repeat C: forward-probe spot check

目的：补足 r9 没启用 forward hook 的边界，抽查中间层 forward tensor 是否仍在可接受容差内。

建议配置：

- `RUN_ID=replay_bsz2_bsz1_4gpu_notf32_pod_0606_r12_forwardprobe`
- `FORWARD_PROBE_SAMPLE_POS=0`
- `UPDATES=1` 或 `UPDATES=3`。如果只做中间层 spot check，可先用 `UPDATES=1` 降低成本；如果要和 r9 完全同长度，则用 `UPDATES=3`。
- `TRACE_SEED=20260605`
- `MASTER_PORT=29512`
- 其他变量与 r9/r10 完全一致。

通过标准：

- `summary.json.status = pass`，或至少 `record_failure_count=0`、`state_failure_count=0`、`state_sync_failure_count=0` 且 forward probe compare 在明确阈值下通过。
- `forward_probe_failure_count = 0`。
- forward probe JSON 中关键层的真实 token/atom prefix compare allclose。

注意：

- Repeat C 是补充中间层证据，不替代 Repeat A 的 exact repeat。
- 如果 Repeat C 在 `DIAGNOSTIC_ATOL/RTOL=1e-4` 下失败，但主 record/state 通过，应沿用 r4/r6 的分析方式，先判断是 fp32 batch-shape 数值尾差还是实际 padding/mask 问题。

## 审计意见补做实验

独立审计指出 r9 的两个 claim boundary：

1. `SAVE_GRAD_TENSORS=false`，不能 claim full-grad tensor 全量保存比较通过。
2. `FORWARD_PROBE_SAMPLE_POS=-1`，不能 claim forward hook 中间层 probe 已通过。

后续补做实验按边界拆分处理：r11 补 full-grad tensor；r13/r18 补 forward hook probe。r12 试图同时补两个边界，但 bsz1 阶段运行过慢，已中止，不作为通过证据。

### r10 exact repeat

| 项目 | 值 |
| --- | --- |
| run id | `replay_bsz2_bsz1_4gpu_notf32_pod_0606_r10_repeat` |
| return code | `0` |
| `summary.status` | `pass` |
| world size | `4` |
| updates | `3` |
| `save_grad_tensors` | `false` |
| `failure_count` | `0` |
| `record/state/state_sync/diagnostic_failure_count` | 全部 `0` |

结论：r10 复现 r9 的 4GPU、3 update replay pass，但仍未补 full-grad tensor 和 forward hook 边界。

### r11 full-grad + strict forward probe

| 项目 | 值 |
| --- | --- |
| run id | `replay_bsz2_bsz1_4gpu_notf32_pod_0606_r11_fullgrad_forward1` |
| run dir | `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign/.cluster_operator/replay_bsz2_bsz1_4gpu_notf32_pod_0606_r11_fullgrad_forward1` |
| return code | `1` |
| `summary.status` | `fail` |
| world size | `4` |
| updates | `1` |
| `save_grad_tensors` | `true` |
| `FORWARD_PROBE_SAMPLE_POS` | `0` |
| `diagnostic_atol/rtol` | `1e-4 / 1e-4` |
| `grad_compare_failure_sum` | `0` |
| `forward_probe_failure_sum` | `4` |
| `record/state/state_sync failure count` | 全部 `0` |

关键事实：

- rank0 保存并比较了 full-grad tensor：
  - `bsz2_gacc5/pre_clip_grad_after_update_0.pt`
  - `bsz2_gacc5/post_clip_grad_after_update_0.pt`
- `bsz1_gacc10` 的 `pre_clip_grad_compare.allclose=true`、`post_clip_grad_compare.allclose=true`。
- r11 失败只来自 strict forward probe sampled values；首个 mismatch 是 `diffusion_transformer.output_a#00`，sampled value 最大绝对差约 `0.0028`，而 stats compare 为 true。

结论：r11 关闭了“full-grad tensor 全量保存比较”边界，但没有关闭 “forward hook 中间层 probe” 边界。

### r12 combined retry

| 项目 | 值 |
| --- | --- |
| 4GPU run id | `replay_bsz2_bsz1_4gpu_notf32_pod_0606_r12_fullgrad_forward5e3` |
| 2GPU run id | `replay_bsz2_bsz1_2gpu_notf32_pod_0606_r12_fullgrad_forward5e3` |
| 目的 | 同时补 full-grad 和 forward-probe |
| 改动 | 新增 forward-probe 专用 `FORWARD_PROBE_ATOL/RTOL=5e-3/5e-3`，主 `DIAGNOSTIC_ATOL/RTOL` 保持 `1e-4/1e-4` |
| 结果 | 中止 |

中止原因：

- 两个 r12 都完成 bsz2 reference、full-grad 保存或部分保存后，在 bsz1 首个 microbatch 阶段长时间无 sample tensor 产出。
- 4GPU r12 已写出 bsz2 侧 40 个 sample tensor、bsz2 侧 4 个 forward probe JSON；加上 bsz1 早期产物，forward probe JSON 合计 8 个；rank0 pre/post full-grad 各约 1.4GB。
- 该 run 没有 `summary.status=pass`，不得作为通过证据。

### r13 forward-probe lightweight replay

| 项目 | 值 |
| --- | --- |
| run id | `replay_bsz2_bsz1_4gpu_notf32_pod_0606_r13_forward2samp5e3` |
| run dir | `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign/.cluster_operator/replay_bsz2_bsz1_4gpu_notf32_pod_0606_r13_forward2samp5e3` |
| return code | `0` |
| `summary.status` | `pass` |
| world size | `4` |
| updates | `1` |
| `per_rank_samples` | `2` |
| `save_grad_tensors` | `false` |
| `FORWARD_PROBE_SAMPLE_POS` | `0` |
| main `ATOL/RTOL` | `5e-4 / 5e-4` |
| diagnostic `ATOL/RTOL` | `1e-4 / 1e-4` |
| forward probe `ATOL/RTOL` | `5e-3 / 5e-3` |
| `failure_count` | `0` |
| `record/state/state_sync/diagnostic_failure_count` | 全部 `0` |
| summed `sample/forward_probe/grad_compare_failure_count` | 全部 `0` |

r13 对脚本的实验控制改动：

- 新增 `--forward-probe-atol/--forward-probe-rtol`，使 forward hook 容差与 sample/grad diagnostic 容差分离。
- 新增 `--per-rank-samples`，默认仍为 `10`；r13 显式设为 `2`，只作为 forward-probe lightweight replay，不替代 r9/r10 的 3 update、10 samples/rank replay。

深查记录：

- `bsz1_gacc10` rank0..3：`micro_count=2`、`sample_compare_failure_count=0`、`forward_probe_failure_count=0`、`grad_compare_failure_count=0`、`record_compare.allclose=true`。
- `bsz1_gacc10` rank0：`state_compare.allclose=true`；所有 rank `state_hashes_synced=true`。
- `bsz1_gacc10` 每个 rank 的 2 个 sample tensor compare 均为 true。
- `bsz1_gacc10` 每个 rank 的 forward probe compare 均为 true。

结论：r13 先关闭了“forward hook 中间层 probe 已启用并通过”的最小边界，但该结论限定在 `per_rank_samples=2`、`updates=1`、forward probe 容差 `5e-3/5e-3` 的 lightweight replay；它不替代 r9/r10 的完整 3 update replay，也不替代 r11 的 full-grad tensor 证据。

### r14 default-kernel forward-probe 诊断失败

| 项目 | 值 |
| --- | --- |
| run id | `replay_bsz2_bsz1_4gpu_notf32_pod_0606_r14_fwd10samp5e3` |
| run dir | `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign/.cluster_operator/replay_bsz2_bsz1_4gpu_notf32_pod_0606_r14_fwd10samp5e3` |
| return code | `1` |
| `summary.status` | `fail` |
| world size | `4` |
| updates | `1` |
| `per_rank_samples` | `10` |
| `save_grad_tensors` | `false` |
| `FORWARD_PROBE_SAMPLE_POS` | `0` |
| forward probe `ATOL/RTOL` | `5e-3 / 5e-3` |
| `failure_count` | `8` |
| `record_failure_count` | `4` |
| `diagnostic_failure_count` | `4` |
| `state_failure_count/state_sync_failure_count` | `0 / 0` |

关键事实：

- `preflight_env.txt` 中没有 `MODEL_DTYPE=fp32`，也没有 `USE_DEEPSPEED_EVO_ATTENTION=false`；replay 脚本在这种情况下默认 `USE_DEEPSPEED_EVO_ATTENTION=true`，dtype 使用配置默认值。
- `bsz1_gacc10` 的 4 个 rank 都出现 `record_compare.allclose=false`，每个 rank 的 `forward_probe_failure_count=1`。
- `grad_compare_failure_count=0`，rank0 `state_compare.allclose=true`，所有 rank 的 grad/state hash sync 仍为 true。

结论：r14 是“控制变量不匹配的诊断失败”，不能用来推翻 r9/r10/r13 在 no-evo/fp32/no-TF32 口径下的结论。它说明 `per_rank_samples=10、FORWARD_PROBE_SAMPLE_POS=0` 的 forward probe 对 kernel/dtype 路径敏感；要补强 r13，必须在与 r9/r10/r13 相同的 `MODEL_DTYPE=fp32`、`USE_DEEPSPEED_EVO_ATTENTION=false` 控制变量下重跑。

### r15/r16/r17 中止记录

| run id | 规模 | 结果 | 说明 |
| --- | --- | --- | --- |
| `replay_bsz2_bsz1_4gpu_notf32_pod_0606_r15_repeat_fwd10samp5e3` | 4GPU | `returncode=143`，无 `summary.json` | 中止，不作为证据 |
| `replay_bsz2_bsz1_2gpu_noevo_pod_0606_r16_fwd10samp5e3` | 2GPU | `returncode=143`，无 `summary.json` | 中止，且缺少 `MODEL_DTYPE=fp32` 控制变量 |
| `replay_bsz2_bsz1_4gpu_noevo_fp32_pod_0606_r17_fwd10samp5e3` | 4GPU | `returncode=143`，无 `summary.json` | 中止，不作为证据 |

### r18 4GPU no-evo/fp32 per-rank-samples=10 forward-probe replay

| 项目 | 值 |
| --- | --- |
| run id | `replay_bsz2_bsz1_4gpu_noevo_fp32_pod_0606_r18_fwd10samp5e3_clean` |
| run dir | `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign/.cluster_operator/replay_bsz2_bsz1_4gpu_noevo_fp32_pod_0606_r18_fwd10samp5e3_clean` |
| return code | `0` |
| `summary.status` | `pass` |
| world size | `4` |
| updates | `1` |
| `per_rank_samples` | `10` |
| `MODEL_DTYPE` | `fp32` |
| `USE_DEEPSPEED_EVO_ATTENTION` | `false` |
| `DISABLE_TF32/NVIDIA_TF32_OVERRIDE` | `true / 0` |
| `save_grad_tensors` | `false` |
| `FORWARD_PROBE_SAMPLE_POS` | `0` |
| main `ATOL/RTOL` | `5e-4 / 5e-4` |
| diagnostic `ATOL/RTOL` | `1e-4 / 1e-4` |
| forward probe `ATOL/RTOL` | `5e-3 / 5e-3` |
| `failure_count` | `0` |
| `record/state/state_sync/diagnostic_failure_count` | 全部 `0` |

深查记录：

- `preflight_env.txt` 明确记录 `MODEL_DTYPE=fp32` 和 `USE_DEEPSPEED_EVO_ATTENTION=false`。
- `bsz1_gacc10` rank0..3 均有 `sample_compare_failure_count=0`、`forward_probe_failure_count=0`、`grad_compare_failure_count=0`、`record_compare.allclose=true`。
- rank0 `state_compare.allclose=true`；所有 rank 的 `pre_clip_grad_hashes_synced=true`、`post_clip_grad_hashes_synced=true`、`state_hashes_synced=true`。

结论：r18 把 r13 的 forward-probe lightweight 证据扩展到单节点 4GPU、`per_rank_samples=10`、1 update、`FORWARD_PROBE_SAMPLE_POS=0` 的 replay。它关闭了 r14 在控制变量不匹配下暴露出的 `per_rank_samples=10` forward-probe 缺口；但它仍不表示 10 个 sample position 都做了 forward hook，也不替代 r9/r10 的 3 update 主训练集成 replay 或 r11 的 full-grad tensor 证据。

### r19 2GPU no-evo/fp32 repeat

| 项目 | 值 |
| --- | --- |
| run id | `replay_bsz2_bsz1_2gpu_noevo_fp32_pod_0606_r19_fwd10samp5e3` |
| run dir | `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign/.cluster_operator/replay_bsz2_bsz1_2gpu_noevo_fp32_pod_0606_r19_fwd10samp5e3` |
| return code | `0` |
| `summary.status` | `pass` |
| world size | `2` |
| updates | `1` |
| `per_rank_samples` | `10` |
| `MODEL_DTYPE` | `fp32` |
| `USE_DEEPSPEED_EVO_ATTENTION` | `false` |
| `failure_count` | `0` |
| `record/state/state_sync/diagnostic_failure_count` | 全部 `0` |

结论：r19 是 r18 同控制变量下的 2GPU repeat，包括 `NVIDIA_TF32_OVERRIDE=0`、`SAVE_GRAD_TENSORS=false`、`FORWARD_PROBE_SAMPLE_POS=0` 和相同阈值，仅 `world_size=2`。它说明该 `per_rank_samples=10、pos0` forward-probe 通过结果不只依赖一次 4GPU 运行；但它不能替代 4GPU/8GPU/16GPU 的 DDP world-size gate，也不提供 full-grad tensor 证据。

### r20 2GPU no-evo/fp32 pos9 forward-probe repeat

| 项目 | 值 |
| --- | --- |
| run id | `replay_bsz2_bsz1_2gpu_noevo_fp32_pod_0607_r20_fwdpos9` |
| run dir | `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign/.cluster_operator/replay_bsz2_bsz1_2gpu_noevo_fp32_pod_0607_r20_fwdpos9` |
| return code | `0` |
| `summary.status` | `pass` |
| world size | `2` |
| updates | `1` |
| `per_rank_samples` | `10` |
| `MODEL_DTYPE` | `fp32` |
| `USE_DEEPSPEED_EVO_ATTENTION` | `false` |
| `DISABLE_TF32/NVIDIA_TF32_OVERRIDE` | `true / 0` |
| `save_grad_tensors` | `false` |
| `FORWARD_PROBE_SAMPLE_POS` | `9` |
| main `ATOL/RTOL` | `5e-4 / 5e-4` |
| diagnostic `ATOL/RTOL` | `1e-4 / 1e-4` |
| forward probe `ATOL/RTOL` | `5e-3 / 5e-3` |
| `failure_count` | `0` |
| `record/state/state_sync/diagnostic_failure_count` | 全部 `0` |

深查记录：

- `preflight_env.txt` 明确记录 `MODEL_DTYPE=fp32`、`USE_DEEPSPEED_EVO_ATTENTION=false` 和 `FORWARD_PROBE_SAMPLE_POS=9`。
- `bsz1_gacc10` rank0..1 均有 `sample_compare_failure_count=0`、`forward_probe_failure_count=0`、`grad_compare_failure_count=0`、`record_compare.allclose=true`。
- rank0 `state_compare.allclose=true`；rank0..1 的 `pre_clip_grad_hashes_synced=true`、`post_clip_grad_hashes_synced=true`、`state_hashes_synced=true`。

结论：r20 是针对独立审计“r18/r19 只覆盖 `FORWARD_PROBE_SAMPLE_POS=0`”措辞边界的定向补强。它说明在同一 no-evo/fp32/no-TF32 控制变量下，2GPU、`per_rank_samples=10`、末尾 sample position `9` 的 forward-probe replay 也通过；但它仍不能替代 4GPU/8GPU/16GPU world-size gate，也不提供 full-grad tensor 证据。

### r22 2GPU full-grad + forward-probe 组合补强

| 项目 | 值 |
| --- | --- |
| run id | `replay_bsz2_bsz1_2gpu_noevo_fp32_pod_0607_r22_fullgrad_fwd2samp5e3` |
| run dir | `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign/.cluster_operator/replay_bsz2_bsz1_2gpu_noevo_fp32_pod_0607_r22_fullgrad_fwd2samp5e3` |
| 运行容器 | `zjow-odesign-pbp-2gpu-clone20260606162953-94144454-jl5md` |
| return code | `0` |
| `summary.status` | `pass` |
| world size | `2` |
| updates | `1` |
| `per_rank_samples` | `2` |
| `MODEL_DTYPE` | `fp32` |
| `USE_DEEPSPEED_EVO_ATTENTION` | `false` |
| `DISABLE_TF32/NVIDIA_TF32_OVERRIDE` | `true / 0` |
| `save_grad_tensors` | `true` |
| `FORWARD_PROBE_SAMPLE_POS` | `0` |
| main `ATOL/RTOL` | `5e-4 / 5e-4` |
| diagnostic `ATOL/RTOL` | `1e-4 / 1e-4` |
| forward probe `ATOL/RTOL` | `5e-3 / 5e-3` |
| `failure_count` | `0` |
| `record/state/state_sync/diagnostic_failure_count` | 全部 `0` |

深查记录：

- `outputs/summary.json` 和顶层 `summary.json` 均存在，`status=pass`。
- `bsz2_gacc5/pre_clip_grad_after_update_0.pt` 大小约 `1.424067262GB`，`bsz2_gacc5/post_clip_grad_after_update_0.pt` 大小约 `1.424071194GB`，作为 full-grad reference 保存。
- `bsz1_gacc10/rank00_records.json` 的 `record_compare.allclose=true`、`state_compare.allclose=true`、`pre_clip_grad_compare.allclose=true`、`post_clip_grad_compare.allclose=true`。
- `bsz1_gacc10` rank0..1 的 `sample_compare_failure_count=0`、`forward_probe_failure_count=0`、`grad_compare_failure_count=0`、`pre_clip_grad_hashes_synced=true`、`post_clip_grad_hashes_synced=true`、`state_hashes_synced=true`。
- 运行快照记录了 runner sha256 `b0082d19231fb10e8723b16c9210b4e505a92b766d72d81bc6e3916763af3836` 和 probe sha256 `24cb2ef0469dfa70d31b19ca7bc7f51241c0d372e734d38c0e3123b23dd0b6eb`。外层共享盘 repo 快照为 `cc95c2f8f915af4d87b1db3f44c4db2df8566a41` 加 dirty runtime 改动；因此 r22 是集群 runtime artifact 证据，不是干净 commit 证据。

结论：r22 在同一个 2GPU、1 update、`per_rank_samples=2` replay 中同时打开 `SAVE_GRAD_TENSORS=true` 和 `FORWARD_PROBE_SAMPLE_POS=0`，并在 no-evo/fp32/no-TF32 控制变量下通过。它补强了 r11/r13 分别提供 full-grad 和 forward-probe 证据的组合边界；但它仍是低成本组合补强，不替代 r9/r10 的 4GPU 3 update 主 replay，不覆盖 `per_rank_samples=10` 的 full-grad 组合，也不覆盖 8GPU/16GPU world-size gate。

### r10-r13 补做实验独立事实审计

在 r10-r13 补做实验完成后，已再次启动独立只读 Codex 审计会话核查 r10/r11/r12/r13 的 summary、returncode、rank record、实际产物和文档措辞。审计 verdict 为 `pass`，并确认：

- r10 只复现 r9 的 4GPU、3 update replay，不补 full-grad 或 forward-probe 边界。
- r11 的 `SAVE_GRAD_TENSORS=true`，rank0 pre/post full-grad 文件存在且约 1.42GB，`pre_clip_grad_compare.allclose=true`、`post_clip_grad_compare.allclose=true`、`grad_compare_failure_count=0`；但 r11 的 forward probe 在 strict `1e-4` 口径失败，不能作为 forward-probe 通过证据。
- r12 没有 `summary.status=pass` 证据，文档必须保持“中止/不作为通过证据”的表述。
- r13 的 `FORWARD_PROBE_SAMPLE_POS=0` 见 `env_node0.txt`，`summary.status=pass`，`forward_probe_atol/rtol=5e-3/5e-3`；rank0..3 的 `forward_probe_failure_count=0`、`sample_compare_failure_count=0`、`grad_compare_failure_count=0`，`record_compare.allclose=true`，rank0 `state_compare.allclose=true`，all ranks `state_hashes_synced=true`。

审计指出的一处文档措辞修正已经完成：r12 forward probe JSON 数量明确为 bsz2 侧 4 个，加上 bsz1 早期产物合计 8 个。当前仍需保留的限制是：r13 只覆盖 `updates=1`、`per_rank_samples=2` 的 forward-probe lightweight replay；r11 只覆盖 `updates=1` 的 full-grad tensor 补证；r22 只覆盖 2GPU、1 update、`per_rank_samples=2` 的 full-grad + forward-probe 组合补强；r9/r10 才是 3 update、10 samples/rank 的主训练集成 replay 证据。r14-r22 是后续新增补做记录，需要单独审计。

## 独立审计

已启动独立只读审计会话核查 r9 真实性：

- agent id: `019e9ccb-a8dc-7a52-999f-6c14ca334f9d`
- nickname: `Euclid`
- 范围：远端 `summary.json`、records、`launch_env.sh`、runner command、本地文档一致性、claim boundary。
- 要求：不修改文件，不启动或停止任务，不使用网络搜索。

审计结果：

- verdict: `pass`。
- 核查确认远端 `summary.json`、`bsz1_gacc10/rank00..03_records.json`、`bsz2_gacc5/rank00..03_records.json`、`launch_env.sh`、`env_node0.txt`、`stdout_stderr_node0.log`、`returncode_node0.txt` 均存在并与 r9 claim 一致。
- 核查确认 `summary.json` 中 `status='pass'`，`world_size=4`，`updates=3`，`trace_seed=20260605`，主阈值 `5e-4`，diagnostic 阈值 `1e-4`，全部 failure count 为 0。
- 核查确认 r9 是 `SMOKE_MODE=1` 的单节点 4GPU DDP replay，不是 16GPU 跨节点 replay，不是正式长训。
- 核查确认 `SAVE_GRAD_TENSORS=false`，所以不能宣称 4GPU 每 rank full-grad tensor 全量保存比较通过。
- 核查确认 `FORWARD_PROBE_SAMPLE_POS=-1`，所以不能宣称 forward hook 细粒度中间层 probe 已通过。
- 审计建议已采纳：显式记录 `SMOKE_MODE=1`，把“4GPU replay”收紧为“单节点 4GPU DDP replay”，把“最后一个 update”改为“最后一条 record (`update_idx=2`)”。
