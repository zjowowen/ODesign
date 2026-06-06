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
