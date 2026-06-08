# ODesign 训练效率 E1 Profiling 记录

本文档记录 2026-06-08 到 2026-06-09 在用户提供的 2GPU H200 长时容器中完成的一次 ODesign E1 baseline profiling。该 run 的目的只是建立训练速度和显存基线，不是 PBP 质量证明，也不说明该 checkpoint 已达到任何目标分数。

## 结论摘要

本次 E1 profiling 成功完成：

- `returncode=0`
- 20 个 optimizer updates
- 100 个 microbatches/rank
- 2 个 rank 共 200 条 profiling JSONL
- 最终保存 checkpoint `19.pt`
- wall time `5921s`，约 `98.7min`

主要性能结论：

- data wait 不是主要瓶颈：排除 cold-start 后，rank0/rank1 的 `data_wait_sec` 均值约 `0.005s`。
- 稳态 microbatch 主要耗时在 forward/backward：排除首条 cold-start 后，rank0/rank1 的 `microbatch_total_sec` 均值约 `55.1s`。
- forward 均值约 `20.9s`，backward 均值约 `30.7s`。
- `torch.cuda.empty_cache()` 每个 microbatch 约 `0.39-0.41s`，不是最大瓶颈，但可作为低风险配置项继续验证。
- GPU CSV 峰值显存约 `116.9GB` / `115.6GB`，其中包含 pod keepalive 进程基线；PyTorch JSONL 记录的训练进程峰值约 `110.1GB` / `108.8GB` allocated。
- 训练过程中出现一次可恢复坏样本跳过：`6b5p_2 does not have MSA for pairing`，训练继续并成功结束。

## Run Contract

| 项目 | 值 |
| --- | --- |
| run id | `effprof_2gpu_baseline_20260608_r1` |
| 容器 | `zjow-odesign-pbp-efficiency-2gpu-4607978-8rw86` |
| hostname | `rjob-e529095eee5a10e-f61eb500bddbbf61-0` |
| GPU | `2 x NVIDIA H200` |
| conda env | `/root/miniconda3/envs/odesign` |
| torch | `2.3.1+cu121` |
| runtime root | `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign` |
| data root | `/mnt/shared-storage-user/ai4sreason/scireason_2/data/odesign` |
| record dir | `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign/.cluster_operator/effprof_2gpu_baseline_20260608_r1` |
| output dir | `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign/outputs/efficiency_profile_2gpu_baseline/effprof_2gpu_baseline_20260608_r1` |

本地 git 分支：

- branch: `feature/odesign-padding-batch`
- pushed fork commit at run closeout: `b3dc608`
- profiling instrumentation commit included in branch history: `f042a09`

runtime snapshot 没有 `.git`，因此以 `file_hashes.txt` 为准。关键 hash：

| 文件 | sha256 |
| --- | --- |
| `scripts/train.py` | `8978a01f3702591bea41e5aef3db8756e3829247667098b4e4839d7c94cdf59a` |
| `configs/exp/train_odesign_base_prot_flex_weighted_before20210930_reso_below4_16gpu_tokenbalanced.yaml` | `b341c763bd03db673da2f23ce78126eabbbf9831a24a3cd76761073812109671` |
| `configs/data/weightedPDB_before20210930_wo_posebusters_reso_below4.yaml` | `594d18df0e4a4dc757e5f3223aadb0ac2eae3128ec7c6747f4ddc82d2b7c0e95` |
| `src/utils/train/train_runner.py` | `e556516eb90322b1aec87b360ca6344f7e1d746a0c9b56aa631c77061e7706be` |
| `src/data/dataloader.py` | `f63e53b1f300f798f9d2674987e884b61a4ae884170f3b8784949a5f45734e35` |
| `src/model/odesign.py` | `cfade2496e05068d435a3b253160a992d36efbb760f5e0d6a2635b401be1586c` |
| `src/model/modules/loss.py` | `bb92c481857f913cf3520684cec82868c0e3d3f005112782585c5ec79cc0881f` |

## Command

命令记录在：

```text
${RECORD_DIR}/command.txt
```

核心命令：

```bash
torchrun \
  --nnodes=1 \
  --nproc_per_node=2 \
  --master_addr=127.0.0.1 \
  --master_port=29608 \
  ./scripts/train.py \
  exp=train_odesign_base_prot_flex_weighted_before20210930_reso_below4_16gpu_tokenbalanced \
  data_root_dir=/mnt/shared-storage-user/ai4sreason/scireason_2/data/odesign \
  ckpt_root_dir=/mnt/shared-storage-user/ai4sreason/scireason_2/data/odesign/ckpt \
  exp.max_steps=20 \
  exp.iters_to_accumulate=5 \
  exp.data.train_batch_size=2 \
  exp.data.epoch_size=1286480 \
  exp.train_crop_size=640 \
  exp.data.train_crop_size=640 \
  exp.diffusion_batch_size=48 \
  exp.loss.diffusion_lddt_chunk_size=1 \
  exp.data.num_dl_workers=4 \
  exp.data.train_sampler.use_token_balanced_sampler=true \
  exp.test_sets=[] \
  exp.eval_first=false \
  exp.eval_interval=20 \
  exp.checkpoint_interval=0 \
  exp.use_wandb=false \
  exp.model.use_deepspeed_evo_attention=true \
  exp.load_checkpoint_path=/mnt/shared-storage-user/ai4sreason/scireason_2/data/odesign/ckpt/protenix_base_default_v0.5.0.pt \
  exp.load_params_only=true \
  exp.skip_load_optimizer=true \
  exp.skip_load_step=true \
  exp.skip_load_scheduler=true \
  exp.load_step_for_scheduler=false
```

Profiling 环境变量：

```bash
ODESIGN_PROFILE_JSONL=${RECORD_DIR}/profile_steps.jsonl
ODESIGN_PROFILE_ALL_RANKS=1
ODESIGN_PROFILE_SYNC_CUDA=1
```

注意：`ODESIGN_PROFILE_SYNC_CUDA=1` 会让阶段计时更可信，但会引入同步开销。若要测真实吞吐，需要另跑 `SYNC_CUDA=0` 对照。

## Artifacts

主要产物：

| 文件 | 说明 |
| --- | --- |
| `env.txt` | 环境、GPU、package 版本 |
| `command.txt` | 完整 torchrun 命令 |
| `file_hashes.txt` | runtime snapshot 关键文件 hash |
| `hydra_config.yaml` | resolved config |
| `hydra_overrides.yaml` | Hydra overrides |
| `stdout_stderr.log` | 训练日志 |
| `gpu_memory.csv` | 2 秒 GPU memory/utilization 采样 |
| `profile_steps_rank00.jsonl` | rank0 microbatch profile |
| `profile_steps_rank01.jsonl` | rank1 microbatch profile |
| `summary.json` | 原始自动汇总 |
| `summary_steady.json` | 二次汇总，修复 GPU CSV 解析并补充 steady-state 统计 |
| `checkpoints.txt` | checkpoint 列表 |
| `returncode.txt` | 退出码 |

checkpoint：

```text
outputs/efficiency_profile_2gpu_baseline/effprof_2gpu_baseline_20260608_r1/checkpoints/19.pt
```

该 checkpoint 只是 profiling run 的训练产物，未做 PBP/ODesignBench 质量评测。

## Timing Results

原始 wall time：

| 指标 | 值 |
| --- | --- |
| start epoch | `1780928360` |
| end epoch | `1780934281` |
| wall time | `5921s` |
| optimizer updates | `20` |
| microbatches/rank | `100` |

rank-level all rows：

| rank | microbatch mean | microbatch p50 | forward mean | backward mean | loss mean | data wait mean | empty cache mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rank0 | `57.45s` | `56.87s` | `23.24s` | `30.64s` | `0.15s` | `0.13s` | `0.39s` |
| rank1 | `57.25s` | `56.91s` | `23.07s` | `30.64s` | `0.18s` | `0.35s` | `0.41s` |

排除每个 rank 第一条 cold-start row 后：

| rank | microbatch mean | microbatch p50 | forward mean | backward mean | data wait mean | empty cache mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rank0 | `55.08s` | `56.26s` | `20.91s` | `30.67s` | `0.005s` | `0.388s` |
| rank1 | `55.10s` | `56.43s` | `20.85s` | `30.67s` | `0.005s` | `0.411s` |

排除第一个 gradient-accumulation window 后：

| rank | microbatch mean | microbatch p50 | forward mean | backward mean | update rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| rank0 | `55.29s` | `57.48s` | `21.26s` | `30.75s` | `19` |
| rank1 | `55.31s` | `57.39s` | `20.89s` | `30.74s` | `19` |

解释：

- all rows 中的 max forward/microbatch 包含 DeepSpeed Evo attention extension 首次编译和加载，不能当作稳态速度。
- 稳态数据等待接近 0，说明本 run 不是 dataloader-bound。
- 当前主要瓶颈是 forward/backward compute 和 checkpoint/recompute 路径。

## Memory Results

GPU CSV 峰值：

| GPU | peak used | total | peak utilization |
| --- | ---: | ---: | ---: |
| 0 | `116937 MiB` | `143771 MiB` | `100%` |
| 1 | `115615 MiB` | `143771 MiB` | `100%` |

PyTorch JSONL 中训练进程 max allocated：

| rank | max allocated | max reserved |
| --- | ---: | ---: |
| rank0 | `110100 MiB` | `113876 MiB` |
| rank1 | `108785 MiB` | `112726 MiB` |

注意：

- `gpu_memory.csv` 是整卡视角，包含 `adjust_gpu_dymanic.py` keepalive 基线。
- JSONL 是训练进程 PyTorch allocator 视角，更适合分析训练显存。

## Data Events

训练过程中出现一次可恢复的数据跳过：

```text
[skip data 308339] 6b5p_2 does not have MSA for pairing
ValueError: 6b5p_2 does not have MSA for pairing
```

这个 traceback 没有导致训练失败，最终 `returncode=0`。但它对 `time_to_target_checkpoint` 是重要信号：正式效率比较需要记录坏样本跳过率，因为跳过坏样本会影响 wall time、数据序列和有效样本吞吐。

## Claim Boundary

本 run 可以支持的结论：

- 在该 2GPU H200 容器、该 runtime snapshot、该数据 root 下，ODesign baseline profiling 可以跑通 20 optimizer updates。
- 该配置在 2GPU 上不是 data-wait bottleneck。
- `diffusion_batch_size=48`、crop 640、bsz2/gacc5 的稳态 microbatch 约 `55s`，训练 20 updates 约 `98.7min`。
- 显存峰值接近 H200 容量上限，但仍未 OOM。

本 run 不能支持的结论：

- 不能证明 checkpoint `19.pt` 有质量价值。
- 不能证明该配置能更快达到 PBP target。
- 不能与 4GPU、8GPU、16GPU run 直接比较主指标，因为资源合同不同。
- 不能把 `ODESIGN_PROFILE_SYNC_CUDA=1` 下的速度直接当作无 instrumentation 的真实吞吐。

## Next Actions

不建议立即扩大同配置到 50-100 optimizer updates，原因是：

- 20 updates 已有 200 条 rank-level microbatch 记录，足以定位首轮瓶颈。
- 按本次 wall time 推算，50 updates 约需 4 小时，100 updates 约需 8 小时。
- 继续同配置长 profiling 的信息增益低于针对瓶颈做对照实验。

建议的下一组实验：

1. `SYNC_CUDA=0` 的 10-20 update 对照：估计 profiling 同步开销和真实吞吐。
2. `empty_cache` 频率对照：每 microbatch、每 optimizer update、关闭。
3. `diffusion_lddt_chunk_size` sweep：`1 -> 2 -> 4`，观察显存是否仍可承受。
4. activation checkpoint granularity sweep：优先看 diffusion/pairformer 的 recompute 开销。
5. `diffusion_batch_size` sweep：在显存峰值接近 110GB 的前提下，谨慎评估 24/32/48，而不是先扩大。
