# ODesign 训练效率 E1 Profiling 记录

本文档记录 2026-06-08 到 2026-06-09 在用户提供的 2GPU H200 长时容器中完成的 ODesign E1 baseline profiling。E1 先完成 20 个 optimizer update 的稳定性验证，再按合同扩展到 50 个 optimizer update。该 profiling 的目的只是建立训练速度和显存基线，不是 PBP 质量证明，也不说明相关 checkpoint 已达到任何目标分数。

## 结论摘要

本次 E1 profiling 成功完成两段 baseline：

| run id | returncode | optimizer updates | profile rows | checkpoint | wall time |
| --- | ---: | ---: | ---: | --- | ---: |
| `effprof_2gpu_baseline_20260608_r1` | `0` | `20` | `100/rank`, `200 total` | `19.pt` | `5921s`, `98.7min` |
| `effprof_2gpu_baseline50_20260609_r1` | `0` | `50` | `250/rank`, `500 total` | `49.pt` | `13234s`, `220.6min` |

主要性能结论：

- data wait 不是主要瓶颈：排除 cold-start 后，rank0/rank1 的 `data_wait_sec` 均值约 `0.005s`。
- 稳态 microbatch 主要耗时在 forward/backward：20-update run 排除首条 cold-start 后约 `55.1s`，50-update 扩展约 `52.15s`。
- 50-update 扩展中，forward 均值约 `18.5-19.0s`，backward 均值约 `30.35s`。
- `torch.cuda.empty_cache()` 每个 microbatch 约 `0.37s`，不是最大瓶颈，但可作为低风险配置项继续验证。
- 50-update 扩展 GPU CSV 峰值显存约 `129.8GB` / `117.2GB`，其中包含 pod keepalive 进程基线；PyTorch JSONL 记录的训练进程峰值约 `122.5GB` / `111.2GB` allocated。
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

50-update 扩展 run 使用同一 runtime 和数据合同，run-specific 记录如下：

| 项目 | 值 |
| --- | --- |
| run id | `effprof_2gpu_baseline50_20260609_r1` |
| record dir | `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign/.cluster_operator/effprof_2gpu_baseline50_20260609_r1` |
| output dir | `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign/outputs/efficiency_profile_2gpu_baseline50/effprof_2gpu_baseline50_20260609_r1` |
| master port | `29615` |
| max steps | `50` |
| eval interval | `50` |
| train_runner hash at launch | `e556516eb90322b1aec87b360ca6344f7e1d746a0c9b56aa631c77061e7706be` |
| train_runner hash after wrapper restore | `e05756199a92a4bf76b1d8a0884d2adc120f64a14eb3f50991599408d8ae1543` |

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

50-update 扩展的完整命令记录在：

```text
${RECORD_DIR}/command.txt
```

与 20-update run 的关键差异：

| 项目 | 20-update | 50-update |
| --- | --- | --- |
| `master_port` | `29608` | `29615` |
| `exp.exp_name` | `efficiency_profile_2gpu_baseline` | `efficiency_profile_2gpu_baseline50` |
| `exp.max_steps` | `20` | `50` |
| `exp.eval_interval` | `20` | `50` |
| output dir | `outputs/efficiency_profile_2gpu_baseline/...` | `outputs/efficiency_profile_2gpu_baseline50/...` |

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

50-update 扩展额外产物：

| 文件 | 说明 |
| --- | --- |
| `summary_steady_v2.json` | 50-update 二次汇总，含 wall time、loss、timing、memory |
| `summary_steady_v2.md` | 50-update Markdown 汇总表 |
| `train_runner_hash_at_launch.txt` | 确认该 run 启动时使用 E1 原始 train runner |

50-update checkpoint：

```text
outputs/efficiency_profile_2gpu_baseline50/effprof_2gpu_baseline50_20260609_r1/checkpoints/49.pt
```

该 checkpoint 同样只是 profiling run 的训练产物，未做 PBP/ODesignBench 质量评测。

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

50-update 扩展 timing：

| 指标 | 值 |
| --- | --- |
| start epoch | `1780979566` |
| end epoch | `1780992800` |
| wall time | `13234s` |
| optimizer updates | `50` |
| microbatches/rank | `250` |

50-update rank-level all rows：

| rank | microbatch mean | microbatch p50 | microbatch p90 | forward mean | backward mean | loss mean | data wait mean | empty cache mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rank0 | `52.23s` | `51.82s` | `64.36s` | `18.95s` | `30.34s` | `0.15s` | `0.06s` | `0.37s` |
| rank1 | `52.15s` | `51.78s` | `64.21s` | `18.50s` | `30.33s` | `0.16s` | `0.14s` | `0.37s` |

50-update 排除每个 rank 第一条 cold-start row 后：

| rank | microbatch mean | microbatch p50 | microbatch p90 | forward mean | backward mean | loss mean | data wait mean | empty cache mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rank0 | `52.15s` | `51.82s` | `64.32s` | `18.98s` | `30.35s` | `0.15s` | `0.005s` | `0.37s` |
| rank1 | `52.16s` | `51.84s` | `64.21s` | `18.49s` | `30.35s` | `0.16s` | `0.006s` | `0.37s` |

50-update loss summary：

| rank | loss first | loss last | loss min | loss max | loss mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| rank0 | `8.663` | `7.447` | `4.072` | `18.980` | `9.591` |
| rank1 | `6.090` | `7.200` | `3.602` | `19.169` | `9.960` |

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

50-update 扩展 GPU CSV 峰值：

| GPU | peak used | total | peak utilization |
| --- | ---: | ---: | ---: |
| 0 | `129769 MiB` | `143771 MiB` | `100%` |
| 1 | `117243 MiB` | `143771 MiB` | `100%` |

50-update 扩展 PyTorch JSONL 中训练进程 max allocated：

| rank | max allocated | max reserved |
| --- | ---: | ---: |
| rank0 | `122475 MiB` | `126708 MiB` |
| rank1 | `111203 MiB` | `114184 MiB` |

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

50-update 扩展中同样出现一次相同的可恢复跳过：

```text
[skip data 308339] 6b5p_2 does not have MSA for pairing
ValueError: 6b5p_2 does not have MSA for pairing
```

50-update 扩展还出现 3 条 RDKit explicit valence warning：

```text
Explicit valence for atom # 0 Al, 5, is greater than permitted
Explicit valence for atom # 0 Al, 5, is greater than permitted
Explicit valence for atom # 8 O, 5, is greater than permitted
```

这些 warning 没有导致训练失败，最终 `returncode=0`，checkpoint `49.pt` 正常保存。

## Claim Boundary

本 run 可以支持的结论：

- 在该 2GPU H200 容器、该 runtime snapshot、该数据 root 下，ODesign baseline profiling 可以跑通 20 optimizer updates。
- 在同一执行面下，ODesign baseline profiling 也可以跑通 50 optimizer updates，并保存最终 checkpoint `49.pt`。
- 该配置在 2GPU 上不是 data-wait bottleneck。
- `diffusion_batch_size=48`、crop 640、bsz2/gacc5 的稳态 microbatch 约 `52-55s`，训练 20 updates 约 `98.7min`，训练 50 updates 约 `220.6min`。
- 显存峰值接近 H200 容量上限，但仍未 OOM。

本 run 不能支持的结论：

- 不能证明 checkpoint `19.pt` 或 `49.pt` 有质量价值。
- 不能证明该配置能更快达到 PBP target。
- 不能与 4GPU、8GPU、16GPU run 直接比较主指标，因为资源合同不同。
- 不能把 `ODESIGN_PROFILE_SYNC_CUDA=1` 下的速度直接当作无 instrumentation 的真实吞吐。

## Next Actions

已按 E1 合同完成从 20 updates 到 50 updates 的扩展。50-update 结果确认：

- 同配置在 2GPU H200 上可连续跑完 50 optimizer updates，`returncode=0`。
- data wait 结论保持不变，不是主要瓶颈。
- forward/backward 仍是主要耗时。
- 显存峰值更高，rank0 PyTorch allocated peak 到约 `122.5GB`，但未 OOM。

建议的下一组实验：

1. `SYNC_CUDA=0` 的 10-20 update 对照：估计 profiling 同步开销和真实吞吐。
2. `empty_cache` 频率对照：每 microbatch、每 optimizer update、关闭。
3. `diffusion_lddt_chunk_size` sweep：`1 -> 2 -> 4`，观察显存是否仍可承受。
4. activation checkpoint granularity sweep：优先看 diffusion/pairformer 的 recompute 开销。
5. `diffusion_batch_size` sweep：在显存峰值接近 110GB 的前提下，谨慎评估 24/32/48，而不是先扩大。

2026-06-09 更新：这组建议是 E1 之后的历史下一步。后续 E2/module-level profiling 已经进一步收敛到 Pairformer/MSA compute path；同时用户明确第一阶段应保持训练 setting 不变。因此当前优先级不是继续配置 sweep，而是按 `docs/odesign_operator_level_acceleration_plan_20260609.zh.md` 先做 operator-level attribution 和等价算子/runtime 优化。
