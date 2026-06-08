# ODesign 训练效率首轮 Profiling 合同

本文档是 `docs/odesign_training_efficiency_roadmap.zh.md` 的第一轮执行合同。目标是在固定资源和固定训练数据下，建立 ODesign 训练速度观测基线，为后续优化 `time_to_target_checkpoint` 做准备。

当前合同是 profiling 合同，不是正式效率结论。它不证明任何 checkpoint 已经 PBP 达标，也不替代后续 PBP/ODesignBench 质量评测。

## 目标

主目标：

- 在 2GPU H200 长时容器上验证 profiling 执行面可用。
- 记录 baseline 训练 step 的 wall-clock breakdown 和显存行为。
- 判断首轮最值得优化的方向：data wait、forward/loss/backward、activation checkpoint、`empty_cache`、diffusion/lDDT chunk 或 DDP 通信。

不在本轮证明：

- 不证明某个训练配置能更快达到 PBP target。
- 不比较不同 GPU 卡数或节点数。
- 不改变训练数据、sampler 或质量评测口径。
- 不把 20-100 step 的短跑速度当作长训质量结论。

## 资源合同

| 项目 | 设置 |
| --- | --- |
| 容器 | `zjow-odesign-pbp-efficiency-2gpu-4607978-8rw86` |
| SSH | `zjow-odesign-pbp-efficiency-2gpu-4607978-8rw86.zhangjinouwen+root.ailab-ai4sdata.pod@h.pjlab.org.cn` |
| 远端 hostname | `rjob-e529095eee5a10e-f61eb500bddbbf61-0` |
| GPU | `2 x NVIDIA H200` |
| Python env | `/root/miniconda3/envs/odesign` |
| torch | `2.3.1+cu121` |
| 计时起点 | 训练命令进入第一条可度量 step 之后；SSH 登录和人工准备时间不计入训练耗时 |

环境准备：

```bash
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}
export PATH=/usr/local/nvidia/bin:${PATH:-}
source /root/miniconda3/etc/profile.d/conda.sh
conda activate odesign
```

当前 GPU 状态核验：

- `nvidia-smi -L` 可见两张 H200。
- 当前只看到 `python adjust_gpu_dymanic.py`，按 H200 pod runbook 记录为保活占卡，不作为业务占用。

## 代码和路径合同

推荐执行面：

```text
/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign
```

原因：

- 该路径是此前正式 padded-batch 训练使用的 runtime snapshot。
- 它没有 `.git`，因此 profiling run 必须记录关键文件 hash。
- 不应在该目录里直接做未记录的代码编辑；若需要 instrumentation，应先在 git worktree 中提交，再同步到 runtime snapshot。

参考 git checkout：

```text
/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_3/SciReasoner-2/reference/ODesign
```

注意：

- 该 checkout 当前有历史 dirty/untracked 内容，不适合作为正式受控实验来源。
- root 访问会触发 git dubious ownership，可用 `git -c safe.directory=<path>` 做只读核查。

首轮 profiling 必须记录这些文件 hash：

- `scripts/train.py`
- `src/utils/train/train_runner.py`
- `src/data/dataloader.py`
- `src/utils/model/padded_collate.py`
- `src/model/odesign.py`
- `src/model/modules/pairformer.py`
- `src/model/modules/diffusion.py`
- `src/model/modules/transformer.py`
- `src/model/modules/loss.py`
- `configs/model/_base.yaml`
- `configs/exp/train_odesign_base_prot_flex.yaml`

## 数据合同

数据根：

```text
/mnt/shared-storage-user/ai4sreason/scireason_2/data/odesign
```

必须可访问的资产：

- `indices/weightedPDB_indices_before_2021-09-30_wo_posebusters_resolution_below_4.csv`
- `mmcif/`
- `mmcif_bioassembly/`
- `mmcif_msa/`
- `seq_to_pdb_index.json`
- `ckpt/protenix_base_default_v0.5.0.pt`

首轮 profiling 默认不做 PBP 质量判定，因此不需要完整跑到 PBP target。若进入 short quality run，则必须另行声明：

- `PBP_TARGET_SCORE`
- checkpoint interval
- PBP 评测脚本和参数
- baseline checkpoint/setting

## 首轮建议实验

### E0：只读环境核验

目的：确认执行面、数据、GPU、conda 可用。

命令类型：

- `nvidia-smi -L`
- `python -c "import torch; ..."`
- `test -d` / `test -f` 检查数据资产
- 关键文件 `sha256sum`

通过标准：

- 2 张 H200 可见。
- `odesign` conda 环境可用。
- 训练数据和 base checkpoint 可访问。
- 记录 runtime snapshot 的关键文件 hash。

当前状态：

- 已完成。

### E1：baseline 2GPU 短 profiling

目的：在不改训练语义的前提下，得到 2GPU step time 和显存基线。

建议资源：

- `1 node x 2 GPU`
- `torchrun --nproc_per_node=2`

建议配置：

- 使用当前 padded-batch 正式训练附近配置。
- `exp.data.train_batch_size=2`
- `exp.iters_to_accumulate=5`
- `exp.diffusion_batch_size=48`
- `exp.loss.diffusion_lddt_chunk_size=1`
- `exp.data.num_dl_workers=4`
- `exp.model.use_deepspeed_evo_attention=true`
- `NVIDIA_TF32_OVERRIDE=1`
- 关闭 eval：`exp.eval_first=false`，`exp.eval_interval=999999999`，`exp.test_sets=[]`
- checkpoint 可先关闭或设为较大 interval，避免 I/O 干扰 profiling。

建议长度：

- 先跑 20 个 optimizer update，确认无 OOM/NaN/启动问题。
- 如果稳定，再跑 50-100 个 optimizer update 形成 baseline。

必须记录：

- 每步训练 elapsed time。
- dataloader wait、H2D copy、forward、loss、backward、optimizer 时间。
- 每步 loss 和 lr。
- 每步或每 2 秒的 GPU memory/utilization。
- `returncode`。

通过标准：

- 训练能进入 step loop。
- 至少 20 个 optimizer update 无 OOM、无 NaN、无非零退出。
- rank0 写出 profiling JSONL 或 stdout 里有足够解析的 step timing。

当前状态：

- 已完成一次 2GPU H200 baseline profiling：`effprof_2gpu_baseline_20260608_r1`。
- 结果：`returncode=0`，20 optimizer updates，100 microbatches/rank，最终 checkpoint `19.pt`。
- 详细记录：`docs/odesign_training_efficiency_e1_profile_20260608.zh.md`。
- 远端 record dir：`/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign/.cluster_operator/effprof_2gpu_baseline_20260608_r1`。
- 关键结论：该 2GPU baseline 不是 data-wait bottleneck；排除 cold-start 后 microbatch mean 约 `55s`，主要瓶颈在 forward/backward compute 和 checkpoint/recompute 路径。
- 不建议直接扩大同配置到 50-100 optimizer updates；20 updates 已有 200 条 rank-level profiling 记录，继续同配置长跑的边际信息增益低于针对瓶颈做 E2 对照。
- E2 targeted profiling 已记录在 `docs/odesign_training_efficiency_e2_profile_20260608.zh.md`。E2A 已排除 `ODESIGN_PROFILE_SYNC_CUDA=1` 是主要开销；E2B/E2C 显示降低或关闭 `TrainRunner.train_step()` 末尾 `torch.cuda.empty_cache()` 只带来约 `1.0-1.2%` 的 10-update 短跑 wall-time 改善，不是主瓶颈。

### E2：最小变量 sweep

只有 E1 稳定后才进入 E2。

建议顺序：

1. `torch.cuda.empty_cache()` 频率：每 microbatch、每 optimizer step、关闭。
2. `exp.loss.diffusion_lddt_chunk_size`：1、2、4。
3. `exp.model.blocks_per_ckpt`：1、2、4。
4. `exp.data.num_dl_workers`：4、8、12。

每个 sweep 只改一个主要变量，并复用 E1 的资源和数据合同。

通过标准：

- 速度改善必须同时给出显存峰值和 loss finite 证据。
- 只要出现 OOM、NaN、checkpoint load mismatch 或数据错误，该候选配置标记为 failed，不继续扩大。

## 日志和产物位置建议

建议使用独立 run id：

```text
effprof_2gpu_baseline_YYYYMMDD_rN
```

建议 record dir：

```text
/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign/.cluster_operator/effprof_2gpu_baseline_YYYYMMDD_rN
```

建议产物：

- `env.txt`
- `command.txt`
- `file_hashes.txt`
- `stdout_stderr.log`
- `gpu_memory.csv`
- `profile_steps_rank0.jsonl`
- `returncode.txt`
- `summary.json`

## Claim Boundary

本轮可以 claim：

- 某资源/数据合同下的短 profiling 是否能跑通。
- 某变量是否改善短跑 step time 或吞吐。
- 某变量是否引入 OOM/NaN/启动失败。

本轮不能 claim：

- 某配置更快达到 PBP target。
- 某配置质量不退化。
- 某配置适合正式长训。
- 不同 GPU 卡数或节点数之间的主指标优劣。

要进入 `time_to_target_checkpoint` 结论，必须补充 short quality run 和 PBP evaluation gate。
