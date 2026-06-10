# ODesign 训练效率 E2 Profiling 记录

本文档记录 E1 baseline 之后的 E2 targeted profiling。E2 的目标是用同一 2GPU H200 资源合同和同一训练数据合同，逐项验证潜在速度瓶颈。本文档只支持速度、显存和稳定性结论，不支持 PBP 质量结论。

## 资源和数据合同

除非单个实验另有说明，E2 复用 E1 的资源和数据合同：

| 项目 | 值 |
| --- | --- |
| 容器 | `zjow-odesign-pbp-efficiency-2gpu-4607978-8rw86` |
| GPU | `2 x NVIDIA H200` |
| runtime root | `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign` |
| data root | `/mnt/shared-storage-user/ai4sreason/scireason_2/data/odesign` |
| base checkpoint | `/mnt/shared-storage-user/ai4sreason/scireason_2/data/odesign/ckpt/protenix_base_default_v0.5.0.pt` |
| training config | `train_odesign_base_prot_flex_weighted_before20210930_reso_below4_16gpu_tokenbalanced` |
| train batch size | `exp.data.train_batch_size=2` |
| gradient accumulation | `exp.iters_to_accumulate=5` |
| crop size | `640` |
| diffusion batch size | `48` |
| diffusion lDDT chunk size | `1` |
| dataloader workers | `4` |

统一解析脚本：

```text
scripts/summarize_efficiency_profile.py
```

该脚本会解析每个 record dir 的 `profile_steps_rank*.jsonl` 和 `gpu_memory.csv`，输出：

- `summary_steady.json` 或指定 JSON 路径。
- 可直接粘贴到文档的 Markdown 表。
- `all_rows`、`drop_first_row`、`drop_first_accumulation_window` 三种统计口径。

## E2A：关闭 profiling CUDA sync

目的：估计 `ODESIGN_PROFILE_SYNC_CUDA=1` 的测量同步开销，得到更接近真实吞吐的 profiling 口径。

| 项目 | 值 |
| --- | --- |
| run id | `effprof_2gpu_sync0_20260608_r1` |
| changed variable | `ODESIGN_PROFILE_SYNC_CUDA=0` |
| baseline | `effprof_2gpu_baseline_20260608_r1` |
| max steps | `10` |
| record dir | `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign/.cluster_operator/effprof_2gpu_sync0_20260608_r1` |
| output dir | `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign/outputs/efficiency_profile_2gpu_sync0/effprof_2gpu_sync0_20260608_r1` |
| returncode | `0` |
| checkpoint | `9.pt` |
| wall time | `2963s` |

与 E1 的稳态对比，统计口径为每个 rank 排除第一条 cold-start row：

| run | rank | rows | microbatch mean | forward mean | backward mean | data wait mean | empty cache mean | PyTorch max allocated |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E1 sync=1 | rank0 | 99 | `55.08s` | `20.91s` | `30.67s` | `0.00s` | `0.39s` | `110100 MiB` |
| E1 sync=1 | rank1 | 99 | `55.10s` | `20.85s` | `30.67s` | `0.00s` | `0.41s` | `108785 MiB` |
| E2A sync=0 | rank0 | 49 | `55.65s` | `20.19s` | `30.00s` | `0.00s` | `0.35s` | `92870 MiB` |
| E2A sync=0 | rank1 | 49 | `55.69s` | `20.52s` | `30.18s` | `0.00s` | `0.40s` | `108785 MiB` |

解释：

- `SYNC_CUDA=0` 没有带来显著 microbatch 加速，说明 E1 的 profiling CUDA sync 不是主要开销。
- E2A 的稳态 microbatch 约 `55.7s`，与 E1 的 `55.1s` 在同一量级。
- E2A 只有 10 个 optimizer update，显存峰值和样本难度会受短样本序列影响；显存比较应优先看是否 OOM 和 PyTorch allocator 峰值范围，不应过度解释单次 rank0/rank1 差异。
- 主瓶颈判断仍然不变：forward/backward compute 和 checkpoint/recompute 路径，而不是 dataloader wait 或 profiling sync。

## Empty Cache 控制代码

为了做 `torch.cuda.empty_cache()` 频率对照，`src/utils/train/train_runner.py` 增加了一个默认不改变行为的环境变量：

```text
ODESIGN_EMPTY_CACHE_POLICY
```

可选值：

| 值 | 行为 |
| --- | --- |
| unset / `microbatch` / `1` / `true` | 默认行为，每个 microbatch 末尾调用 `torch.cuda.empty_cache()` |
| `optimizer_update` / `optimizer` / `optimizer_step` | 只在发生 optimizer update 的 microbatch 末尾调用 |
| `never` / `0` / `false` / `off` | 不在 `TrainRunner.train_step()` 末尾调用 |

注意：该开关目前只控制 `TrainRunner.train_step()` 末尾的 empty-cache 调用，不控制模型模块内部已有的 `torch.cuda.empty_cache()` 调用。

验证：

```text
PYTHONPATH=<runtime root> /root/miniconda3/envs/odesign/bin/python /tmp/odesign_efficiency_launchers/test_train_runner_efficiency_controls.py
```

结果：

```text
Ran 3 tests in 0.000s
OK
```

远端 runtime 当前 `train_runner.py` hash：

```text
e05756199a92a4bf76b1d8a0884d2adc120f64a14eb3f50991599408d8ae1543  src/utils/train/train_runner.py
```

同步前的旧文件已备份在：

```text
<runtime root>/.cluster_operator/runtime_patch_backups/
```

## E2B：Empty Cache 只在 Optimizer Update 调用

目的：测试把 `TrainRunner.train_step()` 末尾的 empty-cache 频率从每个 microbatch 降低到每个 optimizer update，是否能在不 OOM 的前提下降低 wall time。

| 项目 | 值 |
| --- | --- |
| run id | `effprof_2gpu_emptycache_optimizer_20260608_r1` |
| changed variable | `ODESIGN_EMPTY_CACHE_POLICY=optimizer_update` |
| baseline compare | `effprof_2gpu_sync0_20260608_r1` |
| max steps | `10` |
| `ODESIGN_PROFILE_SYNC_CUDA` | `0` |
| record dir | `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign/.cluster_operator/effprof_2gpu_emptycache_optimizer_20260608_r1` |
| returncode | `0` |
| checkpoint | `9.pt` |
| wall time | `2934s` |

通过标准：

- `returncode=0`。
- 两个 rank 都写出 50 条 profiling JSONL。
- `empty_cache_called` 只在 optimizer update row 为 `true`。
- 显存不 OOM，loss finite。
- 与 E2A 比较 microbatch mean、wall time 和 PyTorch allocator 峰值。

已完成的早期策略核验：

```text
profile_steps_rank00.jsonl
[(0, False, False, 0.003), (1, False, False, 0.0), (2, False, False, 0.0), (3, False, False, 0.0), (4, True, True, 0.934), ...]

profile_steps_rank01.jsonl
[(0, False, False, 0.001), (1, False, False, 0.0), (2, False, False, 0.0), (3, False, False, 0.0), (4, True, True, 0.725), ...]
```

元组含义为：

```text
(global_step, optimizer_update, empty_cache_called, empty_cache_sec)
```

这说明 env 开关已经被训练代码实际读取：非 update microbatch 不调用 `empty_cache`，update microbatch 调用。

完成后的策略核验：

```text
profile_steps_rank00.jsonl rows 50 called 10 updates 10 bad 0
profile_steps_rank01.jsonl rows 50 called 10 updates 10 bad 0
```

与 E2A 的稳态对比：

| run | rank | rows | microbatch mean | forward mean | backward mean | data wait mean | empty cache mean | PyTorch max allocated |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E2A sync=0, microbatch empty-cache | rank0 | 49 | `55.65s` | `20.19s` | `30.00s` | `0.00s` | `0.351s` | `92870 MiB` |
| E2A sync=0, microbatch empty-cache | rank1 | 49 | `55.69s` | `20.52s` | `30.18s` | `0.00s` | `0.403s` | `108785 MiB` |
| E2B optimizer-update empty-cache | rank0 | 49 | `55.02s` | `19.82s` | `29.98s` | `0.01s` | `0.114s` | `92870 MiB` |
| E2B optimizer-update empty-cache | rank1 | 49 | `55.04s` | `20.16s` | `30.18s` | `0.01s` | `0.121s` | `108785 MiB` |

解释：

- E2B 成功把 `TrainRunner.train_step()` 末尾 empty-cache 调用次数从每 rank 50 次降到 10 次。
- 10 update 短跑 wall time 从 E2A 的 `2963s` 降到 `2934s`，改善约 `29s`，即约 `1.0%`。
- 稳态 microbatch mean 改善约 `0.6s`，方向符合预期，但幅度较小，仍需用 E2C `never` 补齐上界。
- 显存没有恶化到 OOM；PyTorch max allocated 与 E2A 基本一致。

## E2C：关闭 TrainRunner 末尾 Empty Cache

目的：测试完全关闭 `TrainRunner.train_step()` 末尾 empty-cache 的速度上界和显存风险。

| 项目 | 值 |
| --- | --- |
| run id | `effprof_2gpu_emptycache_never_20260608_r1` |
| changed variable | `ODESIGN_EMPTY_CACHE_POLICY=never` |
| baseline compare | `effprof_2gpu_sync0_20260608_r1` |
| max steps | `10` |
| `ODESIGN_PROFILE_SYNC_CUDA` | `0` |
| record dir | `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign/.cluster_operator/effprof_2gpu_emptycache_never_20260608_r1` |
| returncode | `0` |
| checkpoint | `9.pt` |
| wall time | `2928s` |

通过标准：

- `returncode=0`。
- 两个 rank 都写出 50 条 profiling JSONL。
- 所有 row 的 `empty_cache_called=false`。
- 显存不 OOM，loss finite。
- 与 E2A/E2B 比较 microbatch mean、wall time 和 PyTorch allocator 峰值。

早期策略核验：

```text
profile_steps_rank00.jsonl rows 15 called 0 max_empty_sec 0.0034
profile_steps_rank01.jsonl rows 15 called 0 max_empty_sec 0.0007
```

这说明 `never` 策略已经被训练代码实际读取；当前已写 row 均未调用 `TrainRunner.train_step()` 末尾 empty-cache。

完成后的策略核验：

```text
profile_steps_rank00.jsonl rows 50 called 0 updates 10 max_empty_sec 0.0034
profile_steps_rank01.jsonl rows 50 called 0 updates 10 max_empty_sec 0.0016
```

与 E2A/E2B 的稳态对比：

| run | rank | rows | microbatch mean | forward mean | backward mean | empty cache mean | PyTorch max allocated | PyTorch max reserved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E2A microbatch | rank0 | 49 | `55.65s` | `20.19s` | `30.00s` | `0.351s` | `92870 MiB` | `95336 MiB` |
| E2A microbatch | rank1 | 49 | `55.69s` | `20.52s` | `30.18s` | `0.403s` | `108785 MiB` | `112726 MiB` |
| E2B optimizer_update | rank0 | 49 | `55.02s` | `19.82s` | `29.98s` | `0.114s` | `92870 MiB` | `95432 MiB` |
| E2B optimizer_update | rank1 | 49 | `55.04s` | `20.16s` | `30.18s` | `0.121s` | `108785 MiB` | `112026 MiB` |
| E2C never | rank0 | 49 | `54.84s` | `19.74s` | `30.01s` | `0.000s` | `92870 MiB` | `95434 MiB` |
| E2C never | rank1 | 49 | `54.85s` | `20.08s` | `30.22s` | `0.000s` | `108785 MiB` | `111974 MiB` |

解释：

- 完全关闭 `TrainRunner.train_step()` 末尾 empty-cache 后，10 update 短跑 wall time 从 E2A 的 `2963s` 降到 `2928s`，改善约 `35s`，即约 `1.2%`。
- 相比 E2B，E2C 只额外改善约 `6s` wall time，说明 `optimizer_update` 已拿到主要收益。
- E2C 没有 OOM，PyTorch max allocated 与 E2A/E2B 基本一致；在这个 10 update 窗口内，关闭该处 empty-cache 是安全的。
- 这不是主瓶颈优化。即便采用 E2C，forward/backward 仍占绝大多数时间。

## 当前结论

- 已排除 profiling CUDA sync 是主瓶颈。
- `TrainRunner.train_step()` 末尾的 `empty_cache` 单次约 `0.35-0.40s`；E2B/E2C 证明降低或关闭该处调用可带来约 `1.0-1.2%` 的短跑 wall-time 改善，但不是主瓶颈。
- E2D 显示 `diffusion_lddt_chunk_size=2` 没有带来 wall-time 改善，且显存明显上升；下一优先级应转向 activation checkpoint granularity，而不是继续围绕 profiling overhead 或 lDDT chunk 做文章。

2026-06-09 更新：上述“转向 activation checkpoint granularity”的结论是在尚未加入“固定训练 setting”约束前形成的历史判断。用户已明确第一阶段应通过算子和 runtime 加速提升 forward/backward 速度，因此当前下一优先级改为 PyTorch profiler/NVTX operator-level attribution；`blocks_per_ckpt` 和其他配置 sweep 暂缓到后续阶段。

## E2D：Diffusion lDDT Chunk Size 2

目的：测试把 `exp.loss.diffusion_lddt_chunk_size` 从 `1` 提到 `2` 是否减少 lDDT loss 循环/重算开销。为了只改一个主要变量，E2D 以 E2C 为 baseline，保持 `ODESIGN_EMPTY_CACHE_POLICY=never` 和 `ODESIGN_PROFILE_SYNC_CUDA=0`。

| 项目 | 值 |
| --- | --- |
| run id | `effprof_2gpu_lddtchunk2_20260608_r1` |
| changed variable | `DIFFUSION_LDDT_CHUNK_SIZE=2` |
| baseline compare | `effprof_2gpu_emptycache_never_20260608_r1` |
| max steps | `10` |
| `ODESIGN_EMPTY_CACHE_POLICY` | `never` |
| `ODESIGN_PROFILE_SYNC_CUDA` | `0` |
| record dir | `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign/.cluster_operator/effprof_2gpu_lddtchunk2_20260608_r1` |
| returncode | `0` |
| checkpoint | `9.pt` |
| wall time | `2929s` |

与 E2C 的稳态对比：

| run | rank | rows | microbatch mean | forward mean | loss mean | backward mean | PyTorch max allocated | PyTorch max reserved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E2C chunk1 | rank0 | 49 | `54.84s` | `19.74s` | `0.120s` | `30.01s` | `92870 MiB` | `95434 MiB` |
| E2C chunk1 | rank1 | 49 | `54.85s` | `20.08s` | `0.207s` | `30.22s` | `108785 MiB` | `111974 MiB` |
| E2D chunk2 | rank0 | 49 | `54.90s` | `19.73s` | `0.102s` | `30.09s` | `98527 MiB` | `101014 MiB` |
| E2D chunk2 | rank1 | 49 | `54.91s` | `20.06s` | `0.186s` | `30.32s` | `116422 MiB` | `119254 MiB` |

解释：

- E2D 没有 wall-time 改善：`2928s -> 2929s`，基本持平。
- loss mean 小幅下降约 `0.02s`，但 loss 本身只占总 microbatch 的极小比例。
- 显存明显上升，rank1 PyTorch max allocated 从约 `108.8GB` 上升到约 `116.4GB`。
- 在当前配置下，`diffusion_lddt_chunk_size=2` 不是有效速度优化；继续试 `4` 可能进一步增加显存风险，信息增益低于 activation checkpoint/recompute sweep。

## Claim Boundary

本文档可以支持：

- E2A 在同一 2GPU runtime 和数据合同下成功完成 10 optimizer updates。
- `SYNC_CUDA=0` 没有显著改变稳态 microbatch time。
- empty-cache 策略控制代码有标准库单元测试覆盖，并已同步到远端 runtime。
- E2B 成功完成 10 optimizer updates，并验证 `optimizer_update` 策略实际生效。
- E2C 成功完成 10 optimizer updates，并验证 `never` 策略实际生效；该 10 update 窗口内未出现 OOM。
- E2D 成功完成 10 optimizer updates；`diffusion_lddt_chunk_size=2` 未带来速度改善，并增加显存。

本文档不能支持：

- 不能证明任何 checkpoint 有 PBP 质量价值。
- 不能证明某个效率配置能更快达到 PBP target。
- 不能证明长训中永远关闭 `TrainRunner` 末尾 empty-cache 一定安全；当前只验证了 10 update profiling 窗口。
