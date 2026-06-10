# ODesign 训练效率优化研发路线

本文档规划一个新的研发方向：在固定训练资源和固定训练数据的前提下，减少 ODesign 训练出有效 checkpoint 所需的训练时间。它不是某个单点优化的实现计划，而是未来若干轮 profiling、实验、工程改造和质量验证的路线图。

核心原则：

- 先定位瓶颈，再做优化；不要直接猜某个 kernel、batch size 或 dataloader 参数。
- 同时度量速度、显存、吞吐和质量；不能只看 step time。
- 每个实验只改一个主要变量，除非明确标记为 exploratory。
- 所有正式结论必须有日志、配置、代码版本、artifact 路径和质量评测支撑。

## 目标定义

本项目最关心的目标是：

> 在固定训练资源、固定训练数据、固定评测流程下，最小化首个有效 checkpoint 的训练 wall-clock time。

这里的“有效 checkpoint”定义为：该 checkpoint 经过同一套 PBP 评测流程后，PBP 分数达到预先声明的目标阈值 `PBP_TARGET_SCORE`。`PBP_TARGET_SCORE` 的具体数值应在每轮实验的 evaluation contract 中声明，不能在看到结果后再调整。

主指标：

| 指标 | 定义 | 说明 |
| --- | --- | --- |
| `time_to_target_checkpoint` | 从训练第一步开始，到首个 PBP 达标 checkpoint 写出所需的训练 wall-clock time | 主优化目标；默认不包含 H200 排队时间和后处理评测排队时间 |
| `target_checkpoint_step` | 首个 PBP 达标 checkpoint 的 optimizer step | 判断是“每步更快”还是“更少 step 达标” |
| `target_checkpoint_pbp` | 该 checkpoint 的 PBP 分数 | 必须达到或超过目标阈值 |
| `resource_contract` | GPU 类型、卡数、节点数、CPU/内存、网络设置 | 控制变量；不同资源合同不能直接比较主指标 |
| `data_contract` | 训练数据路径、索引、sampler 设置、crop/mask 配置、checkpoint 起点 | 控制变量；数据变了就不是同一轮效率比较 |

辅助指标分三层：

| 层次 | 指标 | 用途 |
| --- | --- | --- |
| 单步速度 | microbatch time、optimizer step time、samples/sec、tokens/sec、atoms/sec | 判断训练 loop 和 kernel 是否变快 |
| 固定资源利用率 | GPU utilization、GPU memory peak、effective tokens/hour | 判断在同一资源合同下是否更好地利用硬件 |
| 达标效率 | `target_checkpoint_step`、checkpoint 数、`time_to_target_checkpoint` | 判断优化是否真的更快产出有效 checkpoint |

最终目标不是单纯提高 `steps/sec` 或降低 GPU-hour，而是在同一资源和数据条件下更快产出 PBP 达标 checkpoint。GPU-hour 可以作为成本辅助指标，但如果卡数和节点数已经固定，主比较应使用训练 wall-clock time。

## 控制变量合同

每轮效率比较必须先冻结两个合同。

资源合同：

- GPU 类型，例如 H200。
- 节点数和每节点 GPU 数，例如 `1 node x 4 GPU`、`1 node x 8 GPU`、`2 node x 8 GPU`。
- CPU、内存、hostNetwork、gang scheduling、容器 image。
- 训练启动成功后的计时起点；H200 排队时间单独记录，不纳入主训练时间。

数据合同：

- 训练数据 root、索引文件和 mmcif/mmcif_bioassembly/mmcif_msa 路径。
- weighted sampler、epoch size、crop size、mask/crop/augmentation 设置。
- 起始 checkpoint，例如 base checkpoint 或 `35999.pt` resume checkpoint。
- checkpoint interval 和 PBP 评测脚本版本。

只有资源合同和数据合同都相同的 run，才能直接比较 `time_to_target_checkpoint`。

## 当前已知训练路径

当前正式 padded-batch 训练合同使用：

- `exp.data.train_batch_size=2`
- `exp.iters_to_accumulate=5`
- 16GPU world size 下 effective samples/update 为 `2 * 16 * 5 = 160`
- `exp.diffusion_batch_size=48`
- `exp.loss.diffusion_lddt_chunk_size=1`
- `exp.train_crop_size=640`
- `exp.data.num_dl_workers=4`
- `exp.model.use_deepspeed_evo_attention=true`
- `NVIDIA_TF32_OVERRIDE=1`

关键代码路径：

| 路径 | 位置 | 可能影响效率的点 |
| --- | --- | --- |
| dataloader | `src/data/dataloader.py` | weighted sampler、`num_dl_workers`、padded collate、bad sample retry |
| train loop | `src/utils/train/train_runner.py` | `to_device`、forward/loss/backward、grad accumulation、grad clip、optimizer、scheduler、logging、checkpoint/eval |
| model trunk | `src/model/odesign.py`、`src/model/modules/pairformer.py` | pair/token quadratic 路径、DeepSpeed Evo attention、activation checkpoint |
| diffusion | `src/model/modules/diffusion.py`、`src/model/modules/transformer.py` | diffusion sample 维、fine-grained checkpoint、transformer checkpoint |
| loss | `src/model/modules/loss.py` | smooth lDDT sparse/dense 路径、`diffusion_lddt_chunk_size`、checkpointed loss |
| config | `configs/model/_base.yaml`、`configs/exp/*.yaml` | `blocks_per_ckpt`、dtype、chunk size、diffusion batch、eval/checkpoint interval |

现有代码中已经能看到几个候选性能风险：

- `TrainRunner.train_step()` 每个 microbatch 末尾调用 `torch.cuda.empty_cache()`，可能引入同步和 allocator 开销，需要实测。
- 默认 `blocks_per_ckpt=1` 且 diffusion module 启用 fine-grained checkpoint，这有利于省显存，但可能显著增加重算。
- `diffusion_lddt_chunk_size=1` 是安全低显存设置，但可能牺牲 loss 计算吞吐。
- `num_dl_workers=4` 可能不足以喂满多卡，也可能因 GPFS/CPU 争用过高而不是越大越好。
- padded batch 提升 GPU 利用率的同时会带来 padding waste，需要按 token/atom 有效吞吐而不只是 samples/sec 评估。

## Roadmap 总览

### Phase 0：性能观测基线

目标：建立当前正式配置在每个资源合同下的可信性能基线，并记录到首个候选 checkpoint 的训练时间。

交付物：

- `training_efficiency_baseline_report`：记录代码版本、配置、集群资源、数据路径、checkpoint、运行脚本。
- step breakdown：dataloader、H2D copy、forward、loss、backward、grad clip、optimizer、scheduler、logging、checkpoint/eval。
- GPU telemetry：显存峰值、GPU utilization、功耗如果可见、OOM margin。
- throughput 指标：samples/sec、tokens/sec、atoms/sec、effective tokens/hour。
- checkpoint timeline：每个 checkpoint 的写出时间、step、训练 elapsed time。

最低实验：

- 1GPU 短跑，用于排除 DDP 通信。
- 4GPU 单节点短跑，用于观察单机多卡吞吐。
- 8GPU 单节点短跑，用于覆盖正式每节点 GPU 数。
- 16GPU 跨节点短跑，用于观察 NCCL 和跨节点放大后的表现。

通过标准：

- 至少连续 50 到 100 个 optimizer update 有完整 step time 记录；若做短训质量基线，则至少覆盖一个 checkpoint interval。
- 能区分 GPU compute bound、data bound、communication bound 或 checkpoint/recompute bound。
- 记录显存峰值，说明是否有空间换速度。

当前进展：

- 2026-06-08 已完成一次 2GPU H200 E1 baseline profiling，详见 `docs/odesign_training_efficiency_e1_profile_20260608.zh.md`。
- 该 run 完成 20 optimizer updates，100 microbatches/rank，`returncode=0`。
- 2026-06-09 已按 E1 合同完成 50-update 扩展：`effprof_2gpu_baseline50_20260609_r1`，250 microbatches/rank，`returncode=0`，最终 checkpoint `49.pt`。
- 首轮证据显示该配置不是 data-wait bottleneck；排除 cold-start 后 microbatch mean 约 `52-55s`，主要瓶颈在 forward/backward compute 和 checkpoint/recompute 路径。
- 50-update 扩展满足 Phase 0 对 50-100 optimizer update 连续 profiling 的低端要求；如继续做 100-update，应把它定义为额外稳定性确认，而不是当前 E1 gate 的必要条件。

### Phase 1：固定训练 setting 的算子级优化

目标：在不改变模型训练 setting 的前提下，通过 operator-level profiling、kernel dispatch 审计、tensor layout 和等价实现优化提升 forward/backward 速度。

当前阶段的硬约束：

- 不改变 forward 实际经过的模型模块。
- 不改变 loss、训练数据、sampler、crop、augmentation、batch、gradient accumulation、diffusion batch、checkpoint/eval 节奏。
- 不改变模型参数 dtype、训练 dtype、TF32/AMP/DeepSpeed Evo attention 等既有精度策略。
- 不把 activation checkpoint granularity、`diffusion_lddt_chunk_size`、`train_batch_size` 等 memory-speed tradeoff 当作第一批实验。

首要交付物是 `docs/odesign_operator_level_acceleration_plan_20260609.zh.md` 中定义的 operator-level attribution，而不是配置 sweep。

候选方向：

| 类型 | 例子 | 要求 |
| --- | --- | --- |
| operator profiling | PyTorch profiler、NVTX ranges、record_shapes | 只在 profiling 模式打开，不改变默认训练路径 |
| attention/kernel dispatch 审计 | 确认 Pairformer 主路径是否真实使用 DeepSpeed Evo attention 或 fallback 到 stock matmul/softmax | 先审计，不先切换精度或数学路径 |
| tensor layout 优化 | 减少重复 `transpose`、`contiguous`、mask/bias materialize | forward/grad allclose |
| 等价局部融合 | gate、silu、mask multiply、bias add、transition 子路径 | 单模块和真实 batch 数值 gate 通过 |
| 局部 compile | 只对纯局部 helper 或小模块尝试 | 不为 compile 改模型路径；记录 warmup 后速度 |

通过标准：

- Phase A 先给出 Pairformer/MSA 子阶段和 top CUDA kernels 的 attribution。
- Phase B 每个候选必须通过 forward/loss/grad/参数更新 allclose gate。
- 无 profiler speed run 需要同时报告 warmup 后 step time、显存峰值和 loss finite。

### Phase 2：低风险配置优化

目标：在完成固定 setting 的算子级诊断后，再通过配置找到同一资源/数据合同下更快写出 checkpoint 的速度/显存折中。

候选变量：

| 变量 | 候选值 | 假设 | 风险 |
| --- | --- | --- | --- |
| `exp.data.num_dl_workers` | 4, 8, 12, 16 | 增加 worker 可能改善 data stall | CPU/GPFS 争用，worker startup 开销 |
| `exp.data.train_batch_size` | 1, 2, 3, 4 | 更大 per-device batch 可能提升 GPU 吞吐 | padding waste、OOM、数值路径变化 |
| `exp.iters_to_accumulate` | 保持 effective batch 不变 | 对齐样本/update，隔离 batch 变量 | 真实 sampler 顺序仍需记录 |
| `exp.diffusion_batch_size` | 24, 48, 64, 96 | 更大 diffusion sample 可能提高吞吐或质量 | 显存和 loss 开销上升 |
| `exp.loss.diffusion_lddt_chunk_size` | 1, 2, 4, 8 | 更大 chunk 减少循环和重算 | OOM 或 lDDT 峰值过高 |
| `exp.model.blocks_per_ckpt` | 1, 2, 4, None | 更少 checkpoint 减少重算 | 激活显存上升 |
| eval/checkpoint interval | 400, 800, 1600 | 减少 I/O 和 eval 干扰 | checkpoint 粒度变粗，故障损失更大 |

实验规则：

- 每次只改一个主要变量。
- 固定数据路径、sampler/crop 设置、起始 checkpoint、image、训练脚本、GPU 卡数和节点数。
- 记录 speed + memory + loss finite，不把短跑速度等同于质量结论。
- 若变量会改变有效 batch 或训练语义，必须明确标记为非等价探索。
- 对候选配置必须进入短训质量阶段，比较首个 PBP 达标 checkpoint 的训练 elapsed time。

推荐优先级：

1. `torch.cuda.empty_cache()` 频率观测：保留每步、每 N 步、关闭三个模式，先确定是否有明显同步开销。
2. `diffusion_lddt_chunk_size` sweep：从 1 提到 2/4，通常是最直接的 memory-speed tradeoff。
3. `blocks_per_ckpt` sweep：确认当前 checkpoint 是否过度保守。
4. `num_dl_workers` sweep：确认是否 data-bound。
5. `train_batch_size` 和 `diffusion_batch_size` 组合 sweep：寻找单卡显存上限和有效吞吐上限。

### Phase 3：Profiling 驱动的代码级优化

目标：根据 Phase 0/1 的证据，对明确瓶颈做代码改造，以缩短固定资源/固定数据下的 `time_to_target_checkpoint`。

潜在方向：

1. 训练 loop instrumentation
   - 给 dataloader wait、H2D copy、model forward、loss、backward、optimizer step 加结构化计时。
   - 记录到 rank0 JSONL，避免只依赖 stdout。
   - 只在 profiling 模式打开，避免长期训练额外开销。

2. `empty_cache` 策略
   - 如果 profiling 证明每步 `torch.cuda.empty_cache()` 显著拖慢，改为可配置频率。
   - 默认策略应保守，不在未验证的正式训练中直接关闭。

3. Dataloader 和 batch 组装
   - 统计每个 batch 的真实 token/atom、padding ratio、collate time。
   - 如果 padding waste 很高，研究 token/atom bucket 或 token-balanced batch。
   - 如果 data wait 高，研究 worker 数、prefetch、persistent workers、pin memory。

4. Loss 计算路径
   - 比较 sparse/dense smooth lDDT 路径。
   - 调整 `diffusion_lddt_chunk_size` 和 checkpoint 粒度。
   - 对 lDDT/bond/MSE loss 分别计时，避免把 loss 当成黑盒。

5. Activation checkpoint 策略
   - 对 Pairformer、MSA、Diffusion transformer 分别做 checkpoint granularity sweep。
   - 目标是找到 H200 上的速度/显存 Pareto frontier。

6. Attention kernel 路径
   - 比较 DeepSpeed Evo attention on/off、bf16/TF32 路径。
   - 保留 strict replay 和 production training 的边界：性能路径可以用 TF32/DS kernel，等价性诊断不能混用。

### Phase 4：分布式和资源效率优化

目标：分别建立不同资源合同下的 time-to-target 曲线。该阶段不是用更多卡直接赢过更少卡，而是回答“在给定卡数和节点数时，哪套训练配置最快达标”。

实验维度：

- 1GPU、2GPU、4GPU、8GPU 单节点 scaling。
- 16GPU 跨节点 scaling。
- 不同 per-device batch 与 grad accumulation 的组合。
- NCCL env、hostNetwork、gang scheduling 对启动和通信的影响。
- checkpoint/eval/logging 对 rank0 和共享盘的压力。

核心指标：

- strong scaling：固定 global effective batch，增加 GPU 后 step time 是否下降。
- weak scaling：固定 per-GPU workload，增加 GPU 后 throughput 是否线性增加。
- communication overhead：DDP allreduce 占比。
- launch overhead：从 H200 `Inqueue/STARTING` 到训练第一步的时间。
- time-to-target per resource contract：每个固定资源合同下的首个 PBP 达标 checkpoint 时间。

决策边界：

- 如果 16GPU 长时间难调度，而 4/8GPU 在其自身资源合同下能更快稳定产出达标 checkpoint，可以考虑阶段性使用少卡高利用率训练；但这属于不同资源合同之间的研发策略比较，不应混入同一合同内的主指标。
- 如果跨节点通信占比过高，优先优化 batch/accumulation 或单节点吞吐，而不是盲目扩大 world size。

### Phase 5：收敛效率和质量闭环

目标：证明候选效率配置能更快产出 PBP 达标 checkpoint，而不是只在短跑中更快。

质量 gate：

- loss 曲线无异常震荡、NaN、持续退化。
- checkpoint 能正常加载并推理。
- PBP/ODesignBench 指标与历史 best setting 或 bsz1 baseline 比较。
- 对关键 slice 评估：大结构、小结构、高 padding ratio、不同 chain/type。

实验设计：

- 每个候选效率配置先跑短训，产出 checkpoint。
- 用相同 PBP/ODesignBench 流程评测 checkpoint。
- 对每个 checkpoint 记录训练 elapsed time、optimizer step、PBP 分数。
- 选择首个达到 `PBP_TARGET_SCORE` 的 checkpoint 作为该 run 的 `target_checkpoint`。
- 若短训未达标但 loss 曲线健康，再启动更长 resume 或 from0 run。
- 最终比较同一资源/数据合同下的 `time_to_target_checkpoint`，而不是只比较某个 checkpoint 的单点指标。

质量闭环报告必须包含：

- PBP target 分数和选择理由。
- 每个 checkpoint 的 step、写出时间、PBP 分数。
- 首个达标 checkpoint。
- 与 baseline 的 `time_to_target_checkpoint` 差异。
- 未达标 run 的 stop reason：训练时间不足、loss 异常、OOM、评测失败或质量退化。

## 分层实验矩阵

| 层级 | 目的 | 资源 | 运行长度 | 产物 |
| --- | --- | --- | --- | --- |
| micro profile | 拆 step time | 1GPU | 20-50 updates | profiler JSONL、GPU telemetry |
| config sweep | 找 Pareto frontier | 1GPU/4GPU | 50-100 updates | speed/memory 表 |
| scaling sweep | 看多卡效率 | 4GPU/8GPU/16GPU | 50-100 updates | scaling report |
| short quality run | 检查 loss、checkpoint 和早期 PBP 趋势 | 固定资源合同 | 400-1000 updates 或覆盖若干 checkpoint | checkpoint、loss curve、PBP timeline |
| evaluation gate | 质量闭环 | 评测资源 | 每个候选 checkpoint | PBP/ODesignBench report |
| promoted training | 正式验证 | 固定资源合同 | 直到达标或触发 stop criteria | time-to-target report |

## 指标和日志格式建议

每个训练 run 至少记录：

- `run_id`
- commit SHA 和关键文件 hash
- image、cluster、GPU type、replicas、GPU/rank 数
- data root、checkpoint path、config overrides
- resource contract：节点数、每节点 GPU 数、CPU/内存、hostNetwork、image
- data contract：索引、sampler、crop/mask、起点 checkpoint
- target contract：`PBP_TARGET_SCORE`、checkpoint interval、评测脚本版本
- per-step JSONL：
  - `global_step`
  - `optimizer_step`
  - `rank`
  - `data_wait_sec`
  - `to_device_sec`
  - `forward_sec`
  - `loss_sec`
  - `backward_sec`
  - `optimizer_sec`
  - `step_total_sec`
  - `num_tokens_real`
  - `num_atoms_real`
  - `num_tokens_padded`
  - `num_atoms_padded`
  - `padding_ratio`
  - `gpu_mem_allocated_mib`
  - `gpu_mem_reserved_mib`
  - `loss`
  - `lr`
- per-checkpoint JSONL：
  - `checkpoint_step`
  - `checkpoint_path`
  - `train_elapsed_sec`
  - `wall_clock_timestamp`
  - `pbp_score`
  - `pbp_eval_status`
  - `is_target_checkpoint`

rank0 汇总：

- mean/p50/p90 step time。
- warmup 后 samples/sec、tokens/sec、atoms/sec。
- GPU memory peak。
- dataloader wait 占比。
- forward/loss/backward/optimizer 占比。
- checkpoint/eval/logging 造成的 outlier step。
- checkpoint timeline 和首个达标 checkpoint。
- `time_to_target_checkpoint`，如果未达标则写明 `not_reached` 和 stop reason。

## 优先路线建议

第一轮不建议直接改模型结构，也不建议先改训练 setting。推荐路线是：

1. 加 profiling/instrumentation，证明瓶颈。
2. 在固定训练 setting 下做 operator-level attribution，先拆 Pairformer/MSA 的具体热点。
3. 只对 top 热点做等价算子/runtime 优化候选，并通过 forward/loss/grad/参数更新 allclose gate。
4. 再进入无 profiler speed run，比较同一资源/数据合同下的 microbatch time 和显存峰值。
5. 对通过 speed/memory gate 的候选跑到若干 checkpoint，并做 PBP/ODesignBench 评测。
6. 比较同一资源/数据合同下的 `time_to_target_checkpoint`。
7. 再决定是否推广到其他资源合同或进入配置 sweep。

最优先的具体问题：

1. Pairformer forward 里 triangle multiplication、triangle attention、transition、single attention/pair bias 分别占多少。
2. Pairformer backward 剩余时间里有多少来自 activation checkpoint recompute、attention kernel、matmul/einsum、transpose/contiguous 或 autograd 调度。
3. `use_deepspeed_evo_attention=true` 是否真实落到 Pairformer triangle attention 主路径，是否存在 fallback 或 build/cache 问题。
4. 是否存在可等价减少的 mask/bias materialize、layout conversion、无用同步或重复 kernel launch。
5. 只有上述证据不足或已优化后，再回到 `blocks_per_ckpt`、`diffusion_lddt_chunk_size`、batch 等配置变量。

## 风险和边界

- 短跑速度提升不代表能更快产出 PBP 达标 checkpoint。
- TF32、DeepSpeed Evo attention、checkpoint granularity 会影响数值路径；性能实验和 strict replay 诊断必须分开叙述。
- 更大 batch 或 diffusion batch 可能改变优化动态，即使 effective samples/update 不变，也需要质量 gate。
- 只看 samples/sec 可能误导；ODesign 的结构长度差异很大，必须同时看 tokens/sec、atoms/sec 和 padding ratio。
- H200 排队时间不是主训练时间指标的一部分，但会影响研发迭代效率，需要在调度策略中单独记录。
- 不同 GPU 卡数和节点数下的 time-to-target 可以作为资源策略参考，但不能和固定资源合同内的配置优化结论混为一谈。

## 下一步交付物

建议下一份文档/实现是：

1. `docs/odesign_operator_level_acceleration_plan_20260609.zh.md`
   - 明确固定训练 setting 的边界、允许优化类型、数值 gate、速度 gate 和质量 gate。
2. operator-level profiling instrumentation
   - 只在配置开关打开时记录 PyTorch profiler/NVTX ranges，不改变默认训练语义。
3. 2GPU H200 operator trace
   - 在现有 baseline setting 下跑 1-2 optimizer update，拆 Pairformer/MSA 子阶段和 top CUDA kernels。
4. `docs/odesign_operator_level_profile_YYYYMMDD.zh.md`
   - 汇总 triangle multiplication、triangle attention、transition、single attention/pair bias、outer product mean 和 kernel-level attribution。
5. 等价优化候选设计
   - 只针对 top-1/top-2 热点提出候选，并先跑 forward/loss/grad/参数更新 allclose gate。

当前不建议直接启动大规模正式效率训练。先拿到固定资源/固定数据下的 profiling baseline，再决定哪些优化值得进入“跑到 PBP 达标 checkpoint”的长训验证。

## 2026-06-09 E2 更新

E2 targeted profiling 已完成两项低风险检查：

- `ODESIGN_PROFILE_SYNC_CUDA=0` 对照没有显著改变稳态 microbatch time，profiling CUDA sync 不是主要瓶颈。
- `TrainRunner.train_step()` 末尾 `torch.cuda.empty_cache()` 从每 microbatch 调整为每 optimizer update 或关闭后，10 update 短跑 wall time 约改善 `1.0-1.2%`，但 forward/backward 仍占绝大多数时间。
- `exp.loss.diffusion_lddt_chunk_size=2` 相比 `1` 没有带来 wall-time 改善，并显著增加 PyTorch allocator 峰值。

因此，后续效率研发的优先级应前移到：

1. 针对 forward/backward compute path 做更细粒度 profiling，先拆 Pairformer/MSA operator 和 top CUDA kernels。
2. 在固定训练 setting 下设计等价 operator/runtime 优化候选，而不是继续扩大 profiling overhead 或 lDDT chunk sweep。
3. activation checkpoint granularity 和 `blocks_per_ckpt` sweep 暂缓到后续配置优化阶段；只有 operator-level attribution 证明重算是主瓶颈时再推进。
4. 对通过数值 gate 和 speed/memory gate 的候选，再进入短训质量 gate 和 PBP/ODesignBench gate。

## 2026-06-09 Module-Level Profiling 更新

Module-level profiling 已完成，详见 `docs/odesign_module_level_profile_20260609.zh.md`。

关键证据：

- 正式短跑 `module_profile_2gpu_10upd_20260609_r1`：`returncode=0`，`50 rows/rank`，无 hook warning/error。
- Forward 里 Pairformer 平均 `27.640s`，占 `forward` 的 `89.4%`；Diffusion 平均 `3.045s`，占 `forward` 的 `9.8%`。
- Backward hook 近似里 Pairformer 平均 `9.830s`，MSA 平均 `3.228s`，是已观测 hook 时间中最大的两项。
- 该 profiling 是诊断模式，包含 CUDA sync、stage memory peak 和 backward hook 开销；用于归因，不用于无 instrumentation 真实吞吐结论。

2026-06-09 用户进一步明确：训练效率优化第一阶段应保持训练 setting 不变，通过算子和 runtime 加速提升 forward/backward 速度。因此下面这条 module-level profile 后的早期建议被暂缓：

- 暂缓：`exp.model.blocks_per_ckpt` sweep：`1 -> 2 -> 4 -> None`。
- 原因：它虽然不改变模型函数本身，但会改变 activation checkpoint/recompute 策略，属于训练 runtime setting 变量，不适合作为“固定训练 setting”阶段的第一批实验。

新的下一轮优先级：

1. 按 `docs/odesign_operator_level_acceleration_plan_20260609.zh.md` 做 PyTorch profiler/NVTX operator-level attribution。
2. 把 Pairformer/MSA 拆到 triangle multiplication、triangle attention、transition、single attention/pair bias、outer product mean 和 top CUDA kernels。
3. 只对 top-1/top-2 热点设计等价算子/runtime 优化候选。
4. 候选先通过 forward/loss/grad/参数更新 allclose gate，再做无 profiler speed run。
5. 通过 speed/memory gate 后，再进入短训质量 gate 和 PBP/ODesignBench gate。
