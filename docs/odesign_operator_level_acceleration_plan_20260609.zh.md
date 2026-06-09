# ODesign Operator-Level 训练加速计划

本文档承接 `docs/odesign_training_efficiency_roadmap.zh.md` 和 `docs/odesign_module_level_profile_20260609.zh.md`。当前阶段的目标不是寻找新训练 setting，而是在保持训练语义不变的前提下，通过算子、runtime 和实现级优化提升 ODesign forward/backward 速度。

本文是研发计划和执行合同草案，不是速度提升结论。任何“加速有效”的 claim 都必须同时给出数值等价、速度、显存和质量边界证据。

## 当前约束

用户明确要求第一阶段不改变模型训练 setting：

- 不改变 forward 实际经过的模型模块。
- 不改变 loss、数据、sampler、crop、augmentation、batch、gradient accumulation、diffusion batch、checkpoint/eval 节奏。
- 不改变模型参数 dtype、训练 dtype、TF32/AMP/DeepSpeed Evo attention 等既有精度策略。
- 不把 activation checkpoint granularity、`diffusion_lddt_chunk_size`、`train_batch_size` 等 memory-speed tradeoff 当作当前第一优先级。
- 当前阶段只研究等价算子实现、kernel dispatch、runtime 调度、tensor layout、无用同步和低层实现效率。

因此，上一版文档中建议的 `exp.model.blocks_per_ckpt` sweep 暂缓。它仍然是一个有价值的后续 memory-speed 变量，但会改变训练 runtime setting 和 backward 重算路径，不适合作为“固定训练 setting”阶段的第一批实验。

## 已有证据

已完成的 module-level profiling 说明：

| 证据 | 结论 |
| --- | --- |
| `module_profile_2gpu_10upd_20260609_r1` | `returncode=0`，2GPU H200，100 rank-level rows |
| `forward_sec` | mean `30.921s/microbatch` |
| `module_profile_pairformer_sec` | mean `27.640s`，占 forward `89.4%` |
| `module_profile_diffusion_sec` | mean `3.045s`，占 forward `9.8%` |
| `backward_sec` | mean `29.790s/microbatch` |
| backward hook | Pairformer mean `9.830s`，MSA mean `3.228s` |

当前可以确定：

- 主瓶颈在 Pairformer/MSA 相关 forward/backward compute path。
- 现有 module timer 只能定位到大模块，不能定位到具体 CUDA kernel、`aten::matmul`、triangle attention、triangle multiplication、transition 或 checkpoint recompute。
- 下一步需要 operator-level attribution，而不是直接改训练配置。

## 训练语义不变的判定

一个候选优化只有同时满足下列条件，才可以进入 speed gate：

| 维度 | 必须保持不变 |
| --- | --- |
| 模型路径 | `ODesign.forward(mode="train")` 经过同样的 input embedding、MSA、Pairformer、Diffusion、pairwise head、loss 路径 |
| 配置 | 训练 YAML、overrides、batch、accumulation、crop、diffusion batch、loss chunk、checkpoint interval 不变 |
| 精度 | 参数 dtype、训练 dtype、autocast/TF32/DeepSpeed Evo attention 既有策略不变 |
| 数据 | 同一数据 root、index、sampler seed、样本 trace 和 batch 内容 |
| 随机性 | dropout、MSA sampling、diffusion noise、permutation、crop 等受控或复用同一 trace |
| 输出 | 代表性 forward tensors、loss、grad summaries、关键参数更新 allclose |

允许出现的差异：

- 等价 kernel 因 reduction order 不同导致的微小数值差异。
- profiler/NVTX record function 造成的诊断开销。

不允许出现的差异：

- 为了快而跳过某个模块、缩短 block 数、减少 diffusion sample、降低 loss 计算、改变 batch 组成或切换训练精度。
- 只用 loss finite 证明等价。
- 只看 step time，不比较 forward/backward 数值。

## 允许和暂缓的优化类型

### 当前允许优先研究

| 类型 | 例子 | 进入条件 |
| --- | --- | --- |
| operator profiling | PyTorch profiler、NVTX ranges、record_shapes、CUDA kernel table | 默认关闭；只在 profiling run 打开 |
| tensor layout 优化 | 减少不必要的 `contiguous()`、`transpose()` 后重复 materialize、重复 mask/bias 构造 | allclose；不能改变语义 |
| 等价 fused elementwise | gate、silu、mask multiply、bias add 等局部融合 | forward/grad allclose；速度 profile 证明收益 |
| kernel dispatch 审计 | 确认 Pairformer 主路径是否实际走 DeepSpeed Evo attention 或 stock attention | 先审计，不先切换 |
| runtime 初始化优化 | DeepSpeed/extension build cache、避免重复 import/build、避免诊断同步进入正式训练 | 不改变训练数学路径 |
| 编译型优化 | 对局部纯函数尝试 `torch.compile` 或专门 fused kernel | 先小模块 allclose，再真实 batch allclose，再短 profile |

### 当前暂缓

| 类型 | 暂缓原因 |
| --- | --- |
| `blocks_per_ckpt` sweep | 改变 activation checkpoint/recompute 策略，属于 runtime setting 变量 |
| `diffusion_lddt_chunk_size` sweep | 改变 loss 内部 chunking 和显存/速度 tradeoff |
| `train_batch_size` / `iters_to_accumulate` | 影响 batch 形态、sample trace、DDP 梯度平均路径 |
| `diffusion_batch_size` | 改变 diffusion sample 维度和 loss 计算规模 |
| TF32 on/off、bf16/fp32、flash half path | 改变精度或数学路径，不能作为当前固定 setting 优化 |
| 关闭 MSA/Pairformer/Diffusion 子模块 | 改变模型路径，禁止 |

## 需要拆开的代码路径

当前 profile 已经把 forward 主瓶颈收敛到 Pairformer。下一层应拆到这些粒度：

| 模块 | 文件 | 子阶段 |
| --- | --- | --- |
| `PairformerBlock` | `src/model/modules/pairformer.py` | `tri_mul_out`、`tri_mul_in`、`tri_att_start`、`tri_att_end`、`pair_transition`、`attention_pair_bias`、`single_transition` |
| `PairformerStack` | `src/model/modules/pairformer.py` | block index、checkpoint block group、首尾 block outlier |
| `MSABlock` | `src/model/modules/pairformer.py` | `outer_product_mean_msa`、`msa_stack`、inner `pair_stack` |
| `TriangleAttention` | `src/utils/openfold_local/model/triangular_attention.py` | layer norm、mask bias、triangle bias、attention kernel、transpose |
| openfold `Attention` | `src/utils/openfold_local/model/primitives.py` | qkv projection、DeepSpeed Evo / stock attention / LMA / flash branch、gating、output projection |
| `TriangleMultiplicativeUpdate` | `src/utils/openfold_local/model/triangular_multiplicative_update.py` | projections、`_combine_projections` matmul/einsum、out projection、gate |
| `OuterProductMean` | `src/utils/openfold_local/model/outer_product_mean.py` | layer norm、projection、outer/einsum、output projection |
| `Transition` | `src/model/modules/primitives.py` | layer norm、two linear projections、silu-gate multiply、output projection |

一个关键审计点是：Pairformer triangle attention 使用 `src/utils/openfold_local/model/primitives.py` 的 `Attention`，不是 `src/model/modules/primitives.py` 的 `Attention`。因此需要实测生产配置中的 `use_deepspeed_evo_attention=true` 是否真的作用在 Pairformer 主路径，并在 profiler trace 中看到对应 kernel。

## Phase A：Operator-Level Attribution

目标：用最短运行拿到足够细的 Pairformer/MSA 算子耗时表。

建议实现：

1. 保留现有 `ODESIGN_PROFILE_MODULES` 大模块计时。
2. 新增可选 `ODESIGN_PROFILE_PAIRFORMER_DETAIL=1`，在 `PairformerBlock`、`MSABlock`、`TriangleAttention`、`TriangleMultiplicativeUpdate` 周围加 `torch.autograd.profiler.record_function` 或 NVTX range。
3. 用 PyTorch profiler 跑 1-2 个 optimizer update，只抓少量 active microbatch，避免 profiler 文件过大。
4. 输出两类 artifact：
   - profiler trace：用于 Chrome trace / TensorBoard 分析。
   - rank0 summary table：按 `self_cuda_time_total`、`cuda_time_total`、调用次数、shape 聚合。

建议 profiler 参数：

```python
torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
    record_shapes=True,
    profile_memory=True,
    with_stack=False,
    with_modules=True,
)
```

已实现的开关：

| env | 默认 | 作用 |
| --- | --- | --- |
| `ODESIGN_PROFILE_PAIRFORMER_DETAIL` | `0` | 递归打开 Pairformer/MSA/openfold-local 子模块的 `record_function` ranges |
| `ODESIGN_TORCH_PROFILER_DIR` | empty | 非空时启用 PyTorch profiler，并把 TensorBoard trace 写到 `${DIR}/rankXX/` |
| `ODESIGN_TORCH_PROFILER_ALL_RANKS` | `0` | 默认只 profile rank0；设为 `1` 时所有 rank 都写 trace |
| `ODESIGN_TORCH_PROFILER_WAIT` | `1` | profiler schedule wait steps |
| `ODESIGN_TORCH_PROFILER_WARMUP` | `1` | profiler schedule warmup steps |
| `ODESIGN_TORCH_PROFILER_ACTIVE` | `2` | profiler schedule active steps |
| `ODESIGN_TORCH_PROFILER_REPEAT` | `1` | profiler schedule repeat count |
| `ODESIGN_TORCH_PROFILER_RECORD_SHAPES` | `true` | 是否记录 tensor shapes |
| `ODESIGN_TORCH_PROFILER_PROFILE_MEMORY` | `true` | 是否记录 profiler memory |
| `ODESIGN_TORCH_PROFILER_WITH_STACK` | `false` | 是否记录 Python stack；默认关闭以控制 trace 体积 |
| `ODESIGN_TORCH_PROFILER_WITH_MODULES` | `true` | 是否记录 module 信息 |

这些开关默认都不改变训练路径。只有显式设置 `ODESIGN_PROFILE_PAIRFORMER_DETAIL=1` 或 `ODESIGN_TORCH_PROFILER_DIR` 时才增加 profiling 行为。

建议 2GPU trace run 的关键 env：

```bash
export ODESIGN_PROFILE_PAIRFORMER_DETAIL=1
export ODESIGN_TORCH_PROFILER_DIR="${RECORD_DIR}/torch_profiler"
export ODESIGN_TORCH_PROFILER_ALL_RANKS=0
export ODESIGN_TORCH_PROFILER_WAIT=1
export ODESIGN_TORCH_PROFILER_WARMUP=1
export ODESIGN_TORCH_PROFILER_ACTIVE=2
export ODESIGN_TORCH_PROFILER_REPEAT=1
```

如果复用 `.cluster_operator/run_efficiency_profile_e1_2gpu_0608.sh`，这些 env 可以在调用脚本前导出；脚本会把 env 传给 `scripts/train.py`，不需要改训练 override。

运行合同：

| 项目 | 设置 |
| --- | --- |
| 资源 | 用户提供的 2GPU H200 长时 pod |
| 数据 | `/mnt/shared-storage-user/ai4sreason/scireason_2/data/odesign` |
| checkpoint | `/mnt/shared-storage-user/ai4sreason/scireason_2/data/odesign/ckpt/protenix_base_default_v0.5.0.pt` |
| 训练配置 | 与 `module_profile_2gpu_10upd_20260609_r1` 相同 |
| 长度 | 1-2 optimizer update，active 2-4 个 microbatch |
| 成功条件 | trace 写出、无 OOM、无 profiler warning 破坏训练、能定位 Pairformer 子阶段和 top CUDA kernels |

本阶段不 claim 速度提升，只 claim attribution。

## 2026-06-09 实现和 smoke 验证

已实现：

- `src/utils/model/profiling.py`
  - `set_odesign_profile_detail_enabled()` 递归设置 detail flag。
  - `odesign_record_function()` 在 detail flag 打开时进入 `torch.autograd.profiler.record_function()`。
- `src/utils/train/train_runner.py`
  - 新增 `ODESIGN_PROFILE_PAIRFORMER_DETAIL`。
  - 新增 `ODESIGN_TORCH_PROFILER_DIR` 和 schedule/env 控制。
  - 每个 microbatch 结束后调用 `profiler.step()`。
  - profiling JSONL 写出 `profile_pairformer_detail` 和 `torch_profiler_enabled`。
- Pairformer/MSA/openfold-local 子路径已添加 ranges：
  - `odesign.pairformer_block.tri_mul_out`
  - `odesign.pairformer_block.tri_mul_in`
  - `odesign.pairformer_block.tri_att_start`
  - `odesign.pairformer_block.tri_att_end`
  - `odesign.pairformer_block.pair_transition`
  - `odesign.pairformer_block.attention_pair_bias`
  - `odesign.pairformer_block.single_transition`
  - `odesign.msa_block.outer_product_mean`
  - `odesign.msa_block.msa_stack`
  - `odesign.msa_block.pair_stack`
  - `odesign.triangle_attention.*`
  - `odesign.triangle_multiplication.*`
  - `odesign.outer_product_mean.*`
  - `odesign.openfold_attention.prep_qkv`
  - `odesign.openfold_attention.deepspeed_evo`
  - `odesign.openfold_attention.stock`
  - `odesign.openfold_attention.wrap_up`
  - `odesign.transition.*`

已验证：

| gate | evidence |
| --- | --- |
| local py_compile | operator profiling touched files pass |
| remote py_compile | H200 `odesign` env pass |
| remote unit tests | `test_odesign_operator_profiling.py`: `Ran 4 tests OK`; `test_train_runner_efficiency_controls.py`: `Ran 12 tests OK`; `test_odesign_module_profiling.py`: `Ran 2 tests OK` |
| profiler context smoke | `.cluster_operator/operator_profiler_context_smoke_0609/rank00/*.pt.trace.json` 写出 1 个 trace |
| Pairformer range smoke | `.cluster_operator/operator_pairformer_range_smoke_0609/rank00/*.pt.trace.json` 写出 1 个 trace，trace 中包含 `odesign.pairformer_block.tri_mul_out`、`odesign.pairformer_block.tri_att_start`、`odesign.openfold_attention.stock` |

已在真实 2GPU ODesign 训练 trace 中进一步验证：

- `operator_profile_2gpu_1upd_20260609_r1`：`returncode=0`，rank0 写出约 `2.07GB` PyTorch trace。
- `docs/odesign_operator_level_profile_20260609.zh.md` 记录了 run 合同、range 表、top kernel/op 表和结论边界。
- Pairformer openfold-local attention 在生产配置中确认走 `odesign.openfold_attention.deepspeed_evo`，`stock` branch 为 `0`。

仍未验证：

- 尚未设计或验证任何加速候选。
- 尚未跑无 profiler 的 before/after speed benchmark。
- 尚未证明任何优化对 PBP 或 time-to-target 有收益。

## Phase B：等价优化候选

只有 Phase A 证明具体热点后，才进入候选优化。建议按风险从低到高推进。

### B1：DeepSpeed Evo attention 路径审计和缓存优化

问题：

- 当前正式训练配置历史上使用 `exp.model.use_deepspeed_evo_attention=true`。
- openfold-local `Attention` 支持 DeepSpeed Evo branch，但需要确认 Pairformer triangle attention 是否实际进入该 branch。
- 如果 kernel 已经启用，优先优化 extension build/cache、启动 warmup、避免 fallback，而不是切换数学路径。

验证：

- profiler trace 中能看到 DS4Sci Evo attention 或明确看到 stock matmul/softmax fallback。
- 同一 batch 下 before/after forward、loss、grad allclose。
- 只优化初始化/cache 时，不应改变 step 内数值。

### B2：Attention bias/mask 构造和 layout

问题：

- `TriangleAttention.forward()` 每次构造 `mask_bias`、`triangle_bias`，并做 transpose/unsqueeze。
- Pairformer block 里 `tri_att_end` 前后会对 `z` 做 transpose；某些路径可能触发隐式 contiguous 或低效 stride 访问。

候选：

- 减少重复 materialize 的 mask/bias。
- 只在 kernel 必需时做 contiguous。
- 在不改变 shape/broadcast 语义的前提下复用中间 layout。

验证：

- 单个 `PairformerBlock` 小张量 forward/backward allclose。
- 真实 batch 的关键 activation、loss、grad summary allclose。
- profiler 看到 memory copy、transpose、contiguous 或 bias construction 时间下降。

### B3：Triangle multiplication 局部融合

问题：

- `TriangleMultiplicativeUpdate.forward()` 包含 layer norm、多个 linear、sigmoid gate、mask multiply、combine projections、out projection。
- 如果 Phase A 显示其占比高，可以研究局部融合或更高效的 einsum/matmul layout。

候选：

- gate/projection 局部 fusion。
- `_combine_projections` 的 layout/contiguous 优化。
- 避免重复 std 检查或 dtype 分支开销，但不能改变 fp16/bf16 overflow 防护语义。

验证：

- 小模块 `gradcheck` 或 fp32 strict allclose。
- bf16 production dtype 下 loss/grad allclose。
- profiler 证明 hot kernel 数量或 self CUDA time 下降。

### B4：Transition/SwiGLU 等价 fused path

问题：

- `Transition.forward()` 是 `LayerNorm -> Linear a/b -> silu(a) * b -> Linear`。
- 如果 profiler 显示 transition 占比非忽略，可以用等价 fused activation/gating 或 compile 局部模块。

验证：

- dtype 不变。
- forward/grad allclose。
- 真实 batch profile 有收益，否则不合入。

### B5：局部 `torch.compile`

问题：

- Pairformer 整体动态图、checkpoint 和 DDP 可能不适合一次性 compile。
- 但局部纯模块，如 Transition 或小的 bias/mask helper，可能适合 compile。

策略：

- 先只 compile 单个局部 helper 或小模块。
- 禁止为了 compile 改模型路径。
- 若 graph break、显存上升或首次编译时间过大，应停止。

验证：

- 独立单元测试 allclose。
- 真实 2GPU smoke。
- 记录 warmup 后速度，不能把 compile 首次开销混入训练吞吐结论。

## 数值验证 gate

每个候选优化必须至少通过以下三层。

| Gate | 内容 | 通过标准 |
| --- | --- | --- |
| G1 单模块 | 固定 seed、固定输入，比较子模块输出和参数梯度 | fp32 strict allclose；bf16 使用声明 tolerance |
| G2 单 step | 固定真实 batch trace，一次 forward/backward，比较 loss、关键 activation、grad summary | allclose，且无 NaN/OOM |
| G3 多 step replay | 2-3 个 optimizer step，固定 sample trace、随机性和初始 checkpoint，比较每步 loss、grad summary、参数 hash/allclose | 与已有 padding batch replay 口径一致 |

建议 tolerance：

- fp32/no-TF32 诊断：优先 `1e-5` 或更严格，按张量规模记录实际 max diff。
- production bf16/TF32/DeepSpeed 路径：不追求 bitwise，使用事先声明的 `atol/rtol`，并记录 max/mean diff 和 top offender。
- 如果候选 kernel 内部改变 accumulation order，必须把差异归因写清楚，不能只说“通过”。

## 速度和显存 gate

数值通过后再看速度：

| 指标 | 要求 |
| --- | --- |
| microbatch time | warmup 后 mean/p50/p90，与 baseline 同口径比较 |
| top kernel time | Phase A 对应热点应下降，否则只是噪声 |
| GPU memory | peak allocated/reserved 不应超过资源合同可承受范围 |
| loss finite | 全程有限，但 loss finite 不是等价证明 |
| instrumentation overhead | profiling run 和正式 speed run 分开 |

候选优化的合入标准建议：

- 在 2GPU H200 短跑中，warmup 后 microbatch mean 至少改善 `>=3%`，或单一热点 kernel 改善 `>=10%` 且无显存/数值风险。
- 小于 `1-2%` 的改善先记录为低优先级，不急于合入，除非实现极小且风险极低。

## 质量 gate

operator-level 优化即使数值 allclose，也仍需经过短训质量 gate 才能进入正式长训：

1. 2GPU 或 4GPU short quality run，写出 checkpoint。
2. checkpoint 可加载、可推理。
3. PBP/ODesignBench 与当前 baseline 相同流程比较。
4. 同一资源/数据合同下记录 `time_to_target_checkpoint`，如果未达标则记录 `not_reached` 和 stop reason。

当前阶段不要跳过 attribution 和数值 gate 直接启动正式长训。

## 推荐下一步

最直接的下一步是：

1. 新增 `ODESIGN_PROFILE_PAIRFORMER_DETAIL` 或 PyTorch profiler runner，不改默认训练路径。
2. 在 2GPU H200 pod 上跑 1-2 optimizer update 的 operator-level trace。
3. 产出 `docs/odesign_operator_level_profile_YYYYMMDD.zh.md`，列出 Pairformer/MSA 子阶段和 top CUDA kernels。
4. 只选择 top-1 或 top-2 热点设计一个等价优化候选。
5. 对候选先跑 G1/G2/G3 数值 gate，再做无 profiler 的 speed run。

当前不建议马上做：

- `blocks_per_ckpt` sweep。
- 更大 batch 或 diffusion batch sweep。
- 改 TF32/bf16/fp32 策略。
- 改模型结构或减少模块。
