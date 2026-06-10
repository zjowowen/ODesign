# ODesign Operator Sync/Copy 审计和首个等价优化

本文承接 `docs/odesign_operator_level_profile_20260609.zh.md`。目标是在不改变训练 setting、模型路径、loss、batch、diffusion batch、lDDT chunk、精度策略和 DeepSpeed Evo attention 的前提下，继续排查 r1 operator trace 中的 `aten::item/_local_scalar_dense/is_nonzero` 和 `aten::to/_to_copy/copy_` 热点。

本文包含一个已实施的低风险等价优化：默认关闭 MSAStack 训练路径中的 padding roundtrip 全量 tensor assert。它不改变 forward 数学输出；如果需要重新打开该调试校验，可以设置 `ODESIGN_DEBUG_MSA_PADDING_CHECK=1`。

本文仍不是最终加速结论。速度收益必须通过无 PyTorch profiler 的 before/after speed gate 才能 claim。

## 证据来源

| item | value |
| --- | --- |
| trace run | `operator_profile_2gpu_1upd_20260609_r1` |
| resource | 用户提供的 2GPU H200 pod `zjow-odesign-pbp-efficiency-2gpu-4607978-8rw86` |
| record dir | `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/module-profile-runtime-0609/ODesign/.cluster_operator/operator_profile_2gpu_1upd_20260609_r1` |
| production-like fixed setting | `train_batch_size=2`, `iters_to_accumulate=5`, `diffusion_batch_size=48`, `diffusion_lddt_chunk_size=1`, `use_deepspeed_evo_attention=true`, `NVIDIA_TF32_OVERRIDE=1` |
| trace parser | `scripts/summarize_torch_trace_ops.py`, `scripts/summarize_torch_trace_op_ranges.py` |
| parsed artifacts | `${RECORD_DIR}/torch_trace_ops_summary.md`, `${RECORD_DIR}/torch_trace_op_ranges_summary.md` |

## 新增解析工具

新增两个标准库流式解析脚本，避免把 2GB Chrome trace 一次性载入内存：

| file | purpose |
| --- | --- |
| `scripts/summarize_torch_trace_ops.py` | 按 op、input type、input dims 汇总选定 PyTorch `cpu_op` |
| `scripts/summarize_torch_trace_op_ranges.py` | 把选定 PyTorch op 归因到最内层 `odesign.*` `user_annotation` range |

默认关注：

```text
aten::to
aten::_to_copy
aten::copy_
aten::item
aten::_local_scalar_dense
aten::is_nonzero
```

对应测试：

| test | coverage |
| --- | --- |
| `tests/test_torch_trace_ops_summary.py` | shape/type signature 聚合，忽略非目标 category |
| `tests/test_torch_trace_op_ranges_summary.py` | 选最内层 enclosing range，外部 op 归到 `<outside_odesign_range>` |

## r1 Op Shape 观察

r1 选定 op 总数 `294096`。

| op | count | total | interpretation |
| --- | ---: | ---: | --- |
| `aten::copy_` | `110840` | `30409.530ms` | dtype/layout/materialize 路径总量最大 |
| `aten::to` | `78647` | `23505.810ms` | dtype/device/layout 转换 wrapper |
| `aten::_to_copy` | `66247` | `23455.924ms` | `to` 的 copy path |
| `aten::item` | `17548` | `15883.727ms` | Python scalar extraction，同步候选 |
| `aten::_local_scalar_dense` | `17548` | `15874.016ms` | `item` 底层 scalar path |
| `aten::is_nonzero` | `3266` | `6500.078ms` | tensor bool 转 Python bool |

前排 shape bucket 中有三类信号：

1. `msa_stack` scalar sync：`aten::is_nonzero/item/_local_scalar_dense`，input dims `[[]]`，42 次合计约 `6.43s`。
2. 外部大 long tensor copy：`[[2,4,2212]]`、`[[2,4,1990]]`，大概率来自 loss/permutation/coordinate label 或 batch 外围路径，需另加 range 或 stack trace 归因。
3. 模块内参数/activation dtype copy：`[128,128]`、`[128]`、`[384,1536]`、`[16,128]` 等，集中在 triangle multiplication、attention pair bias、transition、openfold attention prep。

## r1 Range 归因

按最内层 `odesign.*` range 归因后，前排如下：

| range | target ops | total |
| --- | ---: | ---: |
| `<outside_odesign_range>` | `129178` | `77691.632ms` |
| `odesign.msa_block.msa_stack` | `11298` | `19438.345ms` |
| `odesign.triangle_multiplication.output` | `19656` | `7570.366ms` |
| `odesign.pairformer_block.attention_pair_bias` | `35616` | `3680.455ms` |
| `odesign.triangle_multiplication.projections` | `27216` | `2696.596ms` |
| `odesign.transition.gate_output` | `5436` | `1414.716ms` |
| `odesign.openfold_attention.prep_qkv` | `13608` | `1151.885ms` |

最关键的 scalar sync 归因：

| range | op | count | total | mean |
| --- | --- | ---: | ---: | ---: |
| `odesign.msa_block.msa_stack` | `aten::is_nonzero` | `42` | `6432.316ms` | `153.150ms` |
| `odesign.msa_block.msa_stack` | `aten::item` | `42` | `6432.254ms` | `153.149ms` |
| `odesign.msa_block.msa_stack` | `aten::_local_scalar_dense` | `42` | `6432.179ms` | `153.147ms` |

这个数量和 `odesign.msa_block.msa_stack` range count `42` 精确一致，说明它大概率是每次 MSAStack forward 固定触发一次的 Python tensor bool，而不是随机日志或偶发异常路径。

## Scalar Sync 根因

### 已排除的候选：triangle multiplication fp16 guard

`src/utils/openfold_local/model/triangular_multiplicative_update.py` 中有：

```python
a_std = a.std()
b_std = b.std()
if is_fp16_enabled() and a_std != 0.0 and b_std != 0.0:
    ...
```

H200 micro probe 显示：

| autocast dtype | `is_fp16_enabled()` | target scalar ops |
| --- | --- | --- |
| `torch.bfloat16` | `False` | 只有 `aten::std`，不触发 `is_nonzero/item` |
| `torch.float16` | `True` | 触发 `aten::is_nonzero`、`aten::item`、`aten::_local_scalar_dense` |

当前 baseline 配置是 `configs/model/_base.yaml: dtype: bf16`，训练入口用 `torch.autocast(device_type="cuda", dtype=torch.bfloat16, cache_enabled=False)`。因此这个 fp16 guard 不能解释当前 r1 的 `msa_stack` 42 次同步；它只是未来若切 fp16 时需要单独处理的潜在问题。

### 已定位的根因：MSAStack padding assert

`src/model/modules/pairformer.py` 训练分支原代码：

```python
m_new = pad_at_dim(
    m, dim=-3, pad_length=(0, self.msa_max_size - m.shape[-3]), value=0
)
assert (_slice_msa_rows(m_new, m.shape[-3]) == m).all()
```

这个 assert 会做一次 GPU tensor 全量比较和 `.all()`，再把 bool tensor 转成 Python bool。该模式正对应 trace 中的：

```text
aten::is_nonzero -> aten::item -> aten::_local_scalar_dense
```

归因链条：

| evidence | conclusion |
| --- | --- |
| `msa_stack` range count 是 `42` | 每次 MSAStack 调用固定一次 |
| `msa_stack` 中 `is_nonzero/item/_local_scalar_dense` count 也是 `42` | 数量完全匹配 |
| 源码中 `MSAStack.forward()` 训练分支有 `assert tensor.all()` | 模式完全匹配 |
| H200 红灯测试 patch `torch.Tensor.all` 后失败在该行 | 证明默认训练路径会触发 `.all()` |
| bf16 micro probe 排除 triangle multiplication fp16 guard | 避免误归因 |

## 已实施改动

把该 assert 改为默认关闭的 debug check：

```python
def _debug_msa_padding_check_enabled() -> bool:
    value = os.environ.get("ODESIGN_DEBUG_MSA_PADDING_CHECK", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _assert_msa_padding_roundtrip(padded_msa: torch.Tensor, msa: torch.Tensor) -> None:
    if not torch.equal(_slice_msa_rows(padded_msa, msa.shape[-3]), msa):
        raise AssertionError("MSA padding roundtrip changed the real MSA rows")
```

训练路径：

```python
m_new = pad_at_dim(
    m, dim=-3, pad_length=(0, self.msa_max_size - m.shape[-3]), value=0
)
if _debug_msa_padding_check_enabled():
    _assert_msa_padding_roundtrip(m_new, m)
```

语义边界：

- 默认正式训练不再做 GPU tensor 全量比较和 Python bool extraction。
- forward 数学输出不变；删除的是只读 sanity check。
- 如果怀疑 padding 工具出错，可设置 `ODESIGN_DEBUG_MSA_PADDING_CHECK=1` 重新打开检查。
- debug 检查本身仍会同步，不应在正式 speed run 中打开。

## 已验证

### TDD 红灯

在远端 H200 `odesign` 环境中，先只加测试、不改实现，调用：

```text
tests/test_msa_token_mask.py::test_msa_stack_train_forward_skips_padding_roundtrip_check_by_default
```

结果按预期失败：

```text
RuntimeError: unexpected tensor all
...
src/model/modules/pairformer.py:675
assert (_slice_msa_rows(m_new, m.shape[-3]) == m).all()
```

### 绿灯

实现后远端 importlib runner 调用 `tests/test_msa_token_mask.py` 中所有 `test_*` 函数：

```text
PASS test_add_single_embedding_to_msa_inserts_msa_axis_for_batched_inputs
PASS test_apply_msa_token_mask_broadcasts_batched_token_mask_over_msa_axis
PASS test_assert_msa_padding_roundtrip_detects_mismatched_real_rows
PASS test_chunk_msa_rows_splits_negative_three_axis
PASS test_msa_stack_inference_forward_chunks_msa_axis_with_batch_prefix
PASS test_msa_stack_train_forward_skips_padding_roundtrip_check_by_default
PASS test_slice_msa_rows_preserves_batch_axis
```

### 小 profiler probe

在远端 H200 上构造一个小 MSAStack training forward，stub 掉 chunk 内部模块，只验证 padding/default path：

```text
out_equal True
```

probe 没有打印任何 `aten::is_nonzero`、`aten::item`、`aten::_local_scalar_dense` 目标事件，说明默认训练路径的该同步点已经移除。

本地系统 Python 缺 `optree`，无法直接 import Pairformer 跑模型测试；模型相关验证均在远端 `odesign` conda 环境完成。

## Copy/To 热点审计

### 已验证事实

模块内 `to/_to_copy/copy_` 主要集中在：

| range | dominant pattern |
| --- | --- |
| `odesign.triangle_multiplication.output` | `[128,128]`、`[128]` 权重/bias 或中间 dtype copy |
| `odesign.triangle_multiplication.projections` | `[128,128]`、`[128]` dtype copy |
| `odesign.pairformer_block.attention_pair_bias` | `[16,128]`、`[384,384]`、`[2,16,24,640]`、`[2,640,640]` |
| `odesign.transition.gate_output` | `[384,1536]` |
| `odesign.openfold_attention.prep_qkv` | `[128,128]` |
| `<outside_odesign_range>` | 大 long tensor `[[2,4,2212]]`、`[[2,4,1990]]` 以及若干 loss/permutation 相关 scalar item |

### 当前推断

OpenFold local 的 `Linear`/`LayerNorm` 为 bf16 输入提供了手动 cast path。例如 `src/utils/openfold_local/model/primitives.py` 中，当输入 `d is torch.bfloat16` 且 DeepSpeed comm 未初始化时，会在 autocast disabled 下调用：

```python
nn.functional.linear(input, self.weight.to(dtype=d), bias)
```

这能解释大量 `[128,128]`、`[128]`、`[384,1536]` 的 float/bf16 copy。但这不是可以直接删除的无用 copy，因为：

- 参数常驻通常仍是 fp32；
- 当前训练显式 `cache_enabled=False`；
- 改 cast/cache 策略可能改变精度、autograd 保存张量或内存峰值；
- DeepSpeed comm 初始化状态会影响分支。

因此 copy/to 优化需要单独 allclose gate，不应和本次 assert 同步优化混在一起。

### 未验证边界

- 还没有用 Python stack trace 精确定位 `<outside_odesign_range>` 的大 long tensor copy。
- 还没有证明 OpenFold local `Linear/LayerNorm` cast 是可安全减少的冗余 copy。
- 还没有无 profiler speed run 证明任何 copy/to 优化收益。

## 下一步实验

### E3A：MSA padding assert removal speed gate

目的：验证移除默认 padding assert 后，真实训练无 PyTorch profiler 的 step time 是否改善。

| item | plan |
| --- | --- |
| baseline code | `c5a5b52`，保留原 assert |
| candidate code | 当前分支，默认 `ODESIGN_DEBUG_MSA_PADDING_CHECK` unset |
| resource | 同一个 2GPU H200 长时 pod，或同规格 H200 pod |
| training setting | 与 r1/r2 相同，固定 `train_batch_size=2`、`gacc=5`、`diffusion_batch_size=48`、`lddt_chunk=1`、bf16、Evo attention、TF32 |
| profiler | 不开 PyTorch profiler；可保留 module JSONL timing，但 before/after 必须一致 |
| length | 建议 5-10 optimizer updates，丢弃 cold start |
| metrics | `forward_sec`、`module_profile_pairformer_sec`、`module_profile_msa_sec` 或 MSA range proxy、`microbatch_total_sec`、peak memory |
| success criterion | 在同设置下 p50/mean 有超过噪声的稳定改善，且 loss finite、无 graph/static DDP 错误 |
| failure criterion | 速度无改善、显存异常、训练报错、或 debug check 打开/关闭导致输出不一致 |

预期：r1 profiler active window 中该同步约占 `6.43s` CPU span，但这不是可直接外推的吞吐收益。正式收益可能小于这个数，也可能被其它同步/通信隐藏。

### E3B：Copy/to 细归因

目的：把剩余 copy/to 分成“参数 dtype cast”、“activation layout materialize”、“外部 loss/permutation 大 tensor copy”三类，再决定是否存在安全优化。

建议：

1. 给 MSAStack 内部加临时细 range：`pad_msa`、`msa_pair_weighted_chunk`、`msa_transition_chunk`。先只在 profiling detail flag 下启用。
2. 对 `<outside_odesign_range>` 增加 loss/permutation/data-to-device 周边 range，而不是直接开全量 `with_stack=1`。
3. 对 OpenFold local `Linear/LayerNorm` 做小模块 profiler：比较 fp32 params + bf16 autocast 下的 cast 次数，确认 `cache_enabled=False` 的影响。
4. 任何减少 cast/cache 的改动必须先做 module-level forward/grad allclose，再做真实 batch single-step allclose。

## 当前状态

| item | status |
| --- | --- |
| scalar sync root cause | `msa_stack` 42 次长同步已定位到训练分支 padding assert |
| scalar sync fix | 已默认关闭该 assert，保留 `ODESIGN_DEBUG_MSA_PADDING_CHECK=1` debug gate |
| unit/micro verification | H200 `odesign` 环境通过 |
| speed claim | 未 claim，等待 E3A |
| copy/to root cause | 已完成 range/shape 归因，尚未进入优化 |
| next deliverable | E3A before/after speed gate 报告，或 E3B copy/to 细归因报告 |
