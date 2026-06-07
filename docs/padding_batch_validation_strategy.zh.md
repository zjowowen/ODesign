# ODesign Padding Batch 分阶段验证方案

本文档用于把 ODesign 从单样本训练路径扩展到 padded batch 训练路径的验证工作拆成三个阶段，并明确每个阶段要证明什么、如何证明、哪些风险尚未覆盖。

核心目标是验证：

> 在真实 token/atom prefix 上，`batch_size_per_device=2` 的一次前后向，应等价于两个 `batch_size_per_device=1` 样本按 DDP/gradient accumulation 语义取平均后的结果。

这里的“等价”不能只看 optimizer step 后的参数是否接近。学习率、Adam 动量、梯度裁剪和较宽的 allclose 容差都可能把 loss/grad 层面的差异掩盖到参数更新里。因此验证顺序必须是：

1. 先比较前向张量、loss 分量和参数梯度。
2. 再比较 pre-clip/post-clip gradient。
3. 最后才比较 optimizer step 后的参数和 optimizer state。

## 术语

本文使用以下符号：

- `B`：padded batch size。
- `T_i`：第 `i` 个样本真实 token 数。
- `A_i`：第 `i` 个样本真实 atom 数。
- `M_i`：第 `i` 个样本真实 MSA row 数。
- `T`：batch 内 `max(T_i)`。
- `A`：batch 内 `max(A_i)`。
- `M`：batch 内 `max(M_i)`。
- `S`：diffusion training sample 数，即 `diffusion_batch_size` 或 `N_sample`。
- `c_s_inputs/c_s/c_z`：ODesign trunk channel 维度。
- “真实 prefix”：每个样本的 `[:T_i]` 或 `[:A_i]` 区域。
- “padding suffix”：每个样本 pad 到 `T` 或 `A` 后的尾部区域。

阶段一只要求真实 prefix 与单样本路径等价，padding suffix 不要求和单样本比较，因为单样本路径里不存在这些 suffix；但 suffix 必须被 mask，不得进入 attention、loss denominator、alignment 和 gradient。

## 总体阶段

### 源码和测试索引

读者做 code review 或复查覆盖率时，优先对照这些文件：

| 主题 | 源码位置 | 测试/文档位置 |
| --- | --- | --- |
| padded collate 和字段角色 | `src/utils/model/padded_collate.py` | `tests/test_padded_collate.py` |
| train dataloader 何时启用 padded collate | `src/data/dataloader.py` | 本文档阶段二 sampler/replay 清单 |
| 模型接口 dataclass | `src/api/data_interface.py`、`src/api/model_interface.py` | `tests/test_padding_batch_end_to_end.py` |
| Pairformer pair mask | `src/model/odesign.py` | `tests/test_padding_batch_end_to_end.py`、`tests/test_padding_batch_training_contract.py` |
| MSA row/helper 轴语义 | `src/utils/model/misc.py`、`src/model/modules/pairformer.py` | `tests/test_msa_token_mask.py` |
| diffusion training shape | `src/model/modules/generator.py` | `tests/test_padding_batch_training_contract.py`、`tests/test_padding_batch_end_to_end.py` |
| token/atom broadcast 与 aggregation | `src/utils/model/misc.py` | `tests/test_padding_batch_training_contract.py` |
| permutation batch 切片/恢复 | `src/utils/permutation/permutation.py` | `tests/test_padding_batch_training_contract.py` |
| loss、mask、reduction | `src/model/modules/loss.py` | `tests/test_padding_batch_training_contract.py` |
| 当前单元测试说明和验证证据 | N/A | `docs/padding_batch_unit_tests.zh.md` |

### 阶段一：单元测试级别对齐

目的：证明每一个被 batch 化的模块都遵守 shape、mask、reduction、gradient 的局部契约。

验收口径：

- `collate_fn_odesign_padded` 能把 ragged token/atom/MSA 样本 pad 成带 batch 轴的数据。
- 所有模型输入接口能接受 legacy unbatched shape 和 new batched shape。
- Pairformer、Diffusion、Permutation、Loss 路径不会把 batch 轴误当成 sample/token/atom 轴。
- padding mask 能阻断 padded token/atom 对 attention、aggregation、alignment、loss 的贡献。
- tiny ODesign 的 `batch=2` 前后向，和两个 `batch=1` 前后向结果平均后，在前向输出、loss、grad、multi-step 参数更新上 allclose。

阶段一的测试原则：

- 对每个底层 helper 写 contract test，直接验证 shape、mask、reduction 和 gradient 阻断。
- 对完整 tiny ODesign 写 e2e test，避免只证明 helper 正确但模块串联后仍错轴。
- 对训练等价性测试必须控制随机性；阶段一不试图证明真实随机训练序列已经等价。
- 比较真实 prefix，不比较 padding suffix 与单样本路径，因为单样本路径没有 suffix。
- loss/metrics、前向张量、参数梯度和 optimizer 后参数都要比较；只比较最终参数不够。

### 阶段二：控制数据集训练对齐

目的：在真实 ODesign 数据、真实 checkpoint、真实训练配置附近，控制所有会破坏比较的随机性，用 replay/integration test 验证 bsz2 和 bsz1 是否真正等价。

验收口径：

- 同 world size、同 rank、同 sample trace。
- 同一 rank 上 bsz2 的一个 optimizer update 与 bsz1 的多个 microbatch accumulation 看到完全相同的样本集合和样本顺序。
- 禁用或固定所有会影响 forward/loss/grad 的随机源。
- 比较 per-sample forward tensors、loss 分量、pre-clip grad、post-clip grad、optimizer step 后 state。

### 阶段三：整体训练与评测

目的：在阶段一和阶段二通过后，再启动完整训练，评估真实收敛、checkpoint 质量和 PBP/ODesignBench 指标。

本文档暂不展开第三阶段，只保留验收原则：

- 使用明确的 training contract 和 evaluation contract。
- 同步记录代码 commit、配置、数据路径、checkpoint 路径、评测脚本、评测结果。
- 正式训练不能用 smoke/replay 结果替代。

## 阶段一：模型数据流和 Shape 改造清单

### 1. Dataset Sample 到 Collate

单样本 dataset 输出结构：

```text
sample = {
  "feature_data": OFeatureData,
  "label_data": OLabelData,
  "label_full_data": OLabelData,
  "basic": dict,
  ...
}
```

改造前训练路径使用 `collate_fn_first`，实质上每个 step 只保留一个样本，绝大多数 tensor 没有 batch 轴。

改造后当 `configs.data.train_batch_size > 1` 时使用 `collate_fn_odesign_padded`，按字段角色 pad 后 `torch.stack(..., dim=0)`。

| 字段角色 | 改造前 shape | 改造后 shape | Padding 规则 | 必须验证 |
| --- | --- | --- | --- | --- |
| token field | `[T_i, ...]` | `[B, T, ...]` | 真实 token copy 到 `[:T_i]`，suffix 置零 | suffix 由 `token_padding_mask` 标记 |
| atom field | `[A_i, ...]` | `[B, A, ...]` | 真实 atom copy 到 `[:A_i]`，suffix 置零 | suffix 由 `atom_padding_mask` 标记 |
| token pair field | `[T_i, T_i, ...]` | `[B, T, T, ...]` | 左上真实 block copy | padding pair 不进 attention/loss |
| atom pair field | `[A_i, A_i, ...]` | `[B, A, A, ...]` | 左上真实 block copy | padding pair 不进 loss |
| MSA token field | `[M_i, T_i, ...]` | `[B, M, T, ...]` | row/token prefix copy | MSA token mask 保留 batch 轴 |
| atom set field | `[K_i, A_i, ...]` | `[B, K, A, ...]` | row/atom prefix copy | atom set 不错切 batch/atom 轴 |
| scalar/same-shape metadata | `[]` 或固定 shape | `[B, ...]` | 能 stack 则 stack | 不参与 padding 推断 |
| ragged metadata/list | list | list of per-sample values | 保留 list | 不误 stack |

新增 mask：

- `feature_data.token_padding_mask`: `[B, T]`，真实 token 为 `False`，padding token 为 `True`。
- `feature_data.atom_padding_mask`: `[B, A]`，真实 atom 为 `False`，padding atom 为 `True`。

已有单元测试：

- `tests/test_padded_collate.py`
  - `test_padded_collate_pads_feature_and_label_tensors`
  - `test_padded_collate_collates_basic_metadata_and_optional_fields`
  - `test_padded_collate_handles_one_dimensional_msa_token_mask`

当前覆盖判断：已覆盖主要字段角色、padding mask、metadata、optional fields。还没有用全量真实 dataset 样本枚举所有字段角色，这是阶段二的 replay/preflight 范畴。

### 2. Interface 拆分

`ODesign.forward(..., mode="train")` 会从 `feature_data/label_data` 拆出：

```text
PairFormerInput.from_feature_data(feature_data)
DiffusionInput.from_feature_data(feature_data)
PermutationInput.from_feature_data(feature_data)
LossInput.from_feature_data(feature_data)
GroundTruth.from_label_data(label_data)
```

| Interface | 改造前主要 shape | 改造后主要 shape | 风险点 | 已有测试 |
| --- | --- | --- | --- | --- |
| `PairFormerInput` | token `[T]`、atom `[A]`、MSA `[M,T]`、pair `[T,T]` | `[B,T]`、`[B,A]`、`[B,M,T]`、`[B,T,T]` | dataclass 自动转换不能丢 batch 轴 | end-to-end forward 覆盖 |
| `DiffusionInput` | atom `[A,...]`、token `[T,...]`、`atom_to_token_idx [A]` | `[B,A,...]`、`[B,T,...]`、`[B,A]` | atom/token 映射需保留 batch prefix | broadcast/aggregate contract tests |
| `PermutationInput` | atom fields `[A]`，`atom_perm_list` list | tensor fields `[B,A]`，collated permutation list | list 和 tensor 切片规则不同 | slice/crop/restore tests |
| `LossInput` | atom masks `[A]`、resolution scalar、rep atom mask `[A]` | atom masks `[B,A]`、resolution `[B]`、rep atom mask `[B,A]` | padding atom 不得参与 loss/alignment | loss contract tests |
| `GroundTruth` | coordinate `[A,3]`、mask `[A]`、bond/label `[A,A]` or `[T,T]` | coordinate `[B,A,3]`、mask `[B,A]`、pair `[B,A,A]` or `[B,T,T]` | `lddt_mask/distance_mask` 要 batch-aware | loss update_label/e2e tests |

阶段一要求：任何 interface class 不应该自己引入新的 reshape 语义，只做字段筛选和类型转换。shape 语义由 collate 和下游模块承担。

### 3. Pairformer Trunk

训练主路径：

```text
PairFormerInput
  -> InputFeatureEmbedder
  -> RelativePositionEncoding
  -> ConstraintTemplateEmbedder
  -> MSAModule
  -> PairformerStack
  -> PairFormerOutput(s_inputs, s, z)
```

改造前：

- `s_inputs`: `[T, c_s_inputs]`
- `s_init/s`: `[T, c_s]`
- `z_init/z`: `[T, T, c_z]`
- `token_bonds`: `[T, T]`
- `msa`: `[M, T, ...]`

改造后：

- `s_inputs`: `[B, T, c_s_inputs]`
- `s_init/s`: `[B, T, c_s]`
- `z_init/z`: `[B, T, T, c_z]`
- `token_bonds`: `[B, T, T]`
- `msa`: `[B, M, T, ...]`
- `pair_mask`: `[B, T, T]`，由 `~token_padding_mask[..., :, None] & ~token_padding_mask[..., None, :]` 生成。

关键改造点：

- `ODesign._make_pair_mask` 从 `token_padding_mask` 生成 pair-level valid mask。
- `ConstraintTemplateEmbedder`、`MSAModule`、`PairformerStack` 必须接收并传递 `pair_mask`。
- MSA helper 必须把 MSA row 轴识别为 `-3`，而不是误切 batch 轴。
- attention bias 必须屏蔽 padded query/key。

已有单元测试：

- `tests/test_padding_batch_end_to_end.py::test_pairformer_paths_receive_token_padding_pair_mask`
- `tests/test_msa_token_mask.py`
  - `_apply_msa_token_mask`
  - `_add_single_embedding_to_msa`
  - `_slice_msa_rows`
  - `_chunk_msa_rows`
  - `MSAStack.inference_forward`
- `tests/test_padding_batch_training_contract.py`
  - `test_attention_pair_bias_applies_standard_attention_mask`
  - `test_pairformer_block_passes_pair_mask_to_single_attention`

当前覆盖判断：batch/pair mask 传递、MSA batch prefix、attention mask 有覆盖。真实大模型中 DeepSpeed EVO attention、memory-efficient kernel、LMA 的所有组合没有在阶段一 tiny config 中完全枚举，属于阶段二/三运行环境验证。

### 4. Pairwise Head

输入：

- `PairFormerOutput.z`: 改造前 `[T, T, c_z]`，改造后 `[B, T, T, c_z]`。

输出：

| 输出 | 改造前 | 改造后 | 验证点 |
| --- | --- | --- | --- |
| `distogram` | `[T, T, n_bins]` | `[B, T, T, n_bins]` | 真实 token block 与单样本一致 |
| `token_bond_type_logits` | `[T, T, n_bond_types]` | `[B, T, T, n_bond_types]` | 真实 token block 与单样本一致 |
| `token_bond_gen_mask` | `[T, T]` | `[B, T, T]` | padding token pair 不进 bond loss |

已有单元测试：

- `tests/test_padding_batch_training_contract.py`
  - `test_bond_type_head_preserves_batch_prefix`
  - `test_bond_type_head_adds_legacy_batch_prefix_for_unbatched_input`
- `tests/test_padding_batch_end_to_end.py`
  - 单步和多步等价性测试比较真实 token block 上的 `distogram` 和 `token_bond_type_logits`。

当前覆盖判断：pairwise head 的 batch prefix 行为和 e2e 输出已覆盖。

### 5. Diffusion Training

训练函数：`sample_diffusion_training(...)`。

改造前：

- `ground_truth.coordinate`: `[A, 3]`
- `ground_truth.coordinate_mask`: `[A]`
- `x_gt_augment`: `[S, A, 3]`
- `sigma`: `[S]`
- `condition_mask`: `[S, A]`
- `x_noisy/x_denoised`: `[S, A, 3]`

改造后：

- `ground_truth.coordinate`: `[B, A, 3]`
- `ground_truth.coordinate_mask`: `[B, A]`
- `batch_prefix = ground_truth.coordinate.shape[:-2] = [B]`
- `x_gt_augment`: `[B, S, A, 3]`
- `sigma`: `[B, S]`
- `condition_mask`: `[B, S, A]`
- `x_noisy/x_denoised`: `[B, S, A, 3]`

关键改造点：

- `_diffusion_batch_prefix` 和 `_diffusion_sample_shape` 必须保留 batch prefix。
- `centre_random_augmentation` 的 mask 支持 `[B, A]`，输出 `[B, S, A, 3]`。
- `TrainingNoiseScheduler.sample_noise_level` 的 size 从 `[S]` 扩展为 `[B, S]`。
- `add_noise_with_condition` 和 `denoise_with_conditon` 需要按 `[B,S,A,3]` broadcast。
- diffusion chunking 只能切 sample axis `S`，不能切 batch axis。
- DiffusionModule 内部 token/atom broadcast、atom-to-token aggregation、attention mask 需支持 `[B,...]` 和 `[B,S,...]`。

已有单元测试：

- `tests/test_padding_batch_training_contract.py`
  - `test_diffusion_sample_shape_preserves_batch_prefix`
  - `test_centre_random_augmentation_accepts_batched_mask`
  - `test_broadcast_token_to_atom_preserves_equal_batch_prefix`
  - `test_broadcast_token_to_atom_expands_batched_index_over_sample_dim`
  - `test_aggregate_atom_to_token_expands_batched_index_over_sample_dim`
  - `test_aggregate_atom_to_token_ignores_masked_atoms_in_mean`
  - `test_gather_pair_embedding_in_dense_trunk_expands_batched_indices_over_sample_dim`
  - `test_diffusion_transformer_mask_blocks_padded_token_gradients`
  - `test_diffusion_transformer_passes_token_attention_mask_to_blocks`
- `tests/test_padding_batch_end_to_end.py`
  - tiny ODesign forward/backward smoke。
  - `batch=2` vs 两个 `batch=1` 平均等价。
  - multi-step optimizer 等价。

当前覆盖判断：Diffusion 的 batch prefix、sample axis、mask、gradient 阻断路径有较强覆盖。真实 production diffusion chunk size、真实大 atom 数、真实 CUDA kernel 的组合仍需要阶段二 replay。

### 6. Symmetric Permutation

训练路径：

```text
symmetric_permutation.permute_diffusion_sample_to_match_label(
  permutation_input, model_output, ground_truth, stage="train"
)
```

改造前：

- `model_output.coordinate`: `[S, A, 3]`
- `ground_truth.coordinate`: `[A, 3]`
- permutation 在单个样本上执行。

改造后：

- `model_output.coordinate`: `[B, S, A, 3]`
- `ground_truth.coordinate`: `[B, A, 3]`
- 先按 batch item 切片，再按真实 atom 长度裁剪 prefix。
- 单样本 permutation 只处理真实 prefix。
- 结果恢复到 padded output 的真实 prefix，padding suffix 保持原值。

关键改造点：

- `_is_batched_structure_tensor`
- `_slice_batch_item`
- `_crop_atom_prefix`
- `_restore_atom_prefix_output`
- `_stack_odesign_outputs`

已有单元测试：

- `tests/test_padding_batch_training_contract.py`
  - `test_is_batched_structure_tensor_identifies_batched_coordinate`
  - `test_slice_batch_item_slices_batch_sized_tensor_but_preserves_arbitrary_lists`
  - `test_slice_batch_item_selects_known_collated_permutation_lists`
  - `test_crop_atom_prefix_uses_real_atom_length_for_atom_axis_fields`
  - `test_restore_atom_prefix_output_updates_real_prefix_and_preserves_padding`
  - `test_stack_odesign_outputs_stacks_tensor_fields_and_preserves_none`

当前覆盖判断：batch 切片/裁剪/恢复基础契约已覆盖。真实 chain permutation heuristic 中存在 `random.choice` 分支，阶段一未要求覆盖；阶段二 replay 需要 seed 或禁用 permutation 来隔离随机性。

### 7. Loss 和 Metrics

`ODesignLoss` 流程：

```text
update_label(loss_input, ground_truth)
  -> distance_mask
  -> distance
  -> lddt_mask

calculate_prediction(pred_output)
  -> pred distance if dense loss path enabled

calculate_losses(...)
  -> smooth_lddt_loss
  -> bond_loss
  -> mse_loss
  -> distogram_loss
  -> bond_type_loss
  -> aggregate_losses
```

改造前：

- `ground_truth.coordinate`: `[A,3]`
- `ground_truth.coordinate_mask`: `[A]`
- `distance_mask/lddt_mask`: `[A,A]`
- `pred_output.coordinate`: `[S,A,3]`
- `pred_output.noise_level`: `[S]`
- diffusion loss per sample intermediate: `[S]`，最后对 `S` 求平均。

改造后：

- `ground_truth.coordinate`: `[B,A,3]`
- `ground_truth.coordinate_mask`: `[B,A]`
- `distance_mask/lddt_mask`: `[B,A,A]`
- `pred_output.coordinate`: `[B,S,A,3]`
- `pred_output.noise_level`: `[B,S]`
- diffusion loss per sample intermediate: `[B,S]`，先对 `S` 求平均，再按 batch prefix reduce。

关键风险：

- batch 轴不能被 flatten 到 pair index 中导致不同样本 pair 混在一起。
- sparse loss 的 `torch.nonzero(mask)` 必须逐 batch item 处理。
- `loss_reduction(mean)` 必须按样本 mean，而不是按所有有效 pair 数全局加权，除非训练契约明确要求全局 pair-weighted mean。
- padding atom 不得进入 `distance_mask`、`lddt_mask`、bond denominator、MSE alignment。
- `diffusion_per_sample_scale` 从 `[S]` 扩展为 `[B,S]` 后要正确 broadcast 到 loss。

已有单元测试：

- `tests/test_padding_batch_training_contract.py`
  - resolution gate/vectorization tests。
  - `_apply_last_dim_mask` batched ragged atom axis tests。
  - `BondTypeLoss` batched labels/masks tests。
  - sparse/dense `BondLoss` tests。
  - sparse/dense `SmoothLDDTLoss` batch prefix、sample axis、chunk、empty mask tests。
  - `DistogramLoss` batched ragged representative atom mask test。
  - `_diffusion_condition_align_mask` padding-aware test。
  - `MSELoss.weighted_rigid_align` align mask test。
- `tests/test_padding_batch_end_to_end.py`
  - 单步和多步比较 scalar loss、metrics、`distance_mask`、`lddt_mask`、grad、参数。

当前覆盖判断：loss 层是阶段一覆盖最强的部分之一。但阶段二真实 replay 已显示 `smooth_lddt_loss` 是最敏感项，因此真实大样本、真实 sparse LDDT 路径仍需要 integration 级别诊断。

### 8. Backward 和 Optimizer

阶段一的最终单元测试契约：

```text
batch=2 padded forward/loss/backward
==
batch=1 sample0 forward + batch=1 sample1 forward
loss = mean(loss0, loss1)
backward(loss)
```

需要比较：

- scalar loss。
- 每个上报 metric。
- `GroundTruth` 真实 prefix，包括 `distance_mask` 和 `lddt_mask`。
- `LossInput` 真实 prefix。
- `model_output` 真实 prefix，包括 coordinate、distogram、bond logits、noise level。
- 每个参数梯度。
- 多个 optimizer step 后的每个参数。

已有单元测试：

- `tests/test_padding_batch_end_to_end.py::test_padding_batch_matches_average_of_single_item_training_steps`
- `tests/test_padding_batch_end_to_end.py::test_padding_batch_matches_microbatch_accumulation_across_optimizer_steps`

这两个测试的随机性控制包括：

- 固定 Torch 和 CUDA seed。
- monkeypatch `centre_random_augmentation` 为确定性 centering、identity rotation、zero translation。
- monkeypatch training noise scheduler，让 sigma 固定，并使用确定性的 noisy coordinate transform。
- 使用同一个初始 state dict、同一个 `current_step` 和同一个 tiny config。

当前覆盖判断：阶段一的核心训练等价性已覆盖，并且 multi-step 测试覆盖 forward、loss、grad、参数更新轨迹。但这仍是“控制随机性后的单元测试等价”，不是“真实 dataloader/真实随机增强/真实 DDP 训练等价”。

## 阶段一：地毯式检查清单

| 检查项 | 必须成立的契约 | 当前覆盖 | 状态 | 后续动作 |
| --- | --- | --- | --- | --- |
| Collate 角色识别 | token/atom/pair/MSA/atom-set 字段按正确轴 pad | `test_padded_collate.py` | 已覆盖 | 阶段二用真实样本 spot-check 字段全集 |
| Padding mask | 真实 prefix 为 False，suffix 为 True | `test_padded_collate.py` | 已覆盖 | 保持 |
| Interface batch 轴 | dataclass 不丢失 `[B,...]` | e2e tests | 已覆盖 | 保持 |
| Pair mask | `token_padding_mask -> pair_mask [B,T,T]` | `test_pairformer_paths_receive_token_padding_pair_mask` | 已覆盖 | 保持 |
| MSA batch prefix | MSA row 轴为 `-3`，batch 轴不被切 | `test_msa_token_mask.py` | 已覆盖 | 阶段二控制 MSA 随机采样 |
| Token/atom broadcast | `[B,A]` index 可扩展到 `[B,S,A]` | training contract tests | 已覆盖 | 保持 |
| Atom-to-token mean | masked atoms 不进入 token mean denominator | `test_aggregate_atom_to_token_ignores_masked_atoms_in_mean` | 已覆盖 | 保持 |
| Diffusion sample axis | batch prefix 和 sample axis 同时保留 | diffusion shape tests + e2e | 已覆盖 | 阶段二用真实 `diffusion_chunk_size` 验证 |
| Diffusion attention mask | padded token 梯度为零 | `test_diffusion_transformer_mask_blocks_padded_token_gradients` | 已覆盖 | 保持 |
| Permutation batch 切片 | 每个 batch item 独立 crop/permute/restore | permutation contract tests | 已覆盖 | 阶段二评估真实 permutation 是否需禁用或 seed |
| LDDT sparse loss | 不混 batch pair index，按 batch prefix reduce | sparse LDDT tests | 已覆盖 | 阶段二重点诊断真实 smooth LDDT |
| Dense LDDT loss | batch/sample axis 保留，chunk 切 sample axis | dense LDDT tests | 已覆盖 | 大样本 dense 仅用于小规模诊断，避免 CPU/GPU 过重 |
| Bond loss | batched pair mask 和 scale 正确 | bond loss tests | 已覆盖 | 保持 |
| Distogram loss | batched rep atom mask 正确 | distogram test | 已覆盖 | 保持 |
| Bond type loss | batched labels/masks 校验 batch size | bond type loss tests | 已覆盖 | 保持 |
| MSE alignment | padding-aware align mask，不让 condition/pad outlier 拖偏 | align mask RED/GREEN + tests | 已覆盖 | 保持 |
| E2E forward/backward | batch=2 vs 两个 batch=1 平均，输出和 grad allclose | e2e single-step | 已覆盖 | 保持 |
| Multi-step optimizer | 多步 loss/metrics/grad/params allclose | e2e multi-step | 已覆盖 | 保持 |
| DDP allreduce | 多 rank 下梯度平均语义一致 | 阶段一未覆盖 | 未覆盖 | 阶段二 replay |
| 真实数据采样 | Dataset crop/featurizer 随机性被控制 | 阶段一未覆盖 | 未覆盖 | 阶段二 replay |
| 生产配置全部 kernel | H image 上真实 block/kernel/chunk 组合 | tiny config 部分覆盖 | 部分覆盖 | 阶段二/三 |

阶段一结论标准：

- 如果只讨论代码局部契约和 tiny ODesign 前后向，当前单元测试已经形成较完整覆盖。
- 如果讨论真实数据、真实 sampler、真实随机增强、DDP allreduce 和生产大样本，必须进入阶段二。
- 因此阶段一可以作为合并代码的必要条件，但不能单独作为启动正式长训的充分条件。

## 阶段二：控制数据集训练对齐

阶段二不是普通短训，而是 replay integration test。它必须把“两个训练方式是否等价”从“统计上可能相似”降级为“同一批样本、同一随机输入、同一模型状态下逐项比较”。

推荐对照：

```text
bsz2 path:
  micro_batch_size=2, iters_to_accumulate=K

bsz1 path:
  micro_batch_size=1, iters_to_accumulate=2K

两者每个 optimizer update 看到相同的 2K 个样本。
```

比较顺序：

1. sample trace 是否完全一致。
2. 每个样本的真实 prefix feature/label 是否一致。
3. 每个样本的 `noise_level` 是否一致。
4. 每个样本的 `pred_coordinate` 是否一致。
5. 每个样本的 `lddt_mask/coordinate_mask` 是否一致。
6. 每个 loss 分量是否一致，尤其 `smooth_lddt_loss`。
7. pre-clip grad tensor 是否一致。
8. post-clip grad tensor 是否一致。
9. optimizer step 后参数和 optimizer state 是否一致。

参数更新 allclose 只能作为最后的弱信号，不能替代前面 1-8 项。

阶段二 replay 产物至少要保留：

- `run_contract.json`：commit、selected file hashes、checkpoint path/hash、config hash、world size、rank 数、dtype/kernel/chunk 设置。
- `trace_rankXX.json`：每个 rank 每个 update/microbatch 的 dataset index、basic metadata、sample seed、fetch 是否成功。
- `forward_rankXX_stepYY.pt`：每个样本真实 prefix 上的 `noise_level`、`pred_coordinate`、`distogram`、`token_bond_type_logits`、`distance_mask`、`lddt_mask`。
- `loss_rankXX_stepYY.json`：总 loss、每个 loss 分量、每个 metric，尤其单独记录 `weighted_smooth_lddt_loss` 或对应 smooth LDDT 分量。
- `grad_rankXX_stepYY.pt`：pre-clip grad、post-clip grad 或至少参数名到 norm/max_abs/hash 的 summary；发现 mismatch 时保存完整 tensor 子集。
- `state_rankXX_stepYY.pt`：optimizer step 后参数/state hash；只有在前面 loss/grad 通过后，state allclose 才能作为最终确认。

## 阶段二：随机性排查清单

| 随机源 | 代码位置/现象 | 如何破坏控制变量 | 控制方法 | 必须记录 |
| --- | --- | --- | --- | --- |
| Weighted sampler | `WeightedSampler` / `DistributedWeightedSampler` 使用 `torch.multinomial(seed + epoch)` | bsz2/bsz1 看到不同样本序列 | 不使用 dataloader 迭代结果作对照；显式生成 per-rank sample trace | `trace_rankXX.json` |
| DDP rank 切分 | distributed sampler 按 rank 切 indices | world size 或 rank assignment 不同会导致样本不同 | bsz2/bsz1 replay 使用同一 world size、同一 rank、同一 trace | world size、rank、trace |
| DataLoader epoch | `set_epoch(counter)` 改变 sampler seed | update 对齐错位 | replay 固定 update_idx 和 trace，不依赖 DataLoader epoch | update_idx |
| bad sample retry | dataset `random_sample_if_failed` 会 `random.choice` 新 index | bsz2/bsz1 某一路径跳过坏样本后序列分叉 | replay 中关闭 `random_sample_if_failed`；记录 invalid samples | invalid trace |
| dataset crop method | `CropData.random_crop_method()` | 同一 data index 可能产出不同 crop | 每次 fetch 前设置 `random/np/torch/cuda` seed；必要时缓存 featurized sample | `sample_seed` |
| ligand-focused crop | `np.random.choice(ref_chain_indices)` | ref chain 不同导致 crop 不同 | 同上；或禁用 ligand focus 相关随机分支 | crop method/ref chain |
| mask generation | `src/utils/data/mask_generator.py` 多处 `np.random`/`random.choice` | condition atoms、masked entity/token/atom 不同 | fetch sample 前固定全局 seed；必要时固定 mask_method/mask_type | sample_seed、mask config |
| reference position augmentation | `random_transform(..., apply_augmentation=True)` | ref_pos 全局旋转/平移不同 | fetch sample 前固定 `np.random`；或关闭 `ref_pos_augment` | featurizer config |
| molecule shuffle | dataset `shuffle_mols` 使用 `np.random.shuffle` | token/atom 顺序不同 | fetch sample 前固定 seed；对 replay 建议关闭或记录顺序 | config、sample_seed |
| symmetry id shuffle | dataset `shuffle_sym_ids` 使用 `np.random.shuffle` | permutation features 不同 | 固定 seed；对 replay 建议关闭或记录 | config、sample_seed |
| MSA row sampling | `sample_indices` 使用 `torch.randint` 和 `torch.randperm` | bsz2 与两个 bsz1 的 MSA rows/row count 不同 | replay monkeypatch 为 deterministic/topk；同时 patch `misc_module` 和 `pairformer_module` 引用 | deterministic MSA flag |
| model cycle count | `np.random.RandomState(current_step).randint(...)` | current_step 不同会导致 `N_cycle` 不同 | bsz2/bsz1 使用相同 `current_step` | current_step |
| condition dropout | `random.random() < condition_embedding_drop_rate` | use_conditioning 分叉 | replay 设 `condition_embedding_drop_rate=0.0` 或固定 seed | config |
| SE(3) augmentation | `centre_random_augmentation` 使用 scipy `Rotation.random` 和 `torch.randn` translation | 坐标 frame 不同，forward/loss/grad 不可比 | replay 替换为 deterministic center/identity rotation/zero translation | hook version |
| diffusion sigma | `sample_noise_level` 使用 `torch.randn` | 每样本 noise level 不同，loss scale 不同 | replay 固定 sigma，例如 `1.0` | `noise_level` |
| diffusion Gaussian noise | `add_noise_with_condition` 使用 `torch.randn_like(x_gt)` | noisy coordinate 不同，pred coordinate 不同 | replay 替换为 deterministic/no-random noise transform | hook version |
| diffusion chunking | sample axis chunk 顺序改变 | 浮点顺序和 checkpoint recompute 可能不同 | 对照中固定 `diffusion_chunk_size`；必要时设 `1` 做压力/隔离 | chunk size |
| symmetric permutation | chain heuristic 里有 `random.choice`；atom permutation 可能按预测选择最优排列 | 等价输入若预测细微不同，permutation 可能离散跳变 | 先跑 identity/disable permutation 隔离；再固定 seed 跑真实 permutation | permutation flag/log |
| CUDA 非确定性 | fused kernels、attention kernels、bf16、checkpoint recompute | 即使输入相同也可能有小量数值差异 | 使用合理但严格容差；比较 loss/grad 时优先 fp32 诊断；记录 kernel config | dtype/kernel/chunk |
| AMP/skip_amp | autocast 和 skip_amp 配置不同 | dtype 路径不同 | bsz2/bsz1 使用同一 config 和同一 env | config hash |
| gradient clipping | clip 会把不同 grad 投影到相近 norm | 参数更新差异被掩盖 | 同时比较 pre-clip 和 post-clip grad | grad norm、grad tensors |
| optimizer/scheduler | Adam state、LR、scheduler step | 参数 allclose 可能掩盖 loss/grad 差异 | 先比较 forward/loss/grad，再比较 state | lr、optimizer state hash |
| checkpoint load | 不同 checkpoint 或 partial load | 初始参数不同 | 记录 checkpoint path/hash，比较 initial state hash | checkpoint hash |
| code/config drift | cluster runtime 和 GitHub branch 不一致 | 诊断不可复现 | 记录 selected file hashes、commit SHA、runner hash | code hashes |

排查优先级：

1. 先证明 sample trace 和 featurized sample 一致。这里不一致时，任何 loss/grad 差异都不能归因到模型 batchwise 改造。
2. 再证明 diffusion 随机输入一致，重点看 `noise_level`、`x_noisy` 或能重建 `x_noisy` 的 hook 产物。
3. 再证明 forward 预测一致，重点看 `pred_coordinate` 的真实 atom prefix。
4. 再证明 loss 中间态一致，当前最敏感项是 smooth LDDT；如果 `pred_coordinate` 一致但 LDDT loss 不一致，优先查 `lddt_mask`、sparse pair index、reduction denominator 和 chunk/recompute。
5. 最后比较 pre-clip grad、post-clip grad、optimizer state。梯度之前的任一层不一致，都不要用参数 allclose 判定等价。

## 阶段二：当前 Replay 诊断证据

截至 2026-06-07，阶段二 replay 的最新结论是：

> 真实样本 replay 中，默认 H200/TF32 数值路径会让 batch shape 不同的 `bsz2` 和 `bsz1` 运行在 `InputFeatureEmbedder.atom_attention_encoder` 内出现 `1e-5 ~ 1e-4` 级差异；关闭 TF32 后，该首个发散点降到 fp32 尾差量级。因此当前主要问题不是 padding 输入错位，而是严格等价 replay 对 CUDA/TF32 batch-shape 数值差异非常敏感。
>
> 在 no-TF32/fp32、禁用 permutation、固定 MSA rows、真实 `35999.pt` checkpoint 和真实样本 trace 下，最新 4GPU DDP replay 已经在 `ATOL/RTOL=5e-4`、`DIAGNOSTIC_ATOL/RTOL=1e-4`、3 个 optimizer update 上通过。这个结论支持“训练集成层面 bsz2 与 bsz1 等价”，但不声称 bitwise 或 `1e-6` 级 strict replay 等价。
>
> 审计指出的两个 r9 边界已经补证：r11 在 4GPU、1 update 下保存并比较 pre/post full-grad tensor 且 grad compare failure 为 0；r13 在 4GPU、`per_rank_samples=2`、1 update 下启用 forward hook probe，并在 forward-probe 专用 `ATOL/RTOL=5e-3` 下通过；r18 进一步在 4GPU、`per_rank_samples=10`、1 update、`FORWARD_PROBE_SAMPLE_POS=0`、no-evo/fp32/no-TF32 控制变量下通过 forward-probe replay；r20 在 2GPU、同控制变量、`FORWARD_PROBE_SAMPLE_POS=9` 下通过；r22 在 2GPU、1 update、`per_rank_samples=2` 下同时打开 `SAVE_GRAD_TENSORS=true` 和 `FORWARD_PROBE_SAMPLE_POS=0` 并通过。r13/r18/r20 是 forward-probe 证据，r11 是 full-grad tensor 证据，r22 是低成本组合补强；它们都不替代 r9/r10 的 3 update 主 replay。
>
> 8GPU 单节点 r8 replay artifact 已在 3 update、`1e-4` diagnostic 口径下通过；16GPU 跨节点 r7 已实际运行到 3 update，但在 `1e-4` diagnostic 口径下有 1 个 `pred_coordinate` sample diagnostic failure，`max_abs=1.17e-4`，record/state/grad sync 均通过。r21 使用同一 16GPU 跨节点配置、把 diagnostic 阈值放宽到 `2e-4` 后已通过 summary 和 rank-record 断言探针。因此当前已有 8GPU gate pass 和 16GPU `2e-4` diagnostic gate pass；16GPU `1e-4` strict replay 仍不是 pass。

已核验的证据链：

1. r2 forward hook：
   - run dir：
     `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign/.cluster_operator/replay_bsz2_bsz1_forward_hook35999_fix2_0606_r2`
   - 结果：replay 按预期失败。
   - `bsz1 loss = 9.140436112880707`，`bsz2 loss = 9.080204248428345`，差异约 `0.06023186445236206`。
   - 当时的全局首个共同 mismatch 出现在 `pairformer_output.s_inputs`。

2. r3 input embedder probe：
   - run dir：
     `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign/.cluster_operator/replay_bsz2_bsz1_forward_hook35999_inputembed_0606_r3`
   - 已确认一致的输入：
     `diffusion_input.x_noisy/ref_pos/ref_mask/atom_to_token_idx/is_condition_atom`，
     以及 `InputFeatureEmbedder` 原始输入 `restype/profile/deletion_mean/is_hotspot_residue/ref_pos/ref_mask/ref_element/ref_charge/ref_atom_name_chars/ref_space_uid/atom_to_token_idx/padding_mask`。
   - `input_atom_attention_encoder.c_skip` 一致。
   - 首个差异出现在无坐标 `AtomAttentionEncoder` 输出：
     `input_atom_attention_encoder.q_skip` 最大 sample 差异约 `6.3329935e-05`，
     `input_atom_attention_encoder.a_token` 最大 sample 差异约 `1.258254e-04`。

3. r4 no-TF32 probe：
   - run dir：
     `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign/.cluster_operator/replay_bsz2_bsz1_forward_hook35999_inputembed_notf32_pod_0606_r4`
   - 环境控制：
     `MODEL_DTYPE=fp32`，
     `USE_DEEPSPEED_EVO_ATTENTION=false`，
     `NVIDIA_TF32_OVERRIDE=0`，
     `DISABLE_TF32=true`，
     `DISABLE_PERMUTATION=true`，
     `DETERMINISTIC_MSA_ROWS=1`。
   - `bsz1_gacc10` 与 `bsz2_gacc5` 的 `update000_pos00` forward probe 已对齐：`sample_idx=0`，`real_atom_len=2027`，`real_token_len=430`，281 个 probe record 全部按 `(base_name, occurrence)` 对齐。
   - 关键差异：
     - `input_atom_attention_encoder.q_skip`: `max_abs = 2.384185791015625e-07`，没有超过 `1e-6` 的坏点。
     - `input_atom_attention_encoder.a_token`: `max_abs = 2.384185791015625e-07`，没有超过 `1e-6` 的坏点。
     - `input_embedder.s_inputs`: `max_abs = 2.384185791015625e-07`，没有超过 `1e-6` 的坏点。
     - `pairformer_output.z_trunk`: `max_abs = 1.6093254089355469e-06`，存在少量 `1e-6` 级尾差，但没有超过 `1e-5` 或 `5e-4` 的坏点。
     - `atom_attention_decoder.x_update`: 最大约 `2.3e-06`，没有超过 `1e-5` 或 `5e-4` 的坏点。
   - 最终 `summary.json`：
     - `status = fail`，但失败来自 `diagnostic_atol=1e-6/diagnostic_rtol=1e-6` 的细粒度诊断阈值。
     - 主 replay 阈值 `atol=5e-4/rtol=5e-4` 下，`record_compare.allclose = true`，`state_compare.allclose = true`，`state_failure_count = 0`，`record_failure_count = 0`。
     - `effective_loss` 差异约 `5.364418029785156e-07`。
     - `grad_summary` 在主阈值下 allclose，最坏 summary 差异为 `grad_abs_sum` 约 `5.735119339078665e-04`，相对差异约 `2.592178175664554e-08`。
     - `state_compare` 在主阈值下 allclose，最坏 tensor 为 `diffusion_module.linear_no_bias_s.weight`，`max_abs = 1.819126191549003e-05`。
     - sample tensor compare 的剩余失败都集中在 `pred_coordinate` 的 `1e-6` 级诊断阈值上；已打印样本中的最大 `pred_coordinate max_abs` 为 `7.62939453125e-06`，`mean_abs` 约 `3e-7 ~ 4e-7`。

4. r5 no-TF32 full-grad probe：
   - run dir：
     `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign/.cluster_operator/replay_bsz2_bsz1_fullgrad_notf32_pod_0606_r5`
   - 环境控制：
     `MODEL_DTYPE=fp32`，
     `USE_DEEPSPEED_EVO_ATTENTION=false`，
     `NVIDIA_TF32_OVERRIDE=0`，
     `DISABLE_TF32=true`，
     `DISABLE_PERMUTATION=true`，
     `DETERMINISTIC_MSA_ROWS=1`，
     `SAVE_GRAD_TENSORS=true`。
   - 阈值：
     主 replay `atol=5e-4/rtol=5e-4`，
     full-grad diagnostic `diagnostic_atol=1e-5/diagnostic_rtol=1e-5`。
   - 产物：
     run dir 总大小约 `4.7G`；
     `bsz2_gacc5/pre_clip_grad_after_update_0.pt` 和
     `bsz2_gacc5/post_clip_grad_after_update_0.pt` 已保存为 reference。
   - 最终 `summary.json`：
     - `status = fail`，失败来自 `pre_clip_grad_compare` 的 `1e-5` 级诊断阈值。
     - `record_compare.allclose = true`，`state_compare.allclose = true`，`sample_compare_failure_count = 0`，`forward_probe_failure_count = 0`。
     - `bsz1 effective_loss = 9.080523490905762`，`bsz2 effective_loss = 9.080522775650024`。
     - `pre_clip_grad_compare.allclose = false`，最坏 tensor 为 `diffusion_module.atom_attention_encoder.linear_no_bias_q.weight`，`max_abs = 1.4086253941059113e-05`。
     - `post_clip_grad_compare.allclose = true`，同一最坏 tensor 的 `max_abs = 2.2284220904111862e-06`。
     - `state_compare.allclose = true`，最坏 tensor 为 `diffusion_module.linear_no_bias_s.weight`，`max_abs = 2.343812957406044e-05`。

5. r6 no-TF32 4GPU DDP replay：
   - run dir：
     `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign/.cluster_operator/replay_bsz2_bsz1_4gpu_notf32_pod_0606_r6`
   - 环境控制：
     `world_size=4`，
     `MODEL_DTYPE=fp32`，
     `USE_DEEPSPEED_EVO_ATTENTION=false`，
     `NVIDIA_TF32_OVERRIDE=0`，
     `DISABLE_TF32=true`，
     `DISABLE_PERMUTATION=true`，
     `DETERMINISTIC_MSA_ROWS=1`。
   - 阈值：
     主 replay `atol=5e-4/rtol=5e-4`，
     forward/sample diagnostic `diagnostic_atol=2e-5/diagnostic_rtol=2e-5`。
   - 最终 `summary.json`：
     - `status = fail`，失败来自 4 个 rank 的 diagnostic compare；`record_failure_count = 0`，`state_failure_count = 0`，`state_sync_failure_count = 0`。
     - 4 个 rank 的 `record_compare.allclose = true`，包括 `effective_loss`、`metrics_scaled_sum`、`grad_summary` 和 lr。
     - 4 个 rank 的 `state_compare.allclose = true`；rank0 最坏 state tensor 为 `module.diffusion_module.atom_attention_encoder.linear_no_bias_q.weight`，`max_abs = 1.3172626495361328e-05`。
     - 4 个 rank 的 `pre_clip_grad_hashes_synced = true`，`post_clip_grad_hashes_synced = true`，`state_hashes_synced = true`，说明 DDP rank 间同步状态一致。
     - diagnostic failure 集中在 forward/sample tensors：`pairformer_output.z_trunk`、`diffusion_transformer.output_a`、`atom_attention_decoder.x_update/diffusion_module.x_update` 和最终 `pred_coordinate`。已观察到的 `pred_coordinate max_abs` 最大约 `9.5367431640625e-05`。

6. r9 no-TF32 4GPU DDP replay，diagnostic 1e-4：
   - run dir：
     `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign/.cluster_operator/replay_bsz2_bsz1_4gpu_notf32_pod_0606_r9_diag1e4`
   - 详细实验日志：
     `docs/padding_batch_replay_r9_experiment_log.zh.md`
   - 运行载体：
     长时 4GPU 交互容器 `zjow-sci2-bs-pbp-4gpu-clone20260606162737-59667537-2g4sk`。
   - 环境控制：
     `SMOKE_MODE=1`，
     `world_size=4`，
     `MODEL_DTYPE=fp32`，
     `NVIDIA_TF32_OVERRIDE=0`，
     `DISABLE_TF32=true`，
     `DISABLE_PERMUTATION=true`，
     `DETERMINISTIC_MSA_ROWS=1`，
     `DENSE_LDDT_MAX_ATOMS=0`，
     `SAVE_GRAD_TENSORS=false`。
   - replay 参数：
     `UPDATES=3`，
     `TRACE_SEED=20260605`，
     主 replay `ATOL=5e-4/RTOL=5e-4`，
     forward/sample/grad diagnostic `DIAGNOSTIC_ATOL=1e-4/DIAGNOSTIC_RTOL=1e-4`。
   - 最终结果：
     `summary.json` 已生成，launcher return code 为 `0`。
   - 最终 `summary.json`：
     - `status = pass`。
     - `world_size = 4`，`updates = 3`。
     - `failure_count = 0`。
     - `record_failure_count = 0`，`state_failure_count = 0`，`state_sync_failure_count = 0`，`diagnostic_failure_count = 0`。
     - `bsz1_gacc10` 的 4 个 rank 均有 3 条 records；每个 rank 的最后一条 record（`update_idx = 2`）都满足 `record_compare.allclose = true`，`diagnostic_failure_count = 0`，`sample_compare_failure_count = 0`，`grad_compare_failure_count = 0`，`state_hashes_synced = true`。
     - `FORWARD_PROBE_SAMPLE_POS=-1`，因此 r9 未启用 forward hook 细粒度 probe；`forward_probe_failure_count = 0` 只能说明没有 forward probe failure，不能作为 forward probe 通过证据。
     - rank0 的 `state_compare.allclose = true` 覆盖 3 个 update；非 rank0 不重复保存完整 state compare，但 all-rank state hash 同步。
   - 结论：
     这是当前阶段二最强的单节点 4GPU DDP replay 证据。它把 r6 中 `2e-5` diagnostic 阈值下的 sample tensor 尾差，收敛为 `1e-4` diagnostic 阈值下的完整 3 update 通过；主训练 record、state、grad summary、sample tensor 和 DDP 同步 hash 均未出现 failure。r9 未启用 forward probe，若要补强中间层 forward tensor 证据，需要另跑 forward-probe repeat。

7. r10 exact repeat：
   - run dir：
     `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign/.cluster_operator/replay_bsz2_bsz1_4gpu_notf32_pod_0606_r10_repeat`
   - 运行配置：
     与 r9 相同的 `world_size=4`、`UPDATES=3`、`TRACE_SEED=20260605`、主阈值 `5e-4`、diagnostic 阈值 `1e-4`，仍为 `SAVE_GRAD_TENSORS=false`、`FORWARD_PROBE_SAMPLE_POS=-1`。
   - 最终结果：
     `returncode_node0.txt = 0`，`summary.status = pass`，`failure_count = 0`，`record_failure_count = 0`，`state_failure_count = 0`，`state_sync_failure_count = 0`，`diagnostic_failure_count = 0`。
   - 结论：
     r10 复现 r9 结论，证明 r9 不是单次偶发；但它不补 full-grad tensor 和 forward hook 边界。

8. r11 full-grad + strict forward probe：
   - run dir：
     `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign/.cluster_operator/replay_bsz2_bsz1_4gpu_notf32_pod_0606_r11_fullgrad_forward1`
   - 运行配置：
     `world_size=4`，`UPDATES=1`，`SAVE_GRAD_TENSORS=true`，`FORWARD_PROBE_SAMPLE_POS=0`，主阈值 `5e-4`，diagnostic 阈值 `1e-4`。
   - 最终结果：
     `summary.status = fail`，但 `record_failure_count = 0`，`state_failure_count = 0`，`state_sync_failure_count = 0`，`grad_compare_failure_sum = 0`。
   - 失败来源：
     `forward_probe_failure_sum = 4`，首个 mismatch 为 `diffusion_transformer.output_a#00`，sampled value 最大绝对差约 `0.0028`，stats compare 仍为 true。
   - 结论：
     r11 关闭 full-grad tensor 全量保存比较边界；不能作为 forward hook probe 通过证据。

9. r13 forward-probe lightweight replay：
   - run dir：
     `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign/.cluster_operator/replay_bsz2_bsz1_4gpu_notf32_pod_0606_r13_forward2samp5e3`
   - 运行配置：
     `world_size=4`，`UPDATES=1`，`PER_RANK_SAMPLES=2`，`SAVE_GRAD_TENSORS=false`，`FORWARD_PROBE_SAMPLE_POS=0`，主阈值 `5e-4`，sample/grad diagnostic 阈值 `1e-4`，forward probe 专用阈值 `5e-3`。
   - 最终结果：
     `returncode_node0.txt = 0`，`summary.status = pass`，`failure_count = 0`，`diagnostic_failure_count = 0`，`record_failure_count = 0`，`state_failure_count = 0`，`state_sync_failure_count = 0`。
   - 深查结果：
     `bsz1_gacc10` rank0..3 均有 `sample_compare_failure_count = 0`，`forward_probe_failure_count = 0`，`grad_compare_failure_count = 0`，`record_compare.allclose = true`；rank0 `state_compare.allclose = true`。
   - 结论：
     r13 先关闭 forward hook 中间层 probe 最小边界；但该结论限定为 4GPU lightweight replay，不能替代 r9/r10 的 3 update、10 samples/rank replay，也不能替代 r11 的 full-grad tensor 证据。

10. r14 default-kernel forward-probe 诊断失败：
   - run dir：
     `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign/.cluster_operator/replay_bsz2_bsz1_4gpu_notf32_pod_0606_r14_fwd10samp5e3`
   - 运行配置：
     `world_size=4`，`UPDATES=1`，`PER_RANK_SAMPLES=10`，`SAVE_GRAD_TENSORS=false`，`FORWARD_PROBE_SAMPLE_POS=0`，主阈值 `5e-4`，sample/grad diagnostic 阈值 `1e-4`，forward probe 专用阈值 `5e-3`。`preflight_env.txt` 没有 `MODEL_DTYPE=fp32`，也没有 `USE_DEEPSPEED_EVO_ATTENTION=false`。
   - 最终结果：
     `returncode_node0.txt = 1`，`summary.status = fail`，`failure_count = 8`，`record_failure_count = 4`，`diagnostic_failure_count = 4`，`state_failure_count = 0`，`state_sync_failure_count = 0`。
   - 深查结果：
     `bsz1_gacc10` rank0..3 均有 `record_compare.allclose = false`，每个 rank 的 `forward_probe_failure_count = 1`；`grad_compare_failure_count = 0`，rank0 `state_compare.allclose = true`，all-rank grad/state hash sync 均为 true。
   - 结论：
     r14 是控制变量不匹配的诊断失败，不作为 no-evo/fp32/no-TF32 口径的反证。它说明 `per_rank_samples=10、FORWARD_PROBE_SAMPLE_POS=0` 的 forward probe 对 dtype/kernel 路径敏感，需要用 r18 这类同控制变量 repeat 补证。

11. r15/r16/r17 中止记录：
   - r15 `replay_bsz2_bsz1_4gpu_notf32_pod_0606_r15_repeat_fwd10samp5e3`、r16 `replay_bsz2_bsz1_2gpu_noevo_pod_0606_r16_fwd10samp5e3`、r17 `replay_bsz2_bsz1_4gpu_noevo_fp32_pod_0606_r17_fwd10samp5e3` 均为人为中止或污染清理相关运行，`returncode=143` 且没有 `summary.json`。
   - 结论：
     这些 run 不作为通过或失败证据；r16 还缺少 `MODEL_DTYPE=fp32` 控制变量。

12. r18 4GPU no-evo/fp32 per-rank-samples=10 forward-probe replay：
   - run dir：
     `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign/.cluster_operator/replay_bsz2_bsz1_4gpu_noevo_fp32_pod_0606_r18_fwd10samp5e3_clean`
   - 运行配置：
     `world_size=4`，`UPDATES=1`，`PER_RANK_SAMPLES=10`，`MODEL_DTYPE=fp32`，`USE_DEEPSPEED_EVO_ATTENTION=false`，`DISABLE_TF32=true`，`NVIDIA_TF32_OVERRIDE=0`，`SAVE_GRAD_TENSORS=false`，`FORWARD_PROBE_SAMPLE_POS=0`，主阈值 `5e-4`，sample/grad diagnostic 阈值 `1e-4`，forward probe 专用阈值 `5e-3`。
   - 最终结果：
     `returncode_node0.txt = 0`，`summary.status = pass`，`failure_count = 0`，`diagnostic_failure_count = 0`，`record_failure_count = 0`，`state_failure_count = 0`，`state_sync_failure_count = 0`。
   - 深查结果：
     `bsz1_gacc10` rank0..3 均有 `sample_compare_failure_count = 0`，`forward_probe_failure_count = 0`，`grad_compare_failure_count = 0`，`record_compare.allclose = true`；rank0 `state_compare.allclose = true`；所有 rank 的 pre/post grad hash 和 state hash sync 均为 true。
   - 结论：
     r18 将 forward-probe 补证从 r13 的 `per_rank_samples=2` 扩展到 4GPU、`per_rank_samples=10`、1 update、`FORWARD_PROBE_SAMPLE_POS=0`；但它不表示 10 个 sample position 都做了 forward hook，也不替代 r9/r10 的 3 update 主 replay 或 r11 的 full-grad tensor 证据。

13. r19 2GPU no-evo/fp32 repeat：
   - run dir：
     `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign/.cluster_operator/replay_bsz2_bsz1_2gpu_noevo_fp32_pod_0606_r19_fwd10samp5e3`
   - 运行配置：
     `world_size=2`，`UPDATES=1`，`PER_RANK_SAMPLES=10`，`MODEL_DTYPE=fp32`，`USE_DEEPSPEED_EVO_ATTENTION=false`，`DISABLE_TF32=true`，`FORWARD_PROBE_SAMPLE_POS=0`，forward probe 专用阈值 `5e-3`。
   - 最终结果：
     `returncode_node0.txt = 0`，`summary.status = pass`，`failure_count = 0`，`record/state/state_sync/diagnostic_failure_count` 全部为 0。
   - 结论：
     r19 是 r18 同控制变量下的 2GPU repeat，包括 `NVIDIA_TF32_OVERRIDE=0`、`SAVE_GRAD_TENSORS=false`、`FORWARD_PROBE_SAMPLE_POS=0` 和相同阈值，仅 `world_size=2`；它支持 r18 的可靠性，但不能替代 4GPU/8GPU/16GPU world-size gate，也不提供 full-grad tensor 证据。

14. r20 2GPU no-evo/fp32 pos9 forward-probe repeat：
   - run dir：
     `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign/.cluster_operator/replay_bsz2_bsz1_2gpu_noevo_fp32_pod_0607_r20_fwdpos9`
   - 运行配置：
     `world_size=2`，`UPDATES=1`，`PER_RANK_SAMPLES=10`，`MODEL_DTYPE=fp32`，`USE_DEEPSPEED_EVO_ATTENTION=false`，`DISABLE_TF32=true`，`NVIDIA_TF32_OVERRIDE=0`，`SAVE_GRAD_TENSORS=false`，`FORWARD_PROBE_SAMPLE_POS=9`，主阈值 `5e-4`，sample/grad diagnostic 阈值 `1e-4`，forward probe 专用阈值 `5e-3`。
   - 最终结果：
     `returncode_node0.txt = 0`，`summary.status = pass`，`failure_count = 0`，`diagnostic_failure_count = 0`，`record_failure_count = 0`，`state_failure_count = 0`，`state_sync_failure_count = 0`。
   - 深查结果：
     `bsz1_gacc10` rank0..1 均有 `sample_compare_failure_count = 0`，`forward_probe_failure_count = 0`，`grad_compare_failure_count = 0`，`record_compare.allclose = true`；rank0 `state_compare.allclose = true`；rank0..1 的 pre/post grad hash 和 state hash sync 均为 true。
   - 结论：
     r20 是针对 `FORWARD_PROBE_SAMPLE_POS=0` 覆盖边界的定向补强，说明 2GPU、`per_rank_samples=10`、末尾 sample position `9` 也通过；但它不能替代 4GPU/8GPU/16GPU world-size gate，也不提供 full-grad tensor 证据。

15. r22 2GPU full-grad + forward-probe 组合补强：
   - run dir：
     `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign/.cluster_operator/replay_bsz2_bsz1_2gpu_noevo_fp32_pod_0607_r22_fullgrad_fwd2samp5e3`
   - 运行配置：
     `world_size=2`，`UPDATES=1`，`PER_RANK_SAMPLES=2`，`MODEL_DTYPE=fp32`，`USE_DEEPSPEED_EVO_ATTENTION=false`，`DISABLE_TF32=true`，`NVIDIA_TF32_OVERRIDE=0`，`SAVE_GRAD_TENSORS=true`，`FORWARD_PROBE_SAMPLE_POS=0`，主阈值 `5e-4`，sample/grad diagnostic 阈值 `1e-4`，forward probe 专用阈值 `5e-3`。
   - 最终结果：
     `returncode_node0.txt = 0`，`summary.status = pass`，`failure_count = 0`，`diagnostic_failure_count = 0`，`record_failure_count = 0`，`state_failure_count = 0`，`state_sync_failure_count = 0`。
   - 深查结果：
     `bsz2_gacc5/pre_clip_grad_after_update_0.pt` 和 `bsz2_gacc5/post_clip_grad_after_update_0.pt` 已保存为 full-grad reference；`bsz1_gacc10/rank00_records.json` 的 `record_compare.allclose = true`、`state_compare.allclose = true`、`pre_clip_grad_compare.allclose = true`、`post_clip_grad_compare.allclose = true`。`bsz1_gacc10` rank0..1 的 `sample_compare_failure_count = 0`、`forward_probe_failure_count = 0`、`grad_compare_failure_count = 0`，pre/post grad hash 和 state hash 均同步。
   - 结论：
     r22 是对 r11 full-grad 和 r13/r18 forward-probe 证据的低成本组合补强，说明二者可以在同一个 2GPU、1 update、`per_rank_samples=2` replay 中同时通过；但它不替代 r9/r10 的 4GPU 3 update 主 replay，也不覆盖 `per_rank_samples=10` 的 full-grad 组合或 8GPU/16GPU world-size gate。

16. r7 no-TF32 16GPU DDP replay：
   - H200 job：
     `zjow-odesign-replay16g-notf32-0606-r7`
   - 提交时间：
     `2026-06-06 17:04 +0800` 左右。
   - 提交入口：
     H200 MCP `submit_experiment`，namespace `ailab-ai4sdata`，charged group `ai4sdata_gpu`。
   - 规模：
     `replicas=2`，每个 replica `gpu=8`、`cpu=160`、`memory=1600512Mi`，等价于正式 `2 node x 8 GPU` world size。
   - runtime gate：
     live 0GPU preflight `zjow-odesign-replay16g-pref-0606-r1` 已 `Succeeded`，用于验证容器内可见共享盘、runner、replay Python、数据根、ckpt 根、`35999.pt` checkpoint 和 `odesign` conda 环境。
   - dry-run gate：
     H200 MCP dry-run manifest 已确认 `mount=gpfs://gpfs1/ai4sreason:/mnt/shared-storage-user/ai4sreason`、`DISTRIBUTED_JOB=true`、`gang-start=true`、`enable-sshd=yes`、`hostNetwork=true`、priority `5`、`restartPolicy=Never` 均按预期渲染。
   - 关键环境控制：
     `MODEL_DTYPE=fp32`，
     `USE_DEEPSPEED_EVO_ATTENTION=false`，
     `NVIDIA_TF32_OVERRIDE=0`，
     `DISABLE_TF32=true`，
     `DISABLE_PERMUTATION=true`，
     `DETERMINISTIC_MSA_ROWS=1`，
     `DENSE_LDDT_MAX_ATOMS=0`，
     `SAVE_GRAD_TENSORS=false`。
   - replay 参数：
     `UPDATES=3`，
     `TRACE_SEED=20260605`，
     主阈值 `ATOL=5e-4/RTOL=5e-4`，
     diagnostic 阈值 `DIAGNOSTIC_ATOL=1e-4/DIAGNOSTIC_RTOL=1e-4`。
   - 输出目录：
     `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign/.cluster_operator/replay_bsz2_bsz1_16gpu_notf32_h200_0606_r7`
   - 最终结果：
     共享盘 artifact 已写出 `outputs/summary.json`，`summary.status = fail`，`world_size = 16`，`updates = 3`，`failure_count = 1`，`diagnostic_failure_count = 1`，`record_failure_count = 0`，`state_failure_count = 0`，`state_sync_failure_count = 0`。
   - 深查结果：
     `bsz2_gacc5` 16 个 rank 共 48 条 records，sample/forward/grad/diagnostic failure 均为 0，record/state/sync 均未失败。`bsz1_gacc10` 16 个 rank 共 48 条 records，其中只有 `rank04_records.json` 的 `update_idx=2` 出现 1 个 sample diagnostic failure；`record_compare.allclose = true`，grad summary allclose，pre/post grad hash 和 state hash sync 均为 true。
   - 失败来源：
     唯一失败样本为 `rank=4`、`update_idx=2`、`sample_pos=1`、`data_index=1205544`、`real_atom_len=1184`，`pred_coordinate` 的 `max_abs = 1.1730194091796875e-04`，略高于 `DIAGNOSTIC_ATOL=1e-4`；`noise_level`、`coordinate_mask`、`lddt_mask` 等控制张量均 allclose。
   - 结论：
     r7 覆盖了正式 `2 node x 8 GPU` world size 和跨节点通信，但没有在 `1e-4` sample diagnostic 口径下完全通过。由于主 record/state/grad sync 全部通过，且唯一差异为 `pred_coordinate` 的 `1.17e-4` 量级尾差，当前判断更接近 16GPU 跨节点 shape-sensitive fp32 数值尾差，而不是 padding/mask/trace 错位。它不能作为 16GPU strict replay pass 证据，但可以作为“16GPU 主训练 record/state/grad sync 无异常”的诊断证据。

17. r21 no-TF32 16GPU DDP replay，diagnostic 2e-4：
   - H200 job：
     `zjow-odesign-replay16g-notf32-0607-r21`
   - 提交入口：
     H200 MCP `submit_experiment`，namespace `ailab-ai4sdata`，charged group `ai4sdata_gpu`。
   - 规模：
     `replicas=2`，每个 replica `gpu=8`、`cpu=160`、`memory_gb=1563`，等价于正式 `2 node x 8 GPU` world size。
   - 关键环境控制：
     `MODEL_DTYPE=fp32`，
     `USE_DEEPSPEED_EVO_ATTENTION=false`，
     `NVIDIA_TF32_OVERRIDE=0`，
     `DISABLE_TF32=true`，
     `DISABLE_PERMUTATION=true`，
     `DETERMINISTIC_MSA_ROWS=1`，
     `DENSE_LDDT_MAX_ATOMS=0`，
     `SAVE_GRAD_TENSORS=false`。
   - replay 参数：
     `UPDATES=3`，
     `PER_RANK_SAMPLES=10`，
     `TRACE_SEED=20260605`，
     主阈值 `ATOL=5e-4/RTOL=5e-4`，
     diagnostic 阈值 `DIAGNOSTIC_ATOL=2e-4/DIAGNOSTIC_RTOL=2e-4`。
   - 输出目录：
     `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign/.cluster_operator/replay_bsz2_bsz1_16gpu_notf32_h200_0607_r21_diag2e4`
   - 外层作业结果：
     H200 MCP `get_job` 显示 `status = Succeeded`，两个 replica `state = SUCCEED`。
   - 断言探针：
     因本地机器和已回收的长时容器无法直接读取共享盘，补交了两个 0GPU 只读断言探针读取 r21 artifact。
     `zjow-odesign-r21-summary-assert-0607-r1` 已 `Succeeded`，证明 `outputs/summary.json` 存在且满足 `status=pass`、`world_size=16`、`updates=3`、`failure_count=0`、`diagnostic_failure_count=0`、`record_failure_count=0`、`state_failure_count=0`、`state_sync_failure_count=0`、`diagnostic_atol/rtol=2e-4/2e-4`。
     `zjow-odesign-r21-records-assert-0607-r1` 已 `Succeeded`，证明 `bsz2_gacc5` 和 `bsz1_gacc10` 的 16 个 rank records 均存在，每 rank 3 条 records、`update_idx=[0,1,2]`；`bsz1_gacc10` records 的 `record_compare.allclose=true`，rank0 的 `state_compare.allclose=true`，sample/forward/grad/diagnostic failure count 均为 0，pre/post grad hash 和 state hash 均同步。
   - 结论：
     r21 是正式 `2 node x 8 GPU` world size 下的 3 update replay pass 证据，但通过口径是 `DIAGNOSTIC_ATOL/RTOL=2e-4`。它支持 r7 的判断：r7 的 `1.17e-4` 单点失败更像跨节点 fp32 数值尾差，而不是 padding/mask/trace bug。r21 不能反向宣称 16GPU 在 `1e-4` diagnostic 口径下也已经通过；`1e-4` strict replay 的失败记录仍以 r7 为准。

18. r8 no-TF32 8GPU 单节点 engineering replay：
   - H200 job：
     `zjow-odesign-replay8g-notf32-0606-r1`
   - 提交时间：
     `2026-06-06 17:13 +0800` 左右。
   - 提交入口：
     H200 MCP `submit_experiment`，namespace `ailab-ai4sdata`，charged group `ai4sdata_gpu`。
   - 规模：
     `replicas=1`，单 replica `gpu=8`、`cpu=160`、`memory=1600512Mi`。
   - 结论边界：
     该任务使用 `SMOKE_MODE=1`，只验证单节点 8 rank DDP 和正式每节点 GPU 数；它不覆盖跨节点通信，也不能替代 r7 的正式 `2 node x 8 GPU` gate。
   - dry-run gate：
     H200 MCP dry-run manifest 已确认 `mount=gpfs://gpfs1/ai4sreason:/mnt/shared-storage-user/ai4sreason`、`DISTRIBUTED_JOB=true`、`enable-sshd=yes`、`hostNetwork=false`、priority `5`、`restartPolicy=Never` 均按预期渲染。
   - 关键环境控制：
     `MODEL_DTYPE=fp32`，
     `USE_DEEPSPEED_EVO_ATTENTION=false`，
     `NVIDIA_TF32_OVERRIDE=0`，
     `DISABLE_TF32=true`，
     `DISABLE_PERMUTATION=true`，
     `DETERMINISTIC_MSA_ROWS=1`，
     `DENSE_LDDT_MAX_ATOMS=0`，
     `SAVE_GRAD_TENSORS=false`。
   - replay 参数：
     `UPDATES=3`，
     `TRACE_SEED=20260605`，
     主阈值 `ATOL=5e-4/RTOL=5e-4`，
     diagnostic 阈值 `DIAGNOSTIC_ATOL=1e-4/DIAGNOSTIC_RTOL=1e-4`。
   - 输出目录：
     `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign/.cluster_operator/replay_bsz2_bsz1_8gpu_notf32_h200_0606_r1`
   - 最终结果：
     共享盘 artifact 已写出 `outputs/summary.json`，`summary.status = pass`，`world_size = 8`，`updates = 3`，`failure_count = 0`，`diagnostic_failure_count = 0`，`record_failure_count = 0`，`state_failure_count = 0`，`state_sync_failure_count = 0`。
   - 深查结果：
     `bsz2_gacc5` 和 `bsz1_gacc10` 各 8 个 rank、共各 24 条 records；sample/forward/grad/diagnostic failure 均为 0，record/state/sync 均未失败。
   - 注意：
     H200 MCP 聚合状态显示该 job 为 `Failed`，但共享盘 replay artifact 的 `outputs/summary.json` 为 `pass`，stdout tail 也显示 replay summary pass。当前以 replay artifact 作为模型等价性证据，以 H200 job 聚合状态作为外层作业状态异常另行记录。
   - 结论：
     r8 通过单节点 8GPU、3 update replay，是当前正式每节点 GPU 数的工程 gate 通过证据；它仍不覆盖跨节点通信，不能替代 r7 的 16GPU gate。

当前判断：

- r3 已排除“输入 feature/label、diffusion noise、MSA rows、padding mask 错位”作为首个发散原因。
- r4 说明默认 replay 失败中的大差异主要来自 TF32/batch-shape 触发的不同 kernel 或累加路径。
- r5 说明在 no-TF32/fp32 下，完整 pre-clip grad 的剩余最坏差异约为 `1.4e-5`，post-clip grad、state、record、sample tensor 和 forward probe 在当前口径下通过。
- r6 说明 4GPU DDP 下每 rank 的 record/state/grad summary 和 DDP 同步 hash 在主阈值下通过；但 strict forward/sample diagnostic 仍会看到 `1e-4` 量级以内的 shape-sensitive fp32 差异。
- r9/r10 说明在 no-TF32/fp32、单节点 4GPU DDP、3 update、`diagnostic_atol/rtol=1e-4` 下，bsz2 与 bsz1 replay 已经完整通过；这是当前训练集成层面的主要通过证据。
- r11 说明 4GPU、1 update 下 pre/post full-grad tensor 全量保存比较通过，关闭了 r9 中 `SAVE_GRAD_TENSORS=false` 的审计边界。
- r13 说明 4GPU、1 update、`per_rank_samples=2` 下 forward hook 中间层 probe 通过；r18 进一步说明同一 no-evo/fp32/no-TF32 控制变量下，4GPU、1 update、`per_rank_samples=10`、`FORWARD_PROBE_SAMPLE_POS=0` 的 forward-probe replay 也通过；r20 说明 2GPU、同控制变量、`FORWARD_PROBE_SAMPLE_POS=9` 也通过。三者共同补上 r9 未启用 forward-probe 的证据，其中 r18 覆盖 4GPU pos0，r20 仅作为 2GPU pos9 定向补强；该证据仍不覆盖所有 sample position，也不覆盖 3 update forward-probe；该结论使用 forward-probe 专用 `ATOL/RTOL=5e-3`。
- r22 说明 2GPU、1 update、`per_rank_samples=2` 下 full-grad tensor compare 和 forward-probe compare 可以在同一个 run 中同时通过，补强了“r11/r13 分拆补证”的审计边界；但它不替代 r9/r10，也不覆盖 3 update full-grad 或 `per_rank_samples=10` full-grad 组合。
- r8 说明单节点 8GPU、3 update replay artifact 已通过，补上了正式每节点 GPU 数的单节点工程 gate；但它不覆盖跨节点通信。
- r7 说明 16GPU、2 node x 8 GPU replay 已实际运行到 3 update，并且主 record/state/grad sync 全部通过；但 `rank=4/update=2/sample_pos=1` 的 `pred_coordinate max_abs=1.17e-4` 略超 `1e-4` diagnostic 阈值，因此不能 claim 16GPU strict replay pass。
- r21 说明同一正式 16GPU world size 在 `DIAGNOSTIC_ATOL/RTOL=2e-4` 下，3 update summary 和 rank-record 断言均通过；这补上了 16GPU 放宽数值尾差口径后的通过证据，但不改变 r7 对 `1e-4` strict 口径的失败事实。
- r14 说明如果缺少 `MODEL_DTYPE=fp32` 和 `USE_DEEPSPEED_EVO_ATTENTION=false` 这些控制变量，`per_rank_samples=10、FORWARD_PROBE_SAMPLE_POS=0` 的 forward probe 会失败；它是 dtype/kernel 敏感性的诊断信号，不是 padding batch 改造的反证。
- 对严格等价 replay，应使用 fp32 诊断配置并关闭 TF32；诊断容差建议将 `2e-5` 视为真实大模型 CUDA fp32 full-grad 路径的强诊断阈值，将 `1e-4` 视为单节点 DDP sample/grad 诊断阈值，将 `5e-3` 视为深层 forward hook sampled-value spot check 的当前工程容差，将 `5e-4` 作为训练集成层面的主阈值。
- 对正式训练，TF32 可以作为性能路径保留，但不能期望 `bsz2` 与 `bsz1` 在 `1e-6` 级别严格 replay 等价。正式训练应更多依赖 loss/grad/state 的合理容差和后续 PBP/ODesignBench 指标，而不是 bitwise 或 near-bitwise 等价。

尚未完成的边界：

- r4 的最终状态仍是 `status=fail`，不能宣称 strict replay 已完全通过；但失败口径已经从默认 TF32 下的 `1e-4` 级首个发散，收敛为 no-TF32/fp32 下的 `1e-6 ~ 1e-5` 级尾差。
- r5 已保存并比较完整 pre-clip/post-clip grad tensor，但仍只覆盖 1GPU、1 update、permutation disabled、deterministic MSA rows 的诊断配置。
- r6 已覆盖单节点 4GPU DDP 的更严格 `2e-5` diagnostic 口径，但未通过 strict forward/sample diagnostic。
- r9/r10 已覆盖单节点 4GPU DDP、3 update 和 `1e-4` diagnostic 通过，但 r9/r10 本身未保存完整 grad tensor、也未启用 forward probe；这些边界分别由 r11 和 r13 补证。
- r11 的 full-grad tensor 证据只覆盖 4GPU、1 update，不覆盖 3 update full-grad tensor 全量保存。
- r13 的 forward-probe 证据只覆盖 4GPU、1 update、`per_rank_samples=2` 的 lightweight replay；r18 已补到 4GPU、1 update、`per_rank_samples=10`、`FORWARD_PROBE_SAMPLE_POS=0`；r20 已补到 2GPU、1 update、`per_rank_samples=10`、`FORWARD_PROBE_SAMPLE_POS=9`；r22 已补到 2GPU、1 update、`per_rank_samples=2` 的 full-grad + forward-probe 组合。这些仍不覆盖所有 sample position，也不覆盖 3 update 全长度 forward-probe 或 full-grad replay。
- r7 已覆盖正式 world size 和跨节点通信的实际执行，但在 `1e-4` sample diagnostic 口径下有 1 个 `pred_coordinate` 尾差失败；r21 已补到 `2e-4` diagnostic 口径下 16GPU replay pass。尚未得到 16GPU `1e-4` strict replay pass。
- r8 已覆盖单节点 8GPU、3 update artifact pass，但 H200 MCP 聚合状态显示外层 job 为 `Failed`；需要后续确认外层状态为何与 replay summary 不一致。

## 阶段二推荐实验矩阵

| 实验 | 目的 | 规模 | 随机控制 | 通过标准 |
| --- | --- | --- | --- | --- |
| 1GPU replay | 排除 DDP，仅看真实数据 forward/loss/grad | 1 node x 1 GPU，1 update | 固定 sample trace、MSA、diffusion、dropout | per-sample tensors、loss、grad allclose |
| 4GPU replay | 覆盖单节点 DDP allreduce | 1 node x 4 GPU，1-3 updates | 同上，rank trace 固定 | 每 rank 和 all-rank grad/state hash 一致 |
| 16GPU replay | 覆盖正式 world size | 2 node x 8 GPU，2-3 updates | 同上 | 每步 loss/grad/state allclose |
| disable permutation replay | 判断 permutation 是否导致离散分叉 | 1GPU/4GPU | identity permutation | 如果通过而真实 permutation 不通过，重点查 permutation |
| deterministic MSA off/on 对照 | 判断 MSA row sampling 是否导致分叉 | 1GPU | 分别固定/不固定 MSA | 固定后通过则 MSA 是主因 |
| diffusion noise off/on 对照 | 判断 diffusion noise 是否导致分叉 | 1GPU | 固定 sigma/noise vs 原始 noise | 固定后通过则 noise 是主因 |

阶段二 hard gates：

- `sample_trace_equal == true`
- `noise_level allclose`
- `pred_coordinate real prefix allclose`
- `lddt_mask real prefix allclose`
- `loss components allclose`
- `pre_clip_grad allclose`
- `post_clip_grad allclose`
- `state_after_update allclose`

如果 `state_after_update allclose` 但 `loss/grad` 不 allclose，结论必须是“参数更新被 LR/optimizer/clip/容差掩盖”，不能判定训练等价。

## 阶段三：整体训练占位

阶段三应在阶段二通过后启动，建议拆成：

1. 正式 4GPU 短训 smoke：快速发现 runtime、保存 checkpoint、eval 脚本问题。
2. 正式 16GPU 从 0 开始训练：验证新 batchwise 训练完整链路。
3. 正式 16GPU resume 长训：验证从已有 checkpoint 继续训练的指标连续性。
4. checkpoint PBP/ODesignBench 测评：比较历史 best setting、bsz1 对照和 bsz2 新训练。

阶段三文档需要另行补充 training contract、evaluation contract、stop criteria、checkpoint 选择和评测报告模板。

## 当前项目状态摘要

阶段一：

- `tests/test_padded_collate.py`
- `tests/test_msa_token_mask.py`
- `tests/test_padding_batch_training_contract.py`
- `tests/test_padding_batch_end_to_end.py`

这些测试共同覆盖了 padding collate、batch prefix、MSA helper、Pairformer mask、Diffusion batch/sample axis、Permutation crop/restore、Loss reductions、single-step 和 multi-step optimizer 等价。

已记录验证证据见 `docs/padding_batch_unit_tests.zh.md`：

- full padding pytest：`zjow-odesign-pad-fullpytest-0602-r17-71919712`，结果 `42 passed in 209.15s`。
- multi-step optimizer 等价 GREEN：`zjow-odesign-pad-multistep-green-0604-r1-98276998`，结果 `1 passed in 200.84s`。

阶段一的未验证边界也必须明确保留：

- 不覆盖真实 weighted sampler/DDP rank trace。
- 不覆盖真实 dataset fetch 中的随机 crop/mask/augmentation 分支。
- 不覆盖真实 diffusion 随机 sigma/noise。
- 不覆盖所有生产 CUDA kernel、bf16/AMP、checkpoint recompute 和大结构 atom 数组合。
- 不覆盖正式长训的收敛性和 PBP/ODesignBench 质量。

阶段二：

- 已有 replay 结果显示只看参数更新不足够。曾出现 `state_compare allclose=True` 但 `loss` 和 `grad_summary` 不 allclose。
- 早期 replay 曾指向 `weighted_smooth_lddt_loss` 是主要差异项，但后续 forward hook 进一步显示，loss 差异之前已经存在 `pred_coordinate`/trunk 表征差异。
- r3/r4 诊断把首个大差异定位到默认 H200/TF32 数值路径下的 `InputFeatureEmbedder.atom_attention_encoder` 输出；关闭 TF32 后该差异降至 fp32 尾差量级。
- r9/r10 已在 no-TF32/fp32/no-evo 控制变量下完成 4GPU、3 update 主 replay；r11 补 full-grad tensor 1 update；r18 补 4GPU、`per_rank_samples=10`、1 update、`FORWARD_PROBE_SAMPLE_POS=0` forward-probe replay；r20 补 2GPU、`FORWARD_PROBE_SAMPLE_POS=9` repeat；r22 补 2GPU、`per_rank_samples=2`、1 update 的 full-grad + forward-probe 组合通过；r21 补 16GPU、3 update、`DIAGNOSTIC_ATOL/RTOL=2e-4` 通过。接下来阶段二的主要缺口是如果坚持 16GPU `1e-4` strict replay 口径仍未通过，以及如果需要更强证据时补更多 sample position 的 forward-probe sweep、3 update forward-probe 或 3 update full-grad tensor 组合。正式训练可以使用性能配置，但 strict replay 不能用 TF32 路径判定 padding 逻辑是否等价。
