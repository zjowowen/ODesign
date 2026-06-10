# Padding Batch 单元测试方案

本文档记录这次 ODesign batchwise padding 改动对应的单元测试和 smoke
测试。核心验证契约是：

> 在控制随机性之后，一个包含两个 padded sample 的 batch=2 前后向结果，
> 应该等价于两个 batch=1 单样本前后向结果按 DDP 语义取平均。

这里的“等价”不是只看 loss 是否有限，而是同时比较前向输出张量和参数梯度。
测试被拆成两层：

- 小粒度 contract tests：隔离验证 padding mask、batch 轴、sample 轴、loss
  reduction 和 attention mask 传递。
- tiny ODesign 端到端训练 smoke：用一个很小但真实的 ODesign 配置跑完整
  forward/loss/backward，并打开非零数量的 Pairformer 和 diffusion blocks。

## 测试文件

- `tests/test_padded_collate.py`
  - 验证 ragged ODesign 样本可以被 pad 成 batch。
  - 验证 `token_padding_mask` 和 `atom_padding_mask`。
  - 验证 padded token、atom、bond、MSA 区域都会被 mask 掉。
  - 验证 basic metadata 和 optional fields 在 collate 后仍保留每个样本的结构。

- `tests/test_msa_token_mask.py`
  - 验证带 batch prefix 的 MSA/token helper 行为。
  - 覆盖 token mask 在 MSA row 轴上的 broadcast。
  - 覆盖 inference helper 中带 batch prefix 的 MSA row chunking。

- `tests/test_padding_batch_training_contract.py`
  - 验证底层 model utilities 和 losses 对 batch prefix 的处理。
  - 覆盖 token/atom 聚合、attention masks、Pairformer mask 传递、diffusion
    transformer mask 传递、sparse/dense losses、resolution gates 和 rigid
    alignment masks。

- `tests/test_padding_batch_end_to_end.py`
  - 构造一个 tiny CUDA ODesign 配置，并在 padded batch 上跑训练 forward 和
    backward。
  - 验证核心的 batch=2 vs 两个 batch=1 平均的训练等价性。
  - 验证同一等价性在多个 optimizer step 上仍成立，包括中间张量、梯度以及每次
    update 后的模型参数。
  - tiny config 中设置了 `pairformer.n_blocks=1`、
    `diffusion_module.atom_encoder.n_blocks=1`、
    `diffusion_module.transformer.n_blocks=1`、
    `diffusion_module.atom_decoder.n_blocks=1`，所以它不是只测 shape 的 stub。

## 核心训练等价性测试

最强的训练级别单元测试是
`tests/test_padding_batch_end_to_end.py` 里的
`test_padding_batch_matches_average_of_single_item_training_steps`。

这个测试构造两个长度不同的样本：

- sample 0：3 tokens，4 atoms
- sample 1：5 tokens，7 atoms

测试流程如下：

1. 使用 `collate_fn_odesign_padded` 把两个样本 collate 成一个 padded batch。
2. 分别把两个样本各自 collate 成单样本 batch，用作 batch=1 对照。
3. 创建两个完全相同的模型，把 batched model 的 state dict 拷贝到
   micro-batch model。
4. 控制训练中的随机性：
   - 固定 Torch 和 CUDA seed；
   - 用确定性的 centering 函数替换 `centre_random_augmentation`；
   - 让训练 diffusion scheduler 返回固定 sigma，并使用确定性的 noisy
     coordinate transform。
5. 对 padded batch 执行一次 forward/loss/backward。
6. 对两个单样本 batch 分别执行 forward/loss，将两个 loss 取平均后调用一次
   backward。
7. 比较以下内容：
   - scalar training loss；
   - `metrics["loss"]`；
   - ground-truth coordinate 的真实 atom prefix；
   - `LossInput.atom_padding_mask` 的真实 atom prefix；
   - 预测 coordinate 的真实 atom prefix；
   - 预测 distogram 的真实 token block；
   - 预测 token-bond-type logits 的真实 token block；
   - 两个模型中每一个实际存在的参数梯度。

前向输出和参数梯度使用
`torch.allclose(..., atol=2e-5, rtol=2e-5)` 比较。这个阈值足够小，可以捕获
padding 泄漏，同时允许 CUDA 浮点计算顺序带来的正常细微差异。

## 多步 Optimizer 等价性测试

`test_padding_batch_matches_microbatch_accumulation_across_optimizer_steps` 把
单步检查扩展成一段很短的 optimizer 轨迹。

这个测试使用和单步测试相同的两个 ragged samples，也使用相同的随机性控制。它先
创建两个初始参数完全相同的模型，然后执行 3 次 SGD optimizer update：

1. padded model 执行一次 batch=2 forward/loss/backward。
2. micro-batch model 分别执行两次 batch=1 forward，把两个 scalar loss 平均后，
   对平均 loss 调用一次 backward。
3. 每次 optimizer step 之前，比较：
   - scalar loss 和每一个上报的 metric；
   - 真实 atom prefix 上的 ground-truth coordinates、coordinate masks、
     `distance_mask` 和 `lddt_mask`；
   - 真实 atom prefix 上的 `LossInput` atom masks 和分子类型 masks；
   - 真实 prefix 上的预测 coordinates、distogram logits、bond-type logits、
     bond-generation masks 和 diffusion noise levels；
   - 每一个参数梯度。
4. 每次 optimizer step 之后，比较每一个模型参数。

padding-only suffix 区域不会和单样本运行比较，因为单样本 batch 里不存在这些
padded suffix。这个测试验证的契约是：真实 token/atom prefix、梯度、loss 以及
update 后的模型参数都保持 allclose。

## Mask 和 Batch 轴 Contract Tests

`tests/test_padding_batch_training_contract.py` 覆盖 end-to-end 测试依赖的底层
契约。

### 结构切片和恢复

- `_is_batched_structure_tensor` 识别带 batch 维的 coordinate-like tensors。
- `_slice_batch_item` 只切 batch-sized tensors 和已知的 collated permutation
  lists。
- `_crop_atom_prefix` 按真实 atom 长度裁剪 atom-axis fields。
- `_restore_atom_prefix_output` 把真实 prefix 写回 padded output，同时不覆盖
  padded suffix。
- `_stack_odesign_outputs` stack tensor fields，并保留 `None` fields。

这组测试用于避免 inference/permutation helpers 把 batch 轴和 atom/sample 轴混淆。

### Token/Atom Broadcast 和 Aggregation

- `broadcast_token_to_atom` 保持 legacy unbatched 行为。
- `broadcast_token_to_atom` 支持相同的 batch prefix。
- `broadcast_token_to_atom` 可以把 batched atom index 扩展到 diffusion sample
  维。
- `aggregate_atom_to_token` 可以把 batched atom index 扩展到 diffusion sample
  维。
- `aggregate_atom_to_token(..., atom_mask=...)` 在计算 token mean 时忽略 padded
  atoms 或其他被 mask 的 atoms。

masked aggregation 测试很关键，因为如果 padded atoms 参与均值，会悄悄污染
token-level representation。

### Pair 和 Diffusion Attention Masks

- `AttentionPairBias` 会把标准 attention mask 转成 invalid key/query pair 上的
  large negative attention bias。
- `DiffusionTransformer` 能阻断 padded token 的梯度：只对真实 token 取 loss 时，
  padded token input 的梯度为零。
- `PairformerBlock` 会把 `pair_mask` 传给 single-token `AttentionPairBias`。
- `DiffusionTransformer` 会把 `attn_mask` 传给每个 block。

这组测试覆盖 token padding mask 阻止 padded tokens 参与 attention 的主要路径。

### Losses 和 Reductions

- Resolution mask 按样本向量化计算。
- Resolution gate 只对 valid examples 求平均。
- Last-dimension atom mask 保留 batched ragged atom axis，并拒绝 shape mismatch。
- `BondTypeLoss` 接受 batched labels/masks，并拒绝 batch size 不匹配。
- Sparse bond loss 保留 batched pair indices、scale、mean reduction，并在没有
  bonds 时返回 differentiable zero。
- Sparse `SmoothLDDTLoss` 保留 batched pair indices，正确 reduce batched
  prefix，沿 diffusion sample axis chunk，并在 empty sparse mask 时返回
  differentiable zero。
- Dense `SmoothLDDTLoss` 同时保留 batch 轴和 diffusion sample 轴。
- `DistogramLoss` 接受 batched ragged representative-atom mask。

这些测试主要防止 batch 轴和 sample 轴被错误 flatten，也防止 padded atoms 进入
loss denominator。

### Diffusion Alignment Masks

- `_diffusion_condition_align_mask` 的 threshold 是 per-example 的，并且
  padding-aware。
- `MSELoss.weighted_rigid_align` 使用 `align_mask` 作为 alignment weights。

`align_mask` 回归问题做过 RED/GREEN 验证：

- RED job：`zjow-odesign-pad-align-red-0602-52266937`
  - 预期失败测试：
    `test_mse_weighted_rigid_align_uses_align_mask_for_alignment_weights`
  - 失败现象说明：`align_mask` 外的 outlier atom 会拖偏 rigid alignment。
- GREEN job：`zjow-odesign-pad-align-green-0602-27388901`
  - 结果：`1 passed in 204.38s`

## 端到端训练 Smoke

`tests/test_padding_batch_end_to_end.py` 里的
`test_padding_batch_train_forward_backward_smoke` 会在两个样本组成的 padded batch
上运行 tiny ODesign，计算 `ODesignLoss`，并调用 `loss.backward()`。

它检查：

- loss 和 `metrics["loss"]` 是 finite；
- 预测坐标的 batch/sample/atom shape 是 `(2, 1, 7)`；
- ground-truth 坐标的 padded atom shape 是 `(2, 7, 3)`；
- 至少一个模型参数有 finite gradient。

tiny config 打开了真实 block 路径：

- Pairformer block count：`1`
- diffusion atom encoder block count：`1`
- diffusion transformer block count：`1`
- diffusion atom decoder block count：`1`

这个 config 尽量关闭了可选的高性能 kernel，但当前 H runtime 仍然需要 CUDA，
因为该环境中的 fused LayerNorm 没有 CPU fallback。

## 如何复跑

在 H 侧 ODesign CUDA 容器中运行：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate odesign
cd /mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign
export PYTHONPATH="$PWD:$PWD/src:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python -m pip install -q "setuptools<81" "pytest"
python -m pytest -q \
  tests/test_padding_batch_training_contract.py \
  tests/test_padding_batch_end_to_end.py
```

本地 macOS checkout 不是可靠的执行环境，因为本地是 Python 3.9，并且没有当前
ODesign image 依赖的 CUDA fused LayerNorm runtime。

如果要跑完整 padding 相关测试集，也可以运行：

```bash
python -m pytest -q \
  tests/test_padded_collate.py \
  tests/test_msa_token_mask.py \
  tests/test_padding_batch_training_contract.py \
  tests/test_padding_batch_end_to_end.py
```

## 最近一次验证证据

这次工作中记录到的最近一次 full padding pytest run 是：

- job：`zjow-odesign-pad-fullpytest-0602-r17-71919712`
- record directory：
  `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/padding-batch-fullpytest-0602-r17`
- 结果：`42 passed in 209.15s`
- runtime selected hashes：
  - `src/model/modules/loss.py`：
    `504eb00696bbdeb461d8b7e0b85aabe20ac99e2ad85d84be92f6ba3b999121c2`
  - `src/model/modules/transformer.py`：
    `b2d7fbcd60a916ac1f689a2993e2413ba460547cb361a73f3eb77a5b3dbfddb2`
  - `tests/test_padding_batch_training_contract.py`：
    `de5162471a92be7eb009f7f7dbd71b1bdca90daf61498834b2202d568b193009`
  - `tests/test_padding_batch_end_to_end.py`：
    `c189b7d43235a8cb398cda5dd941f378bc7b45a92b323998c6bbe6ac7473857e`

最新一次专门的多步 optimizer 等价性验证是：

- RED job：`zjow-odesign-pad-multistep-red-0604-r2-91580369`
  - 结果：预期失败，
    `NameError: name '_run_multi_step_padding_batch_equivalence' is not defined`
  - record directory：
    `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign/.cluster_operator/padding-batch-multistep-red-0604-r2`
- GREEN job：`zjow-odesign-pad-multistep-green-0604-r1-98276998`
  - 结果：`1 passed in 200.84s`
  - record directory：
    `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/bestsetting-padding-runtime-0601/ODesign/.cluster_operator/padding-batch-multistep-green-0604-r1`

## 当前边界

这些测试验证的是 batchwise padding 的训练契约，以及关键 model/loss 路径是否
padding-aware。它们本身不证明：

- 大规模真实数据训练一定收敛；
- 16-GPU 长时间训练稳定性；
- batched evaluation 或 inference 质量；
- 与 padding 无关的其他模型路径完全没有 bug。

另有一个 best-setting short training job 用于覆盖下一层验证，但它的训练证据应
和本文档中的单元测试证据分开看。
