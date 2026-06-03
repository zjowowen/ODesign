# Padding Batch Unit Test Plan

This document records the unit and smoke tests used to validate the ODesign
batchwise padding change. The main contract is:

> After randomness is controlled, one batch of two padded samples must produce
> the same forward tensors and parameter gradients as two single-sample runs
> whose losses are averaged, matching the effective DDP reduction semantics.

The tests are intentionally split into small contract tests and a tiny
end-to-end ODesign training smoke. The small tests isolate padding masks,
batched tensor axes, loss reductions, and attention mask wiring. The end-to-end
smoke exercises the real model path with nonzero Pairformer and diffusion
blocks.

## Files

- `tests/test_padded_collate.py`
  - Verifies that ragged ODesign samples are padded into a batch.
  - Checks `token_padding_mask` and `atom_padding_mask`.
  - Checks that padded token, atom, bond, and MSA regions are masked out.
  - Checks that basic metadata and optional fields are collated without losing
    per-sample structure.

- `tests/test_msa_token_mask.py`
  - Verifies batched MSA/token helper behavior.
  - Covers token-mask broadcast over the MSA row axis.
  - Covers MSA row chunking with a batch prefix in inference helpers.

- `tests/test_padding_batch_training_contract.py`
  - Verifies batch-prefix handling in lower-level model utilities and losses.
  - Covers token/atom aggregation, attention masks, Pairformer mask forwarding,
    diffusion transformer mask forwarding, sparse and dense losses, resolution
    gates, and rigid alignment masks.

- `tests/test_padding_batch_end_to_end.py`
  - Builds a tiny CUDA ODesign config and runs padded batch training forward and
    backward.
  - Verifies the core batch=2 versus two batch=1 training-equivalence contract.
  - Uses `pairformer.n_blocks=1`, `diffusion_module.atom_encoder.n_blocks=1`,
    `diffusion_module.transformer.n_blocks=1`, and
    `diffusion_module.atom_decoder.n_blocks=1`, so this is not only a shape-only
    stub.

## Core Equivalence Test

The strongest unit-level training check is
`test_padding_batch_matches_average_of_single_item_training_steps` in
`tests/test_padding_batch_end_to_end.py`.

It builds two samples with different token and atom lengths:

- sample 0: 3 tokens, 4 atoms
- sample 1: 5 tokens, 7 atoms

The test then:

1. Collates both samples with `collate_fn_odesign_padded`.
2. Collates each sample alone for the single-item comparison runs.
3. Creates two identical models by copying the batched model state dict into
   the micro-batch model.
4. Controls stochastic training effects:
   - seeds Torch and CUDA;
   - replaces `centre_random_augmentation` with a deterministic centering
     function;
   - makes the training diffusion scheduler return a fixed sigma and a
     deterministic noisy coordinate transform.
5. Runs one forward/loss/backward pass on the padded batch.
6. Runs one forward/loss pass for each single item, averages the two losses, and
   calls backward once on the averaged scalar.
7. Compares:
   - scalar training loss;
   - `metrics["loss"]`;
   - ground-truth coordinate prefixes;
   - `LossInput.atom_padding_mask` prefixes;
   - predicted coordinate prefixes;
   - predicted distogram real-token blocks;
   - predicted token-bond-type real-token blocks;
   - every parameter gradient that is present in both models.

The comparison uses `torch.allclose(..., atol=2e-5, rtol=2e-5)` for model
outputs and gradients. This tolerance is small enough to catch padding leakage
while allowing normal CUDA floating-point ordering differences.

## Mask and Batch-Axis Contract Tests

`tests/test_padding_batch_training_contract.py` covers the smaller contracts
that the end-to-end test depends on.

### Structure Slicing and Restoration

- `_is_batched_structure_tensor` identifies batched coordinate-like tensors.
- `_slice_batch_item` slices only batch-sized tensors and known collated
  permutation lists.
- `_crop_atom_prefix` crops atom-axis fields by the real atom length.
- `_restore_atom_prefix_output` writes real prefixes back into padded outputs
  without overwriting padded suffixes.
- `_stack_odesign_outputs` stacks tensor fields and preserves `None` fields.

These tests protect the inference and permutation helpers from confusing batch
axes with atom/sample axes.

### Token/Atom Broadcast and Aggregation

- `broadcast_token_to_atom` keeps legacy unbatched behavior.
- `broadcast_token_to_atom` supports equal batch prefixes.
- `broadcast_token_to_atom` expands batched atom indices over a diffusion sample
  dimension.
- `aggregate_atom_to_token` expands batched atom indices over a diffusion sample
  dimension.
- `aggregate_atom_to_token(..., atom_mask=...)` ignores padded or otherwise
  masked atoms when computing token means.

The masked aggregation test is important because padded atoms can otherwise
silently bias token-level representations.

### Pair and Diffusion Attention Masks

- `AttentionPairBias` turns a standard attention mask into a large negative
  attention bias for invalid key/query pairs.
- `DiffusionTransformer` blocks padded token gradients: a loss on real tokens
  produces zero gradient on the padded token input.
- `PairformerBlock` passes `pair_mask` into its single-token
  `AttentionPairBias`.
- `DiffusionTransformer` forwards `attn_mask` to each block.

These tests cover the main route by which token padding masks prevent padded
tokens from participating in attention.

### Losses and Reductions

- Resolution masks are vectorized per example.
- Resolution gates average only valid examples.
- Last-dimension atom masks preserve batched ragged atom axes and reject shape
  mismatches.
- `BondTypeLoss` accepts batched labels and masks and rejects mismatched batch
  sizes.
- Sparse bond loss preserves batched pair indices, scaling, mean reduction, and
  differentiable zero outputs when no bonds are present.
- Sparse `SmoothLDDTLoss` preserves batched pair indices, reduces batched
  prefixes correctly, chunks over the diffusion sample axis, and returns a
  differentiable zero for an empty sparse mask.
- Dense `SmoothLDDTLoss` preserves both batch and diffusion sample axes.
- `DistogramLoss` accepts a batched ragged representative-atom mask.

These tests are aimed at preventing accidental flattening of batch and sample
axes, and preventing padded atoms from contributing to loss denominators.

### Diffusion Alignment Masks

- `_diffusion_condition_align_mask` computes its threshold per example and is
  padding-aware.
- `MSELoss.weighted_rigid_align` uses `align_mask` for alignment weights.

The `align_mask` regression was verified with a RED/GREEN cycle:

- RED job: `zjow-odesign-pad-align-red-0602-52266937`
  - Expected failure:
    `test_mse_weighted_rigid_align_uses_align_mask_for_alignment_weights`
  - The failure showed that an outlier atom outside `align_mask` could drag the
    rigid alignment.
- GREEN job: `zjow-odesign-pad-align-green-0602-27388901`
  - Result: `1 passed in 204.38s`

## End-to-End Training Smoke

`test_padding_batch_train_forward_backward_smoke` in
`tests/test_padding_batch_end_to_end.py` runs the tiny ODesign model on a
two-sample padded batch, computes `ODesignLoss`, and calls `loss.backward()`.

It checks:

- loss and `metrics["loss"]` are finite;
- predicted coordinates have batch/sample/atom shape `(2, 1, 7)`;
- ground-truth coordinates have padded atom shape `(2, 7, 3)`;
- at least one model parameter has a finite gradient.

The tiny config enables real block paths:

- Pairformer block count: `1`
- diffusion atom encoder block count: `1`
- diffusion transformer block count: `1`
- diffusion atom decoder block count: `1`

The config disables optional high-performance kernels where possible, but the H
runtime still requires CUDA because fused LayerNorm has no CPU fallback in this
environment.

## How To Rerun

Run these tests inside the H ODesign CUDA container:

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

The local macOS checkout is not a reliable execution environment for these
tests because it uses Python 3.9 and does not provide the CUDA fused LayerNorm
runtime expected by the current ODesign image.

For a full padding-related selection, also run:

```bash
python -m pytest -q \
  tests/test_padded_collate.py \
  tests/test_msa_token_mask.py \
  tests/test_padding_batch_training_contract.py \
  tests/test_padding_batch_end_to_end.py
```

## Last Verified Evidence

The latest full padding pytest run recorded during this work was:

- job: `zjow-odesign-pad-fullpytest-0602-r17-71919712`
- record directory:
  `/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/padding-batch-fullpytest-0602-r17`
- result: `42 passed in 209.15s`
- selected hashes at runtime:
  - `src/model/modules/loss.py`:
    `504eb00696bbdeb461d8b7e0b85aabe20ac99e2ad85d84be92f6ba3b999121c2`
  - `src/model/modules/transformer.py`:
    `b2d7fbcd60a916ac1f689a2993e2413ba460547cb361a73f3eb77a5b3dbfddb2`
  - `tests/test_padding_batch_training_contract.py`:
    `de5162471a92be7eb009f7f7dbd71b1bdca90daf61498834b2202d568b193009`
  - `tests/test_padding_batch_end_to_end.py`:
    `c189b7d43235a8cb398cda5dd941f378bc7b45a92b323998c6bbe6ac7473857e`

## Current Boundaries

These tests validate the batchwise padding training contract and the key
padding-aware model/loss paths. They do not by themselves prove:

- broad real-data convergence;
- long-run 16-GPU stability;
- batched evaluation or inference quality;
- absence of bugs in unrelated non-padding model paths.

A separate best-setting short training job was launched to cover the next
validation layer, but that training evidence should be treated separately from
the unit-test evidence in this document.
