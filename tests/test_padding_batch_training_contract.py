import math

import pytest
import torch
from attr import define

from src.api.model_interface import ODesignOutput
from src.model.modules.generator import _diffusion_sample_shape
from src.model.modules.head import BondTypeHead
from src.model.modules.loss import (
    BondLoss,
    BondTypeLoss,
    DistogramLoss,
    _diffusion_condition_align_mask,
    MSELoss,
    SmoothLDDTLoss,
    _apply_last_dim_mask,
    _apply_resolution_gate,
    _valid_resolution_mask,
)
from src.model.modules.primitives import gather_pair_embedding_in_dense_trunk
from src.model.modules.pairformer import PairformerBlock
from src.model.modules.transformer import AttentionPairBias, DiffusionTransformer
from src.utils.permutation.permutation import (
    _crop_atom_prefix,
    _is_batched_structure_tensor,
    _real_atom_len_from_coordinate_mask,
    _restore_atom_prefix_output,
    _slice_batch_item,
    _stack_odesign_outputs,
)
from src.utils.model.misc import (
    aggregate_atom_to_token,
    broadcast_token_to_atom,
    centre_random_augmentation,
)


@define
class BatchSliceFixture:
    tensor_field: torch.Tensor
    list_field: list
    preserved_tensor: torch.Tensor
    preserved_value: str


@define
class CollatedListFixture:
    atom_perm_list: list
    masked_asym_ids: list


@define
class AtomCropFixture:
    coordinate: torch.Tensor
    coordinate_mask: torch.Tensor
    ref_atom_name_chars: torch.Tensor
    ligand_bond_mask: torch.Tensor
    atom_perm_list: list
    metadata: list


def test_is_batched_structure_tensor_identifies_batched_coordinate() -> None:
    assert _is_batched_structure_tensor(torch.zeros(2, 8, 3)) is True
    assert _is_batched_structure_tensor(torch.zeros(8, 3)) is False


def test_slice_batch_item_slices_batch_sized_tensor_but_preserves_arbitrary_lists() -> None:
    data = BatchSliceFixture(
        tensor_field=torch.tensor([[1, 2, 3], [4, 5, 6]]),
        list_field=["first", "second"],
        preserved_tensor=torch.tensor([7, 8, 9]),
        preserved_value="metadata",
    )

    sliced = _slice_batch_item(data, batch_idx=1, batch_size=2)

    assert sliced.tensor_field.tolist() == [4, 5, 6]
    assert sliced.list_field == ["first", "second"]
    assert sliced.preserved_tensor.tolist() == [7, 8, 9]
    assert sliced.preserved_value == "metadata"


def test_slice_batch_item_selects_known_collated_permutation_lists() -> None:
    data = CollatedListFixture(
        atom_perm_list=[[[0, 1], [1, 0]], [[2, 3], [3, 2]]],
        masked_asym_ids=[[10], [20, 21]],
    )

    sliced = _slice_batch_item(data, batch_idx=1, batch_size=2)

    assert sliced.atom_perm_list == [[2, 3], [3, 2]]
    assert sliced.masked_asym_ids == [20, 21]


def test_crop_atom_prefix_uses_real_atom_length_for_atom_axis_fields() -> None:
    data = AtomCropFixture(
        coordinate=torch.arange(5 * 3).reshape(5, 3),
        coordinate_mask=torch.tensor([True, False, True, False, False]),
        ref_atom_name_chars=torch.zeros(5, 4, 3),
        ligand_bond_mask=torch.ones(5, 5),
        atom_perm_list=[[0], [1], [2], [3], [4]],
        metadata=["first", "second"],
    )

    assert _real_atom_len_from_coordinate_mask(data.coordinate_mask) == 3

    cropped = _crop_atom_prefix(data, real_atom_len=3)

    assert cropped.coordinate.shape == (3, 3)
    assert cropped.coordinate_mask.tolist() == [True, False, True]
    assert cropped.ref_atom_name_chars.shape == (3, 4, 3)
    assert cropped.ligand_bond_mask.shape == (3, 3)
    assert cropped.atom_perm_list == [[0], [1], [2]]
    assert cropped.metadata == ["first", "second"]


def test_restore_atom_prefix_output_updates_real_prefix_and_preserves_padding() -> None:
    padded = ODesignOutput(
        coordinate=torch.zeros(2, 5, 3),
        noise_level=torch.tensor([0.1, 0.2]),
        distance=torch.zeros(2, 5, 5),
    )
    real = ODesignOutput(
        coordinate=torch.ones(2, 3, 3),
        noise_level=torch.tensor([9.0, 9.0]),
        distance=torch.ones(2, 3, 3),
    )

    restored = _restore_atom_prefix_output(padded, real, real_atom_len=3)

    assert restored.coordinate[:, :3].tolist() == torch.ones(2, 3, 3).tolist()
    assert restored.coordinate[:, 3:].tolist() == torch.zeros(2, 2, 3).tolist()
    assert restored.distance[:, :3, :3].tolist() == torch.ones(2, 3, 3).tolist()
    assert restored.distance[:, 3:, :].tolist() == torch.zeros(2, 2, 5).tolist()
    assert restored.distance[:, :, 3:].tolist() == torch.zeros(2, 5, 2).tolist()
    assert torch.equal(restored.noise_level, padded.noise_level)


def test_stack_odesign_outputs_stacks_tensor_fields_and_preserves_none() -> None:
    outputs = [
        ODesignOutput(
            coordinate=torch.zeros(3, 8, 3),
            noise_level=torch.tensor([0.1, 0.2, 0.3]),
        ),
        ODesignOutput(
            coordinate=torch.ones(3, 8, 3),
            noise_level=torch.tensor([0.4, 0.5, 0.6]),
        ),
    ]

    stacked = _stack_odesign_outputs(outputs)

    assert stacked.coordinate.shape == (2, 3, 8, 3)
    assert stacked.noise_level.shape == (2, 3)
    assert stacked.distogram is None


def test_bond_type_head_preserves_batch_prefix() -> None:
    head = BondTypeHead(c_z=4, c_hidden=8, no_bond_types=3)

    logits = head(torch.zeros(2, 5, 5, 4))

    assert logits.shape == (2, 5, 5, 3)


def test_bond_type_head_adds_legacy_batch_prefix_for_unbatched_input() -> None:
    head = BondTypeHead(c_z=4, c_hidden=8, no_bond_types=3)

    logits = head(torch.zeros(5, 5, 4))

    assert logits.shape == (1, 5, 5, 3)


def test_diffusion_sample_shape_preserves_batch_prefix() -> None:
    assert _diffusion_sample_shape(torch.zeros(2, 8, 3), 4) == (2, 4, 8, 3)


def test_centre_random_augmentation_accepts_batched_mask() -> None:
    coords = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    mask = torch.tensor(
        [
            [True, True, False, False],
            [True, False, True, False],
        ]
    )

    augmented = centre_random_augmentation(
        coords,
        N_sample=2,
        centre_only=True,
        mask=mask,
        dtype=torch.float32,
    )

    assert augmented.shape == (2, 2, 4, 3)


def test_broadcast_token_to_atom_preserves_legacy_unbatched_index() -> None:
    atom_to_token_idx = torch.tensor([1, 0, 1])
    token = torch.tensor([[10.0], [20.0]])
    sampled_token = torch.tensor(
        [
            [[10.0], [20.0]],
            [[30.0], [40.0]],
        ]
    )

    assert broadcast_token_to_atom(token, atom_to_token_idx).tolist() == [
        [20.0],
        [10.0],
        [20.0],
    ]
    assert broadcast_token_to_atom(sampled_token, atom_to_token_idx).tolist() == [
        [[20.0], [10.0], [20.0]],
        [[40.0], [30.0], [40.0]],
    ]


def test_broadcast_token_to_atom_preserves_equal_batch_prefix() -> None:
    token = torch.tensor(
        [
            [[10.0], [20.0], [30.0]],
            [[40.0], [50.0], [60.0]],
        ]
    )
    atom_to_token_idx = torch.tensor(
        [
            [2, 0],
            [1, 2],
        ]
    )

    atom = broadcast_token_to_atom(token, atom_to_token_idx)

    assert atom.shape == (2, 2, 1)
    assert atom.tolist() == [
        [[30.0], [10.0]],
        [[50.0], [60.0]],
    ]


def test_broadcast_token_to_atom_expands_batched_index_over_sample_dim() -> None:
    token = torch.tensor(
        [
            [
                [[10.0], [20.0], [30.0]],
                [[40.0], [50.0], [60.0]],
            ],
            [
                [[70.0], [80.0], [90.0]],
                [[100.0], [110.0], [120.0]],
            ],
        ]
    )
    atom_to_token_idx = torch.tensor(
        [
            [2, 0],
            [1, 2],
        ]
    )

    atom = broadcast_token_to_atom(token, atom_to_token_idx)

    assert atom.shape == (2, 2, 2, 1)
    assert atom.tolist() == [
        [
            [[30.0], [10.0]],
            [[60.0], [40.0]],
        ],
        [
            [[80.0], [90.0]],
            [[110.0], [120.0]],
        ],
    ]


def test_aggregate_atom_to_token_expands_batched_index_over_sample_dim() -> None:
    atom = torch.tensor(
        [
            [
                [[1.0], [10.0], [3.0]],
                [[5.0], [20.0], [7.0]],
            ],
            [
                [[30.0], [2.0], [4.0]],
                [[40.0], [6.0], [8.0]],
            ],
        ]
    )
    atom_to_token_idx = torch.tensor(
        [
            [0, 1, 0],
            [1, 2, 1],
        ]
    )

    token = aggregate_atom_to_token(
        atom,
        atom_to_token_idx=atom_to_token_idx,
        n_token=3,
        reduce="mean",
    )

    assert token.shape == (2, 2, 3, 1)
    assert token.tolist() == [
        [
            [[2.0], [10.0], [0.0]],
            [[6.0], [20.0], [0.0]],
        ],
        [
            [[0.0], [17.0], [2.0]],
            [[0.0], [24.0], [6.0]],
        ],
    ]


def test_aggregate_atom_to_token_ignores_masked_atoms_in_mean() -> None:
    atom = torch.tensor(
        [
            [
                [[1.0], [10.0], [3.0], [100.0]],
                [[5.0], [20.0], [7.0], [200.0]],
            ]
        ]
    )
    atom_to_token_idx = torch.tensor([[0, 1, 0, 0]])
    atom_mask = torch.tensor([[True, True, True, False]])

    token = aggregate_atom_to_token(
        atom,
        atom_to_token_idx=atom_to_token_idx,
        atom_mask=atom_mask,
        n_token=2,
        reduce="mean",
    )

    assert token.shape == (1, 2, 2, 1)
    assert token.tolist() == [
        [
            [[2.0], [10.0]],
            [[6.0], [20.0]],
        ]
    ]


def test_gather_pair_embedding_in_dense_trunk_expands_batched_indices_over_sample_dim() -> None:
    z_token = torch.arange(2 * 2 * 3 * 3, dtype=torch.float32).reshape(
        2, 2, 3, 3, 1
    )
    idx_q = torch.tensor(
        [
            [[2, 0]],
            [[1, 2]],
        ]
    )
    idx_k = torch.tensor(
        [
            [[0, 1, 2]],
            [[2, 0, 1]],
        ]
    )

    gathered = gather_pair_embedding_in_dense_trunk(
        z_token,
        idx_q=idx_q,
        idx_k=idx_k,
    )

    assert gathered.shape == (2, 2, 1, 2, 3, 1)
    assert gathered[0, 0, 0, :, :, 0].tolist() == [
        [6.0, 7.0, 8.0],
        [0.0, 1.0, 2.0],
    ]
    assert gathered[0, 1, 0, :, :, 0].tolist() == [
        [15.0, 16.0, 17.0],
        [9.0, 10.0, 11.0],
    ]
    assert gathered[1, 0, 0, :, :, 0].tolist() == [
        [23.0, 21.0, 22.0],
        [26.0, 24.0, 25.0],
    ]
    assert gathered[1, 1, 0, :, :, 0].tolist() == [
        [32.0, 30.0, 31.0],
        [35.0, 33.0, 34.0],
    ]


def test_attention_pair_bias_applies_standard_attention_mask() -> None:
    if not torch.cuda.is_available():
        pytest.skip("AttentionPairBias uses fused LayerNorm in the H CUDA runtime.")
    device = torch.device("cuda")
    captured = {}

    class CaptureAttention(torch.nn.Module):
        def forward(self, q_x, kv_x, *, attn_bias=None, **kwargs):
            del kv_x, kwargs
            captured["attn_bias"] = attn_bias
            return torch.zeros_like(q_x)

    module = AttentionPairBias(has_s=False, n_heads=1, c_a=4, c_z=2).to(device)
    module.attention = CaptureAttention()
    attn_mask = torch.tensor(
        [
            [True, True, False],
            [True, True, False],
            [False, False, False],
        ],
        device=device,
    )

    module(
        a=torch.zeros(3, 4, device=device),
        s=None,
        z=torch.zeros(3, 3, 2, device=device),
        attn_mask=attn_mask,
    )

    attn_bias = captured["attn_bias"]
    assert attn_bias.shape == (1, 3, 3)
    assert torch.all(attn_bias[..., :2, :2] == 0)
    assert torch.all(attn_bias[..., :2, 2] < -1e9)


def test_diffusion_transformer_mask_blocks_padded_token_gradients() -> None:
    if not torch.cuda.is_available():
        pytest.skip("DiffusionTransformer uses fused LayerNorm in the H CUDA runtime.")
    device = torch.device("cuda")
    torch.manual_seed(7)
    transformer = DiffusionTransformer(
        c_a=4,
        c_s=4,
        c_z=2,
        n_blocks=1,
        n_heads=1,
    ).to(device)
    transformer.train()

    a = torch.randn(1, 3, 4, device=device, requires_grad=True)
    s = torch.randn(1, 3, 4, device=device)
    z = torch.randn(1, 3, 3, 2, device=device)
    attn_mask = torch.tensor(
        [
            [
                [True, True, False],
                [True, True, False],
                [False, False, False],
            ]
        ],
        device=device,
    )

    out = transformer(a=a, s=s, z=z, attn_mask=attn_mask)
    out[:, :2].sum().backward()

    assert torch.allclose(a.grad[:, 2], torch.zeros_like(a.grad[:, 2]), atol=1e-7)


def test_pairformer_block_passes_pair_mask_to_single_attention() -> None:
    captured = {}

    class ZeroLike(torch.nn.Module):
        def forward(self, x, *args, **kwargs):
            del args, kwargs
            return torch.zeros_like(x)

    class CapturePairBias(torch.nn.Module):
        def forward(self, *, a, s, z, attn_mask=None, **kwargs):
            del s, z, kwargs
            captured["attn_mask"] = attn_mask
            return torch.zeros_like(a)

    block = PairformerBlock(c_s=4, c_z=2, n_heads=1, dropout=0.0)
    block.tri_mul_out = ZeroLike()
    block.tri_mul_in = ZeroLike()
    block.tri_att_start = ZeroLike()
    block.tri_att_end = ZeroLike()
    block.pair_transition = ZeroLike()
    block.single_transition = ZeroLike()
    block.dropout_row = torch.nn.Identity()
    block.attention_pair_bias = CapturePairBias()
    pair_mask = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    block(
        s=torch.zeros(3, 4),
        z=torch.zeros(3, 3, 2),
        pair_mask=pair_mask,
    )

    assert torch.equal(captured["attn_mask"], pair_mask)


def test_diffusion_transformer_passes_token_attention_mask_to_blocks() -> None:
    captured = {}

    class CaptureBlock(torch.nn.Module):
        def forward(self, a, s, z, *, attn_mask=None, **kwargs):
            del kwargs
            captured["attn_mask"] = attn_mask
            return a, s, z

    transformer = DiffusionTransformer(c_a=4, c_s=4, c_z=2, n_blocks=1, n_heads=1)
    transformer.blocks = torch.nn.ModuleList([CaptureBlock()])
    transformer.blocks_per_ckpt = None
    attn_mask = torch.tensor(
        [
            [True, True, False],
            [True, True, False],
            [False, False, False],
        ]
    )

    transformer(
        a=torch.zeros(3, 4),
        s=torch.zeros(3, 4),
        z=torch.zeros(3, 3, 2),
        attn_mask=attn_mask,
    )

    assert torch.equal(captured["attn_mask"], attn_mask)


def test_valid_resolution_mask_vectorizes_per_example() -> None:
    resolution = torch.tensor([1.5, 9.0, 2.5])

    valid_resolution = _valid_resolution_mask(
        resolution,
        min_resolution=1.0,
        max_resolution=4.0,
    )

    assert valid_resolution.device == resolution.device
    assert valid_resolution.dtype == torch.float32
    assert valid_resolution.tolist() == [1.0, 0.0, 1.0]


def test_apply_resolution_gate_averages_only_valid_examples() -> None:
    gated_loss = _apply_resolution_gate(
        torch.tensor([2.0, 10.0, 4.0]),
        torch.tensor([1.0, 0.0, 1.0]),
    )

    assert gated_loss.shape == torch.Size([])
    assert torch.equal(gated_loss, torch.tensor(3.0))
    assert torch.equal(
        _apply_resolution_gate(torch.tensor(5.0), torch.tensor([0.0])),
        torch.tensor(0.0),
    )


def test_apply_last_dim_mask_preserves_batched_ragged_atom_axis() -> None:
    values = torch.arange(2 * 2 * 4, dtype=torch.float32).reshape(2, 2, 4)
    ragged_atom_mask = torch.tensor(
        [
            [True, False, True, False],
            [False, True, True, True],
        ]
    )

    masked = _apply_last_dim_mask(
        values,
        ragged_atom_mask,
        name="ragged_atom_mask",
    )

    assert masked.shape == values.shape
    assert masked.tolist() == [
        [
            [0.0, 0.0, 2.0, 0.0],
            [4.0, 0.0, 6.0, 0.0],
        ],
        [
            [0.0, 9.0, 10.0, 11.0],
            [0.0, 13.0, 14.0, 15.0],
        ],
    ]


def test_apply_last_dim_mask_rejects_atom_axis_mismatch() -> None:
    with pytest.raises(ValueError, match="last dimension"):
        _apply_last_dim_mask(
            torch.zeros(2, 3),
            torch.ones(2, 4, dtype=torch.bool),
            name="atom_mask",
        )


def test_bond_type_loss_expands_legacy_unbatched_labels_and_masks() -> None:
    loss_fn = BondTypeLoss(num_classes=2, eps=0.0, reduction=None)
    logits = torch.zeros(2, 2, 2, 2)
    bond_labels = torch.tensor(
        [
            [0, 1],
            [1, 0],
        ]
    )
    bond_gen_flag = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    loss = loss_fn(logits, bond_labels, bond_gen_flag)

    assert loss.shape == (2,)
    assert torch.allclose(loss, torch.full((2,), math.log(2.0)))


def test_bond_type_loss_accepts_batched_labels_and_masks() -> None:
    loss_fn = BondTypeLoss(num_classes=2, eps=0.0, reduction=None)
    logits = torch.zeros(2, 1, 1, 2)
    bond_labels = torch.tensor([[[0]], [[1]]])
    bond_gen_flag = torch.ones(2, 1, 1)

    loss = loss_fn(logits, bond_labels, bond_gen_flag)

    assert loss.shape == (2,)
    assert torch.allclose(loss, torch.full((2,), math.log(2.0)))


def test_bond_type_loss_rejects_mismatched_batched_labels() -> None:
    loss_fn = BondTypeLoss(num_classes=2, reduction=None)

    with pytest.raises(ValueError, match="batch size"):
        loss_fn(
            torch.zeros(2, 1, 1, 2),
            torch.zeros(3, 1, 1, dtype=torch.long),
            torch.ones(2, 1, 1),
        )


def test_bond_type_loss_rejects_mismatched_batched_masks() -> None:
    loss_fn = BondTypeLoss(num_classes=2, reduction=None)

    with pytest.raises(ValueError, match="batch size"):
        loss_fn(
            torch.zeros(2, 1, 1, 2),
            torch.zeros(2, 1, 1, dtype=torch.long),
            torch.ones(3, 1, 1),
        )


def test_sparse_bond_loss_preserves_batched_pair_indices_and_scale() -> None:
    loss_fn = BondLoss(reduction=None)
    true_coordinate = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
        ]
    )
    pred_coordinate = torch.tensor(
        [
            [
                [
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ],
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                ]
            ],
        ]
    )
    bond_mask = torch.zeros(2, 3, 3)
    bond_mask[0, 0, 1] = 1.0
    bond_mask[1, 1, 2] = 1.0

    loss = loss_fn.sparse_forward(
        pred_coordinate=pred_coordinate,
        true_coordinate=true_coordinate,
        distance_mask=torch.ones_like(bond_mask),
        bond_mask=bond_mask,
        per_sample_scale=torch.tensor([[2.0], [3.0]]),
    )

    assert loss.shape == (2,)
    assert torch.allclose(loss, torch.tensor([2.0, 27.0]))


def test_sparse_bond_loss_reduces_batched_prefix_with_mean_reduction() -> None:
    loss_fn = BondLoss(reduction="mean")
    true_coordinate = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
        ]
    )
    pred_coordinate = torch.tensor(
        [
            [
                [
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ],
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                ]
            ],
        ]
    )
    bond_mask = torch.zeros(2, 3, 3)
    bond_mask[0, 0, 1] = 1.0
    bond_mask[1, 1, 2] = 1.0

    loss = loss_fn.sparse_forward(
        pred_coordinate=pred_coordinate,
        true_coordinate=true_coordinate,
        distance_mask=torch.ones_like(bond_mask),
        bond_mask=bond_mask,
        per_sample_scale=torch.tensor([[2.0], [3.0]]),
    )

    assert loss.shape == torch.Size([])
    assert torch.allclose(loss, torch.tensor(14.5))


def test_sparse_bond_loss_returns_batched_differentiable_zero_without_bonds() -> None:
    loss_fn = BondLoss(reduction=None)
    pred_coordinate = torch.zeros(2, 1, 3, 3, requires_grad=True)

    loss = loss_fn.sparse_forward(
        pred_coordinate=pred_coordinate,
        true_coordinate=torch.zeros(2, 3, 3),
        distance_mask=torch.zeros(2, 3, 3),
        bond_mask=torch.zeros(2, 3, 3),
    )

    assert loss.shape == (2,)
    assert loss.requires_grad
    assert torch.equal(loss, torch.zeros(2))


def test_sparse_smooth_lddt_loss_preserves_batched_pair_indices() -> None:
    loss_fn = SmoothLDDTLoss(reduction=None)
    true_coordinate = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
        ]
    )
    pred_coordinate = torch.tensor(
        [
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.5, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ],
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [4.5, 0.0, 0.0],
                ]
            ],
        ]
    )
    lddt_mask = torch.zeros(2, 3, 3, dtype=torch.bool)
    lddt_mask[0, 0, 1] = True
    lddt_mask[1, 1, 2] = True

    batched_loss = loss_fn.sparse_forward(
        pred_coordinate=pred_coordinate,
        true_coordinate=true_coordinate,
        lddt_mask=lddt_mask,
    )
    expected_loss = torch.stack(
        [
            loss_fn.sparse_forward(
                pred_coordinate=pred_coordinate[i],
                true_coordinate=true_coordinate[i],
                lddt_mask=lddt_mask[i],
            )
            for i in range(2)
        ]
    )

    assert batched_loss.shape == (2,)
    assert torch.allclose(batched_loss, expected_loss)


def test_sparse_smooth_lddt_loss_reduces_batched_prefix_with_mean_reduction() -> None:
    per_example_loss_fn = SmoothLDDTLoss(reduction=None)
    mean_loss_fn = SmoothLDDTLoss(reduction="mean")
    true_coordinate = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
        ]
    )
    pred_coordinate = torch.tensor(
        [
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.5, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ],
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [4.5, 0.0, 0.0],
                ]
            ],
        ]
    )
    lddt_mask = torch.zeros(2, 3, 3, dtype=torch.bool)
    lddt_mask[0, 0, 1] = True
    lddt_mask[1, 1, 2] = True

    per_example_loss = per_example_loss_fn.sparse_forward(
        pred_coordinate=pred_coordinate,
        true_coordinate=true_coordinate,
        lddt_mask=lddt_mask,
    )
    mean_loss = mean_loss_fn.sparse_forward(
        pred_coordinate=pred_coordinate,
        true_coordinate=true_coordinate,
        lddt_mask=lddt_mask,
    )

    assert mean_loss.shape == torch.Size([])
    assert torch.allclose(mean_loss, per_example_loss.mean())


def test_sparse_smooth_lddt_chunk_slices_sample_axis_with_shared_mask() -> None:
    loss_fn = SmoothLDDTLoss(reduction=None)
    true_coordinate = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    pred_coordinate = torch.tensor(
        [
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.5, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0],
                    [0.5, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                ],
            ],
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.5, 0.0, 0.0],
                ],
            ],
        ]
    )
    lddt_mask = torch.zeros(3, 3, dtype=torch.bool)
    lddt_mask[0, 1] = True
    lddt_mask[1, 2] = True

    unchunked_loss = loss_fn.sparse_forward(
        pred_coordinate=pred_coordinate,
        true_coordinate=true_coordinate,
        lddt_mask=lddt_mask,
    )
    chunked_loss = loss_fn.sparse_forward(
        pred_coordinate=pred_coordinate,
        true_coordinate=true_coordinate,
        lddt_mask=lddt_mask,
        diffusion_chunk_size=2,
    )

    assert chunked_loss.shape == (2,)
    assert torch.allclose(chunked_loss, unchunked_loss)


def test_sparse_smooth_lddt_empty_mask_returns_differentiable_zero() -> None:
    loss_fn = SmoothLDDTLoss(reduction=None)
    pred_coordinate = torch.zeros(2, 3, 4, 3, requires_grad=True)

    loss = loss_fn.sparse_forward(
        pred_coordinate=pred_coordinate,
        true_coordinate=torch.zeros(2, 4, 3),
        lddt_mask=torch.zeros(4, 4, dtype=torch.bool),
    )

    assert loss.shape == (2,)
    assert loss.requires_grad
    assert torch.equal(loss, torch.zeros(2))


def test_dense_smooth_lddt_loss_preserves_batched_diffusion_sample_axis() -> None:
    loss_fn = SmoothLDDTLoss(reduction=None)
    true_coordinate = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
        ]
    )
    pred_coordinate = torch.tensor(
        [
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.5, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0],
                    [0.5, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
            ],
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [4.5, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.5, 0.0, 0.0],
                ],
            ],
        ]
    )
    lddt_mask = torch.zeros(2, 3, 3, dtype=torch.bool)
    lddt_mask[0, 0, 1] = True
    lddt_mask[1, 1, 2] = True

    batched_loss = loss_fn.dense_forward(
        pred_coordinate=pred_coordinate,
        true_coordinate=true_coordinate,
        lddt_mask=lddt_mask,
        diffusion_chunk_size=2,
    )
    expected_loss = torch.stack(
        [
            loss_fn.dense_forward(
                pred_coordinate=pred_coordinate[i],
                true_coordinate=true_coordinate[i],
                lddt_mask=lddt_mask[i],
                diffusion_chunk_size=2,
            )
            for i in range(2)
        ]
    )

    assert batched_loss.shape == (2,)
    assert torch.allclose(batched_loss, expected_loss)


def test_dense_smooth_lddt_chunk_supports_real_training_sample_count() -> None:
    loss_fn = SmoothLDDTLoss(reduction=None)
    true_coordinate = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 0.0, 2.0],
            ],
        ]
    )
    pred_coordinate = true_coordinate[:, None, :, :].expand(2, 48, 3, 3).clone()
    sample_offsets = torch.linspace(0.0, 0.47, steps=48).view(1, 48, 1, 1)
    pred_coordinate = pred_coordinate + sample_offsets
    pred_coordinate[0, :, 1, 0] += torch.linspace(0.0, 0.3, steps=48)
    pred_coordinate[1, :, 2, 2] -= torch.linspace(0.0, 0.5, steps=48)

    lddt_mask = torch.zeros(2, 3, 3, dtype=torch.bool)
    lddt_mask[0, 0, 1] = True
    lddt_mask[0, 1, 2] = True
    lddt_mask[1, 0, 2] = True
    lddt_mask[1, 1, 2] = True

    unchunked_loss = loss_fn.dense_forward(
        pred_coordinate=pred_coordinate,
        true_coordinate=true_coordinate,
        lddt_mask=lddt_mask,
    )
    chunked_loss = loss_fn.dense_forward(
        pred_coordinate=pred_coordinate,
        true_coordinate=true_coordinate,
        lddt_mask=lddt_mask,
        diffusion_chunk_size=48,
    )

    assert chunked_loss.shape == (2,)
    assert torch.allclose(chunked_loss, unchunked_loss)


def test_diffusion_condition_align_mask_threshold_is_per_example_and_padding_aware() -> None:
    is_condition_atom = torch.tensor(
        [
            [True, True, False, False],
            [False, False, False, False],
        ]
    )
    atom_padding_mask = torch.tensor(
        [
            [False, False, False, True],
            [False, False, False, True],
        ]
    )

    align_mask = _diffusion_condition_align_mask(
        is_condition_atom=is_condition_atom,
        atom_padding_mask=atom_padding_mask,
        threshold=0.3,
    )

    assert align_mask.tolist() == [
        [True, True, False, False],
        [True, True, True, True],
    ]


def test_mse_weighted_rigid_align_uses_align_mask_for_alignment_weights() -> None:
    loss_fn = MSELoss(reduction=None)
    true_coordinate = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [50.0, 50.0, 50.0],
        ]
    )
    translation = torch.tensor([10.0, 20.0, -3.0])
    pred_coordinate = true_coordinate.clone()
    pred_coordinate[:3] = true_coordinate[:3] + translation
    pred_coordinate[3] = torch.tensor([-100.0, 25.0, 80.0])
    pred_coordinate = pred_coordinate.unsqueeze(dim=0)

    aligned, _ = loss_fn.weighted_rigid_align(
        pred_coordinate=pred_coordinate,
        true_coordinate=true_coordinate,
        coordinate_mask=torch.ones(4),
        is_dna=torch.zeros(4),
        is_rna=torch.zeros(4),
        is_ligand=torch.zeros(4),
        align_mask=torch.tensor([True, True, True, False]),
    )

    assert torch.allclose(
        aligned[0, :3],
        pred_coordinate[0, :3],
        atol=1e-4,
        rtol=1e-4,
    )


def test_distogram_loss_accepts_batched_ragged_rep_atom_mask() -> None:
    loss_fn = DistogramLoss(no_bins=4, reduction=None)
    logits = torch.zeros(2, 3, 3, 4)
    true_coordinate = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [9.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        ]
    )
    coordinate_mask = torch.tensor(
        [
            [True, True, True, False],
            [True, True, True, False],
        ]
    )
    rep_atom_mask = torch.tensor(
        [
            [True, False, True, False],
            [True, True, True, False],
        ]
    )

    loss = loss_fn(
        logits=logits,
        true_coordinate=true_coordinate,
        coordinate_mask=coordinate_mask,
        rep_atom_mask=rep_atom_mask,
    )

    assert loss.shape == (2,)
    assert torch.allclose(loss, torch.full((2,), math.log(4.0)))
