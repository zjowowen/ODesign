import torch
from unittest import mock

from src.model.modules.pairformer import (
    MSAStack,
    _add_single_embedding_to_msa,
    _apply_msa_token_mask,
    _assert_msa_padding_roundtrip,
    _chunk_msa_rows,
    _slice_msa_rows,
)


def test_apply_msa_token_mask_broadcasts_batched_token_mask_over_msa_axis() -> None:
    msa = torch.ones(2, 3, 4, 5)
    mask = torch.tensor(
        [
            [False, True, False, True],
            [True, False, False, False],
        ]
    )

    result = _apply_msa_token_mask(msa.clone(), mask)

    expected = msa.clone()
    expected[0, :, [1, 3], :] = 0
    expected[1, :, [0], :] = 0
    assert torch.equal(result, expected)


def test_add_single_embedding_to_msa_inserts_msa_axis_for_batched_inputs() -> None:
    msa = torch.zeros(2, 3, 4, 5)
    single = torch.arange(2 * 4 * 5, dtype=torch.float32).reshape(2, 4, 5)

    result = _add_single_embedding_to_msa(msa, single)

    expected = single.unsqueeze(-3).expand_as(msa)
    assert torch.equal(result, expected)


def test_slice_msa_rows_preserves_batch_axis() -> None:
    msa = torch.arange(2 * 5 * 3 * 4).reshape(2, 5, 3, 4)

    result = _slice_msa_rows(msa, 3)

    assert torch.equal(result, msa[:, :3, :, :])


def test_chunk_msa_rows_splits_negative_three_axis() -> None:
    msa = torch.arange(2 * 5 * 3 * 4).reshape(2, 5, 3, 4)

    chunks = _chunk_msa_rows(msa, 2)

    assert [tuple(chunk.shape) for chunk in chunks] == [
        (2, 2, 3, 4),
        (2, 2, 3, 4),
        (2, 1, 3, 4),
    ]
    assert torch.equal(torch.cat(chunks, dim=-3), msa)


def test_msa_stack_inference_forward_chunks_msa_axis_with_batch_prefix() -> None:
    pair_weighted_shapes: list[tuple[int, ...]] = []
    transition_shapes: list[tuple[int, ...]] = []

    class PairWeighted(torch.nn.Module):
        def forward(self, m: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
            pair_weighted_shapes.append(tuple(m.shape))
            return torch.ones_like(m)

    class Transition(torch.nn.Module):
        def forward(self, m: torch.Tensor) -> torch.Tensor:
            transition_shapes.append(tuple(m.shape))
            return torch.full_like(m, 2)

    stack = MSAStack(msa_chunk_size=2)
    stack.msa_pair_weighted_averaging = PairWeighted()
    stack.transition_m = Transition()
    stack.eval()

    msa = torch.zeros(2, 5, 3, 4)
    pair = torch.zeros(2, 3, 3, 4)

    result = stack.inference_forward(msa, pair, chunk_size=2)

    assert result is msa
    assert torch.equal(result, torch.full_like(msa, 3))
    expected_chunk_shapes = [(2, 2, 3, 4), (2, 2, 3, 4), (2, 1, 3, 4)]
    assert pair_weighted_shapes == expected_chunk_shapes
    assert transition_shapes == expected_chunk_shapes


def test_msa_stack_train_forward_skips_padding_roundtrip_check_by_default() -> None:
    stack = MSAStack(c_m=4, msa_chunk_size=2, msa_max_size=6)
    stack.dropout_row = torch.nn.Identity()
    stack.train()

    def fake_chunk_forward(
        module: torch.nn.Module,
        m: torch.Tensor,
        z,
        chunk_size: int = 2048,
    ) -> torch.Tensor:
        return torch.zeros_like(m)

    stack.chunk_forward = fake_chunk_forward

    msa = torch.randn(2, 3, 5, 4)
    pair = torch.randn(2, 5, 5, 4)

    with mock.patch.object(
        torch.Tensor,
        "all",
        autospec=True,
        side_effect=RuntimeError("unexpected tensor all"),
    ):
        result = stack(msa, pair)

    assert torch.equal(result, msa)


def test_assert_msa_padding_roundtrip_detects_mismatched_real_rows() -> None:
    original = torch.ones(2, 3, 5, 4)
    padded = torch.zeros(2, 6, 5, 4)

    try:
        _assert_msa_padding_roundtrip(padded, original)
    except AssertionError as exc:
        assert "MSA padding roundtrip" in str(exc)
    else:
        raise AssertionError("expected MSA padding roundtrip check to fail")
