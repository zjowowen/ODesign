import torch

from src.model.modules.pairformer import (
    _add_single_embedding_to_msa,
    _apply_msa_token_mask,
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
