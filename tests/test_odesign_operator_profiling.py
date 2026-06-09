import unittest
from unittest import mock

import torch
import torch.nn as nn

from src.utils.model.profiling import (
    PROFILE_DETAIL_ATTR,
    odesign_record_function,
    set_odesign_profile_detail_enabled,
)

try:
    from src.model.modules.pairformer import PairformerBlock
except ModuleNotFoundError as exc:
    PairformerBlock = None
    PAIRFORMER_IMPORT_ERROR = exc
else:
    PAIRFORMER_IMPORT_ERROR = None


class _Recorder:
    def __init__(self, name: str, calls: list[str]) -> None:
        self.name = name
        self.calls = calls

    def __enter__(self):
        self.calls.append(self.name)
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class ODesignOperatorProfilingTests(unittest.TestCase):
    def test_profile_detail_flag_is_recursive(self) -> None:
        module = nn.Sequential(nn.Linear(2, 2), nn.Sequential(nn.ReLU()))

        set_odesign_profile_detail_enabled(module, True)

        for child in module.modules():
            self.assertIs(getattr(child, PROFILE_DETAIL_ATTR), True)

        set_odesign_profile_detail_enabled(module, False)

        for child in module.modules():
            self.assertIs(getattr(child, PROFILE_DETAIL_ATTR), False)

    def test_record_function_is_noop_when_detail_disabled(self) -> None:
        module = nn.Linear(2, 2)

        with mock.patch.object(torch.autograd.profiler, "record_function") as record_fn:
            with odesign_record_function(module, "odesign.test.disabled"):
                pass

        record_fn.assert_not_called()

    def test_record_function_is_used_when_detail_enabled(self) -> None:
        module = nn.Linear(2, 2)
        set_odesign_profile_detail_enabled(module, True)
        calls = []

        with mock.patch.object(
            torch.autograd.profiler,
            "record_function",
            side_effect=lambda name: _Recorder(name, calls),
        ):
            with odesign_record_function(module, "odesign.test.enabled"):
                pass

        self.assertEqual(calls, ["odesign.test.enabled"])

    @unittest.skipIf(
        PairformerBlock is None,
        f"PairformerBlock dependencies unavailable: {PAIRFORMER_IMPORT_ERROR}",
    )
    def test_pairformer_block_emits_detail_ranges(self) -> None:
        block = PairformerBlock(
            n_heads=1,
            c_z=2,
            c_s=4,
            c_hidden_mul=2,
            c_hidden_pair_att=2,
            no_heads_pair=1,
            dropout=0.0,
        )
        block.eval()
        set_odesign_profile_detail_enabled(block, True)
        s = torch.randn(1, 3, 4)
        z = torch.randn(1, 3, 3, 2)
        pair_mask = torch.ones(1, 3, 3)
        calls = []

        with mock.patch.object(
            torch.autograd.profiler,
            "record_function",
            side_effect=lambda name: _Recorder(name, calls),
        ):
            out_s, out_z = block(
                s,
                z,
                pair_mask,
                use_deepspeed_evo_attention=False,
            )

        self.assertEqual(out_s.shape, s.shape)
        self.assertEqual(out_z.shape, z.shape)
        self.assertIn("odesign.pairformer_block.tri_mul_out", calls)
        self.assertIn("odesign.pairformer_block.tri_mul_in", calls)
        self.assertIn("odesign.pairformer_block.tri_att_start", calls)
        self.assertIn("odesign.pairformer_block.tri_att_end", calls)
        self.assertIn("odesign.pairformer_block.pair_transition", calls)
        self.assertIn("odesign.pairformer_block.attention_pair_bias", calls)
        self.assertIn("odesign.pairformer_block.single_transition", calls)
        self.assertIn("odesign.openfold_attention.stock", calls)


if __name__ == "__main__":
    unittest.main()
