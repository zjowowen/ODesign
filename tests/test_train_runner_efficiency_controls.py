import unittest
from unittest import mock

import torch

from src.utils.train.train_runner import TrainRunner


class TrainRunnerEfficiencyControlTests(unittest.TestCase):
    def test_empty_cache_policy_defaults_to_microbatch(self) -> None:
        self.assertEqual(TrainRunner._normalize_empty_cache_policy(None), "microbatch")
        self.assertEqual(TrainRunner._normalize_empty_cache_policy(""), "microbatch")

    def test_empty_cache_policy_controls_when_cache_is_cleared(self) -> None:
        self.assertIs(TrainRunner._should_empty_cache_for_policy("microbatch", False), True)
        self.assertIs(TrainRunner._should_empty_cache_for_policy("microbatch", True), True)
        self.assertIs(
            TrainRunner._should_empty_cache_for_policy("optimizer_update", False), False
        )
        self.assertIs(
            TrainRunner._should_empty_cache_for_policy("optimizer_update", True), True
        )
        self.assertIs(TrainRunner._should_empty_cache_for_policy("never", False), False)
        self.assertIs(TrainRunner._should_empty_cache_for_policy("never", True), False)

    def test_empty_cache_policy_rejects_invalid_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "ODESIGN_EMPTY_CACHE_POLICY"):
            TrainRunner._normalize_empty_cache_policy("sometimes")

    def test_cuda_stage_profile_disabled_without_profile_record(self) -> None:
        runner = mock.Mock()
        runner.profile_enabled = True
        runner.profile_stage_cuda_peaks = True
        runner.device = torch.device("cuda:0")
        self.assertIs(
            TrainRunner._profile_cuda_stage_enabled(runner, profile_record=None), False
        )

    def test_cuda_stage_profile_records_peak_fields(self) -> None:
        runner = mock.Mock()
        runner.profile_enabled = True
        runner.profile_stage_cuda_peaks = True
        runner.device = torch.device("cuda:0")
        profile_record = {}

        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "synchronize") as synchronize,
            mock.patch.object(torch.cuda, "reset_peak_memory_stats") as reset_peak,
            mock.patch.object(torch.cuda, "memory_allocated", side_effect=[100, 300]),
            mock.patch.object(torch.cuda, "memory_reserved", side_effect=[200, 400]),
            mock.patch.object(torch.cuda, "max_memory_allocated", return_value=500),
            mock.patch.object(torch.cuda, "max_memory_reserved", return_value=600),
        ):
            TrainRunner._profile_cuda_stage_begin(runner, profile_record, "forward")
            TrainRunner._profile_cuda_stage_end(runner, profile_record, "forward")

        self.assertEqual(synchronize.call_count, 2)
        reset_peak.assert_called_once_with(runner.device)
        self.assertEqual(profile_record["forward_gpu_mem_start_allocated_mib"], 100 / 1024**2)
        self.assertEqual(profile_record["forward_gpu_mem_start_reserved_mib"], 200 / 1024**2)
        self.assertEqual(profile_record["forward_gpu_mem_end_allocated_mib"], 300 / 1024**2)
        self.assertEqual(profile_record["forward_gpu_mem_end_reserved_mib"], 400 / 1024**2)
        self.assertEqual(profile_record["forward_gpu_mem_peak_allocated_mib"], 500 / 1024**2)
        self.assertEqual(profile_record["forward_gpu_mem_peak_reserved_mib"], 600 / 1024**2)

if __name__ == "__main__":
    unittest.main()
