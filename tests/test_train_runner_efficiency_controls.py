import unittest
import types
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

    def test_module_profile_setup_registers_backward_hooks(self) -> None:
        runner = mock.Mock()
        runner.profile_sync_cuda = True
        runner.profile_module_backward = True
        runner.device = torch.device("cuda:0")
        runner._profile_module_backward_handles = []
        runner._profile_module_backward_starts = {}
        runner._profile_now = mock.Mock(return_value=1.0)
        runner._profile_model = lambda: runner.model.module
        runner._profile_register_module_backward_hook = mock.Mock(
            wraps=TrainRunner._profile_register_module_backward_hook.__get__(
                runner, TrainRunner
            )
        )
        runner._profile_register_pairwise_backward_hooks = (
            TrainRunner._profile_register_pairwise_backward_hooks.__get__(
                runner, TrainRunner
            )
        )
        runner.model = types.SimpleNamespace(module=types.SimpleNamespace())
        hook_handle = mock.Mock()

        for module_name in [
            "pairformer_stack",
            "msa_module",
            "pairwise_head",
            "diffusion_module",
        ]:
            module = mock.Mock()
            module.register_full_backward_pre_hook.return_value = hook_handle
            module.register_full_backward_hook.return_value = hook_handle
            setattr(runner.model.module, module_name, module)
        runner.model.module.pairwise_head.distogram_head = mock.Mock()
        runner.model.module.pairwise_head.bond_type_head = mock.Mock()

        TrainRunner._profile_setup_model_module_profiling(runner)

        self.assertIs(runner.model.module._odesign_profile_enabled, True)
        self.assertIs(runner.model.module._odesign_profile_sync_cuda, True)
        self.assertEqual(runner.model.module._odesign_profile_device, runner.device)
        self.assertIsNone(runner.model.module._odesign_profile_record)
        self.assertFalse(hasattr(runner.model, "_odesign_profile_enabled"))
        self.assertEqual(len(runner._profile_module_backward_handles), 10)
        registered_names = [
            call.args[0]
            for call in runner._profile_register_module_backward_hook.call_args_list
        ]
        self.assertEqual(
            registered_names,
            [
                "pairformer",
                "msa",
                "pairwise_head",
                "pairwise_head",
                "diffusion_module",
            ],
        )

    def test_module_backward_hook_accumulates_elapsed_time(self) -> None:
        runner = mock.Mock()
        runner._profile_module_backward_handles = []
        runner._profile_module_backward_starts = {}
        runner._profile_now = mock.Mock(side_effect=[2.0, 5.5, 6.0, 7.25])
        record = {}
        runner._profile_get_model_record = mock.Mock(return_value=record)
        module = mock.Mock()
        callbacks = {}

        def register_pre_hook(callback):
            callbacks["pre"] = callback
            return mock.Mock()

        def register_hook(callback):
            callbacks["hook"] = callback
            return mock.Mock()

        module.register_full_backward_pre_hook.side_effect = register_pre_hook
        module.register_full_backward_hook.side_effect = register_hook

        TrainRunner._profile_register_module_backward_hook(
            runner, "pairformer", module
        )
        callbacks["pre"](module, ())
        callbacks["hook"](module, (), ())
        callbacks["pre"](module, ())
        callbacks["hook"](module, (), ())

        self.assertEqual(record["module_profile_backward_pairformer_sec"], 4.75)

    def test_profile_write_records_module_profile_flags(self) -> None:
        runner = mock.Mock()
        runner.profile_enabled = True
        runner.profile_jsonl_path = mock.Mock()
        runner.step = 2
        runner.global_step = 13
        runner.profile_stage_cuda_peaks = True
        runner.profile_modules = True
        runner.profile_module_backward = False
        runner.device = torch.device("cpu")
        mocked_open = mock.mock_open()
        runner.profile_jsonl_path.open = mocked_open

        TrainRunner._profile_write(runner, {"loss": 1.25})

        handle = mocked_open()
        handle.write.assert_called_once()
        written = handle.write.call_args.args[0]
        self.assertIn('"profile_modules": true', written)
        self.assertIn('"profile_module_backward": false', written)
        self.assertIn('"profile_stage_cuda_peaks": true', written)

if __name__ == "__main__":
    unittest.main()
