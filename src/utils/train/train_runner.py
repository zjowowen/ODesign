import os
from pathlib import Path
import logging
import json
import time

import wandb
from ml_collections.config_dict import ConfigDict
from contextlib import nullcontext
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from src.api.model_interface import LossInput, ODesignOutput, GroundTruth
from src.model.odesign import ODesign
from src.model.modules.loss import ODesignLoss
from src.utils.inference.dumper import DataDumper
from src.utils.model.misc import is_loss_nan_check
from src.utils.train.distributed import DIST_WRAPPER
from src.utils.train.lr_scheduler import get_lr_scheduler
from src.utils.train.metrics import SimpleMetricAggregator
from src.utils.model.torch_utils import (
    autocasting_disable_decorator,
    to_device,
    filter_state_dict,
)
from src.utils.model.profiling import set_odesign_profile_detail_enabled
from src.utils.permutation.permutation import SymmetricPermutation

logger = logging.getLogger(__name__)


class TrainRunner(object):
    def __init__(
        self,
        configs: ConfigDict,
        ckpt_dir: str | Path,
        error_dir: str | Path,
        eval_dump_dir: str | Path,
        device: torch.device,
        train_dl: DataLoader,
        test_dls: dict[str, DataLoader],
        model: ODesign,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        loss: ODesignLoss,
        symmetric_permutation: SymmetricPermutation,
        train_metric_wrapper: SimpleMetricAggregator,
    ) -> None:
        
        self.configs = configs
        self.ckpt_dir = ckpt_dir
        self.error_dir = error_dir
        self.eval_dump_dir = eval_dump_dir
        self.device = device
        self.train_dl = train_dl
        self.test_dls = test_dls
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss = loss
        self.symmetric_permutation = symmetric_permutation
        self.train_metric_wrapper = train_metric_wrapper

        # Step means effective step considering accumulation
        self.step = 0
        # Global_step equals to self.step * self.iters_to_accumulate
        self.global_step = 0
        self.start_step = 0
        # Add for grad accumulation, it can increase real batch size
        self.iters_to_accumulate = self.configs.iters_to_accumulate    

        self.profile_jsonl = os.environ.get("ODESIGN_PROFILE_JSONL", "")
        self.profile_all_ranks = os.environ.get("ODESIGN_PROFILE_ALL_RANKS", "0") == "1"
        self.profile_enabled = bool(self.profile_jsonl) and (
            self.profile_all_ranks or DIST_WRAPPER.rank == 0
        )
        self.profile_sync_cuda = os.environ.get("ODESIGN_PROFILE_SYNC_CUDA", "1") != "0"
        self.profile_stage_cuda_peaks = (
            os.environ.get("ODESIGN_PROFILE_STAGE_CUDA_PEAKS", "0") == "1"
        )
        self.profile_modules = os.environ.get("ODESIGN_PROFILE_MODULES", "0") == "1"
        self.profile_module_backward = (
            os.environ.get("ODESIGN_PROFILE_MODULE_BACKWARD", "1") != "0"
        )
        self.profile_pairformer_detail = (
            os.environ.get("ODESIGN_PROFILE_PAIRFORMER_DETAIL", "0") == "1"
        )
        self.torch_profiler_dir = os.environ.get("ODESIGN_TORCH_PROFILER_DIR", "")
        self.torch_profiler_all_ranks = (
            os.environ.get("ODESIGN_TORCH_PROFILER_ALL_RANKS", "0") == "1"
        )
        self.torch_profiler_enabled = bool(self.torch_profiler_dir) and (
            self.torch_profiler_all_ranks or DIST_WRAPPER.rank == 0
        )
        self.torch_profiler_wait = int(
            os.environ.get("ODESIGN_TORCH_PROFILER_WAIT", "1")
        )
        self.torch_profiler_warmup = int(
            os.environ.get("ODESIGN_TORCH_PROFILER_WARMUP", "1")
        )
        self.torch_profiler_active = int(
            os.environ.get("ODESIGN_TORCH_PROFILER_ACTIVE", "2")
        )
        self.torch_profiler_repeat = int(
            os.environ.get("ODESIGN_TORCH_PROFILER_REPEAT", "1")
        )
        self._torch_profiler = None
        self.empty_cache_policy = self._normalize_empty_cache_policy(
            os.environ.get("ODESIGN_EMPTY_CACHE_POLICY")
        )
        self.profile_jsonl_path = None
        self._profile_last_iter_end = time.perf_counter()
        self._profile_module_backward_handles = []
        self._profile_module_backward_starts = {}
        if self.profile_enabled:
            base_path = Path(self.profile_jsonl)
            if self.profile_all_ranks:
                suffix = base_path.suffix or ".jsonl"
                stem = base_path.stem if base_path.suffix else base_path.name
                base_path = base_path.with_name(
                    f"{stem}_rank{DIST_WRAPPER.rank:02d}{suffix}"
                )
            base_path.parent.mkdir(parents=True, exist_ok=True)
            self.profile_jsonl_path = base_path
        if self.profile_enabled and self.profile_modules:
            self._profile_setup_model_module_profiling()
        if self.profile_pairformer_detail:
            self._profile_setup_pairformer_detail_profiling()

        self.load_checkpoint()

    @staticmethod
    def _normalize_empty_cache_policy(value: str | None) -> str:
        raw_value = (value or "").strip().lower()
        if raw_value in ("", "1", "true", "yes", "always", "microbatch"):
            return "microbatch"
        if raw_value in ("optimizer", "optimizer_step", "optimizer_update"):
            return "optimizer_update"
        if raw_value in ("0", "false", "no", "off", "never", "disabled"):
            return "never"
        raise ValueError(
            "ODESIGN_EMPTY_CACHE_POLICY must be one of "
            "microbatch, optimizer_update, or never; got "
            f"{value!r}"
        )

    @staticmethod
    def _should_empty_cache_for_policy(
        policy: str, optimizer_update: bool
    ) -> bool:
        if policy == "microbatch":
            return True
        if policy == "optimizer_update":
            return optimizer_update
        if policy == "never":
            return False
        raise ValueError(f"Unknown empty_cache policy: {policy!r}")

    def _profile_now(self) -> float:
        if (
            self.profile_sync_cuda
            and torch.cuda.is_available()
            and self.device.type == "cuda"
        ):
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def _profile_model(self) -> torch.nn.Module:
        return getattr(self.model, "module", self.model)

    def _profile_set_model_record(self, profile_record: dict | None) -> None:
        self._profile_model()._odesign_profile_record = profile_record

    def _profile_get_model_record(self) -> dict | None:
        return getattr(self._profile_model(), "_odesign_profile_record", None)

    def _profile_setup_model_module_profiling(self) -> None:
        profile_model = self._profile_model()
        profile_model._odesign_profile_enabled = True
        profile_model._odesign_profile_sync_cuda = self.profile_sync_cuda
        profile_model._odesign_profile_device = self.device
        profile_model._odesign_profile_record = None
        if not self.profile_module_backward:
            return
        self._profile_register_module_backward_hook(
            "pairformer", profile_model.pairformer_stack
        )
        self._profile_register_module_backward_hook("msa", profile_model.msa_module)
        self._profile_register_pairwise_backward_hooks(profile_model.pairwise_head)
        self._profile_register_module_backward_hook(
            "diffusion_module", profile_model.diffusion_module
        )

    def _profile_setup_pairformer_detail_profiling(self) -> None:
        set_odesign_profile_detail_enabled(
            self._profile_model(), self.profile_pairformer_detail
        )

    @staticmethod
    def _profile_env_bool(name: str, default: bool) -> bool:
        raw_value = os.environ.get(name)
        if raw_value is None:
            return default
        return raw_value.strip().lower() in ("1", "true", "yes", "on")

    def _profile_torch_profiler_context(self):
        if not self.torch_profiler_enabled:
            return nullcontext()

        trace_dir = Path(self.torch_profiler_dir) / f"rank{DIST_WRAPPER.rank:02d}"
        trace_dir.mkdir(parents=True, exist_ok=True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available() and self.device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        return torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=self.torch_profiler_wait,
                warmup=self.torch_profiler_warmup,
                active=self.torch_profiler_active,
                repeat=self.torch_profiler_repeat,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
            record_shapes=self._profile_env_bool(
                "ODESIGN_TORCH_PROFILER_RECORD_SHAPES", True
            ),
            profile_memory=self._profile_env_bool(
                "ODESIGN_TORCH_PROFILER_PROFILE_MEMORY", True
            ),
            with_stack=self._profile_env_bool(
                "ODESIGN_TORCH_PROFILER_WITH_STACK", False
            ),
            with_modules=self._profile_env_bool(
                "ODESIGN_TORCH_PROFILER_WITH_MODULES", True
            ),
        )

    def _profile_torch_profiler_step(self) -> None:
        if self._torch_profiler is not None:
            self._torch_profiler.step()

    def _profile_register_pairwise_backward_hooks(
        self, pairwise_head: torch.nn.Module
    ) -> None:
        child_modules = [
            getattr(pairwise_head, "distogram_head", None),
            getattr(pairwise_head, "bond_type_head", None),
        ]
        registered_any = False
        for child_module in child_modules:
            if child_module is None:
                continue
            self._profile_register_module_backward_hook(
                "pairwise_head", child_module
            )
            registered_any = True
        if not registered_any:
            self._profile_register_module_backward_hook(
                "pairwise_head", pairwise_head
            )

    def _profile_register_module_backward_hook(
        self, module_name: str, module: torch.nn.Module
    ) -> None:
        def pre_hook(module, grad_output):
            record = self._profile_get_model_record()
            if record is None:
                return
            self._profile_module_backward_starts[module_name] = self._profile_now()

        def hook(module, grad_input, grad_output):
            record = self._profile_get_model_record()
            if record is None:
                return
            start = self._profile_module_backward_starts.pop(module_name, None)
            if start is None:
                return
            key = f"module_profile_backward_{module_name}_sec"
            record[key] = record.get(key, 0.0) + (self._profile_now() - start)

        self._profile_module_backward_handles.append(
            module.register_full_backward_pre_hook(pre_hook)
        )
        self._profile_module_backward_handles.append(
            module.register_full_backward_hook(hook)
        )

    def _profile_cuda_stage_enabled(self, profile_record: dict | None) -> bool:
        return (
            profile_record is not None
            and self.profile_enabled
            and self.profile_stage_cuda_peaks
            and torch.cuda.is_available()
            and self.device.type == "cuda"
        )

    def _profile_cuda_stage_begin(
        self, profile_record: dict | None, stage_name: str
    ) -> None:
        if not self._profile_cuda_stage_enabled(profile_record):
            return
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        profile_record[f"{stage_name}_gpu_mem_start_allocated_mib"] = (
            torch.cuda.memory_allocated(self.device) / 1024**2
        )
        profile_record[f"{stage_name}_gpu_mem_start_reserved_mib"] = (
            torch.cuda.memory_reserved(self.device) / 1024**2
        )

    def _profile_cuda_stage_end(
        self, profile_record: dict | None, stage_name: str
    ) -> None:
        if not self._profile_cuda_stage_enabled(profile_record):
            return
        torch.cuda.synchronize(self.device)
        profile_record[f"{stage_name}_gpu_mem_end_allocated_mib"] = (
            torch.cuda.memory_allocated(self.device) / 1024**2
        )
        profile_record[f"{stage_name}_gpu_mem_end_reserved_mib"] = (
            torch.cuda.memory_reserved(self.device) / 1024**2
        )
        profile_record[f"{stage_name}_gpu_mem_peak_allocated_mib"] = (
            torch.cuda.max_memory_allocated(self.device) / 1024**2
        )
        profile_record[f"{stage_name}_gpu_mem_peak_reserved_mib"] = (
            torch.cuda.max_memory_reserved(self.device) / 1024**2
        )

    @staticmethod
    def _profile_to_int(value):
        if value is None:
            return None
        if torch.is_tensor(value):
            if value.numel() == 0:
                return 0
            return int(value.detach().cpu().sum().item())
        if isinstance(value, (list, tuple)):
            total = 0
            for item in value:
                item_value = TrainRunner._profile_to_int(item)
                if item_value is not None:
                    total += item_value
            return total
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _profile_to_float(value):
        if value is None:
            return None
        if torch.is_tensor(value):
            if value.numel() == 0:
                return 0.0
            return float(value.detach().float().mean().cpu().item())
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _profile_batch_stats(self, batch: dict) -> dict:
        stats = {}
        basic = batch.get("basic", {}) if isinstance(batch, dict) else {}
        feature_data = batch.get("feature_data") if isinstance(batch, dict) else None

        n_token = self._profile_to_int(basic.get("N_token")) if basic else None
        n_atom = self._profile_to_int(basic.get("N_atom")) if basic else None
        if n_token is not None:
            stats["num_tokens_real"] = n_token
        if n_atom is not None:
            stats["num_atoms_real"] = n_atom

        token_padding_mask = getattr(feature_data, "token_padding_mask", None)
        if token_padding_mask is not None:
            stats["num_tokens_padded"] = int(token_padding_mask.numel())
            stats["num_tokens_real_from_mask"] = int(
                (~token_padding_mask.bool()).sum().item()
            )
        atom_padding_mask = getattr(feature_data, "atom_padding_mask", None)
        if atom_padding_mask is not None:
            stats["num_atoms_padded"] = int(atom_padding_mask.numel())
            stats["num_atoms_real_from_mask"] = int(
                (~atom_padding_mask.bool()).sum().item()
            )

        if "num_tokens_real" in stats and "num_tokens_padded" in stats:
            denom = max(stats["num_tokens_padded"], 1)
            stats["token_padding_ratio"] = 1.0 - stats["num_tokens_real"] / denom
        if "num_atoms_real" in stats and "num_atoms_padded" in stats:
            denom = max(stats["num_atoms_padded"], 1)
            stats["atom_padding_ratio"] = 1.0 - stats["num_atoms_real"] / denom

        pdb_id = basic.get("pdb_id") if basic else None
        if pdb_id is not None:
            stats["pdb_id"] = pdb_id
        return stats

    def _profile_write(self, record: dict):
        if not self.profile_enabled or self.profile_jsonl_path is None:
            return
        record = dict(record)
        record["rank"] = DIST_WRAPPER.rank
        record["world_size"] = DIST_WRAPPER.world_size
        record["step"] = int(self.step)
        record["global_step"] = int(self.global_step)
        record["timestamp"] = time.time()
        record["profile_stage_cuda_peaks"] = bool(self.profile_stage_cuda_peaks)
        record["profile_modules"] = bool(self.profile_modules)
        record["profile_module_backward"] = bool(
            self.profile_modules and self.profile_module_backward
        )
        record["profile_pairformer_detail"] = bool(self.profile_pairformer_detail)
        record["torch_profiler_enabled"] = bool(self.torch_profiler_enabled)
        if torch.cuda.is_available() and self.device.type == "cuda":
            record["gpu_mem_allocated_mib"] = (
                torch.cuda.memory_allocated(self.device) / 1024**2
            )
            record["gpu_mem_reserved_mib"] = (
                torch.cuda.memory_reserved(self.device) / 1024**2
            )
            stage_peak_allocated = max(
                (
                    value
                    for key, value in record.items()
                    if key.endswith("_gpu_mem_peak_allocated_mib")
                ),
                default=None,
            )
            stage_peak_reserved = max(
                (
                    value
                    for key, value in record.items()
                    if key.endswith("_gpu_mem_peak_reserved_mib")
                ),
                default=None,
            )
            record["gpu_mem_max_allocated_mib"] = (
                stage_peak_allocated
                if stage_peak_allocated is not None
                else torch.cuda.max_memory_allocated(self.device) / 1024**2
            )
            record["gpu_mem_max_reserved_mib"] = (
                stage_peak_reserved
                if stage_peak_reserved is not None
                else torch.cuda.max_memory_reserved(self.device) / 1024**2
            )
        with self.profile_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def load_checkpoint(self):

        def _load_checkpoint(
            checkpoint_path: str,
            load_params_only: bool,
            skip_load_optimizer: bool = False,
            skip_load_step: bool = False,
            skip_load_scheduler: bool = False,
            load_step_for_scheduler: bool = True,
        ):
            if not os.path.exists(checkpoint_path):
                raise Exception(f"Given checkpoint path not exist [{checkpoint_path}]")
            self.print(
                f"Loading from {checkpoint_path}, strict: {self.configs.load_strict}"
            )
            checkpoint = torch.load(checkpoint_path, self.device)

            sample_key = [k for k in checkpoint["model"].keys()][0]
            self.print(f"Sampled key: {sample_key}")
            if sample_key.startswith("module.") and not (DIST_WRAPPER.world_size > 1):
                # DDP checkpoint has module. prefix
                checkpoint["model"] = {
                    k[len("module.") :]: v for k, v in checkpoint["model"].items()
                }

            # Handle shape mismatches by filtering out parameters with different shapes
            filtered_state_dict = filter_state_dict(
                model_state_dict=self.model.state_dict(),
                ckpt_state_dict=checkpoint["model"],
            )
            
            self.model.load_state_dict(
                state_dict=filtered_state_dict,
                strict=self.configs.load_strict,
            )

            if not load_params_only:
                if not skip_load_optimizer:
                    self.print(f"Loading optimizer state")
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
                if not skip_load_step:
                    self.print(f"Loading checkpoint step")
                    self.step = checkpoint["step"] + 1
                    self.start_step = self.step
                    self.global_step = self.step * self.iters_to_accumulate
                if not skip_load_scheduler:
                    self.print(f"Loading scheduler state")
                    self.lr_scheduler.load_state_dict(checkpoint["scheduler"])
                elif load_step_for_scheduler:
                    assert (
                        not skip_load_step
                    ), "if load_step_for_scheduler is True, you must load step first"
                    # reinitialize LR scheduler using the updated optimizer and step
                    self.lr_scheduler = get_lr_scheduler(
                        self.configs.lr_scheduler,
                        self.optimizer,
                        last_epoch=self.step - 1,
                    )

            self.print(f"Finish loading checkpoint, current step: {self.step}")

        # Load model
        if self.configs.load_checkpoint_path:
            _load_checkpoint(
                self.configs.load_checkpoint_path,
                self.configs.load_params_only,
                skip_load_optimizer=self.configs.skip_load_optimizer,
                skip_load_scheduler=self.configs.skip_load_scheduler,
                skip_load_step=self.configs.skip_load_step,
                load_step_for_scheduler=self.configs.load_step_for_scheduler,
            )
    
    def save_checkpoint(self):
        if DIST_WRAPPER.rank == 0:
            path = f"{self.ckpt_dir}/{self.step}.pt"
            checkpoint = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": (
                    self.lr_scheduler.state_dict()
                    if self.lr_scheduler is not None
                    else None
                ),
                "step": self.step,
            }
            torch.save(checkpoint, path)
            self.print(f"Saved checkpoint to {path}")

    def print(self, msg: str):
        if DIST_WRAPPER.rank == 0:
            logger.info(msg)

    def model_forward(
        self,
        batch: dict,
        mode: str = "train",
    ) -> tuple[ODesignOutput, GroundTruth, LossInput]:
        assert mode in ["train", "eval"]
        pred_output, ground_truth, loss_input = self.model(
            feature_data=batch["feature_data"],
            label_data=batch["label_data"],
            label_full_data=batch["label_full_data"],
            mode=mode,
            current_step=self.step if mode == "train" else None,
            symmetric_permutation=self.symmetric_permutation,
        )
        return pred_output, ground_truth, loss_input

    def get_loss(
        self,
        loss_input: LossInput,
        pred_output: ODesignOutput,
        ground_truth: GroundTruth,
        mode: str = "train",
    ) -> tuple[torch.Tensor, dict]:
        assert mode in ["train", "eval"]

        loss, loss_dict = autocasting_disable_decorator(self.configs.model.skip_amp.loss)(
            self.loss
        )(
            loss_input=loss_input,
            pred_output=pred_output,
            ground_truth=ground_truth,
            mode=mode,
        )
        return loss, loss_dict

    @torch.no_grad()
    def evaluate(self):
        # Init Metric Aggregator
        simple_metric_wrapper = SimpleMetricAggregator(["avg"])
        eval_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.model.dtype]
        enable_amp = (
            torch.autocast(device_type="cuda", dtype=eval_precision)
            if torch.cuda.is_available()
            else nullcontext()
        )
        self.model.eval()

        for test_name, test_dl in self.test_dls.items():
            self.print(f"Testing on {test_name}")
            evaluated_pids = []
            total_batch_num = len(test_dl)
            for index, batch in enumerate(tqdm(test_dl)):
                batch = to_device(batch, self.device)
                pid = batch["basic"]["pdb_id"]

                if index + 1 == total_batch_num and DIST_WRAPPER.world_size > 1:
                    # Gather all pids across ranks for avoiding duplicated evaluations when drop_last = False
                    all_data_ids = DIST_WRAPPER.all_gather_object(evaluated_pids)
                    dedup_ids = set(sum(all_data_ids, []))
                    if pid in dedup_ids:
                        print(
                            f"Rank {DIST_WRAPPER.rank}: Drop data_id {pid} as it is already evaluated."
                        )
                        break
                evaluated_pids.append(pid)

                simple_metrics = {}
                with enable_amp:
                    # Model forward
                    pred_output, ground_truth, loss_input = self.model_forward(batch, mode="eval")
                    # Loss forward
                    _, loss_dict = self.get_loss(loss_input, pred_output, ground_truth, mode="eval")
                    simple_metrics.update(loss_dict)

                    if self.configs.eval_dump:
                        dumper = DataDumper(base_dir=self.eval_dump_dir)
                        dumper.dump(
                            dataset_name="",
                            pdb_id=pid,
                            seed=self.configs.seed,
                            pred_dict=pred_output.to_dict(),
                            atom_array=batch["atom_array"],
                            entity_poly_type=batch["basic"]["entity_poly_type"],
                        )                    

                # Metrics
                for key, value in simple_metrics.items():
                    simple_metric_wrapper.add(
                        key, value, namespace=test_name
                    )

                del batch, simple_metrics
                if index % 5 == 0:
                    # Release some memory periodically
                    torch.cuda.empty_cache()

            metrics = simple_metric_wrapper.calc()
            self.print(f"Step {self.step}, eval {test_name}: {metrics}")
            if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
                wandb.log(metrics, step=self.step)

    def update(self):
        # Clip the gradient
        if self.configs.grad_clip_norm != 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.configs.grad_clip_norm
            )

    def train_step(self, batch: dict, profile_record: dict | None = None):
        self.model.train()
        # FP16 training has not been verified yet
        train_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.model.dtype]
        enable_amp = (
            torch.autocast(
                device_type="cuda", dtype=train_precision, cache_enabled=False
            )
            if torch.cuda.is_available()
            else nullcontext()
        )

        scaler = torch.GradScaler(
            device="cuda" if torch.cuda.is_available() else "cpu",
            enabled=(self.configs.model.dtype == "float16"),
        )

        train_step_start = self._profile_now() if profile_record is not None else None
        with enable_amp:
            forward_start = self._profile_now() if profile_record is not None else None
            self._profile_cuda_stage_begin(profile_record, "forward")
            if self.profile_modules:
                self._profile_set_model_record(profile_record)
            try:
                pred_output, ground_truth, loss_input = self.model_forward(
                    batch, mode="train"
                )
            finally:
                if self.profile_modules:
                    self._profile_set_model_record(None)
            if profile_record is not None:
                self._profile_cuda_stage_end(profile_record, "forward")
                profile_record["forward_sec"] = self._profile_now() - forward_start
            loss_start = self._profile_now() if profile_record is not None else None
            self._profile_cuda_stage_begin(profile_record, "loss")
            if self.profile_modules:
                self._profile_set_model_record(profile_record)
            try:
                loss, loss_dict = self.get_loss(
                    loss_input, pred_output, ground_truth, mode="train"
                )
            finally:
                if self.profile_modules:
                    self._profile_set_model_record(None)
            if profile_record is not None:
                self._profile_cuda_stage_end(profile_record, "loss")
                profile_record["loss_sec"] = self._profile_now() - loss_start
                profile_record["loss"] = self._profile_to_float(loss)
                profile_record["loss_components"] = {
                    key: self._profile_to_float(value)
                    for key, value in loss_dict.items()
                    if "loss" in key
                }

        if self.configs.model.dtype in ["bf16", "fp32"]:
            if is_loss_nan_check(loss):
                self.print(f"Skip iteration with NaN loss: {self.step} steps")
                loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
        backward_start = self._profile_now() if profile_record is not None else None
        self._profile_cuda_stage_begin(profile_record, "backward")
        if self.profile_modules:
            self._profile_set_model_record(profile_record)
        try:
            scaler.scale(loss / self.iters_to_accumulate).backward()
        finally:
            if self.profile_modules:
                self._profile_set_model_record(None)
        if profile_record is not None:
            self._profile_cuda_stage_end(profile_record, "backward")
            profile_record["backward_sec"] = self._profile_now() - backward_start

        did_optimizer_update = False
        # For simplicity, the global training step is used
        if (self.global_step + 1) % self.iters_to_accumulate == 0:
            self.print(
                f"self.step {self.step}, self.iters_to_accumulate: {self.iters_to_accumulate}"
            )
            optimizer_start = self._profile_now() if profile_record is not None else None
            self._profile_cuda_stage_begin(profile_record, "optimizer")
            # Unscales the gradients of optimizer's assigned parameters in-place
            scaler.unscale_(self.optimizer)
            # Do grad clip only
            self.update()
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.lr_scheduler.step()
            did_optimizer_update = True
            if profile_record is not None:
                self._profile_cuda_stage_end(profile_record, "optimizer")
                profile_record["optimizer_sec"] = self._profile_now() - optimizer_start
                profile_record["optimizer_update"] = True
        elif profile_record is not None:
            profile_record["optimizer_sec"] = 0.0
            profile_record["optimizer_update"] = False
        for key, value in loss_dict.items():
            if "loss" not in key:
                continue
            self.train_metric_wrapper.add(key, value, namespace="train")

        should_empty_cache = self._should_empty_cache_for_policy(
            self.empty_cache_policy, did_optimizer_update
        )
        empty_cache_start = self._profile_now() if profile_record is not None else None
        self._profile_cuda_stage_begin(profile_record, "empty_cache")
        if should_empty_cache:
            torch.cuda.empty_cache()
        if profile_record is not None:
            self._profile_cuda_stage_end(profile_record, "empty_cache")
            profile_record["empty_cache_policy"] = self.empty_cache_policy
            profile_record["empty_cache_called"] = bool(should_empty_cache)
            profile_record["empty_cache_sec"] = self._profile_now() - empty_cache_start
            profile_record["train_step_sec"] = self._profile_now() - train_step_start

    def progress_bar(self, desc: str = ""):
        if DIST_WRAPPER.rank != 0:
            return
        if self.global_step % (
            self.configs.eval_interval * self.iters_to_accumulate
        ) == 0 or (not hasattr(self, "_ipbar")):
            # Start a new progress bar
            self._pbar = tqdm(
                range(
                    self.global_step
                    % (self.iters_to_accumulate * self.configs.eval_interval),
                    self.iters_to_accumulate * self.configs.eval_interval,
                )
            )
            self._ipbar = iter(self._pbar)

        step = next(self._ipbar)
        self._pbar.set_description(
            f"[step {self.step}: {step}/{self.iters_to_accumulate * self.configs.eval_interval}] {desc}"
        )
        return

    def run(self):
        """
        Main entry for the TrainRunner.

        This function handles the training process, evaluation, logging, and checkpoint saving.
        """
        if self.configs.eval_only or self.configs.eval_first or self.configs.eval_dump:
            self.evaluate()
            if self.configs.eval_only or self.configs.eval_dump:
                return

        with self._profile_torch_profiler_context() as torch_profiler:
            self._torch_profiler = torch_profiler
            try:
                self._run_training_loop()
            finally:
                self._torch_profiler = None

    def _run_training_loop(self):
        if self.profile_enabled:
            self._profile_last_iter_end = self._profile_now()
        while True:
            for batch in self.train_dl:
                iter_ready_time = self._profile_now() if self.profile_enabled else None
                profile_record = None
                if self.profile_enabled:
                    profile_record = {
                        "data_wait_sec": iter_ready_time - self._profile_last_iter_end,
                    }
                    profile_record.update(self._profile_batch_stats(batch))
                is_update_step = (self.global_step + 1) % self.iters_to_accumulate == 0
                is_last_step = (self.step + 1) == self.configs.max_steps
                step_need_log = (self.step + 1) % self.configs.log_interval == 0

                step_need_eval = (
                    self.configs.eval_interval > 0
                    and (self.step + 1) % self.configs.eval_interval == 0
                )
                step_need_save = (
                    self.configs.checkpoint_interval > 0
                    and (self.step + 1) % self.configs.checkpoint_interval == 0
                )

                is_last_step &= is_update_step
                step_need_log &= is_update_step
                step_need_eval &= is_update_step
                step_need_save &= is_update_step

                to_device_start = self._profile_now() if profile_record is not None else None
                self._profile_cuda_stage_begin(profile_record, "to_device")
                batch = to_device(batch, self.device)
                if profile_record is not None:
                    self._profile_cuda_stage_end(profile_record, "to_device")
                    profile_record["to_device_sec"] = self._profile_now() - to_device_start
                    profile_record["is_update_step"] = bool(is_update_step)
                    profile_record["will_save_checkpoint"] = bool(step_need_save or is_last_step)
                    profile_record["will_eval"] = bool(step_need_eval or is_last_step)
                self.progress_bar()
                self.train_step(batch, profile_record=profile_record)
                if profile_record is not None:
                    profile_record["microbatch_total_sec"] = (
                        self._profile_now() - iter_ready_time
                    )
                    self._profile_write(profile_record)
                self._profile_torch_profiler_step()
                if step_need_log or is_last_step:
                    metrics = self.train_metric_wrapper.calc()
                    self.print(f"Step {self.step} train: {metrics}")
                    last_lr = self.lr_scheduler.get_last_lr()
                    if DIST_WRAPPER.rank == 0:
                        if self.configs.use_wandb:
                            lr_dict = {"train/lr": last_lr[0]}
                            for group_i, group_lr in enumerate(last_lr):
                                lr_dict[f"train/group{group_i}_lr"] = group_lr
                            wandb.log(lr_dict, step=self.step)
                        self.print(f"Step {self.step}, lr: {last_lr}")
                    if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
                        wandb.log(metrics, step=self.step)

                if step_need_save or is_last_step:
                    self.save_checkpoint()

                if step_need_eval or is_last_step:
                    self.evaluate()
                self.global_step += 1
                if self.global_step % self.iters_to_accumulate == 0:
                    self.step += 1
                if self.step >= self.configs.max_steps:
                    self.print(f"Finish training after {self.step} steps")
                    break
                if self.profile_enabled:
                    self._profile_last_iter_end = self._profile_now()
            if self.step >= self.configs.max_steps:
                break
