import torch
import attr
import pytest
from ml_collections.config_dict import ConfigDict

from src.api.model_interface import PairFormerInput
from src.api.data_interface import OFeatureData, OLabelData
from src.model.odesign import ODesign
from src.model.modules import generator as generator_module
from src.model.modules.loss import ODesignLoss
from src.utils.model.padded_collate import collate_fn_odesign_padded
from src.utils.permutation.permutation import SymmetricPermutation


def _move_to_device(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if attr.has(value.__class__):
        return value.__class__(
            **{
                key: _move_to_device(item, device)
                for key, item in value.items()
            }
        )
    return value


def _append_dims(value: torch.Tensor, ndim: int) -> torch.Tensor:
    while value.ndim < ndim:
        value = value.unsqueeze(-1)
    return value


def _deterministic_centre_augmentation(
    x_input_coords: torch.Tensor,
    N_sample: int = 1,
    s_trans: float = 1.0,
    centre_only: bool = False,
    mask: torch.Tensor | None = None,
    eps: float = 1e-12,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del s_trans, centre_only
    if mask is None:
        x_center = x_input_coords.mean(dim=-2, keepdim=True)
    else:
        x_center = (
            (x_input_coords * mask.unsqueeze(-1)).sum(dim=-2)
            / (mask.sum(dim=-1, keepdim=True) + eps)
        ).unsqueeze(-2)

    centered = (x_input_coords - x_center).unsqueeze(-3)
    centered = centered.expand(*centered.shape[:-3], N_sample, *centered.shape[-2:])
    batch_prefix = centered.shape[:-3]
    trans = torch.zeros(*batch_prefix, N_sample, 3, device=x_input_coords.device)
    rot = (
        torch.eye(3, device=x_input_coords.device)
        .view(*([1] * len(batch_prefix)), 1, 3, 3)
        .expand(*batch_prefix, N_sample, 3, 3)
    )
    return (
        centered.to(dtype),
        trans.to(dtype),
        rot.to(dtype),
        x_center.to(dtype),
    )


def _make_training_diffusion_deterministic(model: ODesign, sigma_value: float = 1.0) -> None:
    scheduler = model.train_noise_schedulers["coordinate"]

    def sample_noise_level(
        size: torch.Size,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        return torch.full(size, sigma_value, device=device)

    def add_noise_with_condition(
        x_gt: torch.Tensor,
        sigma: torch.Tensor,
        condition_mask: torch.Tensor,
        scale: bool = True,
    ) -> torch.Tensor:
        del condition_mask
        if not scale:
            return x_gt
        sigma = _append_dims(sigma, x_gt.ndim)
        c_in = 1 / torch.sqrt(scheduler.sigma_data**2 + sigma**2)
        return c_in * x_gt

    scheduler.sample_noise_level = sample_noise_level
    scheduler.add_noise_with_condition = add_noise_with_condition


def _forward_loss(
    model: ODesign,
    loss_fn: ODesignLoss,
    symmetric_permutation: SymmetricPermutation,
    batch: dict,
    current_step: int = 0,
) -> tuple:
    model_output, ground_truth, loss_input = model.forward(
        feature_data=batch["feature_data"],
        label_full_data=batch["label_full_data"],
        label_data=batch["label_data"],
        mode="train",
        current_step=current_step,
        symmetric_permutation=symmetric_permutation,
    )
    loss, metrics = loss_fn(
        loss_input=loss_input,
        pred_output=model_output,
        ground_truth=ground_truth,
        mode="train",
    )
    return model_output, ground_truth, loss_input, loss, metrics


def _named_grads(model: ODesign) -> dict[str, torch.Tensor | None]:
    return {
        name: None if param.grad is None else param.grad.detach().cpu().clone()
        for name, param in model.named_parameters()
    }


def _named_params(model: ODesign) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
    }


def _assert_tensor_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float = 2e-5,
    rtol: float = 2e-5,
) -> None:
    actual = actual.detach().cpu()
    expected = expected.detach().cpu()
    if not torch.is_floating_point(actual):
        assert torch.equal(actual, expected), name
        return

    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        max_abs = (actual - expected).abs().max().item()
        raise AssertionError(
            f"{name}: max_abs_diff={max_abs}, atol={atol}, rtol={rtol}"
        )


def _assert_optional_tensor_close(
    name: str,
    actual: torch.Tensor | None,
    expected: torch.Tensor | None,
    *,
    atol: float = 2e-5,
    rtol: float = 2e-5,
) -> None:
    if actual is None or expected is None:
        assert actual is None and expected is None, name
        return
    _assert_tensor_close(name, actual, expected, atol=atol, rtol=rtol)


def _assert_metrics_average_close(
    step: int,
    batched_metrics: dict[str, torch.Tensor],
    micro_metrics: list[dict[str, torch.Tensor]],
) -> None:
    assert set(batched_metrics) == set(micro_metrics[0])
    for metrics in micro_metrics[1:]:
        assert set(metrics) == set(batched_metrics)

    for key in sorted(batched_metrics):
        expected = torch.stack([metrics[key] for metrics in micro_metrics]).mean()
        _assert_tensor_close(
            f"step {step} metrics[{key}]",
            batched_metrics[key],
            expected,
        )


def _assert_real_prefixes_close(
    step: int,
    batched_output,
    batched_gt,
    batched_loss_input,
    micro_outputs: list,
    micro_gts: list,
    micro_loss_inputs: list,
    token_lens: list[int],
    atom_lens: list[int],
) -> None:
    atom_loss_input_names = [
        "atom_padding_mask",
        "distogram_rep_atom_mask",
        "is_condition_atom",
        "is_dna",
        "is_ligand",
        "is_rna",
    ]
    atom_ground_truth_names = [
        "coordinate",
        "coordinate_mask",
    ]
    atom_pair_ground_truth_names = [
        "distance_mask",
        "lddt_mask",
    ]

    for idx, (num_token, num_atom) in enumerate(zip(token_lens, atom_lens)):
        for name in atom_ground_truth_names:
            _assert_tensor_close(
                f"step {step} ground_truth.{name}[{idx}]",
                getattr(batched_gt, name)[idx, :num_atom],
                getattr(micro_gts[idx], name)[0, :num_atom],
                atol=1e-6,
                rtol=1e-6,
            )
        for name in atom_pair_ground_truth_names:
            _assert_optional_tensor_close(
                f"step {step} ground_truth.{name}[{idx}]",
                getattr(batched_gt, name, None)[idx, :num_atom, :num_atom],
                getattr(micro_gts[idx], name, None)[0, :num_atom, :num_atom],
                atol=1e-6,
                rtol=1e-6,
            )
        for name in atom_loss_input_names:
            _assert_optional_tensor_close(
                f"step {step} loss_input.{name}[{idx}]",
                getattr(batched_loss_input, name)[idx, :num_atom],
                getattr(micro_loss_inputs[idx], name)[0, :num_atom],
                atol=1e-6,
                rtol=1e-6,
            )
        _assert_tensor_close(
            f"step {step} loss_input.resolution[{idx}]",
            batched_loss_input.resolution[idx],
            micro_loss_inputs[idx].resolution[0],
            atol=1e-6,
            rtol=1e-6,
        )

        _assert_optional_tensor_close(
            f"step {step} output.coordinate[{idx}]",
            batched_output["coordinate"][idx, :, :num_atom],
            micro_outputs[idx]["coordinate"][0, :, :num_atom],
        )
        _assert_optional_tensor_close(
            f"step {step} output.distogram[{idx}]",
            batched_output["distogram"][idx, :num_token, :num_token],
            micro_outputs[idx]["distogram"][0, :num_token, :num_token],
        )
        _assert_optional_tensor_close(
            f"step {step} output.token_bond_type_logits[{idx}]",
            batched_output["token_bond_type_logits"][idx, :num_token, :num_token],
            micro_outputs[idx]["token_bond_type_logits"][0, :num_token, :num_token],
        )
        _assert_optional_tensor_close(
            f"step {step} output.token_bond_gen_mask[{idx}]",
            batched_output["token_bond_gen_mask"][idx, :num_token, :num_token],
            micro_outputs[idx]["token_bond_gen_mask"][0, :num_token, :num_token],
        )
        _assert_optional_tensor_close(
            f"step {step} output.noise_level[{idx}]",
            batched_output["noise_level"][idx],
            micro_outputs[idx]["noise_level"][0],
        )


def _assert_named_optional_tensors_close(
    prefix: str,
    actual: dict[str, torch.Tensor | None],
    expected: dict[str, torch.Tensor | None],
) -> int:
    assert set(actual) == set(expected)
    compared = 0
    for name in sorted(actual):
        if actual[name] is None or expected[name] is None:
            assert actual[name] is None and expected[name] is None, name
            continue
        compared += 1
        _assert_tensor_close(f"{prefix}.{name}", actual[name], expected[name])
    return compared


def _assert_named_tensors_close(
    prefix: str,
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> int:
    assert set(actual) == set(expected)
    compared = 0
    for name in sorted(actual):
        compared += 1
        _assert_tensor_close(f"{prefix}.{name}", actual[name], expected[name])
    return compared


def _run_multi_step_padding_batch_equivalence(num_steps: int = 3) -> None:
    device = torch.device("cuda")
    configs = _tiny_configs()
    symmetric_permutation = SymmetricPermutation(configs)

    samples = [
        _sample(3, 4, 1, 1, "short"),
        _sample(5, 7, 1, 3, "long"),
    ]
    token_lens = [3, 5]
    atom_lens = [4, 7]
    batch = _move_to_device(collate_fn_odesign_padded(samples), device)
    single_batches = [
        _move_to_device(collate_fn_odesign_padded([sample]), device)
        for sample in samples
    ]

    torch.manual_seed(123)
    batched_model = ODesign(configs).to(device)
    micro_model = ODesign(configs).to(device)
    micro_model.load_state_dict(batched_model.state_dict())
    _make_training_diffusion_deterministic(batched_model)
    _make_training_diffusion_deterministic(micro_model)
    batched_model.train()
    micro_model.train()

    batched_loss_fn = ODesignLoss(configs)
    micro_loss_fn = ODesignLoss(configs)
    batched_optimizer = torch.optim.SGD(batched_model.parameters(), lr=1e-4)
    micro_optimizer = torch.optim.SGD(micro_model.parameters(), lr=1e-4)

    assert (
        _assert_named_tensors_close(
            "initial_parameter",
            _named_params(batched_model),
            _named_params(micro_model),
        )
        > 0
    )

    for step in range(num_steps):
        batched_optimizer.zero_grad(set_to_none=True)
        micro_optimizer.zero_grad(set_to_none=True)

        batched_output, batched_gt, batched_loss_input, batched_loss, batched_metrics = (
            _forward_loss(
                batched_model,
                batched_loss_fn,
                symmetric_permutation,
                batch,
                current_step=step,
            )
        )
        batched_loss.backward()
        batched_grads = _named_grads(batched_model)

        micro_outputs = []
        micro_gts = []
        micro_loss_inputs = []
        micro_losses = []
        micro_metrics = []
        for single_batch in single_batches:
            output, gt, loss_input, loss, metrics = _forward_loss(
                micro_model,
                micro_loss_fn,
                symmetric_permutation,
                single_batch,
                current_step=step,
            )
            micro_outputs.append(output)
            micro_gts.append(gt)
            micro_loss_inputs.append(loss_input)
            micro_losses.append(loss)
            micro_metrics.append(metrics)
        micro_loss = torch.stack(micro_losses).mean()
        micro_loss.backward()
        micro_grads = _named_grads(micro_model)

        _assert_tensor_close(f"step {step} loss", batched_loss, micro_loss)
        _assert_metrics_average_close(step, batched_metrics, micro_metrics)
        _assert_real_prefixes_close(
            step,
            batched_output,
            batched_gt,
            batched_loss_input,
            micro_outputs,
            micro_gts,
            micro_loss_inputs,
            token_lens,
            atom_lens,
        )
        assert (
            _assert_named_optional_tensors_close(
                f"step {step} grad",
                batched_grads,
                micro_grads,
            )
            > 0
        )

        batched_optimizer.step()
        micro_optimizer.step()
        assert (
            _assert_named_tensors_close(
                f"step {step} parameter_after_step",
                _named_params(batched_model),
                _named_params(micro_model),
            )
            > 0
        )


def _tiny_configs() -> ConfigDict:
    c_token = 16
    c_s_inputs = c_token + 35 + 32 + 1 + 1

    return ConfigDict(
        {
            "diffusion_batch_size": 1,
            "data_condition": [],
            "bond_reconstruction": True,
            "data": {
                "train_batch_size": 2,
                "msa": {
                    "enable": False,
                    "strategy": "topk",
                    "sample_cutoff": {"train": 1, "test": 1},
                    "min_size": {"train": 1, "test": 1},
                },
            },
            "model": {
                "c_s": 16,
                "c_z": 8,
                "c_s_inputs": c_s_inputs,
                "c_atom": 8,
                "c_atompair": 4,
                "c_token": c_token,
                "no_bins": 8,
                "chunk_size": None,
                "diffusion_chunk_size": None,
                "blocks_per_ckpt": None,
                "sigma_data": 16.0,
                "use_memory_efficient_kernel": False,
                "use_deepspeed_evo_attention": False,
                "use_flash": False,
                "use_lma": False,
                "use_xformer": False,
                "dtype": "fp32",
                "loss_metrics_sparse_enable": True,
                "skip_amp": {
                    "loss": True,
                    "sample_diffusion": True,
                    "sample_diffusion_training": True,
                },
                "find_unused_parameters": False,
                "train_noise_schedulers": {
                    "coordinate": {
                        "p_mean": -1.2,
                        "p_std": 1.5,
                        "sigma_data": 16.0,
                    }
                },
                "inference_noise_schedulers": {
                    "coordinate": {
                        "rho": 7,
                        "s_max": 160.0,
                        "s_min": 4e-4,
                        "sigma_data": 16.0,
                        "use_predictor_corrector_sampler": False,
                        "gamma0": 0.0,
                        "gamma_min": 1.0,
                        "noise_scale_lambda": 1.0,
                        "step_scale_eta": 1.0,
                        "partial_diffusion": {"enable": False, "snr": 0.1},
                    }
                },
                "sample_diffusion": {
                    "N_sample": 1,
                    "N_step": 1,
                    "attn_chunk_size": None,
                    "diffusion_chunk_size": None,
                },
                "N_model_seed": 1,
                "N_cycle": 1,
                "condition_embedding_drop_rate": 0.0,
                "input_embedder": {
                    "c_atom": 8,
                    "c_atompair": 4,
                    "c_token": c_token,
                },
                "relative_position_encoding": {
                    "c_z": 8,
                    "r_max": 4,
                    "s_max": 2,
                },
                "msa_module": {
                    "n_blocks": 0,
                    "c_m": 4,
                    "c_s_inputs": c_s_inputs,
                    "c_z": 8,
                    "msa_dropout": 0.0,
                    "pair_dropout": 0.0,
                    "blocks_per_ckpt": None,
                    "msa_chunk_size": 4,
                    "msa_max_size": 4,
                },
                "constraint_distogram_embedder": {
                    "blocks_per_ckpt": None,
                    "c": 4,
                    "c_z": 8,
                    "dropout": 0.0,
                    "n_blocks": 0,
                },
                "pairformer": {
                    "n_blocks": 1,
                    "n_heads": 1,
                    "c_s": 16,
                    "c_z": 8,
                    "dropout": 0.0,
                    "blocks_per_ckpt": None,
                },
                "diffusion_module": {
                    "use_fine_grained_checkpoint": False,
                    "sigma_data": 16.0,
                    "blocks_per_ckpt": None,
                    "c_atom": 8,
                    "c_atompair": 4,
                    "c_s": 16,
                    "c_s_inputs": c_s_inputs,
                    "c_token": c_token,
                    "c_z": 8,
                    "atom_encoder": {
                        "n_blocks": 1,
                        "n_heads": 1,
                        "n_queries": 2,
                        "n_keys": 4,
                    },
                    "transformer": {
                        "n_blocks": 1,
                        "n_heads": 1,
                        "drop_path_rate": 0.0,
                    },
                    "atom_decoder": {
                        "n_blocks": 1,
                        "n_heads": 1,
                        "n_queries": 2,
                        "n_keys": 4,
                    },
                },
                "pairwise_head": {
                    "c_z": 8,
                    "no_bins": 8,
                    "no_bond_types": 2,
                    "bond_reconstruction": True,
                },
            },
            "loss": {
                "diffusion_lddt_chunk_size": None,
                "diffusion_bond_chunk_size": None,
                "diffusion_sparse_loss_enable": True,
                "diffusion_lddt_loss_dense": False,
                "resolution": {"min": 0.1, "max": 4.0},
                "weight": {
                    "alpha_diffusion": 1.0,
                    "alpha_distogram": 1.0,
                    "alpha_bond": 1.0,
                    "smooth_lddt": 1.0,
                    "alpha_bond_type": 1.0,
                },
                "diffusion": {
                    "bond": {"eps": 1e-6},
                    "mse": {
                        "eps": 1e-6,
                        "weight_dna": 5.0,
                        "weight_ligand": 10.0,
                        "weight_mse": 1.0 / 3.0,
                        "weight_rna": 5.0,
                    },
                    "smooth_lddt": {"eps": 1e-6},
                },
                "distogram": {
                    "eps": 1e-6,
                    "max_bin": 21.6875,
                    "min_bin": 2.3125,
                    "no_bins": 8,
                },
                "bond_type": {
                    "alpha": 1.0,
                    "eps": 1e-6,
                    "k": 2.0,
                    "num_classes": 2,
                },
            },
            "chain_permutation": {
                "configs": {
                    "accept_it_as_it_is": True,
                    "enumerate_all_anchor_pairs": False,
                    "find_gt_anchor_first": False,
                    "selection_metric": "aligned_rmsd",
                    "use_center_rmsd": False,
                },
                "permute_by_pocket": False,
                "test": {"diffusion_sample": False},
                "train": {"diffusion_sample": False, "mini_rollout": False},
            },
            "atom_permutation": {
                "global_align_wo_symmetric_atom": False,
                "permute_by_pocket": False,
                "test": {"diffusion_sample": False},
                "train": {"diffusion_sample": False, "mini_rollout": False},
            },
        }
    )


def _feature_data(
    num_token: int,
    num_atom: int,
    num_msa: int,
    offset: int,
) -> OFeatureData:
    token_index = torch.arange(num_token)
    atom_index = torch.arange(num_atom)
    atom_to_token_idx = atom_index % num_token

    ref_element = torch.nn.functional.one_hot(atom_index % 8, num_classes=129).long()
    ref_atom_name_chars = torch.nn.functional.one_hot(
        (atom_index[:, None] + torch.arange(4)[None, :]) % 16,
        num_classes=64,
    ).long()

    feature_dict = {
        "token_index": token_index,
        "residue_index": token_index + offset,
        "asym_id": torch.ones(num_token, dtype=torch.long),
        "entity_id": torch.ones(num_token, dtype=torch.long),
        "sym_id": torch.zeros(num_token, dtype=torch.long),
        "restype": torch.nn.functional.one_hot(
            token_index % 20,
            num_classes=35,
        ).long(),
        "token_bonds": torch.zeros(num_token, num_token, dtype=torch.float32),
        "token_pair_gen_mask": torch.ones(num_token, num_token, dtype=torch.bool),
        "token_bond_gen_mask": torch.ones(num_token, num_token, dtype=torch.bool),
        "ref_pos": (
            torch.arange(num_atom * 3, dtype=torch.float32).reshape(num_atom, 3)
            / 10.0
            + offset
        ),
        "ref_mask": torch.ones(num_atom, dtype=torch.bool),
        "ref_element": ref_element,
        "ref_charge": torch.zeros(num_atom, dtype=torch.float32),
        "ref_atom_name_chars": ref_atom_name_chars,
        "ref_space_uid": atom_index,
        "has_frame": torch.ones(num_token, dtype=torch.bool),
        "frame_atom_index": torch.stack(
            [
                atom_to_token_idx.clamp(max=num_atom - 1)[:num_token],
                atom_to_token_idx.roll(1).clamp(max=num_atom - 1)[:num_token],
                atom_to_token_idx.roll(2).clamp(max=num_atom - 1)[:num_token],
            ],
            dim=-1,
        ),
        "atom_to_token_idx": atom_to_token_idx,
        "atom_to_tokatom_idx": atom_index,
        "is_protein": torch.ones(num_atom, dtype=torch.bool),
        "is_ligand": torch.zeros(num_atom, dtype=torch.bool),
        "is_dna": torch.zeros(num_atom, dtype=torch.bool),
        "is_rna": torch.zeros(num_atom, dtype=torch.bool),
        "resolution": torch.tensor([2.0], dtype=torch.float32),
        "mol_id": atom_index,
        "mol_atom_index": atom_index,
        "entity_mol_id": atom_index,
        "masked_asym_ids": [1],
        "atom_perm_list": [[0] for _ in range(num_atom)],
        "pae_rep_atom_mask": torch.ones(num_atom, dtype=torch.bool),
        "modified_res_mask": torch.ones(num_atom, dtype=torch.bool),
        "is_condition_atom": torch.zeros(num_atom, dtype=torch.bool),
        "distogram_rep_atom_mask": atom_index < num_token,
        "plddt_m_rep_atom_mask": torch.ones(num_atom, dtype=torch.bool),
        "bond_mask": torch.zeros(num_atom, num_atom, dtype=torch.bool),
        "constraint_feature": torch.zeros(num_token, num_token, 1, dtype=torch.float32),
        "msa": torch.zeros(num_msa, num_token, dtype=torch.long),
        "has_deletion": torch.zeros(num_msa, num_token, dtype=torch.bool),
        "deletion_value": torch.zeros(num_msa, num_token, dtype=torch.float32),
        "profile": torch.zeros(num_token, 32, dtype=torch.float32),
        "deletion_mean": torch.zeros(num_token, dtype=torch.float32),
        "msa_token_mask": torch.ones(num_msa, num_token, dtype=torch.bool),
        "prot_pair_num_alignments": torch.tensor([num_msa]),
        "prot_unpair_num_alignments": torch.tensor([0]),
        "rna_pair_num_alignments": torch.tensor([0]),
        "rna_unpair_num_alignments": torch.tensor([0]),
        "template_restype": None,
        "template_all_atom_mask": None,
        "template_all_atom_positions": None,
        "is_hotspot_residue": torch.zeros(num_token, dtype=torch.bool),
        "is_cyclic_token": torch.zeros(num_token, dtype=torch.bool),
    }
    return OFeatureData.from_feature_dict(feature_dict)


def _label_data(num_token: int, num_atom: int, offset: int) -> OLabelData:
    atom_index = torch.arange(num_atom)
    coordinate = (
        torch.arange(num_atom * 3, dtype=torch.float32).reshape(num_atom, 3) / 7.0
        + offset
    )
    label_dict = {
        "coordinate": coordinate,
        "coordinate_mask": torch.ones(num_atom, dtype=torch.bool),
        "token_bond_type_label": torch.zeros(num_token, num_token, dtype=torch.long),
        "ligand_bond_mask": torch.zeros(num_atom, num_atom, dtype=torch.bool),
        "entity_mol_id": atom_index,
        "mol_id": atom_index,
        "mol_atom_index": atom_index,
        "pae_rep_atom_mask": atom_index < num_token,
    }
    return OLabelData.from_label_dict(label_dict)


def _sample(
    num_token: int,
    num_atom: int,
    num_msa: int,
    offset: int,
    pdb_id: str,
) -> dict:
    return {
        "feature_data": _feature_data(num_token, num_atom, num_msa, offset),
        "label_data": _label_data(num_token, num_atom, offset),
        "label_full_data": _label_data(num_token, num_atom, offset),
        "basic": {
            "pdb_id": pdb_id,
            "N_token": torch.tensor([num_token]),
            "N_atom": torch.tensor([num_atom]),
        },
    }


def test_padding_batch_train_forward_backward_smoke() -> None:
    torch.manual_seed(0)
    assert torch.cuda.is_available(), (
        "ODesign end-to-end smoke requires CUDA because the H image uses fused "
        "LayerNorm kernels without a CPU fallback."
    )
    torch.cuda.manual_seed_all(0)
    device = torch.device("cuda")

    configs = _tiny_configs()
    batch = collate_fn_odesign_padded(
        [
            _sample(3, 4, 1, 1, "short"),
            _sample(5, 7, 1, 3, "long"),
        ]
    )
    batch = _move_to_device(batch, device)

    model = ODesign(configs).to(device)
    model.train()
    loss_fn = ODesignLoss(configs)
    symmetric_permutation = SymmetricPermutation(configs)

    model_output, ground_truth, loss_input = model.forward(
        feature_data=batch["feature_data"],
        label_full_data=batch["label_full_data"],
        label_data=batch["label_data"],
        mode="train",
        current_step=0,
        symmetric_permutation=symmetric_permutation,
    )
    loss, metrics = loss_fn(
        loss_input=loss_input,
        pred_output=model_output,
        ground_truth=ground_truth,
        mode="train",
    )
    loss.backward()

    assert torch.isfinite(loss).item()
    assert torch.isfinite(metrics["loss"]).item()
    assert model_output["coordinate"].shape[:3] == (2, 1, 7)
    assert ground_truth.coordinate.shape == (2, 7, 3)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    )


def test_pairformer_paths_receive_token_padding_pair_mask() -> None:
    if not torch.cuda.is_available():
        pytest.skip(
            "ODesign pairformer mask wiring test requires CUDA in the H runtime."
        )
    device = torch.device("cuda")
    configs = _tiny_configs()
    configs.data_condition = ["constraint_distogram"]

    captures = {
        "constraint": [],
        "msa": [],
        "pairformer": [],
    }

    class SpyConstraint(torch.nn.Module):
        def forward(self, input_data, z, *, pair_mask=None, **kwargs):
            del input_data, kwargs
            captures["constraint"].append(pair_mask)
            return torch.zeros_like(z)

    class SpyMSA(torch.nn.Module):
        def forward(self, input_data, z, s_inputs, *, pair_mask=None, **kwargs):
            del input_data, s_inputs, kwargs
            captures["msa"].append(pair_mask)
            return z

    class SpyPairformer(torch.nn.Module):
        def forward(self, s, z, *, pair_mask=None, **kwargs):
            del kwargs
            captures["pairformer"].append(pair_mask)
            return s, z

    batch = collate_fn_odesign_padded(
        [
            _sample(3, 4, 1, 1, "short"),
            _sample(5, 7, 1, 3, "long"),
        ]
    )
    pairformer_input = _move_to_device(
        PairFormerInput.from_feature_data(batch["feature_data"]),
        device,
    )

    model = ODesign(configs).to(device)
    model.constraint_distogram_embedder = SpyConstraint().to(device)
    model.msa_module = SpyMSA().to(device)
    model.pairformer_stack = SpyPairformer().to(device)

    model.get_pairformer_output(pairformer_input, N_cycle=1)

    valid_token_mask = ~pairformer_input.token_padding_mask.bool()
    expected_pair_mask = (
        valid_token_mask[..., :, None] & valid_token_mask[..., None, :]
    )

    for name, masks in captures.items():
        assert len(masks) == 1, name
        assert masks[0] is not None, name
        assert torch.is_floating_point(masks[0]), name
        assert torch.equal(masks[0].bool(), expected_pair_mask), name


def test_padding_batch_matches_average_of_single_item_training_steps(monkeypatch) -> None:
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        pytest.skip(
            "ODesign training equivalence test requires CUDA because the H image "
            "uses fused LayerNorm kernels without a CPU fallback."
        )
    torch.cuda.manual_seed_all(0)
    monkeypatch.setattr(
        generator_module,
        "centre_random_augmentation",
        _deterministic_centre_augmentation,
    )
    device = torch.device("cuda")
    configs = _tiny_configs()
    symmetric_permutation = SymmetricPermutation(configs)

    samples = [
        _sample(3, 4, 1, 1, "short"),
        _sample(5, 7, 1, 3, "long"),
    ]
    batch = _move_to_device(collate_fn_odesign_padded(samples), device)
    single_batches = [
        _move_to_device(collate_fn_odesign_padded([sample]), device)
        for sample in samples
    ]

    torch.manual_seed(123)
    batched_model = ODesign(configs).to(device)
    micro_model = ODesign(configs).to(device)
    micro_model.load_state_dict(batched_model.state_dict())
    _make_training_diffusion_deterministic(batched_model)
    _make_training_diffusion_deterministic(micro_model)
    batched_model.train()
    micro_model.train()

    batched_loss_fn = ODesignLoss(configs)
    micro_loss_fn = ODesignLoss(configs)

    batched_output, batched_gt, batched_loss_input, batched_loss, batched_metrics = (
        _forward_loss(batched_model, batched_loss_fn, symmetric_permutation, batch)
    )
    batched_loss.backward()
    batched_grads = _named_grads(batched_model)

    micro_outputs = []
    micro_gts = []
    micro_loss_inputs = []
    micro_losses = []
    micro_metrics = []
    for single_batch in single_batches:
        output, gt, loss_input, loss, metrics = _forward_loss(
            micro_model,
            micro_loss_fn,
            symmetric_permutation,
            single_batch,
        )
        micro_outputs.append(output)
        micro_gts.append(gt)
        micro_loss_inputs.append(loss_input)
        micro_losses.append(loss)
        micro_metrics.append(metrics)
    micro_loss = torch.stack(micro_losses).mean()
    micro_loss.backward()
    micro_grads = _named_grads(micro_model)

    assert torch.allclose(batched_loss, micro_loss, atol=2e-5, rtol=2e-5)
    assert torch.allclose(
        batched_metrics["loss"],
        torch.stack([metrics["loss"] for metrics in micro_metrics]).mean(),
        atol=2e-5,
        rtol=2e-5,
    )

    token_lens = [3, 5]
    atom_lens = [4, 7]
    for idx, (num_token, num_atom) in enumerate(zip(token_lens, atom_lens)):
        assert torch.allclose(
            batched_gt.coordinate[idx, :num_atom],
            micro_gts[idx].coordinate[0, :num_atom],
            atol=1e-6,
            rtol=1e-6,
        )
        assert torch.equal(
            batched_loss_input.atom_padding_mask[idx, :num_atom],
            micro_loss_inputs[idx].atom_padding_mask[0, :num_atom],
        )
        assert torch.allclose(
            batched_output["coordinate"][idx, :, :num_atom],
            micro_outputs[idx]["coordinate"][0, :, :num_atom],
            atol=2e-5,
            rtol=2e-5,
        )
        assert torch.allclose(
            batched_output["distogram"][idx, :num_token, :num_token],
            micro_outputs[idx]["distogram"][0, :num_token, :num_token],
            atol=2e-5,
            rtol=2e-5,
        )
        assert torch.allclose(
            batched_output["token_bond_type_logits"][idx, :num_token, :num_token],
            micro_outputs[idx]["token_bond_type_logits"][0, :num_token, :num_token],
            atol=2e-5,
            rtol=2e-5,
        )

    compared_grads = 0
    for name, batched_grad in batched_grads.items():
        micro_grad = micro_grads[name]
        if batched_grad is None or micro_grad is None:
            assert batched_grad is None and micro_grad is None, name
            continue
        compared_grads += 1
        assert torch.allclose(
            batched_grad,
            micro_grad,
            atol=2e-5,
            rtol=2e-5,
        ), name
    assert compared_grads > 0


def test_padding_batch_matches_microbatch_accumulation_across_optimizer_steps(
    monkeypatch,
) -> None:
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        pytest.skip(
            "ODesign multi-step training equivalence test requires CUDA because "
            "the H image uses fused LayerNorm kernels without a CPU fallback."
        )
    torch.cuda.manual_seed_all(0)
    monkeypatch.setattr(
        generator_module,
        "centre_random_augmentation",
        _deterministic_centre_augmentation,
    )

    _run_multi_step_padding_batch_equivalence()
