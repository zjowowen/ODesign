import torch
import attr
from ml_collections.config_dict import ConfigDict

from src.api.data_interface import OFeatureData, OLabelData
from src.model.odesign import ODesign
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
                    "n_blocks": 0,
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
                        "n_blocks": 0,
                        "n_heads": 1,
                        "n_queries": 2,
                        "n_keys": 4,
                    },
                    "transformer": {
                        "n_blocks": 0,
                        "n_heads": 1,
                        "drop_path_rate": 0.0,
                    },
                    "atom_decoder": {
                        "n_blocks": 0,
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
