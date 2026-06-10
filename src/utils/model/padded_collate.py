from typing import Any, Mapping, Optional

import attr
import torch

from src.api.data_interface import OFeatureData, OLabelData


TOKEN_FIELDS = {
    "token_index",
    "residue_index",
    "asym_id",
    "entity_id",
    "sym_id",
    "restype",
    "has_frame",
    "frame_atom_index",
    "profile",
    "deletion_mean",
    "is_hotspot_residue",
    "is_cyclic_token",
    "token_padding_mask",
}
ATOM_FIELDS = {
    "ref_pos",
    "ref_mask",
    "ref_element",
    "ref_charge",
    "ref_atom_name_chars",
    "ref_space_uid",
    "atom_to_token_idx",
    "atom_to_tokatom_idx",
    "is_protein",
    "is_ligand",
    "is_dna",
    "is_rna",
    "mol_id",
    "mol_atom_index",
    "entity_mol_id",
    "pae_rep_atom_mask",
    "modified_res_mask",
    "is_condition_atom",
    "distogram_rep_atom_mask",
    "plddt_m_rep_atom_mask",
    "coordinate",
    "coordinate_mask",
    "atom_padding_mask",
}
TOKEN_PAIR_FIELDS = {
    "token_bonds",
    "token_pair_gen_mask",
    "token_bond_gen_mask",
    "constraint_feature",
    "token_bond_type_label",
}
ATOM_PAIR_FIELDS = {
    "bond_mask",
    "ligand_bond_mask",
}
MSA_TOKEN_FIELDS = {
    "msa",
    "has_deletion",
    "deletion_value",
    "msa_token_mask",
    "template_restype",
    "template_all_atom_mask",
    "template_all_atom_positions",
}
ATOM_SET_FIELDS = {
    "chain_1_mask",
    "chain_2_mask",
}


def collate_fn_odesign_padded(samples: list[dict]) -> dict:
    """Collate ODesign samples by padding variable token, atom, and MSA axes."""
    if not samples:
        raise ValueError("collate_fn_odesign_padded requires at least one sample")

    feature_data = [sample["feature_data"] for sample in samples]
    token_lengths = [int(item.token_index.shape[-1]) for item in feature_data]
    atom_lengths = [int(item.atom_to_token_idx.shape[-1]) for item in feature_data]
    msa_lengths = [_num_msa(item) for item in feature_data]

    batch = {
        "feature_data": _collate_data_class(
            OFeatureData,
            feature_data,
            token_lengths=token_lengths,
            atom_lengths=atom_lengths,
            msa_lengths=msa_lengths,
            include_padding_masks=True,
        ),
        "label_data": _collate_data_class(
            OLabelData,
            [sample["label_data"] for sample in samples],
            token_lengths=token_lengths,
            atom_lengths=[
                int(sample["label_data"].coordinate.shape[0]) for sample in samples
            ],
            msa_lengths=[0] * len(samples),
        ),
        "label_full_data": _collate_data_class(
            OLabelData,
            [sample["label_full_data"] for sample in samples],
            token_lengths=_label_token_lengths(
                [sample["label_full_data"] for sample in samples], token_lengths
            ),
            atom_lengths=[
                int(sample["label_full_data"].coordinate.shape[0])
                for sample in samples
            ],
            msa_lengths=[0] * len(samples),
        ),
        "basic": _collate_mapping([sample["basic"] for sample in samples]),
    }

    for key in samples[0].keys() - batch.keys():
        batch[key] = _collate_values([sample[key] for sample in samples])
    return batch


def _collate_data_class(
    cls: type,
    values: list[Any],
    *,
    token_lengths: list[int],
    atom_lengths: list[int],
    msa_lengths: list[int],
    include_padding_masks: bool = False,
) -> Any:
    max_token = max(token_lengths) if token_lengths else 0
    max_atom = max(atom_lengths) if atom_lengths else 0

    kwargs = {}
    for field in attr.fields(cls):
        kwargs[field.name] = _collate_field(
            field.name,
            [getattr(value, field.name) for value in values],
            token_lengths=token_lengths,
            atom_lengths=atom_lengths,
            msa_lengths=msa_lengths,
            max_token=max_token,
            max_atom=max_atom,
        )

    if include_padding_masks:
        device = values[0].token_index.device
        kwargs["token_padding_mask"] = _padding_mask(token_lengths, max_token, device)
        kwargs["atom_padding_mask"] = _padding_mask(atom_lengths, max_atom, device)

    return cls(**kwargs)


def _collate_field(
    name: str,
    values: list[Any],
    *,
    token_lengths: list[int],
    atom_lengths: list[int],
    msa_lengths: list[int],
    max_token: int,
    max_atom: int,
) -> Any:
    real_tensors = [value for value in values if torch.is_tensor(value)]
    if not real_tensors:
        return None if all(value is None for value in values) else list(values)

    role = _field_role(name, real_tensors[0])
    if role is None:
        role = _infer_role(values, token_lengths, atom_lengths, msa_lengths)
    if role is None:
        return _collate_values(values)

    first_real = real_tensors[0]
    max_first_dim = _max_first_dim(values) if role in {"msa_token", "atom_set"} else 0
    padded = []
    for index, value in enumerate(values):
        source = value
        if source is None:
            source = first_real.new_zeros(
                _source_shape_for_missing(
                    role,
                    first_real,
                    token_lengths[index],
                    atom_lengths[index],
                    msa_lengths[index],
                )
            )
        padded.append(
            _pad_tensor(
                source,
                role,
                token_lengths[index],
                atom_lengths[index],
                msa_lengths[index],
                max_token,
                max_atom,
                max_first_dim,
            )
        )
    return torch.stack(padded, dim=0)


def _field_role(name: str, tensor: torch.Tensor) -> Optional[str]:
    if tensor.ndim == 0:
        return None
    if name in TOKEN_PAIR_FIELDS and tensor.ndim >= 2:
        return "token_pair"
    if name in ATOM_PAIR_FIELDS and tensor.ndim >= 2:
        return "atom_pair"
    if name in MSA_TOKEN_FIELDS and tensor.ndim >= 2:
        return "msa_token"
    if name in ATOM_SET_FIELDS and tensor.ndim >= 2:
        return "atom_set"
    if name in TOKEN_FIELDS:
        return "token"
    if name in ATOM_FIELDS:
        return "atom"
    return None


def _infer_role(
    values: list[Any],
    token_lengths: list[int],
    atom_lengths: list[int],
    msa_lengths: list[int],
) -> Optional[str]:
    tensor_infos = [
        (value, token_lengths[index], atom_lengths[index], msa_lengths[index])
        for index, value in enumerate(values)
        if torch.is_tensor(value)
    ]
    if not tensor_infos or tensor_infos[0][0].ndim == 0:
        return None

    def all_match(predicate) -> bool:
        return all(predicate(tensor, token_length, atom_length, msa_length) for tensor, token_length, atom_length, msa_length in tensor_infos)

    if all_match(
        lambda tensor, token_length, _atom_length, _msa_length: tensor.ndim >= 2
        and tensor.shape[0] == token_length
        and tensor.shape[1] == token_length
    ):
        return "token_pair"
    if all_match(
        lambda tensor, _token_length, atom_length, _msa_length: tensor.ndim >= 2
        and tensor.shape[0] == atom_length
        and tensor.shape[1] == atom_length
    ):
        return "atom_pair"
    if all_match(
        lambda tensor, token_length, _atom_length, _msa_length: tensor.ndim >= 2
        and tensor.shape[1] == token_length
    ):
        return "msa_token"
    if all_match(
        lambda tensor, _token_length, atom_length, _msa_length: tensor.ndim >= 2
        and tensor.shape[1] == atom_length
    ):
        return "atom_set"
    if all_match(
        lambda tensor, token_length, _atom_length, _msa_length: tensor.shape[0]
        == token_length
    ):
        return "token"
    if all_match(
        lambda tensor, _token_length, atom_length, _msa_length: tensor.shape[0]
        == atom_length
    ):
        return "atom"
    return None


def _pad_tensor(
    tensor: torch.Tensor,
    role: str,
    token_length: int,
    atom_length: int,
    msa_length: int,
    max_token: int,
    max_atom: int,
    max_first_dim: int,
) -> torch.Tensor:
    target_shape = _target_shape(role, tensor, max_token, max_atom, max_first_dim)
    output = tensor.new_zeros(target_shape)

    if role == "token":
        copy_token = min(token_length, tensor.shape[0])
        output[:copy_token] = tensor[:copy_token]
    elif role == "atom":
        copy_atom = min(atom_length, tensor.shape[0])
        output[:copy_atom] = tensor[:copy_atom]
    elif role == "token_pair":
        copy_token = min(token_length, tensor.shape[0], tensor.shape[1])
        output[:copy_token, :copy_token] = tensor[:copy_token, :copy_token]
    elif role == "atom_pair":
        copy_atom = min(atom_length, tensor.shape[0], tensor.shape[1])
        output[:copy_atom, :copy_atom] = tensor[:copy_atom, :copy_atom]
    elif role == "msa_token":
        copy_rows = min(max_first_dim, tensor.shape[0])
        copy_token = min(token_length, tensor.shape[1])
        output[:copy_rows, :copy_token] = tensor[:copy_rows, :copy_token]
    elif role == "atom_set":
        copy_rows = min(max_first_dim, tensor.shape[0])
        copy_atom = min(atom_length, tensor.shape[1])
        output[:copy_rows, :copy_atom] = tensor[:copy_rows, :copy_atom]
    else:
        raise ValueError(f"Unsupported padding role: {role}")

    return output


def _target_shape(
    role: str,
    tensor: torch.Tensor,
    max_token: int,
    max_atom: int,
    max_first_dim: int,
) -> tuple:
    if role == "token":
        return (max_token, *tensor.shape[1:])
    if role == "atom":
        return (max_atom, *tensor.shape[1:])
    if role == "token_pair":
        return (max_token, max_token, *tensor.shape[2:])
    if role == "atom_pair":
        return (max_atom, max_atom, *tensor.shape[2:])
    if role == "msa_token":
        return (max_first_dim, max_token, *tensor.shape[2:])
    if role == "atom_set":
        return (max_first_dim, max_atom, *tensor.shape[2:])
    raise ValueError(f"Unsupported padding role: {role}")


def _source_shape_for_missing(
    role: str,
    tensor: torch.Tensor,
    token_length: int,
    atom_length: int,
    msa_length: int,
) -> tuple:
    if role == "token":
        return (token_length, *tensor.shape[1:])
    if role == "atom":
        return (atom_length, *tensor.shape[1:])
    if role == "token_pair":
        return (token_length, token_length, *tensor.shape[2:])
    if role == "atom_pair":
        return (atom_length, atom_length, *tensor.shape[2:])
    if role == "msa_token":
        return (0, token_length, *tensor.shape[2:])
    if role == "atom_set":
        return (0, atom_length, *tensor.shape[2:])
    raise ValueError(f"Unsupported padding role: {role}")


def _collate_mapping(values: list[Mapping[str, Any]]) -> dict:
    keys = set().union(*(value.keys() for value in values))
    return {key: _collate_values([value.get(key) for value in values]) for key in keys}


def _collate_values(values: list[Any]) -> Any:
    tensors = [value for value in values if torch.is_tensor(value)]
    if len(tensors) != len(values):
        return list(values)
    if not tensors:
        return None if all(value is None for value in values) else list(values)
    if all(tensor.shape == tensors[0].shape for tensor in tensors):
        return torch.stack(tensors, dim=0)
    return list(values)


def _max_first_dim(values: list[Any]) -> int:
    return max(value.shape[0] for value in values if torch.is_tensor(value))


def _padding_mask(lengths: list[int], max_length: int, device: torch.device) -> torch.Tensor:
    mask = torch.ones((len(lengths), max_length), dtype=torch.bool, device=device)
    for index, length in enumerate(lengths):
        mask[index, :length] = False
    return mask


def _num_msa(feature_data: OFeatureData) -> int:
    if torch.is_tensor(feature_data.msa):
        return int(feature_data.msa.shape[0])
    return 0


def _label_token_lengths(labels: list[OLabelData], fallback: list[int]) -> list[int]:
    lengths = []
    for label, fallback_length in zip(labels, fallback):
        if torch.is_tensor(label.token_bond_type_label):
            lengths.append(int(label.token_bond_type_label.shape[0]))
        else:
            lengths.append(fallback_length)
    return lengths
