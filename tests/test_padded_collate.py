import torch

from src.utils.model.padded_collate import collate_fn_odesign_padded
from src.api.data_interface import OFeatureData, OLabelData


def _feature_data(num_token: int, num_atom: int, num_msa: int, offset: int) -> OFeatureData:
    token_index = torch.arange(num_token) + offset
    atom_index = torch.arange(num_atom) + offset
    atom_to_token_idx = torch.arange(num_atom) % num_token

    feature_dict = {
        "token_index": token_index,
        "residue_index": token_index + 10,
        "asym_id": torch.ones(num_token, dtype=torch.long),
        "entity_id": torch.full((num_token,), offset + 1, dtype=torch.long),
        "sym_id": torch.zeros(num_token, dtype=torch.long),
        "restype": torch.nn.functional.one_hot(
            torch.arange(num_token) % 4, num_classes=4
        ).long(),
        "token_bonds": torch.full((num_token, num_token, 2), float(offset)),
        "token_pair_gen_mask": torch.ones(num_token, num_token, dtype=torch.bool),
        "token_bond_gen_mask": torch.ones(num_token, num_token, dtype=torch.bool),
        "ref_pos": torch.arange(num_atom * 3, dtype=torch.float32).reshape(num_atom, 3)
        + offset,
        "ref_mask": torch.ones(num_atom, dtype=torch.bool),
        "ref_element": torch.nn.functional.one_hot(
            torch.arange(num_atom) % 3, num_classes=3
        ).long(),
        "ref_charge": torch.arange(num_atom, dtype=torch.float32) + offset,
        "ref_atom_name_chars": torch.ones(num_atom, 4, 3, dtype=torch.long),
        "ref_space_uid": atom_index,
        "has_frame": torch.ones(num_token, dtype=torch.bool),
        "frame_atom_index": torch.arange(num_token * 3).reshape(num_token, 3),
        "atom_to_token_idx": atom_to_token_idx,
        "atom_to_tokatom_idx": atom_index,
        "is_protein": torch.ones(num_atom, dtype=torch.bool),
        "is_ligand": torch.zeros(num_atom, dtype=torch.bool),
        "is_dna": torch.zeros(num_atom, dtype=torch.bool),
        "is_rna": torch.zeros(num_atom, dtype=torch.bool),
        "resolution": torch.tensor([2.0 + offset]),
        "mol_id": atom_index,
        "mol_atom_index": atom_index,
        "entity_mol_id": atom_index + 100,
        "masked_asym_ids": [offset],
        "atom_perm_list": [[0, min(1, num_atom - 1)]],
        "pae_rep_atom_mask": torch.ones(num_atom, dtype=torch.bool),
        "modified_res_mask": torch.ones(num_atom, dtype=torch.bool),
        "is_condition_atom": torch.zeros(num_atom, dtype=torch.bool),
        "distogram_rep_atom_mask": torch.ones(num_atom, dtype=torch.bool),
        "plddt_m_rep_atom_mask": torch.ones(num_atom, dtype=torch.bool),
        "bond_mask": torch.ones(num_atom, num_atom, dtype=torch.bool),
        "constraint_feature": torch.full((num_token, num_token, 1), float(offset)),
        "msa": torch.arange(num_msa * num_token).reshape(num_msa, num_token)
        + offset,
        "has_deletion": torch.zeros(num_msa, num_token, dtype=torch.bool),
        "deletion_value": torch.zeros(num_msa, num_token, dtype=torch.float32),
        "profile": torch.ones(num_token, 5, dtype=torch.float32) * offset,
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
    label_dict = {
        "coordinate": torch.arange(num_atom * 3, dtype=torch.float32).reshape(
            num_atom, 3
        )
        + offset,
        "coordinate_mask": torch.ones(num_atom, dtype=torch.bool),
        "token_bond_type_label": torch.full(
            (num_token, num_token), offset, dtype=torch.long
        ),
        "ligand_bond_mask": torch.ones(num_atom, num_atom, dtype=torch.bool),
        "entity_mol_id": torch.arange(num_atom) + offset,
        "mol_id": torch.arange(num_atom) + offset,
        "mol_atom_index": torch.arange(num_atom),
        "pae_rep_atom_mask": torch.ones(num_atom, dtype=torch.bool),
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
        "label_full_data": _label_data(num_token, num_atom, offset + 10),
        "basic": {
            "pdb_id": pdb_id,
            "N_token": torch.tensor([num_token]),
            "N_atom": torch.tensor([num_atom]),
            "entity_poly_type": {"A": "protein", "id": pdb_id},
            "chain_id": ["A", "B"][: max(1, min(2, num_token))],
        },
    }


def test_padded_collate_pads_feature_and_label_tensors() -> None:
    batch = collate_fn_odesign_padded(
        [
            _sample(num_token=2, num_atom=3, num_msa=1, offset=1, pdb_id="short"),
            _sample(num_token=4, num_atom=5, num_msa=3, offset=7, pdb_id="long"),
        ]
    )

    feature_data = batch["feature_data"]
    label_data = batch["label_data"]

    assert feature_data.num_token == 4
    assert feature_data.num_atom == 5
    assert feature_data.token_index.shape == (2, 4)
    assert feature_data.has_frame.shape == (2, 4)
    assert feature_data.frame_atom_index.shape == (2, 4, 3)
    assert feature_data.ref_pos.shape == (2, 5, 3)
    assert feature_data.modified_res_mask.shape == (2, 5)
    assert feature_data.token_bonds.shape == (2, 4, 4, 2)
    assert feature_data.bond_mask.shape == (2, 5, 5)
    assert feature_data.msa.shape == (2, 3, 4)
    assert feature_data.profile.shape == (2, 4, 5)
    assert label_data.coordinate.shape == (2, 5, 3)
    assert label_data.token_bond_type_label.shape == (2, 4, 4)
    assert label_data.ligand_bond_mask.shape == (2, 5, 5)

    assert feature_data.token_padding_mask.tolist() == [
        [False, False, True, True],
        [False, False, False, False],
    ]
    assert feature_data.atom_padding_mask.tolist() == [
        [False, False, False, True, True],
        [False, False, False, False, False],
    ]
    assert feature_data.has_frame[0, 2:].tolist() == [False, False]
    assert feature_data.frame_atom_index[0, 2:].tolist() == [[0, 0, 0], [0, 0, 0]]
    assert feature_data.ref_mask[0, 3:].tolist() == [False, False]
    assert feature_data.modified_res_mask[0, 3:].tolist() == [False, False]
    assert label_data.coordinate_mask[0, 3:].tolist() == [False, False]
    assert feature_data.token_pair_gen_mask[0, 2:, :].any().item() is False
    assert feature_data.token_pair_gen_mask[0, :, 2:].any().item() is False
    assert feature_data.bond_mask[0, 3:, :].any().item() is False
    assert feature_data.bond_mask[0, :, 3:].any().item() is False
    assert feature_data.msa_token_mask[0, 1:, :].any().item() is False
    assert feature_data.msa_token_mask[0, :, 2:].any().item() is False


def test_padded_collate_collates_basic_metadata_and_optional_fields() -> None:
    batch = collate_fn_odesign_padded(
        [
            _sample(num_token=2, num_atom=3, num_msa=1, offset=1, pdb_id="short"),
            _sample(num_token=4, num_atom=5, num_msa=3, offset=7, pdb_id="long"),
        ]
    )

    feature_data = batch["feature_data"]
    basic = batch["basic"]

    assert feature_data.template_restype is None
    assert feature_data.atom_perm_list == [[[0, 1]], [[0, 1]]]
    assert feature_data.masked_asym_ids == [[1], [7]]
    assert basic["N_token"].shape == (2, 1)
    assert basic["N_token"].tolist() == [[2], [4]]
    assert basic["N_atom"].tolist() == [[3], [5]]
    assert basic["pdb_id"] == ["short", "long"]
    assert basic["entity_poly_type"] == [
        {"A": "protein", "id": "short"},
        {"A": "protein", "id": "long"},
    ]
    assert basic["chain_id"] == [["A", "B"], ["A", "B"]]
