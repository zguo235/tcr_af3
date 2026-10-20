"""Focused tests for the TCR heavy-atom RMSD analysis."""

from pathlib import Path

import numpy as np
import pandas as pd

from analyze_tcr_heavy_atom_rmsd import (
    any_within_cutoff,
    fit_rotation_translation,
    identical_residue_keys,
    paired_atom_coordinates,
    prepare_reader_facing_table,
    select_model_zero,
)


class FakeResidue:
    """Minimal residue object exposing the name method used by matching logic."""

    def __init__(self, residue_name: str) -> None:
        self.residue_name = residue_name

    def get_resname(self) -> str:
        return self.residue_name


def test_identical_residue_keys_excludes_sequence_mismatch() -> None:
    key_a = ("alpha", 27, "")
    key_b = ("beta", 109, "")
    reference = {key_a: FakeResidue("ALA"), key_b: FakeResidue("SER")}
    mobile = {key_a: FakeResidue("ALA"), key_b: FakeResidue("TYR")}
    identical, mismatched = identical_residue_keys(reference, mobile)
    assert identical == [key_a]
    assert mismatched == [key_b]


def test_framework_transform_recovers_rigid_motion() -> None:
    reference = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    rotation_forward = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    mobile = reference @ rotation_forward + np.asarray([4.0, -2.0, 3.0])
    rotation, translation, fitted_rmsd = fit_rotation_translation(reference, mobile)
    assert fitted_rmsd < 1e-10
    assert np.allclose(mobile @ rotation + translation, reference)


def test_symmetric_aspartate_atoms_are_swapped_when_better() -> None:
    reference_atoms = {
        "CB": np.asarray([0.0, 0.0, 0.0]),
        "OD1": np.asarray([1.0, 1.0, 0.0]),
        "OD2": np.asarray([1.0, -1.0, 0.0]),
    }
    mobile_atoms = {
        "CB": np.asarray([0.0, 0.0, 0.0]),
        "OD1": np.asarray([1.0, -1.0, 0.0]),
        "OD2": np.asarray([1.0, 1.0, 0.0]),
    }
    reference, mobile, names = paired_atom_coordinates(
        reference_atoms,
        mobile_atoms,
        residue_name="ASP",
        rotation=np.eye(3),
        translation=np.zeros(3),
        sidechain_only=True,
    )
    assert names == ["CB", "OD1", "OD2"]
    assert np.allclose(reference, mobile)


def test_interface_cutoff_uses_any_heavy_atom_pair() -> None:
    residue = np.asarray([[0.0, 0.0, 0.0], [20.0, 20.0, 20.0]])
    partner = np.asarray([[4.9, 0.0, 0.0]])
    assert any_within_cutoff(residue, partner, 5.0)
    assert not any_within_cutoff(residue, partner, 4.0)


def test_select_model_zero_ignores_rank_or_list_order() -> None:
    models = [
        (2, Path("model_2.cif"), None),
        (0, Path("model_0.cif"), Path("confidence_0.json")),
        (1, Path("model_1.cif"), None),
    ]
    selected = select_model_zero(models)
    assert selected[0] == 0
    assert selected[1] == Path("model_0.cif")


def test_reader_facing_table_omits_internal_model_metadata() -> None:
    """Omit internal AF3 identifiers and paths from submitted tables."""
    internal = pd.DataFrame(
        {
            "comparison_type": ["bound_unbound", "af3_model0"],
            "af3_model": [np.nan, 0],
            "reference_path": ["bound.pdb", "reference.pdb"],
            "mobile_path": ["unbound.pdb", "prediction_model_0.cif"],
            "value": [1.0, 2.0],
        }
    )
    observed = prepare_reader_facing_table(internal)
    assert not {"af3_model", "reference_path", "mobile_path"} & set(observed.columns)
    assert observed["comparison_type"].tolist() == [
        "bound_unbound",
        "top_ranked_af3_experimental_bound",
    ]
