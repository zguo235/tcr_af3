"""Regression tests for the AF3-WT structural error-feature analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from helpers.paths import DATA_ROOT, SUPPLEMENTARY_ROOT
from Bio.PDB import Atom, Residue

REPO = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from analyze_af3_wt_error_features import (  # noqa: E402
    PRIMARY_CONTACT_CUTOFF,
    _rng_for_key,
    _within_template_rank_statistic,
    assign_contact_disagreement_strata,
    benjamini_hochberg,
    calculate_template_rmsds,
    classify_mutation_region,
    contact_set_metrics,
    extract_template_context,
    load_common_cohort,
    mutation_rank_association,
    template_spearman_association,
)


def test_contact_disagreement_strata_are_stable_at_third_boundaries() -> None:
    """Assign binary-fraction approximations of one third consistently."""
    values = pd.Series([0.0, 1.0 - 2.0 / 3.0, 0.5, 0.8])
    observed = assign_contact_disagreement_strata(values).tolist()
    assert observed == [
        "No disagreement",
        "Low (0--0.33]",
        "Medium (0.33--0.67]",
        "High (0.67--1.0]",
    ]


def _residue(resseq: int, coordinate: np.ndarray) -> Residue.Residue:
    """Create a minimal alanine residue with one C-alpha atom."""
    residue = Residue.Residue((" ", resseq, " "), "ALA", " ")
    atom = Atom.Atom(
        "CA",
        np.asarray(coordinate, dtype=float),
        0.0,
        1.0,
        " ",
        " CA ",
        resseq,
        element="C",
    )
    residue.add(atom)
    return residue


def test_contact_set_metrics_handles_empty_and_disjoint_sets() -> None:
    tcr_key = ("D", 1, " ")
    pmhc_key = ("C", 1, " ")
    contact = (tcr_key, pmhc_key)

    empty = contact_set_metrics(set(), set())
    assert empty["disagreement"] == 0.0
    assert empty["f1"] == 1.0

    one_sided = contact_set_metrics({contact}, set())
    assert one_sided["disagreement"] == 1.0
    assert one_sided["lost_count"] == 1
    assert one_sided["gained_count"] == 0


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ({"chain": "C", "resseq": 3}, "Peptide"),
        ({"chain": "A", "resseq": 50}, "MHC"),
        ({"chain": "D", "resseq": 10}, "TCR CDR3"),
        ({"chain": "D", "resseq": 20}, "Other TCR CDR"),
        ({"chain": "E", "resseq": 30}, "TCR framework"),
    ],
)
def test_classify_mutation_region(mutation: dict, expected: str) -> None:
    imgt_map = {
        ("D", 10): 108,
        ("D", 20): 30,
        ("E", 30): 80,
    }
    assert classify_mutation_region(mutation, imgt_map) == expected


def test_calculate_template_rmsds_is_rigid_transform_invariant() -> None:
    rotation = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.asarray([7.0, -3.0, 2.0])
    aligned = {}
    imgt_map = {}

    for index in range(12):
        chain = "D" if index < 6 else "E"
        resseq = index + 1
        reference_coordinate = np.asarray(
            [float(index), float(index % 3), float(index % 5)]
        )
        af3_coordinate = np.dot(reference_coordinate - translation, rotation.T)
        key = (chain, resseq, " ")
        aligned[key] = (
            _residue(resseq, reference_coordinate),
            _residue(resseq, af3_coordinate),
        )
        imgt_map[(chain, resseq)] = index + 1

    for index, chain in enumerate(("D", "D", "E", "E"), start=20):
        reference_coordinate = np.asarray(
            [float(index), float(index % 4), float(index % 6)]
        )
        af3_coordinate = np.dot(reference_coordinate - translation, rotation.T)
        key = (chain, index, " ")
        aligned[key] = (
            _residue(index, reference_coordinate),
            _residue(index, af3_coordinate),
        )
        imgt_map[(chain, index)] = 104 + (index - 19)

    metrics = calculate_template_rmsds(aligned, imgt_map)
    assert metrics["framework_rmsd_angstrom"] < 1e-6
    assert metrics["cdr3_rmsd_angstrom"] < 1e-6
    assert metrics["tcr_pmhc_ca_rmsd_angstrom"] < 1e-6


def test_within_template_rank_statistic_weights_templates_equally() -> None:
    frame = pd.DataFrame(
        {
            "template_PDB": ["A"] * 4 + ["B"] * 2,
            "feature": [1, 2, 3, 4, 1, 2],
            "outcome": [10, 20, 30, 40, 3, 7],
        }
    )
    groups = [group for _, group in frame.groupby("template_PDB")]
    estimate, n_templates, n_mutations = _within_template_rank_statistic(
        groups,
        "feature",
        "outcome",
    )
    assert estimate == pytest.approx(1.0)
    assert n_templates == 2
    assert n_mutations == 6


def test_resampling_is_deterministic() -> None:
    template_frame = pd.DataFrame(
        {
            "feature": np.arange(8, dtype=float),
            "outcome": np.arange(8, dtype=float),
            "n_mutations": np.ones(8, dtype=int),
        }
    )
    first = template_spearman_association(
        template_frame,
        "feature",
        "outcome",
        bootstrap_replicates=25,
        permutation_replicates=25,
        seed=20260722,
        analysis_key="determinism-template",
    )
    second = template_spearman_association(
        template_frame,
        "feature",
        "outcome",
        bootstrap_replicates=25,
        permutation_replicates=25,
        seed=20260722,
        analysis_key="determinism-template",
    )
    assert first == second

    mutation_frame = pd.DataFrame(
        {
            "template_PDB": ["A"] * 4 + ["B"] * 4,
            "feature": [1, 2, 3, 4] * 2,
            "outcome": [4, 3, 2, 1, 8, 6, 4, 2],
        }
    )
    first_mutation = mutation_rank_association(
        mutation_frame,
        "feature",
        "outcome",
        bootstrap_replicates=25,
        permutation_replicates=25,
        seed=20260722,
        analysis_key="determinism-mutation",
    )
    second_mutation = mutation_rank_association(
        mutation_frame,
        "feature",
        "outcome",
        bootstrap_replicates=25,
        permutation_replicates=25,
        seed=20260722,
        analysis_key="determinism-mutation",
    )
    assert first_mutation == second_mutation


def test_error_change_rename_preserves_resampling_stream() -> None:
    """Keep published resampling draws stable after renaming the outcome."""
    legacy_rng = _rng_for_key(
        20260722,
        "template|all|BA-DDG|scaffold_penalty|ptm",
    )
    renamed_rng = _rng_for_key(
        20260722,
        "template|all|BA-DDG|absolute_error_change|ptm",
    )
    np.testing.assert_array_equal(
        legacy_rng.integers(0, 37, size=100),
        renamed_rng.integers(0, 37, size=100),
    )


def test_benjamini_hochberg_preserves_indices_and_monotonicity() -> None:
    p_values = pd.Series([0.01, 0.04, np.nan, 0.03], index=[4, 2, 9, 7])
    q_values = benjamini_hochberg(p_values)
    assert np.isnan(q_values.loc[9])
    assert q_values.loc[4] == pytest.approx(0.03)
    assert q_values.loc[2] == pytest.approx(0.04)
    assert q_values.loc[7] == pytest.approx(0.04)


@pytest.mark.integration
def test_common_cohort_counts_and_curation_exclusions() -> None:
    cohort = load_common_cohort(
        DATA_ROOT / "ddg_predictions" / "compare_4models.csv",
        DATA_ROOT / "ATLAS_af" / "index.csv",
        validate_raw_sources=True,
    )
    assert len(cohort) == 438
    assert cohort["template_PDB"].nunique() == 37
    assert int((cohort["num_muts"] == 1).sum()) == 304
    assert "1FYT" not in set(cohort["template_PDB"])
    single_peptide = (
        (cohort["num_muts"] == 1)
        & cohort["mutations"].str.startswith("PEP:C:")
    )
    assert int(single_peptide.sum()) == 67


@pytest.mark.parametrize(
    ("template", "expected_chain_order"),
    [
        ("1MI5", "A;C;D;E"),
        ("3QIB", "A;B;C;D;E"),
    ],
)
@pytest.mark.integration
def test_extract_template_context_for_four_and_five_chain_complexes(
    template: str,
    expected_chain_order: str,
) -> None:
    context = extract_template_context(
        template,
        DATA_ROOT / "ATLAS" / "www" / "structures" / "true_pdb",
        DATA_ROOT / "ATLAS_af" / "wt_output",
        [PRIMARY_CONTACT_CUTOFF],
    )
    row = context.feature_row
    assert row["confidence_chain_order"] == expected_chain_order
    assert np.isfinite(float(row["cdr3_rmsd_angstrom"]))
    assert np.isfinite(float(row["contact_disagreement_5a"]))
    assert 0.0 <= float(row["contact_disagreement_5a"]) <= 1.0
    assert int(row["reference_interface_contact_count_5a"]) > 0
    assert len(context.token_indices) > 300


@pytest.mark.integration
def test_generated_tables_reconcile_manuscript_headline_results() -> None:
    """Verify generated cohort sizes, null result, and peptide-region estimates."""
    supplementary_dir = (
        SUPPLEMENTARY_ROOT
    )
    template_table = pd.read_csv(supplementary_dir / "S15.csv")
    mutation_table = pd.read_csv(supplementary_dir / "S16.csv")
    associations = pd.read_csv(supplementary_dir / "S17.csv")

    assert len(template_table) == 37
    assert len(mutation_table) == 438
    assert len(associations) == 355
    assert int((mutation_table["num_muts"] == 1).sum()) == 304
    assert int(
        (
            (mutation_table["num_muts"] == 1)
            & (mutation_table["single_point_region"] == "Peptide")
        ).sum()
    ) == 67

    contact_counts = (
        assign_contact_disagreement_strata(
            mutation_table["local_contact_disagreement_5a"]
        )
        .value_counts()
        .to_dict()
    )
    assert contact_counts == {
        "No disagreement": 178,
        "Low (0--0.33]": 105,
        "Medium (0.33--0.67]": 119,
        "High (0.67--1.0]": 36,
    }

    template_primary = associations[
        (associations["analysis_level"] == "template")
        & (associations["analysis_type"] == "spearman")
        & (associations["is_primary"] == True)  # noqa: E712
    ]
    assert len(template_primary) == 24
    assert (
        (template_primary["ci95_low"] <= 0.0)
        & (template_primary["ci95_high"] >= 0.0)
    ).all()
    assert float(template_primary["q_value"].min()) == pytest.approx(
        0.6761238761238761
    )
    individual_primary = template_primary[
        template_primary["model_or_contrast"].isin(
            ["RDE-PPI", "Prompt-DDG", "BA-DDG", "H3-DDG"]
        )
    ]
    expected_ranges = {
        "contact_disagreement_5a": (-0.220247, 0.029398),
        "cdr3_rmsd_angstrom": (-0.200332, 0.057373),
        "ptm": (-0.051966, 0.186374),
        "iptm": (-0.013093, 0.254887),
    }
    for feature, (expected_minimum, expected_maximum) in expected_ranges.items():
        estimates = individual_primary.loc[
            individual_primary["feature"] == feature,
            "estimate",
        ]
        assert float(estimates.min()) == pytest.approx(
            expected_minimum,
            abs=5e-7,
        )
        assert float(estimates.max()) == pytest.approx(
            expected_maximum,
            abs=5e-7,
        )

    local_contact = associations[
        (associations["analysis_type"] == "within_template_rank")
        & (associations["feature"] == "local_contact_disagreement_5a")
        & (associations["is_primary"] == True)  # noqa: E712
    ].set_index("model_or_contrast")
    expected_contact = {
        "BA-DDG": (0.445008, -0.003506, 0.483551),
        "H3-DDG": (0.378152, -0.133796, 0.423633),
        "Prompt-DDG": (0.216228, -0.210062, 0.245816),
        "RDE-PPI": (0.166022, -0.059521, 0.179281),
        "BA-DDG minus Prompt-DDG": (0.407340, -0.056754, 0.445730),
        "H3-DDG minus Prompt-DDG": (0.345977, -0.133384, 0.385899),
    }
    for model, (estimate, ci_low, ci_high) in expected_contact.items():
        assert float(local_contact.loc[model, "estimate"]) == pytest.approx(
            estimate,
            abs=5e-7,
        )
        assert float(local_contact.loc[model, "ci95_low"]) == pytest.approx(
            ci_low,
            abs=5e-7,
        )
        assert float(local_contact.loc[model, "ci95_high"]) == pytest.approx(
            ci_high,
            abs=5e-7,
        )
    assert float(local_contact["q_value"].min()) == pytest.approx(
        0.6113886113886113
    )

    peptide_position = associations[
        (associations["analysis_type"] == "within_template_rank")
        & (associations["feature"] == "peptide_position_normalized")
        & (associations["is_primary"] == True)  # noqa: E712
    ]
    assert float(peptide_position["q_value"].min()) == pytest.approx(
        0.7852147852147852
    )

    peptide = associations[
        (associations["analysis_type"] == "equal_template_category_mean")
        & (associations["outcome"] == "absolute_error_change")
        & (associations["feature"] == "single_point_region")
        & (associations["category"] == "Peptide")
    ].set_index("model_or_contrast")
    expected = {
        "BA-DDG": (0.324322, 0.118752, 0.621232),
        "H3-DDG": (0.325777, 0.054314, 0.674982),
        "Prompt-DDG": (0.021515, -0.044058, 0.084316),
        "RDE-PPI": (0.079603, -0.074792, 0.229384),
    }
    for model, (estimate, ci_low, ci_high) in expected.items():
        assert float(peptide.loc[model, "estimate"]) == pytest.approx(
            estimate,
            abs=5e-7,
        )
        assert float(peptide.loc[model, "ci95_low"]) == pytest.approx(
            ci_low,
            abs=5e-7,
        )
        assert float(peptide.loc[model, "ci95_high"]) == pytest.approx(
            ci_high,
            abs=5e-7,
        )
