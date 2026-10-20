"""Regression tests for paired-bootstrap correlation confidence intervals."""

import numpy as np

from helpers.correlation_uncertainty import (
    paired_correlation_bootstrap,
    stable_cell_seed,
)


def test_stable_cell_seed_is_deterministic_and_cell_specific() -> None:
    assert stable_cell_seed(7, "A", "pearson") == stable_cell_seed(7, "A", "pearson")
    assert stable_cell_seed(7, "A", "pearson") != stable_cell_seed(7, "B", "pearson")


def test_perfect_pearson_relationship_has_unit_interval() -> None:
    result = paired_correlation_bootstrap(
        [1, 2, 3, 4, 5],
        [2, 4, 6, 8, 10],
        method="pearson",
        replicates=500,
        seed=11,
    )
    assert result["status"] == "ok"
    assert np.isclose(result["estimate"], 1.0)
    assert np.isclose(result["ci_low"], 1.0)
    assert np.isclose(result["ci_high"], 1.0)


def test_spearman_interval_handles_ties() -> None:
    result = paired_correlation_bootstrap(
        [1, 1, 2, 3, 4],
        [2, 2, 3, 5, 8],
        method="spearman",
        replicates=500,
        seed=19,
    )
    assert result["status"] == "ok"
    assert np.isclose(result["estimate"], 1.0)
    assert result["bootstrap_replicates_valid"] <= 500


def test_small_template_keeps_point_estimate_but_not_interval() -> None:
    result = paired_correlation_bootstrap(
        [1, 2, 3],
        [1, 3, 2],
        method="pearson",
        replicates=100,
        seed=23,
    )
    assert result["status"] == "not_estimable_n_lt_4"
    assert np.isfinite(result["estimate"])
    assert np.isnan(result["ci_low"])
    assert result["bootstrap_replicates_valid"] == 0


def test_constant_input_is_reported_explicitly() -> None:
    result = paired_correlation_bootstrap(
        [1, 1, 1, 1],
        [1, 2, 3, 4],
        method="pearson",
        replicates=100,
        seed=29,
    )
    assert result["status"] == "not_estimable_constant_input"
    assert np.isnan(result["estimate"])
