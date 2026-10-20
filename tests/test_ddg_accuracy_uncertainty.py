import numpy as np
import pytest

from ddg_accuracy_uncertainty import (
    bootstrap_metric_samples,
    calculate_metrics,
    percentile_interval,
    williams_test,
)


def test_calculate_metrics() -> None:
    pred = np.array([1.0, 2.0, 3.0, 5.0])
    exp = np.array([1.0, 2.0, 4.0, 4.0])

    pearson_r, rmse, mae = calculate_metrics(pred, exp)

    assert pearson_r == pytest.approx(np.corrcoef(pred, exp)[0, 1])
    assert rmse == pytest.approx(np.sqrt(0.5))
    assert mae == pytest.approx(0.5)


def test_bootstrap_metric_samples_use_paired_indices() -> None:
    pred = np.array([1.0, 2.0, 3.0, 4.0])
    exp = np.array([2.0, 4.0, 6.0, 8.0])
    indices = np.array([[0, 1, 2, 3], [3, 2, 1, 0], [0, 0, 2, 3]])

    pearson, rmse, mae = bootstrap_metric_samples(pred, exp, indices)

    assert pearson == pytest.approx(np.ones(3))
    expected_diff = pred[indices] - exp[indices]
    assert rmse == pytest.approx(np.sqrt(np.mean(expected_diff**2, axis=1)))
    assert mae == pytest.approx(np.mean(np.abs(expected_diff), axis=1))


def test_percentile_interval() -> None:
    values = np.arange(101, dtype=float)
    assert percentile_interval(values) == pytest.approx((2.5, 97.5))


def test_williams_test_is_symmetric_when_comparisons_are_swapped() -> None:
    forward_t, forward_df, forward_p = williams_test(0.75, 0.72, 0.95, 438)
    reverse_t, reverse_df, reverse_p = williams_test(0.72, 0.75, 0.95, 438)

    assert reverse_t == pytest.approx(-forward_t)
    assert reverse_df == forward_df == 435
    assert reverse_p == pytest.approx(forward_p)


def test_williams_test_matches_rde_mutant_comparison() -> None:
    statistic, degrees_of_freedom, p_value = williams_test(
        r_ensemble=0.745681,
        r_single=0.734038,
        r_between_predictions=0.996491,
        n=438,
    )

    assert statistic == pytest.approx(4.4089, abs=2e-3)
    assert degrees_of_freedom == 435
    assert p_value == pytest.approx(1.31e-5, rel=0.02)
