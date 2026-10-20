"""Deterministic paired-bootstrap confidence intervals for correlations."""

import hashlib
from collections.abc import Iterable

import numpy as np
from scipy.stats import rankdata


def stable_cell_seed(base_seed: int, *keys: object) -> int:
    """Derive a reproducible NumPy seed from a base seed and cell identifiers.

    Args:
        base_seed: User-facing seed recorded with the analysis.
        *keys: Values that uniquely identify one analysis cell.

    Returns:
        An unsigned 32-bit integer suitable for ``numpy.random.default_rng``.
    """
    payload = "\x1f".join(str(value) for value in (base_seed, *keys)).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _rowwise_pearson(x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    """Return one Pearson correlation for every row of two equal 2-D arrays."""
    x_centered = x_values - x_values.mean(axis=1, keepdims=True)
    y_centered = y_values - y_values.mean(axis=1, keepdims=True)
    numerator = np.sum(x_centered * y_centered, axis=1)
    denominator = np.sqrt(
        np.sum(x_centered**2, axis=1) * np.sum(y_centered**2, axis=1)
    )
    correlations = np.full(x_values.shape[0], np.nan, dtype=float)
    valid = denominator > 0
    correlations[valid] = numerator[valid] / denominator[valid]
    return correlations


def _prepare_pairs(
    predicted: Iterable[float], experimental: Iterable[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Convert paired observations to finite one-dimensional float arrays."""
    predicted_array = np.asarray(list(predicted), dtype=float)
    experimental_array = np.asarray(list(experimental), dtype=float)
    if predicted_array.shape != experimental_array.shape:
        raise ValueError("predicted and experimental arrays must have the same shape")
    if predicted_array.ndim != 1:
        raise ValueError("predicted and experimental inputs must be one-dimensional")
    finite = np.isfinite(predicted_array) & np.isfinite(experimental_array)
    return predicted_array[finite], experimental_array[finite]


def paired_correlation_bootstrap(
    predicted: Iterable[float],
    experimental: Iterable[float],
    *,
    method: str,
    replicates: int,
    seed: int,
    minimum_n: int = 4,
    confidence_level: float = 0.95,
) -> dict[str, float | int | str]:
    """Estimate a correlation and its paired nonparametric bootstrap interval.

    Each bootstrap replicate resamples paired prediction/experiment rows within
    the analysis cell. Replicates with a constant resampled vector have an
    undefined correlation and are excluded from the percentile interval.

    Args:
        predicted: Predicted values for one template/model/regime cell.
        experimental: Experimental values paired with ``predicted``.
        method: ``"pearson"`` or ``"spearman"``.
        replicates: Number of bootstrap resamples.
        seed: NumPy random seed for this analysis cell.
        minimum_n: Smallest sample size for which an interval is estimated.
        confidence_level: Central percentile interval width.

    Returns:
        A dictionary containing the point estimate, interval bounds, finite
        replicate count, input sample size, and an explicit status. NaN bounds
        mean the interval is not estimable.
    """
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie between zero and one")

    x_values, y_values = _prepare_pairs(predicted, experimental)
    n_observations = len(x_values)
    result: dict[str, float | int | str] = {
        "estimate": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "n_observations": n_observations,
        "bootstrap_replicates_requested": replicates,
        "bootstrap_replicates_valid": 0,
        "bootstrap_seed": seed,
        "status": "not_estimable",
    }

    if n_observations >= 2 and np.unique(x_values).size >= 2 and np.unique(y_values).size >= 2:
        point_x = rankdata(x_values) if method == "spearman" else x_values
        point_y = rankdata(y_values) if method == "spearman" else y_values
        result["estimate"] = float(
            _rowwise_pearson(point_x[None, :], point_y[None, :])[0]
        )

    if n_observations < minimum_n:
        result["status"] = f"not_estimable_n_lt_{minimum_n}"
        return result
    if np.unique(x_values).size < 2 or np.unique(y_values).size < 2:
        result["status"] = "not_estimable_constant_input"
        return result

    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(
        0, n_observations, size=(replicates, n_observations), endpoint=False
    )
    sampled_x = x_values[sample_indices]
    sampled_y = y_values[sample_indices]
    if method == "spearman":
        sampled_x = rankdata(sampled_x, axis=1)
        sampled_y = rankdata(sampled_y, axis=1)

    bootstrap_correlations = _rowwise_pearson(sampled_x, sampled_y)
    finite_correlations = bootstrap_correlations[np.isfinite(bootstrap_correlations)]
    result["bootstrap_replicates_valid"] = len(finite_correlations)
    if not len(finite_correlations):
        result["status"] = "not_estimable_no_finite_bootstrap_replicates"
        return result

    tail_probability = (1.0 - confidence_level) / 2.0
    result["ci_low"] = float(np.quantile(finite_correlations, tail_probability))
    result["ci_high"] = float(np.quantile(finite_correlations, 1.0 - tail_probability))
    result["status"] = "ok"
    return result
