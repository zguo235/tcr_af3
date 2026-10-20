"""Bootstrap uncertainty and dependent-correlation tests for manuscript Table 2.

The analysis uses the common 438-mutation ATLAS cohort scored by four
structure-based models and the sequence-only PPAC baseline. It writes
confidence intervals for every Table 2 metric and Williams tests comparing
single AF3 structures with arithmetic-mean AF3 ensembles.

Usage:
    conda run -n bioemu1p3 python scripts/ddg_accuracy_uncertainty.py
"""

import argparse
import sys
from pathlib import Path

from helpers.paths import DATA_ROOT, OUTPUT_ROOT, SUPPLEMENTARY_ROOT

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "ddg_predictions" / "atlas_af_eval"))

from aggregate_results import (  # noqa: E402
    ensemble_aggregate,
    load_template_map,
    mutant_ranking_csv,
    parse_ddg_exp,
    wt_ranking_csv,
)

MODELS = [
    ("RDE-PPI", DATA_ROOT / "ddg_predictions" / "atlas_af_eval" / "results"),
    ("Prompt-DDG", DATA_ROOT / "ddg_predictions" / "atlas_promptddg_eval" / "results"),
    ("BA-DDG", DATA_ROOT / "ddg_predictions" / "atlas_baddg_eval" / "results"),
    ("H3-DDG", DATA_ROOT / "ddg_predictions" / "atlas_h3ddg_eval" / "results"),
]

PPAC_RESULTS = DATA_ROOT / "ddg_predictions" / "atlas_ppac_eval" / "results"

# Table 2 order: (row label, structural input, aggregation, internal key).
REGIMES = [
    ("1", "PDB_template", "single", "PDB_template:single"),
    ("2", "wt_af3_single", "single", "wt_af3_single:single"),
    ("3a", "wt_af3_multi", "mean", "wt_af3_multi:mean"),
    ("3b", "wt_af3_multi", "median", "wt_af3_multi:median"),
    ("3c", "wt_af3_multi", "rank_weighted_mean", "wt_af3_multi:rank_weighted_mean"),
    ("4", "mutant_af3_single", "single", "mutant_af3_single:single"),
    ("5a", "mutant_af3_multi", "mean", "mutant_af3_multi:mean"),
    ("5b", "mutant_af3_multi", "median", "mutant_af3_multi:median"),
    (
        "5c",
        "mutant_af3_multi",
        "rank_weighted_mean",
        "mutant_af3_multi:rank_weighted_mean",
    ),
]

COMPARISONS = [
    ("WT", "wt_af3_single:single", "wt_af3_multi:mean"),
    ("mutant", "mutant_af3_single:single", "mutant_af3_multi:mean"),
]


def calculate_metrics(pred: np.ndarray, exp: np.ndarray) -> tuple[float, float, float]:
    """Return Pearson r, RMSE, and MAE for paired one-dimensional arrays."""
    if pred.ndim != 1 or exp.ndim != 1 or pred.shape != exp.shape:
        raise ValueError("pred and exp must be paired one-dimensional arrays")
    if len(pred) < 3:
        raise ValueError("at least three paired observations are required")
    diff = pred - exp
    pearson_r = float(np.corrcoef(pred, exp)[0, 1])
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    return pearson_r, rmse, mae


def bootstrap_metric_samples(
    pred: np.ndarray,
    exp: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate metric values for shared paired bootstrap resamples."""
    if bootstrap_indices.ndim != 2:
        raise ValueError("bootstrap_indices must be a two-dimensional array")
    pred_boot = pred[bootstrap_indices]
    exp_boot = exp[bootstrap_indices]

    pred_centered = pred_boot - pred_boot.mean(axis=1, keepdims=True)
    exp_centered = exp_boot - exp_boot.mean(axis=1, keepdims=True)
    numerator = np.sum(pred_centered * exp_centered, axis=1)
    denominator = np.sqrt(
        np.sum(pred_centered**2, axis=1) * np.sum(exp_centered**2, axis=1)
    )
    pearson = numerator / denominator

    diff = pred_boot - exp_boot
    rmse = np.sqrt(np.mean(diff**2, axis=1))
    mae = np.mean(np.abs(diff), axis=1)
    return pearson, rmse, mae


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    """Return the two-sided 95% percentile interval."""
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def williams_test(
    r_ensemble: float,
    r_single: float,
    r_between_predictions: float,
    n: int,
) -> tuple[float, int, float]:
    """Williams test for two dependent correlations sharing one variable.

    The sign of the statistic follows ``r_ensemble - r_single``.
    """
    if n <= 3:
        raise ValueError("Williams test requires n > 3")
    determinant = (
        1
        + 2 * r_ensemble * r_single * r_between_predictions
        - r_ensemble**2
        - r_single**2
        - r_between_predictions**2
    )
    mean_r = (r_ensemble + r_single) / 2
    denominator = (
        2 * ((n - 1) / (n - 3)) * determinant
        + mean_r**2 * (1 - r_between_predictions) ** 3
    )
    if denominator <= 0:
        raise ValueError("invalid correlation matrix for Williams test")
    statistic = (r_ensemble - r_single) * np.sqrt(
        ((n - 1) * (1 + r_between_predictions)) / denominator
    )
    degrees_of_freedom = n - 3
    p_value = 2 * student_t.sf(abs(statistic), degrees_of_freedom)
    return float(statistic), degrees_of_freedom, float(p_value)


def _prediction_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Select and standardize one prediction column."""
    return (
        df[["row_idx", column]]
        .drop_duplicates(subset="row_idx")
        .rename(columns={column: "ddg_pred"})
    )


def load_model_predictions(
    results_dir: Path,
    template_map: dict[int, str],
) -> dict[str, pd.DataFrame]:
    """Load all nine Table 2 prediction regimes for one model."""
    a1 = pd.read_csv(results_dir / "approach1.csv")
    a2 = pd.read_csv(results_dir / "approach2.csv")
    a3 = pd.read_csv(results_dir / "approach3.csv")
    a4 = pd.read_csv(results_dir / "approach4.csv")
    a5 = pd.read_csv(results_dir / "approach5.csv")
    for frame in (a3, a5):
        frame["ddg_exp"] = frame["ddg_exp"].map(parse_ddg_exp)

    mutant_ensemble = ensemble_aggregate(a3, mutant_ranking_csv, template_map)
    wt_ensemble = ensemble_aggregate(a5, wt_ranking_csv, template_map)
    return {
        "PDB_template:single": _prediction_column(a1, "ddg_pred"),
        "wt_af3_single:single": _prediction_column(a4, "ddg_pred"),
        "wt_af3_multi:mean": _prediction_column(wt_ensemble, "ddg_pred_mean"),
        "wt_af3_multi:median": _prediction_column(wt_ensemble, "ddg_pred_median"),
        "wt_af3_multi:rank_weighted_mean": _prediction_column(
            wt_ensemble, "ddg_pred_rankwt"
        ),
        "mutant_af3_single:single": _prediction_column(a2, "ddg_pred"),
        "mutant_af3_multi:mean": _prediction_column(mutant_ensemble, "ddg_pred_mean"),
        "mutant_af3_multi:median": _prediction_column(
            mutant_ensemble, "ddg_pred_median"
        ),
        "mutant_af3_multi:rank_weighted_mean": _prediction_column(
            mutant_ensemble, "ddg_pred_rankwt"
        ),
    }


def load_ppac_predictions(results_dir: Path = PPAC_RESULTS) -> pd.DataFrame:
    """Load PPAC's single sequence-only prediction for each ATLAS row.

    PPAC does not consume a structural input, so only one prediction is added
    to the uncertainty table and it is excluded from structural-regime and
    single-versus-ensemble comparisons.
    """
    frame = pd.read_csv(results_dir / "approach1.csv")
    if set(frame["source"].dropna().unique()) != {"sequence_only"}:
        raise ValueError("expected PPAC predictions to use source=sequence_only")
    return _prediction_column(frame, "ddg_pred")


def _common_cohort() -> pd.DataFrame:
    """Return the canonical 438-row experimental cohort in stable row order."""
    reference = pd.read_csv(MODELS[0][1] / "approach1.csv")
    reference["ddg_exp"] = reference["ddg_exp"].map(parse_ddg_exp)
    cohort = (
        reference[["row_idx", "ddg_exp"]]
        .drop_duplicates(subset="row_idx")
        .sort_values("row_idx")
        .reset_index(drop=True)
    )
    if len(cohort) != 438 or cohort.isna().any().any():
        raise ValueError(f"expected complete 438-row cohort, found {len(cohort)} rows")
    return cohort


def analyze(
    bootstrap_replicates: int = 10000,
    seed: int = 20260722,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return uncertainty and single-versus-ensemble comparison tables."""
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")

    cohort = _common_cohort()
    exp = cohort["ddg_exp"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    bootstrap_indices = rng.integers(
        0, len(cohort), size=(bootstrap_replicates, len(cohort))
    )

    template_map = load_template_map()
    predictions: dict[tuple[str, str], np.ndarray] = {}
    bootstrap_pearson: dict[tuple[str, str], np.ndarray] = {}
    interval_rows: list[dict[str, object]] = []

    for model_name, results_dir in MODELS:
        model_predictions = load_model_predictions(results_dir, template_map)
        for table_row, regime, aggregation, key in REGIMES:
            indexed = model_predictions[key].set_index("row_idx")["ddg_pred"]
            aligned = cohort["row_idx"].map(indexed)
            if aligned.isna().any():
                missing = cohort.loc[aligned.isna(), "row_idx"].tolist()
                raise ValueError(
                    f"{model_name} {key} is missing predictions for rows {missing[:10]}"
                )
            pred = aligned.to_numpy(dtype=float)
            pearson_r, rmse, mae = calculate_metrics(pred, exp)
            pearson_boot, rmse_boot, mae_boot = bootstrap_metric_samples(
                pred, exp, bootstrap_indices
            )
            pearson_low, pearson_high = percentile_interval(pearson_boot)
            rmse_low, rmse_high = percentile_interval(rmse_boot)
            mae_low, mae_high = percentile_interval(mae_boot)

            predictions[(model_name, key)] = pred
            bootstrap_pearson[(model_name, key)] = pearson_boot
            interval_rows.append(
                {
                    "model": model_name,
                    "table_row": table_row,
                    "structural_input": regime,
                    "aggregation": aggregation,
                    "n_mutations": len(cohort),
                    "pearson_r": pearson_r,
                    "pearson_ci95_low": pearson_low,
                    "pearson_ci95_high": pearson_high,
                    "rmse_kcal_per_mol": rmse,
                    "rmse_ci95_low": rmse_low,
                    "rmse_ci95_high": rmse_high,
                    "mae_kcal_per_mol": mae,
                    "mae_ci95_low": mae_low,
                    "mae_ci95_high": mae_high,
                    "bootstrap_replicates": bootstrap_replicates,
                }
            )

    ppac_indexed = load_ppac_predictions().set_index("row_idx")["ddg_pred"]
    ppac_aligned = cohort["row_idx"].map(ppac_indexed)
    if ppac_aligned.isna().any():
        missing = cohort.loc[ppac_aligned.isna(), "row_idx"].tolist()
        raise ValueError(f"PPAC is missing predictions for rows {missing[:10]}")
    ppac_pred = ppac_aligned.to_numpy(dtype=float)
    ppac_pearson, ppac_rmse, ppac_mae = calculate_metrics(ppac_pred, exp)
    ppac_pearson_boot, ppac_rmse_boot, ppac_mae_boot = bootstrap_metric_samples(
        ppac_pred, exp, bootstrap_indices
    )
    ppac_pearson_low, ppac_pearson_high = percentile_interval(ppac_pearson_boot)
    ppac_rmse_low, ppac_rmse_high = percentile_interval(ppac_rmse_boot)
    ppac_mae_low, ppac_mae_high = percentile_interval(ppac_mae_boot)
    interval_rows.append(
        {
            "model": "PPAC",
            "table_row": "S",
            "structural_input": "sequence_only",
            "aggregation": "not_applicable",
            "n_mutations": len(cohort),
            "pearson_r": ppac_pearson,
            "pearson_ci95_low": ppac_pearson_low,
            "pearson_ci95_high": ppac_pearson_high,
            "rmse_kcal_per_mol": ppac_rmse,
            "rmse_ci95_low": ppac_rmse_low,
            "rmse_ci95_high": ppac_rmse_high,
            "mae_kcal_per_mol": ppac_mae,
            "mae_ci95_low": ppac_mae_low,
            "mae_ci95_high": ppac_mae_high,
            "bootstrap_replicates": bootstrap_replicates,
        }
    )

    comparison_rows: list[dict[str, object]] = []
    for model_name, _ in MODELS:
        for scaffold, single_key, ensemble_key in COMPARISONS:
            single = predictions[(model_name, single_key)]
            ensemble = predictions[(model_name, ensemble_key)]
            r_single = calculate_metrics(single, exp)[0]
            r_ensemble = calculate_metrics(ensemble, exp)[0]
            r_between = float(np.corrcoef(ensemble, single)[0, 1])
            statistic, degrees_of_freedom, p_value = williams_test(
                r_ensemble, r_single, r_between, len(cohort)
            )
            single_boot = bootstrap_pearson[(model_name, single_key)]
            ensemble_boot = bootstrap_pearson[(model_name, ensemble_key)]
            single_low, single_high = percentile_interval(single_boot)
            ensemble_low, ensemble_high = percentile_interval(ensemble_boot)
            delta_low, delta_high = percentile_interval(ensemble_boot - single_boot)
            comparison_rows.append(
                {
                    "model": model_name,
                    "scaffold": scaffold,
                    "single_regime": single_key.split(":", maxsplit=1)[0],
                    "ensemble_regime": ensemble_key.split(":", maxsplit=1)[0],
                    "ensemble_aggregation": "mean",
                    "n_mutations": len(cohort),
                    "pearson_single": r_single,
                    "pearson_single_ci95_low": single_low,
                    "pearson_single_ci95_high": single_high,
                    "pearson_ensemble": r_ensemble,
                    "pearson_ensemble_ci95_low": ensemble_low,
                    "pearson_ensemble_ci95_high": ensemble_high,
                    "pearson_delta": r_ensemble - r_single,
                    "pearson_delta_ci95_low": delta_low,
                    "pearson_delta_ci95_high": delta_high,
                    "single_ensemble_prediction_r": r_between,
                    "williams_t": statistic,
                    "degrees_of_freedom": degrees_of_freedom,
                    "p_value_two_sided": p_value,
                    "bootstrap_replicates": bootstrap_replicates,
                }
            )

    return pd.DataFrame(interval_rows), pd.DataFrame(comparison_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--intervals-out",
        default=str(
            OUTPUT_ROOT / "supplementary_tables" / "S10.csv"
        ),
    )
    parser.add_argument(
        "--comparisons-out",
        default=str(
            OUTPUT_ROOT / "supplementary_tables" / "S11.csv"
        ),
    )
    args = parser.parse_args()

    intervals, comparisons = analyze(args.bootstrap_replicates, args.seed)
    intervals_path = Path(args.intervals_out)
    comparisons_path = Path(args.comparisons_out)
    intervals_path.parent.mkdir(parents=True, exist_ok=True)
    comparisons_path.parent.mkdir(parents=True, exist_ok=True)
    intervals.to_csv(intervals_path, index=False)
    comparisons.to_csv(comparisons_path, index=False)

    print(f"Wrote {intervals_path} ({len(intervals)} rows)")
    print(f"Wrote {comparisons_path} ({len(comparisons)} rows)")
    print(
        comparisons[
            [
                "model",
                "scaffold",
                "pearson_single",
                "pearson_ensemble",
                "pearson_delta",
                "p_value_two_sided",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
