#!/usr/bin/env python3
"""Compare ATLAS prediction accuracy by shared template PDB ID with SKEMPI.

Uses existing RDE-PPI and Prompt-DDG predictions, without inference or training.
Outputs include the estimates used in SI Tables S21--S22.
The original five-scaffold numerical checks are retained; the table formatter
selects experimental WT and AF3-WT single, as reported in the manuscript.
"""

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo / "scripts"))
from helpers.correlation_uncertainty import stable_cell_seed  # noqa: E402
from helpers.atlas_reference import digits_only, load_source_by_index  # noqa: E402
from helpers.paths import DATA_ROOT, OUTPUT_ROOT, SUPPLEMENTARY_ROOT  # noqa: E402

models = {
    "RDE-PPI": ("rde", "atlas_af_eval"),
    "Prompt-DDG": ("prompt", "atlas_promptddg_eval"),
}
regimes = {
    "PDB_template": (1, "single", 1, "Experimental WT"),
    "wt_af3_single": (4, "single", 1, "AF3-WT single"),
    "wt_af3_multi": (5, "mean", 100, "AF3-WT ensemble mean"),
    "mutant_af3_single": (2, "single", 1, "AF3-mutant single"),
    "mutant_af3_multi": (3, "mean", 20, "AF3-mutant ensemble mean"),
}
metric_names = ("pearson_r", "rmse_kcal_per_mol", "mae_kcal_per_mol")
group_names = {
    "all": "All ATLAS records",
    "shared_pdb_id": "Shared template PDB ID",
    "no_shared_pdb_id": "No shared template PDB ID",
}


def metric_values(observed, predicted):
    """Return pooled (Pearson r, RMSE, MAE) for equal-length finite 1-D arrays.

    Correlation is NaN when either variable is constant. Errors use kcal/mol.
    """
    error = predicted - observed
    correlation = (
        float(pearsonr(observed, predicted).statistic)
        if np.ptp(observed) > 0 and np.ptp(predicted) > 0
        else np.nan
    )
    return np.array([correlation, np.sqrt(np.mean(error**2)), np.mean(abs(error))])


def template_statistics(observed, predicted, template_codes):
    """Build per-template sufficient statistics for a single predictor/scaffold.

    Inputs are equally sized 1-D arrays; template_codes are consecutive integers
    starting at zero. Return a (n_templates, 8) array containing count, sums of
    observed/predicted values, their squares/product, and squared/absolute error.
    """
    error = predicted - observed
    values = (
        np.ones(len(observed)), observed, predicted, observed**2,
        predicted**2, observed * predicted, error**2, abs(error),
    )
    return np.column_stack([
        np.bincount(template_codes, weights=value) for value in values
    ])


def metrics_from_template_counts(statistics, template_counts):
    """Compute pooled metrics for bootstrap template multiplicities.

    statistics has shape (n_templates, 8); template_counts has shape
    (n_resamples, n_templates). Return (n_resamples, 3) Pearson/RMSE/MAE values.
    A template occurring twice contributes every one of its records twice.
    Correlation is NaN for a resample with zero variance in either variable.
    """
    totals = template_counts @ statistics
    count, sum_y, sum_p, sum_yy, sum_pp, sum_yp, sum_se, sum_ae = totals.T
    covariance = sum_yp - sum_y * sum_p / count
    variance_y = np.maximum(sum_yy - sum_y**2 / count, 0)
    variance_p = np.maximum(sum_pp - sum_p**2 / count, 0)
    denominator = np.sqrt(variance_y * variance_p)
    correlations = np.full(len(count), np.nan)
    np.divide(covariance, denominator, out=correlations, where=denominator > 0)
    correlations = np.clip(correlations, -1, 1)
    return np.column_stack((correlations, np.sqrt(sum_se / count), sum_ae / count))


def add_intervals(row, estimates, resamples, names):
    """Add estimates, percentile 95% intervals, and valid resample counts to row.

    estimates has one entry per name; resamples has shape (n_resamples, n_names).
    Invalid correlations are counted explicitly; fewer than 90% valid resamples
    raises an error rather than yielding an unreliable interval. Mutates row.
    """
    for index, name in enumerate(names):
        values = resamples[:, index]
        values = values[np.isfinite(values)]
        if len(values) < 0.9 * len(resamples):
            raise ValueError(f"Too few finite bootstrap replicates for {name}")
        low, high = np.quantile(values, [0.025, 0.975])
        row[name] = float(estimates[index])
        row[f"{name}_ci95_low"] = float(low)
        row[f"{name}_ci95_high"] = float(high)
        row[f"{name}_bootstrap_valid"] = len(values)


def load_cohort(workspace, repository):
    """Load the canonical ATLAS cohort and join verified signed predictions.

    Return (cohort, source_paths, template_counts, checks). cohort has one row per
    ATLAS record, all original S6 fields, S16 metadata, group labels, and ten
    prediction columns. Source values remain strings except analysis columns.
    Joins must cover the same 438 records in every model/scaffold setting.
    """
    paths = {
        "atlas": SUPPLEMENTARY_ROOT / "S6.csv",
        "metadata": SUPPLEMENTARY_ROOT / "S16.csv",
        "published_metrics": SUPPLEMENTARY_ROOT / "S10.csv",
        "skempi": workspace / "reviewer3/data/skempi2/skempi_v2.csv",
        "unicode_reference": repository / "ATLAS/www/scoring/Mutants_052915.tsv",
        "prediction_index": repository / "ATLAS_af/index.csv",
    }
    atlas = pd.read_csv(paths["atlas"], dtype=str, keep_default_na=False)
    metadata = pd.read_csv(paths["metadata"], dtype={"atlas_Index": str})
    skempi = pd.read_csv(paths["skempi"], sep=";", dtype=str, keep_default_na=False)
    prediction_index = pd.read_csv(paths["prediction_index"], dtype={"atlas_Index": str}).set_index("row_idx")
    if len(atlas) != 438 or atlas["Index"].duplicated().any():
        raise ValueError("S6 must contain 438 unique ATLAS record IDs")
    if len(metadata) != 438 or metadata["row_idx"].duplicated().any():
        raise ValueError("S16 must contain 438 unique prediction row IDs")
    columns = [
        "atlas_Index", "row_idx", "template_PDB", "mutstr", "mutations",
        "mutation_regions", "mutation_components", "num_muts", "ddg_exp",
    ]
    cohort = atlas.merge(
        metadata[columns], left_on="Index", right_on="atlas_Index",
        how="outer", validate="one_to_one", suffixes=("", "_metadata"),
        indicator=True,
    )
    if not cohort["_merge"].eq("both").all():
        raise ValueError("S6 and S16 record IDs differ")
    if not cohort["template_PDB"].eq(cohort["template_PDB_metadata"]).all():
        raise ValueError("S6 and S16 template assignments differ")
    cohort = cohort.drop(columns=["atlas_Index", "template_PDB_metadata", "_merge"])
    # S6 retains historical Unicode corruption; validate corrected scored values
    # against the existing authoritative source, without modifying the S6 cells.
    source_ddg = load_source_by_index()
    cohort["s6_ddg_text_has_unicode_error"] = cohort["Delta_DeltaG_kcal_per_mol"].str.contains("?", regex=False)
    cohort["analysis_reference_ddg_text"] = cohort["Delta_DeltaG_kcal_per_mol"]
    for index, row in cohort[cohort["s6_ddg_text_has_unicode_error"]].iterrows():
        source_mutation, clean_ddg = source_ddg[row["Index"]]
        if source_mutation != row["TCR_mut"] or digits_only(clean_ddg) != digits_only(row["Delta_DeltaG_kcal_per_mol"]):
            raise ValueError(f"Unicode reference mismatch for ATLAS {row['Index']}")
        cohort.loc[index, "analysis_reference_ddg_text"] = clean_ddg
    original_ddg = cohort["analysis_reference_ddg_text"].map(
        lambda value: float(value.split("±")[0].replace("−", "-").strip())
    )
    np.testing.assert_allclose(original_ddg, cohort["ddg_exp"], atol=1e-6, rtol=0)
    skempi_ids = set(skempi["#Pdb"].str.split("_").str[0].str.upper())
    atlas_ids = set(cohort["template_PDB"])
    shared_ids = atlas_ids & skempi_ids
    if (len(atlas_ids), len(skempi_ids), len(shared_ids)) != (37, 345, 15):
        raise ValueError("Template counts differ from the previously validated audit")
    cohort["overlap_group"] = np.where(
        cohort["template_PDB"].isin(shared_ids), "shared_pdb_id", "no_shared_pdb_id"
    )
    if cohort["overlap_group"].value_counts().to_dict() != {
        "shared_pdb_id": 356, "no_shared_pdb_id": 82,
    }:
        raise ValueError("Unexpected subgroup sizes")
    cohort = cohort.sort_values(["template_PDB", "row_idx"]).reset_index(drop=True)
    if not cohort["row_idx"].map(prediction_index["atlas_Index"]).eq(cohort["Index"]).all():
        raise ValueError("Prediction index and S6 record IDs differ")
    allowed_excluded_ids = set(prediction_index.index[prediction_index["template_PDB"].eq("1FYT")])
    metadata = metadata.set_index("row_idx")
    checks = {
        "record_join": "438 one-to-one matches", "prediction_inputs": {},
        "s6_unicode_corrupted_record_ids": cohort.loc[cohort["s6_ddg_text_has_unicode_error"], "Index"].tolist(),
        "unicode_resolution": "Original S6 fields preserved; S16 scored targets verified against the authoritative Unicode reference",
    }
    for model, (slug, folder) in models.items():
        for regime, (approach, _, expected_samples, _) in regimes.items():
            name = f"{slug}_{regime}"
            path = repository / "ddg_predictions" / folder / "results" / f"approach{approach}.csv"
            paths[name] = path
            raw = pd.read_csv(path)
            expected_ids = set(cohort["row_idx"])
            extra_ids = set(raw["row_idx"]) - expected_ids
            if expected_ids - set(raw["row_idx"]) or not extra_ids <= allowed_excluded_ids:
                raise ValueError(f"Missing or unexpected prediction IDs: {path}")
            # AF3 predictions also contain the two 1FYT records already excluded
            # from the published 438-record cohort; record this explicit filter.
            raw = raw[raw["row_idx"].isin(expected_ids)].copy()
            if not raw["approach"].eq(approach).all():
                raise ValueError(f"Wrong approach identifier: {path}")
            if raw.duplicated(["row_idx", "source"]).any():
                raise ValueError(f"Repeated scaffold samples: {path}")
            sample_counts = raw.groupby("row_idx").size()
            if not sample_counts.eq(expected_samples).all():
                raise ValueError(f"Unexpected number of samples per variant: {path}")
            truth = pd.to_numeric(raw["ddg_exp"])
            np.testing.assert_allclose(
                truth, raw["row_idx"].map(metadata["ddg_exp"]), atol=1e-6, rtol=0
            )
            sign = -1 if regime.startswith("mutant") else 1
            np.testing.assert_allclose(raw["ddg_pred"], sign * raw["ddg_pred_raw"], atol=1e-12)
            if not np.isfinite(raw[["ddg_pred", "ddg_pred_raw"]]).all().all():
                raise ValueError(f"Nonfinite predictions: {path}")
            # Exactly the arithmetic aggregation used for the manuscript mean rows.
            predictions = raw.groupby("row_idx")["ddg_pred"].mean()
            cohort[name] = cohort["row_idx"].map(predictions)
            if regime in {"PDB_template", "wt_af3_single"}:
                reference = "pdb" if regime == "PDB_template" else "af3_wt"
                np.testing.assert_allclose(
                    cohort[name], cohort["row_idx"].map(metadata[f"{slug}_{reference}_prediction"]),
                    atol=1e-12, rtol=0,
                )
            checks["prediction_inputs"][name] = {
                "records": len(sample_counts), "samples_per_record": expected_samples,
                "total_predictions": len(raw), "target_and_sign_checks": "passed",
                "excluded_existing_1FYT_row_ids": sorted(extra_ids),
            }
    template_counts = cohort.groupby(["overlap_group", "template_PDB"]).size().reset_index(name="n_records")
    checks["template_sets"] = {
        "atlas_total": len(atlas_ids), "skempi_total": len(skempi_ids),
        "shared": sorted(shared_ids), "atlas_only": sorted(atlas_ids - skempi_ids),
        "skempi_only": sorted(skempi_ids - atlas_ids),
    }
    return cohort, paths, template_counts, checks


def validate_full_cohort(cohort, reference_path):
    """Recompute ten model/scaffold cells and compare all 30 point metrics to S10.

    Return the largest absolute discrepancy; a discrepancy over 1e-9 raises.
    This compares point estimates only; this audit uses template-block intervals.
    """
    published = pd.read_csv(reference_path)
    differences = []
    for model, (slug, _) in models.items():
        for regime, (_, aggregation, _, _) in regimes.items():
            reference = published[
                published["model"].eq(model) & published["structural_input"].eq(regime)
                & published["aggregation"].eq(aggregation)
            ]
            if len(reference) != 1 or int(reference.iloc[0]["n_mutations"]) != 438:
                raise ValueError(f"Missing published reference for {model} {regime}")
            computed = metric_values(cohort["ddg_exp"].to_numpy(), cohort[f"{slug}_{regime}"].to_numpy())
            expected = reference.iloc[0][list(metric_names)].to_numpy(dtype=float)
            np.testing.assert_allclose(computed, expected, atol=1e-9, rtol=0)
            differences.extend(abs(computed - expected))
    return float(max(differences))


def evaluate_groups(cohort, replicates, seed):
    """Calculate group accuracy, paired scaffold changes, and group contrasts.

    Return three long-form DataFrames. Resample whole templates independently
    within each group, using the same draws across models/scaffolds. Primary
    metrics weight records equally; a companion absolute-error-change estimate
    weights templates equally. All contrasts are descriptive, not causal.
    """
    accuracy_rows, change_rows, contrast_rows = [], [], []
    saved = {}
    for group in group_names:
        frame = cohort if group == "all" else cohort[cohort["overlap_group"].eq(group)]
        codes, templates = pd.factorize(frame["template_PDB"], sort=True)
        n_templates = len(templates)
        rng = np.random.default_rng(stable_cell_seed(seed, group, "template_blocks"))
        multiplicities = rng.multinomial(n_templates, np.full(n_templates, 1 / n_templates), size=replicates)
        observed = frame["ddg_exp"].to_numpy()
        for model, (slug, _) in models.items():
            values = {}
            for regime, (_, aggregation, _, _) in regimes.items():
                predicted = frame[f"{slug}_{regime}"].to_numpy()
                statistics = template_statistics(observed, predicted, codes)
                estimates = metric_values(observed, predicted)
                resamples = metrics_from_template_counts(statistics, multiplicities)
                # Also validates pooled statistics against the independent SciPy calculation.
                np.testing.assert_allclose(
                    metrics_from_template_counts(statistics, np.ones((1, n_templates)))[0],
                    estimates, atol=1e-12, rtol=0,
                )
                template_mae = statistics[:, 7] / statistics[:, 0]
                values[regime] = (estimates, resamples, template_mae)
                row = {
                    "overlap_group": group, "model": model, "structural_input": regime,
                    "aggregation": aggregation, "n_records": len(frame), "n_templates": n_templates,
                    "bootstrap_replicates": replicates,
                }
                add_intervals(row, estimates, resamples, metric_names)
                accuracy_rows.append(row)
            baseline, baseline_resamples, baseline_template_mae = values["PDB_template"]
            for regime in list(regimes)[1:]:
                estimate, resamples, template_mae = values[regime]
                changes = estimate - baseline
                change_resamples = resamples - baseline_resamples
                row = {
                    "overlap_group": group, "model": model, "structural_input": regime,
                    "n_records": len(frame), "n_templates": n_templates,
                    "bootstrap_replicates": replicates,
                }
                add_intervals(row, changes, change_resamples, [f"delta_{name}" for name in metric_names])
                equal_change = template_mae - baseline_template_mae
                equal_resamples = multiplicities @ equal_change / n_templates
                add_intervals(
                    row, [equal_change.mean()], equal_resamples[:, None],
                    ["equal_template_delta_mae_kcal_per_mol"],
                )
                change_rows.append(row)
                saved[(group, model, regime)] = (
                    changes, change_resamples, equal_change.mean(), equal_resamples,
                )
    for model in models:
        for regime in list(regimes)[1:]:
            shared = saved[("shared_pdb_id", model, regime)]
            nonshared = saved[("no_shared_pdb_id", model, regime)]
            row = {
                "model": model, "structural_input": regime,
                "contrast": "no_shared_pdb_id_minus_shared_pdb_id",
                "bootstrap_replicates": replicates,
            }
            add_intervals(
                row, nonshared[0] - shared[0], nonshared[1] - shared[1],
                [f"difference_in_delta_{name}" for name in metric_names],
            )
            add_intervals(
                row, [nonshared[2] - shared[2]], (nonshared[3] - shared[3])[:, None],
                ["difference_in_equal_template_delta_mae_kcal_per_mol"],
            )
            contrast_rows.append(row)
    return pd.DataFrame(accuracy_rows), pd.DataFrame(change_rows), pd.DataFrame(contrast_rows)


def independent_metric_check(cohort, accuracy):
    """Validate every reported accuracy point using Python's standard library.

    Return the maximum discrepancy over the 90 group/model/scaffold metrics.
    A discrepancy above 1e-12 raises; this uses a calculation path independent
    of the vectorized sufficient statistics and SciPy point estimates.
    """
    differences = []
    for _, row in accuracy.iterrows():
        group = row["overlap_group"]
        frame = cohort if group == "all" else cohort[cohort["overlap_group"].eq(group)]
        slug = models[row["model"]][0]
        observed = frame["ddg_exp"].tolist()
        predicted = frame[f"{slug}_{row['structural_input']}"].tolist()
        expected = [
            statistics.correlation(observed, predicted),
            statistics.mean((a - b) ** 2 for a, b in zip(observed, predicted)) ** 0.5,
            statistics.mean(abs(a - b) for a, b in zip(observed, predicted)),
        ]
        differences.extend(abs(float(row[name]) - value) for name, value in zip(metric_names, expected))
    maximum = max(differences)
    if maximum > 1e-12:
        raise ValueError(f"Independent metric discrepancy: {maximum}")
    return maximum


def main():
    """Run the requested subgroup audit and save data, estimates, and provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DATA_ROOT / "frontiersin_revision_v3")
    parser.add_argument("--repository", type=Path, default=DATA_ROOT, help="External data root containing ATLAS and prediction inputs.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_925)
    args = parser.parse_args()
    if args.bootstrap_replicates < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    workspace = args.workspace.resolve()
    output_dir = args.output_dir or OUTPUT_ROOT / "template_overlap"
    cohort, paths, templates, checks = load_cohort(workspace, args.repository.resolve())
    checks["full_cohort_max_metric_difference"] = validate_full_cohort(cohort, paths["published_metrics"])
    print("Cohort joins, prediction coverage/signs, S16 predictions, and S10 metrics verified.", flush=True)
    accuracy, changes, contrasts = evaluate_groups(cohort, args.bootstrap_replicates, args.seed)
    checks["independent_metric_max_difference"] = independent_metric_check(cohort, accuracy)
    output_dir.mkdir(parents=True, exist_ok=True)
    for group, name in (
        ("shared_pdb_id", "atlas_shared_template_records.csv"),
        ("no_shared_pdb_id", "atlas_no_shared_pdb_records.csv"),
    ):
        cohort[cohort["overlap_group"].eq(group)].to_csv(output_dir / name, index=False)
    for frame, name in (
        (templates, "template_groups.csv"), (accuracy, "prediction_accuracy.csv"),
        (changes, "predicted_scaffold_changes.csv"), (contrasts, "between_group_contrasts.csv"),
    ):
        frame.to_csv(output_dir / name, index=False)
    provenance = {
        "question": "ATLAS accuracy and scaffold robustness by shared SKEMPI template PDB ID",
        "bootstrap_unit": "template, with all records retained",
        "bootstrap_replicates": args.bootstrap_replicates, "seed": args.seed,
        "primary_weighting": "equal record", "sensitivity_weighting": "equal template MAE change",
        "sources": {
            key: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for key, path in paths.items()
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "validation": checks,
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Saved subgroup data and analysis: {output_dir}", flush=True)
    print(accuracy.loc[
        ~accuracy["overlap_group"].eq("all") & accuracy["structural_input"].isin(["PDB_template", "wt_af3_single"]),
        ["overlap_group", "model", "structural_input", "n_records", *metric_names],
    ].to_string(index=False))


if __name__ == "__main__":
    main()
