"""Per-template supplementary tables for Pearson r / RMSE / MAE.

For every wild-type TCR-pMHC template in the 4-way intersection (n=438),
compute Pearson r, RMSE, and MAE against ATLAS experimental ΔΔG, for the
five canonical structure-input regimes used in Table 2 (single + ensemble-
mean only), across all four models.

Layout (one row per template):
    columns: template, n_mutants, then 4 models × 5 regimes = 20 metric
    columns labelled `{model}:{regime}` where regime ∈
    {template, wt, wt_multi_mean, mutant, mutant_multi_mean}.

Models (Table 2 column order):
    RDE-PPI / Prompt-DDG / BA-DDG / H3-DDG

Regimes (Table 2 row order, restricted to single + mean aggregators):
    template          = PDB_template (experimental WT)
    wt                = wt_af3_single
    wt_multi_mean     = wt_af3_multi (arithmetic mean over 100 AF3 samples)
    mutant            = mutant_af3_single
    mutant_multi_mean = mutant_af3_multi (arithmetic mean over 20 AF3 samples)

Outputs:
    results/supplementary_tables/S7.csv
    results/supplementary_tables/S8.csv
    results/supplementary_tables/S9.csv
    results/supplementary_tables/S12.csv

RMSE/MAE are in kcal/mol; Pearson is blank when n<3 or any column is
constant. Median and AF3-rank-weighted-mean aggregators are reported in
Table 2 itself but omitted here for compactness — they differ from the
mean aggregator by ≤ 0.04 in Pearson on this benchmark.

Usage:
    python scripts/supp_per_template_metrics.py
"""

import argparse
import sys
from pathlib import Path

from helpers.paths import DATA_ROOT, OUTPUT_ROOT, SUPPLEMENTARY_ROOT

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "ddg_predictions" / "atlas_af_eval"))

from aggregate_results import (  # noqa: E402
    ensemble_aggregate,
    load_template_map,
    mutant_ranking_csv,
    parse_ddg_exp,
    wt_ranking_csv,
)
from helpers.correlation_uncertainty import (  # noqa: E402
    paired_correlation_bootstrap,
    stable_cell_seed,
)

MODELS = [
    ("RDE-PPI",     DATA_ROOT / "ddg_predictions" / "atlas_af_eval"        / "results"),
    ("Prompt-DDG",  DATA_ROOT / "ddg_predictions" / "atlas_promptddg_eval" / "results"),
    ("BA-DDG",      DATA_ROOT / "ddg_predictions" / "atlas_baddg_eval"     / "results"),
    ("H3-DDG",      DATA_ROOT / "ddg_predictions" / "atlas_h3ddg_eval"     / "results"),
]

# (regime, aggregator, short label used as column suffix) — Table 3 row
# order, restricted to single + arithmetic-mean aggregators.
REGIME_AGGREGATORS = [
    ("PDB_template",      "single", "template"),
    ("wt_af3_single",     "single", "wt"),
    ("wt_af3_multi",      "mean",   "wt_multi_mean"),
    ("mutant_af3_single", "single", "mutant"),
    ("mutant_af3_multi",  "mean",   "mutant_multi_mean"),
]


def load_model_predictions(results_dir: Path, template_map: dict[int, str]
                            ) -> dict[tuple[str, str], pd.DataFrame]:
    """Return {(regime, aggregator): DataFrame[row_idx, ddg_pred]} for one model.

    Single-aggregator approaches (1, 2, 4) use `ddg_pred` directly.
    The two ensemble approaches (3, 5) need `ensemble_aggregate` to
    collapse per-sample rows to per-mutant means."""
    a1 = pd.read_csv(results_dir / "approach1.csv")
    a2 = pd.read_csv(results_dir / "approach2.csv")
    a3 = pd.read_csv(results_dir / "approach3.csv")
    a4 = pd.read_csv(results_dir / "approach4.csv")
    a5 = pd.read_csv(results_dir / "approach5.csv")
    for df in (a3, a5):
        df["ddg_exp"] = df["ddg_exp"].map(parse_ddg_exp)
    a3g = ensemble_aggregate(a3, mutant_ranking_csv, template_map)
    a5g = ensemble_aggregate(a5, wt_ranking_csv, template_map)
    return {
        ("PDB_template",      "single"): a1[["row_idx", "ddg_pred"]],
        ("wt_af3_single",     "single"): a4[["row_idx", "ddg_pred"]],
        ("wt_af3_multi",      "mean"):   a5g[["row_idx", "ddg_pred_mean"]].rename(
            columns={"ddg_pred_mean": "ddg_pred"}),
        ("mutant_af3_single", "single"): a2[["row_idx", "ddg_pred"]],
        ("mutant_af3_multi",  "mean"):   a3g[["row_idx", "ddg_pred_mean"]].rename(
            columns={"ddg_pred_mean": "ddg_pred"}),
    }


def metrics(pred: np.ndarray, exp: np.ndarray) -> tuple[float, float, float]:
    """Return (pearson_r, rmse, mae). pearson_r is nan if n<3 or constant."""
    n = len(pred)
    if n == 0:
        return np.nan, np.nan, np.nan
    diff = pred - exp
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    if n >= 3 and len(np.unique(pred)) >= 2 and len(np.unique(exp)) >= 2:
        r, _ = pearsonr(pred, exp)
    else:
        r = np.nan
    return r, rmse, mae


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir",
        default=str(OUTPUT_ROOT / "supplementary_tables"),
        help="Directory for Supplementary Tables S7--S9.",
    )
    ap.add_argument(
        "--ci-out",
        default=str(
            OUTPUT_ROOT / "supplementary_tables" / "S12.csv"
        ),
        help="Long-form Supplementary Table S12 for per-template Pearson CIs.",
    )
    ap.add_argument("--bootstrap-reps", type=int, default=10_000)
    ap.add_argument("--bootstrap-seed", type=int, default=20_260_722)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    template_map = load_template_map()

    # Canonical experimental ΔΔG and template label come from RDE-PPI's
    # approach1.csv (any model would give the same numbers; this also fixes
    # the row_idx universe for the inner join).
    a1_ref = pd.read_csv(MODELS[0][1] / "approach1.csv")
    a1_ref["ddg_exp"] = a1_ref["ddg_exp"].map(parse_ddg_exp)
    exp_lookup = (
        a1_ref[["row_idx", "ddg_exp"]]
        .drop_duplicates(subset="row_idx")
        .set_index("row_idx")["ddg_exp"]
    )
    template_lookup = pd.Series(template_map)

    # Pre-compute every (model, regime, aggregator) prediction column.
    print("Loading + aggregating predictions for 4 models …")
    all_preds: dict[str, dict[tuple[str, str], pd.DataFrame]] = {}
    for name, rd in MODELS:
        all_preds[name] = load_model_predictions(rd, template_map)
        print(f"  {name}: 9 (regime, aggregator) cells loaded")

    # Per-template metric values: results[(metric, template)][f"{model}:{label}"] = value
    results: dict[tuple[str, str], dict[str, float]] = {}
    n_per_template: dict[str, int] = {}
    pearson_ci_rows: list[dict[str, float | int | str]] = []

    for regime, agg, label in REGIME_AGGREGATORS:
        # Inner-join all 4 models on row_idx for this (regime, aggregator).
        merged = pd.DataFrame({"row_idx": list(exp_lookup.index)})
        merged["ddg_exp"] = merged["row_idx"].map(exp_lookup)
        merged["template"] = merged["row_idx"].map(template_lookup)
        for name, _ in MODELS:
            df = all_preds[name][(regime, agg)]
            merged = merged.merge(
                df.rename(columns={"ddg_pred": f"pred_{name}"}),
                on="row_idx", how="inner",
            )
        merged = merged.dropna(subset=["ddg_exp", "template"])
        for name, _ in MODELS:
            merged = merged.dropna(subset=[f"pred_{name}"])

        for tmpl, sub_t in merged.groupby("template", sort=False):
            n = len(sub_t)
            n_per_template[tmpl] = max(n_per_template.get(tmpl, 0), n)
            for metric_name in ("pearson", "rmse", "mae"):
                results.setdefault((metric_name, tmpl), {})
            for name, _ in MODELS:
                r, rmse, mae = metrics(
                    sub_t[f"pred_{name}"].to_numpy(),
                    sub_t["ddg_exp"].to_numpy(),
                )
                col = f"{name}:{label}"
                results[("pearson", tmpl)][col] = r
                results[("rmse",    tmpl)][col] = rmse
                results[("mae",     tmpl)][col] = mae
                cell_seed = stable_cell_seed(
                    args.bootstrap_seed, tmpl, name, label, "pearson"
                )
                uncertainty = paired_correlation_bootstrap(
                    sub_t[f"pred_{name}"].to_numpy(),
                    sub_t["ddg_exp"].to_numpy(),
                    method="pearson",
                    replicates=args.bootstrap_reps,
                    seed=cell_seed,
                )
                pearson_ci_rows.append({
                    "template": tmpl,
                    "n_mutants": n,
                    "model": name,
                    "regime": label,
                    "pearson_r": uncertainty["estimate"],
                    "pearson_ci95_low": uncertainty["ci_low"],
                    "pearson_ci95_high": uncertainty["ci_high"],
                    "bootstrap_replicates_requested": uncertainty[
                        "bootstrap_replicates_requested"
                    ],
                    "bootstrap_replicates_valid": uncertainty[
                        "bootstrap_replicates_valid"
                    ],
                    "bootstrap_seed": uncertainty["bootstrap_seed"],
                    "status": uncertainty["status"],
                })

    # Templates by descending n_mutants. Column order: template, n_mutants,
    # then for each model its 5 regime cells in REGIME_AGGREGATORS order.
    template_order = sorted(n_per_template.keys(),
                             key=lambda t: -n_per_template[t])
    column_order = [
        f"{model}:{label}"
        for model, _ in MODELS
        for _, _, label in REGIME_AGGREGATORS
    ]

    for metric_name, output_name in (("pearson", "S7.csv"),
                                     ("rmse", "S8.csv"),
                                     ("mae", "S9.csv")):
        rows = []
        for tmpl in template_order:
            row = {"template": tmpl, "n_mutants": n_per_template[tmpl]}
            cell = results.get((metric_name, tmpl), {})
            for col in column_order:
                row[col] = cell.get(col, np.nan)
            rows.append(row)
        df = pd.DataFrame(rows, columns=["template", "n_mutants"] + column_order)
        out_path = out_dir / output_name
        df.to_csv(out_path, index=False)
        print(f"Wrote {out_path}  ({len(df)} rows × {len(df.columns)} cols)")

    if args.ci_out:
        ci_columns = [
            "template", "n_mutants", "model", "regime", "pearson_r",
            "pearson_ci95_low", "pearson_ci95_high",
            "bootstrap_replicates_requested", "bootstrap_replicates_valid", "status",
        ]
        ci_df = pd.DataFrame(pearson_ci_rows, columns=ci_columns)
        model_order = {name: index for index, (name, _) in enumerate(MODELS)}
        regime_order = {
            label: index for index, (_, _, label) in enumerate(REGIME_AGGREGATORS)
        }
        template_order_map = {name: index for index, name in enumerate(template_order)}
        ci_df = ci_df.sort_values(
            ["template", "model", "regime"],
            key=lambda series: series.map(
                template_order_map if series.name == "template" else
                model_order if series.name == "model" else regime_order
            ),
        )
        ci_path = Path(args.ci_out)
        ci_path.parent.mkdir(parents=True, exist_ok=True)
        ci_df.to_csv(ci_path, index=False)
        print(f"Wrote {ci_path}  ({len(ci_df)} rows × {len(ci_df.columns)} cols)")


if __name__ == "__main__":
    main()
