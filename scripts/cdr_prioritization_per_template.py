"""Reproduce four-template CDR candidate prioritization (Table 3 and S18).

Retain variants touching at least one IMGT CDR residue and require at least
20 such variants per template. The manuscript cohort contains 1AO7 (126),
1MI5 (23), 2BNR (22), and 3QDJ (20), totaling 191 variants.
"""

import argparse
import sys
from pathlib import Path

from helpers.paths import DATA_ROOT, OUTPUT_ROOT, SUPPLEMENTARY_ROOT

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "ddg_predictions" / "atlas_af_eval"))
sys.path.insert(0, str(REPO / "scripts"))

from aggregate_results import (  # noqa: E402
    ensemble_aggregate,
    load_template_map,
    mutant_ranking_csv,
    parse_ddg_exp,
    wt_ranking_csv,
)
from helpers.cdr_filter import (  # noqa: E402
    build_imgt_map,
    count_unresolved_tcr_tokens,
    row_touches_cdr,
)
from helpers.correlation_uncertainty import (  # noqa: E402
    paired_correlation_bootstrap,
    stable_cell_seed,
)

MODELS = [
    ("RDE-PPI",    DATA_ROOT / "ddg_predictions" / "atlas_af_eval"        / "results"),
    ("Prompt-DDG", DATA_ROOT / "ddg_predictions" / "atlas_promptddg_eval" / "results"),
    ("BA-DDG",     DATA_ROOT / "ddg_predictions" / "atlas_baddg_eval"     / "results"),
    ("H3-DDG",     DATA_ROOT / "ddg_predictions" / "atlas_h3ddg_eval"     / "results"),
]

# (regime, aggregator, short label) — Table 3 row order.
REGIME_AGGREGATORS = [
    ("PDB_template",      "single", "template"),
    ("wt_af3_single",     "single", "wt"),
    ("wt_af3_multi",      "mean",   "wt_multi_mean"),
    ("mutant_af3_single", "single", "mutant"),
    ("mutant_af3_multi",  "mean",   "mutant_multi_mean"),
]

DEFAULT_TEMPLATES = ["1AO7", "1MI5", "2BNR", "3QDJ"]


def load_model_predictions(
    results_dir: Path, template_map: dict[int, str]
) -> dict[tuple[str, str], pd.DataFrame]:
    """Per-(regime, aggregator) DataFrames keyed by row_idx for one model."""
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


def topk_hits(pred: pd.Series, exp: pd.Series, k: int) -> int:
    """Overlap of K mutants with smallest predicted ΔΔG and K with smallest
    experimental ΔΔG (most-stabilizing). K is capped at len(pred)."""
    k = min(k, len(pred))
    if k == 0:
        return 0
    return len(set(pred.nsmallest(k).index) & set(exp.nsmallest(k).index))


def cdr_filter_one_template(
    template: str, pdb_dir: Path, template_rows: pd.DataFrame
) -> tuple[set[int], int, int]:
    """Build IMGT map for one template and return CDR-touching row_idx set.

    Returns ``(cdr_row_ids, n_total, n_unresolved_tcr_tokens)``. Raises
    ``RuntimeError`` from ``build_imgt_map`` if ANARCI or chain checks fail.
    """
    pdb_path = pdb_dir / f"{template}.pdb"
    imgt_map = build_imgt_map(pdb_path, ("D", "E"), template_id=template)
    mut_strings = template_rows["mutations"].tolist()
    n_unresolved = count_unresolved_tcr_tokens(mut_strings, imgt_map)
    mask = template_rows["mutations"].map(lambda m: row_touches_cdr(m, imgt_map))
    cdr_rows = template_rows[mask]
    return set(cdr_rows["row_idx"].astype(int)), len(template_rows), n_unresolved


def restrict_to_evaluation_cohort(
    index_rows: pd.DataFrame, prediction_rows: pd.DataFrame
) -> pd.DataFrame:
    """Restrict ATLAS index metadata to rows in the cleaned evaluation cohort.

    Args:
        index_rows: ATLAS index rows containing a ``row_idx`` column.
        prediction_rows: A cleaned prediction table containing the evaluation
            cohort's ``row_idx`` values.

    Returns:
        A copy of ``index_rows`` containing only evaluation-cohort row IDs.
    """
    evaluation_ids = set(prediction_rows["row_idx"].astype(int))
    return index_rows[index_rows["row_idx"].astype(int).isin(evaluation_ids)].copy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default=str(DATA_ROOT / "ATLAS_af" / "index.csv"))
    ap.add_argument("--pdb-dir",
                    default=str(DATA_ROOT / "ATLAS" / "www" / "structures" / "true_pdb"))
    ap.add_argument("--min-cdr-mutants", type=int, default=20,
                    help="Templates whose CDR-touching mutant count is below this "
                         "are skipped (default: 20, as in the current manuscript).")
    ap.add_argument("--top-k", type=int, nargs="+", default=[3, 5, 10])
    ap.add_argument("--bootstrap-reps", type=int, default=10_000)
    ap.add_argument("--bootstrap-seed", type=int, default=20_260_722)
    ap.add_argument("--templates", nargs="+", default=DEFAULT_TEMPLATES)
    ap.add_argument("--out-table",
                    default=str(OUTPUT_ROOT
                                / "supplementary_tables"
                                / "S18.csv"))
    ap.add_argument("--out-summary",
                    default=None,
                    help="Optional path for an intermediate cross-template summary CSV.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print per-template (n_total, n_cdr) and exit.")
    args = ap.parse_args()

    pdb_dir = Path(args.pdb_dir)
    idx = pd.read_csv(args.index)
    a1_ref = pd.read_csv(MODELS[0][1] / "approach1.csv")
    idx = restrict_to_evaluation_cohort(idx, a1_ref)

    # ----- 1. Per-template CDR filter (always runs; cheap) -----
    qualifying: list[tuple[str, set[int], int, int]] = []  # (template, ids, n_total, n_cdr)
    skip_log: list[tuple[str, str]] = []  # (template, reason)
    print(f"{'template':<10} {'n_total':>8} {'n_cdr':>8} {'unresolved':>11}")
    print("-" * 41)
    for template in args.templates:
        template_rows = idx[idx["template_PDB"] == template].copy()
        if template_rows.empty:
            skip_log.append((template, "no rows in index.csv"))
            print(f"{template:<10} {'-':>8} {'-':>8} {'-':>11}  SKIP (no rows)")
            continue
        try:
            cdr_ids, n_total, n_unresolved = cdr_filter_one_template(
                template, pdb_dir, template_rows
            )
        except RuntimeError as e:
            skip_log.append((template, f"build_imgt_map: {e}"))
            print(f"{template:<10} {len(template_rows):>8} {'?':>8} {'?':>11}  SKIP ({e})")
            continue
        n_cdr = len(cdr_ids)
        marker = "" if n_cdr >= args.min_cdr_mutants else \
            f"  SKIP (n_cdr<{args.min_cdr_mutants})"
        print(f"{template:<10} {n_total:>8} {n_cdr:>8} {n_unresolved:>11}{marker}")
        if n_cdr >= args.min_cdr_mutants:
            qualifying.append((template, cdr_ids, n_total, n_cdr))
        else:
            skip_log.append((template, f"n_cdr={n_cdr} < {args.min_cdr_mutants}"))

    print(f"\n{len(qualifying)} of {len(args.templates)} templates qualify "
          f"(n_cdr >= {args.min_cdr_mutants}).")
    if skip_log:
        print("Skipped templates:")
        for t, reason in skip_log:
            print(f"  {t}: {reason}")

    if args.dry_run:
        return
    if not qualifying:
        sys.exit("No qualifying templates — nothing to compute.")

    # ----- 2. Load + aggregate predictions once for all 4 models -----
    template_map = load_template_map()
    print("\nLoading + aggregating predictions for 4 models …")
    all_preds = {name: load_model_predictions(rd, template_map) for name, rd in MODELS}

    a1_ref["ddg_exp"] = a1_ref["ddg_exp"].map(parse_ddg_exp)
    exp_lookup = (a1_ref[["row_idx", "ddg_exp"]]
                  .drop_duplicates(subset="row_idx")
                  .set_index("row_idx")["ddg_exp"])

    # ----- 3. Per-(template, model, regime) metrics -----
    rows = []
    for template, cdr_ids, n_total, n_cdr in qualifying:
        for model_name, _ in MODELS:
            for regime, agg, label in REGIME_AGGREGATORS:
                df = all_preds[model_name][(regime, agg)].copy()
                df = df[df["row_idx"].isin(cdr_ids)]
                df["ddg_exp"] = df["row_idx"].map(exp_lookup)
                df = df.dropna(subset=["ddg_pred", "ddg_exp"]).set_index("row_idx")
                n = len(df)
                row = {
                    "template": template,
                    "n_total": n_total,
                    "n_cdr": n_cdr,
                    "model": model_name,
                    "regime": label,
                    "n_mutants": n,
                }
                if n >= 3:
                    rho, pval = spearmanr(df["ddg_pred"], df["ddg_exp"])
                    row["spearman_rho"] = float(rho)
                    row["spearman_pvalue"] = float(pval)
                else:
                    row["spearman_rho"] = np.nan
                    row["spearman_pvalue"] = np.nan
                cell_seed = stable_cell_seed(
                    args.bootstrap_seed, template, model_name, label, "spearman"
                )
                uncertainty = paired_correlation_bootstrap(
                    df["ddg_pred"].to_numpy(),
                    df["ddg_exp"].to_numpy(),
                    method="spearman",
                    replicates=args.bootstrap_reps,
                    seed=cell_seed,
                )
                row["spearman_ci95_low"] = uncertainty["ci_low"]
                row["spearman_ci95_high"] = uncertainty["ci_high"]
                row["bootstrap_replicates_requested"] = uncertainty[
                    "bootstrap_replicates_requested"
                ]
                row["bootstrap_replicates_valid"] = uncertainty[
                    "bootstrap_replicates_valid"
                ]
                row["bootstrap_status"] = uncertainty["status"]
                for k in args.top_k:
                    h = topk_hits(df["ddg_pred"], df["ddg_exp"], k)
                    row[f"top{k}_hits"] = h
                    row[f"top{k}_precision"] = h / min(k, n) if min(k, n) > 0 else np.nan
                rows.append(row)

    cols = [
        "template", "n_total", "n_cdr", "model", "regime", "n_mutants",
        "spearman_rho", "spearman_ci95_low", "spearman_ci95_high",
        "spearman_pvalue", "bootstrap_replicates_requested",
        "bootstrap_replicates_valid", "bootstrap_status",
    ]
    for k in args.top_k:
        cols.extend([f"top{k}_hits", f"top{k}_precision"])
    out_df = pd.DataFrame(rows, columns=cols)
    out_path = Path(args.out_table)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}  ({len(out_df)} rows)")

    # ----- 4. Summary statistics across templates per (model, regime) -----
    summary_rows = []
    for model_name, _ in MODELS:
        for _regime, _agg, label in REGIME_AGGREGATORS:
            sub = out_df[(out_df["model"] == model_name) & (out_df["regime"] == label)]
            rho_vals = sub["spearman_rho"].dropna().to_numpy()
            top10_vals = sub["top10_precision"].dropna().to_numpy()
            if rho_vals.size:
                summary_rows.append({
                    "model": model_name,
                    "regime": label,
                    "n_templates": int(rho_vals.size),
                    "median_rho": float(np.median(rho_vals)),
                    "q1_rho": float(np.quantile(rho_vals, 0.25)),
                    "q3_rho": float(np.quantile(rho_vals, 0.75)),
                    "min_rho": float(rho_vals.min()),
                    "max_rho": float(rho_vals.max()),
                    "mean_top10_recovery": float(sub["top10_hits"].mean()),
                    "median_top10_prec": float(np.median(top10_vals))
                    if top10_vals.size else np.nan,
                })
            else:
                summary_rows.append({
                    "model": model_name, "regime": label, "n_templates": 0,
                    "median_rho": np.nan, "q1_rho": np.nan, "q3_rho": np.nan,
                    "min_rho": np.nan, "max_rho": np.nan,
                    "mean_top10_recovery": np.nan,
                    "median_top10_prec": np.nan,
                })
    summary_df = pd.DataFrame(summary_rows)
    if args.out_summary:
        summary_path = Path(args.out_summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_path, index=False)
        print(f"Wrote {summary_path}  ({len(summary_df)} rows)")

    # ----- 5. Console grids -----
    regime_order = [label for _, _, label in REGIME_AGGREGATORS]
    model_order = [m for m, _ in MODELS]
    print("\n=== Median Spearman ρ across qualifying templates ===")
    pivot = (summary_df.pivot_table(index="model", columns="regime",
                                     values="median_rho", aggfunc="first")
                .reindex(index=model_order, columns=regime_order))
    with pd.option_context("display.float_format", "{:+.3f}".format,
                           "display.width", 200):
        print(pivot)
    print("\n=== n_templates per cell ===")
    pivot_n = (summary_df.pivot_table(index="model", columns="regime",
                                       values="n_templates", aggfunc="first")
                .reindex(index=model_order, columns=regime_order))
    print(pivot_n)


if __name__ == "__main__":
    main()
