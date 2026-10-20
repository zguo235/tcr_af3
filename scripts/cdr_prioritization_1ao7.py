"""1AO7 CDR-only candidate-prioritization assessment for all four ΔΔG models.

Filters the 181 ATLAS 1AO7 records to those whose mutation set touches at
least one CDR-loop residue (lenient compound-mutation rule) and runs the
five Table 3 structure-input regimes through RDE-PPI, Prompt-DDG, BA-DDG,
and H3-DDG. Reports per-(model, regime) Spearman ρ + top-K precision
(K = 3, 5, 10) for the candidate-prioritization narrative.

CDR membership uses ANARCI IMGT numbering: CDR1 = 27–38, CDR2 = 56–65,
CDR3 = 105–117. Chain convention for 1AO7's ATLAS true_pdb file:
D = TCR α, E = TCR β (per ATLAS_af/CLAUDE.md "ATLAS → true_pdb chain
mapping" — hard-coded because ANARCI chain identification fails on
some ATLAS PDBs).

Usage:
    conda run -n bioemu1p3 python scripts/cdr_prioritization_1ao7.py
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
    is_cdr,
    row_touches_cdr,
)

MODELS = [
    ("RDE-PPI",    DATA_ROOT / "ddg_predictions" / "atlas_af_eval"        / "results"),
    ("Prompt-DDG", DATA_ROOT / "ddg_predictions" / "atlas_promptddg_eval" / "results"),
    ("BA-DDG",     DATA_ROOT / "ddg_predictions" / "atlas_baddg_eval"     / "results"),
    ("H3-DDG",     DATA_ROOT / "ddg_predictions" / "atlas_h3ddg_eval"     / "results"),
]

# (regime, aggregator, short label used as column header).
REGIME_AGGREGATORS = [
    ("PDB_template",      "single", "template"),
    ("wt_af3_single",     "single", "wt"),
    ("wt_af3_multi",      "mean",   "wt_multi_mean"),
    ("mutant_af3_single", "single", "mutant"),
    ("mutant_af3_multi",  "mean",   "mutant_multi_mean"),
]

def load_model_predictions(results_dir: Path, template_map: dict[int, str]
                            ) -> dict[tuple[str, str], pd.DataFrame]:
    """Per-(regime, aggregator) DataFrames keyed by row_idx for one model.

    Single-prediction approaches (1, 2, 4) are passed through directly;
    `_multi` regimes (3, 5) are reduced to per-mutant means via
    `ensemble_aggregate` (median / rankwt are not needed here)."""
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", default=str(DATA_ROOT / "ATLAS" / "www" / "structures"
                                          / "true_pdb" / "1AO7.pdb"))
    ap.add_argument("--index", default=str(DATA_ROOT / "ATLAS_af" / "index.csv"))
    ap.add_argument("--template", default="1AO7")
    ap.add_argument("--top-k", type=int, nargs="+", default=[3, 5, 10])
    ap.add_argument(
        "--out",
        default=str(OUTPUT_ROOT / "supplementary_tables"
                    / "S19.csv"),
    )
    ap.add_argument(
        "--top10-out",
        default=str(OUTPUT_ROOT / "supplementary_tables"
                    / "S20.csv"),
        help="Per-mutation predicted ranks for the 10 most-stabilizing "
             "ATLAS 1AO7 CDR mutants under each (model, regime) cell.",
    )
    args = ap.parse_args()

    # ----- 1. ANARCI IMGT mapping -----
    imgt_map = build_imgt_map(Path(args.pdb), ("D", "E"), template_id=args.template)
    n_d = sum(1 for (c, _r) in imgt_map if c == "D")
    n_e = sum(1 for (c, _r) in imgt_map if c == "E")
    print(f"IMGT mapping: chain D = {n_d} V-domain residues, "
          f"chain E = {n_e} V-domain residues")
    cdr_d = sorted({r for (c, r) in imgt_map if c == "D" and is_cdr(imgt_map[(c, r)])})
    cdr_e = sorted({r for (c, r) in imgt_map if c == "E" and is_cdr(imgt_map[(c, r)])})
    print(f"  CDR positions D (author resseq): {cdr_d}")
    print(f"  CDR positions E (author resseq): {cdr_e}")

    # ----- 2. Filter 1AO7 ATLAS rows to CDR-touching mutations -----
    idx = pd.read_csv(args.index)
    template_rows = idx[idx["template_PDB"] == args.template].copy()
    template_rows["touches_cdr"] = template_rows["mutations"].map(
        lambda m: row_touches_cdr(m, imgt_map)
    )
    cdr_rows = template_rows[template_rows["touches_cdr"]]
    print(f"\n1AO7 ATLAS rows: {len(template_rows)} total, "
          f"{len(cdr_rows)} touch a CDR loop")
    cdr_row_ids = set(cdr_rows["row_idx"].astype(int))

    # ----- 3. Load + aggregate predictions for all 4 models -----
    template_map = load_template_map()
    print("\nLoading + aggregating predictions for 4 models …")
    all_preds = {name: load_model_predictions(rd, template_map) for name, rd in MODELS}

    # Canonical experimental ΔΔG (any model's approach1 has them).
    a1_ref = pd.read_csv(MODELS[0][1] / "approach1.csv")
    a1_ref["ddg_exp"] = a1_ref["ddg_exp"].map(parse_ddg_exp)
    exp_lookup = (a1_ref[["row_idx", "ddg_exp"]]
                  .drop_duplicates(subset="row_idx")
                  .set_index("row_idx")["ddg_exp"])

    # ----- 4. Per-(model, regime) ranking metrics on the CDR subset -----
    rows = []
    for model_name, _ in MODELS:
        for regime, agg, label in REGIME_AGGREGATORS:
            df = all_preds[model_name][(regime, agg)].copy()
            df = df[df["row_idx"].isin(cdr_row_ids)]
            df["ddg_exp"] = df["row_idx"].map(exp_lookup)
            df = df.dropna(subset=["ddg_pred", "ddg_exp"]).set_index("row_idx")
            n = len(df)
            row = {"model": model_name, "regime": label, "n_mutants": n}
            if n >= 3:
                rho, pval = spearmanr(df["ddg_pred"], df["ddg_exp"])
                row["spearman_rho"] = float(rho)
                row["spearman_pvalue"] = float(pval)
            else:
                row["spearman_rho"] = np.nan
                row["spearman_pvalue"] = np.nan
            for k in args.top_k:
                h = topk_hits(df["ddg_pred"], df["ddg_exp"], k)
                row[f"top{k}_hits"] = h
                row[f"top{k}_precision"] = h / min(k, n) if min(k, n) > 0 else np.nan
            rows.append(row)

    cols = ["model", "regime", "n_mutants", "spearman_rho", "spearman_pvalue"]
    for k in args.top_k:
        cols.extend([f"top{k}_hits", f"top{k}_precision"])
    out_df = pd.DataFrame(rows, columns=cols)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}  ({len(out_df)} rows)")

    # ----- 5. Console grids -----
    regime_order = [label for _, _, label in REGIME_AGGREGATORS]
    model_order = [m for m, _ in MODELS]
    for metric in ["spearman_rho"] + [f"top{k}_precision" for k in args.top_k]:
        pivot = (out_df.pivot_table(index="model", columns="regime",
                                     values=metric, aggfunc="first")
                  .reindex(index=model_order, columns=regime_order))
        print(f"\n=== {metric} (CDR subset, n={len(cdr_rows)}) ===")
        with pd.option_context("display.float_format", "{:+.3f}".format,
                                "display.width", 200):
            print(pivot)

    # ----- 6. Per-mutation predicted ranks for the experimental top-10 -----
    # Within the CDR subset, assign rank 1..N to each mutation under each
    # (model, regime) cell where rank 1 = smallest predicted ΔΔG = top
    # candidate. Then select the 10 mutations with smallest experimental
    # ΔΔG and report their predicted ranks across all 20 cells.
    base = pd.DataFrame({"row_idx": sorted(cdr_row_ids)})
    base["mutations"] = base["row_idx"].map(
        idx.set_index("row_idx")["mutations"].to_dict()
    )
    base["ddg_exp"] = base["row_idx"].map(exp_lookup)
    base = base.dropna(subset=["ddg_exp"])

    for model_name, _ in MODELS:
        for _regime, agg, label in REGIME_AGGREGATORS:
            df = all_preds[model_name][(_regime, agg)]
            df_subset = df[df["row_idx"].isin(set(base["row_idx"]))][
                ["row_idx", "ddg_pred"]
            ].set_index("row_idx")
            # rank ascending: 1 = smallest predicted ΔΔG. NaN predictions
            # get NaN rank (pandas default).
            ranks = df_subset["ddg_pred"].rank(method="min", ascending=True)
            base[f"{model_name}:{label}"] = base["row_idx"].map(ranks)

    top10 = base.sort_values("ddg_exp", ascending=True).head(10).reset_index(drop=True)
    rank_cols = [f"{m}:{label}" for m, _ in MODELS
                 for _, _, label in REGIME_AGGREGATORS]
    top10_out = top10[["mutations", "ddg_exp"] + rank_cols]
    top10_out = top10_out.rename(columns={"mutations": "mutation"})
    # Cast rank columns to nullable integer for clean CSV output.
    for col in rank_cols:
        top10_out[col] = top10_out[col].astype("Int64")
    top10_path = Path(args.top10_out)
    top10_path.parent.mkdir(parents=True, exist_ok=True)
    top10_out.to_csv(top10_path, index=False)
    print(f"\nTop-10 ranking table -> {top10_path}  ({len(top10_out)} rows)")
    print(f"  (ranks are 1-based within the n={len(base)} CDR-mutant cohort; "
          f"1 = lowest predicted ΔΔG)")


if __name__ == "__main__":
    main()
