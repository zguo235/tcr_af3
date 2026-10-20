"""Print Table 1 contents from tcr3d/af_prediction_comparison.csv (and bound-unbound).

Table 1 in the manuscript (`tab:af_rmsd`) reports per-region C-alpha RMSD between
AlphaFold Server's top-ranked prediction (model 0) and the experimental structure
across the 30 TCRFlexDock complexes, alongside the bound-unbound RMSD from
Supplementary Table S2 for reference. AF RMSD values are computed after
superposition on TCR framework C-alpha atoms; the Complex C-alpha row uses the
whole-complex C-alpha RMSD after the same superposition.

This script recomputes the table values from the source CSVs so the manuscript
numbers can be verified against the underlying data. Standard deviation uses the
population (ddof=0) convention to match the manuscript and the histogram-script
summary.
"""

import argparse
from pathlib import Path

from helpers.paths import DATA_ROOT, OUTPUT_ROOT, SUPPLEMENTARY_ROOT

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AF_CSV = DATA_ROOT / "tcr3d" / "af_prediction_comparison.csv"
DEFAULT_BU_CSV = DATA_ROOT / "tcr3d" / "bound_unbound_comparison.csv"

# (display label, AF column in af_prediction_comparison.csv,
#                 BU column in bound_unbound_comparison.csv or None)
# Order mirrors Table 1 row order.
ROWS = [
    ("CDR1 alpha",    "cdr1_alpha_rmsd",  "cdr1_alpha_rmsd"),
    ("CDR2 alpha",    "cdr2_alpha_rmsd",  "cdr2_alpha_rmsd"),
    ("CDR3 alpha",    "cdr3_alpha_rmsd",  "cdr3_alpha_rmsd"),
    ("CDR1 beta",     "cdr1_beta_rmsd",   "cdr1_beta_rmsd"),
    ("CDR2 beta",     "cdr2_beta_rmsd",   "cdr2_beta_rmsd"),
    ("CDR3 beta",     "cdr3_beta_rmsd",   "cdr3_beta_rmsd"),
    ("Framework",     "framework_rmsd",   "framework_rmsd"),
    ("CDR (all)",     "cdr_rmsd",         "cdr_rmsd"),
    # No unbound whole-complex value: the unbound structure contains TCR only.
    ("Whole complex Calpha", "tcr_pmhc_ca_rmsd", None),
]


def stats(s: pd.Series) -> dict:
    """Mean/Std/Min/Max with population std (ddof=0). Returns None values if empty."""
    s = s.dropna()
    if s.empty:
        return {"Mean": None, "Std": None, "Min": None, "Max": None, "n": 0}
    return {
        "Mean": s.mean(),
        "Std":  s.std(ddof=0),
        "Min":  s.min(),
        "Max":  s.max(),
        "n":    int(s.size),
    }


def summarize(af_df: pd.DataFrame, bu_df: pd.DataFrame) -> pd.DataFrame:
    """Build a Region x (AF_*, BU_*) statistics table mirroring Table 1."""
    rows = []
    for label, af_col, bu_col in ROWS:
        if af_col not in af_df.columns:
            raise KeyError(f"missing column {af_col!r} in AF CSV")
        af = stats(af_df[af_col])
        if bu_col is None:
            bu = {"Mean": None, "Std": None, "Min": None, "Max": None, "n": 0}
        else:
            if bu_col not in bu_df.columns:
                raise KeyError(f"missing column {bu_col!r} in BU CSV")
            bu = stats(bu_df[bu_col])
        rows.append({
            "Region":  label,
            "AF_Mean": af["Mean"], "AF_Std": af["Std"],
            "AF_Min":  af["Min"],  "AF_Max": af["Max"],
            "AF_n":    af["n"],
            "BU_Mean": bu["Mean"], "BU_Std": bu["Std"],
            "BU_Min":  bu["Min"],  "BU_Max": bu["Max"],
            "BU_n":    bu["n"],
        })
    return pd.DataFrame(rows)


def fmt_cell(x) -> str:
    """2-decimal format; '---' for missing values to match manuscript."""
    if x is None or pd.isna(x):
        return "---"
    return f"{x:.2f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--af-csv", type=Path, default=DEFAULT_AF_CSV,
                    help=f"AF comparison CSV (default: {DEFAULT_AF_CSV})")
    ap.add_argument("--bu-csv", type=Path, default=DEFAULT_BU_CSV,
                    help=f"bound-unbound CSV (default: {DEFAULT_BU_CSV})")
    ap.add_argument("--model", type=int, default=0,
                    help="AF model rank to use (default: 0 = top-ranked)")
    ap.add_argument("--per-complex-output", type=Path,
                    help="Optional CSV containing the per-complex values in SI Table S3.")
    args = ap.parse_args()

    af_df_all = pd.read_csv(args.af_csv)
    af_df = af_df_all[af_df_all["model"] == args.model].copy()
    bu_df = pd.read_csv(args.bu_csv)
    if args.per_complex_output:
        columns = ["pdb_id", "framework_rmsd", "cdr1_alpha_rmsd",
                   "cdr2_alpha_rmsd", "cdr3_alpha_rmsd", "cdr1_beta_rmsd",
                   "cdr2_beta_rmsd", "cdr3_beta_rmsd", "cdr_rmsd",
                   "tcr_pmhc_ca_rmsd"]
        args.per_complex_output.parent.mkdir(parents=True, exist_ok=True)
        af_df[columns].sort_values("pdb_id").to_csv(args.per_complex_output, index=False)

    print(f"Loaded {len(af_df_all)} rows from {args.af_csv} "
          f"({len(af_df)} after filtering model == {args.model})")
    print(f"Loaded {len(bu_df)} rows from {args.bu_csv}")
    print()

    table = summarize(af_df, bu_df)

    fmt = table.copy()
    for col in ["AF_Mean", "AF_Std", "AF_Min", "AF_Max",
                "BU_Mean", "BU_Std", "BU_Min", "BU_Max"]:
        fmt[col] = fmt[col].map(fmt_cell)
    # Drop the n columns from display; keep them implicit in the load message.
    fmt = fmt.drop(columns=["AF_n", "BU_n"])
    print(fmt.to_string(index=False))


if __name__ == "__main__":
    main()
