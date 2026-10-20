"""Print Supplementary Table S2 contents from tcr3d/bound_unbound_comparison.csv.

Supplementary Table S2 in the SI (`tab:S2`) reports per-region C-alpha RMSD
between bound and unbound TCR structures across the 30 TCRFlexDock matched
pairs, after superposition on framework C-alpha atoms. Statistics: Mean, Std,
Min, Max.

This script recomputes those statistics from the source CSV so the values in
the manuscript can be verified against the underlying data. Standard deviation
uses the population (ddof=0) convention to match the manuscript's Supplementary Table S2.
"""

import argparse
from pathlib import Path

from helpers.paths import DATA_ROOT, OUTPUT_ROOT, SUPPLEMENTARY_ROOT

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = DATA_ROOT / "tcr3d" / "bound_unbound_comparison.csv"

# (display label, CSV column) — order mirrors Supplementary Table S2 row order.
ROWS = [
    ("CDR1 alpha", "cdr1_alpha_rmsd"),
    ("CDR2 alpha", "cdr2_alpha_rmsd"),
    ("CDR3 alpha", "cdr3_alpha_rmsd"),
    ("CDR1 beta",  "cdr1_beta_rmsd"),
    ("CDR2 beta",  "cdr2_beta_rmsd"),
    ("CDR3 beta",  "cdr3_beta_rmsd"),
    ("CDR (all)",  "cdr_rmsd"),
    ("Framework",  "framework_rmsd"),
]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with Mean/Std/Min/Max for each Supplementary Table S2 region.

    Drops rows where the column is NaN before reducing, so a region that is
    missing for one pair does not contaminate the other regions' counts.
    """
    rows = []
    for label, col in ROWS:
        if col not in df.columns:
            raise KeyError(f"missing column {col!r} in input CSV")
        s = df[col].dropna()
        rows.append({
            "Region": label,
            "Mean":   s.mean(),
            "Std":    s.std(ddof=0),
            "Min":    s.min(),
            "Max":    s.max(),
            "n":      int(s.size),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                    help=f"input CSV (default: {DEFAULT_CSV})")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
    print()

    table = summarize(df)
    # Format: 2 decimals to match Supplementary Table S2 in the manuscript.
    fmt = table.copy()
    for col in ["Mean", "Std", "Min", "Max"]:
        fmt[col] = fmt[col].map(lambda x: f"{x:.2f}")
    print(fmt.to_string(index=False))


if __name__ == "__main__":
    main()
