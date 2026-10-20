"""Generate CDR RMSD histograms from bound/unbound TCR comparison data.

Reads tcr3d/bound_unbound_comparison.csv and produces a 2x4 panel figure
showing RMSD distributions for Framework, CDR (whole), and CDR1/2/3 on
alpha and beta chains.

Usage:
    python scripts/plot_cdr_rmsd_histograms.py
"""

import argparse
from pathlib import Path

from helpers.paths import DATA_ROOT, OUTPUT_ROOT, SUPPLEMENTARY_ROOT

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = DATA_ROOT / "tcr3d" / "bound_unbound_comparison.csv"
OUTPUT_PATH = OUTPUT_ROOT / "figures" / "cdr_rmsd_histograms.pdf"

# Layout: 2 rows x 4 cols
# Row 0: Framework, CDR1α, CDR2α, CDR3α
# Row 1: CDR (all), CDR1β, CDR2β, CDR3β
SUBPLOT_LAYOUT = [
    ("Framework", "framework_rmsd"),
    (r"CDR1$\alpha$", "cdr1_alpha_rmsd"),
    (r"CDR2$\alpha$", "cdr2_alpha_rmsd"),
    (r"CDR3$\alpha$", "cdr3_alpha_rmsd"),
    ("CDR (all)", "cdr_rmsd"),
    (r"CDR1$\beta$", "cdr1_beta_rmsd"),
    (r"CDR2$\beta$", "cdr2_beta_rmsd"),
    (r"CDR3$\beta$", "cdr3_beta_rmsd"),
]


def main():
    """Plot the eight bound/unbound RMSD histograms from a comparison CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=CSV_PATH)
    parser.add_argument("--output-pdf", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} pairs from {args.input.name}\n")

    # Print summary statistics
    print(f"{'Loop':<12} {'Mean':>6} {'Std':>6} {'Min':>6} {'Max':>6}")
    print("-" * 40)
    for label, col in SUBPLOT_LAYOUT:
        vals = df[col].values
        m, s, mn, mx = np.mean(vals), np.std(vals), np.min(vals), np.max(vals)
        plain = label.replace("$\\alpha$", "a").replace("$\\beta$", "b")
        print(f"{plain:<12} {m:>6.2f} {s:>6.2f} {mn:>6.2f} {mx:>6.2f}")
    print()

    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharey=True)
    fig.subplots_adjust(hspace=0.35, wspace=0.15)

    bins = np.arange(0, 6.0, 0.5)
    color = "#4878CF"

    for idx, (label, col) in enumerate(SUBPLOT_LAYOUT):
        row, col_idx = divmod(idx, 4)
        ax = axes[row, col_idx]
        vals = df[col].values
        m, s = np.mean(vals), np.std(vals)

        ax.hist(vals, bins=bins, color=color, edgecolor="white", linewidth=0.5, alpha=0.85)
        ax.axvline(m, color="#C44E52", linestyle="--", linewidth=1.5)
        ax.set_title(label, fontsize=12)
        ax.set_xlim(0, 5.5)
        ax.text(
            0.95, 0.90,
            f"{m:.2f} \u00b1 {s:.2f} \u00c5",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
        )

    for ax in axes[1]:
        ax.set_xlabel(r"RMSD ($\AA$)", fontsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel("Count", fontsize=11)

    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_pdf, bbox_inches="tight", dpi=300)
    print(f"Saved figure to {args.output_pdf}")
    plt.close(fig)


if __name__ == "__main__":
    main()
