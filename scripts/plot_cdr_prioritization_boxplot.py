"""Plot per-template CDR candidate-prioritization performance.

The analytical question is how structure-input ranking varies across the four
eligible templates. The 2-by-2 box-and-point layout shows the full per-template
distribution for each predictor while retaining 1AO7 as a visually distinct
reference. It expects the 80-row table produced by
``scripts/cdr_prioritization_per_template.py``.

Chart contract: distribution/box plot; 80 rows at
(template, predictor, regime) grain; neutral boxes with full-range whiskers and
non-1AO7 points, a blue median, and a larger filled orange circle with a dark
outline for 1AO7;
7.4-by-6.4-inch static PDF and PNG outputs for full-width supplementary
placement.

Usage:
    python scripts/plot_cdr_prioritization_boxplot.py \
        --input results/supplementary_tables/S18.csv \
        --output-pdf results/figures/cdr_prioritization_boxplot.pdf \
        --output-png results/figures/cdr_prioritization_boxplot.png
"""

import argparse
from pathlib import Path

from helpers.paths import DATA_ROOT, OUTPUT_ROOT, SUPPLEMENTARY_ROOT

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = SUPPLEMENTARY_ROOT / "S18.csv"
DEFAULT_OUTPUT_PDF = (
    OUTPUT_ROOT / "figures" / "cdr_prioritization_boxplot.pdf"
)
DEFAULT_OUTPUT_PNG = (
    OUTPUT_ROOT / "figures" / "cdr_prioritization_boxplot.png"
)

MODEL_ORDER = ["RDE-PPI", "Prompt-DDG", "BA-DDG", "H3-DDG"]
REGIME_ORDER = ["template", "wt", "wt_multi_mean", "mutant", "mutant_multi_mean"]
REGIME_LABELS = {
    "template": "PDB",
    "wt": "WT\nsingle",
    "wt_multi_mean": "WT\nmean",
    "mutant": "Mutant\nsingle",
    "mutant_multi_mean": "Mutant\nmean",
}


def validate_input(df: pd.DataFrame) -> None:
    """Validate the expected complete template-by-model-by-regime grid.

    Args:
        df: Long-form per-template correlation table.

    Returns:
        None. A ``ValueError`` is raised when required columns or cells are
        missing so a partial figure cannot be exported silently.
    """
    required_columns = {"template", "model", "regime", "spearman_rho"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"input table is missing columns: {sorted(missing_columns)}")
    templates = set(df["template"])
    expected = {
        (template, model, regime)
        for template in templates
        for model in MODEL_ORDER
        for regime in REGIME_ORDER
    }
    actual = set(df[["template", "model", "regime"]].itertuples(index=False, name=None))
    if actual != expected or len(df) != len(expected):
        raise ValueError(
            "input table is not a complete template-by-model-by-regime grid: "
            f"expected {len(expected)} cells, found {len(df)} rows"
        )


def build_figure(df: pd.DataFrame) -> plt.Figure:
    """Build the journal-scale 2-by-2 box-and-point figure.

    Args:
        df: Validated long-form table with one Spearman correlation for each
            template, predictor, and structural-input regime.

    Returns:
        A Matplotlib figure ready for PDF or raster export.
    """
    validate_input(df)
    n_templates = df["template"].nunique()
    templates = sorted(df["template"].unique())
    print(f"Loaded {len(df)} rows over {n_templates} templates: {templates}")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#424242",
            "axes.linewidth": 0.8,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.4), sharex=True, sharey=True)
    flat_axes = axes.ravel()
    x = np.arange(len(REGIME_ORDER))
    box_props = dict(facecolor="#DCE3E7", edgecolor="#455A64", linewidth=0.9)
    median_props = dict(color="#0D47A1", linewidth=1.5)
    whisker_props = dict(color="#455A64", linewidth=0.9)
    cap_props = dict(color="#455A64", linewidth=0.9)

    for panel_index, (ax, model) in enumerate(zip(flat_axes, MODEL_ORDER)):
        sub = df[df["model"] == model]
        per_regime = [
            sub[sub["regime"] == r]["spearman_rho"].dropna().to_numpy()
            for r in REGIME_ORDER
        ]
        ax.boxplot(
            per_regime,
            positions=x,
            widths=0.55,
            patch_artist=True,
            boxprops=box_props,
            medianprops=median_props,
            whiskerprops=whisker_props,
            capprops=cap_props,
            whis=(0, 100),
            showfliers=False,
        )
        # A seeded jitter makes all templates visible and reproducible.
        for i, regime in enumerate(REGIME_ORDER):
            cell = sub[sub["regime"] == regime]
            jitter = np.random.default_rng(seed=i * 7 + len(model)).uniform(
                -0.10, 0.10, size=len(cell)
            )
            for j, (_, row) in enumerate(cell.iterrows()):
                if pd.isna(row["spearman_rho"]):
                    continue
                if row["template"] == "1AO7":
                    ax.scatter(
                        i,
                        row["spearman_rho"],
                        marker="o",
                        s=58,
                        facecolor="#D55E00",
                        edgecolor="#212121",
                        linewidth=0.75,
                        zorder=4,
                        label="1AO7" if i == 0 else None,
                    )
                else:
                    ax.scatter(
                        i + jitter[j],
                        row["spearman_rho"],
                        marker="o",
                        s=23,
                        facecolor="#ECEFF1",
                        alpha=0.92,
                        edgecolor="#607D8B",
                        linewidth=0.65,
                        zorder=3,
                        label=(
                            "Other templates"
                            if i == 0 and j == 0 and row["template"] != "1AO7"
                            else None
                        ),
                    )
        ax.axhline(0.0, color="#616161", linestyle="--", linewidth=0.75, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([REGIME_LABELS[r] for r in REGIME_ORDER], fontsize=7.4)
        ax.set_title(model, fontsize=10.2, fontweight="bold", pad=6)
        ax.set_ylim(-0.45, 1.0)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.65, linestyle=":")
        ax.set_axisbelow(True)
        ax.text(
            -0.13,
            1.04,
            chr(ord("A") + panel_index),
            transform=ax.transAxes,
            fontsize=10.5,
            fontweight="bold",
            va="bottom",
        )

    for ax in axes[:, 0]:
        ax.set_ylabel(r"Per-template Spearman $\rho$", fontsize=9)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            markersize=7.2,
            markerfacecolor="#D55E00",
            markeredgecolor="#212121",
            linestyle="none",
            label="1AO7",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            markersize=5.5,
            markerfacecolor="#ECEFF1",
            markeredgecolor="#607D8B",
            linestyle="none",
            label="Other templates",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=8.2,
        bbox_to_anchor=(0.5, 0.955),
    )

    fig.suptitle(
        f"Per-template CDR candidate-prioritization performance ($n={n_templates}$ templates)",
        fontsize=10.8,
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92), h_pad=1.1, w_pad=1.0)
    return fig


def main() -> None:
    """Parse paths, validate the data, and export the PDF and PNG figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-pdf", default=str(DEFAULT_OUTPUT_PDF))
    parser.add_argument("--output-png", default=str(DEFAULT_OUTPUT_PNG))
    args = parser.parse_args()

    dataframe = pd.read_csv(args.input)
    figure = build_figure(dataframe)
    output_pdf = Path(args.output_pdf)
    output_png = Path(args.output_png)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_pdf, bbox_inches="tight")
    figure.savefig(output_png, bbox_inches="tight", dpi=240)
    plt.close(figure)
    print(f"Wrote {output_pdf}")
    print(f"Wrote {output_png}")


if __name__ == "__main__":
    main()
