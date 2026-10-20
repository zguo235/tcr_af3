"""Plot paired AF3 mutant-ensemble performance changes relative to N=100.

The input is the verified long-form saturation summary table supplied with the
reviewer-response archive. The figure deliberately plots paired changes rather
than overlapping absolute-metric intervals, making the saturation comparison
and its uncertainty directly visible.
"""

import argparse
from pathlib import Path

from helpers.paths import DATA_ROOT, OUTPUT_ROOT, SUPPLEMENTARY_ROOT

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = ["RDE-PPI", "Prompt-DDG", "BA-DDG", "H3-DDG"]
AGGREGATION_ORDER = ["mean", "median", "ranking_weighted"]
ENSEMBLE_SIZE_ORDER = [1, 10, 20, 50, 100]
METRIC_ORDER = ["pearson_r", "rmse", "mae"]

AGGREGATION_STYLE = {
    "mean": {"label": "Mean", "color": "#0D47A1", "marker": "o", "linestyle": "-"},
    "median": {"label": "Median", "color": "#AD1457", "marker": "s", "linestyle": "--"},
    "ranking_weighted": {
        "label": "AF3-rank-weighted mean",
        "color": "#6B7D24",
        "marker": "^",
        "linestyle": "-.",
    },
}

METRIC_LABELS = {
    "pearson_r": r"Change in Pearson $r$",
    "rmse": r"Change in RMSE (kcal mol$^{-1}$)",
    "mae": r"Change in MAE (kcal mol$^{-1}$)",
}


def validate_saturation_table(table: pd.DataFrame) -> None:
    """Validate the complete model/aggregation/size/metric result grid."""
    required_columns = {
        "model",
        "aggregation",
        "ensemble_size",
        "metric",
        "delta_vs_100",
        "delta_ci95_low",
        "delta_ci95_high",
        "n_mutants",
        "bootstrap_replicates",
    }
    missing = sorted(required_columns - set(table.columns))
    if missing:
        raise ValueError(f"saturation table is missing columns {missing}")
    expected_cells = {
        (model, aggregation, size, metric)
        for model in MODEL_ORDER
        for aggregation in AGGREGATION_ORDER
        for size in ENSEMBLE_SIZE_ORDER
        for metric in METRIC_ORDER
    }
    actual_cells = set(
        table[["model", "aggregation", "ensemble_size", "metric"]]
        .itertuples(index=False, name=None)
    )
    missing_cells = expected_cells - actual_cells
    extra_cells = actual_cells - expected_cells
    if missing_cells or extra_cells or len(table) != len(expected_cells):
        raise ValueError(
            "saturation table is not the expected 180-cell grid: "
            f"missing={len(missing_cells)}, extra={len(extra_cells)}, rows={len(table)}"
        )
    if set(table["n_mutants"]) != {100}:
        raise ValueError("saturation analysis must use the fixed 100-mutation cohort")


def build_figure(table: pd.DataFrame) -> plt.Figure:
    """Build the four-by-three paired-difference saturation figure."""
    validate_saturation_table(table)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#424242",
            "axes.linewidth": 0.8,
        }
    )
    fig, axes = plt.subplots(
        len(MODEL_ORDER),
        len(METRIC_ORDER),
        figsize=(11.2, 11.0),
        sharex=True,
        sharey="col",
    )
    x_positions = np.arange(len(ENSEMBLE_SIZE_ORDER), dtype=float)

    for row_index, model in enumerate(MODEL_ORDER):
        for column_index, metric in enumerate(METRIC_ORDER):
            ax = axes[row_index, column_index]
            cell = table[(table["model"] == model) & (table["metric"] == metric)]
            for aggregation in AGGREGATION_ORDER:
                style = AGGREGATION_STYLE[aggregation]
                series = (
                    cell[cell["aggregation"] == aggregation]
                    .set_index("ensemble_size")
                    .loc[ENSEMBLE_SIZE_ORDER]
                )
                values = series["delta_vs_100"].to_numpy(dtype=float)
                lower = series["delta_ci95_low"].to_numpy(dtype=float)
                upper = series["delta_ci95_high"].to_numpy(dtype=float)
                ax.errorbar(
                    x_positions,
                    values,
                    yerr=np.vstack([values - lower, upper - values]),
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=1.2,
                    markersize=4.5,
                    markerfacecolor="white",
                    markeredgewidth=1.0,
                    capsize=2.0,
                    elinewidth=0.9,
                    label=style["label"],
                    zorder=3,
                )
            ax.axhline(0.0, color="#424242", linestyle=":", linewidth=0.9, zorder=1)
            ax.grid(axis="y", color="#D9D9D9", linestyle=":", linewidth=0.7)
            ax.set_axisbelow(True)
            if row_index == 0:
                ax.set_title(METRIC_LABELS[metric], fontsize=10.5, pad=8)
            if column_index == 0:
                ax.set_ylabel(model, fontsize=10.5, fontweight="bold")
            if row_index == len(MODEL_ORDER) - 1:
                ax.set_xticks(x_positions)
                ax.set_xticklabels(ENSEMBLE_SIZE_ORDER)
                ax.set_xlabel("Mutant AF3 ensemble size, N", fontsize=9.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        fontsize=9.5,
    )
    fig.suptitle(
        r"Paired performance change relative to the 100-structure mutant ensemble",
        fontsize=12,
        y=1.025,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965), h_pad=1.0, w_pad=1.0)
    return fig


def main() -> None:
    """Parse CLI paths, build the saturation figure, and save PNG and PDF outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Long-form saturation summary CSV")
    parser.add_argument("--output-png", required=True)
    parser.add_argument("--output-pdf", required=True)
    args = parser.parse_args()

    table = pd.read_csv(args.input)
    figure = build_figure(table)
    output_png = Path(args.output_png)
    output_pdf = Path(args.output_pdf)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=220, bbox_inches="tight")
    figure.savefig(output_pdf, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output_png}")
    print(f"Wrote {output_pdf}")


if __name__ == "__main__":
    main()
