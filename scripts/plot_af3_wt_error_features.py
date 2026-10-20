"""Plot the AF3-WT structural error-feature analysis as manuscript Figure 3.

The four-panel static figure reads Supplementary Tables S15--S17 and shows:

* template-level Spearman correlations for the four prespecified features;
* equal-template mean absolute-error change by local contact-disagreement stratum;
* equal-template mean absolute-error change by single-point mutation region; and
* equal-template mean absolute-error change by normalized peptide-position third.

Usage:
    python scripts/plot_af3_wt_error_features.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from helpers.paths import DATA_ROOT, OUTPUT_ROOT, SUPPLEMENTARY_ROOT

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent
MANUSCRIPT_DIR = OUTPUT_ROOT
SUPPLEMENTARY_DIR = SUPPLEMENTARY_ROOT
FIGURE_DIR = MANUSCRIPT_DIR / "figures"

MODEL_ORDER = ["RDE-PPI", "Prompt-DDG", "BA-DDG", "H3-DDG"]
MODEL_STYLE = {
    "RDE-PPI": {
        "color": "#0D47A1",
        "marker": "o",
        "linestyle": "-",
    },
    "Prompt-DDG": {
        "color": "#F9A825",
        "marker": "s",
        "linestyle": "--",
    },
    "BA-DDG": {
        "color": "#6B7D24",
        "marker": "^",
        "linestyle": "-.",
    },
    "H3-DDG": {
        "color": "#AD1457",
        "marker": "D",
        "linestyle": ":",
    },
}

HEATMAP_FEATURES = [
    "contact_disagreement_5a",
    "cdr3_rmsd_angstrom",
    "ptm",
    "iptm",
]
HEATMAP_LABELS = [
    "Contact\ndisagreement",
    r"CDR3 RMSD" "\n" r"($\AA$)",
    "pTM",
    "ipTM",
]
CONTACT_ORDER = [
    "No disagreement",
    "Low (0--0.33]",
    "Medium (0.33--0.67]",
    "High (0.67--1.0]",
]
REGION_ORDER = [
    "TCR CDR3",
    "Other TCR CDR",
    "TCR framework",
    "Peptide",
    "MHC",
]
PEPTIDE_ORDER = [
    "N-terminal third",
    "Middle third",
    "C-terminal third",
]


def _validate_inputs(
    template_table: pd.DataFrame,
    mutation_table: pd.DataFrame,
    associations: pd.DataFrame,
) -> None:
    """Validate the expected cohort and required plotting records."""
    if len(template_table) != 37 or len(mutation_table) != 438:
        raise ValueError(
            f"expected 37 templates and 438 mutations, found "
            f"{len(template_table)} and {len(mutation_table)}"
        )
    required = {
        "analysis_type",
        "subset",
        "model_or_contrast",
        "outcome",
        "feature",
        "category",
        "estimate",
        "ci95_low",
        "ci95_high",
        "q_value",
    }
    missing = sorted(required - set(associations.columns))
    if missing:
        raise ValueError(f"association table is missing columns {missing}")


def _plot_heatmap(ax: plt.Axes, associations: pd.DataFrame) -> None:
    """Plot requested template-level feature correlations by DDG model."""
    source = associations[
        (associations["analysis_type"] == "spearman")
        & (associations["subset"] == "all_templates")
        & (associations["outcome"] == "absolute_error_change")
        & associations["model_or_contrast"].isin(MODEL_ORDER)
        & associations["feature"].isin(HEATMAP_FEATURES)
    ]
    estimates = source.pivot(
        index="model_or_contrast",
        columns="feature",
        values="estimate",
    ).reindex(index=MODEL_ORDER, columns=HEATMAP_FEATURES)
    q_values = source.pivot(
        index="model_or_contrast",
        columns="feature",
        values="q_value",
    ).reindex(index=MODEL_ORDER, columns=HEATMAP_FEATURES)
    if estimates.isna().any().any():
        raise ValueError("template heatmap has missing model-feature cells")

    limit = max(0.4, float(np.nanmax(np.abs(estimates.to_numpy()))))
    limit = min(1.0, np.ceil(limit * 10.0) / 10.0)
    ax.imshow(
        estimates.to_numpy(dtype=float),
        cmap="PuOr_r",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )
    for row_index, model in enumerate(MODEL_ORDER):
        for column_index, feature in enumerate(HEATMAP_FEATURES):
            estimate = float(estimates.loc[model, feature])
            q_value = q_values.loc[model, feature]
            marker = "*" if pd.notna(q_value) and float(q_value) < 0.05 else ""
            text_color = "white" if abs(estimate) > limit * 0.58 else "#212121"
            ax.text(
                column_index,
                row_index,
                f"{estimate:+.2f}{marker}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=11,
                fontweight="bold" if marker else "normal",
            )
    ax.set_xticks(np.arange(len(HEATMAP_FEATURES)))
    ax.set_xticklabels(HEATMAP_LABELS, fontsize=11)
    ax.set_yticks(np.arange(len(MODEL_ORDER)))
    ax.set_yticklabels(MODEL_ORDER, fontsize=11)
    ax.tick_params(length=0)
    ax.set_title(
        "Template features vs. absolute-error change",
        fontsize=13,
        pad=10,
    )


def _category_counts(
    mutation_table: pd.DataFrame,
    feature: str,
) -> dict[str, int]:
    """Return mutation counts matching one plotted categorical feature."""
    if feature == "local_contact_disagreement_stratum":
        values = (
            mutation_table["local_contact_disagreement_5a"]
            .astype(float)
            .round(12)
        )
        categories = pd.cut(
            values,
            bins=[-np.inf, 0.0, 1.0 / 3.0, 2.0 / 3.0, np.inf],
            labels=CONTACT_ORDER,
            include_lowest=True,
            right=True,
        ).astype(str)
        return categories.value_counts().to_dict()
    if feature == "single_point_region":
        subset = mutation_table[mutation_table["num_muts"] == 1]
        return subset["single_point_region"].value_counts().to_dict()
    if feature == "peptide_position_third":
        subset = mutation_table[
            (mutation_table["num_muts"] == 1)
            & (mutation_table["single_point_region"] == "Peptide")
        ]
        categories = pd.cut(
            subset["peptide_position_normalized"].astype(float),
            bins=[-np.inf, 1.0 / 3.0, 2.0 / 3.0, np.inf],
            labels=PEPTIDE_ORDER,
            include_lowest=True,
            right=True,
        ).astype(str)
        return categories.value_counts().to_dict()
    raise KeyError(feature)


def _plot_category_intervals(
    ax: plt.Axes,
    associations: pd.DataFrame,
    mutation_table: pd.DataFrame,
    *,
    feature: str,
    category_order: list[str],
    title: str,
    connect: bool,
) -> None:
    """Plot equal-template category means and block-bootstrap intervals."""
    source = associations[
        (associations["analysis_type"] == "equal_template_category_mean")
        & (associations["outcome"] == "absolute_error_change")
        & (associations["feature"] == feature)
        & associations["model_or_contrast"].isin(MODEL_ORDER)
    ]
    counts = _category_counts(mutation_table, feature)
    x = np.arange(len(category_order), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(MODEL_ORDER))

    for model_index, model in enumerate(MODEL_ORDER):
        style = MODEL_STYLE[model]
        model_rows = source[source["model_or_contrast"] == model].set_index(
            "category"
        )
        if not set(category_order) <= set(model_rows.index):
            missing = sorted(set(category_order) - set(model_rows.index))
            raise ValueError(f"{feature} is missing {model} categories {missing}")
        means = model_rows.loc[category_order, "estimate"].to_numpy(dtype=float)
        lows = model_rows.loc[category_order, "ci95_low"].to_numpy(dtype=float)
        highs = model_rows.loc[category_order, "ci95_high"].to_numpy(dtype=float)
        positions = x + offsets[model_index]
        ax.errorbar(
            positions,
            means,
            yerr=np.vstack([means - lows, highs - means]),
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"] if connect else "none",
            linewidth=1.2,
            markersize=5.5,
            markerfacecolor="white",
            markeredgewidth=1.2,
            capsize=2.5,
            elinewidth=1.0,
            zorder=3,
        )

    display_labels = [
        f"{category}\n(n={counts.get(category, 0)})"
        for category in category_order
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, fontsize=10)
    ax.set_title(title, fontsize=13, pad=10)
    ax.axhline(0.0, color="#616161", linestyle="--", linewidth=0.8, zorder=1)
    ax.grid(axis="y", color="#D9D9D9", linestyle=":", linewidth=0.8)
    ax.set_axisbelow(True)


def _shared_error_change_limits(associations: pd.DataFrame) -> tuple[float, float]:
    """Return common y-axis limits for the three absolute-error-change panels."""
    feature_names = {
        "local_contact_disagreement_stratum",
        "single_point_region",
        "peptide_position_third",
    }
    source = associations[
        (associations["analysis_type"] == "equal_template_category_mean")
        & (associations["outcome"] == "absolute_error_change")
        & associations["feature"].isin(feature_names)
        & associations["model_or_contrast"].isin(MODEL_ORDER)
    ]
    low = min(0.0, float(source["ci95_low"].min()))
    high = max(0.0, float(source["ci95_high"].max()))
    padding = max(0.08, (high - low) * 0.08)
    return low - padding, high + padding


def build_figure(
    template_table: pd.DataFrame,
    mutation_table: pd.DataFrame,
    associations: pd.DataFrame,
) -> plt.Figure:
    """Build and return the complete 2-by-2 manuscript figure."""
    _validate_inputs(template_table, mutation_table, associations)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#424242",
            "axes.labelcolor": "#212121",
            "xtick.color": "#424242",
            "ytick.color": "#424242",
            "text.color": "#212121",
            "font.size": 11,
            "pdf.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(12.2, 8.8), constrained_layout=True)
    grid = figure.add_gridspec(3, 2, height_ratios=[0.08, 1.0, 1.0])
    legend_axis = figure.add_subplot(grid[0, :])
    legend_axis.axis("off")
    axes = np.asarray(
        [
            [figure.add_subplot(grid[1, 0]), figure.add_subplot(grid[1, 1])],
            [figure.add_subplot(grid[2, 0]), figure.add_subplot(grid[2, 1])],
        ],
        dtype=object,
    )
    _plot_heatmap(axes[0, 0], associations)
    _plot_category_intervals(
        axes[0, 1],
        associations,
        mutation_table,
        feature="local_contact_disagreement_stratum",
        category_order=CONTACT_ORDER,
        title="Absolute-error change by local contact disagreement",
        connect=True,
    )
    _plot_category_intervals(
        axes[1, 0],
        associations,
        mutation_table,
        feature="single_point_region",
        category_order=REGION_ORDER,
        title="Absolute-error change by single-point mutation region",
        connect=False,
    )
    _plot_category_intervals(
        axes[1, 1],
        associations,
        mutation_table,
        feature="peptide_position_third",
        category_order=PEPTIDE_ORDER,
        title="Absolute-error change by normalized peptide position",
        connect=True,
    )

    y_limits = _shared_error_change_limits(associations)
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        ax.set_ylim(*y_limits)
        ax.set_ylabel(
            "Change in absolute error\n"
            r"(AF3-WT $-$ PDB), kcal mol$^{-1}$",
            fontsize=11,
        )
    axes[1, 0].tick_params(axis="x", labelrotation=18)

    for label, ax in zip(("A", "B", "C", "D"), axes.flat):
        ax.text(
            -0.08,
            1.08,
            label,
            transform=ax.transAxes,
            fontsize=15,
            fontweight="bold",
            va="top",
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_STYLE[model]["color"],
            marker=MODEL_STYLE[model]["marker"],
            linestyle=MODEL_STYLE[model]["linestyle"],
            markerfacecolor="white",
            markeredgewidth=1.2,
            linewidth=1.2,
            label=model,
        )
        for model in MODEL_ORDER
    ]
    legend_axis.legend(
        handles=legend_handles,
        loc="center",
        ncol=4,
        frameon=False,
        fontsize=11,
    )
    figure.suptitle(
        r"AF3-WT structural features and $\Delta\Delta G$ prediction error",
        fontsize=15,
    )
    return figure


def main() -> None:
    """Parse arguments, render the figure, and write PDF and PNG outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template-table",
        type=Path,
        default=SUPPLEMENTARY_DIR / "S15.csv",
    )
    parser.add_argument(
        "--mutation-table",
        type=Path,
        default=SUPPLEMENTARY_DIR / "S16.csv",
    )
    parser.add_argument(
        "--associations-table",
        type=Path,
        default=SUPPLEMENTARY_DIR / "S17.csv",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=FIGURE_DIR / "af3_wt_error_features.pdf",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=FIGURE_DIR / "af3_wt_error_features.png",
    )
    args = parser.parse_args()

    template_table = pd.read_csv(args.template_table)
    mutation_table = pd.read_csv(args.mutation_table)
    associations = pd.read_csv(args.associations_table)
    figure = build_figure(template_table, mutation_table, associations)
    for output_path in (args.output_pdf, args.output_png):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_pdf, bbox_inches="tight")
    figure.savefig(args.output_png, bbox_inches="tight", dpi=240)
    plt.close(figure)
    print(f"Wrote {args.output_pdf}")
    print(f"Wrote {args.output_png}")


if __name__ == "__main__":
    main()
