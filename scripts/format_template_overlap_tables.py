#!/usr/bin/env python3
"""Format SI Tables S21--S22 from saved overlap-analysis estimates.

Print editable LaTeX to stdout; no analysis is rerun and no input is modified.
Only experimental WT and AF3-WT single results are included.
"""

import argparse
import csv
from pathlib import Path
from helpers.paths import OUTPUT_ROOT


models = ("RDE-PPI", "Prompt-DDG")
groups = {
    "shared_pdb_id": "Shared PDB ID: 356 records, 15 templates",
    "no_shared_pdb_id": "No shared PDB ID: 82 records, 22 templates",
}
scaffolds = {
    "PDB_template": "Experimental WT",
    "wt_af3_single": "AF3-WT single",
}
accuracy_metrics = ("pearson_r", "rmse_kcal_per_mol", "mae_kcal_per_mol")
change_metrics = (
    "delta_pearson_r", "delta_rmse_kcal_per_mol", "delta_mae_kcal_per_mol",
)


def index_rows(path, key_fields):
    """Read a CSV path into a mapping from key-field tuples to string-valued rows.

    Duplicate keys raise an error so no saved estimate can be silently replaced.
    """
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {tuple(row[field] for field in key_fields): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate result keys in {path}")
    return indexed


def format_interval(row, metric):
    """Return a LaTeX cell with a saved estimate and its 95% CI to three decimals.

    row is a CSV record; metric names the estimate and its two CI suffix columns.
    """
    values = [float(row[field]) for field in (
        metric, f"{metric}_ci95_low", f"{metric}_ci95_high",
    )]
    return rf"${values[0]:.3f}\;[{values[1]:.3f},\,{values[2]:.3f}]$"


def render_tables(source_dir):
    """Return complete red LaTeX tables from the three saved result CSVs.

    source_dir holds the verified overlap-analysis outputs. The returned block
    contains eight accuracy rows, four paired-change rows, and two contrasts,
    with landscape orientation and the existing S1--S20 numbering preserved.
    """
    key_fields = ("overlap_group", "model", "structural_input")
    accuracy = index_rows(source_dir / "prediction_accuracy.csv", key_fields)
    changes = index_rows(source_dir / "predicted_scaffold_changes.csv", key_fields)
    contrasts = index_rows(
        source_dir / "between_group_contrasts.csv",
        ("model", "structural_input"),
    )
    lines = [
        "% BEGIN GENERATED TEMPLATE-OVERLAP TABLES",
        r"\clearpage",
        r"\begin{landscape}",
        r"\makeatletter",
        r"\setlength{\@fptop}{0pt plus 1fil}",
        r"\setlength{\@fpsep}{12pt}",
        r"\setlength{\@fpbot}{0pt plus 1fil}",
        r"\makeatother",
        r"\setcounter{table}{20}",
        r"\begin{table}[p]",
        r"\color{red}",
        r"\caption{Prediction accuracy and changes following replacement of experimental WT structures with AF3-WT single structures, stratified by template PDB-ID overlap with SKEMPI~2.0. The shared-ID group contains 356 ATLAS records on 15 templates; the no-shared-ID group contains 82 records on 22 templates. Panel~A reports pooled Pearson $r$, RMSE, and MAE. Panel~B reports AF3-WT single minus experimental WT performance on the same records: negative $\Delta r$ and positive error changes indicate deterioration. Error metrics and their changes are in kcal/mol. Brackets contain percentile 95\% confidence intervals from 10,000 template-block bootstrap resamples, paired across predictors and structural inputs within each group.}",
        r"\label{tab:S21}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\textbf{A. Prediction accuracy}\par\smallskip",
        r"\begin{tabular}{l l c c c}",
        r"\hline",
        r"\textbf{Predictor} & \textbf{Structural input} & \textbf{Pearson $r$} & \textbf{RMSE} & \textbf{MAE} \\",
        r"\hline",
    ]
    for group, title in groups.items():
        lines.append(r"\multicolumn{5}{l}{\textbf{" + title + r"}} \\")
        for model in models:
            for scaffold, label in scaffolds.items():
                row = accuracy[(group, model, scaffold)]
                cells = [model, label] + [format_interval(row, metric) for metric in accuracy_metrics]
                lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\hline")
    lines.extend([
        r"\end{tabular}",
        r"\par\medskip",
        r"\textbf{B. Changes from experimental WT to AF3-WT single}\par\smallskip",
        r"\begin{tabular}{l c c c}",
        r"\hline",
        r"\textbf{Predictor} & \textbf{$\Delta r$} & \textbf{$\Delta$RMSE} & \textbf{$\Delta$MAE} \\",
        r"\hline",
    ])
    for group, title in groups.items():
        lines.append(r"\multicolumn{4}{l}{\textbf{" + title + r"}} \\")
        for model in models:
            row = changes[(group, model, "wt_af3_single")]
            cells = [model] + [format_interval(row, metric) for metric in change_metrics]
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\hline")
    lines.extend([
        r"\end{tabular}",
        r"\end{table}",
        r"\begin{table}[p]",
        r"\color{red}",
        r"\caption{Between-group differences in the changes from experimental WT to AF3-WT single performance. Each contrast is the change in the no-shared-PDB-ID group (82 records, 22 templates) minus the change in the shared-PDB-ID group (356 records, 15 templates). Negative correlation-change contrasts and positive error-change contrasts indicate greater deterioration in the no-shared-ID group. Error-change contrasts are in kcal/mol. Brackets contain percentile 95\% confidence intervals from 10,000 template-block bootstrap resamples drawn independently within the two groups, with paired experimental/AF3-WT comparisons within each group. These observational contrasts do not isolate the causal effect of training exposure.}",
        r"\label{tab:S22}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{l c c c}",
        r"\hline",
        r"\textbf{Predictor} & \textbf{\shortstack{Difference in\\$\Delta r$}} & \textbf{\shortstack{Difference in\\$\Delta$RMSE}} & \textbf{\shortstack{Difference in\\$\Delta$MAE}} \\",
        r"\hline",
    ])
    for model in models:
        row = contrasts[(model, "wt_af3_single")]
        if row["contrast"] != "no_shared_pdb_id_minus_shared_pdb_id":
            raise ValueError(f"Unexpected contrast direction for {model}")
        cells = [model] + [format_interval(row, f"difference_in_{metric}") for metric in change_metrics]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
        r"\end{landscape}",
        "% END GENERATED TEMPLATE-OVERLAP TABLES",
    ])
    return "\n".join(lines) + "\n"


def main():
    """Parse the source directory argument and print the generated LaTeX tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path,
        default=OUTPUT_ROOT / "template_overlap",
        help="Directory containing the three saved overlap-result CSVs.",
    )
    args = parser.parse_args()
    print(render_tables(args.source_dir), end="")


if __name__ == "__main__":
    main()
