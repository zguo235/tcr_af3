# TCR–pMHC scaffold benchmark: manuscript analysis code

## Installation

The analysis runs on CPU. Create an environment from this folder:

```bash
conda env create -f environment.yml
conda activate tcr-scaffold-analysis
python -m pytest -q
```

`environment.yml` installs Python 3.12 and HMMER 3.4, then the Python versions in
`requirements.txt`. These match the packages used for release validation.

ANARCI assigns IMGT numbering to TCR sequences. The requirements use the
[ANARCI 2026.2.13.2 distribution](https://pypi.org/project/ANARCI/2026.2.13.2/);
the upstream [ANARCI documentation](https://github.com/oxpig/ANARCI#installation)
describes its HMMER dependency and an alternative source installation. No
PyTorch, CUDA, or predictor checkpoint is required by this package.

## Run the analyses

```bash
# Check required files and their reference checksums without running analyses.
python run_analysis.py --data-root /path/to/data --check-inputs --verify-hashes

# Inspect the exact commands.
python run_analysis.py --data-root /path/to/data --dry-run

# Run all available analyses. Results are written to ./results by default.
python run_analysis.py --data-root /path/to/data

# Run selected analyses with a separate output directory.
python run_analysis.py --data-root /path/to/data \
    --output-dir /path/to/output --steps accuracy per_template overlap
```

The data root defaults to `./data`. Published supplementary CSVs default to
`DATA_ROOT/frontiersin_revision_v3/supplementary_tables`; use
`--supplementary-dir /path/to/supplementary_tables` to change that location.
The runner uses the current manuscript settings and does not edit input files.
It stops if required inputs are absent. Check only the steps you intend to run
when you have a subset of the data:

```bash
python run_analysis.py --steps plots \
    --supplementary-dir /path/to/supplementary_tables --check-inputs
python run_analysis.py --steps plots \
    --supplementary-dir /path/to/supplementary_tables
```

All scripts also accept `--help`. When calling a script directly, set the path
variables before starting Python:

## Manuscript-to-code map

Table numbers refer to the current manuscript, not older filenames.

| Manuscript result | Runner step | Scripts and outputs |
| --- | --- | --- |
| Bound/unbound Cα comparison, Figure 1 histograms, Table S2 | `structural` | `compare_bound_unbound.py`, `print_table1_cdr_rmsd.py`, `plot_cdr_rmsd_histograms.py`; per-complex CSV, `S2_summary.txt`, histogram PDF |
| AF3 Cα comparison, Table 1, Table S3 | `structural` | `compare_af_predictions.py`, `print_table2_af_rmsd.py`; 30 model-0 comparisons, `S3.csv`, `Table1_ca_summary.txt` |
| Heavy-atom results in Sections 2.1–2.2, Figure 2, Tables S4–S5 | `heavy_atoms` | `analyze_tcr_heavy_atom_rmsd.py`; `S4.csv`, `S5.csv`, `heavy_atom_summary.csv`, residue-wise figure |
| Table 2 and Tables S10–S11 | `accuracy` | `ddg_accuracy_uncertainty.py`; pooled metrics, confidence intervals, single-versus-ensemble comparisons |
| Tables S7–S9 and S12 | `per_template` | `supp_per_template_metrics.py`; per-template Pearson, RMSE, MAE, and correlation intervals |
| Table 3 and Table S18 | `prioritization` | `cdr_prioritization_per_template.py`; `S18.csv`, `Table3_summary.csv` |
| Table 4 and Tables S19–S20 | `prioritization` | `cdr_prioritization_1ao7.py`; 1AO7 metrics and top-candidate ranks |
| Figure 3 and Tables S15–S17 | `features`, then `plots` | `analyze_af3_wt_error_features.py`, `plot_af3_wt_error_features.py`; feature tables, associations, figure |
| Shared/no-shared PDB-ID analysis, Tables S21–S22 | `overlap` | `analyze_template_overlap_performance.py`, `format_template_overlap_tables.py`; subgroup records, metrics, paired changes, contrasts, editable `S21_S22.tex` |
| Figure S1 | `plots` | `plot_af3_mutant_ensemble_saturation.py`; plots the supplied `S13.csv` |
| Figure S2 | `plots` | `plot_cdr_prioritization_boxplot.py`; plots `S18.csv` |

Generated tables are saved in `results/supplementary_tables`, figures in
`results/figures`, structural intermediates in `results/structural`, and overlap
audit outputs in `results/template_overlap`. If `features` or `prioritization`
is selected together with `plots`, plotting uses the newly generated tables.
Otherwise it uses the supplied supplementary tables.

`Table3_summary.csv` contains the median per-template Spearman correlation and
`mean_top10_recovery`, the arithmetic mean of the four recovery counts (maximum
10), as used in Table 3. Other summary columns retain the original script's
quartiles, ranges, and median precision.

The overlap calculation retains the original engine's five scaffold settings
and diagnostic outputs. The formatter selects only **RDE-PPI and Prompt-DDG,
experimental WT and AF3-WT single**, as reported in the current SI. It omits
equal-template MAE sensitivity results. The LaTeX output is a table fragment;
manuscript prose and document assembly are outside this package.

## Cohorts and analytical conventions

- The global ΔΔG comparison contains **438 ATLAS variants on 37 templates**.
  Use stable `row_idx` and ATLAS `Index` identifiers for joins; they are not
  interchangeable.
- Candidate prioritization uses variants with at least one IMGT CDR mutation
  and templates with at least 20 eligible variants: **1AO7 (126), 1MI5 (23),
  2BNR (22), and 3QDJ (20)**, totaling 191. A compound variant qualifies if any
  substitution touches a CDR.
- ΔΔG is in kcal/mol, in the WT→mutant direction. The supplied `ddg_pred`
  column already includes the sign correction for mutant scaffolds. Do not
  negate it again. More negative values rank as more stabilizing.
- ATLAS single-scaffold predictions use **seed 1/sample 0**. Ensemble
  aggregation uses the supplied per-sample scores. This selection is separate
  from the AlphaFold Server `model_0` used in the 30-complex structural
  benchmark.
- Mean, median, and ranking-score-weighted aggregation are evaluated for Table 2.
  Candidate and per-template analyses use arithmetic means for ensembles.
  Ranking weights are shifted by subtracting the minimum and adding `10^-3`.

## Package contents and validation

- `scripts/`: selected analysis and plotting scripts plus lightweight helpers.
- `ddg_predictions/atlas_af_eval/aggregate_results.py`: shared ensemble
  aggregation functions only.
- `tests/`: focused existing regression tests; data-dependent tests are marked
  `integration` and excluded from the default test command.

To include data-dependent tests after supplying the reference inputs:

The `.gitignore` excludes external data, regenerated results, weights,
environments, and caches. Upload the contents of this folder to the analysis
repository; the package does not create or update a GitHub remote.
