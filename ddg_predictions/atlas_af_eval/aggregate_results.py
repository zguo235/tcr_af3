"""Shared ensemble aggregation for the manuscript analysis scripts.

Arithmetic means, medians, and shifted AF3-ranking-weighted means retain
the original prediction aggregation calculations. Inputs are external CSVs.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from helpers.paths import DATA_ROOT  # noqa: E402
ATLAS_OUT = DATA_ROOT / "ATLAS_af" / "output"
ATLAS_WT_OUT = DATA_ROOT / "ATLAS_af" / "wt_output"
ATLAS_INDEX = DATA_ROOT / "ATLAS_af" / "index.csv"


def parse_ddg_exp(x):
    """ATLAS DDG cells can look like '0.19 ± 0.11' (value ± uncertainty)."""
    if pd.isna(x):
        return np.nan
    s = str(x).split("±")[0].split("?")[0].strip().replace("−", "-")
    try:
        return float(s)
    except ValueError:
        return np.nan


def ranking_weighted_mean(df_ens, ranking_csv_path):
    """Weight per-sample ΔΔG by AF3 ranking_score; nan if any weight missing.

    Args:
        df_ens: DataFrame of ensemble entries for one row_idx — must have
            ``source`` column encoding seed/sample (``..._seed{S}_sample{K}``)
            and ``ddg_pred`` column.
        ranking_csv_path: Path to an AF3 ``*_ranking_scores.csv`` with
            columns ``seed``, ``sample``, ``ranking_score``.
    """
    path = Path(ranking_csv_path) if ranking_csv_path else None
    if path is None or not path.exists():
        return np.nan
    rs = pd.read_csv(path)
    if "ranking_score" not in rs.columns:
        return np.nan
    sub = df_ens.copy()
    sub["seed"] = sub["source"].str.extract(r"seed(\d+)").astype(int)
    sub["sample"] = sub["source"].str.extract(r"sample(\d+)").astype(int)
    merged = sub.merge(rs, on=["seed", "sample"], how="left")
    w = merged["ranking_score"].to_numpy()
    y = merged["ddg_pred"].to_numpy()
    if np.isnan(w).any():
        return np.nan
    w = w - w.min() + 1e-3  # shift so weights are positive
    return float(np.average(y, weights=w))


def mutant_ranking_csv(row_idx: int, template: str) -> Path | None:
    """``mutant_af3_multi`` (approach 3): per-row dir under ATLAS_af/output/."""
    row_dirs = sorted(ATLAS_OUT.glob(f"row_{row_idx:04d}_*"))
    if not row_dirs:
        return None
    hits = list(row_dirs[0].glob("*_ranking_scores.csv"))
    return hits[0] if hits else None


def wt_ranking_csv(row_idx: int, template: str) -> Path | None:
    """``wt_af3_multi`` (approach 5): per-template dir under ATLAS_af/wt_output/."""
    path = ATLAS_WT_OUT / f"wt_{template}" / f"wt_{template}_ranking_scores.csv"
    return path if path.exists() else None


def load_template_map():
    """row_idx (int) -> template_PDB (str) from ATLAS_af/index.csv."""
    idx = pd.read_csv(ATLAS_INDEX)
    return dict(zip(idx["row_idx"].astype(int), idx["template_PDB"].astype(str)))


def ensemble_aggregate(df_ens, ranking_csv_fn, template_map):
    """Reduce one-row-per-sample ensemble to per-row stats + rankwt mean."""
    grouped = (
        df_ens.groupby("row_idx")
        .agg(
            ddg_pred_mean=("ddg_pred", "mean"),
            ddg_pred_median=("ddg_pred", "median"),
            ddg_pred_std=("ddg_pred", "std"),
            ddg_pred_min=("ddg_pred", "min"),
            ddg_pred_max=("ddg_pred", "max"),
            ddg_exp=("ddg_exp", "first"),
        )
        .reset_index()
    )
    grouped["template"] = grouped["row_idx"].map(template_map)
    grouped["ddg_pred_rankwt"] = [
        ranking_weighted_mean(
            df_ens[df_ens["row_idx"] == r],
            ranking_csv_fn(r, t),
        )
        for r, t in zip(grouped["row_idx"], grouped["template"])
    ]
    return grouped

