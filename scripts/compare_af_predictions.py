"""Compare AlphaFold Server predictions with experimental TCR-pMHC structures.

For each of the 30 complexes, compares the selected AF Server model (default 0)
against the reference structure (truncated bound PDB). Computes:
  - Per-CDR RMSD (TCR chains only, IMGT-numbered, framework-superposed)
  - Whole-complex CA RMSD (all chains, best-fit on all CAs)
  - Whole-complex CA RMSD after TCR framework superposition

Merges bound-unbound CDR RMSD for comparison. The retained rank column is 1
because one preselected model is analyzed per complex; model selection does not
use the observed RMSD.

Usage:
    python scripts/compare_af_predictions.py \
        --csv tcr3d/unbound-bound.csv \
        --af-dir tcr3d/AF_predictions \
        --ref-dir tcr3d/alphafold_input \
        --output tcr3d/af_prediction_comparison.csv
"""

import argparse
import json
import logging
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

from helpers.paths import DATA_ROOT, OUTPUT_ROOT, SUPPLEMENTARY_ROOT

import numpy as np
import pandas as pd
from Bio.SVDSuperimposer import SVDSuperimposer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_bound_unbound import (
    CDR_RANGES,
    _residues_to_sequence,
    _standard_residues,
    extract_imgt_cas,
    identify_tcr_chains,
    is_cdr_position,
    load_structure,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def match_chains_by_sequence(af_structure, ref_structure):
    """Match AF prediction chains to reference chains by sequence identity.

    Returns dict mapping af_chain_id -> ref_chain_id.
    """
    def get_chain_seqs(structure):
        seqs = {}
        for chain in structure[0]:
            cid = chain.get_id()
            residues = _standard_residues(chain)
            if len(residues) < 3:
                continue
            seqs[cid] = _residues_to_sequence(residues)
        return seqs

    af_seqs = get_chain_seqs(af_structure)
    ref_seqs = get_chain_seqs(ref_structure)

    chain_map = {}
    used_ref = set()

    for af_cid, af_seq in af_seqs.items():
        best_ref = None
        best_score = -1
        for ref_cid, ref_seq in ref_seqs.items():
            if ref_cid in used_ref:
                continue
            score = SequenceMatcher(None, af_seq, ref_seq, autojunk=False).ratio()
            if score > best_score:
                best_score = score
                best_ref = ref_cid
        if best_ref is not None and best_score > 0.5:
            chain_map[af_cid] = best_ref
            used_ref.add(best_ref)
        else:
            log.warning(f"No match for AF chain {af_cid} (len={len(af_seq)})")

    return chain_map


def _align_and_pair_cas(af_residues, ref_residues):
    """Align two residue lists by sequence and return paired CA coordinates.

    Uses SequenceMatcher to handle N/C-terminal shifts or internal gaps.
    Returns list of (af_ca_coord, ref_ca_coord) pairs.
    """
    af_seq = _residues_to_sequence(af_residues)
    ref_seq = _residues_to_sequence(ref_residues)

    pairs = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, af_seq, ref_seq, autojunk=False).get_opcodes():
        if tag != "equal":
            continue
        for ai, ri in zip(range(i1, i2), range(j1, j2)):
            if "CA" in af_residues[ai] and "CA" in ref_residues[ri]:
                pairs.append((
                    af_residues[ai]["CA"].get_vector().get_array(),
                    ref_residues[ri]["CA"].get_vector().get_array(),
                ))
    return pairs


def compute_complex_ca_rmsd(af_structure, ref_structure, chain_map):
    """Compute whole-complex CA RMSD (best-fit on all CAs).

    Pairs residues by sequence alignment to handle terminal/internal mismatches.
    Returns RMSD float, or None if insufficient atoms.
    """
    af_coords = []
    ref_coords = []

    for af_cid, ref_cid in chain_map.items():
        af_residues = _standard_residues(af_structure[0][af_cid])
        ref_residues = _standard_residues(ref_structure[0][ref_cid])
        for af_ca, ref_ca in _align_and_pair_cas(af_residues, ref_residues):
            af_coords.append(af_ca)
            ref_coords.append(ref_ca)

    if len(af_coords) < 10:
        return None

    af_arr = np.array(af_coords)
    ref_arr = np.array(ref_coords)

    sup = SVDSuperimposer()
    sup.set(ref_arr, af_arr)
    sup.run()
    return round(float(sup.get_rms()), 4)


def compute_complex_rmsd_tcr_superposed(af_structure, ref_structure, chain_map,
                                         af_tcr_chains, ref_tcr_chains):
    """Compute whole-complex CA RMSD after superposing on TCR framework CAs.

    Returns RMSD float, or None if insufficient atoms.
    """
    # Get TCR IMGT CAs for superposition
    af_tcr_cas = extract_imgt_cas(af_structure, af_tcr_chains)
    ref_tcr_cas = extract_imgt_cas(ref_structure, ref_tcr_chains)

    common_keys = sorted(set(af_tcr_cas) & set(ref_tcr_cas))
    fw_keys = [k for k in common_keys if not is_cdr_position(k)]
    if len(fw_keys) < 10:
        return None

    ref_fw = np.array([ref_tcr_cas[k] for k in fw_keys])
    af_fw = np.array([af_tcr_cas[k] for k in fw_keys])

    sup = SVDSuperimposer()
    sup.set(ref_fw, af_fw)
    sup.run()
    rot, tran = sup.get_rotran()

    # Now collect ALL chain CAs and apply the TCR-framework rotation
    af_coords = []
    ref_coords = []
    for af_cid, ref_cid in chain_map.items():
        af_residues = _standard_residues(af_structure[0][af_cid])
        ref_residues = _standard_residues(ref_structure[0][ref_cid])
        for af_ca, ref_ca in _align_and_pair_cas(af_residues, ref_residues):
            af_coords.append(af_ca)
            ref_coords.append(ref_ca)

    if len(af_coords) < 10:
        return None

    af_arr = np.array(af_coords)
    ref_arr = np.array(ref_coords)
    af_sup = np.dot(af_arr, rot) + tran

    rmsd = float(np.sqrt(np.mean(np.sum((ref_arr - af_sup) ** 2, axis=1))))
    return round(rmsd, 4)


def find_af_dir(af_base_dir, pdb_id):
    """Find the AF prediction directory for a given PDB ID."""
    af_base = Path(af_base_dir)
    prefix = pdb_id.lower() + "_"
    for d in af_base.iterdir():
        if d.is_dir() and d.name.startswith(prefix):
            return d
    return None


def find_model_files(af_dir):
    """Find model CIF and confidence JSON files in an AF prediction directory.

    Returns list of (model_index, cif_path, confidence_json_path) sorted by index.
    """
    models = []
    for cif in sorted(af_dir.glob("*_model_*.cif")):
        match = re.search(r"_model_(\d+)\.cif$", cif.name)
        if match:
            idx = int(match.group(1))
            conf_name = cif.name.replace(f"_model_{idx}.cif",
                                          f"_summary_confidences_{idx}.json")
            conf_path = af_dir / conf_name
            models.append((idx, cif, conf_path if conf_path.exists() else None))
    return models


def process_model(af_cif_path, ref_pdb_path, confidence_json_path):
    """Compare one AF model against the reference structure.

    Returns dict with all RMSD metrics and confidence scores.
    """
    af_struct = load_structure(str(af_cif_path))
    ref_struct = load_structure(str(ref_pdb_path))

    # Match chains by sequence
    chain_map = match_chains_by_sequence(af_struct, ref_struct)
    log.info(f"  Chain mapping: {chain_map}")

    # Identify TCR chains in both structures
    af_tcr = identify_tcr_chains(af_struct)
    ref_tcr = identify_tcr_chains(ref_struct)

    results = {}

    # TCR CDR RMSD (reuse existing logic)
    if not ("alpha" in af_tcr and "beta" in af_tcr and "alpha" in ref_tcr and "beta" in ref_tcr):
        log.warning(f"  Could not identify TCR chains: AF={af_tcr}, ref={ref_tcr}")
    if "alpha" in af_tcr and "beta" in af_tcr and "alpha" in ref_tcr and "beta" in ref_tcr:
        af_tcr_chains = [af_tcr["alpha"], af_tcr["beta"]]
        ref_tcr_chains = [ref_tcr["alpha"], ref_tcr["beta"]]

        af_tcr_cas = extract_imgt_cas(af_struct, af_tcr_chains)
        ref_tcr_cas = extract_imgt_cas(ref_struct, ref_tcr_chains)

        common = sorted(set(af_tcr_cas) & set(ref_tcr_cas))
        fw_keys = [k for k in common if not is_cdr_position(k)]

        if len(fw_keys) >= 10:
            ref_fw = np.array([ref_tcr_cas[k] for k in fw_keys])
            af_fw = np.array([af_tcr_cas[k] for k in fw_keys])

            sup = SVDSuperimposer()
            sup.set(ref_fw, af_fw)
            sup.run()
            rot, tran = sup.get_rotran()

            results["framework_rmsd"] = round(float(sup.get_rms()), 4)

            # Apply transform to all common TCR CAs
            all_ref = np.array([ref_tcr_cas[k] for k in common])
            all_af = np.array([af_tcr_cas[k] for k in common])
            all_af_sup = np.dot(all_af, rot) + tran

            def rmsd(a, b):
                return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))

            results["overall_tcr_rmsd"] = round(rmsd(all_ref, all_af_sup), 4)

            # All CDR combined
            idx_map = {k: i for i, k in enumerate(common)}
            all_cdr_keys = [k for k in common if is_cdr_position(k)]
            if all_cdr_keys:
                cdr_idxs = [idx_map[k] for k in all_cdr_keys]
                results["cdr_rmsd"] = round(rmsd(all_ref[cdr_idxs], all_af_sup[cdr_idxs]), 4)

            # Per-CDR per-chain
            for cdr_name, (start, end) in CDR_RANGES.items():
                for ci, chain_label in [(0, "alpha"), (1, "beta")]:
                    cdr_keys = [k for k in common if k[0] == ci and start <= k[1] <= end]
                    col = f"{cdr_name.lower()}_{chain_label}_rmsd"
                    if not cdr_keys:
                        results[col] = None
                        continue
                    idxs = [idx_map[k] for k in cdr_keys]
                    results[col] = round(rmsd(all_ref[idxs], all_af_sup[idxs]), 4)

        # Complex RMSD after TCR framework superposition
        results["tcr_pmhc_ca_rmsd"] = compute_complex_rmsd_tcr_superposed(
            af_struct, ref_struct, chain_map, af_tcr_chains, ref_tcr_chains
        )

    # Whole-complex CA RMSD (best-fit on all CAs)
    results["complex_ca_rmsd"] = compute_complex_ca_rmsd(af_struct, ref_struct, chain_map)

    # Confidence scores
    if confidence_json_path and confidence_json_path.exists():
        with open(confidence_json_path) as f:
            conf = json.load(f)
        results["ranking_score"] = conf.get("ranking_score")
        results["iptm"] = conf.get("iptm")
        results["ptm"] = conf.get("ptm")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare AlphaFold Server predictions with experimental structures"
    )
    parser.add_argument("--csv", required=True, help="Path to unbound-bound.csv")
    parser.add_argument("--af-dir", required=True, help="AF predictions directory")
    parser.add_argument("--ref-dir", required=True, help="Reference PDB directory (alphafold_input)")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--bu-csv",
        default=str(DATA_ROOT / "tcr3d/bound_unbound_comparison.csv"),
        help="Bound-unbound comparison CSV (default: tcr3d/bound_unbound_comparison.csv)",
    )
    parser.add_argument("--model", type=int, default=0, help="Server model index used in the manuscript (0, the top-ranked prediction).")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    af_base = Path(args.af_dir)
    ref_dir = Path(args.ref_dir)

    # Load bound-unbound comparison data
    bu_cols = ["cdr_rmsd", "cdr1_alpha_rmsd", "cdr1_beta_rmsd",
               "cdr2_alpha_rmsd", "cdr2_beta_rmsd",
               "cdr3_alpha_rmsd", "cdr3_beta_rmsd"]
    bu_df = pd.read_csv(args.bu_csv)
    bu_df = bu_df[["complex"] + bu_cols].copy()
    bu_df = bu_df.rename(columns={c: f"bu_{c}" for c in bu_cols})
    bu_df = bu_df.rename(columns={"complex": "pdb_id"})

    rows = []
    for _, meta in df.iterrows():
        pdb_id = meta["complex"]
        mhc_class = meta["mhc_class"]
        tcr_name = meta["tcr_name"]
        pmhc_name = meta["pmhc_name"]

        af_dir = find_af_dir(af_base, pdb_id)
        if af_dir is None:
            log.error(f"No AF prediction directory for {pdb_id}")
            continue

        ref_pdb = ref_dir / f"{pdb_id.lower()}.pdb"
        if not ref_pdb.exists():
            log.error(f"Reference PDB not found: {ref_pdb}")
            continue

        models = [model for model in find_model_files(af_dir) if model[0] == args.model]
        log.info(f"Processing {pdb_id} ({mhc_class}): {len(models)} models")

        for model_idx, cif_path, conf_path in models:
            log.info(f"  Model {model_idx}: {cif_path.name}")
            try:
                metrics = process_model(cif_path, ref_pdb, conf_path)
                row = {
                    "pdb_id": pdb_id,
                    "model": model_idx,
                    "mhc_class": mhc_class,
                    "tcr_name": tcr_name,
                    "pmhc_name": pmhc_name,
                }
                row.update(metrics)
                rows.append(row)
            except Exception as e:
                log.error(f"  Model {model_idx} failed: {e}")

    result_df = pd.DataFrame(rows)

    # Rank models per PDB by complex_ca_rmsd (lower = better)
    result_df["rank"] = (
        result_df.groupby("pdb_id")["complex_ca_rmsd"]
        .rank(method="min")
        .astype("Int64")
    )

    # Merge bound-unbound CDR RMSD
    result_df = result_df.merge(bu_df, on="pdb_id", how="left")

    # Sort by pdb_id and rank
    result_df = result_df.sort_values(["pdb_id", "rank"]).reset_index(drop=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output, index=False)
    log.info(f"\nWrote {len(result_df)} rows to {args.output}")

    # Print summary of the one preselected model per PDB.
    best = result_df[result_df["rank"] == 1].copy()
    summary_cols = ["pdb_id", "model", "complex_ca_rmsd", "tcr_pmhc_ca_rmsd",
                    "cdr_rmsd", "cdr3_beta_rmsd", "ranking_score", "iptm",
                    "bu_cdr_rmsd", "bu_cdr3_beta_rmsd"]
    existing_cols = [c for c in summary_cols if c in best.columns]
    print("\n=== Selected model per PDB ===")
    print(best[existing_cols].to_string(index=False))

    # Print aggregate stats
    print("\n=== Aggregate (selected models) ===")
    for col in ["complex_ca_rmsd", "tcr_pmhc_ca_rmsd", "cdr_rmsd", "cdr3_beta_rmsd"]:
        if col in best.columns:
            vals = best[col].dropna()
            print(f"  {col}: mean={vals.mean():.3f}, median={vals.median():.3f}, "
                  f"min={vals.min():.3f}, max={vals.max():.3f}")


if __name__ == "__main__":
    main()
