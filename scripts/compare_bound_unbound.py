"""Compare CDR loop conformations between bound and unbound TCR structures.

Reads tcr3d/unbound-bound.csv, prepares cleaned PDB files in tcr3d/prepared/,
superposes structures on framework (non-CDR) CA atoms using BioPython's
SVDSuperimposer, and computes per-CDR RMSD for alpha and beta chains.

Numbering: IMGT scheme via ANARCI.

Usage:
    # Single pair
    python compare_bound_unbound.py \
        --bound tcr3d/complex_cif/1ao7_renumbered.cif \
        --unbound tcr3d/tcr_structures/ab_chain/3qh3_AB_aho.pdb

    # Batch from CSV
    python compare_bound_unbound.py \
        --csv tcr3d/unbound-bound.csv \
        --cif-dir tcr3d/complex_cif \
        --ab-chain-dir tcr3d/tcr_structures/ab_chain \
        --output tcr3d/bound_unbound_comparison.csv
"""

import argparse
import logging
import os
from pathlib import Path

from helpers.paths import DATA_ROOT, OUTPUT_ROOT, SUPPLEMENTARY_ROOT

import numpy as np
import pandas as pd
import requests
from anarci import anarci
from Bio.PDB import PDBIO, MMCIFParser, PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1
from Bio.SVDSuperimposer import SVDSuperimposer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# IMGT CDR boundaries
CDR_RANGES = {
    "CDR1": (27, 38),
    "CDR2": (56, 65),
    "CDR3": (105, 117),
}

REPO_ROOT = Path(__file__).resolve().parent.parent
PREPARED_DIR = OUTPUT_ROOT / "tcr3d" / "prepared"
DOWNLOAD_CACHE = PREPARED_DIR / "downloads"


# --- Utilities ---


def load_structure(path):
    """Load a structure from PDB or CIF file."""
    path = str(path)
    parser = MMCIFParser(QUIET=True) if path.endswith(".cif") else PDBParser(QUIET=True)
    return parser.get_structure("s", path)


def download_cif(pdb_id, output_dir=None):
    """Download CIF from RCSB PDB. Returns path to downloaded file."""
    output_dir = Path(output_dir or DOWNLOAD_CACHE)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{pdb_id.upper()}.cif"
    if out_path.exists():
        return str(out_path)
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
    log.info(f"Downloading {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    return str(out_path)


def get_imgt_numbering(sequence):
    """Run ANARCI with IMGT scheme. Returns (numbering, start, end, chain_type) or None.

    start/end are 0-based indices into the input sequence delimiting the V-domain.
    """
    results = anarci(
        [("query", sequence)], scheme="imgt", allow={"A", "B", "D", "G"}
    )
    numbered, details, _ = results
    if numbered[0] is None:
        return None
    numbering, start, end = numbered[0][0]
    return numbering, start, end, details[0][0]["chain_type"]


def _standard_residues(chain):
    """Return list of standard (non-hetero) residues from a chain."""
    return [r for r in chain if r.get_id()[0] == " "]


def _residues_to_sequence(residues):
    """Convert residue list to one-letter sequence. Unknown residues become 'X'."""
    return "".join(protein_letters_3to1.get(r.get_resname(), "X") for r in residues)


def is_cdr_position(key):
    """Check if a CA key (chain_idx, imgt_num, ins_code) falls within any CDR."""
    imgt_num = key[1]
    return any(s <= imgt_num <= e for s, e in CDR_RANGES.values())


# --- TCR chain identification ---


def identify_tcr_chains(structure):
    """Identify TCR alpha and beta chains via ANARCI.

    Returns dict like {'alpha': chain_id, 'beta': chain_id}.
    When multiple candidates exist, picks the pair with closest chain IDs
    (adjacent chains in PDB are usually the same biological assembly).
    """
    candidates = {"A": [], "B": []}

    for chain in structure[0]:
        cid = chain.get_id()
        for r in chain:
            if r.get_id()[0] not in (" ", ""):
                log.warning(
                    f"Chain {cid}: skipping hetero residue "
                    f"{r.get_resname()} (id={r.get_id()})"
                )
        seq = _residues_to_sequence(_standard_residues(chain))
        if len(seq) < 50:
            continue
        result = get_imgt_numbering(seq)
        if result is None:
            continue
        numbering, _start, _end, chain_type = result
        if chain_type in candidates:
            n = sum(1 for (_, _), aa in numbering if aa != "-")
            candidates[chain_type].append((cid, n))

    alphas, betas = candidates["A"], candidates["B"]
    tcr = {}

    if len(alphas) <= 1 and len(betas) <= 1:
        if alphas:
            tcr["alpha"] = alphas[0][0]
        if betas:
            tcr["beta"] = betas[0][0]
    elif len(alphas) > 1 and len(betas) == 1:
        tcr["beta"] = betas[0][0]
        tcr["alpha"] = min(alphas, key=lambda c: abs(ord(c[0]) - ord(tcr["beta"])))[0]
    elif len(betas) > 1 and len(alphas) == 1:
        tcr["alpha"] = alphas[0][0]
        tcr["beta"] = min(betas, key=lambda c: abs(ord(c[0]) - ord(tcr["alpha"])))[0]
    else:
        best = min(
            ((a, b) for a, _ in alphas for b, _ in betas),
            key=lambda p: abs(ord(p[0]) - ord(p[1])),
            default=None,
        )
        if best:
            tcr["alpha"], tcr["beta"] = best

    return tcr


# --- CA extraction with IMGT numbering ---


def extract_imgt_cas(structure, chain_ids):
    """Extract CA coordinates keyed by (chain_idx, imgt_position, insertion_code).

    Args:
        structure: BioPython structure.
        chain_ids: [alpha_chain_id, beta_chain_id].

    Returns dict: {(chain_idx, imgt_num, ins_code): numpy array (3,)}.
    """
    cas = {}
    for ci, cid in enumerate(chain_ids):
        chain = structure[0][cid]
        for r in chain:
            if r.get_id()[0] not in (" ", ""):
                log.warning(
                    f"Chain {cid}: skipping hetero residue "
                    f"{r.get_resname()} (id={r.get_id()})"
                )
        residues = _standard_residues(chain)
        seq = _residues_to_sequence(residues)
        ca_coords = [
            r["CA"].get_vector().get_array() if "CA" in r else None
            for r in residues
        ]

        result = get_imgt_numbering(seq)
        if result is None:
            log.warning(f"ANARCI failed for chain {cid}")
            continue

        numbering, domain_start, _domain_end, _ = result
        seq_idx = domain_start
        for (pos, ins), aa in numbering:
            if aa == "-":
                continue
            if seq_idx < len(ca_coords) and ca_coords[seq_idx] is not None:
                cas[(ci, pos, ins)] = ca_coords[seq_idx]
            seq_idx += 1

    return cas


def load_tcr_cas(path):
    """Load TCR CA atoms with IMGT numbering from a CIF or PDB file.

    Returns (cas_dict, [alpha_chain_id, beta_chain_id]).
    """
    structure = load_structure(path)
    tcr = identify_tcr_chains(structure)
    if "alpha" not in tcr or "beta" not in tcr:
        raise ValueError(f"Could not identify both TCR chains in {path}. Found: {tcr}")
    chain_ids = [tcr["alpha"], tcr["beta"]]
    return extract_imgt_cas(structure, chain_ids), chain_ids


# --- Superposition and RMSD (BioPython SVDSuperimposer) ---


def superpose_and_compute(bound_cas, unbound_cas):
    """Superpose unbound onto bound using framework CAs, compute per-CDR RMSD.

    Uses BioPython SVDSuperimposer for optimal rigid-body superposition.

    Returns dict with framework_rmsd, overall_rmsd, per-CDR per-chain RMSD,
    and residue counts.
    """
    common = sorted(set(bound_cas) & set(unbound_cas))
    if len(common) < 10:
        raise ValueError(f"Only {len(common)} common CA positions found")

    fw_keys = [k for k in common if not is_cdr_position(k)]
    if len(fw_keys) < 10:
        raise ValueError(f"Only {len(fw_keys)} framework positions found")

    log.info(f"Superposing on {len(fw_keys)} framework CAs, total {len(common)} common CAs")
    bound_fw = np.array([bound_cas[k] for k in fw_keys])
    unbound_fw = np.array([unbound_cas[k] for k in fw_keys])

    # Fit superposition on framework CAs
    sup = SVDSuperimposer()
    sup.set(bound_fw, unbound_fw)
    sup.run()
    rot, tran = sup.get_rotran()

    # Apply transformation to all unbound coordinates
    all_bound = np.array([bound_cas[k] for k in common])
    all_unbound = np.array([unbound_cas[k] for k in common])
    all_unbound_sup = np.dot(all_unbound, rot) + tran

    def rmsd(a, b):
        return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))

    results = {
        "framework_rmsd": round(sup.get_rms(), 4),
        "overall_rmsd": round(rmsd(all_bound, all_unbound_sup), 4),
        "n_common_residues": len(common),
        "n_framework_residues": len(fw_keys),
    }

    idx_map = {k: i for i, k in enumerate(common)}

    # All CDR positions combined
    all_cdr_keys = [k for k in common if is_cdr_position(k)]
    if all_cdr_keys:
        cdr_idxs = [idx_map[k] for k in all_cdr_keys]
        results["cdr_rmsd"] = round(rmsd(all_bound[cdr_idxs], all_unbound_sup[cdr_idxs]), 4)
    else:
        results["cdr_rmsd"] = None

    # Per-CDR per-chain
    for cdr_name, (start, end) in CDR_RANGES.items():
        for ci, chain_label in [(0, "alpha"), (1, "beta")]:
            cdr_keys = [k for k in common if k[0] == ci and start <= k[1] <= end]
            col = f"{cdr_name.lower()}_{chain_label}_rmsd"
            if not cdr_keys:
                results[col] = None
                continue
            idxs = [idx_map[k] for k in cdr_keys]
            results[col] = round(rmsd(all_bound[idxs], all_unbound_sup[idxs]), 4)

    return results


# --- File resolution ---


def resolve_bound_file(complex_pdb_id, cif_dir):
    """Find or download the complex CIF file."""
    local = Path(cif_dir) / f"{complex_pdb_id.lower()}_renumbered.cif"
    if local.exists():
        return str(local)
    log.info(f"Complex CIF not found locally for {complex_pdb_id}, downloading")
    return download_cif(complex_pdb_id)


def resolve_unbound_file(unbound_pdb_id, ab_chain_dir):
    """Find or download the unbound TCR file."""
    ab_chain_dir = Path(ab_chain_dir)
    prefix = unbound_pdb_id.lower() + "_"
    candidates = sorted(f for f in os.listdir(ab_chain_dir) if f.startswith(prefix))
    if candidates:
        return str(ab_chain_dir / candidates[0])
    log.info(f"Unbound ab_chain not found locally for {unbound_pdb_id}, downloading")
    return download_cif(unbound_pdb_id)


# --- Single pair and batch ---

def compare_pair(bound_path, unbound_path):
    """Compare a single bound/unbound pair. Returns results dict."""
    bound_cas, bound_chains = load_tcr_cas(bound_path)
    unbound_cas, unbound_chains = load_tcr_cas(unbound_path)
    log.info(f"Bound TCR chains: {bound_chains}, Unbound TCR chains: {unbound_chains}")
    return superpose_and_compute(bound_cas, unbound_cas)


def save_pdb(source_path, output_path):
    """Load a structure and re-save as PDB."""
    structure = load_structure(source_path)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_path))


def process_csv(csv_path, cif_dir, ab_chain_dir, output_csv):
    """Process all pairs from unbound-bound.csv."""
    df = pd.read_csv(csv_path)
    bound_dir = PREPARED_DIR / "bound"
    unbound_dir = PREPARED_DIR / "unbound"
    bound_dir.mkdir(parents=True, exist_ok=True)
    unbound_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, row in df.iterrows():
        complex_id, unbound_id = row["complex"], row["unbound_tcr"]
        log.info(f"Processing {row['id']}: {complex_id} vs {unbound_id}")
        result = dict(row)
        try:
            bound_source = resolve_bound_file(complex_id, cif_dir)
            unbound_source = resolve_unbound_file(unbound_id, ab_chain_dir)

            bound_pdb = bound_dir / f"{complex_id.lower()}_bound.pdb"
            unbound_pdb = unbound_dir / f"{complex_id.lower()}_{unbound_id.lower()}_unbound.pdb"
            if not bound_pdb.exists():
                log.info(f"Saving bound structure to {bound_pdb}")
                save_pdb(bound_source, bound_pdb)
            if not unbound_pdb.exists():
                log.info(f"Saving unbound structure to {unbound_pdb}")
                save_pdb(unbound_source, unbound_pdb)

            result.update(compare_pair(str(bound_pdb), str(unbound_pdb)))
            result["bound_file"] = str(bound_pdb.relative_to(OUTPUT_ROOT))
            result["unbound_file"] = str(unbound_pdb.relative_to(OUTPUT_ROOT))
            result["status"] = "ok"
        except Exception as e:
            log.error(f"  Error: {e}")
            result["status"] = str(e)
        rows.append(result)

    out_df = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    log.info(f"Wrote {len(out_df)} rows to {output_csv}")

    ok = out_df[out_df["status"] == "ok"]
    log.info(f"  {len(ok)}/{len(out_df)} pairs processed successfully")
    for _, r in out_df[out_df["status"] != "ok"].iterrows():
        log.warning(f"  FAILED {r['complex']}: {r['status']}")

    return out_df


# --- CLI ---

def compare_prepared_pairs(csv_path, prepared_dir, output_csv):
    """Compare the 30 curated pairs without downloading or rewriting structures.

    csv_path supplies the pair metadata; prepared_dir contains bound/ and
    unbound/ PDBs. Write one comparison row per pair to output_csv and return
    the resulting DataFrame. Missing or invalid pairs raise an error.
    """
    rows = []
    prepared_dir = Path(prepared_dir)
    for _, metadata in pd.read_csv(csv_path).iterrows():
        complex_id = str(metadata["complex"]).lower()
        unbound_id = str(metadata["unbound_tcr"]).lower()
        bound = prepared_dir / "bound" / f"{complex_id}_bound.pdb"
        unbound = prepared_dir / "unbound" / f"{complex_id}_{unbound_id}_unbound.pdb"
        result = dict(metadata)
        result.update(compare_pair(bound, unbound))
        result.update(bound_file=str(bound), unbound_file=str(unbound), status="ok")
        rows.append(result)
    frame = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return frame


def main():
    parser = argparse.ArgumentParser(
        description="Compare bound vs unbound TCR CDR loop conformations"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv", help="Path to unbound-bound.csv for batch mode")
    group.add_argument("--bound", help="Path to bound complex CIF/PDB file")
    parser.add_argument("--unbound", help="Path to unbound TCR PDB file")
    parser.add_argument("--prepared-dir", type=Path,
                        help="Use existing curated bound/ and unbound/ PDBs without downloading or modifying them.")
    parser.add_argument("--cif-dir", default=str(DATA_ROOT / "tcr3d/complex_cif"))
    parser.add_argument("--ab-chain-dir", default=str(DATA_ROOT / "tcr3d/tcr_structures/ab_chain"))
    parser.add_argument("--output", default=str(OUTPUT_ROOT / "tcr3d/bound_unbound_comparison.csv"))

    args = parser.parse_args()

    if args.bound:
        if not args.unbound:
            parser.error("--unbound is required with --bound")
        results = compare_pair(args.bound, args.unbound)
        print("Results:")
        for k, v in results.items():
            if v is not None:
                print(f"  {k}: {v}")
    elif args.prepared_dir:
        compare_prepared_pairs(args.csv, args.prepared_dir, args.output)
    else:
        process_csv(args.csv, args.cif_dir, args.ab_chain_dir, args.output)


if __name__ == "__main__":
    main()
