"""Compare experimental TCR motion and AF3 model-0 error using heavy atoms.

Structures are aligned on framework C-alpha atoms after IMGT numbering. RMSDs
are then computed for identical residues from common non-hydrogen atom names,
with OXT excluded and chemically symmetric side-chain atoms matched in the
orientation that minimizes their post-superposition distance.
"""

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

from helpers.paths import DATA_ROOT, OUTPUT_ROOT, SUPPLEMENTARY_ROOT

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio.SVDSuperimposer import SVDSuperimposer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_af_predictions import find_af_dir, find_model_files  # noqa: E402
from compare_bound_unbound import (  # noqa: E402
    CDR_RANGES,
    _residues_to_sequence,
    _standard_residues,
    get_imgt_numbering,
    identify_tcr_chains,
    load_structure,
)
from helpers.correlation_uncertainty import stable_cell_seed  # noqa: E402

log = logging.getLogger(__name__)

BACKBONE_ATOMS = {"N", "CA", "C", "O"}
SYMMETRY_SWAPS = {
    "ASP": {"OD1": "OD2", "OD2": "OD1"},
    "GLU": {"OE1": "OE2", "OE2": "OE1"},
    "ARG": {"NH1": "NH2", "NH2": "NH1"},
    "LEU": {"CD1": "CD2", "CD2": "CD1"},
    "VAL": {"CG1": "CG2", "CG2": "CG1"},
    "PHE": {"CD1": "CD2", "CD2": "CD1", "CE1": "CE2", "CE2": "CE1"},
    "TYR": {"CD1": "CD2", "CD2": "CD1", "CE1": "CE2", "CE2": "CE1"},
}

REGION_PREFIXES = [
    "framework",
    "cdr_all",
    "cdr1_alpha",
    "cdr2_alpha",
    "cdr3_alpha",
    "cdr1_beta",
    "cdr2_beta",
    "cdr3_beta",
    "interface",
]

COHORT_LABELS = {
    "bound_unbound": "Experimental bound–unbound",
    "af3_model0": "AF3–experimental bound",
}


def extract_imgt_residues(structure) -> tuple[dict[tuple[str, int, str], object], dict[str, str]]:
    """Return IMGT-keyed alpha/beta residue objects and their structure chain IDs.

    Args:
        structure: A Bio.PDB structure containing a complete alpha/beta TCR.

    Returns:
        ``(residue_map, chain_ids)`` where residue keys are
        ``(alpha_or_beta, IMGT_position, insertion_code)`` and ``chain_ids``
        maps ``alpha`` and ``beta`` to their structure-specific chain IDs.
    """
    chain_ids = identify_tcr_chains(structure)
    if "alpha" not in chain_ids or "beta" not in chain_ids:
        raise ValueError(f"could not identify both TCR chains: {chain_ids}")

    residue_map: dict[tuple[str, int, str], object] = {}
    for chain_label in ("alpha", "beta"):
        chain = structure[0][chain_ids[chain_label]]
        residues = _standard_residues(chain)
        sequence = _residues_to_sequence(residues)
        numbering_result = get_imgt_numbering(sequence)
        if numbering_result is None:
            raise ValueError(f"ANARCI failed for {chain_label} chain {chain.id}")
        numbering, domain_start, _domain_end, _chain_type = numbering_result
        sequence_index = domain_start
        for (position, insertion_code), amino_acid in numbering:
            if amino_acid == "-":
                continue
            if sequence_index >= len(residues):
                break
            residue_map[(chain_label, int(position), str(insertion_code).strip())] = (
                residues[sequence_index]
            )
            sequence_index += 1
    return residue_map, chain_ids


def identical_residue_keys(
    reference_residues: dict[tuple[str, int, str], object],
    mobile_residues: dict[tuple[str, int, str], object],
) -> tuple[list[tuple[str, int, str]], list[tuple[str, int, str]]]:
    """Split common IMGT positions into identical and residue-mismatched keys."""
    common = sorted(set(reference_residues) & set(mobile_residues))
    identical = [
        key
        for key in common
        if reference_residues[key].get_resname() == mobile_residues[key].get_resname()
    ]
    mismatched = [key for key in common if key not in set(identical)]
    return identical, mismatched


def fit_rotation_translation(
    reference_coordinates: np.ndarray, mobile_coordinates: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit mobile coordinates onto reference coordinates and return the transform.

    Args:
        reference_coordinates: ``(n, 3)`` coordinates defining the target frame.
        mobile_coordinates: Paired ``(n, 3)`` coordinates to transform.

    Returns:
        ``(rotation, translation, fitted_rmsd)`` where transformed mobile row
        vectors are calculated as ``mobile @ rotation + translation``.
    """
    if reference_coordinates.shape != mobile_coordinates.shape:
        raise ValueError("reference and mobile coordinate arrays must have equal shape")
    if reference_coordinates.ndim != 2 or reference_coordinates.shape[1] != 3:
        raise ValueError("coordinate arrays must have shape (n, 3)")
    if len(reference_coordinates) < 3:
        raise ValueError("at least three coordinate pairs are required for alignment")
    superimposer = SVDSuperimposer()
    superimposer.set(reference_coordinates, mobile_coordinates)
    superimposer.run()
    rotation, translation = superimposer.get_rotran()
    return rotation, translation, float(superimposer.get_rms())


def _is_heavy_atom(atom) -> bool:
    """Return whether a Bio.PDB atom is a retained non-hydrogen atom."""
    atom_name = atom.get_name().strip().upper()
    if atom_name == "OXT":
        return False
    element = (getattr(atom, "element", "") or "").strip().upper()
    if element in {"H", "D"}:
        return False
    name_without_leading_digits = atom_name.lstrip("0123456789")
    return not name_without_leading_digits.startswith(("H", "D"))


def residue_atom_coordinates(residue) -> dict[str, np.ndarray]:
    """Return retained heavy-atom coordinates from one residue by atom name."""
    return {
        atom.get_name().strip().upper(): np.asarray(atom.get_coord(), dtype=float)
        for atom in residue.get_atoms()
        if _is_heavy_atom(atom)
    }


def paired_atom_coordinates(
    reference_atoms: dict[str, np.ndarray],
    mobile_atoms: dict[str, np.ndarray],
    *,
    residue_name: str,
    rotation: np.ndarray,
    translation: np.ndarray,
    sidechain_only: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Pair common atom names, selecting the best valid symmetry orientation.

    Args:
        reference_atoms: Reference coordinates keyed by atom name.
        mobile_atoms: Mobile coordinates keyed by atom name before alignment.
        residue_name: Three-letter residue name used to select symmetry rules.
        rotation: Mobile-to-reference row-vector rotation.
        translation: Mobile-to-reference translation.
        sidechain_only: If true, omit N, CA, C, and O.

    Returns:
        ``(reference, transformed_mobile, reference_atom_names)``. Both arrays
        have shape ``(n_atoms, 3)``; empty inputs return two empty arrays.
    """
    common_names = sorted(set(reference_atoms) & set(mobile_atoms))
    if sidechain_only:
        common_names = [name for name in common_names if name not in BACKBONE_ATOMS]
    if not common_names:
        empty = np.empty((0, 3), dtype=float)
        return empty, empty.copy(), []

    direct_mapping = {name: name for name in common_names}
    candidate_mappings = [direct_mapping]
    swap = SYMMETRY_SWAPS.get(residue_name)
    if swap:
        symmetric_names = set(swap)
        if symmetric_names <= set(reference_atoms) and symmetric_names <= set(mobile_atoms):
            candidate_mappings.append(
                {name: swap.get(name, name) for name in common_names}
            )

    reference_array = np.asarray([reference_atoms[name] for name in common_names])
    best_mobile = None
    best_squared_error = np.inf
    for mapping in candidate_mappings:
        mobile_array = np.asarray([mobile_atoms[mapping[name]] for name in common_names])
        transformed_mobile = mobile_array @ rotation + translation
        squared_error = float(np.sum((reference_array - transformed_mobile) ** 2))
        if squared_error < best_squared_error:
            best_squared_error = squared_error
            best_mobile = transformed_mobile
    if best_mobile is None:
        raise RuntimeError("no atom mapping was evaluated")
    return reference_array, best_mobile, common_names


def rmsd_from_squared_distances(squared_distances: list[float]) -> float:
    """Return RMSD from pooled per-atom squared distances, or NaN if empty."""
    if not squared_distances:
        return np.nan
    return float(np.sqrt(np.mean(squared_distances)))


def region_for_key(key: tuple[str, int, str]) -> str:
    """Return framework or CDR1/2/3 for one IMGT residue key."""
    _chain, position, _insertion = key
    for cdr_name, (start, end) in CDR_RANGES.items():
        if start <= position <= end:
            return cdr_name
    return "framework"


def any_within_cutoff(
    residue_coordinates: np.ndarray,
    partner_coordinates: np.ndarray,
    cutoff_angstrom: float,
) -> bool:
    """Return whether any residue/partner atom pair lies within the cutoff."""
    if not len(residue_coordinates) or not len(partner_coordinates):
        return False
    differences = residue_coordinates[:, None, :] - partner_coordinates[None, :, :]
    squared_distances = np.sum(differences**2, axis=2)
    return bool(np.any(squared_distances <= cutoff_angstrom**2))


def reference_interface_keys(
    structure,
    tcr_chain_ids: dict[str, str],
    tcr_residues: dict[tuple[str, int, str], object],
    cutoff_angstrom: float,
) -> set[tuple[str, int, str]]:
    """Define reference TCR interface residues against all non-TCR protein chains."""
    tcr_chain_id_set = set(tcr_chain_ids.values())
    partner_coordinates = []
    for chain in structure[0]:
        if chain.id in tcr_chain_id_set:
            continue
        for residue in _standard_residues(chain):
            partner_coordinates.extend(residue_atom_coordinates(residue).values())
    partner_array = np.asarray(partner_coordinates, dtype=float)
    interface = set()
    for key, residue in tcr_residues.items():
        residue_coordinates = np.asarray(
            list(residue_atom_coordinates(residue).values()), dtype=float
        )
        if any_within_cutoff(residue_coordinates, partner_array, cutoff_angstrom):
            interface.add(key)
    return interface


def select_model_zero(models: list[tuple[int, Path, Path | None]]) -> tuple[int, Path, Path | None]:
    """Return the AF model-0 tuple, failing explicitly when it is unavailable."""
    matches = [model for model in models if model[0] == 0]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one AF model 0, found {len(matches)}")
    return matches[0]


def _region_prefixes_for_residue(
    key: tuple[str, int, str], is_interface: bool
) -> list[str]:
    """Return every aggregate region to which a residue contributes."""
    chain, _position, _insertion = key
    region = region_for_key(key)
    prefixes = ["framework"] if region == "framework" else [
        "cdr_all",
        f"{region.lower()}_{chain}",
    ]
    if is_interface:
        prefixes.append("interface")
    return prefixes


def compare_tcr_pair(
    reference_path: Path,
    mobile_path: Path,
    *,
    comparison_type: str,
    metadata: dict[str, object],
    interface_cutoff_angstrom: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Compute pooled regional and residue-wise heavy-atom RMSDs for one pair.

    Args:
        reference_path: Experimental bound/reference complex.
        mobile_path: Experimental unbound TCR or AF3 prediction.
        comparison_type: ``bound_unbound`` or ``af3_model0``.
        metadata: Values copied into the complex and residue output rows.
        interface_cutoff_angstrom: Reference heavy-atom interface cutoff.

    Returns:
        ``(complex_row, residue_rows)`` for Supplementary Tables S4 and S5.
    """
    reference_structure = load_structure(str(reference_path))
    mobile_structure = load_structure(str(mobile_path))
    reference_residues, reference_chains = extract_imgt_residues(reference_structure)
    mobile_residues, _mobile_chains = extract_imgt_residues(mobile_structure)
    identical_keys, mismatched_keys = identical_residue_keys(
        reference_residues, mobile_residues
    )
    interface_keys = reference_interface_keys(
        reference_structure,
        reference_chains,
        reference_residues,
        interface_cutoff_angstrom,
    )

    framework_keys = [
        key
        for key in identical_keys
        if region_for_key(key) == "framework"
        and "CA" in reference_residues[key]
        and "CA" in mobile_residues[key]
    ]
    if len(framework_keys) < 10:
        raise ValueError(f"only {len(framework_keys)} identical framework CA pairs")
    reference_framework = np.asarray(
        [reference_residues[key]["CA"].get_coord() for key in framework_keys],
        dtype=float,
    )
    mobile_framework = np.asarray(
        [mobile_residues[key]["CA"].get_coord() for key in framework_keys],
        dtype=float,
    )
    rotation, translation, framework_ca_fit_rmsd = fit_rotation_translation(
        reference_framework, mobile_framework
    )

    squared_distances: dict[str, dict[str, list[float]]] = {
        prefix: {"all_atom": [], "sidechain": []} for prefix in REGION_PREFIXES
    }
    residue_counts: dict[str, int] = defaultdict(int)
    residue_rows: list[dict[str, object]] = []

    for key in identical_keys:
        reference_residue = reference_residues[key]
        mobile_residue = mobile_residues[key]
        reference_atoms = residue_atom_coordinates(reference_residue)
        mobile_atoms = residue_atom_coordinates(mobile_residue)
        all_reference, all_mobile, _all_names = paired_atom_coordinates(
            reference_atoms,
            mobile_atoms,
            residue_name=reference_residue.get_resname(),
            rotation=rotation,
            translation=translation,
            sidechain_only=False,
        )
        side_reference, side_mobile, _side_names = paired_atom_coordinates(
            reference_atoms,
            mobile_atoms,
            residue_name=reference_residue.get_resname(),
            rotation=rotation,
            translation=translation,
            sidechain_only=True,
        )
        all_squared = np.sum((all_reference - all_mobile) ** 2, axis=1).tolist()
        side_squared = np.sum((side_reference - side_mobile) ** 2, axis=1).tolist()
        is_interface = key in interface_keys
        prefixes = _region_prefixes_for_residue(key, is_interface)
        for prefix in prefixes:
            squared_distances[prefix]["all_atom"].extend(all_squared)
            squared_distances[prefix]["sidechain"].extend(side_squared)
            residue_counts[prefix] += 1

        chain, position, insertion_code = key
        residue_rows.append({
            **metadata,
            "comparison_type": comparison_type,
            "chain": chain,
            "imgt_position": position,
            "insertion_code": insertion_code,
            "region": region_for_key(key),
            "residue_name": reference_residue.get_resname(),
            "is_reference_interface_5a": is_interface,
            "n_all_atoms": len(all_squared),
            "all_atom_rmsd_angstrom": rmsd_from_squared_distances(all_squared),
            "n_sidechain_atoms": len(side_squared),
            "sidechain_rmsd_angstrom": rmsd_from_squared_distances(side_squared),
        })

    mismatch_descriptions = [
        (
            f"{key[0]}:{key[1]}{key[2]}:"
            f"{reference_residues[key].get_resname()}/"
            f"{mobile_residues[key].get_resname()}"
        )
        for key in mismatched_keys
    ]
    complex_row: dict[str, object] = {
        **metadata,
        "comparison_type": comparison_type,
        "reference_path": str(reference_path),
        "mobile_path": str(mobile_path),
        "framework_ca_alignment_rmsd_angstrom": framework_ca_fit_rmsd,
        "n_framework_ca_alignment_residues": len(framework_keys),
        "n_common_imgt_positions": len(set(reference_residues) & set(mobile_residues)),
        "n_identical_residues_compared": len(identical_keys),
        "n_mismatched_residues_excluded": len(mismatched_keys),
        "mismatched_residues": ";".join(mismatch_descriptions),
        "interface_cutoff_angstrom": interface_cutoff_angstrom,
    }
    for prefix in REGION_PREFIXES:
        complex_row[f"{prefix}_n_residues"] = residue_counts[prefix]
        for atom_subset in ("all_atom", "sidechain"):
            values = squared_distances[prefix][atom_subset]
            complex_row[f"{prefix}_n_{atom_subset}_atoms"] = len(values)
            complex_row[f"{prefix}_{atom_subset}_rmsd_angstrom"] = (
                rmsd_from_squared_distances(values)
            )
    return complex_row, residue_rows


def bootstrap_mean_interval(
    values: np.ndarray, *, replicates: int, seed: int
) -> tuple[float, float]:
    """Return a 95% percentile interval for the arithmetic mean."""
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if not len(finite_values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(finite_values), size=(replicates, len(finite_values)))
    bootstrap_means = finite_values[indices].mean(axis=1)
    return tuple(np.quantile(bootstrap_means, [0.025, 0.975]).tolist())


def summarize_paired_complex_metrics(
    complex_table: pd.DataFrame, *, replicates: int, base_seed: int
) -> pd.DataFrame:
    """Summarize both cohorts and paired AF3-minus-experimental differences."""
    rmsd_columns = [
        column
        for column in complex_table.columns
        if column.endswith("_rmsd_angstrom")
        and column != "framework_ca_alignment_rmsd_angstrom"
    ]
    pivot_source = complex_table.set_index(["complex", "comparison_type"])
    rows = []
    for metric in rmsd_columns:
        bound_unbound = pivot_source[metric].unstack("comparison_type")["bound_unbound"]
        af3 = pivot_source[metric].unstack("comparison_type")["af3_model0"]
        paired = pd.concat(
            [bound_unbound.rename("bound_unbound"), af3.rename("af3_model0")], axis=1
        ).dropna()
        differences = paired["af3_model0"].to_numpy() - paired["bound_unbound"].to_numpy()
        seed = stable_cell_seed(base_seed, metric, "paired_mean_difference")
        ci_low, ci_high = bootstrap_mean_interval(
            differences, replicates=replicates, seed=seed
        )
        rows.append({
            "metric": metric,
            "n_pairs": len(paired),
            "bound_unbound_mean": paired["bound_unbound"].mean(),
            "bound_unbound_sd": paired["bound_unbound"].std(ddof=1),
            "af3_model0_mean": paired["af3_model0"].mean(),
            "af3_model0_sd": paired["af3_model0"].std(ddof=1),
            "paired_mean_difference_af3_minus_bound_unbound": differences.mean(),
            "paired_difference_ci95_low": ci_low,
            "paired_difference_ci95_high": ci_high,
            "af3_lower_count": int(
                (paired["af3_model0"] < paired["bound_unbound"]).sum()
            ),
            "bootstrap_replicates": replicates,
            "bootstrap_seed": seed,
        })
    return pd.DataFrame(rows)


def _collapse_insertions(residue_table: pd.DataFrame) -> pd.DataFrame:
    """Collapse insertion-coded residues into one weighted RMSD per base IMGT site."""
    rows = []
    grouping = ["comparison_type", "complex", "chain", "imgt_position"]
    for keys, group in residue_table.groupby(grouping, sort=True):
        atom_counts = group["n_all_atoms"].to_numpy(dtype=float)
        rmsd_values = group["all_atom_rmsd_angstrom"].to_numpy(dtype=float)
        valid = (atom_counts > 0) & np.isfinite(rmsd_values)
        if not np.any(valid):
            continue
        pooled_rmsd = np.sqrt(
            np.sum(atom_counts[valid] * rmsd_values[valid] ** 2)
            / np.sum(atom_counts[valid])
        )
        rows.append({
            "comparison_type": keys[0],
            "complex": keys[1],
            "chain": keys[2],
            "imgt_position": keys[3],
            "all_atom_rmsd_angstrom": pooled_rmsd,
        })
    return pd.DataFrame(rows)


def build_residue_profile_figure(
    residue_table: pd.DataFrame,
    *,
    replicates: int,
    base_seed: int,
    minimum_complexes: int,
) -> plt.Figure:
    """Build alpha- and beta-chain residue-wise RMSD profiles with bootstrap CIs."""
    collapsed = _collapse_insertions(residue_table)
    styles = {
        "bound_unbound": {"color": "#AD1457", "linestyle": "-"},
        "af3_model0": {"color": "#0D47A1", "linestyle": "--"},
    }
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True, sharey=True)
    for ax, chain in zip(axes, ("alpha", "beta")):
        chain_table = collapsed[collapsed["chain"] == chain]
        for comparison_type in ("bound_unbound", "af3_model0"):
            source = chain_table[chain_table["comparison_type"] == comparison_type]
            positions = []
            means = []
            lows = []
            highs = []
            for position, group in source.groupby("imgt_position", sort=True):
                values = group["all_atom_rmsd_angstrom"].dropna().to_numpy(dtype=float)
                if len(values) < minimum_complexes:
                    continue
                seed = stable_cell_seed(base_seed, chain, comparison_type, position)
                ci_low, ci_high = bootstrap_mean_interval(
                    values, replicates=replicates, seed=seed
                )
                positions.append(position)
                means.append(float(values.mean()))
                lows.append(ci_low)
                highs.append(ci_high)
            positions_array = np.asarray(positions, dtype=float)
            means_array = np.asarray(means, dtype=float)
            style = styles[comparison_type]
            ax.plot(
                positions_array,
                means_array,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.4,
                label=COHORT_LABELS[comparison_type],
            )
            ax.fill_between(
                positions_array,
                np.asarray(lows),
                np.asarray(highs),
                color=style["color"],
                alpha=0.16,
                linewidth=0,
            )
        for cdr_name, (start, end) in CDR_RANGES.items():
            ax.axvspan(start, end, color="#B0BEC5", alpha=0.20, linewidth=0)
            ax.text(
                (start + end) / 2,
                1.015,
                cdr_name,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=10,
                color="#455A64",
                clip_on=False,
            )
        ax.set_ylabel(f"TCR {chain}\nmean all-heavy-atom RMSD (Å)", fontsize=11.5)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(axis="y", color="#D9D9D9", linestyle=":", linewidth=0.7)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel("IMGT residue position", fontsize=12)
    axes[-1].set_xlim(1, 128)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, frameon=False, loc="upper left", fontsize=11)
    fig.suptitle(
        "Residue-wise TCR heavy-atom displacement",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def resolve_pair_paths(
    metadata_row: pd.Series,
    *,
    bound_dir: Path,
    unbound_dir: Path,
    af_dir: Path,
    reference_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    """Resolve the experimental pair and AF3 model-0/reference paths for one complex."""
    complex_id = str(metadata_row["complex"]).lower()
    unbound_id = str(metadata_row["unbound_tcr"]).lower()
    bound_path = bound_dir / f"{complex_id}_bound.pdb"
    unbound_path = unbound_dir / f"{complex_id}_{unbound_id}_unbound.pdb"
    reference_path = reference_dir / f"{complex_id}.pdb"
    prediction_directory = find_af_dir(af_dir, complex_id)
    if prediction_directory is None:
        raise FileNotFoundError(f"AF prediction directory missing for {complex_id}")
    _model_index, af_model_path, _confidence_path = select_model_zero(
        find_model_files(prediction_directory)
    )
    for path in (bound_path, unbound_path, reference_path, af_model_path):
        if not path.exists():
            raise FileNotFoundError(path)
    return bound_path, unbound_path, reference_path, af_model_path


def prepare_reader_facing_table(table: pd.DataFrame) -> pd.DataFrame:
    """Remove internal AF3 identifiers and paths from a submitted result table.

    Args:
        table: Per-complex or per-residue analysis table using internal
        ``comparison_type`` labels and, optionally, internal AF3 model and
        structure-path columns.

    Returns:
        A copy suitable for Supplementary Tables S4 and S5. Internal model
        numbers and structure paths are omitted, and the selected AF3
        comparison is labeled as the top-ranked AF3--experimental-bound
        comparison.
    """
    reader_table = table.drop(
        columns=["af3_model", "reference_path", "mobile_path"], errors="ignore"
    ).copy()
    if "comparison_type" in reader_table.columns:
        reader_table["comparison_type"] = reader_table["comparison_type"].replace(
            {"af3_model0": "top_ranked_af3_experimental_bound"}
        )
    return reader_table


def main() -> None:
    """Run all 60 comparisons and write complex, residue, summary, and figure files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default=str(DATA_ROOT / "tcr3d/unbound-bound.csv"))
    parser.add_argument("--bound-dir", default=str(DATA_ROOT / "tcr3d/prepared/bound"))
    parser.add_argument("--unbound-dir", default=str(DATA_ROOT / "tcr3d/prepared/unbound"))
    parser.add_argument("--af-dir", default=str(DATA_ROOT / "tcr3d/AF_predictions"))
    parser.add_argument("--reference-dir", default=str(DATA_ROOT / "tcr3d/alphafold_input"))
    parser.add_argument("--output-complex", required=True)
    parser.add_argument("--output-residue", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-figure-png", required=True)
    parser.add_argument("--output-figure-pdf", required=True)
    parser.add_argument("--interface-cutoff", type=float, default=5.0)
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_722)
    parser.add_argument("--minimum-profile-complexes", type=int, default=15)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    metadata_table = pd.read_csv(args.metadata)
    complex_rows = []
    residue_rows = []
    for _, metadata_row in metadata_table.iterrows():
        complex_id = str(metadata_row["complex"]).upper()
        log.info("Processing %s", complex_id)
        bound_path, unbound_path, reference_path, af_model_path = resolve_pair_paths(
            metadata_row,
            bound_dir=Path(args.bound_dir),
            unbound_dir=Path(args.unbound_dir),
            af_dir=Path(args.af_dir),
            reference_dir=Path(args.reference_dir),
        )
        shared_metadata = {
            "id": int(metadata_row["id"]),
            "complex": complex_id,
            "unbound_tcr": metadata_row["unbound_tcr"],
            "mhc_class": metadata_row["mhc_class"],
            "tcr_name": metadata_row["tcr_name"],
            "pmhc_name": metadata_row["pmhc_name"],
        }
        experimental_row, experimental_residues = compare_tcr_pair(
            bound_path,
            unbound_path,
            comparison_type="bound_unbound",
            metadata={**shared_metadata, "af3_model": np.nan},
            interface_cutoff_angstrom=args.interface_cutoff,
        )
        af3_row, af3_residues = compare_tcr_pair(
            reference_path,
            af_model_path,
            comparison_type="af3_model0",
            metadata={**shared_metadata, "af3_model": 0},
            interface_cutoff_angstrom=args.interface_cutoff,
        )
        complex_rows.extend([experimental_row, af3_row])
        residue_rows.extend(experimental_residues)
        residue_rows.extend(af3_residues)

    complex_table = pd.DataFrame(complex_rows).sort_values(
        ["id", "comparison_type"]
    )
    residue_table = pd.DataFrame(residue_rows).sort_values(
        ["id", "comparison_type", "chain", "imgt_position", "insertion_code"]
    )
    if len(complex_table) != 2 * len(metadata_table):
        raise RuntimeError(
            f"expected {2 * len(metadata_table)} comparison rows, got {len(complex_table)}"
        )

    summary_table = summarize_paired_complex_metrics(
        complex_table,
        replicates=args.bootstrap_reps,
        base_seed=args.bootstrap_seed,
    )
    figure = build_residue_profile_figure(
        residue_table,
        replicates=args.bootstrap_reps,
        base_seed=args.bootstrap_seed,
        minimum_complexes=args.minimum_profile_complexes,
    )

    outputs = {
        "complex": (
            Path(args.output_complex),
            prepare_reader_facing_table(complex_table),
        ),
        "residue": (
            Path(args.output_residue),
            prepare_reader_facing_table(residue_table),
        ),
        "summary": (Path(args.output_summary), summary_table),
    }
    for label, (path, table) in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
        print(f"Wrote {label} table: {path} ({len(table)} rows)")
    output_png = Path(args.output_figure_png)
    output_pdf = Path(args.output_figure_pdf)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=220, bbox_inches="tight")
    figure.savefig(output_pdf, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output_png}")
    print(f"Wrote {output_pdf}")


if __name__ == "__main__":
    main()
