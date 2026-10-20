"""Relate AF3-WT structural features to TCR-pMHC DDG prediction errors.

The analysis uses the common 438-mutation cohort scored by RDE-PPI,
Prompt-DDG, BA-DDG, and H3-DDG. It compares predictions made on the
experimental WT PDB (approach 1) with predictions made on the exact AF3-WT
seed-1/sample-0 scaffold (approach 4), then asks whether the resulting error
change is associated with:

* AF3-WT versus experimental interface-contact disagreement;
* framework-superposed CDR3 and whole-complex RMSD;
* global and interface-local AF3 confidence metrics;
* mutation-local contact disagreement and confidence; and
* mutation region or peptide position.

Template-level associations use one equally weighted observation per WT
template. Mutation-level associations remove between-template rank offsets and
give each template equal total weight. Confidence intervals resample templates
as blocks.

Usage:
    conda run -n bioemu1p3 python scripts/analyze_af3_wt_error_features.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import zlib
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from helpers.paths import DATA_ROOT, OUTPUT_ROOT, SUPPLEMENTARY_ROOT
from typing import Iterable

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser, NeighborSearch, PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1
from Bio.SVDSuperimposer import SVDSuperimposer
from scipy.stats import rankdata, spearmanr

REPO = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO / "scripts"
ATLAS_EVAL_DIR = REPO / "ddg_predictions" / "atlas_af_eval"
for import_dir in (SCRIPTS_DIR, ATLAS_EVAL_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from helpers.atlas_parsing import parse_atlas_mutation, parse_ddg_exp  # noqa: E402
from helpers.cdr_filter import CDR_RANGES, build_imgt_map  # noqa: E402

log = logging.getLogger(__name__)

TCR_CHAINS = {"D", "E"}
PMHC_CHAINS = {"A", "B", "C"}
ANALYSIS_CHAINS = ("A", "B", "C", "D", "E")
PRIMARY_CONTACT_CUTOFF = 5.0
DEFAULT_SENSITIVITY_CUTOFFS = (4.0, 6.0)
CONTACT_STRATUM_LABELS = (
    "No disagreement",
    "Low (0--0.33]",
    "Medium (0.33--0.67]",
    "High (0.67--1.0]",
)

MODEL_SPECS = {
    "RDE-PPI": {
        "slug": "rde",
        "compare_suffix": "rde",
        "results_dir": DATA_ROOT / "ddg_predictions" / "atlas_af_eval" / "results",
    },
    "Prompt-DDG": {
        "slug": "prompt",
        "compare_suffix": "pr",
        "results_dir": DATA_ROOT / "ddg_predictions" / "atlas_promptddg_eval" / "results",
    },
    "BA-DDG": {
        "slug": "ba",
        "compare_suffix": "ba",
        "results_dir": DATA_ROOT / "ddg_predictions" / "atlas_baddg_eval" / "results",
    },
    "H3-DDG": {
        "slug": "h3",
        "compare_suffix": "h3",
        "results_dir": DATA_ROOT / "ddg_predictions" / "atlas_h3ddg_eval" / "results",
    },
}

PRIMARY_TEMPLATE_FEATURES = (
    "contact_disagreement_5a",
    "cdr3_rmsd_angstrom",
    "ptm",
    "iptm",
)
EXPLORATORY_TEMPLATE_FEATURES = (
    "tcr_pmhc_ca_rmsd_angstrom",
    "ranking_score",
    "fraction_disordered",
    "interface_weighted_iptm",
    "interface_mean_pae_angstrom",
    "interface_mean_contact_probability",
    "cdr3_mean_plddt",
    "peptide_mean_plddt",
)
EXPLORATORY_MUTATION_FEATURES = (
    "mutation_site_plddt_mean",
    "mutation_site_plddt_max",
    "mutation_partner_min_pae_mean_angstrom",
    "mutation_partner_min_pae_max_angstrom",
    "local_contact_gained_count_5a",
    "local_contact_lost_count_5a",
    "num_muts",
)

ResidueKey = tuple[str, int, str]
ContactPair = tuple[ResidueKey, ResidueKey]


@dataclass
class TemplateContext:
    """Structural and confidence data reused by all mutations on one template.

    Attributes:
        template: ATLAS WT template PDB identifier.
        aligned_residues: canonical experimental residue key to
            ``(experimental_residue, af3_residue)``.
        imgt_map: ``(chain, author_resseq)`` to IMGT position for chains D/E.
        contacts: contact cutoff to ``(experimental_contacts, af3_contacts)``.
        token_indices: canonical residue key to AF3 confidence-token index.
        pae: AF3 token-by-token predicted aligned error matrix.
        ca_plddt: canonical residue key to AF3 C-alpha pLDDT.
        feature_row: one-row template feature record for Supplementary Table S15.
    """

    template: str
    aligned_residues: dict[ResidueKey, tuple[object, object]]
    imgt_map: dict[tuple[str, int], int]
    contacts: dict[float, tuple[set[ContactPair], set[ContactPair]]]
    token_indices: dict[ResidueKey, int]
    pae: np.ndarray
    ca_plddt: dict[ResidueKey, float]
    feature_row: dict[str, object]


def _standard_residues(chain) -> list:
    """Return standard ATOM residues from a BioPython chain in file order."""
    return [residue for residue in chain if residue.id[0] == " "]


def _residue_sequence(residues: Iterable) -> str:
    """Convert an ordered residue iterable to a one-letter sequence."""
    return "".join(
        protein_letters_3to1.get(residue.get_resname()) or "X"
        for residue in residues
    )


def _load_structure(path: Path):
    """Load the first model from a PDB or mmCIF structure file."""
    parser = MMCIFParser(QUIET=True) if path.suffix.lower() == ".cif" else PDBParser(
        QUIET=True
    )
    return parser.get_structure(path.stem, str(path))[0]


def align_residues_by_chain(
    reference_model,
    af3_model,
    chain_ids: Iterable[str] = ANALYSIS_CHAINS,
) -> tuple[dict[ResidueKey, tuple[object, object]], dict[str, float]]:
    """Align same-ID chains by sequence and return canonical residue pairs.

    Canonical keys retain the experimental PDB author numbering. Equal sequence
    blocks from ``SequenceMatcher`` are used so terminal truncations and internal
    gaps do not shift downstream residue mappings.

    Args:
        reference_model: first BioPython model of the experimental PDB.
        af3_model: first BioPython model of the AF3-WT mmCIF.
        chain_ids: chain IDs eligible for alignment.

    Returns:
        A tuple containing the canonical residue-pair mapping and per-chain
        alignment coverage, measured against the shorter chain.
    """
    reference_ids = {chain.id for chain in reference_model}
    af3_ids = {chain.id for chain in af3_model}
    aligned: dict[ResidueKey, tuple[object, object]] = {}
    coverage: dict[str, float] = {}

    for chain_id in chain_ids:
        if chain_id not in reference_ids or chain_id not in af3_ids:
            continue
        reference_residues = _standard_residues(reference_model[chain_id])
        af3_residues = _standard_residues(af3_model[chain_id])
        reference_sequence = _residue_sequence(reference_residues)
        af3_sequence = _residue_sequence(af3_residues)
        matched = 0
        matcher = SequenceMatcher(
            None,
            reference_sequence,
            af3_sequence,
            autojunk=False,
        )
        for tag, ref_start, ref_end, af_start, af_end in matcher.get_opcodes():
            if tag != "equal":
                continue
            for reference_index, af3_index in zip(
                range(ref_start, ref_end),
                range(af_start, af_end),
            ):
                reference_residue = reference_residues[reference_index]
                af3_residue = af3_residues[af3_index]
                key = (
                    chain_id,
                    int(reference_residue.id[1]),
                    str(reference_residue.id[2]),
                )
                aligned[key] = (reference_residue, af3_residue)
                matched += 1

        denominator = min(len(reference_residues), len(af3_residues))
        coverage[chain_id] = matched / denominator if denominator else 0.0
        minimum_coverage = 0.6 if chain_id == "C" else 0.8
        if coverage[chain_id] < minimum_coverage:
            raise ValueError(
                f"chain {chain_id} alignment coverage {coverage[chain_id]:.3f} "
                f"is below {minimum_coverage:.1f}"
            )

    if not TCR_CHAINS <= {key[0] for key in aligned}:
        raise ValueError("both TCR chains D/E must align between PDB and AF3-WT")
    if "C" not in {key[0] for key in aligned}:
        raise ValueError("peptide chain C must align between PDB and AF3-WT")
    return aligned, coverage


def _is_heavy_atom(atom) -> bool:
    """Return whether a BioPython atom is non-hydrogen."""
    element = str(getattr(atom, "element", "") or "").strip().upper()
    return element != "H" and not str(atom.id).upper().startswith("H")


def build_interface_contacts(
    aligned_residues: dict[ResidueKey, tuple[object, object]],
    structure_index: int,
    cutoff: float,
) -> set[ContactPair]:
    """Build mapped TCR-pMHC residue contacts for one structure.

    A residue pair is a contact when any heavy-atom pair lies within ``cutoff``.
    Contact keys always use experimental author numbering, making the PDB and
    AF3-WT contact sets directly comparable.

    Args:
        aligned_residues: canonical residue mapping.
        structure_index: 0 for experimental residues, 1 for AF3 residues.
        cutoff: heavy-atom distance threshold in angstroms.

    Returns:
        Set of ``(tcr_residue_key, pmhc_residue_key)`` contacts.
    """
    atoms = []
    atom_keys: dict[int, ResidueKey] = {}
    for key, residue_pair in aligned_residues.items():
        if key[0] not in TCR_CHAINS | PMHC_CHAINS:
            continue
        residue = residue_pair[structure_index]
        for atom in residue.get_atoms():
            if not _is_heavy_atom(atom):
                continue
            atoms.append(atom)
            atom_keys[id(atom)] = key

    contacts: set[ContactPair] = set()
    search = NeighborSearch(atoms)
    for atom_a, atom_b in search.search_all(float(cutoff), level="A"):
        key_a = atom_keys[id(atom_a)]
        key_b = atom_keys[id(atom_b)]
        if key_a[0] in TCR_CHAINS and key_b[0] in PMHC_CHAINS:
            contacts.add((key_a, key_b))
        elif key_b[0] in TCR_CHAINS and key_a[0] in PMHC_CHAINS:
            contacts.add((key_b, key_a))
    return contacts


def contact_set_metrics(
    reference_contacts: set[ContactPair],
    af3_contacts: set[ContactPair],
) -> dict[str, float | int]:
    """Summarize agreement between reference and AF3 contact sets.

    If both sets are empty, disagreement is defined as zero. If exactly one set
    is empty, disagreement is one.
    """
    intersection = reference_contacts & af3_contacts
    union = reference_contacts | af3_contacts
    disagreement = 0.0 if not union else 1.0 - len(intersection) / len(union)
    denominator = len(reference_contacts) + len(af3_contacts)
    f1 = 1.0 if denominator == 0 else 2.0 * len(intersection) / denominator
    return {
        "reference_count": len(reference_contacts),
        "af3_count": len(af3_contacts),
        "shared_count": len(intersection),
        "gained_count": len(af3_contacts - reference_contacts),
        "lost_count": len(reference_contacts - af3_contacts),
        "disagreement": float(disagreement),
        "f1": float(f1),
    }


def assign_contact_disagreement_strata(values: pd.Series) -> pd.Series:
    """Assign bounded contact disagreement to four reproducible strata.

    Values are rounded before binning so mathematically exact fractions such as
    one third are not moved across a boundary by binary floating-point noise.
    """
    numeric = values.astype(float)
    if ((numeric < 0.0) | (numeric > 1.0)).any():
        raise ValueError("contact disagreement must lie between zero and one")
    return pd.cut(
        numeric.round(12),
        bins=[-np.inf, 0.0, 1.0 / 3.0, 2.0 / 3.0, np.inf],
        labels=CONTACT_STRATUM_LABELS,
        include_lowest=True,
        right=True,
    ).astype(str)


def _fit_framework_transform(
    aligned_residues: dict[ResidueKey, tuple[object, object]],
    imgt_map: dict[tuple[str, int], int],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit AF3-WT to the PDB on TCR variable-domain framework C-alpha atoms."""
    framework_keys = []
    for key, (reference_residue, af3_residue) in aligned_residues.items():
        imgt_position = imgt_map.get((key[0], key[1]))
        if imgt_position is None or "CA" not in reference_residue or "CA" not in af3_residue:
            continue
        if any(start <= imgt_position <= end for start, end in CDR_RANGES.values()):
            continue
        framework_keys.append(key)

    if len(framework_keys) < 10:
        raise ValueError(
            f"at least 10 aligned framework C-alpha atoms are required; "
            f"found {len(framework_keys)}"
        )
    reference_coordinates = np.asarray(
        [aligned_residues[key][0]["CA"].coord for key in framework_keys],
        dtype=float,
    )
    af3_coordinates = np.asarray(
        [aligned_residues[key][1]["CA"].coord for key in framework_keys],
        dtype=float,
    )
    superimposer = SVDSuperimposer()
    superimposer.set(reference_coordinates, af3_coordinates)
    superimposer.run()
    rotation, translation = superimposer.get_rotran()
    return rotation, translation, float(superimposer.get_rms())


def _transformed_rmsd(
    aligned_residues: dict[ResidueKey, tuple[object, object]],
    keys: Iterable[ResidueKey],
    rotation: np.ndarray,
    translation: np.ndarray,
) -> float:
    """Calculate C-alpha RMSD after applying a pre-fitted AF3 transform."""
    usable_keys = [
        key
        for key in keys
        if "CA" in aligned_residues[key][0] and "CA" in aligned_residues[key][1]
    ]
    if not usable_keys:
        return float("nan")
    reference_coordinates = np.asarray(
        [aligned_residues[key][0]["CA"].coord for key in usable_keys],
        dtype=float,
    )
    af3_coordinates = np.asarray(
        [aligned_residues[key][1]["CA"].coord for key in usable_keys],
        dtype=float,
    )
    transformed = np.dot(af3_coordinates, rotation) + translation
    return float(
        np.sqrt(np.mean(np.sum((reference_coordinates - transformed) ** 2, axis=1)))
    )


def calculate_template_rmsds(
    aligned_residues: dict[ResidueKey, tuple[object, object]],
    imgt_map: dict[tuple[str, int], int],
) -> dict[str, float]:
    """Calculate framework-superposed CDR3 and whole-complex RMSDs."""
    rotation, translation, framework_rmsd = _fit_framework_transform(
        aligned_residues,
        imgt_map,
    )
    cdr3_keys = [
        key
        for key in aligned_residues
        if key[0] in TCR_CHAINS
        and (
            CDR_RANGES["CDR3"][0]
            <= imgt_map.get((key[0], key[1]), -1)
            <= CDR_RANGES["CDR3"][1]
        )
    ]
    alpha_keys = [key for key in cdr3_keys if key[0] == "D"]
    beta_keys = [key for key in cdr3_keys if key[0] == "E"]
    all_keys = [
        key
        for key in aligned_residues
        if key[0] in TCR_CHAINS | PMHC_CHAINS
    ]
    return {
        "framework_rmsd_angstrom": framework_rmsd,
        "cdr3_rmsd_angstrom": _transformed_rmsd(
            aligned_residues,
            cdr3_keys,
            rotation,
            translation,
        ),
        "cdr3_alpha_rmsd_angstrom": _transformed_rmsd(
            aligned_residues,
            alpha_keys,
            rotation,
            translation,
        ),
        "cdr3_beta_rmsd_angstrom": _transformed_rmsd(
            aligned_residues,
            beta_keys,
            rotation,
            translation,
        ),
        "tcr_pmhc_ca_rmsd_angstrom": _transformed_rmsd(
            aligned_residues,
            all_keys,
            rotation,
            translation,
        ),
    }


def _confidence_paths(af3_wt_dir: Path, template: str) -> tuple[Path, Path, Path]:
    """Return model, detailed-confidence, and summary-confidence paths."""
    sample_dir = af3_wt_dir / f"wt_{template}" / "seed-1_sample-0"
    prefix = f"wt_{template}_seed-1_sample-0"
    return (
        sample_dir / f"{prefix}_model.cif",
        sample_dir / f"{prefix}_confidences.json",
        sample_dir / f"{prefix}_summary_confidences.json",
    )


def _symmetrized_matrix_value(matrix: np.ndarray, index_a: int, index_b: int) -> float:
    """Return the mean of both directed entries in a square confidence matrix."""
    return float((matrix[index_a, index_b] + matrix[index_b, index_a]) / 2.0)


def extract_confidence_features(
    aligned_residues: dict[ResidueKey, tuple[object, object]],
    imgt_map: dict[tuple[str, int], int],
    reference_contacts: set[ContactPair],
    detailed_path: Path,
    summary_path: Path,
) -> tuple[dict[str, object], dict[ResidueKey, int], np.ndarray, dict[ResidueKey, float]]:
    """Extract global, interface, regional, and token-level AF3 confidence.

    Returns:
        A tuple containing the template feature dictionary, canonical token
        indices, the PAE matrix, and canonical C-alpha pLDDT values.
    """
    detailed = json.loads(detailed_path.read_text())
    summary = json.loads(summary_path.read_text())
    token_chain_ids = [str(value) for value in detailed["token_chain_ids"]]
    token_residue_ids = [int(value) for value in detailed["token_res_ids"]]
    token_lookup = {
        (chain_id, residue_id): index
        for index, (chain_id, residue_id) in enumerate(
            zip(token_chain_ids, token_residue_ids)
        )
    }
    token_indices: dict[ResidueKey, int] = {}
    ca_plddt: dict[ResidueKey, float] = {}
    for key, (_reference_residue, af3_residue) in aligned_residues.items():
        token_index = token_lookup.get((key[0], int(af3_residue.id[1])))
        if token_index is not None:
            token_indices[key] = token_index
        if "CA" in af3_residue:
            ca_plddt[key] = float(af3_residue["CA"].bfactor)

    pae = np.asarray(detailed["pae"], dtype=float)
    contact_probabilities = np.asarray(detailed["contact_probs"], dtype=float)
    if pae.shape != contact_probabilities.shape or pae.shape[0] != len(token_chain_ids):
        raise ValueError(
            f"confidence matrix shape mismatch: PAE {pae.shape}, "
            f"contact probabilities {contact_probabilities.shape}, "
            f"tokens {len(token_chain_ids)}"
        )

    interface_pae = []
    interface_contact_probability = []
    chain_pair_counts: Counter[tuple[str, str]] = Counter()
    for tcr_key, pmhc_key in reference_contacts:
        if tcr_key not in token_indices or pmhc_key not in token_indices:
            continue
        tcr_index = token_indices[tcr_key]
        pmhc_index = token_indices[pmhc_key]
        interface_pae.append(_symmetrized_matrix_value(pae, tcr_index, pmhc_index))
        interface_contact_probability.append(
            _symmetrized_matrix_value(
                contact_probabilities,
                tcr_index,
                pmhc_index,
            )
        )
        chain_pair_counts[(tcr_key[0], pmhc_key[0])] += 1
    if not interface_pae:
        raise ValueError("no mapped experimental interface contacts have confidence data")

    chain_order = list(dict.fromkeys(token_chain_ids))
    chain_index = {chain_id: index for index, chain_id in enumerate(chain_order)}
    chain_pair_iptm = np.asarray(summary["chain_pair_iptm"], dtype=float)
    if chain_pair_iptm.shape != (len(chain_order), len(chain_order)):
        raise ValueError(
            f"chain_pair_iptm shape {chain_pair_iptm.shape} does not match "
            f"token chain order {chain_order}"
        )
    weighted_iptm_values = []
    weighted_iptm_weights = []
    for (tcr_chain, pmhc_chain), count in chain_pair_counts.items():
        i = chain_index[tcr_chain]
        j = chain_index[pmhc_chain]
        weighted_iptm_values.append(
            (chain_pair_iptm[i, j] + chain_pair_iptm[j, i]) / 2.0
        )
        weighted_iptm_weights.append(count)
    interface_weighted_iptm = float(
        np.average(weighted_iptm_values, weights=weighted_iptm_weights)
    )

    cdr3_plddt = [
        value
        for key, value in ca_plddt.items()
        if key[0] in TCR_CHAINS
        and (
            CDR_RANGES["CDR3"][0]
            <= imgt_map.get((key[0], key[1]), -1)
            <= CDR_RANGES["CDR3"][1]
        )
    ]
    peptide_plddt = [value for key, value in ca_plddt.items() if key[0] == "C"]
    if not cdr3_plddt or not peptide_plddt:
        raise ValueError("CDR3 and peptide pLDDT must both be available")

    features: dict[str, object] = {
        "ptm": float(summary["ptm"]),
        "iptm": float(summary["iptm"]),
        "ranking_score": float(summary["ranking_score"]),
        "fraction_disordered": float(summary["fraction_disordered"]),
        "has_clash": float(summary["has_clash"]),
        "interface_weighted_iptm": interface_weighted_iptm,
        "interface_mean_pae_angstrom": float(np.mean(interface_pae)),
        "interface_mean_contact_probability": float(
            np.mean(interface_contact_probability)
        ),
        "cdr3_mean_plddt": float(np.mean(cdr3_plddt)),
        "peptide_mean_plddt": float(np.mean(peptide_plddt)),
        "confidence_chain_order": ";".join(chain_order),
        "interface_confidence_contact_count": len(interface_pae),
    }
    return features, token_indices, pae, ca_plddt


def extract_template_context(
    template: str,
    pdb_dir: Path,
    af3_wt_dir: Path,
    contact_cutoffs: Iterable[float],
) -> TemplateContext:
    """Extract all reusable structural/confidence features for one WT template."""
    pdb_path = pdb_dir / f"{template}.pdb"
    model_path, detailed_path, summary_path = _confidence_paths(af3_wt_dir, template)
    for required_path in (pdb_path, model_path, detailed_path, summary_path):
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    reference_model = _load_structure(pdb_path)
    af3_model = _load_structure(model_path)
    aligned_residues, coverage = align_residues_by_chain(reference_model, af3_model)
    imgt_map = build_imgt_map(pdb_path, template_id=template)
    rmsd_features = calculate_template_rmsds(aligned_residues, imgt_map)

    contacts: dict[float, tuple[set[ContactPair], set[ContactPair]]] = {}
    feature_row: dict[str, object] = {
        "template_PDB": template,
        "pdb_path": str(pdb_path.relative_to(DATA_ROOT)),
        "aligned_residue_count": len(aligned_residues),
        "minimum_chain_alignment_coverage": min(coverage.values()),
    }
    feature_row.update(rmsd_features)
    for cutoff in sorted(set(float(value) for value in contact_cutoffs)):
        reference_contacts = build_interface_contacts(
            aligned_residues,
            structure_index=0,
            cutoff=cutoff,
        )
        af3_contacts = build_interface_contacts(
            aligned_residues,
            structure_index=1,
            cutoff=cutoff,
        )
        contacts[cutoff] = (reference_contacts, af3_contacts)
        metrics = contact_set_metrics(reference_contacts, af3_contacts)
        suffix = f"{cutoff:g}a"
        feature_row[f"reference_interface_contact_count_{suffix}"] = metrics[
            "reference_count"
        ]
        feature_row[f"af3_interface_contact_count_{suffix}"] = metrics["af3_count"]
        feature_row[f"contact_shared_count_{suffix}"] = metrics["shared_count"]
        feature_row[f"contact_gained_count_{suffix}"] = metrics["gained_count"]
        feature_row[f"contact_lost_count_{suffix}"] = metrics["lost_count"]
        feature_row[f"contact_disagreement_{suffix}"] = metrics["disagreement"]
        feature_row[f"contact_f1_{suffix}"] = metrics["f1"]

    reference_contacts = contacts[PRIMARY_CONTACT_CUTOFF][0]
    (
        confidence_features,
        token_indices,
        pae,
        ca_plddt,
    ) = extract_confidence_features(
        aligned_residues,
        imgt_map,
        reference_contacts,
        detailed_path,
        summary_path,
    )
    feature_row.update(confidence_features)
    return TemplateContext(
        template=template,
        aligned_residues=aligned_residues,
        imgt_map=imgt_map,
        contacts=contacts,
        token_indices=token_indices,
        pae=pae,
        ca_plddt=ca_plddt,
        feature_row=feature_row,
    )


def _canonical_mutation_keys(mutation_string: str) -> tuple[list[dict], list[ResidueKey]]:
    """Parse ATLAS mutation tokens and return author-numbered canonical keys."""
    tokens = [token.strip() for token in mutation_string.split(";") if token.strip()]
    mutations = [parse_atlas_mutation(token) for token in tokens]
    keys = [
        (mutation["chain"], int(mutation["resseq"]), str(mutation["icode"]))
        for mutation in mutations
    ]
    return mutations, keys


def classify_mutation_region(
    mutation: dict,
    imgt_map: dict[tuple[str, int], int],
) -> str:
    """Classify one mutation as MHC, peptide, or an IMGT TCR region."""
    chain = mutation["chain"]
    if chain == "C":
        return "Peptide"
    if chain in {"A", "B"}:
        return "MHC"
    if chain not in TCR_CHAINS:
        raise ValueError(f"unexpected mutation chain {chain!r}")
    imgt_position = imgt_map.get((chain, int(mutation["resseq"])))
    if imgt_position is None:
        raise KeyError(
            f"TCR mutation {chain}:{mutation['resseq']} is absent from the IMGT map"
        )
    if CDR_RANGES["CDR3"][0] <= imgt_position <= CDR_RANGES["CDR3"][1]:
        return "TCR CDR3"
    if any(
        start <= imgt_position <= end
        for name, (start, end) in CDR_RANGES.items()
        if name != "CDR3"
    ):
        return "Other TCR CDR"
    return "TCR framework"


def _peptide_position(
    mutation: dict,
    context: TemplateContext,
) -> tuple[float, float, float]:
    """Return one-based peptide position, peptide length, and normalized position."""
    peptide_keys = sorted(
        (key for key in context.aligned_residues if key[0] == "C"),
        key=lambda key: (
            context.aligned_residues[key][0].id[1],
            context.aligned_residues[key][0].id[2],
        ),
    )
    mutation_key = (
        mutation["chain"],
        int(mutation["resseq"]),
        str(mutation["icode"]),
    )
    position_lookup = {key: index + 1 for index, key in enumerate(peptide_keys)}
    if mutation_key not in position_lookup:
        raise KeyError(f"peptide mutation {mutation_key} is absent from aligned residues")
    position = float(position_lookup[mutation_key])
    length = float(len(peptide_keys))
    normalized = 0.0 if length <= 1 else (position - 1.0) / (length - 1.0)
    return position, length, normalized


def _local_contacts(
    contacts: set[ContactPair],
    mutation_keys: set[ResidueKey],
) -> set[ContactPair]:
    """Return interface contacts involving at least one mutated residue."""
    return {
        contact
        for contact in contacts
        if contact[0] in mutation_keys or contact[1] in mutation_keys
    }


def _mutation_partner_min_pae(
    mutation_key: ResidueKey,
    context: TemplateContext,
) -> float:
    """Return minimum symmetrized PAE from one mutation to the opposite partner."""
    mutation_index = context.token_indices[mutation_key]
    opposite_chains = PMHC_CHAINS if mutation_key[0] in TCR_CHAINS else TCR_CHAINS
    partner_indices = [
        token_index
        for key, token_index in context.token_indices.items()
        if key[0] in opposite_chains
    ]
    if not partner_indices:
        raise ValueError(f"no opposite-partner tokens for mutation {mutation_key}")
    values = [
        _symmetrized_matrix_value(context.pae, mutation_index, partner_index)
        for partner_index in partner_indices
    ]
    return float(min(values))


def extract_mutation_features(
    row: pd.Series,
    context: TemplateContext,
    contact_cutoffs: Iterable[float],
) -> dict[str, object]:
    """Extract mutation-local structure, confidence, region, and peptide features."""
    mutations, mutation_keys = _canonical_mutation_keys(str(row["mutations"]))
    missing_keys = [key for key in mutation_keys if key not in context.aligned_residues]
    if missing_keys:
        raise KeyError(
            f"{context.template} row {row['row_idx']}: mutations absent from "
            f"PDB/AF3 alignment: {missing_keys}"
        )
    missing_tokens = [key for key in mutation_keys if key not in context.token_indices]
    if missing_tokens:
        raise KeyError(
            f"{context.template} row {row['row_idx']}: mutations absent from "
            f"AF3 confidence tokens: {missing_tokens}"
        )

    mutation_key_set = set(mutation_keys)
    features: dict[str, object] = {}
    for cutoff in sorted(set(float(value) for value in contact_cutoffs)):
        reference_contacts, af3_contacts = context.contacts[cutoff]
        local_reference = _local_contacts(reference_contacts, mutation_key_set)
        local_af3 = _local_contacts(af3_contacts, mutation_key_set)
        metrics = contact_set_metrics(local_reference, local_af3)
        suffix = f"{cutoff:g}a"
        features[f"local_reference_contact_count_{suffix}"] = metrics[
            "reference_count"
        ]
        features[f"local_af3_contact_count_{suffix}"] = metrics["af3_count"]
        features[f"local_contact_gained_count_{suffix}"] = metrics["gained_count"]
        features[f"local_contact_lost_count_{suffix}"] = metrics["lost_count"]
        features[f"local_contact_disagreement_{suffix}"] = metrics["disagreement"]
        features[f"local_contact_f1_{suffix}"] = metrics["f1"]

    plddt_values = [context.ca_plddt[key] for key in mutation_keys]
    partner_pae_values = [
        _mutation_partner_min_pae(key, context) for key in mutation_keys
    ]
    regions = [classify_mutation_region(mutation, context.imgt_map) for mutation in mutations]
    features.update(
        {
            "mutation_components": "+".join(
                sorted({token.split(":", maxsplit=1)[0] for token in row["mutations"].split(";")})
            ),
            "mutation_chains": ";".join(mutation["chain"] for mutation in mutations),
            "mutation_regions": ";".join(dict.fromkeys(regions)),
            "single_point_region": regions[0] if len(mutations) == 1 else "",
            "mutation_site_plddt_mean": float(np.mean(plddt_values)),
            "mutation_site_plddt_max": float(np.max(plddt_values)),
            "mutation_partner_min_pae_mean_angstrom": float(
                np.mean(partner_pae_values)
            ),
            "mutation_partner_min_pae_max_angstrom": float(
                np.max(partner_pae_values)
            ),
            "peptide_position": float("nan"),
            "peptide_length": float("nan"),
            "peptide_position_normalized": float("nan"),
        }
    )
    if len(mutations) == 1 and mutations[0]["chain"] == "C":
        position, length, normalized = _peptide_position(mutations[0], context)
        features["peptide_position"] = position
        features["peptide_length"] = length
        features["peptide_position_normalized"] = normalized
    return features


def _prediction_columns(model_name: str) -> tuple[str, str]:
    """Return approach-1 and approach-4 columns in compare_4models.csv."""
    suffix = str(MODEL_SPECS[model_name]["compare_suffix"])
    return f"ddg_pred_a1_{suffix}", f"ddg_pred_a4_{suffix}"


def _reconcile_raw_prediction_sources(predictions: pd.DataFrame) -> None:
    """Verify the merged comparison columns against each model's raw result CSVs."""
    for model_name, specification in MODEL_SPECS.items():
        pdb_column, af3_column = _prediction_columns(model_name)
        results_dir = Path(specification["results_dir"])
        for approach, comparison_column in ((1, pdb_column), (4, af3_column)):
            raw = pd.read_csv(results_dir / f"approach{approach}.csv")
            if raw["row_idx"].duplicated().any():
                raise ValueError(
                    f"{model_name} approach {approach} contains duplicate row_idx values"
                )
            raw_values = raw.set_index("row_idx")["ddg_pred"]
            aligned = predictions["row_idx"].map(raw_values)
            if aligned.isna().any():
                missing = predictions.loc[aligned.isna(), "row_idx"].tolist()
                raise ValueError(
                    f"{model_name} approach {approach} is missing common rows "
                    f"{missing[:10]}"
                )
            if not np.allclose(
                predictions[comparison_column].to_numpy(dtype=float),
                aligned.to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-10,
            ):
                raise ValueError(
                    f"{comparison_column} disagrees with "
                    f"{results_dir / f'approach{approach}.csv'}"
                )


def load_common_cohort(
    predictions_csv: Path,
    index_csv: Path,
    *,
    validate_raw_sources: bool = True,
) -> pd.DataFrame:
    """Load and validate the canonical common prediction/mutation cohort."""
    predictions = pd.read_csv(predictions_csv)
    if len(predictions) != 438 or predictions["row_idx"].duplicated().any():
        raise ValueError(
            f"expected 438 unique prediction rows, found {len(predictions)}"
        )
    required_prediction_columns = ["row_idx", "mutstr", "ddg_exp"]
    for model_name in MODEL_SPECS:
        required_prediction_columns.extend(_prediction_columns(model_name))
    missing_columns = sorted(set(required_prediction_columns) - set(predictions.columns))
    if missing_columns:
        raise ValueError(f"prediction table is missing columns {missing_columns}")
    if predictions[required_prediction_columns].isna().any().any():
        raise ValueError("prediction table contains missing required values")
    if validate_raw_sources:
        _reconcile_raw_prediction_sources(predictions)

    index = pd.read_csv(index_csv)
    if index["row_idx"].duplicated().any():
        raise ValueError("ATLAS_af/index.csv contains duplicate row_idx values")
    cohort = predictions.merge(
        index[
            [
                "row_idx",
                "atlas_Index",
                "template_PDB",
                "mutations",
                "DDG",
                "status",
            ]
        ],
        on="row_idx",
        how="left",
        validate="one_to_one",
    )
    if cohort[["atlas_Index", "template_PDB", "mutations", "DDG"]].isna().any().any():
        raise ValueError("prediction-to-index join lost mutation metadata")
    parsed_index_ddg = cohort["DDG"].map(parse_ddg_exp).to_numpy(dtype=float)
    if not np.allclose(
        cohort["ddg_exp"].to_numpy(dtype=float),
        parsed_index_ddg,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("experimental DDG values disagree between predictions and index")
    if cohort["template_PDB"].nunique() != 37:
        raise ValueError(
            f"expected 37 WT templates, found {cohort['template_PDB'].nunique()}"
        )
    if (cohort["template_PDB"] == "1FYT").any():
        raise ValueError("1FYT must remain excluded because of undefined occupancies")
    cohort["num_muts"] = cohort["mutations"].str.split(";").map(len)
    if int((cohort["num_muts"] == 1).sum()) != 304:
        raise ValueError("expected 304 single-point mutations")
    return cohort.sort_values("row_idx").reset_index(drop=True)


def build_mutation_table(
    cohort: pd.DataFrame,
    contexts: dict[str, TemplateContext],
    contact_cutoffs: Iterable[float],
) -> pd.DataFrame:
    """Combine mutation-local features with per-model predictions and errors."""
    rows: list[dict[str, object]] = []
    for _, source_row in cohort.iterrows():
        template = str(source_row["template_PDB"])
        context = contexts[template]
        output_row: dict[str, object] = {
            "row_idx": int(source_row["row_idx"]),
            "atlas_Index": int(source_row["atlas_Index"]),
            "template_PDB": template,
            "mutations": source_row["mutations"],
            "mutstr": source_row["mutstr"],
            "ddg_exp": float(source_row["ddg_exp"]),
            "num_muts": int(source_row["num_muts"]),
        }
        output_row.update(
            extract_mutation_features(source_row, context, contact_cutoffs)
        )
        for model_name, specification in MODEL_SPECS.items():
            slug = str(specification["slug"])
            pdb_column, af3_column = _prediction_columns(model_name)
            pdb_prediction = float(source_row[pdb_column])
            af3_prediction = float(source_row[af3_column])
            experimental = float(source_row["ddg_exp"])
            pdb_error = abs(pdb_prediction - experimental)
            af3_error = abs(af3_prediction - experimental)
            output_row[f"{slug}_pdb_prediction"] = pdb_prediction
            output_row[f"{slug}_af3_wt_prediction"] = af3_prediction
            output_row[f"{slug}_pdb_absolute_error"] = pdb_error
            output_row[f"{slug}_af3_wt_absolute_error"] = af3_error
            output_row[f"{slug}_absolute_error_change"] = af3_error - pdb_error
        output_row["ba_minus_prompt_absolute_error_change"] = (
            float(output_row["ba_absolute_error_change"])
            - float(output_row["prompt_absolute_error_change"])
        )
        output_row["h3_minus_prompt_absolute_error_change"] = (
            float(output_row["h3_absolute_error_change"])
            - float(output_row["prompt_absolute_error_change"])
        )
        rows.append(output_row)

    mutation_table = pd.DataFrame(rows).sort_values("row_idx").reset_index(drop=True)
    peptide_mask = (
        (mutation_table["num_muts"] == 1)
        & (mutation_table["single_point_region"] == "Peptide")
    )
    if int(peptide_mask.sum()) != 67:
        raise ValueError(
            f"expected 67 single-point peptide mutations, found {peptide_mask.sum()}"
        )
    return mutation_table


def build_template_table(
    contexts: dict[str, TemplateContext],
    mutation_table: pd.DataFrame,
) -> pd.DataFrame:
    """Combine one-row structural features with equal-template outcome summaries."""
    rows = []
    for template in sorted(contexts):
        template_mutations = mutation_table[
            mutation_table["template_PDB"] == template
        ]
        row = dict(contexts[template].feature_row)
        row["n_mutations"] = len(template_mutations)
        for model_name, specification in MODEL_SPECS.items():
            slug = str(specification["slug"])
            row[f"{slug}_pdb_mae"] = float(
                template_mutations[f"{slug}_pdb_absolute_error"].mean()
            )
            row[f"{slug}_af3_wt_mae"] = float(
                template_mutations[f"{slug}_af3_wt_absolute_error"].mean()
            )
            row[f"{slug}_mean_absolute_error_change"] = float(
                template_mutations[f"{slug}_absolute_error_change"].mean()
            )
        row["ba_minus_prompt_mean_absolute_error_change"] = (
            float(row["ba_mean_absolute_error_change"])
            - float(row["prompt_mean_absolute_error_change"])
        )
        row["h3_minus_prompt_mean_absolute_error_change"] = (
            float(row["h3_mean_absolute_error_change"])
            - float(row["prompt_mean_absolute_error_change"])
        )
        rows.append(row)
    template_table = pd.DataFrame(rows).sort_values("template_PDB").reset_index(
        drop=True
    )
    if len(template_table) != 37:
        raise ValueError(f"expected 37 template rows, found {len(template_table)}")
    return template_table


def _rng_for_key(seed: int, key: str) -> np.random.Generator:
    """Create a deterministic independent random generator for one analysis cell."""
    stable_key = key.replace("absolute_error_change", "scaffold_penalty")
    key_crc = zlib.crc32(stable_key.encode("utf-8"))
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(key_crc)]))


def _finite_pair(frame: pd.DataFrame, x_column: str, y_column: str) -> pd.DataFrame:
    """Return rows where both named numeric columns are finite."""
    values = frame[[x_column, y_column]].astype(float)
    return frame[np.isfinite(values[x_column]) & np.isfinite(values[y_column])]


def template_spearman_association(
    frame: pd.DataFrame,
    feature: str,
    outcome: str,
    bootstrap_replicates: int,
    permutation_replicates: int,
    seed: int,
    analysis_key: str,
) -> dict[str, float | int]:
    """Estimate a template-level Spearman association with resampling inference."""
    complete = _finite_pair(frame, feature, outcome)
    if len(complete) < 3:
        raise ValueError(
            f"{analysis_key}: at least three complete templates are required"
        )
    x = complete[feature].to_numpy(dtype=float)
    y = complete[outcome].to_numpy(dtype=float)
    estimate = float(spearmanr(x, y).statistic)
    if not np.isfinite(estimate):
        raise ValueError(f"{analysis_key}: Spearman correlation is undefined")

    rng = _rng_for_key(seed, analysis_key)
    bootstrap_values = []
    for _ in range(bootstrap_replicates):
        indices = rng.integers(0, len(complete), size=len(complete))
        value = float(spearmanr(x[indices], y[indices]).statistic)
        if np.isfinite(value):
            bootstrap_values.append(value)
    if len(bootstrap_values) < max(10, bootstrap_replicates // 2):
        raise ValueError(f"{analysis_key}: too few finite bootstrap correlations")
    ci_low, ci_high = np.quantile(bootstrap_values, [0.025, 0.975])

    p_value = float("nan")
    if permutation_replicates > 0:
        extreme = 0
        for _ in range(permutation_replicates):
            permuted = rng.permutation(y)
            value = float(spearmanr(x, permuted).statistic)
            if abs(value) >= abs(estimate):
                extreme += 1
        p_value = (extreme + 1.0) / (permutation_replicates + 1.0)
    return {
        "estimate": estimate,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_value": p_value,
        "n_templates": len(complete),
        "n_mutations": int(complete["n_mutations"].sum())
        if "n_mutations" in complete
        else len(complete),
    }


def _rank_group(
    group: pd.DataFrame,
    feature: str,
    outcome: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return centered within-template feature/outcome ranks for one group."""
    complete = _finite_pair(group, feature, outcome)
    if (
        len(complete) < 2
        or complete[feature].nunique() < 2
        or complete[outcome].nunique() < 2
    ):
        return None
    x_rank = rankdata(complete[feature].to_numpy(dtype=float), method="average")
    y_rank = rankdata(complete[outcome].to_numpy(dtype=float), method="average")
    return x_rank - x_rank.mean(), y_rank - y_rank.mean()


def _within_template_rank_statistic(
    groups: Iterable[pd.DataFrame],
    feature: str,
    outcome: str,
) -> tuple[float, int, int]:
    """Calculate an equal-template-weighted within-template rank correlation."""
    covariance = 0.0
    variance_x = 0.0
    variance_y = 0.0
    n_templates = 0
    n_mutations = 0
    for group in groups:
        ranked = _rank_group(group, feature, outcome)
        if ranked is None:
            continue
        x_values, y_values = ranked
        n = len(x_values)
        covariance += float(np.mean(x_values * y_values))
        variance_x += float(np.mean(x_values**2))
        variance_y += float(np.mean(y_values**2))
        n_templates += 1
        n_mutations += n
    if n_templates < 2:
        return float("nan"), n_templates, n_mutations
    if variance_x <= 0 or variance_y <= 0:
        return float("nan"), n_templates, n_mutations
    return float(covariance / np.sqrt(variance_x * variance_y)), n_templates, n_mutations


def mutation_rank_association(
    frame: pd.DataFrame,
    feature: str,
    outcome: str,
    bootstrap_replicates: int,
    permutation_replicates: int,
    seed: int,
    analysis_key: str,
) -> dict[str, float | int]:
    """Estimate a within-template rank association with template-block inference."""
    ranked_groups = []
    n_mutations = 0
    for _, group in frame.groupby("template_PDB", sort=True):
        ranked = _rank_group(group, feature, outcome)
        if ranked is None:
            continue
        x_values, y_values = ranked
        ranked_groups.append((x_values, y_values))
        n_mutations += len(x_values)
    n_templates = len(ranked_groups)
    if n_templates < 2:
        raise ValueError(f"{analysis_key}: fewer than two informative templates")

    contributions = np.asarray(
        [
            (
                np.mean(x_values * y_values),
                np.mean(x_values**2),
                np.mean(y_values**2),
            )
            for x_values, y_values in ranked_groups
        ],
        dtype=float,
    )
    covariance, variance_x, variance_y = contributions.sum(axis=0)
    estimate = float(covariance / np.sqrt(variance_x * variance_y))
    if not np.isfinite(estimate):
        raise ValueError(f"{analysis_key}: within-template association is undefined")

    rng = _rng_for_key(seed, analysis_key)
    sampled_indices = rng.integers(
        0,
        n_templates,
        size=(bootstrap_replicates, n_templates),
    )
    sampled_contributions = contributions[sampled_indices].sum(axis=1)
    bootstrap_values = sampled_contributions[:, 0] / np.sqrt(
        sampled_contributions[:, 1] * sampled_contributions[:, 2]
    )
    bootstrap_values = bootstrap_values[np.isfinite(bootstrap_values)]
    if len(bootstrap_values) < max(10, bootstrap_replicates // 2):
        raise ValueError(f"{analysis_key}: too few finite bootstrap associations")
    ci_low, ci_high = np.quantile(bootstrap_values, [0.025, 0.975])

    p_value = float("nan")
    if permutation_replicates > 0:
        signs = rng.choice(
            np.asarray([-1.0, 1.0]),
            size=(permutation_replicates, n_templates),
            replace=True,
        )
        permuted_covariance = signs @ contributions[:, 0]
        permutation_values = permuted_covariance / np.sqrt(
            variance_x * variance_y
        )
        extreme = int(np.sum(np.abs(permutation_values) >= abs(estimate)))
        p_value = (extreme + 1.0) / (permutation_replicates + 1.0)
    return {
        "estimate": estimate,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_value": p_value,
        "n_templates": n_templates,
        "n_mutations": n_mutations,
    }


def equal_template_category_estimates(
    frame: pd.DataFrame,
    category: str,
    outcome: str,
    category_order: Iterable[str],
    bootstrap_replicates: int,
    seed: int,
    analysis_key: str,
) -> list[dict[str, object]]:
    """Estimate category means after averaging within template and category."""
    complete = frame[
        frame[category].notna()
        & (frame[category].astype(str) != "")
        & np.isfinite(frame[outcome].astype(float))
    ].copy()
    category_values = set(complete[category].astype(str))
    ordered_categories = [
        value for value in category_order if value in category_values
    ]
    template_category = complete.pivot_table(
        index="template_PDB",
        columns=category,
        values=outcome,
        aggfunc="mean",
    )
    template_category = template_category.reindex(columns=ordered_categories)
    rng = _rng_for_key(seed, analysis_key)
    sampled_indices = rng.integers(
        0,
        len(template_category),
        size=(bootstrap_replicates, len(template_category)),
    )
    sampled_values = template_category.to_numpy(dtype=float)[sampled_indices]

    rows = []
    for category_index, value in enumerate(ordered_categories):
        cell = complete[complete[category].astype(str) == value]
        template_means = template_category[value].dropna()
        values = np.nanmean(sampled_values[:, :, category_index], axis=1)
        values = values[np.isfinite(values)]
        if len(values) < max(10, bootstrap_replicates // 2):
            raise ValueError(
                f"{analysis_key} {value}: too few finite bootstrap category means"
            )
        ci_low, ci_high = np.quantile(values, [0.025, 0.975])
        rows.append(
            {
                "category": value,
                "estimate": float(template_means.mean()),
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "p_value": float("nan"),
                "n_templates": int(cell["template_PDB"].nunique()),
                "n_mutations": len(cell),
            }
        )
    return rows


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Return Benjamini-Hochberg adjusted q-values with original indexing."""
    finite = p_values.dropna().astype(float)
    if finite.empty:
        return pd.Series(index=p_values.index, dtype=float)
    order = finite.sort_values().index
    ranked = finite.loc[order].to_numpy()
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    result = pd.Series(index=p_values.index, dtype=float)
    result.loc[order] = adjusted
    return result


def _outcome_columns() -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Return model outcomes and pre-specified Prompt-DDG contrast outcomes."""
    model_outcomes = []
    for model_name, specification in MODEL_SPECS.items():
        slug = str(specification["slug"])
        model_outcomes.append(
            (model_name, "absolute_error_change", f"{slug}_absolute_error_change")
        )
        model_outcomes.append(
            (model_name, "af3_wt_absolute_error", f"{slug}_af3_wt_absolute_error")
        )
    contrasts = [
        (
            "BA-DDG minus Prompt-DDG",
            "absolute_error_change_contrast",
            "ba_minus_prompt_absolute_error_change",
        ),
        (
            "H3-DDG minus Prompt-DDG",
            "absolute_error_change_contrast",
            "h3_minus_prompt_absolute_error_change",
        ),
    ]
    return model_outcomes, contrasts


def _template_outcome_column(
    model_or_contrast: str,
    outcome: str,
) -> str:
    """Map a mutation-level outcome description to the S15 template column."""
    if model_or_contrast == "BA-DDG minus Prompt-DDG":
        return "ba_minus_prompt_mean_absolute_error_change"
    if model_or_contrast == "H3-DDG minus Prompt-DDG":
        return "h3_minus_prompt_mean_absolute_error_change"
    slug = str(MODEL_SPECS[model_or_contrast]["slug"])
    if outcome == "absolute_error_change":
        return f"{slug}_mean_absolute_error_change"
    if outcome == "af3_wt_absolute_error":
        return f"{slug}_af3_wt_mae"
    raise KeyError((model_or_contrast, outcome))


def build_association_table(
    template_table: pd.DataFrame,
    mutation_table: pd.DataFrame,
    contact_cutoffs: Iterable[float],
    bootstrap_replicates: int,
    permutation_replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Build long-form template, mutation, category, and sensitivity results."""
    if bootstrap_replicates < 1 or permutation_replicates < 0:
        raise ValueError("bootstrap replicates must be positive and permutations nonnegative")

    rows: list[dict[str, object]] = []
    model_outcomes, contrasts = _outcome_columns()
    all_outcomes = model_outcomes + contrasts
    template_features = PRIMARY_TEMPLATE_FEATURES + EXPLORATORY_TEMPLATE_FEATURES

    for model_or_contrast, outcome, mutation_outcome_column in all_outcomes:
        template_outcome = _template_outcome_column(model_or_contrast, outcome)
        for feature in template_features:
            is_primary = (
                outcome in {"absolute_error_change", "absolute_error_change_contrast"}
                and feature in PRIMARY_TEMPLATE_FEATURES
            )
            key = (
                f"template|all|{model_or_contrast}|{outcome}|{feature}"
            )
            estimate = template_spearman_association(
                template_table,
                feature,
                template_outcome,
                bootstrap_replicates,
                permutation_replicates if is_primary else 0,
                seed,
                key,
            )
            rows.append(
                {
                    "analysis_level": "template",
                    "analysis_type": "spearman",
                    "subset": "all_templates",
                    "model_or_contrast": model_or_contrast,
                    "outcome": outcome,
                    "feature": feature,
                    "category": "",
                    "is_primary": is_primary,
                    "fdr_family": "template_primary"
                    if is_primary
                    else "",
                    "contact_cutoff_angstrom": PRIMARY_CONTACT_CUTOFF
                    if "contact_" in feature
                    else float("nan"),
                    **estimate,
                }
            )

    minimum_three = template_table[template_table["n_mutations"] >= 3]
    for model_or_contrast, outcome, _ in all_outcomes:
        if outcome not in {"absolute_error_change", "absolute_error_change_contrast"}:
            continue
        template_outcome = _template_outcome_column(model_or_contrast, outcome)
        for feature in PRIMARY_TEMPLATE_FEATURES:
            key = (
                f"template|n_ge_3|{model_or_contrast}|{outcome}|{feature}"
            )
            estimate = template_spearman_association(
                minimum_three,
                feature,
                template_outcome,
                bootstrap_replicates,
                0,
                seed,
                key,
            )
            rows.append(
                {
                    "analysis_level": "template",
                    "analysis_type": "spearman_sensitivity",
                    "subset": "templates_n_mutations_ge_3",
                    "model_or_contrast": model_or_contrast,
                    "outcome": outcome,
                    "feature": feature,
                    "category": "",
                    "is_primary": False,
                    "fdr_family": "",
                    "contact_cutoff_angstrom": PRIMARY_CONTACT_CUTOFF
                    if "contact_" in feature
                    else float("nan"),
                    **estimate,
                }
            )

    sensitivity_cutoffs = [
        cutoff
        for cutoff in sorted(set(float(value) for value in contact_cutoffs))
        if cutoff != PRIMARY_CONTACT_CUTOFF
    ]
    for cutoff in sensitivity_cutoffs:
        template_feature = f"contact_disagreement_{cutoff:g}a"
        mutation_feature = f"local_contact_disagreement_{cutoff:g}a"
        for model_or_contrast, outcome, mutation_outcome in all_outcomes:
            if outcome not in {"absolute_error_change", "absolute_error_change_contrast"}:
                continue
            template_outcome = _template_outcome_column(model_or_contrast, outcome)
            template_key = (
                f"template|cutoff_{cutoff:g}|{model_or_contrast}|{outcome}"
            )
            template_estimate = template_spearman_association(
                template_table,
                template_feature,
                template_outcome,
                bootstrap_replicates,
                0,
                seed,
                template_key,
            )
            rows.append(
                {
                    "analysis_level": "template",
                    "analysis_type": "contact_cutoff_sensitivity",
                    "subset": "all_templates",
                    "model_or_contrast": model_or_contrast,
                    "outcome": outcome,
                    "feature": template_feature,
                    "category": "",
                    "is_primary": False,
                    "fdr_family": "",
                    "contact_cutoff_angstrom": cutoff,
                    **template_estimate,
                }
            )
            mutation_key = (
                f"mutation|cutoff_{cutoff:g}|{model_or_contrast}|{outcome}"
            )
            mutation_estimate = mutation_rank_association(
                mutation_table,
                mutation_feature,
                mutation_outcome,
                bootstrap_replicates,
                0,
                seed,
                mutation_key,
            )
            rows.append(
                {
                    "analysis_level": "mutation",
                    "analysis_type": "contact_cutoff_sensitivity",
                    "subset": "all_mutations",
                    "model_or_contrast": model_or_contrast,
                    "outcome": outcome,
                    "feature": mutation_feature,
                    "category": "",
                    "is_primary": False,
                    "fdr_family": "",
                    "contact_cutoff_angstrom": cutoff,
                    **mutation_estimate,
                }
            )

    mutation_feature_specs = [
        ("local_contact_disagreement_5a", "all_mutations", mutation_table, True),
    ]
    for feature in EXPLORATORY_MUTATION_FEATURES:
        mutation_feature_specs.append(
            (feature, "all_mutations", mutation_table, False)
        )
    peptide_subset = mutation_table[
        (mutation_table["num_muts"] == 1)
        & (mutation_table["single_point_region"] == "Peptide")
    ]
    mutation_feature_specs.append(
        (
            "peptide_position_normalized",
            "single_point_peptide_mutations",
            peptide_subset,
            True,
        )
    )

    for model_or_contrast, outcome, mutation_outcome in all_outcomes:
        for feature, subset_name, subset_frame, primary_feature in mutation_feature_specs:
            is_primary = (
                primary_feature
                and outcome in {"absolute_error_change", "absolute_error_change_contrast"}
            )
            key = (
                f"mutation|{subset_name}|{model_or_contrast}|{outcome}|{feature}"
            )
            estimate = mutation_rank_association(
                subset_frame,
                feature,
                mutation_outcome,
                bootstrap_replicates,
                permutation_replicates if is_primary else 0,
                seed,
                key,
            )
            if feature == "peptide_position_normalized":
                fdr_family = "peptide_position_primary" if is_primary else ""
            else:
                fdr_family = "mutation_contact_primary" if is_primary else ""
            rows.append(
                {
                    "analysis_level": "mutation",
                    "analysis_type": "within_template_rank",
                    "subset": subset_name,
                    "model_or_contrast": model_or_contrast,
                    "outcome": outcome,
                    "feature": feature,
                    "category": "",
                    "is_primary": is_primary,
                    "fdr_family": fdr_family,
                    "contact_cutoff_angstrom": PRIMARY_CONTACT_CUTOFF
                    if "contact_" in feature
                    else float("nan"),
                    **estimate,
                }
            )

    single_points = mutation_table[mutation_table["num_muts"] == 1].copy()
    region_order = [
        "TCR CDR3",
        "Other TCR CDR",
        "TCR framework",
        "Peptide",
        "MHC",
    ]
    peptide_categories = ["N-terminal third", "Middle third", "C-terminal third"]
    peptide_subset = peptide_subset.copy()
    peptide_subset["peptide_position_third"] = pd.cut(
        peptide_subset["peptide_position_normalized"],
        bins=[-np.inf, 1.0 / 3.0, 2.0 / 3.0, np.inf],
        labels=peptide_categories,
        include_lowest=True,
        right=True,
    ).astype(str)
    contact_categories = list(CONTACT_STRATUM_LABELS)
    contact_strata = mutation_table.copy()
    contact_strata["local_contact_disagreement_stratum"] = (
        assign_contact_disagreement_strata(
            contact_strata["local_contact_disagreement_5a"]
        )
    )

    for model_name, outcome, mutation_outcome in model_outcomes:
        for category, subset_name, subset_frame, order in (
            (
                "local_contact_disagreement_stratum",
                "all_mutations",
                contact_strata,
                contact_categories,
            ),
            ("single_point_region", "single_point_mutations", single_points, region_order),
            (
                "peptide_position_third",
                "single_point_peptide_mutations",
                peptide_subset,
                peptide_categories,
            ),
        ):
            key = f"category|{category}|{model_name}|{outcome}"
            estimates = equal_template_category_estimates(
                subset_frame,
                category,
                mutation_outcome,
                order,
                bootstrap_replicates,
                seed,
                key,
            )
            for estimate in estimates:
                rows.append(
                    {
                        "analysis_level": "mutation",
                        "analysis_type": "equal_template_category_mean",
                        "subset": subset_name,
                        "model_or_contrast": model_name,
                        "outcome": outcome,
                        "feature": category,
                        "is_primary": False,
                        "fdr_family": "",
                        "contact_cutoff_angstrom": float("nan"),
                        **estimate,
                    }
                )

    rows.append(
        {
            "analysis_level": "template",
            "analysis_type": "excluded_feature",
            "subset": "all_templates",
            "model_or_contrast": "",
            "outcome": "",
            "feature": "has_clash",
            "category": "",
            "is_primary": False,
            "fdr_family": "",
            "contact_cutoff_angstrom": float("nan"),
            "estimate": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "p_value": float("nan"),
            "n_templates": len(template_table),
            "n_mutations": int(template_table["n_mutations"].sum()),
            "notes": "Excluded because all 37 selected AF3-WT models have has_clash=0.",
        }
    )

    associations = pd.DataFrame(rows)
    expected_contact_counts = (
        contact_strata["local_contact_disagreement_stratum"]
        .value_counts()
        .to_dict()
    )
    contact_rows = associations[
        (
            associations["feature"]
            == "local_contact_disagreement_stratum"
        )
        & (associations["analysis_type"] == "equal_template_category_mean")
    ]
    for category, expected_count in expected_contact_counts.items():
        observed_counts = set(
            contact_rows.loc[
                contact_rows["category"] == category,
                "n_mutations",
            ].astype(int)
        )
        if observed_counts != {int(expected_count)}:
            raise ValueError(
                f"contact stratum {category!r} expected {expected_count} mutations, "
                f"association rows contain {sorted(observed_counts)}"
            )
    associations["q_value"] = np.nan
    for family, family_rows in associations[
        associations["fdr_family"].astype(str) != ""
    ].groupby("fdr_family"):
        associations.loc[family_rows.index, "q_value"] = benjamini_hochberg(
            family_rows["p_value"]
        )
    associations["bootstrap_replicates"] = bootstrap_replicates
    associations["permutation_replicates"] = permutation_replicates
    associations["notes"] = associations.get("notes", "").fillna("")
    analysis_level_order = {"template": 0, "mutation": 1}
    associations["_analysis_level_order"] = (
        associations["analysis_level"]
        .map(analysis_level_order)
        .fillna(len(analysis_level_order))
    )
    return associations.sort_values(
        [
            "_analysis_level_order",
            "analysis_type",
            "subset",
            "outcome",
            "model_or_contrast",
            "feature",
            "category",
        ]
    ).drop(columns="_analysis_level_order").reset_index(drop=True)


def run_analysis(
    predictions_csv: Path,
    index_csv: Path,
    pdb_dir: Path,
    af3_wt_dir: Path,
    contact_cutoff: float = PRIMARY_CONTACT_CUTOFF,
    sensitivity_cutoffs: Iterable[float] = DEFAULT_SENSITIVITY_CUTOFFS,
    bootstrap_replicates: int = 1000,
    permutation_replicates: int = 1000,
    seed: int = 20260722,
    *,
    validate_raw_sources: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run feature extraction and return S15, S16, and S17 data frames."""
    if not np.isclose(contact_cutoff, PRIMARY_CONTACT_CUTOFF):
        raise ValueError(
            "the manuscript primary contact cutoff is fixed at 5 angstroms; "
            "use sensitivity_cutoffs for alternatives"
        )
    contact_cutoffs = sorted(
        {
            float(contact_cutoff),
            *(float(value) for value in sensitivity_cutoffs),
        }
    )
    cohort = load_common_cohort(
        predictions_csv,
        index_csv,
        validate_raw_sources=validate_raw_sources,
    )
    contexts: dict[str, TemplateContext] = {}
    for template in sorted(cohort["template_PDB"].unique()):
        log.info("Extracting AF3-WT features for %s", template)
        contexts[template] = extract_template_context(
            template,
            pdb_dir,
            af3_wt_dir,
            contact_cutoffs,
        )
    mutation_table = build_mutation_table(cohort, contexts, contact_cutoffs)
    template_table = build_template_table(contexts, mutation_table)
    associations = build_association_table(
        template_table,
        mutation_table,
        contact_cutoffs,
        bootstrap_replicates,
        permutation_replicates,
        seed,
    )
    return template_table, mutation_table, associations


def main() -> None:
    """Parse command-line arguments, run the analysis, and write S15--S17."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=DATA_ROOT / "ddg_predictions" / "compare_4models.csv",
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=DATA_ROOT / "ATLAS_af" / "index.csv",
    )
    parser.add_argument(
        "--pdb-dir",
        type=Path,
        default=DATA_ROOT / "ATLAS" / "www" / "structures" / "true_pdb",
    )
    parser.add_argument(
        "--af3-wt-dir",
        type=Path,
        default=DATA_ROOT / "ATLAS_af" / "wt_output",
    )
    parser.add_argument(
        "--contact-cutoff",
        type=float,
        default=PRIMARY_CONTACT_CUTOFF,
    )
    parser.add_argument(
        "--sensitivity-cutoffs",
        type=float,
        nargs="*",
        default=list(DEFAULT_SENSITIVITY_CUTOFFS),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--permutation-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--template-output",
        type=Path,
        default=(
            OUTPUT_ROOT
            / "supplementary_tables"
            / "S15.csv"
        ),
    )
    parser.add_argument(
        "--mutation-output",
        type=Path,
        default=(
            OUTPUT_ROOT
            / "supplementary_tables"
            / "S16.csv"
        ),
    )
    parser.add_argument(
        "--associations-output",
        type=Path,
        default=(
            OUTPUT_ROOT
            / "supplementary_tables"
            / "S17.csv"
        ),
    )
    parser.add_argument(
        "--skip-raw-source-reconciliation",
        action="store_true",
        help="Skip checking compare_4models.csv against the eight raw result CSVs.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    template_table, mutation_table, associations = run_analysis(
        args.predictions_csv,
        args.index_csv,
        args.pdb_dir,
        args.af3_wt_dir,
        contact_cutoff=args.contact_cutoff,
        sensitivity_cutoffs=args.sensitivity_cutoffs,
        bootstrap_replicates=args.bootstrap_replicates,
        permutation_replicates=args.permutation_replicates,
        seed=args.seed,
        validate_raw_sources=not args.skip_raw_source_reconciliation,
    )
    for path, frame in (
        (args.template_output, template_table),
        (args.mutation_output, mutation_table),
        (args.associations_output, associations),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        print(f"Wrote {path} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
