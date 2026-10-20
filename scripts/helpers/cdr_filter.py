"""Shared CDR-membership helpers for candidate-prioritization analyses.

Factored out of ``scripts/cdr_prioritization_1ao7.py`` so the single-template
1AO7 analysis and the multi-template extension share one source of truth for
the IMGT CDR rule.

Chain convention for ATLAS ``true_pdb/*.pdb``: D = TCR α, E = TCR β
(see project CLAUDE.md "ATLAS → true_pdb chain mapping"). ANARCI silently
fails on a documented short list of templates (1BD2, 1FO0, 1NAM, 2OL3, 3FFC,
4G8G, 4G9F); none of those carry > 10 ATLAS mutants, so hard-coding D/E is
safe for the n > 10 cohort.
"""

import re
import sys
from pathlib import Path

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1

# Make ``compare_bound_unbound.get_imgt_numbering`` importable from anywhere.
_HERE = Path(__file__).resolve()
_SCRIPTS = _HERE.parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from compare_bound_unbound import get_imgt_numbering  # noqa: E402

CDR_RANGES = {"CDR1": (27, 38), "CDR2": (56, 65), "CDR3": (105, 117)}

# ATLAS index.csv mutation-token format: "TCR:[D|E]:WT_resseq_MT".
TCR_MUT_RE = re.compile(r"^TCR:([DE]):([A-Z])(\d+)([A-Z])$")

# A V-domain should produce ~110 numbered residues after ANARCI. Below this
# threshold the parse almost certainly mis-identified the domain.
_MIN_V_DOMAIN_RESIDUES = 50


def is_cdr(imgt_pos: int) -> bool:
    """True if an IMGT position falls inside CDR1, CDR2, or CDR3."""
    return any(s <= imgt_pos <= e for s, e in CDR_RANGES.values())


def parse_tcr_mutations(mut_string: str) -> list[tuple[str, int]]:
    """Parse ``TCR:D:S25A;TCR:E:T98A`` -> ``[('D', 25), ('E', 98)]``.

    Non-TCR tokens (MHC:, PEP:) are silently skipped — they cannot be CDR
    mutations under the D/E chain convention.
    """
    if not isinstance(mut_string, str) or not mut_string.strip():
        return []
    out = []
    for tok in mut_string.split(";"):
        m = TCR_MUT_RE.match(tok.strip())
        if m:
            chain, _wt, resseq, _mt = m.groups()
            out.append((chain, int(resseq)))
    return out


def build_imgt_map(
    pdb_path: Path,
    chain_ids: tuple[str, ...] = ("D", "E"),
    *,
    template_id: str | None = None,
) -> dict[tuple[str, int], int]:
    """Build ``{(chain_id, author_resseq): imgt_pos}`` via ANARCI per chain.

    Walks each chain's standard-residue list in step with the ANARCI numbering's
    non-gap entries. ANARCI returns ``(numbering, start, end, chain_type)``
    where ``start..end`` are 0-based inclusive indices into the input sequence
    delimiting the V-domain; constant-region / signal-peptide residues are
    intentionally excluded — TCR mutations outside the V-domain cannot be CDR
    mutations under our IMGT rule.

    Args:
        pdb_path: path to the template PDB file.
        chain_ids: chain IDs to map (default ``("D", "E")`` per ATLAS
            convention).
        template_id: optional template name for error messages.

    Raises:
        RuntimeError if a requested chain is missing, ANARCI fails on a chain,
        or the V-domain residue count looks suspicious (< _MIN_V_DOMAIN_RESIDUES).
    """
    tid = template_id or pdb_path.stem
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(tid, str(pdb_path))
    model = structure[0]
    out: dict[tuple[str, int], int] = {}
    for cid in chain_ids:
        if cid not in [c.id for c in model.get_chains()]:
            raise RuntimeError(f"{tid}: chain {cid} missing from {pdb_path.name}")
        chain = model[cid]
        residues = [r for r in chain if r.get_id()[0] == " "]
        seq = "".join(
            protein_letters_3to1.get(r.get_resname()) or "X" for r in residues
        )
        result = get_imgt_numbering(seq)
        if result is None:
            raise RuntimeError(f"{tid}: ANARCI failed on chain {cid}")
        numbering, start, end, _chain_type = result
        v_residues = residues[start:end + 1]
        if len(v_residues) < _MIN_V_DOMAIN_RESIDUES:
            raise RuntimeError(
                f"{tid}: chain {cid} V-domain only {len(v_residues)} residues "
                f"(threshold {_MIN_V_DOMAIN_RESIDUES})"
            )
        i = 0
        for (imgt_pos, _ins_code), aa in numbering:
            if aa == "-":
                continue
            if i >= len(v_residues):
                break
            author_resseq = v_residues[i].get_id()[1]
            out[(cid, author_resseq)] = imgt_pos
            i += 1
    return out


def row_touches_cdr(
    mut_string: str, imgt_map: dict[tuple[str, int], int]
) -> bool:
    """True if any TCR-side mutation token maps to a CDR IMGT position.

    Lenient compound-mutation rule: a row with N tokens passes the filter if
    *any* token is CDR. Tokens that fail to resolve in ``imgt_map`` (e.g. an
    insertion-coded residue that the integer-resseq map drops) are silently
    treated as non-CDR — callers should use ``count_unresolved_tcr_tokens``
    for a sanity check.
    """
    for chain, resseq in parse_tcr_mutations(mut_string):
        imgt = imgt_map.get((chain, resseq))
        if imgt is not None and is_cdr(imgt):
            return True
    return False


def count_unresolved_tcr_tokens(
    mut_strings, imgt_map: dict[tuple[str, int], int]
) -> int:
    """Count TCR mutation tokens whose ``(chain, resseq)`` is not in ``imgt_map``.

    Non-zero counts indicate insertion-code mismatches or out-of-V-domain
    mutations and warrant inspection before publication.
    """
    n = 0
    for s in mut_strings:
        for chain, resseq in parse_tcr_mutations(s):
            if (chain, resseq) not in imgt_map:
                n += 1
    return n
