"""ATLAS parsers extracted unchanged from the prediction dataset module.

These functions require no model code or PyTorch. See CODE_PROVENANCE.json.
"""

def parse_atlas_mutation(tag: str) -> dict:
    """Parse an ATLAS mutation token.

    Args:
        tag: e.g. ``"TCR:D:S25A"`` — component prefix, chain id, then
            ``{wt}{resseq}{mt}``.

    Returns:
        dict with keys ``wt``, ``mt`` (1-letter AA), ``chain`` (str),
        ``resseq`` (int), ``icode`` (str, always a single space for ATLAS).
    """
    _, chain, body = tag.split(":")
    return {
        "wt": body[0],
        "mt": body[-1],
        "chain": chain,
        "resseq": int(body[1:-1]),
        "icode": " ",
    }


def parse_ddg_exp(x) -> float:
    """Parse an ATLAS Delta_DeltaG cell into a float (nan if missing).

    The clean ATLAS format uses Unicode ``±`` as the uncertainty separator
    and ``−`` (U+2212) for negatives — e.g. ``"0.19 ± 0.11"`` or
    ``"−1.08 ± 0.13"``. Missing markers (``\\N``, ``N/A``, empty) map to
    ``nan``.

    A ``?`` in the cell indicates the upstream Unicode-repair pipeline
    (``ATLAS/fix_atlas_ddg.py`` + ``ATLAS_af/scripts/apply_atlas_fix.py``
    + ``bootstrap.py``) did not run — raise so the error is surfaced at
    its source rather than silently coerced.
    """
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s or s in ("\\N", "N/A", "nan", "NaN"):
        return float("nan")
    if "?" in s:
        raise ValueError(
            f"Mangled '?' in DDG cell {x!r}: upstream Unicode-repair "
            "pipeline did not run. Re-run ATLAS/fix_atlas_ddg.py, then "
            "ATLAS_af/scripts/apply_atlas_fix.py + bootstrap.py."
        )
    if "±" in s:
        s = s.split("±")[0].strip()
    s = s.replace("−", "-")  # U+2212 → ASCII hyphen
    return float(s)
