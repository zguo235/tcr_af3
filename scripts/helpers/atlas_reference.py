"""Read the original ATLAS Unicode reference without modifying any dataset."""

import re
from helpers.paths import DATA_ROOT

SOURCE = DATA_ROOT / "ATLAS/www/scoring/Mutants_052915.tsv"
COL_INDEX = 1
COL_TCR_MUT = 7
COL_DDG = 13


def load_source_by_index():
    """Return ``{Index: (TCR_mut, clean_ddg)}`` from Mutants_052915.tsv."""
    out = {}
    with open(SOURCE, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if ln == 1:
                continue
            fields = line.rstrip("\r\n").split("\t")
            out[fields[COL_INDEX - 1]] = (
                fields[COL_TCR_MUT - 1], fields[COL_DDG - 1],
            )
    return out


def digits_only(s):
    return re.sub(r"[^0-9.]", "", s or "")
