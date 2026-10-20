"""Configure external analysis inputs and a separate output directory.

Set environment variables before starting a script; paths are resolved at import
time. No directory is created and no input file is modified by this module.
"""

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("FRONTIERS_DATA_ROOT", PACKAGE_ROOT / "data")).expanduser().resolve()
OUTPUT_ROOT = Path(os.environ.get("FRONTIERS_OUTPUT_ROOT", PACKAGE_ROOT / "results")).expanduser().resolve()
SUPPLEMENTARY_ROOT = Path(
    os.environ.get(
        "FRONTIERS_SUPPLEMENTARY_DIR",
        DATA_ROOT / "frontiersin_revision_v3" / "supplementary_tables",
    )
).expanduser().resolve()
