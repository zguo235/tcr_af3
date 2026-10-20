"""Regression tests for multi-template CDR cohort construction."""

import pandas as pd

from cdr_prioritization_per_template import restrict_to_evaluation_cohort


def test_restrict_to_evaluation_cohort_removes_precleaning_rows() -> None:
    index_rows = pd.DataFrame(
        {
            "row_idx": [10, 11, 12, 13],
            "template_PDB": ["1AO7", "1AO7", "3QIB", "3QIB"],
        }
    )
    prediction_rows = pd.DataFrame({"row_idx": [10, 12]})

    filtered = restrict_to_evaluation_cohort(index_rows, prediction_rows)

    assert filtered["row_idx"].tolist() == [10, 12]
    assert filtered["template_PDB"].tolist() == ["1AO7", "3QIB"]
