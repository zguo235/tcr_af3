"""Focused checks for pooled metrics under whole-template bootstrap sampling."""

import unittest

import numpy as np
from scipy.stats import pearsonr

from analyze_template_overlap_performance import (
    metrics_from_template_counts,
    template_statistics,
)


class TemplateBootstrapTests(unittest.TestCase):
    """Compare sufficient-statistic results with independently expanded records."""

    def test_unequal_template_sizes_and_repeated_templates(self):
        """Every draw must retain the full template and its original record weight."""
        observed = np.array([1.0, 2.0, -1.0, 4.0, 6.0, 0.5])
        predicted = np.array([-0.2, 2.0, 1.0, 3.0, 5.0, -0.3])
        codes = np.array([0, 0, 0, 1, 1, 2])
        counts = np.array([[1, 1, 1], [3, 0, 0], [0, 2, 1], [2, 1, 3]])
        statistics = template_statistics(observed, predicted, codes)
        actual = metrics_from_template_counts(statistics, counts)
        expected = []
        for draw in counts:
            rows = np.concatenate([
                np.tile(np.flatnonzero(codes == template), repetitions)
                for template, repetitions in enumerate(draw)
            ])
            y, p = observed[rows], predicted[rows]
            expected.append([
                pearsonr(y, p).statistic,
                np.sqrt(np.mean((y - p) ** 2)),
                np.mean(np.abs(y - p)),
            ])
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)

    def test_degenerate_correlation_preserves_error_metrics(self):
        """A constant bootstrap target has no correlation, but errors remain valid."""
        observed = np.array([2.0, 2.0, 3.0, 4.0])
        predicted = np.array([1.0, 4.0, 2.0, 5.0])
        codes = np.array([0, 0, 1, 1])
        statistics = template_statistics(observed, predicted, codes)
        result = metrics_from_template_counts(statistics, np.array([[2, 0]]))[0]
        self.assertTrue(np.isnan(result[0]))
        np.testing.assert_allclose(result[1:], [np.sqrt(2.5), 1.5], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
