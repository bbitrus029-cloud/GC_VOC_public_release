"""Public evaluation utilities for audit-time analyses."""

from __future__ import annotations

from src.evaluation.calibration import (
    apply_platt_calibrator,
    compute_brier_before_after,
    compute_calibration_slope_intercept,
    fit_platt_calibrator,
)
from src.evaluation.metrics import (
    compute_calibration_metrics,
    compute_count_metrics,
    compute_full_metric_set,
    compute_threshold_dependent_metrics,
    compute_threshold_free_metrics,
)
from src.evaluation.statistical_tests import (
    compute_idi,
    compute_nri,
    compute_nri_idi,
    delong_roc_test,
    fisher_exact_2x2,
    holm_bonferroni,
    jonckheere_terpstra,
    mcnemar_test,
    spearman_correlation,
)


__all__ = [
    "apply_platt_calibrator",
    "compute_brier_before_after",
    "compute_calibration_metrics",
    "compute_calibration_slope_intercept",
    "compute_count_metrics",
    "compute_full_metric_set",
    "compute_idi",
    "compute_nri",
    "compute_nri_idi",
    "compute_threshold_dependent_metrics",
    "compute_threshold_free_metrics",
    "delong_roc_test",
    "fisher_exact_2x2",
    "fit_platt_calibrator",
    "holm_bonferroni",
    "jonckheere_terpstra",
    "mcnemar_test",
    "spearman_correlation",
]
