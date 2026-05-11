"""Unit tests for label-blind discovery primitives."""

from __future__ import annotations

import numpy as np
import pytest


def _import_discovery_symbols() -> tuple:
    """Import discovery symbols lazily so the pre-implementation test fails cleanly."""
    try:
        from src.data.discovery import (
            compute_kofn_fallback_median,
            compute_otsu_threshold,
            compute_tukey_lower_inner_fence,
            compute_tukey_upper_inner_fence,
            detect_bimodality_kde,
        )
    except ModuleNotFoundError as exc:
        pytest.fail(f"Discovery module is missing: {exc}", pytrace=False)

    return (
        compute_kofn_fallback_median,
        compute_otsu_threshold,
        compute_tukey_lower_inner_fence,
        compute_tukey_upper_inner_fence,
        detect_bimodality_kde,
    )


def _import_run_discovery_threshold_helpers() -> tuple:
    """Import threshold helpers from the discovery script for regression tests."""
    try:
        from scripts.run_discovery import (
            QC5_FALLBACK_PERCENTILE,
            _compute_qc4_response_amplitude_threshold,
            _compute_qc5_saturation_threshold,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.fail(f"Discovery script threshold helpers are missing: {exc}", pytrace=False)

    return (
        QC5_FALLBACK_PERCENTILE,
        _compute_qc4_response_amplitude_threshold,
        _compute_qc5_saturation_threshold,
    )


def test_compute_otsu_threshold_falls_between_known_modes() -> None:
    """Otsu threshold should separate a clearly bimodal synthetic distribution."""
    (
        _compute_kofn_fallback_median,
        compute_otsu_threshold,
        _compute_tukey_lower_inner_fence,
        _compute_tukey_upper_inner_fence,
        _detect_bimodality_kde,
    ) = _import_discovery_symbols()
    values = np.concatenate(
        [
            np.linspace(-3.0, -1.0, 400),
            np.linspace(1.0, 3.0, 400),
        ]
    )

    threshold = compute_otsu_threshold(values)

    assert -1.0 < threshold < 1.0


def test_compute_tukey_upper_inner_fence_matches_hand_calculation() -> None:
    """Upper inner fence should equal Q3 plus one-and-a-half IQR."""
    (
        _compute_kofn_fallback_median,
        _compute_otsu_threshold,
        _compute_tukey_lower_inner_fence,
        compute_tukey_upper_inner_fence,
        _detect_bimodality_kde,
    ) = _import_discovery_symbols()
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)

    threshold = compute_tukey_upper_inner_fence(values)

    assert threshold == pytest.approx(7.0)


def test_compute_tukey_lower_inner_fence_clips_negative_values_to_zero() -> None:
    """Lower inner fence should clip to zero when requested."""
    (
        _compute_kofn_fallback_median,
        _compute_otsu_threshold,
        compute_tukey_lower_inner_fence,
        _compute_tukey_upper_inner_fence,
        _detect_bimodality_kde,
    ) = _import_discovery_symbols()
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)

    threshold = compute_tukey_lower_inner_fence(values, clip_at_zero=True)

    assert threshold == pytest.approx(0.0)


def test_detect_bimodality_kde_returns_true_for_bimodal_values() -> None:
    """KDE peak counting should flag a clearly bimodal distribution."""
    (
        _compute_kofn_fallback_median,
        _compute_otsu_threshold,
        _compute_tukey_lower_inner_fence,
        _compute_tukey_upper_inner_fence,
        detect_bimodality_kde,
    ) = _import_discovery_symbols()
    rng = np.random.default_rng(42)
    values = np.concatenate(
        [
            rng.normal(loc=-4.0, scale=0.15, size=300),
            rng.normal(loc=4.0, scale=0.15, size=300),
        ]
    )

    assert detect_bimodality_kde(values) is True


def test_detect_bimodality_kde_returns_false_for_unimodal_values() -> None:
    """KDE peak counting should reject a single Gaussian mode."""
    (
        _compute_kofn_fallback_median,
        _compute_otsu_threshold,
        _compute_tukey_lower_inner_fence,
        _compute_tukey_upper_inner_fence,
        detect_bimodality_kde,
    ) = _import_discovery_symbols()
    rng = np.random.default_rng(42)
    values = rng.normal(loc=0.0, scale=1.0, size=800)

    assert detect_bimodality_kde(values) is False


def test_detect_bimodality_kde_returns_false_for_constant_values() -> None:
    """Constant values should produce zero strict interior peaks."""
    (
        _compute_kofn_fallback_median,
        _compute_otsu_threshold,
        _compute_tukey_lower_inner_fence,
        _compute_tukey_upper_inner_fence,
        detect_bimodality_kde,
    ) = _import_discovery_symbols()
    values = np.full(6, 5.0, dtype=float)

    assert detect_bimodality_kde(values) is False


def test_compute_kofn_fallback_median_matches_hand_computed_value() -> None:
    """Fallback median should match the median of triggered samples."""
    (
        compute_kofn_fallback_median,
        _compute_otsu_threshold,
        _compute_tukey_lower_inner_fence,
        _compute_tukey_upper_inner_fence,
        _detect_bimodality_kde,
    ) = _import_discovery_symbols()
    triggered_t_baseline_end = np.array([9.0, 11.0, 13.0, 15.0, 17.0], dtype=float)

    fallback_median = compute_kofn_fallback_median(triggered_t_baseline_end)

    assert fallback_median == pytest.approx(13.0)


def test_qc4_threshold_extraction_uses_tukey_lower_fence_unconditionally() -> None:
    """QC-4 should use the Tukey lower fence even for a bimodal pooled |r| distribution."""
    (
        _compute_kofn_fallback_median,
        compute_otsu_threshold,
        compute_tukey_lower_inner_fence,
        _compute_tukey_upper_inner_fence,
        _detect_bimodality_kde,
    ) = _import_discovery_symbols()
    (
        _qc5_fallback_percentile,
        compute_qc4_response_amplitude_threshold,
        _compute_qc5_saturation_threshold,
    ) = _import_run_discovery_threshold_helpers()
    pooled_abs_r = np.concatenate(
        [
            np.linspace(0.30, 0.60, 200),
            np.linspace(10.0, 12.0, 40),
        ]
    )

    threshold, threshold_method = compute_qc4_response_amplitude_threshold(pooled_abs_r)

    assert threshold_method == "tukey_lower_inner_fence"
    assert threshold == pytest.approx(compute_tukey_lower_inner_fence(pooled_abs_r, clip_at_zero=True))
    assert threshold != pytest.approx(compute_otsu_threshold(pooled_abs_r))


def test_qc5_threshold_extraction_uses_p95_fallback_when_iqr_is_zero() -> None:
    """QC-5 should use the P95 fallback when the pooled saturation IQR collapses to zero."""
    (
        _compute_kofn_fallback_median,
        _compute_otsu_threshold,
        _compute_tukey_lower_inner_fence,
        _compute_tukey_upper_inner_fence,
        _detect_bimodality_kde,
    ) = _import_discovery_symbols()
    (
        qc5_fallback_percentile,
        _compute_qc4_response_amplitude_threshold,
        compute_qc5_saturation_threshold,
    ) = _import_run_discovery_threshold_helpers()
    pooled_saturation_fraction = np.concatenate(
        [
            np.full(95, 0.001, dtype=float),
            np.array([0.010, 0.020, 0.030, 0.040, 0.079], dtype=float),
        ]
    )

    threshold, threshold_method = compute_qc5_saturation_threshold(pooled_saturation_fraction)

    assert threshold_method == "p95_fallback"
    assert threshold == pytest.approx(
        np.percentile(pooled_saturation_fraction, qc5_fallback_percentile, method="linear")
    )


def test_qc5_threshold_extraction_uses_tukey_when_iqr_is_nonzero() -> None:
    """QC-5 should keep the Tukey upper inner fence when the pooled saturation IQR is nonzero."""
    (
        _compute_kofn_fallback_median,
        _compute_otsu_threshold,
        _compute_tukey_lower_inner_fence,
        compute_tukey_upper_inner_fence,
        _detect_bimodality_kde,
    ) = _import_discovery_symbols()
    (
        _qc5_fallback_percentile,
        _compute_qc4_response_amplitude_threshold,
        compute_qc5_saturation_threshold,
    ) = _import_run_discovery_threshold_helpers()
    pooled_saturation_fraction = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.015, 0.020], dtype=float)

    threshold, threshold_method = compute_qc5_saturation_threshold(pooled_saturation_fraction)

    assert threshold_method == "tukey_upper_inner_fence"
    assert threshold == pytest.approx(compute_tukey_upper_inner_fence(pooled_saturation_fraction))


def test_qc5_threshold_extraction_halts_when_distribution_is_all_identical_nonzero() -> None:
    """QC-5 should retain the nonzero all-identical halt instead of using the fallback."""
    (
        _compute_kofn_fallback_median,
        _compute_otsu_threshold,
        _compute_tukey_lower_inner_fence,
        _compute_tukey_upper_inner_fence,
        _detect_bimodality_kde,
    ) = _import_discovery_symbols()
    (
        _qc5_fallback_percentile,
        _compute_qc4_response_amplitude_threshold,
        compute_qc5_saturation_threshold,
    ) = _import_run_discovery_threshold_helpers()
    pooled_saturation_fraction = np.full(12, 0.02, dtype=float)

    with pytest.raises(RuntimeError, match="zero variance|all identical|saturation-fraction"):
        compute_qc5_saturation_threshold(pooled_saturation_fraction)
