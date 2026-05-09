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
