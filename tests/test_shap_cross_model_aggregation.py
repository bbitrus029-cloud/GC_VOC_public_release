from __future__ import annotations

from typing import Any

import numpy as np
import pytest


def _import_symbols() -> tuple[Any, ...]:
    """Import shap-cross-model helpers lazily for clear test-first failures."""
    try:
        from src.analysis.shap_cross_model import (
            _aggregate_feature_matrix_to_channels,
            _aggregate_feature_vector_to_channels,
            _build_channel_feature_map,
            _normalize_importance_vector,
        )
    except (AttributeError, ImportError, ModuleNotFoundError) as exc:
        pytest.fail(f"shap_cross_model implementation is missing: {exc}", pytrace=False)

    return (
        _aggregate_feature_matrix_to_channels,
        _aggregate_feature_vector_to_channels,
        _build_channel_feature_map,
        _normalize_importance_vector,
    )


def _make_feature_names() -> list[str]:
    """Create a synthetic feature-first `feature_S#` Pipeline B schema."""
    prefixes = [f"F{index}" for index in range(1, 10)]
    return [f"{prefix}_S{channel}" for prefix in prefixes for channel in range(1, 12)]


def test_build_channel_feature_map_parses_feature_first_schema() -> None:
    """The helper should recover 11 channels with 9 features each."""
    _aggregate_matrix, _aggregate_vector, build_map, _normalize = _import_symbols()

    channel_map = build_map(_make_feature_names())

    assert list(channel_map.keys()) == [f"S{index}" for index in range(1, 12)]
    assert {len(indices) for indices in channel_map.values()} == {9}


def test_aggregate_feature_vector_to_channels_sums_nine_features_per_channel() -> None:
    """A 99-vector should collapse to the expected 11-channel totals."""
    _aggregate_matrix, aggregate_vector, _build_map, normalize = _import_symbols()

    feature_names = _make_feature_names()
    feature_vector = np.asarray(
        [channel for _prefix in range(9) for channel in range(1, 12)],
        dtype=float,
    )

    channel_vector = aggregate_vector(feature_vector, feature_names, absolute=True)
    normalized = normalize(channel_vector)

    expected = np.asarray([9.0 * channel for channel in range(1, 12)], dtype=float)
    assert np.allclose(channel_vector, expected)
    assert normalized.shape == (11,)
    assert normalized.sum() == pytest.approx(1.0)


def test_aggregate_feature_matrix_to_channels_uses_abs_then_mean_over_samples() -> None:
    """A `[N, 99]` matrix should aggregate samplewise magnitudes before averaging."""
    aggregate_matrix, _aggregate_vector, _build_map, _normalize = _import_symbols()

    feature_names = _make_feature_names()
    base = np.asarray(
        [channel for _prefix in range(9) for channel in range(1, 12)],
        dtype=float,
    )
    feature_matrix = np.vstack([base, -base])

    channel_vector = aggregate_matrix(feature_matrix, feature_names, absolute=True)

    expected = np.asarray([9.0 * channel for channel in range(1, 12)], dtype=float)
    assert channel_vector.shape == (11,)
    assert np.allclose(channel_vector, expected)
