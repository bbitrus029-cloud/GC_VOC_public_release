from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest


def _import_symbols() -> tuple[Any, ...]:
    """Import MiniRocket permutation helpers lazily for clear failures."""
    try:
        from src.analysis.shap_cross_model import (
            _extract_minirocket_ridge_importance,
            _permute_channel_across_samples,
        )
    except (AttributeError, ImportError, ModuleNotFoundError) as exc:
        pytest.fail(f"shap_cross_model implementation is missing: {exc}", pytrace=False)

    return _extract_minirocket_ridge_importance, _permute_channel_across_samples


class _IdentityNormalizer:
    """Keep the transformed feature frame unchanged."""

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        return x.copy()


class _MeanChannelTransformer:
    """Collapse `[N, C, L]` inputs to one feature per channel."""

    def transform(self, x: np.ndarray) -> pd.DataFrame:
        channel_means = np.asarray(x, dtype=float).mean(axis=2)
        columns = [f"S{index}" for index in range(1, channel_means.shape[1] + 1)]
        return pd.DataFrame(channel_means, columns=columns)


class _ChannelOneModel:
    """Predict from the first transformed channel only."""

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        channel_one = x.iloc[:, 0].to_numpy(dtype=float)
        probabilities = 1.0 / (1.0 + np.exp(-channel_one))
        return np.column_stack([1.0 - probabilities, probabilities])


def test_permute_channel_across_samples_only_touches_requested_channel() -> None:
    """Raw-signal permutation should leave all non-target channels untouched."""
    _extract_importance, permute_channel = _import_symbols()

    x = np.arange(3 * 4 * 2, dtype=float).reshape(3, 4, 2)
    permutation = np.asarray([2, 0, 1], dtype=int)

    permuted = permute_channel(x, channel_index=1, permutation=permutation)

    assert np.array_equal(permuted[:, :, 0], x[:, :, 0])
    assert np.array_equal(permuted[:, :, 1], x[permutation, :, 1])


def test_extract_minirocket_ridge_importance_returns_non_negative_vector() -> None:
    """Channelwise permutation importance should be non-negative and length 11."""
    extract_importance, _permute_channel = _import_symbols()

    y_true = np.asarray([0] * 10 + [1] * 10, dtype=int)
    x = np.zeros((20, 6, 11), dtype=float)
    x[:10, :, 0] = -3.0
    x[10:, :, 0] = 3.0
    x[:, :, 1:] = np.linspace(-1.0, 1.0, num=20, dtype=float).reshape(20, 1, 1)

    payload = {
        "transformer": _MeanChannelTransformer(),
        "normalizer": _IdentityNormalizer(),
        "model": _ChannelOneModel(),
    }

    importance = extract_importance(payload, x, y_true, n_repeats=5, seed=42)

    assert importance.shape == (11,)
    assert np.all(importance >= 0.0)
    assert importance[0] > 0.0
