"""Unit tests for Pipeline B feature extraction, persistence, and normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pandas.api.types import is_integer_dtype
import pytest

from src.config import GAS_CHANNELS, RANDOM_SEED


FEATURE_FAMILIES = ("MPD", "MND", "AUC", "Mean", "MAS", "MDS", "T90", "ERR", "LMR")


def _import_pipeline_b_symbols() -> tuple[Any, ...]:
    """Import Pipeline B symbols lazily so test-first failures stay clear."""
    try:
        from src.config import N_GAS_CHANNELS
        from src.features.extraction import (
            apply_pipeline_b_normalization,
            compute_pipeline_b_stats,
            extract_pipeline_b_features,
            save_pipeline_b,
        )
    except (AttributeError, ImportError, ModuleNotFoundError) as exc:
        pytest.fail(f"Pipeline B implementation is missing: {exc}", pytrace=False)

    return (
        apply_pipeline_b_normalization,
        compute_pipeline_b_stats,
        extract_pipeline_b_features,
        save_pipeline_b,
        N_GAS_CHANNELS,
    )


def _expected_feature_columns(channel_names: tuple[str, ...]) -> list[str]:
    """Return the canonical family-then-channel Pipeline B feature order."""
    return [f"{family}_{channel_name}" for family in FEATURE_FAMILIES for channel_name in channel_names]


def _trapezoid_1d(signal: NDArray[np.float64]) -> float:
    """Compute the one-dimensional trapezoidal integral without warning-prone helpers."""
    return float(np.sum((signal[:-1] + signal[1:]) * 0.5))


@pytest.fixture
def synthetic_pipeline_b_inputs() -> tuple[list[NDArray[np.float64]], NDArray[np.int64], NDArray[np.int64], tuple[str, ...]]:
    """Provide deterministic trimmed signals aligned to sorted sample identifiers."""
    _apply_normalization, _compute_stats, _extract, _save, n_gas_channels = _import_pipeline_b_symbols()
    rng = np.random.default_rng(RANDOM_SEED)
    n_samples = 5
    clip_length_l = 200
    sample_ids = np.array([101, 103, 107, 109, 113], dtype=np.int64)
    labels = np.array([0, 1, 0, 1, 1], dtype=np.int64)
    time = np.linspace(0.0, 1.0, clip_length_l, dtype=np.float64)
    signals: list[NDArray[np.float64]] = []

    for sample_index in range(n_samples):
        matrix = np.zeros((clip_length_l, n_gas_channels), dtype=np.float64)
        for channel_index in range(n_gas_channels):
            signal = (
                0.15 * np.sin(2.0 * np.pi * (channel_index + 1) * time)
                + 0.05 * np.cos(np.pi * time * (sample_index + 1))
                + rng.normal(loc=0.0, scale=0.01, size=clip_length_l)
            )
            signal += (sample_index - 2) * 0.03
            matrix[:, channel_index] = signal.astype(np.float64)
        signals.append(matrix)

    return signals, sample_ids, labels, tuple(GAS_CHANNELS)


def test_extract_pipeline_b_features_returns_expected_shape_and_column_order(
    synthetic_pipeline_b_inputs: tuple[list[NDArray[np.float64]], NDArray[np.int64], NDArray[np.int64], tuple[str, ...]]
) -> None:
    """Extraction should return 99 feature columns followed by traceability columns."""
    _apply_normalization, _compute_stats, extract_pipeline_b_features, _save, _n_gas_channels = (
        _import_pipeline_b_symbols()
    )
    signals, sample_ids, labels, channel_names = synthetic_pipeline_b_inputs

    features_df = extract_pipeline_b_features(
        corrected_trimmed_signals=signals,
        sample_ids=sample_ids,
        labels=labels,
        channel_names=channel_names,
    )

    expected_columns = _expected_feature_columns(channel_names) + ["sample_id", "label"]
    assert features_df.shape == (5, 101)
    assert features_df.columns.tolist() == expected_columns


def test_extract_pipeline_b_features_matches_known_signal_semantics() -> None:
    """Known signed response curves should produce the prespecified feature values."""
    _apply_normalization, _compute_stats, extract_pipeline_b_features, _save, n_gas_channels = (
        _import_pipeline_b_symbols()
    )
    known_signal = np.array([0.0, 1.0, -2.0, 4.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float64)
    signal_matrix = np.zeros((known_signal.shape[0], n_gas_channels), dtype=np.float64)
    signal_matrix[:, 0] = known_signal

    features_df = extract_pipeline_b_features(
        corrected_trimmed_signals=[signal_matrix],
        sample_ids=np.array([101], dtype=np.int64),
        labels=np.array([1], dtype=np.int64),
        channel_names=tuple(GAS_CHANNELS),
    )
    row = features_df.iloc[0]

    assert row["MPD_S1"] == pytest.approx(4.0)
    assert row["MND_S1"] == pytest.approx(-2.0)
    assert row["AUC_S1"] == pytest.approx(_trapezoid_1d(known_signal))
    assert row["Mean_S1"] == pytest.approx(float(np.mean(known_signal)))
    assert row["MAS_S1"] == pytest.approx(float(np.max(np.diff(known_signal))))
    assert row["MDS_S1"] == pytest.approx(float(np.min(np.diff(known_signal))))
    assert row["T90_S1"] == pytest.approx(3.0)
    assert row["ERR_S1"] == pytest.approx(float(np.mean(known_signal[:2]) / np.mean(known_signal)))
    assert row["LMR_S1"] == pytest.approx(float(np.mean(known_signal[-5:])))


def test_extract_pipeline_b_features_respects_mpd_mnd_boundaries() -> None:
    """Strictly negative and strictly positive signals should clamp the opposite extreme to zero."""
    _apply_normalization, _compute_stats, extract_pipeline_b_features, _save, n_gas_channels = (
        _import_pipeline_b_symbols()
    )
    signal_matrix = np.zeros((10, n_gas_channels), dtype=np.float64)
    signal_matrix[:, 0] = -2.0
    signal_matrix[:, 1] = 3.0

    features_df = extract_pipeline_b_features(
        corrected_trimmed_signals=[signal_matrix],
        sample_ids=np.array([201], dtype=np.int64),
        labels=np.array([0], dtype=np.int64),
        channel_names=tuple(GAS_CHANNELS),
    )

    assert features_df.loc[0, "MPD_S1"] == pytest.approx(0.0)
    assert features_df.loc[0, "MND_S1"] == pytest.approx(-2.0)
    assert features_df.loc[0, "MPD_S2"] == pytest.approx(3.0)
    assert features_df.loc[0, "MND_S2"] == pytest.approx(0.0)


def test_extract_pipeline_b_features_uses_t90_fallback_for_flat_signal() -> None:
    """A flat zero signal should use `L` as the T90 fallback."""
    _apply_normalization, _compute_stats, extract_pipeline_b_features, _save, n_gas_channels = (
        _import_pipeline_b_symbols()
    )
    signal_matrix = np.zeros((10, n_gas_channels), dtype=np.float64)

    features_df = extract_pipeline_b_features(
        corrected_trimmed_signals=[signal_matrix],
        sample_ids=np.array([301], dtype=np.int64),
        labels=np.array([0], dtype=np.int64),
        channel_names=tuple(GAS_CHANNELS),
    )

    assert features_df.loc[0, "T90_S1"] == pytest.approx(10.0)


def test_extract_pipeline_b_features_uses_err_denominator_guard() -> None:
    """Near-zero mean signals should yield `ERR = 0.0` under the denominator guard."""
    _apply_normalization, _compute_stats, extract_pipeline_b_features, _save, n_gas_channels = (
        _import_pipeline_b_symbols()
    )
    guarded_signal = np.array([1.0, -1.0] * 5, dtype=np.float64)
    signal_matrix = np.zeros((guarded_signal.shape[0], n_gas_channels), dtype=np.float64)
    signal_matrix[:, 0] = guarded_signal

    features_df = extract_pipeline_b_features(
        corrected_trimmed_signals=[signal_matrix],
        sample_ids=np.array([401], dtype=np.int64),
        labels=np.array([1], dtype=np.int64),
        channel_names=tuple(GAS_CHANNELS),
    )

    assert features_df.loc[0, "ERR_S1"] == pytest.approx(0.0)


def test_save_pipeline_b_round_trip_preserves_traceability_dtypes(
    tmp_path: Path,
    synthetic_pipeline_b_inputs: tuple[list[NDArray[np.float64]], NDArray[np.int64], NDArray[np.int64], tuple[str, ...]],
) -> None:
    """Saving should write a CSV that reloads with integer traceability columns."""
    _apply_normalization, _compute_stats, extract_pipeline_b_features, save_pipeline_b, _n_gas_channels = (
        _import_pipeline_b_symbols()
    )
    signals, sample_ids, labels, channel_names = synthetic_pipeline_b_inputs
    features_df = extract_pipeline_b_features(
        corrected_trimmed_signals=signals,
        sample_ids=sample_ids,
        labels=labels,
        channel_names=channel_names,
    )
    output_path = tmp_path / "features" / "pipeline_b_raw.csv"

    save_pipeline_b(df=features_df, path=output_path)
    reloaded_df = pd.read_csv(output_path)

    assert reloaded_df.columns.tolist() == features_df.columns.tolist()
    assert is_integer_dtype(reloaded_df["sample_id"])
    assert is_integer_dtype(reloaded_df["label"])


def test_pipeline_b_normalization_uses_train_stats_without_mutating_traceability() -> None:
    """Feature normalization should leave `sample_id` and `label` untouched."""
    apply_pipeline_b_normalization, compute_pipeline_b_stats, _extract, _save, _n_gas_channels = (
        _import_pipeline_b_symbols()
    )
    feature_columns = _expected_feature_columns(tuple(GAS_CHANNELS))
    train_features = {
        feature_name: np.array([-1.0, 1.0], dtype=np.float64) for feature_name in feature_columns
    }
    test_features = {
        feature_name: np.array([0.0, 2.0], dtype=np.float64) for feature_name in feature_columns
    }
    train_df = pd.DataFrame(train_features)
    train_df["sample_id"] = pd.Series([1, 2], dtype="int64")
    train_df["label"] = pd.Series([0, 1], dtype="int64")
    test_df = pd.DataFrame(test_features)
    test_df["sample_id"] = pd.Series([11, 12], dtype="int64")
    test_df["label"] = pd.Series([1, 0], dtype="int64")
    original_test = test_df.copy(deep=True)

    stats = compute_pipeline_b_stats(x_train_df=train_df)
    normalized_test = apply_pipeline_b_normalization(x_df=test_df, stats=stats)

    assert test_df.equals(original_test)
    assert np.allclose(normalized_test[feature_columns].mean(axis=0).to_numpy(), np.ones(len(feature_columns)))
    assert normalized_test["sample_id"].tolist() == [11, 12]
    assert normalized_test["label"].tolist() == [1, 0]


def test_compute_pipeline_b_stats_clamps_zero_std_to_one() -> None:
    """Constant features should clamp their standard deviation to `1.0`."""
    apply_pipeline_b_normalization, compute_pipeline_b_stats, _extract, _save, _n_gas_channels = (
        _import_pipeline_b_symbols()
    )
    feature_columns = _expected_feature_columns(tuple(GAS_CHANNELS))
    train_features = {
        feature_name: np.array([-1.0, 1.0], dtype=np.float64) for feature_name in feature_columns
    }
    train_features["MPD_S1"] = np.array([5.0, 5.0], dtype=np.float64)
    train_df = pd.DataFrame(train_features)
    train_df["sample_id"] = pd.Series([1, 2], dtype="int64")
    train_df["label"] = pd.Series([0, 1], dtype="int64")

    stats = compute_pipeline_b_stats(x_train_df=train_df)
    normalized_df = apply_pipeline_b_normalization(x_df=train_df, stats=stats)

    assert stats["std"]["MPD_S1"] == pytest.approx(1.0)
    assert np.allclose(normalized_df["MPD_S1"].to_numpy(), train_df["MPD_S1"].to_numpy() - 5.0)
