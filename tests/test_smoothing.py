"""Unit tests for Savitzky-Golay smoothing on gas-channel matrices."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest


def _import_smoothing_symbols() -> tuple[Any, ...]:
    """Import smoothing symbols lazily so pre-implementation failures stay clear."""
    try:
        from src.config import N_GAS_CHANNELS, SG_POLYORDER, SG_WINDOW_LENGTH
        from src.data.smoothing import apply_sg_smoothing
    except (AttributeError, ImportError, ModuleNotFoundError) as exc:
        pytest.fail(f"Smoothing implementation is missing: {exc}", pytrace=False)

    return apply_sg_smoothing, N_GAS_CHANNELS, SG_WINDOW_LENGTH, SG_POLYORDER


def _build_integer_matrix(n_rows: int, n_channels: int, *, offset: int = 100) -> NDArray[np.int64]:
    """Build a deterministic integer matrix for smoothing tests."""
    time_index = np.arange(n_rows, dtype=np.int64).reshape(-1, 1)
    channel_offsets = np.arange(n_channels, dtype=np.int64).reshape(1, -1)
    return offset + (time_index % 17) + channel_offsets


def test_apply_sg_smoothing_returns_float64_with_input_shape() -> None:
    """Smoothing should preserve shape while returning float64 values."""
    apply_sg_smoothing, N_GAS_CHANNELS, _SG_WINDOW_LENGTH, _SG_POLYORDER = _import_smoothing_symbols()
    gas_signals = _build_integer_matrix(n_rows=1000, n_channels=N_GAS_CHANNELS)

    smoothed = apply_sg_smoothing(gas_signals)

    assert smoothed.shape == gas_signals.shape
    assert smoothed.dtype == np.float64


def test_apply_sg_smoothing_preserves_negative_response_sign() -> None:
    """A signed gas-channel trace should retain negative values after smoothing."""
    apply_sg_smoothing, N_GAS_CHANNELS, _SG_WINDOW_LENGTH, _SG_POLYORDER = _import_smoothing_symbols()
    time_axis = np.linspace(0.0, 4.0 * np.pi, 100, dtype=np.float64)
    gas_signals = np.zeros((100, N_GAS_CHANNELS), dtype=np.float64)
    gas_signals[:, 0] = 5.0 * np.sin(time_axis)

    smoothed = apply_sg_smoothing(gas_signals)

    assert np.min(smoothed[:, 0]) < 0.0


def test_apply_sg_smoothing_rejects_wrong_column_count() -> None:
    """Smoothing should reject matrices that do not expose the 11 gas channels."""
    apply_sg_smoothing, _N_GAS_CHANNELS, _SG_WINDOW_LENGTH, _SG_POLYORDER = _import_smoothing_symbols()
    gas_signals = np.zeros((1000, 10), dtype=np.int64)

    with pytest.raises(ValueError, match="shape"):
        apply_sg_smoothing(gas_signals)


def test_apply_sg_smoothing_rejects_one_dimensional_input() -> None:
    """Smoothing should reject one-dimensional inputs."""
    apply_sg_smoothing, _N_GAS_CHANNELS, _SG_WINDOW_LENGTH, _SG_POLYORDER = _import_smoothing_symbols()
    gas_signals = np.arange(1000, dtype=np.int64)

    with pytest.raises(ValueError, match="two-dimensional"):
        apply_sg_smoothing(gas_signals)


def test_apply_sg_smoothing_rejects_short_traces() -> None:
    """Traces shorter than the Savitzky-Golay window should halt defensively."""
    apply_sg_smoothing, N_GAS_CHANNELS, SG_WINDOW_LENGTH, _SG_POLYORDER = _import_smoothing_symbols()
    gas_signals = np.zeros((SG_WINDOW_LENGTH - 1, N_GAS_CHANNELS), dtype=np.int64)

    with pytest.raises(ValueError, match="shorter than"):
        apply_sg_smoothing(gas_signals)


def test_apply_sg_smoothing_reduces_high_frequency_noise() -> None:
    """Smoothing should move a noisy trace closer to the low-frequency truth."""
    apply_sg_smoothing, N_GAS_CHANNELS, _SG_WINDOW_LENGTH, _SG_POLYORDER = _import_smoothing_symbols()
    rng = np.random.default_rng(42)
    n_rows = 1000
    sample_index = np.arange(n_rows, dtype=np.float64)
    signal_truth = np.sin(2.0 * np.pi * sample_index / 200.0)
    high_frequency_noise = 0.6 * np.sin(2.0 * np.pi * sample_index / 5.0)
    random_perturbation = 0.05 * rng.normal(size=n_rows)
    noisy_signal = signal_truth + high_frequency_noise + random_perturbation
    s_truth = np.tile(signal_truth.reshape(-1, 1), (1, N_GAS_CHANNELS))
    noisy = np.tile(noisy_signal.reshape(-1, 1), (1, N_GAS_CHANNELS))

    smoothed = apply_sg_smoothing(noisy)

    assert np.std(smoothed[:, 0] - s_truth[:, 0]) < np.std(noisy[:, 0] - s_truth[:, 0])


def test_apply_sg_smoothing_does_not_mutate_input_matrix() -> None:
    """Smoothing should leave the original raw matrix unchanged."""
    apply_sg_smoothing, N_GAS_CHANNELS, _SG_WINDOW_LENGTH, _SG_POLYORDER = _import_smoothing_symbols()
    gas_signals = _build_integer_matrix(n_rows=1000, n_channels=N_GAS_CHANNELS)
    original = gas_signals.copy()

    _ = apply_sg_smoothing(gas_signals)

    assert np.array_equal(gas_signals, original)


def test_apply_sg_smoothing_filters_each_channel_independently() -> None:
    """Smoothing should not leak structure from one channel into another."""
    apply_sg_smoothing, N_GAS_CHANNELS, _SG_WINDOW_LENGTH, _SG_POLYORDER = _import_smoothing_symbols()
    gas_signals = np.zeros((1000, N_GAS_CHANNELS), dtype=np.float64)
    gas_signals[500:, 10] = 1.0

    smoothed = apply_sg_smoothing(gas_signals)

    assert np.allclose(smoothed[:, :10], 0.0, atol=1e-12)
    assert smoothed[100, 10] == pytest.approx(0.0, abs=1e-12)
    assert smoothed[900, 10] == pytest.approx(1.0, abs=1e-12)
    assert np.any(np.abs(smoothed[:, 10]) > 0.0)


def test_apply_sg_smoothing_logs_info_parameters(caplog: pytest.LogCaptureFixture) -> None:
    """Smoothing should log the matrix shape and filter parameters at INFO level."""
    apply_sg_smoothing, N_GAS_CHANNELS, SG_WINDOW_LENGTH, SG_POLYORDER = _import_smoothing_symbols()
    gas_signals = _build_integer_matrix(n_rows=1000, n_channels=N_GAS_CHANNELS)

    with caplog.at_level(logging.INFO):
        _ = apply_sg_smoothing(gas_signals)

    info_messages = [record.message for record in caplog.records if record.levelno == logging.INFO]
    assert len(info_messages) == 1
    assert "n_rows=1000" in info_messages[0]
    assert f"n_channels={N_GAS_CHANNELS}" in info_messages[0]
    assert f"window_length={SG_WINDOW_LENGTH}" in info_messages[0]
    assert f"polyorder={SG_POLYORDER}" in info_messages[0]
