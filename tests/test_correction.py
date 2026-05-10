"""Unit tests for fractional baseline correction and response trimming."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest

from src.config import RANDOM_SEED


def _import_correction_symbols() -> tuple[Any, ...]:
    """Import correction symbols lazily so pre-implementation failures stay clear."""
    try:
        from src.config import N_GAS_CHANNELS
        from src.data.correction import TrimResult, apply_fractional_correction, trim_response_segment
    except (AttributeError, ImportError, ModuleNotFoundError) as exc:
        pytest.fail(f"Correction implementation is missing: {exc}", pytrace=False)

    return TrimResult, apply_fractional_correction, trim_response_segment, N_GAS_CHANNELS


def _build_compensated_matrix(
    *,
    n_rows: int,
    n_gas_channels: int,
    baseline_values: NDArray[np.float64],
    t_baseline_end: int,
) -> NDArray[np.float64]:
    """Construct a deterministic compensated matrix with signed post-baseline dynamics."""
    rng = np.random.default_rng(RANDOM_SEED)
    matrix = np.tile(baseline_values.reshape(1, -1), (n_rows, 1)).astype(np.float64, copy=True)
    response_offsets = rng.normal(loc=0.0, scale=0.5, size=(n_rows - t_baseline_end, n_gas_channels))
    response_offsets[:, 0] -= np.linspace(0.5, 2.5, n_rows - t_baseline_end)
    response_offsets[:, 1] += np.linspace(0.25, 2.25, n_rows - t_baseline_end)
    matrix[t_baseline_end:, :] += response_offsets
    return matrix


def test_apply_fractional_correction_matches_known_fractional_values() -> None:
    """Fractional correction should match the analytical `(x - S0) / S0` formula."""
    _TrimResult, apply_fractional_correction, _trim_response_segment, n_gas_channels = _import_correction_symbols()
    t_baseline_end = 2
    baseline_values = np.arange(20.0, 20.0 + n_gas_channels, dtype=np.float64)
    compensated = np.tile(baseline_values.reshape(1, -1), (5, 1))
    compensated[2:, :] += np.array([0.0, 2.0, -2.0]).reshape(-1, 1)

    corrected = apply_fractional_correction(compensated_gas=compensated, t_baseline_end=t_baseline_end)

    expected = (compensated - baseline_values.reshape(1, -1)) / baseline_values.reshape(1, -1)
    assert np.allclose(corrected, expected)


def test_apply_fractional_correction_uses_denominator_guard_and_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Near-zero baseline channels should warn and use the 1e-8 denominator guard."""
    _TrimResult, apply_fractional_correction, _trim_response_segment, n_gas_channels = _import_correction_symbols()
    compensated = np.full((4, n_gas_channels), 20.0, dtype=np.float64)
    compensated[:2, 0] = 0.0
    compensated[2:, 0] = 2.0
    compensated[:2, 1] = 5.0
    compensated[2:, 1] = 7.5

    with caplog.at_level(logging.WARNING):
        corrected = apply_fractional_correction(compensated_gas=compensated, t_baseline_end=2)

    warning_messages = [record.message for record in caplog.records if record.levelno == logging.WARNING]
    assert corrected[2, 0] == pytest.approx((2.0 - 0.0) / 1e-8)
    assert any("10" in message for message in warning_messages)


def test_apply_fractional_correction_preserves_negative_values_and_input_immutability() -> None:
    """Correction should keep signed responses and avoid mutating the compensated input."""
    _TrimResult, apply_fractional_correction, _trim_response_segment, n_gas_channels = _import_correction_symbols()
    baseline_values = np.arange(30.0, 30.0 + n_gas_channels, dtype=np.float64)
    compensated = _build_compensated_matrix(
        n_rows=8,
        n_gas_channels=n_gas_channels,
        baseline_values=baseline_values,
        t_baseline_end=3,
    )
    original = compensated.copy()

    corrected = apply_fractional_correction(compensated_gas=compensated, t_baseline_end=3)

    assert np.array_equal(compensated, original)
    assert np.min(corrected[:, 0]) < 0.0
    assert np.max(corrected[:, 1]) > 0.0


def test_trim_response_segment_returns_exact_clip_when_enough_samples_are_available() -> None:
    """Enough response samples should trim directly without padding or deficit metadata."""
    TrimResult, _apply_fractional_correction, trim_response_segment, n_gas_channels = _import_correction_symbols()
    corrected = np.arange(10 * n_gas_channels, dtype=np.float64).reshape(10, n_gas_channels)
    original = corrected.copy()

    result = trim_response_segment(
        corrected_gas=corrected,
        t_baseline_end=3,
        clip_length_L=4,
        padding_relative_deficit_threshold=0.2,
    )

    assert isinstance(result, TrimResult)
    assert result.trimmed is not None
    assert result.trimmed.shape == (4, n_gas_channels)
    assert np.array_equal(result.trimmed, corrected[3:7, :])
    assert result.padded is False
    assert result.clip_length_pass is True
    assert result.relative_deficit is None
    assert result.n_available == 7
    assert np.array_equal(corrected, original)


def test_trim_response_segment_pads_when_relative_deficit_is_within_threshold() -> None:
    """Short responses should tail-pad when the deficit does not exceed the threshold."""
    _TrimResult, _apply_fractional_correction, trim_response_segment, n_gas_channels = _import_correction_symbols()
    corrected = np.arange(6 * n_gas_channels, dtype=np.float64).reshape(6, n_gas_channels)

    result = trim_response_segment(
        corrected_gas=corrected,
        t_baseline_end=3,
        clip_length_L=4,
        padding_relative_deficit_threshold=0.25,
    )

    assert result.trimmed is not None
    assert result.trimmed.shape == (4, n_gas_channels)
    assert np.array_equal(result.trimmed[:3, :], corrected[3:, :])
    assert np.array_equal(result.trimmed[3, :], corrected[-1, :])
    assert result.padded is True
    assert result.clip_length_pass is True
    assert result.relative_deficit == pytest.approx(0.25)
    assert result.n_available == 3


def test_trim_response_segment_fails_when_relative_deficit_exceeds_threshold() -> None:
    """Short responses beyond the padding tolerance should fail the clip-length check."""
    _TrimResult, _apply_fractional_correction, trim_response_segment, n_gas_channels = _import_correction_symbols()
    corrected = np.arange(6 * n_gas_channels, dtype=np.float64).reshape(6, n_gas_channels)

    result = trim_response_segment(
        corrected_gas=corrected,
        t_baseline_end=3,
        clip_length_L=5,
        padding_relative_deficit_threshold=0.2,
    )

    assert result.trimmed is None
    assert result.padded is False
    assert result.clip_length_pass is False
    assert result.relative_deficit == pytest.approx(0.4)
    assert result.n_available == 3


def test_trim_response_segment_fails_without_padding_threshold_for_short_responses() -> None:
    """A null padding threshold should reject short responses instead of padding them."""
    _TrimResult, _apply_fractional_correction, trim_response_segment, n_gas_channels = _import_correction_symbols()
    corrected = np.arange(6 * n_gas_channels, dtype=np.float64).reshape(6, n_gas_channels)

    result = trim_response_segment(
        corrected_gas=corrected,
        t_baseline_end=3,
        clip_length_L=5,
        padding_relative_deficit_threshold=None,
    )

    assert result.trimmed is None
    assert result.padded is False
    assert result.clip_length_pass is False
    assert result.relative_deficit == pytest.approx(0.4)
    assert result.n_available == 3
