"""Unit tests for K-of-N baseline onset detection."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest

from src.data.loader import RawRecord


def _import_baseline_symbols() -> tuple[Any, ...]:
    """Import baseline symbols lazily so pre-implementation failures stay clear."""
    try:
        from src.config import (
            BASELINE_CONSECUTIVE_EXCEED,
            BASELINE_K_OF_N,
            T_BASELINE_END_MAX,
            T_BASELINE_END_MIN,
        )
        from src.data.baseline import (
            BaselineDetectionResult,
            detect_baseline_end,
            detect_per_channel_onset,
        )
    except (AttributeError, ImportError, ModuleNotFoundError) as exc:
        pytest.fail(f"Baseline detection implementation is missing: {exc}", pytrace=False)

    return (
        BaselineDetectionResult,
        detect_baseline_end,
        detect_per_channel_onset,
        BASELINE_CONSECUTIVE_EXCEED,
        BASELINE_K_OF_N,
        T_BASELINE_END_MAX,
        T_BASELINE_END_MIN,
    )


def _build_channel_signal(
    n_rows: int,
    onset: int | None,
    *,
    response_value: int = 30,
) -> NDArray[np.int64]:
    """Build one deterministic gas-channel trace with an optional step onset."""
    signal = np.tile(np.array([10, 11], dtype=np.int64), int(np.ceil(n_rows / 2)))[:n_rows]
    signal[:5] = np.array([10, 11, 10, 11, 10], dtype=np.int64)
    if onset is not None:
        signal[onset:] = response_value
    return signal


def _build_raw_record(
    onsets: list[int | None],
    *,
    n_rows: int = 100,
    response_values: list[int] | None = None,
) -> RawRecord:
    """Construct a synthetic raw record directly from deterministic arrays."""
    values = [30] * len(onsets) if response_values is None else response_values
    gas_signals = np.column_stack(
        [
            _build_channel_signal(n_rows=n_rows, onset=onset, response_value=response_value)
            for onset, response_value in zip(onsets, values, strict=True)
        ]
    )
    base = np.arange(n_rows, dtype=np.int64)
    return RawRecord(
        sample_id=101,
        filename="synthetic.csv",
        n_rows=n_rows,
        s0=base.copy(),
        gas_signals=gas_signals,
        temperature=base.copy() + 100,
        humidity=base.copy() + 200,
        label=1,
        disease_category=4,
        collection_time="2026-01-01 08:00:00",
    )


def test_detect_per_channel_onset_returns_first_step_sample() -> None:
    """Per-channel onset should be the first sample of the sustained threshold run."""
    (
        _BaselineDetectionResult,
        _detect_baseline_end,
        detect_per_channel_onset,
        _BASELINE_CONSECUTIVE_EXCEED,
        _BASELINE_K_OF_N,
        _T_BASELINE_END_MAX,
        T_BASELINE_END_MIN,
    ) = _import_baseline_symbols()
    signal = _build_channel_signal(n_rows=20, onset=10)

    onset = detect_per_channel_onset(signal, (T_BASELINE_END_MIN, 20))

    assert onset == 10


def test_detect_per_channel_onset_returns_none_for_small_deviations() -> None:
    """Per-channel onset should remain absent when deviations stay below threshold."""
    (
        _BaselineDetectionResult,
        _detect_baseline_end,
        detect_per_channel_onset,
        _BASELINE_CONSECUTIVE_EXCEED,
        _BASELINE_K_OF_N,
        T_BASELINE_END_MAX,
        T_BASELINE_END_MIN,
    ) = _import_baseline_symbols()
    rng = np.random.default_rng(42)
    signal = np.full(1000, 10.0, dtype=float)
    signal[:5] = np.array([10.0, 11.0, 10.0, 11.0, 10.0], dtype=float)
    signal[5:] = 10.4 + rng.uniform(-0.5, 0.5, size=995)

    onset = detect_per_channel_onset(signal, (T_BASELINE_END_MIN, T_BASELINE_END_MAX))

    assert onset is None


def test_detect_per_channel_onset_accepts_run_starting_at_t_min() -> None:
    """A qualifying run that starts at the minimum search index should count."""
    (
        _BaselineDetectionResult,
        _detect_baseline_end,
        detect_per_channel_onset,
        _BASELINE_CONSECUTIVE_EXCEED,
        _BASELINE_K_OF_N,
        _T_BASELINE_END_MAX,
        T_BASELINE_END_MIN,
    ) = _import_baseline_symbols()
    signal = _build_channel_signal(n_rows=20, onset=T_BASELINE_END_MIN)

    onset = detect_per_channel_onset(signal, (T_BASELINE_END_MIN, 20))

    assert onset == T_BASELINE_END_MIN


def test_detect_per_channel_onset_accepts_run_that_ends_at_t_max_minus_one() -> None:
    """A run that fully fits inside the half-open search window should count."""
    (
        _BaselineDetectionResult,
        _detect_baseline_end,
        detect_per_channel_onset,
        BASELINE_CONSECUTIVE_EXCEED,
        _BASELINE_K_OF_N,
        T_BASELINE_END_MAX,
        T_BASELINE_END_MIN,
    ) = _import_baseline_symbols()
    onset_index = T_BASELINE_END_MAX - BASELINE_CONSECUTIVE_EXCEED
    signal = _build_channel_signal(n_rows=T_BASELINE_END_MAX + 5, onset=onset_index)

    onset = detect_per_channel_onset(signal, (T_BASELINE_END_MIN, T_BASELINE_END_MAX))

    assert onset == onset_index


def test_detect_per_channel_onset_rejects_run_that_spills_past_t_max() -> None:
    """A run that extends beyond the half-open search window should not count."""
    (
        _BaselineDetectionResult,
        _detect_baseline_end,
        detect_per_channel_onset,
        BASELINE_CONSECUTIVE_EXCEED,
        _BASELINE_K_OF_N,
        T_BASELINE_END_MAX,
        T_BASELINE_END_MIN,
    ) = _import_baseline_symbols()
    onset_index = T_BASELINE_END_MAX - BASELINE_CONSECUTIVE_EXCEED + 1
    signal = _build_channel_signal(n_rows=T_BASELINE_END_MAX + 5, onset=onset_index)

    onset = detect_per_channel_onset(signal, (T_BASELINE_END_MIN, T_BASELINE_END_MAX))

    assert onset is None


def test_detect_per_channel_onset_clamps_degenerate_sigma_and_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Degenerate baseline variance should clamp sigma and still detect onset."""
    (
        _BaselineDetectionResult,
        _detect_baseline_end,
        detect_per_channel_onset,
        _BASELINE_CONSECUTIVE_EXCEED,
        _BASELINE_K_OF_N,
        _T_BASELINE_END_MAX,
        T_BASELINE_END_MIN,
    ) = _import_baseline_symbols()
    signal = np.full(20, 5.0, dtype=float)
    signal[10:] = 12.0

    with caplog.at_level(logging.WARNING):
        onset = detect_per_channel_onset(signal, (T_BASELINE_END_MIN, 20))

    warning_records = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert onset == 10
    assert len(warning_records) == 1
    assert "sigma < 1e-8" in warning_records[0].message


def test_detect_baseline_end_uses_fourth_smallest_trigger_for_exact_k_channels() -> None:
    """Sample-level onset should equal the Kth triggered onset when exactly K channels fire."""
    (
        BaselineDetectionResult,
        detect_baseline_end,
        _detect_per_channel_onset,
        _BASELINE_CONSECUTIVE_EXCEED,
        BASELINE_K_OF_N,
        _T_BASELINE_END_MAX,
        _T_BASELINE_END_MIN,
    ) = _import_baseline_symbols()
    trigger_onsets = [7, 9, 11, 13]
    record = _build_raw_record(onsets=trigger_onsets + [None] * 7)

    result = detect_baseline_end(record)

    assert isinstance(result, BaselineDetectionResult)
    assert result.t_baseline_end == 13
    assert result.n_triggered_channels == BASELINE_K_OF_N
    assert result.fallback_used is False
    assert result.per_channel_onsets.tolist() == [7, 9, 11, 13] + [-1] * 7


def test_detect_baseline_end_returns_none_when_only_k_minus_one_channels_trigger() -> None:
    """Sample-level onset should fall back when fewer than K channels trigger."""
    (
        _BaselineDetectionResult,
        detect_baseline_end,
        _detect_per_channel_onset,
        _BASELINE_CONSECUTIVE_EXCEED,
        BASELINE_K_OF_N,
        _T_BASELINE_END_MAX,
        _T_BASELINE_END_MIN,
    ) = _import_baseline_symbols()
    record = _build_raw_record(onsets=[7, 9, 11] + [None] * 8)

    result = detect_baseline_end(record)

    assert result.t_baseline_end is None
    assert result.fallback_used is True
    assert result.n_triggered_channels == BASELINE_K_OF_N - 1


def test_detect_baseline_end_uses_fourth_smallest_trigger_when_all_channels_fire() -> None:
    """Distinct onsets across all channels should still pick the fourth-smallest index."""
    (
        _BaselineDetectionResult,
        detect_baseline_end,
        _detect_per_channel_onset,
        _BASELINE_CONSECUTIVE_EXCEED,
        _BASELINE_K_OF_N,
        _T_BASELINE_END_MAX,
        _T_BASELINE_END_MIN,
    ) = _import_baseline_symbols()
    onsets = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    record = _build_raw_record(onsets=onsets)

    result = detect_baseline_end(record)

    assert result.t_baseline_end == sorted(onsets)[3]
    assert result.n_triggered_channels == 11
    assert result.fallback_used is False


def test_detect_baseline_end_preserves_signal_sign_and_input_array() -> None:
    """Negative responses should trigger without mutating the raw gas-signal array."""
    (
        _BaselineDetectionResult,
        detect_baseline_end,
        _detect_per_channel_onset,
        _BASELINE_CONSECUTIVE_EXCEED,
        _BASELINE_K_OF_N,
        _T_BASELINE_END_MAX,
        _T_BASELINE_END_MIN,
    ) = _import_baseline_symbols()
    record = _build_raw_record(
        onsets=[8, 9, 10, 11] + [None] * 7,
        response_values=[-10, -12, -14, -16] + [30] * 7,
    )
    original_gas_signals = record.gas_signals.copy()

    result = detect_baseline_end(record)

    assert result.t_baseline_end == 11
    assert result.per_channel_onsets[:4].tolist() == [8, 9, 10, 11]
    assert np.array_equal(record.gas_signals, original_gas_signals)
