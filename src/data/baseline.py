"""K-of-N baseline onset detection for raw gas-sensor traces."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.config import (
    BASELINE_CONSECUTIVE_EXCEED,
    BASELINE_K_OF_N,
    BASELINE_SIGMA_MULTIPLIER,
    N_GAS_CHANNELS,
    T_BASELINE_END_MAX,
    T_BASELINE_END_MIN,
    T_BASELINE_WINDOW_INIT,
)
from src.data.loader import RawRecord


LOGGER = logging.getLogger(__name__)

_NO_ONSET_SENTINEL = -1


@dataclass(frozen=True)
class BaselineDetectionResult:
    """Baseline onset-detection outputs for one sample.

    The `per_channel_onsets` array stores one onset index for each gas channel
    `S1` through `S11`. Channels that do not trigger inside the search window
    use the sentinel value `-1`.

    Attributes:
        t_baseline_end: Pass-1 triggered onset index, or `None` when fewer than
            `BASELINE_K_OF_N` channels trigger in the search window.
        fallback_used: Whether the cohort-level fallback median must be applied
            later by the orchestrator.
        per_channel_onsets: Per-channel onset indices with shape `(11,)` and the
            sentinel value `-1` for non-triggered channels.
        n_triggered_channels: Count of channels with a non-sentinel onset.
    """

    t_baseline_end: int | None
    fallback_used: bool
    per_channel_onsets: NDArray[np.int64]
    n_triggered_channels: int


def detect_per_channel_onset(
    channel_signal: NDArray[np.number[Any]], search_window: tuple[int, int]
) -> int | None:
    """Detect the first sustained per-channel onset inside a half-open search window.

    Onset definition (binding for this codebase): the per-channel onset is the
    index of the FIRST sample of the qualifying consecutive run. Concretely, the
    function returns the smallest index t_first such that t_first is in [t_min,
    t_max - BASELINE_CONSECUTIVE_EXCEED + 1) and abs(channel_signal[t_first +
    k] - mu) > BASELINE_SIGMA_MULTIPLIER * sigma for all k in {0, 1, ...,
    BASELINE_CONSECUTIVE_EXCEED - 1}. Rationale: the more common
    signal-processing convention treats onset as the moment the deviation first
    crosses the threshold and stays crossed; this also satisfies Section 3.3's binding
    sentence "the smallest t such that at least K gas channels have a
    per-channel onset at or before t" because the smallest first-sample index
    across channels is by construction also the smallest "by t" reference.

    Args:
        channel_signal: One-dimensional raw signal for one gas channel.
        search_window: Half-open `(t_min, t_max)` search interval.

    Returns:
        The onset index as an `int`, or `None` if no qualifying sustained run is
        found entirely inside the search window.

    Raises:
        ValueError: If the input signal is not one-dimensional, is shorter than
            the baseline window, or the search window is invalid.
    """
    signal = np.asarray(channel_signal)
    if signal.ndim != 1:
        raise ValueError(f"channel_signal must be one-dimensional; got shape {signal.shape}")
    if signal.shape[0] < T_BASELINE_WINDOW_INIT:
        raise ValueError(
            "channel_signal must contain at least "
            f"{T_BASELINE_WINDOW_INIT} samples for baseline statistics; got {signal.shape[0]}"
        )

    t_min, t_max = search_window
    if t_min < 0 or t_max < t_min or t_max > signal.shape[0]:
        raise ValueError(
            "search_window must satisfy 0 <= t_min <= t_max <= len(channel_signal); "
            f"got {search_window} for signal length {signal.shape[0]}"
        )

    signal_float = np.asarray(signal, dtype=float)
    baseline_window = signal_float[:T_BASELINE_WINDOW_INIT]
    mu = float(np.mean(baseline_window))
    sigma = float(np.std(baseline_window))
    if sigma < 1e-8:
        LOGGER.warning("Clamping baseline sigma < 1e-8 to 1.0 for per-channel onset detection.")
        sigma = 1.0

    threshold = BASELINE_SIGMA_MULTIPLIER * sigma
    last_valid_start = t_max - BASELINE_CONSECUTIVE_EXCEED
    for t_first in range(t_min, last_valid_start + 1):
        deviation_window = np.abs(signal_float[t_first : t_first + BASELINE_CONSECUTIVE_EXCEED] - mu)
        if np.all(deviation_window > threshold):
            return int(t_first)

    return None


def detect_baseline_end(record: RawRecord) -> BaselineDetectionResult:
    """Detect the sample-level baseline end using K-of-N voting across gas channels.

    Args:
        record: Raw breath-trace record whose `gas_signals` matrix contains the
            raw `S1` through `S11` gas channels column-wise.

    Returns:
        A `BaselineDetectionResult` with per-channel onset indices, K-of-N
        sample-level onset, and the downstream fallback flag.

    Raises:
        ValueError: If the record is too short for onset detection or its
            `gas_signals` matrix does not match the expected `(n_rows, 11)`
            shape.
    """
    if record.n_rows < T_BASELINE_END_MIN + 1:
        raise ValueError(
            "RawRecord is too short for baseline detection: "
            f"n_rows={record.n_rows} but at least {T_BASELINE_END_MIN + 1} rows are required."
        )
    expected_shape = (record.n_rows, N_GAS_CHANNELS)
    if record.gas_signals.shape != expected_shape:
        raise ValueError(
            "RawRecord gas_signals shape mismatch: "
            f"expected {expected_shape}, got {record.gas_signals.shape}"
        )

    search_window = (T_BASELINE_END_MIN, min(T_BASELINE_END_MAX, record.n_rows))
    per_channel_onsets = np.full(N_GAS_CHANNELS, _NO_ONSET_SENTINEL, dtype=np.int64)
    for channel_index in range(N_GAS_CHANNELS):
        onset = detect_per_channel_onset(record.gas_signals[:, channel_index], search_window)
        if onset is not None:
            per_channel_onsets[channel_index] = np.int64(onset)

    triggered_onsets = per_channel_onsets[per_channel_onsets != _NO_ONSET_SENTINEL]
    n_triggered_channels = int(triggered_onsets.shape[0])
    t_baseline_end: int | None = None
    if n_triggered_channels >= BASELINE_K_OF_N:
        sorted_onsets = np.sort(triggered_onsets)
        t_baseline_end = int(sorted_onsets[BASELINE_K_OF_N - 1])

    fallback_used = t_baseline_end is None
    LOGGER.info(
        "Detected baseline end for sample_id=%s n_triggered_channels=%s t_baseline_end=%s",
        record.sample_id,
        n_triggered_channels,
        "fallback" if t_baseline_end is None else t_baseline_end,
    )
    return BaselineDetectionResult(
        t_baseline_end=t_baseline_end,
        fallback_used=fallback_used,
        per_channel_onsets=per_channel_onsets,
        n_triggered_channels=n_triggered_channels,
    )


__all__ = ["BaselineDetectionResult", "detect_per_channel_onset", "detect_baseline_end"]
