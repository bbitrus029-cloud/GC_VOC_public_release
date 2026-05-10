"""Fractional baseline correction and response trimming utilities."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.config import N_GAS_CHANNELS


LOGGER = logging.getLogger(__name__)
_DENOMINATOR_GUARD = 1e-8
_LOW_BASELINE_WARNING_THRESHOLD = 10.0


def _coerce_gas_matrix(values: NDArray[np.number[Any]], name: str) -> NDArray[np.float64]:
    """Validate and coerce a gas-channel matrix to float64."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional; got shape {matrix.shape}")
    if matrix.shape[1] != N_GAS_CHANNELS:
        raise ValueError(f"{name} must have shape (n_rows, {N_GAS_CHANNELS}); got {matrix.shape}")
    if matrix.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _validate_t_baseline_end(t_baseline_end: int, n_rows: int) -> int:
    """Validate the half-open baseline interval boundary."""
    baseline_end = int(t_baseline_end)
    if baseline_end <= 0 or baseline_end >= n_rows:
        raise ValueError(f"t_baseline_end must satisfy 0 < t_baseline_end < {n_rows}; got {baseline_end}")
    return baseline_end


def _validate_clip_length(clip_length_l: int) -> int:
    """Validate the requested trimmed clip length."""
    clip_length = int(clip_length_l)
    if clip_length <= 0:
        raise ValueError(f"clip_length_L must be positive; got {clip_length}")
    return clip_length


def _validate_padding_threshold(threshold: float | None) -> float | None:
    """Validate the optional relative-deficit padding threshold."""
    if threshold is None:
        return None
    threshold_value = float(threshold)
    if not np.isfinite(threshold_value):
        raise ValueError("padding_relative_deficit_threshold must be finite when provided.")
    if threshold_value < 0.0:
        raise ValueError("padding_relative_deficit_threshold must be non-negative when provided.")
    return threshold_value


def _guarded_denominator(baseline_values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Clamp near-zero denominators while preserving sign when available."""
    guarded = np.array(baseline_values, dtype=np.float64, copy=True)
    near_zero_mask = np.abs(guarded) < _DENOMINATOR_GUARD
    signs = np.where(guarded < 0.0, -1.0, 1.0)
    guarded[near_zero_mask] = signs[near_zero_mask] * _DENOMINATOR_GUARD
    return guarded


@dataclass(frozen=True)
class TrimResult:
    """Outputs of the prespecified response-trimming step.

    Attributes:
        trimmed: The trimmed or padded response matrix, or `None` when the sample
            fails the clip-length rule.
        padded: Whether tail padding was applied.
        clip_length_pass: Whether the response passed the clip-length rule.
        relative_deficit: Relative shortfall `(L - n_available) / L` for short
            responses, else `None`.
        n_available: Available response rows before trimming or padding.
    """

    trimmed: NDArray[np.float64] | None
    padded: bool
    clip_length_pass: bool
    relative_deficit: float | None
    n_available: int


def apply_fractional_correction(
    compensated_gas: NDArray[np.number[Any]],
    t_baseline_end: int,
) -> NDArray[np.float64]:
    """Apply fractional baseline correction channel-wise.

    Args:
        compensated_gas: Compensated gas-channel matrix with shape `(n_rows, 11)`.
        t_baseline_end: Half-open baseline boundary, where `[0, t_baseline_end)`
            is baseline and `[t_baseline_end, n_rows)` is response.

    Returns:
        A corrected gas-channel matrix with the same shape and `numpy.float64`
        dtype.

    Raises:
        ValueError: If the gas matrix is mis-shaped, non-finite, or the baseline
            interval is invalid.
    """
    compensated_matrix = _coerce_gas_matrix(compensated_gas, "compensated_gas")
    baseline_end = _validate_t_baseline_end(t_baseline_end, compensated_matrix.shape[0])

    baseline_values = np.mean(compensated_matrix[:baseline_end, :], axis=0)
    for channel_index, baseline_value in enumerate(baseline_values):
        if abs(float(baseline_value)) < _LOW_BASELINE_WARNING_THRESHOLD:
            LOGGER.warning(
                "Channel index %s has abs(S0_i) below %.0f ADC units during fractional correction.",
                channel_index,
                _LOW_BASELINE_WARNING_THRESHOLD,
            )

    denominator = _guarded_denominator(baseline_values)
    corrected = (compensated_matrix - baseline_values.reshape(1, -1)) / denominator.reshape(1, -1)
    return np.asarray(corrected, dtype=np.float64)


def trim_response_segment(
    corrected_gas: NDArray[np.number[Any]],
    t_baseline_end: int,
    clip_length_L: int,
    padding_relative_deficit_threshold: float | None,
) -> TrimResult:
    """Trim or pad one corrected response segment to the shared clip length.

    Args:
        corrected_gas: Corrected gas-channel matrix with shape `(n_rows, 11)`.
        t_baseline_end: Half-open response start index.
        clip_length_L: Target response length `L`.
        padding_relative_deficit_threshold: Optional relative-deficit threshold
            for tail padding.

    Returns:
        A `TrimResult` describing whether trimming succeeded, whether padding was
        required, and the resulting matrix when available.

    Raises:
        ValueError: If the gas matrix is mis-shaped, the baseline boundary is
            invalid, or the clip parameters are invalid.
    """
    corrected_matrix = _coerce_gas_matrix(corrected_gas, "corrected_gas")
    baseline_end = _validate_t_baseline_end(t_baseline_end, corrected_matrix.shape[0])
    clip_length = _validate_clip_length(clip_length_L)
    deficit_threshold = _validate_padding_threshold(padding_relative_deficit_threshold)

    response_matrix = corrected_matrix[baseline_end:, :]
    n_available = int(response_matrix.shape[0])
    if n_available >= clip_length:
        return TrimResult(
            trimmed=np.array(response_matrix[:clip_length, :], copy=True, dtype=np.float64),
            padded=False,
            clip_length_pass=True,
            relative_deficit=None,
            n_available=n_available,
        )

    relative_deficit = float((clip_length - n_available) / clip_length)
    if deficit_threshold is None or n_available == 0 or relative_deficit > deficit_threshold:
        return TrimResult(
            trimmed=None,
            padded=False,
            clip_length_pass=False,
            relative_deficit=relative_deficit,
            n_available=n_available,
        )

    padding_rows = np.repeat(response_matrix[-1:, :], clip_length - n_available, axis=0)
    trimmed = np.vstack([response_matrix, padding_rows])
    return TrimResult(
        trimmed=np.asarray(trimmed, dtype=np.float64),
        padded=True,
        clip_length_pass=True,
        relative_deficit=relative_deficit,
        n_available=n_available,
    )


__all__ = ["TrimResult", "apply_fractional_correction", "trim_response_segment"]
