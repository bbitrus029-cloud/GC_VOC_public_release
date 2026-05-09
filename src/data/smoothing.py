"""Savitzky-Golay smoothing for gas-channel matrices."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_filter

from src.config import N_GAS_CHANNELS, SG_POLYORDER, SG_WINDOW_LENGTH


LOGGER = logging.getLogger(__name__)


def apply_sg_smoothing(gas_signals: NDArray[np.number[Any]]) -> NDArray[np.float64]:
    """Apply Savitzky-Golay smoothing independently to each gas channel.

    The filter is applied column-wise to the raw `S1` through `S11` gas-signal
    matrix. Because Savitzky-Golay smoothing is a linear filter, it preserves the
    sign of the underlying response rather than rectifying or taking absolute
    values, which satisfies Pillar 2 response-direction preservation.

    The short-trace branch is a defensive contract retained from the locked plan.
    It is unreachable on real cohort data under the current specification because
    QC-2 enforces a `1000`-row minimum while `SG_WINDOW_LENGTH = 11`, but the
    guard remains in place in case a future erratum lowers `MIN_ROWS_QC2` below
    the smoothing window length.

    Args:
        gas_signals: Two-dimensional raw gas-signal matrix with shape
            `(n_rows, 11)` for channels `S1` through `S11`.

    Returns:
        A smoothed gas-signal matrix with shape `(n_rows, 11)` and
        `numpy.float64` dtype.

    Raises:
        ValueError: If `gas_signals` is not two-dimensional, does not expose the
            expected gas-channel count, or is shorter than `SG_WINDOW_LENGTH`.
    """
    gas_signal_matrix = np.asarray(gas_signals)
    if gas_signal_matrix.ndim != 2:
        raise ValueError(f"gas_signals must be a two-dimensional array; got shape {gas_signal_matrix.shape}")
    if gas_signal_matrix.shape[1] != N_GAS_CHANNELS:
        raise ValueError(
            "gas_signals shape mismatch: "
            f"expected second dimension {N_GAS_CHANNELS}, got shape {gas_signal_matrix.shape}"
        )
    if gas_signal_matrix.shape[0] < SG_WINDOW_LENGTH:
        raise ValueError(
            "gas_signals is shorter than the Savitzky-Golay window length: "
            f"n_rows={gas_signal_matrix.shape[0]} window_length={SG_WINDOW_LENGTH}"
        )

    gas_signal_matrix_float64 = np.array(gas_signal_matrix, dtype=np.float64, copy=True)
    LOGGER.info(
        "Applying Savitzky-Golay smoothing n_rows=%s n_channels=%s window_length=%s polyorder=%s",
        gas_signal_matrix_float64.shape[0],
        gas_signal_matrix_float64.shape[1],
        SG_WINDOW_LENGTH,
        SG_POLYORDER,
    )
    smoothed_signals = savgol_filter(
        gas_signal_matrix_float64,
        window_length=SG_WINDOW_LENGTH,
        polyorder=SG_POLYORDER,
        axis=0,
        mode="interp",
    )
    return np.asarray(smoothed_signals, dtype=np.float64)


__all__ = ["apply_sg_smoothing"]
