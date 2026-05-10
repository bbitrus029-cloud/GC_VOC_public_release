"""Pipeline B engineered-feature extraction and normalization utilities."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from src.config import GAS_CHANNELS, N_GAS_CHANNELS


LOGGER = logging.getLogger(__name__)

FEATURE_FAMILIES = ("MPD", "MND", "AUC", "Mean", "MAS", "MDS", "T90", "ERR", "LMR")
EARLY_RESPONSE_FRACTION = 0.20
LATE_RESPONSE_FRACTION = 0.50
T90_THRESHOLD_FRACTION = 0.90
ERR_DENOMINATOR_GUARD = 1e-8


def _validate_channel_names(channel_names: tuple[str, ...]) -> tuple[str, ...]:
    """Validate the ordered gas-channel names used for feature labeling."""
    if len(channel_names) != N_GAS_CHANNELS:
        raise ValueError(f"channel_names must contain {N_GAS_CHANNELS} names; got {len(channel_names)}")
    if tuple(channel_names) != tuple(GAS_CHANNELS):
        raise ValueError(f"channel_names must match GAS_CHANNELS exactly; got {channel_names}")
    return tuple(channel_names)


def _coerce_traceability_array(values: list[int] | NDArray[np.integer[Any]], name: str) -> NDArray[np.int64]:
    """Validate one aligned integer traceability vector."""
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}")
    return array


def _validate_trimmed_signal(
    signal: NDArray[np.float64],
    *,
    expected_length: int | None,
    signal_index: int,
) -> NDArray[np.float64]:
    """Validate one corrected-and-trimmed gas-signal matrix."""
    matrix = np.asarray(signal)
    if matrix.dtype != np.float64:
        raise ValueError(f"corrected_trimmed_signals[{signal_index}] must have dtype float64; got {matrix.dtype}")
    if matrix.ndim != 2:
        raise ValueError(
            f"corrected_trimmed_signals[{signal_index}] must be two-dimensional; got shape {matrix.shape}"
        )
    if matrix.shape[1] != N_GAS_CHANNELS:
        raise ValueError(
            f"corrected_trimmed_signals[{signal_index}] must have shape "
            f"(L, {N_GAS_CHANNELS}); got {matrix.shape}"
        )
    if expected_length is not None and matrix.shape[0] != expected_length:
        raise ValueError(
            "All corrected_trimmed_signals must share the same clip length; "
            f"expected {expected_length}, got {matrix.shape[0]} at index {signal_index}"
        )
    return matrix


def _feature_columns(channel_names: tuple[str, ...]) -> list[str]:
    """Return the canonical family-then-channel feature column order."""
    return [f"{family}_{channel_name}" for family in FEATURE_FAMILIES for channel_name in channel_names]


def _extract_signal_features(signal: NDArray[np.float64]) -> dict[str, float]:
    """Compute the nine prespecified response-curve features for one channel."""
    clip_length = int(signal.shape[0])
    diffs = np.diff(signal)
    peak_abs = float(np.max(np.abs(signal)))
    early_window = int(math.ceil(clip_length * EARLY_RESPONSE_FRACTION))
    late_window = int(math.ceil(clip_length * LATE_RESPONSE_FRACTION))
    signal_mean = float(np.mean(signal))

    if peak_abs == 0.0:
        t90 = float(clip_length)
    else:
        threshold = T90_THRESHOLD_FRACTION * peak_abs
        above_threshold = np.flatnonzero(np.abs(signal) >= threshold)
        t90 = float(clip_length if above_threshold.size == 0 else int(above_threshold[0]))

    if abs(signal_mean) < ERR_DENOMINATOR_GUARD:
        err = 0.0
    else:
        err = float(np.mean(signal[:early_window]) / signal_mean)

    if diffs.size == 0:
        mas = 0.0
        mds = 0.0
    else:
        mas = float(np.max(diffs))
        mds = float(np.min(diffs))

    return {
        "MPD": float(np.max(signal)) if np.any(signal > 0.0) else 0.0,
        "MND": float(np.min(signal)) if np.any(signal < 0.0) else 0.0,
        "AUC": float(np.trapezoid(signal)),
        "Mean": signal_mean,
        "MAS": mas,
        "MDS": mds,
        "T90": t90,
        "ERR": err,
        "LMR": float(np.mean(signal[-late_window:])),
    }


def extract_pipeline_b_features(
    corrected_trimmed_signals: list[NDArray[np.float64]],
    sample_ids: list[int] | NDArray[np.integer[Any]],
    labels: list[int] | NDArray[np.integer[Any]],
    channel_names: tuple[str, ...],
) -> pd.DataFrame:
    """Extract Pipeline B features in family-then-channel order.

    The output columns are ordered as `MPD_S1, MPD_S2, ..., MPD_S11, MND_S1, ...,
    LMR_S11, sample_id, label`. This matches the locked Phase 6 contract:
    feature columns first, traceability columns last.

    Args:
        corrected_trimmed_signals: List of corrected-and-trimmed signal matrices in
            `sample_id`-sorted order. Each matrix must have shape `(L, 11)` and
            `numpy.float64` dtype.
        sample_ids: Sorted sample identifiers aligned 1:1 with
            `corrected_trimmed_signals`.
        labels: Binary labels aligned 1:1 with `corrected_trimmed_signals`.
        channel_names: Ordered gas-channel names used to label the feature columns.

    Returns:
        A DataFrame with `99` feature columns followed by `sample_id` and `label`.

    Raises:
        ValueError: If the inputs are length-misaligned, the signal matrices do not
            share the expected shape, or `channel_names` is invalid.
    """
    channel_name_tuple = _validate_channel_names(channel_names)
    sample_id_array = _coerce_traceability_array(sample_ids, "sample_ids")
    label_array = _coerce_traceability_array(labels, "labels")
    if len(corrected_trimmed_signals) != sample_id_array.shape[0] or len(corrected_trimmed_signals) != label_array.shape[0]:
        raise ValueError(
            "corrected_trimmed_signals, sample_ids, and labels must have the same length; "
            f"got {len(corrected_trimmed_signals)}, {sample_id_array.shape[0]}, and {label_array.shape[0]}"
        )
    if not corrected_trimmed_signals:
        raise ValueError("corrected_trimmed_signals must not be empty.")

    LOGGER.info("Starting Pipeline B feature extraction for %s samples.", len(corrected_trimmed_signals))
    clip_length = corrected_trimmed_signals[0].shape[0]
    feature_rows: list[dict[str, float | int]] = []
    for sample_index, signal in enumerate(corrected_trimmed_signals):
        signal_matrix = _validate_trimmed_signal(signal, expected_length=clip_length, signal_index=sample_index)
        feature_row: dict[str, float | int] = {}
        for family in FEATURE_FAMILIES:
            for channel_index, channel_name in enumerate(channel_name_tuple):
                signal_features = _extract_signal_features(signal_matrix[:, channel_index])
                feature_row[f"{family}_{channel_name}"] = signal_features[family]
        feature_row["sample_id"] = int(sample_id_array[sample_index])
        feature_row["label"] = int(label_array[sample_index])
        feature_rows.append(feature_row)

    columns = _feature_columns(channel_name_tuple) + ["sample_id", "label"]
    features_df = pd.DataFrame(feature_rows, columns=columns)
    features_df["sample_id"] = features_df["sample_id"].astype(np.int64)
    features_df["label"] = features_df["label"].astype(np.int64)
    LOGGER.info("Finished Pipeline B feature extraction with %s rows.", features_df.shape[0])
    return features_df


def compute_pipeline_b_stats(x_train_df: pd.DataFrame) -> dict[str, Any]:
    """Compute training-only Pipeline B normalization statistics.

    This function operates only on the `99` feature columns. The `sample_id` and
    `label` traceability columns are excluded from the mean and standard-deviation
    calculations.

    Args:
        x_train_df: Training feature table with the Pipeline B column contract.

    Returns:
        A dictionary with keys `feature_columns`, `mean`, and `std`.

    Raises:
        TypeError: If `x_train_df` is not a DataFrame.
        ValueError: If the DataFrame does not contain feature columns or omits the
            required traceability columns.
    """
    if not isinstance(x_train_df, pd.DataFrame):
        raise TypeError(f"x_train_df must be a pandas DataFrame; got {type(x_train_df).__name__}")
    if "sample_id" not in x_train_df.columns or "label" not in x_train_df.columns:
        raise ValueError("x_train_df must contain sample_id and label columns.")

    feature_columns = [column for column in x_train_df.columns if column not in {"sample_id", "label"}]
    if not feature_columns:
        raise ValueError("x_train_df must contain at least one feature column.")

    feature_frame = x_train_df.loc[:, feature_columns].astype(np.float64)
    mean = feature_frame.mean(axis=0)
    std = feature_frame.std(axis=0, ddof=0)
    std = std.where(std >= 1e-8, 1.0)
    return {"feature_columns": tuple(feature_columns), "mean": mean, "std": std}


def apply_pipeline_b_normalization(x_df: pd.DataFrame, stats: dict[str, Any]) -> pd.DataFrame:
    """Apply precomputed Pipeline B normalization statistics without mutating input.

    Args:
        x_df: Feature table to normalize.
        stats: Dictionary produced by `compute_pipeline_b_stats`.

    Returns:
        A new DataFrame whose feature columns are z-scored by the supplied
        training statistics while `sample_id` and `label` remain unchanged.

    Raises:
        TypeError: If `x_df` is not a DataFrame.
        ValueError: If `stats` is missing required entries or references absent
            feature columns.
    """
    if not isinstance(x_df, pd.DataFrame):
        raise TypeError(f"x_df must be a pandas DataFrame; got {type(x_df).__name__}")
    if not {"feature_columns", "mean", "std"} <= set(stats):
        raise ValueError("stats must contain feature_columns, mean, and std.")

    feature_columns = list(stats["feature_columns"])
    missing_columns = [column for column in feature_columns if column not in x_df.columns]
    if missing_columns:
        raise ValueError(f"x_df is missing feature columns required by stats: {missing_columns}")

    normalized_df = x_df.copy(deep=True)
    mean = pd.Series(stats["mean"], index=feature_columns, dtype=np.float64)
    std = pd.Series(stats["std"], index=feature_columns, dtype=np.float64)
    normalized_df.loc[:, feature_columns] = (
        normalized_df.loc[:, feature_columns].astype(np.float64).sub(mean, axis="columns").div(std, axis="columns")
    )
    return normalized_df


def save_pipeline_b(df: pd.DataFrame, path: Path) -> None:
    """Save the Pipeline B feature table to CSV.

    Args:
        df: Pipeline B feature table.
        path: Destination CSV path.

    Raises:
        OSError: If the CSV cannot be written.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Saving Pipeline B feature table to %s.", output_path)
    try:
        df.to_csv(output_path, index=False)
    except OSError as exc:
        raise OSError(f"Failed to write Pipeline B CSV to {output_path}: {exc}") from exc
    LOGGER.info("Saved Pipeline B feature table to %s.", output_path)


__all__ = [
    "EARLY_RESPONSE_FRACTION",
    "ERR_DENOMINATOR_GUARD",
    "FEATURE_FAMILIES",
    "LATE_RESPONSE_FRACTION",
    "T90_THRESHOLD_FRACTION",
    "apply_pipeline_b_normalization",
    "compute_pipeline_b_stats",
    "extract_pipeline_b_features",
    "save_pipeline_b",
]
