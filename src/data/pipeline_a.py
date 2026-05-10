"""Pipeline A tensor assembly, persistence, and normalization utilities."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.config import GAS_CHANNELS, N_GAS_CHANNELS


LOGGER = logging.getLogger(__name__)


def _coerce_traceability_array(values: list[int] | NDArray[np.integer[Any]], name: str) -> NDArray[np.int64]:
    """Validate one aligned integer traceability vector."""
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}")
    return array


def _validate_signal_matrix(
    signal: NDArray[np.float64],
    *,
    clip_length_l: int,
    signal_index: int,
) -> NDArray[np.float64]:
    """Validate one corrected-and-trimmed Pipeline A input matrix."""
    matrix = np.asarray(signal)
    if matrix.dtype != np.float64:
        raise ValueError(f"corrected_trimmed_signals[{signal_index}] must have dtype float64; got {matrix.dtype}")
    expected_shape = (clip_length_l, N_GAS_CHANNELS)
    if matrix.shape != expected_shape:
        raise ValueError(
            f"corrected_trimmed_signals[{signal_index}] must have shape {expected_shape}; got {matrix.shape}"
        )
    return matrix


def _validate_tensor(x: NDArray[np.float64], clip_length_l: int) -> NDArray[np.float64]:
    """Validate a Pipeline A tensor shape and dtype."""
    tensor = np.asarray(x)
    if tensor.dtype != np.float64:
        raise ValueError(f"x must have dtype float64; got {tensor.dtype}")
    if tensor.ndim != 3:
        raise ValueError(f"x must be three-dimensional with shape (N, L, 11); got shape {tensor.shape}")
    if tensor.shape[1] != clip_length_l or tensor.shape[2] != N_GAS_CHANNELS:
        raise ValueError(
            f"x must have shape (N, {clip_length_l}, {N_GAS_CHANNELS}); got {tensor.shape}"
        )
    return tensor


@dataclass(frozen=True)
class PipelineAOutput:
    """Pipeline A tensor payload aligned to the final output cohort.

    Attributes:
        x: Float64 tensor with shape `(N, L, 11)`.
        sample_id: Int64 sample identifiers aligned to the first tensor dimension.
        label: Int64 binary labels aligned to the first tensor dimension.
        channel_names: Ordered gas-channel names, `S1` through `S11`.
        clip_length_L: Shared response clip length `L`.
    """

    x: NDArray[np.float64]
    sample_id: NDArray[np.int64]
    label: NDArray[np.int64]
    channel_names: tuple[str, ...]
    clip_length_L: int

    def __post_init__(self) -> None:
        """Validate the stored tensor payload."""
        clip_length_l = int(self.clip_length_L)
        if clip_length_l <= 0:
            raise ValueError(f"clip_length_L must be positive; got {clip_length_l}")

        tensor = _validate_tensor(self.x, clip_length_l)
        sample_id = _coerce_traceability_array(self.sample_id, "sample_id")
        label = _coerce_traceability_array(self.label, "label")
        if tensor.shape[0] != sample_id.shape[0] or tensor.shape[0] != label.shape[0]:
            raise ValueError(
                "x, sample_id, and label must share the same first dimension; "
                f"got {tensor.shape[0]}, {sample_id.shape[0]}, and {label.shape[0]}"
            )
        if tuple(self.channel_names) != tuple(GAS_CHANNELS):
            raise ValueError(f"channel_names must match GAS_CHANNELS exactly; got {self.channel_names}")

        object.__setattr__(self, "x", tensor)
        object.__setattr__(self, "sample_id", sample_id)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "channel_names", tuple(self.channel_names))
        object.__setattr__(self, "clip_length_L", clip_length_l)


def assemble_pipeline_a(
    corrected_trimmed_signals: list[NDArray[np.float64]],
    sample_ids: list[int] | NDArray[np.integer[Any]],
    labels: list[int] | NDArray[np.integer[Any]],
    clip_length_L: int,
) -> PipelineAOutput:
    """Assemble the unnormalized Pipeline A tensor from trimmed gas-signal matrices.

    Args:
        corrected_trimmed_signals: List of corrected-and-trimmed signal matrices in
            `sample_id`-sorted order. Each matrix must have shape `(L, 11)` and
            `numpy.float64` dtype.
        sample_ids: Sorted sample identifiers aligned 1:1 with the signal list.
        labels: Binary labels aligned 1:1 with the signal list.
        clip_length_L: Shared response clip length `L`.

    Returns:
        A validated `PipelineAOutput` dataclass.

    Raises:
        ValueError: If the inputs are empty, length-misaligned, or any signal
            matrix violates the `(L, 11)` float64 contract.
    """
    clip_length_l = int(clip_length_L)
    if clip_length_l <= 0:
        raise ValueError(f"clip_length_L must be positive; got {clip_length_l}")

    sample_id_array = _coerce_traceability_array(sample_ids, "sample_ids")
    label_array = _coerce_traceability_array(labels, "labels")
    if len(corrected_trimmed_signals) != sample_id_array.shape[0] or len(corrected_trimmed_signals) != label_array.shape[0]:
        raise ValueError(
            "corrected_trimmed_signals, sample_ids, and labels must have the same length; "
            f"got {len(corrected_trimmed_signals)}, {sample_id_array.shape[0]}, and {label_array.shape[0]}"
        )
    if not corrected_trimmed_signals:
        raise ValueError("corrected_trimmed_signals must not be empty.")

    LOGGER.info(
        "Starting Pipeline A assembly for %s samples with clip_length_L=%s.",
        len(corrected_trimmed_signals),
        clip_length_l,
    )
    validated_signals = [
        _validate_signal_matrix(signal, clip_length_l=clip_length_l, signal_index=signal_index)
        for signal_index, signal in enumerate(corrected_trimmed_signals)
    ]
    tensor = np.stack(validated_signals, axis=0)
    output = PipelineAOutput(
        x=np.asarray(tensor, dtype=np.float64),
        sample_id=sample_id_array,
        label=label_array,
        channel_names=tuple(GAS_CHANNELS),
        clip_length_L=clip_length_l,
    )
    LOGGER.info("Finished Pipeline A assembly with shape=%s.", output.x.shape)
    return output


def save_pipeline_a(output: PipelineAOutput, path: Path) -> None:
    """Save the Pipeline A payload as a `.npz` archive.

    Args:
        output: Validated Pipeline A payload.
        path: Destination archive path.

    Raises:
        OSError: If the archive cannot be written.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Saving Pipeline A archive to %s.", output_path)
    try:
        np.savez(
            output_path,
            x=output.x,
            sample_id=output.sample_id,
            label=output.label,
            channel_names=np.asarray(output.channel_names),
            clip_length_L=np.int64(output.clip_length_L),
        )
    except OSError as exc:
        raise OSError(f"Failed to write Pipeline A archive to {output_path}: {exc}") from exc
    LOGGER.info("Saved Pipeline A archive to %s.", output_path)


def compute_pipeline_a_stats(x_train: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
    """Compute training-only per-channel normalization statistics over axes `(0, 1)`.

    The mean and standard deviation are computed across the sample axis and time
    axis together, that is `axis=(0, 1)`, which yields one scalar mean and one
    scalar standard deviation for each gas channel. Standard deviations below
    `1e-8` are clamped to `1.0`.

    Args:
        x_train: Training tensor with shape `(N, L, 11)` and `numpy.float64` dtype.

    Returns:
        A dictionary with keys `mean` and `std`, both shaped `(11,)`.

    Raises:
        ValueError: If `x_train` does not satisfy the Pipeline A tensor contract.
    """
    tensor = np.asarray(x_train)
    if tensor.dtype != np.float64:
        raise ValueError(f"x_train must have dtype float64; got {tensor.dtype}")
    if tensor.ndim != 3 or tensor.shape[2] != N_GAS_CHANNELS:
        raise ValueError(f"x_train must have shape (N, L, {N_GAS_CHANNELS}); got {tensor.shape}")

    mean = np.mean(tensor, axis=(0, 1), dtype=np.float64)
    std = np.std(tensor, axis=(0, 1), dtype=np.float64)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float64)
    return {"mean": mean.astype(np.float64), "std": std}


def apply_pipeline_a_normalization(
    x: NDArray[np.float64],
    stats: dict[str, NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Apply precomputed Pipeline A channel-wise normalization without mutating input.

    Args:
        x: Tensor to normalize with shape `(N, L, 11)` and `numpy.float64` dtype.
        stats: Dictionary produced by `compute_pipeline_a_stats`.

    Returns:
        A new normalized tensor with the same shape and `numpy.float64` dtype.

    Raises:
        ValueError: If `x` or `stats` violates the expected contract.
    """
    if not {"mean", "std"} <= set(stats):
        raise ValueError("stats must contain mean and std.")

    tensor = np.asarray(x)
    if tensor.dtype != np.float64:
        raise ValueError(f"x must have dtype float64; got {tensor.dtype}")
    if tensor.ndim != 3 or tensor.shape[2] != N_GAS_CHANNELS:
        raise ValueError(f"x must have shape (N, L, {N_GAS_CHANNELS}); got {tensor.shape}")

    mean = np.asarray(stats["mean"], dtype=np.float64)
    std = np.asarray(stats["std"], dtype=np.float64)
    if mean.shape != (N_GAS_CHANNELS,) or std.shape != (N_GAS_CHANNELS,):
        raise ValueError(
            f"stats mean and std must each have shape ({N_GAS_CHANNELS},); got {mean.shape} and {std.shape}"
        )

    normalized = (tensor.astype(np.float64, copy=True) - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    return np.asarray(normalized, dtype=np.float64)


__all__ = [
    "PipelineAOutput",
    "apply_pipeline_a_normalization",
    "assemble_pipeline_a",
    "compute_pipeline_a_stats",
    "save_pipeline_a",
]
