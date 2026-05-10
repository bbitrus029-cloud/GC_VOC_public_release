"""Prespecified preprocessing quality-control checks and serializers."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from src.config import GAS_CHANNELS, N_GAS_CHANNELS
from src.data.baseline import BaselineDetectionResult
from src.data.discovery import compute_tukey_upper_inner_fence
from src.data.loader import RawRecord


LOGGER = logging.getLogger(__name__)
_NUMERICAL_GUARD = 1e-8
_N_UNSTABLE_THRESHOLD = math.floor(N_GAS_CHANNELS / 2) + 1
_N_RESPOND_THRESHOLD = math.ceil(N_GAS_CHANNELS / 4)


@dataclass(frozen=True)
class QCResult:
    """Per-sample QC outcomes and audit-ready diagnostics.

    Attributes:
        sample_id: Traceability identifier from the raw record.
        qc1_env_sensor_fault_pass: Whether QC-1 passed.
        qc2_acquisition_truncated_pass: Whether QC-2 passed, or `None` when QC-1
            short-circuits later checks.
        qc3_unstable_baseline_pass: Whether QC-3 passed, or `None` when QC-1
            short-circuits later checks.
        qc4_insufficient_response_pass: Whether QC-4 passed, or `None` when QC-1
            short-circuits later checks.
        qc5_sensor_saturation_pass: Whether QC-5 passed, or `None` when QC-1
            short-circuits later checks.
        qc6_excessive_spikes_pass: Whether QC-6 passed, or `None` when QC-1
            short-circuits later checks.
        clip_length_pass: Placeholder for downstream trimming logic.
        fail_reasons: Ordered list of failed QC reason identifiers.
        qc_status: Aggregate QC status string, either `"pass"` or `"fail"`.
        n_rows: Number of rows in the sample trace.
        t_baseline_end: Resolved baseline end index.
        fallback_used: Whether the baseline result used the cohort fallback median.
        s12_baseline_mean: S12 early-baseline mean used by QC-1.
        s13_baseline_mean: S13 early-baseline mean for diagnostics.
        baseline_means: Per-channel baseline means on `[0, t_baseline_end)`.
        baseline_stds: Per-channel baseline standard deviations on `[0, t_baseline_end)`.
        baseline_cvs: Per-channel baseline coefficients of variation for QC-3.
        response_amplitudes: Per-channel signed response amplitudes for QC-4.
        saturation_fractions: Per-channel saturation fractions for QC-5.
        spike_counts: Per-channel spike counts for QC-6.
        env_temperature_sample: Per-sample environmental summary used later by
            environmental compensation. In Phase 4 this equals `s12_baseline_mean`.
        env_humidity_sample: Per-sample environmental summary used later by
            environmental compensation. In Phase 4 this equals `s13_baseline_mean`.
        env_reference_temperature: Placeholder for the later cohort reference temperature.
        env_reference_humidity: Placeholder for the later cohort reference humidity.
        env_temperature_humidity_correlation: Placeholder for the later cohort
            temperature-humidity correlation diagnostic.
        env_multicollinearity_fallback_used: Placeholder for the later
            environmental-compensation fallback flag.
        n_available: Response length after the resolved baseline end.
        clip_length_L: Placeholder for the later resolved clip length.
        relative_deficit: Placeholder for the later trimming relative deficit.
        padded: Placeholder for the later trimming padding flag.
    """

    sample_id: int
    qc1_env_sensor_fault_pass: bool
    qc2_acquisition_truncated_pass: bool | None
    qc3_unstable_baseline_pass: bool | None
    qc4_insufficient_response_pass: bool | None
    qc5_sensor_saturation_pass: bool | None
    qc6_excessive_spikes_pass: bool | None
    clip_length_pass: bool | None
    fail_reasons: list[str]
    qc_status: str
    n_rows: int
    t_baseline_end: int
    fallback_used: bool
    s12_baseline_mean: float
    s13_baseline_mean: float
    baseline_means: NDArray[np.float64]
    baseline_stds: NDArray[np.float64]
    baseline_cvs: NDArray[np.float64]
    response_amplitudes: NDArray[np.float64]
    saturation_fractions: NDArray[np.float64]
    spike_counts: NDArray[np.float64]
    env_temperature_sample: float
    env_humidity_sample: float
    env_reference_temperature: float
    env_reference_humidity: float
    env_temperature_humidity_correlation: float
    env_multicollinearity_fallback_used: bool | None
    n_available: float
    clip_length_L: float
    relative_deficit: float
    padded: bool | None


def _coerce_signal_matrix(gas_signals: NDArray[np.number[Any]], n_rows: int) -> NDArray[np.float64]:
    """Validate and coerce a gas-signal matrix to float64."""
    gas_matrix = np.asarray(gas_signals, dtype=np.float64)
    expected_shape = (n_rows, N_GAS_CHANNELS)
    if gas_matrix.shape != expected_shape:
        raise ValueError(f"gas_signals must have shape {expected_shape}; got {gas_matrix.shape}")
    return gas_matrix


def _coerce_env_trace(values: NDArray[np.number[Any]], name: str, n_rows: int) -> NDArray[np.float64]:
    """Validate and coerce an environmental trace to float64."""
    trace = np.asarray(values, dtype=np.float64)
    if trace.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {trace.shape}")
    if trace.shape[0] != n_rows:
        raise ValueError(f"{name} must have length {n_rows}; got {trace.shape[0]}")
    return trace


def _get_nested_threshold(thresholds: Mapping[str, Any], *keys: str) -> float:
    """Read one numeric threshold from a nested configuration mapping."""
    current_value: Any = thresholds
    traversed: list[str] = []
    for key in keys:
        traversed.append(key)
        if not isinstance(current_value, Mapping) or key not in current_value:
            joined_path = ".".join(traversed)
            raise ValueError(f"Threshold mapping is missing required key: {joined_path}")
        current_value = current_value[key]

    if not isinstance(current_value, (int, float)):
        joined_path = ".".join(keys)
        raise TypeError(f"Threshold value at {joined_path} must be numeric; got {type(current_value).__name__}")
    return float(current_value)


def _compute_env_baseline_means(
    env_temperature: NDArray[np.float64],
    env_humidity: NDArray[np.float64],
    baseline_points: int,
) -> tuple[float, float]:
    """Compute the QC-1 environmental baseline means."""
    if baseline_points <= 0:
        raise ValueError(f"baseline_points must be positive; got {baseline_points}")
    if baseline_points > env_temperature.shape[0] or baseline_points > env_humidity.shape[0]:
        raise ValueError(
            "baseline_points must not exceed the available environmental trace length; "
            f"got {baseline_points} for temperature={env_temperature.shape[0]} humidity={env_humidity.shape[0]}"
        )

    s12_baseline_mean = float(np.mean(env_temperature[:baseline_points]))
    s13_baseline_mean = float(np.mean(env_humidity[:baseline_points]))
    return s12_baseline_mean, s13_baseline_mean


def _evaluate_qc1(
    env_temperature: NDArray[np.float64],
    env_humidity: NDArray[np.float64],
    thresholds: Mapping[str, Any],
    qc1_enabled: bool,
    qc1_fault_side: str,
) -> tuple[bool, float, float]:
    """Evaluate QC-1 and return the pass flag plus S12/S13 baseline means."""
    if qc1_fault_side not in {"low", "high"}:
        raise ValueError(f"qc1_fault_side must be 'low' or 'high'; got {qc1_fault_side!r}")

    baseline_points = int(round(_get_nested_threshold(thresholds, "qc1_env_sensor_fault", "env_sensor_baseline_points")))
    qc1_threshold = _get_nested_threshold(thresholds, "qc1_env_sensor_fault", "threshold")
    s12_baseline_mean, s13_baseline_mean = _compute_env_baseline_means(
        env_temperature=env_temperature,
        env_humidity=env_humidity,
        baseline_points=baseline_points,
    )

    if not qc1_enabled:
        return True, s12_baseline_mean, s13_baseline_mean

    if qc1_fault_side == "low":
        qc1_pass = s12_baseline_mean >= qc1_threshold
    else:
        qc1_pass = s12_baseline_mean <= qc1_threshold

    return qc1_pass, s12_baseline_mean, s13_baseline_mean


def _compute_baseline_statistics(
    gas_signals: NDArray[np.float64],
    t_baseline_end: int,
    sample_id: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute per-channel baseline means, standard deviations, and CVs for QC-3."""
    baseline_slice = gas_signals[:t_baseline_end, :]
    baseline_means = np.mean(baseline_slice, axis=0)
    baseline_stds = np.std(baseline_slice, axis=0, ddof=0)
    baseline_cvs = np.empty(N_GAS_CHANNELS, dtype=np.float64)

    for channel_index in range(N_GAS_CHANNELS):
        denominator = abs(float(baseline_means[channel_index]))
        if denominator < _NUMERICAL_GUARD:
            LOGGER.warning(
                "QC-3 denominator below %.1e for sample_id=%s channel=%s; clamping to 1.0.",
                _NUMERICAL_GUARD,
                sample_id,
                GAS_CHANNELS[channel_index],
            )
            denominator = 1.0
        baseline_cvs[channel_index] = float(baseline_stds[channel_index]) / denominator

    return (
        np.asarray(baseline_means, dtype=np.float64),
        np.asarray(baseline_stds, dtype=np.float64),
        baseline_cvs,
    )


def _evaluate_qc2(n_rows: int, thresholds: Mapping[str, Any]) -> bool:
    """Evaluate QC-2 acquisition truncation."""
    minimum_rows = int(round(_get_nested_threshold(thresholds, "qc2_acquisition_truncated", "min_rows")))
    return n_rows >= minimum_rows


def _evaluate_qc3(baseline_cvs: NDArray[np.float64], thresholds: Mapping[str, Any]) -> bool:
    """Evaluate QC-3 unstable baseline."""
    threshold = _get_nested_threshold(thresholds, "qc3_unstable_baseline", "threshold")
    n_unstable_channels = int(np.count_nonzero(baseline_cvs > threshold))
    return n_unstable_channels < _N_UNSTABLE_THRESHOLD


def _compute_response_amplitudes(
    gas_signals: NDArray[np.float64],
    t_baseline_end: int,
    baseline_means: NDArray[np.float64],
    sample_id: int,
) -> NDArray[np.float64]:
    """Compute per-channel signed response amplitudes for QC-4."""
    response_amplitudes = np.empty(N_GAS_CHANNELS, dtype=np.float64)
    response_slice = gas_signals[t_baseline_end:, :]
    for channel_index in range(N_GAS_CHANNELS):
        channel_response = response_slice[:, channel_index]
        baseline_mean = float(baseline_means[channel_index])
        delta_max = float(np.max(channel_response) - baseline_mean)
        delta_min = float(np.min(channel_response) - baseline_mean)
        dominant_delta = delta_max if abs(delta_max) >= abs(delta_min) else delta_min

        denominator = abs(baseline_mean)
        if denominator < _NUMERICAL_GUARD:
            LOGGER.warning(
                "QC-4 denominator below %.1e for sample_id=%s channel=%s; clamping to 1.0.",
                _NUMERICAL_GUARD,
                sample_id,
                GAS_CHANNELS[channel_index],
            )
            denominator = 1.0
        response_amplitudes[channel_index] = dominant_delta / denominator

    return response_amplitudes


def _evaluate_qc4(response_amplitudes: NDArray[np.float64], thresholds: Mapping[str, Any]) -> bool:
    """Evaluate QC-4 insufficient response."""
    threshold = _get_nested_threshold(thresholds, "qc4_insufficient_response", "threshold")
    n_responding_channels = int(np.count_nonzero(np.abs(response_amplitudes) >= threshold))
    return n_responding_channels >= _N_RESPOND_THRESHOLD


def _compute_saturation_fractions(gas_signals: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute per-channel within-one-unit saturation fractions for QC-5."""
    saturation_fractions = np.empty(N_GAS_CHANNELS, dtype=np.float64)
    for channel_index in range(N_GAS_CHANNELS):
        channel_signal = gas_signals[:, channel_index]
        channel_max = float(np.max(channel_signal))
        is_saturated = np.abs(channel_signal - channel_max) <= 1.0
        saturation_fractions[channel_index] = float(np.mean(is_saturated))
    return saturation_fractions


def _evaluate_qc5(saturation_fractions: NDArray[np.float64], thresholds: Mapping[str, Any]) -> bool:
    """Evaluate QC-5 sensor saturation."""
    threshold = _get_nested_threshold(thresholds, "qc5_sensor_saturation", "threshold")
    return not bool(np.any(saturation_fractions > threshold))


def _compute_spike_counts(gas_signals: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute per-channel Tukey spike counts for QC-6."""
    spike_counts = np.zeros(N_GAS_CHANNELS, dtype=np.float64)
    for channel_index in range(N_GAS_CHANNELS):
        absolute_differences = np.abs(np.diff(gas_signals[:, channel_index]))
        if absolute_differences.size == 0:
            spike_counts[channel_index] = 0.0
            continue
        channel_fence = compute_tukey_upper_inner_fence(absolute_differences)
        spike_counts[channel_index] = float(np.count_nonzero(absolute_differences > channel_fence))
    return spike_counts


def _evaluate_qc6(spike_counts: NDArray[np.float64], thresholds: Mapping[str, Any]) -> bool:
    """Evaluate QC-6 excessive transient spikes."""
    threshold = _get_nested_threshold(thresholds, "qc6_excessive_spikes", "threshold")
    return not bool(np.any(spike_counts > threshold))


def _build_short_circuit_result(
    record: RawRecord,
    baseline_result: BaselineDetectionResult,
    s12_baseline_mean: float,
    s13_baseline_mean: float,
) -> QCResult:
    """Build the QC-1 short-circuit result with NaN diagnostics for QC-2 through QC-6."""
    nan_vector = np.full(N_GAS_CHANNELS, np.nan, dtype=np.float64)
    return QCResult(
        sample_id=record.sample_id,
        qc1_env_sensor_fault_pass=False,
        qc2_acquisition_truncated_pass=None,
        qc3_unstable_baseline_pass=None,
        qc4_insufficient_response_pass=None,
        qc5_sensor_saturation_pass=None,
        qc6_excessive_spikes_pass=None,
        clip_length_pass=None,
        fail_reasons=["qc1_env_sensor_fault"],
        qc_status="fail",
        n_rows=record.n_rows,
        t_baseline_end=int(baseline_result.t_baseline_end),
        fallback_used=baseline_result.fallback_used,
        s12_baseline_mean=s12_baseline_mean,
        s13_baseline_mean=s13_baseline_mean,
        baseline_means=nan_vector.copy(),
        baseline_stds=nan_vector.copy(),
        baseline_cvs=nan_vector.copy(),
        response_amplitudes=nan_vector.copy(),
        saturation_fractions=nan_vector.copy(),
        spike_counts=nan_vector.copy(),
        env_temperature_sample=s12_baseline_mean,
        env_humidity_sample=s13_baseline_mean,
        env_reference_temperature=np.nan,
        env_reference_humidity=np.nan,
        env_temperature_humidity_correlation=np.nan,
        env_multicollinearity_fallback_used=None,
        n_available=np.nan,
        clip_length_L=np.nan,
        relative_deficit=np.nan,
        padded=None,
    )


def run_qc(
    record: RawRecord,
    baseline_result: BaselineDetectionResult,
    gas_signals: NDArray[np.number[Any]],
    env_temperature: NDArray[np.number[Any]],
    env_humidity: NDArray[np.number[Any]],
    thresholds: Mapping[str, Any],
    qc1_enabled: bool,
    qc1_fault_side: str,
) -> QCResult:
    """Run the six prespecified preprocessing QC checks for one sample.

    Args:
        record: Raw-record metadata used for traceability and row-count diagnostics.
        baseline_result: Resolved baseline result for the sample. Its
            `t_baseline_end` must already be an integer index, including the
            cohort fallback assignment when needed.
        gas_signals: Gas-channel matrix to evaluate for QC-3 through QC-6 with
            shape `(n_rows, 11)`.
        env_temperature: S12 environmental trace with shape `(n_rows,)`.
        env_humidity: S13 environmental trace with shape `(n_rows,)`.
        thresholds: Nested threshold mapping loaded from `configs/preprocessing.yaml`.
        qc1_enabled: Whether QC-1 is enabled for this cohort.
        qc1_fault_side: Which side of the QC-1 threshold is faulty: `"low"` or `"high"`.

    Returns:
        A `QCResult` containing per-check pass flags, ordered failure reasons, and
        audit-ready diagnostics.

    Raises:
        TypeError: If `thresholds` is not a mapping.
        ValueError: If array shapes are invalid, the baseline end is unresolved,
            or threshold keys are missing.
    """
    if not isinstance(thresholds, Mapping):
        raise TypeError(f"thresholds must be a mapping; got {type(thresholds).__name__}")
    if baseline_result.t_baseline_end is None:
        raise ValueError("baseline_result.t_baseline_end must be resolved before run_qc is called.")

    gas_matrix = _coerce_signal_matrix(gas_signals=gas_signals, n_rows=record.n_rows)
    temperature_trace = _coerce_env_trace(values=env_temperature, name="env_temperature", n_rows=record.n_rows)
    humidity_trace = _coerce_env_trace(values=env_humidity, name="env_humidity", n_rows=record.n_rows)

    t_baseline_end = int(baseline_result.t_baseline_end)
    if t_baseline_end <= 0 or t_baseline_end >= record.n_rows:
        raise ValueError(
            "baseline_result.t_baseline_end must satisfy 0 < t_baseline_end < n_rows; "
            f"got t_baseline_end={t_baseline_end} n_rows={record.n_rows}"
        )

    qc1_pass, s12_baseline_mean, s13_baseline_mean = _evaluate_qc1(
        env_temperature=temperature_trace,
        env_humidity=humidity_trace,
        thresholds=thresholds,
        qc1_enabled=qc1_enabled,
        qc1_fault_side=qc1_fault_side,
    )
    if not qc1_pass:
        return _build_short_circuit_result(
            record=record,
            baseline_result=baseline_result,
            s12_baseline_mean=s12_baseline_mean,
            s13_baseline_mean=s13_baseline_mean,
        )

    baseline_means, baseline_stds, baseline_cvs = _compute_baseline_statistics(
        gas_signals=gas_matrix,
        t_baseline_end=t_baseline_end,
        sample_id=record.sample_id,
    )
    response_amplitudes = _compute_response_amplitudes(
        gas_signals=gas_matrix,
        t_baseline_end=t_baseline_end,
        baseline_means=baseline_means,
        sample_id=record.sample_id,
    )
    saturation_fractions = _compute_saturation_fractions(gas_signals=gas_matrix)
    spike_counts = _compute_spike_counts(gas_signals=gas_matrix)

    qc2_pass = _evaluate_qc2(n_rows=record.n_rows, thresholds=thresholds)
    qc3_pass = _evaluate_qc3(baseline_cvs=baseline_cvs, thresholds=thresholds)
    qc4_pass = _evaluate_qc4(response_amplitudes=response_amplitudes, thresholds=thresholds)
    qc5_pass = _evaluate_qc5(saturation_fractions=saturation_fractions, thresholds=thresholds)
    qc6_pass = _evaluate_qc6(spike_counts=spike_counts, thresholds=thresholds)

    fail_reasons: list[str] = []
    if not qc2_pass:
        fail_reasons.append("qc2_acquisition_truncated")
    if not qc3_pass:
        fail_reasons.append("qc3_unstable_baseline")
    if not qc4_pass:
        fail_reasons.append("qc4_insufficient_response")
    if not qc5_pass:
        fail_reasons.append("qc5_sensor_saturation")
    if not qc6_pass:
        fail_reasons.append("qc6_excessive_spikes")

    return QCResult(
        sample_id=record.sample_id,
        qc1_env_sensor_fault_pass=True,
        qc2_acquisition_truncated_pass=qc2_pass,
        qc3_unstable_baseline_pass=qc3_pass,
        qc4_insufficient_response_pass=qc4_pass,
        qc5_sensor_saturation_pass=qc5_pass,
        qc6_excessive_spikes_pass=qc6_pass,
        clip_length_pass=None,
        fail_reasons=fail_reasons,
        qc_status="pass" if not fail_reasons else "fail",
        n_rows=record.n_rows,
        t_baseline_end=t_baseline_end,
        fallback_used=baseline_result.fallback_used,
        s12_baseline_mean=s12_baseline_mean,
        s13_baseline_mean=s13_baseline_mean,
        baseline_means=baseline_means,
        baseline_stds=baseline_stds,
        baseline_cvs=baseline_cvs,
        response_amplitudes=response_amplitudes,
        saturation_fractions=saturation_fractions,
        spike_counts=spike_counts,
        env_temperature_sample=s12_baseline_mean,
        env_humidity_sample=s13_baseline_mean,
        env_reference_temperature=np.nan,
        env_reference_humidity=np.nan,
        env_temperature_humidity_correlation=np.nan,
        env_multicollinearity_fallback_used=None,
        n_available=float(record.n_rows - t_baseline_end),
        clip_length_L=np.nan,
        relative_deficit=np.nan,
        padded=None,
    )


def _summary_columns() -> list[str]:
    """Return the canonical QC summary column order."""
    return [
        "sample_id",
        "qc1_env_sensor_fault_pass",
        "qc2_acquisition_truncated_pass",
        "qc3_unstable_baseline_pass",
        "qc4_insufficient_response_pass",
        "qc5_sensor_saturation_pass",
        "qc6_excessive_spikes_pass",
        "clip_length_pass",
        "fail_reasons",
        "qc_status",
    ]


def qc_summary_to_df(results: list[QCResult]) -> pd.DataFrame:
    """Serialize QC summary results to the `qc_summary.csv` schema.

    Args:
        results: Ordered list of `QCResult` objects.

    Returns:
        A DataFrame with the prespecified QC summary columns.
    """
    rows = [
        {
            "sample_id": result.sample_id,
            "qc1_env_sensor_fault_pass": result.qc1_env_sensor_fault_pass,
            "qc2_acquisition_truncated_pass": result.qc2_acquisition_truncated_pass,
            "qc3_unstable_baseline_pass": result.qc3_unstable_baseline_pass,
            "qc4_insufficient_response_pass": result.qc4_insufficient_response_pass,
            "qc5_sensor_saturation_pass": result.qc5_sensor_saturation_pass,
            "qc6_excessive_spikes_pass": result.qc6_excessive_spikes_pass,
            "clip_length_pass": result.clip_length_pass,
            "fail_reasons": ";".join(result.fail_reasons),
            "qc_status": result.qc_status,
        }
        for result in results
    ]
    summary_df = pd.DataFrame(rows, columns=_summary_columns())
    for column in _summary_columns()[1:8]:
        summary_df[column] = pd.Series(summary_df[column], dtype="boolean")
    return summary_df


def _diagnostic_columns() -> list[str]:
    """Return the canonical QC diagnostics column order."""
    columns = [
        "sample_id",
        "n_rows",
        "t_baseline_end",
        "fallback_used",
        "s12_baseline_mean",
        "s13_baseline_mean",
        "env_temperature_sample",
        "env_humidity_sample",
        "env_reference_temperature",
        "env_reference_humidity",
        "env_temperature_humidity_correlation",
        "env_multicollinearity_fallback_used",
        "n_available",
        "clip_length_L",
        "relative_deficit",
        "padded",
    ]
    for channel_name in GAS_CHANNELS:
        prefix = channel_name.lower()
        columns.extend(
            [
                f"{prefix}_baseline_mean",
                f"{prefix}_baseline_std",
                f"{prefix}_baseline_cv",
                f"{prefix}_response_amplitude",
                f"{prefix}_saturation_fraction",
                f"{prefix}_spike_count",
            ]
        )
    return columns


def _diagnostic_row(result: QCResult) -> dict[str, Any]:
    """Convert one QC result into a diagnostics-row dictionary."""
    row: dict[str, Any] = {
        "sample_id": result.sample_id,
        "n_rows": result.n_rows,
        "t_baseline_end": result.t_baseline_end,
        "fallback_used": result.fallback_used,
        "s12_baseline_mean": result.s12_baseline_mean,
        "s13_baseline_mean": result.s13_baseline_mean,
        "env_temperature_sample": result.env_temperature_sample,
        "env_humidity_sample": result.env_humidity_sample,
        "env_reference_temperature": result.env_reference_temperature,
        "env_reference_humidity": result.env_reference_humidity,
        "env_temperature_humidity_correlation": result.env_temperature_humidity_correlation,
        "env_multicollinearity_fallback_used": result.env_multicollinearity_fallback_used,
        "n_available": result.n_available,
        "clip_length_L": result.clip_length_L,
        "relative_deficit": result.relative_deficit,
        "padded": result.padded,
    }
    for channel_index, channel_name in enumerate(GAS_CHANNELS):
        prefix = channel_name.lower()
        row[f"{prefix}_baseline_mean"] = float(result.baseline_means[channel_index])
        row[f"{prefix}_baseline_std"] = float(result.baseline_stds[channel_index])
        row[f"{prefix}_baseline_cv"] = float(result.baseline_cvs[channel_index])
        row[f"{prefix}_response_amplitude"] = float(result.response_amplitudes[channel_index])
        row[f"{prefix}_saturation_fraction"] = float(result.saturation_fractions[channel_index])
        row[f"{prefix}_spike_count"] = float(result.spike_counts[channel_index])
    return row


def qc_diagnostics_to_df(results: list[QCResult]) -> pd.DataFrame:
    """Serialize QC diagnostics to the `qc_diagnostics.csv` schema groups.

    Args:
        results: Ordered list of `QCResult` objects.

    Returns:
        A DataFrame with the prespecified diagnostics groups expanded into columns.
    """
    diagnostics_df = pd.DataFrame([_diagnostic_row(result) for result in results], columns=_diagnostic_columns())
    for boolean_column in ["fallback_used", "env_multicollinearity_fallback_used", "padded"]:
        diagnostics_df[boolean_column] = pd.Series(diagnostics_df[boolean_column], dtype="boolean")
    return diagnostics_df


__all__ = ["QCResult", "qc_diagnostics_to_df", "qc_summary_to_df", "run_qc"]
