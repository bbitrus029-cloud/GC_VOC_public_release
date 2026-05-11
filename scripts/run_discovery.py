"""Run the prespecified label-blind preprocessing discovery workflow.

This script orchestrates the Phase-1 discovery contract from Section 3.2 of the
locked preprocessing plan. It computes cohort-level thresholds and calibration
values only; it does not make final QC exclusion decisions, does not write
per-sample QC audit flags, and does not produce Pipeline A or Pipeline B
outputs. The intermediate `qc1_pass_internal` and `qc2_pass_internal` flags
exist only in memory so the script can scope the sub-cohorts required for the
prespecified QC-3/QC-4/QC-5/QC-6, environmental-compensation, and trimming
threshold computations.

QC-1 fault-side assignment follows Erratum E002. K-of-N fallback-median integer
coercion follows Erratum E003. This module has no CLI argument parsing and is
import-safe; the workflow runs only through `main()`.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import yaml

from src.config import (
    BASELINE_CONSECUTIVE_EXCEED,
    BASELINE_K_OF_N,
    BASELINE_SIGMA_MULTIPLIER,
    BREATH_DATA_ROOT,
    CONFIGS_ROOT,
    GAS_CHANNELS,
    HUMIDITY_CHANNEL,
    METADATA_PATH,
    MIN_ROWS_QC2,
    OUTPUTS_LOGS_ROOT,
    PROCESSED_DATA_ROOT,
    PROJECT_ROOT,
    RANDOM_SEED,
    S0_DROPPED,
    SG_POLYORDER,
    SG_WINDOW_LENGTH,
    T_BASELINE_END_MAX,
    T_BASELINE_END_MIN,
    T_BASELINE_WINDOW_INIT,
    TEMPERATURE_CHANNEL,
)
from src.data import discovery
from src.data.baseline import BaselineDetectionResult, detect_baseline_end
from src.data.loader import RawRecord, load_dataset, load_metadata
from src.data.smoothing import apply_sg_smoothing


LOGGER = logging.getLogger(__name__)

_LABEL_BLINDING_STATEMENT = (
    "Labels were loaded only for sample traceability and were not used for any threshold extraction, "
    "cohort masking, regression fitting, or trimming parameter selection in this discovery run."
)
_PIPELINE_B_FEATURE_SUFFIXES = ("mpd", "mnd", "auc", "mean", "mas", "mds", "t90", "err", "lmr")
_PIPELINE_B_FEATURE_NAMES = [f"{channel.lower()}_{suffix}" for channel in GAS_CHANNELS for suffix in _PIPELINE_B_FEATURE_SUFFIXES]
_T90_FRACTION_OF_PEAK = 0.90
_EARLY_RESPONSE_FRACTION = 0.20
_LATE_MEAN_FRACTION = 0.50
_S0_WARNING_ABS_THRESHOLD = 10
_TEMPERATURE_HUMIDITY_ABS_R_THRESHOLD = 0.80
_ADJUSTED_R2_THRESHOLD = 0.0
QC5_FALLBACK_PERCENTILE = 95  # [signal-processing-literature: prespecified 95th-percentile right-tail outlier boundary per Erratum E004]
_CURRENT_METADATA_PATH: Path | None = None
_CURRENT_BREATH_DATA_ROOT: Path | None = None

_L17_PROVENANCE: dict[str, str] = {
    "provenance_schema_version": "[domain-physical-constant: first preprocessing provenance schema]",
    "random_seed": "[domain-physical-constant: project reproducibility convention]",
    "baseline_detection.t_baseline_window_init": (
        "[domain-physical-constant: SOP-derived early baseline window; acquisition starts before subject blow]"
    ),
    "baseline_detection.t_baseline_end_min": (
        "[domain-physical-constant: same SOP-derived minimum early baseline bound as t_baseline_window_init]"
    ),
    "baseline_detection.t_baseline_end_max": (
        "[domain-physical-constant: approximately 6 seconds at 8.33 Hz, from prompt user-confirmed SOP "
        "requiring prompt blow onset]"
    ),
    "baseline_detection.sigma_multiplier": (
        "[signal-processing-literature: Shewhart 3-sigma control limit; NIST SPC handbook]"
    ),
    "baseline_detection.consecutive_exceed": (
        "[signal-processing-literature: short sustained run confirmation adapted from control-chart run-rule "
        "logic; Western Electric 1956; Montgomery SPC]"
    ),
    "baseline_detection.k_of_n": (
        "[signal-processing-literature: weak-consensus multisensor voting rule; 11 is domain channel count, "
        "denominator 3 encodes more-than-one-third quorum without sentinel-channel selection]"
    ),
    "baseline_detection.fallback_median_float": "[runtime-determined-by-prespecified-algorithm]",
    "baseline_detection.fallback_median": "[runtime-determined-by-prespecified-algorithm]",
    "qc_thresholds.qc1_env_sensor_fault.env_sensor_baseline_points": (
        "[domain-physical-constant: same SOP-derived early baseline window as §3.3 onset detection; "
        "consistent definition of \"device-just-started, no breath yet\" interval used across both QC-1 "
        "environmental fault detection and gas-channel onset detection]"
    ),
    "qc_thresholds.qc1_env_sensor_fault.enabled": "[runtime-determined-by-prespecified-algorithm]",
    "qc_thresholds.qc1_env_sensor_fault.threshold": "[runtime-determined-by-prespecified-algorithm]",
    "qc_thresholds.qc1_env_sensor_fault.fault_side": "[runtime-determined-by-prespecified-algorithm]",
    "qc_thresholds.qc2_acquisition_truncated.min_rows": (
        "[domain-physical-constant: hardware/SOP acquisition length]"
    ),
    "qc_thresholds.qc3_unstable_baseline.threshold": "[runtime-determined-by-prespecified-algorithm]",
    "qc_thresholds.qc4_insufficient_response.threshold": "[runtime-determined-by-prespecified-algorithm]",
    "qc_thresholds.qc4_insufficient_response.threshold_method": "[runtime-determined-by-prespecified-algorithm]",
    "qc_thresholds.qc5_sensor_saturation.threshold": "[runtime-determined-by-prespecified-algorithm]",
    "qc_thresholds.qc5_sensor_saturation.threshold_method": "[runtime-determined-by-prespecified-algorithm]",
    "qc_thresholds.qc6_excessive_spikes.threshold": "[runtime-determined-by-prespecified-algorithm]",
    "smoothing.window_length": (
        "[signal-processing-literature: odd-window Savitzky-Golay local polynomial smoothing; at "
        "approximately 8.33 Hz this is approximately 1.32 seconds, much shorter than MOS response/recovery "
        "dynamics on the order of 10-30 seconds; Savitzky and Golay 1964; Press et al.; MOS denoising "
        "practice]"
    ),
    "smoothing.polyorder": (
        "[signal-processing-literature: cubic local polynomial commonly used to preserve peak/curve shape while "
        "smoothing; Savitzky and Golay 1964; Press et al.]"
    ),
    "environmental_compensation.temperature_humidity_abs_pearson_r_threshold": (
        "[signal-processing-literature: common severe pairwise-correlation screening heuristic, applied "
        "cautiously because VIF rules are context-dependent; O'Brien 2007]"
    ),
    "environmental_compensation.temperature_humidity_pearson_r": "[runtime-determined-by-prespecified-algorithm]",
    "environmental_compensation.multicollinearity_fallback_used": (
        "[runtime-determined-by-prespecified-algorithm]"
    ),
    "environmental_compensation.r2_threshold": (
        "[signal-processing-literature: adjusted R² greater than zero indicates improvement over intercept-only "
        "model; replaces unsupported 0.10 threshold]"
    ),
    "environmental_compensation.t_ref": "[runtime-determined-by-prespecified-algorithm]",
    "environmental_compensation.h_ref": "[runtime-determined-by-prespecified-algorithm]",
    "correction_and_trimming.s0_warning_abs_threshold": (
        "[domain-physical-constant: engineering guard tied to integer ADC-like raw scale supplied in source "
        "facts; warning only, not auto-exclusion]"
    ),
    "correction_and_trimming.clip_length_L": (
        "[runtime-determined-by-prespecified-algorithm: 5th percentile is a robust low-tail summary preserving "
        "at least 95% without truncation; percentile convention follows robust descriptive-statistics practice]"
    ),
    "correction_and_trimming.padding_relative_deficit_threshold": (
        "[runtime-determined-by-prespecified-algorithm]"
    ),
    "pipeline_b_features.t90_fraction_of_peak": (
        "[signal-processing-literature: T0-90%/T90 response-time features are standard gas-sensor response "
        "descriptors; Yan et al. 2015]"
    ),
    "pipeline_b_features.early_response_fraction": (
        "[signal-processing-literature: windowed time-slicing and early transient summaries are established "
        "e-nose features; 0.20 is retained as a prespecified one-fifth early-window partition, not "
        "cohort-derived]"
    ),
    "pipeline_b_features.late_mean_fraction": (
        "[signal-processing-literature: interval mean/windowed time-slicing response summaries are established "
        "e-nose features; 0.50 is retained as a prespecified half-window late-response partition, not "
        "cohort-derived]"
    ),
}

for _channel_name in GAS_CHANNELS:
    _L17_PROVENANCE[f"environmental_compensation.per_channel.{_channel_name}.alpha"] = (
        "[runtime-determined-by-prespecified-algorithm]"
    )
    _L17_PROVENANCE[f"environmental_compensation.per_channel.{_channel_name}.beta_T"] = (
        "[runtime-determined-by-prespecified-algorithm]"
    )
    _L17_PROVENANCE[f"environmental_compensation.per_channel.{_channel_name}.beta_H"] = (
        "[runtime-determined-by-prespecified-algorithm]"
    )
    _L17_PROVENANCE[f"environmental_compensation.per_channel.{_channel_name}.adj_r2"] = (
        "[runtime-determined-by-prespecified-algorithm]"
    )
    _L17_PROVENANCE[f"environmental_compensation.per_channel.{_channel_name}.applies_compensation"] = (
        "[runtime-determined-by-prespecified-algorithm]"
    )

_L17_CITATIONS: dict[str, str] = {
    "baseline_detection.sigma_multiplier": "nist_spc_handbook",
    "baseline_detection.consecutive_exceed": "western_electric_1956",
    "baseline_detection.fallback_median_float": "erratum_e003",
    "baseline_detection.fallback_median": "erratum_e003",
    "qc_thresholds.qc1_env_sensor_fault.fault_side": "erratum_e002",
    "smoothing.window_length": "savitzky_golay_1964",
    "smoothing.polyorder": "savitzky_golay_1964",
    "environmental_compensation.temperature_humidity_abs_pearson_r_threshold": "obrien_2007",
    "pipeline_b_features.t90_fraction_of_peak": "yan_2015",
    "pipeline_b_features.early_response_fraction": "yan_2015",
    "pipeline_b_features.late_mean_fraction": "yan_2015",
}


def _relative_to_project_root(path: Path) -> str:
    """Convert a path to a project-root-relative string when possible."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _ensure_nonempty_distribution(values: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    """Validate that a pooled thresholding distribution is one-dimensional and non-empty."""
    array = np.asarray(values, dtype=float).ravel()
    if array.size == 0:
        raise RuntimeError(f"{name} is empty under the prespecified cohort mask.")
    if not np.isfinite(array).all():
        raise RuntimeError(f"{name} contains non-finite values under the prespecified cohort mask.")
    return array


def _summarize_distribution(values: NDArray[np.float64]) -> dict[str, float]:
    """Return descriptive statistics for a one-dimensional finite distribution."""
    array = _ensure_nonempty_distribution(values, "distribution")
    return {
        "min": float(np.min(array)),
        "q1": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "q3": float(np.percentile(array, 75)),
        "max": float(np.max(array)),
    }


def _compute_qc4_response_amplitude_threshold(values: NDArray[np.float64]) -> tuple[float, str]:
    """Compute the Erratum E004 QC-4 threshold and stable schema method label.

    The discovery YAML/report continue to emit `threshold_method="tukey_lower_inner_fence"` so downstream
    consumers keep a stable field even though Erratum E004 removes the bimodality dispatch and makes Tukey's
    lower inner fence the unconditional QC-4 threshold extractor.
    """
    array = _ensure_nonempty_distribution(values, "response_amplitude pooled distribution")
    threshold = float(discovery.compute_tukey_lower_inner_fence(array, clip_at_zero=True))
    return threshold, "tukey_lower_inner_fence"


def _compute_qc5_saturation_threshold(values: NDArray[np.float64]) -> tuple[float, str]:
    """Compute the Erratum E004 QC-5 threshold with fallback and retained halt semantics."""
    array = _ensure_nonempty_distribution(values, "saturation_fraction pooled distribution")
    q1 = float(np.percentile(array, 25))
    q3 = float(np.percentile(array, 75))
    iqr = q3 - q1
    if iqr == 0.0:
        if np.all(array == array[0]) and array[0] > 0.0:
            raise RuntimeError(
                "The pooled saturation-fraction distribution is all identical at a value greater than 0.0; "
                "the retained QC-5 degenerate-distribution halt applies before the Erratum E004 fallback."
            )
        threshold = float(np.percentile(array, QC5_FALLBACK_PERCENTILE, method="linear"))
        return threshold, "p95_fallback"

    threshold = float(discovery.compute_tukey_upper_inner_fence(array))
    return threshold, "tukey_upper_inner_fence"


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Write the preprocessing YAML with informative path-aware errors."""
    try:
        with path.open("w", encoding="utf-8") as file_handle:
            yaml.safe_dump(payload, file_handle, default_flow_style=False, sort_keys=False)
    except OSError as exc:
        raise OSError(f"Failed to write preprocessing YAML at {path}: {exc}") from exc


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write the discovery report JSON with informative path-aware errors."""
    try:
        with path.open("w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2)
            file_handle.write("\n")
    except OSError as exc:
        raise OSError(f"Failed to write discovery report JSON at {path}: {exc}") from exc


def _setup_logging(log_path: Path) -> None:
    """Configure root logging for the discovery run.

    Args:
        log_path: Target log-file path for this run.
    """
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    run_start_iso = datetime.now(timezone.utc).astimezone().isoformat()
    metadata_path = METADATA_PATH if _CURRENT_METADATA_PATH is None else _CURRENT_METADATA_PATH
    breath_data_root = BREATH_DATA_ROOT if _CURRENT_BREATH_DATA_ROOT is None else _CURRENT_BREATH_DATA_ROOT
    LOGGER.info("Discovery run start time=%s", run_start_iso)
    LOGGER.info("RANDOM_SEED=%s", RANDOM_SEED)
    LOGGER.info("Metadata path=%s", metadata_path.resolve())
    LOGGER.info("Breath data root=%s", breath_data_root.resolve())
    LOGGER.info(_LABEL_BLINDING_STATEMENT)


def _run_pass_one_baseline_detection(records: list[RawRecord]) -> list[BaselineDetectionResult]:
    """Run the first-pass K-of-N baseline detection across all samples.

    Args:
        records: Loaded raw records aligned to metadata rows.

    Returns:
        One baseline-detection result per input record.
    """
    baseline_results = [detect_baseline_end(record) for record in records]
    n_triggered = sum(not result.fallback_used for result in baseline_results)
    n_fallback = sum(result.fallback_used for result in baseline_results)
    LOGGER.info(
        "Pass-1 baseline detection complete: n_triggered=%s n_fallback=%s",
        n_triggered,
        n_fallback,
    )
    return baseline_results


def _assign_pass_two_fallback(
    baseline_results: list[BaselineDetectionResult], fallback_median: int
) -> list[int]:
    """Resolve pass-two baseline ends using the prespecified cohort fallback.

    Args:
        baseline_results: First-pass detection outputs.
        fallback_median: Integer floor-coerced fallback median per Erratum E003.

    Returns:
        Resolved `t_baseline_end` integers aligned to `baseline_results`.
    """
    resolved_baseline_ends: list[int] = []
    n_assigned_fallback = 0
    for result in baseline_results:
        if result.fallback_used:
            resolved_baseline_ends.append(fallback_median)
            n_assigned_fallback += 1
            continue
        if result.t_baseline_end is None:
            raise RuntimeError("Encountered a non-fallback baseline result without `t_baseline_end`.")
        resolved_baseline_ends.append(int(result.t_baseline_end))

    LOGGER.info(
        "Pass-2 fallback assignment complete: fallback_median=%s n_assigned_fallback=%s",
        fallback_median,
        n_assigned_fallback,
    )
    return resolved_baseline_ends


def _compute_baseline_summaries(
    records: list[RawRecord],
    t_baseline_end_resolved: list[int],
    smoothed_gas_signals: list[NDArray[np.float64]],
) -> dict[str, NDArray[np.float64] | NDArray[np.int64]]:
    """Compute per-sample summaries required by QC, regression, and trimming.

    Args:
        records: Loaded raw records aligned to metadata rows.
        t_baseline_end_resolved: Resolved integer baseline-end indices.
        smoothed_gas_signals: Smoothed gas-channel matrices aligned to `records`.

    Returns:
        A dictionary of aligned per-sample arrays used by the discovery workflow.

    Raises:
        ValueError: If the aligned collections do not share the same length or a
            resolved baseline index falls outside the valid half-open interval.
    """
    if not (len(records) == len(t_baseline_end_resolved) == len(smoothed_gas_signals)):
        raise ValueError("records, t_baseline_end_resolved, and smoothed_gas_signals must have the same length.")

    n_samples = len(records)
    n_channels = len(GAS_CHANNELS)
    s12_baseline_mean = np.empty(n_samples, dtype=np.float64)
    t_sample = np.empty(n_samples, dtype=np.float64)
    h_sample = np.empty(n_samples, dtype=np.float64)
    gas_baseline_mean = np.empty((n_samples, n_channels), dtype=np.float64)
    baseline_cv = np.empty((n_samples, n_channels), dtype=np.float64)
    response_amplitude = np.empty((n_samples, n_channels), dtype=np.float64)
    saturation_fraction = np.empty((n_samples, n_channels), dtype=np.float64)
    spike_count = np.empty((n_samples, n_channels), dtype=np.int64)
    n_available = np.empty(n_samples, dtype=np.int64)

    for sample_index, (record, t_baseline_end, smoothed_matrix) in enumerate(
        zip(records, t_baseline_end_resolved, smoothed_gas_signals, strict=True)
    ):
        if t_baseline_end <= 0 or t_baseline_end >= record.n_rows:
            raise ValueError(
                f"Resolved t_baseline_end={t_baseline_end} is invalid for sample_id={record.sample_id} "
                f"with n_rows={record.n_rows}."
            )
        if smoothed_matrix.shape != (record.n_rows, len(GAS_CHANNELS)):
            raise ValueError(
                f"Smoothed gas matrix shape mismatch for sample_id={record.sample_id}: "
                f"expected {(record.n_rows, len(GAS_CHANNELS))}, got {smoothed_matrix.shape}."
            )

        raw_gas = np.asarray(record.gas_signals, dtype=np.float64)
        baseline_raw = raw_gas[:t_baseline_end, :]
        response_raw = raw_gas[t_baseline_end:, :]
        temperature_baseline = np.asarray(record.temperature[:t_baseline_end], dtype=np.float64)
        humidity_baseline = np.asarray(record.humidity[:t_baseline_end], dtype=np.float64)
        smoothed_baseline = np.asarray(smoothed_matrix[:t_baseline_end, :], dtype=np.float64)

        channel_baseline_mean = np.mean(baseline_raw, axis=0)
        channel_baseline_std = np.std(baseline_raw, axis=0)
        baseline_denominator = np.where(np.abs(channel_baseline_mean) < 1e-8, 1.0, np.abs(channel_baseline_mean))
        if np.any(np.abs(channel_baseline_mean) < 1e-8):
            LOGGER.warning(
                "Clamping baseline CV denominator to 1.0 for sample_id=%s zero_near_zero_channels=%s",
                record.sample_id,
                int(np.count_nonzero(np.abs(channel_baseline_mean) < 1e-8)),
            )

        delta_max = np.max(response_raw, axis=0) - channel_baseline_mean
        delta_min = np.min(response_raw, axis=0) - channel_baseline_mean
        signed_delta = np.where(np.abs(delta_max) >= np.abs(delta_min), delta_max, delta_min)
        response_denominator = np.where(np.abs(channel_baseline_mean) < 1e-8, 1.0, np.abs(channel_baseline_mean))

        channel_maxima = np.max(raw_gas, axis=0)
        saturation_mask = np.abs(raw_gas - channel_maxima) <= 1.0

        spike_counts = np.empty(n_channels, dtype=np.int64)
        for channel_index in range(n_channels):
            channel_differences = np.abs(np.diff(raw_gas[:, channel_index]))
            spike_threshold = discovery.compute_tukey_upper_inner_fence(channel_differences)
            spike_counts[channel_index] = np.int64(np.count_nonzero(channel_differences > spike_threshold))

        s12_baseline_mean[sample_index] = float(np.mean(np.asarray(record.temperature[:T_BASELINE_WINDOW_INIT], dtype=float)))
        t_sample[sample_index] = float(np.mean(temperature_baseline))
        h_sample[sample_index] = float(np.mean(humidity_baseline))
        gas_baseline_mean[sample_index, :] = np.mean(smoothed_baseline, axis=0)
        baseline_cv[sample_index, :] = channel_baseline_std / baseline_denominator
        response_amplitude[sample_index, :] = signed_delta / response_denominator
        saturation_fraction[sample_index, :] = np.mean(saturation_mask, axis=0)
        spike_count[sample_index, :] = spike_counts
        n_available[sample_index] = np.int64(record.n_rows - t_baseline_end)

    LOGGER.info("Computed baseline summaries for n_samples=%s.", n_samples)
    return {
        "s12_baseline_mean": s12_baseline_mean,
        "t_sample": t_sample,
        "h_sample": h_sample,
        "gas_baseline_mean": gas_baseline_mean,
        "baseline_cv": baseline_cv,
        "response_amplitude": response_amplitude,
        "saturation_fraction": saturation_fraction,
        "spike_count": spike_count,
        "n_available": n_available,
    }


def _compute_qc1_status(
    s12_baseline_mean: NDArray[np.float64],
) -> tuple[bool, float | None, str | None, NDArray[np.bool_]]:
    """Compute QC-1 enablement, threshold, fault side, and internal pass mask.

    Args:
        s12_baseline_mean: Cohort-wide early-baseline S12 means.

    Returns:
        Tuple of `(qc1_enabled, qc1_threshold, qc1_fault_side, qc1_pass_internal)`.
    """
    qc1_enabled = discovery.detect_bimodality_kde(s12_baseline_mean)
    if not qc1_enabled:
        LOGGER.info("QC-1 disabled: s12_baseline_mean distribution is unimodal under prespecified KDE rule")
        return False, None, None, np.ones(s12_baseline_mean.shape[0], dtype=bool)

    qc1_threshold = float(discovery.compute_otsu_threshold(s12_baseline_mean))
    n_low = int(np.count_nonzero(s12_baseline_mean < qc1_threshold))
    n_high = int(np.count_nonzero(s12_baseline_mean >= qc1_threshold))
    qc1_fault_side = "low" if n_low < n_high else "high"
    if qc1_fault_side == "low":
        qc1_pass_internal = np.asarray(s12_baseline_mean >= qc1_threshold, dtype=bool)
    else:
        qc1_pass_internal = np.asarray(s12_baseline_mean < qc1_threshold, dtype=bool)

    LOGGER.info(
        "QC-1 enabled under prespecified KDE rule: threshold=%.6f fault_side=%s n_low=%s n_high=%s n_failing=%s",
        qc1_threshold,
        qc1_fault_side,
        n_low,
        n_high,
        int(np.count_nonzero(~qc1_pass_internal)),
    )
    return True, qc1_threshold, qc1_fault_side, qc1_pass_internal


def _compute_qc2_status(records: list[RawRecord]) -> NDArray[np.bool_]:
    """Compute the internal QC-2 pass mask used for downstream threshold scoping.

    Args:
        records: Loaded raw records aligned to metadata rows.

    Returns:
        Boolean mask where `True` indicates `n_rows >= MIN_ROWS_QC2`.
    """
    qc2_pass_internal = np.asarray([record.n_rows >= MIN_ROWS_QC2 for record in records], dtype=bool)
    LOGGER.info("QC-2 internal mask computed: n_failing=%s", int(np.count_nonzero(~qc2_pass_internal)))
    return qc2_pass_internal


def main(
    metadata_path: Path | None = None,
    breath_data_root: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    """Execute the full label-blind discovery orchestration workflow.

    Args:
        metadata_path: Optional metadata CSV override. Defaults to
            `src.config.METADATA_PATH`.
        breath_data_root: Optional raw breath-data directory override. Defaults
            to `src.config.BREATH_DATA_ROOT`.
        output_dir: Optional outputs root override. Defaults to the parent of
            `src.config.OUTPUTS_LOGS_ROOT`.

    Raises:
        RuntimeError: If any prespecified cohort mask becomes empty, if no
            pass-one K-of-N triggers exist, or if QC-5 encounters the
            zero-variance nonzero saturation halt condition.
        OSError: If writing the YAML or JSON artifact fails.
        ValueError: If aligned inputs violate the expected shapes.
    """
    metadata_path_resolved = METADATA_PATH if metadata_path is None else Path(metadata_path)
    breath_data_root_resolved = BREATH_DATA_ROOT if breath_data_root is None else Path(breath_data_root)
    output_dir_resolved = OUTPUTS_LOGS_ROOT.parent if output_dir is None else Path(output_dir)

    output_dir_resolved.mkdir(parents=True, exist_ok=True)
    OUTPUTS_LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIGS_ROOT.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_ROOT.mkdir(parents=True, exist_ok=True)

    global _CURRENT_METADATA_PATH
    global _CURRENT_BREATH_DATA_ROOT
    _CURRENT_METADATA_PATH = metadata_path_resolved
    _CURRENT_BREATH_DATA_ROOT = breath_data_root_resolved

    _setup_logging(OUTPUTS_LOGS_ROOT / "discovery.log")

    metadata = load_metadata(metadata_path_resolved)
    records = load_dataset(metadata=metadata, breath_data_root=breath_data_root_resolved)

    baseline_results = _run_pass_one_baseline_detection(records)
    triggered_baseline_ends = np.asarray(
        [float(result.t_baseline_end) for result in baseline_results if not result.fallback_used and result.t_baseline_end is not None],
        dtype=np.float64,
    )
    if triggered_baseline_ends.size == 0:
        raise RuntimeError("K-of-N baseline detection produced zero triggered samples; fallback median is undefined.")

    fallback_median_float = float(discovery.compute_kofn_fallback_median(triggered_baseline_ends))
    fallback_median: int = int(np.floor(fallback_median_float))
    LOGGER.info(
        "K-of-N fallback median: float=%.6f int_floor=%d (per Erratum E003)",
        fallback_median_float,
        fallback_median,
    )
    t_baseline_end_resolved = _assign_pass_two_fallback(baseline_results, fallback_median)

    smoothed_gas_signals = [apply_sg_smoothing(record.gas_signals) for record in records]
    summaries = _compute_baseline_summaries(records, t_baseline_end_resolved, smoothed_gas_signals)

    s12_baseline_mean = np.asarray(summaries["s12_baseline_mean"], dtype=np.float64)
    qc1_enabled, qc1_threshold, qc1_fault_side, qc1_pass_internal = _compute_qc1_status(s12_baseline_mean)
    qc2_pass_internal = _compute_qc2_status(records)

    mask_qc1 = qc1_pass_internal
    mask_qc1_qc2 = np.asarray(qc1_pass_internal & qc2_pass_internal, dtype=bool)
    if int(np.count_nonzero(mask_qc1)) == 0:
        raise RuntimeError("QC-1-passing cohort is empty; QC-3 threshold cannot be computed.")
    if int(np.count_nonzero(mask_qc1_qc2)) == 0:
        raise RuntimeError("QC-1/QC-2-passing cohort is empty; downstream thresholds cannot be computed.")

    baseline_cv_distribution = _ensure_nonempty_distribution(
        np.asarray(summaries["baseline_cv"], dtype=np.float64)[mask_qc1, :].ravel(),
        "baseline_cv pooled distribution",
    )
    baseline_cv_threshold = float(discovery.compute_tukey_upper_inner_fence(baseline_cv_distribution))

    response_amplitude_abs_distribution = _ensure_nonempty_distribution(
        np.abs(np.asarray(summaries["response_amplitude"], dtype=np.float64)[mask_qc1_qc2, :]).ravel(),
        "response_amplitude pooled distribution",
    )
    response_amplitude_threshold, response_amplitude_threshold_method = _compute_qc4_response_amplitude_threshold(
        response_amplitude_abs_distribution
    )
    LOGGER.info(
        "QC-4 threshold extraction complete: threshold=%.6f method=%s",
        response_amplitude_threshold,
        response_amplitude_threshold_method,
    )

    saturation_fraction_distribution = _ensure_nonempty_distribution(
        np.asarray(summaries["saturation_fraction"], dtype=np.float64)[mask_qc1_qc2, :].ravel(),
        "saturation_fraction pooled distribution",
    )
    saturation_threshold, saturation_threshold_method = _compute_qc5_saturation_threshold(
        saturation_fraction_distribution
    )
    LOGGER.info(
        "QC-5 threshold extraction complete: threshold=%.6f method=%s",
        saturation_threshold,
        saturation_threshold_method,
    )

    spike_count_distribution = _ensure_nonempty_distribution(
        np.asarray(summaries["spike_count"], dtype=np.float64)[mask_qc1_qc2, :].ravel(),
        "spike_count pooled distribution",
    )
    spike_count_threshold = float(discovery.compute_tukey_upper_inner_fence(spike_count_distribution))

    t_sample_masked = np.asarray(summaries["t_sample"], dtype=np.float64)[mask_qc1_qc2]
    h_sample_masked = np.asarray(summaries["h_sample"], dtype=np.float64)[mask_qc1_qc2]
    gas_baseline_mean_masked = np.asarray(summaries["gas_baseline_mean"], dtype=np.float64)[mask_qc1_qc2, :]
    fitting_sample_count = int(t_sample_masked.shape[0])
    if fitting_sample_count < 2:
        raise RuntimeError("Environmental compensation requires at least two QC-1/QC-2-passing samples.")

    temperature_humidity_pearson_r = float(discovery.compute_pearson_correlation(t_sample_masked, h_sample_masked))
    multicollinearity_fallback_used = abs(temperature_humidity_pearson_r) > _TEMPERATURE_HUMIDITY_ABS_R_THRESHOLD

    t_ref = float(np.median(t_sample_masked))
    h_ref = float(np.median(h_sample_masked))
    environmental_models: dict[str, dict[str, float | bool]] = {}
    for channel_index, channel_name in enumerate(GAS_CHANNELS):
        channel_baseline = gas_baseline_mean_masked[:, channel_index]
        if multicollinearity_fallback_used:
            model = discovery.select_univariate_fallback(channel_baseline, t_sample_masked, h_sample_masked)
        else:
            model = discovery.fit_environmental_regression(channel_baseline, t_sample_masked, h_sample_masked)

        applies_compensation = bool(
            model["adj_r2"] > _ADJUSTED_R2_THRESHOLD
            and np.isfinite(model["alpha"])
            and np.isfinite(model["beta_T"])
            and np.isfinite(model["beta_H"])
        )
        environmental_models[channel_name] = {
            "alpha": float(model["alpha"]),
            "beta_T": float(model["beta_T"]),
            "beta_H": float(model["beta_H"]),
            "adj_r2": float(model["adj_r2"]),
            "applies_compensation": applies_compensation,
        }

    n_available_masked = np.asarray(summaries["n_available"], dtype=np.float64)[mask_qc1_qc2]
    clip_length_l = int(discovery.compute_clip_length(n_available_masked))
    short_mask = np.asarray(
        mask_qc1_qc2 & (np.asarray(summaries["n_available"], dtype=np.int64) < clip_length_l),
        dtype=bool,
    )
    if int(np.count_nonzero(short_mask)) == 0:
        padding_relative_deficits = np.asarray([], dtype=np.float64)
    else:
        padding_relative_deficits = np.asarray(
            [
                (clip_length_l - int(n_available_value)) / clip_length_l
                for n_available_value in np.asarray(summaries["n_available"], dtype=np.int64)[short_mask]
            ],
            dtype=np.float64,
        )
    padding_relative_deficit_threshold = discovery.compute_padding_deficit_threshold(padding_relative_deficits)

    preprocessing_yaml_path = CONFIGS_ROOT / "preprocessing.yaml"
    discovery_report_path = PROCESSED_DATA_ROOT / "discovery_report.json"
    yaml_payload: dict[str, Any] = {
        "provenance_schema_version": 1,
        "random_seed": RANDOM_SEED,
        "channels": {
            "dropped": S0_DROPPED,
            "gas": list(GAS_CHANNELS),
            "temperature": TEMPERATURE_CHANNEL,
            "humidity": HUMIDITY_CHANNEL,
        },
        "baseline_detection": {
            "t_baseline_window_init": T_BASELINE_WINDOW_INIT,
            "t_baseline_end_min": T_BASELINE_END_MIN,
            "t_baseline_end_max": T_BASELINE_END_MAX,
            "sigma_multiplier": BASELINE_SIGMA_MULTIPLIER,
            "consecutive_exceed": BASELINE_CONSECUTIVE_EXCEED,
            "k_of_n": BASELINE_K_OF_N,
            "fallback_median_float": fallback_median_float,
            "fallback_median": fallback_median,
        },
        "qc_thresholds": {
            "qc1_env_sensor_fault": {
                "env_sensor_baseline_points": T_BASELINE_WINDOW_INIT,
                "enabled": qc1_enabled,
                "threshold": qc1_threshold,
                "fault_side": qc1_fault_side,
            },
            "qc2_acquisition_truncated": {
                "min_rows": MIN_ROWS_QC2,
            },
            "qc3_unstable_baseline": {
                "threshold": baseline_cv_threshold,
            },
            "qc4_insufficient_response": {
                "threshold": response_amplitude_threshold,
                "threshold_method": response_amplitude_threshold_method,
            },
            "qc5_sensor_saturation": {
                "threshold": saturation_threshold,
                "threshold_method": saturation_threshold_method,
            },
            "qc6_excessive_spikes": {
                "threshold": spike_count_threshold,
            },
        },
        "smoothing": {
            "window_length": SG_WINDOW_LENGTH,
            "polyorder": SG_POLYORDER,
        },
        "environmental_compensation": {
            "temperature_humidity_pearson_r": temperature_humidity_pearson_r,
            "temperature_humidity_abs_pearson_r_threshold": _TEMPERATURE_HUMIDITY_ABS_R_THRESHOLD,
            "multicollinearity_fallback_used": multicollinearity_fallback_used,
            "r2_threshold": _ADJUSTED_R2_THRESHOLD,
            "t_ref": t_ref,
            "h_ref": h_ref,
            "per_channel": environmental_models,
        },
        "correction_and_trimming": {
            "fractional_correction_formula": "x_corrected_i(t) = (x_compensated_i(t) - S0_i) / S0_i",
            "s0_warning_abs_threshold": _S0_WARNING_ABS_THRESHOLD,
            "clip_length_L": clip_length_l,
            "padding_relative_deficit_threshold": padding_relative_deficit_threshold,
        },
        "pipeline_b_features": {
            "feature_names": _PIPELINE_B_FEATURE_NAMES,
            "t90_fraction_of_peak": _T90_FRACTION_OF_PEAK,
            "early_response_fraction": _EARLY_RESPONSE_FRACTION,
            "late_mean_fraction": _LATE_MEAN_FRACTION,
        },
        "provenance": {
            "labels": _L17_PROVENANCE,
            "citations": _L17_CITATIONS,
        },
    }
    _write_yaml(preprocessing_yaml_path, yaml_payload)

    discovery_report_payload: dict[str, Any] = {
        "run_metadata": {
            "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
            "code_version": None,
            "seed": RANDOM_SEED,
            "input_path_summaries": {
                "metadata_path": _relative_to_project_root(metadata_path_resolved),
                "breath_data_root": _relative_to_project_root(breath_data_root_resolved),
                "preprocessing_yaml_path": _relative_to_project_root(preprocessing_yaml_path),
                "discovery_report_path": _relative_to_project_root(discovery_report_path),
            },
        },
        "label_blinding_statement": _LABEL_BLINDING_STATEMENT,
        "baseline_detection": {
            "total_samples": len(records),
            "n_triggered": int(triggered_baseline_ends.shape[0]),
            "n_fallback": int(sum(result.fallback_used for result in baseline_results)),
            "fallback_median_float": fallback_median_float,
            "fallback_median": fallback_median,
            "note": "Integer coercion via floor per Erratum E003",
        },
        "qc1_env_sensor_fault": {
            "enabled": qc1_enabled,
            "threshold": qc1_threshold,
            "fault_side": qc1_fault_side,
            "note": "Fault-side direction resolved per Erratum E002",
            "n_passing_internal": int(np.count_nonzero(qc1_pass_internal)),
            "n_failing_internal": int(np.count_nonzero(~qc1_pass_internal)),
            "distribution_summary": _summarize_distribution(s12_baseline_mean),
        },
        "qc3_baseline_cv": {
            "threshold": baseline_cv_threshold,
            "distribution_summary": _summarize_distribution(baseline_cv_distribution),
        },
        "qc4_response_amplitude": {
            "threshold": response_amplitude_threshold,
            "threshold_method": response_amplitude_threshold_method,
            "distribution_summary": _summarize_distribution(response_amplitude_abs_distribution),
        },
        "qc5_saturation": {
            "threshold": saturation_threshold,
            "threshold_method": saturation_threshold_method,
            "distribution_summary": _summarize_distribution(saturation_fraction_distribution),
        },
        "qc6_spikes": {
            "threshold": spike_count_threshold,
            "distribution_summary": _summarize_distribution(spike_count_distribution),
        },
        "environmental_compensation": {
            "fitting_sample_count": fitting_sample_count,
            "temperature_humidity_pearson_r": temperature_humidity_pearson_r,
            "abs_pearson_r_threshold": _TEMPERATURE_HUMIDITY_ABS_R_THRESHOLD,
            "multicollinearity_fallback_used": multicollinearity_fallback_used,
            "r2_threshold": _ADJUSTED_R2_THRESHOLD,
            "t_ref": t_ref,
            "h_ref": h_ref,
            "per_channel": environmental_models,
        },
        "trimming": {
            "clip_length_L": clip_length_l,
            "padding_relative_deficit_threshold": padding_relative_deficit_threshold,
            "n_shorter_than_L": int(np.count_nonzero(short_mask)),
        },
        "halt_or_warning_events": [],
    }
    _write_json(discovery_report_path, discovery_report_payload)

    LOGGER.info(
        "Discovery summary: total_samples=%s n_triggered=%s n_fallback=%s qc1_enabled=%s qc1_fault_side=%s "
        "n_qc1_pass_internal=%s n_qc2_pass_internal=%s n_mask_qc1=%s n_mask_qc1_qc2=%s "
        "multicollinearity_fallback_used=%s clip_length_L=%s padding_relative_deficit_threshold=%s",
        len(records),
        int(triggered_baseline_ends.shape[0]),
        int(sum(result.fallback_used for result in baseline_results)),
        qc1_enabled,
        qc1_fault_side,
        int(np.count_nonzero(qc1_pass_internal)),
        int(np.count_nonzero(qc2_pass_internal)),
        int(np.count_nonzero(mask_qc1)),
        int(np.count_nonzero(mask_qc1_qc2)),
        multicollinearity_fallback_used,
        clip_length_l,
        "null" if padding_relative_deficit_threshold is None else f"{padding_relative_deficit_threshold:.6f}",
    )
    LOGGER.info(
        "Discovery run complete; thresholds written. No final QC exclusion decisions were made by this script "
        "(Phase 4 will apply all 6 QC checks to determine in_final_output per sample)."
    )


if __name__ == "__main__":
    main()
