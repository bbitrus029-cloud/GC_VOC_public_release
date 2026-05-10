"""Unit tests for prespecified preprocessing quality-control checks."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import pytest

from src.data.baseline import BaselineDetectionResult
from src.data.loader import RawRecord


def _import_qc_symbols() -> tuple[Any, ...]:
    """Import QC symbols lazily so pre-implementation failures stay clear."""
    try:
        from src.config import GAS_CHANNELS, N_GAS_CHANNELS
        from src.data.qc import QCResult, qc_diagnostics_to_df, qc_summary_to_df, run_qc
    except (AttributeError, ImportError, ModuleNotFoundError) as exc:
        pytest.fail(f"QC implementation is missing: {exc}", pytrace=False)

    return QCResult, qc_diagnostics_to_df, qc_summary_to_df, run_qc, GAS_CHANNELS, N_GAS_CHANNELS


def _build_thresholds(
    *,
    env_sensor_baseline_points: int = 5,
    min_rows: int = 1000,
    qc3_threshold: float = 10.0,
    qc4_threshold: float = 0.25,
    qc5_threshold: float = 1.1,
    qc6_threshold: float = 50.0,
) -> dict[str, dict[str, float | int]]:
    """Build a synthetic QC threshold dictionary without reading YAML."""
    return {
        "qc1_env_sensor_fault": {
            "env_sensor_baseline_points": env_sensor_baseline_points,
            "threshold": 10.0,
        },
        "qc2_acquisition_truncated": {"min_rows": min_rows},
        "qc3_unstable_baseline": {"threshold": qc3_threshold},
        "qc4_insufficient_response": {"threshold": qc4_threshold},
        "qc5_sensor_saturation": {"threshold": qc5_threshold},
        "qc6_excessive_spikes": {"threshold": qc6_threshold},
    }


def _build_temperature_trace(
    n_rows: int,
    *,
    baseline_values: NDArray[np.float64] | None = None,
    response_value: float = 22.0,
) -> NDArray[np.float64]:
    """Build an environmental trace with a controllable first-five-point baseline."""
    trace = np.full(n_rows, response_value, dtype=np.float64)
    if baseline_values is not None:
        trace[: baseline_values.size] = baseline_values
    return trace


def _build_triangle_channel(
    n_rows: int,
    baseline_rows: int,
    *,
    baseline_value: float,
    peak_value: float,
    end_value: float | None = None,
) -> NDArray[np.float64]:
    """Build one smooth response trace with a narrow maximum and no flat top."""
    if baseline_rows <= 0 or baseline_rows >= n_rows:
        raise ValueError("baseline_rows must be strictly between 0 and n_rows.")

    response_rows = n_rows - baseline_rows
    half = response_rows // 2
    if end_value is None:
        end_value = baseline_value + 0.2 * (peak_value - baseline_value)

    rise = np.linspace(baseline_value, peak_value, half, endpoint=False, dtype=np.float64)
    fall = np.linspace(peak_value, end_value, response_rows - half, dtype=np.float64)
    return np.concatenate([np.full(baseline_rows, baseline_value, dtype=np.float64), rise, fall])


def _build_pass_matrix(
    n_rows: int,
    baseline_rows: int,
    n_gas_channels: int,
) -> NDArray[np.float64]:
    """Build a deterministic gas matrix that passes all QC checks with lenient thresholds."""
    channels = []
    for channel_index in range(n_gas_channels):
        baseline_value = 100.0 + float(channel_index)
        peak_value = baseline_value + 100.0 + 5.0 * float(channel_index)
        channels.append(
            _build_triangle_channel(
                n_rows=n_rows,
                baseline_rows=baseline_rows,
                baseline_value=baseline_value,
                peak_value=peak_value,
            )
        )
    return np.column_stack(channels)


def _build_raw_record(
    gas_signals: NDArray[np.float64],
    *,
    sample_id: int = 101,
    temperature: NDArray[np.float64] | None = None,
    humidity: NDArray[np.float64] | None = None,
) -> RawRecord:
    """Construct a synthetic raw record directly from deterministic arrays."""
    n_rows = int(gas_signals.shape[0])
    s0 = np.zeros(n_rows, dtype=np.int64)
    if temperature is None:
        temperature = _build_temperature_trace(n_rows, baseline_values=np.full(5, 20.0, dtype=np.float64))
    if humidity is None:
        humidity = _build_temperature_trace(n_rows, baseline_values=np.full(5, 800.0, dtype=np.float64), response_value=805.0)

    return RawRecord(
        sample_id=sample_id,
        filename="synthetic.csv",
        n_rows=n_rows,
        s0=s0,
        gas_signals=np.asarray(gas_signals, dtype=np.float64),
        temperature=np.asarray(temperature, dtype=np.float64),
        humidity=np.asarray(humidity, dtype=np.float64),
        label=1,
        disease_category=4,
        collection_time="2026-01-01 08:00:00",
    )


def _build_baseline_result(
    t_baseline_end: int,
    n_gas_channels: int,
    *,
    fallback_used: bool = False,
) -> BaselineDetectionResult:
    """Build a resolved baseline result for QC tests."""
    return BaselineDetectionResult(
        t_baseline_end=t_baseline_end,
        fallback_used=fallback_used,
        per_channel_onsets=np.full(n_gas_channels, t_baseline_end, dtype=np.int64),
        n_triggered_channels=n_gas_channels,
    )


def test_run_qc_fails_qc1_for_low_fault_side_and_short_circuits_diagnostics() -> None:
    """QC-1 low-side failures should short-circuit QC-2 through QC-6 diagnostics."""
    (
        _QCResult,
        _qc_diagnostics_to_df,
        _qc_summary_to_df,
        run_qc,
        _GAS_CHANNELS,
        N_GAS_CHANNELS,
    ) = _import_qc_symbols()
    n_rows = 1000
    baseline_rows = 10
    gas_signals = _build_pass_matrix(n_rows=n_rows, baseline_rows=baseline_rows, n_gas_channels=N_GAS_CHANNELS)
    temperature = _build_temperature_trace(n_rows, baseline_values=np.full(5, 5.0, dtype=np.float64))
    record = _build_raw_record(gas_signals, temperature=temperature)
    baseline_result = _build_baseline_result(t_baseline_end=baseline_rows, n_gas_channels=N_GAS_CHANNELS)

    result = run_qc(
        record=record,
        baseline_result=baseline_result,
        gas_signals=gas_signals,
        env_temperature=record.temperature,
        env_humidity=record.humidity,
        thresholds=_build_thresholds(),
        qc1_enabled=True,
        qc1_fault_side="low",
    )

    assert result.qc1_env_sensor_fault_pass is False
    assert result.fail_reasons == ["qc1_env_sensor_fault"]
    assert result.qc2_acquisition_truncated_pass is None
    assert result.qc3_unstable_baseline_pass is None
    assert result.qc4_insufficient_response_pass is None
    assert result.qc5_sensor_saturation_pass is None
    assert result.qc6_excessive_spikes_pass is None
    assert np.isnan(result.baseline_cvs).all()
    assert np.isnan(result.response_amplitudes).all()
    assert np.isnan(result.saturation_fractions).all()
    assert np.isnan(result.spike_counts).all()


def test_run_qc_uses_fault_side_high_branch_for_same_temperature_baseline() -> None:
    """The same S12 baseline mean should pass or fail depending on the configured fault side."""
    (
        _QCResult,
        _qc_diagnostics_to_df,
        _qc_summary_to_df,
        run_qc,
        _GAS_CHANNELS,
        N_GAS_CHANNELS,
    ) = _import_qc_symbols()
    n_rows = 1000
    baseline_rows = 10
    gas_signals = _build_pass_matrix(n_rows=n_rows, baseline_rows=baseline_rows, n_gas_channels=N_GAS_CHANNELS)
    temperature = _build_temperature_trace(n_rows, baseline_values=np.full(5, 12.0, dtype=np.float64))
    record = _build_raw_record(gas_signals, temperature=temperature)
    baseline_result = _build_baseline_result(t_baseline_end=baseline_rows, n_gas_channels=N_GAS_CHANNELS)
    thresholds = _build_thresholds()

    low_side_result = run_qc(
        record=record,
        baseline_result=baseline_result,
        gas_signals=gas_signals,
        env_temperature=record.temperature,
        env_humidity=record.humidity,
        thresholds=thresholds,
        qc1_enabled=True,
        qc1_fault_side="low",
    )
    high_side_result = run_qc(
        record=record,
        baseline_result=baseline_result,
        gas_signals=gas_signals,
        env_temperature=record.temperature,
        env_humidity=record.humidity,
        thresholds=thresholds,
        qc1_enabled=True,
        qc1_fault_side="high",
    )

    assert low_side_result.s12_baseline_mean == pytest.approx(high_side_result.s12_baseline_mean)
    assert low_side_result.qc1_env_sensor_fault_pass is True
    assert high_side_result.qc1_env_sensor_fault_pass is False


def test_run_qc_skips_qc1_when_disabled_and_evaluates_other_checks() -> None:
    """Disabling QC-1 should force QC-1 pass and still evaluate later checks."""
    (
        _QCResult,
        _qc_diagnostics_to_df,
        _qc_summary_to_df,
        run_qc,
        _GAS_CHANNELS,
        N_GAS_CHANNELS,
    ) = _import_qc_symbols()
    n_rows = 1000
    baseline_rows = 10
    gas_signals = _build_pass_matrix(n_rows=n_rows, baseline_rows=baseline_rows, n_gas_channels=N_GAS_CHANNELS)
    temperature = _build_temperature_trace(n_rows, baseline_values=np.full(5, 2.0, dtype=np.float64))
    record = _build_raw_record(gas_signals, temperature=temperature)
    baseline_result = _build_baseline_result(t_baseline_end=baseline_rows, n_gas_channels=N_GAS_CHANNELS)

    result = run_qc(
        record=record,
        baseline_result=baseline_result,
        gas_signals=gas_signals,
        env_temperature=record.temperature,
        env_humidity=record.humidity,
        thresholds=_build_thresholds(),
        qc1_enabled=False,
        qc1_fault_side="low",
    )

    assert result.qc1_env_sensor_fault_pass is True
    assert result.qc2_acquisition_truncated_pass is True
    assert result.qc3_unstable_baseline_pass is True
    assert result.qc4_insufficient_response_pass is True
    assert result.qc5_sensor_saturation_pass is True
    assert result.qc6_excessive_spikes_pass is True
    assert result.fail_reasons == []


def test_run_qc_fails_qc2_for_short_acquisition() -> None:
    """QC-2 should fail when the row count is below the configured minimum."""
    (
        _QCResult,
        _qc_diagnostics_to_df,
        _qc_summary_to_df,
        run_qc,
        _GAS_CHANNELS,
        N_GAS_CHANNELS,
    ) = _import_qc_symbols()
    n_rows = 999
    baseline_rows = 10
    gas_signals = _build_pass_matrix(n_rows=n_rows, baseline_rows=baseline_rows, n_gas_channels=N_GAS_CHANNELS)
    record = _build_raw_record(gas_signals)
    baseline_result = _build_baseline_result(t_baseline_end=baseline_rows, n_gas_channels=N_GAS_CHANNELS)

    result = run_qc(
        record=record,
        baseline_result=baseline_result,
        gas_signals=gas_signals,
        env_temperature=record.temperature,
        env_humidity=record.humidity,
        thresholds=_build_thresholds(min_rows=1000),
        qc1_enabled=False,
        qc1_fault_side="high",
    )

    assert result.qc2_acquisition_truncated_pass is False
    assert result.fail_reasons == ["qc2_acquisition_truncated"]
    assert result.n_rows == 999


def test_run_qc_fails_qc3_for_majority_unstable_channels_and_clamps_small_baseline_mean(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """QC-3 should fail when six channels exceed the baseline-CV threshold, even with zero means."""
    (
        _QCResult,
        _qc_diagnostics_to_df,
        _qc_summary_to_df,
        run_qc,
        _GAS_CHANNELS,
        N_GAS_CHANNELS,
    ) = _import_qc_symbols()
    n_rows = 1000
    baseline_rows = 10
    gas_signals = _build_pass_matrix(n_rows=n_rows, baseline_rows=baseline_rows, n_gas_channels=N_GAS_CHANNELS)
    alternating_baseline = np.array([-1.0, 1.0] * 5, dtype=np.float64)
    for channel_index in range(6):
        gas_signals[:baseline_rows, channel_index] = alternating_baseline
        gas_signals[baseline_rows:, channel_index] = _build_triangle_channel(
            n_rows=n_rows,
            baseline_rows=baseline_rows,
            baseline_value=0.0,
            peak_value=2.0,
        )[baseline_rows:]
    record = _build_raw_record(gas_signals)
    baseline_result = _build_baseline_result(t_baseline_end=baseline_rows, n_gas_channels=N_GAS_CHANNELS)

    with caplog.at_level(logging.WARNING):
        result = run_qc(
            record=record,
            baseline_result=baseline_result,
            gas_signals=gas_signals,
            env_temperature=record.temperature,
            env_humidity=record.humidity,
            thresholds=_build_thresholds(qc3_threshold=0.5, qc4_threshold=0.5),
            qc1_enabled=False,
            qc1_fault_side="high",
        )

    assert result.qc3_unstable_baseline_pass is False
    assert result.fail_reasons == ["qc3_unstable_baseline"]
    assert result.baseline_cvs[:6] == pytest.approx(np.ones(6))
    assert "QC-3 denominator" in caplog.text


def test_run_qc_fails_qc4_for_insufficient_response_and_clamps_small_baseline_mean(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """QC-4 should fail when fewer than three channels exceed the response threshold."""
    (
        _QCResult,
        _qc_diagnostics_to_df,
        _qc_summary_to_df,
        run_qc,
        _GAS_CHANNELS,
        N_GAS_CHANNELS,
    ) = _import_qc_symbols()
    n_rows = 1000
    baseline_rows = 10
    gas_signals = np.zeros((n_rows, N_GAS_CHANNELS), dtype=np.float64)
    for channel_index in range(N_GAS_CHANNELS):
        peak_value = 1.0 if channel_index < 2 else 0.5
        gas_signals[:, channel_index] = _build_triangle_channel(
            n_rows=n_rows,
            baseline_rows=baseline_rows,
            baseline_value=0.0,
            peak_value=peak_value,
        )
    record = _build_raw_record(gas_signals)
    baseline_result = _build_baseline_result(t_baseline_end=baseline_rows, n_gas_channels=N_GAS_CHANNELS)

    with caplog.at_level(logging.WARNING):
        result = run_qc(
            record=record,
            baseline_result=baseline_result,
            gas_signals=gas_signals,
            env_temperature=record.temperature,
            env_humidity=record.humidity,
            thresholds=_build_thresholds(qc3_threshold=0.5, qc4_threshold=0.75),
            qc1_enabled=False,
            qc1_fault_side="high",
        )

    assert result.qc4_insufficient_response_pass is False
    assert result.fail_reasons == ["qc4_insufficient_response"]
    assert np.count_nonzero(np.abs(result.response_amplitudes) >= 0.75) == 2
    assert "QC-4 denominator" in caplog.text


def test_run_qc_uses_passed_threshold_values_not_hardcoded() -> None:
    """Changing the supplied threshold dictionary should change the QC-4 outcome."""
    (
        _QCResult,
        _qc_diagnostics_to_df,
        _qc_summary_to_df,
        run_qc,
        _GAS_CHANNELS,
        N_GAS_CHANNELS,
    ) = _import_qc_symbols()
    n_rows = 1000
    baseline_rows = 10
    gas_signals = np.zeros((n_rows, N_GAS_CHANNELS), dtype=np.float64)
    for channel_index in range(N_GAS_CHANNELS):
        peak_value = 1.0 if channel_index < 2 else 0.5
        gas_signals[:, channel_index] = _build_triangle_channel(
            n_rows=n_rows,
            baseline_rows=baseline_rows,
            baseline_value=0.0,
            peak_value=peak_value,
        )
    record = _build_raw_record(gas_signals)
    baseline_result = _build_baseline_result(t_baseline_end=baseline_rows, n_gas_channels=N_GAS_CHANNELS)

    strict_result = run_qc(
        record=record,
        baseline_result=baseline_result,
        gas_signals=gas_signals,
        env_temperature=record.temperature,
        env_humidity=record.humidity,
        thresholds=_build_thresholds(qc3_threshold=0.5, qc4_threshold=0.75),
        qc1_enabled=False,
        qc1_fault_side="high",
    )
    lenient_result = run_qc(
        record=record,
        baseline_result=baseline_result,
        gas_signals=gas_signals,
        env_temperature=record.temperature,
        env_humidity=record.humidity,
        thresholds=_build_thresholds(qc3_threshold=0.5, qc4_threshold=0.4),
        qc1_enabled=False,
        qc1_fault_side="high",
    )

    assert strict_result.qc4_insufficient_response_pass is False
    assert lenient_result.qc4_insufficient_response_pass is True


def test_run_qc_fails_qc5_for_within_one_adc_unit_plateau() -> None:
    """QC-5 should use the channel-local maximum and a within-one-unit plateau fraction."""
    (
        _QCResult,
        _qc_diagnostics_to_df,
        _qc_summary_to_df,
        run_qc,
        _GAS_CHANNELS,
        N_GAS_CHANNELS,
    ) = _import_qc_symbols()
    n_rows = 1000
    baseline_rows = 10
    gas_signals = _build_pass_matrix(n_rows=n_rows, baseline_rows=baseline_rows, n_gas_channels=N_GAS_CHANNELS)
    channel = gas_signals[:, 0].copy()
    channel[baseline_rows:200] = np.linspace(100.0, 200.0, 200 - baseline_rows, dtype=np.float64)
    channel[210:510] = 199.5
    channel[510:] = np.linspace(198.0, 130.0, n_rows - 510, dtype=np.float64)
    gas_signals[:, 0] = channel
    record = _build_raw_record(gas_signals)
    baseline_result = _build_baseline_result(t_baseline_end=baseline_rows, n_gas_channels=N_GAS_CHANNELS)

    result = run_qc(
        record=record,
        baseline_result=baseline_result,
        gas_signals=gas_signals,
        env_temperature=record.temperature,
        env_humidity=record.humidity,
            thresholds=_build_thresholds(qc5_threshold=0.1, qc6_threshold=1000.0),
        qc1_enabled=False,
        qc1_fault_side="high",
    )

    assert result.qc5_sensor_saturation_pass is False
    assert result.fail_reasons == ["qc5_sensor_saturation"]
    assert result.saturation_fractions[0] > 0.25


def test_run_qc_fails_qc6_for_excessive_spikes_using_within_sample_tukey_rule() -> None:
    """QC-6 should count spike candidates from each channel's own first-difference fence."""
    (
        _QCResult,
        _qc_diagnostics_to_df,
        _qc_summary_to_df,
        run_qc,
        _GAS_CHANNELS,
        N_GAS_CHANNELS,
    ) = _import_qc_symbols()
    n_rows = 1000
    baseline_rows = 10
    gas_signals = _build_pass_matrix(n_rows=n_rows, baseline_rows=baseline_rows, n_gas_channels=N_GAS_CHANNELS)
    for spike_index in [120, 240, 360]:
        gas_signals[spike_index, 0] += 250.0
        gas_signals[spike_index + 1, 0] -= 250.0
    record = _build_raw_record(gas_signals)
    baseline_result = _build_baseline_result(t_baseline_end=baseline_rows, n_gas_channels=N_GAS_CHANNELS)

    result = run_qc(
        record=record,
        baseline_result=baseline_result,
        gas_signals=gas_signals,
        env_temperature=record.temperature,
        env_humidity=record.humidity,
        thresholds=_build_thresholds(qc6_threshold=4.0),
        qc1_enabled=False,
        qc1_fault_side="high",
    )

    assert result.qc6_excessive_spikes_pass is False
    assert result.fail_reasons == ["qc6_excessive_spikes"]
    assert result.spike_counts[0] >= 6.0


def test_run_qc_accumulates_multiple_failures_without_nan_diagnostics() -> None:
    """QC-2, QC-3, and QC-5 failures should accumulate when QC-1 passes."""
    (
        _QCResult,
        _qc_diagnostics_to_df,
        _qc_summary_to_df,
        run_qc,
        _GAS_CHANNELS,
        N_GAS_CHANNELS,
    ) = _import_qc_symbols()
    n_rows = 999
    baseline_rows = 10
    gas_signals = _build_pass_matrix(n_rows=n_rows, baseline_rows=baseline_rows, n_gas_channels=N_GAS_CHANNELS)
    alternating_baseline = np.array([-1.0, 1.0] * 5, dtype=np.float64)
    for channel_index in range(6):
        gas_signals[:baseline_rows, channel_index] = alternating_baseline
        gas_signals[baseline_rows:, channel_index] = _build_triangle_channel(
            n_rows=n_rows,
            baseline_rows=baseline_rows,
            baseline_value=0.0,
            peak_value=2.0,
        )[baseline_rows:]
    gas_signals[210:510, 0] = 199.5
    gas_signals[510:, 0] = np.linspace(198.0, 130.0, n_rows - 510, dtype=np.float64)
    record = _build_raw_record(gas_signals)
    baseline_result = _build_baseline_result(t_baseline_end=baseline_rows, n_gas_channels=N_GAS_CHANNELS)

    result = run_qc(
        record=record,
        baseline_result=baseline_result,
        gas_signals=gas_signals,
        env_temperature=record.temperature,
        env_humidity=record.humidity,
        thresholds=_build_thresholds(min_rows=1000, qc3_threshold=0.5, qc4_threshold=0.5, qc5_threshold=0.1),
        qc1_enabled=False,
        qc1_fault_side="high",
    )

    assert result.fail_reasons == [
        "qc2_acquisition_truncated",
        "qc3_unstable_baseline",
        "qc5_sensor_saturation",
    ]
    assert not np.isnan(result.baseline_cvs).all()
    assert not np.isnan(result.response_amplitudes).all()
    assert not np.isnan(result.saturation_fractions).all()


def test_qc_serializers_expand_summary_and_diagnostics_columns() -> None:
    """QC serializers should emit the summary schema and representative diagnostic groups."""
    (
        QCResult,
        qc_diagnostics_to_df,
        qc_summary_to_df,
        run_qc,
        GAS_CHANNELS,
        N_GAS_CHANNELS,
    ) = _import_qc_symbols()
    n_rows = 1000
    baseline_rows = 10
    gas_signals = _build_pass_matrix(n_rows=n_rows, baseline_rows=baseline_rows, n_gas_channels=N_GAS_CHANNELS)
    record = _build_raw_record(gas_signals)
    baseline_result = _build_baseline_result(t_baseline_end=baseline_rows, n_gas_channels=N_GAS_CHANNELS)
    result = run_qc(
        record=record,
        baseline_result=baseline_result,
        gas_signals=gas_signals,
        env_temperature=record.temperature,
        env_humidity=record.humidity,
        thresholds=_build_thresholds(),
        qc1_enabled=False,
        qc1_fault_side="high",
    )

    summary_df = qc_summary_to_df([result])
    diagnostics_df = qc_diagnostics_to_df([result])

    assert isinstance(result, QCResult)
    assert list(summary_df.columns) == [
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
    assert summary_df.loc[0, "qc_status"] == "pass"
    assert summary_df.loc[0, "fail_reasons"] == ""
    required_diagnostic_columns = {
        "sample_id",
        "n_rows",
        "t_baseline_end",
        "fallback_used",
        "s12_baseline_mean",
        "s13_baseline_mean",
        "env_temperature_sample",
        "env_humidity_sample",
        "n_available",
        "clip_length_L",
        "relative_deficit",
        "padded",
        f"{GAS_CHANNELS[0].lower()}_baseline_mean",
        f"{GAS_CHANNELS[0].lower()}_baseline_std",
        f"{GAS_CHANNELS[0].lower()}_baseline_cv",
        f"{GAS_CHANNELS[0].lower()}_response_amplitude",
        f"{GAS_CHANNELS[0].lower()}_saturation_fraction",
        f"{GAS_CHANNELS[0].lower()}_spike_count",
    }
    assert required_diagnostic_columns.issubset(set(diagnostics_df.columns))
    assert diagnostics_df.shape[0] == 1
    assert pd.isna(diagnostics_df.loc[0, "clip_length_L"])
