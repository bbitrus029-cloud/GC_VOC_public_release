"""Integration tests for the preprocessing orchestrator and entry wiring."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import pytest

from src.config import GAS_CHANNELS, RANDOM_SEED
from src.data.baseline import BaselineDetectionResult


SOURCE_METADATA_COLUMNS = [
    "sample_id",
    "collection_time",
    "collection_location(0outpatient1inpatient)",
    "disease_category(0HC1BGL2LGIN3HGIN4EGC5AGC)",
    "label",
    "filename",
    "age",
    "sex(0male1female)",
    "BMI",
    "smoking(0no1yes)",
    "drinking(0no1yes)",
    "first_degree_family_history_of_GC(0no1yes)",
    "Hp_Infection(0no1yes)",
    "CEA(0-5μg/L)",
    "CA199(0.1-37μ/ml)",
    "CA724(0.1-10μ/ml)",
]
RAW_COLUMNS = [f"S{index}" for index in range(14)]


def _import_preprocessing_symbols() -> tuple[Any, ...]:
    """Import orchestration symbols lazily so test-first failures stay clear."""
    try:
        import src.data.preprocessing as preprocessing_module
        from src.data.preprocessing import PreprocessingPipeline, PreprocessingResult
    except (AttributeError, ImportError, ModuleNotFoundError) as exc:
        pytest.fail(f"Preprocessing orchestration implementation is missing: {exc}", pytrace=False)

    return preprocessing_module, PreprocessingPipeline, PreprocessingResult


def _build_metadata_frame(sample_specs: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a synthetic metadata table with the loader's required source schema."""
    rows: list[dict[str, Any]] = []
    for offset, spec in enumerate(sample_specs):
        rows.append(
            {
                "sample_id": int(spec["sample_id"]),
                "collection_time": f"2026-01-{offset + 1:02d} 08:00:00",
                "collection_location(0outpatient1inpatient)": offset % 2,
                "disease_category(0HC1BGL2LGIN3HGIN4EGC5AGC)": int(spec.get("disease_category", (offset + 1) % 6)),
                "label": int(spec.get("label", offset % 2)),
                "filename": str(spec["filename"]),
                "age": 40 + offset,
                "sex(0male1female)": offset % 2,
                "BMI": 20.5 + offset,
                "smoking(0no1yes)": 0,
                "drinking(0no1yes)": 1,
                "first_degree_family_history_of_GC(0no1yes)": 0,
                "Hp_Infection(0no1yes)": 1,
                "CEA(0-5μg/L)": 1.2 + offset,
                "CA199(0.1-37μ/ml)": 2.3 + offset,
                "CA724(0.1-10μ/ml)": 3.4 + offset,
            }
        )
    return pd.DataFrame(rows, columns=SOURCE_METADATA_COLUMNS)


def _write_metadata_csv(path: Path, metadata_frame: pd.DataFrame) -> None:
    """Write a metadata fixture using the project's CSV encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_frame.to_csv(path, index=False, encoding="utf-8-sig")


def _build_temperature_trace(
    n_rows: int,
    *,
    baseline_value: float,
    response_value: float | None = None,
) -> NDArray[np.float64]:
    """Build an S12 trace with a controlled five-point baseline."""
    tail_value = baseline_value if response_value is None else response_value
    trace = np.full(n_rows, tail_value, dtype=np.float64)
    trace[:5] = np.full(5, baseline_value, dtype=np.float64)
    return trace


def _build_humidity_trace(
    n_rows: int,
    *,
    baseline_value: float = 800.0,
    response_value: float = 805.0,
) -> NDArray[np.float64]:
    """Build an S13 trace with a controlled five-point baseline."""
    trace = np.full(n_rows, response_value, dtype=np.float64)
    trace[:5] = np.full(5, baseline_value, dtype=np.float64)
    return trace


def _build_channel_signal(
    n_rows: int,
    *,
    onset: int,
    baseline_value: float,
    peak_value: float,
) -> NDArray[np.float64]:
    """Build one gas-channel trace with a stable baseline and a unique response peak."""
    signal = np.full(n_rows, baseline_value, dtype=np.float64)
    signal[:5] = np.array(
        [
            baseline_value,
            baseline_value + 1.0,
            baseline_value,
            baseline_value + 1.0,
            baseline_value,
        ],
        dtype=np.float64,
    )
    if onset < n_rows:
        response_rows = n_rows - onset
        rise_rows = max(1, response_rows // 2)
        fall_rows = response_rows - rise_rows
        response_end_value = baseline_value + 0.2 * (peak_value - baseline_value)
        rise = np.linspace(baseline_value + 10.0, peak_value, rise_rows, endpoint=False, dtype=np.float64)
        fall = np.linspace(peak_value, response_end_value, max(1, fall_rows), dtype=np.float64)
        response = np.concatenate([rise, fall[:fall_rows]]).astype(np.float64, copy=False)
        signal[onset:] = response[:response_rows]
    return signal


def _build_gas_matrix(
    n_rows: int,
    *,
    onsets: list[int],
    peak_scale: float = 4200.0,
) -> NDArray[np.float64]:
    """Build an `(n_rows, 11)` gas matrix with configurable per-channel onsets."""
    if len(onsets) != len(GAS_CHANNELS):
        raise ValueError(f"onsets must provide {len(GAS_CHANNELS)} channel indices.")

    channels: list[NDArray[np.float64]] = []
    for channel_index, onset in enumerate(onsets):
        baseline_value = 100.0 + 5.0 * float(channel_index)
        peak_value = baseline_value + peak_scale + 100.0 * float(channel_index)
        channels.append(
            _build_channel_signal(
                n_rows=n_rows,
                onset=onset,
                baseline_value=baseline_value,
                peak_value=peak_value,
            )
        )
    return np.column_stack(channels)


def _write_raw_csv(
    path: Path,
    *,
    gas_signals: NDArray[np.float64],
    temperature: NDArray[np.float64],
    humidity: NDArray[np.float64],
) -> None:
    """Write one synthetic raw breath CSV with the prespecified channel schema."""
    n_rows = int(gas_signals.shape[0])
    frame = pd.DataFrame(
        {
            "S0": np.zeros(n_rows, dtype=np.int64),
            **{channel_name: gas_signals[:, channel_index] for channel_index, channel_name in enumerate(GAS_CHANNELS)},
            "S12": temperature,
            "S13": humidity,
        },
        columns=RAW_COLUMNS,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")


def _create_synthetic_cohort(
    tmp_path: Path,
    sample_specs: list[dict[str, Any]],
) -> tuple[Path, Path, Path]:
    """Create synthetic metadata and breath-data fixtures for an end-to-end run."""
    metadata_path = tmp_path / "clinical_data" / "metadata.csv"
    breath_data_root = tmp_path / "breath_data"
    output_dir = tmp_path / "processed"
    _write_metadata_csv(metadata_path, _build_metadata_frame(sample_specs))

    for spec in sample_specs:
        n_rows = int(spec.get("n_rows", 1000))
        onsets = list(spec.get("onsets", [25] * len(GAS_CHANNELS)))
        gas_signals = np.asarray(spec.get("gas_signals", _build_gas_matrix(n_rows, onsets=onsets)), dtype=np.float64)
        temperature = np.asarray(
            spec.get(
                "temperature",
                _build_temperature_trace(
                    n_rows,
                    baseline_value=float(spec.get("temperature_baseline", 300.0)),
                    response_value=float(spec.get("temperature_response", 301.0)),
                ),
            ),
            dtype=np.float64,
        )
        humidity = np.asarray(
            spec.get(
                "humidity",
                _build_humidity_trace(
                    n_rows,
                    baseline_value=float(spec.get("humidity_baseline", 800.0)),
                    response_value=float(spec.get("humidity_response", 805.0)),
                ),
            ),
            dtype=np.float64,
        )
        _write_raw_csv(
            breath_data_root / str(spec["filename"]),
            gas_signals=gas_signals,
            temperature=temperature,
            humidity=humidity,
        )

    return metadata_path, breath_data_root, output_dir


def _read_outputs(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three audit CSV outputs from one preprocessing run."""
    sample_index = pd.read_csv(output_dir / "sample_index.csv", keep_default_na=False)
    qc_summary = pd.read_csv(output_dir / "qc_reports" / "qc_summary.csv")
    qc_diagnostics = pd.read_csv(output_dir / "qc_reports" / "qc_diagnostics.csv")
    return sample_index, qc_summary, qc_diagnostics


def _build_baseline_override(t_baseline_end: int) -> BaselineDetectionResult:
    """Build a synthetic baseline-detection result for monkeypatched branches."""
    return BaselineDetectionResult(
        t_baseline_end=t_baseline_end,
        fallback_used=False,
        per_channel_onsets=np.full(len(GAS_CHANNELS), t_baseline_end, dtype=np.int64),
        n_triggered_channels=len(GAS_CHANNELS),
    )


def _expected_qc_diagnostics_columns() -> list[str]:
    """Return the exact diagnostics schema expected from `qc_diagnostics.csv`."""
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


@pytest.fixture
def rng() -> np.random.Generator:
    """Provide a deterministic generator for any synthetic perturbations."""
    return np.random.default_rng(RANDOM_SEED)


def test_preprocessing_pipeline_end_to_end_writes_expected_artifacts(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A tiny synthetic cohort should run end to end and emit all promised artifacts."""
    _module, PreprocessingPipeline, PreprocessingResult = _import_preprocessing_symbols()
    sample_specs = [
        {"sample_id": 104, "filename": "sample_104.csv", "label": 1, "onsets": [25] * len(GAS_CHANNELS)},
        {"sample_id": 101, "filename": "sample_101.csv", "label": 0, "onsets": [41] * len(GAS_CHANNELS)},
        {"sample_id": 108, "filename": "sample_108.csv", "label": 1, "onsets": [25, 26, 27] + [80] * 8},
        {"sample_id": 103, "filename": "sample_103.csv", "label": 0, "onsets": [25] * len(GAS_CHANNELS)},
        {
            "sample_id": 107,
            "filename": "sample_107.csv",
            "label": 1,
            "onsets": [25] * len(GAS_CHANNELS),
            "temperature_baseline": 900.0,
            "temperature_response": 901.0,
        },
        {"sample_id": 102, "filename": "sample_102.csv", "label": 0, "n_rows": 999, "onsets": [25] * len(GAS_CHANNELS)},
        {"sample_id": 105, "filename": "sample_105.csv", "label": 1, "onsets": [25] * len(GAS_CHANNELS)},
        {"sample_id": 106, "filename": "sample_106.csv", "label": 0, "onsets": [25] * len(GAS_CHANNELS)},
    ]
    metadata_path, breath_data_root, output_dir = _create_synthetic_cohort(tmp_path, sample_specs)
    pipeline = PreprocessingPipeline()

    with caplog.at_level("INFO"):
        result = pipeline.run(
            metadata_path=metadata_path,
            breath_data_root=breath_data_root,
            output_dir=output_dir,
        )

    assert isinstance(result, PreprocessingResult)
    assert result.total_samples == 8
    assert result.n_fallback_used == 1
    assert result.pipeline_a_path.exists()
    assert result.pipeline_b_path.exists()
    assert result.sample_index_path.exists()
    assert result.qc_summary_path.exists()
    assert result.qc_diagnostics_path.exists()

    sample_index, qc_summary, qc_diagnostics = _read_outputs(output_dir)
    assert sample_index.shape[0] == 8
    assert qc_summary.shape[0] == 8
    assert qc_diagnostics.shape[0] == 8
    assert "Preprocessing run start" in caplog.text
    assert "Preprocessing summary" in caplog.text


def test_preprocessing_pipeline_preserves_sample_order_and_filters_pipeline_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit files should keep every sample while Pipeline A and B keep only final-output rows."""
    preprocessing_module, PreprocessingPipeline, _PreprocessingResult = _import_preprocessing_symbols()
    sample_specs = [
        {"sample_id": 40, "filename": "sample_40.csv", "label": 1, "onsets": [25] * len(GAS_CHANNELS)},
        {"sample_id": 10, "filename": "sample_10.csv", "label": 0, "n_rows": 999, "onsets": [25] * len(GAS_CHANNELS)},
        {
            "sample_id": 30,
            "filename": "sample_30.csv",
            "label": 1,
            "onsets": [25] * len(GAS_CHANNELS),
            "temperature_baseline": 900.0,
            "temperature_response": 901.0,
        },
        {"sample_id": 20, "filename": "sample_20.csv", "label": 0, "onsets": [25] * len(GAS_CHANNELS)},
        {"sample_id": 50, "filename": "sample_50.csv", "label": 1, "onsets": [41] * len(GAS_CHANNELS)},
    ]
    metadata_path, breath_data_root, output_dir = _create_synthetic_cohort(tmp_path, sample_specs)
    pipeline = PreprocessingPipeline()
    original_detect_baseline_end = preprocessing_module.detect_baseline_end

    def patched_detect_baseline_end(record: Any) -> BaselineDetectionResult:
        if int(record.sample_id) == 20:
            return _build_baseline_override(70)
        return original_detect_baseline_end(record)

    monkeypatch.setattr(preprocessing_module, "detect_baseline_end", patched_detect_baseline_end)

    pipeline.run(metadata_path=metadata_path, breath_data_root=breath_data_root, output_dir=output_dir)

    sample_index, _qc_summary, _qc_diagnostics = _read_outputs(output_dir)
    with np.load(output_dir / "sequences" / "pipeline_a_raw.npz", allow_pickle=False) as archive:
        assert archive["x"].shape == (2, 960, len(GAS_CHANNELS))
        assert archive["sample_id"].tolist() == [40, 50]
    pipeline_b = pd.read_csv(output_dir / "features" / "pipeline_b_raw.csv")

    assert sample_index["sample_id"].tolist() == [10, 20, 30, 40, 50]
    assert pipeline_b.shape == (2, 101)
    assert pipeline_b["sample_id"].tolist() == [40, 50]


def test_preprocessing_pipeline_propagates_qc1_short_circuit(tmp_path: Path) -> None:
    """QC-1 failures should keep NaN diagnostics and remain excluded from model-ready outputs."""
    _module, PreprocessingPipeline, _PreprocessingResult = _import_preprocessing_symbols()
    sample_specs = [
        {
            "sample_id": 101,
            "filename": "sample_101.csv",
            "label": 0,
            "onsets": [25] * len(GAS_CHANNELS),
            "temperature_baseline": 900.0,
            "temperature_response": 901.0,
        },
        {"sample_id": 102, "filename": "sample_102.csv", "label": 1, "onsets": [25] * len(GAS_CHANNELS)},
    ]
    metadata_path, breath_data_root, output_dir = _create_synthetic_cohort(tmp_path, sample_specs)

    PreprocessingPipeline().run(metadata_path=metadata_path, breath_data_root=breath_data_root, output_dir=output_dir)

    sample_index, qc_summary, qc_diagnostics = _read_outputs(output_dir)
    qc1_row = sample_index.loc[sample_index["sample_id"] == 101].iloc[0]
    qc1_summary_row = qc_summary.loc[qc_summary["sample_id"] == 101].iloc[0]
    qc1_diag_row = qc_diagnostics.loc[qc_diagnostics["sample_id"] == 101].iloc[0]
    pipeline_b = pd.read_csv(output_dir / "features" / "pipeline_b_raw.csv")
    with np.load(output_dir / "sequences" / "pipeline_a_raw.npz", allow_pickle=False) as archive:
        pipeline_a_sample_ids = archive["sample_id"].tolist()

    assert qc1_row["fail_reasons"] == "qc1_env_sensor_fault"
    assert bool(qc1_row["in_final_output"]) is False
    assert pd.isna(qc1_summary_row["qc2_acquisition_truncated_pass"])
    assert pd.isna(qc1_summary_row["qc3_unstable_baseline_pass"])
    assert pd.isna(qc1_diag_row["s1_baseline_cv"])
    assert pd.isna(qc1_diag_row["s1_response_amplitude"])
    assert 101 not in pipeline_a_sample_ids
    assert 101 not in pipeline_b["sample_id"].tolist()


def test_preprocessing_pipeline_propagates_clip_length_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Samples that fail the clip-length rule should stay in audit files but leave model inputs."""
    preprocessing_module, PreprocessingPipeline, _PreprocessingResult = _import_preprocessing_symbols()
    sample_specs = [
        {"sample_id": 101, "filename": "sample_101.csv", "label": 0, "onsets": [80] * len(GAS_CHANNELS)},
        {"sample_id": 102, "filename": "sample_102.csv", "label": 1, "onsets": [25] * len(GAS_CHANNELS)},
    ]
    metadata_path, breath_data_root, output_dir = _create_synthetic_cohort(tmp_path, sample_specs)
    pipeline = PreprocessingPipeline()
    original_detect_baseline_end = preprocessing_module.detect_baseline_end

    def patched_detect_baseline_end(record: Any) -> BaselineDetectionResult:
        if int(record.sample_id) == 101:
            return _build_baseline_override(53)
        return original_detect_baseline_end(record)

    monkeypatch.setattr(preprocessing_module, "detect_baseline_end", patched_detect_baseline_end)
    pipeline.run(metadata_path=metadata_path, breath_data_root=breath_data_root, output_dir=output_dir)

    sample_index, qc_summary, qc_diagnostics = _read_outputs(output_dir)
    row = sample_index.loc[sample_index["sample_id"] == 101].iloc[0]
    qc_summary_row = qc_summary.loc[qc_summary["sample_id"] == 101].iloc[0]
    qc_diag_row = qc_diagnostics.loc[qc_diagnostics["sample_id"] == 101].iloc[0]
    pipeline_b = pd.read_csv(output_dir / "features" / "pipeline_b_raw.csv")

    assert "clip_length_insufficient" in row["fail_reasons"]
    assert bool(row["in_final_output"]) is False
    assert bool(qc_summary_row["clip_length_pass"]) is False
    assert qc_diag_row["relative_deficit"] > 0.0125
    assert 101 not in pipeline_b["sample_id"].tolist()


def test_preprocessing_pipeline_includes_padded_sample(tmp_path: Path) -> None:
    """Samples within the padding tolerance should be marked padded and kept."""
    _module, PreprocessingPipeline, _PreprocessingResult = _import_preprocessing_symbols()
    sample_specs = [
        {"sample_id": 101, "filename": "sample_101.csv", "label": 0, "onsets": [41] * len(GAS_CHANNELS)},
        {"sample_id": 102, "filename": "sample_102.csv", "label": 1, "onsets": [25] * len(GAS_CHANNELS)},
    ]
    metadata_path, breath_data_root, output_dir = _create_synthetic_cohort(tmp_path, sample_specs)

    PreprocessingPipeline().run(metadata_path=metadata_path, breath_data_root=breath_data_root, output_dir=output_dir)

    sample_index, qc_summary, qc_diagnostics = _read_outputs(output_dir)
    padded_row = sample_index.loc[sample_index["sample_id"] == 101].iloc[0]
    qc_summary_row = qc_summary.loc[qc_summary["sample_id"] == 101].iloc[0]
    qc_diag_row = qc_diagnostics.loc[qc_diagnostics["sample_id"] == 101].iloc[0]
    pipeline_b = pd.read_csv(output_dir / "features" / "pipeline_b_raw.csv")

    assert bool(padded_row["padded"]) is True
    assert bool(padded_row["in_final_output"]) is True
    assert bool(qc_summary_row["clip_length_pass"]) is True
    assert bool(qc_diag_row["padded"]) is True
    assert 101 in pipeline_b["sample_id"].tolist()


def test_preprocessing_pipeline_propagates_fallback_usage_and_uses_yaml_median(tmp_path: Path) -> None:
    """Fallback-triggered samples should record the YAML median baseline end of 19."""
    _module, PreprocessingPipeline, _PreprocessingResult = _import_preprocessing_symbols()
    sample_specs = [
        {"sample_id": 101, "filename": "sample_101.csv", "label": 0, "onsets": [25, 26, 27] + [80] * 8},
        {"sample_id": 102, "filename": "sample_102.csv", "label": 1, "onsets": [25] * len(GAS_CHANNELS)},
    ]
    metadata_path, breath_data_root, output_dir = _create_synthetic_cohort(tmp_path, sample_specs)

    PreprocessingPipeline().run(metadata_path=metadata_path, breath_data_root=breath_data_root, output_dir=output_dir)

    sample_index, _qc_summary, qc_diagnostics = _read_outputs(output_dir)
    fallback_row = sample_index.loc[sample_index["sample_id"] == 101].iloc[0]
    qc_diag_row = qc_diagnostics.loc[qc_diagnostics["sample_id"] == 101].iloc[0]

    assert bool(fallback_row["fallback_used"]) is True
    assert qc_diag_row["t_baseline_end"] == 19
    assert bool(fallback_row["in_final_output"]) is True


def test_preprocessing_pipeline_never_calls_env_model_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The orchestrator must deserialize environmental coefficients without refitting them."""
    _module, PreprocessingPipeline, _PreprocessingResult = _import_preprocessing_symbols()
    from src.data.env_compensation import EnvCompensationModel

    fit_called = {"value": False}
    original_fit = EnvCompensationModel.fit

    def fail_if_called(*args: Any, **kwargs: Any) -> Any:
        fit_called["value"] = True
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(EnvCompensationModel, "fit", fail_if_called)
    sample_specs = [
        {"sample_id": 101, "filename": "sample_101.csv", "label": 0, "onsets": [25] * len(GAS_CHANNELS)},
        {"sample_id": 102, "filename": "sample_102.csv", "label": 1, "onsets": [25] * len(GAS_CHANNELS)},
    ]
    metadata_path, breath_data_root, output_dir = _create_synthetic_cohort(tmp_path, sample_specs)

    PreprocessingPipeline().run(metadata_path=metadata_path, breath_data_root=breath_data_root, output_dir=output_dir)

    assert fit_called["value"] is False


def test_preprocessing_pipeline_passes_raw_gas_to_run_qc(tmp_path: Path) -> None:
    """The orchestrator should pass raw gas signals, not smoothed signals, into QC."""
    preprocessing_module, PreprocessingPipeline, _PreprocessingResult = _import_preprocessing_symbols()
    gas_signals = np.column_stack(
        [
            np.concatenate(
                [
                    np.array([100 + channel_index, 101 + channel_index, 100 + channel_index, 101 + channel_index, 100 + channel_index], dtype=np.int64),
                        np.arange(995, dtype=np.int64) + 5000 + 10 * channel_index,
                ]
            )
            for channel_index in range(len(GAS_CHANNELS))
        ]
    )
    sample_specs = [
        {
            "sample_id": 101,
            "filename": "sample_101.csv",
            "label": 1,
            "gas_signals": gas_signals,
            "temperature": _build_temperature_trace(1000, baseline_value=300.0, response_value=301.0),
            "humidity": _build_humidity_trace(1000, baseline_value=800.0, response_value=805.0),
        },
        {"sample_id": 102, "filename": "sample_102.csv", "label": 0, "onsets": [25] * len(GAS_CHANNELS)},
    ]
    metadata_path, breath_data_root, output_dir = _create_synthetic_cohort(tmp_path, sample_specs)

    with patch("src.data.preprocessing.run_qc", wraps=preprocessing_module.run_qc) as wrapped_run_qc:
        PreprocessingPipeline().run(
            metadata_path=metadata_path,
            breath_data_root=breath_data_root,
            output_dir=output_dir,
        )

    assert wrapped_run_qc.call_count == 2
    qc_call = next(call for call in wrapped_run_qc.call_args_list if int(call.kwargs["record"].sample_id) == 101)
    assert np.array_equal(qc_call.kwargs["gas_signals"], qc_call.kwargs["record"].gas_signals)
    assert np.array_equal(qc_call.kwargs["gas_signals"], gas_signals)


def test_preprocessing_pipeline_smoothing_runs_only_after_qc_pass(tmp_path: Path) -> None:
    """Smoothing should run exactly once per QC-passing sample."""
    preprocessing_module, PreprocessingPipeline, _PreprocessingResult = _import_preprocessing_symbols()
    sample_specs = [
        {
            "sample_id": 101,
            "filename": "sample_101.csv",
            "label": 0,
            "onsets": [25] * len(GAS_CHANNELS),
            "temperature_baseline": 900.0,
            "temperature_response": 901.0,
        },
        {
            "sample_id": 102,
            "filename": "sample_102.csv",
            "label": 1,
            "onsets": [25] * len(GAS_CHANNELS),
            "temperature_baseline": 900.0,
            "temperature_response": 901.0,
        },
        {"sample_id": 103, "filename": "sample_103.csv", "label": 1, "onsets": [25] * len(GAS_CHANNELS)},
    ]
    metadata_path, breath_data_root, output_dir = _create_synthetic_cohort(tmp_path, sample_specs)

    with patch(
        "src.data.preprocessing.apply_sg_smoothing",
        wraps=preprocessing_module.apply_sg_smoothing,
    ) as wrapped_smoothing:
        PreprocessingPipeline().run(
            metadata_path=metadata_path,
            breath_data_root=breath_data_root,
            output_dir=output_dir,
        )

    assert wrapped_smoothing.call_count == 1


def test_preprocessing_pipeline_skips_post_qc_steps_for_qc1_fail_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """QC-1 failures should not reach compensation, correction, or trimming stages."""
    preprocessing_module, PreprocessingPipeline, _PreprocessingResult = _import_preprocessing_symbols()
    from src.data.env_compensation import EnvCompensationModel

    counts = {"smoothing": 0, "env_apply": 0, "correction": 0, "trim": 0}
    original_smoothing = preprocessing_module.apply_sg_smoothing
    original_env_apply = EnvCompensationModel.apply
    original_correction = preprocessing_module.apply_fractional_correction
    original_trim = preprocessing_module.trim_response_segment

    def counting_smoothing(gas_signals: NDArray[np.float64]) -> NDArray[np.float64]:
        counts["smoothing"] += 1
        return original_smoothing(gas_signals)

    def counting_env_apply(
        self: EnvCompensationModel,
        smoothed_gas: NDArray[np.float64],
        t_sample: float,
        h_sample: float,
    ) -> NDArray[np.float64]:
        counts["env_apply"] += 1
        return original_env_apply(self, smoothed_gas, t_sample, h_sample)

    def counting_correction(
        compensated_gas: NDArray[np.float64],
        t_baseline_end: int,
    ) -> NDArray[np.float64]:
        counts["correction"] += 1
        return original_correction(compensated_gas, t_baseline_end)

    def counting_trim(
        corrected_gas: NDArray[np.float64],
        t_baseline_end: int,
        clip_length_L: int,
        padding_relative_deficit_threshold: float | None,
    ) -> Any:
        counts["trim"] += 1
        return original_trim(
            corrected_gas,
            t_baseline_end,
            clip_length_L,
            padding_relative_deficit_threshold,
        )

    monkeypatch.setattr(preprocessing_module, "apply_sg_smoothing", counting_smoothing)
    monkeypatch.setattr(EnvCompensationModel, "apply", counting_env_apply)
    monkeypatch.setattr(preprocessing_module, "apply_fractional_correction", counting_correction)
    monkeypatch.setattr(preprocessing_module, "trim_response_segment", counting_trim)
    sample_specs = [
        {
            "sample_id": 101,
            "filename": "sample_101.csv",
            "label": 0,
            "onsets": [25] * len(GAS_CHANNELS),
            "temperature_baseline": 900.0,
            "temperature_response": 901.0,
        },
        {"sample_id": 102, "filename": "sample_102.csv", "label": 1, "onsets": [25] * len(GAS_CHANNELS)},
    ]
    metadata_path, breath_data_root, output_dir = _create_synthetic_cohort(tmp_path, sample_specs)

    PreprocessingPipeline().run(metadata_path=metadata_path, breath_data_root=breath_data_root, output_dir=output_dir)

    assert counts == {"smoothing": 1, "env_apply": 1, "correction": 1, "trim": 1}


def test_preprocessing_pipeline_writes_expected_audit_schemas(tmp_path: Path) -> None:
    """The three audit CSVs should follow the exact locked column contracts."""
    _module, PreprocessingPipeline, _PreprocessingResult = _import_preprocessing_symbols()
    sample_specs = [
        {
            "sample_id": 201,
            "filename": "sample_201.csv",
            "label": 0,
            "onsets": [25] * len(GAS_CHANNELS),
            "temperature_baseline": 900.0,
            "temperature_response": 901.0,
        },
        {"sample_id": 202, "filename": "sample_202.csv", "label": 1, "onsets": [25] * len(GAS_CHANNELS)},
    ]
    metadata_path, breath_data_root, output_dir = _create_synthetic_cohort(tmp_path, sample_specs)

    PreprocessingPipeline().run(metadata_path=metadata_path, breath_data_root=breath_data_root, output_dir=output_dir)

    sample_index, qc_summary, qc_diagnostics = _read_outputs(output_dir)
    expected_sample_index_columns = [
        "sample_id",
        "filename",
        "collection_time",
        "label",
        "disease_category",
        "n_rows",
        "env_sensor_ok",
        "qc_status",
        "fail_reasons",
        "fallback_used",
        "padded",
        "in_final_output",
    ]
    expected_qc_summary_columns = [
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

    assert sample_index.columns.tolist() == expected_sample_index_columns
    assert qc_summary.columns.tolist() == expected_qc_summary_columns
    assert qc_diagnostics.columns.tolist() == _expected_qc_diagnostics_columns()

    expected_sample_index = pd.DataFrame(
        [
            {
                "sample_id": 201,
                "filename": "sample_201.csv",
                "collection_time": "2026-01-01 08:00:00",
                "label": 0,
                "disease_category": 1,
                "n_rows": 1000,
                "env_sensor_ok": False,
                "qc_status": "fail",
                "fail_reasons": "qc1_env_sensor_fault",
                "fallback_used": False,
                "padded": False,
                "in_final_output": False,
            },
            {
                "sample_id": 202,
                "filename": "sample_202.csv",
                "collection_time": "2026-01-02 08:00:00",
                "label": 1,
                "disease_category": 2,
                "n_rows": 1000,
                "env_sensor_ok": True,
                "qc_status": "pass",
                "fail_reasons": "",
                "fallback_used": False,
                "padded": False,
                "in_final_output": True,
            },
        ]
    )
    pd.testing.assert_frame_equal(sample_index, expected_sample_index, check_dtype=False)
