"""End-to-end preprocessing orchestration for Stage 1 artifacts.

This module wires the already-implemented preprocessing primitives together for a
full cohort run. It writes the five stage artifacts promised by the locked plan:

- `output_dir / "sequences" / "pipeline_a_raw.npz"`
- `output_dir / "features" / "pipeline_b_raw.csv"`
- `output_dir / "sample_index.csv"`
- `output_dir / "qc_reports" / "qc_summary.csv"`
- `output_dir / "qc_reports" / "qc_diagnostics.csv"`

The `fail_reasons` column representation is the semicolon-separated string used
by `src.data.qc.qc_summary_to_df`, for example
`"qc2_acquisition_truncated;qc5_sensor_saturation"`. The orchestrator uses the
same representation in `sample_index.csv` for audit consistency.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import yaml

from src.config import CONFIGS_ROOT, GAS_CHANNELS, OUTPUTS_LOGS_ROOT
from src.data.baseline import BaselineDetectionResult, detect_baseline_end
from src.data.correction import TrimResult, apply_fractional_correction, trim_response_segment
from src.data.env_compensation import EnvCompensationModel
from src.data.loader import load_dataset, load_metadata
from src.data.pipeline_a import assemble_pipeline_a, save_pipeline_a
from src.data.qc import QCResult, qc_diagnostics_to_df, qc_summary_to_df, run_qc
from src.data.smoothing import apply_sg_smoothing
from src.features.extraction import extract_pipeline_b_features, save_pipeline_b


LOGGER = logging.getLogger(__name__)

PREPROCESSING_CONFIG_PATH = CONFIGS_ROOT / "preprocessing.yaml"
PREPROCESSING_LOG_PATH = OUTPUTS_LOGS_ROOT / "preprocessing.log"

PIPELINE_A_RELATIVE_PATH = Path("sequences") / "pipeline_a_raw.npz"
PIPELINE_B_RELATIVE_PATH = Path("features") / "pipeline_b_raw.csv"
SAMPLE_INDEX_RELATIVE_PATH = Path("sample_index.csv")
QC_SUMMARY_RELATIVE_PATH = Path("qc_reports") / "qc_summary.csv"
QC_DIAGNOSTICS_RELATIVE_PATH = Path("qc_reports") / "qc_diagnostics.csv"

FAIL_REASON_DELIMITER = ";"
LOG_PROGRESS_EVERY = 100
QC_REASON_KEYS = (
    "qc1_env_sensor_fault",
    "qc2_acquisition_truncated",
    "qc3_unstable_baseline",
    "qc4_insufficient_response",
    "qc5_sensor_saturation",
    "qc6_excessive_spikes",
    "clip_length_insufficient",
)


@dataclass(frozen=True)
class PreprocessingResult:
    """Summary metadata for one preprocessing run.

    Attributes:
        total_samples: Total sample count loaded from the sorted metadata table.
        n_qc_pass: Number of samples that pass QC-1 through QC-6 before trimming.
        n_in_final_output: Number of samples that remain after QC and clip-length
            filtering and therefore enter Pipeline A and Pipeline B.
        n_excluded_per_qc_check: Count of exclusions attributed to each QC or
            clip-length rule. Samples that fail multiple checks contribute to
            each corresponding count.
        n_fallback_used: Number of samples whose baseline end came from the YAML
            fallback median.
        n_padded: Number of samples that were tail-padded and still included.
        pipeline_a_path: Output path for `pipeline_a_raw.npz`.
        pipeline_b_path: Output path for `pipeline_b_raw.csv`.
        sample_index_path: Output path for `sample_index.csv`.
        qc_summary_path: Output path for `qc_summary.csv`.
        qc_diagnostics_path: Output path for `qc_diagnostics.csv`.
        preprocessing_log_path: Output path for `preprocessing.log`.
    """

    total_samples: int
    n_qc_pass: int
    n_in_final_output: int
    n_excluded_per_qc_check: dict[str, int]
    n_fallback_used: int
    n_padded: int
    pipeline_a_path: Path
    pipeline_b_path: Path
    sample_index_path: Path
    qc_summary_path: Path
    qc_diagnostics_path: Path
    preprocessing_log_path: Path


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load one YAML mapping with informative path-aware errors.

    Args:
        path: YAML file path to load.

    Returns:
        The parsed YAML mapping.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        OSError: If the YAML file cannot be read.
        ValueError: If the parsed payload is not a mapping.
    """
    try:
        with path.open("r", encoding="utf-8") as file_handle:
            payload = yaml.safe_load(file_handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Failed to read preprocessing YAML at {path}: {exc}") from exc
    except OSError as exc:
        raise OSError(f"Failed to read preprocessing YAML at {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Preprocessing YAML at {path} must deserialize to a mapping.")
    return payload


def _write_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Write one CSV artifact with informative path-aware errors.

    Args:
        df: Data frame to serialize.
        path: Destination CSV path.

    Raises:
        OSError: If the CSV cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(path, index=False)
    except OSError as exc:
        raise OSError(f"Failed to write CSV at {path}: {exc}") from exc


def _join_fail_reasons(fail_reasons: list[str]) -> str:
    """Serialize ordered fail reasons using the audit delimiter."""
    return FAIL_REASON_DELIMITER.join(fail_reasons)


def _resolve_optional_float(value: float | None) -> float:
    """Convert an optional float to a CSV-safe float."""
    return np.nan if value is None else float(value)


def _output_paths(output_dir: Path) -> dict[str, Path]:
    """Construct the deterministic preprocessing artifact paths."""
    return {
        "pipeline_a": output_dir / PIPELINE_A_RELATIVE_PATH,
        "pipeline_b": output_dir / PIPELINE_B_RELATIVE_PATH,
        "sample_index": output_dir / SAMPLE_INDEX_RELATIVE_PATH,
        "qc_summary": output_dir / QC_SUMMARY_RELATIVE_PATH,
        "qc_diagnostics": output_dir / QC_DIAGNOSTICS_RELATIVE_PATH,
    }


class PreprocessingPipeline:
    """Run the full preprocessing stage from raw files to stage artifacts."""

    def __init__(self, config_path: Path | None = None) -> None:
        """Load the locked preprocessing configuration once.

        Args:
            config_path: Optional override for `configs/preprocessing.yaml`.

        Raises:
            FileNotFoundError: If the YAML file is missing.
            OSError: If the YAML file cannot be read.
            ValueError: If required configuration keys are missing or invalid.
        """
        self.config_path = PREPROCESSING_CONFIG_PATH if config_path is None else Path(config_path)
        self.config = _read_yaml_mapping(self.config_path)
        self.baseline_detection = self._require_mapping(self.config, "baseline_detection")
        self.qc_thresholds = self._require_mapping(self.config, "qc_thresholds")
        self.environmental_compensation = self._require_mapping(self.config, "environmental_compensation")
        self.correction_and_trimming = self._require_mapping(self.config, "correction_and_trimming")

        self.fallback_median = int(self.baseline_detection["fallback_median"])
        self.clip_length_L = int(self.correction_and_trimming["clip_length_L"])
        padding_threshold_raw = self.correction_and_trimming["padding_relative_deficit_threshold"]
        self.padding_relative_deficit_threshold = (
            None if padding_threshold_raw is None else float(padding_threshold_raw)
        )

        qc1_config = self._require_mapping(self.qc_thresholds, "qc1_env_sensor_fault")
        self.qc1_enabled = bool(qc1_config["enabled"])
        self.qc1_fault_side = str(qc1_config["fault_side"])
        if self.qc1_fault_side not in {"low", "high"}:
            raise ValueError(f"qc1_fault_side must be 'low' or 'high'; got {self.qc1_fault_side!r}")

        self.env_model = EnvCompensationModel.from_yaml_dict(self.environmental_compensation)
        LOGGER.info(
            "Deserialized environmental compensation model from %s clip_length_L=%s fallback_median=%s",
            self.config_path,
            self.clip_length_L,
            self.fallback_median,
        )

    @staticmethod
    def _require_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
        """Read one required nested mapping from the YAML payload."""
        value = parent.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"Configuration key '{key}' must contain a mapping.")
        return value

    @staticmethod
    def _resolved_baseline_result(result: BaselineDetectionResult, fallback_median: int) -> BaselineDetectionResult:
        """Resolve the final per-sample baseline end using the YAML fallback."""
        if result.fallback_used:
            return BaselineDetectionResult(
                t_baseline_end=fallback_median,
                fallback_used=True,
                per_channel_onsets=result.per_channel_onsets,
                n_triggered_channels=result.n_triggered_channels,
            )
        if result.t_baseline_end is None:
            raise ValueError("Baseline detection returned an unresolved non-fallback baseline end.")
        return BaselineDetectionResult(
            t_baseline_end=int(result.t_baseline_end),
            fallback_used=False,
            per_channel_onsets=result.per_channel_onsets,
            n_triggered_channels=result.n_triggered_channels,
        )

    @staticmethod
    def _sample_index_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
        """Build the locked `sample_index.csv` schema."""
        sample_index_df = pd.DataFrame(
            rows,
            columns=[
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
            ],
        )
        return sample_index_df

    def run(self, metadata_path: Path, breath_data_root: Path, output_dir: Path) -> PreprocessingResult:
        """Execute the full preprocessing cohort pipeline.

        Args:
            metadata_path: Metadata CSV path for the run.
            breath_data_root: Raw breath-data directory for the run.
            output_dir: Root output directory for processed stage artifacts.

        Returns:
            A frozen `PreprocessingResult` describing the run summary and output
            artifact paths.

        Raises:
            RuntimeError: If no sample remains for Pipeline A and B assembly.
            OSError: If any artifact cannot be written.
            ValueError: If the configuration or per-sample intermediate state is invalid.
        """
        metadata_path = Path(metadata_path)
        breath_data_root = Path(breath_data_root)
        output_dir = Path(output_dir)
        paths = _output_paths(output_dir)
        for artifact_path in paths.values():
            artifact_path.parent.mkdir(parents=True, exist_ok=True)

        LOGGER.info("Preprocessing run start")
        metadata = load_metadata(metadata_path).sort_values("sample_id").reset_index(drop=True)
        LOGGER.info("Sorted metadata for preprocessing by sample_id with shape=%s", metadata.shape)
        records = load_dataset(metadata=metadata, breath_data_root=breath_data_root)
        LOGGER.info("Loaded preprocessing dataset total_records=%s", len(records))

        sample_index_rows: list[dict[str, Any]] = []
        qc_results: list[QCResult] = []
        final_trimmed_signals: list[NDArray[np.float64]] = []
        final_sample_ids: list[int] = []
        final_labels: list[int] = []

        n_excluded_per_qc_check = {reason: 0 for reason in QC_REASON_KEYS}
        n_fallback_used = 0
        n_padded = 0
        n_qc_pass = 0

        for sample_index, record in enumerate(records, start=1):
            if sample_index == 1 or sample_index % LOG_PROGRESS_EVERY == 0 or sample_index == len(records):
                LOGGER.info(
                    "Processing sample %s/%s sample_id=%s filename=%s",
                    sample_index,
                    len(records),
                    record.sample_id,
                    record.filename,
                )

            baseline_result = detect_baseline_end(record)
            resolved_baseline = self._resolved_baseline_result(baseline_result, self.fallback_median)
            if resolved_baseline.fallback_used:
                n_fallback_used += 1

            qc_result = run_qc(
                record=record,
                baseline_result=resolved_baseline,
                gas_signals=record.gas_signals,
                env_temperature=record.temperature,
                env_humidity=record.humidity,
                thresholds=self.qc_thresholds,
                qc1_enabled=self.qc1_enabled,
                qc1_fault_side=self.qc1_fault_side,
            )

            final_qc_result = qc_result
            padded = False
            in_final_output = False

            if qc_result.fail_reasons:
                for reason in qc_result.fail_reasons:
                    n_excluded_per_qc_check[reason] += 1
            else:
                n_qc_pass += 1
                smoothed_gas = apply_sg_smoothing(record.gas_signals)
                t_baseline_end = int(resolved_baseline.t_baseline_end)
                t_sample = float(np.mean(np.asarray(record.temperature[:t_baseline_end], dtype=np.float64)))
                h_sample = float(np.mean(np.asarray(record.humidity[:t_baseline_end], dtype=np.float64)))
                compensated_gas = self.env_model.apply(smoothed_gas=smoothed_gas, t_sample=t_sample, h_sample=h_sample)
                corrected_gas = apply_fractional_correction(
                    compensated_gas=compensated_gas,
                    t_baseline_end=t_baseline_end,
                )
                trim_result = trim_response_segment(
                    corrected_gas=corrected_gas,
                    t_baseline_end=t_baseline_end,
                    clip_length_L=self.clip_length_L,
                    padding_relative_deficit_threshold=self.padding_relative_deficit_threshold,
                )
                final_fail_reasons = list(qc_result.fail_reasons)
                if not trim_result.clip_length_pass:
                    final_fail_reasons.append("clip_length_insufficient")
                    n_excluded_per_qc_check["clip_length_insufficient"] += 1
                else:
                    in_final_output = True
                    if trim_result.trimmed is None:
                        raise ValueError("trim_response_segment returned clip_length_pass=True with no trimmed matrix.")
                    final_trimmed_signals.append(trim_result.trimmed)
                    final_sample_ids.append(record.sample_id)
                    final_labels.append(record.label)

                padded = bool(trim_result.padded)
                if padded:
                    n_padded += 1

                final_qc_result = replace(
                    qc_result,
                    clip_length_pass=trim_result.clip_length_pass,
                    fail_reasons=final_fail_reasons,
                    qc_status="pass" if not final_fail_reasons else "fail",
                    env_temperature_sample=t_sample,
                    env_humidity_sample=h_sample,
                    env_reference_temperature=self.env_model.t_ref,
                    env_reference_humidity=self.env_model.h_ref,
                    env_temperature_humidity_correlation=self.env_model.temperature_humidity_pearson_r,
                    env_multicollinearity_fallback_used=self.env_model.multicollinearity_fallback_used,
                    n_available=float(trim_result.n_available),
                    clip_length_L=float(self.clip_length_L),
                    relative_deficit=_resolve_optional_float(trim_result.relative_deficit),
                    padded=trim_result.padded,
                )

            qc_results.append(final_qc_result)
            sample_index_rows.append(
                {
                    "sample_id": record.sample_id,
                    "filename": record.filename,
                    "collection_time": record.collection_time,
                    "label": record.label,
                    "disease_category": record.disease_category,
                    "n_rows": record.n_rows,
                    "env_sensor_ok": bool(final_qc_result.qc1_env_sensor_fault_pass),
                    "qc_status": final_qc_result.qc_status,
                    "fail_reasons": _join_fail_reasons(final_qc_result.fail_reasons),
                    "fallback_used": bool(resolved_baseline.fallback_used),
                    "padded": padded,
                    "in_final_output": in_final_output,
                }
            )

        if not final_trimmed_signals:
            raise RuntimeError("No samples remained in_final_output; cannot assemble Pipeline A or Pipeline B.")

        sample_index_df = self._sample_index_df(sample_index_rows)
        qc_summary_df = qc_summary_to_df(qc_results)
        qc_diagnostics_df = qc_diagnostics_to_df(qc_results)

        pipeline_a_output = assemble_pipeline_a(
            corrected_trimmed_signals=final_trimmed_signals,
            sample_ids=final_sample_ids,
            labels=final_labels,
            clip_length_L=self.clip_length_L,
        )
        save_pipeline_a(output=pipeline_a_output, path=paths["pipeline_a"])

        pipeline_b_df = extract_pipeline_b_features(
            corrected_trimmed_signals=final_trimmed_signals,
            sample_ids=final_sample_ids,
            labels=final_labels,
            channel_names=tuple(GAS_CHANNELS),
        )
        save_pipeline_b(df=pipeline_b_df, path=paths["pipeline_b"])

        _write_dataframe(sample_index_df, paths["sample_index"])
        _write_dataframe(qc_summary_df, paths["qc_summary"])
        _write_dataframe(qc_diagnostics_df, paths["qc_diagnostics"])

        LOGGER.info(
            "Preprocessing summary: total_samples=%s n_qc1_fail=%s n_qc2_fail=%s n_qc3_fail=%s "
            "n_qc4_fail=%s n_qc5_fail=%s n_qc6_fail=%s n_clip_length_fail=%s "
            "n_padded=%s n_fallback_used=%s n_in_final_output=%s",
            len(records),
            n_excluded_per_qc_check["qc1_env_sensor_fault"],
            n_excluded_per_qc_check["qc2_acquisition_truncated"],
            n_excluded_per_qc_check["qc3_unstable_baseline"],
            n_excluded_per_qc_check["qc4_insufficient_response"],
            n_excluded_per_qc_check["qc5_sensor_saturation"],
            n_excluded_per_qc_check["qc6_excessive_spikes"],
            n_excluded_per_qc_check["clip_length_insufficient"],
            n_padded,
            n_fallback_used,
            len(final_trimmed_signals),
        )
        LOGGER.info(
            "Wrote preprocessing artifacts pipeline_a=%s pipeline_b=%s sample_index=%s qc_summary=%s qc_diagnostics=%s",
            paths["pipeline_a"],
            paths["pipeline_b"],
            paths["sample_index"],
            paths["qc_summary"],
            paths["qc_diagnostics"],
        )

        return PreprocessingResult(
            total_samples=len(records),
            n_qc_pass=n_qc_pass,
            n_in_final_output=len(final_trimmed_signals),
            n_excluded_per_qc_check=n_excluded_per_qc_check,
            n_fallback_used=n_fallback_used,
            n_padded=n_padded,
            pipeline_a_path=paths["pipeline_a"],
            pipeline_b_path=paths["pipeline_b"],
            sample_index_path=paths["sample_index"],
            qc_summary_path=paths["qc_summary"],
            qc_diagnostics_path=paths["qc_diagnostics"],
            preprocessing_log_path=PREPROCESSING_LOG_PATH,
        )


__all__ = ["PREPROCESSING_LOG_PATH", "PreprocessingPipeline", "PreprocessingResult"]
