"""Run the end-to-end preprocessing orchestration for Stage 1 artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path

from src.config import BREATH_DATA_ROOT, METADATA_PATH, OUTPUTS_LOGS_ROOT, PROCESSED_DATA_ROOT, RANDOM_SEED
from src.data.preprocessing import PREPROCESSING_LOG_PATH, PreprocessingPipeline


LOGGER = logging.getLogger(__name__)


def _setup_logging(log_path: Path) -> None:
    """Configure root logging for the preprocessing run.

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


def main(
    metadata_path: Path | None = None,
    breath_data_root: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    """Execute the full preprocessing stage.

    Args:
        metadata_path: Optional metadata CSV override.
        breath_data_root: Optional raw breath-data directory override.
        output_dir: Optional processed-data output root override.
    """
    metadata_path = METADATA_PATH if metadata_path is None else Path(metadata_path)
    breath_data_root = BREATH_DATA_ROOT if breath_data_root is None else Path(breath_data_root)
    output_dir = PROCESSED_DATA_ROOT if output_dir is None else Path(output_dir)

    OUTPUTS_LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    _setup_logging(PREPROCESSING_LOG_PATH)

    run_start_iso = datetime.now(timezone.utc).astimezone().isoformat()
    LOGGER.info("Preprocessing run start")
    LOGGER.info("Preprocessing run timestamp=%s", run_start_iso)
    LOGGER.info("RANDOM_SEED=%s", RANDOM_SEED)
    LOGGER.info("Metadata path=%s", metadata_path.resolve())
    LOGGER.info("Breath data root=%s", breath_data_root.resolve())
    LOGGER.info("Output directory=%s", output_dir.resolve())

    pipeline = PreprocessingPipeline()
    result = pipeline.run(
        metadata_path=metadata_path,
        breath_data_root=breath_data_root,
        output_dir=output_dir,
    )
    LOGGER.info(
        "Preprocessing summary: total_samples=%s n_qc1_fail=%s n_qc2_fail=%s n_qc3_fail=%s "
        "n_qc4_fail=%s n_qc5_fail=%s n_qc6_fail=%s n_clip_length_fail=%s "
        "n_padded=%s n_fallback_used=%s n_in_final_output=%s",
        result.total_samples,
        result.n_excluded_per_qc_check["qc1_env_sensor_fault"],
        result.n_excluded_per_qc_check["qc2_acquisition_truncated"],
        result.n_excluded_per_qc_check["qc3_unstable_baseline"],
        result.n_excluded_per_qc_check["qc4_insufficient_response"],
        result.n_excluded_per_qc_check["qc5_sensor_saturation"],
        result.n_excluded_per_qc_check["qc6_excessive_spikes"],
        result.n_excluded_per_qc_check["clip_length_insufficient"],
        result.n_padded,
        result.n_fallback_used,
        result.n_in_final_output,
    )


if __name__ == "__main__":
    main()
