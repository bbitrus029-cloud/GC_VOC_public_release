from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.data_split import _format_path, _prepare_cohort_dataframe
from src.analysis.stage7_part_a_common import _merge_predictions_with_metadata
from src.data.loader import load_metadata


LOGGER = logging.getLogger(__name__)
TUMOR_MARKER_COLUMNS = ("cea", "ca199", "ca724")


def _parse_truncated_numeric(value: object) -> float | None:
    """Parse one tumor-marker value that may include censoring markers.

    Args:
        value: Raw metadata cell value.

    Returns:
        A float when parsing succeeds, otherwise `None`.
    """
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    stripped = str(value).strip()
    if stripped in {"", "-", "/", "?"}:
        return None
    if stripped.startswith((">", "<")):
        stripped = stripped[1:].strip()

    try:
        return float(stripped)
    except ValueError:
        LOGGER.warning("Unparseable tumor-marker value %r converted to missing.", value)
        return None


def _load_final_cohort_extended_metadata(config: dict[str, Any]) -> pd.DataFrame:
    """Load the locked 972-sample cohort with collection time and tumor markers.

    Args:
        config: Parsed `configs/modeling_global.yaml` mapping.

    Returns:
        The final-cohort dataframe sorted by `sample_id`.
    """
    cohort = _prepare_cohort_dataframe(config=config)
    metadata = load_metadata()
    extra_columns = metadata.loc[:, ["sample_id", "collection_time", *TUMOR_MARKER_COLUMNS]].copy()
    for marker_name in TUMOR_MARKER_COLUMNS:
        extra_columns[marker_name] = extra_columns[marker_name].apply(_parse_truncated_numeric)

    merged = cohort.merge(extra_columns, on="sample_id", how="left", validate="one_to_one")
    if merged[["collection_time"]].isna().any().any():
        raise AssertionError("Extended metadata merge left missing collection_time values.")
    LOGGER.info(
        "Loaded final cohort extended metadata: shape=%s, missing_counts=%s",
        merged.shape,
        merged[["collection_time", *TUMOR_MARKER_COLUMNS]].isna().sum().to_dict(),
    )
    return merged.sort_values("sample_id", kind="mergesort").reset_index(drop=True)


def _parse_collection_time_or_raise(frame: pd.DataFrame) -> pd.Series:
    """Parse collection timestamps and raise with sample_id on any failure.

    Args:
        frame: Dataframe containing `sample_id` and `collection_time`.

    Returns:
        Parsed pandas timestamps aligned to `frame`.
    """
    parsed = pd.to_datetime(frame["collection_time"], errors="coerce")
    if parsed.isna().any():
        first_bad_row = frame.loc[parsed.isna(), ["sample_id", "collection_time"]].iloc[0]
        raise AssertionError(
            "Unparseable collection_time encountered for "
            f"sample_id={int(first_bad_row['sample_id'])}, value={first_bad_row['collection_time']!r}."
        )
    return parsed


def _compute_continuous_smd(group_a: pd.Series, group_b: pd.Series) -> float:
    """Compute the absolute standardized mean difference for a continuous variable.

    Args:
        group_a: Left-group numeric values.
        group_b: Right-group numeric values.

    Returns:
        Absolute standardized mean difference.
    """
    values_a = group_a.dropna().astype(float).to_numpy()
    values_b = group_b.dropna().astype(float).to_numpy()
    if len(values_a) == 0 or len(values_b) == 0:
        return float(np.nan)
    mean_difference = float(np.mean(values_a) - np.mean(values_b))
    pooled_sd = float(np.sqrt((np.var(values_a, ddof=1) + np.var(values_b, ddof=1)) / 2.0))
    if pooled_sd < 1e-8:
        return 0.0 if abs(mean_difference) < 1e-8 else abs(mean_difference)
    return abs(mean_difference) / pooled_sd


def _compute_binary_risk_difference(group_a: pd.Series, group_b: pd.Series) -> float:
    """Compute the absolute risk difference for a binary indicator.

    Args:
        group_a: Left-group 0/1 values.
        group_b: Right-group 0/1 values.

    Returns:
        Absolute difference in event proportions.
    """
    values_a = group_a.dropna().astype(float).to_numpy()
    values_b = group_b.dropna().astype(float).to_numpy()
    if len(values_a) == 0 or len(values_b) == 0:
        return float(np.nan)
    return abs(float(np.mean(values_a) - np.mean(values_b)))


def _write_dataframe_csv(path: Path, dataframe: pd.DataFrame) -> None:
    """Persist one dataframe to CSV with project-relative logging.

    Args:
        path: Destination CSV path.
        dataframe: Dataframe to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        dataframe.to_csv(path, index=False)
    except OSError as exc:
        LOGGER.error("Failed to write CSV to %s.", path)
        raise OSError(f"Failed to write CSV to {path}.") from exc
    LOGGER.info("Wrote CSV to %s.", _format_path(path))


def _merge_predictions_with_optional_metadata(
    *,
    prediction_frame: pd.DataFrame,
    metadata_frame: pd.DataFrame,
    required_columns: list[str],
    optional_columns: list[str],
) -> pd.DataFrame:
    """Merge strict non-missing columns and permissive nullable columns.

    Args:
        prediction_frame: Prediction dataframe keyed by `sample_id`.
        metadata_frame: Metadata dataframe keyed by `sample_id`.
        required_columns: Columns that must merge with no missing values.
        optional_columns: Columns that may legitimately remain missing.

    Returns:
        Merged dataframe.
    """
    merged = _merge_predictions_with_metadata(
        prediction_frame=prediction_frame,
        metadata_frame=metadata_frame,
        required_columns=required_columns,
    )
    if optional_columns:
        merged = merged.merge(
            metadata_frame[["sample_id", *optional_columns]],
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
    return merged


__all__ = [
    "TUMOR_MARKER_COLUMNS",
    "_compute_binary_risk_difference",
    "_compute_continuous_smd",
    "_load_final_cohort_extended_metadata",
    "_merge_predictions_with_optional_metadata",
    "_parse_collection_time_or_raise",
    "_parse_truncated_numeric",
    "_write_dataframe_csv",
]
