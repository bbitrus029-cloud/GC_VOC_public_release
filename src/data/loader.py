"""Raw metadata and breath-trace loading utilities for preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from src.config import (
    BREATH_DATA_ROOT,
    GAS_CHANNELS,
    HUMIDITY_CHANNEL,
    METADATA_PATH,
    RAW_CHANNELS,
    RAW_CSV_ENCODING,
    S0_DROPPED,
    TEMPERATURE_CHANNEL,
)


LOGGER = logging.getLogger(__name__)

_METADATA_COLUMN_ALIASES: dict[str, str] = {
    "sample_id": "sample_id",
    "collection_time": "collection_time",
    "collection_location(0outpatient1inpatient)": "collection_location",
    "disease_category(0HC1BGL2LGIN3HGIN4EGC5AGC)": "disease_category",
    "label": "label",
    "filename": "filename",
    "age": "age",
    "sex(0male1female)": "sex",
    "BMI": "bmi",
    "smoking(0no1yes)": "smoking",
    "drinking(0no1yes)": "drinking",
    "first_degree_family_history_of_GC(0no1yes)": "first_degree_family_history_of_gc",
    "Hp_Infection(0no1yes)": "hp_infection",
    "CEA(0-5μg/L)": "cea",
    "CA199(0.1-37μ/ml)": "ca199",
    "CA724(0.1-10μ/ml)": "ca724",
}
_REQUIRED_METADATA_COLUMNS = list(_METADATA_COLUMN_ALIASES.keys())
_REQUIRED_METADATA_ROW_FIELDS = [
    "sample_id",
    "filename",
    "label",
    "disease_category",
    "collection_time",
]


@dataclass(frozen=True)
class RawRecord:
    """Separated raw breath signals and traceability metadata for one sample.

    The signal arrays preserve the integer dtype inferred by `pandas.read_csv`
    for the raw CSV columns. For integer-only source files this is typically
    `numpy.int64`, and the loader does not silently coerce the raw signals to
    floating-point values.

    Attributes:
        sample_id: Integer sample identifier from metadata.
        filename: Raw breath CSV filename from metadata.
        n_rows: Number of rows loaded from the raw CSV.
        s0: Diagnostic S0 channel values with shape `(n_rows,)`.
        gas_signals: Gas-sensor matrix for channels `S1` through `S11` with
            shape `(n_rows, 11)`.
        temperature: Ambient temperature channel `S12` with shape `(n_rows,)`.
        humidity: Ambient humidity channel `S13` with shape `(n_rows,)`.
        label: Binary label from metadata for downstream traceability.
        disease_category: Disease-category code from metadata.
        collection_time: Collection timestamp string from metadata.
    """

    sample_id: int
    filename: str
    n_rows: int
    s0: NDArray[np.integer[Any]]
    gas_signals: NDArray[np.integer[Any]]
    temperature: NDArray[np.integer[Any]]
    humidity: NDArray[np.integer[Any]]
    label: int
    disease_category: int
    collection_time: str


def _summarize_dtypes(frame: pd.DataFrame) -> dict[str, str]:
    """Return a stringified dtype summary for logging."""
    return {column: str(dtype) for column, dtype in frame.dtypes.items()}


def _log_dataframe_summary(frame: pd.DataFrame, path: Path, context: str, sample_id: int | None = None) -> None:
    """Log table shape, dtypes, missing counts, and optional sample traceability."""
    if sample_id is None:
        LOGGER.info(
            "Loaded %s from %s with shape=%s dtypes=%s missing_counts=%s",
            context,
            path,
            frame.shape,
            _summarize_dtypes(frame),
            frame.isna().sum().to_dict(),
        )
        return

    LOGGER.info(
        "Loaded %s from %s for sample_id=%s with shape=%s dtypes=%s missing_counts=%s",
        context,
        path,
        sample_id,
        frame.shape,
        _summarize_dtypes(frame),
        frame.isna().sum().to_dict(),
    )


def _validate_metadata_row(metadata_row: pd.Series) -> None:
    """Ensure the metadata row exposes the fields required by the loader."""
    missing_fields = [field for field in _REQUIRED_METADATA_ROW_FIELDS if field not in metadata_row.index]
    if missing_fields:
        raise ValueError(f"metadata_row is missing required fields: {missing_fields}")


def load_metadata(path: Path | None = None) -> pd.DataFrame:
    """Load and validate the clinical metadata table.

    Args:
        path: Optional metadata CSV path. Defaults to `METADATA_PATH`.

    Returns:
        A positionally indexed metadata frame with the prespecified English
        alias columns.

    Raises:
        FileNotFoundError: If the metadata CSV does not exist.
        OSError: If reading the metadata CSV fails for an I/O reason.
        ValueError: If required source columns are missing or the CSV cannot be
            parsed.
    """
    metadata_path = METADATA_PATH if path is None else Path(path)

    try:
        metadata = pd.read_csv(metadata_path, encoding=RAW_CSV_ENCODING)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Failed to read metadata CSV at {metadata_path}: {exc}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Failed to parse metadata CSV at {metadata_path}: {exc}") from exc
    except OSError as exc:
        raise OSError(f"Failed to read metadata CSV at {metadata_path}: {exc}") from exc

    missing_columns = [column for column in _REQUIRED_METADATA_COLUMNS if column not in metadata.columns]
    if missing_columns:
        raise ValueError(f"Metadata CSV at {metadata_path} is missing required columns: {missing_columns}")

    aliased_metadata = metadata.loc[:, _REQUIRED_METADATA_COLUMNS].rename(columns=_METADATA_COLUMN_ALIASES).reset_index(
        drop=True
    )
    _log_dataframe_summary(frame=aliased_metadata, path=metadata_path, context="metadata")
    return aliased_metadata


def load_raw_record(metadata_row: pd.Series, breath_data_root: Path | None = None) -> RawRecord:
    """Load one raw breath CSV and separate its channels.

    Args:
        metadata_row: One metadata row with aliased columns from
            `load_metadata`.
        breath_data_root: Optional raw breath-data directory. Defaults to
            `BREATH_DATA_ROOT`.

    Returns:
        A `RawRecord` containing separated raw channels and metadata
        traceability fields.

    Raises:
        FileNotFoundError: If the raw CSV does not exist.
        OSError: If reading the raw CSV fails for an I/O reason.
        ValueError: If required metadata fields are missing, the raw CSV cannot
            be parsed, or the raw channel schema does not exactly match
            `RAW_CHANNELS`.
    """
    _validate_metadata_row(metadata_row)

    raw_root = BREATH_DATA_ROOT if breath_data_root is None else Path(breath_data_root)
    filename = str(metadata_row["filename"])
    raw_path = raw_root / filename

    try:
        raw_frame = pd.read_csv(raw_path, encoding=RAW_CSV_ENCODING)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Failed to read raw breath CSV at {raw_path}: {exc}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Failed to parse raw breath CSV at {raw_path}: {exc}") from exc
    except OSError as exc:
        raise OSError(f"Failed to read raw breath CSV at {raw_path}: {exc}") from exc

    actual_columns = raw_frame.columns.tolist()
    if actual_columns != RAW_CHANNELS:
        raise ValueError(
            f"Raw breath CSV schema mismatch at {raw_path}: expected {RAW_CHANNELS}, got {actual_columns}"
        )

    sample_id = int(metadata_row["sample_id"])
    _log_dataframe_summary(frame=raw_frame, path=raw_path, context="raw breath table", sample_id=sample_id)

    s0 = raw_frame[S0_DROPPED].to_numpy(copy=True)
    gas_signals = raw_frame.loc[:, GAS_CHANNELS].to_numpy(copy=True)
    temperature = raw_frame[TEMPERATURE_CHANNEL].to_numpy(copy=True)
    humidity = raw_frame[HUMIDITY_CHANNEL].to_numpy(copy=True)
    n_rows = int(raw_frame.shape[0])

    LOGGER.info("Prepared raw record sample_id=%s filename=%s n_rows=%s", sample_id, filename, n_rows)

    return RawRecord(
        sample_id=sample_id,
        filename=filename,
        n_rows=n_rows,
        s0=s0,
        gas_signals=gas_signals,
        temperature=temperature,
        humidity=humidity,
        label=int(metadata_row["label"]),
        disease_category=int(metadata_row["disease_category"]),
        collection_time=str(metadata_row["collection_time"]),
    )


def load_dataset(
    metadata: pd.DataFrame | None = None, breath_data_root: Path | None = None
) -> list[RawRecord]:
    """Load the full raw-breath dataset aligned to metadata rows.

    Args:
        metadata: Optional metadata table. When omitted, `load_metadata()` is
            called.
        breath_data_root: Optional raw breath-data directory. Defaults to
            `BREATH_DATA_ROOT`.

    Returns:
        A list of `RawRecord` objects aligned 1:1 with the metadata rows.

    Raises:
        FileNotFoundError: If any raw CSV is missing.
        OSError: If any raw CSV read fails for an I/O reason.
        ValueError: If the metadata or any raw CSV violates the prespecified
            schema.
    """
    metadata_frame = load_metadata() if metadata is None else metadata.reset_index(drop=True)
    records = [
        load_raw_record(metadata_row=metadata_row, breath_data_root=breath_data_root)
        for _, metadata_row in metadata_frame.iterrows()
    ]

    if not records:
        LOGGER.info("Loaded dataset with total_records=0.")
        return records

    n_rows = np.asarray([record.n_rows for record in records], dtype=np.int64)
    LOGGER.info(
        "Loaded dataset with total_records=%s min_n_rows=%s median_n_rows=%.1f max_n_rows=%s",
        len(records),
        int(np.min(n_rows)),
        float(np.median(n_rows)),
        int(np.max(n_rows)),
    )
    return records


__all__ = ["RawRecord", "load_metadata", "load_raw_record", "load_dataset"]
