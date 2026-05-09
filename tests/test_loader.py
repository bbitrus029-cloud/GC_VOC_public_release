"""Unit tests for raw metadata and breath-data loading."""

from __future__ import annotations

import codecs
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


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

ALIASED_METADATA_COLUMNS = [
    "sample_id",
    "collection_time",
    "collection_location",
    "disease_category",
    "label",
    "filename",
    "age",
    "sex",
    "bmi",
    "smoking",
    "drinking",
    "first_degree_family_history_of_gc",
    "hp_infection",
    "cea",
    "ca199",
    "ca724",
]
METADATA_ALIAS_MAPPING = dict(zip(SOURCE_METADATA_COLUMNS, ALIASED_METADATA_COLUMNS, strict=True))

RAW_CHANNELS = [f"S{index}" for index in range(14)]
EXPECTED_GAS_CHANNELS = [f"S{index}" for index in range(1, 12)]


def _import_loader_symbols() -> tuple:
    """Import loader symbols lazily so pre-implementation failures stay clear."""
    try:
        from src.data.loader import load_dataset, load_metadata, load_raw_record
    except ModuleNotFoundError as exc:
        pytest.fail(f"Loader module is missing: {exc}", pytrace=False)

    return load_dataset, load_metadata, load_raw_record


def _build_metadata_frame(sample_ids: list[int]) -> pd.DataFrame:
    """Build a synthetic metadata frame with the required source columns."""
    rows = []
    for offset, sample_id in enumerate(sample_ids):
        rows.append(
            {
                "sample_id": sample_id,
                "collection_time": f"2026-01-{offset + 1:02d} 08:00:00",
                "collection_location(0outpatient1inpatient)": offset % 2,
                "disease_category(0HC1BGL2LGIN3HGIN4EGC5AGC)": (offset + 1) % 6,
                "label": offset % 2,
                "filename": f"sample_{sample_id}.csv",
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


def _build_aliased_metadata_frame(sample_ids: list[int]) -> pd.DataFrame:
    """Build a synthetic metadata frame using the loader's aliased schema."""
    return _build_metadata_frame(sample_ids).rename(columns=METADATA_ALIAS_MAPPING)


def _build_raw_frame(n_rows: int, columns: list[str] | None = None) -> pd.DataFrame:
    """Build a synthetic raw breath-data frame with deterministic integer channels."""
    column_names = columns if columns is not None else RAW_CHANNELS
    base = np.arange(n_rows, dtype=np.int64)
    data = {
        column_name: base + (column_index * 1000)
        for column_index, column_name in enumerate(column_names)
    }
    return pd.DataFrame(data, columns=column_names)


def _write_metadata_csv(path: Path, metadata_frame: pd.DataFrame) -> None:
    """Write a metadata CSV fixture using the project encoding."""
    metadata_frame.to_csv(path, index=False, encoding="utf-8-sig")


def _write_raw_csv(path: Path, raw_frame: pd.DataFrame, add_bom: bool = False) -> None:
    """Write a raw breath-data CSV fixture with optional explicit BOM bytes."""
    csv_text = raw_frame.to_csv(index=False)
    if add_bom:
        path.write_bytes(codecs.BOM_UTF8 + csv_text.encode("utf-8"))
        return

    path.write_text(csv_text, encoding="utf-8")


def test_load_metadata_renames_columns_and_logs_summary(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Metadata loading should rename source columns and emit an INFO summary."""
    (
        _load_dataset,
        load_metadata,
        _load_raw_record,
    ) = _import_loader_symbols()
    metadata_path = tmp_path / "metadata.csv"
    _write_metadata_csv(metadata_path, _build_metadata_frame([101, 102, 103]))

    with caplog.at_level(logging.INFO):
        metadata = load_metadata(metadata_path)

    assert metadata.shape == (3, 16)
    assert metadata.columns.tolist() == ALIASED_METADATA_COLUMNS
    assert metadata.index.tolist() == [0, 1, 2]
    assert any("Loaded metadata" in record.message for record in caplog.records)
    assert any("missing_counts" in record.message for record in caplog.records)


def test_load_metadata_raises_for_missing_required_source_column(tmp_path: Path) -> None:
    """Metadata loading should name any missing required source columns."""
    (
        _load_dataset,
        load_metadata,
        _load_raw_record,
    ) = _import_loader_symbols()
    metadata_path = tmp_path / "metadata.csv"
    metadata = _build_metadata_frame([101, 102, 103]).drop(columns=["CA724(0.1-10μ/ml)"])
    _write_metadata_csv(metadata_path, metadata)

    with pytest.raises(ValueError, match="CA724\\(0\\.1-10μ/ml\\)"):
        load_metadata(metadata_path)


def test_load_raw_record_splits_channels_and_handles_utf8_bom(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Raw-record loading should preserve gas-channel ordering and integer signal dtypes."""
    (
        _load_dataset,
        _load_metadata,
        load_raw_record,
    ) = _import_loader_symbols()
    metadata_row = _build_aliased_metadata_frame([101]).iloc[0]
    raw_frame = _build_raw_frame(n_rows=1000)
    raw_path = tmp_path / str(metadata_row["filename"])
    _write_raw_csv(raw_path, raw_frame, add_bom=True)

    with caplog.at_level(logging.INFO):
        record = load_raw_record(metadata_row=metadata_row, breath_data_root=tmp_path)

    assert record.sample_id == 101
    assert record.filename == "sample_101.csv"
    assert record.n_rows == 1000
    assert record.gas_signals.shape == (1000, 11)
    assert record.temperature.shape == (1000,)
    assert record.humidity.shape == (1000,)
    assert record.s0.shape == (1000,)
    assert np.issubdtype(record.s0.dtype, np.integer)
    assert np.issubdtype(record.gas_signals.dtype, np.integer)
    assert np.array_equal(record.gas_signals[:, 0], raw_frame["S1"].to_numpy())
    assert np.array_equal(record.gas_signals[:, -1], raw_frame["S11"].to_numpy())
    assert np.array_equal(record.temperature, raw_frame["S12"].to_numpy())
    assert np.array_equal(record.humidity, raw_frame["S13"].to_numpy())
    assert any("sample_id=101" in record.message for record in caplog.records)
    assert any("n_rows=1000" in record.message for record in caplog.records)


def test_load_raw_record_raises_for_schema_mismatch(tmp_path: Path) -> None:
    """Raw-record loading should fail when the file does not match the 14-column schema."""
    (
        _load_dataset,
        _load_metadata,
        load_raw_record,
    ) = _import_loader_symbols()
    metadata_row = _build_aliased_metadata_frame([101]).iloc[0]
    raw_frame = _build_raw_frame(n_rows=1000, columns=RAW_CHANNELS[:-1])
    raw_path = tmp_path / str(metadata_row["filename"])
    _write_raw_csv(raw_path, raw_frame)

    with pytest.raises(ValueError, match=str(raw_path)):
        load_raw_record(metadata_row=metadata_row, breath_data_root=tmp_path)


def test_load_raw_record_keeps_short_rows_without_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Raw-record loading should record short traces without QC-style exclusion behavior."""
    (
        _load_dataset,
        _load_metadata,
        load_raw_record,
    ) = _import_loader_symbols()
    metadata_row = _build_aliased_metadata_frame([101]).iloc[0]
    raw_path = tmp_path / str(metadata_row["filename"])
    _write_raw_csv(raw_path, _build_raw_frame(n_rows=999))

    with caplog.at_level(logging.INFO):
        record = load_raw_record(metadata_row=metadata_row, breath_data_root=tmp_path)

    assert record.n_rows == 999
    assert record.gas_signals.shape == (999, 11)
    assert not any(log_record.levelno >= logging.WARNING for log_record in caplog.records)


def test_load_dataset_preserves_metadata_order_and_traceability(tmp_path: Path) -> None:
    """Dataset loading should return one record per metadata row in the same order."""
    load_dataset, _load_metadata, _load_raw_record = _import_loader_symbols()
    metadata = _build_aliased_metadata_frame([301, 205, 407])
    for metadata_row, n_rows in zip(metadata.to_dict(orient="records"), [1000, 999, 1001], strict=True):
        raw_path = tmp_path / str(metadata_row["filename"])
        _write_raw_csv(raw_path, _build_raw_frame(n_rows=n_rows))

    records = load_dataset(metadata=metadata, breath_data_root=tmp_path)

    assert len(records) == len(metadata)
    assert [record.sample_id for record in records] == metadata["sample_id"].tolist()
    assert [record.filename for record in records] == metadata["filename"].tolist()
    assert all(record.gas_signals.shape[0] == record.n_rows for record in records)


def test_load_dataset_raises_when_a_raw_file_is_missing(tmp_path: Path) -> None:
    """Dataset loading should stop at the first missing raw CSV instead of skipping it."""
    load_dataset, _load_metadata, _load_raw_record = _import_loader_symbols()
    metadata = _build_aliased_metadata_frame([301, 205, 407])
    for metadata_row in metadata.iloc[:2].to_dict(orient="records"):
        raw_path = tmp_path / str(metadata_row["filename"])
        _write_raw_csv(raw_path, _build_raw_frame(n_rows=1000))

    with pytest.raises(FileNotFoundError):
        load_dataset(metadata=metadata, breath_data_root=tmp_path)
