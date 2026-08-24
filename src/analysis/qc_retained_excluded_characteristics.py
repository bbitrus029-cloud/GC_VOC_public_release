"""Aggregate QC-retained/excluded clinical characteristics.

Participant-capable loading is available only through ``execute_approved`` after
hash, Git, environment, projection, and stale-output gates. No participant row
or identifier is returned from the production entry point or persisted.
"""

from __future__ import annotations

import io
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.analysis.posthoc_final_reporting import (
    SmdResult,
    describe_continuous,
    signed_binary_smd,
    signed_continuous_smd,
)
from src.config import PROJECT_ROOT


PACKAGE_ID = "qc_retained_excluded_characteristics"
SUMMARY_COLUMNS = (
    "variable",
    "level",
    "group",
    "n_total",
    "n_nonmissing",
    "n_missing",
    "count",
    "denominator_n",
    "proportion",
    "mean",
    "sd",
    "median",
    "q1",
    "q3",
    "denominator_rule",
)
SMD_COLUMNS = (
    "variable",
    "level",
    "comparison_type",
    "smd_qc_retained_minus_excluded",
    "n_qc_retained_used",
    "n_qc_excluded_used",
    "estimability_status",
    "nonestimable_reason",
)
DISEASE_LEVELS = ("HC", "BGL", "LGIN", "HGIN", "EGC", "AGC")
DISTRIBUTION_ROWS = (
    ("age", None),
    ("sex", "female"),
    ("endpoint", "yes"),
    *(("disease_category", level) for level in DISEASE_LEVELS),
    ("body_mass_index", None),
    ("smoking_history", "yes"),
    ("alcohol_use_history", "yes"),
    ("first_degree_family_history_of_gastric_cancer", "yes"),
    ("helicobacter_pylori_status", "positive"),
    ("helicobacter_pylori_status", "negative"),
    ("helicobacter_pylori_status", "missing"),
)
MISSINGNESS_VARIABLES = (
    "age",
    "sex",
    "endpoint",
    "disease_category",
    "body_mass_index",
    "smoking_history",
    "alcohol_use_history",
    "first_degree_family_history_of_gastric_cancer",
)
RAW_COLUMN_BY_VARIABLE = {
    "age": "age",
    "sex": "sex(0male1female)",
    "endpoint": "label",
    "disease_category": "disease_category",
    "body_mass_index": "BMI",
    "smoking_history": "smoking(0no1yes)",
    "alcohol_use_history": "drinking(0no1yes)",
    "first_degree_family_history_of_gastric_cancer": "first_degree_family_history_of_GC(0no1yes)",
    "helicobacter_pylori_status": "Hp_Infection(0no1yes)",
}
CONTINUOUS_VARIABLES = {"age", "body_mass_index"}
EXPECTED_METADATA_COLUMNS = (
    "sample_id",
    "age",
    "sex(0male1female)",
    "BMI",
    "smoking(0no1yes)",
    "drinking(0no1yes)",
    "first_degree_family_history_of_GC(0no1yes)",
    "Hp_Infection(0no1yes)",
)
EXPECTED_INDEX_COLUMNS = ("sample_id", "label", "disease_category", "in_final_output")
ALLOWED_STATUS = {"estimable", "non_estimable"}
ALLOWED_REASONS = ("smd_fewer_than_two_nonmissing", "smd_zero_denominator_nonidentical")
ALLOWED_UNTRACKED = {
    ".vscode/",
    "assets/",
    "configs/qc_retained_excluded_characteristics.yaml",
    "src/analysis/qc_retained_excluded_characteristics.py",
    "scripts/run_qc_retained_excluded_characteristics.py",
    "tests/test_qc_retained_excluded_characteristics.py",
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one whole file.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate the package configuration.

    Args:
        path: Project-relative or absolute YAML path.

    Returns:
        Parsed configuration.

    Raises:
        ValueError: If the package identity or fixed contracts differ.
    """
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != f"{PACKAGE_ID}.config.v1":
        raise ValueError("Configuration schema is invalid.")
    if payload.get("package_id") != PACKAGE_ID:
        raise ValueError("Configuration package identity is invalid.")
    if tuple(payload["input_columns"]["raw_metadata"]) != EXPECTED_METADATA_COLUMNS:
        raise ValueError("Raw-metadata projection contract is invalid.")
    if tuple(payload["input_columns"]["sample_index"]) != EXPECTED_INDEX_COLUMNS:
        raise ValueError("Sample-index projection contract is invalid.")
    if tuple(payload["disease_category_codes"]) != DISEASE_LEVELS or tuple(
        payload["disease_category_codes"].values()
    ) != tuple(range(6)):
        raise ValueError("Disease-category coding contract is invalid.")
    if tuple(tuple(row) for row in payload["variable_order"]) != DISTRIBUTION_ROWS:
        raise ValueError("Variable order contract is invalid.")
    if tuple(payload["missingness_variable_order"]) != MISSINGNESS_VARIABLES:
        raise ValueError("Missingness order contract is invalid.")
    if tuple(payload["summary_columns"]) != SUMMARY_COLUMNS or tuple(payload["smd_columns"]) != SMD_COLUMNS:
        raise ValueError("Output schema contract is invalid.")
    return payload


def _validate_relative_path(value: object) -> Path:
    """Return one safe project-relative configured path."""
    if not isinstance(value, str):
        raise ValueError("Configured artifact path is invalid.")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Configured artifact paths must be project-relative.")
    return path


def _verify_hash_records(root: Path, records: Mapping[str, Mapping[str, str]]) -> OrderedDict[str, str]:
    """Verify an ordered collection of whole-file hash records."""
    observed: OrderedDict[str, str] = OrderedDict()
    for role, record in records.items():
        path = root / _validate_relative_path(record.get("path"))
        expected = record.get("sha256")
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError("Configured SHA-256 value is invalid.")
        if not path.is_file():
            raise ValueError("Approved hashed source is unavailable.")
        digest = sha256_file(path)
        if digest != expected:
            raise ValueError("Approved source hash mismatch before participant-capable read.")
        observed[role] = digest
    return observed


def load_authorized_inputs(
    root: Path,
    config: Mapping[str, Any],
    *,
    reader: Callable[..., pd.DataFrame] = pd.read_csv,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Verify every frozen source, then read only the approved projections.

    Args:
        root: Project root.
        config: Validated package contract.
        reader: CSV reader, injectable for synthetic tests.

    Returns:
        Projected raw metadata and authoritative sample index.

    Raises:
        ValueError: If hashes, projections, or schemas differ.
    """
    _verify_hash_records(root, config["inputs"])
    _verify_hash_records(root, config.get("anchors", {}))
    frames: list[pd.DataFrame] = []
    for role in ("raw_metadata", "sample_index"):
        columns = list(config["input_columns"][role])
        path = root / _validate_relative_path(config["inputs"][role]["path"])
        try:
            frame = reader(path, usecols=columns)
        except (TypeError, ValueError) as exc:
            raise ValueError("Approved projected input schema could not be read.") from exc
        if frame.columns.tolist() != columns:
            raise ValueError("Approved projected input column order is invalid.")
        frames.append(frame)
    return frames[0], frames[1]


def _validate_exact_binary(series: pd.Series) -> None:
    """Require exact numeric 0/1/missing coding without coercion."""
    values = series.loc[~series.isna()].to_numpy()
    if values.size == 0:
        return
    if values.dtype.kind not in "biuf":
        raise ValueError("Binary field coding is invalid.")
    numeric = values.astype(float)
    if not np.all(np.isfinite(numeric)) or not np.all((numeric == 0.0) | (numeric == 1.0)):
        raise ValueError("Binary field coding is invalid.")


def _validate_continuous(series: pd.Series) -> None:
    """Require finite numeric values wherever a continuous field is present."""
    describe_continuous(series.tolist())


def _validate_disease_category(series: pd.Series, codes: Mapping[str, int]) -> None:
    """Require exact integer disease codes from the configured closed set."""
    values = series.loc[~series.isna()].to_numpy()
    if values.size == 0:
        return
    if values.dtype.kind not in "biuf":
        raise ValueError("Disease-category coding is invalid.")
    numeric = values.astype(float)
    allowed = np.asarray(tuple(codes.values()), dtype=float)
    if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)) or not np.all(
        np.isin(numeric, allowed)
    ):
        raise ValueError("Disease-category coding is invalid.")


def _validate_and_join(
    metadata: pd.DataFrame,
    sample_index: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Validate source schemas, coding, counts, and a strict one-to-one join."""
    if tuple(metadata.columns) != tuple(config["input_columns"]["raw_metadata"]):
        raise ValueError("Raw-metadata schema is invalid.")
    if tuple(sample_index.columns) != tuple(config["input_columns"]["sample_index"]):
        raise ValueError("Sample-index schema is invalid.")
    expected = config["population_counts"]
    if len(metadata) != int(expected["complete_source"]) or len(sample_index) != int(expected["complete_source"]):
        raise ValueError("Complete source count does not match the approved contract.")
    for frame in (metadata, sample_index):
        if frame["sample_id"].isna().any() or not frame["sample_id"].is_unique:
            raise ValueError("Join keys must be complete and unique.")
    if set(metadata["sample_id"].tolist()) != set(sample_index["sample_id"].tolist()):
        raise ValueError("Join-key populations do not match.")
    for column in EXPECTED_METADATA_COLUMNS[2:]:
        if column != "BMI":
            _validate_exact_binary(metadata[column])
    _validate_exact_binary(sample_index["label"])
    _validate_continuous(metadata["age"])
    _validate_continuous(metadata["BMI"])
    _validate_disease_category(sample_index["disease_category"], config["disease_category_codes"])
    flags = sample_index["in_final_output"]
    if flags.isna().any() or not all(type(value) in {bool, np.bool_} for value in flags.tolist()):
        raise ValueError("QC-retention flag coding is invalid.")
    joined = metadata.merge(sample_index, on="sample_id", how="inner", validate="one_to_one", sort=False)
    retained_n = int(joined["in_final_output"].sum())
    excluded_n = int(len(joined) - retained_n)
    if retained_n != int(expected["qc_retained"]) or excluded_n != int(expected["qc_excluded"]):
        raise ValueError("QC-retained or QC-excluded count does not match the approved contract.")
    endpoint = joined["label"]
    disease = joined["disease_category"]
    complete = endpoint.notna() & disease.notna()
    positive_codes = {
        config["disease_category_codes"][level] for level in ("HGIN", "EGC", "AGC")
    }
    expected_endpoint = disease.loc[complete].astype(int).isin(positive_codes).astype(int)
    if not np.array_equal(endpoint.loc[complete].astype(int).to_numpy(), expected_endpoint.to_numpy()):
        raise ValueError("Endpoint and disease-category coding do not agree.")
    return joined


def _indicator(values: pd.Series, focal: int) -> pd.Series:
    """Create one nullable complete-case focal indicator."""
    return values.map(lambda value: np.nan if pd.isna(value) else int(int(value) == focal))


def _distribution_values(
    frame: pd.DataFrame,
    variable: str,
    level: str | None,
    codes: Mapping[str, int],
) -> pd.Series:
    """Return the exact numeric vector for one planned distribution contrast."""
    raw = frame[RAW_COLUMN_BY_VARIABLE[variable]]
    if variable in CONTINUOUS_VARIABLES or variable in {
        "sex",
        "endpoint",
        "smoking_history",
        "alcohol_use_history",
        "first_degree_family_history_of_gastric_cancer",
    }:
        return raw
    if variable == "disease_category":
        return _indicator(raw, codes[str(level)])
    if variable == "helicobacter_pylori_status":
        if level == "missing":
            return raw.isna().astype(int)
        return _indicator(raw, 1 if level == "positive" else 0).fillna(0).astype(int)
    raise ValueError("Unknown fixed variable inventory entry.")


def _empty_summary_record(variable: str, level: str | None, group: str) -> OrderedDict[str, Any]:
    """Create one fully ordered nullable summary record."""
    record = OrderedDict((column, None) for column in SUMMARY_COLUMNS)
    record.update(variable=variable, level=level, group=group)
    return record


def _apply_summary_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic nullable dtypes for exact CSV serialization."""
    result = frame.copy()
    for column in ("variable", "level", "denominator_rule"):
        result[column] = result[column].astype("string")
    result["group"] = pd.Categorical(result["group"], categories=["qc_retained", "qc_excluded"], ordered=True)
    for column in ("n_total", "n_nonmissing", "n_missing", "count", "denominator_n"):
        result[column] = result[column].astype("Int64")
    for column in ("proportion", "mean", "sd", "median", "q1", "q3"):
        result[column] = result[column].astype("Float64")
    return result.loc[:, SUMMARY_COLUMNS]


def _apply_smd_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic nullable dtypes for exact SMD serialization."""
    result = frame.copy()
    for column in ("variable", "level", "nonestimable_reason"):
        result[column] = result[column].astype("string")
    result["comparison_type"] = pd.Categorical(
        result["comparison_type"], categories=["distribution", "missingness"], ordered=True
    )
    result["estimability_status"] = pd.Categorical(
        result["estimability_status"], categories=["estimable", "non_estimable"], ordered=True
    )
    result["smd_qc_retained_minus_excluded"] = result["smd_qc_retained_minus_excluded"].astype("Float64")
    for column in ("n_qc_retained_used", "n_qc_excluded_used"):
        result[column] = result[column].astype("Int64")
    return result.loc[:, SMD_COLUMNS]


def build_characteristics(
    metadata: pd.DataFrame,
    sample_index: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate the fixed aggregate summary and signed SMD inventories.

    Args:
        metadata: Exact approved clinical projection.
        sample_index: Exact authoritative QC/endpoint/category projection.
        config: Package configuration.

    Returns:
        The 32-row summary and 24-row SMD frames.

    Raises:
        ValueError: If any schema, coding, count, or join contract differs.
    """
    joined = _validate_and_join(metadata, sample_index, config)
    groups = OrderedDict(
        [
            ("qc_retained", joined.loc[joined["in_final_output"]]),
            ("qc_excluded", joined.loc[~joined["in_final_output"]]),
        ]
    )
    codes = config["disease_category_codes"]
    summary_records: list[OrderedDict[str, Any]] = []
    for group, frame in groups.items():
        for variable, level in DISTRIBUTION_ROWS:
            record = _empty_summary_record(variable, level, group)
            raw = frame[RAW_COLUMN_BY_VARIABLE[variable]]
            record["n_total"] = int(len(frame))
            record["n_nonmissing"] = int(raw.notna().sum())
            record["n_missing"] = int(raw.isna().sum())
            if variable in CONTINUOUS_VARIABLES:
                description = describe_continuous(raw.tolist())
                for key in ("mean", "sd", "median", "q1", "q3"):
                    record[key] = description[key]
                record["denominator_rule"] = "not_applicable"
            else:
                values = _distribution_values(frame, variable, level, codes)
                if variable == "helicobacter_pylori_status":
                    denominator = len(frame)
                    record["denominator_rule"] = "group_total"
                else:
                    denominator = int(values.notna().sum())
                    record["denominator_rule"] = "variable_specific_nonmissing"
                count = int(values.dropna().sum())
                record["count"] = count
                record["denominator_n"] = denominator
                record["proportion"] = float(count / denominator) if denominator else None
            summary_records.append(record)

    smd_records: list[OrderedDict[str, Any]] = []
    retained = groups["qc_retained"]
    excluded = groups["qc_excluded"]
    for variable, level in DISTRIBUTION_ROWS:
        left = _distribution_values(retained, variable, level, codes)
        right = _distribution_values(excluded, variable, level, codes)
        result = (
            signed_continuous_smd(left, right)
            if variable in CONTINUOUS_VARIABLES
            else signed_binary_smd(left, right)
        )
        smd_records.append(_smd_record(variable, level, "distribution", left, right, result))
    for variable in MISSINGNESS_VARIABLES:
        column = RAW_COLUMN_BY_VARIABLE[variable]
        left = retained[column].isna().astype(int)
        right = excluded[column].isna().astype(int)
        result = signed_binary_smd(left, right)
        smd_records.append(_smd_record(variable, "missing", "missingness", left, right, result))
    summary = _apply_summary_dtypes(pd.DataFrame(summary_records, columns=SUMMARY_COLUMNS))
    smd = _apply_smd_dtypes(pd.DataFrame(smd_records, columns=SMD_COLUMNS))
    validate_summary(summary, config)
    validate_smd(smd, config)
    return summary, smd


def _smd_record(
    variable: str,
    level: str | None,
    comparison_type: str,
    retained: pd.Series,
    excluded: pd.Series,
    result: SmdResult,
) -> OrderedDict[str, Any]:
    """Build one ordered signed-SMD record from the reused frozen result."""
    return OrderedDict(
        [
            ("variable", variable),
            ("level", level),
            ("comparison_type", comparison_type),
            ("smd_qc_retained_minus_excluded", result.estimate),
            ("n_qc_retained_used", int(retained.notna().sum())),
            ("n_qc_excluded_used", int(excluded.notna().sum())),
            ("estimability_status", result.status),
            ("nonestimable_reason", result.reason),
        ]
    )


def _level_value(value: object) -> str | None:
    """Normalize one nullable CSV level for identity validation."""
    return None if pd.isna(value) else str(value)


def validate_summary(frame: pd.DataFrame, config: Mapping[str, Any]) -> None:
    """Validate the complete aggregate summary contract.

    Args:
        frame: In-memory or read-back summary.
        config: Package configuration.

    Raises:
        ValueError: If schema, order, counts, denominators, or fields differ.
    """
    if frame.columns.tolist() != list(SUMMARY_COLUMNS) or len(frame) != 32:
        raise ValueError("Summary schema or row count is invalid.")
    expected_identity = [
        (variable, level, group)
        for group in ("qc_retained", "qc_excluded")
        for variable, level in DISTRIBUTION_ROWS
    ]
    observed_identity = [
        (str(row.variable), _level_value(row.level), str(row.group)) for row in frame.itertuples()
    ]
    if observed_identity != expected_identity:
        raise ValueError("Summary scientific row identity or order is invalid.")
    expected_counts = config["population_counts"]
    for row in frame.itertuples():
        group_n = int(expected_counts[str(row.group)])
        if int(row.n_total) != group_n or int(row.n_nonmissing) + int(row.n_missing) != group_n:
            raise ValueError("Summary group or completeness counts are invalid.")
        if row.variable in CONTINUOUS_VARIABLES:
            if str(row.denominator_rule) != "not_applicable" or not all(
                pd.isna(value) for value in (row.count, row.denominator_n, row.proportion)
            ):
                raise ValueError("Continuous summary field population is invalid.")
        else:
            expected_rule = (
                "group_total"
                if row.variable == "helicobacter_pylori_status"
                else "variable_specific_nonmissing"
            )
            expected_denominator = group_n if expected_rule == "group_total" else int(row.n_nonmissing)
            if str(row.denominator_rule) != expected_rule or int(row.denominator_n) != expected_denominator:
                raise ValueError("Summary denominator contract is invalid.")
            if not 0 <= int(row.count) <= expected_denominator:
                raise ValueError("Summary count is invalid.")
            expected_proportion = int(row.count) / expected_denominator if expected_denominator else None
            if expected_proportion is None:
                if not pd.isna(row.proportion):
                    raise ValueError("Summary empty-denominator proportion is invalid.")
            elif float(row.proportion) != expected_proportion:
                raise ValueError("Summary proportion does not reconcile.")
    for group in ("qc_retained", "qc_excluded"):
        disease = frame.loc[(frame["group"].astype(str) == group) & (frame["variable"] == "disease_category")]
        if int(disease["count"].sum()) != int(disease.iloc[0]["denominator_n"]):
            raise ValueError("Disease-category counts do not reconcile.")
        hp = frame.loc[(frame["group"].astype(str) == group) & (frame["variable"] == "helicobacter_pylori_status")]
        if int(hp["count"].sum()) != int(hp.iloc[0]["n_total"]):
            raise ValueError("H pylori category counts do not reconcile.")


def validate_smd(frame: pd.DataFrame, config: Mapping[str, Any]) -> None:
    """Validate the complete signed-SMD schema, order, support, and status fields."""
    if frame.columns.tolist() != list(SMD_COLUMNS) or len(frame) != 24:
        raise ValueError("SMD schema or row count is invalid.")
    expected_identity = [(*row, "distribution") for row in DISTRIBUTION_ROWS] + [
        (variable, "missing", "missingness") for variable in MISSINGNESS_VARIABLES
    ]
    observed_identity = [
        (str(row.variable), _level_value(row.level), str(row.comparison_type)) for row in frame.itertuples()
    ]
    if observed_identity != expected_identity:
        raise ValueError("SMD scientific row identity or order is invalid.")
    counts = config["population_counts"]
    for row in frame.itertuples():
        if not 0 <= int(row.n_qc_retained_used) <= int(counts["qc_retained"]) or not 0 <= int(
            row.n_qc_excluded_used
        ) <= int(counts["qc_excluded"]):
            raise ValueError("SMD support count is invalid.")
        status = str(row.estimability_status)
        reason = _level_value(row.nonestimable_reason)
        missing = pd.isna(row.smd_qc_retained_minus_excluded)
        if status not in ALLOWED_STATUS:
            raise ValueError("SMD estimability status is invalid.")
        if status == "estimable" and (missing or reason is not None):
            raise ValueError("Estimable SMD field population is invalid.")
        if status == "non_estimable" and (not missing or reason not in ALLOWED_REASONS):
            raise ValueError("Non-estimable SMD field population is invalid.")


def serialize_csv(frame: pd.DataFrame, columns: Sequence[str]) -> bytes:
    """Serialize one aggregate frame deterministically with empty missing fields."""
    if frame.columns.tolist() != list(columns):
        raise ValueError("CSV serialization schema is invalid.")
    stream = io.StringIO(newline="")
    frame.to_csv(stream, index=False, columns=list(columns), na_rep="", lineterminator="\n")
    return stream.getvalue().encode("utf-8")


def validate_persisted_privacy(payloads: Mapping[str, bytes]) -> None:
    """Reject participant-capable fields, absolute paths, dates, and filenames.

    Args:
        payloads: Named serialized aggregate products.

    Raises:
        ValueError: If any prohibited content is detected. The matched content is never echoed.
    """
    forbidden = (
        "sample_id",
        "participant_id",
        "collection_time",
        "collection_location",
        "breath_filename",
        "breath_data",
        ".tdms",
        '"p_cal"',
        "participant_level",
    )
    for payload in payloads.values():
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Persisted-product privacy validation failed.") from exc
        lowered = text.lower()
        if any(token in lowered for token in forbidden):
            raise ValueError("Persisted-product privacy validation failed.")
        if re.search(r"(?:^|[\"'])/(?:home|tmp|mnt|var|root)/", text, flags=re.MULTILINE):
            raise ValueError("Persisted-product privacy validation failed.")
        if re.search(r"(?<!\d)(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}(?!\d)", text):
            raise ValueError("Persisted-product privacy validation failed.")


def serialize_manifest(payload: Mapping[str, Any]) -> bytes:
    """Serialize compact deterministic JSON after privacy validation."""
    encoded = (json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    validate_persisted_privacy({"manifest": encoded})
    return encoded


def read_and_validate_summary(path: Path, config: Mapping[str, Any]) -> pd.DataFrame:
    """Read a summary CSV back with exact nullable dtypes and validate it."""
    frame = pd.read_csv(path, float_precision="round_trip")
    if frame.columns.tolist() != list(SUMMARY_COLUMNS):
        raise ValueError("Summary schema or row count is invalid.")
    frame = _apply_summary_dtypes(frame)
    validate_summary(frame, config)
    return frame


def read_and_validate_smd(path: Path, config: Mapping[str, Any]) -> pd.DataFrame:
    """Read an SMD CSV back with exact nullable dtypes and validate it."""
    frame = pd.read_csv(path, float_precision="round_trip")
    if frame.columns.tolist() != list(SMD_COLUMNS):
        raise ValueError("SMD schema or row count is invalid.")
    frame = _apply_smd_dtypes(frame)
    validate_smd(frame, config)
    return frame


def read_and_validate_manifest(path: Path) -> dict[str, Any]:
    """Read and validate the deterministic manifest identity."""
    payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=OrderedDict)
    if payload.get("schema_version") != f"{PACKAGE_ID}.manifest.v1" or payload.get("package_id") != PACKAGE_ID:
        raise ValueError("Manifest schema or package identity is invalid.")
    validate_persisted_privacy({"manifest": path.read_bytes()})
    return payload


def _exact_nullable(left: object, right: object) -> bool:
    """Compare persisted scalar values with equal missing values."""
    if pd.isna(left) and pd.isna(right):
        return True
    return bool(left == right)


def _aggregate_scalar(value: object) -> str | int | float | bool | None:
    """Normalize one approved aggregate scalar for a bounded diagnostic."""
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)) and not isinstance(value, (np.bool_, bool)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError("Aggregate reconciliation encountered a nonfinite scalar.")
        return numeric
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return str(value)


def _record_aggregate_mismatch(
    mismatches: list[OrderedDict[str, Any]],
    *,
    kind: str,
    group: str,
    variable: str,
    level: str | None,
    field: str,
    observed: object,
    expected: object,
) -> None:
    """Append one fixed-schema aggregate-only mismatch when scalars differ."""
    if _exact_nullable(observed, expected):
        return
    mismatches.append(
        OrderedDict(
            [
                ("kind", kind),
                ("group", group),
                ("variable", variable),
                ("level", level),
                ("field", field),
                ("observed", _aggregate_scalar(observed)),
                ("expected", _aggregate_scalar(expected)),
            ]
        )
    )


def reconcile_overlap(
    summary: pd.DataFrame,
    smd: pd.DataFrame,
    root: Path,
    config: Mapping[str, Any],
) -> OrderedDict[str, Any]:
    """Require exact agreement with frozen overlapping aggregate evidence."""
    summary_path = root / _validate_relative_path(config["anchors"]["overlap_summary"]["path"])
    smd_path = root / _validate_relative_path(config["anchors"]["overlap_smd"]["path"])
    old_summary = pd.read_csv(summary_path, float_precision="round_trip")
    old_smd = pd.read_csv(smd_path, float_precision="round_trip")
    group_map = {"qc_retained": "included", "qc_excluded": "excluded"}
    overlap_rows = DISTRIBUTION_ROWS[:9]
    summary_fields = (
        "n_total",
        "n_nonmissing",
        "n_missing",
        "count",
        "proportion",
        "mean",
        "sd",
        "median",
        "q1",
        "q3",
    )
    mismatches: list[OrderedDict[str, Any]] = []
    for group, old_group in group_map.items():
        for variable, level in overlap_rows:
            new = summary.loc[
                (summary["group"].astype(str) == group)
                & (summary["variable"] == variable)
                & (summary["level"].isna() if level is None else summary["level"].eq(level))
            ].iloc[0]
            old = old_summary.loc[
                (old_summary["group"] == old_group)
                & (old_summary["variable"] == variable)
                & (old_summary["level"].isna() if level is None else old_summary["level"].eq(level))
            ].iloc[0]
            for field in summary_fields:
                _record_aggregate_mismatch(
                    mismatches,
                    kind="summary_distribution",
                    group=group,
                    variable=variable,
                    level=level,
                    field=field,
                    observed=new[field],
                    expected=old[field],
                )
        for variable in ("age", "sex", "endpoint", "disease_category"):
            new = summary.loc[
                (summary["group"].astype(str) == group) & (summary["variable"] == variable)
            ].iloc[0]
            old = old_summary.loc[
                (old_summary["group"] == old_group) & (old_summary["variable"] == f"missingness__{variable}")
            ].iloc[0]
            _record_aggregate_mismatch(
                mismatches,
                kind="summary_missingness",
                group=group,
                variable=variable,
                level="missing",
                field="count",
                observed=new["n_missing"],
                expected=old["count"],
            )
            _record_aggregate_mismatch(
                mismatches,
                kind="summary_missingness",
                group=group,
                variable=variable,
                level="missing",
                field="proportion",
                observed=int(new["n_missing"]) / int(new["n_total"]),
                expected=old["proportion"],
            )
    for variable, level in overlap_rows:
        new = smd.loc[
            (smd["comparison_type"].astype(str) == "distribution")
            & (smd["variable"] == variable)
            & (smd["level"].isna() if level is None else smd["level"].eq(level))
        ].iloc[0]
        old = old_smd.loc[
            (old_smd["variable"] == variable)
            & (old_smd["level"].isna() if level is None else old_smd["level"].eq(level))
        ].iloc[0]
        pairs = (
            (
                "smd_qc_retained_minus_excluded",
                new["smd_qc_retained_minus_excluded"],
                old["smd_included_minus_excluded"],
            ),
            ("n_qc_retained_used", new["n_qc_retained_used"], old["n_included_nonmissing"]),
            ("n_qc_excluded_used", new["n_qc_excluded_used"], old["n_excluded_nonmissing"]),
            ("estimability_status", str(new["estimability_status"]), str(old["estimability_status"])),
            ("nonestimable_reason", new["nonestimable_reason"], old["nonestimable_reason"]),
        )
        for field, observed, expected in pairs:
            _record_aggregate_mismatch(
                mismatches,
                kind="distribution_smd",
                group="qc_retained_minus_qc_excluded",
                variable=variable,
                level=level,
                field=field,
                observed=observed,
                expected=expected,
            )
    old_missing_smd_rows = {"age": "age", "sex": "sex", "endpoint": "endpoint", "disease_category": "disease_category"}
    for variable, old_variable in old_missing_smd_rows.items():
        new = smd.loc[
            (smd["comparison_type"].astype(str) == "missingness") & (smd["variable"] == variable)
        ].iloc[0]
        old = old_smd.loc[old_smd["variable"] == old_variable].iloc[0]
        if variable == "disease_category":
            old = old_smd.loc[(old_smd["variable"] == old_variable) & (old_smd["level"] == "HC")].iloc[0]
        _record_aggregate_mismatch(
            mismatches,
            kind="missingness_smd",
            group="qc_retained_minus_qc_excluded",
            variable=variable,
            level="missing",
            field="smd_qc_retained_minus_excluded",
            observed=new["smd_qc_retained_minus_excluded"],
            expected=old["missingness_smd_included_minus_excluded"],
        )
    if mismatches:
        raise ValueError(json.dumps(mismatches, ensure_ascii=True, separators=(",", ":"), allow_nan=False))
    return OrderedDict(
        [
            ("summary_distribution_rows", 18),
            ("summary_missingness_variables", 4),
            ("distribution_smd_rows", 9),
            ("missingness_smd_rows", 4),
            ("exact_full_precision_agreement", True),
            ("estimability_state_agreement", True),
        ]
    )


def _parse_count_denominator(value: object) -> tuple[int, int]:
    """Parse one aggregate count/denominator display cell."""
    match = re.fullmatch(r"(\d+)/(\d+) \([0-9.]+%\)", str(value))
    if match is None:
        raise ValueError("Retained-group aggregate anchor format is invalid.")
    return int(match.group(1)), int(match.group(2))


def reconcile_retained_group(
    summary: pd.DataFrame,
    root: Path,
    config: Mapping[str, Any],
) -> OrderedDict[str, Any]:
    """Reconcile newly added retained-group variables to the frozen aggregate anchor."""
    path = root / _validate_relative_path(config["anchors"]["retained_group_summary"]["path"])
    anchor = pd.read_csv(path, usecols=["variable", "Total (N=972)"])
    values = dict(zip(anchor["variable"], anchor["Total (N=972)"], strict=True))
    retained = summary.loc[summary["group"].astype(str) == "qc_retained"]
    bmi = retained.loc[retained["variable"] == "body_mass_index"].iloc[0]
    observed_bmi = f"{float(bmi['median']):.2f} [{float(bmi['q1']):.2f}–{float(bmi['q3']):.2f}]"
    if observed_bmi != str(values["BMI, kg/m² (median [IQR])"]):
        raise ValueError("Retained BMI display-precision reconciliation failed.")
    row_names = {
        "smoking_history": "Current/ever smoker, n (%)",
        "alcohol_use_history": "Current/ever drinker, n (%)",
        "first_degree_family_history_of_gastric_cancer": "First-degree family history of GC, n (%)",
    }
    expected_counts = {
        "smoking_history": 277,
        "alcohol_use_history": 238,
        "first_degree_family_history_of_gastric_cancer": 50,
    }
    for variable, row_name in row_names.items():
        count, denominator = _parse_count_denominator(values[row_name])
        new = retained.loc[retained["variable"] == variable].iloc[0]
        if (count, denominator) != (expected_counts[variable], 972) or (
            int(new["count"]), int(new["denominator_n"])
        ) != (count, denominator):
            raise ValueError("Retained lifestyle/family-history reconciliation failed.")
    hp_positive_anchor = _parse_count_denominator(values["Hp_infection positive, n (%)"])
    hp_missing_anchor = _parse_count_denominator(values["Hp_infection missing, n (%)"])
    hp = retained.loc[retained["variable"] == "helicobacter_pylori_status"].set_index("level")
    positive = int(hp.loc["positive", "count"])
    missing = int(hp.loc["missing", "count"])
    negative = int(hp.loc["negative", "count"])
    if hp_positive_anchor != (206, 878) or hp_missing_anchor != (94, 972):
        raise ValueError("Frozen retained H pylori anchor is invalid.")
    if (positive, missing, negative) != (206, 94, 672) or positive + missing + negative != 972:
        raise ValueError("Retained H pylori three-level reconciliation failed.")
    if not (hp["denominator_n"].astype(int) == 972).all():
        raise ValueError("Retained H pylori group-total denominator was not applied.")
    return OrderedDict(
        [
            ("body_mass_index_two_decimal_display_agreement", True),
            ("smoking_count", 277),
            ("alcohol_count", 238),
            ("family_history_count", 50),
            ("helicobacter_pylori_positive_count", 206),
            ("helicobacter_pylori_negative_count", 672),
            ("helicobacter_pylori_missing_count", 94),
            ("helicobacter_pylori_group_total_denominator", 972),
        ]
    )


def ensure_targets_absent(targets: Mapping[str, Path]) -> None:
    """Refuse final or destination-local temporary output collisions."""
    for target in targets.values():
        temporary = target.with_name(f".{target.name}.tmp")
        if target.exists() or target.is_symlink() or temporary.exists() or temporary.is_symlink():
            raise FileExistsError("An output target or temporary output target already exists.")


def promote_serialized_outputs(targets: Mapping[str, Path], payloads: Mapping[str, bytes]) -> None:
    """Stage all products locally and atomically expose the manifest last.

    Args:
        targets: Ordered summary, SMD, and manifest destinations.
        payloads: Corresponding serialized bytes.

    Raises:
        FileExistsError: If any final or temporary target exists.
        ValueError: If the inventory or staged read-back differs.
    """
    if tuple(targets) != ("summary", "smd", "manifest") or tuple(payloads) != tuple(targets):
        raise ValueError("Promotion inventory must be summary, SMD, then manifest.")
    ensure_targets_absent(targets)
    temporaries: OrderedDict[str, Path] = OrderedDict()
    try:
        for name, target in targets.items():
            if not target.parent.is_dir():
                raise ValueError("Output parent directory is unavailable.")
            temporary = target.with_name(f".{target.name}.tmp")
            with temporary.open("xb") as stream:
                stream.write(payloads[name])
                stream.flush()
                os.fsync(stream.fileno())
            if temporary.read_bytes() != payloads[name]:
                raise ValueError("Staged output byte read-back failed.")
            temporaries[name] = temporary
        for name in ("summary", "smd", "manifest"):
            os.replace(temporaries[name], targets[name])
    finally:
        for temporary in temporaries.values():
            if temporary.exists() and not temporary.is_symlink():
                temporary.unlink()


def _git_output(root: Path, *args: str) -> str:
    """Run one read-only Git command with sanitized failure handling."""
    try:
        completed = subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError("Read-only Git authority validation failed.") from exc
    return completed.stdout.strip()


def _validate_execution_authority(root: Path, config: Mapping[str, Any]) -> str:
    """Require the approved HEAD, tracked cleanliness, environment, and target state."""
    head = _git_output(root, "rev-parse", "HEAD")
    if head != config["expected_git_head"]:
        raise ValueError("Current Git HEAD does not match the approved starting authority.")
    if _git_output(root, "diff", "--name-only") or _git_output(root, "diff", "--cached", "--name-only"):
        raise ValueError("Tracked worktree changes are not authorized.")
    status = _git_output(root, "status", "--short")
    for line in status.splitlines():
        if not line:
            continue
        if not line.startswith("?? ") or line[3:] not in ALLOWED_UNTRACKED:
            raise ValueError("Unexpected worktree state is not authorized.")
    executable = Path(sys.executable).resolve()
    expected_environment = (Path.home() / "miniforge3/envs/gcvocs/bin").resolve()
    if executable.parent != expected_environment:
        raise ValueError("Python is not running from the required gcvocs environment.")
    return head


def _frames_exact(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    """Return whether two scientific frames are exactly identical."""
    try:
        pd.testing.assert_frame_equal(left, right, check_exact=True)
    except AssertionError:
        return False
    return True


def _artifact_records(root: Path, records: Mapping[str, Mapping[str, str]]) -> list[OrderedDict[str, str]]:
    """Build ordered relative-path/hash provenance records."""
    return [
        OrderedDict([("role", role), ("path", str(record["path"])), ("sha256", sha256_file(root / record["path"]))])
        for role, record in records.items()
    ]


def _build_manifest(
    *,
    root: Path,
    config: Mapping[str, Any],
    git_head: str,
    summary_bytes: bytes,
    smd_bytes: bytes,
    overlap: Mapping[str, Any],
    retained_reconciliation: Mapping[str, Any],
) -> OrderedDict[str, Any]:
    """Build the exact deterministic aggregate provenance manifest."""
    code_records = [
        OrderedDict([("role", role), ("path", path), ("sha256", sha256_file(root / path))])
        for role, path in config["code"].items()
    ]
    configuration_path = str(config["configuration"]["path"])
    output_records = [
        OrderedDict(
            [
                ("role", "summary"),
                ("path", config["outputs"]["summary"]["path"]),
                ("sha256", hashlib.sha256(summary_bytes).hexdigest()),
            ]
        ),
        OrderedDict(
            [
                ("role", "smd"),
                ("path", config["outputs"]["smd"]["path"]),
                ("sha256", hashlib.sha256(smd_bytes).hexdigest()),
            ]
        ),
        OrderedDict(
            [
                ("role", "manifest"),
                ("path", config["outputs"]["manifest"]["path"]),
                ("sha256", None),
                ("self_hash_policy", "forbidden_circular_self_hash"),
            ]
        ),
    ]
    return OrderedDict(
        [
            ("schema_version", f"{PACKAGE_ID}.manifest.v1"),
            ("package_id", PACKAGE_ID),
            ("package_status", "plan_orthogonal_descriptive_source"),
            ("git_head", git_head),
            ("inputs", _artifact_records(root, config["inputs"])),
            ("anchors", _artifact_records(root, config["anchors"])),
            ("code", code_records),
            (
                "configuration",
                OrderedDict([("path", configuration_path), ("sha256", sha256_file(root / configuration_path))]),
            ),
            ("outputs", output_records),
            (
                "population_counts",
                OrderedDict(
                    [
                        ("complete_source", int(config["population_counts"]["complete_source"])),
                        ("qc_retained", int(config["population_counts"]["qc_retained"])),
                        ("qc_excluded", int(config["population_counts"]["qc_excluded"])),
                    ]
                ),
            ),
            ("variable_order", [list(row) for row in DISTRIBUTION_ROWS]),
            ("disease_category_order", list(DISEASE_LEVELS)),
            ("helicobacter_pylori_status_order", ["positive", "negative", "missing"]),
            ("missingness_variable_order", list(MISSINGNESS_VARIABLES)),
            (
                "statistical_contract",
                OrderedDict(
                    [
                        (
                            "continuous_smd_formula",
                            "(mean_qc_retained-mean_qc_excluded)/sqrt((var_qc_retained+var_qc_excluded)/2)",
                        ),
                        (
                            "binary_smd_formula",
                            "(p_qc_retained-p_qc_excluded)/sqrt((p_qc_retained*(1-p_qc_retained)"
                            "+p_qc_excluded*(1-p_qc_excluded))/2)",
                        ),
                        ("signed_direction", "qc_retained_minus_qc_excluded"),
                        ("continuous_sd_ddof", 1),
                        ("quantile_method", "linear"),
                        (
                            "denominator_rules",
                            OrderedDict(
                                [
                                    ("continuous", "not_applicable"),
                                    ("binary_and_disease_category", "variable_specific_nonmissing"),
                                    ("helicobacter_pylori_status", "group_total"),
                                ]
                            ),
                        ),
                        ("imputation", "none"),
                        ("nonestimable_reasons", list(ALLOWED_REASONS)),
                    ]
                ),
            ),
            (
                "output_contracts",
                OrderedDict(
                    [
                        ("summary", OrderedDict([("columns", list(SUMMARY_COLUMNS)), ("row_count", 32)])),
                        ("smd", OrderedDict([("columns", list(SMD_COLUMNS)), ("row_count", 24)])),
                        ("csv_missing_representation", "empty_field"),
                        ("scientific_value_rounding", "none"),
                    ]
                ),
            ),
            (
                "environment",
                OrderedDict(
                    [
                        ("python", ".".join(str(value) for value in sys.version_info[:3])),
                        ("numpy", np.__version__),
                        ("pandas", pd.__version__),
                    ]
                ),
            ),
            ("overlap_reconciliation", OrderedDict(overlap)),
            ("retained_group_reconciliation", OrderedDict(retained_reconciliation)),
            (
                "validation",
                OrderedDict(
                    [
                        ("candidate_read_back", True),
                        ("deterministic_rerun", True),
                        ("persisted_product_privacy", True),
                        ("frozen_source_immutability", True),
                    ]
                ),
            ),
        ]
    )


def execute_approved(config_path: Path, *, root: Path = PROJECT_ROOT) -> OrderedDict[str, Any]:
    """Execute the single authorized aggregate-only production package.

    Args:
        config_path: Exact package YAML path.
        root: Project root; injectable only for synthetic tests.

    Returns:
        Aggregate population counts and output hashes.

    Raises:
        ValueError: If any authority, scientific, reconciliation, privacy, or read-back gate fails.
        FileExistsError: If any final or temporary target already exists.
    """
    resolved_config = config_path if config_path.is_absolute() else root / config_path
    config = load_config(resolved_config)
    if resolved_config.resolve() != (root / config["configuration"]["path"]).resolve():
        raise ValueError("Configuration path does not match the approved contract.")
    git_head = _validate_execution_authority(root, config)
    targets = OrderedDict(
        (name, root / _validate_relative_path(record["path"])) for name, record in config["outputs"].items()
    )
    ensure_targets_absent(targets)
    before_inputs = _verify_hash_records(root, config["inputs"])
    before_anchors = _verify_hash_records(root, config["anchors"])
    metadata, sample_index = load_authorized_inputs(root, config)
    summary, smd = build_characteristics(metadata, sample_index, config)
    repeated_summary, repeated_smd = build_characteristics(metadata, sample_index, config)
    if not _frames_exact(summary, repeated_summary) or not _frames_exact(smd, repeated_smd):
        raise ValueError("Deterministic scientific rerun validation failed.")
    overlap = reconcile_overlap(summary, smd, root, config)
    retained_reconciliation = reconcile_retained_group(summary, root, config)
    summary_bytes = serialize_csv(summary, SUMMARY_COLUMNS)
    smd_bytes = serialize_csv(smd, SMD_COLUMNS)
    if summary_bytes != serialize_csv(repeated_summary, SUMMARY_COLUMNS) or smd_bytes != serialize_csv(
        repeated_smd, SMD_COLUMNS
    ):
        raise ValueError("Deterministic serialization validation failed.")
    validate_persisted_privacy({"summary": summary_bytes, "smd": smd_bytes})
    candidate_summary = _apply_summary_dtypes(
        pd.read_csv(io.BytesIO(summary_bytes), float_precision="round_trip")
    )
    candidate_smd = _apply_smd_dtypes(pd.read_csv(io.BytesIO(smd_bytes), float_precision="round_trip"))
    validate_summary(candidate_summary, config)
    validate_smd(candidate_smd, config)
    after_inputs = _verify_hash_records(root, config["inputs"])
    after_anchors = _verify_hash_records(root, config["anchors"])
    if before_inputs != after_inputs or before_anchors != after_anchors:
        raise ValueError("Frozen source immutability validation failed.")
    manifest = _build_manifest(
        root=root,
        config=config,
        git_head=git_head,
        summary_bytes=summary_bytes,
        smd_bytes=smd_bytes,
        overlap=overlap,
        retained_reconciliation=retained_reconciliation,
    )
    manifest_bytes = serialize_manifest(manifest)
    parsed_manifest = json.loads(manifest_bytes, object_pairs_hook=OrderedDict)
    if parsed_manifest != manifest:
        raise ValueError("Manifest candidate read-back validation failed.")
    payloads = OrderedDict([("summary", summary_bytes), ("smd", smd_bytes), ("manifest", manifest_bytes)])
    promote_serialized_outputs(targets, payloads)
    persisted_summary = read_and_validate_summary(targets["summary"], config)
    persisted_smd = read_and_validate_smd(targets["smd"], config)
    read_and_validate_manifest(targets["manifest"])
    if not _frames_exact(summary, persisted_summary) or not _frames_exact(smd, persisted_smd):
        raise ValueError("Promoted scientific output read-back validation failed.")
    final_inputs = _verify_hash_records(root, config["inputs"])
    final_anchors = _verify_hash_records(root, config["anchors"])
    if before_inputs != final_inputs or before_anchors != final_anchors:
        raise ValueError("Frozen sources changed during production execution.")
    return OrderedDict(
        [
            ("complete_source", int(config["population_counts"]["complete_source"])),
            ("qc_retained", int(config["population_counts"]["qc_retained"])),
            ("qc_excluded", int(config["population_counts"]["qc_excluded"])),
            ("summary_sha256", sha256_file(targets["summary"])),
            ("smd_sha256", sha256_file(targets["smd"])),
            ("manifest_sha256", sha256_file(targets["manifest"])),
        ]
    )
