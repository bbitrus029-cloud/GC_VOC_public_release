"""Build the deterministic aggregate prerequisite for Supplementary eFigure 9."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import errno
import hashlib
from importlib import metadata as importlib_metadata
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import tempfile
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from src.analysis.stage7_part_a_common import (
    _load_global_config,
    _load_test_frame_with_metadata,
    _load_view_prediction_frame,
    _merge_predictions_with_metadata,
)
from src.config import PROJECT_ROOT


ANALYSIS_ID = "efigure9_disease_category_prediction_matrix_v1"
ANALYSIS_IDENTITY = "plan_orthogonal_descriptive_aggregation"
MODEL_IDENTIFIER = "minirocket_ridge"
MODEL_DISPLAY_NAME = "MiniRocket + Ridge"
EVALUATION_SET_NAME = "internal testing set"
EXPECTED_EVALUATION_SET_SIZE = 195
EXPECTED_THRESHOLD_P_RAW = 0.3418938786147648
EXPECTED_TASK_CONFIG_SHA256 = "536886da77b5adf86da8e1f67459d03799e884ecebb4fd871c3bf4f7b47bc26d"
EXPECTED_TASK_CONFIG_PATH = "configs/efigure9_disease_category_prediction_matrix.yaml"
PLANNING_REFERENCE_HEAD = "8c44846477c250a03380eda24a5c99ab63294426"
EXPECTED_EXECUTION_COMMAND = (
    "PYTHONDONTWRITEBYTECODE=1 python scripts/run_efigure9_disease_category_prediction_matrix.py "
    "--config configs/efigure9_disease_category_prediction_matrix.yaml"
)
EXPECTED_FINAL_DIRECTORY = "outputs/analysis/efigure9_disease_category_prediction_matrix"
PREDICTION_COLUMNS = ["sample_id", "y_true", "p_raw", "p_cal", "y_pred", "quadrant"]
OUTPUT_COLUMNS = [
    "disease_category_code",
    "disease_category_label",
    "true_class",
    "predicted_class",
    "quadrant",
    "count",
    "disease_category_total",
    "within_disease_category_proportion",
]
DISEASE_CATEGORIES = (
    (0, "HC", 0),
    (1, "BGL", 0),
    (2, "LGIN", 0),
    (3, "HGIN", 1),
    (4, "EGC", 1),
    (5, "AGC", 1),
)
PREDICTED_CLASS_ORDER = (0, 1)
QUADRANT_MAPPING = {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}
TEMP_PREFIX = ".efigure9_disease_category_prediction_matrix_tmp_"
FINAL_FILENAMES = ("audit.json", "disease_category_prediction_matrix.csv", "manifest.json")
IMPLEMENTATION_PATHS = (
    EXPECTED_TASK_CONFIG_PATH,
    "src/analysis/efigure9_disease_category_prediction_matrix.py",
    "scripts/run_efigure9_disease_category_prediction_matrix.py",
    "tests/test_efigure9_disease_category_prediction_matrix.py",
)
EXPECTED_SOURCE_HASHES = {
    "configs/modeling_global.yaml": "c4520df8040adfebdfd1f6ae89376be5484ae3dfeda0cfd2481103df323a2e54",
    "outputs/predictions/binary_minirocket_ridge_test_predictions.csv": (
        "ec2f912cd83d523cdff92f453472daf3816cafe4238d98e5b6d650fefe73de28"
    ),
    "outputs/analysis/binary_minirocket_ridge_threshold_mapping.json": (
        "5b3df63297431e0824e7a11df154151beb7f3b97e8cd69fc972b67de5871af95"
    ),
    "outputs/metrics/binary_minirocket_ridge_test_metrics.csv": (
        "eb60d67d42383a0ff764633e09b271c98398ce461f20c2d094f7592a1b882835"
    ),
    "outputs/analysis/table_one_train_test.csv": (
        "8df9b565bdba4c6af7675608d489de2887bfc806b8408518241845b1d0b68117"
    ),
    "data/processed/sample_index.csv": "3e31635c37c3cc142eade488c00a240ec5487175b531f9ed94837946ddcb403f",
    "data/processed/splits/test_idx.npz": (
        "7fc60acfc54f27e77f60a9813ca829fd61f174a8d3e4ab59aac18f5e05c981c3"
    ),
    "data/raw/clinical_data/metadata.csv": (
        "58b7c08d9eea834e22d9aebe0e19049f5ca313c384262b3cba21ce9763523aa9"
    ),
    "src/config.py": "e10ccfe1ca068b39472a7b3e201cf709f0f017ab502d2cec851cb8914a3a721f",
    "src/analysis/data_split.py": "4c10e56a95d675f47c57922133bd48232893500d515ff50f3e215150261b2ada",
    "src/analysis/stage7_part_a_common.py": (
        "2ee75f19c284e84aa0528d557d9f582cf217180005dd014a5dfa06e995527179"
    ),
    "src/evaluation/metrics.py": "f355df339a9ee1bba6fff98c5bf24ea5f611554204da3257d05d702d8708595c",
}
EXPECTED_RECONCILIATION = {
    "disease_category_totals": {"HC": 32, "BGL": 71, "LGIN": 10, "HGIN": 14, "EGC": 24, "AGC": 44},
    "true_class_totals": {"0": 113, "1": 82},
    "predicted_class_totals": {"0": 105, "1": 90},
    "quadrant_totals": {"TN": 97, "FP": 16, "FN": 8, "TP": 74},
    "total": 195,
}
FORBIDDEN_DISPLAY_TERMS = (
    "low risk",
    "high risk",
    "risk group",
    "risk stratum",
    "predicted negative",
    "predicted positive",
    "model negative",
    "model positive",
)
FORBIDDEN_PARTICIPANT_FIELDS = (
    "sample_id",
    "participant_id",
    "p_raw",
    "p_cal",
    "collection_date",
    "collection_time",
    "filename",
    "raw_path",
    "age",
    "sex",
    "bmi",
    "smoking",
    "drinking",
    "tumor_marker",
)


@dataclass
class AggregationResult:
    """Hold one aggregate matrix and its aggregate validation evidence.

    Attributes:
        matrix: Complete ordered 12-cell matrix.
        calculated_disease_category_totals: Counts keyed by disease-category label.
        calculated_true_class_totals: Counts keyed by integer class serialized as text.
        calculated_predicted_class_totals: Counts keyed by integer class serialized as text.
        calculated_quadrant_totals: Counts keyed by standard quadrant token.
        checks: Aggregate-safe pass/fail validation flags.
    """

    matrix: pd.DataFrame
    calculated_disease_category_totals: dict[str, int]
    calculated_true_class_totals: dict[str, int]
    calculated_predicted_class_totals: dict[str, int]
    calculated_quadrant_totals: dict[str, int]
    checks: dict[str, bool]


def _sha256_bytes(payload: bytes) -> str:
    """Return the SHA-256 digest of bytes."""
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file without exposing its contents."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_hashes(project_root: Path, expected_hashes: dict[str, str]) -> dict[str, str]:
    """Require exact SHA-256 identity for every relevant source.

    Args:
        project_root: Root against which source paths are resolved.
        expected_hashes: Project-relative paths mapped to expected SHA-256 hashes.

    Returns:
        The verified path-to-hash mapping in input order.

    Raises:
        ValueError: If a path is invalid, missing, or has a different digest.
    """
    verified: dict[str, str] = {}
    for relative_path, expected_hash in expected_hashes.items():
        path = Path(relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Relevant source path must be project-relative.")
        source_path = project_root / path
        if not source_path.is_file():
            raise ValueError(f"Relevant source is missing: {relative_path}.")
        observed_hash = _sha256_file(source_path)
        if observed_hash != expected_hash:
            raise ValueError(
                f"Relevant source hash mismatch for {relative_path}: "
                f"expected {expected_hash}, observed {observed_hash}."
            )
        verified[relative_path] = observed_hash
    return verified


def _require_integer_series(series: pd.Series, name: str) -> None:
    """Require a complete integer-typed series."""
    if series.isna().any() or not pd.api.types.is_integer_dtype(series.dtype):
        raise ValueError(f"{name} must contain only binary integer values.")
    values = set(int(value) for value in series.tolist())
    if not values.issubset({0, 1}):
        raise ValueError(f"{name} must contain only binary integer values.")


def _require_unique_identifiers(frame: pd.DataFrame, source_name: str) -> None:
    """Require complete unique identifiers without echoing any identifier."""
    if "sample_id" not in frame.columns:
        raise ValueError(f"{source_name} identifiers must be nonmissing and unique; sample_id is absent.")
    missing_count = int(frame["sample_id"].isna().sum())
    duplicate_count = int(frame["sample_id"].duplicated(keep=False).sum())
    if missing_count or duplicate_count:
        raise ValueError(
            f"{source_name} identifiers must be nonmissing and unique; "
            f"missing_count={missing_count}, duplicate_row_count={duplicate_count}."
        )


def _quadrants(true_class: pd.Series, predicted_class: pd.Series) -> pd.Series:
    """Map binary true and predicted classes to standard quadrant tokens."""
    values = [QUADRANT_MAPPING[(int(true_value), int(predicted_value))] for true_value, predicted_value in zip(
        true_class.tolist(), predicted_class.tolist(), strict=True
    )]
    return pd.Series(values, index=true_class.index, dtype="object")


def build_prediction_matrix(
    prediction_frame: pd.DataFrame,
    metadata_frame: pd.DataFrame,
    threshold_p_raw: float,
    expected_row_count: int,
) -> AggregationResult:
    """Validate aligned frozen predictions and reduce them to the fixed matrix.

    Args:
        prediction_frame: Locked prediction rows with the exact six-column schema.
        metadata_frame: Locked test metadata containing sample_id and disease_category.
        threshold_p_raw: Full-precision locked raw-score threshold.
        expected_row_count: Required number of aligned evaluation rows.

    Returns:
        Aggregate-only matrix, totals, and passed validation flags.

    Raises:
        ValueError: If any schema, alignment, mapping, or consistency gate fails.
    """
    if prediction_frame.columns.tolist() != PREDICTION_COLUMNS:
        raise ValueError("Prediction input schema does not match the exact six-column contract.")
    if len(prediction_frame) != expected_row_count:
        raise ValueError(
            f"Prediction row count must be {expected_row_count}; observed_count={len(prediction_frame)}."
        )
    if len(metadata_frame) != expected_row_count:
        raise ValueError(f"Metadata row count must be {expected_row_count}; observed_count={len(metadata_frame)}.")
    if "disease_category" not in metadata_frame.columns:
        raise ValueError("Metadata must contain disease_category.")

    _require_unique_identifiers(prediction_frame, "Prediction")
    _require_unique_identifiers(metadata_frame, "Metadata")
    prediction_ids = set(prediction_frame["sample_id"].tolist())
    metadata_ids = set(metadata_frame["sample_id"].tolist())
    if prediction_ids != metadata_ids:
        raise ValueError(
            "Prediction and metadata identifier sets must match exactly; "
            f"prediction_count={len(prediction_ids)}, metadata_count={len(metadata_ids)}."
        )

    _require_integer_series(prediction_frame["y_true"], "Stored y_true")
    _require_integer_series(prediction_frame["y_pred"], "Stored y_pred")
    try:
        p_raw = prediction_frame["p_raw"].to_numpy(dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("p_raw must be numeric, finite, and within [0, 1].") from exc
    if not np.isfinite(p_raw).all() or np.any((p_raw < 0.0) | (p_raw > 1.0)):
        raise ValueError("p_raw must be numeric, finite, and within [0, 1].")

    disease_category = metadata_frame["disease_category"]
    if disease_category.isna().any() or not pd.api.types.is_integer_dtype(disease_category.dtype):
        raise ValueError("Disease categories must be complete integer codes 0 through 5.")
    observed_categories = set(int(value) for value in disease_category.tolist())
    if observed_categories != set(range(6)):
        raise ValueError(
            "Disease categories must map exactly to codes 0 through 5; "
            f"observed_category_count={len(observed_categories)}."
        )

    try:
        merged = _merge_predictions_with_metadata(
            prediction_frame=prediction_frame.copy(),
            metadata_frame=metadata_frame[["sample_id", "disease_category"]].copy(),
            required_columns=["disease_category"],
        )
    except Exception as exc:
        raise ValueError("Prediction-to-metadata merge must be strictly one-to-one and complete.") from exc
    if len(merged) != expected_row_count:
        raise ValueError(f"Merged row count must be {expected_row_count}; observed_count={len(merged)}.")

    predicted_class = pd.Series((p_raw >= threshold_p_raw).astype(np.int64), index=merged.index)
    stored_y_pred = merged["y_pred"].to_numpy(dtype=np.int64, copy=False)
    if not np.array_equal(predicted_class.to_numpy(dtype=np.int64), stored_y_pred):
        mismatch_count = int(np.count_nonzero(predicted_class.to_numpy(dtype=np.int64) != stored_y_pred))
        raise ValueError(f"Stored y_pred disagrees with threshold-derived prediction; mismatch_count={mismatch_count}.")

    true_class = merged["disease_category"].map({0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}).astype(np.int64)
    stored_y_true = merged["y_true"].to_numpy(dtype=np.int64, copy=False)
    if not np.array_equal(true_class.to_numpy(dtype=np.int64), stored_y_true):
        mismatch_count = int(np.count_nonzero(true_class.to_numpy(dtype=np.int64) != stored_y_true))
        raise ValueError(
            "Stored y_true disagrees with the disease-category true class; "
            f"mismatch_count={mismatch_count}."
        )

    computed_quadrant = _quadrants(true_class, predicted_class)
    stored_quadrant = merged["quadrant"].astype("string")
    if stored_quadrant.isna().any() or not computed_quadrant.equals(stored_quadrant.astype(object)):
        mismatch_count = int((computed_quadrant.astype("string") != stored_quadrant).fillna(True).sum())
        raise ValueError(f"Stored quadrant disagrees with the fixed quadrant mapping; mismatch_count={mismatch_count}.")

    aggregate_frame = pd.DataFrame(
        {
            "disease_category_code": merged["disease_category"].to_numpy(dtype=np.int64),
            "predicted_class": predicted_class.to_numpy(dtype=np.int64),
        }
    )
    grouped = aggregate_frame.groupby(
        ["disease_category_code", "predicted_class"], sort=True, observed=True
    ).size()
    rows: list[dict[str, int | float | str]] = []
    category_totals: dict[str, int] = {}
    for category_code, category_label, category_true_class in DISEASE_CATEGORIES:
        counts = [int(grouped.get((category_code, predicted_value), 0)) for predicted_value in PREDICTED_CLASS_ORDER]
        category_total = int(sum(counts))
        if category_total <= 0:
            raise ValueError("Every disease category must have a positive aggregate row total.")
        category_totals[category_label] = category_total
        for predicted_value, count in zip(PREDICTED_CLASS_ORDER, counts, strict=True):
            rows.append(
                {
                    "disease_category_code": category_code,
                    "disease_category_label": category_label,
                    "true_class": category_true_class,
                    "predicted_class": predicted_value,
                    "quadrant": QUADRANT_MAPPING[(category_true_class, predicted_value)],
                    "count": count,
                    "disease_category_total": category_total,
                    "within_disease_category_proportion": count / category_total,
                }
            )
    matrix = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    validate_matrix(matrix)

    true_class_totals = {
        str(class_value): int(matrix.loc[matrix["true_class"] == class_value, "count"].sum())
        for class_value in PREDICTED_CLASS_ORDER
    }
    predicted_class_totals = {
        str(class_value): int(matrix.loc[matrix["predicted_class"] == class_value, "count"].sum())
        for class_value in PREDICTED_CLASS_ORDER
    }
    quadrant_totals = {
        token: int(matrix.loc[matrix["quadrant"] == token, "count"].sum())
        for token in ("TN", "FP", "FN", "TP")
    }
    checks = {
        "prediction_schema_passed": True,
        "prediction_row_count_passed": True,
        "prediction_identifiers_complete_unique_passed": True,
        "metadata_row_count_passed": True,
        "metadata_identifiers_complete_unique_passed": True,
        "identifier_set_alignment_passed": True,
        "one_to_one_merge_passed": True,
        "merged_row_count_passed": True,
        "p_raw_range_and_finiteness_passed": True,
        "binary_integer_schema_passed": True,
        "disease_category_completeness_passed": True,
        "threshold_derived_prediction_agreement_passed": True,
        "endpoint_mapping_agreement_passed": True,
        "quadrant_mapping_agreement_passed": True,
        "cartesian_completeness_passed": True,
        "output_schema_passed": True,
        "output_order_passed": True,
        "normalization_passed": True,
        "privacy_passed": True,
        "terminology_passed": True,
    }
    return AggregationResult(
        matrix=matrix,
        calculated_disease_category_totals=category_totals,
        calculated_true_class_totals=true_class_totals,
        calculated_predicted_class_totals=predicted_class_totals,
        calculated_quadrant_totals=quadrant_totals,
        checks=checks,
    )


def validate_matrix(matrix: pd.DataFrame) -> None:
    """Validate the complete aggregate-only CSV contract.

    Args:
        matrix: Candidate 12-cell aggregate matrix.

    Raises:
        ValueError: If schema, ordering, mapping, counts, or normalization differ.
    """
    if matrix.columns.tolist() != OUTPUT_COLUMNS:
        raise ValueError("Output matrix schema does not match the exact contract.")
    if len(matrix) != 12:
        raise ValueError(f"Output matrix must contain exactly 12 rows; observed_count={len(matrix)}.")
    for column in ("disease_category_code", "true_class", "predicted_class", "count", "disease_category_total"):
        if matrix[column].isna().any() or not pd.api.types.is_integer_dtype(matrix[column].dtype):
            raise ValueError(f"Output column {column} must use integer serialization.")
    expected_pairs = [(code, predicted) for code, _, _ in DISEASE_CATEGORIES for predicted in PREDICTED_CLASS_ORDER]
    observed_pairs = list(zip(matrix["disease_category_code"], matrix["predicted_class"], strict=True))
    if observed_pairs != expected_pairs:
        raise ValueError("Output matrix ordering or Cartesian completeness is invalid.")
    category_by_code = {code: (label, true_class) for code, label, true_class in DISEASE_CATEGORIES}
    for row in matrix.itertuples(index=False):
        expected_label, expected_true_class = category_by_code[int(row.disease_category_code)]
        if row.disease_category_label != expected_label or int(row.true_class) != expected_true_class:
            raise ValueError("Output disease-category label or true-class mapping is invalid.")
        if row.quadrant != QUADRANT_MAPPING[(expected_true_class, int(row.predicted_class))]:
            raise ValueError("Output quadrant mapping is invalid.")
        if int(row.count) < 0 or int(row.disease_category_total) <= 0:
            raise ValueError("Output counts and row totals must be nonnegative with positive row totals.")
        proportion = float(row.within_disease_category_proportion)
        if not np.isfinite(proportion) or not 0.0 <= proportion <= 1.0:
            raise ValueError("Output row proportions must be finite and within [0, 1].")
        if not np.isclose(proportion, int(row.count) / int(row.disease_category_total), rtol=0.0, atol=1e-15):
            raise ValueError("Output row proportion does not equal count divided by disease-category total.")
    for _, group in matrix.groupby("disease_category_code", sort=True):
        row_total = int(group["disease_category_total"].iloc[0])
        if group["disease_category_total"].nunique() != 1 or int(group["count"].sum()) != row_total:
            raise ValueError("Disease-category counts do not reconcile to the declared row total.")
        if not np.isclose(
            float(group["within_disease_category_proportion"].sum()), 1.0, rtol=0.0, atol=1e-15
        ):
            raise ValueError("Within-disease-category proportions do not sum to one.")


def serialize_matrix_csv(matrix: pd.DataFrame) -> bytes:
    """Serialize a validated matrix deterministically as UTF-8 LF CSV.

    Args:
        matrix: Candidate aggregate matrix.

    Returns:
        Deterministic CSV bytes.

    Raises:
        ValueError: If the matrix or serialized terminology/privacy contract fails.
    """
    validate_matrix(matrix)
    buffer = io.StringIO(newline="")
    matrix.to_csv(buffer, index=False, lineterminator="\n", float_format="%.17g")
    payload = buffer.getvalue().encode("utf-8")
    lowered = payload.decode("utf-8").lower()
    if any(term in lowered for term in FORBIDDEN_DISPLAY_TERMS):
        raise ValueError("Serialized CSV contains forbidden prediction display terminology.")
    header_fields = set(lowered.splitlines()[0].split(","))
    if header_fields.intersection(FORBIDDEN_PARTICIPANT_FIELDS):
        raise ValueError("Serialized CSV contains a participant-level field.")
    if b"\r" in payload:
        raise ValueError("Serialized CSV must use LF newlines only.")
    return payload


def serialize_json(payload: dict[str, Any]) -> bytes:
    """Serialize a JSON mapping with stable sorted keys and no timestamp.

    Args:
        payload: Aggregate-safe JSON mapping.

    Returns:
        Deterministic UTF-8 JSON bytes.

    Raises:
        ValueError: If a timestamp key, forbidden term, or absolute path is present.
    """
    serialized = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")
    lowered = serialized.decode("utf-8").lower()
    if re.search(r'"[^"\n]*timestamp[^"\n]*"\s*:', lowered):
        raise ValueError("Deterministic JSON must not contain timestamp fields.")
    if any(term in lowered for term in FORBIDDEN_DISPLAY_TERMS):
        raise ValueError("Deterministic JSON contains forbidden prediction display terminology.")
    if re.search(r'(?<![a-z0-9_.-])/(?:home|users|root|tmp|var|opt|etc)/', lowered):
        raise ValueError("Deterministic JSON must not contain absolute paths.")
    return serialized


def build_audit_payload(
    result: AggregationResult,
    analysis_id: str,
    threshold_p_raw: float,
    expected_evaluation_set_size: int,
) -> dict[str, Any]:
    """Build the deterministic aggregate-only audit mapping.

    Args:
        result: Validated aggregation result.
        analysis_id: Fixed analysis identifier.
        threshold_p_raw: Locked raw-score threshold.
        expected_evaluation_set_size: Required evaluation-set size.

    Returns:
        Aggregate-only deterministic audit payload.
    """
    validate_matrix(result.matrix)
    return {
        "analysis_id": analysis_id,
        "analysis_identity": ANALYSIS_IDENTITY,
        "artifact_statement": "Aggregate prerequisite; no figure display-label decision is included.",
        "calculated_disease_category_totals": result.calculated_disease_category_totals,
        "calculated_predicted_class_totals": result.calculated_predicted_class_totals,
        "calculated_quadrant_totals": result.calculated_quadrant_totals,
        "calculated_true_class_totals": result.calculated_true_class_totals,
        "checks": result.checks,
        "determinism_statement": "The operation was deterministic and used no stochastic procedure.",
        "disease_category_order": [
            {"code": code, "label": label} for code, label, _ in DISEASE_CATEGORIES
        ],
        "endpoint_mapping": {
            "0": ["HC", "BGL", "LGIN"],
            "1": ["HGIN", "EGC", "AGC"],
        },
        "evaluation_set": EVALUATION_SET_NAME,
        "expected_evaluation_set_size": expected_evaluation_set_size,
        "model": MODEL_DISPLAY_NAME,
        "participant_persistence_statement": "No participant-level records were persisted.",
        "predicted_class_order": list(PREDICTED_CLASS_ORDER),
        "quadrant_mapping": {f"{true_value},{predicted_value}": token for (
            true_value, predicted_value
        ), token in QUADRANT_MAPPING.items()},
        "score_space": "p_raw",
        "threshold_p_raw": threshold_p_raw,
        "threshold_rule": "p_raw >= threshold_p_raw",
    }


def _parse_table_one_totals(path: Path, expected_size: int) -> dict[str, int]:
    """Read exactly the six unambiguous disease-category aggregate rows."""
    frame = pd.read_csv(path)
    if frame.columns.tolist() != ["variable", "train_summary", "test_summary", "smd"]:
        raise ValueError("Table-one reconciliation schema is unexpected.")
    totals: dict[str, int] = {}
    for code, label, _ in DISEASE_CATEGORIES:
        variable = f"disease_category={code} ({label})"
        rows = frame.loc[frame["variable"] == variable]
        if len(rows) != 1:
            raise ValueError(f"Table-one reconciliation requires exactly one row for disease category {code}.")
        match = re.fullmatch(rf"([0-9]+)/{expected_size} \([0-9]+(?:\.[0-9]+)?%\)", str(rows.iloc[0]["test_summary"]))
        if match is None:
            raise ValueError(f"Table-one test summary is ambiguous for disease category {code}.")
        totals[label] = int(match.group(1))
    return totals


def _parse_metric_quadrants(path: Path) -> dict[str, int]:
    """Read exactly one test-set row for each confusion-count metric."""
    frame = pd.read_csv(path)
    if frame.columns.tolist() != ["scope", "metric", "value", "ci_low", "ci_high"]:
        raise ValueError("Metrics reconciliation schema is unexpected.")
    totals: dict[str, int] = {}
    for token in ("TN", "FP", "FN", "TP"):
        rows = frame.loc[(frame["scope"] == "test") & (frame["metric"] == token.lower())]
        if len(rows) != 1:
            raise ValueError(f"Metrics reconciliation requires exactly one test row for {token}.")
        value = float(rows.iloc[0]["value"])
        if not np.isfinite(value) or not value.is_integer():
            raise ValueError(f"Metrics reconciliation value for {token} must be one integer count.")
        totals[token] = int(value)
    return totals


def reconcile_aggregate_totals(
    result: AggregationResult,
    table_one_path: Path,
    metrics_path: Path,
    expected: dict[str, Any],
    expected_size: int,
) -> None:
    """Require calculated totals, independent aggregates, and frozen anchors to agree.

    Args:
        result: Calculated aggregate result.
        table_one_path: Frozen table-one CSV.
        metrics_path: Frozen metrics CSV.
        expected: Frozen aggregate anchors from the task contract.
        expected_size: Required total number of participants.

    Raises:
        ValueError: If any independently sourced or calculated total differs.
    """
    table_totals = _parse_table_one_totals(table_one_path, expected_size)
    metric_totals = _parse_metric_quadrants(metrics_path)
    calculated_total = int(result.matrix["count"].sum())
    checks = {
        "reconciliation_disease_category_passed": (
            result.calculated_disease_category_totals == table_totals == expected["disease_category_totals"]
        ),
        "reconciliation_true_class_passed": result.calculated_true_class_totals == expected["true_class_totals"],
        "reconciliation_predicted_class_passed": (
            result.calculated_predicted_class_totals == expected["predicted_class_totals"]
        ),
        "reconciliation_quadrant_passed": (
            result.calculated_quadrant_totals == metric_totals == expected["quadrant_totals"]
        ),
        "reconciliation_total_passed": calculated_total == expected_size == int(expected["total"]),
    }
    failed_count = sum(not passed for passed in checks.values())
    if failed_count:
        raise ValueError(f"Aggregate reconciliation failed; failed_check_count={failed_count}.")
    result.checks.update(checks)


def _readback_candidate(directory: Path) -> None:
    """Read and validate a two-file candidate without exposing participant-capable data."""
    inventory = sorted(path.name for path in directory.iterdir())
    if inventory != ["audit.json", "disease_category_prediction_matrix.csv"]:
        raise ValueError("Candidate inventory before manifest is invalid.")
    matrix = pd.read_csv(directory / "disease_category_prediction_matrix.csv")
    for column in ("disease_category_code", "true_class", "predicted_class", "count", "disease_category_total"):
        matrix[column] = pd.to_numeric(matrix[column], errors="raise").astype(np.int64)
    validate_matrix(matrix)
    audit = json.loads((directory / "audit.json").read_text(encoding="utf-8"))
    if not isinstance(audit, dict) or not isinstance(audit.get("checks"), dict):
        raise ValueError("Audit JSON schema is invalid.")
    if not audit["checks"] or any(value is not True for value in audit["checks"].values()):
        raise ValueError("Audit JSON contains an absent or failed validation flag.")
    serialize_json(audit)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename a directory while refusing any existing destination."""
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise OSError("Atomic no-replace rename is unavailable on this platform.")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(f"Final target already exists: {destination.name}.")
        raise OSError(error_number, "Atomic no-replace rename failed.")


def _remove_if_identity_matches(directory: Path, expected_hashes: dict[str, str]) -> bool:
    """Remove an invocation-owned promoted directory only when its identity is unchanged."""
    if not directory.is_dir():
        return False
    inventory = sorted(path.name for path in directory.iterdir())
    if inventory != sorted(expected_hashes):
        return False
    observed = {name: _sha256_file(directory / name) for name in inventory}
    if observed != expected_hashes:
        return False
    shutil.rmtree(directory)
    return True


def publish_validated_package(
    *,
    first_matrix: pd.DataFrame,
    first_audit: dict[str, Any],
    second_matrix: pd.DataFrame,
    second_audit: dict[str, Any],
    manifest_builder: Callable[[], dict[str, Any]],
    output_root: Path,
    final_dir: Path,
) -> dict[str, str]:
    """Validate two deterministic candidates and atomically publish one.

    Args:
        first_matrix: Matrix from the first independent source load.
        first_audit: Audit from the first independent source load.
        second_matrix: Matrix from the second independent source load.
        second_audit: Audit from the second independent source load.
        manifest_builder: Final revalidation callback returning the manifest.
        output_root: Existing parent for task-owned temporary directories.
        final_dir: Absent final output directory.

    Returns:
        SHA-256 hashes for the three promoted output files.

    Raises:
        FileExistsError: If the final target exists at either no-clobber gate.
        ValueError: If candidates differ or violate any output contract.
        OSError: If atomic no-replace promotion is unavailable or fails.
    """
    if final_dir.exists():
        raise FileExistsError(f"Final target already exists: {final_dir.name}.")
    if not output_root.is_dir() or final_dir.parent != output_root:
        raise ValueError("Output root and final target do not satisfy the destination contract.")
    temporary_directories: list[Path] = []
    promoted = False
    promoted_hashes: dict[str, str] = {}
    try:
        for matrix, audit in ((first_matrix, first_audit), (second_matrix, second_audit)):
            temporary = Path(tempfile.mkdtemp(prefix=TEMP_PREFIX, dir=output_root))
            temporary_directories.append(temporary)
            (temporary / "disease_category_prediction_matrix.csv").write_bytes(serialize_matrix_csv(matrix))
            (temporary / "audit.json").write_bytes(serialize_json(audit))
            _readback_candidate(temporary)

        first_csv = (temporary_directories[0] / "disease_category_prediction_matrix.csv").read_bytes()
        second_csv = (temporary_directories[1] / "disease_category_prediction_matrix.csv").read_bytes()
        first_audit_bytes = (temporary_directories[0] / "audit.json").read_bytes()
        second_audit_bytes = (temporary_directories[1] / "audit.json").read_bytes()
        if first_csv != second_csv or _sha256_bytes(first_csv) != _sha256_bytes(second_csv):
            raise ValueError("Independent CSV builds are not byte-identical.")
        if first_audit_bytes != second_audit_bytes or _sha256_bytes(first_audit_bytes) != _sha256_bytes(
            second_audit_bytes
        ):
            raise ValueError("Independent audit builds are not byte-identical.")

        manifest = manifest_builder()
        manifest_bytes = serialize_json(manifest)
        (temporary_directories[0] / "manifest.json").write_bytes(manifest_bytes)
        if sorted(path.name for path in temporary_directories[0].iterdir()) != sorted(FINAL_FILENAMES):
            raise ValueError("Temporary final package inventory is invalid.")
        promoted_hashes = {
            "audit.json": _sha256_bytes(first_audit_bytes),
            "disease_category_prediction_matrix.csv": _sha256_bytes(first_csv),
            "manifest.json": _sha256_bytes(manifest_bytes),
        }
        _rename_noreplace(temporary_directories[0], final_dir)
        promoted = True
        temporary_directories.pop(0)
        shutil.rmtree(temporary_directories.pop(0))

        if sorted(path.name for path in final_dir.iterdir()) != sorted(FINAL_FILENAMES):
            raise ValueError("Promoted output inventory is invalid.")
        observed_hashes = {name: _sha256_file(final_dir / name) for name in FINAL_FILENAMES}
        if observed_hashes != promoted_hashes:
            raise ValueError("Promoted output hashes differ from the validated candidate.")
        _readback_candidate_without_inventory_restriction(final_dir)
        return {
            "audit_sha256": promoted_hashes["audit.json"],
            "csv_sha256": promoted_hashes["disease_category_prediction_matrix.csv"],
            "manifest_sha256": promoted_hashes["manifest.json"],
        }
    except Exception:
        for temporary in temporary_directories:
            if temporary.is_dir() and temporary.parent == output_root and temporary.name.startswith(TEMP_PREFIX):
                shutil.rmtree(temporary)
        if promoted:
            _remove_if_identity_matches(final_dir, promoted_hashes)
        raise


def _readback_candidate_without_inventory_restriction(directory: Path) -> None:
    """Validate a promoted three-file package without rewriting it."""
    matrix = pd.read_csv(directory / "disease_category_prediction_matrix.csv")
    for column in ("disease_category_code", "true_class", "predicted_class", "count", "disease_category_total"):
        matrix[column] = pd.to_numeric(matrix[column], errors="raise").astype(np.int64)
    validate_matrix(matrix)
    audit = json.loads((directory / "audit.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if any(value is not True for value in audit["checks"].values()):
        raise ValueError("Promoted audit contains a failed validation flag.")
    if sorted(manifest["final_output_inventory"]) != sorted(FINAL_FILENAMES):
        raise ValueError("Promoted manifest inventory is invalid.")
    serialize_json(audit)
    serialize_json(manifest)


def _expected_contract_payload() -> dict[str, Any]:
    """Return the complete fixed semantic contract."""
    return {
        "analysis_id": ANALYSIS_ID,
        "analysis_identity": ANALYSIS_IDENTITY,
        "planning_reference_head": PLANNING_REFERENCE_HEAD,
        "model": {"identifier": MODEL_IDENTIFIER, "display_name": MODEL_DISPLAY_NAME},
        "evaluation_set": {"name": EVALUATION_SET_NAME, "expected_size": EXPECTED_EVALUATION_SET_SIZE},
        "score": {
            "space": "p_raw",
            "threshold_p_raw": EXPECTED_THRESHOLD_P_RAW,
            "threshold_rule": "p_raw >= threshold_p_raw",
        },
        "disease_categories": [
            {"code": code, "label": label, "true_class": true_class}
            for code, label, true_class in DISEASE_CATEGORIES
        ],
        "predicted_class_order": list(PREDICTED_CLASS_ORDER),
        "quadrant_mapping": {
            f"{true_class},{predicted_class}": quadrant
            for (true_class, predicted_class), quadrant in QUADRANT_MAPPING.items()
        },
        "prediction_schema": PREDICTION_COLUMNS,
        "output_schema": OUTPUT_COLUMNS,
        "serialization": {"csv_float_format": "%.17g", "json_sort_keys": True, "line_ending": "LF"},
        "sources": EXPECTED_SOURCE_HASHES,
        "reconciliation": EXPECTED_RECONCILIATION,
        "outputs": {"final_directory": EXPECTED_FINAL_DIRECTORY, "inventory": list(FINAL_FILENAMES)},
        "authorized_task_paths": list(IMPLEMENTATION_PATHS),
        "execution_command": EXPECTED_EXECUTION_COMMAND,
        "no_stochastic_procedure": True,
        "participant_level_persistence": False,
        "figure_creation_authorized": False,
    }


def _require_exact_contract_value(observed: Any, expected: Any) -> None:
    """Recursively require exact types, keys, lengths, ordering, and scalar values."""
    if type(observed) is not type(expected):
        raise ValueError("Task configuration semantic contract is invalid.")
    if isinstance(expected, dict):
        if set(observed) != set(expected):
            raise ValueError("Task configuration semantic contract is invalid.")
        for key, expected_value in expected.items():
            _require_exact_contract_value(observed[key], expected_value)
        return
    if isinstance(expected, list):
        if len(observed) != len(expected):
            raise ValueError("Task configuration semantic contract is invalid.")
        for observed_value, expected_value in zip(observed, expected, strict=True):
            _require_exact_contract_value(observed_value, expected_value)
        return
    if observed != expected:
        raise ValueError("Task configuration semantic contract is invalid.")


def _validate_contract_payload(payload: Any) -> None:
    """Validate the complete parsed contract independently of its byte identity."""
    _require_exact_contract_value(payload, _expected_contract_payload())


def _resolve_project_root(project_root: Path) -> Path:
    """Resolve and require one existing project-root directory."""
    try:
        resolved_root = project_root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("Task configuration project root is invalid.") from exc
    if not resolved_root.is_dir():
        raise ValueError("Task configuration project root is invalid.")
    return resolved_root


def _is_within_project(path: Path, project_root: Path) -> bool:
    """Return whether a resolved path remains at or below the project root."""
    return path == project_root or project_root in path.parents


def _validate_project_relative_path(
    relative_path: Any,
    project_root: Path,
    *,
    require_file: bool,
) -> Path:
    """Validate one canonical POSIX project-relative path and its containment."""
    if type(relative_path) is not str or not relative_path:
        raise ValueError("Task configuration contains an invalid project-relative path.")
    pure_path = PurePosixPath(relative_path)
    if (
        pure_path.is_absolute()
        or relative_path != pure_path.as_posix()
        or "\\" in relative_path
        or any(part in {".", ".."} for part in pure_path.parts)
    ):
        raise ValueError("Task configuration contains an invalid project-relative path.")
    candidate = project_root.joinpath(*pure_path.parts)
    try:
        if require_file:
            resolved = candidate.resolve(strict=True)
            if not resolved.is_file():
                raise ValueError("Task configuration contains an invalid project-relative file path.")
        else:
            if candidate.is_symlink():
                raise ValueError("Task configuration contains an invalid output path.")
            if candidate.exists():
                resolved = candidate.resolve(strict=True)
                if not resolved.is_dir():
                    raise ValueError("Task configuration contains an invalid output path.")
            else:
                resolved_parent = candidate.parent.resolve(strict=True)
                resolved = resolved_parent / candidate.name
    except (OSError, RuntimeError) as exc:
        raise ValueError("Task configuration path validation failed.") from exc
    if not _is_within_project(resolved, project_root):
        raise ValueError("Task configuration path escapes the project root.")
    return resolved


def _resolve_fixed_config_path(config_path: Path, project_root: Path) -> Path:
    """Resolve only the fixed task configuration location without path disclosure."""
    if not isinstance(config_path, Path):
        raise ValueError("Task configuration location is invalid.")
    if any(part in {".", ".."} for part in config_path.parts):
        raise ValueError("Task configuration location is invalid.")
    expected_path = project_root / EXPECTED_TASK_CONFIG_PATH
    candidate = config_path if config_path.is_absolute() else project_root / config_path
    if candidate != expected_path:
        raise ValueError("Task configuration location is invalid.")
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("Task configuration location is invalid.") from exc
    if not resolved.is_file() or not _is_within_project(resolved, project_root):
        raise ValueError("Task configuration location is invalid.")
    return resolved


def _validate_contract_paths(payload: dict[str, Any], project_root: Path) -> None:
    """Validate every configured path before the contract can be returned."""
    for relative_path in payload["sources"]:
        _validate_project_relative_path(relative_path, project_root, require_file=True)
    for relative_path in payload["authorized_task_paths"]:
        _validate_project_relative_path(relative_path, project_root, require_file=True)
    _validate_project_relative_path(payload["outputs"]["final_directory"], project_root, require_file=False)


def _load_contract(config_path: Path, project_root: Path | None = None) -> dict[str, Any]:
    """Load the fixed task contract through identity, semantic, and path gates."""
    resolved_root = _resolve_project_root(PROJECT_ROOT if project_root is None else project_root)
    resolved_config = _resolve_fixed_config_path(config_path, resolved_root)
    try:
        config_bytes = resolved_config.read_bytes()
    except OSError as exc:
        raise ValueError("Task configuration could not be read.") from exc
    if _sha256_bytes(config_bytes) != EXPECTED_TASK_CONFIG_SHA256:
        raise ValueError("Task configuration byte identity is invalid.")
    try:
        payload = yaml.safe_load(config_bytes)
    except yaml.YAMLError as exc:
        raise ValueError("Task configuration parsing failed.") from exc
    _validate_contract_payload(payload)
    _validate_contract_paths(payload, resolved_root)
    return payload


def _git_head(project_root: Path) -> str:
    """Return the current Git HEAD through a read-only command."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _relevant_status_is_clean(project_root: Path, source_paths: list[str]) -> bool:
    """Return whether relevant source paths have no tracked worktree overlap."""
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--", *source_paths],
        cwd=project_root,
        check=True,
        capture_output=True,
    )
    return not bool(result.stdout)


def _unrelated_worktree_changes_present(project_root: Path) -> bool:
    """Detect unrelated worktree entries without persisting or returning their paths."""
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=project_root,
        check=True,
        capture_output=True,
    )
    for raw_record in result.stdout.split(b"\0"):
        if not raw_record:
            continue
        decoded = raw_record.decode("utf-8", errors="surrogateescape")
        path = decoded[3:] if len(decoded) >= 4 else decoded
        owned = path in IMPLEMENTATION_PATHS or path.startswith(
            "outputs/analysis/efigure9_disease_category_prediction_matrix/"
        ) or path.startswith(f"outputs/analysis/{TEMP_PREFIX}")
        if not owned:
            return True
    return False


def _load_threshold(path: Path, expected_threshold: float) -> float:
    """Load and require the exact full-precision raw-score threshold."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Threshold mapping JSON could not be loaded.") from exc
    if not isinstance(payload, dict) or set(payload).issuperset({"threshold_p_raw"}) is False:
        raise ValueError("Threshold mapping JSON schema is invalid.")
    threshold = payload["threshold_p_raw"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("threshold_p_raw must be one numeric value.")
    if float(threshold) != expected_threshold:
        raise ValueError(
            f"Locked threshold differs: expected {expected_threshold!r}, observed {float(threshold)!r}."
        )
    return float(threshold)


def _load_one_result(contract: dict[str, Any], project_root: Path) -> tuple[AggregationResult, dict[str, Any]]:
    """Independently load authorized sources and build one aggregate candidate."""
    global_config = _load_global_config(project_root / "configs" / "modeling_global.yaml")
    prediction_frame = _load_view_prediction_frame(MODEL_IDENTIFIER, "test")
    metadata_frame = _load_test_frame_with_metadata(global_config)
    threshold = _load_threshold(
        project_root / "outputs" / "analysis" / "binary_minirocket_ridge_threshold_mapping.json",
        EXPECTED_THRESHOLD_P_RAW,
    )
    result = build_prediction_matrix(
        prediction_frame=prediction_frame,
        metadata_frame=metadata_frame,
        threshold_p_raw=threshold,
        expected_row_count=EXPECTED_EVALUATION_SET_SIZE,
    )
    reconcile_aggregate_totals(
        result=result,
        table_one_path=project_root / "outputs" / "analysis" / "table_one_train_test.csv",
        metrics_path=project_root / "outputs" / "metrics" / "binary_minirocket_ridge_test_metrics.csv",
        expected=contract["reconciliation"],
        expected_size=EXPECTED_EVALUATION_SET_SIZE,
    )
    result.checks["threshold_exact_passed"] = threshold == EXPECTED_THRESHOLD_P_RAW
    result.checks["privacy_passed"] = True
    result.checks["terminology_passed"] = True
    audit = build_audit_payload(
        result=result,
        analysis_id=ANALYSIS_ID,
        threshold_p_raw=threshold,
        expected_evaluation_set_size=EXPECTED_EVALUATION_SET_SIZE,
    )
    return result, audit


def _package_versions() -> dict[str, str]:
    """Return versions of only directly used third-party packages."""
    return {
        "numpy": importlib_metadata.version("numpy"),
        "pandas": importlib_metadata.version("pandas"),
        "PyYAML": importlib_metadata.version("PyYAML"),
    }


def run_aggregation(config_path: Path, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Execute the fixed two-build aggregation and atomic promotion.

    Args:
        config_path: Task-specific configuration path.
        project_root: Active project root, overridable only for controlled tests.

    Returns:
        Aggregate-only execution summary and final file hashes.

    Raises:
        ValueError: If any scientific, privacy, provenance, or reconciliation gate fails.
        FileExistsError: If any permanent target would be overwritten.
        OSError: If deterministic output creation or atomic promotion fails.
    """
    project_root = _resolve_project_root(project_root)
    contract = _load_contract(config_path, project_root)
    final_dir = project_root / contract["outputs"]["final_directory"]
    if final_dir.exists():
        raise FileExistsError("Final eFigure 9 output directory already exists.")
    head_start = _git_head(project_root)
    agents_bytes_start = (project_root / "AGENTS.md").read_bytes()
    agents_hash_start = _sha256_bytes(agents_bytes_start)
    source_hashes = verify_source_hashes(project_root, contract["sources"])
    source_paths = list(contract["sources"])
    if not _relevant_status_is_clean(project_root, source_paths):
        raise ValueError("A relevant source path has an overlapping worktree change.")
    implementation_hashes = {
        path: _sha256_file(project_root / path) for path in contract["authorized_task_paths"]
    }
    unrelated_changes_start = _unrelated_worktree_changes_present(project_root)

    first_result, first_audit = _load_one_result(contract, project_root)
    second_result, second_audit = _load_one_result(contract, project_root)
    csv_sha256 = _sha256_bytes(serialize_matrix_csv(first_result.matrix))
    audit_sha256 = _sha256_bytes(serialize_json(first_audit))
    runtime_state: dict[str, Any] = {}

    def build_manifest_after_final_revalidation() -> dict[str, Any]:
        """Revalidate every mutable authority boundary, then build the manifest."""
        if final_dir.exists():
            raise FileExistsError("Final eFigure 9 output directory appeared before promotion.")
        if (project_root / "AGENTS.md").read_bytes() != agents_bytes_start:
            raise ValueError("AGENTS.md changed during execution and requires renewed review.")
        final_source_hashes = verify_source_hashes(project_root, contract["sources"])
        if final_source_hashes != source_hashes:
            raise ValueError("A relevant source changed during execution.")
        if not _relevant_status_is_clean(project_root, source_paths):
            raise ValueError("A relevant source path acquired an overlapping worktree change.")
        final_implementation_hashes = {
            path: _sha256_file(project_root / path) for path in contract["authorized_task_paths"]
        }
        if final_implementation_hashes != implementation_hashes:
            raise ValueError("An authorized implementation target changed during execution.")
        head_end = _git_head(project_root)
        unrelated_changes_end = _unrelated_worktree_changes_present(project_root)
        runtime_state["head_end"] = head_end
        runtime_state["unrelated_changes_present"] = unrelated_changes_start or unrelated_changes_end
        return {
            "analysis_id": ANALYSIS_ID,
            "execution_command": contract["execution_command"],
            "final_output_hashes": {
                f"{contract['outputs']['final_directory']}/audit.json": audit_sha256,
                f"{contract['outputs']['final_directory']}/disease_category_prediction_matrix.csv": csv_sha256,
            },
            "final_output_inventory": contract["outputs"]["inventory"],
            "git_mutation_performed": False,
            "governing_instructions": {"path": "AGENTS.md", "sha256": agents_hash_start},
            "implementation_hashes": final_implementation_hashes,
            "no_git_mutation_confirmed": True,
            "package_versions": _package_versions(),
            "planning_reference_head": contract["planning_reference_head"],
            "python_version": platform.python_version(),
            "relevant_paths_clean_and_non_overlapping_at_preflight": True,
            "runtime_head_end": head_end,
            "runtime_head_end_matches_planning_reference": head_end == contract["planning_reference_head"],
            "runtime_head_start": head_start,
            "runtime_head_start_matches_planning_reference": head_start == contract["planning_reference_head"],
            "source_hashes": final_source_hashes,
            "unrelated_worktree_changes_present": runtime_state["unrelated_changes_present"],
        }

    output_hashes = publish_validated_package(
        first_matrix=first_result.matrix,
        first_audit=first_audit,
        second_matrix=second_result.matrix,
        second_audit=second_audit,
        manifest_builder=build_manifest_after_final_revalidation,
        output_root=final_dir.parent,
        final_dir=final_dir,
    )
    return {
        "calculated_disease_category_totals": first_result.calculated_disease_category_totals,
        "calculated_predicted_class_totals": first_result.calculated_predicted_class_totals,
        "calculated_quadrant_totals": first_result.calculated_quadrant_totals,
        "calculated_true_class_totals": first_result.calculated_true_class_totals,
        "head_end": runtime_state["head_end"],
        "head_start": head_start,
        "matrix": first_result.matrix.copy(),
        **output_hashes,
    }
