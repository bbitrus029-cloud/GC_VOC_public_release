"""Reporting-closure calculations and future Session 6 orchestration.

All functions are import-safe. Participant-capable loaders are called only by the
explicitly authorized runner after immutable authority and stale-target checks.
"""

from __future__ import annotations

import json
import hashlib
import re
import subprocess
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

from src.analysis.posthoc_final_common import (
    ACCESS_STATE_KEYS,
    build_registry_frame,
    is_utc_iso8601_z,
    sha256_file,
    validate_exact_binary_labels_01,
    validate_freeze_commit,
    validate_preflight_privacy,
    validate_preflight_schema,
    wilson_proportion_ci,
)
from src.analysis.posthoc_final_locked_scores import (
    CURRENT_MANIFEST_RELATIVE_PATH,
    SCORE_BLIND_PREFLIGHT_RELATIVE_PATH,
    _validate_annotated_tag_chain,
    load_effective_posthoc_config,
    validate_amendment_freeze_commit,
    validate_current_runtime_authority,
    validate_execute_git_prerequisites,
    _validate_cross_output_reconciliation,
    _validate_scientific_records,
)
from src.analysis.posthoc_final_date_grouped import validate_analysis3_outputs
from src.analysis.posthoc_final_audit import (
    PrivacyReferences,
    PromotionCandidate,
    build_reporting_access_state,
    build_session_completion_log,
    capture_integrity_snapshot,
    load_privacy_references,
    promote_candidates,
    read_typed_registry_csv,
    stage_candidate_bytes,
    validate_access_state,
    validate_artifact_privacy,
    validate_completed_session_from_disk,
    validate_completion_log_against_bytes,
    validate_csv_frame,
    validate_execution_manifest,
    verify_integrity_snapshot,
)
from src.config import PROJECT_ROOT


DISEASE_LABELS = ("HC", "BGL", "LGIN", "HGIN", "EGC", "AGC")
LOWESS_ESTIMATOR = "statsmodels.nonparametric.smoothers_lowess.lowess"
LOWESS_ROLE = "graphical descriptive estimator, not a probability calibrator"
SESSION0_FREEZE = "3c0ff6b3b40460a2c340ad3193222ee54979109d"
AMENDMENT_FREEZE = "2e08e8fb89d99bb62cca2e5204a13ff09e47d432"
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class SmdResult(NamedTuple):
    """One signed SMD and its frozen estimability state."""

    estimate: float | None
    status: str
    reason: str | None


class ReportingAlignmentEvidence(NamedTuple):
    """In-memory binding of source membership, locked test IDs, labels, and p_cal."""

    final: pd.DataFrame
    excluded: pd.DataFrame
    train_ids: np.ndarray
    test_ids: np.ndarray
    calibration: pd.DataFrame
    expected_test_n: int


class PriorAggregateSemanticEvidence(NamedTuple):
    """Independent aggregate anchors recovered from validated Session 4/5 outputs."""

    internal_test_n: int
    internal_test_n_positive: int
    internal_test_n_negative: int
    date_grouped_train_n: int
    date_grouped_train_n_positive: int
    date_grouped_train_n_negative: int


class ReportingPromotionBundle(NamedTuple):
    """Fully staged and read-back-validated Session 6 promotion state."""

    candidates: tuple[PromotionCandidate, ...]
    log_candidate: Path
    log_target: Path
    log_validator: Callable[[Path], None]


def _as_unique_ids(values: Sequence[object], role: str) -> np.ndarray:
    """Validate one nonempty, one-dimensional, duplicate-free integer ID set."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{role} IDs must be one-dimensional.")
    if array.size == 0:
        raise ValueError(f"{role} IDs must be nonempty.")
    if array.dtype.kind not in "iu":
        raise ValueError(f"{role} IDs must use an integer dtype.")
    result = array.astype(np.int64, copy=False)
    if np.unique(result).size != result.size:
        raise ValueError(f"{role} IDs must be unique.")
    return result


def validate_membership_integrity(
    *,
    source_ids: Sequence[object],
    included_ids: Sequence[object],
    excluded_ids: Sequence[object],
    train_ids: Sequence[object],
    test_ids: Sequence[object],
    expected_counts: tuple[int, int, int, int, int] | None = None,
) -> dict[str, int]:
    """Require exact source and modeling partitions with no overlap.

    Args:
        source_ids: Complete assessed source population.
        included_ids: Final modeling population.
        excluded_ids: Excluded source population.
        train_ids: Locked training population.
        test_ids: Locked internal-test population.
        expected_counts: Optional source/included/excluded/train/test anchors.

    Returns:
        Aggregate counts only.

    Raises:
        ValueError: If IDs duplicate, overlap, omit members, or disagree with counts.
    """
    source = _as_unique_ids(source_ids, "Source")
    included = _as_unique_ids(included_ids, "Included")
    excluded = _as_unique_ids(excluded_ids, "Excluded")
    train = _as_unique_ids(train_ids, "Train")
    test = _as_unique_ids(test_ids, "Test")
    source_set = set(source.tolist())
    included_set = set(included.tolist())
    excluded_set = set(excluded.tolist())
    train_set = set(train.tolist())
    test_set = set(test.tolist())
    if included_set & excluded_set:
        raise ValueError("Included and excluded memberships must be disjoint.")
    if included_set | excluded_set != source_set:
        raise ValueError("Included and excluded memberships must exactly partition the source cohort.")
    if train_set & test_set:
        raise ValueError("Train and test memberships must be disjoint.")
    if train_set | test_set != included_set:
        raise ValueError("Train and test memberships must exactly partition the modeling cohort.")
    counts = {
        "source_n": source.size,
        "included_n": included.size,
        "excluded_n": excluded.size,
        "train_n": train.size,
        "test_n": test.size,
    }
    if expected_counts is not None and tuple(counts.values()) != expected_counts:
        raise ValueError("Membership count anchors do not reconcile.")
    return counts


def _exclusion_contract(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the frozen validated-single-reason contract."""
    return config["outputs"]["scientific_and_closure"]["cohort_flow"][
        "validated_single_exclusion_reason_contract"
    ]


def parse_single_exclusion_reasons(values: pd.Series, config: Mapping[str, Any]) -> pd.Series:
    """Parse semicolon-serialized failures and require exactly one closed-set key.

    Args:
        values: Excluded-row frozen failure-reason strings.
        config: Frozen post-hoc configuration.

    Returns:
        Validated reason keys in original order.

    Raises:
        ValueError: If a value is missing, unknown, malformed, or overlapping.
    """
    allowed = tuple(_exclusion_contract(config)["allowed_exclusion_reason_keys"])
    allowed_set = set(allowed)
    parsed: list[str] = []
    for value in values.tolist():
        if not isinstance(value, str):
            raise ValueError("Every excluded row must carry exactly one valid exclusion reason.")
        pieces = value.split(";")
        if len(pieces) != 1 or not pieces[0] or pieces[0] not in allowed_set:
            raise ValueError("Every excluded row must carry exactly one valid exclusion reason.")
        parsed.append(pieces[0])
    return pd.Series(parsed, index=values.index, dtype="string")


def build_cohort_flow(exclusion_counts: Mapping[str, int], config: Mapping[str, Any]) -> pd.DataFrame:
    """Build the exact configured 11-node aggregate cohort hierarchy.

    Args:
        exclusion_counts: Closed-set, mutually exclusive reason counts.
        config: Frozen post-hoc configuration.

    Returns:
        Typed cohort-flow frame in frozen row order.

    Raises:
        ValueError: If keys, counts, or arithmetic do not reconcile.
    """
    closure = config["reporting_closure"]["cohort_flow"]
    output = config["outputs"]["scientific_and_closure"]["cohort_flow"]
    allowed = tuple(output["validated_single_exclusion_reason_contract"]["allowed_exclusion_reason_keys"])
    if tuple(exclusion_counts) != allowed:
        raise ValueError("Exclusion-count keys do not match the exact frozen order.")
    if any(type(value) is not int or value < 0 for value in exclusion_counts.values()):
        raise ValueError("Exclusion counts must be nonnegative integers.")
    if sum(exclusion_counts.values()) != int(closure["excluded_n"]):
        raise ValueError("Exclusion counts must sum exactly to 68.")
    records: list[OrderedDict[str, Any]] = []
    for planned in output["planned_rows"]:
        record = OrderedDict((column, planned[column]) for column in output["columns"])
        if planned["stage"] == "exclusion":
            reason = planned["node_id"].removeprefix("excluded_")
            record["n"] = exclusion_counts[reason]
        records.append(record)
    frame = build_registry_frame(records, config["row_order_and_schema_validation_registry"]["cohort_flow"])
    validate_cohort_flow(frame, config)
    return frame


def validate_cohort_flow(frame: pd.DataFrame, config: Mapping[str, Any]) -> None:
    """Validate exact flow schema, row identity, and every frozen arithmetic identity."""
    output = config["outputs"]["scientific_and_closure"]["cohort_flow"]
    _validate_registered_frame(frame, config["row_order_and_schema_validation_registry"]["cohort_flow"])
    if frame.columns.tolist() != output["columns"] or len(frame) != output["planned_row_count"]:
        raise ValueError("Cohort-flow schema or row count is invalid.")
    planned = output["planned_rows"]
    if frame["node_id"].astype(str).tolist() != [item["node_id"] for item in planned]:
        raise ValueError("Cohort-flow row identity or order is invalid.")
    for row, expected in zip(frame.to_dict(orient="records"), planned, strict=True):
        for column in output["columns"]:
            if column != "n" and not _same_nullable_value(row[column], expected[column]):
                raise ValueError("Cohort-flow frozen row metadata is invalid.")
    counts = dict(zip(frame["node_id"].astype(str), frame["n"].astype(int), strict=True))
    exclusion_ids = [item["node_id"] for item in planned if item["stage"] == "exclusion"]
    if sum(counts[key] for key in exclusion_ids) != 68:
        raise ValueError("Cohort-flow exclusion children do not reconcile to 68.")
    if counts["assessed_source_cohort"] != 68 + counts["modeling_cohort"] or counts[
        "assessed_source_cohort"
    ] != 1040:
        raise ValueError("Cohort-flow assessed parent arithmetic is invalid.")
    if counts["modeling_cohort"] != counts["train"] + counts["internal_test"] or counts[
        "modeling_cohort"
    ] != 972:
        raise ValueError("Cohort-flow modeling parent arithmetic is invalid.")


def _same_nullable_value(left: Any, right: Any) -> bool:
    """Compare scalar contract values while treating nullable missing values as equal."""
    if pd.isna(left) and pd.isna(right):
        return True
    return bool(left == right)


def _validate_registered_frame(frame: pd.DataFrame, registry: Mapping[str, Any]) -> None:
    """Validate exact aggregate column order, row count, pandas dtypes, and nullability."""
    if frame.columns.tolist() != list(registry["column_order"]):
        raise ValueError("Aggregate frame column order is invalid.")
    expected_rows = registry.get("expected_row_count")
    if expected_rows is not None and len(frame) != int(expected_rows):
        raise ValueError("Aggregate frame row count is invalid.")
    for column in registry["column_order"]:
        if str(frame[column].dtype) != registry["dtype_map"][column]:
            raise ValueError("Aggregate frame dtype is invalid.")
        if not registry["nullable_map"][column] and frame[column].isna().any():
            raise ValueError("Aggregate frame contains a forbidden missing value.")


def load_exact_projection(
    path: Path,
    usecols: Sequence[str],
    expected_usecols: Sequence[str],
    *,
    reader: Callable[..., pd.DataFrame] = pd.read_csv,
) -> pd.DataFrame:
    """Load only one exact ordered CSV projection and reject broader results."""
    requested = list(usecols)
    expected = list(expected_usecols)
    if requested != expected:
        raise ValueError("Requested columns do not match the frozen ordered loader contract.")
    if "collection_location" in requested:
        raise ValueError("The prohibited collection-location field cannot be requested.")
    frame = reader(path, usecols=requested)
    if frame.columns.tolist() != expected:
        raise ValueError("Loaded columns do not exactly match the approved columns and order.")
    return frame


def load_inclusion_inputs(
    raw_metadata_path: Path,
    sample_index_path: Path,
    config: Mapping[str, Any],
    *,
    reader: Callable[..., pd.DataFrame] = pd.read_csv,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the two inclusion-comparison sources through their narrow contracts."""
    usecols = config["loaders"]["per_purpose_ordered_usecols"]
    metadata_columns = usecols["inclusion_exclusion_raw_metadata"]
    index_columns = usecols["inclusion_exclusion_sample_index"]
    metadata = load_exact_projection(raw_metadata_path, metadata_columns, metadata_columns, reader=reader)
    sample_index = load_exact_projection(sample_index_path, index_columns, index_columns, reader=reader)
    return metadata, sample_index


def _numeric_nonmissing(values: Sequence[object]) -> np.ndarray:
    """Return finite numeric nonmissing values while rejecting nonnumeric/infinite cells."""
    series = pd.Series(values)
    nonmissing = series.loc[~series.isna()]
    try:
        numeric = nonmissing.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("A numeric comparison field contains a nonnumeric value.") from exc
    if not np.all(np.isfinite(numeric)):
        raise ValueError("A numeric comparison field contains a nonfinite value.")
    return numeric


def _exact_binary_nonmissing(values: Sequence[object]) -> np.ndarray:
    """Return nonmissing exact numeric 0/1 values without lossy recoding."""
    series = pd.Series(values)
    nonmissing = series.loc[~series.isna()]
    array = nonmissing.to_numpy()
    if array.size == 0:
        return np.asarray([], dtype=np.int64)
    if array.dtype.kind not in "biuf":
        raise ValueError("Binary comparison values must be exact numeric 0 or 1.")
    numeric = array.astype(float)
    if not np.all(np.isfinite(numeric)) or not np.all((numeric == 0.0) | (numeric == 1.0)):
        raise ValueError("Binary comparison values must be exact numeric 0 or 1.")
    return numeric.astype(np.int64)


def describe_continuous(values: Sequence[object]) -> OrderedDict[str, int | float | None]:
    """Calculate retained age summaries with ddof=1 and linear quantiles."""
    series = pd.Series(values)
    clean = _numeric_nonmissing(values)
    result: OrderedDict[str, int | float | None] = OrderedDict(
        [
            ("n_total", int(series.size)),
            ("n_nonmissing", int(clean.size)),
            ("n_missing", int(series.isna().sum())),
            ("mean", None),
            ("sd", None),
            ("median", None),
            ("q1", None),
            ("q3", None),
        ]
    )
    if clean.size:
        result["mean"] = float(np.mean(clean))
        result["median"] = float(np.quantile(clean, 0.5, method="linear"))
        result["q1"] = float(np.quantile(clean, 0.25, method="linear"))
        result["q3"] = float(np.quantile(clean, 0.75, method="linear"))
    if clean.size >= 2:
        result["sd"] = float(np.std(clean, ddof=1))
    return result


def signed_continuous_smd(included: Sequence[object], excluded: Sequence[object]) -> SmdResult:
    """Compute the frozen signed continuous included-minus-excluded SMD."""
    left = _numeric_nonmissing(included)
    right = _numeric_nonmissing(excluded)
    if left.size < 2 or right.size < 2:
        return SmdResult(None, "non_estimable", "smd_fewer_than_two_nonmissing")
    left_mean = float(np.mean(left))
    right_mean = float(np.mean(right))
    denominator = float(np.sqrt((np.var(left, ddof=1) + np.var(right, ddof=1)) / 2.0))
    if denominator == 0.0:
        if left_mean == right_mean:
            return SmdResult(0.0, "estimable", None)
        return SmdResult(None, "non_estimable", "smd_zero_denominator_nonidentical")
    return SmdResult((left_mean - right_mean) / denominator, "estimable", None)


def signed_binary_smd(included: Sequence[object], excluded: Sequence[object]) -> SmdResult:
    """Compute the frozen signed binary-proportion included-minus-excluded SMD."""
    left = _exact_binary_nonmissing(included)
    right = _exact_binary_nonmissing(excluded)
    if left.size == 0 or right.size == 0:
        return SmdResult(None, "non_estimable", "smd_zero_denominator_nonidentical")
    left_p = float(np.mean(left))
    right_p = float(np.mean(right))
    denominator = float(np.sqrt((left_p * (1.0 - left_p) + right_p * (1.0 - right_p)) / 2.0))
    if denominator == 0.0:
        if left_p == right_p:
            return SmdResult(0.0, "estimable", None)
        return SmdResult(None, "non_estimable", "smd_zero_denominator_nonidentical")
    return SmdResult((left_p - right_p) / denominator, "estimable", None)


def _binary_summary(values: pd.Series) -> dict[str, int | float | None]:
    """Summarize one exact binary focal indicator with variable-specific complete cases."""
    clean = _exact_binary_nonmissing(values.tolist())
    count = int(clean.sum())
    return {
        "n_total": int(values.size),
        "n_nonmissing": int(clean.size),
        "n_missing": int(values.isna().sum()),
        "count": count,
        "proportion": float(count / clean.size) if clean.size else None,
    }


def _validate_join_inputs(metadata: pd.DataFrame, sample_index: pd.DataFrame) -> pd.DataFrame:
    """Require strict one-to-one source rows before inclusion calculations."""
    for frame, role in ((metadata, "metadata"), (sample_index, "sample index")):
        if "sample_id" not in frame or frame["sample_id"].isna().any() or not frame["sample_id"].is_unique:
            raise ValueError(f"The {role} join key must be complete and unique.")
    if set(metadata["sample_id"]) != set(sample_index["sample_id"]):
        raise ValueError("Metadata and sample-index populations do not match for one-to-one join.")
    try:
        joined = metadata.merge(sample_index, on="sample_id", how="inner", validate="one_to_one", sort=False)
    except pd.errors.MergeError as exc:
        raise ValueError("Inclusion inputs do not support a one-to-one join.") from exc
    if len(joined) != len(metadata):
        raise ValueError("One-to-one inclusion join lost source members.")
    flags = joined["in_final_output"]
    if flags.dtype != bool and str(flags.dtype) != "boolean":
        if not all(type(value) in {bool, np.bool_} for value in flags.tolist()):
            raise ValueError("Inclusion flags must be exact booleans.")
    return joined


def build_inclusion_comparison(
    metadata: pd.DataFrame,
    sample_index: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the exact 33 summary and nine signed-SMD records from in-memory rows."""
    joined = _validate_join_inputs(metadata, sample_index)
    included = joined.loc[joined["in_final_output"].astype(bool)].copy()
    excluded = joined.loc[~joined["in_final_output"].astype(bool)].copy()
    if included.empty or excluded.empty:
        raise ValueError("Both included and excluded groups are required.")
    excluded_reasons = parse_single_exclusion_reasons(excluded["fail_reasons"], config)
    exclusion_counts = excluded_reasons.value_counts().to_dict()
    summary_contract = config["outputs"]["scientific_and_closure"]["included_excluded_summary"]
    summary_records: list[OrderedDict[str, Any]] = []
    category_codes = dict(zip(DISEASE_LABELS, range(6), strict=True))

    def _values(frame: pd.DataFrame, variable: str, level: str | None, kind: str) -> pd.Series:
        if variable == "age":
            return frame["age"]
        if variable == "sex":
            return frame["sex(0male1female)"]
        if variable == "endpoint":
            return frame["label"]
        if variable == "disease_category":
            raw = frame["disease_category"]
            clean = _numeric_nonmissing(raw.tolist())
            if clean.size and (not np.all(clean == np.floor(clean)) or not np.all(np.isin(clean, np.arange(6)))):
                raise ValueError("Disease-category values must be exact integer codes 0 through 5.")
            return raw.map(
                lambda value: np.nan if pd.isna(value) else int(int(value) == category_codes[str(level)])
            )
        if variable.startswith("missingness__"):
            source = variable.removeprefix("missingness__")
            column = {
                "age": "age",
                "sex": "sex(0male1female)",
                "endpoint": "label",
                "disease_category": "disease_category",
            }[source]
            return frame[column].isna().astype(int)
        if variable == "qc_exclusion_reason" and kind == "excluded_group_count_only":
            return pd.Series([], dtype=int)
        raise ValueError("Unknown configured inclusion-summary record.")

    groups = {"included": included, "excluded": excluded}
    for planned in summary_contract["planned_record_tuples"]:
        group_name = planned["group"]
        frame = groups[group_name]
        record: OrderedDict[str, Any] = OrderedDict((column, None) for column in summary_contract["columns"])
        record.update(variable=planned["variable"], level=planned["level"], group=group_name, n_total=len(frame))
        kind = planned["statistics_kind"]
        if kind == "continuous":
            description = describe_continuous(_values(frame, planned["variable"], planned["level"], kind))
            for key in ("n_nonmissing", "n_missing", "mean", "sd", "median", "q1", "q3"):
                record[key] = description[key]
        elif kind in {"binary_focal", "one_versus_rest", "missingness_binary"}:
            values = _values(frame, planned["variable"], planned["level"], kind)
            record.update(_binary_summary(values))
        elif kind == "excluded_group_count_only":
            count = int(exclusion_counts.get(str(planned["level"]), 0))
            record.update(
                n_nonmissing=len(frame),
                n_missing=0,
                count=count,
                proportion=float(count / len(frame)),
            )
        else:
            raise ValueError("Unknown inclusion-summary statistics kind.")
        summary_records.append(record)

    smd_contract = config["outputs"]["scientific_and_closure"]["included_excluded_smd"]
    smd_records: list[OrderedDict[str, Any]] = []
    for planned in smd_contract["planned_record_tuples"]:
        variable = planned["variable"]
        level = planned["level"]
        if variable == "age":
            left = included["age"]
            right = excluded["age"]
            result = signed_continuous_smd(left, right)
            missingness = signed_binary_smd(left.isna().astype(int), right.isna().astype(int))
        elif variable == "sex":
            left = included["sex(0male1female)"]
            right = excluded["sex(0male1female)"]
            result = signed_binary_smd(left, right)
            missingness = signed_binary_smd(left.isna().astype(int), right.isna().astype(int))
        elif variable == "endpoint":
            left = included["label"]
            right = excluded["label"]
            result = signed_binary_smd(left, right)
            missingness = signed_binary_smd(left.isna().astype(int), right.isna().astype(int))
        else:
            code = category_codes[str(level)]
            left_raw = included["disease_category"]
            right_raw = excluded["disease_category"]
            for raw in (left_raw, right_raw):
                clean = _numeric_nonmissing(raw.tolist())
                if clean.size and (not np.all(clean == np.floor(clean)) or not np.all(np.isin(clean, np.arange(6)))):
                    raise ValueError("Disease-category values must be exact integer codes 0 through 5.")
            left = left_raw.map(lambda value: np.nan if pd.isna(value) else int(int(value) == code))
            right = right_raw.map(lambda value: np.nan if pd.isna(value) else int(int(value) == code))
            result = signed_binary_smd(left, right)
            missingness = signed_binary_smd(left_raw.isna().astype(int), right_raw.isna().astype(int))
        missing_estimate = missingness.estimate if planned.get("missingness_smd_applicable") is True else None
        smd_records.append(
            OrderedDict(
                [
                    ("variable", variable),
                    ("level", level),
                    ("smd_included_minus_excluded", result.estimate),
                    ("missingness_smd_included_minus_excluded", missing_estimate),
                    ("n_included_nonmissing", int(pd.Series(left).notna().sum())),
                    ("n_excluded_nonmissing", int(pd.Series(right).notna().sum())),
                    ("estimability_status", result.status),
                    ("nonestimable_reason", result.reason),
                ]
            )
        )
    summary = build_registry_frame(
        summary_records,
        config["row_order_and_schema_validation_registry"]["included_excluded_summary"],
    )
    smd = build_registry_frame(
        smd_records,
        config["row_order_and_schema_validation_registry"]["included_excluded_smd"],
    )
    return summary, smd


def validate_inclusion_comparison(
    summary: pd.DataFrame,
    smd: pd.DataFrame,
    metadata: pd.DataFrame,
    sample_index: pd.DataFrame,
    config: Mapping[str, Any],
) -> None:
    """Recompute and require exact signed inclusion-comparison outputs."""
    expected_summary, expected_smd = build_inclusion_comparison(metadata, sample_index, config)
    try:
        pd.testing.assert_frame_equal(summary, expected_summary, check_exact=True)
        pd.testing.assert_frame_equal(smd, expected_smd, check_exact=True)
    except AssertionError as exc:
        raise ValueError("Inclusion-comparison output does not match the frozen recompute contract.") from exc


def _validate_calibration_inputs(
    y_true: Sequence[object],
    p_cal: Sequence[object],
    *,
    expected_n: int = 195,
) -> tuple[np.ndarray, np.ndarray]:
    """Require exactly aligned finite binary labels and unit-interval probabilities."""
    labels_raw = np.asarray(y_true)
    scores_raw = np.asarray(p_cal)
    if labels_raw.ndim != 1 or scores_raw.ndim != 1 or labels_raw.size != scores_raw.size:
        raise ValueError(f"Calibration inputs must be aligned one-dimensional vectors of exactly {expected_n} rows.")
    if labels_raw.size != expected_n:
        raise ValueError(f"Calibration inputs must contain exactly {expected_n} rows.")
    if labels_raw.dtype.kind not in "biuf":
        raise ValueError("Calibration labels must contain exact numeric binary values.")
    labels = validate_exact_binary_labels_01(labels_raw)
    try:
        scores = scores_raw.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Calibration probabilities must be numeric.") from exc
    if not np.all(np.isfinite(scores)):
        raise ValueError("Calibration probabilities must be finite.")
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("Calibration probabilities must lie in [0, 1].")
    return labels, scores


def build_calibration_points(
    y_true: Sequence[object],
    p_cal: Sequence[object],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Create deterministic qcut calibration points with Wilson intervals."""
    labels, scores = _validate_calibration_inputs(y_true, p_cal)
    contract = config["outputs"]["scientific_and_closure"]["calibration_points"]
    requested = int(contract["conditional_row_inventory_contract"]["n_requested_bins"])
    categories = pd.qcut(pd.Series(scores), q=requested, duplicates="drop")
    category_count = int(categories.cat.categories.size)
    grouped_indices: list[np.ndarray]
    if category_count == 0 or categories.isna().all():
        grouped_indices = [np.arange(scores.size)]
    else:
        grouped_indices = [
            np.flatnonzero((categories == category).to_numpy()) for category in categories.cat.categories
        ]
        grouped_indices = [indices for indices in grouped_indices if indices.size]
        if not grouped_indices:
            grouped_indices = [np.arange(scores.size)]
    actual = len(grouped_indices)
    ready = actual >= 5
    records: list[OrderedDict[str, Any]] = []
    for index, indices in enumerate(grouped_indices):
        n = int(indices.size)
        events = int(labels[indices].sum())
        low, high = wilson_proportion_ci(events, n)
        records.append(
            OrderedDict(
                [
                    ("bin_index", index),
                    ("n", n),
                    ("events", events),
                    ("nonevents", n - events),
                    ("p_cal_mean", float(np.mean(scores[indices]))),
                    ("observed_fraction", float(events / n)),
                    ("observed_ci_low", low),
                    ("observed_ci_high", high),
                    ("ci_method", "wilson"),
                    ("n_requested_bins", requested),
                    ("n_actual_bins", actual),
                    ("visualization_ready", ready),
                    ("estimability_status", "estimable"),
                    ("nonestimable_reason", None),
                ]
            )
        )
    frame = build_registry_frame(records, config["row_order_and_schema_validation_registry"]["calibration_points"])
    validate_calibration_points(frame, expected_n=labels.size, expected_events=int(labels.sum()), config=config)
    return frame


def validate_calibration_points(
    frame: pd.DataFrame,
    *,
    expected_n: int,
    expected_events: int,
    config: Mapping[str, Any],
) -> None:
    """Validate conditional bin inventory, Wilson states, and exact count identities."""
    contract = config["outputs"]["scientific_and_closure"]["calibration_points"]
    if frame.columns.tolist() != contract["columns"] or not 1 <= len(frame) <= 10:
        raise ValueError("Calibration-point schema or conditional row count is invalid.")
    actual = len(frame)
    if frame["bin_index"].astype(int).tolist() != list(range(actual)):
        raise ValueError("Calibration-point bin order is invalid.")
    if set(frame["n_requested_bins"].astype(int)) != {10} or set(frame["n_actual_bins"].astype(int)) != {actual}:
        raise ValueError("Calibration-point requested/actual bin fields are invalid.")
    if int(frame["n"].sum()) != expected_n or int(frame["events"].sum()) != expected_events:
        raise ValueError("Calibration-point totals do not reconcile.")
    if not np.array_equal(
        frame["n"].to_numpy(dtype=int),
        frame["events"].to_numpy(dtype=int) + frame["nonevents"].to_numpy(dtype=int),
    ):
        raise ValueError("Calibration-point event and nonevent counts do not reconcile.")
    if set(frame["ci_method"].astype(str)) != {"wilson"}:
        raise ValueError("Calibration-point CI method is invalid.")
    if set(frame["estimability_status"].astype(str)) != {"estimable"} or frame[
        "nonestimable_reason"
    ].notna().any():
        raise ValueError("Calibration-point status fields are invalid.")
    if set(frame["visualization_ready"].astype(bool)) != {actual >= 5}:
        raise ValueError("Calibration-point visualization readiness is invalid.")
    _validate_registered_frame(frame, config["row_order_and_schema_validation_registry"]["calibration_points"])
    for row in frame.to_dict(orient="records"):
        n = int(row["n"])
        events = int(row["events"])
        nonevents = int(row["nonevents"])
        if n <= 0 or min(events, nonevents) < 0 or events + nonevents != n:
            raise ValueError("Calibration-point row counts are invalid.")
        if not 0.0 <= float(row["p_cal_mean"]) <= 1.0:
            raise ValueError("Calibration-point mean probability is invalid.")
        expected_fraction = events / n
        if not np.isclose(float(row["observed_fraction"]), expected_fraction, rtol=0.0, atol=1e-15):
            raise ValueError("Calibration-point observed fraction is invalid.")
        expected_low, expected_high = wilson_proportion_ci(events, n)
        if not np.isclose(float(row["observed_ci_low"]), expected_low, rtol=0.0, atol=1e-15) or not np.isclose(
            float(row["observed_ci_high"]), expected_high, rtol=0.0, atol=1e-15
        ):
            raise ValueError("Calibration-point Wilson interval is invalid.")


def _lowess_grid_records(
    *,
    smoothed: np.ndarray | None,
    support: np.ndarray,
    global_reason: str | None,
) -> list[OrderedDict[str, Any]]:
    """Build fixed-grid LOWESS records without persisting support extrema."""
    grid = np.linspace(0.0, 1.0, 101)
    records: list[OrderedDict[str, Any]] = []
    for index, grid_value in enumerate(grid):
        in_support = bool(support[index])
        if global_reason is not None:
            value = None
            clipped = False
            status = "non_estimable"
            reason = global_reason
        elif not in_support:
            value = None
            clipped = False
            status = "not_evaluated"
            reason = "outside_observed_support"
        else:
            if smoothed is None or not np.isfinite(smoothed[index]):
                raise ValueError("LOWESS returned a nonfinite supported-grid value.")
            raw = float(smoothed[index])
            value = float(np.clip(raw, 0.0, 1.0))
            clipped = value != raw
            status = "estimable"
            reason = None
        records.append(
            OrderedDict(
                [
                    ("grid_index", index),
                    ("p_cal_grid", float(grid_value)),
                    ("identity_observed_fraction", float(grid_value)),
                    ("smoothed_observed_fraction", value),
                    ("in_observed_support", in_support),
                    ("boundary_clipped", clipped),
                    ("estimator", LOWESS_ESTIMATOR),
                    ("estimability_status", status),
                    ("nonestimable_reason", reason),
                ]
            )
        )
    return records


def build_calibration_smoother(
    y_true: Sequence[object],
    p_cal: Sequence[object],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Create the frozen tie-stabilized descriptive LOWESS grid."""
    labels, scores = _validate_calibration_inputs(y_true, p_cal)
    grid = np.linspace(0.0, 1.0, 101)
    support = (grid >= float(np.min(scores))) & (grid <= float(np.max(scores)))
    global_reason: str | None = None
    if np.unique(labels).size < 2:
        global_reason = "all_one_outcome_class"
    elif np.unique(scores).size < 2:
        global_reason = "fewer_than_two_distinct_scores"
    fitted: np.ndarray | None = None
    if global_reason is None:
        order = np.argsort(scores, kind="mergesort")
        sorted_scores = scores[order]
        sorted_labels = labels[order].astype(float)
        stabilized = sorted_labels.copy()
        start = 0
        while start < sorted_scores.size:
            stop = start + 1
            while stop < sorted_scores.size and sorted_scores[stop] == sorted_scores[start]:
                stop += 1
            stabilized[start:stop] = float(np.mean(sorted_labels[start:stop]))
            start = stop
        fitted = np.asarray(
            lowess(
                endog=stabilized,
                exog=sorted_scores,
                frac=2.0 / 3.0,
                it=0,
                delta=0.0,
                xvals=grid,
                is_sorted=True,
                missing="none",
                return_sorted=False,
            ),
            dtype=float,
        )
        if fitted.shape != grid.shape:
            raise ValueError("LOWESS output does not match the fixed 101-row grid.")
    records = _lowess_grid_records(smoothed=fitted, support=support, global_reason=global_reason)
    frame = build_registry_frame(records, config["row_order_and_schema_validation_registry"]["calibration_smoother"])
    validate_calibration_smoother(frame, config)
    return frame


def validate_calibration_smoother(frame: pd.DataFrame, config: Mapping[str, Any]) -> None:
    """Validate exact LOWESS row identity, masking, clipping, and status dependencies."""
    contract = config["outputs"]["scientific_and_closure"]["calibration_smoother"]
    _validate_registered_frame(frame, config["row_order_and_schema_validation_registry"]["calibration_smoother"])
    if frame.columns.tolist() != contract["columns"] or len(frame) != 101:
        raise ValueError("Calibration-smoother schema or row count is invalid.")
    grid = np.linspace(0.0, 1.0, 101)
    if frame["grid_index"].astype(int).tolist() != list(range(101)) or not np.array_equal(
        frame["p_cal_grid"].to_numpy(dtype=float), grid
    ):
        raise ValueError("Calibration-smoother fixed grid is invalid.")
    if not np.array_equal(frame["identity_observed_fraction"].to_numpy(dtype=float), grid):
        raise ValueError("Calibration-smoother identity column is invalid.")
    if set(frame["estimator"].astype(str)) != {LOWESS_ESTIMATOR}:
        raise ValueError("Calibration-smoother estimator identity is invalid.")
    allowed = set(config["status_and_reason_registries"]["lowess_row_estimability_status"])
    if not set(frame["estimability_status"].astype(str)).issubset(allowed):
        raise ValueError("Calibration-smoother status enum is invalid.")
    for row in frame.to_dict(orient="records"):
        status = str(row["estimability_status"])
        missing = pd.isna(row["smoothed_observed_fraction"])
        if status == "estimable":
            if missing or row["nonestimable_reason"] is not pd.NA and not pd.isna(row["nonestimable_reason"]):
                raise ValueError("Estimable LOWESS rows require a value and empty reason.")
            if not bool(row["in_observed_support"]):
                raise ValueError("Estimable LOWESS rows must be inside observed support.")
            if not 0.0 <= float(row["smoothed_observed_fraction"]) <= 1.0:
                raise ValueError("Estimable LOWESS values must lie in [0, 1].")
        elif status == "not_evaluated":
            if not missing or str(row["nonestimable_reason"]) != "outside_observed_support":
                raise ValueError("Outside-support LOWESS row population is invalid.")
            if bool(row["in_observed_support"]) or bool(row["boundary_clipped"]):
                raise ValueError("Outside-support LOWESS flags are invalid.")
        else:
            if not missing or str(row["nonestimable_reason"]) not in {
                "all_one_outcome_class",
                "fewer_than_two_distinct_scores",
            }:
                raise ValueError("Globally non-estimable LOWESS row population is invalid.")
            if bool(row["boundary_clipped"]):
                raise ValueError("Globally non-estimable LOWESS rows cannot be clipped.")


def build_reference_standard_timing_audit(
    config: Mapping[str, Any],
    *,
    generated_at_utc: str,
    access_state: Mapping[str, Any],
) -> OrderedDict[str, Any]:
    """Build the deterministic 13-record conservative fact audit."""
    if not is_utc_iso8601_z(generated_at_utc):
        raise ValueError("Fact-audit timestamp is invalid.")
    output = config["outputs"]["scientific_and_closure"]["reference_standard_timing_audit"]
    facts: list[OrderedDict[str, Any]] = []
    item_keys = output["schema"]["leaf_contracts"]["facts"]["item_key_order"]
    for planned in output["planned_fact_records"]:
        facts.append(
            OrderedDict(
                (key, None if key == "source_reference" else planned[key]) for key in item_keys
            )
        )
    payload: OrderedDict[str, Any] = OrderedDict(
        [
            ("schema_version", "posthoc_final_reference_standard_timing_audit.v1"),
            ("package_id", "posthoc_final_supplementary"),
            ("generated_at_utc", generated_at_utc),
            ("facts", facts),
            ("access_state", OrderedDict(access_state)),
            (
                "prohibited_fields",
                [
                    "collection_time",
                    "collection_location",
                    "sample_id",
                    "filename",
                    "participant_level_scores",
                    "exact_clinical_collection_dates",
                ],
            ),
        ]
    )
    validate_reference_standard_timing_audit(payload, config)
    return payload


def _validate_access_shape(access_state: Mapping[str, Any]) -> None:
    """Validate the exact two-scope, nine-boolean access-state shape."""
    if tuple(access_state) != ("session_activity", "cumulative_through_session"):
        raise ValueError("Access-state object order is invalid.")
    for scope in access_state.values():
        if not isinstance(scope, Mapping) or tuple(scope) != tuple(ACCESS_STATE_KEYS) or any(
            type(value) is not bool for value in scope.values()
        ):
            raise ValueError("Access-state scopes must contain nine ordered booleans.")


def validate_reference_standard_timing_audit(
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    """Require exact fact identities, statuses, ordering, and conservative wording."""
    output = config["outputs"]["scientific_and_closure"]["reference_standard_timing_audit"]
    schema = output["schema"]
    if tuple(payload) != tuple(schema["top_level_key_order"]):
        raise ValueError("Fact-audit top-level key order is invalid.")
    if payload["schema_version"] != "posthoc_final_reference_standard_timing_audit.v1" or payload[
        "package_id"
    ] != "posthoc_final_supplementary":
        raise ValueError("Fact-audit identity is invalid.")
    if not is_utc_iso8601_z(payload["generated_at_utc"]):
        raise ValueError("Fact-audit provenance timestamp is invalid.")
    facts = payload["facts"]
    planned = output["planned_fact_records"]
    item_keys = tuple(schema["leaf_contracts"]["facts"]["item_key_order"])
    if not isinstance(facts, list) or len(facts) != 13:
        raise ValueError("Fact-audit fact count is invalid.")
    for observed, expected in zip(facts, planned, strict=True):
        if not isinstance(observed, Mapping) or tuple(observed) != item_keys:
            raise ValueError("Fact-audit item schema is invalid.")
        for key in item_keys[:-1]:
            if observed[key] != expected[key] or type(observed[key]) is not type(expected[key]):
                raise ValueError("A frozen fact identity, wording, or status was changed.")
        source = observed["source_reference"]
        if source is not None and (not isinstance(source, str) or Path(source).is_absolute()):
            raise ValueError("Fact-audit source references must be nullable project-relative paths.")
    _validate_access_shape(payload["access_state"])
    prohibited = schema["leaf_contracts"]["prohibited_fields"]["exact_values_in_order"]
    if payload["prohibited_fields"] != prohibited:
        raise ValueError("Fact-audit prohibited-field inventory is invalid.")


def generate_qc6_lineage_correction(
    config: Mapping[str, Any],
    correction_date_utc: str,
) -> str:
    """Render the deterministic additive QC6 correction note without writing it."""
    if not is_utc_iso8601_z(correction_date_utc):
        raise ValueError("QC6 correction provenance timestamp is invalid.")
    cited = config["reporting_closure"]["qc6_lineage_correction"]["must_cite"]
    return (
        "# Additive QC6 reporting-lineage correction\n\n"
        f"- correction_date_utc: `{correction_date_utc}`\n"
        "- Scope: reporting closure only; this note does not edit the frozen protocol or run summary.\n\n"
        "The operational plan used an any-channel rule: any gas-channel spike count above the configured threshold "
        "fails "
        "QC6. The executed implementation applied `not any(spike_counts > configured_threshold)` pass semantics in "
        f"`src/data/qc.py::_evaluate_qc6` (commit `{cited['code_commit']}`). The frozen threshold was "
        f"`{float(cited['threshold']):.1f}` in `{cited['threshold_config']}` (provenance commit "
        f"`{cited['threshold_provenance_commit']}`), so equality at 211.0 passes under the strict `>` rule.\n\n"
        "The frozen protocol omitted the sample-level aggregation rule, while the historical frozen run summary used "
        "inaccurate six-channel quorum wording "
        f"(commit `{cited['run_summary_commit']}`). Frozen QC artifacts from commit "
        f"`{cited['qc_artifacts_commit']}` recorded zero QC6 failures. Consequently, this wording correction had no "
        "cohort or model impact and changes no analytical result. Future Methods and cohort-flow text must use the "
        "actually executed any-channel rule.\n"
    )


def validate_qc6_lineage_correction(
    text: str,
    config: Mapping[str, Any],
    *,
    correction_date_utc: str,
) -> None:
    """Require byte-deterministic QC6 correction content from the frozen facts."""
    expected = generate_qc6_lineage_correction(config, correction_date_utc)
    if text != expected:
        raise ValueError("QC6 correction text does not match the frozen deterministic content.")


def _git(*args: str) -> str:
    """Run one read-only Git command from the project root."""
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _reporting_paths(config: Mapping[str, Any]) -> tuple[tuple[str, ...], str]:
    """Return the exact Session 6 output inventory and completion-log path."""
    grammar = config["outputs"]["completion_logs"]["machine_validated_log_grammar"]
    binding = grammar["session_bindings_exact"]["reporting"]
    return tuple(binding["ordered_output_inventory"]), str(binding["log_path"])


def reporting_promotion_inventory(config: Mapping[str, Any]) -> tuple[tuple[str, ...], str]:
    """Return the exact nine Session 6 outputs and log destination.

    Args:
        config: Frozen post-hoc configuration.

    Returns:
        Ordered output paths and the completion-log path.

    Raises:
        ValueError: If the configured inventory is not the frozen nine-output contract.
    """
    outputs, log_path = _reporting_paths(config)
    if len(outputs) != 9 or log_path in outputs:
        raise ValueError("Session 6 promotion inventory is invalid.")
    return outputs, log_path


def _classify_paths(paths: Sequence[Path], log_path: Path) -> str:
    """Classify absent, partial, or completed individual-promotion state."""
    present = [path.exists() for path in paths]
    log_present = log_path.exists()
    if not any(present) and not log_present:
        return "clean"
    if all(present) and log_present:
        return "complete"
    return "partial"


def validate_reporting_authority(
    config_path: Path,
    freeze_commit: str,
    amendment_freeze_commit: str,
    *,
    git_runner: Callable[..., str] = _git,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Validate current authority, both freezes, preflight, prerequisites, and stale state."""
    root = (project_root or PROJECT_ROOT).resolve()
    if _COMMIT_PATTERN.fullmatch(freeze_commit) is None or _COMMIT_PATTERN.fullmatch(amendment_freeze_commit) is None:
        raise ValueError("Both freeze commits must be exact lowercase 40-character hashes.")
    resolved = config_path if config_path.is_absolute() else root / config_path
    config = load_effective_posthoc_config(resolved)
    manifest = json.loads((root / CURRENT_MANIFEST_RELATIVE_PATH).read_text(encoding="utf-8"))
    current = validate_current_runtime_authority(
        resolved,
        config,
        manifest,
        require_head_identical=True,
        git_runner=git_runner,
    )
    _validate_annotated_tag_chain(git_runner)
    session0 = validate_freeze_commit(freeze_commit, config, manifest, git_runner=git_runner)
    amendment = validate_amendment_freeze_commit(
        amendment_freeze_commit,
        historical_amendment=manifest["historical_evidence"]["amendment_01"],
        git_runner=git_runner,
    )
    preflight_path = root / SCORE_BLIND_PREFLIGHT_RELATIVE_PATH
    expected_preflight_hash = manifest["historical_evidence"]["completed_session_1"]["artifact_sha256"][
        SCORE_BLIND_PREFLIGHT_RELATIVE_PATH
    ]
    if sha256_file(preflight_path) != expected_preflight_hash:
        raise ValueError("Score-blind preflight hash does not match the frozen authority.")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"), object_pairs_hook=OrderedDict)
    prior = OrderedDict(
        config["historical_consolidation"]["amendment_01"]["access_state"]
        ["step_2_approval_materialization"]["cumulative_through_session"]
    )
    validate_preflight_schema(
        preflight,
        frozen_hash_paths=list(config["frozen_inputs"]["whole_container_sha256"]),
        expected_cumulative_prior=prior,
    )
    validate_preflight_privacy(preflight)
    if preflight["status"] != "pass" or preflight["freeze_commit_validation"]["freeze_commit_id"] != freeze_commit:
        raise ValueError("A passing committed score-blind preflight is required.")
    prior_paths = [
        "configs/post_hoc_final_supplementary.yaml",
        config["outputs"]["session_0"]["methods"],
        config["outputs"]["session_0"]["freeze_manifest"],
        SCORE_BLIND_PREFLIGHT_RELATIVE_PATH,
        *config["outputs"]["future_implementation_paths"],
        *config["outputs"]["completion_logs"]["machine_validated_log_grammar"]["session_bindings_exact"]
        ["locked_scores"]["ordered_output_inventory"],
        config["outputs"]["completion_logs"]["locked_scores"],
        *config["outputs"]["completion_logs"]["machine_validated_log_grammar"]["session_bindings_exact"]
        ["date_grouped"]["ordered_output_inventory"],
        config["outputs"]["completion_logs"]["date_grouped"],
    ]
    validate_execute_git_prerequisites(tuple(dict.fromkeys(prior_paths)), git_runner=git_runner)
    relative_targets, relative_log = reporting_promotion_inventory(config)
    state = _classify_paths([root / path for path in relative_targets], root / relative_log)
    if state != "clean":
        raise ValueError(f"Session 6 target state is {state}; provenance review is required.")
    locked_state, _, _ = validate_completed_session_from_disk(
        project_root=root,
        session_name="locked_scores",
        freeze_commit=freeze_commit,
        cumulative_prior=preflight["access_state"]["cumulative_through_session"],
        config=config,
    )
    date_state, _, _ = validate_completed_session_from_disk(
        project_root=root,
        session_name="date_grouped",
        freeze_commit=freeze_commit,
        cumulative_prior=locked_state["cumulative_through_session"],
        config=config,
    )
    protected_paths = tuple(
        dict.fromkeys(
            [
                *config["hash_map_inventories"]["execution_manifest_input_sha256_keys_exact_order"],
                *config["hash_map_inventories"]["execution_manifest_code_sha256_keys_exact_order"],
                config["authority"]["reviewed_plan_path"],
                config["authority"]["agent_guidance_path"],
            ]
        )
    )
    entry_hashes = capture_integrity_snapshot(
        root,
        protected_paths,
        declared_hashes=config["frozen_inputs"]["whole_container_sha256"],
    )
    return {
        "config": config,
        "manifest": manifest,
        "current_runtime": current,
        "session0_freeze": session0,
        "amendment_freeze": amendment,
        "preflight_cumulative": OrderedDict(preflight["access_state"]["cumulative_through_session"]),
        "cumulative_prior": OrderedDict(date_state["cumulative_through_session"]),
        "entry_hashes": entry_hashes,
        "required_paths": tuple(dict.fromkeys(prior_paths)),
        "project_root": root,
    }


def load_reporting_execution_inputs(authority: Mapping[str, Any]) -> dict[str, Any]:
    """Load only the future Session 6 minimal sources after authority validation."""
    config = authority["config"]
    root = Path(authority.get("project_root", PROJECT_ROOT))
    paths_and_roles = config["frozen_inputs"]["paths_and_roles"]

    def path_for_role(role: str) -> Path:
        """Resolve exactly one configured path for a frozen input role."""
        matches = [path for path, configured_role in paths_and_roles.items() if configured_role == role]
        if len(matches) != 1:
            raise ValueError(f"Frozen input role must resolve exactly once: {role}.")
        return root / matches[0]

    metadata, sample_index = load_inclusion_inputs(
        path_for_role("age_sex_inclusion_descriptors_only_never_persisted_rowwise"),
        path_for_role("master_cohort_qc_category_lineage_and_sole_collection_time_source"),
        config,
    )
    calibration_columns = config["loaders"]["per_purpose_ordered_usecols"]["calibration_loader_test_only"]
    test_predictions = load_exact_projection(
        path_for_role("frozen_internal_test_p_raw_and_sole_p_cal"),
        calibration_columns,
        calibration_columns,
    )
    split_path = path_for_role("locked_777_195_membership")
    with np.load(split_path, allow_pickle=False) as split:
        train_idx = np.asarray(split["train_idx"], dtype=np.int64)
        test_idx = np.asarray(split["test_idx"], dtype=np.int64)
    return {
        "metadata": metadata,
        "sample_index": sample_index,
        "test_predictions": test_predictions,
        "train_idx": train_idx,
        "test_idx": test_idx,
    }


def load_validated_prior_aggregates(authority: Mapping[str, Any]) -> OrderedDict[str, pd.DataFrame]:
    """Load the exact hash-validated Session 4/5 aggregate outputs for reporting."""
    config = authority["config"]
    root = Path(authority.get("project_root", PROJECT_ROOT))
    registry_by_path = {
        entry["path"]: entry
        for entry in config["row_order_and_schema_validation_registry"].values()
        if isinstance(entry, Mapping) and "path" in entry and "column_order" in entry
    }
    frames = OrderedDict(
        (path, read_typed_registry_csv(root / path, registry_by_path[path]))
        for path in _prior_aggregate_inventory(config)
    )
    validate_prior_aggregate_frames(frames, config)
    return frames


def align_reporting_execution_inputs(
    inputs: Mapping[str, Any],
    config: Mapping[str, Any],
) -> ReportingAlignmentEvidence:
    """Bind calibration rows exactly to the locked internal-test membership.

    Args:
        inputs: Narrowly loaded sample index, split indices, and calibration rows.
        config: Frozen post-hoc configuration supplying cohort anchors.

    Returns:
        Deterministically reindexed, label-validated in-memory evidence.

    Raises:
        ValueError: If membership, IDs, labels, scores, or deterministic order fail.
    """
    sample_index = inputs.get("sample_index")
    predictions = inputs.get("test_predictions")
    if not isinstance(sample_index, pd.DataFrame) or not isinstance(predictions, pd.DataFrame):
        raise ValueError("Reporting alignment requires loaded sample-index and calibration frames.")
    if sample_index["sample_id"].isna().any() or not sample_index["sample_id"].is_unique:
        raise ValueError("Sample-index IDs must be complete and unique.")
    final = sample_index.loc[sample_index["in_final_output"].astype(bool)].reset_index(drop=True)
    excluded = sample_index.loc[~sample_index["in_final_output"].astype(bool)].reset_index(drop=True)
    split_arrays: dict[str, np.ndarray] = {}
    for key in ("train_idx", "test_idx"):
        raw = np.asarray(inputs.get(key))
        if raw.ndim != 1 or raw.dtype.kind not in "iu" or raw.size == 0:
            raise ValueError("Locked split indices must be nonempty one-dimensional integer arrays.")
        values = raw.astype(np.int64, copy=False)
        if np.unique(values).size != values.size or np.any(values < 0) or np.any(values >= len(final)):
            raise ValueError("Locked split indices are duplicated or outside the modeling population.")
        if values.size > 1 and np.any(np.diff(values) <= 0):
            raise ValueError("Locked split indices must retain their deterministic increasing order.")
        split_arrays[key] = values
    train_ids = final.iloc[split_arrays["train_idx"]]["sample_id"].to_numpy()
    test_ids = final.iloc[split_arrays["test_idx"]]["sample_id"].to_numpy()
    closure = config["reporting_closure"]["cohort_flow"]
    expected = (
        int(closure["assessed_n"]),
        int(closure["modeling_cohort_n"]),
        int(closure["excluded_n"]),
        int(closure["train_n"]),
        int(closure["internal_test_n"]),
    )
    validate_membership_integrity(
        source_ids=sample_index["sample_id"].to_numpy(),
        included_ids=final["sample_id"].to_numpy(),
        excluded_ids=excluded["sample_id"].to_numpy(),
        train_ids=train_ids,
        test_ids=test_ids,
        expected_counts=expected,
    )
    expected_columns = list(config["loaders"]["per_purpose_ordered_usecols"]["calibration_loader_test_only"])
    if predictions.columns.tolist() != expected_columns:
        raise ValueError("Calibration input columns do not match the exact loader contract.")
    if predictions["sample_id"].isna().any() or not predictions["sample_id"].is_unique:
        raise ValueError("Calibration sample IDs must be complete and unique.")
    if set(predictions["sample_id"].tolist()) != set(test_ids.tolist()):
        raise ValueError("Calibration sample IDs do not exactly match the locked internal-test population.")
    aligned = predictions.set_index("sample_id", verify_integrity=True).loc[test_ids.tolist()].reset_index()
    if aligned["sample_id"].tolist() != test_ids.tolist():
        raise ValueError("Calibration rows could not be deterministically reindexed to locked test order.")
    frozen_labels = validate_exact_binary_labels_01(
        final.iloc[split_arrays["test_idx"]]["label"].to_numpy()
    )
    calibration_labels, scores = _validate_calibration_inputs(
        aligned["y_true"].to_numpy(),
        aligned["p_cal"].to_numpy(),
        expected_n=expected[-1],
    )
    if not np.array_equal(calibration_labels, frozen_labels):
        raise ValueError("Calibration y_true does not exactly equal the aligned frozen sample-index labels.")
    aligned = aligned.copy()
    aligned["y_true"] = calibration_labels
    aligned["p_cal"] = scores
    return ReportingAlignmentEvidence(
        final=final,
        excluded=excluded,
        train_ids=np.asarray(train_ids),
        test_ids=np.asarray(test_ids),
        calibration=aligned,
        expected_test_n=expected[-1],
    )


def calculate_reporting_candidates(authority: Mapping[str, Any], inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Calculate in-memory closure candidates without promotion."""
    config = authority["config"]
    evidence = align_reporting_execution_inputs(inputs, config)
    sample_index = inputs["sample_index"]
    reasons = parse_single_exclusion_reasons(evidence.excluded["fail_reasons"], config)
    allowed = _exclusion_contract(config)["allowed_exclusion_reason_keys"]
    counts = OrderedDict((key, int((reasons == key).sum())) for key in allowed)
    summary, smd = build_inclusion_comparison(inputs["metadata"], sample_index, config)
    predictions = evidence.calibration
    points = build_calibration_points(predictions["y_true"], predictions["p_cal"], config)
    smoother = build_calibration_smoother(predictions["y_true"], predictions["p_cal"], config)
    return {
        "cohort_flow": build_cohort_flow(counts, config),
        "included_excluded_summary": summary,
        "included_excluded_smd": smd,
        "calibration_points": points,
        "calibration_smoother": smoother,
    }


def validate_reporting_candidates(
    candidates: Mapping[str, Any],
    authority: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> None:
    """Recompute every scientific candidate from independently aligned source inputs."""
    config = authority["config"]
    required = (
        "cohort_flow",
        "included_excluded_summary",
        "included_excluded_smd",
        "calibration_points",
        "calibration_smoother",
    )
    if tuple(candidates) != required:
        raise ValueError("Reporting candidate inventory is incomplete or reordered.")
    expected = calculate_reporting_candidates(authority, inputs)
    for name in required:
        if not isinstance(candidates[name], pd.DataFrame):
            raise ValueError("Reporting scientific candidates must be aggregate data frames.")
        observed_frame = candidates[name].copy()
        expected_frame = expected[name].copy()
        for column in observed_frame.columns:
            if str(observed_frame[column].dtype) == "category":
                observed_frame[column] = observed_frame[column].astype("string")
                expected_frame[column] = expected_frame[column].astype("string")
        try:
            pd.testing.assert_frame_equal(
                observed_frame,
                expected_frame,
                check_exact=True,
            )
        except AssertionError as exc:
            raise ValueError(
                "Reporting candidate does not match its independently recomputed source semantics."
            ) from exc
    validate_cohort_flow(candidates["cohort_flow"], config)
    validate_reporting_cross_artifact_identities(candidates, config)
    evidence = align_reporting_execution_inputs(inputs, config)
    points = candidates["calibration_points"]
    validate_calibration_points(
        points,
        expected_n=evidence.expected_test_n,
        expected_events=int(evidence.calibration["y_true"].sum()),
        config=config,
    )
    validate_calibration_smoother(candidates["calibration_smoother"], config)


def validate_reporting_cross_artifact_identities(
    candidates: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    """Validate exact reporting rows and count identities across all aggregate candidates."""
    summary = candidates["included_excluded_summary"]
    smd = candidates["included_excluded_smd"]
    flow = candidates["cohort_flow"]
    registries = config["row_order_and_schema_validation_registry"]
    _validate_registered_frame(summary, registries["included_excluded_summary"])
    _validate_registered_frame(smd, registries["included_excluded_smd"])

    summary_contract = config["outputs"]["scientific_and_closure"]["included_excluded_summary"]
    observed_summary = [
        (row["variable"], None if pd.isna(row["level"]) else row["level"], row["group"])
        for row in summary.to_dict(orient="records")
    ]
    expected_summary = [
        (row["variable"], row["level"], row["group"])
        for row in summary_contract["planned_record_tuples"]
    ]
    if observed_summary != expected_summary:
        raise ValueError("Included/excluded summary row identity or order is invalid.")
    flow_counts = dict(zip(flow["node_id"].astype(str), flow["n"].astype(int), strict=True))
    group_counts = {"included": flow_counts["modeling_cohort"], "excluded": 68}
    for row, planned in zip(summary.to_dict(orient="records"), summary_contract["planned_record_tuples"], strict=True):
        if int(row["n_total"]) != group_counts[str(row["group"])]:
            raise ValueError("Included/excluded summary does not reconcile to cohort flow.")
        if int(row["n_nonmissing"]) + int(row["n_missing"]) != int(row["n_total"]):
            raise ValueError("Included/excluded missingness counts do not reconcile.")
        kind = planned["statistics_kind"]
        if kind == "continuous":
            if not pd.isna(row["count"]) or not pd.isna(row["proportion"]):
                raise ValueError("Continuous summaries cannot populate count or proportion.")
        else:
            count = int(row["count"])
            nonmissing = int(row["n_nonmissing"])
            if not 0 <= count <= nonmissing or not np.isclose(
                float(row["proportion"]), count / nonmissing, rtol=0.0, atol=1e-15
            ):
                raise ValueError("Included/excluded count and proportion do not reconcile.")

    for group in ("included", "excluded"):
        category_rows = summary.loc[
            (summary["group"].astype(str) == group) & (summary["variable"] == "disease_category")
        ]
        if int(category_rows["count"].sum()) != int(category_rows.iloc[0]["n_nonmissing"]):
            raise ValueError("Disease-category one-versus-rest counts do not reconcile.")
        for variable in ("age", "sex", "endpoint", "disease_category"):
            main = summary.loc[
                (summary["group"].astype(str) == group) & (summary["variable"] == variable)
            ].iloc[0]
            missing = summary.loc[
                (summary["group"].astype(str) == group)
                & (summary["variable"] == f"missingness__{variable}")
            ].iloc[0]
            if int(missing["count"]) != int(main["n_missing"]):
                raise ValueError("Explicit missingness row does not reconcile to its source variable.")

    reason_rows = summary.loc[summary["variable"] == "qc_exclusion_reason"]
    if int(reason_rows["count"].sum()) != group_counts["excluded"]:
        raise ValueError("Exclusion-reason summary does not reconcile to the excluded cohort.")
    reason_flow = flow.loc[flow["stage"].astype(str) == "exclusion", "n"].astype(int).tolist()
    if reason_rows["count"].astype(int).tolist() != reason_flow:
        raise ValueError("Exclusion-reason summary and cohort flow disagree.")

    smd_contract = config["outputs"]["scientific_and_closure"]["included_excluded_smd"]
    observed_smd = [
        (row["variable"], None if pd.isna(row["level"]) else row["level"])
        for row in smd.to_dict(orient="records")
    ]
    expected_smd = [(row["variable"], row["level"]) for row in smd_contract["planned_record_tuples"]]
    if observed_smd != expected_smd:
        raise ValueError("Included/excluded SMD row identity or order is invalid.")
    allowed_status = {"estimable", "non_estimable"}
    for row, planned in zip(smd.to_dict(orient="records"), smd_contract["planned_record_tuples"], strict=True):
        status = str(row["estimability_status"])
        if status not in allowed_status:
            raise ValueError("Included/excluded SMD status is invalid.")
        value_missing = pd.isna(row["smd_included_minus_excluded"])
        reason_missing = pd.isna(row["nonestimable_reason"])
        if (status == "estimable" and (value_missing or not reason_missing)) or (
            status == "non_estimable"
            and (not value_missing or str(row["nonestimable_reason"]) not in {
                "smd_fewer_than_two_nonmissing",
                "smd_zero_denominator_nonidentical",
            })
        ):
            raise ValueError("Included/excluded SMD estimability fields are inconsistent.")
        applicable = planned.get("missingness_smd_applicable") is True
        if pd.isna(row["missingness_smd_included_minus_excluded"]) == applicable:
            raise ValueError("Included/excluded missingness-SMD population is invalid.")
        source_rows = summary.loc[summary["variable"] == row["variable"]]
        if row["variable"] == "disease_category":
            source_rows = source_rows.loc[source_rows["level"] == row["level"]]
        included_row = source_rows.loc[source_rows["group"].astype(str) == "included"].iloc[0]
        excluded_row = source_rows.loc[source_rows["group"].astype(str) == "excluded"].iloc[0]
        if int(row["n_included_nonmissing"]) != int(included_row["n_nonmissing"]) or int(
            row["n_excluded_nonmissing"]
        ) != int(excluded_row["n_nonmissing"]):
            raise ValueError("Included/excluded SMD support counts do not reconcile to summaries.")


def _prior_aggregate_inventory(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return exact Session 4/5 scientific aggregate paths in execution order."""
    grammar = config["outputs"]["completion_logs"]["machine_validated_log_grammar"]
    return tuple(
        [
            *grammar["session_bindings_exact"]["locked_scores"]["ordered_output_inventory"],
            *grammar["session_bindings_exact"]["date_grouped"]["ordered_output_inventory"],
        ]
    )


def validate_prior_aggregate_frames(
    frames: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
) -> PriorAggregateSemanticEvidence:
    """Validate exact Session 4/5 schemas, scientific semantics, and cross-output anchors."""
    expected_paths = _prior_aggregate_inventory(config)
    if tuple(frames) != expected_paths:
        raise ValueError("Session 4/5 aggregate inventory is incomplete or reordered.")
    registry = config["row_order_and_schema_validation_registry"]
    registry_by_path = {
        entry["path"]: entry
        for entry in registry.values()
        if isinstance(entry, Mapping) and "path" in entry and "column_order" in entry
    }
    for path, frame in frames.items():
        if path not in registry_by_path:
            raise ValueError("A prior aggregate has no frozen schema registry.")
        validate_csv_frame(frame, registry_by_path[path])
    outputs = config["outputs"]["scientific_and_closure"]
    session4_names = (
        "locked_score_metrics",
        "locked_score_composition",
        "qc5_correctness",
        "within_date_concordance",
    )
    session4_records: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for name in session4_names:
        path = outputs[name]["path"]
        records = frames[path].to_dict(orient="records")
        _validate_scientific_records(name, records, outputs[name])
        session4_records[name] = records
    _validate_cross_output_reconciliation(
        session4_records["locked_score_metrics"],
        session4_records["locked_score_composition"],
        session4_records["qc5_correctness"],
    )
    metrics = frames[outputs["locked_score_metrics"]["path"]]
    primary = metrics.loc[
        (metrics["analysis_id"].astype(str) == "frozen_primary_internal_test_lr")
        & (metrics["view"].astype(str) == "test")
        & (metrics["population_id"].astype(str) == "frozen_primary_internal_test")
    ]
    count_fields = ("n", "n_positive", "n_negative", "tp", "tn", "fp", "fn")
    if len(primary) != 2 or any(primary[field].nunique(dropna=False) != 1 for field in count_fields):
        raise ValueError("Frozen primary internal-test aggregate count anchors are inconsistent.")
    internal_test_n = int(primary.iloc[0]["n"])
    internal_test_positive = int(primary.iloc[0]["n_positive"])
    internal_test_negative = int(primary.iloc[0]["n_negative"])
    if (
        internal_test_n != int(config["reporting_closure"]["cohort_flow"]["internal_test_n"])
        or internal_test_n != internal_test_positive + internal_test_negative
    ):
        raise ValueError("Frozen primary internal-test aggregate anchors do not match cohort authority.")

    validation_path = outputs["date_grouped_validation"]["path"]
    fold_path = outputs["date_grouped_fold_audit"]["path"]
    expected_train_n = int(config["reporting_closure"]["cohort_flow"]["train_n"])
    validation = frames[validation_path]
    folds = frames[fold_path]
    validate_analysis3_outputs(validation, folds, config, expected_n=expected_train_n)
    validation_row = validation.iloc[0]
    train_positive = int(validation_row["n_positive"])
    train_negative = int(validation_row["n_negative"])
    n_date_groups = int(validation_row["n_date_groups"])
    for row in folds.to_dict(orient="records"):
        if (
            int(row["n_fit"]) + int(row["n_validation"]) != expected_train_n
            or int(row["n_fit_positive"]) + int(row["n_fit_negative"]) != int(row["n_fit"])
            or int(row["n_validation_positive"]) + int(row["n_validation_negative"])
            != int(row["n_validation"])
            or int(row["n_fit_positive"]) + int(row["n_validation_positive"]) != train_positive
            or int(row["n_fit_negative"]) + int(row["n_validation_negative"]) != train_negative
            or int(row["n_fit_date_groups"]) + int(row["n_validation_date_groups"]) != n_date_groups
        ):
            raise ValueError("Analysis 3 fold fit/validation totals do not reconcile.")
    return PriorAggregateSemanticEvidence(
        internal_test_n=internal_test_n,
        internal_test_n_positive=internal_test_positive,
        internal_test_n_negative=internal_test_negative,
        date_grouped_train_n=expected_train_n,
        date_grouped_train_n_positive=train_positive,
        date_grouped_train_n_negative=train_negative,
    )


def collect_nonestimability_records(
    artifacts: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[OrderedDict[str, Any]]:
    """Collect retained non-estimable states from validated aggregate frames.

    Args:
        artifacts: Ordered project-relative artifact paths and in-memory values.
        config: Frozen post-hoc configuration.

    Returns:
        Deterministic manifest records, deduplicated by artifact and reason.

    Raises:
        ValueError: If a non-estimable row lacks a declared reason code.
    """
    allowed_reasons = set(config["status_and_reason_registries"]["scientific_result_reason_registry"])
    records: list[OrderedDict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for path, value in artifacts.items():
        if not isinstance(value, pd.DataFrame):
            continue
        status_columns = [column for column in value.columns if str(column).endswith("_status")]
        for status_column in status_columns:
            nonestimable = value[status_column].astype("string") == "non_estimable"
            if not nonestimable.any():
                continue
            if status_column == "estimability_status":
                reason_column = "nonestimable_reason"
            elif status_column == "ci_status":
                reason_column = "ci_nonestimable_reason"
            else:
                reason_column = f"{status_column.removesuffix('_status')}_nonestimable_reason"
            if reason_column not in value.columns or value.loc[nonestimable, reason_column].isna().any():
                raise ValueError("A retained non-estimable result is missing its declared reason code.")
            for reason in value.loc[nonestimable, reason_column].astype(str).tolist():
                if reason not in allowed_reasons:
                    raise ValueError("A retained non-estimable result uses an undeclared reason code.")
                identity = (str(path), reason)
                if identity in seen:
                    continue
                seen.add(identity)
                records.append(
                    OrderedDict(
                        [
                            ("artifact_path", str(path)),
                            ("status", "non_estimable_retained"),
                            ("reason_code", reason),
                            ("note", "Retained under the frozen estimability contract."),
                        ]
                    )
                )
    return records


def _markdown_cell(value: Any) -> str:
    """Render one aggregate value deterministically for the consolidated report."""
    if value is None or value is pd.NA or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render every validated aggregate row without a presentation dependency."""
    columns = frame.columns.tolist()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(_markdown_cell(row[column]) for column in columns) + " |"
        for row in frame.to_dict(orient="records")
    ]
    return "\n".join([header, separator, *rows])


def build_consolidated_report(
    prior_aggregates: Mapping[str, pd.DataFrame],
    scientific_candidates: Mapping[str, pd.DataFrame],
    fact_audit: Mapping[str, Any],
    qc6_note: str,
    config: Mapping[str, Any],
) -> str:
    """Render deterministic Session 6 Markdown from validated machine evidence only."""
    validate_prior_aggregate_frames(prior_aggregates, config)
    validate_reporting_cross_artifact_identities(scientific_candidates, config)
    validate_reference_standard_timing_audit(fact_audit, config)
    scientific_paths = config["outputs"]["scientific_and_closure"]
    audit_path = scientific_paths["integrity_audit"]["path"]
    release_path = scientific_paths["visualization_release"]["path"]
    sections = [
        "# Final post-hoc supplementary analyses",
        "",
        "This post-hoc supplement reports internal same-center, same-device, same-period random-holdout evidence. "
        "It is not external or temporal validation, does not establish deployment readiness or clinical benefit, "
        "and does not support causal interpretation.",
        "",
        "Breath collection followed diagnostic gastroscopy and indicated biopsy and preceded treatment; direct "
        "pre-endoscopy-triage validation remains unavailable. Differential reference standards are retained as a "
        "limitation.",
        "",
        f"Session 7 integrity audit is pending; its authoritative status path is `{audit_path}`.",
        "",
        "Visualization release remains unauthorized and pending independent approval; its authoritative status path is "
        f"`{release_path}`.",
    ]
    for path, frame in [*prior_aggregates.items(), *scientific_candidates.items()]:
        sections.extend(["", f"## `{path}`", "", _markdown_table(frame)])
    sections.extend(["", "## Reference-standard and timing facts", ""])
    for fact in fact_audit["facts"]:
        sections.append(
            f"- {fact['fact_id']} [{fact['evidence_status']}]: {fact['conservative_statement_or_nonclaim']}"
        )
    sections.extend(["", "## QC6 reporting-lineage correction", "", qc6_note.rstrip("\n")])
    return "\n".join(sections) + "\n"


def build_reporting_closure_artifacts(
    authority: Mapping[str, Any],
    scientific_candidates: Mapping[str, pd.DataFrame],
    prior_aggregates: Mapping[str, pd.DataFrame],
    *,
    generated_at_utc: str,
    access_state: Mapping[str, Any],
) -> OrderedDict[str, Any]:
    """Construct the eight deterministic Session 6 artifacts that precede the manifest."""
    config = authority["config"]
    validate_access_state(access_state)
    validate_prior_aggregate_frames(prior_aggregates, config)
    validate_reporting_cross_artifact_identities(scientific_candidates, config)
    fact_audit = build_reference_standard_timing_audit(
        config,
        generated_at_utc=generated_at_utc,
        access_state=access_state,
    )
    qc6_note = generate_qc6_lineage_correction(config, correction_date_utc=generated_at_utc)
    report = build_consolidated_report(
        prior_aggregates,
        scientific_candidates,
        fact_audit,
        qc6_note,
        config,
    )
    scientific_paths = config["outputs"]["scientific_and_closure"]
    values_by_path: dict[str, Any] = {
        scientific_paths["cohort_flow"]["path"]: scientific_candidates["cohort_flow"],
        scientific_paths["included_excluded_summary"]["path"]: scientific_candidates[
            "included_excluded_summary"
        ],
        scientific_paths["included_excluded_smd"]["path"]: scientific_candidates["included_excluded_smd"],
        scientific_paths["calibration_points"]["path"]: scientific_candidates["calibration_points"],
        scientific_paths["calibration_smoother"]["path"]: scientific_candidates["calibration_smoother"],
        scientific_paths["reference_standard_timing_audit"]["path"]: fact_audit,
        scientific_paths["qc6_lineage_correction"]["path"]: qc6_note,
        scientific_paths["consolidated_analyses_document"]["path"]: report,
    }
    inventory, _ = reporting_promotion_inventory(config)
    manifest_path = scientific_paths["execution_manifest"]["path"]
    expected = tuple(path for path in inventory if path != manifest_path)
    artifacts = OrderedDict((path, values_by_path[path]) for path in expected)
    if tuple(artifacts) != expected:
        raise ValueError("Pre-manifest reporting artifact inventory is invalid.")
    return artifacts


def serialize_reporting_closure_artifacts(
    artifacts: Mapping[str, Any],
    config: Mapping[str, Any],
) -> OrderedDict[str, bytes]:
    """Serialize all eight pre-manifest artifacts deterministically."""
    inventory, _ = reporting_promotion_inventory(config)
    manifest_path = config["outputs"]["scientific_and_closure"]["execution_manifest"]["path"]
    expected = tuple(path for path in inventory if path != manifest_path)
    if tuple(artifacts) != expected:
        raise ValueError("Pre-manifest reporting artifact inventory is incomplete or reordered.")
    serialized: OrderedDict[str, bytes] = OrderedDict()
    for path, value in artifacts.items():
        if isinstance(value, pd.DataFrame):
            payload = value.to_csv(index=False, lineterminator="\n").encode("utf-8")
        elif isinstance(value, Mapping):
            payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        elif isinstance(value, str):
            payload = value.encode("utf-8")
        else:
            raise TypeError("Unsupported reporting artifact representation.")
        serialized[path] = payload
    return serialized


def serialize_json_artifact(payload: Mapping[str, Any]) -> bytes:
    """Serialize one ordered JSON artifact with stable UTF-8 formatting."""
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def build_reporting_execution_manifest(
    authority: Mapping[str, Any],
    artifact_bytes: Mapping[str, bytes],
    entry_hashes: Mapping[str, str],
    *,
    command: str,
    generated_at_utc: str,
    access_state: Mapping[str, Any],
    deviations_or_nonestimability: Sequence[Mapping[str, Any]],
) -> OrderedDict[str, Any]:
    """Build the Session 6 manifest from agreed candidate and entry bytes."""
    config = authority["config"]
    if not isinstance(command, str) or not command:
        raise ValueError("Execution-manifest command must be the nonempty actual invocation.")
    inventories = config["hash_map_inventories"]
    input_paths = tuple(inventories["execution_manifest_input_sha256_keys_exact_order"])
    code_paths = tuple(inventories["execution_manifest_code_sha256_keys_exact_order"])
    output_paths = tuple(inventories["execution_manifest_output_sha256_keys_exact_order"])
    if any(path not in entry_hashes for path in (*input_paths, *code_paths)):
        raise ValueError("Execution-manifest entry snapshot is incomplete.")
    if tuple(artifact_bytes) != output_paths:
        raise ValueError("Execution-manifest output candidate inventory is incomplete or reordered.")
    runtime = config["runtime_controls"]
    environment = OrderedDict(
        [
            ("conda_environment_name", runtime["python_environment"]),
            ("python_version", runtime["python_version"]),
            (
                "package_versions",
                OrderedDict(
                    (name, runtime["required_package_versions"][name])
                    for name in ("scikit-learn", "statsmodels", "scipy", "numpy", "pandas", "sktime")
                ),
            ),
            (
                "runtime_controls",
                OrderedDict(
                    [
                        ("OMP_NUM_THREADS", runtime["OMP_NUM_THREADS"]),
                        ("OPENBLAS_NUM_THREADS", runtime["OPENBLAS_NUM_THREADS"]),
                        ("MKL_NUM_THREADS", runtime["MKL_NUM_THREADS"]),
                        ("NUMEXPR_NUM_THREADS", runtime["NUMEXPR_NUM_THREADS"]),
                        (
                            "analysis_3_sklearn_n_jobs",
                            runtime["job_count_controls"]["analysis_3_sklearn_n_jobs"],
                        ),
                    ]
                ),
            ),
        ]
    )
    session_0 = config["outputs"]["session_0"]
    payload: OrderedDict[str, Any] = OrderedDict(
        [
            ("schema_version", "posthoc_final_execution_manifest.v1"),
            ("package_id", "posthoc_final_supplementary"),
            ("generated_at_utc", generated_at_utc),
            ("freeze_commit", authority["session0_freeze"]["freeze_commit_id"]),
            ("commands", [command]),
            ("environment", environment),
            ("input_sha256", OrderedDict((path, entry_hashes[path]) for path in input_paths)),
            (
                "output_sha256",
                OrderedDict((path, hashlib.sha256(artifact_bytes[path]).hexdigest()) for path in output_paths),
            ),
            ("code_sha256", OrderedDict((path, entry_hashes[path]) for path in code_paths)),
            ("config_sha256", entry_hashes[session_0["config"]]),
            ("methods_sha256", entry_hashes[session_0["methods"]]),
            (
                "deterministic_comparison_provenance",
                OrderedDict(
                    [
                        ("candidate_pair_count", 2),
                        ("scientific_payload_agreement", True),
                        ("shared_invocation_utc", generated_at_utc),
                    ]
                ),
            ),
            ("access_state", OrderedDict(access_state)),
            (
                "deviations_or_nonestimability",
                [OrderedDict(item) for item in deviations_or_nonestimability],
            ),
            ("self_hash_policy", "forbidden"),
        ]
    )
    validate_reporting_execution_manifest_binding(
        payload,
        artifact_bytes,
        entry_hashes,
        authority,
        expected_deviations=deviations_or_nonestimability,
    )
    return payload


def validate_reporting_execution_manifest_binding(
    payload: Mapping[str, Any],
    artifact_bytes: Mapping[str, bytes],
    entry_hashes: Mapping[str, str],
    authority: Mapping[str, Any],
    *,
    expected_deviations: Sequence[Mapping[str, Any]],
) -> None:
    """Reconcile every manifest hash with its actual source or candidate bytes."""
    config = authority["config"]
    validate_execution_manifest(payload, config)
    inventories = config["hash_map_inventories"]
    expected_inputs = OrderedDict(
        (path, entry_hashes[path]) for path in inventories["execution_manifest_input_sha256_keys_exact_order"]
    )
    expected_codes = OrderedDict(
        (path, entry_hashes[path]) for path in inventories["execution_manifest_code_sha256_keys_exact_order"]
    )
    expected_outputs = OrderedDict(
        (path, hashlib.sha256(artifact_bytes[path]).hexdigest())
        for path in inventories["execution_manifest_output_sha256_keys_exact_order"]
    )
    if payload["input_sha256"] != expected_inputs:
        raise ValueError("Execution-manifest input hash binding is invalid.")
    if payload["code_sha256"] != expected_codes:
        raise ValueError("Execution-manifest code hash binding is invalid.")
    if payload["output_sha256"] != expected_outputs:
        raise ValueError("Execution-manifest output hash binding is invalid.")
    session_0 = config["outputs"]["session_0"]
    if payload["config_sha256"] != entry_hashes[session_0["config"]] or payload[
        "methods_sha256"
    ] != entry_hashes[session_0["methods"]]:
        raise ValueError("Execution-manifest authority hash binding is invalid.")
    if payload["access_state"] != access_state_ordered(access_state=payload["access_state"]):
        raise ValueError("Execution-manifest access-state binding is invalid.")
    if payload["deviations_or_nonestimability"] != [
        OrderedDict(item) for item in expected_deviations
    ]:
        raise ValueError("Execution-manifest retained-status binding is invalid.")


def access_state_ordered(*, access_state: Mapping[str, Any]) -> OrderedDict[str, Any]:
    """Return an exactly validated ordered access-state copy."""
    validate_access_state(access_state)
    return OrderedDict(
        [
            ("session_activity", OrderedDict(access_state["session_activity"])),
            ("cumulative_through_session", OrderedDict(access_state["cumulative_through_session"])),
        ]
    )


def require_byte_identical_candidate_sets(
    first: Mapping[str, bytes],
    second: Mapping[str, bytes],
) -> None:
    """Require exact ordered, byte-for-byte agreement of two isolated candidates."""
    if tuple(first) != tuple(second) or any(
        not isinstance(first[path], bytes)
        or not isinstance(second[path], bytes)
        or first[path] != second[path]
        for path in first
    ):
        raise ValueError("Isolated reporting candidate bytes disagree.")


def _copy_reporting_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Create an isolated deep value copy for the second deterministic calculation."""
    copied: dict[str, Any] = {}
    for key, value in inputs.items():
        if isinstance(value, pd.DataFrame):
            copied[key] = value.copy(deep=True)
        elif isinstance(value, np.ndarray):
            copied[key] = value.copy()
        else:
            copied[key] = value
    return copied


def _copy_aggregate_frames(frames: Mapping[str, pd.DataFrame]) -> OrderedDict[str, pd.DataFrame]:
    """Create an isolated deep copy of an ordered aggregate-frame inventory."""
    return OrderedDict((path, frame.copy(deep=True)) for path, frame in frames.items())


def _reporting_privacy_validate(
    *,
    path: str,
    value: Any,
    serialized: bytes,
    config: Mapping[str, Any],
    privacy_references: PrivacyReferences,
) -> None:
    """Validate one reporting artifact in both object and serialized form."""
    if isinstance(value, pd.DataFrame):
        kind = "scientific_csv"
    elif isinstance(value, Mapping):
        kind = "structured_json"
    else:
        kind = "markdown_report"
    validate_artifact_privacy(
        value,
        artifact_kind=kind,
        config=config,
        privacy_references=privacy_references,
        artifact_path=path,
    )
    validate_artifact_privacy(
        serialized,
        artifact_kind=kind,
        config=config,
        privacy_references=privacy_references,
        artifact_path=path,
    )


def _validate_staged_reporting_bundle(
    *,
    staged: Mapping[str, Path],
    artifacts: Mapping[str, Any],
    nine_bytes: Mapping[str, bytes],
    manifest: Mapping[str, Any],
    log_text: str,
    authority: Mapping[str, Any],
    inputs: Mapping[str, Any],
    entry_hashes: Mapping[str, str],
    access_state: Mapping[str, Any],
    privacy_references: PrivacyReferences,
    deviations_or_nonestimability: Sequence[Mapping[str, Any]],
) -> None:
    """Read back every staged Session 6 artifact and repeat all applicable checks."""
    config = authority["config"]
    if tuple(staged) != tuple(nine_bytes):
        raise ValueError("Staged reporting inventory is incomplete or reordered.")
    for path, staged_path in staged.items():
        if staged_path.read_bytes() != nine_bytes[path]:
            raise ValueError("A staged reporting candidate changed during readback.")
    scientific_paths = config["outputs"]["scientific_and_closure"]
    name_by_path = {
        scientific_paths["cohort_flow"]["path"]: "cohort_flow",
        scientific_paths["included_excluded_summary"]["path"]: "included_excluded_summary",
        scientific_paths["included_excluded_smd"]["path"]: "included_excluded_smd",
        scientific_paths["calibration_points"]["path"]: "calibration_points",
        scientific_paths["calibration_smoother"]["path"]: "calibration_smoother",
    }
    registry_by_path = {
        entry["path"]: entry
        for entry in config["row_order_and_schema_validation_registry"].values()
        if isinstance(entry, Mapping) and "path" in entry and "column_order" in entry
    }
    readback_frames = OrderedDict(
        (name_by_path[path], read_typed_registry_csv(staged[path], registry_by_path[path]))
        for path in name_by_path
    )
    validate_reporting_candidates(readback_frames, authority, inputs)
    fact_path = scientific_paths["reference_standard_timing_audit"]["path"]
    fact_payload = json.loads(staged[fact_path].read_text(encoding="utf-8"), object_pairs_hook=OrderedDict)
    validate_reference_standard_timing_audit(fact_payload, config)
    if fact_payload != artifacts[fact_path]:
        raise ValueError("Staged reference-standard audit changed during JSON readback.")
    qc6_path = scientific_paths["qc6_lineage_correction"]["path"]
    generated_at = str(fact_payload["generated_at_utc"])
    validate_qc6_lineage_correction(
        staged[qc6_path].read_text(encoding="utf-8"),
        config,
        correction_date_utc=generated_at,
    )
    report_path = scientific_paths["consolidated_analyses_document"]["path"]
    if staged[report_path].read_text(encoding="utf-8") != artifacts[report_path]:
        raise ValueError("Staged consolidated report changed during readback.")
    manifest_path = scientific_paths["execution_manifest"]["path"]
    manifest_readback = json.loads(
        staged[manifest_path].read_text(encoding="utf-8"), object_pairs_hook=OrderedDict
    )
    if manifest_readback != manifest:
        raise ValueError("Staged execution manifest changed during JSON readback.")
    premanifest_bytes = OrderedDict(
        (path, nine_bytes[path])
        for path in config["hash_map_inventories"]["execution_manifest_output_sha256_keys_exact_order"]
    )
    validate_reporting_execution_manifest_binding(
        manifest_readback,
        premanifest_bytes,
        entry_hashes,
        authority,
        expected_deviations=deviations_or_nonestimability,
    )
    for path, value in artifacts.items():
        _reporting_privacy_validate(
            path=path,
            value=value,
            serialized=nine_bytes[path],
            config=config,
            privacy_references=privacy_references,
        )
    validate_artifact_privacy(
        manifest_readback,
        artifact_kind="governance_manifest",
        config=config,
        privacy_references=privacy_references,
        artifact_path=manifest_path,
    )
    validate_artifact_privacy(
        nine_bytes[manifest_path],
        artifact_kind="governance_manifest",
        config=config,
        privacy_references=privacy_references,
        artifact_path=manifest_path,
    )
    validate_completion_log_against_bytes(
        log_text,
        session_name="reporting",
        freeze_commit=authority["session0_freeze"]["freeze_commit_id"],
        output_bytes=nine_bytes,
        access_state=access_state,
        cumulative_prior=authority["cumulative_prior"],
        config=config,
    )
    log_path = config["outputs"]["completion_logs"]["reporting"]
    for log_representation in (log_text, log_text.encode("utf-8")):
        validate_artifact_privacy(
            log_representation,
            artifact_kind="sanitized_log",
            config=config,
            privacy_references=privacy_references,
            artifact_path=log_path,
        )


def _validate_final_git_state(
    *,
    git_runner: Callable[..., str],
    allowed_staged_paths: Sequence[Path],
    project_root: Path,
    required_paths: Sequence[str],
) -> None:
    """Recheck HEAD identity while allowing only the known destination-local temp files."""
    allowed = {
        str(path.resolve().relative_to(project_root.resolve()))
        for path in allowed_staged_paths
    }
    status = git_runner("status", "--porcelain", "--untracked-files=all")
    unexpected: list[str] = []
    for line in status.splitlines():
        path_text = line[3:] if len(line) >= 4 else ""
        if " -> " in path_text:
            path_text = path_text.split(" -> ", 1)[1]
        if path_text not in allowed:
            unexpected.append(path_text)
    if unexpected:
        raise ValueError("Final reporting Git state contains an unexpected path.")
    for path in required_paths:
        git_runner("ls-files", "--error-unmatch", path)
        git_runner("cat-file", "-e", f"HEAD:{path}")
        if git_runner("rev-parse", f"HEAD:{path}") != git_runner("hash-object", path):
            raise ValueError("A committed reporting prerequisite changed before promotion.")


def prepare_reporting_promotion(
    *,
    authority: Mapping[str, Any],
    inputs: Mapping[str, Any],
    prior_aggregates: Mapping[str, pd.DataFrame],
    privacy_references: PrivacyReferences,
    generated_at_utc: str,
    command: str,
    git_runner: Callable[..., str],
) -> ReportingPromotionBundle:
    """Calculate twice, validate, serialize, stage, read back, and freeze Session 6 candidates."""
    config = authority["config"]
    access_state = build_reporting_access_state(authority["cumulative_prior"])
    first_prior = _copy_aggregate_frames(prior_aggregates)
    first_scientific = calculate_reporting_candidates(authority, _copy_reporting_inputs(inputs))
    validate_reporting_candidates(first_scientific, authority, inputs)
    first_artifacts = build_reporting_closure_artifacts(
        authority,
        first_scientific,
        first_prior,
        generated_at_utc=generated_at_utc,
        access_state=access_state,
    )
    first_bytes = serialize_reporting_closure_artifacts(first_artifacts, config)
    second_inputs = _copy_reporting_inputs(inputs)
    second_prior = _copy_aggregate_frames(prior_aggregates)
    second_scientific = calculate_reporting_candidates(authority, second_inputs)
    validate_reporting_candidates(second_scientific, authority, second_inputs)
    second_artifacts = build_reporting_closure_artifacts(
        authority,
        second_scientific,
        second_prior,
        generated_at_utc=generated_at_utc,
        access_state=access_state,
    )
    second_bytes = serialize_reporting_closure_artifacts(second_artifacts, config)
    require_byte_identical_candidate_sets(first_bytes, second_bytes)
    for path, value in first_artifacts.items():
        _reporting_privacy_validate(
            path=path,
            value=value,
            serialized=first_bytes[path],
            config=config,
            privacy_references=privacy_references,
        )
    deviations_or_nonestimability = collect_nonestimability_records(
        OrderedDict([*prior_aggregates.items(), *first_artifacts.items()]),
        config,
    )
    manifest = build_reporting_execution_manifest(
        authority,
        first_bytes,
        authority["entry_hashes"],
        command=command,
        generated_at_utc=generated_at_utc,
        access_state=access_state,
        deviations_or_nonestimability=deviations_or_nonestimability,
    )
    manifest_path = config["outputs"]["scientific_and_closure"]["execution_manifest"]["path"]
    manifest_bytes = serialize_json_artifact(manifest)
    inventory, log_relative = reporting_promotion_inventory(config)
    nine_bytes: OrderedDict[str, bytes] = OrderedDict(
        (path, manifest_bytes if path == manifest_path else first_bytes[path]) for path in inventory
    )
    log_text = build_session_completion_log(
        session_name="reporting",
        freeze_commit=authority["session0_freeze"]["freeze_commit_id"],
        output_bytes=nine_bytes,
        access_state=access_state,
        cumulative_prior=authority["cumulative_prior"],
        generated_at_utc=generated_at_utc,
        config=config,
        reason_codes=sorted(
            {
                str(item["reason_code"])
                for item in deviations_or_nonestimability
                if item["reason_code"] is not None
            }
        ),
    )
    output_validators = OrderedDict(
        (
            path,
            lambda staged_path, expected=nine_bytes[path]: (
                None
                if staged_path.read_bytes() == expected
                else (_ for _ in ()).throw(ValueError("Staged reporting bytes changed."))
            ),
        )
        for path in inventory
    )
    log_bytes = log_text.encode("utf-8")

    def log_validator(path: Path) -> None:
        """Revalidate the staged reporting receipt against exact candidate bytes."""
        staged_text = path.read_text(encoding="utf-8")
        validate_completion_log_against_bytes(
            staged_text,
            session_name="reporting",
            freeze_commit=authority["session0_freeze"]["freeze_commit_id"],
            output_bytes=nine_bytes,
            access_state=access_state,
            cumulative_prior=authority["cumulative_prior"],
            config=config,
        )
        for log_representation in (staged_text, path.read_bytes()):
            validate_artifact_privacy(
                log_representation,
                artifact_kind="sanitized_log",
                config=config,
                privacy_references=privacy_references,
                artifact_path=log_relative,
            )

    candidates, log_candidate, log_target = stage_candidate_bytes(
        project_root=authority["project_root"],
        output_bytes=nine_bytes,
        output_validators=output_validators,
        log_relative_path=log_relative,
        log_bytes=log_bytes,
        log_validator=log_validator,
    )
    staged = OrderedDict((candidate.relative_path, candidate.candidate_path) for candidate in candidates)
    try:
        _validate_staged_reporting_bundle(
            staged=staged,
            artifacts=first_artifacts,
            nine_bytes=nine_bytes,
            manifest=manifest,
            log_text=log_text,
            authority=authority,
            inputs=inputs,
            entry_hashes=authority["entry_hashes"],
            access_state=access_state,
            privacy_references=privacy_references,
            deviations_or_nonestimability=deviations_or_nonestimability,
        )
        verify_integrity_snapshot(authority["project_root"], authority["entry_hashes"])
        _validate_final_git_state(
            git_runner=git_runner,
            allowed_staged_paths=[*staged.values(), log_candidate],
            project_root=authority["project_root"],
            required_paths=authority["required_paths"],
        )
        relative_targets, relative_log = reporting_promotion_inventory(config)
        if _classify_paths(
            [authority["project_root"] / path for path in relative_targets],
            authority["project_root"] / relative_log,
        ) != "clean":
            raise ValueError("Session 6 target state changed before promotion.")
    except Exception:
        for path in [*staged.values(), log_candidate]:
            path.unlink(missing_ok=True)
        raise
    return ReportingPromotionBundle(candidates, log_candidate, log_target, log_validator)


def promote_reporting_candidates(
    bundle: ReportingPromotionBundle,
    authority: Mapping[str, Any],
) -> tuple[Path, ...]:
    """Promote nine exact-target outputs atomically in order and the reporting log last."""
    if not isinstance(bundle, ReportingPromotionBundle):
        raise TypeError("Reporting promotion requires a validated ReportingPromotionBundle.")
    output_paths, _ = reporting_promotion_inventory(authority["config"])
    promote_candidates(
        bundle.candidates,
        expected_relative_paths=output_paths,
        log_candidate=bundle.log_candidate,
        log_target=bundle.log_target,
        log_validator=bundle.log_validator,
        project_root=authority["project_root"],
    )
    return tuple([*(candidate.target_path for candidate in bundle.candidates), bundle.log_target])


def execute_approved(
    config_path: Path,
    freeze_commit: str,
    amendment_freeze_commit: str,
    *,
    project_root: Path | None = None,
    git_runner: Callable[..., str] = _git,
    invocation_clock: Callable[[], str] | None = None,
    command: str | None = None,
) -> tuple[Path, ...]:
    """Run authority, access, dual calculation, staging, validation, and ordered promotion."""
    root = (project_root or PROJECT_ROOT).resolve()
    authority = validate_reporting_authority(
        config_path,
        freeze_commit,
        amendment_freeze_commit,
        git_runner=git_runner,
        project_root=root,
    )
    config = authority["config"]
    privacy_source = root / config["privacy"]["privacy_reference_loader"]["source"]
    privacy_references = load_privacy_references(privacy_source, config, authorized=True)
    inputs = load_reporting_execution_inputs(authority)
    prior_aggregates = load_validated_prior_aggregates(authority)
    timestamp = (invocation_clock or (lambda: datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")))()
    if not is_utc_iso8601_z(timestamp):
        raise ValueError("Reporting invocation clock did not return an exact UTC timestamp.")
    invocation = command or (
        "python scripts/run_posthoc_final_reporting.py "
        f"--config {config_path} --freeze-commit {freeze_commit} "
        f"--amendment-freeze-commit {amendment_freeze_commit} --execute-approved"
    )
    bundle = prepare_reporting_promotion(
        authority=authority,
        inputs=inputs,
        prior_aggregates=prior_aggregates,
        privacy_references=privacy_references,
        generated_at_utc=timestamp,
        command=invocation,
        git_runner=git_runner,
    )
    return promote_reporting_candidates(bundle, authority)
