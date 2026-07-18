"""Shared, privacy-safe helpers for the final post-hoc package."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import tempfile
import warnings
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.proportion import confint_proportions_2indep, proportion_confint

from src.config import PROJECT_ROOT
from src.evaluation.statistical_tests import DEGENERATE_VARIANCE_EPSILON, _fast_delong


LOGGER = logging.getLogger(__name__)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_ASCII_SLASH_DATE_PATTERN = re.compile(r"^([0-9]{4})/([0-9]{1,2})/([0-9]{1,2})$")
_UTC_ISO8601_Z_PATTERN = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
# Nullable pandas dtypes used to read scientific CSV outputs back for validation.
_REGISTRY_PANDAS_DTYPES = {
    "string": "string",
    "category": "string",
    "Int64": "Int64",
    "Float64": "Float64",
    "boolean": "boolean",
}


class AccessStateError(ValueError):
    """Sanitized failure that carries an access-state snapshot without private values.

    Args:
        message: Aggregate-safe failure message with no participant value.
        access_state: Ordered activity/cumulative ledger snapshot at failure time.
    """

    def __init__(self, message: str, access_state: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.access_state = access_state


def is_utc_iso8601_z(value: object) -> bool:
    """Return whether a value is an exact ``YYYY-MM-DDTHH:MM:SSZ`` UTC timestamp.

    Args:
        value: Candidate provenance timestamp.

    Returns:
        ``True`` only for the exact allowlisted UTC ISO-8601 Z grammar.
    """
    if not isinstance(value, str) or _UTC_ISO8601_Z_PATTERN.fullmatch(value) is None:
        return False
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return False
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ") == value
_PREFLIGHT_KEYS = (
    "schema_version",
    "package_id",
    "generated_at_utc",
    "hashes",
    "freeze_commit_validation",
    "aggregate_counts",
    "one_to_one_alignment_status",
    "train_test_disjointness_status",
    "known_anchor_reconciliation",
    "status",
    "reason_code",
    "access_state",
)
_FREEZE_KEYS = (
    "freeze_commit_id",
    "exists",
    "is_ancestor_of_head",
    "parent_is_baseline_head",
    "changed_paths_exact",
    "config_blob_matches",
    "methods_blob_matches",
    "manifest_blob_matches",
    "approval_evidence_matches",
    "active_rule_hashes_match",
)
_ACCESS_KEYS = (
    "prediction_container_bytes_hashed",
    "prediction_schema_or_header_inspected",
    "participant_level_label_values_read",
    "participant_level_score_values_read",
    "participant_level_date_values_read",
    "participant_level_qc_values_read",
    "new_score_derived_results_computed",
    "auxiliary_model_fit_executed",
    "privacy_reference_loader_executed",
)
ACCESS_STATE_KEYS = _ACCESS_KEYS


def sha256_bytes(payload: bytes) -> str:
    """Return the SHA-256 digest of opaque bytes.

    Args:
        payload: Bytes to hash without decoding.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file's complete byte container.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_exact_binary_labels_01(labels: np.ndarray) -> np.ndarray:
    """Validate a finite one-dimensional exact {0, 1} label vector.

    Args:
        labels: Candidate binary outcome labels.

    Returns:
        The validated integer label vector.

    Raises:
        TypeError: If values cannot be checked as numeric labels.
        ValueError: If labels are empty, nonfinite, nonbinary, or multidimensional.
    """
    values = np.asarray(labels)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Labels must be a non-empty one-dimensional vector.")
    try:
        numeric = values.astype(float)
    except (TypeError, ValueError) as exc:
        raise TypeError("Labels must be numeric exact binary values.") from exc
    if not np.all(np.isfinite(numeric)):
        raise ValueError("Labels must be finite.")
    if not np.all((numeric == 0.0) | (numeric == 1.0)):
        raise ValueError("Labels must contain exact values 0 or 1 only.")
    return numeric.astype(int)


def validate_aligned_finite_1d_vectors(**vectors: np.ndarray) -> dict[str, np.ndarray]:
    """Validate aligned non-empty finite one-dimensional numeric vectors.

    Args:
        **vectors: Named vectors to validate.

    Returns:
        Named arrays after validation.

    Raises:
        ValueError: If a vector is malformed, nonfinite, empty, or unaligned.
    """
    if not vectors:
        raise ValueError("At least one vector is required.")
    result: dict[str, np.ndarray] = {}
    lengths: set[int] = set()
    for name, value in vectors.items():
        array = np.asarray(value)
        if array.ndim != 1 or array.size == 0:
            raise ValueError("Vectors must be non-empty and one-dimensional.")
        try:
            numeric = array.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Vectors must be numeric.") from exc
        if not np.all(np.isfinite(numeric)):
            raise ValueError("Vectors must contain finite values.")
        result[name] = numeric
        lengths.add(array.size)
    if len(lengths) != 1:
        raise ValueError("Vectors must have equal length.")
    return result


def parse_date_only_ascii_slash(value: object) -> pd.Timestamp:
    """Parse one exact ASCII slash date without exposing the supplied date.

    Args:
        value: Candidate date-only text.

    Returns:
        A midnight-normalized timestamp.

    Raises:
        ValueError: If the value is missing or not exact ``YYYY/M/D`` text.
    """
    if not isinstance(value, str):
        raise ValueError("Date parsing failed: missing_or_nonstring_count=1.")
    match = _ASCII_SLASH_DATE_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError("Date parsing failed: invalid_date_format_count=1.")
    year, month, day = (int(token) for token in match.groups())
    try:
        parsed = pd.to_datetime(
            f"{year:04d}-{month:02d}-{day:02d}",
            format="%Y-%m-%d",
            exact=True,
            errors="raise",
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Date parsing failed: invalid_date_count=1.") from exc
    parsed = pd.Timestamp(parsed)
    if (
        parsed != parsed.normalize()
        or (parsed.year, parsed.month, parsed.day) != (year, month, day)
    ):
        raise ValueError("Date parsing failed: invalid_date_count=1.")
    return parsed


def parse_date_series_ascii_slash(values: Sequence[object]) -> np.ndarray:
    """Parse slash dates while reporting aggregate-safe failures only.

    Args:
        values: Candidate date-only values retained only in memory.

    Returns:
        Parsed midnight-normalized timestamps in input order.

    Raises:
        ValueError: If any value is malformed, invalid, or maps ambiguously.
    """
    missing_or_nonstring_count = 0
    invalid_date_format_count = 0
    invalid_date_count = 0
    parsed_values: list[pd.Timestamp] = []
    raw_to_normalized: dict[str, pd.Timestamp] = {}
    normalized_to_raw: dict[pd.Timestamp, str] = {}
    for value in values:
        if not isinstance(value, str):
            missing_or_nonstring_count += 1
            continue
        if _ASCII_SLASH_DATE_PATTERN.fullmatch(value) is None:
            invalid_date_format_count += 1
            continue
        try:
            parsed = parse_date_only_ascii_slash(value)
        except ValueError:
            invalid_date_count += 1
            continue
        prior_raw = normalized_to_raw.get(parsed)
        if prior_raw is not None and prior_raw != value:
            raise ValueError("Date parsing failed: normalization_collision_count=1.")
        raw_to_normalized[value] = parsed
        normalized_to_raw[parsed] = value
        parsed_values.append(parsed)

    if missing_or_nonstring_count or invalid_date_format_count or invalid_date_count:
        parts = [
            f"missing_or_nonstring_count={missing_or_nonstring_count}",
            f"invalid_date_format_count={invalid_date_format_count}",
            f"invalid_date_count={invalid_date_count}",
        ]
        raise ValueError(f"Date parsing failed: {', '.join(parts)}.")
    return np.asarray(parsed_values, dtype=object)


def validate_exact_aligned_vectors(
    *,
    expected_length: int | None = None,
    finite_names: Sequence[str] = (),
    **vectors: np.ndarray,
) -> dict[str, np.ndarray]:
    """Validate every named one-dimensional vector against one exact length.

    Args:
        expected_length: Required length. When omitted, the first vector sets it.
        finite_names: Vector names that must contain finite numeric values.
        **vectors: Named vectors to validate.

    Returns:
        Arrays in the caller's insertion order.

    Raises:
        ValueError: If a vector is empty, multidimensional, misaligned, or nonfinite.
    """
    if not vectors:
        raise ValueError("At least one vector is required.")
    required_finite = set(finite_names)
    if not required_finite.issubset(vectors):
        raise ValueError("Finite-vector names must be declared among the supplied vectors.")
    result: dict[str, np.ndarray] = {}
    required_length = expected_length
    for name, value in vectors.items():
        array = np.asarray(value)
        if array.ndim != 1:
            raise ValueError("Vectors must be one-dimensional.")
        if required_length is None:
            required_length = array.size
        if array.size != required_length:
            raise ValueError("Every vector must independently match the expected length.")
        if name in required_finite:
            try:
                numeric = array.astype(float)
            except (TypeError, ValueError) as exc:
                raise ValueError("Finite vectors must be numeric.") from exc
            if not np.all(np.isfinite(numeric)):
                raise ValueError("Finite vectors must contain finite values.")
        result[name] = array
    if required_length == 0:
        raise ValueError("Vectors must be non-empty.")
    return result


def load_csv_with_ordered_usecols(
    path: Path,
    usecols: Sequence[str],
    expected_usecols: Sequence[str],
) -> pd.DataFrame:
    """Load a CSV only when its ordered column request matches the contract.

    Args:
        path: CSV input path.
        usecols: Requested columns in request order.
        expected_usecols: Frozen allowed ordered columns.

    Returns:
        The requested columns in the requested order.

    Raises:
        ValueError: If the request is broad, reordered, or prohibited.
    """
    requested = list(usecols)
    expected = list(expected_usecols)
    if requested != expected:
        raise ValueError("CSV usecols must exactly match the frozen ordered contract.")
    if "collection_location" in requested:
        raise ValueError("Score-blind CSV loader requested a prohibited column.")
    frame = pd.read_csv(path, usecols=requested)
    # pandas returns selected columns in file order; enforce the frozen request order.
    missing = [name for name in requested if name not in frame.columns]
    if missing:
        raise ValueError("CSV reader omitted one or more required columns.")
    return frame.loc[:, requested]


def _rank_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute an AUC with half credit for tied positive-negative pairs."""
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    comparisons = positives[:, None] - negatives[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def single_auroc_delong_ci(y_true: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    """Compute a strictly validated single-AUROC DeLong confidence interval.

    Args:
        y_true: Exact binary labels.
        scores: Finite locked score values.

    Returns:
        Point-estimate and CI records following the frozen non-estimability rules.

    Raises:
        ValueError: If inputs are invalid or the independent/core AUCs disagree.
    """
    vectors = validate_aligned_finite_1d_vectors(y_true=y_true, scores=scores)
    labels = validate_exact_binary_labels_01(vectors["y_true"])
    score_values = vectors["scores"]
    n_positive = int(labels.sum())
    n_negative = int(labels.size - n_positive)
    base: dict[str, Any] = {
        "auc": None,
        "variance": None,
        "delong_core_auc": None,
        "ci_low_unclipped": None,
        "ci_high_unclipped": None,
        "ci_low": None,
        "ci_high": None,
        "ci_level": 0.95,
        "ci_method": "delong_normal_clipped",
        "ci_status": "non_estimable",
        "nonestimable_reason": None,
        "ci_nonestimable_reason": None,
        "bootstrap_seed": None,
        "n_positive": n_positive,
        "n_negative": n_negative,
    }
    if n_positive == 0 or n_negative == 0:
        base["nonestimable_reason"] = "absent_outcome_class"
        base["ci_nonestimable_reason"] = "absent_outcome_class"
        return base

    auc = _rank_auc(labels, score_values)
    order = np.argsort(-labels, kind="mergesort")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        core_aucs, covariance = _fast_delong(score_values[order][None, :], n_positive)
    core_auc = float(core_aucs[0])
    if not np.isclose(auc, core_auc, atol=1e-12, rtol=0.0):
        raise ValueError("Independent and DeLong-core AUROC calculations disagree.")
    base["auc"] = auc
    base["delong_core_auc"] = core_auc
    if n_positive < 2 or n_negative < 2:
        base["ci_nonestimable_reason"] = "insufficient_class_count_for_delong_ci"
        return base

    variance = float(covariance[0, 0])
    base["variance"] = variance
    if not np.isfinite(variance):
        base["ci_nonestimable_reason"] = "delong_variance_nonfinite"
        return base
    if variance < 0.0:
        base["ci_nonestimable_reason"] = "delong_variance_negative"
        return base
    if variance < DEGENERATE_VARIANCE_EPSILON:
        base["ci_nonestimable_reason"] = "delong_variance_below_epsilon"
        return base

    margin = float(norm.ppf(0.975) * np.sqrt(variance))
    lower = auc - margin
    upper = auc + margin
    base.update(
        {
            "ci_low_unclipped": lower,
            "ci_high_unclipped": upper,
            "ci_low": max(0.0, lower),
            "ci_high": min(1.0, upper),
            "ci_status": "estimable",
        }
    )
    return base


def confusion_from_scores(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, int]:
    """Calculate confusion counts using the frozen ``p_raw >= threshold`` rule.

    Args:
        y_true: Exact binary labels.
        scores: Finite raw scores.
        threshold: Frozen raw-score threshold.

    Returns:
        Uncorrected TP, FN, TN, and FP counts.
    """
    vectors = validate_aligned_finite_1d_vectors(y_true=y_true, scores=scores)
    labels = validate_exact_binary_labels_01(vectors["y_true"])
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite.")
    predictions = vectors["scores"] >= threshold
    return {
        "tp": int(np.sum((labels == 1) & predictions)),
        "fn": int(np.sum((labels == 1) & ~predictions)),
        "tn": int(np.sum((labels == 0) & ~predictions)),
        "fp": int(np.sum((labels == 0) & predictions)),
    }


def likelihood_ratio_metrics(tp: int, fn: int, tn: int, fp: int) -> dict[str, Any]:
    """Compute likelihood ratios and log-method intervals under frozen rules.

    Args:
        tp: Uncorrected true-positive count.
        fn: Uncorrected false-negative count.
        tn: Uncorrected true-negative count.
        fp: Uncorrected false-positive count.

    Returns:
        Uncorrected counts, correction status, and estimable LR values.
    """
    counts = {"tp": tp, "fn": fn, "tn": tn, "fp": fp}
    base: dict[str, Any] = {
        "uncorrected_counts": counts,
        "corrected_counts": None,
        "zero_cell_correction": None,
        "estimability_status": "non_estimable",
        "nonestimable_reason": None,
        "lr_plus": None,
        "lr_minus": None,
        "se_log_lr_plus": None,
        "se_log_lr_minus": None,
        "lr_plus_ci_low": None,
        "lr_plus_ci_high": None,
        "lr_minus_ci_low": None,
        "lr_minus_ci_high": None,
    }
    if any(not isinstance(value, (int, np.integer)) or value < 0 for value in counts.values()):
        base["nonestimable_reason"] = "invalid_lr_denominator"
        return base
    positive_total = tp + fn
    negative_total = tn + fp
    if positive_total == 0 or negative_total == 0:
        base["nonestimable_reason"] = "absent_outcome_class"
        return base
    corrected = {key: float(value) for key, value in counts.items()}
    if any(value == 0 for value in counts.values()):
        corrected = {key: value + 0.5 for key, value in corrected.items()}
        base["corrected_counts"] = corrected
        base["zero_cell_correction"] = "haldane_anscombe_all_cells"
    positive_total_star = corrected["tp"] + corrected["fn"]
    negative_total_star = corrected["tn"] + corrected["fp"]
    if min(positive_total_star, negative_total_star, corrected["tp"], corrected["fn"], corrected["tn"], corrected["fp"]) <= 0:
        base["nonestimable_reason"] = "invalid_lr_denominator"
        return base
    lr_plus = (corrected["tp"] / positive_total_star) / (corrected["fp"] / negative_total_star)
    lr_minus = (corrected["fn"] / positive_total_star) / (corrected["tn"] / negative_total_star)
    se_plus = np.sqrt(
        1.0 / corrected["tp"] - 1.0 / positive_total_star + 1.0 / corrected["fp"] - 1.0 / negative_total_star
    )
    se_minus = np.sqrt(
        1.0 / corrected["fn"] - 1.0 / positive_total_star + 1.0 / corrected["tn"] - 1.0 / negative_total_star
    )
    values = np.array([lr_plus, lr_minus, se_plus, se_minus], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        base["nonestimable_reason"] = "nonfinite_lr"
        return base
    z_score = norm.ppf(0.975)
    plus_interval = np.exp(np.log(lr_plus) + np.array([-1.0, 1.0]) * z_score * se_plus)
    minus_interval = np.exp(np.log(lr_minus) + np.array([-1.0, 1.0]) * z_score * se_minus)
    if not np.all(np.isfinite(np.r_[plus_interval, minus_interval])):
        base["nonestimable_reason"] = "nonfinite_lr"
        return base
    base.update(
        {
            "estimability_status": "estimable",
            "lr_plus": float(lr_plus),
            "lr_minus": float(lr_minus),
            "se_log_lr_plus": float(se_plus),
            "se_log_lr_minus": float(se_minus),
            "lr_plus_ci_low": float(plus_interval[0]),
            "lr_plus_ci_high": float(plus_interval[1]),
            "lr_minus_ci_low": float(minus_interval[0]),
            "lr_minus_ci_high": float(minus_interval[1]),
        }
    )
    return base


def wilson_proportion_ci(numerator: int, denominator: int) -> tuple[float, float]:
    """Compute a two-sided 95% Wilson binomial confidence interval.

    Args:
        numerator: Number of successes.
        denominator: Number of observations.

    Returns:
        Lower and upper Wilson confidence limits.
    """
    if denominator <= 0 or numerator < 0 or numerator > denominator:
        raise ValueError("Wilson interval requires 0 <= numerator <= denominator and denominator > 0.")
    low, high = proportion_confint(numerator, denominator, alpha=0.05, method="wilson")
    return float(low), float(high)


def newcombe_accuracy_difference_ci(
    pass_correct: int,
    pass_total: int,
    fail_correct: int,
    fail_total: int,
) -> tuple[float, float]:
    """Compute a Newcombe interval for strict-pass minus strict-fail accuracy.

    Args:
        pass_correct: Correct strict-pass classifications.
        pass_total: Strict-pass total.
        fail_correct: Correct strict-fail classifications.
        fail_total: Strict-fail total.

    Returns:
        Two-sided 95% Newcombe lower and upper limits.
    """
    if not (0 <= pass_correct <= pass_total and 0 <= fail_correct <= fail_total and pass_total > 0 and fail_total > 0):
        raise ValueError("Newcombe interval requires valid positive denominators.")
    low, high = confint_proportions_2indep(
        pass_correct,
        pass_total,
        fail_correct,
        fail_total,
        method="newcomb",
        compare="diff",
        alpha=0.05,
    )
    return float(low), float(high)


def _run_git_command(*args: str) -> str:
    """Run one read-only Git command and return stripped standard output."""
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def validate_freeze_commit(
    freeze_commit: str,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    git_runner: Callable[..., str] = _run_git_command,
    rule_hashes: Mapping[str, str] | None = None,
    artifact_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate the supplied three-path Session 0 freeze commit read-only.

    Args:
        freeze_commit: Required lowercase 40-character commit SHA.
        config: Parsed frozen package configuration.
        manifest: Parsed frozen Session 0 manifest.
        git_runner: Read-only Git command adapter for testing.
        rule_hashes: Active rule content hashes, if already computed.
        artifact_hashes: Current config/methods SHA-256 values, if already computed.

    Returns:
        Ordered freeze-validation booleans.

    Raises:
        ValueError: If any freeze invariant fails.
    """
    if _COMMIT_PATTERN.fullmatch(freeze_commit) is None:
        raise ValueError("Freeze commit must be a lowercase 40-character hexadecimal SHA.")
    authority = config["authority"]
    session_outputs = config["outputs"]["session_0"]
    expected_paths = [session_outputs["config"], session_outputs["methods"], session_outputs["freeze_manifest"]]
    try:
        git_runner("cat-file", "-e", f"{freeze_commit}^{{commit}}")
        exists = True
        git_runner("merge-base", "--is-ancestor", freeze_commit, "HEAD")
        is_ancestor = True
        parents = git_runner("rev-list", "--parents", "-n", "1", freeze_commit).split()
        parent_is_baseline = len(parents) == 2 and parents[1] == authority["baseline_head"]
        changed_paths = git_runner("diff-tree", "--no-commit-id", "--name-only", "-r", freeze_commit).splitlines()
        changed_paths_exact = changed_paths == expected_paths
        commit_blobs = {
            path: git_runner("rev-parse", f"{freeze_commit}:{path}") for path in expected_paths
        }
        working_tree_blobs = {path: git_runner("hash-object", path) for path in expected_paths}
    except (subprocess.CalledProcessError, KeyError) as exc:
        raise ValueError("Freeze-commit provenance validation failed.") from exc

    artifact_records = manifest.get("session_0_artifacts", {})
    artifact_keys = ("config", "methods", "freeze_manifest")
    artifact_paths_match = all(
        artifact_records.get(key, {}).get("path") == path for key, path in zip(artifact_keys, expected_paths, strict=True)
    )
    blob_records_match = all(
        record.get("blob_id") in (None, commit_blobs[path])
        for record, path in zip(
            (artifact_records.get(key, {}) for key in artifact_keys),
            expected_paths,
            strict=True,
        )
    )
    current_artifact_hashes = (
        dict(artifact_hashes)
        if artifact_hashes is not None
        else {
            "config": sha256_file(PROJECT_ROOT / expected_paths[0]),
            "methods": sha256_file(PROJECT_ROOT / expected_paths[1]),
        }
    )
    artifact_sha256_match = all(
        current_artifact_hashes.get(key) == artifact_records.get(key, {}).get("sha256")
        for key in ("config", "methods")
    )
    manifest_expected_paths = manifest.get("expected_session_0_changed_paths")
    manifest_paths_match = manifest_expected_paths == expected_paths
    approval_evidence = authority.get("approval_evidence", {})
    reviewed_plan = {
        "path": authority.get("reviewed_plan_path"),
        "sha256": authority.get("reviewed_plan_sha256"),
    }
    approval_evidence_matches = (
        manifest.get("approval_evidence") == approval_evidence
        and manifest.get("reviewed_plan") == reviewed_plan
        and approval_evidence.get("reviewed_plan_path") == reviewed_plan["path"]
        and approval_evidence.get("reviewed_plan_sha256") == reviewed_plan["sha256"]
        and approval_evidence.get("review_status") == "approved"
        and approval_evidence.get("user_confirmation") is True
    )
    expected_rules = {
        item["path"]: item["sha256"] for item in authority.get("active_rules", []) if "path" in item and "sha256" in item
    }
    actual_rules = rule_hashes if rule_hashes is not None else expected_rules
    manifest_rules = {
        path: record.get("sha256") for path, record in manifest.get("active_rules", {}).items() if isinstance(record, Mapping)
    }
    active_rule_hashes_match = dict(actual_rules) == expected_rules and manifest_rules == expected_rules
    result = {
        "freeze_commit_id": freeze_commit,
        "exists": exists,
        "is_ancestor_of_head": is_ancestor,
        "parent_is_baseline_head": parent_is_baseline,
        "changed_paths_exact": changed_paths_exact and manifest_paths_match,
        "config_blob_matches": (
            artifact_paths_match
            and blob_records_match
            and artifact_sha256_match
            and commit_blobs[expected_paths[0]] == working_tree_blobs[expected_paths[0]]
        ),
        "methods_blob_matches": (
            artifact_paths_match
            and blob_records_match
            and artifact_sha256_match
            and commit_blobs[expected_paths[1]] == working_tree_blobs[expected_paths[1]]
        ),
        "manifest_blob_matches": (
            artifact_paths_match
            and blob_records_match
            and commit_blobs[expected_paths[2]] == working_tree_blobs[expected_paths[2]]
        ),
        "approval_evidence_matches": approval_evidence_matches,
        "active_rule_hashes_match": active_rule_hashes_match,
    }
    if tuple(result) != _FREEZE_KEYS or not all(value is True for key, value in result.items() if key != "freeze_commit_id"):
        raise ValueError("Freeze-commit validation did not satisfy all immutable provenance checks.")
    return result


class AccessStateTracker:
    """Track Session 1 accesses as they occur.

    Args:
        cumulative_start: Optional ordered cumulative ledger from earlier work.

    Raises:
        ValueError: If a supplied cumulative ledger does not match the contract.
    """

    def __init__(self, cumulative_start: Mapping[str, bool] | None = None) -> None:
        initial = OrderedDict((key, False) for key in _ACCESS_KEYS)
        if cumulative_start is not None:
            if tuple(cumulative_start) != _ACCESS_KEYS or any(
                not isinstance(value, bool) for value in cumulative_start.values()
            ):
                raise ValueError("Cumulative access state must contain the exact ordered boolean keys.")
            initial.update(cumulative_start)
        self._activity: OrderedDict[str, bool] = OrderedDict((key, False) for key in _ACCESS_KEYS)
        self._cumulative = initial

    def mark(self, key: str) -> None:
        """Record one actually completed access category.

        Args:
            key: Exact access-state key.

        Raises:
            ValueError: If the key is not part of the frozen ledger.
        """
        if key not in _ACCESS_KEYS:
            raise ValueError("Unknown access-state key.")
        self._activity[key] = True
        self._cumulative[key] = True

    def snapshot(self) -> OrderedDict[str, OrderedDict[str, bool]]:
        """Return the ordered activity and cumulative ledgers."""
        return OrderedDict(
            [
                ("session_activity", OrderedDict(self._activity)),
                ("cumulative_through_session", OrderedDict(self._cumulative)),
            ]
        )


def build_session_1_access_state() -> dict[str, dict[str, bool]]:
    """Build an all-false Session 1 access-state ledger.

    Returns:
        Activity and cumulative ledgers with all required boolean keys.
    """
    return AccessStateTracker().snapshot()


def _validate_sha256(value: object) -> None:
    """Validate a lowercase SHA-256 leaf."""
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError("Preflight hash is not a lowercase SHA-256 string.")


def validate_preflight_schema(
    payload: Mapping[str, Any],
    *,
    frozen_hash_paths: Sequence[str],
    expected_cumulative_prior: Mapping[str, bool] | None = None,
) -> None:
    """Validate complete Session 1 preflight ordering, types, and dependencies.

    Args:
        payload: Candidate aggregate-only JSON mapping.
        frozen_hash_paths: Exact frozen-input hash-map path order.
        expected_cumulative_prior: Optional historical cumulative ledger that the
            candidate's cumulative access state must still contain.

    Raises:
        ValueError: If the candidate violates its frozen JSON contract.
    """
    if tuple(payload) != _PREFLIGHT_KEYS:
        raise ValueError("Preflight top-level keys do not match the frozen order.")
    if payload["schema_version"] != "posthoc_final_score_blind_preflight.v1":
        raise ValueError("Preflight schema version is invalid.")
    if payload["package_id"] != "posthoc_final_supplementary":
        raise ValueError("Preflight package ID is invalid.")
    if not is_utc_iso8601_z(payload["generated_at_utc"]):
        raise ValueError("Preflight provenance timestamp is invalid.")
    hashes = payload["hashes"]
    expected_hashes = (
        "config_sha256",
        "methods_sha256",
        "freeze_manifest_sha256",
        "frozen_input_whole_container_sha256",
    )
    if not isinstance(hashes, Mapping) or tuple(hashes) != expected_hashes:
        raise ValueError("Preflight hashes have an invalid key order.")
    for key in expected_hashes[:3]:
        _validate_sha256(hashes[key])
    containers = hashes["frozen_input_whole_container_sha256"]
    if not isinstance(containers, Mapping) or tuple(containers) != tuple(frozen_hash_paths):
        raise ValueError("Frozen input hash map does not match its exact path order.")
    for value in containers.values():
        _validate_sha256(value)
    freeze = payload["freeze_commit_validation"]
    if not isinstance(freeze, Mapping) or tuple(freeze) != _FREEZE_KEYS:
        raise ValueError("Freeze validation object has an invalid key order.")
    if not isinstance(freeze["freeze_commit_id"], str) or _COMMIT_PATTERN.fullmatch(freeze["freeze_commit_id"]) is None:
        raise ValueError("Freeze validation commit ID is invalid.")
    if any(not isinstance(freeze[key], bool) for key in _FREEZE_KEYS[1:]):
        raise ValueError("Freeze validation flags must be boolean.")
    aggregate_counts = payload["aggregate_counts"]
    aggregate_keys = (
        "source_cohort_n",
        "modeling_cohort_n",
        "train_n",
        "internal_test_n",
        "difficult_spectrum_by_view",
        "date_feasibility",
        "qc5_reconciliation",
    )
    if not isinstance(aggregate_counts, Mapping) or tuple(aggregate_counts) != aggregate_keys:
        raise ValueError("Aggregate counts have an invalid key order.")
    for key, expected in zip(aggregate_keys[:4], (1040, 972, 777, 195), strict=True):
        if type(aggregate_counts[key]) is not int or aggregate_counts[key] != expected:
            raise ValueError("Cohort count anchors are invalid.")
    difficult = aggregate_counts["difficult_spectrum_by_view"]
    if not isinstance(difficult, Mapping) or tuple(difficult) != ("oof", "test"):
        raise ValueError("Difficult-spectrum views have an invalid key order.")
    for view in difficult.values():
        if not isinstance(view, Mapping) or tuple(view) != (
            "n",
            "n_positive",
            "n_negative",
            "disease_category_counts",
        ):
            raise ValueError("Difficult-spectrum view schema is invalid.")
        if any(type(view[key]) is not int or view[key] < 0 for key in ("n", "n_positive", "n_negative")):
            raise ValueError("Difficult-spectrum counts must be nonnegative integers.")
        if view["n"] != view["n_positive"] + view["n_negative"]:
            raise ValueError("Difficult-spectrum totals do not reconcile.")
        disease_counts = view["disease_category_counts"]
        if not isinstance(disease_counts, Mapping) or tuple(disease_counts) != tuple(str(index) for index in range(6)):
            raise ValueError("Difficult-spectrum disease counts have an invalid key order.")
        if any(type(value) is not int or value < 0 for value in disease_counts.values()):
            raise ValueError("Difficult-spectrum disease counts must be nonnegative integers.")
        if disease_counts["0"] != 0 or disease_counts["5"] != 0:
            raise ValueError("Difficult-spectrum excluded disease counts must be zero.")
        if disease_counts["1"] + disease_counts["2"] != view["n_negative"]:
            raise ValueError("Difficult-spectrum negative disease counts do not reconcile.")
        if disease_counts["3"] + disease_counts["4"] != view["n_positive"]:
            raise ValueError("Difficult-spectrum positive disease counts do not reconcile.")
        if sum(disease_counts.values()) != view["n"]:
            raise ValueError("Difficult-spectrum disease counts must reconcile to the restricted population.")
    date_feasibility = aggregate_counts["date_feasibility"]
    if (
        not isinstance(date_feasibility, Mapping)
        or tuple(date_feasibility) != ("n_dates_overall", "n_dates_with_both_endpoint_classes")
        or type(date_feasibility["n_dates_overall"]) is not int
        or type(date_feasibility["n_dates_with_both_endpoint_classes"]) is not int
        or date_feasibility["n_dates_overall"] != 40
        or date_feasibility["n_dates_with_both_endpoint_classes"] != 34
    ):
        raise ValueError("Date feasibility anchors are invalid.")
    qc5_reconciliation = aggregate_counts["qc5_reconciliation"]
    qc5_keys = (
        "n_qc5_evaluable_source_rows",
        "n_strict_fail_among_evaluable",
        "n_strict_pass_among_evaluable",
    )
    if (
        not isinstance(qc5_reconciliation, Mapping)
        or tuple(qc5_reconciliation) != qc5_keys
        or any(type(qc5_reconciliation[key]) is not int for key in qc5_keys)
        or tuple(qc5_reconciliation[key] for key in qc5_keys) != (974, 298, 676)
    ):
        raise ValueError("QC5 reconciliation anchors are invalid.")
    if payload["one_to_one_alignment_status"] not in {"pass", "fail"}:
        raise ValueError("One-to-one alignment status is invalid.")
    if payload["train_test_disjointness_status"] not in {"pass", "fail"}:
        raise ValueError("Train/test disjointness status is invalid.")
    known_anchors = payload["known_anchor_reconciliation"]
    if not isinstance(known_anchors, Mapping) or tuple(known_anchors) != (
        "date_40_34",
        "qc5_974_298_676",
        "status",
    ):
        raise ValueError("Known-anchor reconciliation schema is invalid.")
    anchor_booleans = (known_anchors["date_40_34"], known_anchors["qc5_974_298_676"])
    if any(type(value) is not bool for value in anchor_booleans):
        raise ValueError("Known-anchor reconciliation flags must be JSON booleans.")
    expected_anchor_status = "pass" if all(anchor_booleans) else "fail"
    if known_anchors["status"] != expected_anchor_status:
        raise ValueError("Known-anchor reconciliation status does not match its booleans.")
    if payload["status"] not in {"pass", "fail"}:
        raise ValueError("Preflight status is invalid.")
    reason_code = payload["reason_code"]
    allowed_failure_reasons = {
        "alignment_failure",
        "anchor_mismatch",
        "schema_failure",
        "privacy_failure",
        "freeze_commit_invalid",
    }
    if payload["status"] == "pass" and reason_code not in {"ok", None}:
        raise ValueError("A passing preflight requires reason code 'ok' or null.")
    if payload["status"] == "fail" and (
        not isinstance(reason_code, str) or reason_code not in allowed_failure_reasons
    ):
        raise ValueError("A failed preflight requires an allowed non-null reason code.")
    if known_anchors["status"] == "fail" and (
        payload["status"] != "fail" or reason_code != "anchor_mismatch"
    ):
        raise ValueError("A failed known-anchor reconciliation requires anchor mismatch status.")
    if (
        payload["one_to_one_alignment_status"] == "fail" or payload["train_test_disjointness_status"] == "fail"
    ) and (payload["status"] != "fail" or reason_code != "alignment_failure"):
        raise ValueError("An alignment failure requires alignment failure status.")
    if not all(freeze[key] for key in _FREEZE_KEYS[1:]) and (
        payload["status"] != "fail" or reason_code != "freeze_commit_invalid"
    ):
        raise ValueError("A failed freeze validation requires freeze commit invalid status.")
    access_state = payload["access_state"]
    if not isinstance(access_state, Mapping) or tuple(access_state) != (
        "session_activity",
        "cumulative_through_session",
    ):
        raise ValueError("Access state objects are invalid.")
    for scope in access_state.values():
        if not isinstance(scope, Mapping) or tuple(scope) != _ACCESS_KEYS or any(
            not isinstance(value, bool) for value in scope.values()
        ):
            raise ValueError("Access state fields must be ordered booleans.")
    activity = access_state["session_activity"]
    cumulative = access_state["cumulative_through_session"]
    # Cumulative is the logical carry-forward union: it can never lose an access.
    if any(cumulative[key] < activity[key] for key in _ACCESS_KEYS):
        raise ValueError("Cumulative access state must be a superset of session activity.")
    if expected_cumulative_prior is not None:
        if tuple(expected_cumulative_prior) != _ACCESS_KEYS or any(
            not isinstance(value, bool) for value in expected_cumulative_prior.values()
        ):
            raise ValueError("Expected cumulative prior must be ordered booleans.")
        for key in _ACCESS_KEYS:
            if cumulative[key] < bool(expected_cumulative_prior[key]):
                raise ValueError("Cumulative access state dropped a historical prior access.")
    if payload["status"] == "pass":
        expected_activity = OrderedDict(
            (
                key,
                key
                in {
                    "prediction_container_bytes_hashed",
                    "prediction_schema_or_header_inspected",
                    "participant_level_label_values_read",
                    "participant_level_date_values_read",
                    "participant_level_qc_values_read",
                },
            )
            for key in _ACCESS_KEYS
        )
        if activity != expected_activity:
            raise ValueError("A passing preflight has an invalid Session 1 activity pattern.")
        prior = expected_cumulative_prior or OrderedDict((key, False) for key in _ACCESS_KEYS)
        expected_cumulative = OrderedDict(
            (key, bool(prior[key]) or expected_activity[key]) for key in _ACCESS_KEYS
        )
        if cumulative != expected_cumulative:
            raise ValueError("Passing preflight cumulative state is not the exact prior/activity union.")


def validate_preflight_privacy(payload: Mapping[str, Any]) -> None:
    """Reject participant-level values and other privacy-prohibited output content.

    Args:
        payload: Candidate preflight payload.

    Raises:
        ValueError: If a prohibited value or absolute path is detected.
    """
    forbidden_key_fragments = (
        "participant_row",
        "participant_id",
        "sample_id",
        "score",
        "filename",
        "raw_path",
        "absolute_path",
        "qc_flag",
    )
    prohibited_date = re.compile(r"(?<![0-9])(?:[0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{4}/[0-9]{1,2}/[0-9]{1,2})(?![0-9])")
    policy_keys = {"policy", "prohibited_category", "field_or_context"}
    permitted_structural_keys = set(_ACCESS_KEYS)
    timestamp_keys = {
        "generated_at_utc",
        "timestamp_utc",
        "frozen_at_utc",
        "correction_date_utc",
        "approval_date_utc",
        "authorization_date_utc",
        "completion_utc",
    }

    def inspect(value: Any, key: str | None = None) -> None:
        key_lower = key.lower() if key else ""
        if key_lower and key_lower not in policy_keys and key not in permitted_structural_keys and any(
            fragment in key_lower for fragment in forbidden_key_fragments
        ):
            raise ValueError("Preflight privacy validation found a prohibited field category.")
        if isinstance(value, Mapping):
            for nested_key, nested_value in value.items():
                inspect(nested_value, str(nested_key))
        elif isinstance(value, (list, tuple)):
            for nested_value in value:
                inspect(nested_value, key)
        elif isinstance(value, str):
            if value.startswith("/"):
                raise ValueError("Preflight privacy validation found an absolute path.")
            if key_lower in timestamp_keys:
                # Allowlisted timestamps must still be exact UTC grammar; this
                # rejects a slash clinical date smuggled into a timestamp field.
                if not is_utc_iso8601_z(value):
                    raise ValueError("Preflight privacy validation found an invalid timestamp value.")
                return
            if key_lower not in policy_keys and prohibited_date.search(value):
                raise ValueError("Preflight privacy validation found a prohibited date value.")
            if key_lower not in policy_keys and value.endswith((".csv", ".npz", ".pkl")):
                raise ValueError("Preflight privacy validation found a prohibited filename.")

    inspect(payload)


def atomic_write_json(
    destination: Path,
    payload: Mapping[str, Any],
    schema_validator: Callable[[Mapping[str, Any]], None],
    privacy_validator: Callable[[Mapping[str, Any]], None],
) -> None:
    """Validate and atomically promote a JSON candidate into an absent target.

    Args:
        destination: Final absent JSON target.
        payload: Candidate aggregate-only payload.
        schema_validator: Contract validator for in-memory and read-back data.
        privacy_validator: Privacy validator for in-memory and read-back data.

    Raises:
        FileExistsError: If a stale target already exists.
        ValueError: If validation or read-back validation fails.
    """
    if destination.exists():
        raise FileExistsError("Refusing to overwrite an existing preflight target.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    schema_validator(payload)
    privacy_validator(payload)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        loaded = json.loads(temp_path.read_text(encoding="utf-8"))
        schema_validator(loaded)
        privacy_validator(loaded)
        if destination.exists():
            raise FileExistsError("Refusing to overwrite an existing preflight target.")
        os.replace(temp_path, destination)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


_REGISTRY_DATE_LEAK_PATTERN = re.compile(
    r"(?<![0-9])(?:[0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{4}/[0-9]{1,2}/[0-9]{1,2})(?![0-9])"
)
_KNOWN_CATEGORY_ENUMS: dict[str, frozenset[str]] = {"view": frozenset({"oof", "test"})}


def _validate_registry_cell(column: str, value: object, dtype: str, nullable: bool) -> None:
    """Validate one aggregate CSV cell against its declared dtype and nullability."""
    is_missing = value is None or value is pd.NA or (
        isinstance(value, (float, np.floating)) and np.isnan(value)
    )
    if is_missing:
        if not nullable:
            raise ValueError(f"Non-nullable column '{column}' contains a missing value.")
        return
    if dtype in {"string", "category"}:
        if not isinstance(value, str):
            raise ValueError(f"Column '{column}' must contain string values.")
        if value.startswith("/"):
            raise ValueError(f"Column '{column}' contains an absolute path.")
        if _REGISTRY_DATE_LEAK_PATTERN.search(value):
            raise ValueError(f"Column '{column}' contains a prohibited date value.")
        if value.endswith((".csv", ".npz", ".pkl")):
            raise ValueError(f"Column '{column}' contains a prohibited filename value.")
        enum = _KNOWN_CATEGORY_ENUMS.get(column)
        if enum is not None and value not in enum:
            raise ValueError(f"Column '{column}' contains a value outside its enum.")
    elif dtype == "boolean":
        if type(value) is not bool and not isinstance(value, np.bool_):
            raise ValueError(f"Column '{column}' must contain genuine boolean values.")
    elif dtype == "Int64":
        if type(value) is bool or not isinstance(value, (int, np.integer)):
            raise ValueError(f"Column '{column}' must contain integer values.")
    elif dtype == "Float64":
        if type(value) is bool or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"Column '{column}' must contain numeric values.")
        if not np.isfinite(value):
            raise ValueError(f"Column '{column}' must contain finite populated values.")
    else:
        raise ValueError(f"Column '{column}' declares an unsupported dtype '{dtype}'.")


def validate_registry_records(records: Sequence[Mapping[str, Any]], registry_entry: Mapping[str, Any]) -> None:
    """Validate raw aggregate records against a CSV registry entry before framing.

    Args:
        records: Aggregate-only record mappings in intended row order.
        registry_entry: One ``row_order_and_schema_validation_registry`` entry.

    Raises:
        ValueError: If keys, counts, dtypes, nullability, uniqueness, or privacy fail.
    """
    columns = list(registry_entry["column_order"])
    dtype_map = registry_entry["dtype_map"]
    nullable_map = registry_entry["nullable_map"]
    if not isinstance(records, (list, tuple)) or any(not isinstance(record, Mapping) for record in records):
        raise ValueError("Registry records must be a sequence of mappings.")
    expected_count = registry_entry.get("expected_row_count")
    if expected_count is not None and len(records) != expected_count:
        raise ValueError("Registry output row count does not match the frozen contract.")
    for record in records:
        if set(record) != set(columns):
            raise ValueError("Registry record keys do not exactly match the column contract.")
        for column in columns:
            _validate_registry_cell(column, record[column], dtype_map[column], bool(nullable_map[column]))
    unique_key = registry_entry.get("unique_key_tuple")
    if unique_key:
        seen = [tuple(record[key] for key in unique_key) for record in records]
        if len(set(seen)) != len(seen):
            raise ValueError("Registry unique key tuple is not unique.")


def build_registry_frame(records: Sequence[Mapping[str, Any]], registry_entry: Mapping[str, Any]) -> pd.DataFrame:
    """Validate records then build a typed, column-ordered aggregate frame.

    Args:
        records: Aggregate-only record mappings.
        registry_entry: The CSV registry contract.

    Returns:
        A validated frame with exact column order and declared nullable dtypes.
    """
    validate_registry_records(records, registry_entry)
    columns = list(registry_entry["column_order"])
    dtype_map = registry_entry["dtype_map"]
    frame = pd.DataFrame([{column: record[column] for column in columns} for record in records], columns=columns)
    for column in columns:
        pandas_dtype = _REGISTRY_PANDAS_DTYPES[dtype_map[column]]
        frame[column] = frame[column].astype(pandas_dtype)
        if dtype_map[column] == "category":
            frame[column] = frame[column].astype("category")
    return frame


def _string_cells(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Render a typed frame into empty-field string cells for equivalence checks."""
    rendered: dict[str, list[str]] = {}
    for column in columns:
        cells: list[str] = []
        for value in frame[column].tolist():
            if value is None or value is pd.NA or (isinstance(value, float) and not np.isfinite(value)):
                cells.append("")
            elif isinstance(value, (bool, np.bool_)):
                cells.append("True" if bool(value) else "False")
            else:
                cells.append(str(value))
        rendered[column] = cells
    return pd.DataFrame(rendered, columns=list(columns), dtype="string")


def read_back_registry_csv(path: Path, registry_entry: Mapping[str, Any]) -> pd.DataFrame:
    """Read a promoted aggregate CSV back with the frozen empty-field convention.

    Args:
        path: Serialized CSV path.
        registry_entry: The CSV registry contract.

    Returns:
        The read-back frame as strings with empty fields for missing cells.

    Raises:
        ValueError: If the column order or row count does not reconcile.
    """
    columns = list(registry_entry["column_order"])
    frame = pd.read_csv(path, dtype="string", keep_default_na=False, na_values=[])
    if list(frame.columns) != columns:
        raise ValueError("Read-back CSV columns do not match the frozen order.")
    expected_count = registry_entry.get("expected_row_count")
    if expected_count is not None and len(frame) != expected_count:
        raise ValueError("Read-back CSV row count does not match the frozen contract.")
    return frame


def registry_readback_matches(frame: pd.DataFrame, path: Path, registry_entry: Mapping[str, Any]) -> bool:
    """Return whether a promoted CSV reconstructs its in-memory typed frame.

    Args:
        frame: The validated in-memory typed frame.
        path: The serialized CSV path.
        registry_entry: The CSV registry contract.

    Returns:
        ``True`` only when read-back string cells equal the frame's rendering.
    """
    columns = list(registry_entry["column_order"])
    read_back = read_back_registry_csv(path, registry_entry)
    expected = _string_cells(frame, columns)
    return read_back.reset_index(drop=True).equals(expected.reset_index(drop=True))


def validate_completion_log(
    text: str,
    *,
    grammar: Mapping[str, Any],
    session_binding: Mapping[str, Any],
    expected_freeze_commit: str | None = None,
    expected_activity: Mapping[str, bool] | None = None,
    expected_cumulative: Mapping[str, bool] | None = None,
    cumulative_prior: Mapping[str, bool] | None = None,
    allowed_reason_codes: Sequence[str] = (),
) -> list[list[str]]:
    """Validate a completion-log receipt against the machine-validated grammar.

    Args:
        text: Full serialized log text ending in one trailing newline.
        grammar: The ``machine_validated_log_grammar`` mapping.
        session_binding: The exact ``session_bindings_exact`` entry for this session.
        expected_freeze_commit: Optional freeze commit the record must equal.

    Returns:
        The parsed tab-split records in file order.

    Raises:
        ValueError: If any record order, field, type, or binding contract fails.
    """
    if not text.endswith("\n") or text.endswith("\n\n"):
        raise ValueError("Completion log must end with a single trailing newline.")
    records = [line.split("\t") for line in text[:-1].split("\n")]
    inventory = list(session_binding["ordered_output_inventory"])
    hash_count = int(session_binding["output_hash_record_count"])
    if len(inventory) != hash_count:
        raise ValueError("Session binding hash count and inventory disagree.")
    expected_types = (
        ["header", "session_identity", "freeze_commit"]
        + ["output_hash"] * hash_count
        + [
            "deviations_or_nonestimability_summary",
            "access_state_session_activity",
            "access_state_cumulative_through_session",
            "completion_receipt",
        ]
    )
    if [record[0] for record in records] != expected_types:
        raise ValueError("Completion log record order does not match the frozen grammar.")
    contracts = grammar["record_contracts"]

    def _check_fields(record: Sequence[str], record_key: str) -> None:
        expected_fields = contracts[record_key]["fields_in_order"]
        if len(record) != len(expected_fields):
            raise ValueError(f"Completion log record '{record_key}' has the wrong field count.")

    header, identity, freeze = records[0], records[1], records[2]
    _check_fields(header, "header")
    if (header[1], header[2]) != ("posthoc_final_completion_log.v1", "posthoc_final_supplementary"):
        raise ValueError("Completion log header identity is invalid.")
    if not is_utc_iso8601_z(header[3]):
        raise ValueError("Completion log header timestamp is invalid.")
    _check_fields(identity, "session_identity")
    allowed_sessions = contracts["session_identity"]["field_types"]["session_name"]["allowed_values"]
    bound_session = session_binding.get("session_name")
    if not isinstance(bound_session, str) or identity[1] not in allowed_sessions or identity[1] != bound_session:
        raise ValueError("Completion log session identity is invalid.")
    if identity[2] != "complete":
        raise ValueError("Completion log session identity is not complete.")
    _check_fields(freeze, "freeze_commit")
    if _COMMIT_PATTERN.fullmatch(freeze[1]) is None:
        raise ValueError("Completion log freeze commit is invalid.")
    if expected_freeze_commit is not None and freeze[1] != expected_freeze_commit:
        raise ValueError("Completion log freeze commit does not match the supplied commit.")
    forbidden_hash_targets = set(session_binding.get("must_neither_hash_nor_claim", ())) | {
        session_binding.get("log_path")
    }
    for offset, path in enumerate(inventory):
        record = records[3 + offset]
        _check_fields(record, "output_hash_records")
        if record[1] != path:
            raise ValueError("Completion log output-hash path or order is invalid.")
        if record[1].startswith("/"):
            raise ValueError("Completion log output path must be project-relative.")
        if record[1] in forbidden_hash_targets:
            raise ValueError("Completion log must not hash its own log or the preflight artifact.")
        if _SHA256_PATTERN.fullmatch(record[2]) is None:
            raise ValueError("Completion log output hash is not a lowercase SHA-256 string.")
    deviations = records[3 + hash_count]
    _check_fields(deviations, "deviations_or_nonestimability_summary")
    summary_status = deviations[1]
    if summary_status not in {"none", "present_retained"}:
        raise ValueError("Completion log deviations summary status is invalid.")
    if summary_status == "none" and deviations[2] != "":
        raise ValueError("A 'none' deviations summary must carry no reason codes.")
    if summary_status == "present_retained" and not deviations[2]:
        raise ValueError("A 'present_retained' deviations summary requires reason codes.")
    if deviations[2]:
        reason_codes = deviations[2].split(",")
        if (
            any(not code for code in reason_codes)
            or len(reason_codes) != len(set(reason_codes))
            or reason_codes != sorted(reason_codes)
            or any(code not in set(allowed_reason_codes) for code in reason_codes)
        ):
            raise ValueError("Completion log deviation reason-code serialization is invalid.")
    activity = records[4 + hash_count]
    cumulative = records[5 + hash_count]
    for scope in (activity, cumulative):
        if len(scope) != 1 + len(_ACCESS_KEYS) or any(token not in {"true", "false"} for token in scope[1:]):
            raise ValueError("Completion log access-state record must carry nine ordered booleans.")
    parsed_activity = OrderedDict(zip(_ACCESS_KEYS, (token == "true" for token in activity[1:]), strict=True))
    parsed_cumulative = OrderedDict(zip(_ACCESS_KEYS, (token == "true" for token in cumulative[1:]), strict=True))
    if expected_activity is None or expected_cumulative is None:
        raise ValueError("Completion log validation requires exact caller-supplied access states.")
    if parsed_activity != OrderedDict(expected_activity) or parsed_cumulative != OrderedDict(expected_cumulative):
        raise ValueError("Completion log access state does not match the caller-supplied record.")
    prior = OrderedDict(cumulative_prior or ((key, False) for key in _ACCESS_KEYS))
    union = OrderedDict((key, prior[key] or parsed_activity[key]) for key in _ACCESS_KEYS)
    if parsed_cumulative != union:
        raise ValueError("Completion log cumulative state is not the exact prior/activity union.")
    receipt = records[6 + hash_count]
    _check_fields(receipt, "completion_receipt")
    if not is_utc_iso8601_z(receipt[1]) or receipt[2] != "complete":
        raise ValueError("Completion log completion receipt is invalid.")
    return records
