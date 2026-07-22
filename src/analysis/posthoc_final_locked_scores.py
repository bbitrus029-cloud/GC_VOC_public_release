"""Score-blind Session 1 preflight and locked-score analysis framework."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.analysis.posthoc_final_common import (
    ACCESS_STATE_KEYS,
    AccessStateError,
    AccessStateTracker,
    atomic_write_json,
    build_registry_frame,
    confusion_from_scores,
    likelihood_ratio_metrics,
    load_csv_with_ordered_usecols,
    newcombe_accuracy_difference_ci,
    parse_date_only_ascii_slash,
    parse_date_series_ascii_slash,
    registry_readback_matches,
    sha256_file,
    single_auroc_delong_ci,
    validate_completion_log,
    validate_exact_aligned_vectors,
    validate_exact_binary_labels_01,
    validate_freeze_commit,
    validate_preflight_privacy,
    validate_preflight_schema,
    wilson_proportion_ci,
)
from src.config import PROJECT_ROOT


LOGGER = logging.getLogger(__name__)
_DISEASE_KEYS = tuple(str(value) for value in range(6))
_SATURATION_COLUMNS = tuple(f"s{index}_saturation_fraction" for index in range(1, 12))
SESSION_1_CORRECTIONS_IMPLEMENTED = frozenset({f"C{index:02d}" for index in range(1, 17)})
BASE_CONFIG_RELATIVE_PATH = "configs/post_hoc_final_supplementary.yaml"
CURRENT_MANIFEST_RELATIVE_PATH = "outputs/analysis/posthoc_final_freeze_manifest.json"
SCORE_BLIND_PREFLIGHT_RELATIVE_PATH = "outputs/analysis/posthoc_final_score_blind_preflight.json"
# Exact ordered date-parsing payload approved by Amendment 01 (20 keys).
_APPROVED_DATE_PARSING_PAYLOAD_EXACT: tuple[tuple[str, Any], ...] = (
    ("format", "slash_encoded_date_only_ascii"),
    ("regex_full_match", r"^([0-9]{4})/([0-9]{1,2})/([0-9]{1,2})$"),
    ("regex_digit_class", "ascii_0_to_9_only"),
    ("year_digits", 4),
    ("month_digits", "one_or_two"),
    ("day_digits", "one_or_two"),
    ("parser", "explicit_three_token_integer_parse_then_pd_to_datetime_format_%Y-%m-%d_exact_true_errors_raise"),
    ("midnight_normalization_required", True),
    ("parsed_components_must_equal_original_numeric_tokens", True),
    ("exact_round_trip_text_required", False),
    ("whitespace_or_time_of_day_rejected", True),
    ("iso_yyyy_mm_dd_rejected", True),
    ("unicode_or_non_ascii_digits_rejected", True),
    ("coercion_or_imputation_forbidden", True),
    ("errors_report_aggregate_counts_only", True),
    ("errors_must_not_echo_value_id_or_date", True),
    ("raw_to_normalized_one_to_one_required", True),
    ("normalization_collision_forbidden", True),
    ("canonicalize_in_memory_only", True),
    ("rewrite_sample_index_forbidden", True),
)
SESSION0_FREEZE = "3c0ff6b3b40460a2c340ad3193222ee54979109d"
AMENDMENT_FREEZE_DEFAULT_FOR_TESTS = "2e08e8fb89d99bb62cca2e5204a13ff09e47d432"
AMENDMENT_CHANGED_PATHS = (
    "configs/post_hoc_final_supplementary_pre_execution_amendment_01.yaml",
    "docs/modeling/final_post_hoc_supplementary_pre_execution_amendment_01.md",
    "outputs/analysis/posthoc_final_pre_execution_amendment_01_manifest.json",
)
SESSION_4_SCIENTIFIC_OUTPUT_PATHS = (
    "outputs/analysis/posthoc_final_locked_score_metrics_minirocket_ridge.csv",
    "outputs/analysis/posthoc_final_locked_score_composition_minirocket_ridge.csv",
    "outputs/analysis/posthoc_final_qc5_correctness_minirocket_ridge.csv",
    "outputs/analysis/posthoc_final_within_date_concordance_minirocket_ridge.csv",
)
SESSION_4_COMPLETION_LOG_PATH = "outputs/logs/posthoc_final_locked_scores.log"
SESSION_1_TO_3_IMPLEMENTATION_PATHS = (
    "src/analysis/posthoc_final_common.py",
    "src/analysis/posthoc_final_locked_scores.py",
    "src/analysis/posthoc_final_date_grouped.py",
    "src/analysis/posthoc_final_reporting.py",
    "src/analysis/posthoc_final_audit.py",
    "scripts/run_posthoc_final_locked_scores.py",
    "scripts/run_posthoc_final_date_grouped.py",
    "scripts/run_posthoc_final_reporting.py",
    "scripts/run_posthoc_final_audit.py",
    "tests/test_posthoc_final_common.py",
    "tests/test_posthoc_final_locked_scores.py",
    "tests/test_posthoc_final_date_grouped.py",
    "tests/test_posthoc_final_reporting.py",
    "tests/test_posthoc_final_audit.py",
)
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_DISEASE_LABELS = ("HC", "BGL", "LGIN", "HGIN", "EGC", "AGC")
_THRESHOLD = 0.3418938786147648


def load_posthoc_config(path: Path) -> dict[str, Any]:
    """Load the frozen post-hoc YAML configuration.

    Args:
        path: Configuration file path.

    Returns:
        Parsed configuration mapping.

    Raises:
        ValueError: If the YAML root is not a mapping.
    """
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Post-hoc configuration root must be a mapping.")
    return config


def _canonical_immutable_path(
    path: Path,
    expected_relative_path: str,
    description: str,
    identity_error: str,
) -> Path:
    """Resolve one immutable project path and require exact canonical identity."""
    candidate = path if path.is_absolute() else PROJECT_ROOT / path
    expected = PROJECT_ROOT / expected_relative_path
    try:
        canonical_candidate = candidate.resolve(strict=True)
        canonical_expected = expected.resolve(strict=True)
    except (OSError, RuntimeError):
        raise ValueError(f"The {description} path could not be resolved.") from None
    if canonical_candidate != canonical_expected:
        raise ValueError(identity_error)
    return canonical_expected


def load_effective_posthoc_config(
    config_path: Path,
) -> dict[str, Any]:
    """Load the single consolidated current post-hoc configuration.

    Args:
        config_path: Current consolidated configuration path.

    Returns:
        Validated consolidated configuration.

    Raises:
        ValueError: If the path or embedded approved date-parser contract is invalid.
    """
    config_path = _canonical_immutable_path(
        config_path,
        BASE_CONFIG_RELATIVE_PATH,
        "current configuration",
        "The current configuration path is immutable.",
    )
    config = load_posthoc_config(config_path)
    payload = config.get("loaders", {}).get("date_parsing")
    expected_keys = tuple(key for key, _ in _APPROVED_DATE_PARSING_PAYLOAD_EXACT)
    if not isinstance(payload, Mapping) or tuple(payload) != expected_keys:
        raise ValueError("Consolidated date-parser keys or order do not match the approved contract.")
    for key, value in _APPROVED_DATE_PARSING_PAYLOAD_EXACT:
        if payload[key] != value or type(payload[key]) is not type(value):
            raise ValueError("Consolidated date-parser values do not match the approved contract.")
    return config


def evaluate_conditional_preflight_retry_authorization(
    *,
    approval_evidence_present: bool,
    amendment_freeze_valid: bool,
    corrections_implemented: Sequence[str] | frozenset[str],
    synthetic_tests_passed: bool,
    other_gates_satisfied: bool,
) -> bool:
    """Evaluate the amendment's all-of gate for one retry.

    Args:
        approval_evidence_present: Whether exact independent approval exists.
        amendment_freeze_valid: Whether the amendment commit validates.
        corrections_implemented: Implemented correction identifiers.
        synthetic_tests_passed: Explicit test gate supplied by the invoker.
        other_gates_satisfied: Remaining score-blind authority checks.

    Returns:
        ``True`` only when every required gate is satisfied.
    """
    return (
        approval_evidence_present
        and amendment_freeze_valid
        and frozenset(corrections_implemented) == SESSION_1_CORRECTIONS_IMPLEMENTED
        and synthetic_tests_passed
        and other_gates_satisfied
    )


def _locked_scores_log_relative_path(config: Mapping[str, Any]) -> str:
    """Return the locked-scores completion-log path from either config shape."""
    logs = config["outputs"]["completion_logs"]
    if "paths" in logs:
        return logs["paths"]["locked_scores"]
    return logs["locked_scores"]


def _git(*args: str) -> str:
    """Run one read-only Git command."""
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def validate_execute_git_prerequisites(
    required_paths: Sequence[str],
    *,
    git_runner: Callable[..., str] = _git,
) -> None:
    """Require committed, HEAD-identical, clean execute prerequisites.

    Args:
        required_paths: Exact project-relative paths that must be committed inputs.
        git_runner: Read-only Git adapter for production and temporary-repository tests.

    Raises:
        ValueError: If a prerequisite is absent from HEAD, differs from HEAD, or
            any tracked or untracked worktree path is dirty.
    """
    if not required_paths or len(required_paths) != len(set(required_paths)):
        raise ValueError("Execute prerequisite path inventory is invalid.")
    try:
        if git_runner("status", "--porcelain", "--untracked-files=normal"):
            raise ValueError("Approved execution requires a clean worktree.")
        for path in required_paths:
            if not isinstance(path, str) or not path or path.startswith("/"):
                raise ValueError("Execute prerequisite path inventory is invalid.")
            git_runner("ls-files", "--error-unmatch", path)
            git_runner("cat-file", "-e", f"HEAD:{path}")
            head_blob = git_runner("rev-parse", f"HEAD:{path}")
            worktree_blob = git_runner("hash-object", path)
            if head_blob != worktree_blob:
                raise ValueError("Approved execution requires HEAD-identical committed prerequisites.")
    except subprocess.CalledProcessError as exc:
        raise ValueError("Approved execution requires committed HEAD-identical prerequisites.") from exc


def _validate_annotated_tag_chain(git_runner: Callable[..., str]) -> None:
    """Require the six frozen annotated tags and their ancestry."""
    tags = (
        "preprocessing-prespecified",
        "preprocessing-frozen",
        "modeling-prespecified",
        "data-split-locked",
        "best-model-selected",
        "modeling-frozen",
    )
    try:
        for tag in tags:
            if git_runner("cat-file", "-t", tag) != "tag":
                raise ValueError("Required provenance tag is not annotated.")
            git_runner("merge-base", "--is-ancestor", tag, "HEAD")
        git_runner("merge-base", "--is-ancestor", "modeling-frozen", SESSION0_FREEZE)
    except (subprocess.CalledProcessError, KeyError) as exc:
        raise ValueError("Required frozen tag ancestry validation failed.") from exc


def validate_amendment_freeze_commit(
    amendment_freeze_commit: str,
    *,
    historical_amendment: Mapping[str, Any],
    git_runner: Callable[..., str] = _git,
) -> OrderedDict[str, Any]:
    """Validate Amendment 01 from recorded hashes and immutable Git objects.

    Args:
        amendment_freeze_commit: Externally supplied amendment commit SHA.
        historical_amendment: Consolidated historical Amendment 01 record.
        git_runner: Read-only Git adapter.

    Returns:
        Ordered amendment validation flags.

    Raises:
        ValueError: If any amendment provenance invariant is false.
    """
    if _COMMIT_PATTERN.fullmatch(amendment_freeze_commit) is None:
        raise ValueError("Amendment freeze commit must be lowercase 40-character hexadecimal.")
    if historical_amendment.get("commit_id") != amendment_freeze_commit:
        raise ValueError("Supplied Amendment 01 commit disagrees with historical authority.")
    expected_paths = historical_amendment.get("changed_paths_exact")
    if expected_paths != list(AMENDMENT_CHANGED_PATHS):
        raise ValueError("Amendment freeze changed-path contract is invalid.")
    artifacts = historical_amendment.get("artifacts")
    if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes)) or len(artifacts) != 3:
        raise ValueError("Amendment 01 artifact record is malformed.")
    artifact_records = [dict(item) for item in artifacts if isinstance(item, Mapping)]
    if len(artifact_records) != 3 or [item.get("path") for item in artifact_records] != expected_paths:
        raise ValueError("Amendment 01 artifact paths are invalid.")
    hashes_valid = all(
        isinstance(item.get("sha256"), str) and _SHA256_PATTERN.fullmatch(item["sha256"])
        for item in artifact_records
    )
    blobs_valid = all(isinstance(item.get("git_blob_id"), str) and item["git_blob_id"] for item in artifact_records)
    approval = historical_amendment.get("approval_evidence", {})
    approval_ok = (
        isinstance(approval, Mapping)
        and approval.get("review_status") == "approved"
        and approval.get("user_confirmation") is True
    )
    try:
        git_runner("cat-file", "-e", f"{amendment_freeze_commit}^{{commit}}")
        parents = git_runner("rev-list", "--parents", "-n", "1", amendment_freeze_commit).split()
        changed = git_runner("diff-tree", "--no-commit-id", "--name-only", "-r", amendment_freeze_commit).splitlines()
        blobs = {path: git_runner("rev-parse", f"{amendment_freeze_commit}:{path}") for path in expected_paths}
        git_runner("merge-base", "--is-ancestor", amendment_freeze_commit, "HEAD")
    except (subprocess.CalledProcessError, KeyError) as exc:
        raise ValueError("Amendment freeze provenance validation failed.") from exc
    blob_matches = all(
        blobs[path] == record["git_blob_id"]
        for path, record in zip(expected_paths, artifact_records, strict=True)
    )
    result: OrderedDict[str, Any] = OrderedDict(
        [
            ("amendment_freeze_commit_id", amendment_freeze_commit),
            ("exists", True),
            ("is_ancestor_of_head", True),
            (
                "parent_is_session0_freeze",
                len(parents) == 2 and parents[1] == historical_amendment.get("sole_parent") == SESSION0_FREEZE,
            ),
            ("changed_paths_exact", changed == expected_paths),
            ("overlay_blob_matches", hashes_valid and blobs_valid and blob_matches),
            ("approval_evidence_matches", approval_ok),
        ]
    )
    if not all(value is True for key, value in result.items() if key != "amendment_freeze_commit_id"):
        raise ValueError("Amendment freeze validation did not satisfy all immutable provenance checks.")
    return result


def validate_current_runtime_authority(
    config_path: Path,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    require_head_identical: bool,
    git_runner: Callable[..., str] = _git,
) -> OrderedDict[str, bool]:
    """Validate current local authorities separately from immutable history.

    Args:
        config_path: Canonical consolidated config path.
        config: Parsed consolidated configuration.
        manifest: Parsed consolidated manifest.
        require_head_identical: Whether tracked current records must equal ``HEAD``.
        git_runner: Read-only Git adapter.

    Returns:
        Ordered current-authority validation flags.

    Raises:
        ValueError: If any path, hash, self-hash, or Git-identity check fails.
    """
    try:
        current = manifest["current_runtime_authority"]
        config_record = current["config"]
        methods_record = current["methods"]
        manifest_record = current["manifest"]
        plan_record = current["operational_plan"]
        guidance_record = current["agent_guidance"]
        session_0 = config["outputs"]["session_0"]
        authority = config["authority"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Current runtime authority record is malformed.") from exc
    config_relative = str(config_path.relative_to(PROJECT_ROOT))
    if config_record.get("path") != config_relative or config_relative != BASE_CONFIG_RELATIVE_PATH:
        raise ValueError("Current config path does not match consolidated authority.")
    if methods_record.get("path") != session_0.get("methods"):
        raise ValueError("Current methods path does not match consolidated authority.")
    if manifest_record.get("path") != session_0.get("freeze_manifest") or manifest_record.get(
        "path"
    ) != CURRENT_MANIFEST_RELATIVE_PATH:
        raise ValueError("Current manifest path does not match consolidated authority.")
    if manifest_record.get("self_hash") is not None or manifest_record.get("self_hash_policy") != "forbidden":
        raise ValueError("Current manifest self-hash must remain null and forbidden.")
    if sha256_file(config_path) != config_record.get("sha256"):
        raise ValueError("Current config hash does not match the consolidated manifest.")
    methods_path = PROJECT_ROOT / methods_record["path"]
    if sha256_file(methods_path) != methods_record.get("sha256"):
        raise ValueError("Current methods hash does not match the consolidated manifest.")
    if (
        plan_record.get("path") != authority.get("reviewed_plan_path")
        or plan_record.get("sha256") != authority.get("reviewed_plan_sha256")
        or sha256_file(PROJECT_ROOT / plan_record["path"]) != plan_record.get("sha256")
    ):
        raise ValueError("Reviewed plan hash does not match the current authority.")
    if (
        guidance_record.get("path") != authority.get("agent_guidance_path")
        or guidance_record.get("sha256") != authority.get("agent_guidance_sha256")
        or sha256_file(PROJECT_ROOT / guidance_record["path"]) != guidance_record.get("sha256")
    ):
        raise ValueError("Agent guidance hash does not match the current authority.")
    head_identical = True
    if require_head_identical:
        try:
            for path in (config_relative, methods_record["path"], manifest_record["path"]):
                git_runner("ls-files", "--error-unmatch", path)
                git_runner("cat-file", "-e", f"HEAD:{path}")
                if git_runner("rev-parse", f"HEAD:{path}") != git_runner("hash-object", path):
                    head_identical = False
        except subprocess.CalledProcessError as exc:
            raise ValueError("Current record bundle must be committed and HEAD-identical.") from exc
        if not head_identical:
            raise ValueError("Current record bundle must be committed and HEAD-identical.")
    return OrderedDict(
        [
            ("config_hash_matches", True),
            ("methods_hash_matches", True),
            ("manifest_self_hash_forbidden", True),
            ("operational_plan_hash_matches", True),
            ("agent_guidance_hash_matches", True),
            ("record_bundle_head_identical_when_required", head_identical),
        ]
    )


def validate_pre_access_authority(
    *,
    config_path: Path,
    freeze_commit: str,
    amendment_freeze_commit: str,
    mode: str,
    git_runner: Callable[..., str] = _git,
) -> tuple[dict[str, Any], OrderedDict[str, str], OrderedDict[str, Any]]:
    """Validate every authority gate before participant-level reads.

    Args:
        config_path: Current consolidated config path.
        freeze_commit: Session 0 freeze SHA.
        amendment_freeze_commit: Amendment freeze SHA.
        mode: ``preflight`` or ``execute``.
        git_runner: Read-only Git adapter.

    Returns:
        Effective config, entry frozen-input hashes, and ordered authority records.

    Raises:
        ValueError: If any immutable authority, stale-target, or input-hash gate fails.
    """
    if mode not in {"preflight", "execute"}:
        raise ValueError("Unknown Session 1 mode.")
    config_path = _canonical_immutable_path(
        config_path,
        BASE_CONFIG_RELATIVE_PATH,
        "current configuration",
        "The current configuration path is immutable.",
    )
    base = load_effective_posthoc_config(config_path)
    manifest_path = PROJECT_ROOT / base["outputs"]["session_0"]["freeze_manifest"]
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    current_validation = validate_current_runtime_authority(
        config_path,
        base,
        manifest,
        require_head_identical=mode == "execute",
        git_runner=git_runner,
    )
    _validate_annotated_tag_chain(git_runner)
    freeze_validation = validate_freeze_commit(
        freeze_commit,
        base,
        manifest,
        git_runner=git_runner,
    )
    try:
        amendment = base["historical_consolidation"]["amendment_01"]
        manifest_amendment = manifest["historical_evidence"]["amendment_01"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Historical Amendment 01 record is malformed.") from exc
    comparison_keys = ("commit_id", "sole_parent", "changed_paths_exact", "artifacts", "approval_evidence")
    if not isinstance(amendment, Mapping) or not isinstance(manifest_amendment, Mapping) or any(
        amendment.get(key) != manifest_amendment.get(key) for key in comparison_keys
    ):
        raise ValueError("Historical Amendment 01 config and manifest records disagree.")
    if amendment.get("access_state") != manifest_amendment.get("access_state"):
        raise ValueError("Historical Amendment 01 access-state records disagree.")
    if amendment.get("session_1_corrections") != manifest_amendment.get("session_1_corrections"):
        raise ValueError("Historical Amendment 01 correction records disagree.")
    amendment_validation = validate_amendment_freeze_commit(
        amendment_freeze_commit,
        historical_amendment=amendment,
        git_runner=git_runner,
    )
    outputs = base["outputs"]["scientific_and_closure"]
    configured_session4 = (
        outputs["locked_score_metrics"]["path"],
        outputs["locked_score_composition"]["path"],
        outputs["qc5_correctness"]["path"],
        outputs["within_date_concordance"]["path"],
    )
    if configured_session4 != SESSION_4_SCIENTIFIC_OUTPUT_PATHS:
        raise ValueError("Configured Session 4 scientific outputs do not match the frozen path contract.")
    if _locked_scores_log_relative_path(base) != SESSION_4_COMPLETION_LOG_PATH:
        raise ValueError("Configured Session 4 completion log does not match the frozen path contract.")
    preflight_target = PROJECT_ROOT / outputs["score_blind_preflight"]["path"]
    # Session 4 scientific CSVs and the completion log must be absent in every mode.
    for target in (*SESSION_4_SCIENTIFIC_OUTPUT_PATHS, SESSION_4_COMPLETION_LOG_PATH):
        if (PROJECT_ROOT / target).exists():
            raise ValueError("A declared Session 4 target already exists.")
    container_paths = list(base["frozen_inputs"]["whole_container_sha256"])
    container_hashes = OrderedDict((path, sha256_file(PROJECT_ROOT / path)) for path in container_paths)
    if dict(container_hashes) != base["frozen_inputs"]["whole_container_sha256"]:
        raise ValueError("Frozen input whole-container hashes do not match authority.")
    if mode == "preflight":
        if preflight_target.exists():
            raise ValueError("A score-blind preflight target already exists.")
    else:
        if not preflight_target.exists():
            raise ValueError("Approved execution requires a prior validated score-blind preflight.")
        preflight_relative = str(preflight_target.relative_to(PROJECT_ROOT))
        try:
            configured_paths = tuple(base["outputs"]["future_implementation_paths"])
        except (KeyError, TypeError) as exc:
            raise ValueError("Frozen Session 1-3 implementation path inventory is invalid.") from exc
        if configured_paths != SESSION_1_TO_3_IMPLEMENTATION_PATHS:
            raise ValueError("Frozen Session 1-3 implementation path inventory is invalid.")
        validate_execute_git_prerequisites(
            (*SESSION_1_TO_3_IMPLEMENTATION_PATHS, preflight_relative),
            git_runner=git_runner,
        )
        try:
            with preflight_target.open("r", encoding="utf-8") as handle:
                preflight_payload = json.load(handle)
        except OSError as exc:
            raise ValueError("Approved execution requires a committed HEAD-identical preflight.") from exc
        cumulative_prior = _amendment_cumulative_prior(base)
        validate_preflight_schema(
            preflight_payload,
            frozen_hash_paths=container_paths,
            expected_cumulative_prior=cumulative_prior,
        )
        validate_preflight_privacy(preflight_payload)
        historical_artifacts = manifest["historical_evidence"]["original_session_0"]["artifacts"]
        historical_hashes = {item["path"]: item["sha256"] for item in historical_artifacts}
        expected_authority_hashes = {
            "config_sha256": historical_hashes[base["outputs"]["session_0"]["config"]],
            "methods_sha256": historical_hashes[base["outputs"]["session_0"]["methods"]],
            "freeze_manifest_sha256": historical_hashes[base["outputs"]["session_0"]["freeze_manifest"]],
        }
        if any(preflight_payload["hashes"][key] != value for key, value in expected_authority_hashes.items()):
            raise ValueError("Preflight recorded authority hashes disagree with current authorities.")
        recorded = preflight_payload["hashes"]["frozen_input_whole_container_sha256"]
        if dict(recorded) != dict(container_hashes):
            raise ValueError("Preflight recorded frozen-input hashes disagree with current containers.")
        if preflight_payload["freeze_commit_validation"]["freeze_commit_id"] != freeze_commit:
            raise ValueError("Preflight freeze commit provenance disagrees with the supplied freeze commit.")
        if preflight_payload["status"] != "pass":
            raise ValueError("Approved execution requires a passing prior preflight.")
    entry_hashes = _ordered_entry_integrity_hashes(base, config_path)
    if mode == "execute":
        entry_hashes[SCORE_BLIND_PREFLIGHT_RELATIVE_PATH] = sha256_file(preflight_target)
    authority_records = OrderedDict(
        [
            ("session0_freeze", OrderedDict(freeze_validation)),
            ("amendment_freeze", amendment_validation),
            ("current_runtime", current_validation),
            ("container_hashes", container_hashes),
        ]
    )
    if mode == "execute":
        authority_records["preflight_cumulative"] = OrderedDict(
            preflight_payload["access_state"]["cumulative_through_session"]
        )
    return base, entry_hashes, authority_records


def _ordered_entry_integrity_hashes(
    base: Mapping[str, Any],
    config_path: Path,
) -> OrderedDict[str, str]:
    """Hash the exact ordered C15 source set covering every frozen authority input.

    Args:
        base: Parsed immutable base configuration.
        config_path: Immutable base config path.

    Returns:
        Ordered project-relative path to SHA-256 map for the full integrity set.
    """
    session_0 = base["outputs"]["session_0"]
    ordered: OrderedDict[str, str] = OrderedDict()
    for path in base["frozen_inputs"]["whole_container_sha256"]:
        ordered[path] = sha256_file(PROJECT_ROOT / path)
    extra_paths = [
        str(config_path.relative_to(PROJECT_ROOT)),
        session_0["methods"],
        session_0["freeze_manifest"],
        base["authority"]["reviewed_plan_path"],
        base["authority"]["agent_guidance_path"],
    ]
    for path in extra_paths:
        ordered[path] = sha256_file(PROJECT_ROOT / path)
    return ordered


def _verify_exit_hashes(config: Mapping[str, Any], entry_hashes: Mapping[str, str]) -> None:
    """Re-hash the exact ordered C15 entry set and require unchanged equality."""
    exit_hashes = OrderedDict((path, sha256_file(PROJECT_ROOT / path)) for path in entry_hashes)
    if exit_hashes != OrderedDict(entry_hashes):
        raise ValueError("Frozen inputs changed between entry and pre-promotion hash gates.")
    containers = config["frozen_inputs"]["whole_container_sha256"]
    if any(exit_hashes.get(path) != digest for path, digest in containers.items()):
        raise ValueError("Frozen input containers changed between entry and pre-promotion hash gates.")


def _validate_disease_category_codes(values: object) -> np.ndarray:
    """Validate exact integer disease codes 0--5 without silent float/string coercion.

    Args:
        values: Candidate disease-category values.

    Returns:
        Integer code vector.

    Raises:
        ValueError: If values are non-integer-equivalent, out of range, or coerced text.
    """
    array = np.asarray(values)
    if array.dtype == object or array.dtype.kind in {"U", "S"}:
        raise ValueError("Disease categories must be exact integer codes, not text or objects.")
    numeric = array.astype(float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("Disease categories must be finite integer codes.")
    rounded = np.rint(numeric)
    if not np.array_equal(rounded, numeric):
        raise ValueError("Disease categories must be exact integer-equivalent codes.")
    codes = rounded.astype(int)
    if not np.isin(codes, range(6)).all():
        raise ValueError("Disease categories must be exact integers from 0 through 5.")
    return codes


def _require_bool_values(values: object, field: str) -> np.ndarray:
    """Return a genuine boolean vector, rejecting strings or arbitrary numerics."""
    array = np.asarray(values, dtype=object)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{field} must be a non-empty one-dimensional boolean vector.")
    if any(not isinstance(value, (bool, np.bool_)) for value in array.tolist()):
        raise ValueError(f"{field} must contain genuine boolean values only.")
    return np.array([bool(value) for value in array.tolist()], dtype=bool)


def difficult_spectrum_filter(lineage: pd.DataFrame) -> pd.DataFrame:
    """Select frozen difficult-spectrum disease categories without score access.

    Args:
        lineage: Aligned score-blind prediction and sample-index columns.

    Returns:
        Rows from categories 1--4 in their original order with aggregate counts.

    Raises:
        ValueError: If labels disagree or required fields are invalid.
    """
    required = {"sample_id", "y_true", "label", "disease_category"}
    if not required.issubset(lineage.columns):
        raise ValueError("Difficult-spectrum lineage is missing required columns.")
    labels = validate_exact_binary_labels_01(lineage["label"].to_numpy())
    prediction_labels = validate_exact_binary_labels_01(lineage["y_true"].to_numpy())
    if not np.array_equal(labels, prediction_labels):
        raise ValueError("Prediction labels do not agree with frozen sample-index labels.")
    codes = _validate_disease_category_codes(lineage["disease_category"].to_numpy())
    filtered = lineage.loc[np.isin(codes, [1, 2, 3, 4])].copy()
    filtered_categories = _validate_disease_category_codes(filtered["disease_category"].to_numpy())
    counts = OrderedDict((key, int(np.sum(filtered_categories == int(key)))) for key in _DISEASE_KEYS)
    expected_labels = np.where(filtered_categories >= 3, 1, 0)
    if not np.array_equal(filtered["y_true"].to_numpy(dtype=int), expected_labels):
        raise ValueError("Difficult-spectrum disease categories and endpoint labels disagree.")
    if counts["0"] != 0 or counts["5"] != 0 or sum(counts.values()) != len(filtered):
        raise ValueError("Restricted difficult-spectrum counts do not reconcile.")
    filtered.attrs["disease_category_counts"] = counts
    return filtered


def _canonical_in_memory_date(value: object) -> pd.Timestamp:
    """Normalize one in-memory date that is either slash text or a timestamp."""
    if isinstance(value, pd.Timestamp):
        normalized = value.normalize()
        if value != normalized:
            raise ValueError("Date parsing failed: invalid_date_count=1.")
        return pd.Timestamp(normalized)
    return parse_date_only_ascii_slash(value)


def _comparable_date_statistics(
    labels: np.ndarray,
    scores: np.ndarray,
    dates: np.ndarray,
) -> tuple[list[tuple[float, int]], int, int]:
    """Return aggregate per-date concordance numerators and pair counts."""
    grouped: dict[pd.Timestamp, list[int]] = {}
    for index, raw_date in enumerate(dates):
        date = _canonical_in_memory_date(raw_date)
        grouped.setdefault(date, []).append(index)
    values: list[tuple[float, int]] = []
    noncomparable = 0
    for indices in grouped.values():
        date_labels = labels[indices]
        positives = scores[np.asarray(indices)[date_labels == 1]]
        negatives = scores[np.asarray(indices)[date_labels == 0]]
        pair_count = int(positives.size * negatives.size)
        if pair_count == 0:
            noncomparable += 1
            continue
        comparisons = positives[:, None] - negatives[None, :]
        concordant = float(np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0))
        values.append((concordant, pair_count))
    return values, len(grouped), noncomparable


def within_date_concordance(
    y_true: np.ndarray,
    scores: np.ndarray,
    dates: np.ndarray,
    *,
    view: str,
) -> dict[str, Any]:
    """Calculate pair-weighted within-date conditional concordance.

    Args:
        y_true: Exact binary labels.
        scores: Finite locked raw scores.
        dates: Exact date-only values held only in memory.
        view: Separate OOF or test view label.

    Returns:
        Aggregate estimate, counts, and frozen bootstrap metadata.
    """
    labels = validate_exact_binary_labels_01(y_true)
    vectors = validate_exact_aligned_vectors(
        expected_length=len(labels),
        finite_names=("scores",),
        scores=scores,
        dates=np.asarray(dates, dtype=object),
    )
    values = vectors["scores"].astype(float)
    date_values = vectors["dates"]
    comparable, total_dates, noncomparable_dates = _comparable_date_statistics(labels, values, date_values)
    pair_count = int(sum(item[1] for item in comparable))
    result: dict[str, Any] = {
        "view": view,
        "estimate": None,
        "n_dates": total_dates,
        "n_comparable_dates": len(comparable),
        "n_noncomparable_dates": noncomparable_dates,
        "n_comparable_pairs": pair_count,
        "estimability_status": "non_estimable",
        "nonestimable_reason": None,
        "ci_low": None,
        "ci_high": None,
        "ci_status": "non_estimable",
        "ci_nonestimable_reason": None,
        "ci_method": None,
        "n_bootstrap_attempts": 0,
        "n_bootstrap_valid": 0,
        "bootstrap_seed": None,
    }
    if not comparable:
        result["nonestimable_reason"] = "no_comparable_date_groups"
        result["ci_nonestimable_reason"] = "no_comparable_date_groups"
        return result
    numerators = np.asarray([item[0] for item in comparable], dtype=float)
    denominators = np.asarray([item[1] for item in comparable], dtype=float)
    result["estimate"] = float(numerators.sum() / denominators.sum())
    result["estimability_status"] = "estimable"
    if len(comparable) < 10:
        result["ci_nonestimable_reason"] = "fewer_than_10_comparable_date_groups"
        return result
    rng = np.random.default_rng(42)
    draws = np.empty(2000, dtype=float)
    for draw_index in range(2000):
        selected = rng.integers(0, len(comparable), size=len(comparable))
        draws[draw_index] = numerators[selected].sum() / denominators[selected].sum()
    lower, upper = np.quantile(draws, [0.025, 0.975], method="linear")
    result.update(
        {
            "ci_low": float(lower),
            "ci_high": float(upper),
            "ci_status": "estimable",
            "ci_method": "percentile_bootstrap_linear_quantile",
            "n_bootstrap_attempts": 2000,
            "n_bootstrap_valid": 2000,
            "bootstrap_seed": 42,
        }
    )
    return result


def reconstruct_qc5(diagnostics: pd.DataFrame, summary: pd.DataFrame) -> dict[str, Any]:
    """Reconstruct historical strict QC5 statuses from frozen diagnostics.

    Args:
        diagnostics: Sample IDs and exact lower-case saturation-fraction fields.
        summary: Sample IDs, QC1 pass, and final E005 QC5 pass fields.

    Returns:
        In-memory strict/evaluable status vectors and aggregate reconciliation.

    Raises:
        ValueError: If diagnostics are incomplete, malformed, or contradict E005.
    """
    diagnostic_required = {"sample_id", *_SATURATION_COLUMNS}
    summary_required = {"sample_id", "qc1_env_sensor_fault_pass", "qc5_sensor_saturation_pass"}
    if set(diagnostics.columns) != diagnostic_required or set(summary.columns) != summary_required:
        raise ValueError("QC5 loader columns do not match the frozen contract.")
    if diagnostics["sample_id"].duplicated().any() or summary["sample_id"].duplicated().any():
        raise ValueError("QC5 sample IDs must be unique.")
    qc1 = summary["qc1_env_sensor_fault_pass"]
    if qc1.isna().any():
        raise ValueError("QC1 pass field must be complete boolean data.")
    qc1_bools = _require_bool_values(qc1.to_numpy(), "QC1 pass field")
    evaluable_summary = summary.loc[qc1_bools].copy()
    joined = evaluable_summary.merge(diagnostics, on="sample_id", how="left", validate="one_to_one", sort=False)
    if joined[list(_SATURATION_COLUMNS)].isna().any().any():
        raise ValueError("Every QC1-pass row must have complete QC5 diagnostics.")
    fractions = joined.loc[:, _SATURATION_COLUMNS].to_numpy(dtype=float)
    if not np.all(np.isfinite(fractions)) or np.any(fractions < 0):
        raise ValueError("QC5 saturation fractions must be finite nonnegative values.")
    n_sat = np.sum(fractions > 0.003, axis=1).astype(int)
    strict_pass = n_sat == 0
    e005_pass = n_sat < 7
    summary_e005 = _require_bool_values(joined["qc5_sensor_saturation_pass"].to_numpy(), "QC5 E005 pass field")
    if not np.array_equal(e005_pass, summary_e005):
        raise ValueError("QC5 reconstruction does not reproduce final E005 pass status.")
    return {
        "sample_id": joined["sample_id"].to_numpy(),
        "n_sat": n_sat,
        "strict_pass": strict_pass,
        "e005_pass": e005_pass,
        "n_qc5_evaluable": int(len(joined)),
        "n_strict_fail": int((~strict_pass).sum()),
        "n_strict_pass": int(strict_pass.sum()),
    }


def strict_qc5_correctness_inventory(
    *,
    y_true: np.ndarray,
    scores: np.ndarray,
    strict_pass: np.ndarray,
    disease_category: np.ndarray,
    threshold: float,
    view: str,
) -> list[dict[str, Any]]:
    """Build the fixed 19-row strict-QC5 correctness/composition inventory.

    Args:
        y_true: Exact binary labels.
        scores: Finite locked raw scores.
        strict_pass: Historical strict QC5 pass flags.
        disease_category: Exact disease categories 0 through 5.
        threshold: Frozen raw-score threshold.
        view: Separate OOF or test view.

    Returns:
        Aggregate-only accuracy, difference, outcome, and disease records.
    """
    labels = validate_exact_binary_labels_01(y_true)
    vectors = validate_exact_aligned_vectors(
        expected_length=len(labels),
        finite_names=("scores",),
        scores=scores,
        strict_pass=strict_pass,
        disease_category=disease_category,
    )
    score_values = vectors["scores"].astype(float)
    statuses = _require_bool_values(vectors["strict_pass"], "Strict-QC5 status")
    categories = _validate_disease_category_codes(vectors["disease_category"])
    predicted = score_values >= threshold
    records: list[dict[str, Any]] = []
    accuracy_by_status: dict[str, tuple[int, int]] = {}
    for name, selector in (("strict_pass", statuses.astype(bool)), ("strict_fail", ~statuses.astype(bool))):
        n = int(selector.sum())
        correct = int(np.sum(predicted[selector] == labels[selector]))
        incorrect = n - correct
        record: dict[str, Any] = {
            "analysis_id": "analysis_4_strict_qc5",
            "view": view,
            "record_type": "accuracy",
            "strict_status": name,
            "category_type": "not_applicable",
            "category": "not_applicable",
            "numerator_n": None,
            "denominator_n": n,
            "proportion": None,
            "correct_n": correct,
            "incorrect_n": incorrect,
            "estimate": None,
            "ci_low": None,
            "ci_high": None,
            "ci_level": None,
            "ci_method": None,
            "correction_applied": None,
            "threshold_p_raw": threshold,
            "score_scale": "p_raw",
            "estimability_status": "non_estimable",
            "nonestimable_reason": "empty_strict_status_group",
            "ci_status": "non_estimable",
            "ci_nonestimable_reason": "empty_strict_status_group",
        }
        if n:
            record.update(
                {
                    "estimate": correct / n,
                    "ci_low": wilson_proportion_ci(correct, n)[0],
                    "ci_high": wilson_proportion_ci(correct, n)[1],
                    "ci_level": 0.95,
                    "ci_method": "wilson",
                    "estimability_status": "estimable",
                    "nonestimable_reason": None,
                    "ci_status": "estimable",
                    "ci_nonestimable_reason": None,
                }
            )
        records.append(record)
        accuracy_by_status[name] = (correct, n)
    pass_correct, pass_n = accuracy_by_status["strict_pass"]
    fail_correct, fail_n = accuracy_by_status["strict_fail"]
    difference: dict[str, Any] = {
        "analysis_id": "analysis_4_strict_qc5",
        "view": view,
        "record_type": "accuracy_difference",
        "strict_status": "not_applicable",
        "category_type": "not_applicable",
        "category": "not_applicable",
        "numerator_n": None,
        "denominator_n": None,
        "proportion": None,
        "correct_n": None,
        "incorrect_n": None,
        "estimate": None,
        "ci_low": None,
        "ci_high": None,
        "ci_level": None,
        "ci_method": None,
        "correction_applied": None,
        "threshold_p_raw": threshold,
        "score_scale": "p_raw",
        "estimability_status": "non_estimable",
        "nonestimable_reason": "empty_strict_status_group",
        "ci_status": "non_estimable",
        "ci_nonestimable_reason": "empty_strict_status_group",
    }
    if pass_n and fail_n:
        ci_low, ci_high = newcombe_accuracy_difference_ci(pass_correct, pass_n, fail_correct, fail_n)
        difference.update(
            {
                "estimate": pass_correct / pass_n - fail_correct / fail_n,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ci_level": 0.95,
                "ci_method": "newcombe",
                "estimability_status": "estimable",
                "nonestimable_reason": None,
                "ci_status": "estimable",
                "ci_nonestimable_reason": None,
            }
        )
    records.append(difference)
    for name, selector in (("strict_pass", statuses.astype(bool)), ("strict_fail", ~statuses.astype(bool))):
        denominator = int(selector.sum())
        for outcome, label in ((0, "negative"), (1, "positive")):
            count = int(np.sum(labels[selector] == outcome))
            records.append(
                {
                    "analysis_id": "analysis_4_strict_qc5",
                    "view": view,
                    "record_type": "outcome_composition",
                    "strict_status": name,
                    "category_type": "outcome",
                    "category": label,
                    "numerator_n": count,
                    "denominator_n": denominator,
                    "proportion": (count / denominator) if denominator else None,
                    "correct_n": None,
                    "incorrect_n": None,
                    "estimate": None,
                    "ci_low": None,
                    "ci_high": None,
                    "ci_level": None,
                    "ci_method": None,
                    "correction_applied": None,
                    "threshold_p_raw": None,
                    "score_scale": None,
                    "estimability_status": "estimable" if denominator else "non_estimable",
                    "nonestimable_reason": None if denominator else "empty_strict_status_group",
                    "ci_status": "not_applicable",
                    "ci_nonestimable_reason": "not_applicable",
                }
            )
        for category in range(6):
            count = int(np.sum(categories[selector] == category))
            records.append(
                {
                    "analysis_id": "analysis_4_strict_qc5",
                    "view": view,
                    "record_type": "disease_composition",
                    "strict_status": name,
                    "category_type": "disease_category",
                    "category": _DISEASE_LABELS[category],
                    "numerator_n": count,
                    "denominator_n": denominator,
                    "proportion": (count / denominator) if denominator else None,
                    "correct_n": None,
                    "incorrect_n": None,
                    "estimate": None,
                    "ci_low": None,
                    "ci_high": None,
                    "ci_level": None,
                    "ci_method": None,
                    "correction_applied": None,
                    "threshold_p_raw": None,
                    "score_scale": None,
                    "estimability_status": "estimable" if denominator else "non_estimable",
                    "nonestimable_reason": None if denominator else "empty_strict_status_group",
                    "ci_status": "not_applicable",
                    "ci_nonestimable_reason": "not_applicable",
                }
            )
    if len(records) != 19:
        raise RuntimeError("Strict-QC5 inventory did not retain its frozen 19-row contract.")
    return records


def _validate_numeric_unique_ids(values: pd.Series, name: str) -> np.ndarray:
    """Validate numeric unique sample IDs without coercing identifiers."""
    if values.isna().any() or not pd.api.types.is_numeric_dtype(values) or values.duplicated().any():
        raise ValueError(f"{name} sample IDs must be unique numeric values.")
    return values.to_numpy()


def validate_locked_score_alignment(
    *,
    lineage: pd.DataFrame,
    oof: pd.DataFrame,
    test: pd.DataFrame,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
) -> dict[str, str]:
    """Validate exact OOF/train and test/test membership and labels.

    Args:
        lineage: Source lineage rows.
        oof: Score-blind OOF prediction rows.
        test: Score-blind test prediction rows.
        train_ids: Locked train membership IDs in frozen order.
        test_ids: Locked test membership IDs in frozen order.

    Returns:
        Aggregate alignment and disjointness status strings.

    Raises:
        ValueError: If any membership, order, label, or size invariant fails.
    """
    for name, frame in (("lineage", lineage), ("oof", oof), ("test", test)):
        if "sample_id" not in frame or (name != "lineage" and "y_true" not in frame):
            raise ValueError("Aligned inputs are missing required columns.")
        _validate_numeric_unique_ids(frame["sample_id"], name)
    if "label" not in lineage:
        raise ValueError("Lineage must contain labels.")
    lineage_labels = validate_exact_binary_labels_01(lineage["label"].to_numpy())
    oof_labels = validate_exact_binary_labels_01(oof["y_true"].to_numpy())
    test_labels = validate_exact_binary_labels_01(test["y_true"].to_numpy())
    train_array = np.asarray(train_ids)
    test_array = np.asarray(test_ids)
    if train_array.ndim != 1 or test_array.ndim != 1 or len(np.unique(train_array)) != len(train_array):
        raise ValueError("Locked split memberships must be unique one-dimensional ID vectors.")
    if set(train_array).intersection(test_array):
        raise ValueError("Locked train and test memberships must be disjoint.")
    if not np.array_equal(oof["sample_id"].to_numpy(), train_array):
        raise ValueError("OOF order and membership must exactly equal locked train membership.")
    if not np.array_equal(test["sample_id"].to_numpy(), test_array):
        raise ValueError("Test order and membership must exactly equal locked test membership.")
    index_by_id = lineage.set_index("sample_id")["label"]
    if not np.array_equal(index_by_id.loc[train_array].to_numpy(dtype=int), oof_labels):
        raise ValueError("OOF prediction labels disagree with lineage labels.")
    if not np.array_equal(index_by_id.loc[test_array].to_numpy(dtype=int), test_labels):
        raise ValueError("Test prediction labels disagree with lineage labels.")
    if "in_final_output" not in lineage:
        raise ValueError("Lineage must declare the modeling-cohort membership flag.")
    modeling_ids = lineage.loc[lineage["in_final_output"].astype(bool), "sample_id"].to_numpy()
    combined = np.concatenate([train_array, test_array])
    if len(combined) != len(np.unique(combined)):
        raise ValueError("Locked train and test memberships overlap.")
    if set(combined) != set(modeling_ids) or len(combined) != len(modeling_ids):
        raise ValueError("Locked train and test union must exactly equal the modeling cohort sample IDs.")
    _ = lineage_labels
    return {"one_to_one_alignment_status": "pass", "train_test_disjointness_status": "pass"}


def build_score_blind_preflight(
    *,
    hashes: OrderedDict[str, Any],
    freeze_commit_validation: OrderedDict[str, Any],
    aggregate_counts: OrderedDict[str, Any],
    frozen_hash_paths: Sequence[str],
    access_state: Mapping[str, Any] | None = None,
    alignment: Mapping[str, str] | None = None,
    expected_cumulative_prior: Mapping[str, bool] | None = None,
) -> OrderedDict[str, Any]:
    """Build an ordered, aggregate-only passing Session 1 preflight candidate.

    Args:
        hashes: Ordered hash object.
        freeze_commit_validation: Ordered provenance validation object.
        aggregate_counts: Ordered aggregate count object.
        frozen_hash_paths: Exact frozen-input hash path order.

    Returns:
        A schema-validated preflight candidate.
    """
    if access_state is None:
        tracker = AccessStateTracker(cumulative_start=expected_cumulative_prior)
        for key in (
            "prediction_container_bytes_hashed",
            "prediction_schema_or_header_inspected",
            "participant_level_label_values_read",
            "participant_level_date_values_read",
            "participant_level_qc_values_read",
        ):
            tracker.mark(key)
        access_state = tracker.snapshot()
    alignment = (
        {"one_to_one_alignment_status": "pass", "train_test_disjointness_status": "pass"}
        if alignment is None
        else alignment
    )
    date_anchor = aggregate_counts["date_feasibility"] == OrderedDict(
        [("n_dates_overall", 40), ("n_dates_with_both_endpoint_classes", 34)]
    )
    qc5_anchor = aggregate_counts["qc5_reconciliation"] == OrderedDict(
        [
            ("n_qc5_evaluable_source_rows", 974),
            ("n_strict_fail_among_evaluable", 298),
            ("n_strict_pass_among_evaluable", 676),
        ]
    )
    anchor_status = "pass" if date_anchor and qc5_anchor else "fail"
    payload: OrderedDict[str, Any] = OrderedDict(
        [
            ("schema_version", "posthoc_final_score_blind_preflight.v1"),
            ("package_id", "posthoc_final_supplementary"),
            ("generated_at_utc", datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")),
            ("hashes", hashes),
            ("freeze_commit_validation", freeze_commit_validation),
            ("aggregate_counts", aggregate_counts),
            ("one_to_one_alignment_status", alignment["one_to_one_alignment_status"]),
            ("train_test_disjointness_status", alignment["train_test_disjointness_status"]),
            (
                "known_anchor_reconciliation",
                OrderedDict(
                    [("date_40_34", date_anchor), ("qc5_974_298_676", qc5_anchor), ("status", anchor_status)]
                ),
            ),
            ("status", anchor_status),
            ("reason_code", None if anchor_status == "pass" else "anchor_mismatch"),
            ("access_state", access_state),
        ]
    )
    validate_preflight_schema(
        payload,
        frozen_hash_paths=frozen_hash_paths,
        expected_cumulative_prior=expected_cumulative_prior,
    )
    validate_preflight_privacy(payload)
    return payload


def _split_membership(split_path: Path, lineage: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Map locked modeling-cohort row indices to their ordered sample IDs."""
    if "sample_id" not in lineage or "in_final_output" not in lineage:
        raise ValueError("Split membership mapping requires sample IDs and final-output flags.")
    modeling = lineage.loc[lineage["in_final_output"].astype(bool)].reset_index(drop=True)
    if modeling["sample_id"].isna().any() or modeling["sample_id"].duplicated().any():
        raise ValueError("Modeling-cohort sample IDs must be complete and unique.")
    with np.load(split_path, allow_pickle=False) as split:
        if not {"train_idx", "test_idx"}.issubset(split.files):
            raise ValueError("Frozen split lacks required train_idx/test_idx arrays.")
        train_idx = np.asarray(split["train_idx"])
        test_idx = np.asarray(split["test_idx"])
    for values in (train_idx, test_idx):
        if (
            values.ndim != 1
            or not np.issubdtype(values.dtype, np.integer)
            or len(np.unique(values)) != len(values)
            or np.any(values < 0)
            or np.any(values >= len(modeling))
        ):
            raise ValueError("Frozen split indices must be unique in-range one-dimensional integers.")
    if set(train_idx).intersection(test_idx):
        raise ValueError("Frozen train and test indices must be disjoint.")
    return (
        modeling.loc[train_idx, "sample_id"].to_numpy(),
        modeling.loc[test_idx, "sample_id"].to_numpy(),
    )


def run_score_blind_preflight(
    config_path: Path,
    freeze_commit: str,
    amendment_freeze_commit: str,
    *,
    synthetic_tests_passed: bool = False,
    git_runner: Callable[..., str] = _git,
) -> Path:
    """Run the authorized score-blind preflight without loading score values.

    Args:
        config_path: Frozen package configuration path.
        freeze_commit: Externally supplied full Session 0 freeze SHA.
        amendment_freeze_commit: Externally supplied full amendment freeze SHA.
        synthetic_tests_passed: Explicit retry-test gate; defaults to false.
        git_runner: Read-only Git adapter.

    Returns:
        Promoted aggregate preflight JSON path.

    Raises:
        ValueError: If any score-blind provenance, alignment, or anchor check fails.
    """
    config_path = _canonical_immutable_path(
        config_path,
        BASE_CONFIG_RELATIVE_PATH,
        "current configuration",
        "The current configuration path is immutable.",
    )
    config, entry_hashes, authority = validate_pre_access_authority(
        config_path=config_path,
        freeze_commit=freeze_commit,
        amendment_freeze_commit=amendment_freeze_commit,
        mode="preflight",
        git_runner=git_runner,
    )
    amendment = config["historical_consolidation"]["amendment_01"]
    if amendment.get("immediate_preflight_retry_authorized_at_materialization") is not False:
        raise ValueError("Amendment package incorrectly authorizes immediate preflight retry.")
    if not evaluate_conditional_preflight_retry_authorization(
        approval_evidence_present=bool(authority["amendment_freeze"]["approval_evidence_matches"]),
        amendment_freeze_valid=all(
            value is True
            for key, value in authority["amendment_freeze"].items()
            if key != "amendment_freeze_commit_id"
        ),
        corrections_implemented=SESSION_1_CORRECTIONS_IMPLEMENTED,
        synthetic_tests_passed=synthetic_tests_passed,
        other_gates_satisfied=True,
    ):
        raise ValueError("Conditional preflight retry authorization is not satisfied.")
    cumulative_prior = _amendment_cumulative_prior(config)
    tracker = AccessStateTracker(cumulative_start=cumulative_prior)
    try:
        # Entry-integrity hashing already completed inside the authority gate.
        tracker.mark("prediction_container_bytes_hashed")
        prediction_paths = [
            PROJECT_ROOT / "outputs/predictions/binary_minirocket_ridge_nested_cv_oof.csv",
            PROJECT_ROOT / "outputs/predictions/binary_minirocket_ridge_test_predictions.csv",
        ]
        expected_headers = config["frozen_inputs"]["prediction_static_headers"]
        for prediction_path in prediction_paths:
            relative_path = str(prediction_path.relative_to(PROJECT_ROOT))
            header = list(pd.read_csv(prediction_path, nrows=0).columns)
            tracker.mark("prediction_schema_or_header_inspected")
            if header != expected_headers[relative_path]:
                raise ValueError("Prediction CSV header does not match frozen Session 0 inspection.")
        oof = load_csv_with_ordered_usecols(prediction_paths[0], ["sample_id", "y_true"], ["sample_id", "y_true"])
        tracker.mark("participant_level_label_values_read")
        test = load_csv_with_ordered_usecols(prediction_paths[1], ["sample_id", "y_true"], ["sample_id", "y_true"])
        tracker.mark("participant_level_label_values_read")
        lineage_path = PROJECT_ROOT / "data/processed/sample_index.csv"
        lineage = load_csv_with_ordered_usecols(
            lineage_path,
            ["sample_id", "label", "disease_category", "collection_time", "in_final_output"],
            ["sample_id", "label", "disease_category", "collection_time", "in_final_output"],
        )
        # The lineage read materializes both label and date values at once.
        tracker.mark("participant_level_label_values_read")
        tracker.mark("participant_level_date_values_read")
        train_ids, test_ids = _split_membership(PROJECT_ROOT / "data/processed/splits/train_test_split.npz", lineage)
        alignment = validate_locked_score_alignment(
            lineage=lineage, oof=oof, test=test, train_ids=train_ids, test_ids=test_ids
        )
        if (len(lineage), int(lineage["in_final_output"].sum()), len(oof), len(test)) != (1040, 972, 777, 195):
            raise ValueError("Source, modeling, train, or test count anchor failed.")
        modeling = lineage.loc[lineage["in_final_output"].astype(bool)].reset_index(drop=True)
        date_feasibility = _modeling_date_feasibility(modeling)
        if tuple(date_feasibility.values()) != (40, 34):
            raise ValueError("Date feasibility anchor failed.")
        diagnostics = load_csv_with_ordered_usecols(
            PROJECT_ROOT / "data/processed/qc_reports/qc_diagnostics.csv",
            ["sample_id", *_SATURATION_COLUMNS],
            ["sample_id", *_SATURATION_COLUMNS],
        )
        tracker.mark("participant_level_qc_values_read")
        qc_summary = load_csv_with_ordered_usecols(
            PROJECT_ROOT / "data/processed/qc_reports/qc_summary.csv",
            ["sample_id", "qc1_env_sensor_fault_pass", "qc5_sensor_saturation_pass"],
            ["sample_id", "qc1_env_sensor_fault_pass", "qc5_sensor_saturation_pass"],
        )
        tracker.mark("participant_level_qc_values_read")
        qc5 = reconstruct_qc5(diagnostics, qc_summary)
        qc5_reconciliation = OrderedDict(
            [
                ("n_qc5_evaluable_source_rows", qc5["n_qc5_evaluable"]),
                ("n_strict_fail_among_evaluable", qc5["n_strict_fail"]),
                ("n_strict_pass_among_evaluable", qc5["n_strict_pass"]),
            ]
        )
        if tuple(qc5_reconciliation.values()) != (974, 298, 676):
            raise ValueError("QC5 aggregate reconciliation anchor failed.")
        oof_lineage = oof.merge(lineage, on="sample_id", how="inner", validate="one_to_one", sort=False)
        test_lineage = test.merge(lineage, on="sample_id", how="inner", validate="one_to_one", sort=False)
        difficult_oof = difficult_spectrum_filter(oof_lineage)
        difficult_test = difficult_spectrum_filter(test_lineage)
        aggregate_counts = OrderedDict(
            [
                ("source_cohort_n", int(len(lineage))),
                ("modeling_cohort_n", int(modeling.shape[0])),
                ("train_n", int(len(oof))),
                ("internal_test_n", int(len(test))),
                (
                    "difficult_spectrum_by_view",
                    OrderedDict(
                        [
                            (
                                "oof",
                                OrderedDict(
                                    [
                                        ("n", int(len(difficult_oof))),
                                        ("n_positive", int(difficult_oof["y_true"].sum())),
                                        ("n_negative", int((difficult_oof["y_true"] == 0).sum())),
                                        ("disease_category_counts", difficult_oof.attrs["disease_category_counts"]),
                                    ]
                                ),
                            ),
                            (
                                "test",
                                OrderedDict(
                                    [
                                        ("n", int(len(difficult_test))),
                                        ("n_positive", int(difficult_test["y_true"].sum())),
                                        ("n_negative", int((difficult_test["y_true"] == 0).sum())),
                                        ("disease_category_counts", difficult_test.attrs["disease_category_counts"]),
                                    ]
                                ),
                            ),
                        ]
                    ),
                ),
                ("date_feasibility", date_feasibility),
                ("qc5_reconciliation", qc5_reconciliation),
            ]
        )
        container_paths = list(config["frozen_inputs"]["whole_container_sha256"])
        container_hashes = OrderedDict((path, entry_hashes[path]) for path in container_paths)
        historical_hashes = {
            item["path"]: item["sha256"]
            for item in config["historical_consolidation"]["original_session_0"]["artifacts"]
        }
        hashes = OrderedDict(
            [
                ("config_sha256", historical_hashes[config["outputs"]["session_0"]["config"]]),
                ("methods_sha256", historical_hashes[config["outputs"]["session_0"]["methods"]]),
                (
                    "freeze_manifest_sha256",
                    historical_hashes[config["outputs"]["session_0"]["freeze_manifest"]],
                ),
                ("frozen_input_whole_container_sha256", container_hashes),
            ]
        )
        candidate = build_score_blind_preflight(
            hashes=hashes,
            freeze_commit_validation=authority["session0_freeze"],
            aggregate_counts=aggregate_counts,
            frozen_hash_paths=container_paths,
            access_state=tracker.snapshot(),
            alignment=alignment,
            expected_cumulative_prior=cumulative_prior,
        )
        target = PROJECT_ROOT / SCORE_BLIND_PREFLIGHT_RELATIVE_PATH
        # C15: re-hash the exact ordered entry-integrity set immediately before promotion.
        _verify_exit_hashes(config, entry_hashes)
        atomic_write_json(
            target,
            candidate,
            lambda value: validate_preflight_schema(
                value,
                frozen_hash_paths=container_paths,
                expected_cumulative_prior=cumulative_prior,
            ),
            validate_preflight_privacy,
        )
        return target
    except AccessStateError:
        raise
    except Exception as exc:
        raise AccessStateError(
            "Score-blind preflight failed closed before promotion.", tracker.snapshot()
        ) from exc


def _modeling_date_feasibility(modeling: pd.DataFrame) -> OrderedDict[str, int]:
    """Compute in-memory date-feasibility anchors from validated modeling rows.

    Args:
        modeling: Modeling-cohort rows with slash dates and endpoint labels.

    Returns:
        Ordered overall and both-class date counts computed from parsed dates.
    """
    parsed_dates = parse_date_series_ascii_slash(modeling["collection_time"].tolist())
    date_frame = pd.DataFrame({"date": parsed_dates, "label": modeling["label"].to_numpy()})
    grouped = date_frame.groupby("date", sort=False)["label"].agg(lambda values: set(values))
    return OrderedDict(
        [
            ("n_dates_overall", int(len(grouped))),
            ("n_dates_with_both_endpoint_classes", int(sum(values == {0, 1} for values in grouped))),
        ]
    )


def _amendment_cumulative_prior(config: Mapping[str, Any]) -> OrderedDict[str, bool]:
    """Extract the exact ordered Step-2 cumulative access-state carry-forward ledger."""
    amendment = config.get("historical_consolidation", {}).get("amendment_01", {})
    state = amendment.get("access_state", {}).get("step_2_approval_materialization", {})
    cumulative = state.get("cumulative_through_session", {}) if isinstance(state, Mapping) else {}
    if tuple(cumulative) != ACCESS_STATE_KEYS or any(not isinstance(value, bool) for value in cumulative.values()):
        raise ValueError("Amendment Step-2 cumulative access state is malformed.")
    return OrderedDict((key, bool(cumulative[key])) for key in ACCESS_STATE_KEYS)


def _metric_records(
    *,
    analysis_id: str,
    view: str,
    population_id: str,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    on_first_aggregate: Callable[[], None] | None = None,
) -> list[dict[str, Any]]:
    """Build AUROC, threshold, and likelihood-ratio records for one population.

    Args:
        analysis_id: Owning analysis identifier.
        view: Separate OOF or test view.
        population_id: Owning population identifier.
        labels: Exact binary endpoint labels.
        scores: Finite locked raw scores.
        threshold: Frozen raw-score threshold.
        on_first_aggregate: Optional callback invoked immediately after the first
            score-derived aggregate (the AUROC result) is successfully formed, and
            before any later threshold or likelihood-ratio construction can fail.

    Returns:
        AUROC, sensitivity, specificity, and likelihood-ratio records.
    """
    labels = validate_exact_binary_labels_01(labels)
    values = np.asarray(scores, dtype=float)
    validate_exact_aligned_vectors(expected_length=len(labels), finite_names=("scores",), scores=values)
    n_positive = int(labels.sum())
    n_negative = int(len(labels) - n_positive)
    auc = single_auroc_delong_ci(labels, values)
    if on_first_aggregate is not None:
        on_first_aggregate()
    base = {
        "analysis_id": analysis_id,
        "view": view,
        "population_id": population_id,
        "n": len(labels),
        "n_positive": n_positive,
        "n_negative": n_negative,
    }
    auc_row = {
        **base,
        "metric": "auc_roc",
        "estimate": auc["auc"],
        "ci_low": auc["ci_low"],
        "ci_high": auc["ci_high"],
        "ci_level": auc["ci_level"] if auc["ci_status"] == "estimable" else None,
        "ci_method": auc["ci_method"] if auc["ci_status"] == "estimable" else None,
        "tp": None,
        "tn": None,
        "fp": None,
        "fn": None,
        "sensitivity_unadjusted": None,
        "specificity_unadjusted": None,
        "threshold_p_raw": None,
        "score_scale": "p_raw",
        "zero_cell_correction": "not_applicable",
        "correction_applied": None,
        "estimability_status": auc["ci_status"] if auc["auc"] is None else "estimable",
        "nonestimable_reason": auc["nonestimable_reason"],
        "ci_status": auc["ci_status"],
        "ci_nonestimable_reason": auc["ci_nonestimable_reason"],
    }
    confusion = confusion_from_scores(labels, values, threshold)
    sensitivity = confusion["tp"] / n_positive if n_positive else None
    specificity = confusion["tn"] / n_negative if n_negative else None

    def threshold_row(metric: str) -> dict[str, Any]:
        denominator = n_positive if metric == "sensitivity" else n_negative
        numerator = confusion["tp"] if metric == "sensitivity" else confusion["tn"]
        if denominator:
            ci_low, ci_high = wilson_proportion_ci(numerator, denominator)
            estimate = numerator / denominator
            status, reason, ci_status, ci_reason = "estimable", None, "estimable", None
        else:
            ci_low = ci_high = estimate = None
            status = ci_status = "non_estimable"
            reason = ci_reason = "absent_outcome_class"
        return {
            **base,
            "metric": metric,
            "estimate": estimate,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "ci_level": 0.95 if denominator else None,
            "ci_method": "wilson" if denominator else None,
            **confusion,
            "sensitivity_unadjusted": sensitivity,
            "specificity_unadjusted": specificity,
            "threshold_p_raw": threshold,
            "score_scale": "p_raw",
            "zero_cell_correction": "not_applicable",
            "correction_applied": None,
            "estimability_status": status,
            "nonestimable_reason": reason,
            "ci_status": ci_status,
            "ci_nonestimable_reason": ci_reason,
        }

    lr = likelihood_ratio_metrics(**confusion)
    correction = lr["zero_cell_correction"] or "none"
    lr_rows = []
    for metric, value_key, low_key, high_key in (
        ("lr_positive", "lr_plus", "lr_plus_ci_low", "lr_plus_ci_high"),
        ("lr_negative", "lr_minus", "lr_minus_ci_low", "lr_minus_ci_high"),
    ):
        lr_rows.append(
            {
                **base,
                "metric": metric,
                "estimate": lr[value_key],
                "ci_low": lr[low_key],
                "ci_high": lr[high_key],
                "ci_level": 0.95 if lr["estimability_status"] == "estimable" else None,
                "ci_method": "log_normal_wald" if lr["estimability_status"] == "estimable" else None,
                **confusion,
                "sensitivity_unadjusted": sensitivity,
                "specificity_unadjusted": specificity,
                "threshold_p_raw": threshold,
                "score_scale": "p_raw",
                "zero_cell_correction": correction,
                "correction_applied": lr["zero_cell_correction"] is not None,
                "estimability_status": lr["estimability_status"],
                "nonestimable_reason": lr["nonestimable_reason"],
                "ci_status": lr["estimability_status"],
                "ci_nonestimable_reason": lr["nonestimable_reason"],
            }
        )
    return [auc_row, threshold_row("sensitivity"), threshold_row("specificity"), *lr_rows]


def _composition_records(
    analysis_id: str,
    view: str,
    population_id: str,
    disease_category: np.ndarray,
) -> list[dict[str, Any]]:
    """Build six fixed, zero-preserving disease composition rows."""
    categories = _validate_disease_category_codes(disease_category)
    if categories.ndim != 1:
        raise ValueError("Disease composition requires a one-dimensional category vector.")
    denominator = len(categories)
    return [
        {
            "analysis_id": analysis_id,
            "view": view,
            "population_id": population_id,
            "disease_category_code": code,
            "disease_category_label": _DISEASE_LABELS[code],
            "outcome_label": "negative" if code < 3 else "positive",
            "n": int(np.sum(categories == code)),
            "denominator_n": denominator,
            "proportion": int(np.sum(categories == code)) / denominator if denominator else None,
        }
        for code in range(6)
    ]


def _within_date_record(view: str, inputs: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Translate an in-memory within-date result into its CSV row contract."""
    result = within_date_concordance(inputs["y_true"], inputs["p_raw"], inputs["dates"], view=view)
    labels = validate_exact_binary_labels_01(inputs["y_true"])
    return {
        "analysis_id": "analysis_2_within_date_concordance",
        "view": view,
        "estimate": result["estimate"],
        "ci_low": result["ci_low"],
        "ci_high": result["ci_high"],
        "ci_level": 0.95 if result["ci_status"] == "estimable" else None,
        "ci_method": result["ci_method"],
        "n": len(labels),
        "n_positive": int(labels.sum()),
        "n_negative": int(len(labels) - labels.sum()),
        "n_date_groups_total": result["n_dates"],
        "n_date_groups_comparable": result["n_comparable_dates"],
        "n_date_groups_noncomparable": result["n_noncomparable_dates"],
        "n_comparable_pairs": result["n_comparable_pairs"],
        "n_bootstrap_valid": result["n_bootstrap_valid"],
        "n_bootstrap_attempts": result["n_bootstrap_attempts"],
        "bootstrap_seed": result["bootstrap_seed"],
        "tie_credit": 0.5,
        "estimability_status": result["estimability_status"],
        "nonestimable_reason": result["nonestimable_reason"],
        "ci_status": result["ci_status"],
        "ci_nonestimable_reason": result["ci_nonestimable_reason"],
    }


def _validate_scientific_records(
    name: str,
    records: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> None:
    """Enforce frozen enums, status dependencies, finiteness, and aggregate arithmetic."""
    def is_missing(value: object) -> bool:
        """Return whether one nullable scientific cell is missing."""
        return value is None or value is pd.NA or (
            isinstance(value, (float, np.floating)) and np.isnan(value)
        )

    def require_populated(record: Mapping[str, Any], fields: Sequence[str]) -> None:
        """Require fields that the frozen contract declares populated."""
        if any(is_missing(record[field]) for field in fields):
            raise ValueError("Scientific record is missing a contract-required field.")

    def require_missing(record: Mapping[str, Any], fields: Sequence[str]) -> None:
        """Require fields that the frozen contract declares absent."""
        if any(not is_missing(record[field]) for field in fields):
            raise ValueError("Scientific record populates a contract-forbidden field.")

    def close(left: float, right: float) -> bool:
        """Compare deterministic aggregate quantities without repair."""
        return bool(np.isclose(left, right, rtol=1e-12, atol=1e-12))

    if not isinstance(records, (list, tuple)):
        raise ValueError("Scientific records must be an ordered record sequence.")
    planned_rows = contract.get("planned_rows")
    if planned_rows is not None:
        if not isinstance(planned_rows, list) or not planned_rows:
            raise ValueError("Scientific planned-row contract is malformed.")
        identity_keys = tuple(key for key in planned_rows[0] if key in contract["columns"])
        expected_keys = [tuple(row.get(key) for key in identity_keys) for row in planned_rows]
        actual_keys = [tuple(record.get(key) for key in identity_keys) for record in records]
        if len(actual_keys) != len(set(actual_keys)):
            raise ValueError("Scientific records contain duplicate planned-row identities.")
        if len(actual_keys) != len(expected_keys):
            raise ValueError("Scientific records do not match the planned-row count.")
        if actual_keys != expected_keys:
            raise ValueError("Scientific records do not match the frozen planned-row order.")
    elif name == "within_date_concordance":
        expected_views = list(contract.get("planned_views", ()))
        actual_views = [record.get("view") for record in records]
        if actual_views != expected_views or any(
            record.get("analysis_id") != "analysis_2_within_date_concordance" for record in records
        ):
            raise ValueError("Within-date records do not match the frozen planned-row order.")
    column_contracts = {item["name"]: item for item in contract.get("columns_exact", ())}
    allowed_reasons = set(contract.get("allowed_nonestimability_reasons", ()))
    for record in records:
        if set(record) != set(contract["columns"]):
            raise ValueError("Scientific record keys do not match the frozen column contract.")
        for column, spec in column_contracts.items():
            value = record[column]
            missing = is_missing(value)
            if missing:
                if not spec.get("nullable", False):
                    raise ValueError("Scientific record contains an impermissible missing value.")
                continue
            logical_dtype = spec["logical_dtype"]
            if logical_dtype == "boolean" and type(value) is not bool and not isinstance(value, np.bool_):
                raise ValueError("Scientific record contains a non-boolean flag value.")
            if logical_dtype == "integer" and (
                type(value) is bool or not isinstance(value, (int, np.integer))
            ):
                raise ValueError("Scientific record contains a non-integer count value.")
            if logical_dtype == "float" and (
                type(value) is bool
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value)
            ):
                raise ValueError("Scientific record contains a nonfinite or nonnumeric value.")
            if logical_dtype in {"string", "category"} and not isinstance(value, str):
                raise ValueError("Scientific record contains a non-string categorical value.")
            permitted = spec.get("permitted_categorical_values")
            if permitted is not None and value not in permitted:
                raise ValueError("Scientific record contains a value outside a frozen enum.")
            minimum = spec.get("range_min_inclusive_when_populated")
            maximum = spec.get("range_max_inclusive_when_populated")
            if minimum is not None and value < minimum or maximum is not None and value > maximum:
                raise ValueError("Scientific record contains a value outside its frozen range.")
        if "estimability_status" in record:
            point_estimable = record["estimability_status"] == "estimable"
            # QC5 outcome/disease composition rows carry their point value in
            # ``proportion``, not ``estimate``; every other record type with an
            # estimability status carries its point value in ``estimate``.
            if name == "qc5_correctness" and record.get("record_type") in {
                "outcome_composition",
                "disease_composition",
            }:
                point_value_populated = not is_missing(record.get("proportion"))
            else:
                point_value_populated = not is_missing(record.get("estimate"))
            if point_estimable != point_value_populated:
                raise ValueError("Point status and its point-value field population disagree.")
            reason = record.get("nonestimable_reason")
            reason_missing = is_missing(reason)
            if point_estimable and not reason_missing:
                raise ValueError("Estimable point records cannot carry a reason.")
            if not point_estimable and reason_missing:
                raise ValueError("Non-estimable point records require a reason.")
            if not reason_missing and allowed_reasons and reason not in allowed_reasons:
                raise ValueError("Point non-estimability reason is outside the frozen registry.")
        if "ci_status" in record:
            ci_status = record["ci_status"]
            ci_values = (record.get("ci_low"), record.get("ci_high"), record.get("ci_level"), record.get("ci_method"))
            if ci_status == "estimable" and any(is_missing(value) for value in ci_values):
                raise ValueError("Estimable CI records require all CI fields.")
            if ci_status != "estimable" and any(not is_missing(value) for value in ci_values):
                raise ValueError("Non-estimable/not-applicable CI records must omit CI fields.")
            ci_reason = record.get("ci_nonestimable_reason")
            ci_reason_missing = is_missing(ci_reason)
            if (
                (ci_status == "estimable" and not ci_reason_missing)
                or (ci_status == "not_applicable" and (ci_reason_missing or ci_reason != "not_applicable"))
                or (ci_status == "non_estimable" and ci_reason_missing)
            ):
                raise ValueError("CI status and reason do not reconcile.")
        for count_name in (
            "n", "n_positive", "n_negative", "tp", "tn", "fp", "fn",
            "numerator_n", "denominator_n", "correct_n", "incorrect_n",
            "n_date_groups_total", "n_date_groups_comparable", "n_date_groups_noncomparable",
            "n_comparable_pairs", "n_bootstrap_valid", "n_bootstrap_attempts",
        ):
            value = record.get(count_name)
            if value is not None and (not isinstance(value, (int, np.integer)) or value < 0):
                raise ValueError("Scientific counts must be nonnegative integers.")
        if {"n", "n_positive", "n_negative"}.issubset(record) and record["n"] != (
            record["n_positive"] + record["n_negative"]
        ):
            raise ValueError("Scientific class counts do not reconcile.")
        if all(record.get(key) is not None for key in ("tp", "tn", "fp", "fn")):
            if record["tp"] + record["fn"] != record["n_positive"] or record["tn"] + record["fp"] != record["n_negative"]:
                raise ValueError("Confusion counts do not reconcile.")
        if record.get("numerator_n") is not None and record.get("denominator_n") is not None:
            expected = record["numerator_n"] / record["denominator_n"] if record["denominator_n"] else None
            if record.get("proportion") != expected:
                raise ValueError("Composition proportion does not reconcile.")
        if name == "locked_score_composition" and record.get("n") is not None and record.get("denominator_n") is not None:
            expected = record["n"] / record["denominator_n"] if record["denominator_n"] else None
            if expected is None:
                valid_proportion = is_missing(record["proportion"])
            else:
                valid_proportion = close(float(record["proportion"]), expected)
            if not valid_proportion:
                raise ValueError("Disease composition proportion does not reconcile.")
        if name == "within_date_concordance":
            if record["n_date_groups_total"] != record["n_date_groups_comparable"] + record["n_date_groups_noncomparable"]:
                raise ValueError("Within-date group counts do not reconcile.")
            if record["n_bootstrap_valid"] > record["n_bootstrap_attempts"] or record["tie_credit"] != 0.5:
                raise ValueError("Within-date bootstrap or tie-credit contract is invalid.")
            if record["ci_status"] == "estimable" and (
                record["n_date_groups_comparable"] < 10
                or record["n_bootstrap_valid"] != 2000
                or record["n_bootstrap_attempts"] != 2000
                or record["bootstrap_seed"] != 42
            ):
                raise ValueError("Estimable within-date CI violates the frozen bootstrap contract.")
            if record["ci_status"] != "estimable" and (
                record["n_bootstrap_valid"] != 0
                or record["n_bootstrap_attempts"] != 0
                or not is_missing(record["bootstrap_seed"])
            ):
                raise ValueError("Non-estimable within-date CI must not claim bootstrap execution.")
    if name == "locked_score_metrics":
        rules = contract["field_population_rules_by_metric"]
        for record in records:
            metric = record["metric"]
            rule = rules[metric]
            require_populated(record, rule["must_be_populated"])
            require_missing(record, rule.get("must_be_missing", ()))
            if record["score_scale"] != rule["score_scale"]:
                raise ValueError("Metric score scale does not match the frozen contract.")
            threshold_rule = rule.get("threshold_p_raw_population")
            if threshold_rule == "must_be_missing":
                require_missing(record, ("threshold_p_raw",))
            elif not close(float(record["threshold_p_raw"]), float(threshold_rule)):
                raise ValueError("Metric threshold does not match the frozen contract.")
            correction = rule.get("zero_cell_correction")
            if correction is not None and record["zero_cell_correction"] != correction:
                raise ValueError("Metric correction category does not match the frozen contract.")
            if rule.get("correction_applied") == "must_be_missing":
                require_missing(record, ("correction_applied",))
            if record["estimability_status"] == "estimable":
                require_populated(record, rule.get("populated_when_point_estimable", ()))
            if record["ci_status"] == "estimable":
                require_populated(record, rule.get("populated_when_ci_estimable", ()))
                if record["ci_method"] != rule["ci_method_when_ci_estimable"] or not close(
                    float(record["ci_level"]), float(rule["ci_level_when_ci_applicable"])
                ):
                    raise ValueError("Metric CI method or level does not match the frozen contract.")
            if not is_missing(record["ci_low"]) and not is_missing(record["ci_high"]):
                if record["ci_low"] > record["ci_high"]:
                    raise ValueError("Metric confidence interval bounds are not ordered.")
                if not is_missing(record["estimate"]) and not (
                    record["ci_low"] <= record["estimate"] <= record["ci_high"]
                ):
                    raise ValueError("Metric estimate lies outside its confidence interval.")
            if record["tp"] is not None:
                # Every threshold-dependent row (sensitivity, specificity, and both
                # LR rows) carries the same uncorrected companion sensitivity and
                # specificity pair, independently derived from tp/n_positive and
                # tn/n_negative -- not only the row whose own metric matches.
                expected_sensitivity = record["tp"] / record["n_positive"] if record["n_positive"] else None
                expected_specificity = record["tn"] / record["n_negative"] if record["n_negative"] else None
                for field, expected in (
                    ("sensitivity_unadjusted", expected_sensitivity),
                    ("specificity_unadjusted", expected_specificity),
                ):
                    actual = record[field]
                    if expected is None:
                        if not is_missing(actual):
                            raise ValueError(
                                "Metric companion proportion is unexpectedly populated for an absent class."
                            )
                    elif is_missing(actual) or not close(float(actual), expected):
                        raise ValueError("Metric companion proportion does not reconcile with its confusion counts.")
            if metric in {"sensitivity", "specificity"}:
                numerator = record["tp"] if metric == "sensitivity" else record["tn"]
                denominator = record["n_positive"] if metric == "sensitivity" else record["n_negative"]
                if denominator:
                    if not close(float(record["estimate"]), numerator / denominator):
                        raise ValueError("Metric proportion estimate does not reconcile with its counts.")
                    wilson_low, wilson_high = wilson_proportion_ci(numerator, denominator)
                    if (
                        not close(float(record["ci_low"]), wilson_low)
                        or not close(float(record["ci_high"]), wilson_high)
                        or record["ci_method"] != "wilson"
                        or not close(float(record["ci_level"]), 0.95)
                    ):
                        raise ValueError("Metric Wilson interval does not reconcile with its counts.")
            if metric in {"lr_positive", "lr_negative"}:
                calculated = likelihood_ratio_metrics(
                    record["tp"], record["fn"], record["tn"], record["fp"]
                )
                expected_key = "lr_plus" if metric == "lr_positive" else "lr_minus"
                expected_low = f"{expected_key}_ci_low"
                expected_high = f"{expected_key}_ci_high"
                if record["estimability_status"] != calculated["estimability_status"]:
                    raise ValueError("Likelihood-ratio estimability status is inconsistent with confusion counts.")
                expected_correction = calculated["zero_cell_correction"] or "none"
                if record["zero_cell_correction"] != expected_correction or record["correction_applied"] != (
                    calculated["zero_cell_correction"] is not None
                ):
                    raise ValueError("Likelihood-ratio correction fields are inconsistent with confusion counts.")
                if calculated["estimability_status"] == "estimable" and not (
                    close(float(record["estimate"]), float(calculated[expected_key]))
                    and close(float(record["ci_low"]), float(calculated[expected_low]))
                    and close(float(record["ci_high"]), float(calculated[expected_high]))
                ):
                    raise ValueError("Likelihood-ratio estimate or interval does not reconcile with frozen arithmetic.")
    if name == "locked_score_composition":
        grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
        for record in records:
            grouped.setdefault(
                (record["analysis_id"], record["view"], record["population_id"]), []
            ).append(record)
        for key, rows in grouped.items():
            if [row["disease_category_code"] for row in rows] != list(range(6)):
                raise ValueError("Disease composition code inventory or order is invalid.")
            denominator = rows[0]["denominator_n"]
            if any(row["denominator_n"] != denominator for row in rows) or sum(row["n"] for row in rows) != denominator:
                raise ValueError("Disease composition counts do not reconcile.")
            if key[0] == "analysis_1_difficult_spectrum":
                if rows[0]["n"] != 0 or rows[5]["n"] != 0:
                    raise ValueError("Difficult-spectrum composition must exclude HC and AGC.")
                if rows[1]["n"] + rows[2]["n"] + rows[3]["n"] + rows[4]["n"] != denominator:
                    raise ValueError(
                        "Difficult-spectrum middle-four composition does not reconcile with its denominator."
                    )
    if name == "qc5_correctness":
        rules = contract["field_population_rules_by_record_type"]

        def require_zero_denominator_contract(record: Mapping[str, Any], rule: Mapping[str, Any]) -> None:
            """Enforce one frozen all-missing/exact-status QC5 empty-group sub-contract."""
            for field, expected in rule.items():
                if field not in record:
                    continue
                if expected == "must_be_missing":
                    if not is_missing(record[field]):
                        raise ValueError("QC5 empty-group row populates a contract-forbidden field.")
                elif record[field] != expected:
                    raise ValueError("QC5 empty-group row does not match its frozen status or reason contract.")

        for record in records:
            rule = rules[record["record_type"]]
            require_missing(record, rule.get("must_be_missing_always", ()))
            if not is_missing(record["ci_low"]) and not is_missing(record["ci_high"]):
                if record["ci_low"] > record["ci_high"]:
                    raise ValueError("QC5 confidence interval bounds are not ordered.")
                if not is_missing(record["estimate"]) and not (
                    record["ci_low"] <= record["estimate"] <= record["ci_high"]
                ):
                    raise ValueError("QC5 estimate lies outside its confidence interval.")
            if record["record_type"] == "accuracy":
                require_populated(record, rule["always_populated_including_zeros"])
                if record["category_type"] != "not_applicable" or record["category"] != "not_applicable":
                    raise ValueError("QC5 accuracy record has an invalid comparison identity.")
                if not close(float(record["threshold_p_raw"]), float(rule["threshold_p_raw"])):
                    raise ValueError("QC5 accuracy threshold does not match the frozen contract.")
                if record["score_scale"] != rule["score_scale"]:
                    raise ValueError("QC5 accuracy score scale does not match the frozen contract.")
                if record["correct_n"] + record["incorrect_n"] != record["denominator_n"]:
                    raise ValueError("QC5 accuracy counts do not reconcile.")
                if record["denominator_n"]:
                    expected_estimate = record["correct_n"] / record["denominator_n"]
                    wilson_low, wilson_high = wilson_proportion_ci(record["correct_n"], record["denominator_n"])
                    if (
                        not close(float(record["estimate"]), expected_estimate)
                        or not close(float(record["ci_low"]), wilson_low)
                        or not close(float(record["ci_high"]), wilson_high)
                        or record["ci_method"] != rule["ci_method_when_estimable"]
                        or not close(float(record["ci_level"]), 0.95)
                        or record["estimability_status"] != "estimable"
                        or record["ci_status"] != "estimable"
                    ):
                        raise ValueError("QC5 accuracy estimate or Wilson interval does not reconcile with its counts.")
                else:
                    if record["correct_n"] != 0 or record["incorrect_n"] != 0:
                        raise ValueError("QC5 zero-denominator accuracy row must retain zero correct/incorrect counts.")
                    require_zero_denominator_contract(record, rule["when_strict_status_denominator_zero"])
            elif record["record_type"] == "accuracy_difference":
                require_populated(record, rule["always_populated"])
                if record["strict_status"] != "not_applicable":
                    raise ValueError("QC5 accuracy-difference record has an invalid strict status.")
                if not close(float(record["threshold_p_raw"]), float(rule["threshold_p_raw"])) or (
                    record["score_scale"] != rule["score_scale"]
                ):
                    raise ValueError("QC5 accuracy-difference threshold or score scale does not match the frozen contract.")
            else:
                require_populated(record, rule["always_populate_counts"])
                if record["ci_status"] != rule["ci_status"] or record["ci_nonestimable_reason"] != rule["ci_nonestimable_reason"]:
                    raise ValueError("QC5 composition CI fields violate the frozen contract.")
                if not is_missing(record["threshold_p_raw"]) or not is_missing(record["score_scale"]):
                    raise ValueError("QC5 composition record populates a contract-forbidden threshold or score scale.")
                if record["denominator_n"]:
                    if not close(float(record["proportion"]), record["numerator_n"] / record["denominator_n"]) or (
                        record["estimability_status"] != "estimable"
                    ):
                        raise ValueError("QC5 composition proportion or status does not reconcile with its counts.")
                else:
                    if record["numerator_n"] != 0:
                        raise ValueError("QC5 zero-denominator composition row must retain a zero numerator.")
                    zero_rule = rule["when_strict_status_denominator_zero"]
                    if (
                        not is_missing(record["proportion"])
                        or record["estimability_status"] != zero_rule["estimability_status"]
                        or record["nonestimable_reason"] != zero_rule["nonestimable_reason"]
                    ):
                        raise ValueError("QC5 zero-denominator composition row does not match the frozen contract.")
        for view in ("oof", "test"):
            for strict_status in ("strict_pass", "strict_fail"):
                rows = [
                    record for record in records
                    if record["view"] == view and record["strict_status"] == strict_status
                ]
                accuracy_rows = [record for record in rows if record["record_type"] == "accuracy"]
                if len(accuracy_rows) != 1:
                    raise ValueError("QC5 strict-status group does not have one corresponding accuracy row.")
                denominator = accuracy_rows[0]["denominator_n"]
                category_rows: dict[str, dict[str, Mapping[str, Any]]] = {}
                for record_type, expected_categories in (
                    ("outcome_composition", ("negative", "positive")),
                    ("disease_composition", _DISEASE_LABELS),
                ):
                    composition = [record for record in rows if record["record_type"] == record_type]
                    if tuple(record["category"] for record in composition) != expected_categories:
                        raise ValueError("QC5 composition category inventory or order is invalid.")
                    if any(record["denominator_n"] != denominator for record in composition):
                        raise ValueError(
                            "QC5 composition denominators do not match the corresponding accuracy denominator."
                        )
                    if sum(record["numerator_n"] for record in composition) != denominator:
                        raise ValueError("QC5 composition counts do not reconcile.")
                    category_rows[record_type] = {record["category"]: record for record in composition}
                # Within-QC5 disease-composition class sums must reconcile with the
                # paired outcome-composition negative/positive counts.
                disease = category_rows["disease_composition"]
                outcome = category_rows["outcome_composition"]
                negative_sum = sum(disease[label]["numerator_n"] for label in ("HC", "BGL", "LGIN"))
                positive_sum = sum(disease[label]["numerator_n"] for label in ("HGIN", "EGC", "AGC"))
                if negative_sum != outcome["negative"]["numerator_n"] or positive_sum != outcome["positive"]["numerator_n"]:
                    raise ValueError("QC5 disease composition class totals disagree with outcome composition counts.")
            difference = next(record for record in records if record["view"] == view and record["record_type"] == "accuracy_difference")
            pass_row = next(
                record
                for record in records
                if record["view"] == view and record["record_type"] == "accuracy" and record["strict_status"] == "strict_pass"
            )
            fail_row = next(
                record
                for record in records
                if record["view"] == view and record["record_type"] == "accuracy" and record["strict_status"] == "strict_fail"
            )
            if pass_row["denominator_n"] and fail_row["denominator_n"]:
                expected = pass_row["estimate"] - fail_row["estimate"]
                newcombe_low, newcombe_high = newcombe_accuracy_difference_ci(
                    pass_row["correct_n"], pass_row["denominator_n"], fail_row["correct_n"], fail_row["denominator_n"]
                )
                if (
                    not close(float(difference["estimate"]), float(expected))
                    or not close(float(difference["ci_low"]), newcombe_low)
                    or not close(float(difference["ci_high"]), newcombe_high)
                    or difference["ci_method"] != rules["accuracy_difference"]["ci_method_when_estimable"]
                    or not close(float(difference["ci_level"]), 0.95)
                    or difference["estimability_status"] != "estimable"
                    or difference["ci_status"] != "estimable"
                ):
                    raise ValueError("QC5 accuracy difference or Newcombe interval does not reconcile with the paired rows.")
            else:
                require_zero_denominator_contract(
                    difference, rules["accuracy_difference"]["when_either_strict_status_group_empty"]
                )
    if name == "within_date_concordance":
        rule = contract["field_population_rules"]
        for record in records:
            if not close(float(record["tie_credit"]), float(rule["tie_credit_value"])):
                raise ValueError("Within-date tie credit does not match the frozen contract.")
            if (record["n_date_groups_comparable"] == 0) != (record["n_comparable_pairs"] == 0):
                # Zero comparable date groups must imply zero comparable pairs, and a
                # positive comparable-pair count requires at least one comparable group.
                raise ValueError("Within-date comparable-group and comparable-pair zero-ness must agree.")
            if record["n_comparable_pairs"] == 0:
                if (
                    record["estimability_status"] != "non_estimable"
                    or record["nonestimable_reason"] != "no_comparable_date_groups"
                    or record["ci_status"] != "non_estimable"
                    or record["ci_nonestimable_reason"] != "no_comparable_date_groups"
                ):
                    raise ValueError("Within-date empty-group status does not match the frozen contract.")
            elif record["n_date_groups_comparable"] < 10:
                if (
                    record["estimability_status"] != "estimable"
                    or record["ci_status"] != "non_estimable"
                    or record["ci_nonestimable_reason"] != "fewer_than_10_comparable_date_groups"
                ):
                    raise ValueError("Within-date sparse-group status does not match the frozen contract.")
            elif (
                record["ci_status"] != "estimable"
                or record["ci_method"] != rule["ci_method_when_estimable"]
                or not close(float(record["ci_level"]), 0.95)
                or record["bootstrap_seed"] != rule["bootstrap_seed_when_G_ge_10"]
                or record["n_bootstrap_valid"] != 2000
                or record["n_bootstrap_attempts"] != 2000
            ):
                raise ValueError("Within-date bootstrap metadata does not match the frozen contract.")
            elif record["ci_low"] > record["ci_high"] or not (
                record["ci_low"] <= record["estimate"] <= record["ci_high"]
            ):
                raise ValueError("Within-date confidence interval bounds are not ordered around the estimate.")


def _validate_cross_output_reconciliation(
    metrics: Sequence[Mapping[str, Any]],
    composition: Sequence[Mapping[str, Any]],
    qc_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Reconcile disease/endpoint-class totals across the four record inventories.

    Args:
        metrics: Complete in-memory locked-score metric records.
        composition: Complete in-memory locked-score composition records.
        qc_rows: Complete in-memory QC5 correctness/composition records.

    Raises:
        ValueError: If any cross-output class or count reconciliation fails. No
            record is silently repaired.
    """
    metric_groups: dict[tuple[Any, Any, Any], list[Mapping[str, Any]]] = {}
    for record in metrics:
        metric_groups.setdefault((record["analysis_id"], record["view"], record["population_id"]), []).append(record)
    composition_groups: dict[tuple[Any, Any, Any], list[Mapping[str, Any]]] = {}
    for record in composition:
        composition_groups.setdefault(
            (record["analysis_id"], record["view"], record["population_id"]), []
        ).append(record)
    for key, rows in composition_groups.items():
        rows_by_code = {row["disease_category_code"]: row for row in rows}
        if set(rows_by_code) != set(range(6)):
            raise ValueError("Cross-output composition group does not carry the frozen six disease codes.")
        metric_rows = metric_groups.get(key)
        if not metric_rows:
            raise ValueError("Locked-score composition group has no corresponding metric group.")
        n_values = {row["n"] for row in metric_rows}
        n_positive_values = {row["n_positive"] for row in metric_rows}
        n_negative_values = {row["n_negative"] for row in metric_rows}
        if len(n_values) != 1 or len(n_positive_values) != 1 or len(n_negative_values) != 1:
            raise ValueError("Locked-score metric group does not carry identical class counts across its rows.")
        metric_n = n_values.pop()
        metric_n_positive = n_positive_values.pop()
        metric_n_negative = n_negative_values.pop()
        denominator = rows[0]["denominator_n"]
        if denominator != metric_n:
            raise ValueError("Locked-score composition denominator disagrees with its metric group n.")
        negative_sum = sum(rows_by_code[code]["n"] for code in range(3))
        positive_sum = sum(rows_by_code[code]["n"] for code in range(3, 6))
        if negative_sum != metric_n_negative or positive_sum != metric_n_positive:
            raise ValueError("Locked-score composition class totals disagree with the metric group endpoint counts.")
        if key[0] == "analysis_1_difficult_spectrum":
            if rows_by_code[0]["n"] != 0 or rows_by_code[5]["n"] != 0:
                raise ValueError("Difficult-spectrum composition must exclude HC and AGC.")
            if rows_by_code[1]["n"] + rows_by_code[2]["n"] != metric_n_negative:
                raise ValueError("Difficult-spectrum negative composition disagrees with its metric group.")
            if rows_by_code[3]["n"] + rows_by_code[4]["n"] != metric_n_positive:
                raise ValueError("Difficult-spectrum positive composition disagrees with its metric group.")
        if key[0] == "analysis_4_strict_qc5" and key[2] == "strict_qc5_pass":
            qc5_disease = {
                record["category"]: record
                for record in qc_rows
                if record["view"] == key[1]
                and record["strict_status"] == "strict_pass"
                and record["record_type"] == "disease_composition"
            }
            if set(qc5_disease) != set(_DISEASE_LABELS):
                raise ValueError("QC5 strict-pass disease composition is incomplete for cross-output reconciliation.")
            for code, label in enumerate(_DISEASE_LABELS):
                if rows_by_code[code]["n"] != qc5_disease[label]["numerator_n"]:
                    raise ValueError(
                        "Locked-score strict-pass disease composition disagrees with the QC5 strict-pass rows."
                    )


def _execution_frames(
    config: Mapping[str, Any],
    views: Mapping[str, Mapping[str, np.ndarray]],
    *,
    on_first_aggregate: Callable[[], None] | None = None,
) -> OrderedDict[str, pd.DataFrame]:
    """Compute the four Session 4 aggregate-only output frames.

    Args:
        config: Effective frozen configuration.
        views: Ordered OOF/test vectors.
        on_first_aggregate: Optional external callback invoked at most once,
            immediately after the first AUROC aggregate is formed.

    Returns:
        The four validated, cross-reconciled aggregate output frames.
    """
    required_views = ("oof", "test")
    if tuple(views) != required_views:
        raise ValueError("Locked-score execution requires ordered OOF and test views.")
    threshold = float(config["threshold"]["threshold_p_raw"])
    first_aggregate_notified = False

    def notify_first_aggregate_once() -> None:
        """Notify the caller at the true first-aggregate boundary at most once."""
        nonlocal first_aggregate_notified
        if first_aggregate_notified:
            return
        first_aggregate_notified = True
        if on_first_aggregate is not None:
            on_first_aggregate()

    difficult_metrics: list[dict[str, Any]] = []
    strict_qc5_metrics: list[dict[str, Any]] = []
    difficult_composition: list[dict[str, Any]] = []
    strict_qc5_composition: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    dates: list[dict[str, Any]] = []
    for view in required_views:
        inputs = views[view]
        required = {"y_true", "p_raw", "dates", "disease_category", "strict_pass"}
        if set(inputs) != required:
            raise ValueError("Each locked-score view must supply exactly five approved vectors.")
        labels = validate_exact_binary_labels_01(inputs["y_true"])
        vectors = validate_exact_aligned_vectors(
            expected_length=len(labels),
            finite_names=("p_raw",),
            p_raw=inputs["p_raw"],
            dates=inputs["dates"],
            disease_category=inputs["disease_category"],
            strict_pass=inputs["strict_pass"],
        )
        difficult_mask = np.isin(vectors["disease_category"], [1, 2, 3, 4])
        strict_mask = vectors["strict_pass"].astype(bool)
        difficult_metrics.extend(
            _metric_records(
                analysis_id="analysis_1_difficult_spectrum",
                view=view,
                population_id="difficult_spectrum",
                labels=labels[difficult_mask],
                scores=vectors["p_raw"][difficult_mask],
                threshold=threshold,
                on_first_aggregate=notify_first_aggregate_once,
            )
        )
        strict_qc5_metrics.extend(
            _metric_records(
                analysis_id="analysis_4_strict_qc5",
                view=view,
                population_id="strict_qc5_pass",
                labels=labels[strict_mask],
                scores=vectors["p_raw"][strict_mask],
                threshold=threshold,
                on_first_aggregate=notify_first_aggregate_once,
            )
        )
        difficult_composition.extend(
            _composition_records(
                "analysis_1_difficult_spectrum",
                view,
                "difficult_spectrum",
                vectors["disease_category"][difficult_mask],
            )
        )
        strict_qc5_composition.extend(
            _composition_records(
                "analysis_4_strict_qc5",
                view,
                "strict_qc5_pass",
                vectors["disease_category"][strict_mask],
            )
        )
        qc_rows.extend(
            strict_qc5_correctness_inventory(
                y_true=labels,
                scores=vectors["p_raw"],
                strict_pass=vectors["strict_pass"],
                disease_category=vectors["disease_category"],
                threshold=threshold,
                view=view,
            )
        )
        dates.append(_within_date_record(view, inputs))
    test_inputs = views["test"]
    metrics = [*difficult_metrics, *strict_qc5_metrics]
    metrics.extend(
        _metric_records(
            analysis_id="frozen_primary_internal_test_lr",
            view="test",
            population_id="frozen_primary_internal_test",
            labels=validate_exact_binary_labels_01(test_inputs["y_true"]),
            scores=np.asarray(test_inputs["p_raw"], dtype=float),
            threshold=threshold,
            on_first_aggregate=notify_first_aggregate_once,
        )[3:]
    )
    composition = [*difficult_composition, *strict_qc5_composition]
    contracts = config["outputs"]["scientific_and_closure"]
    values = OrderedDict(
        [
            ("locked_score_metrics", metrics),
            ("locked_score_composition", composition),
            ("qc5_correctness", qc_rows),
            ("within_date_concordance", dates),
        ]
    )
    registry = config.get("row_order_and_schema_validation_registry", {})
    for name, records in values.items():
        contract = contracts[name]
        expected_columns = list(contract["columns"])
        # Every record must carry exactly the frozen columns; do not reindex to invent columns.
        for record in records:
            if set(record) != set(expected_columns):
                raise ValueError("Session 4 record keys do not exactly match the frozen column contract.")
        _validate_scientific_records(name, records, contract)
    # Cross-output class/count reconciliation runs only after every individual
    # record inventory has passed its own schema/semantic validator, and strictly
    # before any DataFrame construction, staging, or serialization.
    _validate_cross_output_reconciliation(metrics, composition, qc_rows)
    frames: OrderedDict[str, pd.DataFrame] = OrderedDict()
    for name, records in values.items():
        contract = contracts[name]
        expected_columns = list(contract["columns"])
        registry_entry = registry.get(name)
        if registry_entry is not None:
            frame = build_registry_frame(records, registry_entry)
        else:
            frame = pd.DataFrame(
                [{column: record[column] for column in expected_columns} for record in records],
                columns=expected_columns,
            )
        if len(frame) != contract["planned_row_count"]:
            raise ValueError("Session 4 output does not match its frozen row inventory.")
        frames[name] = frame
    return frames


def execute_approved_framework(
    views: Mapping[str, Mapping[str, np.ndarray]],
    *,
    config: Mapping[str, Any] | None = None,
    on_first_aggregate: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Compute complete Session 4 results in memory from approved vectors.

    Args:
        views: Ordered OOF/test vectors, including labels, raw scores, dates,
            disease categories, and historical strict-QC5 status.
        config: Effective frozen config. Omitted only for synthetic unit tests.
        on_first_aggregate: Optional callback invoked once, immediately after the
            first score-derived aggregate is successfully formed, and before any
            later threshold, composition, QC5, or within-date construction can
            fail. Idempotent callbacks may be reused across repeated calls.

    Returns:
        Aggregate-only data frames keyed by the four configured CSV contracts.
    """
    if config is None:
        config = load_effective_posthoc_config(PROJECT_ROOT / BASE_CONFIG_RELATIVE_PATH)
    frames = _execution_frames(config, views, on_first_aggregate=on_first_aggregate)
    return {"frames": frames, "scores_read": True}


def _write_atomic_csv(destination: Path, frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Validate and individually promote one aggregate-only CSV."""
    if destination.exists():
        raise FileExistsError("Refusing to overwrite an existing Session 4 target.")
    if list(frame.columns) != list(columns):
        raise ValueError("CSV columns do not match the frozen ordered contract.")
    if frame.astype("string").apply(
        lambda column: column.str.contains(r"[0-9]{4}(?:-[0-9]{2}-[0-9]{2}|/[0-9]{1,2}/[0-9]{1,2})", regex=True, na=False)
    ).to_numpy().any():
        raise ValueError("Aggregate CSV privacy validation found a prohibited date value.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        read_back = pd.read_csv(temporary, dtype="string", keep_default_na=False)
        if list(read_back.columns) != list(columns) or len(read_back) != len(frame):
            raise ValueError("CSV read-back validation failed.")
        if destination.exists():
            raise FileExistsError("Refusing to overwrite an existing Session 4 target.")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _stage_csv(
    destination: Path,
    frame: pd.DataFrame,
    columns: Sequence[str],
    registry_entry: Mapping[str, Any] | None,
) -> Path:
    """Validate a frame and write it to a destination-local temp without promoting.

    Args:
        destination: Final absent CSV target.
        frame: Validated aggregate frame.
        columns: Frozen ordered column contract.
        registry_entry: Optional CSV registry contract for typed read-back.

    Returns:
        The staged destination-local temporary path.

    Raises:
        FileExistsError: If the destination already exists.
        ValueError: If columns, privacy, or read-back validation fails.
    """
    if destination.exists():
        raise FileExistsError("Refusing to overwrite an existing Session 4 target.")
    if list(frame.columns) != list(columns):
        raise ValueError("CSV columns do not match the frozen ordered contract.")
    if frame.astype("string").apply(
        lambda column: column.str.contains(r"[0-9]{4}(?:-[0-9]{2}-[0-9]{2}|/[0-9]{1,2}/[0-9]{1,2})", regex=True, na=False)
    ).to_numpy().any():
        raise ValueError("Aggregate CSV privacy validation found a prohibited date value.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        if registry_entry is not None:
            if not registry_readback_matches(frame, temporary, registry_entry):
                raise ValueError("CSV read-back validation failed against the registry contract.")
        else:
            read_back = pd.read_csv(temporary, dtype="string", keep_default_na=False)
            if list(read_back.columns) != list(columns) or len(read_back) != len(frame):
                raise ValueError("CSV read-back validation failed.")
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


_ALLOWED_NONESTIMABLE_REASON_CODES = frozenset(
    {
        "absent_outcome_class",
        "insufficient_class_count_for_delong_ci",
        "delong_variance_nonfinite",
        "delong_variance_negative",
        "delong_variance_below_epsilon",
        "nonfinite_lr",
        "invalid_lr_denominator",
        "empty_strict_status_group",
        "no_comparable_date_groups",
        "fewer_than_10_comparable_date_groups",
    }
)


def _deviations_summary_from_frames(frames: Mapping[str, pd.DataFrame]) -> tuple[str, str]:
    """Derive the completion-log deviations summary from validated output frames.

    Args:
        frames: The four validated aggregate output frames.

    Returns:
        A ``(summary_status, reason_codes_csv)`` pair; status is ``none`` only when
        no retained non-estimable row exists.
    """
    reason_codes: set[str] = set()
    for frame in frames.values():
        for status_col, reason_col in (
            ("estimability_status", "nonestimable_reason"),
            ("ci_status", "ci_nonestimable_reason"),
        ):
            if status_col not in frame.columns or reason_col not in frame.columns:
                continue
            mask = frame[status_col].astype("string") == "non_estimable"
            for reason in frame.loc[mask, reason_col].astype("string").tolist():
                if reason is None or reason is pd.NA or reason == "":
                    continue
                if reason not in _ALLOWED_NONESTIMABLE_REASON_CODES:
                    raise ValueError("Encountered a non-estimable reason outside the allowed registry.")
                reason_codes.add(str(reason))
    if not reason_codes:
        return "none", ""
    return "present_retained", ",".join(sorted(reason_codes))


def _completion_log(
    *,
    freeze_commit: str,
    output_hashes: Sequence[tuple[str, str]],
    tracker: AccessStateTracker,
    deviations: tuple[str, str],
) -> str:
    """Create the configured tab-separated successful completion receipt."""
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    summary_status, reason_codes_csv = deviations
    records: list[tuple[str, ...]] = [
        ("header", "posthoc_final_completion_log.v1", "posthoc_final_supplementary", now),
        ("session_identity", "locked_scores", "complete"),
        ("freeze_commit", freeze_commit),
    ]
    records.extend(("output_hash", path, digest) for path, digest in output_hashes)
    records.append(("deviations_or_nonestimability_summary", summary_status, reason_codes_csv, ""))
    state = tracker.snapshot()
    records.append(
        ("access_state_session_activity", *(str(value).lower() for value in state["session_activity"].values()))
    )
    records.append(
        (
            "access_state_cumulative_through_session",
            *(str(value).lower() for value in state["cumulative_through_session"].values()),
        )
    )
    records.append(("completion_receipt", now, "complete"))
    return "\n".join("\t".join(record) for record in records) + "\n"


def _load_approved_views(config: Mapping[str, Any], tracker: AccessStateTracker) -> OrderedDict[str, Mapping[str, np.ndarray]]:
    """Load the minimal approved Session 4 inputs after authority validation."""
    loaders = config["loaders"]["per_purpose_ordered_usecols"]
    prediction_paths = OrderedDict(
        [
            ("oof", PROJECT_ROOT / "outputs/predictions/binary_minirocket_ridge_nested_cv_oof.csv"),
            ("test", PROJECT_ROOT / "outputs/predictions/binary_minirocket_ridge_test_predictions.csv"),
        ]
    )
    predictions: OrderedDict[str, pd.DataFrame] = OrderedDict()
    for view, path in prediction_paths.items():
        header = list(pd.read_csv(path, nrows=0).columns)
        if header != config["frozen_inputs"]["prediction_static_headers"][str(path.relative_to(PROJECT_ROOT))]:
            raise ValueError("Prediction CSV header does not match the frozen contract.")
        tracker.mark("prediction_schema_or_header_inspected")
        predictions[view] = load_csv_with_ordered_usecols(
            path,
            loaders["locked_score_execution_oof_or_test"],
            loaders["locked_score_execution_oof_or_test"],
        )
        tracker.mark("participant_level_label_values_read")
        tracker.mark("participant_level_score_values_read")
    lineage = load_csv_with_ordered_usecols(
        PROJECT_ROOT / "data/processed/sample_index.csv",
        ["sample_id", "label", "disease_category", "collection_time", "in_final_output"],
        loaders["prediction_preflight_sample_index_lineage"],
    )
    tracker.mark("participant_level_label_values_read")
    tracker.mark("participant_level_date_values_read")
    diagnostics = load_csv_with_ordered_usecols(
        PROJECT_ROOT / "data/processed/qc_reports/qc_diagnostics.csv",
        loaders["qc5_diagnostics"],
        loaders["qc5_diagnostics"],
    )
    tracker.mark("participant_level_qc_values_read")
    summary = load_csv_with_ordered_usecols(
        PROJECT_ROOT / "data/processed/qc_reports/qc_summary.csv",
        loaders["qc5_summary"],
        loaders["qc5_summary"],
    )
    tracker.mark("participant_level_qc_values_read")
    qc5 = reconstruct_qc5(diagnostics, summary)
    strict = pd.DataFrame({"sample_id": qc5["sample_id"], "strict_pass": qc5["strict_pass"]})
    result: OrderedDict[str, Mapping[str, np.ndarray]] = OrderedDict()
    for view, prediction in predictions.items():
        joined = prediction.merge(
            lineage[["sample_id", "label", "disease_category", "collection_time"]],
            on="sample_id",
            how="inner",
            validate="one_to_one",
            sort=False,
        ).merge(strict, on="sample_id", how="inner", validate="one_to_one", sort=False)
        if len(joined) != len(prediction) or not np.array_equal(
            joined["label"].to_numpy(dtype=int), joined["y_true"].to_numpy(dtype=int)
        ):
            raise ValueError("Locked-score execution alignment failed.")
        result[view] = {
            "y_true": joined["y_true"].to_numpy(),
            "p_raw": joined["p_raw"].to_numpy(),
            "dates": parse_date_series_ascii_slash(joined["collection_time"].tolist()),
            "disease_category": joined["disease_category"].to_numpy(),
            "strict_pass": joined["strict_pass"].to_numpy(),
        }
    return result


def execute_approved(
    config_path: Path,
    freeze_commit: str,
    amendment_freeze_commit: str,
    *,
    git_runner: Callable[..., str] = _git,
) -> tuple[Path, ...]:
    """Run the actual approved Session 4 pipeline and promote outputs last.

    Args:
        config_path: Immutable base config path.
        freeze_commit: Session 0 freeze SHA.
        amendment_freeze_commit: Amendment freeze SHA.
        git_runner: Read-only Git adapter.

    Returns:
        Promoted four CSV paths followed by the completion-log path.

    Raises:
        ValueError: If execution authorization, authority, validation, or integrity fails.
    """
    # The machine gate for Session 4 execution is a prior validated score-blind
    # preflight (verified inside validate_pre_access_authority(mode="execute")),
    # historical dual-freeze validation, current-authority validation, and absent
    # targets. An independent external Session 4 "go" remains a human gate.
    config, entry_hashes, authority = validate_pre_access_authority(
        config_path=config_path,
        freeze_commit=freeze_commit,
        amendment_freeze_commit=amendment_freeze_commit,
        mode="execute",
        git_runner=git_runner,
    )
    cumulative_prior = OrderedDict(
        authority.get("preflight_cumulative", ((key, False) for key in ACCESS_STATE_KEYS))
    )
    tracker = AccessStateTracker(cumulative_start=cumulative_prior)
    tracker.mark("prediction_container_bytes_hashed")

    def _mark_first_aggregate() -> None:
        """Mark the true boundary the instant the first real aggregate forms."""
        tracker.mark("new_score_derived_results_computed")

    try:
        views = _load_approved_views(config, tracker)
        first = execute_approved_framework(
            views, config=config, on_first_aggregate=_mark_first_aggregate
        )["frames"]
        second = execute_approved_framework(
            views, config=config, on_first_aggregate=_mark_first_aggregate
        )["frames"]
    except Exception as exc:
        raise AccessStateError("Approved execution failed closed before promotion.", tracker.snapshot()) from exc
    if set(first) != set(second) or any(
        first[name].to_csv(index=False) != second[name].to_csv(index=False) for name in first
    ):
        raise AccessStateError(
            "Approved execution failed closed before promotion.",
            tracker.snapshot(),
        )
    try:
        _verify_exit_hashes(config, entry_hashes)
    except Exception as exc:
        raise AccessStateError(
            "Approved execution failed closed before promotion.",
            tracker.snapshot(),
        ) from exc
    contracts = config["outputs"]["scientific_and_closure"]
    registry = config.get("row_order_and_schema_validation_registry", {})
    log_path = PROJECT_ROOT / _locked_scores_log_relative_path(config)
    log_temporary = log_path.with_name(f".{log_path.name}.tmp")
    staged: OrderedDict[str, tuple[Path, Path]] = OrderedDict()
    promoted: list[Path] = []
    try:
        # Stage and read-back validate every candidate before any promotion.
        for name, frame in first.items():
            destination = PROJECT_ROOT / contracts[name]["path"]
            temporary = _stage_csv(destination, frame, contracts[name]["columns"], registry.get(name))
            staged[name] = (destination, temporary)
        # Prepare and validate the completion log against the promoted-output hashes.
        output_hashes = [
            (str(destination.relative_to(PROJECT_ROOT)), sha256_file(temporary))
            for destination, temporary in staged.values()
        ]
        deviations = _deviations_summary_from_frames(first)
        log_text = _completion_log(
            freeze_commit=freeze_commit,
            output_hashes=output_hashes,
            tracker=tracker,
            deviations=deviations,
        )
        completion_logs = config["outputs"].get("completion_logs", {})
        grammar = (
            completion_logs.get("machine_validated_log_grammar") if isinstance(completion_logs, Mapping) else None
        )
        if grammar is not None:
            binding = dict(grammar["session_bindings_exact"]["locked_scores"])
            binding.setdefault("session_name", "locked_scores")
            validate_completion_log(
                log_text,
                grammar=grammar,
                session_binding=binding,
                expected_freeze_commit=freeze_commit,
                expected_activity=tracker.snapshot()["session_activity"],
                expected_cumulative=tracker.snapshot()["cumulative_through_session"],
                cumulative_prior=cumulative_prior,
                allowed_reason_codes=tuple(_ALLOWED_NONESTIMABLE_REASON_CODES),
            )
        if log_path.exists():
            raise FileExistsError("Refusing to overwrite an existing completion log.")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_temporary.write_text(log_text, encoding="utf-8")
        # Promote CSVs individually, then the completion log last.
        for name, (destination, temporary) in staged.items():
            if destination.exists():
                raise FileExistsError("Refusing to overwrite an existing Session 4 target.")
            os.replace(temporary, destination)
            promoted.append(destination)
        os.replace(log_temporary, log_path)
        promoted.append(log_path)
    except Exception as exc:
        for path in promoted:
            path.unlink(missing_ok=True)
        for _destination, temporary in staged.values():
            temporary.unlink(missing_ok=True)
        log_temporary.unlink(missing_ok=True)
        raise AccessStateError(
            "Approved execution failed closed before promotion.",
            tracker.snapshot(),
        ) from exc
    return tuple(promoted)
