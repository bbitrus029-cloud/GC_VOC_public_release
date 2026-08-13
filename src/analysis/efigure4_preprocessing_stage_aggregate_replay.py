"""Deterministic aggregate-only replay of frozen preprocessing stages."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
import copy
import hashlib
from dataclasses import dataclass
from enum import Enum, unique
import io
import json
import logging
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import scipy
import yaml

from src.config import RAW_CSV_ENCODING
from src.data.correction import apply_fractional_correction, trim_response_segment
from src.data.env_compensation import EnvCompensationModel
from src.data.smoothing import apply_sg_smoothing


ANALYSIS_ID = "efigure4_preprocessing_stage_aggregate_replay"
ANALYSIS_CLASSIFICATION = "post_freeze_result_blind_plan_orthogonal_descriptive_reporting_support"
APPROVED_PLAN_SHA256 = "302f96e626f60463740330a38bcc1c5962543cf5c4f84ee7a32ba942b6d3529d"
ORIGINAL_PLAN_SHA256 = "33381038e69e153a7b4cf401535c50b8a5123de78d7d2a16644804d6a532515e"
RECOVERY_V2_PLAN_SHA256 = "feacd76d706f9c19c632d4336efa42a17691b1a4fbe40b75759daabf8369eebd"
RECOVERY_V3_PLAN_SHA256 = "487e5e69b5282dc2188e4b85467f736546491527076351411654125e5e4d57ca"
REPORTING_CORRECTION_SHA256 = "ba0ea1b226a94ea476744ec7e8f6b047b546634d766e4bdca01792dbbca8a15c"
NUMERICAL_ENVIRONMENT_CORRECTION_SHA256 = "225aff9168b0e4f13e25cbbf6aa5f66ddc151e82a0e28e056cc0d307e917f83b"
CHANNEL_METADATA_CORRECTION_SHA256 = "a1d208e2a2d53faaa931ffa4f99b95dc9690e62a7ff4c55704210d293612f91b"
ENVIRONMENT_LOCK_SHA256 = "cf3e23de96b76536d6b906e24c2ffbaeb767a70d0940c2d483ac4f2200706cad"
APPROVED_PLAN_PATH = "docs/plans/efigure4_preprocessing_stage_aggregate_replay_recovery_v4_plan.md"
ORIGINAL_PLAN_PATH = "docs/plans/efigure4_preprocessing_stage_aggregate_replay_results_free_plan.md"
RECOVERY_V2_PLAN_PATH = "docs/plans/efigure4_preprocessing_stage_aggregate_replay_recovery_v2_plan.md"
RECOVERY_V3_PLAN_PATH = "docs/plans/efigure4_preprocessing_stage_aggregate_replay_recovery_v3_plan.md"
REPORTING_CORRECTION_PATH = "docs/preprocessing/preprocessing_fallback_denominator_reporting_correction.md"
NUMERICAL_ENVIRONMENT_CORRECTION_PATH = (
    "docs/preprocessing/preprocessing_stage4_numerical_environment_build_drift_correction.md"
)
CHANNEL_METADATA_CORRECTION_PATH = (
    "docs/preprocessing/preprocessing_stage4_conda_explicit_archive_channel_metadata_correction.md"
)
ENVIRONMENT_LOCK_PATH = "configs/efigure4_preprocessing_stage_aggregate_replay_environment_lock.json"
REPORTING_CORRECTION_BYTE_SIZE = 3374
CONFIG_ANOMALY_STATUS = "unresolved local SHA-256 computation/provenance anomaly"
EXPECTED_STAGE_IDS = (
    "raw_response_window",
    "savitzky_golay_smoothed_response_window",
    "environment_compensated_response_window",
    "fractional_baseline_corrected_response_window",
)
EXPECTED_STAGE_UNITS = (
    "recorded_signal_unit",
    "recorded_signal_unit",
    "recorded_signal_unit",
    "fractional_change_from_compensated_baseline",
)
EXPECTED_CHANNELS = tuple(f"S{index}" for index in range(1, 12))
EXPECTED_RAW_HEADER = tuple(f"S{index}" for index in range(14))
ENVIRONMENT_CHANNELS = ("S12", "S13")
RAW_SIGNAL_USECOLS = [*EXPECTED_CHANNELS, *ENVIRONMENT_CHANNELS]
SAMPLE_INDEX_USECOLS = [
    "sample_id",
    "filename",
    "n_rows",
    "fallback_used",
    "padded",
    "in_final_output",
]
QC_DIAGNOSTICS_USECOLS = [
    "sample_id",
    "n_rows",
    "t_baseline_end",
    "fallback_used",
    "env_temperature_sample",
    "env_humidity_sample",
    "n_available",
    "clip_length_L",
    "relative_deficit",
    "padded",
]
AGGREGATE_COLUMNS = [
    "stage_order",
    "stage_id",
    "channel",
    "response_frame",
    "unit",
    "n_participants",
    "q25",
    "median",
    "q75",
]
FORBIDDEN_OUTPUT_FIELDS = (
    "sample_id",
    "participant_id",
    "participant_key",
    "raw_filename",
    "filename",
    "per_sample_path",
    "per_sample_hash",
    "raw_path",
    "collection_date",
    "collection_time",
    "collection_location",
    "label",
    "outcome",
    "disease_category",
    "clinical_variable",
    "age",
    "sex",
    "bmi",
    "smoking",
    "drinking",
    "hp_infection",
    "cea",
    "ca199",
    "ca724",
    "split",
    "split_membership",
    "participant_level_array",
    "participant_level_rows",
)
EXPECTED_ALLOWED_WORKTREE_ENTRIES = (
    ".vscode/settings.json",
    "configs/efigure4_preprocessing_stage_aggregate_replay.yaml",
    ENVIRONMENT_LOCK_PATH,
    REPORTING_CORRECTION_PATH,
    NUMERICAL_ENVIRONMENT_CORRECTION_PATH,
    CHANNEL_METADATA_CORRECTION_PATH,
    "scripts/run_efigure4_preprocessing_stage_aggregate_replay.py",
    "src/analysis/efigure4_preprocessing_stage_aggregate_replay.py",
    "tests/test_efigure4_preprocessing_stage_aggregate_replay.py",
)
FINAL_OUTPUT_DIRECTORY = "outputs/analysis/efigure4_preprocessing_stage_aggregate_replay"
CANDIDATE_ONE_DIRECTORY = "outputs/analysis/.efigure4_preprocessing_stage_aggregate_replay.candidate-one"
CANDIDATE_TWO_DIRECTORY = "outputs/analysis/.efigure4_preprocessing_stage_aggregate_replay.candidate-two"
ENVIRONMENT_LOCK_TOP_LEVEL_KEYS = {
    "schema_version",
    "platform",
    "subdirectory",
    "environment_role",
    "environment_name",
    "source_ledger_sha256",
    "final_ledger_sha256",
    "explicit_local_archive_channel_semantics",
    "packages",
    "critical_numerical_builds",
    "critical_cached_archives",
}
ENVIRONMENT_LOCK_SOURCE_LEDGER_SHA256 = "e6a91ad784c41e2781c03947562f6bbf78d6cc33a8d0551245cb7eb9c54e04f8"
ENVIRONMENT_LOCK_FINAL_LEDGER_SHA256 = "7b11f59909a6e2f5b1bf1f8f49cba060b2e0b3d3e1b2b55f0da2eecfbcb30334"
ENVIRONMENT_LOCK_CHANNEL_SEMANTICS = (
    "unknown_channel_expected_only_for_six_explicit_local_sha256_locked_archives"
)
ENVIRONMENT_PACKAGE_KEYS = ("name", "version", "build", "channel", "subdirectory")
EXPECTED_HISTORICAL_SUBSTITUTIONS = {
    "scipy": ("1.17.1", "py311hbe70eeb_0", "<unknown>", "linux-64"),
    "libblas": ("3.11.0", "6_h5875eb1_mkl", "<unknown>", "linux-64"),
    "libcblas": ("3.11.0", "6_hfef963f_mkl", "<unknown>", "linux-64"),
    "liblapack": ("3.11.0", "6_h5e43f62_mkl", "<unknown>", "linux-64"),
    "mkl": ("2025.3.1", "h0e700b2_11", "<unknown>", "linux-64"),
    "onemkl-license": ("2025.3.1", "hf2ce2f3_11", "<unknown>", "linux-64"),
}
EXPECTED_CRITICAL_NUMERICAL_NAMES = (
    "python",
    "numpy",
    "pandas",
    "pyyaml",
    "scipy",
    "libblas",
    "libcblas",
    "liblapack",
    "mkl",
)
EXPECTED_CRITICAL_ARCHIVES = (
    ("libblas-3.11.0-6_h5875eb1_mkl.conda", "a73ec64c0f60a7733f82a679342bdad88e0230ba8243b12ece13a23aded431f4"),
    ("libcblas-3.11.0-6_hfef963f_mkl.conda", "d98a39a8e61af301bf67bf3fb946baff9686864886560cdd48d5259c080c58a5"),
    ("liblapack-3.11.0-6_h5e43f62_mkl.conda", "8715428e721a63880d4e548375a744f177200a5161aec3ebe533f33eaf7ec3a5"),
    ("mkl-2025.3.1-h0e700b2_11.conda", "e03877a111b613dff7b599fd7146def11eb3517f3813e80aedddeb00843f0bf2"),
    (
        "onemkl-license-2025.3.1-hf2ce2f3_11.conda",
        "857bfc5bfbb3956e56a1db3f73f8850c95ef2d962d78471bfdc5de2319149fdb",
    ),
    ("scipy-1.17.1-py311hbe70eeb_0.conda", "0063a75e9406357ef8ecc2fb38808a2442b269bfc5a854cf2bcd368060d9a2d7"),
)


class RawSignalLoadError(ValueError):
    """Sanitized outer error for participant-capable raw-signal loading."""


@unique
class FailureCode(str, Enum):
    """Closed aggregate-safe execution failure taxonomy."""

    AUTHORIZATION_PLAN_IMPLEMENTATION_HASH_VALIDATION_FAILURE = (
        "AUTHORIZATION_PLAN_IMPLEMENTATION_HASH_VALIDATION_FAILURE"
    )
    GIT_AUTHORITY_OUTPUT_STATE_VALIDATION_FAILURE = "GIT_AUTHORITY_OUTPUT_STATE_VALIDATION_FAILURE"
    ENVIRONMENT_IDENTITY_VALIDATION_FAILURE = "ENVIRONMENT_IDENTITY_VALIDATION_FAILURE"
    FROZEN_SOURCE_IDENTITY_CAPTURE_FAILURE = "FROZEN_SOURCE_IDENTITY_CAPTURE_FAILURE"
    FIRST_MINIMAL_COHORT_TABLE_LOADING_ALIGNMENT_FAILURE = "FIRST_MINIMAL_COHORT_TABLE_LOADING_ALIGNMENT_FAILURE"
    FIRST_RAW_COHORT_ENTRY_DIGEST_FAILURE = "FIRST_RAW_COHORT_ENTRY_DIGEST_FAILURE"
    FIRST_ENVIRONMENTAL_MODEL_DESERIALIZATION_FAILURE = "FIRST_ENVIRONMENTAL_MODEL_DESERIALIZATION_FAILURE"
    FIRST_RAW_SIGNAL_CONTAINMENT_LOADING_FAILURE = "FIRST_RAW_SIGNAL_CONTAINMENT_LOADING_FAILURE"
    FIRST_ENVIRONMENTAL_SUMMARY_RECONCILIATION_FAILURE = (
        "FIRST_ENVIRONMENTAL_SUMMARY_RECONCILIATION_FAILURE"
    )
    FIRST_SAVITZKY_GOLAY_SMOOTHING_FAILURE = "FIRST_SAVITZKY_GOLAY_SMOOTHING_FAILURE"
    FIRST_ENVIRONMENTAL_COMPENSATION_FAILURE = "FIRST_ENVIRONMENTAL_COMPENSATION_FAILURE"
    FIRST_FRACTIONAL_BASELINE_CORRECTION_FAILURE = "FIRST_FRACTIONAL_BASELINE_CORRECTION_FAILURE"
    FIRST_COMMON_TRIMMING_PADDING_RECONCILIATION_FAILURE = (
        "FIRST_COMMON_TRIMMING_PADDING_RECONCILIATION_FAILURE"
    )
    FIRST_FROZEN_PIPELINE_A_LOADING_FAILURE = "FIRST_FROZEN_PIPELINE_A_LOADING_FAILURE"
    FIRST_STAGE4_RECONCILIATION_FAILURE = "FIRST_STAGE4_RECONCILIATION_FAILURE"
    FIRST_AGGREGATION_FAILURE = "FIRST_AGGREGATION_FAILURE"
    FIRST_RAW_COHORT_EXIT_DIGEST_FAILURE = "FIRST_RAW_COHORT_EXIT_DIGEST_FAILURE"
    SECOND_MINIMAL_COHORT_TABLE_LOADING_ALIGNMENT_FAILURE = "SECOND_MINIMAL_COHORT_TABLE_LOADING_ALIGNMENT_FAILURE"
    SECOND_RAW_COHORT_ENTRY_DIGEST_FAILURE = "SECOND_RAW_COHORT_ENTRY_DIGEST_FAILURE"
    SECOND_ENVIRONMENTAL_MODEL_DESERIALIZATION_FAILURE = "SECOND_ENVIRONMENTAL_MODEL_DESERIALIZATION_FAILURE"
    SECOND_RAW_SIGNAL_CONTAINMENT_LOADING_FAILURE = "SECOND_RAW_SIGNAL_CONTAINMENT_LOADING_FAILURE"
    SECOND_ENVIRONMENTAL_SUMMARY_RECONCILIATION_FAILURE = (
        "SECOND_ENVIRONMENTAL_SUMMARY_RECONCILIATION_FAILURE"
    )
    SECOND_SAVITZKY_GOLAY_SMOOTHING_FAILURE = "SECOND_SAVITZKY_GOLAY_SMOOTHING_FAILURE"
    SECOND_ENVIRONMENTAL_COMPENSATION_FAILURE = "SECOND_ENVIRONMENTAL_COMPENSATION_FAILURE"
    SECOND_FRACTIONAL_BASELINE_CORRECTION_FAILURE = "SECOND_FRACTIONAL_BASELINE_CORRECTION_FAILURE"
    SECOND_COMMON_TRIMMING_PADDING_RECONCILIATION_FAILURE = (
        "SECOND_COMMON_TRIMMING_PADDING_RECONCILIATION_FAILURE"
    )
    SECOND_FROZEN_PIPELINE_A_LOADING_FAILURE = "SECOND_FROZEN_PIPELINE_A_LOADING_FAILURE"
    SECOND_STAGE4_RECONCILIATION_FAILURE = "SECOND_STAGE4_RECONCILIATION_FAILURE"
    SECOND_AGGREGATION_FAILURE = "SECOND_AGGREGATION_FAILURE"
    SECOND_RAW_COHORT_EXIT_DIGEST_FAILURE = "SECOND_RAW_COHORT_EXIT_DIGEST_FAILURE"
    CROSS_COMPUTATION_DETERMINISM_RAW_DIGEST_COMPARISON_FAILURE = (
        "CROSS_COMPUTATION_DETERMINISM_RAW_DIGEST_COMPARISON_FAILURE"
    )
    FINAL_SOURCE_IDENTITY_AUTHORITY_REVALIDATION_FAILURE = (
        "FINAL_SOURCE_IDENTITY_AUTHORITY_REVALIDATION_FAILURE"
    )
    AUDIT_PAYLOAD_CONSTRUCTION_FAILURE = "AUDIT_PAYLOAD_CONSTRUCTION_FAILURE"
    SERIALIZED_PRIVACY_COMPLETE_READ_BACK_VALIDATION_FAILURE = (
        "SERIALIZED_PRIVACY_COMPLETE_READ_BACK_VALIDATION_FAILURE"
    )
    FIRST_CANDIDATE_WRITING_VALIDATION_FAILURE = "FIRST_CANDIDATE_WRITING_VALIDATION_FAILURE"
    SECOND_CANDIDATE_WRITING_VALIDATION_FAILURE = "SECOND_CANDIDATE_WRITING_VALIDATION_FAILURE"
    CANDIDATE_COMPARISON_FAILURE = "CANDIDATE_COMPARISON_FAILURE"
    ATOMIC_PROMOTION_FINAL_DIRECTORY_VALIDATION_FAILURE = "ATOMIC_PROMOTION_FINAL_DIRECTORY_VALIDATION_FAILURE"
    UNCLASSIFIED_FAILURE = "UNCLASSIFIED_FAILURE"


class ReplayExecutionError(RuntimeError):
    """Carry exactly one fixed failure code while chaining private causes."""

    def __init__(self, failure_code: FailureCode) -> None:
        """Initialize an execution error from the closed code enum.

        Args:
            failure_code: One member of the immutable failure-code taxonomy.

        Raises:
            TypeError: If a caller attempts to supply a dynamic code.
        """
        if not isinstance(failure_code, FailureCode):
            raise TypeError("failure_code must be a FailureCode member.")
        super().__init__("Aggregate replay execution failed.")
        self._failure_code = failure_code

    @property
    def failure_code(self) -> FailureCode:
        """Return the validated static failure code."""
        return self._failure_code


@dataclass(frozen=True)
class IsolatedFailureCodes:
    """Static failure-code bundle for one isolated computation."""

    minimal_cohort: FailureCode
    raw_digest_entry: FailureCode
    environmental_model: FailureCode
    raw_signal: FailureCode
    environmental_summary: FailureCode
    smoothing: FailureCode
    environmental_compensation: FailureCode
    fractional_correction: FailureCode
    trimming: FailureCode
    pipeline_a: FailureCode
    stage4: FailureCode
    aggregation: FailureCode
    raw_digest_exit: FailureCode


FIRST_ISOLATED_FAILURE_CODES = IsolatedFailureCodes(
    minimal_cohort=FailureCode.FIRST_MINIMAL_COHORT_TABLE_LOADING_ALIGNMENT_FAILURE,
    raw_digest_entry=FailureCode.FIRST_RAW_COHORT_ENTRY_DIGEST_FAILURE,
    environmental_model=FailureCode.FIRST_ENVIRONMENTAL_MODEL_DESERIALIZATION_FAILURE,
    raw_signal=FailureCode.FIRST_RAW_SIGNAL_CONTAINMENT_LOADING_FAILURE,
    environmental_summary=FailureCode.FIRST_ENVIRONMENTAL_SUMMARY_RECONCILIATION_FAILURE,
    smoothing=FailureCode.FIRST_SAVITZKY_GOLAY_SMOOTHING_FAILURE,
    environmental_compensation=FailureCode.FIRST_ENVIRONMENTAL_COMPENSATION_FAILURE,
    fractional_correction=FailureCode.FIRST_FRACTIONAL_BASELINE_CORRECTION_FAILURE,
    trimming=FailureCode.FIRST_COMMON_TRIMMING_PADDING_RECONCILIATION_FAILURE,
    pipeline_a=FailureCode.FIRST_FROZEN_PIPELINE_A_LOADING_FAILURE,
    stage4=FailureCode.FIRST_STAGE4_RECONCILIATION_FAILURE,
    aggregation=FailureCode.FIRST_AGGREGATION_FAILURE,
    raw_digest_exit=FailureCode.FIRST_RAW_COHORT_EXIT_DIGEST_FAILURE,
)
SECOND_ISOLATED_FAILURE_CODES = IsolatedFailureCodes(
    minimal_cohort=FailureCode.SECOND_MINIMAL_COHORT_TABLE_LOADING_ALIGNMENT_FAILURE,
    raw_digest_entry=FailureCode.SECOND_RAW_COHORT_ENTRY_DIGEST_FAILURE,
    environmental_model=FailureCode.SECOND_ENVIRONMENTAL_MODEL_DESERIALIZATION_FAILURE,
    raw_signal=FailureCode.SECOND_RAW_SIGNAL_CONTAINMENT_LOADING_FAILURE,
    environmental_summary=FailureCode.SECOND_ENVIRONMENTAL_SUMMARY_RECONCILIATION_FAILURE,
    smoothing=FailureCode.SECOND_SAVITZKY_GOLAY_SMOOTHING_FAILURE,
    environmental_compensation=FailureCode.SECOND_ENVIRONMENTAL_COMPENSATION_FAILURE,
    fractional_correction=FailureCode.SECOND_FRACTIONAL_BASELINE_CORRECTION_FAILURE,
    trimming=FailureCode.SECOND_COMMON_TRIMMING_PADDING_RECONCILIATION_FAILURE,
    pipeline_a=FailureCode.SECOND_FROZEN_PIPELINE_A_LOADING_FAILURE,
    stage4=FailureCode.SECOND_STAGE4_RECONCILIATION_FAILURE,
    aggregation=FailureCode.SECOND_AGGREGATION_FAILURE,
    raw_digest_exit=FailureCode.SECOND_RAW_COHORT_EXIT_DIGEST_FAILURE,
)
_ACTIVE_ISOLATED_FAILURE_CODES: ContextVar[IsolatedFailureCodes | None] = ContextVar(
    "efigure4_active_isolated_failure_codes",
    default=None,
)


@contextmanager
def _failure_boundary(failure_code: FailureCode) -> Iterator[None]:
    """Chain any internal exception to one fixed aggregate-safe code."""
    try:
        yield
    except ReplayExecutionError:
        raise
    except Exception as exc:
        raise ReplayExecutionError(failure_code) from exc


@contextmanager
def _optional_isolated_failure_boundary(failure_code: FailureCode | None) -> Iterator[None]:
    """Apply an isolated code only during observed authorized orchestration."""
    if failure_code is None:
        yield
        return
    with _failure_boundary(failure_code):
        yield


@contextmanager
def _isolated_failure_scope(failure_codes: IsolatedFailureCodes) -> Iterator[None]:
    """Select the immutable first- or second-computation failure bundle."""
    token = _ACTIVE_ISOLATED_FAILURE_CODES.set(failure_codes)
    try:
        yield
    finally:
        _ACTIVE_ISOLATED_FAILURE_CODES.reset(token)


@dataclass(frozen=True)
class CohortTables:
    """Minimal aligned audit state retained only in memory.

    Attributes:
        final_rows: Canonically ordered final-cohort fields needed for replay.
        source_count: Number of aligned source-audit rows.
        final_count: Number of frozen final-cohort rows.
    """

    final_rows: pd.DataFrame
    source_count: int
    final_count: int


@dataclass(frozen=True)
class RawSignal:
    """Minimal raw S1-S13 signal payload with no participant identifiers.

    Attributes:
        gas_signals: S1-S11 matrix in frozen channel order.
        temperature: Recorded S12 vector.
        humidity: Recorded S13 vector.
        n_rows: Number of raw signal rows.
    """

    gas_signals: NDArray[np.float64]
    temperature: NDArray[np.float64]
    humidity: NDArray[np.float64]
    n_rows: int


@dataclass(frozen=True)
class ParticipantReplay:
    """Four aligned participant-stage windows held only in memory.

    Attributes:
        stages: Exact stage ID to `(clip_length, 11)` replay window.
        n_available: Response rows available before trimming or padding.
        padded: Whether the common last-row tail padding rule was used.
    """

    stages: dict[str, NDArray[np.float64]]
    n_available: int
    padded: bool


@dataclass(frozen=True)
class FrozenPipelineA:
    """Only the four permitted arrays from the frozen Pipeline A archive.

    Attributes:
        x: Frozen corrected/trimmed tensor.
        sample_ids: Frozen first-axis participant order, held only in memory.
        channels: Frozen channel-name order.
        clip_length: Frozen common response length.
    """

    x: NDArray[np.float64]
    sample_ids: NDArray[np.int64]
    channels: NDArray[np.str_]
    clip_length: int


@dataclass(frozen=True)
class ReplayComputation:
    """Aggregate-only result of one isolated full cohort computation.

    Attributes:
        summary: Stage/channel/frame quantile table.
        stage4_reconciliation: Aggregate-safe frozen Pipeline A diagnostics.
    """

    summary: pd.DataFrame
    stage4_reconciliation: dict[str, Any]


@dataclass(frozen=True)
class IsolatedReplayComputation:
    """One self-contained replay with aggregate-safe source digest evidence.

    Attributes:
        summary: Stage/channel/frame quantile table.
        stage4_reconciliation: Aggregate-safe frozen Pipeline A diagnostics.
        raw_digest_entry: Canonical raw-cohort digest before replay loading.
        raw_digest_exit: Canonical raw-cohort digest after replay loading.
        private_filenames: In-memory filenames retained only for output privacy checks.
    """

    summary: pd.DataFrame
    stage4_reconciliation: dict[str, Any]
    raw_digest_entry: str
    raw_digest_exit: str
    private_filenames: tuple[str, ...]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal SHA-256 digest.

    Raises:
        OSError: If the file cannot be read.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def blake2b_file(path: Path) -> str:
    """Return the BLAKE2b-512 digest of one file.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal BLAKE2b-512 digest.

    Raises:
        OSError: If the file cannot be read.
    """
    digest = hashlib.blake2b()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    """Raise a contract error when a required invariant is false."""
    if not condition:
        raise ValueError(message)


def _canonical_package_ledger(packages: Any) -> tuple[list[dict[str, str]], str]:
    """Validate and canonicalize one complete five-field package ledger."""
    _require(isinstance(packages, list) and len(packages) == 324, "Environment package ledger count differs.")
    validated: list[dict[str, str]] = []
    names: set[str] = set()
    for item in packages:
        _require(isinstance(item, dict), "Environment package ledger item is invalid.")
        _require(set(item) == set(ENVIRONMENT_PACKAGE_KEYS), "Environment package ledger item schema differs.")
        _require(
            all(isinstance(item[key], str) and item[key] for key in ENVIRONMENT_PACKAGE_KEYS),
            "Environment package ledger item value is invalid.",
        )
        _require(item["name"] not in names, "Environment package ledger contains a duplicate name.")
        names.add(item["name"])
        validated.append({key: item[key] for key in ENVIRONMENT_PACKAGE_KEYS})
    lines = ["\t".join(item[key] for key in ENVIRONMENT_PACKAGE_KEYS) for item in validated]
    _require(lines == sorted(lines, key=lambda value: value.encode("utf-8")), "Environment package order differs.")
    digest = hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()
    return validated, digest


def load_and_validate_environment_lock(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Load and validate the exact canonical Recovery-v4 environment lock.

    Args:
        path: Environment-lock JSON path.
        expected_sha256: Exact lock-file SHA-256 bound by authorization.

    Returns:
        Validated lock mapping.

    Raises:
        OSError: If the lock cannot be read.
        ValueError: If bytes, schema, ledger, or frozen identities differ.
    """
    _require(bool(re.fullmatch(r"[0-9a-f]{64}", expected_sha256)), "Environment-lock SHA-256 is invalid.")
    lock_path = Path(path)
    payload = lock_path.read_bytes()
    _require(hashlib.sha256(payload).hexdigest() == expected_sha256, "Environment-lock SHA-256 differs.")
    try:
        lock = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Environment lock is not valid canonical JSON.") from exc
    _require(isinstance(lock, dict), "Environment lock must be a mapping.")
    _require(set(lock) == ENVIRONMENT_LOCK_TOP_LEVEL_KEYS, "Environment-lock top-level schema differs.")
    _require(type(lock["schema_version"]) is int and lock["schema_version"] == 2, "Lock schema version differs.")
    _require(lock["platform"] == "linux", "Environment-lock platform differs.")
    _require(lock["subdirectory"] == "linux-64", "Environment-lock subdirectory differs.")
    _require(
        lock["environment_role"] == "efigure4_preprocessing_stage_aggregate_replay_historical_reproduction",
        "Environment-lock role differs.",
    )
    _require(lock["environment_name"] == "gcvocs-efigure4-replay-historical", "Environment-lock name differs.")
    _require(
        lock["source_ledger_sha256"] == ENVIRONMENT_LOCK_SOURCE_LEDGER_SHA256,
        "Environment-lock source-ledger SHA-256 differs.",
    )
    _require(
        lock["final_ledger_sha256"] == ENVIRONMENT_LOCK_FINAL_LEDGER_SHA256,
        "Environment-lock final-ledger SHA-256 differs.",
    )
    _require(
        lock["explicit_local_archive_channel_semantics"] == ENVIRONMENT_LOCK_CHANNEL_SEMANTICS,
        "Environment-lock channel semantics differ.",
    )
    packages, final_digest = _canonical_package_ledger(lock["packages"])
    _require(final_digest == ENVIRONMENT_LOCK_FINAL_LEDGER_SHA256, "Environment package-ledger digest differs.")
    by_name = {item["name"]: item for item in packages}
    for name, expected in EXPECTED_HISTORICAL_SUBSTITUTIONS.items():
        item = by_name.get(name, {})
        actual = tuple(item.get(key) for key in ("version", "build", "channel", "subdirectory"))
        _require(actual == expected, "Historical package substitution identity differs.")
    unknown_names = {item["name"] for item in packages if item["channel"] == "<unknown>"}
    _require(unknown_names == set(EXPECTED_HISTORICAL_SUBSTITUTIONS), "Unexpected installed channel identity.")

    critical = lock["critical_numerical_builds"]
    _require(isinstance(critical, list) and len(critical) == 9, "Critical numerical-build ledger differs.")
    for item in critical:
        _require(
            isinstance(item, dict) and set(item) == set(ENVIRONMENT_PACKAGE_KEYS),
            "Critical build schema differs.",
        )
    _require(
        [item.get("name") for item in critical] == list(EXPECTED_CRITICAL_NUMERICAL_NAMES),
        "Critical numerical-build order differs.",
    )
    _require(critical == [by_name[name] for name in EXPECTED_CRITICAL_NUMERICAL_NAMES], "Critical builds differ.")

    archives = lock["critical_cached_archives"]
    _require(isinstance(archives, list) and len(archives) == 6, "Critical cached-archive ledger differs.")
    _require(
        all(isinstance(item, dict) and set(item) == {"basename", "sha256"} for item in archives),
        "Critical cached-archive item schema differs.",
    )
    _require(
        tuple((item.get("basename"), item.get("sha256")) for item in archives) == EXPECTED_CRITICAL_ARCHIVES,
        "Critical cached-archive identity or order differs.",
    )
    canonical = serialize_json(lock)
    _require(payload == canonical, "Environment-lock JSON serialization is not canonical.")
    return lock


def validate_conda_tool_identity(
    expected: dict[str, Any],
    *,
    resolved_executable: str,
    version_output: str,
    base_records: list[dict[str, str]],
) -> None:
    """Validate the frozen Conda executable, version output, and unique base record.

    Args:
        expected: Locked Conda construction-tool identity.
        resolved_executable: Canonical executable path observed by the caller.
        version_output: Complete stdout from the exact Conda version command.
        base_records: Mapped base-environment records named `conda`.

    Raises:
        ValueError: If any tool identity or record cardinality differs.
    """
    required_keys = {"conda_executable", "conda_version_output", "base_conda_package"}
    _require(isinstance(expected, dict) and set(expected) == required_keys, "Conda tool contract differs.")
    _require(
        isinstance(expected["conda_executable"], str) and expected["conda_executable"],
        "Conda executable contract is invalid.",
    )
    _require(resolved_executable == expected["conda_executable"], "Conda executable identity differs.")
    _require(version_output == f"{expected['conda_version_output']}\n", "Conda version output differs.")
    _require(isinstance(base_records, list) and len(base_records) == 1, "Conda base-record cardinality differs.")
    expected_record = expected["base_conda_package"]
    _require(
        isinstance(expected_record, dict) and set(expected_record) == set(ENVIRONMENT_PACKAGE_KEYS),
        "Conda base-record contract differs.",
    )
    actual_record = base_records[0]
    _require(
        isinstance(actual_record, dict) and set(actual_record) == set(ENVIRONMENT_PACKAGE_KEYS),
        "Conda base record is invalid.",
    )
    _require(actual_record == expected_record, "Conda base-record identity differs.")


def _parse_conda_list_records(payload: str) -> list[dict[str, str]]:
    """Map exact Conda-list JSON records to the canonical five-field schema."""
    try:
        raw_records = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("Conda package ledger is not valid JSON.") from exc
    _require(isinstance(raw_records, list), "Conda package ledger must be a list.")
    records: list[dict[str, str]] = []
    for raw in raw_records:
        _require(isinstance(raw, dict), "Conda package record is invalid.")
        values = {
            "name": raw.get("name"),
            "version": raw.get("version"),
            "build": raw.get("build_string"),
            "channel": raw.get("channel"),
            "subdirectory": raw.get("platform"),
        }
        _require(
            all(isinstance(value, str) and value for value in values.values()),
            "Conda package record field is invalid.",
        )
        records.append(values)  # type: ignore[arg-type]
    return records


def _run_conda_command(executable: Path, arguments: list[str]) -> str:
    """Run one exact read-only Conda observation with sanitized failure handling."""
    result = subprocess.run(
        [str(executable), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or result.stderr:
        raise ValueError("Conda environment observation failed.")
    return result.stdout


def validate_runtime_environment(contract: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Validate the locked dedicated environment without participant-capable access.

    Args:
        contract: Validated Recovery-v4 runtime contract.
        project_root: Repository root containing the immutable environment lock.

    Returns:
        Aggregate-safe environment provenance for audit construction.

    Raises:
        OSError: If a bound lock, executable, prefix, interpreter, or archive is unavailable.
        ValueError: If any lock, tool, interpreter, ledger, build, or archive identity differs.
    """
    root = Path(project_root).resolve(strict=True)
    lock_relative = Path(contract["environment_lock"]["path"])
    _require(not lock_relative.is_absolute(), "Environment-lock path must be project-relative.")
    lock_path = (root / lock_relative).resolve(strict=True)
    _require(lock_path.is_relative_to(root), "Environment-lock path escaped the repository root.")
    lock = load_and_validate_environment_lock(lock_path, contract["environment_lock"]["sha256"])

    identity = contract["environment_identity"]
    conda_path = Path(identity["conda_executable"])
    base_prefix = Path(identity["base_prefix"])
    source_prefix = Path(identity["source_prefix"])
    dedicated_prefix = Path(identity["dedicated_prefix"])
    dedicated_interpreter = Path(identity["dedicated_interpreter"])
    cache_root = Path(identity["cache_root"])
    _require(conda_path.resolve(strict=True) == conda_path, "Conda executable identity differs.")
    _require(base_prefix.resolve(strict=True) == base_prefix, "Conda base-prefix identity differs.")
    _require(source_prefix.resolve(strict=True) == source_prefix, "Conda source-prefix identity differs.")
    _require(
        dedicated_prefix.resolve(strict=True) == dedicated_prefix and not dedicated_prefix.is_symlink(),
        "Dedicated environment identity differs.",
    )
    _require(cache_root.resolve(strict=True) == cache_root, "Conda package-cache identity differs.")
    _require(
        Path(sys.executable).resolve(strict=True) == dedicated_interpreter.resolve(strict=True),
        "Dedicated interpreter identity differs.",
    )
    _require(dedicated_prefix.name == identity["environment_name"], "Dedicated environment name differs.")
    _require(platform.system().lower() == identity["platform"], "Dedicated environment platform differs.")

    version_output = _run_conda_command(conda_path, ["--version"])
    base_records = _parse_conda_list_records(
        _run_conda_command(conda_path, ["list", "--prefix", str(base_prefix), "--json"])
    )
    tool_contract = {
        "conda_executable": identity["conda_executable"],
        "conda_version_output": identity["conda_version_output"],
        "base_conda_package": identity["base_conda_package"],
    }
    validate_conda_tool_identity(
        tool_contract,
        resolved_executable=conda_path.as_posix(),
        version_output=version_output,
        base_records=[record for record in base_records if record["name"] == "conda"],
    )

    live_records = _parse_conda_list_records(
        _run_conda_command(conda_path, ["list", "--prefix", str(dedicated_prefix), "--json"])
    )
    live_records.sort(
        key=lambda item: "\t".join(item[key] for key in ENVIRONMENT_PACKAGE_KEYS).encode("utf-8")
    )
    _, live_digest = _canonical_package_ledger(live_records)
    _require(live_digest == identity["final_ledger_sha256"], "Live environment ledger digest differs.")
    _require(live_records == lock["packages"], "Live environment ledger differs from the lock.")

    for archive in lock["critical_cached_archives"]:
        basename = archive["basename"]
        _require(Path(basename).name == basename, "Cached archive basename is invalid.")
        archive_path = (cache_root / basename).resolve(strict=True)
        _require(archive_path.is_relative_to(cache_root) and archive_path.is_file(), "Cached archive is invalid.")
        _require(sha256_file(archive_path) == archive["sha256"], "Cached archive SHA-256 differs.")

    return {
        "environment_lock_sha256": contract["environment_lock"]["sha256"],
        "environment_name": lock["environment_name"],
        "platform": lock["platform"],
        "subdirectory": lock["subdirectory"],
        "package_count": len(live_records),
        "source_ledger_sha256": lock["source_ledger_sha256"],
        "final_ledger_sha256": lock["final_ledger_sha256"],
        "critical_numerical_builds": lock["critical_numerical_builds"],
        "package_ledger_equality_passed": True,
        "archive_hash_validation_passed": True,
    }


def load_and_validate_contract(path: Path, approved_plan_sha256: str) -> dict[str, Any]:
    """Load and validate the locked aggregate-replay runtime contract.

    Args:
        path: YAML runtime-contract path.
        approved_plan_sha256: Exact user-approved results-free plan digest.

    Returns:
        Validated mutable mapping loaded from YAML.

    Raises:
        OSError: If the contract cannot be read.
        ValueError: If YAML parsing or any locked invariant fails.
    """
    try:
        raw_payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError("Runtime contract is not valid YAML.") from exc
    if not isinstance(raw_payload, dict):
        raise ValueError("Runtime contract must contain a top-level mapping.")

    contract: dict[str, Any] = raw_payload
    approved_plan = contract.get("approved_plan")
    _require(isinstance(approved_plan, dict), "Runtime contract must declare the approved plan.")
    _require(
        approved_plan.get("sha256") == approved_plan_sha256,
        "Runtime contract approved plan hash does not match Mandatory Gate 1.",
    )
    _require(
        approved_plan == {"path": APPROVED_PLAN_PATH, "sha256": APPROVED_PLAN_SHA256},
        "Runtime contract recovery plan identity differs.",
    )
    _require(
        contract.get("original_plan") == {"path": ORIGINAL_PLAN_PATH, "sha256": ORIGINAL_PLAN_SHA256},
        "Runtime contract original plan identity differs.",
    )
    _require(
        contract.get("recovery_v2_plan")
        == {"path": RECOVERY_V2_PLAN_PATH, "sha256": RECOVERY_V2_PLAN_SHA256},
        "Runtime contract recovery-v2 plan identity differs.",
    )
    _require(
        contract.get("recovery_v3_plan")
        == {"path": RECOVERY_V3_PLAN_PATH, "sha256": RECOVERY_V3_PLAN_SHA256},
        "Runtime contract recovery-v3 plan identity differs.",
    )
    _require(
        contract.get("reporting_correction")
        == {
            "path": REPORTING_CORRECTION_PATH,
            "sha256": REPORTING_CORRECTION_SHA256,
            "byte_size": REPORTING_CORRECTION_BYTE_SIZE,
        },
        "Runtime contract reporting-correction identity differs.",
    )
    _require(
        contract.get("numerical_environment_correction")
        == {
            "path": NUMERICAL_ENVIRONMENT_CORRECTION_PATH,
            "sha256": NUMERICAL_ENVIRONMENT_CORRECTION_SHA256,
        },
        "Runtime contract numerical-environment correction identity differs.",
    )
    _require(
        contract.get("channel_metadata_correction")
        == {
            "path": CHANNEL_METADATA_CORRECTION_PATH,
            "sha256": CHANNEL_METADATA_CORRECTION_SHA256,
        },
        "Runtime contract channel-metadata correction identity differs.",
    )
    _require(
        contract.get("environment_lock") == {"path": ENVIRONMENT_LOCK_PATH, "sha256": ENVIRONMENT_LOCK_SHA256},
        "Runtime contract environment-lock identity differs.",
    )
    _require(
        contract.get("environment_identity")
        == {
            "conda_executable": "[PRIVATE_PATH_REDACTED]",
            "conda_version_output": "conda 26.1.0",
            "base_prefix": "[PRIVATE_PATH_REDACTED]",
            "source_prefix": "[PRIVATE_PATH_REDACTED]",
            "dedicated_prefix": "[PRIVATE_PATH_REDACTED]",
            "dedicated_interpreter": (
                "[PRIVATE_PATH_REDACTED]"
            ),
            "cache_root": "[PRIVATE_PATH_REDACTED]",
            "environment_name": "gcvocs-efigure4-replay-historical",
            "platform": "linux",
            "subdirectory": "linux-64",
            "environment_role": "efigure4_preprocessing_stage_aggregate_replay_historical_reproduction",
            "source_ledger_sha256": ENVIRONMENT_LOCK_SOURCE_LEDGER_SHA256,
            "final_ledger_sha256": ENVIRONMENT_LOCK_FINAL_LEDGER_SHA256,
            "explicit_local_archive_channel_semantics": ENVIRONMENT_LOCK_CHANNEL_SEMANTICS,
            "base_conda_package": {
                "name": "conda",
                "version": "26.1.0",
                "build": "py313h78bf25f_0",
                "channel": "conda-forge",
                "subdirectory": "linux-64",
            },
        },
        "Runtime contract environment identity differs.",
    )
    _require(contract.get("analysis_id") == ANALYSIS_ID, "Unexpected analysis ID in runtime contract.")
    _require(
        contract.get("analysis_classification") == ANALYSIS_CLASSIFICATION,
        "Unexpected analysis classification in runtime contract.",
    )
    _require(tuple(contract.get("channels", ())) == EXPECTED_CHANNELS, "Runtime contract channel order differs.")

    stages = contract.get("stages")
    _require(isinstance(stages, list) and len(stages) == 4, "Runtime contract must declare exactly four stages.")
    _require(tuple(stage.get("order") for stage in stages) == (1, 2, 3, 4), "Runtime stage order differs.")
    _require(tuple(stage.get("id") for stage in stages) == EXPECTED_STAGE_IDS, "Runtime stage IDs differ.")
    _require(tuple(stage.get("unit") for stage in stages) == EXPECTED_STAGE_UNITS, "Runtime stage units differ.")

    response_axis = contract.get("response_axis", {})
    _require(
        response_axis == {
            "name": "response_frame",
            "first": 0,
            "last": 959,
            "clip_length": 960,
            "convert_to_seconds": False,
        },
        "Runtime response-axis contract differs.",
    )
    _require(contract.get("padding_relative_deficit_threshold") == 0.0125, "Padding threshold differs.")
    _require(contract.get("numpy_quantile_method") == "linear", "NumPy quantile method differs.")
    _require(contract.get("no_stochastic_operations") is True, "Stochastic operations must be prohibited.")
    _require(contract.get("figure_design_authorization") is False, "Figure design must remain unauthorized.")
    _require(contract.get("release_authorization") is False, "Release must remain unauthorized.")

    population = contract.get("population")
    _require(
        population
        == {
            "expected_source_count": 1040,
            "expected_final_count": 972,
            "expected_excluded_count": 68,
            "expected_source_fallback_count": 60,
            "expected_final_fallback_count": 54,
            "expected_excluded_fallback_count": 6,
            "expected_padded_count": 41,
            "final_cohort_selection_rule": "in_final_output == True",
            "canonical_order": "ascending_sample_id",
        },
        "Runtime population and fallback-denominator contract differs.",
    )

    authority = contract.get("runtime_authority", {})
    _require(
        set(authority)
        == {
            "expected_branch",
            "expected_head",
            "expected_parent",
            "preprocessing_frozen_commit",
            "allowed_worktree_entries",
        },
        "Runtime authority schema differs.",
    )
    _require(authority.get("expected_branch") == "main", "Runtime branch authority differs.")
    _require(
        authority.get("expected_head") == "098d6d5d65d9fb935fa97a30d1353a9694a0dc3a",
        "Runtime HEAD authority differs.",
    )
    _require(
        authority.get("expected_parent") == "8c44846477c250a03380eda24a5c99ab63294426",
        "Runtime parent authority differs.",
    )
    _require(
        authority.get("preprocessing_frozen_commit") == "622cb3664774b3940f7bcf092004cfff62d06822",
        "Runtime frozen-tag authority differs.",
    )
    _require(
        tuple(authority.get("allowed_worktree_entries", ())) == EXPECTED_ALLOWED_WORKTREE_ENTRIES,
        "Runtime authorized worktree entries differ.",
    )

    outputs = contract.get("outputs", {})
    _require(outputs.get("final_directory") == FINAL_OUTPUT_DIRECTORY, "Final output namespace differs.")
    _require(
        outputs.get("candidate_one_directory") == CANDIDATE_ONE_DIRECTORY,
        "Candidate-one output namespace differs.",
    )
    _require(
        outputs.get("candidate_two_directory") == CANDIDATE_TWO_DIRECTORY,
        "Candidate-two output namespace differs.",
    )

    preprocessing_config = contract.get("inputs", {}).get("preprocessing_config", {})
    _require(preprocessing_config.get("sha256") is None, "Preprocessing-config SHA-256 must remain null.")
    _require(
        preprocessing_config.get("sha256_status") == CONFIG_ANOMALY_STATUS,
        "Preprocessing-config SHA-256 anomaly status differs.",
    )
    return contract


def _parse_worktree_status_paths(status_output: str) -> list[str]:
    """Parse project-relative paths from non-porcelain-v2 short Git status output."""
    paths: list[str] = []
    for line in status_output.splitlines():
        if len(line) < 4 or line[2] != " ":
            raise ValueError("Read-only Git worktree status could not be parsed safely.")
        path = line[3:]
        if not path or " -> " in path or "\x00" in path:
            raise ValueError("Read-only Git worktree status path is invalid.")
        paths.append(path)
    return paths


def validate_authority_snapshot(contract: dict[str, Any], actual: dict[str, Any]) -> None:
    """Compare a read-only runtime authority snapshot with the locked contract.

    Args:
        contract: Validated runtime contract.
        actual: Aggregate-safe values collected from read-only Git inspection.

    Raises:
        ValueError: If any authority value differs exactly.
    """
    authority = contract["runtime_authority"]
    _require(actual.get("branch") == authority["expected_branch"], "Runtime branch differs from authority.")
    _require(actual.get("head") == authority["expected_head"], "Runtime HEAD differs from authority.")
    _require(actual.get("sole_parent") == authority["expected_parent"], "Runtime sole parent differs from authority.")
    _require(
        actual.get("preprocessing_frozen_commit") == authority["preprocessing_frozen_commit"],
        "preprocessing-frozen commit differs from authority.",
    )
    _require(actual.get("preprocessing_frozen_is_ancestor") is True, "Frozen tag ancestry check failed.")
    _require(actual.get("frozen_diff_is_empty") is True, "Frozen preprocessing diff is not empty.")
    _require(actual.get("final_output_absent") is True, "Final output namespace already exists.")
    _require(
        actual.get("candidate_one_output_absent") is True,
        "Candidate-one output namespace already exists.",
    )
    _require(
        actual.get("candidate_two_output_absent") is True,
        "Candidate-two output namespace already exists.",
    )
    _require(
        actual.get("approved_plan_sha256") == contract["approved_plan"]["sha256"],
        "Approved recovery plan hash differs from authority.",
    )
    _require(
        actual.get("original_plan_sha256") == contract["original_plan"]["sha256"],
        "Original plan hash differs from authority.",
    )
    _require(
        actual.get("recovery_v2_plan_sha256") == contract["recovery_v2_plan"]["sha256"],
        "Recovery-v2 plan hash differs from authority.",
    )
    _require(
        actual.get("recovery_v3_plan_sha256") == contract["recovery_v3_plan"]["sha256"],
        "Recovery-v3 plan hash differs from authority.",
    )
    _require(
        actual.get("reporting_correction_sha256") == contract["reporting_correction"]["sha256"],
        "Reporting-correction hash differs from authority.",
    )
    _require(
        actual.get("numerical_environment_correction_sha256")
        == contract["numerical_environment_correction"]["sha256"],
        "Numerical-environment correction hash differs from authority.",
    )
    _require(
        actual.get("channel_metadata_correction_sha256") == contract["channel_metadata_correction"]["sha256"],
        "Channel-metadata correction hash differs from authority.",
    )
    _require(
        actual.get("environment_lock_sha256") == contract["environment_lock"]["sha256"],
        "Environment-lock hash differs from authority.",
    )

    worktree_entries = actual.get("worktree_entries", ())
    _require(isinstance(worktree_entries, (list, tuple)), "Worktree scope snapshot is invalid.")
    _require(
        len(worktree_entries) == len(set(worktree_entries))
        and set(worktree_entries) == set(authority["allowed_worktree_entries"]),
        "Worktree scope differs from the authorized entries.",
    )

    actual_blobs = actual.get("input_blobs")
    _require(isinstance(actual_blobs, dict), "Runtime input blob snapshot is missing.")
    expected_blobs = {name: payload["git_blob"] for name, payload in contract["inputs"].items()}
    _require(actual_blobs == expected_blobs, "At least one runtime input blob differs from authority.")


def _require_unique_ids(frame: pd.DataFrame, table_name: str) -> None:
    """Require a populated, integral, unique sample identifier column."""
    if frame["sample_id"].isna().any() or not pd.api.types.is_integer_dtype(frame["sample_id"]):
        raise ValueError(f"{table_name} sample identifiers must be complete integral values.")
    if not frame["sample_id"].is_unique:
        raise ValueError(f"{table_name} sample identifiers must be unique.")


def _require_boolean_column(frame: pd.DataFrame, column: str, table_name: str) -> None:
    """Require one parsed boolean audit column without coercion."""
    if not pd.api.types.is_bool_dtype(frame[column]) or frame[column].isna().any():
        raise ValueError(f"{table_name} audit column must contain complete booleans: {column}.")


def load_minimal_cohort_tables(
    sample_index_path: Path,
    qc_diagnostics_path: Path,
    *,
    expected_source_count: int,
    expected_final_count: int,
    expected_excluded_count: int,
    expected_source_fallback_count: int,
    expected_final_fallback_count: int,
    expected_excluded_fallback_count: int,
    expected_padded_count: int,
) -> CohortTables:
    """Load, align, and validate only the authorized cohort audit columns.

    Args:
        sample_index_path: Frozen source/final-cohort index CSV.
        qc_diagnostics_path: Frozen persisted preprocessing diagnostics CSV.
        expected_source_count: Required complete source-cohort row count.
        expected_final_count: Required final-cohort row count.
        expected_excluded_count: Required pre-transformation exclusion count.
        expected_source_fallback_count: Required source-cohort fallback count.
        expected_final_fallback_count: Required final-cohort fallback count.
        expected_excluded_fallback_count: Required excluded-cohort fallback count.
        expected_padded_count: Required final-cohort padding count.

    Returns:
        Minimal final-cohort audit state in canonical ascending ID order.

    Raises:
        OSError: If either CSV cannot be read.
        ValueError: If schema, dtype, count, uniqueness, or alignment differs.
    """
    try:
        sample_index = pd.read_csv(Path(sample_index_path), usecols=SAMPLE_INDEX_USECOLS)
        diagnostics = pd.read_csv(Path(qc_diagnostics_path), usecols=QC_DIAGNOSTICS_USECOLS)
    except (ValueError, pd.errors.ParserError) as exc:
        raise ValueError("Minimal cohort audit CSV loading failed.") from exc

    if len(sample_index) != expected_source_count or len(diagnostics) != expected_source_count:
        raise ValueError("Minimal cohort audit source row count differs from authority.")
    if expected_source_count - expected_final_count != expected_excluded_count:
        raise ValueError("Expected source/final/excluded cohort arithmetic is inconsistent.")
    if expected_source_fallback_count != expected_final_fallback_count + expected_excluded_fallback_count:
        raise ValueError("Expected source/final/excluded fallback reconciliation differs.")
    _require_unique_ids(sample_index, "sample index")
    _require_unique_ids(diagnostics, "QC diagnostics")
    for column in ("fallback_used", "padded", "in_final_output"):
        _require_boolean_column(sample_index, column, "sample index")
    _require_boolean_column(diagnostics, "fallback_used", "QC diagnostics")
    if set(sample_index["sample_id"]) != set(diagnostics["sample_id"]):
        raise ValueError("Minimal cohort audit alignment is incomplete.")

    try:
        aligned = sample_index.merge(
            diagnostics,
            on="sample_id",
            how="inner",
            sort=False,
            validate="one_to_one",
            suffixes=("_index", "_audit"),
        )
    except pd.errors.MergeError as exc:
        raise ValueError("Minimal cohort audit alignment is not one-to-one.") from exc
    if len(aligned) != expected_source_count:
        raise ValueError("Minimal cohort audit alignment row count differs.")
    for field in ("n_rows", "fallback_used"):
        if not aligned[f"{field}_index"].equals(aligned[f"{field}_audit"]):
            raise ValueError(f"Minimal cohort audit field mismatch: {field}.")

    final_mask = aligned["in_final_output"]
    final = aligned.loc[final_mask].copy()
    excluded = aligned.loc[~final_mask]
    if len(final) != expected_final_count or len(excluded) != expected_excluded_count:
        raise ValueError("Final-cohort count differs from authority.")
    if excluded["padded_index"].any():
        raise ValueError("Excluded sample-index padding status must be false.")
    if excluded["padded_audit"].notna().any():
        raise ValueError("Excluded QC-diagnostics padding status must be missing.")
    final_audit_padding = final["padded_audit"]
    if final_audit_padding.isna().any() or not all(
        isinstance(value, (bool, np.bool_)) for value in final_audit_padding.tolist()
    ):
        raise ValueError("Final QC-diagnostics padding status must contain genuine Boolean scalars.")
    if not np.array_equal(
        final["padded_index"].to_numpy(dtype=bool),
        final_audit_padding.to_numpy(dtype=object),
    ):
        raise ValueError("Final-cohort padding status differs across audit tables.")
    source_fallback_count = int(aligned["fallback_used_index"].sum())
    final_fallback_count = int(final["fallback_used_index"].sum())
    excluded_fallback_count = int(excluded["fallback_used_index"].sum())
    if source_fallback_count != expected_source_fallback_count:
        raise ValueError("Source-cohort fallback count differs from authority.")
    if final_fallback_count != expected_final_fallback_count:
        raise ValueError("Final-cohort fallback count differs from authority.")
    if excluded_fallback_count != expected_excluded_fallback_count:
        raise ValueError("Excluded-cohort fallback count differs from authority.")
    if source_fallback_count != final_fallback_count + excluded_fallback_count:
        raise ValueError("Observed source/final/excluded fallback reconciliation differs.")
    if int(final["padded_index"].sum()) != expected_padded_count:
        raise ValueError("Final-cohort padding count differs from authority.")
    required_numeric = [
        "n_rows_index",
        "t_baseline_end",
        "env_temperature_sample",
        "env_humidity_sample",
        "n_available",
        "clip_length_L",
    ]
    if final[required_numeric].isna().any().any():
        raise ValueError("Final-cohort persisted replay audit contains missing required values.")
    if final["filename"].isna().any() or final["filename"].astype(str).str.strip().eq("").any():
        raise ValueError("Final-cohort raw filename is missing.")
    for column in ("n_rows_index", "t_baseline_end", "n_available", "clip_length_L"):
        values = final[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"Final-cohort audit field must be finite integral values: {column}.")
    environment_values = final[["env_temperature_sample", "env_humidity_sample"]].to_numpy(dtype=np.float64)
    if not np.isfinite(environment_values).all():
        raise ValueError("Final-cohort environmental audit values must be finite.")
    if (
        (final["n_rows_index"] <= 0).any()
        or (final["t_baseline_end"] <= 0).any()
        or (final["t_baseline_end"] >= final["n_rows_index"]).any()
        or (final["clip_length_L"] <= 0).any()
    ):
        raise ValueError("Final-cohort row/onset/clip audit bounds differ.")
    calculated_available = final["n_rows_index"].to_numpy(dtype=np.int64) - final["t_baseline_end"].to_numpy(
        dtype=np.int64
    )
    if not np.array_equal(calculated_available, final["n_available"].to_numpy(dtype=np.int64)):
        raise ValueError("Final-cohort available-response audit count differs.")
    padded_mask = final["padded_index"].to_numpy(dtype=bool)
    available = final["n_available"].to_numpy(dtype=np.int64)
    clip_lengths = final["clip_length_L"].to_numpy(dtype=np.int64)
    relative_deficits = final["relative_deficit"].to_numpy(dtype=np.float64)
    if np.any(padded_mask & (available >= clip_lengths)) or np.any((~padded_mask) & (available < clip_lengths)):
        raise ValueError("Final-cohort padding audit is inconsistent with available response length.")
    expected_deficits = np.divide(
        clip_lengths[padded_mask] - available[padded_mask],
        clip_lengths[padded_mask],
        dtype=np.float64,
    )
    if (
        not np.isfinite(relative_deficits[padded_mask]).all()
        or not np.allclose(
            relative_deficits[padded_mask],
            expected_deficits,
            rtol=0.0,
            atol=1.0e-15,
            equal_nan=False,
        )
        or not np.isnan(relative_deficits[~padded_mask]).all()
    ):
        raise ValueError("Final-cohort relative-deficit audit differs.")

    final_rows = pd.DataFrame(
        {
            "sample_id": final["sample_id"].astype(np.int64),
            "filename": final["filename"].astype(str),
            "n_rows": final["n_rows_index"].astype(np.int64),
            "t_baseline_end": final["t_baseline_end"].astype(np.int64),
            "fallback_used": final["fallback_used_index"].astype(bool),
            "env_temperature_sample": final["env_temperature_sample"].astype(np.float64),
            "env_humidity_sample": final["env_humidity_sample"].astype(np.float64),
            "n_available": final["n_available"].astype(np.int64),
            "clip_length_L": final["clip_length_L"].astype(np.int64),
            "relative_deficit": final["relative_deficit"].astype(np.float64),
            "padded": final["padded_index"].astype(bool),
        }
    ).sort_values("sample_id", kind="stable", ignore_index=True)
    return CohortTables(final_rows=final_rows, source_count=expected_source_count, final_count=expected_final_count)


def _resolve_contained_raw_path(filename: str, raw_root: Path) -> Path:
    """Resolve one private raw filename beneath the authorized root."""
    filename_path = Path(filename)
    if filename_path.is_absolute():
        raise ValueError("Absolute raw filenames are prohibited.")
    if ".." in filename_path.parts:
        raise ValueError("Parent traversal components are prohibited.")
    resolved_root = Path(raw_root).resolve(strict=True)
    raw_path = (resolved_root / filename_path).resolve(strict=True)
    if not raw_path.is_relative_to(resolved_root) or not raw_path.is_file():
        raise ValueError("Raw path containment failed.")
    return raw_path


def _load_raw_signal_private(filename: str, raw_root: Path, expected_n_rows: int) -> RawSignal:
    """Load one contained S1-S13 payload while allowing private context internally."""
    raw_path = _resolve_contained_raw_path(filename, raw_root)

    header = pd.read_csv(raw_path, encoding=RAW_CSV_ENCODING, nrows=0).columns.tolist()
    if header != list(EXPECTED_RAW_HEADER):
        raise ValueError(f"Raw header differs at {raw_path}.")
    frame = pd.read_csv(
        raw_path,
        encoding=RAW_CSV_ENCODING,
        usecols=RAW_SIGNAL_USECOLS,
        dtype=np.float64,
    )
    if frame.columns.tolist() != RAW_SIGNAL_USECOLS:
        raise ValueError(f"Raw signal column order differs at {raw_path}.")
    if len(frame) != int(expected_n_rows):
        raise ValueError(f"Raw signal row count differs at {raw_path}.")
    values = frame.to_numpy(dtype=np.float64, copy=True)
    if values.shape != (int(expected_n_rows), 13) or not np.isfinite(values).all():
        raise ValueError(f"Raw signal values are invalid at {raw_path}.")
    return RawSignal(
        gas_signals=np.asarray(values[:, :11], dtype=np.float64),
        temperature=np.asarray(values[:, 11], dtype=np.float64),
        humidity=np.asarray(values[:, 12], dtype=np.float64),
        n_rows=int(expected_n_rows),
    )


def load_raw_signal(filename: str, raw_root: Path, *, expected_n_rows: int) -> RawSignal:
    """Load one privacy-sensitive raw signal behind a sanitized error boundary.

    Args:
        filename: In-memory filename from the minimal frozen cohort index.
        raw_root: Authorized raw breath-data root.
        expected_n_rows: Persisted row count required for reconciliation.

    Returns:
        S1-S13 values with no identifier or path fields.

    Raises:
        RawSignalLoadError: If containment, schema, parsing, shape, or values fail.
    """
    try:
        return _load_raw_signal_private(filename, raw_root, expected_n_rows)
    except Exception as exc:
        raise RawSignalLoadError("Raw signal loading failed.") from exc


def _finite_matrix(values: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    """Validate one full-length or trimmed gas matrix without altering scale."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(EXPECTED_CHANNELS) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} gas matrix is invalid.")
    return matrix


def replay_participant(
    raw: RawSignal,
    *,
    t_baseline_end: int,
    persisted_temperature_sample: float,
    persisted_humidity_sample: float,
    expected_n_available: int,
    expected_padded: bool,
    env_model: EnvCompensationModel,
    clip_length: int,
    padding_threshold: float,
    environmental_rtol: float,
    environmental_atol: float,
) -> ParticipantReplay:
    """Replay four frozen stages for one participant without persisting arrays.

    Args:
        raw: Minimal S1-S13 signal payload.
        t_baseline_end: Persisted half-open baseline end and response start.
        persisted_temperature_sample: Frozen S12 baseline summary.
        persisted_humidity_sample: Frozen S13 baseline summary.
        expected_n_available: Frozen number of response rows before trimming.
        expected_padded: Frozen last-row padding decision.
        env_model: Deserialized frozen environmental model.
        clip_length: Common response-window length.
        padding_threshold: Frozen relative-deficit padding threshold.
        environmental_rtol: Relative tolerance for S12/S13 reconciliation.
        environmental_atol: Absolute tolerance for S12/S13 reconciliation.

    Returns:
        Four aligned in-memory response windows and common trim diagnostics.

    Raises:
        ValueError: If environmental, transformation, or trim reconciliation fails.
    """
    failure_codes = _ACTIVE_ISOLATED_FAILURE_CODES.get()
    environmental_code = failure_codes.environmental_summary if failure_codes is not None else None
    with _optional_isolated_failure_boundary(environmental_code):
        x0 = _finite_matrix(raw.gas_signals, "Raw")
        baseline_end = int(t_baseline_end)
        if baseline_end <= 0 or baseline_end >= raw.n_rows:
            raise ValueError("Persisted t_baseline_end is invalid.")
        t_sample = float(np.mean(np.asarray(raw.temperature[:baseline_end], dtype=np.float64)))
        h_sample = float(np.mean(np.asarray(raw.humidity[:baseline_end], dtype=np.float64)))
        environment_matches = np.isclose(
            [t_sample, h_sample],
            [float(persisted_temperature_sample), float(persisted_humidity_sample)],
            rtol=float(environmental_rtol),
            atol=float(environmental_atol),
            equal_nan=False,
        )
        if not bool(np.all(environment_matches)):
            raise ValueError("Environmental summary reconciliation failed.")

    helper_loggers = [
        logging.getLogger("src.data.smoothing"),
        logging.getLogger("src.data.correction"),
        logging.getLogger("src.data.env_compensation"),
    ]
    prior_disabled = [logger.disabled for logger in helper_loggers]
    try:
        for logger in helper_loggers:
            logger.disabled = True
        smoothing_code = failure_codes.smoothing if failure_codes is not None else None
        with _optional_isolated_failure_boundary(smoothing_code):
            x1 = _finite_matrix(apply_sg_smoothing(x0), "Smoothed")
        compensation_code = failure_codes.environmental_compensation if failure_codes is not None else None
        with _optional_isolated_failure_boundary(compensation_code):
            x2 = _finite_matrix(env_model.apply(x1, t_sample, h_sample), "Compensated")
        fractional_code = failure_codes.fractional_correction if failure_codes is not None else None
        with _optional_isolated_failure_boundary(fractional_code):
            x3 = _finite_matrix(apply_fractional_correction(x2, baseline_end), "Fractional-corrected")
        trimming_code = failure_codes.trimming if failure_codes is not None else None
        with _optional_isolated_failure_boundary(trimming_code):
            full_stages = dict(zip(EXPECTED_STAGE_IDS, (x0, x1, x2, x3), strict=True))
            trim_results = {
                stage_id: trim_response_segment(matrix, baseline_end, int(clip_length), float(padding_threshold))
                for stage_id, matrix in full_stages.items()
            }
    finally:
        for logger, disabled in zip(helper_loggers, prior_disabled, strict=True):
            logger.disabled = disabled

    trimming_code = failure_codes.trimming if failure_codes is not None else None
    with _optional_isolated_failure_boundary(trimming_code):
        reference_available: int | None = None
        reference_padded: bool | None = None
        trimmed_stages: dict[str, NDArray[np.float64]] = {}
        for stage_id in EXPECTED_STAGE_IDS:
            result = trim_results[stage_id]
            if not result.clip_length_pass or result.trimmed is None:
                raise ValueError("Common response-window trimming failed.")
            trimmed = _finite_matrix(result.trimmed, stage_id)
            if trimmed.shape != (int(clip_length), len(EXPECTED_CHANNELS)):
                raise ValueError("Common response-window shape differs.")
            if reference_available is None:
                reference_available = int(result.n_available)
                reference_padded = bool(result.padded)
            if int(result.n_available) != reference_available or bool(result.padded) is not reference_padded:
                raise ValueError("Four-stage trimming decisions differ.")
            trimmed_stages[stage_id] = np.array(trimmed, dtype=np.float64, copy=True)

        if reference_available != int(expected_n_available):
            raise ValueError("Replay available-response count differs from persisted audit.")
        if reference_padded is not bool(expected_padded):
            raise ValueError("Replay padding status differs from persisted audit.")
    return ParticipantReplay(
        stages=trimmed_stages,
        n_available=int(reference_available),
        padded=bool(reference_padded),
    )


def validate_stage_tensors(
    stage_tensors: dict[str, NDArray[np.float64]],
    *,
    expected_participants: int,
    expected_frames: int,
) -> None:
    """Require all four finite stage tensors to have one exact common shape.

    Args:
        stage_tensors: Mapping from locked stage ID to in-memory tensor.
        expected_participants: Required participant-axis length.
        expected_frames: Required response-frame-axis length.

    Raises:
        ValueError: If IDs, shape, numeric dtype, or finite status differs.
    """
    if tuple(stage_tensors) != EXPECTED_STAGE_IDS:
        raise ValueError("Stage tensor IDs or order differ from the locked contract.")
    expected_shape = (int(expected_participants), int(expected_frames), len(EXPECTED_CHANNELS))
    for stage_id in EXPECTED_STAGE_IDS:
        tensor = np.asarray(stage_tensors[stage_id])
        if tensor.shape != expected_shape:
            raise ValueError(f"Stage tensor shape differs for {stage_id}.")
        if not np.issubdtype(tensor.dtype, np.number):
            raise ValueError(f"Stage tensor dtype is not numeric for {stage_id}.")
        if not np.isfinite(tensor).all():
            raise ValueError(f"Stage tensor must contain only finite values for {stage_id}.")


def aggregate_stage_tensors(
    stage_tensors: dict[str, NDArray[np.float64]],
    *,
    stages: list[dict[str, Any]],
    channels: list[str],
    expected_participants: int,
    expected_frames: int,
    quantile_method: str,
) -> pd.DataFrame:
    """Aggregate each stage independently into exact channel/frame quantiles.

    Args:
        stage_tensors: Four participant-by-frame-by-channel stage tensors.
        stages: Locked ordered stage definitions and unit codes.
        channels: Locked ordered channel IDs.
        expected_participants: Required participant count in every stage.
        expected_frames: Required response-frame count in every stage.
        quantile_method: Required NumPy quantile interpolation method.

    Returns:
        Exact stage-major, channel-major, frame-major aggregate table.

    Raises:
        ValueError: If the tensor or aggregate contract differs.
    """
    validate_stage_tensors(
        stage_tensors,
        expected_participants=expected_participants,
        expected_frames=expected_frames,
    )
    if quantile_method != "linear":
        raise ValueError("Only NumPy quantile method='linear' is authorized.")
    if tuple(channels) != EXPECTED_CHANNELS:
        raise ValueError("Aggregate channel order differs from the locked contract.")
    if tuple(stage["id"] for stage in stages) != EXPECTED_STAGE_IDS:
        raise ValueError("Aggregate stage order differs from the locked contract.")

    rows: list[tuple[int, str, str, int, str, int, float, float, float]] = []
    for stage in stages:
        tensor = np.asarray(stage_tensors[stage["id"]], dtype=np.float64)
        quantiles = np.quantile(tensor, [0.25, 0.50, 0.75], axis=0, method="linear")
        for channel_index, channel in enumerate(channels):
            for frame_index in range(expected_frames):
                rows.append(
                    (
                        int(stage["order"]),
                        str(stage["id"]),
                        str(channel),
                        int(frame_index),
                        str(stage["unit"]),
                        int(expected_participants),
                        float(quantiles[0, frame_index, channel_index]),
                        float(quantiles[1, frame_index, channel_index]),
                        float(quantiles[2, frame_index, channel_index]),
                    )
                )
    summary = pd.DataFrame(rows, columns=AGGREGATE_COLUMNS)
    for column in ("stage_order", "response_frame", "n_participants"):
        summary[column] = summary[column].astype(np.int64)
    for column in ("q25", "median", "q75"):
        summary[column] = summary[column].astype(np.float64)
    validate_aggregate_summary(
        summary,
        stages=stages,
        channels=channels,
        expected_participants=expected_participants,
        expected_frames=expected_frames,
    )
    return summary


def validate_aggregate_summary(
    summary: pd.DataFrame,
    *,
    stages: list[dict[str, Any]],
    channels: list[str],
    expected_participants: int,
    expected_frames: int,
) -> None:
    """Validate the exact aggregate CSV schema, types, ordering, and ranges.

    Args:
        summary: Aggregate stage/channel/frame table.
        stages: Locked stage definitions.
        channels: Locked channel order.
        expected_participants: Required count in every row.
        expected_frames: Required frame count per channel and stage.

    Raises:
        ValueError: If any aggregate output invariant fails.
    """
    if summary.columns.tolist() != AGGREGATE_COLUMNS:
        raise ValueError("Aggregate CSV schema or column order differs.")
    expected_rows = len(stages) * len(channels) * int(expected_frames)
    if len(summary) != expected_rows:
        raise ValueError("Aggregate CSV row count differs.")
    for column in ("stage_order", "response_frame", "n_participants"):
        if summary[column].dtype != np.int64:
            raise ValueError(f"Aggregate CSV integer dtype differs: {column}.")
    for column in ("q25", "median", "q75"):
        if summary[column].dtype != np.float64:
            raise ValueError(f"Aggregate CSV floating dtype differs: {column}.")
    if summary.isna().any().any() or not np.isfinite(summary[["q25", "median", "q75"]].to_numpy()).all():
        raise ValueError("Aggregate CSV contains missing or nonfinite values.")
    if not summary["n_participants"].eq(int(expected_participants)).all():
        raise ValueError("Aggregate CSV participant count differs across rows.")
    if not ((summary["q25"] <= summary["median"]) & (summary["median"] <= summary["q75"])).all():
        raise ValueError("Aggregate quantile ordering differs.")

    expected_keys = [
        (int(stage["order"]), str(stage["id"]), channel, frame, str(stage["unit"]))
        for stage in stages
        for channel in channels
        for frame in range(expected_frames)
    ]
    actual_keys = list(
        summary[["stage_order", "stage_id", "channel", "response_frame", "unit"]].itertuples(
            index=False,
            name=None,
        )
    )
    if actual_keys != expected_keys:
        raise ValueError("Aggregate CSV row ordering or categorical values differ.")


def serialize_aggregate_csv(summary: pd.DataFrame, *, float_format: str) -> bytes:
    """Serialize an already validated aggregate table deterministically.

    Args:
        summary: Exact aggregate output table.
        float_format: Locked pandas floating-point format.

    Returns:
        UTF-8 CSV bytes with LF line endings and no index.

    Raises:
        ValueError: If the floating format differs from the locked contract.
    """
    if float_format != "%.17g":
        raise ValueError("Aggregate CSV float format differs from the locked contract.")
    buffer = io.StringIO(newline="")
    summary.to_csv(buffer, index=False, float_format=float_format, lineterminator="\n")
    return buffer.getvalue().encode("utf-8")


def reconcile_stage4(
    replayed_stage4: NDArray[np.float64],
    replay_sample_ids: NDArray[np.integer[Any]],
    frozen_x: NDArray[np.float64],
    frozen_sample_ids: NDArray[np.integer[Any]],
    frozen_channels: NDArray[Any],
    *,
    frozen_clip_length: int,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    """Reconcile replayed Stage 4 with the permitted frozen Pipeline A arrays.

    Args:
        replayed_stage4: Replayed final-stage tensor.
        replay_sample_ids: Canonical in-memory replay order.
        frozen_x: Frozen Pipeline A tensor.
        frozen_sample_ids: Frozen Pipeline A first-axis order.
        frozen_channels: Frozen ordered channel-name array.
        frozen_clip_length: Frozen scalar clip length.
        rtol: Locked relative tolerance, required to be zero.
        atol: Locked absolute tolerance, required to be `1e-12`.

    Returns:
        Aggregate-safe bitwise, maximum-difference, tolerance, and pass status.

    Raises:
        ValueError: If shape, dtype, order, finite status, or tolerance fails.
    """
    if float(rtol) != 0.0 or float(atol) != 1.0e-12:
        raise ValueError("Stage 4 reconciliation tolerance differs from authority.")
    replay = np.asarray(replayed_stage4)
    frozen = np.asarray(frozen_x)
    if replay.shape != frozen.shape or replay.ndim != 3:
        raise ValueError("Stage 4 reconciliation shape differs.")
    if frozen.shape[1:] != (int(frozen_clip_length), len(EXPECTED_CHANNELS)):
        raise ValueError("Frozen Pipeline A length or channel shape differs.")
    if int(frozen_clip_length) != replay.shape[1]:
        raise ValueError("Stage 4 reconciliation clip length differs.")
    if not np.issubdtype(replay.dtype, np.number) or not np.issubdtype(frozen.dtype, np.number):
        raise ValueError("Stage 4 reconciliation dtype is incompatible.")
    if not np.isfinite(replay).all() or not np.isfinite(frozen).all():
        raise ValueError("Stage 4 reconciliation requires finite tensors.")
    if not np.array_equal(np.asarray(replay_sample_ids), np.asarray(frozen_sample_ids)):
        raise ValueError("Stage 4 reconciliation participant order differs.")
    if tuple(np.asarray(frozen_channels).astype(str).tolist()) != EXPECTED_CHANNELS:
        raise ValueError("Stage 4 reconciliation channel order differs.")

    replay_float = np.asarray(replay, dtype=np.float64)
    frozen_float = np.asarray(frozen, dtype=np.float64)
    maximum_difference = float(np.max(np.abs(replay_float - frozen_float)))
    bitwise_equal = bool(np.array_equal(replay, frozen))
    passed = bool(
        np.allclose(replay_float, frozen_float, rtol=0.0, atol=1.0e-12, equal_nan=False)
        and maximum_difference <= 1.0e-12
    )
    if not passed:
        raise ValueError("Stage 4 reconciliation failed.")
    return {
        "shape_match": True,
        "order_match": True,
        "channel_match": True,
        "clip_length_match": True,
        "bitwise_equal": bitwise_equal,
        "maximum_absolute_difference": maximum_difference,
        "rtol": 0.0,
        "atol": 1.0e-12,
        "passed": True,
    }


def validate_serialized_privacy(
    filename: str,
    payload: bytes,
    *,
    allowed_filenames: list[str],
    private_values: tuple[str, ...] = (),
) -> None:
    """Validate one serialized aggregate candidate without echoing matches.

    Args:
        filename: Candidate basename to check against the exact allowlist.
        payload: Serialized candidate bytes.
        allowed_filenames: Locked complete candidate filename set.
        private_values: In-memory private strings that must never be persisted.

    Raises:
        ValueError: If filename, UTF-8 encoding, schema markers, dates, paths,
            or known private values violate the aggregate-only contract.
    """
    if filename not in allowed_filenames:
        raise ValueError("Unexpected candidate filename.")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Privacy validation failed.") from exc
    lowered = text.lower()
    for field in FORBIDDEN_OUTPUT_FIELDS:
        quoted_key = re.compile(rf'["\']{re.escape(field)}["\']\s*:')
        csv_field = re.compile(rf"(?:^|,){re.escape(field)}(?:,|$)", flags=re.MULTILINE)
        if quoted_key.search(lowered) or csv_field.search(lowered):
            raise ValueError("Privacy validation failed.")
    absolute_posix_path = re.compile(
        r"(?:^|[\s\"'=:])/(?!/)(?:[^/\s\"'<>:,]+/)*[^/\s\"'<>:,}\]]+"
    )
    if absolute_posix_path.search(text) or re.search(r"(?:[a-z]:\\\\|\\\\\\\\wsl)", lowered):
        raise ValueError("Privacy validation failed.")

    without_provenance_timestamp = re.sub(
        r'"provenance_timestamp"\s*:\s*"[^"]*"',
        '"provenance_timestamp":"ALLOWLISTED"',
        text,
        flags=re.IGNORECASE,
    )
    if re.search(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b", without_provenance_timestamp):
        raise ValueError("Privacy validation failed.")
    if any(value and value in text for value in private_values):
        raise ValueError("Privacy validation failed.")


def sha256_bytes(payload: bytes) -> str:
    """Return a lowercase SHA-256 digest for serialized aggregate bytes."""
    return hashlib.sha256(payload).hexdigest()


def serialize_json(payload: dict[str, Any]) -> bytes:
    """Serialize deterministic JSON with sorted keys, indentation, and LF.

    Args:
        payload: Aggregate-safe JSON mapping.

    Returns:
        Deterministic UTF-8 JSON bytes.

    Raises:
        ValueError: If non-finite JSON numbers are present.
        TypeError: If a value is not JSON serializable.
    """
    return (
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False, separators=(",", ": "))
        + "\n"
    ).encode("utf-8")


def _validate_environment_provenance(provenance: Any, contract: dict[str, Any]) -> None:
    """Validate aggregate-safe environment provenance against the immutable lock binding."""
    required_keys = {
        "environment_lock_sha256",
        "critical_numerical_builds",
        "package_ledger_equality_passed",
        "archive_hash_validation_passed",
    }
    if not isinstance(provenance, dict) or set(provenance) != required_keys:
        raise ValueError("Environment provenance schema differs.")
    if provenance["environment_lock_sha256"] != contract["environment_lock"]["sha256"]:
        raise ValueError("Environment-lock provenance hash differs.")
    critical = provenance["critical_numerical_builds"]
    if not isinstance(critical, list) or len(critical) != len(EXPECTED_CRITICAL_NUMERICAL_NAMES):
        raise ValueError("Critical numerical-build provenance differs.")
    if [record.get("name") for record in critical if isinstance(record, dict)] != list(
        EXPECTED_CRITICAL_NUMERICAL_NAMES
    ):
        raise ValueError("Critical numerical-build provenance order differs.")
    if any(
        not isinstance(record, dict)
        or set(record) != set(ENVIRONMENT_PACKAGE_KEYS)
        or any(not isinstance(record[key], str) or not record[key] for key in ENVIRONMENT_PACKAGE_KEYS)
        for record in critical
    ):
        raise ValueError("Critical numerical-build provenance record differs.")
    if provenance["package_ledger_equality_passed"] is not True:
        raise ValueError("Environment package-ledger validation did not pass.")
    if provenance["archive_hash_validation_passed"] is not True:
        raise ValueError("Environment archive validation did not pass.")


def _aggregate_safe_environment_provenance(
    runtime_evidence: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Project runtime evidence onto the exact aggregate-safe persisted schema."""
    safe_keys = (
        "environment_lock_sha256",
        "critical_numerical_builds",
        "package_ledger_equality_passed",
        "archive_hash_validation_passed",
    )
    try:
        provenance = {key: copy.deepcopy(runtime_evidence[key]) for key in safe_keys}
    except (KeyError, TypeError) as exc:
        raise ValueError("Environment provenance is incomplete.") from exc
    _validate_environment_provenance(provenance, contract)
    return provenance


def build_aggregate_audit(
    contract: dict[str, Any],
    *,
    authority: dict[str, Any],
    implementation_hashes: dict[str, str],
    package_versions: dict[str, str],
    environment_provenance: dict[str, Any],
    source_identities: dict[str, Any],
    environmental_summary_reconciliation_passed: bool,
    stage4_reconciliation: dict[str, Any],
    raw_cohort_digest: str,
    source_entry_exit_identity_passed: bool,
    deterministic_rerun_passed: bool,
    privacy_validation_passed: bool,
) -> dict[str, Any]:
    """Build the aggregate-safe deterministic execution audit mapping.

    Args:
        contract: Validated runtime contract.
        authority: Verified aggregate-safe Git/runtime authority snapshot.
        implementation_hashes: Config, module, and script SHA-256 values.
        package_versions: Installed computation package versions.
        environment_provenance: Aggregate-safe locked numerical-build evidence.
        source_identities: Complete aggregate-safe frozen-source identity ledger.
        environmental_summary_reconciliation_passed: S12/S13 audit gate status.
        stage4_reconciliation: Aggregate-safe Pipeline A reconciliation result.
        raw_cohort_digest: One cohort-level canonical raw-source digest.
        source_entry_exit_identity_passed: Declared source identity gate status.
        deterministic_rerun_passed: Two-isolated-computation equality status.
        privacy_validation_passed: Complete candidate privacy gate status.

    Returns:
        Aggregate-only audit mapping ready for deterministic JSON serialization.

    Raises:
        ValueError: If any required status is false or digest/hash shape differs.
    """
    required_hash_keys = {"config_sha256", "module_sha256", "script_sha256"}
    if set(implementation_hashes) != required_hash_keys or any(
        not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in implementation_hashes.values()
    ):
        raise ValueError("Implementation hash contract differs.")
    if not re.fullmatch(r"[0-9a-f]{64}", raw_cohort_digest):
        raise ValueError("Aggregate raw-cohort digest is invalid.")
    persisted_environment = _aggregate_safe_environment_provenance(environment_provenance, contract)
    statuses = (
        environmental_summary_reconciliation_passed,
        stage4_reconciliation.get("passed") is True,
        source_entry_exit_identity_passed,
        deterministic_rerun_passed,
        privacy_validation_passed,
    )
    if not all(statuses):
        raise ValueError("Aggregate audit cannot declare an incomplete execution gate.")

    population = contract["population"]
    stages = contract["stages"]
    return {
        "analysis_classification": contract["analysis_classification"],
        "analysis_id": contract["analysis_id"],
        "approved_plan": dict(contract["approved_plan"]),
        "branch": authority["branch"],
        "channels": list(contract["channels"]),
        "declared_lack_of_stochastic_operations": bool(contract["no_stochastic_operations"]),
        "deterministic_rerun": {"passed": True},
        "environmental_summary_reconciliation": {
            "atol": float(contract["environmental_summary_reconciliation"]["atol"]),
            "passed": True,
            "rtol": float(contract["environmental_summary_reconciliation"]["rtol"]),
        },
        "environment_provenance": persisted_environment,
        "execution_access_state": "authorized_by_exact_mandatory_gate_2",
        "figure_design_authorization": False,
        "finite_value_validation": {"passed": True},
        "head": authority["head"],
        "implementation_hashes": dict(implementation_hashes),
        "original_plan": dict(contract["original_plan"]),
        "package_versions": dict(package_versions),
        "participant_level_persistence": False,
        "population_counts": {
            "excluded_before_post_qc_transformations": int(population["expected_excluded_count"]),
            "fallback_t_baseline_end_excluded_cohort": int(population["expected_excluded_fallback_count"]),
            "fallback_t_baseline_end_final_cohort": int(population["expected_final_fallback_count"]),
            "fallback_t_baseline_end_source_cohort": int(population["expected_source_fallback_count"]),
            "final_cohort": int(population["expected_final_count"]),
            "source_cohort": int(population["expected_source_count"]),
            "tail_padded_final_signals": int(population["expected_padded_count"]),
        },
        "preprocessing_config_sha256": None,
        "preprocessing_config_sha256_status": CONFIG_ANOMALY_STATUS,
        "preprocessing_frozen_commit": authority["preprocessing_frozen_commit"],
        "privacy_validation": {"passed": True},
        "prohibited_field_access": False,
        "quantiles": {
            "method": contract["numpy_quantile_method"],
            "summaries": [item["output_name"] for item in contract["quantiles"]],
        },
        "raw_cohort_sha256": raw_cohort_digest,
        "recovery_v2_plan": dict(contract["recovery_v2_plan"]),
        "recovery_v3_plan": dict(contract["recovery_v3_plan"]),
        "reporting_correction": dict(contract["reporting_correction"]),
        "numerical_environment_correction": dict(contract["numerical_environment_correction"]),
        "channel_metadata_correction": dict(contract["channel_metadata_correction"]),
        "release_authorization": False,
        "response_frames": {
            "first": int(contract["response_axis"]["first"]),
            "last": int(contract["response_axis"]["last"]),
            "length": int(contract["response_axis"]["clip_length"]),
        },
        "source_entry_exit_identity": {"passed": True},
        "source_identities": source_identities,
        "stage4_reconciliation": dict(stage4_reconciliation),
        "stages": [
            {"id": stage["id"], "order": int(stage["order"]), "unit": stage["unit"]} for stage in stages
        ],
        "output_state": contract["output_status"],
    }


def build_run_summary(audit: dict[str, Any]) -> bytes:
    """Build the concise aggregate-only technical run summary.

    Args:
        audit: Validated aggregate execution audit mapping.

    Returns:
        Deterministic UTF-8 Markdown bytes without timestamps or signal values.
    """
    counts = audit["population_counts"]
    stage4 = audit["stage4_reconciliation"]
    environment = audit["environment_provenance"]
    channels = ", ".join(audit["channels"])
    lines = [
        "# eFigure 4 Preprocessing-Stage Aggregate Replay Run Summary",
        "",
        "## Classification",
        "",
        (
            "This is a post-freeze, result-blind, plan-orthogonal descriptive reporting-support replay. It is "
            "aggregate-only"
        ),
        (
            "technical evidence pending independent human review; it is not a prespecified diagnostic or performance "
            "analysis."
        ),
        "",
        "## Population and axes",
        "",
        (
            f"The source audit cohort contained {counts['source_cohort']} records; all and only the "
            f"{counts['final_cohort']} "
        ),
        "QC-retained modeling records were replayed identically across all four stages. The ",
        (
            f"{counts['excluded_before_post_qc_transformations']} excluded records did not undergo the post-QC "
            "transformations."
        ),
        (
            f"Fallback baseline-end resolution occurred for "
            f"{counts['fallback_t_baseline_end_source_cohort']} fallback resolutions among all "
            f"{counts['source_cohort']} source records, "
            f"{counts['fallback_t_baseline_end_final_cohort']} fallback resolutions among the "
            f"{counts['final_cohort']} final records, and "
            f"{counts['fallback_t_baseline_end_excluded_cohort']} fallback resolutions among the "
            f"{counts['excluded_before_post_qc_transformations']} excluded records."
        ),
        (
            f"The denominator-explicit reconciliation was "
            f"{counts['fallback_t_baseline_end_source_cohort']} = "
            f"{counts['fallback_t_baseline_end_final_cohort']} + "
            f"{counts['fallback_t_baseline_end_excluded_cohort']}; tail padding occurred for "
            f"{counts['tail_padded_final_signals']} final records."
        ),
        "Channels remain in frozen order: " + channels + ". The response axis is integer frame 0 through ",
        f"{audit['response_frames']['last']}; it was not converted to seconds.",
        "",
        "## Frozen replay and aggregation",
        "",
        "The replay used the persisted baseline end, frozen Savitzky-Golay smoothing, deserialized environmental",
        "compensation, frozen fractional baseline correction, and the common frozen trim/tail-padding rule. The raw,",
        "smoothed, and compensated common windows are diagnostic projections for aggregate cross-stage comparability;",
        "they do not imply that the historical pipeline cropped those full traces before later transformations.",
        (
            "Each stage was summarized independently by channel and response frame using NumPy linear q25, median, "
            "and q75."
        ),
        "",
        "## Reconciliation, privacy, and determinism",
        "",
        (
            "The replay used the immutable historical numerical environment bound by environment-lock SHA-256 "
            f"{environment['environment_lock_sha256']}."
        ),
        "The allowlisted critical numerical builds were:",
        *[
            (
                f"- {record['name']} {record['version']} {record['build']} "
                f"({record['channel']}, {record['subdirectory']})"
            )
            for record in environment["critical_numerical_builds"]
        ],
        (
            "Complete package-ledger equality and all six cached-archive SHA-256 validations passed at entry and "
            "before promotion."
        ),
        "",
        f"Stage 4 reconciliation passed; bitwise equality was {str(stage4['bitwise_equal']).lower()} and the maximum ",
        (
            f"absolute difference was {stage4['maximum_absolute_difference']:.17g} at rtol=0 and atol=1e-12. "
            "Environmental "
        ),
        "summary reconciliation, finite-value validation, aggregate-only privacy validation, frozen-source entry/exit",
        (
            "identity, and two isolated byte-identical computations all passed. No participant-level array was "
            "persisted and"
        ),
        "no prohibited field was accessed.",
        "",
        "## Interpretation boundaries",
        "",
        "Applying environmental compensation does not prove that environmental confounding was eliminated.",
        "The replay does not prove removal of all noise or artifacts.",
        "It does not assess disease discrimination, clinical utility, model improvement, or mechanism.",
        "Cross-responsive channels do not identify individual VOCs.",
        "The artifact does not authorize figure design or release.",
        "",
        (
            "Status: executed aggregate-only evidence; pending independent human review; not authorized for figure "
            "design or release."
        ),
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def build_candidate_payloads(
    summary: pd.DataFrame,
    *,
    audit: dict[str, Any],
    contract: dict[str, Any],
    source_identities: dict[str, Any],
    implementation_hashes: dict[str, str],
) -> dict[str, bytes]:
    """Build the exact deterministic four-file aggregate-only candidate.

    Args:
        summary: Validated aggregate stage/channel/frame table.
        audit: Aggregate-safe execution audit mapping.
        contract: Validated runtime contract.
        source_identities: Aggregate-safe declared source identity ledger.
        implementation_hashes: Config, module, and script SHA-256 values.

    Returns:
        Ordered filename-to-bytes mapping for the complete candidate.
    """
    if audit.get("source_identities") != source_identities:
        raise ValueError("Audit and manifest source-identity ledgers differ.")
    csv_payload = serialize_aggregate_csv(summary, float_format=contract["serialization"]["csv_float_format"])
    audit_payload = serialize_json(audit)
    run_summary_payload = build_run_summary(audit)
    final_directory = contract["outputs"]["final_directory"]
    manifest = {
        "approved_plan_sha256": contract["approved_plan"]["sha256"],
        "channel_metadata_correction_sha256": contract["channel_metadata_correction"]["sha256"],
        "declared_output_filename_order": list(contract["outputs"]["allowed_filenames"]),
        "environment_provenance": copy.deepcopy(audit["environment_provenance"]),
        "implementation_hashes": dict(implementation_hashes),
        "numerical_environment_correction_sha256": contract["numerical_environment_correction"]["sha256"],
        "original_plan_sha256": contract["original_plan"]["sha256"],
        "outputs": [
            {"path": f"{final_directory}/stage_channel_frame_summary.csv", "sha256": sha256_bytes(csv_payload)},
            {"path": f"{final_directory}/aggregate_audit.json", "sha256": sha256_bytes(audit_payload)},
            {"path": f"{final_directory}/run_summary.md", "sha256": sha256_bytes(run_summary_payload)},
        ],
        "privacy_validation": {"passed": True},
        "read_back_validation": {"passed": True},
        "recovery_v2_plan_sha256": contract["recovery_v2_plan"]["sha256"],
        "recovery_v3_plan_sha256": contract["recovery_v3_plan"]["sha256"],
        "reporting_correction_sha256": contract["reporting_correction"]["sha256"],
        "self_hash": None,
        "self_hash_policy": "forbidden",
        "source_entry_exit_validation": {"passed": True},
        "source_identities": source_identities,
    }
    manifest_payload = serialize_json(manifest)
    return {
        "stage_channel_frame_summary.csv": csv_payload,
        "aggregate_audit.json": audit_payload,
        "run_summary.md": run_summary_payload,
        "manifest.json": manifest_payload,
    }


def _parse_aggregate_csv(
    payload: bytes,
    *,
    contract: dict[str, Any],
    expected_participants: int,
    expected_frames: int,
) -> pd.DataFrame:
    """Parse serialized aggregate CSV with explicit output dtypes."""
    dtype = {
        "stage_order": np.int64,
        "stage_id": str,
        "channel": str,
        "response_frame": np.int64,
        "unit": str,
        "n_participants": np.int64,
        "q25": np.float64,
        "median": np.float64,
        "q75": np.float64,
    }
    try:
        frame = pd.read_csv(io.BytesIO(payload), dtype=dtype)
    except (ValueError, pd.errors.ParserError) as exc:
        raise ValueError("Aggregate CSV read-back failed.") from exc
    validate_aggregate_summary(
        frame,
        stages=contract["stages"],
        channels=contract["channels"],
        expected_participants=expected_participants,
        expected_frames=expected_frames,
    )
    return frame


def _validate_source_identity_ledger(ledger: Any, contract: dict[str, Any]) -> None:
    """Validate the complete aggregate-safe frozen-source identity ledger."""
    if not isinstance(ledger, dict) or set(ledger) != set(contract["inputs"]):
        raise ValueError("Source-identity ledger schema differs.")
    for name, expected in contract["inputs"].items():
        identity = ledger.get(name)
        if not isinstance(identity, dict):
            raise ValueError("Source-identity ledger entry is invalid.")
        if name == "preprocessing_config":
            required_keys = {"path", "git_blob", "sha256", "sha256_status", "byte_size", "blake2b"}
            if set(identity) != required_keys:
                raise ValueError("Preprocessing-config source identity schema differs.")
            if identity != {
                "path": expected["path"],
                "git_blob": expected["git_blob"],
                "sha256": None,
                "sha256_status": CONFIG_ANOMALY_STATUS,
                "byte_size": int(expected["byte_size"]),
                "blake2b": expected["blake2b"],
            }:
                raise ValueError("Preprocessing-config source identity differs.")
            continue
        if set(identity) != {"path", "git_blob", "sha256", "byte_size"}:
            raise ValueError("Ordinary source identity schema differs.")
        if identity.get("path") != expected["path"] or identity.get("git_blob") != expected["git_blob"]:
            raise ValueError("Ordinary source path or Git blob differs.")
        if not isinstance(identity.get("byte_size"), int) or identity["byte_size"] < 0:
            raise ValueError("Ordinary source byte size is invalid.")
        if not isinstance(identity.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", identity["sha256"]):
            raise ValueError("Ordinary source SHA-256 is invalid.")


def _validate_stage4_audit(stage4: Any, contract: dict[str, Any]) -> None:
    """Validate the exact aggregate-safe Stage 4 reconciliation schema."""
    required_keys = {
        "shape_match",
        "order_match",
        "channel_match",
        "clip_length_match",
        "bitwise_equal",
        "maximum_absolute_difference",
        "rtol",
        "atol",
        "passed",
    }
    if not isinstance(stage4, dict) or set(stage4) != required_keys:
        raise ValueError("Stage 4 reconciliation schema differs.")
    for field in ("shape_match", "order_match", "channel_match", "clip_length_match", "passed"):
        if stage4[field] is not True:
            raise ValueError("Stage 4 reconciliation required pass status differs.")
    if not isinstance(stage4["bitwise_equal"], bool):
        raise ValueError("Stage 4 bitwise-equality status is invalid.")
    if float(stage4["rtol"]) != float(contract["reconciliation"]["rtol"]) or float(stage4["atol"]) != float(
        contract["reconciliation"]["atol"]
    ):
        raise ValueError("Stage 4 reconciliation tolerance differs.")
    maximum_difference = stage4["maximum_absolute_difference"]
    if (
        not isinstance(maximum_difference, (int, float))
        or not np.isfinite(float(maximum_difference))
        or float(maximum_difference) < 0.0
        or float(maximum_difference) > float(contract["reconciliation"]["atol"])
    ):
        raise ValueError("Stage 4 maximum absolute difference is invalid.")
    if stage4["bitwise_equal"] is True and float(maximum_difference) != 0.0:
        raise ValueError("Stage 4 bitwise status and maximum difference disagree.")


def _validate_aggregate_audit(
    audit: Any,
    *,
    contract: dict[str, Any],
    expected_participants: int,
    expected_frames: int,
) -> None:
    """Validate the exact aggregate-audit schema and contract-critical content."""
    required_keys = {
        "analysis_classification",
        "analysis_id",
        "approved_plan",
        "branch",
        "channels",
        "declared_lack_of_stochastic_operations",
        "deterministic_rerun",
        "environment_provenance",
        "environmental_summary_reconciliation",
        "execution_access_state",
        "figure_design_authorization",
        "finite_value_validation",
        "head",
        "implementation_hashes",
        "original_plan",
        "recovery_v2_plan",
        "recovery_v3_plan",
        "numerical_environment_correction",
        "channel_metadata_correction",
        "package_versions",
        "participant_level_persistence",
        "population_counts",
        "preprocessing_config_sha256",
        "preprocessing_config_sha256_status",
        "preprocessing_frozen_commit",
        "privacy_validation",
        "prohibited_field_access",
        "quantiles",
        "raw_cohort_sha256",
        "reporting_correction",
        "release_authorization",
        "response_frames",
        "source_entry_exit_identity",
        "source_identities",
        "stage4_reconciliation",
        "stages",
        "output_state",
    }
    if not isinstance(audit, dict) or set(audit) != required_keys:
        raise ValueError("Aggregate audit top-level schema differs.")
    if audit["analysis_id"] != contract["analysis_id"] or audit["analysis_classification"] != contract[
        "analysis_classification"
    ]:
        raise ValueError("Aggregate audit analysis identity differs.")
    if audit["approved_plan"] != contract["approved_plan"]:
        raise ValueError("Aggregate audit approved plan differs.")
    if audit["original_plan"] != contract["original_plan"]:
        raise ValueError("Aggregate audit original plan differs.")
    if audit["recovery_v2_plan"] != contract["recovery_v2_plan"]:
        raise ValueError("Aggregate audit recovery-v2 plan differs.")
    if audit["recovery_v3_plan"] != contract["recovery_v3_plan"]:
        raise ValueError("Aggregate audit recovery-v3 plan differs.")
    if audit["reporting_correction"] != contract["reporting_correction"]:
        raise ValueError("Aggregate audit reporting correction differs.")
    if audit["numerical_environment_correction"] != contract["numerical_environment_correction"]:
        raise ValueError("Aggregate audit numerical-environment correction differs.")
    if audit["channel_metadata_correction"] != contract["channel_metadata_correction"]:
        raise ValueError("Aggregate audit channel-metadata correction differs.")
    _validate_environment_provenance(audit["environment_provenance"], contract)
    if (
        audit["branch"] != contract["runtime_authority"]["expected_branch"]
        or audit["head"] != contract["runtime_authority"]["expected_head"]
        or audit["preprocessing_frozen_commit"]
        != contract["runtime_authority"]["preprocessing_frozen_commit"]
    ):
        raise ValueError("Aggregate audit Git authority differs.")
    if audit["execution_access_state"] != "authorized_by_exact_mandatory_gate_2":
        raise ValueError("Aggregate audit execution access state differs.")

    implementation_hashes = audit["implementation_hashes"]
    required_hash_keys = {"config_sha256", "module_sha256", "script_sha256"}
    if not isinstance(implementation_hashes, dict) or set(implementation_hashes) != required_hash_keys or any(
        not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
        for value in implementation_hashes.values()
    ):
        raise ValueError("Aggregate audit implementation hashes differ.")
    package_versions = audit["package_versions"]
    if not isinstance(package_versions, dict) or set(package_versions) != {
        "python",
        "numpy",
        "pandas",
        "scipy",
        "pyyaml",
    } or any(not isinstance(value, str) or not value for value in package_versions.values()):
        raise ValueError("Aggregate audit package-version ledger differs.")

    population = contract["population"]
    expected_counts = {
        "excluded_before_post_qc_transformations": int(population["expected_excluded_count"]),
        "fallback_t_baseline_end_excluded_cohort": int(population["expected_excluded_fallback_count"]),
        "fallback_t_baseline_end_final_cohort": int(population["expected_final_fallback_count"]),
        "fallback_t_baseline_end_source_cohort": int(population["expected_source_fallback_count"]),
        "final_cohort": int(population["expected_final_count"]),
        "source_cohort": int(population["expected_source_count"]),
        "tail_padded_final_signals": int(population["expected_padded_count"]),
    }
    if audit["population_counts"] != expected_counts or expected_participants != expected_counts["final_cohort"]:
        raise ValueError("Aggregate audit population counts differ.")
    expected_stages = [
        {"id": stage["id"], "order": int(stage["order"]), "unit": stage["unit"]}
        for stage in contract["stages"]
    ]
    if audit["stages"] != expected_stages or audit["channels"] != contract["channels"]:
        raise ValueError("Aggregate audit stage, channel, or unit definitions differ.")
    expected_response_frames = {
        "first": int(contract["response_axis"]["first"]),
        "last": int(contract["response_axis"]["last"]),
        "length": int(contract["response_axis"]["clip_length"]),
    }
    if audit["response_frames"] != expected_response_frames or expected_frames != expected_response_frames["length"]:
        raise ValueError("Aggregate audit response-frame definition differs.")
    expected_quantiles = {
        "method": contract["numpy_quantile_method"],
        "summaries": [item["output_name"] for item in contract["quantiles"]],
    }
    if audit["quantiles"] != expected_quantiles:
        raise ValueError("Aggregate audit quantile definition differs.")

    expected_environment = {
        "atol": float(contract["environmental_summary_reconciliation"]["atol"]),
        "passed": True,
        "rtol": float(contract["environmental_summary_reconciliation"]["rtol"]),
    }
    if audit["environmental_summary_reconciliation"] != expected_environment:
        raise ValueError("Aggregate audit environmental reconciliation differs.")
    for field in ("finite_value_validation", "privacy_validation", "deterministic_rerun", "source_entry_exit_identity"):
        if audit[field] != {"passed": True}:
            raise ValueError("Aggregate audit required pass status differs.")
    _validate_stage4_audit(audit["stage4_reconciliation"], contract)
    _validate_source_identity_ledger(audit["source_identities"], contract)

    if audit["preprocessing_config_sha256"] is not None or audit[
        "preprocessing_config_sha256_status"
    ] != CONFIG_ANOMALY_STATUS:
        raise ValueError("Aggregate audit preprocessing-config anomaly status differs.")
    if not isinstance(audit["raw_cohort_sha256"], str) or not re.fullmatch(
        r"[0-9a-f]{64}", audit["raw_cohort_sha256"]
    ):
        raise ValueError("Aggregate audit raw-cohort digest differs.")
    if audit["declared_lack_of_stochastic_operations"] is not True:
        raise ValueError("Aggregate audit stochastic-operation declaration differs.")
    if audit["participant_level_persistence"] is not False or audit["prohibited_field_access"] is not False:
        raise ValueError("Aggregate audit participant privacy declarations differ.")
    if audit["output_state"] != contract["output_status"]:
        raise ValueError("Aggregate audit output state differs.")
    if audit["figure_design_authorization"] is not False or audit["release_authorization"] is not False:
        raise ValueError("Aggregate audit design or release authorization differs.")


def _validate_manifest(
    manifest: Any,
    *,
    audit: dict[str, Any],
    payloads: dict[str, bytes],
    contract: dict[str, Any],
) -> None:
    """Validate exact manifest schema, ledgers, hashes, and pass statuses."""
    required_keys = {
        "approved_plan_sha256",
        "channel_metadata_correction_sha256",
        "declared_output_filename_order",
        "environment_provenance",
        "implementation_hashes",
        "numerical_environment_correction_sha256",
        "original_plan_sha256",
        "outputs",
        "privacy_validation",
        "read_back_validation",
        "recovery_v2_plan_sha256",
        "recovery_v3_plan_sha256",
        "reporting_correction_sha256",
        "self_hash",
        "self_hash_policy",
        "source_entry_exit_validation",
        "source_identities",
    }
    if not isinstance(manifest, dict) or set(manifest) != required_keys:
        raise ValueError("Manifest top-level schema differs.")
    if manifest["approved_plan_sha256"] != contract["approved_plan"]["sha256"]:
        raise ValueError("Manifest approved plan hash differs.")
    if manifest["original_plan_sha256"] != contract["original_plan"]["sha256"]:
        raise ValueError("Manifest original plan hash differs.")
    if manifest["recovery_v2_plan_sha256"] != contract["recovery_v2_plan"]["sha256"]:
        raise ValueError("Manifest recovery-v2 plan hash differs.")
    if manifest["recovery_v3_plan_sha256"] != contract["recovery_v3_plan"]["sha256"]:
        raise ValueError("Manifest recovery-v3 plan hash differs.")
    if manifest["reporting_correction_sha256"] != contract["reporting_correction"]["sha256"]:
        raise ValueError("Manifest reporting-correction hash differs.")
    if manifest["numerical_environment_correction_sha256"] != contract["numerical_environment_correction"]["sha256"]:
        raise ValueError("Manifest numerical-environment correction hash differs.")
    if manifest["channel_metadata_correction_sha256"] != contract["channel_metadata_correction"]["sha256"]:
        raise ValueError("Manifest channel-metadata correction hash differs.")
    if manifest["environment_provenance"] != audit["environment_provenance"]:
        raise ValueError("Manifest and audit environment provenance differs.")
    _validate_environment_provenance(manifest["environment_provenance"], contract)
    if manifest["implementation_hashes"] != audit["implementation_hashes"]:
        raise ValueError("Manifest and audit implementation hashes differ.")
    if manifest["source_identities"] != audit["source_identities"]:
        raise ValueError("Manifest and audit source identities differ.")
    _validate_source_identity_ledger(manifest["source_identities"], contract)
    allowed = contract["outputs"]["allowed_filenames"]
    if manifest["declared_output_filename_order"] != allowed:
        raise ValueError("Manifest output filename order differs.")
    expected_output_entries = [
        {
            "path": f"{contract['outputs']['final_directory']}/{filename}",
            "sha256": sha256_bytes(payloads[filename]),
        }
        for filename in allowed[:3]
    ]
    if manifest["outputs"] != expected_output_entries:
        raise ValueError("Manifest output hashes differ from candidate bytes.")
    for field in ("privacy_validation", "read_back_validation", "source_entry_exit_validation"):
        if manifest[field] != {"passed": True}:
            raise ValueError("Manifest required pass status differs.")
    if manifest["self_hash"] is not None or manifest["self_hash_policy"] != "forbidden":
        raise ValueError("Manifest self-hash policy differs.")


def validate_candidate_payloads(
    payloads: dict[str, bytes],
    *,
    contract: dict[str, Any],
    expected_participants: int,
    expected_frames: int,
    expected_audit: dict[str, Any],
    private_values: tuple[str, ...] = (),
) -> None:
    """Validate a complete in-memory or read-back aggregate candidate.

    Args:
        payloads: Ordered complete candidate bytes.
        contract: Validated runtime contract.
        expected_participants: Required aggregate participant count.
        expected_frames: Required aggregate response-frame count.
        expected_audit: Exact execution-time aggregate audit mapping.
        private_values: In-memory raw filenames that must not appear.

    Raises:
        ValueError: If filenames, privacy, schema, hashes, or status differs.
    """
    allowed = contract["outputs"]["allowed_filenames"]
    if list(payloads) != allowed:
        raise ValueError("Candidate filename set or order differs.")
    for filename, payload in payloads.items():
        validate_serialized_privacy(
            filename,
            payload,
            allowed_filenames=allowed,
            private_values=private_values,
        )
    summary = _parse_aggregate_csv(
        payloads["stage_channel_frame_summary.csv"],
        contract=contract,
        expected_participants=expected_participants,
        expected_frames=expected_frames,
    )
    if len(summary) != int(contract["outputs"]["expected_aggregate_rows"]):
        raise ValueError("Candidate aggregate row count differs from the contract.")
    try:
        audit = json.loads(payloads["aggregate_audit.json"])
        manifest = json.loads(payloads["manifest.json"])
    except json.JSONDecodeError as exc:
        raise ValueError("Candidate JSON read-back failed.") from exc
    _validate_aggregate_audit(
        audit,
        contract=contract,
        expected_participants=expected_participants,
        expected_frames=expected_frames,
    )
    if audit != expected_audit:
        raise ValueError("Candidate audit differs from the exact execution-time audit.")
    _validate_manifest(manifest, audit=audit, payloads=payloads, contract=contract)
    run_summary = payloads["run_summary.md"].decode("utf-8")
    if payloads["run_summary.md"] != build_run_summary(audit):
        raise ValueError("Run summary differs from the validated aggregate audit.")
    required_boundaries = (
        "Applying environmental compensation does not prove that environmental confounding was eliminated.",
        "The replay does not prove removal of all noise or artifacts.",
        "It does not assess disease discrimination, clinical utility, model improvement, or mechanism.",
        "Cross-responsive channels do not identify individual VOCs.",
        "The artifact does not authorize figure design or release.",
    )
    if not all(boundary in run_summary for boundary in required_boundaries):
        raise ValueError("Run-summary interpretation boundaries differ.")


def require_absent_output_namespace(path: Path) -> None:
    """Fail closed if a planned final or candidate namespace already exists.

    Args:
        path: Exact planned namespace.

    Raises:
        FileExistsError: If the path exists in any form.
    """
    if Path(path).exists() or Path(path).is_symlink():
        raise FileExistsError("Planned output namespace already exists.")


def write_candidate_directory(
    path: Path,
    payloads: dict[str, bytes],
    *,
    contract: dict[str, Any],
    expected_participants: int,
    expected_frames: int,
    expected_audit: dict[str, Any],
    private_values: tuple[str, ...] = (),
) -> None:
    """Write and read back one new complete aggregate-only candidate.

    Args:
        path: New destination-local candidate directory.
        payloads: Complete ordered four-file payload.
        contract: Validated runtime contract.
        expected_participants: Required aggregate participant count.
        expected_frames: Required aggregate response-frame count.
        expected_audit: Exact execution-time aggregate audit mapping.
        private_values: In-memory raw filenames forbidden from outputs.

    Raises:
        FileExistsError: If the candidate path already exists.
        OSError: If writing or reading fails.
        ValueError: If any candidate validation fails.
    """
    candidate = Path(path)
    require_absent_output_namespace(candidate)
    validate_candidate_payloads(
        payloads,
        contract=contract,
        expected_participants=expected_participants,
        expected_frames=expected_frames,
        expected_audit=expected_audit,
        private_values=private_values,
    )
    created = False
    try:
        candidate.mkdir(parents=False, exist_ok=False)
        created = True
        for filename in contract["outputs"]["allowed_filenames"]:
            destination = candidate / filename
            with destination.open("xb") as target:
                target.write(payloads[filename])
            read_back = destination.read_bytes()
            if read_back != payloads[filename]:
                raise ValueError("Candidate file read-back bytes differ.")
            validate_serialized_privacy(
                filename,
                read_back,
                allowed_filenames=contract["outputs"]["allowed_filenames"],
                private_values=private_values,
            )
        read_and_validate_candidate_directory(
            candidate,
            contract=contract,
            expected_participants=expected_participants,
            expected_frames=expected_frames,
            expected_audit=expected_audit,
            private_values=private_values,
        )
    except Exception:
        if created and candidate.exists():
            shutil.rmtree(candidate)
        raise


def read_and_validate_candidate_directory(
    path: Path,
    *,
    contract: dict[str, Any],
    expected_participants: int,
    expected_frames: int,
    expected_audit: dict[str, Any],
    private_values: tuple[str, ...] = (),
) -> dict[str, bytes]:
    """Read back and validate an exact aggregate candidate directory.

    Args:
        path: Candidate or promoted final directory.
        contract: Validated runtime contract.
        expected_participants: Required aggregate participant count.
        expected_frames: Required aggregate response-frame count.
        expected_audit: Exact execution-time aggregate audit mapping.
        private_values: In-memory raw filenames forbidden from outputs.

    Returns:
        Ordered filename-to-read-back-bytes mapping.

    Raises:
        OSError: If directory content cannot be read.
        ValueError: If filename set, bytes, privacy, or schema differs.
    """
    directory = Path(path)
    if not directory.is_dir():
        raise ValueError("Candidate directory is absent or invalid.")
    actual_names = [entry.name for entry in directory.iterdir()]
    allowed = contract["outputs"]["allowed_filenames"]
    if set(actual_names) != set(allowed) or len(actual_names) != len(allowed):
        raise ValueError("Candidate filename set differs.")
    payloads = {filename: (directory / filename).read_bytes() for filename in allowed}
    validate_candidate_payloads(
        payloads,
        contract=contract,
        expected_participants=expected_participants,
        expected_frames=expected_frames,
        expected_audit=expected_audit,
        private_values=private_values,
    )
    return payloads


def promote_candidate_directory(
    candidate_path: Path,
    final_path: Path,
    *,
    contract: dict[str, Any],
    expected_participants: int,
    expected_frames: int,
    expected_audit: dict[str, Any],
    private_values: tuple[str, ...] = (),
) -> None:
    """Atomically rename one validated candidate to an absent final namespace.

    Args:
        candidate_path: Complete destination-local candidate directory.
        final_path: Exact absent final output directory.
        contract: Validated runtime contract.
        expected_participants: Required aggregate participant count.
        expected_frames: Required aggregate response-frame count.
        expected_audit: Exact execution-time aggregate audit mapping.
        private_values: In-memory raw filenames forbidden from outputs.

    Raises:
        FileExistsError: If the final path exists.
        OSError: If atomic same-filesystem rename fails.
        ValueError: If the candidate/final contract differs.
    """
    candidate = Path(candidate_path)
    final = Path(final_path)
    require_absent_output_namespace(final)
    if candidate.parent.resolve() != final.parent.resolve():
        raise ValueError("Candidate promotion must remain destination-local.")
    read_and_validate_candidate_directory(
        candidate,
        contract=contract,
        expected_participants=expected_participants,
        expected_frames=expected_frames,
        expected_audit=expected_audit,
        private_values=private_values,
    )
    candidate.replace(final)
    try:
        read_and_validate_candidate_directory(
            final,
            contract=contract,
            expected_participants=expected_participants,
            expected_frames=expected_frames,
            expected_audit=expected_audit,
            private_values=private_values,
        )
    except Exception:
        if final.exists() and not candidate.exists():
            final.replace(candidate)
        raise


def require_source_identity_match(entry: dict[str, Any], exit_state: dict[str, Any]) -> None:
    """Require exact equality of declared frozen source entry/exit identities.

    Args:
        entry: Source identity ledger captured before computation.
        exit_state: Independently captured ledger before promotion.

    Raises:
        ValueError: If any declared source identity changed.
    """
    if entry != exit_state:
        raise ValueError("Frozen source entry/exit identity mismatch.")


def compute_raw_cohort_digest(cohort_rows: pd.DataFrame, raw_root: Path) -> str:
    """Hash one canonical ordered stream of aggregate raw-cohort file bytes.

    Args:
        cohort_rows: Canonically ordered in-memory frame containing `filename`.
        raw_root: Authorized raw breath-data root.

    Returns:
        One lowercase cohort-level SHA-256 digest.

    Raises:
        OSError: If a contained file cannot be read or changes during reading.
        ValueError: If required fields or raw containment differ.
    """
    if "filename" not in cohort_rows.columns or cohort_rows["filename"].isna().any():
        raise ValueError("Canonical raw-cohort filenames are unavailable in memory.")
    digest = hashlib.sha256()
    digest.update(b"GC_VOC_EFIGURE4_RAW_COHORT_V1\x00")
    for filename in cohort_rows["filename"].astype(str).tolist():
        raw_path = _resolve_contained_raw_path(filename, raw_root)
        expected_size = int(raw_path.stat().st_size)
        digest.update(expected_size.to_bytes(8, byteorder="big", signed=False))
        bytes_read = 0
        with raw_path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
                bytes_read += len(chunk)
        if bytes_read != expected_size or int(raw_path.stat().st_size) != expected_size:
            raise OSError("Raw cohort source changed during digest computation.")
    return digest.hexdigest()


def load_frozen_pipeline_a(path: Path) -> FrozenPipelineA:
    """Load only the four permitted arrays from frozen Pipeline A.

    Args:
        path: Frozen `pipeline_a_raw.npz` path.

    Returns:
        Copied permitted arrays with no label access.

    Raises:
        OSError: If the archive cannot be read.
        ValueError: If required members, shapes, dtypes, or values differ.
    """
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            required = {"x", "sample_id", "channel_names", "clip_length_L"}
            if not required.issubset(set(archive.files)):
                raise ValueError("Frozen Pipeline A is missing a permitted required member.")
            x = np.array(archive["x"], copy=True)
            sample_ids = np.array(archive["sample_id"], copy=True)
            channels = np.array(archive["channel_names"], copy=True)
            clip_value = np.array(archive["clip_length_L"], copy=True)
    except ValueError:
        raise
    except OSError as exc:
        raise OSError("Frozen Pipeline A loading failed.") from exc
    if x.ndim != 3 or not np.issubdtype(x.dtype, np.number) or not np.isfinite(x).all():
        raise ValueError("Frozen Pipeline A tensor is invalid.")
    if sample_ids.ndim != 1 or sample_ids.shape[0] != x.shape[0] or not np.issubdtype(sample_ids.dtype, np.integer):
        raise ValueError("Frozen Pipeline A participant order is invalid.")
    if channels.ndim != 1 or channels.shape[0] != x.shape[2]:
        raise ValueError("Frozen Pipeline A channel array is invalid.")
    if clip_value.size != 1:
        raise ValueError("Frozen Pipeline A clip length is not scalar.")
    return FrozenPipelineA(
        x=np.asarray(x, dtype=np.float64),
        sample_ids=np.asarray(sample_ids, dtype=np.int64),
        channels=np.asarray(channels).astype(str),
        clip_length=int(clip_value.reshape(-1)[0]),
    )


def load_frozen_environmental_model(path: Path) -> EnvCompensationModel:
    """Read frozen preprocessing YAML and deserialize one new environmental model.

    Args:
        path: Frozen preprocessing YAML path.

    Returns:
        Newly deserialized frozen environmental-compensation model.

    Raises:
        OSError: If the YAML file cannot be read.
        ValueError: If YAML or the environmental model payload is invalid.
    """
    try:
        preprocessing_payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError("Frozen preprocessing configuration is not valid YAML.") from exc
    if not isinstance(preprocessing_payload, dict) or not isinstance(
        preprocessing_payload.get("environmental_compensation"), dict
    ):
        raise ValueError("Frozen environmental-compensation configuration is missing.")
    return EnvCompensationModel.from_yaml_dict(preprocessing_payload["environmental_compensation"])


def compute_replay_and_aggregate(
    contract: dict[str, Any],
    cohort: CohortTables,
    *,
    env_model: EnvCompensationModel,
    raw_root: Path,
    pipeline_a_path: Path,
) -> ReplayComputation:
    """Run one isolated full replay, reconcile Stage 4, then aggregate.

    Args:
        contract: Validated runtime contract.
        cohort: Minimal canonical final-cohort audit state.
        env_model: Deserialized frozen environmental compensation model.
        raw_root: Authorized raw breath-data root.
        pipeline_a_path: Frozen Pipeline A archive path.

    Returns:
        Aggregate table and aggregate-safe Stage 4 reconciliation diagnostics.

    Raises:
        RawSignalLoadError: If one raw trace cannot be loaded safely.
        ValueError: If cohort, replay, tensor, or Pipeline A gates fail.
    """
    failure_codes = _ACTIVE_ISOLATED_FAILURE_CODES.get()
    cohort_code = failure_codes.minimal_cohort if failure_codes is not None else None
    with _optional_isolated_failure_boundary(cohort_code):
        expected_participants = int(contract["population"]["expected_final_count"])
        expected_frames = int(contract["response_axis"]["clip_length"])
        if cohort.final_count != expected_participants or len(cohort.final_rows) != expected_participants:
            raise ValueError("Replay cohort count differs from the locked contract.")
        if int(cohort.final_rows["fallback_used"].sum()) != int(
            contract["population"]["expected_final_fallback_count"]
        ):
            raise ValueError("Replay fallback count differs from the locked contract.")
        if int(cohort.final_rows["padded"].sum()) != int(contract["population"]["expected_padded_count"]):
            raise ValueError("Replay padding count differs from the locked contract.")
        if not cohort.final_rows["clip_length_L"].eq(expected_frames).all():
            raise ValueError("Replay persisted clip lengths differ from the locked contract.")

    stage_tensors = {
        stage_id: np.empty((expected_participants, expected_frames, len(EXPECTED_CHANNELS)), dtype=np.float64)
        for stage_id in EXPECTED_STAGE_IDS
    }
    for participant_index, row in enumerate(cohort.final_rows.itertuples(index=False)):
        raw_code = failure_codes.raw_signal if failure_codes is not None else None
        with _optional_isolated_failure_boundary(raw_code):
            raw = load_raw_signal(str(row.filename), raw_root, expected_n_rows=int(row.n_rows))
        replay = replay_participant(
            raw,
            t_baseline_end=int(row.t_baseline_end),
            persisted_temperature_sample=float(row.env_temperature_sample),
            persisted_humidity_sample=float(row.env_humidity_sample),
            expected_n_available=int(row.n_available),
            expected_padded=bool(row.padded),
            env_model=env_model,
            clip_length=expected_frames,
            padding_threshold=float(contract["padding_relative_deficit_threshold"]),
            environmental_rtol=float(contract["environmental_summary_reconciliation"]["rtol"]),
            environmental_atol=float(contract["environmental_summary_reconciliation"]["atol"]),
        )
        for stage_id in EXPECTED_STAGE_IDS:
            stage_tensors[stage_id][participant_index, :, :] = replay.stages[stage_id]
    trimming_code = failure_codes.trimming if failure_codes is not None else None
    with _optional_isolated_failure_boundary(trimming_code):
        validate_stage_tensors(
            stage_tensors,
            expected_participants=expected_participants,
            expected_frames=expected_frames,
        )

    pipeline_code = failure_codes.pipeline_a if failure_codes is not None else None
    with _optional_isolated_failure_boundary(pipeline_code):
        frozen = load_frozen_pipeline_a(pipeline_a_path)
    stage4_code = failure_codes.stage4 if failure_codes is not None else None
    with _optional_isolated_failure_boundary(stage4_code):
        reconciliation = reconcile_stage4(
            stage_tensors[EXPECTED_STAGE_IDS[-1]],
            cohort.final_rows["sample_id"].to_numpy(dtype=np.int64),
            frozen.x,
            frozen.sample_ids,
            frozen.channels,
            frozen_clip_length=frozen.clip_length,
            rtol=float(contract["reconciliation"]["rtol"]),
            atol=float(contract["reconciliation"]["atol"]),
        )
    aggregation_code = failure_codes.aggregation if failure_codes is not None else None
    with _optional_isolated_failure_boundary(aggregation_code):
        summary = aggregate_stage_tensors(
            stage_tensors,
            stages=contract["stages"],
            channels=contract["channels"],
            expected_participants=expected_participants,
            expected_frames=expected_frames,
            quantile_method=contract["numpy_quantile_method"],
        )
    return ReplayComputation(summary=summary, stage4_reconciliation=reconciliation)


def compute_isolated_replay_and_aggregate(
    contract: dict[str, Any],
    *,
    sample_index_path: Path,
    qc_diagnostics_path: Path,
    preprocessing_config_path: Path,
    raw_root: Path,
    pipeline_a_path: Path,
) -> IsolatedReplayComputation:
    """Reload every participant-capable source for one isolated computation.

    Args:
        contract: Validated runtime contract.
        sample_index_path: Frozen minimal cohort-index source.
        qc_diagnostics_path: Frozen minimal persisted preprocessing audit source.
        preprocessing_config_path: Frozen preprocessing YAML source.
        raw_root: Authorized raw breath-data root.
        pipeline_a_path: Frozen Pipeline A archive path.

    Returns:
        Aggregate result plus independent raw-source entry/exit digests.

    Raises:
        OSError: If a declared source cannot be read.
        ValueError: If cohort, YAML, replay, reconciliation, or source gates fail.
    """
    population = contract["population"]
    failure_codes = _ACTIVE_ISOLATED_FAILURE_CODES.get()
    cohort_code = failure_codes.minimal_cohort if failure_codes is not None else None
    with _optional_isolated_failure_boundary(cohort_code):
        cohort = load_minimal_cohort_tables(
            sample_index_path,
            qc_diagnostics_path,
            expected_source_count=int(population["expected_source_count"]),
            expected_final_count=int(population["expected_final_count"]),
            expected_excluded_count=int(population["expected_excluded_count"]),
            expected_source_fallback_count=int(population["expected_source_fallback_count"]),
            expected_final_fallback_count=int(population["expected_final_fallback_count"]),
            expected_excluded_fallback_count=int(population["expected_excluded_fallback_count"]),
            expected_padded_count=int(population["expected_padded_count"]),
        )
    entry_digest_code = failure_codes.raw_digest_entry if failure_codes is not None else None
    with _optional_isolated_failure_boundary(entry_digest_code):
        raw_digest_entry = compute_raw_cohort_digest(cohort.final_rows, raw_root)
    environmental_model_code = failure_codes.environmental_model if failure_codes is not None else None
    with _optional_isolated_failure_boundary(environmental_model_code):
        env_model = load_frozen_environmental_model(preprocessing_config_path)
    replay = compute_replay_and_aggregate(
        contract,
        cohort,
        env_model=env_model,
        raw_root=raw_root,
        pipeline_a_path=pipeline_a_path,
    )
    exit_digest_code = failure_codes.raw_digest_exit if failure_codes is not None else None
    with _optional_isolated_failure_boundary(exit_digest_code):
        raw_digest_exit = compute_raw_cohort_digest(cohort.final_rows, raw_root)
        if raw_digest_entry != raw_digest_exit:
            raise ValueError("Aggregate raw-cohort source changed within an isolated computation.")
    return IsolatedReplayComputation(
        summary=replay.summary,
        stage4_reconciliation=replay.stage4_reconciliation,
        raw_digest_entry=raw_digest_entry,
        raw_digest_exit=raw_digest_exit,
        private_filenames=tuple(cohort.final_rows["filename"].astype(str).tolist()),
    )


def hash_implementation_files(project_root: Path, contract_path: Path) -> dict[str, str]:
    """Hash the three implementation files bound by Mandatory Gate 2.

    Args:
        project_root: Repository root.
        contract_path: Runtime-contract YAML path.

    Returns:
        Exact config, module, and script SHA-256 mapping.

    Raises:
        OSError: If any implementation file cannot be read.
        ValueError: If a path escapes the repository root.
    """
    root = Path(project_root).resolve(strict=True)
    paths = {
        "config_sha256": Path(contract_path).resolve(strict=True),
        "module_sha256": Path(__file__).resolve(strict=True),
        "script_sha256": (root / "scripts" / "run_efigure4_preprocessing_stage_aggregate_replay.py").resolve(
            strict=True
        ),
    }
    if any(not path.is_relative_to(root) for path in paths.values()):
        raise ValueError("Implementation hash path escaped the repository root.")
    return {name: sha256_file(path) for name, path in paths.items()}


def build_execution_authorization(
    plan_sha256: str,
    implementation_hashes: dict[str, str],
    environment_lock_sha256: str,
) -> str:
    """Build the one exact Mandatory Gate 2 authorization line.

    Args:
        plan_sha256: Approved results-free plan SHA-256.
        implementation_hashes: Current config, module, and script digests.
        environment_lock_sha256: Exact canonical environment-lock digest.

    Returns:
        Exact user authorization line required by the CLI.

    Raises:
        ValueError: If any digest or key differs from the contract.
    """
    if not re.fullmatch(r"[0-9a-f]{64}", plan_sha256):
        raise ValueError("Approved plan hash is invalid.")
    required = ("config_sha256", "module_sha256", "script_sha256")
    if tuple(implementation_hashes) != required or any(
        not re.fullmatch(r"[0-9a-f]{64}", implementation_hashes[key]) for key in required
    ):
        raise ValueError("Implementation authorization hashes are invalid.")
    if not re.fullmatch(r"[0-9a-f]{64}", environment_lock_sha256):
        raise ValueError("Environment-lock authorization hash is invalid.")
    return (
        "AUTHORIZE_EFIGURE4_AGGREGATE_REPLAY_EXECUTION "
        f"PLAN_SHA256={plan_sha256} "
        f"CONFIG_SHA256={implementation_hashes['config_sha256']} "
        f"MODULE_SHA256={implementation_hashes['module_sha256']} "
        f"SCRIPT_SHA256={implementation_hashes['script_sha256']} "
        f"ENVIRONMENT_LOCK_SHA256={environment_lock_sha256}"
    )


def _run_git(project_root: Path, arguments: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run one non-mutating Git inspection command with sanitized failures."""
    result = subprocess.run(
        ["git", *arguments],
        cwd=Path(project_root),
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise ValueError("Read-only Git authority inspection failed.")
    return result


def collect_runtime_authority(contract: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Collect the aggregate-safe read-only Git/runtime authority snapshot.

    Args:
        contract: Validated runtime contract.
        project_root: Repository root.

    Returns:
        Snapshot accepted by `validate_authority_snapshot` when exact.

    Raises:
        ValueError: If a Git inspection fails or status output is malformed.
    """
    root = Path(project_root).resolve(strict=True)
    branch = _run_git(root, ["branch", "--show-current"]).stdout.strip()
    head = _run_git(root, ["rev-parse", "HEAD"]).stdout.strip()
    lineage = _run_git(root, ["rev-list", "--parents", "-n", "1", "HEAD"]).stdout.split()
    if len(lineage) != 2 or lineage[0] != head:
        raise ValueError("Runtime HEAD does not have exactly one parent.")
    sole_parent = lineage[1]
    frozen = _run_git(root, ["rev-parse", "preprocessing-frozen^{commit}"]).stdout.strip()
    ancestry = _run_git(root, ["merge-base", "--is-ancestor", "preprocessing-frozen", "HEAD"], check=False)
    input_blobs = {
        name: _run_git(root, ["rev-parse", f"HEAD:{payload['path']}"]).stdout.strip()
        for name, payload in contract["inputs"].items()
    }
    frozen_diff = _run_git(
        root,
        [
            "diff",
            "--quiet",
            contract["runtime_authority"]["preprocessing_frozen_commit"],
            "HEAD",
            "--",
            *contract["frozen_diff_paths"],
        ],
        check=False,
    )
    status_output = _run_git(root, ["status", "--short", "--untracked-files=all"]).stdout
    worktree_entries = _parse_worktree_status_paths(status_output)
    final_path = root / contract["outputs"]["final_directory"]
    candidate_one_path = root / contract["outputs"]["candidate_one_directory"]
    candidate_two_path = root / contract["outputs"]["candidate_two_directory"]
    approved_plan_path = root / contract["approved_plan"]["path"]
    original_plan_path = root / contract["original_plan"]["path"]
    recovery_v2_plan_path = root / contract["recovery_v2_plan"]["path"]
    recovery_v3_plan_path = root / contract["recovery_v3_plan"]["path"]
    reporting_correction_path = root / contract["reporting_correction"]["path"]
    numerical_correction_path = root / contract["numerical_environment_correction"]["path"]
    channel_correction_path = root / contract["channel_metadata_correction"]["path"]
    environment_lock_path = root / contract["environment_lock"]["path"]
    return {
        "branch": branch,
        "head": head,
        "sole_parent": sole_parent,
        "preprocessing_frozen_commit": frozen,
        "preprocessing_frozen_is_ancestor": ancestry.returncode == 0,
        "input_blobs": input_blobs,
        "frozen_diff_is_empty": frozen_diff.returncode == 0,
        "worktree_entries": worktree_entries,
        "final_output_absent": not final_path.exists() and not final_path.is_symlink(),
        "candidate_one_output_absent": not candidate_one_path.exists() and not candidate_one_path.is_symlink(),
        "candidate_two_output_absent": not candidate_two_path.exists() and not candidate_two_path.is_symlink(),
        "approved_plan_sha256": sha256_file(approved_plan_path),
        "original_plan_sha256": sha256_file(original_plan_path),
        "recovery_v2_plan_sha256": sha256_file(recovery_v2_plan_path),
        "recovery_v3_plan_sha256": sha256_file(recovery_v3_plan_path),
        "reporting_correction_sha256": sha256_file(reporting_correction_path),
        "numerical_environment_correction_sha256": sha256_file(numerical_correction_path),
        "channel_metadata_correction_sha256": sha256_file(channel_correction_path),
        "environment_lock_sha256": sha256_file(environment_lock_path),
    }


def collect_package_versions() -> dict[str, str]:
    """Return the exact installed computation package-version ledger."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "pyyaml": yaml.__version__,
    }


def capture_source_identities(contract: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Capture aggregate-safe local identities for every declared frozen input.

    Args:
        contract: Validated runtime contract.
        project_root: Repository root containing every declared path.

    Returns:
        Deterministically ordered source identity mapping.

    Raises:
        OSError: If a source cannot be read.
        ValueError: If a path escapes, an anchor differs, or the anomaly changes.
    """
    root = Path(project_root).resolve(strict=True)
    identities: dict[str, Any] = {}
    for name, expected in contract["inputs"].items():
        relative_path = Path(expected["path"])
        if relative_path.is_absolute():
            raise ValueError("Declared source path must be project-relative.")
        source_path = (root / relative_path).resolve(strict=True)
        if not source_path.is_relative_to(root) or not source_path.is_file():
            raise ValueError("Declared source path escaped the repository root.")
        if name == "preprocessing_config":
            byte_size = int(source_path.stat().st_size)
            blake2b = blake2b_file(source_path)
            if (
                expected.get("sha256") is not None
                or expected.get("sha256_status") != CONFIG_ANOMALY_STATUS
                or byte_size != int(expected["byte_size"])
                or blake2b != expected["blake2b"]
            ):
                raise ValueError("Preprocessing-config anomaly identity differs.")
            identities[name] = {
                "path": relative_path.as_posix(),
                "git_blob": expected["git_blob"],
                "sha256": None,
                "sha256_status": CONFIG_ANOMALY_STATUS,
                "byte_size": byte_size,
                "blake2b": blake2b,
            }
        else:
            identities[name] = {
                "path": relative_path.as_posix(),
                "git_blob": expected["git_blob"],
                "sha256": sha256_file(source_path),
                "byte_size": int(source_path.stat().st_size),
            }
    return identities


def run_authorized_replay(*, authorization: str, project_root: Path, contract_path: Path) -> dict[str, Any]:
    """Execute the Gate-2-authorized aggregate replay.

    Args:
        authorization: Exact Mandatory Gate 2 authorization line.
        project_root: Repository root.
        contract_path: Locked runtime-contract path.

    Returns:
        Aggregate-safe final handoff values.

    Raises:
        OSError: If a declared source or aggregate candidate cannot be read/written.
        ValueError: If authorization, authority, scientific, determinism, or privacy gates fail.
    """
    with _failure_boundary(FailureCode.AUTHORIZATION_PLAN_IMPLEMENTATION_HASH_VALIDATION_FAILURE):
        root = Path(project_root).resolve(strict=True)
        contract_file = Path(contract_path).resolve(strict=True)
        contract = load_and_validate_contract(contract_file, APPROVED_PLAN_SHA256)
        implementation_hashes = hash_implementation_files(root, contract_file)
        expected_authorization = build_execution_authorization(
            APPROVED_PLAN_SHA256,
            implementation_hashes,
            contract["environment_lock"]["sha256"],
        )
        if authorization != expected_authorization:
            raise ValueError("Mandatory Gate 2 execution authorization does not match.")
        approved_plan_path = root / contract["approved_plan"]["path"]
        if sha256_file(approved_plan_path) != APPROVED_PLAN_SHA256:
            raise ValueError("Approved recovery plan hash changed after Mandatory Gate 1.")
        original_plan_path = root / contract["original_plan"]["path"]
        if sha256_file(original_plan_path) != ORIGINAL_PLAN_SHA256:
            raise ValueError("Original results-free plan hash changed after recovery Gate 1.")
        recovery_v2_plan_path = root / contract["recovery_v2_plan"]["path"]
        if sha256_file(recovery_v2_plan_path) != RECOVERY_V2_PLAN_SHA256:
            raise ValueError("Recovery-v2 plan hash changed after Recovery-v4 Gate 1.")
        recovery_v3_plan_path = root / contract["recovery_v3_plan"]["path"]
        if sha256_file(recovery_v3_plan_path) != RECOVERY_V3_PLAN_SHA256:
            raise ValueError("Recovery-v3 plan hash changed after Recovery-v4 Gate 1.")
        reporting_correction_path = root / contract["reporting_correction"]["path"]
        if (
            sha256_file(reporting_correction_path) != REPORTING_CORRECTION_SHA256
            or reporting_correction_path.stat().st_size != REPORTING_CORRECTION_BYTE_SIZE
        ):
            raise ValueError("Reporting-correction identity changed after recovery Gate 1.")
        numerical_correction_path = root / contract["numerical_environment_correction"]["path"]
        if sha256_file(numerical_correction_path) != NUMERICAL_ENVIRONMENT_CORRECTION_SHA256:
            raise ValueError("Numerical-environment correction hash changed after Recovery-v4 Gate 1.")
        channel_correction_path = root / contract["channel_metadata_correction"]["path"]
        if sha256_file(channel_correction_path) != CHANNEL_METADATA_CORRECTION_SHA256:
            raise ValueError("Channel-metadata correction hash changed after Recovery-v4 Gate 1.")
        environment_lock_path = root / contract["environment_lock"]["path"]
        if sha256_file(environment_lock_path) != ENVIRONMENT_LOCK_SHA256:
            raise ValueError("Environment-lock hash changed after Recovery-v4 Gate 1.")

    with _failure_boundary(FailureCode.GIT_AUTHORITY_OUTPUT_STATE_VALIDATION_FAILURE):
        authority = collect_runtime_authority(contract, root)
        validate_authority_snapshot(contract, authority)
        final_path = root / contract["outputs"]["final_directory"]
        candidate_one = root / contract["outputs"]["candidate_one_directory"]
        candidate_two = root / contract["outputs"]["candidate_two_directory"]
        for path in (final_path, candidate_one, candidate_two):
            require_absent_output_namespace(path)

    with _failure_boundary(FailureCode.ENVIRONMENT_IDENTITY_VALIDATION_FAILURE):
        environment_entry = validate_runtime_environment(contract, root)
    package_versions = collect_package_versions()
    with _failure_boundary(FailureCode.FROZEN_SOURCE_IDENTITY_CAPTURE_FAILURE):
        source_entry = capture_source_identities(contract, root)
    population = contract["population"]
    raw_root = root / "data" / "raw" / "breath_data"
    isolated_paths = {
        "sample_index_path": root / contract["inputs"]["sample_index"]["path"],
        "qc_diagnostics_path": root / contract["inputs"]["qc_diagnostics"]["path"],
        "preprocessing_config_path": root / contract["inputs"]["preprocessing_config"]["path"],
        "raw_root": raw_root,
        "pipeline_a_path": root / contract["inputs"]["pipeline_a"]["path"],
    }
    with _isolated_failure_scope(FIRST_ISOLATED_FAILURE_CODES):
        first = compute_isolated_replay_and_aggregate(
            contract,
            **isolated_paths,
        )
    with _isolated_failure_scope(SECOND_ISOLATED_FAILURE_CODES):
        second = compute_isolated_replay_and_aggregate(
            contract,
            **isolated_paths,
        )
    with _failure_boundary(FailureCode.CROSS_COMPUTATION_DETERMINISM_RAW_DIGEST_COMPARISON_FAILURE):
        first_csv = serialize_aggregate_csv(
            first.summary,
            float_format=contract["serialization"]["csv_float_format"],
        )
        second_csv = serialize_aggregate_csv(
            second.summary,
            float_format=contract["serialization"]["csv_float_format"],
        )
        if first_csv != second_csv or serialize_json(first.stage4_reconciliation) != serialize_json(
            second.stage4_reconciliation
        ):
            raise ValueError("Two isolated scientific computations are not byte-identical.")
        if not (
            first.raw_digest_entry
            == first.raw_digest_exit
            == second.raw_digest_entry
            == second.raw_digest_exit
        ):
            raise ValueError("Aggregate raw-cohort source entry/exit digest mismatch.")
        if first.private_filenames != second.private_filenames:
            raise ValueError("Two isolated computation cohorts differ.")

    with _failure_boundary(FailureCode.ENVIRONMENT_IDENTITY_VALIDATION_FAILURE):
        environment_exit = validate_runtime_environment(contract, root)
        if environment_exit != environment_entry:
            raise ValueError("Environment identity changed during execution.")

    with _failure_boundary(FailureCode.FINAL_SOURCE_IDENTITY_AUTHORITY_REVALIDATION_FAILURE):
        source_exit = capture_source_identities(contract, root)
        require_source_identity_match(source_entry, source_exit)
        if sha256_file(approved_plan_path) != APPROVED_PLAN_SHA256:
            raise ValueError("Approved recovery plan hash changed before promotion.")
        if sha256_file(original_plan_path) != ORIGINAL_PLAN_SHA256:
            raise ValueError("Original results-free plan hash changed before promotion.")
        if sha256_file(recovery_v2_plan_path) != RECOVERY_V2_PLAN_SHA256:
            raise ValueError("Recovery-v2 plan hash changed before promotion.")
        if sha256_file(recovery_v3_plan_path) != RECOVERY_V3_PLAN_SHA256:
            raise ValueError("Recovery-v3 plan hash changed before promotion.")
        if (
            sha256_file(reporting_correction_path) != REPORTING_CORRECTION_SHA256
            or reporting_correction_path.stat().st_size != REPORTING_CORRECTION_BYTE_SIZE
        ):
            raise ValueError("Reporting-correction identity changed before promotion.")
        if sha256_file(numerical_correction_path) != NUMERICAL_ENVIRONMENT_CORRECTION_SHA256:
            raise ValueError("Numerical-environment correction hash changed before promotion.")
        if sha256_file(channel_correction_path) != CHANNEL_METADATA_CORRECTION_SHA256:
            raise ValueError("Channel-metadata correction hash changed before promotion.")
        if sha256_file(environment_lock_path) != ENVIRONMENT_LOCK_SHA256:
            raise ValueError("Environment-lock hash changed before promotion.")
        if hash_implementation_files(root, contract_file) != implementation_hashes:
            raise ValueError("Authorized implementation hash changed before promotion.")
        exit_authority = collect_runtime_authority(contract, root)
        validate_authority_snapshot(contract, exit_authority)

    with _failure_boundary(FailureCode.AUDIT_PAYLOAD_CONSTRUCTION_FAILURE):
        audit = build_aggregate_audit(
            contract,
            authority=exit_authority,
            implementation_hashes=implementation_hashes,
            package_versions=package_versions,
            environment_provenance=environment_entry,
            source_identities=source_entry,
            environmental_summary_reconciliation_passed=True,
            stage4_reconciliation=first.stage4_reconciliation,
            raw_cohort_digest=first.raw_digest_entry,
            source_entry_exit_identity_passed=True,
            deterministic_rerun_passed=True,
            privacy_validation_passed=True,
        )
        first_payloads = build_candidate_payloads(
            first.summary,
            audit=audit,
            contract=contract,
            source_identities=source_entry,
            implementation_hashes=implementation_hashes,
        )
        second_payloads = build_candidate_payloads(
            second.summary,
            audit=audit,
            contract=contract,
            source_identities=source_entry,
            implementation_hashes=implementation_hashes,
        )
    with _failure_boundary(FailureCode.CANDIDATE_COMPARISON_FAILURE):
        if first_payloads != second_payloads:
            raise ValueError("Two isolated serialized aggregate candidates are not byte-identical.")

    private_filenames = first.private_filenames
    created_candidates: list[Path] = []
    try:
        with _failure_boundary(FailureCode.FIRST_CANDIDATE_WRITING_VALIDATION_FAILURE):
            write_candidate_directory(
                candidate_one,
                first_payloads,
                contract=contract,
                expected_participants=int(population["expected_final_count"]),
                expected_frames=int(contract["response_axis"]["clip_length"]),
                expected_audit=audit,
                private_values=private_filenames,
            )
        created_candidates.append(candidate_one)
        with _failure_boundary(FailureCode.SECOND_CANDIDATE_WRITING_VALIDATION_FAILURE):
            write_candidate_directory(
                candidate_two,
                second_payloads,
                contract=contract,
                expected_participants=int(population["expected_final_count"]),
                expected_frames=int(contract["response_axis"]["clip_length"]),
                expected_audit=audit,
                private_values=private_filenames,
            )
        created_candidates.append(candidate_two)
        with _failure_boundary(FailureCode.SERIALIZED_PRIVACY_COMPLETE_READ_BACK_VALIDATION_FAILURE):
            first_read_back = read_and_validate_candidate_directory(
                candidate_one,
                contract=contract,
                expected_participants=int(population["expected_final_count"]),
                expected_frames=int(contract["response_axis"]["clip_length"]),
                expected_audit=audit,
                private_values=private_filenames,
            )
            second_read_back = read_and_validate_candidate_directory(
                candidate_two,
                contract=contract,
                expected_participants=int(population["expected_final_count"]),
                expected_frames=int(contract["response_axis"]["clip_length"]),
                expected_audit=audit,
                private_values=private_filenames,
            )
        with _failure_boundary(FailureCode.CANDIDATE_COMPARISON_FAILURE):
            if first_read_back != second_read_back:
                raise ValueError("Validated candidate read-back bytes differ.")
        manifest_sha256 = sha256_bytes(first_payloads["manifest.json"])
        result = {
            "aggregate_row_count": int(len(first.summary)),
            "ending_head": exit_authority["head"],
            "final_output_directory": contract["outputs"]["final_directory"],
            "implementation_hashes": implementation_hashes,
            "manifest_sha256": manifest_sha256,
            "population_counts": audit["population_counts"],
            "stage4_reconciliation": first.stage4_reconciliation,
            "starting_head": authority["head"],
        }
        with _failure_boundary(FailureCode.CANDIDATE_COMPARISON_FAILURE):
            shutil.rmtree(candidate_two)
            created_candidates.remove(candidate_two)
        with _failure_boundary(FailureCode.ATOMIC_PROMOTION_FINAL_DIRECTORY_VALIDATION_FAILURE):
            promote_candidate_directory(
                candidate_one,
                final_path,
                contract=contract,
                expected_participants=int(population["expected_final_count"]),
                expected_frames=int(contract["response_axis"]["clip_length"]),
                expected_audit=audit,
                private_values=private_filenames,
            )
        created_candidates.remove(candidate_one)
        return result
    except Exception:
        for candidate in created_candidates:
            if candidate.exists():
                shutil.rmtree(candidate)
        raise


__all__ = [
    "CohortTables",
    "FailureCode",
    "FIRST_ISOLATED_FAILURE_CODES",
    "FrozenPipelineA",
    "IsolatedReplayComputation",
    "IsolatedFailureCodes",
    "ParticipantReplay",
    "RawSignal",
    "RawSignalLoadError",
    "ReplayComputation",
    "ReplayExecutionError",
    "SECOND_ISOLATED_FAILURE_CODES",
    "aggregate_stage_tensors",
    "build_aggregate_audit",
    "build_candidate_payloads",
    "build_execution_authorization",
    "build_run_summary",
    "capture_source_identities",
    "collect_package_versions",
    "collect_runtime_authority",
    "compute_raw_cohort_digest",
    "compute_isolated_replay_and_aggregate",
    "compute_replay_and_aggregate",
    "hash_implementation_files",
    "load_and_validate_contract",
    "load_and_validate_environment_lock",
    "load_minimal_cohort_tables",
    "load_frozen_environmental_model",
    "load_frozen_pipeline_a",
    "load_raw_signal",
    "replay_participant",
    "reconcile_stage4",
    "read_and_validate_candidate_directory",
    "require_absent_output_namespace",
    "require_source_identity_match",
    "run_authorized_replay",
    "serialize_aggregate_csv",
    "serialize_json",
    "sha256_file",
    "promote_candidate_directory",
    "validate_aggregate_summary",
    "validate_authority_snapshot",
    "validate_conda_tool_identity",
    "validate_runtime_environment",
    "validate_serialized_privacy",
    "validate_stage_tensors",
    "write_candidate_directory",
]
