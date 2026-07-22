"""Fixed date-grouped MiniRocket auxiliary validation for the final post-hoc package."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import tempfile
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedGroupKFold

from src.analysis.posthoc_final_common import (
    ACCESS_STATE_KEYS,
    AccessStateTracker,
    build_registry_frame,
    load_csv_with_ordered_usecols,
    parse_date_series_ascii_slash,
    registry_readback_matches,
    sha256_file,
    validate_completion_log,
    validate_exact_binary_labels_01,
    validate_freeze_commit,
    validate_preflight_privacy,
    validate_preflight_schema,
    validate_registry_records,
)
from src.analysis.posthoc_final_locked_scores import (
    SCORE_BLIND_PREFLIGHT_RELATIVE_PATH,
    SESSION_1_TO_3_IMPLEMENTATION_PATHS,
    SESSION_4_COMPLETION_LOG_PATH,
    SESSION_4_SCIENTIFIC_OUTPUT_PATHS,
    _amendment_cumulative_prior,
    _git,
    _validate_annotated_tag_chain,
    load_effective_posthoc_config,
    validate_amendment_freeze_commit,
    validate_current_runtime_authority,
    validate_execute_git_prerequisites,
)
from src.config import PROJECT_ROOT
from src.training.train_minirocket_ridge import MiniRocketRidgeTrainer


ANALYSIS_ID = "analysis_3_date_grouped_minirocket"
VALIDATION_COLUMNS = [
    "analysis_id",
    "n",
    "n_positive",
    "n_negative",
    "n_date_groups",
    "n_folds",
    "splitter",
    "shuffle",
    "split_seed",
    "split_seed_status",
    "model_seed",
    "date_grouped_auc_roc",
    "date_grouped_auc_ci_low",
    "date_grouped_auc_ci_high",
    "frozen_random_oof_auc_roc",
    "auc_difference_date_grouped_minus_random",
    "delta_ci_low",
    "delta_ci_high",
    "ci_method",
    "n_bootstrap_valid",
    "n_bootstrap_attempts",
    "bootstrap_seed",
    "tie_credit",
    "oof_vector_sha256",
    "date_grouped_auc_status",
    "date_grouped_auc_nonestimable_reason",
    "date_grouped_auc_ci_status",
    "date_grouped_auc_ci_nonestimable_reason",
    "auc_difference_status",
    "auc_difference_nonestimable_reason",
    "auc_difference_ci_status",
    "auc_difference_ci_nonestimable_reason",
]
FOLD_COLUMNS = [
    "fold_id",
    "n_fit",
    "n_validation",
    "n_fit_positive",
    "n_fit_negative",
    "n_validation_positive",
    "n_validation_negative",
    "n_fit_date_groups",
    "n_validation_date_groups",
    "date_overlap_count",
    "calibrator_fit_feasible",
    "validation_auc_roc",
    "estimability_status",
    "nonestimable_reason",
]
FEASIBILITY_REASONS = (
    "insufficient_unique_date_groups",
    "fixed_partition_class_degenerate",
    "outer_fit_calibration_class_count_below_5",
)
BOOTSTRAP_REASON = "bootstrap_valid_replicates_below_2000"
EXPECTED_PIPELINE_SHAPE = (972, 960, 11)
EXPECTED_MODELING_N = 972
EXPECTED_TRAIN_N = 777
EXPECTED_TEST_N = 195
MODEL_SEED = 42
BOOTSTRAP_SEED = 42
BOOTSTRAP_VALID_TARGET = 2000
BOOTSTRAP_ATTEMPT_CAP = 20000
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SESSION_4_ACTIVITY = OrderedDict(
    (
        key,
        key
        in {
            "prediction_container_bytes_hashed",
            "prediction_schema_or_header_inspected",
            "participant_level_label_values_read",
            "participant_level_score_values_read",
            "participant_level_date_values_read",
            "participant_level_qc_values_read",
            "new_score_derived_results_computed",
        },
    )
    for key in ACCESS_STATE_KEYS
)


@dataclass(frozen=True)
class Analysis3Inputs:
    """Aligned in-memory inputs for one isolated Analysis 3 candidate."""

    x: np.ndarray
    y: np.ndarray
    date_groups: np.ndarray
    sample_ids: np.ndarray
    frozen_random_oof_scores: np.ndarray
    frozen_random_oof_auc: float


@dataclass(frozen=True)
class FoldDefinition:
    """One fixed fitting/validation partition."""

    fold_id: int
    fit_idx: np.ndarray
    validation_idx: np.ndarray


@dataclass(frozen=True)
class PartitionResult:
    """Validated fixed partition and aggregate feasibility diagnostics."""

    splitter: Any
    folds: tuple[FoldDefinition, ...]
    fold_rows: tuple[dict[str, Any], ...]
    feasible: bool
    reason: str | None


@dataclass(frozen=True)
class BootstrapResult:
    """Paired date-cluster bootstrap interval results."""

    date_auc_ci: tuple[float, float] | None
    delta_ci: tuple[float, float] | None
    n_valid: int
    n_attempts: int
    seed: int
    nonestimable_reason: str | None


@dataclass(frozen=True)
class CandidateArtifacts:
    """Aggregate-only candidate artifacts."""

    validation: pd.DataFrame
    fold_audit: pd.DataFrame
    oof_digest: str | None


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping."""
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Configuration root must be a mapping.")
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    """Load a JSON mapping."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle, object_pairs_hook=OrderedDict)
    if not isinstance(payload, dict):
        raise ValueError("Governance JSON root must be an object.")
    return payload


def _resolve(relative_path: str) -> Path:
    """Resolve one project-relative path without accepting traversal."""
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Configured path must be project-relative.")
    resolved = (PROJECT_ROOT / path).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        raise ValueError("Configured path escapes the project root.") from None
    return resolved


def _load_current_config(config_path: Path) -> dict[str, Any]:
    """Load the single consolidated current runtime configuration."""
    return load_effective_posthoc_config(config_path)


def _validated_historical_amendment(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Require identical historical Amendment provenance in both current records."""
    try:
        config_record = config["historical_consolidation"]["amendment_01"]
        manifest_record = manifest["historical_evidence"]["amendment_01"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Historical Amendment 01 record is malformed.") from exc
    required_fields = (
        "commit_id",
        "sole_parent",
        "changed_paths_exact",
        "artifacts",
        "approval_evidence",
    )
    if not isinstance(config_record, Mapping) or not isinstance(manifest_record, Mapping):
        raise ValueError("Historical Amendment 01 record is malformed.")
    if any(config_record.get(key) != manifest_record.get(key) for key in required_fields):
        raise ValueError("Historical Amendment 01 config and manifest records disagree.")
    if config_record.get("access_state") != manifest_record.get("access_state"):
        raise ValueError("Historical Amendment 01 access-state records disagree.")
    corrections = config_record.get("session_1_corrections")
    if corrections != manifest_record.get("session_1_corrections"):
        raise ValueError("Historical Amendment 01 correction records disagree.")
    expected_corrections = [
        {"id": f"C{index:02d}", "status": "implemented_in_session_1"}
        for index in range(1, 17)
    ]
    if corrections != expected_corrections:
        raise ValueError("Historical Amendment 01 correction record is incomplete.")
    return config_record


def _date_grouped_target_paths(config: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    """Return the exact two scientific targets and completion receipt."""
    registry = config["row_order_and_schema_validation_registry"]
    validation = _resolve(registry["date_grouped_validation"]["path"])
    folds = _resolve(registry["date_grouped_fold_audit"]["path"])
    return validation, folds, _resolve(_completion_log_relative_path(config, "date_grouped"))


def _completion_log_relative_path(config: Mapping[str, Any], session: str) -> str:
    """Return one configured completion-log path across the frozen config views."""
    logs = config["outputs"]["completion_logs"]
    return logs["paths"][session] if "paths" in logs else logs[session]


def _session4_scientific_paths(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the configured ordered Session 4 scientific output inventory."""
    scientific = config["outputs"]["scientific_and_closure"]
    return (
        scientific["locked_score_metrics"]["path"],
        scientific["locked_score_composition"]["path"],
        scientific["qc5_correctness"]["path"],
        scientific["within_date_concordance"]["path"],
    )


def classify_target_state(validation_path: Path, fold_path: Path, log_path: Path) -> str:
    """Classify the absent, partial, or stale Session 5 target state."""
    output_exists = validation_path.exists() or fold_path.exists()
    log_exists = log_path.exists()
    if not output_exists and not log_exists:
        return "clean"
    if output_exists and not log_exists:
        return "partial_interrupted_promotion"
    return "stale_or_complete"


def _validate_analysis3_git_state(config: Mapping[str, Any], git_runner: Callable[..., str]) -> None:
    """Validate tags, committed prerequisites, and Session 5 target absence."""
    _validate_annotated_tag_chain(git_runner)
    implementation_paths = tuple(config["outputs"]["future_implementation_paths"])
    if implementation_paths != SESSION_1_TO_3_IMPLEMENTATION_PATHS:
        raise ValueError("Frozen Session 1-3 implementation path inventory is invalid.")
    preflight_path = config["outputs"]["scientific_and_closure"]["score_blind_preflight"]["path"]
    if preflight_path != SCORE_BLIND_PREFLIGHT_RELATIVE_PATH:
        raise ValueError("Configured score-blind preflight path is invalid.")
    session4_paths = _session4_scientific_paths(config)
    if session4_paths != SESSION_4_SCIENTIFIC_OUTPUT_PATHS:
        raise ValueError("Configured Session 4 scientific outputs are invalid.")
    session4_log = _completion_log_relative_path(config, "locked_scores")
    if session4_log != SESSION_4_COMPLETION_LOG_PATH:
        raise ValueError("Configured Session 4 completion log is invalid.")
    required = (
        *implementation_paths,
        preflight_path,
        *session4_paths,
        session4_log,
    )
    validate_execute_git_prerequisites(required, git_runner=git_runner)
    state = classify_target_state(*_date_grouped_target_paths(config))
    if state != "clean":
        raise ValueError(f"Session 5 target state is {state}; provenance review is required.")


def _validate_committed_preflight(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    freeze_commit: str,
    git_runner: Callable[..., str],
    cumulative_prior: Mapping[str, bool],
) -> OrderedDict[str, bool]:
    """Validate the committed aggregate score-blind preflight without reopening rows."""
    try:
        relative_path = config["outputs"]["scientific_and_closure"]["score_blind_preflight"]["path"]
        expected_sha256 = manifest["historical_evidence"]["completed_session_1"]["artifact_sha256"][
            relative_path
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError("Score-blind preflight authority record is malformed.") from exc
    if relative_path != SCORE_BLIND_PREFLIGHT_RELATIVE_PATH:
        raise ValueError("Score-blind preflight path does not match the frozen authority.")
    path = _resolve(relative_path)
    if sha256_file(path) != expected_sha256:
        raise ValueError("Score-blind preflight hash does not match the frozen authority.")
    try:
        git_runner("ls-files", "--error-unmatch", relative_path)
        if git_runner("rev-parse", f"HEAD:{relative_path}") != git_runner(
            "hash-object", relative_path
        ):
            raise ValueError("Score-blind preflight is not HEAD-identical.")
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError("Score-blind preflight must be committed and HEAD-identical.") from exc
    payload = _read_json(path)
    frozen_paths = list(config["frozen_inputs"]["whole_container_sha256"])
    validate_preflight_schema(payload, frozen_hash_paths=frozen_paths, expected_cumulative_prior=cumulative_prior)
    validate_preflight_privacy(payload)
    session_0 = config["outputs"]["session_0"]
    try:
        historical_artifacts = manifest["historical_evidence"]["original_session_0"]["artifacts"]
        historical_hashes = {item["path"]: item["sha256"] for item in historical_artifacts}
        expected_authority_hashes = {
            "config_sha256": historical_hashes[session_0["config"]],
            "methods_sha256": historical_hashes[session_0["methods"]],
            "freeze_manifest_sha256": historical_hashes[session_0["freeze_manifest"]],
        }
    except (KeyError, TypeError) as exc:
        raise ValueError("Historical Session 0 preflight authority record is malformed.") from exc
    if any(payload["hashes"][key] != value for key, value in expected_authority_hashes.items()):
        raise ValueError("Score-blind preflight hashes do not match historical Session 0 artifacts.")
    if dict(payload["hashes"]["frozen_input_whole_container_sha256"]) != dict(
        config["frozen_inputs"]["whole_container_sha256"]
    ):
        raise ValueError("Score-blind preflight frozen-input hashes do not match current authority.")
    if payload["status"] != "pass" or payload["reason_code"] is not None:
        raise ValueError("Score-blind preflight did not pass.")
    if payload["freeze_commit_validation"]["freeze_commit_id"] != freeze_commit:
        raise ValueError("Score-blind preflight records the wrong freeze commit.")
    if payload["one_to_one_alignment_status"] != "pass" or payload["train_test_disjointness_status"] != "pass":
        raise ValueError("Score-blind preflight alignment did not pass.")
    if payload["known_anchor_reconciliation"]["status"] != "pass":
        raise ValueError("Score-blind preflight anchor reconciliation did not pass.")
    activity = payload["access_state"]["cumulative_through_session"]
    for key in (
        "participant_level_score_values_read",
        "new_score_derived_results_computed",
        "auxiliary_model_fit_executed",
        "privacy_reference_loader_executed",
    ):
        if activity[key]:
            raise ValueError("Score-blind preflight access boundary is invalid.")
    return OrderedDict((key, bool(activity[key])) for key in ACCESS_STATE_KEYS)


def _validate_session4_completion(
    config: Mapping[str, Any],
    freeze_commit: str,
    cumulative_prior: Mapping[str, bool],
) -> OrderedDict[str, bool]:
    """Validate the committed Session 4 receipt and opaque scientific-file hashes."""
    scientific_paths = _session4_scientific_paths(config)
    if scientific_paths != SESSION_4_SCIENTIFIC_OUTPUT_PATHS:
        raise ValueError("Configured Session 4 scientific outputs are invalid.")
    log_relative = _completion_log_relative_path(config, "locked_scores")
    if log_relative != SESSION_4_COMPLETION_LOG_PATH:
        raise ValueError("Configured Session 4 completion log is invalid.")
    grammar = config["outputs"]["completion_logs"]["machine_validated_log_grammar"]
    binding = dict(grammar["session_bindings_exact"]["locked_scores"])
    binding.setdefault("session_name", "locked_scores")
    expected_cumulative = OrderedDict(
        (key, bool(cumulative_prior[key]) or SESSION_4_ACTIVITY[key]) for key in ACCESS_STATE_KEYS
    )
    log_text = _resolve(log_relative).read_text(encoding="utf-8")
    records = validate_completion_log(
        log_text,
        grammar=grammar,
        session_binding=binding,
        expected_freeze_commit=freeze_commit,
        expected_activity=SESSION_4_ACTIVITY,
        expected_cumulative=expected_cumulative,
        cumulative_prior=cumulative_prior,
        allowed_reason_codes=tuple(config["status_and_reason_registries"]["scientific_result_reason_registry"]),
    )
    hash_records = records[3 : 3 + len(scientific_paths)]
    for record, relative_path in zip(hash_records, scientific_paths, strict=True):
        if record[1] != relative_path or record[2] != sha256_file(_resolve(relative_path)):
            raise ValueError("Session 4 completion-log output hash does not match the committed scientific file.")
    return expected_cumulative


def validate_runtime_contract(config: Mapping[str, Any]) -> None:
    """Require the frozen Python/package/source/threading contract."""
    runtime = config["runtime_controls"]
    analysis_contract = config["analysis_3_date_grouped_minirocket"]
    split_contract = analysis_contract["splitter"]
    bootstrap_contract = analysis_contract["bootstrap"]
    expected_versions = runtime["required_package_versions"]
    environment_name = str(runtime["python_environment"])
    if (
        os.environ.get("CONDA_DEFAULT_ENV") != environment_name
        or Path(sys.executable).resolve().parent.parent.name != environment_name
    ):
        raise RuntimeError("Python environment does not match the frozen runtime contract.")
    if platform.python_version() != str(runtime["python_version"]):
        raise RuntimeError("Python version does not match the frozen runtime contract.")
    for package, expected in expected_versions.items():
        if importlib.metadata.version(package) != str(expected):
            raise RuntimeError("A package version does not match the frozen runtime contract.")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(key) != str(runtime[key]):
            raise RuntimeError("A frozen runtime control does not match the authority.")
    if int(runtime["job_count_controls"]["analysis_3_sklearn_n_jobs"]) != 1:
        raise RuntimeError("The scikit-learn job count must equal one.")
    if (
        int(analysis_contract["outer_fit_fold_calls_when_feasible"]) != 5
        or int(bootstrap_contract["bootstrap_seed"]) != BOOTSTRAP_SEED
        or int(bootstrap_contract["n_valid_target"]) != BOOTSTRAP_VALID_TARGET
        or int(bootstrap_contract["attempt_cap"]) != BOOTSTRAP_ATTEMPT_CAP
        or bootstrap_contract["quantile_method"] != "linear"
        or bootstrap_contract["reject"] != "class_collapsed_draws_only"
    ):
        raise RuntimeError("Configured Analysis 3 fit or bootstrap contract is invalid.")
    import sklearn.model_selection._split as split_module

    split_path = Path(split_module.__file__).resolve()
    if sha256_file(split_path) != split_contract["installed_split_py_sha256"]:
        raise RuntimeError("Installed splitter source hash does not match the frozen contract.")
    if (
        split_contract["shuffle"] is not False
        or split_contract["random_state"] is not None
        or split_contract["split_seed"] is not None
        or split_contract["split_seed_status"] != "not_applicable"
    ):
        raise RuntimeError("Configured fixed splitter behavior does not match the frozen contract.")
    splitter = StratifiedGroupKFold(n_splits=int(split_contract["n_splits"]), shuffle=False)
    if splitter.shuffle is not False or splitter.random_state is not None:
        raise RuntimeError("Installed fixed splitter behavior does not match the frozen contract.")


def _validate_frozen_inputs(config: Mapping[str, Any]) -> OrderedDict[str, str]:
    """Hash every frozen input and the two central training sources."""
    observed: OrderedDict[str, str] = OrderedDict()
    for relative_path, expected in config["frozen_inputs"]["whole_container_sha256"].items():
        digest = sha256_file(_resolve(relative_path))
        if digest != expected:
            raise ValueError("A frozen input hash does not match the authority.")
        observed[relative_path] = digest
    central = config["source_integrity"]["analysis_3_central_training_sources"]
    for relative_path, expected in central.items():
        if sha256_file(_resolve(relative_path)) != expected:
            raise ValueError("A central training-source hash does not match the authority.")
    return observed


def validate_analysis3_authority(
    config_path: Path,
    freeze_commit: str,
    amendment_freeze_commit: str,
    *,
    git_runner: Callable[..., str] = _git,
) -> tuple[dict[str, Any], OrderedDict[str, str], OrderedDict[str, bool]]:
    """Validate current authority, both historical freezes, and every pre-access gate."""
    if not re.fullmatch(r"[0-9a-f]{40}", freeze_commit):
        raise ValueError("Freeze commit must be exactly 40 lowercase hexadecimal characters.")
    if not re.fullmatch(r"[0-9a-f]{40}", amendment_freeze_commit):
        raise ValueError("Amendment freeze commit must be exactly 40 lowercase hexadecimal characters.")
    resolved_config = config_path if config_path.is_absolute() else PROJECT_ROOT / config_path
    if resolved_config.resolve() != (PROJECT_ROOT / "configs/post_hoc_final_supplementary.yaml").resolve():
        raise ValueError("The base configuration path is immutable.")
    base = _load_current_config(resolved_config)
    manifest = _read_json(_resolve(base["outputs"]["session_0"]["freeze_manifest"]))
    validate_current_runtime_authority(
        resolved_config,
        base,
        manifest,
        require_head_identical=True,
        git_runner=git_runner,
    )
    _validate_analysis3_git_state(base, git_runner)
    validate_freeze_commit(
        freeze_commit,
        base,
        manifest,
        git_runner=git_runner,
    )
    amendment = _validated_historical_amendment(base, manifest)
    validate_amendment_freeze_commit(
        amendment_freeze_commit,
        historical_amendment=amendment,
        git_runner=git_runner,
    )
    amendment_prior = _amendment_cumulative_prior(base)
    preflight_cumulative = _validate_committed_preflight(
        base,
        manifest,
        freeze_commit,
        git_runner,
        amendment_prior,
    )
    session4_cumulative = _validate_session4_completion(base, freeze_commit, preflight_cumulative)
    entry_hashes = _validate_frozen_inputs(base)
    validate_runtime_contract(base)
    return base, entry_hashes, session4_cumulative


def _read_csv_exact(path: Path, usecols: Sequence[str], expected: Sequence[str]) -> pd.DataFrame:
    """Read exactly one ordered CSV projection."""
    return load_csv_with_ordered_usecols(path, usecols, expected)


def _require_integer_vector(values: np.ndarray, *, expected_length: int, role: str) -> np.ndarray:
    """Validate one exact integer vector with sanitized errors."""
    array = np.asarray(values)
    if array.ndim != 1 or array.size != expected_length or array.dtype.kind not in "iu":
        raise ValueError(f"{role} vector does not satisfy its integer shape contract.")
    return array.astype(np.int64, copy=False)


def validate_locked_indices(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    modeling_n: int = EXPECTED_MODELING_N,
    train_n: int = EXPECTED_TRAIN_N,
    test_n: int = EXPECTED_TEST_N,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate exact locked train/test membership and return integer vectors."""
    train = _require_integer_vector(train_idx, expected_length=train_n, role="Locked train index")
    test = _require_integer_vector(test_idx, expected_length=test_n, role="Locked test index")
    for index, role in ((train, "train"), (test, "test")):
        if np.unique(index).size != index.size or np.any((index < 0) | (index >= modeling_n)):
            raise ValueError(f"Locked {role} indices are invalid.")
    if np.intersect1d(train, test).size:
        raise ValueError("Locked train and test indices overlap.")
    if not np.array_equal(np.sort(np.concatenate([train, test])), np.arange(modeling_n)):
        raise ValueError("Locked train and test indices are not exhaustive.")
    return train, test


def validate_frozen_comparator(
    y_true: np.ndarray,
    scores: np.ndarray,
    summary: pd.DataFrame,
) -> float:
    """Recompute and cross-check the frozen random-fold OOF AUROC."""
    y = validate_exact_binary_labels_01(y_true)
    p = np.asarray(scores, dtype=np.float64)
    if p.ndim != 1 or p.size != y.size:
        raise ValueError("Frozen comparator score alignment is invalid.")
    if not np.isfinite(p).all():
        raise ValueError("Frozen comparator scores must be finite.")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("Frozen comparator scores must be probabilities.")
    expected_columns = ["scope", "outer_fold", "metric", "value"]
    if list(summary.columns) != expected_columns:
        raise ValueError("Frozen comparator aggregate schema is invalid.")
    mask = (
        (summary["scope"] == "aggregate")
        & (summary["outer_fold"].astype("string") == "aggregate")
        & (summary["metric"] == "auc_roc")
    )
    selected = summary.loc[mask, "value"]
    if len(selected) != 1:
        raise ValueError("Frozen comparator requires exactly one aggregate source row.")
    source_value = float(selected.iloc[0])
    if not np.isfinite(source_value) or not 0.0 <= source_value <= 1.0:
        raise ValueError("Frozen comparator aggregate source value is invalid.")
    observed = rank_auc(y, p)
    if abs(observed - source_value) > 1e-12:
        raise ValueError("Frozen comparator recomputation failed its 1e-12 agreement check.")
    return observed


def load_analysis3_inputs(
    config: Mapping[str, Any],
    *,
    sample_index_path: Path | None = None,
    pipeline_path: Path | None = None,
    split_path: Path | None = None,
    oof_path: Path | None = None,
    summary_path: Path | None = None,
    tracker: AccessStateTracker | None = None,
) -> Analysis3Inputs:
    """Load and validate only the future Session 5 Analysis 3 inputs."""
    sample_path = sample_index_path or PROJECT_ROOT / "data/processed/sample_index.csv"
    tensor_path = pipeline_path or PROJECT_ROOT / "data/processed/sequences/pipeline_a_raw.npz"
    locked_split_path = split_path or PROJECT_ROOT / "data/processed/splits/train_test_split.npz"
    comparator_path = oof_path or PROJECT_ROOT / "outputs/predictions/binary_minirocket_ridge_nested_cv_oof.csv"
    aggregate_path = summary_path or PROJECT_ROOT / "outputs/metrics/binary_minirocket_ridge_nested_cv_summary.csv"
    approved_columns = config["loaders"]["per_purpose_ordered_usecols"]
    sample_columns = ["sample_id", "label", "collection_time", "in_final_output"]
    sample_frame = _read_csv_exact(
        sample_path,
        sample_columns,
        approved_columns["analysis_3_dates_sample_index"],
    )
    if tracker is not None:
        tracker.mark("participant_level_label_values_read")
        tracker.mark("participant_level_date_values_read")
    if sample_frame["in_final_output"].dtype != bool:
        raise ValueError("Final-modeling membership must be an exact boolean field.")
    modeling = sample_frame.loc[sample_frame["in_final_output"]].reset_index(drop=True)
    if len(modeling) != EXPECTED_MODELING_N:
        raise ValueError("Final-modeling membership count is invalid.")
    sample_ids = _require_integer_vector(
        modeling["sample_id"].to_numpy(),
        expected_length=EXPECTED_MODELING_N,
        role="Modeling sample",
    )
    if np.unique(sample_ids).size != sample_ids.size or np.any(np.diff(sample_ids) <= 0):
        raise ValueError("Modeling sample order or uniqueness is invalid.")
    labels = _require_integer_vector(
        modeling["label"].to_numpy(),
        expected_length=EXPECTED_MODELING_N,
        role="Modeling label",
    )
    validate_exact_binary_labels_01(labels)
    dates = parse_date_series_ascii_slash(modeling["collection_time"])

    with np.load(locked_split_path, allow_pickle=False) as archive:
        try:
            train_idx = np.asarray(archive["train_idx"])
            test_idx = np.asarray(archive["test_idx"])
        except KeyError:
            raise ValueError("Locked split is missing a required member.") from None
    train_idx, test_idx = validate_locked_indices(train_idx, test_idx)

    with np.load(tensor_path, allow_pickle=False) as archive:
        try:
            x = np.asarray(archive["x"])
            pipeline_ids = np.asarray(archive["sample_id"])
            pipeline_labels = np.asarray(archive["label"])
        except KeyError:
            raise ValueError("Pipeline A container is missing a required member.") from None
    synthetic_shape = config.get("analysis_3_date_grouped_minirocket", {}).get("expected_pipeline_shape")
    expected_shape = tuple(synthetic_shape) if synthetic_shape is not None else EXPECTED_PIPELINE_SHAPE
    if x.shape != expected_shape or x.dtype != np.dtype("<f8") or not np.isfinite(x).all():
        raise ValueError("Pipeline A tensor shape or finite-value contract is invalid.")
    pipeline_ids = _require_integer_vector(
        pipeline_ids,
        expected_length=EXPECTED_MODELING_N,
        role="Pipeline A sample",
    )
    pipeline_labels = _require_integer_vector(
        pipeline_labels,
        expected_length=EXPECTED_MODELING_N,
        role="Pipeline A label",
    )
    if not np.array_equal(pipeline_ids, sample_ids) or not np.array_equal(pipeline_labels, labels):
        raise ValueError("Pipeline A alignment or label agreement failed.")

    expected_header = config["frozen_inputs"]["prediction_static_headers"][
        "outputs/predictions/binary_minirocket_ridge_nested_cv_oof.csv"
    ]
    observed_header = list(pd.read_csv(comparator_path, nrows=0).columns)
    if tracker is not None:
        tracker.mark("prediction_schema_or_header_inspected")
    if observed_header != expected_header:
        raise ValueError("Frozen OOF prediction header does not match the static contract.")
    oof_columns = ["sample_id", "y_true", "p_raw"]
    oof_frame = _read_csv_exact(
        comparator_path,
        oof_columns,
        approved_columns["locked_score_execution_oof_or_test"],
    )
    if tracker is not None:
        tracker.mark("participant_level_score_values_read")
    if len(oof_frame) != EXPECTED_TRAIN_N:
        raise ValueError("Frozen comparator row count is invalid.")
    oof_ids = _require_integer_vector(
        oof_frame["sample_id"].to_numpy(),
        expected_length=EXPECTED_TRAIN_N,
        role="Frozen comparator sample",
    )
    oof_labels = _require_integer_vector(
        oof_frame["y_true"].to_numpy(),
        expected_length=EXPECTED_TRAIN_N,
        role="Frozen comparator label",
    )
    canonical_ids = sample_ids[train_idx]
    canonical_labels = labels[train_idx]
    if not np.array_equal(oof_ids, canonical_ids) or not np.array_equal(oof_labels, canonical_labels):
        raise ValueError("Frozen comparator does not preserve locked train order and labels.")
    scores = oof_frame["p_raw"].to_numpy(dtype=np.float64)
    if oof_frame["p_raw"].dtype.kind not in "fiu":
        raise ValueError("Frozen comparator scores must have a numeric dtype.")
    summary_columns = ["scope", "outer_fold", "metric", "value"]
    summary = _read_csv_exact(aggregate_path, summary_columns, summary_columns)
    frozen_auc = validate_frozen_comparator(canonical_labels, scores, summary)
    if tracker is not None:
        tracker.mark("new_score_derived_results_computed")
    return Analysis3Inputs(
        x=np.asarray(x[train_idx]),
        y=np.asarray(canonical_labels),
        date_groups=np.asarray(dates, dtype=object)[train_idx],
        sample_ids=np.asarray(canonical_ids),
        frozen_random_oof_scores=np.asarray(scores, dtype=np.float64),
        frozen_random_oof_auc=frozen_auc,
    )


def _fold_diagnostics(
    fold_id: int,
    fit_idx: np.ndarray,
    validation_idx: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    """Build aggregate diagnostics for one fixed fold."""
    y_fit = y[fit_idx]
    y_validation = y[validation_idx]
    fit_groups = set(groups[fit_idx].tolist())
    validation_groups = set(groups[validation_idx].tolist())
    return {
        "fold_id": fold_id,
        "n_fit": int(fit_idx.size),
        "n_validation": int(validation_idx.size),
        "n_fit_positive": int(y_fit.sum()),
        "n_fit_negative": int(fit_idx.size - y_fit.sum()),
        "n_validation_positive": int(y_validation.sum()),
        "n_validation_negative": int(validation_idx.size - y_validation.sum()),
        "n_fit_date_groups": len(fit_groups),
        "n_validation_date_groups": len(validation_groups),
        "date_overlap_count": len(fit_groups & validation_groups),
        "calibrator_fit_feasible": bool(np.sum(y_fit == 0) >= 5 and np.sum(y_fit == 1) >= 5),
        "validation_auc_roc": None,
        "estimability_status": "estimable",
        "nonestimable_reason": None,
    }


def _split_once(y: np.ndarray, groups: np.ndarray) -> tuple[Any, tuple[FoldDefinition, ...]]:
    """Construct and apply the exact fixed splitter once."""
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=False)
    if splitter.shuffle is not False or splitter.random_state is not None:
        raise RuntimeError("Fixed splitter construction violated the frozen contract.")
    placeholder = np.zeros((y.size, 1), dtype=np.uint8)
    folds = tuple(
        FoldDefinition(
            fold_id=fold_id,
            fit_idx=np.asarray(fit_idx, dtype=np.int64),
            validation_idx=np.asarray(validation_idx, dtype=np.int64),
        )
        for fold_id, (fit_idx, validation_idx) in enumerate(splitter.split(placeholder, y, groups), start=1)
    )
    if len(folds) != 5:
        raise RuntimeError("Fixed splitter did not produce exactly five folds.")
    return splitter, folds


def build_fixed_partition(y_true: np.ndarray, date_groups: np.ndarray) -> PartitionResult:
    """Build, validate, and classify the fixed unshuffled date-grouped partition."""
    y = validate_exact_binary_labels_01(y_true)
    groups = np.asarray(date_groups, dtype=object)
    if groups.ndim != 1 or groups.size != y.size or pd.isna(groups).any():
        raise ValueError("Analysis 3 date-group vector is invalid.")
    if pd.unique(groups).size < 5:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=False)
        if splitter.shuffle is not False or splitter.random_state is not None:
            raise RuntimeError("Fixed splitter construction violated the frozen contract.")
        empty_rows = tuple(
            {
                "fold_id": fold_id,
                **{column: None for column in FOLD_COLUMNS[1:12]},
                "estimability_status": "non_estimable",
                "nonestimable_reason": "insufficient_unique_date_groups",
            }
            for fold_id in range(1, 6)
        )
        return PartitionResult(
            splitter=splitter,
            folds=(),
            fold_rows=empty_rows,
            feasible=False,
            reason="insufficient_unique_date_groups",
        )
    splitter, folds = _split_once(y, groups)
    validation_positions = np.concatenate([fold.validation_idx for fold in folds])
    if not np.array_equal(np.sort(validation_positions), np.arange(y.size)):
        raise RuntimeError("Fixed splitter validation coverage is missing or duplicated.")
    rows: list[dict[str, Any]] = []
    class_degenerate = False
    calibration_infeasible = False
    for fold in folds:
        if np.intersect1d(fold.fit_idx, fold.validation_idx).size:
            raise RuntimeError("Fixed splitter fit and validation rows overlap.")
        row = _fold_diagnostics(fold.fold_id, fold.fit_idx, fold.validation_idx, y, groups)
        if row["date_overlap_count"] != 0:
            raise RuntimeError("Fixed splitter fit and validation date groups overlap.")
        if len(np.unique(y[fold.fit_idx])) != 2 or len(np.unique(y[fold.validation_idx])) != 2:
            class_degenerate = True
        if not row["calibrator_fit_feasible"]:
            calibration_infeasible = True
        rows.append(row)
    reason: str | None = None
    if class_degenerate:
        reason = "fixed_partition_class_degenerate"
    elif calibration_infeasible:
        reason = "outer_fit_calibration_class_count_below_5"
    if reason is not None:
        for row in rows:
            row["estimability_status"] = "non_estimable"
            row["nonestimable_reason"] = reason
        return PartitionResult(splitter, folds, tuple(rows), False, reason)
    return PartitionResult(splitter, folds, tuple(rows), True, None)


def rank_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUROC from average ranks, giving ties one-half credit."""
    y = validate_exact_binary_labels_01(y_true)
    p = np.asarray(scores, dtype=np.float64)
    if p.ndim != 1 or p.size != y.size or not np.isfinite(p).all():
        raise ValueError("AUROC scores must be one finite aligned vector.")
    n_positive = int(y.sum())
    n_negative = int(y.size - n_positive)
    if n_positive == 0 or n_negative == 0:
        raise ValueError("AUROC requires both endpoint classes.")
    ranks = rankdata(p, method="average")
    numerator = float(ranks[y == 1].sum() - n_positive * (n_positive + 1) / 2.0)
    return numerator / float(n_positive * n_negative)


def assemble_oof_vector(
    probabilities: np.ndarray,
    validation_positions: np.ndarray,
    *,
    expected_length: int,
) -> np.ndarray:
    """Validate canonical position coverage and return one ordered OOF vector."""
    values = np.asarray(probabilities, dtype=np.float64)
    positions = np.asarray(validation_positions)
    if values.ndim != 1 or positions.ndim != 1 or values.size != positions.size:
        raise ValueError("OOF probabilities and validation positions are not aligned.")
    if values.size != expected_length:
        raise ValueError("OOF vector has the wrong length.")
    if positions.dtype.kind not in "iu":
        raise ValueError("OOF validation positions must be integer positions.")
    if np.unique(positions).size != expected_length:
        raise ValueError("OOF validation positions contain duplicate or missing positions.")
    if not np.array_equal(positions, np.arange(expected_length)):
        raise ValueError("OOF validation position order is not canonical.")
    if not np.isfinite(values).all():
        raise ValueError("OOF probabilities must be finite.")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("OOF probability is outside the unit interval.")
    return values


def canonical_oof_digest(probabilities: np.ndarray, *, expected_length: int) -> str:
    """Hash contiguous little-endian float64 OOF bytes in canonical order."""
    positions = np.arange(expected_length, dtype=np.int64)
    values = assemble_oof_vector(probabilities, positions, expected_length=expected_length)
    canonical = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def paired_date_cluster_bootstrap(
    y_true: np.ndarray,
    date_groups: np.ndarray,
    date_grouped_scores: np.ndarray,
    frozen_random_scores: np.ndarray,
) -> BootstrapResult:
    """Run the fixed paired date-cluster percentile bootstrap."""
    y = validate_exact_binary_labels_01(y_true)
    groups = np.asarray(date_groups, dtype=object)
    date_scores = np.asarray(date_grouped_scores, dtype=np.float64)
    random_scores = np.asarray(frozen_random_scores, dtype=np.float64)
    if (
        groups.ndim != 1
        or groups.size != y.size
        or date_scores.shape != y.shape
        or random_scores.shape != y.shape
        or not np.isfinite(date_scores).all()
        or not np.isfinite(random_scores).all()
    ):
        raise ValueError("Paired bootstrap inputs are not finite aligned vectors.")
    unique_groups = np.asarray(pd.unique(groups), dtype=object)
    group_indices = [np.flatnonzero(groups == group) for group in unique_groups]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    date_values: list[float] = []
    delta_values: list[float] = []
    attempts = 0
    while len(date_values) < BOOTSTRAP_VALID_TARGET and attempts < BOOTSTRAP_ATTEMPT_CAP:
        attempts += 1
        selected = rng.choice(np.arange(unique_groups.size), size=unique_groups.size, replace=True)
        sampled_idx = np.concatenate([group_indices[int(index)] for index in selected])
        sampled_y = y[sampled_idx]
        if np.unique(sampled_y).size < 2:
            continue
        date_auc = rank_auc(sampled_y, date_scores[sampled_idx])
        random_auc = rank_auc(sampled_y, random_scores[sampled_idx])
        date_values.append(date_auc)
        delta_values.append(date_auc - random_auc)
    n_valid = len(date_values)
    if n_valid < BOOTSTRAP_VALID_TARGET:
        return BootstrapResult(None, None, n_valid, attempts, BOOTSTRAP_SEED, BOOTSTRAP_REASON)
    date_limits = np.quantile(np.asarray(date_values), [0.025, 0.975], method="linear")
    delta_limits = np.quantile(np.asarray(delta_values), [0.025, 0.975], method="linear")
    return BootstrapResult(
        (float(date_limits[0]), float(date_limits[1])),
        (float(delta_limits[0]), float(delta_limits[1])),
        n_valid,
        attempts,
        BOOTSTRAP_SEED,
        None,
    )


def _registry(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Return one central aggregate schema registry entry."""
    registry = config["row_order_and_schema_validation_registry"][key]
    expected = VALIDATION_COLUMNS if key == "date_grouped_validation" else FOLD_COLUMNS
    if list(registry["column_order"]) != expected:
        raise ValueError("Central Analysis 3 schema column order is invalid.")
    return registry


def _frame_from_records(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    key: str,
) -> pd.DataFrame:
    """Build one typed frame from central registry records."""
    return build_registry_frame(records, _registry(config, key))


def _validation_base(inputs: Analysis3Inputs) -> dict[str, Any]:
    """Build fixed fields shared by estimable and non-estimable validation rows."""
    return {
        "analysis_id": ANALYSIS_ID,
        "n": int(inputs.y.size),
        "n_positive": int(inputs.y.sum()),
        "n_negative": int(inputs.y.size - inputs.y.sum()),
        "n_date_groups": int(pd.unique(inputs.date_groups).size),
        "n_folds": 5,
        "splitter": "StratifiedGroupKFold",
        "shuffle": False,
        "split_seed": None,
        "split_seed_status": "not_applicable",
        "model_seed": MODEL_SEED,
    }


def build_inestimable_candidate(
    inputs: Analysis3Inputs,
    partition: PartitionResult,
    config: Mapping[str, Any],
) -> CandidateArtifacts:
    """Build retained aggregate rows for a planned pre-fit feasibility failure."""
    if partition.feasible or partition.reason not in FEASIBILITY_REASONS:
        raise ValueError("Inestimable candidate requires one approved feasibility reason.")
    reason = partition.reason
    validation_record = {
        **_validation_base(inputs),
        "date_grouped_auc_roc": None,
        "date_grouped_auc_ci_low": None,
        "date_grouped_auc_ci_high": None,
        "frozen_random_oof_auc_roc": float(inputs.frozen_random_oof_auc),
        "auc_difference_date_grouped_minus_random": None,
        "delta_ci_low": None,
        "delta_ci_high": None,
        "ci_method": None,
        "n_bootstrap_valid": 0,
        "n_bootstrap_attempts": 0,
        "bootstrap_seed": None,
        "tie_credit": 0.5,
        "oof_vector_sha256": None,
        "date_grouped_auc_status": "non_estimable",
        "date_grouped_auc_nonestimable_reason": reason,
        "date_grouped_auc_ci_status": "non_estimable",
        "date_grouped_auc_ci_nonestimable_reason": reason,
        "auc_difference_status": "non_estimable",
        "auc_difference_nonestimable_reason": reason,
        "auc_difference_ci_status": "non_estimable",
        "auc_difference_ci_nonestimable_reason": reason,
    }
    validation = _frame_from_records([validation_record], config, "date_grouped_validation")
    fold_audit = _frame_from_records(partition.fold_rows, config, "date_grouped_fold_audit")
    return CandidateArtifacts(validation, fold_audit, None)


def _default_trainer_factory(config: Mapping[str, Any]) -> MiniRocketRidgeTrainer:
    """Construct the frozen MiniRocket trainer configuration."""
    del config
    return MiniRocketRidgeTrainer(
        global_config_path=PROJECT_ROOT / "configs/modeling_global.yaml",
        model_config_path=PROJECT_ROOT / "configs/modeling_minirocket_ridge.yaml",
    )


def _verify_trainer_contract(trainer: Any, config: Mapping[str, Any]) -> None:
    """Require the frozen MiniRocket, Ridge, and Layer-A settings."""
    model = trainer.model_config
    mini = model["minirocket"]
    calibration = model["calibration_layer_a"]
    configured = config["analysis_3_date_grouped_minirocket"]
    trainer_contract = configured["trainer"]
    expected_mini = trainer_contract["minirocket"]
    if (
        int(mini["num_kernels"]) != int(expected_mini["num_kernels"])
        or int(mini["random_state"]) != int(expected_mini["random_state"])
        or int(expected_mini["random_state"]) != MODEL_SEED
    ):
        raise RuntimeError("MiniRocket configuration differs from the frozen contract.")
    ridge = model["ridge"]
    expected_weight = trainer_contract["ridge_class_weight"]
    if ridge.get("class_weight") != expected_weight or trainer.current_class_weights_ != expected_weight:
        raise RuntimeError("Ridge class weighting differs from the frozen contract.")
    expected_calibration = configured["internal_calibration_unchanged"]
    if (
        calibration["method"] != expected_calibration["method"]
        or int(calibration["cv"]) != int(expected_calibration["cv"])
        or calibration["ensemble"] is not expected_calibration["ensemble"]
    ):
        raise RuntimeError("Layer-A calibration differs from the frozen contract.")


def _extract_probabilities(result: Any) -> np.ndarray:
    """Extract validation probabilities without retaining fitted objects."""
    if isinstance(result, Mapping):
        probabilities = result.get("val_probabilities")
    else:
        probabilities = getattr(result, "val_probabilities", None)
    if probabilities is None:
        raise RuntimeError("Fold result omitted validation probabilities.")
    return np.asarray(probabilities, dtype=np.float64).copy()


def run_isolated_candidate(
    inputs: Analysis3Inputs,
    partition: PartitionResult,
    config: Mapping[str, Any],
    *,
    trainer_factory: Callable[[Mapping[str, Any]], Any] = _default_trainer_factory,
    tracker: AccessStateTracker | None = None,
) -> CandidateArtifacts:
    """Compute one isolated candidate from the caller's validated fixed partition."""
    if not isinstance(partition, PartitionResult):
        raise TypeError("Candidate execution requires a validated PartitionResult.")
    if not partition.feasible:
        return build_inestimable_candidate(inputs, partition, config)
    trainer = trainer_factory(config)
    trainer.x_train_ = inputs.x
    trainer.y_train_ = inputs.y
    trainer.current_class_weights_ = config["analysis_3_date_grouped_minirocket"]["trainer"][
        "ridge_class_weight"
    ]
    _verify_trainer_contract(trainer, config)
    probabilities = np.empty(inputs.y.size, dtype=np.float64)
    assigned = np.zeros(inputs.y.size, dtype=bool)
    fold_records: list[dict[str, Any]] = []
    outer_calls = 0
    for fold, base_row in zip(partition.folds, partition.fold_rows, strict=True):
        if tracker is not None and outer_calls == 0:
            tracker.mark("auxiliary_model_fit_executed")
        result = trainer.fit_fold(
            outer_fold=fold.fold_id,
            inner_train_idx=fold.fit_idx,
            inner_val_idx=fold.validation_idx,
            hparams={},
        )
        outer_calls += 1
        fold_probabilities = _extract_probabilities(result)
        del result
        if fold_probabilities.shape != (fold.validation_idx.size,):
            raise RuntimeError("Fold validation probability length is invalid.")
        if assigned[fold.validation_idx].any():
            raise RuntimeError("A validation position was assigned more than once.")
        probabilities[fold.validation_idx] = fold_probabilities
        assigned[fold.validation_idx] = True
        fold_row = dict(base_row)
        fold_row["validation_auc_roc"] = rank_auc(inputs.y[fold.validation_idx], fold_probabilities)
        fold_records.append(fold_row)
    if outer_calls != 5 or not assigned.all():
        raise RuntimeError("Analysis 3 did not complete exactly five outer fold calls.")
    oof = assemble_oof_vector(probabilities, np.arange(inputs.y.size), expected_length=inputs.y.size)
    digest = canonical_oof_digest(oof, expected_length=inputs.y.size)
    date_auc = rank_auc(inputs.y, oof)
    random_auc = rank_auc(inputs.y, inputs.frozen_random_oof_scores)
    if abs(random_auc - inputs.frozen_random_oof_auc) > 1e-12:
        raise RuntimeError("Frozen random-fold comparator changed after alignment.")
    difference = date_auc - random_auc
    bootstrap = paired_date_cluster_bootstrap(
        inputs.y,
        inputs.date_groups,
        oof,
        inputs.frozen_random_oof_scores,
    )
    ci_estimable = bootstrap.nonestimable_reason is None
    validation_record = {
        **_validation_base(inputs),
        "date_grouped_auc_roc": date_auc,
        "date_grouped_auc_ci_low": bootstrap.date_auc_ci[0] if bootstrap.date_auc_ci else None,
        "date_grouped_auc_ci_high": bootstrap.date_auc_ci[1] if bootstrap.date_auc_ci else None,
        "frozen_random_oof_auc_roc": random_auc,
        "auc_difference_date_grouped_minus_random": difference,
        "delta_ci_low": bootstrap.delta_ci[0] if bootstrap.delta_ci else None,
        "delta_ci_high": bootstrap.delta_ci[1] if bootstrap.delta_ci else None,
        "ci_method": "percentile_bootstrap_linear_quantile",
        "n_bootstrap_valid": bootstrap.n_valid,
        "n_bootstrap_attempts": bootstrap.n_attempts,
        "bootstrap_seed": bootstrap.seed,
        "tie_credit": 0.5,
        "oof_vector_sha256": digest,
        "date_grouped_auc_status": "estimable",
        "date_grouped_auc_nonestimable_reason": None,
        "date_grouped_auc_ci_status": "estimable" if ci_estimable else "non_estimable",
        "date_grouped_auc_ci_nonestimable_reason": bootstrap.nonestimable_reason,
        "auc_difference_status": "estimable",
        "auc_difference_nonestimable_reason": None,
        "auc_difference_ci_status": "estimable" if ci_estimable else "non_estimable",
        "auc_difference_ci_nonestimable_reason": bootstrap.nonestimable_reason,
    }
    validation = _frame_from_records([validation_record], config, "date_grouped_validation")
    fold_audit = _frame_from_records(fold_records, config, "date_grouped_fold_audit")
    artifact = CandidateArtifacts(validation, fold_audit, digest)
    validate_analysis3_outputs(validation, fold_audit, config, expected_n=inputs.y.size)
    return artifact


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a typed frame to registry records with Python missing values."""
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        records.append(
            {
                key: None if value is pd.NA or (isinstance(value, float) and np.isnan(value)) else value
                for key, value in row.items()
            }
        )
    return records


def _require_status_pair(
    row: Mapping[str, Any],
    *,
    status: str,
    reason: str,
    populated: Sequence[str],
) -> None:
    """Validate one point/interval status and reason dependency."""
    observed_status = row[status]
    observed_reason = row[reason]
    values_populated = all(row[column] is not None and not pd.isna(row[column]) for column in populated)
    if observed_status == "estimable":
        if observed_reason is not None and not pd.isna(observed_reason):
            raise ValueError("Estimable field must have an empty non-estimability reason.")
        if not values_populated:
            raise ValueError("Estimable field is missing a required estimate.")
    elif observed_status == "non_estimable":
        if observed_reason is None or pd.isna(observed_reason):
            raise ValueError("Non-estimable field requires an approved reason.")
        if any(row[column] is not None and not pd.isna(row[column]) for column in populated):
            raise ValueError("Non-estimable field must not contain an estimate.")
    else:
        raise ValueError("Analysis 3 status is outside the closed registry.")


def validate_analysis3_outputs(
    validation: pd.DataFrame,
    fold_audit: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    expected_n: int = EXPECTED_TRAIN_N,
) -> None:
    """Validate exact schemas, statuses, reconciliation, and aggregate privacy."""
    if list(validation.columns) != VALIDATION_COLUMNS or list(fold_audit.columns) != FOLD_COLUMNS:
        raise ValueError("Analysis 3 output schema contains an undeclared field or wrong order.")
    validate_registry_records(_records(validation), _registry(config, "date_grouped_validation"))
    validate_registry_records(_records(fold_audit), _registry(config, "date_grouped_fold_audit"))
    if len(validation) != 1 or len(fold_audit) != 5:
        raise ValueError("Analysis 3 row inventory is invalid.")
    row = validation.iloc[0]
    if (
        row["analysis_id"] != ANALYSIS_ID
        or int(row["n"]) != expected_n
        or int(row["n_positive"]) + int(row["n_negative"]) != expected_n
        or int(row["n_folds"]) != 5
        or row["splitter"] != "StratifiedGroupKFold"
        or bool(row["shuffle"])
        or not pd.isna(row["split_seed"])
        or row["split_seed_status"] != "not_applicable"
        or int(row["model_seed"]) != MODEL_SEED
        or float(row["tie_credit"]) != 0.5
    ):
        raise ValueError("Analysis 3 fixed output fields do not match the contract.")
    if fold_audit["fold_id"].astype(int).tolist() != [1, 2, 3, 4, 5]:
        raise ValueError("Analysis 3 fold order is invalid.")
    allowed_reasons = set(FEASIBILITY_REASONS) | {BOOTSTRAP_REASON}
    _require_status_pair(
        row,
        status="date_grouped_auc_status",
        reason="date_grouped_auc_nonestimable_reason",
        populated=["date_grouped_auc_roc"],
    )
    _require_status_pair(
        row,
        status="date_grouped_auc_ci_status",
        reason="date_grouped_auc_ci_nonestimable_reason",
        populated=["date_grouped_auc_ci_low", "date_grouped_auc_ci_high"],
    )
    _require_status_pair(
        row,
        status="auc_difference_status",
        reason="auc_difference_nonestimable_reason",
        populated=["auc_difference_date_grouped_minus_random"],
    )
    _require_status_pair(
        row,
        status="auc_difference_ci_status",
        reason="auc_difference_ci_nonestimable_reason",
        populated=["delta_ci_low", "delta_ci_high"],
    )
    for reason_column in (
        "date_grouped_auc_nonestimable_reason",
        "date_grouped_auc_ci_nonestimable_reason",
        "auc_difference_nonestimable_reason",
        "auc_difference_ci_nonestimable_reason",
    ):
        reason = row[reason_column]
        if reason is not None and not pd.isna(reason) and reason not in allowed_reasons:
            raise ValueError("Analysis 3 output uses a reason outside the closed registry.")
    if row["date_grouped_auc_status"] != row["auc_difference_status"]:
        raise ValueError("Analysis 3 point-estimate statuses disagree.")
    if row["date_grouped_auc_ci_status"] != row["auc_difference_ci_status"]:
        raise ValueError("Analysis 3 confidence-interval statuses disagree.")
    if row["date_grouped_auc_status"] == "estimable":
        if not isinstance(row["oof_vector_sha256"], str) or _SHA256_PATTERN.fullmatch(row["oof_vector_sha256"]) is None:
            raise ValueError("Estimable Analysis 3 output requires the canonical OOF digest.")
        if int(row["bootstrap_seed"]) != BOOTSTRAP_SEED:
            raise ValueError("Post-fit bootstrap seed must equal 42.")
        if row["ci_method"] != "percentile_bootstrap_linear_quantile":
            raise ValueError("Post-fit CI method is invalid.")
        if int(row["n_bootstrap_attempts"]) > BOOTSTRAP_ATTEMPT_CAP:
            raise ValueError("Bootstrap attempts exceed the fixed cap.")
        if int(row["n_bootstrap_valid"]) > BOOTSTRAP_VALID_TARGET:
            raise ValueError("Bootstrap valid count exceeds the fixed target.")
        valid_count = int(row["n_bootstrap_valid"])
        attempt_count = int(row["n_bootstrap_attempts"])
        ci_estimable = row["date_grouped_auc_ci_status"] == "estimable"
        if ci_estimable and valid_count != BOOTSTRAP_VALID_TARGET:
            raise ValueError("Estimable bootstrap CI requires exactly 2,000 valid replicates.")
        if attempt_count < valid_count or attempt_count > BOOTSTRAP_ATTEMPT_CAP:
            raise ValueError("Bootstrap attempt count is inconsistent with valid replicates.")
        if not ci_estimable and (
            row["date_grouped_auc_ci_nonestimable_reason"] != BOOTSTRAP_REASON
            or row["auc_difference_ci_nonestimable_reason"] != BOOTSTRAP_REASON
            or valid_count >= BOOTSTRAP_VALID_TARGET
            or attempt_count != BOOTSTRAP_ATTEMPT_CAP
        ):
            raise ValueError("Post-fit non-estimable bootstrap state is internally inconsistent.")
        if ci_estimable and (
            float(row["date_grouped_auc_ci_low"]) > float(row["date_grouped_auc_ci_high"])
            or float(row["delta_ci_low"]) > float(row["delta_ci_high"])
        ):
            raise ValueError("Bootstrap confidence-interval bounds are reversed.")
        expected_delta = float(row["date_grouped_auc_roc"]) - float(row["frozen_random_oof_auc_roc"])
        if not np.isclose(
            float(row["auc_difference_date_grouped_minus_random"]),
            expected_delta,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("Analysis 3 AUROC difference does not match its declared sign contract.")
        if (fold_audit["date_overlap_count"].astype("Int64") != 0).any():
            raise ValueError("Fold audit reports date-group overlap.")
        if not fold_audit["calibrator_fit_feasible"].astype("boolean").all():
            raise ValueError("Estimable candidate has an infeasible calibrator fold.")
        if fold_audit["validation_auc_roc"].isna().any():
            raise ValueError("Estimable fold audit is missing an AUROC.")
        if int(fold_audit["n_validation"].sum()) != expected_n:
            raise ValueError("Fold validation counts do not reconcile.")
        if int(fold_audit["n_validation_positive"].sum()) != int(row["n_positive"]):
            raise ValueError("Fold positive validation counts do not reconcile.")
        if int(fold_audit["n_validation_negative"].sum()) != int(row["n_negative"]):
            raise ValueError("Fold negative validation counts do not reconcile.")
    else:
        if not pd.isna(row["oof_vector_sha256"]):
            raise ValueError("Pre-fit non-estimability must not emit an OOF digest.")
        if int(row["n_bootstrap_valid"]) != 0 or int(row["n_bootstrap_attempts"]) != 0:
            raise ValueError("Pre-fit non-estimability must have zero bootstrap counts.")
        if not pd.isna(row["bootstrap_seed"]):
            raise ValueError("Pre-fit non-estimability must not emit a bootstrap seed.")
    for fold_row in fold_audit.to_dict(orient="records"):
        if fold_row["estimability_status"] == "estimable":
            if fold_row["nonestimable_reason"] is not None and not pd.isna(fold_row["nonestimable_reason"]):
                raise ValueError("Estimable fold must have an empty reason.")
        elif fold_row["estimability_status"] == "non_estimable":
            if fold_row["nonestimable_reason"] not in FEASIBILITY_REASONS:
                raise ValueError("Non-estimable fold reason is outside the feasibility registry.")
            if fold_row["validation_auc_roc"] is not None and not pd.isna(fold_row["validation_auc_roc"]):
                raise ValueError("Non-estimable fold must not emit an AUROC.")
        else:
            raise ValueError("Fold status is outside the closed registry.")


def serialize_candidate_csvs(candidate: CandidateArtifacts) -> tuple[bytes, bytes]:
    """Serialize the two aggregate frames deterministically."""
    return (
        candidate.validation.to_csv(index=False, lineterminator="\n").encode("utf-8"),
        candidate.fold_audit.to_csv(index=False, lineterminator="\n").encode("utf-8"),
    )


def verify_source_snapshot(snapshot: Mapping[str, str]) -> None:
    """Require exact source hashes without exposing file contents."""
    for path_text, expected in snapshot.items():
        path = Path(path_text)
        path = path if path.is_absolute() else PROJECT_ROOT / path
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError("Required source integrity snapshot changed.")


def write_deterministic_outputs(
    first: CandidateArtifacts,
    second: CandidateArtifacts,
    validation_path: Path,
    fold_path: Path,
    log_path: Path,
    *,
    config: Mapping[str, Any],
    source_entry_hashes: Mapping[str, str],
    source_exit_hashes: Mapping[str, str],
    completion_log_factory: Callable[[str, str, str], bytes],
) -> tuple[Path, Path, Path]:
    """Validate fully, then promote scientific CSVs and the receipt last."""
    if classify_target_state(validation_path, fold_path, log_path) != "clean":
        raise RuntimeError("Session 5 targets are not absent.")
    first_bytes = serialize_candidate_csvs(first)
    second_bytes = serialize_candidate_csvs(second)
    if first_bytes != second_bytes or first.oof_digest != second.oof_digest:
        raise RuntimeError("Isolated deterministic candidates disagree.")
    if OrderedDict(source_entry_hashes) != OrderedDict(source_exit_hashes):
        raise RuntimeError("Entry and exit source integrity snapshots differ.")
    for destination in (validation_path, fold_path, log_path):
        destination.parent.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    payloads = first_bytes
    destinations = (validation_path, fold_path)
    try:
        for destination, payload in zip(destinations, payloads, strict=True):
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
                staged.append(Path(handle.name))
        for path, expected in zip(staged, payloads, strict=True):
            if path.read_bytes() != expected:
                raise RuntimeError("Staged output read-back validation failed.")
        if not registry_readback_matches(
            first.validation,
            staged[0],
            _registry(config, "date_grouped_validation"),
        ):
            raise RuntimeError("Staged validation CSV failed registry read-back.")
        if not registry_readback_matches(
            first.fold_audit,
            staged[1],
            _registry(config, "date_grouped_fold_audit"),
        ):
            raise RuntimeError("Staged fold-audit CSV failed registry read-back.")
        if classify_target_state(validation_path, fold_path, log_path) != "clean":
            raise RuntimeError("Session 5 targets appeared during staged validation.")
        generated_at_utc = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        completion_log_bytes = completion_log_factory(
            generated_at_utc,
            hashlib.sha256(first_bytes[0]).hexdigest(),
            hashlib.sha256(first_bytes[1]).hexdigest(),
        )
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{log_path.name}.",
            suffix=".tmp",
            dir=log_path.parent,
            delete=False,
        ) as handle:
            handle.write(completion_log_bytes)
            handle.flush()
            os.fsync(handle.fileno())
            staged.append(Path(handle.name))
        if staged[2].read_bytes() != completion_log_bytes:
            raise RuntimeError("Staged completion-log read-back validation failed.")
        for staged_path, destination in zip(staged, (validation_path, fold_path, log_path), strict=True):
            if destination.exists():
                raise RuntimeError("Session 5 destination appeared before promotion.")
            os.replace(staged_path, destination)
    except Exception:
        for path in staged:
            if path.exists():
                path.unlink()
        raise
    return validation_path, fold_path, log_path


def _completion_log_text(
    *,
    freeze_commit: str,
    validation_path: str,
    validation_hash: str,
    fold_path: str,
    fold_hash: str,
    tracker: AccessStateTracker,
    reasons: Sequence[str],
    generated_at_utc: str,
) -> str:
    """Build the exact Session 5 completion receipt."""
    unique_reasons = sorted(set(reasons))
    summary = "none" if not unique_reasons else "present_retained"
    state = tracker.snapshot()
    records: list[tuple[str, ...]] = [
        ("header", "posthoc_final_completion_log.v1", "posthoc_final_supplementary", generated_at_utc),
        ("session_identity", "date_grouped", "complete"),
        ("freeze_commit", freeze_commit),
        ("output_hash", validation_path, validation_hash),
        ("output_hash", fold_path, fold_hash),
        ("deviations_or_nonestimability_summary", summary, ",".join(unique_reasons), ""),
        (
            "access_state_session_activity",
            *(str(value).lower() for value in state["session_activity"].values()),
        ),
        (
            "access_state_cumulative_through_session",
            *(str(value).lower() for value in state["cumulative_through_session"].values()),
        ),
        ("completion_receipt", generated_at_utc, "complete"),
    ]
    return "\n".join("\t".join(record) for record in records) + "\n"


def _source_integrity_snapshot(config: Mapping[str, Any]) -> OrderedDict[str, str]:
    """Capture the exact frozen and central-source integrity snapshot."""
    expected = OrderedDict(config["frozen_inputs"]["whole_container_sha256"])
    for path, digest in config["source_integrity"]["analysis_3_central_training_sources"].items():
        expected[path] = digest
    session_0 = config["outputs"]["session_0"]
    scientific = config["outputs"]["scientific_and_closure"]
    completion_logs = config["outputs"]["completion_logs"]
    locked_scores_log = (
        completion_logs["paths"]["locked_scores"]
        if "paths" in completion_logs
        else completion_logs["locked_scores"]
    )
    authority_paths = [
        session_0["config"],
        session_0["methods"],
        session_0["freeze_manifest"],
        config["authority"]["reviewed_plan_path"],
        config["authority"]["agent_guidance_path"],
        scientific["score_blind_preflight"]["path"],
        *config["outputs"]["future_implementation_paths"],
        scientific["locked_score_metrics"]["path"],
        scientific["locked_score_composition"]["path"],
        scientific["qc5_correctness"]["path"],
        scientific["within_date_concordance"]["path"],
        locked_scores_log,
    ]
    observed: OrderedDict[str, str] = OrderedDict()
    for path in (*expected, *authority_paths):
        if path not in observed:
            observed[path] = sha256_file(_resolve(path))
    if any(observed[path] != digest for path, digest in expected.items()):
        raise RuntimeError("A frozen authority source differs from its declared hash.")
    return observed


def execute_approved(
    config_path: Path,
    freeze_commit: str,
    amendment_freeze_commit: str,
) -> tuple[Path, Path, Path]:
    """Execute future approved Session 5 after every fail-closed authority gate."""
    config, _, prior = validate_analysis3_authority(config_path, freeze_commit, amendment_freeze_commit)
    source_entry = _source_integrity_snapshot(config)
    tracker = AccessStateTracker(prior)
    tracker.mark("prediction_container_bytes_hashed")
    inputs = load_analysis3_inputs(config, tracker=tracker)
    partition = build_fixed_partition(inputs.y, inputs.date_groups)
    first = run_isolated_candidate(inputs, partition, config, tracker=tracker)
    second = run_isolated_candidate(inputs, partition, config)
    validate_analysis3_outputs(first.validation, first.fold_audit, config)
    validate_analysis3_outputs(second.validation, second.fold_audit, config)
    source_exit = _source_integrity_snapshot(config)
    validation_path, fold_path, log_path = _date_grouped_target_paths(config)
    reasons = [
        str(value)
        for column in (
            "date_grouped_auc_nonestimable_reason",
            "date_grouped_auc_ci_nonestimable_reason",
            "auc_difference_nonestimable_reason",
            "auc_difference_ci_nonestimable_reason",
        )
        for value in first.validation[column].dropna().tolist()
    ]
    validation_relative = str(validation_path.relative_to(PROJECT_ROOT))
    fold_relative = str(fold_path.relative_to(PROJECT_ROOT))
    grammar = config["outputs"]["completion_logs"]["machine_validated_log_grammar"]
    binding = dict(grammar["session_bindings_exact"]["date_grouped"])
    binding.setdefault("session_name", "date_grouped")
    state = tracker.snapshot()

    def _validated_completion_log(
        generated_at_utc: str,
        validation_hash: str,
        fold_hash: str,
    ) -> bytes:
        log_text = _completion_log_text(
            freeze_commit=freeze_commit,
            validation_path=validation_relative,
            validation_hash=validation_hash,
            fold_path=fold_relative,
            fold_hash=fold_hash,
            tracker=tracker,
            reasons=reasons,
            generated_at_utc=generated_at_utc,
        )
        validate_completion_log(
            log_text,
            grammar=grammar,
            session_binding=binding,
            expected_freeze_commit=freeze_commit,
            expected_activity=state["session_activity"],
            expected_cumulative=state["cumulative_through_session"],
            cumulative_prior=prior,
            allowed_reason_codes=(*FEASIBILITY_REASONS, BOOTSTRAP_REASON),
        )
        return log_text.encode("utf-8")

    return write_deterministic_outputs(
        first,
        second,
        validation_path,
        fold_path,
        log_path,
        config=config,
        source_entry_hashes=source_entry,
        source_exit_hashes=source_exit,
        completion_log_factory=_validated_completion_log,
    )
