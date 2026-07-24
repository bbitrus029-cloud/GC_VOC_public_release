"""Synthetic tests for Session 3 reporting-closure infrastructure."""

from __future__ import annotations

import importlib.util
import inspect
import hashlib
import math
import copy
import json
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pytest
import yaml
from statsmodels.stats.proportion import proportion_confint

from src.analysis import posthoc_final_audit as audit
from src.analysis import posthoc_final_date_grouped as date_grouped
from src.analysis import posthoc_final_locked_scores as locked_scores
from src.analysis import posthoc_final_reporting as reporting
from src.analysis.posthoc_final_common import ACCESS_STATE_KEYS, build_registry_frame
from src.config import PROJECT_ROOT


CONFIG_PATH = PROJECT_ROOT / "configs/post_hoc_final_supplementary.yaml"
SESSION0_FREEZE = "3c0ff6b3b40460a2c340ad3193222ee54979109d"
AMENDMENT_FREEZE = "2e08e8fb89d99bb62cca2e5204a13ff09e47d432"


@pytest.fixture(scope="module")
def config() -> dict[str, Any]:
    """Load the frozen result-free machine contract."""
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _ids(start: int, stop: int) -> np.ndarray:
    """Return a compact synthetic integer-ID range."""
    return np.arange(start, stop, dtype=np.int64)


def _calibration_fixture() -> tuple[np.ndarray, np.ndarray]:
    """Build 195 deterministic labels and probabilities with repeated ties."""
    p_cal = np.repeat(np.linspace(0.05, 0.95, 5), 39)
    y_true = np.tile(np.array([0, 1, 0, 1, 0], dtype=np.int64), 39)
    return y_true, p_cal


def _inclusion_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build aligned synthetic inclusion metadata and sample-index rows."""
    sample_id = np.arange(1, 13, dtype=np.int64)
    metadata = pd.DataFrame(
        {
            "sample_id": sample_id,
            "age": [40.0, 42.0, 44.0, 46.0, 48.0, np.nan, 60.0, 62.0, 64.0, 66.0, 68.0, np.nan],
            "sex(0male1female)": [1, 1, 1, 0, 0, np.nan, 0, 0, 0, 1, 1, np.nan],
        }
    )
    sample_index = pd.DataFrame(
        {
            "sample_id": sample_id,
            "label": [1, 1, 1, 0, 0, np.nan, 0, 0, 0, 1, 1, np.nan],
            "disease_category": [0, 1, 2, 3, 4, np.nan, 0, 1, 2, 3, 5, np.nan],
            "qc_status": ["pass"] * 6 + ["fail"] * 6,
            "fail_reasons": [""] * 6
            + [
                "qc1_env_sensor_fault",
                "qc2_acquisition_truncated",
                "qc3_unstable_baseline",
                "qc4_insufficient_response",
                "qc5_sensor_saturation",
                "qc6_excessive_spikes",
            ],
            "fallback_used": [False] * 12,
            "padded": [False] * 12,
            "in_final_output": [True] * 6 + [False] * 6,
        }
    )
    return metadata, sample_index


def _full_reporting_inputs(config: dict[str, Any]) -> dict[str, Any]:
    """Build complete synthetic Session 6 inputs with exact frozen-size membership."""
    source_n = 1040
    included_n = 972
    sample_ids = np.arange(source_n, dtype=np.int64)
    included = np.arange(source_n) < included_n
    allowed = config["outputs"]["scientific_and_closure"]["cohort_flow"][
        "validated_single_exclusion_reason_contract"
    ]["allowed_exclusion_reason_keys"]
    excluded_reasons = [allowed[0]] * 62 + list(allowed[1:])
    sample_index = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "label": sample_ids % 2,
            "disease_category": sample_ids % 6,
            "qc_status": np.where(included, "pass", "fail"),
            "fail_reasons": [""] * included_n + excluded_reasons,
            "fallback_used": np.zeros(source_n, dtype=bool),
            "padded": np.zeros(source_n, dtype=bool),
            "in_final_output": included,
        }
    )
    metadata = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "age": 40.0 + sample_ids % 31,
            "sex(0male1female)": sample_ids % 2,
        }
    )
    train_idx = np.arange(777, dtype=np.int64)
    test_idx = np.arange(777, 972, dtype=np.int64)
    test_ids = sample_ids[test_idx]
    return {
        "metadata": metadata,
        "sample_index": sample_index,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "test_predictions": pd.DataFrame(
            {
                "sample_id": test_ids[::-1],
                "y_true": (test_ids[::-1] % 2).astype(np.int64),
                "p_cal": np.repeat(np.linspace(0.05, 0.95, 5), 39),
            }
        ),
    }


def _full_reporting_candidates(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Build the five scientific candidate frames through the production calculator."""
    return reporting.calculate_reporting_candidates({"config": config}, _full_reporting_inputs(config))


def _alignment_inputs() -> dict[str, Any]:
    """Build the frozen-size synthetic split and deliberately reversed calibration rows."""
    sample_ids = np.arange(10_000, 11_040, dtype=np.int64)
    included = np.arange(1040) < 972
    sample_index = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "label": sample_ids % 2,
            "disease_category": sample_ids % 6,
            "qc_status": np.where(included, "pass", "fail"),
            "fail_reasons": [""] * 1040,
            "fallback_used": np.zeros(1040, dtype=bool),
            "padded": np.zeros(1040, dtype=bool),
            "in_final_output": included,
        }
    )
    train_idx = np.arange(777, dtype=np.int64)
    test_idx = np.arange(777, 972, dtype=np.int64)
    test_ids = sample_ids[test_idx]
    predictions = pd.DataFrame(
        {
            "sample_id": test_ids[::-1],
            "y_true": (test_ids[::-1] % 2).astype(np.int64),
            "p_cal": np.linspace(0.01, 0.99, 195),
        }
    )
    return {
        "sample_index": sample_index,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "test_predictions": predictions,
    }


def _all_false_access_state() -> OrderedDict[str, OrderedDict[str, bool]]:
    """Return the exact ordered all-false Session 3 ledger."""
    activity = OrderedDict((key, False) for key in ACCESS_STATE_KEYS)
    return OrderedDict(
        [
            ("session_activity", OrderedDict(activity)),
            ("cumulative_through_session", OrderedDict(activity)),
        ]
    )


def _synthetic_prior_aggregates(config: dict[str, Any]) -> OrderedDict[str, pd.DataFrame]:
    """Build scientifically coherent aggregate-only Session 4/5 fixtures."""
    def session4_view(sample_ids: np.ndarray) -> dict[str, np.ndarray]:
        """Create one endpoint-consistent synthetic locked-score view."""
        labels = (sample_ids % 2).astype(np.int64)
        disease = np.empty(sample_ids.size, dtype=np.int64)
        negative = np.flatnonzero(labels == 0)
        positive = np.flatnonzero(labels == 1)
        disease[negative] = np.resize(np.array([1, 2], dtype=np.int64), negative.size)
        disease[positive] = np.resize(np.array([3, 4], dtype=np.int64), positive.size)
        positions = np.arange(sample_ids.size)
        scores = np.clip(
            0.5 + 0.14 * (2 * labels - 1) + 0.3 * np.sin(positions * 0.73),
            0.01,
            0.99,
        )
        return {
            "y_true": labels,
            "p_raw": scores,
            "dates": np.where(positions < sample_ids.size // 2, "2030/1/1", "2030/1/2"),
            "disease_category": disease,
            "strict_pass": positions % 3 != 0,
        }

    views = OrderedDict(
        [
            ("oof", session4_view(np.arange(20_000, 20_777, dtype=np.int64))),
            ("test", session4_view(np.arange(20_777, 20_972, dtype=np.int64))),
        ]
    )
    session4 = locked_scores._execution_frames(config, views)
    n_positive = int(views["oof"]["y_true"].sum())
    n_negative = int(views["oof"]["y_true"].size - n_positive)
    validation_counts = [156, 156, 155, 155, 155]
    validation_positive = [78, 78, 78, 77, 77]
    validation_negative = [78, 78, 77, 78, 78]
    fold_records = []
    for index, (fold_n, fold_positive, fold_negative) in enumerate(
        zip(validation_counts, validation_positive, validation_negative, strict=True),
        start=1,
    ):
        fold_records.append(
            {
                "fold_id": index,
                "n_fit": 777 - fold_n,
                "n_validation": fold_n,
                "n_fit_positive": n_positive - fold_positive,
                "n_fit_negative": n_negative - fold_negative,
                "n_validation_positive": fold_positive,
                "n_validation_negative": fold_negative,
                "n_fit_date_groups": 32,
                "n_validation_date_groups": 8,
                "date_overlap_count": 0,
                "calibrator_fit_feasible": True,
                "validation_auc_roc": 0.68 + index * 0.02,
                "estimability_status": "estimable",
                "nonestimable_reason": None,
            }
        )
    validation_record = {
        "analysis_id": date_grouped.ANALYSIS_ID,
        "n": 777,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_date_groups": 40,
        "n_folds": 5,
        "splitter": "StratifiedGroupKFold",
        "shuffle": False,
        "split_seed": None,
        "split_seed_status": "not_applicable",
        "model_seed": 42,
        "date_grouped_auc_roc": 0.75,
        "date_grouped_auc_ci_low": 0.70,
        "date_grouped_auc_ci_high": 0.80,
        "frozen_random_oof_auc_roc": 0.70,
        "auc_difference_date_grouped_minus_random": 0.05,
        "delta_ci_low": 0.00,
        "delta_ci_high": 0.10,
        "ci_method": "percentile_bootstrap_linear_quantile",
        "n_bootstrap_valid": 2000,
        "n_bootstrap_attempts": 2000,
        "bootstrap_seed": 42,
        "tie_credit": 0.5,
        "oof_vector_sha256": "a" * 64,
        "date_grouped_auc_status": "estimable",
        "date_grouped_auc_nonestimable_reason": None,
        "date_grouped_auc_ci_status": "estimable",
        "date_grouped_auc_ci_nonestimable_reason": None,
        "auc_difference_status": "estimable",
        "auc_difference_nonestimable_reason": None,
        "auc_difference_ci_status": "estimable",
        "auc_difference_ci_nonestimable_reason": None,
    }
    registry = config["row_order_and_schema_validation_registry"]
    validation = build_registry_frame([validation_record], registry["date_grouped_validation"])
    folds = build_registry_frame(fold_records, registry["date_grouped_fold_audit"])
    date_grouped.validate_analysis3_outputs(validation, folds, config, expected_n=777)
    result: OrderedDict[str, pd.DataFrame] = OrderedDict()
    for name, frame in session4.items():
        result[config["outputs"]["scientific_and_closure"][name]["path"]] = frame
    result[registry["date_grouped_validation"]["path"]] = validation
    result[registry["date_grouped_fold_audit"]["path"]] = folds
    return result


def _write_bytes(root: Path, relative: str, payload: bytes) -> None:
    """Write one synthetic fixture container below a pytest temporary root."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def test_prior_aggregate_semantics_reject_wrong_planned_identity_order(config: dict[str, Any]) -> None:
    """Schema-valid Session 4 rows must still retain their frozen scientific identity order."""
    frames = _synthetic_prior_aggregates(config)
    metrics_path = config["outputs"]["scientific_and_closure"]["locked_score_metrics"]["path"]
    changed = OrderedDict((path, frame.copy(deep=True)) for path, frame in frames.items())
    reordered = changed[metrics_path].iloc[[1, 0, *range(2, len(changed[metrics_path]))]].reset_index(drop=True)
    changed[metrics_path] = reordered
    with pytest.raises(ValueError, match="planned-row order"):
        reporting.validate_prior_aggregate_frames(changed, config)


@pytest.mark.parametrize(
    "mutation",
    [
        "confusion_counts",
        "qc5_cross_output_composition",
        "within_date_parent_child",
        "date_grouped_auc_difference",
        "fold_totals",
        "fold_status_reason",
    ],
)
def test_prior_aggregate_semantic_mutations_fail_closed(
    mutation: str,
    config: dict[str, Any],
) -> None:
    """All six persisted Session 4/5 frames retain their mature scientific invariants."""
    frames = _synthetic_prior_aggregates(config)
    changed = OrderedDict((path, frame.copy(deep=True)) for path, frame in frames.items())
    outputs = config["outputs"]["scientific_and_closure"]
    if mutation == "confusion_counts":
        metrics = changed[outputs["locked_score_metrics"]["path"]]
        index = metrics.index[metrics["metric"].astype(str) == "sensitivity"][0]
        metrics.loc[index, "tp"] = int(metrics.loc[index, "tp"]) + 1
    elif mutation == "qc5_cross_output_composition":
        qc5 = changed[outputs["qc5_correctness"]["path"]]
        selector = (
            (qc5["view"].astype(str) == "oof")
            & (qc5["strict_status"].astype(str) == "strict_pass")
            & (qc5["record_type"].astype(str) == "disease_composition")
        )
        bgl = qc5.index[selector & (qc5["category"].astype(str) == "BGL")][0]
        lgin = qc5.index[selector & (qc5["category"].astype(str) == "LGIN")][0]
        denominator = int(qc5.loc[bgl, "denominator_n"])
        qc5.loc[bgl, "numerator_n"] = int(qc5.loc[bgl, "numerator_n"]) + 1
        qc5.loc[lgin, "numerator_n"] = int(qc5.loc[lgin, "numerator_n"]) - 1
        qc5.loc[bgl, "proportion"] = int(qc5.loc[bgl, "numerator_n"]) / denominator
        qc5.loc[lgin, "proportion"] = int(qc5.loc[lgin, "numerator_n"]) / denominator
    elif mutation == "within_date_parent_child":
        dates = changed[outputs["within_date_concordance"]["path"]]
        dates.loc[0, "n_date_groups_total"] = int(dates.loc[0, "n_date_groups_total"]) + 1
    elif mutation == "date_grouped_auc_difference":
        validation = changed[outputs["date_grouped_validation"]["path"]]
        validation.loc[0, "auc_difference_date_grouped_minus_random"] += 0.01
    elif mutation == "fold_totals":
        folds = changed[outputs["date_grouped_fold_audit"]["path"]]
        folds.loc[0, "n_fit"] = int(folds.loc[0, "n_fit"]) + 1
    else:
        folds = changed[outputs["date_grouped_fold_audit"]["path"]]
        folds["estimability_status"] = folds["estimability_status"].astype("string")
        folds.loc[0, "estimability_status"] = "non_estimable"
    with pytest.raises(ValueError):
        reporting.validate_prior_aggregate_frames(changed, config)


def _completion_receipt(
    config: dict[str, Any],
    session_name: str,
    output_bytes: Mapping[str, bytes],
    access_state: Mapping[str, Mapping[str, bool]],
) -> str:
    """Render an exact prior-session receipt with actual synthetic output hashes."""
    binding = config["outputs"]["completion_logs"]["machine_validated_log_grammar"][
        "session_bindings_exact"
    ][session_name]
    records: list[tuple[str, ...]] = [
        ("header", "posthoc_final_completion_log.v1", "posthoc_final_supplementary", "2030-01-02T03:04:05Z"),
        ("session_identity", session_name, "complete"),
        ("freeze_commit", SESSION0_FREEZE),
    ]
    records.extend(
        ("output_hash", path, hashlib.sha256(output_bytes[path]).hexdigest())
        for path in binding["ordered_output_inventory"]
    )
    records.extend(
        [
            ("deviations_or_nonestimability_summary", "none", "", ""),
            (
                "access_state_session_activity",
                *(str(value).lower() for value in access_state["session_activity"].values()),
            ),
            (
                "access_state_cumulative_through_session",
                *(str(value).lower() for value in access_state["cumulative_through_session"].values()),
            ),
            ("completion_receipt", "2030-01-02T03:04:05Z", "complete"),
        ]
    )
    return "\n".join("\t".join(record) for record in records) + "\n"


def rebind_synthetic_completion_output_hash(
    root: Path,
    config: Mapping[str, Any],
    session_name: str,
    output_path: str,
) -> None:
    """Rebind one synthetic completion receipt to deliberately mutated aggregate bytes."""
    binding = config["outputs"]["completion_logs"]["machine_validated_log_grammar"][
        "session_bindings_exact"
    ][session_name]
    log_path = root / binding["log_path"]
    records = [line.split("\t") for line in log_path.read_text(encoding="utf-8").rstrip("\n").split("\n")]
    replacement = hashlib.sha256((root / output_path).read_bytes()).hexdigest()
    matched = 0
    for record in records:
        if record[:2] == ["output_hash", output_path]:
            record[2] = replacement
            matched += 1
    if matched != 1:
        raise ValueError("Synthetic completion receipt output identity is invalid.")
    log_path.write_text("\n".join("\t".join(record) for record in records) + "\n", encoding="utf-8")


def _synthetic_git_runner(
    root: Path,
    config: dict[str, Any],
) -> Any:
    """Create a read-only Git-evidence adapter for a synthetic authority root."""
    original = config["historical_consolidation"]["original_session_0"]
    amendment = config["historical_consolidation"]["amendment_01"]
    original_blobs = {item["path"]: item["git_blob_id"] for item in original["artifacts"]}
    amendment_blobs = {item["path"]: item["git_blob_id"] for item in amendment["artifacts"]}
    tracked_paths = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
    }
    grammar = config["outputs"]["completion_logs"]["machine_validated_log_grammar"]
    for binding in grammar["session_bindings_exact"].values():
        tracked_paths.update(binding["ordered_output_inventory"])
        tracked_paths.add(binding["log_path"])

    def worktree_blob(relative: str) -> str:
        payload = (root / relative).read_bytes()
        return hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()

    def runner(*args: str) -> str:
        if args[:2] == ("status", "--porcelain"):
            return ""
        if args[:2] == ("cat-file", "-t"):
            return "tag"
        if args[:2] == ("cat-file", "-e") or args[:2] == ("merge-base", "--is-ancestor"):
            return ""
        if args[:3] == ("rev-list", "--parents", "-n"):
            commit = args[-1]
            if commit == SESSION0_FREEZE:
                return f"{commit} {original['sole_parent']}"
            if commit == AMENDMENT_FREEZE:
                return f"{commit} {amendment['sole_parent']}"
        if args[:4] == ("diff-tree", "--no-commit-id", "--name-only", "-r"):
            commit = args[-1]
            paths = original["changed_paths_exact"] if commit == SESSION0_FREEZE else amendment["changed_paths_exact"]
            return "\n".join(paths)
        if args[0] == "rev-parse":
            spec = args[1]
            if spec.startswith("HEAD:"):
                return worktree_blob(spec.removeprefix("HEAD:"))
            if spec.startswith(f"{SESSION0_FREEZE}:"):
                return original_blobs[spec.split(":", 1)[1]]
            if spec.startswith(f"{AMENDMENT_FREEZE}:"):
                return amendment_blobs[spec.split(":", 1)[1]]
        if args[0] == "hash-object":
            return worktree_blob(args[1])
        if args[:2] == ("ls-files", "--error-unmatch"):
            if args[2] not in tracked_paths or not (root / args[2]).is_file():
                raise ValueError("Synthetic tracked prerequisite is missing.")
            return args[2]
        raise AssertionError(f"Unexpected synthetic read-only Git command: {args}")

    return runner


def build_synthetic_session6_root(
    root: Path,
    base_config: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    """Create a complete privacy-safe synthetic future Session 6 authority root."""
    config = copy.deepcopy(base_config)
    source_n = 1040
    included_n = 972
    sample_ids = np.arange(20_000, 21_040, dtype=np.int64)
    included = np.arange(source_n) < included_n
    allowed = config["outputs"]["scientific_and_closure"]["cohort_flow"][
        "validated_single_exclusion_reason_contract"
    ]["allowed_exclusion_reason_keys"]
    excluded_reasons = [allowed[0]] * 62 + list(allowed[1:])
    metadata = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "age": 35.0 + sample_ids % 45,
            "sex(0male1female)": sample_ids % 2,
        }
    )
    sample_index = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "label": sample_ids % 2,
            "disease_category": sample_ids % 6,
            "qc_status": np.where(included, "pass", "fail"),
            "fail_reasons": [""] * included_n + excluded_reasons,
            "fallback_used": np.zeros(source_n, dtype=bool),
            "padded": np.zeros(source_n, dtype=bool),
            "in_final_output": included,
            "filename": [f"synthetic-breath-{value}.raw" for value in sample_ids],
            "collection_time": [f"2030/1/{1 + index % 28}" for index in range(source_n)],
        }
    )
    metadata_path = root / "data/raw/clinical_data/metadata.csv"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(metadata_path, index=False)
    index_path = root / "data/processed/sample_index.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    sample_index.to_csv(index_path, index=False)
    split_path = root / "data/processed/splits/train_test_split.npz"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        split_path,
        train_idx=np.arange(777, dtype=np.int64),
        test_idx=np.arange(777, 972, dtype=np.int64),
    )
    test_ids = sample_ids[777:972]
    predictions = pd.DataFrame(
        {
            "sample_id": test_ids[::-1],
            "y_true": (test_ids[::-1] % 2).astype(np.int64),
            "p_cal": np.repeat(np.linspace(0.05, 0.95, 5), 39),
        }
    )
    prediction_path = root / "outputs/predictions/binary_minirocket_ridge_test_predictions.csv"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(prediction_path, index=False)
    hash_only = {
        "data/processed/qc_reports/qc_summary.csv": b"synthetic-qc-summary\n",
        "data/processed/qc_reports/qc_diagnostics.csv": b"synthetic-qc-diagnostics\n",
        "data/processed/sequences/pipeline_a_raw.npz": b"synthetic-sequence-container",
        "outputs/predictions/binary_minirocket_ridge_nested_cv_oof.csv": b"synthetic-oof-container\n",
        "outputs/analysis/binary_minirocket_ridge_threshold_mapping.json": b"{}\n",
        "outputs/metrics/binary_minirocket_ridge_nested_cv_summary.csv": b"synthetic\n",
        "outputs/metrics/binary_minirocket_ridge_test_metrics.csv": b"synthetic\n",
        "configs/modeling_global.yaml": b"synthetic: true\n",
        "configs/modeling_minirocket_ridge.yaml": b"synthetic: true\n",
        "environment.yml": b"name: gcvocs\n",
        "outputs/models/binary_minirocket_ridge_final.pkl": b"synthetic-model-container",
    }
    for path, payload in hash_only.items():
        _write_bytes(root, path, payload)
    copy_paths = {
        config["outputs"]["session_0"]["methods"],
        config["authority"]["reviewed_plan_path"],
        config["authority"]["agent_guidance_path"],
        *config["outputs"]["future_implementation_paths"],
        *config["hash_map_inventories"]["execution_manifest_code_sha256_keys_exact_order"],
    }
    for relative in copy_paths:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT_ROOT / relative, destination)
    preflight_path = config["outputs"]["scientific_and_closure"]["score_blind_preflight"]["path"]
    _write_bytes(root, preflight_path, (PROJECT_ROOT / preflight_path).read_bytes())
    prior_frames = _synthetic_prior_aggregates(config)
    for path, frame in prior_frames.items():
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False, lineterminator="\n")
    frozen_hashes = OrderedDict(
        (path, hashlib.sha256((root / path).read_bytes()).hexdigest())
        for path in config["frozen_inputs"]["whole_container_sha256"]
    )
    config["frozen_inputs"]["whole_container_sha256"] = dict(frozen_hashes)
    config_path = root / "configs/post_hoc_final_supplementary.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    manifest = json.loads((PROJECT_ROOT / reporting.CURRENT_MANIFEST_RELATIVE_PATH).read_text(encoding="utf-8"))
    manifest["current_runtime_authority"]["config"]["sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    manifest_path = root / reporting.CURRENT_MANIFEST_RELATIVE_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    preflight = json.loads((root / preflight_path).read_text(encoding="utf-8"), object_pairs_hook=OrderedDict)
    cumulative = preflight["access_state"]["cumulative_through_session"]
    session4_activity = OrderedDict(
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
    session4_state = audit.build_access_state(session4_activity, cumulative)
    grammar = config["outputs"]["completion_logs"]["machine_validated_log_grammar"]
    session4_bytes = OrderedDict(
        (path, (root / path).read_bytes())
        for path in grammar["session_bindings_exact"]["locked_scores"]["ordered_output_inventory"]
    )
    _write_bytes(
        root,
        grammar["session_bindings_exact"]["locked_scores"]["log_path"],
        _completion_receipt(config, "locked_scores", session4_bytes, session4_state).encode(),
    )
    session5_activity = OrderedDict(
        (
            key,
            key
            in {
                "prediction_container_bytes_hashed",
                "participant_level_label_values_read",
                "participant_level_score_values_read",
                "participant_level_date_values_read",
                "new_score_derived_results_computed",
                "auxiliary_model_fit_executed",
            },
        )
        for key in ACCESS_STATE_KEYS
    )
    session5_state = audit.build_access_state(
        session5_activity,
        session4_state["cumulative_through_session"],
    )
    session5_bytes = OrderedDict(
        (path, (root / path).read_bytes())
        for path in grammar["session_bindings_exact"]["date_grouped"]["ordered_output_inventory"]
    )
    _write_bytes(
        root,
        grammar["session_bindings_exact"]["date_grouped"]["log_path"],
        _completion_receipt(config, "date_grouped", session5_bytes, session5_state).encode(),
    )
    return config, _synthetic_git_runner(root, config)


def test_reporting_module_import_is_side_effect_free(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing reporting code must not read data, create directories, or write files."""
    before = set(tmp_path.iterdir())
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location("isolated_reporting", Path(reporting.__file__))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(tmp_path.iterdir()) == before


def test_reporting_promotion_inventory_is_exact_nine_outputs_plus_log(config: dict[str, Any]) -> None:
    """The production object model exposes every configured Session 6 destination directly."""
    outputs, log_path = reporting.reporting_promotion_inventory(config)
    binding = config["outputs"]["completion_logs"]["machine_validated_log_grammar"][
        "session_bindings_exact"
    ]["reporting"]
    assert outputs == tuple(binding["ordered_output_inventory"])
    assert len(outputs) == 9
    assert log_path == binding["log_path"]


def test_membership_and_parent_child_reconciliation_succeeds() -> None:
    """Exact source/included/excluded and modeling/train/test identities reconcile."""
    result = reporting.validate_membership_integrity(
        source_ids=_ids(0, 10),
        included_ids=_ids(0, 8),
        excluded_ids=_ids(8, 10),
        train_ids=_ids(0, 6),
        test_ids=_ids(6, 8),
        expected_counts=(10, 8, 2, 6, 2),
    )
    assert result == {"source_n": 10, "included_n": 8, "excluded_n": 2, "train_n": 6, "test_n": 2}


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("duplicate", "unique"),
        ("included_excluded_overlap", "disjoint"),
        ("missing_source_member", "partition"),
        ("train_test_overlap", "disjoint"),
        ("missing_modeling_member", "partition"),
        ("arithmetic", "count"),
    ],
)
def test_membership_mutations_fail_closed(mutation: str, match: str) -> None:
    """Duplicates, overlap, omissions, and arithmetic mutations fail before reporting."""
    kwargs: dict[str, Any] = {
        "source_ids": _ids(0, 10),
        "included_ids": _ids(0, 8),
        "excluded_ids": _ids(8, 10),
        "train_ids": _ids(0, 6),
        "test_ids": _ids(6, 8),
        "expected_counts": (10, 8, 2, 6, 2),
    }
    if mutation == "duplicate":
        kwargs["source_ids"] = np.array([0, 0, *range(2, 10)])
    elif mutation == "included_excluded_overlap":
        kwargs["excluded_ids"] = np.array([7, 9])
    elif mutation == "missing_source_member":
        kwargs["excluded_ids"] = np.array([8])
    elif mutation == "train_test_overlap":
        kwargs["test_ids"] = np.array([5, 7])
    elif mutation == "missing_modeling_member":
        kwargs["test_ids"] = np.array([7])
    else:
        kwargs["expected_counts"] = (11, 8, 2, 6, 2)
    with pytest.raises(ValueError, match=match):
        reporting.validate_membership_integrity(**kwargs)


def test_cohort_flow_uses_exact_frozen_hierarchy_and_arithmetic(config: dict[str, Any]) -> None:
    """The configured 11-node hierarchy proves 1040=68+972 and 972=777+195."""
    exclusion_counts = OrderedDict(
        [
            ("qc1_env_sensor_fault", 66),
            ("qc2_acquisition_truncated", 0),
            ("qc3_unstable_baseline", 0),
            ("qc4_insufficient_response", 1),
            ("qc5_sensor_saturation", 0),
            ("qc6_excessive_spikes", 0),
            ("clip_length_insufficient", 1),
        ]
    )
    frame = reporting.build_cohort_flow(exclusion_counts, config)
    reporting.validate_cohort_flow(frame, config)
    counts = dict(zip(frame["node_id"].astype(str), frame["n"].astype(int), strict=True))
    assert counts["assessed_source_cohort"] == 1040
    assert counts["modeling_cohort"] == 972
    assert counts["train"] + counts["internal_test"] == 972
    assert frame["node_id"].tolist() == [
        item["node_id"] for item in config["outputs"]["scientific_and_closure"]["cohort_flow"]["planned_rows"]
    ]


def test_cohort_flow_arithmetic_and_reason_overlap_mutations_fail(config: dict[str, Any]) -> None:
    """Wrong exclusion totals and semicolon-overlap representations are hard failures."""
    reasons = reporting.parse_single_exclusion_reasons(
        pd.Series(["qc1_env_sensor_fault", "clip_length_insufficient"]), config
    )
    assert reasons.tolist() == ["qc1_env_sensor_fault", "clip_length_insufficient"]
    with pytest.raises(ValueError, match="exactly one"):
        reporting.parse_single_exclusion_reasons(
            pd.Series(["qc1_env_sensor_fault;qc2_acquisition_truncated"]), config
        )
    wrong = OrderedDict(
        (key, 0)
        for key in config["outputs"]["scientific_and_closure"]["cohort_flow"]
        ["validated_single_exclusion_reason_contract"]["allowed_exclusion_reason_keys"]
    )
    with pytest.raises(ValueError, match="68"):
        reporting.build_cohort_flow(wrong, config)


def test_inclusion_loader_requests_only_exact_allowlisted_columns(config: dict[str, Any]) -> None:
    """The loader requests no temporal, location, filename, or score column."""
    calls: list[tuple[str, tuple[str, ...]]] = []

    def reader(path: Path, *, usecols: list[str]) -> pd.DataFrame:
        calls.append((path.name, tuple(usecols)))
        if path.name == "metadata.csv":
            return pd.DataFrame(columns=usecols)
        return pd.DataFrame(columns=usecols)

    reporting.load_inclusion_inputs(Path("metadata.csv"), Path("sample_index.csv"), config, reader=reader)
    contracts = config["loaders"]["per_purpose_ordered_usecols"]
    assert calls == [
        ("metadata.csv", tuple(contracts["inclusion_exclusion_raw_metadata"])),
        ("sample_index.csv", tuple(contracts["inclusion_exclusion_sample_index"])),
    ]
    assert not {"collection_time", "collection_location", "filename", "p_raw", "p_cal"}.intersection(
        column for _, columns in calls for column in columns
    )


def test_inclusion_loader_rejects_extra_or_reordered_columns(config: dict[str, Any]) -> None:
    """A broad reader result or caller-supplied extra projection cannot pass the guard."""
    expected = config["loaders"]["per_purpose_ordered_usecols"]

    def broad_reader(path: Path, *, usecols: list[str]) -> pd.DataFrame:
        return pd.DataFrame(columns=[*usecols, "collection_time"])

    with pytest.raises(ValueError, match="columns"):
        reporting.load_inclusion_inputs(Path("metadata.csv"), Path("sample_index.csv"), config, reader=broad_reader)
    with pytest.raises(ValueError, match="ordered"):
        reporting.load_exact_projection(
            Path("metadata.csv"),
            list(reversed(expected["inclusion_exclusion_raw_metadata"])),
            expected["inclusion_exclusion_raw_metadata"],
            reader=lambda path, *, usecols: pd.DataFrame(columns=usecols),
        )


def test_signed_smd_formulas_ddof_linear_quantiles_and_direction() -> None:
    """Age and binary anchors use ddof=1, linear quantiles, and included-minus-excluded signs."""
    included = np.array([1.0, 2.0, 6.0])
    excluded = np.array([7.0, 8.0, 12.0])
    result = reporting.signed_continuous_smd(included, excluded)
    expected = (included.mean() - excluded.mean()) / math.sqrt(
        (included.var(ddof=1) + excluded.var(ddof=1)) / 2.0
    )
    assert result.estimate == pytest.approx(expected)
    assert result.estimate < 0
    description = reporting.describe_continuous(np.array([0.0, 10.0, 20.0, 30.0, np.nan]))
    assert description["sd"] == pytest.approx(np.std([0.0, 10.0, 20.0, 30.0], ddof=1))
    assert (description["q1"], description["median"], description["q3"]) == pytest.approx((7.5, 15.0, 22.5))
    binary = reporting.signed_binary_smd(np.array([1, 1, 1, 0]), np.array([0, 0, 0, 1]))
    assert binary.estimate is not None and binary.estimate > 0


def test_inclusion_comparison_covers_fixed_focal_rows_and_signed_direction(config: dict[str, Any]) -> None:
    """Age, female, Yes, six categories, and missingness use the frozen focal coding."""
    metadata, sample_index = _inclusion_frames()
    summary, smd = reporting.build_inclusion_comparison(metadata, sample_index, config)
    assert len(summary) == 33
    assert list(zip(smd["variable"].astype(str), smd["level"].astype("string"), strict=True))[:3] == [
        ("age", pd.NA),
        ("sex", "female"),
        ("endpoint", "yes"),
    ]
    assert smd["smd_included_minus_excluded"].iloc[0] < 0
    assert smd.loc[smd["variable"] == "sex", "smd_included_minus_excluded"].iloc[0] > 0
    assert smd.loc[smd["variable"] == "endpoint", "smd_included_minus_excluded"].iloc[0] > 0
    disease = smd.loc[smd["variable"] == "disease_category"]
    assert disease["level"].astype(str).tolist() == ["HC", "BGL", "LGIN", "HGIN", "EGC", "AGC"]
    included_categories = sample_index.loc[sample_index["in_final_output"], "disease_category"]
    excluded_categories = sample_index.loc[~sample_index["in_final_output"], "disease_category"]
    for code, observed in enumerate(disease["smd_included_minus_excluded"].tolist()):
        expected = reporting.signed_binary_smd(
            included_categories.map(lambda value: np.nan if pd.isna(value) else int(int(value) == code)),
            excluded_categories.map(lambda value: np.nan if pd.isna(value) else int(int(value) == code)),
        )
        assert observed == pytest.approx(expected.estimate)
    assert disease["missingness_smd_included_minus_excluded"].notna().tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    expected_missingness = reporting.signed_binary_smd(
        included_categories.isna().astype(int),
        excluded_categories.isna().astype(int),
    )
    assert disease.iloc[0]["missingness_smd_included_minus_excluded"] == pytest.approx(
        expected_missingness.estimate
    )
    reporting.validate_inclusion_comparison(summary, smd, metadata, sample_index, config)


def test_inclusion_comparison_rejects_absolute_and_reversed_smd_mutations(config: dict[str, Any]) -> None:
    """Legacy absolute-SMD or reversed-direction mutations cannot validate."""
    metadata, sample_index = _inclusion_frames()
    summary, smd = reporting.build_inclusion_comparison(metadata, sample_index, config)
    for mutation in ("absolute", "reversed"):
        changed = smd.copy()
        values = changed["smd_included_minus_excluded"].astype("Float64")
        changed["smd_included_minus_excluded"] = values.abs() if mutation == "absolute" else -values
        with pytest.raises(ValueError, match="recompute"):
            reporting.validate_inclusion_comparison(summary, changed, metadata, sample_index, config)
    source = inspect.getsource(reporting)
    assert "_compute_continuous_smd" not in source
    assert "_compute_binary_smd" not in source


def test_binary_validation_sparse_age_and_zero_denominator_contract() -> None:
    """Invalid binary values fail; sparse age and unequal degenerate groups retain fixed statuses."""
    with pytest.raises(ValueError, match="exact"):
        reporting.signed_binary_smd(np.array([0, 2]), np.array([0, 1]))
    sparse = reporting.signed_continuous_smd(np.array([1.0, np.nan]), np.array([2.0, 3.0]))
    assert sparse == reporting.SmdResult(None, "non_estimable", "smd_fewer_than_two_nonmissing")
    identical = reporting.signed_binary_smd(np.zeros(3), np.zeros(2))
    assert identical == reporting.SmdResult(0.0, "estimable", None)
    unequal = reporting.signed_binary_smd(np.zeros(3), np.ones(2))
    assert unequal == reporting.SmdResult(None, "non_estimable", "smd_zero_denominator_nonidentical")
    retained = reporting.describe_continuous(np.array([5.0, np.nan]))
    assert retained["n_nonmissing"] == 1 and retained["mean"] == 5.0 and retained["sd"] is None


def test_calibration_points_qcut_collapse_wilson_and_totals(config: dict[str, Any]) -> None:
    """Collapsed bins remain deterministic and reconcile Wilson/count fields exactly."""
    y_true, p_cal = _calibration_fixture()
    first = reporting.build_calibration_points(y_true, p_cal, config)
    second = reporting.build_calibration_points(y_true, p_cal, config)
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert 1 <= len(first) < 10
    assert int(first["n"].sum()) == 195
    assert int(first["events"].sum()) == int(y_true.sum())
    assert np.array_equal(first["n"].to_numpy(), first["events"].to_numpy() + first["nonevents"].to_numpy())
    low, high = proportion_confint(int(first.iloc[0]["events"]), int(first.iloc[0]["n"]), method="wilson")
    assert first.iloc[0]["observed_ci_low"] == pytest.approx(low)
    assert first.iloc[0]["observed_ci_high"] == pytest.approx(high)
    reporting.validate_calibration_points(first, expected_n=195, expected_events=int(y_true.sum()), config=config)


def test_calibration_points_all_ties_produce_one_retained_bin(config: dict[str, Any]) -> None:
    """All tied p_cal values yield one deterministic, non-visualization-ready bin."""
    labels = np.tile(np.array([0, 1, 0], dtype=np.int64), 65)
    frame = reporting.build_calibration_points(labels, np.full(195, 0.4), config)
    assert len(frame) == 1
    assert frame.iloc[0]["n_actual_bins"] == 1
    assert frame.iloc[0]["visualization_ready"] is np.bool_(False) or not frame.iloc[0]["visualization_ready"]
    assert frame.iloc[0]["p_cal_mean"] == pytest.approx(0.4)


def test_reporting_candidates_enforce_all_schemas_and_cross_artifact_identities(config: dict[str, Any]) -> None:
    """Every closure candidate validates together; schema, count, CI, and support mutations fail."""
    candidates = _full_reporting_candidates(config)
    authority = {"config": config}
    inputs = _full_reporting_inputs(config)
    reporting.validate_reporting_candidates(candidates, authority, inputs)

    mutations: list[dict[str, pd.DataFrame]] = []
    wrong_parent = {key: value.copy() for key, value in candidates.items()}
    wrong_parent["included_excluded_summary"].loc[0, "n_total"] = 971
    mutations.append(wrong_parent)
    wrong_status = {key: value.copy() for key, value in candidates.items()}
    wrong_status["included_excluded_smd"]["estimability_status"] = (
        wrong_status["included_excluded_smd"]["estimability_status"].astype("string")
    )
    wrong_status["included_excluded_smd"].loc[0, "estimability_status"] = "unsupported"
    wrong_status["included_excluded_smd"]["estimability_status"] = wrong_status[
        "included_excluded_smd"
    ]["estimability_status"].astype("category")
    mutations.append(wrong_status)
    wrong_wilson = {key: value.copy() for key, value in candidates.items()}
    wrong_wilson["calibration_points"].loc[0, "observed_ci_low"] = 0.99
    mutations.append(wrong_wilson)
    extrapolated = {key: value.copy() for key, value in candidates.items()}
    outside = ~extrapolated["calibration_smoother"]["in_observed_support"].astype(bool)
    extrapolated["calibration_smoother"].loc[outside.idxmax(), "smoothed_observed_fraction"] = 0.5
    mutations.append(extrapolated)
    wrong_mean = {key: value.copy() for key, value in candidates.items()}
    wrong_mean["included_excluded_summary"].loc[0, "mean"] += 0.25
    mutations.append(wrong_mean)
    wrong_quantile = {key: value.copy() for key, value in candidates.items()}
    wrong_quantile["included_excluded_summary"].loc[0, "q1"] += 0.25
    mutations.append(wrong_quantile)
    wrong_smd = {key: value.copy() for key, value in candidates.items()}
    wrong_smd["included_excluded_smd"].loc[0, "smd_included_minus_excluded"] += 0.25
    mutations.append(wrong_smd)
    wrong_calibration_mean = {key: value.copy() for key, value in candidates.items()}
    wrong_calibration_mean["calibration_points"].loc[0, "p_cal_mean"] += 0.01
    mutations.append(wrong_calibration_mean)
    wrong_lowess = {key: value.copy() for key, value in candidates.items()}
    supported_index = wrong_lowess["calibration_smoother"].index[
        wrong_lowess["calibration_smoother"]["in_observed_support"].astype(bool)
    ][0]
    wrong_lowess["calibration_smoother"].loc[supported_index, "smoothed_observed_fraction"] += 0.01
    mutations.append(wrong_lowess)
    for mutation in mutations:
        with pytest.raises(ValueError):
            reporting.validate_reporting_candidates(mutation, authority, inputs)


def test_reporting_closure_has_exact_eight_pre_manifest_artifacts_and_deterministic_bytes(
    config: dict[str, Any],
) -> None:
    """Two isolated closure constructions agree byte-for-byte before manifest creation."""
    authority = {"config": config}
    inputs = _full_reporting_inputs(config)
    scientific = reporting.calculate_reporting_candidates(authority, inputs)
    access_state = audit.build_reporting_access_state(_ledger := OrderedDict((key, False) for key in ACCESS_STATE_KEYS))
    prior = _synthetic_prior_aggregates(config)
    first = reporting.build_reporting_closure_artifacts(
        authority,
        scientific,
        prior,
        generated_at_utc="2030-01-02T03:04:05Z",
        access_state=access_state,
    )
    second = reporting.build_reporting_closure_artifacts(
        authority,
        reporting.calculate_reporting_candidates(authority, inputs),
        prior,
        generated_at_utc="2030-01-02T03:04:05Z",
        access_state=access_state,
    )
    inventory, _ = reporting.reporting_promotion_inventory(config)
    manifest_path = config["outputs"]["scientific_and_closure"]["execution_manifest"]["path"]
    expected = tuple(path for path in inventory if path != manifest_path)
    assert tuple(first) == expected
    first_bytes = reporting.serialize_reporting_closure_artifacts(first, config)
    second_bytes = reporting.serialize_reporting_closure_artifacts(second, config)
    assert first_bytes == second_bytes
    report_path = config["outputs"]["scientific_and_closure"]["consolidated_analyses_document"]["path"]
    report = first_bytes[report_path].decode("utf-8")
    retained_reasons = {
        item["reason_code"]
        for item in reporting.collect_nonestimability_records(prior, config)
    }
    assert retained_reasons
    assert all(reason in report for reason in retained_reasons)
    assert config["interpretation_boundary"]["endpoint_definition"] in report
    assert "enriched diagnostic case-control/two-gate sampling design" in report
    assert (
        "LGIN findings remain exploratory because biopsy can undergrade disease and complete resection pathology was "
        "unavailable."
    ) in report
    assert "Analyses 1-4 are post-hoc model-robustness analyses." in report
    assert "non-model reporting-closure artifacts, not a fifth model analysis" in report
    assert "Pre-endoscopy triage remains a future validation target" in report
    assert "This study does not directly validate that use." in report
    assert "Session 7 integrity audit is pending" in report
    assert "Visualization release remains unauthorized" in report
    assert "outputs/analysis/posthoc_final_integrity_audit.json" in report
    assert "outputs/analysis/posthoc_final_visualization_release.json" in report


def test_reporting_manifest_and_completion_log_bind_actual_candidate_bytes(config: dict[str, Any]) -> None:
    """Manifest/log hashes are computed from bytes and obey the configured self exclusions."""
    authority = {
        "config": config,
        "session0_freeze": {"freeze_commit_id": SESSION0_FREEZE},
    }
    inputs = _full_reporting_inputs(config)
    scientific = reporting.calculate_reporting_candidates(authority, inputs)
    prior_scope = OrderedDict((key, False) for key in ACCESS_STATE_KEYS)
    access_state = audit.build_reporting_access_state(prior_scope)
    prior_aggregates = _synthetic_prior_aggregates(config)
    artifacts = reporting.build_reporting_closure_artifacts(
        authority,
        scientific,
        prior_aggregates,
        generated_at_utc="2030-01-02T03:04:05Z",
        access_state=access_state,
    )
    artifact_bytes = reporting.serialize_reporting_closure_artifacts(artifacts, config)
    deviations = reporting.collect_nonestimability_records(
        OrderedDict([*prior_aggregates.items(), *artifacts.items()]),
        config,
    )
    retained_reasons = sorted({item["reason_code"] for item in deviations})
    assert retained_reasons
    inventories = config["hash_map_inventories"]
    entry_hashes = OrderedDict(
        (path, hashlib.sha256(f"entry:{path}".encode()).hexdigest())
        for path in dict.fromkeys(
            [
                *inventories["execution_manifest_input_sha256_keys_exact_order"],
                *inventories["execution_manifest_code_sha256_keys_exact_order"],
            ]
        )
    )
    manifest = reporting.build_reporting_execution_manifest(
        authority,
        artifact_bytes,
        entry_hashes,
        command="python scripts/run_posthoc_final_reporting.py --config configs/post_hoc_final_supplementary.yaml",
        generated_at_utc="2030-01-02T03:04:05Z",
        access_state=access_state,
        deviations_or_nonestimability=deviations,
    )
    reporting.validate_reporting_execution_manifest_binding(
        manifest,
        artifact_bytes,
        entry_hashes,
        authority,
        expected_deviations=deviations,
    )
    manifest_path = config["outputs"]["scientific_and_closure"]["execution_manifest"]["path"]
    manifest_bytes = reporting.serialize_json_artifact(manifest)
    nine_bytes = OrderedDict()
    for path in reporting.reporting_promotion_inventory(config)[0]:
        nine_bytes[path] = manifest_bytes if path == manifest_path else artifact_bytes[path]
    log_text = audit.build_session_completion_log(
        session_name="reporting",
        freeze_commit=SESSION0_FREEZE,
        output_bytes=nine_bytes,
        access_state=access_state,
        cumulative_prior=prior_scope,
        generated_at_utc="2030-01-02T03:04:05Z",
        config=config,
        reason_codes=retained_reasons,
    )
    audit.validate_completion_log_against_bytes(
        log_text,
        session_name="reporting",
        freeze_commit=SESSION0_FREEZE,
        output_bytes=nine_bytes,
        access_state=access_state,
        cumulative_prior=prior_scope,
        config=config,
    )
    assert manifest_path not in manifest["output_sha256"]
    assert config["outputs"]["completion_logs"]["reporting"] not in manifest["output_sha256"]
    assert all(
        any(
            item["status"] == "non_estimable_retained" and item["reason_code"] == reason
            for item in manifest["deviations_or_nonestimability"]
        )
        for reason in retained_reasons
    )
    assert all(f"\tpresent_retained\t{reason}\t" in log_text for reason in retained_reasons)
    mutated = OrderedDict(nine_bytes)
    mutated[next(iter(mutated))] += b"mutation"
    with pytest.raises(ValueError, match="hash"):
        audit.validate_completion_log_against_bytes(
            log_text,
            session_name="reporting",
            freeze_commit=SESSION0_FREEZE,
            output_bytes=mutated,
            access_state=access_state,
            cumulative_prior=prior_scope,
            config=config,
        )


def test_lowess_exact_grid_tie_invariance_support_mask_and_descriptive_role(config: dict[str, Any]) -> None:
    """The fixed 101-grid smoother is tie-stable, masked, and explicitly descriptive."""
    y_true, p_cal = _calibration_fixture()
    first = reporting.build_calibration_smoother(y_true, p_cal, config)
    order = np.arange(195)
    for value in np.unique(p_cal):
        group = order[p_cal == value]
        order[group] = group[::-1]
    second = reporting.build_calibration_smoother(y_true[order], p_cal[order], config)
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert len(first) == 101
    assert first["grid_index"].astype(int).tolist() == list(range(101))
    assert first["p_cal_grid"].to_numpy(dtype=float) == pytest.approx(np.linspace(0.0, 1.0, 101))
    outside = ~first["in_observed_support"].astype(bool)
    assert first.loc[outside, "smoothed_observed_fraction"].isna().all()
    assert set(first.loc[outside, "estimability_status"].astype(str)) == {"not_evaluated"}
    assert set(first.loc[outside, "nonestimable_reason"].astype(str)) == {"outside_observed_support"}
    assert not first.loc[outside, "boundary_clipped"].astype(bool).any()
    assert "recalibr" not in reporting.LOWESS_ROLE.lower()
    reporting.validate_calibration_smoother(first, config)


def test_lowess_clipping_and_global_nonestimability(config: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Supported out-of-bound fits clip; fixed global reasons retain all 101 missing rows."""
    y_true, p_cal = _calibration_fixture()

    def out_of_bounds(**kwargs: Any) -> np.ndarray:
        return np.linspace(-0.2, 1.2, len(kwargs["xvals"]))

    monkeypatch.setattr(reporting, "lowess", out_of_bounds)
    clipped = reporting.build_calibration_smoother(y_true, p_cal, config)
    supported = clipped["in_observed_support"].astype(bool)
    assert clipped.loc[supported, "smoothed_observed_fraction"].between(0.0, 1.0).all()
    assert clipped.loc[supported, "boundary_clipped"].astype(bool).any()

    one_class = reporting.build_calibration_smoother(np.zeros(195, dtype=int), p_cal, config)
    assert set(one_class["estimability_status"].astype(str)) == {"non_estimable"}
    assert set(one_class["nonestimable_reason"].astype(str)) == {"all_one_outcome_class"}
    assert one_class["smoothed_observed_fraction"].isna().all()
    one_score = reporting.build_calibration_smoother(y_true, np.full(195, 0.5), config)
    assert set(one_score["nonestimable_reason"].astype(str)) == {"fewer_than_two_distinct_scores"}


def test_lowess_invocation_uses_every_frozen_parameter(config: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """The installed smoother is called once with the exact fixed grid and parameter set."""
    y_true, p_cal = _calibration_fixture()
    calls: list[dict[str, Any]] = []

    def capture(**kwargs: Any) -> np.ndarray:
        calls.append(kwargs)
        return np.linspace(0.0, 1.0, 101)

    monkeypatch.setattr(reporting, "lowess", capture)
    reporting.build_calibration_smoother(y_true, p_cal, config)
    assert len(calls) == 1
    call = calls[0]
    assert call["frac"] == pytest.approx(2.0 / 3.0)
    assert call["it"] == 0
    assert call["delta"] == 0.0
    assert call["is_sorted"] is True
    assert call["missing"] == "none"
    assert call["return_sorted"] is False
    assert call["xvals"] == pytest.approx(np.linspace(0.0, 1.0, 101))


def test_reporting_loader_resolves_actual_frozen_path_map(
    tmp_path: Path,
    config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The future loader uses real path keys from the production configuration without broad reads."""
    calls: list[tuple[str, ...]] = []
    usecols = config["loaders"]["per_purpose_ordered_usecols"]
    sample_index = pd.DataFrame(columns=usecols["inclusion_exclusion_sample_index"])
    metadata = pd.DataFrame(columns=usecols["inclusion_exclusion_raw_metadata"])
    predictions = pd.DataFrame(columns=config["loaders"]["per_purpose_ordered_usecols"]["calibration_loader_test_only"])

    def inclusion_loader(
        metadata_path: Path,
        index_path: Path,
        frozen: dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        calls.append((str(metadata_path.relative_to(tmp_path)), str(index_path.relative_to(tmp_path))))
        return metadata, sample_index

    def projection_loader(path: Path, usecols: list[str], expected: list[str]) -> pd.DataFrame:
        calls.append((str(path.relative_to(tmp_path)), *usecols))
        return predictions

    class Archive:
        def __enter__(self) -> dict[str, np.ndarray]:
            return {"train_idx": np.array([0]), "test_idx": np.array([1])}

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(reporting, "load_inclusion_inputs", inclusion_loader)
    monkeypatch.setattr(reporting, "load_exact_projection", projection_loader)
    monkeypatch.setattr(
        reporting.np,
        "load",
        lambda path, allow_pickle: calls.append((str(Path(path).relative_to(tmp_path)),)) or Archive(),
    )
    result = reporting.load_reporting_execution_inputs({"config": config, "project_root": tmp_path})
    assert calls[0] == (
        "data/raw/clinical_data/metadata.csv",
        "data/processed/sample_index.csv",
    )
    assert calls[1][0] == "outputs/predictions/binary_minirocket_ridge_test_predictions.csv"
    assert calls[2] == ("data/processed/splits/train_test_split.npz",)
    assert result["train_idx"].tolist() == [0]


def test_reporting_alignment_reindexes_exact_test_ids_and_binds_labels(config: dict[str, Any]) -> None:
    """Calibration scores are explicitly reordered to the locked test membership and frozen labels."""
    inputs = _alignment_inputs()
    evidence = reporting.align_reporting_execution_inputs(inputs, config)
    expected_ids = inputs["sample_index"].iloc[inputs["test_idx"]]["sample_id"].tolist()
    assert evidence.test_ids.tolist() == expected_ids
    assert evidence.calibration["sample_id"].tolist() == expected_ids
    assert evidence.calibration["y_true"].tolist() == (np.asarray(expected_ids) % 2).tolist()
    assert evidence.expected_test_n == config["reporting_closure"]["cohort_flow"]["internal_test_n"]


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "label", "index_order"])
def test_reporting_alignment_mutations_fail_closed(mutation: str, config: dict[str, Any]) -> None:
    """ID population, uniqueness, split order, and label equality are hard pre-computation gates."""
    inputs = _alignment_inputs()
    predictions = inputs["test_predictions"].copy()
    if mutation == "missing":
        predictions = predictions.iloc[1:].copy()
    elif mutation == "extra":
        predictions.loc[len(predictions)] = [999_999, 1, 0.5]
    elif mutation == "duplicate":
        predictions.loc[1, "sample_id"] = predictions.loc[0, "sample_id"]
    elif mutation == "label":
        predictions.loc[0, "y_true"] = 1 - int(predictions.loc[0, "y_true"])
    else:
        inputs["test_idx"] = np.asarray(inputs["test_idx"])[::-1]
    inputs["test_predictions"] = predictions
    with pytest.raises(ValueError):
        reporting.align_reporting_execution_inputs(inputs, config)


@pytest.mark.parametrize(
    ("labels", "scores", "match"),
    [
        (np.array([0, 1]), np.array([0.1]), "195"),
        (np.r_[np.zeros(194), 2], np.linspace(0.0, 1.0, 195), "exact"),
        (np.tile([0, 1, 0], 65), np.r_[np.linspace(0.0, 1.0, 194), np.nan], "finite"),
        (np.tile([0, 1, 0], 65), np.r_[np.linspace(0.0, 1.0, 194), 1.1], r"\[0, 1\]"),
    ],
)
def test_calibration_and_lowess_hard_input_failures(
    labels: np.ndarray,
    scores: np.ndarray,
    match: str,
    config: dict[str, Any],
) -> None:
    """Length, binary, nonfinite, and probability-range violations are hard failures."""
    for function in (reporting.build_calibration_points, reporting.build_calibration_smoother):
        with pytest.raises(ValueError, match=match):
            function(labels, scores, config)


def test_fact_audit_preserves_exact_statuses_and_rejects_upgrade(config: dict[str, Any]) -> None:
    """The 13 frozen conservative records cannot be upgraded or reordered."""
    payload = reporting.build_reference_standard_timing_audit(
        config,
        generated_at_utc="2030-01-02T03:04:05Z",
        access_state=_all_false_access_state(),
    )
    reporting.validate_reference_standard_timing_audit(payload, config)
    assert [item["fact_id"] for item in payload["facts"]] == [f"RS{index:02d}" for index in range(1, 14)]
    assert {item["evidence_status"] for item in payload["facts"]} == {
        "repository_supported",
        "investigator_attested",
        "not_systematically_recorded",
        "prohibited_claim",
    }
    changed = OrderedDict(payload)
    changed["facts"] = [OrderedDict(item) for item in payload["facts"]]
    changed["facts"][0]["evidence_status"] = "repository_supported"
    with pytest.raises(ValueError, match="frozen fact"):
        reporting.validate_reference_standard_timing_audit(changed, config)


def test_qc6_correction_is_deterministic_complete_and_write_free(config: dict[str, Any]) -> None:
    """The additive note text carries every fixed lineage fact and cannot edit legacy sources."""
    timestamp = "2030-01-02T03:04:05Z"
    first = reporting.generate_qc6_lineage_correction(config, correction_date_utc=timestamp)
    second = reporting.generate_qc6_lineage_correction(config, correction_date_utc=timestamp)
    assert first == second
    for required in (
        "any-channel",
        "any(spike_counts > configured_threshold)",
        "595d7db1525560e5e2e61936a222a71abbfe2762",
        "211.0",
        "421254714d7af7ba4466e1424bb443f26270d464",
        "equality",
        "protocol",
        "six-channel",
        "zero",
        "no cohort or model impact",
        "future Methods",
    ):
        assert required.lower() in first.lower()
    reporting.validate_qc6_lineage_correction(first, config, correction_date_utc=timestamp)
    assert tuple(inspect.signature(reporting.generate_qc6_lineage_correction).parameters) == (
        "config",
        "correction_date_utc",
    )
    assert "write_text" not in inspect.getsource(reporting.generate_qc6_lineage_correction)


def test_reporting_cli_parser_keeps_dual_freeze_and_explicit_authorization() -> None:
    """The actual reporting parser requires both historical commits and execute approval."""
    from scripts import run_posthoc_final_reporting as cli

    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--config",
                "configs/post_hoc_final_supplementary.yaml",
                "--freeze-commit",
                SESSION0_FREEZE,
                "--amendment-freeze-commit",
                AMENDMENT_FREEZE,
            ]
        )
    args = parser.parse_args(
        [
            "--config",
            "configs/post_hoc_final_supplementary.yaml",
            "--freeze-commit",
            SESSION0_FREEZE,
            "--amendment-freeze-commit",
            AMENDMENT_FREEZE,
            "--execute-approved",
        ]
    )
    assert args.config == Path("configs/post_hoc_final_supplementary.yaml")
    assert args.freeze_commit == SESSION0_FREEZE
    assert args.amendment_freeze_commit == AMENDMENT_FREEZE


def test_reporting_real_cli_executes_full_synthetic_chain(
    tmp_path: Path,
    config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The actual parser/main crosses authority, calculation, readback, privacy, and promotion."""
    from scripts import run_posthoc_final_reporting as cli
    from src.analysis import posthoc_final_locked_scores as locked_scores

    real_target = PROJECT_ROOT / "outputs/analysis/posthoc_final_cohort_flow.csv"
    real_target_existed_before = real_target.exists()
    real_target_bytes_before = real_target.read_bytes() if real_target_existed_before else None
    synthetic_config, git_runner = build_synthetic_session6_root(tmp_path, config)
    monkeypatch.setattr(locked_scores, "PROJECT_ROOT", tmp_path)
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file()}
    monkeypatch.chdir(tmp_path)
    exit_code = cli.main(
        [
            "--config",
            "configs/post_hoc_final_supplementary.yaml",
            "--freeze-commit",
            SESSION0_FREEZE,
            "--amendment-freeze-commit",
            AMENDMENT_FREEZE,
            "--execute-approved",
        ],
        project_root=tmp_path,
        git_runner=git_runner,
        invocation_clock=lambda: "2030-01-02T03:04:05Z",
    )
    assert exit_code == 0
    inventory, log_path = reporting.reporting_promotion_inventory(synthetic_config)
    after = {path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file()}
    assert after - before == {Path(path) for path in (*inventory, log_path)}
    assert all((tmp_path / path).is_file() for path in (*inventory, log_path))
    manifest = json.loads(
        (tmp_path / synthetic_config["outputs"]["scientific_and_closure"]["execution_manifest"]["path"])
        .read_text(encoding="utf-8")
    )
    assert manifest["deterministic_comparison_provenance"] == {
        "candidate_pair_count": 2,
        "scientific_payload_agreement": True,
        "shared_invocation_utc": "2030-01-02T03:04:05Z",
    }
    assert real_target.exists() == real_target_existed_before
    if real_target_existed_before:
        assert real_target.read_bytes() == real_target_bytes_before


def test_reporting_real_cli_rejects_schema_shaped_semantically_invalid_prior_aggregate(
    tmp_path: Path,
    config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Session 6 parser/main rejects a hash-bound but fold-incoherent Session 5 CSV."""
    from scripts import run_posthoc_final_reporting as cli
    from src.analysis import posthoc_final_locked_scores as locked_scores

    synthetic_config, git_runner = build_synthetic_session6_root(tmp_path, config)
    monkeypatch.setattr(locked_scores, "PROJECT_ROOT", tmp_path)
    fold_path = synthetic_config["outputs"]["scientific_and_closure"]["date_grouped_fold_audit"]["path"]
    fold_file = tmp_path / fold_path
    folds = pd.read_csv(fold_file)
    folds.loc[0, "n_fit"] = int(folds.loc[0, "n_fit"]) + 1
    folds.to_csv(fold_file, index=False, lineterminator="\n")
    rebind_synthetic_completion_output_hash(tmp_path, synthetic_config, "date_grouped", fold_path)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="fold fit/validation totals"):
        cli.main(
            [
                "--config",
                "configs/post_hoc_final_supplementary.yaml",
                "--freeze-commit",
                SESSION0_FREEZE,
                "--amendment-freeze-commit",
                AMENDMENT_FREEZE,
                "--execute-approved",
            ],
            project_root=tmp_path,
            git_runner=git_runner,
            invocation_clock=lambda: "2030-01-02T03:04:05Z",
        )
    inventory, log_path = reporting.reporting_promotion_inventory(synthetic_config)
    assert not any((tmp_path / path).exists() for path in (*inventory, log_path))


@pytest.mark.parametrize("failure", ["authority", "stale", "frozen_input"])
def test_reporting_real_cli_preconditions_fail_before_participant_loader(
    failure: str,
    tmp_path: Path,
    config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real authority, stale, and entry-hash failures precede the participant-capable loader."""
    from scripts import run_posthoc_final_reporting as cli
    from src.analysis import posthoc_final_locked_scores as locked_scores

    synthetic_config, git_runner = build_synthetic_session6_root(tmp_path, config)
    monkeypatch.setattr(locked_scores, "PROJECT_ROOT", tmp_path)
    touched = {"privacy_loader": False}
    real_privacy_loader = reporting.load_privacy_references

    def forbidden_loader(*args: Any, **kwargs: Any) -> Any:
        touched["privacy_loader"] = True
        return real_privacy_loader(*args, **kwargs)

    monkeypatch.setattr(reporting, "load_privacy_references", forbidden_loader)
    if failure == "authority":
        config_path = tmp_path / "configs/post_hoc_final_supplementary.yaml"
        config_path.write_bytes(config_path.read_bytes() + b"\n")
    elif failure == "stale":
        first_target = reporting.reporting_promotion_inventory(synthetic_config)[0][0]
        _write_bytes(tmp_path, first_target, b"stale")
    else:
        frozen_path = next(iter(synthetic_config["frozen_inputs"]["whole_container_sha256"]))
        (tmp_path / frozen_path).write_bytes(b"mutated-before-entry")
    argv = [
        "--config",
        "configs/post_hoc_final_supplementary.yaml",
        "--freeze-commit",
        SESSION0_FREEZE,
        "--amendment-freeze-commit",
        AMENDMENT_FREEZE,
    ]
    with pytest.raises(ValueError):
        cli.main(
            [*argv, "--execute-approved"],
            project_root=tmp_path,
            git_runner=git_runner,
            invocation_clock=lambda: "2030-01-02T03:04:05Z",
        )
    assert touched == {"privacy_loader": False}


def test_reporting_detects_protected_input_mutation_between_entry_and_exit(
    tmp_path: Path,
    config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A protected source changed after authority capture cannot reach Session 6 promotion."""
    from src.analysis import posthoc_final_locked_scores as locked_scores

    synthetic_config, git_runner = build_synthetic_session6_root(tmp_path, config)
    monkeypatch.setattr(locked_scores, "PROJECT_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    authority = reporting.validate_reporting_authority(
        Path("configs/post_hoc_final_supplementary.yaml"),
        SESSION0_FREEZE,
        AMENDMENT_FREEZE,
        git_runner=git_runner,
        project_root=tmp_path,
    )
    references = audit.load_privacy_references(
        tmp_path / synthetic_config["privacy"]["privacy_reference_loader"]["source"],
        synthetic_config,
        authorized=True,
    )
    inputs = reporting.load_reporting_execution_inputs(authority)
    prior_aggregates = reporting.load_validated_prior_aggregates(authority)
    protected = next(iter(synthetic_config["frozen_inputs"]["whole_container_sha256"]))
    (tmp_path / protected).write_bytes(b"synthetic-exit-mutation")
    with pytest.raises(ValueError, match="entry and exit"):
        reporting.prepare_reporting_promotion(
            authority=authority,
            inputs=inputs,
            prior_aggregates=prior_aggregates,
            privacy_references=references,
            generated_at_utc="2030-01-02T03:04:05Z",
            command="python scripts/run_posthoc_final_reporting.py --synthetic-fixture",
            git_runner=git_runner,
        )
    outputs, log_path = reporting.reporting_promotion_inventory(synthetic_config)
    assert not any((tmp_path / path).exists() for path in (*outputs, log_path))


def test_isolated_reporting_candidate_byte_mismatch_fails() -> None:
    """A deterministic dual-reporting mismatch cannot reach manifest construction."""
    first = OrderedDict([("outputs/one.csv", b"one\n")])
    second = OrderedDict([("outputs/one.csv", b"two\n")])
    with pytest.raises(ValueError, match="candidate bytes disagree"):
        reporting.require_byte_identical_candidate_sets(first, second)
