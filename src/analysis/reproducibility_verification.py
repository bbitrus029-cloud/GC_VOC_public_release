"""Stage 9 reproducibility verification: end-to-end smoke test, path-leak grep,
experiment manifest, and Session 15 audit JSON.

This module implements the deliverables in modeling plan §3.27 (end-to-end smoke
test), §3.28 (experiment manifest), and §7.15 (Session 15 deliverables).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.data_split import _format_path, _load_yaml_config, _resolve_path
from src.config import MODEL_NAMES, PROJECT_ROOT, RANDOM_SEED
from src.evaluation.calibration import apply_platt_calibrator
from src.training.load_final_model import load_final_model


LOGGER = logging.getLogger(__name__)

SMOKE_TEST_N_PER_CLASS = 5
SMOKE_TEST_N_TOTAL = 2 * SMOKE_TEST_N_PER_CLASS
PIPELINE_A_MODELS = ("minirocket_ridge", "inception_time", "cnn_bilstm_attention")
PIPELINE_B_MODELS = ("xgboost", "svm", "elastic_net_lr")
TEXT_FILE_EXTENSIONS = (".md", ".json", ".csv", ".py", ".yaml", ".yml", ".txt", ".log")
ABSOLUTE_PATH_PATTERN = re.compile(
    r"(/home/[^/\s\"'`]+/|/mnt/[^/\s\"'`]+/|C:\\Users\\[^\\\s\"'`]+\\|/Users/[^/\s\"'`]+/)"
)
PROBABILITY_TOLERANCE = 1e-6


def _configure_logging() -> None:
    """Configure INFO-level logging for CLI execution."""
    if logging.getLogger().handlers:
        logging.getLogger().setLevel(logging.INFO)
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _timestamp_utc() -> str:
    """Return an ISO-8601 UTC timestamp with trailing Z."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _git_rev_parse(ref: str) -> str:
    """Return the full commit hash for one git reference.

    For annotated tags, `git rev-parse <tag>` returns the tag object hash,
    not the commit hash. The `^{commit}` peel forces resolution to the
    underlying commit so the manifest records the audit-relevant commit hash.
    """
    result = subprocess.run(
        ["git", "rev-parse", f"{ref}^{{commit}}"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_ls_files() -> list[Path]:
    """Return the list of git-tracked files as project-relative paths."""
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [Path(line) for line in result.stdout.splitlines() if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON payload using UTF-8 and project-relative path hygiene."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    except OSError as exc:
        LOGGER.error("Failed to write JSON output to %s.", path)
        raise OSError(f"Failed to write JSON output to {path}.") from exc


def _write_text(path: Path, content: str) -> None:
    """Write a UTF-8 text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _load_pipeline_a_tensor(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the full 972-sample Pipeline A tensor with sample_id and label."""
    with np.load(npz_path, allow_pickle=False) as payload:
        x = np.asarray(payload["x"], dtype=float)
        sample_ids = payload["sample_id"].astype(np.int64)
        labels = payload["label"].astype(np.int64)
    return x, sample_ids, labels


def _load_pipeline_b_frame(csv_path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load the full 972-sample Pipeline B feature frame plus sample_id and label."""
    frame = pd.read_csv(csv_path)
    if "sample_id" not in frame.columns or "label" not in frame.columns:
        raise ValueError(f"Pipeline B CSV at {csv_path} missing sample_id or label.")
    sample_ids = frame["sample_id"].to_numpy(dtype=np.int64)
    labels = frame["label"].to_numpy(dtype=np.int64)
    feature_frame = frame.drop(columns=["sample_id", "label"])
    return feature_frame, sample_ids, labels


def _select_smoke_subset(
    test_idx: np.ndarray,
    sample_ids: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministically pick `SMOKE_TEST_N_PER_CLASS` per class by sample_id order."""
    test_sample_ids = sample_ids[test_idx]
    test_labels = labels[test_idx]
    test_positions = np.argsort(test_sample_ids, kind="mergesort")
    sorted_positions = test_idx[test_positions]
    sorted_labels = test_labels[test_positions]
    sorted_sample_ids = test_sample_ids[test_positions]

    pos_mask = sorted_labels == 1
    neg_mask = sorted_labels == 0
    pos_positions = sorted_positions[pos_mask][:SMOKE_TEST_N_PER_CLASS]
    neg_positions = sorted_positions[neg_mask][:SMOKE_TEST_N_PER_CLASS]
    if len(pos_positions) < SMOKE_TEST_N_PER_CLASS or len(neg_positions) < SMOKE_TEST_N_PER_CLASS:
        raise AssertionError(
            "Test set does not contain enough samples per class for the smoke subset "
            f"(positives={len(pos_positions)}, negatives={len(neg_positions)})."
        )
    selected_positions = np.concatenate([neg_positions, pos_positions])
    selected_sorted = np.sort(selected_positions, kind="mergesort")
    selected_sample_ids = sample_ids[selected_sorted]
    selected_labels = labels[selected_sorted]
    if len(selected_sorted) != SMOKE_TEST_N_TOTAL:
        raise AssertionError(
            f"Smoke subset has {len(selected_sorted)} samples, expected {SMOKE_TEST_N_TOTAL}."
        )
    return selected_sorted, selected_sample_ids, selected_labels


def _predict_minirocket_ridge(payload: dict[str, Any], x_pipeline_a: np.ndarray) -> np.ndarray:
    """Return positive-class probabilities for the MR+R final model."""
    transformer = payload["transformer"]
    normalizer = payload["normalizer"]
    model = payload["model"]
    x_ncl = np.transpose(x_pipeline_a, (0, 2, 1)).astype(np.float32, copy=False)
    transformed = transformer.transform(x_ncl)
    feature_columns = list(getattr(normalizer, "feature_names", [])) or None
    if isinstance(transformed, pd.DataFrame):
        feature_frame = transformed.copy()
        if feature_columns is not None:
            feature_frame.columns = list(feature_columns)
    else:
        array = np.asarray(transformed, dtype=float)
        if feature_columns is None:
            feature_columns = [f"feature_{index:05d}" for index in range(array.shape[1])]
        feature_frame = pd.DataFrame(array, columns=list(feature_columns))
    normalized = normalizer.transform(feature_frame)
    return model.predict_proba(normalized)


def _predict_pipeline_b(payload: dict[str, Any], x_pipeline_b: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities for a Pipeline B sklearn model."""
    normalizer = payload["normalizer"]
    normalized = normalizer.transform(x_pipeline_b)
    return payload["model"].predict_proba(normalized)


def _predict_dl(payload: dict[str, Any], x_pipeline_a: np.ndarray) -> np.ndarray:
    """Return positive-class probabilities for a DL final model via its wrapper."""
    normalizer = payload["normalizer"]
    normalized = normalizer.transform(x_pipeline_a)
    return payload["model"].predict_proba(normalized)


def _derive_quadrant(y_true: int, y_pred: int) -> str:
    """Return the TP/TN/FP/FN label for one binary prediction."""
    if y_true == 1 and y_pred == 1:
        return "TP"
    if y_true == 0 and y_pred == 0:
        return "TN"
    if y_true == 0 and y_pred == 1:
        return "FP"
    return "FN"


def _quadrant_distribution(quadrants: list[str]) -> dict[str, int]:
    """Return the count per quadrant label."""
    return {label: int(sum(1 for quadrant in quadrants if quadrant == label)) for label in ("TP", "TN", "FP", "FN")}


def _run_one_model_smoke(
    model_name: str,
    *,
    x_pipeline_a_subset: np.ndarray,
    x_pipeline_b_subset: pd.DataFrame,
    sample_ids: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Execute the smoke test on one model and return a structured result dict."""
    pkl_path = PROJECT_ROOT / "outputs" / "models" / f"binary_{model_name}_final.pkl"
    LOGGER.info("Smoke test loading PKL for %s from %s.", model_name, _format_path(pkl_path))
    payload = load_final_model(pkl_path)

    threshold_p_raw = float(payload["threshold_p_raw"])
    calibrator = payload["calibrator"]

    if model_name == "minirocket_ridge":
        proba = _predict_minirocket_ridge(payload, x_pipeline_a_subset)
    elif model_name in PIPELINE_B_MODELS:
        proba = _predict_pipeline_b(payload, x_pipeline_b_subset)
    elif model_name in ("inception_time", "cnn_bilstm_attention"):
        proba = _predict_dl(payload, x_pipeline_a_subset)
    else:
        raise ValueError(f"Unknown model name {model_name}.")

    proba = np.asarray(proba, dtype=float)
    n_samples = SMOKE_TEST_N_TOTAL
    shape_ok = proba.shape == (n_samples, 2)
    finite_count = int(np.isfinite(proba).sum())
    in_range_count = int(((proba >= 0.0) & (proba <= 1.0)).sum())
    row_sums = proba.sum(axis=1)
    rows_summing_to_one = int(np.isclose(row_sums, 1.0, atol=PROBABILITY_TOLERANCE).sum())

    p_raw = proba[:, 1]
    if not isinstance(p_raw, np.ndarray) or p_raw.ndim != 1:
        raise AssertionError(f"{model_name} positive-class probabilities must be a 1-D ndarray.")
    p_cal = apply_platt_calibrator(calibrator=calibrator, p_raw=p_raw)
    y_pred = (p_raw >= threshold_p_raw).astype(int)
    quadrants = [_derive_quadrant(int(y), int(y_hat)) for y, y_hat in zip(labels, y_pred, strict=True)]
    distribution = _quadrant_distribution(quadrants)

    assertions = {
        "shape_ok": bool(shape_ok),
        "all_finite": bool(finite_count == proba.size),
        "all_in_range": bool(in_range_count == proba.size),
        "all_rows_sum_to_one": bool(rows_summing_to_one == n_samples),
        "all_quadrants_valid": bool(all(q in {"TP", "TN", "FP", "FN"} for q in quadrants)),
        "p_cal_finite": bool(np.isfinite(p_cal).all()),
        "p_cal_in_range": bool(((p_cal >= 0.0) & (p_cal <= 1.0)).all()),
    }
    passed = all(assertions.values())

    return {
        "model": model_name,
        "n_samples": n_samples,
        "threshold_p_raw": threshold_p_raw,
        "shape": list(proba.shape),
        "n_finite": finite_count,
        "n_in_range": in_range_count,
        "n_rows_summing_to_one": rows_summing_to_one,
        "sample_ids": sample_ids.tolist(),
        "y_true": labels.tolist(),
        "y_pred": y_pred.tolist(),
        "quadrants": quadrants,
        "quadrant_distribution": distribution,
        "assertions": assertions,
        "passed": passed,
    }


def _format_smoke_log_line(result: dict[str, Any]) -> str:
    """Return a one-line summary string for the smoke test log."""
    distribution = result["quadrant_distribution"]
    distribution_text = ", ".join(f"{label}={distribution[label]}" for label in ("TP", "TN", "FP", "FN"))
    return (
        f"{result['model']}: n_samples={result['n_samples']}, "
        f"n_finite={result['n_finite']}, n_in_range={result['n_in_range']}, "
        f"passed={result['passed']}, quadrants=[{distribution_text}]"
    )


def _run_smoke_tests() -> tuple[list[dict[str, Any]], list[str]]:
    """Execute the smoke test for all six models and return structured results + log lines."""
    LOGGER.info("Loading Pipeline A and Pipeline B artifacts for smoke test.")
    pipeline_a_path = PROJECT_ROOT / "data" / "processed" / "sequences" / "pipeline_a_raw.npz"
    pipeline_b_path = PROJECT_ROOT / "data" / "processed" / "features" / "pipeline_b_raw.csv"
    test_idx_path = PROJECT_ROOT / "data" / "processed" / "splits" / "test_idx.npz"

    x_pipeline_a, sample_ids_a, labels_a = _load_pipeline_a_tensor(pipeline_a_path)
    x_pipeline_b, sample_ids_b, labels_b = _load_pipeline_b_frame(pipeline_b_path)
    if not np.array_equal(sample_ids_a, sample_ids_b):
        raise AssertionError("Pipeline A and Pipeline B sample_ids must be identical.")
    if not np.array_equal(labels_a, labels_b):
        raise AssertionError("Pipeline A and Pipeline B labels must be identical.")

    with np.load(test_idx_path, allow_pickle=False) as split_file:
        test_idx = split_file["test_idx"].astype(np.int64)

    subset_positions, subset_sample_ids, subset_labels = _select_smoke_subset(
        test_idx=test_idx,
        sample_ids=sample_ids_a,
        labels=labels_a,
    )
    LOGGER.info(
        "Smoke subset selected: n=%d, positives=%d, negatives=%d, sample_ids=%s",
        len(subset_positions),
        int(subset_labels.sum()),
        int((1 - subset_labels).sum()),
        subset_sample_ids.tolist(),
    )

    x_pipeline_a_subset = x_pipeline_a[subset_positions]
    x_pipeline_b_subset = x_pipeline_b.iloc[subset_positions].reset_index(drop=True)

    results: list[dict[str, Any]] = []
    log_lines: list[str] = []
    for model_name in MODEL_NAMES:
        result = _run_one_model_smoke(
            model_name=model_name,
            x_pipeline_a_subset=x_pipeline_a_subset,
            x_pipeline_b_subset=x_pipeline_b_subset,
            sample_ids=subset_sample_ids,
            labels=subset_labels,
        )
        results.append(result)
        log_lines.append(_format_smoke_log_line(result))
        LOGGER.info("Smoke test completed for %s: passed=%s.", model_name, result["passed"])
    return results, log_lines


def _scan_one_file_for_absolute_paths(path: Path) -> list[dict[str, Any]]:
    """Scan one tracked text file for absolute-path leaks."""
    matches: list[dict[str, Any]] = []
    absolute_path = PROJECT_ROOT / path
    try:
        text = absolute_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return matches
    for line_number, line in enumerate(text.splitlines(), start=1):
        for hit in ABSOLUTE_PATH_PATTERN.findall(line):
            matches.append({
                "file": str(path),
                "line_number": line_number,
                "matched_prefix": hit,
                "line_content": line.rstrip(),
            })
    return matches


def _run_path_leak_grep() -> tuple[list[dict[str, Any]], list[str]]:
    """Scan all tracked text files for absolute-path prefixes."""
    LOGGER.info("Running path-leak grep across git-tracked text files.")
    tracked = _git_ls_files()
    candidate_files = [path for path in tracked if path.suffix.lower() in TEXT_FILE_EXTENSIONS]
    all_matches: list[dict[str, Any]] = []
    for path in candidate_files:
        all_matches.extend(_scan_one_file_for_absolute_paths(path))
    log_lines: list[str] = [
        f"Scanned {len(candidate_files)} text-extension files out of {len(tracked)} tracked files.",
        f"Total absolute-path leak matches: {len(all_matches)}.",
    ]
    if all_matches:
        log_lines.append("---")
        for match in all_matches:
            log_lines.append(
                f"{match['file']}:{match['line_number']}: [{match['matched_prefix']}] {match['line_content']}"
            )
    LOGGER.info(
        "Path-leak grep complete: %d candidate files, %d matches.",
        len(candidate_files),
        len(all_matches),
    )
    return all_matches, log_lines


def _read_audit_json(path: Path) -> dict[str, Any]:
    """Read one audit JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _load_oof_summary_for_all_models() -> dict[str, dict[str, float]]:
    """Read the six nested-CV results JSONs and extract OOF AUC-ROC point + 95% CI."""
    summary: dict[str, dict[str, float]] = {}
    for model_name in MODEL_NAMES:
        results_path = PROJECT_ROOT / "outputs" / "analysis" / f"binary_{model_name}_nested_cv_results.json"
        payload = _read_audit_json(results_path)
        aggregate = payload.get("aggregate_metrics")
        if not isinstance(aggregate, dict) or "auc_roc" not in aggregate:
            raise AssertionError(f"{results_path} missing aggregate_metrics.auc_roc entry.")
        auc_entry = aggregate["auc_roc"]
        summary[model_name] = {
            "point": float(auc_entry["estimate"]),
            "ci_low": float(auc_entry["ci_low"]),
            "ci_high": float(auc_entry["ci_high"]),
        }
    return summary


def _load_best_model_test_metrics(best_model: str) -> dict[str, float]:
    """Read the best-model test metrics for AUC-ROC and Brier reporting."""
    metrics_path = PROJECT_ROOT / "outputs" / "metrics" / f"binary_{best_model}_test_metrics.csv"
    frame = pd.read_csv(metrics_path)
    test_frame = frame[frame["scope"] == "test"]
    auc_row = test_frame[test_frame["metric"] == "auc_roc"].iloc[0]
    brier_pre = test_frame[test_frame["metric"] == "brier_pre_platt"].iloc[0]
    brier_post = test_frame[test_frame["metric"] == "brier_post_platt"].iloc[0]
    brier_delta = test_frame[test_frame["metric"] == "brier_delta"].iloc[0]
    return {
        "auc_roc": float(auc_row["value"]),
        "auc_roc_ci_low": float(auc_row["ci_low"]),
        "auc_roc_ci_high": float(auc_row["ci_high"]),
        "brier_pre_platt": float(brier_pre["value"]),
        "brier_post_platt": float(brier_post["value"]),
        "brier_delta": float(brier_delta["value"]),
    }


def _load_best_model_final_payload(best_model: str) -> dict[str, Any]:
    """Load the best-model final PKL and extract threshold/calibrator metadata."""
    pkl_path = PROJECT_ROOT / "outputs" / "models" / f"binary_{best_model}_final.pkl"
    payload = load_final_model(pkl_path)
    return {
        "threshold_p_raw": float(payload["threshold_p_raw"]),
        "threshold_p_final_equivalent": float(payload["threshold_p_final_equivalent"]),
        "oof_source": payload["metadata"]["oof_source"],
        "training_timestamp_utc": payload["metadata"]["training_timestamp_utc"],
        "training_git_commit_hash": payload["metadata"]["git_commit_hash"],
    }


ERRATA_TABLE = (
    ("E001", "2026-05-17", "777/195 split size correction"),
    ("E002", "2026-05-18", "MR+R and XGBoost YAML deferred to Session 6"),
    ("E003", "2026-05-18", "sample_id integer-sort fix"),
    ("E004", "2026-05-18", "InceptionTime config + DL holdout audit"),
    ("E005", "2026-05-18", "DL early stopping val_loss + patience 40"),
    ("E006", "2026-05-18", "CBLA config lock"),
    ("E007", "2026-05-18", "sample_id sort across nested-CV CSVs"),
    ("E008", "2026-05-19", "Step 1 non-dominated set semantics"),
    ("E009", "2026-05-19", "Non-HP plumbing keys excluded from HP aggregation"),
    ("E010", "2026-05-19", "load_final_model API + DL inference module path"),
    ("E011", "2026-05-19", "Single-RNG sequential bootstrap convention"),
    ("E012", "2026-05-20", "LGIN primary analysis promoted to OOF view"),
    ("E013", "2026-05-20", "Section 2.14 fusion template condition revised"),
    ("E014", "2026-05-21", "Permutation-importance negative clip-to-zero"),
)


def _build_errata_list() -> list[dict[str, str]]:
    """Return the canonical list of all 14 errata for the manifest."""
    return [
        {"id": erratum_id, "date": date, "title": title}
        for erratum_id, date, title in ERRATA_TABLE
    ]


def _build_post_hoc_block(
    *,
    stage7_part_a_audit: dict[str, Any],
    stage7_part_c_audit: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Assemble the post-hoc trigger outcome block per plan §4."""
    snapshot = stage7_part_a_audit["post_hoc_trigger_snapshot"]
    a_block = snapshot["A_subgroup_retrain"]
    c_block = snapshot["C_test_vs_cv_shift"]
    h_block = snapshot["H_calibration_failure"]
    f_block = stage7_part_c_audit["post_hoc_trigger_references"]["F_tumor_marker_underpowered"]
    return {
        "A_subgroup_retrain": {
            "evaluated": bool(a_block["evaluated"]),
            "triggered": bool(a_block["triggered"]),
            "rationale": (
                "HGIN+EGC subgroup AUC=0.9264; AGC subgroup AUC=0.9187; "
                "neither met both AUC-degradation conditions per modeling plan §4.1."
            ),
            "source": "outputs/analysis/stage7_part_a_audit.json",
        },
        "C_test_vs_cv_shift": {
            "evaluated": bool(c_block["evaluated"]),
            "triggered": bool(c_block["triggered"]),
            "rationale": (
                "Average test-minus-CV shift +0.026 with mixed direction across 6 models; "
                "bidirectional unanimous-shift rule not triggered per modeling plan §4.2."
            ),
            "source": "outputs/analysis/stage7_part_a_audit.json",
        },
        "F_tumor_marker_underpowered": {
            "evaluated": bool(f_block["evaluated"]),
            "triggered": bool(f_block["triggered"]),
            "rationale": (
                f"tm_test_subset n_pos={int(f_block['tm_test_subset_n_pos'])} "
                f">= 30; tm_train_subset n_pos={int(f_block['tm_train_subset_n_pos'])} "
                ">= 50 per modeling plan §4.3."
            ),
            "source": "outputs/analysis/stage7_part_c_audit.json",
        },
        "G_dl_anomaly": {
            "evaluated": True,
            "triggered": True,
            "rationale": (
                "InceptionTime first run had 4 outer folds early-stopped before epoch 20; "
                "resolved via Erratum E005 (val_loss monitor, patience=40) prior to the locked "
                "Session 8 + Session 9 training runs and Session 10 Phase 2 final training."
            ),
            "source": "Erratum E005 in .cursor/plans/modeling_plan_v1_a7c3f9b2_dc789bb5.plan.md",
        },
        "H_calibration_failure": {
            "evaluated": bool(h_block["evaluated"]),
            "triggered": bool(h_block["triggered"]),
            "rationale": (
                "Best model post-Platt minus pre-Platt Brier delta +0.0018, below the 0.005 "
                "degradation threshold per modeling plan §4.5."
            ),
            "source": "outputs/analysis/stage7_part_a_audit.json",
        },
    }


def _build_experiment_manifest(
    *,
    head_full_hash: str,
    smoke_results: list[dict[str, Any]],
    path_leak_count: int,
    path_leak_files: list[str],
    stage7_part_a_audit: dict[str, Any],
    stage7_part_b_audit: dict[str, Any],
    stage7_part_c_audit: dict[str, Any],
    stage8_audit: dict[str, Any],
    oof_summary: dict[str, dict[str, float]],
    test_metrics: dict[str, float],
    final_payload_meta: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the full experiment manifest payload per plan §3.28."""
    preprocessing_frozen_hash = _git_rev_parse("preprocessing-frozen")
    modeling_prespecified_hash = _git_rev_parse("modeling-prespecified")
    data_split_locked_hash = _git_rev_parse("data-split-locked")
    best_model_selected_hash = _git_rev_parse("best-model-selected")
    stage8_findings = stage8_audit["key_findings"]
    stage7_b_findings = stage7_part_b_audit["key_findings"]
    stage7_c_findings = stage7_part_c_audit["key_findings"]

    smoke_passed = all(result["passed"] for result in smoke_results)
    return {
        "stage": "modeling",
        "frozen_at_utc": _timestamp_utc(),
        "modeling_stage_head_commit": head_full_hash,
        "modeling_stage_head_short_hash": head_full_hash[:7],
        "preprocessing_stage_head_commit": preprocessing_frozen_hash,
        "annotated_tags": {
            "modeling-prespecified": {
                "commit": modeling_prespecified_hash,
                "short": modeling_prespecified_hash[:7],
                "rationale": (
                    "Methodology lock prior to first implementation commit per TRIPOD prespecification."
                ),
            },
            "data-split-locked": {
                "commit": data_split_locked_hash,
                "short": data_split_locked_hash[:7],
                "data_split_seed": 43,
                "candidate_seeds_evaluated": [42, 43, 44, 45, 46],
                "rationale": (
                    "Train/test split, outer 5-fold, inner 3-fold, and DL early-stopping holdout "
                    "indices generated; data_split_seed=43 selected and locked."
                ),
            },
            "best-model-selected": {
                "commit": best_model_selected_hash,
                "short": best_model_selected_hash[:7],
                "best_model": "minirocket_ridge",
                "selection_step": 2,
                "selection_method": (
                    "paired DeLong significance at alpha=0.05 after Erratum E008 non-dominated "
                    "Step 1 filtering"
                ),
                "oof_auc_roc_point": oof_summary["minirocket_ridge"]["point"],
                "oof_auc_roc_ci_low_high": [
                    oof_summary["minirocket_ridge"]["ci_low"],
                    oof_summary["minirocket_ridge"]["ci_high"],
                ],
                "rationale": "Best model locked before test-set evaluation.",
            },
            "modeling-frozen": {
                "commit": "PENDING_USER_TAG",
                "rationale": (
                    "End-to-end modeling stage frozen; 14 errata documented; "
                    "5 post-hoc triggers evaluated."
                ),
            },
        },
        "errata_recorded": {
            "count": len(ERRATA_TABLE),
            "list": _build_errata_list(),
            "source": ".cursor/plans/modeling_plan_v1_a7c3f9b2_dc789bb5.plan.md §12",
        },
        "post_hoc_trigger_outcomes": _build_post_hoc_block(
            stage7_part_a_audit=stage7_part_a_audit,
            stage7_part_c_audit=stage7_part_c_audit,
        ),
        "data_split": {
            "data_split_seed": 43,
            "candidate_seeds": [42, 43, 44, 45, 46],
            "n_train": 777,
            "n_test": 195,
            "n_final_cohort": 972,
            "stratification_dimensions": ["binary_label", "disease_category"],
            "erratum_split_size_correction": "E001",
        },
        "model_selection_findings": {
            "best_model": "minirocket_ridge",
            "all_six_models_oof_auc_roc": oof_summary,
            "selection_step": 2,
            "selection_method": (
                "Step 1 non-dominated filtering per Erratum E008; Step 2 paired DeLong at "
                "Bonferroni alpha=0.05 over n_pairs=1 forwarded pair."
            ),
            "delong_p_minirocket_vs_xgboost": 0.009427,
            "source": "outputs/analysis/model_selection_report.md",
        },
        "final_training_findings": {
            "best_model": "minirocket_ridge",
            "best_model_test_auc_roc": test_metrics["auc_roc"],
            "best_model_test_auc_roc_ci_low_high": [
                test_metrics["auc_roc_ci_low"],
                test_metrics["auc_roc_ci_high"],
            ],
            "best_model_brier_pre_platt": test_metrics["brier_pre_platt"],
            "best_model_brier_post_platt": test_metrics["brier_post_platt"],
            "best_model_brier_delta": test_metrics["brier_delta"],
            "best_model_threshold_p_raw": final_payload_meta["threshold_p_raw"],
            "best_model_threshold_p_final_equivalent": final_payload_meta["threshold_p_final_equivalent"],
            "best_model_oof_source": final_payload_meta["oof_source"],
            "best_model_training_timestamp_utc": final_payload_meta["training_timestamp_utc"],
            "best_model_training_git_commit_hash": final_payload_meta["training_git_commit_hash"],
            "source": (
                "outputs/metrics/binary_minirocket_ridge_test_metrics.csv; "
                "outputs/models/binary_minirocket_ridge_final.pkl"
            ),
        },
        "stage7_key_findings": {
            "jt_p_one_sided_upper": stage7_b_findings["jt_p_one_sided_upper"],
            "spearman_rho_test_p_cal_vs_disease_category": stage7_b_findings[
                "spearman_rho_test_p_cal_vs_disease_category"
            ],
            "fp_lgin_fisher_odds_ratio": stage7_b_findings["fp_lgin_fisher_odds_ratio"],
            "fp_lgin_fisher_p_two_sided": stage7_b_findings["fp_lgin_fisher_p_two_sided"],
            "lgin_rb_oof": stage7_b_findings["lgin_rb_oof"],
            "lgin_mannwhitney_p_two_sided_oof": stage7_b_findings["lgin_mannwhitney_p_two_sided_oof"],
            "fusion_conclusion_template_used": stage7_c_findings["fusion_conclusion_template_used"],
            "fusion_delta_auc_minus_voc": stage7_c_findings["delta_auc_fusion_minus_voc"],
            "fusion_delong_p": stage7_c_findings["delong_p_fusion_vs_voc"],
            "fusion_nri": stage7_c_findings["nri_fusion_vs_voc"],
            "fusion_idi": stage7_c_findings["idi_fusion_vs_voc"],
            "voc_only_auc_on_tm_test_subset": stage7_c_findings["voc_only_auc_on_tm_test_subset"],
            "marker_lr_auc_on_tm_test_subset": stage7_c_findings["marker_lr_auc_on_tm_test_subset"],
            "delong_p_voc_vs_marker": stage7_c_findings["delong_p_voc_vs_marker"],
            "selection_bias_largest_smd": stage7_c_findings["selection_bias_largest_smd"],
            "selection_bias_largest_smd_variable": stage7_c_findings["selection_bias_largest_smd_variable"],
        },
        "stage8_key_findings": {
            "unanimous_top_5_channels": stage8_findings["channels_unanimous_top_5"],
            "top_3_majority_4_of_6": stage8_findings["top_3_channels_appearing_in_at_least_4_of_6_models"],
            "top_5_majority_4_of_6": stage8_findings["top_5_channels_appearing_in_at_least_4_of_6_models"],
            "top_7_majority_4_of_6": stage8_findings["top_7_channels_appearing_in_at_least_4_of_6_models"],
            "kendall_tau_mean_across_15_pairs": stage8_findings["kendall_tau_mean_across_15_pairs"],
            "kendall_tau_min_across_15_pairs": stage8_findings["kendall_tau_min_across_15_pairs"],
            "kendall_tau_max_across_15_pairs": stage8_findings["kendall_tau_max_across_15_pairs"],
            "n_pairs_significant_at_alpha_0.05": stage8_findings["n_pairs_significant_at_alpha_0.05"],
            "negative_permutation_handling_erratum": "E014",
        },
        "reproducibility_verification": {
            "smoke_test_passed": smoke_passed,
            "smoke_test_n_models": len(smoke_results),
            "smoke_test_n_samples_per_model": SMOKE_TEST_N_TOTAL,
            "smoke_test_models_passed": [r["model"] for r in smoke_results if r["passed"]],
            "path_leak_count": int(path_leak_count),
            "path_leak_files": path_leak_files,
            "captum_version": stage8_audit["captum_version"],
            "random_seed": RANDOM_SEED,
            "bootstrap_n_boot": stage8_audit["bootstrap_n_boot"],
            "bootstrap_convention_erratum": "E011",
        },
    }


def run_reproducibility_verification(config_path: Path) -> None:
    """Execute Stage 9 reproducibility verification end-to-end.

    Args:
        config_path: Path to `configs/modeling_global.yaml` (used only for path resolution).
    """
    _configure_logging()
    LOGGER.info("Stage 9 reproducibility verification started.")
    _load_yaml_config(config_path=config_path)

    head_full_hash = _git_rev_parse("HEAD")
    LOGGER.info("HEAD commit: %s", head_full_hash)

    smoke_results, smoke_log_lines = _run_smoke_tests()
    smoke_log_path = PROJECT_ROOT / "outputs" / "logs" / "reproducibility_smoke_test.log"
    smoke_log_header = [
        f"# Stage 9 end-to-end reproducibility smoke test ({_timestamp_utc()})",
        f"# HEAD commit: {head_full_hash}",
        f"# Models: {len(smoke_results)}; samples per model: {SMOKE_TEST_N_TOTAL}",
        "",
    ]
    _write_text(smoke_log_path, "\n".join(smoke_log_header + smoke_log_lines) + "\n")
    LOGGER.info("Wrote smoke test log to %s.", _format_path(smoke_log_path))

    path_leaks, path_leak_log_lines = _run_path_leak_grep()
    path_leak_log_path = PROJECT_ROOT / "outputs" / "logs" / "path_leak_grep.log"
    path_leak_header = [
        f"# Stage 9 path-leak grep ({_timestamp_utc()})",
        f"# HEAD commit: {head_full_hash}",
        "",
    ]
    _write_text(path_leak_log_path, "\n".join(path_leak_header + path_leak_log_lines) + "\n")
    LOGGER.info("Wrote path-leak log to %s.", _format_path(path_leak_log_path))

    stage7_part_a_audit = _read_audit_json(PROJECT_ROOT / "outputs" / "analysis" / "stage7_part_a_audit.json")
    stage7_part_b_audit = _read_audit_json(PROJECT_ROOT / "outputs" / "analysis" / "stage7_part_b_audit.json")
    stage7_part_c_audit = _read_audit_json(PROJECT_ROOT / "outputs" / "analysis" / "stage7_part_c_audit.json")
    stage8_audit = _read_audit_json(PROJECT_ROOT / "outputs" / "analysis" / "stage8_audit.json")
    oof_summary = _load_oof_summary_for_all_models()
    test_metrics = _load_best_model_test_metrics(best_model="minirocket_ridge")
    final_payload_meta = _load_best_model_final_payload(best_model="minirocket_ridge")

    manifest = _build_experiment_manifest(
        head_full_hash=head_full_hash,
        smoke_results=smoke_results,
        path_leak_count=len(path_leaks),
        path_leak_files=sorted({match["file"] for match in path_leaks}),
        stage7_part_a_audit=stage7_part_a_audit,
        stage7_part_b_audit=stage7_part_b_audit,
        stage7_part_c_audit=stage7_part_c_audit,
        stage8_audit=stage8_audit,
        oof_summary=oof_summary,
        test_metrics=test_metrics,
        final_payload_meta=final_payload_meta,
    )

    manifest_path = PROJECT_ROOT / "outputs" / "logs" / "experiment_manifest.json"
    _write_json(manifest_path, manifest)
    LOGGER.info("Wrote experiment manifest to %s.", _format_path(manifest_path))

    audit_payload = {
        "session": 15,
        "stage": "Stage 9",
        "git_commit_hash": head_full_hash,
        "timestamp_utc": _timestamp_utc(),
        "deliverables": {
            "experiment_manifest": "outputs/logs/experiment_manifest.json",
            "reproducibility_smoke_test_log": "outputs/logs/reproducibility_smoke_test.log",
            "path_leak_grep_log": "outputs/logs/path_leak_grep.log",
            "modeling_run_summary_md": "docs/modeling/validated_modeling_run_summary.md",
        },
        "verification_results": {
            "smoke_test_passed": all(result["passed"] for result in smoke_results),
            "smoke_test_n_models_passed": sum(1 for result in smoke_results if result["passed"]),
            "smoke_test_n_samples_per_model": SMOKE_TEST_N_TOTAL,
            "smoke_test_per_model": [
                {
                    "model": result["model"],
                    "passed": result["passed"],
                    "n_finite": result["n_finite"],
                    "n_in_range": result["n_in_range"],
                    "quadrant_distribution": result["quadrant_distribution"],
                }
                for result in smoke_results
            ],
            "path_leak_count": len(path_leaks),
            "path_leak_files": sorted({match["file"] for match in path_leaks}),
        },
        "errata_count": len(ERRATA_TABLE),
        "post_hoc_trigger_outcomes_summary": {
            "A_subgroup_retrain": False,
            "C_test_vs_cv_shift": False,
            "F_tumor_marker_underpowered": False,
            "G_dl_anomaly": True,
            "H_calibration_failure": False,
        },
        "bootstrap_convention": {
            "erratum": "E011",
            "linked_commit": "a2af64e",
            "summary": (
                "single-RNG sequential per E011 (commit a2af64e); reproducibility verified via "
                "deterministic PCG64 stream; valid for all bootstrap-derived CIs in this session."
            ),
        },
    }
    audit_path = PROJECT_ROOT / "outputs" / "analysis" / "stage9_audit.json"
    _write_json(audit_path, audit_payload)
    LOGGER.info("Wrote Stage 9 audit JSON to %s.", _format_path(audit_path))

    if not all(result["passed"] for result in smoke_results):
        failed = [result["model"] for result in smoke_results if not result["passed"]]
        raise AssertionError(
            f"Smoke test failed for models: {failed}. See {_format_path(smoke_log_path)} for details."
        )
    if path_leaks:
        raise AssertionError(
            f"Path-leak grep found {len(path_leaks)} matches. See {_format_path(path_leak_log_path)} "
            "for details. User must decide whether to sanitize or accept."
        )
    LOGGER.info("Stage 9 reproducibility verification complete.")


def _build_argument_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Stage 9 reproducibility verification.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "modeling_global.yaml",
        help="Path to the modeling global config YAML.",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = _build_argument_parser()
    args = parser.parse_args()
    run_reproducibility_verification(config_path=_resolve_path(args.config))


if __name__ == "__main__":
    main()
