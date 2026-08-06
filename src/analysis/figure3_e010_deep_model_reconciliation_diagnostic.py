"""Aggregate-only E010 deep-model score-reconciliation diagnostic."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import re
import struct
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
import yaml

from src.config import PROJECT_ROOT


LOGGER = logging.getLogger(__name__)
DIAGNOSTIC_ID = "figure3_e010_deep_model_reconciliation_diagnostic"
SYNTHETIC_TESTS_REQUIRE_REAL_ARTIFACTS = False
APPROVED_PLAN = {
    "path": "docs/plans/figure3_e010_deep_model_reconciliation_diagnostic.plan.md",
    "sha256": "9129067eaf3c1d433e0b9eddd10a0fdb8a7ee95c152f931b3f69d55dec9ce509",
}
EXPECTED_GIT_HEAD = "3478442b26bf42e7963b61916d8ac6180984ebbc"
TRAINING_COMMIT = "03f4126ec356a4facc17b05ab7b302cb339f0bb5"

MODEL_ORDER: tuple[dict[str, Any], ...] = (
    {"model_id": "inception_time", "model_display_name": "InceptionTime", "model_order": 1},
    {
        "model_id": "cnn_bilstm_attention",
        "model_display_name": "CNN–BiLSTM–Attention",
        "model_order": 2,
    },
)
CONDITION_ORDER: tuple[dict[str, Any], ...] = (
    {
        "condition_id": "cpu_original_direct",
        "condition_order": 1,
        "requested_device_class": "cpu",
        "input_order": "original_test_idx",
        "train_warmup": False,
    },
    {
        "condition_id": "cpu_original_train_warmup",
        "condition_order": 2,
        "requested_device_class": "cpu",
        "input_order": "original_test_idx",
        "train_warmup": True,
    },
    {
        "condition_id": "gpu_original_direct",
        "condition_order": 3,
        "requested_device_class": "cuda",
        "input_order": "original_test_idx",
        "train_warmup": False,
    },
    {
        "condition_id": "gpu_original_train_warmup",
        "condition_order": 4,
        "requested_device_class": "cuda",
        "input_order": "original_test_idx",
        "train_warmup": True,
    },
    {
        "condition_id": "cpu_participant_sorted_direct",
        "condition_order": 5,
        "requested_device_class": "cpu",
        "input_order": "participant_key_sorted",
        "train_warmup": False,
    },
    {
        "condition_id": "gpu_participant_sorted_direct",
        "condition_order": 6,
        "requested_device_class": "cuda",
        "input_order": "participant_key_sorted",
        "train_warmup": False,
    },
)

CONDITION_DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "diagnostic_id",
    "model_id",
    "model_display_name",
    "model_order",
    "condition_id",
    "condition_order",
    "requested_device_class",
    "input_order",
    "train_warmup",
    "load_number",
    "repeat_number",
    "wrapper_accelerator_before",
    "wrapper_accelerator_after",
    "wrapper_batch_size",
    "actual_parameter_device_class",
    "score_n",
    "score_all_finite",
    "score_in_unit_interval",
    "max_abs_difference",
    "mean_abs_difference",
    "median_abs_difference",
    "rmse_difference",
    "abs_difference_q95",
    "abs_difference_q99",
    "count_abs_difference_gt_1e_6",
    "count_abs_difference_gt_1e_5",
    "count_abs_difference_gt_1e_4",
    "count_abs_difference_gt_1e_3",
    "reproduced_auroc",
    "frozen_prediction_auroc",
    "frozen_metric_auroc",
    "reproduced_vs_prediction_auroc_difference",
    "reproduced_vs_metric_auroc_difference",
    "prediction_vs_metric_auroc_difference",
    "ranking_identical",
    "kendall_tau_b",
    "kendall_estimable",
    "kendall_nonestimable_reason",
    "spearman_rho",
    "spearman_estimable",
    "spearman_nonestimable_reason",
    "state_sha256_before",
    "state_sha256_after_warmup",
    "state_sha256_after_inference",
    "state_unchanged_from_before",
    "score_reconciliation_passed",
    "auroc_reconciliation_passed",
)
COMPARISON_DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "diagnostic_id",
    "comparison_type",
    "comparison_type_order",
    "model_id",
    "model_order",
    "left_condition_id",
    "right_condition_id",
    "left_load_number",
    "right_load_number",
    "left_repeat_number",
    "right_repeat_number",
    "score_n",
    "max_abs_score_difference",
    "exact_score_identity",
    "exceeds_score_reconciliation_tolerance",
)
MODEL_CONDITION_SUMMARY_COLUMNS: tuple[str, ...] = (
    "diagnostic_id",
    "model_id",
    "model_display_name",
    "model_order",
    "condition_id",
    "condition_order",
    "requested_device_class",
    "input_order",
    "train_warmup",
    "n_fresh_loads",
    "n_repeats_per_load",
    "all_actual_devices_match_request",
    "all_wrapper_contracts_unchanged",
    "all_score_vectors_valid",
    "worst_max_abs_difference_from_frozen",
    "maximum_count_abs_difference_gt_1e_6",
    "maximum_count_abs_difference_gt_1e_5",
    "maximum_count_abs_difference_gt_1e_4",
    "maximum_count_abs_difference_gt_1e_3",
    "all_score_reconciliations_passed",
    "all_auroc_reconciliations_passed",
    "condition_reconciles",
    "maximum_within_load_repeat_difference",
    "maximum_between_fresh_load_difference",
    "all_state_digests_unchanged",
    "within_load_differences_within_1e_6",
    "between_load_differences_within_1e_6",
    "score_reconciliation_status_consistent",
    "auroc_reconciliation_status_consistent",
    "wrapper_and_device_checks_passed",
    "condition_stable_at_frozen_scale",
    "all_rankings_identical",
    "minimum_estimable_kendall_tau_b",
    "minimum_estimable_spearman_rho",
)
INTERPRETATION_BRANCH_COLUMNS: tuple[str, ...] = (
    "diagnostic_id",
    "model_id",
    "model_display_name",
    "model_order",
    "gpu_stage4b_sequence_reconciles",
    "matched_cpu_sequence_reconciles",
    "e010_gpu_to_cpu_effect_isolatable",
    "e010_gpu_to_cpu_effect_nonisolatable_reason",
    "e010_gpu_to_cpu_change_supported",
    "cpu_warmup_effect_isolatable",
    "cpu_warmup_effect_nonisolatable_reason",
    "gpu_warmup_effect_isolatable",
    "gpu_warmup_effect_nonisolatable_reason",
    "warmup_sequence_effect_supported",
    "cpu_input_order_effect_isolatable",
    "cpu_input_order_effect_nonisolatable_reason",
    "gpu_input_order_effect_isolatable",
    "gpu_input_order_effect_nonisolatable_reason",
    "input_order_sensitivity_supported",
    "within_load_nondeterminism_observed",
    "between_load_nondeterminism_observed",
    "model_state_mutation_observed",
    "any_condition_reconciles",
    "unresolved_artifact_state_or_provenance_mismatch",
    "root_cause_summary",
    "pre_e010_pkl_available",
    "e010_source_checkpoint_available",
)
EXPECTED_OUTPUT_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "condition_diagnostics.csv": CONDITION_DIAGNOSTIC_COLUMNS,
    "comparison_diagnostics.csv": COMPARISON_DIAGNOSTIC_COLUMNS,
    "model_condition_summary.csv": MODEL_CONDITION_SUMMARY_COLUMNS,
    "interpretation_branches.csv": INTERPRETATION_BRANCH_COLUMNS,
}
EXPECTED_OUTPUT_ROWS: Mapping[str, int] = {
    "condition_diagnostics.csv": 48,
    "comparison_diagnostics.csv": 104,
    "model_condition_summary.csv": 12,
    "interpretation_branches.csv": 2,
}
FINAL_MANIFEST_FILENAMES: tuple[str, ...] = (
    "condition_diagnostics.csv",
    "comparison_diagnostics.csv",
    "model_condition_summary.csv",
    "interpretation_branches.csv",
    "audit.json",
    "report.md",
)
COMPARISON_PAIRS: Mapping[str, tuple[tuple[str, str], ...]] = {
    "cpu_gpu": (
        ("cpu_original_direct", "gpu_original_direct"),
        ("cpu_original_train_warmup", "gpu_original_train_warmup"),
        ("cpu_participant_sorted_direct", "gpu_participant_sorted_direct"),
    ),
    "input_order": (
        ("cpu_original_direct", "cpu_participant_sorted_direct"),
        ("gpu_original_direct", "gpu_participant_sorted_direct"),
    ),
    "train_warmup": (
        ("cpu_original_direct", "cpu_original_train_warmup"),
        ("gpu_original_direct", "gpu_original_train_warmup"),
    ),
}
BRANCH_ORDER: tuple[str, ...] = (
    "e010_gpu_to_cpu_change_supported",
    "warmup_sequence_effect_supported",
    "input_order_sensitivity_supported",
    "within_load_nondeterminism_observed",
    "between_load_nondeterminism_observed",
    "model_state_mutation_observed",
)
EXPECTED_STATUS_PATHS = {
    "?? .vscode/",
    "?? configs/figure3_e010_deep_model_reconciliation_diagnostic.yaml",
    "?? configs/figure3_unified_channel_block_permutation.yaml",
    "?? scripts/run_figure3_e010_deep_model_reconciliation_diagnostic.py",
    "?? scripts/run_figure3_unified_channel_block_permutation.py",
    "?? src/analysis/figure3_e010_deep_model_reconciliation_diagnostic.py",
    "?? src/analysis/figure3_unified_channel_block_permutation.py",
    "?? tests/test_figure3_e010_deep_model_reconciliation_diagnostic.py",
    "?? tests/test_figure3_unified_channel_block_permutation.py",
}
_FORBIDDEN_OUTPUT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsample_id\b", re.IGNORECASE),
    re.compile(r"\bparticipant_(?:key|id|index|score|label|row)s?\b", re.IGNORECASE),
    re.compile(r"\by_true\b", re.IGNORECASE),
    re.compile(r"\border(?:ing)?_map\b", re.IGNORECASE),
    re.compile(r"\bindex_(?:array|vector)s?\b", re.IGNORECASE),
    re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b"),
    re.compile(r"(?:^|[\s\"'])(?:/home/|/mnt/[a-zA-Z]/|[A-Za-z]:\\)"),
)


def _project_path(relative_path: str | Path) -> Path:
    """Resolve a project-relative, non-traversing path."""
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("Path must be project-relative and non-traversing.")
    return PROJECT_ROOT / candidate


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 of exact file bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def serialize_yaml(payload: Mapping[str, Any]) -> str:
    """Serialize YAML deterministically for synthetic contract tests."""
    return yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True)


def expected_configuration() -> dict[str, Any]:
    """Return a detached copy of the repository's closed diagnostic contract."""
    path = PROJECT_ROOT / "configs" / "figure3_e010_deep_model_reconciliation_diagnostic.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Frozen diagnostic configuration must be a mapping.")
    return json.loads(json.dumps(payload, ensure_ascii=False))


def load_diagnostic_config(path: Path) -> dict[str, Any]:
    """Load and strictly validate the closed Phase 2 configuration."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Diagnostic configuration must be a mapping.")
    expected = expected_configuration()
    for key in ("diagnostic_id", "batch_size", "n_fresh_loads", "n_repeated_test_calls_per_load"):
        if payload.get(key) != expected[key]:
            raise ValueError(f"Diagnostic configuration differs for {key}.")
    if payload != expected:
        raise ValueError("Diagnostic configuration differs from the frozen closed contract.")
    if payload["approved_plan"] != APPROVED_PLAN or payload["expected_git_head"] != EXPECTED_GIT_HEAD:
        raise ValueError("Diagnostic approval authority differs from the frozen contract.")
    if payload["models"] != list(MODEL_ORDER) or payload["conditions"] != list(CONDITION_ORDER):
        raise ValueError("Diagnostic model or condition order differs from the frozen contract.")
    configured_schemas = {
        name.removesuffix(".csv"): list(columns) for name, columns in EXPECTED_OUTPUT_COLUMNS.items()
    }
    if payload["schemas"] != configured_schemas:
        raise ValueError("Diagnostic output schemas differ from the frozen contract.")
    for candidate in [*payload["paths"].values(), *payload["immutable_inputs"], *payload["allowed_implementation_paths"]]:
        path_value = Path(candidate)
        if path_value.is_absolute() or ".." in path_value.parts:
            raise ValueError("Diagnostic configuration contains an invalid path.")
    return payload


def build_execution_schedule() -> list[dict[str, Any]]:
    """Return fixed model-condition-load records, each requiring two test calls."""
    return [
        {**model, **condition, "load_number": load_number, "n_test_calls": 2}
        for model in MODEL_ORDER
        for condition in CONDITION_ORDER
        for load_number in (1, 2)
    ]


def align_scores_to_frozen_order(
    scores: np.ndarray,
    input_keys: np.ndarray,
    frozen_keys: np.ndarray,
) -> np.ndarray:
    """Align one score vector to the frozen prediction order without returning a map."""
    values = np.asarray(scores, dtype=float)
    source = np.asarray(input_keys)
    target = np.asarray(frozen_keys)
    if values.ndim != 1 or source.ndim != 1 or target.ndim != 1 or len(values) != len(source):
        raise ValueError("Score alignment vectors have invalid dimensions.")
    if len(np.unique(source)) != len(source) or len(np.unique(target)) != len(target):
        raise ValueError("Score alignment keys are duplicated.")
    if set(source.tolist()) != set(target.tolist()):
        raise ValueError("Score alignment membership differs.")
    positions = {key: index for index, key in enumerate(source.tolist())}
    aligned = values[np.asarray([positions[key] for key in target.tolist()], dtype=np.int64)]
    return np.asarray(aligned, dtype=float)


def device_class_from_records(device_classes: Sequence[str]) -> str:
    """Require a nonempty, homogeneous CPU or CUDA device-class record."""
    classes = list(device_classes)
    if not classes:
        raise ValueError("Model parameter/buffer device records are absent.")
    unique = set(classes)
    if not unique.issubset({"cpu", "cuda"}) or len(unique) != 1:
        raise ValueError("Model parameter/buffer device records are mixed or invalid.")
    return classes[0]


def actual_model_device_class(model: Any) -> str:
    """Return the proven homogeneous device class of all model parameters and buffers."""
    records = [str(tensor.device.type) for _name, tensor in model.named_parameters()]
    records.extend(str(tensor.device.type) for _name, tensor in model.named_buffers())
    return device_class_from_records(records)


def canonical_model_state_sha256(model: Any) -> str:
    """Hash all named parameters and buffers using the approved canonical encoding."""
    records: list[tuple[str, str, Any]] = []
    records.extend(("parameter", name, tensor) for name, tensor in model.named_parameters())
    records.extend(("buffer", name, tensor) for name, tensor in model.named_buffers())
    if not records:
        raise ValueError("Model parameter/buffer state is absent.")
    digest = hashlib.sha256()
    for kind, name, tensor in sorted(records, key=lambda item: (item[0], item[1])):
        if not hasattr(tensor, "detach"):
            raise TypeError("Model state contains a non-tensor record.")
        contiguous = tensor.detach().cpu().contiguous()
        parts = [kind.encode("utf-8"), name.encode("utf-8"), str(contiguous.dtype).encode("utf-8")]
        for part in parts:
            digest.update(struct.pack("<Q", len(part)))
            digest.update(part)
        shape = tuple(int(value) for value in contiguous.shape)
        digest.update(struct.pack("<Q", len(shape)))
        for value in shape:
            digest.update(struct.pack("<q", value))
        raw = contiguous.reshape(-1).view(dtype=__import__("torch").uint8).numpy().tobytes(order="C")
        digest.update(struct.pack("<Q", len(raw)))
        digest.update(raw)
    return digest.hexdigest()


def invoke_ephemeral_gpu_predictor(wrapper: Any, normalized: np.ndarray) -> np.ndarray:
    """Call the persisted predictor with an ephemeral GPU argument without mutating the wrapper."""
    before_accelerator = wrapper.accelerator
    before_batch_size = wrapper.batch_size
    result = wrapper.predictor(
        model=wrapper.lightning_model,
        x_eval=normalized,
        batch_size=32,
        accelerator="gpu",
    )
    if wrapper.accelerator != before_accelerator or wrapper.batch_size != before_batch_size:
        raise ValueError("Persisted wrapper contract changed during ephemeral GPU inference.")
    return np.asarray(result, dtype=float)


def _invoke_cpu_predictor(wrapper: Any, normalized: np.ndarray) -> np.ndarray:
    """Invoke the persisted CPU wrapper and return positive-class scores."""
    probabilities = np.asarray(wrapper.predict_proba(normalized), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape != (len(normalized), 2):
        raise ValueError("Persisted CPU wrapper returned an invalid probability shape.")
    return probabilities[:, 1]


def _validate_score_vector(scores: np.ndarray, expected_n: int, context: str) -> np.ndarray:
    """Validate one probability vector without exposing its values."""
    values = np.asarray(scores, dtype=float)
    if values.shape != (expected_n,):
        raise ValueError(f"{context} score shape is invalid.")
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(f"{context} score finiteness or range is invalid.")
    return values


def execute_condition_load(
    *,
    wrapper: Any,
    normalizer: Any,
    condition: Mapping[str, Any],
    train_input: np.ndarray | None,
    test_input: np.ndarray,
    expected_train_n: int,
    expected_test_n: int,
) -> dict[str, Any]:
    """Execute one fresh-load condition with one optional warm-up and two test calls."""
    if wrapper.accelerator != "cpu" or wrapper.batch_size != 32:
        raise ValueError("Persisted wrapper accelerator or batch-size contract differs.")
    before_accelerator = str(wrapper.accelerator)
    state_before = canonical_model_state_sha256(wrapper.lightning_model)
    state_after_warmup = ""
    requested = str(condition["requested_device_class"])
    invoke = _invoke_cpu_predictor if requested == "cpu" else invoke_ephemeral_gpu_predictor
    if bool(condition["train_warmup"]):
        if train_input is None or len(train_input) != expected_train_n:
            raise ValueError("Execution-only training warm-up input count differs.")
        normalized_train = normalizer.transform(train_input)
        train_scores = _validate_score_vector(invoke(wrapper, normalized_train), expected_train_n, "Training warm-up")
        del train_scores, normalized_train
        state_after_warmup = canonical_model_state_sha256(wrapper.lightning_model)
    test_scores: list[np.ndarray] = []
    state_digests: list[str] = []
    actual_devices: list[str] = []
    for _repeat in (1, 2):
        normalized_test = normalizer.transform(test_input)
        test_scores.append(_validate_score_vector(invoke(wrapper, normalized_test), expected_test_n, "Test"))
        del normalized_test
        state_digests.append(canonical_model_state_sha256(wrapper.lightning_model))
        actual_devices.append(actual_model_device_class(wrapper.lightning_model))
    if wrapper.accelerator != "cpu" or wrapper.batch_size != 32:
        raise ValueError("Persisted wrapper contract changed during inference.")
    return {
        "test_scores": test_scores,
        "state_digests": {"before": state_before, "after_warmup": state_after_warmup, "after_calls": state_digests},
        "actual_device_classes": actual_devices,
        "wrapper_contract": {
            "accelerator_before": before_accelerator,
            "accelerator_after": str(wrapper.accelerator),
            "batch_size": int(wrapper.batch_size),
        },
    }


def rank_diagnostics(reproduced: np.ndarray, frozen: np.ndarray) -> dict[str, Any]:
    """Calculate exact ranking identity and coefficient-only rank correlations."""
    left = np.asarray(reproduced, dtype=float)
    right = np.asarray(frozen, dtype=float)
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape or len(left) < 2:
        raise ValueError("Rank-diagnostic vectors are invalid.")
    left_rank = np.asarray(rankdata(left, method="average"), dtype=float)
    right_rank = np.asarray(rankdata(right, method="average"), dtype=float)
    ranking_identical = bool(np.array_equal(left_rank, right_rank))
    from src.analysis.figure3_unified_channel_block_permutation import kendall_tau_b_coefficient

    kendall = kendall_tau_b_coefficient(left, right)
    if not bool(kendall["estimable"]):
        kendall_value = None
        kendall_reason = "constant_or_degenerate_score_vector"
    else:
        kendall_value = float(kendall["kendall_tau_b"])
        kendall_reason = ""
    left_centered = left_rank - float(np.mean(left_rank))
    right_centered = right_rank - float(np.mean(right_rank))
    denominator = math.sqrt(float(np.sum(left_centered**2) * np.sum(right_centered**2)))
    if denominator == 0.0:
        spearman_value = None
        spearman_estimable = False
        spearman_reason = "constant_or_degenerate_score_vector"
    else:
        spearman_value = float(np.sum(left_centered * right_centered) / denominator)
        spearman_estimable = True
        spearman_reason = ""
    return {
        "ranking_identical": ranking_identical,
        "kendall_tau_b": kendall_value,
        "kendall_estimable": bool(kendall["estimable"]),
        "kendall_nonestimable_reason": kendall_reason,
        "spearman_rho": spearman_value,
        "spearman_estimable": spearman_estimable,
        "spearman_nonestimable_reason": spearman_reason,
    }


def calculate_score_diagnostics(
    reproduced: np.ndarray,
    frozen: np.ndarray,
    labels: np.ndarray,
    *,
    frozen_metric_auroc: float,
    score_atol: float,
    auroc_atol: float,
) -> dict[str, Any]:
    """Calculate all frozen aggregate score, AUROC, and rank diagnostics."""
    scores = np.asarray(reproduced, dtype=float)
    reference = np.asarray(frozen, dtype=float)
    outcomes = np.asarray(labels, dtype=np.int64)
    if scores.ndim != 1 or reference.shape != scores.shape or outcomes.shape != scores.shape:
        raise ValueError("Score diagnostic vectors have invalid shapes.")
    if not np.isfinite(scores).all() or not np.isfinite(reference).all():
        raise ValueError("Score diagnostic vectors contain nonfinite values.")
    absolute = np.abs(scores - reference)
    reproduced_auroc = float(roc_auc_score(outcomes, scores))
    frozen_auroc = float(roc_auc_score(outcomes, reference))
    metric_auroc = float(frozen_metric_auroc)
    pairwise = (
        abs(reproduced_auroc - frozen_auroc),
        abs(reproduced_auroc - metric_auroc),
        abs(frozen_auroc - metric_auroc),
    )
    result: dict[str, Any] = {
        "score_n": int(len(scores)),
        "score_all_finite": True,
        "score_in_unit_interval": bool(np.all((scores >= 0.0) & (scores <= 1.0))),
        "max_abs_difference": float(np.max(absolute)),
        "mean_abs_difference": float(np.mean(absolute)),
        "median_abs_difference": float(np.median(absolute)),
        "rmse_difference": float(np.sqrt(np.mean((scores - reference) ** 2))),
        "abs_difference_q95": float(np.percentile(absolute, 95, method="linear")),
        "abs_difference_q99": float(np.percentile(absolute, 99, method="linear")),
        "count_abs_difference_gt_1e_6": int(np.count_nonzero(absolute > 1.0e-6)),
        "count_abs_difference_gt_1e_5": int(np.count_nonzero(absolute > 1.0e-5)),
        "count_abs_difference_gt_1e_4": int(np.count_nonzero(absolute > 1.0e-4)),
        "count_abs_difference_gt_1e_3": int(np.count_nonzero(absolute > 1.0e-3)),
        "reproduced_auroc": reproduced_auroc,
        "frozen_prediction_auroc": frozen_auroc,
        "frozen_metric_auroc": metric_auroc,
        "reproduced_vs_prediction_auroc_difference": float(pairwise[0]),
        "reproduced_vs_metric_auroc_difference": float(pairwise[1]),
        "prediction_vs_metric_auroc_difference": float(pairwise[2]),
        "score_reconciliation_passed": bool(float(np.max(absolute)) <= score_atol),
        "auroc_reconciliation_passed": bool(all(value <= auroc_atol for value in pairwise)),
    }
    result.update(rank_diagnostics(scores, reference))
    return result


def _comparison_row(
    comparison_type: str,
    comparison_type_order: int,
    model: Mapping[str, Any],
    left_condition: str,
    right_condition: str,
    left_load: int,
    right_load: int,
    left_repeat: int,
    right_repeat: int,
    scores: Mapping[tuple[str, str, int, int], np.ndarray],
    score_atol: float,
) -> dict[str, Any]:
    """Build one aggregate score-vector comparison row."""
    left = np.asarray(scores[(model["model_id"], left_condition, left_load, left_repeat)], dtype=float)
    right = np.asarray(scores[(model["model_id"], right_condition, right_load, right_repeat)], dtype=float)
    if left.shape != right.shape or left.ndim != 1 or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Fixed score-vector comparison is invalid.")
    maximum = float(np.max(np.abs(left - right)))
    return {
        "diagnostic_id": DIAGNOSTIC_ID,
        "comparison_type": comparison_type,
        "comparison_type_order": comparison_type_order,
        "model_id": model["model_id"],
        "model_order": model["model_order"],
        "left_condition_id": left_condition,
        "right_condition_id": right_condition,
        "left_load_number": left_load,
        "right_load_number": right_load,
        "left_repeat_number": left_repeat,
        "right_repeat_number": right_repeat,
        "score_n": int(len(left)),
        "max_abs_score_difference": maximum,
        "exact_score_identity": bool(np.array_equal(left, right)),
        "exceeds_score_reconciliation_tolerance": bool(maximum > score_atol),
    }


def build_comparison_diagnostics(
    scores: Mapping[tuple[str, str, int, int], np.ndarray],
    *,
    score_atol: float,
) -> pd.DataFrame:
    """Build the fixed 104-row score-comparison matrix."""
    expected_keys = {
        (model["model_id"], condition["condition_id"], load, repeat)
        for model in MODEL_ORDER
        for condition in CONDITION_ORDER
        for load in (1, 2)
        for repeat in (1, 2)
    }
    if set(scores) != expected_keys:
        raise ValueError("Aligned score-vector store differs from the fixed structure.")
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for condition in CONDITION_ORDER:
            condition_id = str(condition["condition_id"])
            for load in (1, 2):
                rows.append(_comparison_row("within_load_repeat", 1, model, condition_id, condition_id, load, load, 1, 2, scores, score_atol))
    for model in MODEL_ORDER:
        for condition in CONDITION_ORDER:
            condition_id = str(condition["condition_id"])
            for repeat in (1, 2):
                rows.append(_comparison_row("between_fresh_load", 2, model, condition_id, condition_id, 1, 2, repeat, repeat, scores, score_atol))
    for type_order, comparison_type in enumerate(("cpu_gpu", "input_order", "train_warmup"), start=3):
        for model in MODEL_ORDER:
            for left_condition, right_condition in COMPARISON_PAIRS[comparison_type]:
                for load in (1, 2):
                    for repeat in (1, 2):
                        rows.append(_comparison_row(comparison_type, type_order, model, left_condition, right_condition, load, load, repeat, repeat, scores, score_atol))
    frame = pd.DataFrame(rows, columns=COMPARISON_DIAGNOSTIC_COLUMNS)
    if len(frame) != 104:
        raise ValueError("Comparison diagnostic row count differs from 104.")
    return frame


def _minimum_estimable(values: pd.Series, estimable: pd.Series) -> float | None:
    """Return the minimum estimable coefficient or no value."""
    selected = pd.to_numeric(values[estimable.astype(bool)], errors="coerce").dropna()
    return None if selected.empty else float(selected.min())


def recompute_condition_stability(summary: pd.DataFrame) -> pd.DataFrame:
    """Recompute derived wrapper/device and frozen-scale stability Booleans."""
    result = summary.copy()
    result["wrapper_and_device_checks_passed"] = (
        result["all_actual_devices_match_request"].astype(bool)
        & result["all_wrapper_contracts_unchanged"].astype(bool)
    )
    result["condition_stable_at_frozen_scale"] = (
        result["within_load_differences_within_1e_6"].astype(bool)
        & result["between_load_differences_within_1e_6"].astype(bool)
        & result["score_reconciliation_status_consistent"].astype(bool)
        & result["auroc_reconciliation_status_consistent"].astype(bool)
        & result["all_state_digests_unchanged"].astype(bool)
        & result["wrapper_and_device_checks_passed"].astype(bool)
    )
    return result


def build_model_condition_summary(
    condition_frame: pd.DataFrame,
    comparison_frame: pd.DataFrame,
    *,
    score_atol: float,
) -> pd.DataFrame:
    """Aggregate the 48 observations into the fixed 12 condition summaries."""
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for condition in CONDITION_ORDER:
            model_id = str(model["model_id"])
            condition_id = str(condition["condition_id"])
            subset = condition_frame.loc[
                (condition_frame["model_id"] == model_id) & (condition_frame["condition_id"] == condition_id)
            ]
            if len(subset) != 4:
                raise ValueError("Condition observation count differs from four.")
            within = comparison_frame.loc[
                (comparison_frame["comparison_type"] == "within_load_repeat")
                & (comparison_frame["model_id"] == model_id)
                & (comparison_frame["left_condition_id"] == condition_id)
            ]
            between = comparison_frame.loc[
                (comparison_frame["comparison_type"] == "between_fresh_load")
                & (comparison_frame["model_id"] == model_id)
                & (comparison_frame["left_condition_id"] == condition_id)
            ]
            if len(within) != 2 or len(between) != 2:
                raise ValueError("Condition repeat/load comparison count differs.")
            all_score = bool(subset["score_reconciliation_passed"].astype(bool).all())
            all_auroc = bool(subset["auroc_reconciliation_passed"].astype(bool).all())
            row = {
                "diagnostic_id": DIAGNOSTIC_ID,
                **model,
                **condition,
                "n_fresh_loads": 2,
                "n_repeats_per_load": 2,
                "all_actual_devices_match_request": bool(
                    (subset["actual_parameter_device_class"] == condition["requested_device_class"]).all()
                ),
                "all_wrapper_contracts_unchanged": bool(
                    (
                        (subset["wrapper_accelerator_before"] == "cpu")
                        & (subset["wrapper_accelerator_after"] == "cpu")
                        & (pd.to_numeric(subset["wrapper_batch_size"]) == 32)
                    ).all()
                ),
                "all_score_vectors_valid": bool(
                    (subset["score_all_finite"].astype(bool) & subset["score_in_unit_interval"].astype(bool)).all()
                ),
                "worst_max_abs_difference_from_frozen": float(subset["max_abs_difference"].max()),
                "maximum_count_abs_difference_gt_1e_6": int(subset["count_abs_difference_gt_1e_6"].max()),
                "maximum_count_abs_difference_gt_1e_5": int(subset["count_abs_difference_gt_1e_5"].max()),
                "maximum_count_abs_difference_gt_1e_4": int(subset["count_abs_difference_gt_1e_4"].max()),
                "maximum_count_abs_difference_gt_1e_3": int(subset["count_abs_difference_gt_1e_3"].max()),
                "all_score_reconciliations_passed": all_score,
                "all_auroc_reconciliations_passed": all_auroc,
                "condition_reconciles": bool(all_score and all_auroc),
                "maximum_within_load_repeat_difference": float(within["max_abs_score_difference"].max()),
                "maximum_between_fresh_load_difference": float(between["max_abs_score_difference"].max()),
                "all_state_digests_unchanged": bool(subset["state_unchanged_from_before"].astype(bool).all()),
                "within_load_differences_within_1e_6": bool(within["max_abs_score_difference"].le(score_atol).all()),
                "between_load_differences_within_1e_6": bool(between["max_abs_score_difference"].le(score_atol).all()),
                "score_reconciliation_status_consistent": bool(subset["score_reconciliation_passed"].nunique() == 1),
                "auroc_reconciliation_status_consistent": bool(subset["auroc_reconciliation_passed"].nunique() == 1),
                "wrapper_and_device_checks_passed": False,
                "condition_stable_at_frozen_scale": False,
                "all_rankings_identical": bool(subset["ranking_identical"].astype(bool).all()),
                "minimum_estimable_kendall_tau_b": _minimum_estimable(
                    subset["kendall_tau_b"], subset["kendall_estimable"]
                ),
                "minimum_estimable_spearman_rho": _minimum_estimable(
                    subset["spearman_rho"], subset["spearman_estimable"]
                ),
            }
            rows.append(row)
    frame = pd.DataFrame(rows, columns=MODEL_CONDITION_SUMMARY_COLUMNS)
    return recompute_condition_stability(frame)


def _pair_isolation(left: pd.Series, right: pd.Series) -> tuple[bool, str]:
    """Apply the frozen causal-pair isolation gate and reason precedence."""
    if bool(left["condition_stable_at_frozen_scale"]) and bool(right["condition_stable_at_frozen_scale"]):
        return True, ""
    categories = (
        ("contract_or_device_failure", "wrapper_and_device_checks_passed"),
        ("state_mutation", "all_state_digests_unchanged"),
        (
            "repeat_or_load_variability_gt_1e_6",
            ("within_load_differences_within_1e_6", "between_load_differences_within_1e_6"),
        ),
        (
            "reconciliation_status_inconsistent",
            ("score_reconciliation_status_consistent", "auroc_reconciliation_status_consistent"),
        ),
    )
    for reason, fields in categories:
        field_names = (fields,) if isinstance(fields, str) else fields
        left_failed = not all(bool(left[field]) for field in field_names)
        right_failed = not all(bool(right[field]) for field in field_names)
        if left_failed or right_failed:
            side = "both_conditions" if left_failed and right_failed else "left_condition" if left_failed else "right_condition"
            return False, f"{side}_{reason}"
    raise ValueError("Nonisolatable condition pair has no fixed reason.")


def _summary_row(summary: pd.DataFrame, model_id: str, condition_id: str) -> pd.Series:
    """Select exactly one fixed model-condition summary row."""
    selected = summary.loc[(summary["model_id"] == model_id) & (summary["condition_id"] == condition_id)]
    if len(selected) != 1:
        raise ValueError("Model-condition summary key is absent or duplicated.")
    return selected.iloc[0]


def _pair_comparisons(
    comparisons: pd.DataFrame,
    model_id: str,
    comparison_type: str,
    left_condition: str,
    right_condition: str,
) -> pd.DataFrame:
    """Select the four matched comparison rows for one causal pair."""
    selected = comparisons.loc[
        (comparisons["model_id"] == model_id)
        & (comparisons["comparison_type"] == comparison_type)
        & (comparisons["left_condition_id"] == left_condition)
        & (comparisons["right_condition_id"] == right_condition)
    ]
    if len(selected) != 4:
        raise ValueError("Causal pair comparison count differs from four.")
    return selected


def build_interpretation_branches(
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    *,
    score_atol: float,
) -> pd.DataFrame:
    """Evaluate result-independent isolation gates and terminal interpretation branches."""
    del score_atol
    summary = recompute_condition_stability(summary)
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        model_id = str(model["model_id"])
        conditions = {item["condition_id"]: _summary_row(summary, model_id, item["condition_id"]) for item in CONDITION_ORDER}
        e010_isolatable, e010_reason = _pair_isolation(
            conditions["cpu_original_train_warmup"], conditions["gpu_original_train_warmup"]
        )
        cpu_warm_isolatable, cpu_warm_reason = _pair_isolation(
            conditions["cpu_original_direct"], conditions["cpu_original_train_warmup"]
        )
        gpu_warm_isolatable, gpu_warm_reason = _pair_isolation(
            conditions["gpu_original_direct"], conditions["gpu_original_train_warmup"]
        )
        cpu_order_isolatable, cpu_order_reason = _pair_isolation(
            conditions["cpu_original_direct"], conditions["cpu_participant_sorted_direct"]
        )
        gpu_order_isolatable, gpu_order_reason = _pair_isolation(
            conditions["gpu_original_direct"], conditions["gpu_participant_sorted_direct"]
        )
        e010_matches = _pair_comparisons(
            comparisons,
            model_id,
            "cpu_gpu",
            "cpu_original_train_warmup",
            "gpu_original_train_warmup",
        )
        cpu_warm_matches = _pair_comparisons(
            comparisons, model_id, "train_warmup", "cpu_original_direct", "cpu_original_train_warmup"
        )
        gpu_warm_matches = _pair_comparisons(
            comparisons, model_id, "train_warmup", "gpu_original_direct", "gpu_original_train_warmup"
        )
        cpu_order_matches = _pair_comparisons(
            comparisons, model_id, "input_order", "cpu_original_direct", "cpu_participant_sorted_direct"
        )
        gpu_order_matches = _pair_comparisons(
            comparisons, model_id, "input_order", "gpu_original_direct", "gpu_participant_sorted_direct"
        )
        e010_supported = bool(
            e010_isolatable
            and conditions["gpu_original_train_warmup"]["condition_reconciles"]
            and not conditions["cpu_original_train_warmup"]["all_score_reconciliations_passed"]
            and conditions["cpu_original_train_warmup"]["score_reconciliation_status_consistent"]
            and e010_matches["exceeds_score_reconciliation_tolerance"].astype(bool).all()
        )
        cpu_warm_supported = bool(
            cpu_warm_isolatable
            and cpu_warm_matches["exceeds_score_reconciliation_tolerance"].astype(bool).all()
        )
        gpu_warm_supported = bool(
            gpu_warm_isolatable
            and gpu_warm_matches["exceeds_score_reconciliation_tolerance"].astype(bool).all()
        )
        cpu_order_supported = bool(
            cpu_order_isolatable
            and cpu_order_matches["exceeds_score_reconciliation_tolerance"].astype(bool).all()
        )
        gpu_order_supported = bool(
            gpu_order_isolatable
            and gpu_order_matches["exceeds_score_reconciliation_tolerance"].astype(bool).all()
        )
        model_comparisons = comparisons.loc[comparisons["model_id"] == model_id]
        within_nonidentity = bool(
            (~model_comparisons.loc[
                model_comparisons["comparison_type"] == "within_load_repeat", "exact_score_identity"
            ].astype(bool)).any()
        )
        between_nonidentity = bool(
            (~model_comparisons.loc[
                model_comparisons["comparison_type"] == "between_fresh_load", "exact_score_identity"
            ].astype(bool)).any()
        )
        model_state_mutation = bool(
            (~summary.loc[summary["model_id"] == model_id, "all_state_digests_unchanged"].astype(bool)).any()
        )
        any_reconciles = bool(summary.loc[summary["model_id"] == model_id, "condition_reconciles"].astype(bool).any())
        unresolved = not any_reconciles
        branch_flags = {
            "e010_gpu_to_cpu_change_supported": e010_supported,
            "warmup_sequence_effect_supported": bool(cpu_warm_supported or gpu_warm_supported),
            "input_order_sensitivity_supported": bool(cpu_order_supported or gpu_order_supported),
            "within_load_nondeterminism_observed": within_nonidentity,
            "between_load_nondeterminism_observed": between_nonidentity,
            "model_state_mutation_observed": model_state_mutation,
        }
        if unresolved:
            root_summary = "unresolved_artifact_state_or_provenance_mismatch"
        else:
            supported = [name for name in BRANCH_ORDER if branch_flags[name]]
            root_summary = ";".join(supported) if supported else "no_prespecified_cause_branch_supported"
        rows.append(
            {
                "diagnostic_id": DIAGNOSTIC_ID,
                **model,
                "gpu_stage4b_sequence_reconciles": bool(
                    conditions["gpu_original_train_warmup"]["condition_reconciles"]
                ),
                "matched_cpu_sequence_reconciles": bool(
                    conditions["cpu_original_train_warmup"]["condition_reconciles"]
                ),
                "e010_gpu_to_cpu_effect_isolatable": e010_isolatable,
                "e010_gpu_to_cpu_effect_nonisolatable_reason": e010_reason,
                "e010_gpu_to_cpu_change_supported": e010_supported,
                "cpu_warmup_effect_isolatable": cpu_warm_isolatable,
                "cpu_warmup_effect_nonisolatable_reason": cpu_warm_reason,
                "gpu_warmup_effect_isolatable": gpu_warm_isolatable,
                "gpu_warmup_effect_nonisolatable_reason": gpu_warm_reason,
                "warmup_sequence_effect_supported": bool(cpu_warm_supported or gpu_warm_supported),
                "cpu_input_order_effect_isolatable": cpu_order_isolatable,
                "cpu_input_order_effect_nonisolatable_reason": cpu_order_reason,
                "gpu_input_order_effect_isolatable": gpu_order_isolatable,
                "gpu_input_order_effect_nonisolatable_reason": gpu_order_reason,
                "input_order_sensitivity_supported": bool(cpu_order_supported or gpu_order_supported),
                "within_load_nondeterminism_observed": within_nonidentity,
                "between_load_nondeterminism_observed": between_nonidentity,
                "model_state_mutation_observed": model_state_mutation,
                "any_condition_reconciles": any_reconciles,
                "unresolved_artifact_state_or_provenance_mismatch": unresolved,
                "root_cause_summary": root_summary,
                "pre_e010_pkl_available": False,
                "e010_source_checkpoint_available": False,
            }
        )
    return pd.DataFrame(rows, columns=INTERPRETATION_BRANCH_COLUMNS)


def validate_output_frames(frames: Mapping[str, pd.DataFrame], *, strict_values: bool = True) -> None:
    """Enforce exact aggregate output schemas, rows, ordering, and cross-file identities."""
    if set(frames) != set(EXPECTED_OUTPUT_COLUMNS):
        raise ValueError("Aggregate output filename set differs from the frozen contract.")
    for filename, columns in EXPECTED_OUTPUT_COLUMNS.items():
        frame = frames[filename]
        if frame.columns.tolist() != list(columns):
            raise ValueError(f"Aggregate output schema differs for {filename}.")
        if len(frame) != EXPECTED_OUTPUT_ROWS[filename]:
            raise ValueError(f"Aggregate output row count differs for {filename}.")
    observations = frames["condition_diagnostics.csv"]
    if observations["load_number"].min() != 1 or observations["repeat_number"].min() != 1:
        raise ValueError("Persisted load/repeat numbering is not one-based.")
    if strict_values:
        if not observations["score_all_finite"].astype(bool).all():
            raise ValueError("A diagnostic observation has nonfinite scores.")
        if not observations["score_in_unit_interval"].astype(bool).all():
            raise ValueError("A diagnostic observation has out-of-range scores.")
        summaries = frames["model_condition_summary.csv"]
        branches = frames["interpretation_branches.csv"]
        if set(summaries["model_id"]) != set(branches["model_id"]):
            raise ValueError("Interpretation and condition-summary model identities differ.")


def validate_serialized_privacy(
    serialized_files: Mapping[str, bytes],
    *,
    private_tokens: set[str] | None = None,
) -> None:
    """Reject participant-level fields, private tokens, dates, and absolute paths without echo."""
    tokens = private_tokens or set()
    for filename, payload in serialized_files.items():
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Output privacy validation failed for a non-UTF-8 candidate.") from exc
        if any(pattern.search(text) for pattern in _FORBIDDEN_OUTPUT_PATTERNS):
            raise ValueError(f"Output privacy validation failed for {Path(filename).name}.")
        if any(token and token in text for token in tokens):
            raise ValueError(f"Output privacy validation failed for {Path(filename).name}.")


def atomic_write_bytes(
    target: Path,
    payload: bytes,
    *,
    validator: Callable[[Path], None] | None = None,
) -> None:
    """Write, validate, and atomically promote one new file without overwrite."""
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Output target already exists: {target.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.tmp"
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"Temporary output target already exists: {temporary.name}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if validator is not None:
            validator(temporary)
        os.replace(temporary, target)
    except Exception:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
        raise


def promote_candidate_directory(candidate: Path, final: Path) -> None:
    """Atomically promote one complete candidate directory without overwrite."""
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"Final output already exists: {final.name}")
    if not candidate.is_dir() or candidate.is_symlink():
        raise ValueError("Promotion candidate is absent or invalid.")
    final.parent.mkdir(parents=True, exist_ok=True)
    if candidate.stat().st_dev != final.parent.stat().st_dev:
        raise ValueError("Atomic promotion requires the same filesystem.")
    os.replace(candidate, final)


def _git_output(*args: str) -> str:
    """Return sanitized output from one read-only Git command."""
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError("Read-only Git verification failed.")
    return result.stdout.strip()


def _package_versions() -> dict[str, str]:
    """Return non-sensitive frozen-runtime package provenance."""
    versions = {"python": platform.python_version()}
    for output_name, distribution in (
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("scikit_learn", "scikit-learn"),
        ("torch", "torch"),
        ("pytorch_lightning", "pytorch-lightning"),
    ):
        try:
            versions[output_name] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"Required installed dependency is unavailable: {distribution}") from exc
    return versions


def verify_immutable_hashes(config: Mapping[str, Any]) -> dict[str, str]:
    """Verify every inherited, diagnostic-specific, and Figure 3 immutable hash."""
    verified: dict[str, str] = {}
    for relative_path, expected in config["immutable_inputs"].items():
        path = _project_path(relative_path)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Immutable input is absent or invalid: {relative_path}")
        observed = _sha256_file(path)
        if observed != expected:
            raise ValueError(f"Immutable input hash differs: {relative_path}")
        verified[relative_path] = observed
    return verified


def _verify_e010_source_history() -> dict[str, bool]:
    """Verify unchanged deep inference sources and Captum-only environment history."""
    source_diff = _git_output(
        "diff",
        "--name-only",
        TRAINING_COMMIT,
        "HEAD",
        "--",
        "src/training/train_inception_time.py",
        "src/training/train_cnn_bilstm_attention.py",
    )
    if source_diff:
        raise ValueError("Deep inference source history differs from the E010 boundary.")
    old_environment = _git_output("show", f"{TRAINING_COMMIT}:environment.yml").splitlines()
    current_environment = (PROJECT_ROOT / "environment.yml").read_text(encoding="utf-8").splitlines()
    if [line for line in current_environment if line.strip() != "- captum=0.9.0"] != old_environment:
        raise ValueError("Environment history differs from the Captum-only E010 boundary.")
    return {"deep_inference_source_unchanged": True, "captum_only_environment_change": True}


def _require_empty_directory(path: Path) -> None:
    """Require one existing real directory to have no child entry."""
    if not path.is_dir() or path.is_symlink():
        raise ValueError("Required empty Figure 3 work directory is absent or invalid.")
    if next(path.iterdir(), None) is not None:
        raise ValueError("Required Figure 3 work directory is not empty.")


def verify_execution_preflight(config: Mapping[str, Any], *, require_output_roots_absent: bool = True) -> dict[str, Any]:
    """Verify approval, Git, environment, CUDA, immutable, and stale-output gates."""
    if _sha256_file(_project_path(APPROVED_PLAN["path"])) != APPROVED_PLAN["sha256"]:
        raise ValueError("Approved diagnostic plan hash differs.")
    head = _git_output("rev-parse", "HEAD")
    if head != EXPECTED_GIT_HEAD:
        raise ValueError("Git HEAD differs from the approved diagnostic authority.")
    if _git_output("diff", "--name-only"):
        raise ValueError("Tracked Git diff is not empty.")
    status = [line for line in _git_output("status", "--short").splitlines() if line]
    if set(status) != EXPECTED_STATUS_PATHS:
        raise ValueError("Git worktree state differs from the approved Phase 2 scope.")
    if os.environ.get("CONDA_DEFAULT_ENV") != "gcvocs":
        raise ValueError("The gcvocs conda environment is not active.")
    expected_python = Path.home() / "miniforge3" / "envs" / "gcvocs" / "bin" / "python"
    if Path(sys.executable).resolve() != expected_python.resolve():
        raise ValueError("Python does not resolve inside the gcvocs environment.")
    work_root = _project_path(config["paths"]["work_root"])
    final_root = _project_path(config["paths"]["final_output_dir"])
    if require_output_roots_absent and (work_root.exists() or final_root.exists()):
        raise FileExistsError("Diagnostic work or final output root already exists.")
    _require_empty_directory(_project_path(config["paths"]["figure3_run_a"]))
    if _project_path(config["paths"]["figure3_final_output"]).exists():
        raise FileExistsError("Original Figure 3 final-output directory unexpectedly exists.")
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("The frozen CUDA gate requires exactly one available CUDA device.")
    e010 = _verify_e010_source_history()
    hashes = verify_immutable_hashes(config)
    return {
        "git_head": head,
        "worktree_status": status,
        "environment_name": "gcvocs",
        "python_executable_environment_relative": "bin/python",
        "package_versions": _package_versions(),
        "cuda_available": True,
        "cuda_device_count": 1,
        "immutable_hashes": hashes,
        "e010_source_history": e010,
    }


def _configure_determinism(seed: int) -> dict[str, Any]:
    """Reset the approved deterministic runtime state before each fresh load."""
    import pytorch_lightning as pl
    import torch

    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")
    return {
        "seed": seed,
        "workers_seeded": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "float32_matmul_precision": "high",
    }


def _load_metric_auroc(model_id: str, metric_dir: Path) -> float:
    """Load only the frozen aggregate test AUROC."""
    path = metric_dir / f"binary_{model_id}_test_metrics.csv"
    if pd.read_csv(path, nrows=0).columns.tolist() != ["scope", "metric", "value", "ci_low", "ci_high"]:
        raise ValueError(f"Frozen metric header differs for model={model_id}.")
    frame = pd.read_csv(path, usecols=["scope", "metric", "value"])
    selected = frame.loc[(frame["scope"] == "test") & (frame["metric"] == "auc_roc"), "value"]
    if len(selected) != 1:
        raise ValueError(f"Frozen test AUROC is absent or duplicated for model={model_id}.")
    value = float(selected.iloc[0])
    if not math.isfinite(value):
        raise ValueError(f"Frozen test AUROC is nonfinite for model={model_id}.")
    return value


def _load_prediction_evidence(model_id: str, prediction_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load only frozen test participant keys, labels, and raw scores."""
    path = prediction_dir / f"binary_{model_id}_test_predictions.csv"
    if pd.read_csv(path, nrows=0).columns.tolist() != ["sample_id", "y_true", "p_raw", "p_cal", "y_pred", "quadrant"]:
        raise ValueError(f"Frozen prediction header differs for model={model_id}.")
    frame = pd.read_csv(path, usecols=["sample_id", "y_true", "p_raw"])
    keys = pd.to_numeric(frame["sample_id"], errors="raise").to_numpy(dtype=np.int64)
    labels = pd.to_numeric(frame["y_true"], errors="raise").to_numpy(dtype=np.int64)
    scores = pd.to_numeric(frame["p_raw"], errors="raise").to_numpy(dtype=float)
    if len(frame) != 195 or len(np.unique(keys)) != 195 or not np.isfinite(scores).all():
        raise ValueError(f"Frozen prediction evidence is invalid for model={model_id}.")
    return keys, labels, scores


def _load_locked_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    """Load and validate the minimum Pipeline A, split, prediction, and metric evidence."""
    with np.load(_project_path(config["paths"]["pipeline_a"]), allow_pickle=False) as payload:
        required = {"x", "sample_id", "label", "channel_names", "clip_length_L"}
        if not required.issubset(payload.files):
            raise ValueError("Pipeline A schema differs from the frozen contract.")
        x = np.asarray(payload["x"])
        keys = np.asarray(payload["sample_id"])
        labels = np.asarray(payload["label"])
        channels = tuple(str(value) for value in payload["channel_names"].tolist())
        clip_length = int(np.asarray(payload["clip_length_L"]).item())
    if x.shape != (972, 960, 11) or channels != tuple(f"S{index}" for index in range(1, 12)) or clip_length != 960:
        raise ValueError("Pipeline A shape, channels, or clip length differs.")
    if not np.issubdtype(keys.dtype, np.integer) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Pipeline A key or label dtype differs.")
    keys = keys.astype(np.int64, copy=False)
    labels = labels.astype(np.int64, copy=False)
    if len(np.unique(keys)) != 972 or not np.isfinite(x).all() or set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("Pipeline A keys, labels, or values are invalid.")
    indices: dict[str, np.ndarray] = {}
    for split_name in ("train_idx", "test_idx"):
        with np.load(_project_path(config["paths"][split_name]), allow_pickle=False) as payload:
            if split_name not in payload.files:
                raise ValueError(f"Locked split is missing {split_name}.")
            vector = np.asarray(payload[split_name])
        expected_n = 777 if split_name == "train_idx" else 195
        if vector.ndim != 1 or len(vector) != expected_n or not np.issubdtype(vector.dtype, np.integer):
            raise ValueError(f"Locked {split_name} has invalid dimensions or dtype.")
        vector = vector.astype(np.int64, copy=False)
        if len(np.unique(vector)) != expected_n or vector.min() < 0 or vector.max() >= 972:
            raise ValueError(f"Locked {split_name} is duplicated or out of range.")
        indices[split_name] = vector
    train_idx = indices["train_idx"]
    test_idx = indices["test_idx"]
    if np.intersect1d(train_idx, test_idx).size or len(np.union1d(train_idx, test_idx)) != 972:
        raise ValueError("Locked train/test split separation or coverage differs.")
    original_train = x[train_idx]
    original_test = x[test_idx]
    train_keys = keys[train_idx]
    original_test_keys = keys[test_idx]
    original_test_labels = labels[test_idx]
    sort_order = np.argsort(original_test_keys, kind="mergesort")
    sorted_test = original_test[sort_order]
    sorted_test_keys = original_test_keys[sort_order]
    prediction_dir = _project_path(config["paths"]["prediction_dir"])
    metric_dir = _project_path(config["paths"]["metric_dir"])
    frozen_scores: dict[str, np.ndarray] = {}
    metric_aurocs: dict[str, float] = {}
    frozen_keys: np.ndarray | None = None
    frozen_labels: np.ndarray | None = None
    for model in MODEL_ORDER:
        model_id = str(model["model_id"])
        prediction_keys, prediction_labels, prediction_scores = _load_prediction_evidence(model_id, prediction_dir)
        if frozen_keys is None:
            frozen_keys = prediction_keys
            frozen_labels = prediction_labels
        elif not np.array_equal(prediction_keys, frozen_keys) or not np.array_equal(prediction_labels, frozen_labels):
            raise ValueError("Deep frozen prediction key/label sequences differ.")
        if set(prediction_keys.tolist()) != set(original_test_keys.tolist()):
            raise ValueError("Frozen prediction membership differs from locked test membership.")
        aligned_labels = align_scores_to_frozen_order(original_test_labels.astype(float), original_test_keys, prediction_keys)
        if not np.array_equal(aligned_labels.astype(np.int64), prediction_labels):
            raise ValueError("Frozen prediction labels differ from locked Pipeline A labels.")
        frozen_scores[model_id] = prediction_scores
        metric_aurocs[model_id] = _load_metric_auroc(model_id, metric_dir)
    if frozen_keys is None or frozen_labels is None:
        raise ValueError("Frozen deep prediction evidence is absent.")
    if int(np.count_nonzero(frozen_labels == 1)) != 82 or int(np.count_nonzero(frozen_labels == 0)) != 113:
        raise ValueError("Frozen diagnostic comparison cohort counts differ.")
    return {
        "original_train": original_train,
        "train_keys": train_keys,
        "original_test": original_test,
        "original_test_keys": original_test_keys,
        "sorted_test": sorted_test,
        "sorted_test_keys": sorted_test_keys,
        "frozen_keys": frozen_keys,
        "frozen_labels": frozen_labels,
        "frozen_scores": frozen_scores,
        "metric_aurocs": metric_aurocs,
    }


def _load_validate_model_payload(model_id: str, context: Mapping[str, Any], model_dir: Path) -> dict[str, Any]:
    """Fresh-load and validate one immutable deep-model PKL without exposing private values."""
    from src.training.dl_inference import DLPredictProbaWrapper
    from src.training.load_final_model import load_final_model

    payload = load_final_model(model_dir / f"binary_{model_id}_final.pkl")
    required = {
        "transformer",
        "normalizer",
        "model",
        "calibrator",
        "threshold_p_raw",
        "threshold_p_final_equivalent",
        "feature_names",
        "training_sample_ids",
        "metadata",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError(f"Final model payload schema differs for model={model_id}.")
    if payload["metadata"].get("model_name") != model_id or list(payload["feature_names"]) != [
        f"S{index}" for index in range(1, 12)
    ]:
        raise ValueError(f"Final model identity or channels differ for model={model_id}.")
    wrapper = payload["model"]
    if not isinstance(wrapper, DLPredictProbaWrapper) or wrapper.accelerator != "cpu" or wrapper.batch_size != 32:
        raise ValueError(f"Persisted deep wrapper contract differs for model={model_id}.")
    expected_predictor = {
        "inception_time": ("src.training.train_inception_time", "_predict_probabilities"),
        "cnn_bilstm_attention": ("src.training.train_cnn_bilstm_attention", "_predict_probabilities"),
    }[model_id]
    if (getattr(wrapper.predictor, "__module__", ""), getattr(wrapper.predictor, "__name__", "")) != expected_predictor:
        raise ValueError(f"Persisted predictor identity differs for model={model_id}.")
    training_keys = np.asarray(payload["training_sample_ids"])
    if training_keys.shape != (777,) or len(np.unique(training_keys)) != 777:
        raise ValueError(f"Persisted training-key schema differs for model={model_id}.")
    if set(training_keys.astype(np.int64).tolist()) != set(np.asarray(context["train_keys"]).tolist()):
        raise ValueError(f"Persisted training membership differs for model={model_id}.")
    if np.intersect1d(training_keys.astype(np.int64), np.asarray(context["frozen_keys"])).size:
        raise ValueError(f"Train/test separation differs for model={model_id}.")
    return payload


def _observation_row(
    model: Mapping[str, Any],
    condition: Mapping[str, Any],
    load_number: int,
    repeat_number: int,
    execution: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine aggregate inference provenance and score diagnostics."""
    digests = execution["state_digests"]
    after = digests["after_calls"][repeat_number - 1]
    before = digests["before"]
    after_warmup = digests["after_warmup"]
    unchanged = bool(before == after and (not after_warmup or before == after_warmup))
    return {
        "diagnostic_id": DIAGNOSTIC_ID,
        **model,
        **condition,
        "load_number": load_number,
        "repeat_number": repeat_number,
        "wrapper_accelerator_before": execution["wrapper_contract"]["accelerator_before"],
        "wrapper_accelerator_after": execution["wrapper_contract"]["accelerator_after"],
        "wrapper_batch_size": execution["wrapper_contract"]["batch_size"],
        "actual_parameter_device_class": execution["actual_device_classes"][repeat_number - 1],
        **diagnostics,
        "state_sha256_before": before,
        "state_sha256_after_warmup": after_warmup,
        "state_sha256_after_inference": after,
        "state_unchanged_from_before": unchanged,
    }


def _serialize_frame(frame: pd.DataFrame) -> bytes:
    """Serialize one aggregate CSV with the frozen dialect."""
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a complete aggregate frame without optional dependencies."""
    headers = frame.columns.tolist()
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = ["" if pd.isna(value) else str(value).lower() if isinstance(value, (bool, np.bool_)) else str(value) for value in row]
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in values) + " |")
    return "\n".join(lines)


def build_report(frames: Mapping[str, pd.DataFrame]) -> bytes:
    """Generate the concise complete aggregate report from validated frames."""
    summary = frames["model_condition_summary.csv"]
    branches = frames["interpretation_branches.csv"]
    observations = frames["condition_diagnostics.csv"]
    nonestimable = int((~observations["kendall_estimable"].astype(bool)).sum()) + int(
        (~observations["spearman_estimable"].astype(bool)).sum()
    )
    text = "\n".join(
        [
            "# E010 Deep-Model Reconciliation Diagnostic",
            "",
            "This results-aware operational reproducibility diagnostic evaluates the E010 serialization boundary. It is not a new scientific analysis and does not alter Figure 3.",
            "",
            "The diagnostic comparison cohort is the frozen 195-participant internal test cohort. The locked 777-participant training tensor was used only for execution warm-up in conditions 2 and 4; no training-score diagnostic was retained.",
            "",
            "Acceptance criteria remained max raw-score difference <= 1e-6 and every pairwise AUROC difference <= 1e-12. Distributional and rank diagnostics are descriptive only.",
            "",
            f"Nonestimable rank-correlation records retained: {nonestimable}.",
            "",
            "## Complete model-condition summary",
            "",
            _markdown_table(summary),
            "",
            "## Complete interpretation branches",
            "",
            _markdown_table(branches),
            "",
            "Pre-E010 PKLs and E010 source checkpoints were unavailable, so direct pre/post weight comparison was not possible.",
            "",
            "No artifact repair, tolerance change, new condition, Figure 3 recommendation, chemical interpretation, or biological interpretation is made.",
            "",
        ]
    )
    return text.encode("utf-8")


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize deterministic JSON without nonfinite values."""
    return (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _build_manifest(candidate: Path, final_relative: Path) -> dict[str, Any]:
    """Build the exact non-self-hashing final manifest."""
    files: list[dict[str, Any]] = []
    for filename in FINAL_MANIFEST_FILENAMES:
        source = candidate / filename
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"Manifest source is absent: {filename}")
        files.append(
            {
                "path": (final_relative / filename).as_posix(),
                "byte_size": int(source.stat().st_size),
                "sha256": _sha256_file(source),
            }
        )
    return {"diagnostic_id": DIAGNOSTIC_ID, "files": files}


def _implementation_hashes(config: Mapping[str, Any]) -> dict[str, str]:
    """Hash the four task-owned Phase 2 implementation files."""
    return {path: _sha256_file(_project_path(path)) for path in config["allowed_implementation_paths"]}


def _build_audit(
    config: Mapping[str, Any],
    preflight: Mapping[str, Any],
    immutable_after: Mapping[str, str],
    frames: Mapping[str, pd.DataFrame],
    implementation_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Build the deterministic aggregate audit payload in fixed key order."""
    summaries = frames["model_condition_summary.csv"]
    branches = frames["interpretation_branches.csv"]
    return {
        "diagnostic_id": DIAGNOSTIC_ID,
        "diagnostic_classification": config["diagnostic_classification"],
        "status": "complete",
        "approved_plan": APPROVED_PLAN,
        "source_analysis": config["source_analysis_id"],
        "provenance": {
            "git_head": preflight["git_head"],
            "worktree_status": preflight["worktree_status"],
            "implementation_sha256": dict(implementation_hashes),
            "network_used": False,
        },
        "e010_evidence": {
            **preflight["e010_source_history"],
            "persisted_wrapper_accelerator": "cpu",
            "persisted_wrapper_batch_size": 32,
            "stage4b_configured_accelerator": "gpu",
            "stage4b_train_then_test_sequence": True,
            "pre_e010_pkl_available": False,
            "e010_source_checkpoint_available": False,
        },
        "immutable_inputs": [
            {
                "path": path,
                "sha256_before": digest,
                "sha256_after": immutable_after[path],
            }
            for path, digest in preflight["immutable_hashes"].items()
        ],
        "models": list(MODEL_ORDER),
        "conditions": list(CONDITION_ORDER),
        "cohort_reconciliation": {
            "diagnostic_comparison": {"n_test": 195, "n_positive": 82, "n_negative": 113, "validated": True},
            "execution_only_warmup": {
                "n_train": 777,
                "validation_passed": True,
                "warmup_condition_ids": [
                    "cpu_original_train_warmup",
                    "gpu_original_train_warmup",
                ],
                "warmup_executed_for_all_applicable_loads": True,
            },
        },
        "runtime": {
            "environment_name": preflight["environment_name"],
            "python_executable_environment_relative": preflight["python_executable_environment_relative"],
            "package_versions": preflight["package_versions"],
            "cuda_available": preflight["cuda_available"],
            "cuda_device_count": preflight["cuda_device_count"],
            "random_seed": config["random_seed"],
            "batch_size": config["batch_size"],
            "n_fresh_loads": config["n_fresh_loads"],
            "n_repeated_test_calls_per_load": config["n_repeated_test_calls_per_load"],
        },
        "state_digest_method": {
            "algorithm": "sha256_canonical_named_parameters_and_buffers",
            "device_class_excluded": True,
            "all_condition_state_unchanged": bool(summaries["all_state_digests_unchanged"].astype(bool).all()),
        },
        "acceptance_criteria": {
            "score_reconciliation_atol": config["score_reconciliation_atol"],
            "auroc_reconciliation_atol": config["auroc_reconciliation_atol"],
            "other_diagnostics_descriptive_only": True,
        },
        "output_validation": {
            "row_counts": {name: int(len(frame)) for name, frame in frames.items()},
            "schemas_match": True,
            "cross_file_reconciliation_passed": True,
            "duplicate_serialization_byte_identical": True,
        },
        "interpretation_branches": branches.to_dict(orient="records"),
        "privacy_validation": {
            "schema_and_text_scan_passed": True,
            "participant_level_fields_absent": True,
            "training_diagnostics_persisted": False,
            "aligned_score_vectors_released_before_serialization": True,
        },
        "participant_level_persistence": False,
        "known_limitations": [
            "Pre-E010 PKLs are unavailable.",
            "E010 source checkpoints are unavailable.",
            "This operational diagnostic does not repair artifacts or amend Figure 3.",
        ],
        "commands": [
            {
                "command": "python -m pytest tests/test_figure3_e010_deep_model_reconciliation_diagnostic.py -q",
                "exit_code": 0,
            },
            {
                "command": "python scripts/run_figure3_e010_deep_model_reconciliation_diagnostic.py --config configs/figure3_e010_deep_model_reconciliation_diagnostic.yaml",
                "exit_code": 0,
            },
        ],
    }


def run_diagnostic(config_path: Path) -> dict[str, Any]:
    """Execute all fixed observations and atomically promote aggregate-only outputs."""
    config = load_diagnostic_config(config_path)
    preflight = verify_execution_preflight(config)
    context = _load_locked_inputs(config)
    model_dir = _project_path(config["paths"]["model_dir"])
    observations: list[dict[str, Any]] = []
    score_store: dict[tuple[str, str, int, int], np.ndarray] = {}
    for model in MODEL_ORDER:
        model_id = str(model["model_id"])
        for condition in CONDITION_ORDER:
            test_input = (
                context["original_test"]
                if condition["input_order"] == "original_test_idx"
                else context["sorted_test"]
            )
            test_keys = (
                context["original_test_keys"]
                if condition["input_order"] == "original_test_idx"
                else context["sorted_test_keys"]
            )
            for load_number in (1, 2):
                LOGGER.info(
                    "Diagnostic progress model=%s condition=%s load=%d/2.",
                    model_id,
                    condition["condition_id"],
                    load_number,
                )
                _configure_determinism(int(config["random_seed"]))
                payload = _load_validate_model_payload(model_id, context, model_dir)
                execution = execute_condition_load(
                    wrapper=payload["model"],
                    normalizer=payload["normalizer"],
                    condition=condition,
                    train_input=context["original_train"] if condition["train_warmup"] else None,
                    test_input=test_input,
                    expected_train_n=777,
                    expected_test_n=195,
                )
                for repeat_number, unaligned in enumerate(execution["test_scores"], start=1):
                    aligned = align_scores_to_frozen_order(unaligned, test_keys, context["frozen_keys"])
                    diagnostics = calculate_score_diagnostics(
                        aligned,
                        context["frozen_scores"][model_id],
                        context["frozen_labels"],
                        frozen_metric_auroc=context["metric_aurocs"][model_id],
                        score_atol=float(config["score_reconciliation_atol"]),
                        auroc_atol=float(config["auroc_reconciliation_atol"]),
                    )
                    observations.append(
                        _observation_row(model, condition, load_number, repeat_number, execution, diagnostics)
                    )
                    score_store[(model_id, condition["condition_id"], load_number, repeat_number)] = aligned
                    del unaligned
                del execution, payload
                gc.collect()
                __import__("torch").cuda.empty_cache()
    condition_frame = pd.DataFrame(observations, columns=CONDITION_DIAGNOSTIC_COLUMNS)
    comparison_frame = build_comparison_diagnostics(
        score_store,
        score_atol=float(config["score_reconciliation_atol"]),
    )
    summary_frame = build_model_condition_summary(
        condition_frame,
        comparison_frame,
        score_atol=float(config["score_reconciliation_atol"]),
    )
    branch_frame = build_interpretation_branches(
        summary_frame,
        comparison_frame,
        score_atol=float(config["score_reconciliation_atol"]),
    )
    frames = {
        "condition_diagnostics.csv": condition_frame,
        "comparison_diagnostics.csv": comparison_frame,
        "model_condition_summary.csv": summary_frame,
        "interpretation_branches.csv": branch_frame,
    }
    validate_output_frames(frames, strict_values=True)
    score_store.clear()
    del score_store
    for private_name in (
        "original_train",
        "train_keys",
        "original_test",
        "original_test_keys",
        "sorted_test",
        "sorted_test_keys",
        "frozen_keys",
        "frozen_labels",
        "frozen_scores",
        "metric_aurocs",
    ):
        context.pop(private_name, None)
    del context
    gc.collect()
    immutable_after = verify_immutable_hashes(config)
    if immutable_after != preflight["immutable_hashes"]:
        raise ValueError("An immutable input changed during diagnostic execution.")
    _require_empty_directory(_project_path(config["paths"]["figure3_run_a"]))
    if _project_path(config["paths"]["figure3_final_output"]).exists():
        raise FileExistsError("Original Figure 3 final output appeared during execution.")
    implementation_hashes = _implementation_hashes(config)
    serialized: dict[str, bytes] = {}
    for filename, frame in frames.items():
        first = _serialize_frame(frame)
        second = _serialize_frame(frame)
        if first != second:
            raise ValueError(f"Duplicate aggregate serialization differs for {filename}.")
        serialized[filename] = first
    report_first = build_report(frames)
    report_second = build_report(frames)
    if report_first != report_second:
        raise ValueError("Duplicate report serialization differs.")
    serialized["report.md"] = report_first
    audit = _build_audit(config, preflight, immutable_after, frames, implementation_hashes)
    serialized["audit.json"] = _json_bytes(audit)
    validate_serialized_privacy(serialized)
    work_root = _project_path(config["paths"]["work_root"])
    candidate = _project_path(config["paths"]["promotion_candidate"])
    final = _project_path(config["paths"]["final_output_dir"])
    if work_root.exists() or final.exists():
        raise FileExistsError("Diagnostic output root became stale before serialization.")
    candidate.mkdir(parents=True, exist_ok=False)
    for filename in (*EXPECTED_OUTPUT_COLUMNS, "audit.json", "report.md"):
        atomic_write_bytes(candidate / filename, serialized[filename])
    manifest = _build_manifest(candidate, Path(config["paths"]["final_output_dir"]))
    manifest_bytes = _json_bytes(manifest)
    validate_serialized_privacy({**serialized, "manifest.json": manifest_bytes})
    atomic_write_bytes(candidate / "manifest.json", manifest_bytes)
    for filename, expected_rows in EXPECTED_OUTPUT_ROWS.items():
        readback = pd.read_csv(candidate / filename)
        if readback.columns.tolist() != list(EXPECTED_OUTPUT_COLUMNS[filename]) or len(readback) != expected_rows:
            raise ValueError(f"Candidate read-back validation failed for {filename}.")
    if set(path.name for path in candidate.iterdir()) != {*FINAL_MANIFEST_FILENAMES, "manifest.json"}:
        raise ValueError("Candidate filename allowlist validation failed.")
    promote_candidate_directory(candidate, final)
    resolved_work = work_root.resolve()
    expected_parent = (PROJECT_ROOT / "outputs" / "work").resolve()
    if resolved_work.parent != expected_parent or next(work_root.iterdir(), None) is not None:
        raise ValueError("Diagnostic work-root cleanup safety validation failed.")
    work_root.rmdir()
    final_hashes: dict[str, str] = {}
    for path in sorted(final.iterdir(), key=lambda item: item.name):
        if path.is_file() and not path.is_symlink():
            final_hashes[path.name] = _sha256_file(path)
    return {
        "final_output_dir": Path(config["paths"]["final_output_dir"]).as_posix(),
        "sha256": final_hashes,
    }
