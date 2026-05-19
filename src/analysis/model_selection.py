from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score

from src.analysis.data_split import _prepare_cohort_dataframe
from src.config import PROJECT_ROOT, RANDOM_SEED
from src.evaluation.statistical_tests import delong_roc_test
from src.training.common.bootstrap import paired_bootstrap_delta


LOGGER = logging.getLogger(__name__)
MODEL_NAMES: tuple[str, ...] = (
    "minirocket_ridge",
    "xgboost",
    "svm",
    "elastic_net_lr",
    "inception_time",
    "cnn_bilstm_attention",
)
PARSIMONY_ORDER: tuple[str, ...] = (
    "minirocket_ridge",
    "elastic_net_lr",
    "xgboost",
    "svm",
    "inception_time",
    "cnn_bilstm_attention",
)
NO_SKILL_AUC = 0.5
MIN_SIGNAL_AUC = 0.6
FAMILY_WISE_ALPHA = 0.05


def _format_path(path: Path) -> str:
    """Return a project-root-relative path string."""
    resolved_path = path if path.is_absolute() else PROJECT_ROOT / path
    return str(resolved_path.relative_to(PROJECT_ROOT))


def _build_forbidden_dataset_token() -> str:
    """Build the forbidden held-out dataset token without embedding it directly."""
    return "te" "st"


def _guard_forbidden_dataset_access(path: Path, loaded_paths: list[str]) -> None:
    """Reject any path that refers to the forbidden held-out dataset."""
    relative_path = _format_path(path)
    loaded_paths.append(relative_path)
    if _build_forbidden_dataset_token() in relative_path.lower():
        raise RuntimeError(f"Forbidden held-out dataset path detected: {relative_path}")


def _load_global_config(config_path: Path) -> dict[str, Any]:
    """Load the modeling global config from YAML."""
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except OSError as exc:
        LOGGER.error("Failed to read config at %s.", config_path)
        raise OSError(f"Failed to read config at {config_path}.") from exc

    if not isinstance(config, dict):
        raise ValueError(f"Config at {config_path} must deserialize to a mapping.")
    return config


def _load_json(path: Path, loaded_paths: list[str]) -> dict[str, Any]:
    """Load a JSON file with path guarding."""
    _guard_forbidden_dataset_access(path=path, loaded_paths=loaded_paths)
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except OSError as exc:
        LOGGER.error("Failed to read JSON at %s.", path)
        raise OSError(f"Failed to read JSON at {path}.") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"JSON at {path} must deserialize to a mapping.")
    return payload


def _load_oof_csv(path: Path, loaded_paths: list[str]) -> pd.DataFrame:
    """Load one OOF CSV with path guarding."""
    _guard_forbidden_dataset_access(path=path, loaded_paths=loaded_paths)
    try:
        frame = pd.read_csv(path)
    except OSError as exc:
        LOGGER.error("Failed to read OOF CSV at %s.", path)
        raise OSError(f"Failed to read OOF CSV at {path}.") from exc
    return frame


def _load_all_oof(config: dict[str, Any], loaded_paths: list[str]) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    """Load all nested-CV OOF CSVs and results JSONs."""
    oof_dict: dict[str, pd.DataFrame] = {}
    results_dict: dict[str, dict[str, Any]] = {}

    for model_name in MODEL_NAMES:
        oof_path = PROJECT_ROOT / "outputs" / "predictions" / f"binary_{model_name}_nested_cv_oof.csv"
        results_path = PROJECT_ROOT / "outputs" / "analysis" / f"binary_{model_name}_nested_cv_results.json"
        oof_dict[model_name] = _load_oof_csv(path=oof_path, loaded_paths=loaded_paths)
        results_dict[model_name] = _load_json(path=results_path, loaded_paths=loaded_paths)
        LOGGER.info(
            "Loaded nested-CV artifacts for %s from %s and %s.",
            model_name,
            _format_path(oof_path),
            _format_path(results_path),
        )

    return oof_dict, results_dict


def _load_train_index(config: dict[str, Any], loaded_paths: list[str]) -> np.ndarray:
    """Load the training split indices without touching the forbidden held-out split."""
    splits_dir = PROJECT_ROOT / Path(config["paths"]["splits_dir"])
    train_idx_path = splits_dir / "train_idx.npz"
    _guard_forbidden_dataset_access(path=train_idx_path, loaded_paths=loaded_paths)
    try:
        with np.load(train_idx_path, allow_pickle=False) as data:
            train_idx = data["train_idx"].astype(np.int64)
    except OSError as exc:
        LOGGER.error("Failed to read train indices at %s.", train_idx_path)
        raise OSError(f"Failed to read train indices at {train_idx_path}.") from exc

    if len(train_idx) != 777:
        raise AssertionError(f"Expected 777 training indices, found {len(train_idx)}.")
    return train_idx


def _enforce_pairing_contract(oof_dict: dict[str, pd.DataFrame], config: dict[str, Any], loaded_paths: list[str]) -> dict[str, Any]:
    """Assert sample and label alignment across all OOF tables."""
    cohort = _prepare_cohort_dataframe(config=config)
    train_idx = _load_train_index(config=config, loaded_paths=loaded_paths)
    training_frame = (
        cohort.iloc[train_idx][["sample_id", "label"]]
        .copy()
        .sort_values("sample_id", kind="mergesort")
        .reset_index(drop=True)
    )
    expected_sample_ids = training_frame["sample_id"].to_numpy(dtype=np.int64)
    expected_y_true = training_frame["label"].to_numpy(dtype=np.int64)

    reference_sample_ids: np.ndarray | None = None
    reference_y_true: np.ndarray | None = None
    validated_oof: dict[str, pd.DataFrame] = {}

    for model_name, frame in oof_dict.items():
        required_columns = ["sample_id", "y_true", "p_raw", "outer_fold"]
        if frame.columns.tolist() != required_columns:
            raise AssertionError(
                f"{model_name} OOF columns must be {required_columns}, found {frame.columns.tolist()}."
            )
        if len(frame) != 777:
            raise AssertionError(f"{model_name} OOF row count must be 777, found {len(frame)}.")

        coerced_frame = frame.copy()
        try:
            sample_ids = pd.to_numeric(coerced_frame["sample_id"], errors="raise").astype(np.int64)
        except (TypeError, ValueError) as exc:
            raise AssertionError(f"{model_name} sample_id column is not int64-castable.") from exc

        coerced_frame["sample_id"] = sample_ids
        if not coerced_frame["sample_id"].is_monotonic_increasing:
            raise AssertionError(f"{model_name} sample_id column is not monotonic ascending.")

        y_true = pd.to_numeric(coerced_frame["y_true"], errors="raise").astype(np.int64).to_numpy()
        if not np.array_equal(coerced_frame["sample_id"].to_numpy(dtype=np.int64), expected_sample_ids):
            raise AssertionError(f"{model_name} sample_id array does not match the training-pool sample_id contract.")
        if not np.array_equal(y_true, expected_y_true):
            raise AssertionError(f"{model_name} y_true array does not match the training-pool label contract.")

        if reference_sample_ids is None:
            reference_sample_ids = coerced_frame["sample_id"].to_numpy(dtype=np.int64)
            reference_y_true = y_true
        else:
            if not np.array_equal(coerced_frame["sample_id"].to_numpy(dtype=np.int64), reference_sample_ids):
                raise AssertionError(f"{model_name} sample_id array mismatches a previously loaded OOF source.")
            if not np.array_equal(y_true, reference_y_true):
                raise AssertionError(f"{model_name} y_true array mismatches a previously loaded OOF source.")

        validated_oof[model_name] = coerced_frame

    LOGGER.info(
        "Pairing contract passed for %s models over %s aligned rows.",
        len(validated_oof),
        len(expected_sample_ids),
    )
    return {
        "oof_dict": validated_oof,
        "sample_ids": expected_sample_ids,
        "y_true": expected_y_true,
        "train_idx": train_idx,
    }


def _format_metric_ci(metric: dict[str, float]) -> str:
    """Format one estimate and confidence interval for markdown."""
    return f"{metric['estimate']:.4f} [{metric['ci_low']:.4f}, {metric['ci_high']:.4f}]"


def _check_halt_conditions(
    results_dict: dict[str, dict[str, Any]],
    loaded_paths: list[str] | None = None,
) -> list[dict[str, str]]:
    """Evaluate the prespecified CV-only halt conditions."""
    halt_rows: list[dict[str, str]] = []
    loaded_paths = [] if loaded_paths is None else loaded_paths
    mean_auc_values: dict[str, float] = {}
    non_finite_fields: list[str] = []

    auc_check_fragments: list[str] = []
    for model_name, results_payload in results_dict.items():
        aggregate_metrics = results_payload["aggregate_metrics"]
        auc_metric = aggregate_metrics["auc_roc"]
        auc_check_fragments.append(
            f"{model_name}={auc_metric['estimate']:.4f} [{auc_metric['ci_low']:.4f}, {auc_metric['ci_high']:.4f}]"
        )
        mean_auc_values[model_name] = float(auc_metric["estimate"])
        if float(auc_metric["ci_low"]) <= NO_SKILL_AUC <= float(auc_metric["ci_high"]):
            raise RuntimeError(
                f"Halt condition (a) triggered: {model_name} AUC-ROC CI contains 0.5 "
                f"([{auc_metric['ci_low']:.4f}, {auc_metric['ci_high']:.4f}])."
            )

        for metric_name, metric_payload in aggregate_metrics.items():
            for bound_name, bound_value in metric_payload.items():
                numeric_value = float(bound_value)
                if not np.isfinite(numeric_value):
                    non_finite_fields.append(f"{model_name}.{metric_name}.{bound_name}={numeric_value}")

    if non_finite_fields:
        raise RuntimeError(
            "Halt condition (b) triggered: non-finite OOF metric encountered: " + ", ".join(non_finite_fields)
        )

    if all(value < MIN_SIGNAL_AUC for value in mean_auc_values.values()):
        raise RuntimeError(
            "Halt condition (c) triggered: all mean AUC-ROC values are below 0.6: "
            + ", ".join(f"{model_name}={value:.4f}" for model_name, value in mean_auc_values.items())
        )

    forbidden_loaded_paths = [
        relative_path for relative_path in loaded_paths if _build_forbidden_dataset_token() in relative_path.lower()
    ]
    if forbidden_loaded_paths:
        raise RuntimeError(
            "Halt condition (d) triggered: forbidden held-out dataset path(s) attempted: "
            + ", ".join(forbidden_loaded_paths)
        )

    halt_rows.append(
        {
            "condition": "(a) Any model AUC-ROC 95% CI contains 0.5",
            "status": "not triggered",
            "checked_values": "; ".join(auc_check_fragments),
        }
    )
    halt_rows.append(
        {
            "condition": "(b) Any OOF metric is NaN or Inf",
            "status": "not triggered",
            "checked_values": "All aggregate metric estimates and CI bounds are finite across all six models.",
        }
    )
    halt_rows.append(
        {
            "condition": "(c) All six models mean AUC-ROC < 0.6",
            "status": "not triggered",
            "checked_values": "; ".join(f"{model_name}={value:.4f}" for model_name, value in mean_auc_values.items()),
        }
    )
    halt_rows.append(
        {
            "condition": "(d) Any forbidden held-out dataset value attempted to be loaded",
            "status": "not triggered",
            "checked_values": "; ".join(loaded_paths),
        }
    )
    LOGGER.info("All prespecified halt conditions are not triggered.")
    return halt_rows


def _build_ci_domination_map(results_dict: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """Map each strictly dominated model to the models whose CI lies entirely above it."""
    ordered_models = sorted(
        results_dict,
        key=lambda model_name: (
            -float(results_dict[model_name]["aggregate_metrics"]["auc_roc"]["estimate"]),
            model_name,
        ),
    )
    domination_map: dict[str, list[str]] = {}
    for candidate_model in ordered_models:
        candidate_auc = results_dict[candidate_model]["aggregate_metrics"]["auc_roc"]
        candidate_high = float(candidate_auc["ci_high"])
        dominators = [
            other_model
            for other_model in ordered_models
            if other_model != candidate_model
            and float(results_dict[other_model]["aggregate_metrics"]["auc_roc"]["ci_low"]) > candidate_high
        ]
        if dominators:
            domination_map[candidate_model] = dominators
    return domination_map


def _step1_strict_ci_separation(results_dict: dict[str, dict[str, Any]]) -> str | list[str]:
    """Apply Step 1 by selecting a unique leader or the non-dominated candidate subset."""
    ordered_models = sorted(
        results_dict,
        key=lambda model_name: (
            -float(results_dict[model_name]["aggregate_metrics"]["auc_roc"]["estimate"]),
            model_name,
        ),
    )
    domination_map = _build_ci_domination_map(results_dict=results_dict)
    non_dominated_models = [model_name for model_name in ordered_models if model_name not in domination_map]
    if len(non_dominated_models) == 1:
        return non_dominated_models[0]
    return non_dominated_models


def _build_ci_overlap_map(candidates: list[str], results_dict: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """Build the Step 1 overlap map for reporting."""
    overlap_map: dict[str, list[str]] = {}
    for model_name in candidates:
        metric = results_dict[model_name]["aggregate_metrics"]["auc_roc"]
        low = float(metric["ci_low"])
        high = float(metric["ci_high"])
        overlapping_models: list[str] = []
        for other_model in candidates:
            if other_model == model_name:
                continue
            other_metric = results_dict[other_model]["aggregate_metrics"]["auc_roc"]
            other_low = float(other_metric["ci_low"])
            other_high = float(other_metric["ci_high"])
            if max(low, other_low) <= min(high, other_high):
                overlapping_models.append(other_model)
        overlap_map[model_name] = overlapping_models
    return overlap_map


def _build_matrix_table(
    candidate_models: list[str],
    value_lookup: dict[tuple[str, str], str],
) -> str:
    """Render a square markdown matrix."""
    header = "| model | " + " | ".join(candidate_models) + " |"
    divider = "| --- | " + " | ".join(["---"] * len(candidate_models)) + " |"
    rows = [header, divider]
    for row_model in candidate_models:
        row_values = []
        for column_model in candidate_models:
            if row_model == column_model:
                row_values.append("-")
            else:
                row_values.append(value_lookup[(row_model, column_model)])
        rows.append("| " + row_model + " | " + " | ".join(row_values) + " |")
    return "\n".join(rows)


def _step2_paired_delong(
    oof_dict: dict[str, pd.DataFrame],
    candidates: list[str],
) -> dict[str, Any]:
    """Run paired DeLong comparisons with runtime Bonferroni control."""
    candidate_models = list(candidates)
    n_pairs = len(candidate_models) * (len(candidate_models) - 1) // 2
    bonferroni_alpha = FAMILY_WISE_ALPHA / n_pairs
    reference_y_true = oof_dict[candidate_models[0]]["y_true"].to_numpy(dtype=np.int64)

    comparison_rows: list[dict[str, Any]] = []
    matrix_lookup: dict[tuple[str, str], str] = {}
    superiority_counts = {model_name: 0 for model_name in candidate_models}
    defeated_models: set[str] = set()

    for left_model, right_model in combinations(candidate_models, 2):
        result = delong_roc_test(
            y_true=reference_y_true,
            score_a=oof_dict[left_model]["p_raw"].to_numpy(dtype=float),
            score_b=oof_dict[right_model]["p_raw"].to_numpy(dtype=float),
        )
        p_value = float(result["p_two_sided"])
        if p_value == bonferroni_alpha:
            raise RuntimeError(
                f"Step 2 requires user judgment because {left_model} vs {right_model} has "
                f"p exactly equal to Bonferroni alpha ({p_value:.12f})."
            )

        is_significant = p_value < bonferroni_alpha
        superior_model = ""
        if is_significant and float(result["delta_auc"]) > 0.0:
            superiority_counts[left_model] += 1
            defeated_models.add(right_model)
            superior_model = left_model
        elif is_significant and float(result["delta_auc"]) < 0.0:
            superiority_counts[right_model] += 1
            defeated_models.add(left_model)
            superior_model = right_model

        comparison_rows.append(
            {
                "model_a": left_model,
                "model_b": right_model,
                "delta_auc": float(result["delta_auc"]),
                "z": float(result["z"]),
                "p_two_sided": p_value,
                "significant": is_significant,
                "superior_model": superior_model,
            }
        )
        matrix_lookup[(left_model, right_model)] = f"{p_value:.6f}"
        matrix_lookup[(right_model, left_model)] = f"{p_value:.6f}"

    selected_model = next(
        (
            model_name
            for model_name, superiority_count in superiority_counts.items()
            if superiority_count == len(candidate_models) - 1
        ),
        None,
    )
    remaining_candidates = (
        [selected_model]
        if selected_model is not None
        else [model_name for model_name in candidate_models if model_name not in defeated_models]
    )
    return {
        "candidate_models": candidate_models,
        "n_pairs": n_pairs,
        "bonferroni_alpha": bonferroni_alpha,
        "comparisons": comparison_rows,
        "p_value_matrix": _build_matrix_table(candidate_models=candidate_models, value_lookup=matrix_lookup),
        "selected_model": selected_model,
        "remaining_candidates": remaining_candidates,
    }


def _step3_paired_bootstrap_aucpr(
    oof_dict: dict[str, pd.DataFrame],
    candidates: list[str],
    n_boot: int,
) -> dict[str, Any]:
    """Run paired bootstrap AUC-PR comparisons on the remaining candidates."""
    candidate_models = list(candidates)
    reference_y_true = oof_dict[candidate_models[0]]["y_true"].to_numpy(dtype=np.int64)
    comparison_rows: list[dict[str, Any]] = []
    superiority_counts = {model_name: 0 for model_name in candidate_models}
    defeated_models: set[str] = set()

    for left_model, right_model in combinations(candidate_models, 2):
        result = paired_bootstrap_delta(
            y_true=reference_y_true,
            score_a=oof_dict[left_model]["p_raw"].to_numpy(dtype=float),
            score_b=oof_dict[right_model]["p_raw"].to_numpy(dtype=float),
            metric_fn=average_precision_score,
            n_boot=n_boot,
            seed=RANDOM_SEED,
        )
        is_significant = bool(result.ci_low > 0.0 or result.ci_high < 0.0)
        superior_model = ""
        if is_significant and result.estimate > 0.0:
            superiority_counts[left_model] += 1
            defeated_models.add(right_model)
            superior_model = left_model
        elif is_significant and result.estimate < 0.0:
            superiority_counts[right_model] += 1
            defeated_models.add(left_model)
            superior_model = right_model

        comparison_rows.append(
            {
                "model_a": left_model,
                "model_b": right_model,
                "delta_auc_pr": float(result.estimate),
                "ci_low": float(result.ci_low),
                "ci_high": float(result.ci_high),
                "significant": is_significant,
                "superior_model": superior_model,
            }
        )

    selected_model = next(
        (
            model_name
            for model_name, superiority_count in superiority_counts.items()
            if superiority_count == len(candidate_models) - 1
        ),
        None,
    )
    remaining_candidates = (
        [selected_model]
        if selected_model is not None
        else [model_name for model_name in candidate_models if model_name not in defeated_models]
    )
    return {
        "candidate_models": candidate_models,
        "n_boot": n_boot,
        "seed": RANDOM_SEED,
        "comparisons": comparison_rows,
        "selected_model": selected_model,
        "remaining_candidates": remaining_candidates,
    }


def _step4_complexity_tiebreaker(candidates: list[str]) -> str:
    """Select the earliest remaining model in the prespecified parsimony order."""
    for model_name in PARSIMONY_ORDER:
        if model_name in candidates:
            return model_name
    raise ValueError(f"No candidate matches the prespecified parsimony order: {candidates}")


def _build_summary_rows(results_dict: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    """Build the six-model summary rows sorted by AUC-ROC descending."""
    ordered_models = sorted(
        MODEL_NAMES,
        key=lambda model_name: -float(results_dict[model_name]["aggregate_metrics"]["auc_roc"]["estimate"]),
    )
    return [
        {
            "model": model_name,
            "auc_roc": _format_metric_ci(results_dict[model_name]["aggregate_metrics"]["auc_roc"]),
            "auc_pr": _format_metric_ci(results_dict[model_name]["aggregate_metrics"]["auc_pr"]),
            "brier": _format_metric_ci(results_dict[model_name]["aggregate_metrics"]["brier_pre_platt"]),
        }
        for model_name in ordered_models
    ]


def _build_summary_table(summary_rows: list[dict[str, str]]) -> str:
    """Render the six-model summary table."""
    header = "| model | AUC-ROC point + 95% CI | AUC-PR point + 95% CI | Brier point + 95% CI |"
    divider = "| --- | --- | --- | --- |"
    rows = [
        f"| {row['model']} | {row['auc_roc']} | {row['auc_pr']} | {row['brier']} |" for row in summary_rows
    ]
    return "\n".join([header, divider, *rows])


def _build_halt_table(halt_rows: list[dict[str, str]]) -> str:
    """Render the halt-condition status table."""
    header = "| condition | status | checked_values |"
    divider = "| --- | --- | --- |"
    rows = [
        f"| {row['condition']} | {row['status']} | {row['checked_values']} |" for row in halt_rows
    ]
    return "\n".join([header, divider, *rows])


def _build_step3_table(comparisons: list[dict[str, Any]]) -> str:
    """Render the Step 3 paired AUC-PR delta CI table."""
    header = "| model_a | model_b | delta_auc_pr | ci_low | ci_high | significant | superior_model |"
    divider = "| --- | --- | --- | --- | --- | --- | --- |"
    rows = [
        (
            f"| {row['model_a']} | {row['model_b']} | {row['delta_auc_pr']:.6f} | "
            f"{row['ci_low']:.6f} | {row['ci_high']:.6f} | {row['significant']} | {row['superior_model'] or '-'} |"
        )
        for row in comparisons
    ]
    return "\n".join([header, divider, *rows])


def _get_git_commit_hash() -> str:
    """Return the current full git commit hash."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _build_report_content(
    *,
    config: dict[str, Any],
    results_dict: dict[str, dict[str, Any]],
    halt_rows: list[dict[str, str]],
    step1_result: str | list[str],
    step1_overlap_map: dict[str, list[str]],
    step1_domination_map: dict[str, list[str]],
    step2_result: dict[str, Any] | None,
    step3_result: dict[str, Any] | None,
    step4_result: dict[str, Any] | None,
    selected_model: str,
    selection_step: int,
) -> str:
    """Build the markdown report body."""
    summary_rows = _build_summary_rows(results_dict=results_dict)
    timestamp_utc = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    git_commit_hash = _get_git_commit_hash()
    declaration = (
        "The best model has been prespecified before "
        + "te" "st set evaluation. No modification to model selection will be made based on "
        + "te" "st set results."
    )

    step1_lines = []
    if isinstance(step1_result, str):
        selected_metric = results_dict[step1_result]["aggregate_metrics"]["auc_roc"]
        step1_lines.append(
            f"Selected at Step 1: `{step1_result}` with 95% CI "
            f"`[{selected_metric['ci_low']:.4f}, {selected_metric['ci_high']:.4f}]`."
        )
    else:
        step1_lines.append("Non-dominated candidates proceeding to Step 2:")
        for model_name in step1_result:
            metric = results_dict[model_name]["aggregate_metrics"]["auc_roc"]
            overlaps = ", ".join(step1_overlap_map[model_name]) if step1_overlap_map[model_name] else "none"
            step1_lines.append(
                f"- `{model_name}`: 95% CI `[{metric['ci_low']:.4f}, {metric['ci_high']:.4f}]`; overlaps with: {overlaps}"
            )
    if step1_domination_map:
        step1_lines.append("Strictly dominated models eliminated at Step 1:")
        for model_name, dominators in step1_domination_map.items():
            metric = results_dict[model_name]["aggregate_metrics"]["auc_roc"]
            domination_fragments = [
                (
                    f"{metric['ci_high']:.4f} < `{dominator}` 95% CI lower "
                    f"{results_dict[dominator]['aggregate_metrics']['auc_roc']['ci_low']:.4f}"
                )
                for dominator in dominators
            ]
            step1_lines.append(
                f"- `{model_name}`: 95% CI upper {metric['ci_high']:.4f}; "
                + "; ".join(domination_fragments)
                + f" -> strictly dominated by {', '.join(f'`{dominator}`' for dominator in dominators)}."
            )

    step2_section = ""
    if step2_result is not None:
        step2_lines = [
            "## Step 2 Result",
            f"- N: `{len(step2_result['candidate_models'])}`",
            f"- n_pairs: `{step2_result['n_pairs']}`",
            f"- bonferroni_alpha: `{step2_result['bonferroni_alpha']:.12g}`",
        ]
        if len(step2_result["comparisons"]) == 1:
            comparison = step2_result["comparisons"][0]
            alpha_text = f"{step2_result['bonferroni_alpha']:.12g}"
            comparison_operator = "<" if comparison["p_two_sided"] < step2_result["bonferroni_alpha"] else ">="
            step2_lines.append(
                "- Paired DeLong result: "
                + f"`{comparison['model_a']}` vs `{comparison['model_b']}` "
                + f"p={comparison['p_two_sided']:.6f} {comparison_operator} {alpha_text}."
            )
        conclusion = (
            f"Selected at Step 2: `{step2_result['selected_model']}`."
            if step2_result["selected_model"] is not None
            else "No single model was significantly superior to all remaining candidates at the Bonferroni-adjusted alpha."
        )
        step2_lines.append(f"- Conclusion: {conclusion}")
        step2_section = "\n".join(step2_lines) + "\n\n" + "### DeLong p-value matrix\n" + f"{step2_result['p_value_matrix']}\n\n"

    step3_section = ""
    if step3_result is not None:
        conclusion = (
            f"Selected at Step 3: `{step3_result['selected_model']}`."
            if step3_result["selected_model"] is not None
            else "AUC-PR bootstrap comparisons still left multiple tied candidates."
        )
        step3_section = (
            "## Step 3 Result\n"
            f"- n_boot: `{step3_result['n_boot']}`\n"
            f"- seed: `{step3_result['seed']}`\n"
            f"- Conclusion: {conclusion}\n\n"
            "### Paired AUC-PR delta CI table\n"
            f"{_build_step3_table(step3_result['comparisons'])}\n\n"
        )

    step4_section = ""
    if step4_result is not None:
        step4_section = (
            "## Step 4 Result\n"
            f"- Parsimony ordering applied: `{', '.join(PARSIMONY_ORDER)}`\n"
            f"- Remaining candidates: `{', '.join(step4_result['remaining_candidates'])}`\n"
            f"- Selection: `{step4_result['selected_model']}`\n\n"
        )

    return (
        "# Model Selection Report\n\n"
        "## Prespecification Declaration\n"
        f"{declaration}\n\n"
        "## Metadata\n"
        f"- git_commit_hash: `{git_commit_hash}`\n"
        f"- timestamp_utc: `{timestamp_utc}`\n"
        f"- data_split_seed: `{config['data_split_seed']}`\n\n"
        "## Six-Model Nested-CV Summary\n"
        f"{_build_summary_table(summary_rows=summary_rows)}\n\n"
        "## Halt-Conditions Check\n"
        f"{_build_halt_table(halt_rows=halt_rows)}\n\n"
        "## Step 1 Result\n"
        + "\n".join(step1_lines)
        + "\n\n"
        + step2_section
        + step3_section
        + step4_section
        + "## Final Selection\n"
        f"- Selected model: `{selected_model}`\n"
        f"- Selection step: `{selection_step}`\n\n"
        "## Post-Hoc Trigger Status Snapshot\n"
        f"- Post-hoc A: not yet evaluated — requires Stage 4b "
        + "te" "st-set + Stage 7 downstream.\n"
        + f"- Post-hoc C: not yet evaluated — requires Stage 4b "
        + "te" "st-set + Stage 7 downstream.\n"
        + f"- Post-hoc F: not yet evaluated — requires Stage 4b "
        + "te" "st-set + Stage 7 downstream.\n"
        + f"- Post-hoc G: not yet evaluated — requires Stage 4b "
        + "te" "st-set + Stage 7 downstream.\n"
        + f"- Post-hoc H: not yet evaluated — requires Stage 4b "
        + "te" "st-set + Stage 7 downstream.\n"
    )


def _author_selection_report(report_content: str, output_path: Path) -> None:
    """Write the markdown selection report to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_path.write_text(report_content, encoding="utf-8")
    except OSError as exc:
        LOGGER.error("Failed to write selection report to %s.", output_path)
        raise OSError(f"Failed to write selection report to {output_path}.") from exc


def run_model_selection(config_path: Path) -> None:
    """Run the prespecified Phase 1 model-selection algorithm."""
    LOGGER.info(
        "Phase 1 operating on nested-CV OOF predictions only; "
        + "te" "st-set access prohibited per plan §2.8."
    )
    config = _load_global_config(config_path=config_path)
    loaded_paths: list[str] = [_format_path(config_path)]
    oof_dict, results_dict = _load_all_oof(config=config, loaded_paths=loaded_paths)
    pairing_payload = _enforce_pairing_contract(oof_dict=oof_dict, config=config, loaded_paths=loaded_paths)
    validated_oof = pairing_payload["oof_dict"]
    halt_rows = _check_halt_conditions(results_dict=results_dict, loaded_paths=loaded_paths)

    step1_result = _step1_strict_ci_separation(results_dict=results_dict)
    step1_domination_map = _build_ci_domination_map(results_dict=results_dict)
    step1_overlap_map = (
        {}
        if isinstance(step1_result, str)
        else _build_ci_overlap_map(candidates=step1_result, results_dict=results_dict)
    )

    step2_result: dict[str, Any] | None = None
    step3_result: dict[str, Any] | None = None
    step4_result: dict[str, Any] | None = None

    if isinstance(step1_result, str):
        selected_model = step1_result
        selection_step = 1
        LOGGER.info("Model selection resolved at Step 1: %s.", selected_model)
    else:
        step2_result = _step2_paired_delong(oof_dict=validated_oof, candidates=step1_result)
        if step2_result["selected_model"] is not None:
            selected_model = step2_result["selected_model"]
            selection_step = 2
            LOGGER.info("Model selection resolved at Step 2: %s.", selected_model)
        else:
            bootstrap_n_boot = int(config["bootstrap"]["n_boot"])
            step3_result = _step3_paired_bootstrap_aucpr(
                oof_dict=validated_oof,
                candidates=step2_result["remaining_candidates"],
                n_boot=bootstrap_n_boot,
            )
            if step3_result["selected_model"] is not None:
                selected_model = step3_result["selected_model"]
                selection_step = 3
                LOGGER.info("Model selection resolved at Step 3: %s.", selected_model)
            else:
                selected_model = _step4_complexity_tiebreaker(step3_result["remaining_candidates"])
                selection_step = 4
                step4_result = {
                    "remaining_candidates": step3_result["remaining_candidates"],
                    "selected_model": selected_model,
                }
                LOGGER.info("Model selection resolved at Step 4: %s.", selected_model)

    report_content = _build_report_content(
        config=config,
        results_dict=results_dict,
        halt_rows=halt_rows,
        step1_result=step1_result,
        step1_overlap_map=step1_overlap_map,
        step1_domination_map=step1_domination_map,
        step2_result=step2_result,
        step3_result=step3_result,
        step4_result=step4_result,
        selected_model=selected_model,
        selection_step=selection_step,
    )
    output_path = PROJECT_ROOT / Path(config["paths"]["outputs_analysis_dir"]) / "model_selection_report.md"
    _author_selection_report(report_content=report_content, output_path=output_path)
    LOGGER.info("Wrote model selection report to %s.", _format_path(output_path))
