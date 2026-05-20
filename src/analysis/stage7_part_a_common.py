from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.analysis.data_split import (
    _format_path,
    _load_yaml_config,
    _prepare_cohort_dataframe,
    _resolve_path,
    _write_text,
)
from src.config import MODEL_NAMES, PROJECT_ROOT, RANDOM_SEED
from src.evaluation.calibration import compute_calibration_slope_intercept
from src.evaluation.metrics import (
    compute_count_metrics,
    compute_threshold_dependent_metrics,
    compute_threshold_free_metrics,
)
from src.training.common.bootstrap import bootstrap_metric
from src.training.load_final_model import load_final_model


LOGGER = logging.getLogger(__name__)
TEST_PREDICTION_COLUMNS = ["sample_id", "y_true", "p_raw", "p_cal", "y_pred", "quadrant"]
TRAIN_IN_SAMPLE_COLUMNS = ["sample_id", "y_true", "p_raw", "p_cal", "y_pred", "quadrant"]
OOF_COLUMNS = ["sample_id", "y_true", "p_raw", "outer_fold"]
BOOTSTRAP_CONVENTION_TEXT = (
    "single-RNG sequential per E011 (commit a2af64e); reproducibility verified via deterministic "
    "PCG64 stream; valid for all bootstrap-derived CIs in this session."
)
THREE_VIEW_METRIC_NAMES = (
    "auc_roc",
    "auc_pr",
    "brier",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "f1",
    "balanced_accuracy",
    "mcc",
    "accuracy",
    "tp",
    "tn",
    "fp",
    "fn",
    "n_total",
    "n_pos",
    "n_neg",
    "prevalence",
)
BEST_MODEL_SUBGROUP_METRIC_NAMES = (
    "auc_roc",
    "auc_pr",
    "brier",
    "calibration_slope",
    "calibration_intercept",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "f1",
    "balanced_accuracy",
    "mcc",
    "accuracy",
    "tp",
    "tn",
    "fp",
    "fn",
    "n_total",
    "n_pos",
    "n_neg",
    "prevalence",
)
THRESHOLD_FREE_METRIC_NAMES = ("auc_roc", "auc_pr", "brier")
CLINICAL_TRANSLATION_METRIC_NAMES = ("sensitivity", "specificity", "ppv", "npv", "nne")


def _load_global_config(config_path: Path) -> dict[str, Any]:
    """Load the global modeling config."""
    return _load_yaml_config(config_path=config_path)


def _timestamp_utc() -> str:
    """Return an ISO-8601 UTC timestamp with trailing Z."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON payload using UTF-8 and project-relative path hygiene."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    except OSError as exc:
        LOGGER.error("Failed to write JSON output to %s.", path)
        raise OSError(f"Failed to write JSON output to {path}.") from exc


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON mapping from disk."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        LOGGER.error("Failed to read JSON at %s.", path)
        raise OSError(f"Failed to read JSON at {path}.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON at {path} must deserialize to a mapping.")
    return payload


def _load_prediction_frame(path: Path, *, comment: str | None = None) -> pd.DataFrame:
    """Load a prediction CSV with optional comment stripping."""
    try:
        frame = pd.read_csv(path, comment=comment)
    except OSError as exc:
        LOGGER.error("Failed to read prediction CSV at %s.", path)
        raise OSError(f"Failed to read prediction CSV at {path}.") from exc
    return frame


def _validate_columns(frame: pd.DataFrame, expected_columns: list[str], path: Path) -> pd.DataFrame:
    """Validate exact CSV columns and coerce sample_id/y_true numerics."""
    if frame.columns.tolist() != expected_columns:
        raise AssertionError(
            f"{_format_path(path)} columns must be {expected_columns}, found {frame.columns.tolist()}."
        )
    coerced = frame.copy()
    coerced["sample_id"] = pd.to_numeric(coerced["sample_id"], errors="raise").astype(np.int64)
    coerced["y_true"] = pd.to_numeric(coerced["y_true"], errors="raise").astype(np.int64)
    return coerced


def _prediction_path(model_name: str, view_name: str) -> Path:
    """Resolve one prediction CSV path for a model/view pair."""
    if view_name == "test":
        filename = f"binary_{model_name}_test_predictions.csv"
    elif view_name == "nested_cv_oof":
        filename = f"binary_{model_name}_nested_cv_oof.csv"
    elif view_name == "train_in_sample":
        filename = f"binary_{model_name}_train_in_sample.csv"
    else:
        raise ValueError(f"Unsupported prediction view: {view_name}")
    return PROJECT_ROOT / "outputs" / "predictions" / filename


def _load_view_prediction_frame(model_name: str, view_name: str) -> pd.DataFrame:
    """Load one prediction source for Session 11."""
    path = _prediction_path(model_name=model_name, view_name=view_name)
    if view_name == "nested_cv_oof":
        frame = _load_prediction_frame(path)
        return _validate_columns(frame=frame, expected_columns=OOF_COLUMNS, path=path)
    if view_name == "train_in_sample":
        frame = _load_prediction_frame(path, comment="#")
        return _validate_columns(frame=frame, expected_columns=TRAIN_IN_SAMPLE_COLUMNS, path=path)
    frame = _load_prediction_frame(path)
    return _validate_columns(frame=frame, expected_columns=TEST_PREDICTION_COLUMNS, path=path)


def _load_model_payload(model_name: str) -> dict[str, Any]:
    """Load one final PKL and verify threshold metadata against the mapping JSON."""
    model_path = PROJECT_ROOT / "outputs" / "models" / f"binary_{model_name}_final.pkl"
    payload = load_final_model(model_path)
    mapping_path = PROJECT_ROOT / "outputs" / "analysis" / f"binary_{model_name}_threshold_mapping.json"
    mapping_payload = _read_json(mapping_path)

    threshold_p_raw = float(payload["threshold_p_raw"])
    threshold_p_final_equivalent = float(payload["threshold_p_final_equivalent"])
    mapping_threshold_p_raw = float(mapping_payload["threshold_p_raw"])
    mapping_threshold_p_final_equivalent = float(mapping_payload["threshold_p_final_equivalent"])

    if not np.isclose(threshold_p_raw, mapping_threshold_p_raw, atol=0.0, rtol=0.0):
        raise AssertionError(
            f"{model_name} threshold_p_raw mismatch between PKL and threshold mapping JSON: "
            f"{threshold_p_raw} vs {mapping_threshold_p_raw}."
        )
    if not np.isclose(threshold_p_final_equivalent, mapping_threshold_p_final_equivalent, atol=0.0, rtol=0.0):
        raise AssertionError(
            f"{model_name} threshold_p_final_equivalent mismatch between PKL and threshold mapping JSON: "
            f"{threshold_p_final_equivalent} vs {mapping_threshold_p_final_equivalent}."
        )
    return payload


def _load_test_frame_with_metadata(config: dict[str, Any]) -> pd.DataFrame:
    """Load the locked 195-row test cohort joined to metadata."""
    cohort = _prepare_cohort_dataframe(config=config)
    test_idx_path = _resolve_path(config["paths"]["splits_dir"]) / "test_idx.npz"
    try:
        with np.load(test_idx_path, allow_pickle=False) as split_file:
            test_idx = split_file["test_idx"].astype(np.int64)
    except OSError as exc:
        LOGGER.error("Failed to read test indices at %s.", test_idx_path)
        raise OSError(f"Failed to read test indices at {test_idx_path}.") from exc

    test_frame = cohort.iloc[test_idx].copy().sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    if len(test_frame) != 195:
        raise AssertionError(f"Expected 195 rows in the locked test split, found {len(test_frame)}.")
    return test_frame


def _merge_predictions_with_metadata(
    prediction_frame: pd.DataFrame,
    metadata_frame: pd.DataFrame,
    *,
    required_columns: list[str],
) -> pd.DataFrame:
    """Merge prediction rows onto a metadata frame using sample_id."""
    merged = prediction_frame.merge(
        metadata_frame[["sample_id", *required_columns]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if merged[required_columns].isna().any().any():
        raise AssertionError(f"Metadata merge left missing values in required columns {required_columns}.")
    return merged


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return a safe division result or NaN for zero denominator."""
    if denominator == 0:
        return float(np.nan)
    return float(numerator / denominator)


def _compute_prevalence_adjusted_metrics(
    *,
    sensitivity: float,
    specificity: float,
    prevalence: float,
) -> dict[str, float]:
    """Compute prevalence-adjusted PPV, NPV, and NNE from sensitivity/specificity."""
    false_positive_rate = 1.0 - specificity
    false_negative_rate = 1.0 - sensitivity
    ppv = _safe_ratio(
        sensitivity * prevalence,
        (sensitivity * prevalence) + (false_positive_rate * (1.0 - prevalence)),
    )
    npv = _safe_ratio(
        specificity * (1.0 - prevalence),
        (specificity * (1.0 - prevalence)) + (false_negative_rate * prevalence),
    )
    nne = float(np.nan) if np.isnan(ppv) or ppv == 0.0 else float(1.0 / ppv)
    return {"ppv": ppv, "npv": npv, "nne": nne}


def _metric_callable(
    metric_name: str,
    *,
    threshold_p_raw: float,
    prevalence: float | None = None,
) -> tuple[np.ndarray | None, Callable[[np.ndarray, np.ndarray], float]]:
    """Return the score source selector and metric callable for bootstrap estimation."""

    def _threshold_free(metric_key: str) -> Callable[[np.ndarray, np.ndarray], float]:
        return lambda y_true, p_raw: float(compute_threshold_free_metrics(y_true=y_true, p_raw=p_raw)[metric_key])

    def _threshold_dependent(metric_key: str) -> Callable[[np.ndarray, np.ndarray], float]:
        return lambda y_true, p_raw: float(
            compute_threshold_dependent_metrics(
                y_true=y_true,
                p_raw=p_raw,
                threshold_p_raw=threshold_p_raw,
            )[metric_key]
        )

    def _count_metric(metric_key: str) -> Callable[[np.ndarray, np.ndarray], float]:
        return lambda y_true, p_raw: float(
            compute_count_metrics(y_true=y_true, p_raw=p_raw, threshold_p_raw=threshold_p_raw)[metric_key]
        )

    if metric_name == "auc_roc":
        return None, _threshold_free("auc_roc")
    if metric_name == "auc_pr":
        return None, _threshold_free("auc_pr")
    if metric_name == "brier":
        return None, _threshold_free("brier_pre_platt")
    if metric_name in {"sensitivity", "specificity", "ppv", "npv", "f1", "balanced_accuracy", "mcc", "accuracy"}:
        return None, _threshold_dependent(metric_name)
    if metric_name in {"tp", "tn", "fp", "fn", "n_total", "n_pos", "n_neg", "prevalence"}:
        return None, _count_metric(metric_name)
    if metric_name == "calibration_slope":
        return np.array([], dtype=float), lambda y_true, p_cal: float(
            compute_calibration_slope_intercept(p_cal=p_cal, y_true=y_true)["slope"]
        )
    if metric_name == "calibration_intercept":
        return np.array([], dtype=float), lambda y_true, p_cal: float(
            compute_calibration_slope_intercept(p_cal=p_cal, y_true=y_true)["intercept"]
        )
    if metric_name in {"ppv_adjusted", "npv_adjusted", "nne_adjusted"}:
        if prevalence is None:
            raise ValueError("prevalence is required for prevalence-adjusted clinical translation metrics.")

        def _adjusted_metric(y_true: np.ndarray, p_raw: np.ndarray) -> float:
            threshold_metrics = compute_threshold_dependent_metrics(
                y_true=y_true,
                p_raw=p_raw,
                threshold_p_raw=threshold_p_raw,
            )
            adjusted = _compute_prevalence_adjusted_metrics(
                sensitivity=float(threshold_metrics["sensitivity"]),
                specificity=float(threshold_metrics["specificity"]),
                prevalence=prevalence,
            )
            mapping = {
                "ppv_adjusted": adjusted["ppv"],
                "npv_adjusted": adjusted["npv"],
                "nne_adjusted": adjusted["nne"],
            }
            return float(mapping[metric_name])

        return None, _adjusted_metric

    raise ValueError(f"Unsupported metric name: {metric_name}")


def _bootstrap_metric_from_frame(
    prediction_frame: pd.DataFrame,
    *,
    metric_name: str,
    threshold_p_raw: float,
    n_boot: int,
    seed: int,
    prevalence: float | None = None,
) -> dict[str, float]:
    """Compute one point estimate and percentile CI from a prediction frame."""
    y_true = prediction_frame["y_true"].to_numpy(dtype=np.int64)
    p_raw = prediction_frame["p_raw"].to_numpy(dtype=float)
    p_cal = prediction_frame["p_cal"].to_numpy(dtype=float) if "p_cal" in prediction_frame.columns else None
    score_selector, metric_fn = _metric_callable(
        metric_name=metric_name,
        threshold_p_raw=threshold_p_raw,
        prevalence=prevalence,
    )
    score_array = p_cal if score_selector is not None else p_raw
    if score_array is None:
        raise ValueError(f"Metric {metric_name} requires p_cal but no p_cal column is present.")
    result = bootstrap_metric(
        y_true=y_true,
        y_score=score_array,
        metric_fn=metric_fn,
        n_boot=n_boot,
        seed=seed,
    )
    return {"estimate": result.estimate, "ci_low": result.ci_low, "ci_high": result.ci_high}


def _build_metric_output_frame(metric_rows: list[dict[str, Any]], column_order: list[str]) -> pd.DataFrame:
    """Build one ordered long-format output frame."""
    frame = pd.DataFrame(metric_rows)
    if frame.empty:
        return pd.DataFrame(columns=column_order)
    return frame.loc[:, column_order]


__all__ = [
    "BEST_MODEL_SUBGROUP_METRIC_NAMES",
    "BOOTSTRAP_CONVENTION_TEXT",
    "CLINICAL_TRANSLATION_METRIC_NAMES",
    "OOF_COLUMNS",
    "TEST_PREDICTION_COLUMNS",
    "THREE_VIEW_METRIC_NAMES",
    "THRESHOLD_FREE_METRIC_NAMES",
    "TRAIN_IN_SAMPLE_COLUMNS",
    "_bootstrap_metric_from_frame",
    "_build_metric_output_frame",
    "_compute_prevalence_adjusted_metrics",
    "_format_path",
    "_get_git_commit_hash",
    "_load_global_config",
    "_load_model_payload",
    "_load_prediction_frame",
    "_load_test_frame_with_metadata",
    "_load_view_prediction_frame",
    "_merge_predictions_with_metadata",
    "_read_json",
    "_resolve_path",
    "_safe_ratio",
    "_timestamp_utc",
    "_write_json",
    "_write_text",
    "MODEL_NAMES",
    "PROJECT_ROOT",
    "RANDOM_SEED",
]
