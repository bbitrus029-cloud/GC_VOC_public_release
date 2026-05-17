"""Authoritative audit-time metric definitions."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.evaluation.calibration import (
    compute_brier_before_after,
    compute_calibration_slope_intercept,
)


LOGGER = logging.getLogger(__name__)


def _validate_binary_inputs(y_true: np.ndarray, scores: np.ndarray, scores_name: str) -> None:
    """Validate aligned binary labels and score arrays."""
    if not isinstance(y_true, np.ndarray) or y_true.ndim != 1:
        raise TypeError("y_true must be a one-dimensional numpy array.")
    if not isinstance(scores, np.ndarray) or scores.ndim != 1:
        raise TypeError(f"{scores_name} must be a one-dimensional numpy array.")
    if len(y_true) != len(scores):
        raise ValueError(f"y_true and {scores_name} must have the same length.")
    if len(y_true) == 0:
        raise ValueError("y_true and score arrays must be non-empty.")


def _coerce_binary_labels(y_true: np.ndarray) -> np.ndarray:
    """Coerce a binary label vector to {0, 1}."""
    unique_classes = np.unique(y_true)
    if len(unique_classes) == 1:
        return np.zeros_like(y_true, dtype=int) if unique_classes[0] == 0 else np.ones_like(y_true, dtype=int)
    if len(unique_classes) != 2:
        raise ValueError("y_true must contain one or two classes only.")

    return (y_true == unique_classes.max()).astype(int)


def _warn_undefined(metric_name: str, reason: str) -> float:
    """Log and return `np.nan` for undefined metrics."""
    LOGGER.warning("%s is undefined: %s", metric_name, reason)
    return float(np.nan)


def _safe_ratio(numerator: float, denominator: float, metric_name: str) -> float:
    """Compute a safe ratio or `np.nan` when the denominator is zero."""
    if denominator == 0:
        return _warn_undefined(metric_name, "denominator is zero")
    return float(numerator / denominator)


def compute_threshold_free_metrics(y_true: np.ndarray, p_raw: np.ndarray) -> dict[str, float]:
    """Compute threshold-free metrics from raw probabilities.

    Args:
        y_true: Binary ground-truth labels.
        p_raw: Raw predicted probabilities for the positive class.

    Returns:
        A dictionary containing `auc_roc`, `auc_pr`, and `brier_pre_platt`.
    """
    _validate_binary_inputs(y_true=y_true, scores=p_raw, scores_name="p_raw")
    binary_labels = _coerce_binary_labels(y_true)
    if len(np.unique(binary_labels)) < 2:
        auc_roc = _warn_undefined("auc_roc", "only one class is present in y_true")
        auc_pr = _warn_undefined("auc_pr", "only one class is present in y_true")
    else:
        auc_roc = float(roc_auc_score(binary_labels, p_raw))
        auc_pr = float(average_precision_score(binary_labels, p_raw))

    return {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "brier_pre_platt": float(brier_score_loss(binary_labels, p_raw)),
    }


def compute_count_metrics(y_true: np.ndarray, p_raw: np.ndarray, threshold_p_raw: float) -> dict[str, float]:
    """Compute confusion counts and prevalence in raw-probability space."""
    _validate_binary_inputs(y_true=y_true, scores=p_raw, scores_name="p_raw")
    binary_labels = _coerce_binary_labels(y_true)

    y_pred = (p_raw >= threshold_p_raw).astype(int)
    tp = int(np.sum((binary_labels == 1) & (y_pred == 1)))
    tn = int(np.sum((binary_labels == 0) & (y_pred == 0)))
    fp = int(np.sum((binary_labels == 0) & (y_pred == 1)))
    fn = int(np.sum((binary_labels == 1) & (y_pred == 0)))
    n_total = int(len(binary_labels))
    n_pos = int(np.sum(binary_labels == 1))
    n_neg = int(np.sum(binary_labels == 0))

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n_total": n_total,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "prevalence": float(n_pos / n_total),
    }


def compute_threshold_dependent_metrics(
    y_true: np.ndarray,
    p_raw: np.ndarray,
    threshold_p_raw: float,
) -> dict[str, float]:
    """Compute threshold-dependent metrics in raw-probability space only."""
    count_metrics = compute_count_metrics(y_true=y_true, p_raw=p_raw, threshold_p_raw=threshold_p_raw)

    tp = count_metrics["tp"]
    tn = count_metrics["tn"]
    fp = count_metrics["fp"]
    fn = count_metrics["fn"]
    sensitivity = _safe_ratio(tp, tp + fn, "sensitivity")
    specificity = _safe_ratio(tn, tn + fp, "specificity")
    ppv = _safe_ratio(tp, tp + fp, "ppv")
    npv = _safe_ratio(tn, tn + fn, "npv")
    f1 = _safe_ratio(2 * tp, (2 * tp) + fp + fn, "f1")
    if np.isnan(sensitivity) or np.isnan(specificity):
        balanced_accuracy = _warn_undefined(
            "balanced_accuracy",
            "sensitivity or specificity is undefined",
        )
    else:
        balanced_accuracy = float(np.mean([sensitivity, specificity]))

    mcc_denominator = float(np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = _safe_ratio((tp * tn) - (fp * fn), mcc_denominator, "mcc")
    accuracy = float((tp + tn) / count_metrics["n_total"])
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
        "accuracy": accuracy,
    }


def compute_calibration_metrics(y_true: np.ndarray, p_raw: np.ndarray, p_cal: np.ndarray) -> dict[str, float]:
    """Compute pre/post-Platt Brier plus calibrated slope and intercept."""
    _validate_binary_inputs(y_true=y_true, scores=p_raw, scores_name="p_raw")
    _validate_binary_inputs(y_true=y_true, scores=p_cal, scores_name="p_cal")
    binary_labels = _coerce_binary_labels(y_true)

    brier_summary = compute_brier_before_after(p_raw=p_raw, p_cal=p_cal, y_true=binary_labels)
    if len(np.unique(binary_labels)) < 2:
        calibration_slope = _warn_undefined("calibration_slope", "only one class is present in y_true")
        calibration_intercept = _warn_undefined("calibration_intercept", "only one class is present in y_true")
    else:
        slope_intercept = compute_calibration_slope_intercept(p_cal=p_cal, y_true=binary_labels)
        calibration_slope = slope_intercept["slope"]
        calibration_intercept = slope_intercept["intercept"]

    return {
        **brier_summary,
        "calibration_slope": calibration_slope,
        "calibration_intercept": calibration_intercept,
    }


def compute_full_metric_set(
    y_true: np.ndarray,
    p_raw: np.ndarray,
    p_cal: np.ndarray,
    threshold_p_raw: float,
) -> dict[str, float]:
    """Compute the full audit-time metric set.

    Args:
        y_true: Binary ground-truth labels.
        p_raw: Raw predicted probabilities for the positive class.
        p_cal: Platt-calibrated probabilities for the positive class.
        threshold_p_raw: Fixed decision threshold in raw-probability space.

    Returns:
        A dictionary containing threshold-free, calibration, threshold-dependent,
        and count metrics.
    """
    threshold_free_metrics = compute_threshold_free_metrics(y_true=y_true, p_raw=p_raw)
    calibration_metrics = compute_calibration_metrics(y_true=y_true, p_raw=p_raw, p_cal=p_cal)
    threshold_dependent_metrics = compute_threshold_dependent_metrics(
        y_true=y_true,
        p_raw=p_raw,
        threshold_p_raw=threshold_p_raw,
    )
    count_metrics = compute_count_metrics(y_true=y_true, p_raw=p_raw, threshold_p_raw=threshold_p_raw)

    return {
        **threshold_free_metrics,
        **calibration_metrics,
        **threshold_dependent_metrics,
        **count_metrics,
    }
