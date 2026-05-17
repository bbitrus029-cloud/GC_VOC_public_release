"""Calibration utilities for audit-time evaluation."""

from __future__ import annotations

import logging

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


LOGGER = logging.getLogger(__name__)
PROBABILITY_EPSILON = 1e-7


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
    """Coerce a one-class or two-class label vector to {0, 1}."""
    unique_classes = np.unique(y_true)
    if len(unique_classes) == 1:
        return np.zeros_like(y_true, dtype=int) if unique_classes[0] == 0 else np.ones_like(y_true, dtype=int)
    if len(unique_classes) != 2:
        raise ValueError("y_true must contain one or two classes only.")

    return (y_true == unique_classes.max()).astype(int)


def fit_platt_calibrator(p_raw: np.ndarray, y_true: np.ndarray, seed: int) -> LogisticRegression:
    """Fit a Platt calibrator in raw-probability space.

    Args:
        p_raw: Raw predicted probabilities for the positive class.
        y_true: Binary ground-truth labels.
        seed: Deterministic random seed for the logistic-regression solver.

    Returns:
        A fitted one-dimensional logistic-regression calibrator.

    Raises:
        TypeError: If the inputs are not one-dimensional numpy arrays.
        ValueError: If the arrays are empty, misaligned, or not binary.
    """
    _validate_binary_inputs(y_true=y_true, scores=p_raw, scores_name="p_raw")
    binary_labels = _coerce_binary_labels(y_true)
    if len(np.unique(binary_labels)) != 2:
        raise ValueError("Platt calibration requires exactly two classes in y_true.")

    calibrator = LogisticRegression(
        solver="lbfgs",
        random_state=seed,
        penalty=None,
    )
    calibrator.fit(p_raw.reshape(-1, 1), binary_labels)
    return calibrator


def apply_platt_calibrator(calibrator: LogisticRegression, p_raw: np.ndarray) -> np.ndarray:
    """Apply a fitted Platt calibrator to raw probabilities.

    Args:
        calibrator: A fitted logistic-regression calibrator.
        p_raw: Raw predicted probabilities for the positive class.

    Returns:
        Calibrated probabilities with shape `(n_samples,)`.

    Raises:
        TypeError: If `p_raw` is not a one-dimensional numpy array.
    """
    if not isinstance(p_raw, np.ndarray) or p_raw.ndim != 1:
        raise TypeError("p_raw must be a one-dimensional numpy array.")

    return calibrator.predict_proba(p_raw.reshape(-1, 1))[:, 1]


def compute_calibration_slope_intercept(p_cal: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    """Estimate calibration slope and intercept from calibrated probabilities.

    Args:
        p_cal: Calibrated predicted probabilities for the positive class.
        y_true: Binary ground-truth labels.

    Returns:
        A dictionary containing `slope`, `intercept`, and `n_used`.
    """
    _validate_binary_inputs(y_true=y_true, scores=p_cal, scores_name="p_cal")
    binary_labels = _coerce_binary_labels(y_true)
    if len(np.unique(binary_labels)) != 2:
        LOGGER.warning("Calibration slope and intercept are undefined when only one class is present.")
        return {"slope": float(np.nan), "intercept": float(np.nan), "n_used": int(len(binary_labels))}

    clipped_probabilities = np.clip(p_cal, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    logit_probabilities = logit(clipped_probabilities).reshape(-1, 1)

    calibration_model = LogisticRegression(
        solver="lbfgs",
        penalty=None,
    )
    calibration_model.fit(logit_probabilities, binary_labels)
    return {
        "slope": float(calibration_model.coef_[0, 0]),
        "intercept": float(calibration_model.intercept_[0]),
        "n_used": int(len(binary_labels)),
    }


def compute_brier_before_after(p_raw: np.ndarray, p_cal: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    """Compute pre-Platt and post-Platt Brier scores.

    Args:
        p_raw: Raw predicted probabilities for the positive class.
        p_cal: Calibrated predicted probabilities for the positive class.
        y_true: Binary ground-truth labels.

    Returns:
        A dictionary containing `brier_pre_platt`, `brier_post_platt`, and
        `brier_delta`.
    """
    _validate_binary_inputs(y_true=y_true, scores=p_raw, scores_name="p_raw")
    _validate_binary_inputs(y_true=y_true, scores=p_cal, scores_name="p_cal")
    binary_labels = _coerce_binary_labels(y_true)

    brier_pre_platt = float(brier_score_loss(binary_labels, p_raw))
    brier_post_platt = float(brier_score_loss(binary_labels, p_cal))
    return {
        "brier_pre_platt": brier_pre_platt,
        "brier_post_platt": brier_post_platt,
        "brier_delta": float(brier_post_platt - brier_pre_platt),
    }
