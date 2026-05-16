from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


LOGGER = logging.getLogger(__name__)


def _warn_undefined(metric_name: str, reason: str) -> float:
    """Log and return `np.nan` for undefined metrics."""
    LOGGER.warning("%s is undefined: %s", metric_name, reason)
    return float(np.nan)


def _safe_ratio(numerator: float, denominator: float, metric_name: str) -> float:
    """Compute a safe ratio or `np.nan` when the denominator is zero."""
    if denominator == 0:
        return _warn_undefined(metric_name, "denominator is zero")
    return float(numerator / denominator)


class MetricsCalculator:
    """Training-facing metric wrapper for binary classification.

    Operational-plan references: §2.11 and §5.5.
    """

    def compute_binary_metrics(
        self,
        y_true: np.ndarray,
        p_raw: np.ndarray,
        threshold: float,
    ) -> dict[str, float]:
        """Compute threshold-free and threshold-dependent binary metrics.

        Args:
            y_true: Binary ground-truth labels.
            p_raw: Raw predicted probabilities for the positive class.
            threshold: Fixed decision threshold in raw-probability space.

        Returns:
            A dictionary containing discrimination, calibration, confusion-count,
            and threshold-dependent metrics.
        """
        self._validate_inputs(y_true=y_true, p_raw=p_raw)
        threshold_free = self.compute_threshold_free(y_true=y_true, p_raw=p_raw)

        y_pred = (p_raw >= threshold).astype(int)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        n_total = int(len(y_true))
        n_pos = int(np.sum(y_true == 1))
        n_neg = int(np.sum(y_true == 0))

        sensitivity = _safe_ratio(tp, tp + fn, "sensitivity")
        specificity = _safe_ratio(tn, tn + fp, "specificity")
        ppv = _safe_ratio(tp, tp + fp, "ppv")
        npv = _safe_ratio(tn, tn + fn, "npv")
        f1 = _safe_ratio(2 * tp, (2 * tp) + fp + fn, "f1")
        balanced_accuracy = (
            float(np.nanmean([sensitivity, specificity]))
            if not np.isnan(sensitivity) and not np.isnan(specificity)
            else _warn_undefined("balanced_accuracy", "one or more component recalls are undefined")
        )
        mcc_denominator = float(np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        mcc = _safe_ratio((tp * tn) - (fp * fn), mcc_denominator, "mcc")
        accuracy = float(np.mean(y_true == y_pred))

        return {
            **threshold_free,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "ppv": ppv,
            "npv": npv,
            "f1": f1,
            "balanced_accuracy": balanced_accuracy,
            "mcc": mcc,
            "accuracy": accuracy,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "n_total": n_total,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "prevalence": float(n_pos / n_total),
        }

    def compute_threshold_free(self, y_true: np.ndarray, p_raw: np.ndarray) -> dict[str, float]:
        """Compute threshold-free binary metrics.

        Args:
            y_true: Binary ground-truth labels.
            p_raw: Raw predicted probabilities for the positive class.

        Returns:
            A dictionary with AUC-ROC, AUC-PR, and Brier score.
        """
        self._validate_inputs(y_true=y_true, p_raw=p_raw)
        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            LOGGER.warning("Threshold-free discrimination metrics are undefined when only one class is present.")
            auc_roc = float(np.nan)
            auc_pr = float(np.nan)
        else:
            auc_roc = float(roc_auc_score(y_true, p_raw))
            auc_pr = float(average_precision_score(y_true, p_raw))

        return {
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "brier": float(brier_score_loss(y_true, p_raw)),
        }

    @staticmethod
    def _validate_inputs(y_true: np.ndarray, p_raw: np.ndarray) -> None:
        """Validate binary metric inputs."""
        if not isinstance(y_true, np.ndarray) or y_true.ndim != 1:
            raise TypeError("y_true must be a one-dimensional numpy array.")
        if not isinstance(p_raw, np.ndarray) or p_raw.ndim != 1:
            raise TypeError("p_raw must be a one-dimensional numpy array.")
        if len(y_true) != len(p_raw):
            raise ValueError("y_true and p_raw must have the same length.")
        if len(y_true) == 0:
            raise ValueError("y_true and p_raw must be non-empty.")
