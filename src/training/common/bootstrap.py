from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    """Point estimate and percentile confidence interval container.

    Operational-plan reference: §5.6.
    """

    estimate: float
    ci_low: float
    ci_high: float


def _validate_bootstrap_inputs(y_true: np.ndarray, y_score: np.ndarray) -> None:
    """Validate bootstrap input arrays."""
    if not isinstance(y_true, np.ndarray) or y_true.ndim != 1:
        raise TypeError("y_true must be a one-dimensional numpy array.")
    if not isinstance(y_score, np.ndarray) or y_score.ndim != 1:
        raise TypeError("y_score must be a one-dimensional numpy array.")
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length.")
    if len(np.unique(y_true)) != 2:
        raise ValueError("Bootstrap utilities require exactly two classes in y_true.")


def _draw_stratified_indices(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample bootstrap indices with class balance preserved."""
    sampled_indices: list[np.ndarray] = []
    for class_label in np.unique(y_true):
        class_indices = np.flatnonzero(y_true == class_label)
        sampled_indices.append(rng.choice(class_indices, size=len(class_indices), replace=True))
    combined = np.concatenate(sampled_indices)
    return rng.permutation(combined)


def bootstrap_metric(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int,
    seed: int,
) -> BootstrapResult:
    """Estimate a metric and percentile CI by stratified bootstrap.

    Operational-plan references: §2.11 and §5.6.

    Args:
        y_true: Binary labels.
        y_score: Continuous scores aligned to `y_true`.
        metric_fn: Metric function applied as `metric_fn(y_true, y_score)`.
        n_boot: Number of bootstrap replicates.
        seed: Deterministic bootstrap seed.

    Returns:
        The full-sample point estimate with 2.5 and 97.5 percentile bounds.
    """
    _validate_bootstrap_inputs(y_true=y_true, y_score=y_score)
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1.")

    rng = np.random.default_rng(seed)
    estimate = float(metric_fn(y_true, y_score))
    bootstrap_values = np.empty(n_boot, dtype=float)
    for bootstrap_idx in range(n_boot):
        sampled_indices = _draw_stratified_indices(y_true=y_true, rng=rng)
        bootstrap_values[bootstrap_idx] = float(metric_fn(y_true[sampled_indices], y_score[sampled_indices]))

    ci_low, ci_high = np.percentile(bootstrap_values, [2.5, 97.5])
    return BootstrapResult(estimate=estimate, ci_low=float(ci_low), ci_high=float(ci_high))


def paired_bootstrap_delta(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int,
    seed: int,
) -> BootstrapResult:
    """Estimate a paired metric delta and percentile CI by shared bootstrap draws.

    Operational-plan references: §2.8 and §5.6.

    Args:
        y_true: Binary labels.
        score_a: Scores from model A.
        score_b: Scores from model B.
        metric_fn: Metric function applied as `metric_fn(y_true, y_score)`.
        n_boot: Number of bootstrap replicates.
        seed: Deterministic bootstrap seed.

    Returns:
        The paired delta estimate `metric_a - metric_b` with percentile bounds.
    """
    _validate_bootstrap_inputs(y_true=y_true, y_score=score_a)
    _validate_bootstrap_inputs(y_true=y_true, y_score=score_b)
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1.")

    rng = np.random.default_rng(seed)
    estimate = float(metric_fn(y_true, score_a) - metric_fn(y_true, score_b))
    bootstrap_values = np.empty(n_boot, dtype=float)
    for bootstrap_idx in range(n_boot):
        sampled_indices = _draw_stratified_indices(y_true=y_true, rng=rng)
        bootstrap_values[bootstrap_idx] = float(
            metric_fn(y_true[sampled_indices], score_a[sampled_indices])
            - metric_fn(y_true[sampled_indices], score_b[sampled_indices])
        )

    ci_low, ci_high = np.percentile(bootstrap_values, [2.5, 97.5])
    return BootstrapResult(estimate=estimate, ci_low=float(ci_low), ci_high=float(ci_high))
