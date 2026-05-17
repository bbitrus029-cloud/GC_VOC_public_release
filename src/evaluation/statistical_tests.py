"""Statistical tests for audit-time evaluation."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.stats import fisher_exact, norm, spearmanr


LOGGER = logging.getLogger(__name__)
DEGENERATE_VARIANCE_EPSILON = 1e-12


def _validate_aligned_vectors(**vectors: np.ndarray) -> None:
    """Validate one-dimensional aligned numpy arrays."""
    lengths: set[int] = set()
    for name, vector in vectors.items():
        if not isinstance(vector, np.ndarray) or vector.ndim != 1:
            raise TypeError(f"{name} must be a one-dimensional numpy array.")
        lengths.add(len(vector))

    if not lengths or min(lengths) == 0:
        raise ValueError("Input vectors must be non-empty.")
    if len(lengths) != 1:
        raise ValueError("All input vectors must have the same length.")


def _coerce_strict_binary_labels(y_true: np.ndarray) -> np.ndarray:
    """Coerce a binary label vector to {0, 1}."""
    unique_classes = np.unique(y_true)
    if len(unique_classes) != 2:
        raise ValueError("y_true must contain exactly two classes.")

    return (y_true == unique_classes.max()).astype(int)


def _compute_midrank(values: np.ndarray) -> np.ndarray:
    """Compute one-based midranks for a vector with ties."""
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_midranks = np.zeros(len(values), dtype=float)

    start = 0
    while start < len(values):
        end = start
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        sorted_midranks[start:end] = 0.5 * (start + end - 1) + 1.0
        start = end

    midranks = np.empty(len(values), dtype=float)
    midranks[sorted_indices] = sorted_midranks
    return midranks


def _fast_delong(predictions_sorted: np.ndarray, positive_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute DeLong AUCs and covariance matrix for paired score vectors."""
    negative_count = predictions_sorted.shape[1] - positive_count
    positive_scores = predictions_sorted[:, :positive_count]
    negative_scores = predictions_sorted[:, positive_count:]

    tx = np.empty_like(positive_scores, dtype=float)
    ty = np.empty_like(negative_scores, dtype=float)
    tz = np.empty_like(predictions_sorted, dtype=float)

    for classifier_idx in range(predictions_sorted.shape[0]):
        tx[classifier_idx] = _compute_midrank(positive_scores[classifier_idx])
        ty[classifier_idx] = _compute_midrank(negative_scores[classifier_idx])
        tz[classifier_idx] = _compute_midrank(predictions_sorted[classifier_idx])

    aucs = tz[:, :positive_count].sum(axis=1) / (positive_count * negative_count)
    aucs -= (positive_count + 1.0) / (2.0 * negative_count)

    v01 = (tz[:, :positive_count] - tx) / negative_count
    v10 = 1.0 - (tz[:, positive_count:] - ty) / positive_count

    covariance_positive = np.atleast_2d(np.cov(v01))
    covariance_negative = np.atleast_2d(np.cov(v10))
    covariance = covariance_positive / positive_count + covariance_negative / negative_count
    return aucs, covariance


def delong_roc_test(y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray) -> dict[str, float]:
    """Run a paired DeLong ROC comparison.

    Args:
        y_true: Binary ground-truth labels.
        score_a: Scores from model A.
        score_b: Scores from model B.

    Returns:
        A dictionary containing AUCs, covariance terms, z-statistic, and the
        two-sided p-value.
    """
    _validate_aligned_vectors(y_true=y_true, score_a=score_a, score_b=score_b)
    binary_labels = _coerce_strict_binary_labels(y_true)
    positive_count = int(binary_labels.sum())
    negative_count = len(binary_labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("DeLong requires both positive and negative samples.")

    positive_first_order = np.argsort(-binary_labels)
    predictions_sorted = np.vstack([score_a[positive_first_order], score_b[positive_first_order]])
    aucs, covariance = _fast_delong(predictions_sorted=predictions_sorted, positive_count=positive_count)

    var_a = float(covariance[0, 0])
    var_b = float(covariance[1, 1])
    cov_ab = float(covariance[0, 1])
    delta_auc = float(aucs[0] - aucs[1])
    var_delta = float(var_a + var_b - (2.0 * cov_ab))
    if var_delta < DEGENERATE_VARIANCE_EPSILON:
        LOGGER.warning("DeLong variance is degenerate; returning z=0 and p_two_sided=1.0.")
        z_score = 0.0
        p_two_sided = 1.0
    else:
        z_score = float(delta_auc / np.sqrt(var_delta))
        p_two_sided = float(2.0 * norm.sf(abs(z_score)))

    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "delta_auc": delta_auc,
        "var_a": var_a,
        "var_b": var_b,
        "cov_ab": cov_ab,
        "var_delta": var_delta,
        "z": z_score,
        "p_two_sided": p_two_sided,
    }


def jonckheere_terpstra(values: np.ndarray, ordered_groups: np.ndarray) -> dict[str, float]:
    """Run the Jonckheere-Terpstra trend test for ordered groups.

    Implementation follows the asymptotic-normal approximation used by
    `npordtests::JtTest`, including its upper-tail p-value convention. The
    reported `p_one_sided_upper` corresponds to npordtests' `Asymp. p-value`
    field and is the plan-relevant quantity for prespecified directional
    trend tests (operational plan §2.19).

    The implementation has no continuity correction.

    Args:
        values: Numeric observations.
        ordered_groups: Ordered group labels aligned to ``values``. Group
            order is taken as ``np.unique(ordered_groups)`` ascending.

    Returns:
        A dictionary containing:
        - ``J``: integer JT statistic with 0.5-weighted ties.
        - ``mean_j``: null-hypothesis mean of J.
        - ``variance_j``: null-hypothesis variance of J under no-ties.
        - ``z``: standardized statistic ``(J - mean_j) / sqrt(variance_j)``.
        - ``p_one_sided_upper``: upper-tail one-sided p-value
          ``P(Z >= z_obs)``, matching npordtests' `Asymp. p-value` output.
          This is the quantity anchored against npordtests in
          ``tests/test_jt_npordtests.py`` and used by plan §2.19.
        - ``p_two_sided``: two-sided p-value
          ``2 * P(Z >= |z_obs|)``, provided for completeness only.
    """
    _validate_aligned_vectors(values=values, ordered_groups=ordered_groups)

    sorted_group_labels = np.unique(ordered_groups)
    group_values = [values[ordered_groups == label] for label in sorted_group_labels]
    group_sizes = np.array([len(group) for group in group_values], dtype=float)
    if np.any(group_sizes == 0):
        raise ValueError("Each ordered group must contain at least one observation.")

    jt_statistic = 0.0
    for left_idx in range(len(group_values) - 1):
        for right_idx in range(left_idx + 1, len(group_values)):
            left_group = group_values[left_idx]
            right_group = group_values[right_idx]
            pairwise_differences = right_group[:, None] - left_group[None, :]
            jt_statistic += float(np.sum(pairwise_differences > 0))
            jt_statistic += float(0.5 * np.sum(pairwise_differences == 0))

    total_n = float(len(values))
    mean_j = float((total_n**2 - np.sum(group_sizes**2)) / 4.0)
    variance_j = float(
        (
            total_n**2 * (2.0 * total_n + 3.0)
            - np.sum(group_sizes**2 * (2.0 * group_sizes + 3.0))
        )
        / 72.0
    )
    if variance_j < DEGENERATE_VARIANCE_EPSILON:
        LOGGER.warning("Jonckheere-Terpstra variance is degenerate; returning z=0 and p=1.0.")
        z_score = 0.0
        p_one_sided_upper = 1.0
        p_two_sided = 1.0
    else:
        z_score = float((jt_statistic - mean_j) / np.sqrt(variance_j))
        p_one_sided_upper = float(norm.sf(z_score))
        p_two_sided = float(2.0 * norm.sf(abs(z_score)))

    return {
        "J": int(round(jt_statistic)),
        "mean_j": mean_j,
        "variance_j": variance_j,
        "z": z_score,
        "p_one_sided_upper": p_one_sided_upper,
        "p_two_sided": p_two_sided,
    }


def holm_bonferroni(
    p_values: Sequence[tuple[str, float]],
    alpha: float = 0.05,
) -> list[dict[str, Any]]:
    """Apply Holm-Bonferroni adjustment to named p-values.

    Args:
        p_values: Sequence of `(test_name, raw_p)` tuples in original test order.
        alpha: Family-wise error-rate target.

    Returns:
        A list of dictionaries in the original input order containing
        `test_name`, `raw_p`, `rank`, `holm_adjusted_p`, and `reject_null`.
    """
    if not p_values:
        raise ValueError("p_values must be non-empty.")

    indexed_p_values = [
        {"index": index, "test_name": test_name, "raw_p": float(raw_p)}
        for index, (test_name, raw_p) in enumerate(p_values)
    ]
    sorted_p_values = sorted(indexed_p_values, key=lambda item: item["raw_p"])
    m_tests = len(sorted_p_values)

    running_max = 0.0
    for rank, item in enumerate(sorted_p_values, start=1):
        holm_value = min((m_tests - rank + 1) * item["raw_p"], 1.0)
        running_max = max(running_max, holm_value)
        item["rank"] = rank
        item["holm_adjusted_p"] = running_max
        item["reject_null"] = bool(item["holm_adjusted_p"] <= alpha)

    sorted_by_input = sorted(sorted_p_values, key=lambda item: item["index"])
    return [
        {
            "test_name": item["test_name"],
            "raw_p": item["raw_p"],
            "rank": item["rank"],
            "holm_adjusted_p": item["holm_adjusted_p"],
            "reject_null": item["reject_null"],
        }
        for item in sorted_by_input
    ]


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Compute Spearman correlation with pairwise-complete handling."""
    _validate_aligned_vectors(x=x, y=y)

    valid_mask = ~(np.isnan(x) | np.isnan(y))
    filtered_x = x[valid_mask]
    filtered_y = y[valid_mask]
    if len(filtered_x) < 2:
        LOGGER.warning("Spearman correlation is undefined with fewer than two complete observations.")
        return {"rho": float(np.nan), "p_two_sided": float(np.nan), "n": int(len(filtered_x))}

    rho, p_value = spearmanr(filtered_x, filtered_y)
    return {"rho": float(rho), "p_two_sided": float(p_value), "n": int(len(filtered_x))}


def compute_nri(y_true: np.ndarray, p_old: np.ndarray, p_new: np.ndarray) -> dict[str, float]:
    """Compute continuous net reclassification improvement.

    Args:
        y_true: Binary ground-truth labels.
        p_old: Baseline model probabilities.
        p_new: New model probabilities.

    Returns:
        A dictionary containing overall NRI, event/non-event components, and
        event counts. This is the continuous NRI of Pencina 2011 with no
        category cutpoints.
    """
    _validate_aligned_vectors(y_true=y_true, p_old=p_old, p_new=p_new)
    binary_labels = _coerce_strict_binary_labels(y_true)

    event_mask = binary_labels == 1
    nonevent_mask = binary_labels == 0
    n_events = int(np.sum(event_mask))
    n_nonevents = int(np.sum(nonevent_mask))
    if n_events == 0 or n_nonevents == 0:
        LOGGER.warning("Continuous NRI is undefined when one class is absent.")
        return {
            "nri": float(np.nan),
            "nri_events": float(np.nan),
            "nri_nonevents": float(np.nan),
            "n_events": n_events,
            "n_nonevents": n_nonevents,
        }

    delta_scores = p_new - p_old
    nri_events = float(np.mean(delta_scores[event_mask] > 0) - np.mean(delta_scores[event_mask] < 0))
    nri_nonevents = float(np.mean(delta_scores[nonevent_mask] < 0) - np.mean(delta_scores[nonevent_mask] > 0))
    return {
        "nri": float(nri_events + nri_nonevents),
        "nri_events": nri_events,
        "nri_nonevents": nri_nonevents,
        "n_events": n_events,
        "n_nonevents": n_nonevents,
    }


def compute_idi(y_true: np.ndarray, p_old: np.ndarray, p_new: np.ndarray) -> dict[str, float]:
    """Compute integrated discrimination improvement."""
    _validate_aligned_vectors(y_true=y_true, p_old=p_old, p_new=p_new)
    binary_labels = _coerce_strict_binary_labels(y_true)

    event_mask = binary_labels == 1
    nonevent_mask = binary_labels == 0
    if not np.any(event_mask) or not np.any(nonevent_mask):
        LOGGER.warning("IDI is undefined when one class is absent.")
        return {
            "idi": float(np.nan),
            "idi_events": float(np.nan),
            "idi_nonevents": float(np.nan),
        }

    idi_events = float(np.mean(p_new[event_mask]) - np.mean(p_old[event_mask]))
    idi_nonevents = float(np.mean(p_new[nonevent_mask]) - np.mean(p_old[nonevent_mask]))
    return {
        "idi": float(idi_events - idi_nonevents),
        "idi_events": idi_events,
        "idi_nonevents": idi_nonevents,
    }


def compute_nri_idi(y_true: np.ndarray, p_old: np.ndarray, p_new: np.ndarray) -> dict[str, float]:
    """Compute both continuous NRI and IDI in one call."""
    return {
        **compute_nri(y_true=y_true, p_old=p_old, p_new=p_new),
        **compute_idi(y_true=y_true, p_old=p_old, p_new=p_new),
    }


def fisher_exact_2x2(table: np.ndarray | tuple[int, int, int, int]) -> dict[str, float]:
    """Compute Fisher's exact test for a 2x2 table.

    Args:
        table: A 2x2 numpy array or a 4-tuple of integer counts.

    Returns:
        A dictionary containing `odds_ratio` and `p_two_sided`.
    """
    table_array = np.asarray(table, dtype=float)
    if table_array.shape == (4,):
        table_array = table_array.reshape(2, 2)
    if table_array.shape != (2, 2):
        raise ValueError("table must be a 2x2 array or a 4-tuple.")

    exact_odds_ratio, p_value = fisher_exact(table_array, alternative="two-sided")
    if np.any(table_array == 0):
        LOGGER.warning("Zero cell detected; applying Haldane correction to the odds ratio.")
        corrected_table = table_array + 0.5
        odds_ratio = float((corrected_table[0, 0] * corrected_table[1, 1]) / (corrected_table[0, 1] * corrected_table[1, 0]))
    else:
        odds_ratio = float(exact_odds_ratio)

    return {"odds_ratio": odds_ratio, "p_two_sided": float(p_value)}


def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> dict[str, float]:
    """McNemar is intentionally stubbed because the plan does not use it."""
    raise NotImplementedError(
        "McNemar is not used by any prespecified analysis in this study; remove if unneeded."
    )
