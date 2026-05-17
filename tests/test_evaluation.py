from __future__ import annotations

import logging

import numpy as np
import pytest
from scipy.special import expit
from scipy.stats import fisher_exact
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from statsmodels.stats.multitest import multipletests

from src.config import RANDOM_SEED
from src.evaluation.calibration import (
    apply_platt_calibrator,
    compute_brier_before_after,
    compute_calibration_slope_intercept,
    fit_platt_calibrator,
)
from src.evaluation.metrics import (
    compute_calibration_metrics,
    compute_count_metrics,
    compute_full_metric_set,
    compute_threshold_dependent_metrics,
    compute_threshold_free_metrics,
)
from src.evaluation.statistical_tests import (
    compute_idi,
    compute_nri,
    compute_nri_idi,
    fisher_exact_2x2,
    holm_bonferroni,
    mcnemar_test,
    spearman_correlation,
)


@pytest.fixture
def binary_metric_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return a deterministic binary-evaluation fixture."""
    y_true = np.array([0, 0, 1, 1, 1, 0], dtype=int)
    p_raw = np.array([0.10, 0.80, 0.51, 0.90, 0.49, 0.20], dtype=float)
    p_cal = np.array([0.02, 0.60, 0.55, 0.95, 0.45, 0.05], dtype=float)
    threshold_p_raw = 0.50
    return y_true, p_raw, p_cal, threshold_p_raw


def test_compute_full_metric_set_matches_ground_truth(
    binary_metric_fixture: tuple[np.ndarray, np.ndarray, np.ndarray, float],
) -> None:
    """Full metric set should match sklearn/scipy ground truth."""
    y_true, p_raw, p_cal, threshold_p_raw = binary_metric_fixture

    metrics = compute_full_metric_set(
        y_true=y_true,
        p_raw=p_raw,
        p_cal=p_cal,
        threshold_p_raw=threshold_p_raw,
    )
    y_pred = (p_raw >= threshold_p_raw).astype(int)

    assert metrics["auc_roc"] == pytest.approx(roc_auc_score(y_true, p_raw))
    assert metrics["auc_pr"] == pytest.approx(average_precision_score(y_true, p_raw))
    assert metrics["brier_pre_platt"] == pytest.approx(brier_score_loss(y_true, p_raw))
    assert metrics["brier_post_platt"] == pytest.approx(brier_score_loss(y_true, p_cal))
    assert metrics["brier_delta"] == pytest.approx(
        brier_score_loss(y_true, p_cal) - brier_score_loss(y_true, p_raw)
    )
    assert metrics["sensitivity"] == pytest.approx(recall_score(y_true, y_pred))
    assert metrics["specificity"] == pytest.approx(recall_score(y_true, y_pred, pos_label=0))
    assert metrics["ppv"] == pytest.approx(precision_score(y_true, y_pred))
    assert metrics["npv"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(f1_score(y_true, y_pred))
    assert metrics["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y_true, y_pred))
    assert metrics["mcc"] == pytest.approx(matthews_corrcoef(y_true, y_pred))
    assert metrics["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))
    assert metrics["tp"] == 2
    assert metrics["tn"] == 2
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["n_total"] == 6
    assert metrics["n_pos"] == 3
    assert metrics["n_neg"] == 3
    assert metrics["prevalence"] == pytest.approx(0.5)


def test_compute_full_metric_set_uses_raw_space_for_threshold_metrics() -> None:
    """Thresholded metrics should be derived only from raw probabilities."""
    y_true = np.array([0, 1, 1, 0], dtype=int)
    p_raw = np.array([0.49, 0.51, 0.80, 0.20], dtype=float)
    p_cal = np.array([0.90, 0.91, 0.95, 0.05], dtype=float)

    full_metrics = compute_full_metric_set(
        y_true=y_true,
        p_raw=p_raw,
        p_cal=p_cal,
        threshold_p_raw=0.50,
    )
    raw_metrics = compute_threshold_dependent_metrics(y_true=y_true, p_raw=p_raw, threshold_p_raw=0.50)
    count_metrics = compute_count_metrics(y_true=y_true, p_raw=p_raw, threshold_p_raw=0.50)

    for key in ("sensitivity", "specificity", "ppv", "npv", "f1", "balanced_accuracy", "mcc", "accuracy"):
        assert full_metrics[key] == pytest.approx(raw_metrics[key])

    for key in ("tp", "tn", "fp", "fn"):
        assert full_metrics[key] == count_metrics[key]


def test_compute_full_metric_set_returns_nan_for_one_class_inputs(caplog: pytest.LogCaptureFixture) -> None:
    """Undefined metrics should return NaN and emit warnings."""
    y_true = np.zeros(5, dtype=int)
    p_raw = np.array([0.05, 0.10, 0.20, 0.30, 0.40], dtype=float)
    p_cal = np.array([0.06, 0.11, 0.21, 0.31, 0.41], dtype=float)

    with caplog.at_level(logging.WARNING):
        metrics = compute_full_metric_set(y_true=y_true, p_raw=p_raw, p_cal=p_cal, threshold_p_raw=0.50)

    assert np.isnan(metrics["auc_roc"])
    assert np.isnan(metrics["auc_pr"])
    assert np.isnan(metrics["sensitivity"])
    assert metrics["brier_pre_platt"] == pytest.approx(brier_score_loss(y_true, p_raw))
    assert metrics["brier_post_platt"] == pytest.approx(brier_score_loss(y_true, p_cal))
    assert any("undefined" in record.message.lower() for record in caplog.records)


def test_compute_calibration_slope_intercept_recovers_perfect_calibration() -> None:
    """Perfectly calibrated probabilities should yield slope ~1 and intercept ~0."""
    logit_values = np.linspace(-2.0, 2.0, 9)
    probabilities = expit(logit_values)

    repeated_probabilities = np.repeat(probabilities, 1000)
    y_true_blocks = [
        np.concatenate([np.ones(int(round(probability * 1000))), np.zeros(1000 - int(round(probability * 1000)))])
        for probability in probabilities
    ]
    y_true = np.concatenate(y_true_blocks).astype(int)

    metrics = compute_calibration_slope_intercept(p_cal=repeated_probabilities, y_true=y_true)

    assert metrics["slope"] == pytest.approx(1.0, abs=0.05)
    assert metrics["intercept"] == pytest.approx(0.0, abs=0.05)
    assert metrics["n_used"] == len(y_true)


def test_compute_brier_before_after_returns_zero_delta_for_identity() -> None:
    """Identity calibration should keep the Brier score unchanged."""
    y_true = np.array([0, 1, 0, 1], dtype=int)
    p_raw = np.array([0.10, 0.80, 0.30, 0.70], dtype=float)

    brier_metrics = compute_brier_before_after(p_raw=p_raw, p_cal=p_raw.copy(), y_true=y_true)

    assert brier_metrics["brier_pre_platt"] == pytest.approx(brier_score_loss(y_true, p_raw))
    assert brier_metrics["brier_post_platt"] == pytest.approx(brier_score_loss(y_true, p_raw))
    assert brier_metrics["brier_delta"] == pytest.approx(0.0)


def test_compute_threshold_free_metrics_returns_pre_platt_brier() -> None:
    """Threshold-free helper should expose raw-space Brier semantics only."""
    y_true = np.array([0, 0, 1, 1], dtype=int)
    p_raw = np.array([0.10, 0.40, 0.60, 0.90], dtype=float)

    metrics = compute_threshold_free_metrics(y_true=y_true, p_raw=p_raw)

    assert set(metrics) == {"auc_roc", "auc_pr", "brier_pre_platt"}
    assert metrics["brier_pre_platt"] == pytest.approx(brier_score_loss(y_true, p_raw))


def test_compute_calibration_metrics_returns_pre_and_post_platt_brier() -> None:
    """Calibration helper should return both Brier states plus slope/intercept."""
    y_true = np.array([0, 0, 1, 1], dtype=int)
    p_raw = np.array([0.10, 0.30, 0.70, 0.90], dtype=float)
    p_cal = np.array([0.05, 0.20, 0.80, 0.95], dtype=float)

    metrics = compute_calibration_metrics(y_true=y_true, p_raw=p_raw, p_cal=p_cal)

    assert metrics["brier_pre_platt"] == pytest.approx(brier_score_loss(y_true, p_raw))
    assert metrics["brier_post_platt"] == pytest.approx(brier_score_loss(y_true, p_cal))
    assert metrics["brier_delta"] == pytest.approx(
        brier_score_loss(y_true, p_cal) - brier_score_loss(y_true, p_raw)
    )
    assert set(metrics) == {
        "brier_pre_platt",
        "brier_post_platt",
        "brier_delta",
        "calibration_slope",
        "calibration_intercept",
    }


def test_compute_count_metrics_returns_confusion_counts() -> None:
    """Count helper should expose confusion counts and prevalence."""
    y_true = np.array([0, 1, 0, 1], dtype=int)
    p_raw = np.array([0.20, 0.80, 0.60, 0.70], dtype=float)

    metrics = compute_count_metrics(y_true=y_true, p_raw=p_raw, threshold_p_raw=0.50)

    assert metrics == {
        "tp": 2,
        "tn": 1,
        "fp": 1,
        "fn": 0,
        "n_total": 4,
        "n_pos": 2,
        "n_neg": 2,
        "prevalence": 0.5,
    }


def test_holm_bonferroni_matches_statsmodels_reference() -> None:
    """Holm-Bonferroni adjustment should match statsmodels."""
    raw_p_values = [
        ("test_a", 0.01),
        ("test_b", 0.04),
        ("test_c", 0.03),
        ("test_d", 0.20),
    ]

    adjusted = holm_bonferroni(raw_p_values, alpha=0.05)
    reference_reject, reference_adjusted, _, _ = multipletests(
        [value for _, value in raw_p_values],
        alpha=0.05,
        method="holm",
    )

    assert [item["test_name"] for item in adjusted] == [name for name, _ in raw_p_values]
    assert [item["rank"] for item in adjusted] == [1, 3, 2, 4]
    assert [item["holm_adjusted_p"] for item in adjusted] == pytest.approx(reference_adjusted)
    assert [item["reject_null"] for item in adjusted] == reference_reject.tolist()


def test_compute_nri_and_idi_cover_identity_and_deterministic_improvement() -> None:
    """NRI and IDI should have their canonical edge-case values."""
    y_true = np.array([0, 0, 1, 1], dtype=int)
    p_old = np.full_like(y_true, 0.5, dtype=float)
    p_new_identity = p_old.copy()
    p_new_perfect = y_true.astype(float)

    nri_identity = compute_nri(y_true=y_true, p_old=p_old, p_new=p_new_identity)
    idi_identity = compute_idi(y_true=y_true, p_old=p_old, p_new=p_new_identity)
    combined_identity = compute_nri_idi(y_true=y_true, p_old=p_old, p_new=p_new_identity)

    assert nri_identity["nri"] == pytest.approx(0.0)
    assert nri_identity["nri_events"] == pytest.approx(0.0)
    assert nri_identity["nri_nonevents"] == pytest.approx(0.0)
    assert idi_identity["idi"] == pytest.approx(0.0)
    assert idi_identity["idi_events"] == pytest.approx(0.0)
    assert idi_identity["idi_nonevents"] == pytest.approx(0.0)
    assert combined_identity["nri"] == pytest.approx(0.0)
    assert combined_identity["idi"] == pytest.approx(0.0)

    nri_perfect = compute_nri(y_true=y_true, p_old=p_old, p_new=p_new_perfect)
    idi_perfect = compute_idi(y_true=y_true, p_old=p_old, p_new=p_new_perfect)

    assert nri_perfect["nri"] == pytest.approx(2.0)
    assert nri_perfect["nri_events"] == pytest.approx(1.0)
    assert nri_perfect["nri_nonevents"] == pytest.approx(1.0)
    assert idi_perfect["idi"] > 0.0


def test_fisher_exact_2x2_matches_scipy_reference() -> None:
    """Fisher exact helper should match scipy on standard tables."""
    table = np.array([[8, 2], [1, 5]], dtype=int)

    result = fisher_exact_2x2(table)
    reference_odds_ratio, reference_p_value = fisher_exact(table, alternative="two-sided")

    assert result["odds_ratio"] == pytest.approx(reference_odds_ratio)
    assert result["p_two_sided"] == pytest.approx(reference_p_value)


def test_spearman_correlation_uses_pairwise_complete_observations() -> None:
    """Spearman helper should drop NaN pairs and report the used sample size."""
    x = np.array([1.0, 2.0, 3.0, 4.0, np.nan], dtype=float)
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=float)

    result = spearman_correlation(x=x, y=y)

    assert result["rho"] == pytest.approx(1.0)
    assert result["n"] == 4


def test_fit_and_apply_platt_calibrator_preserve_rank_order() -> None:
    """A sigmoid calibrator should preserve the rank order of monotonic scores."""
    p_raw = np.linspace(0.05, 0.95, 30)
    y_true = (p_raw > 0.55).astype(int)

    calibrator = fit_platt_calibrator(p_raw=p_raw, y_true=y_true, seed=RANDOM_SEED)
    p_cal = apply_platt_calibrator(calibrator=calibrator, p_raw=p_raw)

    assert np.array_equal(np.argsort(p_raw), np.argsort(p_cal))
    assert np.all((p_cal >= 0.0) & (p_cal <= 1.0))


def test_mcnemar_test_is_stubbed_when_not_prespecified() -> None:
    """McNemar should remain a documented stub when not used in this study."""
    with pytest.raises(NotImplementedError):
        mcnemar_test(
            y_true=np.array([0, 1], dtype=int),
            y_pred_a=np.array([0, 1], dtype=int),
            y_pred_b=np.array([0, 1], dtype=int),
        )
