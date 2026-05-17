from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.training.common import (
    FoldGenerator,
    MetricsCalculator,
    bootstrap_metric,
    create_dl_holdout_indices,
    fit_pipeline_a_normalizer,
    fit_pipeline_b_normalizer,
    paired_bootstrap_delta,
)
from src.training.common.fold_generator import _stratify_with_merge_ladder


@pytest.fixture
def train_test_fixture() -> tuple[np.ndarray, np.ndarray]:
    """Return labels and disease categories for split tests."""
    disease_categories = np.array(
        [0] * 4
        + [1] * 4
        + [2] * 4
        + [3] * 5
        + [4] * 5
        + [5] * 5,
        dtype=int,
    )
    labels = (disease_categories >= 3).astype(int)
    return labels, disease_categories


@pytest.fixture
def outer_fold_fixture() -> tuple[np.ndarray, np.ndarray]:
    """Return labels and disease categories for outer-fold tests."""
    disease_categories = np.repeat(np.arange(6, dtype=int), 10)
    labels = (disease_categories >= 3).astype(int)
    return labels, disease_categories


@pytest.fixture
def pipeline_a_fixture() -> np.ndarray:
    """Return a Pipeline A tensor with one constant channel."""
    rng = np.random.default_rng(7)
    x = rng.normal(loc=0.0, scale=1.0, size=(12, 8, 11))
    x[..., 0] = 3.14
    return x


@pytest.fixture
def pipeline_b_fixture() -> tuple[pd.DataFrame, list[str]]:
    """Return a Pipeline B frame with one constant feature."""
    rng = np.random.default_rng(11)
    feature_names = [f"f_{index:02d}" for index in range(99)]
    x = pd.DataFrame(
        rng.normal(loc=0.0, scale=1.0, size=(20, 99)),
        columns=feature_names,
    )
    x["f_00"] = 5.0
    return x, feature_names


@pytest.fixture
def binary_metric_fixture() -> tuple[np.ndarray, np.ndarray, float]:
    """Return a small binary classification example."""
    y_true = np.array([0, 0, 1, 1, 1, 0], dtype=int)
    p_raw = np.array([0.10, 0.40, 0.35, 0.80, 0.65, 0.20], dtype=float)
    threshold = 0.50
    return y_true, p_raw, threshold


def test_generate_train_test_split_is_deterministic_and_stratified(
    train_test_fixture: tuple[np.ndarray, np.ndarray],
) -> None:
    """Train/test split should be deterministic, exclusive, and label-balanced."""
    labels, disease_categories = train_test_fixture
    generator = FoldGenerator()

    train_idx_a, test_idx_a = generator.generate_train_test_split(
        labels=labels,
        disease_categories=disease_categories,
        seed=42,
    )
    train_idx_b, test_idx_b = generator.generate_train_test_split(
        labels=labels,
        disease_categories=disease_categories,
        seed=42,
    )

    assert np.array_equal(train_idx_a, train_idx_b)
    assert np.array_equal(test_idx_a, test_idx_b)
    assert len(train_idx_a) == 21
    assert len(test_idx_a) == 6
    assert set(train_idx_a).isdisjoint(set(test_idx_a))
    assert sorted(np.concatenate([train_idx_a, test_idx_a]).tolist()) == list(range(len(labels)))

    full_counts = np.bincount(labels, minlength=2)
    train_counts = np.bincount(labels[train_idx_a], minlength=2)
    test_counts = np.bincount(labels[test_idx_a], minlength=2)

    for class_index, full_count in enumerate(full_counts):
        expected_train = full_count * len(train_idx_a) / len(labels)
        expected_test = full_count * len(test_idx_a) / len(labels)
        assert abs(train_counts[class_index] - expected_train) <= 1
        assert abs(test_counts[class_index] - expected_test) <= 1


def test_generate_outer_folds_returns_five_deterministic_partitions(
    outer_fold_fixture: tuple[np.ndarray, np.ndarray],
) -> None:
    """Outer folds should partition the training set exactly and reproducibly."""
    labels, disease_categories = outer_fold_fixture
    generator = FoldGenerator()

    folds_a = generator.generate_outer_folds(
        train_labels=labels,
        train_disease_categories=disease_categories,
        seed=42,
    )
    folds_b = generator.generate_outer_folds(
        train_labels=labels,
        train_disease_categories=disease_categories,
        seed=42,
    )

    assert len(folds_a) == 5
    assert len(folds_b) == 5

    all_val_indices: list[int] = []
    for fold_a, fold_b in zip(folds_a, folds_b, strict=True):
        assert fold_a.outer_fold == fold_b.outer_fold
        assert np.array_equal(fold_a.train_idx, fold_b.train_idx)
        assert np.array_equal(fold_a.val_idx, fold_b.val_idx)
        assert len(fold_a.val_idx) == 12
        assert len(fold_a.inner_folds) == 3
        assert set(fold_a.train_idx).isdisjoint(set(fold_a.val_idx))
        all_val_indices.extend(fold_a.val_idx.tolist())

    assert sorted(all_val_indices) == list(range(len(labels)))


def test_stratify_with_merge_ladder_logs_merge_event(caplog: pytest.LogCaptureFixture) -> None:
    """Merge ladder should collapse sparse LGIN strata and log the event."""
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=int)
    disease_categories = np.array([0, 0, 1, 1, 2, 3, 3, 4, 4, 5], dtype=int)

    with caplog.at_level(logging.INFO):
        stratify_vector = _stratify_with_merge_ladder(
            labels=labels,
            disease_categories=disease_categories,
            minimum_count=2,
        )

    assert stratify_vector.shape == labels.shape
    assert any("merge" in record.message.lower() for record in caplog.records)


def test_pipeline_a_normalizer_transforms_fit_tensor_without_mutation(
    pipeline_a_fixture: np.ndarray,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Pipeline A normalizer should z-score without mutating inputs."""
    x_train = pipeline_a_fixture.copy()
    x_before = x_train.copy()

    with caplog.at_level(logging.WARNING):
        normalizer = fit_pipeline_a_normalizer(x_train)

    transformed = normalizer.transform(x_train)

    assert np.array_equal(x_train, x_before)
    assert transformed.shape == x_train.shape
    assert np.allclose(transformed[..., 0], 0.0)
    assert np.allclose(transformed[..., 1:].mean(axis=(0, 1)), 0.0, atol=1e-10)
    assert np.allclose(transformed[..., 1:].std(axis=(0, 1)), 1.0, atol=1e-10)
    assert any("channel" in record.message.lower() for record in caplog.records)


def test_pipeline_b_normalizer_aligns_feature_order_without_mutation(
    pipeline_b_fixture: tuple[pd.DataFrame, list[str]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Pipeline B normalizer should preserve feature alignment and clamp constants."""
    x_train, feature_names = pipeline_b_fixture
    x_before = x_train.copy(deep=True)

    with caplog.at_level(logging.WARNING):
        normalizer = fit_pipeline_b_normalizer(x_train, feature_names)

    shuffled = x_train[feature_names[::-1]]
    transformed = normalizer.transform(shuffled)

    pd.testing.assert_frame_equal(x_train, x_before)
    assert transformed.columns.tolist() == feature_names
    assert np.allclose(transformed["f_00"].to_numpy(), 0.0)
    assert np.allclose(
        transformed[feature_names[1:]].to_numpy().mean(axis=0),
        np.zeros(98),
        atol=1e-10,
    )
    assert any("feature" in record.message.lower() for record in caplog.records)


def test_compute_binary_metrics_matches_sklearn(
    binary_metric_fixture: tuple[np.ndarray, np.ndarray, float],
) -> None:
    """Binary metrics should agree with sklearn on a small fixture."""
    y_true, p_raw, threshold = binary_metric_fixture
    calculator = MetricsCalculator()
    metrics = calculator.compute_binary_metrics(y_true=y_true, p_raw=p_raw, threshold=threshold)

    y_pred = (p_raw >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    assert metrics["auc_roc"] == pytest.approx(roc_auc_score(y_true, p_raw))
    assert metrics["auc_pr"] == pytest.approx(average_precision_score(y_true, p_raw))
    assert metrics["brier_pre_platt"] == pytest.approx(brier_score_loss(y_true, p_raw))
    assert metrics["sensitivity"] == pytest.approx(recall_score(y_true, y_pred))
    assert metrics["specificity"] == pytest.approx(tn / (tn + fp))
    assert metrics["ppv"] == pytest.approx(precision_score(y_true, y_pred))
    assert metrics["npv"] == pytest.approx(tn / (tn + fn))
    assert metrics["f1"] == pytest.approx(f1_score(y_true, y_pred))
    assert metrics["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y_true, y_pred))
    assert metrics["mcc"] == pytest.approx(matthews_corrcoef(y_true, y_pred))
    assert metrics["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))
    assert metrics["tp"] == tp
    assert metrics["tn"] == tn
    assert metrics["fp"] == fp
    assert metrics["fn"] == fn
    assert metrics["n_total"] == len(y_true)
    assert metrics["n_pos"] == int(y_true.sum())
    assert metrics["n_neg"] == int((1 - y_true).sum())
    assert metrics["prevalence"] == pytest.approx(float(y_true.mean()))


def test_compute_binary_metrics_returns_nan_for_missing_positive_class(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Degenerate no-positive case should warn and return NaN instead of raising."""
    calculator = MetricsCalculator()
    y_true = np.zeros(5, dtype=int)
    p_raw = np.array([0.05, 0.10, 0.20, 0.30, 0.40], dtype=float)

    with caplog.at_level(logging.WARNING):
        metrics = calculator.compute_binary_metrics(y_true=y_true, p_raw=p_raw, threshold=0.50)

    assert np.isnan(metrics["auc_roc"])
    assert np.isnan(metrics["auc_pr"])
    assert np.isnan(metrics["sensitivity"])
    assert metrics["brier_pre_platt"] == pytest.approx(brier_score_loss(y_true, p_raw))
    assert any("class" in record.message.lower() for record in caplog.records)


def test_bootstrap_metric_is_deterministic() -> None:
    """Bootstrap metric should be deterministic under a fixed seed."""
    y_true = np.array([0] * 12 + [1] * 12, dtype=int)
    y_score = np.linspace(0.05, 0.95, num=len(y_true))

    result_a = bootstrap_metric(
        y_true=y_true,
        y_score=y_score,
        metric_fn=roc_auc_score,
        n_boot=200,
        seed=42,
    )
    result_b = bootstrap_metric(
        y_true=y_true,
        y_score=y_score,
        metric_fn=roc_auc_score,
        n_boot=200,
        seed=42,
    )

    assert result_a == result_b


def test_paired_bootstrap_delta_is_zero_for_identical_scores() -> None:
    """Paired bootstrap delta should be exactly zero for identical score vectors."""
    y_true = np.array([0] * 10 + [1] * 10, dtype=int)
    y_score = np.linspace(0.10, 0.90, num=len(y_true))

    result = paired_bootstrap_delta(
        y_true=y_true,
        score_a=y_score,
        score_b=y_score.copy(),
        metric_fn=roc_auc_score,
        n_boot=200,
        seed=42,
    )

    assert result.estimate == pytest.approx(0.0)
    assert result.ci_low == pytest.approx(0.0)
    assert result.ci_high == pytest.approx(0.0)


def test_create_dl_holdout_indices_preserves_strata() -> None:
    """DL holdout split should be non-overlapping and disease-stratified."""
    train_idx = np.arange(48, dtype=int)
    disease_categories = np.repeat(np.arange(6, dtype=int), 8)

    kept_train_idx, holdout_idx = create_dl_holdout_indices(
        train_idx=train_idx,
        disease_categories=disease_categories,
        holdout_fraction=0.125,
        seed=42,
    )

    assert len(holdout_idx) == 6
    assert set(kept_train_idx).isdisjoint(set(holdout_idx))
    assert sorted(np.concatenate([kept_train_idx, holdout_idx]).tolist()) == train_idx.tolist()
    holdout_counts = np.bincount(disease_categories[holdout_idx], minlength=6)
    assert holdout_counts.tolist() == [1, 1, 1, 1, 1, 1]
