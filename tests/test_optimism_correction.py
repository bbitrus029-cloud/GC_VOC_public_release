from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.config import PROJECT_ROOT


def _import_module_or_fail():
    """Import the optimism-correction module or fail with a readable message."""
    try:
        return importlib.import_module("src.analysis.optimism_correction")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Missing module src.analysis.optimism_correction: {exc}", pytrace=False)


class _FakeMiniRocketTransformer:
    """Deterministic stand-in transformer for fast fixture-based tests."""

    def __init__(self) -> None:
        self.fit_calls = 0
        self.feature_columns = ["kernel_00000", "kernel_00001"]

    def fit_transform(self, x: np.ndarray) -> pd.DataFrame:
        self.fit_calls += 1
        return self.transform(x)

    def transform(self, x: np.ndarray) -> pd.DataFrame:
        values = np.column_stack(
            [
                x.mean(axis=(1, 2)),
                x[:, :, 0].mean(axis=1),
            ]
        )
        return pd.DataFrame(values, columns=self.feature_columns)


def _build_fixture_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Create a small balanced raw-signal fixture with strong class separation."""
    negative = np.zeros((6, 4, 2), dtype=float)
    positive = np.ones((6, 4, 2), dtype=float)

    negative += np.linspace(0.00, 0.05, num=6, dtype=float).reshape(-1, 1, 1)
    positive += np.linspace(0.00, 0.05, num=6, dtype=float).reshape(-1, 1, 1)

    x = np.concatenate([negative, positive], axis=0)
    y = np.array([0] * len(negative) + [1] * len(positive), dtype=np.int64)
    return x, y


def _build_frozen_config(module):
    """Construct the minimal frozen-config fixture for the desired API."""
    return module.FrozenMiniRocketRidgeConfig(
        transformer_params={
            "num_kernels": 84,
            "max_dilations_per_kernel": 32,
            "random_state": 42,
        },
        ridge_params={
            "alpha": 1.0,
            "class_weight": "balanced",
            "copy_X": True,
            "fit_intercept": True,
            "max_iter": None,
            "positive": False,
            "random_state": None,
            "solver": "auto",
            "tol": 1e-4,
        },
    )


def test_optimism_sequence_is_bit_identical_for_seed_42() -> None:
    """Two full runs with the same seed should yield identical optimism draws."""
    module = _import_module_or_fail()
    x_train, y_train = _build_fixture_arrays()
    frozen_config = _build_frozen_config(module)

    first = module.run_optimism_correction_from_arrays(
        x_train=x_train,
        y_train=y_train,
        frozen_config=frozen_config,
        n_boot=25,
        random_seed=42,
        transformer_factory=lambda _: _FakeMiniRocketTransformer(),
    )
    second = module.run_optimism_correction_from_arrays(
        x_train=x_train,
        y_train=y_train,
        frozen_config=frozen_config,
        n_boot=25,
        random_seed=42,
        transformer_factory=lambda _: _FakeMiniRocketTransformer(),
    )

    assert np.array_equal(first.optimism_b, second.optimism_b)


def test_stratified_bootstrap_preserves_both_classes() -> None:
    """Every stratified bootstrap draw should keep both classes represented."""
    module = _import_module_or_fail()
    _, y_train = _build_fixture_arrays()
    rng = np.random.default_rng(42)

    for _ in range(20):
        sampled_indices = module._draw_stratified_bootstrap_indices(y_true=y_train, rng=rng)
        assert set(np.unique(y_train[sampled_indices])) == {0, 1}


def test_transform_is_fit_exactly_once_across_the_whole_procedure() -> None:
    """The MiniRocket transform should be fit exactly once per full run."""
    module = _import_module_or_fail()
    x_train, y_train = _build_fixture_arrays()
    frozen_config = _build_frozen_config(module)

    result = module.run_optimism_correction_from_arrays(
        x_train=x_train,
        y_train=y_train,
        frozen_config=frozen_config,
        n_boot=12,
        random_seed=42,
        transformer_factory=lambda _: _FakeMiniRocketTransformer(),
    )

    assert result.transform_fit_count == 1


def test_auc_helper_matches_sklearn_on_known_fixture() -> None:
    """The internal AUC helper should match sklearn exactly on a known fixture."""
    module = _import_module_or_fail()
    y_true = np.array([0, 0, 1, 1], dtype=np.int64)
    scores = np.array([0.10, 0.20, 0.70, 0.80], dtype=float)

    assert module._compute_auc_roc(y_true=y_true, decision_scores=scores) == roc_auc_score(y_true, scores)


def test_fixture_run_does_not_write_new_model_artifacts() -> None:
    """The fixture run must not create files under outputs/models."""
    module = _import_module_or_fail()
    x_train, y_train = _build_fixture_arrays()
    frozen_config = _build_frozen_config(module)
    models_dir = PROJECT_ROOT / "outputs" / "models"
    before = sorted(path.name for path in models_dir.iterdir())

    module.run_optimism_correction_from_arrays(
        x_train=x_train,
        y_train=y_train,
        frozen_config=frozen_config,
        n_boot=10,
        random_seed=42,
        transformer_factory=lambda _: _FakeMiniRocketTransformer(),
    )

    after = sorted(path.name for path in models_dir.iterdir())
    assert after == before


def test_apparent_and_corrected_auc_are_finite_and_ordered() -> None:
    """The corrected AUC should be finite, bounded, and not exceed the apparent AUC."""
    module = _import_module_or_fail()
    x_train, y_train = _build_fixture_arrays()
    frozen_config = _build_frozen_config(module)

    result = module.run_optimism_correction_from_arrays(
        x_train=x_train,
        y_train=y_train,
        frozen_config=frozen_config,
        n_boot=25,
        random_seed=42,
        transformer_factory=lambda _: _FakeMiniRocketTransformer(),
    )

    assert np.isfinite(result.auc_full_apparent)
    assert np.isfinite(result.auc_optimism_corrected)
    assert 0.0 <= result.auc_full_apparent <= 1.0
    assert 0.0 <= result.auc_optimism_corrected <= 1.0
    assert result.auc_optimism_corrected <= result.auc_full_apparent
