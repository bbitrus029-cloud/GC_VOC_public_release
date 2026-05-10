"""Unit tests for environmental compensation fitting, application, and YAML I/O."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest

from src.config import RANDOM_SEED


def _import_env_compensation_symbols() -> tuple[Any, ...]:
    """Import environmental-compensation symbols lazily for TDD-friendly failures."""
    try:
        from src.config import GAS_CHANNELS, N_GAS_CHANNELS
        from src.data.env_compensation import EnvCompensationModel
    except (AttributeError, ImportError, ModuleNotFoundError) as exc:
        pytest.fail(f"Environmental compensation implementation is missing: {exc}", pytrace=False)

    return EnvCompensationModel, GAS_CHANNELS, N_GAS_CHANNELS


def _build_baseline_matrix(
    t_samples: NDArray[np.float64],
    h_samples: NDArray[np.float64],
    alpha: NDArray[np.float64],
    beta_t: NDArray[np.float64],
    beta_h: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Construct a synthetic baseline matrix from known coefficients."""
    return (
        alpha.reshape(1, -1)
        + np.outer(t_samples, beta_t)
        + np.outer(h_samples, beta_h)
    )


def _assert_models_close(left: Any, right: Any) -> None:
    """Assert that two compensation models carry numerically equivalent values."""
    assert np.allclose(left.alpha, right.alpha)
    assert np.allclose(left.beta_t, right.beta_t)
    assert np.allclose(left.beta_h, right.beta_h)
    assert np.allclose(left.adjusted_r2, right.adjusted_r2)
    assert tuple(left.applies_compensation) == tuple(right.applies_compensation)
    assert left.t_ref == pytest.approx(right.t_ref)
    assert left.h_ref == pytest.approx(right.h_ref)
    assert left.multicollinearity_fallback_used is right.multicollinearity_fallback_used


@pytest.fixture
def independent_environment_samples() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Provide non-collinear synthetic temperature and humidity summaries."""
    rng = np.random.default_rng(RANDOM_SEED)
    t_samples = np.linspace(20.0, 32.0, 36, dtype=np.float64)
    h_samples = 55.0 + rng.normal(loc=0.0, scale=2.5, size=t_samples.shape[0])
    return t_samples, h_samples


@pytest.fixture
def known_multivariable_baselines(
    independent_environment_samples: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Provide exact synthetic baselines with known multivariable coefficients."""
    _EnvCompensationModel, _GAS_CHANNELS, n_gas_channels = _import_env_compensation_symbols()
    t_samples, h_samples = independent_environment_samples
    alpha = np.linspace(-5.0, 5.0, n_gas_channels, dtype=np.float64)
    beta_t = np.array([0.9, -0.4, 0.6, -0.2, 0.3, 0.7, -0.5, 0.2, 0.15, 0.1, 0.0], dtype=np.float64)
    beta_h = np.array([0.15, 0.5, -0.3, 0.25, 0.4, -0.35, 0.1, 0.45, -0.2, 0.3, 0.0], dtype=np.float64)
    baseline_means = _build_baseline_matrix(t_samples, h_samples, alpha, beta_t, beta_h)
    baseline_means[:, -1] = alpha[-1]
    return baseline_means, t_samples, h_samples, alpha, beta_t, beta_h


@pytest.fixture
def highly_correlated_environment_samples() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Provide temperature and humidity summaries with absolute Pearson r above 0.80."""
    t_samples = np.linspace(18.0, 30.0, 40, dtype=np.float64)
    h_samples = 10.0 + 2.5 * t_samples + 0.2 * np.sin(t_samples)
    return t_samples, h_samples


def test_fit_recovers_known_coefficients_without_multicollinearity(
    known_multivariable_baselines: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ],
) -> None:
    """Fit should recover the planted coefficients on exact linear synthetic data."""
    EnvCompensationModel, _GAS_CHANNELS, _N_GAS_CHANNELS = _import_env_compensation_symbols()
    baseline_means, t_samples, h_samples, alpha, beta_t, beta_h = known_multivariable_baselines

    model = EnvCompensationModel.fit(
        baseline_means=baseline_means,
        t_samples=t_samples,
        h_samples=h_samples,
    )

    assert np.allclose(model.alpha[:-1], alpha[:-1], atol=1e-10)
    assert np.allclose(model.beta_t[:-1], beta_t[:-1], atol=1e-10)
    assert np.allclose(model.beta_h[:-1], beta_h[:-1], atol=1e-10)
    assert np.all(np.asarray(model.adjusted_r2[:-1]) > 0.999999999)
    assert tuple(model.applies_compensation[:-1]) == (True,) * 10
    assert model.multicollinearity_fallback_used is False
    assert model.t_ref == pytest.approx(float(np.median(t_samples)))
    assert model.h_ref == pytest.approx(float(np.median(h_samples)))


def test_fit_uses_univariate_fallback_when_environment_predictors_are_multicollinear(
    highly_correlated_environment_samples: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    """Fallback should zero the weaker predictor coefficient for each channel."""
    EnvCompensationModel, _GAS_CHANNELS, n_gas_channels = _import_env_compensation_symbols()
    t_samples, h_samples = highly_correlated_environment_samples
    alpha = np.linspace(1.0, 3.0, n_gas_channels, dtype=np.float64)
    beta_t = np.zeros(n_gas_channels, dtype=np.float64)
    beta_h = np.zeros(n_gas_channels, dtype=np.float64)
    beta_t[0] = 1.8
    beta_h[1] = -0.9
    beta_t[2:] = 0.4
    baseline_means = _build_baseline_matrix(t_samples, h_samples, alpha, beta_t, beta_h)

    model = EnvCompensationModel.fit(
        baseline_means=baseline_means,
        t_samples=t_samples,
        h_samples=h_samples,
    )

    assert model.multicollinearity_fallback_used is True
    assert model.beta_h[0] == pytest.approx(0.0)
    assert model.beta_t[0] != pytest.approx(0.0)
    assert model.beta_t[1] == pytest.approx(0.0)
    assert model.beta_h[1] != pytest.approx(0.0)
    assert np.all(
        np.logical_xor(
            np.isclose(model.beta_t, 0.0),
            np.isclose(model.beta_h, 0.0),
        )
    )


def test_fit_disables_compensation_for_channel_without_environment_dependence(
    known_multivariable_baselines: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ],
) -> None:
    """A constant-baseline channel should remain unadjusted under the R-squared gate."""
    EnvCompensationModel, _GAS_CHANNELS, _N_GAS_CHANNELS = _import_env_compensation_symbols()
    baseline_means, t_samples, h_samples, alpha, _beta_t, _beta_h = known_multivariable_baselines

    model = EnvCompensationModel.fit(
        baseline_means=baseline_means,
        t_samples=t_samples,
        h_samples=h_samples,
    )

    assert model.alpha[-1] == pytest.approx(alpha[-1])
    assert model.beta_t[-1] == pytest.approx(0.0)
    assert model.beta_h[-1] == pytest.approx(0.0)
    assert model.adjusted_r2[-1] <= 0.0
    assert model.applies_compensation[-1] is False


def test_apply_returns_float64_same_shape_and_does_not_mutate_input() -> None:
    """Applying compensation should preserve shape, dtype, sign, and input immutability."""
    EnvCompensationModel, _GAS_CHANNELS, n_gas_channels = _import_env_compensation_symbols()
    smoothed = np.full((8, n_gas_channels), 12.0, dtype=np.float64)
    smoothed[:, 0] = np.linspace(-8.0, -1.0, 8, dtype=np.float64)
    smoothed[:, 1] = np.linspace(3.0, 10.0, 8, dtype=np.float64)
    original = smoothed.copy()
    model = EnvCompensationModel(
        alpha=tuple(np.zeros(n_gas_channels, dtype=float)),
        beta_t=tuple(np.array([0.2, -0.1] + [0.0] * 9, dtype=float)),
        beta_h=tuple(np.array([0.05, 0.2] + [0.0] * 9, dtype=float)),
        adjusted_r2=tuple(np.array([0.8, 0.7] + [0.0] * 9, dtype=float)),
        applies_compensation=(True, True) + (False,) * 9,
        t_ref=25.0,
        h_ref=60.0,
        multicollinearity_fallback_used=False,
    )

    compensated = model.apply(smoothed_gas=smoothed, t_sample=26.0, h_sample=61.0)

    assert compensated.shape == smoothed.shape
    assert compensated.dtype == np.float64
    assert np.array_equal(smoothed, original)
    assert np.min(compensated[:, 0]) < 0.0
    assert np.max(compensated[:, 1]) > 0.0
    assert np.allclose(compensated[:, 2:], smoothed[:, 2:])


def test_yaml_round_trip_preserves_model_values() -> None:
    """YAML serialization helpers should round-trip the full model payload."""
    EnvCompensationModel, _GAS_CHANNELS, n_gas_channels = _import_env_compensation_symbols()
    model = EnvCompensationModel(
        alpha=tuple(np.linspace(-1.0, 1.0, n_gas_channels, dtype=float)),
        beta_t=tuple(np.linspace(0.1, 1.1, n_gas_channels, dtype=float)),
        beta_h=tuple(np.linspace(-0.4, 0.6, n_gas_channels, dtype=float)),
        adjusted_r2=tuple(np.linspace(-0.2, 0.9, n_gas_channels, dtype=float)),
        applies_compensation=tuple(index % 2 == 0 for index in range(n_gas_channels)),
        t_ref=23.5,
        h_ref=58.25,
        multicollinearity_fallback_used=True,
    )

    serialized = model.to_yaml_dict()
    restored = EnvCompensationModel.from_yaml_dict(serialized)

    _assert_models_close(restored, model)
