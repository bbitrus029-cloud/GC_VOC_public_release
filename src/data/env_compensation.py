"""Environmental compensation fitting, application, and YAML serialization."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from src.config import GAS_CHANNELS, N_GAS_CHANNELS
from src.data.discovery import (
    compute_pearson_correlation,
    fit_environmental_regression,
    select_univariate_fallback,
)


LOGGER = logging.getLogger(__name__)
R2_THRESHOLD = 0.0  # [signal-processing-literature: adjusted R² greater than zero indicates improvement over intercept-only model; replaces unsupported 0.10 threshold]
PEARSON_MULTICOLLINEARITY_THRESHOLD = 0.80  # [signal-processing-literature: common regression-screening heuristic, used cautiously; O'Brien 2007 caveat]


def _coerce_channel_values(values: Sequence[float], name: str) -> tuple[float, ...]:
    """Validate one per-channel numeric sequence."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}")
    if array.shape[0] != N_GAS_CHANNELS:
        raise ValueError(f"{name} must have length {N_GAS_CHANNELS}; got {array.shape[0]}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return tuple(float(value) for value in array)


def _coerce_channel_flags(values: Sequence[bool], name: str) -> tuple[bool, ...]:
    """Validate one per-channel boolean sequence."""
    array = np.asarray(values, dtype=bool)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}")
    if array.shape[0] != N_GAS_CHANNELS:
        raise ValueError(f"{name} must have length {N_GAS_CHANNELS}; got {array.shape[0]}")
    return tuple(bool(value) for value in array)


def _coerce_finite_scalar(value: float, name: str) -> float:
    """Validate one finite scalar."""
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite.")
    return scalar


def _coerce_baseline_matrix(baseline_means: NDArray[np.float64]) -> NDArray[np.float64]:
    """Validate the per-sample baseline matrix."""
    matrix = np.asarray(baseline_means, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"baseline_means must be two-dimensional; got shape {matrix.shape}")
    if matrix.shape[1] != N_GAS_CHANNELS:
        raise ValueError(
            f"baseline_means must have shape (n_samples, {N_GAS_CHANNELS}); got {matrix.shape}"
        )
    if matrix.shape[0] == 0:
        raise ValueError("baseline_means must contain at least one sample.")
    if not np.isfinite(matrix).all():
        raise ValueError("baseline_means must contain only finite values.")
    return matrix


def _coerce_environment_samples(values: NDArray[np.float64], name: str, n_samples: int) -> NDArray[np.float64]:
    """Validate one aligned environmental summary vector."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}")
    if array.shape[0] != n_samples:
        raise ValueError(f"{name} must have length {n_samples}; got {array.shape[0]}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


@dataclass(frozen=True)
class EnvCompensationModel:
    """Per-channel environmental compensation parameters and YAML payload.

    Attributes:
        alpha: Per-channel regression intercepts for `S1` through `S11`.
        beta_t: Per-channel temperature coefficients.
        beta_h: Per-channel humidity coefficients.
        adjusted_r2: Per-channel adjusted R-squared values.
        applies_compensation: Whether the compensation formula is active for each
            gas channel.
        t_ref: Cohort reference temperature summary.
        h_ref: Cohort reference humidity summary.
        multicollinearity_fallback_used: Whether the pairwise temperature-humidity
            multicollinearity guard forced univariate fitting.
        temperature_humidity_pearson_r: Pairwise Pearson correlation between the
            fitting-cohort temperature and humidity summaries.
        temperature_humidity_abs_pearson_r_threshold: Absolute Pearson threshold
            used to trigger the univariate fallback.
        r2_threshold: Adjusted-R-squared threshold used to activate
            compensation.
    """

    alpha: tuple[float, ...]
    beta_t: tuple[float, ...]
    beta_h: tuple[float, ...]
    adjusted_r2: tuple[float, ...]
    applies_compensation: tuple[bool, ...]
    t_ref: float
    h_ref: float
    multicollinearity_fallback_used: bool
    temperature_humidity_pearson_r: float = 0.0
    temperature_humidity_abs_pearson_r_threshold: float = PEARSON_MULTICOLLINEARITY_THRESHOLD
    r2_threshold: float = R2_THRESHOLD

    def __post_init__(self) -> None:
        """Validate and normalize all stored coefficients."""
        object.__setattr__(self, "alpha", _coerce_channel_values(self.alpha, "alpha"))
        object.__setattr__(self, "beta_t", _coerce_channel_values(self.beta_t, "beta_t"))
        object.__setattr__(self, "beta_h", _coerce_channel_values(self.beta_h, "beta_h"))
        object.__setattr__(self, "adjusted_r2", _coerce_channel_values(self.adjusted_r2, "adjusted_r2"))
        object.__setattr__(
            self,
            "applies_compensation",
            _coerce_channel_flags(self.applies_compensation, "applies_compensation"),
        )
        object.__setattr__(self, "t_ref", _coerce_finite_scalar(self.t_ref, "t_ref"))
        object.__setattr__(self, "h_ref", _coerce_finite_scalar(self.h_ref, "h_ref"))
        object.__setattr__(
            self,
            "temperature_humidity_pearson_r",
            _coerce_finite_scalar(self.temperature_humidity_pearson_r, "temperature_humidity_pearson_r"),
        )
        object.__setattr__(
            self,
            "temperature_humidity_abs_pearson_r_threshold",
            _coerce_finite_scalar(
                self.temperature_humidity_abs_pearson_r_threshold,
                "temperature_humidity_abs_pearson_r_threshold",
            ),
        )
        object.__setattr__(self, "r2_threshold", _coerce_finite_scalar(self.r2_threshold, "r2_threshold"))
        object.__setattr__(
            self,
            "multicollinearity_fallback_used",
            bool(self.multicollinearity_fallback_used),
        )

    @classmethod
    def fit(
        cls,
        baseline_means: NDArray[np.float64],
        t_samples: NDArray[np.float64],
        h_samples: NDArray[np.float64],
    ) -> EnvCompensationModel:
        """Fit per-channel environmental compensation from a QC-filtered cohort.

        Args:
            baseline_means: Smoothed per-sample gas-channel baseline means with
                shape `(n_samples, 11)`.
            t_samples: Per-sample temperature baseline summaries with shape
                `(n_samples,)`.
            h_samples: Per-sample humidity baseline summaries with shape
                `(n_samples,)`.

        Returns:
            A frozen `EnvCompensationModel` carrying the fitted coefficients,
            activation flags, and YAML-serializable metadata.

        Raises:
            ValueError: If the input arrays are mis-shaped, non-finite, or
                length-misaligned.
        """
        baseline_matrix = _coerce_baseline_matrix(baseline_means)
        temperature_samples = _coerce_environment_samples(
            values=t_samples,
            name="t_samples",
            n_samples=baseline_matrix.shape[0],
        )
        humidity_samples = _coerce_environment_samples(
            values=h_samples,
            name="h_samples",
            n_samples=baseline_matrix.shape[0],
        )
        pearson_r = compute_pearson_correlation(temperature_samples, humidity_samples)
        use_univariate_fallback = abs(pearson_r) > PEARSON_MULTICOLLINEARITY_THRESHOLD

        alpha_values: list[float] = []
        beta_t_values: list[float] = []
        beta_h_values: list[float] = []
        adjusted_r2_values: list[float] = []
        apply_flags: list[bool] = []

        for channel_index in range(N_GAS_CHANNELS):
            channel_baseline = baseline_matrix[:, channel_index]
            if use_univariate_fallback:
                fit_result = select_univariate_fallback(
                    baseline_means=channel_baseline,
                    t_samples=temperature_samples,
                    h_samples=humidity_samples,
                )
            else:
                fit_result = fit_environmental_regression(
                    baseline_means=channel_baseline,
                    t_samples=temperature_samples,
                    h_samples=humidity_samples,
                )

            coefficients_are_finite = np.isfinite(
                [fit_result["alpha"], fit_result["beta_T"], fit_result["beta_H"], fit_result["adj_r2"]]
            ).all()
            applies_compensation = bool(coefficients_are_finite and fit_result["adj_r2"] > R2_THRESHOLD)
            if applies_compensation:
                alpha_value = float(fit_result["alpha"])
                beta_t_value = float(fit_result["beta_T"])
                beta_h_value = float(fit_result["beta_H"])
            else:
                alpha_value = float(np.mean(channel_baseline))
                beta_t_value = 0.0
                beta_h_value = 0.0

            alpha_values.append(alpha_value)
            beta_t_values.append(beta_t_value)
            beta_h_values.append(beta_h_value)
            adjusted_r2_values.append(float(fit_result["adj_r2"]))
            apply_flags.append(applies_compensation)

        t_ref = float(np.median(temperature_samples))
        h_ref = float(np.median(humidity_samples))
        LOGGER.info(
            "Fitted environmental compensation n_samples=%s multicollinearity_fallback_used=%s t_ref=%.6f h_ref=%.6f",
            baseline_matrix.shape[0],
            use_univariate_fallback,
            t_ref,
            h_ref,
        )
        return cls(
            alpha=tuple(alpha_values),
            beta_t=tuple(beta_t_values),
            beta_h=tuple(beta_h_values),
            adjusted_r2=tuple(adjusted_r2_values),
            applies_compensation=tuple(apply_flags),
            t_ref=t_ref,
            h_ref=h_ref,
            multicollinearity_fallback_used=use_univariate_fallback,
            temperature_humidity_pearson_r=float(pearson_r),
            temperature_humidity_abs_pearson_r_threshold=PEARSON_MULTICOLLINEARITY_THRESHOLD,
            r2_threshold=R2_THRESHOLD,
        )

    def apply(
        self,
        smoothed_gas: NDArray[np.number[Any]],
        t_sample: float,
        h_sample: float,
    ) -> NDArray[np.float64]:
        """Apply the prespecified compensation formula to one smoothed sample.

        Args:
            smoothed_gas: Smoothed gas-channel matrix with shape `(n_rows, 11)`.
            t_sample: Baseline temperature summary for the sample being
                compensated.
            h_sample: Baseline humidity summary for the sample being
                compensated.

        Returns:
            A compensated gas-channel matrix with shape `(n_rows, 11)` and
            `numpy.float64` dtype.

        Raises:
            ValueError: If the gas matrix is mis-shaped or either scalar
                environmental summary is non-finite.
        """
        smoothed_matrix = np.asarray(smoothed_gas, dtype=np.float64)
        if smoothed_matrix.ndim != 2:
            raise ValueError(f"smoothed_gas must be two-dimensional; got shape {smoothed_matrix.shape}")
        if smoothed_matrix.shape[1] != N_GAS_CHANNELS:
            raise ValueError(
                f"smoothed_gas must have shape (n_rows, {N_GAS_CHANNELS}); got {smoothed_matrix.shape}"
            )
        temperature_sample = _coerce_finite_scalar(t_sample, "t_sample")
        humidity_sample = _coerce_finite_scalar(h_sample, "h_sample")

        correction = (
            np.asarray(self.beta_t, dtype=np.float64) * (temperature_sample - self.t_ref)
            + np.asarray(self.beta_h, dtype=np.float64) * (humidity_sample - self.h_ref)
        )
        apply_mask = np.asarray(self.applies_compensation, dtype=np.float64)
        compensated = np.array(smoothed_matrix, copy=True, dtype=np.float64)
        compensated -= correction.reshape(1, -1) * apply_mask.reshape(1, -1)
        return compensated

    def to_yaml_dict(self) -> dict[str, Any]:
        """Serialize the model to the `environmental_compensation` YAML schema."""
        per_channel = {
            channel_name: {
                "alpha": self.alpha[channel_index],
                "beta_T": self.beta_t[channel_index],
                "beta_H": self.beta_h[channel_index],
                "adj_r2": self.adjusted_r2[channel_index],
                "applies_compensation": self.applies_compensation[channel_index],
            }
            for channel_index, channel_name in enumerate(GAS_CHANNELS)
        }
        return {
            "temperature_humidity_pearson_r": self.temperature_humidity_pearson_r,
            "temperature_humidity_abs_pearson_r_threshold": self.temperature_humidity_abs_pearson_r_threshold,
            "multicollinearity_fallback_used": self.multicollinearity_fallback_used,
            "r2_threshold": self.r2_threshold,
            "t_ref": self.t_ref,
            "h_ref": self.h_ref,
            "per_channel": per_channel,
        }

    @classmethod
    def from_yaml_dict(cls, yaml_dict: Mapping[str, Any]) -> EnvCompensationModel:
        """Deserialize a model from the `environmental_compensation` YAML schema.

        Args:
            yaml_dict: Mapping loaded from the `environmental_compensation` YAML
                section.

        Returns:
            A validated `EnvCompensationModel`.

        Raises:
            TypeError: If `yaml_dict` is not a mapping.
            ValueError: If required keys or per-channel payloads are missing.
        """
        if not isinstance(yaml_dict, Mapping):
            raise TypeError(f"yaml_dict must be a mapping; got {type(yaml_dict).__name__}")
        per_channel = yaml_dict.get("per_channel")
        if not isinstance(per_channel, Mapping):
            raise ValueError("yaml_dict must contain a per_channel mapping.")

        alpha_values: list[float] = []
        beta_t_values: list[float] = []
        beta_h_values: list[float] = []
        adjusted_r2_values: list[float] = []
        apply_flags: list[bool] = []
        for channel_name in GAS_CHANNELS:
            channel_payload = per_channel.get(channel_name)
            if not isinstance(channel_payload, Mapping):
                raise ValueError(f"per_channel must contain a mapping for {channel_name}.")
            alpha_values.append(float(channel_payload["alpha"]))
            beta_t_values.append(float(channel_payload["beta_T"]))
            beta_h_values.append(float(channel_payload["beta_H"]))
            adjusted_r2_values.append(float(channel_payload["adj_r2"]))
            apply_flags.append(bool(channel_payload["applies_compensation"]))

        return cls(
            alpha=tuple(alpha_values),
            beta_t=tuple(beta_t_values),
            beta_h=tuple(beta_h_values),
            adjusted_r2=tuple(adjusted_r2_values),
            applies_compensation=tuple(apply_flags),
            t_ref=float(yaml_dict["t_ref"]),
            h_ref=float(yaml_dict["h_ref"]),
            multicollinearity_fallback_used=bool(yaml_dict["multicollinearity_fallback_used"]),
            temperature_humidity_pearson_r=float(yaml_dict["temperature_humidity_pearson_r"]),
            temperature_humidity_abs_pearson_r_threshold=float(
                yaml_dict["temperature_humidity_abs_pearson_r_threshold"]
            ),
            r2_threshold=float(yaml_dict["r2_threshold"]),
        )


__all__ = [
    "EnvCompensationModel",
    "PEARSON_MULTICOLLINEARITY_THRESHOLD",
    "R2_THRESHOLD",
]
