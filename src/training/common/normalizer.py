from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd

from src.config import GAS_CHANNELS, PIPELINE_B_FEATURE_COUNT


LOGGER = logging.getLogger(__name__)
SIGMA_CLAMP_THRESHOLD = 1e-8


@dataclass(frozen=True)
class PipelineANormalizer:
    """Fold-level z-score normalizer for Pipeline A tensors.

    Operational-plan references: §2.4 and §5.4.
    """

    mean_: np.ndarray
    scale_: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Return a z-scored copy of the input tensor.

        Args:
            x: Tensor of shape `[N, L, 11]`.

        Returns:
            A transformed tensor with the same shape as `x`.

        Raises:
            TypeError: If `x` is not a numpy array.
            ValueError: If the tensor shape is invalid.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy array.")
        if x.ndim != 3 or x.shape[-1] != len(self.mean_):
            raise ValueError("Pipeline A inputs must have shape [N, L, n_channels].")
        return (x.astype(float, copy=True) - self.mean_.reshape(1, 1, -1)) / self.scale_.reshape(1, 1, -1)


@dataclass(frozen=True)
class PipelineBNormalizer:
    """Fold-level z-score normalizer for Pipeline B feature tables.

    Operational-plan references: §2.4 and §5.4.
    """

    mean_: np.ndarray
    scale_: np.ndarray
    feature_names: list[str]

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Return a z-scored copy aligned to the fitted feature order.

        Args:
            x: Feature table containing all fitted feature names.

        Returns:
            A transformed dataframe whose columns follow `feature_names`.

        Raises:
            TypeError: If `x` is not a pandas dataframe.
            ValueError: If required features are missing.
        """
        if not isinstance(x, pd.DataFrame):
            raise TypeError("x must be a pandas.DataFrame.")
        missing_features = [feature for feature in self.feature_names if feature not in x.columns]
        if missing_features:
            raise ValueError(f"Missing features for Pipeline B normalization: {missing_features}.")

        aligned = x.loc[:, self.feature_names].copy()
        transformed_values = (aligned.to_numpy(dtype=float) - self.mean_) / self.scale_
        return pd.DataFrame(transformed_values, columns=self.feature_names, index=aligned.index)


def _clamp_small_scales(scale: np.ndarray, names: list[str], kind: str) -> np.ndarray:
    """Clamp numerically unstable standard deviations to `1.0`."""
    clamped_scale = scale.copy()
    for index, value in enumerate(clamped_scale):
        if value < SIGMA_CLAMP_THRESHOLD:
            clamped_scale[index] = 1.0
            LOGGER.warning(
                "Clamped %s scale below %.1e for %s '%s' at index %s.",
                kind,
                SIGMA_CLAMP_THRESHOLD,
                kind,
                names[index],
                index,
            )
    return clamped_scale


def fit_pipeline_a_normalizer(x_train: np.ndarray) -> PipelineANormalizer:
    """Fit the Pipeline A fold-level z-score normalizer on train-only data.

    Operational-plan references: §2.4 and §5.4.

    Args:
        x_train: Train-only tensor of shape `[N, L, 11]`.

    Returns:
        A fitted `PipelineANormalizer`.
    """
    if not isinstance(x_train, np.ndarray):
        raise TypeError("x_train must be a numpy array.")
    if x_train.ndim != 3 or x_train.shape[-1] != len(GAS_CHANNELS):
        raise ValueError(f"x_train must have shape [N, L, {len(GAS_CHANNELS)}].")

    mean_ = x_train.mean(axis=(0, 1), dtype=float)
    scale_ = x_train.std(axis=(0, 1), dtype=float)
    clamped_scale = _clamp_small_scales(scale=scale_, names=list(GAS_CHANNELS), kind="channel")
    return PipelineANormalizer(mean_=mean_, scale_=clamped_scale)


def fit_pipeline_b_normalizer(x_train: pd.DataFrame, feature_names: list[str]) -> PipelineBNormalizer:
    """Fit the Pipeline B fold-level z-score normalizer on train-only data.

    Operational-plan references: §2.4 and §5.4.

    Args:
        x_train: Train-only feature table.
        feature_names: Ordered feature names expected by downstream models.

    Returns:
        A fitted `PipelineBNormalizer`.
    """
    if not isinstance(x_train, pd.DataFrame):
        raise TypeError("x_train must be a pandas.DataFrame.")
    if len(feature_names) != PIPELINE_B_FEATURE_COUNT:
        raise ValueError(
            f"feature_names must contain exactly {PIPELINE_B_FEATURE_COUNT} entries for Pipeline B."
        )
    missing_features = [feature for feature in feature_names if feature not in x_train.columns]
    if missing_features:
        raise ValueError(f"Missing features in x_train: {missing_features}.")

    aligned = x_train.loc[:, feature_names]
    mean_ = aligned.mean(axis=0).to_numpy(dtype=float)
    scale_ = aligned.std(axis=0, ddof=0).to_numpy(dtype=float)
    clamped_scale = _clamp_small_scales(scale=scale_, names=feature_names, kind="feature")
    return PipelineBNormalizer(mean_=mean_, scale_=clamped_scale, feature_names=list(feature_names))
