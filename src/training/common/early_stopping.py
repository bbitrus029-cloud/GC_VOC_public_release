from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


LOGGER = logging.getLogger(__name__)
DEFAULT_DL_HOLDOUT_FRACTION = 0.125


def create_dl_holdout_indices(
    train_idx: np.ndarray,
    disease_categories: np.ndarray,
    holdout_fraction: float = DEFAULT_DL_HOLDOUT_FRACTION,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a stratified deep-learning early-stopping holdout split.

    Operational-plan references: §2.23, §5.8, and §6.1.

    Args:
        train_idx: Absolute indices for the current outer-training subset.
        disease_categories: Full-cohort disease categories aligned to absolute indices.
        holdout_fraction: Fraction of `train_idx` reserved for early stopping.
        seed: Deterministic split seed.

    Returns:
        A tuple of `(kept_train_idx, holdout_idx)` absolute index arrays.

    Raises:
        TypeError: If the input arrays are not one-dimensional numpy arrays.
        ValueError: If the split fraction or input lengths are invalid.
    """
    if not isinstance(train_idx, np.ndarray) or train_idx.ndim != 1:
        raise TypeError("train_idx must be a one-dimensional numpy array.")
    if not isinstance(disease_categories, np.ndarray) or disease_categories.ndim != 1:
        raise TypeError("disease_categories must be a one-dimensional numpy array.")
    if len(train_idx) == 0:
        raise ValueError("train_idx must contain at least one index.")
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1.")
    if np.any(train_idx < 0) or np.any(train_idx >= len(disease_categories)):
        raise ValueError("train_idx contains out-of-bounds positions for disease_categories.")

    train_strata = disease_categories[train_idx]
    unique_counts = np.unique(train_strata, return_counts=True)[1]
    holdout_size = int(np.ceil(len(train_idx) * holdout_fraction))
    if np.min(unique_counts) < 2:
        raise ValueError("Each disease-category stratum needs at least two samples for a DL holdout split.")
    if holdout_size < len(np.unique(train_strata)):
        raise ValueError("Holdout size is too small to preserve disease-category stratification.")

    kept_train_idx, holdout_idx = train_test_split(
        train_idx,
        test_size=holdout_fraction,
        random_state=seed,
        shuffle=True,
        stratify=train_strata,
    )
    LOGGER.info(
        "Created DL holdout split with seed=%s, holdout_fraction=%.3f, train_n=%s, holdout_n=%s.",
        seed,
        holdout_fraction,
        len(kept_train_idx),
        len(holdout_idx),
    )
    return np.sort(kept_train_idx.astype(int)), np.sort(holdout_idx.astype(int))


def save_dl_holdout(path: Path, train_idx: np.ndarray, holdout_idx: np.ndarray) -> None:
    """Persist deep-learning holdout indices to a compressed `.npz` file.

    Operational-plan references: §2.23 and §5.8.

    Args:
        path: Output file path for the compressed split artifact.
        train_idx: Absolute indices retained for training after holdout extraction.
        holdout_idx: Absolute indices reserved for early stopping.

    Raises:
        TypeError: If `path` is not a pathlib path object.
        ValueError: If the index arrays are not one-dimensional.
        OSError: If writing the file fails.
    """
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path instance.")
    if train_idx.ndim != 1 or holdout_idx.ndim != 1:
        raise ValueError("train_idx and holdout_idx must be one-dimensional arrays.")

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        np.savez_compressed(path, train_idx=train_idx.astype(int), holdout_idx=holdout_idx.astype(int))
    except OSError as exc:
        LOGGER.error("Failed to save DL holdout indices to %s.", path)
        raise OSError(f"Failed to save DL holdout indices to {path}.") from exc
