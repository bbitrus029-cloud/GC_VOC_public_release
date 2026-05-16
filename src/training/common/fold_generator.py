from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.training.common.early_stopping import create_dl_holdout_indices


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FoldIndices:
    """Container for one outer-fold split and its nested inner folds.

    Operational-plan reference: §5.3.
    """

    outer_fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    inner_folds: tuple[tuple[np.ndarray, np.ndarray], ...]


def _append_merge_event(message: str) -> None:
    """Record a merge-ladder event through logging only.

    Session 2 keeps this helper side-effect free with respect to tracked output
    files. Session 5 orchestration is the intended consumer that may choose to
    persist these events to `outputs/logs/fold_generator_merge_events.log` while
    generating the locked split artifacts.
    """
    LOGGER.info(message)


def _validate_split_inputs(labels: np.ndarray, disease_categories: np.ndarray) -> None:
    """Validate split-generator input arrays."""
    if not isinstance(labels, np.ndarray) or labels.ndim != 1:
        raise TypeError("labels must be a one-dimensional numpy array.")
    if not isinstance(disease_categories, np.ndarray) or disease_categories.ndim != 1:
        raise TypeError("disease_categories must be a one-dimensional numpy array.")
    if len(labels) != len(disease_categories):
        raise ValueError("labels and disease_categories must have the same length.")
    if len(labels) == 0:
        raise ValueError("labels and disease_categories must be non-empty.")


def _build_joint_strata(labels: np.ndarray, disease_categories: np.ndarray) -> np.ndarray:
    """Build the joint label-disease stratification vector."""
    return np.array(
        [f"{int(label)}_{int(category)}" for label, category in zip(labels, disease_categories, strict=True)],
        dtype=object,
    )


def _strata_are_valid(strata: np.ndarray, minimum_count: int, max_n_strata: int | None) -> bool:
    """Check whether a stratification vector is usable for downstream splitting."""
    _, counts = np.unique(strata, return_counts=True)
    if np.min(counts) < minimum_count:
        return False
    if max_n_strata is not None and len(counts) > max_n_strata:
        return False
    return True


def _stratify_with_merge_ladder(
    labels: np.ndarray,
    disease_categories: np.ndarray,
    minimum_count: int,
    max_n_strata: int | None = None,
) -> np.ndarray:
    """Return the actual stratification vector after the prespecified merge ladder.

    Operational-plan references: §3.7, §5.3, and §8.

    The runtime contingency in §8 only authorizes a merge-ladder fallback for
    LGIN underpopulation during joint label-by-disease stratification. The
    helper therefore first attempts the fully joint vector, then merges
    `disease_category == 2` (LGIN) into `disease_category == 1` (BGL), and
    finally falls back to label-only stratification if the joint vector remains
    invalid for the requested split geometry.

    Args:
        labels: Binary labels aligned to the cohort order.
        disease_categories: Disease-category codes aligned to the cohort order.
        minimum_count: Minimum required count in every stratum.
        max_n_strata: Optional upper bound on the number of strata.

    Returns:
        The stratification vector actually used for sklearn splitting.
    """
    _validate_split_inputs(labels=labels, disease_categories=disease_categories)
    if minimum_count < 1:
        raise ValueError("minimum_count must be at least 1.")

    joint_strata = _build_joint_strata(labels=labels, disease_categories=disease_categories)
    if _strata_are_valid(joint_strata, minimum_count=minimum_count, max_n_strata=max_n_strata):
        return joint_strata

    merged_categories = disease_categories.copy()
    if np.any(merged_categories == 2):
        merged_categories = np.where(merged_categories == 2, 1, merged_categories)
        _append_merge_event(
            "Applied merge ladder: merged LGIN disease_category=2 into BGL disease_category=1 for stratification."
        )
        merged_joint_strata = _build_joint_strata(labels=labels, disease_categories=merged_categories)
        if _strata_are_valid(merged_joint_strata, minimum_count=minimum_count, max_n_strata=max_n_strata):
            return merged_joint_strata

    _append_merge_event("Merge ladder exhausted; falling back to label-only stratification.")
    label_only_strata = labels.astype(str)
    if not _strata_are_valid(label_only_strata, minimum_count=minimum_count, max_n_strata=max_n_strata):
        raise ValueError("Unable to construct a valid stratification vector after exhausting the merge ladder.")
    return label_only_strata


class FoldGenerator:
    """Generate in-memory train/test, nested-CV, and DL holdout indices.

    Operational-plan references: §3.7 and §5.3.
    """

    def __init__(
        self,
        train_fraction: float = 0.80,
        outer_n_folds: int = 5,
        inner_n_folds: int = 3,
        dl_holdout_fraction: float = 0.125,
    ) -> None:
        """Initialize the fold generator with prespecified split geometry.

        Args:
            train_fraction: Fraction assigned to the main training split.
            outer_n_folds: Number of outer CV folds.
            inner_n_folds: Number of inner CV folds.
            dl_holdout_fraction: Fraction reserved for DL early stopping.

        Raises:
            ValueError: If any geometry parameter is outside its valid range.
        """
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1.")
        if outer_n_folds < 2 or inner_n_folds < 2:
            raise ValueError("outer_n_folds and inner_n_folds must both be at least 2.")
        if not 0.0 < dl_holdout_fraction < 1.0:
            raise ValueError("dl_holdout_fraction must be between 0 and 1.")

        self.train_fraction = train_fraction
        self.outer_n_folds = outer_n_folds
        self.inner_n_folds = inner_n_folds
        self.dl_holdout_fraction = dl_holdout_fraction
        self.train_idx_: np.ndarray | None = None
        self.test_idx_: np.ndarray | None = None
        self.outer_folds_: tuple[FoldIndices, ...] = ()
        self.dl_holdouts_: list[tuple[np.ndarray, np.ndarray]] = []

    def generate_train_test_split(
        self,
        labels: np.ndarray,
        disease_categories: np.ndarray,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate the stratified train/test split in memory.

        Operational-plan references: §3.7 and §5.3.

        Args:
            labels: Binary labels aligned to cohort order.
            disease_categories: Disease-category codes aligned to cohort order.
            seed: Deterministic split seed.

        Returns:
            Sorted `(train_idx, test_idx)` arrays in cohort-index space.
        """
        _validate_split_inputs(labels=labels, disease_categories=disease_categories)

        n_test = int(np.ceil(len(labels) * (1.0 - self.train_fraction)))
        stratify_vector = _stratify_with_merge_ladder(
            labels=labels,
            disease_categories=disease_categories,
            minimum_count=2,
            max_n_strata=n_test,
        )
        cohort_idx = np.arange(len(labels), dtype=int)
        train_idx, test_idx = train_test_split(
            cohort_idx,
            train_size=self.train_fraction,
            random_state=seed,
            shuffle=True,
            stratify=stratify_vector,
        )
        self.train_idx_ = np.sort(train_idx.astype(int))
        self.test_idx_ = np.sort(test_idx.astype(int))
        LOGGER.info(
            "Generated train/test split with seed=%s, train_n=%s, test_n=%s.",
            seed,
            len(self.train_idx_),
            len(self.test_idx_),
        )
        return self.train_idx_, self.test_idx_

    def generate_outer_folds(
        self,
        train_labels: np.ndarray,
        train_disease_categories: np.ndarray,
        seed: int,
    ) -> tuple[FoldIndices, ...]:
        """Generate outer and inner fold indices in memory.

        Operational-plan references: §3.7 and §5.3.

        Args:
            train_labels: Binary labels for the main training split.
            train_disease_categories: Disease-category codes for the training split.
            seed: Deterministic CV seed.

        Returns:
            A tuple of `FoldIndices` records, one per outer fold.
        """
        _validate_split_inputs(labels=train_labels, disease_categories=train_disease_categories)

        stratify_vector = _stratify_with_merge_ladder(
            labels=train_labels,
            disease_categories=train_disease_categories,
            minimum_count=self.outer_n_folds,
        )
        skf_outer = StratifiedKFold(
            n_splits=self.outer_n_folds,
            shuffle=True,
            random_state=seed,
        )

        fold_records: list[FoldIndices] = []
        training_idx = np.arange(len(train_labels), dtype=int)
        for outer_fold, (outer_train_idx, outer_val_idx) in enumerate(
            skf_outer.split(training_idx, stratify_vector),
        ):
            inner_labels = train_labels[outer_train_idx]
            inner_disease_categories = train_disease_categories[outer_train_idx]
            inner_strata = _stratify_with_merge_ladder(
                labels=inner_labels,
                disease_categories=inner_disease_categories,
                minimum_count=self.inner_n_folds,
            )
            skf_inner = StratifiedKFold(
                n_splits=self.inner_n_folds,
                shuffle=True,
                random_state=seed,
            )
            inner_folds = tuple(
                (
                    outer_train_idx[inner_train_idx].astype(int),
                    outer_train_idx[inner_val_idx].astype(int),
                )
                for inner_train_idx, inner_val_idx in skf_inner.split(outer_train_idx, inner_strata)
            )
            fold_records.append(
                FoldIndices(
                    outer_fold=outer_fold,
                    train_idx=np.sort(outer_train_idx.astype(int)),
                    val_idx=np.sort(outer_val_idx.astype(int)),
                    inner_folds=inner_folds,
                )
            )

        self.outer_folds_ = tuple(fold_records)
        LOGGER.info(
            "Generated %s outer folds and %s inner folds per outer fold with seed=%s.",
            self.outer_n_folds,
            self.inner_n_folds,
            seed,
        )
        return self.outer_folds_

    def generate_dl_holdout(
        self,
        train_idx: np.ndarray,
        disease_categories: np.ndarray,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a stratified DL early-stopping holdout for one training subset.

        Operational-plan references: §2.23 and §5.3.

        Args:
            train_idx: Absolute indices for the current outer-training subset.
            disease_categories: Full-cohort disease-category array.
            seed: Deterministic split seed.

        Returns:
            A tuple of `(kept_train_idx, holdout_idx)` arrays.
        """
        kept_train_idx, holdout_idx = create_dl_holdout_indices(
            train_idx=train_idx,
            disease_categories=disease_categories,
            holdout_fraction=self.dl_holdout_fraction,
            seed=seed,
        )
        self.dl_holdouts_.append((kept_train_idx, holdout_idx))
        return kept_train_idx, holdout_idx

    def save_splits(self, output_dir: Path) -> None:
        """Persist generated indices to compressed `.npz` files.

        Operational-plan references: §3.7 and §5.3.

        Args:
            output_dir: Directory that receives the split artifacts.

        Raises:
            TypeError: If `output_dir` is not a pathlib path.
            ValueError: If required split state has not been generated yet.
            OSError: If writing any artifact fails.
        """
        if not isinstance(output_dir, Path):
            raise TypeError("output_dir must be a pathlib.Path instance.")
        if self.train_idx_ is None or self.test_idx_ is None:
            raise ValueError("Call generate_train_test_split() before save_splits().")
        if not self.outer_folds_:
            raise ValueError("Call generate_outer_folds() before save_splits().")

        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            np.savez_compressed(
                output_dir / "train_test_split.npz",
                train_idx=self.train_idx_,
                test_idx=self.test_idx_,
            )

            for fold in self.outer_folds_:
                inner_train_indices = np.array([pair[0] for pair in fold.inner_folds], dtype=object)
                inner_val_indices = np.array([pair[1] for pair in fold.inner_folds], dtype=object)
                np.savez_compressed(
                    output_dir / f"outer_fold_{fold.outer_fold}.npz",
                    outer_fold=np.array([fold.outer_fold], dtype=int),
                    train_idx=fold.train_idx,
                    val_idx=fold.val_idx,
                    inner_train_indices=inner_train_indices,
                    inner_val_indices=inner_val_indices,
                )

            # pragma: SESSION-5-CONSUMER
            for outer_fold, (train_idx, holdout_idx) in enumerate(self.dl_holdouts_):
                np.savez_compressed(
                    output_dir / f"dl_holdout_{outer_fold}.npz",
                    train_idx=train_idx,
                    holdout_idx=holdout_idx,
                )
        except OSError as exc:
            LOGGER.error("Failed to save split artifacts under %s.", output_dir)
            raise OSError(f"Failed to save split artifacts under {output_dir}.") from exc
