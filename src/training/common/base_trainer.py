from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
import json
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_ROOT, RANDOM_SEED
from src.training.common.fold_generator import FoldIndices


LOGGER = logging.getLogger(__name__)


class BaseNestedCVTrainer(ABC):
    """Shared nested-CV template for all prespecified model trainers.

    Operational-plan references: §2.7, §3.9, §3.10, and §5.7.
    """

    model_name: str = ""
    pipeline: Literal["A", "B"]
    class_weight_mode: Literal["scale_pos_weight", "balanced", "dl_two_class"]
    inner_n_folds: int = 3
    outer_n_folds: int = 5
    hyperparameter_candidates: Sequence[dict[str, Any]] = ()

    def __init__(
        self,
        global_config_path: Path | None = None,
        model_config_path: Path | None = None,
    ) -> None:
        """Initialize trainer state and optional config paths."""
        self.global_config_path = global_config_path
        self.model_config_path = model_config_path
        self.sample_index_: pd.DataFrame | None = None
        self.sample_ids_: np.ndarray | None = None
        self.x_: np.ndarray | pd.DataFrame | None = None
        self.y_: np.ndarray | None = None
        self.x_train_: np.ndarray | pd.DataFrame | None = None
        self.y_train_: np.ndarray | None = None
        self.feature_names_: list[str] | None = None
        self.train_idx_: np.ndarray | None = None
        self.test_idx_: np.ndarray | None = None
        self.outer_folds_: tuple[FoldIndices, ...] = ()
        self.current_class_weights_: Any = None
        self.nested_cv_oof_: pd.DataFrame | None = None
        self.nested_cv_per_fold_: pd.DataFrame | None = None
        self.nested_cv_summary_: pd.DataFrame | None = None
        self.nested_cv_results_: dict[str, Any] | None = None

    def load_data(self) -> None:
        """Load Stage 1 artifacts and enforce the alignment contract.

        Operational-plan references: §1.3 and §5.7.

        Raises:
            ValueError: If `pipeline` is invalid or schema expectations fail.
            AssertionError: If ordered sample IDs or labels do not align.
            OSError: If reading required artifacts fails.
        """
        sample_index_path = PROCESSED_DATA_ROOT / "sample_index.csv"
        try:
            sample_index = pd.read_csv(sample_index_path)
        except OSError as exc:
            LOGGER.error("Failed to read sample index from %s.", sample_index_path)
            raise OSError(f"Failed to read sample index from {sample_index_path}.") from exc

        LOGGER.info(
            "Loaded sample index from %s with shape=%s, dtypes=%s, missing=%s.",
            sample_index_path,
            sample_index.shape,
            sample_index.dtypes.astype(str).to_dict(),
            sample_index.isna().sum().to_dict(),
        )
        sample_index = sample_index.loc[sample_index["in_final_output"] == True].copy()  # noqa: E712
        sample_index.sort_values("sample_id", inplace=True)
        sample_index.reset_index(drop=True, inplace=True)

        sample_ids = sample_index["sample_id"].astype(str).to_numpy()
        labels = sample_index["label"].astype(int).to_numpy()

        if self.pipeline == "A":
            sequence_path = PROCESSED_DATA_ROOT / "sequences" / "pipeline_a_raw.npz"
            try:
                sequence_data = np.load(sequence_path, allow_pickle=False)
            except OSError as exc:
                LOGGER.error("Failed to read Pipeline A tensor from %s.", sequence_path)
                raise OSError(f"Failed to read Pipeline A tensor from {sequence_path}.") from exc

            x = sequence_data["x"]
            pipeline_sample_ids = sequence_data["sample_id"].astype(str)
            pipeline_labels = sequence_data["label"].astype(int)
            channel_names = [str(name) for name in sequence_data["channel_names"].tolist()]
            order = np.argsort(pipeline_sample_ids)
            ordered_sample_ids = pipeline_sample_ids[order]
            ordered_labels = pipeline_labels[order]

            assert np.array_equal(
                sample_ids,
                ordered_sample_ids,
            ), "Mismatch between sample_index.csv and pipeline_a_raw.npz sample_id ordering."
            assert np.array_equal(labels, ordered_labels), "Mismatch between sample_index.csv and pipeline_a_raw.npz labels."

            self.x_ = x[order]
            self.feature_names_ = channel_names
        elif self.pipeline == "B":
            feature_path = PROCESSED_DATA_ROOT / "features" / "pipeline_b_raw.csv"
            try:
                feature_table = pd.read_csv(feature_path)
            except OSError as exc:
                LOGGER.error("Failed to read Pipeline B features from %s.", feature_path)
                raise OSError(f"Failed to read Pipeline B features from {feature_path}.") from exc

            LOGGER.info(
                "Loaded Pipeline B features from %s with shape=%s, dtypes=%s, missing=%s.",
                feature_path,
                feature_table.shape,
                feature_table.dtypes.astype(str).to_dict(),
                feature_table.isna().sum().to_dict(),
            )
            feature_table.sort_values("sample_id", inplace=True)
            feature_table.reset_index(drop=True, inplace=True)

            ordered_sample_ids = feature_table["sample_id"].astype(str).to_numpy()
            ordered_labels = feature_table["label"].astype(int).to_numpy()
            assert np.array_equal(
                sample_ids,
                ordered_sample_ids,
            ), "Mismatch between sample_index.csv and pipeline_b_raw.csv sample_id ordering."
            assert np.array_equal(labels, ordered_labels), "Mismatch between sample_index.csv and pipeline_b_raw.csv labels."

            self.feature_names_ = [
                column for column in feature_table.columns.tolist() if column not in {"sample_id", "label"}
            ]
            self.x_ = feature_table.loc[:, self.feature_names_].copy()
        else:
            raise ValueError("pipeline must be set to either 'A' or 'B' in the concrete trainer.")

        self.sample_index_ = sample_index
        self.sample_ids_ = sample_ids
        self.y_ = labels
        LOGGER.info(
            "Alignment contract passed for model=%s, pipeline=%s, n_samples=%s.",
            self.model_name,
            self.pipeline,
            len(self.y_),
        )

    @abstractmethod
    def build_estimator(self, hparams: dict[str, Any], class_weights: Any) -> Any:
        """Build the model-family-specific estimator.

        Operational-plan reference: §5.7.
        """

    @abstractmethod
    def fit_fold(
        self,
        outer_fold: int,
        inner_train_idx: np.ndarray,
        inner_val_idx: np.ndarray,
        hparams: dict[str, Any],
    ) -> dict[str, Any]:
        """Fit one train/validation split and return validation artifacts.

        Operational-plan reference: §5.7.

        IMPORTANT: ``inner_train_idx`` and ``inner_val_idx`` are train-set-relative
        indices in ``[0, len(self.y_train_))``. Concrete trainers MUST slice
        ``self.x_train_`` and ``self.y_train_`` (NOT ``self.x_`` / ``self.y_``,
        which are full-cohort arrays of length 972 and would misalign with the
        fold indices).

        The returned dictionary must at least include:
        - ``score``: scalar validation score used for hyperparameter selection
        - ``val_probabilities``: one probability per ``inner_val_idx``
        """

    def run_nested_cv(self) -> None:
        """Run the prespecified outer-5 by inner-3 nested-CV workflow.

        Operational-plan references: §2.7, §3.9, §3.10, and §5.7.
        """
        if self.y_ is None or self.x_ is None or self.sample_ids_ is None:
            self.load_data()
        self._load_split_state()

        assert self.y_ is not None
        assert self.sample_ids_ is not None

        candidate_hparams = self._get_hyperparameter_candidates()
        oof_probabilities = np.full(len(self.y_), np.nan, dtype=float)
        oof_fold_ids = np.full(len(self.y_), -1, dtype=int)
        per_fold_records: list[dict[str, Any]] = []

        for fold in self.outer_folds_:
            best_hparams: dict[str, Any] | None = None
            best_inner_score = float("-inf")
            best_candidate_record: dict[str, Any] | None = None

            for candidate_idx, hparams in enumerate(candidate_hparams):
                inner_scores: list[float] = []
                for inner_fold_idx, (inner_train_idx, inner_val_idx) in enumerate(fold.inner_folds):
                    LOGGER.info(
                        "Running nested CV model=%s outer_fold=%s inner_fold=%s random_seed=%s.",
                        self.model_name,
                        fold.outer_fold,
                        inner_fold_idx,
                        RANDOM_SEED,
                    )
                    self.current_class_weights_ = self._compute_inner_train_class_weights(
                        self.y_train_[inner_train_idx]
                    )
                    fold_result = self.fit_fold(
                        outer_fold=fold.outer_fold,
                        inner_train_idx=inner_train_idx,
                        inner_val_idx=inner_val_idx,
                        hparams=hparams,
                    )
                    if "score" not in fold_result:
                        raise KeyError("fit_fold() must return a 'score' entry.")
                    inner_scores.append(float(fold_result["score"]))

                mean_inner_score = float(np.nanmean(inner_scores))
                if mean_inner_score > best_inner_score:
                    best_inner_score = mean_inner_score
                    best_hparams = dict(hparams)
                    best_candidate_record = {
                        "outer_fold": fold.outer_fold,
                        "candidate_idx": candidate_idx,
                        "candidate_hparams": dict(hparams),
                        "inner_scores": inner_scores,
                        "mean_inner_score": mean_inner_score,
                    }

            if best_hparams is None or best_candidate_record is None:
                raise ValueError("No hyperparameter candidate produced a valid inner-fold score.")

            LOGGER.info(
                "Training selected outer-fold model for model=%s outer_fold=%s random_seed=%s with hparams=%s.",
                self.model_name,
                fold.outer_fold,
                RANDOM_SEED,
                json.dumps(best_hparams, sort_keys=True),
            )
            self.current_class_weights_ = self._compute_inner_train_class_weights(self.y_train_[fold.train_idx])
            outer_result = self.fit_fold(
                outer_fold=fold.outer_fold,
                inner_train_idx=fold.train_idx,
                inner_val_idx=fold.val_idx,
                hparams=best_hparams,
            )
            if "score" not in outer_result or "val_probabilities" not in outer_result:
                raise KeyError("Outer-fold fit_fold() result must contain 'score' and 'val_probabilities'.")

            val_probabilities = np.asarray(outer_result["val_probabilities"], dtype=float)
            if len(val_probabilities) != len(fold.val_idx):
                raise ValueError("fit_fold() returned a validation probability vector with the wrong length.")

            oof_probabilities[fold.val_idx] = val_probabilities
            oof_fold_ids[fold.val_idx] = fold.outer_fold
            per_fold_records.append(
                {
                    **best_candidate_record,
                    "selected_hparams": best_hparams,
                    "outer_score": float(outer_result["score"]),
                    "oof_status": outer_result.get("oof_status", "ok"),
                }
            )

        self.nested_cv_oof_ = pd.DataFrame(
            {
                "sample_id": self.sample_ids_,
                "label": self.y_,
                "outer_fold": oof_fold_ids,
                "p_raw": oof_probabilities,
            }
        )
        self.nested_cv_per_fold_ = pd.DataFrame(per_fold_records)
        outer_scores = self.nested_cv_per_fold_["outer_score"].to_numpy(dtype=float)
        self.nested_cv_summary_ = pd.DataFrame(
            [
                {
                    "model_name": self.model_name,
                    "mean_outer_score": float(np.nanmean(outer_scores)),
                    "std_outer_score": float(np.nanstd(outer_scores, ddof=0)),
                    "n_outer_folds": len(self.outer_folds_),
                }
            ]
        )
        self.nested_cv_results_ = {
            "model_name": self.model_name,
            "n_samples": len(self.y_),
            "outer_n_folds": self.outer_n_folds,
            "inner_n_folds": self.inner_n_folds,
        }

    def _compute_inner_train_class_weights(self, y_train: np.ndarray) -> Any:
        """Compute fold-specific class weighting per the prespecified rule.

        Operational-plan references: §2.7 and §5.7.
        """
        n_pos = int(np.sum(y_train == 1))
        n_neg = int(np.sum(y_train == 0))
        if n_pos == 0 or n_neg == 0:
            raise ValueError("Inner-train subset must contain both classes to compute class weights.")

        n_total = n_pos + n_neg
        if self.class_weight_mode == "scale_pos_weight":
            return float(n_neg / n_pos)
        if self.class_weight_mode == "balanced":
            return "balanced"
        if self.class_weight_mode == "dl_two_class":
            return np.array(
                [
                    n_total / (2.0 * n_neg),
                    n_total / (2.0 * n_pos),
                ],
                dtype=float,
            )
        raise ValueError(f"Unsupported class_weight_mode: {self.class_weight_mode}.")

    def _get_hyperparameter_candidates(self) -> list[dict[str, Any]]:
        """Return the hyperparameter candidates defined by the concrete trainer."""
        if not self.hyperparameter_candidates:
            raise ValueError("Concrete trainer must define non-empty hyperparameter_candidates.")
        return [dict(candidate) for candidate in self.hyperparameter_candidates]

    def _load_split_state(self) -> None:
        """Load the locked split artifacts generated in Session 5."""
        split_dir = PROCESSED_DATA_ROOT / "splits"
        train_test_path = split_dir / "train_test_split.npz"
        if not train_test_path.exists():
            raise FileNotFoundError(f"Expected locked train/test split artifact at {train_test_path}.")

        try:
            train_test = np.load(train_test_path, allow_pickle=False)
        except OSError as exc:
            LOGGER.error("Failed to read split artifact from %s.", train_test_path)
            raise OSError(f"Failed to read split artifact from {train_test_path}.") from exc

        self.train_idx_ = train_test["train_idx"].astype(int)
        self.test_idx_ = train_test["test_idx"].astype(int)

        fold_records: list[FoldIndices] = []
        for outer_fold in range(self.outer_n_folds):
            fold_path = split_dir / f"outer_fold_{outer_fold}.npz"
            if not fold_path.exists():
                raise FileNotFoundError(f"Expected outer-fold split artifact at {fold_path}.")
            try:
                fold_data = np.load(fold_path, allow_pickle=True)
            except OSError as exc:
                LOGGER.error("Failed to read outer-fold artifact from %s.", fold_path)
                raise OSError(f"Failed to read outer-fold artifact from {fold_path}.") from exc

            inner_train_indices = tuple(
                np.asarray(indices, dtype=int) for indices in fold_data["inner_train_indices"].tolist()
            )
            inner_val_indices = tuple(
                np.asarray(indices, dtype=int) for indices in fold_data["inner_val_indices"].tolist()
            )
            inner_folds = tuple(zip(inner_train_indices, inner_val_indices, strict=True))
            fold_records.append(
                FoldIndices(
                    outer_fold=int(fold_data["outer_fold"][0]),
                    train_idx=fold_data["train_idx"].astype(int),
                    val_idx=fold_data["val_idx"].astype(int),
                    inner_folds=inner_folds,
                )
            )

        self.outer_folds_ = tuple(fold_records)

        # Derive train-set-relative views so that outer/inner indices (which
        # generate_outer_folds emits in [0, len(train_idx)) space) line up with
        # the arrays they slice. Without this bridge, self.y_[inner_train_idx]
        # silently grabs full-cohort samples at wrong positions because
        # self.y_ has length 972 while the fold indices are in [0, 778).
        # Operational-plan references: §1.3 (alignment contract) and §5.7.
        if self.y_ is None or self.x_ is None:
            raise ValueError("load_data() must complete before _load_split_state() derives train views.")
        self.y_train_ = self.y_[self.train_idx_]
        if isinstance(self.x_, np.ndarray):
            self.x_train_ = self.x_[self.train_idx_]
        else:
            self.x_train_ = self.x_.iloc[self.train_idx_].reset_index(drop=True)

    @abstractmethod
    def save_outputs(self) -> None:
        """Persist nested-CV outputs for the concrete trainer.

        Operational-plan references: §3.9, §3.10, and §5.7.

        Concrete trainers must write:
        - `nested_cv_oof.csv`
        - `nested_cv_per_fold.csv`
        - `nested_cv_summary.csv`
        - `nested_cv_results.json`
        """
