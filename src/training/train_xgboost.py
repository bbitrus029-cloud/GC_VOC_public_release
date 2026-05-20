"""XGBoost nested cross-validation for Session 6.

This trainer follows the locked Stage 4a Session 6 contract for the Pipeline B
XGBoost model:

- `BaseNestedCVTrainer.load_data()` aligns the full 972-row cohort and exposes
  the 99 engineered Pipeline B features plus `sample_id` traceability.
- `BaseNestedCVTrainer._load_split_state()` derives `self.x_train_` and
  `self.y_train_` in the locked 777-row training-set order.
- The Session 6 alignment probe confirmed that nested inner indices are
  absolute against that 777-row training-set order. To preserve strict outer
  fold isolation, this module remaps those absolute indices into outer-train
  local positions before passing them into `RandomizedSearchCV`.
- Per-inner-fold normalization is enforced with an sklearn `Pipeline` whose
  first step is a `PipelineBNormalizer` adapter, ensuring the z-score fit is
  recomputed on each inner-train subset only.
- `scale_pos_weight` is recomputed inside a thin `XGBClassifier` subclass on
  every fit call from the labels actually seen by that fit call, so inner-train
  class imbalance never leaks across folds.
- Session 6 does not fit the outer OOF Platt calibrator yet; summary metrics
  therefore use `p_cal=p_raw`. Threshold-dependent metrics are reported with
  the fixed placeholder threshold `0.5`, while the prespecified Youden
  threshold is deferred to Session 10 Phase 2.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
import yaml
from xgboost import XGBClassifier

from src.config import PROJECT_ROOT, RANDOM_SEED
from src.evaluation.metrics import compute_full_metric_set
from src.training.common.base_trainer import BaseNestedCVTrainer
from src.training.common.bootstrap import bootstrap_metric
from src.training.common.normalizer import PipelineBNormalizer, fit_pipeline_b_normalizer


LOGGER = logging.getLogger(__name__)
SESSION6_THRESHOLD_P_RAW = 0.5
_SCALE_POS_WEIGHT_HISTORY: list[float] = []


@dataclass(frozen=True)
class _SerializableOuterResult:
    """Serializable outer-fold metadata for JSON output."""

    outer_fold: int
    best_hparams: dict[str, Any]
    metrics: dict[str, float]
    fit_time: float
    predict_time: float
    scale_pos_weight_range: dict[str, float]
    best_estimator_serializable_params: dict[str, Any]


class _PipelineBNormalizerAdapter(BaseEstimator, TransformerMixin):
    """sklearn adapter that refits Pipeline B normalization within each split."""

    def __init__(self) -> None:
        """Initialize the adapter with deferred fitted state."""
        self.normalizer_: PipelineBNormalizer | None = None

    def fit(self, x: pd.DataFrame, y: np.ndarray | None = None) -> "_PipelineBNormalizerAdapter":
        """Fit fold-specific normalization from the current training subset.

        Args:
            x: Training feature matrix.
            y: Unused sklearn compatibility argument.

        Returns:
            The fitted adapter instance.
        """
        del y
        if not isinstance(x, pd.DataFrame):
            raise TypeError("PipelineBNormalizerAdapter expects a pandas.DataFrame input.")
        self.normalizer_ = fit_pipeline_b_normalizer(x_train=x, feature_names=x.columns.tolist())
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Transform a feature matrix with the fitted normalizer.

        Args:
            x: Feature matrix containing the fitted 99-feature schema.

        Returns:
            A normalized dataframe with the same feature order.
        """
        if self.normalizer_ is None:
            raise ValueError("PipelineBNormalizerAdapter must be fitted before transform().")
        if not isinstance(x, pd.DataFrame):
            raise TypeError("PipelineBNormalizerAdapter expects a pandas.DataFrame input.")
        return self.normalizer_.transform(x)


class _InnerTrainBalancedXGBClassifier(XGBClassifier):
    """XGBClassifier that recomputes `scale_pos_weight` on every fit call."""

    def fit(self, x: Any, y: Any, **kwargs: Any) -> "_InnerTrainBalancedXGBClassifier":
        """Fit after recomputing `scale_pos_weight` from the current labels.

        Args:
            x: Training features.
            y: Binary labels for the current fit call.
            **kwargs: Additional keyword arguments forwarded to XGBoost.

        Returns:
            The fitted estimator.
        """
        y_array = np.asarray(y, dtype=int)
        n_pos = int(np.sum(y_array == 1))
        n_neg = int(np.sum(y_array == 0))
        if n_pos == 0 or n_neg == 0:
            raise ValueError("scale_pos_weight requires both classes in the current fit subset.")
        scale_pos_weight = float(n_neg / n_pos)
        _SCALE_POS_WEIGHT_HISTORY.append(scale_pos_weight)
        self.set_params(scale_pos_weight=scale_pos_weight)
        LOGGER.info(
            "Computed scale_pos_weight=%.6f from n_neg=%s and n_pos=%s for the current XGBoost fit call.",
            scale_pos_weight,
            n_neg,
            n_pos,
        )
        return super().fit(x, y_array, **kwargs)


def _configure_logging() -> None:
    """Configure INFO-level logging for CLI execution."""
    if logging.getLogger().handlers:
        logging.getLogger().setLevel(logging.INFO)
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML config file into a dictionary."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except OSError as exc:
        raise OSError(f"Failed to read YAML config from {path}.") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping in {path}.")
    return loaded


def _resolve_project_root(global_config_path: Path | None, model_config_path: Path | None) -> Path:
    """Resolve the project root from config paths when available."""
    anchor = model_config_path or global_config_path
    if anchor is None:
        return PROJECT_ROOT
    return anchor.resolve().parents[1]


def _resolve_path(project_root: Path, value: str) -> Path:
    """Resolve a config path relative to the project root when needed."""
    candidate = Path(value)
    return candidate if candidate.is_absolute() else project_root / candidate


def _ensure_parent_dir(path: Path) -> None:
    """Create the parent directory for an output path."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _json_default(value: Any) -> Any:
    """Convert numpy scalars and arrays into JSON-serializable objects."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 form."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _get_git_commit_hash(project_root: Path) -> str:
    """Return the current git commit hash or `unknown` if unavailable."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        LOGGER.warning("Unable to determine git commit hash from the current repository state.")
        return "unknown"
    return result.stdout.strip()


def _make_param_distribution(spec: Any) -> Any:
    """Convert YAML search-space specs into scipy distributions or discrete lists."""
    if isinstance(spec, list):
        return spec
    if not isinstance(spec, dict):
        raise TypeError("Each XGBoost search-space entry must be a list or mapping.")

    dist_type = str(spec["type"])
    if dist_type == "randint":
        return stats.randint(low=int(spec["low"]), high=int(spec["high"]))
    if dist_type == "uniform":
        return stats.uniform(loc=float(spec["loc"]), scale=float(spec["scale"]))
    if dist_type == "loguniform":
        return stats.loguniform(a=float(spec["a"]), b=float(spec["b"]))
    raise ValueError(f"Unsupported distribution type: {dist_type}.")


def _make_bootstrap_metric_fn(metric_name: str) -> Any:
    """Create a metric callback for the shared bootstrap utility."""

    def _metric_fn(y_true: np.ndarray, p_raw: np.ndarray) -> float:
        metrics = compute_full_metric_set(
            y_true=y_true,
            p_raw=p_raw,
            p_cal=p_raw,
            threshold_p_raw=SESSION6_THRESHOLD_P_RAW,
        )
        return float(metrics[metric_name])

    return _metric_fn


def _remap_inner_cv_indices(
    outer_train_idx: np.ndarray,
    inner_folds: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Remap absolute train-set indices into outer-train-local positions."""
    lookup = {int(abs_idx): local_idx for local_idx, abs_idx in enumerate(outer_train_idx.tolist())}
    remapped_folds: list[tuple[np.ndarray, np.ndarray]] = []
    for inner_train_abs, inner_val_abs in inner_folds:
        try:
            remapped_train = np.asarray([lookup[int(index)] for index in inner_train_abs], dtype=int)
            remapped_val = np.asarray([lookup[int(index)] for index in inner_val_abs], dtype=int)
        except KeyError as exc:
            raise ValueError("Encountered an inner-fold index outside the outer-train subset.") from exc
        remapped_folds.append((remapped_train, remapped_val))
    return remapped_folds


def _strip_search_prefix(best_params: dict[str, Any]) -> dict[str, Any]:
    """Strip sklearn pipeline prefixes from best-parameter names."""
    stripped: dict[str, Any] = {}
    for key, value in best_params.items():
        stripped[key.split("__", maxsplit=1)[-1]] = value
    return stripped


def _collect_serializable_params(params: dict[str, Any]) -> dict[str, Any]:
    """Filter a parameter dictionary down to JSON-serializable values."""
    serializable: dict[str, Any] = {}
    for key, value in params.items():
        try:
            json.dumps(value, default=_json_default)
        except TypeError:
            continue
        serializable[key] = value
    return serializable


class XGBoostTrainer(BaseNestedCVTrainer):
    """Nested-CV trainer for the prespecified XGBoost model."""

    model_name = "xgboost"
    pipeline = "B"
    class_weight_mode = "scale_pos_weight"
    hyperparameter_candidates = ({},)

    def __init__(
        self,
        global_config_path: Path | None = None,
        model_config_path: Path | None = None,
    ) -> None:
        """Initialize config state, output paths, and fold geometry.

        Args:
            global_config_path: Optional path to `configs/modeling_global.yaml`.
            model_config_path: Optional path to `configs/modeling_xgboost.yaml`.
        """
        project_root = _resolve_project_root(global_config_path=global_config_path, model_config_path=model_config_path)
        resolved_global = global_config_path or project_root / "configs" / "modeling_global.yaml"
        resolved_model = model_config_path or project_root / "configs" / "modeling_xgboost.yaml"
        super().__init__(global_config_path=resolved_global, model_config_path=resolved_model)

        self.project_root = project_root
        self.global_config = _load_yaml_config(self.global_config_path)
        self.model_config = _load_yaml_config(self.model_config_path)
        self.outer_n_folds = int(self.global_config["cv"]["outer_folds"])
        self.inner_n_folds = int(self.global_config["cv"]["inner_folds"])
        self.bootstrap_n_boot = int(self.global_config["bootstrap"]["n_boot"])
        self.output_paths = {
            key: _resolve_path(project_root=self.project_root, value=value)
            for key, value in self.model_config["paths"].items()
            if key != "log"
        }

    def load_data(self) -> None:
        """Load aligned Stage 1 artifacts for Pipeline B."""
        super().load_data()
        if not isinstance(self.x_, pd.DataFrame):
            raise TypeError("Pipeline B trainer expected a pandas.DataFrame after load_data().")
        LOGGER.info(
            "Loaded Pipeline B feature matrix for model=%s with shape=%s and feature_count=%s.",
            self.model_name,
            self.x_.shape,
            self.x_.shape[1],
        )

    def build_estimator(self, hparams: dict[str, Any], class_weights: Any) -> Pipeline:
        """Build the per-fold sklearn pipeline for XGBoost.

        Args:
            hparams: Tree hyperparameters selected for the current fit.
            class_weights: Unused placeholder from the abstract interface.

        Returns:
            A `Pipeline` with fold-specific normalization and XGBoost.
        """
        del class_weights
        fixed_config = self.model_config["xgboost_fixed"]
        # E009 requires the trainer-side exclusion set to match
        # `NON_HP_KEYS_PER_MODEL["xgboost"]` in `train_final_models.py`.
        excluded_from_splat = {"n_jobs", "random_state", "tree_method"}
        search_hparams = {
            key: value for key, value in hparams.items() if key not in excluded_from_splat
        }
        classifier = _InnerTrainBalancedXGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method=str(fixed_config["tree_method"]),
            n_jobs=int(fixed_config["n_jobs"]),
            random_state=int(hparams["random_state"]),
            verbosity=0,
            **search_hparams,
        )
        return Pipeline(
            steps=[
                ("norm", _PipelineBNormalizerAdapter()),
                ("clf", classifier),
            ]
        )

    def fit_fold(
        self,
        outer_fold: int,
        inner_train_idx: np.ndarray,
        inner_val_idx: np.ndarray,
        hparams: dict[str, Any],
    ) -> dict[str, Any]:
        """Fit one fixed-hyperparameter XGBoost split and return validation artifacts.

        Args:
            outer_fold: Outer-fold identifier used for logging.
            inner_train_idx: Absolute indices into the locked 777-row train order.
            inner_val_idx: Absolute indices into the locked 777-row train order.
            hparams: Selected XGBoost hyperparameters plus the fold random state.

        Returns:
            Validation probabilities and timing diagnostics.
        """
        if not isinstance(self.x_train_, pd.DataFrame) or self.y_train_ is None:
            raise ValueError("Split state must be loaded before fit_fold().")

        x_inner_train = self.x_train_.iloc[inner_train_idx].reset_index(drop=True)
        x_inner_val = self.x_train_.iloc[inner_val_idx].reset_index(drop=True)
        y_inner_train = self.y_train_[inner_train_idx]
        y_inner_val = self.y_train_[inner_val_idx]

        estimator = self.build_estimator(hparams=hparams, class_weights=self.current_class_weights_)
        fit_start = perf_counter()
        estimator.fit(x_inner_train, y_inner_train)
        fit_time = perf_counter() - fit_start

        predict_start = perf_counter()
        val_probabilities = estimator.predict_proba(x_inner_val)[:, 1]
        predict_time = perf_counter() - predict_start

        if len(np.unique(y_inner_val)) < 2:
            score = float("nan")
        else:
            score = float(roc_auc_score(y_inner_val, val_probabilities))

        LOGGER.info(
            "Completed fixed-parameter XGBoost fit for outer_fold=%s with fit_time=%.3fs and predict_time=%.3fs.",
            outer_fold,
            fit_time,
            predict_time,
        )
        return {
            "score": score,
            "val_probabilities": val_probabilities,
            "fit_time": fit_time,
            "predict_time": predict_time,
        }

    def run(self, outer_fold_limit: int | None = None) -> None:
        """Execute the Session 6 nested-CV workflow.

        Args:
            outer_fold_limit: Optional number of outer folds to run, used by smoke tests.
        """
        self.load_data()
        self._load_split_state()
        if not isinstance(self.x_train_, pd.DataFrame) or self.y_ is None or self.y_train_ is None:
            raise ValueError("Data and split state must be available before run().")
        if self.train_idx_ is None or self.sample_ids_ is None:
            raise ValueError("Train indices and sample IDs must be available before run().")

        LOGGER.info("XGBoost anti-stall lock active: tree_method='exact', n_jobs=1")
        LOGGER.info(
            "Session 6 OOF metrics computed with p_cal=p_raw (Layer-B calibrator deferred to Session 10 Phase 2)."
        )
        LOGGER.info(
            "Session 6 nested-CV summaries use threshold_p_raw=%.2f until the final Youden threshold is fit in Session 10 Phase 2.",
            SESSION6_THRESHOLD_P_RAW,
        )
        train_sample_ids = self.sample_ids_[self.train_idx_]
        held_out_count = len(self.y_) - len(self.train_idx_)
        LOGGER.info(
            "Locked split derives train_n=%s and heldout_n=%s from final cohort n=%s for model=%s.",
            len(self.train_idx_),
            held_out_count,
            len(self.y_),
            self.model_name,
        )

        selected_folds = self.outer_folds_[:outer_fold_limit] if outer_fold_limit is not None else self.outer_folds_
        oof_records: list[dict[str, Any]] = []
        per_fold_metric_records: list[dict[str, Any]] = []
        per_fold_results: list[_SerializableOuterResult] = []

        search_space = self.model_config["search_space"]
        search_n_iter = int(self.model_config["search"]["n_iter"])
        search_scoring = str(self.model_config["search"]["scoring"])
        fixed_config = self.model_config["xgboost_fixed"]

        for fold in selected_folds:
            outer_random_state = RANDOM_SEED + int(fold.outer_fold)
            LOGGER.info(
                "Starting XGBoost outer_fold=%s with random_state=%s.",
                fold.outer_fold,
                outer_random_state,
            )
            outer_train_idx = fold.train_idx.astype(int)
            outer_val_idx = fold.val_idx.astype(int)
            x_outer_train = self.x_train_.iloc[outer_train_idx].reset_index(drop=True)
            y_outer_train = self.y_train_[outer_train_idx]
            x_outer_val = self.x_train_.iloc[outer_val_idx].reset_index(drop=True)
            y_outer_val = self.y_train_[outer_val_idx]
            sample_ids_outer_val = train_sample_ids[outer_val_idx]

            inner_cv = _remap_inner_cv_indices(outer_train_idx=outer_train_idx, inner_folds=fold.inner_folds)
            param_distributions = {
                f"clf__{param_name}": _make_param_distribution(spec)
                for param_name, spec in search_space.items()
            }
            base_pipeline = self.build_estimator(
                hparams={"random_state": outer_random_state},
                class_weights=self.current_class_weights_,
            )

            _SCALE_POS_WEIGHT_HISTORY.clear()
            search = RandomizedSearchCV(
                estimator=base_pipeline,
                param_distributions=param_distributions,
                n_iter=search_n_iter,
                scoring=search_scoring,
                cv=inner_cv,
                random_state=outer_random_state,
                n_jobs=1,
                refit=True,
                error_score="raise",
            )
            search.fit(x_outer_train, y_outer_train)

            for mean_fit_time, params in zip(
                search.cv_results_["mean_fit_time"],
                search.cv_results_["params"],
                strict=True,
            ):
                if float(mean_fit_time) > 300.0:
                    LOGGER.warning(
                        "Observed XGBoost candidate mean_fit_time=%.3fs (>300s) for params=%s.",
                        float(mean_fit_time),
                        json.dumps(_strip_search_prefix(params), sort_keys=True, default=_json_default),
                    )

            predict_start = perf_counter()
            val_probabilities = search.best_estimator_.predict_proba(x_outer_val)[:, 1]
            predict_time = perf_counter() - predict_start

            best_hparams = _strip_search_prefix(search.best_params_)
            best_hparams["tree_method"] = str(fixed_config["tree_method"])
            best_hparams["n_jobs"] = int(fixed_config["n_jobs"])
            best_hparams["random_state"] = outer_random_state
            best_hparams_json = json.dumps(best_hparams, sort_keys=True, default=_json_default)

            if len(np.unique(y_outer_val)) < 2:
                outer_score = float("nan")
            else:
                outer_score = float(roc_auc_score(y_outer_val, val_probabilities))

            if not _SCALE_POS_WEIGHT_HISTORY:
                raise ValueError("No scale_pos_weight values were recorded during XGBoost search.")
            scale_range = {
                "min": float(np.min(_SCALE_POS_WEIGHT_HISTORY)),
                "max": float(np.max(_SCALE_POS_WEIGHT_HISTORY)),
            }
            LOGGER.info(
                "outer_fold=%s scale_pos_weight range during inner-search/refit: min=%.6f max=%.6f.",
                fold.outer_fold,
                scale_range["min"],
                scale_range["max"],
            )

            fold_metrics = compute_full_metric_set(
                y_true=y_outer_val,
                p_raw=val_probabilities,
                p_cal=val_probabilities,
                threshold_p_raw=SESSION6_THRESHOLD_P_RAW,
            )
            per_fold_metric_records.append({"scope": "outer_fold", "outer_fold": fold.outer_fold, **fold_metrics})

            serializable_params = _collect_serializable_params(search.best_estimator_.get_params(deep=True))
            per_fold_results.append(
                _SerializableOuterResult(
                    outer_fold=int(fold.outer_fold),
                    best_hparams=best_hparams,
                    metrics=fold_metrics,
                    fit_time=float(search.refit_time_),
                    predict_time=float(predict_time),
                    scale_pos_weight_range=scale_range,
                    best_estimator_serializable_params=serializable_params,
                )
            )

            for sample_id, y_true, p_raw in zip(sample_ids_outer_val, y_outer_val, val_probabilities, strict=True):
                oof_records.append(
                    {
                        "sample_id": int(sample_id),
                        "y_true": int(y_true),
                        "p_raw": float(p_raw),
                        "outer_fold": int(fold.outer_fold),
                        "best_hparams_json": best_hparams_json,
                    }
                )

            LOGGER.info(
                "Completed XGBoost outer_fold=%s with outer_auc_roc=%.6f, refit_time=%.3fs, predict_time=%.3fs.",
                fold.outer_fold,
                outer_score,
                float(search.refit_time_),
                predict_time,
            )

        oof_frame = pd.DataFrame(oof_records)
        if oof_frame.empty:
            raise ValueError("XGBoost nested CV produced no OOF records.")
        if not np.isfinite(oof_frame["p_raw"]).all():
            raise ValueError("XGBoost nested CV produced non-finite OOF probabilities.")

        oof_frame.sort_values("sample_id", kind="mergesort", inplace=True)
        oof_frame.reset_index(drop=True, inplace=True)
        aggregate_metrics = compute_full_metric_set(
            y_true=oof_frame["y_true"].to_numpy(dtype=int),
            p_raw=oof_frame["p_raw"].to_numpy(dtype=float),
            p_cal=oof_frame["p_raw"].to_numpy(dtype=float),
            threshold_p_raw=SESSION6_THRESHOLD_P_RAW,
        )

        summary_records: list[dict[str, Any]] = []
        for fold_record in per_fold_metric_records:
            outer_fold = int(fold_record["outer_fold"])
            for metric_name, metric_value in fold_record.items():
                if metric_name in {"scope", "outer_fold"}:
                    continue
                summary_records.append(
                    {
                        "scope": "outer_fold",
                        "outer_fold": outer_fold,
                        "metric": metric_name,
                        "value": metric_value,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                    }
                )

        y_oof = oof_frame["y_true"].to_numpy(dtype=int)
        p_oof = oof_frame["p_raw"].to_numpy(dtype=float)
        aggregate_with_ci: dict[str, dict[str, float]] = {}
        for metric_name, metric_value in aggregate_metrics.items():
            bootstrap_result = bootstrap_metric(
                y_true=y_oof,
                y_score=p_oof,
                metric_fn=_make_bootstrap_metric_fn(metric_name),
                n_boot=self.bootstrap_n_boot,
                seed=RANDOM_SEED,
            )
            aggregate_with_ci[metric_name] = {
                "estimate": float(metric_value),
                "ci_low": float(bootstrap_result.ci_low),
                "ci_high": float(bootstrap_result.ci_high),
            }
            summary_records.append(
                {
                    "scope": "aggregate",
                    "outer_fold": "aggregate",
                    "metric": metric_name,
                    "value": float(metric_value),
                    "ci_low": float(bootstrap_result.ci_low),
                    "ci_high": float(bootstrap_result.ci_high),
                }
            )

        self.nested_cv_oof_ = oof_frame.loc[:, ["sample_id", "y_true", "p_raw", "outer_fold"]].copy()
        per_fold_frame = oof_frame.loc[:, ["sample_id", "y_true", "p_raw", "outer_fold", "best_hparams_json"]].copy()
        per_fold_frame.sort_values("sample_id", kind="mergesort", inplace=True)
        per_fold_frame.reset_index(drop=True, inplace=True)
        self.nested_cv_per_fold_ = per_fold_frame
        self.nested_cv_summary_ = pd.DataFrame(summary_records)
        self.nested_cv_results_ = {
            "model_name": self.model_name,
            "train_sample_count": int(len(self.train_idx_)),
            "oof_row_count": int(len(self.nested_cv_oof_)),
            "threshold_p_raw_placeholder": SESSION6_THRESHOLD_P_RAW,
            "layer_b_calibrator_deferred": True,
            "per_fold_results": [result.__dict__ for result in per_fold_results],
            "aggregate_metrics": aggregate_with_ci,
            "git_commit_hash": _get_git_commit_hash(self.project_root),
            "training_timestamp_utc": _utc_now_iso(),
        }
        self.save_outputs()

    def save_outputs(self) -> None:
        """Persist the Session 6 XGBoost nested-CV outputs."""
        if (
            self.nested_cv_oof_ is None
            or self.nested_cv_per_fold_ is None
            or self.nested_cv_summary_ is None
            or self.nested_cv_results_ is None
        ):
            raise ValueError("Nested-CV outputs are unavailable; run() must complete before save_outputs().")

        for output_path in self.output_paths.values():
            _ensure_parent_dir(output_path)

        self.nested_cv_oof_.to_csv(self.output_paths["nested_cv_oof"], index=False)
        self.nested_cv_per_fold_.to_csv(self.output_paths["nested_cv_per_fold"], index=False)
        self.nested_cv_summary_.to_csv(self.output_paths["nested_cv_summary"], index=False)
        with self.output_paths["nested_cv_results"].open("w", encoding="utf-8") as handle:
            json.dump(self.nested_cv_results_, handle, indent=2, sort_keys=True, default=_json_default)

        LOGGER.info("Saved XGBoost nested-CV outputs under outputs/{predictions,metrics,analysis}/.")


def run_nested_cv(config_path: Path) -> None:
    """Run XGBoost nested CV from a model config path.

    Args:
        config_path: Path to `configs/modeling_xgboost.yaml`.
    """
    trainer = XGBoostTrainer(model_config_path=config_path)
    trainer.run()


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the XGBoost trainer."""
    parser = argparse.ArgumentParser(description="Run Session 6 XGBoost nested CV.")
    parser.add_argument("--config", required=True, type=Path, help="Path to modeling_xgboost.yaml.")
    parser.add_argument(
        "--global-config",
        type=Path,
        default=None,
        help="Optional path to modeling_global.yaml.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for Session 6 XGBoost nested CV."""
    _configure_logging()
    args = _parse_args()
    trainer = XGBoostTrainer(
        global_config_path=args.global_config,
        model_config_path=args.config,
    )
    trainer.run()


if __name__ == "__main__":
    main()
