"""MiniRocket + Ridge nested cross-validation for Session 6.

This trainer follows the locked Stage 4a Session 6 contract for the
MiniRocket+Ridge Pipeline A model:

- `BaseNestedCVTrainer.load_data()` aligns the full 972-row cohort across
  `sample_index.csv`, `pipeline_a_raw.npz`, and `pipeline_b_raw.csv`.
- `BaseNestedCVTrainer._load_split_state()` then derives `self.x_train_` and
  `self.y_train_` in the locked 777-row training-set order.
- The alignment probe for Session 6 confirmed that both `outer_fold_{i}.npz`
  outer indices and nested inner indices are absolute against that 777-row
  training-set order, so this module slices `self.x_train_` / `self.y_train_`
  directly with those indices.
- Raw Pipeline A tensors are stored as `[N, 960, 11]`; MiniRocket receives the
  required channels-first layout `[N, 11, 960]`.
- MiniRocket outputs a feature matrix rather than a raw signal tensor, so
  Session 6 applies Pipeline B-style z-score statistics on the post-transform
  feature matrix, not `PipelineANormalizer`.
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
import os
from pathlib import Path
import subprocess
import tempfile
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score
from sktime.transformations.panel.rocket import MiniRocketMultivariate
import yaml

from src.config import PROJECT_ROOT, RANDOM_SEED
from src.evaluation.metrics import compute_full_metric_set
from src.training.common.base_trainer import BaseNestedCVTrainer
from src.training.common.bootstrap import bootstrap_metric
from src.training.common.normalizer import SIGMA_CLAMP_THRESHOLD


LOGGER = logging.getLogger(__name__)
SESSION6_THRESHOLD_P_RAW = 0.5


@dataclass(frozen=True)
class _FeatureMatrixNormalizer:
    """Pipeline B-style z-score normalizer for post-MiniRocket features."""

    mean_: np.ndarray
    scale_: np.ndarray
    feature_names: list[str]

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted z-score transform to a feature matrix.

        Args:
            x: Feature matrix with the fitted feature columns.

        Returns:
            A transformed dataframe with the same column order.
        """
        aligned = x.loc[:, self.feature_names].copy()
        transformed = (aligned.to_numpy(dtype=float) - self.mean_) / self.scale_
        return pd.DataFrame(transformed, columns=self.feature_names, index=aligned.index)


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


def _ensure_numba_cache_dir() -> None:
    """Set a writable numba cache directory for MiniRocket runtime compilation."""
    if os.environ.get("NUMBA_CACHE_DIR"):
        return
    cache_dir = Path(tempfile.gettempdir()) / "gcvoc_numba_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)
    LOGGER.info("Configured a temporary numba cache directory for MiniRocket runtime.")


def _coerce_feature_frame(values: Any, columns: list[str] | None = None) -> pd.DataFrame:
    """Convert MiniRocket output into a pandas feature frame."""
    if isinstance(values, pd.DataFrame):
        frame = values.copy()
        if columns is not None:
            frame.columns = columns
        return frame
    array = np.asarray(values, dtype=float)
    feature_columns = columns or [f"kernel_{index:05d}" for index in range(array.shape[1])]
    return pd.DataFrame(array, columns=feature_columns)


def _clamp_feature_scales(scale: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """Clamp near-zero feature scales to the prespecified fallback value."""
    clamped = scale.copy()
    for index, value in enumerate(clamped):
        if value < SIGMA_CLAMP_THRESHOLD:
            clamped[index] = 1.0
            LOGGER.warning(
                "Clamped feature scale below %.1e for post-MiniRocket feature '%s' at index %s.",
                SIGMA_CLAMP_THRESHOLD,
                feature_names[index],
                index,
            )
    return clamped


def _fit_feature_matrix_normalizer(x_train: pd.DataFrame) -> _FeatureMatrixNormalizer:
    """Fit Pipeline B-style z-score statistics on a feature matrix."""
    feature_names = x_train.columns.tolist()
    mean_ = x_train.mean(axis=0).to_numpy(dtype=float)
    scale_ = x_train.std(axis=0, ddof=0).to_numpy(dtype=float)
    return _FeatureMatrixNormalizer(
        mean_=mean_,
        scale_=_clamp_feature_scales(scale=scale_, feature_names=feature_names),
        feature_names=feature_names,
    )


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


class MiniRocketRidgeTrainer(BaseNestedCVTrainer):
    """Nested-CV trainer for the prespecified MiniRocket+Ridge model."""

    model_name = "minirocket_ridge"
    pipeline = "A"
    class_weight_mode = "balanced"
    hyperparameter_candidates = ({},)

    def __init__(
        self,
        global_config_path: Path | None = None,
        model_config_path: Path | None = None,
    ) -> None:
        """Initialize config state, output paths, and fold geometry.

        Args:
            global_config_path: Optional path to `configs/modeling_global.yaml`.
            model_config_path: Optional path to `configs/modeling_minirocket_ridge.yaml`.
        """
        project_root = _resolve_project_root(global_config_path=global_config_path, model_config_path=model_config_path)
        resolved_global = global_config_path or project_root / "configs" / "modeling_global.yaml"
        resolved_model = model_config_path or project_root / "configs" / "modeling_minirocket_ridge.yaml"
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
        """Load aligned Stage 1 artifacts for Pipeline A."""
        super().load_data()
        if not isinstance(self.x_, np.ndarray):
            raise TypeError("Pipeline A trainer expected a numpy array after load_data().")
        if self.feature_names_ is None:
            raise ValueError("Pipeline A trainer expected channel names after load_data().")

        LOGGER.info(
            "Loaded Pipeline A tensor for model=%s with shape=%s, channel_order=%s, clip_length=%s.",
            self.model_name,
            self.x_.shape,
            self.feature_names_,
            self.x_.shape[1],
        )

    def build_estimator(self, hparams: dict[str, Any], class_weights: Any) -> CalibratedClassifierCV:
        """Build the calibrated Ridge classifier used after MiniRocket.

        Args:
            hparams: Unused fixed-configuration hyperparameters.
            class_weights: Fold-specific class weighting mode.

        Returns:
            A configured `CalibratedClassifierCV` wrapper.
        """
        del hparams
        ridge_estimator = RidgeClassifier(class_weight=class_weights)
        calibration_config = self.model_config["calibration_layer_a"]
        return CalibratedClassifierCV(
            estimator=ridge_estimator,
            method=str(calibration_config["method"]),
            cv=int(calibration_config["cv"]),
            ensemble=bool(calibration_config["ensemble"]),
        )

    def fit_fold(
        self,
        outer_fold: int,
        inner_train_idx: np.ndarray,
        inner_val_idx: np.ndarray,
        hparams: dict[str, Any],
    ) -> dict[str, Any]:
        """Fit one MiniRocket+Ridge split and return validation artifacts.

        Args:
            outer_fold: Outer-fold identifier used for logging.
            inner_train_idx: Absolute indices into the locked 777-row train order.
            inner_val_idx: Absolute indices into the locked 777-row train order.
            hparams: Unused fixed-configuration hyperparameters.

        Returns:
            Validation probabilities, score, timing diagnostics, and fitted objects.
        """
        del hparams
        if not isinstance(self.x_train_, np.ndarray) or self.y_train_ is None:
            raise ValueError("Split state must be loaded before fit_fold().")

        _ensure_numba_cache_dir()
        x_inner_train = self.x_train_[inner_train_idx]
        x_inner_val = self.x_train_[inner_val_idx]
        y_inner_train = self.y_train_[inner_train_idx]
        y_inner_val = self.y_train_[inner_val_idx]

        x_inner_train_ncl = np.transpose(x_inner_train, (0, 2, 1)).astype(np.float32, copy=False)
        x_inner_val_ncl = np.transpose(x_inner_val, (0, 2, 1)).astype(np.float32, copy=False)

        minirocket_config = self.model_config["minirocket"]
        transformer = MiniRocketMultivariate(
            num_kernels=int(minirocket_config["num_kernels"]),
            random_state=int(minirocket_config["random_state"]),
        )

        transform_fit_start = perf_counter()
        train_features = _coerce_feature_frame(transformer.fit_transform(x_inner_train_ncl))
        transform_fit_time = perf_counter() - transform_fit_start

        transform_predict_start = perf_counter()
        val_features = _coerce_feature_frame(
            transformer.transform(x_inner_val_ncl),
            columns=train_features.columns.tolist(),
        )
        transform_predict_time = perf_counter() - transform_predict_start

        LOGGER.info(
            "MiniRocket transform wall-clock outer_fold=%s fit_transform=%.3fs transform=%.3fs.",
            outer_fold,
            transform_fit_time,
            transform_predict_time,
        )

        normalizer = _fit_feature_matrix_normalizer(train_features)
        LOGGER.info(
            "MiniRocket post-transform normalization uses PipelineBNormalizer-style stats on [%s, %s] feature matrix.",
            train_features.shape[0],
            train_features.shape[1],
        )
        train_features_norm = normalizer.transform(train_features)
        val_features_norm = normalizer.transform(val_features)

        estimator = self.build_estimator(hparams={}, class_weights=self.current_class_weights_)
        estimator_fit_start = perf_counter()
        estimator.fit(train_features_norm, y_inner_train)
        estimator_fit_time = perf_counter() - estimator_fit_start

        prediction_start = perf_counter()
        val_probabilities = estimator.predict_proba(val_features_norm)[:, 1]
        prediction_time = perf_counter() - prediction_start

        if len(np.unique(y_inner_val)) < 2:
            score = float("nan")
        else:
            score = float(roc_auc_score(y_inner_val, val_probabilities))

        return {
            "score": score,
            "val_probabilities": val_probabilities,
            "fit_time": transform_fit_time + estimator_fit_time,
            "predict_time": transform_predict_time + prediction_time,
            "transform_fit_time": transform_fit_time,
            "transform_predict_time": transform_predict_time,
            "normalizer": normalizer,
            "transformer": transformer,
        }

    def run(self, outer_fold_limit: int | None = None) -> None:
        """Execute the Session 6 nested-CV workflow.

        Args:
            outer_fold_limit: Optional number of outer folds to run, used by smoke tests.
        """
        self.load_data()
        self._load_split_state()
        if self.y_ is None or self.train_idx_ is None or self.sample_ids_ is None or self.y_train_ is None:
            raise ValueError("Data and split state must be available before run().")

        train_sample_ids = self.sample_ids_[self.train_idx_]
        held_out_count = len(self.y_) - len(self.train_idx_)
        LOGGER.info(
            "Locked split derives train_n=%s and heldout_n=%s from final cohort n=%s for model=%s.",
            len(self.train_idx_),
            held_out_count,
            len(self.y_),
            self.model_name,
        )
        LOGGER.info(
            "Session 6 OOF metrics computed with p_cal=p_raw (Layer-B calibrator deferred to Session 10 Phase 2)."
        )
        LOGGER.info(
            "Session 6 nested-CV summaries use threshold_p_raw=%.2f until the final Youden threshold is fit in Session 10 Phase 2.",
            SESSION6_THRESHOLD_P_RAW,
        )

        selected_folds = self.outer_folds_[:outer_fold_limit] if outer_fold_limit is not None else self.outer_folds_
        oof_records: list[dict[str, Any]] = []
        per_fold_metric_records: list[dict[str, Any]] = []
        per_fold_results: list[dict[str, Any]] = []

        for fold in selected_folds:
            diagnostic_scores: list[float] = []
            for inner_fold_idx, (inner_train_idx, inner_val_idx) in enumerate(fold.inner_folds):
                self.current_class_weights_ = self._compute_inner_train_class_weights(self.y_train_[inner_train_idx])
                diagnostic_result = self.fit_fold(
                    outer_fold=fold.outer_fold,
                    inner_train_idx=inner_train_idx,
                    inner_val_idx=inner_val_idx,
                    hparams={},
                )
                diagnostic_scores.append(float(diagnostic_result["score"]))
                LOGGER.info(
                    "MiniRocket diagnostic inner AUC outer_fold=%s inner_fold=%s score=%.6f.",
                    fold.outer_fold,
                    inner_fold_idx,
                    diagnostic_result["score"],
                )

            LOGGER.info(
                "MiniRocket diagnostic inner AUC outer_fold=%s mean=%.6f across %s inner folds.",
                fold.outer_fold,
                float(np.nanmean(diagnostic_scores)),
                len(diagnostic_scores),
            )

            self.current_class_weights_ = self._compute_inner_train_class_weights(self.y_train_[fold.train_idx])
            outer_result = self.fit_fold(
                outer_fold=fold.outer_fold,
                inner_train_idx=fold.train_idx,
                inner_val_idx=fold.val_idx,
                hparams={},
            )
            y_outer_val = self.y_train_[fold.val_idx]
            sample_ids_outer_val = train_sample_ids[fold.val_idx]
            val_probabilities = np.asarray(outer_result["val_probabilities"], dtype=float)
            best_hparams_json = json.dumps({}, sort_keys=True)

            fold_metrics = compute_full_metric_set(
                y_true=y_outer_val,
                p_raw=val_probabilities,
                p_cal=val_probabilities,
                threshold_p_raw=SESSION6_THRESHOLD_P_RAW,
            )
            per_fold_metric_records.append({"scope": "outer_fold", "outer_fold": fold.outer_fold, **fold_metrics})
            per_fold_results.append(
                {
                    "outer_fold": fold.outer_fold,
                    "best_hparams": {},
                    "diagnostic_inner_auc_scores": diagnostic_scores,
                    "diagnostic_inner_auc_mean": float(np.nanmean(diagnostic_scores)),
                    "metrics": fold_metrics,
                    "fit_time": float(outer_result["fit_time"]),
                    "predict_time": float(outer_result["predict_time"]),
                    "transform_fit_time": float(outer_result["transform_fit_time"]),
                    "transform_predict_time": float(outer_result["transform_predict_time"]),
                }
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

        oof_frame = pd.DataFrame(oof_records)
        if oof_frame.empty:
            raise ValueError("MiniRocket nested CV produced no OOF records.")
        if not np.isfinite(oof_frame["p_raw"]).all():
            raise ValueError("MiniRocket nested CV produced non-finite OOF probabilities.")

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
            "per_fold_results": per_fold_results,
            "aggregate_metrics": aggregate_with_ci,
            "git_commit_hash": _get_git_commit_hash(self.project_root),
            "training_timestamp_utc": _utc_now_iso(),
        }
        self.save_outputs()

    def save_outputs(self) -> None:
        """Persist the Session 6 MiniRocket nested-CV outputs."""
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

        LOGGER.info("Saved MiniRocket nested-CV outputs under outputs/{predictions,metrics,analysis}/.")


def run_nested_cv(config_path: Path) -> None:
    """Run MiniRocket+Ridge nested CV from a model config path.

    Args:
        config_path: Path to `configs/modeling_minirocket_ridge.yaml`.
    """
    trainer = MiniRocketRidgeTrainer(model_config_path=config_path)
    trainer.run()


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the MiniRocket trainer."""
    parser = argparse.ArgumentParser(description="Run Session 6 MiniRocket+Ridge nested CV.")
    parser.add_argument("--config", required=True, type=Path, help="Path to modeling_minirocket_ridge.yaml.")
    parser.add_argument(
        "--global-config",
        type=Path,
        default=None,
        help="Optional path to modeling_global.yaml.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for Session 6 MiniRocket nested CV."""
    _configure_logging()
    args = _parse_args()
    trainer = MiniRocketRidgeTrainer(
        global_config_path=args.global_config,
        model_config_path=args.config,
    )
    trainer.run()


if __name__ == "__main__":
    main()
