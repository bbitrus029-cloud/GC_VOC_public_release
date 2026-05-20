"""Phase 2 final-model training entry points.

This module fills the Session 10 Phase 1 skeleton for Stage 4b final training.
It reuses the established trainer build helpers rather than duplicating model
construction logic:

- `MiniRocketRidgeTrainer.build_estimator` from `src.training.train_minirocket_ridge`
- `XGBoostTrainer.build_estimator` from `src.training.train_xgboost`
- `SVMTrainer.build_estimator` from `src.training.train_svm`
- `ElasticNetLRTrainer.build_estimator` from `src.training.train_elastic_net_lr`
- `InceptionTimeTrainer.build_estimator` / `_train_single_fit` from
  `src.training.train_inception_time`
- `CNNBiLSTMAttentionTrainer.build_estimator` / `_train_single_fit` from
  `src.training.train_cnn_bilstm_attention`

Operational plan §2.5 OOF reuse-or-rerun rule:

- "If every hyperparameter equals the aggregated final hyperparameter in all 5 folds, reuse
  `binary_<m>_nested_cv_oof.csv` and set `oof_source = "nested_cv_reused"`."
- "Otherwise run one-time 5-fold CV on the 777 training samples using final hyperparameters
  and the existing outer fold indices, save `binary_<m>_final_hp_oof.csv`, and set
  `oof_source = "final_hp_rerun"`."
- "The OOF predictions used for Youden threshold and Platt calibrator fitting come from the
  5 outer-fold models ... They do NOT come from the final model itself, which is refit on
  the entire 777 training set."

Per-model non-HP plumbing key exclusion list `NON_HP_KEYS_PER_MODEL` applies
the §2.5 aggregation rule to genuine HP search dimensions only. Excluded keys
are restored from canonical config sources per E009.

For Stage 4b DL refits, `final_dl_es_holdout.npz` provides the prespecified
679/98 early-stopping partition over the locked 777 training samples. The
Pipeline A normalizer is therefore fitted on the 679-sample early-stopping
train subset and then applied to the 98-sample holdout, the full 777-sample
training set for in-sample diagnostics, and the 195-sample test set. This
matches the Session 10 alignment readback and avoids leaking holdout or test
statistics into the DL fit.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import pickle
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score, roc_curve
from sktime.transformations.panel.rocket import MiniRocketMultivariate
import torch
import yaml

from src.config import GAS_CHANNELS, MODEL_NAMES, PROJECT_ROOT, RANDOM_SEED
from src.evaluation.calibration import apply_platt_calibrator, fit_platt_calibrator
from src.evaluation.metrics import compute_full_metric_set
from src.training.common.runtime import ensure_numba_cache_dir
from src.training.dl_inference import DLPredictProbaWrapper
from src.training.train_cnn_bilstm_attention import (
    CNNBiLSTMAttentionTrainer,
    _predict_probabilities as _predict_cbla_probabilities,
)
from src.training.train_elastic_net_lr import ElasticNetLRTrainer
from src.training.train_inception_time import (
    InceptionTimeTrainer,
    _predict_probabilities as _predict_inception_probabilities,
)
from src.training.train_minirocket_ridge import (
    MiniRocketRidgeTrainer,
    _coerce_feature_frame as _coerce_minirocket_feature_frame,
    _fit_feature_matrix_normalizer,
)
from src.training.train_svm import SVMTrainer
from src.training.train_xgboost import XGBoostTrainer
from src.training.common.normalizer import fit_pipeline_a_normalizer, fit_pipeline_b_normalizer


LOGGER = logging.getLogger(__name__)
PIPELINE_A_MODELS = {"minirocket_ridge", "inception_time", "cnn_bilstm_attention"}
PIPELINE_B_MODELS = {"xgboost", "svm", "elastic_net_lr"}
TRAINER_REGISTRY = {
    "minirocket_ridge": MiniRocketRidgeTrainer,
    "inception_time": InceptionTimeTrainer,
    "cnn_bilstm_attention": CNNBiLSTMAttentionTrainer,
    "xgboost": XGBoostTrainer,
    "svm": SVMTrainer,
    "elastic_net_lr": ElasticNetLRTrainer,
}
MODEL_CONFIG_FILENAMES = {
    model_name: f"modeling_{model_name}.yaml" for model_name in MODEL_NAMES
}
NON_HP_KEYS_PER_MODEL = {
    "xgboost": {"random_state", "n_jobs", "tree_method"},
    "svm": {"class_weight", "kernel", "probability", "random_state"},
    "elastic_net_lr": {"class_weight", "max_iter", "penalty", "random_state", "solver"},
    "inception_time": set(),
    "cnn_bilstm_attention": set(),
    "minirocket_ridge": set(),
}
THRESHOLD_MAPPING_NOTE = (
    "Platt calibration is monotonic and does not change AUC-ROC or AUC-PR; it may change "
    "Brier score, calibration slope, and calibration intercept. All threshold-dependent metrics "
    "are computed once in p_raw space using threshold_p_raw."
)
TRAIN_IN_SAMPLE_WARNING = (
    "# WARNING: In-sample predictions are an overfitting upper bound and are not used for any claim."
)


@dataclass(frozen=True)
class FinalModelSummary:
    """Compact per-model summary for the orchestration log."""

    model_name: str
    oof_source: str
    threshold_p_raw: float
    threshold_p_final_equivalent: float
    test_auc_roc: float
    test_auc_roc_ci_low: float
    test_auc_roc_ci_high: float


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


def _resolve_path(path_str: str) -> Path:
    """Resolve a project-root-relative path."""
    candidate = Path(path_str)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _ensure_parent_dir(path: Path) -> None:
    """Create the parent directory for an output file."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _relative_path_str(path: Path) -> str:
    """Return a project-relative path string for logs."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


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


def _json_default(value: Any) -> Any:
    """Convert numpy-rich objects into JSON-serializable primitives."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return _relative_path_str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _mode_with_first_tiebreak(values: list[Any]) -> Any:
    """Return the modal value, breaking ties by first appearance."""
    counts = Counter(values)
    max_count = max(counts.values())
    for value in values:
        if counts[value] == max_count:
            return value
    raise ValueError("mode computation received an empty list.")


def _canonicalize_mode_value(value: Any) -> Any:
    """Convert nested lists into hashable tuples for mode counting."""
    if isinstance(value, list):
        return tuple(_canonicalize_mode_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_canonicalize_mode_value(item) for item in value)
    return value


def _restore_mode_value(reference: Any, canonical_value: Any) -> Any:
    """Restore the original collection type after modal aggregation."""
    if isinstance(reference, list):
        return list(canonical_value)
    if isinstance(reference, tuple):
        return tuple(canonical_value)
    return canonical_value


def _extract_per_fold_best_hparams(results_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the five outer-fold best-hyperparameter dictionaries."""
    if "per_fold_results" in results_payload:
        per_fold_results = results_payload["per_fold_results"]
        if not isinstance(per_fold_results, list):
            raise TypeError("per_fold_results must be a list.")
        extracted = [dict(entry["best_hparams"]) for entry in per_fold_results]
    elif "per_fold_best_hparams" in results_payload:
        extracted = [dict(entry) for entry in results_payload["per_fold_best_hparams"]]
    else:
        raise KeyError("Could not find per-fold best hyperparameters in nested-CV results payload.")

    if len(extracted) != 5:
        raise ValueError(f"Expected 5 outer-fold best-hyperparameter dictionaries, found {len(extracted)}.")
    return extracted


def _restore_canonical_non_hp_values(
    model_name: str,
    global_config: dict[str, Any],
    model_config: dict[str, Any],
) -> dict[str, Any]:
    """Restore E009 non-HP plumbing keys from canonical config sources."""
    non_hp_keys = NON_HP_KEYS_PER_MODEL[model_name]
    canonical_values: dict[str, Any] = {}

    if "random_state" in non_hp_keys:
        canonical_values["random_state"] = int(global_config.get("random_seed", RANDOM_SEED))

    if model_name == "xgboost":
        fixed_config = model_config["xgboost_fixed"]
        if "n_jobs" in non_hp_keys:
            canonical_values["n_jobs"] = int(fixed_config["n_jobs"])
        if "tree_method" in non_hp_keys:
            canonical_values["tree_method"] = str(fixed_config["tree_method"])
    elif model_name == "svm":
        fixed_config = model_config["svc_fixed"]
        if "class_weight" in non_hp_keys:
            canonical_values["class_weight"] = str(fixed_config["class_weight"])
        if "kernel" in non_hp_keys:
            canonical_values["kernel"] = str(fixed_config["kernel"])
        if "probability" in non_hp_keys:
            canonical_values["probability"] = bool(fixed_config["probability"])
    elif model_name == "elastic_net_lr":
        fixed_config = model_config["lr_fixed"]
        if "class_weight" in non_hp_keys:
            canonical_values["class_weight"] = str(fixed_config["class_weight"])
        if "max_iter" in non_hp_keys:
            canonical_values["max_iter"] = int(fixed_config["max_iter"])
        if "penalty" in non_hp_keys:
            canonical_values["penalty"] = str(fixed_config["penalty"])
        if "solver" in non_hp_keys:
            canonical_values["solver"] = str(fixed_config["solver"])

    return canonical_values


def _aggregate_final_hparams(
    model_name: str,
    per_fold_best_hparams: list[dict[str, Any]],
    global_config: dict[str, Any],
    model_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Aggregate final hyperparameters by the plan's median/mode rules."""
    if len(per_fold_best_hparams) != 5:
        raise ValueError("Phase 2 hyperparameter aggregation requires exactly five folds.")

    non_hp_keys = NON_HP_KEYS_PER_MODEL[model_name]
    filtered_per_fold_hparams = [
        {
            key: value
            for key, value in fold_hparams.items()
            if key not in non_hp_keys
        }
        for fold_hparams in per_fold_best_hparams
    ]
    all_keys = sorted({key for fold_hparams in filtered_per_fold_hparams for key in fold_hparams})
    final_hparams: dict[str, Any] = {}
    consistency_flags: dict[str, bool] = {}

    for key in all_keys:
        values = [fold_hparams[key] for fold_hparams in filtered_per_fold_hparams]
        consistency_flags[key] = len({_canonicalize_mode_value(value) for value in values}) == 1

        first_value = values[0]
        if isinstance(first_value, (bool, np.bool_)):
            canonical_values = [_canonicalize_mode_value(bool(value)) for value in values]
            final_hparams[key] = bool(_mode_with_first_tiebreak(canonical_values))
        elif isinstance(first_value, (int, np.integer)) and not isinstance(first_value, (bool, np.bool_)):
            final_hparams[key] = int(round(float(np.median(np.asarray(values, dtype=float)))))
        elif isinstance(first_value, (float, np.floating)):
            final_hparams[key] = float(np.median(np.asarray(values, dtype=float)))
        elif isinstance(first_value, (str, list, tuple)):
            canonical_values = [_canonicalize_mode_value(value) for value in values]
            modal_value = _mode_with_first_tiebreak(canonical_values)
            final_hparams[key] = _restore_mode_value(first_value, modal_value)
        else:
            raise TypeError(f"Unsupported hyperparameter type for key '{key}': {type(first_value).__name__}.")

    final_hparams.update(
        _restore_canonical_non_hp_values(
            model_name=model_name,
            global_config=global_config,
            model_config=model_config,
        )
    )
    return final_hparams, consistency_flags


def _load_nested_cv_results(path: Path) -> dict[str, Any]:
    """Load a nested-CV results JSON payload."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except OSError as exc:
        raise OSError(f"Failed to read nested-CV results from {path}.") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON mapping in {path}.")
    return loaded


def _load_oof_frame(path: Path) -> pd.DataFrame:
    """Load and validate an OOF prediction table."""
    try:
        frame = pd.read_csv(path)
    except OSError as exc:
        raise OSError(f"Failed to read OOF predictions from {path}.") from exc

    required_columns = ["sample_id", "y_true", "p_raw", "outer_fold"]
    if frame.columns.tolist() != required_columns:
        raise ValueError(f"Expected OOF columns {required_columns}, found {frame.columns.tolist()} in {path}.")
    if len(frame) != 777:
        raise ValueError(f"Expected 777 OOF rows in {path}, found {len(frame)}.")

    validated = frame.copy()
    validated["sample_id"] = pd.to_numeric(validated["sample_id"], errors="raise").astype(np.int64)
    validated["y_true"] = pd.to_numeric(validated["y_true"], errors="raise").astype(int)
    validated["p_raw"] = pd.to_numeric(validated["p_raw"], errors="raise").astype(float)
    validated["outer_fold"] = pd.to_numeric(validated["outer_fold"], errors="raise").astype(int)
    validated.sort_values("sample_id", kind="mergesort", inplace=True)
    validated.reset_index(drop=True, inplace=True)
    if not validated["sample_id"].is_monotonic_increasing:
        raise ValueError(f"OOF sample_id ordering is not monotonic increasing in {path}.")
    if not np.isfinite(validated["p_raw"]).all():
        raise ValueError(f"Detected non-finite OOF probabilities in {path}.")
    return validated


def _get_training_sample_ids(trainer: Any) -> np.ndarray:
    """Return the locked 777 training sample IDs as sorted int64."""
    if trainer.sample_index_ is None or trainer.train_idx_ is None:
        raise ValueError("Trainer split state must be loaded before reading training sample IDs.")
    training_sample_ids = (
        trainer.sample_index_.iloc[trainer.train_idx_]["sample_id"].to_numpy(dtype=np.int64, copy=True)
    )
    if not np.all(np.diff(training_sample_ids) >= 0):
        raise ValueError("Training sample IDs must be sorted ascending.")
    return training_sample_ids


def _get_test_sample_ids(trainer: Any) -> np.ndarray:
    """Return the locked 195 test sample IDs as sorted int64."""
    if trainer.sample_index_ is None or trainer.test_idx_ is None:
        raise ValueError("Trainer split state must be loaded before reading test sample IDs.")
    test_sample_ids = trainer.sample_index_.iloc[trainer.test_idx_]["sample_id"].to_numpy(dtype=np.int64, copy=True)
    if not np.all(np.diff(test_sample_ids) >= 0):
        raise ValueError("Test sample IDs must be sorted ascending.")
    return test_sample_ids


def _fit_youden_threshold(y_true: np.ndarray, p_raw: np.ndarray) -> float:
    """Fit the raw-probability Youden threshold from OOF predictions."""
    fpr, tpr, thresholds = roc_curve(y_true, p_raw)
    youden = tpr - fpr
    finite_mask = np.isfinite(thresholds)
    if finite_mask.any():
        finite_youden = youden[finite_mask]
        best_value = np.max(finite_youden)
        best_idx = np.flatnonzero(finite_mask & (youden == best_value))[0]
    else:
        best_idx = int(np.argmax(youden))
    return float(thresholds[best_idx])


def _draw_stratified_indices(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample bootstrap indices with the class balance preserved."""
    sampled_indices: list[np.ndarray] = []
    for class_label in np.unique(y_true):
        class_indices = np.flatnonzero(y_true == class_label)
        sampled_indices.append(rng.choice(class_indices, size=len(class_indices), replace=True))
    combined = np.concatenate(sampled_indices)
    return rng.permutation(combined)


def _bootstrap_metric_rows(
    y_true: np.ndarray,
    p_raw: np.ndarray,
    p_cal: np.ndarray,
    threshold_p_raw: float,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap the full metric set and return one tidy dataframe."""
    point_estimates = compute_full_metric_set(
        y_true=y_true,
        p_raw=p_raw,
        p_cal=p_cal,
        threshold_p_raw=threshold_p_raw,
    )
    metric_names = list(point_estimates)
    bootstrap_values = {metric_name: np.empty(n_boot, dtype=float) for metric_name in metric_names}
    rng = np.random.default_rng(seed)

    for bootstrap_idx in range(n_boot):
        sampled_indices = _draw_stratified_indices(y_true=y_true, rng=rng)
        sampled_metrics = compute_full_metric_set(
            y_true=y_true[sampled_indices],
            p_raw=p_raw[sampled_indices],
            p_cal=p_cal[sampled_indices],
            threshold_p_raw=threshold_p_raw,
        )
        for metric_name in metric_names:
            bootstrap_values[metric_name][bootstrap_idx] = float(sampled_metrics[metric_name])

    rows: list[dict[str, Any]] = []
    for metric_name in metric_names:
        ci_low, ci_high = np.nanpercentile(bootstrap_values[metric_name], [2.5, 97.5])
        rows.append(
            {
                "scope": "test",
                "metric": metric_name,
                "value": float(point_estimates[metric_name]),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
            }
        )
    return pd.DataFrame(rows)


def _quadrant_label(y_true: int, y_pred: int) -> str:
    """Return the confusion-quadrant label for one binary prediction."""
    if y_true == 1 and y_pred == 1:
        return "TP"
    if y_true == 0 and y_pred == 0:
        return "TN"
    if y_true == 0 and y_pred == 1:
        return "FP"
    return "FN"


def _build_prediction_frame(
    sample_ids: np.ndarray,
    y_true: np.ndarray,
    p_raw: np.ndarray,
    p_cal: np.ndarray,
    threshold_p_raw: float,
) -> pd.DataFrame:
    """Build the common prediction dataframe schema."""
    y_pred = (p_raw >= threshold_p_raw).astype(int)
    frame = pd.DataFrame(
        {
            "sample_id": sample_ids.astype(np.int64, copy=False),
            "y_true": y_true.astype(int, copy=False),
            "p_raw": p_raw.astype(float, copy=False),
            "p_cal": p_cal.astype(float, copy=False),
            "y_pred": y_pred.astype(int, copy=False),
        }
    )
    frame["quadrant"] = [
        _quadrant_label(y_true=int(y_true_i), y_pred=int(y_pred_i))
        for y_true_i, y_pred_i in zip(frame["y_true"], frame["y_pred"], strict=True)
    ]
    frame.sort_values("sample_id", kind="mergesort", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def _write_prediction_csv(path: Path, frame: pd.DataFrame, warning_header: str | None = None) -> None:
    """Write a prediction dataframe, optionally with a warning comment header."""
    _ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        if warning_header is not None:
            handle.write(f"{warning_header}\n")
        frame.to_csv(handle, index=False)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload with deterministic formatting."""
    _ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")


def _write_pickle(path: Path, payload: Any) -> None:
    """Write a pickle payload to disk."""
    _ensure_parent_dir(path)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def _predict_positive_probabilities(
    model_name: str,
    transformer: Any,
    normalizer: Any,
    model: Any,
    x_input: np.ndarray | pd.DataFrame,
) -> np.ndarray:
    """Run end-to-end positive-class inference for one model family."""
    if model_name == "minirocket_ridge":
        if not isinstance(x_input, np.ndarray):
            raise TypeError("MiniRocket inference expects a raw Pipeline A numpy tensor.")
        x_input_ncl = np.transpose(x_input, (0, 2, 1)).astype(np.float32, copy=False)
        features = _coerce_minirocket_feature_frame(
            transformer.transform(x_input_ncl),
            columns=list(normalizer.feature_names),
        )
        normalized = normalizer.transform(features)
    elif model_name in PIPELINE_B_MODELS:
        if not isinstance(x_input, pd.DataFrame):
            raise TypeError("Pipeline B inference expects a pandas dataframe.")
        normalized = normalizer.transform(x_input)
    elif model_name in PIPELINE_A_MODELS:
        if not isinstance(x_input, np.ndarray):
            raise TypeError("DL inference expects a raw Pipeline A numpy tensor.")
        normalized = normalizer.transform(x_input)
    else:
        raise ValueError(f"Unsupported model name: {model_name}.")

    probabilities = model.predict_proba(normalized)
    probabilities_array = np.asarray(probabilities, dtype=float)
    if probabilities_array.ndim != 2 or probabilities_array.shape[1] != 2:
        raise ValueError(f"Expected binary predict_proba output with shape [N, 2], found {probabilities_array.shape}.")
    return probabilities_array[:, 1]


def _rerun_oof_for_minirocket(trainer: MiniRocketRidgeTrainer, final_hparams: dict[str, Any]) -> pd.DataFrame:
    """Rerun the one-time final-hyperparameter OOF workflow for MiniRocket."""
    del final_hparams
    if not isinstance(trainer.y_train_, np.ndarray):
        raise ValueError("MiniRocket trainer split state is incomplete.")

    train_sample_ids = _get_training_sample_ids(trainer)
    oof_records: list[dict[str, Any]] = []
    for fold in trainer.outer_folds_:
        trainer.current_class_weights_ = trainer._compute_inner_train_class_weights(trainer.y_train_[fold.train_idx])
        fold_result = trainer.fit_fold(
            outer_fold=fold.outer_fold,
            inner_train_idx=fold.train_idx,
            inner_val_idx=fold.val_idx,
            hparams={},
        )
        fold_probabilities = np.asarray(fold_result["val_probabilities"], dtype=float)
        for sample_id, y_true, p_raw in zip(
            train_sample_ids[fold.val_idx.astype(int)],
            trainer.y_train_[fold.val_idx.astype(int)],
            fold_probabilities,
            strict=True,
        ):
            oof_records.append(
                {
                    "sample_id": int(sample_id),
                    "y_true": int(y_true),
                    "p_raw": float(p_raw),
                    "outer_fold": int(fold.outer_fold),
                }
            )

    oof_frame = pd.DataFrame(oof_records).sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    return oof_frame


def _rerun_oof_for_pipeline_b(
    trainer: XGBoostTrainer | SVMTrainer | ElasticNetLRTrainer,
    final_hparams: dict[str, Any],
) -> pd.DataFrame:
    """Rerun the one-time final-hyperparameter OOF workflow for a Pipeline B model."""
    if not isinstance(trainer.y_train_, np.ndarray):
        raise ValueError("Pipeline B trainer split state is incomplete.")

    train_sample_ids = _get_training_sample_ids(trainer)
    oof_records: list[dict[str, Any]] = []
    for fold in trainer.outer_folds_:
        trainer.current_class_weights_ = trainer._compute_inner_train_class_weights(trainer.y_train_[fold.train_idx])
        fold_result = trainer.fit_fold(
            outer_fold=fold.outer_fold,
            inner_train_idx=fold.train_idx,
            inner_val_idx=fold.val_idx,
            hparams=final_hparams,
        )
        fold_probabilities = np.asarray(fold_result["val_probabilities"], dtype=float)
        for sample_id, y_true, p_raw in zip(
            train_sample_ids[fold.val_idx.astype(int)],
            trainer.y_train_[fold.val_idx.astype(int)],
            fold_probabilities,
            strict=True,
        ):
            oof_records.append(
                {
                    "sample_id": int(sample_id),
                    "y_true": int(y_true),
                    "p_raw": float(p_raw),
                    "outer_fold": int(fold.outer_fold),
                }
            )

    oof_frame = pd.DataFrame(oof_records).sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    return oof_frame


def _rerun_oof_for_dl(
    trainer: InceptionTimeTrainer | CNNBiLSTMAttentionTrainer,
    final_hparams: dict[str, Any],
    predictor: Callable[[Any, np.ndarray, int, str], np.ndarray],
) -> pd.DataFrame:
    """Rerun the one-time final-hyperparameter OOF workflow for a DL model."""
    if not isinstance(trainer.x_, np.ndarray) or not isinstance(trainer.x_train_, np.ndarray) or trainer.y_ is None:
        raise ValueError("DL trainer split state is incomplete.")

    train_sample_ids = _get_training_sample_ids(trainer)
    oof_records: list[dict[str, Any]] = []
    for fold in trainer.outer_folds_:
        es_train_idx, es_holdout_idx = trainer._load_dl_holdout(outer_fold=fold.outer_fold)
        if len(np.union1d(es_train_idx, es_holdout_idx)) != len(fold.train_idx):
            raise ValueError(
                f"DL holdout cardinality mismatch for outer_fold={fold.outer_fold}: "
                f"{len(np.union1d(es_train_idx, es_holdout_idx))} vs {len(fold.train_idx)}."
            )

        outer_class_weights = np.asarray(trainer._compute_inner_train_class_weights(trainer.y_[es_train_idx]), dtype=float)
        LOGGER.info(
            "Computed final-hyperparameter OOF class weights for model=%s outer_fold=%s: %s",
            trainer.model_name,
            fold.outer_fold,
            outer_class_weights.tolist(),
        )
        x_es_train = trainer.x_[es_train_idx]
        x_es_holdout = trainer.x_[es_holdout_idx]
        y_es_train = trainer.y_[es_train_idx]
        y_es_holdout = trainer.y_[es_holdout_idx]

        outer_normalizer = fit_pipeline_a_normalizer(x_es_train)
        x_es_train_norm = outer_normalizer.transform(x_es_train)
        x_es_holdout_norm = outer_normalizer.transform(x_es_holdout)
        fit_result = trainer._train_single_fit(
            x_train=x_es_train_norm,
            y_train=y_es_train,
            x_val=x_es_holdout_norm,
            y_val=y_es_holdout,
            hparams=final_hparams,
            class_weights=outer_class_weights,
            fit_label=f"outer{fold.outer_fold}_final_hp_rerun",
        )
        if fit_result.oof_status != "ok" or fit_result.best_model is None:
            raise RuntimeError(f"DL final-hyperparameter OOF rerun failed for {trainer.model_name} outer_fold={fold.outer_fold}.")

        x_outer_val = trainer.x_train_[fold.val_idx.astype(int)]
        y_outer_val = trainer.y_train_[fold.val_idx.astype(int)]
        x_outer_val_norm = outer_normalizer.transform(x_outer_val)
        fold_probabilities = predictor(
            model=fit_result.best_model,
            x_eval=x_outer_val_norm,
            batch_size=int(final_hparams["batch_size"]),
            accelerator=str(trainer.model_config["training"]["accelerator"]),
        )
        if not np.isfinite(fold_probabilities).all():
            raise ValueError(f"Non-finite OOF probabilities detected for {trainer.model_name} outer_fold={fold.outer_fold}.")

        LOGGER.info(
            "Completed final-hyperparameter OOF rerun for model=%s outer_fold=%s with best_epoch=%s, "
            "early_stopping_triggered=%s, nan_retry_count=%s.",
            trainer.model_name,
            fold.outer_fold,
            fit_result.best_epoch,
            fit_result.early_stopping_triggered,
            fit_result.nan_retry_count,
        )
        for sample_id, y_true, p_raw in zip(
            train_sample_ids[fold.val_idx.astype(int)],
            y_outer_val,
            fold_probabilities,
            strict=True,
        ):
            oof_records.append(
                {
                    "sample_id": int(sample_id),
                    "y_true": int(y_true),
                    "p_raw": float(p_raw),
                    "outer_fold": int(fold.outer_fold),
                }
            )

    oof_frame = pd.DataFrame(oof_records).sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    return oof_frame


def _load_final_dl_holdout() -> tuple[np.ndarray, np.ndarray]:
    """Load the Session 10 final DL early-stopping holdout artifact."""
    holdout_path = PROJECT_ROOT / "data" / "processed" / "splits" / "final_dl_es_holdout.npz"
    if not holdout_path.exists():
        raise FileNotFoundError(f"Expected final DL early-stopping holdout at {holdout_path}.")
    holdout_data = np.load(holdout_path, allow_pickle=False)
    if set(holdout_data.files) != {"train_idx", "holdout_idx"}:
        raise ValueError(f"Unexpected keys in {holdout_path}: {holdout_data.files}.")
    train_idx = holdout_data["train_idx"].astype(int)
    holdout_idx = holdout_data["holdout_idx"].astype(int)
    if len(train_idx) != 679 or len(holdout_idx) != 98 or len(np.union1d(train_idx, holdout_idx)) != 777:
        raise ValueError(
            "final_dl_es_holdout.npz must partition the 777 training samples into 679 train + 98 holdout."
        )
    return train_idx, holdout_idx


def _fit_final_minirocket_model(
    trainer: MiniRocketRidgeTrainer,
) -> tuple[Any, Any, Any, Any]:
    """Fit the Stage 4b final MiniRocket model and return its components."""
    if not isinstance(trainer.x_train_, np.ndarray) or trainer.y_train_ is None:
        raise ValueError("MiniRocket trainer split state is incomplete.")

    ensure_numba_cache_dir()
    class_weights = trainer._compute_inner_train_class_weights(trainer.y_train_)
    x_train_ncl = np.transpose(trainer.x_train_, (0, 2, 1)).astype(np.float32, copy=False)
    transformer_config = trainer.model_config["minirocket"]
    transformer = MiniRocketMultivariate(
        num_kernels=int(transformer_config["num_kernels"]),
        random_state=int(transformer_config["random_state"]),
    )
    train_features = _coerce_minirocket_feature_frame(transformer.fit_transform(x_train_ncl))
    normalizer = _fit_feature_matrix_normalizer(train_features)
    train_features_norm = normalizer.transform(train_features)
    model = trainer.build_estimator(hparams={}, class_weights=class_weights)
    model.fit(train_features_norm, trainer.y_train_)
    LOGGER.info(
        "Final refit class weights for model=%s: %s.",
        trainer.model_name,
        class_weights,
    )
    LOGGER.info(
        "MiniRocket refit used num_kernels=%s random_state=%s and persisted a non-None transformer.",
        transformer_config["num_kernels"],
        transformer_config["random_state"],
    )
    return transformer, normalizer, model, class_weights


def _fit_final_pipeline_b_model(
    trainer: XGBoostTrainer | SVMTrainer | ElasticNetLRTrainer,
    final_hparams: dict[str, Any],
) -> tuple[Any, Any, Any, Any]:
    """Fit the Stage 4b final Pipeline B model and return its components."""
    if not isinstance(trainer.x_train_, pd.DataFrame) or trainer.y_train_ is None or trainer.feature_names_ is None:
        raise ValueError("Pipeline B trainer split state is incomplete.")

    class_weights = trainer._compute_inner_train_class_weights(trainer.y_train_)
    normalizer = fit_pipeline_b_normalizer(trainer.x_train_, feature_names=list(trainer.feature_names_))
    x_train_norm = normalizer.transform(trainer.x_train_)
    pipeline = trainer.build_estimator(hparams=final_hparams, class_weights=class_weights)
    model = pipeline.named_steps["clf"]
    model.fit(x_train_norm, trainer.y_train_)
    LOGGER.info(
        "Final refit class weights for model=%s: %s.",
        trainer.model_name,
        class_weights,
    )
    return None, normalizer, model, class_weights


def _fit_final_dl_model(
    trainer: InceptionTimeTrainer | CNNBiLSTMAttentionTrainer,
    final_hparams: dict[str, Any],
    predictor: Callable[[Any, np.ndarray, int, str], np.ndarray],
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Fit the Stage 4b final DL model and return its components plus diagnostics."""
    if not isinstance(trainer.x_, np.ndarray) or trainer.y_ is None:
        raise ValueError("DL trainer split state is incomplete.")

    final_train_idx, final_holdout_idx = _load_final_dl_holdout()
    class_weights = np.asarray(trainer._compute_inner_train_class_weights(trainer.y_[final_train_idx]), dtype=float)
    trainer.csv_logger_root = PROJECT_ROOT / "outputs" / "logs" / "lightning_csv" / f"binary_{trainer.model_name}_final"
    x_es_train = trainer.x_[final_train_idx]
    x_es_holdout = trainer.x_[final_holdout_idx]
    y_es_train = trainer.y_[final_train_idx]
    y_es_holdout = trainer.y_[final_holdout_idx]
    normalizer = fit_pipeline_a_normalizer(x_es_train)
    x_es_train_norm = normalizer.transform(x_es_train)
    x_es_holdout_norm = normalizer.transform(x_es_holdout)
    fit_result = trainer._train_single_fit(
        x_train=x_es_train_norm,
        y_train=y_es_train,
        x_val=x_es_holdout_norm,
        y_val=y_es_holdout,
        hparams=final_hparams,
        class_weights=class_weights,
        fit_label="stage4b_final",
    )
    if fit_result.oof_status != "ok" or fit_result.best_model is None:
        raise RuntimeError(f"Final DL refit failed for {trainer.model_name}.")

    best_model_cpu = fit_result.best_model.to("cpu")
    wrapper = DLPredictProbaWrapper(
        lightning_model=best_model_cpu,
        predictor=predictor,
        batch_size=int(final_hparams["batch_size"]),
        accelerator=str(trainer.model_config["training"]["accelerator"]),
    )
    LOGGER.info(
        "Final refit class weights for model=%s: %s.",
        trainer.model_name,
        class_weights.tolist(),
    )
    LOGGER.info(
        "DL final refit diagnostics for model=%s: best_epoch=%s, early_stopping_triggered=%s, nan_retry_count=%s.",
        trainer.model_name,
        fit_result.best_epoch,
        fit_result.early_stopping_triggered,
        fit_result.nan_retry_count,
    )
    diagnostics = {
        "class_weights": class_weights.tolist(),
        "best_epoch": int(fit_result.best_epoch),
        "early_stopping_triggered": bool(fit_result.early_stopping_triggered),
        "nan_retry_count": int(fit_result.nan_retry_count),
    }
    return None, normalizer, wrapper, diagnostics


def _train_or_reuse_oof(
    model_name: str,
    trainer: Any,
    final_hparams: dict[str, Any],
    consistency_flags: dict[str, bool],
    nested_cv_oof_path: Path,
    final_hp_oof_path: Path,
) -> tuple[str, pd.DataFrame]:
    """Apply the §2.5 OOF reuse-or-rerun rule for one model."""
    if all(consistency_flags.values()):
        oof_source = "nested_cv_reused"
        oof_frame = _load_oof_frame(nested_cv_oof_path)
    else:
        oof_source = "final_hp_rerun"
        if model_name == "minirocket_ridge":
            oof_frame = _rerun_oof_for_minirocket(trainer=trainer, final_hparams=final_hparams)
        elif model_name in PIPELINE_B_MODELS:
            oof_frame = _rerun_oof_for_pipeline_b(trainer=trainer, final_hparams=final_hparams)
        elif model_name == "inception_time":
            oof_frame = _rerun_oof_for_dl(
                trainer=trainer,
                final_hparams=final_hparams,
                predictor=_predict_inception_probabilities,
            )
        elif model_name == "cnn_bilstm_attention":
            oof_frame = _rerun_oof_for_dl(
                trainer=trainer,
                final_hparams=final_hparams,
                predictor=_predict_cbla_probabilities,
            )
        else:
            raise ValueError(f"Unsupported model name for OOF rerun: {model_name}.")
        _ensure_parent_dir(final_hp_oof_path)
        oof_frame.to_csv(final_hp_oof_path, index=False)
        LOGGER.info(
            "Saved final-hyperparameter OOF predictions for model=%s to %s.",
            model_name,
            _relative_path_str(final_hp_oof_path),
        )

    LOGGER.info(
        "Phase 2 OOF decision for model=%s: oof_source=%s, consistent_hparams=%s, inconsistent_hparams=%s, final_hparams=%s.",
        model_name,
        oof_source,
        int(sum(consistency_flags.values())),
        int(len(consistency_flags) - sum(consistency_flags.values())),
        json.dumps(final_hparams, sort_keys=True, default=_json_default),
    )
    return oof_source, oof_frame


def train_final_model(model_name: str, global_config_path: Path, model_config_path: Path) -> FinalModelSummary:
    """Train one final model artifact in Session 10 Phase 2.

    Args:
        model_name: Canonical model name.
        global_config_path: Path to `configs/modeling_global.yaml`.
        model_config_path: Path to the model-specific YAML config.

    Returns:
        Compact test-set summary for the orchestration log.
    """
    model_start = perf_counter()
    global_config = _load_yaml_config(global_config_path)
    model_config = _load_yaml_config(model_config_path)
    trainer_cls = TRAINER_REGISTRY[model_name]
    trainer = trainer_cls(global_config_path=global_config_path, model_config_path=model_config_path)
    trainer.load_data()
    trainer._load_split_state()

    LOGGER.info(
        "Stage 4b final training operating on the 777 locked training samples; test set NOT loaded until "
        "threshold/calibrator are fit from OOF predictions. model=%s",
        model_name,
    )

    nested_cv_results_path = _resolve_path(str(model_config["paths"]["nested_cv_results"]))
    nested_cv_oof_path = _resolve_path(str(model_config["paths"]["nested_cv_oof"]))
    outputs_analysis_dir = PROJECT_ROOT / "outputs" / "analysis"
    outputs_models_dir = PROJECT_ROOT / "outputs" / "models"
    outputs_predictions_dir = PROJECT_ROOT / "outputs" / "predictions"
    outputs_metrics_dir = PROJECT_ROOT / "outputs" / "metrics"

    training_timestamp_utc = _utc_now_iso()
    git_commit_hash = _get_git_commit_hash(PROJECT_ROOT)
    nested_cv_results = _load_nested_cv_results(nested_cv_results_path)
    per_fold_best_hparams = _extract_per_fold_best_hparams(nested_cv_results)
    final_hparams, consistency_flags = _aggregate_final_hparams(
        model_name=model_name,
        per_fold_best_hparams=per_fold_best_hparams,
        global_config=global_config,
        model_config=model_config,
    )

    final_hp_oof_path = outputs_predictions_dir / f"binary_{model_name}_final_hp_oof.csv"
    oof_source, oof_frame = _train_or_reuse_oof(
        model_name=model_name,
        trainer=trainer,
        final_hparams=final_hparams,
        consistency_flags=consistency_flags,
        nested_cv_oof_path=nested_cv_oof_path,
        final_hp_oof_path=final_hp_oof_path,
    )

    oof_y_true = oof_frame["y_true"].to_numpy(dtype=int)
    oof_p_raw = oof_frame["p_raw"].to_numpy(dtype=float)
    threshold_p_raw = _fit_youden_threshold(y_true=oof_y_true, p_raw=oof_p_raw)
    calibrator = fit_platt_calibrator(p_raw=oof_p_raw, y_true=oof_y_true, seed=RANDOM_SEED)
    threshold_p_final_equivalent = float(
        apply_platt_calibrator(calibrator=calibrator, p_raw=np.asarray([threshold_p_raw], dtype=float))[0]
    )

    threshold_calibrator_payload = {
        "threshold_p_raw": float(threshold_p_raw),
        "threshold_p_final_equivalent": float(threshold_p_final_equivalent),
        "calibrator_object": calibrator,
        "fit_inputs_n_pos": int(np.sum(oof_y_true == 1)),
        "fit_inputs_n_neg": int(np.sum(oof_y_true == 0)),
        "fit_inputs_n_total": int(len(oof_y_true)),
        "oof_source": oof_source,
    }
    threshold_calibrator_path = outputs_models_dir / f"binary_{model_name}_threshold_calibrator.pkl"
    _write_pickle(threshold_calibrator_path, threshold_calibrator_payload)

    threshold_mapping_path = outputs_analysis_dir / f"binary_{model_name}_threshold_mapping.json"
    _write_json(
        threshold_mapping_path,
        {
            "threshold_p_raw": float(threshold_p_raw),
            "threshold_p_final_equivalent": float(threshold_p_final_equivalent),
            "monotonicity_note": THRESHOLD_MAPPING_NOTE,
        },
    )

    final_hp_decision_path = outputs_analysis_dir / f"binary_{model_name}_final_hp_decision.json"
    _write_json(
        final_hp_decision_path,
        {
            "per_fold_best_hparams": per_fold_best_hparams,
            "final_hparams": final_hparams,
            "per_hp_consistency_flags": consistency_flags,
            "oof_source": oof_source,
            "git_commit_hash": git_commit_hash,
            "training_timestamp_utc": training_timestamp_utc,
        },
    )

    if model_name == "minirocket_ridge":
        transformer, normalizer, model, class_weight_log = _fit_final_minirocket_model(trainer=trainer)
    elif model_name in PIPELINE_B_MODELS:
        transformer, normalizer, model, class_weight_log = _fit_final_pipeline_b_model(
            trainer=trainer,
            final_hparams=final_hparams,
        )
    elif model_name == "inception_time":
        transformer, normalizer, model, class_weight_log = _fit_final_dl_model(
            trainer=trainer,
            final_hparams=final_hparams,
            predictor=_predict_inception_probabilities,
        )
    elif model_name == "cnn_bilstm_attention":
        transformer, normalizer, model, class_weight_log = _fit_final_dl_model(
            trainer=trainer,
            final_hparams=final_hparams,
            predictor=_predict_cbla_probabilities,
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}.")

    training_sample_ids = _get_training_sample_ids(trainer)
    test_sample_ids = _get_test_sample_ids(trainer)
    feature_names = list(GAS_CHANNELS) if model_name in PIPELINE_A_MODELS else list(trainer.feature_names_)

    final_pkl_payload = {
        "transformer": transformer,
        "normalizer": normalizer,
        "model": model,
        "calibrator": calibrator,
        "threshold_p_raw": float(threshold_p_raw),
        "threshold_p_final_equivalent": float(threshold_p_final_equivalent),
        "feature_names": feature_names,
        "training_sample_ids": training_sample_ids.astype(np.int64, copy=False),
        "metadata": {
            "model_name": model_name,
            "final_hparams": final_hparams,
            "oof_source": oof_source,
            "git_commit_hash": git_commit_hash,
            "training_timestamp_utc": training_timestamp_utc,
        },
    }
    final_pkl_path = outputs_models_dir / f"binary_{model_name}_final.pkl"
    _write_pickle(final_pkl_path, final_pkl_payload)
    if model_name == "minirocket_ridge" and transformer is None:
        raise RuntimeError("MiniRocket final.pkl must persist a non-None fitted transformer.")

    if model_name in PIPELINE_A_MODELS:
        x_train_eval = trainer.x_train_
        x_test_eval = trainer.x_[trainer.test_idx_]
    else:
        x_train_eval = trainer.x_train_
        x_test_eval = trainer.x_.iloc[trainer.test_idx_].reset_index(drop=True)

    train_p_raw = _predict_positive_probabilities(
        model_name=model_name,
        transformer=transformer,
        normalizer=normalizer,
        model=model,
        x_input=x_train_eval,
    )
    train_p_cal = apply_platt_calibrator(calibrator=calibrator, p_raw=train_p_raw)
    train_predictions = _build_prediction_frame(
        sample_ids=training_sample_ids,
        y_true=trainer.y_train_,
        p_raw=train_p_raw,
        p_cal=train_p_cal,
        threshold_p_raw=threshold_p_raw,
    )
    train_in_sample_path = outputs_predictions_dir / f"binary_{model_name}_train_in_sample.csv"
    _write_prediction_csv(train_in_sample_path, train_predictions, warning_header=TRAIN_IN_SAMPLE_WARNING)

    test_y_true = trainer.y_[trainer.test_idx_]
    test_p_raw = _predict_positive_probabilities(
        model_name=model_name,
        transformer=transformer,
        normalizer=normalizer,
        model=model,
        x_input=x_test_eval,
    )
    test_p_cal = apply_platt_calibrator(calibrator=calibrator, p_raw=test_p_raw)
    test_predictions = _build_prediction_frame(
        sample_ids=test_sample_ids,
        y_true=test_y_true,
        p_raw=test_p_raw,
        p_cal=test_p_cal,
        threshold_p_raw=threshold_p_raw,
    )
    test_predictions_path = outputs_predictions_dir / f"binary_{model_name}_test_predictions.csv"
    _write_prediction_csv(test_predictions_path, test_predictions)

    test_metrics = _bootstrap_metric_rows(
        y_true=test_y_true,
        p_raw=test_p_raw,
        p_cal=test_p_cal,
        threshold_p_raw=threshold_p_raw,
        n_boot=int(global_config["bootstrap"]["n_boot"]),
        seed=RANDOM_SEED,
    )
    test_metrics_path = outputs_metrics_dir / f"binary_{model_name}_test_metrics.csv"
    _ensure_parent_dir(test_metrics_path)
    test_metrics.to_csv(test_metrics_path, index=False)

    test_auc_row = test_metrics.loc[test_metrics["metric"] == "auc_roc"].iloc[0]
    training_wall_clock = perf_counter() - model_start
    LOGGER.info(
        "Completed Stage 4b final model=%s with oof_source=%s, threshold_p_raw=%.6f, "
        "threshold_p_final_equivalent=%.6f, test_auc_roc=%.6f, training_wall_clock=%.3fs.",
        model_name,
        oof_source,
        threshold_p_raw,
        threshold_p_final_equivalent,
        float(test_auc_row["value"]),
        training_wall_clock,
    )
    LOGGER.info("Refit class-weight record for model=%s: %s.", model_name, class_weight_log)
    if model_name == "minirocket_ridge":
        LOGGER.info("Confirmed MiniRocket final.pkl persisted a non-None fitted transformer.")

    return FinalModelSummary(
        model_name=model_name,
        oof_source=oof_source,
        threshold_p_raw=float(threshold_p_raw),
        threshold_p_final_equivalent=float(threshold_p_final_equivalent),
        test_auc_roc=float(test_auc_row["value"]),
        test_auc_roc_ci_low=float(test_auc_row["ci_low"]),
        test_auc_roc_ci_high=float(test_auc_row["ci_high"]),
    )


def run_all_final_models(global_config_path: Path) -> None:
    """Train all final model artifacts in Session 10 Phase 2.

    Args:
        global_config_path: Path to `configs/modeling_global.yaml`.
    """
    total_start = perf_counter()
    summaries: list[FinalModelSummary] = []
    for model_name in MODEL_NAMES:
        model_config_path = PROJECT_ROOT / "configs" / MODEL_CONFIG_FILENAMES[model_name]
        summaries.append(
            train_final_model(
                model_name=model_name,
                global_config_path=global_config_path,
                model_config_path=model_config_path,
            )
        )

    LOGGER.info("Stage 4b final-training summary across all six models:")
    for summary in summaries:
        LOGGER.info(
            "model=%s test_auc_roc=%.6f (95%% CI %.6f-%.6f) oof_source=%s threshold_p_raw=%.6f "
            "threshold_p_final_equivalent=%.6f",
            summary.model_name,
            summary.test_auc_roc,
            summary.test_auc_roc_ci_low,
            summary.test_auc_roc_ci_high,
            summary.oof_source,
            summary.threshold_p_raw,
            summary.threshold_p_final_equivalent,
        )
    LOGGER.info("Completed all Stage 4b final training in %.3fs.", perf_counter() - total_start)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Phase 2 entry point."""
    parser = argparse.ArgumentParser(description="Run Session 10 Phase 2 final model training.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "modeling_global.yaml",
        help="Path to configs/modeling_global.yaml.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for `python -m src.training.train_final_models`."""
    _configure_logging()
    args = _parse_args()
    pl.seed_everything(RANDOM_SEED, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")
    run_all_final_models(global_config_path=args.config)


if __name__ == "__main__":
    main()
