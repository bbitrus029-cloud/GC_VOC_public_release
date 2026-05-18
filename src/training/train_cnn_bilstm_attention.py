"""CNN-BiLSTM-Attention nested cross-validation for Session 9.

This trainer mirrors the Session 8 InceptionTime nested-CV flow while adding
attention-weight extraction for the outer-fold OOF validation samples:

- `BaseNestedCVTrainer.load_data()` aligns the full 972-row cohort across
  `sample_index.csv` and `pipeline_a_raw.npz`.
- `BaseNestedCVTrainer._load_split_state()` then derives `self.x_train_` and
  `self.y_train_` in the locked 777-row training-set order.
- The outer-fold artifacts use the 777-row training-subset index space, so the
  inner-CV search and OOF prediction steps follow the two-step access pattern
  `x_train_777 = x_[train_idx_972]` then `x_train_777[idx_777]`.
- The DL early-stopping holdout artifacts use the 972-row cohort index space,
  so the outer-fold refit step indexes the full aligned tensor directly with
  `x_[dl_holdout_idx_972]` and applies the fitted normalizer to the 777-space
  OOF validation subset afterwards.
- `PipelineANormalizer` is fit on `[N, L, C]` tensors only; `VOCDataModule`
  handles the required `[N, C, L]` transpose internally.
- `CNNBiLSTMAttentionClassifier` accepts hyperparameters via the `hparams` dict,
  receives fold-specific class weights through its `class_weights` property, and
  exposes `get_attention_weights(x)` returning `[batch, time]` attention maps.
- Session 9 does not fit Layer-B Platt calibration yet; summary metrics
  therefore use `p_cal=p_raw`, and threshold-dependent metrics use the fixed
  placeholder threshold `0.5` until Session 10 Phase 2.
- Attention weights are saved as an architectural sanity check only and are
  excluded from the cross-model SHAP/Kendall Tau matrix per plan §2.21 and
  modeling-guardrails §6.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import N_GAS_CHANNELS, RANDOM_SEED
from src.evaluation.metrics import compute_full_metric_set
from src.models.cnn_bilstm_attention import CNNBiLSTMAttentionClassifier
from src.training.common.base_trainer import BaseNestedCVTrainer
from src.training.common.bootstrap import bootstrap_metric
from src.training.common.normalizer import fit_pipeline_a_normalizer
from src.training.train_inception_time import (
    _build_data_module,
    _configure_determinism,
    _configure_logging,
    _ensure_parent_dir,
    _get_git_commit_hash,
    _json_default,
    _load_yaml_config,
    _NaNLossMonitor,
    _relative_path_str,
    _resolve_path,
    _resolve_project_root,
    _summarize_metrics_csv,
    _utc_now_iso,
)


LOGGER = logging.getLogger(__name__)
SESSION9_THRESHOLD_P_RAW = 0.5


@dataclass(frozen=True)
class _FitResult:
    """Container for one training fit outcome."""

    score: float
    val_probabilities: np.ndarray | None
    best_model: CNNBiLSTMAttentionClassifier | None
    fit_time: float
    predict_time: float
    best_epoch: int
    epochs_run: int
    early_stopping_triggered: bool
    val_auc_curve_summary: list[dict[str, float]]
    nan_retry_count: int
    oof_status: str


def _make_bootstrap_metric_fn(metric_name: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Create a metric callback for the shared bootstrap utility."""

    def _metric_fn(y_true: np.ndarray, p_raw: np.ndarray) -> float:
        metrics = compute_full_metric_set(
            y_true=y_true,
            p_raw=p_raw,
            p_cal=p_raw,
            threshold_p_raw=SESSION9_THRESHOLD_P_RAW,
        )
        return float(metrics[metric_name])

    return _metric_fn


def _model_device(accelerator: str) -> torch.device:
    """Return the concrete inference device for the configured accelerator."""
    if accelerator == "gpu" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _predict_probabilities(
    model: CNNBiLSTMAttentionClassifier,
    x_eval: np.ndarray,
    batch_size: int,
    accelerator: str,
) -> np.ndarray:
    """Predict positive-class probabilities for a normalized `[N, L, C]` tensor."""
    device = _model_device(accelerator=accelerator)
    model = model.to(device)
    model.eval()

    x_eval_ncl = np.transpose(x_eval, (0, 2, 1)).astype(np.float32, copy=False)
    dataset = TensorDataset(torch.from_numpy(x_eval_ncl))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for (batch_x,) in dataloader:
            logits = model(batch_x.to(device))
            batch_probabilities = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            probabilities.append(batch_probabilities)
    if not probabilities:
        return np.empty(0, dtype=float)
    return np.concatenate(probabilities).astype(float, copy=False)


def _extract_attention_weights(
    model: CNNBiLSTMAttentionClassifier,
    x_eval: np.ndarray,
    batch_size: int,
    accelerator: str,
) -> np.ndarray:
    """Extract `[N, T_attention]` attention weights for a normalized `[N, L, C]` tensor."""
    device = _model_device(accelerator=accelerator)
    model = model.to(device)
    model.eval()

    x_eval_ncl = np.transpose(x_eval, (0, 2, 1)).astype(np.float32, copy=False)
    dataset = TensorDataset(torch.from_numpy(x_eval_ncl))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    attention_batches: list[np.ndarray] = []
    with torch.no_grad():
        for (batch_x,) in dataloader:
            batch_attention = model.get_attention_weights(batch_x.to(device)).detach().cpu().numpy()
            if batch_attention.ndim != 2:
                raise ValueError(
                    f"Expected attention weights with shape [batch, time], got {batch_attention.shape}."
                )
            attention_batches.append(batch_attention)
    if not attention_batches:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(attention_batches, axis=0).astype(np.float32, copy=False)


def _coerce_sample_ids_to_int64(sample_ids: np.ndarray) -> np.ndarray:
    """Convert aligned sample IDs to the required `int64` artifact representation."""
    try:
        return sample_ids.astype(np.int64)
    except ValueError as exc:
        raise ValueError("CBLA attention artifact requires sample IDs convertible to int64.") from exc


def _save_attention_artifact(
    project_root: Path,
    attention_path: Path,
    per_fold_attention: list[dict[str, Any]],
    git_commit_hash: str,
    training_timestamp_utc: str,
) -> tuple[bool, int]:
    """Persist the Session 9 attention sanity-check artifact."""
    if not per_fold_attention:
        return False, 0

    sample_id_list = [_coerce_sample_ids_to_int64(record["sample_ids"]) for record in per_fold_attention]
    attention_list = [np.asarray(record["attention_weights"], dtype=np.float32) for record in per_fold_attention]
    outer_fold_list = [
        np.full(attention.shape[0], int(record["outer_fold"]), dtype=np.int64)
        for record, attention in zip(per_fold_attention, attention_list, strict=True)
    ]

    attention_widths = {attention.shape[1] for attention in attention_list}
    if len(attention_widths) != 1:
        raise ValueError(f"Inconsistent attention widths across folds: {sorted(attention_widths)}.")

    sample_ids = np.concatenate(sample_id_list, axis=0)
    attention_weights = np.concatenate(attention_list, axis=0)
    outer_fold = np.concatenate(outer_fold_list, axis=0)
    if len(np.unique(sample_ids)) != len(sample_ids):
        raise ValueError("Attention artifact sample IDs must be unique across outer folds.")

    sort_order = np.argsort(sample_ids)
    sample_ids = sample_ids[sort_order]
    attention_weights = attention_weights[sort_order]
    outer_fold = outer_fold[sort_order]
    t_attention = int(attention_weights.shape[1])

    _ensure_parent_dir(attention_path)
    np.savez_compressed(
        attention_path,
        sample_id=sample_ids.astype(np.int64, copy=False),
        attention_weights=attention_weights.astype(np.float32, copy=False),
        outer_fold=outer_fold.astype(np.int64, copy=False),
        T_attention=np.array(t_attention, dtype=np.int64),
        git_commit_hash=np.array(git_commit_hash),
        training_timestamp_utc=np.array(training_timestamp_utc),
    )
    LOGGER.info(
        "Saved CBLA attention sanity artifact to %s with shape=%s.",
        _relative_path_str(project_root, attention_path),
        attention_weights.shape,
    )
    return True, t_attention


class CNNBiLSTMAttentionTrainer(BaseNestedCVTrainer):
    """Nested-CV trainer for the prespecified CNN-BiLSTM-Attention model."""

    model_name = "cnn_bilstm_attention"
    pipeline = "A"
    class_weight_mode = "dl_two_class"
    hyperparameter_candidates = ({},)

    def __init__(
        self,
        global_config_path: Path | None = None,
        model_config_path: Path | None = None,
    ) -> None:
        """Initialize config state, output paths, and fold geometry."""
        project_root = _resolve_project_root(global_config_path=global_config_path, model_config_path=model_config_path)
        resolved_global = global_config_path or project_root / "configs" / "modeling_global.yaml"
        resolved_model = model_config_path or project_root / "configs" / "modeling_cnn_bilstm_attention.yaml"
        super().__init__(global_config_path=resolved_global, model_config_path=resolved_model)

        self.project_root = project_root
        self.global_config = _load_yaml_config(self.global_config_path)
        self.model_config = _load_yaml_config(self.model_config_path)
        self.outer_n_folds = int(self.global_config["cv"]["outer_folds"])
        self.inner_n_folds = int(self.global_config["cv"]["inner_folds"])
        self.bootstrap_n_boot = int(self.global_config["bootstrap"]["n_boot"])
        self.hyperparameter_candidates = tuple(
            dict(candidate) for candidate in self.model_config["hyperparameter_candidates"]
        )
        self.output_paths = {
            key: _resolve_path(project_root=self.project_root, value=value)
            for key, value in self.model_config["paths"].items()
        }
        self.csv_logger_root = _resolve_path(
            project_root=self.project_root,
            value=self.model_config["training"]["csv_logger"]["path"],
        )
        self.per_fold_results_: list[dict[str, Any]] = []
        self.aggregate_metrics_with_ci_: dict[str, dict[str, float]] = {}

    def load_data(self) -> None:
        """Load aligned Stage 1 artifacts for Pipeline A."""
        super().load_data()
        if not isinstance(self.x_, np.ndarray):
            raise TypeError("Pipeline A trainer expected a numpy array after load_data().")
        LOGGER.info(
            "Loaded Pipeline A tensor for model=%s with shape=%s and n_channels=%s.",
            self.model_name,
            self.x_.shape,
            self.x_.shape[-1],
        )

    def build_estimator(self, hparams: dict[str, Any], class_weights: Any) -> CNNBiLSTMAttentionClassifier:
        """Build the CBLA LightningModule for one fit."""
        model = CNNBiLSTMAttentionClassifier(n_channels=N_GAS_CHANNELS, n_classes=2, hparams=hparams)
        model.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        return model

    def fit_fold(
        self,
        outer_fold: int,
        inner_train_idx: np.ndarray,
        inner_val_idx: np.ndarray,
        hparams: dict[str, Any],
    ) -> dict[str, Any]:
        """Satisfy the abstract interface; Session 9 uses the custom `run()` loop."""
        raise NotImplementedError(
            "Session 9 uses CNNBiLSTMAttentionTrainer.run() instead of BaseNestedCVTrainer.run_nested_cv()."
        )

    def _load_dl_holdout(self, outer_fold: int) -> tuple[np.ndarray, np.ndarray]:
        """Load the locked DL early-stopping holdout artifact for one outer fold."""
        holdout_path = self.project_root / "data" / "processed" / "splits" / f"dl_holdout_{outer_fold}.npz"
        if not holdout_path.exists():
            raise FileNotFoundError(f"Expected DL holdout artifact at {holdout_path}.")
        holdout_data = np.load(holdout_path, allow_pickle=True)
        return holdout_data["train_idx"].astype(int), holdout_data["holdout_idx"].astype(int)

    def _train_single_fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        hparams: dict[str, Any],
        class_weights: np.ndarray,
        fit_label: str,
    ) -> _FitResult:
        """Train one CBLA model with the locked retry policy."""
        training_config = self.model_config["training"]
        early_stopping_config = training_config["early_stopping"]
        nan_retry_config = training_config["nan_retry"]
        max_retries = int(nan_retry_config["max_retries"]) if nan_retry_config["enabled"] else 0
        batch_size = int(hparams["batch_size"])
        accelerator = str(training_config["accelerator"])
        devices = int(training_config["devices"])

        current_hparams = dict(hparams)
        for attempt in range(max_retries + 1):
            retry_count = attempt
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            LOGGER.info(
                "Starting %s attempt=%s with hparams=%s and class_weights=%s.",
                fit_label,
                attempt + 1,
                json.dumps(current_hparams, sort_keys=True, default=_json_default),
                np.asarray(class_weights, dtype=float).tolist(),
            )

            datamodule = _build_data_module(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                batch_size=batch_size,
                seed=RANDOM_SEED,
            )
            model = self.build_estimator(hparams=current_hparams, class_weights=class_weights)
            nan_monitor = _NaNLossMonitor()
            early_stopping = EarlyStopping(
                monitor=str(early_stopping_config["monitor"]),
                mode=str(early_stopping_config["mode"]),
                patience=int(early_stopping_config["patience"]),
            )
            csv_logger = CSVLogger(
                save_dir=str(self.csv_logger_root.parent),
                name=self.csv_logger_root.name,
            )
            checkpoint_callback = ModelCheckpoint(
                dirpath=Path(csv_logger.log_dir) / "checkpoints",
                filename="best",
                monitor=str(early_stopping_config["monitor"]),
                mode=str(early_stopping_config["mode"]),
                save_top_k=1,
            )
            LOGGER.info(
                "CSVLogger active for %s at %s.",
                fit_label,
                _relative_path_str(self.project_root, Path(csv_logger.log_dir)),
            )

            pl.seed_everything(RANDOM_SEED, workers=True)
            trainer = pl.Trainer(
                max_epochs=int(training_config["max_epochs"]),
                precision=training_config["precision"],
                deterministic=bool(training_config["deterministic"]),
                accelerator=accelerator,
                devices=devices,
                logger=csv_logger,
                callbacks=[nan_monitor, early_stopping, checkpoint_callback],
                enable_progress_bar=False,
                enable_model_summary=False,
                log_every_n_steps=1,
                num_sanity_val_steps=0,
            )

            fit_start = perf_counter()
            non_finite_loss_detected = False
            try:
                trainer.fit(model, datamodule=datamodule)
            except ValueError as exc:
                if nan_monitor.nan_detected or "non-finite" in str(exc).lower():
                    non_finite_loss_detected = True
                    LOGGER.warning(
                        "Detected non-finite %s loss during %s attempt=%s.",
                        nan_monitor.stage or "unknown",
                        fit_label,
                        attempt + 1,
                    )
                else:
                    raise
            fit_time = perf_counter() - fit_start

            metric_summary = _summarize_metrics_csv(Path(csv_logger.log_dir) / "metrics.csv")
            LOGGER.info(
                "%s val_auc per epoch: %s",
                fit_label,
                metric_summary["val_auc_curve_summary"],
            )

            if fit_time > 300.0:
                LOGGER.warning(
                    "Observed fit_time=%.3fs (>300s) for %s with hparams=%s.",
                    fit_time,
                    fit_label,
                    json.dumps(current_hparams, sort_keys=True, default=_json_default),
                )

            if non_finite_loss_detected:
                if attempt < max_retries and nan_retry_config["lr_halving"]:
                    current_hparams["lr"] = float(current_hparams["lr"]) / 2.0
                    LOGGER.warning(
                        "Retrying %s after non-finite loss with halved lr=%.6g.",
                        fit_label,
                        float(current_hparams["lr"]),
                    )
                    continue

                LOGGER.error(
                    "Non-finite loss persisted after %s retries for %s with hparams=%s.",
                    retry_count,
                    fit_label,
                    json.dumps(current_hparams, sort_keys=True, default=_json_default),
                )
                return _FitResult(
                    score=float("nan"),
                    val_probabilities=None,
                    best_model=None,
                    fit_time=fit_time,
                    predict_time=0.0,
                    best_epoch=int(metric_summary["best_epoch"]),
                    epochs_run=int(metric_summary["epochs_run"]),
                    early_stopping_triggered=bool(early_stopping.stopped_epoch > 0),
                    val_auc_curve_summary=list(metric_summary["val_auc_curve_summary"]),
                    nan_retry_count=retry_count,
                    oof_status="failed",
                )

            best_model = model
            if checkpoint_callback.best_model_path:
                best_model = CNNBiLSTMAttentionClassifier.load_from_checkpoint(
                    checkpoint_callback.best_model_path,
                    n_channels=N_GAS_CHANNELS,
                    n_classes=2,
                    hparams=current_hparams,
                    map_location=_model_device(accelerator=accelerator),
                )

            predict_start = perf_counter()
            val_probabilities = _predict_probabilities(
                model=best_model,
                x_eval=x_val,
                batch_size=batch_size,
                accelerator=accelerator,
            )
            predict_time = perf_counter() - predict_start

            if not np.isfinite(val_probabilities).all():
                if attempt < max_retries and nan_retry_config["lr_halving"]:
                    current_hparams["lr"] = float(current_hparams["lr"]) / 2.0
                    LOGGER.warning(
                        "Retrying %s after non-finite probabilities with halved lr=%.6g.",
                        fit_label,
                        float(current_hparams["lr"]),
                    )
                    continue
                LOGGER.error(
                    "Non-finite probabilities persisted after %s retries for %s.",
                    retry_count,
                    fit_label,
                )
                return _FitResult(
                    score=float("nan"),
                    val_probabilities=None,
                    best_model=None,
                    fit_time=fit_time,
                    predict_time=predict_time,
                    best_epoch=int(metric_summary["best_epoch"]),
                    epochs_run=int(metric_summary["epochs_run"]),
                    early_stopping_triggered=bool(early_stopping.stopped_epoch > 0),
                    val_auc_curve_summary=list(metric_summary["val_auc_curve_summary"]),
                    nan_retry_count=retry_count,
                    oof_status="failed",
                )

            score = float("nan") if len(np.unique(y_val)) < 2 else float(roc_auc_score(y_val, val_probabilities))
            LOGGER.info(
                "Completed %s with score=%.6f, best_epoch=%s, epochs_run=%s, early_stopping_triggered=%s, "
                "nan_retry_count=%s.",
                fit_label,
                score,
                int(metric_summary["best_epoch"]),
                int(metric_summary["epochs_run"]),
                bool(early_stopping.stopped_epoch > 0),
                retry_count,
            )
            return _FitResult(
                score=score,
                val_probabilities=val_probabilities,
                best_model=best_model,
                fit_time=fit_time,
                predict_time=predict_time,
                best_epoch=int(metric_summary["best_epoch"]),
                epochs_run=int(metric_summary["epochs_run"]),
                early_stopping_triggered=bool(early_stopping.stopped_epoch > 0),
                val_auc_curve_summary=list(metric_summary["val_auc_curve_summary"]),
                nan_retry_count=retry_count,
                oof_status="ok",
            )

        raise RuntimeError("Unexpected retry loop exit in _train_single_fit().")

    def run(self, outer_fold_limit: int | None = None) -> None:
        """Execute the Session 9 nested-CV workflow."""
        _configure_determinism()
        self.load_data()
        self._load_split_state()

        if not isinstance(self.x_, np.ndarray) or not isinstance(self.x_train_, np.ndarray):
            raise TypeError("CBLA trainer requires numpy Pipeline A tensors.")
        if self.y_ is None or self.y_train_ is None or self.train_idx_ is None or self.sample_ids_ is None:
            raise ValueError("Data and split state must be available before run().")

        LOGGER.info(
            "Session 9 uses the two-step 777-space access pattern for outer-fold artifacts: "
            "x_train_777 = x_[train_idx_972], then x_train_777[fold_idx_777]."
        )
        LOGGER.info(
            "Session 9 uses direct 972-space access for DL holdouts: x_[dl_holdout_idx_972]."
        )
        LOGGER.info(
            "Configured mixed precision mode=%s.",
            self.model_config["training"]["precision"],
        )
        LOGGER.info(
            "Attention weights saved as architectural sanity check; excluded from the cross-model SHAP matrix "
            "per plan §2.21 and modeling-guardrails §6."
        )

        train_sample_ids = self.sample_ids_[self.train_idx_]
        selected_folds = self.outer_folds_[:outer_fold_limit] if outer_fold_limit is not None else self.outer_folds_
        oof_records: list[dict[str, Any]] = []
        per_fold_metric_records: list[dict[str, Any]] = []
        per_fold_results: list[dict[str, Any]] = []
        per_fold_best_epoch: list[int] = []
        per_fold_early_stopping_triggered: list[bool] = []
        per_fold_nan_retry_count: list[int] = []
        per_fold_failed_status: list[bool] = []
        attention_per_fold: list[dict[str, Any]] = []

        for fold in selected_folds:
            LOGGER.info("Starting CNN-BiLSTM-Attention outer_fold=%s.", fold.outer_fold)
            candidate_summaries: list[dict[str, Any]] = []
            best_candidate_hparams: dict[str, Any] | None = None
            best_candidate_score = float("-inf")

            for candidate_idx, candidate_hparams in enumerate(self.hyperparameter_candidates):
                inner_scores: list[float] = []
                inner_retry_counts: list[int] = []
                inner_failures: list[bool] = []

                for inner_fold_idx, (inner_train_idx, inner_val_idx) in enumerate(fold.inner_folds):
                    inner_class_weights = np.asarray(
                        self._compute_inner_train_class_weights(self.y_train_[inner_train_idx]),
                        dtype=float,
                    )
                    LOGGER.info(
                        "Computed inner class weights for outer_fold=%s candidate=%s inner_fold=%s: %s",
                        fold.outer_fold,
                        candidate_idx,
                        inner_fold_idx,
                        inner_class_weights.tolist(),
                    )

                    x_inner_train = self.x_train_[inner_train_idx]
                    x_inner_val = self.x_train_[inner_val_idx]
                    y_inner_train = self.y_train_[inner_train_idx]
                    y_inner_val = self.y_train_[inner_val_idx]
                    normalizer = fit_pipeline_a_normalizer(x_inner_train)
                    x_inner_train_norm = normalizer.transform(x_inner_train)
                    x_inner_val_norm = normalizer.transform(x_inner_val)

                    fit_result = self._train_single_fit(
                        x_train=x_inner_train_norm,
                        y_train=y_inner_train,
                        x_val=x_inner_val_norm,
                        y_val=y_inner_val,
                        hparams=candidate_hparams,
                        class_weights=inner_class_weights,
                        fit_label=f"outer{fold.outer_fold}_candidate{candidate_idx}_inner{inner_fold_idx}",
                    )
                    inner_scores.append(float(fit_result.score))
                    inner_retry_counts.append(int(fit_result.nan_retry_count))
                    inner_failures.append(fit_result.oof_status == "failed")

                candidate_mean_score = (
                    float(np.nanmean(inner_scores)) if not np.all(np.isnan(inner_scores)) else float("nan")
                )
                candidate_summaries.append(
                    {
                        "candidate_idx": candidate_idx,
                        "candidate_hparams": dict(candidate_hparams),
                        "inner_scores": inner_scores,
                        "mean_inner_score": candidate_mean_score,
                        "nan_retry_count_total": int(np.sum(inner_retry_counts)),
                        "any_inner_failure": bool(any(inner_failures)),
                    }
                )
                LOGGER.info(
                    "Outer_fold=%s candidate=%s mean_inner_auc=%.6f from scores=%s.",
                    fold.outer_fold,
                    candidate_idx,
                    candidate_mean_score,
                    inner_scores,
                )
                if not np.isnan(candidate_mean_score) and candidate_mean_score > best_candidate_score:
                    best_candidate_score = candidate_mean_score
                    best_candidate_hparams = dict(candidate_hparams)

            if best_candidate_hparams is None:
                raise ValueError(f"No valid CBLA hyperparameter candidate succeeded for outer_fold={fold.outer_fold}.")

            es_train_idx, es_holdout_idx = self._load_dl_holdout(outer_fold=fold.outer_fold)
            if len(np.union1d(es_train_idx, es_holdout_idx)) != len(fold.train_idx):
                raise ValueError(
                    f"DL holdout cardinality mismatch for outer_fold={fold.outer_fold}: "
                    f"{len(np.union1d(es_train_idx, es_holdout_idx))} vs {len(fold.train_idx)}."
                )

            outer_class_weights = np.asarray(
                self._compute_inner_train_class_weights(self.y_[es_train_idx]),
                dtype=float,
            )
            LOGGER.info(
                "Computed outer refit class weights for outer_fold=%s: %s",
                fold.outer_fold,
                outer_class_weights.tolist(),
            )

            x_es_train = self.x_[es_train_idx]
            x_es_holdout = self.x_[es_holdout_idx]
            y_es_train = self.y_[es_train_idx]
            y_es_holdout = self.y_[es_holdout_idx]
            outer_normalizer = fit_pipeline_a_normalizer(x_es_train)
            x_es_train_norm = outer_normalizer.transform(x_es_train)
            x_es_holdout_norm = outer_normalizer.transform(x_es_holdout)

            outer_fit_result = self._train_single_fit(
                x_train=x_es_train_norm,
                y_train=y_es_train,
                x_val=x_es_holdout_norm,
                y_val=y_es_holdout,
                hparams=best_candidate_hparams,
                class_weights=outer_class_weights,
                fit_label=f"outer{fold.outer_fold}_refit",
            )

            outer_failed = outer_fit_result.oof_status == "failed"
            per_fold_best_epoch.append(int(outer_fit_result.best_epoch))
            per_fold_early_stopping_triggered.append(bool(outer_fit_result.early_stopping_triggered))
            per_fold_nan_retry_count.append(int(outer_fit_result.nan_retry_count))
            per_fold_failed_status.append(bool(outer_failed))

            outer_predict_time = outer_fit_result.predict_time
            fold_metrics: dict[str, float] | None = None
            if not outer_failed and outer_fit_result.val_probabilities is not None:
                x_outer_val = self.x_train_[fold.val_idx.astype(int)]
                y_outer_val = self.y_train_[fold.val_idx.astype(int)]
                sample_ids_outer_val = train_sample_ids[fold.val_idx.astype(int)]
                x_outer_val_norm = outer_normalizer.transform(x_outer_val)
                if outer_fit_result.best_model is None:
                    outer_failed = True
                    per_fold_failed_status[-1] = True
                else:
                    predict_start = perf_counter()
                    fold_probabilities = _predict_probabilities(
                        model=outer_fit_result.best_model,
                        x_eval=x_outer_val_norm,
                        batch_size=int(best_candidate_hparams["batch_size"]),
                        accelerator=str(self.model_config["training"]["accelerator"]),
                    )
                    outer_predict_time = perf_counter() - predict_start
                    if not np.isfinite(fold_probabilities).all():
                        outer_failed = True
                        per_fold_failed_status[-1] = True
                    else:
                        fold_attention = _extract_attention_weights(
                            model=outer_fit_result.best_model,
                            x_eval=x_outer_val_norm,
                            batch_size=int(best_candidate_hparams["batch_size"]),
                            accelerator=str(self.model_config["training"]["accelerator"]),
                        )
                        if not np.isfinite(fold_attention).all():
                            raise ValueError(f"Non-finite attention weights detected for outer_fold={fold.outer_fold}.")
                        attention_per_fold.append(
                            {
                                "outer_fold": int(fold.outer_fold),
                                "sample_ids": sample_ids_outer_val.copy(),
                                "attention_weights": fold_attention.copy(),
                            }
                        )
                        LOGGER.info(
                            "Extracted attention weights for outer_fold=%s with shape=%s.",
                            fold.outer_fold,
                            fold_attention.shape,
                        )
                        fold_metrics = compute_full_metric_set(
                            y_true=y_outer_val,
                            p_raw=fold_probabilities,
                            p_cal=fold_probabilities,
                            threshold_p_raw=SESSION9_THRESHOLD_P_RAW,
                        )
                        per_fold_metric_records.append({"scope": "outer_fold", "outer_fold": fold.outer_fold, **fold_metrics})
                        best_hparams_json = json.dumps(best_candidate_hparams, sort_keys=True, default=_json_default)
                        for sample_id, y_true, p_raw in zip(
                            sample_ids_outer_val,
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
                                    "best_hparams_json": best_hparams_json,
                                }
                            )

            per_fold_results.append(
                {
                    "outer_fold": int(fold.outer_fold),
                    "best_hparams": dict(best_candidate_hparams),
                    "candidate_summaries": candidate_summaries,
                    "metrics": fold_metrics or {},
                    "fit_time": float(outer_fit_result.fit_time),
                    "predict_time": float(outer_predict_time),
                    "best_epoch": int(outer_fit_result.best_epoch),
                    "epochs_run": int(outer_fit_result.epochs_run),
                    "early_stopping_triggered": bool(outer_fit_result.early_stopping_triggered),
                    "val_auc_curve_summary": list(outer_fit_result.val_auc_curve_summary),
                    "nan_retry_count": int(outer_fit_result.nan_retry_count),
                    "oof_status": outer_fit_result.oof_status,
                }
            )
            LOGGER.info(
                "Completed outer_fold=%s with best_hparams=%s, best_epoch=%s, early_stopping_triggered=%s, "
                "nan_retry_count=%s, oof_status=%s.",
                fold.outer_fold,
                json.dumps(best_candidate_hparams, sort_keys=True, default=_json_default),
                int(outer_fit_result.best_epoch),
                bool(outer_fit_result.early_stopping_triggered),
                int(outer_fit_result.nan_retry_count),
                outer_fit_result.oof_status,
            )

        oof_frame = pd.DataFrame(oof_records)
        if not oof_frame.empty:
            oof_frame.sort_values("sample_id", kind="mergesort", inplace=True)
            oof_frame.reset_index(drop=True, inplace=True)

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

        aggregate_metrics_with_ci: dict[str, dict[str, float]] = {}
        if not oof_frame.empty:
            aggregate_metrics = compute_full_metric_set(
                y_true=oof_frame["y_true"].to_numpy(dtype=int),
                p_raw=oof_frame["p_raw"].to_numpy(dtype=float),
                p_cal=oof_frame["p_raw"].to_numpy(dtype=float),
                threshold_p_raw=SESSION9_THRESHOLD_P_RAW,
            )
            y_oof = oof_frame["y_true"].to_numpy(dtype=int)
            p_oof = oof_frame["p_raw"].to_numpy(dtype=float)
            for metric_name, metric_value in aggregate_metrics.items():
                bootstrap_result = bootstrap_metric(
                    y_true=y_oof,
                    y_score=p_oof,
                    metric_fn=_make_bootstrap_metric_fn(metric_name),
                    n_boot=self.bootstrap_n_boot,
                    seed=RANDOM_SEED,
                )
                aggregate_metrics_with_ci[metric_name] = {
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

        folds_early_stopped_before_epoch_20 = int(
            np.sum(
                [
                    bool(stopped) and int(best_epoch) < 20
                    for stopped, best_epoch in zip(
                        per_fold_early_stopping_triggered,
                        per_fold_best_epoch,
                        strict=True,
                    )
                ]
            )
        )
        any_fold_nan_after_retry = bool(any(per_fold_failed_status))
        trigger_fires = bool(folds_early_stopped_before_epoch_20 >= 3 or any_fold_nan_after_retry)
        if trigger_fires:
            LOGGER.warning(
                "Post-hoc G trigger fires for %s: folds_early_stopped_before_epoch_20=%s, any_fold_nan_after_retry=%s.",
                self.model_name,
                folds_early_stopped_before_epoch_20,
                any_fold_nan_after_retry,
            )

        git_commit_hash = _get_git_commit_hash(self.project_root)
        training_timestamp_utc = _utc_now_iso()
        attention_extracted, t_attention = _save_attention_artifact(
            project_root=self.project_root,
            attention_path=self.output_paths["attention_npz"],
            per_fold_attention=attention_per_fold,
            git_commit_hash=git_commit_hash,
            training_timestamp_utc=training_timestamp_utc,
        )

        self.nested_cv_oof_ = (
            oof_frame.loc[:, ["sample_id", "y_true", "p_raw", "outer_fold"]].copy()
            if not oof_frame.empty
            else pd.DataFrame(columns=["sample_id", "y_true", "p_raw", "outer_fold"])
        )
        if not oof_frame.empty:
            per_fold_frame = oof_frame.loc[:, ["sample_id", "y_true", "p_raw", "outer_fold", "best_hparams_json"]].copy()
            per_fold_frame.sort_values("sample_id", kind="mergesort", inplace=True)
            per_fold_frame.reset_index(drop=True, inplace=True)
            self.nested_cv_per_fold_ = per_fold_frame
        else:
            self.nested_cv_per_fold_ = pd.DataFrame(
                columns=["sample_id", "y_true", "p_raw", "outer_fold", "best_hparams_json"]
            )
        self.nested_cv_summary_ = pd.DataFrame(summary_records)
        self.nested_cv_results_ = {
            "model_name": self.model_name,
            "train_sample_count": int(len(self.train_idx_)),
            "oof_row_count": int(len(self.nested_cv_oof_)),
            "threshold_p_raw_placeholder": SESSION9_THRESHOLD_P_RAW,
            "layer_b_calibrator_deferred": True,
            "per_fold_results": per_fold_results,
            "aggregate_metrics": aggregate_metrics_with_ci,
            "per_fold_best_epoch": per_fold_best_epoch,
            "per_fold_early_stopping_triggered": per_fold_early_stopping_triggered,
            "per_fold_nan_retry_count": per_fold_nan_retry_count,
            "per_fold_failed_status": per_fold_failed_status,
            "posthoc_G_trigger_check": {
                "folds_early_stopped_before_epoch_20": folds_early_stopped_before_epoch_20,
                "any_fold_nan_after_retry": any_fold_nan_after_retry,
                "trigger_fires": trigger_fires,
            },
            "attention_extracted": attention_extracted,
            "attention_npz_path": _relative_path_str(self.project_root, self.output_paths["attention_npz"]),
            "T_attention": int(t_attention),
            "git_commit_hash": git_commit_hash,
            "training_timestamp_utc": training_timestamp_utc,
        }
        self.per_fold_results_ = per_fold_results
        self.aggregate_metrics_with_ci_ = aggregate_metrics_with_ci
        self.save_outputs()

    def save_outputs(self) -> None:
        """Persist the Session 9 CBLA nested-CV outputs."""
        if (
            self.nested_cv_oof_ is None
            or self.nested_cv_per_fold_ is None
            or self.nested_cv_summary_ is None
            or self.nested_cv_results_ is None
        ):
            raise ValueError("Nested-CV outputs are unavailable; run() must complete before save_outputs().")

        for output_key, output_path in self.output_paths.items():
            if output_key == "attention_npz":
                continue
            _ensure_parent_dir(output_path)

        self.nested_cv_oof_.to_csv(self.output_paths["nested_cv_oof"], index=False)
        self.nested_cv_per_fold_.to_csv(self.output_paths["nested_cv_per_fold"], index=False)
        self.nested_cv_summary_.to_csv(self.output_paths["nested_cv_summary"], index=False)
        with self.output_paths["nested_cv_results"].open("w", encoding="utf-8") as handle:
            json.dump(self.nested_cv_results_, handle, indent=2, sort_keys=True, default=_json_default)

        LOGGER.info("Saved CBLA nested-CV outputs under outputs/{predictions,metrics,analysis,shap}/.")


def run_nested_cv(config_path: Path) -> None:
    """Run CBLA nested CV from a model config path."""
    trainer = CNNBiLSTMAttentionTrainer(model_config_path=config_path)
    trainer.run()


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the CBLA trainer."""
    parser = argparse.ArgumentParser(description="Run Session 9 CNN-BiLSTM-Attention nested CV.")
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to modeling_cnn_bilstm_attention.yaml.",
    )
    parser.add_argument(
        "--global-config",
        type=Path,
        default=None,
        help="Optional path to modeling_global.yaml.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for Session 9 CNN-BiLSTM-Attention nested CV."""
    _configure_logging()
    args = _parse_args()
    trainer = CNNBiLSTMAttentionTrainer(
        global_config_path=args.global_config,
        model_config_path=args.config,
    )
    trainer.run()


if __name__ == "__main__":
    main()
