"""Stage 8 SHAP cross-model channel-importance consistency analysis."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
import torch

from src.analysis.stage7_part_a_common import (
    BOOTSTRAP_CONVENTION_TEXT,
    MODEL_NAMES,
    PROJECT_ROOT,
    RANDOM_SEED,
    _format_path,
    _get_git_commit_hash,
    _load_global_config,
    _load_view_prediction_frame,
    _resolve_path,
    _timestamp_utc,
    _write_json,
    _write_text,
)
from src.config import GAS_CHANNELS
from src.training.load_final_model import load_final_model


LOGGER = logging.getLogger(__name__)
PIPELINE_B_CHANNEL_FEATURE_COUNT = 9
PERMUTATION_REPEATS = 10
IG_STEPS = 50
IG_BATCH_SIZE_GPU = 8
IG_BATCH_SIZE_CPU = 4
CHANNEL_ORDER = list(GAS_CHANNELS)
CBLA_ATTENTION_NPZ = PROJECT_ROOT / "outputs" / "shap" / "binary_cnn_bilstm_attention_nested_cv_attention.npz"
REPRESENTATIVE_SIGNALS_NPZ = PROJECT_ROOT / "outputs" / "shap" / "binary_minirocket_ridge_representative_samples_signals.npz"
REPRESENTATIVE_SAMPLE_CSV = PROJECT_ROOT / "outputs" / "analysis" / "misclassification_representative_samples_minirocket_ridge.csv"


def _configure_logging() -> None:
    """Configure INFO-level logging for CLI execution."""
    if logging.getLogger().handlers:
        logging.getLogger().setLevel(logging.INFO)
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _write_dataframe_csv(path: Path, dataframe: pd.DataFrame) -> None:
    """Write one dataframe to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def _safe_format_path(path: Path) -> str:
    """Return a project-relative path when possible, else the absolute string."""
    try:
        return _format_path(path)
    except ValueError:
        return str(path)


def _coerce_feature_frame(values: Any, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Convert a transformer output into a feature dataframe."""
    if isinstance(values, pd.DataFrame):
        frame = values.copy()
        if columns is not None:
            frame.columns = list(columns)
        return frame
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got shape {array.shape}.")
    if columns is None:
        columns = [f"feature_{index:05d}" for index in range(array.shape[1])]
    return pd.DataFrame(array, columns=list(columns))


def _load_pipeline_a_bundle(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load the full aligned Pipeline A tensor bundle."""
    try:
        with np.load(path, allow_pickle=False) as payload:
            x = np.asarray(payload["x"], dtype=float)
            sample_ids = payload["sample_id"].astype(np.int64)
            labels = payload["label"].astype(np.int64)
            channel_names = [str(name) for name in payload["channel_names"].tolist()]
    except OSError as exc:
        LOGGER.error("Failed to read Pipeline A tensor at %s.", path)
        raise OSError(f"Failed to read Pipeline A tensor at {path}.") from exc
    LOGGER.info("Loaded Pipeline A tensor with shape=%s and channel_names=%s.", x.shape, channel_names)
    return x, sample_ids, labels, channel_names


def _load_pipeline_b_frame(path: Path) -> pd.DataFrame:
    """Load the full aligned Pipeline B feature table."""
    try:
        frame = pd.read_csv(path)
    except OSError as exc:
        LOGGER.error("Failed to read Pipeline B features at %s.", path)
        raise OSError(f"Failed to read Pipeline B features at {path}.") from exc
    ordered = frame.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    LOGGER.info(
        "Loaded Pipeline B table with shape=%s, first_feature_columns=%s.",
        ordered.shape,
        ordered.columns[:12].tolist(),
    )
    return ordered


def _load_test_indices(path: Path) -> np.ndarray:
    """Load the locked test-set positional indices."""
    try:
        with np.load(path, allow_pickle=False) as payload:
            test_idx = payload["test_idx"].astype(np.int64)
    except OSError as exc:
        LOGGER.error("Failed to read test indices at %s.", path)
        raise OSError(f"Failed to read test indices at {path}.") from exc
    LOGGER.info("Loaded test indices with n_test=%s.", len(test_idx))
    return test_idx


def _assert_pipeline_alignment(
    pipeline_a_sample_ids: np.ndarray,
    pipeline_a_labels: np.ndarray,
    pipeline_b_frame: pd.DataFrame,
) -> list[str]:
    """Assert that Pipeline A and B stay aligned after sorting by sample_id."""
    pipeline_b_sample_ids = pipeline_b_frame["sample_id"].to_numpy(dtype=np.int64)
    pipeline_b_labels = pipeline_b_frame["label"].to_numpy(dtype=np.int64)
    if not np.array_equal(pipeline_a_sample_ids, pipeline_b_sample_ids):
        raise AssertionError("Pipeline A and Pipeline B sample_id arrays are not aligned.")
    if not np.array_equal(pipeline_a_labels, pipeline_b_labels):
        raise AssertionError("Pipeline A and Pipeline B labels are not aligned.")
    feature_names = [column for column in pipeline_b_frame.columns.tolist() if column not in {"sample_id", "label"}]
    LOGGER.info("Pipeline A / B alignment passed for n_samples=%s.", len(pipeline_a_sample_ids))
    return feature_names


def _safe_kendall_tau(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    """Return a finite Kendall tau statistic and p-value."""
    result = kendalltau(left, right)
    tau = float(result.statistic) if result.statistic is not None else float("nan")
    p_value = float(result.pvalue) if result.pvalue is not None else float("nan")
    if np.isnan(tau):
        return 0.0, 1.0
    if np.isnan(p_value):
        p_value = 1.0
    return tau, p_value


def _compute_kendall_tau_summary(
    left: np.ndarray,
    right: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    """Compute Kendall tau, p-value, and a bootstrap percentile confidence interval."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape:
        raise ValueError("left and right vectors must have identical shape.")
    if left.ndim != 1:
        raise ValueError("Kendall tau vectors must be one-dimensional.")
    tau, p_value = _safe_kendall_tau(left, right)
    rng = np.random.default_rng(seed)
    bootstrap_values = np.empty(n_boot, dtype=float)
    sample_indices = np.arange(len(left), dtype=int)
    for bootstrap_idx in range(n_boot):
        resample = rng.choice(sample_indices, size=len(sample_indices), replace=True)
        bootstrap_values[bootstrap_idx] = _safe_kendall_tau(left[resample], right[resample])[0]
    if not np.isfinite(bootstrap_values).all():
        raise ValueError("Bootstrap Kendall tau values contain non-finite entries.")
    ci_low, ci_high = np.percentile(bootstrap_values, [2.5, 97.5])
    return {
        "kendall_tau": tau,
        "p_value": p_value,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def _build_channel_feature_map(feature_names: Sequence[str]) -> dict[str, list[int]]:
    """Map a 99-feature Pipeline B schema onto the 11 gas channels."""
    channel_map = {channel_name: [] for channel_name in CHANNEL_ORDER}
    for feature_index, feature_name in enumerate(feature_names):
        suffix = str(feature_name).rsplit("_", 1)[-1]
        if suffix not in channel_map:
            raise ValueError(f"Unable to parse channel suffix from feature '{feature_name}'.")
        channel_map[suffix].append(feature_index)
    counts = {channel_name: len(indices) for channel_name, indices in channel_map.items()}
    if set(counts.values()) != {PIPELINE_B_CHANNEL_FEATURE_COUNT}:
        raise ValueError(f"Expected 9 features per channel, found counts={counts}.")
    return channel_map


def _aggregate_feature_vector_to_channels(
    feature_values: np.ndarray,
    feature_names: Sequence[str],
    *,
    absolute: bool,
) -> np.ndarray:
    """Aggregate a 99-feature vector into one 11-channel vector."""
    vector = np.asarray(feature_values, dtype=float)
    if vector.ndim != 1:
        raise ValueError("feature_values must be one-dimensional.")
    channel_map = _build_channel_feature_map(feature_names)
    working = np.abs(vector) if absolute else vector.copy()
    return np.asarray(
        [working[channel_map[channel_name]].sum(dtype=float) for channel_name in CHANNEL_ORDER],
        dtype=float,
    )


def _aggregate_feature_matrix_to_channels(
    feature_matrix: np.ndarray,
    feature_names: Sequence[str],
    *,
    absolute: bool,
) -> np.ndarray:
    """Aggregate a `[N, 99]` matrix into the mean 11-channel vector."""
    matrix = np.asarray(feature_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("feature_matrix must be two-dimensional.")
    channel_map = _build_channel_feature_map(feature_names)
    working = np.abs(matrix) if absolute else matrix.copy()
    return np.asarray(
        [
            working[:, channel_map[channel_name]].sum(axis=1, dtype=float).mean(dtype=float)
            for channel_name in CHANNEL_ORDER
        ],
        dtype=float,
    )


def _clip_negative_importance(values: np.ndarray, *, label: str) -> np.ndarray:
    """Clamp permutation noise below zero to preserve non-negative importances."""
    clipped = np.asarray(values, dtype=float).copy()
    negative_mask = clipped < 0.0
    if negative_mask.any():
        LOGGER.warning(
            "Clipped %s negative importance values to zero for indices=%s.",
            label,
            np.flatnonzero(negative_mask).tolist(),
        )
        clipped[negative_mask] = 0.0
    return clipped


def _normalize_importance_vector(values: np.ndarray) -> np.ndarray:
    """L1-normalize an 11-channel importance vector."""
    vector = np.asarray(values, dtype=float)
    total = float(vector.sum())
    if total <= 0.0:
        LOGGER.warning("Importance vector sum is non-positive; falling back to uniform normalization.")
        return np.full(len(vector), 1.0 / len(vector), dtype=float)
    return vector / total


def _permute_channel_across_samples(x: np.ndarray, *, channel_index: int, permutation: np.ndarray) -> np.ndarray:
    """Permute one raw-signal channel across samples while leaving others unchanged."""
    permuted = np.asarray(x, dtype=float).copy()
    permuted[:, :, channel_index] = permuted[permutation, :, channel_index]
    return permuted


def _permute_channel_within_sample(signal: np.ndarray, *, channel_index: int, permutation: np.ndarray) -> np.ndarray:
    """Permute one raw-signal channel within a single sample across time."""
    permuted = np.asarray(signal, dtype=float).copy()
    permuted[:, channel_index] = signal[permutation, channel_index]
    return permuted


def _predict_minirocket_probabilities(payload: dict[str, Any], x_eval: np.ndarray) -> np.ndarray:
    """Predict positive-class probabilities for the final MiniRocket+Ridge model."""
    transformer = payload["transformer"]
    normalizer = payload["normalizer"]
    model = payload["model"]
    x_eval_ncl = np.transpose(x_eval, (0, 2, 1)).astype(np.float32, copy=False)
    feature_columns = list(getattr(normalizer, "feature_names", [])) or None
    transformed = transformer.transform(x_eval_ncl)
    feature_frame = _coerce_feature_frame(transformed, columns=feature_columns)
    normalized = normalizer.transform(feature_frame)
    return model.predict_proba(normalized)[:, 1].astype(float, copy=False)


def _extract_xgboost_importance(payload: dict[str, Any], x_test_pipeline_b: pd.DataFrame) -> np.ndarray:
    """Compute the 11-channel TreeSHAP aggregate for XGBoost."""
    import shap

    x_normalized = payload["normalizer"].transform(x_test_pipeline_b)
    explainer = shap.TreeExplainer(payload["model"])
    shap_values = explainer.shap_values(x_normalized)
    shap_array = np.asarray(shap_values, dtype=float)
    if shap_array.ndim == 3:
        shap_array = shap_array[..., -1]
    if shap_array.ndim != 2:
        raise ValueError(f"Expected TreeSHAP output shape [N, 99], got {shap_array.shape}.")
    return _aggregate_feature_matrix_to_channels(
        shap_array,
        getattr(payload["normalizer"], "feature_names", x_normalized.columns.tolist()),
        absolute=True,
    )


def _extract_elastic_net_importance(payload: dict[str, Any]) -> np.ndarray:
    """Compute the 11-channel absolute-coefficient aggregate for elastic-net LR."""
    coefficients = np.asarray(payload["model"].coef_[0], dtype=float)
    return _aggregate_feature_vector_to_channels(
        coefficients,
        getattr(payload["normalizer"], "feature_names", payload["feature_names"]),
        absolute=True,
    )


def _extract_svm_importance(
    payload: dict[str, Any],
    x_test_pipeline_b: pd.DataFrame,
    y_test: np.ndarray,
    *,
    n_repeats: int,
    seed: int,
) -> np.ndarray:
    """Compute the 11-channel permutation-importance aggregate for the SVM."""
    x_normalized = payload["normalizer"].transform(x_test_pipeline_b)
    result = permutation_importance(
        estimator=payload["model"],
        X=x_normalized,
        y=np.asarray(y_test, dtype=int),
        n_repeats=n_repeats,
        scoring="roc_auc",
        random_state=seed,
    )
    raw = _aggregate_feature_vector_to_channels(
        np.asarray(result.importances_mean, dtype=float),
        getattr(payload["normalizer"], "feature_names", x_normalized.columns.tolist()),
        absolute=False,
    )
    return _clip_negative_importance(raw, label="svm")


def _extract_minirocket_ridge_importance(
    payload: dict[str, Any],
    x_test_pipeline_a: np.ndarray,
    y_test: np.ndarray,
    *,
    n_repeats: int = PERMUTATION_REPEATS,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Compute raw-signal channel permutation importance for MiniRocket+Ridge."""
    baseline_probabilities = _predict_minirocket_probabilities(payload, x_test_pipeline_a)
    baseline_auc = float(roc_auc_score(np.asarray(y_test, dtype=int), baseline_probabilities))
    rng = np.random.default_rng(seed)
    raw = np.zeros(len(CHANNEL_ORDER), dtype=float)
    for channel_index in range(len(CHANNEL_ORDER)):
        drops: list[float] = []
        for _ in range(n_repeats):
            permutation = rng.permutation(len(x_test_pipeline_a))
            permuted = _permute_channel_across_samples(
                x_test_pipeline_a,
                channel_index=channel_index,
                permutation=permutation,
            )
            permuted_probabilities = _predict_minirocket_probabilities(payload, permuted)
            permuted_auc = float(roc_auc_score(np.asarray(y_test, dtype=int), permuted_probabilities))
            drops.append(max(0.0, baseline_auc - permuted_auc))
        raw[channel_index] = float(np.mean(drops))
    return _clip_negative_importance(raw, label="minirocket_ridge")


def _prepare_dl_tensor(payload: dict[str, Any], x_eval: np.ndarray) -> np.ndarray:
    """Normalize Pipeline A data and transpose it to `[N, C, L]`."""
    normalized = payload["normalizer"].transform(x_eval)
    return np.transpose(normalized, (0, 2, 1)).astype(np.float32, copy=False)


def _attribute_with_optional_cudnn_guard(
    ig: Any,
    batch: torch.Tensor,
    *,
    target: int,
    n_steps: int,
    disable_cudnn: bool,
) -> torch.Tensor:
    """Run one Captum attribution call with an optional CuDNN-disabled guard."""
    if disable_cudnn:
        with torch.backends.cudnn.flags(enabled=False):
            return ig.attribute(batch, target=target, n_steps=n_steps)
    return ig.attribute(batch, target=target, n_steps=n_steps)


def _extract_dl_ig_importance(
    payload: dict[str, Any],
    x_test_pipeline_a: np.ndarray,
    *,
    batch_size: int,
    n_steps: int,
    disable_cudnn: bool = False,
) -> tuple[np.ndarray, float | None]:
    """Compute an 11-channel Integrated Gradients importance vector."""
    from captum.attr import IntegratedGradients

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lightning_model = payload["model"].lightning_model.to(device)
    lightning_model.eval()
    x_test_ncl = _prepare_dl_tensor(payload, x_test_pipeline_a)
    ig = IntegratedGradients(lightning_model)
    channel_sums = np.zeros(len(CHANNEL_ORDER), dtype=float)
    peak_memory_mb: float | None = None
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    if disable_cudnn and device.type == "cuda":
        LOGGER.info("Running Integrated Gradients with CuDNN disabled for RNN backward compatibility.")
    for start in range(0, len(x_test_ncl), batch_size):
        stop = min(start + batch_size, len(x_test_ncl))
        batch = torch.from_numpy(x_test_ncl[start:stop]).to(device)
        batch.requires_grad_(True)
        attributions = _attribute_with_optional_cudnn_guard(
            ig,
            batch,
            target=1,
            n_steps=n_steps,
            disable_cudnn=disable_cudnn and device.type == "cuda",
        )
        attr_numpy = attributions.detach().cpu().numpy()
        if not np.isfinite(attr_numpy).all():
            raise ValueError("Integrated Gradients returned NaN or Inf values.")
        channel_sums += np.abs(attr_numpy).sum(axis=2).sum(axis=0)
    if device.type == "cuda":
        peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0**2))
    lightning_model.to("cpu")
    return channel_sums / len(x_test_ncl), peak_memory_mb


def _extract_inception_time_importance(payload: dict[str, Any], x_test_pipeline_a: np.ndarray) -> tuple[np.ndarray, float | None]:
    """Compute Integrated Gradients channel importance for InceptionTime."""
    return _extract_dl_ig_importance(
        payload,
        x_test_pipeline_a,
        batch_size=IG_BATCH_SIZE_GPU if torch.cuda.is_available() else IG_BATCH_SIZE_CPU,
        n_steps=IG_STEPS,
    )


def _extract_cbla_importance(payload: dict[str, Any], x_test_pipeline_a: np.ndarray) -> tuple[np.ndarray, float | None]:
    """Compute Integrated Gradients channel importance for CNN-BiLSTM-Attention."""
    return _extract_dl_ig_importance(
        payload,
        x_test_pipeline_a,
        batch_size=IG_BATCH_SIZE_GPU if torch.cuda.is_available() else IG_BATCH_SIZE_CPU,
        n_steps=IG_STEPS,
        disable_cudnn=True,
    )


def _top_k_channels(importance_vector: np.ndarray, *, k: int) -> list[str]:
    """Return the channel names in descending importance order for the top-K set."""
    order = np.argsort(-np.asarray(importance_vector, dtype=float), kind="mergesort")[:k]
    return [CHANNEL_ORDER[index] for index in order.tolist()]


def _compute_top_k_converging_rows(
    importance_map: dict[str, np.ndarray],
    *,
    top_k_values: Iterable[int] = (3, 5, 7),
    model_thresholds: Iterable[int] = (3, 4, 5, 6),
) -> list[dict[str, Any]]:
    """Compute converging-channel rows for the locked K and majority thresholds."""
    rows: list[dict[str, Any]] = []
    for top_k in top_k_values:
        counter: Counter[str] = Counter()
        for importance_vector in importance_map.values():
            counter.update(_top_k_channels(importance_vector, k=top_k))
        for threshold in model_thresholds:
            channels = [
                channel_name
                for channel_name in CHANNEL_ORDER
                if counter.get(channel_name, 0) >= int(threshold)
            ]
            rows.append(
                {
                    "K": int(top_k),
                    "M_models_threshold": int(threshold),
                    "channels": channels,
                    "channel_count": len(channels),
                }
            )
    return rows


def _save_channel_importance_csv(
    output_path: Path,
    *,
    raw_values: np.ndarray,
    normalized_values: np.ndarray,
) -> None:
    """Write one model-specific channel-importance CSV."""
    dataframe = pd.DataFrame(
        {
            "channel_index": np.arange(1, len(CHANNEL_ORDER) + 1, dtype=np.int64),
            "channel_name": CHANNEL_ORDER,
            "importance_raw": np.asarray(raw_values, dtype=float),
            "importance_normalized": np.asarray(normalized_values, dtype=float),
        }
    )
    _write_dataframe_csv(output_path, dataframe)
    LOGGER.info("Wrote channel-importance CSV to %s.", _safe_format_path(output_path))


def _compute_single_sample_minirocket_importance(
    payload: dict[str, Any],
    signal: np.ndarray,
    *,
    n_repeats: int,
    seed: int,
) -> np.ndarray:
    """Compute one representative-sample channel-importance row for MiniRocket+Ridge."""
    baseline_probability = float(_predict_minirocket_probabilities(payload, signal[np.newaxis, ...])[0])
    rng = np.random.default_rng(seed)
    raw = np.zeros(len(CHANNEL_ORDER), dtype=float)
    for channel_index in range(len(CHANNEL_ORDER)):
        drops: list[float] = []
        for _ in range(n_repeats):
            permutation = rng.permutation(signal.shape[0])
            permuted_signal = _permute_channel_within_sample(
                signal,
                channel_index=channel_index,
                permutation=permutation,
            )
            permuted_probability = float(_predict_minirocket_probabilities(payload, permuted_signal[np.newaxis, ...])[0])
            drops.append(max(0.0, baseline_probability - permuted_probability))
        raw[channel_index] = float(np.mean(drops))
    return raw


def _save_representative_samples_shap_npz(
    *,
    reference_signal_path: Path,
    output_path: Path,
    shap_values: np.ndarray,
) -> None:
    """Write the Session 14 representative-samples SHAP NPZ aligned to the signal NPZ."""
    with np.load(reference_signal_path, allow_pickle=True) as reference_payload:
        sample_ids = reference_payload["sample_id"].astype(np.int64)
        quadrants = reference_payload["quadrant"].astype(str)
    shap_array = np.asarray(shap_values, dtype=float)
    if shap_array.shape != (len(sample_ids), len(CHANNEL_ORDER)):
        raise ValueError(
            f"Expected SHAP array shape {(len(sample_ids), len(CHANNEL_ORDER))}, got {shap_array.shape}."
        )
    normalized = np.vstack([_normalize_importance_vector(row) for row in shap_array])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        sample_id=sample_ids,
        quadrant=quadrants.astype("U8"),
        shap=normalized.astype(np.float32, copy=False),
    )
    LOGGER.info("Wrote representative-samples SHAP NPZ to %s.", _safe_format_path(output_path))


def _validate_representative_sample_order(reference_signal_path: Path, reference_csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate the locked representative-sample ordering contract."""
    with np.load(reference_signal_path, allow_pickle=True) as reference_payload:
        sample_ids = reference_payload["sample_id"].astype(np.int64)
        quadrants = reference_payload["quadrant"].astype(str)
        signals = np.asarray(reference_payload["signal"], dtype=float)
    csv_frame = pd.read_csv(reference_csv_path)
    csv_sample_ids = csv_frame["sample_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(sample_ids, csv_sample_ids):
        raise AssertionError("Representative sample_id ordering diverges between the Session 12 CSV and NPZ.")
    LOGGER.info("Representative sample ordering matches Session 12 CSV and signal NPZ 1:1.")
    return sample_ids, quadrants, signals


def _render_top_three_table(model_name: str, normalized_values: np.ndarray) -> list[str]:
    """Render the markdown rows for one model's top-3 channels."""
    top_channels = _top_k_channels(normalized_values, k=3)
    lines = [
        f"### `{model_name}`",
        "",
        "| channel | normalized_weight |",
        "| --- | --- |",
    ]
    for channel_name in top_channels:
        channel_index = CHANNEL_ORDER.index(channel_name)
        lines.append(f"| {channel_name} | {normalized_values[channel_index]:.6f} |")
    lines.append("")
    return lines


def _aggregation_policy_markdown_table() -> str:
    """Return the locked aggregation-policy table for the markdown footer."""
    return "\n".join(
        [
            "| Model | Raw attribution | Time aggregation | Feature aggregation | Final |",
            "| --- | --- | --- | --- | --- |",
            "| `xgboost` | TreeSHAP `[195, 99]` on test set | n/a | `sum(|SHAP|)` over 9 features per channel, then mean over 195 samples | `[11]` |",
            "| `elastic_net_lr` | `|coef|` `[99]` from PKL `model.coef_[0]` | n/a | `sum(|coef|)` over 9 features per channel | `[11]` |",
            "| `svm` | permutation importance `[99]` on test set with n_repeats=10, scoring='roc_auc' | n/a | `sum(perm_imp)` over 9 features per channel | `[11]` |",
            "| `minirocket_ridge` | permutation importance per raw signal channel directly on test set with n_repeats=10, scoring='roc_auc' | n/a | direct | `[11]` |",
            "| `inception_time` | Captum IG on Pipeline A test set `[195, 11, 960]`, n_steps=50, target=1 | `sum(|IG|)` over 960 timepoints per sample | mean over 195 samples | `[11]` |",
            "| `cnn_bilstm_attention` | Same as InceptionTime | Same | Same | `[11]` |",
        ]
    )


def _build_markdown_report(
    *,
    importance_map: dict[str, np.ndarray],
    kendall_frame: pd.DataFrame,
    top_k_frame: pd.DataFrame,
    wall_time_minutes: dict[str, float],
    cbla_attention_path: Path,
    git_commit_hash: str,
    timestamp_utc: str,
) -> str:
    """Build the Session 14 markdown summary."""
    lines = [
        "# SHAP Cross-Model Consistency",
        "",
        "## Methodology",
        "",
        "- `xgboost`: TreeSHAP on the normalized 99-feature test matrix, aggregated to channels by summing absolute SHAP across the 9 features mapped to each channel.",
        "- `elastic_net_lr`: absolute normalized-input coefficients aggregated from 99 features to 11 channels.",
        "- `svm`: normalized-input permutation importance with `n_repeats=10` and `scoring='roc_auc'`, aggregated from 99 features to 11 channels.",
        "- `minirocket_ridge`: raw-signal channel permutation importance with `n_repeats=10`, recomputing the MiniRocket transform and calibrated ridge probabilities after each channel permutation.",
        "- `inception_time`: Captum Integrated Gradients on the positive-class logit (`target=1`) with `n_steps=50`, summed over 960 time points and averaged across the 195-sample test set.",
        "- `cnn_bilstm_attention`: same Integrated Gradients protocol as InceptionTime; attention weights remain a separate architectural sanity artifact and are excluded from the Kendall tau matrix.",
        "",
        "## Top-3 Channels Per Model",
        "",
    ]
    for model_name in MODEL_NAMES:
        lines.extend(_render_top_three_table(model_name, importance_map[model_name]))
    lines.extend(
        [
            "## Kendall Tau Matrix",
            "",
            "| model_a | model_b | kendall_tau | p_value | ci_low | ci_high |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in kendall_frame.itertuples(index=False):
        lines.append(
            f"| {row.model_a} | {row.model_b} | {row.kendall_tau:.6f} | {row.p_value:.6f} | {row.ci_low:.6f} | {row.ci_high:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Top-K Converging Channels",
            "",
            "| K | M_models_threshold | channels | channel_count |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in top_k_frame.itertuples(index=False):
        channel_text = ", ".join(row.channels) if isinstance(row.channels, list) else str(row.channels)
        lines.append(f"| {row.K} | {row.M_models_threshold} | {channel_text} | {row.channel_count} |")
    lines.extend(
        [
            "",
            "## Implementation Note",
            "",
            "The DL final PKLs depend on the private `_predict_probabilities` functions in `src/training/train_inception_time.py` and `src/training/train_cnn_bilstm_attention.py`, as locked by Erratum E010.",
            "",
            "## Bootstrap Convention",
            "",
            f"- {BOOTSTRAP_CONVENTION_TEXT}",
            "- Kendall tau bootstrap resamples the 11 channel pairs with the same single-RNG sequential convention documented by Erratum E011.",
            "",
            "## CBLA Attention Weights",
            "",
            f"- Reference artifact: `{_format_path(cbla_attention_path)}`",
            "- The attention artifact is referenced for architectural sanity only and is excluded from the cross-model Kendall tau matrix, consistent with plan §2.21 and the Jain & Wallace 2019 / Wiegreffe & Pinter 2019 caution.",
            "",
            "## Wall Time",
            "",
            f"- xgboost_treeshap: {wall_time_minutes['xgboost_treeshap']:.3f} minutes",
            f"- svm_permutation: {wall_time_minutes['svm_permutation']:.3f} minutes",
            f"- minirocket_ridge_permutation_on_raw_signal: {wall_time_minutes['minirocket_ridge_permutation_on_raw_signal']:.3f} minutes",
            f"- inception_time_ig: {wall_time_minutes['inception_time_ig']:.3f} minutes",
            f"- cnn_bilstm_attention_ig: {wall_time_minutes['cnn_bilstm_attention_ig']:.3f} minutes",
            f"- total: {wall_time_minutes['total']:.3f} minutes",
            "",
            "## Locked Aggregation Policy",
            "",
            _aggregation_policy_markdown_table(),
            "",
            f"Footer: git_commit_hash={git_commit_hash}; timestamp_utc={timestamp_utc}",
            "",
        ]
    )
    return "\n".join(lines)


def _paths_for_outputs() -> dict[str, Path]:
    """Return the Stage 8 output paths."""
    outputs_shap = _resolve_path("outputs/shap")
    outputs_analysis = _resolve_path("outputs/analysis")
    return {
        "kendall": outputs_shap / "cross_model_kendall_tau.csv",
        "top_k": outputs_shap / "cross_model_top_k_converging.csv",
        "report": outputs_analysis / "shap_cross_model_consistency.md",
        "audit": outputs_analysis / "stage8_audit.json",
        "representative_shap": outputs_shap / "binary_minirocket_ridge_representative_samples_shap.npz",
    }


def run_shap_cross_model(config_path: Path) -> None:
    """Run the full Stage 8 channel-importance consistency analysis."""
    config = _load_global_config(config_path=config_path)
    torch.set_float32_matmul_precision("high")
    git_commit_hash = _get_git_commit_hash()
    timestamp_utc = _timestamp_utc()
    pipeline_a_path = PROJECT_ROOT / "data" / "processed" / "sequences" / "pipeline_a_raw.npz"
    pipeline_b_path = PROJECT_ROOT / "data" / "processed" / "features" / "pipeline_b_raw.csv"
    test_idx_path = PROJECT_ROOT / "data" / "processed" / "splits" / "test_idx.npz"
    x_pipeline_a, sample_ids, labels, channel_names = _load_pipeline_a_bundle(pipeline_a_path)
    pipeline_b = _load_pipeline_b_frame(pipeline_b_path)
    pipeline_b_feature_names = _assert_pipeline_alignment(sample_ids, labels, pipeline_b)
    if channel_names != CHANNEL_ORDER:
        raise AssertionError(f"Unexpected Pipeline A channel order: {channel_names}")
    test_idx = _load_test_indices(test_idx_path)
    x_test_pipeline_a = x_pipeline_a[test_idx]
    y_test = labels[test_idx]
    sample_ids_test = sample_ids[test_idx]
    x_test_pipeline_b = pipeline_b.iloc[test_idx].reset_index(drop=True).loc[:, pipeline_b_feature_names]
    cbla_attention_path = CBLA_ATTENTION_NPZ
    if not cbla_attention_path.exists():
        raise FileNotFoundError(f"Expected CBLA attention artifact at {cbla_attention_path}.")
    LOGGER.info("Using CBLA attention reference artifact at %s.", _safe_format_path(cbla_attention_path))
    for model_name in MODEL_NAMES:
        prediction_frame = _load_view_prediction_frame(model_name=model_name, view_name="test")
        prediction_sample_ids = prediction_frame["sample_id"].to_numpy(dtype=np.int64)
        if not np.array_equal(prediction_sample_ids, sample_ids_test):
            raise AssertionError(f"Test prediction sample_id ordering mismatch for model={model_name}.")
    payloads = {
        model_name: load_final_model(PROJECT_ROOT / "outputs" / "models" / f"binary_{model_name}_final.pkl")
        for model_name in MODEL_NAMES
    }
    outputs = _paths_for_outputs()
    raw_importance: dict[str, np.ndarray] = {}
    normalized_importance: dict[str, np.ndarray] = {}
    wall_time_minutes = {
        "xgboost_treeshap": 0.0,
        "svm_permutation": 0.0,
        "minirocket_ridge_permutation_on_raw_signal": 0.0,
        "inception_time_ig": 0.0,
        "cnn_bilstm_attention_ig": 0.0,
        "total": 0.0,
    }
    total_start = perf_counter()
    extraction_plan = (
        ("xgboost", lambda: _extract_xgboost_importance(payloads["xgboost"], x_test_pipeline_b), "xgboost_treeshap"),
        ("elastic_net_lr", lambda: _extract_elastic_net_importance(payloads["elastic_net_lr"]), None),
        (
            "svm",
            lambda: _extract_svm_importance(
                payloads["svm"],
                x_test_pipeline_b,
                y_test,
                n_repeats=PERMUTATION_REPEATS,
                seed=RANDOM_SEED,
            ),
            "svm_permutation",
        ),
        (
            "minirocket_ridge",
            lambda: _extract_minirocket_ridge_importance(
                payloads["minirocket_ridge"],
                x_test_pipeline_a,
                y_test,
                n_repeats=PERMUTATION_REPEATS,
                seed=RANDOM_SEED,
            ),
            "minirocket_ridge_permutation_on_raw_signal",
        ),
        (
            "inception_time",
            lambda: _extract_inception_time_importance(payloads["inception_time"], x_test_pipeline_a),
            "inception_time_ig",
        ),
        (
            "cnn_bilstm_attention",
            lambda: _extract_cbla_importance(payloads["cnn_bilstm_attention"], x_test_pipeline_a),
            "cnn_bilstm_attention_ig",
        ),
    )
    for model_name, extractor, wall_time_key in extraction_plan:
        LOGGER.info("Starting channel-importance extraction for model=%s.", model_name)
        start = perf_counter()
        extracted = extractor()
        elapsed_minutes = (perf_counter() - start) / 60.0
        if isinstance(extracted, tuple):
            raw_vector = np.asarray(extracted[0], dtype=float)
            peak_memory_mb = extracted[1]
            if peak_memory_mb is not None:
                LOGGER.info("Integrated Gradients peak GPU memory for model=%s: %.3f MB.", model_name, peak_memory_mb)
        else:
            raw_vector = np.asarray(extracted, dtype=float)
        if wall_time_key is not None:
            wall_time_minutes[wall_time_key] = elapsed_minutes
        raw_importance[model_name] = raw_vector
        normalized_importance[model_name] = _normalize_importance_vector(raw_vector)
        csv_path = _resolve_path("outputs/shap") / f"binary_{model_name}_channel_importance.csv"
        _save_channel_importance_csv(
            csv_path,
            raw_values=raw_importance[model_name],
            normalized_values=normalized_importance[model_name],
        )
        LOGGER.info("Completed model=%s channel-importance extraction in %.3f minutes.", model_name, elapsed_minutes)
    wall_time_minutes["total"] = (perf_counter() - total_start) / 60.0
    kendall_rows = []
    for model_a, model_b in itertools.combinations(MODEL_NAMES, 2):
        summary = _compute_kendall_tau_summary(
            normalized_importance[model_a],
            normalized_importance[model_b],
            n_boot=int(config["bootstrap"]["n_boot"]),
            seed=RANDOM_SEED,
        )
        kendall_rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "kendall_tau": summary["kendall_tau"],
                "p_value": summary["p_value"],
                "ci_low": summary["ci_low"],
                "ci_high": summary["ci_high"],
            }
        )
    kendall_frame = pd.DataFrame(kendall_rows)
    _write_dataframe_csv(outputs["kendall"], kendall_frame)
    LOGGER.info("Wrote cross-model Kendall tau CSV to %s.", _safe_format_path(outputs["kendall"]))
    top_k_rows = _compute_top_k_converging_rows(normalized_importance)
    top_k_frame = pd.DataFrame(top_k_rows)
    _write_dataframe_csv(outputs["top_k"], top_k_frame)
    LOGGER.info("Wrote top-K converging-channel CSV to %s.", _safe_format_path(outputs["top_k"]))
    representative_sample_ids, representative_quadrants, representative_signals = _validate_representative_sample_order(
        REPRESENTATIVE_SIGNALS_NPZ,
        REPRESENTATIVE_SAMPLE_CSV,
    )
    representative_raw = np.vstack(
        [
            _compute_single_sample_minirocket_importance(
                payloads["minirocket_ridge"],
                representative_signals[sample_index],
                n_repeats=PERMUTATION_REPEATS,
                seed=RANDOM_SEED + sample_index,
            )
            for sample_index in range(len(representative_sample_ids))
        ]
    )
    _save_representative_samples_shap_npz(
        reference_signal_path=REPRESENTATIVE_SIGNALS_NPZ,
        output_path=outputs["representative_shap"],
        shap_values=representative_raw,
    )
    report_text = _build_markdown_report(
        importance_map=normalized_importance,
        kendall_frame=kendall_frame,
        top_k_frame=top_k_frame,
        wall_time_minutes=wall_time_minutes,
        cbla_attention_path=cbla_attention_path,
        git_commit_hash=git_commit_hash,
        timestamp_utc=timestamp_utc,
    )
    _write_text(outputs["report"], report_text)
    LOGGER.info("Wrote markdown report to %s.", _safe_format_path(outputs["report"]))
    max_row = kendall_frame.sort_values("kendall_tau", ascending=False, kind="mergesort").iloc[0]
    min_row = kendall_frame.sort_values("kendall_tau", ascending=True, kind="mergesort").iloc[0]
    top_k_lookup = {
        (int(row["K"]), int(row["M_models_threshold"])): list(row["channels"])
        for row in top_k_rows
    }
    audit_payload = {
        "session": 14,
        "stage": "Stage 8",
        "session_scope": "all_6_models_cross_model_consistency",
        "git_commit_hash": git_commit_hash,
        "timestamp_utc": timestamp_utc,
        "data_split_seed": int(config["data_split_seed"]),
        "bootstrap_n_boot": int(config["bootstrap"]["n_boot"]),
        "random_seed": int(config["random_seed"]),
        "bootstrap_convention": {
            "erratum": "E011",
            "linked_commit": "a2af64e",
            "summary": BOOTSTRAP_CONVENTION_TEXT,
        },
        "captum_version": None,
        "captum_dependency_commit": "82e3516",
        "channel_aggregation_policy": {
            "xgboost": "TreeSHAP on test n=195, sum(|SHAP|) over 9 features per channel, mean over samples, L1-normalized",
            "elastic_net_lr": "abs(coef_) on Pipeline B 99 features, sum over 9 features per channel, L1-normalized",
            "svm": "permutation_importance on Pipeline B 99 features, n_repeats=10, scoring='roc_auc', sum over 9 features per channel, L1-normalized",
            "minirocket_ridge": "permutation importance on raw Pipeline A channels, n_repeats=10, scoring='roc_auc', direct 11-channel output, L1-normalized",
            "inception_time": "Captum IntegratedGradients on Pipeline A test n=195, n_steps=50, target=1, sum(|IG|) over 960 timepoints, mean over samples, L1-normalized",
            "cnn_bilstm_attention": "Captum IntegratedGradients on Pipeline A test n=195, n_steps=50, target=1, sum(|IG|) over 960 timepoints, mean over samples, L1-normalized",
        },
        "deliverables": {
            "channel_importance_csvs": [
                _format_path(_resolve_path("outputs/shap") / "binary_minirocket_ridge_channel_importance.csv"),
                _format_path(_resolve_path("outputs/shap") / "binary_xgboost_channel_importance.csv"),
                _format_path(_resolve_path("outputs/shap") / "binary_svm_channel_importance.csv"),
                _format_path(_resolve_path("outputs/shap") / "binary_elastic_net_lr_channel_importance.csv"),
                _format_path(_resolve_path("outputs/shap") / "binary_inception_time_channel_importance.csv"),
                _format_path(_resolve_path("outputs/shap") / "binary_cnn_bilstm_attention_channel_importance.csv"),
            ],
            "cross_model_kendall_tau_csv": _format_path(outputs["kendall"]),
            "cross_model_top_k_converging_csv": _format_path(outputs["top_k"]),
            "representative_samples_shap_npz": _format_path(outputs["representative_shap"]),
            "consistency_markdown": _format_path(outputs["report"]),
            "cbla_attention_reference_note": _format_path(cbla_attention_path),
        },
        "post_hoc_trigger_references": {
            "A_subgroup_retrain": "Session 11 audit JSON: triggered=false",
            "C_test_vs_cv_shift": "Session 11 audit JSON: triggered=false",
            "F_tumor_marker_underpowered": "Session 13 audit JSON: triggered=false",
            "G_dl_anomaly": "Erratum E005 (Session 8/9 mitigation)",
            "H_calibration_failure": "Session 11 audit JSON: triggered=false",
            "lgin_primary_view": "Session 12 audit JSON: nested_cv_oof per Erratum E012",
            "fusion_template_logic": "Session 13 audit JSON: voc_alone_preferable per Erratum E013",
        },
        "wall_time_minutes": {key: float(value) for key, value in wall_time_minutes.items()},
        "key_findings": {
            "kendall_tau_max_across_15_pairs": {
                "pair": f"{max_row['model_a']}_vs_{max_row['model_b']}",
                "tau": float(max_row["kendall_tau"]),
                "ci_low_high": [float(max_row["ci_low"]), float(max_row["ci_high"])],
            },
            "kendall_tau_min_across_15_pairs": {
                "pair": f"{min_row['model_a']}_vs_{min_row['model_b']}",
                "tau": float(min_row["kendall_tau"]),
                "ci_low_high": [float(min_row["ci_low"]), float(min_row["ci_high"])],
            },
            "kendall_tau_mean_across_15_pairs": float(kendall_frame["kendall_tau"].mean()),
            "n_pairs_significant_at_alpha_0.05": int((kendall_frame["p_value"] < 0.05).sum()),
            "top_3_channels_appearing_in_at_least_4_of_6_models": top_k_lookup[(3, 4)],
            "top_5_channels_appearing_in_at_least_4_of_6_models": top_k_lookup[(5, 4)],
            "top_7_channels_appearing_in_at_least_4_of_6_models": top_k_lookup[(7, 4)],
            "channels_unanimous_top_5": top_k_lookup[(5, 6)],
        },
    }
    try:
        from captum import __version__ as captum_version

        audit_payload["captum_version"] = captum_version
    except ImportError:
        audit_payload["captum_version"] = None
    _write_json(outputs["audit"], audit_payload)
    LOGGER.info("Wrote Stage 8 audit JSON to %s.", _safe_format_path(outputs["audit"]))


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Stage 8 analysis module."""
    parser = argparse.ArgumentParser(description="Run Stage 8 SHAP cross-model consistency analysis.")
    parser.add_argument("--config", required=True, type=Path, help="Path to modeling_global.yaml.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point for Stage 8 SHAP cross-model consistency."""
    _configure_logging()
    args = _parse_args()
    run_shap_cross_model(config_path=args.config)


if __name__ == "__main__":
    main()
