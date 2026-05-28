"""Post-hoc optimism-corrected validation for Plan 1 Session 2.

This module implements the Harrell-style bootstrap optimism correction required
by Plan 1 §3.4 steps 1-9 for the frozen best model `minirocket_ridge`.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score
from sktime.transformations.panel.rocket import MiniRocketMultivariate

from src.analysis.data_split import _format_path, _resolve_path, _write_text
from src.analysis.sample_size import POST_HOC_DECLARATION, load_post_hoc_config
from src.config import CLIP_LENGTH_L, N_GAS_CHANNELS, PROJECT_ROOT, RANDOM_SEED
from src.training.load_final_model import load_final_model


LOGGER = logging.getLogger(__name__)

SAMPLE_INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "sample_index.csv"
SEQUENCE_PATH = PROJECT_ROOT / "data" / "processed" / "sequences" / "pipeline_a_raw.npz"
TRAIN_IDX_PATH = PROJECT_ROOT / "data" / "processed" / "splits" / "train_idx.npz"
FINAL_MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "binary_minirocket_ridge_final.pkl"

FROZEN_OOF_AUC = 0.9246  # [runtime-determined-by-prespecified-algorithm: docs/modeling/validated_modeling_run_summary.md]
FROZEN_OOF_CI_LOW = 0.9048  # [runtime-determined-by-prespecified-algorithm: docs/modeling/validated_modeling_run_summary.md]
FROZEN_OOF_CI_HIGH = 0.9427  # [runtime-determined-by-prespecified-algorithm: docs/modeling/validated_modeling_run_summary.md]
FROZEN_TEST_AUC = 0.9223  # [runtime-determined-by-prespecified-algorithm: docs/modeling/validated_modeling_run_summary.md]
FROZEN_TEST_CI_LOW = 0.8831  # [runtime-determined-by-prespecified-algorithm: docs/modeling/validated_modeling_run_summary.md]
FROZEN_TEST_CI_HIGH = 0.9585  # [runtime-determined-by-prespecified-algorithm: docs/modeling/validated_modeling_run_summary.md]

EXPECTED_FINAL_PKL_KEYS = {
    "transformer",
    "normalizer",
    "model",
    "calibrator",
    "threshold_p_raw",
    "threshold_p_final_equivalent",
    "feature_names",
    "training_sample_ids",
    "metadata",
}
EXPECTED_FINAL_METADATA_KEYS = {
    "model_name",
    "final_hparams",
    "oof_source",
    "git_commit_hash",
    "training_timestamp_utc",
}
SIGMA_CLAMP_THRESHOLD = 1e-8


@dataclass(frozen=True)
class FrozenMiniRocketRidgeConfig:
    """Recovered frozen configuration for the transient Plan 1 §3.4 procedure.

    Plan 1 §3.4 step 2 allows recovery from read-only metadata or by unwrapping
    the fitted frozen objects when metadata is empty.

    Attributes:
        transformer_params: Parameters used to rebuild `MiniRocketMultivariate`.
        ridge_params: Parameters used to rebuild the plain `RidgeClassifier`.
    """

    transformer_params: dict[str, Any]
    ridge_params: dict[str, Any]


@dataclass(frozen=True)
class LockedTrainingData:
    """Aligned locked 777-row training data for Plan 1 §3.4 step 1.

    Attributes:
        x_train: Raw Pipeline A tensor aligned to ascending integer `sample_id`.
        y_train: Binary labels aligned to `x_train`.
        train_sample_ids: Ascending integer `sample_id` values for the 777-row train set.
    """

    x_train: np.ndarray
    y_train: np.ndarray
    train_sample_ids: np.ndarray


@dataclass(frozen=True)
class FeatureMatrixNormalizer:
    """Pipeline B-style z-score normalizer for post-MiniRocket feature matrices.

    Plan 1 §3.4 step 2 requires z-score normalization to be fit on the current
    bootstrap sample and then applied to the original 777-row feature matrix.

    Attributes:
        mean_: Per-feature mean.
        scale_: Per-feature standard deviation after clamping.
        feature_names: Ordered feature names expected on transform.
    """

    mean_: np.ndarray
    scale_: np.ndarray
    feature_names: list[str]

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted z-score transform to one feature matrix.

        Args:
            x: Feature matrix containing all fitted feature names.

        Returns:
            A transformed dataframe with the fitted feature order.

        Raises:
            TypeError: If `x` is not a pandas dataframe.
            ValueError: If required features are missing.
        """
        if not isinstance(x, pd.DataFrame):
            raise TypeError("x must be a pandas.DataFrame.")
        missing_features = [feature for feature in self.feature_names if feature not in x.columns]
        if missing_features:
            raise ValueError(f"Missing features for post-MiniRocket normalization: {missing_features}.")
        aligned = x.loc[:, self.feature_names].copy()
        transformed = (aligned.to_numpy(dtype=float) - self.mean_) / self.scale_
        return pd.DataFrame(transformed, columns=self.feature_names, index=aligned.index)


@dataclass(frozen=True)
class OptimismCorrectionResult:
    """Structured result for Plan 1 §3.4 steps 5-7.

    Attributes:
        transformer_params: Recovered MiniRocket parameters.
        ridge_params: Recovered Ridge parameters.
        auc_bootstrap_apparent: Bootstrap-sample apparent AUC values.
        auc_original_from_bootstrap_model: Original-777 AUC values from bootstrap-fitted models.
        optimism_b: Per-bootstrap optimism values.
        auc_full_apparent: Apparent AUC on the full 777 training samples.
        mean_optimism: Mean bootstrap optimism.
        auc_optimism_corrected: Full-sample apparent AUC minus mean optimism.
        corrected_ci_low: 2.5 percentile of corrected bootstrap values.
        corrected_ci_high: 97.5 percentile of corrected bootstrap values.
        transform_fit_count: Diagnostic count for the one-time MiniRocket fit.
    """

    transformer_params: dict[str, Any]
    ridge_params: dict[str, Any]
    auc_bootstrap_apparent: np.ndarray
    auc_original_from_bootstrap_model: np.ndarray
    optimism_b: np.ndarray
    auc_full_apparent: float
    mean_optimism: float
    auc_optimism_corrected: float
    corrected_ci_low: float
    corrected_ci_high: float
    transform_fit_count: int

    def to_csv_frame(self) -> pd.DataFrame:
        """Render deterministic per-bootstrap and summary rows for the CSV output."""
        bootstrap_frame = pd.DataFrame(
            {
                "record_type": "bootstrap",
                "bootstrap_index": np.arange(len(self.optimism_b), dtype=int),
                "auc_bootstrap_apparent": self.auc_bootstrap_apparent,
                "auc_original_from_bootstrap_model": self.auc_original_from_bootstrap_model,
                "optimism_b": self.optimism_b,
                "auc_full_apparent": np.nan,
                "mean_optimism": np.nan,
                "auc_optimism_corrected": np.nan,
                "corrected_ci_low": np.nan,
                "corrected_ci_high": np.nan,
            }
        )
        summary_frame = pd.DataFrame(
            [
                {
                    "record_type": "summary",
                    "bootstrap_index": -1,
                    "auc_bootstrap_apparent": np.nan,
                    "auc_original_from_bootstrap_model": np.nan,
                    "optimism_b": np.nan,
                    "auc_full_apparent": self.auc_full_apparent,
                    "mean_optimism": self.mean_optimism,
                    "auc_optimism_corrected": self.auc_optimism_corrected,
                    "corrected_ci_low": self.corrected_ci_low,
                    "corrected_ci_high": self.corrected_ci_high,
                }
            ]
        )
        return pd.concat([bootstrap_frame, summary_frame], ignore_index=True)


def _timestamp_utc() -> str:
    """Return an ISO-8601 UTC timestamp with trailing `Z`."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _get_git_commit_hash() -> str:
    """Return the current full git commit hash."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _coerce_feature_frame(values: Any, columns: list[str] | None = None) -> pd.DataFrame:
    """Convert one MiniRocket feature output into a dataframe."""
    if isinstance(values, pd.DataFrame):
        frame = values.copy()
        if columns is not None:
            frame.columns = columns
        return frame
    array = np.asarray(values, dtype=float)
    feature_columns = columns or [f"kernel_{index:05d}" for index in range(array.shape[1])]
    return pd.DataFrame(array, columns=feature_columns)


def _clamp_feature_scales(scale: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """Clamp near-zero feature standard deviations to `1.0`."""
    clamped = scale.copy()
    for index, value in enumerate(clamped):
        if value < SIGMA_CLAMP_THRESHOLD:
            clamped[index] = 1.0
            LOGGER.warning(
                "Clamped post-MiniRocket feature scale below %.1e for feature '%s' at index %s.",
                SIGMA_CLAMP_THRESHOLD,
                feature_names[index],
                index,
            )
    return clamped


def _fit_feature_matrix_normalizer(x_train: pd.DataFrame) -> FeatureMatrixNormalizer:
    """Fit the Plan 1 §3.4 step-2 z-score normalizer on one feature matrix."""
    feature_names = x_train.columns.tolist()
    mean_ = x_train.mean(axis=0).to_numpy(dtype=float)
    scale_ = x_train.std(axis=0, ddof=0).to_numpy(dtype=float)
    return FeatureMatrixNormalizer(
        mean_=mean_,
        scale_=_clamp_feature_scales(scale=scale_, feature_names=feature_names),
        feature_names=feature_names,
    )


def _compute_auc_roc(y_true: np.ndarray, decision_scores: np.ndarray) -> float:
    """Compute AUC-ROC from binary labels and continuous decision scores."""
    y_array = np.asarray(y_true, dtype=np.int64)
    score_array = np.asarray(decision_scores, dtype=float)
    if y_array.ndim != 1 or score_array.ndim != 1:
        raise ValueError("AUC inputs must be one-dimensional arrays.")
    if len(y_array) != len(score_array):
        raise ValueError("AUC inputs must have the same length.")
    if len(np.unique(y_array)) != 2:
        raise ValueError("AUC-ROC requires both classes to be present.")
    if not np.isfinite(score_array).all():
        raise ValueError("AUC decision scores must be finite.")
    return float(roc_auc_score(y_array, score_array))


def _draw_stratified_bootstrap_indices(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw one stratified bootstrap sample with both classes preserved.

    Plan 1 §3.4 step 4 uses the single-RNG sequential convention from Erratum
    E011 and applies stratification by `y_true`.
    """
    sampled_indices: list[np.ndarray] = []
    for class_label in np.unique(y_true):
        class_indices = np.flatnonzero(y_true == class_label)
        sampled_indices.append(rng.choice(class_indices, size=len(class_indices), replace=True))
    combined = np.concatenate(sampled_indices)
    return rng.permutation(combined)


def _resolve_transformer_factory(
    transformer_factory: Callable[[dict[str, Any]], Any] | None,
) -> Callable[[dict[str, Any]], Any]:
    """Return the caller-supplied transformer factory or the real MiniRocket builder."""
    return transformer_factory or _build_minirocket_transformer


def _build_minirocket_transformer(transformer_params: dict[str, Any]) -> MiniRocketMultivariate:
    """Build the one-time MiniRocket transformer from recovered frozen parameters."""
    return MiniRocketMultivariate(**transformer_params)


def recover_frozen_minirocket_ridge_config(final_model_path: Path = FINAL_MODEL_PATH) -> FrozenMiniRocketRidgeConfig:
    """Recover the transient Plan 1 §3.4 step-2 configuration from the frozen PKL.

    The frozen payload is opened read-only via `load_final_model()`. If
    `metadata.final_hparams` is empty, the configuration is recovered by
    unwrapping the fitted transformer and the underlying `RidgeClassifier`.

    Args:
        final_model_path: Path to `outputs/models/binary_minirocket_ridge_final.pkl`.

    Returns:
        Recovered MiniRocket and Ridge parameters for the transient procedure.

    Raises:
        ValueError: If the required frozen configuration cannot be recovered.
    """
    payload = load_final_model(final_model_path)
    if set(payload.keys()) != EXPECTED_FINAL_PKL_KEYS:
        raise ValueError(
            "Frozen final PKL does not match the expected 9-key schema for Plan 1 §3.4 configuration recovery."
        )

    metadata = payload["metadata"]
    if set(metadata.keys()) != EXPECTED_FINAL_METADATA_KEYS:
        raise ValueError("Frozen final PKL metadata does not match the expected 5-key schema.")

    transformer = payload["transformer"]
    if transformer is None or not hasattr(transformer, "get_params"):
        raise ValueError("Failed to recover the fitted MiniRocket transformer from the frozen PKL.")
    transformer_params = transformer.get_params(deep=False)
    recovered_transformer_params = {
        key: transformer_params[key]
        for key in ("num_kernels", "max_dilations_per_kernel", "random_state", "n_jobs")
        if key in transformer_params
    }
    if not recovered_transformer_params or "num_kernels" not in recovered_transformer_params:
        raise ValueError("Failed to recover MiniRocket hyperparameters from the frozen transformer.")

    fitted_model = payload["model"]
    if isinstance(fitted_model, RidgeClassifier):
        ridge_estimator = fitted_model
    elif hasattr(fitted_model, "estimator"):
        ridge_estimator = fitted_model.estimator
    elif hasattr(fitted_model, "base_estimator"):
        ridge_estimator = fitted_model.base_estimator
    else:
        ridge_estimator = None

    if ridge_estimator is None or not hasattr(ridge_estimator, "get_params"):
        raise ValueError("Failed to recover the plain RidgeClassifier configuration from the frozen PKL.")
    ridge_params = ridge_estimator.get_params(deep=False)
    if "alpha" not in ridge_params or "class_weight" not in ridge_params:
        raise ValueError("Recovered RidgeClassifier parameters are missing alpha and/or class_weight.")

    LOGGER.info("Recovered frozen MiniRocket params from %s: %s", _format_path(final_model_path), recovered_transformer_params)
    LOGGER.info("Recovered frozen RidgeClassifier params from %s: %s", _format_path(final_model_path), ridge_params)
    return FrozenMiniRocketRidgeConfig(
        transformer_params=dict(recovered_transformer_params),
        ridge_params=dict(ridge_params),
    )


def _load_locked_training_data() -> LockedTrainingData:
    """Load the aligned locked 777-row training set for Plan 1 §3.4 step 1.

    Returns:
        The aligned training tensor, labels, and ascending integer sample IDs.

    Raises:
        AssertionError: If the sample-id or label alignment contract fails.
        ValueError: If any shape or dtype expectation is violated.
    """
    try:
        sample_index = pd.read_csv(SAMPLE_INDEX_PATH)
    except OSError as exc:
        LOGGER.error("Failed to read sample index from %s.", SAMPLE_INDEX_PATH)
        raise OSError(f"Failed to read sample index from {SAMPLE_INDEX_PATH}.") from exc
    LOGGER.info(
        "Loaded sample index from %s with shape=%s, dtypes=%s, missing=%s.",
        _format_path(SAMPLE_INDEX_PATH),
        sample_index.shape,
        sample_index.dtypes.astype(str).to_dict(),
        sample_index.isna().sum().to_dict(),
    )
    final_sample_index = sample_index.loc[sample_index["in_final_output"].fillna(False)].copy()
    final_sample_index["sample_id"] = pd.to_numeric(final_sample_index["sample_id"], errors="raise").astype(np.int64)
    final_sample_index["label"] = pd.to_numeric(final_sample_index["label"], errors="raise").astype(np.int64)
    final_sample_index.sort_values("sample_id", kind="mergesort", inplace=True)
    final_sample_index.reset_index(drop=True, inplace=True)

    try:
        with np.load(SEQUENCE_PATH, allow_pickle=False) as sequence_npz:
            x_full = np.asarray(sequence_npz["x"])
            sequence_sample_ids = np.asarray(sequence_npz["sample_id"], dtype=np.int64)
            sequence_labels = np.asarray(sequence_npz["label"], dtype=np.int64)
    except OSError as exc:
        LOGGER.error("Failed to read Pipeline A tensor from %s.", SEQUENCE_PATH)
        raise OSError(f"Failed to read Pipeline A tensor from {SEQUENCE_PATH}.") from exc
    LOGGER.info(
        "Loaded Pipeline A NPZ from %s with x_shape=%s, x_dtype=%s, sample_id_dtype=%s, label_dtype=%s.",
        _format_path(SEQUENCE_PATH),
        x_full.shape,
        x_full.dtype,
        sequence_sample_ids.dtype,
        sequence_labels.dtype,
    )
    if x_full.ndim != 3 or x_full.shape[1] != CLIP_LENGTH_L or x_full.shape[2] != N_GAS_CHANNELS:
        raise ValueError(
            f"Expected Pipeline A tensor shape [N, {CLIP_LENGTH_L}, {N_GAS_CHANNELS}], received {x_full.shape}."
        )

    order = np.argsort(sequence_sample_ids, kind="stable")
    ordered_x_full = x_full[order]
    ordered_sequence_sample_ids = sequence_sample_ids[order]
    ordered_sequence_labels = sequence_labels[order]
    expected_sample_ids = final_sample_index["sample_id"].to_numpy(dtype=np.int64)
    expected_labels = final_sample_index["label"].to_numpy(dtype=np.int64)
    assert np.array_equal(
        expected_sample_ids,
        ordered_sequence_sample_ids,
    ), "Mismatch between sample_index.csv and pipeline_a_raw.npz sample_id ordering."
    assert np.array_equal(
        expected_labels,
        ordered_sequence_labels,
    ), "Mismatch between sample_index.csv and pipeline_a_raw.npz labels."

    try:
        with np.load(TRAIN_IDX_PATH, allow_pickle=False) as train_idx_npz:
            train_idx_names = list(train_idx_npz.files)
            train_idx = np.asarray(train_idx_npz["train_idx"], dtype=np.int64)
    except OSError as exc:
        LOGGER.error("Failed to read train_idx artifact from %s.", TRAIN_IDX_PATH)
        raise OSError(f"Failed to read train_idx artifact from {TRAIN_IDX_PATH}.") from exc
    LOGGER.info(
        "Loaded train-index NPZ from %s with arrays=%s and length=%s.",
        _format_path(TRAIN_IDX_PATH),
        train_idx_names,
        len(train_idx),
    )
    if len(train_idx) != 777:
        raise ValueError(f"Expected 777 train indices in {TRAIN_IDX_PATH}, found {len(train_idx)}.")

    x_train = ordered_x_full[train_idx]
    y_train = ordered_sequence_labels[train_idx]
    train_sample_ids = ordered_sequence_sample_ids[train_idx]
    if not np.issubdtype(train_sample_ids.dtype, np.integer):
        raise ValueError("Locked training sample IDs must be integer dtype.")
    if not np.all(np.diff(train_sample_ids) >= 0):
        raise AssertionError("Locked training sample IDs are not monotonically increasing per Erratum E007.")

    LOGGER.info(
        "Prepared locked 777-row training set with x_shape=%s, y_shape=%s, first_sample_ids=%s.",
        x_train.shape,
        y_train.shape,
        train_sample_ids[:10].tolist(),
    )
    return LockedTrainingData(x_train=x_train, y_train=y_train, train_sample_ids=train_sample_ids)


def run_optimism_correction_from_arrays(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    frozen_config: FrozenMiniRocketRidgeConfig,
    n_boot: int,
    random_seed: int,
    transformer_factory: Callable[[dict[str, Any]], Any] | None = None,
) -> OptimismCorrectionResult:
    """Run Plan 1 §3.4 steps 2-6 on preloaded arrays without writing outputs.

    Args:
        x_train: Raw Pipeline A tensor of shape `[N, L, 11]`.
        y_train: Binary labels aligned to `x_train`.
        frozen_config: Recovered frozen MiniRocket and Ridge parameters.
        n_boot: Number of bootstrap replicates.
        random_seed: Locked single-RNG sequential seed.
        transformer_factory: Optional transformer factory used by tests to inject
            a deterministic spy transformer.

    Returns:
        Structured optimism-correction result.

    Raises:
        ValueError: If any Plan 1 §3.4 halt condition is triggered.
    """
    if not isinstance(x_train, np.ndarray):
        raise TypeError("x_train must be a numpy array.")
    if not isinstance(y_train, np.ndarray):
        raise TypeError("y_train must be a numpy array.")
    if x_train.ndim != 3:
        raise ValueError("x_train must have shape [N, L, C].")
    if y_train.ndim != 1 or len(y_train) != len(x_train):
        raise ValueError("y_train must be one-dimensional and aligned to x_train.")
    if len(np.unique(y_train)) != 2:
        raise ValueError("Optimism correction requires exactly two classes in y_train.")
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1.")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise TypeError("random_seed must be an integer.")

    build_transformer = _resolve_transformer_factory(transformer_factory)
    x_train_ncl = np.transpose(x_train, (0, 2, 1)).astype(np.float32, copy=False)

    transformer = build_transformer(frozen_config.transformer_params)
    train_features = _coerce_feature_frame(transformer.fit_transform(x_train_ncl))
    transform_fit_count = int(getattr(transformer, "fit_calls", 1))
    LOGGER.info(
        "Fit MiniRocket exactly once on the full training tensor with feature_shape=%s and transform_fit_count=%s.",
        train_features.shape,
        transform_fit_count,
    )

    full_normalizer = _fit_feature_matrix_normalizer(train_features)
    train_features_full_norm = full_normalizer.transform(train_features)
    full_model = RidgeClassifier(**frozen_config.ridge_params)
    full_model.fit(train_features_full_norm, y_train)
    full_scores = np.asarray(full_model.decision_function(train_features_full_norm), dtype=float)
    auc_full_apparent = _compute_auc_roc(y_true=y_train, decision_scores=full_scores)
    if not np.isfinite(auc_full_apparent):
        raise ValueError("Plan 1 §3.4 halt: auc_full_apparent is NaN or Inf.")

    rng = np.random.default_rng(random_seed)
    auc_bootstrap_apparent = np.empty(n_boot, dtype=float)
    auc_original_from_bootstrap_model = np.empty(n_boot, dtype=float)
    for bootstrap_index in range(n_boot):
        sampled_indices = _draw_stratified_bootstrap_indices(y_true=y_train, rng=rng)
        sampled_labels = y_train[sampled_indices]
        if len(np.unique(sampled_labels)) != 2:
            raise ValueError(
                f"Plan 1 §3.4 halt: bootstrap iteration {bootstrap_index} is degenerate despite stratification."
            )

        bootstrap_features = train_features.iloc[sampled_indices].reset_index(drop=True)
        bootstrap_normalizer = _fit_feature_matrix_normalizer(bootstrap_features)
        bootstrap_features_norm = bootstrap_normalizer.transform(bootstrap_features)
        bootstrap_model = RidgeClassifier(**frozen_config.ridge_params)
        bootstrap_model.fit(bootstrap_features_norm, sampled_labels)

        bootstrap_scores = np.asarray(bootstrap_model.decision_function(bootstrap_features_norm), dtype=float)
        auc_bootstrap_apparent[bootstrap_index] = _compute_auc_roc(
            y_true=sampled_labels,
            decision_scores=bootstrap_scores,
        )

        original_features_from_bootstrap_model = bootstrap_normalizer.transform(train_features)
        original_scores = np.asarray(
            bootstrap_model.decision_function(original_features_from_bootstrap_model),
            dtype=float,
        )
        auc_original_from_bootstrap_model[bootstrap_index] = _compute_auc_roc(
            y_true=y_train,
            decision_scores=original_scores,
        )

    optimism_b = auc_bootstrap_apparent - auc_original_from_bootstrap_model
    mean_optimism = float(np.mean(optimism_b))
    auc_optimism_corrected = float(auc_full_apparent - mean_optimism)
    corrected_values = auc_full_apparent - optimism_b
    corrected_ci_low, corrected_ci_high = np.percentile(corrected_values, [2.5, 97.5])

    if not np.isfinite(auc_optimism_corrected):
        raise ValueError("Plan 1 §3.4 halt: auc_optimism_corrected is NaN or Inf.")

    return OptimismCorrectionResult(
        transformer_params=dict(frozen_config.transformer_params),
        ridge_params=dict(frozen_config.ridge_params),
        auc_bootstrap_apparent=auc_bootstrap_apparent,
        auc_original_from_bootstrap_model=auc_original_from_bootstrap_model,
        optimism_b=optimism_b,
        auc_full_apparent=float(auc_full_apparent),
        mean_optimism=mean_optimism,
        auc_optimism_corrected=auc_optimism_corrected,
        corrected_ci_low=float(corrected_ci_low),
        corrected_ci_high=float(corrected_ci_high),
        transform_fit_count=transform_fit_count,
    )


def analyze_optimism_correction(config_path: Path) -> OptimismCorrectionResult:
    """Run the full read-only Plan 1 §3.4 analysis without writing outputs.

    Args:
        config_path: Path to `configs/post_hoc_supplementary.yaml`.

    Returns:
        Structured optimism-correction result.
    """
    config = load_post_hoc_config(config_path)
    frozen_config = recover_frozen_minirocket_ridge_config(FINAL_MODEL_PATH)
    locked_training_data = _load_locked_training_data()

    LOGGER.info(
        "Running optimism correction with random_seed=%s, n_boot=%s, metric=%s.",
        config["random_seed"],
        config["optimism_correction"]["n_boot"],
        config["optimism_correction"]["metric"],
    )
    LOGGER.info(
        "Frozen comparator metrics: OOF AUC=%.4f [%.4f, %.4f], test AUC=%.4f [%.4f, %.4f].",
        FROZEN_OOF_AUC,
        FROZEN_OOF_CI_LOW,
        FROZEN_OOF_CI_HIGH,
        FROZEN_TEST_AUC,
        FROZEN_TEST_CI_LOW,
        FROZEN_TEST_CI_HIGH,
    )

    result = run_optimism_correction_from_arrays(
        x_train=locked_training_data.x_train,
        y_train=locked_training_data.y_train,
        frozen_config=frozen_config,
        n_boot=int(config["optimism_correction"]["n_boot"]),
        random_seed=int(config["random_seed"]),
    )
    LOGGER.info("Recovered transformer params: %s", result.transformer_params)
    LOGGER.info("Recovered RidgeClassifier params: %s", result.ridge_params)
    LOGGER.info("auc_full_apparent=%.6f", result.auc_full_apparent)
    LOGGER.info("mean_optimism=%.6f", result.mean_optimism)
    LOGGER.info("auc_optimism_corrected=%.6f", result.auc_optimism_corrected)
    LOGGER.info(
        "corrected_percentile_interval=[%.6f, %.6f]",
        result.corrected_ci_low,
        result.corrected_ci_high,
    )
    return result


def _config_lines_for_markdown(result: OptimismCorrectionResult) -> list[str]:
    """Render the recovered transient configuration bullets for the markdown report."""
    transformer_params = result.transformer_params
    ridge_params = result.ridge_params
    return [
        "- MiniRocket params recovered read-only from the frozen PKL transformer:",
        (
            f"  - `num_kernels = {int(transformer_params['num_kernels'])}` "
            "[runtime-determined-by-prespecified-algorithm: recovered read-only from frozen final PKL]"
        ),
        (
            f"  - `max_dilations_per_kernel = {int(transformer_params['max_dilations_per_kernel'])}` "
            "[runtime-determined-by-prespecified-algorithm: recovered read-only from frozen final PKL]"
        ),
        (
            f"  - `random_state = {int(transformer_params['random_state'])}` "
            "[runtime-determined-by-prespecified-algorithm: recovered read-only from frozen final PKL]"
        ),
        (
            f"  - `n_jobs = {int(transformer_params['n_jobs'])}` "
            "[runtime-determined-by-prespecified-algorithm: recovered read-only from frozen final PKL]"
        ),
        "- RidgeClassifier params recovered read-only from the frozen PKL model wrapper:",
        (
            f"  - `alpha = {float(ridge_params['alpha']):.1f}` "
            "[runtime-determined-by-prespecified-algorithm: recovered read-only from frozen final PKL]"
        ),
        f"  - `class_weight = {ridge_params['class_weight']!r}`",
        f"  - `copy_X = {ridge_params['copy_X']!r}`",
        f"  - `fit_intercept = {ridge_params['fit_intercept']!r}`",
        f"  - `max_iter = {ridge_params['max_iter']!r}`",
        f"  - `positive = {ridge_params['positive']!r}`",
        f"  - `random_state = {ridge_params['random_state']!r}`",
        f"  - `solver = {ridge_params['solver']!r}`",
        (
            f"  - `tol = {float(ridge_params['tol']):.4f}` "
            "[runtime-determined-by-prespecified-algorithm: recovered read-only from frozen final PKL]"
        ),
        "- The transient procedure omits Layer A sigmoid calibration, fits no Platt Layer B calibrator, and fits no threshold because Plan 1 §3.4 targets discrimination AUC only.",
        "- Platt calibration is monotonic and does not change AUC-ROC or AUC-PR; it may change Brier score, calibration slope, and calibration intercept.",
    ]


def _build_markdown_report(
    *,
    result: OptimismCorrectionResult,
    generated_at_utc: str,
    git_commit_hash: str,
    markdown_path: Path,
    csv_path: Path,
) -> str:
    """Build the Plan 1 §3.4 markdown report."""
    config_lines = "\n".join(_config_lines_for_markdown(result))
    return (
        "# Post-Hoc Optimism-Corrected Validation (`minirocket_ridge`)\n\n"
        f"> {POST_HOC_DECLARATION}\n\n"
        f"- `generated_at_utc = {generated_at_utc}`\n"
        f"- `git_commit_hash = {git_commit_hash}`\n"
        f"- `output_markdown = {_format_path(markdown_path)}`\n"
        f"- `output_csv = {_format_path(csv_path)}`\n\n"
        "## Recovered Transient Configuration\n\n"
        f"{config_lines}\n\n"
        "## Optimism-Correction Results\n\n"
        f"- `auc_full_apparent = {result.auc_full_apparent:.6f}`\n"
        f"- `mean_optimism = {result.mean_optimism:.6f}`\n"
        f"- `auc_optimism_corrected = {result.auc_optimism_corrected:.6f}`\n"
        f"- Descriptive percentile interval: `[{result.corrected_ci_low:.6f}, {result.corrected_ci_high:.6f}]`\n\n"
        "## Comparison To Frozen Primary Evidence\n\n"
        f"- Frozen nested-CV OOF AUC-ROC: `0.9246 [0.9048-0.9427]`\n"
        f"- Frozen test AUC-ROC: `0.9223 [0.8831-0.9585]`\n"
        f"- Supplementary optimism-corrected training AUC-ROC: `{result.auc_optimism_corrected:.6f} [{result.corrected_ci_low:.6f}-{result.corrected_ci_high:.6f}]`\n\n"
        "The frozen OOF and test AUC values remain the primary evidence for model discrimination. "
        "The optimism-corrected value reported here is post-hoc supplementary context only and does not "
        "modify any frozen primary-analysis artifact.\n"
    )


def generate_optimism_correction_outputs(
    config_path: Path,
    *,
    generated_at_utc: str | None = None,
    git_commit_hash: str | None = None,
) -> OptimismCorrectionResult:
    """Run Plan 1 §3.4 and write the deterministic CSV plus markdown report.

    Args:
        config_path: Path to `configs/post_hoc_supplementary.yaml`.
        generated_at_utc: Optional externally supplied UTC timestamp for the markdown header.
        git_commit_hash: Optional externally supplied `HEAD` commit hash for the markdown header.

    Returns:
        Structured optimism-correction result.
    """
    config = load_post_hoc_config(config_path)
    result = analyze_optimism_correction(config_path)
    timestamp_value = generated_at_utc or _timestamp_utc()
    git_hash_value = git_commit_hash or _get_git_commit_hash()

    outputs_analysis_dir = _resolve_path(config["paths"]["outputs_analysis_dir"])
    csv_path = outputs_analysis_dir / "posthoc_optimism_correction_minirocket_ridge.csv"
    markdown_path = outputs_analysis_dir / "posthoc_optimism_correction_minirocket_ridge.md"

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv_frame().to_csv(csv_path, index=False, float_format="%.17g")
    markdown = _build_markdown_report(
        result=result,
        generated_at_utc=timestamp_value,
        git_commit_hash=git_hash_value,
        markdown_path=markdown_path,
        csv_path=csv_path,
    )
    _write_text(markdown_path, markdown)
    LOGGER.info(
        "Wrote optimism-correction outputs to %s and %s.",
        _format_path(csv_path),
        _format_path(markdown_path),
    )
    return result
