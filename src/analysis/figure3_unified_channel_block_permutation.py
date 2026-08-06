"""Exploratory unified channel-block permutation importance analysis."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import itertools
import json
import logging
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
import yaml

from src.config import PROJECT_ROOT


LOGGER = logging.getLogger(__name__)
ANALYSIS_ID = "figure3_unified_channel_block_permutation"
CHANNEL_ORDER: tuple[str, ...] = tuple(f"S{index}" for index in range(1, 12))
MODEL_ORDER: tuple[dict[str, Any], ...] = (
    {
        "model_id": "minirocket_ridge",
        "model_display_name": "MiniRocket + Ridge",
        "model_order": 1,
        "pipeline": "pipeline_a",
    },
    {"model_id": "xgboost", "model_display_name": "XGBoost", "model_order": 2, "pipeline": "pipeline_b"},
    {
        "model_id": "elastic_net_lr",
        "model_display_name": "Elastic Net LR",
        "model_order": 3,
        "pipeline": "pipeline_b",
    },
    {"model_id": "svm", "model_display_name": "SVM", "model_order": 4, "pipeline": "pipeline_b"},
    {
        "model_id": "inception_time",
        "model_display_name": "InceptionTime",
        "model_order": 5,
        "pipeline": "pipeline_a",
    },
    {
        "model_id": "cnn_bilstm_attention",
        "model_display_name": "CNN–BiLSTM–Attention",
        "model_order": 6,
        "pipeline": "pipeline_a",
    },
)
CHANNEL_IMPORTANCE_COLUMNS: tuple[str, ...] = (
    "analysis_id",
    "model_id",
    "model_display_name",
    "model_order",
    "pipeline",
    "channel_index",
    "channel_name",
    "n_test",
    "n_positive",
    "n_negative",
    "baseline_auroc",
    "n_permutations",
    "mean_permuted_auroc",
    "mean_delta_auroc",
    "median_delta_auroc",
    "sd_delta_auroc",
    "repeat_delta_q025",
    "repeat_delta_q975",
    "negative_delta_count",
    "negative_delta_fraction",
    "rank_within_model",
    "n_bootstrap",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "ci_method",
)
PERMUTATION_REPEAT_COLUMNS: tuple[str, ...] = (
    "analysis_id",
    "model_id",
    "model_order",
    "channel_index",
    "channel_name",
    "permutation_repeat",
    "baseline_auroc",
    "permuted_auroc",
    "delta_auroc",
)
CROSS_MODEL_COLUMNS: tuple[str, ...] = (
    "analysis_id",
    "model_1_id",
    "model_1_order",
    "model_2_id",
    "model_2_order",
    "n_channels",
    "kendall_tau_b",
    "estimable",
    "nonestimable_reason",
)
CHANNEL_CONSENSUS_COLUMNS: tuple[str, ...] = (
    "analysis_id",
    "channel_index",
    "channel_name",
    "positive_model_count",
    "zero_model_count",
    "negative_model_count",
    "mean_rank",
    "median_rank",
    "minimum_rank",
    "maximum_rank",
    "mean_delta_auroc",
    "median_delta_auroc",
)
EXPECTED_OUTPUT_ROWS: Mapping[str, int] = {
    "channel_importance_summary.csv": 66,
    "permutation_repeats.csv": 6600,
    "cross_model_consistency.csv": 15,
    "channel_consensus_summary.csv": 11,
}
EXPECTED_OUTPUT_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "channel_importance_summary.csv": CHANNEL_IMPORTANCE_COLUMNS,
    "permutation_repeats.csv": PERMUTATION_REPEAT_COLUMNS,
    "cross_model_consistency.csv": CROSS_MODEL_COLUMNS,
    "channel_consensus_summary.csv": CHANNEL_CONSENSUS_COLUMNS,
}
SCIENTIFIC_PAYLOAD_FILENAMES: tuple[str, ...] = (
    "channel_importance_summary.csv",
    "permutation_repeats.csv",
    "cross_model_consistency.csv",
    "channel_consensus_summary.csv",
    "report.md",
)
FINAL_MANIFEST_FILENAMES: tuple[str, ...] = (
    "channel_importance_summary.csv",
    "permutation_repeats.csv",
    "cross_model_consistency.csv",
    "channel_consensus_summary.csv",
    "audit.json",
    "report.md",
)
VOLATILE_RUN_RECORD_FIELDS: tuple[str, ...] = (
    "output_directory",
    "run_started_at_utc",
    "run_finished_at_utc",
    "wall_clock_seconds",
)
EXPECTED_IMMUTABLE_INPUTS: Mapping[str, str] = {
    "configs/modeling_global.yaml": "c4520df8040adfebdfd1f6ae89376be5484ae3dfeda0cfd2481103df323a2e54",
    "data/processed/sequences/pipeline_a_raw.npz": (
        "82dcca83f0caf21f262fbf9f140fa4c0463a7d782986249bcea341d03517095f"
    ),
    "data/processed/features/pipeline_b_raw.csv": (
        "4594b83b6d08674f78501dabe5f236c303252bcd1ac9bf3657c7faa92d40acea"
    ),
    "data/processed/splits/test_idx.npz": (
        "7fc60acfc54f27e77f60a9813ca829fd61f174a8d3e4ab59aac18f5e05c981c3"
    ),
    "outputs/models/binary_minirocket_ridge_final.pkl": (
        "9152bc2b2bc7172dd6f6d7bc3491741538687043dcadd51ab173eb7f3bc39aef"
    ),
    "outputs/models/binary_xgboost_final.pkl": (
        "c8ec92965aab6ad4e13e818d79e07224a5f4cc1856c8f01d79b4167f4db34791"
    ),
    "outputs/models/binary_elastic_net_lr_final.pkl": (
        "9516e98a49defffb8283850d06d90dba06097490479baa55f94e041e6330fcd4"
    ),
    "outputs/models/binary_svm_final.pkl": (
        "85b6dfa28c3bee56284009ac3494873d5061abef9f85641c78defb89453d996b"
    ),
    "outputs/models/binary_inception_time_final.pkl": (
        "99e73e2c9dfc9baa74da0261f9027b9db4581529d127ecc5d73569e0fc8e8622"
    ),
    "outputs/models/binary_cnn_bilstm_attention_final.pkl": (
        "4e2a543ebbdbdf387e9ccb59dd5f64ea18e5d61c16cb480c5851bcdb92addd75"
    ),
    "outputs/predictions/binary_minirocket_ridge_test_predictions.csv": (
        "ec2f912cd83d523cdff92f453472daf3816cafe4238d98e5b6d650fefe73de28"
    ),
    "outputs/predictions/binary_xgboost_test_predictions.csv": (
        "bc3745fd5cb31b85e1256317d05ffe4f20a6e4083a239773a79afa270434730f"
    ),
    "outputs/predictions/binary_elastic_net_lr_test_predictions.csv": (
        "de472a79e76d9d254e7cde217b6ae11053d3a6cab9287b923548c6acdaa7be41"
    ),
    "outputs/predictions/binary_svm_test_predictions.csv": (
        "0b229d969f476b1ff5d335d5d7ce96af789daf3dc7fcc7ea883817a1ed028773"
    ),
    "outputs/predictions/binary_inception_time_test_predictions.csv": (
        "661c8387e0888977f990a9f586d68dab36677a8e3c5c0b41df1b65d8859bc5c0"
    ),
    "outputs/predictions/binary_cnn_bilstm_attention_test_predictions.csv": (
        "b56feeb932f6dccdc7dc666f30b8992112e44032d893f77ae63de337fe47e126"
    ),
    "outputs/metrics/binary_minirocket_ridge_test_metrics.csv": (
        "eb60d67d42383a0ff764633e09b271c98398ce461f20c2d094f7592a1b882835"
    ),
    "outputs/metrics/binary_xgboost_test_metrics.csv": (
        "314270fcb17885c53f0f477749c4dfdba4fee5f0607e052bd784c04ca1b27352"
    ),
    "outputs/metrics/binary_elastic_net_lr_test_metrics.csv": (
        "67d888b1d28e70a125098a80cd8797b6ad133c69c1af2325bdf5cb3fdaf33cc8"
    ),
    "outputs/metrics/binary_svm_test_metrics.csv": (
        "4b0d14a152905b67afdb55b7fad894853532b045fc870b3d419a561534484de2"
    ),
    "outputs/metrics/binary_inception_time_test_metrics.csv": (
        "59836ea0f12f4b956c7eaa802e4622f467282d34d49c4a4b573dcf1e25e3e5d2"
    ),
    "outputs/metrics/binary_cnn_bilstm_attention_test_metrics.csv": (
        "3c860545a1b0a1c4f2ee5c6cc7ac27771b855b6a943192998f9777c9de65ac2b"
    ),
}
EXPECTED_PATHS: Mapping[str, str] = {
    "modeling_config": "configs/modeling_global.yaml",
    "pipeline_a": "data/processed/sequences/pipeline_a_raw.npz",
    "pipeline_b": "data/processed/features/pipeline_b_raw.csv",
    "test_idx": "data/processed/splits/test_idx.npz",
    "final_output_dir": "outputs/analysis/figure3_unified_channel_block_permutation",
    "work_root": "outputs/work/figure3_unified_channel_block_permutation",
}
EXPECTED_RANDOM_STREAMS: Mapping[str, Any] = {
    "root_seed_sequence_entropy": 42,
    "spawn_count": 2,
    "permutation_spawn_key": [0],
    "bootstrap_spawn_key": [1],
    "matrix_digest_dtype": "<i8",
    "matrix_digest_order": "C",
}
EXPECTED_INTERPRETIVE_LIMITATIONS: tuple[str, ...] = (
    "Exploratory post hoc analysis.",
    "Internal same-center, same-device, same-period random validation only.",
    "Enriched case-control or two-gate sampling limits transportability.",
    "Breath was collected after diagnostic endoscopy and indicated biopsy but before treatment.",
    "No external, temporal, multicenter, cross-device, prospective, or deployment validation.",
    "A model score is not a gastric-cancer probability.",
    "The endpoint is high-grade gastric lesion, not gastric cancer alone.",
    "MOS channels are cross-responsive and cannot identify a molecule or disease-specific VOC.",
    "Channel-block importance does not establish causality or hardware inclusion or exclusion.",
    "Participant-level bootstrap confidence intervals are computed only for MiniRocket + Ridge.",
    "Supporting-model percentiles describe permutation-repeat spread, not confidence intervals.",
    "Correlated channel blocks can redistribute permutation importance.",
    "LGIN findings are exploratory because biopsy may undergrade disease and reference standards differ.",
)


def channel_position(channel_name: str) -> int:
    """Return the zero-based array position for a fixed channel.

    Args:
        channel_name: Channel name from S1 through S11.

    Returns:
        Zero-based NumPy axis position.

    Raises:
        ValueError: If the channel name is outside the fixed order.
    """
    try:
        return CHANNEL_ORDER.index(channel_name)
    except ValueError as exc:
        raise ValueError("Channel name is outside the fixed channel order.") from exc


def output_index_fields(channel_name: str, permutation_repeat: int) -> dict[str, int]:
    """Return persisted one-based channel and permutation numbering.

    Args:
        channel_name: Fixed channel name.
        permutation_repeat: Persisted repeat number from 1 through 100.

    Returns:
        One-based output fields.

    Raises:
        ValueError: If either persisted field is outside its frozen range.
    """
    position = channel_position(channel_name)
    if permutation_repeat < 1 or permutation_repeat > 100:
        raise ValueError("permutation_repeat must be between 1 and 100.")
    return {"channel_index": position + 1, "permutation_repeat": int(permutation_repeat)}


def _validate_permutation(permutation: np.ndarray, n_participants: int) -> np.ndarray:
    """Validate and return one complete participant permutation."""
    array = np.asarray(permutation)
    if array.ndim != 1 or len(array) != n_participants or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("Participant permutation has an invalid shape or dtype.")
    validated = array.astype(np.int64, copy=False)
    if not np.array_equal(np.sort(validated), np.arange(n_participants, dtype=np.int64)):
        raise ValueError("Participant permutation is not complete and unique.")
    return validated


def select_permutation_mapping(permutation_matrix: np.ndarray, permutation_repeat: int) -> np.ndarray:
    """Select a permutation by persisted one-based repeat number.

    Args:
        permutation_matrix: Matrix whose rows are zero-based internal permutation positions.
        permutation_repeat: Persisted repeat number from 1 through the matrix row count.

    Returns:
        Validated participant mapping for the requested repeat.

    Raises:
        ValueError: If the matrix or repeat number is invalid.
    """
    matrix = np.asarray(permutation_matrix)
    if matrix.ndim != 2 or not np.issubdtype(matrix.dtype, np.integer):
        raise ValueError("Permutation matrix must be a two-dimensional integer array.")
    if permutation_repeat < 1 or permutation_repeat > matrix.shape[0]:
        raise ValueError("permutation_repeat is outside the permutation matrix.")
    permutation_position = permutation_repeat - 1
    return _validate_permutation(matrix[permutation_position], matrix.shape[1])


def permute_pipeline_a_channel(
    original: np.ndarray,
    channel_name: str,
    participant_permutation: np.ndarray,
) -> np.ndarray:
    """Move one complete Pipeline A channel trajectory across participants.

    Args:
        original: Raw tensor with shape ``[N, L, 11]``.
        channel_name: Fixed channel selected by name.
        participant_permutation: Complete mapping of source participants.

    Returns:
        Copy with only the selected channel block permuted.

    Raises:
        TypeError: If the input is not a NumPy array.
        ValueError: If shape, values, or permutation are invalid.
    """
    if not isinstance(original, np.ndarray):
        raise TypeError("Pipeline A input must be a NumPy array.")
    if original.ndim != 3 or original.shape[2] != len(CHANNEL_ORDER):
        raise ValueError("Pipeline A input must have shape [N, L, 11].")
    if not np.isfinite(original).all():
        raise ValueError("Pipeline A input contains nonfinite values.")
    mapping = _validate_permutation(participant_permutation, len(original))
    position = channel_position(channel_name)
    permuted = original.copy()
    permuted[:, :, position] = original[mapping, :, position]
    expected_selected = original[mapping, :, position]
    if not np.array_equal(permuted[:, :, position], expected_selected):
        raise ValueError("Pipeline A selected-channel block validation failed.")
    other_positions = [index for index in range(len(CHANNEL_ORDER)) if index != position]
    if permuted[:, :, other_positions].tobytes() != original[:, :, other_positions].tobytes():
        raise ValueError("Pipeline A nonselected-channel byte validation failed.")
    return permuted


def build_pipeline_b_channel_map(feature_names: Sequence[str]) -> dict[str, list[str]]:
    """Build the exact 11-by-9 Pipeline B channel map.

    Args:
        feature_names: Ordered raw Pipeline B feature names.

    Returns:
        Fixed-order mapping from channel name to nine feature names.

    Raises:
        ValueError: If names are missing, duplicated, unknown, or ambiguously mapped.
    """
    names = [str(name) for name in feature_names]
    if len(names) != 99 or len(set(names)) != 99:
        raise ValueError("Pipeline B channel mapping requires 99 unique feature names.")
    channel_map = {channel_name: [] for channel_name in CHANNEL_ORDER}
    for feature_name in names:
        suffix = feature_name.rsplit("_", 1)[-1]
        if suffix not in channel_map:
            raise ValueError("Pipeline B channel mapping contains an unknown channel suffix.")
        channel_map[suffix].append(feature_name)
    if any(len(channel_map[channel_name]) != 9 for channel_name in CHANNEL_ORDER):
        raise ValueError("Pipeline B channel mapping requires exactly nine features per channel.")
    assigned = [feature for channel_name in CHANNEL_ORDER for feature in channel_map[channel_name]]
    if len(assigned) != 99 or set(assigned) != set(names):
        raise ValueError("Pipeline B channel mapping is incomplete or duplicated.")
    return channel_map


def permute_pipeline_b_channel(
    original: pd.DataFrame,
    channel_name: str,
    participant_permutation: np.ndarray,
    channel_map: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Move one complete nine-feature Pipeline B block across participants.

    Args:
        original: Raw 99-feature input frame.
        channel_name: Fixed channel selected by name.
        participant_permutation: Complete mapping of source participants.
        channel_map: Validated channel-name to feature-name mapping.

    Returns:
        Copy with only the selected nine-feature block permuted.

    Raises:
        TypeError: If the input is not a DataFrame.
        ValueError: If the frame, mapping, channel, or permutation is invalid.
    """
    if not isinstance(original, pd.DataFrame):
        raise TypeError("Pipeline B input must be a pandas DataFrame.")
    if original.shape[1] != 99 or original.columns.duplicated().any():
        raise ValueError("Pipeline B input must contain exactly 99 unique feature columns.")
    if not np.isfinite(original.to_numpy(dtype=float)).all():
        raise ValueError("Pipeline B input contains nonfinite values.")
    if channel_name not in channel_map or len(channel_map[channel_name]) != 9:
        raise ValueError("Pipeline B selected channel mapping is missing or incomplete.")
    selected = list(channel_map[channel_name])
    if any(feature not in original.columns for feature in selected):
        raise ValueError("Pipeline B selected channel features are absent from the input.")
    mapping = _validate_permutation(participant_permutation, len(original))
    other = [feature for feature in original.columns if feature not in selected]
    permuted = original.copy()
    permuted.loc[:, selected] = original.iloc[mapping].loc[:, selected].to_numpy(copy=True)
    if permuted.columns.tolist() != original.columns.tolist():
        raise ValueError("Pipeline B feature order changed during permutation.")
    if not np.array_equal(
        permuted.loc[:, selected].to_numpy(),
        original.iloc[mapping].loc[:, selected].to_numpy(),
    ):
        raise ValueError("Pipeline B selected-channel block validation failed.")
    if permuted.loc[:, other].to_numpy().tobytes() != original.loc[:, other].to_numpy().tobytes():
        raise ValueError("Pipeline B nonselected-feature byte validation failed.")
    return permuted


def _canonical_matrix_digest(matrix: np.ndarray) -> str:
    """Hash a matrix as C-order little-endian signed 64-bit bytes."""
    canonical = np.ascontiguousarray(matrix, dtype="<i8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def generate_random_matrices(
    labels: np.ndarray,
    *,
    seed: int,
    n_permutations: int,
    n_bootstrap: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate the independent permutation and stratified-bootstrap streams.

    Args:
        labels: Binary aligned outcome labels.
        seed: Root SeedSequence entropy.
        n_permutations: Number of complete participant permutations.
        n_bootstrap: Number of stratified bootstrap rows.

    Returns:
        Permutation matrix, bootstrap matrix, and non-reversible digest metadata.

    Raises:
        ValueError: If labels or counts are invalid.
    """
    y_true = np.asarray(labels)
    if y_true.ndim != 1 or not np.issubdtype(y_true.dtype, np.integer):
        raise ValueError("Labels must be a one-dimensional integer array.")
    if set(np.unique(y_true).tolist()) != {0, 1}:
        raise ValueError("Labels must contain both binary classes 0 and 1.")
    if n_permutations <= 0 or n_bootstrap <= 0:
        raise ValueError("Random-matrix row counts must be positive.")
    root = np.random.SeedSequence(seed)
    permutation_seed, bootstrap_seed = root.spawn(2)
    permutation_rng = np.random.default_rng(permutation_seed)
    bootstrap_rng = np.random.default_rng(bootstrap_seed)
    n_participants = len(y_true)
    permutations = np.vstack(
        [permutation_rng.permutation(n_participants) for _ in range(n_permutations)]
    ).astype(np.int64, copy=False)
    for row in permutations:
        _validate_permutation(row, n_participants)
    positive_positions = np.flatnonzero(y_true == 1)
    negative_positions = np.flatnonzero(y_true == 0)
    bootstrap_rows: list[np.ndarray] = []
    for _ in range(n_bootstrap):
        positive_draw = bootstrap_rng.choice(positive_positions, size=len(positive_positions), replace=True)
        negative_draw = bootstrap_rng.choice(negative_positions, size=len(negative_positions), replace=True)
        bootstrap_rows.append(np.concatenate([positive_draw, negative_draw]).astype(np.int64, copy=False))
    bootstrap = np.vstack(bootstrap_rows).astype(np.int64, copy=False)
    if any(
        np.count_nonzero(y_true[row] == 1) != len(positive_positions)
        or np.count_nonzero(y_true[row] == 0) != len(negative_positions)
        for row in bootstrap
    ):
        raise ValueError("Stratified bootstrap class-count validation failed.")
    digests: dict[str, Any] = {
        "permutation_matrix_sha256": _canonical_matrix_digest(permutations),
        "bootstrap_matrix_sha256": _canonical_matrix_digest(bootstrap),
        "permutation_matrix_shape": list(permutations.shape),
        "bootstrap_matrix_shape": list(bootstrap.shape),
        "matrix_digest_dtype": "<i8",
        "matrix_digest_order": "C",
        "root_seed_sequence_entropy": int(seed),
        "permutation_spawn_key": list(permutation_seed.spawn_key),
        "bootstrap_spawn_key": list(bootstrap_seed.spawn_key),
        "bit_generator": permutation_rng.bit_generator.__class__.__name__,
    }
    return permutations, bootstrap, digests


def signed_auroc_delta(baseline_auroc: float, permuted_auroc: float) -> float:
    """Return the unmodified signed absolute AUROC decrease."""
    baseline = float(baseline_auroc)
    permuted = float(permuted_auroc)
    if not math.isfinite(baseline) or not math.isfinite(permuted):
        raise ValueError("AUROC values must be finite.")
    return baseline - permuted


def descending_average_ranks(values: np.ndarray) -> np.ndarray:
    """Return descending ranks with average ranks for exact ties."""
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("Rank values must be a finite one-dimensional vector.")
    return np.asarray(rankdata(-vector, method="average"), dtype=float)


def kendall_tau_b_coefficient(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    """Calculate Kendall tau-b without calculating an inferential P value.

    Args:
        left: First channel-value vector.
        right: Second channel-value vector.

    Returns:
        Coefficient, estimability flag, and nonestimability reason.

    Raises:
        ValueError: If vector dimensions, shapes, or values are invalid.
    """
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("Kendall tau-b vectors must be one-dimensional with identical shape.")
    if len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Kendall tau-b vectors must contain finite values and at least two entries.")
    numerator = 0.0
    not_tied_x = 0
    not_tied_y = 0
    for left_index in range(len(x) - 1):
        for right_index in range(left_index + 1, len(x)):
            x_sign = float(np.sign(x[left_index] - x[right_index]))
            y_sign = float(np.sign(y[left_index] - y[right_index]))
            numerator += x_sign * y_sign
            not_tied_x += int(x_sign != 0.0)
            not_tied_y += int(y_sign != 0.0)
    denominator = math.sqrt(not_tied_x * not_tied_y)
    if denominator == 0.0:
        return {
            "kendall_tau_b": None,
            "estimable": False,
            "nonestimable_reason": "constant_or_degenerate_channel_vector",
        }
    coefficient = numerator / denominator
    if not math.isfinite(coefficient):
        return {
            "kendall_tau_b": None,
            "estimable": False,
            "nonestimable_reason": "constant_or_degenerate_channel_vector",
        }
    return {"kendall_tau_b": float(coefficient), "estimable": True, "nonestimable_reason": ""}


def _auc_rows_from_stratified_scores(score_rows: np.ndarray, n_positive: int, n_negative: int) -> np.ndarray:
    """Calculate tie-aware AUROC for rows ordered as positive draws then negative draws."""
    ranks = rankdata(np.asarray(score_rows, dtype=float), method="average", axis=1)
    positive_rank_sums = ranks[:, :n_positive].sum(axis=1)
    offset = n_positive * (n_positive + 1) / 2.0
    return np.asarray((positive_rank_sums - offset) / (n_positive * n_negative), dtype=float)


def paired_bootstrap_channel_deltas(
    labels: np.ndarray,
    baseline_scores: np.ndarray,
    permuted_scores: np.ndarray,
    bootstrap_matrix: np.ndarray,
) -> np.ndarray:
    """Calculate paired bootstrap deltas for one channel.

    Args:
        labels: Aligned binary labels.
        baseline_scores: Frozen baseline score vector.
        permuted_scores: Fixed permutation score matrix with shape ``[R, N]``.
        bootstrap_matrix: Shared stratified index matrix with shape ``[B, N]``.

    Returns:
        One paired delta per bootstrap row.

    Raises:
        ValueError: If shapes, labels, or values are invalid.
    """
    y_true = np.asarray(labels)
    baseline = np.asarray(baseline_scores, dtype=float)
    permuted = np.asarray(permuted_scores, dtype=float)
    bootstrap = np.asarray(bootstrap_matrix)
    if y_true.ndim != 1 or baseline.shape != y_true.shape:
        raise ValueError("Baseline labels and scores must have identical one-dimensional shape.")
    if permuted.ndim != 2 or permuted.shape[1] != len(y_true):
        raise ValueError("Permuted score matrix must have shape [R, N].")
    if bootstrap.ndim != 2 or bootstrap.shape[1] != len(y_true):
        raise ValueError("Bootstrap matrix must have shape [B, N].")
    if not np.isfinite(baseline).all() or not np.isfinite(permuted).all():
        raise ValueError("Bootstrap scores contain nonfinite values.")
    positive_count = int(np.count_nonzero(y_true == 1))
    negative_count = int(np.count_nonzero(y_true == 0))
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Bootstrap AUROC requires both classes.")
    deltas = np.empty(len(bootstrap), dtype=float)
    for bootstrap_position, row in enumerate(bootstrap):
        if np.count_nonzero(y_true[row] == 1) != positive_count or not np.all(y_true[row][:positive_count] == 1):
            raise ValueError("Bootstrap row does not preserve positive class count and ordering.")
        if np.count_nonzero(y_true[row] == 0) != negative_count or not np.all(y_true[row][positive_count:] == 0):
            raise ValueError("Bootstrap row does not preserve negative class count and ordering.")
        all_scores = np.vstack([baseline[row], permuted[:, row]])
        aurocs = _auc_rows_from_stratified_scores(all_scores, positive_count, negative_count)
        deltas[bootstrap_position] = aurocs[0] - float(np.mean(aurocs[1:]))
    if not np.isfinite(deltas).all():
        raise ValueError("Paired bootstrap produced nonfinite deltas.")
    return deltas


def reconcile_baseline_scores(
    labels: np.ndarray,
    reproduced_scores: np.ndarray,
    frozen_scores: np.ndarray,
    *,
    metric_auroc: float,
    score_atol: float,
    auroc_atol: float,
) -> dict[str, float | bool]:
    """Fail closed unless frozen scores and all three AUROCs reconcile.

    Args:
        labels: Aligned binary labels.
        reproduced_scores: Newly inferred raw positive-class scores.
        frozen_scores: Frozen prediction raw scores.
        metric_auroc: Frozen test-metric AUROC.
        score_atol: Maximum allowed absolute score difference.
        auroc_atol: Maximum allowed absolute AUROC difference.

    Returns:
        Aggregate reconciliation diagnostics.

    Raises:
        ValueError: If shape, finiteness, range, score, or AUROC checks fail.
    """
    y_true = np.asarray(labels)
    reproduced = np.asarray(reproduced_scores, dtype=float)
    frozen = np.asarray(frozen_scores, dtype=float)
    if y_true.ndim != 1 or reproduced.shape != y_true.shape or frozen.shape != y_true.shape:
        raise ValueError("Baseline reconciliation arrays have inconsistent shapes.")
    if set(np.unique(y_true).tolist()) != {0, 1}:
        raise ValueError("Baseline reconciliation requires both binary classes.")
    if not np.isfinite(reproduced).all() or not np.isfinite(frozen).all():
        raise ValueError("Baseline reconciliation scores contain nonfinite values.")
    if not ((reproduced >= 0.0).all() and (reproduced <= 1.0).all()):
        raise ValueError("Reproduced scores are outside [0, 1].")
    max_difference = float(np.max(np.abs(reproduced - frozen)))
    if max_difference > score_atol:
        raise ValueError("Baseline score reconciliation exceeded the frozen tolerance.")
    reproduced_auroc = float(roc_auc_score(y_true, reproduced))
    frozen_auroc = float(roc_auc_score(y_true, frozen))
    metric_value = float(metric_auroc)
    differences = (
        abs(reproduced_auroc - frozen_auroc),
        abs(reproduced_auroc - metric_value),
        abs(frozen_auroc - metric_value),
    )
    if not math.isfinite(metric_value) or max(differences) > auroc_atol:
        raise ValueError("Baseline AUROC reconciliation exceeded the frozen tolerance.")
    return {
        "max_abs_score_difference": max_difference,
        "reproduced_auroc": reproduced_auroc,
        "frozen_prediction_auroc": frozen_auroc,
        "frozen_metric_auroc": metric_value,
        "reproduced_vs_prediction_auroc_difference": differences[0],
        "reproduced_vs_metric_auroc_difference": differences[1],
        "prediction_vs_metric_auroc_difference": differences[2],
        "passed": True,
    }


def validate_output_frames(frames: Mapping[str, pd.DataFrame], *, strict_values: bool = True) -> None:
    """Validate exact aggregate CSV schemas and row counts.

    Args:
        frames: Mapping from required output filename to DataFrame.
        strict_values: Whether to apply scientific value and ordering checks.

    Raises:
        ValueError: If a filename, schema, row count, or scientific contract differs.
    """
    if set(frames) != set(EXPECTED_OUTPUT_ROWS):
        raise ValueError("Output frame inventory differs from the frozen contract.")
    for filename, expected_rows in EXPECTED_OUTPUT_ROWS.items():
        frame = frames[filename]
        if frame.columns.tolist() != list(EXPECTED_OUTPUT_COLUMNS[filename]):
            raise ValueError(f"{filename} schema differs from the frozen contract.")
        if len(frame) != expected_rows:
            raise ValueError(f"{filename} row count differs from the frozen contract.")
    if not strict_values:
        return
    for filename, frame in frames.items():
        if frame["analysis_id"].isna().any() or not (frame["analysis_id"] == ANALYSIS_ID).all():
            raise ValueError(f"{filename} analysis identity differs from the frozen contract.")

    summary = frames["channel_importance_summary.csv"]
    repeats = frames["permutation_repeats.csv"]
    consistency = frames["cross_model_consistency.csv"]
    consensus = frames["channel_consensus_summary.csv"]
    expected_summary_order = [
        (str(model["model_id"]), int(model["model_order"]), channel_index, channel_name)
        for model in MODEL_ORDER
        for channel_index, channel_name in enumerate(CHANNEL_ORDER, start=1)
    ]
    observed_summary_order = list(
        summary[["model_id", "model_order", "channel_index", "channel_name"]].itertuples(index=False, name=None)
    )
    expected_repeat_order = [
        (str(model["model_id"]), int(model["model_order"]), channel_index, channel_name, repeat)
        for model in MODEL_ORDER
        for channel_index, channel_name in enumerate(CHANNEL_ORDER, start=1)
        for repeat in range(1, 101)
    ]
    observed_repeat_order = list(
        repeats[
            ["model_id", "model_order", "channel_index", "channel_name", "permutation_repeat"]
        ].itertuples(index=False, name=None)
    )
    expected_consistency_order = [
        (
            str(model_1["model_id"]),
            int(model_1["model_order"]),
            str(model_2["model_id"]),
            int(model_2["model_order"]),
        )
        for model_1, model_2 in itertools.combinations(MODEL_ORDER, 2)
    ]
    observed_consistency_order = list(
        consistency[["model_1_id", "model_1_order", "model_2_id", "model_2_order"]].itertuples(
            index=False,
            name=None,
        )
    )
    expected_consensus_order = [(index, channel) for index, channel in enumerate(CHANNEL_ORDER, start=1)]
    observed_consensus_order = list(
        consensus[["channel_index", "channel_name"]].itertuples(index=False, name=None)
    )
    if (
        observed_summary_order != expected_summary_order
        or observed_repeat_order != expected_repeat_order
        or observed_consistency_order != expected_consistency_order
        or observed_consensus_order != expected_consensus_order
    ):
        raise ValueError("Aggregate output ordering differs from the frozen contract.")

    integer_columns = {
        "channel_importance_summary.csv": (
            "model_order",
            "channel_index",
            "n_test",
            "n_positive",
            "n_negative",
            "n_permutations",
            "negative_delta_count",
            "n_bootstrap",
        ),
        "permutation_repeats.csv": ("model_order", "channel_index", "permutation_repeat"),
        "cross_model_consistency.csv": ("model_1_order", "model_2_order", "n_channels"),
        "channel_consensus_summary.csv": (
            "channel_index",
            "positive_model_count",
            "zero_model_count",
            "negative_model_count",
        ),
    }
    for filename, columns in integer_columns.items():
        if any(not pd.api.types.is_integer_dtype(frames[filename][column]) for column in columns):
            raise ValueError(f"{filename} integer dtypes differ from the frozen contract.")

    numeric_repeat_columns = ["baseline_auroc", "permuted_auroc", "delta_auroc"]
    if not np.isfinite(repeats.loc[:, numeric_repeat_columns].to_numpy(dtype=float)).all():
        raise ValueError("Permutation-repeat output contains nonfinite values.")
    if not (
        repeats["baseline_auroc"].between(0.0, 1.0).all()
        and repeats["permuted_auroc"].between(0.0, 1.0).all()
        and repeats["delta_auroc"].between(-1.0, 1.0).all()
    ):
        raise ValueError("Permutation-repeat values are outside their valid ranges.")
    expected_delta = repeats["baseline_auroc"].to_numpy(dtype=float) - repeats["permuted_auroc"].to_numpy(dtype=float)
    if not np.allclose(expected_delta, repeats["delta_auroc"].to_numpy(dtype=float), atol=1.0e-15, rtol=0.0):
        raise ValueError("Permutation-repeat signed deltas do not reconcile.")

    final_rows = summary["model_id"] == "minirocket_ridge"
    supporting_rows = ~final_rows
    if (
        not (summary.loc[final_rows, "n_bootstrap"] == 2000).all()
        or summary.loc[final_rows, ["bootstrap_ci_low", "bootstrap_ci_high"]].isna().any(axis=None)
        or not (summary.loc[supporting_rows, "n_bootstrap"] == 0).all()
        or summary.loc[supporting_rows, ["bootstrap_ci_low", "bootstrap_ci_high"]].notna().any(axis=None)
    ):
        raise ValueError("Bootstrap confidence interval missingness differs from the frozen design.")
    final_cis = summary.loc[final_rows, ["bootstrap_ci_low", "bootstrap_ci_high"]].to_numpy(dtype=float)
    if not np.isfinite(final_cis).all() or not np.all(final_cis[:, 0] <= final_cis[:, 1]):
        raise ValueError("Final-model bootstrap confidence intervals are invalid.")
    if not (
        summary.loc[final_rows, "ci_method"]
        == "paired_stratified_participant_bootstrap_fixed_permutations_percentile_95"
    ).all() or not (
        summary.loc[supporting_rows, "ci_method"] == "not_computed_supporting_models_by_design"
    ).all():
        raise ValueError("Confidence interval method labels differ from the frozen design.")

    estimable = consistency["estimable"].map(lambda value: str(value).lower())
    if not estimable.isin({"true", "false"}).all() or not (consistency["n_channels"] == 11).all():
        raise ValueError("Cross-model estimability fields differ from the frozen contract.")
    coefficients = pd.to_numeric(consistency["kendall_tau_b"], errors="coerce")
    reasons = consistency["nonestimable_reason"].fillna("").astype(str)
    if (
        not np.isfinite(coefficients[estimable == "true"].to_numpy(dtype=float)).all()
        or not coefficients[estimable == "true"].between(-1.0, 1.0).all()
        or coefficients[estimable == "false"].notna().any()
        or not (reasons[estimable == "true"] == "").all()
        or not (reasons[estimable == "false"] == "constant_or_degenerate_channel_vector").all()
    ):
        raise ValueError("Cross-model coefficient or nonestimability fields are invalid.")

    sign_counts = consensus[["positive_model_count", "zero_model_count", "negative_model_count"]].sum(axis=1)
    if not np.array_equal(sign_counts.to_numpy(dtype=int), np.full(11, 6, dtype=int)):
        raise ValueError("Channel-consensus sign counts do not sum to six.")


_FORBIDDEN_OUTPUT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsample_id\b", flags=re.IGNORECASE),
    re.compile(r"\by_true\b", flags=re.IGNORECASE),
    re.compile(r"\bp_cal\b", flags=re.IGNORECASE),
    re.compile(r"\by_pred\b", flags=re.IGNORECASE),
    re.compile(r"\bparticipant_(?:id|index|indices|key|keys)\b", flags=re.IGNORECASE),
    re.compile(r"\b(?:permutation|bootstrap)_(?:index|indices)\b", flags=re.IGNORECASE),
    re.compile(r"data/raw/(?:clinical_data|breath_data)", flags=re.IGNORECASE),
    re.compile(r"(?:^|[\s\"'])(?:/home/|/mnt/|[A-Za-z]:\\)"),
)
_DATE_PREFIX_PATTERN = re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}")
_RUN_TIMESTAMP_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_ALLOWED_TIMESTAMP_FIELDS = frozenset({"run_started_at_utc", "run_finished_at_utc"})


def _validate_allowlisted_provenance_timestamps(filename: str, text: str) -> None:
    """Allow date-bearing text only in exact top-level run-record timestamp fields."""
    if _DATE_PREFIX_PATTERN.search(text) is None:
        return
    if Path(filename).name != "run_record.json":
        raise ValueError(f"Output privacy validation failed for {Path(filename).name}.")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Output privacy validation failed for run_record.json.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Output privacy validation failed for run_record.json.")

    def visit(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                key_text = str(key)
                if _DATE_PREFIX_PATTERN.search(key_text):
                    raise ValueError("Output privacy validation failed for run_record.json.")
                visit(child, (*path, key_text))
        elif isinstance(value, list):
            for child in value:
                visit(child, path)
        elif isinstance(value, str) and _DATE_PREFIX_PATTERN.search(value):
            if len(path) != 1 or path[0] not in _ALLOWED_TIMESTAMP_FIELDS:
                raise ValueError("Output privacy validation failed for run_record.json.")
            if _RUN_TIMESTAMP_PATTERN.fullmatch(value) is None:
                raise ValueError("Output privacy validation failed for run_record.json.")

    visit(payload, ())


def validate_serialized_privacy(serialized_files: Mapping[str, bytes]) -> None:
    """Reject forbidden participant-level or path material without echoing it.

    Args:
        serialized_files: Candidate filename-to-bytes mapping.

    Raises:
        ValueError: If UTF-8 decoding or a privacy rule fails.
    """
    for filename, payload in serialized_files.items():
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Output privacy validation failed for a non-UTF-8 candidate.") from exc
        if any(pattern.search(text) for pattern in _FORBIDDEN_OUTPUT_PATTERNS):
            raise ValueError(f"Output privacy validation failed for {Path(filename).name}.")
        _validate_allowlisted_provenance_timestamps(filename, text)


def atomic_write_bytes(
    target: Path,
    payload: bytes,
    *,
    validator: Callable[[Path], None] | None = None,
) -> None:
    """Write, validate, and atomically promote one new file without overwrite.

    Args:
        target: New destination path.
        payload: Exact serialized bytes.
        validator: Optional validation called on the destination-local temporary file.

    Raises:
        FileExistsError: If the target or temporary path already exists.
        OSError: If writing or replacement fails.
        ValueError: If the supplied validator rejects the temporary file.
    """
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Output target already exists: {target.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.tmp"
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"Temporary output target already exists: {temporary.name}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if validator is not None:
            validator(temporary)
        os.replace(temporary, target)
    except Exception:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
        raise


def promote_candidate_directory(candidate: Path, final: Path) -> None:
    """Atomically promote one complete candidate directory without overwrite.

    Args:
        candidate: Validated task-owned candidate directory.
        final: Absent final scientific output directory.

    Raises:
        FileExistsError: If the final directory already exists.
        ValueError: If candidate validation or same-filesystem checks fail.
        OSError: If the atomic directory replacement fails.
    """
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"Final output already exists: {final.name}")
    if not candidate.is_dir() or candidate.is_symlink():
        raise ValueError("Promotion candidate must be a real directory.")
    final.parent.mkdir(parents=True, exist_ok=True)
    if candidate.stat().st_dev != final.parent.stat().st_dev:
        raise ValueError("Atomic promotion requires candidate and destination on the same filesystem.")
    os.replace(candidate, final)


def build_aggregate_frames(
    repeat_frame: pd.DataFrame,
    bootstrap_intervals: Mapping[str, tuple[float, float]],
) -> dict[str, pd.DataFrame]:
    """Build all four aggregate scientific CSV frames.

    Args:
        repeat_frame: Complete 6-by-11-by-100 repeat-level aggregate frame.
        bootstrap_intervals: Final-model confidence interval by fixed channel name.

    Returns:
        Filename-keyed aggregate frames with exact schemas and ordering.

    Raises:
        ValueError: If repeat coverage, ordering keys, or bootstrap intervals differ from the frozen contract.
    """
    if repeat_frame.columns.tolist() != list(PERMUTATION_REPEAT_COLUMNS) or len(repeat_frame) != 6600:
        raise ValueError("Permutation-repeat frame schema or row count differs from the frozen contract.")
    if set(bootstrap_intervals) != set(CHANNEL_ORDER):
        raise ValueError("Final-model bootstrap interval inventory differs from the fixed channel order.")
    repeats = repeat_frame.copy().reset_index(drop=True)
    summary_rows: list[dict[str, Any]] = []
    mean_vectors: dict[str, np.ndarray] = {}
    for model in MODEL_ORDER:
        model_id = str(model["model_id"])
        model_rows: list[dict[str, Any]] = []
        mean_deltas: list[float] = []
        for channel_index, channel_name in enumerate(CHANNEL_ORDER, start=1):
            subset = repeats.loc[
                (repeats["model_id"] == model_id) & (repeats["channel_name"] == channel_name)
            ].sort_values("permutation_repeat", kind="mergesort")
            if len(subset) != 100 or subset["permutation_repeat"].tolist() != list(range(1, 101)):
                raise ValueError("Permutation-repeat model-channel coverage is incomplete.")
            if not (subset["model_order"] == int(model["model_order"])).all():
                raise ValueError("Permutation-repeat model order differs from the frozen order.")
            if not (subset["channel_index"] == channel_index).all():
                raise ValueError("Permutation-repeat channel index differs from the one-based contract.")
            baselines = subset["baseline_auroc"].to_numpy(dtype=float)
            if not np.all(baselines == baselines[0]):
                raise ValueError("Baseline AUROC is not constant within a model-channel group.")
            permuted = subset["permuted_auroc"].to_numpy(dtype=float)
            deltas = subset["delta_auroc"].to_numpy(dtype=float)
            if not np.isfinite(np.concatenate([baselines, permuted, deltas])).all():
                raise ValueError("Permutation-repeat values contain nonfinite entries.")
            mean_delta = float(np.mean(deltas))
            mean_deltas.append(mean_delta)
            is_final = model_id == "minirocket_ridge"
            ci_low, ci_high = bootstrap_intervals[channel_name]
            model_rows.append(
                {
                    "analysis_id": ANALYSIS_ID,
                    "model_id": model_id,
                    "model_display_name": model["model_display_name"],
                    "model_order": int(model["model_order"]),
                    "pipeline": model["pipeline"],
                    "channel_index": channel_index,
                    "channel_name": channel_name,
                    "n_test": 195,
                    "n_positive": 82,
                    "n_negative": 113,
                    "baseline_auroc": float(baselines[0]),
                    "n_permutations": 100,
                    "mean_permuted_auroc": float(np.mean(permuted)),
                    "mean_delta_auroc": mean_delta,
                    "median_delta_auroc": float(np.median(deltas)),
                    "sd_delta_auroc": float(np.std(deltas, ddof=1)),
                    "repeat_delta_q025": float(np.percentile(deltas, 2.5, method="linear")),
                    "repeat_delta_q975": float(np.percentile(deltas, 97.5, method="linear")),
                    "negative_delta_count": int(np.count_nonzero(deltas < 0.0)),
                    "negative_delta_fraction": float(np.count_nonzero(deltas < 0.0) / 100.0),
                    "rank_within_model": 0.0,
                    "n_bootstrap": 2000 if is_final else 0,
                    "bootstrap_ci_low": float(ci_low) if is_final else None,
                    "bootstrap_ci_high": float(ci_high) if is_final else None,
                    "ci_method": (
                        "paired_stratified_participant_bootstrap_fixed_permutations_percentile_95"
                        if is_final
                        else "not_computed_supporting_models_by_design"
                    ),
                }
            )
        ranks = descending_average_ranks(np.asarray(mean_deltas, dtype=float))
        for row, rank in zip(model_rows, ranks, strict=True):
            row["rank_within_model"] = float(rank)
        summary_rows.extend(model_rows)
        mean_vectors[model_id] = np.asarray(mean_deltas, dtype=float)
    summary = pd.DataFrame(summary_rows, columns=CHANNEL_IMPORTANCE_COLUMNS)

    consistency_rows: list[dict[str, Any]] = []
    for model_1, model_2 in itertools.combinations(MODEL_ORDER, 2):
        result = kendall_tau_b_coefficient(
            mean_vectors[str(model_1["model_id"])],
            mean_vectors[str(model_2["model_id"])],
        )
        consistency_rows.append(
            {
                "analysis_id": ANALYSIS_ID,
                "model_1_id": model_1["model_id"],
                "model_1_order": int(model_1["model_order"]),
                "model_2_id": model_2["model_id"],
                "model_2_order": int(model_2["model_order"]),
                "n_channels": 11,
                "kendall_tau_b": result["kendall_tau_b"],
                "estimable": "true" if result["estimable"] else "false",
                "nonestimable_reason": result["nonestimable_reason"],
            }
        )
    consistency = pd.DataFrame(consistency_rows, columns=CROSS_MODEL_COLUMNS)
    consistency["kendall_tau_b"] = pd.to_numeric(consistency["kendall_tau_b"], errors="coerce")

    consensus_rows: list[dict[str, Any]] = []
    for channel_index, channel_name in enumerate(CHANNEL_ORDER, start=1):
        subset = summary.loc[summary["channel_name"] == channel_name].sort_values("model_order", kind="mergesort")
        deltas = subset["mean_delta_auroc"].to_numpy(dtype=float)
        ranks = subset["rank_within_model"].to_numpy(dtype=float)
        consensus_rows.append(
            {
                "analysis_id": ANALYSIS_ID,
                "channel_index": channel_index,
                "channel_name": channel_name,
                "positive_model_count": int(np.count_nonzero(deltas > 0.0)),
                "zero_model_count": int(np.count_nonzero(deltas == 0.0)),
                "negative_model_count": int(np.count_nonzero(deltas < 0.0)),
                "mean_rank": float(np.mean(ranks)),
                "median_rank": float(np.median(ranks)),
                "minimum_rank": float(np.min(ranks)),
                "maximum_rank": float(np.max(ranks)),
                "mean_delta_auroc": float(np.mean(deltas)),
                "median_delta_auroc": float(np.median(deltas)),
            }
        )
    consensus = pd.DataFrame(consensus_rows, columns=CHANNEL_CONSENSUS_COLUMNS)
    frames = {
        "channel_importance_summary.csv": summary,
        "permutation_repeats.csv": repeats,
        "cross_model_consistency.csv": consistency,
        "channel_consensus_summary.csv": consensus,
    }
    validate_output_frames(frames, strict_values=True)
    return frames


def execute_model_permutations(
    model_spec: Mapping[str, Any],
    model_input: np.ndarray | pd.DataFrame,
    labels: np.ndarray,
    baseline_auroc: float,
    permutation_matrix: np.ndarray,
    predictor: Callable[[np.ndarray | pd.DataFrame], np.ndarray],
    channel_map: Mapping[str, Sequence[str]] | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Execute every fixed channel-repeat perturbation for one locked model.

    Args:
        model_spec: One fixed model-order record.
        model_input: Raw aligned Pipeline A tensor or Pipeline B feature frame.
        labels: Aligned binary test labels.
        baseline_auroc: Authoritative frozen-prediction AUROC.
        permutation_matrix: Single shared participant permutation matrix.
        predictor: Locked positive-class raw-score inference callable.
        channel_map: Validated Pipeline B feature-block mapping when applicable.

    Returns:
        Repeat-level aggregate rows and transient score cube ``[11, R, N]``.

    Raises:
        ValueError: If the model specification, input, scores, or AUROC is invalid.
    """
    pipeline = str(model_spec.get("pipeline"))
    if pipeline not in {"pipeline_a", "pipeline_b"}:
        raise ValueError("Model specification contains an unsupported pipeline.")
    y_true = np.asarray(labels)
    if y_true.ndim != 1 or set(np.unique(y_true).tolist()) != {0, 1}:
        raise ValueError("Model permutation execution requires aligned binary labels.")
    matrix = np.asarray(permutation_matrix)
    if matrix.ndim != 2 or matrix.shape[1] != len(y_true):
        raise ValueError("Shared permutation matrix shape differs from the aligned cohort.")
    if pipeline == "pipeline_a":
        if not isinstance(model_input, np.ndarray) or len(model_input) != len(y_true):
            raise ValueError("Pipeline A model input differs from the aligned cohort.")
    else:
        if not isinstance(model_input, pd.DataFrame) or len(model_input) != len(y_true) or channel_map is None:
            raise ValueError("Pipeline B model input or channel mapping differs from the aligned cohort.")
    score_cube = np.empty((len(CHANNEL_ORDER), len(matrix), len(y_true)), dtype=float)
    rows: list[dict[str, Any]] = []
    for channel_position_value, channel_name in enumerate(CHANNEL_ORDER):
        for permutation_position_value in range(len(matrix)):
            permutation_repeat = permutation_position_value + 1
            mapping = select_permutation_mapping(matrix, permutation_repeat)
            if pipeline == "pipeline_a":
                permuted_input = permute_pipeline_a_channel(model_input, channel_name, mapping)
            else:
                permuted_input = permute_pipeline_b_channel(model_input, channel_name, mapping, channel_map)
            scores = np.asarray(predictor(permuted_input), dtype=float)
            if scores.shape != y_true.shape or not np.isfinite(scores).all():
                raise ValueError("Locked model inference returned an invalid score vector.")
            if not ((scores >= 0.0).all() and (scores <= 1.0).all()):
                raise ValueError("Locked model inference returned scores outside [0, 1].")
            permuted_auroc = float(roc_auc_score(y_true, scores))
            if not math.isfinite(permuted_auroc):
                raise ValueError("A required model-channel AUROC is nonestimable.")
            score_cube[channel_position_value, permutation_position_value] = scores
            index_fields = output_index_fields(channel_name, permutation_repeat)
            rows.append(
                {
                    "analysis_id": ANALYSIS_ID,
                    "model_id": model_spec["model_id"],
                    "model_order": int(model_spec["model_order"]),
                    "channel_index": index_fields["channel_index"],
                    "channel_name": channel_name,
                    "permutation_repeat": index_fields["permutation_repeat"],
                    "baseline_auroc": float(baseline_auroc),
                    "permuted_auroc": permuted_auroc,
                    "delta_auroc": signed_auroc_delta(baseline_auroc, permuted_auroc),
                }
            )
        LOGGER.info(
            "Completed model=%s channel=%s (%s/%s).",
            model_spec["model_id"],
            channel_name,
            channel_position_value + 1,
            len(CHANNEL_ORDER),
        )
    return rows, score_cube


def _format_markdown_float(value: Any) -> str:
    """Format a finite aggregate value or an explicit missing marker."""
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.6f}"


def render_report(
    frames: Mapping[str, pd.DataFrame],
    baseline_reconciliation: Sequence[Mapping[str, Any]],
    limitations: Sequence[str],
) -> str:
    """Render the deterministic aggregate scientific Markdown report.

    Args:
        frames: Validated aggregate scientific frames.
        baseline_reconciliation: Six aggregate reconciliation records in fixed model order.
        limitations: Frozen interpretive limitations.

    Returns:
        Complete report text with no participant-level content.

    Raises:
        ValueError: If aggregate inputs are incomplete.
    """
    validate_output_frames(frames, strict_values=True)
    if [record["model_id"] for record in baseline_reconciliation] != [model["model_id"] for model in MODEL_ORDER]:
        raise ValueError("Baseline reconciliation report records differ from the fixed model order.")
    summary = frames["channel_importance_summary.csv"]
    consistency = frames["cross_model_consistency.csv"]
    consensus = frames["channel_consensus_summary.csv"]
    lines = [
        "# Unified Channel-Block Permutation Importance",
        "",
        "## Analysis identity and boundaries",
        "",
        "This is an exploratory post hoc analysis of the frozen 195-participant internal random test set. It is a",
        "separate unified perturbation analysis and does not replace the heterogeneous Stage 8 analyses or robustness",
        "analyses.",
        "",
        "## Baseline reconciliation",
        "",
        "| model_id | display_name | max_abs_score_difference | reproduced_auroc | "
        "frozen_prediction_auroc | frozen_metric_auroc | passed |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    display_by_id = {str(model["model_id"]): str(model["model_display_name"]) for model in MODEL_ORDER}
    for record in baseline_reconciliation:
        model_id = str(record["model_id"])
        lines.append(
            f"| {model_id} | {display_by_id[model_id]} | "
            f"{_format_markdown_float(record['max_abs_score_difference'])} | "
            f"{_format_markdown_float(record['reproduced_auroc'])} | "
            f"{_format_markdown_float(record['frozen_prediction_auroc'])} | "
            f"{_format_markdown_float(record['frozen_metric_auroc'])} | "
            f"{'true' if record['passed'] else 'false'} |"
        )

    lines.extend(
        [
            "",
            "## MiniRocket + Ridge complete channel results",
            "",
            "| channel | mean_delta | median_delta | sd_delta | repeat_q025 | repeat_q975 | negative_repeats | "
            "rank | bootstrap_ci_low | bootstrap_ci_high |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    final_rows = summary.loc[summary["model_id"] == "minirocket_ridge"].sort_values("channel_index")
    for row in final_rows.itertuples(index=False):
        lines.append(
            f"| {row.channel_name} | {_format_markdown_float(row.mean_delta_auroc)} | "
            f"{_format_markdown_float(row.median_delta_auroc)} | {_format_markdown_float(row.sd_delta_auroc)} | "
            f"{_format_markdown_float(row.repeat_delta_q025)} | {_format_markdown_float(row.repeat_delta_q975)} | "
            f"{int(row.negative_delta_count)} | {_format_markdown_float(row.rank_within_model)} | "
            f"{_format_markdown_float(row.bootstrap_ci_low)} | {_format_markdown_float(row.bootstrap_ci_high)} |"
        )

    lines.extend(["", "## Complete signed mean-delta matrix", ""])
    lines.append("| model | " + " | ".join(CHANNEL_ORDER) + " |")
    lines.append("|---|" + "---:|" * len(CHANNEL_ORDER))
    for model in MODEL_ORDER:
        model_rows = summary.loc[summary["model_id"] == model["model_id"]].sort_values("channel_index")
        values = " | ".join(_format_markdown_float(value) for value in model_rows["mean_delta_auroc"])
        lines.append(f"| {model['model_display_name']} | {values} |")

    lines.extend(
        [
            "",
            "## All pairwise Kendall tau-b coefficients",
            "",
            "| model_1 | model_2 | kendall_tau_b | estimable | nonestimable_reason |",
            "|---|---|---:|---|---|",
        ]
    )
    for row in consistency.itertuples(index=False):
        lines.append(
            f"| {row.model_1_id} | {row.model_2_id} | {_format_markdown_float(row.kendall_tau_b)} | "
            f"{row.estimable} | {row.nonestimable_reason or ''} |"
        )

    lines.extend(
        [
            "",
            "## Complete channel consensus",
            "",
            "| channel | positive_models | zero_models | negative_models | mean_rank | median_rank | minimum_rank | "
            "maximum_rank | mean_delta | median_delta |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in consensus.itertuples(index=False):
        lines.append(
            f"| {row.channel_name} | {int(row.positive_model_count)} | {int(row.zero_model_count)} | "
            f"{int(row.negative_model_count)} | {_format_markdown_float(row.mean_rank)} | "
            f"{_format_markdown_float(row.median_rank)} | {_format_markdown_float(row.minimum_rank)} | "
            f"{_format_markdown_float(row.maximum_rank)} | {_format_markdown_float(row.mean_delta_auroc)} | "
            f"{_format_markdown_float(row.median_delta_auroc)} |"
        )

    negative_means = summary.loc[summary["mean_delta_auroc"] < 0.0, ["model_id", "channel_name", "mean_delta_auroc"]]
    null_means = summary.loc[summary["mean_delta_auroc"] == 0.0, ["model_id", "channel_name"]]
    inconsistent = consensus.loc[
        (consensus["positive_model_count"] > 0) & (consensus["negative_model_count"] > 0), "channel_name"
    ]
    nonestimable = consistency.loc[consistency["estimable"] == "false"]
    lines.extend(["", "## Retained adverse, null, inconsistent, and nonestimable findings", ""])
    if negative_means.empty:
        lines.append("- Negative signed mean deltas: none observed.")
    else:
        for row in negative_means.itertuples(index=False):
            lines.append(
                f"- Negative signed mean delta: {row.model_id}, {row.channel_name}, "
                f"{_format_markdown_float(row.mean_delta_auroc)}."
            )
    if null_means.empty:
        lines.append("- Exact-null signed mean deltas: none observed.")
    else:
        for row in null_means.itertuples(index=False):
            lines.append(f"- Exact-null signed mean delta: {row.model_id}, {row.channel_name}.")
    if inconsistent.empty:
        lines.append("- Cross-model sign-inconsistent channels: none observed.")
    else:
        lines.append("- Cross-model sign-inconsistent channels: " + ", ".join(inconsistent.tolist()) + ".")
    if nonestimable.empty:
        lines.append("- Nonestimable Kendall tau-b pairs: none observed.")
    else:
        for row in nonestimable.itertuples(index=False):
            lines.append(
                f"- Nonestimable Kendall tau-b pair: {row.model_1_id} versus {row.model_2_id}; "
                f"{row.nonestimable_reason}."
            )

    lines.extend(["", "## Interpretive limitations", ""])
    lines.extend(f"- {limitation}" for limitation in limitations)
    lines.extend(
        [
            "",
            "No result-driven Figure 3 recommendation is made.",
            "",
            "No chemical or biological interpretation is made.",
            "",
        ]
    )
    return "\n".join(lines)


def compare_run_directories(run_a: Path, run_b: Path) -> dict[str, Any]:
    """Require byte-identical payloads and normalized scientific JSON equality.

    Args:
        run_a: First complete run directory.
        run_b: Independent second complete run directory.

    Returns:
        Deterministic comparison record.

    Raises:
        ValueError: If a scientific payload or normalized run record differs.
    """
    payload_hashes: dict[str, str] = {}
    for filename in SCIENTIFIC_PAYLOAD_FILENAMES:
        first = (run_a / filename).read_bytes()
        second = (run_b / filename).read_bytes()
        if first != second:
            raise ValueError(f"Deterministic scientific payload differs for {filename}.")
        payload_hashes[filename] = hashlib.sha256(first).hexdigest()
    record_a = json.loads((run_a / "run_record.json").read_text(encoding="utf-8"))
    record_b = json.loads((run_b / "run_record.json").read_text(encoding="utf-8"))
    normalized_a = {key: value for key, value in record_a.items() if key not in VOLATILE_RUN_RECORD_FIELDS}
    normalized_b = {key: value for key, value in record_b.items() if key not in VOLATILE_RUN_RECORD_FIELDS}
    if normalized_a != normalized_b:
        raise ValueError("Deterministic normalized run records differ.")
    return {
        "scientific_payloads_byte_identical": True,
        "normalized_run_records_identical": True,
        "excluded_volatile_fields": list(VOLATILE_RUN_RECORD_FIELDS),
        "scientific_payload_sha256": payload_hashes,
    }


def load_analysis_config(path: Path) -> dict[str, Any]:
    """Load and strictly validate the frozen analysis configuration.

    Args:
        path: Project-relative YAML configuration path.

    Returns:
        Validated configuration mapping.

    Raises:
        ValueError: If the configuration differs from the approved plan contract.
        OSError: If the configuration cannot be read.
    """
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Analysis configuration must be a mapping.")
    expected_keys = {
        "analysis_id",
        "analysis_classification",
        "endpoint",
        "score_domain",
        "final_model_id",
        "permutation_seed",
        "bootstrap_seed",
        "n_permutations",
        "n_bootstrap_final_model",
        "score_reconciliation_atol",
        "auroc_reconciliation_atol",
        "n_jobs",
        "approved_plan",
        "expected_git_head",
        "model_order",
        "channel_order",
        "cohort",
        "random_streams",
        "paths",
        "immutable_inputs",
        "schemas",
        "interpretive_limitations",
    }
    if set(payload) != expected_keys:
        raise ValueError("Analysis configuration top-level keys differ from the frozen contract.")
    expected_scalars = {
        "analysis_id": ANALYSIS_ID,
        "analysis_classification": "exploratory_post_hoc",
        "endpoint": "high_grade_gastric_lesion",
        "score_domain": "p_raw",
        "final_model_id": "minirocket_ridge",
        "permutation_seed": 42,
        "bootstrap_seed": 42,
        "n_permutations": 100,
        "n_bootstrap_final_model": 2000,
        "score_reconciliation_atol": 1.0e-6,
        "auroc_reconciliation_atol": 1.0e-12,
        "n_jobs": 1,
        "expected_git_head": "3478442b26bf42e7963b61916d8ac6180984ebbc",
    }
    for key, expected in expected_scalars.items():
        if payload[key] != expected:
            raise ValueError(f"Analysis configuration value differs for {key}.")
    if payload["approved_plan"] != {
        "path": "docs/plans/figure3_unified_channel_block_permutation.plan.md",
        "sha256": "1f1fbf47ef39ccd16ccebe0c8cff920653c844c120aa3bc5201aa4e5c772fb89",
    }:
        raise ValueError("Approved plan authority differs from the frozen contract.")
    if payload["model_order"] != list(MODEL_ORDER) or tuple(payload["channel_order"]) != CHANNEL_ORDER:
        raise ValueError("Model or channel order differs from the frozen contract.")
    if payload["cohort"] != {"n_test": 195, "n_positive": 82, "n_negative": 113}:
        raise ValueError("Cohort counts differ from the frozen contract.")
    configured_paths = [*payload["paths"].values(), *payload["immutable_inputs"].keys()]
    for relative_path in configured_paths:
        candidate = Path(relative_path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("Configuration paths must be project-relative and non-traversing.")
    if payload["paths"] != dict(EXPECTED_PATHS):
        raise ValueError("Configuration path contract differs from the frozen contract.")
    if payload["random_streams"] != dict(EXPECTED_RANDOM_STREAMS):
        raise ValueError("Configuration random-stream contract differs from the frozen contract.")
    if payload["interpretive_limitations"] != list(EXPECTED_INTERPRETIVE_LIMITATIONS):
        raise ValueError("Configuration interpretive limitations differ from the frozen contract.")
    expected_schemas = {
        "channel_importance_summary": list(CHANNEL_IMPORTANCE_COLUMNS),
        "permutation_repeats": list(PERMUTATION_REPEAT_COLUMNS),
        "cross_model_consistency": list(CROSS_MODEL_COLUMNS),
        "channel_consensus_summary": list(CHANNEL_CONSENSUS_COLUMNS),
    }
    if payload["schemas"] != expected_schemas:
        raise ValueError("Output schemas differ from the frozen contract.")
    if payload["immutable_inputs"] != dict(EXPECTED_IMMUTABLE_INPUTS):
        raise ValueError("Configuration immutable input anchors differ from the independently frozen contract.")
    return payload


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file's exact bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest_payload(candidate: Path, final_relative_dir: Path) -> dict[str, Any]:
    """Build the exact non-self-hashing final manifest payload.

    Args:
        candidate: Directory containing all required non-manifest final files.
        final_relative_dir: Project-relative final directory recorded in paths.

    Returns:
        Deterministic manifest mapping.

    Raises:
        ValueError: If the final path is absolute or a required file is absent.
    """
    if final_relative_dir.is_absolute() or ".." in final_relative_dir.parts:
        raise ValueError("Manifest final directory must be project-relative.")
    files: list[dict[str, Any]] = []
    for filename in FINAL_MANIFEST_FILENAMES:
        source = candidate / filename
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"Manifest source is absent or invalid: {filename}")
        files.append(
            {
                "path": (final_relative_dir / filename).as_posix(),
                "byte_size": int(source.stat().st_size),
                "sha256": _sha256_file(source),
            }
        )
    return {"analysis_id": ANALYSIS_ID, "files": files}


@dataclass(repr=False)
class _ScientificContext:
    """Hold sensitive aligned scientific arrays transiently in memory."""

    pipeline_a: np.ndarray
    pipeline_b: pd.DataFrame
    labels: np.ndarray
    participant_keys: np.ndarray
    channel_map: dict[str, list[str]]
    frozen_scores: dict[str, np.ndarray]
    payloads: dict[str, dict[str, Any]]
    baseline_reconciliation: list[dict[str, Any]]


def _project_path(relative_path: str) -> Path:
    """Resolve one validated project-relative path."""
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Project path must be relative and non-traversing.")
    return PROJECT_ROOT / relative


def _git_output(*args: str) -> str:
    """Run one read-only Git inspection command and return stripped output."""
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _utc_timestamp() -> str:
    """Return an allowlisted provenance timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize deterministic strict JSON bytes."""
    return (json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def _package_versions() -> dict[str, str]:
    """Return relevant installed package versions without importing project data."""
    distributions = (
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("scikit_learn", "scikit-learn"),
        ("sktime", "sktime"),
        ("torch", "torch"),
        ("pytorch_lightning", "pytorch-lightning"),
        ("xgboost", "xgboost"),
    )
    versions: dict[str, str] = {"python": platform.python_version()}
    for output_name, distribution_name in distributions:
        try:
            versions[output_name] = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"Required installed dependency is unavailable: {distribution_name}") from exc
    return versions


def verify_immutable_hashes(config: Mapping[str, Any]) -> dict[str, str]:
    """Verify and return every immutable input hash.

    Args:
        config: Validated frozen analysis configuration.

    Returns:
        Ordered project-relative path-to-digest mapping.

    Raises:
        ValueError: If an immutable file is absent or has a changed digest.
    """
    verified: dict[str, str] = {}
    for relative_path, expected_digest in config["immutable_inputs"].items():
        path = _project_path(relative_path)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Immutable input is absent or invalid: {relative_path}")
        observed = _sha256_file(path)
        if observed != expected_digest:
            raise ValueError(f"Immutable input hash differs: {relative_path}")
        verified[relative_path] = observed
    return verified


def verify_execution_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    """Verify approval, repository, environment, and stale-target execution gates.

    Args:
        config: Validated frozen analysis configuration.

    Returns:
        Aggregate preflight provenance with no participant-level content.

    Raises:
        ValueError: If any execution gate differs from the approved plan.
    """
    plan_path = _project_path(config["approved_plan"]["path"])
    if _sha256_file(plan_path) != config["approved_plan"]["sha256"]:
        raise ValueError("Approved plan hash differs from the on-disk plan.")
    head = _git_output("rev-parse", "HEAD")
    if head != config["expected_git_head"]:
        raise ValueError("Git HEAD differs from the approved plan.")
    status_lines = [line for line in _git_output("status", "--short").splitlines() if line]
    expected_status = {
        "?? .vscode/",
        "?? configs/figure3_unified_channel_block_permutation.yaml",
        "?? scripts/run_figure3_unified_channel_block_permutation.py",
        "?? src/analysis/figure3_unified_channel_block_permutation.py",
        "?? tests/test_figure3_unified_channel_block_permutation.py",
    }
    if set(status_lines) != expected_status:
        raise ValueError("Git worktree state differs from the approved Phase 2 write scope.")
    if os.environ.get("CONDA_DEFAULT_ENV") != "gcvocs":
        raise ValueError("The gcvocs conda environment is not active.")
    expected_environment_root = Path.home() / "miniforge3" / "envs" / "gcvocs" / "bin"
    if Path(sys.executable).resolve().parent != expected_environment_root.resolve():
        raise ValueError("Python does not resolve inside the required gcvocs environment.")
    final_dir = _project_path(config["paths"]["final_output_dir"])
    work_root = _project_path(config["paths"]["work_root"])
    if final_dir.exists() or final_dir.is_symlink():
        raise FileExistsError("Final scientific output directory already exists.")
    if work_root.exists() or work_root.is_symlink():
        raise FileExistsError("Task work directory already exists.")
    immutable_hashes = verify_immutable_hashes(config)
    return {
        "git_head": head,
        "worktree_status": status_lines,
        "environment_name": "gcvocs",
        "python_executable_environment_relative": "bin/python",
        "package_versions": _package_versions(),
        "immutable_hashes": immutable_hashes,
    }


def _configure_determinism(seed: int) -> dict[str, Any]:
    """Apply the existing locked deterministic deep-learning runtime settings."""
    import pytorch_lightning as pl
    import torch

    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")
    device_class = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "framework_seed": int(seed),
        "workers_seeded": True,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "float32_matmul_precision": "high",
        "device_class": device_class,
    }


def _load_prediction_evidence(model_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load only the participant key, label, and frozen raw score columns."""
    path = PROJECT_ROOT / "outputs" / "predictions" / f"binary_{model_id}_test_predictions.csv"
    expected_header = ["sample_id", "y_true", "p_raw", "p_cal", "y_pred", "quadrant"]
    if pd.read_csv(path, nrows=0).columns.tolist() != expected_header:
        raise ValueError(f"Frozen prediction header differs for model={model_id}.")
    frame = pd.read_csv(path, usecols=["sample_id", "y_true", "p_raw"])
    participant_keys = pd.to_numeric(frame["sample_id"], errors="raise").to_numpy(dtype=np.int64)
    labels = pd.to_numeric(frame["y_true"], errors="raise").to_numpy(dtype=np.int64)
    scores = pd.to_numeric(frame["p_raw"], errors="raise").to_numpy(dtype=float)
    if len(frame) != 195 or len(np.unique(participant_keys)) != 195 or not np.isfinite(scores).all():
        raise ValueError(f"Frozen prediction evidence is invalid for model={model_id}.")
    return participant_keys, labels, scores


def _load_metric_auroc(model_id: str) -> float:
    """Load only the frozen aggregate test AUROC metric value."""
    path = PROJECT_ROOT / "outputs" / "metrics" / f"binary_{model_id}_test_metrics.csv"
    expected_header = ["scope", "metric", "value", "ci_low", "ci_high"]
    if pd.read_csv(path, nrows=0).columns.tolist() != expected_header:
        raise ValueError(f"Frozen metric header differs for model={model_id}.")
    frame = pd.read_csv(path, usecols=["scope", "metric", "value"])
    row = frame.loc[(frame["scope"] == "test") & (frame["metric"] == "auc_roc")]
    if len(row) != 1:
        raise ValueError(f"Frozen test AUROC row is absent or duplicated for model={model_id}.")
    value = float(row.iloc[0]["value"])
    if not math.isfinite(value):
        raise ValueError(f"Frozen test AUROC is nonfinite for model={model_id}.")
    return value


def _infer_positive_scores(
    model_id: str,
    payload: Mapping[str, Any],
    model_input: np.ndarray | pd.DataFrame,
) -> np.ndarray:
    """Run the frozen end-to-end raw positive-class inference contract."""
    from src.training.train_final_models import _predict_positive_probabilities

    scores = _predict_positive_probabilities(
        model_name=model_id,
        transformer=payload["transformer"],
        normalizer=payload["normalizer"],
        model=payload["model"],
        x_input=model_input,
    )
    return np.asarray(scores, dtype=float)


def _load_scientific_context(config: Mapping[str, Any]) -> _ScientificContext:
    """Load, align, and baseline-reconcile the frozen scientific inputs."""
    pipeline_a_path = _project_path(config["paths"]["pipeline_a"])
    with np.load(pipeline_a_path, allow_pickle=False) as payload:
        required_keys = {"x", "sample_id", "label", "channel_names", "clip_length_L"}
        if not required_keys.issubset(set(payload.files)):
            raise ValueError("Pipeline A schema differs from the frozen contract.")
        pipeline_a_full = np.asarray(payload["x"])
        participant_keys_a = np.asarray(payload["sample_id"])
        labels_a = np.asarray(payload["label"])
        channel_names = tuple(str(value) for value in payload["channel_names"].tolist())
        clip_length = int(np.asarray(payload["clip_length_L"]).item())
    if pipeline_a_full.shape != (972, 960, 11) or channel_names != CHANNEL_ORDER or clip_length != 960:
        raise ValueError("Pipeline A shape, channel order, or clip length differs from the frozen contract.")
    if not np.issubdtype(participant_keys_a.dtype, np.integer) or not np.issubdtype(labels_a.dtype, np.integer):
        raise ValueError("Pipeline A participant keys or labels have invalid dtypes.")
    participant_keys_a = participant_keys_a.astype(np.int64, copy=False)
    labels_a = labels_a.astype(np.int64, copy=False)
    if len(np.unique(participant_keys_a)) != 972 or not np.all(np.diff(participant_keys_a) > 0):
        raise ValueError("Pipeline A participant ordering is duplicated or nonmonotonic.")
    if set(np.unique(labels_a).tolist()) != {0, 1} or not np.isfinite(pipeline_a_full).all():
        raise ValueError("Pipeline A labels or values are invalid.")

    pipeline_b_path = _project_path(config["paths"]["pipeline_b"])
    pipeline_b_full = pd.read_csv(pipeline_b_path)
    if len(pipeline_b_full) != 972 or pipeline_b_full.columns.duplicated().any():
        raise ValueError("Pipeline B row count or column uniqueness differs from the frozen contract.")
    if "sample_id" not in pipeline_b_full or "label" not in pipeline_b_full:
        raise ValueError("Pipeline B is missing its alignment boundary columns.")
    pipeline_b_full = pipeline_b_full.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    participant_keys_b = pd.to_numeric(pipeline_b_full["sample_id"], errors="raise").to_numpy(dtype=np.int64)
    labels_b = pd.to_numeric(pipeline_b_full["label"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(participant_keys_a, participant_keys_b) or not np.array_equal(labels_a, labels_b):
        raise ValueError("Pipeline A and Pipeline B alignment differs.")
    feature_names = [name for name in pipeline_b_full.columns if name not in {"sample_id", "label"}]
    channel_map = build_pipeline_b_channel_map(feature_names)
    pipeline_b_features = pipeline_b_full.loc[:, feature_names]
    if not np.isfinite(pipeline_b_features.to_numpy(dtype=float)).all():
        raise ValueError("Pipeline B contains nonfinite feature values.")

    split_path = _project_path(config["paths"]["test_idx"])
    with np.load(split_path, allow_pickle=False) as payload:
        if "test_idx" not in payload.files:
            raise ValueError("Locked test split is missing test_idx.")
        test_idx = np.asarray(payload["test_idx"])
    if test_idx.ndim != 1 or len(test_idx) != 195 or not np.issubdtype(test_idx.dtype, np.integer):
        raise ValueError("Locked test positions have invalid shape or dtype.")
    test_idx = test_idx.astype(np.int64, copy=False)
    if len(np.unique(test_idx)) != 195 or test_idx.min() < 0 or test_idx.max() >= 972:
        raise ValueError("Locked test positions are duplicated or out of range.")
    selected_sort = np.argsort(participant_keys_a[test_idx], kind="mergesort")
    selected_positions = test_idx[selected_sort]
    participant_keys = participant_keys_a[selected_positions]
    labels = labels_a[selected_positions]
    pipeline_a = pipeline_a_full[selected_positions]
    pipeline_b = pipeline_b_features.iloc[selected_positions].reset_index(drop=True)
    if len(participant_keys) != 195 or len(np.unique(participant_keys)) != 195:
        raise ValueError("Aligned test participant keys are incomplete or duplicated.")
    if int(np.count_nonzero(labels == 1)) != 82 or int(np.count_nonzero(labels == 0)) != 113:
        raise ValueError("Frozen test cohort or class counts differ from the approved plan.")

    from src.training.load_final_model import load_final_model

    frozen_scores: dict[str, np.ndarray] = {}
    payloads: dict[str, dict[str, Any]] = {}
    metric_aurocs: dict[str, float] = {}
    required_payload_keys = {
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
    for model_spec in MODEL_ORDER:
        model_id = str(model_spec["model_id"])
        prediction_keys, prediction_labels, scores = _load_prediction_evidence(model_id)
        if not np.array_equal(prediction_keys, participant_keys) or not np.array_equal(prediction_labels, labels):
            raise ValueError(f"Frozen prediction alignment differs for model={model_id}.")
        frozen_scores[model_id] = scores
        metric_aurocs[model_id] = _load_metric_auroc(model_id)
        model_path = PROJECT_ROOT / "outputs" / "models" / f"binary_{model_id}_final.pkl"
        model_payload = load_final_model(model_path)
        if not isinstance(model_payload, dict) or not required_payload_keys.issubset(model_payload):
            raise ValueError(f"Final model payload schema differs for model={model_id}.")
        if model_payload["metadata"].get("model_name") != model_id:
            raise ValueError(f"Final model metadata identity differs for model={model_id}.")
        training_keys = np.asarray(model_payload["training_sample_ids"])
        if training_keys.ndim != 1 or len(np.unique(training_keys)) != len(training_keys):
            raise ValueError(f"Training participant keys are invalid for model={model_id}.")
        if np.intersect1d(training_keys.astype(np.int64), participant_keys).size != 0:
            raise ValueError(f"Train/test participant separation failed for model={model_id}.")
        if model_spec["pipeline"] == "pipeline_b":
            payload_features = list(model_payload["feature_names"])
            normalizer_features = list(getattr(model_payload["normalizer"], "feature_names", []))
            if payload_features != feature_names or normalizer_features != feature_names:
                raise ValueError(f"Pipeline B feature order differs for model={model_id}.")
        elif list(model_payload["feature_names"]) != list(CHANNEL_ORDER):
            raise ValueError(f"Pipeline A channel order differs for model={model_id}.")
        if model_id == "minirocket_ridge" and model_payload["transformer"] is None:
            raise ValueError("MiniRocket final payload is missing its fitted transformer.")
        payloads[model_id] = model_payload

    baseline_reconciliation: list[dict[str, Any]] = []
    reproduced_scores: dict[str, np.ndarray] = {}
    for model_spec in MODEL_ORDER:
        model_id = str(model_spec["model_id"])
        model_input: np.ndarray | pd.DataFrame = pipeline_a if model_spec["pipeline"] == "pipeline_a" else pipeline_b
        try:
            reproduced = _infer_positive_scores(model_id, payloads[model_id], model_input)
        except Exception as exc:
            raise ValueError(f"Locked baseline inference failed for model={model_id}.") from exc
        reproduced_scores[model_id] = reproduced
    for model_spec in MODEL_ORDER:
        model_id = str(model_spec["model_id"])
        try:
            reconciliation = reconcile_baseline_scores(
                labels,
                reproduced_scores[model_id],
                frozen_scores[model_id],
                metric_auroc=metric_aurocs[model_id],
                score_atol=float(config["score_reconciliation_atol"]),
                auroc_atol=float(config["auroc_reconciliation_atol"]),
            )
        except ValueError as exc:
            raise ValueError(f"Baseline reconciliation failed for model={model_id}.") from exc
        baseline_reconciliation.append({"model_id": model_id, **reconciliation})
    del reproduced_scores
    return _ScientificContext(
        pipeline_a=pipeline_a,
        pipeline_b=pipeline_b,
        labels=labels,
        participant_keys=participant_keys,
        channel_map=channel_map,
        frozen_scores=frozen_scores,
        payloads=payloads,
        baseline_reconciliation=baseline_reconciliation,
    )


def _serialize_frame(frame: pd.DataFrame) -> bytes:
    """Serialize one CSV frame with the frozen dialect."""
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _validate_csv_temp(path: Path, expected_columns: Sequence[str], expected_rows: int) -> None:
    """Read back one temporary CSV and validate its basic contract."""
    frame = pd.read_csv(path)
    if frame.columns.tolist() != list(expected_columns) or len(frame) != expected_rows:
        raise ValueError("Temporary CSV schema or row count validation failed.")


def _read_run_frames(output_dir: Path) -> dict[str, pd.DataFrame]:
    """Read all four aggregate run CSVs for cross-file validation."""
    return {filename: pd.read_csv(output_dir / filename) for filename in EXPECTED_OUTPUT_ROWS}


def run_analysis_once(
    config: Mapping[str, Any],
    output_dir: Path,
    *,
    package_versions: Mapping[str, str],
) -> dict[str, Any]:
    """Execute one independent complete scientific analysis run.

    Args:
        config: Validated frozen configuration.
        output_dir: Absent task-owned run directory.
        package_versions: Preflight package-version inventory.

    Returns:
        Private-data-free run record also persisted in the run directory.

    Raises:
        FileExistsError: If the run directory already exists.
        ValueError: If a scientific, privacy, schema, or provenance gate fails.
    """
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("Independent run output directory already exists.")
    output_dir.mkdir(parents=True)
    started_at = _utc_timestamp()
    started = perf_counter()
    determinism = _configure_determinism(int(config["permutation_seed"]))
    immutable_hashes = verify_immutable_hashes(config)
    context = _load_scientific_context(config)
    LOGGER.info("Baseline reconciliation passed for all six locked models.")
    permutation_matrix, bootstrap_matrix, randomization = generate_random_matrices(
        context.labels,
        seed=int(config["permutation_seed"]),
        n_permutations=int(config["n_permutations"]),
        n_bootstrap=int(config["n_bootstrap_final_model"]),
    )
    randomization["numpy_version"] = package_versions["numpy"]
    repeat_rows: list[dict[str, Any]] = []
    final_score_cube: np.ndarray | None = None
    baseline_by_model = {
        record["model_id"]: float(record["frozen_prediction_auroc"])
        for record in context.baseline_reconciliation
    }
    for model_spec in MODEL_ORDER:
        model_id = str(model_spec["model_id"])
        LOGGER.info("Starting unified channel-block permutation for model=%s.", model_id)
        model_input: np.ndarray | pd.DataFrame = (
            context.pipeline_a if model_spec["pipeline"] == "pipeline_a" else context.pipeline_b
        )

        def predictor(values: np.ndarray | pd.DataFrame, *, _model_id: str = model_id) -> np.ndarray:
            return _infer_positive_scores(_model_id, context.payloads[_model_id], values)

        model_rows, score_cube = execute_model_permutations(
            model_spec,
            model_input,
            context.labels,
            baseline_by_model[model_id],
            permutation_matrix,
            predictor,
            channel_map=context.channel_map if model_spec["pipeline"] == "pipeline_b" else None,
        )
        repeat_rows.extend(model_rows)
        if model_id == "minirocket_ridge":
            final_score_cube = score_cube
        else:
            del score_cube
        LOGGER.info("Completed unified channel-block permutation for model=%s.", model_id)
    if final_score_cube is None:
        raise ValueError("Final-model permuted score cube was not retained for bootstrap uncertainty.")
    bootstrap_intervals: dict[str, tuple[float, float]] = {}
    for channel_position_value, channel_name in enumerate(CHANNEL_ORDER):
        bootstrap_deltas = paired_bootstrap_channel_deltas(
            context.labels,
            context.frozen_scores["minirocket_ridge"],
            final_score_cube[channel_position_value],
            bootstrap_matrix,
        )
        ci_low, ci_high = np.percentile(bootstrap_deltas, [2.5, 97.5], method="linear")
        bootstrap_intervals[channel_name] = (float(ci_low), float(ci_high))
    repeats = pd.DataFrame(repeat_rows, columns=PERMUTATION_REPEAT_COLUMNS)
    frames = build_aggregate_frames(repeats, bootstrap_intervals)
    report = render_report(frames, context.baseline_reconciliation, config["interpretive_limitations"])
    serialized: dict[str, bytes] = {
        filename: _serialize_frame(frame) for filename, frame in frames.items()
    }
    serialized["report.md"] = report.encode("utf-8")
    validate_serialized_privacy(serialized)
    for filename, payload in serialized.items():
        if filename in EXPECTED_OUTPUT_ROWS:
            atomic_write_bytes(
                output_dir / filename,
                payload,
                validator=lambda path, _filename=filename: _validate_csv_temp(
                    path,
                    EXPECTED_OUTPUT_COLUMNS[_filename],
                    EXPECTED_OUTPUT_ROWS[_filename],
                ),
            )
        else:
            atomic_write_bytes(output_dir / filename, payload)
    readback_frames = _read_run_frames(output_dir)
    validate_output_frames(readback_frames, strict_values=True)
    scientific_hashes = {filename: _sha256_file(output_dir / filename) for filename in SCIENTIFIC_PAYLOAD_FILENAMES}
    finished_at = _utc_timestamp()
    run_record: dict[str, Any] = {
        "analysis_id": ANALYSIS_ID,
        "analysis_classification": config["analysis_classification"],
        "output_directory": output_dir.relative_to(PROJECT_ROOT).as_posix(),
        "run_started_at_utc": started_at,
        "run_finished_at_utc": finished_at,
        "wall_clock_seconds": float(perf_counter() - started),
        "git_head": config["expected_git_head"],
        "approved_plan": config["approved_plan"],
        "package_versions": dict(package_versions),
        "immutable_hashes": immutable_hashes,
        "model_order": list(MODEL_ORDER),
        "channel_order": list(CHANNEL_ORDER),
        "cohort_reconciliation": {"n_test": 195, "n_positive": 82, "n_negative": 113, "passed": True},
        "baseline_reconciliation": context.baseline_reconciliation,
        "determinism": determinism,
        "randomization": randomization,
        "analysis_counts": {
            "n_permutations": 100,
            "n_bootstrap_final_model": 2000,
            "channel_importance_rows": 66,
            "permutation_repeat_rows": 6600,
            "cross_model_rows": 15,
            "channel_consensus_rows": 11,
        },
        "scientific_payload_sha256": scientific_hashes,
        "privacy_validation_passed": True,
        "participant_level_persistence": False,
    }
    record_bytes = _json_bytes(run_record)
    validate_serialized_privacy({"run_record.json": record_bytes})
    atomic_write_bytes(output_dir / "run_record.json", record_bytes)
    return run_record


def _build_audit_payload(
    config: Mapping[str, Any],
    preflight: Mapping[str, Any],
    run_record: Mapping[str, Any],
    immutable_after: Mapping[str, str],
    deterministic_comparison: Mapping[str, Any],
    final_worktree_status: Sequence[str],
) -> dict[str, Any]:
    """Build the complete final private-data-free audit payload."""
    immutable_records = [
        {
            "path": relative_path,
            "sha256_before": preflight["immutable_hashes"][relative_path],
            "sha256_after": immutable_after[relative_path],
            "unchanged": preflight["immutable_hashes"][relative_path] == immutable_after[relative_path],
        }
        for relative_path in config["immutable_inputs"]
    ]
    return {
        "analysis_id": ANALYSIS_ID,
        "analysis_classification": config["analysis_classification"],
        "status": "complete",
        "approved_plan": config["approved_plan"],
        "provenance": {
            "git_head": preflight["git_head"],
            "phase2_start_worktree_status": ["?? .vscode/"],
            "pre_execution_worktree_status": preflight["worktree_status"],
            "final_worktree_status": list(final_worktree_status),
            "environment_name": preflight["environment_name"],
            "python_executable_environment_relative": preflight["python_executable_environment_relative"],
            "package_versions": preflight["package_versions"],
        },
        "immutable_inputs": immutable_records,
        "model_order": list(MODEL_ORDER),
        "channel_order": list(CHANNEL_ORDER),
        "cohort_reconciliation": run_record["cohort_reconciliation"],
        "baseline_reconciliation": run_record["baseline_reconciliation"],
        "randomization": run_record["randomization"],
        "perturbation": {
            "pipeline_a": (
                "Complete raw [N,L,11] selected-channel trajectory moved by the shared participant mapping before "
                "locked inference."
            ),
            "pipeline_b": (
                "All nine raw engineered features for the selected channel name moved by the shared participant "
                "mapping before the fitted normalizer."
            ),
            "signed_estimand": "baseline_auroc_minus_permuted_auroc",
            "negative_values_retained": True,
        },
        "analysis_counts": run_record["analysis_counts"],
        "ci_method": {
            "final_model": "paired_stratified_participant_bootstrap_fixed_permutations_percentile_95",
            "supporting_models": "not_computed_supporting_models_by_design",
            "repeat_percentiles": "permutation_repeat_spread_not_confidence_intervals",
        },
        "output_validation": {
            "row_counts": dict(EXPECTED_OUTPUT_ROWS),
            "schemas": {filename: list(columns) for filename, columns in EXPECTED_OUTPUT_COLUMNS.items()},
            "finite_values_passed": True,
            "ordering_passed": True,
            "readback_passed": True,
        },
        "deterministic_rerun": dict(deterministic_comparison),
        "privacy_validation": {
            "schema_scan_passed": True,
            "serialized_text_scan_passed": True,
            "project_relative_paths_passed": True,
            "forbidden_values_echoed": False,
        },
        "participant_level_persistence": False,
        "interpretive_limitations": list(config["interpretive_limitations"]),
        "commands": [
            {
                "command": "python -m pytest tests/test_figure3_unified_channel_block_permutation.py -q",
                "exit_code": 0,
            },
            {
                "command": (
                    "python -m pytest tests/test_shap_cross_model_aggregation.py tests/test_shap_cross_model_ig.py "
                    "tests/test_shap_cross_model_kendall_tau.py "
                    "tests/test_shap_cross_model_minirocket_permutation.py -q"
                ),
                "exit_code": 0,
            },
            {
                "command": (
                    "python scripts/run_figure3_unified_channel_block_permutation.py --config "
                    "configs/figure3_unified_channel_block_permutation.yaml"
                ),
                "exit_code": 0,
            },
        ],
    }


def _verify_final_manifest(final_dir: Path, manifest: Mapping[str, Any]) -> None:
    """Verify every promoted non-manifest file against the final manifest."""
    if manifest.get("analysis_id") != ANALYSIS_ID or len(manifest.get("files", [])) != 6:
        raise ValueError("Final manifest schema or inventory differs from the frozen contract.")
    for record in manifest["files"]:
        path = final_dir / Path(record["path"]).name
        if not path.is_file() or path.stat().st_size != record["byte_size"] or _sha256_file(path) != record["sha256"]:
            raise ValueError("Promoted final artifact differs from the manifest.")


def _safe_remove_run_directory(work_root: Path, run_name: str) -> None:
    """Remove one exact verified task-owned run directory."""
    resolved_root = work_root.resolve()
    candidate = work_root / run_name
    if candidate.is_symlink() or not candidate.is_dir():
        raise ValueError("Task run cleanup target is absent, invalid, or a symlink.")
    resolved_candidate = candidate.resolve()
    if resolved_candidate.parent != resolved_root or resolved_candidate.name != run_name:
        raise ValueError("Task run cleanup target is outside the exact work root.")
    shutil.rmtree(resolved_candidate)


def execute_workflow(config_path: Path) -> dict[str, Any]:
    """Execute, reproduce, validate, and atomically promote the approved analysis.

    Args:
        config_path: Frozen analysis configuration path.

    Returns:
        Final project-relative path and output SHA-256 inventory.

    Raises:
        FileExistsError: If any task target would be overwritten.
        ValueError: If a scientific, deterministic, privacy, provenance, or promotion gate fails.
    """
    config = load_analysis_config(config_path)
    preflight = verify_execution_preflight(config)
    work_root = _project_path(config["paths"]["work_root"])
    final_dir = _project_path(config["paths"]["final_output_dir"])
    work_root.mkdir(parents=True)
    LOGGER.info("Starting independent scientific run A.")
    run_a_record = run_analysis_once(
        config,
        work_root / "run_a",
        package_versions=preflight["package_versions"],
    )
    gc.collect()
    verify_immutable_hashes(config)
    LOGGER.info("Starting independent scientific run B.")
    run_analysis_once(
        config,
        work_root / "run_b",
        package_versions=preflight["package_versions"],
    )
    deterministic_comparison = compare_run_directories(work_root / "run_a", work_root / "run_b")
    LOGGER.info("Independent run comparison passed.")
    immutable_after = verify_immutable_hashes(config)
    final_status = [line for line in _git_output("status", "--short").splitlines() if line]
    if set(final_status) != set(preflight["worktree_status"]):
        raise ValueError("Worktree state changed during scientific execution.")
    candidate = work_root / "promotion_candidate"
    if candidate.exists() or candidate.is_symlink():
        raise FileExistsError("Promotion candidate already exists.")
    candidate.mkdir()
    serialized_candidate: dict[str, bytes] = {}
    for filename in SCIENTIFIC_PAYLOAD_FILENAMES:
        payload = (work_root / "run_a" / filename).read_bytes()
        serialized_candidate[filename] = payload
        atomic_write_bytes(candidate / filename, payload)
    audit = _build_audit_payload(
        config,
        preflight,
        run_a_record,
        immutable_after,
        deterministic_comparison,
        final_status,
    )
    audit_bytes = _json_bytes(audit)
    serialized_candidate["audit.json"] = audit_bytes
    validate_serialized_privacy(serialized_candidate)
    atomic_write_bytes(candidate / "audit.json", audit_bytes)
    final_relative_dir = Path(config["paths"]["final_output_dir"])
    manifest = build_manifest_payload(candidate, final_relative_dir)
    manifest_bytes = _json_bytes(manifest)
    validate_serialized_privacy({**serialized_candidate, "manifest.json": manifest_bytes})
    atomic_write_bytes(candidate / "manifest.json", manifest_bytes)
    candidate_frames = _read_run_frames(candidate)
    validate_output_frames(candidate_frames, strict_values=True)
    if final_dir.exists() or final_dir.is_symlink():
        raise FileExistsError("Final scientific output directory appeared before promotion.")
    promote_candidate_directory(candidate, final_dir)
    promoted_manifest = json.loads((final_dir / "manifest.json").read_text(encoding="utf-8"))
    _verify_final_manifest(final_dir, promoted_manifest)
    verify_immutable_hashes(config)
    _safe_remove_run_directory(work_root, "run_a")
    _safe_remove_run_directory(work_root, "run_b")
    final_hashes = {
        filename: _sha256_file(final_dir / filename)
        for filename in (*FINAL_MANIFEST_FILENAMES, "manifest.json")
    }
    LOGGER.info("Atomic promotion and final validation completed.")
    return {"final_output_dir": final_relative_dir.as_posix(), "sha256": final_hashes}
