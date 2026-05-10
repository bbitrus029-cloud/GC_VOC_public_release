"""Unit tests for Pipeline A assembly, persistence, and normalization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest

from src.config import GAS_CHANNELS, RANDOM_SEED


def _import_pipeline_a_symbols() -> tuple[Any, ...]:
    """Import Pipeline A symbols lazily so test-first failures stay clear."""
    try:
        from src.config import N_GAS_CHANNELS
        from src.data.pipeline_a import (
            PipelineAOutput,
            apply_pipeline_a_normalization,
            assemble_pipeline_a,
            compute_pipeline_a_stats,
            save_pipeline_a,
        )
    except (AttributeError, ImportError, ModuleNotFoundError) as exc:
        pytest.fail(f"Pipeline A implementation is missing: {exc}", pytrace=False)

    return (
        PipelineAOutput,
        apply_pipeline_a_normalization,
        assemble_pipeline_a,
        compute_pipeline_a_stats,
        save_pipeline_a,
        N_GAS_CHANNELS,
    )


@pytest.fixture
def synthetic_pipeline_a_inputs() -> tuple[list[NDArray[np.float64]], NDArray[np.int64], NDArray[np.int64], int]:
    """Provide deterministic trimmed signals aligned to sorted sample identifiers."""
    _PipelineAOutput, _apply_normalization, _assemble, _compute_stats, _save, n_gas_channels = (
        _import_pipeline_a_symbols()
    )
    rng = np.random.default_rng(RANDOM_SEED)
    clip_length_l = 10
    sample_ids = np.array([101, 103, 107, 109, 113], dtype=np.int64)
    labels = np.array([0, 1, 0, 1, 1], dtype=np.int64)
    signals: list[NDArray[np.float64]] = []

    for sample_index in range(sample_ids.shape[0]):
        signal = rng.normal(loc=0.0, scale=0.25, size=(clip_length_l, n_gas_channels)).astype(np.float64)
        signal[:, 0] -= np.linspace(0.1, 1.1, clip_length_l, dtype=np.float64)
        signal[:, 1] += np.linspace(0.2, 1.2, clip_length_l, dtype=np.float64)
        signal[:, 2] += sample_index * 0.05
        signals.append(signal)

    return signals, sample_ids, labels, clip_length_l


def test_assemble_pipeline_a_returns_expected_shape_alignment_and_signs(
    synthetic_pipeline_a_inputs: tuple[list[NDArray[np.float64]], NDArray[np.int64], NDArray[np.int64], int]
) -> None:
    """Assembly should stack aligned `(L, 11)` matrices without altering signed values."""
    PipelineAOutput, _apply_normalization, assemble_pipeline_a, _compute_stats, _save, _n_gas_channels = (
        _import_pipeline_a_symbols()
    )
    signals, sample_ids, labels, clip_length_l = synthetic_pipeline_a_inputs

    output = assemble_pipeline_a(
        corrected_trimmed_signals=signals,
        sample_ids=sample_ids,
        labels=labels,
        clip_length_L=clip_length_l,
    )

    assert isinstance(output, PipelineAOutput)
    assert output.x.shape == (5, 10, len(GAS_CHANNELS))
    assert output.x.dtype == np.float64
    assert np.array_equal(output.sample_id, sample_ids)
    assert np.array_equal(output.label, labels)
    assert output.channel_names == tuple(GAS_CHANNELS)
    assert output.clip_length_L == clip_length_l
    assert np.min(output.x[..., 0]) < 0.0
    assert np.array_equal(output.x[0], signals[0])


def test_save_pipeline_a_round_trip_preserves_required_arrays(
    tmp_path: Path,
    synthetic_pipeline_a_inputs: tuple[list[NDArray[np.float64]], NDArray[np.int64], NDArray[np.int64], int],
) -> None:
    """Saving should emit the full `.npz` payload with content-equal arrays."""
    _PipelineAOutput, _apply_normalization, assemble_pipeline_a, _compute_stats, save_pipeline_a, _n_gas_channels = (
        _import_pipeline_a_symbols()
    )
    signals, sample_ids, labels, clip_length_l = synthetic_pipeline_a_inputs
    output = assemble_pipeline_a(
        corrected_trimmed_signals=signals,
        sample_ids=sample_ids,
        labels=labels,
        clip_length_L=clip_length_l,
    )
    output_path = tmp_path / "sequences" / "pipeline_a_raw.npz"

    save_pipeline_a(output=output, path=output_path)

    with np.load(output_path, allow_pickle=False) as archive:
        assert set(archive.files) == {"x", "sample_id", "label", "channel_names", "clip_length_L"}
        assert np.array_equal(archive["x"], output.x)
        assert np.array_equal(archive["sample_id"], output.sample_id)
        assert np.array_equal(archive["label"], output.label)
        assert tuple(archive["channel_names"].tolist()) == output.channel_names
        assert int(archive["clip_length_L"]) == output.clip_length_L


def test_compute_pipeline_a_stats_recovers_known_channel_mean_and_std() -> None:
    """Training statistics should reduce over axes `(0, 1)` to per-channel scalars."""
    _PipelineAOutput, _apply_normalization, _assemble, compute_pipeline_a_stats, _save, n_gas_channels = (
        _import_pipeline_a_symbols()
    )
    x_train = np.zeros((2, 4, n_gas_channels), dtype=np.float64)
    x_train[..., 0] = np.array([[3.0, 7.0, 3.0, 7.0], [3.0, 7.0, 3.0, 7.0]], dtype=np.float64)
    x_train[..., 1] = 2.5

    stats = compute_pipeline_a_stats(x_train=x_train)

    assert np.isclose(stats["mean"][0], 5.0)
    assert np.isclose(stats["std"][0], 2.0)
    assert stats["mean"].shape == (n_gas_channels,)
    assert stats["std"].shape == (n_gas_channels,)


def test_apply_pipeline_a_normalization_uses_train_stats_and_preserves_input() -> None:
    """Applying train-only stats should keep a one-sigma test shift centered at one."""
    _PipelineAOutput, apply_pipeline_a_normalization, _assemble, compute_pipeline_a_stats, _save, n_gas_channels = (
        _import_pipeline_a_symbols()
    )
    base = np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=np.float64).reshape(2, 2, 1)
    x_train = np.tile(base, (1, 1, n_gas_channels))
    x_test = x_train + 1.0
    original_test = x_test.copy()

    stats = compute_pipeline_a_stats(x_train=x_train)
    normalized_test = apply_pipeline_a_normalization(x=x_test, stats=stats)

    assert np.array_equal(x_test, original_test)
    assert normalized_test.dtype == np.float64
    assert np.allclose(np.mean(normalized_test, axis=(0, 1)), np.ones(n_gas_channels))


def test_compute_pipeline_a_stats_clamps_zero_std_to_one() -> None:
    """Constant channels should clamp their standard deviation to `1.0`."""
    _PipelineAOutput, apply_pipeline_a_normalization, _assemble, compute_pipeline_a_stats, _save, n_gas_channels = (
        _import_pipeline_a_symbols()
    )
    x_train = np.zeros((2, 3, n_gas_channels), dtype=np.float64)
    x_train[..., 0] = 5.0
    x_train[..., 1] = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], dtype=np.float64)

    stats = compute_pipeline_a_stats(x_train=x_train)
    normalized = apply_pipeline_a_normalization(x=x_train, stats=stats)

    assert stats["std"][0] == pytest.approx(1.0)
    assert np.allclose(normalized[..., 0], x_train[..., 0] - 5.0)
