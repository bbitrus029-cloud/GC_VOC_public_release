"""End-to-end smoke test for Stage 9 reproducibility verification.

Verifies that for all six final PKLs, deterministic inference on a 10-sample
subset (5 negatives + 5 positives) produces a `[10, 2]` probability matrix
with finite values in `[0, 1]`, rows summing to `1.0`, valid TP/TN/FP/FN
quadrants, and a finite calibrated probability vector.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.reproducibility_verification import (
    PIPELINE_A_MODELS,
    PIPELINE_B_MODELS,
    SMOKE_TEST_N_PER_CLASS,
    SMOKE_TEST_N_TOTAL,
    _load_pipeline_a_tensor,
    _load_pipeline_b_frame,
    _run_one_model_smoke,
    _select_smoke_subset,
)
from src.config import MODEL_NAMES, PROJECT_ROOT


@pytest.fixture(scope="module")
def smoke_subset() -> tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    """Build the 10-sample subset once per module."""
    pipeline_a_path = PROJECT_ROOT / "data" / "processed" / "sequences" / "pipeline_a_raw.npz"
    pipeline_b_path = PROJECT_ROOT / "data" / "processed" / "features" / "pipeline_b_raw.csv"
    test_idx_path = PROJECT_ROOT / "data" / "processed" / "splits" / "test_idx.npz"

    x_pipeline_a, sample_ids, labels = _load_pipeline_a_tensor(pipeline_a_path)
    x_pipeline_b_frame, sample_ids_b, labels_b = _load_pipeline_b_frame(pipeline_b_path)
    assert np.array_equal(sample_ids, sample_ids_b), "Pipeline A/B sample_ids mismatch."
    assert np.array_equal(labels, labels_b), "Pipeline A/B labels mismatch."

    with np.load(test_idx_path, allow_pickle=False) as split_file:
        test_idx = split_file["test_idx"].astype(np.int64)

    selected_positions, selected_sample_ids, selected_labels = _select_smoke_subset(
        test_idx=test_idx,
        sample_ids=sample_ids,
        labels=labels,
    )
    x_pipeline_a_subset = x_pipeline_a[selected_positions]
    x_pipeline_b_subset = x_pipeline_b_frame.iloc[selected_positions].reset_index(drop=True)
    return x_pipeline_a_subset, x_pipeline_b_subset, selected_sample_ids, selected_labels


def test_smoke_subset_geometry(
    smoke_subset: tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray],
) -> None:
    """The deterministic subset has 5 positives + 5 negatives in sample_id order."""
    x_pipeline_a_subset, x_pipeline_b_subset, sample_ids, labels = smoke_subset
    assert x_pipeline_a_subset.shape == (SMOKE_TEST_N_TOTAL, 960, 11)
    assert len(x_pipeline_b_subset) == SMOKE_TEST_N_TOTAL
    assert sample_ids.shape == (SMOKE_TEST_N_TOTAL,)
    assert labels.shape == (SMOKE_TEST_N_TOTAL,)
    assert int(labels.sum()) == SMOKE_TEST_N_PER_CLASS
    assert int((1 - labels).sum()) == SMOKE_TEST_N_PER_CLASS
    assert np.array_equal(sample_ids, np.sort(sample_ids))


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_smoke_one_model_end_to_end(
    model_name: str,
    smoke_subset: tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray],
) -> None:
    """Each model's PKL runs end-to-end inference on the 10-sample subset."""
    x_pipeline_a_subset, x_pipeline_b_subset, sample_ids, labels = smoke_subset
    result = _run_one_model_smoke(
        model_name=model_name,
        x_pipeline_a_subset=x_pipeline_a_subset,
        x_pipeline_b_subset=x_pipeline_b_subset,
        sample_ids=sample_ids,
        labels=labels,
    )
    assert result["model"] == model_name
    assert result["n_samples"] == SMOKE_TEST_N_TOTAL
    assert tuple(result["shape"]) == (SMOKE_TEST_N_TOTAL, 2)
    assertions = result["assertions"]
    assert assertions["shape_ok"], f"{model_name}: probability shape != (10, 2)"
    assert assertions["all_finite"], f"{model_name}: non-finite probabilities"
    assert assertions["all_in_range"], f"{model_name}: probabilities outside [0, 1]"
    assert assertions["all_rows_sum_to_one"], f"{model_name}: rows do not sum to 1.0"
    assert assertions["all_quadrants_valid"], f"{model_name}: quadrant labels outside {{TP,TN,FP,FN}}"
    assert assertions["p_cal_finite"], f"{model_name}: non-finite calibrated probabilities"
    assert assertions["p_cal_in_range"], f"{model_name}: calibrated probabilities outside [0, 1]"
    assert result["passed"], f"{model_name}: overall smoke test did not pass"


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_smoke_pipeline_assignment_matches_config(model_name: str) -> None:
    """Each of the six models is classified as Pipeline A or Pipeline B exactly once."""
    is_a = model_name in PIPELINE_A_MODELS
    is_b = model_name in PIPELINE_B_MODELS
    assert is_a ^ is_b, f"{model_name} must belong to exactly one pipeline."


def test_quadrant_derivation_consistency(
    smoke_subset: tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray],
) -> None:
    """`y_pred = (p_raw >= threshold_p_raw)` produces self-consistent quadrants on MR+R."""
    x_pipeline_a_subset, x_pipeline_b_subset, sample_ids, labels = smoke_subset
    result = _run_one_model_smoke(
        model_name="minirocket_ridge",
        x_pipeline_a_subset=x_pipeline_a_subset,
        x_pipeline_b_subset=x_pipeline_b_subset,
        sample_ids=sample_ids,
        labels=labels,
    )
    quadrants = result["quadrants"]
    for y_true, y_pred, quadrant in zip(result["y_true"], result["y_pred"], quadrants, strict=True):
        if y_true == 1 and y_pred == 1:
            assert quadrant == "TP"
        elif y_true == 0 and y_pred == 0:
            assert quadrant == "TN"
        elif y_true == 0 and y_pred == 1:
            assert quadrant == "FP"
        else:
            assert quadrant == "FN"
