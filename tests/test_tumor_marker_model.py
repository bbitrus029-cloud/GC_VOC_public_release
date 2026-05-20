from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest


def _import_model_module_or_fail():
    """Import the Session 13 tumor-marker model module or fail clearly."""
    try:
        return importlib.import_module("src.analysis.tumor_marker_model")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Missing module src.analysis.tumor_marker_model: {exc}", pytrace=False)


def _import_common_module_or_fail():
    """Import the Session 13 common helper module or fail clearly."""
    try:
        return importlib.import_module("src.analysis.stage7_part_c_common")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Missing module src.analysis.stage7_part_c_common: {exc}", pytrace=False)


def test_complete_case_subsets_filter_synthetic_fixture_correctly() -> None:
    """Synthetic complete-case filtering should respect train/test partitions and marker missingness."""
    common = _import_common_module_or_fail()
    frame = pd.DataFrame(
        {
            "sample_id": [1, 2, 3, 4, 5, 6],
            "label": [1, 0, 1, 0, 1, 0],
            "cea": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "ca199": [1.0, 2.0, 3.0, 4.0, np.nan, 6.0],
            "ca724": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan],
            "age": [60, 61, 62, 63, 64, 65],
            "sex": [0, 1, 0, 1, 0, 1],
            "bmi": [22.1, 23.2, 24.3, 25.4, 26.5, 27.6],
            "smoking": [0, 1, 0, 1, 0, 1],
            "drinking": [1, 0, 1, 0, 1, 0],
            "family_history": [0, 0, 1, 1, 0, 0],
            "hp_infection": [1, 1, 0, 0, 1, np.nan],
            "disease_category": [0, 1, 2, 3, 4, 5],
            "collection_location": [0, 0, 1, 1, 1, 0],
            "p_raw": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "p_cal": [0.11, 0.21, 0.31, 0.41, 0.51, 0.61],
        }
    )

    train_frame, test_frame = common._build_complete_case_subsets(
        cohort_frame=frame,
        train_idx=np.asarray([0, 1, 2, 3]),
        test_idx=np.asarray([4, 5]),
    )

    assert train_frame["sample_id"].tolist() == [1, 2, 4]
    assert test_frame["sample_id"].tolist() == []


def test_logistic_regression_fit_helper_rejects_nan_inputs_before_estimator_fit() -> None:
    """Any estimator fit helper must guard against NaN features before calling fit()."""
    module = _import_model_module_or_fail()

    class RecordingEstimator:
        def __init__(self) -> None:
            self.seen_nan = None

        def fit(self, x_matrix: np.ndarray, y_vector: np.ndarray) -> "RecordingEstimator":
            self.seen_nan = bool(np.isnan(x_matrix).any())
            return self

    estimator = RecordingEstimator()
    feature_frame = pd.DataFrame(
        {
            "cea": [1.0, 2.0, 3.0],
            "ca199": [4.0, 5.0, 6.0],
            "ca724": [7.0, 8.0, 9.0],
        }
    )
    y_true = np.asarray([0, 1, 1], dtype=int)

    module._fit_logistic_regression_model(
        estimator=estimator,
        feature_frame=feature_frame,
        y_true=y_true,
    )

    assert estimator.seen_nan is False


def test_selection_bias_smd_formulas_match_continuous_and_binary_definitions() -> None:
    """Continuous and binary SMD helpers should follow the locked formulas exactly."""
    common = _import_common_module_or_fail()
    continuous_a = pd.Series([1.0, 2.0, 3.0, 4.0])
    continuous_b = pd.Series([2.0, 3.0, 4.0, 5.0])
    binary_a = pd.Series([1, 1, 0, 0])
    binary_b = pd.Series([1, 0, 0, 0])

    expected_continuous = abs(continuous_a.mean() - continuous_b.mean()) / np.sqrt(
        (continuous_a.var(ddof=1) + continuous_b.var(ddof=1)) / 2.0
    )
    p_a = float(binary_a.mean())
    p_b = float(binary_b.mean())
    expected_binary = abs(p_a - p_b) / np.sqrt(
        ((p_a * (1.0 - p_a)) + (p_b * (1.0 - p_b))) / 2.0
    )

    assert common._compute_continuous_smd(continuous_a, continuous_b) == pytest.approx(expected_continuous)
    assert common._compute_binary_smd(binary_a, binary_b) == pytest.approx(expected_binary)


def test_selection_bias_variable_list_matches_plan_exactly() -> None:
    """Selection-bias SMD variables must match the prespecified 10-variable list exactly."""
    common = _import_common_module_or_fail()

    assert list(common.SELECTION_BIAS_VARIABLES) == [
        "age",
        "sex",
        "bmi",
        "smoking",
        "drinking",
        "family_history",
        "hp_infection",
        "disease_category",
        "collection_location",
        "label",
    ]
