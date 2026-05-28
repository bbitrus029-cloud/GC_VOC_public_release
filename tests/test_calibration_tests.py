from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2, norm


EXPECTED_TEST_COLUMNS = ["sample_id", "y_true", "p_raw", "p_cal", "y_pred", "quadrant"]


def _import_module_or_fail():
    """Import the calibration-tests module or fail with a readable message."""
    try:
        return importlib.import_module("src.analysis.calibration_tests")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Missing module src.analysis.calibration_tests: {exc}", pytrace=False)


def _ten_group_fixture() -> pd.DataFrame:
    """Build a hand-computable fixture with one observation per HL decile."""
    probabilities = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95], dtype=float)
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64)
    predictions = (probabilities >= 0.50).astype(np.int64)
    return pd.DataFrame(
        {
            "sample_id": np.arange(1, 11, dtype=np.int64),
            "y_true": labels,
            "p_raw": probabilities,
            "p_cal": probabilities,
            "y_pred": predictions,
            "quadrant": ["TN", "TN", "TN", "TN", "TN", "TP", "TP", "TP", "TP", "TP"],
        },
        columns=EXPECTED_TEST_COLUMNS,
    )


def _collapsed_group_fixture() -> pd.DataFrame:
    """Build a fixture whose duplicated probabilities collapse qcut bins."""
    probabilities = np.array([0.10, 0.10, 0.20, 0.20, 0.30, 0.30, 0.40, 0.40, 0.50, 0.50], dtype=float)
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    predictions = (probabilities >= 0.30).astype(np.int64)
    quadrants = ["TN", "FN", "TN", "FN", "FP", "TP", "FP", "TP", "FP", "TP"]
    return pd.DataFrame(
        {
            "sample_id": np.arange(1, 11, dtype=np.int64),
            "y_true": labels,
            "p_raw": probabilities,
            "p_cal": probabilities,
            "y_pred": predictions,
            "quadrant": quadrants,
        },
        columns=EXPECTED_TEST_COLUMNS,
    )


def test_run_calibration_tests_on_frame_is_bit_identical() -> None:
    """Two runs on the same fixture should return identical statistics."""
    module = _import_module_or_fail()
    frame = _ten_group_fixture()

    first = module.run_calibration_tests_on_frame(frame=frame, hosmer_lemeshow_groups=10)
    second = module.run_calibration_tests_on_frame(frame=frame, hosmer_lemeshow_groups=10)

    assert first == second


def test_hosmer_lemeshow_statistic_matches_hand_computed_fixture() -> None:
    """HL statistic should match the hand-computed ten-decile fixture."""
    module = _import_module_or_fail()
    frame = _ten_group_fixture()

    statistic, degrees_of_freedom, p_value = module.compute_hosmer_lemeshow_test(
        y_true=frame["y_true"].to_numpy(dtype=np.int64),
        p_cal=frame["p_cal"].to_numpy(dtype=float),
        hosmer_lemeshow_groups=10,
    )

    assert statistic == pytest.approx(3.838157714318705, rel=0.0, abs=1e-12)
    assert degrees_of_freedom == 8
    assert p_value == pytest.approx(chi2.sf(statistic, 8), rel=0.0, abs=1e-15)


def test_spiegelhalter_components_match_hand_computed_fixture() -> None:
    """Observed Brier, expected Brier, variance, and z should match the fixture."""
    module = _import_module_or_fail()
    frame = _ten_group_fixture()

    observed_brier, expected_brier, variance, z_statistic, p_value = module.compute_spiegelhalter_z_test(
        y_true=frame["y_true"].to_numpy(dtype=np.int64),
        p_cal=frame["p_cal"].to_numpy(dtype=float),
    )

    assert observed_brier == pytest.approx(0.0825, rel=0.0, abs=1e-15)
    assert expected_brier == pytest.approx(0.1675, rel=0.0, abs=1e-15)
    assert variance == pytest.approx(0.0034155, rel=0.0, abs=1e-15)
    assert z_statistic == pytest.approx(-1.4544265066281457, rel=0.0, abs=1e-12)
    assert p_value == pytest.approx(2.0 * norm.sf(abs(z_statistic)), rel=0.0, abs=1e-15)


def test_hosmer_lemeshow_p_value_uses_df_8() -> None:
    """Plan 1 fixes the HL p-value degrees of freedom at 8."""
    module = _import_module_or_fail()
    frame = _ten_group_fixture()

    statistic, degrees_of_freedom, p_value = module.compute_hosmer_lemeshow_test(
        y_true=frame["y_true"].to_numpy(dtype=np.int64),
        p_cal=frame["p_cal"].to_numpy(dtype=float),
        hosmer_lemeshow_groups=10,
    )

    assert degrees_of_freedom == 8
    assert p_value == pytest.approx(chi2.sf(statistic, 8), rel=0.0, abs=1e-15)


def test_hosmer_lemeshow_halts_when_qcut_yields_fewer_than_ten_groups() -> None:
    """Duplicate probabilities must trigger the prespecified <10-group halt."""
    module = _import_module_or_fail()
    frame = _collapsed_group_fixture()

    with pytest.raises(ValueError, match="fewer than 10 groups"):
        module.run_calibration_tests_on_frame(frame=frame, hosmer_lemeshow_groups=10)


def test_p_cal_range_guard_raises_for_invalid_probabilities() -> None:
    """Probabilities outside [0, 1] must trigger the Plan 1 guard."""
    module = _import_module_or_fail()
    frame = _ten_group_fixture()
    frame.loc[0, "p_cal"] = 1.10

    with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
        module.run_calibration_tests_on_frame(frame=frame, hosmer_lemeshow_groups=10)
