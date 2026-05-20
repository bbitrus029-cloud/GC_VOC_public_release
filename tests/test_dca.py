from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest


def _import_module_or_fail():
    """Import the DCA module or fail with a readable message."""
    try:
        return importlib.import_module("src.analysis.dca")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Missing module src.analysis.dca: {exc}", pytrace=False)


def test_compute_net_benefit_at_threshold_matches_formula() -> None:
    """Model and treat-all net benefit should match the prespecified formula."""
    module = _import_module_or_fail()
    y_true = np.array([1, 1, 0, 0], dtype=int)
    p_raw = np.array([0.9, 0.8, 0.7, 0.1], dtype=float)

    result = module._compute_net_benefit_at_threshold(
        y_true=y_true,
        p_raw=p_raw,
        threshold=0.5,
        prevalence=0.5,
    )

    assert result["model"] == pytest.approx(0.25)
    assert result["treat_all"] == pytest.approx(0.0)
    assert result["treat_none"] == pytest.approx(0.0)


def test_treat_none_curve_is_identically_zero() -> None:
    """Treat-none net benefit should be zero for every threshold and scenario."""
    module = _import_module_or_fail()
    y_true = np.array([1, 0, 1, 0], dtype=int)
    p_raw = np.array([0.9, 0.2, 0.8, 0.1], dtype=float)

    output_frame = module._build_dca_output_frame(
        y_true=y_true,
        p_raw=p_raw,
        threshold_grid=module._build_threshold_grid(),
        prevalence_scenarios=[0.01, 0.05, 0.10],
    )

    treat_none_rows = output_frame.loc[output_frame["curve"] == "treat_none"]
    assert not treat_none_rows.empty
    assert (treat_none_rows["net_benefit"] == 0.0).all()


def test_threshold_grid_has_locked_length_and_bounds() -> None:
    """The DCA threshold grid should match the prespecified 0.01-to-0.98 range."""
    module = _import_module_or_fail()

    threshold_grid = module._build_threshold_grid()

    assert len(threshold_grid) == 98
    assert threshold_grid[0] == pytest.approx(0.01)
    assert threshold_grid[-1] == pytest.approx(0.98)


def test_build_dca_output_frame_has_expected_long_schema() -> None:
    """The persisted DCA table should use the required long-format columns."""
    module = _import_module_or_fail()
    y_true = np.array([1, 0, 1, 0], dtype=int)
    p_raw = np.array([0.9, 0.2, 0.8, 0.1], dtype=float)

    output_frame = module._build_dca_output_frame(
        y_true=y_true,
        p_raw=p_raw,
        threshold_grid=np.array([0.1, 0.2], dtype=float),
        prevalence_scenarios=[0.01],
    )

    assert isinstance(output_frame, pd.DataFrame)
    assert output_frame.columns.tolist() == [
        "prevalence_scenario",
        "threshold",
        "curve",
        "net_benefit",
    ]
    assert set(output_frame["curve"]) == {"model", "treat_all", "treat_none"}
