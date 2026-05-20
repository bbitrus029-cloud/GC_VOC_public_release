from __future__ import annotations

import importlib

import pandas as pd
import pytest


def _import_module_or_fail():
    """Import the stratified-robustness module or fail with a readable message."""
    try:
        return importlib.import_module("src.analysis.stratified_robustness")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Missing module src.analysis.stratified_robustness: {exc}", pytrace=False)


def test_exactly_seven_stratification_variables_are_defined() -> None:
    """Stratified robustness should cover the locked seven variables and no others."""
    module = _import_module_or_fail()

    assert tuple(module.STRATIFICATION_VARIABLES) == (
        "sex",
        "age_at_60",
        "bmi_at_24",
        "smoking",
        "drinking",
        "family_history",
        "hp_infection",
    )


def test_gate_stratum_returns_omission_reason_for_small_cells() -> None:
    """Failed gating should retain the row and record why the stratum was omitted."""
    module = _import_module_or_fail()

    omission_reason = module._gate_stratum(n_total=9, n_pos=1, n_neg=8)

    assert omission_reason == "n_total=9 < 10; n_pos=1 < 2"


def test_gate_stratum_accepts_valid_cell_sizes() -> None:
    """A stratum meeting all prespecified minimum counts should pass gating."""
    module = _import_module_or_fail()

    omission_reason = module._gate_stratum(n_total=12, n_pos=4, n_neg=8)

    assert omission_reason == ""


def test_hp_infection_encoding_produces_three_strata() -> None:
    """Hp infection should keep 0, 1, and missing as distinct strata."""
    module = _import_module_or_fail()
    series = pd.Series([0.0, 1.0, None, 1.0, 0.0], dtype=float)

    encoded = module._encode_hp_infection_strata(series)

    assert encoded.tolist() == ["0", "1", "missing", "1", "0"]
    assert set(encoded.tolist()) == {"0", "1", "missing"}


def test_build_stratified_output_frame_has_expected_schema() -> None:
    """Each per-variable CSV should retain omission_reason alongside metric fields."""
    module = _import_module_or_fail()

    output_frame = module._build_stratified_output_frame(
        metric_rows=[
            {
                "model": "minirocket_ridge",
                "stratum_value": "female",
                "n_total": 20,
                "n_pos": 8,
                "n_neg": 12,
                "metric_name": "auc_roc",
                "estimate": 0.9,
                "ci_low": 0.8,
                "ci_high": 1.0,
                "omission_reason": "",
            }
        ]
    )

    assert output_frame.columns.tolist() == [
        "model",
        "stratum_value",
        "n_total",
        "n_pos",
        "n_neg",
        "metric_name",
        "estimate",
        "ci_low",
        "ci_high",
        "omission_reason",
    ]
