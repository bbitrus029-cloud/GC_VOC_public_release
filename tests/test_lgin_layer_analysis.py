from __future__ import annotations

import importlib
import re

import pandas as pd
import pytest


def _import_module_or_fail():
    """Import the LGIN-layer-analysis module or fail with a readable message."""
    try:
        return importlib.import_module("src.analysis.lgin_layer_analysis")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Missing module src.analysis.lgin_layer_analysis: {exc}", pytrace=False)


def test_rank_biserial_matches_mann_whitney_formula() -> None:
    """Rank-biserial correlation should equal 2U / (n1*n2) - 1."""
    module = _import_module_or_fail()

    rb = module._compute_rank_biserial_from_u(u_statistic=5.0, n_group_b=3, n_group_c=2)

    assert rb == pytest.approx((2.0 * 5.0 / (3 * 2)) - 1.0)


def test_within_b_stability_gating_retains_small_cells_as_omitted_rows() -> None:
    """Within-B stability should keep rows for underpowered levels with omission reasons."""
    module = _import_module_or_fail()
    b_group_frame = pd.DataFrame(
        {
            "sample_id": [1, 2, 3, 4, 5, 6],
            "p_cal": [0.91, 0.87, 0.82, 0.78, 0.75, 0.74],
            "age": [61, 64, 63, 62, 59, 58],
            "sex": [0, 0, 0, 0, 0, 1],
            "bmi": [25.0, 25.5, 26.0, 24.5, 23.0, 22.0],
            "smoking": [1, 1, 1, 1, 1, 0],
            "drinking": [0, 0, 0, 0, 0, 1],
            "family_history": [0, 0, 0, 0, 0, 1],
            "hp_infection": [1, 1, 1, 1, 1, 0],
        }
    )

    stability_frame = module._build_within_b_stability_frame(
        b_group_frame=b_group_frame,
        age_cutpoint=60,
        bmi_cutpoint=24,
        threshold_probability=0.5,
    )

    omitted_row = stability_frame.loc[
        (stability_frame["variable_name"] == "sex") & (stability_frame["level_value"] == "1")
    ].iloc[0]
    assert omitted_row["omission_reason"] == "n_in_level=1 < 5"


def test_lgin_within_b_output_schema_is_locked() -> None:
    """Within-B stability CSV companions should retain the required descriptive columns."""
    module = _import_module_or_fail()

    output_frame = module._build_within_b_output_frame(
        [
            {
                "variable_name": "sex",
                "level_value": "0",
                "n_in_level": 5,
                "median_p_cal": 0.8,
                "n_above_0_5": 5,
                "omission_reason": "",
            }
        ]
    )

    assert output_frame.columns.tolist() == [
        "variable_name",
        "level_value",
        "n_in_level",
        "median_p_cal",
        "n_above_0_5",
        "omission_reason",
    ]


def test_synthetic_oof_primary_metrics_are_non_degenerate() -> None:
    """A synthetic OOF B/C split with n_B=35 and n_C=4 should yield finite primary metrics."""
    module = _import_module_or_fail()
    b_group = pd.DataFrame({"p_cal": [0.82 + (index * 0.001) for index in range(35)]})
    c_group = pd.DataFrame({"p_cal": [0.15, 0.18, 0.21, 0.24]})

    mw_result = module.mannwhitneyu(
        b_group["p_cal"].to_numpy(dtype=float),
        c_group["p_cal"].to_numpy(dtype=float),
        alternative="two-sided",
    )
    rb = module._compute_rank_biserial_from_u(
        u_statistic=float(mw_result.statistic),
        n_group_b=len(b_group),
        n_group_c=len(c_group),
    )

    assert mw_result.pvalue == mw_result.pvalue
    assert -1.0 <= rb <= 1.0


def test_separation_criterion_gate_logic_handles_edge_cases() -> None:
    """The E012 separation rule should be strict on both the effect-size and p-value thresholds."""
    module = _import_module_or_fail()

    assert module._evaluate_separation_criterion(rank_biserial=0.51, p_two_sided=0.049) is True
    assert module._evaluate_separation_criterion(rank_biserial=0.49, p_two_sided=0.049) is False
    assert module._evaluate_separation_criterion(rank_biserial=0.51, p_two_sided=0.051) is False


def test_test_view_section_declares_non_estimable_when_c_group_is_empty() -> None:
    """The supplementary test-view section should explicitly declare the B/C comparison non-estimable."""
    module = _import_module_or_fail()
    test_frame = pd.DataFrame(
        {
            "sample_id": [1, 2],
            "p_cal": [0.7, 0.8],
            "age": [60, 61],
            "sex": [0, 1],
            "bmi": [24.0, 25.0],
            "smoking": [0, 1],
            "drinking": [0, 0],
            "family_history": [0, 1],
            "hp_infection": [1, 1],
            "cea": [1.0, 2.0],
            "ca199": [3.0, 4.0],
            "ca724": [5.0, 6.0],
            "collection_location": [1, 1],
        }
    )

    section = module._render_test_view_section(test_frame)

    assert "non-estimable" in section


def test_footer_mentions_erratum_e012() -> None:
    """The markdown footer should cite Erratum E012 explicitly."""
    module = _import_module_or_fail()

    footer = module._render_footer(git_commit_hash="abc123", timestamp_utc="2026-05-20T12:00:00Z")

    assert "Erratum E012" in footer


def test_primary_section_lists_c_group_values_individually() -> None:
    """The primary OOF section should list the four C-group calibrated probabilities explicitly."""
    module = _import_module_or_fail()
    primary_frame = module._build_primary_oof_output_frame(
        [
            {
                "rb": 0.75,
                "U": 120.0,
                "p_two_sided": 0.01,
                "median_p_cal_B": 0.81,
                "median_p_cal_C": 0.22,
                "q1_p_cal_B": 0.77,
                "q3_p_cal_B": 0.86,
                "q1_p_cal_C": 0.18,
                "q3_p_cal_C": 0.26,
                "n_above_0.5_B": 35,
                "n_above_0.5_C": 0,
                "n_B": 35,
                "n_C": 4,
                "directional_verification_passed": True,
                "separation_criterion_passed": True,
            }
        ]
    )

    section = module._render_primary_analysis_section(
        primary_frame=primary_frame,
        c_group_values=[0.123, 0.245, 0.187, 0.301],
        confounding_lines=[],
    )

    assert re.search(r"C-group p_cal values:\s*\[[^\]]+\]", section) is not None
