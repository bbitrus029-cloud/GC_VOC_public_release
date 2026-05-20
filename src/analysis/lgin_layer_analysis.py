"""LGIN exploratory analysis with OOF-primary execution per Erratum E012.

The primary LGIN analysis uses the nested-CV OOF view, not the test view,
under Erratum E012. The OOF CSV stores `p_raw` only, so this module computes
`p_cal` by applying the final PKL's fitted Platt calibrator to the OOF `p_raw`
values. This is internally consistent because the calibrator itself was fit on
OOF predictions in Stage 4b per plan §2.5. No model, split, calibrator, or
threshold is refit here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from src.analysis.stage7_part_a_common import (
    _build_metric_output_frame,
    _format_path,
    _get_git_commit_hash,
    _load_global_config,
    _load_model_payload,
    _load_view_prediction_frame,
    _resolve_path,
    _timestamp_utc,
    _write_text,
)
from src.analysis.stage7_part_b_common import (
    _compute_binary_risk_difference,
    _compute_continuous_smd,
    _load_final_cohort_extended_metadata,
    _merge_predictions_with_optional_metadata,
    _write_dataframe_csv,
)
from src.evaluation.calibration import apply_platt_calibrator


LOGGER = logging.getLogger(__name__)
ERRATUM_E012_SUMMARY = (
    "LGIN primary analysis promoted from test view to OOF view due to test-view n_C=0; "
    "OOF n_C=4 sustains the prespecified analysis with documented power caveats; "
    "test view retained as descriptive sensitivity supplement."
)
HONESTY_PARAGRAPH = (
    "The prespecified LGIN B-vs-C separation analysis is reported on the nested-CV out-of-fold "
    "prediction view (n_LGIN=39, n_B=35, n_C=4) because the locked test set contains zero outpatient "
    "LGIN samples (n_C_test=0), rendering the test-view analysis non-estimable. OOF predictions are "
    "outer-fold-model-unseen by nested CV construction and are statistically equivalent to an "
    "independent prediction set for within-cohort calibration-structure hypothesis testing. Power is "
    "limited by n_C=4; results are reported with explicit transparency on the C-group sample values and "
    "Mann-Whitney null distribution constraints. The test view is reported as a descriptive sensitivity supplement."
)
WITHIN_B_VARIABLES = (
    "age_at_60",
    "sex",
    "bmi_at_24",
    "smoking",
    "drinking",
    "family_history",
    "hp_infection",
)
BASELINE_SMD_VARIABLES = (
    ("age", "continuous"),
    ("sex", "binary"),
    ("bmi", "continuous"),
    ("smoking", "binary"),
    ("drinking", "binary"),
    ("family_history", "binary"),
    ("hp_infection", "binary"),
    ("cea", "continuous"),
    ("ca199", "continuous"),
    ("ca724", "continuous"),
)
EXPECTED_OOF_COUNTS = {"n_lgin": 39, "n_b": 35, "n_c": 4}


def _compute_rank_biserial_from_u(*, u_statistic: float, n_group_b: int, n_group_c: int) -> float:
    """Compute the rank-biserial correlation from a Mann-Whitney U statistic.

    This module passes B-group `p_cal` values as the first argument to
    `mannwhitneyu`, so the convention is `rb = 2U / (n_B * n_C) - 1`.

    Args:
        u_statistic: Mann-Whitney U computed with B-group scores first.
        n_group_b: Number of B-group samples.
        n_group_c: Number of C-group samples.

    Returns:
        Rank-biserial correlation.
    """
    return float((2.0 * u_statistic / (n_group_b * n_group_c)) - 1.0)


def _evaluate_separation_criterion(*, rank_biserial: float, p_two_sided: float) -> bool:
    """Evaluate the E012 separation criterion.

    Args:
        rank_biserial: Rank-biserial correlation.
        p_two_sided: Two-sided Mann-Whitney p-value.

    Returns:
        True when the E012 separation gate passes.
    """
    return bool(abs(rank_biserial) >= 0.5 and p_two_sided < 0.05)


def _encode_hp_level(value: object) -> str:
    """Encode Hp infection levels for within-B stability tables."""
    return "missing" if pd.isna(value) else str(int(value))


def _derive_within_b_columns(b_group_frame: pd.DataFrame, *, age_cutpoint: int, bmi_cutpoint: int) -> pd.DataFrame:
    """Add the seven locked within-B stability variables.

    Args:
        b_group_frame: B-group LGIN dataframe.
        age_cutpoint: Locked age cutpoint.
        bmi_cutpoint: Locked BMI cutpoint.

    Returns:
        Derived dataframe.
    """
    derived = b_group_frame.copy()
    derived["age_at_60"] = derived["age"].apply(lambda value: f"<{age_cutpoint}" if value < age_cutpoint else f">={age_cutpoint}")
    derived["bmi_at_24"] = derived["bmi"].apply(
        lambda value: f"<{bmi_cutpoint}" if value < bmi_cutpoint else f">={bmi_cutpoint}"
    )
    derived["sex"] = derived["sex"].astype(int).astype(str)
    derived["smoking"] = derived["smoking"].astype(int).astype(str)
    derived["drinking"] = derived["drinking"].astype(int).astype(str)
    derived["family_history"] = derived["family_history"].astype(int).astype(str)
    derived["hp_infection"] = derived["hp_infection"].apply(_encode_hp_level)
    return derived


def _build_within_b_stability_frame(
    *,
    b_group_frame: pd.DataFrame,
    age_cutpoint: int,
    bmi_cutpoint: int,
    threshold_probability: float,
) -> pd.DataFrame:
    """Build the within-B stability descriptive table with omission rows.

    Args:
        b_group_frame: LGIN B-group dataframe.
        age_cutpoint: Locked age cutpoint.
        bmi_cutpoint: Locked BMI cutpoint.
        threshold_probability: Fixed probability threshold for descriptive counts above 0.5.

    Returns:
        Long-format within-B stability dataframe.
    """
    derived = _derive_within_b_columns(b_group_frame, age_cutpoint=age_cutpoint, bmi_cutpoint=bmi_cutpoint)
    rows: list[dict[str, Any]] = []
    for variable_name in WITHIN_B_VARIABLES:
        for level_value in sorted(derived[variable_name].astype(str).unique()):
            level_frame = derived.loc[derived[variable_name].astype(str) == level_value].reset_index(drop=True)
            n_in_level = int(len(level_frame))
            omission_reason = "" if n_in_level >= 5 else f"n_in_level={n_in_level} < 5"
            if omission_reason:
                rows.append(
                    {
                        "variable_name": variable_name,
                        "level_value": level_value,
                        "n_in_level": n_in_level,
                        "median_p_cal": float("nan"),
                        "n_above_0_5": float("nan"),
                        "omission_reason": omission_reason,
                    }
                )
                continue
            rows.append(
                {
                    "variable_name": variable_name,
                    "level_value": level_value,
                    "n_in_level": n_in_level,
                    "median_p_cal": float(level_frame["p_cal"].median()),
                    "n_above_0_5": int((level_frame["p_cal"] > threshold_probability).sum()),
                    "omission_reason": "",
                }
            )
    return _build_within_b_output_frame(rows)


def _build_within_b_output_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the within-B stability CSV companion dataframe.

    Args:
        rows: Within-B stability rows.

    Returns:
        Ordered dataframe.
    """
    return _build_metric_output_frame(
        rows,
        ["variable_name", "level_value", "n_in_level", "median_p_cal", "n_above_0_5", "omission_reason"],
    )


def _build_smd_output_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the OOF baseline-SMD CSV companion dataframe.

    Args:
        rows: SMD rows.

    Returns:
        Ordered dataframe.
    """
    return _build_metric_output_frame(
        rows,
        ["variable", "mean_B", "mean_C", "smd", "n_B_non_missing", "n_C_non_missing"],
    )


def _build_primary_oof_output_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the single-row primary OOF summary CSV.

    Args:
        rows: Primary-summary rows.

    Returns:
        Ordered dataframe.
    """
    return _build_metric_output_frame(
        rows,
        [
            "rb",
            "U",
            "p_two_sided",
            "median_p_cal_B",
            "median_p_cal_C",
            "q1_p_cal_B",
            "q3_p_cal_B",
            "q1_p_cal_C",
            "q3_p_cal_C",
            "n_above_0.5_B",
            "n_above_0.5_C",
            "n_B",
            "n_C",
            "directional_verification_passed",
            "separation_criterion_passed",
        ],
    )


def _build_test_view_output_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the descriptive test-view CSV companion dataframe.

    Args:
        rows: Test-view descriptive rows.

    Returns:
        Ordered dataframe.
    """
    return _build_metric_output_frame(
        rows,
        ["sample_id", "p_cal", "age", "sex", "bmi", "smoking", "drinking", "family_history", "hp_infection", "cea", "ca199", "ca724"],
    )


def _format_distribution(series: pd.Series) -> tuple[float, float, float]:
    """Return median, first quartile, and third quartile.

    Args:
        series: Numeric series.

    Returns:
        Median, Q1, and Q3.
    """
    numeric = series.dropna().astype(float)
    return float(numeric.median()), float(numeric.quantile(0.25)), float(numeric.quantile(0.75))


def _render_primary_analysis_section(primary_frame: pd.DataFrame, c_group_values: list[float], confounding_lines: list[str]) -> str:
    """Render the primary OOF analysis markdown section.

    Args:
        primary_frame: Single-row primary summary dataframe.
        c_group_values: Individual C-group calibrated probabilities.
        confounding_lines: Markdown bullet lines for confounding rebuttal.

    Returns:
        Markdown section string.
    """
    row = primary_frame.iloc[0]
    c_values_text = ", ".join(f"{value:.6f}" for value in c_group_values)
    lines = [
        "## Primary Analysis (OOF View)",
        "",
        f"- n_B = {int(row['n_B'])}",
        f"- n_C = {int(row['n_C'])}",
        f"- U = {float(row['U']):.6f}",
        f"- rb = {float(row['rb']):.6f}",
        f"- p_two_sided = {float(row['p_two_sided']):.6g}",
        f"- median_p_cal_B = {float(row['median_p_cal_B']):.6f}",
        f"- median_p_cal_C = {float(row['median_p_cal_C']):.6f}",
        (
            f"- IQR_B = [{float(row['q1_p_cal_B']):.6f}, {float(row['q3_p_cal_B']):.6f}], "
            f"IQR_C = [{float(row['q1_p_cal_C']):.6f}, {float(row['q3_p_cal_C']):.6f}]"
        ),
        f"- n_above_0.5_B = {int(row['n_above_0.5_B'])}",
        f"- n_above_0.5_C = {int(row['n_above_0.5_C'])}",
        f"- directional_verification_passed = {bool(row['directional_verification_passed'])}",
        f"- separation_criterion_passed = {bool(row['separation_criterion_passed'])}",
        f"- C-group p_cal values: [{c_values_text}]",
        "",
        "### Power Caveat",
        (
            "Power is limited by n_C=4. The minimum achievable two-sided Mann-Whitney p-value under the null at "
            "n_B=35 and n_C=4 is approximately 1.22e-5. Power to detect Cohen-scale |rb| >= 0.5 at alpha=0.05 is "
            "approximately 0.5-0.6. Bootstrap CIs are not computed because n_C=4 produces degenerate resamples."
        ),
        "",
        "### Confounding-By-Indication Rebuttal",
        *confounding_lines,
    ]
    return "\n".join(lines)


def _render_test_view_section(test_frame: pd.DataFrame) -> str:
    """Render the descriptive supplementary test-view markdown section.

    Args:
        test_frame: Test-view LGIN dataframe.

    Returns:
        Markdown section string.
    """
    median_p_cal, q1_p_cal, q3_p_cal = _format_distribution(test_frame["p_cal"])
    age_median, age_q1, age_q3 = _format_distribution(test_frame["age"])
    bmi_median, bmi_q1, bmi_q3 = _format_distribution(test_frame["bmi"])
    def _prevalence(series: pd.Series) -> float:
        return float(100.0 * series.dropna().astype(float).mean())

    def _marker_summary(series: pd.Series) -> str:
        numeric = series.dropna().astype(float)
        if numeric.empty:
            return f"median=nan; missing={int(series.isna().sum())}"
        return f"median={float(numeric.median()):.6f}; missing={int(series.isna().sum())}"

    lines = [
        "## Sensitivity / Supplementary Analysis (Test View)",
        "",
        (
            f"- Test-view B/C separation is non-estimable because n_C_test=0. "
            f"All {len(test_frame)} LGIN test samples are in the B-group."
        ),
        f"- n_above_0.5 = {int((test_frame['p_cal'] > 0.5).sum())} / {len(test_frame)}",
        f"- median_p_cal = {median_p_cal:.6f}",
        f"- p_cal_IQR = [{q1_p_cal:.6f}, {q3_p_cal:.6f}]",
        f"- age median [Q1, Q3] = {age_median:.6f} [{age_q1:.6f}, {age_q3:.6f}]",
        f"- sex prevalence (%) = {_prevalence(test_frame['sex']):.2f}",
        f"- BMI median [Q1, Q3] = {bmi_median:.6f} [{bmi_q1:.6f}, {bmi_q3:.6f}]",
        f"- smoking prevalence (%) = {_prevalence(test_frame['smoking']):.2f}",
        f"- drinking prevalence (%) = {_prevalence(test_frame['drinking']):.2f}",
        f"- family_history prevalence (%) = {_prevalence(test_frame['family_history']):.2f}",
        f"- hp_infection prevalence (%) = {_prevalence(test_frame['hp_infection']):.2f}",
        f"- CEA: {_marker_summary(test_frame['cea'])}",
        f"- CA19-9: {_marker_summary(test_frame['ca199'])}",
        f"- CA72-4: {_marker_summary(test_frame['ca724'])}",
    ]
    return "\n".join(lines)


def _render_footer(*, git_commit_hash: str, timestamp_utc: str) -> str:
    """Render the E012 footer block.

    Args:
        git_commit_hash: Current git commit hash.
        timestamp_utc: Current UTC timestamp.

    Returns:
        Markdown footer string.
    """
    return "\n".join(
        [
            "## Footer",
            "",
            f"- Erratum E012: {ERRATUM_E012_SUMMARY}",
            f"- git_commit_hash = `{git_commit_hash}`",
            f"- timestamp_utc = `{timestamp_utc}`",
            f'- Honesty paragraph: "{HONESTY_PARAGRAPH}"',
        ]
    )


def _build_oof_lgin_frame(*, best_model: str, metadata: pd.DataFrame, calibrator: Any) -> pd.DataFrame:
    """Build the OOF LGIN dataframe with calibrated probabilities.

    Args:
        best_model: Canonical best-model short name.
        metadata: Final-cohort metadata dataframe.
        calibrator: Fitted Stage 4b Platt calibrator from the PKL.

    Returns:
        OOF LGIN dataframe with `p_cal`.
    """
    oof_frame = _load_view_prediction_frame(model_name=best_model, view_name="nested_cv_oof")
    oof_frame = oof_frame.copy()
    oof_frame["p_cal"] = apply_platt_calibrator(calibrator=calibrator, p_raw=oof_frame["p_raw"].to_numpy(dtype=float))
    merged = _merge_predictions_with_optional_metadata(
        prediction_frame=oof_frame,
        metadata_frame=metadata,
        required_columns=["disease_category", "collection_location", "age", "sex", "bmi", "smoking", "drinking", "family_history"],
        optional_columns=["hp_infection", "cea", "ca199", "ca724"],
    )
    return merged.loc[merged["disease_category"] == 2].reset_index(drop=True)


def _build_test_lgin_frame(*, best_model: str, metadata: pd.DataFrame) -> pd.DataFrame:
    """Build the test-view LGIN dataframe.

    Args:
        best_model: Canonical best-model short name.
        metadata: Final-cohort metadata dataframe.

    Returns:
        Test-view LGIN dataframe.
    """
    test_frame = _load_view_prediction_frame(model_name=best_model, view_name="test")
    merged = _merge_predictions_with_optional_metadata(
        prediction_frame=test_frame,
        metadata_frame=metadata,
        required_columns=["disease_category", "collection_location", "age", "sex", "bmi", "smoking", "drinking", "family_history"],
        optional_columns=["hp_infection", "cea", "ca199", "ca724"],
    )
    return merged.loc[merged["disease_category"] == 2].reset_index(drop=True)


def run_lgin_layer_analysis(config_path: Path, best_model: str) -> None:
    """Run the E012-authorized LGIN exploratory analysis.

    Args:
        config_path: Path to `configs/modeling_global.yaml`.
        best_model: Canonical best-model short name.
    """
    config = _load_global_config(config_path=config_path)
    metadata = _load_final_cohort_extended_metadata(config=config)
    model_payload = _load_model_payload(best_model)
    calibrator = model_payload["calibrator"]
    if calibrator is None:
        raise AssertionError("LGIN OOF primary analysis requires a fitted PKL calibrator.")

    oof_lgin_frame = _build_oof_lgin_frame(best_model=best_model, metadata=metadata, calibrator=calibrator)
    n_lgin_oof = int(len(oof_lgin_frame))
    b_group_oof = oof_lgin_frame.loc[oof_lgin_frame["collection_location"] == 1].reset_index(drop=True)
    c_group_oof = oof_lgin_frame.loc[oof_lgin_frame["collection_location"] == 0].reset_index(drop=True)
    n_b_oof = int(len(b_group_oof))
    n_c_oof = int(len(c_group_oof))
    LOGGER.info("LGIN OOF cohort sizes: n_LGIN=%s, n_B=%s, n_C=%s.", n_lgin_oof, n_b_oof, n_c_oof)
    if any(
        abs(actual - expected) > 1
        for actual, expected in [
            (n_lgin_oof, EXPECTED_OOF_COUNTS["n_lgin"]),
            (n_b_oof, EXPECTED_OOF_COUNTS["n_b"]),
            (n_c_oof, EXPECTED_OOF_COUNTS["n_c"]),
        ]
    ):
        raise AssertionError(
            "OOF LGIN counts drifted beyond the E012 tolerance: "
            f"n_LGIN={n_lgin_oof}, n_B={n_b_oof}, n_C={n_c_oof}."
        )
    if n_c_oof == 0:
        raise AssertionError("OOF LGIN C-group is empty despite E012 authorization.")

    test_lgin_frame = _build_test_lgin_frame(best_model=best_model, metadata=metadata)
    n_b_test = int((test_lgin_frame["collection_location"] == 1).sum())
    n_c_test = int((test_lgin_frame["collection_location"] == 0).sum())
    LOGGER.info("LGIN test cohort sizes: n_LGIN=%s, n_B=%s, n_C=%s.", len(test_lgin_frame), n_b_test, n_c_test)

    mw_result = mannwhitneyu(
        b_group_oof["p_cal"].to_numpy(dtype=float),
        c_group_oof["p_cal"].to_numpy(dtype=float),
        alternative="two-sided",
    )
    if np.isnan(float(mw_result.statistic)) or np.isnan(float(mw_result.pvalue)):
        raise AssertionError("LGIN OOF Mann-Whitney U returned NaN.")
    rank_biserial = _compute_rank_biserial_from_u(
        u_statistic=float(mw_result.statistic),
        n_group_b=n_b_oof,
        n_group_c=n_c_oof,
    )
    directional_verification = bool(float(b_group_oof["p_cal"].median()) > float(c_group_oof["p_cal"].median()))
    separation_criterion_passed = _evaluate_separation_criterion(
        rank_biserial=rank_biserial,
        p_two_sided=float(mw_result.pvalue),
    )
    primary_row = {
        "rb": rank_biserial,
        "U": float(mw_result.statistic),
        "p_two_sided": float(mw_result.pvalue),
        "median_p_cal_B": float(b_group_oof["p_cal"].median()),
        "median_p_cal_C": float(c_group_oof["p_cal"].median()),
        "q1_p_cal_B": float(b_group_oof["p_cal"].quantile(0.25)),
        "q3_p_cal_B": float(b_group_oof["p_cal"].quantile(0.75)),
        "q1_p_cal_C": float(c_group_oof["p_cal"].quantile(0.25)),
        "q3_p_cal_C": float(c_group_oof["p_cal"].quantile(0.75)),
        "n_above_0.5_B": int((b_group_oof["p_cal"] > 0.5).sum()),
        "n_above_0.5_C": int((c_group_oof["p_cal"] > 0.5).sum()),
        "n_B": n_b_oof,
        "n_C": n_c_oof,
        "directional_verification_passed": directional_verification,
        "separation_criterion_passed": separation_criterion_passed,
    }
    primary_frame = _build_primary_oof_output_frame([primary_row])
    primary_path = _resolve_path(config["paths"]["outputs_analysis_dir"]) / f"lgin_primary_oof_{best_model}.csv"
    _write_dataframe_csv(primary_path, primary_frame)

    smd_rows: list[dict[str, Any]] = []
    for variable_name, variable_type in BASELINE_SMD_VARIABLES:
        non_missing_b = b_group_oof[variable_name].dropna()
        non_missing_c = c_group_oof[variable_name].dropna()
        mean_b = float(non_missing_b.astype(float).mean()) if not non_missing_b.empty else float("nan")
        mean_c = float(non_missing_c.astype(float).mean()) if not non_missing_c.empty else float("nan")
        smd_value = (
            _compute_binary_risk_difference(b_group_oof[variable_name], c_group_oof[variable_name])
            if variable_type == "binary"
            else _compute_continuous_smd(b_group_oof[variable_name], c_group_oof[variable_name])
        )
        smd_rows.append(
            {
                "variable": variable_name,
                "mean_B": mean_b,
                "mean_C": mean_c,
                "smd": smd_value,
                "n_B_non_missing": int(len(non_missing_b)),
                "n_C_non_missing": int(len(non_missing_c)),
            }
        )
    smd_frame = _build_smd_output_frame(smd_rows)
    smd_path = _resolve_path(config["paths"]["outputs_analysis_dir"]) / f"lgin_baseline_smd_oof_{best_model}.csv"
    _write_dataframe_csv(smd_path, smd_frame)

    within_b_frame = _build_within_b_stability_frame(
        b_group_frame=b_group_oof,
        age_cutpoint=int(config["age_cutpoint"]),
        bmi_cutpoint=int(config["bmi_cutpoint"]),
        threshold_probability=0.5,
    )
    within_b_path = _resolve_path(config["paths"]["outputs_analysis_dir"]) / f"lgin_within_b_stability_oof_{best_model}.csv"
    _write_dataframe_csv(within_b_path, within_b_frame)
    gated_within_b = within_b_frame.loc[within_b_frame["omission_reason"] == ""].reset_index(drop=True)
    if gated_within_b.empty:
        within_b_state = "no gated levels"
        within_b_pass_overall = False
    elif bool((gated_within_b["median_p_cal"] > 0.5).all()):
        within_b_state = "all gated levels pass"
        within_b_pass_overall = True
    else:
        within_b_state = "some levels fail"
        within_b_pass_overall = False

    test_output_frame = _build_test_view_output_frame(
        test_lgin_frame.loc[
            :,
            ["sample_id", "p_cal", "age", "sex", "bmi", "smoking", "drinking", "family_history", "hp_infection", "cea", "ca199", "ca724"],
        ]
        .sort_values("sample_id", kind="mergesort")
        .to_dict("records")
    )
    test_path = _resolve_path(config["paths"]["outputs_analysis_dir"]) / f"lgin_descriptive_test_{best_model}.csv"
    _write_dataframe_csv(test_path, test_output_frame)

    confounding_lines = []
    for variable_name in ("smoking", "drinking", "hp_infection"):
        prevalence_b = float(100.0 * b_group_oof[variable_name].dropna().astype(float).mean())
        prevalence_c = float(100.0 * c_group_oof[variable_name].dropna().astype(float).mean())
        higher_in_c = prevalence_c > prevalence_b
        confounding_lines.append(
            f"- {variable_name}_B = {prevalence_b:.2f}%, {variable_name}_C = {prevalence_c:.2f}%, higher_in_C = {higher_in_c}"
        )

    markdown_lines = [
        "# LGIN Exploratory Layer Analysis",
        "",
        f"- Best model: `{best_model}`",
        f"- Primary OOF CSV: `{_format_path(primary_path)}`",
        f"- Baseline SMD CSV: `{_format_path(smd_path)}`",
        f"- Within-B stability CSV: `{_format_path(within_b_path)}`",
        f"- Test descriptive CSV: `{_format_path(test_path)}`",
        "",
        _render_primary_analysis_section(
            primary_frame=primary_frame,
            c_group_values=c_group_oof["p_cal"].astype(float).tolist(),
            confounding_lines=confounding_lines,
        ),
        "",
        "## Within-B Stability (OOF View)",
        "",
        f"- overall_pass_state = {within_b_state}",
        f"- within_b_pass_overall = {within_b_pass_overall}",
        f"- gated_levels_evaluated = {len(gated_within_b)}",
        "",
        "| variable_name | level_value | n_in_level | median_p_cal | n_above_0_5 | omission_reason |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in within_b_frame.iterrows():
        median_text = "" if pd.isna(row["median_p_cal"]) else f"{float(row['median_p_cal']):.6f}"
        count_text = "" if pd.isna(row["n_above_0_5"]) else str(int(row["n_above_0_5"]))
        markdown_lines.append(
            "| "
            f"{row['variable_name']} | {row['level_value']} | {int(row['n_in_level'])} | {median_text} | {count_text} | "
            f"{row['omission_reason']} |"
        )

    markdown_lines.extend(
        [
            "",
            _render_test_view_section(test_lgin_frame),
            "",
            _render_footer(git_commit_hash=_get_git_commit_hash(), timestamp_utc=_timestamp_utc()),
        ]
    )
    markdown_path = _resolve_path(config["paths"]["outputs_analysis_dir"]) / "lgin_layer_analysis.md"
    _write_text(markdown_path, "\n".join(markdown_lines) + "\n")
    LOGGER.info("Wrote LGIN layer markdown to %s.", _format_path(markdown_path))


__all__ = [
    "BASELINE_SMD_VARIABLES",
    "ERRATUM_E012_SUMMARY",
    "HONESTY_PARAGRAPH",
    "WITHIN_B_VARIABLES",
    "_build_primary_oof_output_frame",
    "_build_test_view_output_frame",
    "_build_within_b_output_frame",
    "_build_within_b_stability_frame",
    "_compute_rank_biserial_from_u",
    "_evaluate_separation_criterion",
    "_render_footer",
    "_render_primary_analysis_section",
    "_render_test_view_section",
    "run_lgin_layer_analysis",
]
