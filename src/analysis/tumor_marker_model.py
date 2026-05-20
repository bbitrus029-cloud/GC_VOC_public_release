from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.stage7_part_a_common import (
    _bootstrap_metric_from_frame,
    _format_path,
    _load_model_payload,
    _resolve_path,
    _write_text,
)
from src.analysis.stage7_part_c_common import (
    MARKER_FEATURE_NAMES,
    RANDOM_SEED,
    SELECTION_BIAS_VARIABLES,
    _assemble_tumor_marker_subsets,
    _assert_expected_subset_sizes,
    _build_final_pkl_payload,
    _build_outer_fold_oof_predictions,
    _compute_selection_bias_rows,
    _create_logistic_regression_estimator,
    _fit_logistic_regression_model,
    _fit_youden_threshold,
    _load_global_config,
    _load_outer_fold_definitions,
    _predict_logistic_regression_probabilities,
    _save_final_pkl_compliant,
    _timestamp_utc,
    _write_dataframe_csv,
)
from src.evaluation.calibration import apply_platt_calibrator, fit_platt_calibrator
from src.evaluation.metrics import compute_threshold_dependent_metrics
from src.evaluation.statistical_tests import compute_nri_idi, delong_roc_test


LOGGER = logging.getLogger(__name__)
TM_SELECTION_BIAS_MD = Path("outputs/analysis/tumor_marker_selection_bias.md")
TM_SELECTION_BIAS_CSV = Path("outputs/analysis/tumor_marker_selection_bias.csv")


def _build_prediction_frame(
    *,
    sample_ids: pd.Series,
    labels: pd.Series,
    p_raw: np.ndarray,
    p_cal: np.ndarray,
) -> pd.DataFrame:
    """Build one compact prediction frame for metric computation."""
    return pd.DataFrame(
        {
            "sample_id": sample_ids.to_numpy(dtype=np.int64),
            "y_true": labels.to_numpy(dtype=np.int64),
            "p_raw": np.asarray(p_raw, dtype=float),
            "p_cal": np.asarray(p_cal, dtype=float),
        }
    )


def _fit_marker_model(
    config_path: Path,
    best_model: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit the tumor-marker LR model and return test predictions plus summary values."""
    config = _load_global_config(config_path=config_path)
    subset_bundle = _assemble_tumor_marker_subsets(config_path=config_path, best_model=best_model)
    _assert_expected_subset_sizes(subset_bundle.tm_train_subset, subset_bundle.tm_test_subset)

    outer_folds = _load_outer_fold_definitions(config)
    marker_oof = _build_outer_fold_oof_predictions(
        train_cohort=subset_bundle.train_cohort,
        tm_train_subset=subset_bundle.tm_train_subset,
        feature_names=MARKER_FEATURE_NAMES,
        outer_folds=outer_folds,
    )
    marker_calibrator = fit_platt_calibrator(
        p_raw=marker_oof["p_raw"].to_numpy(dtype=float),
        y_true=marker_oof["y_true"].to_numpy(dtype=np.int64),
        seed=RANDOM_SEED,
    )
    marker_threshold_p_raw = _fit_youden_threshold(
        y_true=marker_oof["y_true"].to_numpy(dtype=np.int64),
        p_raw=marker_oof["p_raw"].to_numpy(dtype=float),
    )
    marker_threshold_p_final_equivalent = float(
        apply_platt_calibrator(
            calibrator=marker_calibrator,
            p_raw=np.asarray([marker_threshold_p_raw], dtype=float),
        )[0]
    )

    final_model = _create_logistic_regression_estimator()
    _fit_logistic_regression_model(
        estimator=final_model,
        feature_frame=subset_bundle.tm_train_subset.loc[:, list(MARKER_FEATURE_NAMES)],
        y_true=subset_bundle.tm_train_subset["label"].to_numpy(dtype=np.int64),
    )
    marker_p_raw_test = _predict_logistic_regression_probabilities(
        estimator=final_model,
        feature_frame=subset_bundle.tm_test_subset.loc[:, list(MARKER_FEATURE_NAMES)],
    )
    marker_p_cal_test = apply_platt_calibrator(marker_calibrator, marker_p_raw_test)
    marker_test_frame = _build_prediction_frame(
        sample_ids=subset_bundle.tm_test_subset["sample_id"],
        labels=subset_bundle.tm_test_subset["label"],
        p_raw=marker_p_raw_test,
        p_cal=marker_p_cal_test,
    )

    if np.isnan(final_model.coef_).any():
        raise AssertionError("Marker-LR coefficients contain NaN values.")

    model_path = _resolve_path("outputs/models") / f"binary_{best_model}_marker_lr.pkl"
    payload = _build_final_pkl_payload(
        model_name="marker_lr",
        final_model=final_model,
        calibrator=marker_calibrator,
        threshold_p_raw=marker_threshold_p_raw,
        threshold_p_final_equivalent=marker_threshold_p_final_equivalent,
        feature_names=MARKER_FEATURE_NAMES,
        training_sample_ids=subset_bundle.tm_train_subset["sample_id"].to_numpy(dtype=np.int64),
    )
    _save_final_pkl_compliant(model_path, payload)

    summary = {
        "tm_train_subset_n_total": int(len(subset_bundle.tm_train_subset)),
        "tm_train_subset_n_pos": int(subset_bundle.tm_train_subset["label"].sum()),
        "tm_test_subset_n_total": int(len(subset_bundle.tm_test_subset)),
        "tm_test_subset_n_pos": int(subset_bundle.tm_test_subset["label"].sum()),
        "marker_threshold_p_raw": float(marker_threshold_p_raw),
        "marker_threshold_p_final_equivalent": float(marker_threshold_p_final_equivalent),
    }
    return subset_bundle.tm_test_subset, marker_test_frame, summary


def _build_selection_bias_markdown(selection_bias_frame: pd.DataFrame) -> str:
    """Render the mandatory tumor-marker selection-bias markdown report."""
    largest_row = selection_bias_frame.sort_values("smd", ascending=False, kind="mergesort").iloc[0]
    notable_variables = selection_bias_frame.loc[selection_bias_frame["smd"] >= 0.1, "variable"].tolist()
    lines = [
        "# Tumor Marker Selection Bias",
        "",
        "Tumor marker missingness is MNAR; the marker-LR cohort is enriched in higher-prevalence samples relative to the full test set, consistent with real-world selective testing on patients with stronger clinical indication. VOC analysis is performed universally on all 195 test samples without selection.",
        "",
        f"- Largest SMD: `{largest_row['variable']}` = `{largest_row['smd']:.4f}`",
        f"- Variables with SMD >= 0.1: `{notable_variables}`",
        f"- Locked variable list: `{list(SELECTION_BIAS_VARIABLES)}`",
        "",
        "## Selection-Bias Table",
        "",
        "| variable | value_in_tm_subset | value_in_full_test | smd | n_non_missing_tm | n_non_missing_full | interpretation |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in selection_bias_frame.itertuples(index=False):
        lines.append(
            "| "
            f"{row.variable} | {row.value_in_tm_subset:.6f} | {row.value_in_full_test:.6f} | {row.smd:.6f} | "
            f"{row.n_non_missing_tm} | {row.n_non_missing_full} | {row.interpretation} |"
        )
    return "\n".join(lines) + "\n"


def _build_comparison_output_frame(
    *,
    best_model: str,
    tm_test_subset: pd.DataFrame,
    marker_test_frame: pd.DataFrame,
    marker_threshold_p_raw: float,
    n_boot: int,
) -> pd.DataFrame:
    """Build the long-format VOC-only vs marker-LR comparison frame."""
    voc_frame = _build_prediction_frame(
        sample_ids=tm_test_subset["sample_id"],
        labels=tm_test_subset["label"],
        p_raw=tm_test_subset["voc_p_raw"].to_numpy(dtype=float),
        p_cal=tm_test_subset["voc_p_cal"].to_numpy(dtype=float),
    )
    voc_model_payload = _load_model_payload(best_model)
    voc_auc = _bootstrap_metric_from_frame(
        prediction_frame=voc_frame,
        metric_name="auc_roc",
        threshold_p_raw=float(voc_model_payload["threshold_p_raw"]),
        n_boot=n_boot,
        seed=RANDOM_SEED,
    )
    marker_auc = _bootstrap_metric_from_frame(
        prediction_frame=marker_test_frame,
        metric_name="auc_roc",
        threshold_p_raw=0.5,
        n_boot=n_boot,
        seed=RANDOM_SEED,
    )

    y_true = voc_frame["y_true"].to_numpy(dtype=np.int64)
    delong_result = delong_roc_test(
        y_true=y_true,
        score_a=voc_frame["p_raw"].to_numpy(dtype=float),
        score_b=marker_test_frame["p_raw"].to_numpy(dtype=float),
    )
    nri_idi = compute_nri_idi(
        y_true=y_true,
        p_old=marker_test_frame["p_cal"].to_numpy(dtype=float),
        p_new=voc_frame["p_cal"].to_numpy(dtype=float),
    )
    voc_threshold_metrics = compute_threshold_dependent_metrics(
        y_true=y_true,
        p_raw=voc_frame["p_raw"].to_numpy(dtype=float),
        threshold_p_raw=float(voc_model_payload["threshold_p_raw"]),
    )
    marker_threshold_metrics = compute_threshold_dependent_metrics(
        y_true=y_true,
        p_raw=marker_test_frame["p_raw"].to_numpy(dtype=float),
        threshold_p_raw=float(marker_threshold_p_raw),
    )

    rows: list[dict[str, Any]] = [
        {
            "metric": "auc_roc",
            "model": "voc_only",
            "estimate": float(voc_auc["estimate"]),
            "ci_low": float(voc_auc["ci_low"]),
            "ci_high": float(voc_auc["ci_high"]),
            "comparison_stat": "",
            "comparison_value": np.nan,
        },
        {
            "metric": "auc_roc",
            "model": "marker_lr",
            "estimate": float(marker_auc["estimate"]),
            "ci_low": float(marker_auc["ci_low"]),
            "ci_high": float(marker_auc["ci_high"]),
            "comparison_stat": "",
            "comparison_value": np.nan,
        },
    ]
    for metric_name in (
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "f1",
        "balanced_accuracy",
        "mcc",
        "accuracy",
    ):
        rows.append(
            {
                "metric": metric_name,
                "model": "voc_only",
                "estimate": float(voc_threshold_metrics[metric_name]),
                "ci_low": np.nan,
                "ci_high": np.nan,
                "comparison_stat": "",
                "comparison_value": np.nan,
            }
        )
        rows.append(
            {
                "metric": metric_name,
                "model": "marker_lr",
                "estimate": float(marker_threshold_metrics[metric_name]),
                "ci_low": np.nan,
                "ci_high": np.nan,
                "comparison_stat": "",
                "comparison_value": np.nan,
            }
        )
    rows.extend(
        [
            {
                "metric": "delta_auc_voc_minus_marker",
                "model": "comparison",
                "estimate": float(delong_result["delta_auc"]),
                "ci_low": np.nan,
                "ci_high": np.nan,
                "comparison_stat": "delong_z",
                "comparison_value": float(delong_result["z"]),
            },
            {
                "metric": "delta_auc_voc_minus_marker",
                "model": "comparison",
                "estimate": float(delong_result["delta_auc"]),
                "ci_low": np.nan,
                "ci_high": np.nan,
                "comparison_stat": "delong_p_two_sided",
                "comparison_value": float(delong_result["p_two_sided"]),
            },
            {
                "metric": "nri",
                "model": "comparison",
                "estimate": float(nri_idi["nri"]),
                "ci_low": np.nan,
                "ci_high": np.nan,
                "comparison_stat": "nri_events",
                "comparison_value": float(nri_idi["nri_events"]),
            },
            {
                "metric": "nri",
                "model": "comparison",
                "estimate": float(nri_idi["nri"]),
                "ci_low": np.nan,
                "ci_high": np.nan,
                "comparison_stat": "nri_nonevents",
                "comparison_value": float(nri_idi["nri_nonevents"]),
            },
            {
                "metric": "idi",
                "model": "comparison",
                "estimate": float(nri_idi["idi"]),
                "ci_low": np.nan,
                "ci_high": np.nan,
                "comparison_stat": "idi_events",
                "comparison_value": float(nri_idi["idi_events"]),
            },
            {
                "metric": "idi",
                "model": "comparison",
                "estimate": float(nri_idi["idi"]),
                "ci_low": np.nan,
                "ci_high": np.nan,
                "comparison_stat": "idi_nonevents",
                "comparison_value": float(nri_idi["idi_nonevents"]),
            },
        ]
    )
    return pd.DataFrame(rows)


def _build_comparison_markdown(
    *,
    best_model: str,
    comparison_frame: pd.DataFrame,
    tm_train_n: int,
    tm_train_pos: int,
    tm_test_n: int,
    tm_test_pos: int,
) -> str:
    """Render the tumor-marker comparison markdown report."""
    voc_auc_row = comparison_frame.loc[
        (comparison_frame["metric"] == "auc_roc") & (comparison_frame["model"] == "voc_only")
    ].iloc[0]
    marker_auc_row = comparison_frame.loc[
        (comparison_frame["metric"] == "auc_roc") & (comparison_frame["model"] == "marker_lr")
    ].iloc[0]
    delong_row = comparison_frame.loc[comparison_frame["comparison_stat"] == "delong_p_two_sided"].iloc[0]
    delong_z_row = comparison_frame.loc[comparison_frame["comparison_stat"] == "delong_z"].iloc[0]
    nri_row = comparison_frame.loc[
        (comparison_frame["metric"] == "nri") & (comparison_frame["comparison_stat"] == "nri_events")
    ].iloc[0]
    idi_row = comparison_frame.loc[
        (comparison_frame["metric"] == "idi") & (comparison_frame["comparison_stat"] == "idi_events")
    ].iloc[0]
    lines = [
        f"# Tumor Marker LR Comparison - {best_model}",
        "",
        f"- tm_train_subset: n={tm_train_n}, n_pos={tm_train_pos}",
        f"- tm_test_subset: n={tm_test_n}, n_pos={tm_test_pos}",
        "- Selection-bias context: see `outputs/analysis/tumor_marker_selection_bias.md`.",
        "",
        "## AUC Summary",
        "",
        f"- VOC-only AUC-ROC = {voc_auc_row['estimate']:.6f} (95% CI {voc_auc_row['ci_low']:.6f}-{voc_auc_row['ci_high']:.6f})",
        f"- Marker-LR AUC-ROC = {marker_auc_row['estimate']:.6f} (95% CI {marker_auc_row['ci_low']:.6f}-{marker_auc_row['ci_high']:.6f})",
        f"- Delta AUC (VOC - Marker) = {delong_row['estimate']:.6f}",
        f"- Paired DeLong z = {delong_z_row['comparison_value']:.6f}, p_two_sided = {delong_row['comparison_value']:.6g}",
        f"- Continuous NRI = {nri_row['estimate']:.6f}",
        f"- IDI = {idi_row['estimate']:.6f}",
        "",
        "## Interpretation",
        "",
        "This comparison is restricted to the same 135 complete-case test samples for both models. The direction and strength of the effect are reported directly from the paired DeLong, NRI, and IDI statistics without attributing any change in AUC to Platt calibration.",
    ]
    return "\n".join(lines) + "\n"


def run_tumor_marker_model(config_path: Path, best_model: str) -> None:
    """Run the Session 13 tumor-marker LR comparison workflow.

    Args:
        config_path: Path to `configs/modeling_global.yaml`.
        best_model: Locked best-model short name.
    """
    config = _load_global_config(config_path=config_path)
    tm_test_subset, marker_test_frame, summary = _fit_marker_model(config_path=config_path, best_model=best_model)
    subset_bundle = _assemble_tumor_marker_subsets(config_path=config_path, best_model=best_model)

    selection_bias_rows = _compute_selection_bias_rows(
        tm_test_subset=subset_bundle.tm_test_subset,
        full_test_frame=subset_bundle.test_cohort,
    )
    selection_bias_frame = pd.DataFrame(selection_bias_rows)
    selection_bias_csv_path = _resolve_path(str(TM_SELECTION_BIAS_CSV))
    selection_bias_md_path = _resolve_path(str(TM_SELECTION_BIAS_MD))
    _write_dataframe_csv(selection_bias_csv_path, selection_bias_frame)
    _write_text(selection_bias_md_path, _build_selection_bias_markdown(selection_bias_frame))

    comparison_frame = _build_comparison_output_frame(
        best_model=best_model,
        tm_test_subset=tm_test_subset,
        marker_test_frame=marker_test_frame,
        marker_threshold_p_raw=float(summary["marker_threshold_p_raw"]),
        n_boot=int(config["bootstrap"]["n_boot"]),
    )
    comparison_csv_path = _resolve_path("outputs/analysis") / f"tumor_marker_lr_comparison_{best_model}.csv"
    comparison_md_path = _resolve_path("outputs/analysis") / f"tumor_marker_lr_comparison_{best_model}.md"
    _write_dataframe_csv(comparison_csv_path, comparison_frame)
    _write_text(
        comparison_md_path,
        _build_comparison_markdown(
            best_model=best_model,
            comparison_frame=comparison_frame,
            tm_train_n=int(summary["tm_train_subset_n_total"]),
            tm_train_pos=int(summary["tm_train_subset_n_pos"]),
            tm_test_n=int(summary["tm_test_subset_n_total"]),
            tm_test_pos=int(summary["tm_test_subset_n_pos"]),
        ),
    )
    LOGGER.info(
        "Completed tumor-marker LR workflow for model=%s with tm_train_subset_n=%s tm_test_subset_n=%s at %s.",
        best_model,
        summary["tm_train_subset_n_total"],
        summary["tm_test_subset_n_total"],
        _timestamp_utc(),
    )


__all__ = [
    "SELECTION_BIAS_VARIABLES",
    "_fit_logistic_regression_model",
    "run_tumor_marker_model",
]
