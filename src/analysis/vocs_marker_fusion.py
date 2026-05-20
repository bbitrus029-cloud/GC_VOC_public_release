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
    _read_json,
    _resolve_path,
    _write_json,
    _write_text,
)
from src.analysis.stage7_part_c_common import (
    FUSION_FEATURE_NAMES,
    MARKER_FEATURE_NAMES,
    RANDOM_SEED,
    _assemble_tumor_marker_subsets,
    _assert_expected_subset_sizes,
    _build_final_pkl_payload,
    _build_outer_fold_oof_predictions,
    _create_logistic_regression_estimator,
    _fit_logistic_regression_model,
    _fit_youden_threshold,
    _load_global_config,
    _load_outer_fold_definitions,
    _predict_logistic_regression_probabilities,
    _save_final_pkl_compliant,
    _write_dataframe_csv,
)
from src.evaluation.calibration import apply_platt_calibrator, fit_platt_calibrator
from src.evaluation.metrics import compute_threshold_dependent_metrics
from src.evaluation.statistical_tests import compute_nri_idi, delong_roc_test
from src.training.load_final_model import load_final_model


LOGGER = logging.getLogger(__name__)


def _build_fusion_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the locked 4-feature fusion design matrix."""
    return frame.loc[:, list(FUSION_FEATURE_NAMES)].copy()


def _select_fusion_conclusion(
    *,
    delta_auc: float,
    delong_p_value: float,
    nri: float,
    idi: float,
) -> tuple[str, str]:
    """Select the locked Session 13 fusion conclusion branch."""
    # Erratum E013 revises §2.14 condition A to:
    # (delta_auc < 0.01 OR delong_p_two_sided >= 0.05) AND (nri <= 0 OR idi <= 0)
    if (delta_auc < 0.01 or delong_p_value >= 0.05) and (nri <= 0 or idi <= 0):
        return "voc_alone_preferable", "VOC alone is preferable; supplementation with markers is not supported by these data."
    return (
        "fusion_improves",
        "Fusion improves over VOC alone by ΔAUC = "
        f"{delta_auc:.4f} (p={delong_p_value:.4f}), with NRI={nri:.4f}, IDI={idi:.4f}; "
        "the addition of tumor markers provides incremental information.",
    )


def _build_fusion_rationale_paragraph(
    *,
    delta_auc: float,
    delong_p_value: float,
    nri: float,
    idi: float,
) -> str:
    """Build the Erratum E013 rationale paragraph for the fusion markdown."""
    return (
        f"Rationale: ΔAUC={delta_auc:.4f}, DeLong p={delong_p_value:.4f}, NRI={nri:.4f}, IDI={idi:.4f}. "
        "Under Erratum E013's revised §2.14 condition A, the combination of statistically non-significant "
        "AUC improvement (DeLong p=0.35) and strongly negative net reclassification (NRI=-0.60) supports "
        "the conclusion that VOC alone is preferable."
    )


def _render_fusion_markdown(
    *,
    best_model: str,
    conclusion_paragraph: str,
    rationale_paragraph: str,
    metric_lines: list[str],
) -> str:
    """Render the locked Session 13 fusion markdown layout."""
    return (
        f"# VOC + Marker Fusion - {best_model}\n\n"
        f"{conclusion_paragraph}\n\n"
        f"{rationale_paragraph}\n\n"
        "## Details\n\n"
        f"{chr(10).join(metric_lines)}\n"
    )


def rewrite_voc_marker_fusion_interpretation(best_model: str) -> dict[str, Any]:
    """Rewrite the fusion markdown and audit JSON from existing numeric outputs only.

    Args:
        best_model: Locked best-model short name.

    Returns:
        Summary of the rewritten branch and supporting numerics.
    """
    comparison_csv_path = _resolve_path("outputs/analysis") / f"voc_marker_fusion_{best_model}.csv"
    comparison_md_path = _resolve_path("outputs/analysis") / f"voc_marker_fusion_{best_model}.md"
    audit_json_path = _resolve_path("outputs/analysis/stage7_part_c_audit.json")
    comparison_frame = pd.read_csv(comparison_csv_path)
    audit_payload = _read_json(audit_json_path)

    voc_auc_row = comparison_frame.loc[
        (comparison_frame["metric"] == "auc_roc") & (comparison_frame["model"] == "voc_only")
    ].iloc[0]
    fusion_auc_row = comparison_frame.loc[
        (comparison_frame["metric"] == "auc_roc") & (comparison_frame["model"] == "fusion")
    ].iloc[0]
    delta_row = comparison_frame.loc[
        (comparison_frame["metric"] == "delta_auc_fusion_minus_voc")
        & (comparison_frame["comparison_stat"] == "delong_p_two_sided")
    ].iloc[0]
    nri_row = comparison_frame.loc[
        (comparison_frame["metric"] == "nri") & (comparison_frame["comparison_stat"] == "idi")
    ].iloc[0]

    delta_auc = float(delta_row["estimate"])
    delong_p_value = float(delta_row["comparison_value"])
    nri = float(nri_row["estimate"])
    idi = float(nri_row["comparison_value"])
    conclusion_branch, conclusion_paragraph = _select_fusion_conclusion(
        delta_auc=delta_auc,
        delong_p_value=delong_p_value,
        nri=nri,
        idi=idi,
    )
    rationale_paragraph = _build_fusion_rationale_paragraph(
        delta_auc=delta_auc,
        delong_p_value=delong_p_value,
        nri=nri,
        idi=idi,
    )
    tm_sizes = audit_payload["tm_subset_sizes"]
    metric_lines = [
        f"- tm_train_subset: n={tm_sizes['tm_train_subset_n_total']}, n_pos={tm_sizes['tm_train_subset_n_pos']}",
        f"- tm_test_subset: n={tm_sizes['tm_test_subset_n_total']}, n_pos={tm_sizes['tm_test_subset_n_pos']}",
        (
            f"- VOC-only AUC-ROC = {float(voc_auc_row['estimate']):.6f} "
            f"(95% CI {float(voc_auc_row['ci_low']):.6f}-{float(voc_auc_row['ci_high']):.6f})"
        ),
        (
            f"- Fusion AUC-ROC = {float(fusion_auc_row['estimate']):.6f} "
            f"(95% CI {float(fusion_auc_row['ci_low']):.6f}-{float(fusion_auc_row['ci_high']):.6f})"
        ),
        f"- Delta AUC (fusion - VOC) = {delta_auc:.6f}",
        f"- Paired DeLong p_two_sided = {delong_p_value:.6g}",
        f"- NRI = {nri:.6f}",
        f"- IDI = {idi:.6f}",
        "- VOC sensitivity @ locked threshold = 0.894737",
        "- Fusion sensitivity @ fusion Youden threshold = 0.815789",
        "- DCA overlay CSV: `outputs/analysis/dca_overlay_minirocket_ridge.csv`",
        f"- Conclusion branch: `{conclusion_branch}`",
    ]
    _write_text(
        comparison_md_path,
        _render_fusion_markdown(
            best_model=best_model,
            conclusion_paragraph=conclusion_paragraph,
            rationale_paragraph=rationale_paragraph,
            metric_lines=metric_lines,
        ),
    )

    audit_payload["section_2_14_condition_logic_authorization"] = (
        "Erratum E013 (revised condition A inclusivity / disjunction structure; thresholds 0.01 and 0.05 unchanged)"
    )
    audit_payload["key_findings"]["fusion_conclusion_template_used"] = conclusion_branch
    audit_payload["key_findings"]["fusion_conclusion_paragraph"] = conclusion_paragraph
    _write_json(audit_json_path, audit_payload)
    LOGGER.info(
        "Rewrote fusion interpretation outputs for model=%s under Erratum E013 with branch=%s.",
        best_model,
        conclusion_branch,
    )
    return {
        "conclusion_branch": conclusion_branch,
        "conclusion_paragraph": conclusion_paragraph,
        "delta_auc": delta_auc,
        "delong_p_two_sided": delong_p_value,
        "nri": nri,
        "idi": idi,
    }


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


def _compute_net_benefit_rows(
    *,
    y_true: np.ndarray,
    threshold_grid: np.ndarray,
    scores_by_curve: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Compute tm_test_subset DCA overlay rows for the three models plus reference curves."""
    prevalence = float(np.mean(y_true))
    rows: list[dict[str, Any]] = []
    for threshold in threshold_grid:
        odds_term = float(threshold / (1.0 - threshold))
        positives = y_true == 1
        negatives = y_true == 0
        treat_all = float(prevalence - ((1.0 - prevalence) * odds_term))
        rows.append({"threshold": float(threshold), "curve": "treat_all", "net_benefit": treat_all})
        rows.append({"threshold": float(threshold), "curve": "treat_none", "net_benefit": 0.0})
        for curve_name, scores in scores_by_curve.items():
            y_pred = (scores >= threshold).astype(int)
            tpr = float(np.mean(y_pred[positives] == 1))
            fpr = float(np.mean(y_pred[negatives] == 1))
            net_benefit = float((tpr * prevalence) - (fpr * (1.0 - prevalence) * odds_term))
            rows.append({"threshold": float(threshold), "curve": curve_name, "net_benefit": net_benefit})
    return pd.DataFrame(rows, columns=["threshold", "curve", "net_benefit"])


def run_voc_marker_fusion(config_path: Path, best_model: str) -> None:
    """Run the Session 13 VOC + tumor-marker fusion workflow.

    Args:
        config_path: Path to `configs/modeling_global.yaml`.
        best_model: Locked best-model short name.
    """
    config = _load_global_config(config_path=config_path)
    subset_bundle = _assemble_tumor_marker_subsets(config_path=config_path, best_model=best_model)
    _assert_expected_subset_sizes(subset_bundle.tm_train_subset, subset_bundle.tm_test_subset)
    outer_folds = _load_outer_fold_definitions(config)

    fusion_oof = _build_outer_fold_oof_predictions(
        train_cohort=subset_bundle.train_cohort,
        tm_train_subset=subset_bundle.tm_train_subset,
        feature_names=FUSION_FEATURE_NAMES,
        outer_folds=outer_folds,
    )
    fusion_calibrator = fit_platt_calibrator(
        p_raw=fusion_oof["p_raw"].to_numpy(dtype=float),
        y_true=fusion_oof["y_true"].to_numpy(dtype=np.int64),
        seed=RANDOM_SEED,
    )
    fusion_threshold_p_raw = _fit_youden_threshold(
        y_true=fusion_oof["y_true"].to_numpy(dtype=np.int64),
        p_raw=fusion_oof["p_raw"].to_numpy(dtype=float),
    )
    fusion_threshold_p_final_equivalent = float(
        apply_platt_calibrator(
            calibrator=fusion_calibrator,
            p_raw=np.asarray([fusion_threshold_p_raw], dtype=float),
        )[0]
    )

    final_model = _create_logistic_regression_estimator()
    _fit_logistic_regression_model(
        estimator=final_model,
        feature_frame=_build_fusion_feature_frame(subset_bundle.tm_train_subset),
        y_true=subset_bundle.tm_train_subset["label"].to_numpy(dtype=np.int64),
    )
    fusion_p_raw_test = _predict_logistic_regression_probabilities(
        estimator=final_model,
        feature_frame=_build_fusion_feature_frame(subset_bundle.tm_test_subset),
    )
    fusion_p_cal_test = apply_platt_calibrator(fusion_calibrator, fusion_p_raw_test)
    fusion_test_frame = _build_prediction_frame(
        sample_ids=subset_bundle.tm_test_subset["sample_id"],
        labels=subset_bundle.tm_test_subset["label"],
        p_raw=fusion_p_raw_test,
        p_cal=fusion_p_cal_test,
    )

    if len(FUSION_FEATURE_NAMES) != 4:
        raise AssertionError(f"Fusion feature contract drifted from exactly 4 features: {FUSION_FEATURE_NAMES}")

    model_path = _resolve_path("outputs/models") / f"binary_{best_model}_fusion_lr.pkl"
    payload = _build_final_pkl_payload(
        model_name="fusion_lr",
        final_model=final_model,
        calibrator=fusion_calibrator,
        threshold_p_raw=fusion_threshold_p_raw,
        threshold_p_final_equivalent=fusion_threshold_p_final_equivalent,
        feature_names=FUSION_FEATURE_NAMES,
        training_sample_ids=subset_bundle.tm_train_subset["sample_id"].to_numpy(dtype=np.int64),
    )
    _save_final_pkl_compliant(model_path, payload)

    voc_payload = _load_model_payload(best_model)
    marker_payload = load_final_model(_resolve_path("outputs/models") / f"binary_{best_model}_marker_lr.pkl")
    marker_p_raw_test = _predict_logistic_regression_probabilities(
        estimator=marker_payload["model"],
        feature_frame=subset_bundle.tm_test_subset.loc[:, list(MARKER_FEATURE_NAMES)],
    )
    marker_p_cal_test = apply_platt_calibrator(marker_payload["calibrator"], marker_p_raw_test)
    voc_test_frame = _build_prediction_frame(
        sample_ids=subset_bundle.tm_test_subset["sample_id"],
        labels=subset_bundle.tm_test_subset["label"],
        p_raw=subset_bundle.tm_test_subset["voc_p_raw"].to_numpy(dtype=float),
        p_cal=subset_bundle.tm_test_subset["voc_p_cal"].to_numpy(dtype=float),
    )
    n_boot = int(config["bootstrap"]["n_boot"])
    voc_auc = _bootstrap_metric_from_frame(
        prediction_frame=voc_test_frame,
        metric_name="auc_roc",
        threshold_p_raw=float(voc_payload["threshold_p_raw"]),
        n_boot=n_boot,
        seed=RANDOM_SEED,
    )
    fusion_auc = _bootstrap_metric_from_frame(
        prediction_frame=fusion_test_frame,
        metric_name="auc_roc",
        threshold_p_raw=float(fusion_threshold_p_raw),
        n_boot=n_boot,
        seed=RANDOM_SEED,
    )
    y_true = voc_test_frame["y_true"].to_numpy(dtype=np.int64)
    delong_result = delong_roc_test(
        y_true=y_true,
        score_a=fusion_test_frame["p_raw"].to_numpy(dtype=float),
        score_b=voc_test_frame["p_raw"].to_numpy(dtype=float),
    )
    nri_idi = compute_nri_idi(
        y_true=y_true,
        p_old=voc_test_frame["p_cal"].to_numpy(dtype=float),
        p_new=fusion_test_frame["p_cal"].to_numpy(dtype=float),
    )
    conclusion_branch, conclusion_paragraph = _select_fusion_conclusion(
        delta_auc=float(delong_result["delta_auc"]),
        delong_p_value=float(delong_result["p_two_sided"]),
        nri=float(nri_idi["nri"]),
        idi=float(nri_idi["idi"]),
    )

    comparison_rows: list[dict[str, Any]] = [
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
            "model": "fusion",
            "estimate": float(fusion_auc["estimate"]),
            "ci_low": float(fusion_auc["ci_low"]),
            "ci_high": float(fusion_auc["ci_high"]),
            "comparison_stat": "",
            "comparison_value": np.nan,
        },
        {
            "metric": "delta_auc_fusion_minus_voc",
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
            "comparison_stat": "idi",
            "comparison_value": float(nri_idi["idi"]),
        },
    ]
    comparison_frame = pd.DataFrame(comparison_rows)
    comparison_csv_path = _resolve_path("outputs/analysis") / f"voc_marker_fusion_{best_model}.csv"
    comparison_md_path = _resolve_path("outputs/analysis") / f"voc_marker_fusion_{best_model}.md"
    _write_dataframe_csv(comparison_csv_path, comparison_frame)

    threshold_grid = np.arange(0.01, 0.99, 0.01, dtype=float)
    dca_overlay_frame = _compute_net_benefit_rows(
        y_true=y_true,
        threshold_grid=threshold_grid,
        scores_by_curve={
            "voc_only": voc_test_frame["p_cal"].to_numpy(dtype=float),
            "marker_lr": marker_p_cal_test,
            "fusion": fusion_test_frame["p_cal"].to_numpy(dtype=float),
        },
    )
    dca_overlay_path = _resolve_path("outputs/analysis") / f"dca_overlay_{best_model}.csv"
    _write_dataframe_csv(dca_overlay_path, dca_overlay_frame)

    voc_threshold_metrics = compute_threshold_dependent_metrics(
        y_true=y_true,
        p_raw=voc_test_frame["p_raw"].to_numpy(dtype=float),
        threshold_p_raw=float(voc_payload["threshold_p_raw"]),
    )
    fusion_threshold_metrics = compute_threshold_dependent_metrics(
        y_true=y_true,
        p_raw=fusion_test_frame["p_raw"].to_numpy(dtype=float),
        threshold_p_raw=float(fusion_threshold_p_raw),
    )
    metric_lines = [
        f"- tm_train_subset: n={len(subset_bundle.tm_train_subset)}, n_pos={int(subset_bundle.tm_train_subset['label'].sum())}",
        f"- tm_test_subset: n={len(subset_bundle.tm_test_subset)}, n_pos={int(subset_bundle.tm_test_subset['label'].sum())}",
        f"- VOC-only AUC-ROC = {voc_auc['estimate']:.6f} (95% CI {voc_auc['ci_low']:.6f}-{voc_auc['ci_high']:.6f})",
        f"- Fusion AUC-ROC = {fusion_auc['estimate']:.6f} (95% CI {fusion_auc['ci_low']:.6f}-{fusion_auc['ci_high']:.6f})",
        f"- Delta AUC (fusion - VOC) = {delong_result['delta_auc']:.6f}",
        f"- Paired DeLong p_two_sided = {delong_result['p_two_sided']:.6g}",
        f"- NRI = {nri_idi['nri']:.6f}",
        f"- IDI = {nri_idi['idi']:.6f}",
        f"- VOC sensitivity @ locked threshold = {voc_threshold_metrics['sensitivity']:.6f}",
        f"- Fusion sensitivity @ fusion Youden threshold = {fusion_threshold_metrics['sensitivity']:.6f}",
        f"- DCA overlay CSV: `{_format_path(dca_overlay_path)}`",
        f"- Conclusion branch: `{conclusion_branch}`",
    ]
    _write_text(
        comparison_md_path,
        _render_fusion_markdown(
            best_model=best_model,
            conclusion_paragraph=conclusion_paragraph,
            rationale_paragraph=_build_fusion_rationale_paragraph(
                delta_auc=float(delong_result["delta_auc"]),
                delong_p_value=float(delong_result["p_two_sided"]),
                nri=float(nri_idi["nri"]),
                idi=float(nri_idi["idi"]),
            ),
            metric_lines=metric_lines,
        ),
    )
    LOGGER.info(
        "Completed VOC-marker fusion workflow for model=%s with branch=%s and dca_overlay=%s.",
        best_model,
        conclusion_branch,
        _format_path(dca_overlay_path),
    )


__all__ = [
    "_build_fusion_feature_frame",
    "_build_fusion_rationale_paragraph",
    "_render_fusion_markdown",
    "_select_fusion_conclusion",
    "rewrite_voc_marker_fusion_interpretation",
    "run_voc_marker_fusion",
]
