from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.stage7_part_a_common import (
    BEST_MODEL_SUBGROUP_METRIC_NAMES,
    MODEL_NAMES,
    RANDOM_SEED,
    THRESHOLD_FREE_METRIC_NAMES,
    _bootstrap_metric_from_frame,
    _build_metric_output_frame,
    _format_path,
    _load_global_config,
    _load_model_payload,
    _load_test_frame_with_metadata,
    _load_view_prediction_frame,
    _merge_predictions_with_metadata,
    _resolve_path,
    _timestamp_utc,
    _write_text,
)


LOGGER = logging.getLogger(__name__)
BEST_MODEL_SUBGROUP_METRICS = BEST_MODEL_SUBGROUP_METRIC_NAMES


def _build_shared_negative_pool(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    """Return the shared negative pool of HC, BGL, and LGIN samples."""
    return prediction_frame.loc[prediction_frame["disease_category"].isin([0, 1, 2])].reset_index(drop=True)


def _build_subgroup_frames(prediction_frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build the two prespecified subgroup frames from the shared negative pool."""
    negative_pool = _build_shared_negative_pool(prediction_frame)
    hgin_egc_positive = prediction_frame.loc[prediction_frame["disease_category"].isin([3, 4])]
    agc_positive = prediction_frame.loc[prediction_frame["disease_category"] == 5]
    return {
        "hgin_egc": pd.concat([negative_pool, hgin_egc_positive], ignore_index=True),
        "agc": pd.concat([negative_pool, agc_positive], ignore_index=True),
    }


def _compute_best_model_subgroup_metric_rows(
    *,
    grouping: str,
    prediction_frame: pd.DataFrame,
    threshold_p_raw: float,
    n_boot: int,
    seed: int,
    model_object: Any | None = None,
) -> list[dict[str, float | str]]:
    """Compute the best-model Tier B2 metric rows from persisted predictions only."""
    del model_object
    return [
        {
            "grouping": grouping,
            "view": "test",
            "metric_name": metric_name,
            **_bootstrap_metric_from_frame(
                prediction_frame=prediction_frame,
                metric_name=metric_name,
                threshold_p_raw=threshold_p_raw,
                n_boot=n_boot,
                seed=seed,
            ),
        }
        for metric_name in BEST_MODEL_SUBGROUP_METRICS
    ]


def _build_best_model_subgroup_output_frame(metric_rows: list[dict[str, float | str]]) -> pd.DataFrame:
    """Build the best-model subgroup companion CSV frame."""
    return _build_metric_output_frame(
        metric_rows,
        ["grouping", "view", "metric_name", "estimate", "ci_low", "ci_high"],
    )


def _build_all_models_subgroup_output_frame(metric_rows: list[dict[str, float | str]]) -> pd.DataFrame:
    """Build one all-models subgroup threshold-free frame."""
    return _build_metric_output_frame(
        metric_rows,
        ["model", "metric_name", "estimate", "ci_low", "ci_high"],
    )


def _evaluate_posthoc_a_trigger(
    *,
    full_auc_summary: dict[str, float],
    subgroup_auc_summaries: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Evaluate the post-hoc A subgroup-retrain trigger without executing it."""
    trigger_details: dict[str, dict[str, Any]] = {}
    triggered = False
    full_point = float(full_auc_summary["estimate"])
    for subgroup_name, summary in subgroup_auc_summaries.items():
        condition_1 = float(summary["estimate"]) < (full_point - 0.05)
        condition_2 = float(summary["ci_high"]) < full_point
        subgroup_triggered = bool(condition_1 and condition_2)
        triggered = triggered or subgroup_triggered
        trigger_details[subgroup_name] = {
            "estimate": float(summary["estimate"]),
            "ci_low": float(summary["ci_low"]),
            "ci_high": float(summary["ci_high"]),
            "condition_1_auc_drop_gt_0_05": condition_1,
            "condition_2_ci_high_below_full_point": condition_2,
            "triggered": subgroup_triggered,
        }
    return {"triggered": triggered, "details": trigger_details}


def run_subgroup_analysis(config_path: Path, best_model: str) -> None:
    """Run the Session 11 subgroup analyses without any refitting.

    Args:
        config_path: Path to `configs/modeling_global.yaml`.
        best_model: Canonical best-model short name.
    """
    config = _load_global_config(config_path=config_path)
    test_metadata = _load_test_frame_with_metadata(config=config)
    best_model_payload = _load_model_payload(best_model)
    best_threshold = float(best_model_payload["threshold_p_raw"])
    n_boot = int(config["bootstrap"]["n_boot"])

    best_prediction_frame = _load_view_prediction_frame(model_name=best_model, view_name="test")
    best_prediction_frame = _merge_predictions_with_metadata(
        prediction_frame=best_prediction_frame,
        metadata_frame=test_metadata,
        required_columns=["disease_category"],
    )
    subgroup_frames = _build_subgroup_frames(best_prediction_frame)

    best_model_rows = _compute_best_model_subgroup_metric_rows(
        grouping="full_test",
        prediction_frame=best_prediction_frame,
        threshold_p_raw=best_threshold,
        n_boot=n_boot,
        seed=RANDOM_SEED,
    )
    subgroup_auc_summaries: dict[str, dict[str, float]] = {}
    for subgroup_name, subgroup_frame in subgroup_frames.items():
        subgroup_rows = _compute_best_model_subgroup_metric_rows(
            grouping=subgroup_name,
            prediction_frame=subgroup_frame,
            threshold_p_raw=best_threshold,
            n_boot=n_boot,
            seed=RANDOM_SEED,
        )
        best_model_rows.extend(subgroup_rows)
        subgroup_auc_summaries[subgroup_name] = next(
            row for row in subgroup_rows if row["metric_name"] == "auc_roc"
        )

    best_model_output = _build_best_model_subgroup_output_frame(best_model_rows)
    best_model_csv_path = _resolve_path(config["paths"]["outputs_analysis_dir"]) / "three_view_best_model_with_subgroups.csv"
    best_model_csv_path.parent.mkdir(parents=True, exist_ok=True)
    best_model_output.to_csv(best_model_csv_path, index=False)

    threshold_free_outputs: dict[str, Path] = {
        "hgin_egc": _resolve_path(config["paths"]["outputs_analysis_dir"]) / "subgroup_hgin_egc_all_models.csv",
        "agc": _resolve_path(config["paths"]["outputs_analysis_dir"]) / "subgroup_agc_all_models.csv",
    }
    threshold_free_rows: dict[str, list[dict[str, float | str]]] = {"hgin_egc": [], "agc": []}
    for model_name in MODEL_NAMES:
        model_payload = _load_model_payload(model_name)
        threshold_p_raw = float(model_payload["threshold_p_raw"])
        prediction_frame = _load_view_prediction_frame(model_name=model_name, view_name="test")
        prediction_frame = _merge_predictions_with_metadata(
            prediction_frame=prediction_frame,
            metadata_frame=test_metadata,
            required_columns=["disease_category"],
        )
        model_subgroups = _build_subgroup_frames(prediction_frame)
        for subgroup_name, subgroup_frame in model_subgroups.items():
            for metric_name in THRESHOLD_FREE_METRIC_NAMES:
                threshold_free_rows[subgroup_name].append(
                    {
                        "model": model_name,
                        "metric_name": metric_name,
                        **_bootstrap_metric_from_frame(
                            prediction_frame=subgroup_frame,
                            metric_name=metric_name,
                            threshold_p_raw=threshold_p_raw,
                            n_boot=n_boot,
                            seed=RANDOM_SEED,
                        ),
                    }
                )
        LOGGER.info("Computed subgroup outputs for model=%s.", model_name)

    for subgroup_name, output_path in threshold_free_outputs.items():
        output_frame = _build_all_models_subgroup_output_frame(threshold_free_rows[subgroup_name])
        output_frame.to_csv(output_path, index=False)

    full_auc_summary = next(row for row in best_model_rows if row["grouping"] == "full_test" and row["metric_name"] == "auc_roc")
    trigger_snapshot = _evaluate_posthoc_a_trigger(
        full_auc_summary=full_auc_summary,
        subgroup_auc_summaries=subgroup_auc_summaries,
    )

    summary_markdown_path = _resolve_path(config["paths"]["outputs_analysis_dir"]) / "subgroup_analysis_summary.md"
    summary_lines = [
        "# Subgroup Analysis Summary",
        "",
        f"- Best model: `{best_model}`",
        f"- Best-model subgroup CSV: `{_format_path(best_model_csv_path)}`",
        f"- HGIN+EGC all-models CSV: `{_format_path(threshold_free_outputs['hgin_egc'])}`",
        f"- AGC all-models CSV: `{_format_path(threshold_free_outputs['agc'])}`",
        f"- timestamp_utc: `{_timestamp_utc()}`",
        "",
        "## Subgroup Sizes",
    ]
    for subgroup_name, subgroup_frame in subgroup_frames.items():
        n_total = len(subgroup_frame)
        n_pos = int((subgroup_frame["y_true"] == 1).sum())
        n_neg = int((subgroup_frame["y_true"] == 0).sum())
        summary_lines.append(f"- `{subgroup_name}`: n_total={n_total}, n_pos={n_pos}, n_neg={n_neg}")
    summary_lines.extend(
        [
            "",
            "## Post-Hoc A Snapshot",
            f"- triggered: `{trigger_snapshot['triggered']}`",
        ]
    )
    for subgroup_name, details in trigger_snapshot["details"].items():
        summary_lines.append(
            f"- `{subgroup_name}`: estimate={details['estimate']:.6f}, "
            f"ci=[{details['ci_low']:.6f}, {details['ci_high']:.6f}], "
            f"condition_1={details['condition_1_auc_drop_gt_0_05']}, "
            f"condition_2={details['condition_2_ci_high_below_full_point']}, "
            f"triggered={details['triggered']}"
        )
    _write_text(summary_markdown_path, "\n".join(summary_lines) + "\n")
    LOGGER.info(
        "Wrote subgroup outputs to %s, %s, %s, and %s.",
        _format_path(best_model_csv_path),
        _format_path(threshold_free_outputs["hgin_egc"]),
        _format_path(threshold_free_outputs["agc"]),
        _format_path(summary_markdown_path),
    )
