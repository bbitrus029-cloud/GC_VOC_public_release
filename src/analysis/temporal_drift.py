from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.stage7_part_a_common import (
    MODEL_NAMES,
    RANDOM_SEED,
    THRESHOLD_FREE_METRIC_NAMES,
    _bootstrap_metric_from_frame,
    _build_metric_output_frame,
    _format_path,
    _load_global_config,
    _load_model_payload,
    _load_view_prediction_frame,
    _merge_predictions_with_metadata,
    _resolve_path,
    _write_text,
)
from src.analysis.stage7_part_b_common import (
    _load_final_cohort_extended_metadata,
    _parse_collection_time_or_raise,
    _write_dataframe_csv,
)


LOGGER = logging.getLogger(__name__)


def _split_by_collection_time_median(prediction_frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split one prediction frame into first and second halves by collection time.

    Ties at the exact median timestamp stay in the first half, matching the
    locked Session 12 rule.

    Args:
        prediction_frame: Prediction frame containing parsed `collection_time`.

    Returns:
        Mapping with `first_half` and `second_half` dataframes.
    """
    ordered = prediction_frame.sort_values(["collection_time", "sample_id"], kind="mergesort").reset_index(drop=True)
    median_timestamp = ordered["collection_time"].median()
    first_half = ordered.loc[ordered["collection_time"] <= median_timestamp].reset_index(drop=True)
    second_half = ordered.loc[ordered["collection_time"] > median_timestamp].reset_index(drop=True)
    return {"first_half": first_half, "second_half": second_half}


def _build_temporal_output_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the long-format temporal-drift output frame.

    Args:
        rows: Temporal-drift metric rows.

    Returns:
        Ordered dataframe.
    """
    return _build_metric_output_frame(
        rows,
        [
            "model",
            "half",
            "time_range_start",
            "time_range_end",
            "n_total",
            "n_pos",
            "n_neg",
            "metric_name",
            "estimate",
            "ci_low",
            "ci_high",
        ],
    )


def _compute_temporal_rows(
    *,
    model_name: str,
    split_frames: dict[str, pd.DataFrame],
    threshold_p_raw: float,
    n_boot: int,
) -> list[dict[str, Any]]:
    """Compute temporal-drift metric rows for one model and one view.

    Args:
        model_name: Canonical model short name.
        split_frames: Mapping of temporal halves to prediction frames.
        threshold_p_raw: Fixed raw-space threshold.
        n_boot: Bootstrap replicate count.

    Returns:
        Long-format metric rows.
    """
    metric_rows: list[dict[str, Any]] = []
    for half_name, half_frame in split_frames.items():
        n_total = int(len(half_frame))
        n_pos = int((half_frame["y_true"] == 1).sum())
        n_neg = int((half_frame["y_true"] == 0).sum())
        for metric_name in THRESHOLD_FREE_METRIC_NAMES:
            metric_rows.append(
                {
                    "model": model_name,
                    "half": half_name,
                    "time_range_start": half_frame["collection_time"].min().isoformat(),
                    "time_range_end": half_frame["collection_time"].max().isoformat(),
                    "n_total": n_total,
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "metric_name": metric_name,
                    **_bootstrap_metric_from_frame(
                        prediction_frame=half_frame,
                        metric_name=metric_name,
                        threshold_p_raw=threshold_p_raw,
                        n_boot=n_boot,
                        seed=RANDOM_SEED,
                    ),
                }
            )
    return metric_rows


def run_temporal_drift(config_path: Path) -> None:
    """Run the all-model OOF and test temporal-drift sensitivity analysis.

    Args:
        config_path: Path to `configs/modeling_global.yaml`.
    """
    config = _load_global_config(config_path=config_path)
    metadata = _load_final_cohort_extended_metadata(config=config)
    metadata["collection_time"] = _parse_collection_time_or_raise(metadata)
    n_boot = int(config["bootstrap"]["n_boot"])

    oof_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    summary_lines = [
        "# Temporal Drift Summary",
        "",
        "- Interpretation flag: a second-half AUC-ROC point estimate drop greater than 0.05 is descriptive only and does not trigger Post-hoc C.",
        "",
        "| model | test_auc_roc_first | test_auc_roc_second | test_delta_second_minus_first | oof_delta_second_minus_first | flag_test_drop_gt_0_05 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for model_name in MODEL_NAMES:
        model_payload = _load_model_payload(model_name)
        threshold_p_raw = float(model_payload["threshold_p_raw"])

        view_rows: dict[str, list[dict[str, Any]]] = {}
        for view_name in ("test", "nested_cv_oof"):
            prediction_frame = _load_view_prediction_frame(model_name=model_name, view_name=view_name)
            merged = _merge_predictions_with_metadata(
                prediction_frame=prediction_frame,
                metadata_frame=metadata,
                required_columns=["collection_time"],
            )
            merged["collection_time"] = _parse_collection_time_or_raise(merged)
            split_frames = _split_by_collection_time_median(merged)
            if view_name == "test":
                current_rows = _compute_temporal_rows(
                    model_name=model_name,
                    split_frames=split_frames,
                    threshold_p_raw=threshold_p_raw,
                    n_boot=n_boot,
                )
                test_rows.extend(current_rows)
                view_rows["test"] = current_rows
            else:
                current_rows = _compute_temporal_rows(
                    model_name=model_name,
                    split_frames=split_frames,
                    threshold_p_raw=threshold_p_raw,
                    n_boot=n_boot,
                )
                oof_rows.extend(current_rows)
                view_rows["nested_cv_oof"] = current_rows

        first_test_auc = next(
            row["estimate"]
            for row in view_rows["test"]
            if row["half"] == "first_half" and row["metric_name"] == "auc_roc"
        )
        second_test_auc = next(
            row["estimate"]
            for row in view_rows["test"]
            if row["half"] == "second_half" and row["metric_name"] == "auc_roc"
        )
        first_oof_auc = next(
            row["estimate"]
            for row in view_rows["nested_cv_oof"]
            if row["half"] == "first_half" and row["metric_name"] == "auc_roc"
        )
        second_oof_auc = next(
            row["estimate"]
            for row in view_rows["nested_cv_oof"]
            if row["half"] == "second_half" and row["metric_name"] == "auc_roc"
        )
        test_delta = float(second_test_auc - first_test_auc)
        oof_delta = float(second_oof_auc - first_oof_auc)
        summary_lines.append(
            "| "
            f"{model_name} | {float(first_test_auc):.6f} | {float(second_test_auc):.6f} | {test_delta:.6f} | "
            f"{oof_delta:.6f} | {test_delta < -0.05} |"
        )
        LOGGER.info(
            "Computed temporal drift for model=%s: test_delta_auc_roc=%.6f, oof_delta_auc_roc=%.6f.",
            model_name,
            test_delta,
            oof_delta,
        )

    oof_output = _build_temporal_output_frame(oof_rows)
    test_output = _build_temporal_output_frame(test_rows)
    oof_path = _resolve_path(config["paths"]["outputs_analysis_dir"]) / "temporal_drift_oof_all_models.csv"
    test_path = _resolve_path(config["paths"]["outputs_analysis_dir"]) / "temporal_drift_test_all_models.csv"
    summary_path = _resolve_path(config["paths"]["outputs_analysis_dir"]) / "temporal_drift_summary.md"
    _write_dataframe_csv(oof_path, oof_output)
    _write_dataframe_csv(test_path, test_output)
    _write_text(summary_path, "\n".join(summary_lines) + "\n")
    LOGGER.info(
        "Wrote temporal drift outputs to %s, %s, and %s.",
        _format_path(oof_path),
        _format_path(test_path),
        _format_path(summary_path),
    )


__all__ = [
    "_build_temporal_output_frame",
    "_split_by_collection_time_median",
    "run_temporal_drift",
]
