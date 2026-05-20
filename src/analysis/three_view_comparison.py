from __future__ import annotations

"""Tier B1 three-view comparison for all six final models.

The Stage 4b `train_in_sample.csv` artifacts begin with a leading `# WARNING`
line documenting that in-sample predictions are an overfitting upper bound and
are not used for claims. This module therefore loads train-in-sample CSVs with
`pd.read_csv(path, comment="#")` so the warning line is skipped during parsing.
"""

import logging
from pathlib import Path

import pandas as pd

from src.analysis.stage7_part_a_common import (
    MODEL_NAMES,
    RANDOM_SEED,
    THREE_VIEW_METRIC_NAMES,
    _bootstrap_metric_from_frame,
    _build_metric_output_frame,
    _format_path,
    _load_global_config,
    _load_model_payload,
    _load_view_prediction_frame,
    _resolve_path,
    _timestamp_utc,
    _write_text,
)


LOGGER = logging.getLogger(__name__)
SUPPORTED_VIEWS = ("train_in_sample", "nested_cv_oof", "test")


def _load_prediction_frame(path: Path) -> pd.DataFrame:
    """Load a train-in-sample CSV while skipping the leading warning comment line."""
    return pd.read_csv(path, comment="#")


def _extract_model_thresholds(model_payloads: dict[str, dict[str, float]]) -> dict[str, float]:
    """Extract the locked per-model raw-probability thresholds."""
    return {model_name: float(payload["threshold_p_raw"]) for model_name, payload in model_payloads.items()}


def _compute_view_metrics(
    *,
    prediction_frame: pd.DataFrame,
    threshold_p_raw: float,
    n_boot: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Compute the Tier B1 metric bundle for one model/view pair."""
    return {
        metric_name: _bootstrap_metric_from_frame(
            prediction_frame=prediction_frame,
            metric_name=metric_name,
            threshold_p_raw=threshold_p_raw,
            n_boot=n_boot,
            seed=seed,
        )
        for metric_name in THREE_VIEW_METRIC_NAMES
    }


def _build_three_view_output_frame(metric_rows: list[dict[str, float | str]]) -> pd.DataFrame:
    """Build the persisted long-format three-view output."""
    return _build_metric_output_frame(
        metric_rows,
        ["model", "view", "metric_name", "estimate", "ci_low", "ci_high"],
    )


def run_three_view_comparison(config_path: Path, best_model: str) -> None:
    """Run the all-model three-view comparison for Session 11.

    Args:
        config_path: Path to `configs/modeling_global.yaml`.
        best_model: Locked best-model name, used only for audit sidecar context.
    """
    del best_model
    config = _load_global_config(config_path=config_path)
    n_boot = int(config["bootstrap"]["n_boot"])
    model_payloads = {model_name: _load_model_payload(model_name) for model_name in MODEL_NAMES}
    threshold_map = _extract_model_thresholds(model_payloads)

    metric_rows: list[dict[str, float | str]] = []
    for model_name in MODEL_NAMES:
        for view_name in SUPPORTED_VIEWS:
            if view_name == "train_in_sample":
                prediction_frame = _load_prediction_frame(
                    _resolve_path(f"outputs/predictions/binary_{model_name}_train_in_sample.csv")
                )
            else:
                prediction_frame = _load_view_prediction_frame(model_name=model_name, view_name=view_name)
            metric_map = _compute_view_metrics(
                prediction_frame=prediction_frame,
                threshold_p_raw=threshold_map[model_name],
                n_boot=n_boot,
                seed=RANDOM_SEED,
            )
            for metric_name, summary in metric_map.items():
                metric_rows.append(
                    {
                        "model": model_name,
                        "view": view_name,
                        "metric_name": metric_name,
                        "estimate": summary["estimate"],
                        "ci_low": summary["ci_low"],
                        "ci_high": summary["ci_high"],
                    }
                )
            LOGGER.info(
                "Computed three-view metrics for model=%s, view=%s using threshold_p_raw=%.6f.",
                model_name,
                view_name,
                threshold_map[model_name],
            )

    output_frame = _build_three_view_output_frame(metric_rows)
    csv_path = _resolve_path(config["paths"]["outputs_analysis_dir"]) / "three_view_all_models_full_set.csv"
    markdown_path = _resolve_path(config["paths"]["outputs_analysis_dir"]) / "three_view_all_models_full_set.md"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(csv_path, index=False)

    markdown = (
        "# Three-View All-Models Full Set\n\n"
        "In-sample predictions are an overfitting upper-bound diagnostic and are not used for any claim.\n\n"
        f"- Output CSV: `{_format_path(csv_path)}`\n"
        f"- Views: `{SUPPORTED_VIEWS}`\n"
        f"- n_boot: `{n_boot}`\n"
        f"- random_seed: `{RANDOM_SEED}`\n"
        f"- timestamp_utc: `{_timestamp_utc()}`\n"
    )
    _write_text(markdown_path, markdown)
    LOGGER.info(
        "Wrote three-view outputs to %s and %s.",
        _format_path(csv_path),
        _format_path(markdown_path),
    )
