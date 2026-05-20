from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.stage7_part_a_common import (
    _format_path,
    _get_git_commit_hash,
    _load_global_config,
    _load_prediction_frame,
    _resolve_path,
    _timestamp_utc,
    _write_text,
)


LOGGER = logging.getLogger(__name__)
POLICY_THRESHOLDS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)


def _build_threshold_grid() -> np.ndarray:
    """Return the prespecified 98-point DCA threshold grid."""
    return np.arange(0.01, 0.99, 0.01, dtype=float)


def _compute_net_benefit_at_threshold(
    *,
    y_true: np.ndarray,
    p_raw: np.ndarray,
    threshold: float,
    prevalence: float,
) -> dict[str, float]:
    """Compute model, treat-all, and treat-none net benefit at one threshold."""
    y_pred = (p_raw >= threshold).astype(int)
    positives = y_true == 1
    negatives = y_true == 0
    tpr = float(np.mean(y_pred[positives] == 1))
    fpr = float(np.mean(y_pred[negatives] == 1))
    odds_term = float(threshold / (1.0 - threshold))
    model_net_benefit = float((tpr * prevalence) - (fpr * (1.0 - prevalence) * odds_term))
    treat_all_net_benefit = float(prevalence - ((1.0 - prevalence) * odds_term))
    return {
        "model": model_net_benefit,
        "treat_all": treat_all_net_benefit,
        "treat_none": 0.0,
    }


def _build_dca_output_frame(
    *,
    y_true: np.ndarray,
    p_raw: np.ndarray,
    threshold_grid: np.ndarray,
    prevalence_scenarios: list[float],
) -> pd.DataFrame:
    """Build the long-format DCA output frame."""
    metric_rows: list[dict[str, float | str]] = []
    for prevalence in prevalence_scenarios:
        for threshold in threshold_grid:
            net_benefits = _compute_net_benefit_at_threshold(
                y_true=y_true,
                p_raw=p_raw,
                threshold=float(threshold),
                prevalence=float(prevalence),
            )
            for curve_name, net_benefit in net_benefits.items():
                metric_rows.append(
                    {
                        "prevalence_scenario": float(prevalence),
                        "threshold": float(threshold),
                        "curve": curve_name,
                        "net_benefit": float(net_benefit),
                    }
                )
    return pd.DataFrame(metric_rows, columns=["prevalence_scenario", "threshold", "curve", "net_benefit"])


def _policy_threshold_table(policy_rows: pd.DataFrame) -> str:
    """Render the policy-threshold DCA summary table in plain markdown."""
    header = "| prevalence_scenario | threshold | net_benefit |"
    divider = "| --- | --- | --- |"
    body = [
        f"| {row.prevalence_scenario:.2f} | {row.threshold:.2f} | {row.net_benefit:.6f} |"
        for row in policy_rows.itertuples(index=False)
    ]
    return "\n".join([header, divider, *body])


def run_dca(config_path: Path, best_model: str) -> None:
    """Run Stage 7 Part A decision-curve analysis for the locked best model.

    Args:
        config_path: Path to `configs/modeling_global.yaml`.
        best_model: Canonical short model name, expected to be `minirocket_ridge`.
    """
    config = _load_global_config(config_path=config_path)
    prediction_path = Path("outputs/predictions") / f"binary_{best_model}_test_predictions.csv"
    prediction_frame = _load_prediction_frame(_resolve_path(str(prediction_path)))
    y_true = prediction_frame["y_true"].to_numpy(dtype=np.int64)
    p_raw = prediction_frame["p_raw"].to_numpy(dtype=float)
    prevalence_test = float(np.mean(y_true))
    threshold_grid = _build_threshold_grid()
    prevalence_scenarios = [float(value) for value in config["prevalence_scenarios"]]

    output_frame = _build_dca_output_frame(
        y_true=y_true,
        p_raw=p_raw,
        threshold_grid=threshold_grid,
        prevalence_scenarios=prevalence_scenarios,
    )
    csv_path = _resolve_path(config["paths"]["outputs_analysis_dir"]) / f"dca_{best_model}.csv"
    markdown_path = _resolve_path(config["paths"]["outputs_analysis_dir"]) / f"dca_{best_model}.md"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(csv_path, index=False)

    policy_rows = output_frame.loc[
        output_frame["curve"] == "model",
        ["prevalence_scenario", "threshold", "net_benefit"],
    ]
    policy_rows = policy_rows.loc[policy_rows["threshold"].isin(POLICY_THRESHOLDS)].reset_index(drop=True)
    policy_table = _policy_threshold_table(policy_rows)
    markdown = (
        f"# DCA — {best_model}\n\n"
        "Decision curve analysis on the locked test predictions only. TPR/FPR are computed on the empirical test set "
        "at each threshold and reweighted to the prespecified prevalence scenarios; no resampling or refitting is used.\n\n"
        f"- Input predictions: `{_format_path(_resolve_path(str(prediction_path)))}`\n"
        f"- Output CSV: `{_format_path(csv_path)}`\n"
        f"- Empirical test prevalence: `{prevalence_test:.6f}`\n"
        f"- Threshold grid: `np.arange(0.01, 0.99, 0.01)`\n"
        f"- Prevalence scenarios: `{prevalence_scenarios}`\n"
        f"- git_commit_hash: `{_get_git_commit_hash()}`\n"
        f"- timestamp_utc: `{_timestamp_utc()}`\n\n"
        "## Policy-Relevant Thresholds\n\n"
        f"{policy_table}\n"
    )
    _write_text(markdown_path, markdown)
    LOGGER.info(
        "Wrote DCA outputs for %s to %s and %s.",
        best_model,
        _format_path(csv_path),
        _format_path(markdown_path),
    )
