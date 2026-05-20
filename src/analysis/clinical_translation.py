from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.stage7_part_a_common import (
    CLINICAL_TRANSLATION_METRIC_NAMES,
    RANDOM_SEED,
    _bootstrap_metric_from_frame,
    _compute_prevalence_adjusted_metrics,
    _format_path,
    _get_git_commit_hash,
    _load_global_config,
    _load_model_payload,
    _load_prediction_frame,
    _resolve_path,
    _timestamp_utc,
    _write_text,
)
from src.evaluation.metrics import compute_threshold_dependent_metrics


LOGGER = logging.getLogger(__name__)


def _load_prediction_frame_wrapper(path: Path) -> pd.DataFrame:
    """Load one standard Stage 4b prediction CSV."""
    return _load_prediction_frame(path)


def _assign_binary_risk_tier(
    *,
    probabilities: np.ndarray,
    threshold_p_raw: float,
    tier_mode: str = "binary",
) -> np.ndarray:
    """Assign the prespecified binary high/low risk tier from raw probabilities."""
    if tier_mode != "binary":
        raise ValueError("Session 11 clinical translation supports the binary tier only.")
    return (probabilities >= threshold_p_raw).astype(int)


def _assign_binary_risk_tier_from_frame(
    *,
    prediction_frame: pd.DataFrame,
    threshold_p_raw: float,
    tier_mode: str = "binary",
) -> pd.Series:
    """Assign risk tiers from a prediction frame using `p_raw` only."""
    return pd.Series(
        _assign_binary_risk_tier(
            probabilities=prediction_frame["p_raw"].to_numpy(dtype=float),
            threshold_p_raw=threshold_p_raw,
            tier_mode=tier_mode,
        ),
        index=prediction_frame.index,
        dtype=int,
    )


def _build_clinical_translation_output_frame(metric_rows: list[dict[str, float | str]]) -> pd.DataFrame:
    """Build the persisted long-format clinical translation output."""
    return pd.DataFrame(
        metric_rows,
        columns=["prevalence_scenario", "metric_name", "estimate", "ci_low", "ci_high"],
    )


def run_clinical_translation(config_path: Path, best_model: str) -> None:
    """Run the prespecified binary two-tier clinical translation analysis.

    Args:
        config_path: Path to `configs/modeling_global.yaml`.
        best_model: Canonical short model name, expected to be `minirocket_ridge`.
    """
    config = _load_global_config(config_path=config_path)
    prediction_path = _resolve_path(f"outputs/predictions/binary_{best_model}_test_predictions.csv")
    prediction_frame = _load_prediction_frame_wrapper(prediction_path)
    model_payload = _load_model_payload(best_model)
    threshold_p_raw = float(model_payload["threshold_p_raw"])
    prediction_frame = prediction_frame.copy()
    prediction_frame["high_risk"] = _assign_binary_risk_tier_from_frame(
        prediction_frame=prediction_frame,
        threshold_p_raw=threshold_p_raw,
    )

    y_true = prediction_frame["y_true"].to_numpy(dtype=np.int64)
    p_raw = prediction_frame["p_raw"].to_numpy(dtype=float)
    threshold_metrics = compute_threshold_dependent_metrics(
        y_true=y_true,
        p_raw=p_raw,
        threshold_p_raw=threshold_p_raw,
    )
    prevalence_scenarios = [float(value) for value in config["prevalence_scenarios"]]
    n_boot = int(config["bootstrap"]["n_boot"])

    metric_rows: list[dict[str, float | str]] = []
    for prevalence in prevalence_scenarios:
        for metric_name in CLINICAL_TRANSLATION_METRIC_NAMES:
            if metric_name == "sensitivity":
                summary = _bootstrap_metric_from_frame(
                    prediction_frame=prediction_frame,
                    metric_name="sensitivity",
                    threshold_p_raw=threshold_p_raw,
                    n_boot=n_boot,
                    seed=RANDOM_SEED,
                )
            elif metric_name == "specificity":
                summary = _bootstrap_metric_from_frame(
                    prediction_frame=prediction_frame,
                    metric_name="specificity",
                    threshold_p_raw=threshold_p_raw,
                    n_boot=n_boot,
                    seed=RANDOM_SEED,
                )
            else:
                adjusted_metric_name = {
                    "ppv": "ppv_adjusted",
                    "npv": "npv_adjusted",
                    "nne": "nne_adjusted",
                }[metric_name]
                summary = _bootstrap_metric_from_frame(
                    prediction_frame=prediction_frame,
                    metric_name=adjusted_metric_name,
                    threshold_p_raw=threshold_p_raw,
                    n_boot=n_boot,
                    seed=RANDOM_SEED,
                    prevalence=prevalence,
                )
            metric_rows.append(
                {
                    "prevalence_scenario": prevalence,
                    "metric_name": metric_name,
                    "estimate": summary["estimate"],
                    "ci_low": summary["ci_low"],
                    "ci_high": summary["ci_high"],
                }
            )

    output_frame = _build_clinical_translation_output_frame(metric_rows)
    csv_path = _resolve_path(config["paths"]["outputs_analysis_dir"]) / f"two_tier_risk_stratification_{best_model}.csv"
    markdown_path = _resolve_path(config["paths"]["outputs_analysis_dir"]) / f"two_tier_risk_stratification_{best_model}.md"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(csv_path, index=False)

    binary_metrics = _compute_prevalence_adjusted_metrics(
        sensitivity=float(threshold_metrics["sensitivity"]),
        specificity=float(threshold_metrics["specificity"]),
        prevalence=float(np.mean(y_true)),
    )
    markdown = (
        f"# Two-Tier Risk Stratification — {best_model}\n\n"
        "All threshold-dependent metrics computed in p_raw space using threshold_p_raw per "
        "modeling-guardrails §5; threshold_p_final_equivalent is reported in "
        f"`binary_{best_model}_threshold_mapping.json` for clinical communication only.\n\n"
        "The Session 11 implementation uses the locked binary high-risk definition "
        "`high_risk = (p_raw >= threshold_p_raw)` and explicitly rejects any three-tier configuration.\n\n"
        f"- Input predictions: `{_format_path(prediction_path)}`\n"
        f"- Output CSV: `{_format_path(csv_path)}`\n"
        f"- threshold_p_raw: `{threshold_p_raw:.6f}`\n"
        f"- Prevalence scenarios: `{prevalence_scenarios}`\n"
        f"- Empirical-test sensitivity: `{threshold_metrics['sensitivity']:.6f}`\n"
        f"- Empirical-test specificity: `{threshold_metrics['specificity']:.6f}`\n"
        f"- Empirical-test PPV/NPV/NNE at observed prevalence: "
        f"`{binary_metrics['ppv']:.6f}` / `{binary_metrics['npv']:.6f}` / `{binary_metrics['nne']:.6f}`\n"
        f"- git_commit_hash: `{_get_git_commit_hash()}`\n"
        f"- timestamp_utc: `{_timestamp_utc()}`\n\n"
        "REQUIRES-USER-VERIFICATION: prevalence scenarios `[0.01, 0.05, 0.10]` correspond to the plan §6.1 "
        "clinical scenario ledger and should retain that caveat in manuscript-facing text unless separately confirmed.\n"
    )
    _write_text(markdown_path, markdown)
    LOGGER.info(
        "Wrote clinical translation outputs for %s to %s and %s.",
        best_model,
        _format_path(csv_path),
        _format_path(markdown_path),
    )
