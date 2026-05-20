from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
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
    _resolve_path,
    _write_text,
)
from src.analysis.stage7_part_b_common import (
    _compute_binary_risk_difference,
    _compute_continuous_smd,
    _load_final_cohort_extended_metadata,
    _merge_predictions_with_optional_metadata,
    _write_dataframe_csv,
)
from src.evaluation.statistical_tests import fisher_exact_2x2


LOGGER = logging.getLogger(__name__)
SUBSET_ORDER = ("full", "fallback_used_false", "padded_false", "both_false")


def _build_subset_frames(prediction_frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Construct the four locked fallback/padded sensitivity subsets.

    Args:
        prediction_frame: Test prediction frame merged with fallback/padded flags.

    Returns:
        Mapping from subset name to dataframe.
    """
    return {
        "full": prediction_frame.reset_index(drop=True),
        "fallback_used_false": prediction_frame.loc[~prediction_frame["fallback_used"].astype(bool)].reset_index(drop=True),
        "padded_false": prediction_frame.loc[~prediction_frame["padded"].astype(bool)].reset_index(drop=True),
        "both_false": prediction_frame.loc[
            (~prediction_frame["fallback_used"].astype(bool)) & (~prediction_frame["padded"].astype(bool))
        ].reset_index(drop=True),
    }


def _build_binary_association_table(indicator: pd.Series, outcome: pd.Series) -> np.ndarray:
    """Build a 2x2 table for one binary indicator against one binary outcome.

    Args:
        indicator: Indicator series such as `fallback_used`.
        outcome: Binary outcome series such as `misclassified`.

    Returns:
        A 2x2 numpy array with rows `[indicator=True, indicator=False]` and
        columns `[outcome=True, outcome=False]`.
    """
    indicator_bool = indicator.astype(bool)
    outcome_bool = outcome.astype(bool)
    return np.array(
        [
            [int((indicator_bool & outcome_bool).sum()), int((indicator_bool & ~outcome_bool).sum())],
            [int((~indicator_bool & outcome_bool).sum()), int((~indicator_bool & ~outcome_bool).sum())],
        ],
        dtype=int,
    )


def _compute_subset_metric_rows(
    *,
    model_name: str,
    subset_name: str,
    prediction_frame: pd.DataFrame,
    threshold_p_raw: float,
    n_boot: int,
    seed: int,
    model_object: Any | None = None,
) -> list[dict[str, Any]]:
    """Compute the three locked threshold-free metrics from persisted predictions.

    Args:
        model_name: Canonical model short name.
        subset_name: Sensitivity subset name.
        prediction_frame: Subset prediction frame.
        threshold_p_raw: Fixed raw-space threshold from the final model payload.
        n_boot: Bootstrap replicate count.
        seed: Deterministic bootstrap seed.
        model_object: Unused sentinel to verify that no refit occurs.

    Returns:
        List of long-format metric rows.
    """
    del model_object
    n_total = int(len(prediction_frame))
    n_pos = int((prediction_frame["y_true"] == 1).sum())
    n_neg = int((prediction_frame["y_true"] == 0).sum())
    return [
        {
            "model": model_name,
            "subset": subset_name,
            "n_total": n_total,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "metric_name": metric_name,
            **_bootstrap_metric_from_frame(
                prediction_frame=prediction_frame,
                metric_name=metric_name,
                threshold_p_raw=threshold_p_raw,
                n_boot=n_boot,
                seed=seed,
            ),
        }
        for metric_name in THRESHOLD_FREE_METRIC_NAMES
    ]


def _build_fallback_output_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the all-models fallback/padded sensitivity CSV frame.

    Args:
        rows: Long-format metric rows.

    Returns:
        Ordered dataframe.
    """
    return _build_metric_output_frame(
        rows,
        ["model", "subset", "n_total", "n_pos", "n_neg", "metric_name", "estimate", "ci_low", "ci_high"],
    )


def run_fallback_padded_sensitivity(config_path: Path, best_model: str) -> None:
    """Run the all-model fallback/padded sensitivity analysis.

    Args:
        config_path: Path to `configs/modeling_global.yaml`.
        best_model: Canonical best-model short name for Fisher exact interpretation.
    """
    config = _load_global_config(config_path=config_path)
    metadata = _load_final_cohort_extended_metadata(config=config)
    n_boot = int(config["bootstrap"]["n_boot"])

    metric_rows: list[dict[str, Any]] = []
    for model_name in MODEL_NAMES:
        model_payload = _load_model_payload(model_name)
        prediction_frame = _load_view_prediction_frame(model_name=model_name, view_name="test")
        merged = _merge_predictions_with_optional_metadata(
            prediction_frame=prediction_frame,
            metadata_frame=metadata,
            required_columns=["fallback_used", "padded"],
            optional_columns=[],
        )
        subset_frames = _build_subset_frames(merged)
        for subset_name in SUBSET_ORDER:
            metric_rows.extend(
                _compute_subset_metric_rows(
                    model_name=model_name,
                    subset_name=subset_name,
                    prediction_frame=subset_frames[subset_name],
                    threshold_p_raw=float(model_payload["threshold_p_raw"]),
                    n_boot=n_boot,
                    seed=RANDOM_SEED,
                )
            )
        LOGGER.info("Computed fallback/padded sensitivity metrics for model=%s.", model_name)
    output_frame = _build_fallback_output_frame(metric_rows)
    output_path = _resolve_path(config["paths"]["outputs_analysis_dir"]) / "fallback_padded_sensitivity_all_models.csv"
    _write_dataframe_csv(output_path, output_frame)

    subset_characteristics_rows: list[dict[str, Any]] = []
    baseline_variables = (
        ("age", "continuous"),
        ("sex", "binary"),
        ("bmi", "continuous"),
        ("smoking", "binary"),
        ("drinking", "binary"),
        ("family_history", "binary"),
        ("hp_infection", "binary"),
        ("disease_category", "continuous"),
    )
    for indicator_name in ("fallback_used", "padded"):
        positive_group = metadata.loc[metadata[indicator_name] == 1].reset_index(drop=True)
        negative_group = metadata.loc[metadata[indicator_name] == 0].reset_index(drop=True)
        for variable_name, variable_type in baseline_variables:
            smd_value = (
                _compute_binary_risk_difference(positive_group[variable_name], negative_group[variable_name])
                if variable_type == "binary"
                else _compute_continuous_smd(positive_group[variable_name], negative_group[variable_name])
            )
            subset_characteristics_rows.append(
                {
                    "indicator": indicator_name,
                    "variable_name": variable_name,
                    "variable_type": variable_type,
                    "group_true_n": int(len(positive_group)),
                    "group_false_n": int(len(negative_group)),
                    "smd": smd_value,
                }
            )
    subset_characteristics_frame = _build_metric_output_frame(
        subset_characteristics_rows,
        ["indicator", "variable_name", "variable_type", "group_true_n", "group_false_n", "smd"],
    )
    subset_characteristics_path = (
        _resolve_path(config["paths"]["outputs_analysis_dir"]) / "fallback_padded_subset_characteristics.csv"
    )
    _write_dataframe_csv(subset_characteristics_path, subset_characteristics_frame)

    best_prediction_frame = _load_view_prediction_frame(model_name=best_model, view_name="test")
    best_merged = _merge_predictions_with_optional_metadata(
        prediction_frame=best_prediction_frame,
        metadata_frame=metadata,
        required_columns=["fallback_used", "padded"],
        optional_columns=[],
    )
    misclassified = best_merged["y_pred"] != best_merged["y_true"]
    fallback_table = _build_binary_association_table(best_merged["fallback_used"], misclassified)
    padded_table = _build_binary_association_table(best_merged["padded"], misclassified)
    fallback_result = fisher_exact_2x2(fallback_table)
    padded_result = fisher_exact_2x2(padded_table)

    markdown_lines = [
        "# Fallback/Padded Misclassification Association",
        "",
        f"- Best model: `{best_model}`",
        f"- All-models sensitivity CSV: `{_format_path(output_path)}`",
        f"- Subset-characteristics CSV: `{_format_path(subset_characteristics_path)}`",
        "",
        "## Fallback Used vs Misclassification",
        f"- table = {fallback_table.tolist()}",
        f"- odds_ratio = {fallback_result['odds_ratio']:.6f}",
        f"- p_two_sided = {fallback_result['p_two_sided']:.6g}",
        "",
        "## Padded vs Misclassification",
        f"- table = {padded_table.tolist()}",
        f"- odds_ratio = {padded_result['odds_ratio']:.6f}",
        f"- p_two_sided = {padded_result['p_two_sided']:.6g}",
        "",
        "## Interpretation",
        (
            "- Per §2.16: p < 0.05 with odds_ratio > 2 suggests association with misclassification; "
            "p < 0.05 with odds_ratio < 0.5 suggests counterintuitively better classification; "
            "p >= 0.05 suggests no detectable association."
        ),
    ]
    markdown_path = _resolve_path(config["paths"]["outputs_analysis_dir"]) / f"fallback_padded_misclass_{best_model}.md"
    _write_text(markdown_path, "\n".join(markdown_lines) + "\n")
    LOGGER.info("Wrote fallback/padded misclassification markdown to %s.", _format_path(markdown_path))


__all__ = [
    "_build_binary_association_table",
    "_build_fallback_output_frame",
    "_build_subset_frames",
    "_compute_subset_metric_rows",
    "run_fallback_padded_sensitivity",
]
