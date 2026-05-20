from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from src.analysis.data_split import DISEASE_CATEGORY_LABELS
from src.analysis.stage7_part_a_common import (
    _build_metric_output_frame,
    _format_path,
    _load_global_config,
    _load_test_frame_with_metadata,
    _load_view_prediction_frame,
    _merge_predictions_with_metadata,
    _resolve_path,
    _write_text,
)
from src.evaluation.statistical_tests import holm_bonferroni, jonckheere_terpstra, spearman_correlation


LOGGER = logging.getLogger(__name__)
JT_PRIMARY_PVALUE_KEY = "p_one_sided_upper"
ADJACENT_PAIR_DEFINITIONS = [
    ("HC-BGL", 0, 1),
    ("BGL-LGIN", 1, 2),
    ("LGIN-HGIN", 2, 3),
    ("HGIN-EGC", 3, 4),
    ("EGC-AGC", 4, 5),
]


def _build_progression_output_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the per-sample progression CSV frame.

    Args:
        rows: Per-sample records.

    Returns:
        Ordered progression dataframe.
    """
    return _build_metric_output_frame(rows, ["sample_id", "y_true", "p_cal", "disease_category"])


def _apply_holm_to_adjacent_pairs(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach Holm-Bonferroni outputs while preserving clinical pair order.

    Args:
        pair_rows: Adjacent-pair rows containing `pair_name` and `raw_p`.

    Returns:
        Pair rows augmented with Holm-adjusted fields in original order.
    """
    adjusted = holm_bonferroni([(row["pair_name"], float(row["raw_p"])) for row in pair_rows], alpha=0.05)
    adjusted_by_name = {row["test_name"]: row for row in adjusted}
    return [
        {
            **row,
            "holm_adjusted_p": adjusted_by_name[row["pair_name"]]["holm_adjusted_p"],
            "reject_null": adjusted_by_name[row["pair_name"]]["reject_null"],
        }
        for row in pair_rows
    ]


def _compute_group_descriptives(prediction_frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Compute one descriptive row per disease category.

    Args:
        prediction_frame: Best-model test predictions merged with disease category.

    Returns:
        List of descriptive summary rows.
    """
    descriptive_rows: list[dict[str, Any]] = []
    for disease_category, label in DISEASE_CATEGORY_LABELS.items():
        category_frame = prediction_frame.loc[prediction_frame["disease_category"] == disease_category].reset_index(drop=True)
        q1 = float(category_frame["p_cal"].quantile(0.25))
        q3 = float(category_frame["p_cal"].quantile(0.75))
        descriptive_rows.append(
            {
                "disease_category": disease_category,
                "label": label,
                "n": int(len(category_frame)),
                "median_p_cal": float(category_frame["p_cal"].median()),
                "q1_p_cal": q1,
                "q3_p_cal": q3,
                "iqr_p_cal": float(q3 - q1),
            }
        )
    return descriptive_rows


def run_progression_trajectory(config_path: Path, best_model: str) -> None:
    """Run the best-model disease-progression trajectory analysis.

    This module intentionally uses `p_cal`, not `p_raw`, because plan §2.19
    defines a descriptive progression-distribution analysis rather than a
    threshold-dependent decision analysis.

    Args:
        config_path: Path to `configs/modeling_global.yaml`.
        best_model: Canonical best-model short name.
    """
    config = _load_global_config(config_path=config_path)
    test_metadata = _load_test_frame_with_metadata(config=config)
    prediction_frame = _load_view_prediction_frame(model_name=best_model, view_name="test")
    merged = _merge_predictions_with_metadata(
        prediction_frame=prediction_frame,
        metadata_frame=test_metadata,
        required_columns=["disease_category"],
    )
    LOGGER.info("Loaded progression trajectory inputs for model=%s with n=%s.", best_model, len(merged))

    output_frame = merged.loc[:, ["sample_id", "y_true", "p_cal", "disease_category"]].sort_values(
        ["disease_category", "p_cal", "sample_id"],
        kind="mergesort",
    )
    output_frame = _build_progression_output_frame(output_frame.to_dict("records"))
    output_csv_path = _resolve_path(config["paths"]["outputs_analysis_dir"]) / f"progression_trajectory_{best_model}.csv"
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(output_csv_path, index=False)
    LOGGER.info("Wrote progression per-sample CSV to %s.", _format_path(output_csv_path))

    jt_result = jonckheere_terpstra(
        values=merged["p_cal"].to_numpy(dtype=float),
        ordered_groups=merged["disease_category"].to_numpy(dtype=int),
    )
    if JT_PRIMARY_PVALUE_KEY not in jt_result:
        raise AssertionError(
            f"jonckheere_terpstra must return {JT_PRIMARY_PVALUE_KEY}; found keys {sorted(jt_result)}."
        )
    spearman_result = spearman_correlation(
        x=merged["p_cal"].to_numpy(dtype=float),
        y=merged["disease_category"].astype(float).to_numpy(),
    )

    pair_rows: list[dict[str, Any]] = []
    for pair_name, left_category, right_category in ADJACENT_PAIR_DEFINITIONS:
        left_values = merged.loc[merged["disease_category"] == left_category, "p_cal"].to_numpy(dtype=float)
        right_values = merged.loc[merged["disease_category"] == right_category, "p_cal"].to_numpy(dtype=float)
        mann_whitney_result = mannwhitneyu(left_values, right_values, alternative="two-sided")
        if np.isnan(float(mann_whitney_result.statistic)) or np.isnan(float(mann_whitney_result.pvalue)):
            raise AssertionError(f"Mann-Whitney U returned NaN for pair {pair_name}.")
        pair_rows.append(
            {
                "pair_name": pair_name,
                "n_left": int(len(left_values)),
                "n_right": int(len(right_values)),
                "u_statistic": float(mann_whitney_result.statistic),
                "raw_p": float(mann_whitney_result.pvalue),
            }
        )
    pair_rows = _apply_holm_to_adjacent_pairs(pair_rows)
    descriptives = _compute_group_descriptives(merged)

    markdown_lines = [
        "# Disease Progression Trajectory",
        "",
        f"- Best model: `{best_model}`",
        f"- Per-sample CSV: `{_format_path(output_csv_path)}`",
        "- Analysis space: `p_cal` (descriptive distribution only; no threshold-dependent decision metric is computed here).",
        "",
        "## Jonckheere-Terpstra",
        (
            f"- J = {jt_result['J']}, mean_j = {jt_result['mean_j']:.6f}, variance_j = {jt_result['variance_j']:.6f}, "
            f"z = {jt_result['z']:.6f}, p_one_sided_upper = {jt_result[JT_PRIMARY_PVALUE_KEY]:.6g}"
        ),
        "- Primary p-value is `p_one_sided_upper` per the locked npordtests-aligned convention.",
        "",
        "## Spearman Correlation",
        (
            f"- rho = {spearman_result['rho']:.6f}, p_two_sided = {spearman_result['p_two_sided']:.6g}, "
            f"n = {spearman_result['n']}"
        ),
        "",
        "## Adjacent-Pair Mann-Whitney Tests",
        "",
        "| Pair | n_left | n_right | U | raw_p | holm_adjusted_p | reject_null |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in pair_rows:
        markdown_lines.append(
            "| "
            f"{row['pair_name']} | {row['n_left']} | {row['n_right']} | {row['u_statistic']:.6f} | "
            f"{row['raw_p']:.6g} | {row['holm_adjusted_p']:.6g} | {row['reject_null']} |"
        )

    markdown_lines.extend(
        [
            "",
            "## Per-Group Descriptives",
            "",
            "| Category | n | median_p_cal | q1_p_cal | q3_p_cal | iqr_p_cal |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in descriptives:
        markdown_lines.append(
            "| "
            f"{row['label']} | {row['n']} | {row['median_p_cal']:.6f} | {row['q1_p_cal']:.6f} | "
            f"{row['q3_p_cal']:.6f} | {row['iqr_p_cal']:.6f} |"
        )
    markdown_lines.extend(
        [
            "",
            "## Interpretation Guidance",
            "- A negative Spearman rho or a large `p_one_sided_upper` would argue against a monotone increase in calibrated risk across the ordered disease trajectory.",
            "- The adjacent-pair tests are descriptive complements to the global JT trend test and are Holm-corrected across exactly five prespecified adjacent comparisons.",
        ]
    )
    output_markdown_path = _resolve_path(config["paths"]["outputs_analysis_dir"]) / f"progression_trajectory_{best_model}.md"
    _write_text(output_markdown_path, "\n".join(markdown_lines) + "\n")
    LOGGER.info("Wrote progression trajectory markdown to %s.", _format_path(output_markdown_path))


__all__ = [
    "ADJACENT_PAIR_DEFINITIONS",
    "JT_PRIMARY_PVALUE_KEY",
    "_apply_holm_to_adjacent_pairs",
    "_build_progression_output_frame",
    "run_progression_trajectory",
]
