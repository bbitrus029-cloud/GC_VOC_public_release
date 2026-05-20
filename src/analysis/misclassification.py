from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.data_split import DISEASE_CATEGORY_LABELS
from src.analysis.stage7_part_a_common import (
    _format_path,
    _load_global_config,
    _load_model_payload,
    _load_view_prediction_frame,
    _resolve_path,
    _write_text,
)
from src.analysis.stage7_part_b_common import (
    TUMOR_MARKER_COLUMNS,
    _load_final_cohort_extended_metadata,
    _merge_predictions_with_optional_metadata,
    _write_dataframe_csv,
)
from src.evaluation.statistical_tests import fisher_exact_2x2


LOGGER = logging.getLogger(__name__)
QUADRANT_ORDER = ("TP", "TN", "FP", "FN")
BAND_ORDER = ("low", "mid", "high")


def _assign_selection_bands(quadrant_frame: pd.DataFrame) -> pd.DataFrame:
    """Assign low/mid/high distance bands within one quadrant.

    Args:
        quadrant_frame: One-quadrant dataframe sorted arbitrarily.

    Returns:
        Dataframe with `distance_to_threshold` and `selection_band`.
    """
    ordered = quadrant_frame.copy().sort_values(["distance_to_threshold", "sample_id"], kind="mergesort").reset_index(
        drop=True
    )
    n_rows = len(ordered)
    first_cut = int(np.ceil(n_rows / 3.0))
    second_cut = int(np.ceil((2.0 * n_rows) / 3.0))
    band_labels: list[str] = []
    for row_idx in range(n_rows):
        if row_idx < first_cut:
            band_labels.append("low")
        elif row_idx < second_cut:
            band_labels.append("mid")
        else:
            band_labels.append("high")
    ordered["selection_band"] = band_labels
    return ordered


def _select_representative_samples_by_band(
    *,
    prediction_frame: pd.DataFrame,
    threshold_p_raw: float,
    max_per_quadrant: int = 5,
) -> pd.DataFrame:
    """Select representative samples per quadrant by distance band and subtype diversity.

    Args:
        prediction_frame: Best-model test prediction frame with `quadrant`.
        threshold_p_raw: Fixed raw-space threshold for distance calculations.
        max_per_quadrant: Maximum number of representatives per quadrant.

    Returns:
        Representative-sample dataframe.
    """
    selected_frames: list[pd.DataFrame] = []
    for quadrant in QUADRANT_ORDER:
        quadrant_frame = prediction_frame.loc[prediction_frame["quadrant"] == quadrant].copy()
        if quadrant_frame.empty:
            continue
        quadrant_frame["distance_to_threshold"] = (quadrant_frame["p_raw"] - threshold_p_raw).abs()
        quadrant_frame = _assign_selection_bands(quadrant_frame)

        chosen_sample_ids: set[int] = set()
        chosen_categories: set[int] = set()
        chosen_rows: list[pd.Series] = []

        for selection_band in BAND_ORDER:
            band_candidates = quadrant_frame.loc[
                (quadrant_frame["selection_band"] == selection_band)
                & (~quadrant_frame["sample_id"].isin(chosen_sample_ids))
            ].copy()
            if band_candidates.empty:
                continue
            band_candidates["is_new_category"] = ~band_candidates["disease_category"].isin(chosen_categories)
            band_candidates = band_candidates.sort_values(
                ["is_new_category", "distance_to_threshold", "sample_id"],
                ascending=[False, True, True],
                kind="mergesort",
            )
            chosen_row = band_candidates.iloc[0]
            chosen_rows.append(chosen_row)
            chosen_sample_ids.add(int(chosen_row["sample_id"]))
            chosen_categories.add(int(chosen_row["disease_category"]))

        if len(chosen_rows) < max_per_quadrant:
            remaining = quadrant_frame.loc[~quadrant_frame["sample_id"].isin(chosen_sample_ids)].copy()
            while len(chosen_rows) < max_per_quadrant and not remaining.empty:
                remaining["is_new_category"] = ~remaining["disease_category"].isin(chosen_categories)
                remaining = remaining.sort_values(
                    ["is_new_category", "distance_to_threshold", "sample_id"],
                    ascending=[False, True, True],
                    kind="mergesort",
                )
                chosen_row = remaining.iloc[0]
                chosen_rows.append(chosen_row)
                chosen_sample_ids.add(int(chosen_row["sample_id"]))
                chosen_categories.add(int(chosen_row["disease_category"]))
                remaining = remaining.iloc[1:].copy()

        selected_quadrant = pd.DataFrame(chosen_rows).drop(columns=["is_new_category"], errors="ignore")
        selected_frames.append(selected_quadrant)

    if not selected_frames:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "y_true",
                "y_pred",
                "p_raw",
                "p_cal",
                "quadrant",
                "distance_to_threshold",
                "disease_category",
                "selection_band",
            ]
        )
    selected = pd.concat(selected_frames, ignore_index=True)
    return selected.loc[
        :,
        [
            "sample_id",
            "y_true",
            "y_pred",
            "p_raw",
            "p_cal",
            "quadrant",
            "distance_to_threshold",
            "disease_category",
            "selection_band",
        ],
    ].sort_values(["quadrant", "distance_to_threshold", "sample_id"], kind="mergesort").reset_index(drop=True)


def _render_false_negative_section(fn_frame: pd.DataFrame) -> str:
    """Render the mandatory false-negative detail markdown section.

    Args:
        fn_frame: False-negative dataframe.

    Returns:
        Markdown string for the FN section.
    """
    lines = ["## False Negatives"]
    if fn_frame.empty:
        lines.extend(["", "No false negatives in the test set."])
        return "\n".join(lines)

    lines.append("")
    for _, row in fn_frame.sort_values("sample_id", kind="mergesort").iterrows():
        disease_label = DISEASE_CATEGORY_LABELS.get(int(row["disease_category"]), str(int(row["disease_category"])))
        lines.append(
            "- "
            f"sample_id={int(row['sample_id'])}, p_cal={float(row['p_cal']):.6f}, "
            f"disease_category={disease_label}, age={float(row['age']):.1f}, sex={int(row['sex'])}, "
            f"cea={row['cea']}, ca199={row['ca199']}, ca724={row['ca724']}"
        )
    return "\n".join(lines)


def _build_fp_lgin_table(prediction_frame: pd.DataFrame) -> np.ndarray:
    """Build the locked FP-LGIN Fisher exact 2x2 table.

    Args:
        prediction_frame: Best-model test prediction frame with `quadrant` and `disease_category`.

    Returns:
        A 2x2 numpy array with rows `[FP, non-FP]` and columns `[LGIN, non-LGIN]`.
    """
    fp_mask = prediction_frame["quadrant"] == "FP"
    lgin_mask = prediction_frame["disease_category"] == 2
    return np.array(
        [
            [int((fp_mask & lgin_mask).sum()), int((fp_mask & ~lgin_mask).sum())],
            [int((~fp_mask & lgin_mask).sum()), int((~fp_mask & ~lgin_mask).sum())],
        ],
        dtype=int,
    )


def _write_representative_signal_npz(
    *,
    representative_frame: pd.DataFrame,
    pipeline_a_path: Path,
    output_path: Path,
) -> None:
    """Persist representative raw signals for later SHAP Session 14 use.

    Args:
        representative_frame: Representative-sample dataframe.
        pipeline_a_path: Path to `pipeline_a_raw.npz`.
        output_path: Destination NPZ path.
    """
    try:
        with np.load(pipeline_a_path, allow_pickle=False) as payload:
            sample_ids = payload["sample_id"].astype(np.int64)
            signals = payload["x"]
    except OSError as exc:
        LOGGER.error("Failed to read Pipeline A raw tensor at %s.", pipeline_a_path)
        raise OSError(f"Failed to read Pipeline A raw tensor at {pipeline_a_path}.") from exc

    sample_id_to_index = {int(sample_id): idx for idx, sample_id in enumerate(sample_ids.tolist())}
    representative_ids = representative_frame["sample_id"].astype(int).tolist()
    missing_sample_ids = [sample_id for sample_id in representative_ids if sample_id not in sample_id_to_index]
    if missing_sample_ids:
        raise AssertionError(f"Representative samples missing from pipeline_a_raw.npz: {missing_sample_ids}")

    extracted = np.stack([signals[sample_id_to_index[sample_id]] for sample_id in representative_ids], axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        np.savez_compressed(
            output_path,
            sample_id=np.array(representative_ids, dtype=np.int64),
            quadrant=representative_frame["quadrant"].astype(object).to_numpy(),
            signal=extracted,
        )
    except OSError as exc:
        LOGGER.error("Failed to write representative signal NPZ to %s.", output_path)
        raise OSError(f"Failed to write representative signal NPZ to {output_path}.") from exc
    try:
        logged_path = _format_path(output_path)
    except ValueError:
        logged_path = str(output_path)
    LOGGER.info("Wrote representative signal NPZ to %s.", logged_path)


def _summarize_median_q1_q3(series: pd.Series) -> tuple[float | None, float | None, float | None]:
    """Summarize one numeric series with median and quartiles.

    Args:
        series: Numeric pandas series.

    Returns:
        Median, first quartile, and third quartile.
    """
    non_missing = series.dropna().astype(float)
    if non_missing.empty:
        return None, None, None
    return float(non_missing.median()), float(non_missing.quantile(0.25)), float(non_missing.quantile(0.75))


def _summarize_percent(series: pd.Series) -> float | None:
    """Return the percentage of truthy values in a binary series.

    Args:
        series: Binary pandas series.

    Returns:
        Percentage in [0, 100].
    """
    non_missing = series.dropna().astype(float)
    if non_missing.empty:
        return None
    return float(100.0 * non_missing.mean())


def run_misclassification_analysis(config_path: Path, best_model: str) -> None:
    """Run the best-model misclassification Layer 2 analysis.

    Args:
        config_path: Path to `configs/modeling_global.yaml`.
        best_model: Canonical best-model short name.
    """
    config = _load_global_config(config_path=config_path)
    model_payload = _load_model_payload(best_model)
    threshold_p_raw = float(model_payload["threshold_p_raw"])
    metadata = _load_final_cohort_extended_metadata(config=config)
    prediction_frame = _load_view_prediction_frame(model_name=best_model, view_name="test")
    merged = _merge_predictions_with_optional_metadata(
        prediction_frame=prediction_frame,
        metadata_frame=metadata,
        required_columns=["disease_category", "age", "sex", "bmi", "smoking", "drinking"],
        optional_columns=["hp_infection", *TUMOR_MARKER_COLUMNS],
    )
    LOGGER.info("Loaded misclassification inputs for model=%s with n=%s.", best_model, len(merged))

    per_sample_frame = merged.loc[
        :,
        [
            "sample_id",
            "y_true",
            "y_pred",
            "p_cal",
            "quadrant",
            "disease_category",
            "age",
            "sex",
            "bmi",
            "smoking",
            "drinking",
            "hp_infection",
            "cea",
            "ca199",
            "ca724",
        ],
    ].sort_values("sample_id", kind="mergesort")
    per_sample_path = _resolve_path(config["paths"]["outputs_analysis_dir"]) / f"misclassification_per_sample_{best_model}.csv"
    _write_dataframe_csv(per_sample_path, per_sample_frame)

    representative_frame = _select_representative_samples_by_band(
        prediction_frame=merged,
        threshold_p_raw=threshold_p_raw,
        max_per_quadrant=5,
    )
    representative_path = (
        _resolve_path(config["paths"]["outputs_analysis_dir"]) / f"misclassification_representative_samples_{best_model}.csv"
    )
    _write_dataframe_csv(representative_path, representative_frame)

    signal_npz_path = _resolve_path("outputs/shap") / f"binary_{best_model}_representative_samples_signals.npz"
    pipeline_a_path = _resolve_path("data/processed/sequences/pipeline_a_raw.npz")
    _write_representative_signal_npz(
        representative_frame=representative_frame,
        pipeline_a_path=pipeline_a_path,
        output_path=signal_npz_path,
    )

    fp_lgin_table = _build_fp_lgin_table(merged)
    fp_lgin_result = fisher_exact_2x2(fp_lgin_table)
    false_negative_frame = merged.loc[merged["quadrant"] == "FN"].reset_index(drop=True)

    markdown_lines = [
        "# Misclassification Four-Quadrant Analysis",
        "",
        f"- Best model: `{best_model}`",
        f"- Per-sample CSV: `{_format_path(per_sample_path)}`",
        f"- Representative samples CSV: `{_format_path(representative_path)}`",
        f"- Representative raw-signal NPZ: `{_format_path(signal_npz_path)}`",
        "- SHAP values for these representative samples will be generated in Session 14 (Cross-Model SHAP Consistency).",
        "",
        "## Quadrant Counts",
    ]
    for quadrant in QUADRANT_ORDER:
        quadrant_frame = merged.loc[merged["quadrant"] == quadrant].reset_index(drop=True)
        markdown_lines.append(f"- `{quadrant}`: n={len(quadrant_frame)}")

    for quadrant in QUADRANT_ORDER:
        quadrant_frame = merged.loc[merged["quadrant"] == quadrant].reset_index(drop=True)
        median_p_cal, q1_p_cal, q3_p_cal = _summarize_median_q1_q3(quadrant_frame["p_cal"])
        age_median, age_q1, age_q3 = _summarize_median_q1_q3(quadrant_frame["age"])
        bmi_median, bmi_q1, bmi_q3 = _summarize_median_q1_q3(quadrant_frame["bmi"])
        cea_median, _, _ = _summarize_median_q1_q3(quadrant_frame["cea"])
        ca199_median, _, _ = _summarize_median_q1_q3(quadrant_frame["ca199"])
        ca724_median, _, _ = _summarize_median_q1_q3(quadrant_frame["ca724"])
        markdown_lines.extend(["", f"## {quadrant}", ""])
        markdown_lines.append(f"- p_cal median [Q1, Q3]: {median_p_cal} [{q1_p_cal}, {q3_p_cal}]")
        markdown_lines.append(f"- age median [Q1, Q3]: {age_median} [{age_q1}, {age_q3}]")
        markdown_lines.append(f"- BMI median [Q1, Q3]: {bmi_median} [{bmi_q1}, {bmi_q3}]")
        markdown_lines.append(f"- sex=1 (%): {_summarize_percent(quadrant_frame['sex'])}")
        markdown_lines.append(f"- smoking=1 (%): {_summarize_percent(quadrant_frame['smoking'])}")
        markdown_lines.append(f"- drinking=1 (%): {_summarize_percent(quadrant_frame['drinking'])}")
        markdown_lines.append(f"- Hp infection=1 (%): {_summarize_percent(quadrant_frame['hp_infection'])}")
        markdown_lines.append(f"- CEA median: {cea_median}; missing={int(quadrant_frame['cea'].isna().sum())}")
        markdown_lines.append(f"- CA19-9 median: {ca199_median}; missing={int(quadrant_frame['ca199'].isna().sum())}")
        markdown_lines.append(f"- CA72-4 median: {ca724_median}; missing={int(quadrant_frame['ca724'].isna().sum())}")
        markdown_lines.append("- disease subtype distribution:")
        for disease_category, label in DISEASE_CATEGORY_LABELS.items():
            count = int((quadrant_frame["disease_category"] == disease_category).sum())
            proportion = float(100.0 * count / len(quadrant_frame)) if len(quadrant_frame) else 0.0
            markdown_lines.append(f"  - {label}: {count} ({proportion:.1f}%)")

    markdown_lines.extend(
        [
            "",
            _render_false_negative_section(false_negative_frame),
            "",
            "## FP-LGIN Fisher Exact Test",
            f"- table = {fp_lgin_table.tolist()}",
            f"- odds_ratio = {fp_lgin_result['odds_ratio']:.6f}",
            f"- p_two_sided = {fp_lgin_result['p_two_sided']:.6g}",
            (
                "- Interpretation guidance: a significant p-value with odds_ratio > 2 suggests LGIN is over-represented "
                "among false positives, which would be descriptive evidence that the model confuses some LGIN samples "
                "with higher-risk lesions. This does not trigger Post-hoc A, which is defined only by subgroup AUC degradation."
            ),
        ]
    )
    markdown_path = (
        _resolve_path(config["paths"]["outputs_analysis_dir"]) / f"misclassification_four_quadrant_{best_model}.md"
    )
    _write_text(markdown_path, "\n".join(markdown_lines) + "\n")
    LOGGER.info("Wrote misclassification markdown to %s.", _format_path(markdown_path))


__all__ = [
    "_build_fp_lgin_table",
    "_render_false_negative_section",
    "_select_representative_samples_by_band",
    "_write_representative_signal_npz",
    "run_misclassification_analysis",
]
