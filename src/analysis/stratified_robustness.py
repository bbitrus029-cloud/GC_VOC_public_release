from __future__ import annotations

import logging
from pathlib import Path

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
    _load_test_frame_with_metadata,
    _load_view_prediction_frame,
    _merge_predictions_with_metadata,
    _resolve_path,
)


LOGGER = logging.getLogger(__name__)
STRATIFICATION_VARIABLES = (
    "sex",
    "age_at_60",
    "bmi_at_24",
    "smoking",
    "drinking",
    "family_history",
    "hp_infection",
)


def _encode_hp_infection_strata(series: pd.Series) -> pd.Series:
    """Encode Hp infection as 0, 1, or missing."""
    return series.apply(lambda value: "missing" if pd.isna(value) else str(int(value)))


def _gate_stratum(*, n_total: int, n_pos: int, n_neg: int) -> str:
    """Return an omission reason string when the prespecified gate fails."""
    reasons: list[str] = []
    if n_total < 10:
        reasons.append(f"n_total={n_total} < 10")
    if n_pos < 2:
        reasons.append(f"n_pos={n_pos} < 2")
    if n_neg < 2:
        reasons.append(f"n_neg={n_neg} < 2")
    return "; ".join(reasons)


def _build_stratified_output_frame(metric_rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build one persisted stratified-robustness output frame."""
    return _build_metric_output_frame(
        metric_rows,
        [
            "model",
            "stratum_value",
            "n_total",
            "n_pos",
            "n_neg",
            "metric_name",
            "estimate",
            "ci_low",
            "ci_high",
            "omission_reason",
        ],
    )


def _derive_stratification_columns(test_metadata: pd.DataFrame, *, age_cutpoint: int, bmi_cutpoint: int) -> pd.DataFrame:
    """Add the seven Session 11 stratification columns to the test metadata."""
    derived = test_metadata.copy()
    derived["age_at_60"] = derived["age"].apply(lambda value: f"<{age_cutpoint}" if value < age_cutpoint else f">={age_cutpoint}")
    derived["bmi_at_24"] = derived["bmi"].apply(lambda value: f"<{bmi_cutpoint}" if value < bmi_cutpoint else f">={bmi_cutpoint}")
    derived["hp_infection"] = _encode_hp_infection_strata(derived["hp_infection"])
    return derived


def run_stratified_robustness(config_path: Path) -> None:
    """Run the all-models stratified robustness analysis for the seven locked variables.

    Args:
        config_path: Path to `configs/modeling_global.yaml`.
    """
    config = _load_global_config(config_path=config_path)
    test_metadata = _load_test_frame_with_metadata(config=config)
    test_metadata = _derive_stratification_columns(
        test_metadata,
        age_cutpoint=int(config["age_cutpoint"]),
        bmi_cutpoint=int(config["bmi_cutpoint"]),
    )
    n_boot = int(config["bootstrap"]["n_boot"])

    for variable_name in STRATIFICATION_VARIABLES:
        metric_rows: list[dict[str, object]] = []
        non_omitted_row_count = 0
        for model_name in MODEL_NAMES:
            model_payload = _load_model_payload(model_name)
            threshold_p_raw = float(model_payload["threshold_p_raw"])
            prediction_frame = _load_view_prediction_frame(model_name=model_name, view_name="test")
            merged = _merge_predictions_with_metadata(
                prediction_frame=prediction_frame,
                metadata_frame=test_metadata,
                required_columns=[variable_name],
            )
            for stratum_value in sorted(merged[variable_name].astype(str).unique()):
                stratum_frame = merged.loc[merged[variable_name].astype(str) == stratum_value].reset_index(drop=True)
                n_total = len(stratum_frame)
                n_pos = int((stratum_frame["y_true"] == 1).sum())
                n_neg = int((stratum_frame["y_true"] == 0).sum())
                omission_reason = _gate_stratum(n_total=n_total, n_pos=n_pos, n_neg=n_neg)
                if omission_reason:
                    metric_rows.append(
                        {
                            "model": model_name,
                            "stratum_value": stratum_value,
                            "n_total": n_total,
                            "n_pos": n_pos,
                            "n_neg": n_neg,
                            "metric_name": "",
                            "estimate": float("nan"),
                            "ci_low": float("nan"),
                            "ci_high": float("nan"),
                            "omission_reason": omission_reason,
                        }
                    )
                    continue

                non_omitted_row_count += 1
                for metric_name in THRESHOLD_FREE_METRIC_NAMES:
                    metric_rows.append(
                        {
                            "model": model_name,
                            "stratum_value": stratum_value,
                            "n_total": n_total,
                            "n_pos": n_pos,
                            "n_neg": n_neg,
                            "metric_name": metric_name,
                            "omission_reason": "",
                            **_bootstrap_metric_from_frame(
                                prediction_frame=stratum_frame,
                                metric_name=metric_name,
                                threshold_p_raw=threshold_p_raw,
                                n_boot=n_boot,
                                seed=RANDOM_SEED,
                            ),
                        }
                    )
        if non_omitted_row_count == 0:
            raise AssertionError(f"All strata were omitted for variable {variable_name}, indicating metadata corruption.")

        output_frame = _build_stratified_output_frame(metric_rows)
        output_path = (
            _resolve_path(config["paths"]["outputs_analysis_dir"]) / f"stratified_robustness_{variable_name}_all_models.csv"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_frame.to_csv(output_path, index=False)
        LOGGER.info("Wrote stratified robustness output for %s to %s.", variable_name, _format_path(output_path))
