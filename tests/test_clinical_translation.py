from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _import_module_or_fail():
    """Import the clinical translation module or fail with a readable message."""
    try:
        return importlib.import_module("src.analysis.clinical_translation")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Missing module src.analysis.clinical_translation: {exc}", pytrace=False)


def test_prevalence_adjusted_ppv_npv_and_nne_follow_formula() -> None:
    """Clinical translation should compute the prespecified prevalence-adjusted quantities."""
    module = _import_module_or_fail()

    result = module._compute_prevalence_adjusted_metrics(
        sensitivity=0.8,
        specificity=0.9,
        prevalence=0.1,
    )

    expected_ppv = (0.8 * 0.1) / ((0.8 * 0.1) + (0.1 * 0.9))
    expected_npv = (0.9 * 0.9) / ((0.9 * 0.9) + (0.2 * 0.1))

    assert result["ppv"] == pytest.approx(expected_ppv)
    assert result["npv"] == pytest.approx(expected_npv)
    assert result["nne"] == pytest.approx(1.0 / expected_ppv)


def test_assign_binary_risk_tier_rejects_three_tier_configuration() -> None:
    """Only the binary plan-locked risk-tier scheme should be accepted."""
    module = _import_module_or_fail()

    with pytest.raises(ValueError, match="binary tier"):
        module._assign_binary_risk_tier(
            probabilities=np.array([0.1, 0.8], dtype=float),
            threshold_p_raw=0.5,
            tier_mode="three_tier",
        )


def test_assign_binary_risk_tier_uses_p_raw_not_p_cal() -> None:
    """High-risk assignment must be driven by p_raw rather than calibrated probabilities."""
    module = _import_module_or_fail()
    prediction_frame = pd.DataFrame(
        {
            "sample_id": [1, 2],
            "y_true": [0, 1],
            "p_raw": [0.30, 0.60],
            "p_cal": [0.95, 0.05],
        }
    )

    assigned = module._assign_binary_risk_tier_from_frame(
        prediction_frame=prediction_frame,
        threshold_p_raw=0.5,
        tier_mode="binary",
    )

    assert assigned.tolist() == [0, 1]


def test_build_clinical_translation_output_frame_has_expected_schema() -> None:
    """The persisted long-format output should use the required columns."""
    module = _import_module_or_fail()

    output_frame = module._build_clinical_translation_output_frame(
        metric_rows=[
            {
                "prevalence_scenario": 0.01,
                "metric_name": "ppv",
                "estimate": 0.2,
                "ci_low": 0.1,
                "ci_high": 0.3,
            }
        ]
    )

    assert output_frame.columns.tolist() == [
        "prevalence_scenario",
        "metric_name",
        "estimate",
        "ci_low",
        "ci_high",
    ]


def test_load_prediction_frame_reads_standard_csv(tmp_path: Path) -> None:
    """Prediction loading should preserve the expected Stage 4b columns."""
    module = _import_module_or_fail()
    csv_path = tmp_path / "test_predictions.csv"
    csv_path.write_text(
        "sample_id,y_true,p_raw,p_cal,y_pred,quadrant\n"
        "1,0,0.20,0.30,0,TN\n"
        "2,1,0.80,0.70,1,TP\n",
        encoding="utf-8",
    )

    loaded = module._load_prediction_frame(csv_path)

    assert loaded.columns.tolist() == ["sample_id", "y_true", "p_raw", "p_cal", "y_pred", "quadrant"]
