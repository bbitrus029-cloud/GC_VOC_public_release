from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _import_module_or_fail():
    """Import the three-view module or fail with a readable message."""
    try:
        return importlib.import_module("src.analysis.three_view_comparison")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Missing module src.analysis.three_view_comparison: {exc}", pytrace=False)


def test_model_threshold_mapping_preserves_per_model_thresholds() -> None:
    """Each model should retain its own threshold rather than sharing a global one."""
    module = _import_module_or_fail()

    threshold_map = module._extract_model_thresholds(
        {
            "minirocket_ridge": {"threshold_p_raw": 0.34},
            "xgboost": {"threshold_p_raw": 0.52},
        }
    )

    assert threshold_map == {"minirocket_ridge": 0.34, "xgboost": 0.52}


def test_supported_views_exclude_the_622_subset() -> None:
    """Tier B1 should report only the three plan-locked views."""
    module = _import_module_or_fail()

    assert module.SUPPORTED_VIEWS == ("train_in_sample", "nested_cv_oof", "test")
    assert "outer_train_subset_622" not in module.SUPPORTED_VIEWS


def test_load_prediction_frame_skips_leading_warning_comment(tmp_path: Path) -> None:
    """Train-in-sample CSV loading should ignore the leading warning line."""
    module = _import_module_or_fail()
    csv_path = tmp_path / "train_in_sample.csv"
    csv_path.write_text(
        "# WARNING: In-sample predictions are an overfitting upper bound and are not used for any claim.\n"
        "sample_id,y_true,p_raw,p_cal,y_pred,quadrant\n"
        "2,1,0.81,0.93,1,TP\n"
        "3,0,0.25,0.12,0,TN\n",
        encoding="utf-8",
    )

    loaded = module._load_prediction_frame(csv_path)

    assert loaded.columns.tolist() == ["sample_id", "y_true", "p_raw", "p_cal", "y_pred", "quadrant"]
    assert loaded["sample_id"].tolist() == [2, 3]


def test_build_three_view_output_frame_has_expected_schema() -> None:
    """The three-view output should persist in long format."""
    module = _import_module_or_fail()

    output_frame = module._build_three_view_output_frame(
        metric_rows=[
            {
                "model": "minirocket_ridge",
                "view": "test",
                "metric_name": "auc_roc",
                "estimate": 0.9,
                "ci_low": 0.8,
                "ci_high": 1.0,
            }
        ]
    )

    assert isinstance(output_frame, pd.DataFrame)
    assert output_frame.columns.tolist() == [
        "model",
        "view",
        "metric_name",
        "estimate",
        "ci_low",
        "ci_high",
    ]


def test_threshold_dependent_metrics_use_passed_model_specific_threshold() -> None:
    """The metric computation should respect the threshold supplied for that model-view pair."""
    module = _import_module_or_fail()
    prediction_frame = pd.DataFrame(
        {
            "sample_id": [1, 2, 3, 4],
            "y_true": [1, 1, 0, 0],
            "p_raw": [0.6, 0.4, 0.3, 0.2],
            "p_cal": [0.9, 0.8, 0.1, 0.2],
        }
    )

    higher_threshold_metrics = module._compute_view_metrics(
        prediction_frame=prediction_frame,
        threshold_p_raw=0.5,
        n_boot=5,
        seed=42,
    )
    lower_threshold_metrics = module._compute_view_metrics(
        prediction_frame=prediction_frame,
        threshold_p_raw=0.35,
        n_boot=5,
        seed=42,
    )

    assert higher_threshold_metrics["sensitivity"]["estimate"] != lower_threshold_metrics["sensitivity"]["estimate"]
