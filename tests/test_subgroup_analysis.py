from __future__ import annotations

import importlib
from unittest.mock import Mock

import pandas as pd
import pytest


def _import_module_or_fail():
    """Import the subgroup-analysis module or fail with a readable message."""
    try:
        return importlib.import_module("src.analysis.subgroup_analysis")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Missing module src.analysis.subgroup_analysis: {exc}", pytrace=False)


def _build_prediction_frame() -> pd.DataFrame:
    """Create a compact synthetic subgroup-analysis frame."""
    return pd.DataFrame(
        {
            "sample_id": [1, 2, 3, 4, 5, 6],
            "y_true": [0, 0, 0, 1, 1, 1],
            "p_raw": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
            "p_cal": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
            "disease_category": [0, 1, 2, 3, 4, 5],
        }
    )


def test_shared_negative_pool_is_hc_bgl_lgin_only() -> None:
    """Both subgroups should share the same negative pool from categories 0, 1, and 2."""
    module = _import_module_or_fail()
    prediction_frame = _build_prediction_frame()

    negative_pool = module._build_shared_negative_pool(prediction_frame)

    assert negative_pool["sample_id"].tolist() == [1, 2, 3]
    assert negative_pool["disease_category"].tolist() == [0, 1, 2]


def test_hgin_egc_and_agc_subgroups_use_disjoint_positive_sets() -> None:
    """The two subgroup definitions should follow the locked category partitions."""
    module = _import_module_or_fail()
    prediction_frame = _build_prediction_frame()

    subgroup_frames = module._build_subgroup_frames(prediction_frame)

    assert subgroup_frames["hgin_egc"]["sample_id"].tolist() == [1, 2, 3, 4, 5]
    assert subgroup_frames["agc"]["sample_id"].tolist() == [1, 2, 3, 6]


def test_compute_best_model_subgroup_metrics_does_not_refit_model() -> None:
    """Subgroup evaluation should consume persisted predictions without calling fit()."""
    module = _import_module_or_fail()
    prediction_frame = _build_prediction_frame()
    subgroup_frame = prediction_frame.loc[prediction_frame["sample_id"].isin([1, 2, 3, 4, 5])].reset_index(drop=True)
    fake_model = Mock()

    metric_rows = module._compute_best_model_subgroup_metric_rows(
        grouping="hgin_egc",
        prediction_frame=subgroup_frame,
        threshold_p_raw=0.5,
        n_boot=5,
        seed=42,
        model_object=fake_model,
    )

    fake_model.fit.assert_not_called()
    assert metric_rows


def test_subgroup_output_excludes_nne() -> None:
    """NNE belongs only to Tier A clinical translation and must stay out of subgroup outputs."""
    module = _import_module_or_fail()

    assert "nne" not in module.BEST_MODEL_SUBGROUP_METRICS


def test_build_best_model_subgroup_output_frame_has_expected_schema() -> None:
    """The best-model subgroup companion CSV should use the required long schema."""
    module = _import_module_or_fail()

    output_frame = module._build_best_model_subgroup_output_frame(
        metric_rows=[
            {
                "grouping": "hgin_egc",
                "view": "test",
                "metric_name": "auc_roc",
                "estimate": 0.9,
                "ci_low": 0.8,
                "ci_high": 1.0,
            }
        ]
    )

    assert output_frame.columns.tolist() == [
        "grouping",
        "view",
        "metric_name",
        "estimate",
        "ci_low",
        "ci_high",
    ]
