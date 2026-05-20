from __future__ import annotations

import importlib
from unittest.mock import Mock

import pandas as pd
import pytest


def _import_module_or_fail():
    """Import the fallback/padded sensitivity module or fail with a readable message."""
    try:
        return importlib.import_module("src.analysis.fallback_padded_sensitivity")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Missing module src.analysis.fallback_padded_sensitivity: {exc}", pytrace=False)


def _build_fixture() -> pd.DataFrame:
    """Create a compact synthetic test-set frame with the two engineering flags."""
    return pd.DataFrame(
        {
            "sample_id": [1, 2, 3, 4, 5, 6],
            "y_true": [0, 1, 0, 1, 0, 1],
            "y_pred": [0, 1, 1, 0, 0, 1],
            "p_raw": [0.1, 0.9, 0.8, 0.2, 0.3, 0.7],
            "p_cal": [0.1, 0.9, 0.8, 0.2, 0.3, 0.7],
            "fallback_used": [False, True, False, True, False, False],
            "padded": [False, False, True, False, True, False],
        }
    )


def test_four_locked_subsets_are_constructed_correctly() -> None:
    """Sensitivity analysis should use the full, fallback-free, padded-free, and both-free subsets."""
    module = _import_module_or_fail()
    prediction_frame = _build_fixture()

    subset_frames = module._build_subset_frames(prediction_frame)

    assert sorted(subset_frames) == [
        "both_false",
        "fallback_used_false",
        "full",
        "padded_false",
    ]
    assert len(subset_frames["full"]) == 6
    assert subset_frames["fallback_used_false"]["sample_id"].tolist() == [1, 3, 5, 6]
    assert subset_frames["padded_false"]["sample_id"].tolist() == [1, 2, 4, 6]
    assert subset_frames["both_false"]["sample_id"].tolist() == [1, 6]


def test_binary_association_table_is_2x2() -> None:
    """Best-model Fisher inputs should be persisted as a 2x2 contingency table."""
    module = _import_module_or_fail()
    prediction_frame = _build_fixture()

    table = module._build_binary_association_table(
        indicator=prediction_frame["fallback_used"],
        outcome=(prediction_frame["y_pred"] != prediction_frame["y_true"]),
    )

    assert table.shape == (2, 2)
    assert table.sum() == len(prediction_frame)


def test_subset_metric_rows_do_not_refit_any_model() -> None:
    """Fallback/padded sensitivity must consume persisted predictions only."""
    module = _import_module_or_fail()
    prediction_frame = _build_fixture()
    fake_model = Mock()

    metric_rows = module._compute_subset_metric_rows(
        model_name="minirocket_ridge",
        subset_name="full",
        prediction_frame=prediction_frame,
        threshold_p_raw=0.5,
        n_boot=5,
        seed=42,
        model_object=fake_model,
    )

    fake_model.fit.assert_not_called()
    assert len(metric_rows) == 3
    assert {row["metric_name"] for row in metric_rows} == {"auc_roc", "auc_pr", "brier"}


def test_fallback_output_schema_is_locked() -> None:
    """The all-models sensitivity CSV should keep the required long-format schema."""
    module = _import_module_or_fail()

    output_frame = module._build_fallback_output_frame(
        [
            {
                "model": "minirocket_ridge",
                "subset": "full",
                "n_total": 195,
                "n_pos": 82,
                "n_neg": 113,
                "metric_name": "auc_roc",
                "estimate": 0.9,
                "ci_low": 0.8,
                "ci_high": 1.0,
            }
        ]
    )

    assert output_frame.columns.tolist() == [
        "model",
        "subset",
        "n_total",
        "n_pos",
        "n_neg",
        "metric_name",
        "estimate",
        "ci_low",
        "ci_high",
    ]
