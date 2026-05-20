from __future__ import annotations

import importlib

import pandas as pd
import pytest


def _import_module_or_fail():
    """Import the temporal-drift module or fail with a readable message."""
    try:
        return importlib.import_module("src.analysis.temporal_drift")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Missing module src.analysis.temporal_drift: {exc}", pytrace=False)


def test_collection_time_split_places_median_ties_in_first_half() -> None:
    """Samples tied at the median collection timestamp should deterministically stay in the first half."""
    module = _import_module_or_fail()
    prediction_frame = pd.DataFrame(
        {
            "sample_id": [1, 2, 3, 4, 5, 6],
            "collection_time": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                    "2024-01-03",
                    "2024-01-04",
                ]
            ),
        }
    )

    split_frames = module._split_by_collection_time_median(prediction_frame)

    assert split_frames["first_half"]["sample_id"].tolist() == [1, 2, 3, 4, 5]
    assert split_frames["second_half"]["sample_id"].tolist() == [6]


def test_temporal_output_schema_is_shared_by_oof_and_test_views() -> None:
    """The OOF and test temporal-drift CSVs should use the same locked schema."""
    module = _import_module_or_fail()

    output_frame = module._build_temporal_output_frame(
        [
            {
                "model": "minirocket_ridge",
                "half": "first_half",
                "time_range_start": "2024-01-01T00:00:00",
                "time_range_end": "2024-03-01T00:00:00",
                "n_total": 100,
                "n_pos": 40,
                "n_neg": 60,
                "metric_name": "auc_roc",
                "estimate": 0.9,
                "ci_low": 0.8,
                "ci_high": 1.0,
            }
        ]
    )

    assert output_frame.columns.tolist() == [
        "model",
        "half",
        "time_range_start",
        "time_range_end",
        "n_total",
        "n_pos",
        "n_neg",
        "metric_name",
        "estimate",
        "ci_low",
        "ci_high",
    ]
