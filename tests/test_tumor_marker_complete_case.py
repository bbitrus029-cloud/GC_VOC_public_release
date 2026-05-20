from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest


def _import_module_or_fail():
    """Import the Session 13 common helper module or fail clearly."""
    try:
        return importlib.import_module("src.analysis.stage7_part_c_common")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Missing module src.analysis.stage7_part_c_common: {exc}", pytrace=False)


def test_ca199_european_decimal_coercion_handles_commas_and_invalid_tokens() -> None:
    """European decimal strings should parse while invalid tokens become NaN."""
    module = _import_module_or_fail()

    coerced = module._coerce_ca199_series(pd.Series(["14,07", "abc", np.nan, "8.5"]))

    assert coerced.iloc[0] == pytest.approx(14.07)
    assert np.isnan(coerced.iloc[1])
    assert np.isnan(coerced.iloc[2])
    assert coerced.iloc[3] == pytest.approx(8.5)


def test_complete_case_mask_requires_all_three_markers_present() -> None:
    """Complete-case filtering must require non-missing CEA, CA19-9, and CA72-4 together."""
    module = _import_module_or_fail()
    frame = pd.DataFrame(
        {
            "cea": [1.0, np.nan, 3.0, 4.0],
            "ca199": [2.0, 2.0, np.nan, 5.0],
            "ca724": [3.0, 3.0, 3.0, np.nan],
        }
    )

    mask = module._compute_complete_case_mask(frame)

    assert mask.tolist() == [True, False, False, False]


def test_partition_by_train_test_indices_splits_without_overlap() -> None:
    """Locked train/test indices should partition a frame deterministically."""
    module = _import_module_or_fail()
    frame = pd.DataFrame(
        {
            "sample_id": [10, 11, 12, 13, 14],
            "value": [1, 2, 3, 4, 5],
        }
    )

    train_frame, test_frame = module._partition_frame_by_indices(
        frame=frame,
        train_idx=np.asarray([0, 1, 2]),
        test_idx=np.asarray([3, 4]),
    )

    assert train_frame["sample_id"].tolist() == [10, 11, 12]
    assert test_frame["sample_id"].tolist() == [13, 14]
    assert set(train_frame["sample_id"]).isdisjoint(set(test_frame["sample_id"]))
