from __future__ import annotations

from typing import Any

import numpy as np
import pytest


def _import_symbols() -> Any:
    """Import top-k helpers lazily for clear failures."""
    try:
        from src.analysis.shap_cross_model import _compute_top_k_converging_rows
    except (AttributeError, ImportError, ModuleNotFoundError) as exc:
        pytest.fail(f"shap_cross_model implementation is missing: {exc}", pytrace=False)

    return _compute_top_k_converging_rows


def test_top_k_converging_rows_capture_majority_thresholds() -> None:
    """The helper should count channel memberships across K and M thresholds."""
    compute_rows = _import_symbols()

    importance_map = {
        "m1": np.asarray([0.40, 0.30, 0.20, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]),
        "m2": np.asarray([0.39, 0.31, 0.21, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02]),
        "m3": np.asarray([0.38, 0.32, 0.22, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]),
        "m4": np.asarray([0.37, 0.33, 0.23, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.00]),
        "m5": np.asarray([0.36, 0.34, 0.05, 0.04, 0.03, 0.30, 0.20, 0.10, 0.02, 0.01, 0.00]),
        "m6": np.asarray([0.35, 0.30, 0.04, 0.03, 0.02, 0.29, 0.21, 0.11, 0.01, 0.00, 0.00]),
    }

    rows = compute_rows(importance_map)
    row_lookup = {(int(row["K"]), int(row["M_models_threshold"])): row for row in rows}

    row_k3_m4 = row_lookup[(3, 4)]
    assert row_k3_m4["channels"] == ["S1", "S2", "S3"]
    assert row_k3_m4["channel_count"] == 3

    row_k5_m6 = row_lookup[(5, 6)]
    assert row_k5_m6["channels"] == ["S1", "S2"]
    assert row_k5_m6["channel_count"] == 2
