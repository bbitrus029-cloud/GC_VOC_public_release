from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _import_module_or_fail():
    """Import the misclassification-analysis module or fail with a readable message."""
    try:
        return importlib.import_module("src.analysis.misclassification")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Missing module src.analysis.misclassification: {exc}", pytrace=False)


def _build_representative_fixture() -> pd.DataFrame:
    """Create a compact fixture spanning all four quadrants."""
    rows: list[dict[str, object]] = []
    sample_id = 1
    for quadrant, p_raw_values, y_true, y_pred in [
        ("TP", [0.51, 0.56, 0.61, 0.72, 0.88, 0.95], 1, 1),
        ("TN", [0.49, 0.44, 0.39, 0.28, 0.12, 0.05], 0, 0),
        ("FP", [0.52, 0.57, 0.63, 0.74, 0.86, 0.93], 0, 1),
        ("FN", [0.48, 0.43, 0.37, 0.26, 0.14, 0.07], 1, 0),
    ]:
        for offset, p_raw in enumerate(p_raw_values):
            rows.append(
                {
                    "sample_id": sample_id,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "p_raw": p_raw,
                    "p_cal": min(max(p_raw + 0.03, 0.0), 1.0),
                    "quadrant": quadrant,
                    "disease_category": [0, 1, 2, 3, 4, 5][offset],
                }
            )
            sample_id += 1
    return pd.DataFrame(rows)


def test_representative_selection_caps_each_quadrant_at_five_samples() -> None:
    """Representative selection should never persist more than five samples per quadrant."""
    module = _import_module_or_fail()
    prediction_frame = _build_representative_fixture()

    selected = module._select_representative_samples_by_band(
        prediction_frame=prediction_frame,
        threshold_p_raw=0.5,
        max_per_quadrant=5,
    )

    assert selected.groupby("quadrant").size().max() <= 5
    assert set(selected["selection_band"].unique()).issubset({"low", "mid", "high"})


def test_false_negative_section_is_present_even_when_empty() -> None:
    """The markdown must keep the FN section even when the test set has zero false negatives."""
    module = _import_module_or_fail()

    fn_section = module._render_false_negative_section(
        pd.DataFrame(columns=["sample_id", "p_cal", "disease_category", "age", "sex", "cea", "ca199", "ca724"])
    )

    assert "## False Negatives" in fn_section
    assert "No false negatives in the test set" in fn_section


def test_fp_lgin_table_uses_fp_vs_non_fp_and_lgin_vs_non_lgin() -> None:
    """FP-LGIN Fisher input should follow the locked 2x2 cell definition."""
    module = _import_module_or_fail()
    prediction_frame = pd.DataFrame(
        {
            "sample_id": [1, 2, 3, 4, 5, 6],
            "quadrant": ["FP", "FP", "TN", "TP", "FN", "TN"],
            "disease_category": [2, 1, 2, 5, 4, 0],
        }
    )

    table = module._build_fp_lgin_table(prediction_frame)

    assert table.shape == (2, 2)
    assert table.tolist() == [[1, 1], [1, 3]]


def test_representative_signal_npz_has_locked_keys_and_shapes(tmp_path: Path) -> None:
    """Representative raw-signal persistence should preserve sample order and quadrant labels."""
    module = _import_module_or_fail()
    pipeline_a_path = tmp_path / "pipeline_a_raw.npz"
    output_path = tmp_path / "representative_signals.npz"
    np.savez_compressed(
        pipeline_a_path,
        sample_id=np.array([11, 12, 13], dtype=np.int64),
        x=np.zeros((3, 960, 11), dtype=np.float32),
        label=np.array([0, 1, 0], dtype=np.int64),
    )
    representative_frame = pd.DataFrame(
        {
            "sample_id": [12, 11],
            "quadrant": ["FP", "TP"],
        }
    )

    module._write_representative_signal_npz(
        representative_frame=representative_frame,
        pipeline_a_path=pipeline_a_path,
        output_path=output_path,
    )

    with np.load(output_path, allow_pickle=True) as payload:
        assert sorted(payload.files) == ["quadrant", "sample_id", "signal"]
        assert payload["sample_id"].tolist() == [12, 11]
        assert payload["quadrant"].tolist() == ["FP", "TP"]
        assert payload["signal"].shape == (2, 960, 11)
