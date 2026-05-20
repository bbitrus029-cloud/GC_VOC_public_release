from __future__ import annotations

import importlib

import pytest


def _import_module_or_fail():
    """Import the progression-trajectory module or fail with a readable message."""
    try:
        return importlib.import_module("src.analysis.progression_trajectory")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Missing module src.analysis.progression_trajectory: {exc}", pytrace=False)


def test_primary_jt_pvalue_key_is_one_sided_upper() -> None:
    """The locked primary JT p-value key should be the upper-tail one-sided field."""
    module = _import_module_or_fail()

    assert module.JT_PRIMARY_PVALUE_KEY == "p_one_sided_upper"


def test_adjacent_pairs_follow_locked_progression_order() -> None:
    """Adjacent-pair testing should cover exactly the five prespecified category transitions."""
    module = _import_module_or_fail()

    assert module.ADJACENT_PAIR_DEFINITIONS == [
        ("HC-BGL", 0, 1),
        ("BGL-LGIN", 1, 2),
        ("LGIN-HGIN", 2, 3),
        ("HGIN-EGC", 3, 4),
        ("EGC-AGC", 4, 5),
    ]


def test_holm_adjustment_preserves_adjacent_pair_input_order() -> None:
    """The persisted adjacent-pair rows should remain in the original clinical ordering."""
    module = _import_module_or_fail()

    adjusted_rows = module._apply_holm_to_adjacent_pairs(
        [
            {"pair_name": "HC-BGL", "raw_p": 0.04},
            {"pair_name": "BGL-LGIN", "raw_p": 0.01},
            {"pair_name": "LGIN-HGIN", "raw_p": 0.20},
            {"pair_name": "HGIN-EGC", "raw_p": 0.03},
            {"pair_name": "EGC-AGC", "raw_p": 0.50},
        ]
    )

    assert [row["pair_name"] for row in adjusted_rows] == [
        "HC-BGL",
        "BGL-LGIN",
        "LGIN-HGIN",
        "HGIN-EGC",
        "EGC-AGC",
    ]
    assert all("holm_adjusted_p" in row for row in adjusted_rows)
    assert all("reject_null" in row for row in adjusted_rows)


def test_progression_csv_output_schema_is_locked() -> None:
    """The per-sample progression CSV should retain the prespecified four columns."""
    module = _import_module_or_fail()

    output_frame = module._build_progression_output_frame(
        [
            {
                "sample_id": 1,
                "y_true": 1,
                "p_cal": 0.8,
                "disease_category": 4,
            }
        ]
    )

    assert output_frame.columns.tolist() == ["sample_id", "y_true", "p_cal", "disease_category"]
