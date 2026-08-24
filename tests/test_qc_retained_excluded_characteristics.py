"""Synthetic tests for the QC-retained/excluded characteristics package."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import OrderedDict
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from src.config import PROJECT_ROOT


MODULE_PATH = PROJECT_ROOT / "src/analysis/qc_retained_excluded_characteristics.py"
CONFIG_PATH = PROJECT_ROOT / "configs/qc_retained_excluded_characteristics.yaml"
SUMMARY_COLUMNS = [
    "variable",
    "level",
    "group",
    "n_total",
    "n_nonmissing",
    "n_missing",
    "count",
    "denominator_n",
    "proportion",
    "mean",
    "sd",
    "median",
    "q1",
    "q3",
    "denominator_rule",
]
SMD_COLUMNS = [
    "variable",
    "level",
    "comparison_type",
    "smd_qc_retained_minus_excluded",
    "n_qc_retained_used",
    "n_qc_excluded_used",
    "estimability_status",
    "nonestimable_reason",
]
ROW_ORDER = [
    ("age", None),
    ("sex", "female"),
    ("endpoint", "yes"),
    ("disease_category", "HC"),
    ("disease_category", "BGL"),
    ("disease_category", "LGIN"),
    ("disease_category", "HGIN"),
    ("disease_category", "EGC"),
    ("disease_category", "AGC"),
    ("body_mass_index", None),
    ("smoking_history", "yes"),
    ("alcohol_use_history", "yes"),
    ("first_degree_family_history_of_gastric_cancer", "yes"),
    ("helicobacter_pylori_status", "positive"),
    ("helicobacter_pylori_status", "negative"),
    ("helicobacter_pylori_status", "missing"),
]
MISSINGNESS_ORDER = [
    ("age", "missing"),
    ("sex", "missing"),
    ("endpoint", "missing"),
    ("disease_category", "missing"),
    ("body_mass_index", "missing"),
    ("smoking_history", "missing"),
    ("alcohol_use_history", "missing"),
    ("first_degree_family_history_of_gastric_cancer", "missing"),
]


def _module() -> ModuleType:
    """Load the production module after first proving the target exists."""
    assert MODULE_PATH.exists(), "production analysis module is not implemented"
    spec = importlib.util.spec_from_file_location("qc_characteristics_under_test", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Synthetic test could not load the production module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config() -> dict[str, Any]:
    """Return a minimal synthetic contract with deliberately small count anchors."""
    return {
        "schema_version": "qc_retained_excluded_characteristics.config.v1",
        "package_id": "qc_retained_excluded_characteristics",
        "expected_git_head": "a" * 40,
        "population_counts": {"complete_source": 6, "qc_retained": 4, "qc_excluded": 2},
        "disease_category_codes": {"HC": 0, "BGL": 1, "LGIN": 2, "HGIN": 3, "EGC": 4, "AGC": 5},
        "input_columns": {
            "raw_metadata": [
                "sample_id",
                "age",
                "sex(0male1female)",
                "BMI",
                "smoking(0no1yes)",
                "drinking(0no1yes)",
                "first_degree_family_history_of_GC(0no1yes)",
                "Hp_Infection(0no1yes)",
            ],
            "sample_index": ["sample_id", "label", "disease_category", "in_final_output"],
        },
        "inputs": {
            "raw_metadata": {"path": "inputs/metadata.csv", "sha256": "0" * 64},
            "sample_index": {"path": "inputs/sample_index.csv", "sha256": "0" * 64},
        },
        "anchors": {},
        "code": {"path": "src/analysis/qc_retained_excluded_characteristics.py"},
        "configuration": {"path": "configs/qc_retained_excluded_characteristics.yaml"},
        "outputs": {
            "summary": {"path": "out/summary.csv"},
            "smd": {"path": "out/smd.csv"},
            "manifest": {"path": "out/manifest.json"},
        },
    }


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build deidentified rows covering missingness and every planned category."""
    metadata = pd.DataFrame(
        {
            "sample_id": [
                "SYNTHETIC-101",
                "SYNTHETIC-102",
                "SYNTHETIC-103",
                "SYNTHETIC-104",
                "SYNTHETIC-105",
                "SYNTHETIC-106",
            ],
            "age": [10.0, 20.0, 30.0, np.nan, 40.0, 50.0],
            "sex(0male1female)": [0, 1, np.nan, 1, 0, 1],
            "BMI": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan],
            "smoking(0no1yes)": [0, 1, 1, np.nan, 0, 1],
            "drinking(0no1yes)": [0, 0, 1, 1, np.nan, 1],
            "first_degree_family_history_of_GC(0no1yes)": [0, 1, 0, np.nan, 0, 1],
            "Hp_Infection(0no1yes)": [1, 0, np.nan, 1, 0, np.nan],
        }
    )
    sample_index = pd.DataFrame(
        {
            "sample_id": [
                "SYNTHETIC-106",
                "SYNTHETIC-105",
                "SYNTHETIC-104",
                "SYNTHETIC-103",
                "SYNTHETIC-102",
                "SYNTHETIC-101",
            ],
            "label": [1, 1, np.nan, 0, 0, 0],
            "disease_category": [5, 4, np.nan, 2, 1, 0],
            "in_final_output": [False, False, True, True, True, True],
        }
    )
    return metadata, sample_index


def _round_trip_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build a synthetic cohort whose exact 10/27 proportion stresses CSV parsing."""
    source_n = 29
    retained_n = 27
    sample_ids = [f"ROUNDTRIP-{index:02d}" for index in range(source_n)]
    metadata = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "age": np.arange(source_n, dtype=float) + 20.0,
            "sex(0male1female)": [1] * 10 + [0] * (source_n - 10),
            "BMI": np.arange(source_n, dtype=float) / 10.0 + 20.0,
            "smoking(0no1yes)": [0, 1] * 14 + [0],
            "drinking(0no1yes)": [1, 0] * 14 + [1],
            "first_degree_family_history_of_GC(0no1yes)": [0] * source_n,
            "Hp_Infection(0no1yes)": [1, 0] * 14 + [1],
        }
    )
    disease = np.resize(np.arange(6, dtype=int), source_n)
    sample_index = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "label": np.isin(disease, [3, 4, 5]).astype(int),
            "disease_category": disease,
            "in_final_output": [True] * retained_n + [False] * (source_n - retained_n),
        }
    )
    config = _config()
    config["population_counts"] = {
        "complete_source": source_n,
        "qc_retained": retained_n,
        "qc_excluded": source_n - retained_n,
    }
    return metadata, sample_index, config


def _build() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the real calculator on the common synthetic fixture."""
    metadata, sample_index = _inputs()
    return _module().build_characteristics(metadata, sample_index, _config())


def _overlap_config(tmp_path: Path) -> dict[str, Any]:
    """Return aggregate-only anchor paths for direct reconciliation tests."""
    config = _config()
    config["anchors"] = {
        "overlap_summary": {"path": "anchors/frozen_overlap_summary.csv"},
        "overlap_smd": {"path": "anchors/frozen_overlap_smd.csv"},
        "retained_group_summary": {"path": "anchors/frozen_retained_summary.csv"},
    }
    (tmp_path / "anchors").mkdir()
    return config


def _write_independent_overlap_anchors(tmp_path: Path) -> None:
    """Write frozen-style aggregate CSV literals independently of new frames."""
    summary_text = """variable,level,group,n_total,n_nonmissing,n_missing,count,proportion,mean,sd,median,q1,q3
age,,included,972,972,0,,,53.641975308641975,14.95091203771298,56.0,42.0,65.0
sex,female,included,972,972,0,360,0.37037037037037035,,,,,
endpoint,yes,included,972,972,0,413,0.42489711934156377,,,,,
disease_category,HC,included,972,972,0,158,0.16255144032921812,,,,,
disease_category,BGL,included,972,972,0,352,0.36213991769547327,,,,,
disease_category,LGIN,included,972,972,0,49,0.050411522633744855,,,,,
disease_category,HGIN,included,972,972,0,71,0.07304526748971193,,,,,
disease_category,EGC,included,972,972,0,122,0.12551440329218108,,,,,
disease_category,AGC,included,972,972,0,220,0.22633744855967078,,,,,
missingness__age,missing,included,972,972,0,0,0.0,,,,,
missingness__sex,missing,included,972,972,0,0,0.0,,,,,
missingness__endpoint,missing,included,972,972,0,0,0.0,,,,,
missingness__disease_category,missing,included,972,972,0,0,0.0,,,,,
age,,excluded,68,68,0,,,55.838235294117645,13.691276549531482,59.0,47.25,64.25
sex,female,excluded,68,68,0,27,0.39705882352941174,,,,,
endpoint,yes,excluded,68,68,0,33,0.4852941176470588,,,,,
disease_category,HC,excluded,68,68,0,5,0.07352941176470588,,,,,
disease_category,BGL,excluded,68,68,0,26,0.38235294117647056,,,,,
disease_category,LGIN,excluded,68,68,0,4,0.058823529411764705,,,,,
disease_category,HGIN,excluded,68,68,0,5,0.07352941176470588,,,,,
disease_category,EGC,excluded,68,68,0,15,0.22058823529411764,,,,,
disease_category,AGC,excluded,68,68,0,13,0.19117647058823528,,,,,
missingness__age,missing,excluded,68,68,0,0,0.0,,,,,
missingness__sex,missing,excluded,68,68,0,0,0.0,,,,,
missingness__endpoint,missing,excluded,68,68,0,0,0.0,,,,,
missingness__disease_category,missing,excluded,68,68,0,0,0.0,,,,,
"""
    smd_text = (
        "variable,level,smd_included_minus_excluded,missingness_smd_included_minus_excluded,"
        "n_included_nonmissing,n_excluded_nonmissing,estimability_status,nonestimable_reason\n"
        """age,,-0.15321030158277832,0.0,972,68,estimable,
sex,female,-0.054902469106964016,0.0,972,68,estimable,
endpoint,yes,-0.1215077278214748,0.0,972,68,estimable,
disease_category,HC,0.27856725340104865,0.0,972,68,estimable,
disease_category,BGL,-0.04182311254585088,,972,68,estimable,
disease_category,LGIN,-0.0370257818394607,,972,68,estimable,
disease_category,HGIN,-0.001857752958149615,,972,68,estimable,
disease_category,EGC,-0.2533323123359978,,972,68,estimable,
disease_category,AGC,0.086594879257065,,972,68,estimable,
"""
    )
    (tmp_path / "anchors/frozen_overlap_summary.csv").write_text(summary_text, encoding="utf-8")
    (tmp_path / "anchors/frozen_overlap_smd.csv").write_text(smd_text, encoding="utf-8")


def _independent_new_overlap_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build nullable aggregate frames without calling production calculators."""
    group_values = {
        "qc_retained": {
            "n": 972,
            "age": (53.641975308641975, 14.95091203771298, 56.0, 42.0, 65.0),
            "counts": [360, 413, 158, 352, 49, 71, 122, 220],
            "proportions": [
                0.37037037037037035,
                0.42489711934156377,
                0.16255144032921812,
                0.36213991769547327,
                0.050411522633744855,
                0.07304526748971193,
                0.12551440329218108,
                0.22633744855967078,
            ],
        },
        "qc_excluded": {
            "n": 68,
            "age": (55.838235294117645, 13.691276549531482, 59.0, 47.25, 64.25),
            "counts": [27, 33, 5, 26, 4, 5, 15, 13],
            "proportions": [
                0.39705882352941174,
                0.4852941176470588,
                0.07352941176470588,
                0.38235294117647056,
                0.058823529411764705,
                0.07352941176470588,
                0.22058823529411764,
                0.19117647058823528,
            ],
        },
    }
    summary_records: list[dict[str, object]] = []
    categorical_rows = ROW_ORDER[1:9]
    for group, values in group_values.items():
        n = int(values["n"])
        mean, sd, median, q1, q3 = values["age"]
        summary_records.append(
            {
                "variable": "age",
                "level": None,
                "group": group,
                "n_total": n,
                "n_nonmissing": n,
                "n_missing": 0,
                "count": None,
                "denominator_n": None,
                "proportion": None,
                "mean": mean,
                "sd": sd,
                "median": median,
                "q1": q1,
                "q3": q3,
                "denominator_rule": "not_applicable",
            }
        )
        for (variable, level), count, proportion in zip(
            categorical_rows,
            values["counts"],
            values["proportions"],
            strict=True,
        ):
            summary_records.append(
                {
                    "variable": variable,
                    "level": level,
                    "group": group,
                    "n_total": n,
                    "n_nonmissing": n,
                    "n_missing": 0,
                    "count": count,
                    "denominator_n": n,
                    "proportion": proportion,
                    "mean": None,
                    "sd": None,
                    "median": None,
                    "q1": None,
                    "q3": None,
                    "denominator_rule": "variable_specific_nonmissing",
                }
            )
    summary = pd.DataFrame(summary_records, columns=SUMMARY_COLUMNS)
    for column in ("variable", "level", "denominator_rule"):
        summary[column] = summary[column].astype("string")
    summary["group"] = pd.Categorical(
        summary["group"], categories=["qc_retained", "qc_excluded"], ordered=True
    )
    for column in ("n_total", "n_nonmissing", "n_missing", "count", "denominator_n"):
        summary[column] = summary[column].astype("Int64")
    for column in ("proportion", "mean", "sd", "median", "q1", "q3"):
        summary[column] = summary[column].astype("Float64")

    distribution_smd = [
        ("age", None, -0.15321030158277832),
        ("sex", "female", -0.054902469106964016),
        ("endpoint", "yes", -0.1215077278214748),
        ("disease_category", "HC", 0.27856725340104865),
        ("disease_category", "BGL", -0.04182311254585088),
        ("disease_category", "LGIN", -0.0370257818394607),
        ("disease_category", "HGIN", -0.001857752958149615),
        ("disease_category", "EGC", -0.2533323123359978),
        ("disease_category", "AGC", 0.086594879257065),
    ]
    smd_records = [
        {
            "variable": variable,
            "level": level,
            "comparison_type": "distribution",
            "smd_qc_retained_minus_excluded": estimate,
            "n_qc_retained_used": 972,
            "n_qc_excluded_used": 68,
            "estimability_status": "estimable",
            "nonestimable_reason": None,
        }
        for variable, level, estimate in distribution_smd
    ]
    smd_records.extend(
        {
            "variable": variable,
            "level": "missing",
            "comparison_type": "missingness",
            "smd_qc_retained_minus_excluded": 0.0,
            "n_qc_retained_used": 972,
            "n_qc_excluded_used": 68,
            "estimability_status": "estimable",
            "nonestimable_reason": None,
        }
        for variable in ("age", "sex", "endpoint", "disease_category")
    )
    smd = pd.DataFrame(smd_records, columns=SMD_COLUMNS)
    for column in ("variable", "level", "nonestimable_reason"):
        smd[column] = smd[column].astype("string")
    smd["comparison_type"] = pd.Categorical(
        smd["comparison_type"], categories=["distribution", "missingness"], ordered=True
    )
    smd["estimability_status"] = pd.Categorical(
        smd["estimability_status"], categories=["estimable", "non_estimable"], ordered=True
    )
    smd["smd_qc_retained_minus_excluded"] = smd["smd_qc_retained_minus_excluded"].astype("Float64")
    smd["n_qc_retained_used"] = smd["n_qc_retained_used"].astype("Int64")
    smd["n_qc_excluded_used"] = smd["n_qc_excluded_used"].astype("Int64")
    return summary, smd


def _independent_retained_summary() -> pd.DataFrame:
    """Build the new retained aggregate rows without production helpers."""
    rows = [
        ("body_mass_index", None, 972, 972, 0, None, None, None, 24.20, 22.00, 26.23, "not_applicable"),
        (
            "smoking_history",
            "yes",
            972,
            972,
            0,
            277,
            972,
            277 / 972,
            None,
            None,
            None,
            "variable_specific_nonmissing",
        ),
        (
            "alcohol_use_history",
            "yes",
            972,
            972,
            0,
            238,
            972,
            238 / 972,
            None,
            None,
            None,
            "variable_specific_nonmissing",
        ),
        (
            "first_degree_family_history_of_gastric_cancer",
            "yes",
            972,
            972,
            0,
            50,
            972,
            50 / 972,
            None,
            None,
            None,
            "variable_specific_nonmissing",
        ),
        ("helicobacter_pylori_status", "positive", 972, 878, 94, 206, 972, 206 / 972, None, None, None, "group_total"),
        ("helicobacter_pylori_status", "negative", 972, 878, 94, 672, 972, 672 / 972, None, None, None, "group_total"),
        ("helicobacter_pylori_status", "missing", 972, 878, 94, 94, 972, 94 / 972, None, None, None, "group_total"),
    ]
    records = []
    for variable, level, total, nonmissing, missing, count, denominator, proportion, median, q1, q3, rule in rows:
        records.append(
            {
                "variable": variable,
                "level": level,
                "group": "qc_retained",
                "n_total": total,
                "n_nonmissing": nonmissing,
                "n_missing": missing,
                "count": count,
                "denominator_n": denominator,
                "proportion": proportion,
                "mean": None,
                "sd": None,
                "median": median,
                "q1": q1,
                "q3": q3,
                "denominator_rule": rule,
            }
        )
    frame = pd.DataFrame(records, columns=SUMMARY_COLUMNS)
    frame["group"] = pd.Categorical(frame["group"], categories=["qc_retained", "qc_excluded"], ordered=True)
    frame["variable"] = frame["variable"].astype("string")
    frame["level"] = frame["level"].astype("string")
    return frame


def test_reconcile_overlap_accepts_independent_default_and_nullable_dtype_fixtures(tmp_path: Path) -> None:
    """Representation differences must not defeat exact equal aggregate scalars."""
    module = _module()
    config = _overlap_config(tmp_path)
    _write_independent_overlap_anchors(tmp_path)
    summary, smd = _independent_new_overlap_frames()

    result = module.reconcile_overlap(summary, smd, tmp_path, config)

    assert result == OrderedDict(
        [
            ("summary_distribution_rows", 18),
            ("summary_missingness_variables", 4),
            ("distribution_smd_rows", 9),
            ("missingness_smd_rows", 4),
            ("exact_full_precision_agreement", True),
            ("estimability_state_agreement", True),
        ]
    )


def test_reconcile_overlap_reports_all_safe_aggregate_scalar_mismatches(tmp_path: Path) -> None:
    """The diagnostic must identify every mismatch without participant-capable context."""
    module = _module()
    config = _overlap_config(tmp_path)
    _write_independent_overlap_anchors(tmp_path)
    summary, smd = _independent_new_overlap_frames()
    summary.loc[
        (summary["group"].astype(str) == "qc_retained") & (summary["variable"] == "age"), "mean"
    ] = 54.0
    summary.loc[
        (summary["group"].astype(str) == "qc_retained") & (summary["variable"] == "sex"), "count"
    ] = 361
    smd.loc[
        (smd["comparison_type"].astype(str) == "distribution") & (smd["variable"] == "endpoint"),
        "smd_qc_retained_minus_excluded",
    ] = -0.12

    with pytest.raises(ValueError) as exc_info:
        module.reconcile_overlap(summary, smd, tmp_path, config)

    diagnostic = json.loads(str(exc_info.value))
    assert diagnostic == [
        {
            "kind": "summary_distribution",
            "group": "qc_retained",
            "variable": "age",
            "level": None,
            "field": "mean",
            "observed": 54.0,
            "expected": 53.641975308641975,
        },
        {
            "kind": "summary_distribution",
            "group": "qc_retained",
            "variable": "sex",
            "level": "female",
            "field": "count",
            "observed": 361,
            "expected": 360,
        },
        {
            "kind": "distribution_smd",
            "group": "qc_retained_minus_qc_excluded",
            "variable": "endpoint",
            "level": "yes",
            "field": "smd_qc_retained_minus_excluded",
            "observed": -0.12,
            "expected": -0.1215077278214748,
        },
    ]
    expected_keys = {"kind", "group", "variable", "level", "field", "observed", "expected"}
    assert all(set(record) == expected_keys for record in diagnostic)
    message = str(exc_info.value)
    forbidden = (
        "SYNTHETIC-PARTICIPANT-007",
        str(tmp_path),
        "2030-01-01",
        ".csv",
        "sample_id",
        "collection_time",
        "collection_location",
        "breath_filename",
    )
    assert all(value not in message for value in forbidden)


def test_reconcile_overlap_round_trips_frozen_full_precision_float_text(tmp_path: Path) -> None:
    """Frozen decimal text must recover the exact independently recomputed floats."""
    module = _module()
    config = _overlap_config(tmp_path)
    _write_independent_overlap_anchors(tmp_path)
    summary, smd = _independent_new_overlap_frames()
    categorical = summary["variable"].isin({"sex", "endpoint", "disease_category"})
    summary.loc[categorical, "proportion"] = (
        summary.loc[categorical, "count"].astype(float)
        / summary.loc[categorical, "denominator_n"].astype(float)
    )
    exact_smd = {
        ("age", None): -0.15321030158277832,
        ("sex", "female"): -0.054902469106964016,
        ("endpoint", "yes"): -0.1215077278214748,
        ("disease_category", "HC"): 0.27856725340104865,
        ("disease_category", "BGL"): -0.04182311254585088,
        ("disease_category", "LGIN"): -0.0370257818394607,
        ("disease_category", "HGIN"): -0.001857752958149615,
        ("disease_category", "EGC"): -0.2533323123359978,
        ("disease_category", "AGC"): 0.086594879257065,
    }
    for (variable, level), estimate in exact_smd.items():
        selector = (smd["comparison_type"].astype(str) == "distribution") & (smd["variable"] == variable)
        selector &= smd["level"].isna() if level is None else smd["level"].eq(level)
        smd.loc[selector, "smd_qc_retained_minus_excluded"] = estimate

    result = module.reconcile_overlap(summary, smd, tmp_path, config)

    assert result["exact_full_precision_agreement"] is True


def test_reconcile_retained_group_accepts_independent_h_pylori_denominators(tmp_path: Path) -> None:
    """The old nonmissing anchor and new complete-group H pylori denominator must coexist."""
    module = _module()
    config = _overlap_config(tmp_path)
    retained_anchor = """variable,Total (N=972)
"BMI, kg/m² (median [IQR])",24.20 [22.00–26.23]
"Current/ever smoker, n (%)",277/972 (28.5%)
"Current/ever drinker, n (%)",238/972 (24.5%)
"First-degree family history of GC, n (%)",50/972 (5.1%)
"Hp_infection positive, n (%)",206/878 (23.5%)
"Hp_infection missing, n (%)",94/972 (9.7%)
"""
    (tmp_path / "anchors/frozen_retained_summary.csv").write_text(retained_anchor, encoding="utf-8")

    result = module.reconcile_retained_group(_independent_retained_summary(), tmp_path, config)

    assert result["body_mass_index_two_decimal_display_agreement"] is True
    assert result["helicobacter_pylori_positive_count"] == 206
    assert result["helicobacter_pylori_negative_count"] == 672
    assert result["helicobacter_pylori_missing_count"] == 94
    assert result["helicobacter_pylori_group_total_denominator"] == 972


def test_config_and_all_authorized_implementation_targets_exist() -> None:
    """Omitting any one authorized implementation artifact must fail the package contract."""
    expected = [
        CONFIG_PATH,
        MODULE_PATH,
        PROJECT_ROOT / "scripts/run_qc_retained_excluded_characteristics.py",
    ]
    assert all(path.is_file() for path in expected)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["package_id"] == "qc_retained_excluded_characteristics"


def test_exact_schemas_row_counts_and_scientific_order_are_preserved() -> None:
    """Schema drift, row omission, or reordering must fail the fixed inventory."""
    summary, smd = _build()
    assert summary.columns.tolist() == SUMMARY_COLUMNS
    assert smd.columns.tolist() == SMD_COLUMNS
    assert len(summary) == 32
    assert len(smd) == 24
    summary_identity = [
        (row.variable, None if pd.isna(row.level) else row.level)
        for row in summary.loc[summary["group"] == "qc_retained"].itertuples()
    ]
    assert summary_identity == ROW_ORDER
    assert summary["group"].tolist() == ["qc_retained"] * 16 + ["qc_excluded"] * 16
    smd_identity = [
        (row.variable, None if pd.isna(row.level) else row.level, row.comparison_type)
        for row in smd.itertuples()
    ]
    assert smd_identity == [(*identity, "distribution") for identity in ROW_ORDER] + [
        (*identity, "missingness") for identity in MISSINGNESS_ORDER
    ]


def test_continuous_statistics_use_sample_sd_and_linear_quantiles() -> None:
    """Population SD or a non-linear quantile method must fail hand-derived values."""
    summary, _ = _build()
    age = summary.loc[(summary["group"] == "qc_retained") & (summary["variable"] == "age")].iloc[0]
    assert age["n_total"] == 4
    assert age["n_nonmissing"] == 3
    assert age["n_missing"] == 1
    assert age["mean"] == 20.0
    assert age["sd"] == 10.0
    assert age["median"] == 20.0
    assert age["q1"] == 15.0
    assert age["q3"] == 25.0
    assert pd.isna(age["count"])
    assert pd.isna(age["denominator_n"])
    assert age["denominator_rule"] == "not_applicable"


def test_signed_continuous_and_binary_smd_direction_is_retained_minus_excluded() -> None:
    """Reversing either SMD subtraction direction must fail independent calculations."""
    _, smd = _build()
    age = smd.loc[(smd["variable"] == "age") & (smd["comparison_type"] == "distribution")].iloc[0]
    expected_age = (20.0 - 45.0) / np.sqrt((100.0 + 50.0) / 2.0)
    assert age["smd_qc_retained_minus_excluded"] == pytest.approx(expected_age)
    sex = smd.loc[(smd["variable"] == "sex") & (smd["comparison_type"] == "distribution")].iloc[0]
    retained_p = 2.0 / 3.0
    excluded_p = 1.0 / 2.0
    expected_sex = (retained_p - excluded_p) / np.sqrt(
        (retained_p * (1.0 - retained_p) + excluded_p * (1.0 - excluded_p)) / 2.0
    )
    assert sex["smd_qc_retained_minus_excluded"] == pytest.approx(expected_sex)


def test_all_disease_and_h_pylori_levels_use_their_fixed_denominators() -> None:
    """Dropping a zero category or using the old H pylori denominator must fail."""
    summary, _ = _build()
    retained = summary.loc[summary["group"] == "qc_retained"]
    disease = retained.loc[retained["variable"] == "disease_category"]
    assert disease["level"].tolist() == ["HC", "BGL", "LGIN", "HGIN", "EGC", "AGC"]
    assert disease["count"].tolist() == [1, 1, 1, 0, 0, 0]
    assert disease["denominator_n"].tolist() == [3] * 6
    hp = retained.loc[retained["variable"] == "helicobacter_pylori_status"]
    assert hp["level"].tolist() == ["positive", "negative", "missing"]
    assert hp["count"].tolist() == [2, 1, 1]
    assert hp["denominator_n"].tolist() == [4, 4, 4]
    assert hp["proportion"].tolist() == [0.5, 0.25, 0.25]
    assert hp["n_nonmissing"].tolist() == [3, 3, 3]
    assert hp["n_missing"].tolist() == [1, 1, 1]
    assert hp["denominator_rule"].tolist() == ["group_total"] * 3


def test_missingness_rows_and_nonestimable_states_are_never_dropped() -> None:
    """Result-dependent row suppression or false estimability must fail."""
    _, smd = _build()
    observed_missingness = smd.loc[
        smd["comparison_type"] == "missingness", ["variable", "level"]
    ].apply(tuple, axis=1)
    assert observed_missingness.tolist() == MISSINGNESS_ORDER
    bmi = smd.loc[
        (smd["variable"] == "body_mass_index") & (smd["comparison_type"] == "distribution")
    ].iloc[0]
    assert pd.isna(bmi["smd_qc_retained_minus_excluded"])
    assert bmi["estimability_status"] == "non_estimable"
    assert bmi["nonestimable_reason"] == "smd_fewer_than_two_nonmissing"
    assert bmi["n_qc_retained_used"] == 4
    assert bmi["n_qc_excluded_used"] == 1
    hp_missing = smd.loc[
        (smd["variable"] == "helicobacter_pylori_status") & (smd["level"] == "missing")
    ]
    assert len(hp_missing) == 1


@pytest.mark.parametrize(
    ("frame_name", "column", "invalid"),
    [
        ("metadata", "sex(0male1female)", 2),
        ("metadata", "smoking(0no1yes)", "1"),
        ("metadata", "drinking(0no1yes)", -1),
        ("metadata", "first_degree_family_history_of_GC(0no1yes)", 0.5),
        ("metadata", "Hp_Infection(0no1yes)", 3),
        ("sample_index", "label", 2),
        ("sample_index", "disease_category", 6),
    ],
)
def test_invalid_binary_or_category_coding_fails_closed(frame_name: str, column: str, invalid: object) -> None:
    """Lossy coercion of malformed clinical coding must fail before summaries."""
    metadata, sample_index = _inputs()
    frame = metadata if frame_name == "metadata" else sample_index
    if isinstance(invalid, str):
        frame[column] = frame[column].astype(object)
    frame.loc[0, column] = invalid
    with pytest.raises(ValueError, match="coding"):
        _module().build_characteristics(metadata, sample_index, _config())


@pytest.mark.parametrize("mutation", ["duplicate_metadata", "duplicate_index", "population_mismatch"])
def test_duplicate_or_mismatched_join_keys_fail_without_identity_disclosure(mutation: str) -> None:
    """A non-one-to-one join must fail with a sanitized message."""
    metadata, sample_index = _inputs()
    if mutation == "duplicate_metadata":
        metadata.loc[1, "sample_id"] = metadata.loc[0, "sample_id"]
    elif mutation == "duplicate_index":
        sample_index.loc[1, "sample_id"] = sample_index.loc[0, "sample_id"]
    else:
        sample_index.loc[0, "sample_id"] = "SYNTHETIC-999"
    with pytest.raises(ValueError) as exc_info:
        _module().build_characteristics(metadata, sample_index, _config())
    message = str(exc_info.value)
    assert all(str(value) not in message for value in metadata["sample_id"].tolist())
    assert "SYNTHETIC-999" not in message


@pytest.mark.parametrize(
    "counts",
    [
        {"complete_source": 7, "qc_retained": 4, "qc_excluded": 2},
        {"complete_source": 6, "qc_retained": 5, "qc_excluded": 1},
        {"complete_source": 6, "qc_retained": 3, "qc_excluded": 3},
    ],
)
def test_incorrect_source_retained_or_excluded_counts_fail_closed(counts: dict[str, int]) -> None:
    """Any departure from the configured 1040/972/68 analogue must halt."""
    metadata, sample_index = _inputs()
    config = _config()
    config["population_counts"] = counts
    with pytest.raises(ValueError, match="count"):
        _module().build_characteristics(metadata, sample_index, config)


def test_hash_mismatch_halts_before_participant_capable_reader(tmp_path: Path) -> None:
    """A bad whole-file hash must prevent even a projected source read."""
    module = _module()
    config = _config()
    for record in config["inputs"].values():
        path = tmp_path / record["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic bytes\n", encoding="utf-8")
    calls: list[tuple[object, ...]] = []

    def forbidden_reader(*args: object, **kwargs: object) -> pd.DataFrame:
        calls.append(args)
        raise RuntimeError("reader must not be called")

    with pytest.raises(ValueError, match="hash"):
        module.load_authorized_inputs(tmp_path, config, reader=forbidden_reader)
    assert calls == []


def test_exact_input_projection_is_enforced(tmp_path: Path) -> None:
    """Broader or reordered usecols must fail the minimum-column contract."""
    module = _module()
    config = _config()
    metadata, sample_index = _inputs()
    for key, frame in (("raw_metadata", metadata), ("sample_index", sample_index)):
        path = tmp_path / config["inputs"][key]["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        config["inputs"][key]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    observed_usecols: list[list[str]] = []

    def recording_reader(path: Path, *, usecols: list[str]) -> pd.DataFrame:
        observed_usecols.append(list(usecols))
        return pd.read_csv(path, usecols=usecols)

    loaded_metadata, loaded_index = module.load_authorized_inputs(tmp_path, config, reader=recording_reader)
    assert observed_usecols == [config["input_columns"]["raw_metadata"], config["input_columns"]["sample_index"]]
    assert loaded_metadata.columns.tolist() == config["input_columns"]["raw_metadata"]
    assert loaded_index.columns.tolist() == config["input_columns"]["sample_index"]


def test_zero_denominator_nonidentical_binary_smd_uses_frozen_reason() -> None:
    """Opposite constant groups must not emit an infinite binary SMD."""
    module = _module()
    result = module.signed_binary_smd([1, 1], [0, 0])
    assert result.estimate is None
    assert result.status == "non_estimable"
    assert result.reason == "smd_zero_denominator_nonidentical"


def test_stale_output_refusal_and_atomic_manifest_last_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing targets must survive, while a clean promotion exposes the manifest last."""
    module = _module()
    targets = OrderedDict(
        [
            ("summary", tmp_path / "summary.csv"),
            ("smd", tmp_path / "smd.csv"),
            ("manifest", tmp_path / "manifest.json"),
        ]
    )
    targets["summary"].write_bytes(b"sentinel\n")
    with pytest.raises(FileExistsError, match="output target"):
        module.promote_serialized_outputs(targets, {name: name.encode() for name in targets})
    assert targets["summary"].read_bytes() == b"sentinel\n"
    targets["summary"].unlink()
    promoted: list[str] = []
    real_replace = module.os.replace

    def record_replace(source: Path, destination: Path) -> None:
        promoted.append(destination.name)
        real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", record_replace)
    module.promote_serialized_outputs(targets, {name: name.encode() for name in targets})
    assert promoted == ["summary.csv", "smd.csv", "manifest.json"]
    assert not list(tmp_path.glob(".*.tmp"))


def test_serialization_is_deterministic_and_uses_empty_csv_missing_fields() -> None:
    """Runtime variation or textual null sentinels must fail byte reproducibility."""
    module = _module()
    summary, smd = _build()
    first_summary = module.serialize_csv(summary, SUMMARY_COLUMNS)
    second_summary = module.serialize_csv(summary.copy(deep=True), SUMMARY_COLUMNS)
    first_smd = module.serialize_csv(smd, SMD_COLUMNS)
    assert first_summary == second_summary
    assert b",NA," not in first_summary
    assert b"None" not in first_summary + first_smd
    assert b"null" not in first_summary + first_smd
    manifest = OrderedDict([("schema_version", "synthetic.v1"), ("checks", {"deterministic": True})])
    assert module.serialize_manifest(manifest) == module.serialize_manifest(manifest)
    assert b"\n  " not in module.serialize_manifest(manifest)


@pytest.mark.parametrize(
    "payload",
    [
        b"sample_id,metric\n101,1\n",
        b'{"path":"[PRIVATE_PATH_REDACTED]"}\n',
        b'{"collection_time":"2030-01-01"}\n',
        b'{"breath_filename":"private_sample.tdms"}\n',
        b'{"collection_location":"ward"}\n',
        b'{"p_cal":0.8}\n',
    ],
)
def test_persisted_privacy_rejects_ids_paths_dates_filenames_and_unapproved_fields(payload: bytes) -> None:
    """A persisted participant-capable or unapproved field must fail without echoing it."""
    with pytest.raises(ValueError, match="privacy") as exc_info:
        _module().validate_persisted_privacy({"synthetic": payload})
    assert payload.decode("utf-8").strip() not in str(exc_info.value)


def test_validators_read_serialized_outputs_back_with_exact_contract(tmp_path: Path) -> None:
    """Malformed persisted schemas or row identities must fail read-back validation."""
    module = _module()
    summary, smd = _build()
    summary_path = tmp_path / "summary.csv"
    smd_path = tmp_path / "smd.csv"
    manifest_path = tmp_path / "manifest.json"
    summary_path.write_bytes(module.serialize_csv(summary, SUMMARY_COLUMNS))
    smd_path.write_bytes(module.serialize_csv(smd, SMD_COLUMNS))
    manifest = OrderedDict(
        [
            ("schema_version", "qc_retained_excluded_characteristics.manifest.v1"),
            ("package_id", "qc_retained_excluded_characteristics"),
        ]
    )
    manifest_path.write_bytes(module.serialize_manifest(manifest))
    module.read_and_validate_summary(summary_path, _config())
    module.read_and_validate_smd(smd_path, _config())
    assert module.read_and_validate_manifest(manifest_path)["package_id"] == "qc_retained_excluded_characteristics"
    broken = summary.drop(columns=["denominator_rule"])
    summary_path.write_bytes(module.serialize_csv(broken, broken.columns.tolist()))
    with pytest.raises(ValueError, match="schema"):
        module.read_and_validate_summary(summary_path, _config())


def test_scientific_csv_read_back_round_trips_exact_float_payloads(tmp_path: Path) -> None:
    """Candidate and persisted readers must preserve exact calculated float payloads."""
    module = _module()
    metadata, sample_index, config = _round_trip_inputs()
    summary, smd = module.build_characteristics(metadata, sample_index, config)
    summary_path = tmp_path / "summary.csv"
    smd_path = tmp_path / "smd.csv"
    summary_path.write_bytes(module.serialize_csv(summary, SUMMARY_COLUMNS))
    smd_path.write_bytes(module.serialize_csv(smd, SMD_COLUMNS))

    read_summary = module.read_and_validate_summary(summary_path, config)
    read_smd = module.read_and_validate_smd(smd_path, config)

    pd.testing.assert_frame_equal(read_summary, summary, check_exact=True)
    pd.testing.assert_frame_equal(read_smd, smd, check_exact=True)


def test_safe_aggregate_products_contain_no_synthetic_participant_values() -> None:
    """Participant keys and values used for joins must never reach serialized products."""
    module = _module()
    summary, smd = _build()
    serialized = {
        "summary": module.serialize_csv(summary, SUMMARY_COLUMNS),
        "smd": module.serialize_csv(smd, SMD_COLUMNS),
    }
    module.validate_persisted_privacy(serialized)
    combined = b"".join(serialized.values())
    assert b"sample_id" not in combined
    metadata, _ = _inputs()
    assert all(str(value).encode() not in combined for value in metadata["sample_id"])


def test_manifest_serialization_rejects_runtime_timestamp_and_absolute_path() -> None:
    """Nondeterministic time or host-specific provenance must not enter the manifest."""
    module = _module()
    with pytest.raises(ValueError, match="privacy"):
        module.serialize_manifest({"generated_at_utc": "2030-01-01T00:00:00Z"})
    with pytest.raises(ValueError, match="privacy"):
        module.serialize_manifest({"path": "/tmp/private"})


def test_production_config_freezes_exact_counts_columns_variables_and_outputs() -> None:
    """Changing the approved production contract must fail explicit structural assertions."""
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["population_counts"] == {"complete_source": 1040, "qc_retained": 972, "qc_excluded": 68}
    assert config["variable_order"] == [list(item) for item in ROW_ORDER]
    assert config["missingness_variable_order"] == [item[0] for item in MISSINGNESS_ORDER]
    assert config["input_columns"]["raw_metadata"] == _config()["input_columns"]["raw_metadata"]
    assert config["input_columns"]["sample_index"] == _config()["input_columns"]["sample_index"]
    assert [record["path"] for record in config["outputs"].values()] == [
        "outputs/analysis/qc_retained_excluded_characteristics_summary.csv",
        "outputs/analysis/qc_retained_excluded_characteristics_smd.csv",
        "outputs/analysis/qc_retained_excluded_characteristics_manifest.json",
    ]


def test_manifest_bytes_are_valid_compact_json() -> None:
    """Pretty printing or non-JSON serialization must fail the compact contract."""
    module = _module()
    payload = OrderedDict([("schema_version", "synthetic.v1"), ("value", 1)])
    encoded = module.serialize_manifest(payload)
    assert json.loads(encoded) == payload
    assert encoded == b'{"schema_version":"synthetic.v1","value":1}\n'
