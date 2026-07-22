"""Contract tests for Session 1 locked-score framework."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
from collections import OrderedDict
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

import src.analysis.posthoc_final_locked_scores as locked_scores

from scripts.run_posthoc_final_locked_scores import build_parser, main
from src.analysis.posthoc_final_common import AccessStateError, AccessStateTracker
from src.analysis.posthoc_final_locked_scores import (
    AMENDMENT_CHANGED_PATHS,
    AMENDMENT_FREEZE_DEFAULT_FOR_TESTS,
    SCORE_BLIND_PREFLIGHT_RELATIVE_PATH,
    SESSION0_FREEZE,
    SESSION_1_CORRECTIONS_IMPLEMENTED,
    SESSION_4_COMPLETION_LOG_PATH,
    SESSION_4_SCIENTIFIC_OUTPUT_PATHS,
    build_score_blind_preflight,
    difficult_spectrum_filter,
    evaluate_conditional_preflight_retry_authorization,
    execute_approved,
    execute_approved_framework,
    _require_bool_values,
    _split_membership,
    _validate_disease_category_codes,
    _write_atomic_csv,
    reconstruct_qc5,
    run_score_blind_preflight,
    strict_qc5_correctness_inventory,
    validate_amendment_freeze_commit,
    validate_locked_score_alignment,
    validate_pre_access_authority,
    within_date_concordance,
)
from src.config import PROJECT_ROOT


CURRENT_MANIFEST_RELATIVE_PATH = "outputs/analysis/posthoc_final_freeze_manifest.json"


def _six_category_frame() -> pd.DataFrame:
    """Return a score-free synthetic lineage fixture with all disease categories."""
    return pd.DataFrame(
        {
            "sample_id": [10, 11, 12, 13, 14, 15],
            "y_true": [0, 0, 0, 1, 1, 1],
            "label": [0, 0, 0, 1, 1, 1],
            "disease_category": [0, 1, 2, 3, 4, 5],
            "collection_time": ["2025/1/1"] * 6,
            "in_final_output": [True] * 6,
        }
    )


def test_difficult_spectrum_keeps_only_categories_one_through_four() -> None:
    """Difficult spectrum includes 1/2 negatives and 3/4 positives in source order."""
    filtered = difficult_spectrum_filter(_six_category_frame())
    assert filtered["sample_id"].tolist() == [11, 12, 13, 14]
    assert filtered["y_true"].tolist() == [0, 0, 1, 1]
    assert filtered["disease_category"].tolist() == [1, 2, 3, 4]
    assert filtered.attrs["disease_category_counts"] == OrderedDict(
        [("0", 0), ("1", 1), ("2", 1), ("3", 1), ("4", 1), ("5", 0)]
    )


def test_difficult_spectrum_rejects_label_disagreement() -> None:
    """Lineage labels must agree before spectrum filtering."""
    fixture = _six_category_frame()
    fixture.loc[3, "label"] = 0
    with pytest.raises(ValueError):
        difficult_spectrum_filter(fixture)


def test_oof_and_test_difficult_spectrum_views_remain_separate() -> None:
    """Filtering never pools source view rows."""
    oof = _six_category_frame()
    test = _six_category_frame().assign(sample_id=lambda value: value["sample_id"] + 100)
    result = {
        "oof": difficult_spectrum_filter(oof),
        "test": difficult_spectrum_filter(test),
    }
    assert set(result["oof"]["sample_id"]).isdisjoint(set(result["test"]["sample_id"]))


def test_within_date_concordance_pair_weighting_and_ties() -> None:
    """The estimator weights dates by all comparable positive-negative pairs."""
    result = within_date_concordance(
        y_true=np.array([1, 0, 1, 0, 0]),
        scores=np.array([0.6, 0.4, 0.5, 0.5, 0.1]),
        dates=np.array(["2025/1/1", "2025/1/1", "2025/1/2", "2025/1/2", "2025/1/3"]),
        view="oof",
    )
    # Date one: 1/1 concordant; date two: tie = 1/2; date three is non-comparable.
    assert result["estimate"] == pytest.approx(0.75)
    assert result["n_dates"] == 3
    assert result["n_comparable_dates"] == 2
    assert result["n_noncomparable_dates"] == 1
    assert result["n_comparable_pairs"] == 2
    assert result["ci_status"] == "non_estimable"
    assert result["ci_nonestimable_reason"] == "fewer_than_10_comparable_date_groups"
    assert result["n_bootstrap_attempts"] == result["n_bootstrap_valid"] == 0
    assert "date" not in str(result).lower() or "n_dates" in result


def test_within_date_concordance_uses_independent_rng_streams() -> None:
    """An OOF call cannot advance the deterministic test-view bootstrap stream."""
    dates = np.repeat([f"2025/1/{index}" for index in range(1, 11)], 2)
    y_true = np.tile([1, 0], 10)
    scores = np.linspace(0.0, 1.0, 20)
    _ = within_date_concordance(y_true, scores, dates, view="oof")
    after_oof = within_date_concordance(y_true, scores, dates, view="test")
    direct = within_date_concordance(y_true, scores, dates, view="test")
    assert after_oof["n_bootstrap_attempts"] == after_oof["n_bootstrap_valid"] == 2000
    assert after_oof["bootstrap_seed"] == 42
    assert after_oof["ci_method"] == "percentile_bootstrap_linear_quantile"
    assert after_oof["ci_low"] == direct["ci_low"]
    assert after_oof["ci_high"] == direct["ci_high"]


def test_within_date_concordance_no_comparable_dates_is_nonestimable() -> None:
    """No date with both classes retains the planned non-estimable result."""
    result = within_date_concordance(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.2, 0.8, 0.9]),
        np.array(["2025/1/1", "2025/1/1", "2025/1/2", "2025/1/2"]),
        view="test",
    )
    assert result["estimate"] is None
    assert result["estimability_status"] == "non_estimable"
    assert result["nonestimable_reason"] == "no_comparable_date_groups"


def _qc_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build three synthetic QC5 rows, including threshold equality."""
    diagnostic = pd.DataFrame(
        {
            "sample_id": [1, 2, 3],
            **{
                f"s{index}_saturation_fraction": [
                    0.003 if index == 1 else 0.0,
                    0.004 if index == 1 else 0.0,
                    0.004 if index <= 7 else 0.0,
                ]
                for index in range(1, 12)
            },
        }
    )
    summary = pd.DataFrame(
        {
            "sample_id": [1, 2, 3, 4],
            "qc1_env_sensor_fault_pass": [True, True, True, False],
            "qc5_sensor_saturation_pass": [True, True, False, pd.NA],
        }
    )
    return diagnostic, summary


def test_qc5_reconstruction_uses_lowercase_fields_and_strict_threshold() -> None:
    """QC5 treats equality as pass, reconstructs E005, and retains statuses."""
    diagnostic, summary = _qc_fixture()
    result = reconstruct_qc5(diagnostic, summary)
    assert result["n_sat"].tolist() == [0, 1, 7]
    assert result["strict_pass"].tolist() == [True, False, False]
    assert result["e005_pass"].tolist() == [True, True, False]
    assert result["n_qc5_evaluable"] == 3
    assert result["n_strict_fail"] == 2
    assert result["n_strict_pass"] == 1


def test_qc5_requires_qc1_and_e005_reconciliation() -> None:
    """Missing diagnostics and E005 disagreement fail closed."""
    diagnostic, summary = _qc_fixture()
    with pytest.raises(ValueError):
        reconstruct_qc5(diagnostic.iloc[:2], summary)
    summary.loc[1, "qc5_sensor_saturation_pass"] = False
    with pytest.raises(ValueError):
        reconstruct_qc5(diagnostic, summary)


def test_qc5_anchor_reconciliation_supports_974_298_676_logic() -> None:
    """Aggregate-only synthetic counts reproduce the required QC5 arithmetic."""
    diagnostic = pd.DataFrame(
        {
            "sample_id": np.arange(974),
            **{
                f"s{index}_saturation_fraction": np.r_[np.repeat(0.004 if index == 1 else 0.0, 298), np.zeros(676)]
                for index in range(1, 12)
            },
        }
    )
    summary = pd.DataFrame(
        {
            "sample_id": np.arange(974),
            "qc1_env_sensor_fault_pass": True,
            "qc5_sensor_saturation_pass": True,
        }
    )
    result = reconstruct_qc5(diagnostic, summary)
    assert (result["n_qc5_evaluable"], result["n_strict_fail"], result["n_strict_pass"]) == (974, 298, 676)


def test_qc5_correctness_and_composition_inventory_retains_empty_status_rows() -> None:
    """Each view has 19 planned QC5 records even when one strict status is empty."""
    rows = strict_qc5_correctness_inventory(
        y_true=np.array([0, 1]),
        scores=np.array([0.1, 0.9]),
        strict_pass=np.array([True, True]),
        disease_category=np.array([1, 3]),
        threshold=0.5,
        view="oof",
    )
    assert len(rows) == 19
    assert [row["record_type"] for row in rows].count("accuracy") == 2
    assert [row["record_type"] for row in rows].count("accuracy_difference") == 1
    assert [row["record_type"] for row in rows].count("outcome_composition") == 4
    assert [row["record_type"] for row in rows].count("disease_composition") == 12
    strict_fail_accuracy = next(
        row for row in rows if row["record_type"] == "accuracy" and row["strict_status"] == "strict_fail"
    )
    assert strict_fail_accuracy["denominator_n"] == 0
    assert strict_fail_accuracy["estimability_status"] == "non_estimable"
    assert strict_fail_accuracy["nonestimable_reason"] == "empty_strict_status_group"


def test_disease_category_codes_reject_lossy_float_and_string() -> None:
    """Disease codes require exact integer-equivalent values without lossy casts."""
    with pytest.raises(ValueError):
        _validate_disease_category_codes(np.array([1.5, 2.0]))
    with pytest.raises(ValueError):
        _validate_disease_category_codes(np.array(["1", "2"]))
    codes = _validate_disease_category_codes(np.array([1.0, 2.0, 3.0]))
    assert codes.tolist() == [1, 2, 3]


def test_require_bool_values_rejects_string_true_and_int_one() -> None:
    """Genuine booleans are required; ``"True"`` and ``1`` are rejected."""
    with pytest.raises(ValueError):
        _require_bool_values(np.array(["True", "False"]), "QC1 pass field")
    with pytest.raises(ValueError):
        _require_bool_values(np.array([1, 0]), "QC5 E005 pass field")
    validated = _require_bool_values(np.array([True, False]), "strict_pass")
    assert validated.tolist() == [True, False]


def test_strict_qc5_difference_emits_newcombe_when_both_groups_present() -> None:
    """The accuracy-difference row emits the Newcombe method when estimable."""
    rows = strict_qc5_correctness_inventory(
        y_true=np.array([0, 1, 0, 1]),
        scores=np.array([0.1, 0.9, 0.2, 0.8]),
        strict_pass=np.array([True, True, False, False]),
        disease_category=np.array([1, 3, 2, 4]),
        threshold=0.5,
        view="oof",
    )
    difference = next(row for row in rows if row["record_type"] == "accuracy_difference")
    assert difference["estimability_status"] == "estimable"
    assert difference["ci_method"] == "newcombe"
    assert difference["ci_status"] == "estimable"


def test_alignment_rejects_union_matching_length_but_wrong_cohort_ids() -> None:
    """A train/test union of the right size but wrong IDs is rejected."""
    lineage = pd.DataFrame(
        {
            "sample_id": [10, 11, 12, 13, 14, 15, 16],
            "label": [0, 0, 0, 1, 1, 1, 0],
            "in_final_output": [True, True, True, True, True, True, False],
        }
    )
    oof = pd.DataFrame({"sample_id": [10, 11, 12], "y_true": [0, 0, 0]})
    test = pd.DataFrame({"sample_id": [13, 14, 16], "y_true": [1, 1, 0]})
    with pytest.raises(ValueError):
        validate_locked_score_alignment(
            lineage=lineage,
            oof=oof,
            test=test,
            train_ids=np.array([10, 11, 12]),
            test_ids=np.array([13, 14, 16]),
        )


def test_alignment_validator_requires_numeric_unique_ordered_ids_and_split_contract() -> None:
    """Prediction, lineage, and split inputs require exact one-to-one membership."""
    lineage = _six_category_frame()
    oof = lineage.iloc[:3][["sample_id", "y_true"]].copy()
    test = lineage.iloc[3:][["sample_id", "y_true"]].copy()
    validation = validate_locked_score_alignment(
        lineage=lineage,
        oof=oof,
        test=test,
        train_ids=np.array([10, 11, 12]),
        test_ids=np.array([13, 14, 15]),
    )
    assert validation["one_to_one_alignment_status"] == "pass"
    assert validation["train_test_disjointness_status"] == "pass"
    with pytest.raises(ValueError):
        validate_locked_score_alignment(
            lineage=lineage,
            oof=oof.iloc[::-1],
            test=test,
            train_ids=np.array([10, 11, 12]),
            test_ids=np.array([13, 14, 15]),
        )
    with pytest.raises(ValueError):
        validate_locked_score_alignment(
            lineage=lineage,
            oof=oof.assign(y_true=[1, 0, 0]),
            test=test,
            train_ids=np.array([10, 11, 12]),
            test_ids=np.array([13, 14, 15]),
        )


def test_split_membership_maps_modeling_row_indices_to_sample_ids(tmp_path: Path) -> None:
    """Locked train/test arrays index the reset modeling cohort, not sample IDs."""
    lineage = pd.DataFrame(
        {
            "sample_id": [900, 20, 800, 10, 700],
            "in_final_output": [False, True, True, True, False],
        }
    )
    split_path = tmp_path / "split.npz"
    np.savez(split_path, train_idx=np.array([2, 0]), test_idx=np.array([1]))

    train_ids, test_ids = _split_membership(split_path, lineage)

    assert train_ids.tolist() == [10, 20]
    assert test_ids.tolist() == [800]


def test_preflight_nested_schema_orders_and_dependencies() -> None:
    """Synthetic preflight has exact key order, types, enums, and null dependency."""
    payload = build_score_blind_preflight(
        hashes=OrderedDict(
            [
                ("config_sha256", "a" * 64),
                ("methods_sha256", "b" * 64),
                ("freeze_manifest_sha256", "c" * 64),
                ("frozen_input_whole_container_sha256", OrderedDict([("fixture", "d" * 64)])),
            ]
        ),
        freeze_commit_validation=OrderedDict(
            [
                ("freeze_commit_id", "3c0ff6b3b40460a2c340ad3193222ee54979109d"),
                ("exists", True),
                ("is_ancestor_of_head", True),
                ("parent_is_baseline_head", True),
                ("changed_paths_exact", True),
                ("config_blob_matches", True),
                ("methods_blob_matches", True),
                ("manifest_blob_matches", True),
                ("approval_evidence_matches", True),
                ("active_rule_hashes_match", True),
            ]
        ),
        aggregate_counts=OrderedDict(
            [
                ("source_cohort_n", 1040),
                ("modeling_cohort_n", 972),
                ("train_n", 777),
                ("internal_test_n", 195),
                (
                    "difficult_spectrum_by_view",
                    OrderedDict(
                        [
                            (
                                view,
                                OrderedDict(
                                    [
                                        ("n", 4),
                                        ("n_positive", 2),
                                        ("n_negative", 2),
                                        (
                                            "disease_category_counts",
                                            OrderedDict(
                                                [("0", 0), ("1", 1), ("2", 1), ("3", 1), ("4", 1), ("5", 0)]
                                            ),
                                        ),
                                    ]
                                ),
                            )
                            for view in ("oof", "test")
                        ]
                    ),
                ),
                (
                    "date_feasibility",
                    OrderedDict([("n_dates_overall", 40), ("n_dates_with_both_endpoint_classes", 34)]),
                ),
                (
                    "qc5_reconciliation",
                    OrderedDict(
                        [
                            ("n_qc5_evaluable_source_rows", 974),
                            ("n_strict_fail_among_evaluable", 298),
                            ("n_strict_pass_among_evaluable", 676),
                        ]
                    ),
                ),
            ]
        ),
        frozen_hash_paths=["fixture"],
    )
    assert list(payload) == [
        "schema_version",
        "package_id",
        "generated_at_utc",
        "hashes",
        "freeze_commit_validation",
        "aggregate_counts",
        "one_to_one_alignment_status",
        "train_test_disjointness_status",
        "known_anchor_reconciliation",
        "status",
        "reason_code",
        "access_state",
    ]
    assert payload["status"] == "pass" and payload["reason_code"] is None
    assert list(payload["hashes"]) == [
        "config_sha256",
        "methods_sha256",
        "freeze_manifest_sha256",
        "frozen_input_whole_container_sha256",
    ]
    assert payload["access_state"]["session_activity"]["participant_level_score_values_read"] is False
    assert payload["known_anchor_reconciliation"] == OrderedDict(
        [("date_40_34", True), ("qc5_974_298_676", True), ("status", "pass")]
    )


def test_execute_framework_uses_synthetic_values_only() -> None:
    """Framework execution accepts explicit synthetic views without invoking a CLI."""
    inputs = {
        "y_true": np.array([0, 1, 0, 1]),
        "p_raw": np.array([0.1, 0.8, 0.2, 0.9]),
        "dates": np.array(["2025/1/1", "2025/1/1", "2025/1/2", "2025/1/2"]),
        "disease_category": np.array([1, 3, 2, 4]),
        "strict_pass": np.array([True, False, True, False]),
    }
    result = execute_approved_framework(
        {
            "oof": inputs,
            "test": inputs,
        }
    )
    assert len(result["frames"]["locked_score_metrics"]) == 22
    assert len(result["frames"]["qc5_correctness"]) == 38
    assert result["scores_read"] is True


def test_scientific_validator_rejects_planned_rows_in_wrong_order() -> None:
    """A valid scientific record inventory cannot be silently reordered."""
    inputs = {
        "y_true": np.array([0, 1, 0, 1]),
        "p_raw": np.array([0.1, 0.8, 0.2, 0.9]),
        "dates": np.array(["2025/1/1", "2025/1/1", "2025/1/2", "2025/1/2"]),
        "disease_category": np.array([1, 3, 2, 4]),
        "strict_pass": np.array([True, False, True, False]),
    }
    config = locked_scores.load_effective_posthoc_config(
        PROJECT_ROOT / "configs/post_hoc_final_supplementary.yaml"
    )
    frame = locked_scores._execution_frames(
        config,
        OrderedDict([("oof", inputs), ("test", inputs)]),
    )["locked_score_metrics"]
    records = frame.to_dict(orient="records")
    records[0], records[1] = records[1], records[0]

    with pytest.raises(ValueError, match="planned-row order"):
        locked_scores._validate_scientific_records(
            "locked_score_metrics",
            records,
            config["outputs"]["scientific_and_closure"]["locked_score_metrics"],
        )


def _synthetic_scientific_records() -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Create a complete valid scientific-record fixture entirely in memory."""
    inputs = {
        "y_true": np.array([0, 1, 0, 1]),
        "p_raw": np.array([0.1, 0.8, 0.2, 0.9]),
        "dates": np.array(["2025/1/1", "2025/1/1", "2025/1/2", "2025/1/2"]),
        "disease_category": np.array([1, 3, 2, 4]),
        "strict_pass": np.array([True, False, True, False]),
    }
    config = locked_scores.load_effective_posthoc_config(
        PROJECT_ROOT / "configs/post_hoc_final_supplementary.yaml"
    )
    frames = locked_scores._execution_frames(config, OrderedDict([("oof", inputs), ("test", inputs)]))
    return config, {name: frame.to_dict(orient="records") for name, frame in frames.items()}


def _synthetic_framework_views() -> OrderedDict[str, Mapping[str, np.ndarray]]:
    """Return two ordered synthetic views for direct framework-boundary tests."""
    inputs = {
        "y_true": np.array([0, 1, 0, 1]),
        "p_raw": np.array([0.1, 0.8, 0.2, 0.9]),
        "dates": np.array(["2025/1/1", "2025/1/1", "2025/1/2", "2025/1/2"]),
        "disease_category": np.array([1, 3, 2, 4]),
        "strict_pass": np.array([True, False, True, False]),
    }
    return OrderedDict([("oof", inputs), ("test", inputs)])


def test_framework_accepts_hc_and_agc_in_strict_qc5_composition() -> None:
    """Strict-QC5 composition retains endpoint-consistent HC and AGC rows."""
    inputs = {
        "y_true": np.array([0, 0, 0, 1, 1, 1]),
        "p_raw": np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]),
        "dates": np.array(
            ["2025/1/1", "2025/1/2", "2025/1/3", "2025/1/1", "2025/1/2", "2025/1/3"]
        ),
        "disease_category": np.array([0, 1, 2, 3, 4, 5]),
        "strict_pass": np.array([True, False, False, False, False, True]),
    }

    result = execute_approved_framework(OrderedDict([("oof", inputs), ("test", inputs)]))

    composition = result["frames"]["locked_score_composition"]
    strict_oof = composition[
        (composition["analysis_id"] == "analysis_4_strict_qc5") & (composition["view"] == "oof")
    ]
    assert strict_oof["disease_category_code"].tolist() == list(range(6))
    assert strict_oof["n"].tolist() == [1, 0, 0, 0, 0, 1]
    assert strict_oof["denominator_n"].tolist() == [2] * 6


def _record(records: list[dict[str, Any]], **expected: object) -> dict[str, Any]:
    """Return the unique synthetic record matching the supplied public fields."""
    matches = [
        record
        for record in records
        if all(record.get(field) == value for field, value in expected.items())
    ]
    assert len(matches) == 1
    return matches[0]


def _apply_field_mutation(record: dict[str, Any], updates: Mapping[str, Any]) -> None:
    """Mutate exactly the declared real fields in place, proving the target exists.

    Args:
        record: The record to mutate; every key in ``updates`` must already exist.
        updates: New values keyed by real, already-present field name.

    Raises:
        AssertionError: If a target field is absent, an extra/missing key results,
            or an unrelated field's value changes.
    """
    for field in updates:
        assert field in record, f"Mutation target field {field!r} does not exist in the record."
    original_keys = set(record)
    unrelated_before = {key: value for key, value in record.items() if key not in updates}
    record.update(updates)
    assert set(record) == original_keys, "Mutation introduced or removed a record key."
    for key, before_value in unrelated_before.items():
        after_value = record[key]
        same = before_value == after_value or (
            isinstance(before_value, float)
            and isinstance(after_value, float)
            and np.isnan(before_value)
            and np.isnan(after_value)
        )
        assert same, f"Mutation unexpectedly changed unrelated field {key!r}."


def test_qc5_composition_denominator_must_match_accuracy_denominator() -> None:
    """An individually valid composition denominator cannot differ from its accuracy group."""
    config, records_by_name = _synthetic_scientific_records()
    records = records_by_name["qc5_correctness"]
    accuracy = _record(records, record_type="accuracy", strict_status="strict_pass", view="oof")
    composition = _record(
        records,
        record_type="outcome_composition",
        strict_status="strict_pass",
        category="negative",
        view="oof",
    )
    mutated_denominator = composition["denominator_n"] + 1
    _apply_field_mutation(
        composition,
        {
            "denominator_n": mutated_denominator,
            "proportion": composition["numerator_n"] / mutated_denominator,
        },
    )
    assert composition["proportion"] == composition["numerator_n"] / composition["denominator_n"]
    assert composition["denominator_n"] != accuracy["denominator_n"]

    with pytest.raises(
        ValueError,
        match="QC5 composition denominators do not match the corresponding accuracy denominator",
    ):
        locked_scores._validate_scientific_records(
            "qc5_correctness",
            records,
            config["outputs"]["scientific_and_closure"]["qc5_correctness"],
        )


def test_qc5_zero_accuracy_denominator_governs_all_composition_rows() -> None:
    """An empty accuracy group requires matching zero/missing composition rows."""
    config, _ = _synthetic_scientific_records()
    threshold = config["threshold"]["threshold_p_raw"]
    records: list[dict[str, Any]] = []
    for view in ("oof", "test"):
        records.extend(
            strict_qc5_correctness_inventory(
                y_true=np.array([0, 1]),
                scores=np.array([0.1, 0.9]),
                strict_pass=np.array([True, True]),
                disease_category=np.array([1, 3]),
                threshold=threshold,
                view=view,
            )
        )
    for view in ("oof", "test"):
        accuracy = _record(records, record_type="accuracy", strict_status="strict_fail", view=view)
        composition = [
            record
            for record in records
            if record["view"] == view
            and record["strict_status"] == "strict_fail"
            and record["record_type"] in {"outcome_composition", "disease_composition"}
        ]
        assert accuracy["denominator_n"] == 0
        assert all(record["denominator_n"] == 0 for record in composition)
        assert all(record["numerator_n"] == 0 for record in composition)
        assert all(record["proportion"] is None for record in composition)
        assert all(record["estimability_status"] == "non_estimable" for record in composition)
        assert all(record["nonestimable_reason"] == "empty_strict_status_group" for record in composition)

    empty_accuracy = _record(records, record_type="accuracy", strict_status="strict_fail", view="oof")
    _apply_field_mutation(empty_accuracy, {"ci_low": pd.NA})
    locked_scores._validate_scientific_records(
        "qc5_correctness",
        records,
        config["outputs"]["scientific_and_closure"]["qc5_correctness"],
    )


@pytest.mark.parametrize("missing_estimate", [float("nan"), pd.NA], ids=["nan", "pd_NA"])
def test_sparse_within_date_missing_point_estimate_fails_closed(missing_estimate: object) -> None:
    """A sparse estimable row cannot use a recognized missing point representation."""
    config, records_by_name = _synthetic_scientific_records()
    records = records_by_name["within_date_concordance"]
    record = records[0]
    assert 1 <= record["n_date_groups_comparable"] < 10
    assert record["estimability_status"] == "estimable"
    assert record["ci_status"] == "non_estimable"
    _apply_field_mutation(record, {"estimate": missing_estimate})

    with pytest.raises(ValueError, match="Point status and its point-value field population disagree"):
        locked_scores._validate_scientific_records(
            "within_date_concordance",
            records,
            config["outputs"]["scientific_and_closure"]["within_date_concordance"],
        )


def test_estimable_ci_nan_field_fails_as_missing_required_ci() -> None:
    """NaN cannot masquerade as a populated field of an estimable CI."""
    config, _ = _synthetic_scientific_records()
    records = _ge_10_group_within_date_records()
    _apply_field_mutation(records[0], {"ci_low": float("nan")})

    with pytest.raises(ValueError, match="Estimable CI records require all CI fields"):
        locked_scores._validate_scientific_records(
            "within_date_concordance",
            records,
            config["outputs"]["scientific_and_closure"]["within_date_concordance"],
        )


@pytest.mark.parametrize(
    ("reason_field", "expected_message"),
    [
        ("nonestimable_reason", "Non-estimable point records require a reason"),
        ("ci_nonestimable_reason", "CI status and reason do not reconcile"),
    ],
)
def test_nonestimable_required_reason_pd_na_fails_closed(
    reason_field: str,
    expected_message: str,
) -> None:
    """pd.NA cannot satisfy a required point or CI non-estimability reason."""
    config, _ = _synthetic_scientific_records()
    records = _zero_comparable_within_date_records()
    _apply_field_mutation(records[0], {reason_field: pd.NA})

    with pytest.raises(ValueError, match=expected_message):
        locked_scores._validate_scientific_records(
            "within_date_concordance",
            records,
            config["outputs"]["scientific_and_closure"]["within_date_concordance"],
        )


@pytest.mark.parametrize("reason_field", ["nonestimable_reason", "ci_nonestimable_reason"])
def test_estimable_nullable_reason_pd_na_is_treated_as_missing(reason_field: str) -> None:
    """pd.NA is permitted wherever an estimable point or CI requires a missing reason."""
    config, records_by_name = _synthetic_scientific_records()
    if reason_field == "nonestimable_reason":
        records = records_by_name["within_date_concordance"]
        assert records[0]["estimability_status"] == "estimable"
    else:
        records = _ge_10_group_within_date_records()
        assert records[0]["ci_status"] == "estimable"
    _apply_field_mutation(records[0], {reason_field: pd.NA})

    locked_scores._validate_scientific_records(
        "within_date_concordance",
        records,
        config["outputs"]["scientific_and_closure"]["within_date_concordance"],
    )


def test_framework_callback_fires_once_per_successful_computation() -> None:
    """Each complete framework call notifies its external callback exactly once."""
    config, _ = _synthetic_scientific_records()
    callback_count = 0

    def count_callback() -> None:
        nonlocal callback_count
        callback_count += 1

    execute_approved_framework(_synthetic_framework_views(), config=config, on_first_aggregate=count_callback)
    assert callback_count == 1
    execute_approved_framework(_synthetic_framework_views(), config=config, on_first_aggregate=count_callback)
    assert callback_count == 2


def test_framework_callback_does_not_fire_before_first_aggregate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failure inside the first AUROC helper leaves the callback untouched."""
    config, _ = _synthetic_scientific_records()
    callback_count = 0

    def count_callback() -> None:
        nonlocal callback_count
        callback_count += 1

    monkeypatch.setattr(
        "src.analysis.posthoc_final_locked_scores.single_auroc_delong_ci",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("synthetic pre-aggregate failure")),
    )
    with pytest.raises(ValueError, match="synthetic pre-aggregate failure"):
        execute_approved_framework(_synthetic_framework_views(), config=config, on_first_aggregate=count_callback)
    assert callback_count == 0


def test_framework_callback_fires_once_when_later_computation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-AUROC failure retains one and only one external notification."""
    config, _ = _synthetic_scientific_records()
    callback_count = 0

    def count_callback() -> None:
        nonlocal callback_count
        callback_count += 1

    monkeypatch.setattr(
        "src.analysis.posthoc_final_locked_scores.strict_qc5_correctness_inventory",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("synthetic post-aggregate failure")),
    )
    with pytest.raises(ValueError, match="synthetic post-aggregate failure"):
        execute_approved_framework(_synthetic_framework_views(), config=config, on_first_aggregate=count_callback)
    assert callback_count == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "missing",
        "extra",
        "wrong_identity",
        "invalid_enum",
        "nan",
        "positive_infinity",
        "negative_infinity",
        "non_boolean",
        "required_null",
        "forbidden_populated",
        "estimable_missing_estimate",
        "nonestimable_populated_estimate",
        "missing_reason",
        "inverted_ci",
        "estimate_outside_ci",
        "wrong_threshold",
        "wrong_score_scale",
        "wrong_ci_contract",
        "wrong_ci_level",
        "class_total_mismatch",
        "confusion_mismatch",
        "sensitivity_mismatch",
        "lr_correction_mismatch",
        "wrong_sensitivity_estimate",
        "wrong_specificity_estimate",
        "wrong_companion_specificity_on_sensitivity_row",
        "wrong_companion_sensitivity_on_specificity_row",
        "wrong_lr_sensitivity_unadjusted",
        "wrong_lr_specificity_unadjusted",
        "wrong_wilson_bound_ordered_contains_estimate",
        "wrong_wilson_method",
        "wrong_wilson_level",
    ],
)
def test_locked_metric_semantic_mutations_fail_closed(mutation: str) -> None:
    """Each isolated locked-metric semantic mutation is rejected."""
    config, records_by_name = _synthetic_scientific_records()
    records = records_by_name["locked_score_metrics"]
    sensitivity = _record(records, analysis_id="analysis_1_difficult_spectrum", metric="sensitivity", view="oof")
    specificity = _record(records, analysis_id="analysis_1_difficult_spectrum", metric="specificity", view="oof")
    lr_positive = _record(records, analysis_id="analysis_1_difficult_spectrum", metric="lr_positive", view="oof")
    auc = _record(records, analysis_id="analysis_1_difficult_spectrum", metric="auc_roc", view="oof")

    target: dict[str, Any] | None = None
    updates: dict[str, Any] = {}

    if mutation == "duplicate":
        records[1] = dict(records[0])
    elif mutation == "missing":
        records.pop()
    elif mutation == "extra":
        records.append(dict(records[-1]))
    elif mutation == "wrong_identity":
        records[0]["metric"] = "unplanned_metric"
    elif mutation == "invalid_enum":
        target, updates = sensitivity, {"estimability_status": "invalid"}
    elif mutation == "nan":
        target, updates = sensitivity, {"estimate": float("nan")}
    elif mutation == "positive_infinity":
        target, updates = sensitivity, {"estimate": float("inf")}
    elif mutation == "negative_infinity":
        target, updates = sensitivity, {"estimate": float("-inf")}
    elif mutation == "non_boolean":
        target, updates = lr_positive, {"correction_applied": 1}
    elif mutation == "required_null":
        target, updates = sensitivity, {"n": None}
    elif mutation == "forbidden_populated":
        target, updates = auc, {"tp": 0}
    elif mutation == "estimable_missing_estimate":
        target, updates = sensitivity, {"estimate": None}
    elif mutation == "nonestimable_populated_estimate":
        target, updates = sensitivity, {
            "estimability_status": "non_estimable",
            "ci_status": "non_estimable",
            "nonestimable_reason": "zero_denominator",
        }
    elif mutation == "missing_reason":
        target, updates = sensitivity, {
            "estimability_status": "non_estimable",
            "ci_status": "non_estimable",
            "estimate": None,
            "nonestimable_reason": None,
        }
    elif mutation == "inverted_ci":
        target, updates = sensitivity, {"ci_low": sensitivity["ci_high"], "ci_high": sensitivity["ci_low"]}
    elif mutation == "estimate_outside_ci":
        target, updates = sensitivity, {"ci_low": sensitivity["estimate"] + 0.01}
    elif mutation == "wrong_threshold":
        target, updates = sensitivity, {"threshold_p_raw": 0.123}
    elif mutation == "wrong_score_scale":
        target, updates = sensitivity, {"score_scale": "calibrated"}
    elif mutation == "wrong_ci_contract":
        target, updates = sensitivity, {"ci_method": "wrong_method"}
    elif mutation == "wrong_ci_level":
        target, updates = sensitivity, {"ci_level": 0.5}
    elif mutation == "class_total_mismatch":
        target, updates = sensitivity, {"n_positive": sensitivity["n_positive"] + 1}
    elif mutation == "confusion_mismatch":
        target, updates = sensitivity, {"tp": sensitivity["tp"] - 1}
    elif mutation == "sensitivity_mismatch":
        target, updates = sensitivity, {"sensitivity_unadjusted": 0.123}
    elif mutation == "lr_correction_mismatch":
        target, updates = lr_positive, {"zero_cell_correction": 0.5}
    elif mutation == "wrong_sensitivity_estimate":
        target, updates = sensitivity, {"estimate": 0.01}
    elif mutation == "wrong_specificity_estimate":
        target, updates = specificity, {"estimate": 0.01}
    elif mutation == "wrong_companion_specificity_on_sensitivity_row":
        # Own metric untouched; only its companion specificity is wrong.
        target, updates = sensitivity, {"specificity_unadjusted": 0.01}
    elif mutation == "wrong_companion_sensitivity_on_specificity_row":
        target, updates = specificity, {"sensitivity_unadjusted": 0.01}
    elif mutation == "wrong_lr_sensitivity_unadjusted":
        # LR rows carry the same companion pair; this was previously unchecked.
        target, updates = lr_positive, {"sensitivity_unadjusted": 0.01}
    elif mutation == "wrong_lr_specificity_unadjusted":
        target, updates = lr_positive, {"specificity_unadjusted": 0.01}
    elif mutation == "wrong_wilson_bound_ordered_contains_estimate":
        # Halving keeps ci_low <= ci_high and ci_low <= estimate, so only the
        # Wilson-recomputation check -- not ordering/containment -- can reject it.
        target, updates = sensitivity, {"ci_low": sensitivity["ci_low"] / 2}
    elif mutation == "wrong_wilson_method":
        target, updates = sensitivity, {"ci_method": "normal_approximation"}
    elif mutation == "wrong_wilson_level":
        target, updates = sensitivity, {"ci_level": 0.90}
    else:
        raise AssertionError(f"Unknown mutation: {mutation}")

    if target is not None:
        _apply_field_mutation(target, updates)
        if mutation == "wrong_wilson_bound_ordered_contains_estimate":
            assert sensitivity["ci_low"] <= sensitivity["ci_high"]
            assert sensitivity["ci_low"] <= sensitivity["estimate"] <= sensitivity["ci_high"]

    with pytest.raises(ValueError):
        locked_scores._validate_scientific_records(
            "locked_score_metrics",
            records,
            config["outputs"]["scientific_and_closure"]["locked_score_metrics"],
        )


@pytest.mark.parametrize(
    "mutation",
    ["proportion_mismatch", "disease_total_mismatch", "wrong_order", "unplanned_disease_category_code"],
)
def test_composition_semantic_mutations_fail_closed(mutation: str) -> None:
    """Composition count and disease-contract mutations are rejected."""
    config, records_by_name = _synthetic_scientific_records()
    records = records_by_name["locked_score_composition"]
    bgl = _record(
        records,
        analysis_id="analysis_1_difficult_spectrum",
        disease_category_label="BGL",
        view="oof",
    )

    if mutation == "proportion_mismatch":
        _apply_field_mutation(bgl, {"proportion": 0.123})
    elif mutation == "disease_total_mismatch":
        _apply_field_mutation(bgl, {"n": bgl["n"] + 1})
    elif mutation == "wrong_order":
        # A row-order swap within one planned group; the key-preservation guard
        # does not apply because no field value is mutated, only row position.
        records[0], records[1] = records[1], records[0]
    else:
        _apply_field_mutation(bgl, {"disease_category_code": 99})

    with pytest.raises(ValueError):
        locked_scores._validate_scientific_records(
            "locked_score_composition",
            records,
            config["outputs"]["scientific_and_closure"]["locked_score_composition"],
        )


def test_composition_negative_to_positive_transfer_violates_class_reconciliation() -> None:
    """A denominator-preserving, proportion-consistent disease transfer must still
    fail cross-output reconciliation because the endpoint-class totals change."""
    config, records_by_name = _synthetic_scientific_records()
    metrics = records_by_name["locked_score_metrics"]
    composition = records_by_name["locked_score_composition"]
    qc_rows = records_by_name["qc5_correctness"]
    # Sanity: the unmutated inventories reconcile before the transfer.
    locked_scores._validate_cross_output_reconciliation(metrics, composition, qc_rows)

    bgl = _record(composition, analysis_id="analysis_1_difficult_spectrum", disease_category_label="BGL", view="oof")
    hgin = _record(composition, analysis_id="analysis_1_difficult_spectrum", disease_category_label="HGIN", view="oof")
    hc = _record(composition, analysis_id="analysis_1_difficult_spectrum", disease_category_label="HC", view="oof")
    agc = _record(composition, analysis_id="analysis_1_difficult_spectrum", disease_category_label="AGC", view="oof")
    denominator = bgl["denominator_n"]
    assert bgl["n"] > 0

    _apply_field_mutation(bgl, {"n": bgl["n"] - 1, "proportion": (bgl["n"] - 1) / denominator})
    _apply_field_mutation(hgin, {"n": hgin["n"] + 1, "proportion": (hgin["n"] + 1) / denominator})

    group = [
        row for row in composition
        if row["analysis_id"] == "analysis_1_difficult_spectrum" and row["view"] == "oof"
    ]
    # The grand total and the HC/AGC exclusions are preserved by construction.
    assert sum(row["n"] for row in group) == denominator
    assert hc["n"] == 0 and agc["n"] == 0

    with pytest.raises(ValueError, match="class"):
        locked_scores._validate_cross_output_reconciliation(metrics, composition, qc_rows)


def test_composition_disagrees_with_metric_class_totals() -> None:
    """Composition must match its metric group even when every metric row in the
    group remains mutually consistent with its siblings."""
    config, records_by_name = _synthetic_scientific_records()
    metrics = records_by_name["locked_score_metrics"]
    composition = records_by_name["locked_score_composition"]
    qc_rows = records_by_name["qc5_correctness"]
    locked_scores._validate_cross_output_reconciliation(metrics, composition, qc_rows)

    group = [
        record for record in metrics
        if record["analysis_id"] == "analysis_1_difficult_spectrum"
        and record["view"] == "oof"
        and record["population_id"] == "difficult_spectrum"
    ]
    assert group
    for record in group:
        _apply_field_mutation(record, {"n_positive": record["n_positive"] + 1, "n_negative": record["n_negative"] - 1})

    with pytest.raises(ValueError, match="class"):
        locked_scores._validate_cross_output_reconciliation(metrics, composition, qc_rows)


def test_strict_pass_composition_disagrees_with_qc5_disease_rows() -> None:
    """Locked-score strict-pass disease composition must match QC5's own
    strict-pass disease-composition rows for the same view."""
    config, records_by_name = _synthetic_scientific_records()
    metrics = records_by_name["locked_score_metrics"]
    composition = records_by_name["locked_score_composition"]
    qc_rows = records_by_name["qc5_correctness"]
    locked_scores._validate_cross_output_reconciliation(metrics, composition, qc_rows)

    qc5_bgl = _record(
        qc_rows, record_type="disease_composition", strict_status="strict_pass", category="BGL", view="oof"
    )
    _apply_field_mutation(qc5_bgl, {"numerator_n": qc5_bgl["numerator_n"] + 1})

    with pytest.raises(ValueError, match="strict-pass"):
        locked_scores._validate_cross_output_reconciliation(metrics, composition, qc_rows)


@pytest.mark.parametrize(
    "mutation",
    [
        "accuracy_mismatch",
        "difference_mismatch",
        "record_type_population_mismatch",
        "wrong_accuracy_threshold",
        "wrong_difference_threshold",
        "wrong_accuracy_wilson_bound_ordered",
        "wrong_accuracy_wilson_method",
        "wrong_accuracy_wilson_level",
        "denominator_zero_accuracy_invalid_state",
        "wrong_difference_newcombe_bound_ordered",
        "wrong_difference_newcombe_method",
        "wrong_difference_newcombe_level",
        "empty_paired_group_invalid_difference",
        "composition_populated_threshold",
        "composition_populated_score_scale",
        "composition_zero_denominator_wrong_status",
        "disease_outcome_transfer_violates_reconciliation",
    ],
)
def test_qc5_semantic_mutations_fail_closed(mutation: str) -> None:
    """QC5 per-record-type arithmetic and field dependencies are rejected."""
    config, records_by_name = _synthetic_scientific_records()
    records = records_by_name["qc5_correctness"]
    accuracy = _record(records, record_type="accuracy", strict_status="strict_pass", view="oof")
    fail_accuracy = _record(records, record_type="accuracy", strict_status="strict_fail", view="oof")
    difference = _record(records, record_type="accuracy_difference", view="oof")
    outcome_negative = _record(
        records, record_type="outcome_composition", strict_status="strict_pass", category="negative", view="oof"
    )

    target: dict[str, Any] | None = None
    updates: dict[str, Any] = {}

    if mutation == "accuracy_mismatch":
        target, updates = accuracy, {"estimate": 0.123}
    elif mutation == "difference_mismatch":
        target, updates = difference, {"estimate": 0.123}
    elif mutation == "record_type_population_mismatch":
        target, updates = accuracy, {"strict_status": "not_applicable"}
    elif mutation == "wrong_accuracy_threshold":
        target, updates = accuracy, {"threshold_p_raw": 0.123}
    elif mutation == "wrong_difference_threshold":
        target, updates = difference, {"threshold_p_raw": 0.123}
    elif mutation == "wrong_accuracy_wilson_bound_ordered":
        # Halving the real Wilson lower bound keeps it ordered and containing the
        # estimate, so only the Wilson-recomputation check can reject it.
        target, updates = accuracy, {"ci_low": accuracy["ci_low"] / 2}
    elif mutation == "wrong_accuracy_wilson_method":
        target, updates = accuracy, {"ci_method": "normal_approximation"}
    elif mutation == "wrong_accuracy_wilson_level":
        target, updates = accuracy, {"ci_level": 0.90}
    elif mutation == "denominator_zero_accuracy_invalid_state":
        # Denominator/correct/incorrect are honestly emptied, but estimate and
        # status are wrongly left populated as if the group were still estimable.
        target, updates = accuracy, {"denominator_n": 0, "correct_n": 0, "incorrect_n": 0}
    elif mutation == "wrong_difference_newcombe_bound_ordered":
        target, updates = difference, {"ci_low": difference["ci_low"] - 0.05}
    elif mutation == "wrong_difference_newcombe_method":
        target, updates = difference, {"ci_method": "wilson"}
    elif mutation == "wrong_difference_newcombe_level":
        target, updates = difference, {"ci_level": 0.90}
    elif mutation == "empty_paired_group_invalid_difference":
        # The strict-fail group is honestly emptied, but the paired difference
        # row is wrongly left populated/estimable instead of following suit.
        _apply_field_mutation(
            fail_accuracy,
            {
                "denominator_n": 0,
                "correct_n": 0,
                "incorrect_n": 0,
                "estimate": None,
                "ci_low": None,
                "ci_high": None,
                "ci_level": None,
                "ci_method": None,
                "estimability_status": "non_estimable",
                "nonestimable_reason": "empty_strict_status_group",
                "ci_status": "non_estimable",
                "ci_nonestimable_reason": "empty_strict_status_group",
            },
        )
    elif mutation == "composition_populated_threshold":
        target, updates = outcome_negative, {"threshold_p_raw": 0.5}
    elif mutation == "composition_populated_score_scale":
        target, updates = outcome_negative, {"score_scale": "p_raw"}
    elif mutation == "composition_zero_denominator_wrong_status":
        # Numerator/denominator/proportion are honestly emptied, but the point
        # status is wrongly left "estimable" for the now-empty strict group.
        target, updates = outcome_negative, {"numerator_n": 0, "denominator_n": 0, "proportion": None}
    elif mutation == "disease_outcome_transfer_violates_reconciliation":
        bgl = _record(
            records, record_type="disease_composition", strict_status="strict_pass", category="BGL", view="oof"
        )
        hgin = _record(
            records, record_type="disease_composition", strict_status="strict_pass", category="HGIN", view="oof"
        )
        denominator = bgl["denominator_n"]
        assert bgl["numerator_n"] > 0
        _apply_field_mutation(
            bgl, {"numerator_n": bgl["numerator_n"] - 1, "proportion": (bgl["numerator_n"] - 1) / denominator}
        )
        _apply_field_mutation(
            hgin, {"numerator_n": hgin["numerator_n"] + 1, "proportion": (hgin["numerator_n"] + 1) / denominator}
        )
    else:
        raise AssertionError(f"Unknown mutation: {mutation}")

    if target is not None:
        _apply_field_mutation(target, updates)
        if mutation in {"wrong_accuracy_wilson_bound_ordered", "wrong_difference_newcombe_bound_ordered"}:
            assert target["ci_low"] <= target["ci_high"]
            assert target["ci_low"] <= target["estimate"] <= target["ci_high"]

    with pytest.raises(ValueError):
        locked_scores._validate_scientific_records(
            "qc5_correctness",
            records,
            config["outputs"]["scientific_and_closure"]["qc5_correctness"],
        )


def _ge_10_group_within_date_records() -> list[dict[str, Any]]:
    """Build a valid two-view within-date record fixture with >=10 comparable groups.

    Eight dates are concordant (positive score above negative) and two are
    discordant, so cluster-bootstrap resampling yields a non-degenerate
    distribution (``ci_low < estimate < ci_high``) rather than a zero-width CI.
    """
    dates = np.repeat([f"2025/1/{index}" for index in range(1, 11)], 2)
    y_true = np.tile([1, 0], 10)
    p_raw = np.array([0.9, 0.1] * 8 + [0.1, 0.9] * 2)
    inputs = {"y_true": y_true, "p_raw": p_raw, "dates": dates}
    return [locked_scores._within_date_record(view, inputs) for view in ("oof", "test")]


def _zero_comparable_within_date_records() -> list[dict[str, Any]]:
    """Build a valid two-view within-date record fixture with zero comparable dates."""
    inputs = {
        "y_true": np.array([0, 0, 1, 1]),
        "p_raw": np.array([0.1, 0.2, 0.8, 0.9]),
        "dates": np.array(["2025/1/1", "2025/1/1", "2025/1/2", "2025/1/2"]),
    }
    return [locked_scores._within_date_record(view, inputs) for view in ("oof", "test")]


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_bootstrap_attempts_field",
        "inverted_ci",
        "estimate_outside_ordered_ci",
        "wrong_bootstrap_method",
        "wrong_bootstrap_level",
        "wrong_bootstrap_seed",
        "wrong_bootstrap_valid_count",
        "zero_pairs_contradictory_group_count",
        "sparse_group_status_mismatch",
        "no_group_reason_mismatch",
    ],
)
def test_within_date_real_field_mutations_fail_closed(mutation: str) -> None:
    """Field-real mutations to within-date records are rejected at their real branch."""
    config, default_records_by_name = _synthetic_scientific_records()
    contract = config["outputs"]["scientific_and_closure"]["within_date_concordance"]

    if mutation == "sparse_group_status_mismatch":
        # The default fixture yields 2 comparable dates (1 <= G < 10).
        records = default_records_by_name["within_date_concordance"]
        record = records[0]
        assert 1 <= record["n_date_groups_comparable"] < 10
        target, updates = record, {"estimability_status": "non_estimable"}
    elif mutation in {"zero_pairs_contradictory_group_count", "no_group_reason_mismatch"}:
        records = _zero_comparable_within_date_records()
        record = records[0]
        assert record["n_comparable_pairs"] == 0
        if mutation == "zero_pairs_contradictory_group_count":
            # Total is kept reconciled (comparable + noncomparable unchanged) so
            # only the comparable-group/pair zero-ness agreement check can fire.
            target, updates = record, {
                "n_date_groups_comparable": record["n_date_groups_comparable"] + 1,
                "n_date_groups_noncomparable": record["n_date_groups_noncomparable"] - 1,
            }
        else:
            target, updates = record, {"nonestimable_reason": "fewer_than_10_comparable_date_groups"}
    else:
        records = _ge_10_group_within_date_records()
        record = records[0]
        assert record["ci_status"] == "estimable"
        if mutation == "wrong_bootstrap_attempts_field":
            target, updates = record, {"n_bootstrap_attempts": record["n_bootstrap_attempts"] - 1}
        elif mutation == "inverted_ci":
            target, updates = record, {"ci_low": record["ci_high"], "ci_high": record["ci_low"]}
        elif mutation == "estimate_outside_ordered_ci":
            # Collapsing the upper bound to the original lower bound keeps the
            # interval ordered (low == high) while excluding the point estimate.
            target, updates = record, {"ci_high": record["ci_low"]}
        elif mutation == "wrong_bootstrap_method":
            target, updates = record, {"ci_method": "wrong_method"}
        elif mutation == "wrong_bootstrap_level":
            target, updates = record, {"ci_level": 0.90}
        elif mutation == "wrong_bootstrap_seed":
            target, updates = record, {"bootstrap_seed": 7}
        elif mutation == "wrong_bootstrap_valid_count":
            target, updates = record, {"n_bootstrap_valid": record["n_bootstrap_valid"] - 1}
        else:
            raise AssertionError(f"Unknown mutation: {mutation}")

    _apply_field_mutation(target, updates)
    if mutation == "estimate_outside_ordered_ci":
        assert target["ci_low"] <= target["ci_high"]
        assert not (target["ci_low"] <= target["estimate"] <= target["ci_high"])

    with pytest.raises(ValueError):
        locked_scores._validate_scientific_records("within_date_concordance", records, contract)


def test_cli_requires_explicit_mode_and_freeze_commit() -> None:
    """Score access never occurs by default and modes are mutually exclusive."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--config", "fixture.yaml"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--config",
                "fixture.yaml",
                "--freeze-commit",
                "not-a-commit",
                "--amendment-freeze-commit",
                "2e08e8fb89d99bb62cca2e5204a13ff09e47d432",
                "--preflight-score-blind",
            ]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--config",
                "fixture.yaml",
                "--freeze-commit",
                "3c0ff6b3b40460a2c340ad3193222ee54979109d",
                "--amendment-freeze-commit",
                "2e08e8fb89d99bb62cca2e5204a13ff09e47d432",
                "--preflight-score-blind",
                "--execute-approved",
            ]
        )
    args: argparse.Namespace = parser.parse_args(
        [
            "--config",
            "fixture.yaml",
            "--freeze-commit",
            "3c0ff6b3b40460a2c340ad3193222ee54979109d",
            "--amendment-freeze-commit",
            "2e08e8fb89d99bb62cca2e5204a13ff09e47d432",
            "--preflight-score-blind",
        ]
    )
    assert args.preflight_score_blind is True and args.execute_approved is False


def test_session_one_declares_exact_session_four_outputs_and_completion_last() -> None:
    """The repaired framework owns exactly the configured Session 4 output contract."""
    from src.analysis.posthoc_final_locked_scores import (
        SESSION_4_COMPLETION_LOG_PATH,
        SESSION_4_SCIENTIFIC_OUTPUT_PATHS,
    )

    source = Path("src/analysis/posthoc_final_locked_scores.py").read_text(encoding="utf-8")
    script = Path("scripts/run_posthoc_final_locked_scores.py").read_text(encoding="utf-8")
    assert SESSION_4_SCIENTIFIC_OUTPUT_PATHS == (
        "outputs/analysis/posthoc_final_locked_score_metrics_minirocket_ridge.csv",
        "outputs/analysis/posthoc_final_locked_score_composition_minirocket_ridge.csv",
        "outputs/analysis/posthoc_final_qc5_correctness_minirocket_ridge.csv",
        "outputs/analysis/posthoc_final_within_date_concordance_minirocket_ridge.csv",
    )
    assert SESSION_4_COMPLETION_LOG_PATH == "outputs/logs/posthoc_final_locked_scores.log"
    for target in (*SESSION_4_SCIENTIFIC_OUTPUT_PATHS, SESSION_4_COMPLETION_LOG_PATH):
        assert target in source
    assert "def execute_approved(" in source
    assert "def _write_atomic_csv(" in source
    assert "def _stage_csv(" in source
    assert "os.replace(log_temporary, log_path)" in source
    # Every scientific CSV is staged and promoted before the completion log is promoted last.
    assert source.index("for name, frame in first.items()") < source.index("os.replace(log_temporary, log_path)")
    for name, promotion in (
        ("os.replace(temporary, destination)", "os.replace(log_temporary, log_path)"),
    ):
        assert source.index(name) < source.index(promotion)
    assert "--execute-approved" in script
    assert "--amendment-freeze-commit" in script
    assert "--confirm-reviewed-session1-corrections-and-tests" in script


def test_conditional_retry_gate_requires_all_frozen_conditions() -> None:
    """A second preflight remains unauthorized unless every amendment gate is true."""
    assert (
        evaluate_conditional_preflight_retry_authorization(
            approval_evidence_present=True,
            amendment_freeze_valid=True,
            corrections_implemented=SESSION_1_CORRECTIONS_IMPLEMENTED,
            synthetic_tests_passed=True,
            other_gates_satisfied=True,
        )
        is True
    )
    assert (
        evaluate_conditional_preflight_retry_authorization(
            approval_evidence_present=True,
            amendment_freeze_valid=True,
            corrections_implemented=SESSION_1_CORRECTIONS_IMPLEMENTED,
            synthetic_tests_passed=False,
            other_gates_satisfied=True,
        )
        is False
    )


def test_cli_preflight_defaults_block_retry_without_synthetic_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI score-blind mode never invents synthetic-test evidence."""
    captured: dict[str, Any] = {}

    def fake_preflight(
        config: Path,
        freeze_commit: str,
        amendment_freeze_commit: str,
        *,
        synthetic_tests_passed: bool = False,
        git_runner: Any = None,
    ) -> Path:
        captured["synthetic_tests_passed"] = synthetic_tests_passed
        captured["freeze_commit"] = freeze_commit
        captured["amendment_freeze_commit"] = amendment_freeze_commit
        return Path("outputs/analysis/posthoc_final_score_blind_preflight.json")

    monkeypatch.setattr("scripts.run_posthoc_final_locked_scores.run_score_blind_preflight", fake_preflight)
    code = main(
        [
            "--config",
            "configs/post_hoc_final_supplementary.yaml",
            "--freeze-commit",
            SESSION0_FREEZE,
            "--amendment-freeze-commit",
            AMENDMENT_FREEZE_DEFAULT_FOR_TESTS,
            "--preflight-score-blind",
        ]
    )
    assert code == 0
    assert captured["synthetic_tests_passed"] is False
    assert captured["freeze_commit"] == SESSION0_FREEZE
    assert captured["amendment_freeze_commit"] == AMENDMENT_FREEZE_DEFAULT_FOR_TESTS


def _write_synthetic_consolidated_config(tmp_path: Path) -> Path:
    """Write a minimal current config with the approved parser embedded directly."""
    base_path = tmp_path / "configs/post_hoc_final_supplementary.yaml"
    base_path.parent.mkdir(parents=True, exist_ok=True)
    base = {"loaders": {"date_parsing": dict(locked_scores._APPROVED_DATE_PARSING_PAYLOAD_EXACT)}}
    base_path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    return base_path


def test_effective_config_accepts_project_root_relative_base_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The approved project-root-relative base path resolves to the frozen file."""
    _write_synthetic_consolidated_config(tmp_path)
    monkeypatch.setattr("src.analysis.posthoc_final_locked_scores.PROJECT_ROOT", tmp_path)

    effective = locked_scores.load_effective_posthoc_config(Path("configs/post_hoc_final_supplementary.yaml"))

    assert effective["loaders"]["date_parsing"] == dict(locked_scores._APPROVED_DATE_PARSING_PAYLOAD_EXACT)


def test_effective_config_accepts_canonical_absolute_base_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The canonical absolute base path resolves to the same frozen file."""
    base_path = _write_synthetic_consolidated_config(tmp_path)
    monkeypatch.setattr("src.analysis.posthoc_final_locked_scores.PROJECT_ROOT", tmp_path)

    effective = locked_scores.load_effective_posthoc_config(base_path.resolve())

    assert effective["loaders"]["date_parsing"] == dict(locked_scores._APPROVED_DATE_PARSING_PAYLOAD_EXACT)


@pytest.mark.parametrize(
    "candidate",
    [
        Path("configs/wrong.yaml"),
        Path("../outside/post_hoc_final_supplementary.yaml"),
    ],
)
def test_effective_config_rejects_wrong_relative_and_traversal_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate: Path,
) -> None:
    """Existing wrong relative files and traversal outside the project fail closed."""
    base_path = _write_synthetic_consolidated_config(tmp_path)
    wrong_path = (tmp_path / candidate).resolve()
    wrong_path.parent.mkdir(parents=True, exist_ok=True)
    wrong_path.write_bytes(base_path.read_bytes())
    monkeypatch.setattr("src.analysis.posthoc_final_locked_scores.PROJECT_ROOT", tmp_path)

    with pytest.raises(ValueError, match="current configuration path is immutable"):
        locked_scores.load_effective_posthoc_config(candidate)


def test_effective_config_rejects_same_named_file_elsewhere(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid-looking same-named file outside the fixed directory is not accepted."""
    base_path = _write_synthetic_consolidated_config(tmp_path)
    lookalike = tmp_path / "lookalike/post_hoc_final_supplementary.yaml"
    lookalike.parent.mkdir(parents=True)
    lookalike.write_bytes(base_path.read_bytes())
    monkeypatch.setattr("src.analysis.posthoc_final_locked_scores.PROJECT_ROOT", tmp_path)

    with pytest.raises(ValueError, match="current configuration path is immutable"):
        locked_scores.load_effective_posthoc_config(lookalike)


def test_effective_config_missing_path_fails_closed_without_echo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing candidate produces a concise error without exposing its value."""
    _write_synthetic_consolidated_config(tmp_path)
    missing = Path("configs/private-missing-name.yaml")
    monkeypatch.setattr("src.analysis.posthoc_final_locked_scores.PROJECT_ROOT", tmp_path)

    with pytest.raises(ValueError) as excinfo:
        locked_scores.load_effective_posthoc_config(missing)

    assert "could not be resolved" in str(excinfo.value)
    assert str(missing) not in str(excinfo.value)
    assert excinfo.value.__cause__ is None


def test_consolidated_config_requires_no_live_amendment_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact approved date contract loads after every live Amendment file is absent."""
    base_path = _write_synthetic_consolidated_config(tmp_path)
    monkeypatch.setattr("src.analysis.posthoc_final_locked_scores.PROJECT_ROOT", tmp_path)

    effective = locked_scores.load_effective_posthoc_config(base_path)

    assert effective["loaders"]["date_parsing"] == dict(locked_scores._APPROVED_DATE_PARSING_PAYLOAD_EXACT)
    assert not any("amendment_01" in str(path) for path in tmp_path.rglob("*"))


def test_repository_config_contains_exact_approved_date_contract() -> None:
    """The current repository config embeds the ordered 20-key Amendment parser verbatim."""
    config = yaml.safe_load((PROJECT_ROOT / locked_scores.BASE_CONFIG_RELATIVE_PATH).read_text(encoding="utf-8"))
    parser = config["loaders"]["date_parsing"]

    assert tuple(parser.items()) == locked_scores._APPROVED_DATE_PARSING_PAYLOAD_EXACT


def test_consolidated_config_equals_git_session0_plus_exact_amendment_and_governance() -> None:
    """Every non-governance field equals Session 0 with only the approved parser replaced."""
    git_environment = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}

    def git_yaml(commit: str, path: str) -> dict[str, Any]:
        completed = subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            env=git_environment,
        )
        return yaml.safe_load(completed.stdout)

    original = git_yaml(SESSION0_FREEZE, locked_scores.BASE_CONFIG_RELATIVE_PATH)
    amendment = git_yaml(AMENDMENT_FREEZE_DEFAULT_FOR_TESTS, AMENDMENT_CHANGED_PATHS[0])
    current = yaml.safe_load((PROJECT_ROOT / locked_scores.BASE_CONFIG_RELATIVE_PATH).read_text(encoding="utf-8"))
    expected = copy.deepcopy(original)
    expected["loaders"]["date_parsing"] = amendment["overlay_application_contract"][
        "overlay_date_parsing_payload"
    ]
    assert original["loaders"]["date_parsing"]["format"] == "%Y-%m-%d"
    assert tuple(expected["loaders"]["date_parsing"].items()) == locked_scores._APPROVED_DATE_PARSING_PAYLOAD_EXACT

    current_for_comparison = copy.deepcopy(current)
    for key in (
        "record_bundle_status",
        "consolidation_date",
        "consolidation_creates_new_session_0",
        "consolidation_backdates_amended_parser_to_original_session_0",
        "historical_consolidation",
    ):
        current_for_comparison.pop(key)
    current_authority = current_for_comparison.pop("authority")
    expected.pop("authority")
    current_release = current_for_comparison["outputs"]["scientific_and_closure"]["visualization_release"]["schema"]
    original_release = expected["outputs"]["scientific_and_closure"]["visualization_release"]["schema"]
    current_release["leaf_contracts"]["reviewed_plan_path"]["allowed_value"] = original_release["leaf_contracts"][
        "reviewed_plan_path"
    ]["allowed_value"]
    current_release["leaf_contracts"]["reviewed_plan_sha256"]["allowed_value"] = original_release[
        "leaf_contracts"
    ]["reviewed_plan_sha256"]["allowed_value"]

    assert current_for_comparison == expected
    assert current_authority["reviewed_plan_path"] == (
        "docs/plans/acquisition_context_robustness_audit_97707a26.plan.md"
    )
    assert current_authority["reviewed_plan_sha256"] == (
        "ea38735dce9383adeff839ca5c8bdcd084b5f065f5fd68eeda924cdaa522a452"
    )
    assert current_authority["agent_guidance_path"] == "AGENTS.md"
    assert "active_rules" not in current_authority


def test_repository_current_authority_is_single_config_and_not_cursor_runtime() -> None:
    """Only current project paths are active; legacy paths remain historical evidence."""
    config = yaml.safe_load((PROJECT_ROOT / locked_scores.BASE_CONFIG_RELATIVE_PATH).read_text(encoding="utf-8"))
    authority = config["authority"]

    assert authority["reviewed_plan_path"] == "docs/plans/acquisition_context_robustness_audit_97707a26.plan.md"
    assert authority["agent_guidance_path"] == "AGENTS.md"
    assert "active_rules" not in authority
    assert "overlay_application_contract" not in config
    assert "amendment_overlay" not in config


def test_consolidated_corrections_and_cumulative_ledger_are_unchanged() -> None:
    """C01--C16 and the ordered Amendment carry-forward booleans survive consolidation."""
    config = yaml.safe_load((PROJECT_ROOT / locked_scores.BASE_CONFIG_RELATIVE_PATH).read_text(encoding="utf-8"))
    amendment = config["historical_consolidation"]["amendment_01"]
    correction_ids = tuple(item["id"] for item in amendment["session_1_corrections"])
    cumulative = amendment["access_state"]["step_2_approval_materialization"]["cumulative_through_session"]

    assert correction_ids == tuple(f"C{index:02d}" for index in range(1, 17))
    assert tuple(cumulative) == tuple(AccessStateTracker().snapshot()["session_activity"])
    assert tuple(cumulative.values()) == (True, True, True, False, True, False, False, False, False)


def test_current_runtime_hashes_validate_separately_from_historical_hashes() -> None:
    """Current config/method/plan/guidance hashes match the consolidated manifest."""
    config_path = PROJECT_ROOT / locked_scores.BASE_CONFIG_RELATIVE_PATH
    config = locked_scores.load_effective_posthoc_config(config_path)
    manifest_path = PROJECT_ROOT / CURRENT_MANIFEST_RELATIVE_PATH
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert hasattr(locked_scores, "validate_current_runtime_authority")
    result = locked_scores.validate_current_runtime_authority(
        config_path,
        config,
        manifest,
        require_head_identical=False,
    )

    assert all(result.values())


def test_preserved_preflight_hashes_bind_to_historical_session0_not_current_bundle() -> None:
    """The byte-preserved preflight retains its original Session 0 authority hashes."""
    manifest = json.loads((PROJECT_ROOT / CURRENT_MANIFEST_RELATIVE_PATH).read_text(encoding="utf-8"))
    preflight = json.loads(
        (PROJECT_ROOT / SCORE_BLIND_PREFLIGHT_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    historical = manifest["historical_evidence"]["original_session_0"]["artifacts"]
    historical_by_path = {item["path"]: item["sha256"] for item in historical}

    assert preflight["hashes"]["config_sha256"] == historical_by_path[locked_scores.BASE_CONFIG_RELATIVE_PATH]
    assert preflight["hashes"]["methods_sha256"] == historical_by_path[
        "docs/modeling/final_post_hoc_supplementary_methods.md"
    ]
    assert preflight["hashes"]["freeze_manifest_sha256"] == historical_by_path[
        CURRENT_MANIFEST_RELATIVE_PATH
    ]
    assert preflight["hashes"]["config_sha256"] != manifest["current_runtime_authority"]["config"]["sha256"]


def test_cli_relative_config_reaches_preflight_with_canonical_frozen_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real CLI/preflight path boundary canonicalizes before governance work."""
    base_path = _write_synthetic_consolidated_config(tmp_path)
    captured: dict[str, Any] = {}

    class BoundaryReached(RuntimeError):
        """Stop after observing the canonical pre-access authority argument."""

    def observe_authority(**kwargs: Any) -> Any:
        captured["config_path"] = kwargs["config_path"]
        raise BoundaryReached

    monkeypatch.setattr("src.analysis.posthoc_final_locked_scores.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        "src.analysis.posthoc_final_locked_scores.validate_pre_access_authority",
        observe_authority,
    )
    monkeypatch.setattr(
        "src.analysis.posthoc_final_locked_scores.AccessStateTracker",
        lambda *_args, **_kwargs: pytest.fail("tracker creation must remain downstream"),
    )
    monkeypatch.setattr(
        "src.analysis.posthoc_final_locked_scores.load_csv_with_ordered_usecols",
        lambda *_args, **_kwargs: pytest.fail("participant reads must remain downstream"),
    )

    with pytest.raises(BoundaryReached):
        main(
            [
                "--config",
                "configs/post_hoc_final_supplementary.yaml",
                "--freeze-commit",
                SESSION0_FREEZE,
                "--amendment-freeze-commit",
                AMENDMENT_FREEZE_DEFAULT_FOR_TESTS,
                "--preflight-score-blind",
                "--confirm-reviewed-session1-corrections-and-tests",
            ]
        )

    assert captured["config_path"] == base_path.resolve()


def test_amendment_freeze_validation_uses_historical_commit_blobs_without_live_files() -> None:
    """Amendment history remains verifiable without a worktree overlay or manifest."""
    commit = AMENDMENT_FREEZE_DEFAULT_FOR_TESTS
    calls: list[tuple[str, ...]] = []
    blob_ids = ("blob-a", "blob-b", "blob-c")
    amendment = {
        "commit_id": commit,
        "sole_parent": SESSION0_FREEZE,
        "changed_paths_exact": list(AMENDMENT_CHANGED_PATHS),
        "artifacts": [
            {"path": path, "sha256": chr(97 + index) * 64, "git_blob_id": blob_ids[index]}
            for index, path in enumerate(AMENDMENT_CHANGED_PATHS)
        ],
        "approval_evidence": {"review_status": "approved", "user_confirmation": True},
    }

    def fake_git(*args: str) -> str:
        calls.append(args)
        outputs = {
            ("cat-file", "-e", f"{commit}^{{commit}}"): "",
            ("rev-list", "--parents", "-n", "1", commit): f"{commit} {SESSION0_FREEZE}",
            ("diff-tree", "--no-commit-id", "--name-only", "-r", commit): "\n".join(AMENDMENT_CHANGED_PATHS) + "\n",
            ("merge-base", "--is-ancestor", commit, "HEAD"): "",
        }
        for path, blob_id in zip(AMENDMENT_CHANGED_PATHS, blob_ids, strict=True):
            outputs[("rev-parse", f"{commit}:{path}")] = blob_id
        return outputs[args]

    result = validate_amendment_freeze_commit(commit, historical_amendment=amendment, git_runner=fake_git)
    assert result["approval_evidence_matches"] is True
    assert result["overlay_blob_matches"] is True
    assert not any(call[0] == "hash-object" for call in calls)


def test_repository_historical_chain_validates_from_git_objects_only() -> None:
    """Both completed freezes validate without reading any live Amendment path."""
    config = locked_scores.load_effective_posthoc_config(PROJECT_ROOT / locked_scores.BASE_CONFIG_RELATIVE_PATH)
    manifest = json.loads((PROJECT_ROOT / CURRENT_MANIFEST_RELATIVE_PATH).read_text(encoding="utf-8"))

    session_0 = locked_scores.validate_freeze_commit(SESSION0_FREEZE, config, manifest)
    amendment = validate_amendment_freeze_commit(
        AMENDMENT_FREEZE_DEFAULT_FOR_TESTS,
        historical_amendment=config["historical_consolidation"]["amendment_01"],
    )

    config_amendment = config["historical_consolidation"]["amendment_01"]
    manifest_amendment = manifest["historical_evidence"]["amendment_01"]
    for key in ("commit_id", "sole_parent", "changed_paths_exact", "artifacts", "approval_evidence"):
        assert config_amendment[key] == manifest_amendment[key]
    assert config_amendment["access_state"] == manifest_amendment["access_state"]
    assert config_amendment["session_1_corrections"] == manifest_amendment["session_1_corrections"]
    assert all(value is True for key, value in session_0.items() if key != "freeze_commit_id")
    assert all(value is True for key, value in amendment.items() if key != "amendment_freeze_commit_id")


def test_execute_git_prerequisites_use_real_temporary_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Committed paths pass while untracked, staged, modified, or dirty paths fail."""
    repository = tmp_path / "repository"
    repository.mkdir()

    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    required = [
        *locked_scores.SESSION_1_TO_3_IMPLEMENTATION_PATHS,
        SCORE_BLIND_PREFLIGHT_RELATIVE_PATH,
    ]
    git("init")
    git("config", "user.email", "contact-withheld.invalid")
    git("config", "user.name", "Synthetic Test")
    for relative in required:
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("committed\n", encoding="utf-8")
    (repository / "unrelated_tracked.txt").write_text("committed\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "synthetic baseline")
    monkeypatch.setattr("src.analysis.posthoc_final_locked_scores.PROJECT_ROOT", repository)

    locked_scores.validate_execute_git_prerequisites(required)

    for relative, staged in (
        (required[0], False),
        (SCORE_BLIND_PREFLIGHT_RELATIVE_PATH, False),
        (SCORE_BLIND_PREFLIGHT_RELATIVE_PATH, True),
        ("unrelated.txt", False),
        ("unrelated_tracked.txt", False),
    ):
        (repository / relative).write_text("temporary mutation\n", encoding="utf-8")
        if staged:
            git("add", relative)
        with pytest.raises(ValueError):
            locked_scores.validate_execute_git_prerequisites(required)
        git("reset", "--hard", "HEAD")
        (repository / "unrelated.txt").unlink(missing_ok=True)

    (repository / SCORE_BLIND_PREFLIGHT_RELATIVE_PATH).write_text("staged\n", encoding="utf-8")
    git("add", SCORE_BLIND_PREFLIGHT_RELATIVE_PATH)
    (repository / SCORE_BLIND_PREFLIGHT_RELATIVE_PATH).write_text("staged and worktree\n", encoding="utf-8")
    with pytest.raises(ValueError):
        locked_scores.validate_execute_git_prerequisites(required)

    git("reset", "--hard", "HEAD")
    git("rm", "--cached", SCORE_BLIND_PREFLIGHT_RELATIVE_PATH)
    with pytest.raises(ValueError):
        locked_scores.validate_execute_git_prerequisites(required)
    git("reset", "--hard", "HEAD")
    git("rm", "src/analysis/posthoc_final_date_grouped.py")
    git("commit", "-m", "remove required session path")
    with pytest.raises(ValueError):
        locked_scores.validate_execute_git_prerequisites(required)


def _install_real_framework_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Wire authority and view loading with synthetic data; the real framework, real
    per-aggregate callback, and real CSV staging all run unmocked.

    Only ``validate_pre_access_authority``, ``_load_approved_views``,
    ``_verify_exit_hashes``, and ``sha256_file`` are replaced, so that the access
    boundary is exercised through the actual production statistical helpers and
    the actual ``execute_approved_framework``/``_stage_csv`` call chain.
    """
    config = locked_scores.load_effective_posthoc_config(
        PROJECT_ROOT / "configs/post_hoc_final_supplementary.yaml"
    )
    entry_hashes = OrderedDict([("fixture", "a" * 64)])

    def fake_authority(**_kwargs: Any) -> tuple[dict[str, Any], OrderedDict[str, str], OrderedDict[str, Any]]:
        return config, entry_hashes, OrderedDict()

    def fake_load(_config: Mapping[str, Any], tracker: AccessStateTracker) -> OrderedDict[str, Any]:
        for key in (
            "participant_level_label_values_read",
            "participant_level_date_values_read",
            "participant_level_qc_values_read",
            "participant_level_score_values_read",
        ):
            tracker.mark(key)
        inputs = {
            "y_true": np.array([0, 1, 0, 1]),
            "p_raw": np.array([0.1, 0.9, 0.2, 0.8]),
            "dates": np.array(["2025/1/1", "2025/1/1", "2025/1/2", "2025/1/2"]),
            "disease_category": np.array([1, 3, 2, 4]),
            "strict_pass": np.array([True, True, False, False]),
        }
        return OrderedDict([("oof", inputs), ("test", inputs)])

    monkeypatch.setattr("src.analysis.posthoc_final_locked_scores.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        "src.analysis.posthoc_final_locked_scores.validate_pre_access_authority", fake_authority
    )
    monkeypatch.setattr("src.analysis.posthoc_final_locked_scores._load_approved_views", fake_load)
    monkeypatch.setattr(
        "src.analysis.posthoc_final_locked_scores._verify_exit_hashes", lambda _config, _entry: None
    )
    monkeypatch.setattr("src.analysis.posthoc_final_locked_scores.sha256_file", lambda _path: "b" * 64)
    return config


@pytest.mark.parametrize(
    ("failure_point", "derived_expected"),
    [
        pytest.param("before_first_real_aggregate", False, id="before_first_real_aggregate"),
        pytest.param("after_first_real_aggregate", True, id="after_first_real_aggregate"),
        pytest.param("staging_after_full_computation", True, id="staging_after_full_computation"),
    ],
)
def test_execute_access_state_marks_score_derived_results_at_real_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
    derived_expected: bool,
) -> None:
    """The real per-aggregate callback -- not a mocked whole-framework return -- gates the flag.

    Only the failure point differs across cases; the framework, its AUROC helper,
    its QC5 helper, and CSV staging are the actual production functions.
    """
    _install_real_framework_environment(tmp_path, monkeypatch)
    if failure_point == "before_first_real_aggregate":
        # The very first score-derived helper (called inside the very first
        # _metric_records invocation) fails before ever returning an aggregate.
        monkeypatch.setattr(
            "src.analysis.posthoc_final_locked_scores.single_auroc_delong_ci",
            lambda *_a, **_k: (_ for _ in ()).throw(ValueError("synthetic failure before any real aggregate")),
        )
    elif failure_point == "after_first_real_aggregate":
        # The first AUROC aggregate is allowed to form (firing the callback);
        # a distinct, later production computation step then fails.
        monkeypatch.setattr(
            "src.analysis.posthoc_final_locked_scores.strict_qc5_correctness_inventory",
            lambda *_a, **_k: (_ for _ in ()).throw(ValueError("synthetic failure in a later production step")),
        )
    else:
        # Both framework computations complete for real; only staging fails.
        monkeypatch.setattr(
            "src.analysis.posthoc_final_locked_scores._stage_csv",
            lambda *_a, **_k: (_ for _ in ()).throw(ValueError("synthetic staging failure")),
        )

    with pytest.raises(AccessStateError) as excinfo:
        execute_approved(
            tmp_path / "configs/post_hoc_final_supplementary.yaml",
            SESSION0_FREEZE,
            AMENDMENT_FREEZE_DEFAULT_FOR_TESTS,
        )
    snapshot = excinfo.value.access_state
    assert snapshot is not None
    assert snapshot["session_activity"]["participant_level_score_values_read"] is True
    assert snapshot["session_activity"]["new_score_derived_results_computed"] is derived_expected
    assert snapshot["session_activity"]["auxiliary_model_fit_executed"] is False


def test_execute_access_state_marks_true_on_full_successful_real_framework(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fully successful real execution promotes a completion log recording the flag true."""
    _install_real_framework_environment(tmp_path, monkeypatch)
    promoted = execute_approved(
        tmp_path / "configs/post_hoc_final_supplementary.yaml",
        SESSION0_FREEZE,
        AMENDMENT_FREEZE_DEFAULT_FOR_TESTS,
    )
    log_path = promoted[-1]
    assert log_path.name.endswith(".log")
    activity_line = next(
        line for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("access_state_session_activity")
    )
    keys = list(locked_scores.ACCESS_STATE_KEYS)
    values = activity_line.split("\t")[1:]
    assert values[keys.index("new_score_derived_results_computed")] == "true"
    assert values[keys.index("participant_level_score_values_read")] == "true"
    assert values[keys.index("auxiliary_model_fit_executed")] == "false"


def test_score_blind_mode_cannot_promote_session_four_outputs(tmp_path: Path) -> None:
    """Wrong-mode helpers refuse Session 4 targets and leave no temporary residue."""
    target = tmp_path / "posthoc_final_locked_score_metrics_minirocket_ridge.csv"
    frame = pd.DataFrame([{"analysis_id": "x"}])
    with pytest.raises(ValueError):
        _write_atomic_csv(target, frame, ["analysis_id", "view"])
    assert not target.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_execute_approved_orchestration_promotes_log_last_with_mocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mocked approved execution validates dual freeze, promotes CSVs, then the log."""
    contracts = {
        "locked_score_metrics": {
            "path": str(tmp_path / "metrics.csv"),
            "columns": ["analysis_id", "view"],
            "planned_row_count": 1,
        },
        "locked_score_composition": {
            "path": str(tmp_path / "composition.csv"),
            "columns": ["analysis_id", "view"],
            "planned_row_count": 1,
        },
        "qc5_correctness": {
            "path": str(tmp_path / "qc5.csv"),
            "columns": ["analysis_id", "view"],
            "planned_row_count": 1,
        },
        "within_date_concordance": {
            "path": str(tmp_path / "within.csv"),
            "columns": ["analysis_id", "view"],
            "planned_row_count": 1,
        },
    }
    config = {
        "outputs": {
            "scientific_and_closure": contracts,
            "completion_logs": {"locked_scores": str(tmp_path / "locked_scores.log")},
            "session_0": {
                "config": "configs/post_hoc_final_supplementary.yaml",
                "methods": "docs/modeling/final_post_hoc_supplementary_methods.md",
                "freeze_manifest": "outputs/analysis/posthoc_final_freeze_manifest.json",
            },
        },
        "frozen_inputs": {"whole_container_sha256": OrderedDict([("fixture", "a" * 64)])},
        "threshold": {"threshold_p_raw": 0.3418938786147648},
        "loaders": {"per_purpose_ordered_usecols": {}},
    }
    entry_hashes = OrderedDict([("fixture", "a" * 64)])
    access_order: list[str] = []

    def fake_authority(**kwargs: Any) -> tuple[dict[str, Any], OrderedDict[str, str], OrderedDict[str, Any]]:
        access_order.append("authority")
        assert kwargs["mode"] == "execute"
        assert kwargs["freeze_commit"] == SESSION0_FREEZE
        assert kwargs["amendment_freeze_commit"] == AMENDMENT_FREEZE_DEFAULT_FOR_TESTS
        return config, entry_hashes, OrderedDict()

    def fake_load(_config: Mapping[str, Any], tracker: AccessStateTracker) -> OrderedDict[str, Any]:
        access_order.append("load")
        tracker.mark("participant_level_label_values_read")
        tracker.mark("participant_level_date_values_read")
        tracker.mark("participant_level_qc_values_read")
        tracker.mark("participant_level_score_values_read")
        inputs = {
            "y_true": np.array([0, 1]),
            "p_raw": np.array([0.1, 0.9]),
            "dates": np.array(["2025/1/1", "2025/1/1"]),
            "disease_category": np.array([1, 3]),
            "strict_pass": np.array([True, True]),
        }
        return OrderedDict([("oof", inputs), ("test", inputs)])

    def fake_framework(views: Any, *, config: Any = None, on_first_aggregate: Any = None) -> dict[str, Any]:
        access_order.append("compute")
        if on_first_aggregate is not None:
            on_first_aggregate()
        frames = OrderedDict(
            (name, pd.DataFrame([{"analysis_id": name, "view": "oof"}])) for name in contracts
        )
        return {"frames": frames, "scores_read": True}

    # All writes resolve under tmp_path by monkeypatching the module PROJECT_ROOT.
    monkeypatch.setattr("src.analysis.posthoc_final_locked_scores.PROJECT_ROOT", tmp_path)
    for name, contract in contracts.items():
        contract["path"] = f"outputs/analysis/_synth_{name}.csv"
    config["outputs"]["completion_logs"]["locked_scores"] = "outputs/logs/_synth_locked_scores.log"
    monkeypatch.setattr(
        "src.analysis.posthoc_final_locked_scores.validate_pre_access_authority",
        fake_authority,
    )
    monkeypatch.setattr("src.analysis.posthoc_final_locked_scores._load_approved_views", fake_load)
    monkeypatch.setattr("src.analysis.posthoc_final_locked_scores.execute_approved_framework", fake_framework)
    monkeypatch.setattr(
        "src.analysis.posthoc_final_locked_scores._verify_exit_hashes",
        lambda _config, _entry: access_order.append("exit_hash"),
    )
    monkeypatch.setattr(
        "src.analysis.posthoc_final_locked_scores.sha256_file",
        lambda path: "b" * 64,
    )

    promoted = execute_approved(
        tmp_path / "configs/post_hoc_final_supplementary.yaml",
        SESSION0_FREEZE,
        AMENDMENT_FREEZE_DEFAULT_FOR_TESTS,
    )
    assert access_order[:4] == ["authority", "load", "compute", "compute"]
    assert "exit_hash" in access_order
    assert access_order.index("authority") < access_order.index("load")
    assert [path.name for path in promoted] == [
        "_synth_locked_score_metrics.csv",
        "_synth_locked_score_composition.csv",
        "_synth_qc5_correctness.csv",
        "_synth_within_date_concordance.csv",
        "_synth_locked_scores.log",
    ]
    assert promoted[-1].name.endswith(".log")
    assert promoted[-1].read_text(encoding="utf-8").startswith("header\t")
    assert not list(promoted[-1].parent.glob("._synth_locked_scores.log.tmp"))
    # Every promoted target lives strictly under the temporary project root.
    for path in promoted:
        assert tmp_path in path.parents


def test_preflight_retry_fails_before_participant_access_without_test_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Authority may pass, but missing synthetic-test evidence still blocks reads."""
    calls: list[str] = []

    def fake_authority(**kwargs: Any) -> tuple[dict[str, Any], OrderedDict[str, str], OrderedDict[str, Any]]:
        calls.append("authority")
        return (
            {
                "outputs": {
                    "session_0": {
                        "methods": "docs/modeling/final_post_hoc_supplementary_methods.md",
                        "freeze_manifest": "outputs/analysis/posthoc_final_freeze_manifest.json",
                    }
                },
                "frozen_inputs": {"whole_container_sha256": OrderedDict(), "prediction_static_headers": {}},
                "historical_consolidation": {
                    "amendment_01": {
                        "immediate_preflight_retry_authorized_at_materialization": False,
                        "access_state": {
                            "step_2_approval_materialization": {
                                "cumulative_through_session": OrderedDict(
                                    (key, False) for key in AccessStateTracker().snapshot()["session_activity"]
                                )
                            }
                        },
                    }
                },
            },
            OrderedDict(),
            OrderedDict(
                [
                    (
                        "amendment_freeze",
                        OrderedDict(
                            [
                                ("amendment_freeze_commit_id", AMENDMENT_FREEZE_DEFAULT_FOR_TESTS),
                                ("exists", True),
                                ("is_ancestor_of_head", True),
                                ("parent_is_session0_freeze", True),
                                ("changed_paths_exact", True),
                                ("overlay_blob_matches", True),
                                ("approval_evidence_matches", True),
                            ]
                        ),
                    ),
                    ("session0_freeze", OrderedDict()),
                    ("rule_hashes", OrderedDict()),
                ]
            ),
        )

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        calls.append("participant_read")
        raise AssertionError("participant values must not be read")

    monkeypatch.setattr(
        "src.analysis.posthoc_final_locked_scores.validate_pre_access_authority",
        fake_authority,
    )
    monkeypatch.setattr("src.analysis.posthoc_final_locked_scores.load_csv_with_ordered_usecols", boom)
    with pytest.raises(ValueError, match="Conditional preflight retry authorization"):
        run_score_blind_preflight(
            PROJECT_ROOT / "configs/post_hoc_final_supplementary.yaml",
            SESSION0_FREEZE,
            AMENDMENT_FREEZE_DEFAULT_FOR_TESTS,
            synthetic_tests_passed=False,
        )
    assert calls == ["authority"]


# --- CLI confirmation flag (correction A) --------------------------------------------------------


def test_cli_confirm_flag_passes_through_with_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """The explicit confirmation flag reaches the preflight as synthetic_tests_passed."""
    captured: dict[str, Any] = {}

    def fake_preflight(
        config: Path,
        freeze_commit: str,
        amendment_freeze_commit: str,
        *,
        synthetic_tests_passed: bool = False,
        git_runner: Any = None,
    ) -> Path:
        captured["synthetic_tests_passed"] = synthetic_tests_passed
        return Path("outputs/analysis/posthoc_final_score_blind_preflight.json")

    monkeypatch.setattr("scripts.run_posthoc_final_locked_scores.run_score_blind_preflight", fake_preflight)
    code = main(
        [
            "--config",
            "configs/post_hoc_final_supplementary.yaml",
            "--freeze-commit",
            SESSION0_FREEZE,
            "--amendment-freeze-commit",
            AMENDMENT_FREEZE_DEFAULT_FOR_TESTS,
            "--preflight-score-blind",
            "--confirm-reviewed-session1-corrections-and-tests",
        ]
    )
    assert code == 0
    assert captured["synthetic_tests_passed"] is True


def test_cli_confirm_flag_rejected_with_execute(monkeypatch: pytest.MonkeyPatch) -> None:
    """The confirmation flag is inadmissible for approved execution and blocks all access."""
    executed: list[str] = []

    monkeypatch.setattr(
        "scripts.run_posthoc_final_locked_scores.execute_approved",
        lambda *args, **kwargs: executed.append("execute"),
    )
    with pytest.raises(SystemExit):
        main(
            [
                "--config",
                "configs/post_hoc_final_supplementary.yaml",
                "--freeze-commit",
                SESSION0_FREEZE,
                "--amendment-freeze-commit",
                AMENDMENT_FREEZE_DEFAULT_FOR_TESTS,
                "--execute-approved",
                "--confirm-reviewed-session1-corrections-and-tests",
            ]
        )
    assert executed == []


# --- C16 orchestration of the actual run_score_blind_preflight (correction F) ---------------------

_PREDICTION_HEADER = ["sample_id", "y_true", "p_raw", "p_cal"]
_OOF_RELATIVE = "outputs/predictions/binary_minirocket_ridge_nested_cv_oof.csv"
_TEST_RELATIVE = "outputs/predictions/binary_minirocket_ridge_test_predictions.csv"
_FORBIDDEN_COLUMNS = {"p_raw", "p_cal", "collection_location"}


# Mapping of ordered-usecols load failures to their C16 stage identifiers.
_LOAD_FAIL_STAGES = {
    "oof": "oof_ordered_usecols",
    "test": "test_ordered_usecols",
    "lineage": "sample_index_label_date_read",
    "diagnostics": "qc_diagnostics_load",
    "qc_summary": "qc_summary_load",
}
# The full happy-path call sequence of the mocked score-blind preflight body.
_PREFLIGHT_HAPPY_CALLS = [
    "oof",
    "test",
    "lineage",
    "split",
    "alignment",
    "date_feasibility",
    "diagnostics",
    "qc_summary",
    "qc5",
    "difficult",
    "difficult",
    "build",
    "exit_hash",
    "write",
]


def _install_preflight_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_stage: str | None = None,
    step2_cumulative: OrderedDict[str, bool] | None = None,
) -> dict[str, Any]:
    """Wire a mocked score-blind preflight environment rooted at ``tmp_path``.

    Args:
        tmp_path: Temporary project root for all synthetic I/O.
        monkeypatch: Pytest monkeypatch fixture.
        fail_stage: Optional post-authority C16 stage identifier to fail closed at.
        step2_cumulative: Optional amendment Step-2 cumulative carry-forward ledger.

    Returns:
        A knobs mapping exposing the requested-usecols log, the captured
        ``build_score_blind_preflight`` access-state snapshot, and the call log.
    """
    _write_synthetic_consolidated_config(tmp_path)
    monkeypatch.setattr("src.analysis.posthoc_final_locked_scores.PROJECT_ROOT", tmp_path)
    for relative in (_OOF_RELATIVE, _TEST_RELATIVE):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(",".join(_PREDICTION_HEADER) + "\n", encoding="utf-8")

    container_hashes = OrderedDict([("data/processed/sample_index.csv", "a" * 64)])
    oof_header = ["sample_id", "wrong"] if fail_stage == "prediction_header_inspection" else list(_PREDICTION_HEADER)
    cumulative = step2_cumulative or OrderedDict(
        (key, False) for key in AccessStateTracker().snapshot()["session_activity"]
    )
    config = {
        "frozen_inputs": {
            "whole_container_sha256": container_hashes,
            "prediction_static_headers": {
                _OOF_RELATIVE: oof_header,
                _TEST_RELATIVE: list(_PREDICTION_HEADER),
            },
        },
        "outputs": {
            "session_0": {
                "config": "configs/post_hoc_final_supplementary.yaml",
                "methods": "docs/modeling/final_post_hoc_supplementary_methods.md",
                "freeze_manifest": "outputs/analysis/posthoc_final_freeze_manifest.json",
            }
        },
        "historical_consolidation": {
            "original_session_0": {
                "artifacts": [
                    {"path": "configs/post_hoc_final_supplementary.yaml", "sha256": "c" * 64},
                    {
                        "path": "docs/modeling/final_post_hoc_supplementary_methods.md",
                        "sha256": "c" * 64,
                    },
                    {
                        "path": "outputs/analysis/posthoc_final_freeze_manifest.json",
                        "sha256": "c" * 64,
                    },
                ]
            },
            "amendment_01": {
                "immediate_preflight_retry_authorized_at_materialization": False,
                "access_state": {
                    "step_2_approval_materialization": {"cumulative_through_session": cumulative}
                },
            }
        },
    }
    entry_hashes = OrderedDict(container_hashes)
    authority = OrderedDict(
        [
            (
                "amendment_freeze",
                OrderedDict(
                    [
                        ("amendment_freeze_commit_id", AMENDMENT_FREEZE_DEFAULT_FOR_TESTS),
                        ("exists", True),
                        ("is_ancestor_of_head", True),
                        ("parent_is_session0_freeze", True),
                        ("changed_paths_exact", True),
                        ("overlay_blob_matches", True),
                        ("approval_evidence_matches", True),
                    ]
                ),
            ),
            ("session0_freeze", OrderedDict([("freeze_commit_id", SESSION0_FREEZE)])),
        ]
    )
    monkeypatch.setattr(
        "src.analysis.posthoc_final_locked_scores.validate_pre_access_authority",
        lambda **_kwargs: (config, entry_hashes, authority),
    )
    lineage = pd.DataFrame(
        {
            "sample_id": list(range(1040)),
            "label": [index % 2 for index in range(1040)],
            "disease_category": [index % 6 for index in range(1040)],
            "collection_time": ["2025/1/1"] * 1040,
            "in_final_output": [True] * 972 + [False] * 68,
        }
    )
    oof = pd.DataFrame({"sample_id": list(range(777)), "y_true": [index % 2 for index in range(777)]})
    test = pd.DataFrame(
        {"sample_id": list(range(777, 972)), "y_true": [index % 2 for index in range(195)]}
    )
    diagnostics = pd.DataFrame({"sample_id": [0, 1]})
    qc_summary = pd.DataFrame({"sample_id": [0, 1]})
    requested_usecols: list[list[str]] = []
    calls: list[str] = []

    def fake_load(path: Path, usecols: Any, expected: Any) -> pd.DataFrame:
        columns = list(usecols)
        requested_usecols.append(columns)
        assert not (_FORBIDDEN_COLUMNS & set(columns))
        name = Path(path).name
        if "in_final_output" in columns:
            label, frame = "lineage", lineage
        elif any(column.endswith("_saturation_fraction") for column in columns):
            label, frame = "diagnostics", diagnostics
        elif "qc1_env_sensor_fault_pass" in columns:
            label, frame = "qc_summary", qc_summary
        elif "oof" in name:
            label, frame = "oof", oof
        else:
            label, frame = "test", test
        calls.append(label)
        if fail_stage == _LOAD_FAIL_STAGES.get(label):
            raise ValueError(f"synthetic {label} load failure")
        return frame.copy()

    monkeypatch.setattr("src.analysis.posthoc_final_locked_scores.load_csv_with_ordered_usecols", fake_load)

    def fake_split(_path: Any, _lineage: Any) -> tuple[Any, Any]:
        calls.append("split")
        if fail_stage == "split_membership":
            raise ValueError("synthetic split-membership failure")
        return (list(range(777)), list(range(777, 972)))

    monkeypatch.setattr("src.analysis.posthoc_final_locked_scores._split_membership", fake_split)

    def fake_alignment(**_kwargs: Any) -> dict[str, str]:
        calls.append("alignment")
        if fail_stage == "split_alignment_validation":
            raise ValueError("synthetic alignment failure")
        return {"one_to_one_alignment_status": "pass", "train_test_disjointness_status": "pass"}

    monkeypatch.setattr(
        "src.analysis.posthoc_final_locked_scores.validate_locked_score_alignment",
        fake_alignment,
    )

    def fake_date(_modeling: Any) -> OrderedDict[str, int]:
        calls.append("date_feasibility")
        if fail_stage == "slash_date_parsing":
            raise ValueError("Date parsing failed: invalid_date_format_count=1.")
        return OrderedDict([("n_dates_overall", 40), ("n_dates_with_both_endpoint_classes", 34)])

    monkeypatch.setattr("src.analysis.posthoc_final_locked_scores._modeling_date_feasibility", fake_date)

    def fake_qc5(_diagnostics: Any, _summary: Any) -> dict[str, int]:
        calls.append("qc5")
        if fail_stage == "qc5_reconciliation":
            return {"n_qc5_evaluable": 1, "n_strict_fail": 0, "n_strict_pass": 1}
        return {"n_qc5_evaluable": 974, "n_strict_fail": 298, "n_strict_pass": 676}

    monkeypatch.setattr("src.analysis.posthoc_final_locked_scores.reconstruct_qc5", fake_qc5)

    def fake_difficult(frame: pd.DataFrame) -> pd.DataFrame:
        calls.append("difficult")
        view = pd.DataFrame({"y_true": [0, 1]})
        view.attrs["disease_category_counts"] = OrderedDict((str(index), 0) for index in range(6))
        return view

    monkeypatch.setattr("src.analysis.posthoc_final_locked_scores.difficult_spectrum_filter", fake_difficult)
    monkeypatch.setattr("src.analysis.posthoc_final_locked_scores.sha256_file", lambda _path: "a" * 64)

    def fake_exit(_config: Any, _entry: Any) -> None:
        calls.append("exit_hash")
        if fail_stage == "exit_rehash":
            raise ValueError("synthetic exit re-hash failure")

    monkeypatch.setattr("src.analysis.posthoc_final_locked_scores._verify_exit_hashes", fake_exit)

    captured: dict[str, Any] = {}

    def fake_build(**kwargs: Any) -> OrderedDict[str, Any]:
        calls.append("build")
        if fail_stage == "candidate_schema_validation":
            raise ValueError("synthetic candidate schema validation failure")
        if fail_stage == "candidate_privacy_validation":
            raise ValueError("synthetic candidate privacy validation failure")
        captured["access_state"] = kwargs["access_state"]
        return OrderedDict([("status", "pass")])

    monkeypatch.setattr("src.analysis.posthoc_final_locked_scores.build_score_blind_preflight", fake_build)

    def fake_write(target: Path, candidate: Any, *_validators: Any) -> None:
        calls.append("write")
        if fail_stage in {"temporary_candidate_read_back", "final_promotion"}:
            raise ValueError(f"synthetic {fail_stage} failure")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("candidate", encoding="utf-8")

    monkeypatch.setattr("src.analysis.posthoc_final_locked_scores.atomic_write_json", fake_write)
    return {
        "config": config,
        "requested_usecols": requested_usecols,
        "captured": captured,
        "calls": calls,
        "target": tmp_path / "outputs/analysis/posthoc_final_score_blind_preflight.json",
    }


def test_run_score_blind_preflight_success_is_score_blind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fully mocked successful preflight marks only score-blind accesses and promotes."""
    env = _install_preflight_environment(tmp_path, monkeypatch)
    target = run_score_blind_preflight(
        tmp_path / "configs/post_hoc_final_supplementary.yaml",
        SESSION0_FREEZE,
        AMENDMENT_FREEZE_DEFAULT_FOR_TESTS,
        synthetic_tests_passed=True,
    )
    assert target == env["target"]
    assert target.exists()
    activity = env["captured"]["access_state"]["session_activity"]
    assert activity["prediction_container_bytes_hashed"] is True
    assert activity["prediction_schema_or_header_inspected"] is True
    assert activity["participant_level_label_values_read"] is True
    assert activity["participant_level_date_values_read"] is True
    assert activity["participant_level_qc_values_read"] is True
    assert activity["participant_level_score_values_read"] is False
    assert activity["new_score_derived_results_computed"] is False
    assert activity["auxiliary_model_fit_executed"] is False
    assert activity["privacy_reference_loader_executed"] is False
    for columns in env["requested_usecols"]:
        assert not (_FORBIDDEN_COLUMNS & set(columns))


def test_run_score_blind_preflight_success_preserves_step2_cumulative_prior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Success carries forward the real amendment Step-2 cumulative prior (correction E)."""
    prior = OrderedDict((key, False) for key in AccessStateTracker().snapshot()["session_activity"])
    for key in (
        "prediction_container_bytes_hashed",
        "prediction_schema_or_header_inspected",
        "participant_level_label_values_read",
        "participant_level_date_values_read",
    ):
        prior[key] = True
    env = _install_preflight_environment(tmp_path, monkeypatch, step2_cumulative=prior)
    target = run_score_blind_preflight(
        tmp_path / "configs/post_hoc_final_supplementary.yaml",
        SESSION0_FREEZE,
        AMENDMENT_FREEZE_DEFAULT_FOR_TESTS,
        synthetic_tests_passed=True,
    )
    assert target.exists()
    assert env["calls"] == _PREFLIGHT_HAPPY_CALLS
    state = env["captured"]["access_state"]
    activity = state["session_activity"]
    cumulative = state["cumulative_through_session"]
    # This session performs only score-blind reads; QC is newly read this session.
    assert activity["participant_level_qc_values_read"] is True
    assert activity["participant_level_score_values_read"] is False
    # Cumulative retains every Step-2 prior true flag and unions this session's activity.
    for key in (
        "prediction_container_bytes_hashed",
        "prediction_schema_or_header_inspected",
        "participant_level_label_values_read",
        "participant_level_date_values_read",
        "participant_level_qc_values_read",
    ):
        assert cumulative[key] is True
    assert cumulative["participant_level_score_values_read"] is False
    assert cumulative["new_score_derived_results_computed"] is False
    assert cumulative["auxiliary_model_fit_executed"] is False


@pytest.mark.parametrize(
    ("stage", "call_count", "labels_expected", "date_expected", "qc_expected"),
    [
        ("prediction_header_inspection", 0, False, False, False),
        ("oof_ordered_usecols", 1, False, False, False),
        ("test_ordered_usecols", 2, True, False, False),
        ("sample_index_label_date_read", 3, True, False, False),
        ("split_membership", 4, True, True, False),
        ("split_alignment_validation", 5, True, True, False),
        ("slash_date_parsing", 6, True, True, False),
        ("qc_diagnostics_load", 7, True, True, False),
        ("qc_summary_load", 8, True, True, True),
        ("qc5_reconciliation", 9, True, True, True),
        ("candidate_schema_validation", 12, True, True, True),
        ("candidate_privacy_validation", 12, True, True, True),
        ("exit_rehash", 13, True, True, True),
        ("temporary_candidate_read_back", 14, True, True, True),
        ("final_promotion", 14, True, True, True),
    ],
)
def test_run_score_blind_preflight_post_authority_stages_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    call_count: int,
    labels_expected: bool,
    date_expected: bool,
    qc_expected: bool,
) -> None:
    """Every post-authority C16 stage fails closed with a sanitized AccessStateError."""
    env = _install_preflight_environment(tmp_path, monkeypatch, fail_stage=stage)
    with pytest.raises(AccessStateError) as excinfo:
        run_score_blind_preflight(
            tmp_path / "configs/post_hoc_final_supplementary.yaml",
            SESSION0_FREEZE,
            AMENDMENT_FREEZE_DEFAULT_FOR_TESTS,
            synthetic_tests_passed=True,
        )
    message = str(excinfo.value)
    assert "failed closed" in message
    assert "sample_id" not in message
    assert "2025/" not in message and "2025-" not in message
    # Exact earlier calls occurred and no later call was made.
    assert env["calls"] == _PREFLIGHT_HAPPY_CALLS[:call_count]
    snapshot = excinfo.value.access_state
    assert snapshot is not None
    activity = snapshot["session_activity"]
    assert activity["prediction_container_bytes_hashed"] is True
    assert activity["prediction_schema_or_header_inspected"] is True
    assert activity["participant_level_label_values_read"] is labels_expected
    assert activity["participant_level_date_values_read"] is date_expected
    assert activity["participant_level_qc_values_read"] is qc_expected
    assert activity["participant_level_score_values_read"] is False
    assert activity["new_score_derived_results_computed"] is False
    assert activity["auxiliary_model_fit_executed"] is False
    assert activity["privacy_reference_loader_executed"] is False
    # p_raw, p_cal, and collection_location are never requested from any loader.
    for columns in env["requested_usecols"]:
        assert not (_FORBIDDEN_COLUMNS & set(columns))
    assert not env["target"].exists()
    assert not list(tmp_path.rglob("*.tmp")) and not list(tmp_path.rglob(".*.tmp"))


# --- C16 pre-access authority stages via the real validate_pre_access_authority (correction F) -----


def _valid_freeze_validation() -> OrderedDict[str, Any]:
    """Return a passing Session 0 freeze-validation record."""
    return OrderedDict(
        [
            ("freeze_commit_id", SESSION0_FREEZE),
            ("exists", True),
            ("is_ancestor_of_head", True),
            ("parent_is_baseline_head", True),
            ("changed_paths_exact", True),
            ("config_blob_matches", True),
            ("methods_blob_matches", True),
            ("manifest_blob_matches", True),
            ("approval_evidence_matches", True),
            ("active_rule_hashes_match", True),
        ]
    )


def _valid_amendment_validation() -> OrderedDict[str, Any]:
    """Return a passing amendment freeze-validation record."""
    return OrderedDict(
        [
            ("amendment_freeze_commit_id", AMENDMENT_FREEZE_DEFAULT_FOR_TESTS),
            ("exists", True),
            ("is_ancestor_of_head", True),
            ("parent_is_session0_freeze", True),
            ("changed_paths_exact", True),
            ("overlay_blob_matches", True),
            ("approval_evidence_matches", True),
        ]
    )


def _install_authority_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_stage: str | None = None,
) -> dict[str, Any]:
    """Wire the real ``validate_pre_access_authority`` orchestrator over mocked dependencies.

    Args:
        tmp_path: Temporary project root for all synthetic I/O.
        monkeypatch: Pytest monkeypatch fixture.
        fail_stage: Optional pre-access C16 stage identifier to fail closed at.

    Returns:
        A knobs mapping with the base config, container path, and a participant-read log.
    """
    _write_synthetic_consolidated_config(tmp_path)
    monkeypatch.setattr("src.analysis.posthoc_final_locked_scores.PROJECT_ROOT", tmp_path)
    plan_rel = "docs/plans/approved.plan.md"
    guidance_rel = "AGENTS.md"
    container_rel = "data/processed/sample_index.csv"
    methods_rel = "docs/modeling/final_post_hoc_supplementary_methods.md"
    manifest_rel = CURRENT_MANIFEST_RELATIVE_PATH
    cumulative = OrderedDict(
        (
            key,
            key
            in {
                "prediction_container_bytes_hashed",
                "prediction_schema_or_header_inspected",
                "participant_level_label_values_read",
                "participant_level_date_values_read",
            },
        )
        for key in AccessStateTracker().snapshot()["session_activity"]
    )
    amendment = {
        "commit_id": AMENDMENT_FREEZE_DEFAULT_FOR_TESTS,
        "sole_parent": SESSION0_FREEZE,
        "changed_paths_exact": list(AMENDMENT_CHANGED_PATHS),
        "artifacts": [
            {"path": path, "sha256": chr(97 + index) * 64, "git_blob_id": f"blob-{index}"}
            for index, path in enumerate(AMENDMENT_CHANGED_PATHS)
        ],
        "approval_evidence": {"review_status": "approved", "user_confirmation": True},
        "immediate_preflight_retry_authorized_at_materialization": False,
        "access_state": {
            "step_2_approval_materialization": {"cumulative_through_session": cumulative}
        },
    }
    historical_session_0 = {
        "commit_id": SESSION0_FREEZE,
        "artifacts": [
            {"path": locked_scores.BASE_CONFIG_RELATIVE_PATH, "sha256": "c" * 64},
            {"path": methods_rel, "sha256": "c" * 64},
            {"path": manifest_rel, "sha256": "c" * 64},
        ],
    }
    base = {
        "authority": {
            "reviewed_plan_path": plan_rel,
            "reviewed_plan_sha256": "a" * 64,
            "agent_guidance_path": guidance_rel,
            "agent_guidance_sha256": "b" * 64,
        },
        "historical_consolidation": {
            "original_session_0": historical_session_0,
            "amendment_01": amendment,
        },
        "outputs": {
            "session_0": {
                "config": "configs/post_hoc_final_supplementary.yaml",
                "methods": methods_rel,
                "freeze_manifest": manifest_rel,
            },
            "scientific_and_closure": {
                "locked_score_metrics": {"path": SESSION_4_SCIENTIFIC_OUTPUT_PATHS[0]},
                "locked_score_composition": {"path": SESSION_4_SCIENTIFIC_OUTPUT_PATHS[1]},
                "qc5_correctness": {"path": SESSION_4_SCIENTIFIC_OUTPUT_PATHS[2]},
                "within_date_concordance": {"path": SESSION_4_SCIENTIFIC_OUTPUT_PATHS[3]},
                "score_blind_preflight": {"path": SCORE_BLIND_PREFLIGHT_RELATIVE_PATH},
            },
            "completion_logs": {"locked_scores": SESSION_4_COMPLETION_LOG_PATH},
            "future_implementation_paths": dict.fromkeys(
                locked_scores.SESSION_1_TO_3_IMPLEMENTATION_PATHS,
                "not_yet_created",
            ),
        },
        "frozen_inputs": {"whole_container_sha256": {container_rel: "d" * 64}},
        "loaders": {"date_parsing": dict(locked_scores._APPROVED_DATE_PARSING_PAYLOAD_EXACT)},
    }
    manifest = {
        "historical_evidence": {
            "original_session_0": historical_session_0,
            "amendment_01": amendment,
        },
        "current_runtime_authority": {
            "config": {"path": locked_scores.BASE_CONFIG_RELATIVE_PATH, "sha256": "c" * 64},
            "methods": {"path": methods_rel, "sha256": "c" * 64},
            "manifest": {"path": manifest_rel, "self_hash": None, "self_hash_policy": "forbidden"},
            "operational_plan": {"path": plan_rel, "sha256": "a" * 64},
            "agent_guidance": {"path": guidance_rel, "sha256": "b" * 64},
            "cutover_commit_id": None,
        },
    }
    sha_by_rel = {
        plan_rel: "a" * 64,
        guidance_rel: "b" * 64,
        container_rel: "d" * 64,
        locked_scores.BASE_CONFIG_RELATIVE_PATH: "c" * 64,
        methods_rel: "c" * 64,
    }
    manifest_path = tmp_path / manifest_rel
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def fake_load_config(_path: Path) -> dict[str, Any]:
        return base

    monkeypatch.setattr("src.analysis.posthoc_final_locked_scores.load_posthoc_config", fake_load_config)

    def fake_sha(path: Path) -> str:
        rel = str(Path(path).relative_to(tmp_path))
        if fail_stage == "active_plan_validation" and rel == plan_rel:
            return "f" * 64
        if fail_stage == "agent_guidance_validation" and rel == guidance_rel:
            return "f" * 64
        if fail_stage == "current_authority_validation" and rel == methods_rel:
            return "f" * 64
        if fail_stage == "entry_hashing" and rel == container_rel:
            return "f" * 64
        return sha_by_rel.get(rel, "c" * 64)

    monkeypatch.setattr("src.analysis.posthoc_final_locked_scores.sha256_file", fake_sha)

    def fake_tags(_git_runner: Any) -> None:
        if fail_stage == "annotated_tag_validation":
            raise ValueError("synthetic annotated-tag validation failure")

    monkeypatch.setattr("src.analysis.posthoc_final_locked_scores._validate_annotated_tag_chain", fake_tags)

    def fake_freeze(*_args: Any, **_kwargs: Any) -> OrderedDict[str, Any]:
        if fail_stage == "original_session0_freeze_validation":
            raise ValueError("synthetic Session 0 freeze validation failure")
        return _valid_freeze_validation()

    monkeypatch.setattr("src.analysis.posthoc_final_locked_scores.validate_freeze_commit", fake_freeze)

    def fake_amendment(*_args: Any, **_kwargs: Any) -> OrderedDict[str, Any]:
        if fail_stage == "amendment_freeze_validation":
            raise ValueError("synthetic amendment freeze validation failure")
        return _valid_amendment_validation()

    monkeypatch.setattr(
        "src.analysis.posthoc_final_locked_scores.validate_amendment_freeze_commit", fake_amendment
    )

    def fake_effective(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        if fail_stage == "consolidated_config_validation":
            raise ValueError("synthetic consolidated-config failure")
        return base

    monkeypatch.setattr("src.analysis.posthoc_final_locked_scores.load_effective_posthoc_config", fake_effective)

    if fail_stage == "stale_target_completion_log_validation":
        stale = tmp_path / SESSION_4_SCIENTIFIC_OUTPUT_PATHS[0]
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_text("stale", encoding="utf-8")

    participant_calls: list[str] = []

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        participant_calls.append("participant_read")
        raise AssertionError("participant values must not be read before authority passes")

    monkeypatch.setattr("src.analysis.posthoc_final_locked_scores.load_csv_with_ordered_usecols", boom)
    return {"base": base, "container_rel": container_rel, "participant_calls": participant_calls}


@pytest.mark.parametrize(
    ("stage", "message_fragment"),
    [
        ("active_plan_validation", "Reviewed plan hash"),
        ("agent_guidance_validation", "Agent guidance hash"),
        ("current_authority_validation", "Current methods hash"),
        ("annotated_tag_validation", "synthetic annotated-tag validation failure"),
        ("original_session0_freeze_validation", "synthetic Session 0 freeze validation failure"),
        ("amendment_freeze_validation", "synthetic amendment freeze validation failure"),
        ("consolidated_config_validation", "synthetic consolidated-config failure"),
        ("stale_target_completion_log_validation", "already exists"),
        ("entry_hashing", "whole-container hashes do not match"),
    ],
)
def test_run_score_blind_preflight_pre_access_stages_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    message_fragment: str,
) -> None:
    """Every pre-access authority stage raises a plain ValueError before any read."""
    env = _install_authority_environment(tmp_path, monkeypatch, fail_stage=stage)
    with pytest.raises(ValueError) as excinfo:
        run_score_blind_preflight(
            tmp_path / "configs/post_hoc_final_supplementary.yaml",
            SESSION0_FREEZE,
            AMENDMENT_FREEZE_DEFAULT_FOR_TESTS,
            synthetic_tests_passed=True,
            git_runner=lambda *_args: "",
        )
    # Pre-access failures are plain ValueErrors, not the post-access AccessStateError.
    assert not isinstance(excinfo.value, AccessStateError)
    # The exact injected stage is the one that fails closed.
    assert message_fragment in str(excinfo.value)
    assert "sample_id" not in str(excinfo.value)
    # Labels, dates, QC, and scores remain unread when authority fails.
    assert env["participant_calls"] == []
    assert not (tmp_path / SCORE_BLIND_PREFLIGHT_RELATIVE_PATH).exists()
    assert not list(tmp_path.rglob("*.tmp")) and not list(tmp_path.rglob(".*.tmp"))


# --- Mode-specific preflight target semantics (correction B) --------------------------------------


def _synthetic_aggregate_counts() -> OrderedDict[str, Any]:
    """Return the frozen aggregate-count block accepted by the preflight schema."""
    return OrderedDict(
        [
            ("source_cohort_n", 1040),
            ("modeling_cohort_n", 972),
            ("train_n", 777),
            ("internal_test_n", 195),
            (
                "difficult_spectrum_by_view",
                OrderedDict(
                    [
                        (
                            view,
                            OrderedDict(
                                [
                                    ("n", 4),
                                    ("n_positive", 2),
                                    ("n_negative", 2),
                                    (
                                        "disease_category_counts",
                                        OrderedDict(
                                            [("0", 0), ("1", 1), ("2", 1), ("3", 1), ("4", 1), ("5", 0)]
                                        ),
                                    ),
                                ]
                            ),
                        )
                        for view in ("oof", "test")
                    ]
                ),
            ),
            (
                "date_feasibility",
                OrderedDict([("n_dates_overall", 40), ("n_dates_with_both_endpoint_classes", 34)]),
            ),
            (
                "qc5_reconciliation",
                OrderedDict(
                    [
                        ("n_qc5_evaluable_source_rows", 974),
                        ("n_strict_fail_among_evaluable", 298),
                        ("n_strict_pass_among_evaluable", 676),
                    ]
                ),
            ),
        ]
    )


def _synthetic_preflight_payload(container_rel: str, container_hash: str, freeze_commit: str) -> OrderedDict[str, Any]:
    """Build a schema-valid passing preflight payload for a single frozen container."""
    hashes = OrderedDict(
        [
            ("config_sha256", "c" * 64),
            ("methods_sha256", "c" * 64),
            ("freeze_manifest_sha256", "c" * 64),
            ("frozen_input_whole_container_sha256", OrderedDict([(container_rel, container_hash)])),
        ]
    )
    freeze = OrderedDict(
        [
            ("freeze_commit_id", freeze_commit),
            ("exists", True),
            ("is_ancestor_of_head", True),
            ("parent_is_baseline_head", True),
            ("changed_paths_exact", True),
            ("config_blob_matches", True),
            ("methods_blob_matches", True),
            ("manifest_blob_matches", True),
            ("approval_evidence_matches", True),
            ("active_rule_hashes_match", True),
        ]
    )
    return build_score_blind_preflight(
        hashes=hashes,
        freeze_commit_validation=freeze,
        aggregate_counts=_synthetic_aggregate_counts(),
        frozen_hash_paths=[container_rel],
    )


def _authority_kwargs(tmp_path: Path, mode: str) -> dict[str, Any]:
    """Return canonical validate_pre_access_authority keyword arguments."""
    def clean_git(*args: str) -> str:
        if args and args[0] in {"rev-parse", "hash-object"}:
            return "a" * 40
        return ""

    return {
        "config_path": tmp_path / "configs/post_hoc_final_supplementary.yaml",
        "freeze_commit": SESSION0_FREEZE,
        "amendment_freeze_commit": AMENDMENT_FREEZE_DEFAULT_FOR_TESTS,
        "mode": mode,
        "git_runner": clean_git,
    }


def test_authority_preflight_mode_rejects_existing_preflight_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preflight mode refuses to run when the preflight target already exists."""
    _install_authority_environment(tmp_path, monkeypatch)
    target = tmp_path / SCORE_BLIND_PREFLIGHT_RELATIVE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        validate_pre_access_authority(**_authority_kwargs(tmp_path, "preflight"))


def test_authority_execute_mode_rejects_missing_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute mode refuses to run without a prior validated preflight."""
    _install_authority_environment(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        validate_pre_access_authority(**_authority_kwargs(tmp_path, "execute"))


def test_authority_execute_mode_rejects_invalid_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute mode rejects a preflight artifact that fails schema validation."""
    _install_authority_environment(tmp_path, monkeypatch)
    target = tmp_path / SCORE_BLIND_PREFLIGHT_RELATIVE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        validate_pre_access_authority(**_authority_kwargs(tmp_path, "execute"))


def test_authority_execute_mode_accepts_valid_preflight_with_targets_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute mode accepts a schema-valid preflight when Session 4 targets are absent."""
    env = _install_authority_environment(tmp_path, monkeypatch)
    payload = _synthetic_preflight_payload(env["container_rel"], "d" * 64, SESSION0_FREEZE)
    target = tmp_path / SCORE_BLIND_PREFLIGHT_RELATIVE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload), encoding="utf-8")
    result = validate_pre_access_authority(**_authority_kwargs(tmp_path, "execute"))
    assert isinstance(result, tuple) and len(result) == 3
    _effective, _entry_hashes, authority = result
    assert "session0_freeze" in authority and "amendment_freeze" in authority
    for path in SESSION_4_SCIENTIFIC_OUTPUT_PATHS:
        assert not (tmp_path / path).exists()
    assert env["participant_calls"] == []


# --- All-or-none Session 4 promotion failure injection (correction G) -----------------------------


def _install_execute_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Wire a fully mocked approved-execution environment rooted at ``tmp_path``."""
    contracts = {
        name: {
            "path": f"outputs/analysis/_synth_{name}.csv",
            "columns": ["analysis_id", "view"],
            "planned_row_count": 1,
        }
        for name in (
            "locked_score_metrics",
            "locked_score_composition",
            "qc5_correctness",
            "within_date_concordance",
        )
    }
    config = {
        "outputs": {
            "scientific_and_closure": contracts,
            "completion_logs": {"locked_scores": "outputs/logs/_synth_locked_scores.log"},
            "session_0": {
                "config": "configs/post_hoc_final_supplementary.yaml",
                "methods": "docs/modeling/final_post_hoc_supplementary_methods.md",
                "freeze_manifest": "outputs/analysis/posthoc_final_freeze_manifest.json",
            },
        },
        "frozen_inputs": {"whole_container_sha256": OrderedDict([("fixture", "a" * 64)])},
        "threshold": {"threshold_p_raw": 0.3418938786147648},
        "loaders": {"per_purpose_ordered_usecols": {}},
    }
    entry_hashes = OrderedDict([("fixture", "a" * 64)])

    def fake_authority(**_kwargs: Any) -> tuple[dict[str, Any], OrderedDict[str, str], OrderedDict[str, Any]]:
        return config, entry_hashes, OrderedDict()

    def fake_load(_config: Mapping[str, Any], tracker: AccessStateTracker) -> OrderedDict[str, Any]:
        for key in (
            "participant_level_label_values_read",
            "participant_level_date_values_read",
            "participant_level_qc_values_read",
            "participant_level_score_values_read",
        ):
            tracker.mark(key)
        inputs = {
            "y_true": np.array([0, 1]),
            "p_raw": np.array([0.1, 0.9]),
            "dates": np.array(["2025/1/1", "2025/1/1"]),
            "disease_category": np.array([1, 3]),
            "strict_pass": np.array([True, True]),
        }
        return OrderedDict([("oof", inputs), ("test", inputs)])

    def fake_framework(
        _views: Any, *, config: Any = None, on_first_aggregate: Any = None
    ) -> dict[str, Any]:
        if on_first_aggregate is not None:
            on_first_aggregate()
        frames = OrderedDict(
            (name, pd.DataFrame([{"analysis_id": name, "view": "oof"}])) for name in contracts
        )
        return {"frames": frames, "scores_read": True}

    monkeypatch.setattr("src.analysis.posthoc_final_locked_scores.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        "src.analysis.posthoc_final_locked_scores.validate_pre_access_authority", fake_authority
    )
    monkeypatch.setattr("src.analysis.posthoc_final_locked_scores._load_approved_views", fake_load)
    monkeypatch.setattr(
        "src.analysis.posthoc_final_locked_scores.execute_approved_framework", fake_framework
    )
    monkeypatch.setattr(
        "src.analysis.posthoc_final_locked_scores._verify_exit_hashes", lambda _config, _entry: None
    )
    monkeypatch.setattr("src.analysis.posthoc_final_locked_scores.sha256_file", lambda _path: "b" * 64)
    return {"config": config, "contracts": contracts}


@pytest.mark.parametrize(
    "fail",
    [
        "stage_metrics",
        "stage_composition",
        "stage_qc5",
        "stage_within",
        "promote_2",
        "promote_3",
        "promote_4",
        "log_construction",
        "log_promotion",
    ],
)
def test_execute_approved_all_or_none_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail: str,
) -> None:
    """Any staging, promotion, or completion-log failure leaves zero outputs and temporaries."""
    _install_execute_environment(tmp_path, monkeypatch)
    staging_stages = {"stage_metrics": 1, "stage_composition": 2, "stage_qc5": 3, "stage_within": 4}
    promote_stages = {"promote_2": 2, "promote_3": 3, "promote_4": 4, "log_promotion": 5}
    if fail in staging_stages:
        real_stage = locked_scores._stage_csv
        counter = {"n": 0}

        def staging(destination: Path, frame: Any, columns: Any, registry_entry: Any) -> Path:
            counter["n"] += 1
            if counter["n"] == staging_stages[fail]:
                raise ValueError("synthetic staging failure")
            return real_stage(destination, frame, columns, registry_entry)

        monkeypatch.setattr("src.analysis.posthoc_final_locked_scores._stage_csv", staging)
    elif fail in promote_stages:
        real_replace = os.replace
        counter = {"n": 0}

        def replace(source: Any, destination: Any) -> Any:
            counter["n"] += 1
            if counter["n"] == promote_stages[fail]:
                raise OSError("synthetic promotion failure")
            return real_replace(source, destination)

        monkeypatch.setattr(os, "replace", replace)
    else:

        def boom_log(**_kwargs: Any) -> str:
            raise ValueError("synthetic completion-log construction failure")

        monkeypatch.setattr("src.analysis.posthoc_final_locked_scores._completion_log", boom_log)

    with pytest.raises((ValueError, OSError)):
        execute_approved(
            tmp_path / "configs/post_hoc_final_supplementary.yaml",
            SESSION0_FREEZE,
            AMENDMENT_FREEZE_DEFAULT_FOR_TESTS,
        )
    analysis_dir = tmp_path / "outputs/analysis"
    logs_dir = tmp_path / "outputs/logs"
    assert not list(analysis_dir.glob("_synth_*.csv"))
    assert not (logs_dir / "_synth_locked_scores.log").exists()
    assert not list(tmp_path.rglob("*.tmp")) and not list(tmp_path.rglob(".*.tmp"))
