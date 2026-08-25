"""Focused tests for the aggregate LGIN disposition source-closure package."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts import run_lgin_postdiagnosis_disposition_source_closure as closure_cli
from src.analysis import lgin_postdiagnosis_disposition_source_closure as closure


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/lgin_postdiagnosis_disposition_source_closure.yaml"


def _config() -> dict[str, Any]:
    """Return an independent copy of the production contract."""
    return copy.deepcopy(closure.load_config(CONFIG_PATH))


def _primary_source() -> pd.DataFrame:
    """Return one synthetic primary aggregate using frozen literal anchors."""
    return pd.DataFrame(
        [
            {
                "rb": 0.9428571428571428,
                "U": 136.0,
                "p_two_sided": 0.000291789765473976,
                "median_p_cal_B": 0.8,
                "median_p_cal_C": 0.05,
                "q1_p_cal_B": 0.4,
                "q3_p_cal_B": 0.9,
                "q1_p_cal_C": 0.03,
                "q3_p_cal_C": 0.09,
                "n_above_0.5_B": 24,
                "n_above_0.5_C": 0,
                "n_B": 35,
                "n_C": 4,
                "directional_verification_passed": True,
                "separation_criterion_passed": True,
            }
        ]
    )


def _audit() -> dict[str, Any]:
    """Return the authorized aggregate-only LGIN audit projection."""
    return {
        "git_commit_hash": "09538ab9780f5fbfe4f9263c350c952716b612a6",
        "post_hoc_trigger_references": {
            "lgin_primary_view": "nested_cv_oof",
            "lgin_primary_view_authorization": "Erratum E012",
            "lgin_n_b_oof": 35,
            "lgin_n_c_oof": 4,
            "lgin_separation_criterion_passed": True,
            "lgin_directional_verification_passed": True,
            "lgin_rb": 0.9428571428571428,
            "lgin_mannwhitney_p_two_sided": 0.0002917897654739,
            "lgin_within_b_stability_levels_evaluated": 13,
            "lgin_within_b_stability_pass_overall": True,
            "lgin_test_view": "descriptive_sensitivity_only_due_to_n_c_test_zero",
            "lgin_n_b_test": 10,
            "lgin_n_c_test": 0,
        },
    }


def _small_lgin_frame() -> pd.DataFrame:
    """Return a synthetic aggregate-construction population with missingness."""
    return pd.DataFrame(
        {
            "collection_location": [1, 1, 1, 0, 0],
            "age": [10.0, 20.0, 30.0, 40.0, 50.0],
            "hp_infection": [1.0, 0.0, np.nan, 1.0, np.nan],
        }
    )


def _small_baseline_config() -> dict[str, Any]:
    """Return a two-variable synthetic baseline contract."""
    config = _config()
    config["anchors"]["oof"] = {
        "n_lgin": 5,
        "n_inpatient_treatment_proxy": 3,
        "n_outpatient_followup_proxy": 2,
    }
    config["baseline_variables"] = [
        {
            "variable_order": 1,
            "variable_key": "age",
            "source_column": "age",
            "variable_type": "continuous",
            "event_level": None,
            "inpatient_n_nonmissing": 3,
            "outpatient_n_nonmissing": 2,
            "difference_metric": "absolute_standardized_mean_difference",
        },
        {
            "variable_order": 2,
            "variable_key": "hp_infection",
            "source_column": "hp_infection",
            "variable_type": "binary",
            "event_level": "positive",
            "event_code": 1,
            "inpatient_n_nonmissing": 2,
            "outpatient_n_nonmissing": 1,
            "inpatient_event_count": 1,
            "outpatient_event_count": 1,
            "difference_metric": "absolute_proportion_difference",
        },
    ]
    return config


def _small_legacy_baseline() -> pd.DataFrame:
    """Return the independent two-row aggregate anchor."""
    age_difference = 25.0 / np.sqrt(75.0)
    return pd.DataFrame(
        [
            {
                "variable": "age",
                "mean_B": 20.0,
                "mean_C": 45.0,
                "smd": age_difference,
                "n_B_non_missing": 3,
                "n_C_non_missing": 2,
            },
            {
                "variable": "hp_infection",
                "mean_B": 0.5,
                "mean_C": 1.0,
                "smd": 0.5,
                "n_B_non_missing": 2,
                "n_C_non_missing": 1,
            },
        ]
    )


def _stability_source() -> pd.DataFrame:
    """Return a complete synthetic legacy stability inventory."""
    rows = []
    for _, source_variable, source_level, _, _ in _config()["stability_mapping"]:
        small = (source_variable, source_level) in {("family_history", "1"), ("hp_infection", "1")}
        n_in_level = 2 if source_variable == "family_history" and source_level == "1" else 3 if small else 5
        rows.append(
            {
                "variable_name": source_variable,
                "level_value": source_level,
                "n_in_level": n_in_level,
                "median_p_cal": np.nan if small else 0.75,
                "n_above_0_5": np.nan if small else 4.0,
                "omission_reason": f"n_in_level={n_in_level} < 5" if small else "",
            }
        )
    return pd.DataFrame(rows)


def _participant_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return synthetic projected metadata, sample index, and OOF membership."""
    identifiers = ["SYNTH-101", "SYNTH-102", "SYNTH-103", "SYNTH-104", "SYNTH-105"]
    metadata = pd.DataFrame(
        {
            "sample_id": identifiers,
            "disease_category": [2, 2, 2, 2, 1],
            "label": [0, 0, 0, 0, 0],
            "collection_location": [1, 1, 1, 0, 0],
            "age": [50.0, 60.0, 70.0, 55.0, 45.0],
            "sex": [0, 1, 1, 0, 1],
            "bmi": [20.0, 21.0, 22.0, 23.0, 24.0],
            "smoking": [0, 1, 0, 0, 1],
            "drinking": [0, 1, 0, 0, 1],
            "first_degree_family_history_of_gc": [0, 0, 1, 0, 0],
            "hp_infection": [0.0, 1.0, np.nan, 0.0, 1.0],
            "cea": [1.0, 2.0, 3.0, 4.0, 5.0],
            "ca199": [1.0, 2.0, 3.0, 4.0, 5.0],
            "ca724": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    sample_index = metadata.loc[:, ["sample_id", "label", "disease_category"]].copy()
    oof = pd.DataFrame({"sample_id": identifiers[:4], "y_true": [0, 0, 0, 0]})
    config = _config()
    config["anchors"]["oof"].update(
        n_lgin=4,
        n_inpatient_treatment_proxy=3,
        n_outpatient_followup_proxy=1,
    )
    return metadata, sample_index, oof, config


def test_production_configuration_freezes_identity_schemas_and_outputs() -> None:
    """Package renaming, schema drift, or output renumbering must fail."""
    config = closure.load_config(CONFIG_PATH)

    assert config["package_id"] == "lgin_postdiagnosis_disposition_source_closure"
    assert tuple(config["schemas"]["primary_estimability"]) == closure.PRIMARY_COLUMNS
    assert tuple(config["schemas"]["baseline_context"]) == closure.BASELINE_COLUMNS
    assert tuple(config["schemas"]["within_inpatient_stability"]) == closure.STABILITY_COLUMNS
    assert [record["path"] for record in config["outputs"].values()] == [
        "outputs/analysis/lgin_postdiagnosis_disposition_primary_estimability.csv",
        "outputs/analysis/lgin_postdiagnosis_disposition_baseline_context.csv",
        "outputs/analysis/lgin_postdiagnosis_disposition_within_inpatient_stability.csv",
        "outputs/analysis/lgin_postdiagnosis_disposition_source_closure_manifest.json",
    ]
    assert [row["variable_key"] for row in config["baseline_variables"]] == [
        "age",
        "sex",
        "bmi",
        "smoking",
        "drinking",
        "family_history",
        "hp_infection",
        "cea",
        "ca199",
        "ca724",
    ]
    assert config["proxy_mapping"]["raw_values"] == {
        1: "inpatient_treatment_proxy",
        0: "outpatient_followup_proxy",
    }
    assert config["anchors"]["oof"]["n_lgin"] == 39
    assert config["anchors"]["locked_internal_test"]["n_outpatient_followup_proxy"] == 0


def test_forbidden_sources_are_not_in_the_input_inventory() -> None:
    """Adding either participant-level legacy artifact to production inputs must fail."""
    config = _config()
    inventory = {record["path"] for record in config["scientific_inputs"]}
    assert inventory.isdisjoint(config["forbidden_inputs"])
    assert len(inventory) == 18


def test_primary_source_preserves_oof_precision_and_empty_test_statistics() -> None:
    """Test statistics must not leak into the non-estimable internal-test row."""
    result = closure.build_primary_estimability(_primary_source(), _audit(), _config())

    assert result.columns.tolist() == list(closure.PRIMARY_COLUMNS)
    assert result["analysis_view"].tolist() == ["training_nested_cv_oof", "locked_internal_test"]
    oof, test = result.iloc[0], result.iloc[1]
    assert oof["p_two_sided"] == 0.000291789765473976
    assert oof["rank_biserial_correlation"] == 0.9428571428571428
    assert oof["n_lgin"] == 39
    assert oof["n_inpatient_treatment_proxy"] == 35
    assert oof["n_outpatient_followup_proxy"] == 4
    assert bool(oof["directional_verification_passed"])
    assert bool(oof["separation_criterion_passed"])
    assert test["comparison_estimability_status"] == "not_estimable"
    assert test["nonestimable_reason"] == "outpatient_followup_proxy_n_zero"
    assert test["n_lgin"] == 10
    unavailable = list(closure.PRIMARY_COLUMNS[7:])
    assert test[unavailable].isna().all()


def test_primary_source_rejects_audit_disagreement_beyond_strict_tolerance() -> None:
    """A materially different audit P value must not be silently preferred."""
    audit = _audit()
    audit["post_hoc_trigger_references"]["lgin_mannwhitney_p_two_sided"] = 0.001
    with pytest.raises(ValueError, match="aggregate reconciliation"):
        closure.build_primary_estimability(_primary_source(), audit, _config())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lgin_primary_view", "test"),
        ("lgin_primary_view_authorization", "unapproved"),
        ("lgin_within_b_stability_levels_evaluated", 12),
        ("lgin_within_b_stability_pass_overall", False),
    ],
)
def test_primary_source_rejects_wrong_audit_identity(field: str, value: object) -> None:
    """Audit view, authority, and stability closure must retain their frozen identity."""
    audit = _audit()
    audit["post_hoc_trigger_references"][field] = value
    with pytest.raises(ValueError, match="aggregate reconciliation"):
        closure.build_primary_estimability(_primary_source(), audit, _config())


def test_continuous_and_binary_baseline_summaries_use_distinct_metrics() -> None:
    """Continuous SMD and binary proportion difference must never share one label."""
    result = closure.build_baseline_context(_small_lgin_frame(), _small_legacy_baseline(), _small_baseline_config())

    assert result.columns.tolist() == list(closure.BASELINE_COLUMNS)
    age, hp = result.iloc[0], result.iloc[1]
    assert age["inpatient_sd"] == 10.0
    assert age["inpatient_median"] == 20.0
    assert age["inpatient_q1"] == 15.0
    assert age["inpatient_q3"] == 25.0
    assert age["outpatient_sd"] == pytest.approx(np.sqrt(50.0))
    assert age["difference_metric"] == "absolute_standardized_mean_difference"
    assert age["absolute_difference"] == pytest.approx(25.0 / np.sqrt(75.0))
    assert pd.isna(age["inpatient_event_count"])
    assert hp["inpatient_n_nonmissing"] == 2
    assert hp["inpatient_n_missing"] == 1
    assert hp["inpatient_event_count"] == 1
    assert hp["inpatient_denominator_n"] == 2
    assert hp["inpatient_proportion"] == 0.5
    assert hp["outpatient_n_nonmissing"] == 1
    assert hp["outpatient_n_missing"] == 1
    assert hp["outpatient_event_count"] == 1
    assert hp["outpatient_denominator_n"] == 1
    assert hp["outpatient_proportion"] == 1.0
    assert hp["difference_metric"] == "absolute_proportion_difference"
    assert hp["absolute_difference"] == 0.5
    assert pd.isna(hp["inpatient_mean"])


def test_baseline_reconciliation_rejects_wrong_legacy_metric() -> None:
    """A changed frozen difference estimate must fail the aggregate reconciliation."""
    legacy = _small_legacy_baseline()
    legacy.loc[legacy["variable"] == "hp_infection", "smd"] = 0.25
    with pytest.raises(ValueError, match="aggregate reconciliation"):
        closure.build_baseline_context(_small_lgin_frame(), legacy, _small_baseline_config())


def test_binary_coding_and_expected_event_counts_fail_closed() -> None:
    """Invalid 0/1 coding or an event-count mismatch must not be coerced."""
    frame = _small_lgin_frame()
    frame.loc[0, "hp_infection"] = 2
    with pytest.raises(ValueError, match="Binary field coding"):
        closure.build_baseline_context(frame, _small_legacy_baseline(), _small_baseline_config())
    frame = _small_lgin_frame()
    config = _small_baseline_config()
    config["baseline_variables"][1]["inpatient_event_count"] = 2
    with pytest.raises(ValueError, match="event count"):
        closure.build_baseline_context(frame, _small_legacy_baseline(), config)


def test_participant_join_requires_complete_unique_matching_keys_without_disclosure() -> None:
    """Duplicate, missing, or unmatched keys must fail with sanitized errors."""
    metadata, sample_index, oof, config = _participant_frames()
    mutations = []
    duplicate = metadata.copy()
    duplicate.loc[1, "sample_id"] = duplicate.loc[0, "sample_id"]
    mutations.append((duplicate, sample_index, oof))
    missing = sample_index.copy()
    missing.loc[0, "sample_id"] = None
    mutations.append((metadata, missing, oof))
    unmatched = oof.copy()
    unmatched.loc[0, "sample_id"] = "SYNTH-UNMATCHED"
    mutations.append((metadata, sample_index, unmatched))
    for bad_metadata, bad_index, bad_oof in mutations:
        with pytest.raises(ValueError) as exc_info:
            closure.build_lgin_participant_context(bad_metadata, bad_index, bad_oof, config)
        message = str(exc_info.value)
        assert "SYNTH-" not in message
        assert "identifier" not in message.lower()


def test_participant_join_closes_oof_lgin_groups_and_binary_coding() -> None:
    """The strict OOF membership join must retain only coded LGIN proxy groups."""
    metadata, sample_index, oof, config = _participant_frames()
    result = closure.build_lgin_participant_context(metadata, sample_index, oof, config)
    assert len(result) == 4
    assert result["collection_location"].value_counts().to_dict() == {1: 3, 0: 1}
    assert "sample_id" not in result.columns
    assert set(result.columns) == {
        "collection_location",
        "age",
        "sex",
        "bmi",
        "smoking",
        "drinking",
        "family_history",
        "hp_infection",
        "cea",
        "ca199",
        "ca724",
    }
    metadata.loc[0, "sex"] = 2
    with pytest.raises(ValueError, match="Binary field coding"):
        closure.build_lgin_participant_context(metadata, sample_index, oof, config)


def test_participant_join_rejects_inconsistent_endpoint_without_disclosure() -> None:
    """A metadata/index endpoint conflict must fail without exposing join values."""
    metadata, sample_index, oof, config = _participant_frames()
    metadata.loc[0, "label"] = 1
    with pytest.raises(ValueError, match="alignment") as exc_info:
        closure.build_lgin_participant_context(metadata, sample_index, oof, config)
    assert "SYNTH-" not in str(exc_info.value)


def test_participant_validation_is_restricted_to_oof_lgin_baseline_rows() -> None:
    """Irrelevant non-OOF baseline missingness must not broaden the authorized population."""
    metadata, sample_index, oof, config = _participant_frames()
    metadata.loc[4, ["age", "sex", "bmi", "smoking", "drinking"]] = np.nan
    result = closure.build_lgin_participant_context(metadata, sample_index, oof, config)
    assert len(result) == 4


def test_hash_mismatch_precedes_every_participant_capable_callback(tmp_path: Path) -> None:
    """A bad pinned hash must halt before any participant-capable loader runs."""
    source = tmp_path / "source.csv"
    source.write_text("synthetic\n", encoding="utf-8")
    config = _config()
    config["scientific_inputs"] = [
        {
            "role": "synthetic",
            "path": "source.csv",
            "sha256": "0" * 64,
            "participant_capable": True,
        }
    ]
    calls: list[str] = []

    def forbidden(*args: object, **kwargs: object) -> pd.DataFrame:
        calls.append("called")
        raise RuntimeError("must not run")

    with pytest.raises(ValueError, match="hash mismatch"):
        closure.load_authorized_participant_context(
            tmp_path,
            config,
            metadata_loader=forbidden,
            sample_index_loader=forbidden,
            alignment_checker=forbidden,
            oof_reader=forbidden,
        )
    assert calls == []


def test_hash_inventory_opens_only_configured_synthetic_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hash verification must not probe forbidden or alternative paths."""
    source = tmp_path / "allowed.bin"
    source.write_bytes(b"allowed bytes\n")
    records = [
        {
            "role": "synthetic",
            "path": "allowed.bin",
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "participant_capable": False,
        }
    ]
    opened: list[Path] = []
    real_open = Path.open

    def tracked_open(self: Path, *args: object, **kwargs: object) -> Any:
        opened.append(self)
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)
    closure.verify_scientific_inputs(tmp_path, records)
    assert opened == [source]


def test_production_pinned_inventory_is_byte_identical() -> None:
    """Every configured frozen input must still match its independently pinned digest."""
    config = _config()
    observed = closure.verify_scientific_inputs(PROJECT_ROOT, config["scientific_inputs"])
    expected = OrderedDict((record["role"], record["sha256"]) for record in config["scientific_inputs"])
    assert observed == expected


def test_frozen_aggregate_loader_preserves_round_trip_float_text() -> None:
    """Default-parser loss of a frozen full-precision decimal must fail."""
    _, baseline, _, _ = closure._load_aggregate_sources(PROJECT_ROOT, _config())
    bmi_mean = baseline.loc[baseline["variable"] == "bmi", "mean_B"].iloc[0]
    ca199_mean = baseline.loc[baseline["variable"] == "ca199", "mean_B"].iloc[0]
    assert bmi_mean == 23.691428571428574
    assert ca199_mean == 11.252500000000001


@pytest.mark.parametrize(
    ("mutation", "expected_message"),
    [
        ("head", "HEAD"),
        ("tracked", "Tracked worktree"),
        ("staged", "Tracked worktree"),
        ("status", "Unexpected worktree"),
    ],
)
def test_execution_authority_rejects_unapproved_git_state(mutation: str, expected_message: str) -> None:
    """Wrong HEAD, tracked changes, or unapproved status entries must halt."""
    config = _config()
    responses = {
        ("rev-parse", "HEAD"): config["expected_git_head"],
        ("diff", "--name-only"): "",
        ("diff", "--cached", "--name-only"): "",
        ("status", "--short"): "?? .vscode/\n?? assets/",
    }
    if mutation == "head":
        responses[("rev-parse", "HEAD")] = "0" * 40
    elif mutation == "tracked":
        responses[("diff", "--name-only")] = "existing.py"
    elif mutation == "staged":
        responses[("diff", "--cached", "--name-only")] = "existing.py"
    else:
        responses[("status", "--short")] += "\n?? unexpected.txt"

    def fake_git(root: Path, *args: str) -> str:
        return responses[args]

    with pytest.raises(ValueError, match=expected_message):
        closure.validate_execution_authority(
            PROJECT_ROOT,
            config,
            git_output=fake_git,
            executable=Path.home() / "miniforge3/envs/gcvocs/bin/python",
        )


def test_stability_mapping_and_small_level_gate_are_exact() -> None:
    """Changing level semantics or retaining n<5 summaries must fail."""
    result = closure.build_stability_source(_stability_source(), _config())
    assert result.columns.tolist() == list(closure.STABILITY_COLUMNS)
    assert len(result) == 15
    assert result[["variable_key", "level_key"]].apply(tuple, axis=1).tolist() == [
        ("age", "lt_60"),
        ("age", "ge_60"),
        ("sex", "male"),
        ("sex", "female"),
        ("bmi", "lt_24"),
        ("bmi", "ge_24"),
        ("smoking", "no"),
        ("smoking", "yes"),
        ("drinking", "no"),
        ("drinking", "yes"),
        ("family_history", "no"),
        ("family_history", "yes"),
        ("hp_infection", "negative"),
        ("hp_infection", "positive"),
        ("hp_infection", "missing"),
    ]
    small = result.loc[result["estimability_status"] == "not_reported_small_level"]
    assert len(small) == 2
    assert small["n_in_level"].tolist() == [2, 3]
    assert small["median_p_cal"].isna().all()
    assert small["n_p_cal_gt_0_5"].isna().all()
    assert (small["nonestimable_reason"] == "n_in_level_below_5").all()
    assert (result["estimability_status"] == "estimable").sum() == 13


def test_serialization_is_deterministic_and_round_trips_numeric_payloads() -> None:
    """CSV serialization must be byte-stable with empty missing fields and exact floats."""
    primary = closure.build_primary_estimability(_primary_source(), _audit(), _config())
    first = closure.serialize_csv(primary, closure.PRIMARY_COLUMNS)
    second = closure.serialize_csv(primary.copy(deep=True), closure.PRIMARY_COLUMNS)
    assert first == second
    assert b"None" not in first and b"nan" not in first.lower()
    restored = pd.read_csv(io.BytesIO(first), float_precision="round_trip")
    assert restored.loc[0, "p_two_sided"] == primary.loc[0, "p_two_sided"]
    assert pd.isna(restored.loc[1, "p_two_sided"])
    assert first.endswith(b"\n") and b"\r\n" not in first


def test_output_collision_refusal_and_manifest_last_atomic_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing targets survive, and a successful publish exposes the manifest last."""
    targets = OrderedDict(
        [
            ("primary_estimability", tmp_path / "primary.csv"),
            ("baseline_context", tmp_path / "baseline.csv"),
            ("within_inpatient_stability", tmp_path / "stability.csv"),
            ("manifest", tmp_path / "manifest.json"),
        ]
    )
    payloads = OrderedDict((name, f"{name}\n".encode()) for name in targets)
    targets["primary_estimability"].write_bytes(b"sentinel\n")
    with pytest.raises(FileExistsError, match="output target"):
        closure.promote_serialized_outputs(targets, payloads)
    assert targets["primary_estimability"].read_bytes() == b"sentinel\n"
    targets["primary_estimability"].unlink()
    promoted: list[str] = []
    real_replace = os.replace

    def record_replace(source: Path, destination: Path) -> None:
        promoted.append(destination.name)
        real_replace(source, destination)

    monkeypatch.setattr(closure.os, "replace", record_replace)
    closure.promote_serialized_outputs(targets, payloads)
    assert promoted == ["primary.csv", "baseline.csv", "stability.csv", "manifest.json"]
    assert not list(tmp_path.glob(".*.tmp"))


def test_destination_local_temporary_collision_is_never_overwritten(tmp_path: Path) -> None:
    """A stale destination-local temporary must survive collision refusal."""
    targets = OrderedDict(
        [
            ("primary_estimability", tmp_path / "primary.csv"),
            ("baseline_context", tmp_path / "baseline.csv"),
            ("within_inpatient_stability", tmp_path / "stability.csv"),
            ("manifest", tmp_path / "manifest.json"),
        ]
    )
    payloads = OrderedDict((name, b"payload\n") for name in targets)
    stale = tmp_path / ".baseline.csv.tmp"
    stale.write_bytes(b"sentinel\n")
    with pytest.raises(FileExistsError, match="temporary output target"):
        closure.promote_serialized_outputs(targets, payloads)
    assert stale.read_bytes() == b"sentinel\n"
    assert not any(path.exists() for path in targets.values())


def test_promotion_failure_removes_every_new_final_and_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial replacement failure must leave no apparently complete product."""
    targets = OrderedDict(
        [
            ("primary_estimability", tmp_path / "primary.csv"),
            ("baseline_context", tmp_path / "baseline.csv"),
            ("within_inpatient_stability", tmp_path / "stability.csv"),
            ("manifest", tmp_path / "manifest.json"),
        ]
    )
    payloads = OrderedDict((name, b"payload\n") for name in targets)
    real_replace = os.replace
    replacements = 0

    def fail_second(source: Path, destination: Path) -> None:
        nonlocal replacements
        replacements += 1
        if replacements == 2:
            raise OSError("synthetic promotion failure")
        real_replace(source, destination)

    monkeypatch.setattr(closure.os, "replace", fail_second)
    with pytest.raises(OSError, match="synthetic promotion failure"):
        closure.promote_serialized_outputs(targets, payloads)
    assert not any(path.exists() for path in targets.values())
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize(
    "payload",
    [
        b"sample_id,metric\nSYNTH-101,1\n",
        b'{"path":"[PRIVATE_PATH_REDACTED]"}\n',
        b'{"collection_time":"2030-01-01"}\n',
        b'{"breath_filename":"private_record.csv"}\n',
        b'{"participant_prediction":0.8}\n',
    ],
)
def test_persisted_privacy_rejects_ids_paths_dates_filenames_and_predictions(payload: bytes) -> None:
    """Unsafe persisted content must fail without echoing the matched value."""
    with pytest.raises(ValueError, match="privacy") as exc_info:
        closure.validate_persisted_privacy({"synthetic": payload})
    assert payload.decode("utf-8").strip() not in str(exc_info.value)


def test_manifest_output_hash_validation_detects_changed_promoted_bytes(tmp_path: Path) -> None:
    """Manifest hashes must agree with all three promoted CSV byte streams."""
    targets = OrderedDict(
        [
            ("primary_estimability", tmp_path / "primary.csv"),
            ("baseline_context", tmp_path / "baseline.csv"),
            ("within_inpatient_stability", tmp_path / "stability.csv"),
        ]
    )
    for name, path in targets.items():
        path.write_bytes(f"{name}\n".encode())
    manifest = {
        "outputs": [
            {"role": name, "path": path.name, "sha256": closure.sha256_file(path)}
            for name, path in targets.items()
        ]
    }
    closure.validate_manifest_output_hashes(manifest, targets)
    targets["baseline_context"].write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="output hash"):
        closure.validate_manifest_output_hashes(manifest, targets)


def test_safe_products_cannot_contain_synthetic_join_keys_or_rows() -> None:
    """Participant-capable synthetic fields must disappear before serialization."""
    metadata, sample_index, oof, config = _participant_frames()
    lgin = closure.build_lgin_participant_context(metadata, sample_index, oof, config)
    baseline_config = _small_baseline_config()
    baseline = closure.build_baseline_context(_small_lgin_frame(), _small_legacy_baseline(), baseline_config)
    payload = closure.serialize_csv(baseline, closure.BASELINE_COLUMNS)
    closure.validate_persisted_privacy({"baseline": payload})
    assert b"sample_id" not in payload
    assert all(value.encode() not in payload for value in metadata["sample_id"])
    assert "collection_location" in lgin.columns
    assert "collection_location" not in baseline.columns


def test_manifest_candidate_round_trips_with_order_and_numeric_mapping_keys() -> None:
    """JSON key conversion must not change the deterministic manifest payload."""
    config = _config()
    observed = OrderedDict((record["role"], record["sha256"]) for record in config["scientific_inputs"])
    csv_payloads = OrderedDict((name, f"{name}\n".encode()) for name in closure.OUTPUT_ORDER[:3])
    manifest = closure._build_manifest(
        root=PROJECT_ROOT,
        config=config,
        git_head=config["expected_git_head"],
        observed_hashes=observed,
        csv_payloads=csv_payloads,
    )
    restored = json.loads(closure.serialize_manifest(manifest), object_pairs_hook=OrderedDict)
    assert restored == manifest


def test_manifest_serialization_is_compact_deterministic_and_host_independent() -> None:
    """Runtime timestamps and absolute host paths must never enter deterministic provenance."""
    payload = OrderedDict([("schema_version", "synthetic.v1"), ("status", "closed")])
    assert closure.serialize_manifest(payload) == b'{"schema_version":"synthetic.v1","status":"closed"}\n'
    with pytest.raises(ValueError, match="privacy"):
        closure.serialize_manifest({"generated_at_utc": "2030-01-01T00:00:00Z"})
    with pytest.raises(ValueError, match="privacy"):
        closure.serialize_manifest({"path": "/tmp/private"})


def test_cli_failure_is_compact_and_does_not_echo_exception_content(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A fail-closed CLI error must not emit a traceback, path, or private value."""

    def fail(_config_path: Path) -> OrderedDict[str, Any]:
        raise ValueError("SYNTH-PRIVATE [PRIVATE_PATH_REDACTED]")

    monkeypatch.setattr(closure_cli, "execute_approved", fail)
    exit_code = closure_cli.main(["--config", "synthetic.yaml", "--execute-approved"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == "Aggregate source closure failed closed.\n"
