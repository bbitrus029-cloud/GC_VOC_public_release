"""Synthetic tests for the eFigure 4 aggregate preprocessing-stage replay."""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = PROJECT_ROOT / "configs" / "efigure4_preprocessing_stage_aggregate_replay.yaml"
ENVIRONMENT_LOCK_PATH = PROJECT_ROOT / "configs" / "efigure4_preprocessing_stage_aggregate_replay_environment_lock.json"
ORIGINAL_PLAN_SHA256 = "33381038e69e153a7b4cf401535c50b8a5123de78d7d2a16644804d6a532515e"
RECOVERY_V2_PLAN_SHA256 = "feacd76d706f9c19c632d4336efa42a17691b1a4fbe40b75759daabf8369eebd"
RECOVERY_V3_PLAN_SHA256 = "487e5e69b5282dc2188e4b85467f736546491527076351411654125e5e4d57ca"
APPROVED_PLAN_SHA256 = "302f96e626f60463740330a38bcc1c5962543cf5c4f84ee7a32ba942b6d3529d"
RECOVERY_PLAN_SHA256 = APPROVED_PLAN_SHA256
REPORTING_CORRECTION_SHA256 = "ba0ea1b226a94ea476744ec7e8f6b047b546634d766e4bdca01792dbbca8a15c"
NUMERICAL_ENVIRONMENT_CORRECTION_SHA256 = "225aff9168b0e4f13e25cbbf6aa5f66ddc151e82a0e28e056cc0d307e917f83b"
CHANNEL_METADATA_CORRECTION_SHA256 = "a1d208e2a2d53faaa931ffa4f99b95dc9690e62a7ff4c55704210d293612f91b"
ENVIRONMENT_LOCK_SHA256 = "cf3e23de96b76536d6b906e24c2ffbaeb767a70d0940c2d483ac4f2200706cad"
SOURCE_LEDGER_SHA256 = "e6a91ad784c41e2781c03947562f6bbf78d6cc33a8d0551245cb7eb9c54e04f8"
FINAL_LEDGER_SHA256 = "7b11f59909a6e2f5b1bf1f8f49cba060b2e0b3d3e1b2b55f0da2eecfbcb30334"
EXPECTED_IMPLEMENTATION_HASHES = {
    "config_sha256": "106a16978edc134ef72984d23a07aebe4a3de7ea3f8c81e4f5622eaa1c78f0b5",
    "module_sha256": "c27b5d33d966251946d7d8731fc0e415d833bf6bae68aa8321e527ad5e28bc1d",
    "script_sha256": "5ebf84d3ce8ebbdf7455360113491267293b9acb18d9f2f2d9934a4152a8aec4",
}

EXPECTED_FAILURE_CODES = {
    "AUTHORIZATION_PLAN_IMPLEMENTATION_HASH_VALIDATION_FAILURE",
    "GIT_AUTHORITY_OUTPUT_STATE_VALIDATION_FAILURE",
    "ENVIRONMENT_IDENTITY_VALIDATION_FAILURE",
    "FROZEN_SOURCE_IDENTITY_CAPTURE_FAILURE",
    "FIRST_MINIMAL_COHORT_TABLE_LOADING_ALIGNMENT_FAILURE",
    "FIRST_RAW_COHORT_ENTRY_DIGEST_FAILURE",
    "FIRST_ENVIRONMENTAL_MODEL_DESERIALIZATION_FAILURE",
    "FIRST_RAW_SIGNAL_CONTAINMENT_LOADING_FAILURE",
    "FIRST_ENVIRONMENTAL_SUMMARY_RECONCILIATION_FAILURE",
    "FIRST_SAVITZKY_GOLAY_SMOOTHING_FAILURE",
    "FIRST_ENVIRONMENTAL_COMPENSATION_FAILURE",
    "FIRST_FRACTIONAL_BASELINE_CORRECTION_FAILURE",
    "FIRST_COMMON_TRIMMING_PADDING_RECONCILIATION_FAILURE",
    "FIRST_FROZEN_PIPELINE_A_LOADING_FAILURE",
    "FIRST_STAGE4_RECONCILIATION_FAILURE",
    "FIRST_AGGREGATION_FAILURE",
    "FIRST_RAW_COHORT_EXIT_DIGEST_FAILURE",
    "SECOND_MINIMAL_COHORT_TABLE_LOADING_ALIGNMENT_FAILURE",
    "SECOND_RAW_COHORT_ENTRY_DIGEST_FAILURE",
    "SECOND_ENVIRONMENTAL_MODEL_DESERIALIZATION_FAILURE",
    "SECOND_RAW_SIGNAL_CONTAINMENT_LOADING_FAILURE",
    "SECOND_ENVIRONMENTAL_SUMMARY_RECONCILIATION_FAILURE",
    "SECOND_SAVITZKY_GOLAY_SMOOTHING_FAILURE",
    "SECOND_ENVIRONMENTAL_COMPENSATION_FAILURE",
    "SECOND_FRACTIONAL_BASELINE_CORRECTION_FAILURE",
    "SECOND_COMMON_TRIMMING_PADDING_RECONCILIATION_FAILURE",
    "SECOND_FROZEN_PIPELINE_A_LOADING_FAILURE",
    "SECOND_STAGE4_RECONCILIATION_FAILURE",
    "SECOND_AGGREGATION_FAILURE",
    "SECOND_RAW_COHORT_EXIT_DIGEST_FAILURE",
    "CROSS_COMPUTATION_DETERMINISM_RAW_DIGEST_COMPARISON_FAILURE",
    "FINAL_SOURCE_IDENTITY_AUTHORITY_REVALIDATION_FAILURE",
    "AUDIT_PAYLOAD_CONSTRUCTION_FAILURE",
    "SERIALIZED_PRIVACY_COMPLETE_READ_BACK_VALIDATION_FAILURE",
    "FIRST_CANDIDATE_WRITING_VALIDATION_FAILURE",
    "SECOND_CANDIDATE_WRITING_VALIDATION_FAILURE",
    "CANDIDATE_COMPARISON_FAILURE",
    "ATOMIC_PROMOTION_FINAL_DIRECTORY_VALIDATION_FAILURE",
    "UNCLASSIFIED_FAILURE",
}


def _import_analysis() -> Any:
    """Import the replay module lazily so the initial TDD failure is explicit."""
    try:
        from src.analysis import efigure4_preprocessing_stage_aggregate_replay as analysis
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.fail(f"eFigure 4 replay implementation is missing: {exc}", pytrace=False)
    return analysis


def test_contract_has_exact_frozen_values_and_stage_order() -> None:
    """The runtime contract should preserve every reviewed scientific constant."""
    analysis = _import_analysis()

    contract = analysis.load_and_validate_contract(CONTRACT_PATH, APPROVED_PLAN_SHA256)

    assert contract["analysis_id"] == "efigure4_preprocessing_stage_aggregate_replay"
    assert contract["analysis_classification"] == (
        "post_freeze_result_blind_plan_orthogonal_descriptive_reporting_support"
    )
    assert contract["approved_plan"]["sha256"] == APPROVED_PLAN_SHA256
    assert contract["original_plan"]["sha256"] == ORIGINAL_PLAN_SHA256
    assert contract["runtime_authority"]["expected_branch"] == "main"
    assert contract["runtime_authority"]["expected_head"] == "098d6d5d65d9fb935fa97a30d1353a9694a0dc3a"
    assert contract["runtime_authority"]["preprocessing_frozen_commit"] == (
        "622cb3664774b3940f7bcf092004cfff62d06822"
    )
    assert contract["population"] == {
        "expected_source_count": 1040,
        "expected_final_count": 972,
        "expected_excluded_count": 68,
        "expected_source_fallback_count": 60,
        "expected_final_fallback_count": 54,
        "expected_excluded_fallback_count": 6,
        "expected_padded_count": 41,
        "final_cohort_selection_rule": "in_final_output == True",
        "canonical_order": "ascending_sample_id",
    }
    assert contract["channels"] == [f"S{index}" for index in range(1, 12)]
    assert contract["response_axis"] == {
        "name": "response_frame",
        "first": 0,
        "last": 959,
        "clip_length": 960,
        "convert_to_seconds": False,
    }
    assert contract["padding_relative_deficit_threshold"] == 0.0125
    assert [stage["id"] for stage in contract["stages"]] == [
        "raw_response_window",
        "savitzky_golay_smoothed_response_window",
        "environment_compensated_response_window",
        "fractional_baseline_corrected_response_window",
    ]
    assert [stage["unit"] for stage in contract["stages"]] == [
        "recorded_signal_unit",
        "recorded_signal_unit",
        "recorded_signal_unit",
        "fractional_change_from_compensated_baseline",
    ]
    assert contract["numpy_quantile_method"] == "linear"
    assert contract["reconciliation"] == {"rtol": 0.0, "atol": 1.0e-12, "equal_nan": False}
    assert contract["environmental_summary_reconciliation"] == {
        "rtol": 0.0,
        "atol": 1.0e-12,
        "equal_nan": False,
    }
    assert contract["serialization"]["csv_float_format"] == "%.17g"
    assert contract["outputs"]["expected_aggregate_rows"] == 42240
    assert contract["no_stochastic_operations"] is True
    assert contract["output_status"] == "executed_aggregate_only_pending_independent_review"
    assert contract["figure_design_authorization"] is False
    assert contract["release_authorization"] is False


def test_recovery_v4_environment_lock_has_exact_schema_and_six_substitutions() -> None:
    """The canonical lock should bind the complete ledger and exact local-archive semantics."""
    analysis = _import_analysis()

    lock = analysis.load_and_validate_environment_lock(ENVIRONMENT_LOCK_PATH, ENVIRONMENT_LOCK_SHA256)

    assert set(lock) == {
        "schema_version",
        "platform",
        "subdirectory",
        "environment_role",
        "environment_name",
        "source_ledger_sha256",
        "final_ledger_sha256",
        "explicit_local_archive_channel_semantics",
        "packages",
        "critical_numerical_builds",
        "critical_cached_archives",
    }
    assert lock["schema_version"] == 2
    assert type(lock["schema_version"]) is int
    assert lock["source_ledger_sha256"] == SOURCE_LEDGER_SHA256
    assert lock["final_ledger_sha256"] == FINAL_LEDGER_SHA256
    assert lock["explicit_local_archive_channel_semantics"] == (
        "unknown_channel_expected_only_for_six_explicit_local_sha256_locked_archives"
    )
    assert len(lock["packages"]) == 324
    assert len({item["name"] for item in lock["packages"]}) == 324
    assert {item["name"] for item in lock["packages"] if item["channel"] == "<unknown>"} == {
        "scipy",
        "libblas",
        "libcblas",
        "liblapack",
        "mkl",
        "onemkl-license",
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "schema_version",
        "missing_key",
        "additional_key",
        "misspelled_key",
        "repodata_record_metadata",
        "source_digest",
        "final_digest",
        "channel_semantics",
    ],
)
def test_recovery_v4_environment_lock_rejects_top_level_contract_tamper(tmp_path: Path, mutation: str) -> None:
    """Every top-level lock-schema or frozen-scalar mutation must fail closed."""
    analysis = _import_analysis()
    payload = json.loads(ENVIRONMENT_LOCK_PATH.read_text(encoding="utf-8"))
    if mutation == "schema_version":
        payload["schema_version"] = "2"
    elif mutation == "missing_key":
        payload.pop("platform")
    elif mutation == "additional_key":
        payload["unexpected"] = "value"
    elif mutation == "misspelled_key":
        payload["environmentRole"] = payload.pop("environment_role")
    elif mutation == "repodata_record_metadata":
        payload["repodata_record.json"] = {"channel": "mutable-transport-metadata"}
    elif mutation == "source_digest":
        payload["source_ledger_sha256"] = "0" * 64
    elif mutation == "final_digest":
        payload["final_ledger_sha256"] = "0" * 64
    else:
        payload["explicit_local_archive_channel_semantics"] = "unreviewed"
    altered = tmp_path / "lock.json"
    altered.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        analysis.load_and_validate_environment_lock(altered, analysis.sha256_file(altered))


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_package",
        "extra_package",
        "duplicate_package",
        "wrong_version",
        "wrong_build",
        "wrong_subdirectory",
        "unknown_on_retained_package",
        "wrong_substitution_channel",
        "wrong_archive_hash",
        "nested_local_url",
        "nested_absolute_path",
    ],
)
def test_recovery_v4_environment_lock_rejects_package_or_archive_tamper(tmp_path: Path, mutation: str) -> None:
    """Ledger identity includes every package field and each substituted archive hash."""
    analysis = _import_analysis()
    payload = json.loads(ENVIRONMENT_LOCK_PATH.read_text(encoding="utf-8"))
    packages = payload["packages"]
    scipy = next(item for item in packages if item["name"] == "scipy")
    retained = next(item for item in packages if item["name"] == "numpy")
    if mutation == "missing_package":
        packages.pop()
    elif mutation == "extra_package":
        packages.append(
            {
                "name": "synthetic-extra",
                "version": "1",
                "build": "0",
                "channel": "conda-forge",
                "subdirectory": "linux-64",
            }
        )
    elif mutation == "duplicate_package":
        packages.append(copy.deepcopy(packages[0]))
    elif mutation == "wrong_version":
        scipy["version"] = "1.17.2"
    elif mutation == "wrong_build":
        scipy["build"] = "py311hbe70eeb_1"
    elif mutation == "wrong_subdirectory":
        scipy["subdirectory"] = "noarch"
    elif mutation == "unknown_on_retained_package":
        retained["channel"] = "<unknown>"
    elif mutation == "wrong_substitution_channel":
        scipy["channel"] = "conda-forge"
    elif mutation == "wrong_archive_hash":
        payload["critical_cached_archives"][0]["sha256"] = "0" * 64
    elif mutation == "nested_local_url":
        scipy["url"] = "file:///private/cache/archive.conda"
    else:
        payload["critical_cached_archives"][0]["path"] = "/private/cache/archive.conda"
    altered = tmp_path / "lock.json"
    altered.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        analysis.load_and_validate_environment_lock(altered, analysis.sha256_file(altered))


def _synthetic_conda_identity() -> dict[str, Any]:
    """Return the exact reviewed Conda construction-tool identity without invoking Conda."""
    return {
        "conda_executable": "/approved/conda",
        "conda_version_output": "conda 26.1.0",
        "base_conda_package": {
            "name": "conda",
            "version": "26.1.0",
            "build": "py313h78bf25f_0",
            "channel": "conda-forge",
            "subdirectory": "linux-64",
        },
    }


def test_exact_conda_construction_tool_identity_is_accepted() -> None:
    """The reviewed executable, version output, and unique base record should pass together."""
    analysis = _import_analysis()
    identity = _synthetic_conda_identity()

    analysis.validate_conda_tool_identity(
        identity,
        resolved_executable="/approved/conda",
        version_output="conda 26.1.0\n",
        base_records=[copy.deepcopy(identity["base_conda_package"])],
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "executable",
        "version_output",
        "version",
        "build",
        "channel",
        "subdirectory",
        "missing_record",
        "duplicate_record",
    ],
)
def test_conda_construction_tool_identity_rejects_every_mismatch(mutation: str) -> None:
    """A construction-tool mismatch must stop before any environment or data operation."""
    analysis = _import_analysis()
    identity = _synthetic_conda_identity()
    resolved_executable = "/approved/conda"
    version_output = "conda 26.1.0\n"
    records = [copy.deepcopy(identity["base_conda_package"])]
    if mutation == "executable":
        resolved_executable = "/unapproved/conda"
    elif mutation == "version_output":
        version_output = "conda 26.0.0\n"
    elif mutation in {"version", "build", "channel", "subdirectory"}:
        records[0][mutation] = "wrong"
    elif mutation == "missing_record":
        records = []
    else:
        records.append(copy.deepcopy(records[0]))

    with pytest.raises(ValueError):
        analysis.validate_conda_tool_identity(
            identity,
            resolved_executable=resolved_executable,
            version_output=version_output,
            base_records=records,
        )


def test_contract_rejects_altered_plan_hash(tmp_path: Path) -> None:
    """A contract carrying any other approved-plan hash must fail closed."""
    analysis = _import_analysis()
    payload = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    payload["approved_plan"]["sha256"] = "0" * 64
    altered_path = tmp_path / "contract.yaml"
    altered_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="approved plan"):
        analysis.load_and_validate_contract(altered_path, APPROVED_PLAN_SHA256)


def test_recovery_contract_binds_complete_v4_chain_and_explicit_fallback_denominators() -> None:
    """Recovery-v4 must bind immutable history, corrections, lock, and fallback denominators."""
    analysis = _import_analysis()

    contract = analysis.load_and_validate_contract(CONTRACT_PATH, RECOVERY_PLAN_SHA256)

    assert contract["approved_plan"] == {
        "path": "docs/plans/efigure4_preprocessing_stage_aggregate_replay_recovery_v4_plan.md",
        "sha256": RECOVERY_PLAN_SHA256,
    }
    assert contract["original_plan"] == {
        "path": "docs/plans/efigure4_preprocessing_stage_aggregate_replay_results_free_plan.md",
        "sha256": ORIGINAL_PLAN_SHA256,
    }
    assert contract["reporting_correction"] == {
        "path": "docs/preprocessing/preprocessing_fallback_denominator_reporting_correction.md",
        "sha256": REPORTING_CORRECTION_SHA256,
        "byte_size": 3374,
    }
    assert contract["recovery_v2_plan"]["sha256"] == RECOVERY_V2_PLAN_SHA256
    assert contract["recovery_v3_plan"]["sha256"] == RECOVERY_V3_PLAN_SHA256
    assert contract["numerical_environment_correction"]["sha256"] == NUMERICAL_ENVIRONMENT_CORRECTION_SHA256
    assert contract["channel_metadata_correction"]["sha256"] == CHANNEL_METADATA_CORRECTION_SHA256
    assert contract["environment_lock"] == {
        "path": "configs/efigure4_preprocessing_stage_aggregate_replay_environment_lock.json",
        "sha256": ENVIRONMENT_LOCK_SHA256,
    }
    assert contract["population"] == {
        "expected_source_count": 1040,
        "expected_final_count": 972,
        "expected_excluded_count": 68,
        "expected_source_fallback_count": 60,
        "expected_final_fallback_count": 54,
        "expected_excluded_fallback_count": 6,
        "expected_padded_count": 41,
        "final_cohort_selection_rule": "in_final_output == True",
        "canonical_order": "ascending_sample_id",
    }
    assert "expected_fallback_count" not in contract["population"]


def test_recovery_contract_rejects_obsolete_ambiguous_fallback_key(tmp_path: Path) -> None:
    """The superseded single-denominator fallback key must fail contract loading."""
    analysis = _import_analysis()
    payload = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    payload["population"]["expected_fallback_count"] = 60
    altered_path = tmp_path / "contract.yaml"
    altered_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="fallback|population"):
        analysis.load_and_validate_contract(altered_path, RECOVERY_PLAN_SHA256)


def test_recovery_contract_declares_exact_uncollapsed_worktree_and_output_namespaces() -> None:
    """Recovery-v4 should bind the exact uncollapsed worktree with no concurrent exception."""
    analysis = _import_analysis()
    contract = analysis.load_and_validate_contract(CONTRACT_PATH, RECOVERY_PLAN_SHA256)

    assert "concurrent_worktree_exception" not in contract["runtime_authority"]
    assert contract["runtime_authority"]["allowed_worktree_entries"] == [
        ".vscode/settings.json",
        "configs/efigure4_preprocessing_stage_aggregate_replay.yaml",
        "configs/efigure4_preprocessing_stage_aggregate_replay_environment_lock.json",
        "docs/preprocessing/preprocessing_fallback_denominator_reporting_correction.md",
        "docs/preprocessing/preprocessing_stage4_numerical_environment_build_drift_correction.md",
        "docs/preprocessing/preprocessing_stage4_conda_explicit_archive_channel_metadata_correction.md",
        "scripts/run_efigure4_preprocessing_stage_aggregate_replay.py",
        "src/analysis/efigure4_preprocessing_stage_aggregate_replay.py",
        "tests/test_efigure4_preprocessing_stage_aggregate_replay.py",
    ]
    assert contract["outputs"]["final_directory"] == (
        "outputs/analysis/efigure4_preprocessing_stage_aggregate_replay"
    )
    assert contract["outputs"]["candidate_one_directory"] == (
        "outputs/analysis/.efigure4_preprocessing_stage_aggregate_replay.candidate-one"
    )
    assert contract["outputs"]["candidate_two_directory"] == (
        "outputs/analysis/.efigure4_preprocessing_stage_aggregate_replay.candidate-two"
    )


def _load_contract() -> dict[str, Any]:
    """Load the real results-free runtime contract for synthetic tests."""
    analysis = _import_analysis()
    return analysis.load_and_validate_contract(CONTRACT_PATH, APPROVED_PLAN_SHA256)


def _authority_snapshot(contract: dict[str, Any]) -> dict[str, Any]:
    """Build a matching read-only authority snapshot."""
    return {
        "branch": contract["runtime_authority"]["expected_branch"],
        "head": contract["runtime_authority"]["expected_head"],
        "sole_parent": contract["runtime_authority"]["expected_parent"],
        "preprocessing_frozen_commit": contract["runtime_authority"]["preprocessing_frozen_commit"],
        "preprocessing_frozen_is_ancestor": True,
        "input_blobs": {name: payload["git_blob"] for name, payload in contract["inputs"].items()},
        "frozen_diff_is_empty": True,
        "worktree_entries": contract["runtime_authority"]["allowed_worktree_entries"],
        "final_output_absent": True,
        "candidate_one_output_absent": True,
        "candidate_two_output_absent": True,
        "approved_plan_sha256": APPROVED_PLAN_SHA256,
        "original_plan_sha256": ORIGINAL_PLAN_SHA256,
        "recovery_v2_plan_sha256": RECOVERY_V2_PLAN_SHA256,
        "recovery_v3_plan_sha256": RECOVERY_V3_PLAN_SHA256,
        "reporting_correction_sha256": REPORTING_CORRECTION_SHA256,
        "numerical_environment_correction_sha256": NUMERICAL_ENVIRONMENT_CORRECTION_SHA256,
        "channel_metadata_correction_sha256": CHANNEL_METADATA_CORRECTION_SHA256,
        "environment_lock_sha256": ENVIRONMENT_LOCK_SHA256,
    }


def _validate_live_repository_lifecycle_state(
    analysis: Any,
    contract: dict[str, Any],
    actual: dict[str, Any],
    *,
    implementation_hashes: dict[str, str],
    project_root: Path,
) -> None:
    """Validate exact pre-execution or post-promotion repository state for live tests."""
    if implementation_hashes != EXPECTED_IMPLEMENTATION_HASHES:
        raise ValueError("Bound implementation hash differs in the live lifecycle state.")
    root = Path(project_root).resolve(strict=True)
    final_relative = Path(contract["outputs"]["final_directory"])
    if final_relative.is_absolute():
        raise ValueError("Final output path is not project-relative.")
    final_path = root / final_relative
    if not final_path.exists() and not final_path.is_symlink():
        analysis.validate_authority_snapshot(contract, actual)
        return

    if final_path.is_symlink() or not final_path.is_dir():
        raise ValueError("Post-promotion final namespace is not a real directory.")
    allowed_filenames = tuple(contract["outputs"]["allowed_filenames"])
    entries = list(final_path.iterdir())
    if (
        len(entries) != len(allowed_filenames)
        or {entry.name for entry in entries} != set(allowed_filenames)
        or any(entry.is_symlink() or not entry.is_file() for entry in entries)
    ):
        raise ValueError("Post-promotion final namespace contents differ.")

    candidate_paths = [
        root / contract["outputs"]["candidate_one_directory"],
        root / contract["outputs"]["candidate_two_directory"],
    ]
    if any(path.exists() or path.is_symlink() for path in candidate_paths):
        raise ValueError("A post-promotion candidate namespace exists.")
    if actual.get("final_output_absent") is not False:
        raise ValueError("Post-promotion final namespace state differs.")
    if actual.get("candidate_one_output_absent") is not True or actual.get("candidate_two_output_absent") is not True:
        raise ValueError("Post-promotion candidate snapshot differs.")

    expected_worktree = [
        *contract["runtime_authority"]["allowed_worktree_entries"],
        *[f"{final_relative.as_posix()}/{filename}" for filename in allowed_filenames],
    ]
    observed_worktree = actual.get("worktree_entries")
    if (
        not isinstance(observed_worktree, (list, tuple))
        or len(observed_worktree) != len(set(observed_worktree))
        or set(observed_worktree) != set(expected_worktree)
    ):
        raise ValueError("Post-promotion worktree scope differs.")

    pre_execution_view = copy.deepcopy(actual)
    pre_execution_view["final_output_absent"] = True
    pre_execution_view["worktree_entries"] = list(contract["runtime_authority"]["allowed_worktree_entries"])
    analysis.validate_authority_snapshot(contract, pre_execution_view)


def test_authority_snapshot_rejects_altered_head_or_input_blob() -> None:
    """HEAD and every declared input blob must match exactly, never semantically."""
    analysis = _import_analysis()
    contract = _load_contract()
    matching = _authority_snapshot(contract)
    analysis.validate_authority_snapshot(contract, matching)

    altered_head = {**matching, "head": "0" * 40}
    with pytest.raises(ValueError, match="HEAD"):
        analysis.validate_authority_snapshot(contract, altered_head)

    altered_blobs = {**matching, "input_blobs": dict(matching["input_blobs"])}
    altered_blobs["input_blobs"]["pipeline_a"] = "0" * 40
    with pytest.raises(ValueError, match="blob"):
        analysis.validate_authority_snapshot(contract, altered_blobs)


def test_exact_worktree_scope_rejects_collapsed_or_committed_efigure9_entries() -> None:
    """Only exact all-untracked paths pass; collapsed or committed eFigure 9 paths do not."""
    analysis = _import_analysis()
    contract = analysis.load_and_validate_contract(CONTRACT_PATH, RECOVERY_PLAN_SHA256)
    matching = _authority_snapshot(contract)
    analysis.validate_authority_snapshot(contract, matching)
    collapsed = {
        **matching,
        "worktree_entries": [
            ".vscode/" if path == ".vscode/settings.json" else path for path in matching["worktree_entries"]
        ],
    }
    with pytest.raises(ValueError, match="Worktree"):
        analysis.validate_authority_snapshot(contract, collapsed)
    with_efigure9 = {
        **matching,
        "worktree_entries": [
            *matching["worktree_entries"],
            "configs/efigure9_disease_category_prediction_matrix.yaml",
        ],
    }
    with pytest.raises(ValueError, match="Worktree"):
        analysis.validate_authority_snapshot(contract, with_efigure9)


@pytest.mark.parametrize(
    "unexpected_path",
    [
        "tests/test_efigure9_disease_category_prediction_matrix_extra.py",
        "outputs/analysis/efigure9_disease_category_prediction_matrix_extra/result.json",
        "docs/efigure4_unapproved_lifecycle_change.md",
    ],
)
def test_similarly_named_or_unapproved_worktree_entries_remain_rejected(unexpected_path: str) -> None:
    """The concurrent exception must not broaden to similar names or other eFigure 4 changes."""
    analysis = _import_analysis()
    contract = analysis.load_and_validate_contract(CONTRACT_PATH, RECOVERY_PLAN_SHA256)
    matching = _authority_snapshot(contract)
    matching.update(
        {
            "approved_plan_sha256": RECOVERY_PLAN_SHA256,
            "original_plan_sha256": ORIGINAL_PLAN_SHA256,
            "reporting_correction_sha256": REPORTING_CORRECTION_SHA256,
            "candidate_one_output_absent": True,
            "candidate_two_output_absent": True,
            "worktree_entries": [*matching["worktree_entries"], unexpected_path],
        }
    )

    with pytest.raises(ValueError, match="Worktree"):
        analysis.validate_authority_snapshot(contract, matching)


@pytest.mark.parametrize(
    ("field", "altered"),
    [
        ("branch", "unapproved-branch"),
        ("head", "0" * 40),
        ("sole_parent", "0" * 40),
        ("preprocessing_frozen_is_ancestor", False),
        ("frozen_diff_is_empty", False),
        ("approved_plan_sha256", "0" * 64),
        ("original_plan_sha256", "0" * 64),
        ("recovery_v2_plan_sha256", "0" * 64),
        ("recovery_v3_plan_sha256", "0" * 64),
        ("reporting_correction_sha256", "0" * 64),
        ("numerical_environment_correction_sha256", "0" * 64),
        ("channel_metadata_correction_sha256", "0" * 64),
        ("environment_lock_sha256", "0" * 64),
        ("final_output_absent", False),
        ("candidate_one_output_absent", False),
        ("candidate_two_output_absent", False),
    ],
)
def test_recovery_authority_remains_fail_closed_outside_concurrent_exception(field: str, altered: Any) -> None:
    """Plans, Git authority, and every eFigure 4 output namespace remain immutable gates."""
    analysis = _import_analysis()
    contract = analysis.load_and_validate_contract(CONTRACT_PATH, RECOVERY_PLAN_SHA256)
    matching = _authority_snapshot(contract)
    matching.update(
        {
            "approved_plan_sha256": RECOVERY_PLAN_SHA256,
            "original_plan_sha256": ORIGINAL_PLAN_SHA256,
            "reporting_correction_sha256": REPORTING_CORRECTION_SHA256,
            "candidate_one_output_absent": True,
            "candidate_two_output_absent": True,
        }
    )
    matching[field] = altered

    with pytest.raises(ValueError):
        analysis.validate_authority_snapshot(contract, matching)


SAMPLE_INDEX_USECOLS = [
    "sample_id",
    "filename",
    "n_rows",
    "fallback_used",
    "padded",
    "in_final_output",
]
QC_DIAGNOSTICS_USECOLS = [
    "sample_id",
    "n_rows",
    "t_baseline_end",
    "fallback_used",
    "env_temperature_sample",
    "env_humidity_sample",
    "n_available",
    "clip_length_L",
    "relative_deficit",
    "padded",
]


def _write_minimal_synthetic_audits(tmp_path: Path) -> tuple[Path, Path]:
    """Create synthetic audit CSVs with prohibited distractor columns."""
    sample_index = pd.DataFrame(
        {
            "sample_id": [901, 902, 903],
            "filename": ["synthetic_a.csv", "synthetic_b.csv", "synthetic_c.csv"],
            "n_rows": [12, 12, 12],
            "fallback_used": [False, True, False],
            "padded": [False, False, False],
            "in_final_output": [True, True, False],
            "label": [0, 1, 0],
            "disease_category": [0, 3, 1],
            "collection_time": ["PRIVATE_DATE_A", "PRIVATE_DATE_B", "PRIVATE_DATE_C"],
            "collection_location": [0, 1, 0],
            "age": [40, 50, 60],
            "split": ["train", "test", "train"],
        }
    )
    qc_diagnostics = pd.DataFrame(
        {
            "sample_id": [901, 902, 903],
            "n_rows": [12, 12, 12],
            "t_baseline_end": [3, 3, 3],
            "fallback_used": [False, True, False],
            "env_temperature_sample": [10.0, 11.0, 12.0],
            "env_humidity_sample": [20.0, 21.0, 22.0],
            "n_available": [9, 9, 9],
            "clip_length_L": [8, 8, 8],
            "relative_deficit": [np.nan, np.nan, np.nan],
            "padded": [False, False, np.nan],
            "label": [0, 1, 0],
            "clinical_variable": [1.1, 2.2, 3.3],
        }
    )
    index_path = tmp_path / "sample_index.csv"
    audit_path = tmp_path / "qc_diagnostics.csv"
    sample_index.to_csv(index_path, index=False)
    qc_diagnostics.to_csv(audit_path, index=False)
    return index_path, audit_path


def test_minimal_csv_loading_uses_explicit_usecols_and_aligns_one_to_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only participant-capable fields explicitly approved by the plan may load."""
    analysis = _import_analysis()
    index_path, audit_path = _write_minimal_synthetic_audits(tmp_path)
    calls: list[list[str]] = []
    original_read_csv = analysis.pd.read_csv

    def recording_read_csv(*args: Any, **kwargs: Any) -> pd.DataFrame:
        calls.append(list(kwargs["usecols"]))
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(analysis.pd, "read_csv", recording_read_csv)
    cohort = analysis.load_minimal_cohort_tables(
        index_path,
        audit_path,
        expected_source_count=3,
        expected_final_count=2,
        expected_excluded_count=1,
        expected_source_fallback_count=1,
        expected_final_fallback_count=1,
        expected_excluded_fallback_count=0,
        expected_padded_count=0,
    )

    assert calls == [SAMPLE_INDEX_USECOLS, QC_DIAGNOSTICS_USECOLS]
    assert cohort.source_count == 3
    assert cohort.final_count == 2
    assert cohort.final_rows["sample_id"].tolist() == [901, 902]
    loaded_columns = set(cohort.final_rows.columns)
    assert loaded_columns <= set(SAMPLE_INDEX_USECOLS + QC_DIAGNOSTICS_USECOLS)
    assert not loaded_columns.intersection(
        {"label", "disease_category", "collection_time", "collection_location", "age", "split", "clinical_variable"}
    )


def test_minimal_csv_accepts_nullable_diagnostics_padding_only_for_excluded_rows(tmp_path: Path) -> None:
    """Persisted diagnostics padding is Boolean for final rows and missing for excluded rows."""
    analysis = _import_analysis()
    index_path, audit_path = _write_minimal_synthetic_audits(tmp_path)
    diagnostics = pd.read_csv(audit_path)

    assert diagnostics["padded"].dtype == object
    assert all(isinstance(value, (bool, np.bool_)) for value in diagnostics.loc[:1, "padded"])
    assert pd.isna(diagnostics.loc[2, "padded"])

    cohort = analysis.load_minimal_cohort_tables(
        index_path,
        audit_path,
        expected_source_count=3,
        expected_final_count=2,
        expected_excluded_count=1,
        expected_source_fallback_count=1,
        expected_final_fallback_count=1,
        expected_excluded_fallback_count=0,
        expected_padded_count=0,
    )

    assert cohort.final_rows["padded"].tolist() == [False, False]


def test_minimal_csv_validates_source_final_and_excluded_fallback_counts_separately(tmp_path: Path) -> None:
    """The synthetic source fallback count must reconcile exactly as final plus excluded."""
    analysis = _import_analysis()
    index_path, audit_path = _write_minimal_synthetic_audits(tmp_path)

    cohort = analysis.load_minimal_cohort_tables(
        index_path,
        audit_path,
        expected_source_count=3,
        expected_final_count=2,
        expected_excluded_count=1,
        expected_source_fallback_count=1,
        expected_final_fallback_count=1,
        expected_excluded_fallback_count=0,
        expected_padded_count=0,
    )

    assert cohort.source_count == 3
    assert cohort.final_count == 2


@pytest.mark.parametrize(
    ("source_fallback", "final_fallback", "excluded_fallback"),
    [
        (2, 1, 1),
        (1, 0, 1),
        (1, 1, 1),
        (1, 0, 0),
    ],
)
def test_minimal_csv_rejects_each_fallback_denominator_or_reconciliation_failure(
    tmp_path: Path,
    source_fallback: int,
    final_fallback: int,
    excluded_fallback: int,
) -> None:
    """Every denominator and the source-equals-final-plus-excluded identity must fail closed."""
    analysis = _import_analysis()
    index_path, audit_path = _write_minimal_synthetic_audits(tmp_path)

    with pytest.raises(ValueError, match="fallback|reconciliation"):
        analysis.load_minimal_cohort_tables(
            index_path,
            audit_path,
            expected_source_count=3,
            expected_final_count=2,
            expected_excluded_count=1,
            expected_source_fallback_count=source_fallback,
            expected_final_fallback_count=final_fallback,
            expected_excluded_fallback_count=excluded_fallback,
            expected_padded_count=0,
        )


@pytest.mark.parametrize(
    "failure_mode",
    [
        "included_diagnostics_missing",
        "excluded_diagnostics_nonmissing",
        "excluded_index_true",
        "diagnostics_non_boolean",
        "included_padding_mismatch",
    ],
)
def test_minimal_csv_rejects_invalid_nullable_diagnostics_padding(tmp_path: Path, failure_mode: str) -> None:
    """Nullable diagnostics padding must follow exact post-QC transformation topology."""
    analysis = _import_analysis()
    index_path, audit_path = _write_minimal_synthetic_audits(tmp_path)
    sample_index = pd.read_csv(index_path)
    diagnostics = pd.read_csv(audit_path)
    if failure_mode == "included_diagnostics_missing":
        diagnostics.loc[0, "padded"] = np.nan
    elif failure_mode == "excluded_diagnostics_nonmissing":
        diagnostics.loc[2, "padded"] = False
    elif failure_mode == "excluded_index_true":
        sample_index.loc[2, "padded"] = True
    elif failure_mode == "diagnostics_non_boolean":
        diagnostics["padded"] = diagnostics["padded"].astype(object)
        diagnostics.loc[0, "padded"] = "not-a-boolean"
    else:
        diagnostics.loc[0, "padded"] = True
    sample_index.to_csv(index_path, index=False)
    diagnostics.to_csv(audit_path, index=False)

    with pytest.raises(ValueError, match="padding|padded|Boolean"):
        analysis.load_minimal_cohort_tables(
            index_path,
            audit_path,
            expected_source_count=3,
            expected_final_count=2,
            expected_excluded_count=1,
            expected_source_fallback_count=1,
            expected_final_fallback_count=1,
            expected_excluded_fallback_count=0,
            expected_padded_count=0,
        )


@pytest.mark.parametrize("failure_mode", ["duplicate", "missing", "audit_mismatch"])
def test_minimal_csv_alignment_fails_closed(tmp_path: Path, failure_mode: str) -> None:
    """Duplicate, incomplete, or inconsistent audit alignment must halt."""
    analysis = _import_analysis()
    index_path, audit_path = _write_minimal_synthetic_audits(tmp_path)
    audit = pd.read_csv(audit_path)
    if failure_mode == "duplicate":
        audit.loc[2, "sample_id"] = audit.loc[1, "sample_id"]
    elif failure_mode == "missing":
        audit = audit.iloc[:2, :]
    else:
        audit.loc[0, "n_rows"] = 13
    audit.to_csv(audit_path, index=False)

    with pytest.raises(ValueError, match="alignment|unique|audit"):
        analysis.load_minimal_cohort_tables(
            index_path,
            audit_path,
            expected_source_count=3,
            expected_final_count=2,
            expected_excluded_count=1,
            expected_source_fallback_count=1,
            expected_final_fallback_count=1,
            expected_excluded_fallback_count=0,
            expected_padded_count=0,
        )


@pytest.mark.parametrize("failure_mode", ["missing_filename", "available_count", "relative_deficit"])
def test_minimal_csv_rejects_invalid_raw_resolution_or_trim_audit(tmp_path: Path, failure_mode: str) -> None:
    """Raw resolution and persisted trim arithmetic must be complete and internally exact."""
    analysis = _import_analysis()
    index_path, audit_path = _write_minimal_synthetic_audits(tmp_path)
    sample_index = pd.read_csv(index_path)
    audit = pd.read_csv(audit_path)
    if failure_mode == "missing_filename":
        sample_index.loc[0, "filename"] = np.nan
    elif failure_mode == "available_count":
        audit.loc[0, "n_available"] = 8
    else:
        sample_index.loc[0, "padded"] = True
        audit.loc[0, "padded"] = True
        audit.loc[0, "n_available"] = 7
        audit.loc[0, "relative_deficit"] = 0.5
    sample_index.to_csv(index_path, index=False)
    audit.to_csv(audit_path, index=False)

    with pytest.raises(ValueError, match="filename|available|deficit|padding"):
        analysis.load_minimal_cohort_tables(
            index_path,
            audit_path,
            expected_source_count=3,
            expected_final_count=2,
            expected_excluded_count=1,
            expected_source_fallback_count=1,
            expected_final_fallback_count=1,
            expected_excluded_fallback_count=0,
            expected_padded_count=1 if failure_mode == "relative_deficit" else 0,
        )


def _write_synthetic_raw_signal(path: Path, *, n_rows: int = 16, alter_header: bool = False) -> np.ndarray:
    """Write one synthetic S0-S13 trace and return its S1-S13 values."""
    raw = np.empty((n_rows, 14), dtype=np.float64)
    row_axis = np.arange(n_rows, dtype=np.float64).reshape(-1, 1)
    raw[:, :] = 100.0 + row_axis + np.arange(14, dtype=np.float64).reshape(1, -1)
    raw[:, 1] -= 150.0
    columns = [f"S{index}" for index in range(14)]
    if alter_header:
        columns[-1] = "unexpected"
    pd.DataFrame(raw, columns=columns).to_csv(path, index=False)
    return raw[:, 1:]


def test_raw_loader_validates_header_containment_and_loads_only_s1_through_s13(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw loading should be contained, header-exact, minimal, finite, and unlinkable."""
    analysis = _import_analysis()
    raw_root = tmp_path / "breath_data"
    raw_root.mkdir()
    expected = _write_synthetic_raw_signal(raw_root / "synthetic.csv")
    calls: list[dict[str, Any]] = []
    original_read_csv = analysis.pd.read_csv

    def recording_read_csv(*args: Any, **kwargs: Any) -> pd.DataFrame:
        calls.append(dict(kwargs))
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(analysis.pd, "read_csv", recording_read_csv)
    loaded = analysis.load_raw_signal("synthetic.csv", raw_root, expected_n_rows=16)

    assert np.array_equal(loaded.gas_signals, expected[:, :11])
    assert np.array_equal(loaded.temperature, expected[:, 11])
    assert np.array_equal(loaded.humidity, expected[:, 12])
    assert loaded.gas_signals.dtype == np.float64
    assert calls[-1]["usecols"] == [f"S{index}" for index in range(1, 14)]
    assert "S0" not in calls[-1]["usecols"]


def test_raw_loader_sanitizes_errors_and_preserves_internal_cause(tmp_path: Path) -> None:
    """Raw failures should retain a cause internally but expose no private context."""
    analysis = _import_analysis()
    raw_root = tmp_path / "breath_data"
    raw_root.mkdir()
    private_name = "private_trace_name.csv"
    _write_synthetic_raw_signal(raw_root / private_name, alter_header=True)

    with pytest.raises(analysis.RawSignalLoadError) as captured:
        analysis.load_raw_signal(private_name, raw_root, expected_n_rows=16)

    assert str(captured.value) == "Raw signal loading failed."
    assert private_name not in str(captured.value)
    assert str(raw_root) not in str(captured.value)
    assert captured.value.__cause__ is not None

    with pytest.raises(analysis.RawSignalLoadError, match="Raw signal loading failed"):
        analysis.load_raw_signal("../escape.csv", raw_root, expected_n_rows=16)


def test_raw_loader_rejects_parent_component_even_when_normalized_target_stays_inside_root(tmp_path: Path) -> None:
    """Every parent component is forbidden even when resolution lands inside the raw root."""
    analysis = _import_analysis()
    raw_root = tmp_path / "breath_data"
    nested = raw_root / "nested"
    nested.mkdir(parents=True)
    _write_synthetic_raw_signal(raw_root / "synthetic.csv")

    with pytest.raises(analysis.RawSignalLoadError) as captured:
        analysis.load_raw_signal("nested/../synthetic.csv", raw_root, expected_n_rows=16)

    assert str(captured.value) == "Raw signal loading failed."
    assert "nested" not in str(captured.value)
    assert "synthetic.csv" not in str(captured.value)
    assert str(raw_root) not in str(captured.value)
    assert captured.value.__cause__ is not None


def _synthetic_env_model() -> Any:
    """Build a deserialized zero-shift environmental model without fitting."""
    from src.data.env_compensation import EnvCompensationModel

    return EnvCompensationModel.from_yaml_dict(
        {
            "temperature_humidity_pearson_r": 0.0,
            "temperature_humidity_abs_pearson_r_threshold": 0.8,
            "multicollinearity_fallback_used": False,
            "r2_threshold": 0.0,
            "t_ref": 0.0,
            "h_ref": 0.0,
            "per_channel": {
                f"S{index}": {
                    "alpha": 0.0,
                    "beta_T": 0.0,
                    "beta_H": 0.0,
                    "adj_r2": 0.0,
                    "applies_compensation": False,
                }
                for index in range(1, 12)
            },
        }
    )


def _synthetic_loaded_signal(n_rows: int = 16) -> Any:
    """Create one in-memory synthetic signal with signed gas values."""
    analysis = _import_analysis()
    row_axis = np.arange(n_rows, dtype=np.float64).reshape(-1, 1)
    gas = 50.0 + row_axis + np.arange(11, dtype=np.float64).reshape(1, -1)
    gas[:, 0] -= 80.0
    temperature = 10.0 + row_axis[:, 0]
    humidity = 20.0 + row_axis[:, 0]
    return analysis.RawSignal(gas_signals=gas, temperature=temperature, humidity=humidity, n_rows=n_rows)


def test_replay_uses_persisted_onset_and_frozen_helpers_for_all_four_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The replay should call each frozen transform and trim all stages identically."""
    analysis = _import_analysis()
    raw = _synthetic_loaded_signal()
    model = _synthetic_env_model()
    calls = {"smooth": 0, "correct": 0, "trim": []}
    original_smoothing = analysis.apply_sg_smoothing
    original_correction = analysis.apply_fractional_correction
    original_trim = analysis.trim_response_segment

    def recording_smoothing(values: np.ndarray) -> np.ndarray:
        calls["smooth"] += 1
        return original_smoothing(values)

    def recording_correction(values: np.ndarray, t_baseline_end: int) -> np.ndarray:
        calls["correct"] += 1
        assert t_baseline_end == 3
        return original_correction(values, t_baseline_end)

    def recording_trim(
        corrected_gas: np.ndarray,
        t_baseline_end: int,
        clip_length_L: int,
        padding_relative_deficit_threshold: float,
    ) -> Any:
        calls["trim"].append((t_baseline_end, clip_length_L, padding_relative_deficit_threshold))
        return original_trim(corrected_gas, t_baseline_end, clip_length_L, padding_relative_deficit_threshold)

    monkeypatch.setattr(analysis, "apply_sg_smoothing", recording_smoothing)
    monkeypatch.setattr(analysis, "apply_fractional_correction", recording_correction)
    monkeypatch.setattr(analysis, "trim_response_segment", recording_trim)
    replay = analysis.replay_participant(
        raw,
        t_baseline_end=3,
        persisted_temperature_sample=float(np.mean(raw.temperature[:3])),
        persisted_humidity_sample=float(np.mean(raw.humidity[:3])),
        expected_n_available=13,
        expected_padded=False,
        env_model=model,
        clip_length=8,
        padding_threshold=0.25,
        environmental_rtol=0.0,
        environmental_atol=1.0e-12,
    )

    assert calls == {"smooth": 1, "correct": 1, "trim": [(3, 8, 0.25)] * 4}
    assert tuple(replay.stages) == analysis.EXPECTED_STAGE_IDS
    assert all(stage.shape == (8, 11) for stage in replay.stages.values())
    assert np.array_equal(replay.stages["raw_response_window"], raw.gas_signals[3:11, :])
    assert np.min(replay.stages["raw_response_window"][:, 0]) < 0.0
    assert replay.n_available == 13
    assert replay.padded is False


def test_replay_never_calls_onset_qc_pipeline_or_environment_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Persisted replay must never rediscover onset/QC or fit any model."""
    analysis = _import_analysis()
    from src.data import baseline, preprocessing
    from src.data.env_compensation import EnvCompensationModel

    def prohibited(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("prohibited routine called")

    monkeypatch.setattr(baseline, "detect_baseline_end", prohibited)
    monkeypatch.setattr(preprocessing, "run_qc", prohibited)
    monkeypatch.setattr(preprocessing.PreprocessingPipeline, "run", prohibited)
    monkeypatch.setattr(EnvCompensationModel, "fit", prohibited)
    raw = _synthetic_loaded_signal()

    replay = analysis.replay_participant(
        raw,
        t_baseline_end=3,
        persisted_temperature_sample=float(np.mean(raw.temperature[:3])),
        persisted_humidity_sample=float(np.mean(raw.humidity[:3])),
        expected_n_available=13,
        expected_padded=False,
        env_model=_synthetic_env_model(),
        clip_length=8,
        padding_threshold=0.25,
        environmental_rtol=0.0,
        environmental_atol=1.0e-12,
    )

    assert replay.stages["fractional_baseline_corrected_response_window"].shape == (8, 11)


def test_replay_rejects_environmental_summary_or_padding_mismatch() -> None:
    """Persisted environmental and trim audit facts are mandatory gates."""
    analysis = _import_analysis()
    raw = _synthetic_loaded_signal()
    common = {
        "raw": raw,
        "t_baseline_end": 3,
        "persisted_temperature_sample": float(np.mean(raw.temperature[:3])),
        "persisted_humidity_sample": float(np.mean(raw.humidity[:3])),
        "expected_n_available": 13,
        "expected_padded": False,
        "env_model": _synthetic_env_model(),
        "clip_length": 8,
        "padding_threshold": 0.25,
        "environmental_rtol": 0.0,
        "environmental_atol": 1.0e-12,
    }
    with pytest.raises(ValueError, match="Environmental summary"):
        analysis.replay_participant(**{**common, "persisted_temperature_sample": 999.0})
    with pytest.raises(ValueError, match="padding"):
        analysis.replay_participant(**{**common, "expected_padded": True})


def _small_stage_tensors() -> dict[str, np.ndarray]:
    """Build four deterministic `(4 participants, 3 frames, 11 channels)` tensors."""
    base = np.arange(4 * 3 * 11, dtype=np.float64).reshape(4, 3, 11) - 100.0
    return {
        stage_id: base + 1000.0 * stage_index
        for stage_index, stage_id in enumerate(
            [
                "raw_response_window",
                "savitzky_golay_smoothed_response_window",
                "environment_compensated_response_window",
                "fractional_baseline_corrected_response_window",
            ]
        )
    }


def test_replay_stage_values_are_only_frozen_transform_then_common_trim() -> None:
    """Stages must retain frozen scale/sign with no normalization or extra operation."""
    analysis = _import_analysis()
    raw = _synthetic_loaded_signal()
    model = _synthetic_env_model()
    replay = analysis.replay_participant(
        raw,
        t_baseline_end=3,
        persisted_temperature_sample=float(np.mean(raw.temperature[:3])),
        persisted_humidity_sample=float(np.mean(raw.humidity[:3])),
        expected_n_available=13,
        expected_padded=False,
        env_model=model,
        clip_length=8,
        padding_threshold=0.25,
        environmental_rtol=0.0,
        environmental_atol=1.0e-12,
    )

    expected_x1 = analysis.apply_sg_smoothing(raw.gas_signals)
    expected_x2 = model.apply(expected_x1, float(np.mean(raw.temperature[:3])), float(np.mean(raw.humidity[:3])))
    expected_x3 = analysis.apply_fractional_correction(expected_x2, 3)
    assert np.array_equal(replay.stages["raw_response_window"], raw.gas_signals[3:11])
    assert np.array_equal(replay.stages["savitzky_golay_smoothed_response_window"], expected_x1[3:11])
    assert np.array_equal(replay.stages["environment_compensated_response_window"], expected_x2[3:11])
    assert np.array_equal(replay.stages["fractional_baseline_corrected_response_window"], expected_x3[3:11])
    assert np.min(replay.stages["raw_response_window"]) < 0.0
    assert np.min(replay.stages["savitzky_golay_smoothed_response_window"]) < 0.0
    assert np.min(replay.stages["environment_compensated_response_window"]) < 0.0


def test_stage_tensor_validation_requires_identical_exact_shapes_and_finite_values() -> None:
    """Every stage must have the same exact participant/frame/channel tensor shape."""
    analysis = _import_analysis()
    tensors = _small_stage_tensors()
    analysis.validate_stage_tensors(tensors, expected_participants=4, expected_frames=3)

    altered_shape = dict(tensors)
    altered_shape[analysis.EXPECTED_STAGE_IDS[2]] = altered_shape[analysis.EXPECTED_STAGE_IDS[2]][:, :2, :]
    with pytest.raises(ValueError, match="shape"):
        analysis.validate_stage_tensors(altered_shape, expected_participants=4, expected_frames=3)

    nonfinite = {name: values.copy() for name, values in tensors.items()}
    nonfinite[analysis.EXPECTED_STAGE_IDS[0]][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        analysis.validate_stage_tensors(nonfinite, expected_participants=4, expected_frames=3)


def test_aggregate_uses_linear_quantiles_and_exact_schema_order_dtypes() -> None:
    """Aggregation should emit the exact stage/channel/frame-major CSV contract."""
    analysis = _import_analysis()
    tensors = _small_stage_tensors()
    stages = [
        {"order": index + 1, "id": stage_id, "unit": "unit_a" if index < 3 else "unit_b"}
        for index, stage_id in enumerate(analysis.EXPECTED_STAGE_IDS)
    ]

    summary = analysis.aggregate_stage_tensors(
        tensors,
        stages=stages,
        channels=[f"S{index}" for index in range(1, 12)],
        expected_participants=4,
        expected_frames=3,
        quantile_method="linear",
    )

    assert summary.columns.tolist() == [
        "stage_order",
        "stage_id",
        "channel",
        "response_frame",
        "unit",
        "n_participants",
        "q25",
        "median",
        "q75",
    ]
    assert len(summary) == 4 * 11 * 3
    assert summary.iloc[:3]["response_frame"].tolist() == [0, 1, 2]
    assert summary.iloc[:3]["channel"].tolist() == ["S1", "S1", "S1"]
    assert summary.iloc[3]["channel"] == "S2"
    assert summary.iloc[33]["stage_id"] == analysis.EXPECTED_STAGE_IDS[1]
    expected = np.quantile(tensors[analysis.EXPECTED_STAGE_IDS[0]][:, 0, 0], [0.25, 0.5, 0.75], method="linear")
    assert np.array_equal(summary.loc[0, ["q25", "median", "q75"]].to_numpy(dtype=np.float64), expected)
    assert all(summary[column].dtype == np.int64 for column in ["stage_order", "response_frame", "n_participants"])
    assert all(summary[column].dtype == np.float64 for column in ["q25", "median", "q75"])
    assert summary["n_participants"].eq(4).all()
    assert (summary["q25"] <= summary["median"]).all()
    assert (summary["median"] <= summary["q75"]).all()
    assert not summary.isna().any().any()

    payload = analysis.serialize_aggregate_csv(summary, float_format="%.17g")
    assert payload.startswith(b"stage_order,stage_id,channel,response_frame,unit,n_participants,q25,median,q75\n")
    assert b"\r\n" not in payload


def test_real_output_row_count_formula_and_contract_are_exact() -> None:
    """The fixed real aggregate contract is four stages by 11 channels by 960 frames."""
    contract = _load_contract()
    assert len(contract["stages"]) * len(contract["channels"]) * contract["response_axis"]["clip_length"] == 42240
    assert contract["outputs"]["expected_aggregate_rows"] == 42240


def test_stage4_reconciliation_enforces_order_and_frozen_tolerance() -> None:
    """Stage 4 accepts only exact alignment and maximum difference at most 1e-12."""
    analysis = _import_analysis()
    frozen = np.arange(2 * 3 * 11, dtype=np.float64).reshape(2, 3, 11) / 10.0
    sample_ids = np.array([801, 802], dtype=np.int64)
    channels = np.array([f"S{index}" for index in range(1, 12)])

    exact = analysis.reconcile_stage4(
        frozen.copy(),
        sample_ids,
        frozen,
        sample_ids.copy(),
        channels,
        frozen_clip_length=3,
        rtol=0.0,
        atol=1.0e-12,
    )
    assert exact["bitwise_equal"] is True
    assert exact["maximum_absolute_difference"] == 0.0
    assert exact["passed"] is True

    within = frozen.copy()
    within[0, 0, 0] += 5.0e-13
    accepted = analysis.reconcile_stage4(
        within,
        sample_ids,
        frozen,
        sample_ids,
        channels,
        frozen_clip_length=3,
        rtol=0.0,
        atol=1.0e-12,
    )
    assert accepted["bitwise_equal"] is False
    assert accepted["maximum_absolute_difference"] <= 1.0e-12
    assert accepted["passed"] is True

    outside = frozen.copy()
    outside[0, 0, 0] += 2.0e-12
    with pytest.raises(ValueError, match="Stage 4 reconciliation"):
        analysis.reconcile_stage4(
            outside,
            sample_ids,
            frozen,
            sample_ids,
            channels,
            frozen_clip_length=3,
            rtol=0.0,
            atol=1.0e-12,
        )
    with pytest.raises(ValueError, match="order"):
        analysis.reconcile_stage4(
            frozen,
            sample_ids[::-1],
            frozen,
            sample_ids,
            channels,
            frozen_clip_length=3,
            rtol=0.0,
            atol=1.0e-12,
        )


@pytest.mark.parametrize(
    "forbidden_payload",
    [
        b'{"sample_id": 901}',
        b'{"participant_key": "private"}',
        b'{"raw_filename": "private.csv"}',
        b'{"per_sample_hash": "abc"}',
        b'{"raw_path": "[PRIVATE_PATH_REDACTED]"}',
        b'{"collection_date": "2026-01-02"}',
        b'{"label": 1}',
        b'{"outcome": 0}',
        b'{"disease_category": 3}',
        b'{"collection_location": 1}',
        b'{"age": 50}',
        b'{"participant_level_array": [1.0, 2.0]}',
    ],
)
def test_output_privacy_rejects_forbidden_participant_material(forbidden_payload: bytes) -> None:
    """Serialized outputs must reject every prohibited participant-capable category."""
    analysis = _import_analysis()
    with pytest.raises(ValueError, match="Privacy validation failed"):
        analysis.validate_serialized_privacy(
            "aggregate_audit.json",
            forbidden_payload,
            allowed_filenames=_load_contract()["outputs"]["allowed_filenames"],
        )


def test_output_privacy_allows_aggregate_provenance_and_rejects_private_values_or_filename() -> None:
    """Aggregate-safe IDs, counts, hashes, versions, and project-relative provenance remain valid."""
    analysis = _import_analysis()
    allowed_filenames = _load_contract()["outputs"]["allowed_filenames"]
    safe_payload = (
        b'{"stage_id":"raw_response_window","channel":"S11","response_frame":959,'
        b'"n_participants":972,"unit":"recorded_signal_unit","sha256":"' + b"a" * 64
        + b'","numpy":"2.3.5","source_path":"data/processed/sample_index.csv",'
        b'"provenance_timestamp":"2026-08-12T00:00:00Z",'
        b'"participant_level_persistence":false,"prohibited_field_access":false}'
    )
    analysis.validate_serialized_privacy(
        "aggregate_audit.json",
        safe_payload,
        allowed_filenames=allowed_filenames,
        private_values=("PRIVATE_RAW_NAME.csv",),
    )

    with pytest.raises(ValueError, match="Privacy validation failed"):
        analysis.validate_serialized_privacy(
            "aggregate_audit.json",
            b'{"note":"PRIVATE_RAW_NAME.csv"}',
            allowed_filenames=allowed_filenames,
            private_values=("PRIVATE_RAW_NAME.csv",),
        )
    with pytest.raises(ValueError, match="Unexpected candidate filename"):
        analysis.validate_serialized_privacy(
            "unexpected.json",
            safe_payload,
            allowed_filenames=allowed_filenames,
        )


@pytest.mark.parametrize("absolute_path", ["/tmp/private.csv", "/var/data/file.csv", "/etc/private"])
def test_output_privacy_rejects_every_absolute_posix_path(absolute_path: str) -> None:
    """Absolute POSIX paths are forbidden regardless of their top-level directory."""
    analysis = _import_analysis()
    payload = analysis.serialize_json({"note": absolute_path})

    with pytest.raises(ValueError, match="Privacy validation failed"):
        analysis.validate_serialized_privacy(
            "aggregate_audit.json",
            payload,
            allowed_filenames=_load_contract()["outputs"]["allowed_filenames"],
        )


def test_output_privacy_keeps_approved_project_relative_paths_allowed() -> None:
    """Approved project-relative provenance paths remain valid."""
    analysis = _import_analysis()
    payload = analysis.serialize_json(
        {
            "approved_plan_path": "docs/plans/efigure4_preprocessing_stage_aggregate_replay_results_free_plan.md",
            "source_path": "data/processed/qc_reports/qc_diagnostics.csv",
        }
    )

    analysis.validate_serialized_privacy(
        "aggregate_audit.json",
        payload,
        allowed_filenames=_load_contract()["outputs"]["allowed_filenames"],
    )


def _small_contract() -> dict[str, Any]:
    """Return a contract copy narrowed only for synthetic aggregate fixtures."""
    contract = copy.deepcopy(_load_contract())
    contract["population"].update(
        {
            "expected_source_count": 5,
            "expected_final_count": 4,
            "expected_excluded_count": 1,
            "expected_source_fallback_count": 1,
            "expected_final_fallback_count": 1,
            "expected_excluded_fallback_count": 0,
            "expected_padded_count": 0,
        }
    )
    contract["response_axis"].update({"last": 2, "clip_length": 3})
    contract["outputs"]["expected_aggregate_rows"] = 132
    return contract


def _small_summary(analysis: Any, contract: dict[str, Any]) -> pd.DataFrame:
    """Build one valid synthetic aggregate table using the locked stage metadata."""
    return analysis.aggregate_stage_tensors(
        _small_stage_tensors(),
        stages=contract["stages"],
        channels=contract["channels"],
        expected_participants=4,
        expected_frames=3,
        quantile_method="linear",
    )


def _small_reconciliation(analysis: Any) -> dict[str, Any]:
    """Build an aggregate-safe exact synthetic Stage 4 reconciliation result."""
    frozen = _small_stage_tensors()[analysis.EXPECTED_STAGE_IDS[-1]]
    ids = np.arange(4, dtype=np.int64)
    channels = np.array([f"S{index}" for index in range(1, 12)])
    return analysis.reconcile_stage4(
        frozen,
        ids,
        frozen.copy(),
        ids.copy(),
        channels,
        frozen_clip_length=3,
        rtol=0.0,
        atol=1.0e-12,
    )


def _synthetic_environment_provenance() -> dict[str, Any]:
    """Return the exact aggregate-safe environment provenance fixture."""
    lock = json.loads(ENVIRONMENT_LOCK_PATH.read_text(encoding="utf-8"))
    return {
        "environment_lock_sha256": ENVIRONMENT_LOCK_SHA256,
        "critical_numerical_builds": lock["critical_numerical_builds"],
        "package_ledger_equality_passed": True,
        "archive_hash_validation_passed": True,
    }


def _small_payloads(
    environment_provenance: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any], dict[str, bytes]]:
    """Build a complete deterministic synthetic four-file candidate."""
    analysis = _import_analysis()
    contract = _small_contract()
    authority = _authority_snapshot(contract)
    implementation_hashes = {
        "config_sha256": "1" * 64,
        "module_sha256": "2" * 64,
        "script_sha256": "3" * 64,
    }
    source_identities = {
        name: (
            {
                "path": payload["path"],
                "git_blob": payload["git_blob"],
                "sha256": None,
                "sha256_status": "unresolved local SHA-256 computation/provenance anomaly",
                "byte_size": int(payload["byte_size"]),
                "blake2b": payload["blake2b"],
            }
            if name == "preprocessing_config"
            else {
                "path": payload["path"],
                "git_blob": payload["git_blob"],
                "sha256": "4" * 64,
                "byte_size": 1,
            }
        )
        for name, payload in contract["inputs"].items()
    }
    audit = analysis.build_aggregate_audit(
        contract,
        authority=authority,
        implementation_hashes=implementation_hashes,
        package_versions={
            "python": "3.11.15",
            "numpy": "2.3.5",
            "pandas": "2.3.3",
            "scipy": "1.17.1",
            "pyyaml": "6.0.3",
        },
        environment_provenance=environment_provenance or _synthetic_environment_provenance(),
        source_identities=source_identities,
        environmental_summary_reconciliation_passed=True,
        stage4_reconciliation=_small_reconciliation(analysis),
        raw_cohort_digest="5" * 64,
        source_entry_exit_identity_passed=True,
        deterministic_rerun_passed=True,
        privacy_validation_passed=True,
    )
    payloads = analysis.build_candidate_payloads(
        _small_summary(analysis, contract),
        audit=audit,
        contract=contract,
        source_identities=source_identities,
        implementation_hashes=implementation_hashes,
    )
    return analysis, contract, payloads


def test_candidate_payloads_are_byte_deterministic_and_manifest_has_no_self_hash() -> None:
    """Two isolated serializations must match and the manifest must forbid self-hashing."""
    analysis, contract, first = _small_payloads()
    _analysis, _contract, second = _small_payloads()

    assert list(first) == contract["outputs"]["allowed_filenames"]
    assert first == second
    manifest = analysis.json.loads(first["manifest.json"])
    assert manifest["self_hash"] is None
    assert manifest["self_hash_policy"] == "forbidden"
    assert "manifest.json" not in [entry["path"] for entry in manifest["outputs"]]
    expected_audit = analysis.json.loads(first["aggregate_audit.json"])
    analysis.validate_candidate_payloads(
        first,
        contract=contract,
        expected_participants=4,
        expected_frames=3,
        expected_audit=expected_audit,
    )


def test_run_summary_precisely_limits_the_excluded_cohort_statement_to_post_qc_transformations() -> None:
    """Summary wording must not imply that excluded records skipped onset detection or QC."""
    _analysis, _contract, payloads = _small_payloads()
    summary = payloads["run_summary.md"].decode("utf-8")

    assert "did not undergo the post-QC transformations" in summary
    assert "were not transformed" not in summary


def test_audit_read_back_and_run_summary_use_only_denominator_explicit_fallback_fields() -> None:
    """All aggregate provenance must state source, final, and excluded fallback denominators."""
    analysis, contract, payloads = _small_payloads()
    audit = analysis.json.loads(payloads["aggregate_audit.json"])
    summary = payloads["run_summary.md"].decode("utf-8")

    assert audit["population_counts"] == {
        "excluded_before_post_qc_transformations": 1,
        "fallback_t_baseline_end_source_cohort": 1,
        "fallback_t_baseline_end_final_cohort": 1,
        "fallback_t_baseline_end_excluded_cohort": 0,
        "final_cohort": 4,
        "source_cohort": 5,
        "tail_padded_final_signals": 0,
    }
    assert "fallback_t_baseline_end" not in audit["population_counts"]
    assert "1 fallback resolutions among all 5 source records" in summary
    assert "1 fallback resolutions among the 4 final records" in summary
    assert "0 fallback resolutions among the 1 excluded records" in summary
    assert "1 = 1 + 0" in summary
    expected_audit = analysis.json.loads(payloads["aggregate_audit.json"])
    analysis.validate_candidate_payloads(
        payloads,
        contract=contract,
        expected_participants=4,
        expected_frames=3,
        expected_audit=expected_audit,
    )


def test_audit_and_manifest_contain_the_same_complete_source_identity_ledger() -> None:
    """Both JSON artifacts must retain one exact aggregate-safe source ledger."""
    analysis, _contract, payloads = _small_payloads()
    audit = analysis.json.loads(payloads["aggregate_audit.json"])
    manifest = analysis.json.loads(payloads["manifest.json"])

    assert audit["source_identities"] == manifest["source_identities"]
    assert set(audit["source_identities"]) == set(_load_contract()["inputs"])
    preprocessing_identity = audit["source_identities"]["preprocessing_config"]
    assert preprocessing_identity["sha256"] is None
    assert preprocessing_identity["sha256_status"] == "unresolved local SHA-256 computation/provenance anomaly"


def test_audit_manifest_and_summary_bind_complete_recovery_and_environment_provenance() -> None:
    """Every aggregate provenance ledger must bind the complete immutable recovery chain."""
    analysis, contract, payloads = _small_payloads()
    audit = analysis.json.loads(payloads["aggregate_audit.json"])
    manifest = analysis.json.loads(payloads["manifest.json"])

    assert audit["approved_plan"] == contract["approved_plan"]
    assert audit["original_plan"] == contract["original_plan"]
    assert audit["recovery_v2_plan"] == contract["recovery_v2_plan"]
    assert audit["recovery_v3_plan"] == contract["recovery_v3_plan"]
    assert audit["reporting_correction"] == contract["reporting_correction"]
    assert audit["numerical_environment_correction"] == contract["numerical_environment_correction"]
    assert audit["channel_metadata_correction"] == contract["channel_metadata_correction"]
    assert audit["environment_provenance"] == _synthetic_environment_provenance()
    assert manifest["approved_plan_sha256"] == APPROVED_PLAN_SHA256
    assert manifest["original_plan_sha256"] == ORIGINAL_PLAN_SHA256
    assert manifest["recovery_v2_plan_sha256"] == RECOVERY_V2_PLAN_SHA256
    assert manifest["recovery_v3_plan_sha256"] == RECOVERY_V3_PLAN_SHA256
    assert manifest["reporting_correction_sha256"] == REPORTING_CORRECTION_SHA256
    assert manifest["numerical_environment_correction_sha256"] == NUMERICAL_ENVIRONMENT_CORRECTION_SHA256
    assert manifest["channel_metadata_correction_sha256"] == CHANNEL_METADATA_CORRECTION_SHA256
    assert manifest["environment_provenance"] == _synthetic_environment_provenance()
    summary = payloads["run_summary.md"].decode("utf-8")
    assert ENVIRONMENT_LOCK_SHA256 in summary
    for record in _synthetic_environment_provenance()["critical_numerical_builds"]:
        assert f"{record['name']} {record['version']} {record['build']}" in summary


def test_persisted_environment_provenance_excludes_internal_runtime_metadata() -> None:
    """Internal ledger details and operational values must not enter aggregate outputs."""
    runtime_evidence = {
        **_synthetic_environment_provenance(),
        "environment_name": "synthetic-private-name",
        "platform": "linux",
        "subdirectory": "linux-64",
        "package_count": 324,
        "source_ledger_sha256": SOURCE_LEDGER_SHA256,
        "final_ledger_sha256": FINAL_LEDGER_SHA256,
        "private_operational_path": "/synthetic/private/prefix",
    }

    analysis, _contract, payloads = _small_payloads(runtime_evidence)
    audit = analysis.json.loads(payloads["aggregate_audit.json"])
    manifest = analysis.json.loads(payloads["manifest.json"])

    assert audit["environment_provenance"] == _synthetic_environment_provenance()
    assert manifest["environment_provenance"] == _synthetic_environment_provenance()
    assert b"/synthetic/private/prefix" not in b"".join(payloads.values())


@pytest.mark.parametrize(
    "category",
    [
        "approved_plan",
        "original_plan",
        "recovery_v2_plan",
        "recovery_v3_plan",
        "reporting_correction",
        "numerical_environment_correction",
        "channel_metadata_correction",
        "environment_provenance",
        "environment_lock_hash",
        "environment_build_ledger",
        "environment_ledger_status",
        "environment_archive_status",
        "implementation_hashes",
        "source_identities",
        "package_versions",
        "population_counts",
        "stage_definitions",
        "channel_definitions",
        "frame_definitions",
        "quantile_definitions",
        "stage4_reconciliation",
        "environmental_status",
        "finite_status",
        "privacy_status",
        "determinism_status",
        "source_entry_exit_status",
        "participant_persistence",
        "prohibited_access",
        "output_status",
        "figure_design",
        "release_authorization",
        "manifest_output_hashes",
        "manifest_original_plan",
        "manifest_recovery_v2_plan",
        "manifest_recovery_v3_plan",
        "manifest_reporting_correction",
        "manifest_numerical_environment_correction",
        "manifest_channel_metadata_correction",
        "manifest_environment_provenance",
        "manifest_output_order",
        "manifest_privacy_status",
        "manifest_read_back_status",
        "manifest_source_entry_exit_status",
        "manifest_self_hash_policy",
    ],
)
def test_candidate_read_back_rejects_every_material_audit_or_manifest_tamper(category: str) -> None:
    """Every contract-critical JSON category must be reconciled, not merely parseable."""
    analysis, contract, original = _small_payloads()
    expected_audit = analysis.json.loads(original["aggregate_audit.json"])
    payloads = dict(original)
    audit = analysis.json.loads(payloads["aggregate_audit.json"])
    manifest = analysis.json.loads(payloads["manifest.json"])
    audit_categories = {
        "approved_plan",
        "original_plan",
        "recovery_v2_plan",
        "recovery_v3_plan",
        "reporting_correction",
        "numerical_environment_correction",
        "channel_metadata_correction",
        "environment_provenance",
        "environment_lock_hash",
        "environment_build_ledger",
        "environment_ledger_status",
        "environment_archive_status",
        "implementation_hashes",
        "source_identities",
        "package_versions",
        "population_counts",
        "stage_definitions",
        "channel_definitions",
        "frame_definitions",
        "quantile_definitions",
        "stage4_reconciliation",
        "environmental_status",
        "finite_status",
        "privacy_status",
        "determinism_status",
        "source_entry_exit_status",
        "participant_persistence",
        "prohibited_access",
        "output_status",
        "figure_design",
        "release_authorization",
    }
    if category == "approved_plan":
        audit["approved_plan"].pop("path")
    elif category == "original_plan":
        audit["original_plan"].pop("path")
    elif category == "recovery_v2_plan":
        audit["recovery_v2_plan"].pop("path")
    elif category == "recovery_v3_plan":
        audit["recovery_v3_plan"].pop("path")
    elif category == "reporting_correction":
        audit["reporting_correction"]["byte_size"] += 1
    elif category == "numerical_environment_correction":
        audit["numerical_environment_correction"].pop("path")
    elif category == "channel_metadata_correction":
        audit["channel_metadata_correction"].pop("path")
    elif category == "environment_provenance":
        audit.pop("environment_provenance")
    elif category == "environment_lock_hash":
        audit["environment_provenance"]["environment_lock_sha256"] = "0" * 64
    elif category == "environment_build_ledger":
        audit["environment_provenance"]["critical_numerical_builds"][0]["build"] = "wrong"
    elif category == "environment_ledger_status":
        audit["environment_provenance"]["package_ledger_equality_passed"] = False
    elif category == "environment_archive_status":
        audit["environment_provenance"]["archive_hash_validation_passed"] = False
    elif category == "implementation_hashes":
        audit["implementation_hashes"]["module_sha256"] = "a" * 64
    elif category == "source_identities":
        audit["source_identities"].pop(next(iter(audit["source_identities"])))
    elif category == "package_versions":
        audit["package_versions"].pop("scipy")
    elif category == "population_counts":
        audit["population_counts"]["final_cohort"] += 1
    elif category == "stage_definitions":
        audit["stages"][0].pop("unit")
    elif category == "channel_definitions":
        audit["channels"] = audit["channels"][:-1]
    elif category == "frame_definitions":
        audit["response_frames"]["last"] += 1
    elif category == "quantile_definitions":
        audit["quantiles"]["method"] = "nearest"
    elif category == "stage4_reconciliation":
        audit["stage4_reconciliation"]["passed"] = False
    elif category == "environmental_status":
        audit["environmental_summary_reconciliation"]["passed"] = False
    elif category == "finite_status":
        audit["finite_value_validation"]["passed"] = False
    elif category == "privacy_status":
        audit["privacy_validation"]["passed"] = False
    elif category == "determinism_status":
        audit["deterministic_rerun"]["passed"] = False
    elif category == "source_entry_exit_status":
        audit["source_entry_exit_identity"]["passed"] = False
    elif category == "participant_persistence":
        audit["participant_level_persistence"] = True
    elif category == "prohibited_access":
        audit["prohibited_field_access"] = True
    elif category == "output_status":
        audit["output_state"] = "tampered"
    elif category == "figure_design":
        audit["figure_design_authorization"] = True
    elif category == "release_authorization":
        audit["release_authorization"] = True
    elif category == "manifest_output_hashes":
        manifest["outputs"][0]["sha256"] = "b" * 64
    elif category == "manifest_original_plan":
        manifest["original_plan_sha256"] = "b" * 64
    elif category == "manifest_recovery_v2_plan":
        manifest["recovery_v2_plan_sha256"] = "b" * 64
    elif category == "manifest_recovery_v3_plan":
        manifest["recovery_v3_plan_sha256"] = "b" * 64
    elif category == "manifest_reporting_correction":
        manifest["reporting_correction_sha256"] = "b" * 64
    elif category == "manifest_numerical_environment_correction":
        manifest["numerical_environment_correction_sha256"] = "b" * 64
    elif category == "manifest_channel_metadata_correction":
        manifest["channel_metadata_correction_sha256"] = "b" * 64
    elif category == "manifest_environment_provenance":
        manifest["environment_provenance"]["package_ledger_equality_passed"] = False
    elif category == "manifest_output_order":
        manifest["declared_output_filename_order"] = list(reversed(manifest["declared_output_filename_order"]))
    elif category == "manifest_privacy_status":
        manifest["privacy_validation"]["passed"] = False
    elif category == "manifest_read_back_status":
        manifest["read_back_validation"]["passed"] = False
    elif category == "manifest_source_entry_exit_status":
        manifest["source_entry_exit_validation"]["passed"] = False
    elif category == "manifest_self_hash_policy":
        manifest["self_hash"] = "c" * 64

    if category in audit_categories:
        payloads["aggregate_audit.json"] = analysis.serialize_json(audit)
        for output in manifest["outputs"]:
            if output["path"].endswith("/aggregate_audit.json"):
                output["sha256"] = analysis.sha256_bytes(payloads["aggregate_audit.json"])
    payloads["manifest.json"] = analysis.serialize_json(manifest)

    with pytest.raises(ValueError):
        analysis.validate_candidate_payloads(
            payloads,
            contract=contract,
            expected_participants=4,
            expected_frames=3,
            expected_audit=expected_audit,
        )


@pytest.mark.parametrize("category", ["implementation_hashes", "source_identities", "stage4_values"])
def test_candidate_read_back_rejects_coordinated_tamper_against_exact_execution_audit(category: str) -> None:
    """Audit/manifest agreement cannot replace equality to the execution-time audit."""
    analysis, contract, original = _small_payloads()
    expected_audit = analysis.json.loads(original["aggregate_audit.json"])
    payloads = dict(original)
    audit = copy.deepcopy(expected_audit)
    manifest = analysis.json.loads(payloads["manifest.json"])
    if category == "implementation_hashes":
        audit["implementation_hashes"]["module_sha256"] = "a" * 64
        manifest["implementation_hashes"] = copy.deepcopy(audit["implementation_hashes"])
    elif category == "source_identities":
        ordinary_name = next(name for name in audit["source_identities"] if name != "preprocessing_config")
        audit["source_identities"][ordinary_name]["sha256"] = "b" * 64
        manifest["source_identities"] = copy.deepcopy(audit["source_identities"])
    else:
        audit["stage4_reconciliation"]["bitwise_equal"] = False
        audit["stage4_reconciliation"]["maximum_absolute_difference"] = 5.0e-13
    payloads["aggregate_audit.json"] = analysis.serialize_json(audit)
    for output in manifest["outputs"]:
        if output["path"].endswith("/aggregate_audit.json"):
            output["sha256"] = analysis.sha256_bytes(payloads["aggregate_audit.json"])
    payloads["manifest.json"] = analysis.serialize_json(manifest)

    with pytest.raises(ValueError):
        analysis.validate_candidate_payloads(
            payloads,
            contract=contract,
            expected_participants=4,
            expected_frames=3,
            expected_audit=expected_audit,
        )


def test_existing_or_partial_final_namespace_fails_without_overwrite(tmp_path: Path) -> None:
    """Any stale final path must halt while preserving its existing material."""
    analysis = _import_analysis()
    final_path = tmp_path / "final"
    final_path.mkdir()
    sentinel = final_path / "partial.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        analysis.require_absent_output_namespace(final_path)

    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_candidate_is_written_read_back_validated_and_promoted_atomically(tmp_path: Path) -> None:
    """A complete candidate should survive read-back and one atomic directory rename."""
    analysis, contract, payloads = _small_payloads()
    expected_audit = analysis.json.loads(payloads["aggregate_audit.json"])
    candidate_path = tmp_path / ".candidate-one"
    final_path = tmp_path / "final"

    analysis.write_candidate_directory(
        candidate_path,
        payloads,
        contract=contract,
        expected_participants=4,
        expected_frames=3,
        expected_audit=expected_audit,
    )
    read_back = analysis.read_and_validate_candidate_directory(
        candidate_path,
        contract=contract,
        expected_participants=4,
        expected_frames=3,
        expected_audit=expected_audit,
    )
    assert read_back == payloads

    analysis.promote_candidate_directory(
        candidate_path,
        final_path,
        contract=contract,
        expected_participants=4,
        expected_frames=3,
        expected_audit=expected_audit,
    )
    assert not candidate_path.exists()
    assert final_path.is_dir()
    assert {path.name for path in final_path.iterdir()} == set(contract["outputs"]["allowed_filenames"])


def test_failed_candidate_write_leaves_no_final_or_apparently_complete_candidate(tmp_path: Path) -> None:
    """Invalid output must clean only its new candidate and never create final state."""
    analysis, contract, payloads = _small_payloads()
    expected_audit = analysis.json.loads(payloads["aggregate_audit.json"])
    candidate_path = tmp_path / ".candidate-failed"
    final_path = tmp_path / "final"
    invalid = dict(payloads)
    invalid["unexpected.json"] = invalid.pop("manifest.json")

    with pytest.raises(ValueError, match="filename"):
        analysis.write_candidate_directory(
            candidate_path,
            invalid,
            contract=contract,
            expected_participants=4,
            expected_frames=3,
            expected_audit=expected_audit,
        )

    assert not candidate_path.exists()
    assert not final_path.exists()


def test_unexpected_file_in_complete_candidate_fails_validation(tmp_path: Path) -> None:
    """A complete candidate directory may contain exactly four declared files."""
    analysis, contract, payloads = _small_payloads()
    expected_audit = analysis.json.loads(payloads["aggregate_audit.json"])
    candidate_path = tmp_path / ".candidate-extra"
    analysis.write_candidate_directory(
        candidate_path,
        payloads,
        contract=contract,
        expected_participants=4,
        expected_frames=3,
        expected_audit=expected_audit,
    )
    (candidate_path / "unexpected.txt").write_text("aggregate", encoding="utf-8")

    with pytest.raises(ValueError, match="filename"):
        analysis.read_and_validate_candidate_directory(
            candidate_path,
            contract=contract,
            expected_participants=4,
            expected_frames=3,
            expected_audit=expected_audit,
        )


def test_frozen_source_entry_exit_mismatch_fails_before_promotion() -> None:
    """Any declared source identity change must be rejected exactly."""
    analysis = _import_analysis()
    entry = {"source": {"sha256": "a" * 64, "git_blob": "b" * 40}}
    analysis.require_source_identity_match(entry, copy.deepcopy(entry))
    changed = copy.deepcopy(entry)
    changed["source"]["sha256"] = "c" * 64

    with pytest.raises(ValueError, match="source entry/exit"):
        analysis.require_source_identity_match(entry, changed)


def test_raw_cohort_digest_uses_canonical_bytes_not_ids_filenames_or_paths(tmp_path: Path) -> None:
    """The one digest should depend on ordered file bytes only, with unambiguous lengths."""
    analysis = _import_analysis()
    first_root = tmp_path / "root_a"
    second_root = tmp_path / "root_b"
    first_root.mkdir()
    second_root.mkdir()
    (first_root / "private_a.csv").write_bytes(b"alpha")
    (first_root / "private_b.csv").write_bytes(b"beta-data")
    (second_root / "unrelated_x.csv").write_bytes(b"alpha")
    (second_root / "unrelated_y.csv").write_bytes(b"beta-data")
    first_rows = pd.DataFrame({"sample_id": [901, 902], "filename": ["private_a.csv", "private_b.csv"]})
    second_rows = pd.DataFrame({"sample_id": [111, 222], "filename": ["unrelated_x.csv", "unrelated_y.csv"]})

    first_digest = analysis.compute_raw_cohort_digest(first_rows, first_root)
    second_digest = analysis.compute_raw_cohort_digest(second_rows, second_root)

    assert first_digest == second_digest
    assert len(first_digest) == 64
    reversed_rows = first_rows.iloc[::-1].reset_index(drop=True)
    assert analysis.compute_raw_cohort_digest(reversed_rows, first_root) != first_digest


def test_pipeline_a_loader_accesses_only_permitted_arrays(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Frozen NPZ loading must never access the label member."""
    analysis = _import_analysis()
    accessed: list[str] = []
    values = {
        "x": np.zeros((2, 3, 11), dtype=np.float64),
        "sample_id": np.array([901, 902], dtype=np.int64),
        "channel_names": np.array([f"S{index}" for index in range(1, 12)]),
        "clip_length_L": np.array(3, dtype=np.int64),
    }

    class FakeArchive:
        files = ["x", "sample_id", "label", "channel_names", "clip_length_L"]

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def __getitem__(self, key: str) -> np.ndarray:
            accessed.append(key)
            if key == "label":
                raise AssertionError("label array accessed")
            return values[key]

    monkeypatch.setattr(analysis.np, "load", lambda *args, **kwargs: FakeArchive())
    loaded = analysis.load_frozen_pipeline_a(tmp_path / "synthetic.npz")

    assert accessed == ["x", "sample_id", "channel_names", "clip_length_L"]
    assert loaded.x.shape == (2, 3, 11)
    assert loaded.clip_length == 3


def _write_replay_raw_file(path: Path, *, offset: float) -> tuple[float, float]:
    """Write a deterministic 16-row raw trace and return persisted S12/S13 means."""
    values = np.empty((16, 14), dtype=np.float64)
    row_axis = np.arange(16, dtype=np.float64).reshape(-1, 1)
    values[:] = offset + 100.0 + row_axis + np.arange(14, dtype=np.float64).reshape(1, -1)
    values[:, 1] -= 130.0
    pd.DataFrame(values, columns=[f"S{index}" for index in range(14)]).to_csv(path, index=False)
    return float(np.mean(values[:3, 12])), float(np.mean(values[:3, 13]))


def test_two_isolated_synthetic_full_replays_serialize_identically(tmp_path: Path) -> None:
    """Independent loader/transform/reconciliation/aggregation runs must be byte-identical."""
    analysis = _import_analysis()
    raw_root = tmp_path / "breath_data"
    raw_root.mkdir()
    t_first, h_first = _write_replay_raw_file(raw_root / "synthetic_first.csv", offset=0.0)
    t_second, h_second = _write_replay_raw_file(raw_root / "synthetic_second.csv", offset=10.0)
    rows = pd.DataFrame(
        {
            "sample_id": np.array([901, 902], dtype=np.int64),
            "filename": ["synthetic_first.csv", "synthetic_second.csv"],
            "n_rows": np.array([16, 16], dtype=np.int64),
            "t_baseline_end": np.array([3, 3], dtype=np.int64),
            "fallback_used": [False, False],
            "env_temperature_sample": np.array([t_first, t_second], dtype=np.float64),
            "env_humidity_sample": np.array([h_first, h_second], dtype=np.float64),
            "n_available": np.array([13, 13], dtype=np.int64),
            "clip_length_L": np.array([8, 8], dtype=np.int64),
            "relative_deficit": np.array([np.nan, np.nan]),
            "padded": [False, False],
        }
    )
    cohort = analysis.CohortTables(final_rows=rows, source_count=2, final_count=2)
    contract = _small_contract()
    contract["population"].update(
        {
            "expected_source_count": 2,
            "expected_final_count": 2,
            "expected_excluded_count": 0,
            "expected_source_fallback_count": 0,
            "expected_final_fallback_count": 0,
            "expected_excluded_fallback_count": 0,
            "expected_padded_count": 0,
        }
    )
    contract["response_axis"].update({"last": 7, "clip_length": 8})
    contract["outputs"]["expected_aggregate_rows"] = 352
    model = _synthetic_env_model()

    stage4_rows: list[np.ndarray] = []
    for row in rows.itertuples(index=False):
        raw = analysis.load_raw_signal(row.filename, raw_root, expected_n_rows=int(row.n_rows))
        replay = analysis.replay_participant(
            raw,
            t_baseline_end=int(row.t_baseline_end),
            persisted_temperature_sample=float(row.env_temperature_sample),
            persisted_humidity_sample=float(row.env_humidity_sample),
            expected_n_available=int(row.n_available),
            expected_padded=bool(row.padded),
            env_model=model,
            clip_length=8,
            padding_threshold=0.25,
            environmental_rtol=0.0,
            environmental_atol=1.0e-12,
        )
        stage4_rows.append(replay.stages[analysis.EXPECTED_STAGE_IDS[-1]])
    pipeline_path = tmp_path / "pipeline_a.npz"
    np.savez(
        pipeline_path,
        x=np.stack(stage4_rows),
        sample_id=rows["sample_id"].to_numpy(),
        label=np.array([0, 1], dtype=np.int64),
        channel_names=np.array(contract["channels"]),
        clip_length_L=np.int64(8),
    )

    first = analysis.compute_replay_and_aggregate(
        contract,
        cohort,
        env_model=model,
        raw_root=raw_root,
        pipeline_a_path=pipeline_path,
    )
    second = analysis.compute_replay_and_aggregate(
        contract,
        cohort,
        env_model=model,
        raw_root=raw_root,
        pipeline_a_path=pipeline_path,
    )

    first_csv = analysis.serialize_aggregate_csv(first.summary, float_format="%.17g")
    second_csv = analysis.serialize_aggregate_csv(second.summary, float_format="%.17g")
    assert first_csv == second_csv
    assert analysis.serialize_json(first.stage4_reconciliation) == analysis.serialize_json(second.stage4_reconciliation)


def test_contract_computations_reload_every_participant_capable_source_and_use_distinct_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two contract computations must share no cohort table, YAML model, raw load, or Pipeline A load."""
    analysis = _import_analysis()
    raw_root = tmp_path / "breath_data"
    raw_root.mkdir()
    t_first, h_first = _write_replay_raw_file(raw_root / "synthetic_first.csv", offset=0.0)
    t_second, h_second = _write_replay_raw_file(raw_root / "synthetic_second.csv", offset=10.0)
    sample_index_path = tmp_path / "sample_index.csv"
    qc_path = tmp_path / "qc_diagnostics.csv"
    pd.DataFrame(
        {
            "sample_id": [901, 902],
            "filename": ["synthetic_first.csv", "synthetic_second.csv"],
            "n_rows": [16, 16],
            "fallback_used": [False, False],
            "padded": [False, False],
            "in_final_output": [True, True],
        }
    ).to_csv(sample_index_path, index=False)
    pd.DataFrame(
        {
            "sample_id": [901, 902],
            "n_rows": [16, 16],
            "t_baseline_end": [3, 3],
            "fallback_used": [False, False],
            "env_temperature_sample": [t_first, t_second],
            "env_humidity_sample": [h_first, h_second],
            "n_available": [13, 13],
            "clip_length_L": [8, 8],
            "relative_deficit": [np.nan, np.nan],
            "padded": [False, False],
        }
    ).to_csv(qc_path, index=False)
    preprocessing_path = tmp_path / "preprocessing.yaml"
    preprocessing_path.write_text(
        yaml.safe_dump({"environmental_compensation": _synthetic_env_model().to_yaml_dict()}, sort_keys=False),
        encoding="utf-8",
    )
    contract = _small_contract()
    contract["population"].update(
        {
            "expected_source_count": 2,
            "expected_final_count": 2,
            "expected_excluded_count": 0,
            "expected_source_fallback_count": 0,
            "expected_final_fallback_count": 0,
            "expected_excluded_fallback_count": 0,
            "expected_padded_count": 0,
        }
    )
    contract["response_axis"].update({"last": 7, "clip_length": 8})
    contract["outputs"]["expected_aggregate_rows"] = 352

    stage4_rows: list[np.ndarray] = []
    seed_model = _synthetic_env_model()
    for filename, temperature, humidity in zip(
        ["synthetic_first.csv", "synthetic_second.csv"],
        [t_first, t_second],
        [h_first, h_second],
        strict=True,
    ):
        raw = analysis.load_raw_signal(filename, raw_root, expected_n_rows=16)
        replay = analysis.replay_participant(
            raw,
            t_baseline_end=3,
            persisted_temperature_sample=temperature,
            persisted_humidity_sample=humidity,
            expected_n_available=13,
            expected_padded=False,
            env_model=seed_model,
            clip_length=8,
            padding_threshold=0.0125,
            environmental_rtol=0.0,
            environmental_atol=1.0e-12,
        )
        stage4_rows.append(replay.stages[analysis.EXPECTED_STAGE_IDS[-1]])
    pipeline_path = tmp_path / "pipeline_a.npz"
    np.savez(
        pipeline_path,
        x=np.stack(stage4_rows),
        sample_id=np.array([901, 902], dtype=np.int64),
        label=np.array([0, 1], dtype=np.int64),
        channel_names=np.array(contract["channels"]),
        clip_length_L=np.int64(8),
    )

    counts = {"cohort": 0, "environment": 0, "raw": 0, "pipeline_a": 0}
    models: list[Any] = []
    original_cohort = analysis.load_minimal_cohort_tables
    original_environment = analysis.load_frozen_environmental_model
    original_raw = analysis.load_raw_signal
    original_pipeline_a = analysis.load_frozen_pipeline_a

    def counting_cohort(*args: Any, **kwargs: Any) -> Any:
        counts["cohort"] += 1
        return original_cohort(*args, **kwargs)

    def counting_environment(*args: Any, **kwargs: Any) -> Any:
        counts["environment"] += 1
        model = original_environment(*args, **kwargs)
        models.append(model)
        return model

    def counting_raw(*args: Any, **kwargs: Any) -> Any:
        counts["raw"] += 1
        return original_raw(*args, **kwargs)

    def counting_pipeline_a(*args: Any, **kwargs: Any) -> Any:
        counts["pipeline_a"] += 1
        return original_pipeline_a(*args, **kwargs)

    monkeypatch.setattr(analysis, "load_minimal_cohort_tables", counting_cohort)
    monkeypatch.setattr(analysis, "load_frozen_environmental_model", counting_environment)
    monkeypatch.setattr(analysis, "load_raw_signal", counting_raw)
    monkeypatch.setattr(analysis, "load_frozen_pipeline_a", counting_pipeline_a)

    first = analysis.compute_isolated_replay_and_aggregate(
        contract,
        sample_index_path=sample_index_path,
        qc_diagnostics_path=qc_path,
        preprocessing_config_path=preprocessing_path,
        raw_root=raw_root,
        pipeline_a_path=pipeline_path,
    )
    second = analysis.compute_isolated_replay_and_aggregate(
        contract,
        sample_index_path=sample_index_path,
        qc_diagnostics_path=qc_path,
        preprocessing_config_path=preprocessing_path,
        raw_root=raw_root,
        pipeline_a_path=pipeline_path,
    )

    assert counts == {"cohort": 2, "environment": 2, "raw": 4, "pipeline_a": 2}
    assert len(models) == 2
    assert models[0] is not models[1]
    assert analysis.serialize_aggregate_csv(first.summary, float_format="%.17g") == analysis.serialize_aggregate_csv(
        second.summary,
        float_format="%.17g",
    )
    assert analysis.serialize_json(first.stage4_reconciliation) == analysis.serialize_json(
        second.stage4_reconciliation
    )


def test_failure_code_taxonomy_is_closed_static_and_distinguishes_both_computations() -> None:
    """Only the reviewed fixed codes may be carried by the execution exception."""
    analysis = _import_analysis()

    assert {code.value for code in analysis.FailureCode} == EXPECTED_FAILURE_CODES
    assert analysis.FIRST_ISOLATED_FAILURE_CODES.raw_signal is analysis.FailureCode(
        "FIRST_RAW_SIGNAL_CONTAINMENT_LOADING_FAILURE"
    )
    assert analysis.SECOND_ISOLATED_FAILURE_CODES.raw_signal is analysis.FailureCode(
        "SECOND_RAW_SIGNAL_CONTAINMENT_LOADING_FAILURE"
    )
    assert analysis.FIRST_ISOLATED_FAILURE_CODES != analysis.SECOND_ISOLATED_FAILURE_CODES
    with pytest.raises(TypeError, match="FailureCode"):
        analysis.ReplayExecutionError("DYNAMIC_PRIVATE_CODE")
    private_cause = ValueError("synthetic private cause")
    with pytest.raises(analysis.ReplayExecutionError) as captured:
        with analysis._failure_boundary(analysis.FailureCode.FIRST_AGGREGATION_FAILURE):
            raise private_cause
    assert captured.value.failure_code is analysis.FailureCode.FIRST_AGGREGATION_FAILURE
    assert captured.value.__cause__ is private_cause


@pytest.mark.parametrize("computation", ["first", "second"])
@pytest.mark.parametrize(
    ("boundary", "code_field"),
    [
        ("cohort", "minimal_cohort"),
        ("entry_digest", "raw_digest_entry"),
        ("environmental_model", "environmental_model"),
        ("exit_digest", "raw_digest_exit"),
    ],
)
def test_isolated_source_boundaries_emit_computation_specific_codes(
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    code_field: str,
    computation: str,
) -> None:
    """Each isolated source boundary must retain first/second identity."""
    analysis = _import_analysis()
    failure_codes = (
        analysis.FIRST_ISOLATED_FAILURE_CODES
        if computation == "first"
        else analysis.SECOND_ISOLATED_FAILURE_CODES
    )
    rows = pd.DataFrame({"filename": ["synthetic.csv"]})
    cohort = analysis.CohortTables(final_rows=rows, source_count=1, final_count=1)
    digest_calls = 0

    def load_cohort(*_args: Any, **_kwargs: Any) -> Any:
        if boundary == "cohort":
            raise RuntimeError("synthetic private cohort failure")
        return cohort

    def digest(*_args: Any, **_kwargs: Any) -> str:
        nonlocal digest_calls
        digest_calls += 1
        if (boundary == "entry_digest" and digest_calls == 1) or (
            boundary == "exit_digest" and digest_calls == 2
        ):
            raise RuntimeError("synthetic private digest failure")
        return "a" * 64

    def load_environment(*_args: Any, **_kwargs: Any) -> object:
        if boundary == "environmental_model":
            raise RuntimeError("synthetic private YAML failure")
        return object()

    monkeypatch.setattr(analysis, "load_minimal_cohort_tables", load_cohort)
    monkeypatch.setattr(analysis, "compute_raw_cohort_digest", digest)
    monkeypatch.setattr(analysis, "load_frozen_environmental_model", load_environment)
    monkeypatch.setattr(
        analysis,
        "compute_replay_and_aggregate",
        lambda *_args, **_kwargs: analysis.ReplayComputation(pd.DataFrame(), {}),
    )
    contract = {
        "population": {
            "expected_source_count": 1,
            "expected_final_count": 1,
            "expected_excluded_count": 0,
            "expected_source_fallback_count": 0,
            "expected_final_fallback_count": 0,
            "expected_excluded_fallback_count": 0,
            "expected_padded_count": 0,
        }
    }

    with analysis._isolated_failure_scope(failure_codes):
        with pytest.raises(analysis.ReplayExecutionError) as captured:
            analysis.compute_isolated_replay_and_aggregate(
                contract,
                sample_index_path=Path("synthetic-index.csv"),
                qc_diagnostics_path=Path("synthetic-qc.csv"),
                preprocessing_config_path=Path("synthetic-preprocessing.yaml"),
                raw_root=Path("synthetic-raw"),
                pipeline_a_path=Path("synthetic-pipeline.npz"),
            )
    assert captured.value.failure_code is getattr(failure_codes, code_field)


@pytest.mark.parametrize("computation", ["first", "second"])
@pytest.mark.parametrize(
    ("boundary", "code_field"),
    [
        ("environmental_summary", "environmental_summary"),
        ("smoothing", "smoothing"),
        ("environmental_compensation", "environmental_compensation"),
        ("fractional_correction", "fractional_correction"),
        ("trimming", "trimming"),
    ],
)
def test_participant_transform_boundaries_emit_computation_specific_codes(
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    code_field: str,
    computation: str,
) -> None:
    """Each frozen participant transform boundary must have one static code."""
    analysis = _import_analysis()
    failure_codes = (
        analysis.FIRST_ISOLATED_FAILURE_CODES
        if computation == "first"
        else analysis.SECOND_ISOLATED_FAILURE_CODES
    )
    raw = analysis.RawSignal(
        gas_signals=np.arange(88, dtype=np.float64).reshape(8, 11) + 100.0,
        temperature=np.arange(8, dtype=np.float64) + 20.0,
        humidity=np.arange(8, dtype=np.float64) + 40.0,
        n_rows=8,
    )

    class SyntheticModel:
        def apply(self, values: np.ndarray, *_args: Any) -> np.ndarray:
            if boundary == "environmental_compensation":
                raise RuntimeError("synthetic private compensation failure")
            return values.copy()

    def smoothing(values: np.ndarray) -> np.ndarray:
        if boundary == "smoothing":
            raise RuntimeError("synthetic private smoothing failure")
        return values.copy()

    def correction(values: np.ndarray, _baseline_end: int) -> np.ndarray:
        if boundary == "fractional_correction":
            raise RuntimeError("synthetic private fractional failure")
        return values.copy()

    original_trim = analysis.trim_response_segment

    def trimming(*args: Any, **kwargs: Any) -> Any:
        if boundary == "trimming":
            raise RuntimeError("synthetic private trimming failure")
        return original_trim(*args, **kwargs)

    monkeypatch.setattr(analysis, "apply_sg_smoothing", smoothing)
    monkeypatch.setattr(analysis, "apply_fractional_correction", correction)
    monkeypatch.setattr(analysis, "trim_response_segment", trimming)
    persisted_temperature = float(np.mean(raw.temperature[:2]))
    if boundary == "environmental_summary":
        persisted_temperature += 1.0

    with analysis._isolated_failure_scope(failure_codes):
        with pytest.raises(analysis.ReplayExecutionError) as captured:
            analysis.replay_participant(
                raw,
                t_baseline_end=2,
                persisted_temperature_sample=persisted_temperature,
                persisted_humidity_sample=float(np.mean(raw.humidity[:2])),
                expected_n_available=6,
                expected_padded=False,
                env_model=SyntheticModel(),
                clip_length=4,
                padding_threshold=0.0125,
                environmental_rtol=0.0,
                environmental_atol=1.0e-12,
            )
    assert captured.value.failure_code is getattr(failure_codes, code_field)


@pytest.mark.parametrize("computation", ["first", "second"])
@pytest.mark.parametrize(
    ("boundary", "code_field"),
    [
        ("raw_signal", "raw_signal"),
        ("tensor_validation", "trimming"),
        ("pipeline_a", "pipeline_a"),
        ("stage4", "stage4"),
        ("aggregation", "aggregation"),
    ],
)
def test_replay_assembly_boundaries_emit_computation_specific_codes(
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    code_field: str,
    computation: str,
) -> None:
    """Raw loading, assembly, reconciliation, and aggregation must be classified."""
    analysis = _import_analysis()
    failure_codes = (
        analysis.FIRST_ISOLATED_FAILURE_CODES
        if computation == "first"
        else analysis.SECOND_ISOLATED_FAILURE_CODES
    )
    contract = _small_contract()
    contract["population"].update(
        {
            "expected_final_count": 1,
            "expected_source_fallback_count": 0,
            "expected_final_fallback_count": 0,
            "expected_excluded_fallback_count": 0,
            "expected_padded_count": 0,
        }
    )
    contract["response_axis"].update({"last": 2, "clip_length": 3})
    rows = pd.DataFrame(
        {
            "sample_id": [901],
            "filename": ["synthetic.csv"],
            "n_rows": [5],
            "t_baseline_end": [2],
            "fallback_used": [False],
            "env_temperature_sample": [20.0],
            "env_humidity_sample": [40.0],
            "n_available": [3],
            "clip_length_L": [3],
            "relative_deficit": [np.nan],
            "padded": [False],
        }
    )
    cohort = analysis.CohortTables(final_rows=rows, source_count=1, final_count=1)
    stages = {stage_id: np.zeros((3, 11), dtype=np.float64) for stage_id in analysis.EXPECTED_STAGE_IDS}
    participant = analysis.ParticipantReplay(stages=stages, n_available=3, padded=False)
    frozen = analysis.FrozenPipelineA(
        x=np.zeros((1, 3, 11), dtype=np.float64),
        sample_ids=np.array([901], dtype=np.int64),
        channels=np.array(contract["channels"]),
        clip_length=3,
    )

    def raw_loader(*_args: Any, **_kwargs: Any) -> Any:
        if boundary == "raw_signal":
            raise RuntimeError("synthetic private raw failure")
        return object()

    def validate_tensors(*_args: Any, **_kwargs: Any) -> None:
        if boundary == "tensor_validation":
            raise RuntimeError("synthetic private tensor failure")

    def pipeline_loader(*_args: Any, **_kwargs: Any) -> Any:
        if boundary == "pipeline_a":
            raise RuntimeError("synthetic private Pipeline A failure")
        return frozen

    def reconcile(*_args: Any, **_kwargs: Any) -> dict[str, bool]:
        if boundary == "stage4":
            raise RuntimeError("synthetic private Stage 4 failure")
        return {"passed": True}

    def aggregate(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        if boundary == "aggregation":
            raise RuntimeError("synthetic private aggregate failure")
        return pd.DataFrame()

    monkeypatch.setattr(analysis, "load_raw_signal", raw_loader)
    monkeypatch.setattr(analysis, "replay_participant", lambda *_args, **_kwargs: participant)
    monkeypatch.setattr(analysis, "validate_stage_tensors", validate_tensors)
    monkeypatch.setattr(analysis, "load_frozen_pipeline_a", pipeline_loader)
    monkeypatch.setattr(analysis, "reconcile_stage4", reconcile)
    monkeypatch.setattr(analysis, "aggregate_stage_tensors", aggregate)

    with analysis._isolated_failure_scope(failure_codes):
        with pytest.raises(analysis.ReplayExecutionError) as captured:
            analysis.compute_replay_and_aggregate(
                contract,
                cohort,
                env_model=object(),
                raw_root=Path("synthetic-raw"),
                pipeline_a_path=Path("synthetic-pipeline.npz"),
            )
    assert captured.value.failure_code is getattr(failure_codes, code_field)


def _configure_synthetic_run_orchestration(
    analysis: Any,
    monkeypatch: pytest.MonkeyPatch,
    failure_boundary: str | None,
) -> dict[str, list[Any]]:
    """Patch authorized orchestration with aggregate-only synthetic stand-ins."""
    contract = {
        "approved_plan": {
            "path": "docs/plans/efigure4_preprocessing_stage_aggregate_replay_recovery_v4_plan.md",
            "sha256": APPROVED_PLAN_SHA256,
        },
        "original_plan": {
            "path": "docs/plans/efigure4_preprocessing_stage_aggregate_replay_results_free_plan.md",
            "sha256": ORIGINAL_PLAN_SHA256,
        },
        "reporting_correction": {
            "path": "docs/preprocessing/preprocessing_fallback_denominator_reporting_correction.md",
            "sha256": REPORTING_CORRECTION_SHA256,
            "byte_size": 3374,
        },
        "recovery_v2_plan": {
            "path": "docs/plans/efigure4_preprocessing_stage_aggregate_replay_recovery_v2_plan.md",
            "sha256": RECOVERY_V2_PLAN_SHA256,
        },
        "recovery_v3_plan": {
            "path": "docs/plans/efigure4_preprocessing_stage_aggregate_replay_recovery_v3_plan.md",
            "sha256": RECOVERY_V3_PLAN_SHA256,
        },
        "numerical_environment_correction": {
            "path": "docs/preprocessing/preprocessing_stage4_numerical_environment_build_drift_correction.md",
            "sha256": NUMERICAL_ENVIRONMENT_CORRECTION_SHA256,
        },
        "channel_metadata_correction": {
            "path": "docs/preprocessing/preprocessing_stage4_conda_explicit_archive_channel_metadata_correction.md",
            "sha256": CHANNEL_METADATA_CORRECTION_SHA256,
        },
        "environment_lock": {
            "path": "configs/efigure4_preprocessing_stage_aggregate_replay_environment_lock.json",
            "sha256": ENVIRONMENT_LOCK_SHA256,
        },
        "population": {"expected_final_count": 1},
        "response_axis": {"clip_length": 3},
        "outputs": {
            "final_directory": "outputs/analysis/synthetic-only",
            "candidate_one_directory": "outputs/analysis/.synthetic-only.candidate-one",
            "candidate_two_directory": "outputs/analysis/.synthetic-only.candidate-two",
        },
        "inputs": {
            "sample_index": {"path": "synthetic-index.csv"},
            "qc_diagnostics": {"path": "synthetic-qc.csv"},
            "preprocessing_config": {"path": "synthetic-preprocessing.yaml"},
            "pipeline_a": {"path": "synthetic-pipeline.npz"},
        },
        "serialization": {"csv_float_format": "%.17g"},
    }
    implementation_hashes = {
        "config_sha256": "a" * 64,
        "module_sha256": "b" * 64,
        "script_sha256": "c" * 64,
    }
    state: dict[str, list[Any]] = {
        "environment_calls": [],
        "source_calls": [],
        "isolated_calls": [],
        "write_calls": [],
        "read_calls": [],
        "scopes": [],
    }
    payloads = {
        "stage_channel_frame_summary.csv": b"summary",
        "aggregate_audit.json": b"audit",
        "run_summary.md": b"summary-md",
        "manifest.json": b"manifest",
    }

    def collect_authority(*_args: Any, **_kwargs: Any) -> dict[str, str]:
        if failure_boundary == "git_authority":
            raise RuntimeError("synthetic private Git failure")
        return {"head": "synthetic-head"}

    def capture_sources(*_args: Any, **_kwargs: Any) -> dict[str, dict[str, str]]:
        state["source_calls"].append("capture")
        if failure_boundary == "source_identity" and len(state["source_calls"]) == 1:
            raise RuntimeError("synthetic private source failure")
        if failure_boundary == "final_revalidation" and len(state["source_calls"]) == 2:
            raise RuntimeError("synthetic private final source failure")
        return {"synthetic": {"sha256": "d" * 64}}

    def isolated(*_args: Any, **_kwargs: Any) -> Any:
        state["isolated_calls"].append("compute")
        state["scopes"].append(analysis._ACTIVE_ISOLATED_FAILURE_CODES.get())
        digest = "e" * 64
        if failure_boundary == "cross_computation" and len(state["isolated_calls"]) == 2:
            digest = "f" * 64
        return analysis.IsolatedReplayComputation(
            summary=pd.DataFrame({"synthetic": [1]}),
            stage4_reconciliation={"passed": True},
            raw_digest_entry=digest,
            raw_digest_exit=digest,
            private_filenames=("synthetic.csv",),
        )

    def build_audit(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        if failure_boundary == "audit_payload":
            raise RuntimeError("synthetic private audit failure")
        return {"population_counts": {"final_cohort": 1}}

    def write_candidate(*_args: Any, **_kwargs: Any) -> None:
        state["write_calls"].append("write")
        if failure_boundary == "first_candidate" and len(state["write_calls"]) == 1:
            raise RuntimeError("synthetic private first candidate failure")
        if failure_boundary == "second_candidate" and len(state["write_calls"]) == 2:
            raise RuntimeError("synthetic private second candidate failure")

    def read_candidate(*_args: Any, **_kwargs: Any) -> dict[str, bytes]:
        state["read_calls"].append("read")
        if failure_boundary == "serialized_read_back" and len(state["read_calls"]) == 1:
            raise RuntimeError("synthetic private read-back failure")
        if failure_boundary == "candidate_comparison" and len(state["read_calls"]) == 2:
            return {**payloads, "manifest.json": b"different"}
        return payloads

    def validate_environment(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        state["environment_calls"].append("validate")
        if failure_boundary == "environment":
            raise RuntimeError("synthetic private environment failure")
        return {
            "environment_lock_sha256": ENVIRONMENT_LOCK_SHA256,
            "critical_numerical_builds": [],
            "package_ledger_equality_passed": True,
            "archive_hash_validation_passed": True,
        }

    def promote(*_args: Any, **_kwargs: Any) -> None:
        if failure_boundary == "promotion":
            raise RuntimeError("synthetic private promotion failure")

    def synthetic_sha256(path: Path, *_args: Any, **_kwargs: Any) -> str:
        relative = Path(path).name
        if relative == "efigure4_preprocessing_stage_aggregate_replay_results_free_plan.md":
            return ORIGINAL_PLAN_SHA256
        if relative == "preprocessing_fallback_denominator_reporting_correction.md":
            return REPORTING_CORRECTION_SHA256
        if relative == "efigure4_preprocessing_stage_aggregate_replay_recovery_v2_plan.md":
            return RECOVERY_V2_PLAN_SHA256
        if relative == "efigure4_preprocessing_stage_aggregate_replay_recovery_v3_plan.md":
            return RECOVERY_V3_PLAN_SHA256
        if relative == "preprocessing_stage4_numerical_environment_build_drift_correction.md":
            return NUMERICAL_ENVIRONMENT_CORRECTION_SHA256
        if relative == "preprocessing_stage4_conda_explicit_archive_channel_metadata_correction.md":
            return CHANNEL_METADATA_CORRECTION_SHA256
        if relative == "efigure4_preprocessing_stage_aggregate_replay_environment_lock.json":
            return ENVIRONMENT_LOCK_SHA256
        return APPROVED_PLAN_SHA256

    monkeypatch.setattr(analysis, "load_and_validate_contract", lambda *_args, **_kwargs: contract)
    monkeypatch.setattr(analysis, "hash_implementation_files", lambda *_args, **_kwargs: implementation_hashes)
    monkeypatch.setattr(analysis, "build_execution_authorization", lambda *_args, **_kwargs: "synthetic-gate")
    monkeypatch.setattr(analysis, "sha256_file", synthetic_sha256)
    monkeypatch.setattr(analysis, "collect_runtime_authority", collect_authority)
    monkeypatch.setattr(analysis, "validate_authority_snapshot", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(analysis, "require_absent_output_namespace", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(analysis, "validate_runtime_environment", validate_environment, raising=False)
    monkeypatch.setattr(analysis, "collect_package_versions", lambda: {"python": "synthetic"})
    monkeypatch.setattr(analysis, "capture_source_identities", capture_sources)
    monkeypatch.setattr(analysis, "compute_isolated_replay_and_aggregate", isolated)
    monkeypatch.setattr(analysis, "serialize_aggregate_csv", lambda *_args, **_kwargs: b"same")
    monkeypatch.setattr(analysis, "serialize_json", lambda *_args, **_kwargs: b"same")
    monkeypatch.setattr(analysis, "require_source_identity_match", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(analysis, "build_aggregate_audit", build_audit)
    monkeypatch.setattr(analysis, "build_candidate_payloads", lambda *_args, **_kwargs: payloads)
    monkeypatch.setattr(analysis, "write_candidate_directory", write_candidate)
    monkeypatch.setattr(analysis, "read_and_validate_candidate_directory", read_candidate)
    monkeypatch.setattr(analysis, "sha256_bytes", lambda *_args, **_kwargs: "1" * 64)
    monkeypatch.setattr(analysis.shutil, "rmtree", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(analysis, "promote_candidate_directory", promote)
    return state


@pytest.mark.parametrize(
    ("boundary", "expected_code"),
    [
        ("git_authority", "GIT_AUTHORITY_OUTPUT_STATE_VALIDATION_FAILURE"),
        ("environment", "ENVIRONMENT_IDENTITY_VALIDATION_FAILURE"),
        ("source_identity", "FROZEN_SOURCE_IDENTITY_CAPTURE_FAILURE"),
        ("cross_computation", "CROSS_COMPUTATION_DETERMINISM_RAW_DIGEST_COMPARISON_FAILURE"),
        ("final_revalidation", "FINAL_SOURCE_IDENTITY_AUTHORITY_REVALIDATION_FAILURE"),
        ("audit_payload", "AUDIT_PAYLOAD_CONSTRUCTION_FAILURE"),
        ("first_candidate", "FIRST_CANDIDATE_WRITING_VALIDATION_FAILURE"),
        ("second_candidate", "SECOND_CANDIDATE_WRITING_VALIDATION_FAILURE"),
        ("serialized_read_back", "SERIALIZED_PRIVACY_COMPLETE_READ_BACK_VALIDATION_FAILURE"),
        ("candidate_comparison", "CANDIDATE_COMPARISON_FAILURE"),
        ("promotion", "ATOMIC_PROMOTION_FINAL_DIRECTORY_VALIDATION_FAILURE"),
    ],
)
def test_post_authorization_orchestration_boundaries_emit_fixed_codes(
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    expected_code: str,
) -> None:
    """Every nonparticipant orchestration boundary must emit its fixed code."""
    analysis = _import_analysis()
    _configure_synthetic_run_orchestration(analysis, monkeypatch, boundary)

    with pytest.raises(analysis.ReplayExecutionError) as captured:
        analysis.run_authorized_replay(
            authorization="synthetic-gate",
            project_root=PROJECT_ROOT,
            contract_path=CONTRACT_PATH,
        )
    assert captured.value.failure_code is analysis.FailureCode(expected_code)


def test_authorized_orchestration_selects_first_then_second_diagnostic_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two unchanged scientific calls must receive distinct diagnostic context."""
    analysis = _import_analysis()
    state = _configure_synthetic_run_orchestration(analysis, monkeypatch, None)

    result = analysis.run_authorized_replay(
        authorization="synthetic-gate",
        project_root=PROJECT_ROOT,
        contract_path=CONTRACT_PATH,
    )

    assert state["scopes"] == [analysis.FIRST_ISOLATED_FAILURE_CODES, analysis.SECOND_ISOLATED_FAILURE_CODES]
    assert state["environment_calls"] == ["validate", "validate"]
    assert result["aggregate_row_count"] == 1


def test_environment_failure_stops_before_every_participant_capable_or_write_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The environment gate must precede sources, computations, and candidate creation."""
    analysis = _import_analysis()
    state = _configure_synthetic_run_orchestration(analysis, monkeypatch, "environment")

    with pytest.raises(analysis.ReplayExecutionError) as captured:
        analysis.run_authorized_replay(
            authorization="synthetic-gate",
            project_root=PROJECT_ROOT,
            contract_path=CONTRACT_PATH,
        )

    assert captured.value.failure_code is analysis.FailureCode.ENVIRONMENT_IDENTITY_VALIDATION_FAILURE
    assert state["environment_calls"] == ["validate"]
    assert state["source_calls"] == []
    assert state["isolated_calls"] == []
    assert state["write_calls"] == []


def test_cli_sanitizes_internal_private_error_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI should show one fixed code and no private cause or traceback."""
    analysis = _import_analysis()
    cli = importlib.import_module("scripts.run_efigure4_preprocessing_stage_aggregate_replay")
    private_message = (
        "raw-name-private.csv /tmp/private.csv C:\\private\\raw.csv "
        "\\\\wsl$\\Ubuntu\\private participant-901 2026-08-11 987654.321"
    )

    def fail_run(*args: Any, **kwargs: Any) -> Any:
        try:
            raise ValueError(private_message)
        except ValueError as exc:
            raise analysis.ReplayExecutionError(
                analysis.FailureCode.FIRST_RAW_SIGNAL_CONTAINMENT_LOADING_FAILURE
            ) from exc

    monkeypatch.setattr(cli, "run_authorized_replay", fail_run)
    exit_code = cli.main(["--authorization", "synthetic-invalid-gate"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == (
        "ERROR: Aggregate replay execution failed. "
        "FAILURE_CODE=FIRST_RAW_SIGNAL_CONTAINMENT_LOADING_FAILURE\n"
    )
    assert not any(private_value in captured.err for private_value in private_message.split())
    assert "Traceback" not in captured.err


def test_cli_maps_unexpected_private_exception_only_to_unclassified_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An unexpected exception must expose only the fixed fallback code."""
    cli = importlib.import_module("scripts.run_efigure4_preprocessing_stage_aggregate_replay")

    def fail_unexpected(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("/var/private/raw.csv participant-777 2026-08-11 123456.789")

    monkeypatch.setattr(cli, "run_authorized_replay", fail_unexpected)
    exit_code = cli.main(["--authorization", "synthetic-invalid-gate"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == "ERROR: Aggregate replay execution failed. FAILURE_CODE=UNCLASSIFIED_FAILURE\n"
    assert "Traceback" not in captured.err
    assert "private" not in captured.err.lower()
    assert "777" not in captured.err
    assert "2026-08-11" not in captured.err
    assert "123456.789" not in captured.err


def test_successful_cli_output_remains_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Diagnostic classification must not alter successful CLI output."""
    cli = importlib.import_module("scripts.run_efigure4_preprocessing_stage_aggregate_replay")
    result = {
        "final_output_directory": "outputs/analysis/synthetic",
        "aggregate_row_count": 24,
        "manifest_sha256": "a" * 64,
    }
    monkeypatch.setattr(cli, "run_authorized_replay", lambda **_kwargs: result)

    exit_code = cli.main(["--authorization", "synthetic-gate"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out == (
        "Aggregate replay execution completed.\n"
        "Final output: outputs/analysis/synthetic\n"
        "Aggregate rows: 24\n"
        f"Manifest SHA-256: {'a' * 64}\n"
    )


def test_execution_authorization_line_uses_current_plan_config_module_and_script_hashes() -> None:
    """The Gate 2 line should bind the plan, implementation, CLI, and environment lock."""
    analysis = _import_analysis()
    implementation_hashes = analysis.hash_implementation_files(PROJECT_ROOT, CONTRACT_PATH)
    line = analysis.build_execution_authorization(
        APPROVED_PLAN_SHA256,
        implementation_hashes,
        ENVIRONMENT_LOCK_SHA256,
    )

    assert line == (
        "AUTHORIZE_EFIGURE4_AGGREGATE_REPLAY_EXECUTION "
        f"PLAN_SHA256={APPROVED_PLAN_SHA256} "
        f"CONFIG_SHA256={implementation_hashes['config_sha256']} "
        f"MODULE_SHA256={implementation_hashes['module_sha256']} "
        f"SCRIPT_SHA256={implementation_hashes['script_sha256']} "
        f"ENVIRONMENT_LOCK_SHA256={ENVIRONMENT_LOCK_SHA256}"
    )


def test_wrong_execution_authorization_fails_before_participant_access(monkeypatch: pytest.MonkeyPatch) -> None:
    """A mismatched Gate 2 token must halt before any participant-capable operation."""
    analysis = _import_analysis()

    def prohibited(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("participant-capable operation reached")

    monkeypatch.setattr(analysis, "load_minimal_cohort_tables", prohibited)
    monkeypatch.setattr(analysis, "compute_raw_cohort_digest", prohibited)
    monkeypatch.setattr(analysis, "compute_replay_and_aggregate", prohibited)
    monkeypatch.setattr(analysis, "validate_runtime_environment", prohibited, raising=False)

    with pytest.raises(analysis.ReplayExecutionError) as captured:
        analysis.run_authorized_replay(
            authorization="not-the-exact-gate",
            project_root=PROJECT_ROOT,
            contract_path=CONTRACT_PATH,
        )
    assert captured.value.failure_code is analysis.FailureCode.AUTHORIZATION_PLAN_IMPLEMENTATION_HASH_VALIDATION_FAILURE


def test_read_only_runtime_authority_collection_matches_current_approved_scope() -> None:
    """Read-only authority inspection should accept only the exact current lifecycle state."""
    analysis = _import_analysis()
    contract = _load_contract()
    actual = analysis.collect_runtime_authority(contract, PROJECT_ROOT)
    implementation_hashes = analysis.hash_implementation_files(PROJECT_ROOT, CONTRACT_PATH)

    _validate_live_repository_lifecycle_state(
        analysis,
        contract,
        actual,
        implementation_hashes=implementation_hashes,
        project_root=PROJECT_ROOT,
    )


def test_runtime_authority_observes_worktree_with_all_untracked_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runtime worktree observation must use the exact noncollapsing Git-status interface."""
    analysis = _import_analysis()
    contract = _load_contract()
    observed_arguments: list[tuple[str, ...]] = []
    original_run_git = analysis._run_git

    def observe(project_root: Path, arguments: list[str], *, check: bool = True) -> Any:
        observed_arguments.append(tuple(arguments))
        return original_run_git(project_root, arguments, check=check)

    monkeypatch.setattr(analysis, "_run_git", observe)
    actual = analysis.collect_runtime_authority(contract, PROJECT_ROOT)
    implementation_hashes = analysis.hash_implementation_files(PROJECT_ROOT, CONTRACT_PATH)
    _validate_live_repository_lifecycle_state(
        analysis,
        contract,
        actual,
        implementation_hashes=implementation_hashes,
        project_root=PROJECT_ROOT,
    )

    assert ("status", "--short", "--untracked-files=all") in observed_arguments
    assert ("status", "--short", "--untracked-files=normal") not in observed_arguments


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_output",
        "extra_output",
        "candidate_present",
        "unexpected_worktree_path",
        "altered_authority",
    ],
)
def test_post_promotion_lifecycle_validation_rejects_every_state_tamper(tmp_path: Path, mutation: str) -> None:
    """Post-promotion validation must reject incomplete, stale, or unauthorized state."""
    analysis = _import_analysis()
    contract = _load_contract()
    final_path = tmp_path / contract["outputs"]["final_directory"]
    final_path.mkdir(parents=True)
    for filename in contract["outputs"]["allowed_filenames"]:
        (final_path / filename).write_bytes(b"synthetic aggregate-only fixture")
    actual = _authority_snapshot(contract)
    actual["final_output_absent"] = False
    actual["worktree_entries"] = [
        *contract["runtime_authority"]["allowed_worktree_entries"],
        *[
            f"{contract['outputs']['final_directory']}/{filename}"
            for filename in contract["outputs"]["allowed_filenames"]
        ],
    ]
    if mutation == "missing_output":
        (final_path / contract["outputs"]["allowed_filenames"][0]).unlink()
    elif mutation == "extra_output":
        (final_path / "unexpected.json").write_bytes(b"synthetic unexpected fixture")
    elif mutation == "candidate_present":
        candidate = tmp_path / contract["outputs"]["candidate_one_directory"]
        candidate.mkdir(parents=True)
    elif mutation == "unexpected_worktree_path":
        actual["worktree_entries"].append("outputs/analysis/unapproved.txt")
    else:
        actual["head"] = "0" * 40

    with pytest.raises(ValueError):
        _validate_live_repository_lifecycle_state(
            analysis,
            contract,
            actual,
            implementation_hashes=EXPECTED_IMPLEMENTATION_HASHES,
            project_root=tmp_path,
        )


def test_package_version_collection_is_exactly_the_required_computation_set() -> None:
    """Runtime evidence should name Python, NumPy, pandas, SciPy, and PyYAML only."""
    analysis = _import_analysis()
    versions = analysis.collect_package_versions()
    assert versions == {
        "python": "3.11.15",
        "numpy": "2.3.5",
        "pandas": "2.3.3",
        "scipy": "1.17.1",
        "pyyaml": "6.0.3",
    }


def test_source_identity_capture_keeps_config_sha256_null_and_uses_blake2b(tmp_path: Path) -> None:
    """The known configuration anomaly must use size/blob/BLAKE2b and a null SHA-256."""
    analysis = _import_analysis()
    config_path = tmp_path / "config.yaml"
    ordinary_path = tmp_path / "ordinary.txt"
    config_path.write_bytes(b"nonempty synthetic config")
    ordinary_path.write_bytes(b"ordinary synthetic source")
    blake2b = analysis.hashlib.blake2b(config_path.read_bytes()).hexdigest()
    contract = {
        "inputs": {
            "preprocessing_config": {
                "path": "config.yaml",
                "git_blob": "1" * 40,
                "sha256": None,
                "sha256_status": "unresolved local SHA-256 computation/provenance anomaly",
                "byte_size": config_path.stat().st_size,
                "blake2b": blake2b,
            },
            "ordinary": {"path": "ordinary.txt", "git_blob": "2" * 40},
        }
    }

    identities = analysis.capture_source_identities(contract, tmp_path)

    assert identities["preprocessing_config"] == {
        "path": "config.yaml",
        "git_blob": "1" * 40,
        "sha256": None,
        "sha256_status": "unresolved local SHA-256 computation/provenance anomaly",
        "byte_size": config_path.stat().st_size,
        "blake2b": blake2b,
    }
    assert identities["ordinary"]["sha256"] == analysis.sha256_file(ordinary_path)
