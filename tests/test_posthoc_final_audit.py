"""Synthetic tests for Session 3 integrity-audit and promotion infrastructure."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from src.analysis import posthoc_final_audit as audit
from src.analysis import posthoc_final_reporting as reporting
from src.analysis.posthoc_final_common import ACCESS_STATE_KEYS, wilson_proportion_ci
from src.config import PROJECT_ROOT
from tests.test_posthoc_final_reporting import (
    build_synthetic_session6_root,
    rebind_synthetic_completion_output_hash,
)


CONFIG_PATH = PROJECT_ROOT / "configs/post_hoc_final_supplementary.yaml"
SESSION0_FREEZE = "3c0ff6b3b40460a2c340ad3193222ee54979109d"
AMENDMENT_FREEZE = "2e08e8fb89d99bb62cca2e5204a13ff09e47d432"
SHA = "a" * 64
COMMIT = "b" * 40


@pytest.fixture(scope="module")
def config() -> dict[str, Any]:
    """Load the frozen result-free machine contract."""
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _ledger(values: dict[str, bool] | None = None) -> OrderedDict[str, bool]:
    """Build one exact ordered access-ledger scope."""
    values = values or {}
    return OrderedDict((key, bool(values.get(key, False))) for key in ACCESS_STATE_KEYS)


def _access_state(
    activity: dict[str, bool] | None = None,
    cumulative: dict[str, bool] | None = None,
) -> OrderedDict[str, OrderedDict[str, bool]]:
    """Build the exact two-scope ordered access-state object."""
    return OrderedDict(
        [
            ("session_activity", _ledger(activity)),
            ("cumulative_through_session", _ledger(cumulative if cumulative is not None else activity)),
        ]
    )


def _reference_frame() -> pd.DataFrame:
    """Build one aggregate CSV frame with frozen nullable dtypes."""
    return pd.DataFrame(
        {
            "variable": pd.Series(["age"], dtype="string"),
            "level": pd.Series([pd.NA], dtype="string"),
            "group": pd.Series(["included"], dtype="category"),
            "n_total": pd.Series([10], dtype="Int64"),
            "n_nonmissing": pd.Series([9], dtype="Int64"),
            "n_missing": pd.Series([1], dtype="Int64"),
            "count": pd.Series([pd.NA], dtype="Int64"),
            "proportion": pd.Series([pd.NA], dtype="Float64"),
            "mean": pd.Series([50.0], dtype="Float64"),
            "sd": pd.Series([4.0], dtype="Float64"),
            "median": pd.Series([50.0], dtype="Float64"),
            "q1": pd.Series([47.0], dtype="Float64"),
            "q3": pd.Series([53.0], dtype="Float64"),
        }
    )


def _valid_execution_manifest(config: dict[str, Any]) -> OrderedDict[str, Any]:
    """Build a complete synthetic execution manifest from frozen inventories."""
    inventory = config["hash_map_inventories"]
    return OrderedDict(
        [
            ("schema_version", "posthoc_final_execution_manifest.v1"),
            ("package_id", "posthoc_final_supplementary"),
            ("generated_at_utc", "2030-01-02T03:04:05Z"),
            ("freeze_commit", SESSION0_FREEZE),
            ("commands", ["python scripts/run_posthoc_final_reporting.py --synthetic"]),
            (
                "environment",
                OrderedDict(
                    [
                        ("conda_environment_name", "gcvocs"),
                        ("python_version", "3.11.15"),
                        (
                            "package_versions",
                            OrderedDict(
                                [
                                    ("scikit-learn", "1.7.2"),
                                    ("statsmodels", "0.14.6"),
                                    ("scipy", "1.17.1"),
                                    ("numpy", "2.3.5"),
                                    ("pandas", "2.3.3"),
                                    ("sktime", "0.40.1"),
                                ]
                            ),
                        ),
                        (
                            "runtime_controls",
                            OrderedDict(
                                [
                                    ("OMP_NUM_THREADS", "1"),
                                    ("OPENBLAS_NUM_THREADS", "1"),
                                    ("MKL_NUM_THREADS", "1"),
                                    ("NUMEXPR_NUM_THREADS", "1"),
                                    ("analysis_3_sklearn_n_jobs", 1),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
            (
                "input_sha256",
                OrderedDict((key, SHA) for key in inventory["execution_manifest_input_sha256_keys_exact_order"]),
            ),
            (
                "output_sha256",
                OrderedDict((key, SHA) for key in inventory["execution_manifest_output_sha256_keys_exact_order"]),
            ),
            (
                "code_sha256",
                OrderedDict((key, SHA) for key in inventory["execution_manifest_code_sha256_keys_exact_order"]),
            ),
            ("config_sha256", SHA),
            ("methods_sha256", SHA),
            (
                "deterministic_comparison_provenance",
                OrderedDict(
                    [
                        ("candidate_pair_count", 2),
                        ("scientific_payload_agreement", True),
                        ("shared_invocation_utc", "2030-01-02T03:04:05Z"),
                    ]
                ),
            ),
            ("access_state", _access_state()),
            ("deviations_or_nonestimability", []),
            ("self_hash_policy", "forbidden"),
        ]
    )


def _valid_integrity_audit(config: dict[str, Any]) -> OrderedDict[str, Any]:
    """Build a complete synthetic Session 7 audit payload."""
    registry = config["outputs"]["scientific_and_closure"]["integrity_audit"]["schema"]
    checks = [
        OrderedDict(
            [
                ("check_id", item["check_id"]),
                ("display_order", item["display_order"]),
                ("check_name", item["check_name"]),
                ("status", "pass"),
                ("reason_code", None),
                ("evidence_note", None),
            ]
        )
        for item in registry["leaf_contracts"]["checks"]["ordered_check_registry"]
    ]
    return OrderedDict(
        [
            ("schema_version", "posthoc_final_integrity_audit.v1"),
            ("package_id", "posthoc_final_supplementary"),
            ("generated_at_utc", "2030-01-02T03:04:05Z"),
            ("freeze_commit", SESSION0_FREEZE),
            ("checks", checks),
            (
                "prior_artifact_sha256",
                OrderedDict(
                    (key, SHA)
                    for key in config["hash_map_inventories"]
                    ["integrity_audit_prior_artifact_sha256_keys_exact_order"]
                ),
            ),
            ("consolidated_report_unchanged_sha256", SHA),
            ("visualization_v2_authorized", False),
            ("access_state", _access_state()),
            ("self_hash_policy", "forbidden"),
        ]
    )


def _valid_release(config: dict[str, Any]) -> OrderedDict[str, Any]:
    """Build the future authorization payload without writing it."""
    return OrderedDict(
        [
            ("schema_version", "posthoc_final_visualization_release.v1"),
            ("package_id", "posthoc_final_supplementary"),
            ("reviewed_plan_path", config["authority"]["reviewed_plan_path"]),
            ("reviewed_plan_sha256", config["authority"]["reviewed_plan_sha256"]),
            ("config_sha256", SHA),
            ("methods_sha256", SHA),
            ("integrity_audit_sha256", SHA),
            ("integrity_audit_commit", COMMIT),
            ("audit_log_sha256", SHA),
            ("approval_status", "approved"),
            ("reviewer_designation_or_role", "Independent synthetic reviewer"),
            ("user_confirmation", True),
            ("authorization_date_utc", "2030-01-02T03:04:05Z"),
        ]
    )


def _completion_log(config: dict[str, Any], session_name: str) -> str:
    """Build one synthetic completion receipt in the frozen line grammar."""
    binding = config["outputs"]["completion_logs"]["machine_validated_log_grammar"][
        "session_bindings_exact"
    ][session_name]
    boolean_tokens = "\t".join("false" for _ in ACCESS_STATE_KEYS)
    lines = [
        "header\tposthoc_final_completion_log.v1\tposthoc_final_supplementary\t2030-01-02T03:04:05Z",
        f"session_identity\t{session_name}\tcomplete",
        f"freeze_commit\t{SESSION0_FREEZE}",
    ]
    lines.extend(f"output_hash\t{path}\t{SHA}" for path in binding["ordered_output_inventory"])
    lines.extend(
        [
            "deviations_or_nonestimability_summary\tnone\t\t",
            f"access_state_session_activity\t{boolean_tokens}",
            f"access_state_cumulative_through_session\t{boolean_tokens}",
            "completion_receipt\t2030-01-02T03:04:05Z\tcomplete",
        ]
    )
    return "\n".join(lines) + "\n"


def test_audit_module_import_is_side_effect_free(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing audit infrastructure performs no validation, directory creation, or write."""
    before = set(tmp_path.iterdir())
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location("isolated_audit", Path(audit.__file__))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(tmp_path.iterdir()) == before


def test_audit_promotion_inventory_is_exact_json_plus_log(config: dict[str, Any]) -> None:
    """The production object model exposes both configured Session 7 destinations directly."""
    outputs, log_path = audit.audit_promotion_inventory(config)
    binding = config["outputs"]["completion_logs"]["machine_validated_log_grammar"][
        "session_bindings_exact"
    ]["audit"]
    assert outputs == tuple(binding["ordered_output_inventory"])
    assert outputs == ("outputs/analysis/posthoc_final_integrity_audit.json",)
    assert log_path == binding["log_path"]


def test_session3_access_ledger_is_exact_and_score_blind() -> None:
    """Session 3 activity truthfully records no real values, scores, result, fit, or privacy scan."""
    state = audit.build_session3_access_state()
    assert tuple(state) == ("session_activity", "cumulative_through_session")
    assert tuple(state["session_activity"]) == tuple(ACCESS_STATE_KEYS)
    assert not any(state["session_activity"].values())
    assert not any(state["cumulative_through_session"].values())
    audit.validate_access_state(state, expected_activity=_ledger(), cumulative_prior=_ledger())


def test_session3_activity_can_carry_forward_exact_historical_cumulative_state(config: dict[str, Any]) -> None:
    """Synthetic Session 3 activity is false while prior score-blind accesses remain cumulative."""
    prior = OrderedDict(
        config["historical_consolidation"]["completed_session_1"]["score_blind_preflight"]["access_state"]
        ["cumulative_through_session"]
    )
    state = audit.build_session3_access_state(prior)
    assert not any(state["session_activity"].values())
    assert state["cumulative_through_session"] == prior
    audit.validate_access_state(state, expected_activity=_ledger(), cumulative_prior=prior)


def test_access_ledger_union_and_impossible_dependencies() -> None:
    """Cumulative union is exact and score-derived/fit/privacy states require their prerequisite accesses."""
    prior = _ledger({"participant_level_label_values_read": True})
    activity = _ledger({"participant_level_score_values_read": True, "new_score_derived_results_computed": True})
    state = audit.build_access_state(activity, prior)
    assert state["cumulative_through_session"]["participant_level_label_values_read"]
    assert state["cumulative_through_session"]["participant_level_score_values_read"]
    audit.validate_access_state(state, expected_activity=activity, cumulative_prior=prior)
    impossible = _access_state({"new_score_derived_results_computed": True})
    with pytest.raises(ValueError, match="score-derived"):
        audit.validate_access_state(impossible)
    impossible_fit = _access_state({"auxiliary_model_fit_executed": True})
    with pytest.raises(ValueError, match="model fit"):
        audit.validate_access_state(impossible_fit)
    impossible_privacy = _access_state({"privacy_reference_loader_executed": True})
    with pytest.raises(ValueError, match="privacy"):
        audit.validate_access_state(impossible_privacy)

    externally_validated_prior = _access_state(
        {},
        {"participant_level_label_values_read": True, "participant_level_date_values_read": True},
    )
    audit.validate_access_state(externally_validated_prior)


def test_session6_and_session7_access_ledgers_are_truthful_cumulative_unions() -> None:
    """Later-session ledgers preserve history and record their actual authorized access patterns."""
    prior = _ledger(
        {
            "prediction_container_bytes_hashed": True,
            "participant_level_label_values_read": True,
            "participant_level_score_values_read": True,
            "new_score_derived_results_computed": True,
        }
    )
    session6 = audit.build_reporting_access_state(prior)
    assert session6["session_activity"] == _ledger(
        {
            "prediction_container_bytes_hashed": True,
            "prediction_schema_or_header_inspected": True,
            "participant_level_label_values_read": True,
            "participant_level_score_values_read": True,
            "participant_level_date_values_read": True,
            "participant_level_qc_values_read": True,
            "new_score_derived_results_computed": True,
            "privacy_reference_loader_executed": True,
        }
    )
    assert session6["session_activity"]["auxiliary_model_fit_executed"] is False
    session7 = audit.build_audit_access_state(session6["cumulative_through_session"])
    assert session7["session_activity"] == _ledger(
        {
            "participant_level_date_values_read": True,
            "privacy_reference_loader_executed": True,
        }
    )
    assert session7["cumulative_through_session"] == session6["cumulative_through_session"]


def test_privacy_policy_names_and_timestamps_are_context_allowlisted(config: dict[str, Any]) -> None:
    """Approved policy field names and provenance timestamps do not create false positives."""
    payload = OrderedDict(
        [
            ("generated_at_utc", "2030-01-02T03:04:05Z"),
            (
                "policy",
                "Fields sample_id, filename, collection_time, and collection_location are prohibited in "
                "participant outputs.",
            ),
            ("artifact_path", "outputs/analysis/posthoc_final_integrity_audit.json"),
        ]
    )
    audit.validate_artifact_privacy(payload, artifact_kind="governance_manifest", config=config)


def test_privacy_policy_context_does_not_allow_values_dates_or_raw_filenames(config: dict[str, Any]) -> None:
    """Policy wording may name fields, but it cannot conceal a clinical date or raw filename value."""
    for value in ("2031-02-03", "synthetic_breath_0042.csv"):
        with pytest.raises(ValueError) as error:
            audit.validate_artifact_privacy(
                {"policy": f"Prohibited participant value: {value}"},
                artifact_kind="governance_manifest",
                config=config,
            )
        assert value not in str(error.value)


@pytest.mark.parametrize(
    "value",
    [
        "SYNTHETIC-PARTICIPANT-0042",
        "2031/2/3",
        "2031-02-03",
        "synthetic_breath_0042.csv",
        "/private/synthetic/repository/output.json",
        r"[PRIVATE_PATH_REDACTED]",
    ],
)
def test_privacy_rejects_values_dates_filenames_ids_and_absolute_paths_without_echo(
    value: str,
    config: dict[str, Any],
) -> None:
    """Privacy failures are sanitized and never echo the matched synthetic value."""
    references = audit.PrivacyReferences.from_columns(
        sample_ids=["SYNTHETIC-PARTICIPANT-0042"],
        filenames=["synthetic_breath_0042.csv"],
        collection_times=["2031/2/3"],
    )
    with pytest.raises(ValueError) as error:
        audit.validate_artifact_privacy(
            {"evidence_note": value},
            artifact_kind="structured_json",
            config=config,
            privacy_references=references,
        )
    assert value not in str(error.value)


def test_privacy_rejects_participant_row_schema(config: dict[str, Any]) -> None:
    """A synthetic participant row fails even though policy text may name its fields."""
    with pytest.raises(ValueError, match="participant_level_field_or_row"):
        audit.validate_artifact_privacy(
            [{"sample_id": 42, "p_cal": 0.6}],
            artifact_kind="scientific_csv",
            config=config,
        )


def test_privacy_scans_dataframes_serialized_bytes_and_embedded_paths(config: dict[str, Any]) -> None:
    """Every production representation crosses the same sanitized privacy boundary."""
    references = audit.PrivacyReferences.from_columns(
        sample_ids=[42, "SYNTH-ID-42"],
        filenames=["synthetic_secret.raw"],
        collection_times=["2031/2/3"],
    )
    unsafe_payloads = [
        (pd.DataFrame({"sample_id": ["SYNTH-ID-42"], "value": [0.2]}), "scientific_csv"),
        (b"sample_id,value\nSYNTH-ID-42,0.2\n", "scientific_csv"),
        ("sample_id,value\nSYNTH-ID-42,0.2\n", "scientific_csv"),
        (b'{"note":"SYNTH-ID-42"}', "structured_json"),
        ({"note": "SYNTH-ID-42"}, "governance_manifest"),
        ("prefix /private/synthetic/output.json suffix", "markdown_report"),
        ("synthetic row sample_id=42 was retained", "markdown_report"),
        (
            "header\tposthoc_final_completion_log.v1\tposthoc_final_supplementary\t2030-01-02T03:04:05Z\n"
            "evidence\tSYNTH-ID-42\n",
            "sanitized_log",
        ),
    ]
    for payload, artifact_kind in unsafe_payloads:
        with pytest.raises(audit.PrivacyValidationError) as error:
            audit.validate_artifact_privacy(
                payload,
                artifact_kind=artifact_kind,
                config=config,
                privacy_references=references,
                artifact_path="outputs/analysis/synthetic.json",
            )
        assert "SYNTH-ID-42" not in str(error.value)
        assert "/private/synthetic/output.json" not in str(error.value)
    audit.validate_artifact_privacy(
        pd.DataFrame({"aggregate_n": [42]}),
        artifact_kind="scientific_csv",
        config=config,
        privacy_references=references,
        artifact_path="outputs/analysis/aggregate.csv",
    )


def test_hash_policy_includes_every_configured_code_container(config: dict[str, Any]) -> None:
    """Manifest code-container subjects are approved whole containers, never value hashes."""
    for path in config["hash_map_inventories"]["execution_manifest_code_sha256_keys_exact_order"]:
        audit.validate_hash_subject("whole_container", path, config)


def test_privacy_reference_loader_is_explicit_isolated_and_no_echo(config: dict[str, Any]) -> None:
    """The deny-set loader requires authorization and requests exactly its three frozen columns."""
    calls: list[tuple[str, ...]] = []

    def reader(path: Path, *, usecols: list[str]) -> pd.DataFrame:
        calls.append(tuple(usecols))
        return pd.DataFrame(
            {
                "sample_id": ["SYNTH-1"],
                "filename": ["synthetic_file.csv"],
                "collection_time": ["2030/1/2"],
            }
        )

    with pytest.raises(PermissionError, match="explicit"):
        audit.load_privacy_references(Path("sample_index.csv"), config, authorized=False, reader=reader)
    assert calls == []
    references = audit.load_privacy_references(Path("sample_index.csv"), config, authorized=True, reader=reader)
    assert calls == [tuple(config["loaders"]["per_purpose_ordered_usecols"]["privacy_reference_sample_index"])]
    assert not hasattr(references, "sample_ids")
    with pytest.raises(ValueError) as error:
        audit.validate_artifact_privacy(
            {"note": "SYNTH-1"},
            artifact_kind="structured_json",
            config=config,
            privacy_references=references,
        )
    assert "SYNTH-1" not in str(error.value)

    private_loader_detail = "SYNTHETIC-PRIVATE-LOADER-VALUE"

    def failing_reader(path: Path, *, usecols: list[str]) -> pd.DataFrame:
        raise ValueError(private_loader_detail)

    with pytest.raises(ValueError) as loader_error:
        audit.load_privacy_references(
            Path("sample_index.csv"),
            config,
            authorized=True,
            reader=failing_reader,
        )
    assert private_loader_detail not in str(loader_error.value)


def test_hash_policy_accepts_declared_containers_and_rejects_individual_values(config: dict[str, Any]) -> None:
    """Whole-container hashes pass only in declared maps; individual field/value hashing is forbidden."""
    keys = config["hash_map_inventories"]["execution_manifest_output_sha256_keys_exact_order"]
    values = OrderedDict((key, SHA) for key in keys)
    audit.validate_hash_map(values, expected_keys=keys, forbidden_self_paths=())
    with pytest.raises(ValueError, match="individual"):
        audit.validate_hash_subject("individual_value", "sample_id", config)
    audit.validate_hash_subject("whole_container", "data/processed/sample_index.csv", config)


def test_execution_manifest_exact_schema_hash_inventories_and_non_circularity(config: dict[str, Any]) -> None:
    """The Session 6 manifest excludes itself/log and enforces exact ordered hash maps."""
    payload = _valid_execution_manifest(config)
    audit.validate_execution_manifest(payload, config)
    assert "outputs/analysis/posthoc_final_execution_manifest.json" not in payload["output_sha256"]
    assert "outputs/logs/posthoc_final_reporting.log" not in payload["output_sha256"]
    mutations: list[OrderedDict[str, Any]] = []
    reordered = copy.deepcopy(payload)
    reordered["input_sha256"] = OrderedDict(reversed(tuple(reordered["input_sha256"].items())))
    mutations.append(reordered)
    self_hash = copy.deepcopy(payload)
    self_hash["output_sha256"]["outputs/analysis/posthoc_final_execution_manifest.json"] = SHA
    mutations.append(self_hash)
    wrong_hash = copy.deepcopy(payload)
    first = next(iter(wrong_hash["code_sha256"]))
    wrong_hash["code_sha256"][first] = "not-a-hash"
    mutations.append(wrong_hash)
    wrong_state = copy.deepcopy(payload)
    wrong_state["deterministic_comparison_provenance"]["scientific_payload_agreement"] = False
    mutations.append(wrong_state)
    for mutation in mutations:
        with pytest.raises(ValueError):
            audit.validate_execution_manifest(mutation, config)
    wrong_type = copy.deepcopy(payload)
    wrong_type["freeze_commit"] = int("1" * 40)
    with pytest.raises(ValueError):
        audit.validate_execution_manifest(wrong_type, config)


def test_integrity_audit_exact_checks_hashes_additive_state_and_non_circularity(config: dict[str, Any]) -> None:
    """Session 7 hashes only prior artifacts, retains all checks, and fixes authorization false."""
    payload = _valid_integrity_audit(config)
    audit.validate_integrity_audit(payload, config)
    assert "outputs/analysis/posthoc_final_integrity_audit.json" not in payload["prior_artifact_sha256"]
    assert "outputs/logs/posthoc_final_audit.log" not in payload["prior_artifact_sha256"]
    changed = copy.deepcopy(payload)
    changed["checks"][0]["check_id"] = "IA02"
    with pytest.raises(ValueError, match="check"):
        audit.validate_integrity_audit(changed, config)
    changed = copy.deepcopy(payload)
    changed["checks"][0]["status"] = "fail"
    with pytest.raises(ValueError, match="reason"):
        audit.validate_integrity_audit(changed, config)
    changed = copy.deepcopy(payload)
    changed["visualization_v2_authorized"] = True
    with pytest.raises(ValueError, match="visualization"):
        audit.validate_integrity_audit(changed, config)
    changed = copy.deepcopy(payload)
    changed["checks"][0]["display_order"] = True
    with pytest.raises(ValueError, match="check"):
        audit.validate_integrity_audit(changed, config)


def test_ia_registry_executes_every_check_and_retains_sanitized_failure(config: dict[str, Any]) -> None:
    """No IA status defaults to pass; all twelve callables execute and one failure blocks certification."""
    registry = config["outputs"]["scientific_and_closure"]["integrity_audit"]["schema"][
        "leaf_contracts"
    ]["checks"]["ordered_check_registry"]
    events: list[str] = []
    secret = "SYNTHETIC-PRIVATE-MATCH"

    def make_check(name: str) -> Any:
        def check() -> str:
            events.append(name)
            if name == "privacy_and_path_sanitization":
                raise ValueError(secret)
            return f"executed:{name}"

        return check

    functions = OrderedDict((item["check_name"], make_check(item["check_name"])) for item in registry)
    checks = audit.execute_integrity_check_registry(config, functions)
    assert events == [item["check_name"] for item in registry]
    assert [item["status"] for item in checks].count("fail") == 1
    assert all(item["evidence_note"] for item in checks)
    failed = next(item for item in checks if item["status"] == "fail")
    assert failed["reason_code"] == "integrity_check_failed"
    assert secret not in failed["evidence_note"]
    assert secret not in json.dumps(checks)
    with pytest.raises(ValueError, match="failed IA check"):
        audit.require_all_integrity_checks_pass(checks)


@pytest.mark.parametrize("session_name", ["reporting", "audit"])
def test_reporting_and_audit_completion_logs_are_exact_and_non_circular(
    session_name: str,
    config: dict[str, Any],
) -> None:
    """Reporting hashes its manifest and audit hashes its JSON, while neither receipt hashes itself."""
    binding = config["outputs"]["completion_logs"]["machine_validated_log_grammar"][
        "session_bindings_exact"
    ][session_name]
    if session_name == "reporting":
        assert "outputs/analysis/posthoc_final_execution_manifest.json" in binding["ordered_output_inventory"]
    assert binding["log_path"] not in binding["ordered_output_inventory"]
    state = _ledger()
    audit.validate_session_completion_log(
        _completion_log(config, session_name),
        session_name=session_name,
        freeze_commit=SESSION0_FREEZE,
        expected_activity=state,
        expected_cumulative=state,
        cumulative_prior=state,
        config=config,
    )
    mutated = _completion_log(config, session_name).replace(
        binding["ordered_output_inventory"][0],
        binding["log_path"],
        1,
    )
    with pytest.raises(ValueError):
        audit.validate_session_completion_log(
            mutated,
            session_name=session_name,
            freeze_commit=SESSION0_FREEZE,
            expected_activity=state,
            expected_cumulative=state,
            cumulative_prior=state,
            config=config,
        )


def test_cross_artifact_hashes_and_report_identity_are_checked(config: dict[str, Any]) -> None:
    """Recorded hashes must equal prior bytes and the Session 6 report must remain unchanged."""
    payload = _valid_integrity_audit(config)
    prior_bytes = OrderedDict(
        (key, f"synthetic-{index}".encode())
        for index, key in enumerate(payload["prior_artifact_sha256"])
    )
    for key, value in prior_bytes.items():
        payload["prior_artifact_sha256"][key] = hashlib.sha256(value).hexdigest()
    report_path = config["outputs"]["scientific_and_closure"]["consolidated_analyses_document"]["path"]
    payload["consolidated_report_unchanged_sha256"] = payload["prior_artifact_sha256"][report_path]
    audit.validate_prior_artifact_hashes(payload, prior_bytes, config)
    mutated = OrderedDict(prior_bytes)
    mutated[report_path] += b"mutation"
    with pytest.raises(ValueError, match="hash"):
        audit.validate_prior_artifact_hashes(payload, mutated, config)


def test_entry_exit_integrity_snapshot_detects_intervening_mutation(tmp_path: Path) -> None:
    """Protected container bytes are rehashed and compared before the first promotion."""
    paths = ("inputs/one.bin", "inputs/two.bin")
    for index, relative in enumerate(paths):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic-{index}".encode())
    snapshot = audit.capture_integrity_snapshot(tmp_path, paths)
    audit.verify_integrity_snapshot(tmp_path, snapshot)
    (tmp_path / paths[1]).write_bytes(b"mutated")
    with pytest.raises(ValueError, match="changed between entry and exit"):
        audit.verify_integrity_snapshot(tmp_path, snapshot)


def test_csv_schema_dtype_order_and_required_rows_are_exact(config: dict[str, Any]) -> None:
    """Column order, nullable dtypes, and configured row identity cannot drift."""
    frame = _reference_frame()
    registry = copy.deepcopy(config["row_order_and_schema_validation_registry"]["included_excluded_summary"])
    registry["expected_row_count"] = 1
    audit.validate_csv_frame(frame, registry)
    with pytest.raises(ValueError, match="column order"):
        audit.validate_csv_frame(frame.loc[:, list(reversed(frame.columns))], registry)
    wrong_dtype = frame.copy()
    wrong_dtype["n_total"] = wrong_dtype["n_total"].astype("int64")
    with pytest.raises(ValueError, match="dtype"):
        audit.validate_csv_frame(wrong_dtype, registry)
    wrong_enum = frame.copy()
    wrong_enum["group"] = pd.Series(["unknown"], dtype="category")
    registry["allowed_values"] = {"group": ["included", "excluded"]}
    with pytest.raises(ValueError, match="enum"):
        audit.validate_csv_frame(wrong_enum, registry)


def test_target_state_stale_and_partial_are_detected_before_access(tmp_path: Path) -> None:
    """Unexpected targets and interrupted output-without-log states fail closed."""
    outputs = [tmp_path / "one.csv", tmp_path / "two.json"]
    log = tmp_path / "completion.log"
    assert audit.classify_promotion_state(outputs, log) == "clean"
    outputs[0].write_text("candidate", encoding="utf-8")
    assert audit.classify_promotion_state(outputs, log) == "partial"
    with pytest.raises(ValueError, match="partial"):
        audit.require_clean_promotion_state(outputs, log)
    outputs[1].write_text("candidate", encoding="utf-8")
    assert audit.classify_promotion_state(outputs, log) == "partial"
    log.write_text("receipt", encoding="utf-8")
    assert audit.classify_promotion_state(outputs, log) == "complete"
    with pytest.raises(ValueError, match="complete"):
        audit.require_clean_promotion_state(outputs, log)


def test_promotion_validates_all_candidates_then_promotes_atomically_log_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every candidate validates before any destination replacement and the receipt moves last."""
    target_dir = tmp_path / "outputs"
    target_dir.mkdir()
    candidates: list[audit.PromotionCandidate] = []
    events: list[str] = []
    for name in ("one.csv", "two.json"):
        candidate = target_dir / f".{name}.candidate"
        candidate.write_text(name, encoding="utf-8")
        target = target_dir / name
        candidates.append(
            audit.PromotionCandidate(
                relative_path=f"outputs/{name}",
                candidate_path=candidate,
                target_path=target,
                validator=lambda path, name=name: events.append(f"validate:{name}"),
            )
        )
    log_candidate = target_dir / ".completion.log.candidate"
    log_candidate.write_text("complete", encoding="utf-8")
    log_target = target_dir / "completion.log"
    real_replace = os.replace

    def recording_replace(source: Path, destination: Path) -> None:
        events.append(f"promote:{Path(destination).name}")
        real_replace(source, destination)

    monkeypatch.setattr(audit.os, "replace", recording_replace)
    expected_relative_paths = [candidate.relative_path for candidate in candidates]
    with pytest.raises(ValueError, match="order"):
        audit.promote_candidates(
            list(reversed(candidates)),
            expected_relative_paths=expected_relative_paths,
            log_candidate=log_candidate,
            log_target=log_target,
            log_validator=lambda path: events.append("validate:completion.log"),
        )
    assert events == []
    audit.promote_candidates(
        candidates,
        expected_relative_paths=expected_relative_paths,
        log_candidate=log_candidate,
        log_target=log_target,
        log_validator=lambda path: events.append("validate:completion.log"),
    )
    assert events == [
        "validate:one.csv",
        "validate:two.json",
        "validate:completion.log",
        "promote:one.csv",
        "promote:two.json",
        "promote:completion.log",
    ]


def test_promotion_binds_each_candidate_to_exact_project_target(tmp_path: Path) -> None:
    """A valid candidate cannot redirect an output to another destination."""
    output = tmp_path / "outputs" / "one.json"
    output.parent.mkdir(parents=True)
    candidate_path = output.with_name(".one.json.candidate")
    candidate_path.write_text("{}", encoding="utf-8")
    wrong_target = tmp_path / "elsewhere" / "one.json"
    log_target = tmp_path / "outputs" / "completion.log"
    log_candidate = log_target.with_name(".completion.log.candidate")
    log_candidate.write_text("complete", encoding="utf-8")
    candidate = audit.PromotionCandidate("outputs/one.json", candidate_path, wrong_target, lambda path: None)
    with pytest.raises(ValueError, match="exact configured target"):
        audit.promote_candidates(
            [candidate],
            expected_relative_paths=["outputs/one.json"],
            log_candidate=log_candidate,
            log_target=log_target,
            log_validator=lambda path: None,
            project_root=tmp_path,
        )


def test_promotion_validation_failure_writes_nothing_and_interruption_is_retained(tmp_path: Path) -> None:
    """Pre-promotion failure writes nothing; a mid-promotion interruption is not silently cleaned."""
    directory = tmp_path / "out"
    directory.mkdir()
    good = directory / ".good"
    bad = directory / ".bad"
    log_candidate = directory / ".log"
    for path in (good, bad, log_candidate):
        path.write_text("synthetic", encoding="utf-8")
    candidates = [
        audit.PromotionCandidate("out/good", good, directory / "good", lambda path: None),
        audit.PromotionCandidate(
            "out/bad",
            bad,
            directory / "bad",
            lambda path: (_ for _ in ()).throw(ValueError("invalid candidate")),
        ),
    ]
    with pytest.raises(ValueError, match="invalid"):
        audit.promote_candidates(
            candidates,
            expected_relative_paths=[candidate.relative_path for candidate in candidates],
            log_candidate=log_candidate,
            log_target=directory / "receipt",
            log_validator=lambda path: None,
        )
    assert not (directory / "good").exists()
    assert not (directory / "bad").exists()


def test_mid_promotion_interruption_retains_partial_state_without_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A replacement failure after one output leaves a detectable partial state and no completion log."""
    directory = tmp_path / "out"
    directory.mkdir()
    candidate_paths = [directory / ".one", directory / ".two"]
    target_paths = [directory / "one", directory / "two"]
    for path in candidate_paths:
        path.write_text("synthetic", encoding="utf-8")
    log_candidate = directory / ".receipt"
    log_target = directory / "receipt"
    log_candidate.write_text("complete", encoding="utf-8")
    candidates = [
        audit.PromotionCandidate(f"out/{target.name}", candidate, target, lambda path: None)
        for candidate, target in zip(candidate_paths, target_paths, strict=True)
    ]
    real_replace = os.replace
    calls = 0

    def interrupt(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic interruption")
        real_replace(source, destination)

    monkeypatch.setattr(audit.os, "replace", interrupt)
    with pytest.raises(OSError, match="interruption"):
        audit.promote_candidates(
            candidates,
            expected_relative_paths=[candidate.relative_path for candidate in candidates],
            log_candidate=log_candidate,
            log_target=log_target,
            log_validator=lambda path: None,
        )
    assert target_paths[0].exists()
    assert not target_paths[1].exists()
    assert not log_target.exists()
    assert audit.classify_promotion_state(target_paths, log_target) == "partial"


def test_session7_scope_is_additive_and_consolidated_report_immutable(config: dict[str, Any]) -> None:
    """Only the audit and its log may be added; the Session 6 report hash cannot change."""
    report_hash = SHA
    allowed = [
        config["outputs"]["scientific_and_closure"]["integrity_audit"]["path"],
        config["outputs"]["completion_logs"]["audit"],
    ]
    audit.validate_session7_additive_scope(
        changed_paths=allowed,
        report_hash_before=report_hash,
        report_hash_after=report_hash,
        config=config,
    )
    with pytest.raises(ValueError, match="additive"):
        audit.validate_session7_additive_scope(
            changed_paths=[
                *allowed,
                config["outputs"]["scientific_and_closure"]["consolidated_analyses_document"]["path"],
            ],
            report_hash_before=report_hash,
            report_hash_after="c" * 64,
            config=config,
        )


def test_future_visualization_release_schema_rejects_premature_or_incomplete_authorization(
    config: dict[str, Any],
) -> None:
    """The future release payload validates only with complete approved references and is never created here."""
    payload = _valid_release(config)
    audit.validate_visualization_release(payload, config, references_valid=True, file_exists=True)
    for mutation in ("approval", "confirmation", "reference", "absence", "order"):
        changed = copy.deepcopy(payload)
        references_valid = True
        file_exists = True
        if mutation == "approval":
            changed["approval_status"] = "pending"
        elif mutation == "confirmation":
            changed["user_confirmation"] = False
        elif mutation == "reference":
            references_valid = False
        elif mutation == "absence":
            file_exists = False
        else:
            changed = OrderedDict(reversed(tuple(changed.items())))
        with pytest.raises(ValueError):
            audit.validate_visualization_release(
                changed,
                config,
                references_valid=references_valid,
                file_exists=file_exists,
            )
    assert not (PROJECT_ROOT / config["outputs"]["scientific_and_closure"]["visualization_release"]["path"]).exists()


def test_audit_cli_has_dual_freeze_without_execute_authorization_flag() -> None:
    """The actual audit parser preserves both commit flags and has no execute approval flag."""
    from scripts import run_posthoc_final_audit as cli

    parser = cli.build_parser()
    actions = {option for action in parser._actions for option in action.option_strings}
    assert "--freeze-commit" in actions
    assert "--amendment-freeze-commit" in actions
    assert "--execute-approved" not in actions
    args = parser.parse_args(
        [
            "--config",
            "configs/post_hoc_final_supplementary.yaml",
            "--freeze-commit",
            SESSION0_FREEZE,
            "--amendment-freeze-commit",
            AMENDMENT_FREEZE,
        ]
    )
    assert args.config == Path("configs/post_hoc_final_supplementary.yaml")


def _complete_synthetic_reporting(
    tmp_path: Path,
    config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], Any]:
    """Run the real reporting CLI to establish a synthetic Session 6 handoff."""
    from scripts import run_posthoc_final_reporting as reporting_cli
    from src.analysis import posthoc_final_locked_scores as locked_scores

    synthetic_config, git_runner = build_synthetic_session6_root(tmp_path, config)
    monkeypatch.setattr(locked_scores, "PROJECT_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    assert reporting_cli.main(
        [
            "--config",
            "configs/post_hoc_final_supplementary.yaml",
            "--freeze-commit",
            SESSION0_FREEZE,
            "--amendment-freeze-commit",
            AMENDMENT_FREEZE,
            "--execute-approved",
        ],
        project_root=tmp_path,
        git_runner=git_runner,
        invocation_clock=lambda: "2030-01-02T03:04:05Z",
    ) == 0
    return synthetic_config, git_runner


def _synthetic_audit_context(
    tmp_path: Path,
    config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], Any, dict[str, Any], audit.PrivacyReferences, audit.AuditCheckContext]:
    """Establish and load a complete synthetic Session 6 audit handoff."""
    synthetic_config, git_runner = _complete_synthetic_reporting(tmp_path, config, monkeypatch)
    authority = audit.validate_audit_authority(
        Path("configs/post_hoc_final_supplementary.yaml"),
        SESSION0_FREEZE,
        AMENDMENT_FREEZE,
        git_runner=git_runner,
        project_root=tmp_path,
    )
    references = audit.load_privacy_references(
        tmp_path / synthetic_config["privacy"]["privacy_reference_loader"]["source"],
        synthetic_config,
        authorized=True,
    )
    prior_inputs = audit.load_prior_audit_inputs(authority)
    context = audit.build_audit_check_context(authority, prior_inputs, references)
    return synthetic_config, git_runner, authority, references, context


def test_ia01_and_ia03_reject_schema_valid_session5_fold_incoherence(
    tmp_path: Path,
    config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit scientific checks must execute the Session 4/5 semantic validator."""
    synthetic_config, _, _, _, context = _synthetic_audit_context(tmp_path, config, monkeypatch)
    fold_path = synthetic_config["outputs"]["scientific_and_closure"]["date_grouped_fold_audit"]["path"]
    frames = OrderedDict((path, frame.copy(deep=True)) for path, frame in context.frames.items())
    frames[fold_path].loc[0, "n_fit"] = int(frames[fold_path].loc[0, "n_fit"]) + 1
    changed = context._replace(frames=frames)
    with pytest.raises(ValueError, match="fold fit/validation totals"):
        audit._check_schema_dtype_order_validation(changed)
    with pytest.raises(ValueError, match="fold fit/validation totals"):
        audit._check_parent_child_count_reconciliation(changed)


def test_audit_calibration_counts_use_independent_session4_event_anchor(
    tmp_path: Path,
    config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coordinated internally valid calibration count changes fail against Session 4 evidence."""
    synthetic_config, _, _, _, context = _synthetic_audit_context(tmp_path, config, monkeypatch)
    points_path = synthetic_config["outputs"]["scientific_and_closure"]["calibration_points"]["path"]
    frames = OrderedDict((path, frame.copy(deep=True)) for path, frame in context.frames.items())
    points = frames[points_path]
    index = points.index[points["nonevents"].astype(int) > 0][0]
    n = int(points.loc[index, "n"])
    events = int(points.loc[index, "events"]) + 1
    points.loc[index, "events"] = events
    points.loc[index, "nonevents"] = n - events
    points.loc[index, "observed_fraction"] = events / n
    low, high = wilson_proportion_ci(events, n)
    points.loc[index, "observed_ci_low"] = low
    points.loc[index, "observed_ci_high"] = high
    changed = context._replace(frames=frames)
    with pytest.raises(ValueError, match="event anchor"):
        audit._check_schema_dtype_order_validation(changed)
    with pytest.raises(ValueError, match="event anchor"):
        audit._check_parent_child_count_reconciliation(changed)


def test_audit_independently_rejects_rehashed_qc6_statement_mutation(
    tmp_path: Path,
    config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rebinding downstream hashes cannot legitimize a changed frozen QC6 statement."""
    synthetic_config, _, _, _, context = _synthetic_audit_context(tmp_path, config, monkeypatch)
    scientific = synthetic_config["outputs"]["scientific_and_closure"]
    qc6_path = scientific["qc6_lineage_correction"]["path"]
    report_path = scientific["consolidated_analyses_document"]["path"]
    manifest_path = scientific["execution_manifest"]["path"]
    reporting_log_path = synthetic_config["outputs"]["completion_logs"]["reporting"]
    changed_inputs = OrderedDict(context.prior_inputs)
    changed_note = changed_inputs[qc6_path].decode("utf-8").replace("any-channel rule", "single-channel rule")
    changed_inputs[qc6_path] = changed_note.encode("utf-8")
    changed_report = reporting.build_consolidated_report(
        audit._session4_5_frames(context),
        audit._session6_reporting_frames(context),
        context.fact_audit,
        changed_note,
        synthetic_config,
    )
    changed_inputs[report_path] = changed_report.encode("utf-8")
    changed_manifest = copy.deepcopy(context.execution_manifest)
    changed_manifest["output_sha256"][qc6_path] = hashlib.sha256(changed_inputs[qc6_path]).hexdigest()
    changed_manifest["output_sha256"][report_path] = hashlib.sha256(changed_inputs[report_path]).hexdigest()
    changed_inputs[manifest_path] = reporting.serialize_json_artifact(changed_manifest)
    records = [line.split("\t") for line in context.reporting_log.rstrip("\n").split("\n")]
    rebound = {
        qc6_path: hashlib.sha256(changed_inputs[qc6_path]).hexdigest(),
        report_path: hashlib.sha256(changed_inputs[report_path]).hexdigest(),
        manifest_path: hashlib.sha256(changed_inputs[manifest_path]).hexdigest(),
    }
    for record in records:
        if record[0] == "output_hash" and record[1] in rebound:
            record[2] = rebound[record[1]]
    changed_log = "\n".join("\t".join(record) for record in records) + "\n"
    changed_inputs[reporting_log_path] = changed_log.encode("utf-8")
    entry_hashes = OrderedDict(context.entry_hashes)
    for path in (qc6_path, report_path, manifest_path, reporting_log_path):
        entry_hashes[path] = hashlib.sha256(changed_inputs[path]).hexdigest()
    changed = context._replace(
        prior_inputs=changed_inputs,
        entry_hashes=entry_hashes,
        execution_manifest=changed_manifest,
        consolidated_report=changed_report,
        reporting_log=changed_log,
        reporting_report_anchor=rebound[report_path],
    )
    assert audit._check_non_circular_manifest_log_provenance(changed).startswith("executed:")
    assert audit._check_negative_result_retention(changed).startswith("executed:")
    with pytest.raises(ValueError, match="QC6"):
        audit._check_schema_dtype_order_validation(changed)


@pytest.mark.parametrize(
    "unexpected_path",
    [
        "outputs/analysis/posthoc_final_undeclared.csv",
        "outputs/figures/unauthorized.png",
        "outputs/models/unauthorized.joblib",
        "outputs/predictions/participant_values.csv",
        "outputs/analysis/fold_assignments.csv",
    ],
)
def test_ia06_rejects_unexpected_supplementary_and_new_prohibited_artifacts(
    tmp_path: Path,
    config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    unexpected_path: str,
) -> None:
    """IA06 rejects undeclared supplement, figure, model, prediction, and assignment outputs."""
    _, _, _, _, context = _synthetic_audit_context(tmp_path, config, monkeypatch)
    assert audit._check_stale_output_and_namespace(context).startswith("executed:")
    target = tmp_path / unexpected_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"synthetic unauthorized artifact\n")
    with pytest.raises(ValueError, match="namespace"):
        audit._check_stale_output_and_namespace(context)


@pytest.mark.parametrize(
    "unexpected_path",
    [
        "outputs/figures/committed_after_freeze.png",
        "outputs/models/committed_after_freeze.joblib",
        "outputs/predictions/committed_after_freeze.csv",
        "outputs/analysis/committed_fold_assignments.csv",
    ],
)
def test_ia06_rejects_tracked_prohibited_artifacts_created_after_freeze(
    tmp_path: Path,
    config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    unexpected_path: str,
) -> None:
    """Being tracked at Session 7 cannot legitimize a prohibited post-freeze artifact."""
    _, _, _, _, context = _synthetic_audit_context(tmp_path, config, monkeypatch)
    target = tmp_path / unexpected_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"synthetic committed-after-freeze artifact\n")
    base_runner = context.authority["git_runner"]

    def post_freeze_runner(*args: str) -> str:
        if args[:2] == ("ls-files", "--error-unmatch") and args[2] == unexpected_path:
            return unexpected_path
        if args[:2] == ("cat-file", "-e") and args[2] == f"{SESSION0_FREEZE}:{unexpected_path}":
            raise ValueError("Synthetic path did not exist at the Session 0 freeze.")
        return base_runner(*args)

    changed_authority = dict(context.authority)
    changed_authority["git_runner"] = post_freeze_runner
    changed = context._replace(authority=changed_authority)
    with pytest.raises(ValueError, match="namespace"):
        audit._check_stale_output_and_namespace(changed)


def test_audit_real_cli_executes_checks_staging_readback_and_log_last_promotion(
    tmp_path: Path,
    config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real audit parser/main executes IA01--IA12 and promotes exactly JSON then log."""
    from scripts import run_posthoc_final_audit as cli

    synthetic_config, git_runner = _complete_synthetic_reporting(tmp_path, config, monkeypatch)
    report_path = synthetic_config["outputs"]["scientific_and_closure"]["consolidated_analyses_document"]["path"]
    report_hash_before = hashlib.sha256((tmp_path / report_path).read_bytes()).hexdigest()
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file()}
    assert cli.main(
        [
            "--config",
            "configs/post_hoc_final_supplementary.yaml",
            "--freeze-commit",
            SESSION0_FREEZE,
            "--amendment-freeze-commit",
            AMENDMENT_FREEZE,
        ],
        project_root=tmp_path,
        git_runner=git_runner,
        invocation_clock=lambda: "2030-01-02T03:04:05Z",
    ) == 0
    output_paths, log_path = audit.audit_promotion_inventory(synthetic_config)
    after = {path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file()}
    assert after - before == {Path(output_paths[0]), Path(log_path)}
    payload = json.loads((tmp_path / output_paths[0]).read_text(encoding="utf-8"))
    assert len(payload["checks"]) == 12
    assert {item["status"] for item in payload["checks"]} == {"pass"}
    assert all(item["evidence_note"].startswith("executed:") for item in payload["checks"])
    assert hashlib.sha256((tmp_path / report_path).read_bytes()).hexdigest() == report_hash_before
    assert not (PROJECT_ROOT / output_paths[0]).exists()


def test_audit_real_cli_rejects_rehashed_schema_shaped_semantically_invalid_prior(
    tmp_path: Path,
    config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Session 7 parser/main cannot certify a coherently rehashed fold-incoherent prior CSV."""
    from scripts import run_posthoc_final_audit as cli

    synthetic_config, git_runner = _complete_synthetic_reporting(tmp_path, config, monkeypatch)
    scientific = synthetic_config["outputs"]["scientific_and_closure"]
    fold_path = scientific["date_grouped_fold_audit"]["path"]
    fold_file = tmp_path / fold_path
    folds = pd.read_csv(fold_file)
    folds.loc[0, "n_fit"] = int(folds.loc[0, "n_fit"]) + 1
    folds.to_csv(fold_file, index=False, lineterminator="\n")
    rebind_synthetic_completion_output_hash(tmp_path, synthetic_config, "date_grouped", fold_path)
    date_log_path = synthetic_config["outputs"]["completion_logs"]["date_grouped"]
    manifest_path = scientific["execution_manifest"]["path"]
    manifest_file = tmp_path / manifest_path
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"), object_pairs_hook=OrderedDict)
    manifest["input_sha256"][fold_path] = hashlib.sha256(fold_file.read_bytes()).hexdigest()
    manifest["input_sha256"][date_log_path] = hashlib.sha256((tmp_path / date_log_path).read_bytes()).hexdigest()
    manifest_file.write_bytes(reporting.serialize_json_artifact(manifest))
    rebind_synthetic_completion_output_hash(tmp_path, synthetic_config, "reporting", manifest_path)
    with pytest.raises(ValueError, match="failed IA check"):
        cli.main(
            [
                "--config",
                "configs/post_hoc_final_supplementary.yaml",
                "--freeze-commit",
                SESSION0_FREEZE,
                "--amendment-freeze-commit",
                AMENDMENT_FREEZE,
            ],
            project_root=tmp_path,
            git_runner=git_runner,
            invocation_clock=lambda: "2030-01-02T03:04:05Z",
        )
    output_paths, log_path = audit.audit_promotion_inventory(synthetic_config)
    assert not any((tmp_path / path).exists() for path in (*output_paths, log_path))


@pytest.mark.parametrize("failure", ["authority", "stale"])
def test_audit_real_cli_preconditions_fail_before_privacy_loader(
    failure: str,
    tmp_path: Path,
    config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real current-runtime and stale-target gates precede audit privacy-reference access."""
    from scripts import run_posthoc_final_audit as cli

    synthetic_config, git_runner = _complete_synthetic_reporting(tmp_path, config, monkeypatch)
    touched = {"privacy_loader": False}
    real_loader = audit.load_privacy_references

    def recording_loader(*args: Any, **kwargs: Any) -> Any:
        touched["privacy_loader"] = True
        return real_loader(*args, **kwargs)

    monkeypatch.setattr(audit, "load_privacy_references", recording_loader)
    if failure == "authority":
        config_path = tmp_path / "configs/post_hoc_final_supplementary.yaml"
        config_path.write_bytes(config_path.read_bytes() + b"\n")
    else:
        output_paths, _ = audit.audit_promotion_inventory(synthetic_config)
        target = tmp_path / output_paths[0]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("stale", encoding="utf-8")
    with pytest.raises(ValueError):
        cli.main(
            [
                "--config",
                "configs/post_hoc_final_supplementary.yaml",
                "--freeze-commit",
                SESSION0_FREEZE,
                "--amendment-freeze-commit",
                AMENDMENT_FREEZE,
            ],
            project_root=tmp_path,
            git_runner=git_runner,
            invocation_clock=lambda: "2030-01-02T03:04:05Z",
        )
    assert touched == {"privacy_loader": False}


def test_actual_ia_failure_blocks_audit_staging_and_false_completion(
    tmp_path: Path,
    config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A genuinely executed IA12 failure leaves neither an audit candidate nor receipt."""
    synthetic_config, git_runner = _complete_synthetic_reporting(tmp_path, config, monkeypatch)
    authority = audit.validate_audit_authority(
        Path("configs/post_hoc_final_supplementary.yaml"),
        SESSION0_FREEZE,
        AMENDMENT_FREEZE,
        git_runner=git_runner,
        project_root=tmp_path,
    )
    privacy_source = tmp_path / synthetic_config["privacy"]["privacy_reference_loader"]["source"]
    references = audit.load_privacy_references(privacy_source, synthetic_config, authorized=True)
    prior_inputs = audit.load_prior_audit_inputs(authority)
    authority["config"]["visualization_v2_authorized"] = True
    with pytest.raises(ValueError, match="failed IA check"):
        audit.prepare_audit_promotion(
            authority=authority,
            prior_inputs=prior_inputs,
            privacy_references=references,
            generated_at_utc="2030-01-02T03:04:05Z",
            git_runner=git_runner,
        )
    output_paths, log_path = audit.audit_promotion_inventory(synthetic_config)
    assert not any((tmp_path / path).exists() for path in (*output_paths, log_path))


def test_audit_detects_entry_exit_mutation_and_report_handoff_mismatch(
    tmp_path: Path,
    config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protected-source mutation and a wrong Session 6 report anchor each fail closed."""
    synthetic_config, git_runner = _complete_synthetic_reporting(tmp_path, config, monkeypatch)
    authority = audit.validate_audit_authority(
        Path("configs/post_hoc_final_supplementary.yaml"),
        SESSION0_FREEZE,
        AMENDMENT_FREEZE,
        git_runner=git_runner,
        project_root=tmp_path,
    )
    references = audit.load_privacy_references(
        tmp_path / synthetic_config["privacy"]["privacy_reference_loader"]["source"],
        synthetic_config,
        authorized=True,
    )
    prior_inputs = audit.load_prior_audit_inputs(authority)
    context = audit.build_audit_check_context(authority, prior_inputs, references)
    wrong_anchor = context._replace(reporting_report_anchor="0" * 64)
    with pytest.raises(ValueError, match="handoff anchor"):
        audit._check_unchanged_consolidated_report_hash(wrong_anchor)
    retained_reasons = sorted(
        {
            str(value)
            for frame in context.frames.values()
            for column in ("nonestimable_reason", "ci_nonestimable_reason")
            if column in frame.columns
            for value in frame[column].dropna().tolist()
            if str(value)
        }
    )
    assert retained_reasons
    incomplete_report = context._replace(
        consolidated_report=context.consolidated_report.replace(retained_reasons[0], "omitted_reason")
    )
    with pytest.raises(ValueError, match="deterministic complete aggregate rendering"):
        audit._check_negative_result_retention(incomplete_report)
    protected = next(iter(synthetic_config["frozen_inputs"]["whole_container_sha256"]))
    (tmp_path / protected).write_bytes(b"synthetic-exit-mutation")
    with pytest.raises(ValueError, match="failed IA check"):
        audit.prepare_audit_promotion(
            authority=authority,
            prior_inputs=prior_inputs,
            privacy_references=references,
            generated_at_utc="2030-01-02T03:04:05Z",
            git_runner=git_runner,
        )
    output_paths, log_path = audit.audit_promotion_inventory(synthetic_config)
    assert not any((tmp_path / path).exists() for path in (*output_paths, log_path))


def test_isolated_audit_candidate_byte_mismatch_fails() -> None:
    """A deterministic dual-audit mismatch cannot reach staging."""
    with pytest.raises(ValueError, match="candidate bytes disagree"):
        audit.require_byte_identical_audit_candidates(b'{"candidate":1}\n', b'{"candidate":2}\n')
