"""Privacy, provenance, promotion, and future Session 7 audit infrastructure."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from src.analysis.posthoc_final_common import (
    ACCESS_STATE_KEYS,
    is_utc_iso8601_z,
    sha256_file,
    validate_completion_log,
    validate_freeze_commit,
    validate_preflight_privacy,
    validate_preflight_schema,
)
from src.analysis.posthoc_final_locked_scores import (
    CURRENT_MANIFEST_RELATIVE_PATH,
    SCORE_BLIND_PREFLIGHT_RELATIVE_PATH,
    _validate_annotated_tag_chain,
    load_effective_posthoc_config,
    validate_amendment_freeze_commit,
    validate_current_runtime_authority,
    validate_execute_git_prerequisites,
)
from src.config import PROJECT_ROOT


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_CLINICAL_DATE_PATTERN = re.compile(
    r"(?<![0-9])(?:[0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{4}/[0-9]{1,2}/[0-9]{1,2})(?![0-9])"
)
_RAW_FILENAME_PATTERN = re.compile(r"(?<![/\\\w])[^/\\\s]+(?:\.csv|\.npz|\.pkl)\b", re.IGNORECASE)
_WINDOWS_ABSOLUTE_PATTERN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/](?:[^\t\r\n ]+)")
_POSIX_ABSOLUTE_PATTERN = re.compile(r"(?<![A-Za-z0-9_.-])/(?:[^/\s]+/)+[^/\s]+")
_PARTICIPANT_COLUMNS = frozenset(
    {
        "sample_id",
        "participant_id",
        "p_raw",
        "p_cal",
        "filename",
        "collection_time",
        "collection_location",
        "participant_row",
        "participant_level_scores",
    }
)
_POLICY_KEYS = frozenset(
    {
        "policy",
        "prohibited_category",
        "field_or_context",
        "prohibited_fields",
        "prohibited_stronger_wording",
        "permitted_manuscript_implication",
    }
)
_TIMESTAMP_KEYS = frozenset(
    {
        "generated_at_utc",
        "timestamp_utc",
        "frozen_at_utc",
        "correction_date_utc",
        "approval_date_utc",
        "authorization_date_utc",
        "completion_utc",
        "shared_invocation_utc",
    }
)


class PromotionCandidate(NamedTuple):
    """One validated destination-local candidate awaiting atomic promotion."""

    relative_path: str
    candidate_path: Path
    target_path: Path
    validator: Callable[[Path], None]


class PrivacyValidationError(ValueError):
    """Sanitized artifact-privacy failure with no matched private value."""

    def __init__(self, *, artifact_path: str, category: str, context: str, match_count: int = 1) -> None:
        self.artifact_path = artifact_path
        self.category = category
        self.context = context
        self.match_count = match_count
        super().__init__(
            "Artifact privacy validation failed: "
            f"artifact={artifact_path}; category={category}; context={context}; match_count={match_count}."
        )


class PrivacyReferences:
    """Opaque in-memory deny sets for explicitly authorized privacy scans."""

    __slots__ = ("_sample_strings", "_sample_numbers", "_filenames", "_collection_times")

    def __init__(
        self,
        *,
        sample_strings: frozenset[str],
        sample_numbers: frozenset[int],
        filenames: frozenset[str],
        collection_times: frozenset[str],
    ) -> None:
        self._sample_strings = sample_strings
        self._sample_numbers = sample_numbers
        self._filenames = filenames
        self._collection_times = collection_times

    @classmethod
    def from_columns(
        cls,
        *,
        sample_ids: Sequence[object],
        filenames: Sequence[object],
        collection_times: Sequence[object],
    ) -> PrivacyReferences:
        """Build an opaque deny set without hashing or exposing individual values."""
        sample_strings: set[str] = set()
        sample_numbers: set[int] = set()
        for value in sample_ids:
            if value is None or pd.isna(value):
                continue
            if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
                sample_numbers.add(int(value))
            else:
                text = str(value)
                if re.fullmatch(r"[+-]?[0-9]+", text):
                    sample_numbers.add(int(text))
                else:
                    sample_strings.add(text)
        filename_values = frozenset(str(value) for value in filenames if value is not None and not pd.isna(value))
        date_values = frozenset(
            str(value) for value in collection_times if value is not None and not pd.isna(value)
        )
        return cls(
            sample_strings=frozenset(sample_strings),
            sample_numbers=frozenset(sample_numbers),
            filenames=filename_values,
            collection_times=date_values,
        )

    def category_for(self, value: object, *, field_name: str | None, embedded_text: bool) -> str | None:
        """Return a prohibited category using typed, context-aware matching."""
        field = field_name.lower() if isinstance(field_name, str) else ""
        participant_field = field in _PARTICIPANT_COLUMNS
        if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
            if participant_field and int(value) in self._sample_numbers:
                return "participant_identifier"
            return None
        if not isinstance(value, str):
            return None
        if value in self._filenames or (embedded_text and any(item in value for item in self._filenames)):
            return "filename"
        if value in self._collection_times or (
            embedded_text and any(item in value for item in self._collection_times)
        ):
            return "clinical_date"
        if value in self._sample_strings:
            return "participant_identifier"
        if embedded_text:
            for item in self._sample_strings:
                if re.search(rf"(?<![A-Za-z0-9]){re.escape(item)}(?![A-Za-z0-9])", value):
                    return "participant_identifier"
            numeric_identifiers = re.finditer(
                r"(?i)\b(?:sample|participant|subject)[ _-]?id\b\s*[:=#]?\s*([+-]?[0-9]+)\b",
                value,
            )
            if any(int(match.group(1)) in self._sample_numbers for match in numeric_identifiers):
                return "participant_identifier"
        if participant_field and re.fullmatch(r"[+-]?[0-9]+", value) and int(value) in self._sample_numbers:
            return "participant_identifier"
        return None


def _ordered_boolean_scope(values: Mapping[str, bool] | None = None) -> OrderedDict[str, bool]:
    """Create one exact nine-key boolean scope."""
    source = values or {}
    if source and tuple(source) != tuple(ACCESS_STATE_KEYS):
        raise ValueError("Access-state keys or order are invalid.")
    result = OrderedDict((key, bool(source.get(key, False))) for key in ACCESS_STATE_KEYS)
    if source and any(type(source[key]) is not bool for key in ACCESS_STATE_KEYS):
        raise ValueError("Access-state values must be genuine booleans.")
    return result


def build_access_state(
    session_activity: Mapping[str, bool],
    cumulative_prior: Mapping[str, bool],
) -> OrderedDict[str, OrderedDict[str, bool]]:
    """Build the exact cumulative union of prior and current access activity."""
    activity = _ordered_boolean_scope(session_activity)
    prior = _ordered_boolean_scope(cumulative_prior)
    cumulative = OrderedDict((key, prior[key] or activity[key]) for key in ACCESS_STATE_KEYS)
    state = OrderedDict(
        [
            ("session_activity", activity),
            ("cumulative_through_session", cumulative),
        ]
    )
    validate_access_state(state, expected_activity=activity, cumulative_prior=prior)
    return state


def build_session3_access_state(
    cumulative_prior: Mapping[str, bool] | None = None,
) -> OrderedDict[str, OrderedDict[str, bool]]:
    """Return Session 3's all-false activity while preserving any prior cumulative access."""
    empty = OrderedDict((key, False) for key in ACCESS_STATE_KEYS)
    return build_access_state(empty, cumulative_prior or empty)


def build_reporting_access_state(
    cumulative_prior: Mapping[str, bool],
) -> OrderedDict[str, OrderedDict[str, bool]]:
    """Build the truthful future Session 6 participant/score/privacy ledger.

    Args:
        cumulative_prior: Validated cumulative state through Session 5.

    Returns:
        Exact activity and cumulative ledgers for reporting closure.
    """
    true_keys = {
        "prediction_container_bytes_hashed",
        "prediction_schema_or_header_inspected",
        "participant_level_label_values_read",
        "participant_level_score_values_read",
        "participant_level_date_values_read",
        "participant_level_qc_values_read",
        "new_score_derived_results_computed",
        "privacy_reference_loader_executed",
    }
    activity = OrderedDict((key, key in true_keys) for key in ACCESS_STATE_KEYS)
    return build_access_state(activity, cumulative_prior)


def build_audit_access_state(
    cumulative_prior: Mapping[str, bool],
) -> OrderedDict[str, OrderedDict[str, bool]]:
    """Build the truthful future Session 7 aggregate-audit/privacy ledger.

    Args:
        cumulative_prior: Validated cumulative state through Session 6.

    Returns:
        Exact activity and cumulative ledgers for the additive audit.
    """
    activity = OrderedDict(
        (
            key,
            key in {"participant_level_date_values_read", "privacy_reference_loader_executed"},
        )
        for key in ACCESS_STATE_KEYS
    )
    return build_access_state(activity, cumulative_prior)


def validate_access_state(
    access_state: Mapping[str, Any],
    *,
    expected_activity: Mapping[str, bool] | None = None,
    cumulative_prior: Mapping[str, bool] | None = None,
) -> None:
    """Validate exact access ordering, union semantics, and impossible dependencies."""
    if tuple(access_state) != ("session_activity", "cumulative_through_session"):
        raise ValueError("Access-state object keys or order are invalid.")
    activity = _ordered_boolean_scope(access_state["session_activity"])
    cumulative = _ordered_boolean_scope(access_state["cumulative_through_session"])
    if expected_activity is not None and activity != _ordered_boolean_scope(expected_activity):
        raise ValueError("Session activity does not match the expected state.")
    if cumulative_prior is None:
        if any(activity[key] and not cumulative[key] for key in ACCESS_STATE_KEYS):
            raise ValueError("Cumulative access state must retain every current-session access.")
    else:
        prior = _ordered_boolean_scope(cumulative_prior)
        expected_cumulative = OrderedDict((key, prior[key] or activity[key]) for key in ACCESS_STATE_KEYS)
        if cumulative != expected_cumulative:
            raise ValueError("Cumulative access state is not the exact prior/activity union.")
    for scope in (activity, cumulative):
        if scope["new_score_derived_results_computed"] and not scope["participant_level_score_values_read"]:
            raise ValueError("A score-derived result requires participant-level score access.")
        if scope["auxiliary_model_fit_executed"] and not (
            scope["participant_level_label_values_read"]
            and scope["participant_level_date_values_read"]
            and scope["new_score_derived_results_computed"]
        ):
            raise ValueError("An auxiliary model fit has an impossible access-state dependency.")
        if scope["privacy_reference_loader_executed"] and not scope["participant_level_date_values_read"]:
            raise ValueError("A privacy-reference load must record its date-value access.")


def load_privacy_references(
    sample_index_path: Path,
    config: Mapping[str, Any],
    *,
    authorized: bool,
    reader: Callable[..., pd.DataFrame] = pd.read_csv,
) -> PrivacyReferences:
    """Load the isolated exact-value deny sets only under explicit authorization."""
    if authorized is not True:
        raise PermissionError("Privacy-reference access requires explicit later-session authorization.")
    contract = config["privacy"]["privacy_reference_loader"]
    expected = list(contract["usecols"])
    if expected != list(config["loaders"]["per_purpose_ordered_usecols"]["privacy_reference_sample_index"]):
        raise ValueError("Privacy-reference loader contracts disagree.")
    try:
        frame = reader(sample_index_path, usecols=expected)
    except (OSError, TypeError, ValueError, UnicodeError, pd.errors.ParserError):
        raise ValueError("Privacy-reference source could not be loaded under its exact contract.") from None
    if frame.columns.tolist() != expected:
        raise ValueError("Privacy-reference loader returned a broader or reordered schema.")
    return PrivacyReferences.from_columns(
        sample_ids=frame["sample_id"].tolist(),
        filenames=frame["filename"].tolist(),
        collection_times=frame["collection_time"].tolist(),
    )


def _privacy_failure(category: str, *, artifact_path: str, context: str) -> PrivacyValidationError:
    """Create one sanitized failure containing no detected value."""
    return PrivacyValidationError(artifact_path=artifact_path, category=category, context=context)


def _is_absolute_path(value: str) -> bool:
    """Return whether text is an absolute POSIX or Windows path."""
    return _POSIX_ABSOLUTE_PATTERN.search(value) is not None or _WINDOWS_ABSOLUTE_PATTERN.search(value) is not None


def validate_artifact_privacy(
    payload: Any,
    *,
    artifact_kind: str,
    config: Mapping[str, Any],
    privacy_references: PrivacyReferences | None = None,
    artifact_path: str = "candidate_artifact",
) -> None:
    """Apply schema-aware privacy checks without echoing any detected value."""
    allowed_kinds = frozenset(config["privacy"]["artifact_specific_allowlists"])
    kind_alias = {
        "governance_manifest": "governance_manifests",
        "structured_json": "structured_json",
        "scientific_csv": "scientific_csv",
        "markdown_report": "markdown_reports",
        "sanitized_log": "sanitized_logs",
        "methods_configuration": "methods_configuration",
    }
    normalized_kind = kind_alias.get(artifact_kind, artifact_kind)
    if normalized_kind not in allowed_kinds:
        raise ValueError("Unknown artifact-specific privacy policy.")

    path = Path(artifact_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Privacy validation requires a sanitized project-relative artifact path.")

    def fail(category: str, context: str) -> None:
        raise _privacy_failure(category, artifact_path=artifact_path, context=context)

    def inspect(value: Any, key: str | None = None, *, in_policy: bool = False) -> None:
        key_lower = key.lower() if isinstance(key, str) else ""
        policy_context = in_policy or key_lower in _POLICY_KEYS
        if privacy_references is not None:
            category = privacy_references.category_for(value, field_name=key, embedded_text=isinstance(value, str))
            if category is not None:
                fail(category, "field_or_text")
        if key_lower in _PARTICIPANT_COLUMNS and normalized_kind == "scientific_csv" and not policy_context:
            fail("participant_level_field_or_row", "column")
        if isinstance(value, pd.DataFrame):
            if normalized_kind == "scientific_csv" and set(map(str, value.columns)).intersection(
                _PARTICIPANT_COLUMNS
            ):
                fail("participant_level_field_or_row", "dataframe_columns")
            for column in value.columns:
                for cell in value[column].tolist():
                    inspect(cell, str(column), in_policy=policy_context)
        elif isinstance(value, Mapping):
            if normalized_kind == "scientific_csv" and not policy_context and set(map(str, value)).intersection(
                _PARTICIPANT_COLUMNS
            ):
                fail("participant_level_field_or_row", "mapping_keys")
            for nested_key, nested_value in value.items():
                inspect(nested_value, str(nested_key), in_policy=policy_context)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                inspect(nested, key, in_policy=policy_context)
        elif isinstance(value, bytes):
            try:
                decoded = value.decode("utf-8")
            except UnicodeDecodeError:
                fail("non_utf8_serialization", "bytes")
            if normalized_kind == "scientific_csv":
                try:
                    frame = pd.read_csv(io.StringIO(decoded), dtype="string", keep_default_na=False)
                except (pd.errors.ParserError, UnicodeError) as exc:
                    raise ValueError("Scientific CSV privacy input is not parseable UTF-8 CSV.") from exc
                inspect(frame, key, in_policy=policy_context)
            elif normalized_kind in {"structured_json", "governance_manifests"}:
                try:
                    parsed = json.loads(decoded, object_pairs_hook=OrderedDict)
                except json.JSONDecodeError as exc:
                    raise ValueError("Structured JSON privacy input is not valid UTF-8 JSON.") from exc
                inspect(parsed, key, in_policy=policy_context)
            else:
                inspect(decoded, key, in_policy=policy_context)
        elif isinstance(value, str):
            if _is_absolute_path(value):
                fail("absolute_path", "text")
            if key_lower in _TIMESTAMP_KEYS:
                if not is_utc_iso8601_z(value):
                    fail("invalid_provenance_timestamp", "timestamp_field")
                return
            date_scan_value = value
            if normalized_kind == "markdown_reports":
                timestamp_names = "|".join(re.escape(name) for name in sorted(_TIMESTAMP_KEYS))
                date_scan_value = re.sub(
                    rf"(?:{timestamp_names})\s*:\s*`?[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}"
                    rf"T[0-9]{{2}}:[0-9]{{2}}:[0-9]{{2}}Z`?",
                    "allowlisted_provenance_timestamp",
                    date_scan_value,
                )
            if _CLINICAL_DATE_PATTERN.search(date_scan_value):
                fail("clinical_date", "text")
            if _RAW_FILENAME_PATTERN.search(value):
                fail("filename", "text")
            lowered = value.lower()
            unsupported = (
                "generally under three months",
                "confirmed central pathology review",
                "external validation demonstrated",
                "validated pre-endoscopy triage",
            )
            def unsupported_affirmed(phrase: str) -> bool:
                """Reject affirmative overclaims while permitting explicit conservative nonclaims."""
                if phrase not in lowered:
                    return False
                quoted_phrase = re.escape(phrase)
                return re.search(rf"\b(?:no|not|does not)\s+['\"]?{quoted_phrase}", lowered) is None

            if not policy_context and any(unsupported_affirmed(phrase) for phrase in unsupported):
                fail("unsupported_claim", "text")

    if normalized_kind == "sanitized_logs" and isinstance(payload, (str, bytes)):
        try:
            log_text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        except UnicodeDecodeError:
            fail("non_utf8_serialization", "log_bytes")
        for line in log_text.rstrip("\n").split("\n"):
            fields = line.split("\t")
            if not fields:
                continue
            timestamp_positions = (
                {3} if fields[0] == "header" else ({1} if fields[0] == "completion_receipt" else set())
            )
            for index, field in enumerate(fields):
                if index in timestamp_positions:
                    if not is_utc_iso8601_z(field):
                        fail("invalid_provenance_timestamp", "log_timestamp")
                else:
                    inspect(field, in_policy=False)
        return
    inspect(payload)


def validate_hash_subject(subject_kind: str, subject: str, config: Mapping[str, Any]) -> None:
    """Distinguish approved whole-container provenance from prohibited individual hashes."""
    if subject_kind != "whole_container":
        raise ValueError("Hashing an individual participant value, row, date, filename, or element is prohibited.")
    declared = set(config["frozen_inputs"]["whole_container_sha256"])
    declared.update(config["hash_map_inventories"]["execution_manifest_output_sha256_keys_exact_order"])
    declared.update(config["hash_map_inventories"]["execution_manifest_code_sha256_keys_exact_order"])
    declared.update(config["hash_map_inventories"]["integrity_audit_prior_artifact_sha256_keys_exact_order"])
    if subject not in declared:
        raise ValueError("Whole-container hash subject is not declared by the frozen contract.")


def validate_hash_map(
    values: Mapping[str, str],
    *,
    expected_keys: Sequence[str],
    forbidden_self_paths: Sequence[str],
) -> None:
    """Validate exact ordered SHA-256 maps and reject self/circular references."""
    if not isinstance(values, Mapping) or tuple(values) != tuple(expected_keys):
        raise ValueError("Hash-map keys or order do not match the frozen inventory.")
    if set(values).intersection(forbidden_self_paths):
        raise ValueError("Hash-map self-reference or circular reference is forbidden.")
    if any(not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None for value in values.values()):
        raise ValueError("Hash-map values must be lowercase SHA-256 strings.")


def capture_integrity_snapshot(
    project_root: Path,
    relative_paths: Sequence[str],
    *,
    declared_hashes: Mapping[str, str] | None = None,
) -> OrderedDict[str, str]:
    """Hash an exact ordered protected-container inventory without reading values."""
    if not relative_paths or len(relative_paths) != len(set(relative_paths)):
        raise ValueError("Protected integrity path inventory is empty or duplicated.")
    root = project_root.resolve()
    snapshot: OrderedDict[str, str] = OrderedDict()
    for relative_text in relative_paths:
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Protected integrity paths must be project-relative.")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            raise ValueError("Protected integrity path escapes the project root.") from None
        if not path.is_file():
            raise ValueError("A protected integrity container is missing.")
        snapshot[relative_text] = hashlib.sha256(path.read_bytes()).hexdigest()
    if declared_hashes is not None:
        for path, digest in declared_hashes.items():
            if path not in snapshot or snapshot[path] != digest:
                raise ValueError("A protected frozen container differs from its declared hash.")
    return snapshot


def verify_integrity_snapshot(project_root: Path, entry_snapshot: Mapping[str, str]) -> None:
    """Rehash the exact entry inventory and fail if any protected byte changed."""
    exit_snapshot = capture_integrity_snapshot(project_root, tuple(entry_snapshot))
    if exit_snapshot != OrderedDict(entry_snapshot):
        raise ValueError("A protected input changed between entry and exit integrity checks.")


def _validate_environment(environment: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    """Validate the exact frozen environment and runtime-control object."""
    contract = schema["leaf_contracts"]["environment"]
    if tuple(environment) != tuple(contract["key_order"]):
        raise ValueError("Execution-manifest environment key order is invalid.")
    if environment["conda_environment_name"] != "gcvocs" or environment["python_version"] != "3.11.15":
        raise ValueError("Execution-manifest environment identity is invalid.")
    for name in ("package_versions", "runtime_controls"):
        nested_contract = contract["leaves"][name]
        nested = environment[name]
        if tuple(nested) != tuple(nested_contract["key_order"]):
            raise ValueError("Execution-manifest nested environment order is invalid.")
        for key, leaf in nested_contract["leaves"].items():
            expected = leaf.get("allowed_value", leaf.get("fixed_value"))
            if nested[key] != expected or type(nested[key]) is not type(expected):
                raise ValueError("Execution-manifest environment value is invalid.")


def validate_execution_manifest(payload: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    """Validate the complete non-circular Session 6 execution-manifest schema."""
    schema = config["outputs"]["scientific_and_closure"]["execution_manifest"]["schema"]
    if tuple(payload) != tuple(schema["top_level_key_order"]):
        raise ValueError("Execution-manifest top-level keys or order are invalid.")
    if payload["schema_version"] != schema["schema_version_fixed"] or payload["package_id"] != schema[
        "package_id_fixed"
    ]:
        raise ValueError("Execution-manifest identity is invalid.")
    if (
        not is_utc_iso8601_z(payload["generated_at_utc"])
        or not isinstance(payload["freeze_commit"], str)
        or _COMMIT_PATTERN.fullmatch(payload["freeze_commit"]) is None
    ):
        raise ValueError("Execution-manifest provenance fields are invalid.")
    commands = payload["commands"]
    if (
        not isinstance(commands, list)
        or not commands
        or any(not isinstance(item, str) or not item for item in commands)
    ):
        raise ValueError("Execution-manifest command inventory is invalid.")
    _validate_environment(payload["environment"], schema)
    inventories = config["hash_map_inventories"]
    validate_hash_map(
        payload["input_sha256"],
        expected_keys=inventories["execution_manifest_input_sha256_keys_exact_order"],
        forbidden_self_paths=(),
    )
    validate_hash_map(
        payload["output_sha256"],
        expected_keys=inventories["execution_manifest_output_sha256_keys_exact_order"],
        forbidden_self_paths=schema["exclude_from_own_hash_map"],
    )
    validate_hash_map(
        payload["code_sha256"],
        expected_keys=inventories["execution_manifest_code_sha256_keys_exact_order"],
        forbidden_self_paths=(),
    )
    for key in ("config_sha256", "methods_sha256"):
        if not isinstance(payload[key], str) or _SHA256_PATTERN.fullmatch(payload[key]) is None:
            raise ValueError("Execution-manifest authority hash is invalid.")
    deterministic = payload["deterministic_comparison_provenance"]
    if tuple(deterministic) != ("candidate_pair_count", "scientific_payload_agreement", "shared_invocation_utc"):
        raise ValueError("Execution-manifest deterministic-comparison schema is invalid.")
    if deterministic["candidate_pair_count"] != 2 or deterministic["scientific_payload_agreement"] is not True:
        raise ValueError("Execution-manifest deterministic candidate agreement is invalid.")
    if not is_utc_iso8601_z(deterministic["shared_invocation_utc"]):
        raise ValueError("Execution-manifest shared invocation timestamp is invalid.")
    validate_access_state(payload["access_state"])
    deviations = payload["deviations_or_nonestimability"]
    if not isinstance(deviations, list):
        raise ValueError("Execution-manifest deviation inventory is invalid.")
    allowed_status = {"none", "deviation_recorded", "non_estimable_retained"}
    for item in deviations:
        if not isinstance(item, Mapping) or tuple(item) != ("artifact_path", "status", "reason_code", "note"):
            raise ValueError("Execution-manifest deviation item schema is invalid.")
        if not isinstance(item["artifact_path"], str) or _is_absolute_path(item["artifact_path"]):
            raise ValueError("Execution-manifest deviation path must be project-relative.")
        if item["status"] not in allowed_status:
            raise ValueError("Execution-manifest deviation status is invalid.")
        if item["reason_code"] is not None and not isinstance(item["reason_code"], str):
            raise ValueError("Execution-manifest deviation reason is invalid.")
        if item["note"] is not None and not isinstance(item["note"], str):
            raise ValueError("Execution-manifest deviation note is invalid.")
    if payload["self_hash_policy"] != "forbidden":
        raise ValueError("Execution-manifest self-hash policy is invalid.")
    validate_artifact_privacy(payload, artifact_kind="governance_manifest", config=config)


def validate_integrity_audit(payload: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    """Validate the complete additive, non-circular Session 7 integrity audit."""
    schema = config["outputs"]["scientific_and_closure"]["integrity_audit"]["schema"]
    if tuple(payload) != tuple(schema["top_level_key_order"]):
        raise ValueError("Integrity-audit top-level keys or order are invalid.")
    if payload["schema_version"] != schema["schema_version_fixed"] or payload["package_id"] != schema[
        "package_id_fixed"
    ]:
        raise ValueError("Integrity-audit identity is invalid.")
    if (
        not is_utc_iso8601_z(payload["generated_at_utc"])
        or not isinstance(payload["freeze_commit"], str)
        or _COMMIT_PATTERN.fullmatch(payload["freeze_commit"]) is None
    ):
        raise ValueError("Integrity-audit provenance fields are invalid.")
    check_contract = schema["leaf_contracts"]["checks"]
    checks = payload["checks"]
    registry = check_contract["ordered_check_registry"]
    if not isinstance(checks, list) or len(checks) != len(registry):
        raise ValueError("Integrity-audit check inventory is invalid.")
    for item, expected in zip(checks, registry, strict=True):
        if not isinstance(item, Mapping) or tuple(item) != tuple(check_contract["item_key_order"]):
            raise ValueError("Integrity-audit check schema is invalid.")
        if (
            not isinstance(item["check_id"], str)
            or type(item["display_order"]) is not int
            or not isinstance(item["check_name"], str)
            or not isinstance(item["status"], str)
        ):
            raise ValueError("Integrity-audit check field types are invalid.")
        identity = (item["check_id"], item["display_order"], item["check_name"])
        if identity != (expected["check_id"], expected["display_order"], expected["check_name"]):
            raise ValueError("Integrity-audit check identity or order is invalid.")
        if item["status"] not in {"pass", "fail"}:
            raise ValueError("Integrity-audit check status is invalid.")
        if item["status"] == "pass" and item["reason_code"] not in {None, "ok"}:
            raise ValueError("A passing integrity check has an invalid reason.")
        if item["status"] == "fail" and (
            not isinstance(item["reason_code"], str) or not item["reason_code"]
        ):
            raise ValueError("A failed integrity check requires a nonempty reason.")
        if item["evidence_note"] is not None and not isinstance(item["evidence_note"], str):
            raise ValueError("Integrity-audit evidence note is invalid.")
    validate_hash_map(
        payload["prior_artifact_sha256"],
        expected_keys=config["hash_map_inventories"]["integrity_audit_prior_artifact_sha256_keys_exact_order"],
        forbidden_self_paths=schema["exclude_from_own_hash_map"],
    )
    if _SHA256_PATTERN.fullmatch(str(payload["consolidated_report_unchanged_sha256"])) is None:
        raise ValueError("Integrity-audit consolidated-report hash is invalid.")
    if payload["visualization_v2_authorized"] is not False:
        raise ValueError("Integrity-audit visualization authorization must remain false.")
    validate_access_state(payload["access_state"])
    if payload["self_hash_policy"] != "forbidden":
        raise ValueError("Integrity-audit self-hash policy is invalid.")
    validate_artifact_privacy(payload, artifact_kind="governance_manifest", config=config)


def execute_integrity_check_registry(
    config: Mapping[str, Any],
    check_functions: Mapping[str, Callable[[], str]],
) -> list[OrderedDict[str, Any]]:
    """Execute IA01--IA12 in frozen order and retain sanitized deterministic evidence."""
    registry = config["outputs"]["scientific_and_closure"]["integrity_audit"]["schema"][
        "leaf_contracts"
    ]["checks"]["ordered_check_registry"]
    names = tuple(item["check_name"] for item in registry)
    if tuple(check_functions) != names or any(not callable(function) for function in check_functions.values()):
        raise ValueError("Integrity-check function inventory is incomplete or reordered.")
    checks: list[OrderedDict[str, Any]] = []
    for item in registry:
        name = item["check_name"]
        try:
            evidence = check_functions[name]()
            if not isinstance(evidence, str) or not evidence or "\n" in evidence or "\t" in evidence:
                raise ValueError("Integrity-check evidence is invalid.")
            status = "pass"
            reason_code: str | None = None
            evidence_note = evidence
        except Exception:
            status = "fail"
            reason_code = "integrity_check_failed"
            evidence_note = f"executed:{name}; result=failed; sensitive_details_withheld"
        checks.append(
            OrderedDict(
                [
                    ("check_id", item["check_id"]),
                    ("display_order", item["display_order"]),
                    ("check_name", name),
                    ("status", status),
                    ("reason_code", reason_code),
                    ("evidence_note", evidence_note),
                ]
            )
        )
    return checks


def require_all_integrity_checks_pass(checks: Sequence[Mapping[str, Any]]) -> None:
    """Block audit completion whenever any executed IA check failed."""
    failed_ids = [str(item.get("check_id", "unknown")) for item in checks if item.get("status") != "pass"]
    if len(checks) != 12 or failed_ids:
        identifiers = ",".join(failed_ids) if failed_ids else "inventory"
        raise ValueError(
            f"Integrity audit contains a failed IA check ({identifiers}); completion and promotion are forbidden."
        )


def validate_session_completion_log(
    text: str,
    *,
    session_name: str,
    freeze_commit: str,
    expected_activity: Mapping[str, bool],
    expected_cumulative: Mapping[str, bool],
    cumulative_prior: Mapping[str, bool],
    config: Mapping[str, Any],
) -> list[list[str]]:
    """Validate the exact reporting or audit receipt inventory and non-circular hash policy."""
    if session_name not in {"reporting", "audit"}:
        raise ValueError("Session 3 completion-log infrastructure is limited to reporting and audit.")
    grammar = config["outputs"]["completion_logs"]["machine_validated_log_grammar"]
    binding = dict(grammar["session_bindings_exact"][session_name])
    binding["session_name"] = session_name
    return validate_completion_log(
        text,
        grammar=grammar,
        session_binding=binding,
        expected_freeze_commit=freeze_commit,
        expected_activity=expected_activity,
        expected_cumulative=expected_cumulative,
        cumulative_prior=cumulative_prior,
        allowed_reason_codes=config["status_and_reason_registries"]["scientific_result_reason_registry"],
    )


def build_session_completion_log(
    *,
    session_name: str,
    freeze_commit: str,
    output_bytes: Mapping[str, bytes],
    access_state: Mapping[str, Any],
    cumulative_prior: Mapping[str, bool],
    generated_at_utc: str,
    config: Mapping[str, Any],
    reason_codes: Sequence[str] = (),
) -> str:
    """Build and validate one hash-bound reporting or audit completion receipt."""
    if not is_utc_iso8601_z(generated_at_utc):
        raise ValueError("Completion-log invocation timestamp is invalid.")
    validate_access_state(access_state, cumulative_prior=cumulative_prior)
    grammar = config["outputs"]["completion_logs"]["machine_validated_log_grammar"]
    binding = grammar["session_bindings_exact"][session_name]
    inventory = tuple(binding["ordered_output_inventory"])
    if tuple(output_bytes) != inventory or any(not isinstance(value, bytes) for value in output_bytes.values()):
        raise ValueError("Completion-log output byte inventory is incomplete or reordered.")
    unique_reasons = sorted(set(reason_codes))
    if list(reason_codes) != unique_reasons and reason_codes:
        raise ValueError("Completion-log reason codes must be unique and sorted.")
    summary_status = "none" if not unique_reasons else "present_retained"
    records: list[tuple[str, ...]] = [
        ("header", "posthoc_final_completion_log.v1", "posthoc_final_supplementary", generated_at_utc),
        ("session_identity", session_name, "complete"),
        ("freeze_commit", freeze_commit),
    ]
    records.extend(
        ("output_hash", path, hashlib.sha256(output_bytes[path]).hexdigest()) for path in inventory
    )
    records.extend(
        [
            (
                "deviations_or_nonestimability_summary",
                summary_status,
                ",".join(unique_reasons),
                "",
            ),
            (
                "access_state_session_activity",
                *(str(value).lower() for value in access_state["session_activity"].values()),
            ),
            (
                "access_state_cumulative_through_session",
                *(str(value).lower() for value in access_state["cumulative_through_session"].values()),
            ),
            ("completion_receipt", generated_at_utc, "complete"),
        ]
    )
    text = "\n".join("\t".join(record) for record in records) + "\n"
    validate_completion_log_against_bytes(
        text,
        session_name=session_name,
        freeze_commit=freeze_commit,
        output_bytes=output_bytes,
        access_state=access_state,
        cumulative_prior=cumulative_prior,
        config=config,
    )
    return text


def validate_completion_log_against_bytes(
    text: str,
    *,
    session_name: str,
    freeze_commit: str,
    output_bytes: Mapping[str, bytes],
    access_state: Mapping[str, Any],
    cumulative_prior: Mapping[str, bool],
    config: Mapping[str, Any],
) -> None:
    """Validate receipt grammar and reconcile every recorded hash to actual bytes."""
    records = validate_session_completion_log(
        text,
        session_name=session_name,
        freeze_commit=freeze_commit,
        expected_activity=access_state["session_activity"],
        expected_cumulative=access_state["cumulative_through_session"],
        cumulative_prior=cumulative_prior,
        config=config,
    )
    binding = config["outputs"]["completion_logs"]["machine_validated_log_grammar"][
        "session_bindings_exact"
    ][session_name]
    inventory = tuple(binding["ordered_output_inventory"])
    if tuple(output_bytes) != inventory:
        raise ValueError("Completion-log audited byte inventory is incomplete or reordered.")
    for index, path in enumerate(inventory, start=3):
        content = output_bytes[path]
        if not isinstance(content, bytes) or records[index][2] != hashlib.sha256(content).hexdigest():
            raise ValueError("Completion-log output hash does not match the candidate bytes.")
    validate_artifact_privacy(
        text,
        artifact_kind="sanitized_log",
        config=config,
        artifact_path=str(binding["log_path"]),
    )


def _completion_log_access_state(
    text: str,
    *,
    session_name: str,
    config: Mapping[str, Any],
) -> OrderedDict[str, OrderedDict[str, bool]]:
    """Parse the two fixed access records before full caller-bound validation."""
    binding = config["outputs"]["completion_logs"]["machine_validated_log_grammar"][
        "session_bindings_exact"
    ][session_name]
    records = [line.split("\t") for line in text.rstrip("\n").split("\n")]
    hash_count = int(binding["output_hash_record_count"])
    activity_index = 4 + hash_count
    cumulative_index = 5 + hash_count
    if len(records) <= cumulative_index:
        raise ValueError("Completion log is incomplete.")
    scopes: list[OrderedDict[str, bool]] = []
    for index in (activity_index, cumulative_index):
        tokens = records[index][1:]
        if len(tokens) != len(ACCESS_STATE_KEYS) or any(token not in {"true", "false"} for token in tokens):
            raise ValueError("Completion-log access record is invalid.")
        scopes.append(OrderedDict(zip(ACCESS_STATE_KEYS, (token == "true" for token in tokens), strict=True)))
    return OrderedDict(
        [
            ("session_activity", scopes[0]),
            ("cumulative_through_session", scopes[1]),
        ]
    )


def validate_completed_session_from_disk(
    *,
    project_root: Path,
    session_name: str,
    freeze_commit: str,
    cumulative_prior: Mapping[str, bool],
    config: Mapping[str, Any],
) -> tuple[OrderedDict[str, OrderedDict[str, bool]], OrderedDict[str, bytes], str]:
    """Validate one prior completion receipt, byte hashes, and cumulative access union."""
    grammar = config["outputs"]["completion_logs"]["machine_validated_log_grammar"]
    binding = grammar["session_bindings_exact"][session_name]
    output_bytes = OrderedDict(
        (path, (project_root / path).read_bytes()) for path in binding["ordered_output_inventory"]
    )
    log_path = project_root / str(binding["log_path"])
    text = log_path.read_text(encoding="utf-8")
    state = _completion_log_access_state(text, session_name=session_name, config=config)
    validate_access_state(state, cumulative_prior=cumulative_prior)
    if session_name in {"reporting", "audit"}:
        validate_completion_log_against_bytes(
            text,
            session_name=session_name,
            freeze_commit=freeze_commit,
            output_bytes=output_bytes,
            access_state=state,
            cumulative_prior=cumulative_prior,
            config=config,
        )
    else:
        bound = dict(binding)
        bound["session_name"] = session_name
        records = validate_completion_log(
            text,
            grammar=grammar,
            session_binding=bound,
            expected_freeze_commit=freeze_commit,
            expected_activity=state["session_activity"],
            expected_cumulative=state["cumulative_through_session"],
            cumulative_prior=cumulative_prior,
            allowed_reason_codes=config["status_and_reason_registries"]["scientific_result_reason_registry"],
        )
        for index, path in enumerate(binding["ordered_output_inventory"], start=3):
            if records[index][2] != hashlib.sha256(output_bytes[path]).hexdigest():
                raise ValueError("Prior completion-log hash does not match its output bytes.")
        validate_artifact_privacy(
            text,
            artifact_kind="sanitized_log",
            config=config,
            artifact_path=str(binding["log_path"]),
        )
    return state, output_bytes, text


def read_typed_registry_csv(path: Path, registry: Mapping[str, Any]) -> pd.DataFrame:
    """Read one aggregate CSV with its exact nullable pandas dtype registry."""
    dtype_map = {column: registry["dtype_map"][column] for column in registry["column_order"]}
    try:
        frame = pd.read_csv(
            path,
            keep_default_na=True,
            float_precision="round_trip",
        )
        frame = frame.astype(dtype_map)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError("Aggregate CSV could not be read under its frozen dtype schema.") from exc
    validate_csv_frame(frame, registry)
    return frame


def validate_prior_artifact_hashes(
    payload: Mapping[str, Any],
    prior_artifact_bytes: Mapping[str, bytes],
    config: Mapping[str, Any],
) -> None:
    """Verify every prior audited byte container and the unchanged Session 6 report identity."""
    expected = config["hash_map_inventories"]["integrity_audit_prior_artifact_sha256_keys_exact_order"]
    if tuple(prior_artifact_bytes) != tuple(expected):
        raise ValueError("Prior-artifact byte inventory is incomplete or reordered.")
    for path, content in prior_artifact_bytes.items():
        if not isinstance(content, bytes) or hashlib.sha256(content).hexdigest() != payload[
            "prior_artifact_sha256"
        ][path]:
            raise ValueError("A prior-artifact hash does not match its audited bytes.")
    report_path = config["outputs"]["scientific_and_closure"]["consolidated_analyses_document"]["path"]
    if payload["consolidated_report_unchanged_sha256"] != payload["prior_artifact_sha256"][report_path]:
        raise ValueError("The consolidated-report unchanged hash does not reconcile.")


def validate_visualization_release(
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    references_valid: bool,
    file_exists: bool,
) -> None:
    """Validate the future release gate without creating or authorizing it in Session 3."""
    schema = config["outputs"]["scientific_and_closure"]["visualization_release"]["schema"]
    if tuple(payload) != tuple(schema["top_level_key_order"]):
        raise ValueError("Visualization-release keys or order are invalid.")
    leaves = schema["leaf_contracts"]
    for key, contract in leaves.items():
        value = payload[key]
        if contract["logical_type"] == "string" and not isinstance(value, str):
            raise ValueError("Visualization-release string field has the wrong type.")
        if contract["logical_type"] == "boolean" and type(value) is not bool:
            raise ValueError("Visualization-release boolean field has the wrong type.")
        if "allowed_value" in contract and value != contract["allowed_value"]:
            raise ValueError("Visualization-release fixed field is invalid.")
        if "allowed_values" in contract and value not in contract["allowed_values"]:
            raise ValueError("Visualization-release enum field is invalid.")
        pattern = contract.get("pattern")
        if pattern == "sha256_64_lowercase_hex" and _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("Visualization-release SHA-256 field is invalid.")
        if pattern == "git_commit_sha1_40_lowercase_hex" and _COMMIT_PATTERN.fullmatch(value) is None:
            raise ValueError("Visualization-release commit field is invalid.")
        if pattern == "utc_iso8601_z_allowlisted_provenance_timestamp" and not is_utc_iso8601_z(value):
            raise ValueError("Visualization-release timestamp is invalid.")
    if payload["approval_status"] != "approved" or payload["user_confirmation"] is not True:
        raise ValueError("Visualization-release approval is incomplete.")
    if references_valid is not True or file_exists is not True:
        raise ValueError("Visualization-release references or existence gate are incomplete.")
    validate_artifact_privacy(payload, artifact_kind="governance_manifest", config=config)


def validate_csv_frame(frame: pd.DataFrame, registry: Mapping[str, Any]) -> None:
    """Validate exact aggregate CSV column order, dtypes, row count, and optional enums."""
    columns = list(registry["column_order"])
    if frame.columns.tolist() != columns:
        raise ValueError("CSV column order does not match the frozen schema.")
    expected_count = registry.get("expected_row_count")
    if expected_count is not None and len(frame) != expected_count:
        raise ValueError("CSV row count does not match the frozen schema.")
    for column in columns:
        expected_dtype = registry["dtype_map"][column]
        observed = str(frame[column].dtype)
        if observed != expected_dtype:
            raise ValueError(f"CSV column '{column}' has the wrong dtype.")
        if not registry["nullable_map"][column] and frame[column].isna().any():
            raise ValueError(f"CSV column '{column}' violates non-nullability.")
        allowed = registry.get("allowed_values", {}).get(column)
        if allowed is not None and not set(frame[column].dropna().astype(str)).issubset(set(allowed)):
            raise ValueError(f"CSV column '{column}' contains a value outside its enum.")


def classify_promotion_state(target_paths: Sequence[Path], completion_log_path: Path) -> str:
    """Classify a future session's absent, partial, or complete individual promotions."""
    target_presence = [path.exists() for path in target_paths]
    log_present = completion_log_path.exists()
    if not any(target_presence) and not log_present:
        return "clean"
    if all(target_presence) and log_present:
        return "complete"
    return "partial"


def audit_promotion_inventory(config: Mapping[str, Any]) -> tuple[tuple[str, ...], str]:
    """Return the exact Session 7 audit output and completion-log destination.

    Args:
        config: Frozen post-hoc configuration.

    Returns:
        Ordered audit output paths and the completion-log path.

    Raises:
        ValueError: If the configured inventory is not the single-output audit contract.
    """
    grammar = config["outputs"]["completion_logs"]["machine_validated_log_grammar"]
    binding = grammar["session_bindings_exact"]["audit"]
    outputs = tuple(binding["ordered_output_inventory"])
    log_path = str(binding["log_path"])
    if outputs != (config["outputs"]["scientific_and_closure"]["integrity_audit"]["path"],):
        raise ValueError("Session 7 promotion inventory is invalid.")
    if log_path in outputs:
        raise ValueError("Session 7 completion log cannot hash itself.")
    return outputs, log_path


def require_clean_promotion_state(target_paths: Sequence[Path], completion_log_path: Path) -> None:
    """Refuse stale complete or partial-interruption states before participant access."""
    state = classify_promotion_state(target_paths, completion_log_path)
    if state != "clean":
        raise ValueError(f"Target state is {state}; provenance review is required before execution.")


def promote_candidates(
    candidates: Sequence[PromotionCandidate],
    *,
    expected_relative_paths: Sequence[str],
    log_candidate: Path,
    log_target: Path,
    log_validator: Callable[[Path], None],
    project_root: Path | None = None,
) -> None:
    """Validate all candidates, then atomically promote each output and the log last."""
    if not candidates or len({candidate.relative_path for candidate in candidates}) != len(candidates):
        raise ValueError("Promotion candidate inventory is empty or duplicated.")
    if tuple(candidate.relative_path for candidate in candidates) != tuple(expected_relative_paths):
        raise ValueError("Promotion candidates do not match the exact frozen output order.")
    if project_root is not None:
        root = project_root.resolve()
        for candidate in candidates:
            relative = Path(candidate.relative_path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Promotion candidate path is not project-relative.")
            if candidate.target_path.resolve() != (root / relative).resolve():
                raise ValueError("Promotion candidate is not bound to its exact configured target.")
    targets = [candidate.target_path for candidate in candidates]
    require_clean_promotion_state(targets, log_target)
    for candidate in candidates:
        if candidate.candidate_path.parent.resolve() != candidate.target_path.parent.resolve():
            raise ValueError("Each candidate must be destination-local.")
        if not candidate.candidate_path.is_file():
            raise ValueError("A promotion candidate is missing.")
        candidate.validator(candidate.candidate_path)
    if log_candidate.parent.resolve() != log_target.parent.resolve() or not log_candidate.is_file():
        raise ValueError("The completion-log candidate must be destination-local and present.")
    log_validator(log_candidate)
    require_clean_promotion_state(targets, log_target)
    for candidate in candidates:
        if candidate.target_path.exists():
            raise ValueError("A promotion target appeared immediately before replacement.")
        os.replace(candidate.candidate_path, candidate.target_path)
    if log_target.exists():
        raise ValueError("The completion-log target appeared immediately before replacement.")
    os.replace(log_candidate, log_target)


def stage_candidate_bytes(
    *,
    project_root: Path,
    output_bytes: Mapping[str, bytes],
    output_validators: Mapping[str, Callable[[Path], None]],
    log_relative_path: str,
    log_bytes: bytes,
    log_validator: Callable[[Path], None],
) -> tuple[tuple[PromotionCandidate, ...], Path, Path]:
    """Write destination-local candidates, read them back, and validate before promotion."""
    if tuple(output_bytes) != tuple(output_validators):
        raise ValueError("Staging byte and validator inventories are incomplete or reordered.")
    root = project_root.resolve()
    staged_paths: list[Path] = []
    candidates: list[PromotionCandidate] = []

    def stage(relative_text: str, payload: bytes) -> tuple[Path, Path]:
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Staging destination must be project-relative.")
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            raise ValueError("Staging destination escapes the project root.") from None
        if target.exists():
            raise ValueError("Refusing to stage over an existing configured target.")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            candidate_path = Path(handle.name)
        staged_paths.append(candidate_path)
        if candidate_path.parent.resolve() != target.parent.resolve() or candidate_path.read_bytes() != payload:
            raise ValueError("Destination-local staged candidate failed byte readback.")
        return candidate_path, target

    try:
        for relative_path, payload in output_bytes.items():
            if not isinstance(payload, bytes):
                raise TypeError("Staged artifact payloads must be serialized bytes.")
            candidate_path, target_path = stage(relative_path, payload)
            validator = output_validators[relative_path]
            validator(candidate_path)
            candidates.append(
                PromotionCandidate(relative_path, candidate_path, target_path, validator)
            )
        log_candidate, log_target = stage(log_relative_path, log_bytes)
        log_validator(log_candidate)
    except Exception:
        for path in staged_paths:
            path.unlink(missing_ok=True)
        raise
    return tuple(candidates), log_candidate, log_target


def validate_session7_additive_scope(
    *,
    changed_paths: Sequence[str],
    report_hash_before: str,
    report_hash_after: str,
    config: Mapping[str, Any],
) -> None:
    """Require exact additive Session 7 paths and an unchanged Session 6 report."""
    expected = [
        config["outputs"]["scientific_and_closure"]["integrity_audit"]["path"],
        config["outputs"]["completion_logs"]["audit"],
    ]
    if list(changed_paths) != expected or report_hash_before != report_hash_after:
        raise ValueError("Session 7 must remain additive and leave the consolidated report unchanged.")


def _git(*args: str) -> str:
    """Run one read-only Git command from the project root."""
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def validate_audit_authority(
    config_path: Path,
    freeze_commit: str,
    amendment_freeze_commit: str,
    *,
    git_runner: Callable[..., str] = _git,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Validate current authority, both freezes, prior artifacts, and clean audit targets."""
    root = (project_root or PROJECT_ROOT).resolve()
    if _COMMIT_PATTERN.fullmatch(freeze_commit) is None or _COMMIT_PATTERN.fullmatch(amendment_freeze_commit) is None:
        raise ValueError("Both freeze commits must be exact lowercase 40-character hashes.")
    resolved = config_path if config_path.is_absolute() else root / config_path
    config = load_effective_posthoc_config(resolved)
    manifest = json.loads((root / CURRENT_MANIFEST_RELATIVE_PATH).read_text(encoding="utf-8"))
    current = validate_current_runtime_authority(
        resolved,
        config,
        manifest,
        require_head_identical=True,
        git_runner=git_runner,
    )
    _validate_annotated_tag_chain(git_runner)
    session0 = validate_freeze_commit(freeze_commit, config, manifest, git_runner=git_runner)
    amendment = validate_amendment_freeze_commit(
        amendment_freeze_commit,
        historical_amendment=manifest["historical_evidence"]["amendment_01"],
        git_runner=git_runner,
    )
    prior_paths = [
        *config["outputs"]["future_implementation_paths"],
        *config["hash_map_inventories"]["integrity_audit_prior_artifact_sha256_keys_exact_order"],
    ]
    validate_execute_git_prerequisites(tuple(dict.fromkeys(prior_paths)), git_runner=git_runner)
    binding = config["outputs"]["completion_logs"]["machine_validated_log_grammar"]["session_bindings_exact"]["audit"]
    targets = [root / path for path in binding["ordered_output_inventory"]]
    require_clean_promotion_state(targets, root / binding["log_path"])
    release_path = root / config["outputs"]["scientific_and_closure"]["visualization_release"]["path"]
    if release_path.exists():
        raise ValueError("Visualization release must remain absent through Session 7.")
    preflight_path = root / SCORE_BLIND_PREFLIGHT_RELATIVE_PATH
    expected_preflight_hash = manifest["historical_evidence"]["completed_session_1"]["artifact_sha256"][
        SCORE_BLIND_PREFLIGHT_RELATIVE_PATH
    ]
    if sha256_file(preflight_path) != expected_preflight_hash:
        raise ValueError("Score-blind preflight hash does not match the frozen authority.")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"), object_pairs_hook=OrderedDict)
    prior = OrderedDict(
        config["historical_consolidation"]["amendment_01"]["access_state"]
        ["step_2_approval_materialization"]["cumulative_through_session"]
    )
    validate_preflight_schema(
        preflight,
        frozen_hash_paths=list(config["frozen_inputs"]["whole_container_sha256"]),
        expected_cumulative_prior=prior,
    )
    validate_preflight_privacy(preflight)
    if preflight["status"] != "pass" or preflight["freeze_commit_validation"]["freeze_commit_id"] != freeze_commit:
        raise ValueError("A passing committed score-blind preflight is required.")
    cumulative = preflight["access_state"]["cumulative_through_session"]
    reporting_cumulative_prior: Mapping[str, bool] | None = None
    for session_name in ("locked_scores", "date_grouped", "reporting"):
        if session_name == "reporting":
            reporting_cumulative_prior = OrderedDict(cumulative)
        state, _, _ = validate_completed_session_from_disk(
            project_root=root,
            session_name=session_name,
            freeze_commit=freeze_commit,
            cumulative_prior=cumulative,
            config=config,
        )
        cumulative = state["cumulative_through_session"]
    protected_paths = tuple(
        dict.fromkeys(
            [
                *config["frozen_inputs"]["whole_container_sha256"],
                *config["hash_map_inventories"]["integrity_audit_prior_artifact_sha256_keys_exact_order"],
                *config["hash_map_inventories"]["execution_manifest_code_sha256_keys_exact_order"],
                *config["outputs"]["future_implementation_paths"],
                config["authority"]["reviewed_plan_path"],
                config["authority"]["agent_guidance_path"],
            ]
        )
    )
    entry_hashes = capture_integrity_snapshot(
        root,
        protected_paths,
        declared_hashes=config["frozen_inputs"]["whole_container_sha256"],
    )
    return {
        "config": config,
        "manifest": manifest,
        "current_runtime": current,
        "session0_freeze": session0,
        "amendment_freeze": amendment,
        "cumulative_prior": OrderedDict(cumulative),
        "reporting_cumulative_prior": OrderedDict(reporting_cumulative_prior or cumulative),
        "entry_hashes": entry_hashes,
        "required_paths": tuple(dict.fromkeys(prior_paths)),
        "project_root": root,
        "git_runner": git_runner,
    }


def load_prior_audit_inputs(authority: Mapping[str, Any]) -> OrderedDict[str, bytes]:
    """Load the exact prior aggregate/artifact byte inventory after authority validation."""
    config = authority["config"]
    root = Path(authority.get("project_root", PROJECT_ROOT))
    paths = config["hash_map_inventories"]["integrity_audit_prior_artifact_sha256_keys_exact_order"]
    return OrderedDict((path, (root / path).read_bytes()) for path in paths)


class AuditCheckContext(NamedTuple):
    """Validated aggregate-only evidence available to IA01--IA12."""

    authority: Mapping[str, Any]
    prior_inputs: Mapping[str, bytes]
    privacy_references: PrivacyReferences
    entry_hashes: Mapping[str, str]
    frames: Mapping[str, pd.DataFrame]
    fact_audit: Mapping[str, Any]
    execution_manifest: Mapping[str, Any]
    consolidated_report: str
    reporting_log: str
    reporting_report_anchor: str


class AuditPromotionBundle(NamedTuple):
    """Fully staged and read-back-validated Session 7 promotion state."""

    candidates: tuple[PromotionCandidate, ...]
    log_candidate: Path
    log_target: Path
    log_validator: Callable[[Path], None]


def _audit_registry_frames(
    authority: Mapping[str, Any],
    prior_inputs: Mapping[str, bytes],
) -> OrderedDict[str, pd.DataFrame]:
    """Read every prior scientific CSV under its frozen typed registry."""
    config = authority["config"]
    root = Path(authority["project_root"])
    registry_by_path = {
        entry["path"]: entry
        for entry in config["row_order_and_schema_validation_registry"].values()
        if isinstance(entry, Mapping) and "path" in entry and "column_order" in entry
    }
    expected_paths = tuple(path for path in prior_inputs if path in registry_by_path)
    frames: OrderedDict[str, pd.DataFrame] = OrderedDict()
    for path in expected_paths:
        if (root / path).read_bytes() != prior_inputs[path]:
            raise ValueError("A prior CSV changed after the audit byte load.")
        frames[path] = read_typed_registry_csv(root / path, registry_by_path[path])
    return frames


def _reporting_log_hash_map(text: str, config: Mapping[str, Any]) -> OrderedDict[str, str]:
    """Parse the already grammar-validated reporting output-hash records."""
    binding = config["outputs"]["completion_logs"]["machine_validated_log_grammar"][
        "session_bindings_exact"
    ]["reporting"]
    records = [line.split("\t") for line in text.rstrip("\n").split("\n")]
    hashes = OrderedDict()
    for index, path in enumerate(binding["ordered_output_inventory"], start=3):
        if len(records[index]) != 3 or records[index][0] != "output_hash" or records[index][1] != path:
            raise ValueError("Reporting completion-log hash inventory is invalid.")
        hashes[path] = records[index][2]
    return hashes


def build_audit_check_context(
    authority: Mapping[str, Any],
    prior_inputs: Mapping[str, bytes],
    privacy_references: PrivacyReferences,
) -> AuditCheckContext:
    """Parse prior aggregate evidence and bind the Session 6 report handoff anchor."""
    config = authority["config"]
    expected = tuple(config["hash_map_inventories"]["integrity_audit_prior_artifact_sha256_keys_exact_order"])
    if tuple(prior_inputs) != expected:
        raise ValueError("Audit prior-input inventory is incomplete or reordered.")
    frames = _audit_registry_frames(authority, prior_inputs)
    facts_path = config["outputs"]["scientific_and_closure"]["reference_standard_timing_audit"]["path"]
    manifest_path = config["outputs"]["scientific_and_closure"]["execution_manifest"]["path"]
    report_path = config["outputs"]["scientific_and_closure"]["consolidated_analyses_document"]["path"]
    log_path = config["outputs"]["completion_logs"]["reporting"]
    fact_audit = json.loads(prior_inputs[facts_path].decode("utf-8"), object_pairs_hook=OrderedDict)
    execution_manifest = json.loads(
        prior_inputs[manifest_path].decode("utf-8"), object_pairs_hook=OrderedDict
    )
    report = prior_inputs[report_path].decode("utf-8")
    reporting_log = prior_inputs[log_path].decode("utf-8")
    anchor = _reporting_log_hash_map(reporting_log, config)[report_path]
    return AuditCheckContext(
        authority=authority,
        prior_inputs=prior_inputs,
        privacy_references=privacy_references,
        entry_hashes=authority["entry_hashes"],
        frames=frames,
        fact_audit=fact_audit,
        execution_manifest=execution_manifest,
        consolidated_report=report,
        reporting_log=reporting_log,
        reporting_report_anchor=anchor,
    )


def _session6_reporting_frames(context: AuditCheckContext) -> OrderedDict[str, pd.DataFrame]:
    """Map the five Session 6 reporting frames back to their production identities."""
    outputs = context.authority["config"]["outputs"]["scientific_and_closure"]
    return OrderedDict(
        [
            ("cohort_flow", context.frames[outputs["cohort_flow"]["path"]]),
            ("included_excluded_summary", context.frames[outputs["included_excluded_summary"]["path"]]),
            ("included_excluded_smd", context.frames[outputs["included_excluded_smd"]["path"]]),
            ("calibration_points", context.frames[outputs["calibration_points"]["path"]]),
            ("calibration_smoother", context.frames[outputs["calibration_smoother"]["path"]]),
        ]
    )


def _session4_5_frames(context: AuditCheckContext) -> OrderedDict[str, pd.DataFrame]:
    """Return the exact persisted Session 4/5 aggregate inventory for semantic validation."""
    config = context.authority["config"]
    grammar = config["outputs"]["completion_logs"]["machine_validated_log_grammar"]
    paths = (
        *grammar["session_bindings_exact"]["locked_scores"]["ordered_output_inventory"],
        *grammar["session_bindings_exact"]["date_grouped"]["ordered_output_inventory"],
    )
    return OrderedDict((path, context.frames[path]) for path in paths)


def _check_schema_dtype_order_validation(context: AuditCheckContext) -> str:
    """Execute IA01 exact CSV/JSON schema, dtype, and order checks."""
    from src.analysis import posthoc_final_reporting as reporting

    config = context.authority["config"]
    prior_evidence = reporting.validate_prior_aggregate_frames(_session4_5_frames(context), config)
    reporting_frames = _session6_reporting_frames(context)
    reporting.validate_cohort_flow(reporting_frames["cohort_flow"], config)
    reporting.validate_reporting_cross_artifact_identities(reporting_frames, config)
    flow_n = int(
        reporting_frames["cohort_flow"].loc[
            reporting_frames["cohort_flow"]["node_id"] == "internal_test", "n"
        ].iloc[0]
    )
    points = reporting_frames["calibration_points"]
    if (
        int(points["n"].sum()) != prior_evidence.internal_test_n
        or int(points["events"].sum()) != prior_evidence.internal_test_n_positive
        or int(points["nonevents"].sum()) != prior_evidence.internal_test_n_negative
    ):
        raise ValueError("Calibration totals do not match the independent Session 4 event anchor.")
    reporting.validate_calibration_points(
        points,
        expected_n=prior_evidence.internal_test_n,
        expected_events=prior_evidence.internal_test_n_positive,
        config=config,
    )
    if flow_n != prior_evidence.internal_test_n:
        raise ValueError("Calibration totals do not match the independent Session 4 event anchor.")
    reporting.validate_calibration_smoother(reporting_frames["calibration_smoother"], config)
    reporting.validate_reference_standard_timing_audit(context.fact_audit, config)
    qc6_path = config["outputs"]["scientific_and_closure"]["qc6_lineage_correction"]["path"]
    try:
        qc6_note = context.prior_inputs[qc6_path].decode("utf-8")
    except (KeyError, UnicodeDecodeError) as exc:
        raise ValueError("QC6 correction artifact is missing or is not valid UTF-8.") from exc
    reporting.validate_qc6_lineage_correction(
        qc6_note,
        config,
        correction_date_utc=str(context.fact_audit["generated_at_utc"]),
    )
    validate_execution_manifest(context.execution_manifest, config)
    return f"executed:schema_dtype_order_validation; csv_artifact_count={len(context.frames)}; json_artifact_count=2"


def _check_planned_row_and_status_retention(context: AuditCheckContext) -> str:
    """Execute IA02 planned row, status, and reason completeness checks."""
    config = context.authority["config"]
    point_allowed = set(config["status_and_reason_registries"]["point_estimability_status"])
    ci_allowed = set(config["status_and_reason_registries"]["ci_status"])
    lowess_allowed = set(config["status_and_reason_registries"]["lowess_row_estimability_status"])
    lowess_path = config["outputs"]["scientific_and_closure"]["calibration_smoother"]["path"]
    reason_count = 0
    for path, frame in context.frames.items():
        status_contracts = (
            (
                "estimability_status",
                "nonestimable_reason",
                lowess_allowed if path == lowess_path else point_allowed,
                {"non_estimable", "not_evaluated"} if path == lowess_path else {"non_estimable"},
            ),
            ("ci_status", "ci_nonestimable_reason", ci_allowed, {"non_estimable"}),
        )
        for status_column, reason_column, allowed, reason_required_statuses in status_contracts:
            if status_column not in frame.columns:
                continue
            statuses = set(frame[status_column].dropna().astype(str))
            if not statuses.issubset(allowed):
                raise ValueError("A prior artifact contains an undeclared estimability status.")
            if reason_column in frame.columns:
                reason_required = frame[status_column].astype("string").isin(reason_required_statuses)
                if frame.loc[reason_required, reason_column].isna().any():
                    raise ValueError("A non-estimable retained row is missing its reason code.")
                reason_count += int(frame.loc[reason_required, reason_column].notna().sum())
    return f"executed:planned_row_and_status_retention; retained_reason_count={reason_count}"


def _check_parent_child_count_reconciliation(context: AuditCheckContext) -> str:
    """Execute IA03 parent-child and calibration-count reconciliation."""
    from src.analysis import posthoc_final_reporting as reporting

    frames = _session6_reporting_frames(context)
    config = context.authority["config"]
    prior_evidence = reporting.validate_prior_aggregate_frames(_session4_5_frames(context), config)
    reporting.validate_cohort_flow(frames["cohort_flow"], config)
    reporting.validate_reporting_cross_artifact_identities(frames, config)
    internal_n = int(
        frames["cohort_flow"].loc[frames["cohort_flow"]["node_id"] == "internal_test", "n"].iloc[0]
    )
    if int(frames["calibration_points"]["n"].sum()) != internal_n:
        raise ValueError("Calibration bins do not reconcile to the internal-test cohort-flow node.")
    if (
        internal_n != prior_evidence.internal_test_n
        or int(frames["calibration_points"]["events"].sum())
        != prior_evidence.internal_test_n_positive
        or int(frames["calibration_points"]["nonevents"].sum())
        != prior_evidence.internal_test_n_negative
    ):
        raise ValueError("Calibration totals do not match the independent Session 4 event anchor.")
    return f"executed:parent_child_count_reconciliation; internal_test_n={internal_n}"


def _check_frozen_source_rule_tag_stability(context: AuditCheckContext) -> str:
    """Execute IA04 source-hash, rule, and tag stability checks."""
    authority = context.authority
    for section in ("current_runtime", "session0_freeze", "amendment_freeze"):
        values = authority[section]
        identity_fields = {"freeze_commit_id", "amendment_freeze_commit_id"}
        if any(value is not True for key, value in values.items() if key not in identity_fields):
            raise ValueError("A frozen authority or tag-stability flag is false.")
    declared = authority["config"]["frozen_inputs"]["whole_container_sha256"]
    if any(context.entry_hashes[path] != digest for path, digest in declared.items()):
        raise ValueError("A frozen source differs from its declared hash.")
    return f"executed:frozen_source_rule_tag_stability; frozen_container_count={len(declared)}"


def _check_hash_chain_and_ancestry(context: AuditCheckContext) -> str:
    """Execute IA05 prior-container hash and validated ancestry checks."""
    expected = context.authority["config"]["hash_map_inventories"][
        "integrity_audit_prior_artifact_sha256_keys_exact_order"
    ]
    if tuple(context.prior_inputs) != tuple(expected):
        raise ValueError("Prior hash-chain inventory is invalid.")
    for path, content in context.prior_inputs.items():
        if hashlib.sha256(content).hexdigest() != context.entry_hashes[path]:
            raise ValueError("A prior artifact does not match its entry integrity anchor.")
    for section in ("session0_freeze", "amendment_freeze"):
        if any(
            value is not True
            for key, value in context.authority[section].items()
            if key not in {"freeze_commit_id", "amendment_freeze_commit_id"}
        ):
            raise ValueError("Historical ancestry validation did not pass.")
    return f"executed:hash_chain_and_ancestry; prior_artifact_count={len(expected)}"


def _check_stale_output_and_namespace(context: AuditCheckContext) -> str:
    """Execute IA06 exact namespace and clean additive-target checks."""
    config = context.authority["config"]
    root = Path(context.authority["project_root"])
    outputs, log_path = audit_promotion_inventory(config)
    require_clean_promotion_state([root / path for path in outputs], root / log_path)
    release = root / config["outputs"]["scientific_and_closure"]["visualization_release"]["path"]
    if release.exists():
        raise ValueError("Visualization release exists before authorization.")
    allowed_paths = {
        *config["hash_map_inventories"]["integrity_audit_prior_artifact_sha256_keys_exact_order"],
        *outputs,
        log_path,
        str(release.relative_to(root)),
    }
    search_roots = (
        root / "outputs",
        root / "docs/preprocessing",
        root / "docs/modeling",
    )
    git_runner = context.authority["git_runner"]
    freeze_commit = context.authority["session0_freeze"]["freeze_commit_id"]
    scanned_count = 0
    violation_count = 0
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for path in search_root.rglob("*"):
            if not path.is_file():
                continue
            scanned_count += 1
            relative = str(path.relative_to(root))
            undeclared_supplement = "posthoc_final" in path.name and relative not in allowed_paths
            lower_name = path.name.lower()
            prohibited_kind = (
                relative.startswith("outputs/figures/")
                or relative.startswith("outputs/models/")
                or relative.startswith("outputs/predictions/")
                or ("fold" in lower_name and "assign" in lower_name)
            )
            existed_at_freeze = False
            if prohibited_kind:
                try:
                    git_runner("cat-file", "-e", f"{freeze_commit}:{relative}")
                    existed_at_freeze = True
                except Exception:
                    existed_at_freeze = False
            try:
                git_runner("ls-files", "--error-unmatch", relative)
                tracked = True
            except Exception:
                tracked = False
            newly_generated = relative not in allowed_paths and not tracked
            prohibited_post_freeze = prohibited_kind and relative not in allowed_paths and not existed_at_freeze
            if undeclared_supplement or newly_generated or prohibited_post_freeze:
                violation_count += 1
    if violation_count:
        raise ValueError(
            "Artifact namespace contains unexpected supplementary or newly generated prohibited artifacts "
            f"(match_count={violation_count})."
        )
    return (
        "executed:stale_output_and_namespace; audit_targets_absent=2; "
        f"filesystem_artifact_count={scanned_count}"
    )


def _check_privacy_and_path_sanitization(context: AuditCheckContext) -> str:
    """Execute IA07 privacy scans across every scientific/reporting representation."""
    config = context.authority["config"]
    scientific = config["outputs"]["scientific_and_closure"]
    scanned = 0
    for path, frame in context.frames.items():
        validate_artifact_privacy(
            frame,
            artifact_kind="scientific_csv",
            config=config,
            privacy_references=context.privacy_references,
            artifact_path=path,
        )
        validate_artifact_privacy(
            context.prior_inputs[path],
            artifact_kind="scientific_csv",
            config=config,
            privacy_references=context.privacy_references,
            artifact_path=path,
        )
        scanned += 1
    structured = {
        scientific["reference_standard_timing_audit"]["path"]: context.fact_audit,
        scientific["execution_manifest"]["path"]: context.execution_manifest,
    }
    for path, payload in structured.items():
        kind = "governance_manifest" if path == scientific["execution_manifest"]["path"] else "structured_json"
        validate_artifact_privacy(
            payload,
            artifact_kind=kind,
            config=config,
            privacy_references=context.privacy_references,
            artifact_path=path,
        )
        validate_artifact_privacy(
            context.prior_inputs[path],
            artifact_kind=kind,
            config=config,
            privacy_references=context.privacy_references,
            artifact_path=path,
        )
        scanned += 1
    for path in (
        scientific["qc6_lineage_correction"]["path"],
        scientific["consolidated_analyses_document"]["path"],
    ):
        validate_artifact_privacy(
            context.prior_inputs[path],
            artifact_kind="markdown_report",
            config=config,
            privacy_references=context.privacy_references,
            artifact_path=path,
        )
        scanned += 1
    reporting_log_path = config["outputs"]["completion_logs"]["reporting"]
    validate_artifact_privacy(
        context.reporting_log,
        artifact_kind="sanitized_log",
        config=config,
        privacy_references=context.privacy_references,
        artifact_path=reporting_log_path,
    )
    scanned += 1
    return f"executed:privacy_and_path_sanitization; artifact_representation_count={scanned}"


def _check_negative_result_retention(context: AuditCheckContext) -> str:
    """Execute IA08 retention of every non-estimable reason in the report."""
    from src.analysis import posthoc_final_reporting as reporting

    config = context.authority["config"]
    prior_paths = reporting._prior_aggregate_inventory(config)
    prior_frames = OrderedDict((path, context.frames[path]) for path in prior_paths)
    reporting_frames = _session6_reporting_frames(context)
    qc6_path = config["outputs"]["scientific_and_closure"]["qc6_lineage_correction"]["path"]
    expected_report = reporting.build_consolidated_report(
        prior_frames,
        reporting_frames,
        context.fact_audit,
        context.prior_inputs[qc6_path].decode("utf-8"),
        config,
    )
    if context.consolidated_report != expected_report:
        raise ValueError("The consolidated report is not the deterministic complete aggregate rendering.")
    reasons: set[str] = set()
    for frame in context.frames.values():
        for reason_column in ("nonestimable_reason", "ci_nonestimable_reason"):
            if reason_column in frame.columns:
                reasons.update(str(value) for value in frame[reason_column].dropna().tolist() if str(value))
    if any(reason not in context.consolidated_report for reason in reasons):
        raise ValueError("The consolidated report omits a retained non-estimable result reason.")
    required_pending = (
        "Session 7 integrity audit is pending",
        "Visualization release remains unauthorized",
        config["outputs"]["scientific_and_closure"]["integrity_audit"]["path"],
        config["outputs"]["scientific_and_closure"]["visualization_release"]["path"],
    )
    if any(statement not in context.consolidated_report for statement in required_pending):
        raise ValueError("The consolidated report overstates downstream completion.")
    return f"executed:negative_result_retention; retained_reason_count={len(reasons)}"


def _check_source_immutability(context: AuditCheckContext) -> str:
    """Execute IA09 entry/current source immutability comparison."""
    verify_integrity_snapshot(Path(context.authority["project_root"]), context.entry_hashes)
    return f"executed:source_immutability; protected_container_count={len(context.entry_hashes)}"


def _check_non_circular_manifest_log_provenance(context: AuditCheckContext) -> str:
    """Execute IA10 Session 6 manifest/log hash reconciliation and self exclusions."""
    from src.analysis import posthoc_final_reporting as reporting

    config = context.authority["config"]
    output_paths = config["hash_map_inventories"]["execution_manifest_output_sha256_keys_exact_order"]
    output_bytes = OrderedDict((path, context.prior_inputs[path]) for path in output_paths)
    reporting.validate_reporting_execution_manifest_binding(
        context.execution_manifest,
        output_bytes,
        context.entry_hashes,
        context.authority,
        expected_deviations=reporting.collect_nonestimability_records(context.frames, config),
    )
    binding = config["outputs"]["completion_logs"]["machine_validated_log_grammar"][
        "session_bindings_exact"
    ]["reporting"]
    session6_bytes = OrderedDict((path, context.prior_inputs[path]) for path in binding["ordered_output_inventory"])
    state = context.execution_manifest["access_state"]
    validate_completion_log_against_bytes(
        context.reporting_log,
        session_name="reporting",
        freeze_commit=context.authority["session0_freeze"]["freeze_commit_id"],
        output_bytes=session6_bytes,
        access_state=state,
        cumulative_prior=context.authority["reporting_cumulative_prior"],
        config=config,
    )
    return "executed:non_circular_manifest_log_provenance; manifest_self_hash=false; log_self_hash=false"


def _check_unchanged_consolidated_report_hash(context: AuditCheckContext) -> str:
    """Execute IA11 current report versus Session 6 completion-log anchor comparison."""
    digest = hashlib.sha256(context.consolidated_report.encode("utf-8")).hexdigest()
    if digest != context.reporting_report_anchor:
        raise ValueError("The consolidated report differs from the Session 6 handoff anchor.")
    return "executed:unchanged_consolidated_report_hash; handoff_anchor_match=true"


def _check_visualization_authorization_gate_false(context: AuditCheckContext) -> str:
    """Execute IA12 fixed false authorization and absent release-file gate."""
    config = context.authority["config"]
    release = Path(context.authority["project_root"]) / config["outputs"]["scientific_and_closure"][
        "visualization_release"
    ]["path"]
    if config["visualization_v2_authorized"] is not False or release.exists():
        raise ValueError("Visualization authorization is not fixed false through Session 7.")
    return "executed:visualization_authorization_gate_false; authorized=false; release_absent=true"


def _audit_check_functions(context: AuditCheckContext) -> OrderedDict[str, Callable[[], str]]:
    """Bind the exact configured IA names to evidence-producing implementations."""
    functions: OrderedDict[str, Callable[[], str]] = OrderedDict(
        [
            ("schema_dtype_order_validation", lambda: _check_schema_dtype_order_validation(context)),
            ("planned_row_and_status_retention", lambda: _check_planned_row_and_status_retention(context)),
            ("parent_child_count_reconciliation", lambda: _check_parent_child_count_reconciliation(context)),
            ("frozen_source_rule_tag_stability", lambda: _check_frozen_source_rule_tag_stability(context)),
            ("hash_chain_and_ancestry", lambda: _check_hash_chain_and_ancestry(context)),
            ("stale_output_and_namespace", lambda: _check_stale_output_and_namespace(context)),
            ("privacy_and_path_sanitization", lambda: _check_privacy_and_path_sanitization(context)),
            ("negative_result_retention", lambda: _check_negative_result_retention(context)),
            ("source_immutability", lambda: _check_source_immutability(context)),
            (
                "non_circular_manifest_log_provenance",
                lambda: _check_non_circular_manifest_log_provenance(context),
            ),
            ("unchanged_consolidated_report_hash", lambda: _check_unchanged_consolidated_report_hash(context)),
            (
                "visualization_authorization_gate_false",
                lambda: _check_visualization_authorization_gate_false(context),
            ),
        ]
    )
    return functions


def calculate_integrity_audit_candidate(
    authority: Mapping[str, Any],
    prior_inputs: Mapping[str, bytes],
    *,
    privacy_references: PrivacyReferences,
    generated_at_utc: str,
    access_state: Mapping[str, Any],
) -> OrderedDict[str, Any]:
    """Execute every IA check and build the aggregate-only candidate with evidence."""
    config = authority["config"]
    if not is_utc_iso8601_z(generated_at_utc):
        raise ValueError("Audit invocation timestamp is invalid.")
    validate_access_state(access_state, cumulative_prior=authority["cumulative_prior"])
    context = build_audit_check_context(authority, prior_inputs, privacy_references)
    checks = execute_integrity_check_registry(config, _audit_check_functions(context))
    hashes = OrderedDict((path, hashlib.sha256(prior_inputs[path]).hexdigest()) for path in prior_inputs)
    report_path = config["outputs"]["scientific_and_closure"]["consolidated_analyses_document"]["path"]
    payload = OrderedDict(
        [
            ("schema_version", "posthoc_final_integrity_audit.v1"),
            ("package_id", "posthoc_final_supplementary"),
            ("generated_at_utc", generated_at_utc),
            ("freeze_commit", authority["session0_freeze"]["freeze_commit_id"]),
            ("checks", checks),
            ("prior_artifact_sha256", hashes),
            ("consolidated_report_unchanged_sha256", hashes[report_path]),
            ("visualization_v2_authorized", False),
            ("access_state", OrderedDict(access_state)),
            ("self_hash_policy", "forbidden"),
        ]
    )
    return payload


def validate_integrity_audit_candidate(
    candidate: Mapping[str, Any],
    authority: Mapping[str, Any],
    prior_inputs: Mapping[str, bytes] | None = None,
    *,
    require_success: bool = True,
) -> None:
    """Validate schema, executed evidence, prior hashes, and the all-pass completion gate."""
    validate_integrity_audit(candidate, authority["config"])
    for item in candidate["checks"]:
        if not isinstance(item["evidence_note"], str) or not item["evidence_note"].startswith("executed:"):
            raise ValueError("Every integrity status requires executed deterministic evidence.")
    if prior_inputs is not None:
        validate_prior_artifact_hashes(candidate, prior_inputs, authority["config"])
    if require_success:
        require_all_integrity_checks_pass(candidate["checks"])


def serialize_integrity_audit_candidate(candidate: Mapping[str, Any]) -> bytes:
    """Serialize one ordered integrity-audit candidate deterministically."""
    return (json.dumps(candidate, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def require_byte_identical_audit_candidates(first: bytes, second: bytes) -> None:
    """Require byte-identical isolated Session 7 candidate serializations."""
    if not isinstance(first, bytes) or not isinstance(second, bytes) or first != second:
        raise ValueError("Isolated integrity-audit candidate bytes disagree.")


def _validate_audit_final_git_state(
    *,
    git_runner: Callable[..., str],
    allowed_staged_paths: Sequence[Path],
    project_root: Path,
    required_paths: Sequence[str],
) -> None:
    """Recheck Session 7 Git cleanliness and prerequisite HEAD identity before promotion."""
    allowed = {
        str(path.resolve().relative_to(project_root.resolve()))
        for path in allowed_staged_paths
    }
    status = git_runner("status", "--porcelain", "--untracked-files=all")
    for line in status.splitlines():
        path_text = line[3:] if len(line) >= 4 else ""
        if " -> " in path_text:
            path_text = path_text.split(" -> ", 1)[1]
        if path_text not in allowed:
            raise ValueError("Final audit Git state contains an unexpected path.")
    for path in required_paths:
        git_runner("ls-files", "--error-unmatch", path)
        git_runner("cat-file", "-e", f"HEAD:{path}")
        if git_runner("rev-parse", f"HEAD:{path}") != git_runner("hash-object", path):
            raise ValueError("A committed audit prerequisite changed before promotion.")


def prepare_audit_promotion(
    *,
    authority: Mapping[str, Any],
    prior_inputs: Mapping[str, bytes],
    privacy_references: PrivacyReferences,
    generated_at_utc: str,
    git_runner: Callable[..., str],
) -> AuditPromotionBundle:
    """Execute IA01--IA12 twice, validate, stage/read back, and freeze Session 7 candidates."""
    config = authority["config"]
    access_state = build_audit_access_state(authority["cumulative_prior"])
    first = calculate_integrity_audit_candidate(
        authority,
        prior_inputs,
        privacy_references=privacy_references,
        generated_at_utc=generated_at_utc,
        access_state=access_state,
    )
    second = calculate_integrity_audit_candidate(
        authority,
        OrderedDict((path, bytes(value)) for path, value in prior_inputs.items()),
        privacy_references=privacy_references,
        generated_at_utc=generated_at_utc,
        access_state=access_state,
    )
    first_bytes = serialize_integrity_audit_candidate(first)
    second_bytes = serialize_integrity_audit_candidate(second)
    require_byte_identical_audit_candidates(first_bytes, second_bytes)
    validate_integrity_audit_candidate(first, authority, prior_inputs, require_success=True)
    audit_path, log_relative = audit_promotion_inventory(config)
    output_bytes = OrderedDict([(audit_path[0], first_bytes)])
    log_text = build_session_completion_log(
        session_name="audit",
        freeze_commit=authority["session0_freeze"]["freeze_commit_id"],
        output_bytes=output_bytes,
        access_state=access_state,
        cumulative_prior=authority["cumulative_prior"],
        generated_at_utc=generated_at_utc,
        config=config,
    )
    validate_artifact_privacy(
        first,
        artifact_kind="governance_manifest",
        config=config,
        privacy_references=privacy_references,
        artifact_path=audit_path[0],
    )
    validate_artifact_privacy(
        first_bytes,
        artifact_kind="governance_manifest",
        config=config,
        privacy_references=privacy_references,
        artifact_path=audit_path[0],
    )

    def candidate_validator(path: Path) -> None:
        """Read back and validate the exact staged audit JSON."""
        if path.read_bytes() != first_bytes:
            raise ValueError("Staged integrity-audit bytes changed during readback.")
        payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=OrderedDict)
        if payload != first:
            raise ValueError("Staged integrity-audit JSON changed during readback.")
        validate_integrity_audit_candidate(payload, authority, prior_inputs, require_success=True)
        validate_artifact_privacy(
            payload,
            artifact_kind="governance_manifest",
            config=config,
            privacy_references=privacy_references,
            artifact_path=audit_path[0],
        )

    def log_validator(path: Path) -> None:
        """Read back and reconcile the staged audit completion receipt."""
        staged_text = path.read_text(encoding="utf-8")
        validate_completion_log_against_bytes(
            staged_text,
            session_name="audit",
            freeze_commit=authority["session0_freeze"]["freeze_commit_id"],
            output_bytes=output_bytes,
            access_state=access_state,
            cumulative_prior=authority["cumulative_prior"],
            config=config,
        )
        for log_representation in (staged_text, path.read_bytes()):
            validate_artifact_privacy(
                log_representation,
                artifact_kind="sanitized_log",
                config=config,
                privacy_references=privacy_references,
                artifact_path=log_relative,
            )

    candidates, log_candidate, log_target = stage_candidate_bytes(
        project_root=authority["project_root"],
        output_bytes=output_bytes,
        output_validators=OrderedDict([(audit_path[0], candidate_validator)]),
        log_relative_path=log_relative,
        log_bytes=log_text.encode("utf-8"),
        log_validator=log_validator,
    )
    report_path = config["outputs"]["scientific_and_closure"]["consolidated_analyses_document"]["path"]
    report_hash_before = hashlib.sha256(prior_inputs[report_path]).hexdigest()
    try:
        candidate_validator(candidates[0].candidate_path)
        log_validator(log_candidate)
        verify_integrity_snapshot(Path(authority["project_root"]), authority["entry_hashes"])
        report_hash_after = hashlib.sha256((Path(authority["project_root"]) / report_path).read_bytes()).hexdigest()
        validate_session7_additive_scope(
            changed_paths=[audit_path[0], log_relative],
            report_hash_before=report_hash_before,
            report_hash_after=report_hash_after,
            config=config,
        )
        _validate_audit_final_git_state(
            git_runner=git_runner,
            allowed_staged_paths=[candidates[0].candidate_path, log_candidate],
            project_root=Path(authority["project_root"]),
            required_paths=authority["required_paths"],
        )
        require_clean_promotion_state(
            [Path(authority["project_root"]) / audit_path[0]],
            Path(authority["project_root"]) / log_relative,
        )
    except Exception:
        candidates[0].candidate_path.unlink(missing_ok=True)
        log_candidate.unlink(missing_ok=True)
        raise
    return AuditPromotionBundle(candidates, log_candidate, log_target, log_validator)


def promote_integrity_audit(
    bundle: AuditPromotionBundle,
    authority: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Promote the audit JSON first and its validated completion log last."""
    if not isinstance(bundle, AuditPromotionBundle):
        raise TypeError("Audit promotion requires a validated AuditPromotionBundle.")
    output_paths, _ = audit_promotion_inventory(authority["config"])
    promote_candidates(
        bundle.candidates,
        expected_relative_paths=output_paths,
        log_candidate=bundle.log_candidate,
        log_target=bundle.log_target,
        log_validator=bundle.log_validator,
        project_root=Path(authority["project_root"]),
    )
    return bundle.candidates[0].target_path, bundle.log_target


def execute_audit(
    config_path: Path,
    freeze_commit: str,
    amendment_freeze_commit: str,
    *,
    project_root: Path | None = None,
    git_runner: Callable[..., str] = _git,
    invocation_clock: Callable[[], str] | None = None,
) -> tuple[Path, Path]:
    """Run authority, aggregate access, executed checks, staging, and log-last promotion."""
    root = (project_root or PROJECT_ROOT).resolve()
    authority = validate_audit_authority(
        config_path,
        freeze_commit,
        amendment_freeze_commit,
        git_runner=git_runner,
        project_root=root,
    )
    config = authority["config"]
    privacy_source = root / config["privacy"]["privacy_reference_loader"]["source"]
    privacy_references = load_privacy_references(privacy_source, config, authorized=True)
    prior_inputs = load_prior_audit_inputs(authority)
    generated_at_utc = (
        invocation_clock or (lambda: datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))
    )()
    if not is_utc_iso8601_z(generated_at_utc):
        raise ValueError("Audit invocation clock did not return an exact UTC timestamp.")
    bundle = prepare_audit_promotion(
        authority=authority,
        prior_inputs=prior_inputs,
        privacy_references=privacy_references,
        generated_at_utc=generated_at_utc,
        git_runner=git_runner,
    )
    return promote_integrity_audit(bundle, authority)
