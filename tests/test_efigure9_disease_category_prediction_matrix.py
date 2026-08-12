"""Synthetic tests for the eFigure 9 disease-category prediction matrix."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pandas as pd
import pytest
import yaml


EXPECTED_COLUMNS = [
    "disease_category_code",
    "disease_category_label",
    "true_class",
    "predicted_class",
    "quadrant",
    "count",
    "disease_category_total",
    "within_disease_category_proportion",
]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_CONFIG_RELATIVE_PATH = Path("configs/efigure9_disease_category_prediction_matrix.yaml")


def _create_synthetic_contract_project(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    """Create a synthetic project tree containing the exact frozen contract bytes."""
    synthetic_root = tmp_path / "synthetic_project"
    config_path = synthetic_root / TASK_CONFIG_RELATIVE_PATH
    config_bytes = (PROJECT_ROOT / TASK_CONFIG_RELATIVE_PATH).read_bytes()
    payload = yaml.safe_load(config_bytes)
    assert isinstance(payload, dict)

    required_files = [*payload["sources"], *payload["authorized_task_paths"]]
    for relative_path in required_files:
        path = synthetic_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path != config_path:
            path.write_bytes(b"synthetic fixture\n")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_bytes(config_bytes)
    (synthetic_root / "outputs" / "analysis").mkdir(parents=True, exist_ok=True)
    return synthetic_root, config_path, payload


def _write_mutated_contract(config_path: Path, payload: dict[str, Any]) -> None:
    """Write one deliberately non-frozen synthetic contract variant."""
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _mutated_contract(payload: dict[str, Any], mutation: str) -> dict[str, Any]:
    """Return one independently specified invalid semantic contract variant."""
    altered = copy.deepcopy(payload)
    first_source = "configs/modeling_global.yaml"
    if mutation == "empty_sources":
        altered["sources"] = {}
    elif mutation == "source_removed":
        altered["sources"].pop(first_source)
    elif mutation == "source_added":
        altered["sources"]["synthetic/extra.txt"] = "0" * 64
    elif mutation == "source_modified":
        altered["sources"][first_source] = "0" * 64
    elif mutation == "escaping_output":
        altered["outputs"]["final_directory"] = "../escaped_output"
    elif mutation == "escaping_authorized_path":
        altered["authorized_task_paths"][1] = "../escaped_source.py"
    elif mutation == "substituted_authorized_path":
        altered["authorized_task_paths"][1] = "src/analysis/substitute.py"
    elif mutation == "reconciliation":
        altered["reconciliation"]["total"] = 196
    elif mutation == "serialization":
        altered["serialization"]["csv_float_format"] = "%.6f"
    elif mutation == "execution_command":
        altered["execution_command"] = "python substitute.py"
    elif mutation == "planning_reference_head":
        altered["planning_reference_head"] = "0" * 40
    elif mutation == "score_space":
        altered["score"]["space"] = "p_cal"
    elif mutation == "threshold_rule":
        altered["score"]["threshold_rule"] = "p_raw > threshold_p_raw"
    elif mutation == "unexpected_top_level_key":
        altered["unexpected"] = "value"
    elif mutation == "unexpected_nested_key":
        altered["model"]["unexpected"] = "value"
    elif mutation == "missing_key":
        altered.pop("serialization")
    elif mutation == "bool_for_int":
        altered["evaluation_set"]["expected_size"] = True
    else:
        raise ValueError(f"Unknown synthetic mutation: {mutation}.")
    return altered


@pytest.fixture
def synthetic_contract_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, Path, Path, dict[str, Any]]:
    """Provide the loader with a complete synthetic project root."""
    analysis = _import_analysis()
    synthetic_root, config_path, payload = _create_synthetic_contract_project(tmp_path)
    monkeypatch.setattr(analysis, "PROJECT_ROOT", synthetic_root)
    return analysis, synthetic_root, config_path, payload


def test_exact_frozen_contract_is_accepted_at_fixed_synthetic_location(
    synthetic_contract_project: tuple[Any, Path, Path, dict[str, Any]],
) -> None:
    """Rejecting the exact frozen bytes at the fixed contained location must fail."""
    analysis, _, config_path, expected_payload = synthetic_contract_project

    assert analysis._load_contract(config_path) == expected_payload


@pytest.mark.parametrize(
    "mutation",
    [
        "empty_sources",
        "source_removed",
        "source_added",
        "source_modified",
        "escaping_output",
        "escaping_authorized_path",
        "substituted_authorized_path",
        "reconciliation",
        "serialization",
        "execution_command",
        "planning_reference_head",
        "score_space",
        "threshold_rule",
        "unexpected_top_level_key",
        "unexpected_nested_key",
        "missing_key",
        "bool_for_int",
    ],
)
def test_loader_rejects_every_modified_contract_before_use(
    synthetic_contract_project: tuple[Any, Path, Path, dict[str, Any]],
    mutation: str,
) -> None:
    """Any byte or semantic contract mutation must fail before returning a payload."""
    analysis, _, config_path, payload = synthetic_contract_project
    _write_mutated_contract(config_path, _mutated_contract(payload, mutation))

    with pytest.raises(ValueError, match="configuration") as exc_info:
        analysis._load_contract(config_path)

    assert str(config_path.parent.parent) not in str(exc_info.value)


@pytest.mark.parametrize(
    "mutation",
    [
        "empty_sources",
        "source_removed",
        "source_added",
        "source_modified",
        "escaping_output",
        "escaping_authorized_path",
        "substituted_authorized_path",
        "reconciliation",
        "serialization",
        "execution_command",
        "planning_reference_head",
        "score_space",
        "threshold_rule",
        "unexpected_top_level_key",
        "unexpected_nested_key",
        "missing_key",
        "bool_for_int",
    ],
)
def test_semantic_defense_rejects_every_contract_mutation(
    synthetic_contract_project: tuple[Any, Path, Path, dict[str, Any]],
    mutation: str,
) -> None:
    """Semantic validation must independently reject mutations after identity verification."""
    analysis, _, _, payload = synthetic_contract_project

    with pytest.raises(ValueError, match="configuration"):
        analysis._validate_contract_payload(_mutated_contract(payload, mutation))


def test_identical_contract_bytes_at_wrong_path_are_rejected(
    synthetic_contract_project: tuple[Any, Path, Path, dict[str, Any]],
) -> None:
    """The correct bytes must not authorize an alternate configuration location."""
    analysis, synthetic_root, config_path, _ = synthetic_contract_project
    alternate_path = synthetic_root / "configs" / "alternate.yaml"
    alternate_path.write_bytes(config_path.read_bytes())

    with pytest.raises(ValueError, match="configuration") as exc_info:
        analysis._load_contract(alternate_path)

    assert str(synthetic_root) not in str(exc_info.value)


def test_configuration_path_with_parent_traversal_is_rejected(
    synthetic_contract_project: tuple[Any, Path, Path, dict[str, Any]],
) -> None:
    """Lexical parent traversal must fail even when it resolves to the fixed file."""
    analysis, synthetic_root, _, _ = synthetic_contract_project
    traversing_path = synthetic_root / "configs" / "nested" / ".." / TASK_CONFIG_RELATIVE_PATH.name

    with pytest.raises(ValueError, match="configuration"):
        analysis._load_contract(traversing_path)


def test_configuration_symlink_escaping_project_is_rejected_without_target_disclosure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fixed-location config symlink must not resolve beyond the project root."""
    analysis = _import_analysis()
    synthetic_root, config_path, _ = _create_synthetic_contract_project(tmp_path)
    outside_path = tmp_path / "outside_contract.yaml"
    outside_path.write_bytes(config_path.read_bytes())
    config_path.unlink()
    config_path.symlink_to(outside_path)
    monkeypatch.setattr(analysis, "PROJECT_ROOT", synthetic_root)

    with pytest.raises(ValueError, match="configuration") as exc_info:
        analysis._load_contract(config_path)

    assert str(outside_path) not in str(exc_info.value)


def test_source_symlink_escaping_project_is_rejected_without_target_disclosure(
    synthetic_contract_project: tuple[Any, Path, Path, dict[str, Any]],
    tmp_path: Path,
) -> None:
    """A configured source symlink must not resolve beyond the project root."""
    analysis, synthetic_root, config_path, payload = synthetic_contract_project
    source_path = synthetic_root / next(iter(payload["sources"]))
    outside_path = tmp_path / "outside_source.txt"
    outside_path.write_text("outside\n", encoding="utf-8")
    source_path.unlink()
    source_path.symlink_to(outside_path)

    with pytest.raises(ValueError, match="configuration") as exc_info:
        analysis._load_contract(config_path)

    assert str(outside_path) not in str(exc_info.value)


def test_invalid_contract_is_rejected_before_configured_paths_are_accessed(
    synthetic_contract_project: tuple[Any, Path, Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No downstream source, output, or Git access may precede complete validation."""
    analysis, synthetic_root, config_path, payload = synthetic_contract_project
    _write_mutated_contract(config_path, _mutated_contract(payload, "empty_sources"))
    downstream_calls: list[str] = []

    def forbidden_downstream_call(*args: Any, **kwargs: Any) -> Any:
        downstream_calls.append("called")
        raise AssertionError("Downstream access occurred before contract rejection.")

    monkeypatch.setattr(analysis, "_git_head", forbidden_downstream_call)
    monkeypatch.setattr(analysis, "verify_source_hashes", forbidden_downstream_call)

    with pytest.raises(ValueError, match="configuration"):
        analysis.run_aggregation(config_path=config_path, project_root=synthetic_root)

    assert downstream_calls == []


def _import_analysis() -> Any:
    """Import lazily so the required initial TDD failure is explicit."""
    try:
        return importlib.import_module("src.analysis.efigure9_disease_category_prediction_matrix")
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.fail(f"eFigure 9 implementation module is missing: {exc}", pytrace=False)


def _synthetic_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return aligned, deidentified prediction and disease-category fixtures."""
    sample_ids = list(range(1001, 1013))
    disease_categories = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    p_raw = [0.20, 0.80, 0.10, 0.90, 0.30, 0.70, 0.20, 0.80, 0.10, 0.90, 0.30, 0.70]
    y_true = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    y_pred = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    quadrant = ["TN", "FP", "TN", "FP", "TN", "FP", "FN", "TP", "FN", "TP", "FN", "TP"]
    predictions = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "y_true": y_true,
            "p_raw": p_raw,
            "p_cal": p_raw,
            "y_pred": y_pred,
            "quadrant": quadrant,
        }
    )
    metadata = pd.DataFrame({"sample_id": sample_ids, "disease_category": disease_categories})
    return predictions, metadata


def _build_result() -> Any:
    """Build the standard synthetic aggregation result."""
    analysis = _import_analysis()
    predictions, metadata = _synthetic_frames()
    return analysis.build_prediction_matrix(
        prediction_frame=predictions,
        metadata_frame=metadata,
        threshold_p_raw=0.5,
        expected_row_count=12,
    )


def _audit_payload(result: Any) -> dict[str, Any]:
    """Build a deterministic synthetic audit payload."""
    analysis = _import_analysis()
    return analysis.build_audit_payload(
        result=result,
        analysis_id="efigure9_disease_category_prediction_matrix_v1",
        threshold_p_raw=0.5,
        expected_evaluation_set_size=12,
    )


def _manifest_payload() -> dict[str, Any]:
    """Return a minimal aggregate-safe manifest for publication tests."""
    return {
        "analysis_id": "efigure9_disease_category_prediction_matrix_v1",
        "execution_command": "synthetic-test-command",
        "final_output_inventory": [
            "audit.json",
            "disease_category_prediction_matrix.csv",
            "manifest.json",
        ],
        "git_mutation_performed": False,
    }


def test_complete_cartesian_product_retains_zero_cells_in_fixed_order() -> None:
    """Dropping an unobserved cell or changing canonical ordering must fail."""
    analysis = _import_analysis()
    predictions, metadata = _synthetic_frames()
    predictions = predictions.iloc[::2].reset_index(drop=True)
    metadata = metadata.loc[metadata["sample_id"].isin(predictions["sample_id"])].reset_index(drop=True)

    result = analysis.build_prediction_matrix(
        prediction_frame=predictions,
        metadata_frame=metadata,
        threshold_p_raw=0.5,
        expected_row_count=6,
    )

    assert result.matrix[["disease_category_code", "predicted_class"]].values.tolist() == [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [2, 0],
        [2, 1],
        [3, 0],
        [3, 1],
        [4, 0],
        [4, 1],
        [5, 0],
        [5, 1],
    ]
    assert result.matrix["count"].tolist() == [1, 0] * 6


def test_endpoint_quadrant_counts_and_row_proportions_are_exact() -> None:
    """Wrong endpoint, quadrant, count, or row normalization logic must fail."""
    result = _build_result()

    assert result.matrix.columns.tolist() == EXPECTED_COLUMNS
    assert result.matrix["disease_category_label"].tolist() == [
        "HC",
        "HC",
        "BGL",
        "BGL",
        "LGIN",
        "LGIN",
        "HGIN",
        "HGIN",
        "EGC",
        "EGC",
        "AGC",
        "AGC",
    ]
    assert result.matrix["true_class"].tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    assert result.matrix["quadrant"].tolist() == [
        "TN",
        "FP",
        "TN",
        "FP",
        "TN",
        "FP",
        "FN",
        "TP",
        "FN",
        "TP",
        "FN",
        "TP",
    ]
    assert result.matrix["count"].tolist() == [1] * 12
    assert result.matrix["disease_category_total"].tolist() == [2] * 12
    assert result.matrix["within_disease_category_proportion"].tolist() == [0.5] * 12
    assert result.calculated_true_class_totals == {"0": 6, "1": 6}
    assert result.calculated_predicted_class_totals == {"0": 6, "1": 6}
    assert result.calculated_quadrant_totals == {"TN": 3, "FP": 3, "FN": 3, "TP": 3}


@pytest.mark.parametrize("frame_name", ["prediction", "metadata"])
def test_duplicate_participant_identifiers_are_rejected(frame_name: str) -> None:
    """Allowing a duplicated identifier in either source must fail closed."""
    analysis = _import_analysis()
    predictions, metadata = _synthetic_frames()
    target = predictions if frame_name == "prediction" else metadata
    target.loc[1, "sample_id"] = target.loc[0, "sample_id"]

    with pytest.raises(ValueError, match="identifiers must be nonmissing and unique"):
        analysis.build_prediction_matrix(predictions, metadata, 0.5, 12)


def test_misaligned_participant_identifier_sets_are_rejected() -> None:
    """Allowing unequal source identifier sets must fail closed without listing IDs."""
    analysis = _import_analysis()
    predictions, metadata = _synthetic_frames()
    metadata.loc[0, "sample_id"] = 9001

    with pytest.raises(ValueError, match="identifier sets must match exactly") as exc_info:
        analysis.build_prediction_matrix(predictions, metadata, 0.5, 12)
    assert "1001" not in str(exc_info.value)
    assert "9001" not in str(exc_info.value)


@pytest.mark.parametrize(
    ("column", "row", "replacement", "message"),
    [
        ("y_true", 0, 1, "true class"),
        ("y_pred", 0, 1, "threshold-derived prediction"),
        ("quadrant", 0, "FP", "quadrant"),
        ("p_raw", 0, 0.6, "threshold-derived prediction"),
    ],
)
def test_inconsistent_stored_or_threshold_derived_values_are_rejected(
    column: str,
    row: int,
    replacement: Any,
    message: str,
) -> None:
    """Any disagreement with the frozen recomputation must fail closed."""
    analysis = _import_analysis()
    predictions, metadata = _synthetic_frames()
    predictions.loc[row, column] = replacement

    with pytest.raises(ValueError, match=message):
        analysis.build_prediction_matrix(predictions, metadata, 0.5, 12)


def test_output_schema_excludes_participant_fields_and_prediction_display_wording() -> None:
    """Participant fields and future display terminology must never serialize."""
    analysis = _import_analysis()
    result = _build_result()
    csv_bytes = analysis.serialize_matrix_csv(result.matrix)
    audit_bytes = analysis.serialize_json(_audit_payload(result))
    csv_text = csv_bytes.decode("utf-8").lower()
    combined = (csv_bytes + audit_bytes).decode("utf-8").lower()

    assert result.matrix.columns.tolist() == EXPECTED_COLUMNS
    for forbidden_csv_field in ("sample_id", "p_raw", "p_cal"):
        assert forbidden_csv_field not in csv_text
    for forbidden in (
        "low risk",
        "high risk",
        "risk group",
        "risk stratum",
        "predicted negative",
        "predicted positive",
        "model negative",
        "model positive",
    ):
        assert forbidden not in combined


def test_existing_final_target_is_never_overwritten(tmp_path: Path) -> None:
    """Replacing or merging an existing final directory must fail."""
    analysis = _import_analysis()
    result = _build_result()
    audit = _audit_payload(result)
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    sentinel = final_dir / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        analysis.publish_validated_package(
            first_matrix=result.matrix,
            first_audit=audit,
            second_matrix=result.matrix.copy(deep=True),
            second_audit=copy.deepcopy(audit),
            manifest_builder=_manifest_payload,
            output_root=tmp_path,
            final_dir=final_dir,
        )

    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_csv_and_audit_serialization_are_deterministic() -> None:
    """Nondeterministic row, float, key, or newline serialization must fail."""
    analysis = _import_analysis()
    first = _build_result()
    second = _build_result()

    first_csv = analysis.serialize_matrix_csv(first.matrix)
    second_csv = analysis.serialize_matrix_csv(second.matrix)
    first_audit = analysis.serialize_json(_audit_payload(first))
    second_audit = analysis.serialize_json(_audit_payload(second))

    assert first_csv == second_csv
    assert first_audit == second_audit
    assert first_csv.endswith(b"\n") and b"\r\n" not in first_csv
    assert first_audit.endswith(b"\n")
    assert hashlib.sha256(first_csv).hexdigest() == hashlib.sha256(second_csv).hexdigest()
    assert list(json.loads(first_audit).keys()) == sorted(json.loads(first_audit).keys())


def test_relevant_source_hashes_are_checked_and_changes_are_detected(tmp_path: Path) -> None:
    """Skipping a relevant-source immutability recheck must fail."""
    analysis = _import_analysis()
    source = tmp_path / "source.txt"
    source.write_text("frozen\n", encoding="utf-8")
    expected = hashlib.sha256(b"frozen\n").hexdigest()

    assert analysis.verify_source_hashes(tmp_path, {"source.txt": expected}) == {"source.txt": expected}
    source.write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source hash mismatch") as exc_info:
        analysis.verify_source_hashes(tmp_path, {"source.txt": expected})
    assert "changed" not in str(exc_info.value)


def test_validation_failure_leaves_no_final_directory_or_temporary_package(tmp_path: Path) -> None:
    """A pre-promotion validation failure must leave no apparent artifact."""
    analysis = _import_analysis()
    result = _build_result()
    invalid_matrix = result.matrix.iloc[:-1].copy()
    final_dir = tmp_path / "final"

    with pytest.raises(ValueError, match="exactly 12 rows"):
        analysis.publish_validated_package(
            first_matrix=invalid_matrix,
            first_audit=_audit_payload(result),
            second_matrix=invalid_matrix.copy(deep=True),
            second_audit=_audit_payload(result),
            manifest_builder=_manifest_payload,
            output_root=tmp_path,
            final_dir=final_dir,
        )

    assert not final_dir.exists()
    assert not list(tmp_path.glob(".efigure9_disease_category_prediction_matrix_tmp_*"))


def test_valid_package_has_exact_inventory_and_readback_contract(tmp_path: Path) -> None:
    """Promotion must publish exactly the validated CSV, audit, and manifest."""
    analysis = _import_analysis()
    first = _build_result()
    second = _build_result()
    first_audit = _audit_payload(first)
    second_audit = _audit_payload(second)
    final_dir = tmp_path / "final"

    result = analysis.publish_validated_package(
        first_matrix=first.matrix,
        first_audit=first_audit,
        second_matrix=second.matrix,
        second_audit=second_audit,
        manifest_builder=_manifest_payload,
        output_root=tmp_path,
        final_dir=final_dir,
    )

    assert sorted(path.name for path in final_dir.iterdir()) == [
        "audit.json",
        "disease_category_prediction_matrix.csv",
        "manifest.json",
    ]
    readback = pd.read_csv(final_dir / "disease_category_prediction_matrix.csv")
    assert readback.columns.tolist() == EXPECTED_COLUMNS
    assert len(readback) == 12
    assert result["csv_sha256"] == hashlib.sha256(
        (final_dir / "disease_category_prediction_matrix.csv").read_bytes()
    ).hexdigest()
    assert not list(tmp_path.glob(".efigure9_disease_category_prediction_matrix_tmp_*"))


def test_runner_is_importable_by_direct_script_execution(tmp_path: Path) -> None:
    """The prescribed direct-script command must resolve project imports."""
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_efigure9_disease_category_prediction_matrix.py"),
            "--help",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env={"PYTHONDONTWRITEBYTECODE": "1"},
    )

    assert completed.returncode == 0, completed.stderr
    assert "--config" in completed.stdout
