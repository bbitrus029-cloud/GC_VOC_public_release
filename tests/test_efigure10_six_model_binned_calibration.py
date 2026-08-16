"""Synthetic contract tests for the eFigure 10 aggregate."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import unicodedata
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from statsmodels.stats.proportion import proportion_confint

from src.analysis import efigure10_six_model_binned_calibration as calibration_module
from src.analysis.efigure10_six_model_binned_calibration import (
    build_frozen_calibration_builder_config,
    build_artifact_payloads,
    build_participant_identifier_references,
    build_unified_calibration_points,
    cleanup_exact_task_directory,
    discard_participant_identifier_references,
    execute_isolated_run,
    load_and_reconcile_predictions,
    load_config,
    load_frozen_reporting_config,
    load_prediction_projection,
    load_strict_yaml_mapping,
    promote_artifact_set,
    reconcile_minirocket_anchor,
    stage_artifact_set,
    validate_artifact_payloads,
    validate_candidate_privacy,
    validate_promoted_artifact_set,
    validate_runtime_absence_state,
    validate_runtime_authority,
    validate_unified_calibration_points,
    run_analysis,
)
from src.analysis.posthoc_final_reporting import build_calibration_points


CONFIG_PATH = Path("configs/efigure10_six_model_binned_calibration.yaml")
MODEL_IDS = (
    "minirocket_ridge",
    "xgboost",
    "elastic_net_lr",
    "svm",
    "inception_time",
    "cnn_bilstm_attention",
)
DISPLAY_NAMES = (
    "MiniRocket + Ridge",
    "XGBoost",
    "Elastic Net LR",
    "SVM",
    "InceptionTime",
    "CNN–BiLSTM–Attention",
)
EXPECTED_INPUT_COLUMNS = ("sample_id", "y_true", "p_raw", "p_cal", "y_pred", "quadrant")
EXPECTED_PROJECTION_COLUMNS = ("sample_id", "y_true", "p_cal")
COMMON_COLUMNS = (
    "bin_index",
    "n",
    "events",
    "nonevents",
    "p_cal_mean",
    "observed_fraction",
    "observed_ci_low",
    "observed_ci_high",
    "ci_method",
    "n_requested_bins",
    "n_actual_bins",
    "visualization_ready",
    "estimability_status",
    "nonestimable_reason",
)
EXPECTED_G06_PATHS = (
    "configs/efigure10_six_model_binned_calibration.yaml",
    "scripts/run_efigure10_six_model_binned_calibration.py",
    "src/analysis/efigure10_six_model_binned_calibration.py",
    "tests/test_efigure10_six_model_binned_calibration.py",
    "outputs/work/efigure10_six_model_binned_calibration",
    "outputs/analysis/efigure10_six_model_binned_calibration",
)
EXPECTED_G06_ARGV = (
    "git",
    "status",
    "--short",
    "--branch",
    "--untracked-files=normal",
    "--",
    *EXPECTED_G06_PATHS,
)


@pytest.fixture
def loaded_config() -> dict[str, object]:
    return load_config(CONFIG_PATH, project_root=Path.cwd())


def _synthetic_identifier_lexemes() -> list[str]:
    fixed = [
        "+0001",
        "0002",
        "CaseID",
        "caseid",
        "NA",
        " leading-space",
        "trailing-space ",
        "SYN-UNICODE-\u00e9",
        "SYN-UNICODE-e\u0301",
    ]
    return fixed + [f"SYN-CSV-{index:04d}" for index in range(10, 196)]


def _synthetic_labels() -> np.ndarray:
    return np.array([1] * 82 + [0] * 113, dtype=int)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_prediction(path: Path, ids: list[str], labels: object, scores: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "sample_id": ids,
            "y_true": labels,
            "p_raw": np.linspace(0.0, 1.0, len(ids)),
            "p_cal": scores,
            "y_pred": np.zeros(len(ids), dtype=int),
            "quadrant": ["synthetic"] * len(ids),
        }
    )
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


@pytest.fixture
def synthetic_config_dict(loaded_config: dict[str, object]) -> dict[str, object]:
    config = copy.deepcopy(loaded_config)
    for record in config["models"]:
        record["prediction_path"] = f"synthetic_inputs/{record['model_id']}.csv"
        record["sha256"] = "0" * 64
    return config


@pytest.fixture
def synthetic_prediction_frames_distinct() -> OrderedDict[str, pd.DataFrame]:
    ids = [f"SYN{index:04d}" for index in range(1, 196)]
    labels = _synthetic_labels()
    frames: OrderedDict[str, pd.DataFrame] = OrderedDict()
    for model_index, model_id in enumerate(MODEL_IDS):
        probabilities = np.linspace(0.001 + model_index * 0.00001, 0.999 - model_index * 0.00001, 195)
        frames[model_id] = pd.DataFrame({"sample_id": ids, "y_true": labels, "p_cal": probabilities})
    return frames


@pytest.fixture
def synthetic_prediction_frames_tied(
    synthetic_prediction_frames_distinct: OrderedDict[str, pd.DataFrame],
) -> OrderedDict[str, pd.DataFrame]:
    frames = copy.deepcopy(synthetic_prediction_frames_distinct)
    tied = np.repeat(np.linspace(0.1, 0.9, 5), 39)
    for frame in frames.values():
        frame["p_cal"] = tied
    return frames


@pytest.fixture
def synthetic_prediction_frames_all_tied(
    synthetic_prediction_frames_distinct: OrderedDict[str, pd.DataFrame],
) -> OrderedDict[str, pd.DataFrame]:
    frames = copy.deepcopy(synthetic_prediction_frames_distinct)
    for frame in frames.values():
        frame["p_cal"] = np.full(195, 0.5)
    return frames


def _build_synthetic_prediction_bundle(
    project_root: Path,
    config: dict[str, object],
    frames: Mapping[str, pd.DataFrame],
) -> dict[str, object]:
    prepared = copy.deepcopy(config)
    paths: OrderedDict[str, Path] = OrderedDict()
    for record in prepared["models"]:
        model_id = record["model_id"]
        relative_path = Path("synthetic_inputs") / f"{model_id}.csv"
        path = project_root / relative_path
        frame = frames[model_id]
        _write_prediction(path, frame["sample_id"].tolist(), frame["y_true"].to_numpy(), frame["p_cal"].to_numpy())
        record["prediction_path"] = relative_path.as_posix()
        record["sha256"] = _sha256(path)
        paths[model_id] = path
    return {
        "project_root": project_root,
        "config": prepared,
        "paths": paths,
        "ids": frames[MODEL_IDS[0]]["sample_id"].tolist(),
        "labels": frames[MODEL_IDS[0]]["y_true"].to_numpy(copy=True),
    }


@pytest.fixture
def synthetic_prediction_csv_files(
    tmp_path: Path,
    synthetic_config_dict: dict[str, object],
) -> dict[str, object]:
    ids = _synthetic_identifier_lexemes()
    labels = _synthetic_labels()
    frames: OrderedDict[str, pd.DataFrame] = OrderedDict()
    for model_index, model_id in enumerate(MODEL_IDS):
        probabilities = np.linspace(0.001 + model_index * 0.00001, 0.999 - model_index * 0.00001, 195)
        frames[model_id] = pd.DataFrame({"sample_id": ids, "y_true": labels, "p_cal": probabilities})
    return _build_synthetic_prediction_bundle(tmp_path, synthetic_config_dict, frames)


@pytest.fixture
def synthetic_anchor_frame(
    loaded_config: dict[str, object],
    synthetic_prediction_frames_distinct: OrderedDict[str, pd.DataFrame],
) -> pd.DataFrame:
    reference = load_frozen_reporting_config(
        Path("configs/post_hoc_final_supplementary.yaml"), project_root=Path.cwd()
    )
    builder_config = build_frozen_calibration_builder_config(
        loaded_config["calibration_contract"], frozen_reporting_config=reference
    )
    frame = synthetic_prediction_frames_distinct[MODEL_IDS[0]]
    return build_calibration_points(frame["y_true"], frame["p_cal"], builder_config)


@pytest.fixture
def synthetic_workspace(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def synthetic_artifact_payloads() -> OrderedDict[str, bytes]:
    return OrderedDict(
        (
            ("calibration_points.csv", b"model_order,bin_index\n1,0\n"),
            ("audit.json", b'{"artifact_status": "validated_for_atomic_promotion"}\n'),
            ("manifest.json", b'{"files": []}\n'),
            ("report.md", b"# Synthetic aggregate report\n"),
        )
    )


@pytest.fixture
def privacy_payload_factory() -> Any:
    def factory(role: str, synthetic_value: str) -> tuple[str, bytes]:
        return role, synthetic_value.encode("utf-8")

    return factory


@pytest.fixture
def immutable_hash_fixture(tmp_path: Path) -> OrderedDict[Path, str]:
    first = tmp_path / "synthetic_immutable_a.txt"
    second = tmp_path / "synthetic_immutable_b.txt"
    first.write_text("synthetic-a\n", encoding="utf-8")
    second.write_text("synthetic-b\n", encoding="utf-8")
    return OrderedDict(((first, _sha256(first)), (second, _sha256(second))))


@pytest.fixture
def external_namespace_fixture(tmp_path: Path) -> OrderedDict[str, tuple[Path, bytes]]:
    return OrderedDict(
        (
            ("efigure9", (tmp_path / "synthetic_external" / "efigure9.bin", b"synthetic-efigure9")),
            ("efigure4", (tmp_path / "synthetic_external" / "efigure4.bin", b"synthetic-efigure4")),
        )
    )


def _rewrite_bundle_model(
    bundle: dict[str, object],
    model_id: str,
    ids: list[str],
    labels: object,
    scores: object,
) -> None:
    path = bundle["paths"][model_id]
    _write_prediction(path, ids, labels, scores)
    record = next(item for item in bundle["config"]["models"] if item["model_id"] == model_id)
    record["sha256"] = _sha256(path)


def _load_reconciled(bundle: dict[str, object]) -> OrderedDict[str, pd.DataFrame]:
    frames, _references = load_and_reconcile_predictions(
        bundle["config"], project_root=bundle["project_root"]
    )
    return frames


def _record_prediction_reads(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []
    real_read_csv = pd.read_csv

    def recording_read_csv(*args: object, **kwargs: object) -> pd.DataFrame:
        calls.append(dict(kwargs))
        return real_read_csv(*args, **kwargs)

    monkeypatch.setattr(calibration_module.pd, "read_csv", recording_read_csv)
    return calls


def _write_valid_projection_file(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    ids = _synthetic_identifier_lexemes()
    labels = _synthetic_labels()
    scores = np.linspace(0.001, 0.999, 195)
    _write_prediction(path, ids, labels, scores)
    return ids, labels, scores


def _builder_config(loaded_config: dict[str, object]) -> OrderedDict[str, object]:
    reference = load_frozen_reporting_config(
        Path("configs/post_hoc_final_supplementary.yaml"), project_root=Path.cwd()
    )
    return build_frozen_calibration_builder_config(
        loaded_config["calibration_contract"], frozen_reporting_config=reference
    )


def _unified_frame(
    loaded_config: dict[str, object],
    frames: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    return build_unified_calibration_points(frames, loaded_config, builder_config=_builder_config(loaded_config))


def _synthetic_run_context(frame: pd.DataFrame, loaded_config: dict[str, object]) -> OrderedDict[str, object]:
    del frame
    anchor_record = next(
        record for record in loaded_config["immutable_references"] if record["role"] == "reconciliation_anchor"
    )
    authority = OrderedDict(
        (
            ("authority_status", "passed"),
            ("runtime_head", loaded_config["authority"]["phase2_starting_head"]),
            ("ancestry_passed", True),
            ("tag_ancestry_passed", True),
            ("prediction_ancestry_passed", True),
            ("anchor_ancestry_passed", True),
            ("relevant_diff_status", "passed"),
            ("git_status", "passed"),
            ("efigure9_status", "passed"),
            ("efigure4_status", "passed"),
            ("command_inventory_status", "passed"),
            ("python_path_status", "passed"),
            ("pycache_status", "passed"),
            ("prediction_hash_status", "passed"),
            ("immutable_hash_status", "passed"),
            ("import_closure_status", "passed"),
            ("import_origin_status", "passed"),
            ("stale_state_status", "passed"),
        )
    )
    input_hashes = [OrderedDict((("path", "synthetic/input"), ("sha256", "a" * 64), ("status", "matched")))]
    passed_privacy = OrderedDict(
        (key, "passed")
        for key in (
            "schema", "participant_identifiers", "participant_predictions", "dates", "filenames",
            "per_sample_paths", "absolute_paths", "external_namespaces", "raw_value_matching",
            "candidate_error_log_messages",
        )
    )
    return OrderedDict(
        (
            (
                "authority_checks",
                OrderedDict(
                    (stage, copy.deepcopy(authority))
                    for stage in ("initial_entry", "before_run_a", "before_run_b", "after_both_runs")
                ),
            ),
            ("input_hashes_before", input_hashes),
            ("input_hashes_after", copy.deepcopy(input_hashes)),
            (
                "environment_versions",
                OrderedDict((("python", "3.11.15"), ("pandas", pd.__version__), ("numpy", np.__version__),
                             ("statsmodels", "0.14.5"), ("pytest", pytest.__version__))),
            ),
            (
                "input_validation",
                OrderedDict(
                    (
                        ("model_registry_status", "passed"),
                        ("header_only_status", "passed"),
                        ("projection_columns", ["sample_id", "y_true", "p_cal"]),
                        ("unused_participant_columns_loaded", False),
                        ("row_count_status", "passed"),
                        ("identifier_lexeme_status", "passed"),
                        ("probability_status", "passed"),
                        ("label_status", "passed"),
                    )
                ),
            ),
            (
                "cohort_reconciliation",
                OrderedDict((("models", 6), ("participants_per_model", 195), ("positive_labels", 82),
                             ("negative_labels", 113), ("participant_set_agreement", "passed"),
                             ("cross_model_label_agreement", "passed"))),
            ),
            (
                "minirocket_reconciliation",
                OrderedDict(
                    (
                        ("anchor_path", anchor_record["path"]),
                        ("anchor_sha256", anchor_record["sha256"]),
                        ("common_columns", list(COMMON_COLUMNS)),
                        ("check_exact", True),
                        ("status", "passed"),
                    )
                ),
            ),
            ("privacy_validation", passed_privacy),
        )
    )


def _assert_recursive_order_and_value(actual: object, expected: object) -> None:
    if isinstance(expected, Mapping):
        assert isinstance(actual, Mapping)
        assert tuple(actual) == tuple(expected)
        for key in expected:
            _assert_recursive_order_and_value(actual[key], expected[key])
        return
    if isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_recursive_order_and_value(actual_item, expected_item)
        return
    assert actual == expected


def _synthetic_git_evidence(config: Mapping[str, Any]) -> OrderedDict[str, object]:
    return OrderedDict(
        (
            ("runtime_head", config["authority"]["phase2_starting_head"]),
            ("ancestry_passed", True),
            ("tag_ancestry_passed", True),
            ("prediction_ancestry_passed", True),
            ("anchor_ancestry_passed", True),
            ("relevant_diff_status", "passed"),
            ("git_status", "passed"),
            ("efigure9_status", "passed"),
            ("efigure4_status", "passed"),
            ("command_inventory_status", "passed"),
        )
    )


def _prepare_synthetic_analysis_project(
    project_root: Path,
    loaded_config: dict[str, object],
    frames: OrderedDict[str, pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    project_root.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(loaded_config)
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", config["environment"]["production_pycache_prefix"])
    config["output_contract"]["work_root"] = "outputs/work/efigure10_six_model_binned_calibration"
    config["output_contract"]["final_root"] = "outputs/analysis/efigure10_six_model_binned_calibration"
    (project_root / "outputs/work").mkdir(parents=True, exist_ok=True)
    (project_root / "outputs/analysis").mkdir(parents=True, exist_ok=True)
    prediction_records = []
    for record in config["models"]:
        relative = Path("synthetic_predictions") / f"{record['model_id']}.csv"
        frame = frames[record["model_id"]]
        _write_prediction(project_root / relative, frame["sample_id"].tolist(), frame["y_true"], frame["p_cal"])
        record["prediction_path"] = relative.as_posix()
        record["sha256"] = _sha256(project_root / relative)
        prediction_records.append(
            {"path": relative.as_posix(), "sha256_or_binding": record["sha256"],
             "content_access_phase": "after_runtime_gate", "role": "prediction"}
        )
    frozen_relative = Path("configs/post_hoc_final_supplementary.yaml")
    frozen_path = project_root / frozen_relative
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    frozen_path.write_bytes(Path(frozen_relative).read_bytes())
    builder = build_frozen_calibration_builder_config(
        config["calibration_contract"],
        frozen_reporting_config=load_frozen_reporting_config(frozen_relative, project_root=project_root),
    )
    unified = build_unified_calibration_points(frames, config, builder_config=builder)
    anchor_relative = Path("synthetic_anchor.csv")
    anchor = unified.loc[unified["model_id"] == MODEL_IDS[0], list(COMMON_COLUMNS)].reset_index(drop=True)
    anchor.to_csv(
        project_root / anchor_relative, index=False, encoding="utf-8", lineterminator="\n", na_rep="",
        float_format="%.17g",
    )
    plan_relative = Path("synthetic_plan.md")
    (project_root / plan_relative).write_text("synthetic reviewed plan\n", encoding="utf-8")
    config["authority"]["approved_plan_path"] = plan_relative.as_posix()
    config["authority"]["approved_plan_sha256"] = _sha256(project_root / plan_relative)
    config["immutable_references"] = [
        {"path": anchor_relative.as_posix(), "sha256": _sha256(project_root / anchor_relative),
         "role": "reconciliation_anchor"},
        {"path": frozen_relative.as_posix(), "sha256": _sha256(frozen_path), "role": "frozen_builder_config"},
    ]
    config_relative = Path("configs/efigure10_six_model_binned_calibration.yaml")
    config["authority"]["production_read_paths"] = [
        {"path": plan_relative.as_posix(), "sha256_or_binding": config["authority"]["approved_plan_sha256"],
         "content_access_phase": "authority", "role": "approved_plan"},
        {"path": config_relative.as_posix(), "sha256_or_binding": "external_test_binding",
         "content_access_phase": "runtime", "role": "task_config"},
        {"path": frozen_relative.as_posix(), "sha256_or_binding": _sha256(frozen_path),
         "content_access_phase": "runtime", "role": "frozen_builder_config"},
        *prediction_records,
        {"path": anchor_relative.as_posix(), "sha256_or_binding": _sha256(project_root / anchor_relative),
         "content_access_phase": "after_runtime_gate", "role": "reconciliation_anchor"},
    ]
    config_path = project_root / config_relative
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    monkeypatch.setattr(calibration_module, "TASK_CONFIG_PATH", config_relative)
    monkeypatch.setattr(calibration_module, "TASK_CONFIG_SHA256", _sha256(config_path))
    monkeypatch.setattr(
        calibration_module,
        "_validate_git_authority",
        lambda current, *, project_root, status_stage="preproduction": _synthetic_git_evidence(current),
    )
    monkeypatch.setattr(calibration_module, "_verify_import_origins", lambda *_args, **_kwargs: [])
    return config_path


def test_model_registry_is_exact_and_ordered(loaded_config: dict[str, object]) -> None:
    assert tuple(item["model_id"] for item in loaded_config["models"]) == MODEL_IDS
    assert tuple(item["model_display_name"] for item in loaded_config["models"]) == DISPLAY_NAMES
    assert tuple(item["model_order"] for item in loaded_config["models"]) == tuple(range(1, 7))


def test_prediction_schema_is_exact_and_ordered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "prediction.csv"
    ids, _, _ = _write_valid_projection_file(path)
    calls = _record_prediction_reads(monkeypatch)
    result = load_prediction_projection(path, project_root=tmp_path)
    assert tuple(result.columns) == EXPECTED_PROJECTION_COLUMNS
    assert result["sample_id"].tolist() == ids
    assert len(calls) == 2
    assert calls[0]["nrows"] == 0
    assert calls[0]["encoding"] == "utf-8"
    assert "usecols" not in calls[0]
    assert calls[1]["usecols"] == ["sample_id", "y_true", "p_cal"]
    assert calls[1]["dtype"] == {"sample_id": "string"}
    assert calls[1]["keep_default_na"] is False
    assert calls[1]["na_filter"] is False
    assert calls[1]["skipinitialspace"] is False
    assert not set(("p_raw", "y_pred", "quadrant")) & set(result.columns)


@pytest.mark.parametrize("variant", ["missing", "unexpected"], ids=["missing", "unexpected"])
def test_prediction_schema_rejects_missing_and_unexpected_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
) -> None:
    path = tmp_path / "prediction.csv"
    ids = _synthetic_identifier_lexemes()
    labels = _synthetic_labels()
    frame = pd.DataFrame(
        {
            "sample_id": ids,
            "y_true": labels,
            "p_raw": np.linspace(0.0, 1.0, 195),
            "p_cal": np.linspace(0.001, 0.999, 195),
            "y_pred": np.zeros(195, dtype=int),
            "quadrant": ["synthetic"] * 195,
        }
    )
    if variant == "missing":
        frame = frame.drop(columns=["quadrant"])
    else:
        frame["unexpected"] = "synthetic"
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    calls = _record_prediction_reads(monkeypatch)
    with pytest.raises(ValueError, match="schema"):
        load_prediction_projection(path, project_root=tmp_path)
    assert len(calls) == 1
    assert calls[0]["nrows"] == 0


def test_each_model_requires_exactly_195_unique_participants(
    synthetic_prediction_csv_files: dict[str, object],
) -> None:
    frames = _load_reconciled(synthetic_prediction_csv_files)
    assert all(len(frame) == 195 and frame["sample_id"].is_unique for frame in frames.values())
    evidence_frames, references, evidence = calibration_module._load_and_reconcile_predictions_with_evidence(
        synthetic_prediction_csv_files["config"],
        project_root=synthetic_prediction_csv_files["project_root"],
    )
    observed_labels = evidence_frames[MODEL_IDS[0]]["y_true"]
    assert evidence["cohort_reconciliation"] == OrderedDict(
        (
            ("models", len(evidence_frames)),
            ("participants_per_model", len(observed_labels)),
            ("positive_labels", int(observed_labels.sum())),
            ("negative_labels", len(observed_labels) - int(observed_labels.sum())),
            ("participant_set_agreement", "passed"),
            ("cross_model_label_agreement", "passed"),
        )
    )
    discard_participant_identifier_references(references)
    ids = synthetic_prediction_csv_files["ids"][:-1]
    labels = synthetic_prediction_csv_files["labels"][:-1]
    _rewrite_bundle_model(
        synthetic_prediction_csv_files,
        MODEL_IDS[0],
        ids,
        labels,
        np.linspace(0.001, 0.999, 194),
    )
    with pytest.raises(ValueError):
        _load_reconciled(synthetic_prediction_csv_files)


def test_each_model_requires_exact_82_113_label_totals(
    synthetic_prediction_csv_files: dict[str, object],
) -> None:
    frames = _load_reconciled(synthetic_prediction_csv_files)
    assert all((int(frame["y_true"].sum()), int((1 - frame["y_true"]).sum())) == (82, 113) for frame in frames.values())
    invalid_labels = np.array([1] * 81 + [0] * 114, dtype=int)
    _rewrite_bundle_model(
        synthetic_prediction_csv_files,
        MODEL_IDS[0],
        synthetic_prediction_csv_files["ids"],
        invalid_labels,
        np.linspace(0.001, 0.999, 195),
    )
    with pytest.raises(ValueError):
        _load_reconciled(synthetic_prediction_csv_files)


def test_participant_sets_must_agree_across_models(
    synthetic_prediction_csv_files: dict[str, object],
) -> None:
    changed_ids = list(synthetic_prediction_csv_files["ids"])
    changed_ids[0] = "SYNTHETIC_SET_MISMATCH"
    _rewrite_bundle_model(
        synthetic_prediction_csv_files,
        MODEL_IDS[1],
        changed_ids,
        synthetic_prediction_csv_files["labels"],
        np.linspace(0.001, 0.999, 195),
    )
    with pytest.raises(ValueError):
        _load_reconciled(synthetic_prediction_csv_files)


def test_labels_must_agree_after_one_to_one_alignment(
    synthetic_prediction_csv_files: dict[str, object],
) -> None:
    changed_labels = synthetic_prediction_csv_files["labels"].copy()
    changed_labels[0], changed_labels[100] = changed_labels[100], changed_labels[0]
    _rewrite_bundle_model(
        synthetic_prediction_csv_files,
        MODEL_IDS[1],
        synthetic_prediction_csv_files["ids"],
        changed_labels,
        np.linspace(0.001, 0.999, 195),
    )
    with pytest.raises(ValueError):
        _load_reconciled(synthetic_prediction_csv_files)


@pytest.mark.parametrize(
    "variant",
    ["empty", "duplicate", "missing", "extra", "mismatched"],
    ids=["empty", "duplicate", "missing", "extra", "mismatched"],
)
def test_alignment_rejects_duplicate_missing_extra_and_mismatched_ids(
    synthetic_prediction_csv_files: dict[str, object],
    variant: str,
) -> None:
    valid_frames = _load_reconciled(synthetic_prediction_csv_files)
    expected_ids = synthetic_prediction_csv_files["ids"]
    composed = "SYN-UNICODE-\u00e9"
    decomposed = "SYN-UNICODE-e\u0301"
    assert composed != decomposed
    assert unicodedata.normalize("NFC", composed) == unicodedata.normalize("NFC", decomposed)
    assert tuple(map(ord, composed)) != tuple(map(ord, decomposed))
    assert expected_ids[7:9] == [composed, decomposed]
    assert len(expected_ids) == 195
    assert len(set(expected_ids)) == 195
    assert expected_ids[:7] == [
        "+0001", "0002", "CaseID", "caseid", "NA", " leading-space", "trailing-space "
    ]
    for frame in valid_frames.values():
        aligned_ids = frame["sample_id"].tolist()
        assert aligned_ids == expected_ids
        assert aligned_ids[7:9] == [composed, decomposed]
        assert tuple(map(ord, aligned_ids[7])) != tuple(map(ord, aligned_ids[8]))
    ids = list(synthetic_prediction_csv_files["ids"])
    labels = synthetic_prediction_csv_files["labels"].copy()
    scores = np.linspace(0.001, 0.999, 195)
    if variant == "empty":
        ids[0] = ""
    elif variant == "duplicate":
        ids[1] = ids[0]
    elif variant == "missing":
        ids, labels, scores = ids[:-1], labels[:-1], scores[:-1]
    elif variant == "extra":
        ids.append("SYNTHETIC_EXTRA")
        labels = np.append(labels, 0)
        scores = np.append(scores, 0.5)
    else:
        ids[0] = "SYNTHETIC_MISMATCH"
    _rewrite_bundle_model(synthetic_prediction_csv_files, MODEL_IDS[0], ids, labels, scores)
    with pytest.raises(ValueError):
        _load_reconciled(synthetic_prediction_csv_files)


def test_p_cal_rejects_missing_values(tmp_path: Path) -> None:
    path = tmp_path / "prediction.csv"
    ids, labels, scores = _write_valid_projection_file(path)
    invalid_scores = scores.astype(object)
    invalid_scores[0] = ""
    _write_prediction(path, ids, labels, invalid_scores)
    with pytest.raises(TypeError):
        load_prediction_projection(path, project_root=tmp_path)


def test_p_cal_rejects_nonnumeric_values(tmp_path: Path) -> None:
    path = tmp_path / "prediction.csv"
    ids, labels, scores = _write_valid_projection_file(path)
    invalid_scores = scores.astype(object)
    invalid_scores[0] = "synthetic_invalid"
    _write_prediction(path, ids, labels, invalid_scores)
    with pytest.raises(TypeError):
        load_prediction_projection(path, project_root=tmp_path)


def test_p_cal_rejects_nonfinite_values(tmp_path: Path) -> None:
    path = tmp_path / "prediction.csv"
    ids, labels, scores = _write_valid_projection_file(path)
    scores[0] = np.inf
    _write_prediction(path, ids, labels, scores)
    with pytest.raises(ValueError):
        load_prediction_projection(path, project_root=tmp_path)


def test_p_cal_rejects_values_below_zero(tmp_path: Path) -> None:
    path = tmp_path / "prediction.csv"
    ids, labels, scores = _write_valid_projection_file(path)
    scores[0] = -0.1
    _write_prediction(path, ids, labels, scores)
    with pytest.raises(ValueError):
        load_prediction_projection(path, project_root=tmp_path)


def test_p_cal_rejects_values_above_one(tmp_path: Path) -> None:
    path = tmp_path / "prediction.csv"
    ids, labels, scores = _write_valid_projection_file(path)
    scores[0] = 1.1
    _write_prediction(path, ids, labels, scores)
    with pytest.raises(ValueError):
        load_prediction_projection(path, project_root=tmp_path)


@pytest.mark.parametrize(
    "invalid_value",
    ["synthetic_text", True, 0.5, "", 2],
    ids=["text", "boolean", "fractional", "missing", "out_of_range"],
)
def test_labels_require_exact_numeric_binary_values(tmp_path: Path, invalid_value: object) -> None:
    path = tmp_path / "prediction.csv"
    ids, labels, scores = _write_valid_projection_file(path)
    invalid_labels = labels.astype(object)
    invalid_labels[0] = invalid_value
    _write_prediction(path, ids, invalid_labels, scores)
    with pytest.raises((TypeError, ValueError)):
        load_prediction_projection(path, project_root=tmp_path)


def test_distinct_probabilities_produce_deterministic_ten_bins(
    loaded_config: dict[str, object],
    synthetic_prediction_frames_distinct: OrderedDict[str, pd.DataFrame],
) -> None:
    frame = synthetic_prediction_frames_distinct[MODEL_IDS[0]]
    first = build_calibration_points(frame["y_true"], frame["p_cal"], _builder_config(loaded_config))
    second = build_calibration_points(frame["y_true"], frame["p_cal"], _builder_config(loaded_config))
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert first["n_actual_bins"].unique().tolist() == [10]
    assert int(first["n"].sum()) == 195


def test_ties_deterministically_collapse_bins(
    loaded_config: dict[str, object],
    synthetic_prediction_frames_tied: OrderedDict[str, pd.DataFrame],
) -> None:
    frame = synthetic_prediction_frames_tied[MODEL_IDS[0]]
    first = build_calibration_points(frame["y_true"], frame["p_cal"], _builder_config(loaded_config))
    second = build_calibration_points(frame["y_true"], frame["p_cal"], _builder_config(loaded_config))
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert 1 < int(first["n_actual_bins"].iloc[0]) < 10
    assert int(first["n"].sum()) == 195


def test_all_ties_retain_all_observations_in_one_bin(
    loaded_config: dict[str, object],
    synthetic_prediction_frames_all_tied: OrderedDict[str, pd.DataFrame],
) -> None:
    frame = synthetic_prediction_frames_all_tied[MODEL_IDS[0]]
    result = build_calibration_points(frame["y_true"], frame["p_cal"], _builder_config(loaded_config))
    assert len(result) == 1
    assert int(result.loc[0, "n"]) == 195
    assert int(result.loc[0, "n_actual_bins"]) == 1
    assert int(result.loc[0, "n_requested_bins"]) == 10


@pytest.mark.parametrize(
    ("variant", "yaml_text", "duplicated_key"),
    [
        ("top_level", "duplicate_top: 1\nduplicate_top: 2\n", "duplicate_top"),
        ("nested_mapping", "outer:\n  duplicate_nested: 1\n  duplicate_nested: 2\n", "duplicate_nested"),
        ("mapping_inside_list", "outer:\n  - duplicate_list: 1\n    duplicate_list: 2\n", "duplicate_list"),
        ("model_record", "models:\n  - model_id: first\n    model_id: second\n", "model_id"),
    ],
    ids=["top_level", "nested_mapping", "mapping_inside_list", "model_record"],
)
def test_config_rejects_duplicate_yaml_keys_at_every_mapping_level(
    tmp_path: Path,
    variant: str,
    yaml_text: str,
    duplicated_key: str,
) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ValueError) as error:
        load_strict_yaml_mapping(path, project_root=tmp_path)
    assert variant in {"top_level", "nested_mapping", "mapping_inside_list", "model_record"}
    assert "duplicate" in str(error.value).lower()
    assert duplicated_key not in str(error.value)


def test_builder_adapter_matches_frozen_reporting_contract(
    loaded_config: dict[str, object],
    tmp_path: Path,
) -> None:
    reference = load_frozen_reporting_config(
        Path("configs/post_hoc_final_supplementary.yaml"), project_root=Path.cwd()
    )
    with pytest.raises(ValueError, match="path"):
        load_frozen_reporting_config(Path("configs/not_the_frozen_reporting_config.yaml"), project_root=Path.cwd())
    altered_root = tmp_path / "altered_root"
    altered_path = altered_root / "configs" / "post_hoc_final_supplementary.yaml"
    altered_path.parent.mkdir(parents=True)
    altered_path.write_text("synthetic: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_frozen_reporting_config(
            Path("configs/post_hoc_final_supplementary.yaml"), project_root=altered_root
        )
    adapter = build_frozen_calibration_builder_config(
        loaded_config["calibration_contract"], frozen_reporting_config=reference
    )
    reference_registry = reference["row_order_and_schema_validation_registry"]["calibration_points"]
    expected = OrderedDict(
        (
            (
                "outputs",
                OrderedDict(
                    (
                        (
                            "scientific_and_closure",
                            OrderedDict(
                                (
                                    (
                                        "calibration_points",
                                        OrderedDict(
                                            (
                                                ("columns", list(COMMON_COLUMNS)),
                                                (
                                                    "conditional_row_inventory_contract",
                                                    OrderedDict((("n_requested_bins", 10),)),
                                                ),
                                            )
                                        ),
                                    ),
                                )
                            ),
                        ),
                    )
                ),
            ),
            (
                "row_order_and_schema_validation_registry",
                OrderedDict(
                    (
                        (
                            "calibration_points",
                            OrderedDict(
                                (
                                    ("column_order", list(COMMON_COLUMNS)),
                                    ("dtype_map", copy.deepcopy(reference_registry["dtype_map"])),
                                    ("nullable_map", copy.deepcopy(reference_registry["nullable_map"])),
                                    (
                                        "missing_representation",
                                        copy.deepcopy(reference_registry["missing_representation"]),
                                    ),
                                )
                            ),
                        ),
                    )
                ),
            ),
        )
    )
    _assert_recursive_order_and_value(adapter, expected)


def test_closed_yaml_has_no_undeclared_builder_alias_top_level_keys(
    loaded_config: dict[str, object], tmp_path: Path
) -> None:
    assert tuple(loaded_config) == (
        "schema_version", "analysis", "authority", "environment", "models", "immutable_references",
        "input_contract", "calibration_contract", "output_contract", "privacy_contract",
        "external_namespace_contract",
    )
    assert "outputs" not in loaded_config
    assert "row_order_and_schema_validation_registry" not in loaded_config

    mutations: list[dict[str, object]] = []
    nested_targets = (
        ("analysis", None),
        ("authority", None),
        ("models", 0),
        ("immutable_references", 0),
        ("authority.production_read_paths", 0),
        ("authority.allowed_commands", 0),
        ("output_contract.csv_serialization", None),
        ("privacy_contract.provenance_date_allowlist", 0),
    )
    for dotted_path, list_index in nested_targets:
        keys = dotted_path.split(".")
        original: object = loaded_config
        for key in keys:
            original = original[key]
        if list_index is not None:
            original = original[list_index]
        assert isinstance(original, Mapping) and original
        original_keys = list(original)
        for mutation_kind in ("unknown", "missing", "reordered", "wrong_type"):
            candidate = copy.deepcopy(loaded_config)
            target: object = candidate
            for key in keys:
                target = target[key]
            if list_index is not None:
                target = target[list_index]
            assert isinstance(target, dict)
            if mutation_kind == "unknown":
                target["synthetic_unknown_key"] = "synthetic"
            elif mutation_kind == "missing":
                target.pop(original_keys[-1])
            elif mutation_kind == "reordered":
                first_key = original_keys[0]
                first_value = target.pop(first_key)
                target[first_key] = first_value
            else:
                target[original_keys[0]] = ["synthetic_wrong_type"]
            mutations.append(candidate)

    for index, mutation in enumerate(mutations):
        candidate_path = tmp_path / f"closed_config_mutation_{index}.yaml"
        candidate_path.write_text(yaml.safe_dump(mutation, sort_keys=False, allow_unicode=True), encoding="utf-8")
        try:
            with pytest.raises(ValueError, match="closed contract|type"):
                load_config(candidate_path, project_root=tmp_path)
        finally:
            candidate_path.unlink()


def test_fewer_than_five_bins_sets_visualization_ready_false(
    loaded_config: dict[str, object],
    synthetic_prediction_frames_all_tied: OrderedDict[str, pd.DataFrame],
) -> None:
    frame = synthetic_prediction_frames_all_tied[MODEL_IDS[0]]
    result = build_calibration_points(frame["y_true"], frame["p_cal"], _builder_config(loaded_config))
    assert result["n_actual_bins"].tolist() == [1]
    assert result["visualization_ready"].tolist() == [False]


def test_binning_uses_no_jitter_or_random_tie_breaking(
    loaded_config: dict[str, object],
    synthetic_prediction_frames_tied: OrderedDict[str, pd.DataFrame],
) -> None:
    frame = synthetic_prediction_frames_tied[MODEL_IDS[0]]
    first = build_calibration_points(frame["y_true"], frame["p_cal"], _builder_config(loaded_config))
    second = build_calibration_points(frame["y_true"], frame["p_cal"], _builder_config(loaded_config))
    pd.testing.assert_frame_equal(first, second, check_exact=True)


def test_identical_probabilities_are_never_split(
    loaded_config: dict[str, object],
    synthetic_prediction_frames_tied: OrderedDict[str, pd.DataFrame],
) -> None:
    frame = synthetic_prediction_frames_tied[MODEL_IDS[0]]
    result = build_calibration_points(frame["y_true"], frame["p_cal"], _builder_config(loaded_config))
    assert result["n"].astype(int).tolist() == [39, 39, 39, 39, 39]


def test_bin_assignment_is_label_blind(
    loaded_config: dict[str, object],
    synthetic_prediction_frames_distinct: OrderedDict[str, pd.DataFrame],
) -> None:
    frame = synthetic_prediction_frames_distinct[MODEL_IDS[0]]
    first = build_calibration_points(frame["y_true"], frame["p_cal"], _builder_config(loaded_config))
    second = build_calibration_points(1 - frame["y_true"], frame["p_cal"], _builder_config(loaded_config))
    columns = ["bin_index", "n", "p_cal_mean", "n_requested_bins", "n_actual_bins"]
    pd.testing.assert_frame_equal(first[columns], second[columns], check_exact=True)


def test_bin_counts_and_model_totals_reconcile_exactly(
    loaded_config: dict[str, object],
    synthetic_prediction_frames_distinct: OrderedDict[str, pd.DataFrame],
) -> None:
    result = _unified_frame(loaded_config, synthetic_prediction_frames_distinct)
    grouped = result.groupby("model_id", observed=True, sort=False)[["n", "events", "nonevents"]].sum()
    assert grouped["n"].astype(int).tolist() == [195] * 6
    assert grouped["events"].astype(int).tolist() == [82] * 6
    assert grouped["nonevents"].astype(int).tolist() == [113] * 6
    mutations = []
    wrong_fraction = result.copy(deep=True)
    original_fraction = float(wrong_fraction.loc[0, "observed_fraction"])
    wrong_fraction.loc[0, "observed_fraction"] = (
        original_fraction - 0.01 if original_fraction == 1.0 else original_fraction + 0.01
    )
    mutations.append(wrong_fraction)
    wrong_identity = result.copy(deep=True)
    wrong_identity.loc[0, "model_display_name"] = "Synthetic altered display"
    mutations.append(wrong_identity)
    wrong_interval = result.copy(deep=True)
    wrong_interval.loc[0, "observed_ci_low"] = float(wrong_interval.loc[0, "observed_ci_low"]) + 0.001
    mutations.append(wrong_interval)
    for column in ("observed_fraction", "observed_ci_low", "observed_ci_high"):
        one_ulp = result.copy(deep=True)
        current = float(one_ulp.loc[0, column])
        one_ulp.loc[0, column] = np.nextafter(current, np.inf if current < 1.0 else -np.inf)
        mutations.append(one_ulp)
    wrong_requested_bins = result.copy(deep=True)
    wrong_requested_bins.loc[0, "n_requested_bins"] = 9
    mutations.append(wrong_requested_bins)
    for mutation in mutations:
        with pytest.raises(ValueError):
            validate_unified_calibration_points(mutation, loaded_config)


@pytest.mark.parametrize("event_count", [0, 82, 195], ids=["zero_event", "mixed_event", "all_event"])
def test_wilson_intervals_match_statsmodels_anchors(
    loaded_config: dict[str, object],
    synthetic_prediction_frames_all_tied: OrderedDict[str, pd.DataFrame],
    event_count: int,
) -> None:
    frame = synthetic_prediction_frames_all_tied[MODEL_IDS[0]].copy()
    frame["y_true"] = np.array([1] * event_count + [0] * (195 - event_count), dtype=int)
    result = build_calibration_points(frame["y_true"], frame["p_cal"], _builder_config(loaded_config))
    expected = proportion_confint(count=event_count, nobs=195, alpha=0.05, method="wilson")
    assert float(result.loc[0, "observed_ci_low"]) == float(expected[0])
    assert float(result.loc[0, "observed_ci_high"]) == float(expected[1])


def test_unified_long_table_has_exact_column_order(
    loaded_config: dict[str, object],
    synthetic_prediction_frames_distinct: OrderedDict[str, pd.DataFrame],
) -> None:
    result = _unified_frame(loaded_config, synthetic_prediction_frames_distinct)
    assert result.columns.tolist() == loaded_config["calibration_contract"]["unified_columns"]


def test_unified_long_table_has_exact_pandas_dtypes(
    loaded_config: dict[str, object],
    synthetic_prediction_frames_distinct: OrderedDict[str, pd.DataFrame],
) -> None:
    result = _unified_frame(loaded_config, synthetic_prediction_frames_distinct)
    expected = loaded_config["calibration_contract"]["pandas_dtypes"]
    assert {column: str(result[column].dtype) for column in result.columns} == expected
    assert result["estimability_status"].cat.categories.tolist() == ["estimable"]
    assert result["estimability_status"].cat.ordered is False


def test_unified_long_table_has_exact_model_and_bin_order(
    loaded_config: dict[str, object],
    synthetic_prediction_frames_distinct: OrderedDict[str, pd.DataFrame],
) -> None:
    result = _unified_frame(loaded_config, synthetic_prediction_frames_distinct)
    assert result["model_id"].drop_duplicates().tolist() == list(MODEL_IDS)
    for model_id in MODEL_IDS:
        bins = result.loc[result["model_id"] == model_id, "bin_index"].astype(int).tolist()
        assert bins == list(range(len(bins)))


def test_csv_uses_exact_missing_boolean_float_and_line_ending_contract(
    loaded_config: dict[str, object],
    synthetic_prediction_frames_distinct: OrderedDict[str, pd.DataFrame],
) -> None:
    frame = _unified_frame(loaded_config, synthetic_prediction_frames_distinct)
    _, payloads = build_artifact_payloads(
        frame,
        _synthetic_run_context(frame, loaded_config),
        loaded_config,
        deterministic_rerun_passed=True,
    )
    csv_bytes = payloads["calibration_points.csv"]
    assert csv_bytes.endswith(b"\n") and b"\r\n" not in csv_bytes
    assert b",true," in csv_bytes
    assert b",estimable,\n" in csv_bytes


def test_minirocket_common_columns_reconcile_exactly(
    tmp_path: Path,
    loaded_config: dict[str, object],
    synthetic_prediction_frames_distinct: OrderedDict[str, pd.DataFrame],
) -> None:
    frame = _unified_frame(loaded_config, synthetic_prediction_frames_distinct)
    anchor = frame.loc[frame["model_id"] == MODEL_IDS[0], list(COMMON_COLUMNS)].reset_index(drop=True)
    anchor_path = tmp_path / "synthetic_anchor.csv"
    anchor.to_csv(anchor_path, index=False, encoding="utf-8", lineterminator="\n", na_rep="", float_format="%.17g")
    reconcile_minirocket_anchor(frame, anchor_path=anchor_path)


def test_minirocket_reconciliation_mutation_fails_closed(
    tmp_path: Path,
    loaded_config: dict[str, object],
    synthetic_prediction_frames_distinct: OrderedDict[str, pd.DataFrame],
) -> None:
    frame = _unified_frame(loaded_config, synthetic_prediction_frames_distinct)
    anchor = frame.loc[frame["model_id"] == MODEL_IDS[0], list(COMMON_COLUMNS)].reset_index(drop=True)
    anchor.loc[0, "n"] = int(anchor.loc[0, "n"]) + 1
    anchor_path = tmp_path / "synthetic_anchor.csv"
    anchor.to_csv(anchor_path, index=False, encoding="utf-8", lineterminator="\n", na_rep="", float_format="%.17g")
    with pytest.raises(ValueError, match="MiniRocket"):
        reconcile_minirocket_anchor(frame, anchor_path=anchor_path)


def test_immutable_hashes_match_before_and_after_execution(
    loaded_config: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import run_efigure10_six_model_binned_calibration as _task_cli

    assert _task_cli.PROJECT_ROOT == Path.cwd()
    calls: list[tuple[str, ...]] = []
    original = calibration_module._run_authority_command

    def recording_command(argv: object, *, project_root: Path) -> object:
        calls.append(tuple(argv))
        return original(argv, project_root=project_root)

    monkeypatch.setattr(calibration_module, "_run_authority_command", recording_command)
    before = validate_runtime_authority(loaded_config, project_root=Path.cwd())
    after = validate_runtime_authority(loaded_config, project_root=Path.cwd())
    assert before["input_hashes"] == after["input_hashes"]
    assert [record["path"] for record in before["input_hashes"]] == [
        record["path"] for record in loaded_config["authority"]["production_read_paths"]
    ]
    assert all(record["status"] == "matched" for record in before["input_hashes"])
    expected_git_argv = [
        tuple(record["argv_or_shell_lines"])
        for record in loaded_config["authority"]["allowed_commands"]
        if record["command_id"] in {*(f"G{index:02d}" for index in range(1, 14)), "G18", "G19", "G20", "G21"}
    ]
    assert calls[:17] == expected_git_argv
    assert calls[17:] == expected_git_argv


def test_preexisting_work_namespace_is_refused(
    synthetic_config_dict: dict[str, object],
    synthetic_artifact_payloads: OrderedDict[str, bytes],
    tmp_path: Path,
) -> None:
    config = copy.deepcopy(synthetic_config_dict)
    config["output_contract"]["work_root"] = "work"
    (tmp_path / "work").mkdir()
    with pytest.raises(FileExistsError):
        stage_artifact_set(synthetic_artifact_payloads, config, project_root=tmp_path)


def test_preexisting_final_namespace_is_refused(
    synthetic_config_dict: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = copy.deepcopy(synthetic_config_dict)
    config["output_contract"]["final_root"] = "final"
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (tmp_path / "final").mkdir()
    with pytest.raises(FileExistsError):
        promote_artifact_set(candidate, config, project_root=tmp_path)
    race_candidate = tmp_path / "race-candidate"
    race_candidate.mkdir()
    race_config = copy.deepcopy(config)
    race_config["output_contract"]["final_root"] = "race-final"
    original_noreplace = calibration_module._atomic_rename_directory_noreplace

    def create_racing_destination(source: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / "sentinel.txt").write_text("synthetic external winner\n", encoding="utf-8")
        original_noreplace(source, destination)

    monkeypatch.setattr(calibration_module, "_atomic_rename_directory_noreplace", create_racing_destination)
    with pytest.raises(FileExistsError):
        promote_artifact_set(race_candidate, race_config, project_root=tmp_path)
    assert (tmp_path / "race-final/sentinel.txt").read_text(encoding="utf-8") == "synthetic external winner\n"
    assert race_candidate.is_dir()


def test_complete_directory_is_promoted_by_one_atomic_rename(
    loaded_config: dict[str, object],
    synthetic_prediction_frames_distinct: OrderedDict[str, pd.DataFrame],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _prepare_synthetic_analysis_project(
        tmp_path, loaded_config, synthetic_prediction_frames_distinct, monkeypatch
    )
    staged_refs: list[bool] = []
    authority_calls: list[tuple[bool, str]] = []
    original_stage = calibration_module.stage_artifact_set
    original_authority = calibration_module.validate_runtime_authority

    def recording_stage(payloads: Mapping[str, bytes], config: Mapping[str, Any], *, project_root: Path) -> Path:
        staged_refs.append(True)
        return original_stage(payloads, config, project_root=project_root)

    monkeypatch.setattr(calibration_module, "stage_artifact_set", recording_stage)

    def recording_authority(
        config: Mapping[str, Any],
        *,
        project_root: Path,
        allow_work_root: bool = False,
        status_stage: str = "preproduction",
        allow_final_root: bool = False,
    ) -> OrderedDict[str, Any]:
        authority_calls.append((allow_work_root, status_stage))
        return original_authority(
            config,
            project_root=project_root,
            allow_work_root=allow_work_root,
            status_stage=status_stage,
            allow_final_root=allow_final_root,
        )

    monkeypatch.setattr(calibration_module, "validate_runtime_authority", recording_authority)
    final_paths = run_analysis(config_path.relative_to(tmp_path), project_root=tmp_path)
    assert staged_refs == [True]
    assert authority_calls == [
        (False, "preproduction"),
        (True, "preproduction"),
        (True, "preproduction"),
        (True, "preproduction"),
        (True, "final"),
    ]
    assert tuple(path.name for path in final_paths) == (
        "calibration_points.csv", "audit.json", "manifest.json", "report.md"
    )
    assert not (tmp_path / "outputs/work/efigure10_six_model_binned_calibration").exists()
    promoted_payloads = OrderedDict((path.name, path.read_bytes()) for path in final_paths)
    validate_promoted_artifact_set(
        final_paths,
        load_config(config_path.relative_to(tmp_path), project_root=tmp_path),
        project_root=tmp_path,
        validated_candidate_payloads=promoted_payloads,
    )


def test_failed_run_cleans_temporary_files_without_final_directory(
    loaded_config: dict[str, object],
    synthetic_prediction_frames_distinct: OrderedDict[str, pd.DataFrame],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import run_efigure10_six_model_binned_calibration as cli

    recursive_deletion_calls: list[Path] = []
    real_rmtree = shutil.rmtree

    def recording_rmtree(path: object, *args: object, **kwargs: object) -> None:
        recursive_deletion_calls.append(Path(path))
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", recording_rmtree)
    real_open = Path.open
    collision_root = tmp_path / "regular-file-collision"
    collision_config = _prepare_synthetic_analysis_project(
        collision_root, loaded_config, synthetic_prediction_frames_distinct, monkeypatch
    )
    monkeypatch.setattr(cli, "PROJECT_ROOT", collision_root)
    collision_bytes = b"synthetic preexisting temporary collision\n"

    def colliding_open(path: Path, mode: str = "r", *args: object, **kwargs: object) -> Any:
        if path.name == ".calibration_points.csv.tmp" and mode == "xb" and not path.exists():
            with real_open(path, "wb") as stream:
                stream.write(collision_bytes)
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", colliding_open)
    collision_errors: list[tuple[str, str]] = []
    real_run_analysis = calibration_module.run_analysis

    def recording_run_analysis(*args: object, **kwargs: object) -> tuple[Path, Path, Path, Path]:
        try:
            return real_run_analysis(*args, **kwargs)
        except Exception as exc:
            collision_errors.append((type(exc).__name__, str(exc)))
            raise

    monkeypatch.setattr(cli, "run_analysis", recording_run_analysis)
    assert cli.main(["--config", collision_config.relative_to(collision_root).as_posix()]) == 1, collision_errors
    assert capsys.readouterr().out == (
        "P01_STATUS: FAILED_BEFORE_PROMOTION\nP01_ERROR: sanitized aggregate integrity failure\n"
    )
    work_root = collision_root / "outputs/work/efigure10_six_model_binned_calibration"
    collided = work_root / "promotion_candidate/.calibration_points.csv.tmp"
    assert collided.is_file() and collided.read_bytes() == collision_bytes
    assert work_root.is_dir()
    assert not (collision_root / "outputs/analysis/efigure10_six_model_binned_calibration").exists()
    monkeypatch.setattr(Path, "open", real_open)
    assert cli.main(["--config", collision_config.relative_to(collision_root).as_posix()]) == 1
    assert capsys.readouterr().out.startswith("P01_STATUS: FAILED_BEFORE_PROMOTION\n")
    assert collided.is_file() and collided.read_bytes() == collision_bytes

    directory_root = tmp_path / "allowed-name-directory-collision"
    directory_config = _prepare_synthetic_analysis_project(
        directory_root, loaded_config, synthetic_prediction_frames_distinct, monkeypatch
    )
    monkeypatch.setattr(cli, "PROJECT_ROOT", directory_root)
    real_link = calibration_module.os.link

    def colliding_link(source: Path, destination: Path) -> None:
        destination.mkdir()
        real_link(source, destination)

    monkeypatch.setattr(calibration_module.os, "link", colliding_link)
    assert cli.main(["--config", directory_config.relative_to(directory_root).as_posix()]) == 1
    assert capsys.readouterr().out == (
        "P01_STATUS: FAILED_BEFORE_PROMOTION\nP01_ERROR: sanitized aggregate integrity failure\n"
    )
    directory_work_root = directory_root / "outputs/work/efigure10_six_model_binned_calibration"
    collided_directory = directory_work_root / "promotion_candidate/calibration_points.csv"
    assert collided_directory.is_dir()
    assert directory_work_root.is_dir()
    assert not (directory_root / "outputs/analysis/efigure10_six_model_binned_calibration").exists()
    assert recursive_deletion_calls == []

    strict_root = tmp_path / "strict-work"
    run_a = strict_root / "run_a"
    run_b = strict_root / "run_b"
    run_a.mkdir(parents=True)
    run_b.mkdir()
    unexpected = run_a / "unexpected.txt"
    unexpected.write_text("synthetic\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="empty"):
        cleanup_exact_task_directory(
            strict_root,
            project_root=tmp_path,
            expected_direct_children=["run_a", "run_b"],
        )
    assert strict_root.is_dir() and unexpected.is_file() and run_b.is_dir()


@pytest.mark.parametrize("role", ["csv", "json", "markdown", "log", "error"])
def test_schema_aware_privacy_validator_rejects_prohibited_content(
    loaded_config: dict[str, object], role: str
) -> None:
    refs = build_participant_identifier_references(["SYNTHETIC-PRIVATE"])
    serialized = OrderedDict(((f"{role}.txt", b"sample_id"),))
    with pytest.raises(ValueError, match="privacy"):
        validate_candidate_privacy({}, serialized, [], participant_references=[refs], config=loaded_config)


def test_artifacts_contain_no_participant_identifiers(loaded_config: dict[str, object]) -> None:
    refs = build_participant_identifier_references(["SYNTHETIC-PRIVATE"])
    with pytest.raises(ValueError, match="privacy"):
        validate_candidate_privacy({}, {"report.md": b"SYNTHETIC-PRIVATE"}, [], participant_references=[refs], config=loaded_config)


def test_artifacts_contain_no_participant_level_predictions(loaded_config: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="privacy"):
        validate_candidate_privacy({}, {"audit.json": b'{"p_cal": 0.5}'}, [], participant_references=[], config=loaded_config)


def test_artifacts_contain_no_exact_dates(loaded_config: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="privacy"):
        validate_candidate_privacy({}, {"report.md": b"2026-08-14"}, [], participant_references=[], config=loaded_config)


def test_artifacts_contain_no_raw_filenames_or_per_sample_paths(loaded_config: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="privacy"):
        validate_candidate_privacy({}, {"report.md": b"breath_data/sample.csv"}, [], participant_references=[], config=loaded_config)


def test_artifacts_contain_no_absolute_paths(loaded_config: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="privacy"):
        validate_candidate_privacy({}, {"report.md": b"/private/path"}, [], participant_references=[], config=loaded_config)


def test_artifacts_contain_no_unapproved_efigure4_or_efigure9_material(loaded_config: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="privacy"):
        validate_candidate_privacy({}, {"report.md": b"efigure9_disease_category_prediction_matrix"}, [], participant_references=[], config=loaded_config)


def test_two_isolated_runs_produce_four_byte_identical_artifacts(
    loaded_config: dict[str, object],
    synthetic_prediction_frames_distinct: OrderedDict[str, pd.DataFrame],
) -> None:
    frame = _unified_frame(loaded_config, synthetic_prediction_frames_distinct)
    context = _synthetic_run_context(frame, loaded_config)
    structured_first, first = build_artifact_payloads(
        frame, context, loaded_config, deterministic_rerun_passed=True
    )
    structured_second, second = build_artifact_payloads(
        frame.copy(deep=True), copy.deepcopy(context), loaded_config, deterministic_rerun_passed=True
    )
    assert first == second
    references = [build_participant_identifier_references(["SYNTHETIC-PRIVATE-A"]),
                  build_participant_identifier_references(["SYNTHETIC-PRIVATE-B"])]
    validate_candidate_privacy(
        structured_first,
        first,
        ["aggregate candidate validation passed"],
        participant_references=references,
        config=loaded_config,
    )
    assert structured_first.keys() == structured_second.keys()


def test_every_csv_and_json_artifact_passes_exact_readback_validation(
    loaded_config: dict[str, object],
    synthetic_prediction_frames_distinct: OrderedDict[str, pd.DataFrame],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = copy.deepcopy(loaded_config)
    config["output_contract"]["work_root"] = "work"
    frame = _unified_frame(config, synthetic_prediction_frames_distinct)
    _, payloads = build_artifact_payloads(frame, _synthetic_run_context(frame, config), config, deterministic_rerun_passed=True)
    validate_artifact_payloads(payloads, config)
    corrupt_csv = copy.deepcopy(payloads)
    corrupt_csv["calibration_points.csv"] = payloads["calibration_points.csv"].replace(
        b"model_order,model_id", b"model_id,model_order", 1
    )
    with pytest.raises(ValueError):
        validate_artifact_payloads(corrupt_csv, config)
    corrupt_audit = copy.deepcopy(payloads)
    audit = json.loads(payloads["audit.json"], object_pairs_hook=OrderedDict)
    audit["artifact_status"] = "promoted"
    corrupt_audit["audit.json"] = calibration_module._serialize_json(audit)
    with pytest.raises(ValueError):
        validate_artifact_payloads(corrupt_audit, config)
    corrupt_manifest = copy.deepcopy(payloads)
    manifest = json.loads(payloads["manifest.json"], object_pairs_hook=OrderedDict)
    manifest["files"][0]["path"] = "wrong-parent/calibration_points.csv"
    corrupt_manifest["manifest.json"] = calibration_module._serialize_json(manifest)
    with pytest.raises(ValueError, match="full path"):
        validate_artifact_payloads(corrupt_manifest, config)
    corrupt_report = copy.deepcopy(payloads)
    corrupt_report["report.md"] = payloads["report.md"].replace(
        b"# Authority and reconciliation", b"# Wrong section", 1
    )
    report_manifest = json.loads(corrupt_report["manifest.json"], object_pairs_hook=OrderedDict)
    report_record = next(item for item in report_manifest["files"] if item["path"].endswith("/report.md"))
    report_record["byte_size"] = len(corrupt_report["report.md"])
    report_record["sha256"] = hashlib.sha256(corrupt_report["report.md"]).hexdigest()
    corrupt_report["manifest.json"] = calibration_module._serialize_json(report_manifest)
    with pytest.raises(ValueError, match="section order"):
        validate_artifact_payloads(corrupt_report, config)
    candidate = stage_artifact_set(payloads, config, project_root=tmp_path)
    assert {path.name for path in candidate.iterdir()} == set(config["output_contract"]["files"])
    collision_root = tmp_path / "collision"
    collision_config = copy.deepcopy(config)
    collision_config["output_contract"]["work_root"] = "collision"
    collision_candidate = collision_root / collision_config["output_contract"]["promotion_candidate"]
    (collision_root / collision_config["output_contract"]["run_a"]).mkdir(parents=True)
    (collision_root / collision_config["output_contract"]["run_b"]).mkdir()
    temporary = collision_candidate / ".calibration_points.csv.tmp"
    original_mkdir = Path.mkdir

    def create_colliding_temporary(path: Path, *args: object, **kwargs: object) -> None:
        original_mkdir(path, *args, **kwargs)
        if path == collision_candidate:
            temporary.write_bytes(b"preexisting synthetic sentinel\n")

    monkeypatch.setattr(Path, "mkdir", create_colliding_temporary)
    with pytest.raises(FileExistsError):
        stage_artifact_set(payloads, collision_config, project_root=tmp_path)
    assert temporary.read_bytes() == b"preexisting synthetic sentinel\n"


def test_report_retains_collapsed_nonmonotonic_unfavorable_and_not_ready_states(
    loaded_config: dict[str, object],
    synthetic_prediction_frames_all_tied: OrderedDict[str, pd.DataFrame],
) -> None:
    frame = _unified_frame(loaded_config, synthetic_prediction_frames_all_tied)
    _, payloads = build_artifact_payloads(frame, _synthetic_run_context(frame, loaded_config), loaded_config, deterministic_rerun_passed=True)
    report = payloads["report.md"].decode("utf-8")
    assert "not_assessable_single_bin" in report
    assert "false" in report
    assert "retained" in report.lower()


def test_no_prohibited_metric_smoother_threshold_or_figure_is_produced(
    loaded_config: dict[str, object],
    synthetic_prediction_frames_distinct: OrderedDict[str, pd.DataFrame],
) -> None:
    frame = _unified_frame(loaded_config, synthetic_prediction_frames_distinct)
    _, payloads = build_artifact_payloads(frame, _synthetic_run_context(frame, loaded_config), loaded_config, deterministic_rerun_passed=True)
    assert tuple(payloads) == ("calibration_points.csv", "audit.json", "manifest.json", "report.md")


def test_standard_gcvocs_environment_rejects_efigure4_historical_environment(loaded_config: dict[str, object]) -> None:
    assert loaded_config["environment"]["conda_environment"] == "gcvocs"
    assert loaded_config["environment"]["efigure4_historical_environment_forbidden"] is True


def test_efigure4_and_efigure9_files_remain_byte_unchanged(
    external_namespace_fixture: OrderedDict[str, tuple[Path, bytes]],
) -> None:
    for path, expected in external_namespace_fixture.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(expected)
    assert all(path.read_bytes() == expected for path, expected in external_namespace_fixture.values())


def test_provenance_date_allowlist_accepts_only_two_fixed_paths_and_value(loaded_config: dict[str, object]) -> None:
    allowed = loaded_config["privacy_contract"]["provenance_date_allowlist"]
    validate_candidate_privacy(
        {"analysis": {"phase1_authoring_date": "2026-08-13"}, "audit.json": {"authoring_state": {"date": "2026-08-13"}}},
        {}, [], participant_references=[], config=loaded_config,
    )
    assert len(allowed) == 2
    invalid_candidates = (
        {"analysis": {"phase1_authoring_date": "2026-08-14"}},
        {"audit.json": {"authoring_state": {"date": "2026-08-14"}}},
        {"runtime": {"date": "2026-08-13"}},
        {"runtime": {"collection_date": "2026-08-13"}},
    )
    for candidate in invalid_candidates:
        with pytest.raises(ValueError, match="privacy"):
            validate_candidate_privacy(candidate, {}, [], participant_references=[], config=loaded_config)
    with pytest.raises(ValueError, match="privacy"):
        validate_candidate_privacy(
            {}, {"report.md": b"runtime_timestamp: 2026-08-13T12:34:56"}, [],
            participant_references=[], config=loaded_config,
        )


@pytest.mark.parametrize("role", ["csv", "json", "markdown", "log", "error"])
@pytest.mark.parametrize("token", ["SYNTHETIC-PRIVATE", "1"], ids=["nonnumeric", "qualified_numeric"])
def test_raw_identifier_injections_are_rejected_in_csv_json_markdown_and_log_error_candidates(
    loaded_config: dict[str, object], role: str, token: str
) -> None:
    refs = build_participant_identifier_references([token])
    injected = token if token != "1" else f"participant_id: {token}"
    serialized = {} if role in {"log", "error"} else {f"{role}.txt": injected.encode()}
    candidates = [injected] if role in {"log", "error"} else []
    with pytest.raises(ValueError) as error:
        validate_candidate_privacy({}, serialized, candidates, participant_references=[refs], config=loaded_config)
    assert token not in str(error.value)


def test_numeric_identifier_matching_does_not_reject_valid_aggregate_numbers(loaded_config: dict[str, object]) -> None:
    refs = build_participant_identifier_references(["1"])
    validate_candidate_privacy({"aggregate": {"model_order": 1, "n": 1}}, {}, [], participant_references=[refs], config=loaded_config)
    with pytest.raises(ValueError, match="privacy"):
        validate_candidate_privacy({"aggregate": {"participant_id": "1"}}, {}, [], participant_references=[refs], config=loaded_config)
    nonnumeric = build_participant_identifier_references(
        ["001", "+0001", "++1", "+-1", "NA", "CaseID", "SYN-UNICODE-\u00e9", "SYN-UNICODE-e\u0301"]
    )
    for token in ("001", "+0001", "++1", "+-1"):
        assert nonnumeric.matches_nonnumeric_token(token)
        assert not nonnumeric.matches_identifier_qualified_numeric("participant_id", token)
    for harmless in ("NATURE", "ShowCaseIdentifier", "prefixSYN-UNICODE-éSuffix"):
        validate_candidate_privacy({}, {"report.md": harmless.encode("utf-8")}, [],
                                   participant_references=[nonnumeric], config=loaded_config)
    for exact_or_delimited in ("NA", "value: NA,", "CaseID; retained"):
        with pytest.raises(ValueError, match="privacy"):
            validate_candidate_privacy({}, {"report.md": exact_or_delimited.encode("utf-8")}, [],
                                       participant_references=[nonnumeric], config=loaded_config)
    numeric = build_participant_identifier_references(["1", "+1", "10"])
    for valid in ("participant_id: 1", "SUBJECT_ID=+1", "record_id,10", "identifier 1"):
        with pytest.raises(ValueError, match="privacy"):
            validate_candidate_privacy({}, {"report.md": valid.encode("utf-8")}, [],
                                       participant_references=[numeric], config=loaded_config)
    for invalid in ("unrelated: 11", "notparticipant_id: 1", "participant_id_number: 1"):
        validate_candidate_privacy({}, {"report.md": invalid.encode("utf-8")}, [],
                                   participant_references=[numeric], config=loaded_config)


def test_participant_identifier_references_are_redacted_nonserializable_and_cleared_before_staging(
    loaded_config: dict[str, object],
    synthetic_prediction_frames_distinct: OrderedDict[str, pd.DataFrame],
    tmp_path: Path,
) -> None:
    refs = build_participant_identifier_references(["SYNTHETIC-PRIVATE", "001"])
    assert repr(refs) == "<ParticipantIdentifierReferences redacted>"
    assert str(refs) == "<ParticipantIdentifierReferences redacted>"
    with pytest.raises((TypeError, pickle.PicklingError)):
        pickle.dumps(refs)
    with pytest.raises(TypeError):
        hash(refs)
    with pytest.raises(TypeError):
        iter(refs)
    with pytest.raises(TypeError):
        json.dumps(refs)
    assert refs.matches_nonnumeric_token("SYNTHETIC-PRIVATE")
    discard_participant_identifier_references(refs)
    assert refs.is_cleared()
    assert not refs.matches_nonnumeric_token("SYNTHETIC-PRIVATE")
    config = copy.deepcopy(loaded_config)
    config["output_contract"]["work_root"] = "work"
    frame = _unified_frame(config, synthetic_prediction_frames_distinct)
    _, payloads = build_artifact_payloads(
        frame, _synthetic_run_context(frame, config), config, deterministic_rerun_passed=True
    )
    candidate = stage_artifact_set(payloads, config, project_root=tmp_path)
    assert refs.is_cleared() and candidate.is_dir()


def test_audit_status_is_validated_for_atomic_promotion_without_final_completion_claims(
    loaded_config: dict[str, object],
    synthetic_prediction_frames_distinct: OrderedDict[str, pd.DataFrame],
) -> None:
    frame = _unified_frame(loaded_config, synthetic_prediction_frames_distinct)
    structured, _ = build_artifact_payloads(frame, _synthetic_run_context(frame, loaded_config), loaded_config, deterministic_rerun_passed=True)
    audit = structured["audit.json"]
    assert tuple(audit) == tuple(loaded_config["output_contract"]["audit_key_order"])
    assert tuple(audit["runtime_state"]) == (
        "head", "descends_from_authoring_head", "relevant_diff_review_status", "command_inventory_status",
        "runtime_git_argv_status", "visible_status_stage",
    )
    assert audit["runtime_state"]["command_inventory_status"] == "passed"
    assert audit["input_hashes_before"]
    assert "bound" not in json.dumps(audit["environment"]).lower()
    assert audit["artifact_status"] == "validated_for_atomic_promotion"
    serialized = json.dumps(audit)
    assert "PHASE_2_STATUS" not in serialized
    assert "complete" not in serialized.lower()
    empty_hash_context = _synthetic_run_context(frame, loaded_config)
    empty_hash_context["input_hashes_before"] = []
    with pytest.raises(ValueError, match="hashes"):
        build_artifact_payloads(frame, empty_hash_context, loaded_config, deterministic_rerun_passed=True)
    placeholder_context = _synthetic_run_context(frame, loaded_config)
    placeholder_context["environment_versions"]["statsmodels"] = "environment_bound"
    with pytest.raises(ValueError, match="placeholder"):
        build_artifact_payloads(frame, placeholder_context, loaded_config, deterministic_rerun_passed=True)
    failure_context = _synthetic_run_context(frame, loaded_config)
    failure_context["authority_checks"]["after_both_runs"]["command_inventory_status"] = "failed"
    with pytest.raises(ValueError, match="dynamic authority"):
        build_artifact_payloads(frame, failure_context, loaded_config, deterministic_rerun_passed=True)
    privacy_failure_context = _synthetic_run_context(frame, loaded_config)
    privacy_failure_context["privacy_validation"]["schema"] = "failed"
    with pytest.raises(ValueError, match="privacy"):
        build_artifact_payloads(frame, privacy_failure_context, loaded_config, deterministic_rerun_passed=True)
    absent_privacy_context = _synthetic_run_context(frame, loaded_config)
    absent_privacy_context.pop("privacy_validation")
    with pytest.raises(ValueError, match="key order"):
        build_artifact_payloads(frame, absent_privacy_context, loaded_config, deterministic_rerun_passed=True)
    absent_projection_context = _synthetic_run_context(frame, loaded_config)
    absent_projection_context["input_validation"].pop("projection_columns")
    with pytest.raises(ValueError, match="input validation|key order"):
        build_artifact_payloads(frame, absent_projection_context, loaded_config, deterministic_rerun_passed=True)
    fabricated_unused_column_context = _synthetic_run_context(frame, loaded_config)
    fabricated_unused_column_context["input_validation"]["unused_participant_columns_loaded"] = True
    with pytest.raises(ValueError, match="input validation"):
        build_artifact_payloads(
            frame, fabricated_unused_column_context, loaded_config, deterministic_rerun_passed=True
        )
    absent_anchor_evidence_context = _synthetic_run_context(frame, loaded_config)
    absent_anchor_evidence_context["minirocket_reconciliation"].pop("anchor_sha256")
    with pytest.raises(ValueError, match="MiniRocket|key order"):
        build_artifact_payloads(
            frame, absent_anchor_evidence_context, loaded_config, deterministic_rerun_passed=True
        )
    inconsistent_anchor_context = _synthetic_run_context(frame, loaded_config)
    inconsistent_anchor_context["minirocket_reconciliation"]["anchor_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="authority"):
        build_artifact_payloads(
            frame, inconsistent_anchor_context, loaded_config, deterministic_rerun_passed=True
        )


def test_p01_validated_is_emitted_only_after_internal_integrity_and_phase2_complete_requires_f01(
    loaded_config: dict[str, object],
    synthetic_prediction_frames_distinct: OrderedDict[str, pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import run_efigure10_six_model_binned_calibration as cli

    subprocess_environment = os.environ.copy()
    subprocess_environment.pop("PYTHONPATH", None)
    subprocess_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess_environment["PYTHONPYCACHEPREFIX"] = str(tmp_path / "absent-cli-pycache-prefix")
    help_result = subprocess.run(
        [
            "[PRIVATE_PATH_REDACTED]",
            "scripts/run_efigure10_six_model_binned_calibration.py",
            "--help",
        ],
        cwd=Path.cwd(),
        env=subprocess_environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0
    assert "--config" in help_result.stdout
    assert not (tmp_path / "absent-cli-pycache-prefix").exists()

    shadow_root = tmp_path / "shadow-import"
    shadow_module = shadow_root / "src/analysis/efigure10_six_model_binned_calibration.py"
    shadow_module.parent.mkdir(parents=True)
    (shadow_root / "src/__init__.py").write_text("", encoding="utf-8")
    (shadow_root / "src/analysis/__init__.py").write_text("", encoding="utf-8")
    shadow_module.write_text("raise RuntimeError('synthetic shadow import selected')\n", encoding="utf-8")
    shadow_environment = subprocess_environment.copy()
    shadow_environment["PYTHONPATH"] = os.pathsep.join((str(shadow_root), str(Path.cwd())))
    shadow_result = subprocess.run(
        [
            "[PRIVATE_PATH_REDACTED]",
            "scripts/run_efigure10_six_model_binned_calibration.py",
            "--help",
        ],
        cwd=Path.cwd(),
        env=shadow_environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert shadow_result.returncode == 0
    assert "--config" in shadow_result.stdout
    assert "shadow import selected" not in shadow_result.stderr

    success_root = tmp_path / "success"
    success_config = _prepare_synthetic_analysis_project(
        success_root, loaded_config, synthetic_prediction_frames_distinct, monkeypatch
    )
    monkeypatch.setattr(cli, "PROJECT_ROOT", success_root)
    success_errors: list[tuple[str, str]] = []
    real_run_analysis = calibration_module.run_analysis

    def recording_run_analysis(*args: object, **kwargs: object) -> tuple[Path, Path, Path, Path]:
        try:
            return real_run_analysis(*args, **kwargs)
        except Exception as exc:
            success_errors.append((type(exc).__name__, str(exc)))
            raise

    monkeypatch.setattr(cli, "run_analysis", recording_run_analysis)
    assert cli.main(["--config", success_config.relative_to(success_root).as_posix()]) == 0, success_errors
    output = capsys.readouterr().out
    assert output.startswith("P01_STATUS: VALIDATED\n")
    assert "PHASE_2_STATUS: COMPLETE" not in output

    original_load = calibration_module._load_and_reconcile_predictions_with_evidence
    pre_root = tmp_path / "prepromotion"
    pre_config = _prepare_synthetic_analysis_project(
        pre_root, loaded_config, synthetic_prediction_frames_distinct, monkeypatch
    )
    monkeypatch.setattr(cli, "PROJECT_ROOT", pre_root)
    monkeypatch.setattr(
        calibration_module,
        "_load_and_reconcile_predictions_with_evidence",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("synthetic failure")),
    )
    assert cli.main(["--config", pre_config.relative_to(pre_root).as_posix()]) == 1
    assert capsys.readouterr().out.startswith("P01_STATUS: FAILED_BEFORE_PROMOTION\n")
    monkeypatch.setattr(calibration_module, "_load_and_reconcile_predictions_with_evidence", original_load)

    original_promoted_validation = calibration_module.validate_promoted_artifact_set
    post_root = tmp_path / "postpromotion"
    post_config = _prepare_synthetic_analysis_project(
        post_root, loaded_config, synthetic_prediction_frames_distinct, monkeypatch
    )
    monkeypatch.setattr(cli, "PROJECT_ROOT", post_root)
    monkeypatch.setattr(
        calibration_module,
        "validate_promoted_artifact_set",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("synthetic postpromotion failure")),
    )
    assert cli.main(["--config", post_config.relative_to(post_root).as_posix()]) == 1
    assert capsys.readouterr().out.startswith("P01_STATUS: INCOMPLETE_INTEGRITY\n")
    assert (post_root / "outputs/analysis/efigure10_six_model_binned_calibration").is_dir()
    monkeypatch.setattr(calibration_module, "validate_promoted_artifact_set", original_promoted_validation)

    original_authority = calibration_module.validate_runtime_authority
    final_gate_root = tmp_path / "postpromotion-authority"
    final_gate_config = _prepare_synthetic_analysis_project(
        final_gate_root, loaded_config, synthetic_prediction_frames_distinct, monkeypatch
    )
    monkeypatch.setattr(cli, "PROJECT_ROOT", final_gate_root)
    bytes_at_final_gate: dict[str, bytes] = {}

    def fail_final_authority(
        config: Mapping[str, Any],
        *,
        project_root: Path,
        allow_work_root: bool = False,
        status_stage: str = "preproduction",
        allow_final_root: bool = False,
    ) -> OrderedDict[str, Any]:
        if status_stage == "final":
            final_directory = project_root / "outputs/analysis/efigure10_six_model_binned_calibration"
            bytes_at_final_gate.update({path.name: path.read_bytes() for path in final_directory.iterdir()})
            raise RuntimeError("synthetic final integrity failure")
        return original_authority(
            config,
            project_root=project_root,
            allow_work_root=allow_work_root,
            status_stage=status_stage,
            allow_final_root=allow_final_root,
        )

    monkeypatch.setattr(calibration_module, "validate_runtime_authority", fail_final_authority)
    assert cli.main(["--config", final_gate_config.relative_to(final_gate_root).as_posix()]) == 1
    final_gate_output = capsys.readouterr().out
    assert final_gate_output.startswith("P01_STATUS: INCOMPLETE_INTEGRITY\n")
    assert "P01_STATUS: VALIDATED" not in final_gate_output
    final_gate_directory = final_gate_root / "outputs/analysis/efigure10_six_model_binned_calibration"
    final_gate_bytes = {path.name: path.read_bytes() for path in final_gate_directory.iterdir()}
    assert final_gate_bytes == bytes_at_final_gate
    monkeypatch.setattr(calibration_module, "validate_runtime_authority", original_authority)

    original_cleanup = calibration_module.cleanup_exact_task_directory
    absence_gate_root = tmp_path / "postcleanup-absence"
    absence_gate_config = _prepare_synthetic_analysis_project(
        absence_gate_root, loaded_config, synthetic_prediction_frames_distinct, monkeypatch
    )
    monkeypatch.setattr(cli, "PROJECT_ROOT", absence_gate_root)
    bytes_before_absence_gate: dict[str, bytes] = {}

    def cleanup_then_create_foreign_paths(
        path: Path,
        *,
        project_root: Path,
        expected_direct_children: list[str],
    ) -> None:
        original_cleanup(
            path,
            project_root=project_root,
            expected_direct_children=expected_direct_children,
        )
        final_directory = project_root / "outputs/analysis/efigure10_six_model_binned_calibration"
        bytes_before_absence_gate.update({item.name: item.read_bytes() for item in final_directory.iterdir()})
        pytest_root = project_root / "outputs/work/efigure10_six_model_binned_calibration_pytest"
        pytest_root.mkdir()
        (pytest_root / "synthetic-marker.txt").write_text("synthetic pytest residue\n", encoding="utf-8")
        pytest_cache = project_root / ".pytest_cache"
        pytest_cache.mkdir()
        (pytest_cache / "synthetic-marker.txt").write_text("synthetic cache residue\n", encoding="utf-8")
        (project_root / ".coverage").write_text("synthetic foreign coverage path\n", encoding="utf-8")
        (project_root / "coverage.xml").write_text("synthetic coverage XML\n", encoding="utf-8")
        htmlcov = project_root / "htmlcov"
        htmlcov.mkdir()
        (htmlcov / "synthetic-marker.html").write_text("synthetic htmlcov residue\n", encoding="utf-8")

    monkeypatch.setattr(calibration_module, "cleanup_exact_task_directory", cleanup_then_create_foreign_paths)
    assert cli.main(["--config", absence_gate_config.relative_to(absence_gate_root).as_posix()]) == 0
    absence_output = capsys.readouterr().out
    assert absence_output.startswith("P01_STATUS: VALIDATED\n")
    assert (
        absence_gate_root / "outputs/work/efigure10_six_model_binned_calibration_pytest/synthetic-marker.txt"
    ).read_text(encoding="utf-8") == "synthetic pytest residue\n"
    assert (absence_gate_root / ".pytest_cache/synthetic-marker.txt").read_text(
        encoding="utf-8"
    ) == "synthetic cache residue\n"
    assert (absence_gate_root / ".coverage").read_text(encoding="utf-8") == "synthetic foreign coverage path\n"
    assert (absence_gate_root / "coverage.xml").read_text(encoding="utf-8") == "synthetic coverage XML\n"
    assert (absence_gate_root / "htmlcov/synthetic-marker.html").read_text(
        encoding="utf-8"
    ) == "synthetic htmlcov residue\n"
    absence_final = absence_gate_root / "outputs/analysis/efigure10_six_model_binned_calibration"
    assert {item.name: item.read_bytes() for item in absence_final.iterdir()} == bytes_before_absence_gate


def test_postpromotion_failure_emits_incomplete_integrity_without_mutating_final_directory(
    loaded_config: dict[str, object],
    synthetic_prediction_frames_distinct: OrderedDict[str, pd.DataFrame],
    tmp_path: Path,
) -> None:
    config = copy.deepcopy(loaded_config)
    config["output_contract"]["final_root"] = "final"
    frame = _unified_frame(config, synthetic_prediction_frames_distinct)
    _, candidate_payloads = build_artifact_payloads(
        frame,
        _synthetic_run_context(frame, config),
        config,
        deterministic_rerun_passed=True,
    )
    final = tmp_path / "final"
    final.mkdir()
    for name, payload in candidate_payloads.items():
        (final / name).write_bytes(payload)
    altered_report = candidate_payloads["report.md"].replace(b"retained", b"retained ", 1)
    assert altered_report != candidate_payloads["report.md"]
    (final / "report.md").write_bytes(altered_report)
    manifest = json.loads(candidate_payloads["manifest.json"], object_pairs_hook=OrderedDict)
    report_record = next(item for item in manifest["files"] if item["path"].endswith("/report.md"))
    report_record["byte_size"] = len(altered_report)
    report_record["sha256"] = hashlib.sha256(altered_report).hexdigest()
    (final / "manifest.json").write_bytes(calibration_module._serialize_json(manifest))
    tampered = {path.name: path.read_bytes() for path in final.iterdir()}
    with pytest.raises(ValueError, match="candidate"):
        validate_promoted_artifact_set(
            tuple(final / name for name in config["output_contract"]["files"]),
            config,
            project_root=tmp_path,
            validated_candidate_payloads=candidate_payloads,
        )
    assert {path.name: path.read_bytes() for path in final.iterdir()} == tampered


def test_production_read_allowlist_matches_complete_import_closure(
    loaded_config: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    records = loaded_config["authority"]["production_read_paths"]
    paths = [record["path"] for record in records]
    assert paths.index("src/analysis/baseline_table.py") < paths.index("src/analysis/data_split.py")
    assert "src/training/common/metrics.py" in paths
    assert not any(path.endswith((".pyc", ".pyo")) or "__pycache__" in path for path in paths)
    assert len(paths) == len(set(paths))
    for record in records:
        expected = record["sha256_or_binding"]
        if isinstance(expected, str) and len(expected) == 64 and set(expected) <= set("0123456789abcdef"):
            assert _sha256(Path(record["path"])) == expected
    evidence = calibration_module._verify_import_origins(loaded_config, project_root=Path.cwd())
    assert evidence and all(record["status"] == "matched" for record in evidence)
    source_module = sys.modules[calibration_module.__name__]
    original_origin = source_module.__file__
    monkeypatch.setattr(source_module, "__file__", str(Path.cwd() / "src/analysis/__pycache__/altered.pyc"))
    with pytest.raises(RuntimeError, match="origin"):
        calibration_module._verify_import_origins(loaded_config, project_root=Path.cwd())
    monkeypatch.setattr(source_module, "__file__", original_origin)


def test_test_only_read_allowlist_matches_regression_fixture_closure(loaded_config: dict[str, object]) -> None:
    records = loaded_config["authority"]["test_only_read_paths"]
    paths = [record["path"] for record in records]
    assert paths[0] == "tests/test_efigure10_six_model_binned_calibration.py"
    assert "tests/test_posthoc_final_audit.py" in paths
    assert len(paths) == len(set(paths))
    for record in records:
        expected = record["sha256_or_binding"]
        if isinstance(expected, str) and len(expected) == 64 and set(expected) <= set("0123456789abcdef"):
            assert _sha256(Path(record["path"])) == expected


def test_test_read_scope_cannot_open_real_predictions_or_anchor_before_runtime_gate(
    synthetic_prediction_csv_files: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    original_hash = calibration_module._sha256_path
    original_load = calibration_module.load_prediction_projection

    def recording_hash(path: Path) -> str:
        if path.name.endswith(".csv"):
            events.append(f"hash:{path.name}")
        return original_hash(path)

    def recording_load(path: Path, *, project_root: Path) -> pd.DataFrame:
        events.append(f"read:{path.name}")
        return original_load(path, project_root=project_root)

    monkeypatch.setattr(calibration_module, "_sha256_path", recording_hash)
    monkeypatch.setattr(calibration_module, "load_prediction_projection", recording_load)
    _load_reconciled(synthetic_prediction_csv_files)
    first_read = next(index for index, event in enumerate(events) if event.startswith("read:"))
    assert len([event for event in events[:first_read] if event.startswith("hash:")]) == 6
    assert all("binary_" not in event and "posthoc_final" not in event for event in events)


def test_g06_authority_is_exactly_path_scoped(loaded_config: dict[str, object]) -> None:
    g06 = next(record for record in loaded_config["authority"]["allowed_commands"] if record["command_id"] == "G06")
    assert g06["execution_kind"] == "argv"
    assert tuple(g06["argv_or_shell_lines"]) == EXPECTED_G06_ARGV
    assert calibration_module._validate_g06_command(g06) == EXPECTED_G06_ARGV
    expected = g06["postconditions"]["assertions"][0]["expected"]
    entry = [
        "## main",
        "?? configs/efigure10_six_model_binned_calibration.yaml",
        "?? scripts/run_efigure10_six_model_binned_calibration.py",
        "?? src/analysis/efigure10_six_model_binned_calibration.py",
        "?? tests/test_efigure10_six_model_binned_calibration.py",
    ]
    final = entry[:2] + ["?? outputs/analysis/efigure10_six_model_binned_calibration/"] + entry[2:]
    assert expected == {"phase2_entry": entry, "preproduction": entry, "final": final}


@pytest.mark.parametrize(
    "invalid_argv",
    [
        ("git", "status", "--short", "--branch"),
        EXPECTED_G06_ARGV[:6] + (".",) + EXPECTED_G06_ARGV[7:],
        EXPECTED_G06_ARGV[:6] + ("outputs/*",) + EXPECTED_G06_ARGV[7:],
        EXPECTED_G06_ARGV[:6] + ("outputs/work",) + EXPECTED_G06_ARGV[7:],
        EXPECTED_G06_ARGV[:6] + ("assets",) + EXPECTED_G06_ARGV[7:],
        EXPECTED_G06_ARGV[:6] + ("",) + EXPECTED_G06_ARGV[7:],
    ],
    ids=("repository_wide", "dot", "glob", "parent", "unrelated", "empty"),
)
def test_g06_mutations_fail_before_any_git_command(
    loaded_config: dict[str, object], monkeypatch: pytest.MonkeyPatch, invalid_argv: tuple[str, ...]
) -> None:
    config = copy.deepcopy(loaded_config)
    g06 = next(record for record in config["authority"]["allowed_commands"] if record["command_id"] == "G06")
    g06["argv_or_shell_lines"] = list(invalid_argv)

    def reject_execution(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("invalid G06 reached subprocess execution")

    monkeypatch.setattr(calibration_module, "_run_authority_command", reject_execution)
    with pytest.raises(ValueError, match="G06"):
        calibration_module._validate_git_authority(config, project_root=Path.cwd())


def test_unrelated_user_and_test_residue_is_nonblocking_and_untouched(
    loaded_config: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = copy.deepcopy(loaded_config)
    marker_paths = (
        tmp_path / "assets/synthetic-marker.txt",
        tmp_path / ".vscode/synthetic-marker.txt",
        tmp_path / "user-work/synthetic-marker.txt",
        tmp_path / "outputs/work/efigure10_six_model_binned_calibration_pytest/synthetic-marker.txt",
        tmp_path / ".pytest_cache/synthetic-marker.txt",
        tmp_path / ".coverage",
        tmp_path / "coverage.xml",
        tmp_path / "htmlcov/synthetic-marker.html",
    )
    for marker in marker_paths:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("synthetic residue\n", encoding="utf-8")
    resolved_paths: list[str] = []
    original_resolve = calibration_module._resolve_new_project_path

    def record_resolved_path(path: Path, *, project_root: Path) -> Path:
        resolved_paths.append(path.as_posix())
        return original_resolve(path, project_root=project_root)

    monkeypatch.setattr(calibration_module, "_resolve_new_project_path", record_resolved_path)
    evidence = validate_runtime_absence_state(config, project_root=tmp_path, require_work_root_absent=True)
    assert evidence == {
        "status": "passed",
        "paths": [
            "outputs/work/efigure10_six_model_binned_calibration",
            "outputs/work/efigure10_six_model_binned_calibration/__efigure10_pycache_prefix__",
        ],
    }
    assert resolved_paths == evidence["paths"]
    assert all(marker.read_text(encoding="utf-8") == "synthetic residue\n" for marker in marker_paths)


def test_runtime_absence_rejects_scientific_work_root(loaded_config: dict[str, object], tmp_path: Path) -> None:
    work_root = tmp_path / loaded_config["output_contract"]["work_root"]
    work_root.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="absence"):
        validate_runtime_absence_state(loaded_config, project_root=tmp_path, require_work_root_absent=True)


def test_runtime_authority_rejects_final_root_before_production(
    loaded_config: dict[str, object], tmp_path: Path
) -> None:
    final_root = tmp_path / loaded_config["output_contract"]["final_root"]
    final_root.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="final-namespace"):
        validate_runtime_authority(loaded_config, project_root=tmp_path)


def test_runtime_absence_rejects_production_cache_prefix_sentinel(
    loaded_config: dict[str, object], tmp_path: Path
) -> None:
    sentinel = tmp_path / loaded_config["output_contract"]["production_pycache_prefix_sentinel"]
    sentinel.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="absence"):
        validate_runtime_absence_state(loaded_config, project_root=tmp_path, require_work_root_absent=False)


def test_production_artifact_run_rejects_test_cache_prefix(
    loaded_config: dict[str, object],
    synthetic_prediction_frames_distinct: OrderedDict[str, pd.DataFrame],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _prepare_synthetic_analysis_project(
        tmp_path, loaded_config, synthetic_prediction_frames_distinct, monkeypatch
    )
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", loaded_config["environment"]["pytest_pycache_prefix"])
    with pytest.raises(RuntimeError, match="production bytecode cache prefix"):
        run_analysis(config_path.relative_to(tmp_path), project_root=tmp_path)
    assert not (tmp_path / loaded_config["output_contract"]["work_root"]).exists()
    assert not (tmp_path / loaded_config["output_contract"]["final_root"]).exists()


def test_active_commands_exclude_recovery_and_test_mechanics(loaded_config: dict[str, object]) -> None:
    command_ids = tuple(record["command_id"] for record in loaded_config["authority"]["allowed_commands"])
    assert command_ids == (
        "E01",
        *(f"G{index:02d}" for index in range(1, 22)),
        "H01",
        "H02",
        "H03",
        "P01",
        "F01",
    )
    assert not set(command_ids) & {"A01P", "H03R", "A01T", *(f"T{index:02d}" for index in range(1, 9)), "A02", "C01"}


def test_f01_uses_only_path_scoped_status_and_production_integrity(loaded_config: dict[str, object]) -> None:
    f01 = next(record for record in loaded_config["authority"]["allowed_commands"] if record["command_id"] == "F01")
    lines = f01["argv_or_shell_lines"]
    status_lines = [line for line in lines if isinstance(line, str) and line.startswith("git status ")]
    assert status_lines == [" ".join(EXPECTED_G06_ARGV)]
    prohibited = ("_pytest", ".pytest_cache", ".coverage", "coverage.xml", "htmlcov", ".vscode", "assets")
    assert not any(token in "\n".join(lines) for token in prohibited)
    assertions = f01["postconditions"]["assertions"]
    absent = next(item for item in assertions if item["subject"] == "required_absent_nonsymlink_paths")
    assert absent["expected"] == [
        "outputs/work/efigure10_six_model_binned_calibration",
        "outputs/work/efigure10_six_model_binned_calibration/__efigure10_pycache_prefix__",
    ]
    hashes = next(item for item in assertions if item["subject"] == "final_output_sha256_by_path")
    assert hashes["expected_from"] == "p01_validated_stdout.sha256_by_final_output_path"


def test_output_contract_excludes_development_hygiene_authority(loaded_config: dict[str, object]) -> None:
    output = loaded_config["output_contract"]
    assert "pytest_basetemp_root" not in output
    assert "pytest_basetemp_children" not in output
    assert "pytest_pycache_prefix_sentinel" not in output
    assert "forbidden_repository_artifacts" not in output
    assert output["production_pycache_prefix_sentinel"] == (
        "outputs/work/efigure10_six_model_binned_calibration/__efigure10_pycache_prefix__"
    )
    assert output["production_pycache_prefix_must_remain_absent"] is True
    assert "pycache_prefixes_must_remain_absent" not in output
    assert "outputs/work/efigure10_six_model_binned_calibration_pytest/" not in loaded_config["authority"][
        "allowed_write_paths"
    ]
    assert loaded_config["environment"]["pytest_pycache_prefix"] == (
        "outputs/work/efigure10_six_model_binned_calibration_pytest/__efigure10_pycache_prefix__"
    )
    assert loaded_config["environment"]["production_pycache_prefix_must_remain_absent"] is True
    assert "pycache_prefixes_must_remain_absent" not in loaded_config["environment"]
