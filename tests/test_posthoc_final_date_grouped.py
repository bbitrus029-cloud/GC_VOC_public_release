"""Synthetic tests for the fixed date-grouped MiniRocket auxiliary validation."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
import os
import struct
from collections import OrderedDict
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from src.analysis import posthoc_final_date_grouped as date_grouped_module
from src.analysis.posthoc_final_common import ACCESS_STATE_KEYS, AccessStateTracker, parse_date_series_ascii_slash
from src.analysis.posthoc_final_date_grouped import (
    ANALYSIS_ID,
    FOLD_COLUMNS,
    SCORE_BLIND_PREFLIGHT_RELATIVE_PATH,
    VALIDATION_COLUMNS,
    Analysis3Inputs,
    CandidateArtifacts,
    assemble_oof_vector,
    build_fixed_partition,
    build_inestimable_candidate,
    canonical_oof_digest,
    classify_target_state,
    execute_approved,
    load_analysis3_inputs,
    paired_date_cluster_bootstrap,
    rank_auc,
    run_isolated_candidate,
    serialize_candidate_csvs,
    validate_analysis3_authority,
    validate_analysis3_outputs,
    validate_frozen_comparator,
    validate_locked_indices,
    validate_runtime_contract,
    verify_source_snapshot,
    write_deterministic_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG_PATH = PROJECT_ROOT / "configs" / "post_hoc_final_supplementary.yaml"


def test_module_imports_after_commit_b() -> None:
    """The Session 2 module imports against the consolidated runtime interfaces."""
    assert callable(date_grouped_module.execute_approved)


def test_current_config_loading_uses_one_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    """The consolidated config loader receives no retired overlay path."""
    calls: list[tuple[Path, ...]] = []
    expected: dict[str, Any] = {"package_id": "synthetic"}

    def _load(*args: Path) -> dict[str, Any]:
        calls.append(args)
        return expected

    monkeypatch.setattr(date_grouped_module, "load_effective_posthoc_config", _load)
    config_path = Path("configs/post_hoc_final_supplementary.yaml")

    assert date_grouped_module._load_current_config(config_path) is expected
    assert calls == [(config_path,)]


def _synthetic_historical_amendment() -> dict[str, Any]:
    """Return a complete synthetic historical Amendment record."""
    cumulative = OrderedDict((key, False) for key in ACCESS_STATE_KEYS)
    return {
        "commit_id": "b" * 40,
        "sole_parent": "a" * 40,
        "changed_paths_exact": ["history/config", "history/methods", "history/manifest"],
        "artifacts": [
            {"path": f"history/{name}", "sha256": character * 64, "git_blob_id": f"blob-{name}"}
            for name, character in (("config", "a"), ("methods", "b"), ("manifest", "c"))
        ],
        "approval_evidence": {"review_status": "approved", "user_confirmation": True},
        "access_state": {
            "step_2_approval_materialization": {"cumulative_through_session": cumulative}
        },
        "session_1_corrections": [
            {"id": f"C{index:02d}", "status": "implemented_in_session_1"}
            for index in range(1, 17)
        ],
    }


def test_historical_amendment_records_compare_exact_fields_access_and_corrections() -> None:
    """Historical config/manifest records must agree before Git-object validation."""
    amendment = _synthetic_historical_amendment()
    config = {"historical_consolidation": {"amendment_01": amendment}}
    manifest = {"historical_evidence": {"amendment_01": copy.deepcopy(amendment)}}

    assert date_grouped_module._validated_historical_amendment(config, manifest) is amendment

    for mutation in ("sole_parent", "access_state", "session_1_corrections"):
        changed = copy.deepcopy(manifest)
        changed_record = changed["historical_evidence"]["amendment_01"]
        if mutation == "sole_parent":
            changed_record[mutation] = "c" * 40
        elif mutation == "access_state":
            changed_record[mutation]["step_2_approval_materialization"]["cumulative_through_session"][
                "participant_level_date_values_read"
            ] = True
        else:
            changed_record[mutation][-1]["status"] = "missing"
        with pytest.raises(ValueError, match="Historical Amendment"):
            date_grouped_module._validated_historical_amendment(config, changed)


def test_analysis3_authority_separates_current_and_historical_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Current files and historical Git anchors validate through distinct interfaces."""
    amendment = _synthetic_historical_amendment()
    config = {
        "historical_consolidation": {"amendment_01": amendment},
        "outputs": {
            "session_0": {"freeze_manifest": "outputs/analysis/synthetic-manifest.json"}
        },
    }
    manifest = {"historical_evidence": {"amendment_01": copy.deepcopy(amendment)}}
    prior = OrderedDict((key, False) for key in ACCESS_STATE_KEYS)
    events: list[str] = []

    monkeypatch.setattr(date_grouped_module, "_load_current_config", lambda _: events.append("load") or config)
    monkeypatch.setattr(date_grouped_module, "_read_json", lambda _: manifest)

    def _current(*args: Any, **kwargs: Any) -> None:
        assert args[1:] == (config, manifest)
        assert kwargs["require_head_identical"] is True
        events.append("current")

    monkeypatch.setattr(date_grouped_module, "validate_current_runtime_authority", _current)
    monkeypatch.setattr(
        date_grouped_module,
        "_validate_analysis3_git_state",
        lambda *args, **kwargs: events.append("git"),
    )

    def _session0(*args: Any, **kwargs: Any) -> None:
        assert args[1:] == (config, manifest)
        assert set(kwargs) == {"git_runner"}
        events.append("session0")

    monkeypatch.setattr(date_grouped_module, "validate_freeze_commit", _session0)

    def _compare(*args: Any) -> dict[str, Any]:
        assert args == (config, manifest)
        events.append("compare")
        return amendment

    monkeypatch.setattr(date_grouped_module, "_validated_historical_amendment", _compare)

    def _amendment(commit: str, **kwargs: Any) -> None:
        assert commit == "b" * 40
        assert kwargs["historical_amendment"] is amendment
        assert set(kwargs) == {"historical_amendment", "git_runner"}
        events.append("amendment")

    monkeypatch.setattr(date_grouped_module, "validate_amendment_freeze_commit", _amendment)
    monkeypatch.setattr(
        date_grouped_module,
        "_amendment_cumulative_prior",
        lambda value: events.append("prior") or prior,
    )

    def _preflight(*args: Any) -> OrderedDict[str, bool]:
        assert args[0:2] == (config, manifest)
        events.append("preflight")
        return prior

    monkeypatch.setattr(date_grouped_module, "_validate_committed_preflight", _preflight)
    monkeypatch.setattr(
        date_grouped_module,
        "_validate_session4_completion",
        lambda *args: events.append("session4") or prior,
    )
    monkeypatch.setattr(
        date_grouped_module,
        "_validate_frozen_inputs",
        lambda value: events.append("frozen") or OrderedDict(),
    )
    monkeypatch.setattr(
        date_grouped_module,
        "validate_runtime_contract",
        lambda value: events.append("runtime"),
    )

    observed, hashes, cumulative = validate_analysis3_authority(
        BASE_CONFIG_PATH,
        "a" * 40,
        "b" * 40,
        git_runner=lambda *_args: "",
    )

    assert observed is config
    assert hashes == OrderedDict()
    assert cumulative == prior
    assert events == [
        "load",
        "current",
        "git",
        "session0",
        "compare",
        "amendment",
        "prior",
        "preflight",
        "session4",
        "frozen",
        "runtime",
    ]


def test_current_runtime_authority_uses_internally_consistent_synthetic_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Current authority validation succeeds only for matching temporary records."""
    relative_paths = {
        "config": "configs/post_hoc_final_supplementary.yaml",
        "methods": "docs/modeling/final_post_hoc_supplementary_methods.md",
        "manifest": "outputs/analysis/posthoc_final_freeze_manifest.json",
        "plan": "docs/plans/approved.plan.md",
        "guidance": "AGENTS.md",
    }
    for key, relative_path in relative_paths.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic-{key}".encode())
    digest = lambda key: hashlib.sha256((tmp_path / relative_paths[key]).read_bytes()).hexdigest()
    config = {
        "outputs": {
            "session_0": {
                "methods": relative_paths["methods"],
                "freeze_manifest": relative_paths["manifest"],
            }
        },
        "authority": {
            "reviewed_plan_path": relative_paths["plan"],
            "reviewed_plan_sha256": digest("plan"),
            "agent_guidance_path": relative_paths["guidance"],
            "agent_guidance_sha256": digest("guidance"),
        },
    }
    manifest = {
        "current_runtime_authority": {
            "config": {"path": relative_paths["config"], "sha256": digest("config")},
            "methods": {"path": relative_paths["methods"], "sha256": digest("methods")},
            "manifest": {
                "path": relative_paths["manifest"],
                "self_hash": None,
                "self_hash_policy": "forbidden",
            },
            "operational_plan": {"path": relative_paths["plan"], "sha256": digest("plan")},
            "agent_guidance": {
                "path": relative_paths["guidance"],
                "sha256": digest("guidance"),
            },
        }
    }
    monkeypatch.setattr("src.analysis.posthoc_final_locked_scores.PROJECT_ROOT", tmp_path)

    result = date_grouped_module.validate_current_runtime_authority(
        tmp_path / relative_paths["config"],
        config,
        manifest,
        require_head_identical=False,
    )
    assert all(result.values())

    (tmp_path / relative_paths["guidance"]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Agent guidance hash"):
        date_grouped_module.validate_current_runtime_authority(
            tmp_path / relative_paths["config"],
            config,
            manifest,
            require_head_identical=False,
        )


def test_historical_amendment_validation_uses_git_objects_without_live_files() -> None:
    """Historical validation consumes recorded paths and Git blobs only."""
    historical = _base_config()["historical_consolidation"]["amendment_01"]
    commit = historical["commit_id"]
    paths = historical["changed_paths_exact"]
    calls: list[tuple[str, ...]] = []

    def _git(*args: str) -> str:
        calls.append(args)
        if args == ("cat-file", "-e", f"{commit}^{{commit}}"):
            return ""
        if args == ("rev-list", "--parents", "-n", "1", commit):
            return f"{commit} {historical['sole_parent']}"
        if args == ("diff-tree", "--no-commit-id", "--name-only", "-r", commit):
            return "\n".join(paths)
        if args == ("merge-base", "--is-ancestor", commit, "HEAD"):
            return ""
        if args[0] == "rev-parse":
            path = args[1].split(":", maxsplit=1)[1]
            record = next(item for item in historical["artifacts"] if item["path"] == path)
            return record["git_blob_id"]
        raise AssertionError("Unexpected synthetic Git call")

    result = date_grouped_module.validate_amendment_freeze_commit(
        commit,
        historical_amendment=historical,
        git_runner=_git,
    )
    assert all(value is True for key, value in result.items() if key != "amendment_freeze_commit_id")
    assert not any(call[0] == "hash-object" for call in calls)


def test_preflight_hashes_bind_to_historical_session0_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The preserved preflight hash domain remains the original Session 0 bundle."""
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_bytes(b"synthetic preserved preflight")
    preflight_sha = hashlib.sha256(preflight_path.read_bytes()).hexdigest()
    historical_hashes = {
        "configs/post_hoc_final_supplementary.yaml": "a" * 64,
        "docs/modeling/final_post_hoc_supplementary_methods.md": "b" * 64,
        "outputs/analysis/posthoc_final_freeze_manifest.json": "c" * 64,
    }
    current_hashes = {key: "f" * 64 for key in historical_hashes}
    config = {
        "frozen_inputs": {"whole_container_sha256": {"synthetic/input": "d" * 64}},
        "outputs": {
            "session_0": {
                "config": "configs/post_hoc_final_supplementary.yaml",
                "methods": "docs/modeling/final_post_hoc_supplementary_methods.md",
                "freeze_manifest": "outputs/analysis/posthoc_final_freeze_manifest.json",
            },
            "scientific_and_closure": {
                "score_blind_preflight": {"path": SCORE_BLIND_PREFLIGHT_RELATIVE_PATH}
            },
        },
    }
    manifest = {
        "current_runtime_authority": {
            "config": {"sha256": current_hashes[config["outputs"]["session_0"]["config"]]},
            "methods": {"sha256": current_hashes[config["outputs"]["session_0"]["methods"]]},
        },
        "historical_evidence": {
            "original_session_0": {
                "artifacts": [
                    {"path": path, "sha256": digest}
                    for path, digest in historical_hashes.items()
                ]
            },
            "completed_session_1": {
                "artifact_sha256": {SCORE_BLIND_PREFLIGHT_RELATIVE_PATH: preflight_sha}
            },
        },
    }
    prior = OrderedDict((key, False) for key in ACCESS_STATE_KEYS)
    cumulative = OrderedDict((key, False) for key in ACCESS_STATE_KEYS)
    payload = {
        "hashes": {
            "config_sha256": historical_hashes[config["outputs"]["session_0"]["config"]],
            "methods_sha256": historical_hashes[config["outputs"]["session_0"]["methods"]],
            "freeze_manifest_sha256": historical_hashes[
                config["outputs"]["session_0"]["freeze_manifest"]
            ],
            "frozen_input_whole_container_sha256": {"synthetic/input": "d" * 64},
        },
        "status": "pass",
        "reason_code": None,
        "freeze_commit_validation": {"freeze_commit_id": "a" * 40},
        "one_to_one_alignment_status": "pass",
        "train_test_disjointness_status": "pass",
        "known_anchor_reconciliation": {"status": "pass"},
        "access_state": {"cumulative_through_session": cumulative},
    }
    monkeypatch.setattr(
        date_grouped_module,
        "_resolve",
        lambda value: preflight_path if value == SCORE_BLIND_PREFLIGHT_RELATIVE_PATH else tmp_path / str(value),
    )
    monkeypatch.setattr(date_grouped_module, "_read_json", lambda _: payload)
    monkeypatch.setattr(date_grouped_module, "validate_preflight_schema", lambda *args, **kwargs: None)
    monkeypatch.setattr(date_grouped_module, "validate_preflight_privacy", lambda *args, **kwargs: None)

    def _git(*args: str) -> str:
        if args[0] in {"ls-files", "rev-parse", "hash-object"}:
            return "same-blob"
        raise AssertionError("Unexpected synthetic Git call")

    assert date_grouped_module._validate_committed_preflight(
        config,
        manifest,
        "a" * 40,
        _git,
        prior,
    ) == cumulative

    payload["hashes"]["config_sha256"] = current_hashes[config["outputs"]["session_0"]["config"]]
    with pytest.raises(ValueError, match="historical Session 0"):
        date_grouped_module._validate_committed_preflight(
            config,
            manifest,
            "a" * 40,
            _git,
            prior,
        )


def test_source_integrity_inventory_uses_only_current_live_authorities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The snapshot covers current prerequisites without historical live paths."""
    frozen_path = "data/frozen.bin"
    central_path = "src/training/central.py"
    current_paths = [
        "configs/post_hoc_final_supplementary.yaml",
        "docs/modeling/final_post_hoc_supplementary_methods.md",
        "outputs/analysis/posthoc_final_freeze_manifest.json",
        "docs/plans/approved.plan.md",
        "AGENTS.md",
    ]
    preflight_path = SCORE_BLIND_PREFLIGHT_RELATIVE_PATH
    implementation_paths = ["src/analysis/implementation.py", "scripts/run.py", "tests/test_run.py"]
    session4_paths = [f"outputs/analysis/session4-{index}.csv" for index in range(4)]
    session4_log = "outputs/logs/session4.log"
    all_paths = [
        frozen_path,
        central_path,
        *current_paths,
        preflight_path,
        *implementation_paths,
        *session4_paths,
        session4_log,
    ]
    for index, relative_path in enumerate(all_paths):
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic-{index}".encode())
    digest = lambda relative: hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest()
    config = {
        "authority": {
            "reviewed_plan_path": current_paths[3],
            "agent_guidance_path": current_paths[4],
        },
        "frozen_inputs": {"whole_container_sha256": {frozen_path: digest(frozen_path)}},
        "source_integrity": {
            "analysis_3_central_training_sources": {central_path: digest(central_path)}
        },
        "outputs": {
            "session_0": {
                "config": current_paths[0],
                "methods": current_paths[1],
                "freeze_manifest": current_paths[2],
            },
            "future_implementation_paths": dict.fromkeys(implementation_paths, "required"),
            "scientific_and_closure": {
                "score_blind_preflight": {"path": preflight_path},
                "locked_score_metrics": {"path": session4_paths[0]},
                "locked_score_composition": {"path": session4_paths[1]},
                "qc5_correctness": {"path": session4_paths[2]},
                "within_date_concordance": {"path": session4_paths[3]},
            },
            "completion_logs": {"locked_scores": session4_log},
        },
    }
    monkeypatch.setattr(date_grouped_module, "PROJECT_ROOT", tmp_path)

    snapshot = date_grouped_module._source_integrity_snapshot(config)

    assert list(snapshot) == all_paths
    assert all(snapshot[path] == digest(path) for path in all_paths)
    assert all("amendment" not in path.lower() for path in snapshot)


def test_git_prerequisites_reject_changed_configured_session4_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configured prerequisite paths cannot diverge from the frozen inventory."""
    config = _base_config()
    config["outputs"]["scientific_and_closure"]["locked_score_metrics"]["path"] = (
        "outputs/analysis/synthetic-changed.csv"
    )
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(date_grouped_module, "_validate_annotated_tag_chain", lambda _: None)
    monkeypatch.setattr(
        date_grouped_module,
        "validate_execute_git_prerequisites",
        lambda required, **kwargs: calls.append(tuple(required)),
    )

    with pytest.raises(ValueError, match="Session 4 scientific outputs"):
        date_grouped_module._validate_analysis3_git_state(config, lambda *_args: "")
    assert calls == []


def _base_config() -> dict[str, Any]:
    with BASE_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _completion_log_factory(_: str, __: str, ___: str) -> bytes:
    return b"complete\n"


def _session4_prior() -> OrderedDict[str, bool]:
    return OrderedDict(
        (key, key not in {"auxiliary_model_fit_executed", "privacy_reference_loader_executed"})
        for key in ACCESS_STATE_KEYS
    )


def _synthetic_inputs() -> Analysis3Inputs:
    groups = np.repeat(np.arange(10), 6)
    labels = np.tile(np.array([0, 1, 0, 1, 0, 1], dtype=np.int64), 10)
    dates = np.array([date(2025, 1, int(group) + 1) for group in groups], dtype=object)
    x = np.arange(60 * 8 * 2, dtype=np.float64).reshape(60, 8, 2) / 1000.0
    random_scores = np.where(labels == 1, 0.75, 0.25).astype(np.float64)
    return Analysis3Inputs(
        x=x,
        y=labels,
        date_groups=dates,
        sample_ids=np.arange(1001, 1061, dtype=np.int64),
        frozen_random_oof_scores=random_scores,
        frozen_random_oof_auc=1.0,
    )


class _FitResult:
    def __init__(self, probabilities: np.ndarray) -> None:
        self.val_probabilities = probabilities
        self.transformer = object()
        self.normalizer = object()
        self.estimator = object()


class _TrainerDouble:
    instances: list["_TrainerDouble"] = []

    def __init__(self, global_config_path: Path, model_config_path: Path) -> None:
        del global_config_path, model_config_path
        self.model_config = {
            "minirocket": {"num_kernels": 10000, "random_state": 42},
            "ridge": {"class_weight": "balanced"},
            "calibration_layer_a": {"method": "sigmoid", "cv": 5, "ensemble": True},
        }
        self.random_seed = 42
        self.calls: list[tuple[np.ndarray, np.ndarray, int]] = []
        self.run_called = False
        self.save_called = False
        self.__class__.instances.append(self)

    def run(self) -> None:
        self.run_called = True
        raise AssertionError("Broad trainer run() is forbidden.")

    def save_outputs(self) -> None:
        self.save_called = True
        raise AssertionError("Trainer serialization is forbidden.")

    def fit_fold(
        self,
        outer_fold: int,
        inner_train_idx: np.ndarray,
        inner_val_idx: np.ndarray,
        hparams: dict[str, Any],
    ) -> _FitResult:
        assert hparams == {}
        self.calls.append((inner_train_idx.copy(), inner_val_idx.copy(), outer_fold))
        probabilities = 0.15 + 0.7 * self.y_train_[inner_val_idx].astype(np.float64)
        return _FitResult(probabilities)


def _trainer_factory(_config: dict[str, Any]) -> _TrainerDouble:
    return _TrainerDouble(Path("global.yaml"), Path("model.yaml"))


def test_amended_ascii_date_parser_accepts_only_slash_dates() -> None:
    parsed = parse_date_series_ascii_slash(pd.Series(["2025/1/2", "2025/11/29"], dtype="string"))
    assert parsed.tolist() == [pd.Timestamp("2025-01-02"), pd.Timestamp("2025-11-29")]

    forbidden = [
        "2025-01-02",
        " 2025/1/2",
        "2025/1/2 ",
        "2025/1/2 00:00",
        "2025/1/2/x",
        "２０２５/1/2",
        "2025/2/30",
    ]
    for value in forbidden:
        with pytest.raises(ValueError) as exc_info:
            parse_date_series_ascii_slash(pd.Series([value], dtype="string"))
        assert value not in str(exc_info.value)
    with pytest.raises(ValueError, match="normalization_collision_count=1"):
        parse_date_series_ascii_slash(pd.Series(["2025/1/2", "2025/01/02"], dtype="string"))


def test_fixed_splitter_contract_and_hard_coded_assignment() -> None:
    inputs = _synthetic_inputs()
    result = build_fixed_partition(inputs.y, inputs.date_groups)

    assert result.feasible is True
    assert result.reason is None
    assert result.splitter.shuffle is False
    assert result.splitter.random_state is None
    observed_groups = [
        tuple(sorted({(value - date(2025, 1, 1)).days for value in inputs.date_groups[fold.validation_idx]}))
        for fold in result.folds
    ]
    assert observed_groups == [(0, 5), (2, 6), (3, 7), (1, 8), (4, 9)]
    validation_positions = np.concatenate([fold.validation_idx for fold in result.folds])
    assert np.array_equal(np.sort(validation_positions), np.arange(inputs.y.size))
    for fold in result.folds:
        assert set(inputs.date_groups[fold.fit_idx]).isdisjoint(inputs.date_groups[fold.validation_idx])


def test_splitter_integrity_failures_are_hard_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _synthetic_inputs()

    class _BadSplitter:
        n_splits = 5
        shuffle = False
        random_state = None

        def __init__(self, n_splits: int, shuffle: bool) -> None:
            assert n_splits == 5
            assert shuffle is False

        def split(self, x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Any:
            del x, y, groups
            for _ in range(5):
                yield np.arange(50), np.arange(10)

    monkeypatch.setattr("src.analysis.posthoc_final_date_grouped.StratifiedGroupKFold", _BadSplitter)
    with pytest.raises(RuntimeError, match="validation coverage"):
        build_fixed_partition(inputs.y, inputs.date_groups)


def test_feasibility_reasons_are_closed_and_prevent_fit() -> None:
    inputs = _synthetic_inputs()
    too_few_dates = replace(inputs, date_groups=np.repeat(np.arange(4), 15))
    result = build_fixed_partition(too_few_dates.y, too_few_dates.date_groups)
    assert result.feasible is False
    assert result.reason == "insufficient_unique_date_groups"
    assert len(result.fold_rows) == 5

    degenerate_y = np.repeat(np.array([0, 1, 0, 1, 0]), 12)
    degenerate_groups = np.repeat(np.arange(5), 12)
    result = build_fixed_partition(degenerate_y, degenerate_groups)
    assert result.feasible is False
    assert result.reason == "fixed_partition_class_degenerate"

    sparse_positive_y = np.zeros(50, dtype=np.int64)
    sparse_positive_y[[0, 10, 20, 30, 40]] = 1
    result = build_fixed_partition(sparse_positive_y, np.repeat(np.arange(10), 5))
    assert result.feasible is False
    assert result.reason == "outer_fit_calibration_class_count_below_5"

    registered_reasons = set(_base_config()["status_and_reason_registries"]["scientific_result_reason_registry"])
    assert {
        "insufficient_unique_date_groups",
        "fixed_partition_class_degenerate",
        "outer_fit_calibration_class_count_below_5",
    }.issubset(registered_reasons)


def test_inestimable_candidate_retains_exact_row_inventory_without_fit() -> None:
    inputs = replace(_synthetic_inputs(), date_groups=np.repeat(np.arange(4), 15))
    partition = build_fixed_partition(inputs.y, inputs.date_groups)
    artifact = build_inestimable_candidate(inputs, partition, _base_config())

    assert artifact.validation.shape == (1, len(VALIDATION_COLUMNS))
    assert artifact.fold_audit.shape == (5, len(FOLD_COLUMNS))
    row = artifact.validation.iloc[0]
    assert row["n_bootstrap_valid"] == 0
    assert row["n_bootstrap_attempts"] == 0
    assert pd.isna(row["bootstrap_seed"])
    assert pd.isna(row["oof_vector_sha256"])
    assert row["date_grouped_auc_nonestimable_reason"] == "insufficient_unique_date_groups"


def test_fixed_partition_invokes_installed_splitter_once(monkeypatch: pytest.MonkeyPatch) -> None:
    real_splitter = date_grouped_module.StratifiedGroupKFold
    calls = {"construct": 0, "split": 0}

    class _CountingSplitter:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls["construct"] += 1
            self._delegate = real_splitter(*args, **kwargs)
            self.shuffle = self._delegate.shuffle
            self.random_state = self._delegate.random_state

        def split(self, *args: Any, **kwargs: Any) -> Any:
            calls["split"] += 1
            return self._delegate.split(*args, **kwargs)

    monkeypatch.setattr(date_grouped_module, "StratifiedGroupKFold", _CountingSplitter)
    inputs = _synthetic_inputs()
    partition = build_fixed_partition(inputs.y, inputs.date_groups)
    assert partition.feasible
    assert calls == {"construct": 1, "split": 1}


def test_minirocket_adapter_makes_exactly_five_outer_calls() -> None:
    _TrainerDouble.instances.clear()
    inputs = _synthetic_inputs()
    partition = build_fixed_partition(inputs.y, inputs.date_groups)
    artifact = run_isolated_candidate(
        inputs,
        partition,
        _base_config(),
        trainer_factory=_trainer_factory,
    )

    trainer = _TrainerDouble.instances[-1]
    assert len(trainer.calls) == 5
    assert [call[2] for call in trainer.calls] == [1, 2, 3, 4, 5]
    assert trainer.current_class_weights_ == "balanced"
    assert trainer.model_config["minirocket"] == {"num_kernels": 10000, "random_state": 42}
    assert trainer.model_config["calibration_layer_a"] == {"method": "sigmoid", "cv": 5, "ensemble": True}
    assert trainer.run_called is False
    assert trainer.save_called is False
    assert artifact.validation.iloc[0]["date_grouped_auc_status"] == "estimable"


def test_minirocket_adapter_cross_checks_configured_trainer_contract() -> None:
    """The consolidated Analysis 3 trainer settings remain runtime authority."""
    _TrainerDouble.instances.clear()
    inputs = _synthetic_inputs()
    partition = build_fixed_partition(inputs.y, inputs.date_groups)
    changed = _base_config()
    changed["analysis_3_date_grouped_minirocket"]["trainer"]["minirocket"]["num_kernels"] = 9999

    with pytest.raises(RuntimeError, match="MiniRocket configuration"):
        run_isolated_candidate(inputs, partition, changed, trainer_factory=_trainer_factory)
    assert len(_TrainerDouble.instances) == 1
    assert _TrainerDouble.instances[0].calls == []


def test_two_isolated_candidates_make_exactly_ten_outer_calls() -> None:
    """Two fresh candidates each execute the fixed five-fold partition once."""
    _TrainerDouble.instances.clear()
    inputs = _synthetic_inputs()
    partition = build_fixed_partition(inputs.y, inputs.date_groups)

    first = run_isolated_candidate(inputs, partition, _base_config(), trainer_factory=_trainer_factory)
    second = run_isolated_candidate(inputs, partition, _base_config(), trainer_factory=_trainer_factory)

    assert len(_TrainerDouble.instances) == 2
    assert _TrainerDouble.instances[0] is not _TrainerDouble.instances[1]
    assert [len(trainer.calls) for trainer in _TrainerDouble.instances] == [5, 5]
    assert sum(len(trainer.calls) for trainer in _TrainerDouble.instances) == 10
    assert serialize_candidate_csvs(first) == serialize_candidate_csvs(second)
    assert first.validation.equals(second.validation)
    assert first.oof_digest == second.oof_digest


def test_two_prefit_nonestimable_candidates_make_zero_outer_calls() -> None:
    """The complete two-candidate invocation fits nothing after pre-fit failure."""
    _TrainerDouble.instances.clear()
    inputs = replace(_synthetic_inputs(), date_groups=np.repeat(np.arange(4), 15))
    partition = build_fixed_partition(inputs.y, inputs.date_groups)

    first = run_isolated_candidate(inputs, partition, _base_config(), trainer_factory=_trainer_factory)
    second = run_isolated_candidate(inputs, partition, _base_config(), trainer_factory=_trainer_factory)

    assert _TrainerDouble.instances == []
    assert first.oof_digest is None and second.oof_digest is None
    assert serialize_candidate_csvs(first) == serialize_candidate_csvs(second)


def test_future_session5_access_states_for_feasible_and_prefit_nonestimable() -> None:
    expected_feasible = OrderedDict(
        [
            ("prediction_container_bytes_hashed", True),
            ("prediction_schema_or_header_inspected", True),
            ("participant_level_label_values_read", True),
            ("participant_level_score_values_read", True),
            ("participant_level_date_values_read", True),
            ("participant_level_qc_values_read", False),
            ("new_score_derived_results_computed", True),
            ("auxiliary_model_fit_executed", True),
            ("privacy_reference_loader_executed", False),
        ]
    )
    tracker = AccessStateTracker(_session4_prior())
    for key, value in expected_feasible.items():
        if value and key != "auxiliary_model_fit_executed":
            tracker.mark(key)
    inputs = _synthetic_inputs()
    partition = build_fixed_partition(inputs.y, inputs.date_groups)
    run_isolated_candidate(inputs, partition, _base_config(), trainer_factory=_trainer_factory, tracker=tracker)
    assert tracker.snapshot()["session_activity"] == expected_feasible
    assert tracker.snapshot()["cumulative_through_session"] == OrderedDict(
        (key, _session4_prior()[key] or expected_feasible[key]) for key in ACCESS_STATE_KEYS
    )

    expected_nonestimable = OrderedDict(expected_feasible)
    expected_nonestimable["auxiliary_model_fit_executed"] = False
    nonestimable_tracker = AccessStateTracker(_session4_prior())
    for key, value in expected_nonestimable.items():
        if value:
            nonestimable_tracker.mark(key)
    limited = replace(inputs, date_groups=np.repeat(np.arange(4), 15))
    limited_partition = build_fixed_partition(limited.y, limited.date_groups)
    _TrainerDouble.instances.clear()
    run_isolated_candidate(
        limited,
        limited_partition,
        _base_config(),
        trainer_factory=_trainer_factory,
        tracker=nonestimable_tracker,
    )
    assert not _TrainerDouble.instances
    assert nonestimable_tracker.snapshot()["session_activity"] == expected_nonestimable
    assert nonestimable_tracker.snapshot()["cumulative_through_session"] == OrderedDict(
        (key, _session4_prior()[key] or expected_nonestimable[key]) for key in ACCESS_STATE_KEYS
    )


def test_production_builds_one_partition_and_reuses_it_for_both_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _synthetic_inputs()
    config = _base_config()
    artifact = build_inestimable_candidate(
        replace(inputs, date_groups=np.repeat(np.arange(4), 15)),
        build_fixed_partition(inputs.y, np.repeat(np.arange(4), 15)),
        config,
    )
    partition = object()
    builds: list[tuple[np.ndarray, np.ndarray]] = []
    received: list[object] = []

    monkeypatch.setattr(
        date_grouped_module,
        "validate_analysis3_authority",
        lambda *a, **k: (config, OrderedDict(), _session4_prior()),
    )
    monkeypatch.setattr(date_grouped_module, "_source_integrity_snapshot", lambda _: OrderedDict())
    monkeypatch.setattr(date_grouped_module, "load_analysis3_inputs", lambda *a, **k: inputs)

    def _build(y: np.ndarray, groups: np.ndarray) -> object:
        builds.append((y, groups))
        return partition

    def _candidate(
        candidate_inputs: Analysis3Inputs,
        candidate_partition: object,
        candidate_config: dict[str, Any],
        **kwargs: Any,
    ) -> CandidateArtifacts:
        del candidate_inputs, candidate_config, kwargs
        received.append(candidate_partition)
        return artifact

    monkeypatch.setattr(date_grouped_module, "build_fixed_partition", _build)
    monkeypatch.setattr(date_grouped_module, "run_isolated_candidate", _candidate)
    monkeypatch.setattr(date_grouped_module, "validate_analysis3_outputs", lambda *a, **k: None)
    del tmp_path
    targets = (
        PROJECT_ROOT / "outputs/analysis/synthetic-validation.csv",
        PROJECT_ROOT / "outputs/analysis/synthetic-folds.csv",
        PROJECT_ROOT / "outputs/logs/synthetic-completion.log",
    )
    monkeypatch.setattr(date_grouped_module, "_date_grouped_target_paths", lambda _: targets)
    monkeypatch.setattr(date_grouped_module, "write_deterministic_outputs", lambda *a, **k: targets)

    assert execute_approved(BASE_CONFIG_PATH, "a" * 40, "b" * 40) == targets
    assert len(builds) == 1
    assert received == [partition, partition]


def test_real_fit_fold_normalization_uses_fitting_rows_only(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.training import train_minirocket_ridge as trainer_module

    class _TransformerDouble:
        def __init__(self, num_kernels: int, random_state: int) -> None:
            assert num_kernels == 10000
            assert random_state == 42

        def fit_transform(self, x: np.ndarray) -> pd.DataFrame:
            return pd.DataFrame({"feature": x[:, 0, 0].astype(float)})

        def transform(self, x: np.ndarray) -> pd.DataFrame:
            return pd.DataFrame({"feature": x[:, 0, 0].astype(float)})

    class _EstimatorDouble:
        def __init__(self) -> None:
            self.fit_matrix: np.ndarray | None = None
            self.validation_matrix: np.ndarray | None = None

        def fit(self, x: pd.DataFrame, y: np.ndarray) -> "_EstimatorDouble":
            del y
            self.fit_matrix = x.to_numpy(dtype=float)
            return self

        def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
            self.validation_matrix = x.to_numpy(dtype=float)
            return np.tile(np.array([[0.5, 0.5]]), (len(x), 1))

    monkeypatch.setattr(trainer_module, "MiniRocketMultivariate", _TransformerDouble)
    monkeypatch.setattr(trainer_module, "_ensure_numba_cache_dir", lambda: None)
    trainer = trainer_module.MiniRocketRidgeTrainer(
        global_config_path=PROJECT_ROOT / "configs" / "modeling_global.yaml",
        model_config_path=PROJECT_ROOT / "configs" / "modeling_minirocket_ridge.yaml",
    )
    values = np.array([0.0, 2.0, 4.0, 6.0, 1000.0, 1002.0])
    trainer.x_train_ = np.repeat(values[:, None, None], 4, axis=1)
    trainer.y_train_ = np.array([0, 1, 0, 1, 0, 1])
    trainer.current_class_weights_ = "balanced"
    estimator = _EstimatorDouble()
    monkeypatch.setattr(trainer, "build_estimator", lambda hparams, class_weights: estimator)

    result = trainer.fit_fold(
        outer_fold=1,
        inner_train_idx=np.arange(4),
        inner_val_idx=np.arange(4, 6),
        hparams={},
    )

    assert result["normalizer"].mean_.tolist() == [3.0]
    assert estimator.fit_matrix is not None
    assert estimator.fit_matrix.mean() == pytest.approx(0.0)
    assert estimator.validation_matrix is not None
    assert estimator.validation_matrix.min() > 100.0


def test_canonical_digest_has_independent_little_endian_anchor() -> None:
    values = np.array([0.125, 0.5, 0.875], dtype=np.float64)
    independently_packed = struct.pack("<3d", 0.125, 0.5, 0.875)
    assert hashlib.sha256(independently_packed).hexdigest() == (
        "114bd39a0442d4291e61c43455e716c1af8f041c38ab8b8e3f4b4398f86d29c8"
    )
    assert canonical_oof_digest(values, expected_length=3) == (
        "114bd39a0442d4291e61c43455e716c1af8f041c38ab8b8e3f4b4398f86d29c8"
    )
    assert canonical_oof_digest(values.byteswap().view(values.dtype.newbyteorder()), expected_length=3) == (
        "114bd39a0442d4291e61c43455e716c1af8f041c38ab8b8e3f4b4398f86d29c8"
    )


@pytest.mark.parametrize(
    ("values", "positions", "message"),
    [
        ([0.1, 0.2], [0, 0], "positions"),
        ([0.1, 0.2], [1, 0], "order"),
        ([0.1], [0], "length"),
        ([0.1, np.nan], [0, 1], "finite"),
        ([0.1, np.inf], [0, 1], "finite"),
        ([0.1, 1.1], [0, 1], "probability"),
    ],
)
def test_oof_assembly_rejects_invalid_vectors(
    values: list[float],
    positions: list[int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        assemble_oof_vector(
            np.asarray(values, dtype=np.float64),
            np.asarray(positions, dtype=np.int64),
            expected_length=2,
        )


def test_rank_auc_awards_half_credit_to_ties() -> None:
    assert rank_auc(np.array([0, 1]), np.array([0.5, 0.5])) == 0.5


def test_paired_cluster_bootstrap_is_fixed_and_estimable(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _synthetic_inputs()
    observed_seeds: list[int] = []
    real_default_rng = np.random.default_rng

    def _recording_rng(seed: int) -> np.random.Generator:
        observed_seeds.append(seed)
        return real_default_rng(seed)

    monkeypatch.setattr("src.analysis.posthoc_final_date_grouped.np.random.default_rng", _recording_rng)
    result = paired_date_cluster_bootstrap(
        inputs.y,
        inputs.date_groups,
        0.2 + 0.6 * inputs.y,
        inputs.frozen_random_oof_scores,
    )
    assert observed_seeds == [42]
    assert result.n_valid == 2000
    assert result.n_attempts == 2000
    assert result.seed == 42
    assert result.date_auc_ci is not None
    assert result.delta_ci is not None
    assert result.nonestimable_reason is None


def test_paired_cluster_bootstrap_below_target_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _synthetic_inputs()

    class _CollapsedRng:
        def choice(self, values: np.ndarray, size: int, replace: bool) -> np.ndarray:
            assert replace is True
            return np.repeat(values[0], size)

    monkeypatch.setattr(
        "src.analysis.posthoc_final_date_grouped.np.random.default_rng",
        lambda seed: _CollapsedRng(),
    )
    result = paired_date_cluster_bootstrap(
        inputs.y,
        inputs.y.astype(object),
        0.2 + 0.6 * inputs.y,
        inputs.frozen_random_oof_scores,
    )
    assert result.n_valid == 0
    assert result.n_attempts == 20000
    assert result.date_auc_ci is None
    assert result.delta_ci is None
    assert result.nonestimable_reason == "bootstrap_valid_replicates_below_2000"


def test_paired_bootstrap_reuses_cluster_multiplicities_and_linear_quantiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each paired score uses identical sampled rows and linear percentile limits."""
    labels = np.tile(np.array([0, 1], dtype=np.int64), 3)
    groups = np.repeat(np.arange(3), 2).astype(object)
    date_scores = np.arange(6, dtype=np.float64) / 10.0
    random_scores = 10.0 + np.arange(6, dtype=np.float64)

    class _SequenceRng:
        def __init__(self) -> None:
            self.draws = iter((np.array([0, 0, 2]), np.array([2, 1, 1])))

        def choice(self, values: np.ndarray, size: int, replace: bool) -> np.ndarray:
            assert np.array_equal(values, np.arange(3))
            assert size == 3 and replace is True
            return next(self.draws)

    real_rank_auc = date_grouped_module.rank_auc
    score_calls: list[np.ndarray] = []

    def _capture_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
        score_calls.append(np.asarray(scores).copy())
        return real_rank_auc(y_true, scores)

    real_quantile = np.quantile
    quantile_methods: list[str] = []

    def _capture_quantile(values: np.ndarray, quantiles: list[float], *, method: str) -> np.ndarray:
        quantile_methods.append(method)
        return real_quantile(values, quantiles, method=method)

    monkeypatch.setattr(date_grouped_module, "BOOTSTRAP_VALID_TARGET", 2)
    monkeypatch.setattr(date_grouped_module, "BOOTSTRAP_ATTEMPT_CAP", 3)
    monkeypatch.setattr(date_grouped_module.np.random, "default_rng", lambda seed: _SequenceRng())
    monkeypatch.setattr(date_grouped_module, "rank_auc", _capture_auc)
    monkeypatch.setattr(date_grouped_module.np, "quantile", _capture_quantile)

    result = paired_date_cluster_bootstrap(labels, groups, date_scores, random_scores)

    assert result.n_valid == 2 and result.n_attempts == 2
    assert len(score_calls) == 4
    for date_call, random_call in zip(score_calls[::2], score_calls[1::2], strict=True):
        sampled_positions = np.rint(date_call * 10.0).astype(int)
        assert np.array_equal(random_call, 10.0 + sampled_positions)
    assert quantile_methods == ["linear", "linear"]


def test_frozen_comparator_requires_unique_aggregate_source_and_exact_alignment() -> None:
    inputs = _synthetic_inputs()
    summary = pd.DataFrame(
        [{"scope": "aggregate", "outer_fold": "aggregate", "metric": "auc_roc", "value": 1.0}]
    )
    observed = validate_frozen_comparator(inputs.y, inputs.frozen_random_oof_scores, summary)
    assert observed == 1.0

    with pytest.raises(ValueError, match="exactly one"):
        validate_frozen_comparator(
            inputs.y,
            inputs.frozen_random_oof_scores,
            pd.concat([summary, summary], ignore_index=True),
        )
    mismatch = summary.copy()
    mismatch.loc[0, "value"] = 0.8
    with pytest.raises(ValueError, match="agreement"):
        validate_frozen_comparator(inputs.y, inputs.frozen_random_oof_scores, mismatch)
    bad_scores = inputs.frozen_random_oof_scores.copy()
    bad_scores[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_frozen_comparator(inputs.y, bad_scores, summary)


def test_output_schemas_and_cross_field_mutations() -> None:
    inputs = _synthetic_inputs()
    partition = build_fixed_partition(inputs.y, inputs.date_groups)
    artifact = run_isolated_candidate(inputs, partition, _base_config(), trainer_factory=_trainer_factory)
    validate_analysis3_outputs(artifact.validation, artifact.fold_audit, _base_config(), expected_n=60)
    assert list(artifact.validation.columns) == VALIDATION_COLUMNS
    assert list(artifact.fold_audit.columns) == FOLD_COLUMNS
    assert artifact.fold_audit["fold_id"].tolist() == [1, 2, 3, 4, 5]
    assert str(artifact.validation["split_seed"].dtype) == "Int64"
    assert artifact.validation.iloc[0]["split_seed_status"] == "not_applicable"

    mutations = {
        "shuffle": True,
        "split_seed": 42,
        "split_seed_status": "applicable",
        "oof_vector_sha256": None,
        "date_grouped_auc_status": "non_estimable",
        "date_grouped_auc_ci_status": "non_estimable",
        "auc_difference_status": "non_estimable",
        "auc_difference_ci_status": "non_estimable",
    }
    for column, value in mutations.items():
        changed = artifact.validation.copy()
        if isinstance(changed[column].dtype, pd.CategoricalDtype):
            changed[column] = changed[column].astype("string")
        changed.loc[0, column] = value
        with pytest.raises((TypeError, ValueError)):
            validate_analysis3_outputs(changed, artifact.fold_audit, _base_config(), expected_n=60)

    changed_folds = artifact.fold_audit.copy()
    changed_folds.loc[0, "date_overlap_count"] = 1
    with pytest.raises(ValueError):
        validate_analysis3_outputs(artifact.validation, changed_folds, _base_config(), expected_n=60)
    changed_folds = artifact.fold_audit.copy()
    changed_folds.loc[0, "calibrator_fit_feasible"] = False
    with pytest.raises(ValueError):
        validate_analysis3_outputs(artifact.validation, changed_folds, _base_config(), expected_n=60)


def test_candidate_serialization_is_byte_deterministic() -> None:
    inputs = _synthetic_inputs()
    partition = build_fixed_partition(inputs.y, inputs.date_groups)
    first = run_isolated_candidate(inputs, partition, _base_config(), trainer_factory=_trainer_factory)
    second = run_isolated_candidate(inputs, partition, _base_config(), trainer_factory=_trainer_factory)
    first_bytes = serialize_candidate_csvs(first)
    second_bytes = serialize_candidate_csvs(second)
    assert first_bytes == second_bytes
    assert first.oof_digest == second.oof_digest


def test_target_state_distinguishes_clean_stale_and_partial(tmp_path: Path) -> None:
    validation = tmp_path / "validation.csv"
    folds = tmp_path / "folds.csv"
    log = tmp_path / "completion.log"
    assert classify_target_state(validation, folds, log) == "clean"
    validation.write_text("partial", encoding="utf-8")
    assert classify_target_state(validation, folds, log) == "partial_interrupted_promotion"
    log.write_text("receipt", encoding="utf-8")
    assert classify_target_state(validation, folds, log) == "stale_or_complete"


def test_deterministic_mismatch_or_source_mutation_promotes_nothing(tmp_path: Path) -> None:
    inputs = _synthetic_inputs()
    partition = build_fixed_partition(inputs.y, inputs.date_groups)
    artifact = run_isolated_candidate(inputs, partition, _base_config(), trainer_factory=_trainer_factory)
    mismatched = CandidateArtifacts(
        validation=artifact.validation.assign(model_seed=7),
        fold_audit=artifact.fold_audit,
        oof_digest=artifact.oof_digest,
    )
    targets = (tmp_path / "validation.csv", tmp_path / "folds.csv", tmp_path / "completion.log")
    log_generations: list[str] = []

    def _counted_factory(*_: str) -> bytes:
        log_generations.append("generated")
        return b"complete\n"

    with pytest.raises(RuntimeError, match="deterministic"):
        write_deterministic_outputs(
            artifact,
            mismatched,
            *targets,
            config=_base_config(),
            source_entry_hashes={"source.py": "a" * 64},
            source_exit_hashes={"source.py": "a" * 64},
            completion_log_factory=_counted_factory,
        )
    assert not any(path.exists() for path in targets)
    assert not log_generations

    with pytest.raises(RuntimeError, match="source integrity"):
        write_deterministic_outputs(
            artifact,
            artifact,
            *targets,
            config=_base_config(),
            source_entry_hashes={"source.py": "a" * 64},
            source_exit_hashes={"source.py": "b" * 64},
            completion_log_factory=_counted_factory,
        )
    assert not any(path.exists() for path in targets)
    assert not log_generations


def test_staged_registry_failure_precedes_log_generation_and_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _synthetic_inputs()
    partition = build_fixed_partition(inputs.y, inputs.date_groups)
    artifact = run_isolated_candidate(inputs, partition, _base_config(), trainer_factory=_trainer_factory)
    targets = (tmp_path / "validation.csv", tmp_path / "folds.csv", tmp_path / "completion.log")
    log_generations: list[str] = []
    monkeypatch.setattr(date_grouped_module, "registry_readback_matches", lambda *a, **k: False)

    def _counted_factory(*_: str) -> bytes:
        log_generations.append("generated")
        return b"complete\n"

    with pytest.raises(RuntimeError, match="registry read-back"):
        write_deterministic_outputs(
            artifact,
            artifact,
            *targets,
            config=_base_config(),
            source_entry_hashes={"source.py": "a" * 64},
            source_exit_hashes={"source.py": "a" * 64},
            completion_log_factory=_counted_factory,
        )
    assert not log_generations
    assert not any(path.exists() for path in targets)
    assert not list(tmp_path.glob("*.tmp"))


def test_output_first_log_last_and_no_temp_leftovers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _synthetic_inputs()
    partition = build_fixed_partition(inputs.y, inputs.date_groups)
    artifact = run_isolated_candidate(inputs, partition, _base_config(), trainer_factory=_trainer_factory)
    targets = (tmp_path / "validation.csv", tmp_path / "folds.csv", tmp_path / "completion.log")
    order: list[str] = []
    real_replace = os.replace

    def _record_replace(source: str | Path, destination: str | Path) -> None:
        order.append(Path(destination).name)
        real_replace(source, destination)

    monkeypatch.setattr("src.analysis.posthoc_final_date_grouped.os.replace", _record_replace)
    write_deterministic_outputs(
        artifact,
        artifact,
        *targets,
        config=_base_config(),
        source_entry_hashes={"source.py": "a" * 64},
        source_exit_hashes={"source.py": "a" * 64},
        completion_log_factory=_completion_log_factory,
    )
    assert order == ["validation.csv", "folds.csv", "completion.log"]
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    ("failing_target", "expected_present"),
    [
        ("folds.csv", {"validation.csv"}),
        ("completion.log", {"validation.csv", "folds.csv"}),
    ],
)
def test_promotion_failure_retains_partial_scientific_state_without_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_target: str,
    expected_present: set[str],
) -> None:
    inputs = _synthetic_inputs()
    partition = build_fixed_partition(inputs.y, inputs.date_groups)
    artifact = run_isolated_candidate(inputs, partition, _base_config(), trainer_factory=_trainer_factory)
    targets = (tmp_path / "validation.csv", tmp_path / "folds.csv", tmp_path / "completion.log")
    real_replace = os.replace

    def _failing_replace(source: str | Path, destination: str | Path) -> None:
        if Path(destination).name == failing_target:
            raise OSError("simulated promotion failure")
        real_replace(source, destination)

    monkeypatch.setattr(date_grouped_module.os, "replace", _failing_replace)
    with pytest.raises(OSError, match="simulated"):
        write_deterministic_outputs(
            artifact,
            artifact,
            *targets,
            config=_base_config(),
            source_entry_hashes={"source.py": "a" * 64},
            source_exit_hashes={"source.py": "a" * 64},
            completion_log_factory=_completion_log_factory,
        )
    assert {path.name for path in targets if path.exists()} == expected_present
    assert not targets[2].exists()
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(RuntimeError, match="not absent"):
        write_deterministic_outputs(
            artifact,
            artifact,
            *targets,
            config=_base_config(),
            source_entry_hashes={"source.py": "a" * 64},
            source_exit_hashes={"source.py": "a" * 64},
            completion_log_factory=_completion_log_factory,
        )


def test_source_snapshot_detects_changes_without_echoing_private_content(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("safe = 1\n", encoding="utf-8")
    snapshot = {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
    verify_source_snapshot(snapshot)
    source.write_text("private-sentinel\n", encoding="utf-8")
    with pytest.raises(RuntimeError) as exc_info:
        verify_source_snapshot(snapshot)
    assert "private-sentinel" not in str(exc_info.value)


def test_minimal_loader_uses_exact_columns_and_preserves_locked_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _base_config()
    analysis = config["analysis_3_date_grouped_minirocket"]
    sample_index = tmp_path / "sample_index.csv"
    pipeline = tmp_path / "pipeline.npz"
    split = tmp_path / "split.npz"
    oof = tmp_path / "oof.csv"
    summary = tmp_path / "summary.csv"
    ids = np.arange(1001, 1973, dtype=np.int64)
    labels = np.tile(np.array([0, 1], dtype=np.int64), 486)
    dates = [f"2025/{month}/{day}" for month in range(1, 13) for day in range(1, 28)]
    pd.DataFrame(
        {
            "sample_id": ids,
            "label": labels,
            "collection_time": [dates[index % len(dates)] for index in range(972)],
            "in_final_output": True,
        }
    ).to_csv(sample_index, index=False)
    x = np.zeros((972, 2, 2), dtype=np.float64)
    np.savez(pipeline, x=x, sample_id=ids, label=labels)
    train_idx = np.arange(777, dtype=np.int64)
    test_idx = np.arange(777, 972, dtype=np.int64)
    np.savez(split, train_idx=train_idx, test_idx=test_idx)
    train_labels = labels[train_idx]
    scores = 0.2 + 0.6 * train_labels
    pd.DataFrame(
        {
            "sample_id": ids[train_idx],
            "y_true": train_labels,
            "p_raw": scores,
            "outer_fold": np.tile(np.arange(1, 6), 156)[:777],
        }
    ).to_csv(oof, index=False)
    pd.DataFrame(
        [{"scope": "aggregate", "outer_fold": "aggregate", "metric": "auc_roc", "value": 1.0}]
    ).to_csv(summary, index=False)
    analysis["expected_pipeline_shape"] = [972, 2, 2]

    observed_usecols: list[tuple[str, ...]] = []
    real_read_csv = pd.read_csv

    def _record_csv(*args: Any, **kwargs: Any) -> pd.DataFrame:
        if "usecols" in kwargs:
            observed_usecols.append(tuple(kwargs["usecols"]))
        return real_read_csv(*args, **kwargs)

    monkeypatch.setattr("src.analysis.posthoc_final_date_grouped.pd.read_csv", _record_csv)
    tracker = AccessStateTracker(OrderedDict((key, False) for key in ACCESS_STATE_KEYS))
    loaded = load_analysis3_inputs(
        config,
        sample_index_path=sample_index,
        pipeline_path=pipeline,
        split_path=split,
        oof_path=oof,
        summary_path=summary,
        tracker=tracker,
    )
    assert loaded.sample_ids.tolist() == ids[train_idx].tolist()
    assert observed_usecols == [
        ("sample_id", "label", "collection_time", "in_final_output"),
        ("sample_id", "y_true", "p_raw"),
        ("scope", "outer_fold", "metric", "value"),
    ]
    assert tracker.snapshot()["session_activity"] == OrderedDict(
        [
            ("prediction_container_bytes_hashed", False),
            ("prediction_schema_or_header_inspected", True),
            ("participant_level_label_values_read", True),
            ("participant_level_score_values_read", True),
            ("participant_level_date_values_read", True),
            ("participant_level_qc_values_read", False),
            ("new_score_derived_results_computed", True),
            ("auxiliary_model_fit_executed", False),
            ("privacy_reference_loader_executed", False),
        ]
    )
    assert "collection_location" not in inspect.getsource(load_analysis3_inputs)


def test_loader_rejects_test_overlap_before_candidate_fit(tmp_path: Path) -> None:
    source = inspect.getsource(load_analysis3_inputs)
    assert "test_predictions" not in source
    assert "internal_test" not in source
    assert "load_data(" not in source
    assert "_load_split_state(" not in source


@pytest.mark.parametrize(
    ("train_idx", "test_idx", "message"),
    [
        (np.array([0, 0, 2]), np.array([3, 4]), "train indices"),
        (np.array([0, 1, 5]), np.array([2, 3]), "train indices"),
        (np.array([0, 1, 2]), np.array([2, 3]), "overlap"),
        (np.array([0, 1, 2]), np.array([3, 3]), "test indices"),
        (np.array([0, 1, 2]), np.array([3, 5]), "test indices"),
    ],
)
def test_locked_index_mutations_fail_before_splitting(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_locked_indices(train_idx, test_idx, modeling_n=5, train_n=3, test_n=2)


def test_runtime_contract_requires_exact_versions_threads_and_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _base_config()
    runtime = config["runtime_controls"]
    monkeypatch.setenv("CONDA_DEFAULT_ENV", runtime["python_environment"])
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(variable, runtime[variable])
    validate_runtime_contract(config)
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "wrong-environment")
    with pytest.raises(RuntimeError, match="Python environment"):
        validate_runtime_contract(config)
    monkeypatch.setenv("CONDA_DEFAULT_ENV", runtime["python_environment"])
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    with pytest.raises(RuntimeError, match="runtime control"):
        validate_runtime_contract(config)
    monkeypatch.setenv("OMP_NUM_THREADS", runtime["OMP_NUM_THREADS"])
    changed = copy.deepcopy(config)
    changed["runtime_controls"]["job_count_controls"]["analysis_3_sklearn_n_jobs"] = 2
    with pytest.raises(RuntimeError, match="job count"):
        validate_runtime_contract(changed)
    changed = copy.deepcopy(config)
    changed["analysis_3_date_grouped_minirocket"]["bootstrap"]["n_valid_target"] = 1999
    with pytest.raises(RuntimeError, match="bootstrap contract"):
        validate_runtime_contract(changed)


@pytest.mark.parametrize(
    "stage",
    [
        "current_runtime",
        "git_prerequisites",
        "session0_history",
        "amendment_record",
        "amendment_git",
        "preflight",
        "session4_receipt",
        "frozen_inputs",
        "runtime",
    ],
)
def test_authority_failure_stops_before_loader_or_fit(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    """Every authority-stage failure precedes participant loading and fitting."""
    calls: list[str] = []

    def _authority(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise ValueError(f"synthetic authority failure: {stage}")

    monkeypatch.setattr(date_grouped_module, "validate_analysis3_authority", _authority)
    monkeypatch.setattr(
        date_grouped_module,
        "load_analysis3_inputs",
        lambda *args, **kwargs: calls.append("loader"),
    )
    monkeypatch.setattr(
        date_grouped_module,
        "run_isolated_candidate",
        lambda *args, **kwargs: calls.append("fit"),
    )

    with pytest.raises(ValueError, match=stage):
        execute_approved(BASE_CONFIG_PATH, "a" * 40, "b" * 40)
    assert calls == []


def test_real_config_cumulative_prior_and_preflight_chain_has_no_missing_key() -> None:
    assert "session_1_expected" not in inspect.getsource(date_grouped_module)
    base = _base_config()
    manifest = json.loads(
        (PROJECT_ROOT / "outputs/analysis/posthoc_final_freeze_manifest.json").read_text(encoding="utf-8")
    )
    prior = date_grouped_module._amendment_cumulative_prior(base)
    preflight = date_grouped_module._validate_committed_preflight(
        base,
        manifest,
        "3c0ff6b3b40460a2c340ad3193222ee54979109d",
        date_grouped_module._git,
        prior,
    )
    assert tuple(preflight) == ACCESS_STATE_KEYS


def test_malformed_amendment_and_invalid_preflight_union_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _base_config()
    malformed = copy.deepcopy(base)
    del malformed["historical_consolidation"]["amendment_01"]["access_state"][
        "step_2_approval_materialization"
    ]["cumulative_through_session"][
        "participant_level_date_values_read"
    ]
    with pytest.raises(ValueError, match="cumulative access state"):
        date_grouped_module._amendment_cumulative_prior(malformed)

    manifest = json.loads(
        (PROJECT_ROOT / "outputs/analysis/posthoc_final_freeze_manifest.json").read_text(encoding="utf-8")
    )
    prior = date_grouped_module._amendment_cumulative_prior(base)
    preflight_path = PROJECT_ROOT / date_grouped_module.SCORE_BLIND_PREFLIGHT_RELATIVE_PATH
    payload = json.loads(preflight_path.read_text(encoding="utf-8"))
    payload["access_state"]["cumulative_through_session"]["participant_level_score_values_read"] = True
    monkeypatch.setattr(date_grouped_module, "_read_json", lambda _: payload)
    with pytest.raises(ValueError, match="cumulative"):
        date_grouped_module._validate_committed_preflight(
            base,
            manifest,
            "3c0ff6b3b40460a2c340ad3193222ee54979109d",
            date_grouped_module._git,
            prior,
        )


@pytest.mark.parametrize("mutation", ["malformed", "wrong_order", "wrong_hash", "wrong_activity"])
def test_session4_completion_receipt_mutations_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    config = _base_config()
    prior = OrderedDict((key, False) for key in ACCESS_STATE_KEYS)
    activity = date_grouped_module.SESSION_4_ACTIVITY
    cumulative = OrderedDict((key, activity[key]) for key in ACCESS_STATE_KEYS)
    scientific_paths: dict[str, Path] = {}
    hash_records: list[list[str]] = []
    for index, relative_path in enumerate(date_grouped_module.SESSION_4_SCIENTIFIC_OUTPUT_PATHS):
        path = tmp_path / f"scientific-{index}.csv"
        path.write_bytes(f"aggregate-{index}\n".encode())
        scientific_paths[relative_path] = path
        hash_records.append(["output_hash", relative_path, hashlib.sha256(path.read_bytes()).hexdigest()])
    records = [
        ["header", "posthoc_final_completion_log.v1", "posthoc_final_supplementary", "2026-01-01T00:00:00Z"],
        ["session_identity", "locked_scores", "complete"],
        ["freeze_commit", "3c0ff6b3b40460a2c340ad3193222ee54979109d"],
        *hash_records,
        ["deviations_or_nonestimability_summary", "none", "", ""],
        ["access_state_session_activity", *(str(value).lower() for value in activity.values())],
        ["access_state_cumulative_through_session", *(str(value).lower() for value in cumulative.values())],
        ["completion_receipt", "2026-01-01T00:00:00Z", "complete"],
    ]
    if mutation == "malformed":
        records[-1].pop()
    elif mutation == "wrong_order":
        records[3], records[4] = records[4], records[3]
    elif mutation == "wrong_hash":
        records[3][2] = "0" * 64
    else:
        records[-3][1] = "false"
    log_path = tmp_path / "session4.log"
    log_path.write_text("\n".join("\t".join(record) for record in records) + "\n", encoding="utf-8")
    paths = {**scientific_paths, date_grouped_module.SESSION_4_COMPLETION_LOG_PATH: log_path}
    monkeypatch.setattr(date_grouped_module, "_resolve", lambda value: paths[str(value)])
    with pytest.raises(ValueError):
        date_grouped_module._validate_session4_completion(
            config,
            "3c0ff6b3b40460a2c340ad3193222ee54979109d",
            prior,
        )


def test_real_cli_parser_requires_confirmation_and_dual_lowercase_commits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = PROJECT_ROOT / "scripts" / "run_posthoc_final_date_grouped.py"
    spec = importlib.util.spec_from_file_location("run_posthoc_final_date_grouped", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    called: list[tuple[str, str, str]] = []
    monkeypatch.setattr(module, "execute_approved", lambda c, f, a: called.append((str(c), f, a)))

    valid = [
        "--config",
        "configs/post_hoc_final_supplementary.yaml",
        "--freeze-commit",
        "3c0ff6b3b40460a2c340ad3193222ee54979109d",
        "--amendment-freeze-commit",
        "2e08e8fb89d99bb62cca2e5204a13ff09e47d432",
        "--execute-approved",
    ]
    assert module.main(valid) == 0
    assert len(called) == 1

    for invalid in (
        valid[:-1],
        [*valid[:3], "ABC", *valid[4:]],
        valid[:4] + valid[6:],
        [*valid, "--internal-test-predictions", "forbidden.csv"],
    ):
        with pytest.raises(SystemExit):
            module.main(invalid)
    assert len(called) == 1


def test_cli_bootstraps_project_root_before_src_import() -> None:
    """The CLI establishes its import root before importing project modules."""
    source = (PROJECT_ROOT / "scripts" / "run_posthoc_final_date_grouped.py").read_text(encoding="utf-8")
    root_position = source.index("PROJECT_ROOT = Path(__file__).resolve().parents[1]")
    insert_position = source.index("sys.path.insert(0, str(PROJECT_ROOT))")
    src_import_position = source.index("from src.analysis.posthoc_final_date_grouped import execute_approved")

    assert root_position < insert_position < src_import_position


def test_static_production_source_has_no_forbidden_execution_paths() -> None:
    module_source = inspect.getsource(
        __import__("src.analysis.posthoc_final_date_grouped", fromlist=["posthoc_final_date_grouped"])
    )
    script_source = (PROJECT_ROOT / "scripts" / "run_posthoc_final_date_grouped.py").read_text(encoding="utf-8")
    combined = module_source + script_source
    assert "shuffle=True" not in combined
    assert ".load_data(" not in combined
    assert "._load_split_state(" not in combined
    assert ".run(" not in combined
    assert ".save_outputs(" not in combined
    assert ".pkl" not in combined
    assert "to_pickle(" not in combined
    assert "test_predictions" not in combined
    assert "fold_assignment" not in combined
    assert "collection_location" not in combined


def test_static_source_has_no_live_amendment_or_competing_authority_dependency() -> None:
    """Historical Amendment evidence is never a current filesystem authority."""
    module_source = inspect.getsource(date_grouped_module)
    script_source = (PROJECT_ROOT / "scripts" / "run_posthoc_final_date_grouped.py").read_text(encoding="utf-8")
    combined = module_source + script_source
    forbidden = (
        "AMENDMENT_" + "OVERLAY_RELATIVE_PATH",
        "AMENDMENT_" + "MANIFEST_RELATIVE_PATH",
        "post_hoc_final_supplementary_pre_execution_" + "amendment_01.yaml",
        "final_post_hoc_supplementary_pre_execution_" + "amendment_01.md",
        "posthoc_final_pre_execution_" + "amendment_01_manifest.json",
        "active_" + "rules",
        "EXPECTED_" + "PREFLIGHT_SHA256",
    )

    assert all(token not in combined for token in forbidden)


def test_privacy_rejects_extra_participant_or_date_fields_without_echo(
    tmp_path: Path,
) -> None:
    inputs = _synthetic_inputs()
    partition = build_fixed_partition(inputs.y, inputs.date_groups)
    artifact = run_isolated_candidate(inputs, partition, _base_config(), trainer_factory=_trainer_factory)
    private_value = "synthetic-private-id-777"
    changed = artifact.validation.copy()
    changed[private_value] = private_value
    with pytest.raises(ValueError) as exc_info:
        validate_analysis3_outputs(changed, artifact.fold_audit, _base_config(), expected_n=60)
    assert private_value not in str(exc_info.value)

    prohibited = [
        tmp_path / "participant_predictions.csv",
        tmp_path / "date_lookup.csv",
        tmp_path / "date_group.csv",
        tmp_path / "model.pkl",
    ]
    assert all(not path.exists() for path in prohibited)
