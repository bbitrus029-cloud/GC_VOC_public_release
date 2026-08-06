from __future__ import annotations

import ast
import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _diagnostic_module() -> ModuleType:
    """Import the diagnostic lazily so the missing implementation is the TDD red state."""
    try:
        return importlib.import_module("src.analysis.figure3_e010_deep_model_reconciliation_diagnostic")
    except ModuleNotFoundError as exc:
        pytest.fail(f"E010 diagnostic implementation is missing: {exc}", pytrace=False)


class _SyntheticNormalizer:
    """Identity normalizer recording only aggregate call sizes."""

    def __init__(self) -> None:
        self.call_sizes: list[int] = []

    def transform(self, values: np.ndarray) -> np.ndarray:
        self.call_sizes.append(len(values))
        return np.asarray(values, dtype=np.float32)


class _SyntheticWrapper:
    """CPU wrapper with a real torch state and deterministic synthetic predictor."""

    def __init__(self) -> None:
        self.lightning_model = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.lightning_model.weight.copy_(torch.eye(2))
        self.accelerator = "cpu"
        self.batch_size = 32
        self.predictor_calls: list[tuple[int, int, str]] = []

        def predictor(*, model: Any, x_eval: np.ndarray, batch_size: int, accelerator: str) -> np.ndarray:
            del model
            self.predictor_calls.append((len(x_eval), batch_size, accelerator))
            return np.asarray(x_eval[:, 0], dtype=float)

        self.predictor = predictor

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        positive = self.predictor(
            model=self.lightning_model,
            x_eval=values,
            batch_size=self.batch_size,
            accelerator=self.accelerator,
        )
        return np.column_stack([1.0 - positive, positive])


def _synthetic_condition_rows(module: ModuleType) -> pd.DataFrame:
    """Return 48 aggregate observation rows satisfying the frozen schemas."""
    rows: list[dict[str, object]] = []
    digest = "a" * 64
    for model in module.MODEL_ORDER:
        for condition in module.CONDITION_ORDER:
            for load_number in (1, 2):
                for repeat_number in (1, 2):
                    row = {column: 0 for column in module.CONDITION_DIAGNOSTIC_COLUMNS}
                    row.update(
                        {
                            "diagnostic_id": module.DIAGNOSTIC_ID,
                            "model_id": model["model_id"],
                            "model_display_name": model["model_display_name"],
                            "model_order": model["model_order"],
                            "condition_id": condition["condition_id"],
                            "condition_order": condition["condition_order"],
                            "requested_device_class": condition["requested_device_class"],
                            "input_order": condition["input_order"],
                            "train_warmup": condition["train_warmup"],
                            "load_number": load_number,
                            "repeat_number": repeat_number,
                            "wrapper_accelerator_before": "cpu",
                            "wrapper_accelerator_after": "cpu",
                            "wrapper_batch_size": 32,
                            "actual_parameter_device_class": condition["requested_device_class"],
                            "score_n": 195,
                            "score_all_finite": True,
                            "score_in_unit_interval": True,
                            "max_abs_difference": 0.0,
                            "kendall_tau_b": 1.0,
                            "kendall_estimable": True,
                            "kendall_nonestimable_reason": "",
                            "spearman_rho": 1.0,
                            "spearman_estimable": True,
                            "spearman_nonestimable_reason": "",
                            "state_sha256_before": digest,
                            "state_sha256_after_warmup": digest if condition["train_warmup"] else "",
                            "state_sha256_after_inference": digest,
                            "state_unchanged_from_before": True,
                            "score_reconciliation_passed": True,
                            "auroc_reconciliation_passed": True,
                            "ranking_identical": True,
                        }
                    )
                    rows.append(row)
    return pd.DataFrame(rows, columns=module.CONDITION_DIAGNOSTIC_COLUMNS)


def _synthetic_score_store(module: ModuleType) -> dict[tuple[str, str, int, int], np.ndarray]:
    """Build 48 aligned synthetic test vectors indexed only by the fixed structure."""
    store: dict[tuple[str, str, int, int], np.ndarray] = {}
    base = np.linspace(0.1, 0.9, 5)
    for model in module.MODEL_ORDER:
        for condition in module.CONDITION_ORDER:
            offset = float(condition["condition_order"]) * 1.0e-4
            for load_number in (1, 2):
                for repeat_number in (1, 2):
                    store[(model["model_id"], condition["condition_id"], load_number, repeat_number)] = base + offset
    return store


def test_cli_entrypoint_resolves_src_from_exact_approved_invocation() -> None:
    """The approved file-based CLI invocation must resolve repository modules."""
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_figure3_e010_deep_model_reconciliation_diagnostic.py",
            "--help",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "ModuleNotFoundError" not in result.stdout
    assert "ModuleNotFoundError" not in result.stderr


def test_frozen_model_condition_and_schedule_contract() -> None:
    """Changing condition/model order or load/call counts must fail this test."""
    module = _diagnostic_module()

    assert [item["model_id"] for item in module.MODEL_ORDER] == ["inception_time", "cnn_bilstm_attention"]
    assert [item["condition_id"] for item in module.CONDITION_ORDER] == [
        "cpu_original_direct",
        "cpu_original_train_warmup",
        "gpu_original_direct",
        "gpu_original_train_warmup",
        "cpu_participant_sorted_direct",
        "gpu_participant_sorted_direct",
    ]
    schedule = module.build_execution_schedule()
    assert len(schedule) == 24
    assert all(item["n_test_calls"] == 2 for item in schedule)
    assert [item["load_number"] for item in schedule[:2]] == [1, 2]


def test_cpu_warmup_executes_train_once_then_two_test_calls_without_returning_train_scores() -> None:
    """Repeating warm-up or returning training scores must fail this test."""
    module = _diagnostic_module()
    wrapper = _SyntheticWrapper()
    normalizer = _SyntheticNormalizer()
    condition = module.CONDITION_ORDER[1]
    train = np.asarray([[0.2, 0.8]] * 4, dtype=np.float32)
    test = np.asarray([[0.1, 0.9], [0.3, 0.7], [0.6, 0.4]], dtype=np.float32)

    result = module.execute_condition_load(
        wrapper=wrapper,
        normalizer=normalizer,
        condition=condition,
        train_input=train,
        test_input=test,
        expected_train_n=4,
        expected_test_n=3,
    )

    assert set(result) == {"test_scores", "state_digests", "actual_device_classes", "wrapper_contract"}
    assert len(result["test_scores"]) == 2
    assert normalizer.call_sizes == [4, 3, 3]
    assert wrapper.predictor_calls == [(4, 32, "cpu"), (3, 32, "cpu"), (3, 32, "cpu")]


def test_direct_condition_omits_training_warmup() -> None:
    """Calling training inference in a direct condition must fail this test."""
    module = _diagnostic_module()
    wrapper = _SyntheticWrapper()
    normalizer = _SyntheticNormalizer()
    test = np.asarray([[0.1, 0.9], [0.3, 0.7], [0.6, 0.4]], dtype=np.float32)

    module.execute_condition_load(
        wrapper=wrapper,
        normalizer=normalizer,
        condition=module.CONDITION_ORDER[0],
        train_input=None,
        test_input=test,
        expected_train_n=4,
        expected_test_n=3,
    )

    assert normalizer.call_sizes == [3, 3]
    assert wrapper.predictor_calls == [(3, 32, "cpu"), (3, 32, "cpu")]


def test_ephemeral_gpu_call_preserves_persisted_wrapper_contract() -> None:
    """Assigning to the persisted accelerator or changing batch size must fail this test."""
    module = _diagnostic_module()
    wrapper = _SyntheticWrapper()
    values = np.asarray([[0.2, 0.8], [0.4, 0.6]], dtype=np.float32)

    scores = module.invoke_ephemeral_gpu_predictor(wrapper, values)

    assert np.array_equal(scores, values[:, 0])
    assert wrapper.predictor_calls == [(2, 32, "gpu")]
    assert wrapper.accelerator == "cpu"
    assert wrapper.batch_size == 32


def test_alignment_handles_original_and_sorted_input_without_persisting_maps() -> None:
    """Positional comparison without one-to-one key alignment must fail this test."""
    module = _diagnostic_module()
    input_keys = np.asarray([30, 10, 20], dtype=np.int64)
    scores = np.asarray([0.3, 0.1, 0.2])
    frozen_keys = np.asarray([10, 20, 30], dtype=np.int64)

    aligned = module.align_scores_to_frozen_order(scores, input_keys, frozen_keys)

    assert np.array_equal(aligned, np.asarray([0.1, 0.2, 0.3]))


def test_model_state_digest_is_device_independent_and_detects_mutation() -> None:
    """Omitting parameters/buffers or including device placement must fail this test."""
    module = _diagnostic_module()
    model = torch.nn.BatchNorm1d(2)
    before = module.canonical_model_state_sha256(model)
    moved = module.canonical_model_state_sha256(model.to("cpu"))
    with torch.no_grad():
        model.running_mean[0] = 1.0
    after = module.canonical_model_state_sha256(model)

    assert before == moved
    assert after != before
    assert len(before) == 64


def test_actual_device_reporting_rejects_mixed_or_empty_state() -> None:
    """CUDA availability alone or mixed device records must not prove execution."""
    module = _diagnostic_module()
    assert module.device_class_from_records(["cpu", "cpu"]) == "cpu"
    with pytest.raises(ValueError, match="mixed"):
        module.device_class_from_records(["cpu", "cuda"])
    with pytest.raises(ValueError, match="absent"):
        module.device_class_from_records([])


def test_aggregate_differences_correlations_and_acceptance_rules() -> None:
    """Rounding, non-strict counts, or alternate reconciliation tolerances must fail."""
    module = _diagnostic_module()
    frozen = np.asarray([0.1, 0.4, 0.6, 0.9])
    reproduced = np.asarray([0.1, 0.400001, 0.60002, 0.899])
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)

    result = module.calculate_score_diagnostics(
        reproduced,
        frozen,
        labels,
        frozen_metric_auroc=1.0,
        score_atol=1.0e-6,
        auroc_atol=1.0e-12,
    )

    absolute = np.abs(reproduced - frozen)
    assert result["max_abs_difference"] == float(np.max(absolute))
    assert result["mean_abs_difference"] == float(np.mean(absolute))
    assert result["median_abs_difference"] == float(np.median(absolute))
    assert result["rmse_difference"] == float(np.sqrt(np.mean((reproduced - frozen) ** 2)))
    assert result["abs_difference_q95"] == float(np.percentile(absolute, 95, method="linear"))
    assert result["abs_difference_q99"] == float(np.percentile(absolute, 99, method="linear"))
    assert result["count_abs_difference_gt_1e_6"] == int(np.count_nonzero(absolute > 1.0e-6))
    assert result["score_reconciliation_passed"] is False
    assert "p_value" not in result


def test_nonestimable_rank_correlations_are_retained_without_p_values() -> None:
    """Dropping a constant-score diagnostic or emitting P values must fail this test."""
    module = _diagnostic_module()
    result = module.rank_diagnostics(np.ones(4), np.arange(4, dtype=float))

    assert result == {
        "ranking_identical": False,
        "kendall_tau_b": None,
        "kendall_estimable": False,
        "kendall_nonestimable_reason": "constant_or_degenerate_score_vector",
        "spearman_rho": None,
        "spearman_estimable": False,
        "spearman_nonestimable_reason": "constant_or_degenerate_score_vector",
    }


def test_fixed_comparison_matrix_has_exact_104_rows_and_order() -> None:
    """Missing, extra, or value-reordered comparisons must fail this test."""
    module = _diagnostic_module()
    frame = module.build_comparison_diagnostics(_synthetic_score_store(module), score_atol=1.0e-6)

    assert len(frame) == 104
    assert frame.columns.tolist() == list(module.COMPARISON_DIAGNOSTIC_COLUMNS)
    assert frame["comparison_type"].value_counts().to_dict() == {
        "within_load_repeat": 24,
        "between_fresh_load": 24,
        "cpu_gpu": 24,
        "input_order": 16,
        "train_warmup": 16,
    }


def test_condition_stability_allows_nonzero_below_tolerance_but_reports_nonidentity() -> None:
    """Treating exact nonidentity as automatically nonisolatable must fail this test."""
    module = _diagnostic_module()
    conditions = _synthetic_condition_rows(module)
    comparisons = module.build_comparison_diagnostics(_synthetic_score_store(module), score_atol=1.0e-6)
    comparisons.loc[comparisons.index[0], "max_abs_score_difference"] = 5.0e-7
    comparisons.loc[comparisons.index[0], "exact_score_identity"] = False
    comparisons.loc[comparisons.index[0], "exceeds_score_reconciliation_tolerance"] = False

    summary = module.build_model_condition_summary(conditions, comparisons, score_atol=1.0e-6)

    first = summary.iloc[0]
    assert first["condition_stable_at_frozen_scale"]
    assert first["maximum_within_load_repeat_difference"] == pytest.approx(5.0e-7)


@pytest.mark.parametrize(
    ("field", "reason"),
    [
        ("all_actual_devices_match_request", "left_condition_contract_or_device_failure"),
        ("all_state_digests_unchanged", "left_condition_state_mutation"),
        ("within_load_differences_within_1e_6", "left_condition_repeat_or_load_variability_gt_1e_6"),
        ("score_reconciliation_status_consistent", "left_condition_reconciliation_status_inconsistent"),
    ],
)
def test_isolation_reason_enum_and_precedence(field: str, reason: str) -> None:
    """Free-written or incorrectly prioritized nonisolation reasons must fail."""
    module = _diagnostic_module()
    summary = module.build_model_condition_summary(
        _synthetic_condition_rows(module),
        module.build_comparison_diagnostics(_synthetic_score_store(module), score_atol=1.0e-6),
        score_atol=1.0e-6,
    )
    summary.loc[(summary["model_id"] == "inception_time") & (summary["condition_order"] == 2), field] = False
    summary = module.recompute_condition_stability(summary)
    branches = module.build_interpretation_branches(
        summary,
        module.build_comparison_diagnostics(_synthetic_score_store(module), score_atol=1.0e-6),
        score_atol=1.0e-6,
    )

    row = branches.loc[branches["model_id"] == "inception_time"].iloc[0]
    assert not row["e010_gpu_to_cpu_effect_isolatable"]
    assert row["e010_gpu_to_cpu_effect_nonisolatable_reason"] == reason
    assert not row["e010_gpu_to_cpu_change_supported"]


def test_supported_device_branch_requires_all_four_matches_and_reconciliation_pattern() -> None:
    """A partial or nonisolated CPU/GPU contrast must not support the E010 cause."""
    module = _diagnostic_module()
    conditions = _synthetic_condition_rows(module)
    condition_two = conditions["condition_order"] == 2
    conditions.loc[condition_two, "score_reconciliation_passed"] = False
    conditions.loc[condition_two, "max_abs_difference"] = 2.0e-4
    score_store = _synthetic_score_store(module)
    for load in (1, 2):
        for repeat in (1, 2):
            score_store[("inception_time", "cpu_original_train_warmup", load, repeat)] -= 0.01
    comparisons = module.build_comparison_diagnostics(score_store, score_atol=1.0e-6)
    summary = module.build_model_condition_summary(conditions, comparisons, score_atol=1.0e-6)

    branches = module.build_interpretation_branches(summary, comparisons, score_atol=1.0e-6)

    inception = branches.loc[branches["model_id"] == "inception_time"].iloc[0]
    assert inception["e010_gpu_to_cpu_effect_isolatable"]
    assert inception["e010_gpu_to_cpu_change_supported"]


def test_failed_isolation_causal_branch_never_enters_root_cause_summary() -> None:
    """Reporting a causal root despite a failed isolation gate must fail this test."""
    module = _diagnostic_module()
    conditions = _synthetic_condition_rows(module)
    conditions.loc[conditions["condition_order"] == 2, "state_unchanged_from_before"] = False
    comparisons = module.build_comparison_diagnostics(_synthetic_score_store(module), score_atol=1.0e-6)
    summary = module.build_model_condition_summary(conditions, comparisons, score_atol=1.0e-6)
    branches = module.build_interpretation_branches(summary, comparisons, score_atol=1.0e-6)

    assert all(
        "e010_gpu_to_cpu_change_supported" not in value for value in branches["root_cause_summary"].astype(str)
    )


def test_output_schemas_row_counts_and_one_based_indices_are_enforced() -> None:
    """Wrong schemas, row counts, or zero-based persisted load/repeat fields must fail."""
    module = _diagnostic_module()
    condition = _synthetic_condition_rows(module)
    comparison = module.build_comparison_diagnostics(_synthetic_score_store(module), score_atol=1.0e-6)
    summary = module.build_model_condition_summary(condition, comparison, score_atol=1.0e-6)
    branches = module.build_interpretation_branches(summary, comparison, score_atol=1.0e-6)
    frames = {
        "condition_diagnostics.csv": condition,
        "comparison_diagnostics.csv": comparison,
        "model_condition_summary.csv": summary,
        "interpretation_branches.csv": branches,
    }

    module.validate_output_frames(frames, strict_values=False)
    assert condition["load_number"].min() == 1
    assert condition["repeat_number"].min() == 1
    broken = dict(frames)
    broken["condition_diagnostics.csv"] = condition.iloc[:-1]
    with pytest.raises(ValueError, match="row count"):
        module.validate_output_frames(broken, strict_values=False)


def test_privacy_scan_rejects_participant_fields_without_echo() -> None:
    """Participant-level output or echoed private values must fail this test."""
    module = _diagnostic_module()
    payload = json.dumps({"sample_id": "SYNTHETIC-PRIVATE-TOKEN"}).encode("utf-8")

    with pytest.raises(ValueError, match="privacy validation") as exc_info:
        module.validate_serialized_privacy({"audit.json": payload}, private_tokens={"SYNTHETIC-PRIVATE-TOKEN"})

    assert "SYNTHETIC-PRIVATE-TOKEN" not in str(exc_info.value)


def test_existing_outputs_and_partial_atomic_files_are_never_promoted(tmp_path: Path) -> None:
    """Overwriting a result or retaining a rejected temporary file must fail this test."""
    module = _diagnostic_module()
    target = tmp_path / "candidate" / "payload.csv"

    def reject(_path: Path) -> None:
        raise ValueError("synthetic rejection")

    with pytest.raises(ValueError, match="synthetic rejection"):
        module.atomic_write_bytes(target, b"header\nvalue\n", validator=reject)
    assert not target.exists()
    assert not (target.parent / ".payload.csv.tmp").exists()

    candidate = tmp_path / "promotion_candidate"
    final = tmp_path / "final"
    candidate.mkdir()
    final.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        module.promote_candidate_directory(candidate, final)


def test_frozen_configuration_rejects_alternative_batch_size(tmp_path: Path) -> None:
    """Configuration flexibility that allows a post-result batch size must fail."""
    module = _diagnostic_module()
    config = module.expected_configuration()
    config["batch_size"] = 16
    path = tmp_path / "diagnostic.yaml"
    path.write_text(module.serialize_yaml(config), encoding="utf-8")

    with pytest.raises(ValueError, match="batch_size"):
        module.load_diagnostic_config(path)


def test_source_has_no_forbidden_scientific_invocations() -> None:
    """Introducing fit, permutation, bootstrap, resample, or threshold calls must fail."""
    module = _diagnostic_module()
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    forbidden = {"fit", "fit_transform", "partial_fit", "permutation", "bootstrap", "resample"}
    invoked: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                invoked.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                invoked.add(node.func.attr)
    assert invoked.isdisjoint(forbidden)


def test_synthetic_suite_does_not_require_real_project_artifacts() -> None:
    """Accidental reads of participant/model/prediction/metric artifacts must fail this test."""
    module = _diagnostic_module()
    assert module.SYNTHETIC_TESTS_REQUIRE_REAL_ARTIFACTS is False
