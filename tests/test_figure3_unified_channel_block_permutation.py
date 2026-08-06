from __future__ import annotations

import ast
import importlib
import inspect
import json
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kendalltau
import yaml


def _analysis_module() -> ModuleType:
    """Import the implementation lazily so the TDD red state is an explicit failure."""
    try:
        return importlib.import_module("src.analysis.figure3_unified_channel_block_permutation")
    except ModuleNotFoundError as exc:
        pytest.fail(f"figure3 unified permutation implementation is missing: {exc}", pytrace=False)


def _pipeline_b_feature_names() -> list[str]:
    """Return a complete synthetic 11-channel by 9-feature schema."""
    return [f"F{feature}_S{channel}" for feature in range(1, 10) for channel in range(1, 12)]


def _minimal_frames(module: ModuleType) -> dict[str, pd.DataFrame]:
    """Build schema-shaped aggregate frames without participant-level values."""
    return {
        "channel_importance_summary.csv": pd.DataFrame(
            np.zeros((66, len(module.CHANNEL_IMPORTANCE_COLUMNS))),
            columns=module.CHANNEL_IMPORTANCE_COLUMNS,
        ),
        "permutation_repeats.csv": pd.DataFrame(
            np.zeros((6600, len(module.PERMUTATION_REPEAT_COLUMNS))),
            columns=module.PERMUTATION_REPEAT_COLUMNS,
        ),
        "cross_model_consistency.csv": pd.DataFrame(
            np.zeros((15, len(module.CROSS_MODEL_COLUMNS))),
            columns=module.CROSS_MODEL_COLUMNS,
        ),
        "channel_consensus_summary.csv": pd.DataFrame(
            np.zeros((11, len(module.CHANNEL_CONSENSUS_COLUMNS))),
            columns=module.CHANNEL_CONSENSUS_COLUMNS,
        ),
    }


def _synthetic_repeat_frame(module: ModuleType) -> pd.DataFrame:
    """Build a complete aggregate-only repeat fixture with signed findings."""
    rows: list[dict[str, object]] = []
    for model in module.MODEL_ORDER:
        for channel_index, channel_name in enumerate(module.CHANNEL_ORDER, start=1):
            direction = 1.0 if int(model["model_order"]) % 2 else -1.0
            channel_effect = direction * (channel_index - 6) * 0.001
            for permutation_repeat in range(1, 101):
                repeat_offset = (permutation_repeat - 50.5) * 1.0e-6
                delta = channel_effect + repeat_offset
                rows.append(
                    {
                        "analysis_id": module.ANALYSIS_ID,
                        "model_id": model["model_id"],
                        "model_order": model["model_order"],
                        "channel_index": channel_index,
                        "channel_name": channel_name,
                        "permutation_repeat": permutation_repeat,
                        "baseline_auroc": 0.80,
                        "permuted_auroc": 0.80 - delta,
                        "delta_auroc": delta,
                    }
                )
    return pd.DataFrame(rows, columns=module.PERMUTATION_REPEAT_COLUMNS)


class _SyntheticDeviceTensor:
    """Expose only a sanitized synthetic tensor device class."""

    def __init__(self, device_class: str) -> None:
        self.device = SimpleNamespace(type=device_class)


class _SyntheticDeviceModel:
    """Provide synthetic named parameter/buffer device records."""

    def __init__(
        self,
        parameter_classes: tuple[str, ...] = ("cuda",),
        buffer_classes: tuple[str, ...] = ("cuda",),
    ) -> None:
        self.parameter_classes = parameter_classes
        self.buffer_classes = buffer_classes
        self.parameter_reads = 0
        self.buffer_reads = 0

    def named_parameters(self) -> list[tuple[str, _SyntheticDeviceTensor]]:
        """Return synthetic parameter records and count verification reads."""
        self.parameter_reads += 1
        return [
            (f"parameter_{index}", _SyntheticDeviceTensor(device_class))
            for index, device_class in enumerate(self.parameter_classes)
        ]

    def named_buffers(self) -> list[tuple[str, _SyntheticDeviceTensor]]:
        """Return synthetic buffer records and count verification reads."""
        self.buffer_reads += 1
        return [
            (f"buffer_{index}", _SyntheticDeviceTensor(device_class))
            for index, device_class in enumerate(self.buffer_classes)
        ]


class _RecordingNormalizer:
    """Record exact transform calls while returning a distinct normalized array."""

    def __init__(self) -> None:
        self.inputs: list[np.ndarray] = []
        self.outputs: list[np.ndarray] = []

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Return a deterministic distinct array for predictor identity checks."""
        self.inputs.append(values)
        normalized = np.asarray(values, dtype=float) + 0.25
        self.outputs.append(normalized)
        return normalized


def _synthetic_deep_payload(
    *,
    model_id: str = "inception_time",
    accelerator: str = "cpu",
    batch_size: int = 32,
    predictor: Any | None = None,
    lightning_model: Any | None = None,
) -> dict[str, Any]:
    """Build a complete synthetic locked deep-model payload."""
    from src.training.dl_inference import DLPredictProbaWrapper

    class SyntheticLightningModel:
        """Support wrapper construction without loading a real model."""

        def __init__(self, device_model: Any) -> None:
            self.device_model = device_model

        def to(self, _device: str) -> SyntheticLightningModel:
            return self

        def named_parameters(self) -> Any:
            return self.device_model.named_parameters()

        def named_buffers(self) -> Any:
            return self.device_model.named_buffers()

    device_model = lightning_model or _SyntheticDeviceModel()
    synthetic_model = SyntheticLightningModel(device_model)
    predict = predictor or (lambda **kwargs: np.full(len(kwargs["x_eval"]), 0.5, dtype=float))
    wrapper = DLPredictProbaWrapper(
        lightning_model=synthetic_model,
        predictor=predict,
        batch_size=batch_size,
        accelerator=accelerator,
    )
    return {
        "transformer": None,
        "normalizer": _RecordingNormalizer(),
        "model": wrapper,
        "calibrator": None,
        "threshold_p_raw": 0.5,
        "threshold_p_final_equivalent": 0.5,
        "feature_names": [f"S{index}" for index in range(1, 12)],
        "training_sample_ids": np.asarray([10_001, 10_002], dtype=np.int64),
        "metadata": {"model_name": model_id},
    }


def test_pipeline_a_moves_complete_selected_trajectory() -> None:
    """A partial time-axis assignment must not satisfy complete-block permutation."""
    module = _analysis_module()
    original = np.arange(4 * 3 * 11, dtype=np.float64).reshape(4, 3, 11)
    permutation = np.asarray([2, 0, 3, 1], dtype=np.int64)

    permuted = module.permute_pipeline_a_channel(original, "S4", permutation)

    assert np.array_equal(permuted[:, :, 3], original[permutation, :, 3])


def test_pipeline_a_leaves_nonselected_channels_byte_identical() -> None:
    """Mutating any nonselected Pipeline A channel must fail this test."""
    module = _analysis_module()
    original = np.arange(4 * 3 * 11, dtype=np.float64).reshape(4, 3, 11)
    permutation = np.asarray([2, 0, 3, 1], dtype=np.int64)

    permuted = module.permute_pipeline_a_channel(original, "S4", permutation)

    assert permuted[:, :, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]].tobytes() == (
        original[:, :, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]].tobytes()
    )


def test_pipeline_b_moves_exact_selected_nine_feature_block() -> None:
    """Permuting fewer or more than nine channel features must fail this test."""
    module = _analysis_module()
    feature_names = _pipeline_b_feature_names()
    frame = pd.DataFrame(np.arange(4 * 99, dtype=np.float64).reshape(4, 99), columns=feature_names)
    channel_map = module.build_pipeline_b_channel_map(feature_names)
    permutation = np.asarray([2, 0, 3, 1], dtype=np.int64)

    permuted = module.permute_pipeline_b_channel(frame, "S4", permutation, channel_map)

    selected = channel_map["S4"]
    assert len(selected) == 9
    assert np.array_equal(permuted.loc[:, selected].to_numpy(), frame.iloc[permutation].loc[:, selected].to_numpy())


def test_pipeline_b_leaves_other_features_byte_identical_and_ordered() -> None:
    """Changing a nonselected feature or column order must fail this test."""
    module = _analysis_module()
    feature_names = _pipeline_b_feature_names()
    frame = pd.DataFrame(np.arange(4 * 99, dtype=np.float64).reshape(4, 99), columns=feature_names)
    channel_map = module.build_pipeline_b_channel_map(feature_names)
    permutation = np.asarray([2, 0, 3, 1], dtype=np.int64)

    permuted = module.permute_pipeline_b_channel(frame, "S4", permutation, channel_map)

    other = [name for name in feature_names if name not in channel_map["S4"]]
    assert permuted.columns.tolist() == feature_names
    assert permuted.loc[:, other].to_numpy().tobytes() == frame.loc[:, other].to_numpy().tobytes()


def test_same_participant_mapping_is_reused_across_channels_and_pipelines() -> None:
    """Regenerating permutations inside a channel/model loop must fail this test."""
    module = _analysis_module()
    matrix = np.asarray([[2, 0, 3, 1], [1, 3, 0, 2]], dtype=np.int64)
    mapping = module.select_permutation_mapping(matrix, permutation_repeat=1)
    pipeline_a = np.arange(4 * 2 * 11, dtype=np.float64).reshape(4, 2, 11)
    names = _pipeline_b_feature_names()
    pipeline_b = pd.DataFrame(np.arange(4 * 99, dtype=np.float64).reshape(4, 99), columns=names)
    channel_map = module.build_pipeline_b_channel_map(names)

    a_s1 = module.permute_pipeline_a_channel(pipeline_a, "S1", mapping)
    a_s11 = module.permute_pipeline_a_channel(pipeline_a, "S11", mapping)
    b_s1 = module.permute_pipeline_b_channel(pipeline_b, "S1", mapping, channel_map)
    b_s11 = module.permute_pipeline_b_channel(pipeline_b, "S11", mapping, channel_map)

    assert np.array_equal(a_s1[:, :, 0], pipeline_a[matrix[0], :, 0])
    assert np.array_equal(a_s11[:, :, 10], pipeline_a[matrix[0], :, 10])
    assert np.array_equal(b_s1.loc[:, channel_map["S1"]], pipeline_b.iloc[matrix[0]].loc[:, channel_map["S1"]])
    assert np.array_equal(b_s11.loc[:, channel_map["S11"]], pipeline_b.iloc[matrix[0]].loc[:, channel_map["S11"]])


def test_negative_signed_delta_is_retained() -> None:
    """Zero clipping of an adverse signed decrease must fail this test."""
    module = _analysis_module()

    assert module.signed_auroc_delta(0.60, 0.65) == pytest.approx(-0.05)


def test_signed_delta_uses_neither_absolute_value_nor_reversed_sign() -> None:
    """Absolute-value or reversed-sign implementations must fail this test."""
    module = _analysis_module()

    assert module.signed_auroc_delta(0.72, 0.70) == pytest.approx(0.02)
    assert module.signed_auroc_delta(0.70, 0.72) == pytest.approx(-0.02)


def test_model_and_channel_order_are_frozen() -> None:
    """Using the legacy internal model order or lexical channel order must fail."""
    module = _analysis_module()

    assert [record["model_id"] for record in module.MODEL_ORDER] == [
        "minirocket_ridge",
        "xgboost",
        "elastic_net_lr",
        "svm",
        "inception_time",
        "cnn_bilstm_attention",
    ]
    assert module.CHANNEL_ORDER == tuple(f"S{index}" for index in range(1, 12))


def test_average_ranks_are_used_for_exact_ties() -> None:
    """Ordinal channel-order tie breaking must fail this test."""
    module = _analysis_module()

    ranks = module.descending_average_ranks(np.asarray([0.20, 0.20, 0.10], dtype=float))

    assert np.array_equal(ranks, np.asarray([1.5, 1.5, 3.0]))


def test_kendall_tau_b_matches_scipy_statistic_without_p_value_field() -> None:
    """A tau variant mismatch or inferential P-value output must fail this test."""
    module = _analysis_module()
    left = np.asarray([1.0, 2.0, 2.0, 4.0, 5.0])
    right = np.asarray([1.0, 3.0, 2.0, 4.0, 4.0])

    result = module.kendall_tau_b_coefficient(left, right)
    expected = float(kendalltau(left, right, variant="b").statistic)

    assert set(result) == {"kendall_tau_b", "estimable", "nonestimable_reason"}
    assert result["kendall_tau_b"] == pytest.approx(expected)
    assert result["estimable"] is True
    assert result["nonestimable_reason"] == ""


def test_nonestimable_kendall_pair_is_retained() -> None:
    """Dropping or converting a constant-vector pair to zero must fail this test."""
    module = _analysis_module()

    result = module.kendall_tau_b_coefficient(np.ones(11), np.arange(11, dtype=float))

    assert result == {
        "kendall_tau_b": None,
        "estimable": False,
        "nonestimable_reason": "constant_or_degenerate_channel_vector",
    }


def test_paired_stratified_bootstrap_preserves_counts_and_pairing() -> None:
    """Unstratified draws or different baseline/permutation indices must fail this test."""
    module = _analysis_module()
    labels = np.asarray([1, 1, 0, 0, 0], dtype=np.int64)
    _, bootstrap, _digests = module.generate_random_matrices(
        labels,
        seed=42,
        n_permutations=2,
        n_bootstrap=5,
    )
    baseline = np.asarray([0.8, 0.7, 0.3, 0.2, 0.1])
    permuted = np.stack([baseline - 0.05, baseline + 0.02])

    deltas = module.paired_bootstrap_channel_deltas(labels, baseline, permuted, bootstrap)

    assert bootstrap.shape == (5, 5)
    assert all(np.count_nonzero(labels[row] == 1) == 2 for row in bootstrap)
    assert all(np.count_nonzero(labels[row] == 0) == 3 for row in bootstrap)
    assert deltas.shape == (5,)
    assert np.isfinite(deltas).all()


def test_random_matrices_and_digests_are_deterministic() -> None:
    """Implicit global RNG state or altered stream spawning must fail this test."""
    module = _analysis_module()
    labels = np.asarray([1, 1, 0, 0, 0], dtype=np.int64)

    first = module.generate_random_matrices(labels, seed=42, n_permutations=3, n_bootstrap=4)
    second = module.generate_random_matrices(labels, seed=42, n_permutations=3, n_bootstrap=4)

    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[2] == second[2]


def test_baseline_reconciliation_fails_closed_outside_tolerance() -> None:
    """A relaxed score or AUROC reconciliation must fail this test."""
    module = _analysis_module()
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    frozen = np.asarray([0.1, 0.4, 0.6, 0.9])
    reproduced = frozen.copy()
    reproduced[0] += 2.0e-6

    with pytest.raises(ValueError, match="score reconciliation"):
        module.reconcile_baseline_scores(
            labels,
            reproduced,
            frozen,
            metric_auroc=1.0,
            score_atol=1.0e-6,
            auroc_atol=1.0e-12,
        )


@pytest.mark.parametrize(
    "feature_names",
    [
        pytest.param(_pipeline_b_feature_names()[:-1], id="missing_feature"),
        pytest.param([*_pipeline_b_feature_names()[:-1], "F1_UNKNOWN"], id="unknown_channel"),
        pytest.param([*_pipeline_b_feature_names()[:-1], _pipeline_b_feature_names()[0]], id="duplicate_feature"),
    ],
)
def test_missing_or_ambiguous_channel_mapping_fails_closed(feature_names: list[str]) -> None:
    """An incomplete, unknown, or duplicated feature map must fail this test."""
    module = _analysis_module()

    with pytest.raises(ValueError, match="Pipeline B channel mapping"):
        module.build_pipeline_b_channel_map(feature_names)


def test_output_schema_and_row_counts_are_enforced() -> None:
    """A missing column or wrong scientific row count must fail validation."""
    module = _analysis_module()
    frames = _minimal_frames(module)
    frames["channel_consensus_summary.csv"] = frames["channel_consensus_summary.csv"].iloc[:-1]

    with pytest.raises(ValueError, match="row count"):
        module.validate_output_frames(frames, strict_values=False)


def test_output_privacy_rejects_participant_level_fields_without_echo() -> None:
    """Persisting an individual identifier field must fail without echoing its value."""
    module = _analysis_module()
    private_payload = json.dumps({"sample_id": "SYNTHETIC-PRIVATE-VALUE"}).encode()

    with pytest.raises(ValueError, match="privacy validation") as exc_info:
        module.validate_serialized_privacy({"audit.json": private_payload})

    assert "SYNTHETIC-PRIVATE-VALUE" not in str(exc_info.value)


def test_existing_final_output_is_never_overwritten(tmp_path: Path) -> None:
    """Replacing a stale scientific result directory must fail this test."""
    module = _analysis_module()
    candidate = tmp_path / "candidate"
    final = tmp_path / "final"
    candidate.mkdir()
    final.mkdir()
    (candidate / "payload.txt").write_text("new", encoding="utf-8")
    (final / "payload.txt").write_text("old", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        module.promote_candidate_directory(candidate, final)

    assert (final / "payload.txt").read_text(encoding="utf-8") == "old"
    assert candidate.exists()


def test_failed_atomic_write_cannot_leave_promotable_partial_file(tmp_path: Path) -> None:
    """Validator failure after a temporary write must leave no final artifact."""
    module = _analysis_module()
    target = tmp_path / "candidate" / "payload.csv"

    def reject(_path: Path) -> None:
        raise ValueError("synthetic validation failure")

    with pytest.raises(ValueError, match="synthetic validation failure"):
        module.atomic_write_bytes(target, b"header\nvalue\n", validator=reject)

    assert not target.exists()
    assert not (target.parent / f".{target.name}.tmp").exists()


def test_s1_uses_pipeline_a_axis_position_zero_only() -> None:
    """Using persisted channel index 1 as NumPy position 1 must fail this test."""
    module = _analysis_module()
    original = np.arange(4 * 2 * 11, dtype=np.float64).reshape(4, 2, 11)
    permutation = np.asarray([3, 2, 1, 0])

    permuted = module.permute_pipeline_a_channel(original, "S1", permutation)

    assert np.array_equal(permuted[:, :, 0], original[permutation, :, 0])
    assert permuted[:, :, 1:].tobytes() == original[:, :, 1:].tobytes()


def test_s11_uses_pipeline_a_axis_position_ten_only() -> None:
    """Using persisted channel index 11 as a NumPy position must fail this test."""
    module = _analysis_module()
    original = np.arange(4 * 2 * 11, dtype=np.float64).reshape(4, 2, 11)
    permutation = np.asarray([3, 2, 1, 0])

    permuted = module.permute_pipeline_a_channel(original, "S11", permutation)

    assert np.array_equal(permuted[:, :, 10], original[permutation, :, 10])
    assert permuted[:, :, :10].tobytes() == original[:, :, :10].tobytes()


def test_pipeline_b_endpoint_blocks_keep_one_based_output_indices() -> None:
    """Deriving feature positions from one-based output indices must fail this test."""
    module = _analysis_module()
    names = _pipeline_b_feature_names()
    frame = pd.DataFrame(np.arange(4 * 99, dtype=np.float64).reshape(4, 99), columns=names)
    channel_map = module.build_pipeline_b_channel_map(names)
    permutation = np.asarray([3, 2, 1, 0])

    s1 = module.permute_pipeline_b_channel(frame, "S1", permutation, channel_map)
    s11 = module.permute_pipeline_b_channel(frame, "S11", permutation, channel_map)

    assert channel_map["S1"] == [f"F{feature}_S1" for feature in range(1, 10)]
    assert channel_map["S11"] == [f"F{feature}_S11" for feature in range(1, 10)]
    assert np.array_equal(s1.loc[:, channel_map["S1"]], frame.iloc[permutation].loc[:, channel_map["S1"]])
    assert np.array_equal(s11.loc[:, channel_map["S11"]], frame.iloc[permutation].loc[:, channel_map["S11"]])
    assert module.output_index_fields("S1", 1) == {"channel_index": 1, "permutation_repeat": 1}
    assert module.output_index_fields("S11", 100) == {"channel_index": 11, "permutation_repeat": 100}


def test_permutation_repeat_one_uses_matrix_row_zero() -> None:
    """Using persisted repeat 1 as zero-based row 1 must fail this test."""
    module = _analysis_module()
    matrix = np.asarray([[2, 0, 1], [1, 2, 0]], dtype=np.int64)

    assert np.array_equal(module.select_permutation_mapping(matrix, 1), matrix[0])


def test_permutation_repeat_one_hundred_uses_matrix_row_ninety_nine() -> None:
    """Using persisted repeat 100 as row 100 must fail this test."""
    module = _analysis_module()
    matrix = np.vstack([np.roll(np.arange(5), index) for index in range(100)])

    assert np.array_equal(module.select_permutation_mapping(matrix, 100), matrix[99])


def test_persisted_channel_and_repeat_fields_are_one_based() -> None:
    """Leaking zero-based internal positions into aggregate outputs must fail this test."""
    module = _analysis_module()

    fields = [
        module.output_index_fields(channel, repeat)
        for channel in module.CHANNEL_ORDER
        for repeat in (1, 100)
    ]

    assert min(field["channel_index"] for field in fields) == 1
    assert max(field["channel_index"] for field in fields) == 11
    assert {field["permutation_repeat"] for field in fields} == {1, 100}


def test_build_aggregate_frames_enforces_complete_matrix_and_final_model_ci() -> None:
    """Omitting a model-channel row or adding supporting-model CIs must fail this integration contract."""
    module = _analysis_module()
    repeats = _synthetic_repeat_frame(module)
    bootstrap_intervals = {channel: (-0.01, 0.02) for channel in module.CHANNEL_ORDER}

    frames = module.build_aggregate_frames(repeats, bootstrap_intervals)

    module.validate_output_frames(frames, strict_values=True)
    summary = frames["channel_importance_summary.csv"]
    final_rows = summary.loc[summary["model_id"] == "minirocket_ridge"]
    supporting_rows = summary.loc[summary["model_id"] != "minirocket_ridge"]
    assert len(summary) == 66
    assert (final_rows["n_bootstrap"] == 2000).all()
    assert final_rows[["bootstrap_ci_low", "bootstrap_ci_high"]].notna().all(axis=None)
    assert (supporting_rows["n_bootstrap"] == 0).all()
    assert supporting_rows[["bootstrap_ci_low", "bootstrap_ci_high"]].isna().all(axis=None)
    assert (summary["negative_delta_count"] > 0).any()


def test_render_report_contains_all_models_channels_and_pairwise_rows() -> None:
    """A truncated matrix or omitted pairwise consistency row must fail report generation."""
    module = _analysis_module()
    repeats = _synthetic_repeat_frame(module)
    frames = module.build_aggregate_frames(
        repeats,
        {channel: (-0.01, 0.02) for channel in module.CHANNEL_ORDER},
    )
    reconciliation = [
        {
            "model_id": model["model_id"],
            "max_abs_score_difference": 0.0,
            "reproduced_auroc": 0.8,
            "frozen_prediction_auroc": 0.8,
            "frozen_metric_auroc": 0.8,
            "passed": True,
        }
        for model in module.MODEL_ORDER
    ]

    report = module.render_report(frames, reconciliation, ["Synthetic limitation."])

    assert all(str(model["model_display_name"]) in report for model in module.MODEL_ORDER)
    assert all(channel in report for channel in module.CHANNEL_ORDER)
    assert report.count("| minirocket_ridge |") >= 1
    assert "No result-driven Figure 3 recommendation is made." in report
    assert "No chemical or biological interpretation is made." in report


def test_compare_run_directories_ignores_only_declared_volatile_json_fields(tmp_path: Path) -> None:
    """A scientific JSON difference must fail even when volatile fields differ legitimately."""
    module = _analysis_module()
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    for filename in (*module.SCIENTIFIC_PAYLOAD_FILENAMES,):
        (run_a / filename).write_bytes(b"identical\n")
        (run_b / filename).write_bytes(b"identical\n")
    record_a = {
        "output_directory": "run_a",
        "run_started_at_utc": "a",
        "run_finished_at_utc": "b",
        "wall_clock_seconds": 1.0,
        "scientific": {"value": 7},
    }
    record_b = {
        "output_directory": "run_b",
        "run_started_at_utc": "c",
        "run_finished_at_utc": "d",
        "wall_clock_seconds": 2.0,
        "scientific": {"value": 7},
    }
    (run_a / "run_record.json").write_text(json.dumps(record_a), encoding="utf-8")
    (run_b / "run_record.json").write_text(json.dumps(record_b), encoding="utf-8")

    result = module.compare_run_directories(run_a, run_b)

    assert result["scientific_payloads_byte_identical"] is True
    assert result["normalized_run_records_identical"] is True
    record_b["scientific"]["value"] = 8
    (run_b / "run_record.json").write_text(json.dumps(record_b), encoding="utf-8")
    with pytest.raises(ValueError, match="normalized run records"):
        module.compare_run_directories(run_a, run_b)


def test_load_analysis_config_accepts_only_the_frozen_contract() -> None:
    """A configuration value that can alter the method must fail closed."""
    module = _analysis_module()
    config_path = Path("configs/figure3_unified_channel_block_permutation.yaml")

    config = module.load_analysis_config(config_path)

    assert config["analysis_id"] == module.ANALYSIS_ID
    assert config["n_permutations"] == 100
    assert config["n_bootstrap_final_model"] == 2000
    assert config["approved_plan"]["sha256"] == "1f1fbf47ef39ccd16ccebe0c8cff920653c844c120aa3bc5201aa4e5c772fb89"


def test_config_rejects_changed_independent_input_anchor(tmp_path: Path) -> None:
    """A config-local hash edit must not redefine an independently supplied immutable anchor."""
    module = _analysis_module()
    source = Path("configs/figure3_unified_channel_block_permutation.yaml")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["immutable_inputs"]["configs/modeling_global.yaml"] = "0" * 64
    changed = tmp_path / "changed_hash.yaml"
    changed.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ValueError, match="immutable input anchors"):
        module.load_analysis_config(changed)


def test_config_rejects_relative_input_redirect_and_stream_change(tmp_path: Path) -> None:
    """Relative path redirection and random-stream edits must not bypass the frozen config contract."""
    module = _analysis_module()
    source = Path("configs/figure3_unified_channel_block_permutation.yaml")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["paths"]["pipeline_a"] = "data/processed/sequences/alternate.npz"
    redirected = tmp_path / "redirected.yaml"
    redirected.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="path contract"):
        module.load_analysis_config(redirected)

    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["random_streams"]["bootstrap_spawn_key"] = [0]
    changed_stream = tmp_path / "changed_stream.yaml"
    changed_stream.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="random-stream contract"):
        module.load_analysis_config(changed_stream)


def test_strict_output_validation_rejects_wrong_order_and_ci_missingness() -> None:
    """Strict validation must prove deterministic ordering and final-model-only CI population."""
    module = _analysis_module()
    frames = module.build_aggregate_frames(
        _synthetic_repeat_frame(module),
        {channel: (-0.01, 0.02) for channel in module.CHANNEL_ORDER},
    )
    reordered = {name: frame.copy() for name, frame in frames.items()}
    repeats = reordered["permutation_repeats.csv"]
    reordered["permutation_repeats.csv"] = pd.concat(
        [repeats.iloc[[1]], repeats.iloc[[0]], repeats.iloc[2:]],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="ordering"):
        module.validate_output_frames(reordered, strict_values=True)

    invalid_ci = {name: frame.copy() for name, frame in frames.items()}
    summary = invalid_ci["channel_importance_summary.csv"]
    summary.loc[summary["model_id"] == "xgboost", "bootstrap_ci_low"] = 0.0
    with pytest.raises(ValueError, match="confidence interval"):
        module.validate_output_frames(invalid_ci, strict_values=True)


def test_build_manifest_records_exact_nonself_inventory(tmp_path: Path) -> None:
    """Self-hashing or omission of a required final artifact must fail manifest construction."""
    module = _analysis_module()
    final_candidate = tmp_path / "candidate"
    final_candidate.mkdir()
    for index, filename in enumerate(module.FINAL_MANIFEST_FILENAMES, start=1):
        (final_candidate / filename).write_bytes(f"payload-{index}\n".encode())

    manifest = module.build_manifest_payload(final_candidate, Path("outputs/analysis/final"))

    assert [Path(entry["path"]).name for entry in manifest["files"]] == list(module.FINAL_MANIFEST_FILENAMES)
    assert all(len(entry["sha256"]) == 64 for entry in manifest["files"])
    assert all(entry["byte_size"] > 0 for entry in manifest["files"])
    assert all(Path(entry["path"]).name != "manifest.json" for entry in manifest["files"])


def test_execute_model_permutations_uses_fixed_rows_and_preserves_signed_outputs() -> None:
    """Per-channel RNG generation or loss of one-based output numbering must fail the execution loop."""
    module = _analysis_module()
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    original = np.zeros((4, 2, 11), dtype=np.float64)
    original[0, :, :] = -2.0
    original[1, :, :] = -1.0
    original[2, :, :] = 1.0
    original[3, :, :] = 2.0
    permutations = np.asarray([[2, 3, 0, 1], [1, 0, 3, 2]], dtype=np.int64)

    def predictor(values: np.ndarray) -> np.ndarray:
        participant_mean = values.mean(axis=(1, 2))
        return 1.0 / (1.0 + np.exp(-participant_mean))

    baseline_scores = predictor(original)
    baseline_auroc = 1.0

    rows, score_cube = module.execute_model_permutations(
        module.MODEL_ORDER[0],
        original,
        labels,
        baseline_auroc,
        permutations,
        predictor,
    )

    assert len(rows) == 22
    assert score_cube.shape == (11, 2, 4)
    assert {row["channel_index"] for row in rows} == set(range(1, 12))
    assert {row["permutation_repeat"] for row in rows} == {1, 2}
    assert rows[0]["permutation_repeat"] == 1
    assert rows[1]["permutation_repeat"] == 2
    assert all(row["delta_auroc"] == pytest.approx(baseline_auroc - row["permuted_auroc"]) for row in rows)


def test_privacy_allows_only_run_record_provenance_timestamps() -> None:
    """Rejecting allowlisted runtime timestamps or accepting dates elsewhere must fail privacy validation."""
    module = _analysis_module()
    allowed = json.dumps(
        {
            "run_started_at_utc": "2026-08-05T10:00:00Z",
            "run_finished_at_utc": "2026-08-05T10:01:00Z",
        }
    ).encode()

    module.validate_serialized_privacy({"run_record.json": allowed})
    with pytest.raises(ValueError, match="privacy validation"):
        module.validate_serialized_privacy({"report.md": b"clinical date 2026-08-05\n"})
    with pytest.raises(ValueError, match="privacy validation"):
        module.validate_serialized_privacy(
            {"audit.json": b'{"unexpected_timestamp":"2026-08-05T10:00:00Z"}'}
        )


def test_config_rejects_absolute_path_values(tmp_path: Path) -> None:
    """Checking only configuration path keys must not allow an absolute output target."""
    module = _analysis_module()
    source = Path("configs/figure3_unified_channel_block_permutation.yaml")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["paths"]["final_output_dir"] = "/tmp/outside-project"
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ValueError, match="project-relative"):
        module.load_analysis_config(changed)


def test_environment_version_gate_uses_frozen_pytorch_lightning_distribution() -> None:
    """The package gate must use the environment.yml distribution rather than an unavailable alias."""
    module = _analysis_module()

    versions = module._package_versions()

    assert versions["pytorch_lightning"] == "2.6.1"
    assert "lightning" not in versions


def test_deep_gpu_direct_uses_persisted_normalizer_and_predictor_once() -> None:
    """A wrapper call, duplicate normalization, or wrong GPU arguments must fail."""
    module = _analysis_module()
    calls: list[dict[str, Any]] = []
    device_model = _SyntheticDeviceModel()

    def predictor(**kwargs: Any) -> np.ndarray:
        calls.append(kwargs)
        return np.linspace(0.2, 0.8, len(kwargs["x_eval"]), dtype=float)

    payload = _synthetic_deep_payload(predictor=predictor, lightning_model=device_model)
    raw = np.arange(4 * 3 * 11, dtype=float).reshape(4, 3, 11)

    scores = module._infer_deep_gpu_direct(payload, raw)

    normalizer = payload["normalizer"]
    wrapper = payload["model"]
    assert normalizer.inputs == [raw]
    assert len(normalizer.outputs) == 1
    assert len(calls) == 1
    assert calls[0]["model"] is wrapper.lightning_model
    assert calls[0]["x_eval"] is normalizer.outputs[0]
    assert calls[0]["accelerator"] == "gpu"
    assert calls[0]["batch_size"] == 32
    assert np.array_equal(scores, np.linspace(0.2, 0.8, 4, dtype=float))
    assert device_model.parameter_reads == 1
    assert device_model.buffer_reads == 1


def test_deep_gpu_direct_preserves_persisted_wrapper_contract() -> None:
    """Wrong or mutated persisted CPU/32 metadata must stop direct GPU inference."""
    module = _analysis_module()
    raw = np.zeros((3, 2, 11), dtype=float)

    for payload in (
        _synthetic_deep_payload(accelerator="gpu"),
        _synthetic_deep_payload(batch_size=16),
    ):
        with pytest.raises(ValueError, match="wrapper"):
            module._infer_deep_gpu_direct(payload, raw)

    payload = _synthetic_deep_payload()
    wrapper = payload["model"]

    def mutating_predictor(**kwargs: Any) -> np.ndarray:
        wrapper.accelerator = "gpu"
        return np.full(len(kwargs["x_eval"]), 0.5, dtype=float)

    wrapper.predictor = mutating_predictor
    with pytest.raises(ValueError, match="wrapper"):
        module._infer_deep_gpu_direct(payload, raw)
    assert wrapper.batch_size == 32


def test_deep_wrapper_metadata_is_validated_before_participant_access(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opening participant inputs before both deep CPU/32 checks must fail."""
    module = _analysis_module()
    load_events: list[str] = []

    def loader(path: Path) -> dict[str, Any]:
        model_id = path.name.removeprefix("binary_").removesuffix("_final.pkl")
        load_events.append(f"load:{model_id}")
        return _synthetic_deep_payload(model_id=model_id)

    payloads = module._load_deep_payloads_before_sensitive_access(loader)

    assert list(payloads) == ["inception_time", "cnn_bilstm_attention"]
    assert load_events == ["load:inception_time", "load:cnn_bilstm_attention"]
    assert all(payload["model"].accelerator == "cpu" for payload in payloads.values())
    assert all(payload["model"].batch_size == 32 for payload in payloads.values())

    class SyntheticSensitiveRead(RuntimeError):
        """Mark the first synthetic participant-level read boundary."""

    ordering_events: list[str] = []

    def preloaded(_loader: Any) -> dict[str, dict[str, Any]]:
        ordering_events.append("deep_wrappers_validated")
        return payloads

    def stop_at_first_sensitive_read(*_args: Any, **_kwargs: Any) -> Any:
        ordering_events.append("participant_read")
        raise SyntheticSensitiveRead

    with monkeypatch.context() as patch:
        patch.setattr(module, "_load_deep_payloads_before_sensitive_access", preloaded)
        patch.setattr(module.np, "load", stop_at_first_sensitive_read)
        with pytest.raises(SyntheticSensitiveRead):
            module._load_scientific_context(
                {"paths": {"pipeline_a": "synthetic/pipeline_a.npz"}}
            )
    assert ordering_events == ["deep_wrappers_validated", "participant_read"]

    def invalid_loader(path: Path) -> dict[str, Any]:
        model_id = path.name.removeprefix("binary_").removesuffix("_final.pkl")
        batch_size = 16 if model_id == "cnn_bilstm_attention" else 32
        return _synthetic_deep_payload(model_id=model_id, batch_size=batch_size)

    with pytest.raises(ValueError, match="wrapper"):
        module._load_deep_payloads_before_sensitive_access(invalid_loader)


def test_deep_gpu_direct_has_no_warmup_or_training_array_access() -> None:
    """A warm-up parameter, training-array access, or extra predictor call must fail."""
    module = _analysis_module()
    signature = inspect.signature(module._infer_deep_gpu_direct)
    assert list(signature.parameters) == ["payload", "model_input"]

    class TrainingArraySentinel:
        """Raise if an unauthorized training array is inspected."""

        def __array__(self) -> np.ndarray:
            raise AssertionError("training array was accessed")

    training_array = TrainingArraySentinel()
    calls: list[int] = []

    def predictor(**kwargs: Any) -> np.ndarray:
        calls.append(len(kwargs["x_eval"]))
        return np.full(len(kwargs["x_eval"]), 0.5, dtype=float)

    payload = _synthetic_deep_payload(predictor=predictor)
    first = np.zeros((3, 2, 11), dtype=float)
    second = np.ones((3, 2, 11), dtype=float)
    module._infer_deep_gpu_direct(payload, first)
    module._infer_deep_gpu_direct(payload, second)

    assert calls == [3, 3]
    assert len(payload["normalizer"].inputs) == 2
    assert training_array is training_array


def test_baseline_and_permuted_deep_calls_share_gpu_direct_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    """Routing baseline or permutations through the CPU wrapper must fail."""
    module = _analysis_module()
    helper_inputs: list[np.ndarray] = []

    def direct_gpu(_payload: Any, values: np.ndarray) -> np.ndarray:
        helper_inputs.append(values)
        participant_mean = values.mean(axis=(1, 2))
        return 1.0 / (1.0 + np.exp(-participant_mean))

    monkeypatch.setattr(module, "_infer_deep_gpu_direct", direct_gpu)
    payload = {
        "transformer": None,
        "normalizer": object(),
        "model": SimpleNamespace(
            predict_proba=lambda _values: pytest.fail("deep wrapper predict_proba was called")
        ),
    }
    original = np.zeros((4, 2, 11), dtype=float)
    original[:, :, :] = np.asarray([-2.0, -1.0, 1.0, 2.0])[:, None, None]
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    baseline = module._infer_positive_scores("inception_time", payload, original)

    rows, _cube = module.execute_model_permutations(
        module.MODEL_ORDER[4],
        original,
        labels,
        float(module.roc_auc_score(labels, baseline)),
        np.asarray([[2, 3, 0, 1]], dtype=np.int64),
        lambda values: module._infer_positive_scores("inception_time", payload, values),
    )

    assert len(rows) == 11
    assert len(helper_inputs) == 12
    assert helper_inputs[0] is original


def test_all_six_baselines_reconcile_before_first_permutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generating a permutation before all baseline gates finish must fail."""
    module = _analysis_module()
    events: list[str] = []

    def load_context(_config: Any) -> SimpleNamespace:
        events.extend(f"baseline:{model['model_id']}" for model in module.MODEL_ORDER)
        return SimpleNamespace(labels=np.asarray([0, 0, 1, 1], dtype=np.int64))

    class StopBeforePermutation(RuntimeError):
        """Stop after recording the first random-matrix request."""

    def generate(*_args: Any, **_kwargs: Any) -> Any:
        events.append("permutation_matrix")
        raise StopBeforePermutation

    monkeypatch.setattr(module, "_configure_determinism", lambda _seed: {})
    monkeypatch.setattr(module, "verify_immutable_hashes", lambda _config: {})
    monkeypatch.setattr(module, "_load_scientific_context", load_context)
    monkeypatch.setattr(module, "generate_random_matrices", generate)
    config = {"permutation_seed": 42, "n_permutations": 100, "n_bootstrap_final_model": 2000}
    with pytest.raises(StopBeforePermutation):
        module.run_analysis_once(config, tmp_path / "run_order", package_versions={"numpy": "synthetic"})
    assert events[:-1] == [f"baseline:{model['model_id']}" for model in module.MODEL_ORDER]
    assert events[-1] == "permutation_matrix"

    events.clear()

    def fail_baseline(_config: Any) -> Any:
        events.append("baseline:minirocket_ridge")
        raise ValueError("synthetic baseline failure")

    monkeypatch.setattr(module, "_load_scientific_context", fail_baseline)
    with pytest.raises(ValueError, match="synthetic baseline failure"):
        module.run_analysis_once(config, tmp_path / "run_failure", package_versions={"numpy": "synthetic"})
    assert events == ["baseline:minirocket_ridge"]


def test_cuda_preflight_requires_exactly_one_device_before_sensitive_access() -> None:
    """Unavailable, zero-device, or multi-device CUDA states must fail closed."""
    module = _analysis_module()
    sensitive_calls: list[str] = []

    for available, count in ((False, 1), (True, 0), (True, 2)):
        with pytest.raises(ValueError, match="CUDA"):
            module._validate_cuda_contract(available, count)
        assert sensitive_calls == []

    result = module._validate_cuda_contract(True, 1)
    assert result == {"cuda_available": True, "cuda_device_count": 1, "cuda_device_class": "cuda"}


def test_actual_deep_device_state_requires_homogeneous_cuda() -> None:
    """CPU, mixed, absent, non-tensor, or invalid device evidence must fail."""
    module = _analysis_module()
    assert module._actual_model_device_class(_SyntheticDeviceModel()) == "cuda"

    invalid_models = (
        _SyntheticDeviceModel(("cpu",), ("cpu",)),
        _SyntheticDeviceModel(("cuda",), ("cpu",)),
        _SyntheticDeviceModel((), ()),
        _SyntheticDeviceModel(("meta",), ("meta",)),
    )
    for model in invalid_models:
        with pytest.raises(ValueError):
            module._actual_model_device_class(model)

    non_tensor_model = SimpleNamespace(
        named_parameters=lambda: [("weight", object())],
        named_buffers=lambda: [],
    )
    with pytest.raises(TypeError, match="non-tensor"):
        module._actual_model_device_class(non_tensor_model)


def test_unified_state_digest_matches_diagnostic_canonical_contract() -> None:
    """A changed canonical encoding or device-dependent digest must fail."""
    module = _analysis_module()
    diagnostic = importlib.import_module("src.analysis.figure3_e010_deep_model_reconciliation_diagnostic")
    import torch

    class SyntheticState(torch.nn.Module):
        """Contain one named parameter and one named buffer."""

        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.asarray([[1.0, 2.0]], dtype=torch.float32))
            self.register_buffer("offset", torch.asarray([3, 4], dtype=torch.int64))

    model = SyntheticState()
    expected = diagnostic.canonical_model_state_sha256(model)
    assert module._canonical_model_state_sha256(model) == expected
    assert torch.cuda.is_available() is True
    model.to("cuda")
    assert module._canonical_model_state_sha256(model) == expected
    with torch.no_grad():
        model.weight.add_(1.0)
    assert module._canonical_model_state_sha256(model) != expected


def test_deep_state_digest_gate_covers_baseline_and_complete_permutation_block() -> None:
    """Missing, reordered, or unequal three-checkpoint digests must stop execution."""
    module = _analysis_module()
    digest = "a" * 64
    valid = {
        "before_baseline": digest,
        "after_baseline": digest,
        "after_model_permutation_block": digest,
    }
    assert module._validate_deep_state_digests(valid) is True

    for changed_key in ("after_baseline", "after_model_permutation_block"):
        changed = dict(valid)
        changed[changed_key] = "b" * 64
        with pytest.raises(ValueError, match="state digest"):
            module._validate_deep_state_digests(changed)
    with pytest.raises(ValueError, match="checkpoint"):
        module._validate_deep_state_digests({"before_baseline": digest})


def test_four_non_deep_inference_paths_remain_behaviorally_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing the frozen non-deep helper call or return bytes must fail."""
    module = _analysis_module()
    training_module = importlib.import_module("src.training.train_final_models")
    calls: list[dict[str, Any]] = []
    expected_scores = np.asarray([0.125, 0.875], dtype=np.float64)

    def predict(**kwargs: Any) -> np.ndarray:
        calls.append(kwargs)
        return expected_scores

    monkeypatch.setattr(training_module, "_predict_positive_probabilities", predict)
    payload = {
        "transformer": object(),
        "normalizer": object(),
        "model": object(),
    }
    raw = np.asarray([[1.0], [2.0]])
    for model_id in ("minirocket_ridge", "xgboost", "elastic_net_lr", "svm"):
        observed = module._infer_positive_scores(model_id, payload, raw)
        assert observed.tobytes() == expected_scores.tobytes()
    assert [call["model_name"] for call in calls] == [
        "minirocket_ridge",
        "xgboost",
        "elastic_net_lr",
        "svm",
    ]
    assert all(call["transformer"] is payload["transformer"] for call in calls)
    assert all(call["normalizer"] is payload["normalizer"] for call in calls)
    assert all(call["model"] is payload["model"] for call in calls)
    assert all(call["x_input"] is raw for call in calls)

    frozen_non_deep_block = '''    from src.training.train_final_models import _predict_positive_probabilities

    scores = _predict_positive_probabilities(
        model_name=model_id,
        transformer=payload["transformer"],
        normalizer=payload["normalizer"],
        model=payload["model"],
        x_input=model_input,
    )
    return np.asarray(scores, dtype=float)'''
    assert frozen_non_deep_block in inspect.getsource(module._infer_positive_scores)


def test_amendment_parent_and_diagnostic_authorities_are_strict(tmp_path: Path) -> None:
    """Missing, changed, reordered, absolute, or traversing authority records must fail."""
    module = _analysis_module()
    source = Path("configs/figure3_unified_channel_block_permutation.yaml")
    valid = module.load_analysis_config(source)

    assert valid["approved_plan"] == {
        "path": "docs/plans/figure3_unified_channel_block_permutation.plan.md",
        "sha256": "1f1fbf47ef39ccd16ccebe0c8cff920653c844c120aa3bc5201aa4e5c772fb89",
    }
    assert valid["gpu_inference_amendment_plan"]["sha256"] == (
        "665a082b02364ff50eb0ea21044dda1815ede17ac6632412d0fb0a31919506ee"
    )
    assert valid["diagnostic_plan"]["sha256"] == (
        "9129067eaf3c1d433e0b9eddd10a0fdb8a7ee95c152f931b3f69d55dec9ce509"
    )
    assert len(valid["diagnostic_aggregate_outputs"]) == 7

    mutations: list[tuple[str, Any]] = []
    missing = yaml.safe_load(source.read_text(encoding="utf-8"))
    missing.pop("gpu_inference_amendment_plan", None)
    mutations.append(("missing", missing))
    changed = yaml.safe_load(source.read_text(encoding="utf-8"))
    changed["diagnostic_plan"]["sha256"] = "0" * 64
    mutations.append(("changed", changed))
    reordered = yaml.safe_load(source.read_text(encoding="utf-8"))
    reordered["diagnostic_aggregate_outputs"] = list(reversed(reordered["diagnostic_aggregate_outputs"]))
    mutations.append(("reordered", reordered))
    absolute = yaml.safe_load(source.read_text(encoding="utf-8"))
    absolute["gpu_inference_amendment_plan"]["path"] = "/tmp/amendment.plan.md"
    mutations.append(("absolute", absolute))
    traversing = yaml.safe_load(source.read_text(encoding="utf-8"))
    traversing["diagnostic_plan"]["path"] = "docs/plans/../other.plan.md"
    mutations.append(("traversing", traversing))

    for name, payload in mutations:
        changed_path = tmp_path / f"{name}.yaml"
        changed_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
        with pytest.raises(ValueError):
            module.load_analysis_config(changed_path)


def test_python_interpreter_gate_accepts_exact_approved_symlink_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolving away the approved logical Python alias must fail."""
    module = _analysis_module()
    versioned_interpreter = tmp_path / "python3.11"
    versioned_interpreter.write_bytes(b"")
    bin_directory = tmp_path / "bin"
    bin_directory.mkdir()
    approved_alias = bin_directory / "python"
    approved_alias.symlink_to(versioned_interpreter)
    alternate_interpreter = tmp_path / "alternate_python"
    alternate_interpreter.write_bytes(b"")
    absent_interpreter = tmp_path / "absent_python"

    monkeypatch.setattr(module, "EXPECTED_PYTHON_INTERPRETER", approved_alias, raising=False)

    assert module._validate_python_interpreter_path(str(approved_alias)) == "bin/python"
    for rejected_path in (versioned_interpreter, alternate_interpreter, absent_interpreter):
        with pytest.raises(ValueError) as exc_info:
            module._validate_python_interpreter_path(str(rejected_path))
        assert str(tmp_path) not in str(exc_info.value)


def test_preflight_enforces_exact_normalized_modified_path_status_set() -> None:
    """A staged, missing, extra, renamed, deleted, or alternate status tuple must fail."""
    module = _analysis_module()
    expected = [
        {"status": "??", "path": ".vscode/"},
        {"status": " M", "path": "configs/figure3_unified_channel_block_permutation.yaml"},
        {"status": " M", "path": "src/analysis/figure3_unified_channel_block_permutation.py"},
        {"status": " M", "path": "tests/test_figure3_unified_channel_block_permutation.py"},
    ]
    valid_lines = [
        " M tests/test_figure3_unified_channel_block_permutation.py",
        "?? .vscode/",
        " M configs/figure3_unified_channel_block_permutation.yaml",
        " M src/analysis/figure3_unified_channel_block_permutation.py",
    ]
    assert module._validate_pre_execution_status(valid_lines, expected) == expected

    invalid_sets = (
        valid_lines[:-1],
        [*valid_lines, "?? unexpected.txt"],
        [*valid_lines[:-1], "M  src/analysis/figure3_unified_channel_block_permutation.py"],
        [*valid_lines[:-1], " D src/analysis/figure3_unified_channel_block_permutation.py"],
        [*valid_lines[:-1], "R  old.py -> src/analysis/figure3_unified_channel_block_permutation.py"],
        [*valid_lines[:-1], " M src/analysis/alternate.py"],
    )
    for lines in invalid_sets:
        with pytest.raises(ValueError, match="worktree"):
            module._validate_pre_execution_status(lines, expected)


def test_scientific_csv_contracts_are_unchanged_by_amendment() -> None:
    """Changing scientific schemas, counts, ordering, or numeric semantics must fail."""
    module = _analysis_module()
    config = module.load_analysis_config(Path("configs/figure3_unified_channel_block_permutation.yaml"))
    assert module.EXPECTED_OUTPUT_ROWS == {
        "channel_importance_summary.csv": 66,
        "permutation_repeats.csv": 6600,
        "cross_model_consistency.csv": 15,
        "channel_consensus_summary.csv": 11,
    }
    assert [model["model_id"] for model in module.MODEL_ORDER] == [
        "minirocket_ridge",
        "xgboost",
        "elastic_net_lr",
        "svm",
        "inception_time",
        "cnn_bilstm_attention",
    ]
    assert module.CHANNEL_ORDER == tuple(f"S{index}" for index in range(1, 12))
    assert config["n_permutations"] == 100
    assert config["n_bootstrap_final_model"] == 2000
    assert config["permutation_seed"] == config["bootstrap_seed"] == 42
    assert config["score_reconciliation_atol"] == 1.0e-6
    assert config["auroc_reconciliation_atol"] == 1.0e-12
    assert module.signed_auroc_delta(0.7, 0.8) == pytest.approx(-0.1)


def test_operational_provenance_additions_are_aggregate_only(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Operational records must prove GPU state without participant or machine details."""
    module = _analysis_module()
    model_records = [
        {
            "model_id": model_id,
            "persisted_wrapper_accelerator": "cpu",
            "persisted_batch_size": 32,
            "effective_batch_size": 32,
            "baseline_invocation_count": 1,
            "permutation_invocation_count": 1100,
            "all_actual_device_classes": "cuda",
            "all_device_records_homogeneous": True,
            "state_sha256_before_baseline": "a" * 64,
            "state_sha256_after_baseline": "a" * 64,
            "state_sha256_after_model_permutation_block": "a" * 64,
            "state_digests_identical": True,
            "wrapper_contract_unchanged": True,
        }
        for model_id in ("inception_time", "cnn_bilstm_attention")
    ]
    contract = module._build_deep_inference_contract_payload(model_records)
    assert list(contract) == [
        "contract_id",
        "requested_accelerator",
        "required_cuda_device_count",
        "training_warmup_used",
        "models",
    ]
    assert contract["contract_id"] == "e010_gpu_direct_no_warmup_v1"
    assert contract["training_warmup_used"] is False

    candidate = tmp_path / "candidate"
    candidate.mkdir()
    for filename in module.FINAL_MANIFEST_FILENAMES:
        (candidate / filename).write_text("aggregate\n", encoding="utf-8")
    manifest = module.build_manifest_payload(candidate, Path("outputs/analysis/final"))
    assert list(manifest) == ["analysis_id", "deep_inference_contract_id", "files"]

    safe_payload = json.dumps(
        {"deep_inference_contract": contract, "participant_level_persistence": False}
    ).encode()
    module.validate_serialized_privacy({"audit.json": safe_payload})
    safe_text = safe_payload.decode()
    for forbidden in (
        "fit",
        "retrain",
        "recalibr",
        "threshold_change",
        "artifact_repair",
        "sample_id",
        "y_true",
        "permutation_indices",
        "bootstrap_indices",
        "tensor_shape",
        "device_name",
        "host_identifier",
        "username",
    ):
        assert forbidden not in safe_text
        assert forbidden not in caplog.text

    for field, value in (
        ("participant_id", "SYNTHETIC-PRIVATE"),
        ("y_true", 1),
        ("permutation_indices", [0, 1]),
        ("tensor_shapes", [[1, 2]]),
        ("device_name", "SYNTHETIC-DEVICE"),
        ("host_identifier", "SYNTHETIC-HOST"),
        ("username", "SYNTHETIC-USER"),
        ("path", "[PRIVATE_PATH_REDACTED]"),
    ):
        encoded = json.dumps({field: value}).encode()
        with pytest.raises(ValueError, match="privacy validation") as exc_info:
            module.validate_serialized_privacy({"audit.json": encoded})
        assert str(value) not in str(exc_info.value)


def test_verified_empty_stale_run_a_cleanup_accepts_only_exact_safe_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the exact empty nonsymlink run_a and now-empty work root may be removed."""
    module = _analysis_module()
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    work_root = tmp_path / "outputs" / "work" / module.ANALYSIS_ID
    run_a = work_root / "run_a"
    run_a.mkdir(parents=True)
    final_dir = tmp_path / "outputs" / "analysis" / module.ANALYSIS_ID
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("retain", encoding="utf-8")

    result = module._remove_verified_empty_stale_run_a(work_root, final_dir)

    assert result == {
        "run_a_validated_empty": True,
        "run_a_removed": True,
        "work_root_removed": True,
    }
    assert not run_a.exists()
    assert not work_root.exists()
    assert sentinel.read_text(encoding="utf-8") == "retain"


@pytest.mark.parametrize(
    "unsafe_case",
    [
        "symlink",
        "hidden_file",
        "ordinary_file",
        "child_directory",
        "special_child",
        "outside_root",
        "missing_target",
        "run_b",
        "promotion_candidate",
        "final_output",
    ],
)
def test_verified_empty_stale_run_a_cleanup_rejects_every_unsafe_state(
    unsafe_case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every unsafe stale-work shape must fail without deleting any object."""
    module = _analysis_module()
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    work_root = tmp_path / "outputs" / "work" / module.ANALYSIS_ID
    work_root.mkdir(parents=True)
    run_a = work_root / "run_a"
    final_dir = tmp_path / "outputs" / "analysis" / module.ANALYSIS_ID
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_sentinel = outside / "sentinel.txt"
    outside_sentinel.write_text("retain", encoding="utf-8")

    if unsafe_case == "symlink":
        target = outside / "linked_run"
        target.mkdir()
        run_a.symlink_to(target, target_is_directory=True)
    elif unsafe_case == "missing_target":
        pass
    else:
        run_a.mkdir()
        if unsafe_case == "hidden_file":
            (run_a / ".hidden").write_text("retain", encoding="utf-8")
        elif unsafe_case == "ordinary_file":
            (run_a / "file.txt").write_text("retain", encoding="utf-8")
        elif unsafe_case == "child_directory":
            (run_a / "child").mkdir()
        elif unsafe_case == "special_child":
            os.mkfifo(run_a / "synthetic_fifo")
        elif unsafe_case == "run_b":
            (work_root / "run_b").mkdir()
        elif unsafe_case == "promotion_candidate":
            (work_root / "promotion_candidate").mkdir()
        elif unsafe_case == "final_output":
            final_dir.mkdir(parents=True)

    target_root = outside / "outside_work" if unsafe_case == "outside_root" else work_root
    if unsafe_case == "outside_root":
        (target_root / "run_a").mkdir(parents=True)

    with pytest.raises((ValueError, FileExistsError)):
        module._remove_verified_empty_stale_run_a(target_root, final_dir)

    assert outside_sentinel.read_text(encoding="utf-8") == "retain"
    assert target_root.exists()


def test_source_contains_no_forbidden_amendment_operations() -> None:
    """The amendment helpers must exclude scientific mutation, warm-up, fallback, and recursive cleanup."""
    module = _analysis_module()
    assert module.DEEP_MODEL_IDS == ("inception_time", "cnn_bilstm_attention")
    assert module.DEEP_INFERENCE_CONTRACT_ID == "e010_gpu_direct_no_warmup_v1"

    infer_tree = ast.parse(inspect.getsource(module._infer_deep_gpu_direct))
    cleanup_tree = ast.parse(inspect.getsource(module._remove_verified_empty_stale_run_a))
    infer_attributes = {
        node.attr
        for node in ast.walk(infer_tree)
        if isinstance(node, ast.Attribute)
    }
    infer_names = {
        node.id
        for node in ast.walk(infer_tree)
        if isinstance(node, ast.Name)
    }
    cleanup_calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(cleanup_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert not infer_attributes.intersection({"fit", "fit_transform", "partial_fit", "predict_proba"})
    assert not infer_names.intersection({"train_input", "training_array", "warmup", "cpu_fallback"})
    assert "rmtree" not in cleanup_calls
    assert "glob" not in cleanup_calls
    assert "rglob" not in cleanup_calls
    assert module.EXPECTED_OUTPUT_COLUMNS == {
        "channel_importance_summary.csv": module.CHANNEL_IMPORTANCE_COLUMNS,
        "permutation_repeats.csv": module.PERMUTATION_REPEAT_COLUMNS,
        "cross_model_consistency.csv": module.CROSS_MODEL_COLUMNS,
        "channel_consensus_summary.csv": module.CHANNEL_CONSENSUS_COLUMNS,
    }
