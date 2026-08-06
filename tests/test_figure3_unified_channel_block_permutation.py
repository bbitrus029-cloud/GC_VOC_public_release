from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import ModuleType

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
