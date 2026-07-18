"""Contract tests for Session 1 shared post-hoc utilities."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.posthoc_final_common import (
    DEGENERATE_VARIANCE_EPSILON,
    atomic_write_json,
    AccessStateError,
    AccessStateTracker,
    build_registry_frame,
    build_session_1_access_state,
    confusion_from_scores,
    is_utc_iso8601_z,
    likelihood_ratio_metrics,
    load_csv_with_ordered_usecols,
    newcombe_accuracy_difference_ci,
    parse_date_only_ascii_slash,
    parse_date_series_ascii_slash,
    read_back_registry_csv,
    registry_readback_matches,
    sha256_bytes,
    sha256_file,
    single_auroc_delong_ci,
    validate_aligned_finite_1d_vectors,
    validate_completion_log,
    validate_exact_aligned_vectors,
    validate_exact_binary_labels_01,
    validate_freeze_commit,
    validate_preflight_privacy,
    validate_preflight_schema,
    validate_registry_records,
    wilson_proportion_ci,
)


def test_statistical_tests_source_hash_is_frozen() -> None:
    """The wrapper must depend on the approved immutable DeLong core."""
    source_path = Path("src/evaluation/statistical_tests.py")
    assert sha256_file(source_path) == "ef565e50c5fb380db200098adfca454eedda89a8ea71a14dc075680db9d4d5d0"
    assert DEGENERATE_VARIANCE_EPSILON == 1e-12


def test_sha256_helpers_hash_exact_bytes(tmp_path: Path) -> None:
    """Byte and file hashing must agree without decoding input."""
    payload = b"\x00opaque-container\n"
    path = tmp_path / "opaque.bin"
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert sha256_bytes(payload) == expected
    assert sha256_file(path) == expected


def test_delong_ordinary_separation_matches_independent_anchor() -> None:
    """Hard-coded anchor derives from hand midranks and the DeLong formula."""
    # Independent anchor: positive/negative placement midranks give AUC 8/9;
    # v01/v10 sample covariances yield 2/81, then z=norm.ppf(.975).
    result = single_auroc_delong_ci(
        np.array([0, 0, 0, 1, 1, 1]), np.array([0.1, 0.2, 0.4, 0.35, 0.6, 0.8])
    )
    assert result["auc"] == pytest.approx(0.8888888888888888)
    assert result["variance"] == pytest.approx(0.024691358024691357)
    assert result["ci_low_unclipped"] == pytest.approx(0.5809102612556272)
    assert result["ci_high_unclipped"] == pytest.approx(1.1968675165221505)
    assert result["ci_low"] == pytest.approx(0.5809102612556272)
    assert result["ci_high"] == 1.0
    assert result["ci_method"] == "delong_normal_clipped"
    assert result["ci_status"] == "estimable"
    assert result["bootstrap_seed"] is None


def test_delong_ties_match_independent_anchor_and_core_crosscheck() -> None:
    """AUC and variance use half credit for ties and agree with DeLong core."""
    # Independent midrank calculation: tied 0.5 values have rank 2; AUC=3/4
    # and the one-class DeLong covariance diagonal is 1/16.
    result = single_auroc_delong_ci(np.array([0, 0, 1, 1]), np.array([0.5, 0.5, 0.5, 0.9]))
    assert result["auc"] == pytest.approx(0.75)
    assert result["variance"] == pytest.approx(0.0625)
    assert abs(result["auc"] - result["delong_core_auc"]) <= 1e-12


@pytest.mark.parametrize("labels", [np.array([0, 0, 1]), np.array([0, 1, 1])])
def test_delong_single_class_observation_keeps_point_but_not_ci(labels: np.ndarray) -> None:
    """A DeLong CI needs at least two observations per outcome class."""
    result = single_auroc_delong_ci(labels, np.array([0.1, 0.4, 0.8]))
    assert result["auc"] is not None
    assert result["ci_status"] == "non_estimable"
    assert result["ci_nonestimable_reason"] == "insufficient_class_count_for_delong_ci"
    assert result["ci_low"] is None and result["ci_high"] is None


@pytest.mark.parametrize("labels", [np.array([0, 0]), np.array([1, 1])])
def test_delong_absent_outcome_class_is_nonestimable(labels: np.ndarray) -> None:
    """An absent outcome class cannot provide an AUROC point or CI."""
    result = single_auroc_delong_ci(labels, np.array([0.1, 0.9]))
    assert result["auc"] is None
    assert result["ci_status"] == "non_estimable"
    assert result["nonestimable_reason"] == "absent_outcome_class"
    assert result["ci_nonestimable_reason"] == "absent_outcome_class"


@pytest.mark.parametrize(
    ("scores", "labels"),
    [
        (np.array([0.5, 0.5, 0.5, 0.5]), np.array([0, 0, 1, 1])),
        (np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])),
    ],
)
def test_delong_degenerate_variance_never_emits_zero_width_ci(scores: np.ndarray, labels: np.ndarray) -> None:
    """Constant-score and perfect-separation inputs retain only point estimates."""
    result = single_auroc_delong_ci(labels, scores)
    assert result["auc"] is not None
    assert result["variance"] == 0.0
    assert result["ci_status"] == "non_estimable"
    assert result["ci_nonestimable_reason"] == "delong_variance_below_epsilon"
    assert result["ci_low"] is None and result["ci_high"] is None


@pytest.mark.parametrize(
    ("labels", "scores"),
    [
        (np.array([0, np.nan]), np.array([0.1, 0.9])),
        (np.array([0, 1]), np.array([0.1, np.inf])),
        (np.array([2, 3]), np.array([0.1, 0.9])),
    ],
)
def test_delong_rejects_nonfinite_and_nonbinary_inputs(labels: np.ndarray, scores: np.ndarray) -> None:
    """Labels are exact finite 0/1 values; arbitrary binary recoding is forbidden."""
    with pytest.raises((TypeError, ValueError)):
        single_auroc_delong_ci(labels, scores)


def test_delong_clips_lower_bound_for_inverted_scores() -> None:
    """The final normal lower confidence limit is clipped to zero."""
    result = single_auroc_delong_ci(
        np.array([1, 1, 1, 0, 0, 0]), np.array([0.1, 0.2, 0.4, 0.35, 0.6, 0.8])
    )
    assert result["ci_low_unclipped"] < 0.0
    assert result["ci_low"] == 0.0


def test_vector_and_binary_validators_require_strict_alignment() -> None:
    """Validation rejects missing, nonfinite, multidimensional, and recoded inputs."""
    validated = validate_aligned_finite_1d_vectors(
        y_true=np.array([0, 1]), scores=np.array([0.2, 0.8])
    )
    assert list(validated) == ["y_true", "scores"]
    assert np.array_equal(validate_exact_binary_labels_01(np.array([0, 1, 0])), np.array([0, 1, 0]))
    with pytest.raises(ValueError):
        validate_aligned_finite_1d_vectors(values=np.array([[1.0]]))
    with pytest.raises(ValueError):
        validate_aligned_finite_1d_vectors(left=np.array([1.0]), right=np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        validate_exact_binary_labels_01(np.array([0, 2]))
    with pytest.raises(ValueError):
        validate_exact_aligned_vectors(
            expected_length=2,
            finite_names=("scores",),
            scores=np.array([0.1, 0.2]),
            labels=np.array([0]),
        )


def test_confusion_threshold_includes_exactly_equal_score() -> None:
    """The frozen threshold comparison is p_raw >= threshold."""
    result = confusion_from_scores(
        y_true=np.array([1, 1, 0, 0]), scores=np.array([0.5, 0.49, 0.5, 0.1]), threshold=0.5
    )
    assert result == {"tp": 1, "fn": 1, "tn": 1, "fp": 1}


def test_likelihood_ratios_uncorrected_points_and_log_ci() -> None:
    """The LR contract preserves uncorrected counts and standard log intervals."""
    result = likelihood_ratio_metrics(tp=3, fn=1, tn=2, fp=1)
    assert result["lr_plus"] == pytest.approx(2.25)
    assert result["lr_minus"] == pytest.approx(0.375)
    assert result["se_log_lr_plus"] == pytest.approx(np.sqrt(1 / 3 - 1 / 4 + 1 - 1 / 3))
    assert result["zero_cell_correction"] is None
    assert result["lr_plus_ci_low"] < result["lr_plus"] < result["lr_plus_ci_high"]
    assert result["lr_minus_ci_low"] < result["lr_minus"] < result["lr_minus_ci_high"]


def test_likelihood_ratios_haldane_corrects_all_cells() -> None:
    """Any zero confusion cell corrects all four LR cells only."""
    result = likelihood_ratio_metrics(tp=4, fn=0, tn=3, fp=1)
    assert result["zero_cell_correction"] == "haldane_anscombe_all_cells"
    assert result["corrected_counts"] == {"tp": 4.5, "fn": 0.5, "tn": 3.5, "fp": 1.5}
    assert result["uncorrected_counts"] == {"tp": 4, "fn": 0, "tn": 3, "fp": 1}


@pytest.mark.parametrize("counts", [(0, 0, 3, 1), (3, 1, 0, 0)])
def test_likelihood_ratios_absent_class_is_not_invented(counts: tuple[int, int, int, int]) -> None:
    """The correction does not manufacture an absent outcome class."""
    result = likelihood_ratio_metrics(*counts)
    assert result["estimability_status"] == "non_estimable"
    assert result["nonestimable_reason"] == "absent_outcome_class"


def test_likelihood_ratios_reject_invalid_denominator() -> None:
    """Negative counts produce a stable invalid-denominator failure."""
    result = likelihood_ratio_metrics(tp=1, fn=-1, tn=2, fp=1)
    assert result["estimability_status"] == "non_estimable"
    assert result["nonestimable_reason"] == "invalid_lr_denominator"


def test_wilson_and_newcombe_intervals_match_statsmodels_anchors() -> None:
    """Configured statsmodels Wilson and Newcombe methods are used exactly."""
    assert wilson_proportion_ci(3, 4) == pytest.approx((0.30064184258240184, 0.9544127391902995))
    assert wilson_proportion_ci(2, 3) == pytest.approx((0.2076596008020477, 0.9385080552796037))
    assert wilson_proportion_ci(5, 7) == pytest.approx((0.3589344518326191, 0.9177810759959432))
    assert newcombe_accuracy_difference_ci(4, 5, 2, 6) == pytest.approx(
        (-0.09424348411669564, 0.7543889036266154)
    )


@pytest.mark.parametrize("value", ["2025/1/31", "2000/02/29"])
def test_date_parser_accepts_only_exact_date_only_text(value: str) -> None:
    """Date parsing allows exact ASCII slash date strings only."""
    assert parse_date_only_ascii_slash(value).strftime("%Y-%m-%d") in {"2025-01-31", "2000-02-29"}


@pytest.mark.parametrize("value", [" 2025/1/31", "2025/1/31 ", "2025/1/31T00:00:00", "2025-01-31", "２０２５/1/1", None, "bad"])
def test_date_parser_rejects_without_echoing_private_value(value: object) -> None:
    """Date failures report aggregate-safe categories, never the supplied value."""
    with pytest.raises(ValueError) as error:
        parse_date_only_ascii_slash(value)
    assert str(value) not in str(error.value)


def test_date_series_parser_reports_only_aggregate_failure_counts() -> None:
    """Series parsing rejects all malformed values without exposing individual dates."""
    with pytest.raises(ValueError) as error:
        parse_date_series_ascii_slash(["2025/1/1", None, "2025-01-02", "2025/02/30"])
    message = str(error.value)
    assert "missing_or_nonstring_count=1" in message
    assert "invalid_date_format_count=1" in message
    assert "invalid_date_count=1" in message
    assert "2025-01-02" not in message


def test_date_parser_accepts_one_and_two_digit_month_day_components() -> None:
    """Valid ASCII slash grammar accepts one- or two-digit month and day tokens."""
    one_digit = parse_date_only_ascii_slash("2025/1/2")
    two_digit = parse_date_only_ascii_slash("2025/01/02")
    mixed = parse_date_series_ascii_slash(["2025/12/31", "2025/9/9"])
    assert one_digit.strftime("%Y-%m-%d") == "2025-01-02"
    assert two_digit.strftime("%Y-%m-%d") == "2025-01-02"
    assert [value.strftime("%Y-%m-%d") for value in mixed] == ["2025-12-31", "2025-09-09"]


def test_date_series_parser_rejects_normalization_collision_without_echo() -> None:
    """Distinct raw slash spellings that normalize identically fail closed."""
    with pytest.raises(ValueError) as error:
        parse_date_series_ascii_slash(["2025/1/2", "2025/01/02"])
    message = str(error.value)
    assert "normalization_collision_count=1" in message
    assert "2025/1/2" not in message
    assert "2025/01/02" not in message


def test_preflight_privacy_rejects_iso_and_slash_date_leaks_without_echo() -> None:
    """Privacy validation rejects both ISO-like and slash-like leaked date strings."""
    payload = _valid_preflight_payload()
    payload["note"] = "leaked 2025-07-17"
    with pytest.raises(ValueError) as error:
        validate_preflight_privacy(payload)
    assert "2025-07-17" not in str(error.value)
    payload = _valid_preflight_payload()
    payload["note"] = "leaked 2025/7/17"
    with pytest.raises(ValueError) as error:
        validate_preflight_privacy(payload)
    assert "2025/7/17" not in str(error.value)


def test_ordered_usecols_loader_rejects_broad_or_reordered_requests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every CSV read must request precisely the frozen ordered column list."""
    path = tmp_path / "table.csv"
    path.write_text("sample_id,y_true,p_raw\n1,0,0.1\n", encoding="utf-8")
    intercepted: list[list[str]] = []
    original = pd.read_csv

    def recording_read_csv(*args: object, **kwargs: object) -> pd.DataFrame:
        intercepted.append(list(kwargs["usecols"]))  # type: ignore[index]
        return original(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", recording_read_csv)
    frame = load_csv_with_ordered_usecols(path, ["sample_id", "y_true"], ["sample_id", "y_true"])
    assert list(frame.columns) == ["sample_id", "y_true"]
    assert intercepted == [["sample_id", "y_true"]]
    with pytest.raises(ValueError):
        load_csv_with_ordered_usecols(path, ["y_true", "sample_id"], ["sample_id", "y_true"])
    with pytest.raises(ValueError):
        load_csv_with_ordered_usecols(path, ["sample_id", "y_true", "p_raw"], ["sample_id", "y_true"])


def test_ordered_usecols_loader_allows_scores_only_when_expected(tmp_path: Path) -> None:
    """Score columns are permitted only when the caller's contract explicitly includes them."""
    path = tmp_path / "table.csv"
    path.write_text("sample_id,y_true,p_raw,p_cal,collection_location\n1,0,0.1,0.2,x\n", encoding="utf-8")
    frame = load_csv_with_ordered_usecols(path, ["sample_id", "p_raw"], ["sample_id", "p_raw"])
    assert list(frame.columns) == ["sample_id", "p_raw"]
    with pytest.raises(ValueError):
        load_csv_with_ordered_usecols(path, ["sample_id", "p_raw"], ["sample_id", "y_true"])
    with pytest.raises(ValueError):
        load_csv_with_ordered_usecols(path, ["sample_id", "collection_location"], ["sample_id", "collection_location"])


def test_ordered_usecols_loader_returns_request_order_not_file_order(tmp_path: Path) -> None:
    """Returned frames must use the frozen request order even when the file order differs."""
    path = tmp_path / "lineage.csv"
    path.write_text(
        "sample_id,collection_time,label,disease_category,in_final_output\n1,2025-01-01,0,1,True\n",
        encoding="utf-8",
    )
    requested = ["sample_id", "label", "disease_category", "collection_time", "in_final_output"]
    frame = load_csv_with_ordered_usecols(path, requested, requested)
    assert list(frame.columns) == requested


def test_freeze_commit_validation_uses_mocked_read_only_git() -> None:
    """Freeze validation requires exact ancestry, parent, paths, blobs, and approval evidence."""
    calls: list[tuple[str, ...]] = []
    commit = "3c0ff6b3b40460a2c340ad3193222ee54979109d"

    def fake_git(*args: str) -> str:
        calls.append(args)
        outputs = {
            ("cat-file", "-e", f"{commit}^{{commit}}"): "",
            ("merge-base", "--is-ancestor", commit, "HEAD"): "",
            ("rev-list", "--parents", "-n", "1", commit): f"{commit} cbb87b0380d7129798002ff4ea8b869c4b786582",
            ("diff-tree", "--no-commit-id", "--name-only", "-r", commit): (
                "configs/post_hoc_final_supplementary.yaml\n"
                "docs/modeling/final_post_hoc_supplementary_methods.md\n"
                "outputs/analysis/posthoc_final_freeze_manifest.json\n"
            ),
            ("rev-parse", f"{commit}:configs/post_hoc_final_supplementary.yaml"): "configblob",
            ("rev-parse", f"{commit}:docs/modeling/final_post_hoc_supplementary_methods.md"): "methodsblob",
            ("rev-parse", f"{commit}:outputs/analysis/posthoc_final_freeze_manifest.json"): "manifestblob",
            ("hash-object", "configs/post_hoc_final_supplementary.yaml"): "configblob",
            ("hash-object", "docs/modeling/final_post_hoc_supplementary_methods.md"): "methodsblob",
            ("hash-object", "outputs/analysis/posthoc_final_freeze_manifest.json"): "manifestblob",
        }
        return outputs[args]

    config = {
        "authority": {
            "baseline_head": "cbb87b0380d7129798002ff4ea8b869c4b786582",
            "reviewed_plan_path": ".cursor/plans/approved.plan.md",
            "reviewed_plan_sha256": "b" * 64,
            "approval_evidence": {
                "reviewed_plan_path": ".cursor/plans/approved.plan.md",
                "reviewed_plan_sha256": "b" * 64,
                "review_status": "approved",
                "user_confirmation": True,
            },
            "active_rules": [{"path": "rule", "sha256": "a" * 64}],
        },
        "outputs": {
            "session_0": {
                "config": "configs/post_hoc_final_supplementary.yaml",
                "methods": "docs/modeling/final_post_hoc_supplementary_methods.md",
                "freeze_manifest": "outputs/analysis/posthoc_final_freeze_manifest.json",
            }
        },
    }
    manifest = {
        "session_0_artifacts": {
            "config": {
                "path": "configs/post_hoc_final_supplementary.yaml",
                "sha256": "c" * 64,
            },
            "methods": {
                "path": "docs/modeling/final_post_hoc_supplementary_methods.md",
                "sha256": "d" * 64,
            },
            "freeze_manifest": {"path": "outputs/analysis/posthoc_final_freeze_manifest.json"},
        },
        "expected_session_0_changed_paths": [
            "configs/post_hoc_final_supplementary.yaml",
            "docs/modeling/final_post_hoc_supplementary_methods.md",
            "outputs/analysis/posthoc_final_freeze_manifest.json",
        ],
        "approval_evidence": {
            "reviewed_plan_path": ".cursor/plans/approved.plan.md",
            "reviewed_plan_sha256": "b" * 64,
            "review_status": "approved",
            "user_confirmation": True,
        },
        "reviewed_plan": {"path": ".cursor/plans/approved.plan.md", "sha256": "b" * 64},
        "active_rules": {"rule": {"sha256": "a" * 64}},
    }
    result = validate_freeze_commit(
        commit,
        config,
        manifest,
        git_runner=fake_git,
        rule_hashes={"rule": "a" * 64},
        artifact_hashes={"config": "c" * 64, "methods": "d" * 64},
    )
    assert all(value is True for key, value in result.items() if key != "freeze_commit_id")
    assert result["freeze_commit_id"] == commit
    assert ("hash-object", "configs/post_hoc_final_supplementary.yaml") in calls
    assert all("status" not in call and "reset" not in call for call in calls)


def _valid_preflight_payload() -> dict[str, object]:
    """Build a minimal valid aggregate-only Session 1 preflight."""
    hashes = OrderedDict(
        [
            ("config_sha256", "a" * 64),
            ("methods_sha256", "b" * 64),
            ("freeze_manifest_sha256", "c" * 64),
            ("frozen_input_whole_container_sha256", OrderedDict([("fixture", "d" * 64)])),
        ]
    )
    return {
        "schema_version": "posthoc_final_score_blind_preflight.v1",
        "package_id": "posthoc_final_supplementary",
        "generated_at_utc": "2026-07-17T00:00:00Z",
        "hashes": hashes,
        "freeze_commit_validation": {
            "freeze_commit_id": "3c0ff6b3b40460a2c340ad3193222ee54979109d",
            "exists": True,
            "is_ancestor_of_head": True,
            "parent_is_baseline_head": True,
            "changed_paths_exact": True,
            "config_blob_matches": True,
            "methods_blob_matches": True,
            "manifest_blob_matches": True,
            "approval_evidence_matches": True,
            "active_rule_hashes_match": True,
        },
        "aggregate_counts": OrderedDict(
            [
                ("source_cohort_n", 1040),
                ("modeling_cohort_n", 972),
                ("train_n", 777),
                ("internal_test_n", 195),
                (
                    "difficult_spectrum_by_view",
                    OrderedDict(
                        [
                            (
                                view,
                                OrderedDict(
                                    [
                                        ("n", 4),
                                        ("n_positive", 2),
                                        ("n_negative", 2),
                                        (
                                            "disease_category_counts",
                                            OrderedDict(
                                                [("0", 0), ("1", 1), ("2", 1), ("3", 1), ("4", 1), ("5", 0)]
                                            ),
                                        ),
                                    ]
                                ),
                            )
                            for view in ("oof", "test")
                        ]
                    ),
                ),
                (
                    "date_feasibility",
                    OrderedDict([("n_dates_overall", 40), ("n_dates_with_both_endpoint_classes", 34)]),
                ),
                (
                    "qc5_reconciliation",
                    OrderedDict(
                        [
                            ("n_qc5_evaluable_source_rows", 974),
                            ("n_strict_fail_among_evaluable", 298),
                            ("n_strict_pass_among_evaluable", 676),
                        ]
                    ),
                ),
            ]
        ),
        "one_to_one_alignment_status": "pass",
        "train_test_disjointness_status": "pass",
        "known_anchor_reconciliation": OrderedDict(
            [("date_40_34", True), ("qc5_974_298_676", True), ("status", "pass")]
        ),
        "status": "pass",
        "reason_code": None,
        "access_state": _successful_preflight_access_state(),
    }


def _successful_preflight_access_state() -> OrderedDict[str, OrderedDict[str, bool]]:
    """Return the exact successful score-blind Session 1 activity and cumulative state."""
    tracker = AccessStateTracker()
    for key in (
        "prediction_container_bytes_hashed",
        "prediction_schema_or_header_inspected",
        "participant_level_label_values_read",
        "participant_level_date_values_read",
        "participant_level_qc_values_read",
    ):
        tracker.mark(key)
    return tracker.snapshot()


def test_preflight_schema_and_privacy_validation_enforce_order_and_allowlists() -> None:
    """Preflight accepts aggregate values and policy names, rejecting private output."""
    payload = _valid_preflight_payload()
    validate_preflight_schema(payload, frozen_hash_paths=["fixture"])
    validate_preflight_privacy(payload)
    assert list(payload) == [
        "schema_version",
        "package_id",
        "generated_at_utc",
        "hashes",
        "freeze_commit_validation",
        "aggregate_counts",
        "one_to_one_alignment_status",
        "train_test_disjointness_status",
        "known_anchor_reconciliation",
        "status",
        "reason_code",
        "access_state",
    ]
    payload["policy"] = "sample_id filename collection_time collection_location permitted by policy"
    validate_preflight_privacy(payload)
    payload["participant_rows"] = [{"sample_id": 1}]
    with pytest.raises(ValueError):
        validate_preflight_privacy(payload)
    payload = _valid_preflight_payload()
    payload["absolute_path"] = "[PRIVATE_PATH_REDACTED]"
    with pytest.raises(ValueError):
        validate_preflight_privacy(payload)
    payload = _valid_preflight_payload()
    payload["note"] = "2025/1/2"
    with pytest.raises(ValueError) as error:
        validate_preflight_privacy(payload)
    assert "2025/1/2" not in str(error.value)
    payload = _valid_preflight_payload()
    payload["known_anchor_reconciliation"]["date_40_34"] = False  # type: ignore[index]
    with pytest.raises(ValueError):
        validate_preflight_schema(payload, frozen_hash_paths=["fixture"])


def test_atomic_json_write_validates_reads_back_and_refuses_overwrite(tmp_path: Path) -> None:
    """Promotion happens only after validation and never replaces a stale target."""
    target = tmp_path / "preflight.json"
    payload = _valid_preflight_payload()
    atomic_write_json(
        target,
        payload,
        schema_validator=lambda data: validate_preflight_schema(data, frozen_hash_paths=["fixture"]),
        privacy_validator=validate_preflight_privacy,
    )
    assert json.loads(target.read_text(encoding="utf-8"))["status"] == "pass"
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(FileExistsError):
        atomic_write_json(target, payload, lambda _: None, lambda _: None)


def test_session_1_access_state_is_complete_and_score_blind() -> None:
    """The tracker starts false and records only actual completed access."""
    state = build_session_1_access_state()
    for scope in ("session_activity", "cumulative_through_session"):
        assert list(state[scope]) == [
            "prediction_container_bytes_hashed",
            "prediction_schema_or_header_inspected",
            "participant_level_label_values_read",
            "participant_level_score_values_read",
            "participant_level_date_values_read",
            "participant_level_qc_values_read",
            "new_score_derived_results_computed",
            "auxiliary_model_fit_executed",
            "privacy_reference_loader_executed",
        ]
        assert state[scope] == {
            "prediction_container_bytes_hashed": False,
            "prediction_schema_or_header_inspected": False,
            "participant_level_label_values_read": False,
            "participant_level_score_values_read": False,
            "participant_level_date_values_read": False,
            "participant_level_qc_values_read": False,
            "new_score_derived_results_computed": False,
            "auxiliary_model_fit_executed": False,
            "privacy_reference_loader_executed": False,
        }
    tracker = AccessStateTracker()
    tracker.mark("participant_level_date_values_read")
    assert tracker.snapshot()["session_activity"]["participant_level_qc_values_read"] is False


# --- UTC grammar and sanitized access-state error (correction C) ----------------------------------


@pytest.mark.parametrize("value", ["2025-01-02T03:04:05Z", "2026-12-31T23:59:59Z"])
def test_is_utc_iso8601_z_accepts_exact_grammar(value: str) -> None:
    """Only the exact ``YYYY-MM-DDTHH:MM:SSZ`` grammar is accepted."""
    assert is_utc_iso8601_z(value) is True


@pytest.mark.parametrize(
    "value",
    ["2025-01-02T03:04:05", "2025/01/02T03:04:05Z", "2025-01-02 03:04:05Z", "2025-1-2T3:4:5Z", 12345, None],
)
def test_is_utc_iso8601_z_rejects_non_exact(value: object) -> None:
    """Missing Z, slash dates, loose components, and non-strings are rejected."""
    assert is_utc_iso8601_z(value) is False


def test_access_state_error_carries_sanitized_snapshot() -> None:
    """The sanitized error exposes an access-state snapshot without private echoes."""
    snapshot = AccessStateTracker().snapshot()
    error = AccessStateError("failed closed before promotion.", snapshot)
    assert error.access_state is snapshot
    assert "failed closed" in str(error)


# --- CSV registry validators (correction D) -------------------------------------------------------


def _synthetic_registry_entry() -> dict[str, object]:
    """Return a synthetic CSV registry contract exercising every supported dtype."""
    columns = ["analysis_id", "view", "n", "estimate", "flag"]
    return {
        "expected_row_count": 2,
        "column_order": columns,
        "dtype_map": {
            "analysis_id": "string",
            "view": "category",
            "n": "Int64",
            "estimate": "Float64",
            "flag": "boolean",
        },
        "nullable_map": {
            "analysis_id": False,
            "view": False,
            "n": False,
            "estimate": True,
            "flag": False,
        },
        "unique_key_tuple": ["analysis_id", "view"],
    }


def _synthetic_registry_records() -> list[dict[str, object]]:
    """Return two aggregate records that satisfy the synthetic registry contract."""
    return [
        {"analysis_id": "a1", "view": "oof", "n": 10, "estimate": 0.5, "flag": True},
        {"analysis_id": "a2", "view": "test", "n": 20, "estimate": None, "flag": False},
    ]


def test_build_registry_frame_and_readback_roundtrip(tmp_path: Path) -> None:
    """A validated frame reads back equivalently under the empty-field convention."""
    entry = _synthetic_registry_entry()
    frame = build_registry_frame(_synthetic_registry_records(), entry)
    assert list(frame.columns) == entry["column_order"]
    path = tmp_path / "aggregate.csv"
    frame.to_csv(path, index=False)
    read_back = read_back_registry_csv(path, entry)
    assert list(read_back.columns) == entry["column_order"]
    assert read_back.loc[1, "estimate"] == ""
    assert registry_readback_matches(frame, path, entry) is True


def test_validate_registry_records_rejects_key_and_count_violations() -> None:
    """Extra keys, missing keys, and wrong row counts are rejected."""
    entry = _synthetic_registry_entry()
    with_extra = _synthetic_registry_records()
    with_extra[0] = {**with_extra[0], "unexpected": 1}
    with pytest.raises(ValueError):
        validate_registry_records(with_extra, entry)
    with pytest.raises(ValueError):
        validate_registry_records(_synthetic_registry_records()[:1], entry)


def test_validate_registry_records_rejects_dtype_and_uniqueness() -> None:
    """Boolean-for-integer coercion and duplicate keys are rejected."""
    entry = _synthetic_registry_entry()
    bad_int = _synthetic_registry_records()
    bad_int[0]["n"] = True
    with pytest.raises(ValueError):
        validate_registry_records(bad_int, entry)
    duplicate = _synthetic_registry_records()
    duplicate[1] = {**duplicate[1], "analysis_id": "a1", "view": "oof"}
    with pytest.raises(ValueError):
        validate_registry_records(duplicate, entry)


@pytest.mark.parametrize("leak", ["2025-01-02", "2025/1/2", "/abs/path", "artifact.csv"])
def test_validate_registry_records_rejects_privacy_leaks(leak: str) -> None:
    """String cells with dates, absolute paths, or filenames are rejected."""
    entry = _synthetic_registry_entry()
    records = _synthetic_registry_records()
    records[0]["analysis_id"] = leak
    with pytest.raises(ValueError):
        validate_registry_records(records, entry)


# --- Completion-log grammar validator (correction H) ----------------------------------------------


def _completion_grammar() -> dict[str, object]:
    """Return a synthetic completion-log grammar accepted by the validator."""
    return {
        "record_contracts": {
            "header": {"fields_in_order": ["record_type", "schema_version", "package_id", "generated_at_utc"]},
            "session_identity": {
                "fields_in_order": ["record_type", "session_name", "completion_status"],
                "field_types": {"session_name": {"allowed_values": ["locked_scores", "date_grouped"]}},
            },
            "freeze_commit": {"fields_in_order": ["record_type", "freeze_commit"]},
            "output_hash_records": {"fields_in_order": ["record_type", "output_path", "sha256"]},
            "deviations_or_nonestimability_summary": {
                "fields_in_order": ["record_type", "summary_status", "reason_codes_csv", "note"]
            },
            "completion_receipt": {"fields_in_order": ["record_type", "completion_utc", "completion_status"]},
        }
    }


def _completion_binding() -> dict[str, object]:
    """Return a synthetic session binding for the locked-scores completion log."""
    return {
        "session_name": "locked_scores",
        "log_path": "outputs/logs/posthoc_final_locked_scores.log",
        "output_hash_record_count": 2,
        "ordered_output_inventory": [
            "outputs/analysis/one.csv",
            "outputs/analysis/two.csv",
        ],
        "must_neither_hash_nor_claim": ["outputs/analysis/posthoc_final_score_blind_preflight.json"],
    }


def _valid_completion_log() -> str:
    """Assemble a completion-log text that satisfies the synthetic grammar."""
    commit = "a" * 40
    digest = "b" * 64
    lines = [
        "\t".join(["header", "posthoc_final_completion_log.v1", "posthoc_final_supplementary", "2025-01-02T03:04:05Z"]),
        "\t".join(["session_identity", "locked_scores", "complete"]),
        "\t".join(["freeze_commit", commit]),
        "\t".join(["output_hash", "outputs/analysis/one.csv", digest]),
        "\t".join(["output_hash", "outputs/analysis/two.csv", digest]),
        "\t".join(["deviations_or_nonestimability_summary", "none", "", ""]),
        "\t".join(["access_state_session_activity", *(["true"] * 9)]),
        "\t".join(["access_state_cumulative_through_session", *(["true"] * 9)]),
        "\t".join(["completion_receipt", "2025-01-02T03:04:06Z", "complete"]),
    ]
    return "\n".join(lines) + "\n"


def test_validate_completion_log_accepts_valid_receipt() -> None:
    """A well-formed locked-scores completion log validates and returns its records."""
    expected = OrderedDict((key, True) for key in AccessStateTracker().snapshot()["session_activity"])
    prior = OrderedDict((key, False) for key in expected)
    records = validate_completion_log(
        _valid_completion_log(),
        grammar=_completion_grammar(),
        session_binding=_completion_binding(),
        expected_freeze_commit="a" * 40,
        expected_activity=expected,
        expected_cumulative=expected,
        cumulative_prior=prior,
    )
    assert records[0][0] == "header"
    assert records[-1][0] == "completion_receipt"


def test_validate_completion_log_rejects_self_hash_and_preflight_claim() -> None:
    """The log must not hash its own path or the preflight artifact."""
    text = _valid_completion_log().replace(
        "outputs/analysis/two.csv",
        "outputs/analysis/posthoc_final_score_blind_preflight.json",
    )
    binding = _completion_binding()
    binding["ordered_output_inventory"] = [
        "outputs/analysis/one.csv",
        "outputs/analysis/posthoc_final_score_blind_preflight.json",
    ]
    with pytest.raises(ValueError):
        validate_completion_log(text, grammar=_completion_grammar(), session_binding=binding)


def test_validate_completion_log_rejects_order_and_summary_violations() -> None:
    """Record reordering and inconsistent deviation summaries are rejected."""
    reordered = _valid_completion_log().split("\n")
    reordered[1], reordered[2] = reordered[2], reordered[1]
    with pytest.raises(ValueError):
        validate_completion_log(
            "\n".join(reordered),
            grammar=_completion_grammar(),
            session_binding=_completion_binding(),
        )
    bad_summary = _valid_completion_log().replace(
        "deviations_or_nonestimability_summary\tnone\t\t",
        "deviations_or_nonestimability_summary\tnone\tsome_reason\t",
    )
    with pytest.raises(ValueError):
        validate_completion_log(
            bad_summary,
            grammar=_completion_grammar(),
            session_binding=_completion_binding(),
        )


# --- Nested preflight-schema mutations (correction C) ---------------------------------------------


def _drop_status(payload: dict[str, object]) -> dict[str, object]:
    """Remove a required top-level key so the frozen ordering breaks."""
    del payload["status"]
    return payload


def _add_unexpected_key(payload: dict[str, object]) -> dict[str, object]:
    """Add an unexpected top-level key beyond the frozen contract."""
    payload["unexpected"] = 1
    return payload


def _reorder_top_level(payload: dict[str, object]) -> dict[str, object]:
    """Swap the first two top-level keys to violate the frozen order."""
    items = list(payload.items())
    items[0], items[1] = items[1], items[0]
    return dict(items)


def _bad_schema_version(payload: dict[str, object]) -> dict[str, object]:
    """Set a wrong fixed schema version string."""
    payload["schema_version"] = "posthoc_final_score_blind_preflight.v2"
    return payload


def _bad_package_id(payload: dict[str, object]) -> dict[str, object]:
    """Set a wrong fixed package identifier."""
    payload["package_id"] = "some_other_package"
    return payload


def _timestamp_missing_z(payload: dict[str, object]) -> dict[str, object]:
    """Drop the trailing UTC ``Z`` from the provenance timestamp."""
    payload["generated_at_utc"] = "2026-07-17T00:00:00"
    return payload


def _timestamp_slash_date(payload: dict[str, object]) -> dict[str, object]:
    """Smuggle a slash clinical date into the provenance timestamp field."""
    payload["generated_at_utc"] = "2026/7/17T00:00:00Z"
    return payload


def _reorder_hashes(payload: dict[str, object]) -> dict[str, object]:
    """Swap the first two hash keys to violate the frozen hash order."""
    hashes = payload["hashes"]  # type: ignore[assignment]
    items = list(hashes.items())  # type: ignore[union-attr]
    items[0], items[1] = items[1], items[0]
    payload["hashes"] = OrderedDict(items)
    return payload


def _non_hex_hash_value(payload: dict[str, object]) -> dict[str, object]:
    """Replace a hash leaf with a non-hexadecimal 64-character string."""
    payload["hashes"]["config_sha256"] = "z" * 64  # type: ignore[index]
    return payload


def _bool_for_int_cohort(payload: dict[str, object]) -> dict[str, object]:
    """Use a boolean where an exact integer cohort count is required."""
    payload["aggregate_counts"]["source_cohort_n"] = True  # type: ignore[index]
    return payload


def _wrong_cohort_anchor(payload: dict[str, object]) -> dict[str, object]:
    """Set a cohort anchor to a value other than the frozen count."""
    payload["aggregate_counts"]["source_cohort_n"] = 999  # type: ignore[index]
    return payload


def _difficult_totals_mismatch(payload: dict[str, object]) -> dict[str, object]:
    """Break the difficult-spectrum total reconciliation for a view."""
    payload["aggregate_counts"]["difficult_spectrum_by_view"]["oof"]["n_positive"] = 5  # type: ignore[index]
    return payload


def _reorder_disease_counts(payload: dict[str, object]) -> dict[str, object]:
    """Reorder the nested disease-category count keys for a view."""
    view = payload["aggregate_counts"]["difficult_spectrum_by_view"]["oof"]  # type: ignore[index]
    items = list(view["disease_category_counts"].items())
    items[0], items[1] = items[1], items[0]
    view["disease_category_counts"] = OrderedDict(items)
    return payload


def _bad_date_anchor(payload: dict[str, object]) -> dict[str, object]:
    """Change the frozen date-feasibility anchor away from 40/34."""
    payload["aggregate_counts"]["date_feasibility"]["n_dates_overall"] = 41  # type: ignore[index]
    return payload


def _bad_qc5_anchor(payload: dict[str, object]) -> dict[str, object]:
    """Change the frozen QC5 reconciliation anchor away from 974/298/676."""
    payload["aggregate_counts"]["qc5_reconciliation"]["n_strict_fail_among_evaluable"] = 1  # type: ignore[index]
    return payload


def _freeze_flag_non_bool(payload: dict[str, object]) -> dict[str, object]:
    """Use a non-boolean value for a freeze-validation flag."""
    payload["freeze_commit_validation"]["exists"] = "yes"  # type: ignore[index]
    return payload


def _reorder_freeze_flags(payload: dict[str, object]) -> dict[str, object]:
    """Reorder two freeze-validation keys to violate the frozen order."""
    freeze = payload["freeze_commit_validation"]  # type: ignore[assignment]
    items = list(freeze.items())  # type: ignore[union-attr]
    items[1], items[2] = items[2], items[1]
    payload["freeze_commit_validation"] = dict(items)
    return payload


def _bad_status_enum(payload: dict[str, object]) -> dict[str, object]:
    """Use a status value outside the allowed pass/fail set."""
    payload["status"] = "maybe"
    return payload


def _pass_with_reason(payload: dict[str, object]) -> dict[str, object]:
    """A passing preflight must not carry a non-null failure reason."""
    payload["reason_code"] = "anchor_mismatch"
    return payload


def _fail_without_reason(payload: dict[str, object]) -> dict[str, object]:
    """A failed preflight must carry an allowed non-null reason code."""
    payload["status"] = "fail"
    payload["reason_code"] = None
    return payload


def _fail_with_bogus_reason(payload: dict[str, object]) -> dict[str, object]:
    """A failed preflight cannot use a reason outside the allowed registry."""
    payload["status"] = "fail"
    payload["reason_code"] = "bogus_reason"
    return payload


def _reorder_access_state(payload: dict[str, object]) -> dict[str, object]:
    """Reorder the access-state activity keys away from the frozen ledger."""
    activity = payload["access_state"]["session_activity"]  # type: ignore[index]
    items = list(activity.items())
    items[0], items[1] = items[1], items[0]
    payload["access_state"]["session_activity"] = OrderedDict(items)  # type: ignore[index]
    return payload


def _non_bool_access_state(payload: dict[str, object]) -> dict[str, object]:
    """Use a non-boolean value inside the access-state ledger."""
    payload["access_state"]["session_activity"]["prediction_container_bytes_hashed"] = "yes"  # type: ignore[index]
    return payload


def _success_with_score_access(payload: dict[str, object]) -> dict[str, object]:
    """A passing score-blind preflight cannot record any score access."""
    access = payload["access_state"]  # type: ignore[assignment]
    access["session_activity"]["participant_level_score_values_read"] = True
    access["cumulative_through_session"]["participant_level_score_values_read"] = True
    return payload


def _cumulative_less_than_activity(payload: dict[str, object]) -> dict[str, object]:
    """Cumulative access can never lose an access recorded in the session."""
    payload["access_state"]["cumulative_through_session"]["participant_level_label_values_read"] = False  # type: ignore[index]
    return payload


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(_drop_status, id="missing_top_level_key"),
        pytest.param(_add_unexpected_key, id="extra_top_level_key"),
        pytest.param(_reorder_top_level, id="reordered_top_level_keys"),
        pytest.param(_bad_schema_version, id="wrong_schema_version"),
        pytest.param(_bad_package_id, id="wrong_package_id"),
        pytest.param(_timestamp_missing_z, id="timestamp_missing_z"),
        pytest.param(_timestamp_slash_date, id="timestamp_slash_date"),
        pytest.param(_reorder_hashes, id="reordered_hash_keys"),
        pytest.param(_non_hex_hash_value, id="non_hex_hash_value"),
        pytest.param(_bool_for_int_cohort, id="bool_where_int_required"),
        pytest.param(_wrong_cohort_anchor, id="wrong_cohort_anchor_value"),
        pytest.param(_difficult_totals_mismatch, id="difficult_totals_mismatch"),
        pytest.param(_reorder_disease_counts, id="reordered_disease_counts"),
        pytest.param(_bad_date_anchor, id="wrong_date_anchor_value"),
        pytest.param(_bad_qc5_anchor, id="wrong_qc5_anchor_value"),
        pytest.param(_freeze_flag_non_bool, id="freeze_flag_non_bool"),
        pytest.param(_reorder_freeze_flags, id="reordered_freeze_flags"),
        pytest.param(_bad_status_enum, id="invalid_status_enum"),
        pytest.param(_pass_with_reason, id="pass_with_non_null_reason"),
        pytest.param(_fail_without_reason, id="fail_without_reason"),
        pytest.param(_fail_with_bogus_reason, id="fail_with_bogus_reason"),
        pytest.param(_reorder_access_state, id="reordered_access_state"),
        pytest.param(_non_bool_access_state, id="non_bool_access_state"),
        pytest.param(_success_with_score_access, id="success_records_score_access"),
        pytest.param(_cumulative_less_than_activity, id="cumulative_less_than_activity"),
    ],
)
def test_validate_preflight_schema_rejects_nested_mutations(mutation) -> None:
    """Every important nested schema level is rejected when mutated off-contract."""
    payload = mutation(_valid_preflight_payload())
    with pytest.raises(ValueError):
        validate_preflight_schema(payload, frozen_hash_paths=["fixture"])


def test_validate_preflight_schema_enforces_expected_cumulative_prior() -> None:
    """Success semantics require cumulative access to retain every historical prior."""
    payload = _valid_preflight_payload()
    access = payload["access_state"]  # type: ignore[assignment]
    prior = OrderedDict((key, False) for key in access["cumulative_through_session"])
    prior["new_score_derived_results_computed"] = True
    with pytest.raises(ValueError):
        validate_preflight_schema(payload, frozen_hash_paths=["fixture"], expected_cumulative_prior=prior)
