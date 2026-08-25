"""Aggregate-only LGIN postdiagnosis disposition source closure.

Participant-capable loading is confined to :func:`execute_approved` after
authority, collision, and whole-file hash gates. Participant join keys are
discarded before construction functions return and are never persisted.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import sys
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.analysis.data_split import _assert_stage1_alignment, _load_final_sample_index
from src.analysis.stage7_part_b_common import (
    _compute_binary_risk_difference,
    _compute_continuous_smd,
    _parse_truncated_numeric,
)
from src.analysis import stage7_part_b_common
from src.config import PROJECT_ROOT
from src.data.loader import load_metadata


PACKAGE_ID = "lgin_postdiagnosis_disposition_source_closure"
PRIMARY_COLUMNS = (
    "analysis_view",
    "analysis_role",
    "n_lgin",
    "n_inpatient_treatment_proxy",
    "n_outpatient_followup_proxy",
    "comparison_estimability_status",
    "nonestimable_reason",
    "rank_biserial_correlation",
    "mann_whitney_u",
    "p_two_sided",
    "median_p_cal_inpatient",
    "q1_p_cal_inpatient",
    "q3_p_cal_inpatient",
    "median_p_cal_outpatient",
    "q1_p_cal_outpatient",
    "q3_p_cal_outpatient",
    "n_p_cal_gt_0_5_inpatient",
    "n_p_cal_gt_0_5_outpatient",
    "directional_verification_passed",
    "separation_criterion_passed",
)
BASELINE_COLUMNS = (
    "variable_order",
    "variable_key",
    "variable_type",
    "event_level",
    "inpatient_n_total",
    "inpatient_n_nonmissing",
    "inpatient_n_missing",
    "inpatient_event_count",
    "inpatient_denominator_n",
    "inpatient_proportion",
    "inpatient_mean",
    "inpatient_sd",
    "inpatient_median",
    "inpatient_q1",
    "inpatient_q3",
    "outpatient_n_total",
    "outpatient_n_nonmissing",
    "outpatient_n_missing",
    "outpatient_event_count",
    "outpatient_denominator_n",
    "outpatient_proportion",
    "outpatient_mean",
    "outpatient_sd",
    "outpatient_median",
    "outpatient_q1",
    "outpatient_q3",
    "denominator_rule",
    "difference_metric",
    "absolute_difference",
    "estimability_status",
    "nonestimable_reason",
)
STABILITY_COLUMNS = (
    "variable_order",
    "variable_key",
    "level_order",
    "level_key",
    "n_in_level",
    "median_p_cal",
    "n_p_cal_gt_0_5",
    "estimability_status",
    "nonestimable_reason",
)
PRIMARY_SOURCE_COLUMNS = (
    "rb",
    "U",
    "p_two_sided",
    "median_p_cal_B",
    "median_p_cal_C",
    "q1_p_cal_B",
    "q3_p_cal_B",
    "q1_p_cal_C",
    "q3_p_cal_C",
    "n_above_0.5_B",
    "n_above_0.5_C",
    "n_B",
    "n_C",
    "directional_verification_passed",
    "separation_criterion_passed",
)
LEGACY_BASELINE_COLUMNS = (
    "variable",
    "mean_B",
    "mean_C",
    "smd",
    "n_B_non_missing",
    "n_C_non_missing",
)
LEGACY_STABILITY_COLUMNS = (
    "variable_name",
    "level_value",
    "n_in_level",
    "median_p_cal",
    "n_above_0_5",
    "omission_reason",
)
OUTPUT_ORDER = (
    "primary_estimability",
    "baseline_context",
    "within_inpatient_stability",
    "manifest",
)
EXPECTED_OUTPUT_PATHS = (
    "outputs/analysis/lgin_postdiagnosis_disposition_primary_estimability.csv",
    "outputs/analysis/lgin_postdiagnosis_disposition_baseline_context.csv",
    "outputs/analysis/lgin_postdiagnosis_disposition_within_inpatient_stability.csv",
    "outputs/analysis/lgin_postdiagnosis_disposition_source_closure_manifest.json",
)
EXPECTED_IMPLEMENTATION_PATHS = (
    "src/analysis/lgin_postdiagnosis_disposition_source_closure.py",
    "scripts/run_lgin_postdiagnosis_disposition_source_closure.py",
    "tests/test_lgin_postdiagnosis_disposition_source_closure.py",
)
EXPECTED_BASELINE_VARIABLES = (
    "age",
    "sex",
    "bmi",
    "smoking",
    "drinking",
    "family_history",
    "hp_infection",
    "cea",
    "ca199",
    "ca724",
)
EXPECTED_STABILITY_IDENTITIES = (
    ("age", "lt_60"),
    ("age", "ge_60"),
    ("sex", "male"),
    ("sex", "female"),
    ("bmi", "lt_24"),
    ("bmi", "ge_24"),
    ("smoking", "no"),
    ("smoking", "yes"),
    ("drinking", "no"),
    ("drinking", "yes"),
    ("family_history", "no"),
    ("family_history", "yes"),
    ("hp_infection", "negative"),
    ("hp_infection", "positive"),
    ("hp_infection", "missing"),
)
EXPECTED_METADATA_COLUMNS = (
    "sample_id",
    "disease_category",
    "label",
    "collection_location",
    "age",
    "sex",
    "bmi",
    "smoking",
    "drinking",
    "first_degree_family_history_of_gc",
    "hp_infection",
    "cea",
    "ca199",
    "ca724",
)
EXPECTED_SAMPLE_INDEX_COLUMNS = ("sample_id", "label", "disease_category")
EXPECTED_OOF_COLUMNS = ("sample_id", "y_true")
FORBIDDEN_INPUTS = {
    "outputs/analysis/lgin_descriptive_test_minirocket_ridge.csv",
    "outputs/analysis/lgin_layer_analysis.md",
}


def _validate_relative_path(value: object) -> Path:
    """Return one safe project-relative configured path."""
    if not isinstance(value, str):
        raise ValueError("Configured artifact path is invalid.")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Configured artifact paths must be project-relative.")
    return path


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one whole file.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate the fixed source-closure configuration.

    Args:
        path: Configuration YAML path.

    Returns:
        Parsed configuration mapping.

    Raises:
        ValueError: If any fixed package contract differs.
    """
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Configuration must deserialize to a mapping.")
    config = dict(payload)
    if config.get("schema_version") != f"{PACKAGE_ID}.config.v1" or config.get("package_id") != PACKAGE_ID:
        raise ValueError("Configuration package identity is invalid.")
    if config.get("package_status") != "source_closure_only_not_publication_table":
        raise ValueError("Configuration package status is invalid.")
    if config.get("intended_downstream_consumer") != "supplementary_etable_7":
        raise ValueError("Configuration downstream consumer is invalid.")
    schemas = config.get("schemas", {})
    expected_schemas = {
        "primary_estimability": PRIMARY_COLUMNS,
        "baseline_context": BASELINE_COLUMNS,
        "within_inpatient_stability": STABILITY_COLUMNS,
    }
    if any(tuple(schemas.get(name, ())) != columns for name, columns in expected_schemas.items()):
        raise ValueError("Configuration output schema contract is invalid.")
    outputs = config.get("outputs", {})
    output_paths = tuple(record.get("path") for record in outputs.values())
    if tuple(outputs) != OUTPUT_ORDER or output_paths != EXPECTED_OUTPUT_PATHS:
        raise ValueError("Configuration output inventory is invalid.")
    implementation = config.get("implementation", {})
    if tuple(implementation.values()) != EXPECTED_IMPLEMENTATION_PATHS:
        raise ValueError("Configuration implementation inventory is invalid.")
    if config.get("configuration", {}).get("path") != "configs/lgin_postdiagnosis_disposition_source_closure.yaml":
        raise ValueError("Configuration self-path is invalid.")
    baseline_variables = config.get("baseline_variables", [])
    if tuple(record.get("variable_key") for record in baseline_variables) != EXPECTED_BASELINE_VARIABLES:
        raise ValueError("Configuration baseline variable order is invalid.")
    if tuple(int(record.get("variable_order", -1)) for record in baseline_variables) != tuple(range(1, 11)):
        raise ValueError("Configuration baseline variable numbering is invalid.")
    stability_mapping = config.get("stability_mapping", [])
    stability_identities = tuple((row[3], row[4]) for row in stability_mapping)
    if len(stability_mapping) != 15 or stability_identities != EXPECTED_STABILITY_IDENTITIES:
        raise ValueError("Configuration stability mapping is invalid.")
    inventory = config.get("scientific_inputs", [])
    if len(inventory) != 18:
        raise ValueError("Configuration scientific input inventory is invalid.")
    paths: list[str] = []
    for record in inventory:
        path_value = record.get("path")
        _validate_relative_path(path_value)
        digest = record.get("sha256")
        if re.fullmatch(r"[0-9a-f]{64}", str(digest)) is None:
            raise ValueError("Configuration scientific input hash is invalid.")
        if not isinstance(record.get("participant_capable"), bool):
            raise ValueError("Configuration participant-capable flag is invalid.")
        paths.append(str(path_value))
    if len(set(paths)) != len(paths) or set(paths) & FORBIDDEN_INPUTS:
        raise ValueError("Configuration scientific input inventory is unsafe.")
    if set(config.get("forbidden_inputs", ())) != FORBIDDEN_INPUTS:
        raise ValueError("Configuration forbidden input contract is invalid.")
    projection = config.get("participant_projection", {})
    if tuple(projection.get("metadata_alias_columns", ())) != EXPECTED_METADATA_COLUMNS:
        raise ValueError("Configuration metadata projection is invalid.")
    if tuple(projection.get("sample_index_columns_after_alignment", ())) != EXPECTED_SAMPLE_INDEX_COLUMNS:
        raise ValueError("Configuration sample-index projection is invalid.")
    if tuple(projection.get("oof_prediction_columns", ())) != EXPECTED_OOF_COLUMNS:
        raise ValueError("Configuration OOF projection is invalid.")
    if config.get("proxy_mapping", {}).get("raw_values") != {
        1: "inpatient_treatment_proxy",
        0: "outpatient_followup_proxy",
    }:
        raise ValueError("Configuration proxy mapping is invalid.")
    statistics = config.get("statistics", {})
    if statistics.get("continuous_sd_ddof") != 1 or statistics.get("quantile_method") != "linear":
        raise ValueError("Configuration descriptive-statistic contract is invalid.")
    if float(statistics.get("reconciliation_absolute_tolerance", -1.0)) != 1e-15:
        raise ValueError("Configuration reconciliation tolerance is invalid.")
    if float(statistics.get("reconciliation_relative_tolerance", -1.0)) != 0.0:
        raise ValueError("Configuration relative tolerance is invalid.")
    return config


def verify_scientific_inputs(root: Path, records: Sequence[Mapping[str, Any]]) -> OrderedDict[str, str]:
    """Verify every configured scientific input without parsing it.

    Args:
        root: Project root.
        records: Ordered scientific input records.

    Returns:
        Ordered role-to-observed-hash mapping.

    Raises:
        ValueError: If a path, hash, role, or file differs.
    """
    observed: OrderedDict[str, str] = OrderedDict()
    for record in records:
        role = record.get("role")
        if not isinstance(role, str) or role in observed:
            raise ValueError("Scientific input role inventory is invalid.")
        relative = _validate_relative_path(record.get("path"))
        expected = record.get("sha256")
        if re.fullmatch(r"[0-9a-f]{64}", str(expected)) is None:
            raise ValueError("Scientific input SHA-256 contract is invalid.")
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError("Approved scientific input is unavailable.")
        digest = sha256_file(path)
        if digest != expected:
            raise ValueError("Approved scientific input hash mismatch.")
        observed[role] = digest
    return observed


def _input_path(root: Path, config: Mapping[str, Any], role: str) -> Path:
    """Resolve one configured scientific input by role."""
    matches = [record for record in config["scientific_inputs"] if record["role"] == role]
    if len(matches) != 1:
        raise ValueError("Scientific input role resolution failed.")
    return root / _validate_relative_path(matches[0]["path"])


def _numeric_close(left: object, right: object, config: Mapping[str, Any]) -> bool:
    """Compare finite scalars with the locked strict floating tolerance."""
    try:
        left_float = float(left)
        right_float = float(right)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(left_float) or not np.isfinite(right_float):
        return False
    statistics = config["statistics"]
    return bool(
        np.isclose(
            left_float,
            right_float,
            atol=float(statistics["reconciliation_absolute_tolerance"]),
            rtol=float(statistics["reconciliation_relative_tolerance"]),
        )
    )


def _integer(value: object, message: str) -> int:
    """Return one exact non-boolean integer or raise a sanitized error."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(numeric) or numeric != np.floor(numeric):
        raise ValueError(message)
    return int(numeric)


def _validate_exact_binary(series: pd.Series, *, allow_missing: bool) -> None:
    """Require numeric 0/1 coding without string coercion."""
    if not allow_missing and series.isna().any():
        raise ValueError("Binary field coding is invalid.")
    values = series.loc[~series.isna()].tolist()
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError("Binary field coding is invalid.")
        numeric = float(value)
        if not np.isfinite(numeric) or numeric not in {0.0, 1.0}:
            raise ValueError("Binary field coding is invalid.")


def _validate_numeric(series: pd.Series, *, allow_missing: bool) -> None:
    """Require finite numeric values wherever a continuous field is present."""
    if not allow_missing and series.isna().any():
        raise ValueError("Continuous field coding is invalid.")
    for value in series.loc[~series.isna()].tolist():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError("Continuous field coding is invalid.")
        if not np.isfinite(float(value)):
            raise ValueError("Continuous field coding is invalid.")


def _apply_primary_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic nullable dtypes to the primary source."""
    result = frame.copy()
    for column in (
        "analysis_view",
        "analysis_role",
        "comparison_estimability_status",
        "nonestimable_reason",
    ):
        result[column] = result[column].astype("string")
    for column in (
        "n_lgin",
        "n_inpatient_treatment_proxy",
        "n_outpatient_followup_proxy",
        "n_p_cal_gt_0_5_inpatient",
        "n_p_cal_gt_0_5_outpatient",
    ):
        result[column] = result[column].astype("Int64")
    for column in (
        "rank_biserial_correlation",
        "mann_whitney_u",
        "p_two_sided",
        "median_p_cal_inpatient",
        "q1_p_cal_inpatient",
        "q3_p_cal_inpatient",
        "median_p_cal_outpatient",
        "q1_p_cal_outpatient",
        "q3_p_cal_outpatient",
    ):
        result[column] = result[column].astype("Float64")
    for column in ("directional_verification_passed", "separation_criterion_passed"):
        result[column] = result[column].astype("boolean")
    return result.loc[:, PRIMARY_COLUMNS]


def build_primary_estimability(
    primary_source: pd.DataFrame,
    audit: Mapping[str, Any],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Map frozen OOF and audit aggregates to two analysis-view rows.

    Args:
        primary_source: Frozen one-row OOF aggregate.
        audit: Authorized LGIN audit projection.
        config: Validated package contract.

    Returns:
        Two-row primary/estimability source.

    Raises:
        ValueError: If schemas, anchors, or aggregate sources disagree.
    """
    if tuple(primary_source.columns) != PRIMARY_SOURCE_COLUMNS or len(primary_source) != 1:
        raise ValueError("Primary aggregate source schema is invalid.")
    references = audit.get("post_hoc_trigger_references")
    if not isinstance(references, Mapping):
        raise ValueError("Audit LGIN aggregate projection is invalid.")
    if audit.get("git_commit_hash") != config["upstream_stage7_part_b_generation_commit"]:
        raise ValueError("Audit generation commit is invalid.")
    required_audit_fields = tuple(config["anchors"]["audit_lgin_fields"])
    if any(field not in references for field in required_audit_fields):
        raise ValueError("Audit LGIN aggregate projection is incomplete.")
    if (
        references["lgin_primary_view"] != "nested_cv_oof"
        or references["lgin_primary_view_authorization"] != "Erratum E012"
        or references["lgin_within_b_stability_levels_evaluated"] != 13
        or references["lgin_within_b_stability_pass_overall"] is not True
    ):
        raise ValueError("Primary aggregate reconciliation failed.")
    row = primary_source.iloc[0]
    oof_anchor = config["anchors"]["oof"]
    exact_pairs = (
        (row["n_B"], oof_anchor["n_inpatient_treatment_proxy"]),
        (row["n_C"], oof_anchor["n_outpatient_followup_proxy"]),
        (row["U"], oof_anchor["mann_whitney_u"]),
        (row["directional_verification_passed"], oof_anchor["directional_verification_passed"]),
        (row["separation_criterion_passed"], oof_anchor["separation_criterion_passed"]),
        (references["lgin_n_b_oof"], oof_anchor["n_inpatient_treatment_proxy"]),
        (references["lgin_n_c_oof"], oof_anchor["n_outpatient_followup_proxy"]),
        (references["lgin_directional_verification_passed"], True),
        (references["lgin_separation_criterion_passed"], True),
    )
    if any(left != right for left, right in exact_pairs):
        raise ValueError("Primary aggregate reconciliation failed.")
    numeric_pairs = (
        (row["rb"], oof_anchor["rank_biserial_correlation"]),
        (references["lgin_rb"], row["rb"]),
        (references["lgin_mannwhitney_p_two_sided"], row["p_two_sided"]),
    )
    if any(not _numeric_close(left, right, config) for left, right in numeric_pairs):
        raise ValueError("Primary aggregate reconciliation failed.")
    n_lgin = int(row["n_B"]) + int(row["n_C"])
    if n_lgin != int(oof_anchor["n_lgin"]):
        raise ValueError("Primary aggregate reconciliation failed.")
    test_anchor = config["anchors"]["locked_internal_test"]
    if (
        references["lgin_n_b_test"] != test_anchor["n_inpatient_treatment_proxy"]
        or references["lgin_n_c_test"] != test_anchor["n_outpatient_followup_proxy"]
        or references["lgin_test_view"] != "descriptive_sensitivity_only_due_to_n_c_test_zero"
    ):
        raise ValueError("Internal-test estimability reconciliation failed.")
    rows = [
        OrderedDict(
            [
                ("analysis_view", "training_nested_cv_oof"),
                ("analysis_role", "exploratory_primary"),
                ("n_lgin", n_lgin),
                ("n_inpatient_treatment_proxy", int(row["n_B"])),
                ("n_outpatient_followup_proxy", int(row["n_C"])),
                ("comparison_estimability_status", "estimable"),
                ("nonestimable_reason", None),
                ("rank_biserial_correlation", float(row["rb"])),
                ("mann_whitney_u", float(row["U"])),
                ("p_two_sided", float(row["p_two_sided"])),
                ("median_p_cal_inpatient", float(row["median_p_cal_B"])),
                ("q1_p_cal_inpatient", float(row["q1_p_cal_B"])),
                ("q3_p_cal_inpatient", float(row["q3_p_cal_B"])),
                ("median_p_cal_outpatient", float(row["median_p_cal_C"])),
                ("q1_p_cal_outpatient", float(row["q1_p_cal_C"])),
                ("q3_p_cal_outpatient", float(row["q3_p_cal_C"])),
                ("n_p_cal_gt_0_5_inpatient", int(row["n_above_0.5_B"])),
                ("n_p_cal_gt_0_5_outpatient", int(row["n_above_0.5_C"])),
                ("directional_verification_passed", bool(row["directional_verification_passed"])),
                ("separation_criterion_passed", bool(row["separation_criterion_passed"])),
            ]
        ),
        OrderedDict(
            [
                ("analysis_view", "locked_internal_test"),
                ("analysis_role", "estimability_check_only"),
                ("n_lgin", int(test_anchor["n_lgin"])),
                ("n_inpatient_treatment_proxy", int(test_anchor["n_inpatient_treatment_proxy"])),
                ("n_outpatient_followup_proxy", int(test_anchor["n_outpatient_followup_proxy"])),
                ("comparison_estimability_status", "not_estimable"),
                ("nonestimable_reason", "outpatient_followup_proxy_n_zero"),
                *[(column, None) for column in PRIMARY_COLUMNS[7:]],
            ]
        ),
    ]
    result = _apply_primary_dtypes(pd.DataFrame(rows, columns=PRIMARY_COLUMNS))
    validate_primary(result, config)
    return result


def validate_primary(frame: pd.DataFrame, config: Mapping[str, Any]) -> None:
    """Validate the exact two-row primary/estimability contract."""
    if frame.columns.tolist() != list(PRIMARY_COLUMNS) or len(frame) != 2:
        raise ValueError("Primary output schema or row count is invalid.")
    if frame["analysis_view"].astype(str).tolist() != ["training_nested_cv_oof", "locked_internal_test"]:
        raise ValueError("Primary output row order is invalid.")
    oof, test = frame.iloc[0], frame.iloc[1]
    anchor = config["anchors"]["oof"]
    if (
        int(oof["n_lgin"]) != int(anchor["n_lgin"])
        or int(oof["n_inpatient_treatment_proxy"]) != int(anchor["n_inpatient_treatment_proxy"])
        or int(oof["n_outpatient_followup_proxy"]) != int(anchor["n_outpatient_followup_proxy"])
        or str(oof["comparison_estimability_status"]) != "estimable"
    ):
        raise ValueError("Primary OOF count or estimability contract is invalid.")
    test_anchor = config["anchors"]["locked_internal_test"]
    if (
        int(test["n_lgin"]) != int(test_anchor["n_lgin"])
        or int(test["n_inpatient_treatment_proxy"]) != int(test_anchor["n_inpatient_treatment_proxy"])
        or int(test["n_outpatient_followup_proxy"]) != 0
        or str(test["comparison_estimability_status"]) != "not_estimable"
        or str(test["nonestimable_reason"]) != "outpatient_followup_proxy_n_zero"
        or not test.loc[list(PRIMARY_COLUMNS[7:])].isna().all()
    ):
        raise ValueError("Internal-test structural unavailability contract is invalid.")


def _apply_baseline_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic nullable dtypes to baseline context."""
    result = frame.copy()
    for column in (
        "variable_key",
        "variable_type",
        "event_level",
        "denominator_rule",
        "difference_metric",
        "estimability_status",
        "nonestimable_reason",
    ):
        result[column] = result[column].astype("string")
    integer_columns = (
        "variable_order",
        "inpatient_n_total",
        "inpatient_n_nonmissing",
        "inpatient_n_missing",
        "inpatient_event_count",
        "inpatient_denominator_n",
        "outpatient_n_total",
        "outpatient_n_nonmissing",
        "outpatient_n_missing",
        "outpatient_event_count",
        "outpatient_denominator_n",
    )
    for column in integer_columns:
        result[column] = result[column].astype("Int64")
    float_columns = (
        "inpatient_proportion",
        "inpatient_mean",
        "inpatient_sd",
        "inpatient_median",
        "inpatient_q1",
        "inpatient_q3",
        "outpatient_proportion",
        "outpatient_mean",
        "outpatient_sd",
        "outpatient_median",
        "outpatient_q1",
        "outpatient_q3",
        "absolute_difference",
    )
    for column in float_columns:
        result[column] = result[column].astype("Float64")
    return result.loc[:, BASELINE_COLUMNS]


def _continuous_summary(series: pd.Series) -> dict[str, float]:
    """Compute approved additive continuous summaries."""
    values = series.dropna().astype(float)
    return {
        "sd": float(values.std(ddof=1)),
        "median": float(values.median()),
        "q1": float(values.quantile(0.25, interpolation="linear")),
        "q3": float(values.quantile(0.75, interpolation="linear")),
    }


def build_baseline_context(
    lgin_frame: pd.DataFrame,
    legacy_baseline: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Build additive group summaries and reconcile frozen differences.

    Args:
        lgin_frame: Identifier-free OOF LGIN baseline fields.
        legacy_baseline: Frozen mixed-metric aggregate source.
        config: Package contract or a bounded synthetic analogue.

    Returns:
        Ordered aggregate baseline-context rows.

    Raises:
        ValueError: If coding, counts, definitions, or aggregates disagree.
    """
    if tuple(legacy_baseline.columns) != LEGACY_BASELINE_COLUMNS:
        raise ValueError("Legacy baseline aggregate schema is invalid.")
    variables = list(config["baseline_variables"])
    expected_variables = [record["variable_key"] for record in variables]
    if legacy_baseline["variable"].tolist() != expected_variables:
        raise ValueError("Legacy baseline aggregate variable order is invalid.")
    if "collection_location" not in lgin_frame.columns:
        raise ValueError("Proxy-group field is unavailable.")
    _validate_exact_binary(lgin_frame["collection_location"], allow_missing=False)
    inpatient = lgin_frame.loc[lgin_frame["collection_location"] == 1].reset_index(drop=True)
    outpatient = lgin_frame.loc[lgin_frame["collection_location"] == 0].reset_index(drop=True)
    oof_anchor = config["anchors"]["oof"]
    if (
        len(lgin_frame) != int(oof_anchor["n_lgin"])
        or len(inpatient) != int(oof_anchor["n_inpatient_treatment_proxy"])
        or len(outpatient) != int(oof_anchor["n_outpatient_followup_proxy"])
    ):
        raise ValueError("OOF LGIN proxy-group counts do not match the approved contract.")
    rows: list[OrderedDict[str, Any]] = []
    for variable, legacy_row in zip(variables, legacy_baseline.to_dict("records"), strict=True):
        source_column = variable["source_column"]
        if source_column not in lgin_frame.columns:
            raise ValueError("Approved baseline field is unavailable.")
        left = inpatient[source_column]
        right = outpatient[source_column]
        variable_type = variable["variable_type"]
        if variable_type == "binary":
            _validate_exact_binary(left, allow_missing=True)
            _validate_exact_binary(right, allow_missing=True)
        elif variable_type == "continuous":
            _validate_numeric(left, allow_missing=True)
            _validate_numeric(right, allow_missing=True)
        else:
            raise ValueError("Approved baseline variable type is invalid.")
        left_nonmissing = left.dropna().astype(float)
        right_nonmissing = right.dropna().astype(float)
        expected_left_n = int(variable["inpatient_n_nonmissing"])
        expected_right_n = int(variable["outpatient_n_nonmissing"])
        if (
            len(left_nonmissing) != expected_left_n
            or len(right_nonmissing) != expected_right_n
            or int(legacy_row["n_B_non_missing"]) != expected_left_n
            or int(legacy_row["n_C_non_missing"]) != expected_right_n
        ):
            raise ValueError("Baseline nonmissing-count aggregate reconciliation failed.")
        left_mean = float(left_nonmissing.mean())
        right_mean = float(right_nonmissing.mean())
        if not _numeric_close(left_mean, legacy_row["mean_B"], config) or not _numeric_close(
            right_mean, legacy_row["mean_C"], config
        ):
            raise ValueError("Baseline group-mean aggregate reconciliation failed.")
        difference = (
            _compute_binary_risk_difference(left, right)
            if variable_type == "binary"
            else _compute_continuous_smd(left, right)
        )
        if not _numeric_close(difference, legacy_row["smd"], config):
            raise ValueError("Baseline difference aggregate reconciliation failed.")
        record = OrderedDict((column, None) for column in BASELINE_COLUMNS)
        record.update(
            variable_order=int(variable["variable_order"]),
            variable_key=variable["variable_key"],
            variable_type=variable_type,
            event_level=variable.get("event_level"),
            inpatient_n_total=len(inpatient),
            inpatient_n_nonmissing=expected_left_n,
            inpatient_n_missing=len(inpatient) - expected_left_n,
            outpatient_n_total=len(outpatient),
            outpatient_n_nonmissing=expected_right_n,
            outpatient_n_missing=len(outpatient) - expected_right_n,
            denominator_rule="variable_specific_nonmissing",
            difference_metric=variable["difference_metric"],
            absolute_difference=float(legacy_row["smd"]),
            estimability_status="estimable",
        )
        if variable_type == "binary":
            event_code = int(variable["event_code"])
            left_events = int((left_nonmissing == event_code).sum())
            right_events = int((right_nonmissing == event_code).sum())
            if left_events != int(variable["inpatient_event_count"]) or right_events != int(
                variable["outpatient_event_count"]
            ):
                raise ValueError("Baseline binary event count does not match the approved contract.")
            record.update(
                inpatient_event_count=left_events,
                inpatient_denominator_n=expected_left_n,
                inpatient_proportion=float(legacy_row["mean_B"]),
                outpatient_event_count=right_events,
                outpatient_denominator_n=expected_right_n,
                outpatient_proportion=float(legacy_row["mean_C"]),
            )
        else:
            left_summary = _continuous_summary(left)
            right_summary = _continuous_summary(right)
            record.update(
                inpatient_mean=float(legacy_row["mean_B"]),
                inpatient_sd=left_summary["sd"],
                inpatient_median=left_summary["median"],
                inpatient_q1=left_summary["q1"],
                inpatient_q3=left_summary["q3"],
                outpatient_mean=float(legacy_row["mean_C"]),
                outpatient_sd=right_summary["sd"],
                outpatient_median=right_summary["median"],
                outpatient_q1=right_summary["q1"],
                outpatient_q3=right_summary["q3"],
            )
        rows.append(record)
    result = _apply_baseline_dtypes(pd.DataFrame(rows, columns=BASELINE_COLUMNS))
    validate_baseline(result, config)
    return result


def validate_baseline(frame: pd.DataFrame, config: Mapping[str, Any]) -> None:
    """Validate the baseline-context schema, ordering, fields, and missingness."""
    variables = list(config["baseline_variables"])
    if frame.columns.tolist() != list(BASELINE_COLUMNS) or len(frame) != len(variables):
        raise ValueError("Baseline output schema or row count is invalid.")
    if frame["variable_key"].astype(str).tolist() != [record["variable_key"] for record in variables]:
        raise ValueError("Baseline output variable order is invalid.")
    for row, variable in zip(frame.to_dict("records"), variables, strict=True):
        for prefix in ("inpatient", "outpatient"):
            if int(row[f"{prefix}_n_nonmissing"]) + int(row[f"{prefix}_n_missing"]) != int(
                row[f"{prefix}_n_total"]
            ):
                raise ValueError("Baseline missingness reconciliation is invalid.")
        if row["denominator_rule"] != "variable_specific_nonmissing" or row["estimability_status"] != "estimable":
            raise ValueError("Baseline denominator or estimability contract is invalid.")
        if variable["variable_type"] == "continuous":
            event_fields = (
                "inpatient_event_count",
                "inpatient_denominator_n",
                "inpatient_proportion",
                "outpatient_event_count",
                "outpatient_denominator_n",
                "outpatient_proportion",
            )
            if not all(pd.isna(row[field]) for field in event_fields):
                raise ValueError("Continuous baseline event fields must be structurally empty.")
            if row["difference_metric"] != "absolute_standardized_mean_difference":
                raise ValueError("Continuous baseline difference label is invalid.")
        else:
            continuous_fields = (
                "inpatient_mean",
                "inpatient_sd",
                "inpatient_median",
                "inpatient_q1",
                "inpatient_q3",
                "outpatient_mean",
                "outpatient_sd",
                "outpatient_median",
                "outpatient_q1",
                "outpatient_q3",
            )
            if not all(pd.isna(row[field]) for field in continuous_fields):
                raise ValueError("Binary baseline continuous fields must be structurally empty.")
            if row["difference_metric"] != "absolute_proportion_difference":
                raise ValueError("Binary baseline difference label is invalid.")


def _apply_stability_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic nullable dtypes to stability rows."""
    result = frame.copy()
    for column in ("variable_key", "level_key", "estimability_status", "nonestimable_reason"):
        result[column] = result[column].astype("string")
    for column in ("variable_order", "level_order", "n_in_level", "n_p_cal_gt_0_5"):
        result[column] = result[column].astype("Int64")
    result["median_p_cal"] = result["median_p_cal"].astype("Float64")
    return result.loc[:, STABILITY_COLUMNS]


def build_stability_source(legacy_stability: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Map the frozen within-inpatient stability aggregate with an n>=5 gate.

    Args:
        legacy_stability: Frozen 15-row stability aggregate.
        config: Validated package contract.

    Returns:
        Truthfully mapped and gated 15-row source.

    Raises:
        ValueError: If mapping, order, gates, or aggregate fields differ.
    """
    if tuple(legacy_stability.columns) != LEGACY_STABILITY_COLUMNS or len(legacy_stability) != 15:
        raise ValueError("Legacy stability aggregate schema or row count is invalid.")
    if legacy_stability[["variable_name", "level_value"]].duplicated().any():
        raise ValueError("Legacy stability aggregate identities are not unique.")
    minimum_n = int(config["statistics"]["stability_minimum_n"])
    rows: list[OrderedDict[str, Any]] = []
    level_counters: dict[str, int] = {}
    for variable_order, source_variable, source_level, variable_key, level_key in config["stability_mapping"]:
        match = legacy_stability.loc[
            (legacy_stability["variable_name"].astype(str) == str(source_variable))
            & (legacy_stability["level_value"].astype(str) == str(source_level))
        ]
        if len(match) != 1:
            raise ValueError("Legacy stability mapping reconciliation failed.")
        source = match.iloc[0]
        n_in_level = _integer(source["n_in_level"], "Legacy stability level count is invalid.")
        level_counters[variable_key] = level_counters.get(variable_key, 0) + 1
        reportable = n_in_level >= minimum_n
        omission = source["omission_reason"]
        omission_empty = pd.isna(omission) or str(omission) == ""
        if reportable:
            if not omission_empty or pd.isna(source["median_p_cal"]) or pd.isna(source["n_above_0_5"]):
                raise ValueError("Legacy stability estimable field population is invalid.")
            median = float(source["median_p_cal"])
            count = _integer(source["n_above_0_5"], "Legacy stability threshold count is invalid.")
            if not np.isfinite(median) or count < 0 or count > n_in_level:
                raise ValueError("Legacy stability estimable aggregate is invalid.")
            status = "estimable"
            reason = None
        else:
            if omission_empty or not pd.isna(source["median_p_cal"]) or not pd.isna(source["n_above_0_5"]):
                raise ValueError("Legacy stability small-level field population is invalid.")
            median = None
            count = None
            status = "not_reported_small_level"
            reason = "n_in_level_below_5"
        rows.append(
            OrderedDict(
                [
                    ("variable_order", int(variable_order)),
                    ("variable_key", variable_key),
                    ("level_order", level_counters[variable_key]),
                    ("level_key", level_key),
                    ("n_in_level", n_in_level),
                    ("median_p_cal", median),
                    ("n_p_cal_gt_0_5", count),
                    ("estimability_status", status),
                    ("nonestimable_reason", reason),
                ]
            )
        )
    result = _apply_stability_dtypes(pd.DataFrame(rows, columns=STABILITY_COLUMNS))
    validate_stability(result, config)
    return result


def validate_stability(frame: pd.DataFrame, config: Mapping[str, Any]) -> None:
    """Validate exact stability identities and small-level omission fields."""
    if frame.columns.tolist() != list(STABILITY_COLUMNS) or len(frame) != 15:
        raise ValueError("Stability output schema or row count is invalid.")
    identities = list(zip(frame["variable_key"].astype(str), frame["level_key"].astype(str), strict=True))
    if tuple(identities) != EXPECTED_STABILITY_IDENTITIES:
        raise ValueError("Stability output identity order is invalid.")
    small = frame.loc[frame["estimability_status"].astype(str) == "not_reported_small_level"]
    if (
        len(small) != 2
        or list(zip(small["variable_key"].astype(str), small["level_key"].astype(str), strict=True))
        != [("family_history", "yes"), ("hp_infection", "positive")]
        or small["n_in_level"].astype(int).tolist() != [2, 3]
        or not small["median_p_cal"].isna().all()
        or not small["n_p_cal_gt_0_5"].isna().all()
        or not (small["nonestimable_reason"].astype(str) == "n_in_level_below_5").all()
    ):
        raise ValueError("Stability small-level gate is invalid.")
    estimable = frame.loc[frame["estimability_status"].astype(str) == "estimable"]
    if len(estimable) != 13 or estimable[["median_p_cal", "n_p_cal_gt_0_5"]].isna().any().any():
        raise ValueError("Stability estimable-level contract is invalid.")


def build_lgin_participant_context(
    metadata: pd.DataFrame,
    sample_index: pd.DataFrame,
    oof_membership: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Validate strict joins and return identifier-free OOF LGIN baseline fields.

    Args:
        metadata: Minimum projected aliased metadata.
        sample_index: Final-cohort key, label, and disease projection.
        oof_membership: OOF key and endpoint projection only.
        config: Package contract or a bounded synthetic analogue.

    Returns:
        Identifier-free LGIN baseline fields.

    Raises:
        ValueError: If schemas, joins, coding, labels, disease, or counts differ.
    """
    if tuple(metadata.columns) != EXPECTED_METADATA_COLUMNS:
        raise ValueError("Projected metadata schema is invalid.")
    if tuple(sample_index.columns) != EXPECTED_SAMPLE_INDEX_COLUMNS:
        raise ValueError("Projected sample-index schema is invalid.")
    if tuple(oof_membership.columns) != EXPECTED_OOF_COLUMNS:
        raise ValueError("Projected OOF membership schema is invalid.")
    for frame in (metadata, sample_index, oof_membership):
        if frame["sample_id"].isna().any() or not frame["sample_id"].is_unique:
            raise ValueError("Join keys must be complete and unique.")
    metadata_keys = set(metadata["sample_id"].tolist())
    index_keys = set(sample_index["sample_id"].tolist())
    oof_keys = set(oof_membership["sample_id"].tolist())
    if not index_keys.issubset(metadata_keys) or not oof_keys.issubset(index_keys):
        raise ValueError("Join-key populations do not match.")
    _validate_exact_binary(sample_index["label"], allow_missing=False)
    _validate_exact_binary(oof_membership["y_true"], allow_missing=False)
    disease = sample_index["disease_category"]
    if disease.isna().any():
        raise ValueError("Disease-category coding is invalid.")
    for value in disease.tolist():
        numeric = _integer(value, "Disease-category coding is invalid.")
        if numeric not in range(6):
            raise ValueError("Disease-category coding is invalid.")
    renamed = metadata.rename(
        columns={
            "label": "metadata_label",
            "disease_category": "metadata_disease_category",
            "first_degree_family_history_of_gc": "family_history",
        }
    )
    cohort = sample_index.merge(renamed, on="sample_id", how="left", validate="one_to_one", sort=False)
    if cohort.isna().all(axis=1).any():
        raise ValueError("Final-cohort metadata alignment failed.")
    _validate_exact_binary(cohort["metadata_label"], allow_missing=False)
    metadata_disease = cohort["metadata_disease_category"]
    if metadata_disease.isna().any():
        raise ValueError("Disease-category coding is invalid.")
    for value in metadata_disease.tolist():
        numeric = _integer(value, "Disease-category coding is invalid.")
        if numeric not in range(6):
            raise ValueError("Disease-category coding is invalid.")
    if not np.array_equal(cohort["label"].to_numpy(), cohort["metadata_label"].to_numpy()) or not np.array_equal(
        cohort["disease_category"].to_numpy(), cohort["metadata_disease_category"].to_numpy()
    ):
        raise ValueError("Endpoint or disease-category alignment failed.")
    merged = oof_membership.merge(cohort, on="sample_id", how="left", validate="one_to_one", sort=False)
    if merged["label"].isna().any() or not np.array_equal(merged["y_true"].to_numpy(), merged["label"].to_numpy()):
        raise ValueError("OOF endpoint alignment failed.")
    lgin = merged.loc[merged["disease_category"] == int(config["anchors"]["lgin_disease_category"])].copy()
    if not (lgin["label"] == 0).all():
        raise ValueError("LGIN endpoint coding is invalid.")
    for column in ("collection_location", "sex", "smoking", "drinking", "family_history"):
        _validate_exact_binary(lgin[column], allow_missing=False)
    _validate_exact_binary(lgin["hp_infection"], allow_missing=True)
    for column in ("age", "bmi"):
        _validate_numeric(lgin[column], allow_missing=False)
    for column in ("cea", "ca199", "ca724"):
        _validate_numeric(lgin[column], allow_missing=True)
    anchor = config["anchors"]["oof"]
    inpatient_n = int((lgin["collection_location"] == 1).sum())
    outpatient_n = int((lgin["collection_location"] == 0).sum())
    if (
        len(lgin) != int(anchor["n_lgin"])
        or inpatient_n != int(anchor["n_inpatient_treatment_proxy"])
        or outpatient_n != int(anchor["n_outpatient_followup_proxy"])
    ):
        raise ValueError("OOF LGIN proxy-group counts do not match the approved contract.")
    output_columns = (
        "collection_location",
        "age",
        "sex",
        "bmi",
        "smoking",
        "drinking",
        "family_history",
        "hp_infection",
        "cea",
        "ca199",
        "ca724",
    )
    return lgin.loc[:, output_columns].reset_index(drop=True)


def load_authorized_participant_context(
    root: Path,
    config: Mapping[str, Any],
    *,
    metadata_loader: Callable[..., pd.DataFrame] = load_metadata,
    sample_index_loader: Callable[..., pd.DataFrame] = _load_final_sample_index,
    alignment_checker: Callable[..., Any] = _assert_stage1_alignment,
    oof_reader: Callable[..., pd.DataFrame] = pd.read_csv,
) -> pd.DataFrame:
    """Verify all hashes, then perform the single participant-capable read.

    Args:
        root: Project root.
        config: Validated package contract.
        metadata_loader: Established metadata loader, injectable for tests.
        sample_index_loader: Established final-index loader, injectable for tests.
        alignment_checker: Established Stage 1 alignment check, injectable for tests.
        oof_reader: Projected OOF CSV reader, injectable for tests.

    Returns:
        Identifier-free OOF LGIN baseline fields.

    Raises:
        ValueError: If any pre-read hash, loader, projection, or alignment gate fails.
    """
    verify_scientific_inputs(root, config["scientific_inputs"])
    modeling_path = _input_path(root, config, "global_modeling_contract")
    try:
        modeling_config = yaml.safe_load(modeling_path.read_text(encoding="utf-8"))
        if not isinstance(modeling_config, dict):
            raise ValueError
    except (OSError, ValueError, yaml.YAMLError):
        raise ValueError("Approved global modeling contract could not be loaded.") from None
    try:
        full_index = sample_index_loader(modeling_config)
        alignment_checker(full_index)
        sample_index = full_index.loc[:, list(EXPECTED_SAMPLE_INDEX_COLUMNS)].copy()
        del full_index
    except (OSError, ValueError, KeyError, AssertionError, TypeError):
        raise ValueError("Approved final-cohort alignment loading failed.") from None
    try:
        full_metadata = metadata_loader()
        metadata = full_metadata.loc[:, list(EXPECTED_METADATA_COLUMNS)].copy()
        del full_metadata
    except (OSError, ValueError, KeyError, TypeError):
        raise ValueError("Approved participant-capable metadata loading failed.") from None
    previous_disabled = stage7_part_b_common.LOGGER.disabled
    stage7_part_b_common.LOGGER.disabled = True
    try:
        for marker in ("cea", "ca199", "ca724"):
            metadata[marker] = metadata[marker].apply(_parse_truncated_numeric)
    finally:
        stage7_part_b_common.LOGGER.disabled = previous_disabled
    oof_path = _input_path(root, config, "locked_training_oof_membership")
    try:
        oof = oof_reader(oof_path, usecols=list(EXPECTED_OOF_COLUMNS))
    except (OSError, ValueError, KeyError, TypeError):
        raise ValueError("Approved OOF membership projection could not be loaded.") from None
    if tuple(oof.columns) != EXPECTED_OOF_COLUMNS:
        raise ValueError("Approved OOF membership projection is invalid.")
    result = build_lgin_participant_context(metadata, sample_index, oof, config)
    del metadata, sample_index, oof
    return result


def _git_output(root: Path, *args: str) -> str:
    """Run one read-only Git command with sanitized failure handling."""
    try:
        completed = subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise ValueError("Read-only Git authority validation failed.") from exc
    return completed.stdout.strip()


def validate_execution_authority(
    root: Path,
    config: Mapping[str, Any],
    *,
    git_output: Callable[..., str] = _git_output,
    executable: Path | None = None,
) -> str:
    """Require the approved HEAD, worktree inventory, and Python environment.

    Args:
        root: Project root.
        config: Validated package contract.
        git_output: Read-only Git adapter, injectable for tests.
        executable: Python executable override for tests.

    Returns:
        Verified full Git HEAD.

    Raises:
        ValueError: If authority, worktree, or environment differs.
    """
    head = git_output(root, "rev-parse", "HEAD")
    if head != config["expected_git_head"]:
        raise ValueError("Current Git HEAD does not match the approved starting authority.")
    if git_output(root, "diff", "--name-only") or git_output(root, "diff", "--cached", "--name-only"):
        raise ValueError("Tracked worktree changes are not authorized.")
    allowed = set(config["allowed_untracked_paths"])
    for line in git_output(root, "status", "--short").splitlines():
        if line and (not line.startswith("?? ") or line[3:] not in allowed):
            raise ValueError("Unexpected worktree state is not authorized.")
    running = Path(sys.executable if executable is None else executable).resolve()
    expected_parent = (Path.home() / "miniforge3/envs/gcvocs/bin").resolve()
    if running.parent != expected_parent:
        raise ValueError("Python is not running from the required gcvocs environment.")
    return head


def serialize_csv(frame: pd.DataFrame, columns: Sequence[str]) -> bytes:
    """Serialize one aggregate frame with deterministic empty missing fields."""
    if frame.columns.tolist() != list(columns):
        raise ValueError("CSV serialization schema is invalid.")
    stream = io.StringIO(newline="")
    frame.to_csv(stream, index=False, columns=list(columns), na_rep="", lineterminator="\n")
    return stream.getvalue().encode("utf-8")


def validate_persisted_privacy(payloads: Mapping[str, bytes]) -> None:
    """Reject direct identifiers, dates, raw-record names, predictions, and host paths."""
    forbidden_tokens = (
        "sample_id",
        "participant_id",
        "collection_time",
        "breath_filename",
        "participant_prediction",
        ".tdms",
    )
    for name, payload in payloads.items():
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Persisted-product privacy validation failed.") from exc
        lowered = text.lower()
        if any(token in lowered for token in forbidden_tokens):
            raise ValueError("Persisted-product privacy validation failed.")
        if re.search(r"(?:^|[\"'])/(?:home|tmp|mnt|var|root)/", text, flags=re.MULTILINE):
            raise ValueError("Persisted-product privacy validation failed.")
        if re.search(r"(?<!\d)(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}(?!\d)", text):
            raise ValueError("Persisted-product privacy validation failed.")
        if "collection_location" in lowered:
            if name != "manifest" or lowered.count("collection_location") != 1:
                raise ValueError("Persisted-product privacy validation failed.")
            try:
                manifest = json.loads(text)
                if manifest["proxy_mapping"]["legacy_source_field"] != "collection_location":
                    raise KeyError
            except (json.JSONDecodeError, KeyError, TypeError):
                raise ValueError("Persisted-product privacy validation failed.") from None


def serialize_manifest(payload: Mapping[str, Any]) -> bytes:
    """Serialize deterministic compact JSON after privacy validation."""
    encoded = (json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode(
        "utf-8"
    )
    validate_persisted_privacy({"manifest": encoded})
    return encoded


def ensure_targets_absent(targets: Mapping[str, Path]) -> None:
    """Refuse final and destination-local temporary collisions."""
    for target in targets.values():
        temporary = target.with_name(f".{target.name}.tmp")
        if target.exists() or target.is_symlink() or temporary.exists() or temporary.is_symlink():
            raise FileExistsError("An output target or temporary output target already exists.")


def promote_serialized_outputs(targets: Mapping[str, Path], payloads: Mapping[str, bytes]) -> None:
    """Stage all four products and atomically expose the manifest last.

    Args:
        targets: Ordered three-CSV and manifest destinations.
        payloads: Corresponding validated bytes.

    Raises:
        FileExistsError: If a final or temporary path already exists.
        ValueError: If inventory or staged bytes differ.
        OSError: If staging or promotion fails. Any newly exposed final is removed.
    """
    if tuple(targets) != OUTPUT_ORDER or tuple(payloads) != OUTPUT_ORDER:
        raise ValueError("Promotion inventory must contain three CSVs followed by the manifest.")
    ensure_targets_absent(targets)
    temporaries: OrderedDict[str, Path] = OrderedDict()
    promoted: list[Path] = []
    try:
        for name, target in targets.items():
            if not target.parent.is_dir():
                raise ValueError("Output parent directory is unavailable.")
            temporary = target.with_name(f".{target.name}.tmp")
            with temporary.open("xb") as stream:
                stream.write(payloads[name])
                stream.flush()
                os.fsync(stream.fileno())
            if temporary.read_bytes() != payloads[name]:
                raise ValueError("Staged output byte read-back failed.")
            temporaries[name] = temporary
        for name in OUTPUT_ORDER:
            os.replace(temporaries[name], targets[name])
            promoted.append(targets[name])
    except Exception:
        for target in reversed(promoted):
            if target.exists() and not target.is_symlink():
                target.unlink()
        raise
    finally:
        for temporary in temporaries.values():
            if temporary.exists() and not temporary.is_symlink():
                temporary.unlink()


def _frames_exact(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    """Return whether two scientific frames are exactly identical."""
    try:
        pd.testing.assert_frame_equal(left, right, check_exact=True)
    except AssertionError:
        return False
    return True


def _read_primary_bytes(payload: bytes, config: Mapping[str, Any]) -> pd.DataFrame:
    """Read and validate candidate primary bytes."""
    frame = pd.read_csv(io.BytesIO(payload), float_precision="round_trip")
    if frame.columns.tolist() != list(PRIMARY_COLUMNS):
        raise ValueError("Primary candidate schema is invalid.")
    result = _apply_primary_dtypes(frame)
    validate_primary(result, config)
    return result


def _read_baseline_bytes(payload: bytes, config: Mapping[str, Any]) -> pd.DataFrame:
    """Read and validate candidate baseline bytes."""
    frame = pd.read_csv(io.BytesIO(payload), float_precision="round_trip")
    if frame.columns.tolist() != list(BASELINE_COLUMNS):
        raise ValueError("Baseline candidate schema is invalid.")
    result = _apply_baseline_dtypes(frame)
    validate_baseline(result, config)
    return result


def _read_stability_bytes(payload: bytes, config: Mapping[str, Any]) -> pd.DataFrame:
    """Read and validate candidate stability bytes."""
    frame = pd.read_csv(io.BytesIO(payload), float_precision="round_trip")
    if frame.columns.tolist() != list(STABILITY_COLUMNS):
        raise ValueError("Stability candidate schema is invalid.")
    result = _apply_stability_dtypes(frame)
    validate_stability(result, config)
    return result


def read_and_validate_primary(path: Path, config: Mapping[str, Any]) -> pd.DataFrame:
    """Read and validate the promoted primary/estimability CSV."""
    return _read_primary_bytes(path.read_bytes(), config)


def read_and_validate_baseline(path: Path, config: Mapping[str, Any]) -> pd.DataFrame:
    """Read and validate the promoted baseline-context CSV."""
    return _read_baseline_bytes(path.read_bytes(), config)


def read_and_validate_stability(path: Path, config: Mapping[str, Any]) -> pd.DataFrame:
    """Read and validate the promoted stability CSV."""
    return _read_stability_bytes(path.read_bytes(), config)


def read_and_validate_manifest(path: Path) -> OrderedDict[str, Any]:
    """Read and validate deterministic manifest identity and privacy."""
    payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=OrderedDict)
    if payload.get("schema_version") != f"{PACKAGE_ID}.manifest.v1" or payload.get("package_id") != PACKAGE_ID:
        raise ValueError("Manifest schema or package identity is invalid.")
    validate_persisted_privacy({"manifest": path.read_bytes()})
    return payload


def validate_manifest_output_hashes(manifest: Mapping[str, Any], targets: Mapping[str, Path]) -> None:
    """Require manifest hashes to match the three promoted CSVs."""
    records = manifest.get("outputs")
    if not isinstance(records, list) or len(records) != 3:
        raise ValueError("Manifest output hash inventory is invalid.")
    by_role = {record.get("role"): record for record in records if isinstance(record, Mapping)}
    for role in OUTPUT_ORDER[:3]:
        if role not in targets or role not in by_role or sha256_file(targets[role]) != by_role[role].get("sha256"):
            raise ValueError("Manifest output hash agreement failed.")


def _load_aggregate_sources(
    root: Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, OrderedDict[str, Any]]:
    """Load the three aggregate CSVs and only the authorized LGIN audit projection."""
    try:
        primary = pd.read_csv(
            _input_path(root, config, "primary_oof_aggregate"),
            float_precision="round_trip",
        )
        baseline = pd.read_csv(
            _input_path(root, config, "legacy_baseline_aggregate"),
            float_precision="round_trip",
        )
        stability = pd.read_csv(
            _input_path(root, config, "legacy_stability_aggregate"),
            float_precision="round_trip",
        )
        full_audit = json.loads(_input_path(root, config, "stage7_part_b_aggregate_audit").read_text(encoding="utf-8"))
        references = full_audit["post_hoc_trigger_references"]
        audit_projection = OrderedDict(
            [
                ("git_commit_hash", full_audit["git_commit_hash"]),
                (
                    "post_hoc_trigger_references",
                    OrderedDict((field, references[field]) for field in config["anchors"]["audit_lgin_fields"]),
                ),
            ]
        )
        del full_audit, references
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        raise ValueError("Approved aggregate source loading failed.") from None
    return primary, baseline, stability, audit_projection


def _artifact_records(
    config: Mapping[str, Any],
    observed_hashes: Mapping[str, str],
) -> list[OrderedDict[str, Any]]:
    """Build ordered scientific input provenance records."""
    return [
        OrderedDict(
            [
                ("role", record["role"]),
                ("path", record["path"]),
                ("sha256", observed_hashes[record["role"]]),
                ("participant_capable", bool(record["participant_capable"])),
            ]
        )
        for record in config["scientific_inputs"]
    ]


def _build_manifest(
    *,
    root: Path,
    config: Mapping[str, Any],
    git_head: str,
    observed_hashes: Mapping[str, str],
    csv_payloads: Mapping[str, bytes],
) -> OrderedDict[str, Any]:
    """Build the exact deterministic aggregate provenance manifest."""
    implementation_records = [
        OrderedDict([("role", role), ("path", path), ("sha256", sha256_file(root / path))])
        for role, path in config["implementation"].items()
    ]
    outputs = [
        OrderedDict(
            [
                ("role", role),
                ("path", config["outputs"][role]["path"]),
                ("sha256", hashlib.sha256(csv_payloads[role]).hexdigest()),
            ]
        )
        for role in OUTPUT_ORDER[:3]
    ]
    baseline_definitions = [
        OrderedDict(
            [
                ("variable_order", int(record["variable_order"])),
                ("variable_key", record["variable_key"]),
                ("variable_type", record["variable_type"]),
                ("event_level", record.get("event_level")),
                ("difference_metric", record["difference_metric"]),
            ]
        )
        for record in config["baseline_variables"]
    ]
    stability_mapping = [
        OrderedDict(
            [
                ("variable_order", int(row[0])),
                ("legacy_variable", row[1]),
                ("legacy_level", row[2]),
                ("variable_key", row[3]),
                ("level_key", row[4]),
            ]
        )
        for row in config["stability_mapping"]
    ]
    return OrderedDict(
        [
            ("schema_version", f"{PACKAGE_ID}.manifest.v1"),
            ("package_id", PACKAGE_ID),
            ("package_status", "source_closure_only_not_publication_table"),
            ("intended_downstream_consumer", config["intended_downstream_consumer"]),
            ("closure_git_head", git_head),
            ("upstream_stage7_part_b_generation_commit", config["upstream_stage7_part_b_generation_commit"]),
            (
                "configuration",
                OrderedDict(
                    [
                        ("path", config["configuration"]["path"]),
                        ("sha256", sha256_file(root / config["configuration"]["path"])),
                    ]
                ),
            ),
            ("implementation", implementation_records),
            ("scientific_inputs", _artifact_records(config, observed_hashes)),
            ("outputs", outputs),
            (
                "manifest",
                OrderedDict(
                    [
                        ("path", config["outputs"]["manifest"]["path"]),
                        ("self_hash_policy", "not_computed_to_avoid_recursive_self_hash"),
                    ]
                ),
            ),
            (
                "proxy_mapping",
                OrderedDict(
                    [
                        ("legacy_source_field", "collection_location"),
                        (
                            "raw_values",
                            OrderedDict(
                                [
                                    ("1", "inpatient_treatment_proxy"),
                                    ("0", "outpatient_followup_proxy"),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
            ("analysis_view_roles", OrderedDict(config["analysis_views"])),
            (
                "baseline_contract",
                OrderedDict(
                    [
                        ("variables", baseline_definitions),
                        ("denominator_rule", "variable_specific_nonmissing"),
                        ("continuous_sd_ddof", 1),
                        ("quantile_method", "pandas_linear"),
                        (
                            "difference_metric_mapping",
                            OrderedDict(
                                [
                                    ("continuous", "absolute_standardized_mean_difference"),
                                    ("binary", "absolute_proportion_difference"),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
            (
                "stability_contract",
                OrderedDict(
                    [
                        ("minimum_n_for_aggregate_score_summary", 5),
                        ("small_level_status", "not_reported_small_level"),
                        ("small_level_reason", "n_in_level_below_5"),
                        ("level_mapping", stability_mapping),
                        ("reportable_levels", 13),
                        ("small_levels", 2),
                    ]
                ),
            ),
            (
                "primary_and_test_closure",
                OrderedDict(
                    [
                        ("training_nested_cv_oof", OrderedDict(config["anchors"]["oof"])),
                        ("locked_internal_test", OrderedDict(config["anchors"]["locked_internal_test"])),
                        ("aggregate_reconciliation_status", "passed"),
                    ]
                ),
            ),
            (
                "privacy_contract",
                OrderedDict(
                    [
                        ("participant_join_keys_transient_in_memory_only", True),
                        ("participant_level_output_produced", False),
                        ("test_prediction_rows_read", False),
                        ("raw_breath_records_read", False),
                        ("forbidden_sources_read", False),
                        ("persisted_outputs_aggregate_only", True),
                    ]
                ),
            ),
            (
                "explicit_non_actions",
                OrderedDict(
                    [
                        ("model_changed", False),
                        ("split_changed", False),
                        ("calibrator_changed", False),
                        ("threshold_changed", False),
                        ("new_inference_performed", False),
                        ("publication_table_generated", False),
                    ]
                ),
            ),
            (
                "interpretation_contract",
                OrderedDict(
                    [
                        ("population", "locked_cross_sectional_lgin_disease_category_2"),
                        ("oof_view", "internal_development_nested_cv_out_of_fold"),
                        ("test_view", "aggregate_counts_and_estimability_only"),
                        ("p_cal_scale", "internally_calibrated_model_score_not_target_population_absolute_risk"),
                        ("descriptive_criterion", "p_cal_gt_0_5_not_a_clinical_cutoff"),
                        ("reference_ascertainment", "biopsy_based_and_may_undergrade_disease"),
                        ("analysis_scope", "exploratory_postdiagnosis_disposition_proxy"),
                    ]
                ),
            ),
            (
                "validation",
                OrderedDict(
                    [
                        ("candidate_read_back", True),
                        ("deterministic_reconstruction", True),
                        ("persisted_privacy", True),
                        ("frozen_input_immutability", True),
                        ("primary_test_counts_reconciled", True),
                        ("legacy_baseline_reconciled", True),
                        ("legacy_stability_reconciled", True),
                    ]
                ),
            ),
        ]
    )


def _remove_new_outputs(targets: Mapping[str, Path]) -> None:
    """Remove only outputs created after a verified all-absent precondition."""
    for target in reversed(list(targets.values())):
        if target.exists() and not target.is_symlink():
            target.unlink()


def execute_approved(config_path: Path, *, root: Path = PROJECT_ROOT) -> OrderedDict[str, Any]:
    """Execute the single authorized aggregate-only source closure.

    Args:
        config_path: Exact package YAML path.
        root: Project root; injectable only for synthetic tests.

    Returns:
        Aggregate row counts and four output hashes.

    Raises:
        ValueError: If any authority, scientific, privacy, or read-back gate fails.
        FileExistsError: If any final or temporary output already exists.
    """
    resolved_config = config_path if config_path.is_absolute() else root / config_path
    config = load_config(resolved_config)
    approved_config = (root / config["configuration"]["path"]).resolve()
    if resolved_config.resolve() != approved_config:
        raise ValueError("Configuration path does not match the approved contract.")
    git_head = validate_execution_authority(root, config)
    targets = OrderedDict(
        (name, root / _validate_relative_path(record["path"])) for name, record in config["outputs"].items()
    )
    ensure_targets_absent(targets)
    before_hashes = verify_scientific_inputs(root, config["scientific_inputs"])
    primary_source, legacy_baseline, legacy_stability, audit = _load_aggregate_sources(root, config)
    lgin_frame = load_authorized_participant_context(root, config)
    primary = build_primary_estimability(primary_source, audit, config)
    baseline = build_baseline_context(lgin_frame, legacy_baseline, config)
    stability = build_stability_source(legacy_stability, config)
    repeated_primary = build_primary_estimability(primary_source, audit, config)
    repeated_baseline = build_baseline_context(lgin_frame, legacy_baseline, config)
    repeated_stability = build_stability_source(legacy_stability, config)
    del lgin_frame
    if not all(
        (
            _frames_exact(primary, repeated_primary),
            _frames_exact(baseline, repeated_baseline),
            _frames_exact(stability, repeated_stability),
        )
    ):
        raise ValueError("Deterministic aggregate reconstruction failed.")
    csv_payloads = OrderedDict(
        [
            ("primary_estimability", serialize_csv(primary, PRIMARY_COLUMNS)),
            ("baseline_context", serialize_csv(baseline, BASELINE_COLUMNS)),
            ("within_inpatient_stability", serialize_csv(stability, STABILITY_COLUMNS)),
        ]
    )
    if (
        csv_payloads["primary_estimability"] != serialize_csv(repeated_primary, PRIMARY_COLUMNS)
        or csv_payloads["baseline_context"] != serialize_csv(repeated_baseline, BASELINE_COLUMNS)
        or csv_payloads["within_inpatient_stability"] != serialize_csv(repeated_stability, STABILITY_COLUMNS)
    ):
        raise ValueError("Deterministic aggregate serialization failed.")
    validate_persisted_privacy(csv_payloads)
    candidate_primary = _read_primary_bytes(csv_payloads["primary_estimability"], config)
    candidate_baseline = _read_baseline_bytes(csv_payloads["baseline_context"], config)
    candidate_stability = _read_stability_bytes(csv_payloads["within_inpatient_stability"], config)
    if not all(
        (
            _frames_exact(primary, candidate_primary),
            _frames_exact(baseline, candidate_baseline),
            _frames_exact(stability, candidate_stability),
        )
    ):
        raise ValueError("Candidate aggregate numeric round-trip validation failed.")
    after_construction_hashes = verify_scientific_inputs(root, config["scientific_inputs"])
    if before_hashes != after_construction_hashes:
        raise ValueError("Frozen scientific inputs changed during aggregate construction.")
    manifest = _build_manifest(
        root=root,
        config=config,
        git_head=git_head,
        observed_hashes=before_hashes,
        csv_payloads=csv_payloads,
    )
    manifest_bytes = serialize_manifest(manifest)
    parsed_manifest = json.loads(manifest_bytes, object_pairs_hook=OrderedDict)
    if parsed_manifest != manifest:
        raise ValueError("Manifest candidate read-back validation failed.")
    payloads = OrderedDict([*csv_payloads.items(), ("manifest", manifest_bytes)])
    validate_persisted_privacy(payloads)
    promoted = False
    try:
        promote_serialized_outputs(targets, payloads)
        promoted = True
        persisted_primary = read_and_validate_primary(targets["primary_estimability"], config)
        persisted_baseline = read_and_validate_baseline(targets["baseline_context"], config)
        persisted_stability = read_and_validate_stability(targets["within_inpatient_stability"], config)
        persisted_manifest = read_and_validate_manifest(targets["manifest"])
        if not all(
            (
                _frames_exact(primary, persisted_primary),
                _frames_exact(baseline, persisted_baseline),
                _frames_exact(stability, persisted_stability),
            )
        ):
            raise ValueError("Promoted aggregate numeric read-back validation failed.")
        validate_manifest_output_hashes(persisted_manifest, OrderedDict(list(targets.items())[:3]))
        final_hashes = verify_scientific_inputs(root, config["scientific_inputs"])
        if before_hashes != final_hashes:
            raise ValueError("Frozen scientific inputs changed during production execution.")
    except Exception:
        if promoted:
            _remove_new_outputs(targets)
        raise
    return OrderedDict(
        [
            ("primary_rows", len(primary)),
            ("baseline_rows", len(baseline)),
            ("stability_rows", len(stability)),
            ("primary_sha256", sha256_file(targets["primary_estimability"])),
            ("baseline_sha256", sha256_file(targets["baseline_context"])),
            ("stability_sha256", sha256_file(targets["within_inpatient_stability"])),
            ("manifest_sha256", sha256_file(targets["manifest"])),
        ]
    )


__all__ = [
    "BASELINE_COLUMNS",
    "PRIMARY_COLUMNS",
    "STABILITY_COLUMNS",
    "build_baseline_context",
    "build_lgin_participant_context",
    "build_primary_estimability",
    "build_stability_source",
    "execute_approved",
    "load_authorized_participant_context",
    "load_config",
    "promote_serialized_outputs",
    "read_and_validate_baseline",
    "read_and_validate_manifest",
    "read_and_validate_primary",
    "read_and_validate_stability",
    "serialize_csv",
    "serialize_manifest",
    "sha256_file",
    "validate_execution_authority",
    "validate_manifest_output_hashes",
    "validate_persisted_privacy",
    "verify_scientific_inputs",
]
