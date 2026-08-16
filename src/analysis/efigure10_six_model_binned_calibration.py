"""Build the frozen six-model eFigure 10 calibration aggregate."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import importlib.metadata
import io
import json
import os
import re
import subprocess
import sys
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import yaml

from src.analysis.posthoc_final_common import wilson_proportion_ci
from src.analysis.posthoc_final_reporting import build_calibration_points


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
TASK_CONFIG_PATH: Final[Path] = Path("configs/efigure10_six_model_binned_calibration.yaml")
TASK_CONFIG_SHA256: Final[str] = "97d8973c1daaa93221eec486aa87a83c876c4ce2e5ca381a096e6f9b6a4ebf2d"
TASK_CONFIG_SCHEMA_SHA256: Final[str] = "97d8973c1daaa93221eec486aa87a83c876c4ce2e5ca381a096e6f9b6a4ebf2d"
FROZEN_REPORTING_CONFIG_PATH: Final[Path] = Path("configs/post_hoc_final_supplementary.yaml")
FROZEN_REPORTING_CONFIG_SHA256: Final[str] = (
    "32a0ffc546053f01e4f2a2b63542b4404d5ce4f9d3f85fcd91b43d4cf06158f9"
)
EXPECTED_TOP_LEVEL_KEYS: Final[tuple[str, ...]] = (
    "schema_version",
    "analysis",
    "authority",
    "environment",
    "models",
    "immutable_references",
    "input_contract",
    "calibration_contract",
    "output_contract",
    "privacy_contract",
    "external_namespace_contract",
)
EXPECTED_MODEL_KEYS: Final[tuple[str, ...]] = (
    "model_order",
    "model_id",
    "model_display_name",
    "prediction_path",
    "sha256",
)
G06_PATHS: Final[tuple[str, ...]] = (
    "configs/efigure10_six_model_binned_calibration.yaml",
    "scripts/run_efigure10_six_model_binned_calibration.py",
    "src/analysis/efigure10_six_model_binned_calibration.py",
    "tests/test_efigure10_six_model_binned_calibration.py",
    "outputs/work/efigure10_six_model_binned_calibration",
    "outputs/analysis/efigure10_six_model_binned_calibration",
)
G06_ARGV: Final[tuple[str, ...]] = (
    "git",
    "status",
    "--short",
    "--branch",
    "--untracked-files=normal",
    "--",
    *G06_PATHS,
)
TASK_CONFIG_SCHEMA_PATH: Final[Path] = Path("configs/efigure10_six_model_binned_calibration.yaml")
COMMON_COLUMNS: Final[tuple[str, ...]] = (
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
INTERPRETATION_BOUNDARIES: Final[tuple[str, ...]] = (
    "This is an internal, same-center, same-device, same-period random holdout, not external or temporal validation.",
    "The study uses an enriched diagnostic case-control/two-gate sample.",
    "Breath was collected after diagnostic endoscopy and indicated biopsy but before treatment.",
    "The aggregation does not establish population-screening calibration.",
    "It does not establish pre-endoscopy triage validity.",
    "It does not establish target-population absolute-risk calibration.",
    "It does not establish external transportability.",
    "It does not establish deployment readiness, clinical utility, clinical benefit, or causality.",
    "p_cal is not a gastric-cancer probability.",
    "Bins are model-specific, data-dependent quantile groups.",
    "The same bin number across models does not represent the same probability interval or participant set.",
    "Wilson intervals quantify uncertainty in the observed positive proportion within a data-dependent bin.",
    "They are not confidence bands for the entire calibration function.",
    "They are not confidence intervals for the fitted Platt calibrator.",
    "They do not provide simultaneous coverage.",
    "The analysis supports descriptive calibration characterization only.",
    "It must not rank models by calibration or support a generic well-calibrated claim.",
    "Mixed, unfavorable, nonmonotonic, collapsed-bin, inconclusive, and not-visualization-ready results are retained.",
    "LGIN-related interpretation remains exploratory because biopsy may undergrade disease and reference standards "
    "differ; this differential-reference-standard limitation remains explicit.",
)


class ParticipantIdentifierReferences:
    """Hold transient identifier tokens without exposing or serializing them."""

    __slots__ = ("__nonnumeric_tokens", "__numeric_tokens", "__cleared")
    __hash__ = None

    _NUMERIC_GRAMMAR: Final[re.Pattern[str]] = re.compile(r"[+-]?(?:0|[1-9][0-9]*)\Z")
    _IDENTIFIER_QUALIFIERS: Final[frozenset[str]] = frozenset(
        {"sample_id", "participant_id", "subject_id", "record_id", "identifier"}
    )

    def __init__(self, sample_ids: Sequence[str]) -> None:
        """Create references from already validated exact identifier lexemes."""
        nonnumeric: set[str] = set()
        numeric: set[str] = set()
        for sample_id in sample_ids:
            if not isinstance(sample_id, str):
                raise TypeError("Participant identifier reference input must contain strings")
            if self._NUMERIC_GRAMMAR.fullmatch(sample_id):
                numeric.add(sample_id)
            else:
                nonnumeric.add(sample_id)
        self.__nonnumeric_tokens = nonnumeric
        self.__numeric_tokens = numeric
        self.__cleared = False

    def __repr__(self) -> str:
        """Return a redacted representation."""
        return "<ParticipantIdentifierReferences redacted>"

    __str__ = __repr__

    def matches_nonnumeric_token(self, candidate: str) -> bool:
        """Return whether a complete nonnumeric identifier token is present."""
        if not isinstance(candidate, str):
            raise TypeError("Privacy candidate must be a string")
        return not self.__cleared and candidate in self.__nonnumeric_tokens

    def matches_identifier_qualified_numeric(self, qualifier: str, candidate: str) -> bool:
        """Match a numeric token only when it appears in an identifier field."""
        if not isinstance(qualifier, str) or not isinstance(candidate, str):
            raise TypeError("Privacy qualifier and candidate must be strings")
        return (
            not self.__cleared
            and qualifier.casefold() in self._IDENTIFIER_QUALIFIERS
            and candidate in self.__numeric_tokens
        )

    def matches_serialized_text(self, candidate: str, qualifiers: Sequence[str]) -> bool:
        """Return whether serialized text contains an identifier token."""
        if not isinstance(candidate, str):
            raise TypeError("Privacy candidate must be a string")
        if self.__cleared:
            return False
        for token in self.__nonnumeric_tokens:
            if candidate == token:
                return True
            boundary_pattern = rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])"
            if re.search(boundary_pattern, candidate):
                return True
        approved_qualifiers = tuple(
            qualifier for qualifier in qualifiers if qualifier.casefold() in self._IDENTIFIER_QUALIFIERS
        )
        for qualifier in approved_qualifiers:
            for token in self.__numeric_tokens:
                pattern = (
                    rf"(?i:(?<![A-Za-z0-9_]){re.escape(qualifier)})"
                    rf"(?:[ \t]*[:=][ \t]*|,[ \t]*|[ \t])"
                    rf"{re.escape(token)}(?![A-Za-z0-9_])"
                )
                if re.search(pattern, candidate):
                    return True
        return False

    def is_cleared(self) -> bool:
        """Return whether all transient tokens have been discarded."""
        return self.__cleared

    def _clear(self) -> None:
        """Clear all private tokens in place."""
        self.__nonnumeric_tokens.clear()
        self.__numeric_tokens.clear()
        self.__cleared = True

    def __reduce_ex__(self, protocol: int) -> object:
        """Reject serialization of transient identifier references."""
        del protocol
        raise TypeError("Participant identifier references are not serializable")


class _UniqueSafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate and merge keys recursively."""


def _construct_unique_mapping(
    loader: yaml.SafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[str, Any]:
    """Construct one mapping while rejecting duplicate and merge keys."""
    mapping: dict[str, Any] = {}
    for key_node, value_node in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge" or key_node.value == "<<":
            raise ValueError("YAML merge/default expansion is forbidden")
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ValueError("YAML mapping contains an unhashable key") from exc
        if duplicate:
            raise ValueError("YAML contains a duplicate mapping key")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _sha256_path(path: Path) -> str:
    """Return one whole-file SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_project_path(path: Path, *, project_root: Path) -> Path:
    """Resolve an existing nonsymlink path beneath the project root."""
    if not isinstance(path, Path) or not isinstance(project_root, Path):
        raise TypeError("Paths must be pathlib.Path instances")
    root = project_root.resolve(strict=True)
    candidate = path if path.is_absolute() else root / path
    if candidate.is_symlink():
        raise ValueError("Authorized path must not be a symbolic link")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError("Required project-relative path is absent") from exc
    if not resolved.is_relative_to(root):
        raise ValueError("Authorized path escapes the project root")
    return resolved


def load_strict_yaml_mapping(
    path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Load a safe YAML mapping with recursive duplicate-key rejection.

    Args:
        path: Project-contained YAML path.
        project_root: Root used for path confinement.

    Returns:
        The insertion-ordered decoded mapping.

    Raises:
        FileNotFoundError: If the path is absent.
        ValueError: If path or YAML safety and mapping rules fail.
    """
    resolved = _resolve_project_path(path, project_root=project_root)
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            loaded = yaml.load(handle, Loader=_UniqueSafeLoader)
    except yaml.YAMLError as exc:
        raise ValueError("YAML failed strict safe loading") from exc
    if not isinstance(loaded, dict):
        raise ValueError("YAML top level must be a mapping")
    return loaded


def load_frozen_reporting_config(
    path: Path,
    *,
    project_root: Path,
) -> OrderedDict[str, Any]:
    """Load only the hash-bound frozen reporting projection.

    This compatibility boundary is limited to the immutable reporting YAML,
    whose two known identical legacy duplicate declarations cannot be loaded
    by the general task-configuration loader.

    Args:
        path: Exact canonical frozen reporting configuration path.
        project_root: Root used for path confinement.

    Returns:
        The ordered builder-only projection required by eFigure 10.

    Raises:
        ValueError: If the path, hash, YAML, or required projection differs.
    """
    if not isinstance(path, Path) or path != FROZEN_REPORTING_CONFIG_PATH:
        raise ValueError("Frozen reporting compatibility path is not authorized")
    resolved = _resolve_project_path(path, project_root=project_root)
    frozen_bytes = resolved.read_bytes()
    if hashlib.sha256(frozen_bytes).hexdigest() != FROZEN_REPORTING_CONFIG_SHA256:
        raise ValueError("Frozen reporting configuration hash does not match authority")
    try:
        loaded = yaml.safe_load(frozen_bytes.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError("Frozen reporting configuration failed safe loading") from exc
    if not isinstance(loaded, Mapping):
        raise ValueError("Frozen reporting configuration must be a mapping")
    try:
        calibration_points = loaded["outputs"]["scientific_and_closure"]["calibration_points"]
        registry = loaded["row_order_and_schema_validation_registry"]["calibration_points"]
        projection = OrderedDict(
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
                                                    ("columns", _copy_ordered(calibration_points["columns"])),
                                                    (
                                                        "conditional_row_inventory_contract",
                                                        OrderedDict(
                                                            (
                                                                (
                                                                    "n_requested_bins",
                                                                    calibration_points[
                                                                        "conditional_row_inventory_contract"
                                                                    ]["n_requested_bins"],
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
                                        ("column_order", _copy_ordered(registry["column_order"])),
                                        ("dtype_map", _copy_ordered(registry["dtype_map"])),
                                        ("nullable_map", _copy_ordered(registry["nullable_map"])),
                                        (
                                            "missing_representation",
                                            _copy_ordered(registry["missing_representation"]),
                                        ),
                                    )
                                ),
                            ),
                        )
                    ),
                ),
            )
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("Frozen reporting builder projection is incomplete") from exc
    return projection


def _require_key_order(mapping: object, expected: Sequence[str], role: str) -> None:
    """Require an exact insertion-ordered mapping key inventory."""
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{role} must be a mapping")
    if tuple(mapping) != tuple(expected):
        raise ValueError(f"{role} key order does not match the closed contract")


def _closed_structure_matches(candidate: object, authority: object) -> bool:
    """Return whether a value matches an independently hash-bound closed structure."""
    if isinstance(authority, Mapping):
        return (
            isinstance(candidate, Mapping)
            and tuple(candidate) == tuple(authority)
            and all(_closed_structure_matches(candidate[key], authority[key]) for key in authority)
        )
    if isinstance(authority, list):
        if not isinstance(candidate, list):
            return False
        if not authority:
            return not candidate
        for item in candidate:
            if not any(_closed_structure_matches(item, template) for template in authority):
                return False
        return True
    return type(candidate) is type(authority)


def _load_closed_config_structure_authority() -> dict[str, Any]:
    """Load the exact canonical task configuration as structure-only authority."""
    resolved = _resolve_project_path(TASK_CONFIG_SCHEMA_PATH, project_root=PROJECT_ROOT)
    if _sha256_path(resolved) != TASK_CONFIG_SCHEMA_SHA256:
        raise ValueError("Canonical task configuration structure authority is invalid")
    return load_strict_yaml_mapping(TASK_CONFIG_SCHEMA_PATH, project_root=PROJECT_ROOT)


def load_config(config_path: Path, *, project_root: Path) -> dict[str, Any]:
    """Load and validate the closed task configuration.

    Args:
        config_path: Project-relative task configuration path.
        project_root: Root used for path confinement.

    Returns:
        The validated configuration mapping without inserted defaults.

    Raises:
        ValueError: If the configuration is not closed and ordered.
    """
    config = load_strict_yaml_mapping(config_path, project_root=project_root)
    _require_key_order(config, EXPECTED_TOP_LEVEL_KEYS, "Task configuration")
    structure_authority = _load_closed_config_structure_authority()
    if not _closed_structure_matches(config, structure_authority):
        raise ValueError("Task configuration nested structure does not match the closed contract")
    if config["schema_version"] != 1:
        raise ValueError("Task configuration schema version is invalid")
    models = config["models"]
    if not isinstance(models, list) or len(models) != 6:
        raise ValueError("Model registry must contain exactly six records")
    for record in models:
        _require_key_order(record, EXPECTED_MODEL_KEYS, "Model registry record")
    model_orders = [record["model_order"] for record in models]
    if model_orders != list(range(1, 7)):
        raise ValueError("Model registry order is invalid")
    if len({record["model_id"] for record in models}) != 6:
        raise ValueError("Model registry identifiers must be unique")
    return config


def _copy_ordered(value: Any) -> Any:
    """Copy nested mappings and lists while retaining mapping order."""
    if isinstance(value, Mapping):
        return OrderedDict((key, _copy_ordered(item)) for key, item in value.items())
    if isinstance(value, list):
        return [_copy_ordered(item) for item in value]
    return value


def _recursive_equal(actual: object, expected: object) -> bool:
    """Compare nested values including mapping key order."""
    if isinstance(expected, pd.DataFrame):
        if not isinstance(actual, pd.DataFrame):
            return False
        try:
            pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        except AssertionError:
            return False
        return True
    if isinstance(expected, pd.Series):
        if not isinstance(actual, pd.Series):
            return False
        try:
            pd.testing.assert_series_equal(actual, expected, check_exact=True)
        except AssertionError:
            return False
        return True
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and tuple(actual) == tuple(expected)
            and all(_recursive_equal(actual[key], expected[key]) for key in expected)
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_recursive_equal(left, right) for left, right in zip(actual, expected, strict=True))
        )
    return actual == expected


def build_frozen_calibration_builder_config(
    calibration_contract: Mapping[str, Any],
    *,
    frozen_reporting_config: Mapping[str, Any],
) -> OrderedDict[str, Any]:
    """Derive and validate the pure in-memory frozen-builder adapter.

    Args:
        calibration_contract: Already validated task calibration mapping.
        frozen_reporting_config: Immutable reporting configuration.

    Returns:
        The exact two-key builder-only configuration mapping.

    Raises:
        ValueError: If the task and frozen projections differ.
    """
    if not isinstance(calibration_contract, Mapping) or not isinstance(frozen_reporting_config, Mapping):
        raise TypeError("Builder adapter inputs must be mappings")
    common_columns = list(calibration_contract.get("common_columns", []))
    requested_bins = calibration_contract.get("n_requested_bins")
    try:
        frozen_points = frozen_reporting_config["outputs"]["scientific_and_closure"]["calibration_points"]
        frozen_registry = frozen_reporting_config["row_order_and_schema_validation_registry"]["calibration_points"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Frozen reporting builder contract is incomplete") from exc
    adapter: OrderedDict[str, Any] = OrderedDict(
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
                                                ("columns", common_columns),
                                                (
                                                    "conditional_row_inventory_contract",
                                                    OrderedDict((("n_requested_bins", requested_bins),)),
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
                                    ("column_order", common_columns),
                                    ("dtype_map", _copy_ordered(frozen_registry["dtype_map"])),
                                    ("nullable_map", _copy_ordered(frozen_registry["nullable_map"])),
                                    (
                                        "missing_representation",
                                        _copy_ordered(frozen_registry["missing_representation"]),
                                    ),
                                )
                            ),
                        ),
                    )
                ),
            ),
        )
    )
    frozen_projection = OrderedDict(
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
                                                ("columns", _copy_ordered(frozen_points["columns"])),
                                                (
                                                    "conditional_row_inventory_contract",
                                                    OrderedDict(
                                                        (
                                                            (
                                                                "n_requested_bins",
                                                                frozen_points[
                                                                    "conditional_row_inventory_contract"
                                                                ]["n_requested_bins"],
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
                                    ("column_order", _copy_ordered(frozen_registry["column_order"])),
                                    ("dtype_map", _copy_ordered(frozen_registry["dtype_map"])),
                                    ("nullable_map", _copy_ordered(frozen_registry["nullable_map"])),
                                    (
                                        "missing_representation",
                                        _copy_ordered(frozen_registry["missing_representation"]),
                                    ),
                                )
                            ),
                        ),
                    )
                ),
            ),
        )
    )
    if not _recursive_equal(adapter, frozen_projection):
        raise ValueError("Task calibration contract differs from the frozen reporting contract")
    return adapter


def _validate_exact_numeric_column(series: pd.Series, *, role: str) -> pd.Series:
    """Parse one CSV numeric column without repairing invalid values."""
    try:
        numeric = pd.to_numeric(series, errors="raise")
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{role} must contain exact numeric values") from exc
    if numeric.isna().any():
        raise TypeError(f"{role} must not contain missing values")
    return numeric


def _load_prediction_projection_with_evidence(
    prediction_path: Path,
    *,
    project_root: Path,
) -> tuple[pd.DataFrame, OrderedDict[str, Any]]:
    """Read and validate one minimal projection with operation-derived evidence.

    Args:
        prediction_path: Allowlisted prediction CSV path.
        project_root: Root used for path confinement.

    Returns:
        The three-column frame and aggregate-only validation evidence.

    Raises:
        TypeError: If labels or probabilities are not numeric.
        ValueError: If schema, identifiers, labels, or probabilities are invalid.
    """
    resolved = _resolve_project_path(prediction_path, project_root=project_root)
    header = pd.read_csv(resolved, nrows=0, encoding="utf-8")
    expected_header = ("sample_id", "y_true", "p_raw", "p_cal", "y_pred", "quadrant")
    if tuple(header.columns) != expected_header:
        raise ValueError("Prediction schema does not match the exact ordered contract")
    frame = pd.read_csv(
        resolved,
        usecols=["sample_id", "y_true", "p_cal"],
        dtype={"sample_id": "string"},
        keep_default_na=False,
        na_filter=False,
        skipinitialspace=False,
        encoding="utf-8",
    )
    frame = frame.loc[:, ["sample_id", "y_true", "p_cal"]]
    if frame["sample_id"].isna().any() or (frame["sample_id"] == "").any():
        raise ValueError("Prediction identifiers must be present")
    if not frame["sample_id"].is_unique:
        raise ValueError("Prediction identifiers must be unique")
    labels = _validate_exact_numeric_column(frame["y_true"], role="Prediction labels")
    label_values = labels.to_numpy()
    if not np.all(np.isfinite(label_values.astype(float))):
        raise ValueError("Prediction labels must be finite")
    if not np.all(np.equal(label_values, np.floor(label_values.astype(float)))):
        raise ValueError("Prediction labels must be exact integers")
    if not set(label_values.tolist()).issubset({0, 1}):
        raise ValueError("Prediction labels must be binary")
    scores = _validate_exact_numeric_column(frame["p_cal"], role="Calibrated probabilities")
    score_values = scores.to_numpy(dtype=float)
    if not np.isfinite(score_values).all():
        raise ValueError("Calibrated probabilities must be finite")
    if ((score_values < 0.0) | (score_values > 1.0)).any():
        raise ValueError("Calibrated probabilities must be within zero and one")
    frame["y_true"] = labels.astype("int64")
    frame["p_cal"] = scores.astype("float64")
    evidence: OrderedDict[str, Any] = OrderedDict(
        (
            ("header_only_status", "passed"),
            ("projection_columns", list(frame.columns)),
            ("unused_participant_columns_loaded", False),
            ("identifier_lexeme_status", "passed"),
            ("probability_status", "passed"),
            ("label_status", "passed"),
        )
    )
    return frame, evidence


def load_prediction_projection(
    prediction_path: Path,
    *,
    project_root: Path,
) -> pd.DataFrame:
    """Read an exact header followed by the minimal prediction projection."""
    frame, _evidence = _load_prediction_projection_with_evidence(
        prediction_path,
        project_root=project_root,
    )
    frame.attrs["validation_evidence"] = _evidence
    return frame


def build_participant_identifier_references(
    sample_ids: Sequence[str],
) -> ParticipantIdentifierReferences:
    """Build opaque references from exact aligned identifier lexemes."""
    return ParticipantIdentifierReferences(sample_ids)


def _load_and_reconcile_predictions_with_evidence(
    config: Mapping[str, Any],
    *,
    project_root: Path,
) -> tuple[
    OrderedDict[str, pd.DataFrame],
    ParticipantIdentifierReferences,
    OrderedDict[str, Any],
]:
    """Load and reconcile six projections with aggregate-only evidence.

    Args:
        config: Validated task configuration.
        project_root: Root used for path confinement.

    Returns:
        Ordered frames, opaque references, and validation evidence.

    Raises:
        ValueError: If hashes, totals, identifiers, or labels disagree.
    """
    if not isinstance(config, Mapping):
        raise TypeError("Configuration must be a mapping")
    models = config.get("models")
    input_contract = config.get("input_contract")
    if not isinstance(models, list) or not isinstance(input_contract, Mapping):
        raise ValueError("Prediction loading configuration is incomplete")
    expected_count = input_contract["participant_count"]
    expected_positive = input_contract["positive_count"]
    expected_negative = input_contract["negative_count"]
    loaded: OrderedDict[str, pd.DataFrame] = OrderedDict()
    reference_ids: list[str] | None = None
    reference_labels: pd.Series | None = None
    resolved_predictions: OrderedDict[str, Path] = OrderedDict()
    projection_evidence: OrderedDict[str, Any] | None = None
    for record in models:
        relative_path = Path(record["prediction_path"])
        resolved = _resolve_project_path(relative_path, project_root=project_root)
        if _sha256_path(resolved) != record["sha256"]:
            raise ValueError("Prediction whole-file hash does not match authority")
        resolved_predictions[record["model_id"]] = resolved
    for record in models:
        relative_path = Path(record["prediction_path"])
        frame = load_prediction_projection(relative_path, project_root=project_root)
        current_evidence = frame.attrs.pop("validation_evidence", None)
        if not isinstance(current_evidence, Mapping):
            raise ValueError("Prediction projection validation evidence is absent")
        if projection_evidence is None:
            projection_evidence = current_evidence
        elif not _recursive_equal(projection_evidence, current_evidence):
            raise ValueError("Prediction projection validation evidence is inconsistent")
        if len(frame) != expected_count or not frame["sample_id"].is_unique:
            raise ValueError("Prediction cohort size or uniqueness is invalid")
        positives = int(frame["y_true"].sum())
        negatives = len(frame) - positives
        if (positives, negatives) != (expected_positive, expected_negative):
            raise ValueError("Prediction endpoint totals do not match authority")
        model_id = record["model_id"]
        current_ids = frame["sample_id"].tolist()
        if reference_ids is None:
            reference_ids = current_ids
            reference_labels = frame["y_true"].reset_index(drop=True)
            aligned = frame.reset_index(drop=True)
        else:
            if set(current_ids) != set(reference_ids):
                raise ValueError("Prediction participant sets do not match")
            aligned = frame.set_index("sample_id", drop=False).loc[reference_ids].reset_index(drop=True)
            if not aligned["y_true"].equals(reference_labels):
                raise ValueError("Prediction labels do not agree after alignment")
        loaded[model_id] = aligned
    if reference_ids is None or reference_labels is None or projection_evidence is None:
        raise ValueError("Prediction model registry is empty")
    references = build_participant_identifier_references(reference_ids)
    input_validation: OrderedDict[str, Any] = OrderedDict(
        (
            ("model_registry_status", "passed"),
            ("header_only_status", projection_evidence["header_only_status"]),
            ("projection_columns", projection_evidence["projection_columns"]),
            (
                "unused_participant_columns_loaded",
                projection_evidence["unused_participant_columns_loaded"],
            ),
            ("row_count_status", "passed"),
            ("identifier_lexeme_status", projection_evidence["identifier_lexeme_status"]),
            ("probability_status", projection_evidence["probability_status"]),
            ("label_status", projection_evidence["label_status"]),
        )
    )
    observed_positive = int(reference_labels.sum())
    cohort_reconciliation: OrderedDict[str, Any] = OrderedDict(
        (
            ("models", len(loaded)),
            ("participants_per_model", len(reference_ids)),
            ("positive_labels", observed_positive),
            ("negative_labels", len(reference_ids) - observed_positive),
            ("participant_set_agreement", "passed"),
            ("cross_model_label_agreement", "passed"),
        )
    )
    evidence: OrderedDict[str, Any] = OrderedDict(
        (
            ("input_validation", input_validation),
            ("cohort_reconciliation", cohort_reconciliation),
        )
    )
    return loaded, references, evidence


def load_and_reconcile_predictions(
    config: Mapping[str, Any],
    *,
    project_root: Path,
) -> tuple[OrderedDict[str, pd.DataFrame], ParticipantIdentifierReferences]:
    """Load six projections and require exact cohort and label alignment."""
    loaded, references, _evidence = _load_and_reconcile_predictions_with_evidence(
        config,
        project_root=project_root,
    )
    return loaded, references


def _coerce_unified_dtypes(frame: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Apply the exact unified pandas dtype contract."""
    dtype_map = config["calibration_contract"]["pandas_dtypes"]
    result = frame.copy()
    for column, dtype_name in dtype_map.items():
        if dtype_name == "category":
            result[column] = pd.Categorical(result[column], categories=["estimable"], ordered=False)
        else:
            result[column] = result[column].astype(dtype_name)
    return result


def build_unified_calibration_points(
    predictions: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
    *,
    builder_config: Mapping[str, Any],
) -> pd.DataFrame:
    """Call the frozen builder once per model and construct the unified table.

    Args:
        predictions: Ordered aligned three-column frames by model ID.
        config: Validated task configuration.
        builder_config: Exact immutable frozen-builder adapter.

    Returns:
        The exact typed 17-column aggregate.

    Raises:
        ValueError: If model identities or the unified contract disagree.
    """
    if not isinstance(predictions, Mapping) or not isinstance(config, Mapping):
        raise TypeError("Unified calibration inputs must be mappings")
    expected_ids = [record["model_id"] for record in config["models"]]
    if list(predictions) != expected_ids:
        raise ValueError("Prediction model order does not match the registry")
    pieces: list[pd.DataFrame] = []
    for record in config["models"]:
        frame = predictions[record["model_id"]]
        points = build_calibration_points(frame["y_true"], frame["p_cal"], builder_config).copy()
        points.insert(0, "model_display_name", record["model_display_name"])
        points.insert(0, "model_id", record["model_id"])
        points.insert(0, "model_order", record["model_order"])
        pieces.append(points)
    unified = pd.concat(pieces, ignore_index=True)
    unified = unified.loc[:, config["calibration_contract"]["unified_columns"]]
    unified = _coerce_unified_dtypes(unified, config)
    validate_unified_calibration_points(unified, config)
    return unified


def validate_unified_calibration_points(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> None:
    """Validate schema, types, order, bin identities, and aggregate totals."""
    if not isinstance(frame, pd.DataFrame) or not isinstance(config, Mapping):
        raise TypeError("Unified validation requires a DataFrame and mapping")
    contract = config["calibration_contract"]
    if frame.columns.tolist() != contract["unified_columns"]:
        raise ValueError("Unified calibration columns are invalid")
    actual_dtypes = {column: str(frame[column].dtype) for column in frame.columns}
    if actual_dtypes != contract["pandas_dtypes"]:
        raise ValueError("Unified calibration dtypes are invalid")
    if frame.isna().drop(columns=["nonestimable_reason"]).any().any():
        raise ValueError("Unified calibration contains unexpected missing values")
    if not frame["nonestimable_reason"].isna().all():
        raise ValueError("Unified calibration reason must be null for estimable bins")
    model_records = config["models"]
    expected_models = [record["model_id"] for record in model_records]
    if frame["model_id"].drop_duplicates().tolist() != expected_models:
        raise ValueError("Unified calibration model order is invalid")
    if frame["model_id"].isna().any() or set(frame["model_id"].astype(str)) != set(expected_models):
        raise ValueError("Unified calibration model identities are invalid")
    input_contract = config["input_contract"]
    requested_bins = int(config["calibration_contract"].get("n_requested_bins", 10))
    if "n_requested_bins" not in config["calibration_contract"]:
        requested_bins = 10
    for record in model_records:
        subset = frame.loc[frame["model_id"] == record["model_id"]]
        if subset.empty:
            raise ValueError("Unified calibration model group is missing")
        positions = np.flatnonzero(frame["model_id"].to_numpy() == record["model_id"])
        if not np.array_equal(positions, np.arange(positions[0], positions[0] + len(positions))):
            raise ValueError("Unified calibration model group is not contiguous")
        if not (subset["model_order"] == record["model_order"]).all():
            raise ValueError("Unified calibration model order identity is invalid")
        if not (subset["model_display_name"] == record["model_display_name"]).all():
            raise ValueError("Unified calibration display identity is invalid")
        actual_bins = int(subset["n_actual_bins"].iloc[0])
        if not 1 <= actual_bins <= 10 or len(subset) != actual_bins:
            raise ValueError("Unified calibration actual-bin row count is invalid")
        if subset["bin_index"].astype(int).tolist() != list(range(actual_bins)):
            raise ValueError("Unified calibration bin order is invalid")
        if int(subset["n"].sum()) != input_contract["participant_count"]:
            raise ValueError("Unified calibration participant totals are invalid")
        if int(subset["events"].sum()) != input_contract["positive_count"]:
            raise ValueError("Unified calibration event totals are invalid")
        if int(subset["nonevents"].sum()) != input_contract["negative_count"]:
            raise ValueError("Unified calibration nonevent totals are invalid")
        if not (subset["n"] == subset["events"] + subset["nonevents"]).all():
            raise ValueError("Unified calibration bin counts do not reconcile")
        if not (subset["n"] > 0).all() or not (subset[["events", "nonevents"]] >= 0).all().all():
            raise ValueError("Unified calibration bin counts are invalid")
        if not (subset["n_requested_bins"] == requested_bins).all():
            raise ValueError("Unified calibration requested-bin identity is invalid")
        if not (subset["n_actual_bins"] == actual_bins).all():
            raise ValueError("Unified calibration actual-bin identities are invalid")
        if not (subset["visualization_ready"] == (actual_bins >= 5)).all():
            raise ValueError("Unified calibration readiness state is invalid")
        if not (subset["ci_method"] == "wilson").all():
            raise ValueError("Unified calibration confidence interval method is invalid")
        if not (subset["estimability_status"].astype(str) == "estimable").all():
            raise ValueError("Unified calibration estimability state is invalid")
        for row in subset.to_dict(orient="records"):
            n = int(row["n"])
            events = int(row["events"])
            values = (
                float(row["p_cal_mean"]),
                float(row["observed_fraction"]),
                float(row["observed_ci_low"]),
                float(row["observed_ci_high"]),
            )
            if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
                raise ValueError("Unified calibration probability or interval value is invalid")
            expected_fraction = events / n
            if values[1] != expected_fraction:
                raise ValueError("Unified calibration observed fraction is invalid")
            expected_low, expected_high = wilson_proportion_ci(events, n)
            if values[2] != expected_low or values[3] != expected_high:
                raise ValueError("Unified calibration Wilson interval is invalid")
            if values[2] > values[3]:
                raise ValueError("Unified calibration confidence interval ordering is invalid")


def _read_common_calibration_csv(anchor_path: Path) -> pd.DataFrame:
    """Read a synthetic or authorized anchor with exact common dtypes."""
    dtype_map: dict[str, str] = {
        "bin_index": "Int64",
        "n": "Int64",
        "events": "Int64",
        "nonevents": "Int64",
        "p_cal_mean": "Float64",
        "observed_fraction": "Float64",
        "observed_ci_low": "Float64",
        "observed_ci_high": "Float64",
        "ci_method": "string",
        "n_requested_bins": "Int64",
        "n_actual_bins": "Int64",
        "visualization_ready": "boolean",
        "nonestimable_reason": "string",
    }
    anchor = pd.read_csv(
        anchor_path,
        dtype=dtype_map,
        true_values=["true", "True"],
        false_values=["false", "False"],
        keep_default_na=True,
    )
    anchor["estimability_status"] = pd.Categorical(
        anchor["estimability_status"], categories=["estimable"], ordered=False
    )
    return anchor


def reconcile_minirocket_anchor(
    frame: pd.DataFrame,
    *,
    anchor_path: Path,
) -> OrderedDict[str, Any]:
    """Require exact equality with the MiniRocket common-column anchor."""
    if not isinstance(frame, pd.DataFrame) or not isinstance(anchor_path, Path):
        raise TypeError("MiniRocket reconciliation inputs are invalid")
    common_columns = frame.columns.tolist()[3:]
    calculated = frame.loc[frame["model_id"] == "minirocket_ridge", common_columns].reset_index(drop=True)
    try:
        anchor = _read_common_calibration_csv(anchor_path)
        pd.testing.assert_frame_equal(calculated, anchor, check_exact=True)
    except (AssertionError, OSError, ValueError, TypeError) as exc:
        raise ValueError("MiniRocket exact reconciliation failed") from exc
    return OrderedDict(
        (
            ("common_columns", common_columns),
            ("check_exact", True),
            ("status", "passed"),
        )
    )


def _serialize_csv(frame: pd.DataFrame, config: Mapping[str, Any]) -> bytes:
    """Serialize the authoritative aggregate with exact CSV settings."""
    serializable = frame.copy()
    serializable["visualization_ready"] = serializable["visualization_ready"].map(
        {True: "true", False: "false"}
    )
    stream = io.StringIO(newline="")
    serializable.to_csv(
        stream,
        index=False,
        sep=",",
        quotechar='"',
        doublequote=True,
        escapechar=None,
        lineterminator="\n",
        na_rep="",
        float_format="%.17g",
    )
    return stream.getvalue().encode("utf-8")


def _serialize_json(value: object) -> bytes:
    """Serialize one deterministic JSON object."""
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=False,
            indent=2,
            separators=(",", ": "),
        )
        + "\n"
    ).encode("utf-8")


def _model_states(frame: pd.DataFrame, config: Mapping[str, Any]) -> list[OrderedDict[str, Any]]:
    """Derive aggregate-only model states in frozen registry order."""
    states: list[OrderedDict[str, Any]] = []
    for record in config["models"]:
        subset = frame.loc[frame["model_id"] == record["model_id"]]
        observed = subset["observed_fraction"].astype(float).tolist()
        if len(observed) == 1:
            sequence_state = "not_assessable_single_bin"
        elif any(later < earlier for earlier, later in zip(observed, observed[1:], strict=False)):
            sequence_state = "nonmonotonic"
        else:
            sequence_state = "monotonic_nondecreasing"
        states.append(
            OrderedDict(
                (
                    ("model_order", record["model_order"]),
                    ("model_id", record["model_id"]),
                    ("model_display_name", record["model_display_name"]),
                    ("n_requested_bins", int(subset["n_requested_bins"].iloc[0])),
                    ("n_actual_bins", int(subset["n_actual_bins"].iloc[0])),
                    ("visualization_ready", bool(subset["visualization_ready"].iloc[0])),
                    ("estimability_status", "estimable"),
                    ("nonestimable_reason", None),
                    ("observed_fraction_sequence_state", sequence_state),
                    ("all_rows_retained", True),
                )
            )
        )
    return states


def _build_report(states: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> bytes:
    """Build the deterministic aggregate-only Markdown report."""
    lines = [
        "# Identity",
        "",
        f"Analysis: {config['analysis']['analysis_id']}",
        "",
        "# Authority and reconciliation",
        "",
        "Frozen calibrated probabilities; exact MiniRocket reconciliation required.",
        "",
        "# Model aggregate states",
        "",
        "| model_order | model_id | requested_bins | actual_bins | visualization_ready | estimability | sequence_state | all_rows_retained |",
        "|---:|---|---:|---:|---|---|---|---|",
    ]
    for state in states:
        lines.append(
            "| {model_order} | {model_id} | {n_requested_bins} | {n_actual_bins} | {ready} | "
            "{estimability_status} | {observed_fraction_sequence_state} | true |".format(
                **state, ready=str(state["visualization_ready"]).lower()
            )
        )
    lines.extend(
        [
            "",
            "# Result-retention statement",
            "",
            "All favorable, unfavorable, mixed, inconclusive, collapsed-bin, nonmonotonic, and not-ready states are retained.",
            "",
            "# Interpretation boundaries",
            "",
        ]
    )
    lines.extend(f"- {boundary}" for boundary in INTERPRETATION_BOUNDARIES)
    lines.append("")
    return "\n".join(lines).encode("utf-8")


def _validated_run_context(run_context: Mapping[str, Any]) -> OrderedDict[str, Any]:
    """Validate and copy the aggregate-only evidence used by the audit."""
    required = (
        "authority_checks",
        "input_hashes_before",
        "input_hashes_after",
        "environment_versions",
        "input_validation",
        "cohort_reconciliation",
        "minirocket_reconciliation",
        "privacy_validation",
    )
    _require_key_order(run_context, required, "Run context")
    authority_checks = run_context["authority_checks"]
    if not isinstance(authority_checks, Mapping) or tuple(authority_checks) != (
        "initial_entry",
        "before_run_a",
        "before_run_b",
        "after_both_runs",
    ):
        raise ValueError("Run context authority stages are incomplete")
    for evidence in authority_checks.values():
        if not isinstance(evidence, Mapping) or evidence.get("authority_status") != "passed":
            raise ValueError("Run context contains unsupported authority evidence")
        for status_key in (
            "relevant_diff_status",
            "git_status",
            "efigure9_status",
            "efigure4_status",
            "command_inventory_status",
            "python_path_status",
            "pycache_status",
            "prediction_hash_status",
            "immutable_hash_status",
            "import_closure_status",
            "import_origin_status",
            "stale_state_status",
        ):
            if evidence.get(status_key) != "passed":
                raise ValueError("Run context contains a failed dynamic authority gate")
        for boolean_key in (
            "ancestry_passed",
            "tag_ancestry_passed",
            "prediction_ancestry_passed",
            "anchor_ancestry_passed",
        ):
            if evidence.get(boolean_key) is not True:
                raise ValueError("Run context contains a failed ancestry gate")
    before = run_context["input_hashes_before"]
    after = run_context["input_hashes_after"]
    if not isinstance(before, list) or not before or not _recursive_equal(before, after):
        raise ValueError("Run context input hashes are absent or changed")
    versions = run_context["environment_versions"]
    required_versions = ("python", "pandas", "numpy", "statsmodels", "pytest")
    if not isinstance(versions, Mapping) or tuple(versions) != required_versions:
        raise ValueError("Run context environment versions are incomplete")
    if any(not isinstance(value, str) or not value or "bound" in value for value in versions.values()):
        raise ValueError("Run context environment versions contain placeholders")
    for key in ("input_validation", "cohort_reconciliation", "minirocket_reconciliation", "privacy_validation"):
        if not isinstance(run_context[key], Mapping):
            raise ValueError("Run context aggregate validation evidence is incomplete")
    input_validation = run_context["input_validation"]
    _require_key_order(
        input_validation,
        (
            "model_registry_status",
            "header_only_status",
            "projection_columns",
            "unused_participant_columns_loaded",
            "row_count_status",
            "identifier_lexeme_status",
            "probability_status",
            "label_status",
        ),
        "Run context input validation",
    )
    if (
        input_validation["projection_columns"] != ["sample_id", "y_true", "p_cal"]
        or input_validation["unused_participant_columns_loaded"] is not False
        or any(
            input_validation[key] != "passed"
            for key in (
                "model_registry_status",
                "header_only_status",
                "row_count_status",
                "identifier_lexeme_status",
                "probability_status",
                "label_status",
            )
        )
    ):
        raise ValueError("Run context input validation did not pass")
    minirocket = run_context["minirocket_reconciliation"]
    _require_key_order(
        minirocket,
        ("anchor_path", "anchor_sha256", "common_columns", "check_exact", "status"),
        "Run context MiniRocket reconciliation",
    )
    if (
        not isinstance(minirocket["anchor_path"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", minirocket["anchor_sha256"])
        or minirocket["common_columns"] != list(COMMON_COLUMNS)
        or minirocket["check_exact"] is not True
        or minirocket["status"] != "passed"
    ):
        raise ValueError("Run context MiniRocket reconciliation did not pass")
    if any(value != "passed" for value in run_context["privacy_validation"].values()):
        raise ValueError("Run context privacy validation did not pass")
    cohort = run_context["cohort_reconciliation"]
    if (
        cohort.get("models") != 6
        or cohort.get("participants_per_model") != 195
        or cohort.get("positive_labels") != 82
        or cohort.get("negative_labels") != 113
        or cohort.get("participant_set_agreement") != "passed"
        or cohort.get("cross_model_label_agreement") != "passed"
    ):
        raise ValueError("Run context cohort reconciliation did not pass")
    return _copy_ordered(run_context)


def build_artifact_payloads(
    frame: pd.DataFrame,
    run_context: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    deterministic_rerun_passed: bool,
) -> tuple[OrderedDict[str, object], OrderedDict[str, bytes]]:
    """Construct four deterministic aggregate-only artifact payloads."""
    validate_unified_calibration_points(frame, config)
    if not deterministic_rerun_passed:
        raise ValueError("Deterministic rerun equality is required")
    evidence = _validated_run_context(run_context)
    anchor_record = next(
        record for record in config["immutable_references"] if record["role"] == "reconciliation_anchor"
    )
    if (
        evidence["minirocket_reconciliation"]["anchor_path"] != anchor_record["path"]
        or evidence["minirocket_reconciliation"]["anchor_sha256"] != anchor_record["sha256"]
        or evidence["minirocket_reconciliation"]["common_columns"]
        != config["calibration_contract"]["common_columns"]
    ):
        raise ValueError("Run context MiniRocket reconciliation authority is inconsistent")
    states = _model_states(frame, config)
    csv_bytes = _serialize_csv(frame, config)
    audit = OrderedDict(
        (
            ("schema_version", 1),
            ("analysis_id", config["analysis"]["analysis_id"]),
            ("timing_classification", config["analysis"]["timing_classification"]),
            ("functional_role", config["analysis"]["functional_role"]),
            (
                "plan_authority",
                OrderedDict(
                    (
                        ("path", config["authority"]["approved_plan_path"]),
                        ("sha256", config["authority"]["approved_plan_sha256"]),
                        ("approval_status", "independently_reviewed"),
                    )
                ),
            ),
            (
                "authoring_state",
                OrderedDict(
                    (
                        ("date", config["analysis"]["phase1_authoring_date"]),
                        ("head", config["authority"]["phase1_authoring_head"]),
                        ("parent", "098d6d5d65d9fb935fa97a30d1353a9694a0dc3a"),
                        ("second_parent", "8c44846477c250a03380eda24a5c99ab63294426"),
                    )
                ),
            ),
            (
                "runtime_state",
                OrderedDict(
                    (
                        ("head", evidence["authority_checks"]["after_both_runs"]["runtime_head"]),
                        ("descends_from_authoring_head", evidence["authority_checks"]["after_both_runs"]["ancestry_passed"]),
                        ("relevant_diff_review_status", evidence["authority_checks"]["after_both_runs"]["relevant_diff_status"]),
                        ("command_inventory_status", evidence["authority_checks"]["after_both_runs"]["command_inventory_status"]),
                        ("runtime_git_argv_status", evidence["authority_checks"]["after_both_runs"]["git_status"]),
                        ("visible_status_stage", "prepromotion"),
                    )
                ),
            ),
            (
                "freeze_chain",
                OrderedDict(
                    (
                        ("tag_object", config["authority"]["modeling_tag_object"]),
                        ("tag_commit", config["authority"]["modeling_tag_commit"]),
                        ("prediction_source_commit", config["authority"]["prediction_source_commit"]),
                        ("anchor_source_commit", config["authority"]["anchor_source_commit"]),
                        ("tag_ancestry_passed", evidence["authority_checks"]["after_both_runs"]["tag_ancestry_passed"]),
                        ("prediction_ancestry_passed", evidence["authority_checks"]["after_both_runs"]["prediction_ancestry_passed"]),
                        ("anchor_ancestry_passed", evidence["authority_checks"]["after_both_runs"]["anchor_ancestry_passed"]),
                        ("runtime_ancestry_passed", evidence["authority_checks"]["after_both_runs"]["ancestry_passed"]),
                    )
                ),
            ),
            (
                "environment",
                OrderedDict(
                    (
                        ("dependency_authority", "environment.yml"),
                        ("conda_environment", "gcvocs"),
                        ("python_version", evidence["environment_versions"]["python"]),
                        ("pandas_version", evidence["environment_versions"]["pandas"]),
                        ("numpy_version", evidence["environment_versions"]["numpy"]),
                        ("statsmodels_version", evidence["environment_versions"]["statsmodels"]),
                        ("pytest_version", evidence["environment_versions"]["pytest"]),
                        ("python_path_gate", evidence["authority_checks"]["after_both_runs"]["python_path_status"]),
                        ("efigure4_environment_rejected", True),
                        ("package_mutation_performed", False),
                        ("python_dont_write_bytecode", "1"),
                        ("pytest_disable_plugin_autoload", "1"),
                        ("pytest_pycache_prefix", config["environment"]["pytest_pycache_prefix"]),
                        ("production_pycache_prefix", config["environment"]["production_pycache_prefix"]),
                        ("pycache_prefixes_remained_absent", evidence["authority_checks"]["after_both_runs"]["pycache_status"] == "passed"),
                        ("source_tree_bytecode_read_performed", False),
                    )
                ),
            ),
            ("input_hashes_before", evidence["input_hashes_before"]),
            (
                "input_validation",
                OrderedDict(
                    (
                        ("model_registry_status", evidence["input_validation"]["model_registry_status"]),
                        ("header_only_status", evidence["input_validation"]["header_only_status"]),
                        ("projection_columns", evidence["input_validation"]["projection_columns"]),
                        (
                            "unused_participant_columns_loaded",
                            evidence["input_validation"]["unused_participant_columns_loaded"],
                        ),
                        ("row_count_status", evidence["input_validation"]["row_count_status"]),
                        ("identifier_lexeme_status", evidence["input_validation"]["identifier_lexeme_status"]),
                        ("probability_status", evidence["input_validation"]["probability_status"]),
                        ("label_status", evidence["input_validation"]["label_status"]),
                    )
                ),
            ),
            (
                "cohort_reconciliation",
                OrderedDict(
                    (
                        ("models", evidence["cohort_reconciliation"]["models"]),
                        ("participants_per_model", evidence["cohort_reconciliation"]["participants_per_model"]),
                        ("positive_labels", evidence["cohort_reconciliation"]["positive_labels"]),
                        ("negative_labels", evidence["cohort_reconciliation"]["negative_labels"]),
                        ("participant_set_agreement", evidence["cohort_reconciliation"]["participant_set_agreement"]),
                        ("cross_model_label_agreement", evidence["cohort_reconciliation"]["cross_model_label_agreement"]),
                    )
                ),
            ),
            ("model_bin_states", states),
            (
                "minirocket_reconciliation",
                OrderedDict(
                    (
                        ("anchor_path", evidence["minirocket_reconciliation"]["anchor_path"]),
                        ("anchor_sha256", evidence["minirocket_reconciliation"]["anchor_sha256"]),
                        ("common_columns", evidence["minirocket_reconciliation"]["common_columns"]),
                        ("check_exact", evidence["minirocket_reconciliation"]["check_exact"]),
                        ("status", evidence["minirocket_reconciliation"]["status"]),
                    )
                ),
            ),
            (
                "source_immutability",
                OrderedDict(
                    (
                        ("method_reference_hashes_prepromotion", evidence["authority_checks"]["after_both_runs"]["immutable_hash_status"]),
                        ("import_closure_hashes_prepromotion", evidence["authority_checks"]["after_both_runs"]["import_closure_status"]),
                        ("prediction_hashes_prepromotion", evidence["authority_checks"]["after_both_runs"]["prediction_hash_status"]),
                        ("project_imports_resolved_to_source", evidence["authority_checks"]["after_both_runs"]["import_origin_status"] == "passed"),
                        ("source_tree_bytecode_reads", False),
                        ("efigure9_git_comparison", evidence["authority_checks"]["after_both_runs"]["efigure9_status"]),
                        ("efigure4_git_comparison", evidence["authority_checks"]["after_both_runs"]["efigure4_status"]),
                        ("all_unchanged", _recursive_equal(evidence["input_hashes_before"], evidence["input_hashes_after"])),
                    )
                ),
            ),
            (
                "privacy_validation",
                OrderedDict(
                    (
                        ("schema", evidence["privacy_validation"]["schema"]),
                        ("participant_identifiers", evidence["privacy_validation"]["participant_identifiers"]),
                        ("participant_predictions", evidence["privacy_validation"]["participant_predictions"]),
                        ("dates", evidence["privacy_validation"]["dates"]),
                        ("filenames", evidence["privacy_validation"]["filenames"]),
                        ("per_sample_paths", evidence["privacy_validation"]["per_sample_paths"]),
                        ("absolute_paths", evidence["privacy_validation"]["absolute_paths"]),
                        ("external_namespaces", evidence["privacy_validation"]["external_namespaces"]),
                        ("raw_value_matching", evidence["privacy_validation"]["raw_value_matching"]),
                        ("candidate_error_log_messages", evidence["privacy_validation"]["candidate_error_log_messages"]),
                        ("network_transmission", False),
                    )
                ),
            ),
            (
                "prepromotion_validation",
                OrderedDict(
                    (
                        ("scientific_payloads", "passed"),
                        ("serialization_contracts", "passed"),
                        ("candidate_manifest_inputs", "passed"),
                        ("candidate_write_readback_required", True),
                        ("atomic_rename_pending", True),
                    )
                ),
            ),
            (
                "deterministic_rerun",
                OrderedDict(
                    (
                        ("run_a_status", "passed"),
                        ("run_b_status", "passed"),
                        ("four_artifacts_byte_identical", True),
                        ("volatile_fields_present", False),
                    )
                ),
            ),
            (
                "interpretation_boundaries",
                list(INTERPRETATION_BOUNDARIES),
            ),
            ("artifact_status", "validated_for_atomic_promotion"),
        )
    )
    audit_bytes = _serialize_json(audit)
    report_bytes = _build_report(states, config)
    final_root = config["output_contract"]["final_root"]
    manifest_files = []
    for name, payload in (
        ("calibration_points.csv", csv_bytes),
        ("audit.json", audit_bytes),
        ("report.md", report_bytes),
    ):
        manifest_files.append(
            OrderedDict(
                (
                    ("path", f"{final_root}/{name}"),
                    ("byte_size", len(payload)),
                    ("sha256", hashlib.sha256(payload).hexdigest()),
                )
            )
        )
    manifest = OrderedDict(
        (("schema_version", 1), ("analysis_id", config["analysis"]["analysis_id"]), ("files", manifest_files))
    )
    structured: OrderedDict[str, object] = OrderedDict(
        (
            ("calibration_points.csv", frame.copy(deep=True)),
            ("audit.json", audit),
            ("manifest.json", manifest),
            ("report.md", report_bytes.decode("utf-8")),
        )
    )
    payloads: OrderedDict[str, bytes] = OrderedDict(
        (
            ("calibration_points.csv", csv_bytes),
            ("audit.json", audit_bytes),
            ("manifest.json", _serialize_json(manifest)),
            ("report.md", report_bytes),
        )
    )
    return structured, payloads


def _walk_structured(value: object, path: tuple[str, ...] = ()) -> Sequence[tuple[tuple[str, ...], object]]:
    """Return path-qualified scalar values for schema-aware privacy checks."""
    items: list[tuple[tuple[str, ...], object]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = path + (str(key),)
            items.append((child_path, child))
            items.extend(_walk_structured(child, child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            items.extend(_walk_structured(child, path + (str(index),)))
    return items


def _privacy_failure() -> ValueError:
    """Create a sanitized privacy exception."""
    return ValueError("Candidate privacy validation failed")


def validate_candidate_privacy(
    structured_payloads: Mapping[str, object],
    serialized_payloads: Mapping[str, bytes],
    log_and_error_candidates: Sequence[str],
    *,
    participant_references: Sequence[ParticipantIdentifierReferences],
    config: Mapping[str, Any],
) -> OrderedDict[str, str]:
    """Validate candidate structures and bytes without echoing private matches."""
    privacy = config["privacy_contract"]
    forbidden_fields = set(privacy["forbidden_field_names"])
    qualifiers = set(privacy["raw_identifier_field_qualifiers"])
    allowed_dates = {
        ("analysis", "phase1_authoring_date"): "2026-08-13",
        ("audit.json", "authoring_state", "date"): "2026-08-13",
    }
    for path, value in _walk_structured(structured_payloads):
        if path and path[-1] in forbidden_fields:
            raise _privacy_failure()
        if isinstance(value, str):
            date_match = re.fullmatch(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", value)
            if date_match and allowed_dates.get(path) != value:
                raise _privacy_failure()
            for refs in participant_references:
                if refs.matches_nonnumeric_token(value):
                    raise _privacy_failure()
                if path and path[-1] in qualifiers and refs.matches_identifier_qualified_numeric(path[-1], value):
                    raise _privacy_failure()
    byte_candidates = list(serialized_payloads.items()) + [
        ("log_or_error", item.encode("utf-8")) for item in log_and_error_candidates
    ]
    forbidden_fragments = privacy["forbidden_path_fragments"]
    text_forbidden_fields = forbidden_fields - {"p_cal"}
    field_pattern = re.compile(
        r"(?<![A-Za-z0-9_])(?:"
        + "|".join(re.escape(item) for item in text_forbidden_fields)
        + r")(?![A-Za-z0-9_])"
    )
    for role, payload in byte_candidates:
        text = payload.decode("utf-8", errors="strict")
        if any(fragment in text for fragment in forbidden_fragments):
            raise _privacy_failure()
        parsed_json: object | None = None
        if role.endswith(".json"):
            try:
                parsed_json = json.loads(text)
            except json.JSONDecodeError:
                parsed_json = None
        if parsed_json is not None:
            validate_candidate_privacy(
                {role: parsed_json},
                {},
                [],
                participant_references=participant_references,
                config=config,
            )
        elif role.endswith(".csv"):
            headers = text.splitlines()[0].split(",") if text.splitlines() else []
            if any(header in forbidden_fields for header in headers):
                raise _privacy_failure()
        elif field_pattern.search(text):
            raise _privacy_failure()
        if re.search(r"(?:^|[\s\"'])/|[A-Za-z]:\\|file://|wsl://|~/|/home/", text):
            raise _privacy_failure()
        if parsed_json is None and re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", text):
            raise _privacy_failure()
        for refs in participant_references:
            if refs.matches_serialized_text(text, tuple(qualifiers)):
                raise _privacy_failure()
            for token in re.findall(r"[A-Za-z0-9_+\-\u0080-\uffff]+", text):
                if refs.matches_nonnumeric_token(token):
                    raise _privacy_failure()
    return OrderedDict(
        (key, "passed")
        for key in (
            "schema",
            "participant_identifiers",
            "participant_predictions",
            "dates",
            "filenames",
            "per_sample_paths",
            "absolute_paths",
            "external_namespaces",
            "raw_value_matching",
            "candidate_error_log_messages",
        )
    )


def discard_participant_identifier_references(
    references: ParticipantIdentifierReferences,
) -> None:
    """Clear and discard all transient participant identifier tokens."""
    if not isinstance(references, ParticipantIdentifierReferences):
        raise TypeError("Participant references have the wrong type")
    references._clear()


def _resolve_new_project_path(path: Path, *, project_root: Path) -> Path:
    """Resolve a not-yet-existing path confined beneath the project root."""
    root = project_root.resolve(strict=True)
    candidate = path if path.is_absolute() else root / path
    if candidate.is_symlink():
        raise ValueError("Task path must not be a symbolic link")
    resolved = candidate.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise ValueError("Task path escapes the project root")
    return resolved


def _readback_calibration_csv(payload: bytes, config: Mapping[str, Any]) -> pd.DataFrame:
    """Parse, type, validate, and exactly reserialize the scientific CSV."""
    if not payload.endswith(b"\n") or b"\r" in payload:
        raise ValueError("Calibration CSV line-ending contract failed")
    try:
        frame = pd.read_csv(
            io.BytesIO(payload),
            encoding="utf-8",
            keep_default_na=False,
            na_filter=False,
            float_precision="round_trip",
        )
    except (UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ValueError("Calibration CSV read-back failed") from exc
    if frame.columns.tolist() != config["calibration_contract"]["unified_columns"]:
        raise ValueError("Calibration CSV column contract failed")
    frame["visualization_ready"] = frame["visualization_ready"].map(
        {"true": True, "false": False, True: True, False: False}
    )
    if frame["visualization_ready"].isna().any():
        raise ValueError("Calibration CSV Boolean contract failed")
    frame["nonestimable_reason"] = frame["nonestimable_reason"].replace("", pd.NA)
    try:
        frame = _coerce_unified_dtypes(frame, config)
    except (TypeError, ValueError) as exc:
        raise ValueError("Calibration CSV dtype contract failed") from exc
    validate_unified_calibration_points(frame, config)
    if _serialize_csv(frame, config) != payload:
        raise ValueError("Calibration CSV byte-round-trip contract failed")
    return frame


def _readback_ordered_json(payload: bytes, *, role: str) -> OrderedDict[str, Any]:
    """Decode deterministic JSON while retaining mapping key order."""
    if not payload.endswith(b"\n") or b"\r" in payload:
        raise ValueError(f"{role} JSON line-ending contract failed")
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=OrderedDict)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{role} JSON read-back failed") from exc
    if not isinstance(value, OrderedDict) or _serialize_json(value) != payload:
        raise ValueError(f"{role} JSON deterministic round-trip failed")
    return value


def validate_artifact_payloads(payloads: Mapping[str, bytes], config: Mapping[str, Any]) -> None:
    """Validate all four serialized artifacts and their cross-file reconciliation."""
    expected_names = config["output_contract"]["files"]
    if list(payloads) != expected_names or any(not isinstance(value, bytes) for value in payloads.values()):
        raise ValueError("Artifact payload inventory is invalid")
    _readback_calibration_csv(payloads["calibration_points.csv"], config)
    audit = _readback_ordered_json(payloads["audit.json"], role="Audit")
    _require_key_order(audit, config["output_contract"]["audit_key_order"], "Audit")
    nested_orders = {
        "plan_authority": ("path", "sha256", "approval_status"),
        "authoring_state": ("date", "head", "parent", "second_parent"),
        "runtime_state": (
            "head", "descends_from_authoring_head", "relevant_diff_review_status", "command_inventory_status",
            "runtime_git_argv_status", "visible_status_stage",
        ),
        "freeze_chain": (
            "tag_object", "tag_commit", "prediction_source_commit", "anchor_source_commit", "tag_ancestry_passed",
            "prediction_ancestry_passed", "anchor_ancestry_passed", "runtime_ancestry_passed",
        ),
        "environment": (
            "dependency_authority", "conda_environment", "python_version", "pandas_version", "numpy_version",
            "statsmodels_version", "pytest_version", "python_path_gate", "efigure4_environment_rejected",
            "package_mutation_performed", "python_dont_write_bytecode", "pytest_disable_plugin_autoload",
            "pytest_pycache_prefix", "production_pycache_prefix", "pycache_prefixes_remained_absent",
            "source_tree_bytecode_read_performed",
        ),
        "input_validation": (
            "model_registry_status", "header_only_status", "projection_columns", "unused_participant_columns_loaded",
            "row_count_status", "identifier_lexeme_status", "probability_status", "label_status",
        ),
        "cohort_reconciliation": (
            "models", "participants_per_model", "positive_labels", "negative_labels", "participant_set_agreement",
            "cross_model_label_agreement",
        ),
        "minirocket_reconciliation": ("anchor_path", "anchor_sha256", "common_columns", "check_exact", "status"),
        "source_immutability": (
            "method_reference_hashes_prepromotion", "import_closure_hashes_prepromotion",
            "prediction_hashes_prepromotion", "project_imports_resolved_to_source", "source_tree_bytecode_reads",
            "efigure9_git_comparison", "efigure4_git_comparison", "all_unchanged",
        ),
        "privacy_validation": (
            "schema", "participant_identifiers", "participant_predictions", "dates", "filenames", "per_sample_paths",
            "absolute_paths", "external_namespaces", "raw_value_matching", "candidate_error_log_messages",
            "network_transmission",
        ),
        "prepromotion_validation": (
            "scientific_payloads", "serialization_contracts", "candidate_manifest_inputs",
            "candidate_write_readback_required", "atomic_rename_pending",
        ),
        "deterministic_rerun": (
            "run_a_status", "run_b_status", "four_artifacts_byte_identical", "volatile_fields_present",
        ),
    }
    for key, order in nested_orders.items():
        _require_key_order(audit[key], order, f"Audit {key}")
    for record in audit["input_hashes_before"]:
        _require_key_order(record, ("path", "sha256", "status"), "Audit input hash record")
    model_state_order = (
        "model_order", "model_id", "model_display_name", "n_requested_bins", "n_actual_bins",
        "visualization_ready", "estimability_status", "nonestimable_reason",
        "observed_fraction_sequence_state", "all_rows_retained",
    )
    for record in audit["model_bin_states"]:
        _require_key_order(record, model_state_order, "Audit model state")
    if audit["artifact_status"] != "validated_for_atomic_promotion":
        raise ValueError("Audit artifact status is invalid")
    if audit["interpretation_boundaries"] != list(INTERPRETATION_BOUNDARIES):
        raise ValueError("Audit interpretation boundaries are incomplete")
    audit_text = payloads["audit.json"].decode("utf-8")
    forbidden_claims = ("PHASE_2_STATUS", "P01_STATUS", "promoted_read_back_complete")
    if any(claim in audit_text for claim in forbidden_claims):
        raise ValueError("Audit contains an unsupported final-state claim")
    manifest = _readback_ordered_json(payloads["manifest.json"], role="Manifest")
    _require_key_order(manifest, config["output_contract"]["manifest_key_order"], "Manifest")
    expected_manifest_names = ["calibration_points.csv", "audit.json", "report.md"]
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 3:
        raise ValueError("Manifest file inventory is invalid")
    final_root = config["output_contract"]["final_root"]
    for item, name in zip(files, expected_manifest_names, strict=True):
        _require_key_order(item, ("path", "byte_size", "sha256"), "Manifest file record")
        if item["path"] != f"{final_root}/{name}":
            raise ValueError("Manifest full path reconciliation failed")
        artifact = payloads[name]
        if item["byte_size"] != len(artifact) or item["sha256"] != hashlib.sha256(artifact).hexdigest():
            raise ValueError("Manifest size or hash reconciliation failed")
    report = payloads["report.md"]
    if not report.endswith(b"\n") or report.endswith(b"\n\n") or b"\r" in report:
        raise ValueError("Report line-ending contract failed")
    try:
        report_text = report.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Report UTF-8 contract failed") from exc
    headings = [line[2:] for line in report_text.splitlines() if line.startswith("# ")]
    if headings != config["output_contract"]["markdown_serialization"]["section_order"]:
        raise ValueError("Report section order is invalid")
    if any(f"- {boundary}" not in report_text for boundary in INTERPRETATION_BOUNDARIES):
        raise ValueError("Report interpretation boundaries are incomplete")


def stage_artifact_set(
    payloads: Mapping[str, bytes],
    config: Mapping[str, Any],
    *,
    project_root: Path,
) -> Path:
    """Stage the exact four-file candidate with destination-local temporaries."""
    expected_names = config["output_contract"]["files"]
    if list(payloads) != expected_names:
        raise ValueError("Candidate artifact inventory is invalid")
    work_root = _resolve_new_project_path(Path(config["output_contract"]["work_root"]), project_root=project_root)
    if work_root.exists():
        if work_root.is_symlink():
            raise FileExistsError("Scientific work root is not reusable")
        existing = sorted(path.name for path in work_root.iterdir())
        expected_existing = sorted([config["output_contract"]["run_a"], config["output_contract"]["run_b"]])
        if existing != expected_existing:
            raise FileExistsError("Scientific work root is not reusable")
    else:
        work_root.mkdir()
    validate_artifact_payloads(payloads, config)
    candidate = work_root / config["output_contract"]["promotion_candidate"]
    if candidate.exists() or candidate.is_symlink():
        raise FileExistsError("Promotion candidate already exists")
    candidate.mkdir()
    for name, payload in payloads.items():
        if not isinstance(payload, bytes):
            raise TypeError("Artifact payloads must be bytes")
        temporary = candidate / f".{name}.tmp"
        final = candidate / name
        try:
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, final)
            temporary.unlink()
        except FileExistsError as exc:
            raise FileExistsError("Candidate artifact target already exists") from exc
        if final.read_bytes() != payload:
            raise ValueError("Candidate artifact read-back failed")
    disk_payloads = OrderedDict((name, (candidate / name).read_bytes()) for name in expected_names)
    validate_artifact_payloads(disk_payloads, config)
    return candidate


def _validate_manifest(directory: Path, config: Mapping[str, Any]) -> None:
    """Validate the self-excluding manifest against three artifact files."""
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        expected_names = ["calibration_points.csv", "audit.json", "report.md"]
        expected_paths = [f"{config['output_contract']['final_root']}/{name}" for name in expected_names]
        if [item["path"] for item in manifest["files"]] != expected_paths:
            raise ValueError
        for item, name in zip(manifest["files"], expected_names, strict=True):
            payload = (directory / name).read_bytes()
            if item["byte_size"] != len(payload) or item["sha256"] != hashlib.sha256(payload).hexdigest():
                raise ValueError
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError) as exc:
        raise ValueError("Candidate manifest reconciliation failed") from exc


def promote_artifact_set(
    candidate_directory: Path,
    config: Mapping[str, Any],
    *,
    project_root: Path,
) -> tuple[Path, Path, Path, Path]:
    """Expose the complete candidate through one atomic directory rename."""
    candidate = _resolve_project_path(candidate_directory, project_root=project_root)
    final_root = _resolve_new_project_path(Path(config["output_contract"]["final_root"]), project_root=project_root)
    if final_root.exists() or final_root.is_symlink():
        raise FileExistsError("Final artifact namespace already exists")
    if candidate.stat().st_dev != final_root.parent.stat().st_dev:
        raise RuntimeError("Atomic promotion requires one filesystem")
    _atomic_rename_directory_noreplace(candidate, final_root)
    return tuple(final_root / name for name in config["output_contract"]["files"])  # type: ignore[return-value]


def _atomic_rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename one directory without permitting destination replacement."""
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("Atomic no-replace promotion is unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError("Final artifact namespace already exists")
        raise OSError(error_number, "Atomic no-replace promotion failed")


def validate_promoted_artifact_set(
    final_paths: Sequence[Path],
    config: Mapping[str, Any],
    *,
    project_root: Path,
    validated_candidate_payloads: Mapping[str, bytes],
) -> None:
    """Read back and reconcile an immutable promoted four-file package."""
    expected_names = config["output_contract"]["files"]
    if [path.name for path in final_paths] != expected_names:
        raise ValueError("Promoted artifact order is invalid")
    resolved = [_resolve_project_path(path, project_root=project_root) for path in final_paths]
    if any(path.is_symlink() or not path.is_file() for path in resolved):
        raise ValueError("Promoted artifacts must be regular nonsymlink files")
    directory = resolved[0].parent
    if sorted(path.name for path in directory.iterdir()) != sorted(expected_names):
        raise ValueError("Promoted artifact inventory is invalid")
    disk_payloads = OrderedDict((name, (directory / name).read_bytes()) for name in expected_names)
    if list(validated_candidate_payloads) != expected_names or any(
        disk_payloads[name] != validated_candidate_payloads[name] for name in expected_names
    ):
        raise ValueError("Promoted artifacts differ from the validated candidate bytes")
    validate_artifact_payloads(disk_payloads, config)


def cleanup_exact_task_directory(
    path: Path,
    *,
    project_root: Path,
    expected_direct_children: Sequence[str],
) -> None:
    """Remove one confined task directory after exact child validation."""
    resolved = _resolve_project_path(path, project_root=project_root)
    if resolved.is_symlink() or not resolved.is_dir():
        raise RuntimeError("Task cleanup root is invalid")
    if sorted(child.name for child in resolved.iterdir()) != sorted(expected_direct_children):
        raise RuntimeError("Task cleanup child inventory is invalid")
    children = list(resolved.iterdir())
    if any(child.is_symlink() or not child.is_dir() for child in children):
        raise RuntimeError("Task cleanup children must be nonsymlink directories")
    if any(any(child.iterdir()) for child in children):
        raise RuntimeError("Task cleanup run directories must be empty")
    for child_name in expected_direct_children:
        (resolved / child_name).rmdir()
    resolved.rmdir()
    if resolved.exists() or resolved.is_symlink():
        raise RuntimeError("Task cleanup did not remove the exact root")


def validate_runtime_absence_state(
    config: Mapping[str, Any],
    *,
    project_root: Path,
    require_work_root_absent: bool,
) -> OrderedDict[str, Any]:
    """Require only the scientific work root and production cache sentinel absent."""
    if not isinstance(config, Mapping):
        raise TypeError("Runtime absence validation requires a configuration mapping")
    relative_paths: list[str] = []
    if require_work_root_absent:
        relative_paths.append(config["output_contract"]["work_root"])
    relative_paths.append(config["output_contract"]["production_pycache_prefix_sentinel"])
    for relative_path in relative_paths:
        if not isinstance(relative_path, str) or not relative_path:
            raise RuntimeError("Runtime absence path authority is invalid")
        candidate = _resolve_new_project_path(Path(relative_path), project_root=project_root)
        if candidate.exists() or candidate.is_symlink():
            raise RuntimeError("Runtime required-absence integrity gate failed")
    return OrderedDict(
        (
            ("status", "passed"),
            ("paths", list(relative_paths)),
        )
    )


def _hash_record(path: Path) -> OrderedDict[str, str]:
    """Build one aggregate-safe project-relative hash record."""
    return OrderedDict((("path", path.as_posix()), ("sha256", _sha256_path(path)), ("status", "matched")))


def _run_authority_command(argv: Sequence[str], *, project_root: Path) -> subprocess.CompletedProcess[str]:
    """Run one exact read-only authority command."""
    return subprocess.run(list(argv), cwd=project_root, check=False, capture_output=True, text=True)


def _installed_versions() -> OrderedDict[str, str]:
    """Return actual installed versions without placeholders."""
    return OrderedDict(
        (
            ("python", ".".join(str(item) for item in sys.version_info[:3])),
            ("pandas", pd.__version__),
            ("numpy", np.__version__),
            ("statsmodels", importlib.metadata.version("statsmodels")),
            ("pytest", importlib.metadata.version("pytest")),
        )
    )


def _validate_runtime_environment(config: Mapping[str, Any], *, project_root: Path) -> OrderedDict[str, Any]:
    """Validate interpreter, environment flags, and the production cache sentinel."""
    expected_python = Path("[PRIVATE_PATH_REDACTED]").resolve(strict=True)
    if Path(sys.executable).resolve(strict=True) != expected_python:
        raise RuntimeError("Runtime interpreter gate failed")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != config["environment"]["python_dont_write_bytecode"]:
        raise RuntimeError("Runtime bytecode environment gate failed")
    prefix = os.environ.get("PYTHONPYCACHEPREFIX")
    allowed_prefixes = {
        config["environment"]["pytest_pycache_prefix"],
        config["environment"]["production_pycache_prefix"],
    }
    if prefix not in allowed_prefixes:
        raise RuntimeError("Runtime bytecode cache prefix gate failed")
    production_prefix = config["environment"]["production_pycache_prefix"]
    production_sentinel = config["output_contract"]["production_pycache_prefix_sentinel"]
    if (
        config["environment"]["production_pycache_prefix_must_remain_absent"] is not True
        or config["output_contract"]["production_pycache_prefix_must_remain_absent"] is not True
        or production_sentinel != production_prefix
    ):
        raise RuntimeError("Runtime production bytecode cache authority is inconsistent")
    candidate = _resolve_new_project_path(Path(production_sentinel), project_root=project_root)
    if candidate.exists() or candidate.is_symlink():
        raise RuntimeError("Runtime bytecode cache sentinel gate failed")
    return OrderedDict(
        (
            ("python_path_status", "passed"),
            ("pycache_status", "passed"),
            ("versions", _installed_versions()),
        )
    )


def _verify_read_closure(config: Mapping[str, Any], *, project_root: Path) -> list[OrderedDict[str, str]]:
    """Hash the exact production read closure in configured order."""
    records: list[OrderedDict[str, str]] = []
    for authority in config["authority"]["production_read_paths"]:
        relative = Path(authority["path"])
        resolved = _resolve_project_path(relative, project_root=project_root)
        digest = _sha256_path(resolved)
        expected = authority["sha256_or_binding"]
        if isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected) and digest != expected:
            raise RuntimeError("Production read-closure hash gate failed")
        records.append(OrderedDict((("path", relative.as_posix()), ("sha256", digest), ("status", "matched"))))
    if [record["path"] for record in records] != [
        record["path"] for record in config["authority"]["production_read_paths"]
    ]:
        raise RuntimeError("Production read-closure order gate failed")
    return records


def _module_name_for_source_path(relative: Path) -> str:
    """Return the canonical import name for one project Python source path."""
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _verify_import_origins(
    config: Mapping[str, Any], *, project_root: Path
) -> list[OrderedDict[str, str]]:
    """Require every configured runtime module to originate from its exact source."""
    source_roles = {
        "task_source",
        "task_cli",
        "import_closure",
        "incidental_import_only",
        "project_root_import",
        "frozen_builder",
        "frozen_builder_dependency",
        "score_provenance",
    }
    evidence: list[OrderedDict[str, str]] = []
    for record in config["authority"]["production_read_paths"]:
        relative = Path(record["path"])
        if (
            record["role"] not in source_roles
            or record["content_access_phase"] == "hash_only"
            or relative.suffix != ".py"
        ):
            continue
        expected = _resolve_project_path(relative, project_root=project_root)
        module_name = _module_name_for_source_path(relative)
        module = sys.modules.get(module_name)
        if module is None and record["role"] == "task_cli":
            main_module = sys.modules.get("__main__")
            main_origin = getattr(main_module, "__file__", None)
            if main_origin is not None and Path(main_origin).resolve(strict=True) == expected:
                module = main_module
                module_name = "__main__"
        if module is None:
            raise RuntimeError("Project import origin module is missing")
        origin_value = getattr(module, "__file__", None)
        if not isinstance(origin_value, str):
            raise RuntimeError("Project import origin is missing")
        try:
            origin = Path(origin_value).resolve(strict=True)
        except OSError as exc:
            raise RuntimeError("Project import origin gate failed") from exc
        if (
            origin != expected
            or origin.suffix != ".py"
            or "__pycache__" in origin.parts
            or origin_value.endswith((".pyc", ".pyo"))
        ):
            raise RuntimeError("Project import origin gate failed")
        evidence.append(
            OrderedDict(
                (("module", module_name), ("path", relative.as_posix()), ("origin", relative.as_posix()),
                 ("status", "matched"))
            )
        )
    return evidence


def _validate_g06_command(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Validate and return the exact path-scoped Git status argv."""
    if not isinstance(record, Mapping) or record.get("command_id") != "G06":
        raise ValueError("G06 authority record is missing or invalid")
    argv = record.get("argv_or_shell_lines")
    if record.get("execution_kind") != "argv" or not isinstance(argv, list):
        raise ValueError("G06 must use argv execution")
    if any(type(item) is not str for item in argv):
        raise ValueError("G06 argv must contain only strings")
    candidate = tuple(argv)
    prefix = candidate[:6]
    pathspecs = candidate[6:]
    if prefix != G06_ARGV[:6] or len(pathspecs) != len(G06_PATHS):
        raise ValueError("G06 command prefix or pathspec count is invalid")
    if any(
        not pathspec
        or pathspec == "."
        or Path(pathspec).is_absolute()
        or ".." in Path(pathspec).parts
        or any(character in pathspec for character in "*?[]")
        for pathspec in pathspecs
    ):
        raise ValueError("G06 contains a broad or unsafe pathspec")
    if pathspecs != G06_PATHS or candidate != G06_ARGV:
        raise ValueError("G06 must contain exactly the authorized eFigure 10 pathspecs")
    return candidate


def _validate_git_authority(
    config: Mapping[str, Any], *, project_root: Path, status_stage: str = "preproduction"
) -> OrderedDict[str, Any]:
    """Execute the exact runtime Git command inventory and validate outputs."""
    command_by_id = {record["command_id"]: record for record in config["authority"]["allowed_commands"]}
    try:
        g06 = command_by_id["G06"]
    except KeyError as exc:
        raise ValueError("G06 authority record is missing") from exc
    _validate_g06_command(g06)
    command_ids = [f"G{index:02d}" for index in range(1, 14)] + ["G18", "G19", "G20", "G21"]
    outputs: dict[str, list[str]] = {}
    for command_id in command_ids:
        record = command_by_id[command_id]
        argv = record["argv_or_shell_lines"]
        completed = _run_authority_command(argv, project_root=project_root)
        if completed.returncode != record["expected_status"]:
            raise RuntimeError("Runtime Git authority command failed")
        outputs[command_id] = completed.stdout.rstrip("\n").splitlines() if completed.stdout else []
    expected_scalars = {
        "G01": [project_root.as_posix()],
        "G02": ["main"],
        "G03": [config["authority"]["phase2_starting_head"]],
        "G04": ["098d6d5d65d9fb935fa97a30d1353a9694a0dc3a"],
        "G05": ["8c44846477c250a03380eda24a5c99ab63294426"],
        "G07": [config["authority"]["modeling_tag_object"]],
        "G08": [config["authority"]["modeling_tag_commit"]],
        "G09": ["tag"],
        "G10": [],
        "G11": [],
        "G12": [],
        "G13": [],
        "G18": [],
        "G19": [f".gitignore:17:/docs/plans/\t{config['authority']['approved_plan_path']}"],
        "G20": [],
        "G21": [],
    }
    for command_id, expected in expected_scalars.items():
        if outputs[command_id] != expected:
            raise RuntimeError("Runtime Git authority output gate failed")
    expected_status = command_by_id["G06"]["postconditions"]["assertions"][0]["expected"][status_stage]
    if outputs["G06"] != expected_status:
        raise RuntimeError("Runtime visible-status gate failed")
    return OrderedDict(
        (
            ("runtime_head", outputs["G03"][0]),
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


def validate_runtime_authority(
    config: Mapping[str, Any],
    *,
    project_root: Path,
    allow_work_root: bool = False,
    status_stage: str = "preproduction",
    allow_final_root: bool = False,
) -> OrderedDict[str, Any]:
    """Validate closed runtime authority before any participant-row read."""
    if not isinstance(config, Mapping):
        raise TypeError("Runtime authority requires a configuration mapping")
    root = project_root.resolve(strict=True)
    work_root = _resolve_new_project_path(Path(config["output_contract"]["work_root"]), project_root=root)
    final_root = _resolve_new_project_path(Path(config["output_contract"]["final_root"]), project_root=root)
    validate_runtime_absence_state(
        config,
        project_root=root,
        require_work_root_absent=not allow_work_root,
    )
    if allow_final_root:
        if final_root.is_symlink() or not final_root.is_dir():
            raise RuntimeError("Runtime final-namespace integrity gate failed")
        if sorted(path.name for path in final_root.iterdir()) != sorted(config["output_contract"]["files"]):
            raise RuntimeError("Runtime final artifact inventory gate failed")
    elif final_root.exists() or final_root.is_symlink():
        raise RuntimeError("Runtime final-namespace stale-state gate failed")
    if not allow_work_root and (work_root.exists() or work_root.is_symlink()):
        raise RuntimeError("Runtime work-namespace stale-state gate failed")
    if allow_work_root:
        if work_root.is_symlink() or not work_root.is_dir():
            raise RuntimeError("Runtime work-namespace gate failed")
        allowed_children = {config["output_contract"]["run_a"], config["output_contract"]["run_b"]}
        if {child.name for child in work_root.iterdir()} != allowed_children:
            raise RuntimeError("Runtime work-namespace child gate failed")
    canonical_config = _resolve_project_path(TASK_CONFIG_PATH, project_root=root)
    if _sha256_path(canonical_config) != TASK_CONFIG_SHA256:
        raise RuntimeError("Canonical task configuration hash gate failed")
    if not _recursive_equal(load_config(TASK_CONFIG_PATH, project_root=root), config):
        raise RuntimeError("Runtime configuration differs from canonical authority")
    plan_path = Path(config["authority"]["approved_plan_path"])
    resolved_plan = _resolve_project_path(plan_path, project_root=root)
    if _sha256_path(resolved_plan) != config["authority"]["approved_plan_sha256"]:
        raise RuntimeError("Approved plan hash gate failed")
    git_evidence = _validate_git_authority(config, project_root=root, status_stage=status_stage)
    environment_evidence = _validate_runtime_environment(config, project_root=root)
    input_hashes = _verify_read_closure(config, project_root=root)
    prediction_paths = {record["prediction_path"] for record in config["models"]}
    hashed_predictions = {record["path"] for record in input_hashes if record["path"] in prediction_paths}
    if hashed_predictions != prediction_paths:
        raise RuntimeError("Prediction hash inventory gate failed")
    for record in config["models"]:
        actual = next(item["sha256"] for item in input_hashes if item["path"] == record["prediction_path"])
        if actual != record["sha256"]:
            raise RuntimeError("Prediction hash gate failed")
    for reference in config["immutable_references"]:
        actual = next(item["sha256"] for item in input_hashes if item["path"] == reference["path"])
        if actual != reference["sha256"]:
            raise RuntimeError("Immutable reference hash gate failed")
    import_origin_evidence = _verify_import_origins(config, project_root=root)
    return OrderedDict(
        (
            ("authority_status", "passed"),
            *git_evidence.items(),
            ("python_path_status", environment_evidence["python_path_status"]),
            ("pycache_status", environment_evidence["pycache_status"]),
            ("environment_versions", environment_evidence["versions"]),
            ("input_hashes", input_hashes),
            ("prediction_hash_status", "passed"),
            ("immutable_hash_status", "passed"),
            ("import_closure_status", "passed" if input_hashes else "failed"),
            ("import_origin_status", "passed" if all(
                item["status"] == "matched" for item in import_origin_evidence
            ) else "failed"),
            ("import_origin_evidence", import_origin_evidence),
            ("stale_state_status", "passed"),
        )
    )


def execute_isolated_run(
    config: Mapping[str, Any],
    *,
    builder_config: Mapping[str, Any],
    project_root: Path,
    run_directory: Path,
    authority_stage: str = "isolated_run",
) -> tuple[pd.DataFrame, OrderedDict[str, Any], ParticipantIdentifierReferences]:
    """Execute one in-memory aggregate run without writing scientific files."""
    run_path = _resolve_project_path(run_directory, project_root=project_root)
    if any(run_path.iterdir()):
        raise FileExistsError("Isolated run directory must be empty")
    authority = validate_runtime_authority(config, project_root=project_root, allow_work_root=True)
    predictions, references, validation_evidence = _load_and_reconcile_predictions_with_evidence(
        config,
        project_root=project_root,
    )
    frame = build_unified_calibration_points(predictions, config, builder_config=builder_config)
    anchor_record = next(item for item in config["immutable_references"] if item["role"] == "reconciliation_anchor")
    minirocket_result = reconcile_minirocket_anchor(frame, anchor_path=project_root / anchor_record["path"])
    minirocket_evidence: OrderedDict[str, Any] = OrderedDict(
        (
            ("anchor_path", anchor_record["path"]),
            ("anchor_sha256", anchor_record["sha256"]),
            ("common_columns", minirocket_result["common_columns"]),
            ("check_exact", minirocket_result["check_exact"]),
            ("status", minirocket_result["status"]),
        )
    )
    context = OrderedDict(
        (
            ("authority_stage", authority_stage),
            ("authority", authority),
            ("input_validation", validation_evidence["input_validation"]),
            ("cohort_reconciliation", validation_evidence["cohort_reconciliation"]),
            ("minirocket_reconciliation", minirocket_evidence),
            ("model_bin_states", _model_states(frame, config)),
        )
    )
    del predictions
    return frame, context, references


def run_analysis(
    config_path: Path,
    *,
    project_root: Path = PROJECT_ROOT,
) -> tuple[Path, Path, Path, Path]:
    """Orchestrate two deterministic runs and atomic four-file promotion."""
    config = load_config(config_path, project_root=project_root)
    if os.environ.get("PYTHONPYCACHEPREFIX") != config["environment"]["production_pycache_prefix"]:
        raise RuntimeError("Artifact production bytecode cache prefix gate failed")
    initial_authority = validate_runtime_authority(config, project_root=project_root)
    frozen = load_frozen_reporting_config(FROZEN_REPORTING_CONFIG_PATH, project_root=project_root)
    builder_config = build_frozen_calibration_builder_config(config["calibration_contract"], frozen_reporting_config=frozen)
    work_root = _resolve_new_project_path(Path(config["output_contract"]["work_root"]), project_root=project_root)
    final_root = _resolve_new_project_path(Path(config["output_contract"]["final_root"]), project_root=project_root)
    if work_root.exists() or work_root.is_symlink():
        raise FileExistsError("Scientific work root already exists")
    if final_root.exists() or final_root.is_symlink():
        raise FileExistsError("Final artifact namespace already exists")
    work_root.mkdir()
    run_a = work_root / config["output_contract"]["run_a"]
    run_b = work_root / config["output_contract"]["run_b"]
    run_a.mkdir()
    run_b.mkdir()
    try:
        frame_a, context_a, refs_a = execute_isolated_run(
            config,
            builder_config=builder_config,
            project_root=project_root,
            run_directory=run_a,
            authority_stage="before_run_a",
        )
        frame_b, context_b, refs_b = execute_isolated_run(
            config,
            builder_config=builder_config,
            project_root=project_root,
            run_directory=run_b,
            authority_stage="before_run_b",
        )
        pd.testing.assert_frame_equal(frame_a, frame_b, check_exact=True)
        aggregate_context_a = OrderedDict((key, value) for key, value in context_a.items() if key not in {"authority_stage", "authority"})
        aggregate_context_b = OrderedDict((key, value) for key, value in context_b.items() if key not in {"authority_stage", "authority"})
        if not _recursive_equal(aggregate_context_a, aggregate_context_b):
            raise ValueError("Isolated aggregate contexts differ")
        after_authority = validate_runtime_authority(config, project_root=project_root, allow_work_root=True)
        if not _recursive_equal(initial_authority["input_hashes"], after_authority["input_hashes"]):
            raise RuntimeError("Runtime input hashes changed across isolated runs")
        privacy_passed = validate_candidate_privacy(
            OrderedDict((("calibration_points.csv", frame_a),)),
            OrderedDict((("calibration_points.csv", _serialize_csv(frame_a, config)),)),
            [],
            participant_references=[refs_a, refs_b],
            config=config,
        )
        shared_context = OrderedDict(
            (
                (
                    "authority_checks",
                    OrderedDict(
                        (
                            ("initial_entry", initial_authority),
                            ("before_run_a", context_a["authority"]),
                            ("before_run_b", context_b["authority"]),
                            ("after_both_runs", after_authority),
                        )
                    ),
                ),
                ("input_hashes_before", initial_authority["input_hashes"]),
                ("input_hashes_after", after_authority["input_hashes"]),
                ("environment_versions", after_authority["environment_versions"]),
                ("input_validation", context_a["input_validation"]),
                ("cohort_reconciliation", context_a["cohort_reconciliation"]),
                ("minirocket_reconciliation", context_a["minirocket_reconciliation"]),
                ("privacy_validation", privacy_passed),
            )
        )
        structured_a, payloads_a = build_artifact_payloads(frame_a, shared_context, config, deterministic_rerun_passed=True)
        structured_b, payloads_b = build_artifact_payloads(frame_b, shared_context, config, deterministic_rerun_passed=True)
        if payloads_a != payloads_b or not _recursive_equal(structured_a, structured_b):
            raise ValueError("Isolated artifact payloads differ")
        final_privacy_evidence = validate_candidate_privacy(
            structured_a,
            payloads_a,
            [],
            participant_references=[refs_a, refs_b],
            config=config,
        )
        if final_privacy_evidence != privacy_passed:
            raise RuntimeError("Candidate privacy evidence is inconsistent")
        discard_participant_identifier_references(refs_a)
        discard_participant_identifier_references(refs_b)
        if not refs_a.is_cleared() or not refs_b.is_cleared():
            raise RuntimeError("Participant references were not cleared before staging")
        candidate = stage_artifact_set(payloads_a, config, project_root=project_root)
        final_paths = promote_artifact_set(candidate, config, project_root=project_root)
        validate_promoted_artifact_set(
            final_paths,
            config,
            project_root=project_root,
            validated_candidate_payloads=payloads_a,
        )
        postpromotion_authority = validate_runtime_authority(
            config,
            project_root=project_root,
            allow_work_root=True,
            status_stage="final",
            allow_final_root=True,
        )
        if not _recursive_equal(initial_authority["input_hashes"], postpromotion_authority["input_hashes"]):
            raise RuntimeError("Post-promotion runtime input hashes changed")
        cleanup_exact_task_directory(
            work_root,
            project_root=project_root,
            expected_direct_children=[config["output_contract"]["run_a"], config["output_contract"]["run_b"]],
        )
        validate_runtime_absence_state(
            config,
            project_root=project_root,
            require_work_root_absent=True,
        )
        return final_paths
    except Exception:
        raise
