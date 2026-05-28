"""Post-hoc calibration goodness-of-fit tests for Plan 1 Session 4.

This module implements Plan 1 §3.6 steps 1-4 for the frozen best model
`minirocket_ridge`. It validates the locked test-prediction artifact against the
post-hoc alignment contract, then computes the descriptive Hosmer-Lemeshow and
Spiegelhalter calibration tests without modifying any frozen primary artifact.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from src.analysis.sample_size import (
    FROZEN_CALIBRATION_INTERCEPT,
    FROZEN_CALIBRATION_SLOPE,
    FROZEN_POST_PLATT_BRIER,
    POST_HOC_DECLARATION,
    load_post_hoc_config,
)
from src.analysis.stage7_part_a_common import (
    _format_path,
    _get_git_commit_hash,
    _resolve_path,
    _timestamp_utc,
    _write_text,
)
from src.config import PROJECT_ROOT, RANDOM_SEED


LOGGER = logging.getLogger(__name__)

EXPECTED_TEST_COLUMNS = ["sample_id", "y_true", "p_raw", "p_cal", "y_pred", "quadrant"]
SAMPLE_INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "sample_index.csv"
TEST_PREDICTION_PATH = PROJECT_ROOT / "outputs" / "predictions" / "binary_minirocket_ridge_test_predictions.csv"
HOSMER_LEMESHOW_GROUPS = 10  # [ml-methodology-literature: standard HL decile grouping]
HOSMER_LEMESHOW_DF = 8  # [ml-methodology-literature: G - 2 with G = 10]


@dataclass(frozen=True)
class CalibrationTestResult:
    """Structured result for Plan 1 §3.6 steps 2-4.

    Attributes:
        hosmer_lemeshow_statistic: HL grouped chi-square statistic.
        hosmer_lemeshow_df: HL degrees of freedom fixed by the decile design.
        hosmer_lemeshow_p_value: Survival-function p-value for the HL statistic.
        spiegelhalter_z: Spiegelhalter calibration z-statistic.
        spiegelhalter_p_value: Two-sided p-value for the Spiegelhalter z-statistic.
        observed_brier: Observed Brier score from the locked `p_cal`.
        expected_brier: Expected Brier score under calibration.
        variance: Spiegelhalter variance term.
    """

    hosmer_lemeshow_statistic: float
    hosmer_lemeshow_df: int
    hosmer_lemeshow_p_value: float
    spiegelhalter_z: float
    spiegelhalter_p_value: float
    observed_brier: float
    expected_brier: float
    variance: float


def _read_csv(path: Path) -> pd.DataFrame:
    """Read one CSV from disk.

    Args:
        path: CSV path to load.

    Returns:
        Loaded dataframe.

    Raises:
        OSError: If the CSV cannot be read.
    """
    try:
        return pd.read_csv(path)
    except OSError as exc:
        LOGGER.error("Failed to read CSV from %s.", path)
        raise OSError(f"Failed to read CSV from {path}.") from exc


def _log_frame_details(frame: pd.DataFrame, path: Path) -> None:
    """Log shape, dtypes, and missing-value counts for one dataframe."""
    LOGGER.info(
        "Loaded %s with shape=%s, dtypes=%s, missing=%s.",
        _format_path(path),
        frame.shape,
        frame.dtypes.astype(str).to_dict(),
        frame.isna().sum().to_dict(),
    )


def _load_sample_index() -> pd.DataFrame:
    """Load the final-cohort sample index for the Plan 1 §3.6 step-1 checks."""
    sample_index = _read_csv(SAMPLE_INDEX_PATH)
    _log_frame_details(sample_index, SAMPLE_INDEX_PATH)

    final_sample_index = sample_index.loc[sample_index["in_final_output"].fillna(False)].copy()
    final_sample_index["sample_id"] = pd.to_numeric(final_sample_index["sample_id"], errors="raise").astype(np.int64)
    final_sample_index["label"] = pd.to_numeric(final_sample_index["label"], errors="raise").astype(np.int64)
    final_sample_index.sort_values("sample_id", kind="mergesort", inplace=True)
    final_sample_index.reset_index(drop=True, inplace=True)
    return final_sample_index.loc[:, ["sample_id", "label"]]


def _validate_test_prediction_frame(frame: pd.DataFrame, sample_index: pd.DataFrame) -> pd.DataFrame:
    """Validate the locked test-prediction CSV against the post-hoc contract.

    Args:
        frame: Raw prediction dataframe.
        sample_index: Final-cohort `sample_index.csv` subset with labels.

    Returns:
        Validated dataframe with numeric coercions applied.

    Raises:
        AssertionError: If column order, sample-id ordering, subset membership, or
            label alignment violate the Plan 1 §3.6 step-1 contract.
        ValueError: If score columns contain invalid values or `p_cal` leaves
            the unit interval.
    """
    if frame.columns.tolist() != EXPECTED_TEST_COLUMNS:
        raise AssertionError(
            f"{_format_path(TEST_PREDICTION_PATH)} columns must be {EXPECTED_TEST_COLUMNS}, "
            f"found {frame.columns.tolist()}."
        )

    validated = frame.copy()
    validated["sample_id"] = pd.to_numeric(validated["sample_id"], errors="raise").astype(np.int64)
    validated["y_true"] = pd.to_numeric(validated["y_true"], errors="raise").astype(np.int64)
    validated["p_raw"] = pd.to_numeric(validated["p_raw"], errors="raise").astype(float)
    validated["p_cal"] = pd.to_numeric(validated["p_cal"], errors="raise").astype(float)
    validated["y_pred"] = pd.to_numeric(validated["y_pred"], errors="raise").astype(np.int64)

    assert validated["sample_id"].is_monotonic_increasing, (
        f"{_format_path(TEST_PREDICTION_PATH)} sample_id must be monotonically increasing per Erratum E007."
    )
    assert validated["sample_id"].is_unique, f"{_format_path(TEST_PREDICTION_PATH)} sample_id values must be unique."
    assert set(validated["sample_id"].tolist()) < set(sample_index["sample_id"].tolist()), (
        f"{_format_path(TEST_PREDICTION_PATH)} sample_id values must be a strict subset of "
        "sample_index.csv[in_final_output == True]."
    )

    if not np.isfinite(validated["p_raw"].to_numpy(dtype=float)).all():
        raise ValueError(f"{_format_path(TEST_PREDICTION_PATH)} p_raw contains NaN or Inf.")
    if not np.isfinite(validated["p_cal"].to_numpy(dtype=float)).all():
        raise ValueError(f"{_format_path(TEST_PREDICTION_PATH)} p_cal contains NaN or Inf.")
    if ((validated["p_cal"] < 0.0) | (validated["p_cal"] > 1.0)).any():
        raise ValueError(f"{_format_path(TEST_PREDICTION_PATH)} p_cal contains values outside [0, 1].")

    merged = validated.merge(sample_index, on="sample_id", how="left", validate="one_to_one")
    assert not merged["label"].isna().any(), (
        f"{_format_path(TEST_PREDICTION_PATH)} contains sample_ids missing from sample_index.csv."
    )
    assert np.array_equal(
        merged["y_true"].to_numpy(dtype=np.int64),
        merged["label"].to_numpy(dtype=np.int64),
    ), f"{_format_path(TEST_PREDICTION_PATH)} y_true does not match sample_index.csv labels."
    return validated


def _validate_calibration_arrays(y_true: np.ndarray, p_cal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate the one-dimensional inputs used by Plan 1 §3.6 steps 2-3."""
    y_array = np.asarray(y_true, dtype=np.int64)
    p_array = np.asarray(p_cal, dtype=float)
    if y_array.ndim != 1 or p_array.ndim != 1:
        raise ValueError("y_true and p_cal must be one-dimensional.")
    if len(y_array) != len(p_array):
        raise ValueError("y_true and p_cal must have the same length.")
    if len(y_array) == 0:
        raise ValueError("Calibration tests require at least one observation.")
    if not np.isfinite(p_array).all():
        raise ValueError("p_cal contains NaN or Inf.")
    if np.any((p_array < 0.0) | (p_array > 1.0)):
        raise ValueError("p_cal contains values outside [0, 1].")
    unique_labels = set(np.unique(y_array).tolist())
    if not unique_labels <= {0, 1}:
        raise ValueError(f"y_true must be binary with values in {{0, 1}}, found {sorted(unique_labels)}.")
    return y_array, p_array


def compute_hosmer_lemeshow_test(
    *,
    y_true: np.ndarray,
    p_cal: np.ndarray,
    hosmer_lemeshow_groups: int,
) -> tuple[float, int, float]:
    """Compute the Plan 1 §3.6 step-2 Hosmer-Lemeshow decile test.

    Args:
        y_true: Binary labels aligned to `p_cal`.
        p_cal: Frozen Platt-calibrated probabilities.
        hosmer_lemeshow_groups: Required number of HL decile groups.

    Returns:
        Tuple of `(statistic, degrees_of_freedom, p_value)`.

    Raises:
        ValueError: If the prespecified structural halts fire, including fewer
            than 10 groups, zero expected counts, or non-finite p-values.
    """
    if hosmer_lemeshow_groups != HOSMER_LEMESHOW_GROUPS:
        raise ValueError(
            f"hosmer_lemeshow_groups must equal {HOSMER_LEMESHOW_GROUPS} per Plan 1 §3.6, "
            f"received {hosmer_lemeshow_groups}."
        )

    y_array, p_array = _validate_calibration_arrays(y_true=y_true, p_cal=p_cal)
    working_frame = pd.DataFrame({"y_true": y_array, "p_cal": p_array}).sort_values("p_cal", kind="mergesort")
    working_frame.reset_index(drop=True, inplace=True)
    working_frame["hl_group"] = pd.qcut(working_frame["p_cal"], q=hosmer_lemeshow_groups, duplicates="drop")

    observed_groups = int(working_frame["hl_group"].nunique(dropna=True))
    if observed_groups != hosmer_lemeshow_groups:
        raise ValueError(
            "Plan 1 §3.6 halt: Hosmer-Lemeshow decile grouping yielded fewer than 10 groups after qcut duplicate "
            f"dropping (observed {observed_groups})."
        )

    statistic = 0.0
    for _, group_frame in working_frame.groupby("hl_group", observed=False):
        observed_events = float(group_frame["y_true"].sum())
        expected_events = float(group_frame["p_cal"].sum())
        observed_nonevents = float(len(group_frame) - observed_events)
        expected_nonevents = float((1.0 - group_frame["p_cal"]).sum())
        if expected_events <= 0.0 or expected_nonevents <= 0.0:
            raise ValueError("Plan 1 §3.6 halt: expected group count is zero in the Hosmer-Lemeshow table.")
        statistic += ((observed_events - expected_events) ** 2) / expected_events
        statistic += ((observed_nonevents - expected_nonevents) ** 2) / expected_nonevents

    degrees_of_freedom = HOSMER_LEMESHOW_DF
    p_value = float(chi2.sf(statistic, degrees_of_freedom))
    if not np.isfinite(p_value):
        raise ValueError("Plan 1 §3.6 halt: Hosmer-Lemeshow p-value is NaN or Inf.")
    return float(statistic), degrees_of_freedom, p_value


def compute_spiegelhalter_z_test(
    *,
    y_true: np.ndarray,
    p_cal: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Compute the Plan 1 §3.6 step-3 Spiegelhalter z test.

    Args:
        y_true: Binary labels aligned to `p_cal`.
        p_cal: Frozen Platt-calibrated probabilities.

    Returns:
        Tuple of `(observed_brier, expected_brier, variance, z_statistic, p_value)`.

    Raises:
        ValueError: If the variance is non-positive or any derived value is
            non-finite.
    """
    y_array, p_array = _validate_calibration_arrays(y_true=y_true, p_cal=p_cal)
    observed_brier = float(np.mean((y_array - p_array) ** 2))
    expected_brier = float(np.mean(p_array * (1.0 - p_array)))
    variance = float(np.sum(p_array * (1.0 - p_array) * ((1.0 - (2.0 * p_array)) ** 2)) / (len(p_array) ** 2))
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError("Plan 1 §3.6 halt: Spiegelhalter variance must be positive and finite.")

    z_statistic = float((observed_brier - expected_brier) / np.sqrt(variance))
    p_value = float(2.0 * norm.sf(abs(z_statistic)))
    if not np.isfinite(z_statistic) or not np.isfinite(p_value):
        raise ValueError("Plan 1 §3.6 halt: Spiegelhalter z-statistic or p-value is NaN or Inf.")
    return observed_brier, expected_brier, variance, z_statistic, p_value


def run_calibration_tests_on_frame(
    *,
    frame: pd.DataFrame,
    hosmer_lemeshow_groups: int,
) -> CalibrationTestResult:
    """Run Plan 1 §3.6 steps 1-4 on one in-memory prediction frame.

    Args:
        frame: Test-prediction dataframe with the locked six-column schema.
        hosmer_lemeshow_groups: Required HL group count from the Plan 1 config.

    Returns:
        Structured scalar result with both descriptive calibration tests.
    """
    if frame.columns.tolist() != EXPECTED_TEST_COLUMNS:
        raise AssertionError(f"In-memory frame columns must be {EXPECTED_TEST_COLUMNS}, found {frame.columns.tolist()}.")

    validated = frame.copy()
    validated["sample_id"] = pd.to_numeric(validated["sample_id"], errors="raise").astype(np.int64)
    validated["y_true"] = pd.to_numeric(validated["y_true"], errors="raise").astype(np.int64)
    validated["p_raw"] = pd.to_numeric(validated["p_raw"], errors="raise").astype(float)
    validated["p_cal"] = pd.to_numeric(validated["p_cal"], errors="raise").astype(float)
    validated["y_pred"] = pd.to_numeric(validated["y_pred"], errors="raise").astype(np.int64)
    if ((validated["p_cal"] < 0.0) | (validated["p_cal"] > 1.0)).any():
        raise ValueError("p_cal contains values outside [0, 1].")

    hosmer_lemeshow_statistic, hosmer_lemeshow_df, hosmer_lemeshow_p_value = compute_hosmer_lemeshow_test(
        y_true=validated["y_true"].to_numpy(dtype=np.int64),
        p_cal=validated["p_cal"].to_numpy(dtype=float),
        hosmer_lemeshow_groups=hosmer_lemeshow_groups,
    )
    observed_brier, expected_brier, variance, spiegelhalter_z, spiegelhalter_p_value = compute_spiegelhalter_z_test(
        y_true=validated["y_true"].to_numpy(dtype=np.int64),
        p_cal=validated["p_cal"].to_numpy(dtype=float),
    )
    return CalibrationTestResult(
        hosmer_lemeshow_statistic=hosmer_lemeshow_statistic,
        hosmer_lemeshow_df=hosmer_lemeshow_df,
        hosmer_lemeshow_p_value=hosmer_lemeshow_p_value,
        spiegelhalter_z=spiegelhalter_z,
        spiegelhalter_p_value=spiegelhalter_p_value,
        observed_brier=observed_brier,
        expected_brier=expected_brier,
        variance=variance,
    )


def analyze_calibration_tests(config_path: Path) -> CalibrationTestResult:
    """Run the full read-only Plan 1 §3.6 analysis on the locked test predictions.

    Args:
        config_path: Path to `configs/post_hoc_supplementary.yaml`.

    Returns:
        Structured scalar result for downstream CSV and markdown writing.
    """
    config = load_post_hoc_config(config_path)
    hosmer_lemeshow_groups = int(config["calibration_tests"]["hosmer_lemeshow_groups"])
    sample_index = _load_sample_index()
    prediction_frame = _read_csv(TEST_PREDICTION_PATH)
    _log_frame_details(prediction_frame, TEST_PREDICTION_PATH)
    validated_frame = _validate_test_prediction_frame(prediction_frame, sample_index)

    LOGGER.info(
        "Running calibration GOF tests with random_seed=%s and hosmer_lemeshow_groups=%s.",
        RANDOM_SEED,
        hosmer_lemeshow_groups,
    )
    result = run_calibration_tests_on_frame(
        frame=validated_frame,
        hosmer_lemeshow_groups=hosmer_lemeshow_groups,
    )
    LOGGER.info(
        "Hosmer-Lemeshow result: statistic=%.10f, df=%s, p_value=%.10f.",
        result.hosmer_lemeshow_statistic,
        result.hosmer_lemeshow_df,
        result.hosmer_lemeshow_p_value,
    )
    LOGGER.info(
        "Spiegelhalter result: observed_brier=%.10f, expected_brier=%.10f, variance=%.10f, z=%.10f, p_value=%.10f.",
        result.observed_brier,
        result.expected_brier,
        result.variance,
        result.spiegelhalter_z,
        result.spiegelhalter_p_value,
    )
    return result


def _build_csv_frame(result: CalibrationTestResult) -> pd.DataFrame:
    """Build the deterministic one-row CSV payload for Plan 1 §3.6."""
    return pd.DataFrame(
        [
            {
                "hosmer_lemeshow_statistic": result.hosmer_lemeshow_statistic,
                "hosmer_lemeshow_df": result.hosmer_lemeshow_df,
                "hosmer_lemeshow_p_value": result.hosmer_lemeshow_p_value,
                "spiegelhalter_z": result.spiegelhalter_z,
                "spiegelhalter_p_value": result.spiegelhalter_p_value,
                "observed_brier": result.observed_brier,
                "expected_brier": result.expected_brier,
                "variance": result.variance,
            }
        ],
        columns=[
            "hosmer_lemeshow_statistic",
            "hosmer_lemeshow_df",
            "hosmer_lemeshow_p_value",
            "spiegelhalter_z",
            "spiegelhalter_p_value",
            "observed_brier",
            "expected_brier",
            "variance",
        ],
    )


def _build_markdown_report(
    *,
    result: CalibrationTestResult,
    generated_at_utc: str,
    git_commit_hash: str,
    csv_path: Path,
    markdown_path: Path,
) -> str:
    """Build the Plan 1 §3.6 markdown report."""
    return (
        "# Post-Hoc Calibration Goodness-Of-Fit Tests (`minirocket_ridge`)\n\n"
        f"> {POST_HOC_DECLARATION}\n\n"
        f"- `generated_at_utc = {generated_at_utc}`\n"
        f"- `git_commit_hash = {git_commit_hash}`\n"
        f"- `output_csv = {_format_path(csv_path)}`\n"
        f"- `output_markdown = {_format_path(markdown_path)}`\n\n"
        "## Locked Method Inputs\n\n"
        f"- `hosmer_lemeshow_groups = {HOSMER_LEMESHOW_GROUPS}` "
        "`[ml-methodology-literature: standard HL decile grouping]`\n"
        f"- `hosmer_lemeshow_df = {HOSMER_LEMESHOW_DF}` "
        "`[ml-methodology-literature: G - 2 with G = 10]`\n\n"
        "## Frozen Calibration Context\n\n"
        f"- Frozen post-Platt Brier: `{FROZEN_POST_PLATT_BRIER:.4f}` "
        "`[runtime-determined-by-prespecified-algorithm: docs/modeling/validated_modeling_run_summary.md]`\n"
        f"- Frozen calibration slope: `{FROZEN_CALIBRATION_SLOPE:.4f}` "
        "`[runtime-determined-by-prespecified-algorithm: docs/modeling/validated_modeling_run_summary.md]`\n"
        f"- Frozen calibration intercept: `{FROZEN_CALIBRATION_INTERCEPT:.4f}` "
        "`[runtime-determined-by-prespecified-algorithm: docs/modeling/validated_modeling_run_summary.md]`\n\n"
        "Platt calibration is monotonic and does not change AUC-ROC or AUC-PR; it may change Brier score, "
        "calibration slope, and calibration intercept.\n\n"
        "## Descriptive Supplementary Results\n\n"
        f"- Hosmer-Lemeshow chi-square statistic: `{result.hosmer_lemeshow_statistic:.10f}` "
        f"with `df = {result.hosmer_lemeshow_df}` and `p = {result.hosmer_lemeshow_p_value:.10f}`\n"
        f"- Spiegelhalter z-statistic: `{result.spiegelhalter_z:.10f}` with "
        f"`p = {result.spiegelhalter_p_value:.10f}`\n"
        f"- Observed Brier from locked `p_cal`: `{result.observed_brier:.10f}`\n"
        f"- Expected Brier under calibration: `{result.expected_brier:.10f}`\n"
        f"- Spiegelhalter variance: `{result.variance:.10f}`\n\n"
        "Both statistics are descriptive supplementary evidence only, not pass/fail gates. They are interpreted "
        "alongside the frozen post-Platt Brier, slope, and intercept reported in the validated modeling run summary.\n"
    )


def generate_calibration_test_outputs(
    config_path: Path,
    *,
    generated_at_utc: str | None = None,
    git_commit_hash: str | None = None,
) -> CalibrationTestResult:
    """Run Plan 1 §3.6 and write the deterministic CSV plus markdown report.

    Args:
        config_path: Path to `configs/post_hoc_supplementary.yaml`.
        generated_at_utc: Optional externally supplied UTC timestamp for the markdown header.
        git_commit_hash: Optional externally supplied `HEAD` hash for the markdown header.

    Returns:
        Structured scalar result.
    """
    config = load_post_hoc_config(config_path)
    result = analyze_calibration_tests(config_path)
    timestamp_value = generated_at_utc or _timestamp_utc()
    git_hash_value = git_commit_hash or _get_git_commit_hash()

    outputs_analysis_dir = _resolve_path(config["paths"]["outputs_analysis_dir"])
    csv_path = outputs_analysis_dir / "posthoc_calibration_gof_minirocket_ridge.csv"
    markdown_path = outputs_analysis_dir / "posthoc_calibration_gof_minirocket_ridge.md"

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    _build_csv_frame(result).to_csv(csv_path, index=False, float_format="%.17g")
    markdown = _build_markdown_report(
        result=result,
        generated_at_utc=timestamp_value,
        git_commit_hash=git_hash_value,
        csv_path=csv_path,
        markdown_path=markdown_path,
    )
    _write_text(markdown_path, markdown)
    LOGGER.info(
        "Wrote calibration GOF outputs to %s and %s.",
        _format_path(csv_path),
        _format_path(markdown_path),
    )
    return result
