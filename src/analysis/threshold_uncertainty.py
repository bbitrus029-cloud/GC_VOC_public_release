"""Post-hoc threshold-uncertainty analysis for Plan 1 Session 3.

This module implements Plan 1 §3.5 steps 1-4b for the frozen best model
`minirocket_ridge`. It keeps the operating-point threshold sensitivity table and
the standard DCA-curve background table separate, and it reuses the existing
`src.analysis.dca` net-benefit helper rather than re-implementing the formula.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from src.analysis.dca import _build_threshold_grid, _compute_net_benefit_at_threshold
from src.analysis.sample_size import POST_HOC_DECLARATION, load_post_hoc_config
from src.analysis.stage7_part_a_common import (
    _format_path,
    _get_git_commit_hash,
    _load_global_config,
    _resolve_path,
    _timestamp_utc,
    _write_text,
)
from src.config import PROJECT_ROOT, RANDOM_SEED


LOGGER = logging.getLogger(__name__)

EXPECTED_OOF_COLUMNS = ["sample_id", "y_true", "p_raw", "outer_fold"]
EXPECTED_TEST_COLUMNS = ["sample_id", "y_true", "p_raw", "p_cal", "y_pred", "quadrant"]
THRESHOLD_KIND_ORDER = ("lower", "point", "upper")

SAMPLE_INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "sample_index.csv"
OOF_PREDICTION_PATH = PROJECT_ROOT / "outputs" / "predictions" / "binary_minirocket_ridge_nested_cv_oof.csv"
TEST_PREDICTION_PATH = PROJECT_ROOT / "outputs" / "predictions" / "binary_minirocket_ridge_test_predictions.csv"
THRESHOLD_MAPPING_PATH = PROJECT_ROOT / "outputs" / "analysis" / "binary_minirocket_ridge_threshold_mapping.json"
MODELING_GLOBAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "modeling_global.yaml"


@dataclass(frozen=True)
class ThresholdUncertaintyResult:
    """Structured result for Plan 1 §3.5 steps 1-4b.

    Attributes:
        frozen_threshold_p_raw: Raw-space threshold read from the frozen mapping.
        recomputed_threshold_p_raw: Recomputed OOF-space Youden threshold.
        agreement_delta: Absolute difference between frozen and recomputed thresholds.
        threshold_ci_low: Lower percentile bootstrap threshold bound.
        threshold_ci_high: Upper percentile bootstrap threshold bound.
        bootstrap_thresholds: All stratified bootstrap threshold draws.
        operating_point_frame: Fixed-threshold operating-point sensitivity table.
        dca_curve_frame: Standard DCA-curve background table.
    """

    frozen_threshold_p_raw: float
    recomputed_threshold_p_raw: float
    agreement_delta: float
    threshold_ci_low: float
    threshold_ci_high: float
    bootstrap_thresholds: np.ndarray
    operating_point_frame: pd.DataFrame
    dca_curve_frame: pd.DataFrame


def _read_json(path: Path) -> dict[str, object]:
    """Read one JSON object from disk."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        LOGGER.error("Failed to read JSON from %s.", path)
        raise OSError(f"Failed to read JSON from {path}.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON at {path} must deserialize to a mapping.")
    return payload


def _read_csv(path: Path) -> pd.DataFrame:
    """Read one CSV from disk."""
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
    """Load the final-cohort sample index required by the post-hoc alignment checks."""
    sample_index = _read_csv(SAMPLE_INDEX_PATH)
    _log_frame_details(sample_index, SAMPLE_INDEX_PATH)

    final_sample_index = sample_index.loc[sample_index["in_final_output"].fillna(False)].copy()
    final_sample_index["sample_id"] = pd.to_numeric(final_sample_index["sample_id"], errors="raise").astype(np.int64)
    final_sample_index["label"] = pd.to_numeric(final_sample_index["label"], errors="raise").astype(np.int64)
    final_sample_index.sort_values("sample_id", kind="mergesort", inplace=True)
    final_sample_index.reset_index(drop=True, inplace=True)
    return final_sample_index.loc[:, ["sample_id", "label"]]


def _validate_prediction_frame(
    frame: pd.DataFrame,
    *,
    path: Path,
    sample_index: pd.DataFrame,
    expected_columns: list[str],
) -> pd.DataFrame:
    """Validate one prediction artifact against the post-hoc alignment contract."""
    if frame.columns.tolist() != expected_columns:
        raise AssertionError(
            f"{_format_path(path)} columns must be {expected_columns}, found {frame.columns.tolist()}."
        )

    validated = frame.copy()
    validated["sample_id"] = pd.to_numeric(validated["sample_id"], errors="raise").astype(np.int64)
    validated["y_true"] = pd.to_numeric(validated["y_true"], errors="raise").astype(np.int64)
    validated["p_raw"] = pd.to_numeric(validated["p_raw"], errors="raise").astype(float)

    if not np.issubdtype(validated["sample_id"].dtype, np.integer):
        raise AssertionError(f"{_format_path(path)} sample_id must have integer dtype.")
    if not validated["sample_id"].is_monotonic_increasing:
        raise AssertionError(f"{_format_path(path)} sample_id must be monotonically increasing per Erratum E007.")
    if not validated["sample_id"].is_unique:
        raise AssertionError(f"{_format_path(path)} sample_id values must be unique.")
    if not np.isfinite(validated["p_raw"].to_numpy(dtype=float)).all():
        raise ValueError(f"{_format_path(path)} p_raw contains NaN or Inf.")

    if "outer_fold" in expected_columns:
        validated["outer_fold"] = pd.to_numeric(validated["outer_fold"], errors="raise").astype(np.int64)

    sample_id_set = set(sample_index["sample_id"].tolist())
    prediction_id_set = set(validated["sample_id"].tolist())
    if not prediction_id_set < sample_id_set:
        raise AssertionError(
            f"{_format_path(path)} sample_id values must be a strict subset of sample_index.csv[in_final_output == True]."
        )

    merged = validated.merge(sample_index, on="sample_id", how="left", validate="one_to_one")
    if merged["label"].isna().any():
        raise AssertionError(f"{_format_path(path)} contains sample_ids missing from sample_index.csv.")
    if not np.array_equal(
        merged["y_true"].to_numpy(dtype=np.int64),
        merged["label"].to_numpy(dtype=np.int64),
    ):
        raise AssertionError(f"{_format_path(path)} y_true does not match sample_index.csv labels.")

    return validated


def _load_locked_prediction_artifacts() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the frozen OOF and test prediction artifacts."""
    sample_index = _load_sample_index()

    oof_frame = _read_csv(OOF_PREDICTION_PATH)
    _log_frame_details(oof_frame, OOF_PREDICTION_PATH)
    validated_oof = _validate_prediction_frame(
        oof_frame,
        path=OOF_PREDICTION_PATH,
        sample_index=sample_index,
        expected_columns=EXPECTED_OOF_COLUMNS,
    )

    test_frame = _read_csv(TEST_PREDICTION_PATH)
    _log_frame_details(test_frame, TEST_PREDICTION_PATH)
    validated_test = _validate_prediction_frame(
        test_frame,
        path=TEST_PREDICTION_PATH,
        sample_index=sample_index,
        expected_columns=EXPECTED_TEST_COLUMNS,
    )
    validated_test["p_cal"] = pd.to_numeric(validated_test["p_cal"], errors="raise").astype(float)
    validated_test["y_pred"] = pd.to_numeric(validated_test["y_pred"], errors="raise").astype(np.int64)
    if not np.isfinite(validated_test["p_cal"].to_numpy(dtype=float)).all():
        raise ValueError(f"{_format_path(TEST_PREDICTION_PATH)} p_cal contains NaN or Inf.")

    return validated_oof, validated_test


def _validate_binary_arrays(y_true: np.ndarray, p_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate one binary-label and score pair."""
    y_array = np.asarray(y_true, dtype=np.int64)
    p_array = np.asarray(p_raw, dtype=float)
    if y_array.ndim != 1 or p_array.ndim != 1:
        raise ValueError("y_true and p_raw must be one-dimensional.")
    if len(y_array) != len(p_array):
        raise ValueError("y_true and p_raw must have the same length.")
    if len(np.unique(y_array)) != 2:
        raise ValueError("Threshold analysis requires both classes to be present.")
    if not np.isfinite(p_array).all():
        raise ValueError("p_raw contains NaN or Inf.")
    return y_array, p_array


def compute_youden_threshold(
    *,
    y_true: np.ndarray,
    p_raw: np.ndarray,
    tie_break: str,
) -> float:
    """Compute the raw-space Youden threshold from observed score cutpoints.

    Args:
        y_true: Binary labels.
        p_raw: Raw probabilities aligned to `y_true`.
        tie_break: Tie-break rule. Session 3 requires `"smallest_threshold"`.

    Returns:
        The selected raw-space threshold.

    Raises:
        ValueError: If inputs are invalid or the tie-break rule is unsupported.
    """
    if tie_break != "smallest_threshold":
        raise ValueError(f"Unsupported tie-break rule: {tie_break!r}.")

    y_array, p_array = _validate_binary_arrays(y_true=y_true, p_raw=p_raw)
    fpr, tpr, thresholds = roc_curve(y_array, p_array)
    finite_mask = np.isfinite(thresholds)
    if not finite_mask.any():
        raise ValueError("No finite observed score cutpoints are available for Youden thresholding.")

    youden_values = tpr[finite_mask] - fpr[finite_mask]
    finite_thresholds = thresholds[finite_mask]
    best_youden = np.max(youden_values)
    best_thresholds = finite_thresholds[youden_values == best_youden]
    threshold = float(np.min(best_thresholds))

    if not np.isfinite(threshold):
        raise ValueError("Recomputed Youden threshold is NaN or Inf.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Recomputed Youden threshold must lie in [0, 1], received {threshold}.")
    return threshold


def draw_stratified_bootstrap_indices(*, y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw one stratified bootstrap sample using the single-RNG sequential convention."""
    y_array = np.asarray(y_true, dtype=np.int64)
    negative_indices = np.flatnonzero(y_array == 0)
    positive_indices = np.flatnonzero(y_array == 1)
    if len(negative_indices) == 0 or len(positive_indices) == 0:
        raise ValueError("Stratified bootstrap requires both classes to be present.")

    sampled_negative = rng.choice(negative_indices, size=len(negative_indices), replace=True)
    sampled_positive = rng.choice(positive_indices, size=len(positive_indices), replace=True)
    return np.concatenate([sampled_negative, sampled_positive]).astype(np.int64, copy=False)


def bootstrap_youden_thresholds(
    *,
    y_true: np.ndarray,
    p_raw: np.ndarray,
    n_boot: int,
    random_seed: int,
    tie_break: str,
) -> np.ndarray:
    """Bootstrap the raw-space Youden threshold over stratified OOF resamples.

    Args:
        y_true: Binary labels.
        p_raw: Raw probabilities aligned to `y_true`.
        n_boot: Number of bootstrap replicates.
        random_seed: Locked single-RNG sequential seed.
        tie_break: Tie-break rule for threshold selection.

    Returns:
        A deterministic bootstrap threshold array of shape `(n_boot,)`.

    Raises:
        ValueError: If any bootstrap halt condition is triggered.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1.")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise TypeError("random_seed must be an integer.")

    y_array, p_array = _validate_binary_arrays(y_true=y_true, p_raw=p_raw)
    rng = np.random.default_rng(random_seed)
    thresholds = np.empty(n_boot, dtype=float)

    for bootstrap_index in range(n_boot):
        sampled_indices = draw_stratified_bootstrap_indices(y_true=y_array, rng=rng)
        sampled_y = y_array[sampled_indices]
        sampled_p = p_array[sampled_indices]
        if len(np.unique(sampled_y)) != 2:
            raise ValueError(
                f"Plan 1 §3.5 halt: bootstrap iteration {bootstrap_index} lacks both classes despite stratification."
            )

        thresholds[bootstrap_index] = compute_youden_threshold(
            y_true=sampled_y,
            p_raw=sampled_p,
            tie_break=tie_break,
        )
        if not np.isfinite(thresholds[bootstrap_index]):
            raise ValueError(
                f"Plan 1 §3.5 halt: bootstrap threshold at iteration {bootstrap_index} is NaN or Inf."
            )
        if not 0.0 <= thresholds[bootstrap_index] <= 1.0:
            raise ValueError(
                f"Plan 1 §3.5 halt: bootstrap threshold at iteration {bootstrap_index} is outside [0, 1]."
            )

    return thresholds


def compute_threshold_percentile_ci(
    *,
    bootstrap_thresholds: np.ndarray,
    ci_percentiles: Iterable[float],
) -> tuple[float, float]:
    """Compute the percentile bootstrap CI for the threshold distribution."""
    threshold_array = np.asarray(bootstrap_thresholds, dtype=float)
    percentile_values = tuple(float(value) for value in ci_percentiles)
    if len(percentile_values) != 2:
        raise ValueError("ci_percentiles must contain exactly two values.")

    lower_ci, upper_ci = np.percentile(threshold_array, percentile_values)
    lower = float(lower_ci)
    upper = float(upper_ci)
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("Plan 1 §3.5 halt: threshold CI bounds are NaN or Inf.")
    if not 0.0 <= lower <= 1.0 or not 0.0 <= upper <= 1.0:
        raise ValueError("Plan 1 §3.5 halt: threshold CI bounds fall outside [0, 1].")
    if lower > upper:
        raise ValueError("Plan 1 §3.5 halt: threshold CI lower bound exceeds the upper bound.")
    return lower, upper


def compute_operating_point_net_benefits(
    *,
    y_true: np.ndarray,
    p_raw: np.ndarray,
    threshold: float,
    prevalence: float,
) -> dict[str, float]:
    """Compute TPR/FPR and net benefits at one fixed operating point.

    Args:
        y_true: Binary labels.
        p_raw: Raw probabilities aligned to `y_true`.
        threshold: Operating-point threshold in raw-probability space.
        prevalence: Clinical prevalence scenario.

    Returns:
        Mapping containing TPR, FPR, and the three net-benefit quantities.

    Raises:
        ValueError: If any metric is non-finite or inputs are invalid.
    """
    y_array, p_array = _validate_binary_arrays(y_true=y_true, p_raw=p_raw)
    if not 0.0 <= threshold < 1.0:
        raise ValueError(f"threshold must lie in [0, 1), received {threshold}.")
    if not 0.0 <= prevalence <= 1.0:
        raise ValueError(f"prevalence must lie in [0, 1], received {prevalence}.")

    y_pred = (p_array >= threshold).astype(np.int64)
    positive_mask = y_array == 1
    negative_mask = y_array == 0
    tpr = float(np.mean(y_pred[positive_mask] == 1))
    fpr = float(np.mean(y_pred[negative_mask] == 1))

    net_benefits = _compute_net_benefit_at_threshold(
        y_true=y_array,
        p_raw=p_array,
        threshold=float(threshold),
        prevalence=float(prevalence),
    )
    result = {
        "tpr": tpr,
        "fpr": fpr,
        "nb_model": float(net_benefits["model"]),
        "nb_treat_all": float(net_benefits["treat_all"]),
        "nb_treat_none": float(net_benefits["treat_none"]),
    }
    if not np.isfinite(np.array(list(result.values()), dtype=float)).all():
        raise ValueError("Plan 1 §3.5 halt: operating-point net-benefit values are NaN or Inf.")
    return result


def _build_operating_point_frame(
    *,
    y_true: np.ndarray,
    p_raw: np.ndarray,
    threshold_values: dict[str, float],
    prevalence_scenarios: list[float],
) -> pd.DataFrame:
    """Build the fixed-threshold operating-point sensitivity table."""
    rows: list[dict[str, float | str]] = []
    for threshold_kind in THRESHOLD_KIND_ORDER:
        threshold_value = threshold_values[threshold_kind]
        for prevalence in prevalence_scenarios:
            metrics = compute_operating_point_net_benefits(
                y_true=y_true,
                p_raw=p_raw,
                threshold=threshold_value,
                prevalence=prevalence,
            )
            rows.append(
                {
                    "threshold_kind": threshold_kind,
                    "threshold_value": float(threshold_value),
                    "prevalence": float(prevalence),
                    "tpr": float(metrics["tpr"]),
                    "fpr": float(metrics["fpr"]),
                    "nb_model": float(metrics["nb_model"]),
                    "nb_treat_all": float(metrics["nb_treat_all"]),
                    "nb_treat_none": float(metrics["nb_treat_none"]),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "threshold_kind",
            "threshold_value",
            "prevalence",
            "tpr",
            "fpr",
            "nb_model",
            "nb_treat_all",
            "nb_treat_none",
        ],
    )


def _build_dca_curve_frame(
    *,
    y_true: np.ndarray,
    p_raw: np.ndarray,
    prevalence_scenarios: list[float],
) -> pd.DataFrame:
    """Build the standard DCA-curve background table for Plan 2 rendering."""
    rows: list[dict[str, float]] = []
    threshold_grid = _build_threshold_grid()
    for prevalence in prevalence_scenarios:
        for grid_threshold in threshold_grid:
            metrics = compute_operating_point_net_benefits(
                y_true=y_true,
                p_raw=p_raw,
                threshold=float(grid_threshold),
                prevalence=float(prevalence),
            )
            rows.append(
                {
                    "prevalence": float(prevalence),
                    "grid_threshold": float(grid_threshold),
                    "nb_model": float(metrics["nb_model"]),
                    "nb_treat_all": float(metrics["nb_treat_all"]),
                    "nb_treat_none": float(metrics["nb_treat_none"]),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["prevalence", "grid_threshold", "nb_model", "nb_treat_all", "nb_treat_none"],
    )


def analyze_threshold_uncertainty(config_path: Path) -> ThresholdUncertaintyResult:
    """Run the full read-only Plan 1 §3.5 analysis without writing outputs.

    Args:
        config_path: Path to `configs/post_hoc_supplementary.yaml`.

    Returns:
        Structured threshold-uncertainty result.

    Raises:
        ValueError: If any Session 3 halt condition is triggered.
        AssertionError: If any alignment assertion fails.
    """
    config = load_post_hoc_config(config_path)
    global_config = _load_global_config(MODELING_GLOBAL_CONFIG_PATH)
    oof_frame, test_frame = _load_locked_prediction_artifacts()

    threshold_mapping = _read_json(THRESHOLD_MAPPING_PATH)
    frozen_threshold_p_raw = float(threshold_mapping["threshold_p_raw"])
    prevalence_scenarios = [float(value) for value in global_config["prevalence_scenarios"]]
    threshold_config = config["threshold_uncertainty"]

    LOGGER.info(
        "Running threshold-uncertainty analysis with random_seed=%s, n_boot=%s, ci_percentiles=%s, tie_break=%s.",
        config["random_seed"],
        threshold_config["n_boot"],
        threshold_config["ci_percentiles"],
        threshold_config["tie_break"],
    )
    LOGGER.info("Using prevalence_scenarios=%s from %s.", prevalence_scenarios, _format_path(MODELING_GLOBAL_CONFIG_PATH))

    oof_y_true = oof_frame["y_true"].to_numpy(dtype=np.int64)
    oof_p_raw = oof_frame["p_raw"].to_numpy(dtype=float)
    test_y_true = test_frame["y_true"].to_numpy(dtype=np.int64)
    test_p_raw = test_frame["p_raw"].to_numpy(dtype=float)

    recomputed_threshold_p_raw = compute_youden_threshold(
        y_true=oof_y_true,
        p_raw=oof_p_raw,
        tie_break=str(threshold_config["tie_break"]),
    )
    agreement_delta = float(abs(recomputed_threshold_p_raw - frozen_threshold_p_raw))
    LOGGER.info("Recomputed threshold_p_raw=%.15f", recomputed_threshold_p_raw)
    LOGGER.info("Frozen threshold_p_raw from mapping JSON=%.15f", frozen_threshold_p_raw)
    LOGGER.info("Agreement delta vs frozen threshold=%.15f", agreement_delta)
    if agreement_delta > 1e-4:
        raise ValueError(
            "Plan 1 §3.5 halt: recomputed OOF point threshold disagrees with the frozen threshold mapping by more than 1e-4."
        )

    bootstrap_thresholds = bootstrap_youden_thresholds(
        y_true=oof_y_true,
        p_raw=oof_p_raw,
        n_boot=int(threshold_config["n_boot"]),
        random_seed=int(config["random_seed"]),
        tie_break=str(threshold_config["tie_break"]),
    )
    threshold_ci_low, threshold_ci_high = compute_threshold_percentile_ci(
        bootstrap_thresholds=bootstrap_thresholds,
        ci_percentiles=threshold_config["ci_percentiles"],
    )
    LOGGER.info(
        "Bootstrap threshold percentile CI=[%.15f, %.15f]",
        threshold_ci_low,
        threshold_ci_high,
    )

    threshold_values = {
        "lower": float(threshold_ci_low),
        "point": float(recomputed_threshold_p_raw),
        "upper": float(threshold_ci_high),
    }
    operating_point_frame = _build_operating_point_frame(
        y_true=test_y_true,
        p_raw=test_p_raw,
        threshold_values=threshold_values,
        prevalence_scenarios=prevalence_scenarios,
    )
    dca_curve_frame = _build_dca_curve_frame(
        y_true=test_y_true,
        p_raw=test_p_raw,
        prevalence_scenarios=prevalence_scenarios,
    )

    return ThresholdUncertaintyResult(
        frozen_threshold_p_raw=float(frozen_threshold_p_raw),
        recomputed_threshold_p_raw=float(recomputed_threshold_p_raw),
        agreement_delta=agreement_delta,
        threshold_ci_low=float(threshold_ci_low),
        threshold_ci_high=float(threshold_ci_high),
        bootstrap_thresholds=bootstrap_thresholds,
        operating_point_frame=operating_point_frame,
        dca_curve_frame=dca_curve_frame,
    )


def _render_operating_point_table(operating_point_frame: pd.DataFrame) -> str:
    """Render the operating-point table as markdown."""
    header = (
        "| threshold_kind | threshold_value | prevalence | tpr | fpr | nb_model | nb_treat_all | nb_treat_none |"
    )
    divider = "| --- | --- | --- | --- | --- | --- | --- | --- |"
    rows = [
        (
            f"| {row.threshold_kind} | {row.threshold_value:.15f} | {row.prevalence:.2f} | "
            f"{row.tpr:.6f} | {row.fpr:.6f} | {row.nb_model:.6f} | "
            f"{row.nb_treat_all:.6f} | {row.nb_treat_none:.6f} |"
        )
        for row in operating_point_frame.itertuples(index=False)
    ]
    return "\n".join([header, divider, *rows])


def _build_markdown_report(
    *,
    result: ThresholdUncertaintyResult,
    generated_at_utc: str,
    git_commit_hash: str,
    markdown_path: Path,
    operating_point_csv_path: Path,
    dca_curve_csv_path: Path,
) -> str:
    """Build the Plan 1 §3.5 markdown report."""
    operating_point_table = _render_operating_point_table(result.operating_point_frame)
    return (
        "# Post-Hoc Threshold-Uncertainty Sensitivity (`minirocket_ridge`)\n\n"
        f"> {POST_HOC_DECLARATION}\n\n"
        f"- `generated_at_utc = {generated_at_utc}`\n"
        f"- `git_commit_hash = {git_commit_hash}`\n"
        f"- `output_operating_point_csv = {_format_path(operating_point_csv_path)}`\n"
        f"- `output_dca_curve_csv = {_format_path(dca_curve_csv_path)}`\n"
        f"- `output_markdown = {_format_path(markdown_path)}`\n\n"
        "## Threshold Agreement And Bootstrap CI\n\n"
        f"- Recomputed `threshold_p_raw = {result.recomputed_threshold_p_raw:.15f}`\n"
        f"- Frozen mapping `threshold_p_raw = {result.frozen_threshold_p_raw:.15f}`\n"
        f"- Absolute agreement delta = `{result.agreement_delta:.15f}` (required `<= 1e-4`)\n"
        f"- 95% percentile bootstrap CI for `threshold_p_raw`: "
        f"`[{result.threshold_ci_low:.15f}, {result.threshold_ci_high:.15f}]`\n\n"
        "The frozen `threshold_p_raw` remains the operative threshold. This post-hoc sensitivity analysis "
        "quantifies uncertainty around that operating point but does not change any frozen primary-analysis artifact.\n\n"
        "## Operating-Point Net-Benefit Sensitivity\n\n"
        "This table is the fixed-threshold threshold-uncertainty result: the lower, point, and upper OOF-derived "
        "operating-point thresholds are applied to the locked test predictions at the three prespecified prevalence "
        "scenarios.\n\n"
        f"{operating_point_table}\n\n"
        "## Orthogonality Of The Two Tables\n\n"
        "The operating-point uncertainty table and the DCA-curve background table are different quantities. "
        "The operating-point table holds three fixed thresholds (`lower`, `point`, `upper`) and summarizes their "
        "TPR, FPR, and net benefit. The DCA-curve background table sweeps the full 98-point threshold grid for each "
        "prevalence scenario and is independent of the chosen operating point. Plan 2 will overlay the three "
        "operating points as markers on top of those per-prevalence DCA curves.\n\n"
        "Plan 1 renders no figure. The standard DCA-curve background data is persisted to "
        f"`{_format_path(dca_curve_csv_path)}` for downstream Plan 2 visualization.\n"
    )


def generate_threshold_uncertainty_outputs(
    config_path: Path,
    *,
    generated_at_utc: str | None = None,
    git_commit_hash: str | None = None,
) -> ThresholdUncertaintyResult:
    """Run Plan 1 §3.5 and write the two deterministic CSV outputs plus markdown.

    Args:
        config_path: Path to `configs/post_hoc_supplementary.yaml`.
        generated_at_utc: Optional externally supplied UTC timestamp for the markdown header.
        git_commit_hash: Optional externally supplied `HEAD` commit hash for the markdown header.

    Returns:
        Structured threshold-uncertainty result.
    """
    config = load_post_hoc_config(config_path)
    result = analyze_threshold_uncertainty(config_path)
    timestamp_value = generated_at_utc or _timestamp_utc()
    git_hash_value = git_commit_hash or _get_git_commit_hash()

    outputs_analysis_dir = _resolve_path(config["paths"]["outputs_analysis_dir"])
    operating_point_csv_path = outputs_analysis_dir / "posthoc_threshold_uncertainty_minirocket_ridge.csv"
    dca_curve_csv_path = outputs_analysis_dir / "posthoc_threshold_uncertainty_minirocket_ridge_dca_curve_data.csv"
    markdown_path = outputs_analysis_dir / "posthoc_threshold_uncertainty_minirocket_ridge.md"

    operating_point_csv_path.parent.mkdir(parents=True, exist_ok=True)
    result.operating_point_frame.to_csv(operating_point_csv_path, index=False, float_format="%.17g")
    result.dca_curve_frame.to_csv(dca_curve_csv_path, index=False, float_format="%.17g")
    markdown = _build_markdown_report(
        result=result,
        generated_at_utc=timestamp_value,
        git_commit_hash=git_hash_value,
        markdown_path=markdown_path,
        operating_point_csv_path=operating_point_csv_path,
        dca_curve_csv_path=dca_curve_csv_path,
    )
    _write_text(markdown_path, markdown)
    LOGGER.info(
        "Wrote threshold-uncertainty outputs to %s, %s, and %s.",
        _format_path(operating_point_csv_path),
        _format_path(dca_curve_csv_path),
        _format_path(markdown_path),
    )
    return result
