"""Post-hoc observed-prevalence DCA addendum for Session 9A.

This module generates the missing observed/internal-test prevalence decision
curve for the frozen `minirocket_ridge` test predictions. It reuses the locked
raw-probability DCA formula and the post-hoc prediction-validation helpers
without refitting, recalibrating, or modifying any primary modeling artifact.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.dca import _compute_net_benefit_at_threshold
from src.analysis.sample_size import POST_HOC_DECLARATION, load_post_hoc_config
from src.analysis.stage7_part_a_common import (
    _format_path,
    _get_git_commit_hash,
    _resolve_path,
    _timestamp_utc,
    _write_text,
)
from src.analysis.threshold_uncertainty import (
    EXPECTED_TEST_COLUMNS,
    _load_sample_index,
    _log_frame_details,
    _read_csv,
    _validate_prediction_frame,
)
from src.config import PROJECT_ROOT


LOGGER = logging.getLogger(__name__)

PROBABILITY_COLUMN = "p_raw"
LABEL_COLUMN = "y_true"
PREVALENCE_SCENARIO = "observed_internal_test"
CURVE_ORDER = ("model", "treat_all", "treat_none")

TEST_PREDICTION_PATH = PROJECT_ROOT / "outputs" / "predictions" / "binary_minirocket_ridge_test_predictions.csv"
GRID_SOURCE_PATH = (
    PROJECT_ROOT / "outputs" / "analysis" / "posthoc_threshold_uncertainty_minirocket_ridge_dca_curve_data.csv"
)
OPERATING_POINT_SOURCE_PATH = PROJECT_ROOT / "outputs" / "analysis" / "posthoc_threshold_uncertainty_minirocket_ridge.csv"
PRIMARY_DCA_PATH = PROJECT_ROOT / "outputs" / "analysis" / "dca_minirocket_ridge.csv"
PRIMARY_DCA_MARKDOWN_PATH = PROJECT_ROOT / "outputs" / "analysis" / "dca_minirocket_ridge.md"

GRID_SOURCE_COLUMNS = ["prevalence", "grid_threshold", "nb_model", "nb_treat_all", "nb_treat_none"]
OPERATING_POINT_COLUMNS = [
    "threshold_kind",
    "threshold_value",
    "prevalence",
    "tpr",
    "fpr",
    "nb_model",
    "nb_treat_all",
    "nb_treat_none",
]
PRIMARY_DCA_COLUMNS = ["prevalence_scenario", "threshold", "curve", "net_benefit"]
OUTPUT_COLUMNS = [
    "prevalence_scenario",
    "prevalence",
    "threshold",
    "curve",
    "net_benefit",
    "n",
    "n_events",
    "n_nonevents",
]


@dataclass(frozen=True)
class ObservedPrevalenceDcaResult:
    """Structured result for the Session 9A observed-prevalence DCA addendum.

    Attributes:
        observed_prevalence: Empirical event fraction in the locked test set.
        n: Locked internal-test sample size.
        n_events: Number of positive labels in the locked test set.
        n_nonevents: Number of negative labels in the locked test set.
        probability_column: Prediction column used for DCA.
        label_column: Binary label column used for DCA.
        threshold_grid_source: Project-relative path of the selected threshold source.
        existing_prevalence_scenarios: Prevalence scenarios already present in the
            primary DCA artifact.
        primary_grid_matches_source: Whether the primary DCA artifact uses the same
            threshold grid as the selected post-hoc source artifact.
        output_frame: Long-format observed-prevalence DCA table.
    """

    observed_prevalence: float
    n: int
    n_events: int
    n_nonevents: int
    probability_column: str
    label_column: str
    threshold_grid_source: str
    existing_prevalence_scenarios: tuple[float, ...]
    primary_grid_matches_source: bool
    output_frame: pd.DataFrame


def _read_text(path: Path) -> str:
    """Read one UTF-8 text file from disk."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        LOGGER.error("Failed to read text from %s.", path)
        raise OSError(f"Failed to read text from {path}.") from exc


def _validate_exact_columns(frame: pd.DataFrame, expected_columns: list[str], path: Path) -> pd.DataFrame:
    """Validate the exact column order for one persisted artifact."""
    if frame.columns.tolist() != expected_columns:
        raise AssertionError(
            f"{_format_path(path)} columns must be {expected_columns}, found {frame.columns.tolist()}."
        )
    return frame.copy()


def _load_locked_test_prediction_frame() -> pd.DataFrame:
    """Load and validate the frozen internal-test prediction artifact."""
    sample_index = _load_sample_index()
    prediction_frame = _read_csv(TEST_PREDICTION_PATH)
    _log_frame_details(prediction_frame, TEST_PREDICTION_PATH)
    validated = _validate_prediction_frame(
        prediction_frame,
        path=TEST_PREDICTION_PATH,
        sample_index=sample_index,
        expected_columns=EXPECTED_TEST_COLUMNS,
    )
    validated["p_cal"] = pd.to_numeric(validated["p_cal"], errors="raise").astype(float)
    validated["y_pred"] = pd.to_numeric(validated["y_pred"], errors="raise").astype(np.int64)
    if not np.isfinite(validated["p_cal"].to_numpy(dtype=float)).all():
        raise ValueError(f"{_format_path(TEST_PREDICTION_PATH)} p_cal contains NaN or Inf.")
    return validated


def _load_grid_source_frame() -> pd.DataFrame:
    """Load the existing post-hoc DCA curve data used as the threshold source."""
    grid_source = _read_csv(GRID_SOURCE_PATH)
    _log_frame_details(grid_source, GRID_SOURCE_PATH)
    return _validate_exact_columns(grid_source, GRID_SOURCE_COLUMNS, GRID_SOURCE_PATH)


def _load_operating_point_source_frame() -> pd.DataFrame:
    """Load the existing threshold-uncertainty operating-point artifact."""
    operating_point_source = _read_csv(OPERATING_POINT_SOURCE_PATH)
    _log_frame_details(operating_point_source, OPERATING_POINT_SOURCE_PATH)
    return _validate_exact_columns(operating_point_source, OPERATING_POINT_COLUMNS, OPERATING_POINT_SOURCE_PATH)


def _load_primary_dca_frame() -> pd.DataFrame:
    """Load the existing primary DCA artifact for consistency checks."""
    primary_dca = _read_csv(PRIMARY_DCA_PATH)
    _log_frame_details(primary_dca, PRIMARY_DCA_PATH)
    return _validate_exact_columns(primary_dca, PRIMARY_DCA_COLUMNS, PRIMARY_DCA_PATH)


def _extract_threshold_grid(frame: pd.DataFrame, *, column: str, path: Path) -> np.ndarray:
    """Extract one persisted threshold grid and verify it is finite and safe."""
    threshold_grid = frame[column].drop_duplicates().to_numpy(dtype=float)
    if threshold_grid.size == 0:
        raise ValueError(f"{_format_path(path)} does not contain any threshold values.")
    if not np.isfinite(threshold_grid).all():
        raise ValueError(f"{_format_path(path)} contains NaN or Inf threshold values.")
    if np.any((threshold_grid < 0.0) | (threshold_grid >= 1.0)):
        raise ValueError(
            f"{_format_path(path)} contains threshold values outside the safe DCA interval [0, 1)."
        )
    return threshold_grid


def _validate_primary_dca_markdown(markdown_text: str) -> None:
    """Validate that the primary DCA markdown still records its key provenance lines."""
    required_snippets = [
        _format_path(TEST_PREDICTION_PATH),
        _format_path(PRIMARY_DCA_PATH),
        "np.arange(0.01, 0.99, 0.01)",
    ]
    for snippet in required_snippets:
        if snippet not in markdown_text:
            raise ValueError(
                f"{_format_path(PRIMARY_DCA_MARKDOWN_PATH)} is missing the expected provenance snippet {snippet!r}."
            )


def _build_output_frame(
    *,
    y_true: np.ndarray,
    p_raw: np.ndarray,
    threshold_grid: np.ndarray,
) -> tuple[pd.DataFrame, float, int, int, int]:
    """Build the long-format observed-prevalence DCA output frame."""
    observed_prevalence = float(np.mean(y_true))
    n = int(len(y_true))
    n_events = int(np.sum(y_true == 1))
    n_nonevents = int(np.sum(y_true == 0))

    rows: list[dict[str, float | int | str]] = []
    for threshold in threshold_grid:
        net_benefits = _compute_net_benefit_at_threshold(
            y_true=y_true,
            p_raw=p_raw,
            threshold=float(threshold),
            prevalence=observed_prevalence,
        )
        for curve_name in CURVE_ORDER:
            rows.append(
                {
                    "prevalence_scenario": PREVALENCE_SCENARIO,
                    "prevalence": observed_prevalence,
                    "threshold": float(threshold),
                    "curve": curve_name,
                    "net_benefit": float(net_benefits[curve_name]),
                    "n": n,
                    "n_events": n_events,
                    "n_nonevents": n_nonevents,
                }
            )

    output_frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    return output_frame, observed_prevalence, n, n_events, n_nonevents


def analyze_observed_prevalence_dca(config_path: Path) -> ObservedPrevalenceDcaResult:
    """Run the read-only observed-prevalence DCA analysis without writing outputs.

    Args:
        config_path: Path to `configs/post_hoc_supplementary.yaml`.

    Returns:
        Structured observed-prevalence DCA result.
    """
    load_post_hoc_config(config_path)

    test_prediction_frame = _load_locked_test_prediction_frame()
    grid_source_frame = _load_grid_source_frame()
    _load_operating_point_source_frame()
    primary_dca_frame = _load_primary_dca_frame()
    primary_dca_markdown = _read_text(PRIMARY_DCA_MARKDOWN_PATH)
    _validate_primary_dca_markdown(primary_dca_markdown)

    threshold_grid = _extract_threshold_grid(grid_source_frame, column="grid_threshold", path=GRID_SOURCE_PATH)
    primary_threshold_grid = _extract_threshold_grid(primary_dca_frame, column="threshold", path=PRIMARY_DCA_PATH)
    primary_grid_matches_source = bool(
        np.allclose(primary_threshold_grid, threshold_grid, atol=1e-15, rtol=0.0)
    )

    existing_prevalence_scenarios = tuple(
        float(value) for value in sorted(primary_dca_frame["prevalence_scenario"].unique().tolist())
    )
    LOGGER.info(
        "Observed-prevalence DCA uses probability_column=%s, label_column=%s, threshold_grid_source=%s, primary_grid_matches_source=%s.",
        PROBABILITY_COLUMN,
        LABEL_COLUMN,
        _format_path(GRID_SOURCE_PATH),
        primary_grid_matches_source,
    )
    LOGGER.info("Existing primary DCA prevalence_scenarios=%s.", existing_prevalence_scenarios)

    y_true = test_prediction_frame[LABEL_COLUMN].to_numpy(dtype=np.int64)
    p_raw = test_prediction_frame[PROBABILITY_COLUMN].to_numpy(dtype=float)
    output_frame, observed_prevalence, n, n_events, n_nonevents = _build_output_frame(
        y_true=y_true,
        p_raw=p_raw,
        threshold_grid=threshold_grid,
    )
    LOGGER.info(
        "Computed observed-prevalence DCA for n=%s, n_events=%s, n_nonevents=%s, observed_prevalence=%.15f.",
        n,
        n_events,
        n_nonevents,
        observed_prevalence,
    )

    return ObservedPrevalenceDcaResult(
        observed_prevalence=observed_prevalence,
        n=n,
        n_events=n_events,
        n_nonevents=n_nonevents,
        probability_column=PROBABILITY_COLUMN,
        label_column=LABEL_COLUMN,
        threshold_grid_source=_format_path(GRID_SOURCE_PATH),
        existing_prevalence_scenarios=existing_prevalence_scenarios,
        primary_grid_matches_source=primary_grid_matches_source,
        output_frame=output_frame,
    )


def _build_markdown_report(
    *,
    result: ObservedPrevalenceDcaResult,
    output_csv_path: Path,
    output_markdown_path: Path,
    generated_at_utc: str,
    git_commit_hash: str,
) -> str:
    """Build the Session 9A provenance markdown report."""
    input_paths = [
        TEST_PREDICTION_PATH,
        GRID_SOURCE_PATH,
        OPERATING_POINT_SOURCE_PATH,
        PRIMARY_DCA_PATH,
        PRIMARY_DCA_MARKDOWN_PATH,
    ]
    input_lines = "\n".join(f"- `{_format_path(path)}`" for path in input_paths)
    output_lines = "\n".join(f"- `{_format_path(path)}`" for path in [output_csv_path, output_markdown_path])
    grid_match_text = "yes" if result.primary_grid_matches_source else "no"
    prevalence_scenarios_text = ", ".join(f"{value:.2f}" for value in result.existing_prevalence_scenarios)
    return (
        "# Post-Hoc Observed-Prevalence DCA (`minirocket_ridge`)\n\n"
        "Purpose: persist the missing observed/internal-test prevalence decision-curve artifact needed only for the "
        "Session 9 eFigure 10 observed-prevalence panel.\n\n"
        f"{POST_HOC_DECLARATION}\n\n"
        "## Inputs\n\n"
        f"{input_lines}\n\n"
        "## Outputs\n\n"
        f"{output_lines}\n\n"
        "## Locked Computation Choices\n\n"
        f"- Prediction column used: `{result.probability_column}`\n"
        f"- Label column used: `{result.label_column}`\n"
        f"- Probability scale used: `raw`\n"
        f"- Threshold-grid source used: `{result.threshold_grid_source}`\n"
        f"- Primary DCA grid matches selected source: `{grid_match_text}`\n"
        f"- Existing primary DCA prevalence scenarios inspected: `{prevalence_scenarios_text}`\n"
        f"- Observed internal-test prevalence: `{result.observed_prevalence:.17g}`\n"
        f"- `n = {result.n}`\n"
        f"- `n_events = {result.n_events}`\n"
        f"- `n_nonevents = {result.n_nonevents}`\n\n"
        "## Net-Benefit Formulas\n\n"
        "- Model net benefit: `nb_model = TP / n - FP / n * pt / (1 - pt)`\n"
        "- Treat-all net benefit: "
        "`nb_treat_all = observed_prevalence - (1 - observed_prevalence) * pt / (1 - pt)`\n"
        "- Treat-none net benefit: `nb_treat_none = 0`\n\n"
        "## Integrity Statements\n\n"
        "- No model was refit.\n"
        "- No prediction artifact was modified.\n"
        "- No threshold, calibrator, split, fold, normalizer, transformer, or primary-analysis artifact was modified.\n"
        "- This artifact exists only to supply the observed/internal-test prevalence panel for Session 9 eFigure 10.\n\n"
        "## Limitation\n\n"
        "- The observed internal-test prevalence reflects the enriched internal-test cohort and should not be interpreted "
        "as natural-population screening prevalence.\n"
        "- The `0.01`, `0.05`, and `0.10` scenario curves remain the deployment-oriented prevalence-scenario analyses.\n\n"
        "## Provenance\n\n"
        f"- `generated_at_utc = {generated_at_utc}`\n"
        f"- `git_commit_hash = {git_commit_hash}`\n"
    )


def generate_observed_prevalence_dca_outputs(
    config_path: Path,
    *,
    generated_at_utc: str | None = None,
    git_commit_hash: str | None = None,
) -> ObservedPrevalenceDcaResult:
    """Generate the observed-prevalence DCA CSV and provenance markdown.

    Args:
        config_path: Path to `configs/post_hoc_supplementary.yaml`.
        generated_at_utc: Optional externally supplied UTC timestamp.
        git_commit_hash: Optional externally supplied `HEAD` commit hash.

    Returns:
        Structured observed-prevalence DCA result.
    """
    config = load_post_hoc_config(config_path)
    result = analyze_observed_prevalence_dca(config_path)

    outputs_analysis_dir = _resolve_path(config["paths"]["outputs_analysis_dir"])
    output_csv_path = outputs_analysis_dir / "posthoc_observed_prevalence_dca_minirocket_ridge.csv"
    output_markdown_path = outputs_analysis_dir / "posthoc_observed_prevalence_dca_minirocket_ridge.md"
    timestamp_value = generated_at_utc or _timestamp_utc()
    git_hash_value = git_commit_hash or _get_git_commit_hash()

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    result.output_frame.to_csv(output_csv_path, index=False, float_format="%.17g")
    markdown = _build_markdown_report(
        result=result,
        output_csv_path=output_csv_path,
        output_markdown_path=output_markdown_path,
        generated_at_utc=timestamp_value,
        git_commit_hash=git_hash_value,
    )
    _write_text(output_markdown_path, markdown)
    LOGGER.info(
        "Wrote observed-prevalence DCA outputs to %s and %s.",
        _format_path(output_csv_path),
        _format_path(output_markdown_path),
    )
    return result
