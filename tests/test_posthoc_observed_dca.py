from __future__ import annotations

import builtins
import hashlib
import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.sample_size import POST_HOC_DECLARATION
from src.config import PROJECT_ROOT


CONFIG_PATH = PROJECT_ROOT / "configs" / "post_hoc_supplementary.yaml"
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_posthoc_observed_prevalence_dca.py"
OUTPUT_CSV_PATH = PROJECT_ROOT / "outputs" / "analysis" / "posthoc_observed_prevalence_dca_minirocket_ridge.csv"
OUTPUT_MARKDOWN_PATH = PROJECT_ROOT / "outputs" / "analysis" / "posthoc_observed_prevalence_dca_minirocket_ridge.md"
PREDICTION_PATH = PROJECT_ROOT / "outputs" / "predictions" / "binary_minirocket_ridge_test_predictions.csv"
GRID_SOURCE_PATH = (
    PROJECT_ROOT / "outputs" / "analysis" / "posthoc_threshold_uncertainty_minirocket_ridge_dca_curve_data.csv"
)
OPERATING_POINT_SOURCE_PATH = PROJECT_ROOT / "outputs" / "analysis" / "posthoc_threshold_uncertainty_minirocket_ridge.csv"
DCA_SOURCE_PATH = PROJECT_ROOT / "outputs" / "analysis" / "dca_minirocket_ridge.csv"
DCA_MARKDOWN_PATH = PROJECT_ROOT / "outputs" / "analysis" / "dca_minirocket_ridge.md"
OVERLAY_PATH = PROJECT_ROOT / "outputs" / "analysis" / "dca_overlay_minirocket_ridge.csv"
SOURCE_ARTIFACT_PATHS = [
    PREDICTION_PATH,
    GRID_SOURCE_PATH,
    OPERATING_POINT_SOURCE_PATH,
    DCA_SOURCE_PATH,
    DCA_MARKDOWN_PATH,
]
REQUIRED_COLUMNS = [
    "prevalence_scenario",
    "prevalence",
    "threshold",
    "curve",
    "net_benefit",
    "n",
    "n_events",
    "n_nonevents",
]


def _import_module_or_fail():
    """Import the observed-prevalence DCA module or fail with a readable message."""
    try:
        return importlib.import_module("src.analysis.observed_prevalence_dca")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Missing module src.analysis.observed_prevalence_dca: {exc}", pytrace=False)


def _run_cli() -> None:
    """Run the observed-prevalence DCA CLI and assert success."""
    assert SCRIPT_PATH.exists(), f"Missing CLI script: {SCRIPT_PATH}"
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--config", str(CONFIG_PATH)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(
            "Observed-prevalence DCA CLI failed.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}",
            pytrace=False,
        )


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest for one artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_path(path: Path) -> str:
    """Return the project-relative string for one path."""
    return str(path.relative_to(PROJECT_ROOT))


def _load_prediction_frame() -> pd.DataFrame:
    """Load the frozen internal-test prediction artifact."""
    return pd.read_csv(PREDICTION_PATH)


def _load_output_frame() -> pd.DataFrame:
    """Load the generated observed-prevalence DCA output."""
    return pd.read_csv(OUTPUT_CSV_PATH)


def _compute_expected_net_benefits(
    *,
    y_true: np.ndarray,
    p_hat: np.ndarray,
    threshold: float,
) -> tuple[float, float, float]:
    """Compute the three required DCA net-benefit values directly from counts."""
    n = len(y_true)
    predicted_positive = p_hat >= threshold
    tp = int(np.sum(predicted_positive & (y_true == 1)))
    fp = int(np.sum(predicted_positive & (y_true == 0)))
    prevalence = float(np.mean(y_true))
    odds_term = float(threshold / (1.0 - threshold))
    nb_model = (tp / n) - ((fp / n) * odds_term)
    nb_treat_all = prevalence - ((1.0 - prevalence) * odds_term)
    return float(nb_model), float(nb_treat_all), 0.0


def test_cli_generates_observed_prevalence_outputs() -> None:
    """The CLI should generate the observed-prevalence CSV and markdown outputs."""
    _run_cli()
    assert OUTPUT_CSV_PATH.exists()
    assert OUTPUT_MARKDOWN_PATH.exists()


def test_output_schema_counts_and_prevalence_match_frozen_predictions() -> None:
    """The generated CSV should expose the required schema and frozen-cohort counts."""
    _run_cli()
    prediction_frame = _load_prediction_frame()
    output_frame = _load_output_frame()

    expected_prevalence = float(prediction_frame["y_true"].mean())
    expected_n = int(len(prediction_frame))
    expected_events = int(prediction_frame["y_true"].sum())
    expected_nonevents = int((prediction_frame["y_true"] == 0).sum())

    assert output_frame.columns.tolist() == REQUIRED_COLUMNS
    assert set(output_frame["prevalence_scenario"]) == {"observed_internal_test"}
    assert set(output_frame["curve"]) == {"model", "treat_all", "treat_none"}
    assert output_frame["prevalence"].nunique() == 1
    assert output_frame["prevalence"].iloc[0] == pytest.approx(expected_prevalence, rel=0.0, abs=1e-12)
    assert set(output_frame["n"]) == {expected_n}
    assert set(output_frame["n_events"]) == {expected_events}
    assert set(output_frame["n_nonevents"]) == {expected_nonevents}


def test_threshold_grid_and_net_benefits_match_persisted_sources() -> None:
    """The output should reuse the persisted threshold grid and direct DCA formulas."""
    _run_cli()
    prediction_frame = _load_prediction_frame()
    output_frame = _load_output_frame()
    grid_source = pd.read_csv(GRID_SOURCE_PATH)

    expected_thresholds = grid_source["grid_threshold"].drop_duplicates().to_numpy(dtype=float)
    observed_thresholds = output_frame["threshold"].drop_duplicates().to_numpy(dtype=float)
    representative_thresholds = [
        float(expected_thresholds[0]),
        float(expected_thresholds[len(expected_thresholds) // 2]),
        float(expected_thresholds[-1]),
    ]

    assert np.allclose(observed_thresholds, expected_thresholds, atol=1e-15, rtol=0.0)
    for threshold in observed_thresholds:
        threshold_rows = output_frame.loc[output_frame["threshold"] == threshold]
        assert threshold_rows["curve"].tolist() == ["model", "treat_all", "treat_none"]

    y_true = prediction_frame["y_true"].to_numpy(dtype=np.int64)
    p_raw = prediction_frame["p_raw"].to_numpy(dtype=float)
    for threshold in representative_thresholds:
        expected_model, expected_treat_all, expected_treat_none = _compute_expected_net_benefits(
            y_true=y_true,
            p_hat=p_raw,
            threshold=threshold,
        )
        threshold_rows = output_frame.loc[output_frame["threshold"] == threshold].set_index("curve")
        assert threshold_rows.loc["model", "net_benefit"] == pytest.approx(expected_model, rel=0.0, abs=1e-12)
        assert threshold_rows.loc["treat_all", "net_benefit"] == pytest.approx(
            expected_treat_all,
            rel=0.0,
            abs=1e-12,
        )
        assert threshold_rows.loc["treat_none", "net_benefit"] == pytest.approx(
            expected_treat_none,
            rel=0.0,
            abs=1e-15,
        )

    treat_none_rows = output_frame.loc[output_frame["curve"] == "treat_none", "net_benefit"]
    assert np.allclose(treat_none_rows.to_numpy(dtype=float), 0.0, atol=0.0, rtol=0.0)


def test_generation_does_not_modify_existing_source_artifacts() -> None:
    """Running the generator should leave all required input artifacts unchanged."""
    hashes_before = {path: _sha256(path) for path in SOURCE_ARTIFACT_PATHS}

    _run_cli()

    hashes_after = {path: _sha256(path) for path in SOURCE_ARTIFACT_PATHS}
    assert hashes_after == hashes_before


def test_generation_does_not_use_overlay_or_pkl_artifacts() -> None:
    """The generation logic must not read the tumor-marker overlay or any PKL file."""
    module = _import_module_or_fail()

    overlay_accesses: list[Path] = []
    pkl_accesses: list[Path] = []
    original_read_csv = pd.read_csv
    original_builtin_open = builtins.open
    original_path_open = Path.open

    def patched_read_csv(filepath_or_buffer, *args, **kwargs):  # type: ignore[no-untyped-def]
        """Block accidental reads from the tumor-marker overlay artifact."""
        candidate_path = Path(filepath_or_buffer) if isinstance(filepath_or_buffer, (str, Path)) else None
        if candidate_path is not None and candidate_path.name == OVERLAY_PATH.name:
            overlay_accesses.append(candidate_path)
            raise AssertionError("dca_overlay_minirocket_ridge.csv must not be used.")
        return original_read_csv(filepath_or_buffer, *args, **kwargs)

    def patched_builtin_open(file, *args, **kwargs):  # type: ignore[no-untyped-def]
        """Record any direct `.pkl` access attempted by the generation logic."""
        try:
            candidate_path = Path(file)
        except TypeError:
            candidate_path = None
        if candidate_path is not None and candidate_path.suffix == ".pkl":
            pkl_accesses.append(candidate_path)
        return original_builtin_open(file, *args, **kwargs)

    def patched_path_open(self: Path, *args, **kwargs):  # type: ignore[no-untyped-def]
        """Record any `Path.open()` call that targets a `.pkl` path."""
        if self.suffix == ".pkl":
            pkl_accesses.append(self)
        return original_path_open(self, *args, **kwargs)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(pd, "read_csv", patched_read_csv)
    monkeypatch.setattr(builtins, "open", patched_builtin_open)
    monkeypatch.setattr(Path, "open", patched_path_open)
    try:
        module.generate_observed_prevalence_dca_outputs(config_path=CONFIG_PATH)
    finally:
        monkeypatch.undo()

    assert overlay_accesses == []
    assert pkl_accesses == []


def test_markdown_contains_post_hoc_declaration_and_artifact_paths() -> None:
    """The markdown should record the mandatory declaration and provenance paths."""
    _run_cli()
    markdown = OUTPUT_MARKDOWN_PATH.read_text(encoding="utf-8")

    assert POST_HOC_DECLARATION in markdown
    for path in [
        PREDICTION_PATH,
        GRID_SOURCE_PATH,
        OPERATING_POINT_SOURCE_PATH,
        DCA_SOURCE_PATH,
        DCA_MARKDOWN_PATH,
        OUTPUT_CSV_PATH,
        OUTPUT_MARKDOWN_PATH,
    ]:
        assert _relative_path(path) in markdown
