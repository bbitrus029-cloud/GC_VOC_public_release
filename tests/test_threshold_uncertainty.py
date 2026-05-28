from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import dca as dca_module
from src.config import PROJECT_ROOT


CONFIG_PATH = PROJECT_ROOT / "configs" / "post_hoc_supplementary.yaml"
OOF_PATH = PROJECT_ROOT / "outputs" / "predictions" / "binary_minirocket_ridge_nested_cv_oof.csv"
THRESHOLD_MAPPING_PATH = PROJECT_ROOT / "outputs" / "analysis" / "binary_minirocket_ridge_threshold_mapping.json"


def _import_module_or_fail():
    """Import the threshold-uncertainty module or fail with a readable message."""
    try:
        return importlib.import_module("src.analysis.threshold_uncertainty")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Missing module src.analysis.threshold_uncertainty: {exc}", pytrace=False)


def _load_real_oof_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Load the locked OOF labels and raw probabilities for MR+R."""
    oof_frame = pd.read_csv(OOF_PATH)
    return (
        oof_frame["y_true"].to_numpy(dtype=np.int64),
        oof_frame["p_raw"].to_numpy(dtype=float),
    )


def _load_frozen_threshold() -> float:
    """Load the frozen raw-space threshold from the mapping JSON."""
    payload = json.loads(THRESHOLD_MAPPING_PATH.read_text(encoding="utf-8"))
    return float(payload["threshold_p_raw"])


def test_bootstrap_threshold_array_is_bit_identical_for_seed_42() -> None:
    """Two bootstrap runs with seed 42 should be bit-identical."""
    module = _import_module_or_fail()
    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    p_raw = np.array([0.10, 0.20, 0.35, 0.55, 0.70, 0.85], dtype=float)

    first = module.bootstrap_youden_thresholds(
        y_true=y_true,
        p_raw=p_raw,
        n_boot=25,
        random_seed=42,
        tie_break="smallest_threshold",
    )
    second = module.bootstrap_youden_thresholds(
        y_true=y_true,
        p_raw=p_raw,
        n_boot=25,
        random_seed=42,
        tie_break="smallest_threshold",
    )

    assert np.array_equal(first, second)


def test_recomputed_youden_threshold_agrees_with_frozen_mapping() -> None:
    """The recomputed OOF Youden threshold should match the frozen mapping within 1e-4."""
    module = _import_module_or_fail()
    y_true, p_raw = _load_real_oof_arrays()

    recomputed_threshold = module.compute_youden_threshold(
        y_true=y_true,
        p_raw=p_raw,
        tie_break="smallest_threshold",
    )

    assert recomputed_threshold == pytest.approx(_load_frozen_threshold(), rel=0.0, abs=1e-4)


def test_smallest_threshold_is_used_for_tied_youden_maxima() -> None:
    """Tied Youden maxima should resolve to the smallest threshold."""
    module = _import_module_or_fail()
    y_true = np.array([1, 1, 0, 0], dtype=np.int64)
    p_raw = np.array([0.60, 0.40, 0.50, 0.30], dtype=float)

    threshold = module.compute_youden_threshold(
        y_true=y_true,
        p_raw=p_raw,
        tie_break="smallest_threshold",
    )

    assert threshold == pytest.approx(0.40, rel=0.0, abs=1e-12)


def test_net_benefit_matches_direct_dca_call() -> None:
    """Operating-point NB should match a direct call to the imported DCA helper."""
    module = _import_module_or_fail()
    y_true = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    p_raw = np.array([0.20, 0.60, 0.40, 0.70, 0.90], dtype=float)
    threshold = 0.40
    prevalence = 0.10

    operating_point = module.compute_operating_point_net_benefits(
        y_true=y_true,
        p_raw=p_raw,
        threshold=threshold,
        prevalence=prevalence,
    )
    direct = dca_module._compute_net_benefit_at_threshold(
        y_true=y_true,
        p_raw=p_raw,
        threshold=threshold,
        prevalence=prevalence,
    )

    assert operating_point["nb_model"] == pytest.approx(direct["model"])
    assert operating_point["nb_treat_all"] == pytest.approx(direct["treat_all"])
    assert operating_point["nb_treat_none"] == pytest.approx(direct["treat_none"])


def test_stratified_bootstrap_indices_preserve_both_classes() -> None:
    """Every stratified bootstrap draw should preserve both classes."""
    module = _import_module_or_fail()
    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    rng = np.random.default_rng(42)

    for _ in range(20):
        sampled_indices = module.draw_stratified_bootstrap_indices(y_true=y_true, rng=rng)
        assert set(np.unique(y_true[sampled_indices])) == {0, 1}


def test_threshold_ci_bounds_stay_in_unit_interval() -> None:
    """Bootstrap CI bounds should stay within [0, 1] and bracket the point threshold."""
    module = _import_module_or_fail()
    y_true, p_raw = _load_real_oof_arrays()

    point_threshold = module.compute_youden_threshold(
        y_true=y_true,
        p_raw=p_raw,
        tie_break="smallest_threshold",
    )
    bootstrap_thresholds = module.bootstrap_youden_thresholds(
        y_true=y_true,
        p_raw=p_raw,
        n_boot=200,
        random_seed=42,
        tie_break="smallest_threshold",
    )
    lower_ci, upper_ci = module.compute_threshold_percentile_ci(
        bootstrap_thresholds=bootstrap_thresholds,
        ci_percentiles=(2.5, 97.5),
    )

    assert 0.0 <= lower_ci <= 1.0
    assert 0.0 <= upper_ci <= 1.0
    assert lower_ci <= point_threshold <= upper_ci
