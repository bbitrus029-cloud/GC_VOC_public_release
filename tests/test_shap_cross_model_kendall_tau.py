from __future__ import annotations

from typing import Any

import numpy as np
import pytest


def _import_symbols() -> Any:
    """Import Kendall tau helpers lazily for clear failures."""
    try:
        from src.analysis.shap_cross_model import _compute_kendall_tau_summary
    except (AttributeError, ImportError, ModuleNotFoundError) as exc:
        pytest.fail(f"shap_cross_model implementation is missing: {exc}", pytrace=False)

    return _compute_kendall_tau_summary


def test_kendall_tau_summary_returns_one_for_perfect_agreement() -> None:
    """Identical rankings should produce tau=1."""
    compute_summary = _import_symbols()

    values = np.arange(1, 12, dtype=float)
    summary = compute_summary(values, values, n_boot=200, seed=42)

    assert summary["kendall_tau"] == pytest.approx(1.0)
    assert np.isfinite(summary["p_value"])
    assert summary["ci_low"] <= summary["kendall_tau"] <= summary["ci_high"]


def test_kendall_tau_summary_returns_negative_one_for_reverse_ranking() -> None:
    """Reverse rankings should produce tau=-1."""
    compute_summary = _import_symbols()

    left = np.arange(1, 12, dtype=float)
    right = left[::-1]
    summary = compute_summary(left, right, n_boot=200, seed=42)

    assert summary["kendall_tau"] == pytest.approx(-1.0)
    assert summary["ci_low"] <= summary["kendall_tau"] <= summary["ci_high"]


def test_kendall_tau_summary_is_near_zero_for_mixed_order() -> None:
    """A mixed ordering should yield a weak correlation near zero."""
    compute_summary = _import_symbols()

    left = np.arange(1, 12, dtype=float)
    right = np.asarray([1, 11, 2, 10, 3, 9, 4, 8, 5, 7, 6], dtype=float)
    summary = compute_summary(left, right, n_boot=200, seed=42)

    assert abs(summary["kendall_tau"]) < 0.5
    assert np.isfinite(summary["ci_low"])
    assert np.isfinite(summary["ci_high"])
