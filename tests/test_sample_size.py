from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from src.config import PROJECT_ROOT


CONFIG_PATH = PROJECT_ROOT / "configs" / "post_hoc_supplementary.yaml"


def _import_module_or_fail():
    """Import the sample-size module or fail with a readable message."""
    try:
        return importlib.import_module("src.analysis.sample_size")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Missing module src.analysis.sample_size: {exc}", pytrace=False)


def test_cox_snell_r_squared_simulation_is_bit_identical() -> None:
    """The deterministic Cox-Snell simulation should be bit-identical across calls."""
    module = _import_module_or_fail()

    first = module._simulate_cox_snell_r_squared(
        anticipated_auc=0.90,
        anticipated_prevalence=0.50,
        cox_snell_simulation_n=1_000_000,
        random_seed=42,
    )
    second = module._simulate_cox_snell_r_squared(
        anticipated_auc=0.90,
        anticipated_prevalence=0.50,
        cox_snell_simulation_n=1_000_000,
        random_seed=42,
    )

    assert first == second


def test_b1_closed_form_matches_known_inputs() -> None:
    """Criterion B1 should match the hand-computed known-input fixture."""
    module = _import_module_or_fail()

    n_b1 = module._compute_n_b1(anticipated_prevalence=0.50, prevalence_margin_of_error=0.05)

    assert n_b1 == 385


def test_b2_raises_when_candidate_predictor_count_exceeds_thirty() -> None:
    """Criterion B2 must halt when p > 30 per Plan 1 §3.3."""
    module = _import_module_or_fail()

    with pytest.raises(ValueError, match="p <= 30"):
        module._compute_n_b2(
            candidate_predictor_parameters=31,
            anticipated_prevalence=0.50,
            target_mape=0.05,
        )


def test_b3_closed_form_matches_fixture_value() -> None:
    """Criterion B3 should match the closed-form fixture at R2_cs=0.30."""
    module = _import_module_or_fail()
    expected = 617

    n_b3 = module._compute_n_b3(
        candidate_predictor_parameters=25,
        target_shrinkage=0.90,
        r2_cs=0.30,
    )

    assert n_b3 == expected


def test_b4_shrinkage_derivation_matches_prespecified_equation() -> None:
    """The derived S_b4 should satisfy the prespecified algebraic identity."""
    module = _import_module_or_fail()
    r2_cs = 0.30
    r2_cs_max = 0.75
    target_r2_nagelkerke_optimism = 0.05

    s_b4 = module._compute_s_b4(
        r2_cs=r2_cs,
        target_r2_nagelkerke_optimism=target_r2_nagelkerke_optimism,
        r2_cs_max=r2_cs_max,
    )

    assert s_b4 == pytest.approx(
        r2_cs / (r2_cs + (target_r2_nagelkerke_optimism * r2_cs_max)),
        rel=0.0,
        abs=0.0,
    )
    assert s_b4 * (r2_cs + (target_r2_nagelkerke_optimism * r2_cs_max)) == pytest.approx(
        r2_cs,
        rel=0.0,
        abs=1e-15,
    )


def test_locked_inputs_return_shortfall_disclosure_structure() -> None:
    """Locked inputs should expose the B2 shortfall disclosure pathway."""
    module = _import_module_or_fail()

    result = module.analyze_sample_size(CONFIG_PATH)

    assert result.cohort_meets_all_criteria is False
    assert result.shortfall_criteria

    b2_shortfall = next(
        (entry for entry in result.shortfall_criteria if entry["criterion"] == "B2"),
        None,
    )
    assert b2_shortfall is not None
    assert b2_shortfall["n_required_for_criterion"] == 1374
    assert b2_shortfall["shortfall"] == 1374 - 972
    assert b2_shortfall["shortfall"] > 0
