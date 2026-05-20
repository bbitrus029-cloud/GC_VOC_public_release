from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def _import_module_or_fail(module_name: str):
    """Import a module or fail the test with a readable message."""
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.fail(f"Missing module {module_name}: {exc}")


def _metric_triplet(estimate: float, ci_low: float, ci_high: float) -> dict[str, float]:
    """Build the nested metric mapping used in nested-CV result JSON files."""
    return {
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def test_step1_strict_ci_separation_selects_unique_leader() -> None:
    """Step 1 should select a model whose lower CI exceeds all other upper CIs."""
    module = _import_module_or_fail("src.analysis.model_selection")

    results_dict = {
        "minirocket_ridge": {"aggregate_metrics": {"auc_roc": _metric_triplet(0.94, 0.91, 0.96)}},
        "xgboost": {"aggregate_metrics": {"auc_roc": _metric_triplet(0.88, 0.84, 0.90)}},
        "svm": {"aggregate_metrics": {"auc_roc": _metric_triplet(0.86, 0.82, 0.89)}},
    }

    selected_or_candidates = module._step1_strict_ci_separation(results_dict)

    assert selected_or_candidates == "minirocket_ridge"


def test_step1_strict_ci_separation_returns_overlap_candidates() -> None:
    """Step 1 should keep the overlapping models when no strict separation exists."""
    module = _import_module_or_fail("src.analysis.model_selection")

    results_dict = {
        "minirocket_ridge": {"aggregate_metrics": {"auc_roc": _metric_triplet(0.92, 0.88, 0.95)}},
        "elastic_net_lr": {"aggregate_metrics": {"auc_roc": _metric_triplet(0.91, 0.87, 0.94)}},
        "xgboost": {"aggregate_metrics": {"auc_roc": _metric_triplet(0.90, 0.86, 0.93)}},
    }

    selected_or_candidates = module._step1_strict_ci_separation(results_dict)

    assert isinstance(selected_or_candidates, list)
    assert set(selected_or_candidates) == {"minirocket_ridge", "elastic_net_lr", "xgboost"}


def test_step1_filters_dominated_models_when_no_unique_leader() -> None:
    """Step 1 should forward only the non-dominated subset to Step 2."""
    module = _import_module_or_fail("src.analysis.model_selection")

    results_dict = {
        "model_a": {"aggregate_metrics": {"auc_roc": _metric_triplet(0.93, 0.90, 0.96)}},
        "model_b": {"aggregate_metrics": {"auc_roc": _metric_triplet(0.92, 0.89, 0.95)}},
        "model_c": {"aggregate_metrics": {"auc_roc": _metric_triplet(0.84, 0.80, 0.88)}},
        "model_d": {"aggregate_metrics": {"auc_roc": _metric_triplet(0.78, 0.70, 0.84)}},
    }

    selected_or_candidates = module._step1_strict_ci_separation(results_dict)
    domination_map = module._build_ci_domination_map(results_dict)

    assert isinstance(selected_or_candidates, list)
    assert set(selected_or_candidates) == {"model_a", "model_b"}
    assert domination_map == {
        "model_c": ["model_a", "model_b"],
        "model_d": ["model_a", "model_b"],
    }


def test_step1_filters_six_model_real_cohort_pattern() -> None:
    """Step 1 should reproduce the observed non-dominated cohort pattern."""
    module = _import_module_or_fail("src.analysis.model_selection")

    results_dict = {
        "minirocket_ridge": {"aggregate_metrics": {"auc_roc": _metric_triplet(0.9246, 0.9048, 0.9427)}},
        "xgboost": {"aggregate_metrics": {"auc_roc": _metric_triplet(0.8981, 0.8756, 0.9188)}},
        "svm": {"aggregate_metrics": {"auc_roc": _metric_triplet(0.8662, 0.8410, 0.8901)}},
        "elastic_net_lr": {"aggregate_metrics": {"auc_roc": _metric_triplet(0.8615, 0.8351, 0.8861)}},
        "cnn_bilstm_attention": {"aggregate_metrics": {"auc_roc": _metric_triplet(0.8243, 0.7950, 0.8518)}},
        "inception_time": {"aggregate_metrics": {"auc_roc": _metric_triplet(0.8085, 0.7756, 0.8396)}},
    }

    selected_or_candidates = module._step1_strict_ci_separation(results_dict)
    domination_map = module._build_ci_domination_map(results_dict)

    assert isinstance(selected_or_candidates, list)
    assert set(selected_or_candidates) == {"minirocket_ridge", "xgboost"}
    assert set(domination_map) == {
        "svm",
        "elastic_net_lr",
        "cnn_bilstm_attention",
        "inception_time",
    }
    assert domination_map["svm"] == ["minirocket_ridge"]
    assert domination_map["elastic_net_lr"] == ["minirocket_ridge"]
    assert domination_map["cnn_bilstm_attention"] == ["minirocket_ridge", "xgboost"]
    assert domination_map["inception_time"] == ["minirocket_ridge", "xgboost", "svm"]


def test_step4_complexity_tiebreaker_uses_prespecified_order() -> None:
    """Step 4 should choose the earliest remaining model in the plan order."""
    module = _import_module_or_fail("src.analysis.model_selection")

    selected_model = module._step4_complexity_tiebreaker(
        ["cnn_bilstm_attention", "xgboost", "elastic_net_lr"]
    )

    assert selected_model == "elastic_net_lr"


def test_check_halt_conditions_raises_for_auc_ci_containing_no_skill() -> None:
    """Selection must halt if any AUC-ROC confidence interval contains 0.5."""
    module = _import_module_or_fail("src.analysis.model_selection")

    results_dict = {
        "minirocket_ridge": {
            "aggregate_metrics": {
                "auc_roc": _metric_triplet(0.71, 0.49, 0.84),
                "auc_pr": _metric_triplet(0.70, 0.60, 0.80),
                "brier_pre_platt": _metric_triplet(0.18, 0.16, 0.20),
            }
        },
        "xgboost": {
            "aggregate_metrics": {
                "auc_roc": _metric_triplet(0.81, 0.76, 0.86),
                "auc_pr": _metric_triplet(0.80, 0.74, 0.85),
                "brier_pre_platt": _metric_triplet(0.14, 0.12, 0.16),
            }
        },
    }

    with pytest.raises(RuntimeError, match="contains 0.5"):
        module._check_halt_conditions(results_dict)


def test_train_final_models_phase2_helpers_are_exposed() -> None:
    """Phase 2 final-training module should expose the aggregation helpers."""
    module = _import_module_or_fail("src.training.train_final_models")
    assert callable(module.train_final_model)
    assert callable(module.run_all_final_models)
    assert callable(module._extract_per_fold_best_hparams)
    assert callable(module._aggregate_final_hparams)
