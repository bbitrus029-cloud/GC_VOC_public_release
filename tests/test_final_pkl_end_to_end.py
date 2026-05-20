from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import GAS_CHANNELS, MODEL_NAMES, PIPELINE_B_FEATURE_COUNT
from src.evaluation.calibration import apply_platt_calibrator
from src.training.load_final_model import load_final_model
from tests.test_train_elastic_net_lr_smoke import elastic_net_lr_smoke_fixture
from tests.test_train_svm_smoke import svm_smoke_fixture
from tests.test_train_xgboost_smoke import xgboost_smoke_fixture


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_A_MODELS = {"minirocket_ridge", "inception_time", "cnn_bilstm_attention"}
PIPELINE_B_MODELS = {"xgboost", "svm", "elastic_net_lr"}
EXPECTED_FINAL_PKL_KEYS = {
    "transformer",
    "normalizer",
    "model",
    "calibrator",
    "threshold_p_raw",
    "threshold_p_final_equivalent",
    "feature_names",
    "training_sample_ids",
    "metadata",
}
EXPECTED_METADATA_KEYS = {
    "model_name",
    "final_hparams",
    "oof_source",
    "git_commit_hash",
    "training_timestamp_utc",
}


def _import_module_or_fail(module_name: str):
    """Import a module or fail the test with a readable message."""
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.fail(f"Missing module {module_name}: {exc}")


def _coerce_feature_frame(values: object, columns: list[str]) -> pd.DataFrame:
    """Convert MiniRocket features into a dataframe for the stored normalizer."""
    if isinstance(values, pd.DataFrame):
        frame = values.copy()
        frame.columns = columns
        return frame

    array = np.asarray(values, dtype=float)
    return pd.DataFrame(array, columns=columns)


def _load_pipeline_a_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Load the Stage 1 Pipeline A tensor and aligned sample IDs."""
    payload = np.load(
        PROJECT_ROOT / "data" / "processed" / "sequences" / "pipeline_a_raw.npz",
        allow_pickle=False,
    )
    return payload["x"], payload["sample_id"].astype(np.int64)


def _load_pipeline_b_frame() -> pd.DataFrame:
    """Load the Stage 1 Pipeline B feature table."""
    return pd.read_csv(PROJECT_ROOT / "data" / "processed" / "features" / "pipeline_b_raw.csv")


def _lookup_pipeline_a_sample(sample_id: int) -> np.ndarray:
    """Return one raw `[1, L, C]` Pipeline A sample by sample ID."""
    x_all, sample_ids = _load_pipeline_a_arrays()
    match_idx = np.flatnonzero(sample_ids == sample_id)
    assert len(match_idx) == 1
    return x_all[match_idx[0] : match_idx[0] + 1]


def _lookup_pipeline_b_sample(sample_id: int, feature_names: list[str]) -> pd.DataFrame:
    """Return one raw `[1, 99]` Pipeline B sample by sample ID."""
    frame = _load_pipeline_b_frame()
    matched = frame.loc[frame["sample_id"].astype(np.int64) == sample_id, feature_names]
    assert len(matched) == 1
    return matched.reset_index(drop=True)


def test_extract_per_fold_best_hparams_reads_shared_schema() -> None:
    """Phase 2 should extract five per-fold best-hparam dicts from the shared JSON shape."""
    module = _import_module_or_fail("src.training.train_final_models")
    payload = {
        "per_fold_results": [
            {"outer_fold": 0, "best_hparams": {"a": 1}},
            {"outer_fold": 1, "best_hparams": {"a": 2}},
            {"outer_fold": 2, "best_hparams": {"a": 3}},
            {"outer_fold": 3, "best_hparams": {"a": 4}},
            {"outer_fold": 4, "best_hparams": {"a": 5}},
        ]
    }

    extracted = module._extract_per_fold_best_hparams(payload)

    assert extracted == [{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}, {"a": 5}]


def test_aggregate_final_hparams_excludes_non_hp_keys_per_e009() -> None:
    """Phase 2 should aggregate only searched HPs and restore canonical plumbing keys."""
    module = _import_module_or_fail("src.training.train_final_models")
    global_config = {"random_seed": 42}
    model_config = {"xgboost_fixed": {"n_jobs": 1, "tree_method": "exact"}}
    per_fold_best_hparams = [
        {
            "float_hp": 0.1,
            "int_hp": 3,
            "cat_hp": "a",
            "bool_hp": True,
            "list_hp": [1, 2],
            "random_state": 42,
            "n_jobs": 8,
            "tree_method": "hist",
        },
        {
            "float_hp": 0.3,
            "int_hp": 5,
            "cat_hp": "b",
            "bool_hp": False,
            "list_hp": [1, 2],
            "random_state": 43,
            "n_jobs": 6,
            "tree_method": "approx",
        },
        {
            "float_hp": 0.2,
            "int_hp": 5,
            "cat_hp": "b",
            "bool_hp": False,
            "list_hp": [3, 4],
            "random_state": 44,
            "n_jobs": 4,
            "tree_method": "hist",
        },
        {
            "float_hp": 0.4,
            "int_hp": 7,
            "cat_hp": "b",
            "bool_hp": False,
            "list_hp": [1, 2],
            "random_state": 45,
            "n_jobs": 2,
            "tree_method": "approx",
        },
        {
            "float_hp": 0.5,
            "int_hp": 9,
            "cat_hp": "c",
            "bool_hp": False,
            "list_hp": [1, 2],
            "random_state": 46,
            "n_jobs": 3,
            "tree_method": "hist",
        },
    ]

    final_hparams, flags = module._aggregate_final_hparams(
        model_name="xgboost",
        per_fold_best_hparams=per_fold_best_hparams,
        global_config=global_config,
        model_config=model_config,
    )

    assert final_hparams == {
        "bool_hp": False,
        "cat_hp": "b",
        "float_hp": 0.3,
        "int_hp": 5,
        "list_hp": [1, 2],
        "random_state": 42,
        "n_jobs": 1,
        "tree_method": "exact",
    }
    assert flags == {
        "bool_hp": False,
        "cat_hp": False,
        "float_hp": False,
        "int_hp": False,
        "list_hp": False,
    }


def test_xgboost_build_estimator_accepts_restored_plumbing_keys(
    xgboost_smoke_fixture: dict[str, Path],
) -> None:
    """XGBoost build_estimator should ignore restored plumbing keys in the splat."""
    module = _import_module_or_fail("src.training.train_xgboost")
    trainer = module.XGBoostTrainer(
        global_config_path=xgboost_smoke_fixture["global_config_path"],
        model_config_path=xgboost_smoke_fixture["model_config_path"],
    )

    final_hparams = {
        "colsample_bytree": 0.7,
        "gamma": 1.0,
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 6,
        "n_estimators": 50,
        "reg_alpha": 0.05,
        "reg_lambda": 0.1,
        "subsample": 0.7,
        "random_state": 42,
        "n_jobs": 1,
        "tree_method": "exact",
    }
    estimator = trainer.build_estimator(hparams=final_hparams, class_weights=None)

    classifier = estimator.named_steps["clf"]
    assert estimator is not None
    assert classifier.n_estimators == 50
    assert classifier.max_depth == 5
    assert classifier.learning_rate == 0.05
    assert classifier.tree_method == "exact"
    assert classifier.n_jobs == 1
    assert classifier.random_state == 42


def test_svm_build_estimator_accepts_restored_plumbing_keys(
    svm_smoke_fixture: dict[str, Path],
) -> None:
    """SVM build_estimator should ignore restored plumbing keys in the splat."""
    module = _import_module_or_fail("src.training.train_svm")
    trainer = module.SVMTrainer(
        global_config_path=svm_smoke_fixture["global_config_path"],
        model_config_path=svm_smoke_fixture["model_config_path"],
    )

    final_hparams = {
        "C": 1.0,
        "class_weight": "balanced",
        "gamma": "scale",
        "kernel": "rbf",
        "random_state": 42,
        "probability": True,
    }
    estimator = trainer.build_estimator(hparams=final_hparams, class_weights=None)

    classifier = estimator.named_steps["clf"]
    svm_params = classifier.get_params()
    assert estimator is not None
    assert svm_params["C"] == 1.0
    assert svm_params["gamma"] == "scale"
    assert svm_params["probability"] is True
    assert svm_params["random_state"] == 42
    assert svm_params["kernel"] == "rbf"
    assert svm_params["class_weight"] == "balanced"


def test_elastic_net_build_estimator_accepts_restored_plumbing_keys(
    elastic_net_lr_smoke_fixture: dict[str, Path],
) -> None:
    """Elastic-net build_estimator should ignore restored plumbing keys in the splat."""
    module = _import_module_or_fail("src.training.train_elastic_net_lr")
    trainer = module.ElasticNetLRTrainer(
        global_config_path=elastic_net_lr_smoke_fixture["global_config_path"],
        model_config_path=elastic_net_lr_smoke_fixture["model_config_path"],
    )

    final_hparams = {
        "C": 1.0,
        "class_weight": "balanced",
        "l1_ratio": 0.9,
        "random_state": 42,
        "max_iter": 5000,
        "penalty": "elasticnet",
        "solver": "saga",
    }
    estimator = trainer.build_estimator(hparams=final_hparams, class_weights=None)

    classifier = estimator.named_steps["clf"]
    lr_params = classifier.get_params()
    assert estimator is not None
    assert lr_params["C"] == 1.0
    assert lr_params["l1_ratio"] == 0.9
    assert lr_params["random_state"] == 42
    assert lr_params["max_iter"] == 5000
    assert lr_params["penalty"] == "elasticnet"
    assert lr_params["solver"] == "saga"
    assert lr_params["class_weight"] == "balanced"


def test_final_pkl_end_to_end_smoke() -> None:
    """Every generated final PKL should support one-sample end-to-end inference."""
    final_pkl_paths = {
        model_name: PROJECT_ROOT / "outputs" / "models" / f"binary_{model_name}_final.pkl"
        for model_name in MODEL_NAMES
    }
    missing = [model_name for model_name, path in final_pkl_paths.items() if not path.exists()]
    if missing:
        pytest.skip(f"Final PKLs are not generated yet: {missing}")

    for model_name, path in final_pkl_paths.items():
        artifact = load_final_model(path)
        assert set(artifact) == EXPECTED_FINAL_PKL_KEYS
        assert set(artifact["metadata"]) == EXPECTED_METADATA_KEYS
        assert artifact["metadata"]["model_name"] == model_name
        assert artifact["metadata"]["oof_source"] in {"nested_cv_reused", "final_hp_rerun"}

        training_sample_ids = np.asarray(artifact["training_sample_ids"], dtype=np.int64)
        assert training_sample_ids.dtype == np.int64
        assert training_sample_ids.ndim == 1
        assert len(training_sample_ids) == 777
        assert np.all(np.diff(training_sample_ids) >= 0)

        if model_name in PIPELINE_A_MODELS:
            assert artifact["feature_names"] == list(GAS_CHANNELS)
            raw_sample = _lookup_pipeline_a_sample(int(training_sample_ids[0]))
            if model_name == "minirocket_ridge":
                assert artifact["transformer"] is not None
                transformed = artifact["transformer"].transform(
                    np.transpose(raw_sample, (0, 2, 1)).astype(np.float32, copy=False)
                )
                transformed_frame = _coerce_feature_frame(
                    transformed,
                    columns=list(artifact["normalizer"].feature_names),
                )
                normalized = artifact["normalizer"].transform(transformed_frame)
            else:
                assert artifact["transformer"] is None
                normalized = artifact["normalizer"].transform(raw_sample)
        else:
            assert model_name in PIPELINE_B_MODELS
            assert artifact["transformer"] is None
            assert len(artifact["feature_names"]) == PIPELINE_B_FEATURE_COUNT
            raw_sample = _lookup_pipeline_b_sample(
                sample_id=int(training_sample_ids[0]),
                feature_names=list(artifact["feature_names"]),
            )
            normalized = artifact["normalizer"].transform(raw_sample)

        p_raw = np.asarray(artifact["model"].predict_proba(normalized))[:, 1]
        p_cal = apply_platt_calibrator(artifact["calibrator"], p_raw)

        assert p_raw.shape == (1,)
        assert p_cal.shape == (1,)
        assert np.isfinite(p_raw).all()
        assert np.isfinite(p_cal).all()
        assert ((0.0 <= p_raw) & (p_raw <= 1.0)).all()
        assert ((0.0 <= p_cal) & (p_cal <= 1.0)).all()
