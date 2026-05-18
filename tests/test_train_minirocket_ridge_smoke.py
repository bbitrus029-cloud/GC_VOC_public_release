from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


EXPECTED_FEATURE_PREFIXES = ("MPD", "MND", "AUC", "Mean", "MAS", "MDS", "T90", "ERR", "LMR")


@pytest.fixture
def minirocket_smoke_fixture(tmp_path: Path) -> dict[str, Path]:
    """Create a tiny aligned Stage 1 + split fixture for MR+R smoke tests."""
    project_root = tmp_path
    processed_root = project_root / "data" / "processed"
    splits_dir = processed_root / "splits"
    features_dir = processed_root / "features"
    sequences_dir = processed_root / "sequences"
    outputs_dir = project_root / "outputs"
    configs_dir = project_root / "configs"

    splits_dir.mkdir(parents=True)
    features_dir.mkdir(parents=True)
    sequences_dir.mkdir(parents=True)
    outputs_dir.mkdir(parents=True)
    configs_dir.mkdir(parents=True)

    sample_ids = np.array([f"S{index:03d}" for index in range(20)], dtype="U4")
    labels = np.array([index % 2 for index in range(20)], dtype=int)

    sample_index = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "label": labels,
            "in_final_output": True,
        }
    )
    sample_index.to_csv(processed_root / "sample_index.csv", index=False)

    rng = np.random.default_rng(42)
    x_pipeline_a = rng.normal(loc=0.0, scale=1.0, size=(20, 960, 11)).astype(np.float32)
    np.savez_compressed(
        sequences_dir / "pipeline_a_raw.npz",
        x=x_pipeline_a,
        sample_id=sample_ids,
        label=labels,
        channel_names=np.array([f"S{index}" for index in range(1, 12)], dtype="U3"),
        clip_length_L=np.array(960, dtype=int),
    )

    feature_columns: list[str] = []
    feature_values: dict[str, np.ndarray] = {}
    for feature_name in EXPECTED_FEATURE_PREFIXES:
        for sensor_idx in range(1, 12):
            column_name = f"{feature_name}_S{sensor_idx}"
            feature_columns.append(column_name)
            feature_values[column_name] = rng.normal(loc=0.0, scale=1.0, size=20)

    pipeline_b = pd.DataFrame(feature_values, columns=feature_columns)
    pipeline_b["sample_id"] = sample_ids
    pipeline_b["label"] = labels
    pipeline_b.to_csv(features_dir / "pipeline_b_raw.csv", index=False)

    train_idx = np.arange(20, dtype=int)
    test_idx = np.array([], dtype=int)
    np.savez_compressed(splits_dir / "train_test_split.npz", train_idx=train_idx, test_idx=test_idx)
    np.savez_compressed(splits_dir / "train_idx.npz", train_idx=train_idx)
    np.savez_compressed(splits_dir / "test_idx.npz", test_idx=test_idx)

    outer_train_idx = np.arange(0, 16, dtype=int)
    outer_val_idx = np.arange(16, 20, dtype=int)
    inner_train_indices = np.array(
        [
            np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=int),
            np.array([0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15], dtype=int),
        ],
        dtype=object,
    )
    inner_val_indices = np.array(
        [
            np.array([0, 1, 2, 3], dtype=int),
            np.array([4, 5, 6, 7], dtype=int),
        ],
        dtype=object,
    )
    np.savez_compressed(
        splits_dir / "outer_fold_0.npz",
        outer_fold=np.array([0], dtype=int),
        train_idx=outer_train_idx,
        val_idx=outer_val_idx,
        inner_train_indices=inner_train_indices,
        inner_val_indices=inner_val_indices,
    )

    global_config = {
        "random_seed": 42,
        "data_split_seed": 43,
        "cv": {"outer_folds": 1, "inner_folds": 2},
        "bootstrap": {"n_boot": 10},
    }
    global_config_path = configs_dir / "modeling_global.yaml"
    global_config_path.write_text(yaml.safe_dump(global_config), encoding="utf-8")

    model_config = {
        "model_name": "minirocket_ridge",
        "pipeline": "A",
        "input_layout": "NCL",
        "minirocket": {"num_kernels": 256, "random_state": 42},
        "ridge": {"classifier": "RidgeClassifier", "class_weight": "balanced"},
        "calibration_layer_a": {"method": "sigmoid", "cv": 2, "ensemble": True},
        "paths": {
            "nested_cv_oof": str(outputs_dir / "predictions" / "binary_minirocket_ridge_nested_cv_oof.csv"),
            "nested_cv_per_fold": str(outputs_dir / "predictions" / "binary_minirocket_ridge_nested_cv_per_fold.csv"),
            "nested_cv_summary": str(outputs_dir / "metrics" / "binary_minirocket_ridge_nested_cv_summary.csv"),
            "nested_cv_results": str(outputs_dir / "analysis" / "binary_minirocket_ridge_nested_cv_results.json"),
        },
    }
    model_config_path = configs_dir / "modeling_minirocket_ridge.yaml"
    model_config_path.write_text(yaml.safe_dump(model_config), encoding="utf-8")

    return {
        "project_root": project_root,
        "processed_root": processed_root,
        "global_config_path": global_config_path,
        "model_config_path": model_config_path,
        "oof_path": Path(model_config["paths"]["nested_cv_oof"]),
        "per_fold_path": Path(model_config["paths"]["nested_cv_per_fold"]),
        "summary_path": Path(model_config["paths"]["nested_cv_summary"]),
        "results_path": Path(model_config["paths"]["nested_cv_results"]),
    }


def test_train_minirocket_ridge_smoke(minirocket_smoke_fixture: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    """MiniRocket+Ridge trainer should emit finite probabilities on a tiny fixture."""
    from src.training import common as training_common
    from src.training.common import base_trainer
    from src.training import train_minirocket_ridge as trainer_module

    monkeypatch.setattr(base_trainer, "PROCESSED_DATA_ROOT", minirocket_smoke_fixture["processed_root"])
    monkeypatch.setattr(training_common.base_trainer, "PROCESSED_DATA_ROOT", minirocket_smoke_fixture["processed_root"])

    trainer = trainer_module.MiniRocketRidgeTrainer(
        global_config_path=minirocket_smoke_fixture["global_config_path"],
        model_config_path=minirocket_smoke_fixture["model_config_path"],
    )
    trainer.run(outer_fold_limit=1)

    oof = pd.read_csv(minirocket_smoke_fixture["oof_path"])
    assert oof.columns.tolist() == ["sample_id", "y_true", "p_raw", "outer_fold"]
    assert oof["p_raw"].between(0.0, 1.0).all()
    assert np.isfinite(oof["p_raw"]).all()
    assert not oof["p_raw"].isna().any()

    assert minirocket_smoke_fixture["per_fold_path"].exists()
    assert minirocket_smoke_fixture["summary_path"].exists()
    assert minirocket_smoke_fixture["results_path"].exists()
