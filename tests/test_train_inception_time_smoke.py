from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


@pytest.fixture
def inception_time_smoke_fixture(tmp_path: Path) -> dict[str, Path]:
    """Create a tiny aligned Stage 1 + split fixture for InceptionTime smoke tests."""
    project_root = tmp_path
    processed_root = project_root / "data" / "processed"
    splits_dir = processed_root / "splits"
    sequences_dir = processed_root / "sequences"
    outputs_dir = project_root / "outputs"
    configs_dir = project_root / "configs"

    splits_dir.mkdir(parents=True)
    sequences_dir.mkdir(parents=True)
    outputs_dir.mkdir(parents=True)
    configs_dir.mkdir(parents=True)

    sample_ids = np.arange(20, dtype=np.int64)
    labels = np.array([index % 2 for index in range(20)], dtype=int)
    disease_categories = np.array(
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0, 1, 2, 3, 4, 5, 0, 5],
        dtype=int,
    )

    sample_index = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "label": labels,
            "disease_category": disease_categories,
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

    train_idx = np.arange(20, dtype=int)
    test_idx = np.array([], dtype=int)
    np.savez_compressed(splits_dir / "train_test_split.npz", train_idx=train_idx, test_idx=test_idx)
    np.savez_compressed(splits_dir / "train_idx.npz", train_idx=train_idx)
    np.savez_compressed(splits_dir / "test_idx.npz", test_idx=test_idx)

    outer_train_idx = np.arange(0, 16, dtype=int)
    outer_val_idx = np.arange(16, 20, dtype=int)
    inner_train_indices = np.array(
        [
            np.array([4, 5, 6, 7, 8, 9, 10, 11], dtype=int),
        ],
        dtype=object,
    )
    inner_val_indices = np.array(
        [
            np.array([12, 13, 14, 15], dtype=int),
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
    np.savez_compressed(
        splits_dir / "dl_holdout_0.npz",
        train_idx=np.arange(0, 14, dtype=int),
        holdout_idx=np.arange(14, 16, dtype=int),
    )

    global_config = {
        "random_seed": 42,
        "data_split_seed": 43,
        "cv": {"outer_folds": 1, "inner_folds": 1},
        "bootstrap": {"n_boot": 10},
    }
    global_config_path = configs_dir / "modeling_global.yaml"
    global_config_path.write_text(yaml.safe_dump(global_config), encoding="utf-8")

    model_config = {
        "provenance": {
            "operational_plan": ".cursor/plans/modeling_plan_v1_a7c3f9b2_dc789bb5.plan.md",
            "governing_sections": ["§2.2", "§2.7", "§2.23", "§6.2"],
            "errata": ["E004"],
        },
        "model_name": "inception_time",
        "pipeline": "A",
        "input_layout": "NCL",
        "random_seed": 42,
        "class_imbalance": {"mechanism": "dl_two_class_weighted_loss"},
        "hyperparameter_candidates": [
            {
                "depth": 3,
                "n_filters": 8,
                "kernel_sizes": [9, 19, 39],
                "dropout": 0.1,
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "batch_size": 4,
                "bottleneck_channels": 8,
            }
        ],
        "training": {
            "max_epochs": 2,
            "precision": "32-true",
            "deterministic": True,
            "accelerator": "cpu",
            "devices": 1,
            "early_stopping": {"monitor": "val_auc", "mode": "max", "patience": 1},
            "csv_logger": {"path": "outputs/logs/lightning_csv/binary_inception_time"},
            "nan_retry": {"enabled": True, "lr_halving": True, "max_retries": 1},
        },
        "paths": {
            "nested_cv_oof": str(outputs_dir / "predictions" / "binary_inception_time_nested_cv_oof.csv"),
            "nested_cv_per_fold": str(outputs_dir / "predictions" / "binary_inception_time_nested_cv_per_fold.csv"),
            "nested_cv_summary": str(outputs_dir / "metrics" / "binary_inception_time_nested_cv_summary.csv"),
            "nested_cv_results": str(outputs_dir / "analysis" / "binary_inception_time_nested_cv_results.json"),
            "log": str(outputs_dir / "logs" / "train_inception_time.log"),
        },
    }
    model_config_path = configs_dir / "modeling_inception_time.yaml"
    model_config_path.write_text(yaml.safe_dump(model_config, sort_keys=False), encoding="utf-8")

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


def test_train_inception_time_smoke(
    inception_time_smoke_fixture: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """InceptionTime trainer should emit finite probabilities and DL audit fields."""
    from src.training import common as training_common
    from src.training.common import base_trainer
    from src.training import train_inception_time as trainer_module

    monkeypatch.setattr(base_trainer, "PROCESSED_DATA_ROOT", inception_time_smoke_fixture["processed_root"])
    monkeypatch.setattr(training_common.base_trainer, "PROCESSED_DATA_ROOT", inception_time_smoke_fixture["processed_root"])

    trainer = trainer_module.InceptionTimeTrainer(
        global_config_path=inception_time_smoke_fixture["global_config_path"],
        model_config_path=inception_time_smoke_fixture["model_config_path"],
    )
    trainer.run(outer_fold_limit=1)

    oof = pd.read_csv(inception_time_smoke_fixture["oof_path"])
    assert oof.columns.tolist() == ["sample_id", "y_true", "p_raw", "outer_fold"]
    assert oof["p_raw"].between(0.0, 1.0).all()
    assert np.isfinite(oof["p_raw"]).all()
    assert not oof["p_raw"].isna().any()

    per_fold = pd.read_csv(inception_time_smoke_fixture["per_fold_path"])
    best_hparams = json.loads(per_fold.loc[0, "best_hparams_json"])
    assert best_hparams["depth"] == 3
    assert best_hparams["batch_size"] == 4

    summary = pd.read_csv(inception_time_smoke_fixture["summary_path"])
    assert not summary.empty

    results = json.loads(inception_time_smoke_fixture["results_path"].read_text(encoding="utf-8"))
    for field_name in [
        "per_fold_best_epoch",
        "per_fold_early_stopping_triggered",
        "per_fold_nan_retry_count",
        "per_fold_failed_status",
        "posthoc_G_trigger_check",
    ]:
        assert field_name in results
