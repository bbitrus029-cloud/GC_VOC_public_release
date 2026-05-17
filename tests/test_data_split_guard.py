from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pytest
import yaml

from src.config import PROJECT_ROOT
from src.analysis.data_split import SeedScore, _select_best_seed, generate_and_save_splits


def _write_config(tmp_path: Path, data_split_seed: object, splits_dir: Path | None = None) -> Path:
    """Write a minimal temporary modeling config for guard tests."""
    config_path = tmp_path / "modeling_global.yaml"
    config = {
        "provenance_schema_version": 1,
        "random_seed": 42,
        "data_split_seed": data_split_seed,
        "data_split_seed_candidates": [42, 43, 44, 45, 46],
        "split": {"train_fraction": 0.80},
        "cv": {"outer_folds": 5, "inner_folds": 3},
        "dl": {"holdout_fraction": 0.125},
        "bootstrap": {"n_boot": 2000},
        "calibration": {"method": "sigmoid"},
        "paths": {
            "sample_index": "data/processed/sample_index.csv",
            "splits_dir": str(splits_dir or Path("data/processed/splits")),
            "outputs_analysis_dir": "outputs/analysis",
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def test_generate_and_save_splits_raises_on_tbd_seed(tmp_path: Path) -> None:
    """Unlocked seed should halt before any split work is attempted."""
    config_path = _write_config(tmp_path=tmp_path, data_split_seed="__TBD__")

    with pytest.raises(ValueError, match="__TBD__"):
        generate_and_save_splits(config_path)


def test_generate_and_save_splits_raises_on_integer_outside_candidate_set(tmp_path: Path) -> None:
    """Out-of-set integer seed should be rejected explicitly."""
    config_path = _write_config(tmp_path=tmp_path, data_split_seed=99)

    with pytest.raises(ValueError, match=r"99"):
        generate_and_save_splits(config_path)

    with pytest.raises(ValueError, match=r"\[42, 43, 44, 45, 46\]"):
        generate_and_save_splits(config_path)


def test_generate_and_save_splits_raises_on_non_integer_seed(tmp_path: Path) -> None:
    """Non-integer seed values should be rejected before split generation."""
    config_path = _write_config(tmp_path=tmp_path, data_split_seed="abc")

    with pytest.raises(ValueError):
        generate_and_save_splits(config_path)


def test_generate_and_save_splits_writes_final_dl_holdout(tmp_path: Path) -> None:
    """Locked split generation should persist the Stage 4b final DL holdout."""
    splits_dir = PROJECT_ROOT / ".pytest_tmp" / tmp_path.name / "splits"
    shutil.rmtree(splits_dir.parent, ignore_errors=True)
    config_path = _write_config(tmp_path=tmp_path, data_split_seed=43, splits_dir=splits_dir)
    try:
        generate_and_save_splits(config_path)

        final_holdout_path = splits_dir / "final_dl_es_holdout.npz"
        assert final_holdout_path.exists()

        with np.load(splits_dir / "train_idx.npz", allow_pickle=False) as train_split:
            expected_train_idx = train_split["train_idx"].astype(int)

        with np.load(final_holdout_path, allow_pickle=False) as final_holdout:
            assert set(final_holdout.files) == {"train_idx", "holdout_idx"}
            final_train_idx = final_holdout["train_idx"].astype(int)
            holdout_idx = final_holdout["holdout_idx"].astype(int)

        assert len(final_train_idx) + len(holdout_idx) == len(expected_train_idx) == 777
        assert len(holdout_idx) in {97, 98}
        assert len(np.intersect1d(final_train_idx, holdout_idx)) == 0
        assert np.array_equal(np.sort(np.concatenate([final_train_idx, holdout_idx])), np.sort(expected_train_idx))
    finally:
        shutil.rmtree(splits_dir.parent, ignore_errors=True)


def test_seed_selection_scoring_lexicographic() -> None:
    """Lexicographic seed selection should fall through tied levels correctly."""
    seed_scores = [
        SeedScore(
            seed=42,
            label_smd=0.010000,
            disease_category_smd=0.020000,
            age_smd=0.080000,
            bmi_smd=0.030000,
            sex_smd=0.020000,
            smoking_smd=0.020000,
            drinking_smd=0.010000,
            family_history_smd=0.010000,
            hp_smd=0.010000,
            fallback_used_smd=0.005000,
            padded_smd=0.004000,
        ),
        SeedScore(
            seed=43,
            label_smd=0.0100004,
            disease_category_smd=0.0200004,
            age_smd=0.020000,
            bmi_smd=0.030000,
            sex_smd=0.050000,
            smoking_smd=0.020000,
            drinking_smd=0.010000,
            family_history_smd=0.010000,
            hp_smd=0.010000,
            fallback_used_smd=0.005000,
            padded_smd=0.004000,
        ),
        SeedScore(
            seed=44,
            label_smd=0.0100002,
            disease_category_smd=0.0200002,
            age_smd=0.020000,
            bmi_smd=0.030000,
            sex_smd=0.020000,
            smoking_smd=0.020000,
            drinking_smd=0.010000,
            family_history_smd=0.010000,
            hp_smd=0.010000,
            fallback_used_smd=0.010000,
            padded_smd=0.004000,
        ),
    ]

    selected_score, selected_level = _select_best_seed(seed_scores)

    assert selected_score.seed == 44
    assert selected_level == 4
