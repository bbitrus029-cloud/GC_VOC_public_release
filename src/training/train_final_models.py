"""Phase 2 final-model training entry points.

Operational plan §2.5 OOF reuse-or-rerun rule:

- "If every hyperparameter equals the aggregated final hyperparameter in all 5 folds, reuse
  `binary_<m>_nested_cv_oof.csv` and set `oof_source = "nested_cv_reused"`."
- "Otherwise run one-time 5-fold CV on the 777 training samples using final hyperparameters
  and the existing outer fold indices, save `binary_<m>_final_hp_oof.csv`, and set
  `oof_source = "final_hp_rerun"`."
- "The OOF predictions used for Youden threshold and Platt calibrator fitting come from the
  5 outer-fold models ... They do NOT come from the final model itself, which is refit on
  the entire 777 training set."
"""

from __future__ import annotations

from pathlib import Path


def train_final_model(model_name: str, global_config_path: Path, model_config_path: Path) -> None:
    """Train one final model artifact in Session 10 Phase 2.

    Args:
        model_name: Canonical model name.
        global_config_path: Path to `configs/modeling_global.yaml`.
        model_config_path: Path to the model-specific YAML config.

    Raises:
        NotImplementedError: Always in Session 10 Phase 1.
    """
    raise NotImplementedError("Filled in Session 10 Phase 2")


def run_all_final_models(global_config_path: Path) -> None:
    """Train all final model artifacts in Session 10 Phase 2.

    Args:
        global_config_path: Path to `configs/modeling_global.yaml`.

    Raises:
        NotImplementedError: Always in Session 10 Phase 1.
    """
    raise NotImplementedError("Filled in Session 10 Phase 2")
