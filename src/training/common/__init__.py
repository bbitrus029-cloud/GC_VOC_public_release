"""Common training utilities shared across nested-CV model trainers."""

from src.training.common.base_trainer import BaseNestedCVTrainer
from src.training.common.bootstrap import BootstrapResult, bootstrap_metric, paired_bootstrap_delta
from src.training.common.early_stopping import create_dl_holdout_indices, save_dl_holdout
from src.training.common.fold_generator import FoldGenerator, FoldIndices
from src.training.common.metrics import MetricsCalculator
from src.training.common.normalizer import (
    PipelineANormalizer,
    PipelineBNormalizer,
    fit_pipeline_a_normalizer,
    fit_pipeline_b_normalizer,
)


__all__ = [
    "BaseNestedCVTrainer",
    "BootstrapResult",
    "FoldGenerator",
    "FoldIndices",
    "MetricsCalculator",
    "PipelineANormalizer",
    "PipelineBNormalizer",
    "bootstrap_metric",
    "create_dl_holdout_indices",
    "fit_pipeline_a_normalizer",
    "fit_pipeline_b_normalizer",
    "paired_bootstrap_delta",
    "save_dl_holdout",
]
