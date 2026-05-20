from __future__ import annotations

from typing import Any, Callable

import numpy as np


class DLPredictProbaWrapper:
    """Pickleable CPU-safe wrapper exposing sklearn-like `predict_proba`."""

    def __init__(
        self,
        lightning_model: Any,
        predictor: Callable[[Any, np.ndarray, int, str], np.ndarray],
        batch_size: int,
        accelerator: str,
    ) -> None:
        self.lightning_model = lightning_model.to("cpu")
        self.predictor = predictor
        self.batch_size = batch_size
        self.accelerator = accelerator

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return a two-column probability matrix for binary classification."""
        p_positive = self.predictor(
            model=self.lightning_model,
            x_eval=x,
            batch_size=self.batch_size,
            accelerator=self.accelerator,
        )
        return np.column_stack([1.0 - p_positive, p_positive]).astype(float, copy=False)
