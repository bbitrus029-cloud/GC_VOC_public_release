"""Inference for the audited public model bundle; no fitting or source-data access."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
FAMILIES = (
    "cnn_bilstm_attention", "elastic_net_lr", "inception_time", "minirocket_ridge", "svm", "xgboost",
)

def load_model(model_name: str, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Load only a model authenticated by the release-contained model index.

    Args:
        model_name: A key in MODEL_INDEX.json.
        project_root: Root containing the public package.

    Returns:
        The verified fitted model envelope.

    Raises:
        ValueError: If the model identity, file path, or checksum is invalid.
    """
    index = json.loads((project_root / "MODEL_INDEX.json").read_text())
    if model_name not in index["models"]:
        raise ValueError("Unknown release model")
    item = index["models"][model_name]
    path = project_root / item["path"]
    if not path.resolve().is_relative_to(project_root.resolve()) or path.is_symlink():
        raise ValueError("Unsafe model path")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != item["sha256"]:
        raise ValueError("Model checksum mismatch")
    os.environ.setdefault("NUMBA_CACHE_DIR", str(project_root / ".release-runtime" / "numba"))
    return pickle.loads(content)

def predict_scores(
    bundle: dict[str, Any], inputs: np.ndarray | pd.DataFrame, deep_runtime: str = "stored",
) -> dict[str, np.ndarray | None]:
    """Apply frozen inference, calibration, and the existing raw-score threshold.

    Args:
        bundle: A provenance-verified fitted model envelope.
        inputs: Processed signals, ordered features, or marker/fusion feature values.
        deep_runtime: Stored wrapper behavior, or the recorded frozen_gpu inference contract.

    Returns:
        Raw scores, calibrated scores, and frozen-threshold decisions if available.

    Raises:
        ValueError: If input dimensions, class mapping, or model outputs are invalid.
        TypeError: If a signal model is given a feature table or vice versa.
    """
    name = bundle["metadata"]["model_name"]
    features = list(bundle["feature_names"])
    if name == "minirocket_ridge":
        if not isinstance(inputs, np.ndarray) or inputs.ndim != 3:
            raise TypeError("MiniRocket requires a processed N by L by C array")
        transformed = bundle["transformer"].transform(
            np.transpose(inputs, (0, 2, 1)).astype(np.float32, copy=False)
        )
        matrix = transformed.to_numpy() if isinstance(transformed, pd.DataFrame) else np.asarray(transformed)
        frame = pd.DataFrame(matrix, columns=list(bundle["normalizer"].feature_names))
        normalized = bundle["normalizer"].transform(frame)
    elif name in {"cnn_bilstm_attention", "inception_time"}:
        if not isinstance(inputs, np.ndarray) or inputs.ndim != 3 or inputs.shape[-1] != len(features):
            raise TypeError("Deep models require a processed N by L by C array")
        normalized = bundle["normalizer"].transform(inputs)
    else:
        if not isinstance(inputs, pd.DataFrame) or list(inputs.columns) != features:
            raise TypeError("Feature inputs must match the saved feature order exactly")
        if bundle["normalizer"] is not None:
            normalized = bundle["normalizer"].transform(inputs)
        else:
            normalized = inputs.to_numpy(dtype=float)
    if deep_runtime not in {"stored", "frozen_gpu"}:
        raise ValueError("Unknown deep inference runtime")
    if deep_runtime == "frozen_gpu" and name in {"cnn_bilstm_attention", "inception_time"}:
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("The recorded GPU contract requires one CUDA device")
        wrapper = bundle["model"]
        if wrapper.batch_size != 32 or wrapper.accelerator != "cpu":
            raise ValueError("Saved wrapper does not match the recorded inference contract")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("high")
        positive = wrapper.predictor(
            model=wrapper.lightning_model, x_eval=normalized, batch_size=32, accelerator="gpu"
        )
        probabilities = np.column_stack([1.0 - positive, positive]).astype(float, copy=False)
    else:
        probabilities = np.asarray(bundle["model"].predict_proba(normalized), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("Expected two-class probability output")
    classes = getattr(bundle["model"], "classes_", np.array([0, 1]))
    if not np.array_equal(classes, [0, 1]):
        raise ValueError("Unexpected positive-class mapping")
    raw = probabilities[:, 1]
    calibrated = np.asarray(bundle["calibrator"].predict_proba(raw.reshape(-1, 1))[:, 1], dtype=float)
    if not np.isfinite(raw).all() or not np.isfinite(calibrated).all():
        raise ValueError("Nonfinite model output")
    if np.any((raw < 0) | (raw > 1) | (calibrated < 0) | (calibrated > 1)):
        raise ValueError("Model scores outside the probability scale")
    threshold = bundle["threshold_p_raw"]
    decisions = None if threshold is None else (raw >= float(threshold)).astype(np.int64)
    return {"p_raw": raw, "p_cal": calibrated, "y_pred": decisions}
