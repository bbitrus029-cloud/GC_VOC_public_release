from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from src.training.common.runtime import ensure_numba_cache_dir
from src.training.dl_inference import DLPredictProbaWrapper  # noqa: F401


def load_final_model(path: Path | str) -> dict[str, Any]:
    """Load a Stage 4b final PKL with runtime prerequisites configured.

    Args:
        path: Absolute or project-relative path to a `binary_<m>_final.pkl` artifact.

    Returns:
        The deserialized final-model payload.
    """
    ensure_numba_cache_dir()
    return pickle.loads(Path(path).read_bytes())
