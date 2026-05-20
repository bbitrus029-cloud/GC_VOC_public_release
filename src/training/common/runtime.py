from __future__ import annotations

import os
from pathlib import Path
import tempfile


def ensure_numba_cache_dir() -> None:
    """Set a writable numba cache directory for MiniRocket runtime compilation."""
    if os.environ.get("NUMBA_CACHE_DIR"):
        return
    cache_dir = Path(tempfile.gettempdir()) / "gcvoc_numba_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)
