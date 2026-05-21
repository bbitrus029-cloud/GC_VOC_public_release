from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest


def _import_symbols() -> Any:
    """Import representative-sample helpers lazily for clear failures."""
    try:
        from src.analysis.shap_cross_model import _save_representative_samples_shap_npz
    except (AttributeError, ImportError, ModuleNotFoundError) as exc:
        pytest.fail(f"shap_cross_model implementation is missing: {exc}", pytrace=False)

    return _save_representative_samples_shap_npz


def test_representative_samples_shap_npz_preserves_reference_order(tmp_path: Path) -> None:
    """The SHAP NPZ should keep sample_id and quadrant aligned 1:1 with the signal NPZ."""
    save_npz = _import_symbols()

    reference_path = tmp_path / "representative_signals.npz"
    output_path = tmp_path / "representative_shap.npz"
    reference_sample_ids = np.asarray([101, 202], dtype=np.int64)
    reference_quadrants = np.asarray(["FN", "TP"], dtype="U2")
    reference_signals = np.zeros((2, 960, 11), dtype=np.float32)
    raw_shap = np.asarray(
        [
            np.arange(1, 12, dtype=float),
            np.arange(11, 0, -1, dtype=float),
        ]
    )

    np.savez_compressed(
        reference_path,
        sample_id=reference_sample_ids,
        quadrant=reference_quadrants,
        signal=reference_signals,
    )

    save_npz(reference_signal_path=reference_path, output_path=output_path, shap_values=raw_shap)

    written = np.load(output_path, allow_pickle=True)
    assert np.array_equal(written["sample_id"], reference_sample_ids)
    assert np.array_equal(written["quadrant"].astype(str), reference_quadrants.astype(str))
    assert written["shap"].shape == (2, 11)
    assert np.allclose(written["shap"].sum(axis=1), np.ones(2))
