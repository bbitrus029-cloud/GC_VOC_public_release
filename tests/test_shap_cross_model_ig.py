from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import pytest


def _import_symbols() -> Any:
    """Import IG helpers lazily for clear failures."""
    try:
        from src.analysis.shap_cross_model import _attribute_with_optional_cudnn_guard
    except (AttributeError, ImportError, ModuleNotFoundError) as exc:
        pytest.fail(f"shap_cross_model IG helper is missing: {exc}", pytrace=False)

    return _attribute_with_optional_cudnn_guard


class _DummyIntegratedGradients:
    """Capture the attribution call without invoking Captum internals."""

    def __init__(self) -> None:
        self.called = False

    def attribute(self, batch: torch.Tensor, *, target: int, n_steps: int) -> torch.Tensor:
        self.called = True
        return torch.ones_like(batch)


def test_attribute_with_optional_cudnn_guard_disables_cudnn_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    """The helper should wrap IG attribution in `cudnn.flags(enabled=False)` when requested."""
    attribute_with_guard = _import_symbols()
    ig = _DummyIntegratedGradients()
    batch = torch.ones((2, 11, 8), dtype=torch.float32)
    recorded: dict[str, object] = {"enabled": None, "entered": False}

    class _Guard:
        def __enter__(self) -> None:
            recorded["entered"] = True

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            return None

    def _fake_flags(*, enabled: bool) -> _Guard:
        recorded["enabled"] = enabled
        return _Guard()

    monkeypatch.setattr(torch.backends.cudnn, "flags", _fake_flags)

    output = attribute_with_guard(ig, batch, target=1, n_steps=50, disable_cudnn=True)

    assert ig.called is True
    assert recorded["entered"] is True
    assert recorded["enabled"] is False
    assert output.shape == batch.shape


def test_attribute_with_optional_cudnn_guard_skips_guard_when_not_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    """The helper should call IG directly when no CuDNN guard is needed."""
    attribute_with_guard = _import_symbols()
    ig = _DummyIntegratedGradients()
    batch = torch.ones((2, 11, 8), dtype=torch.float32)
    called = {"flags": False}

    def _unexpected_flags(*, enabled: bool) -> nullcontext[None]:
        called["flags"] = True
        return nullcontext()

    monkeypatch.setattr(torch.backends.cudnn, "flags", _unexpected_flags)

    output = attribute_with_guard(ig, batch, target=1, n_steps=50, disable_cudnn=False)

    assert ig.called is True
    assert called["flags"] is False
    assert output.shape == batch.shape
