from __future__ import annotations

import numpy as np
import pytorch_lightning as pl
import pytest
import torch

from src.models import CNNBiLSTMAttentionClassifier, InceptionTimeClassifier, VOCDataModule


pl.seed_everything(42, workers=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("high")


@pytest.fixture
def batch_fixture() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a synthetic batch with the Session 4 tensor geometry."""
    x = torch.randn(4, 11, 960)
    y = torch.randint(0, 2, (4,))
    return x, y


@pytest.fixture
def inception_hparams() -> dict[str, object]:
    """Return conservative InceptionTime hyperparameters for shape tests."""
    return {
        "n_filters": 16,
        "depth": 3,
        "kernel_sizes": [9, 19, 39],
        "bottleneck_channels": 8,
        "use_residual": True,
        "dropout": 0.10,
        "lr": 1e-3,
        "weight_decay": 1e-4,
    }


@pytest.fixture
def cbla_hparams() -> dict[str, object]:
    """Return conservative CNN-BiLSTM-Attention hyperparameters for shape tests."""
    return {
        "cnn_channels": [16, 32],
        "cnn_kernel_size": 5,
        "lstm_hidden": 16,
        "lstm_layers": 1,
        "attention_dim": 16,
        "dropout": 0.10,
        "lr": 1e-3,
        "weight_decay": 1e-4,
    }


def test_inception_time_forward_returns_finite_logits(
    batch_fixture: tuple[torch.Tensor, torch.Tensor],
    inception_hparams: dict[str, object],
) -> None:
    """InceptionTime forward pass should produce finite binary logits."""
    x, _ = batch_fixture
    model = InceptionTimeClassifier(n_channels=11, n_classes=2, hparams=inception_hparams)

    logits = model(x)

    assert logits.shape == (4, 2)
    assert torch.isfinite(logits).all()


def test_cnn_bilstm_attention_forward_returns_finite_logits(
    batch_fixture: tuple[torch.Tensor, torch.Tensor],
    cbla_hparams: dict[str, object],
) -> None:
    """CNN-BiLSTM-Attention forward pass should produce finite binary logits."""
    x, _ = batch_fixture
    model = CNNBiLSTMAttentionClassifier(n_channels=11, n_classes=2, hparams=cbla_hparams)

    logits = model(x)

    assert logits.shape == (4, 2)
    assert torch.isfinite(logits).all()


def test_cnn_bilstm_attention_returns_raw_attention_weights(
    batch_fixture: tuple[torch.Tensor, torch.Tensor],
    cbla_hparams: dict[str, object],
) -> None:
    """Attention helper should expose a finite [B, T] tensor only."""
    x, _ = batch_fixture
    model = CNNBiLSTMAttentionClassifier(n_channels=11, n_classes=2, hparams=cbla_hparams)

    attention_weights = model.get_attention_weights(x)

    assert attention_weights.shape[0] == 4
    assert len(attention_weights.shape) == 2
    assert attention_weights.shape[1] > 0
    assert torch.isfinite(attention_weights).all()


@pytest.mark.parametrize("model_class,hparams_fixture", [
    (InceptionTimeClassifier, "inception_hparams"),
    (CNNBiLSTMAttentionClassifier, "cbla_hparams"),
])
def test_configure_optimizers_returns_optimizer(
    request: pytest.FixtureRequest,
    model_class: type[pl.LightningModule],
    hparams_fixture: str,
) -> None:
    """Both classifiers should expose a valid optimizer configuration."""
    hparams = request.getfixturevalue(hparams_fixture)
    model = model_class(n_channels=11, n_classes=2, hparams=hparams)

    optimizers = model.configure_optimizers()

    if isinstance(optimizers, tuple):
        optimizer = optimizers[0]
    elif isinstance(optimizers, list):
        optimizer = optimizers[0]
    else:
        optimizer = optimizers

    assert isinstance(optimizer, torch.optim.Optimizer)


def test_voc_data_module_transposes_batches_and_is_seed_deterministic() -> None:
    """DataModule should transpose [N, L, C] to [B, C, L] and preserve seeded order."""
    rng = np.random.default_rng(42)
    x_np = rng.normal(size=(8, 960, 11)).astype(np.float32)
    y_np = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    train_idx = np.arange(6, dtype=np.int64)
    val_idx = np.array([6, 7], dtype=np.int64)

    data_module_a = VOCDataModule(
        x=x_np,
        y=y_np,
        train_idx=train_idx,
        val_idx=val_idx,
        batch_size=2,
        seed=42,
    )
    data_module_b = VOCDataModule(
        x=x_np,
        y=y_np,
        train_idx=train_idx,
        val_idx=val_idx,
        batch_size=2,
        seed=42,
    )

    data_module_a.setup()
    data_module_b.setup()

    first_train_batch_a = next(iter(data_module_a.train_dataloader()))
    first_train_batch_b = next(iter(data_module_b.train_dataloader()))

    batch_x_a, batch_y_a = first_train_batch_a
    batch_x_b, batch_y_b = first_train_batch_b

    assert batch_x_a.shape == (2, 11, 960)
    assert torch.equal(batch_x_a, batch_x_b)
    assert torch.equal(batch_y_a, batch_y_b)


def test_inception_time_weight_initialization_is_deterministic(
    batch_fixture: tuple[torch.Tensor, torch.Tensor],
    inception_hparams: dict[str, object],
) -> None:
    """Resetting the global seed should reproduce identical logits."""
    x, _ = batch_fixture

    pl.seed_everything(42, workers=True)
    model_a = InceptionTimeClassifier(n_channels=11, n_classes=2, hparams=inception_hparams)

    pl.seed_everything(42, workers=True)
    model_b = InceptionTimeClassifier(n_channels=11, n_classes=2, hparams=inception_hparams)

    model_a.eval()
    model_b.eval()

    with torch.no_grad():
        logits_a = model_a(x)
        logits_b = model_b(x)

    assert torch.allclose(logits_a, logits_b)
