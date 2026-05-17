from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics.classification import AUROC


def _compute_brier_score(probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return the mean squared error between probabilities and binary labels."""
    return torch.mean((probabilities - targets.float()) ** 2)


class _AttentionPooling(nn.Module):
    """Aggregate temporal features with learned scalar attention.

    Args:
        input_dim: Feature width of each time step.
        attention_dim: Hidden width of the attention scoring network.
    """

    def __init__(self, input_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, sequence_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool a `[B, T, F]` tensor using normalized attention weights.

        Args:
            sequence_features: Temporal feature tensor shaped `[batch, time, features]`.

        Returns:
            Tuple of pooled features `[batch, features]` and raw attention weights
            `[batch, time]`.
        """
        scores = self.attention(sequence_features).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(sequence_features * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class CNNBiLSTMAttentionClassifier(pl.LightningModule):
    """CNN-BiLSTM-Attention classifier for Pipeline A binary prediction.

    Per operational plan §2.21, attention weights are architectural sanity output
    only; they are persisted in Session 9 as a supplementary artifact and are NOT
    included in the cross-model Kendall Tau consistency matrix.

    Args:
        n_channels: Number of input gas channels.
        n_classes: Number of output classes.
        hparams: Hyperparameter dictionary for the architecture and optimizer.
    """

    DEFAULT_HPARAMS: dict[str, Any] = {
        "cnn_channels": (16, 32),
        "cnn_kernel_size": 5,
        "lstm_hidden": 16,
        "lstm_layers": 1,
        "attention_dim": 16,
        "dropout": 0.10,
        "lr": 1e-3,
        "weight_decay": 1e-4,
    }

    def __init__(self, n_channels: int, n_classes: int, hparams: dict[str, Any]) -> None:
        super().__init__()
        if n_channels <= 0:
            raise ValueError("n_channels must be positive.")
        if n_classes <= 1:
            raise ValueError("n_classes must be greater than 1.")

        merged_hparams = {**self.DEFAULT_HPARAMS, **hparams}
        cnn_channels = tuple(int(channel) for channel in merged_hparams["cnn_channels"])
        if len(cnn_channels) == 0:
            raise ValueError("cnn_channels must include at least one convolution width.")

        self.save_hyperparameters(
            {
                "n_channels": n_channels,
                "n_classes": n_classes,
                "hparams": merged_hparams,
            }
        )
        self.n_classes = n_classes
        self.lr = float(merged_hparams["lr"])
        self.weight_decay = float(merged_hparams["weight_decay"])
        self.dropout = nn.Dropout(p=float(merged_hparams["dropout"]))

        kernel_size = int(merged_hparams["cnn_kernel_size"])
        conv_layers: list[nn.Module] = []
        in_channels = n_channels
        for out_channels in cnn_channels:
            conv_layers.extend(
                [
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                ]
            )
            in_channels = out_channels
        self.cnn = nn.Sequential(*conv_layers)

        lstm_hidden = int(merged_hparams["lstm_hidden"])
        lstm_layers = int(merged_hparams["lstm_layers"])
        self.bilstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=float(merged_hparams["dropout"]) if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attention_pool = _AttentionPooling(
            input_dim=lstm_hidden * 2,
            attention_dim=int(merged_hparams["attention_dim"]),
        )
        self.classifier = nn.Linear(lstm_hidden * 2, n_classes)
        self.val_auc_metric = AUROC(task="binary")
        self.register_buffer("_class_weights_buffer", torch.empty(0, dtype=torch.float32), persistent=False)

    @property
    def class_weights(self) -> torch.Tensor | None:
        """Return fold-specific class weights when available.

        Returns:
            The buffered class-weight tensor, or `None` when uniform weighting is
            requested for synthetic or unset cases.
        """
        if self._class_weights_buffer.numel() == 0:
            return None
        return self._class_weights_buffer

    @class_weights.setter
    def class_weights(self, value: torch.Tensor | None) -> None:
        """Update the buffered class-weight tensor.

        Args:
            value: New class-weight tensor or `None` to disable weighting.
        """
        if value is None:
            self._class_weights_buffer = torch.empty(0, device=self.device, dtype=torch.float32)
            return
        if value.ndim != 1 or value.numel() != self.n_classes:
            raise ValueError("class_weights must be a 1D tensor with one weight per class.")
        self._class_weights_buffer = value.detach().to(device=self.device, dtype=torch.float32)

    def _extract_sequence_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the CNN and BiLSTM backbone before attention pooling.

        Args:
            x: Input tensor shaped `[batch, channels, length]`.

        Returns:
            Tuple of BiLSTM features `[batch, time, hidden*2]` and raw attention
            weights `[batch, time]`.
        """
        if x.ndim != 3:
            raise ValueError("Expected x to have shape [batch, channels, length].")
        cnn_features = self.cnn(x)
        temporal_features = cnn_features.transpose(1, 2)
        lstm_features, _ = self.bilstm(temporal_features)
        return self.attention_pool(lstm_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits for a `[B, 11, 960]` input tensor.

        Args:
            x: Input tensor shaped `[batch, channels, length]`.

        Returns:
            Logits shaped `[batch, n_classes]`.
        """
        pooled_features, _ = self._extract_sequence_features(x)
        return self.classifier(self.dropout(pooled_features))

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw attention weights for architectural sanity inspection.

        Args:
            x: Input tensor shaped `[batch, channels, length]`.

        Returns:
            Attention weights shaped `[batch, time]`.
        """
        _, attention_weights = self._extract_sequence_features(x)
        return attention_weights

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Compute and log the training loss for one mini-batch.

        Args:
            batch: Tuple of input tensor and integer labels.
            batch_idx: Unused Lightning batch index.

        Returns:
            Cross-entropy loss for the batch.
        """
        del batch_idx
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss(weight=self.class_weights)(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Compute and log validation loss plus threshold-free metrics.

        Args:
            batch: Tuple of input tensor and integer labels.
            batch_idx: Unused Lightning batch index.

        Returns:
            Cross-entropy loss for the validation batch.
        """
        del batch_idx
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss(weight=self.class_weights)(logits, y)
        probabilities = torch.softmax(logits, dim=1)[:, 1]
        val_auc = self.val_auc_metric(probabilities, y)
        val_brier = _compute_brier_score(probabilities, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_auc", val_auc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_brier", val_brier, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Build the optimizer for Session 8+ trainer wiring.

        Returns:
            AdamW optimizer configured from the hyperparameter dictionary.
        """
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
