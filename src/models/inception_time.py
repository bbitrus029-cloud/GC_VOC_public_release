from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics.classification import AUROC


def _compute_brier_score(probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return the mean squared error between probabilities and binary labels."""
    return torch.mean((probabilities - targets.float()) ** 2)


class _InceptionBlock(nn.Module):
    """Apply parallel temporal convolutions with a pooled shortcut branch.

    Args:
        in_channels: Number of input channels.
        n_filters: Number of filters per convolution branch.
        kernel_sizes: Odd convolution kernel sizes for the three branches.
        bottleneck_channels: Bottleneck width used before the multi-scale branches.
    """

    def __init__(
        self,
        in_channels: int,
        n_filters: int,
        kernel_sizes: tuple[int, int, int],
        bottleneck_channels: int,
    ) -> None:
        super().__init__()
        branch_in_channels = in_channels
        if in_channels > 1:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            branch_in_channels = bottleneck_channels
        else:
            self.bottleneck = nn.Identity()

        self.conv_branches = nn.ModuleList(
            [
                nn.Conv1d(
                    branch_in_channels,
                    n_filters,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False,
                )
                for kernel_size in kernel_sizes
            ]
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False),
        )
        self.batch_norm = nn.BatchNorm1d(num_features=n_filters * 4)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform a `[B, C, L]` tensor into concatenated inception features.

        Args:
            x: Input tensor shaped `[batch, channels, length]`.

        Returns:
            Concatenated block output shaped `[batch, 4 * n_filters, length]`.
        """
        bottleneck_x = self.bottleneck(x)
        branches = [conv(bottleneck_x) for conv in self.conv_branches]
        branches.append(self.pool_branch(x))
        concatenated = torch.cat(branches, dim=1)
        return self.activation(self.batch_norm(concatenated))


class _ResidualProjection(nn.Module):
    """Project a residual path to the current feature width.

    Args:
        in_channels: Residual input width.
        out_channels: Residual output width.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project the residual branch with a pointwise convolution.

        Args:
            x: Input tensor shaped `[batch, channels, length]`.

        Returns:
            Residual tensor aligned to the current feature width.
        """
        return self.projection(x)


class InceptionTimeClassifier(pl.LightningModule):
    """InceptionTime classifier for Pipeline A binary prediction.

    This small-cohort variant follows the InceptionTime pattern from Fawaz et al.
    (Data Mining and Knowledge Discovery, 2020) with conservative defaults that
    keep synthetic Session 4 tests lightweight while preserving the required
    `[B, 11, 960]` input geometry.

    Args:
        n_channels: Number of input gas channels.
        n_classes: Number of output classes.
        hparams: Hyperparameter dictionary for the architecture and optimizer.
    """

    DEFAULT_HPARAMS: dict[str, Any] = {
        "n_filters": 16,
        "depth": 3,
        "kernel_sizes": (9, 19, 39),
        "bottleneck_channels": 8,
        "use_residual": True,
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
        kernel_sizes = tuple(int(size) for size in merged_hparams["kernel_sizes"])
        if len(kernel_sizes) != 3:
            raise ValueError("kernel_sizes must contain exactly three entries.")

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

        n_filters = int(merged_hparams["n_filters"])
        bottleneck_channels = int(merged_hparams["bottleneck_channels"])
        depth = int(merged_hparams["depth"])
        use_residual = bool(merged_hparams["use_residual"])

        blocks: list[_InceptionBlock] = []
        residual_projections: list[nn.Module] = []
        current_channels = n_channels
        residual_source_channels = n_channels
        for block_index in range(depth):
            block = _InceptionBlock(
                in_channels=current_channels,
                n_filters=n_filters,
                kernel_sizes=kernel_sizes,
                bottleneck_channels=bottleneck_channels,
            )
            blocks.append(block)
            next_channels = n_filters * 4
            if use_residual and block_index % 3 == 2:
                residual_projections.append(_ResidualProjection(residual_source_channels, next_channels))
                residual_source_channels = next_channels
            else:
                residual_projections.append(nn.Identity())
            current_channels = next_channels

        self.blocks = nn.ModuleList(blocks)
        self.residual_projections = nn.ModuleList(residual_projections)
        self.use_residual = use_residual
        self.global_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.classifier = nn.Linear(current_channels, n_classes)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits for a `[B, 11, 960]` input tensor.

        Args:
            x: Input tensor shaped `[batch, channels, length]`.

        Returns:
            Logits shaped `[batch, n_classes]`.
        """
        if x.ndim != 3:
            raise ValueError("Expected x to have shape [batch, channels, length].")

        residual_source = x
        current = x
        for block_index, block in enumerate(self.blocks):
            current = block(current)
            if self.use_residual and block_index % 3 == 2:
                residual = self.residual_projections[block_index](residual_source)
                current = torch.relu(current + residual)
                residual_source = current

        pooled = self.global_pool(current).squeeze(-1)
        return self.classifier(self.dropout(pooled))

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
