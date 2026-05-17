from __future__ import annotations

import logging

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset


LOGGER = logging.getLogger(__name__)


class VOCDataModule(pl.LightningDataModule):
    """Lightning DataModule for deterministic VOC train/validation batches.

    Args:
        x: Input array shaped `[N, L, C]`.
        y: Label array shaped `[N]`.
        train_idx: Training-row indices.
        val_idx: Validation-row indices.
        batch_size: Mini-batch size.
        seed: Random seed for deterministic train-loader shuffling.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        batch_size: int,
        seed: int,
    ) -> None:
        super().__init__()
        if x.ndim != 3:
            raise ValueError("x must have shape [N, L, C].")
        if y.ndim != 1:
            raise ValueError("y must have shape [N].")
        if len(x) != len(y):
            raise ValueError("x and y must contain the same number of samples.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self.x = x
        self.y = y
        self.train_idx = np.asarray(train_idx, dtype=np.int64)
        self.val_idx = np.asarray(val_idx, dtype=np.int64)
        self.batch_size = batch_size
        self.seed = seed
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.train_dataset: TensorDataset | None = None
        self.val_dataset: TensorDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Prepare transposed train and validation datasets.

        Args:
            stage: Optional Lightning stage string, unused in Session 4.
        """
        del stage
        x_train = np.transpose(self.x[self.train_idx], (0, 2, 1)).astype(np.float32, copy=False)
        x_val = np.transpose(self.x[self.val_idx], (0, 2, 1)).astype(np.float32, copy=False)
        y_train = self.y[self.train_idx].astype(np.int64, copy=False)
        y_val = self.y[self.val_idx].astype(np.int64, copy=False)

        LOGGER.info(
            "VOCDataModule transposed train/val tensors from [N, L, C] to [N, C, L]: "
            "train %s -> %s, val %s -> %s",
            self.x[self.train_idx].shape,
            x_train.shape,
            self.x[self.val_idx].shape,
            x_val.shape,
        )

        self.train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        self.val_dataset = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))

    def train_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        """Return the deterministic shuffled training DataLoader.

        Returns:
            Training DataLoader with seeded shuffling and `num_workers=0`.
        """
        if self.train_dataset is None:
            raise ValueError("setup() must be called before train_dataloader().")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            generator=self.generator,
        )

    def val_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        """Return the deterministic non-shuffled validation DataLoader.

        Returns:
            Validation DataLoader with `shuffle=False` and `generator=None`.
        """
        if self.val_dataset is None:
            raise ValueError("setup() must be called before val_dataloader().")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            generator=None,
        )
