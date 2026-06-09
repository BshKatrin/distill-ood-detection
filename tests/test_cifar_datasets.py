"""Tests for CIFAR dataloader selection."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch.utils.data import Dataset

from distill_ood_detection.config import DatasetConfig
from distill_ood_detection.datasets.cifar10 import build_cifar_loaders


class TinyImageDataset(Dataset[tuple[torch.Tensor, int]]):
    """Small image dataset matching the torchvision CIFAR constructor shape."""

    def __init__(
        self,
        root: object,
        train: bool,
        download: bool,
        transform: object | None,
    ) -> None:
        self._length = 10 if train else 4
        self._transform = transform

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.zeros(3, 32, 32), index


class CifarDatasetTests(unittest.TestCase):
    """Validate CIFAR-10 and CIFAR-100 loader construction."""

    def test_builds_cifar100_loaders(self) -> None:
        config = DatasetConfig(
            name="cifar100",
            batch_size=2,
            num_workers=0,
            validation_fraction=0.2,
        )

        with patch(
            "distill_ood_detection.datasets.cifar10.CIFAR100",
            TinyImageDataset,
        ):
            loaders = build_cifar_loaders(config, seed=123)

        self.assertEqual(len(loaders.train.dataset), 8)
        self.assertEqual(len(loaders.validation.dataset), 2)
        self.assertEqual(len(loaders.test.dataset), 4)


if __name__ == "__main__":
    unittest.main()
