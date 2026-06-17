"""Tests for image dataloader selection."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch.utils.data import Dataset

from distill_ood_detection.config import DatasetConfig
from distill_ood_detection.datasets.inference import build_id_loaders


class TinyCifarDataset(Dataset[tuple[torch.Tensor, int]]):
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


class TinyImageNetDataset(Dataset[tuple[torch.Tensor, int]]):
    """Small image dataset matching the torchvision ImageNet constructor shape."""

    seen_splits: list[str] = []

    def __init__(
        self,
        root: object,
        split: str,
        transform: object | None,
    ) -> None:
        self._length = 10 if split == "train" else 4
        self._transform = transform
        self.seen_splits.append(split)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.zeros(3, 224, 224), index


class ImageDatasetTests(unittest.TestCase):
    """Validate in-distribution image loader construction."""

    def test_builds_cifar100_loaders(self) -> None:
        config = DatasetConfig(
            name="cifar100",
            batch_size=2,
            num_workers=0,
            validation_fraction=0.2,
        )

        with patch(
            "distill_ood_detection.datasets.inference.CIFAR100",
            TinyCifarDataset,
        ):
            loaders = build_id_loaders(config, seed=123)

        self.assertEqual(len(loaders.train.dataset), 8)
        self.assertEqual(len(loaders.validation.dataset), 2)
        self.assertEqual(len(loaders.test.dataset), 4)

    def test_builds_imagenet_loaders_with_validation_as_test_split(self) -> None:
        TinyImageNetDataset.seen_splits = []
        config = DatasetConfig(
            name="imagenet",
            batch_size=2,
            num_workers=0,
            validation_fraction=0.2,
        )

        with patch(
            "distill_ood_detection.datasets.inference.ImageNet",
            TinyImageNetDataset,
        ):
            loaders = build_id_loaders(config, seed=123)

        self.assertEqual(len(loaders.train.dataset), 8)
        self.assertEqual(len(loaders.validation.dataset), 2)
        self.assertEqual(len(loaders.test.dataset), 4)
        self.assertEqual(TinyImageNetDataset.seen_splits, ["train", "val"])


if __name__ == "__main__":
    unittest.main()
