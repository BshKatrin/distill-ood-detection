"""Tests for image dataloader selection."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar
from unittest.mock import patch

import torch
from torch.utils.data import Dataset

from distill_ood_detection.config import (
    DatasetConfig,
    ImageListConfig,
    OODDatasetConfig,
)
from distill_ood_detection.datasets.inference import (
    build_id_loaders,
    build_in_distribution_full_train_loader,
    build_ood_loaders,
)


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

    seen_splits: ClassVar[list[str]] = []

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

    def test_builds_complete_official_train_loader_for_activation_export(self) -> None:
        config = DatasetConfig(name="cifar100", batch_size=2, num_workers=0)

        with patch(
            "distill_ood_detection.datasets.inference.CIFAR100",
            TinyCifarDataset,
        ):
            named_loader = build_in_distribution_full_train_loader(config)

        self.assertEqual(named_loader.name, "cifar100_train")
        self.assertEqual(named_loader.split, "train")
        self.assertEqual(len(named_loader.loader.dataset), 10)

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

    def test_uses_fixed_imagenet200_manifests_and_ood_group(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = {}
            for split, count in (("train", 3), ("validation", 1), ("test", 2)):
                path = root / f"{split}.txt"
                path.write_text(
                    "".join(f"{split}_{index}.jpg {index}\n" for index in range(count)),
                    encoding="utf-8",
                )
                manifests[split] = ImageListConfig(imglist_path=str(path))
            ood_manifest = root / "ssb_hard.txt"
            ood_manifest.write_text("ood.jpg -1\n", encoding="utf-8")
            config = DatasetConfig(
                name="imagenet200",
                data_dir=str(root),
                batch_size=2,
                num_workers=0,
                image_lists=manifests,
                ood_datasets=(
                    OODDatasetConfig(
                        name="ssb_hard",
                        group="near",
                        image_list=ImageListConfig(imglist_path=str(ood_manifest)),
                    ),
                ),
            )

            loaders = build_id_loaders(config, seed=123)
            ood_loader = build_ood_loaders(config)[0]

        self.assertEqual(len(loaders.train.dataset), 3)
        self.assertEqual(len(loaders.validation.dataset), 1)
        self.assertEqual(len(loaders.test.dataset), 2)
        self.assertEqual(ood_loader.name, "ssb_hard_test")
        self.assertEqual(ood_loader.group, "near")


if __name__ == "__main__":
    unittest.main()
