"""Datasets used for probability inference and OOD analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, MNIST, SVHN
from torchvision.transforms import Compose, Grayscale, Normalize, Resize, ToTensor

from distill_ood_detection.config import DatasetConfig, OODDatasetConfig
from distill_ood_detection.datasets.cifar10 import (
    CIFAR10_MEAN,
    CIFAR10_STD,
)


@dataclass(frozen=True)
class NamedLoader:
    """Named dataloader for later artifact metadata."""

    name: str
    split: str
    loader: DataLoader[tuple[torch.Tensor, int]]


def build_in_distribution_test_loader(config: DatasetConfig) -> NamedLoader:
    """Build the in-distribution CIFAR-10 test loader."""

    if config.name != "cifar10":
        raise ValueError(f"Unsupported in-distribution dataset: {config.name}")
    transform = Compose([ToTensor(), Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    test_dataset = CIFAR10(
        root=Path(config.data_dir),
        train=False,
        download=True,
        transform=transform,
    )
    return NamedLoader(
        name=f"{config.name}_test",
        split="test",
        loader=_loader(test_dataset, config),
    )


def build_ood_loaders(config: DatasetConfig) -> list[NamedLoader]:
    """Build configured OOD dataset loaders."""

    return [
        NamedLoader(
            name=f"{ood.name}_{ood.split}",
            split=ood.split,
            loader=_loader(_ood_dataset(config, ood), config),
        )
        for ood in config.ood_datasets
    ]


def _ood_dataset(
    dataset_config: DatasetConfig,
    ood_config: OODDatasetConfig,
) -> Dataset[tuple[torch.Tensor, int]]:
    data_dir = Path(dataset_config.data_dir)
    if ood_config.name == "cifar100":
        return HuggingFaceImageDataset(
            dataset_id="uoft-cs/cifar100",
            split=ood_config.split,
            image_key="img",
            label_key="fine_label",
            transform=_cifar_like_transform(),
            cache_dir=data_dir / "huggingface",
        )

    transform = _cifar_like_transform(ood_config.name)
    if ood_config.name == "mnist":
        return MNIST(
            root=data_dir,
            train=ood_config.split == "train",
            download=True,
            transform=transform,
        )
    if ood_config.name == "svhn":
        return SVHN(
            root=data_dir,
            split=ood_config.split,
            download=True,
            transform=transform,
        )
    raise ValueError(f"Unsupported OOD dataset: {ood_config.name}")


def _cifar_like_transform(name: str | None = None) -> Compose:
    return Compose(
        [
            Resize((32, 32)),
            _channels_transform(name),
            ToTensor(),
            Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def _channels_transform(name: str | None) -> object:
    if name == "mnist":
        return Grayscale(num_output_channels=3)
    return _Identity()


def _loader(
    dataset: Dataset[tuple[torch.Tensor, int]],
    config: DatasetConfig,
) -> DataLoader[tuple[torch.Tensor, int]]:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


class _Identity:
    def __call__(self, image: object) -> object:
        return image


class HuggingFaceImageDataset(Dataset[tuple[torch.Tensor, int]]):
    """Torch dataset wrapper around a Hugging Face image dataset split."""

    def __init__(
        self,
        dataset_id: str,
        split: str,
        image_key: str,
        label_key: str,
        transform: Compose,
        cache_dir: Path,
    ) -> None:
        self._dataset = load_dataset(
            dataset_id,
            split=split,
            cache_dir=str(cache_dir),
        )
        self._image_key = image_key
        self._label_key = label_key
        self._transform = transform

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = cast(dict[str, Any], self._dataset[index])
        image = sample[self._image_key]
        if not hasattr(image, "mode"):
            raise TypeError(
                f"Expected PIL image-like object in '{self._image_key}', got {type(image)!r}"
            )
        image = image.convert("RGB")
        label = int(sample[self._label_key])
        return self._transform(image), label
