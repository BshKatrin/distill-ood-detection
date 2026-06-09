"""Datasets used for probability inference and OOD analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, SVHN
from torchvision.transforms import Compose, Grayscale, Normalize, Resize, ToTensor

from distill_ood_detection.config import DatasetConfig, OODDatasetConfig
from distill_ood_detection.datasets.cifar10 import (
    cifar_normalization,
    cifar_transform,
    split_train_validation,
)


@dataclass(frozen=True)
class NamedLoader:
    """Named dataloader for later artifact metadata."""

    name: str
    split: str
    loader: DataLoader[tuple[torch.Tensor, int]]


def build_in_distribution_test_loader(config: DatasetConfig) -> NamedLoader:
    """Build the in-distribution CIFAR test loader."""

    test_dataset = _cifar_dataset_class(config.name)(
        root=Path(config.data_dir),
        train=False,
        download=True,
        transform=cifar_transform(config.name),
    )
    return NamedLoader(
        name=f"{config.name}_test",
        split="test",
        loader=_loader(test_dataset, config),
    )


def build_in_distribution_train_loader(config: DatasetConfig, seed: int) -> NamedLoader:
    """Build the in-distribution CIFAR training split loader."""

    train_subset, _ = _cifar_train_validation_subsets(config, seed)
    return NamedLoader(
        name=f"{config.name}_train",
        split="train",
        loader=_loader(train_subset, config),
    )


def build_in_distribution_validation_loader(config: DatasetConfig, seed: int) -> NamedLoader:
    """Build the in-distribution CIFAR validation split loader."""

    _, validation_subset = _cifar_train_validation_subsets(config, seed)
    return NamedLoader(
        name=f"{config.name}_validation",
        split="validation",
        loader=_loader(validation_subset, config),
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
    transform = _cifar_like_transform(dataset_config.name, ood_config.name)
    if ood_config.name == "cifar10":
        return CIFAR10(
            root=data_dir,
            train=ood_config.split == "train",
            download=True,
            transform=transform,
        )
    if ood_config.name == "cifar100":
        return CIFAR100(
            root=data_dir,
            train=ood_config.split == "train",
            download=True,
            transform=transform,
        )

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


def _cifar_train_validation_subsets(
    config: DatasetConfig,
    seed: int,
) -> tuple[Dataset[tuple[torch.Tensor, int]], Dataset[tuple[torch.Tensor, int]]]:
    train_dataset = _cifar_dataset_class(config.name)(
        root=Path(config.data_dir),
        train=True,
        download=True,
        transform=cifar_transform(config.name),
    )
    return split_train_validation(
        train_dataset,
        validation_fraction=config.validation_fraction,
        seed=seed,
    )


def _cifar_dataset_class(name: str) -> type[CIFAR10] | type[CIFAR100]:
    if name == "cifar10":
        return CIFAR10
    if name == "cifar100":
        return CIFAR100
    raise ValueError(f"Unsupported in-distribution dataset: {name}")


def _cifar_like_transform(id_name: str, ood_name: str | None = None) -> Compose:
    mean, std = cifar_normalization(id_name)
    return Compose(
        [
            Resize((32, 32)),
            _channels_transform(ood_name),
            ToTensor(),
            Normalize(mean, std),
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

