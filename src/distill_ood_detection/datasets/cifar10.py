"""CIFAR data loading for distillation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import Compose, Normalize, ToTensor

from distill_ood_detection.config import DatasetConfig

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


@dataclass(frozen=True)
class DataLoaders:
    """Train, validation, and test dataloaders."""

    train: DataLoader[tuple[torch.Tensor, int]]
    validation: DataLoader[tuple[torch.Tensor, int]]
    test: DataLoader[tuple[torch.Tensor, int]]


def build_cifar10_loaders(config: DatasetConfig, seed: int) -> DataLoaders:
    """Build deterministic CIFAR-10 train/validation/test dataloaders."""

    if config.name != "cifar10":
        raise ValueError(f"Expected dataset.name='cifar10', got {config.name!r}")
    return build_cifar_loaders(config, seed)


def build_cifar_loaders(config: DatasetConfig, seed: int) -> DataLoaders:
    """Build deterministic CIFAR-10 or CIFAR-100 train/validation/test dataloaders."""

    transform = cifar_transform(config.name)
    data_dir = Path(config.data_dir)
    dataset_class = _cifar_dataset_class(config.name)
    train_dataset = dataset_class(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = dataset_class(root=data_dir, train=False, download=True, transform=transform)
    train_subset, validation_subset = split_train_validation(
        train_dataset,
        validation_fraction=config.validation_fraction,
        seed=seed,
    )
    return DataLoaders(
        train=_loader(train_subset, config, shuffle=True),
        validation=_loader(validation_subset, config, shuffle=False),
        test=_loader(test_dataset, config, shuffle=False),
    )


def cifar_transform(name: str) -> Compose:
    """Return the normalized transform for a supported CIFAR dataset."""

    mean, std = cifar_normalization(name)
    return Compose([ToTensor(), Normalize(mean, std)])


def cifar_normalization(name: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return channel normalization statistics for a supported CIFAR dataset."""

    if name == "cifar10":
        return CIFAR10_MEAN, CIFAR10_STD
    if name == "cifar100":
        return CIFAR100_MEAN, CIFAR100_STD
    raise ValueError(f"Unsupported CIFAR dataset: {name}")


def _cifar_dataset_class(name: str) -> type[CIFAR10] | type[CIFAR100]:
    if name == "cifar10":
        return CIFAR10
    if name == "cifar100":
        return CIFAR100
    raise ValueError(f"Unsupported in-distribution dataset: {name}")


def split_train_validation(
    dataset: Dataset[tuple[torch.Tensor, int]],
    validation_fraction: float,
    seed: int,
) -> tuple[Subset[tuple[torch.Tensor, int]], Subset[tuple[torch.Tensor, int]]]:
    """Split a dataset into train and validation subsets."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    validation_size = int(len(dataset) * validation_fraction)
    train_size = len(dataset) - validation_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, validation_subset = random_split(
        dataset,
        [train_size, validation_size],
        generator=generator,
    )
    return train_subset, validation_subset


def _loader(
    dataset: Dataset[tuple[torch.Tensor, int]],
    config: DatasetConfig,
    shuffle: bool,
) -> DataLoader[tuple[torch.Tensor, int]]:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
