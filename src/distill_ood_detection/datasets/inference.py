"""Datasets used for probability inference and OOD analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet, MNIST, SVHN
from torchvision.transforms import CenterCrop, Compose, Lambda, Normalize, Resize, ToTensor

from distill_ood_detection.config import DatasetConfig, OODDatasetConfig


class IDPreprocessingConfig(TypedDict):
    """Image preprocessing settings for an in-distribution dataset."""

    pre_size: int
    image_size: int
    normalization: tuple[tuple[float, float, float], tuple[float, float, float]]


ID_PREPROCESSING: dict[str, IDPreprocessingConfig] = {
    "cifar10": {
        "pre_size": 32,
        "image_size": 32,
        "normalization": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    },
    "cifar100": {
        "pre_size": 32,
        "image_size": 32,
        "normalization": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    },
    "imagenet": {
        "pre_size": 256,
        "image_size": 224,
        "normalization": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    },
}


@dataclass(frozen=True)
class NamedLoader:
    """Named dataloader for later artifact metadata."""

    name: str
    split: str
    loader: DataLoader[tuple[torch.Tensor, int]]


@dataclass(frozen=True)
class DataLoaders:
    """Train, validation, and test dataloaders."""

    train: DataLoader[tuple[torch.Tensor, int]]
    validation: DataLoader[tuple[torch.Tensor, int]]
    test: DataLoader[tuple[torch.Tensor, int]]


def build_id_loaders(config: DatasetConfig, seed: int) -> DataLoaders:
    """Build deterministic in-distribution train/validation/test dataloaders."""

    train_subset, validation_subset = split_train_validation(
        _id_dataset(config, split="train"),
        validation_fraction=config.validation_fraction,
        seed=seed,
    )
    return DataLoaders(
        train=_loader(train_subset, config, shuffle=True),
        validation=_loader(validation_subset, config),
        test=_loader(_id_dataset(config, split="test"), config),
    )


def build_in_distribution_test_loader(config: DatasetConfig) -> NamedLoader:
    """Build the in-distribution test loader."""

    return NamedLoader(
        name=f"{config.name}_test",
        split="test",
        loader=_loader(_id_dataset(config, split="test"), config),
    )


def build_in_distribution_train_loader(config: DatasetConfig, seed: int) -> NamedLoader:
    """Build the in-distribution training split loader."""

    train_subset, _ = _id_train_validation_subsets(config, seed)
    return NamedLoader(
        name=f"{config.name}_train",
        split="train",
        loader=_loader(train_subset, config),
    )


def build_in_distribution_validation_loader(config: DatasetConfig, seed: int) -> NamedLoader:
    """Build the in-distribution validation split loader."""

    _, validation_subset = _id_train_validation_subsets(config, seed)
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
    return _torchvision_dataset(
        name=ood_config.name,
        root=Path(dataset_config.data_dir),
        split=ood_config.split,
        transform=_inference_transform(dataset_config.name),
    )


def _id_train_validation_subsets(
    config: DatasetConfig,
    seed: int,
) -> tuple[Dataset[tuple[torch.Tensor, int]], Dataset[tuple[torch.Tensor, int]]]:
    return split_train_validation(
        _id_dataset(config, split="train"),
        validation_fraction=config.validation_fraction,
        seed=seed,
    )


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


def _id_dataset(config: DatasetConfig, split: str) -> Dataset[tuple[torch.Tensor, int]]:
    _id_preprocessing(config.name)
    return _torchvision_dataset(
        name=config.name,
        root=Path(config.data_dir),
        split=split,
        transform=_inference_transform(config.name),
    )


def _torchvision_dataset(
    name: str,
    root: Path,
    split: str,
    transform: Compose,
) -> Dataset[tuple[torch.Tensor, int]]:
    if name == "cifar10":
        return CIFAR10(root=root, train=split == "train", download=True, transform=transform)
    if name == "cifar100":
        return CIFAR100(root=root, train=split == "train", download=True, transform=transform)
    if name == "imagenet":
        return ImageNet(root=root, split=_imagenet_split(split), transform=transform)
    if name == "mnist":
        return MNIST(root=root, train=split == "train", download=True, transform=transform)
    if name == "svhn":
        return SVHN(root=root, split=split, download=True, transform=transform)
    raise ValueError(f"Unsupported dataset: {name}")


def _imagenet_split(split: str) -> str:
    if split == "test":
        return "val"
    return split


def _inference_transform(id_name: str) -> Compose:
    """Return the standard image transform used for inference datasets."""

    preprocessing = _id_preprocessing(id_name)
    mean, std = preprocessing["normalization"]
    return Compose(
        [
            Lambda(lambda image: image.convert("RGB")),
            Resize(preprocessing["pre_size"]),
            CenterCrop(preprocessing["image_size"]),
            ToTensor(),
            Normalize(mean, std),
        ]
    )


def _id_preprocessing(name: str) -> IDPreprocessingConfig:
    try:
        return ID_PREPROCESSING[name]
    except KeyError as error:
        raise ValueError(f"Unsupported in-distribution dataset: {name}") from error


def _loader(
    dataset: Dataset[tuple[torch.Tensor, int]],
    config: DatasetConfig,
    shuffle: bool = False,
) -> DataLoader[tuple[torch.Tensor, int]]:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
