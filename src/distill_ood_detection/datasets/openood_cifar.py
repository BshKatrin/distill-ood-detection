"""Fixed OpenOOD v1.5 evaluation datasets for CIFAR-10 and CIFAR-100."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from distill_ood_detection.config import (
    DatasetConfig,
    ImageListConfig,
    OODDatasetConfig,
)

OPENOOD_CIFAR_PROTOCOL = "openood_v1_5"

OPENOOD_CIFAR_DATASETS: dict[str, tuple[tuple[str, str, int], ...]] = {
    "cifar10": (
        ("cifar100", "near", 9_000),
        ("tin", "near", 7_793),
        ("mnist", "far", 70_000),
        ("svhn", "far", 26_032),
        ("texture", "far", 5_640),
        ("places365", "far", 35_195),
    ),
    "cifar100": (
        ("cifar10", "near", 10_000),
        ("tin", "near", 6_526),
        ("mnist", "far", 70_000),
        ("svhn", "far", 26_032),
        ("texture", "far", 5_640),
        ("places365", "far", 33_773),
    ),
}

OPENOOD_CIFAR_ID_TEST_SIZE = 9_000


def build_openood_cifar_dataset_config(
    source: DatasetConfig,
    openood_root: Path,
) -> DatasetConfig:
    """Replace evaluation splits with fixed OpenOOD v1.5 image manifests.

    The source preprocessing is retained because the evaluated Hugging Face
    classifier and students were trained with those input statistics.
    """

    try:
        ood_datasets = OPENOOD_CIFAR_DATASETS[source.name]
    except KeyError as error:
        raise ValueError(
            "OpenOOD CIFAR evaluation requires CIFAR-10 or CIFAR-100 as ID"
        ) from error

    data_root = openood_root / "data"
    manifest_root = data_root / "benchmark_imglist" / source.name
    image_root = data_root / "images_classic"
    return replace(
        source,
        data_dir=str(image_root),
        image_lists={
            "test": ImageListConfig(
                imglist_path=str(manifest_root / f"test_{source.name}.txt"),
                data_dir=str(image_root),
            )
        },
        ood_datasets=tuple(
            OODDatasetConfig(
                name=name,  # type: ignore[arg-type]
                split="test",
                group=group,  # type: ignore[arg-type]
                image_list=ImageListConfig(
                    imglist_path=str(manifest_root / f"test_{name}.txt"),
                    data_dir=str(image_root),
                ),
            )
            for name, group, _count in ood_datasets
        ),
    )


def expected_openood_cifar_sizes(id_dataset: str) -> dict[str, int]:
    """Return expected manifest sizes keyed by exported dataset name."""

    try:
        ood_datasets = OPENOOD_CIFAR_DATASETS[id_dataset]
    except KeyError as error:
        raise ValueError(f"Unsupported OpenOOD CIFAR ID dataset: {id_dataset}") from error
    return {
        f"{id_dataset}_test": OPENOOD_CIFAR_ID_TEST_SIZE,
        **{f"{name}_test": count for name, _group, count in ood_datasets},
    }


def validate_openood_cifar_manifests(config: DatasetConfig) -> dict[str, int]:
    """Validate every fixed manifest and return its non-empty row count."""

    manifests = {
        f"{config.name}_test": config.image_lists["test"],
        **{
            f"{dataset.name}_{dataset.split}": dataset.image_list
            for dataset in config.ood_datasets
        },
    }
    expected = expected_openood_cifar_sizes(config.name)
    actual: dict[str, int] = {}
    for name, image_list in manifests.items():
        if image_list is None:
            raise ValueError(f"OpenOOD dataset {name} has no image-list manifest")
        path = Path(image_list.imglist_path)
        if not path.is_file():
            raise FileNotFoundError(f"OpenOOD manifest not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            count = sum(bool(line.strip()) for line in handle)
        actual[name] = count
        if count != expected[name]:
            raise ValueError(
                f"Unexpected OpenOOD manifest size for {name}: {count} != {expected[name]}"
            )
    return actual
