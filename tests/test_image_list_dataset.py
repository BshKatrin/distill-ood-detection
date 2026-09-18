"""Tests for OpenOOD-style image-list datasets."""

from pathlib import Path

import pytest
from PIL import Image
from torchvision.transforms import ToTensor

from distill_ood_detection.config import ImageListConfig
from distill_ood_detection.datasets.image_list import ImageListDataset


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(path)


def test_loads_manifest_and_strips_openood_prefix(tmp_path: Path) -> None:
    _write_image(tmp_path / "images" / "train" / "example.jpg")
    manifest = tmp_path / "train.txt"
    manifest.write_text("imagenet_1k/train/example.jpg 17\n", encoding="utf-8")
    dataset = ImageListDataset(
        ImageListConfig(
            imglist_path=str(manifest),
            data_dir=str(tmp_path / "images"),
            strip_prefix="imagenet_1k",
        ),
        default_data_dir=tmp_path,
        transform=ToTensor(),
    )

    image, label = dataset[0]

    assert tuple(image.shape) == (3, 8, 8)
    assert label == 17


def test_resolves_flat_manifest_name_in_class_directories(tmp_path: Path) -> None:
    expected = tmp_path / "images" / "val" / "n0001" / "ILSVRC.jpg"
    _write_image(expected)
    manifest = tmp_path / "test.txt"
    manifest.write_text("imagenet_1k/val/ILSVRC.jpg 4\n", encoding="utf-8")
    dataset = ImageListDataset(
        ImageListConfig(
            imglist_path=str(manifest),
            data_dir=str(tmp_path / "images"),
            strip_prefix="imagenet_1k",
            basename_search_dir="val",
        ),
        default_data_dir=tmp_path,
        transform=ToTensor(),
    )

    assert dataset.samples == [(expected, 4)]


def test_rejects_parent_directory_manifest_path(tmp_path: Path) -> None:
    manifest = tmp_path / "unsafe.txt"
    manifest.write_text("../outside.jpg -1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="safe relative paths"):
        ImageListDataset(
            ImageListConfig(imglist_path=str(manifest)),
            default_data_dir=tmp_path,
            transform=ToTensor(),
        )
