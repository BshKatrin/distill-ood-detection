"""Tests for PixMix corruption and mixing-set loading."""

from __future__ import annotations

from pathlib import Path
import tempfile

from PIL import Image
import torch

from distill_ood_detection.config import DatasetConfig, PixMixConfig
from distill_ood_detection.datasets.pixmix import build_pixmix_mixing_provider
from distill_ood_detection.distillation.pixmix import sample_pixmix


def test_pixmix_returns_base_and_corrupted_views() -> None:
    images = torch.zeros(3, 3, 32, 32)
    mixing_images = torch.rand(3, 3, 32, 32)
    config = PixMixConfig(
        mixing_iterations=4,
        beta=3.0,
        augmentation_severity=3.0,
        all_ops=True,
        working_size=32,
    )

    torch.manual_seed(123)
    batch = sample_pixmix(
        images,
        mixing_images,
        config,
        ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    )

    assert batch.clean_images.shape == images.shape
    assert batch.perturbed_images.shape == images.shape
    assert torch.isfinite(batch.perturbed_images).all()
    assert torch.all(batch.perturbed_images >= 0.0)
    assert torch.all(batch.perturbed_images <= 1.0)
    assert not torch.equal(batch.clean_images, batch.perturbed_images)


def test_pixmix_resizes_working_views_back_to_teacher_input() -> None:
    images = torch.zeros(2, 3, 224, 224)
    mixing_images = torch.rand(2, 3, 32, 32)

    batch = sample_pixmix(
        images,
        mixing_images,
        PixMixConfig(mixing_iterations=1, working_size=32),
        ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    )

    assert batch.clean_images.shape == images.shape
    assert batch.perturbed_images.shape == images.shape


def test_pixmix_mixing_provider_cycles_rgb_images() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        class_dir = root / "fractals"
        class_dir.mkdir()
        for index in range(3):
            image = Image.new("RGB", (40, 40), color=(index * 20, 50, 100))
            image.save(class_dir / f"{index}.png")

        provider = build_pixmix_mixing_provider(
            DatasetConfig(
                data_dir=directory,
                batch_size=2,
                num_workers=0,
            ),
            PixMixConfig(
                mixing_set_path=".",
                working_size=32,
            ),
            seed=42,
        )

        sampled = provider.sample(
            batch_size=5,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    assert sampled.shape == (5, 3, 32, 32)
    assert torch.all(sampled >= 0.0)
    assert torch.all(sampled <= 1.0)
