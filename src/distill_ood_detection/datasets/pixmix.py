"""Mixing-set loading for PixMix pixel corruptions."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Lambda, RandomCrop, Resize, ToTensor

from distill_ood_detection.config import DatasetConfig, PixMixConfig


class PixMixMixingProvider:
    """Cycle through shuffled mixing-set image batches."""

    def __init__(
        self,
        loader: DataLoader[tuple[torch.Tensor, int]],
    ) -> None:
        self.loader = loader
        self._iterator = iter(loader)
        self._buffer: torch.Tensor | None = None

    def sample(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return one unnormalized RGB mixing image per requested example."""

        batches: list[torch.Tensor] = []
        remaining = batch_size
        if self._buffer is not None:
            take = min(remaining, self._buffer.shape[0])
            batches.append(self._buffer[:take])
            self._buffer = self._buffer[take:] if take < self._buffer.shape[0] else None
            remaining -= take

        while remaining > 0:
            images = self._next_batch()
            take = min(remaining, images.shape[0])
            batches.append(images[:take])
            if take < images.shape[0]:
                self._buffer = images[take:]
            remaining -= take

        return torch.cat(batches, dim=0).to(
            device=device,
            dtype=dtype,
            non_blocking=True,
        )

    def _next_batch(self) -> torch.Tensor:
        try:
            images, _labels = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.loader)
            images, _labels = next(self._iterator)
        return images


def build_pixmix_mixing_provider(
    dataset_config: DatasetConfig,
    pixmix_config: PixMixConfig,
    seed: int,
) -> PixMixMixingProvider:
    """Build a deterministic, worker-prefetched PixMix mixing-set provider."""

    if pixmix_config.mixing_set_path is None:
        raise ValueError("PixMix requires pixmix.mixing_set_path")
    configured_path = Path(pixmix_config.mixing_set_path).expanduser()
    root = (
        configured_path
        if configured_path.is_absolute()
        else Path(dataset_config.data_dir) / configured_path
    )
    if not root.is_dir():
        raise FileNotFoundError(f"PixMix mixing set does not exist: {root}")

    working_size = pixmix_config.working_size
    resize_size = working_size + working_size // 8
    dataset = ImageFolder(
        root=root,
        transform=Compose(
            [
                Lambda(lambda image: image.convert("RGB")),
                Resize(resize_size),
                RandomCrop(working_size),
                ToTensor(),
            ]
        ),
    )
    if len(dataset) == 0:
        raise ValueError(f"PixMix mixing set contains no images: {root}")

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=dataset_config.batch_size,
        shuffle=True,
        num_workers=dataset_config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=dataset_config.num_workers > 0,
        generator=generator,
    )
    return PixMixMixingProvider(loader)
