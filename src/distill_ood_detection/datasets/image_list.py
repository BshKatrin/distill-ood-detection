"""Image datasets defined by OpenOOD-style path and label manifests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path, PurePosixPath

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

from distill_ood_detection.config import ImageListConfig

ImageTransform = Callable[[Image.Image], torch.Tensor]


class ImageListDataset(Dataset[tuple[torch.Tensor, int]]):
    """Load images from a text manifest containing ``relative_path label`` rows."""

    def __init__(
        self,
        config: ImageListConfig,
        default_data_dir: Path,
        transform: ImageTransform,
    ) -> None:
        self.imglist_path = Path(config.imglist_path).expanduser()
        self.data_dir = Path(config.data_dir or default_data_dir).expanduser()
        self.transform = transform
        relative_samples = _read_image_list(
            self.imglist_path,
            strip_prefix=config.strip_prefix,
        )
        self.samples = _resolve_samples(
            relative_samples,
            data_dir=self.data_dir,
            basename_search_dir=config.basename_search_dir,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        image = default_loader(path)
        return self.transform(image), label


def _read_image_list(
    path: Path,
    strip_prefix: str | None,
) -> list[tuple[PurePosixPath, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"Image-list manifest not found: {path}")
    prefix_parts = PurePosixPath(strip_prefix).parts if strip_prefix else ()
    samples: list[tuple[PurePosixPath, int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                raw_relative_path, raw_label = line.rsplit(maxsplit=1)
                label = int(raw_label)
            except ValueError as error:
                raise ValueError(
                    f"Invalid image-list row at {path}:{line_number}: {line!r}"
                ) from error
            relative_path = PurePosixPath(raw_relative_path)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(
                    f"Image-list paths must be safe relative paths: {relative_path}"
                )
            if prefix_parts:
                if relative_path.parts[: len(prefix_parts)] != prefix_parts:
                    raise ValueError(
                        f"Image-list path {relative_path} does not start with "
                        f"configured prefix {strip_prefix!r}"
                    )
                relative_path = PurePosixPath(*relative_path.parts[len(prefix_parts) :])
            samples.append((relative_path, label))
    if not samples:
        raise ValueError(f"Image-list manifest is empty: {path}")
    return samples


def _resolve_samples(
    samples: list[tuple[PurePosixPath, int]],
    data_dir: Path,
    basename_search_dir: str | None,
) -> list[tuple[Path, int]]:
    resolved = [
        (data_dir.joinpath(*relative.parts), label) for relative, label in samples
    ]
    if basename_search_dir is None:
        return resolved

    missing_names = {path.name for path, _ in resolved if not path.is_file()}
    if not missing_names:
        return resolved
    search_root = data_dir / basename_search_dir
    if not search_root.is_dir():
        raise FileNotFoundError(f"Basename search directory not found: {search_root}")
    basename_index: dict[str, Path] = {}
    for candidate in search_root.rglob("*"):
        if not candidate.is_file() or candidate.name not in missing_names:
            continue
        previous = basename_index.get(candidate.name)
        if previous is not None:
            raise ValueError(
                f"Duplicate basename {candidate.name!r} under {search_root}: "
                f"{previous} and {candidate}"
            )
        basename_index[candidate.name] = candidate
    unresolved = sorted(missing_names - basename_index.keys())
    if unresolved:
        preview = ", ".join(unresolved[:5])
        raise FileNotFoundError(
            f"Could not resolve {len(unresolved)} manifest images under {search_root}; "
            f"first missing names: {preview}"
        )
    return [(basename_index.get(path.name, path), label) for path, label in resolved]
