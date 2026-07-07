"""Summarize clean and pixel-masked teacher pooled embeddings."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from distill_ood_detection.config import DatasetConfig, OODDatasetConfig, TeacherConfig
from distill_ood_detection.datasets.inference import (
    NamedLoader,
    build_in_distribution_test_loader,
    build_ood_loaders,
)
from distill_ood_detection.evaluation.ood_metrics import ood_detection_metrics
from distill_ood_detection.models.teacher import ResNetFeatureForwarder, load_teacher
from distill_ood_detection.utils import resolve_device, set_seed

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT
    / "reports"
    / "outputs"
    / "json"
    / "feature_denoising_pixel_mask_pooled_embeddings_cifar10_resnet18_layer4.json"
)
DEFAULT_ARTIFACT = (
    ROOT
    / "reports"
    / "outputs"
    / "artifacts"
    / "feature_denoising_pixel_mask_pooled_embeddings_cifar10_resnet18_layer4.pt"
)


def main() -> None:
    """Export clean/masked pooled embedding distribution summaries."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--mask-block-count", type=int, default=2)
    parser.add_argument("--mask-scale-min", type=float, default=0.15)
    parser.add_argument("--mask-scale-max", type=float, default=0.20)
    parser.add_argument("--mask-aspect-ratio-min", type=float, default=0.75)
    parser.add_argument("--mask-aspect-ratio-max", type=float, default=1.50)
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    dataset = DatasetConfig(
        name="cifar10",
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        ood_datasets=(
            OODDatasetConfig(name="mnist", split="test"),
            OODDatasetConfig(name="svhn", split="test"),
            OODDatasetConfig(name="cifar100", split="test"),
        ),
    )
    teacher_config = TeacherConfig(
        hf_model_id="edadaltocg/resnet18_cifar10",
        revision="main",
        num_classes=10,
    )
    teacher = load_teacher(teacher_config, device)
    forwarder = ResNetFeatureForwarder(teacher, "layer4").to(device)
    forwarder.eval()

    loaders = [build_in_distribution_test_loader(dataset), *build_ood_loaders(dataset)]
    mask_config = PixelMaskConfig(
        block_count=args.mask_block_count,
        scale_min=args.mask_scale_min,
        scale_max=args.mask_scale_max,
        aspect_ratio_min=args.mask_aspect_ratio_min,
        aspect_ratio_max=args.mask_aspect_ratio_max,
    )
    embeddings = {
        named_loader.name: collect_embeddings(
            named_loader=named_loader,
            forwarder=forwarder,
            device=device,
            max_samples=args.max_samples,
            mask_config=mask_config,
        )
        for named_loader in loaders
    }
    summary = summarize_embeddings(
        embeddings=embeddings,
        id_dataset="cifar10_test",
        metadata={
            "teacher": asdict(teacher_config),
            "dataset": asdict(dataset),
            "feature_layer": "layer4",
            "pooling": "global_average",
            "mask": asdict(mask_config),
            "max_samples": args.max_samples,
            "seed": args.seed,
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": summary["metadata"],
            "embeddings": embeddings,
        },
        args.artifact,
    )


@torch.no_grad()
def collect_embeddings(
    named_loader: NamedLoader,
    forwarder: ResNetFeatureForwarder,
    device: torch.device,
    max_samples: int,
    mask_config: PixelMaskConfig,
) -> dict[str, torch.Tensor]:
    """Collect clean/masked pooled teacher embeddings for one dataset."""

    clean_batches: list[torch.Tensor] = []
    masked_batches: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    mask_fractions: list[torch.Tensor] = []
    seen = 0
    for images, batch_labels in named_loader.loader:
        remaining = max_samples - seen
        if remaining <= 0:
            break
        images = images[:remaining].to(device)
        batch_labels = batch_labels[:remaining]
        masks = sample_pixel_block_keep_mask(images, mask_config)
        masked_images = images * masks
        clean_features = forwarder.forward_to_features(images)
        masked_features = forwarder.forward_to_features(masked_images)
        clean_batches.append(global_average_pool(clean_features).cpu())
        masked_batches.append(global_average_pool(masked_features).cpu())
        labels.append(batch_labels.cpu())
        mask_fractions.append((1.0 - masks[:, :1]).flatten(start_dim=1).mean(dim=1).cpu())
        seen += images.shape[0]
    return {
        "clean": torch.cat(clean_batches, dim=0),
        "masked": torch.cat(masked_batches, dim=0),
        "labels": torch.cat(labels, dim=0),
        "mask_fraction": torch.cat(mask_fractions, dim=0),
    }


def global_average_pool(features: torch.Tensor) -> torch.Tensor:
    """Pool spatial feature maps into one vector per sample."""

    return features.mean(dim=(2, 3))


def sample_pixel_block_keep_mask(
    images: torch.Tensor,
    config: PixelMaskConfig,
) -> torch.Tensor:
    """Sample Feature Denoising rectangular keep masks for normalized images."""

    if images.ndim != 4:
        raise ValueError("images must have shape (B, C, H, W)")
    batch_size, _channels, height, width = images.shape
    keep_mask = torch.ones(
        batch_size,
        1,
        height,
        width,
        device=images.device,
        dtype=images.dtype,
    )
    for row in range(batch_size):
        for _ in range(config.block_count):
            top, left, block_height, block_width = sample_rectangle(
                height=height,
                width=width,
                scale_min=config.scale_min,
                scale_max=config.scale_max,
                aspect_ratio_min=config.aspect_ratio_min,
                aspect_ratio_max=config.aspect_ratio_max,
                device=images.device,
            )
            keep_mask[
                row,
                :,
                top : top + block_height,
                left : left + block_width,
            ] = 0.0
    return keep_mask


def sample_rectangle(
    height: int,
    width: int,
    scale_min: float,
    scale_max: float,
    aspect_ratio_min: float,
    aspect_ratio_max: float,
    device: torch.device,
) -> tuple[int, int, int, int]:
    """Sample a rectangle by area scale and aspect ratio."""

    area = height * width
    scale = float(torch.empty((), device=device).uniform_(scale_min, scale_max).item())
    ratio = float(
        torch.empty((), device=device)
        .uniform_(aspect_ratio_min, aspect_ratio_max)
        .item()
    )
    target_area = max(1.0, area * scale)
    block_height = max(1, min(height, int(round((target_area / ratio) ** 0.5))))
    block_width = max(1, min(width, int(round((target_area * ratio) ** 0.5))))
    top = int(torch.randint(0, height - block_height + 1, (1,), device=device).item())
    left = int(torch.randint(0, width - block_width + 1, (1,), device=device).item())
    return top, left, block_height, block_width


def summarize_embeddings(
    embeddings: dict[str, dict[str, torch.Tensor]],
    id_dataset: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Summarize embedding distributions and simple OOD diagnostic scores."""

    id_clean = embeddings[id_dataset]["clean"].numpy()
    id_masked = embeddings[id_dataset]["masked"].numpy()
    clean_centroid = id_clean.mean(axis=0, keepdims=True)
    masked_centroid = id_masked.mean(axis=0, keepdims=True)
    dataset_summaries = {}
    derived_scores = {}
    for name, values in embeddings.items():
        clean = values["clean"].numpy()
        masked = values["masked"].numpy()
        delta = masked - clean
        scores = {
            "clean_norm": np.linalg.norm(clean, axis=1),
            "masked_norm": np.linalg.norm(masked, axis=1),
            "clean_masked_mse": np.mean(delta**2, axis=1),
            "clean_masked_cosine": cosine_similarity(clean, masked),
            "clean_centroid_distance": np.linalg.norm(clean - clean_centroid, axis=1),
            "masked_centroid_distance": np.linalg.norm(masked - masked_centroid, axis=1),
            "mask_fraction": values["mask_fraction"].numpy(),
        }
        dataset_summaries[name] = {
            key: describe(score_values)
            for key, score_values in scores.items()
        }
        derived_scores[name] = scores

    ood_metrics = {}
    for ood_name in sorted(name for name in embeddings if name != id_dataset):
        ood_metrics[ood_name] = {}
        for score_name in (
            "clean_norm",
            "masked_norm",
            "clean_masked_mse",
            "clean_masked_cosine",
            "clean_centroid_distance",
            "masked_centroid_distance",
        ):
            id_values = derived_scores[id_dataset][score_name]
            ood_values = derived_scores[ood_name][score_name]
            for sign_name, sign in (("positive", 1.0), ("negative", -1.0)):
                labels = np.concatenate(
                    [
                        np.ones_like(id_values, dtype=int),
                        np.zeros_like(ood_values, dtype=int),
                    ]
                )
                signed_scores = np.concatenate([sign * id_values, sign * ood_values])
                ood_metrics[ood_name][f"{sign_name}_{score_name}"] = (
                    ood_detection_metrics(labels, signed_scores)
                )
    return {
        "version": 1,
        "metadata": metadata,
        "datasets": dataset_summaries,
        "ood_metrics": ood_metrics,
    }


def describe(values: np.ndarray) -> dict[str, float]:
    """Return compact distribution statistics for one vector."""

    quantiles = np.quantile(values, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q25": float(quantiles[2]),
        "q50": float(quantiles[3]),
        "q75": float(quantiles[4]),
        "q95": float(quantiles[5]),
        "q99": float(quantiles[6]),
    }


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return rowwise cosine similarity."""

    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return numerator / np.maximum(denominator, 1.0e-12)


@dataclass(frozen=True)
class PixelMaskConfig:
    """image block mask settings."""

    block_count: int
    scale_min: float
    scale_max: float
    aspect_ratio_min: float
    aspect_ratio_max: float


if __name__ == "__main__":
    main()
