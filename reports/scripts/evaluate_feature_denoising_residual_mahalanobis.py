"""Evaluate ID-calibrated residual scores for a trained Feature Denoising student."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from distill_ood_detection.config import ExperimentConfig, load_config
from distill_ood_detection.datasets.inference import (
    NamedLoader,
    build_in_distribution_test_loader,
    build_in_distribution_train_loader,
    build_ood_loaders,
)
from distill_ood_detection.distillation.feature_denoising import sample_feature_denoising_pca_batch
from distill_ood_detection.evaluation.ood_metrics import ood_detection_metrics
from distill_ood_detection.models.student import build_student
from distill_ood_detection.models.teacher import ResNetFeatureForwarder, load_teacher
from distill_ood_detection.utils import resolve_device, set_seed

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "students"
    / "feature_denoising"
    / "spatial_token_prediction"
    / "cifar_10"
    / "resnet18"
    / "token_layer4_blocks2_targets6.yaml"
)
DEFAULT_OUTPUT = (
    ROOT
    / "reports"
    / "outputs"
    / "json"
    / "feature_denoising_spatial_token_residual_mahalanobis_cifar10_layer4_blocks2_targets6.json"
)


def main() -> None:
    """Run residual-direction OOD diagnostics."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", choices=("best", "latest"), default="best")
    parser.add_argument("--fit-samples", type=int, default=5000)
    parser.add_argument("--eval-samples", type=int, default=5000)
    parser.add_argument("--fit-draws", type=int, default=3)
    parser.add_argument("--eval-draws", type=int, default=10)
    parser.add_argument("--variance-eps", type=float, default=1.0e-6)
    args = parser.parse_args()

    config = load_config(args.config)
    if config.strategy.name != "feature_denoising":
        raise ValueError("config must use strategy.name: feature_denoising")
    set_seed(config.training.defaults.seed)
    device = resolve_device(config.training.defaults.device)
    teacher = load_teacher(config.teacher, device)
    forwarder = ResNetFeatureForwarder(teacher, config.student.feature_layer).to(device)
    forwarder.eval()
    student = build_student(config.student).to(device)
    student.load_state_dict(
        torch.load(
            student_checkpoint_path(config, args.checkpoint),
            map_location=device,
            weights_only=True,
        )
    )
    student.eval()

    fit_loader = build_in_distribution_train_loader(
        config.dataset,
        seed=config.training.defaults.seed,
    )
    fit_residuals = collect_residual_matrix(
        student=student,
        forwarder=forwarder,
        loader=fit_loader.loader,
        device=device,
        config=config,
        max_samples=args.fit_samples,
        draws=args.fit_draws,
    )
    residual_stats = fit_diagonal_residual_stats(
        fit_residuals,
        variance_eps=args.variance_eps,
    )

    loaders = [
        build_in_distribution_test_loader(config.dataset),
        *build_ood_loaders(config.dataset),
    ]
    score_by_dataset = {
        named_loader.name: collect_residual_scores(
            student=student,
            forwarder=forwarder,
            loader=named_loader.loader,
            device=device,
            config=config,
            stats=residual_stats,
            max_samples=args.eval_samples,
            draws=args.eval_draws,
        )
        for named_loader in loaders
    }
    output = summarize_scores(
        score_by_dataset=score_by_dataset,
        id_dataset=f"{config.dataset.name}_test",
        metadata={
            "config_path": str(args.config),
            "run_dir": config.run_dir,
            "method": config.strategy.feature_denoising.method,
            "checkpoint": args.checkpoint,
            "fit_samples": args.fit_samples,
            "eval_samples": args.eval_samples,
            "fit_draws": args.fit_draws,
            "eval_draws": args.eval_draws,
            "variance_eps": args.variance_eps,
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)


@torch.no_grad()
def collect_residual_matrix(
    student: nn.Module,
    forwarder: ResNetFeatureForwarder,
    loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
    config: ExperimentConfig,
    max_samples: int,
    draws: int,
) -> torch.Tensor:
    """Collect flattened residual vectors for fitting ID residual statistics."""

    residuals: list[torch.Tensor] = []
    seen = 0
    for images, _labels in loader:
        remaining = max_samples - seen
        if remaining <= 0:
            break
        images = images[:remaining].to(device)
        features = forwarder.forward_to_features(images)
        for _ in range(draws):
            batch = sample_feature_denoising_pca_batch(features, config.strategy.feature_denoising)
            predictions = student(batch.student_inputs)
            residuals.append((predictions - batch.targets).flatten(start_dim=1).cpu())
        seen += images.shape[0]
    return torch.cat(residuals, dim=0)


def fit_diagonal_residual_stats(
    residuals: torch.Tensor,
    variance_eps: float,
) -> dict[str, torch.Tensor]:
    """Fit diagonal Gaussian residual statistics from ID residuals."""

    mean = residuals.mean(dim=0)
    variance = residuals.var(dim=0, unbiased=False).clamp_min(variance_eps)
    return {"mean": mean, "variance": variance}


@torch.no_grad()
def collect_residual_scores(
    student: nn.Module,
    forwarder: ResNetFeatureForwarder,
    loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
    config: ExperimentConfig,
    stats: dict[str, torch.Tensor],
    max_samples: int,
    draws: int,
) -> dict[str, list[float]]:
    """Collect raw and ID-calibrated residual scores for one dataset."""

    mean = stats["mean"].to(device)
    variance = stats["variance"].to(device)
    raw_mse: list[torch.Tensor] = []
    diag_mahalanobis: list[torch.Tensor] = []
    centered_cosine: list[torch.Tensor] = []
    seen = 0
    for images, _labels in loader:
        remaining = max_samples - seen
        if remaining <= 0:
            break
        images = images[:remaining].to(device)
        features = forwarder.forward_to_features(images)
        draw_raw = []
        draw_mahalanobis = []
        draw_cosine = []
        for _ in range(draws):
            batch = sample_feature_denoising_pca_batch(features, config.strategy.feature_denoising)
            predictions = student(batch.student_inputs)
            residual = (predictions - batch.targets).flatten(start_dim=1)
            centered = residual - mean
            draw_raw.append(residual.pow(2).mean(dim=1))
            draw_mahalanobis.append((centered.pow(2) / variance).mean(dim=1))
            draw_cosine.append(cosine_similarity(residual, mean.expand_as(residual)))
        raw_mse.append(torch.stack(draw_raw, dim=1).mean(dim=1).cpu())
        diag_mahalanobis.append(
            torch.stack(draw_mahalanobis, dim=1).mean(dim=1).cpu()
        )
        centered_cosine.append(torch.stack(draw_cosine, dim=1).mean(dim=1).cpu())
        seen += images.shape[0]
    return {
        "raw_mse": torch.cat(raw_mse).tolist(),
        "diag_mahalanobis": torch.cat(diag_mahalanobis).tolist(),
        "residual_mean_cosine": torch.cat(centered_cosine).tolist(),
    }


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return rowwise cosine similarity."""

    numerator = (left * right).sum(dim=1)
    denominator = left.norm(dim=1) * right.norm(dim=1)
    return numerator / denominator.clamp_min(1.0e-12)


def summarize_scores(
    score_by_dataset: dict[str, dict[str, list[float]]],
    id_dataset: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Summarize score distributions and OOD metrics."""

    datasets = {
        dataset: {
            score_name: describe(np.asarray(values))
            for score_name, values in scores.items()
        }
        for dataset, scores in score_by_dataset.items()
    }
    id_scores = score_by_dataset[id_dataset]
    ood_metrics = {}
    for ood_dataset, ood_scores in score_by_dataset.items():
        if ood_dataset == id_dataset:
            continue
        ood_metrics[ood_dataset] = {}
        for score_name, id_values_raw in id_scores.items():
            id_values = np.asarray(id_values_raw)
            ood_values = np.asarray(ood_scores[score_name])
            labels = np.concatenate(
                [
                    np.ones_like(id_values, dtype=int),
                    np.zeros_like(ood_values, dtype=int),
                ]
            )
            for sign_name, sign in (("negative", -1.0), ("positive", 1.0)):
                signed_scores = np.concatenate([sign * id_values, sign * ood_values])
                ood_metrics[ood_dataset][f"{sign_name}_{score_name}"] = (
                    ood_detection_metrics(labels, signed_scores)
                )
    return {
        "version": 1,
        "metadata": metadata,
        "datasets": datasets,
        "ood_metrics": ood_metrics,
    }


def describe(values: np.ndarray) -> dict[str, float]:
    """Return compact distribution statistics."""

    quantiles = np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "q05": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "q50": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "q95": float(quantiles[4]),
    }


def student_checkpoint_path(config: ExperimentConfig, checkpoint: str) -> Path:
    """Return the trained Feature Denoising student checkpoint path from metrics."""

    metrics_path = Path(config.run_dir) / config.strategy.feature_denoising.method / "metrics.json"
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    path = Path(metrics[f"{checkpoint}_checkpoint_path"])
    if not path.is_absolute():
        path = ROOT / path
    return path


if __name__ == "__main__":
    main()
