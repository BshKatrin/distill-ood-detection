"""Classification metrics."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from distill_ood_detection.config import DistillationMethod, PerturbationConfig
from distill_ood_detection.distillation.losses import distillation_loss
from distill_ood_detection.distillation.perturbation import (
    PcaProjector,
    build_unperturbed_perturbation_batch,
    sample_perturbation,
    teacher_target_features,
)


@torch.no_grad()
def accuracy(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
    feature_extractor: nn.Module | None = None,
    perturbation_forwarder: nn.Module | None = None,
    perturbation_config: PerturbationConfig | None = None,
    pca_projector: PcaProjector | None = None,
    apply_perturbation: bool = False,
) -> float:
    """Compute top-1 classification accuracy."""

    model.eval()
    if feature_extractor is not None:
        feature_extractor.eval()
    if perturbation_forwarder is not None:
        perturbation_forwarder.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if perturbation_forwarder is not None:
            if perturbation_config is None:
                raise ValueError("perturbation_config is required for perturbation accuracy")
            features = perturbation_forwarder.forward_to_features(images)
            logits_sum = None
            draws = perturbation_config.evaluation_draws if apply_perturbation else 1
            for _ in range(draws):
                perturbation_batch = _evaluation_perturbation_batch(
                    features,
                    perturbation_config,
                    apply_perturbation=apply_perturbation,
                    pca_projector=pca_projector,
                )
                logits = model(perturbation_batch.student_inputs)
                logits_sum = logits if logits_sum is None else logits_sum + logits
            assert logits_sum is not None
            predictions = (logits_sum / draws).argmax(dim=1)
        else:
            inputs = images
            if feature_extractor is not None:
                _, inputs = feature_extractor(images)
            predictions = model(inputs).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.numel()
    return correct / total


@torch.no_grad()
def distillation_metrics(
    method: DistillationMethod,
    teacher: nn.Module,
    student: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
    temperature: float = 1.0,
    alpha: float = 0.5,
    feature_extractor: nn.Module | None = None,
    perturbation_forwarder: nn.Module | None = None,
    perturbation_config: PerturbationConfig | None = None,
    pca_projector: PcaProjector | None = None,
    apply_perturbation: bool = False,
    split: Literal["validation", "test"] = "validation",
) -> dict[str, float]:
    """Compute student accuracy and distillation metrics for one data split."""

    teacher.eval()
    student.eval()
    if feature_extractor is not None:
        feature_extractor.eval()
    if perturbation_forwarder is not None:
        perturbation_forwarder.eval()
    correct = 0
    total = 0
    total_kl = 0.0
    total_distillation_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if perturbation_forwarder is not None:
            if perturbation_config is None:
                raise ValueError(
                    f"perturbation_config is required for perturbation {split} metrics"
                )
            features = perturbation_forwarder.forward_to_features(images)
            perturbation_batch = _evaluation_perturbation_batch(
                features,
                perturbation_config,
                apply_perturbation=apply_perturbation,
                pca_projector=pca_projector,
            )
            teacher_logits = perturbation_forwarder.forward_from_features(
                teacher_target_features(perturbation_batch, perturbation_config)
            )
            student_inputs = perturbation_batch.student_inputs
        elif feature_extractor is None:
            teacher_logits = teacher(images)
            student_inputs = images
        else:
            teacher_logits, student_inputs = feature_extractor(images)
        student_logits = student(student_inputs)
        predictions = student_logits.argmax(dim=1)
        teacher_probabilities = F.softmax(teacher_logits, dim=1)
        student_log_probabilities = F.log_softmax(student_logits, dim=1)
        kl_divergence = F.kl_div(
            student_log_probabilities,
            teacher_probabilities,
            reduction="batchmean",
        )

        batch_size = labels.numel()
        correct += (predictions == labels).sum().item()
        total += batch_size
        total_kl += kl_divergence.item() * batch_size
        total_distillation_loss += (
            distillation_loss(
                method,
                student_logits,
                teacher_logits,
                labels=labels,
                temperature=temperature,
                alpha=alpha,
            ).item()
            * batch_size
        )

    return {
        f"{split}_accuracy": correct / total,
        f"{split}_distillation_loss": total_distillation_loss / total,
        f"{split}_kl_divergence": total_kl / total,
    }


def _evaluation_perturbation_batch(
    features: torch.Tensor,
    config: PerturbationConfig,
    apply_perturbation: bool,
    pca_projector: PcaProjector | None = None,
):
    if apply_perturbation:
        return sample_perturbation(features, config, pca_projector=pca_projector)
    return build_unperturbed_perturbation_batch(
        features,
        config,
        pca_projector=pca_projector,
    )
