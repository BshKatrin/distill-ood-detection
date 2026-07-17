"""Classification metrics."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from distill_ood_detection.config import DistillationMethod, PerturbationConfig
from distill_ood_detection.datasets.pixmix import PixMixMixingProvider
from distill_ood_detection.distillation.losses import distillation_loss
from distill_ood_detection.distillation.perturbation import (
    PcaProjector,
    build_pixel_student_inputs,
    build_sequential_clipping_batch,
    build_unperturbed_pixel_params,
    build_unperturbed_perturbation_batch,
    clipping_teacher_target_logits,
    forward_clean_from_clipping_start,
    forward_to_clipping_start,
    sample_pixel_augmentation,
    sample_perturbation,
    teacher_target_features,
)
from distill_ood_detection.distillation.pixmix import sample_pixmix


@torch.no_grad()
def accuracy(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
    feature_extractor: nn.Module | None = None,
    perturbation_forwarder: nn.Module | None = None,
    perturbation_config: PerturbationConfig | None = None,
    pca_projector: PcaProjector | None = None,
    image_normalization: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
    pixmix_provider: PixMixMixingProvider | None = None,
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
            if perturbation_config.method in {"pixel_augmentation", "pixmix"}:
                predictions = _pixel_accuracy_predictions(
                    model,
                    perturbation_forwarder,
                    images,
                    perturbation_config,
                    image_normalization,
                    pixmix_provider,
                    apply_perturbation=apply_perturbation,
                )
            elif perturbation_config.method == "clipping":
                start_features = forward_to_clipping_start(
                    perturbation_forwarder.teacher,
                    images,
                    perturbation_config,
                )
                logits_sum = None
                draws = perturbation_config.evaluation_draws if apply_perturbation else 1
                for _ in range(draws):
                    clipping_batch = build_sequential_clipping_batch(
                        perturbation_forwarder.teacher,
                        start_features,
                        perturbation_config,
                        apply_perturbation=apply_perturbation,
                    )
                    logits = model(clipping_batch.student_inputs)
                    logits_sum = logits if logits_sum is None else logits_sum + logits
                assert logits_sum is not None
                predictions = (logits_sum / draws).argmax(dim=1)
            else:
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
    image_normalization: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
    pixmix_provider: PixMixMixingProvider | None = None,
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
            if perturbation_config.method in {"pixel_augmentation", "pixmix"}:
                batch_metrics = _pixel_distillation_batch_metrics(
                    method=method,
                    teacher=teacher,
                    student=student,
                    perturbation_forwarder=perturbation_forwarder,
                    images=images,
                    labels=labels,
                    perturbation_config=perturbation_config,
                    image_normalization=image_normalization,
                    pixmix_provider=pixmix_provider,
                    temperature=temperature,
                    alpha=alpha,
                    apply_perturbation=apply_perturbation,
                )
                batch_size = labels.numel()
                correct += batch_metrics["correct"]
                total += batch_size
                total_kl += batch_metrics["kl_divergence"] * batch_size
                total_distillation_loss += batch_metrics["distillation_loss"] * batch_size
                continue
            if perturbation_config.method == "clipping":
                start_features = forward_to_clipping_start(
                    perturbation_forwarder.teacher,
                    images,
                    perturbation_config,
                )
                clipping_batch = build_sequential_clipping_batch(
                    perturbation_forwarder.teacher,
                    start_features,
                    perturbation_config,
                    apply_perturbation=apply_perturbation,
                )
                clean_logits = (
                    forward_clean_from_clipping_start(
                        perturbation_forwarder.teacher,
                        start_features,
                        perturbation_config,
                    )
                    if perturbation_config.teacher_target == "clean"
                    else clipping_batch.teacher_logits
                )
                teacher_logits = clipping_teacher_target_logits(
                    clipping_batch,
                    clean_logits,
                    perturbation_config,
                )
                student_inputs = clipping_batch.student_inputs
            else:
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


def _pixel_accuracy_predictions(
    model: nn.Module,
    perturbation_forwarder: nn.Module,
    images: torch.Tensor,
    config: PerturbationConfig,
    image_normalization: tuple[tuple[float, float, float], tuple[float, float, float]] | None,
    pixmix_provider: PixMixMixingProvider | None,
    apply_perturbation: bool,
) -> torch.Tensor:
    logits_sum = None
    draws = config.evaluation_draws if apply_perturbation else 1
    for _ in range(draws):
        student_inputs, _ = _pixel_student_inputs_and_teacher_logits(
            teacher=None,
            perturbation_forwarder=perturbation_forwarder,
            images=images,
            config=config,
            image_normalization=image_normalization,
            pixmix_provider=pixmix_provider,
            apply_perturbation=apply_perturbation,
        )
        logits = model(student_inputs)
        logits_sum = logits if logits_sum is None else logits_sum + logits
    assert logits_sum is not None
    return (logits_sum / draws).argmax(dim=1)


def _pixel_distillation_batch_metrics(
    method: DistillationMethod,
    teacher: nn.Module,
    student: nn.Module,
    perturbation_forwarder: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    perturbation_config: PerturbationConfig,
    image_normalization: tuple[tuple[float, float, float], tuple[float, float, float]] | None,
    pixmix_provider: PixMixMixingProvider | None,
    temperature: float,
    alpha: float,
    apply_perturbation: bool,
) -> dict[str, float | int]:
    logits_sum = None
    total_kl = 0.0
    total_loss = 0.0
    draws = perturbation_config.evaluation_draws if apply_perturbation else 1
    for _ in range(draws):
        student_inputs, teacher_logits = _pixel_student_inputs_and_teacher_logits(
            teacher=teacher,
            perturbation_forwarder=perturbation_forwarder,
            images=images,
            config=perturbation_config,
            image_normalization=image_normalization,
            pixmix_provider=pixmix_provider,
            apply_perturbation=apply_perturbation,
        )
        student_logits = student(student_inputs)
        teacher_probabilities = F.softmax(teacher_logits, dim=1)
        student_log_probabilities = F.log_softmax(student_logits, dim=1)
        total_kl += F.kl_div(
            student_log_probabilities,
            teacher_probabilities,
            reduction="batchmean",
        ).item()
        total_loss += distillation_loss(
            method,
            student_logits,
            teacher_logits,
            labels=labels,
            temperature=temperature,
            alpha=alpha,
        ).item()
        logits_sum = student_logits if logits_sum is None else logits_sum + student_logits
    assert logits_sum is not None
    predictions = (logits_sum / draws).argmax(dim=1)
    return {
        "correct": int((predictions == labels).sum().item()),
        "kl_divergence": total_kl / draws,
        "distillation_loss": total_loss / draws,
    }


def _pixel_student_inputs_and_teacher_logits(
    teacher: nn.Module | None,
    perturbation_forwarder: nn.Module,
    images: torch.Tensor,
    config: PerturbationConfig,
    image_normalization: tuple[tuple[float, float, float], tuple[float, float, float]] | None,
    pixmix_provider: PixMixMixingProvider | None,
    apply_perturbation: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if apply_perturbation:
        if image_normalization is None:
            raise ValueError("image_normalization is required for pixel perturbations")
        if config.method == "pixmix":
            if pixmix_provider is None:
                raise ValueError("pixmix_provider is required for PixMix evaluation")
            pixel_batch = sample_pixmix(
                images,
                pixmix_provider.sample(images.shape[0], images.device, images.dtype),
                config.pixmix,
                image_normalization,
            )
            params = images.new_empty((images.shape[0], 0))
        else:
            pixel_batch = sample_pixel_augmentation(images, config, image_normalization)
            params = pixel_batch.normalized_transform_params
        perturbed_logits, features = perturbation_forwarder(pixel_batch.perturbed_images)
        clean_images = pixel_batch.clean_images
    else:
        perturbed_logits, features = perturbation_forwarder(images)
        parameter_count = 0 if config.method == "pixmix" else 6
        params = build_unperturbed_pixel_params(images, parameter_count)
        clean_images = images
    student_inputs = build_pixel_student_inputs(
        features,
        params,
        embedding_pool=config.embedding_pool,
    )
    if teacher is None or config.teacher_target == "perturbed":
        return student_inputs, perturbed_logits
    return student_inputs, teacher(clean_images)


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
