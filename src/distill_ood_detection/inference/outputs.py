"""Shared helpers for logit and probability inference artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from distill_ood_detection.config import PerturbationConfig, TreeDistillationMode
from distill_ood_detection.datasets.pixmix import PixMixMixingProvider
from distill_ood_detection.distillation.perturbation import (
    PcaProjector,
    build_pixel_student_inputs,
    build_sequential_clipping_batch,
    build_unperturbed_pixel_params,
    build_unperturbed_perturbation_batch,
    forward_clean_from_clipping_start,
    forward_to_clipping_start,
    sample_pixel_augmentation,
    sample_perturbation,
    teacher_target_features,
)
from distill_ood_detection.distillation.pixmix import sample_pixmix


@dataclass(frozen=True)
class ModelOutputs:
    """Batched model outputs collected over a dataloader."""

    logits: torch.Tensor
    probabilities: torch.Tensor
    labels: torch.Tensor


@torch.no_grad()
def collect_model_outputs(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
) -> ModelOutputs:
    """Collect logits, softmax probabilities, and labels for a model."""

    logits_batches: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    model.eval()
    for images, batch_labels in loader:
        images = images.to(device)
        logits = model(images)
        logits_batches.append(logits.cpu())
        probabilities.append(F.softmax(logits, dim=1).cpu())
        labels.append(batch_labels.cpu())
    return ModelOutputs(
        logits=torch.cat(logits_batches, dim=0),
        probabilities=torch.cat(probabilities, dim=0),
        labels=torch.cat(labels, dim=0),
    )


@torch.no_grad()
def collect_feature_model_outputs(
    model: nn.Module,
    feature_extractor: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
) -> ModelOutputs:
    """Collect outputs for a student that consumes teacher features."""

    logits_batches: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    model.eval()
    feature_extractor.eval()
    for images, batch_labels in loader:
        images = images.to(device)
        _, features = feature_extractor(images)
        logits = model(features)
        logits_batches.append(logits.cpu())
        probabilities.append(F.softmax(logits, dim=1).cpu())
        labels.append(batch_labels.cpu())
    return ModelOutputs(
        logits=torch.cat(logits_batches, dim=0),
        probabilities=torch.cat(probabilities, dim=0),
        labels=torch.cat(labels, dim=0),
    )


@torch.no_grad()
def collect_perturbation_model_outputs(
    model: nn.Module,
    perturbation_forwarder: nn.Module,
    perturbation_config: PerturbationConfig,
    loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
    apply_perturbation: bool = False,
    pca_projector: PcaProjector | None = None,
    image_normalization: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
    pixmix_provider: PixMixMixingProvider | None = None,
) -> ModelOutputs:
    """Collect per-draw outputs for a perturbation-aware student."""

    logits_batches: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    model.eval()
    perturbation_forwarder.eval()
    for images, batch_labels in loader:
        images = images.to(device)
        if perturbation_config.method in {"pixel_augmentation", "pixmix"}:
            outputs = _collect_pixel_model_batch_outputs(
                model,
                perturbation_forwarder,
                perturbation_config,
                images,
                apply_perturbation=apply_perturbation,
                image_normalization=image_normalization,
                pixmix_provider=pixmix_provider,
            )
            logits_batches.append(outputs[0].cpu())
            probabilities.append(outputs[1].cpu())
            labels.append(batch_labels.cpu())
            continue
        if perturbation_config.method == "clipping":
            start_features = forward_to_clipping_start(
                perturbation_forwarder.teacher,
                images,
                perturbation_config,
            )
            draw_logits = []
            draw_probabilities = []
            batch_size = images.shape[0]
            evaluation_draws = _evaluation_draws(
                perturbation_config,
                apply_perturbation,
            )
            for draw_count in _draw_chunks(evaluation_draws, batch_size):
                clipping_batch = build_sequential_clipping_batch(
                    perturbation_forwarder.teacher,
                    start_features.repeat_interleave(draw_count, dim=0),
                    perturbation_config,
                    apply_perturbation=apply_perturbation,
                )
                logits = model(clipping_batch.student_inputs).reshape(
                    batch_size,
                    draw_count,
                    -1,
                )
                draw_logits.append(logits)
                draw_probabilities.append(F.softmax(logits, dim=-1))
            logits_batches.append(torch.cat(draw_logits, dim=1).cpu())
            probabilities.append(torch.cat(draw_probabilities, dim=1).cpu())
            labels.append(batch_labels.cpu())
            continue
        features = perturbation_forwarder.forward_to_features(images)
        draw_logits: list[torch.Tensor] = []
        draw_probabilities: list[torch.Tensor] = []
        batch_size = features.shape[0]
        evaluation_draws = _evaluation_draws(perturbation_config, apply_perturbation)
        for draw_count in _draw_chunks(evaluation_draws, batch_size):
            expanded_features = features.repeat_interleave(draw_count, dim=0)
            perturbation_batch = _evaluation_perturbation_batch(
                expanded_features,
                perturbation_config,
                apply_perturbation=apply_perturbation,
                pca_projector=pca_projector,
            )
            logits = model(perturbation_batch.student_inputs)
            logits = logits.reshape(batch_size, draw_count, -1)
            draw_logits.append(logits)
            draw_probabilities.append(F.softmax(logits, dim=-1))
        logits_batches.append(torch.cat(draw_logits, dim=1).cpu())
        probabilities.append(torch.cat(draw_probabilities, dim=1).cpu())
        labels.append(batch_labels.cpu())
    return ModelOutputs(
        logits=torch.cat(logits_batches, dim=0),
        probabilities=torch.cat(probabilities, dim=0),
        labels=torch.cat(labels, dim=0),
    )


@torch.no_grad()
def collect_perturbed_teacher_outputs(
    perturbation_forwarder: nn.Module,
    perturbation_config: PerturbationConfig,
    loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
    apply_perturbation: bool = False,
    pca_projector: PcaProjector | None = None,
    image_normalization: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
    pixmix_provider: PixMixMixingProvider | None = None,
) -> ModelOutputs:
    """Collect per-draw perturbed teacher logits and probabilities."""

    logits_batches: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    perturbation_forwarder.eval()
    for images, batch_labels in loader:
        images = images.to(device)
        if perturbation_config.method in {"pixel_augmentation", "pixmix"}:
            outputs = _collect_pixel_teacher_batch_outputs(
                perturbation_forwarder,
                perturbation_config,
                images,
                apply_perturbation=apply_perturbation,
                image_normalization=image_normalization,
                pixmix_provider=pixmix_provider,
            )
            logits_batches.append(outputs[0].cpu())
            probabilities.append(outputs[1].cpu())
            labels.append(batch_labels.cpu())
            continue
        if perturbation_config.method == "clipping":
            start_features = forward_to_clipping_start(
                perturbation_forwarder.teacher,
                images,
                perturbation_config,
            )
            clean_logits = (
                forward_clean_from_clipping_start(
                    perturbation_forwarder.teacher,
                    start_features,
                    perturbation_config,
                )
                if perturbation_config.teacher_target == "clean"
                else None
            )
            draw_logits = []
            draw_probabilities = []
            batch_size = images.shape[0]
            evaluation_draws = _evaluation_draws(
                perturbation_config,
                apply_perturbation,
            )
            for draw_count in _draw_chunks(evaluation_draws, batch_size):
                clipping_batch = build_sequential_clipping_batch(
                    perturbation_forwarder.teacher,
                    start_features.repeat_interleave(draw_count, dim=0),
                    perturbation_config,
                    apply_perturbation=apply_perturbation,
                )
                selected_logits = (
                    clean_logits.repeat_interleave(draw_count, dim=0)
                    if clean_logits is not None
                    else clipping_batch.teacher_logits
                )
                logits = selected_logits.reshape(batch_size, draw_count, -1)
                draw_logits.append(logits)
                draw_probabilities.append(F.softmax(logits, dim=-1))
            logits_batches.append(torch.cat(draw_logits, dim=1).cpu())
            probabilities.append(torch.cat(draw_probabilities, dim=1).cpu())
            labels.append(batch_labels.cpu())
            continue
        features = perturbation_forwarder.forward_to_features(images)
        draw_logits: list[torch.Tensor] = []
        draw_probabilities: list[torch.Tensor] = []
        batch_size = features.shape[0]
        evaluation_draws = _evaluation_draws(perturbation_config, apply_perturbation)
        for draw_count in _draw_chunks(evaluation_draws, batch_size):
            expanded_features = features.repeat_interleave(draw_count, dim=0)
            perturbation_batch = _evaluation_perturbation_batch(
                expanded_features,
                perturbation_config,
                apply_perturbation=apply_perturbation,
                pca_projector=pca_projector,
            )
            logits = perturbation_forwarder.forward_from_features(
                teacher_target_features(perturbation_batch, perturbation_config)
            )
            logits = logits.reshape(batch_size, draw_count, -1)
            draw_logits.append(logits)
            draw_probabilities.append(F.softmax(logits, dim=-1))
        logits_batches.append(torch.cat(draw_logits, dim=1).cpu())
        probabilities.append(torch.cat(draw_probabilities, dim=1).cpu())
        labels.append(batch_labels.cpu())
    return ModelOutputs(
        logits=torch.cat(logits_batches, dim=0),
        probabilities=torch.cat(probabilities, dim=0),
        labels=torch.cat(labels, dim=0),
    )


def _collect_pixel_model_batch_outputs(
    model: nn.Module,
    perturbation_forwarder: nn.Module,
    config: PerturbationConfig,
    images: torch.Tensor,
    apply_perturbation: bool,
    image_normalization: tuple[tuple[float, float, float], tuple[float, float, float]] | None,
    pixmix_provider: PixMixMixingProvider | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    draw_logits: list[torch.Tensor] = []
    draw_probabilities: list[torch.Tensor] = []
    batch_size = images.shape[0]
    evaluation_draws = _evaluation_draws(config, apply_perturbation)
    for _ in range(evaluation_draws):
        student_inputs, _ = _pixel_student_inputs_and_teacher_logits(
            perturbation_forwarder=perturbation_forwarder,
            config=config,
            images=images,
            apply_perturbation=apply_perturbation,
            image_normalization=image_normalization,
            pixmix_provider=pixmix_provider,
        )
        logits = model(student_inputs).reshape(batch_size, 1, -1)
        draw_logits.append(logits)
        draw_probabilities.append(F.softmax(logits, dim=-1))
    return torch.cat(draw_logits, dim=1), torch.cat(draw_probabilities, dim=1)


def _collect_pixel_teacher_batch_outputs(
    perturbation_forwarder: nn.Module,
    config: PerturbationConfig,
    images: torch.Tensor,
    apply_perturbation: bool,
    image_normalization: tuple[tuple[float, float, float], tuple[float, float, float]] | None,
    pixmix_provider: PixMixMixingProvider | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    draw_logits: list[torch.Tensor] = []
    draw_probabilities: list[torch.Tensor] = []
    batch_size = images.shape[0]
    evaluation_draws = _evaluation_draws(config, apply_perturbation)
    for _ in range(evaluation_draws):
        _, logits = _pixel_student_inputs_and_teacher_logits(
            perturbation_forwarder=perturbation_forwarder,
            config=config,
            images=images,
            apply_perturbation=apply_perturbation,
            image_normalization=image_normalization,
            pixmix_provider=pixmix_provider,
        )
        logits = logits.reshape(batch_size, 1, -1)
        draw_logits.append(logits)
        draw_probabilities.append(F.softmax(logits, dim=-1))
    return torch.cat(draw_logits, dim=1), torch.cat(draw_probabilities, dim=1)


def _pixel_student_inputs_and_teacher_logits(
    perturbation_forwarder: nn.Module,
    config: PerturbationConfig,
    images: torch.Tensor,
    apply_perturbation: bool,
    image_normalization: tuple[tuple[float, float, float], tuple[float, float, float]] | None,
    pixmix_provider: PixMixMixingProvider | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if apply_perturbation:
        if image_normalization is None:
            raise ValueError("image_normalization is required for pixel perturbations")
        if config.method == "pixmix":
            if pixmix_provider is None:
                raise ValueError("pixmix_provider is required for PixMix inference")
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
        logits, features = perturbation_forwarder(pixel_batch.perturbed_images)
        if config.teacher_target == "clean":
            logits = perturbation_forwarder.teacher(pixel_batch.clean_images)
    else:
        logits, features = perturbation_forwarder(images)
        params = build_unperturbed_pixel_params(
            images,
            0 if config.method == "pixmix" else 6,
        )
    return (
        build_pixel_student_inputs(
            features,
            params,
            embedding_pool=config.embedding_pool,
        ),
        logits,
    )


def save_model_outputs(
    path: Path,
    outputs: ModelOutputs,
    metadata: dict[str, object],
) -> None:
    """Save logits, probabilities, labels, and metadata to a torch artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **metadata,
            "logits": outputs.logits,
            "probabilities": outputs.probabilities,
            "labels": outputs.labels,
        },
        path,
    )


def collect_tree_model_outputs(
    model: Any,
    mode: TreeDistillationMode,
    loader: DataLoader[tuple[torch.Tensor, int]],
) -> ModelOutputs:
    """Collect logits, probabilities, and labels from a tree-based student."""

    feature_batches: list[np.ndarray] = []
    label_batches: list[torch.Tensor] = []
    for images, batch_labels in loader:
        feature_batches.append(torch.flatten(images, start_dim=1).numpy().astype(np.float32))
        label_batches.append(batch_labels.cpu())

    features = np.concatenate(feature_batches, axis=0)
    logits, probabilities = predict_tree_model_outputs(model, mode, features)
    return ModelOutputs(
        logits=torch.from_numpy(logits),
        probabilities=torch.from_numpy(probabilities),
        labels=torch.cat(label_batches, dim=0),
    )


@torch.no_grad()
def collect_feature_tree_model_outputs(
    model: Any,
    mode: TreeDistillationMode,
    feature_extractor: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
) -> ModelOutputs:
    """Collect outputs for a tree student that consumes teacher features."""

    feature_batches: list[np.ndarray] = []
    label_batches: list[torch.Tensor] = []
    feature_extractor.eval()
    for images, batch_labels in loader:
        images = images.to(device)
        _, features = feature_extractor(images)
        feature_batches.append(
            torch.flatten(features, start_dim=1).cpu().numpy().astype(np.float32)
        )
        label_batches.append(batch_labels.cpu())

    features_array = np.concatenate(feature_batches, axis=0)
    logits, probabilities = predict_tree_model_outputs(model, mode, features_array)
    return ModelOutputs(
        logits=torch.from_numpy(logits),
        probabilities=torch.from_numpy(probabilities),
        labels=torch.cat(label_batches, dim=0),
    )


@torch.no_grad()
def collect_perturbation_tree_model_outputs(
    model: Any,
    mode: TreeDistillationMode,
    perturbation_forwarder: nn.Module,
    perturbation_config: PerturbationConfig,
    loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
    apply_perturbation: bool = False,
    pca_projector: PcaProjector | None = None,
) -> ModelOutputs:
    """Collect per-draw outputs from a perturbation-aware tree student."""

    if perturbation_config.method in {"pixel_augmentation", "pixmix"}:
        raise ValueError("pixel-space perturbations do not support random-forest inference")
    logits_batches: list[torch.Tensor] = []
    probability_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    perturbation_forwarder.eval()
    for images, batch_labels in loader:
        images = images.to(device)
        if perturbation_config.method == "clipping":
            start_features = forward_to_clipping_start(
                perturbation_forwarder.teacher,
                images,
                perturbation_config,
            )
            draw_logits = []
            draw_probabilities = []
            batch_size = images.shape[0]
            evaluation_draws = _evaluation_draws(
                perturbation_config,
                apply_perturbation,
            )
            for draw_count in _draw_chunks(evaluation_draws, batch_size):
                clipping_batch = build_sequential_clipping_batch(
                    perturbation_forwarder.teacher,
                    start_features.repeat_interleave(draw_count, dim=0),
                    perturbation_config,
                    apply_perturbation=apply_perturbation,
                )
                student_features = (
                    clipping_batch.student_inputs.cpu().numpy().astype(np.float32)
                )
                logits, probabilities = predict_tree_model_outputs(
                    model,
                    mode,
                    student_features,
                )
                draw_logits.append(logits.reshape(batch_size, draw_count, -1))
                draw_probabilities.append(
                    probabilities.reshape(batch_size, draw_count, -1)
                )
            logits_batches.append(torch.from_numpy(np.concatenate(draw_logits, axis=1)))
            probability_batches.append(
                torch.from_numpy(np.concatenate(draw_probabilities, axis=1))
            )
            label_batches.append(batch_labels.cpu())
            continue
        features = perturbation_forwarder.forward_to_features(images)
        draw_logits: list[np.ndarray] = []
        draw_probabilities: list[np.ndarray] = []
        batch_size = features.shape[0]
        evaluation_draws = _evaluation_draws(perturbation_config, apply_perturbation)
        for draw_count in _draw_chunks(evaluation_draws, batch_size):
            expanded_features = features.repeat_interleave(draw_count, dim=0)
            perturbation_batch = _evaluation_perturbation_batch(
                expanded_features,
                perturbation_config,
                apply_perturbation=apply_perturbation,
                pca_projector=pca_projector,
            )
            student_features = (
                torch.flatten(perturbation_batch.student_inputs, start_dim=1)
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            logits, probabilities = predict_tree_model_outputs(model, mode, student_features)
            draw_logits.append(logits.reshape(batch_size, draw_count, -1))
            draw_probabilities.append(probabilities.reshape(batch_size, draw_count, -1))
        logits_batches.append(torch.from_numpy(np.concatenate(draw_logits, axis=1)))
        probability_batches.append(torch.from_numpy(np.concatenate(draw_probabilities, axis=1)))
        label_batches.append(batch_labels.cpu())
    return ModelOutputs(
        logits=torch.cat(logits_batches, dim=0),
        probabilities=torch.cat(probability_batches, dim=0),
        labels=torch.cat(label_batches, dim=0),
    )


def predict_tree_model_outputs(
    model: Any,
    mode: TreeDistillationMode,
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict logit/probability outputs from a fitted tree-based student."""

    predictions = np.asarray(model.predict(features), dtype=np.float32)
    if mode == "logits":
        return predictions, _softmax(predictions)
    raise ValueError(f"Unsupported tree distillation mode: {mode}")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


def _draw_chunks(total_draws: int, batch_size: int, max_examples: int = 2048) -> tuple[int, ...]:
    """Split stochastic draws into bounded expanded-batch chunks."""

    chunk_size = max(1, min(total_draws, max_examples // batch_size))
    chunks = []
    remaining = total_draws
    while remaining > 0:
        current = min(chunk_size, remaining)
        chunks.append(current)
        remaining -= current
    return tuple(chunks)


def _evaluation_draws(config: PerturbationConfig, apply_perturbation: bool) -> int:
    return config.evaluation_draws if apply_perturbation else 1


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
