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

from distill_ood_detection.config import TreeDistillationMode


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


def predict_tree_model_outputs(
    model: Any,
    mode: TreeDistillationMode,
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict logit/probability outputs from a fitted tree-based student."""

    predictions = np.asarray(model.predict(features), dtype=np.float32)
    if mode == "logits":
        return predictions, _softmax(predictions)

    probabilities = np.clip(predictions, 1e-12, None)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    zero_rows = row_sums.squeeze(axis=1) <= 0.0
    if np.any(zero_rows):
        probabilities[zero_rows] = 1.0 / probabilities.shape[1]
        row_sums = probabilities.sum(axis=1, keepdims=True)
    probabilities = probabilities / row_sums
    return np.log(probabilities).astype(np.float32), probabilities.astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)
