"""Classification metrics."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def accuracy(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
) -> float:
    """Compute top-1 classification accuracy."""

    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        predictions = model(images).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.numel()
    return correct / total


@torch.no_grad()
def distillation_validation_metrics(
    teacher: nn.Module,
    student: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
) -> dict[str, float]:
    """Compute validation accuracy and KL divergence from teacher to student."""

    teacher.eval()
    student.eval()
    correct = 0
    total = 0
    total_kl = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        teacher_logits = teacher(images)
        student_logits = student(images)
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

    return {
        "validation_accuracy": correct / total,
        "validation_kl_divergence": total_kl / total,
    }
