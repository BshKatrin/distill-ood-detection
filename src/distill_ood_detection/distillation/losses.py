"""Loss functions for student distillation."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from distill_ood_detection.config import DistillationMethod


def _center_logits(logits: torch.Tensor) -> torch.Tensor:
    """Center logits per sample to remove constant-offset invariance."""

    return logits - logits.mean(dim=1, keepdim=True)


def distillation_loss(
    method: DistillationMethod,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor | None = None,
    temperature: float = 1.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Compute the configured distillation objective."""

    teacher_probabilities = F.softmax(teacher_logits, dim=1)
    if method == "mse_softmax":
        student_probabilities = F.softmax(student_logits, dim=1)
        return F.mse_loss(student_probabilities, teacher_probabilities)

    if method == "mse_logits":
        centered_student_logits = _center_logits(student_logits)
        centered_teacher_logits = _center_logits(teacher_logits)
        return F.mse_loss(centered_student_logits, centered_teacher_logits)

    if method == "cross_entropy":
        if labels is None:
            raise ValueError("labels are required for cross_entropy distillation")
        scaled_teacher_probabilities = F.softmax(teacher_logits / temperature, dim=1)
        scaled_student_log_probabilities = F.log_softmax(student_logits / temperature, dim=1)
        loss_soft = (
            -(scaled_teacher_probabilities * scaled_student_log_probabilities).sum(dim=1).mean()
        )
        loss_hard = F.cross_entropy(student_logits, labels)
        return alpha * loss_soft + (1.0 - alpha) * loss_hard
    raise ValueError(f"Unsupported distillation method: {method}")
