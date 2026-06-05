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

    if method == "cross_entropy_softmax":
        student_log_probabilities = F.log_softmax(student_logits, dim=1)
        return -(teacher_probabilities * student_log_probabilities).sum(dim=1).mean()
    raise ValueError(f"Unsupported distillation method: {method}")
