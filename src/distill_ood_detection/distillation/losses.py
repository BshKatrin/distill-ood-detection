"""Loss functions for student distillation."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from distill_ood_detection.config import DistillationMethod


def distillation_loss(
    method: DistillationMethod,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> torch.Tensor:
    """Compute the configured distillation objective."""

    teacher_probabilities = F.softmax(teacher_logits, dim=1)
    if method == "mse":
        teacher_probabilities = F.softmax(teacher_logits, dim=1)
        student_probabilities = F.softmax(student_logits, dim=1)
        return F.mse_loss(student_probabilities, teacher_probabilities)
    if method == "cross_entropy_probabilities":
        student_log_probabilities = F.log_softmax(student_logits, dim=1)
        return -(teacher_probabilities * student_log_probabilities).sum(dim=1).mean()
    raise ValueError(f"Unsupported distillation method: {method}")
