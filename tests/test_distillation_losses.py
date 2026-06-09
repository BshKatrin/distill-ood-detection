"""Tests for distillation loss formulas."""

from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

from distill_ood_detection.distillation.losses import distillation_loss


class DistillationLossTests(unittest.TestCase):
    """Validate configured distillation objectives."""

    def test_cross_entropy_scales_soft_loss_by_temperature_squared(self) -> None:
        student_logits = torch.tensor([[1.0, 0.0, -1.0], [0.5, 2.0, -0.5]])
        teacher_logits = torch.tensor([[0.0, 2.0, 1.0], [1.5, -0.5, 0.0]])
        labels = torch.tensor([1, 0])
        temperature = 4.0
        alpha = 0.7

        teacher_probabilities = F.softmax(teacher_logits / temperature, dim=1)
        student_log_probabilities = F.log_softmax(student_logits / temperature, dim=1)
        loss_soft = -(teacher_probabilities * student_log_probabilities).sum(dim=1).mean()
        loss_hard = F.cross_entropy(student_logits, labels)
        expected = alpha * (temperature**2) * loss_soft + (1.0 - alpha) * loss_hard

        actual = distillation_loss(
            "cross_entropy",
            student_logits,
            teacher_logits,
            labels=labels,
            temperature=temperature,
            alpha=alpha,
        )

        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
