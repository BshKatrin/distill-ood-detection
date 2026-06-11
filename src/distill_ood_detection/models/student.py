"""Student model definitions."""

from __future__ import annotations

from functools import reduce
from operator import mul

import torch
from torch import nn
from torchvision.ops import MLP

from distill_ood_detection.config import StudentConfig


class LinearStudent(nn.Module):
    """A single linear classifier over flattened image pixels."""

    def __init__(self, input_shape: tuple[int, ...], num_classes: int) -> None:
        super().__init__()
        input_dim = reduce(mul, input_shape, 1)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return student logits for a batch of images."""

        return self.classifier(torch.flatten(images, start_dim=1))


class MLPStudent(nn.Module):
    """A configurable multilayer perceptron over flattened image pixels."""

    def __init__(
        self,
        input_shape: tuple[int, ...],
        num_classes: int,
        hidden_channels: tuple[int, ...],
    ) -> None:
        super().__init__()
        input_dim = reduce(mul, input_shape, 1)
        self.classifier = MLP(input_dim, [*hidden_channels, num_classes])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return student logits for a batch of images."""

        return self.classifier(torch.flatten(images, start_dim=1))


def build_student(config: StudentConfig) -> nn.Module:
    """Create the configured student model."""

    if config.kind == "linear":
        return LinearStudent(config.input_shape, config.num_classes)
    if config.kind == "mlp":
        return MLPStudent(
            input_shape=config.input_shape,
            num_classes=config.num_classes,
            hidden_channels=config.hidden_channels,
        )
    raise ValueError(f"Unsupported student kind: {config.kind}")
