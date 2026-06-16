"""Tests for raw teacher activation export artifacts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from distill_ood_detection.config import parse_teacher_activation_config
from distill_ood_detection.datasets.inference import NamedLoader
from distill_ood_detection.experiments.export_teacher_activations import (
    export_loader_activations,
)


class ToyTeacher(nn.Module):
    """Small teacher with two named activation layers."""

    def __init__(self) -> None:
        super().__init__()
        self.layer3 = nn.Conv2d(3, 4, kernel_size=1)
        self.layer4 = nn.Conv2d(4, 5, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(5, 2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return logits for a batch of images."""

        features = self.layer4(self.layer3(images))
        pooled = torch.flatten(self.pool(features), start_dim=1)
        return self.classifier(pooled)


class TeacherActivationExportTests(unittest.TestCase):
    """Validate teacher activation config parsing and saved artifacts."""

    def test_parses_teacher_activation_config_layers(self) -> None:
        config = parse_teacher_activation_config(
            {
                "experiment_name": "teacher_acts",
                "layers": ["layer3", "layer4"],
                "dataset": {
                    "name": "cifar10",
                    "ood_datasets": [{"name": "mnist", "split": "test"}],
                },
            }
        )

        self.assertEqual(config.layers, ("layer3", "layer4"))
        self.assertEqual(config.dataset.ood_datasets[0].name, "mnist")

    def test_exports_one_pt_file_per_layer(self) -> None:
        teacher = ToyTeacher()
        images = torch.zeros(3, 3, 8, 8)
        labels = torch.tensor([0, 1, 0])
        loader = DataLoader(TensorDataset(images, labels), batch_size=2)
        named_loader = NamedLoader(name="toy_test", split="test", loader=loader)

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            artifacts = export_loader_activations(
                teacher=teacher,
                layers=("layer3", "layer4"),
                named_loader=named_loader,
                output_dir=output_dir,
                device=torch.device("cpu"),
                metadata={"model": "teacher", "dataset": "toy_test", "split": "test"},
            )

            self.assertEqual(
                sorted(path.name for path in (output_dir / "toy_test").glob("*.pt")),
                ["layer3.pt", "layer4.pt"],
            )
            self.assertEqual([artifact["layer"] for artifact in artifacts], ["layer3", "layer4"])

            layer3 = torch.load(
                output_dir / "toy_test" / "layer3.pt",
                map_location="cpu",
                weights_only=False,
            )
            layer4 = torch.load(
                output_dir / "toy_test" / "layer4.pt",
                map_location="cpu",
                weights_only=False,
            )

        self.assertEqual(tuple(layer3["activations"].shape), (3, 4, 8, 8))
        self.assertEqual(tuple(layer4["activations"].shape), (3, 5, 8, 8))
        self.assertEqual(tuple(layer3["labels"].shape), (3,))
        self.assertEqual(layer3["layer"], "layer3")


if __name__ == "__main__":
    unittest.main()
