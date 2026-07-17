"""Tests for activation-subspace raw inference export helpers."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from distill_ood_detection.cli import build_parser
from distill_ood_detection.experiments.export_activation_subspace_inference import (
    infer_activation_subspace_student,
)


class ActivationSubspaceInferenceTests(unittest.TestCase):
    """Validate component coordinates, outputs, and CLI grouping."""

    def test_decisive_inference_exports_logits_and_probabilities(self) -> None:
        student = nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            student.weight.copy_(
                torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
            )
        embeddings = torch.tensor([[2.0, 3.0, 5.0, 7.0]])

        outputs = infer_activation_subspace_student(
            student=student,
            embeddings=embeddings,
            right_basis=torch.eye(4),
            decisive_dimension=2,
            component="decisive",
            target="projected_logits",
            batch_size=1,
            device=torch.device("cpu"),
        )

        torch.testing.assert_close(
            outputs["logits"],
            torch.tensor([[2.0, 3.0, -1.0]]),
        )
        torch.testing.assert_close(
            outputs["probabilities"].sum(dim=1),
            torch.ones(1),
        )

    def test_insignificant_inference_exports_reconstructed_coordinates(self) -> None:
        student = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            student.weight.copy_(torch.eye(2))

        outputs = infer_activation_subspace_student(
            student=student,
            embeddings=torch.tensor([[2.0, 3.0, 5.0, 7.0]]),
            right_basis=torch.eye(4),
            decisive_dimension=2,
            component="insignificant",
            target="coordinates",
            batch_size=1,
            device=torch.device("cpu"),
        )

        self.assertEqual(set(outputs), {"reconstructed_coordinates"})
        torch.testing.assert_close(
            outputs["reconstructed_coordinates"],
            torch.tensor([[5.0, 7.0]]),
        )

    def test_decisive_coordinate_inference_exports_reconstruction(self) -> None:
        outputs = infer_activation_subspace_student(
            student=nn.Identity(),
            embeddings=torch.tensor([[2.0, 3.0, 5.0, 7.0]]),
            right_basis=torch.eye(4),
            decisive_dimension=2,
            component="decisive",
            target="coordinates",
            batch_size=1,
            device=torch.device("cpu"),
        )

        self.assertEqual(set(outputs), {"reconstructed_coordinates"})
        torch.testing.assert_close(
            outputs["reconstructed_coordinates"],
            torch.tensor([[2.0, 3.0]]),
        )

    def test_cli_accepts_multiple_configs(self) -> None:
        args = build_parser().parse_args(
            [
                "export-activation-subspace-inference",
                "--config",
                "first.yaml",
                "--config",
                "second.yaml",
                "--teacher-output-dir",
                "runs/teachers/example",
            ]
        )

        self.assertEqual([str(path) for path in args.config], ["first.yaml", "second.yaml"])
        self.assertEqual(args.checkpoint, "best")


if __name__ == "__main__":
    unittest.main()
