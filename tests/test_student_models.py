"""Tests for neural-network student models."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from distill_ood_detection.config import StudentConfig, load_config, parse_config
from distill_ood_detection.models.student import MLPStudent, build_student


class StudentModelTests(unittest.TestCase):
    """Validate configurable PyTorch student models."""

    def test_loads_mlp_config(self) -> None:
        config = load_config(Path("configs/baseline/cifar_10/mlp.yaml"))

        self.assertEqual(config.student.kind, "mlp")
        self.assertEqual(config.student.hidden_channels, (1024, 512, 256))
        self.assertEqual(
            config.training.enabled_methods(),
            ("cross_entropy", "mse_logits"),
        )

    def test_loads_feature_linear_config(self) -> None:
        config = load_config(Path("configs/baseline/cifar_10/feature_linear_layer3.yaml"))

        self.assertEqual(config.student.kind, "linear")
        self.assertEqual(config.student.feature_layer, "layer3")
        self.assertEqual(config.student.input_shape, (256, 8, 8))
        self.assertEqual(config.student.hidden_channels, ())

    def test_builds_mlp_student(self) -> None:
        config = StudentConfig(
            kind="mlp",
            input_shape=(3, 32, 32),
            hidden_channels=(64, 32, 16),
            num_classes=10,
        )

        student = build_student(config)
        logits = student(torch.zeros(2, 3, 32, 32))

        self.assertIsInstance(student, MLPStudent)
        self.assertEqual(tuple(logits.shape), (2, 10))

    def test_rejects_mlp_without_hidden_channels(self) -> None:
        raw = {
            "experiment_name": "bad_mlp_config",
            "student": {
                "kind": "mlp",
                "input_shape": [3, 32, 32],
                "num_classes": 10,
            },
        }

        with self.assertRaises(ValueError):
            parse_config(raw)

    def test_rejects_feature_layer_for_random_forest(self) -> None:
        raw = {
            "experiment_name": "bad_feature_tree_config",
            "student": {
                "kind": "random_forest",
                "feature_layer": "layer3",
            },
        }

        with self.assertRaises(ValueError):
            parse_config(raw)

    def test_parses_kl_divergence_and_perturbation_strategy(self) -> None:
        raw = {
            "experiment_name": "perturbation_config",
            "student": {
                "kind": "linear",
                "feature_layer": "layer3",
                "input_shape": [16385],
                "num_classes": 10,
            },
            "strategy": {
                "name": "perturbation",
                "perturbation": {
                    "u_min": 0.2,
                    "u_max": 0.8,
                    "clipping_mode": "channel_dependent",
                    "evaluation_draws": 3,
                },
            },
            "training": {
                "methods": {
                    "kl_divergence": {},
                },
            },
        }

        config = parse_config(raw)

        self.assertEqual(config.strategy.name, "perturbation")
        self.assertEqual(config.strategy.perturbation.clipping_mode, "channel_dependent")
        self.assertEqual(config.strategy.perturbation.evaluation_draws, 3)
        self.assertEqual(config.student.input_shape, (16385,))
        self.assertEqual(config.training.enabled_methods(), ("kl_divergence",))

    def test_rejects_perturbation_strategy_without_feature_layer(self) -> None:
        raw = {
            "experiment_name": "bad_perturbation_config",
            "strategy": {"name": "perturbation"},
        }

        with self.assertRaises(ValueError):
            parse_config(raw)


if __name__ == "__main__":
    unittest.main()
