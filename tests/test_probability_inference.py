"""Tests for probability inference experiment helpers."""

from __future__ import annotations

import unittest
from pathlib import Path

from distill_ood_detection.config import parse_config
from distill_ood_detection.experiments.infer_probabilities import _probability_output_dir


class ProbabilityInferenceTests(unittest.TestCase):
    """Validate probability artifact path choices."""

    def test_baseline_probability_output_dir_keeps_legacy_path(self) -> None:
        config = parse_config({"experiment_name": "baseline_config", "run_dir": "runs/tests/baseline_config"})

        output_dir = _probability_output_dir(Path("runs/baseline_config"), config, False)

        self.assertEqual(output_dir, Path("runs/baseline_config/probabilities"))

    def test_perturbation_probability_output_dirs_are_mode_specific(self) -> None:
        config = parse_config(
            {
                "experiment_name": "perturbation_config",
                "run_dir": "runs/tests/perturbation_config",
                "student": {
                    "kind": "linear",
                    "feature_layer": "layer3",
                    "input_shape": [16385],
                    "num_classes": 10,
                },
                "strategy": {"name": "perturbation"},
            }
        )

        unperturbed_dir = _probability_output_dir(
            Path("runs/perturbation_config"),
            config,
            False,
        )
        perturbed_dir = _probability_output_dir(
            Path("runs/perturbation_config"),
            config,
            True,
        )

        self.assertEqual(
            unperturbed_dir,
            Path("runs/perturbation_config/probabilities/unperturbed"),
        )
        self.assertEqual(
            perturbed_dir,
            Path("runs/perturbation_config/probabilities/perturbed"),
        )

    def test_aggressive_perturbation_config_uses_lower_percentile_range(self) -> None:
        config = parse_config(
            {
                "experiment_name": "perturbation_config",
                "run_dir": "runs/tests/perturbation_config",
                "student": {
                    "kind": "linear",
                    "feature_layer": "layer3",
                    "input_shape": [16385],
                    "num_classes": 10,
                },
                "strategy": {
                    "name": "perturbation",
                    "perturbation": {
                        "u_min": 0.0,
                        "u_max": 0.5,
                    },
                },
            }
        )

        self.assertEqual(config.strategy.perturbation.u_min, 0.0)
        self.assertEqual(config.strategy.perturbation.u_max, 0.5)

    def test_mc_dropout_perturbation_config_parses_dropout_probability(self) -> None:
        config = parse_config(
            {
                "experiment_name": "mc_dropout_config",
                "run_dir": "runs/tests/mc_dropout_config",
                "student": {
                    "kind": "linear",
                    "feature_layer": "layer4",
                    "input_shape": [8192],
                    "num_classes": 10,
                },
                "strategy": {
                    "name": "perturbation",
                    "perturbation": {
                        "method": "mc_dropout",
                        "dropout_probability": 0.5,
                        "dropout_mode": "channel",
                        "evaluation_draws": 50,
                    },
                },
            }
        )

        self.assertEqual(config.strategy.perturbation.method, "mc_dropout")
        self.assertEqual(config.strategy.perturbation.dropout_probability, 0.5)
        self.assertEqual(config.strategy.perturbation.dropout_mode, "channel")
        self.assertEqual(config.strategy.perturbation.evaluation_draws, 50)

    def test_pca_projection_config_parses_components_and_activation_path(self) -> None:
        config = parse_config(
            {
                "experiment_name": "pca_config",
                "run_dir": "runs/tests/pca_config",
                "student": {
                    "kind": "linear",
                    "feature_layer": "layer4",
                    "input_shape": [128],
                    "num_classes": 10,
                },
                "strategy": {
                    "name": "perturbation",
                    "perturbation": {
                        "method": "pca_projection",
                        "pca_components": 128,
                        "pca_activation_path": "runs/teacher/teacher_activations/cifar10_train/layer4.pt",
                    },
                },
            }
        )

        self.assertEqual(config.strategy.perturbation.method, "pca_projection")
        self.assertEqual(config.strategy.perturbation.pca_components, 128)
        self.assertEqual(
            config.strategy.perturbation.pca_activation_path,
            "runs/teacher/teacher_activations/cifar10_train/layer4.pt",
        )


if __name__ == "__main__":
    unittest.main()
