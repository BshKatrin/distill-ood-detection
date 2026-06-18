"""Tests for probability inference experiment helpers."""

from __future__ import annotations

import unittest
from pathlib import Path

from distill_ood_detection.config import parse_config
from distill_ood_detection.experiments.infer_probabilities import _probability_output_dir


class ProbabilityInferenceTests(unittest.TestCase):
    """Validate probability artifact path choices."""

    def test_baseline_probability_output_dir_keeps_legacy_path(self) -> None:
        config = parse_config({"experiment_name": "baseline_config"})

        output_dir = _probability_output_dir(Path("runs/baseline_config"), config, False)

        self.assertEqual(output_dir, Path("runs/baseline_config/probabilities"))

    def test_perturbation_probability_output_dirs_are_mode_specific(self) -> None:
        config = parse_config(
            {
                "experiment_name": "perturbation_config",
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


if __name__ == "__main__":
    unittest.main()
