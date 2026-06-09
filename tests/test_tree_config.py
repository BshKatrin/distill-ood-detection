"""Tests for tree-based distillation configuration."""

from __future__ import annotations

import unittest
from pathlib import Path

from distill_ood_detection.config import load_config, parse_config


class TreeConfigTests(unittest.TestCase):
    """Validate random-forest student configuration parsing."""

    def test_loads_random_forest_config(self) -> None:
        config = load_config(Path("configs/distill_random_forest_cifar10.yaml"))

        self.assertEqual(config.student.kind, "random_forest")
        self.assertEqual(config.training.enabled_methods(), ())
        self.assertEqual(config.tree.enabled_modes(), ("logits",))
        self.assertEqual(config.tree.random_forest.n_estimators, 200)

    def test_rejects_cross_entropy_mode(self) -> None:
        raw = {
            "experiment_name": "bad_tree_config",
            "student": {"kind": "random_forest"},
            "tree": {
                "methods": {
                    "cross_entropy": {},
                }
            },
        }

        with self.assertRaises(ValueError):
            parse_config(raw)

    def test_rejects_legacy_softmax_mode(self) -> None:
        raw = {
            "experiment_name": "bad_tree_config",
            "student": {"kind": "random_forest"},
            "tree": {
                "methods": {
                    "softmax": {},
                }
            },
        }

        with self.assertRaises(ValueError):
            parse_config(raw)

    def test_rejects_temperature_for_logits_mode(self) -> None:
        raw = {
            "experiment_name": "bad_tree_config",
            "tree": {
                "methods": {
                    "logits": {"temperature": 2.0},
                }
            },
        }

        with self.assertRaises(ValueError):
            parse_config(raw)


if __name__ == "__main__":
    unittest.main()
