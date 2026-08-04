"""Tests for predictive-entropy and BALD ensemble OOD Scores."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from distill_ood_detection.cli import build_parser
from distill_ood_detection.evaluation import (
    ensemble_bald,
    ensemble_predictive_entropy,
)
from distill_ood_detection.experiments.export_subspace_ensemble_scores import (
    SUBSPACE_ENSEMBLE_SCORE_SIGNS,
    detection_metrics_by_score,
    subspace_ensemble_score_vectors,
)


class SubspaceEnsembleScoreTests(unittest.TestCase):
    """Validate entropy definitions, signs, aggregation, and CLI commands."""

    def setUp(self) -> None:
        self.probabilities = np.array(
            [
                [[0.8, 0.2], [0.8, 0.2]],
                [[1.0, 0.0], [0.0, 1.0]],
            ]
        )

    def test_predictive_entropy_uses_mean_distribution(self) -> None:
        entropy = ensemble_predictive_entropy(self.probabilities)

        expected_first = -(0.8 * np.log(0.8) + 0.2 * np.log(0.2))
        np.testing.assert_allclose(entropy, [expected_first, np.log(2.0)])

    def test_bald_subtracts_expected_member_entropy(self) -> None:
        bald = ensemble_bald(self.probabilities)

        np.testing.assert_allclose(bald, [0.0, np.log(2.0)])

    def test_signed_scores_are_negative_raw_uncertainties(self) -> None:
        np.testing.assert_allclose(
            ensemble_predictive_entropy(self.probabilities, signed=True),
            -ensemble_predictive_entropy(self.probabilities),
        )
        np.testing.assert_allclose(
            ensemble_bald(self.probabilities, signed=True),
            -ensemble_bald(self.probabilities),
        )

    def test_export_vectors_follow_documented_signs(self) -> None:
        raw, scores = subspace_ensemble_score_vectors(
            torch.tensor(self.probabilities)
        )

        self.assertEqual(set(raw), set(SUBSPACE_ENSEMBLE_SCORE_SIGNS))
        for name, sign in SUBSPACE_ENSEMBLE_SCORE_SIGNS.items():
            torch.testing.assert_close(scores[name], raw[name] * sign)

    def test_detection_metrics_include_macro_values(self) -> None:
        metrics = detection_metrics_by_score(
            id_scores={"bald": torch.tensor([-0.1, -0.2])},
            ood_scores_by_dataset={
                "ood_test": {"bald": torch.tensor([-0.8, -0.9])}
            },
        )

        self.assertEqual(metrics["bald"]["macro"]["roc_auc"], 1.0)
        self.assertEqual(metrics["bald"]["macro"]["fpr_at_95_tpr"], 0.0)

    def test_cli_exposes_inference_and_score_exports(self) -> None:
        inference = build_parser().parse_args(
            [
                "export-subspace-ensemble-inference",
                "--config",
                "ensemble.yaml",
            ]
        )
        scores = build_parser().parse_args(
            [
                "export-subspace-ensemble-scores",
                "--config",
                "ensemble.yaml",
            ]
        )

        self.assertEqual(str(inference.config), "ensemble.yaml")
        self.assertEqual(inference.checkpoint, "best")
        self.assertEqual(str(scores.config), "ensemble.yaml")
        self.assertEqual(scores.checkpoint, "best")


if __name__ == "__main__":
    unittest.main()
