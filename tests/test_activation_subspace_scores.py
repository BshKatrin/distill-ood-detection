"""Tests for activation-subspace OOD Score export helpers."""

from __future__ import annotations

import unittest

import torch

from distill_ood_detection.cli import build_parser
from distill_ood_detection.experiments.export_activation_subspace_scores import (
    DECISIVE_SCORE_SIGNS,
    INSIGNIFICANT_SCORE_SIGNS,
    decisive_score_vectors,
    detection_metrics_by_score,
    insignificant_score_vectors,
)


class ActivationSubspaceScoreTests(unittest.TestCase):
    """Validate score definitions, signs, and CLI config grouping."""

    def test_decisive_scores_follow_documented_signs(self) -> None:
        teacher_logits = torch.tensor([[2.0, 0.0], [0.5, 1.0]])
        student_logits = torch.tensor([[1.5, 0.5], [1.0, 0.0]])

        raw_metrics, scores = decisive_score_vectors(
            teacher_logits,
            student_logits,
        )

        self.assertEqual(set(raw_metrics), set(DECISIVE_SCORE_SIGNS))
        for name, sign in DECISIVE_SCORE_SIGNS.items():
            with self.subTest(name=name):
                torch.testing.assert_close(scores[name], raw_metrics[name] * sign)

    def test_insignificant_scores_use_confirmed_relative_l2_definition(self) -> None:
        targets = torch.tensor([[3.0, 4.0], [1.0, 0.0]])
        reconstructions = torch.tensor([[0.0, 0.0], [1.0, 0.0]])

        raw_metrics, scores = insignificant_score_vectors(
            targets,
            reconstructions,
        )

        torch.testing.assert_close(
            raw_metrics["raw_reconstruction_error"],
            torch.tensor([12.5, 0.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            raw_metrics["relative_reconstruction_error"],
            torch.tensor([1.0, 0.0], dtype=torch.float64),
        )
        for name, sign in INSIGNIFICANT_SCORE_SIGNS.items():
            with self.subTest(name=name):
                torch.testing.assert_close(scores[name], raw_metrics[name] * sign)

    def test_cli_accepts_score_export_configs(self) -> None:
        args = build_parser().parse_args(
            [
                "export-activation-subspace-scores",
                "--config",
                "first.yaml",
                "--config",
                "second.yaml",
                "--teacher-embedding-dir",
                "runs/teachers/example",
            ]
        )

        self.assertEqual(len(args.config), 2)
        self.assertEqual(args.checkpoint, "best")

    def test_detection_metrics_include_per_ood_and_macro_values(self) -> None:
        results = detection_metrics_by_score(
            id_scores={"example": torch.tensor([0.8, 0.9, 1.0])},
            ood_scores_by_dataset={
                "far_test": {"example": torch.tensor([0.0, 0.1])},
                "near_test": {"example": torch.tensor([0.2, 0.3])},
            },
        )

        self.assertEqual(len(results["example"]["ood"]), 2)
        self.assertEqual(results["example"]["macro"]["roc_auc"], 1.0)
        self.assertEqual(results["example"]["macro"]["fpr_at_95_tpr"], 0.0)


if __name__ == "__main__":
    unittest.main()
