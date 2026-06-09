"""Tests for tree student training targets."""

from __future__ import annotations

import unittest

import numpy as np

from distill_ood_detection.experiments.train_tree_student import _target_matrix


class TreeStudentTrainingTests(unittest.TestCase):
    """Validate random-forest distillation target construction."""

    def test_logits_mode_centers_teacher_logits_per_sample(self) -> None:
        logits = np.array(
            [
                [1.0, 2.0, 6.0],
                [-3.0, 3.0, 9.0],
            ],
            dtype=np.float32,
        )

        targets = _target_matrix(
            mode="logits",
            teacher_logits=logits,
        )

        expected = logits - logits.mean(axis=1, keepdims=True)
        np.testing.assert_allclose(targets, expected)
        np.testing.assert_allclose(targets.mean(axis=1), np.zeros(2), atol=1e-7)

    def test_rejects_removed_cross_entropy_mode(self) -> None:
        logits = np.array(
            [
                [1.0, 2.0, 0.0],
                [-1.0, 0.0, 3.0],
            ],
            dtype=np.float32,
        )

        with self.assertRaises(ValueError):
            _target_matrix(mode="cross_entropy", teacher_logits=logits)


if __name__ == "__main__":
    unittest.main()
