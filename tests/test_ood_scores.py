"""Tests for teacher-student OOD score functions."""

from __future__ import annotations

import unittest

import numpy as np

from distill_ood_detection.evaluation import (
    absolute_max_probability_difference,
    logit_l2_distance,
    max_probability_difference,
    student_teacher_kl_divergence,
)
from distill_ood_detection.evaluation.ood_scores import SIGNS


class OODScoreTests(unittest.TestCase):
    """Validate OOD score formulas."""

    def test_max_probability_difference(self) -> None:
        teacher = np.array([[0.7, 0.3], [0.1, 0.9]])
        student = np.array([[0.4, 0.6], [0.2, 0.8]])

        np.testing.assert_allclose(
            max_probability_difference(teacher, student),
            np.array([0.1, 0.1]),
        )

    def test_absolute_max_probability_difference(self) -> None:
        teacher = np.array([[0.7, 0.3], [0.1, 0.9]])
        student = np.array([[0.4, 0.6], [0.95, 0.05]])

        np.testing.assert_allclose(
            absolute_max_probability_difference(teacher, student),
            np.array([0.1, 0.05]),
        )

    def test_student_teacher_kl_divergence(self) -> None:
        teacher = np.array([[0.7, 0.3], [0.1, 0.9]])
        student = np.array([[0.4, 0.6], [0.2, 0.8]])

        np.testing.assert_allclose(
            student_teacher_kl_divergence(teacher, student),
            np.array(
                [
                    0.7 * np.log(0.7 / 0.4) + 0.3 * np.log(0.3 / 0.6),
                    0.1 * np.log(0.1 / 0.2) + 0.9 * np.log(0.9 / 0.8),
                ]
            ),
        )

    def test_logit_l2_distance_uses_centered_logits(self) -> None:
        teacher = np.array([[1.0, 2.0, 3.0], [0.0, 4.0, 8.0]])
        student = np.array([[6.0, 7.0, 8.0], [2.0, 4.0, 6.0]])

        np.testing.assert_allclose(
            logit_l2_distance(teacher, student),
            np.array([0.0, np.sqrt(8.0)]),
        )

    def test_signed_scores_follow_id_positive_convention(self) -> None:
        teacher = np.array([[0.7, 0.3], [0.1, 0.9]])
        student = np.array([[0.4, 0.6], [0.2, 0.8]])
        teacher_logits = np.array([[1.0, 2.0], [3.0, 5.0]])
        student_logits = np.array([[2.0, 2.0], [0.0, 1.0]])

        np.testing.assert_allclose(
            max_probability_difference(teacher, student, signed=True),
            np.array([0.1, 0.1]),
        )
        np.testing.assert_allclose(
            absolute_max_probability_difference(teacher, student, signed=True),
            np.array([-0.1, -0.1]),
        )
        np.testing.assert_allclose(
            student_teacher_kl_divergence(teacher, student, signed=True),
            -student_teacher_kl_divergence(teacher, student),
        )
        np.testing.assert_allclose(
            logit_l2_distance(teacher_logits, student_logits, signed=True),
            -logit_l2_distance(teacher_logits, student_logits),
        )

    def test_defines_sign_for_every_ood_score(self) -> None:
        self.assertEqual(
            set(SIGNS),
            {
                "absolute_max_probability_difference",
                "logit_l2_distance",
                "max_probability_difference",
                "student_teacher_kl_divergence",
            },
        )

    def test_rejects_shape_mismatch(self) -> None:
        teacher = np.array([[0.7, 0.3]])
        student = np.array([[0.4, 0.3], [0.6, 0.2]])

        with self.assertRaises(ValueError):
            max_probability_difference(teacher, student)

        with self.assertRaises(ValueError):
            logit_l2_distance(teacher, student)

    def test_student_teacher_kl_divergence_rejects_zero_student_support(self) -> None:
        teacher = np.array([[0.8, 0.2]])
        student = np.array([[1.0, 0.0]])

        with self.assertRaises(ValueError):
            student_teacher_kl_divergence(teacher, student)


if __name__ == "__main__":
    unittest.main()
