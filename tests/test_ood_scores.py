"""Tests for teacher-student OOD score functions."""

from __future__ import annotations

import unittest

import numpy as np

from distill_ood_detection.evaluation import (
    absolute_max_probability_difference,
    confidence_ratio,
    max_probability_difference,
)


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

    def test_confidence_ratio(self) -> None:
        teacher = np.array([[0.7, 0.3], [0.1, 0.9]])
        student = np.array([[0.4, 0.6], [0.2, 0.8]])

        np.testing.assert_allclose(
            confidence_ratio(teacher, student),
            np.log(np.array([0.7, 0.9])) - np.log(np.array([0.6, 0.8])),
        )

    def test_rejects_shape_mismatch(self) -> None:
        teacher = np.array([[0.7, 0.3]])
        student = np.array([[0.4, 0.3], [0.6, 0.2]])

        with self.assertRaises(ValueError):
            max_probability_difference(teacher, student)

    def test_confidence_ratio_rejects_zero_max_probability(self) -> None:
        teacher = np.array([[0.0, 0.0]])
        student = np.array([[0.4, 0.6]])

        with self.assertRaises(ValueError):
            confidence_ratio(teacher, student)


if __name__ == "__main__":
    unittest.main()
