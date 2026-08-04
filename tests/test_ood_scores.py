"""Tests for teacher-student OOD score functions."""

from __future__ import annotations

import unittest

import numpy as np

from distill_ood_detection.evaluation import (
    absolute_energy_gap,
    absolute_max_probability_difference,
    energy,
    energy_gap,
    logit_l2_distance,
    max_probability_difference,
    student_energy,
    student_msp,
    student_teacher_kl_divergence,
    student_teacher_kl_divergence_from_logits,
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

    def test_student_teacher_kl_divergence_from_logits(self) -> None:
        teacher_logits = np.array([[2.0, 1.0], [-1.0, 3.0]])
        student_logits = np.array([[1.0, 2.0], [0.0, 2.0]])
        teacher = np.exp(teacher_logits) / np.exp(teacher_logits).sum(
            axis=-1,
            keepdims=True,
        )
        student = np.exp(student_logits) / np.exp(student_logits).sum(
            axis=-1,
            keepdims=True,
        )

        np.testing.assert_allclose(
            student_teacher_kl_divergence_from_logits(
                teacher_logits,
                student_logits,
            ),
            student_teacher_kl_divergence(teacher, student),
        )

    def test_logit_kl_remains_finite_after_probability_underflow(self) -> None:
        teacher_logits = np.array([[0.0, -10.0]])
        student_logits = np.array([[0.0, -1000.0]])

        divergence = student_teacher_kl_divergence_from_logits(
            teacher_logits,
            student_logits,
        )

        self.assertTrue(np.isfinite(divergence).all())

    def test_logit_l2_distance_uses_centered_logits(self) -> None:
        teacher = np.array([[1.0, 2.0, 3.0], [0.0, 4.0, 8.0]])
        student = np.array([[6.0, 7.0, 8.0], [2.0, 4.0, 6.0]])

        np.testing.assert_allclose(
            logit_l2_distance(teacher, student),
            np.array([0.0, np.sqrt(8.0)]),
        )

    def test_energy_uses_temperature_scaled_logsumexp(self) -> None:
        logits = np.array([[1.0, 2.0, 3.0], [0.0, -1.0, -2.0]])
        temperature = 2.0
        shifted = logits / temperature
        expected = temperature * (
            shifted.max(axis=1)
            + np.log(np.exp(shifted - shifted.max(axis=1, keepdims=True)).sum(axis=1))
        )

        np.testing.assert_allclose(
            energy(logits, temperature=temperature),
            expected,
        )

    def test_energy_gap(self) -> None:
        teacher = np.array([[1.0, 2.0], [0.0, 4.0]])
        student = np.array([[0.0, 1.0], [1.0, 1.0]])

        np.testing.assert_allclose(
            energy_gap(teacher, student),
            energy(teacher) - energy(student),
        )
        np.testing.assert_allclose(
            absolute_energy_gap(teacher, student),
            np.abs(energy(teacher) - energy(student)),
        )

    def test_student_only_scores(self) -> None:
        probabilities = np.array([[0.7, 0.3], [0.1, 0.9]])
        logits = np.array([[1.0, 2.0], [0.0, 4.0]])

        np.testing.assert_allclose(student_msp(probabilities), [0.7, 0.9])
        np.testing.assert_allclose(student_energy(logits), energy(logits))

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
        np.testing.assert_allclose(
            energy(teacher_logits, signed=True),
            energy(teacher_logits),
        )
        np.testing.assert_allclose(
            energy_gap(teacher_logits, student_logits, signed=True),
            energy_gap(teacher_logits, student_logits),
        )
        np.testing.assert_allclose(
            absolute_energy_gap(teacher_logits, student_logits, signed=True),
            -absolute_energy_gap(teacher_logits, student_logits),
        )

    def test_defines_sign_for_every_ood_score(self) -> None:
        self.assertEqual(
            set(SIGNS),
            {
                "absolute_max_probability_difference",
                "absolute_energy_gap",
                "energy",
                "energy_gap",
                "logit_l2_distance",
                "max_probability_difference",
                "student_teacher_kl_divergence",
                "student_msp",
                "student_energy",
                "feature_denoising_pca_reconstruction_error",
                "feature_denoising_spatial_reconstruction_error",
                "feature_denoising_spatial_block_residual_reconstruction_error",
                "feature_denoising_channel_reconstruction_error",
                "feature_denoising_channel_residual_reconstruction_error",
                "feature_denoising_confusion_channel_replacement_reconstruction_error",
                "feature_denoising_confusion_channel_replacement_residual_reconstruction_error",
                "feature_denoising_spatial_token_prediction_error",
                "feature_denoising_pixel_embedding_prediction_error",
                "feature_denoising_pixel_augmented_embedding_prediction_error",
                "feature_denoising_pixel_multilayer_prediction_error",
                "feature_denoising_pixel_multilayer_l234_prediction_error",
                "ensemble_predictive_entropy",
                "ensemble_bald",
            },
        )

    def test_rejects_shape_mismatch(self) -> None:
        teacher = np.array([[0.7, 0.3]])
        student = np.array([[0.4, 0.3], [0.6, 0.2]])

        with self.assertRaises(ValueError):
            max_probability_difference(teacher, student)

        with self.assertRaises(ValueError):
            logit_l2_distance(teacher, student)

        with self.assertRaises(ValueError):
            energy_gap(teacher, student)

    def test_rejects_invalid_energy_temperature(self) -> None:
        logits = np.array([[1.0, 2.0]])

        with self.assertRaises(ValueError):
            energy(logits, temperature=0.0)

        with self.assertRaises(ValueError):
            energy(logits, temperature=-1.0)

    def test_student_teacher_kl_divergence_rejects_zero_student_support(self) -> None:
        teacher = np.array([[0.8, 0.2]])
        student = np.array([[1.0, 0.0]])

        with self.assertRaises(ValueError):
            student_teacher_kl_divergence(teacher, student)

    def test_averages_per_draw_scores_for_perturbation_outputs(self) -> None:
        teacher = np.array(
            [
                [[0.7, 0.3], [0.6, 0.4]],
                [[0.1, 0.9], [0.3, 0.7]],
            ]
        )
        student = np.array(
            [
                [[0.4, 0.6], [0.5, 0.5]],
                [[0.2, 0.8], [0.4, 0.6]],
            ]
        )

        np.testing.assert_allclose(
            max_probability_difference(teacher, student),
            np.array([0.1, 0.1]),
        )
        np.testing.assert_allclose(
            max_probability_difference(teacher, student, average_draws=False),
            np.array([[0.1, 0.1], [0.1, 0.1]]),
        )

        per_draw_kl = np.sum(teacher * np.log(teacher / student), axis=-1)
        np.testing.assert_allclose(
            student_teacher_kl_divergence(teacher, student),
            per_draw_kl.mean(axis=1),
        )

    def test_averages_per_draw_energy_scores_for_perturbation_outputs(self) -> None:
        teacher = np.array(
            [
                [[1.0, 3.0], [2.0, 4.0]],
                [[0.0, 4.0], [1.0, 5.0]],
            ]
        )
        student = np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[1.0, 3.0], [1.0, 4.0]],
            ]
        )

        per_draw_energy = energy(teacher, average_draws=False)
        per_draw_gap = energy_gap(teacher, student, average_draws=False)

        np.testing.assert_allclose(
            energy(teacher),
            per_draw_energy.mean(axis=1),
        )
        np.testing.assert_allclose(
            energy_gap(teacher, student),
            per_draw_gap.mean(axis=1),
        )
        np.testing.assert_allclose(
            absolute_energy_gap(teacher, student),
            np.abs(per_draw_gap).mean(axis=1),
        )

    def test_averages_per_draw_logit_scores_for_perturbation_outputs(self) -> None:
        teacher = np.array(
            [
                [[1.0, 3.0], [2.0, 4.0]],
                [[0.0, 4.0], [1.0, 5.0]],
            ]
        )
        student = np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[1.0, 3.0], [1.0, 4.0]],
            ]
        )

        per_draw = logit_l2_distance(teacher, student, average_draws=False)

        np.testing.assert_allclose(
            logit_l2_distance(teacher, student),
            per_draw.mean(axis=1),
        )


if __name__ == "__main__":
    unittest.main()
