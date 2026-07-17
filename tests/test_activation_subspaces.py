"""Tests for activation-subspace Feature Denoising diagnostics."""

from __future__ import annotations

import unittest

import torch

from distill_ood_detection.evaluation.activation_subspaces import (
    classifier_svd,
    feature_denoising_subspace_errors,
    select_balanced_subspace_dimension,
)


class ActivationSubspaceTests(unittest.TestCase):
    """Validate SVD decomposition, balancing, and component errors."""

    def test_classifier_svd_includes_classifier_nullspace(self) -> None:
        weight = torch.tensor([[3.0, 0.0, 0.0], [0.0, 2.0, 0.0]])

        singular_values, right_basis = classifier_svd(weight)

        torch.testing.assert_close(singular_values, torch.tensor([3.0, 2.0]))
        self.assertEqual(tuple(right_basis.shape), (3, 3))
        torch.testing.assert_close(weight @ right_basis[-1], torch.zeros(2))

    def test_selects_split_with_closest_mean_component_norms(self) -> None:
        basis = torch.eye(3)
        activations = torch.tensor([[3.0, 4.0, 0.0], [3.0, 0.0, 4.0]])

        dimension, gaps = select_balanced_subspace_dimension(basis, activations)

        self.assertEqual(dimension, 1)
        torch.testing.assert_close(gaps, torch.tensor([5.0, 1.0, 2.0]))

    def test_computes_errors_and_relative_improvement_per_subspace(self) -> None:
        target = torch.zeros(1, 4)
        context = torch.tensor([[2.0, 0.0, 4.0, 0.0]])
        prediction = torch.tensor([[1.0, 1.0, 2.0, 2.0]])

        scores = feature_denoising_subspace_errors(
            right_basis=torch.eye(4),
            decisive_dimension=2,
            context=context,
            prediction=prediction,
            target=target,
        )

        torch.testing.assert_close(
            scores["decisive_reconstruction_error"], torch.tensor([1.0])
        )
        torch.testing.assert_close(scores["decisive_identity_error"], torch.tensor([2.0]))
        torch.testing.assert_close(
            scores["decisive_relative_improvement"], torch.tensor([0.5])
        )
        torch.testing.assert_close(
            scores["insignificant_reconstruction_error"], torch.tensor([4.0])
        )
        torch.testing.assert_close(
            scores["insignificant_identity_error"], torch.tensor([8.0])
        )
        torch.testing.assert_close(
            scores["insignificant_relative_improvement"], torch.tensor([0.5])
        )

    def test_component_errors_match_explicit_orthogonal_projections(self) -> None:
        generator = torch.Generator().manual_seed(7)
        basis, _triangular = torch.linalg.qr(torch.randn(5, 5, generator=generator))
        context = torch.randn(3, 5, generator=generator)
        prediction = torch.randn(3, 5, generator=generator)
        target = torch.randn(3, 5, generator=generator)

        scores = feature_denoising_subspace_errors(
            right_basis=basis,
            decisive_dimension=2,
            context=context,
            prediction=prediction,
            target=target,
        )

        prediction_coordinates = (prediction - target) @ basis.T
        expected_decisive = prediction_coordinates[:, :2].square().mean(dim=1)
        expected_insignificant = prediction_coordinates[:, 2:].square().mean(dim=1)
        torch.testing.assert_close(
            scores["decisive_reconstruction_error"], expected_decisive
        )
        torch.testing.assert_close(
            scores["insignificant_reconstruction_error"], expected_insignificant
        )


if __name__ == "__main__":
    unittest.main()
