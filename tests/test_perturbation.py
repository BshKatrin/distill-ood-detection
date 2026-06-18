"""Tests for perturbation-based distillation helpers."""

from __future__ import annotations

import unittest

import torch

from distill_ood_detection.config import PerturbationConfig
from distill_ood_detection.distillation.perturbation import (
    _clip_single_feature_map,
    build_unperturbed_perturbation_batch,
    sample_clipping_perturbation,
)


class PerturbationTests(unittest.TestCase):
    """Validate clipping perturbation behavior."""

    def test_clipping_modes_return_perturbation_aware_inputs(self) -> None:
        features = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
        for clipping_mode, u_shape, input_shape in (
            ("constant", (2, 1), (2, 49)),
            ("spatial_dependent", (2, 16), (2, 64)),
            ("channel_dependent", (2, 3), (2, 51)),
        ):
            with self.subTest(clipping_mode=clipping_mode):
                torch.manual_seed(123)
                config = PerturbationConfig(
                    u_min=0.25,
                    u_max=0.75,
                    clipping_mode=clipping_mode,
                )

                batch = sample_clipping_perturbation(features, config)

                self.assertEqual(tuple(batch.perturbed_features.shape), tuple(features.shape))
                self.assertEqual(tuple(batch.perturbations.shape), u_shape)
                self.assertEqual(tuple(batch.student_inputs.shape), input_shape)
                self.assertTrue(torch.all(batch.perturbed_features <= features))
                self.assertTrue(torch.all(batch.perturbations >= 0.25))
                self.assertTrue(torch.all(batch.perturbations <= 0.75))
                torch.testing.assert_close(
                    batch.student_inputs[:, :48],
                    torch.flatten(batch.perturbed_features, start_dim=1),
                )
                torch.testing.assert_close(
                    batch.student_inputs[:, 48:],
                    batch.perturbations,
                )

    def test_rejects_non_convolutional_features(self) -> None:
        config = PerturbationConfig()

        with self.assertRaises(ValueError):
            sample_clipping_perturbation(torch.zeros(2, 3), config)

    def test_unperturbed_batch_preserves_features_with_neutral_perturbation(self) -> None:
        features = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
        config = PerturbationConfig(clipping_mode="channel_dependent")

        batch = build_unperturbed_perturbation_batch(features, config)

        torch.testing.assert_close(batch.perturbed_features, features)
        torch.testing.assert_close(batch.percentiles, torch.ones(2, 3))
        torch.testing.assert_close(batch.perturbations, torch.ones(2, 3))
        torch.testing.assert_close(batch.student_inputs[:, :48], torch.flatten(features, start_dim=1))
        torch.testing.assert_close(batch.student_inputs[:, 48:], torch.ones(2, 3))

    def test_vectorized_clipping_matches_torch_quantile_reference(self) -> None:
        feature_map = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(3, 4, 4)

        spatial_percentiles = torch.linspace(0.1, 0.9, steps=16).reshape(4, 4)
        spatial_actual = _clip_single_feature_map(
            feature_map,
            spatial_percentiles,
            PerturbationConfig(clipping_mode="spatial_dependent"),
        )
        spatial_thresholds = torch.stack(
            [
                torch.stack(
                    [
                        torch.quantile(feature_map[:, row, column], spatial_percentiles[row, column])
                        for column in range(4)
                    ]
                )
                for row in range(4)
            ]
        )
        spatial_expected = torch.minimum(feature_map, spatial_thresholds.unsqueeze(0))
        torch.testing.assert_close(spatial_actual, spatial_expected)

        channel_percentiles = torch.tensor([0.1, 0.5, 0.9])
        channel_actual = _clip_single_feature_map(
            feature_map,
            channel_percentiles,
            PerturbationConfig(clipping_mode="channel_dependent"),
        )
        channel_thresholds = torch.stack(
            [
                torch.quantile(feature_map[channel].flatten(), channel_percentiles[channel])
                for channel in range(3)
            ]
        )
        channel_expected = torch.minimum(feature_map, channel_thresholds[:, None, None])
        torch.testing.assert_close(channel_actual, channel_expected)


if __name__ == "__main__":
    unittest.main()
