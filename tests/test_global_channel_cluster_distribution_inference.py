"""Tensor-level tests for global channel-cluster distribution export."""

from __future__ import annotations

import torch
from torch import nn

from distill_ood_detection.config import FeatureDenoisingConfig
from distill_ood_detection.distillation.feature_denoising import (
    ChannelGroups,
    collect_channel_group_cluster_improvements,
    per_sample_channel_group_hidden_mse,
)


def test_cluster_hidden_mse_is_per_sample_and_group() -> None:
    targets = torch.zeros(2, 4, 1, 1)
    predictions = torch.tensor(
        [
            [[[1.0]], [[2.0]], [[3.0]], [[4.0]]],
            [[[4.0]], [[3.0]], [[2.0]], [[1.0]]],
        ]
    )
    keep_mask = torch.tensor(
        [
            [[[0.0]], [[1.0]], [[0.0]], [[1.0]]],
            [[[1.0]], [[0.0]], [[1.0]], [[0.0]]],
        ]
    )
    groups = ChannelGroups(
        membership=torch.tensor(
            [[True, True, False, False], [False, False, True, True]]
        ),
        distance_threshold=0.5,
        source_path="toy.pt",
    )

    errors = per_sample_channel_group_hidden_mse(
        predictions,
        targets,
        keep_mask,
        groups,
    )

    torch.testing.assert_close(errors, torch.tensor([[1.0, 9.0], [9.0, 1.0]]))


def test_cluster_improvements_average_draws_per_image() -> None:
    class IdentityForwarder(nn.Module):
        def forward_to_features(self, images: torch.Tensor) -> torch.Tensor:
            return images

    class ZeroResidualStudent(nn.Module):
        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(features)

    features = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4, 1, 1)
    labels = torch.arange(4)
    groups = ChannelGroups(
        membership=torch.tensor(
            [[True, True, False, False], [False, False, True, True]]
        ),
        distance_threshold=0.5,
        source_path="toy.pt",
    )
    config = FeatureDenoisingConfig(
        method="channel_group_stratified_masked_residual_reconstruction",
        channel_group_min_size=1,
        channel_group_mask_fraction=0.5,
        evaluation_draws=3,
    )

    result = collect_channel_group_cluster_improvements(
        student=ZeroResidualStudent(),
        loader=[(features, labels)],
        device=torch.device("cpu"),
        perturbation_forwarder=IdentityForwarder(),
        feature_denoising_config=config,
        channel_groups=groups,
    )

    assert tuple(result["raw_reconstruction_error"].shape) == (4, 2)
    assert tuple(result["absolute_improvement"].shape) == (4, 2)
    assert tuple(result["relative_improvement"].shape) == (4, 2)
    torch.testing.assert_close(
        result["absolute_improvement"],
        torch.zeros(4, 2),
    )
    torch.testing.assert_close(
        result["relative_improvement"],
        torch.zeros(4, 2),
    )
    assert bool((result["raw_reconstruction_error"] > 0).all())
