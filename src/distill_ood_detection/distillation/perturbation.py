"""Perturbation helpers for stochastic distillation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from distill_ood_detection.config import PerturbationConfig


@dataclass(frozen=True)
class PerturbationBatch:
    """Perturbed teacher features and perturbation-aware student inputs."""

    student_inputs: torch.Tensor
    original_features: torch.Tensor
    perturbations: torch.Tensor
    perturbed_features: torch.Tensor
    percentiles: torch.Tensor


def sample_clipping_perturbation(
    features: torch.Tensor,
    config: PerturbationConfig,
) -> PerturbationBatch:
    """Sample clipping perturbations and concatenate ``z_tilde`` with ``u``.

    The shape of ``u`` follows the clipping mode: scalar for ``constant``,
    spatial map for ``spatial_dependent``, and channel vector for
    ``channel_dependent``.
    """

    if features.ndim != 4:
        raise ValueError(
            "clipping perturbation expects convolutional features with shape "
            "(batch, channels, height, width)"
        )
    percentiles = _sample_percentiles(features, config)
    perturbed = torch.stack(
        [
            _clip_single_feature_map(features[index], percentiles[index], config)
            for index in range(features.shape[0])
        ],
        dim=0,
    )
    perturbations = torch.flatten(percentiles, start_dim=1)
    student_inputs = torch.cat(
        [
            torch.flatten(perturbed, start_dim=1),
            perturbations,
        ],
        dim=1,
    )
    return PerturbationBatch(
        student_inputs=student_inputs,
        original_features=features,
        perturbations=perturbations,
        perturbed_features=perturbed,
        percentiles=percentiles,
    )


def _sample_percentiles(
    features: torch.Tensor,
    config: PerturbationConfig,
) -> torch.Tensor:
    batch_size, channels, height, width = features.shape
    if config.clipping_mode == "constant":
        shape = (batch_size, 1)
    elif config.clipping_mode == "spatial_dependent":
        shape = (batch_size, height, width)
    elif config.clipping_mode == "channel_dependent":
        shape = (batch_size, channels)
    else:
        raise ValueError(f"Unsupported clipping mode: {config.clipping_mode}")
    return torch.empty(
        shape,
        device=features.device,
        dtype=features.dtype,
    ).uniform_(config.u_min, config.u_max)


def _clip_single_feature_map(
    feature_map: torch.Tensor,
    percentiles: torch.Tensor,
    config: PerturbationConfig,
) -> torch.Tensor:
    if config.clipping_mode == "constant":
        threshold = torch.quantile(feature_map.flatten(), percentiles.squeeze())
        return torch.minimum(feature_map, threshold)
    if config.clipping_mode == "spatial_dependent":
        _, height, width = feature_map.shape
        thresholds = torch.empty(
            (height, width),
            device=feature_map.device,
            dtype=feature_map.dtype,
        )
        for row in range(height):
            for column in range(width):
                thresholds[row, column] = torch.quantile(
                    feature_map[:, row, column],
                    percentiles[row, column],
                )
        return torch.minimum(feature_map, thresholds.unsqueeze(0))
    if config.clipping_mode == "channel_dependent":
        channels = feature_map.shape[0]
        flattened_channels = feature_map.reshape(channels, -1)
        thresholds = torch.empty(
            channels,
            device=feature_map.device,
            dtype=feature_map.dtype,
        )
        for channel in range(channels):
            thresholds[channel] = torch.quantile(
                flattened_channels[channel],
                percentiles[channel],
            )
        return torch.minimum(feature_map, thresholds[:, None, None])
    raise ValueError(f"Unsupported clipping mode: {config.clipping_mode}")
