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
    perturbed = _clip_feature_batch(features, percentiles, config)
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


def _clip_feature_batch(
    features: torch.Tensor,
    percentiles: torch.Tensor,
    config: PerturbationConfig,
) -> torch.Tensor:
    batch_size, channels, height, width = features.shape
    if config.clipping_mode == "constant":
        thresholds = _rowwise_quantile(features.reshape(batch_size, -1), percentiles.squeeze(1))
        return torch.minimum(features, thresholds[:, None, None, None])
    if config.clipping_mode == "spatial_dependent":
        spatial_values = features.permute(0, 2, 3, 1).reshape(
            batch_size * height * width,
            channels,
        )
        thresholds = _rowwise_quantile(
            spatial_values,
            percentiles.reshape(batch_size * height * width),
        ).reshape(batch_size, height, width)
        return torch.minimum(features, thresholds[:, None, :, :])
    if config.clipping_mode == "channel_dependent":
        channel_values = features.reshape(batch_size * channels, height * width)
        thresholds = _rowwise_quantile(
            channel_values,
            percentiles.reshape(batch_size * channels),
        ).reshape(batch_size, channels)
        return torch.minimum(features, thresholds[:, :, None, None])
    raise ValueError(f"Unsupported clipping mode: {config.clipping_mode}")


def _clip_single_feature_map(
    feature_map: torch.Tensor,
    percentiles: torch.Tensor,
    config: PerturbationConfig,
) -> torch.Tensor:
    if config.clipping_mode == "constant":
        batched_percentiles = percentiles.reshape(1, 1)
    else:
        batched_percentiles = percentiles.unsqueeze(0)
    return _clip_feature_batch(feature_map.unsqueeze(0), batched_percentiles, config).squeeze(0)


def _rowwise_quantile(values: torch.Tensor, percentiles: torch.Tensor) -> torch.Tensor:
    """Compute one linear-interpolated quantile per row."""

    if values.ndim != 2:
        raise ValueError("values must have shape (rows, columns)")
    if percentiles.ndim != 1 or percentiles.shape[0] != values.shape[0]:
        raise ValueError("percentiles must have one value per row")
    sorted_values = values.sort(dim=1).values
    positions = percentiles * (values.shape[1] - 1)
    lower_indices = positions.floor().long()
    upper_indices = positions.ceil().long()
    weights = positions - lower_indices.to(dtype=positions.dtype)
    lower_values = sorted_values.gather(1, lower_indices[:, None]).squeeze(1)
    upper_values = sorted_values.gather(1, upper_indices[:, None]).squeeze(1)
    return lower_values + weights * (upper_values - lower_values)
