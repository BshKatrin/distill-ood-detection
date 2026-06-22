"""Perturbation helpers for stochastic distillation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class PcaProjector:
    """Fixed PCA projection from flattened teacher features."""

    mean: torch.Tensor
    components: torch.Tensor

    def to(self, device: torch.device) -> PcaProjector:
        """Move PCA tensors to a device."""

        return PcaProjector(
            mean=self.mean.to(device),
            components=self.components.to(device),
        )

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        """Project convolutional features onto PCA components."""

        flat_features = torch.flatten(features, start_dim=1)
        return (flat_features - self.mean) @ self.components.T


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


def sample_perturbation(
    features: torch.Tensor,
    config: PerturbationConfig,
    pca_projector: PcaProjector | None = None,
) -> PerturbationBatch:
    """Sample the perturbation configured for stochastic distillation."""

    if config.method == "clipping":
        return sample_clipping_perturbation(features, config)
    if config.method == "mc_dropout":
        return sample_mc_dropout_perturbation(features, config)
    if config.method == "pca_projection":
        if pca_projector is None:
            raise ValueError("pca_projector is required for PCA projection")
        return build_pca_projection_batch(features, pca_projector)
    raise ValueError(f"Unsupported perturbation method: {config.method}")


def sample_mc_dropout_perturbation(
    features: torch.Tensor,
    config: PerturbationConfig,
) -> PerturbationBatch:
    """Apply Monte-Carlo dropout to student features.

    The student receives dropped features, while ``perturbed_features`` remains
    the original tensor so the teacher target is the unperturbed continuation.
    """

    if features.ndim != 4:
        raise ValueError(
            "Monte-Carlo dropout perturbation expects convolutional features "
            "with shape (batch, channels, height, width)"
        )
    mask = _sample_dropout_mask(features, config)
    keep_probability = 1.0 - config.dropout_probability
    dropped = features * mask / keep_probability
    return PerturbationBatch(
        student_inputs=torch.flatten(dropped, start_dim=1),
        original_features=features,
        perturbations=torch.flatten(mask.expand_as(features), start_dim=1),
        perturbed_features=features,
        percentiles=mask,
    )


def build_unperturbed_perturbation_batch(
    features: torch.Tensor,
    config: PerturbationConfig,
    pca_projector: PcaProjector | None = None,
) -> PerturbationBatch:
    """Build perturbation-aware inputs without modifying teacher features.

    The neutral perturbation vector is all ones. This preserves the input shape
    expected by perturbation-aware students while leaving ``z`` unchanged.
    """

    if config.method == "mc_dropout":
        return build_unperturbed_mc_dropout_batch(features, config)
    if config.method == "pca_projection":
        if pca_projector is None:
            raise ValueError("pca_projector is required for PCA projection")
        return build_pca_projection_batch(features, pca_projector)
    if features.ndim != 4:
        raise ValueError(
            "unperturbed perturbation inputs expect convolutional features with shape "
            "(batch, channels, height, width)"
        )
    percentiles = torch.ones(
        _percentile_shape(features, config),
        device=features.device,
        dtype=features.dtype,
    )
    perturbations = torch.flatten(percentiles, start_dim=1)
    student_inputs = torch.cat(
        [
            torch.flatten(features, start_dim=1),
            perturbations,
        ],
        dim=1,
    )
    return PerturbationBatch(
        student_inputs=student_inputs,
        original_features=features,
        perturbations=perturbations,
        perturbed_features=features,
        percentiles=percentiles,
    )


def build_unperturbed_mc_dropout_batch(
    features: torch.Tensor,
    config: PerturbationConfig,
) -> PerturbationBatch:
    """Build MC-dropout-shaped inputs without applying dropout."""

    if features.ndim != 4:
        raise ValueError(
            "unperturbed Monte-Carlo dropout inputs expect convolutional features "
            "with shape (batch, channels, height, width)"
        )
    mask = torch.ones_like(features)
    return PerturbationBatch(
        student_inputs=torch.flatten(features, start_dim=1),
        original_features=features,
        perturbations=torch.flatten(mask, start_dim=1),
        perturbed_features=features,
        percentiles=mask,
    )


def build_pca_projection_batch(
    features: torch.Tensor,
    projector: PcaProjector,
) -> PerturbationBatch:
    """Project features through a fitted PCA basis for student input."""

    if features.ndim != 4:
        raise ValueError(
            "PCA projection expects convolutional features with shape "
            "(batch, channels, height, width)"
        )
    projected = projector.transform(features)
    empty = features.new_empty((features.shape[0], 0))
    return PerturbationBatch(
        student_inputs=projected,
        original_features=features,
        perturbations=empty,
        perturbed_features=features,
        percentiles=empty,
    )


def fit_pca_projector_from_activations(
    activation_path: Path,
    n_components: int,
    expected_dataset: str | None = None,
) -> PcaProjector:
    """Fit PCA from a training-split teacher activation artifact.

    Args:
        activation_path: Exported teacher activation artifact.
        n_components: Number of principal components to retain.
        expected_dataset: Expected artifact dataset key, such as
            ``cifar10_train``. When provided, both the dataset key and training
            split metadata are validated before fitting.
    """

    artifact = torch.load(activation_path, map_location="cpu", weights_only=False)
    if expected_dataset is not None:
        if artifact.get("dataset") != expected_dataset or artifact.get("split") != "train":
            raise ValueError(
                "PCA must be fitted from the complete ID training-split activation "
                f"artifact {expected_dataset!r}; got dataset={artifact.get('dataset')!r}, "
                f"split={artifact.get('split')!r}: {activation_path}"
            )
    activations = artifact.get("activations")
    if not torch.is_tensor(activations):
        raise ValueError(f"Activation artifact is missing tensor 'activations': {activation_path}")
    values = torch.flatten(activations.float(), start_dim=1)
    if n_components > values.shape[1]:
        raise ValueError(
            f"pca_components={n_components} exceeds flattened activation dimension "
            f"{values.shape[1]}"
        )
    mean = values.mean(dim=0)
    centered = values - mean
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    return PcaProjector(
        mean=mean,
        components=vh[:n_components].contiguous(),
    )


def save_pca_projector(path: Path, projector: PcaProjector, metadata: dict[str, object]) -> None:
    """Save a fitted PCA projector artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **metadata,
            "mean": projector.mean.cpu(),
            "components": projector.components.cpu(),
        },
        path,
    )


def load_pca_projector(path: Path, device: torch.device | None = None) -> PcaProjector:
    """Load a fitted PCA projector artifact."""

    artifact = torch.load(path, map_location="cpu", weights_only=False)
    mean = artifact.get("mean")
    components = artifact.get("components")
    if not torch.is_tensor(mean) or not torch.is_tensor(components):
        raise ValueError(f"PCA projector artifact is missing tensors: {path}")
    projector = PcaProjector(mean=mean.float(), components=components.float())
    if device is not None:
        projector = projector.to(device)
    return projector


def pca_projector_path(experiment_dir: Path) -> Path:
    """Return the standard PCA projector artifact path for an experiment."""

    return experiment_dir / "pca_projector.pt"


def _sample_dropout_mask(
    features: torch.Tensor,
    config: PerturbationConfig,
) -> torch.Tensor:
    batch_size, channels, height, width = features.shape
    if config.dropout_mode == "element":
        mask_shape = features.shape
    elif config.dropout_mode == "channel":
        mask_shape = (batch_size, channels, 1, 1)
    elif config.dropout_mode == "spatial":
        mask_shape = (batch_size, 1, height, width)
    else:
        raise ValueError(f"Unsupported dropout mode: {config.dropout_mode}")
    keep_probability = 1.0 - config.dropout_probability
    return torch.empty(
        mask_shape,
        device=features.device,
        dtype=features.dtype,
    ).bernoulli_(keep_probability)


def _sample_percentiles(
    features: torch.Tensor,
    config: PerturbationConfig,
) -> torch.Tensor:
    return torch.empty(
        _percentile_shape(features, config),
        device=features.device,
        dtype=features.dtype,
    ).uniform_(config.u_min, config.u_max)


def _percentile_shape(
    features: torch.Tensor,
    config: PerturbationConfig,
) -> tuple[int, ...]:
    batch_size, channels, height, width = features.shape
    if config.clipping_mode == "constant":
        return (batch_size, 1)
    elif config.clipping_mode == "spatial_dependent":
        return (batch_size, height, width)
    elif config.clipping_mode == "channel_dependent":
        return (batch_size, channels)
    else:
        raise ValueError(f"Unsupported clipping mode: {config.clipping_mode}")


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
