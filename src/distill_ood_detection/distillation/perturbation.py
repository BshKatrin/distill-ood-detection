"""Perturbation helpers for stochastic distillation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from distill_ood_detection.config import ClippingLayerConfig, PerturbationConfig


RESNET_FEATURE_LAYERS = ("layer1", "layer2", "layer3", "layer4")


@dataclass(frozen=True)
class PerturbationBatch:
    """Perturbed teacher features and perturbation-aware student inputs."""

    student_inputs: torch.Tensor
    original_features: torch.Tensor
    perturbations: torch.Tensor
    perturbed_features: torch.Tensor
    percentiles: torch.Tensor


@dataclass(frozen=True)
class PixelAugmentationBatch:
    """Pixel-perturbed images and transform parameters for student conditioning."""

    clean_images: torch.Tensor
    perturbed_images: torch.Tensor
    transform_params: torch.Tensor
    normalized_transform_params: torch.Tensor


@dataclass(frozen=True)
class SequentialClippingBatch:
    """Final teacher state and student inputs after sequential layer clipping."""

    student_inputs: torch.Tensor
    final_features: torch.Tensor
    perturbations: dict[str, torch.Tensor]
    teacher_logits: torch.Tensor


@dataclass(frozen=True)
class PcaProjector:
    """Fixed PCA projection from flattened teacher features."""

    mean: torch.Tensor
    components: torch.Tensor
    component_stds: torch.Tensor | None = None

    def to(self, device: torch.device) -> PcaProjector:
        """Move PCA tensors to a device."""

        return PcaProjector(
            mean=self.mean.to(device),
            components=self.components.to(device),
            component_stds=(
                self.component_stds.to(device)
                if self.component_stds is not None
                else None
            ),
        )

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        """Project convolutional features onto PCA components."""

        flat_features = torch.flatten(features, start_dim=1)
        mean, components = self._aligned_tensors(features)
        return (flat_features - mean) @ components.T

    def whitened_transform(self, features: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
        """Project features and scale PCA coordinates to ID unit variance."""

        projected = self.transform(features)
        if self.component_stds is None:
            raise ValueError("PCA projector is missing component_stds for whitening")
        stds = self.component_stds.to(device=projected.device, dtype=projected.dtype)
        return projected / stds.clamp_min(eps)

    def projection_matrix(self, reference: torch.Tensor) -> torch.Tensor:
        """Return PCA eigenvectors as ``Q`` with shape ``(original_dim, pca_dim)``."""

        _, components = self._aligned_tensors(reference)
        return components.T

    def inverse_transform(
        self,
        projected_features: torch.Tensor,
        feature_shape: tuple[int, ...],
    ) -> torch.Tensor:
        """Reconstruct projected features in the original feature shape."""

        mean, components = self._aligned_tensors(projected_features)
        reconstructed = projected_features @ components + mean
        return reconstructed.reshape(projected_features.shape[0], *feature_shape)

    def _aligned_tensors(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.mean.to(device=reference.device, dtype=reference.dtype),
            self.components.to(device=reference.device, dtype=reference.dtype),
        )


def clipping_start_layer(config: PerturbationConfig) -> str:
    """Return the earliest configured clipping layer."""

    if config.method != "clipping" or not config.clipping_layers:
        raise ValueError("A non-empty clipping_layers mapping is required")
    return next(
        layer_name
        for layer_name in RESNET_FEATURE_LAYERS
        if layer_name in config.clipping_layers
    )


def forward_to_clipping_start(
    teacher: torch.nn.Module,
    images: torch.Tensor,
    config: PerturbationConfig,
) -> torch.Tensor:
    """Run a ResNet teacher to the input of its earliest clipped layer."""

    start_layer = clipping_start_layer(config)
    _validate_resnet_teacher(teacher)
    x = teacher.conv1(images)
    x = teacher.bn1(x)
    x = teacher.relu(x)
    x = teacher.maxpool(x)
    for layer_name in RESNET_FEATURE_LAYERS:
        if layer_name == start_layer:
            return x
        x = getattr(teacher, layer_name)(x)
    raise RuntimeError(f"Clipping start layer was not reached: {start_layer}")


def forward_clean_from_clipping_start(
    teacher: torch.nn.Module,
    start_features: torch.Tensor,
    config: PerturbationConfig,
) -> torch.Tensor:
    """Continue a clean ResNet forward pass from the clipping start."""

    _validate_resnet_teacher(teacher)
    start_index = RESNET_FEATURE_LAYERS.index(clipping_start_layer(config))
    x = start_features
    for layer_name in RESNET_FEATURE_LAYERS[start_index:]:
        x = getattr(teacher, layer_name)(x)
    return _resnet_logits_from_layer4(teacher, x)


def build_sequential_clipping_batch(
    teacher: torch.nn.Module,
    start_features: torch.Tensor,
    config: PerturbationConfig,
    apply_perturbation: bool = True,
) -> SequentialClippingBatch:
    """Apply configured clipping layers in ResNet forward order."""

    if config.method != "clipping":
        raise ValueError("Sequential clipping requires perturbation method 'clipping'")
    _validate_resnet_teacher(teacher)
    start_index = RESNET_FEATURE_LAYERS.index(clipping_start_layer(config))
    x = start_features
    perturbations: dict[str, torch.Tensor] = {}
    for layer_name in RESNET_FEATURE_LAYERS[start_index:]:
        x = getattr(teacher, layer_name)(x)
        layer_config = config.clipping_layers.get(layer_name)
        if layer_config is None:
            continue
        if apply_perturbation:
            percentiles = _sample_percentiles(x, layer_config)
            x = _clip_feature_batch(x, percentiles, layer_config)
        else:
            percentiles = torch.ones(
                _percentile_shape(x, layer_config),
                device=x.device,
                dtype=x.dtype,
            )
        perturbations[layer_name] = torch.flatten(percentiles, start_dim=1)

    final_features = x
    student_parts = [
        pool_embedding_features(final_features, config.embedding_pool),
        *(
            perturbations[layer_name]
            for layer_name in RESNET_FEATURE_LAYERS
            if layer_name in perturbations
        ),
    ]
    return SequentialClippingBatch(
        student_inputs=torch.cat(student_parts, dim=1),
        final_features=final_features,
        perturbations=perturbations,
        teacher_logits=_resnet_logits_from_layer4(teacher, final_features),
    )


def clipping_teacher_target_logits(
    batch: SequentialClippingBatch,
    clean_logits: torch.Tensor,
    config: PerturbationConfig,
) -> torch.Tensor:
    """Select clean or sequentially clipped teacher logits."""

    if config.teacher_target == "clean":
        return clean_logits
    if config.teacher_target == "perturbed":
        return batch.teacher_logits
    raise ValueError(f"Unsupported teacher target: {config.teacher_target}")


def sample_clipping_perturbation(
    features: torch.Tensor,
    config: ClippingLayerConfig,
    embedding_pool: str = "flatten",
) -> PerturbationBatch:
    """Clip one feature map and concatenate its pooled representation with ``u``.

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
            pool_embedding_features(perturbed, embedding_pool),
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

    if config.method == "mc_dropout":
        return sample_mc_dropout_perturbation(features, config)
    if config.method == "pca_projection":
        if pca_projector is None:
            raise ValueError("pca_projector is required for PCA projection")
        return build_pca_projection_batch(features, pca_projector)
    if config.method == "pca_masked_projection":
        if pca_projector is None:
            raise ValueError("pca_projector is required for masked PCA projection")
        return sample_pca_masked_projection_batch(features, config, pca_projector)
    if config.method == "clipping":
        raise ValueError(
            "clipping requires sequential teacher forwarding; "
            "use build_sequential_clipping_batch"
        )
    raise ValueError(f"Unsupported perturbation method: {config.method}")


def sample_pixel_augmentation(
    images: torch.Tensor,
    config: PerturbationConfig,
    normalization: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> PixelAugmentationBatch:
    """Sample and apply a mild pixel-space augmentation to normalized images."""

    if images.ndim != 4:
        raise ValueError(
            "pixel augmentation expects images with shape "
            "(batch, channels, height, width)"
        )
    if images.shape[1] != 3:
        raise ValueError("pixel augmentation expects RGB images with three channels")
    raw_params, normalized_params = sample_pixel_augmentation_params(images, config)
    perturbed_images = apply_pixel_augmentation(images, raw_params, normalization)
    return PixelAugmentationBatch(
        clean_images=images,
        perturbed_images=perturbed_images,
        transform_params=raw_params,
        normalized_transform_params=normalized_params,
    )


def sample_pixel_augmentation_params(
    images: torch.Tensor,
    config: PerturbationConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample raw and normalized pixel augmentation parameters for a batch."""

    if images.ndim != 4:
        raise ValueError(
            "pixel augmentation parameter sampling expects images with shape "
            "(batch, channels, height, width)"
        )
    batch_size, _, height, width = images.shape
    device = images.device
    dtype = images.dtype
    max_dx = int(round(width * config.translate_fraction))
    max_dy = int(round(height * config.translate_fraction))
    angle = images.new_empty(batch_size).uniform_(
        -config.rotation_degrees,
        config.rotation_degrees,
    )
    dx = _sample_integer_offsets(batch_size, max_dx, device, dtype)
    dy = _sample_integer_offsets(batch_size, max_dy, device, dtype)
    scale = images.new_empty(batch_size).uniform_(config.scale_min, config.scale_max)
    brightness = images.new_empty(batch_size).uniform_(
        1.0 - config.brightness_delta,
        1.0 + config.brightness_delta,
    )
    contrast = images.new_empty(batch_size).uniform_(
        1.0 - config.contrast_delta,
        1.0 + config.contrast_delta,
    )
    raw_params = torch.stack([angle, dx, dy, scale, brightness, contrast], dim=1)
    normalized_params = torch.stack(
        [
            _normalize_delta(angle, config.rotation_degrees),
            _normalize_delta(dx, float(max_dx)),
            _normalize_delta(dy, float(max_dy)),
            _normalize_delta(scale - 1.0, _scale_delta(config)),
            _normalize_delta(brightness - 1.0, config.brightness_delta),
            _normalize_delta(contrast - 1.0, config.contrast_delta),
        ],
        dim=1,
    )
    return raw_params, normalized_params


def apply_pixel_augmentation(
    images: torch.Tensor,
    raw_params: torch.Tensor,
    normalization: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> torch.Tensor:
    """Apply raw pixel augmentation parameters to normalized RGB image tensors."""

    if raw_params.shape != (images.shape[0], 6):
        raise ValueError("pixel augmentation parameters must have shape (batch, 6)")
    unnormalized = _unnormalize_images(images, normalization).clamp(0.0, 1.0)
    augmented = []
    for image, params in zip(unnormalized, raw_params, strict=True):
        angle, dx, dy, scale, brightness, contrast = params.tolist()
        transformed = TF.affine(
            image,
            angle=angle,
            translate=[int(round(dx)), int(round(dy))],
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=[0.0, 0.0, 0.0],
        )
        transformed = TF.adjust_brightness(transformed, brightness)
        transformed = TF.adjust_contrast(transformed, contrast)
        augmented.append(transformed.clamp(0.0, 1.0))
    return _normalize_images(torch.stack(augmented, dim=0), normalization)


def build_pixel_student_inputs(
    features: torch.Tensor,
    normalized_transform_params: torch.Tensor,
    embedding_pool: str = "flatten",
) -> torch.Tensor:
    """Pool teacher features and append pixel-corruption conditioning values."""

    if features.shape[0] != normalized_transform_params.shape[0]:
        raise ValueError("features and transform parameters must have the same batch size")
    pooled_features = pool_pixel_features(features, embedding_pool)
    return torch.cat(
        [
            pooled_features,
            normalized_transform_params.to(device=features.device, dtype=features.dtype),
        ],
        dim=1,
    )


def pool_pixel_features(features: torch.Tensor, embedding_pool: str) -> torch.Tensor:
    """Return flattened or global-average-pooled teacher features."""

    return pool_embedding_features(features, embedding_pool)


def pool_embedding_features(features: torch.Tensor, embedding_pool: str) -> torch.Tensor:
    """Return flattened or global-average-pooled convolutional features."""

    if embedding_pool == "flatten":
        return torch.flatten(features, start_dim=1)
    if embedding_pool == "avg":
        if features.ndim != 4:
            raise ValueError(
                "global average pooling expects convolutional features with shape "
                "(batch, channels, height, width)"
            )
        return features.mean(dim=(-2, -1))
    raise ValueError(f"Unsupported embedding pool: {embedding_pool}")


def build_unperturbed_pixel_params(
    images: torch.Tensor,
    parameter_count: int = 6,
) -> torch.Tensor:
    """Return neutral conditioning values for clean pixel inputs."""

    if images.ndim != 4:
        raise ValueError(
            "unperturbed pixel parameters expect images with shape "
            "(batch, channels, height, width)"
        )
    return images.new_zeros((images.shape[0], parameter_count))


def sample_mc_dropout_perturbation(
    features: torch.Tensor,
    config: PerturbationConfig,
) -> PerturbationBatch:
    """Apply Monte-Carlo dropout to student features."""

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
        perturbed_features=dropped,
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
    if config.method == "pca_masked_projection":
        if pca_projector is None:
            raise ValueError("pca_projector is required for masked PCA projection")
        return build_unperturbed_pca_masked_projection_batch(features, pca_projector)
    if config.method == "clipping":
        raise ValueError(
            "clipping requires sequential teacher forwarding; "
            "use build_sequential_clipping_batch"
        )
    raise ValueError(f"Unsupported perturbation method: {config.method}")


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
    reconstructed = projector.inverse_transform(projected, tuple(features.shape[1:]))
    empty = features.new_empty((features.shape[0], 0))
    return PerturbationBatch(
        student_inputs=projected,
        original_features=features,
        perturbations=empty,
        perturbed_features=reconstructed,
        percentiles=empty,
    )


def sample_pca_masked_projection_batch(
    features: torch.Tensor,
    config: PerturbationConfig,
    projector: PcaProjector,
) -> PerturbationBatch:
    """Project features after masking PCA components."""

    keep_mask = _sample_pca_keep_mask(features, config, projector)
    return _build_pca_masked_projection_batch(features, projector, keep_mask)


def build_unperturbed_pca_masked_projection_batch(
    features: torch.Tensor,
    projector: PcaProjector,
) -> PerturbationBatch:
    """Build masked-PCA-shaped inputs with every PCA component kept."""

    keep_mask = features.new_ones((features.shape[0], projector.components.shape[0]))
    return _build_pca_masked_projection_batch(features, projector, keep_mask)


def teacher_target_features(
    batch: PerturbationBatch,
    config: PerturbationConfig,
) -> torch.Tensor:
    """Select clean or perturbed features for the teacher continuation."""

    if config.teacher_target == "clean":
        return batch.original_features
    if config.teacher_target == "perturbed":
        return batch.perturbed_features
    raise ValueError(f"Unsupported teacher target: {config.teacher_target}")


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
    values = torch.flatten(activations, start_dim=1).to(dtype=torch.float32)
    if n_components > values.shape[1]:
        raise ValueError(
            f"pca_components={n_components} exceeds flattened activation dimension "
            f"{values.shape[1]}"
        )
    mean = values.mean(dim=0)
    components = _fit_truncated_pca_components(values, n_components)
    projected = (values - mean) @ components.T
    component_stds = projected.std(dim=0, unbiased=False)
    return PcaProjector(
        mean=mean,
        components=components,
        component_stds=component_stds,
    )


def save_pca_projector(path: Path, projector: PcaProjector, metadata: dict[str, object]) -> None:
    """Save a fitted PCA projector artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **metadata,
            "mean": projector.mean.cpu(),
            "components": projector.components.cpu(),
            "component_stds": (
                projector.component_stds.cpu()
                if projector.component_stds is not None
                else None
            ),
        },
        path,
    )


def load_pca_projector(path: Path, device: torch.device | None = None) -> PcaProjector:
    """Load a fitted PCA projector artifact."""

    artifact = torch.load(path, map_location="cpu", weights_only=False)
    mean = artifact.get("mean")
    components = artifact.get("components")
    component_stds = artifact.get("component_stds")
    if not torch.is_tensor(mean) or not torch.is_tensor(components):
        raise ValueError(f"PCA projector artifact is missing tensors: {path}")
    projector = PcaProjector(
        mean=mean.float(),
        components=components.float(),
        component_stds=(
            component_stds.float()
            if torch.is_tensor(component_stds)
            else None
        ),
    )
    if device is not None:
        projector = projector.to(device)
    return projector


def pca_projector_path(experiment_dir: Path) -> Path:
    """Return the standard PCA projector artifact path for an experiment."""

    return experiment_dir / "pca_projector.pt"


def _fit_truncated_pca_components(
    values: torch.Tensor,
    n_components: int,
    oversampling: int = 20,
    niter: int = 4,
) -> torch.Tensor:
    """Fit leading PCA components without computing a full SVD."""

    q = min(n_components + oversampling, min(values.shape))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        _, _, v = torch.pca_lowrank(values, q=q, center=True, niter=niter)
    return v[:, :n_components].T.contiguous()


def _build_pca_masked_projection_batch(
    features: torch.Tensor,
    projector: PcaProjector,
    keep_mask: torch.Tensor,
) -> PerturbationBatch:
    if features.ndim != 4:
        raise ValueError(
            "masked PCA projection expects convolutional features with shape "
            "(batch, channels, height, width)"
        )
    flat_features = torch.flatten(features, start_dim=1)
    mean, _ = projector._aligned_tensors(features)
    centered = flat_features - mean
    projection_matrix = projector.projection_matrix(features)
    expected_mask_shape = (features.shape[0], projection_matrix.shape[1])
    if keep_mask.shape != expected_mask_shape:
        raise ValueError(
            "PCA keep mask must have shape "
            f"{expected_mask_shape}; got {tuple(keep_mask.shape)}"
        )
    keep_mask = keep_mask.to(device=features.device, dtype=features.dtype)
    masked_projection_matrices = projection_matrix.unsqueeze(0) * keep_mask.unsqueeze(1)
    masked_projected = torch.bmm(
        centered.unsqueeze(1),
        masked_projection_matrices,
    ).squeeze(1)
    reconstructed = projector.inverse_transform(
        masked_projected,
        tuple(features.shape[1:]),
    )
    student_inputs = torch.cat([masked_projected, keep_mask], dim=1)
    return PerturbationBatch(
        student_inputs=student_inputs,
        original_features=features,
        perturbations=keep_mask,
        perturbed_features=reconstructed,
        percentiles=keep_mask,
    )


def _sample_pca_keep_mask(
    features: torch.Tensor,
    config: PerturbationConfig,
    projector: PcaProjector,
) -> torch.Tensor:
    keep_probability = 1.0 - config.pca_mask_probability
    return torch.empty(
        (features.shape[0], projector.components.shape[0]),
        device=features.device,
        dtype=features.dtype,
    ).bernoulli_(keep_probability)


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
    config: ClippingLayerConfig,
) -> torch.Tensor:
    return torch.empty(
        _percentile_shape(features, config),
        device=features.device,
        dtype=features.dtype,
    ).uniform_(config.u_min, config.u_max)


def _percentile_shape(
    features: torch.Tensor,
    config: ClippingLayerConfig,
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
    config: ClippingLayerConfig,
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
    config: ClippingLayerConfig,
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


def _validate_resnet_teacher(teacher: torch.nn.Module) -> None:
    for name in (
        "conv1",
        "bn1",
        "relu",
        "maxpool",
        "avgpool",
        "fc",
        *RESNET_FEATURE_LAYERS,
    ):
        if not hasattr(teacher, name):
            raise ValueError(f"Clipping teacher is missing ResNet submodule: {name}")


def _resnet_logits_from_layer4(
    teacher: torch.nn.Module,
    features: torch.Tensor,
) -> torch.Tensor:
    pooled = teacher.avgpool(features)
    return teacher.fc(torch.flatten(pooled, start_dim=1))


def _sample_integer_offsets(
    batch_size: int,
    max_offset: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if max_offset == 0:
        return torch.zeros(batch_size, device=device, dtype=dtype)
    return torch.randint(
        low=-max_offset,
        high=max_offset + 1,
        size=(batch_size,),
        device=device,
    ).to(dtype=dtype)


def _normalize_delta(values: torch.Tensor, denominator: float) -> torch.Tensor:
    if denominator == 0.0:
        return torch.zeros_like(values)
    return values / denominator


def _scale_delta(config: PerturbationConfig) -> float:
    return max(abs(1.0 - config.scale_min), abs(config.scale_max - 1.0))


def _normalization_tensors(
    images: torch.Tensor,
    normalization: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    mean, std = normalization
    mean_tensor = torch.tensor(mean, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return mean_tensor, std_tensor


def _unnormalize_images(
    images: torch.Tensor,
    normalization: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> torch.Tensor:
    mean, std = _normalization_tensors(images, normalization)
    return images * std + mean


def _normalize_images(
    images: torch.Tensor,
    normalization: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> torch.Tensor:
    mean, std = _normalization_tensors(images, normalization)
    return (images - mean) / std
