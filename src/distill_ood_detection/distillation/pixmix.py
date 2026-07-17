"""PixMix pixel corruption following the reference CIFAR implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch.nn import functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from distill_ood_detection.config import PixMixConfig


@dataclass(frozen=True)
class PixMixBatch:
    """Paper-style base views and their PixMix-corrupted counterparts."""

    clean_images: torch.Tensor
    perturbed_images: torch.Tensor


def sample_pixmix(
    images: torch.Tensor,
    mixing_images: torch.Tensor,
    config: PixMixConfig,
    normalization: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> PixMixBatch:
    """Apply PixMix to normalized RGB images."""

    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("PixMix expects RGB images with shape (batch, 3, height, width)")
    if mixing_images.shape != (
        images.shape[0],
        3,
        config.working_size,
        config.working_size,
    ):
        raise ValueError(
            "PixMix mixing images must have shape "
            f"(batch, 3, {config.working_size}, {config.working_size})"
        )

    target_size = tuple(images.shape[-2:])
    working_images = F.interpolate(
        _unnormalize(images, normalization).clamp(0.0, 1.0),
        size=(config.working_size, config.working_size),
        mode="bilinear",
        align_corners=False,
    )
    clean_views: list[torch.Tensor] = []
    perturbed_views: list[torch.Tensor] = []
    for image, mixing_image in zip(working_images, mixing_images, strict=True):
        clean_view = _sample_cifar_base_view(image)
        mixed = _sample_single_pixmix(clean_view, mixing_image, config)
        clean_views.append(clean_view)
        perturbed_views.append(mixed)

    clean = torch.stack(clean_views)
    perturbed = torch.stack(perturbed_views)
    if target_size != (config.working_size, config.working_size):
        clean = F.interpolate(
            clean,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        perturbed = F.interpolate(
            perturbed,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
    return PixMixBatch(
        clean_images=_normalize(clean.clamp(0.0, 1.0), normalization),
        perturbed_images=_normalize(perturbed.clamp(0.0, 1.0), normalization),
    )


def _sample_cifar_base_view(image: torch.Tensor) -> torch.Tensor:
    if torch.rand(()) < 0.5:
        image = TF.hflip(image)
    padding = image.shape[-1] // 8
    padded = TF.pad(image, [padding, padding, padding, padding], fill=0.0)
    max_top = padded.shape[-2] - image.shape[-2]
    max_left = padded.shape[-1] - image.shape[-1]
    top = int(torch.randint(max_top + 1, ()).item())
    left = int(torch.randint(max_left + 1, ()).item())
    return TF.crop(
        img=padded,
        top=top,
        left=left,
        height=image.shape[-2],
        width=image.shape[-1],
    )


def _sample_single_pixmix(
    clean_image: torch.Tensor,
    mixing_image: torch.Tensor,
    config: PixMixConfig,
) -> torch.Tensor:
    mixed = (
        _sample_augmentation(clean_image, config)
        if torch.rand(()) < 0.5
        else clean_image.clone()
    )
    round_count = int(
        torch.randint(config.mixing_iterations + 1, ()).item()
    )
    for _ in range(round_count):
        other = (
            _sample_augmentation(clean_image, config)
            if torch.rand(()) < 0.5
            else mixing_image
        )
        if torch.rand(()) < 0.5:
            mixed = _additive_mix(mixed, other, config.beta)
        else:
            mixed = _multiplicative_mix(mixed, other, config.beta)
        mixed = mixed.clamp(0.0, 1.0)
    return mixed


def _sample_augmentation(image: torch.Tensor, config: PixMixConfig) -> torch.Tensor:
    operation_count = 13 if config.all_ops else 9
    operation = int(torch.randint(operation_count, ()).item())
    severity = float(
        torch.empty(()).uniform_(
            0.1,
            config.augmentation_severity,
        ).item()
    )
    if operation == 0:
        return TF.autocontrast(image)
    if operation == 1:
        return _uint8_operation(image, TF.equalize)
    if operation == 2:
        bits = 4 - int(severity * 4 / 10)
        return _uint8_operation(image, lambda value: TF.posterize(value, bits))
    if operation == 3:
        degrees = int(severity * 30 / 10)
        if torch.rand(()) < 0.5:
            degrees = -degrees
        return TF.rotate(
            image,
            degrees,
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )
    if operation == 4:
        threshold = 1.0 - int(severity * 256 / 10) / 255.0
        return TF.solarize(image, threshold)
    if operation in {5, 6}:
        shear_factor = severity * 0.3 / 10.0
        if torch.rand(()) < 0.5:
            shear_factor = -shear_factor
        shear_degrees = math.degrees(math.atan(shear_factor))
        shear = [shear_degrees, 0.0] if operation == 5 else [0.0, shear_degrees]
        return TF.affine(
            image,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=shear,
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )
    if operation in {7, 8}:
        offset = int(severity * image.shape[-1] / 30)
        if torch.rand(()) < 0.5:
            offset = -offset
        translate = [offset, 0] if operation == 7 else [0, offset]
        return TF.affine(
            image,
            angle=0.0,
            translate=translate,
            scale=1.0,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )

    factor = severity * 1.8 / 10.0 + 0.1
    if operation == 9:
        return TF.adjust_saturation(image, factor)
    if operation == 10:
        return TF.adjust_contrast(image, factor)
    if operation == 11:
        return TF.adjust_brightness(image, factor)
    if operation == 12:
        return TF.adjust_sharpness(image, factor)
    raise RuntimeError(f"Unsupported PixMix operation index: {operation}")


def _uint8_operation(
    image: torch.Tensor,
    operation: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    quantized = (image.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    return operation(quantized).to(dtype=image.dtype) / 255.0


def _mixing_coefficients(beta: float) -> tuple[float, float]:
    beta_value = torch.tensor(beta)
    one = torch.tensor(1.0)
    if torch.rand(()) < 0.5:
        a = float(torch.distributions.Beta(beta_value, one).sample().item())
        b = float(torch.distributions.Beta(one, beta_value).sample().item())
    else:
        a = 1.0 + float(torch.distributions.Beta(one, beta_value).sample().item())
        b = -float(torch.distributions.Beta(one, beta_value).sample().item())
    return a, b


def _additive_mix(image: torch.Tensor, other: torch.Tensor, beta: float) -> torch.Tensor:
    a, b = _mixing_coefficients(beta)
    centered_image = image * 2.0 - 1.0
    centered_other = other * 2.0 - 1.0
    return (a * centered_image + b * centered_other + 1.0) / 2.0


def _multiplicative_mix(
    image: torch.Tensor,
    other: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    a, b = _mixing_coefficients(beta)
    scaled_image = image * 2.0
    scaled_other = (other * 2.0).clamp_min(1.0e-37)
    return scaled_image.pow(a) * scaled_other.pow(b) / 2.0


def _normalization_tensors(
    images: torch.Tensor,
    normalization: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    mean, std = normalization
    shape = (1, len(mean), 1, 1)
    return (
        images.new_tensor(mean).reshape(shape),
        images.new_tensor(std).reshape(shape),
    )


def _unnormalize(
    images: torch.Tensor,
    normalization: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> torch.Tensor:
    mean, std = _normalization_tensors(images, normalization)
    return images * std + mean


def _normalize(
    images: torch.Tensor,
    normalization: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> torch.Tensor:
    mean, std = _normalization_tensors(images, normalization)
    return (images - mean) / std
