"""Feature Denoising PCA reconstruction training and scoring."""

from __future__ import annotations

import time
from dataclasses import dataclass
from dataclasses import asdict
from pathlib import Path

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from distill_ood_detection.config import (
    ExperimentConfig,
    FeatureDenoisingConfig,
    OptimizerConfig,
    PerturbationConfig,
    ResolvedTrainingMethodConfig,
)
from distill_ood_detection.datasets.pixmix import PixMixMixingProvider
from distill_ood_detection.distillation.perturbation import PcaProjector
from distill_ood_detection.distillation.perturbation import sample_pixel_augmentation
from distill_ood_detection.distillation.pixmix import sample_pixmix
from distill_ood_detection.distillation.train import build_optimizer
from distill_ood_detection.evaluation.activation_subspaces import (
    feature_denoising_subspace_errors,
)
from distill_ood_detection.utils import write_json

FEATURE_DENOISING_RECONSTRUCTION_METHOD = "pca_masked_reconstruction"
FEATURE_DENOISING_RECONSTRUCTION_SCORE = "feature_denoising_pca_reconstruction_error"
FEATURE_DENOISING_SPATIAL_RECONSTRUCTION_SCORE = "feature_denoising_spatial_reconstruction_error"
FEATURE_DENOISING_SPATIAL_BLOCK_RESIDUAL_RECONSTRUCTION_SCORE = (
    "feature_denoising_spatial_block_residual_reconstruction_error"
)
FEATURE_DENOISING_CHANNEL_RECONSTRUCTION_SCORE = "feature_denoising_channel_reconstruction_error"
FEATURE_DENOISING_CHANNEL_RESIDUAL_RECONSTRUCTION_SCORE = (
    "feature_denoising_channel_residual_reconstruction_error"
)
FEATURE_DENOISING_CONFUSION_CHANNEL_REPLACEMENT_RECONSTRUCTION_SCORE = (
    "feature_denoising_confusion_channel_replacement_reconstruction_error"
)
FEATURE_DENOISING_CONFUSION_CHANNEL_REPLACEMENT_RESIDUAL_RECONSTRUCTION_SCORE = (
    "feature_denoising_confusion_channel_replacement_residual_reconstruction_error"
)
FEATURE_DENOISING_SPATIAL_TOKEN_PREDICTION_SCORE = "feature_denoising_spatial_token_prediction_error"
FEATURE_DENOISING_PIXEL_EMBEDDING_PREDICTION_SCORE = "feature_denoising_pixel_embedding_prediction_error"
FEATURE_DENOISING_PIXEL_AUGMENTED_EMBEDDING_PREDICTION_SCORE = (
    "feature_denoising_pixel_augmented_embedding_prediction_error"
)
FEATURE_DENOISING_PIXEL_MULTILAYER_PREDICTION_SCORE = "feature_denoising_pixel_multilayer_prediction_error"
FEATURE_DENOISING_PIXEL_MULTILAYER_L234_PREDICTION_SCORE = (
    "feature_denoising_pixel_multilayer_l234_prediction_error"
)


@dataclass(frozen=True)
class FeatureNormalizer:
    """Channelwise feature normalization fitted from ID training activations."""

    mean: torch.Tensor
    std: torch.Tensor

    def to(self, device: torch.device) -> FeatureNormalizer:
        """Move normalization tensors to a device."""

        return FeatureNormalizer(
            mean=self.mean.to(device),
            std=self.std.to(device),
        )

    def normalize(self, features: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
        """Normalize feature maps channelwise."""

        mean = self.mean.to(device=features.device, dtype=features.dtype)
        std = self.std.to(device=features.device, dtype=features.dtype)
        return (features - mean) / std.clamp_min(eps)


@dataclass(frozen=True)
class ClassChannelCorruptionBank:
    """Class statistics, confusion pairs, and activation-map prototypes."""

    confusion_matrix: torch.Tensor
    confusing_class: torch.Tensor
    class_mean: torch.Tensor
    class_std: torch.Tensor
    prototypes: torch.Tensor
    prototype_source_indices: torch.Tensor
    validation_class_counts: torch.Tensor
    mean_teacher_probabilities: torch.Tensor


def train_feature_denoising_student(
    student: nn.Module,
    train_loader: DataLoader[tuple[torch.Tensor, int]],
    validation_loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
    optimizer_config: OptimizerConfig,
    training_config: ResolvedTrainingMethodConfig,
    output_dir: Path,
    perturbation_forwarder: nn.Module,
    feature_denoising_config: FeatureDenoisingConfig,
    pca_projector: PcaProjector | None = None,
    feature_normalizer: FeatureNormalizer | None = None,
    class_channel_corruption: ClassChannelCorruptionBank | None = None,
    image_normalization: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
    pixmix_provider: PixMixMixingProvider | None = None,
    mlflow_enabled: bool = False,
) -> dict[str, float | int | str]:
    """Train a student to reconstruct clean teacher representations."""

    student.to(device)
    perturbation_forwarder.to(device)
    perturbation_forwarder.eval()
    optimizer = build_optimizer(student, optimizer_config)
    history: list[dict[str, float | int]] = []
    best_validation_reconstruction_loss = float("inf")
    checkpoint_path = output_dir / "latest_student.pt"
    best_checkpoint_path = output_dir / "best_student.pt"
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    checked_student_input_shape = False

    for epoch in range(1, training_config.epochs + 1):
        student.train()
        total_loss = 0.0
        total_examples = 0
        progress = tqdm(train_loader, desc=f"{feature_denoising_config.method} epoch {epoch}", leave=False)
        for step, (images, labels) in enumerate(progress, start=1):
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                batch = sample_feature_denoising_batch(
                    images,
                    perturbation_forwarder,
                    feature_denoising_config,
                    pca_projector,
                    feature_normalizer,
                    class_channel_corruption,
                    labels,
                    image_normalization,
                    pixmix_provider,
                )
            if not checked_student_input_shape:
                _validate_student_input_shape(student, batch.student_inputs)
                checked_student_input_shape = True
            predictions = feature_denoising_predictions(
                student,
                batch,
                feature_denoising_config,
            )
            loss = hidden_component_mse(
                predictions,
                batch.targets,
                batch.keep_mask,
            )
            loss.backward()
            optimizer.step()

            batch_size = images.shape[0]
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            if step % training_config.log_every_steps == 0:
                progress.set_postfix(loss=f"{total_loss / total_examples:.4f}")

        train_loss = total_loss / total_examples
        validation_loss = reconstruction_loss(
            student,
            validation_loader,
            device,
            perturbation_forwarder,
            feature_denoising_config,
            pca_projector,
            feature_normalizer,
            class_channel_corruption,
            image_normalization,
            pixmix_provider,
        )
        if validation_loss < best_validation_reconstruction_loss:
            best_validation_reconstruction_loss = validation_loss
            torch.save(student.state_dict(), best_checkpoint_path)
        epoch_record = {
            "epoch": epoch,
            "reconstruction_loss": train_loss,
            "validation_reconstruction_loss": validation_loss,
        }
        history.append(epoch_record)
        write_json(output_dir / "history.json", {"history": history})
        if mlflow_enabled:
            mlflow.log_metrics(epoch_record, step=epoch)

    torch.save(student.state_dict(), checkpoint_path)
    summary = {
        "method": feature_denoising_config.method,
        "epochs": training_config.epochs,
        "best_validation_reconstruction_loss": best_validation_reconstruction_loss,
        "final_validation_reconstruction_loss": history[-1]["validation_reconstruction_loss"],
        "latest_checkpoint_path": str(checkpoint_path),
        "best_checkpoint_path": str(best_checkpoint_path),
        "checkpoint_path": str(checkpoint_path),
        "seconds": round(time.time() - started_at, 3),
    }
    write_json(output_dir / "metrics.json", summary)
    return summary


def reconstruction_loss(
    student: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
    perturbation_forwarder: nn.Module,
    feature_denoising_config: FeatureDenoisingConfig,
    pca_projector: PcaProjector | None = None,
    feature_normalizer: FeatureNormalizer | None = None,
    class_channel_corruption: ClassChannelCorruptionBank | None = None,
    image_normalization: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
    pixmix_provider: PixMixMixingProvider | None = None,
) -> float:
    """Return average hidden-component reconstruction loss for a loader."""

    student.eval()
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            batch = sample_feature_denoising_batch(
                images,
                perturbation_forwarder,
                feature_denoising_config,
                pca_projector,
                feature_normalizer,
                class_channel_corruption,
                labels,
                image_normalization,
                pixmix_provider,
            )
            predictions = feature_denoising_predictions(
                student,
                batch,
                feature_denoising_config,
            )
            loss = hidden_component_mse(predictions, batch.targets, batch.keep_mask)
            batch_size = images.shape[0]
            total_loss += loss.item() * batch_size
            total_examples += batch_size
    return total_loss / total_examples


def collect_feature_denoising_reconstruction_scores(
    student: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
    perturbation_forwarder: nn.Module,
    feature_denoising_config: FeatureDenoisingConfig,
    pca_projector: PcaProjector | None = None,
    feature_normalizer: FeatureNormalizer | None = None,
    class_channel_corruption: ClassChannelCorruptionBank | None = None,
    image_normalization: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
    pixmix_provider: PixMixMixingProvider | None = None,
    activation_subspace_basis: torch.Tensor | None = None,
    decisive_subspace_dimension: int | None = None,
) -> dict[str, torch.Tensor]:
    """Collect sign-adjusted reconstruction OOD scores for one dataset.

    When an activation-subspace basis and split dimension are supplied, the
    function also averages exact per-draw reconstruction diagnostics over the
    decisive and insignificant classifier subspaces.
    """

    if (activation_subspace_basis is None) != (decisive_subspace_dimension is None):
        raise ValueError(
            "activation_subspace_basis and decisive_subspace_dimension must be "
            "provided together"
        )

    student.eval()
    labels = []
    raw_errors = []
    scores = []
    identity_errors = []
    improvements = []
    cosine_similarities = []
    targets = []
    contexts = []
    predicted_embeddings = []
    component_scores: dict[str, list[torch.Tensor]] = {}
    subspace_scores: dict[str, list[torch.Tensor]] = {}
    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device)
            batch_labels = batch_labels.to(device)
            inference_class_ids: torch.Tensor | None = None
            features = (
                None
                if feature_denoising_config.method
                in {
                    "pixel_masked_embedding_prediction",
                    "pixel_augmented_embedding_prediction",
                    "pixel_masked_multilayer_prediction",
                    "pixel_masked_multilayer_l234_prediction",
                    "confusion_channel_replacement_reconstruction",
                    "confusion_channel_replacement_residual_reconstruction",
                }
                else _teacher_features(perturbation_forwarder, images)
            )
            if (
                feature_denoising_config.method
                in {
                    "confusion_channel_replacement_reconstruction",
                    "confusion_channel_replacement_residual_reconstruction",
                }
            ):
                teacher_logits, features = perturbation_forwarder(images)
                inference_class_ids = teacher_logits.argmax(dim=1)
            draw_errors = []
            draw_identity_errors = []
            draw_improvements = []
            draw_cosines = []
            draw_contexts = []
            draw_predictions = []
            draw_targets = []
            draw_component_scores: dict[str, list[torch.Tensor]] = {}
            draw_subspace_scores: dict[str, list[torch.Tensor]] = {}
            for _ in range(feature_denoising_config.evaluation_draws):
                if feature_denoising_config.method == "pixel_masked_embedding_prediction":
                    batch = sample_pixel_masked_embedding_batch(
                        images,
                        perturbation_forwarder,
                        feature_denoising_config,
                    )
                elif feature_denoising_config.method == "pixel_augmented_embedding_prediction":
                    batch = sample_pixel_augmented_embedding_batch(
                        images,
                        perturbation_forwarder,
                        feature_denoising_config,
                        image_normalization,
                        pixmix_provider,
                    )
                elif feature_denoising_config.method == "pixel_masked_multilayer_prediction":
                    batch = sample_pixel_masked_multilayer_batch(
                        images,
                        perturbation_forwarder,
                        feature_denoising_config,
                    )
                elif feature_denoising_config.method == "pixel_masked_multilayer_l234_prediction":
                    batch = sample_pixel_masked_multilayer_l234_batch(
                        images,
                        perturbation_forwarder,
                        feature_denoising_config,
                    )
                else:
                    if features is None:
                        raise RuntimeError("Feature tensor was not prepared")
                    batch = sample_feature_denoising_pca_batch(
                        features,
                        feature_denoising_config,
                        pca_projector,
                        feature_normalizer,
                        class_channel_corruption,
                        inference_class_ids,
                    )
                batch_predictions = feature_denoising_predictions(
                    student,
                    batch,
                    feature_denoising_config,
                )
                if activation_subspace_basis is not None:
                    if not isinstance(batch.student_inputs, torch.Tensor):
                        raise ValueError(
                            "Activation-subspace diagnostics require tensor student inputs"
                        )
                    assert decisive_subspace_dimension is not None
                    diagnostics = feature_denoising_subspace_errors(
                        right_basis=activation_subspace_basis,
                        decisive_dimension=decisive_subspace_dimension,
                        context=batch.student_inputs,
                        prediction=batch_predictions,
                        target=batch.targets,
                    )
                    for name, values in diagnostics.items():
                        draw_subspace_scores.setdefault(name, []).append(values)
                prediction_error = per_sample_hidden_component_mse(
                    batch_predictions,
                    batch.targets,
                    batch.keep_mask,
                )
                draw_errors.append(prediction_error)
                if feature_denoising_config.method in {
                    "spatial_block_residual_reconstruction",
                    "channel_masked_residual_reconstruction",
                    "confusion_channel_replacement_reconstruction",
                    "confusion_channel_replacement_residual_reconstruction",
                }:
                    if batch.corrupted_features is None:
                        raise RuntimeError(
                            "Residual Feature Denoising batch is missing corrupted features"
                        )
                    identity_error = per_sample_hidden_component_mse(
                        batch.corrupted_features,
                        batch.targets,
                        batch.keep_mask,
                    )
                    draw_identity_errors.append(identity_error)
                    draw_improvements.append(identity_error - prediction_error)
                if (
                    feature_denoising_config.method
                    in {
                        "pixel_masked_embedding_prediction",
                        "pixel_augmented_embedding_prediction",
                        "pixel_masked_multilayer_prediction",
                        "pixel_masked_multilayer_l234_prediction",
                    }
                    and isinstance(batch.student_inputs, torch.Tensor)
                ):
                    if batch.student_inputs.shape == batch.targets.shape:
                        identity_error = per_sample_hidden_component_mse(
                            batch.student_inputs,
                            batch.targets,
                            batch.keep_mask,
                        )
                        draw_identity_errors.append(identity_error)
                        draw_improvements.append(identity_error - prediction_error)
                        draw_cosines.append(
                            per_sample_cosine_similarity(batch_predictions, batch.targets)
                        )
                    else:
                        components = multilayer_component_scores(
                            batch.student_inputs,
                            batch_predictions,
                            batch.targets,
                        )
                        for name, values in components.items():
                            draw_component_scores.setdefault(name, []).append(values)
                    draw_contexts.append(batch.student_inputs)
                    draw_predictions.append(batch_predictions)
                    draw_targets.append(batch.targets)
            batch_raw_errors = torch.stack(draw_errors, dim=1).mean(dim=1)
            labels.append(batch_labels.cpu())
            raw_errors.append(batch_raw_errors.cpu())
            scores.append((-batch_raw_errors).cpu())
            if draw_identity_errors:
                identity_errors.append(torch.stack(draw_identity_errors, dim=1).mean(dim=1).cpu())
                improvements.append(torch.stack(draw_improvements, dim=1).mean(dim=1).cpu())
            if draw_cosines:
                cosine_similarities.append(torch.stack(draw_cosines, dim=1).mean(dim=1).cpu())
                contexts.append(torch.stack(draw_contexts, dim=1).mean(dim=1).cpu())
                predicted_embeddings.append(torch.stack(draw_predictions, dim=1).mean(dim=1).cpu())
                if draw_targets:
                    targets.append(torch.stack(draw_targets, dim=1).mean(dim=1).cpu())
            if draw_component_scores:
                for name, values in draw_component_scores.items():
                    component_scores.setdefault(name, []).append(
                        torch.stack(values, dim=1).mean(dim=1).cpu()
                    )
                contexts.append(torch.stack(draw_contexts, dim=1).mean(dim=1).cpu())
                predicted_embeddings.append(torch.stack(draw_predictions, dim=1).mean(dim=1).cpu())
                if draw_targets:
                    targets.append(torch.stack(draw_targets, dim=1).mean(dim=1).cpu())
            batch_subspace_scores = {
                name: torch.stack(values, dim=1).mean(dim=1)
                for name, values in draw_subspace_scores.items()
            }
            for component in ("decisive", "insignificant"):
                reconstruction_key = f"{component}_reconstruction_error"
                identity_key = f"{component}_identity_error"
                if reconstruction_key not in batch_subspace_scores:
                    continue
                reconstruction_error = batch_subspace_scores[reconstruction_key]
                identity_error = batch_subspace_scores[identity_key]
                batch_subspace_scores[f"{component}_improvement"] = (
                    identity_error - reconstruction_error
                )
                batch_subspace_scores[f"{component}_relative_improvement"] = (
                    identity_error - reconstruction_error
                ) / identity_error.clamp_min(1.0e-12)
            for name, values in batch_subspace_scores.items():
                subspace_scores.setdefault(name, []).append(
                    values.cpu()
                )
    result = {
        "labels": torch.cat(labels, dim=0),
        "raw_reconstruction_error": torch.cat(raw_errors, dim=0),
        "scores": torch.cat(scores, dim=0),
    }
    if identity_errors:
        result.update(
            {
                "identity_error": torch.cat(identity_errors, dim=0),
                "improvement": torch.cat(improvements, dim=0),
            }
        )
    if cosine_similarities:
        result.update(
            {
                "cosine_similarity": torch.cat(cosine_similarities, dim=0),
                "z_context": torch.cat(contexts, dim=0),
                "z_pred": torch.cat(predicted_embeddings, dim=0),
                "z_target": torch.cat(targets, dim=0),
            }
        )
    if component_scores:
        result.update(
            {
                name: torch.cat(values, dim=0)
                for name, values in component_scores.items()
            }
        )
        result.update(
            {
                "z_context": torch.cat(contexts, dim=0),
                "z_pred": torch.cat(predicted_embeddings, dim=0),
                "z_target": torch.cat(targets, dim=0),
            }
        )
    result.update(
        {
            name: torch.cat(values, dim=0)
            for name, values in subspace_scores.items()
        }
    )
    return result


def sample_feature_denoising_batch(
    images: torch.Tensor,
    perturbation_forwarder: nn.Module,
    config: FeatureDenoisingConfig,
    projector: PcaProjector | None = None,
    feature_normalizer: FeatureNormalizer | None = None,
    class_channel_corruption: ClassChannelCorruptionBank | None = None,
    class_ids: torch.Tensor | None = None,
    image_normalization: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
    pixmix_provider: PixMixMixingProvider | None = None,
) -> FeatureDenoisingPcaBatch:
    """Sample one Feature Denoising training batch from normalized input images."""

    if config.method == "pixel_masked_embedding_prediction":
        return sample_pixel_masked_embedding_batch(
            images,
            perturbation_forwarder,
            config,
        )
    if config.method == "pixel_augmented_embedding_prediction":
        return sample_pixel_augmented_embedding_batch(
            images,
            perturbation_forwarder,
            config,
            image_normalization,
            pixmix_provider,
        )
    if config.method == "pixel_masked_multilayer_prediction":
        return sample_pixel_masked_multilayer_batch(
            images,
            perturbation_forwarder,
            config,
        )
    if config.method == "pixel_masked_multilayer_l234_prediction":
        return sample_pixel_masked_multilayer_l234_batch(
            images,
            perturbation_forwarder,
            config,
        )
    return sample_feature_denoising_pca_batch(
        _teacher_features(perturbation_forwarder, images),
        config,
        projector,
        feature_normalizer,
        class_channel_corruption,
        class_ids,
    )


def sample_feature_denoising_pca_batch(
    features: torch.Tensor,
    config: FeatureDenoisingConfig,
    projector: PcaProjector | None = None,
    feature_normalizer: FeatureNormalizer | None = None,
    class_channel_corruption: ClassChannelCorruptionBank | None = None,
    class_ids: torch.Tensor | None = None,
) -> FeatureDenoisingPcaBatch:
    """Sample a masked reconstruction batch for the configured Feature Denoising method."""

    if config.method == "pca_masked_reconstruction":
        if projector is None:
            raise ValueError("pca_projector is required for PCA Feature Denoising reconstruction")
        targets = projector.whitened_transform(features)
        keep_mask = sample_feature_denoising_keep_mask(targets, config.pca_mask_probability)
    elif config.method == "spatial_masked_reconstruction":
        targets = features
        keep_mask = sample_spatial_keep_mask(targets, config.mask_probability)
    elif config.method == "spatial_block_residual_reconstruction":
        if feature_normalizer is None:
            raise ValueError(
                "feature_normalizer is required for spatial block residual reconstruction"
            )
        targets = features
        hidden_mask = sample_spatial_block_hidden_mask(
            targets,
            config.spatial_mask_block_sizes,
        )
        keep_mask = 1.0 - hidden_mask
        channel_mean = feature_normalizer.mean.to(
            device=targets.device,
            dtype=targets.dtype,
        )
        if (
            channel_mean.ndim != 4
            or channel_mean.shape[0] != 1
            or channel_mean.shape[1] != targets.shape[1]
            or channel_mean.shape[2:] != (1, 1)
        ):
            raise ValueError(
                "feature_normalizer.mean must have shape (1, C, 1, 1) "
                "matching the teacher feature channels"
            )
        corrupted_features = targets * keep_mask + channel_mean * hidden_mask
        return FeatureDenoisingPcaBatch(
            student_inputs=torch.cat((corrupted_features, hidden_mask), dim=1),
            targets=targets,
            keep_mask=keep_mask,
            corrupted_features=corrupted_features,
        )
    elif config.method == "channel_masked_reconstruction":
        targets = features
        keep_mask = sample_channel_keep_mask(targets, config.mask_probability)
    elif config.method == "channel_masked_residual_reconstruction":
        targets = features
        keep_mask = sample_channel_keep_mask(targets, config.mask_probability)
        corrupted_features = targets * keep_mask
        return FeatureDenoisingPcaBatch(
            student_inputs=corrupted_features,
            targets=targets,
            keep_mask=keep_mask,
            corrupted_features=corrupted_features,
        )
    elif (
        config.method
        in {
            "confusion_channel_replacement_reconstruction",
            "confusion_channel_replacement_residual_reconstruction",
        }
    ):
        if class_channel_corruption is None:
            raise ValueError(
                "class_channel_corruption is required for confusion-channel "
                "replacement"
            )
        if class_ids is None:
            raise ValueError(
                "class_ids are required for confusion-channel replacement"
            )
        return sample_confusion_channel_replacement_batch(
            features=features,
            class_ids=class_ids,
            config=config,
            bank=class_channel_corruption,
        )
    elif config.method == "spatial_token_prediction":
        return sample_spatial_token_prediction_batch(features, config)
    else:
        raise ValueError(f"Unsupported Feature Denoising method: {config.method}")
    return FeatureDenoisingPcaBatch(
        student_inputs=targets * keep_mask,
        targets=targets,
        keep_mask=keep_mask,
    )


class FeatureDenoisingPcaBatch:
    """Student inputs and targets for one Feature Denoising PCA reconstruction batch."""

    def __init__(
        self,
        student_inputs: torch.Tensor | dict[str, torch.Tensor],
        targets: torch.Tensor,
        keep_mask: torch.Tensor,
        corrupted_features: torch.Tensor | None = None,
    ) -> None:
        self.student_inputs = student_inputs
        self.targets = targets
        self.keep_mask = keep_mask
        self.corrupted_features = corrupted_features


def sample_spatial_token_prediction_batch(
    features: torch.Tensor,
    config: FeatureDenoisingConfig,
) -> FeatureDenoisingPcaBatch:
    """Sample visible context and target tokens for spatial token prediction."""

    if features.ndim != 4:
        raise ValueError("spatial token Feature Denoising expects features with shape (B, C, H, W)")
    batch_size, channels, height, width = features.shape
    token_count = height * width
    flat_features = features.flatten(start_dim=2).transpose(1, 2)
    target_token_count = _resolve_target_token_count(config, token_count)
    visible_tokens = torch.zeros(
        batch_size,
        token_count,
        channels,
        device=features.device,
        dtype=features.dtype,
    )
    visible_positions = torch.zeros(
        batch_size,
        token_count,
        device=features.device,
        dtype=torch.long,
    )
    visible_padding_mask = torch.ones(
        batch_size,
        token_count,
        device=features.device,
        dtype=torch.bool,
    )
    target_positions = torch.zeros(
        batch_size,
        target_token_count,
        device=features.device,
        dtype=torch.long,
    )
    target_tokens = torch.zeros(
        batch_size,
        target_token_count,
        channels,
        device=features.device,
        dtype=features.dtype,
    )
    all_positions = torch.arange(token_count, device=features.device, dtype=torch.long)
    for row in range(batch_size):
        target_indices, context_indices = sample_spatial_token_indices(
            height=height,
            width=width,
            config=config,
            device=features.device,
        )
        target_positions[row] = target_indices
        target_tokens[row] = flat_features[row, target_indices]
        visible_count = context_indices.shape[0]
        visible_tokens[row, :visible_count] = flat_features[row, context_indices]
        visible_positions[row, :visible_count] = all_positions[context_indices]
        visible_padding_mask[row, :visible_count] = False
    keep_mask = torch.zeros_like(target_tokens)
    return FeatureDenoisingPcaBatch(
        student_inputs={
            "visible_tokens": visible_tokens,
            "visible_positions": visible_positions,
            "visible_padding_mask": visible_padding_mask,
            "target_positions": target_positions,
        },
        targets=target_tokens,
        keep_mask=keep_mask,
    )


def sample_pixel_masked_embedding_batch(
    images: torch.Tensor,
    perturbation_forwarder: nn.Module,
    config: FeatureDenoisingConfig,
) -> FeatureDenoisingPcaBatch:
    """Predict clean pooled teacher embeddings from pixel-masked image embeddings."""

    if config.embedding_pool not in {"avg", "cls"}:
        raise ValueError(
            "pixel_masked_embedding_prediction supports embedding_pool='avg' or 'cls'"
        )
    keep_mask = sample_pixel_block_keep_mask(images, config)
    masked_images = images * keep_mask
    target_features = perturbation_forwarder.forward_to_features(images)
    context_features = perturbation_forwarder.forward_to_features(masked_images)
    targets = pool_teacher_features(target_features, config)
    contexts = pool_teacher_features(context_features, config)
    return FeatureDenoisingPcaBatch(
        student_inputs=contexts,
        targets=targets,
        keep_mask=torch.zeros_like(targets),
    )


def sample_pixel_augmented_embedding_batch(
    images: torch.Tensor,
    perturbation_forwarder: nn.Module,
    config: FeatureDenoisingConfig,
    image_normalization: tuple[tuple[float, float, float], tuple[float, float, float]] | None,
    pixmix_provider: PixMixMixingProvider | None = None,
) -> FeatureDenoisingPcaBatch:
    """Predict clean pooled teacher embeddings from pixel-augmented embeddings."""

    if config.embedding_pool not in {"avg", "cls"}:
        raise ValueError(
            "pixel_augmented_embedding_prediction supports embedding_pool='avg' or 'cls'"
        )
    if image_normalization is None:
        raise ValueError(
            "image_normalization is required for pixel_augmented_embedding_prediction"
        )
    if config.pixel_augmentation_method == "pixmix":
        if pixmix_provider is None:
            raise ValueError(
                "pixmix_provider is required for PixMix Feature Denoising"
            )
        pixel_batch = sample_pixmix(
            images,
            pixmix_provider.sample(images.shape[0], images.device, images.dtype),
            config.pixmix,
            image_normalization,
        )
    else:
        pixel_batch = sample_pixel_augmentation(
            images,
            feature_denoising_pixel_augmentation_config(config),
            image_normalization,
        )
    target_features = perturbation_forwarder.forward_to_features(
        pixel_batch.clean_images
    )
    context_features = perturbation_forwarder.forward_to_features(
        pixel_batch.perturbed_images
    )
    targets = pool_teacher_features(target_features, config)
    contexts = pool_teacher_features(context_features, config)
    return FeatureDenoisingPcaBatch(
        student_inputs=contexts,
        targets=targets,
        keep_mask=torch.zeros_like(targets),
    )


def feature_denoising_pixel_augmentation_config(config: FeatureDenoisingConfig) -> PerturbationConfig:
    """Build a pixel-augmentation config from Feature Denoising augmentation fields."""

    return PerturbationConfig(
        method="pixel_augmentation",
        rotation_degrees=config.rotation_degrees,
        translate_fraction=config.translate_fraction,
        scale_min=config.scale_min,
        scale_max=config.scale_max,
        brightness_delta=config.brightness_delta,
        contrast_delta=config.contrast_delta,
        evaluation_draws=config.evaluation_draws,
    )


def sample_pixel_masked_multilayer_batch(
    images: torch.Tensor,
    perturbation_forwarder: nn.Module,
    config: FeatureDenoisingConfig,
) -> FeatureDenoisingPcaBatch:
    """Predict clean layer3/layer4/logit targets from masked layer3/layer4 context."""

    return sample_pixel_masked_multilayer_state_batch(
        images=images,
        perturbation_forwarder=perturbation_forwarder,
        config=config,
        include_layer2=False,
    )


def sample_pixel_masked_multilayer_l234_batch(
    images: torch.Tensor,
    perturbation_forwarder: nn.Module,
    config: FeatureDenoisingConfig,
) -> FeatureDenoisingPcaBatch:
    """Predict clean layer2/layer3/layer4/logit targets from masked context."""

    return sample_pixel_masked_multilayer_state_batch(
        images=images,
        perturbation_forwarder=perturbation_forwarder,
        config=config,
        include_layer2=True,
    )


def sample_pixel_masked_multilayer_state_batch(
    images: torch.Tensor,
    perturbation_forwarder: nn.Module,
    config: FeatureDenoisingConfig,
    include_layer2: bool,
) -> FeatureDenoisingPcaBatch:
    """Predict clean multilayer teacher state from masked multilayer context."""

    keep_mask = sample_pixel_block_keep_mask(images, config)
    masked_images = images * keep_mask
    clean = forward_multilayer_teacher_state(
        perturbation_forwarder,
        images,
        config,
        include_layer2=include_layer2,
    )
    masked = forward_multilayer_teacher_state(
        perturbation_forwarder,
        masked_images,
        config,
        include_layer2=include_layer2,
    )
    context_parts = masked.feature_parts()
    target_parts = [*clean.feature_parts(), clean.centered_logits]
    return FeatureDenoisingPcaBatch(
        student_inputs=torch.cat(context_parts, dim=1),
        targets=torch.cat(target_parts, dim=1),
        keep_mask=torch.zeros_like(torch.cat(target_parts, dim=1)),
    )


@dataclass(frozen=True)
class MultilayerTeacherState:
    """Pooled teacher features and centered logits from one forward pass."""

    layer2: torch.Tensor | None
    layer3: torch.Tensor
    layer4: torch.Tensor
    centered_logits: torch.Tensor

    def feature_parts(self) -> list[torch.Tensor]:
        """Return pooled feature tensors in concatenation order."""

        parts = []
        if self.layer2 is not None:
            parts.append(self.layer2)
        parts.extend([self.layer3, self.layer4])
        return parts


def forward_multilayer_teacher_state(
    perturbation_forwarder: nn.Module,
    images: torch.Tensor,
    config: FeatureDenoisingConfig,
    include_layer2: bool,
) -> MultilayerTeacherState:
    """Return pooled teacher feature state and centered logits."""

    teacher = perturbation_forwarder.teacher
    x = teacher.conv1(images)
    x = teacher.bn1(x)
    x = teacher.relu(x)
    x = teacher.maxpool(x)
    x = teacher.layer1(x)
    layer2_features = teacher.layer2(x)
    layer3_features = teacher.layer3(layer2_features)
    layer4_features = teacher.layer4(layer3_features)
    pooled = teacher.avgpool(layer4_features)
    logits = teacher.fc(torch.flatten(pooled, start_dim=1))
    centered_logits = logits - logits.mean(dim=1, keepdim=True)
    return MultilayerTeacherState(
        layer2=(
            pool_teacher_features(layer2_features, config)
            if include_layer2
            else None
        ),
        layer3=pool_teacher_features(layer3_features, config),
        layer4=pool_teacher_features(layer4_features, config),
        centered_logits=centered_logits,
    )


def sample_pixel_block_keep_mask(
    images: torch.Tensor,
    config: FeatureDenoisingConfig,
) -> torch.Tensor:
    """Sample image block keep masks in normalized image space."""

    if images.ndim != 4:
        raise ValueError("pixel image masking expects images with shape (B, C, H, W)")
    batch_size, _channels, height, width = images.shape
    keep_mask = torch.ones(
        batch_size,
        1,
        height,
        width,
        device=images.device,
        dtype=images.dtype,
    )
    for row in range(batch_size):
        for _ in range(config.image_mask_block_count):
            top, left, block_height, block_width = _sample_rectangle(
                height=height,
                width=width,
                scale_min=config.image_mask_scale_min,
                scale_max=config.image_mask_scale_max,
                aspect_ratio_min=config.image_mask_aspect_ratio_min,
                aspect_ratio_max=config.image_mask_aspect_ratio_max,
                device=images.device,
            )
            keep_mask[
                row,
                :,
                top : top + block_height,
                left : left + block_width,
            ] = 0.0
    return keep_mask


def pool_teacher_features(features: torch.Tensor, config: FeatureDenoisingConfig) -> torch.Tensor:
    """Pool teacher feature maps according to the Feature Denoising embedding settings."""

    if config.embedding_pool == "cls":
        if features.ndim != 2:
            raise ValueError("cls embedding pooling expects CLS feature vectors")
        return features
    if config.embedding_pool != "avg":
        raise ValueError("Only avg and cls pooling are supported for Feature Denoising embeddings")
    if features.ndim != 4:
        raise ValueError("avg embedding pooling expects feature maps")
    return features.mean(dim=(2, 3))


def sample_spatial_token_indices(
    height: int,
    width: int,
    config: FeatureDenoisingConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample fixed target tokens and visible context tokens on a feature grid."""

    token_count = height * width
    target_token_count = _resolve_target_token_count(config, token_count)
    target_mask = torch.zeros(token_count, device=device, dtype=torch.bool)
    max_attempts = max(8, config.target_block_count * 8)
    for _ in range(max_attempts):
        rectangle = _sample_rectangle(
            height=height,
            width=width,
            scale_min=config.target_block_scale_min,
            scale_max=config.target_block_scale_max,
            aspect_ratio_min=config.target_aspect_ratio_min,
            aspect_ratio_max=config.target_aspect_ratio_max,
            device=device,
        )
        target_mask[_rectangle_positions(rectangle, width, device)] = True
        if int(target_mask.sum().item()) >= target_token_count:
            break
    target_positions = target_mask.nonzero(as_tuple=False).flatten()
    if target_positions.shape[0] < target_token_count:
        remaining = (~target_mask).nonzero(as_tuple=False).flatten()
        fill_order = torch.randperm(remaining.shape[0], device=device)
        fill = remaining[fill_order[: target_token_count - target_positions.shape[0]]]
        target_positions = torch.cat([target_positions, fill], dim=0)
    if target_positions.shape[0] > target_token_count:
        keep_order = torch.randperm(target_positions.shape[0], device=device)
        target_positions = target_positions[keep_order[:target_token_count]]
    target_positions = target_positions.sort().values

    context_rectangle = _sample_rectangle(
        height=height,
        width=width,
        scale_min=config.context_scale_min,
        scale_max=config.context_scale_max,
        aspect_ratio_min=1.0,
        aspect_ratio_max=1.0,
        device=device,
    )
    context_mask = torch.zeros(token_count, device=device, dtype=torch.bool)
    context_mask[_rectangle_positions(context_rectangle, width, device)] = True
    context_mask[target_positions] = False
    context_positions = context_mask.nonzero(as_tuple=False).flatten()
    if context_positions.numel() == 0:
        non_targets = torch.ones(token_count, device=device, dtype=torch.bool)
        non_targets[target_positions] = False
        context_positions = non_targets.nonzero(as_tuple=False).flatten()
    if context_positions.numel() == 0:
        raise ValueError("spatial_token_prediction requires at least one visible token")
    return target_positions, context_positions.sort().values


def _resolve_target_token_count(config: FeatureDenoisingConfig, token_count: int) -> int:
    if config.target_token_count > 0:
        requested = config.target_token_count
    else:
        average_scale = (config.target_block_scale_min + config.target_block_scale_max) / 2.0
        requested = round(config.target_block_count * average_scale * token_count)
    return max(1, min(token_count - 1, requested))


def _sample_rectangle(
    height: int,
    width: int,
    scale_min: float,
    scale_max: float,
    aspect_ratio_min: float,
    aspect_ratio_max: float,
    device: torch.device,
) -> tuple[int, int, int, int]:
    token_count = height * width
    scale = _uniform(scale_min, scale_max, device)
    ratio = _uniform(aspect_ratio_min, aspect_ratio_max, device)
    target_area = max(1.0, scale * token_count)
    rectangle_height = int(round((target_area / ratio) ** 0.5))
    rectangle_width = int(round((target_area * ratio) ** 0.5))
    rectangle_height = max(1, min(height, rectangle_height))
    rectangle_width = max(1, min(width, rectangle_width))
    top = int(torch.randint(0, height - rectangle_height + 1, (1,), device=device).item())
    left = int(torch.randint(0, width - rectangle_width + 1, (1,), device=device).item())
    return top, left, rectangle_height, rectangle_width


def _rectangle_positions(
    rectangle: tuple[int, int, int, int],
    width: int,
    device: torch.device,
) -> torch.Tensor:
    top, left, height, rectangle_width = rectangle
    rows = torch.arange(top, top + height, device=device)
    columns = torch.arange(left, left + rectangle_width, device=device)
    grid_rows, grid_columns = torch.meshgrid(rows, columns, indexing="ij")
    return (grid_rows * width + grid_columns).flatten().long()


def _uniform(min_value: float, max_value: float, device: torch.device) -> float:
    if min_value == max_value:
        return min_value
    sample = torch.empty((), device=device).uniform_(min_value, max_value)
    return float(sample.item())


def sample_feature_denoising_keep_mask(projected: torch.Tensor, mask_probability: float) -> torch.Tensor:
    """Sample keep masks, forcing at least one hidden component per sample."""

    keep_mask = torch.empty_like(projected).bernoulli_(1.0 - mask_probability)
    all_kept = keep_mask.bool().all(dim=1)
    if all_kept.any():
        rows = all_kept.nonzero(as_tuple=False).flatten()
        columns = torch.randint(
            low=0,
            high=keep_mask.shape[1],
            size=(rows.shape[0],),
            device=keep_mask.device,
        )
        keep_mask[rows, columns] = 0.0
    return keep_mask


def sample_spatial_keep_mask(features: torch.Tensor, mask_probability: float) -> torch.Tensor:
    """Sample spatial keep masks, forcing at least one hidden location."""

    if features.ndim != 4:
        raise ValueError("spatial Feature Denoising masking expects features with shape (B, C, H, W)")
    keep_mask = torch.empty(
        (features.shape[0], 1, features.shape[2], features.shape[3]),
        device=features.device,
        dtype=features.dtype,
    ).bernoulli_(1.0 - mask_probability)
    flat = keep_mask.flatten(start_dim=1)
    all_kept = flat.bool().all(dim=1)
    if all_kept.any():
        rows = all_kept.nonzero(as_tuple=False).flatten()
        columns = torch.randint(
            low=0,
            high=flat.shape[1],
            size=(rows.shape[0],),
            device=flat.device,
        )
        flat[rows, columns] = 0.0
    return flat.reshape_as(keep_mask)


def sample_spatial_block_hidden_mask(
    features: torch.Tensor,
    block_sizes: tuple[int, ...],
) -> torch.Tensor:
    """Sample one uniformly selected square hidden block per feature map."""

    if features.ndim != 4:
        raise ValueError(
            "spatial block Feature Denoising expects features with shape "
            "(B, C, H, W)"
        )
    if not block_sizes or any(block_size <= 0 for block_size in block_sizes):
        raise ValueError("spatial block sizes must contain positive integers")
    batch_size, _channels, height, width = features.shape
    if any(block_size > min(height, width) for block_size in block_sizes):
        raise ValueError("spatial block sizes must fit the feature map")

    hidden_mask = torch.zeros(
        (batch_size, 1, height, width),
        device=features.device,
        dtype=features.dtype,
    )
    size_indices = torch.randint(
        low=0,
        high=len(block_sizes),
        size=(batch_size,),
        device=features.device,
    )
    rows = torch.arange(height, device=features.device).reshape(1, height, 1)
    columns = torch.arange(width, device=features.device).reshape(1, 1, width)
    for size_index, block_size in enumerate(block_sizes):
        selected = size_indices == size_index
        tops = torch.randint(
            low=0,
            high=height - block_size + 1,
            size=(batch_size,),
            device=features.device,
        )
        lefts = torch.randint(
            low=0,
            high=width - block_size + 1,
            size=(batch_size,),
            device=features.device,
        )
        spatial_mask = (
            selected.reshape(-1, 1, 1)
            & (rows >= tops.reshape(-1, 1, 1))
            & (rows < (tops + block_size).reshape(-1, 1, 1))
            & (columns >= lefts.reshape(-1, 1, 1))
            & (columns < (lefts + block_size).reshape(-1, 1, 1))
        )
        hidden_mask[:, 0].masked_fill_(spatial_mask, 1.0)
    return hidden_mask


def sample_channel_keep_mask(features: torch.Tensor, mask_probability: float) -> torch.Tensor:
    """Sample channel keep masks, forcing at least one hidden channel."""

    if features.ndim != 4:
        raise ValueError("channel Feature Denoising masking expects features with shape (B, C, H, W)")
    keep_mask = torch.empty(
        (features.shape[0], features.shape[1], 1, 1),
        device=features.device,
        dtype=features.dtype,
    ).bernoulli_(1.0 - mask_probability)
    flat = keep_mask.flatten(start_dim=1)
    all_kept = flat.bool().all(dim=1)
    if all_kept.any():
        rows = all_kept.nonzero(as_tuple=False).flatten()
        columns = torch.randint(
            low=0,
            high=flat.shape[1],
            size=(rows.shape[0],),
            device=flat.device,
        )
        flat[rows, columns] = 0.0
    return flat.reshape_as(keep_mask)


def sample_confusion_channel_replacement_batch(
    features: torch.Tensor,
    class_ids: torch.Tensor,
    config: FeatureDenoisingConfig,
    bank: ClassChannelCorruptionBank,
) -> FeatureDenoisingPcaBatch:
    """Replace selected channels with adjusted confusing-class prototypes."""

    if features.ndim != 4:
        raise ValueError(
            "confusion-channel replacement expects features with shape "
            "(B, C, H, W)"
        )
    batch_size, channel_count, height, width = features.shape
    class_ids_cpu = class_ids.detach().long().flatten().cpu()
    if class_ids_cpu.shape[0] != batch_size:
        raise ValueError("class_ids must contain one class per feature map")
    num_classes = bank.confusing_class.shape[0]
    if (
        class_ids_cpu.numel() > 0
        and (
            int(class_ids_cpu.min()) < 0
            or int(class_ids_cpu.max()) >= num_classes
        )
    ):
        raise ValueError("class_ids contain a class outside the corruption bank")
    expected_shape = (channel_count, height, width)
    if tuple(bank.prototypes.shape[2:]) != expected_shape:
        raise ValueError(
            "Corruption-bank prototype shape does not match features: "
            f"{tuple(bank.prototypes.shape[2:])} != {expected_shape}"
        )
    if tuple(bank.class_mean.shape) != (num_classes, channel_count, 1, 1):
        raise ValueError(
            "Corruption-bank class statistics do not match feature channels"
        )

    replacement_count = max(1, round(config.mask_probability * channel_count))
    replacement_count = min(channel_count, replacement_count)
    selected_channels = torch.rand(batch_size, channel_count).topk(
        replacement_count,
        dim=1,
        largest=False,
    ).indices
    prototype_indices = torch.randint(
        low=0,
        high=bank.prototypes.shape[1],
        size=(batch_size,),
    )
    donor_classes = bank.confusing_class[class_ids_cpu]
    donor_channels = bank.prototypes[
        donor_classes[:, None],
        prototype_indices[:, None],
        selected_channels,
    ]
    source_mean = bank.class_mean[class_ids_cpu[:, None], selected_channels]
    source_std = bank.class_std[class_ids_cpu[:, None], selected_channels]
    donor_mean = bank.class_mean[donor_classes[:, None], selected_channels]
    donor_std = bank.class_std[donor_classes[:, None], selected_channels]

    dtype = features.dtype
    device = features.device
    donor_channels = donor_channels.to(device=device, dtype=dtype)
    source_mean = source_mean.to(device=device, dtype=dtype)
    source_std = source_std.to(device=device, dtype=dtype)
    donor_mean = donor_mean.to(device=device, dtype=dtype)
    donor_std = donor_std.to(device=device, dtype=dtype)
    adjusted_donor = source_mean + source_std * (
        donor_channels - donor_mean
    ) / donor_std.clamp_min(config.class_statistics_epsilon)

    selected_channels_device = selected_channels.to(device)
    scatter_indices = selected_channels_device[:, :, None, None].expand(
        -1,
        -1,
        height,
        width,
    )
    corrupted_features = features.clone()
    corrupted_features.scatter_(1, scatter_indices, adjusted_donor)
    keep_mask = torch.ones(
        batch_size,
        channel_count,
        1,
        1,
        device=device,
        dtype=dtype,
    )
    keep_mask.scatter_(
        1,
        selected_channels_device[:, :, None, None],
        0.0,
    )
    return FeatureDenoisingPcaBatch(
        student_inputs=corrupted_features,
        targets=features,
        keep_mask=keep_mask,
        corrupted_features=corrupted_features,
    )


def hidden_component_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    keep_mask: torch.Tensor,
) -> torch.Tensor:
    """Return batch-mean MSE on hidden PCA components only."""

    return per_sample_hidden_component_mse(predictions, targets, keep_mask).mean()


def per_sample_hidden_component_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    keep_mask: torch.Tensor,
) -> torch.Tensor:
    """Return one hidden-component MSE value per sample."""

    hidden_mask = (1.0 - keep_mask).expand_as(targets)
    squared_error = (predictions - targets).pow(2) * hidden_mask
    denominator = hidden_mask.flatten(start_dim=1).sum(dim=1).clamp_min(1.0)
    return squared_error.flatten(start_dim=1).sum(dim=1) / denominator


def per_sample_cosine_similarity(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Return rowwise cosine similarity between flattened predictions and targets."""

    predicted = predictions.flatten(start_dim=1)
    target = targets.flatten(start_dim=1)
    numerator = (predicted * target).sum(dim=1)
    denominator = predicted.norm(dim=1) * target.norm(dim=1)
    return numerator / denominator.clamp_min(1.0e-12)


def multilayer_component_scores(
    contexts: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return per-sample component scores for multilayer concatenated prediction."""

    feature_dims = infer_multilayer_feature_dims(contexts.shape[1])
    scores = {}
    start = 0
    for layer_name, dim in feature_dims:
        end = start + dim
        context_part = contexts[:, start:end]
        predicted_part = predictions[:, start:end]
        target_part = targets[:, start:end]
        prediction_error = per_sample_mse(predicted_part, target_part)
        identity_error = per_sample_mse(context_part, target_part)
        scores[f"{layer_name}_prediction_error"] = prediction_error
        scores[f"{layer_name}_identity_error"] = identity_error
        scores[f"{layer_name}_improvement"] = identity_error - prediction_error
        scores[f"{layer_name}_cosine_similarity"] = per_sample_cosine_similarity(
            predicted_part,
            target_part,
        )
        start = end
    predicted_logits = predictions[:, start:]
    target_logits = targets[:, start:]
    scores["logits_prediction_error"] = per_sample_mse(
        predicted_logits,
        target_logits,
    )
    scores["logits_cosine_similarity"] = per_sample_cosine_similarity(
        predicted_logits,
        target_logits,
    )
    return scores


def infer_multilayer_feature_dims(context_dim: int) -> tuple[tuple[str, int], ...]:
    """Infer pooled feature dimensions from concatenated context width."""

    if context_dim == 768:
        return (("layer3", 256), ("layer4", 512))
    if context_dim == 896:
        return (("layer2", 128), ("layer3", 256), ("layer4", 512))
    if context_dim == 3584:
        return (("layer2", 512), ("layer3", 1024), ("layer4", 2048))
    raise ValueError(f"Unsupported multilayer context dimension: {context_dim}")


def per_sample_mse(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return rowwise MSE over flattened tensors."""

    return (left - right).pow(2).flatten(start_dim=1).mean(dim=1)


def feature_denoising_predictions(
    student: nn.Module,
    batch: FeatureDenoisingPcaBatch,
    config: FeatureDenoisingConfig,
) -> torch.Tensor:
    """Return reconstructed targets from direct or residual student outputs."""

    outputs = student(batch.student_inputs)
    if config.method not in {
        "spatial_block_residual_reconstruction",
        "channel_masked_residual_reconstruction",
        "confusion_channel_replacement_residual_reconstruction",
    }:
        return outputs
    if batch.corrupted_features is None:
        raise RuntimeError(
            "Residual Feature Denoising batch is missing corrupted features"
        )
    return batch.corrupted_features + outputs


def feature_denoising_metadata(config: ExperimentConfig, checkpoint: str | None = None) -> dict[str, object]:
    """Return common metadata for Feature Denoising reconstruction artifacts."""

    metadata = {
        "strategy": config.strategy.name,
        "method": config.strategy.feature_denoising.method,
        "student_kind": config.student.kind,
        "feature_layer": config.student.feature_layer,
        "score": feature_denoising_score_name(config.strategy.feature_denoising.method),
        "score_sign": -1,
        "feature_denoising": asdict(config.strategy.feature_denoising),
    }
    if checkpoint is not None:
        metadata["checkpoint"] = checkpoint
    return metadata


def feature_denoising_score_name(method: str) -> str:
    """Return the OOD Score name for a Feature Denoising method."""

    if method == "pca_masked_reconstruction":
        return FEATURE_DENOISING_RECONSTRUCTION_SCORE
    if method == "spatial_masked_reconstruction":
        return FEATURE_DENOISING_SPATIAL_RECONSTRUCTION_SCORE
    if method == "spatial_block_residual_reconstruction":
        return FEATURE_DENOISING_SPATIAL_BLOCK_RESIDUAL_RECONSTRUCTION_SCORE
    if method == "channel_masked_reconstruction":
        return FEATURE_DENOISING_CHANNEL_RECONSTRUCTION_SCORE
    if method == "channel_masked_residual_reconstruction":
        return FEATURE_DENOISING_CHANNEL_RESIDUAL_RECONSTRUCTION_SCORE
    if method == "confusion_channel_replacement_reconstruction":
        return (
            FEATURE_DENOISING_CONFUSION_CHANNEL_REPLACEMENT_RECONSTRUCTION_SCORE
        )
    if method == "confusion_channel_replacement_residual_reconstruction":
        return (
            FEATURE_DENOISING_CONFUSION_CHANNEL_REPLACEMENT_RESIDUAL_RECONSTRUCTION_SCORE
        )
    if method == "spatial_token_prediction":
        return FEATURE_DENOISING_SPATIAL_TOKEN_PREDICTION_SCORE
    if method == "pixel_masked_embedding_prediction":
        return FEATURE_DENOISING_PIXEL_EMBEDDING_PREDICTION_SCORE
    if method == "pixel_augmented_embedding_prediction":
        return FEATURE_DENOISING_PIXEL_AUGMENTED_EMBEDDING_PREDICTION_SCORE
    if method == "pixel_masked_multilayer_prediction":
        return FEATURE_DENOISING_PIXEL_MULTILAYER_PREDICTION_SCORE
    if method == "pixel_masked_multilayer_l234_prediction":
        return FEATURE_DENOISING_PIXEL_MULTILAYER_L234_PREDICTION_SCORE
    raise ValueError(f"Unsupported Feature Denoising method: {method}")


@torch.no_grad()
def fit_class_channel_corruption_bank(
    validation_loader: DataLoader[tuple[torch.Tensor, int]],
    confusion_loader: DataLoader[tuple[torch.Tensor, int]],
    prototype_loader: DataLoader[tuple[torch.Tensor, int]],
    perturbation_forwarder: nn.Module,
    device: torch.device,
    num_classes: int,
    prototype_count: int,
) -> ClassChannelCorruptionBank:
    """Fit class statistics, confusion pairs, and activation-map prototypes."""

    if num_classes <= 1:
        raise ValueError("Class-channel corruption requires at least two classes")
    if prototype_count <= 0:
        raise ValueError("prototype_count must be positive")
    perturbation_forwarder.to(device)
    perturbation_forwarder.eval()

    validation_class_counts = torch.zeros(num_classes, dtype=torch.long)
    value_sum: torch.Tensor | None = None
    squared_value_sum: torch.Tensor | None = None
    value_count = torch.zeros(num_classes, dtype=torch.long)
    feature_shape: tuple[int, int, int] | None = None

    for images, labels in validation_loader:
        labels_cpu = labels.detach().long().flatten().cpu()
        if labels_cpu.shape[0] != images.shape[0]:
            raise ValueError("Validation labels must match the image batch")
        if (
            labels_cpu.numel() > 0
            and (
                int(labels_cpu.min()) < 0
                or int(labels_cpu.max()) >= num_classes
            )
        ):
            raise ValueError("Validation labels contain an unknown class")
        features = perturbation_forwarder.forward_to_features(images.to(device))
        if features.ndim != 4:
            raise ValueError(
                "Class-channel corruption requires 4D teacher feature maps"
            )
        current_shape = tuple(features.shape[1:])
        if feature_shape is None:
            feature_shape = current_shape
            channel_count = features.shape[1]
            value_sum = torch.zeros(
                num_classes,
                channel_count,
                dtype=torch.float64,
            )
            squared_value_sum = torch.zeros_like(value_sum)
        elif current_shape != feature_shape:
            raise ValueError("Teacher feature shape changed across validation batches")

        labels_device = labels_cpu.to(device)
        for class_id_tensor in labels_cpu.unique():
            class_id = int(class_id_tensor)
            class_mask = labels_device == class_id
            class_features = features[class_mask]
            example_count = class_features.shape[0]
            validation_class_counts[class_id] += example_count
            assert value_sum is not None
            assert squared_value_sum is not None
            value_sum[class_id] += class_features.sum(
                dim=(0, 2, 3)
            ).double().cpu()
            squared_value_sum[class_id] += class_features.square().sum(
                dim=(0, 2, 3)
            ).double().cpu()
            value_count[class_id] += (
                example_count * class_features.shape[2] * class_features.shape[3]
            )

    if (
        feature_shape is None
        or value_sum is None
        or squared_value_sum is None
    ):
        raise ValueError("Cannot fit class-channel corruption from an empty validation loader")
    missing_validation_classes = (
        validation_class_counts == 0
    ).nonzero(as_tuple=False).flatten()
    if missing_validation_classes.numel() > 0:
        raise ValueError(
            "Validation split is missing classes required for class-channel "
            f"corruption: {missing_validation_classes.tolist()}"
        )

    mean_flat = value_sum / value_count[:, None]
    variance_flat = (
        squared_value_sum / value_count[:, None] - mean_flat.square()
    ).clamp_min(0.0)
    class_mean = mean_flat.float().reshape(num_classes, -1, 1, 1)
    class_std = variance_flat.sqrt().float().reshape(num_classes, -1, 1, 1)

    confusion_matrix = torch.zeros(
        num_classes,
        num_classes,
        dtype=torch.long,
    )
    confusion_class_counts = torch.zeros(num_classes, dtype=torch.long)
    probability_sum = torch.zeros(
        num_classes,
        num_classes,
        dtype=torch.float64,
    )
    for images, labels in confusion_loader:
        labels_cpu = labels.detach().long().flatten().cpu()
        if labels_cpu.shape[0] != images.shape[0]:
            raise ValueError("Confusion labels must match the image batch")
        if (
            labels_cpu.numel() > 0
            and (
                int(labels_cpu.min()) < 0
                or int(labels_cpu.max()) >= num_classes
            )
        ):
            raise ValueError("Confusion labels contain an unknown class")
        logits, _features = perturbation_forwarder(images.to(device))
        if logits.ndim != 2 or logits.shape[1] != num_classes:
            raise ValueError(
                "Teacher logits do not match the configured class count"
            )
        predictions = logits.argmax(dim=1).detach().cpu()
        flat_pairs = labels_cpu * num_classes + predictions
        confusion_matrix += torch.bincount(
            flat_pairs,
            minlength=num_classes * num_classes,
        ).reshape(num_classes, num_classes)
        probabilities = logits.softmax(dim=1)
        labels_device = labels_cpu.to(device)
        for class_id_tensor in labels_cpu.unique():
            class_id = int(class_id_tensor)
            class_mask = labels_device == class_id
            example_count = int(class_mask.sum())
            confusion_class_counts[class_id] += example_count
            probability_sum[class_id] += probabilities[class_mask].sum(
                dim=0
            ).double().cpu()
    missing_confusion_classes = (
        confusion_class_counts == 0
    ).nonzero(as_tuple=False).flatten()
    if missing_confusion_classes.numel() > 0:
        raise ValueError(
            "Confusion split is missing classes required for class-channel "
            f"corruption: {missing_confusion_classes.tolist()}"
        )
    mean_teacher_probabilities = (
        probability_sum / confusion_class_counts[:, None]
    ).float()

    confusing_class = torch.empty(num_classes, dtype=torch.long)
    for class_id in range(num_classes):
        off_diagonal_counts = confusion_matrix[class_id].clone()
        off_diagonal_counts[class_id] = -1
        if int(off_diagonal_counts.max()) > 0:
            confusing_class[class_id] = off_diagonal_counts.argmax()
        else:
            fallback_probabilities = mean_teacher_probabilities[
                class_id
            ].clone()
            fallback_probabilities[class_id] = -torch.inf
            confusing_class[class_id] = fallback_probabilities.argmax()

    prototypes_by_class: list[list[torch.Tensor]] = [
        [] for _ in range(num_classes)
    ]
    source_indices_by_class: list[list[int]] = [
        [] for _ in range(num_classes)
    ]
    source_offset = 0
    for images, labels in prototype_loader:
        features = perturbation_forwarder.forward_to_features(images.to(device))
        if features.ndim != 4 or tuple(features.shape[1:]) != feature_shape:
            raise ValueError(
                "Prototype feature maps do not match validation feature shape"
            )
        features_cpu = features.detach().float().cpu()
        labels_cpu = labels.detach().long().flatten().cpu()
        for row, class_id_tensor in enumerate(labels_cpu):
            class_id = int(class_id_tensor)
            if class_id < 0 or class_id >= num_classes:
                raise ValueError("Prototype labels contain an unknown class")
            if len(prototypes_by_class[class_id]) >= prototype_count:
                continue
            prototypes_by_class[class_id].append(features_cpu[row])
            source_indices_by_class[class_id].append(source_offset + row)
        source_offset += images.shape[0]
        if all(
            len(class_prototypes) == prototype_count
            for class_prototypes in prototypes_by_class
        ):
            break

    missing_prototypes = {
        class_id: prototype_count - len(class_prototypes)
        for class_id, class_prototypes in enumerate(prototypes_by_class)
        if len(class_prototypes) < prototype_count
    }
    if missing_prototypes:
        raise ValueError(
            "Prototype split does not contain enough examples per class: "
            f"{missing_prototypes}"
        )
    prototypes = torch.stack(
        [
            torch.stack(class_prototypes, dim=0)
            for class_prototypes in prototypes_by_class
        ],
        dim=0,
    )
    prototype_source_indices = torch.tensor(
        source_indices_by_class,
        dtype=torch.long,
    )
    return ClassChannelCorruptionBank(
        confusion_matrix=confusion_matrix,
        confusing_class=confusing_class,
        class_mean=class_mean,
        class_std=class_std,
        prototypes=prototypes,
        prototype_source_indices=prototype_source_indices,
        validation_class_counts=validation_class_counts,
        mean_teacher_probabilities=mean_teacher_probabilities,
    )


def save_class_channel_corruption_bank(
    path: Path,
    bank: ClassChannelCorruptionBank,
    metadata: dict[str, object],
) -> None:
    """Save a class-channel corruption bank and its provenance."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "artifact_version": 1,
            **metadata,
            "confusion_matrix": bank.confusion_matrix.cpu(),
            "confusing_class": bank.confusing_class.cpu(),
            "class_mean": bank.class_mean.cpu(),
            "class_std": bank.class_std.cpu(),
            "prototypes": bank.prototypes.cpu(),
            "prototype_source_indices": bank.prototype_source_indices.cpu(),
            "validation_class_counts": bank.validation_class_counts.cpu(),
            "mean_teacher_probabilities": (
                bank.mean_teacher_probabilities.cpu()
            ),
        },
        path,
    )


def load_class_channel_corruption_bank(
    path: Path,
    expected_prototype_count: int | None = None,
) -> ClassChannelCorruptionBank:
    """Load and validate a class-channel corruption bank."""

    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if artifact.get("artifact_version") != 1:
        raise ValueError(
            f"Unsupported class-channel corruption artifact version: {path}"
        )
    tensor_names = (
        "confusion_matrix",
        "confusing_class",
        "class_mean",
        "class_std",
        "prototypes",
        "prototype_source_indices",
        "validation_class_counts",
        "mean_teacher_probabilities",
    )
    tensors = {}
    for name in tensor_names:
        value = artifact.get(name)
        if not torch.is_tensor(value):
            raise ValueError(
                f"Class-channel corruption artifact is missing tensor {name!r}: "
                f"{path}"
            )
        tensors[name] = value
    num_classes = tensors["confusing_class"].shape[0]
    prototypes = tensors["prototypes"]
    if prototypes.ndim != 5 or prototypes.shape[0] != num_classes:
        raise ValueError(
            f"Class-channel corruption prototypes have an invalid shape: {path}"
        )
    if (
        expected_prototype_count is not None
        and prototypes.shape[1] != expected_prototype_count
    ):
        raise ValueError(
            "Class-channel corruption prototype count differs from config: "
            f"{prototypes.shape[1]} != {expected_prototype_count}"
        )
    expected_square = (num_classes, num_classes)
    if tuple(tensors["confusion_matrix"].shape) != expected_square:
        raise ValueError(f"Invalid confusion-matrix shape in {path}")
    if tuple(tensors["mean_teacher_probabilities"].shape) != expected_square:
        raise ValueError(f"Invalid mean-probability shape in {path}")
    if tuple(tensors["class_mean"].shape) != (
        num_classes,
        prototypes.shape[2],
        1,
        1,
    ):
        raise ValueError(f"Invalid class-statistics shape in {path}")
    if tensors["class_std"].shape != tensors["class_mean"].shape:
        raise ValueError(f"Class mean/std shapes differ in {path}")
    if tuple(tensors["prototype_source_indices"].shape) != tuple(
        prototypes.shape[:2]
    ):
        raise ValueError(f"Invalid prototype source-index shape in {path}")
    if tuple(tensors["validation_class_counts"].shape) != (num_classes,):
        raise ValueError(f"Invalid validation class-count shape in {path}")
    confusing_class = tensors["confusing_class"].long()
    if (
        confusing_class.ndim != 1
        or confusing_class.numel() != num_classes
        or int(confusing_class.min()) < 0
        or int(confusing_class.max()) >= num_classes
        or torch.any(
            confusing_class == torch.arange(num_classes, dtype=torch.long)
        )
    ):
        raise ValueError(f"Invalid confusing-class mapping in {path}")
    return ClassChannelCorruptionBank(
        confusion_matrix=tensors["confusion_matrix"].long(),
        confusing_class=confusing_class,
        class_mean=tensors["class_mean"].float(),
        class_std=tensors["class_std"].float(),
        prototypes=prototypes.float(),
        prototype_source_indices=tensors["prototype_source_indices"].long(),
        validation_class_counts=tensors["validation_class_counts"].long(),
        mean_teacher_probabilities=tensors[
            "mean_teacher_probabilities"
        ].float(),
    )


def class_channel_corruption_path(experiment_dir: Path) -> Path:
    """Return the standard class-channel corruption artifact path."""

    return experiment_dir / "class_channel_corruption.pt"


def fit_feature_normalizer_from_activations(
    activation_path: Path,
    expected_dataset: str | None = None,
) -> FeatureNormalizer:
    """Fit channelwise normalization from a training-split activation artifact."""

    artifact = torch.load(activation_path, map_location="cpu", weights_only=False)
    if expected_dataset is not None:
        if artifact.get("dataset") != expected_dataset or artifact.get("split") != "train":
            raise ValueError(
                "Feature normalization must be fitted from the complete ID "
                f"training-split activation artifact {expected_dataset!r}; got "
                f"dataset={artifact.get('dataset')!r}, split={artifact.get('split')!r}: "
                f"{activation_path}"
            )
    activations = artifact.get("activations")
    if not torch.is_tensor(activations) or activations.ndim != 4:
        raise ValueError(
            f"Activation artifact is missing 4D tensor 'activations': {activation_path}"
        )
    values = activations.to(dtype=torch.float32)
    mean = values.mean(dim=(0, 2, 3), keepdim=True)
    std = values.std(dim=(0, 2, 3), keepdim=True, unbiased=False)
    return FeatureNormalizer(mean=mean, std=std)


def fit_feature_normalizer_from_loader(
    loader: DataLoader[tuple[torch.Tensor, int]],
    perturbation_forwarder: nn.Module,
    device: torch.device,
) -> FeatureNormalizer:
    """Fit channelwise feature statistics from an ID training loader."""

    perturbation_forwarder.to(device)
    perturbation_forwarder.eval()
    value_sum: torch.Tensor | None = None
    squared_value_sum: torch.Tensor | None = None
    value_count = 0
    with torch.no_grad():
        for images, _labels in loader:
            features = _teacher_features(
                perturbation_forwarder,
                images.to(device),
            )
            batch_sum = features.sum(dim=(0, 2, 3)).double().cpu()
            batch_squared_sum = (
                features.square().sum(dim=(0, 2, 3)).double().cpu()
            )
            if value_sum is None:
                value_sum = torch.zeros_like(batch_sum)
                squared_value_sum = torch.zeros_like(batch_squared_sum)
            value_sum += batch_sum
            assert squared_value_sum is not None
            squared_value_sum += batch_squared_sum
            value_count += features.shape[0] * features.shape[2] * features.shape[3]
    if value_sum is None or squared_value_sum is None or value_count == 0:
        raise ValueError("Cannot fit feature normalization from an empty loader")
    mean_flat = value_sum / value_count
    variance_flat = squared_value_sum / value_count - mean_flat.square()
    return FeatureNormalizer(
        mean=mean_flat.float().reshape(1, -1, 1, 1),
        std=variance_flat.clamp_min(0.0).sqrt().float().reshape(1, -1, 1, 1),
    )


def save_feature_normalizer(
    path: Path,
    normalizer: FeatureNormalizer,
    metadata: dict[str, object],
) -> None:
    """Save a feature normalization artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **metadata,
            "mean": normalizer.mean.cpu(),
            "std": normalizer.std.cpu(),
        },
        path,
    )


def load_feature_normalizer(
    path: Path,
    device: torch.device | None = None,
) -> FeatureNormalizer:
    """Load a feature normalization artifact."""

    artifact = torch.load(path, map_location="cpu", weights_only=False)
    mean = artifact.get("mean")
    std = artifact.get("std")
    if not torch.is_tensor(mean) or not torch.is_tensor(std):
        raise ValueError(f"Feature normalizer artifact is missing tensors: {path}")
    normalizer = FeatureNormalizer(mean=mean.float(), std=std.float())
    if device is not None:
        normalizer = normalizer.to(device)
    return normalizer


def feature_normalizer_path(experiment_dir: Path) -> Path:
    """Return the standard feature normalizer artifact path for an experiment."""

    return experiment_dir / "feature_normalizer.pt"


def _teacher_features(
    perturbation_forwarder: nn.Module,
    images: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        return perturbation_forwarder.forward_to_features(images)


def _validate_student_input_shape(
    student: nn.Module,
    student_inputs: torch.Tensor | dict[str, torch.Tensor],
) -> None:
    if isinstance(student_inputs, dict):
        return
    first_parameter = next(student.parameters(), None)
    if first_parameter is None or first_parameter.ndim < 2:
        return
    expected_dim = first_parameter.shape[1]
    actual_dim = (
        student_inputs.shape[1]
        if first_parameter.ndim == 4 and student_inputs.ndim == 4
        else torch.flatten(student_inputs, start_dim=1).shape[1]
    )
    if actual_dim != expected_dim:
        raise ValueError(
            "student.input_shape does not match Feature Denoising inputs: "
            f"expected flattened dimension {expected_dim}, got {actual_dim}"
        )
