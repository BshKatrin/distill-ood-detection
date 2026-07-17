"""Export Feature Denoising errors in ActSub classifier subspaces."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader

from distill_ood_detection.config import ExperimentConfig
from distill_ood_detection.datasets.inference import (
    build_in_distribution_test_loader,
    build_in_distribution_train_loader,
    build_ood_loaders,
    dataset_normalization,
)
from distill_ood_detection.datasets.pixmix import build_pixmix_mixing_provider
from distill_ood_detection.distillation.feature_denoising import (
    collect_feature_denoising_reconstruction_scores,
    feature_denoising_metadata,
    pool_teacher_features,
)
from distill_ood_detection.evaluation.activation_subspaces import (
    classifier_svd,
    select_balanced_subspace_dimension,
)
from distill_ood_detection.experiments.export_feature_denoising_scores import (
    _student_checkpoint_path,
)
from distill_ood_detection.models.student import build_student
from distill_ood_detection.models.teacher import build_feature_forwarder, load_teacher
from distill_ood_detection.utils import resolve_device, set_seed, write_json

CheckpointSelection = Literal["best", "latest"]
SUBSPACE_SCORE_KEYS = (
    "decisive_reconstruction_error",
    "decisive_identity_error",
    "decisive_improvement",
    "decisive_relative_improvement",
    "insignificant_reconstruction_error",
    "insignificant_identity_error",
    "insignificant_improvement",
    "insignificant_relative_improvement",
)


def run_feature_denoising_subspace_error_export(
    config: ExperimentConfig,
    checkpoint: CheckpointSelection = "best",
) -> dict[str, object]:
    """Export exact per-draw errors in decisive and insignificant subspaces."""

    if config.strategy.name != "feature_denoising":
        raise ValueError(
            "Feature Denoising subspace export requires strategy.name: feature_denoising"
        )
    if config.student.feature_layer != "layer4":
        raise ValueError(
            "Classifier-weight activation subspaces are only defined for pooled layer4 "
            f"embeddings; got {config.student.feature_layer!r}"
        )
    if config.strategy.feature_denoising.method not in {
        "pixel_masked_embedding_prediction",
        "pixel_augmented_embedding_prediction",
    }:
        raise ValueError(
            "Activation-subspace export currently requires a single-layer pixel "
            "embedding prediction method"
        )

    training_defaults = config.training.defaults
    set_seed(training_defaults.seed)
    device = resolve_device(training_defaults.device)
    experiment_dir = Path(config.run_dir)
    teacher = load_teacher(config.teacher, device)
    forwarder = build_feature_forwarder(teacher, config.student.feature_layer)
    forwarder.to(device)
    forwarder.eval()
    train_loader = build_in_distribution_train_loader(
        config.dataset,
        training_defaults.seed,
    )
    train_targets = _collect_clean_train_embeddings(
        loader=train_loader.loader,
        device=device,
        forwarder=forwarder,
        config=config,
    )
    classifier_weight = _resnet_classifier_weight(teacher)
    singular_values, right_basis = classifier_svd(classifier_weight)
    decisive_dimension, norm_gaps = select_balanced_subspace_dimension(
        right_basis,
        train_targets,
    )
    if decisive_dimension in {0, right_basis.shape[0]}:
        raise ValueError(
            "ActSub selected an empty activation subspace; cannot compute normalized "
            f"component errors (k={decisive_dimension})"
        )

    student = build_student(config.student)
    checkpoint_path = _student_checkpoint_path(
        experiment_dir,
        config.strategy.feature_denoising.method,
        checkpoint,
    )
    student.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    student.to(device)
    student.eval()

    output_dir = experiment_dir / "feature_denoising_subspace_errors"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "right_basis": right_basis.cpu(),
            "singular_values": singular_values.cpu(),
            "decisive_dimension": decisive_dimension,
            "insignificant_dimension": right_basis.shape[0] - decisive_dimension,
            "norm_gaps": norm_gaps.cpu(),
            "classifier_weight_shape": tuple(classifier_weight.shape),
            "fit_dataset": f"{config.dataset.name}_train",
            "fit_samples": train_targets.shape[0],
        },
        output_dir / "subspace.pt",
    )

    loaders = [
        build_in_distribution_test_loader(config.dataset),
        *build_ood_loaders(config.dataset),
    ]
    artifacts = []
    image_normalization = dataset_normalization(config.dataset)
    pixmix_provider = (
        build_pixmix_mixing_provider(
            config.dataset,
            config.strategy.feature_denoising.pixmix,
            training_defaults.seed,
        )
        if (
            config.strategy.feature_denoising.method
            == "pixel_augmented_embedding_prediction"
            and config.strategy.feature_denoising.pixel_augmentation_method == "pixmix"
        )
        else None
    )
    for named_loader in loaders:
        set_seed(training_defaults.seed)
        all_scores = collect_feature_denoising_reconstruction_scores(
            student=student,
            loader=named_loader.loader,
            device=device,
            perturbation_forwarder=forwarder,
            feature_denoising_config=config.strategy.feature_denoising,
            image_normalization=image_normalization,
            pixmix_provider=pixmix_provider,
            activation_subspace_basis=right_basis,
            decisive_subspace_dimension=decisive_dimension,
        )
        scores = {
            "labels": all_scores["labels"],
            "full_reconstruction_error": all_scores["raw_reconstruction_error"],
            "full_identity_error": all_scores["identity_error"],
            "full_relative_improvement": (
                all_scores["identity_error"] - all_scores["raw_reconstruction_error"]
            )
            / all_scores["identity_error"].clamp_min(1.0e-12),
            **{key: all_scores[key] for key in SUBSPACE_SCORE_KEYS},
        }
        path = output_dir / named_loader.name / f"student_{checkpoint}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                **scores,
                "metadata": {
                    **feature_denoising_metadata(config, checkpoint=checkpoint),
                    "dataset": named_loader.name,
                    "split": named_loader.split,
                    "decisive_dimension": decisive_dimension,
                    "insignificant_dimension": right_basis.shape[0]
                    - decisive_dimension,
                    "checkpoint_path": str(checkpoint_path),
                },
            },
            path,
        )
        artifacts.append(
            {
                "dataset": named_loader.name,
                "split": named_loader.split,
                "path": str(path),
            }
        )

    manifest = {
        "version": 1,
        "experiment_name": config.experiment_name,
        "dataset": asdict(config.dataset),
        "teacher": asdict(config.teacher),
        "strategy": asdict(config.strategy),
        "checkpoint_selection": checkpoint,
        "subspace": {
            "method": "classifier_weight_svd_actsub_norm_balance",
            "fit_dataset": f"{config.dataset.name}_train",
            "fit_samples": train_targets.shape[0],
            "activation_dimension": right_basis.shape[0],
            "decisive_dimension": decisive_dimension,
            "insignificant_dimension": right_basis.shape[0] - decisive_dimension,
            "path": str(output_dir / "subspace.pt"),
        },
        "artifacts": artifacts,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def _collect_clean_train_embeddings(
    *,
    loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
    forwarder: torch.nn.Module,
    config: ExperimentConfig,
) -> torch.Tensor:
    """Collect clean pooled layer4 targets used only to select ActSub ``k``."""

    targets = []
    with torch.no_grad():
        for images, _labels in loader:
            features = forwarder.forward_to_features(images.to(device))
            targets.append(
                pool_teacher_features(
                    features,
                    config.strategy.feature_denoising,
                ).cpu()
            )
    return torch.cat(targets, dim=0).to(device)


def _resnet_classifier_weight(teacher: torch.nn.Module) -> torch.Tensor:
    classifier = getattr(teacher, "fc", None)
    if not isinstance(classifier, torch.nn.Linear):
        raise ValueError("Layer4 activation subspaces require a ResNet linear fc head")
    return classifier.weight.detach()
