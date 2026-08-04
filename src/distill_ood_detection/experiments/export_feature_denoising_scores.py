"""Export Feature Denoising reconstruction OOD score artifacts."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Literal

import torch

from distill_ood_detection.config import ExperimentConfig
from distill_ood_detection.datasets.inference import (
    build_in_distribution_test_loader,
    build_in_distribution_train_loader,
    build_in_distribution_validation_loader,
    build_ood_loaders,
    dataset_normalization,
)
from distill_ood_detection.datasets.pixmix import build_pixmix_mixing_provider
from distill_ood_detection.distillation.feature_denoising import (
    class_channel_corruption_path,
    collect_feature_denoising_reconstruction_scores,
    feature_denoising_metadata,
    feature_normalizer_path,
    load_class_channel_corruption_bank,
    load_feature_normalizer,
)
from distill_ood_detection.distillation.perturbation import (
    load_pca_projector,
    pca_projector_path,
)
from distill_ood_detection.models.student import build_student
from distill_ood_detection.models.teacher import build_feature_forwarder, load_teacher
from distill_ood_detection.utils import resolve_device, set_seed, write_json

CheckpointSelection = Literal["best", "latest"]


def run_feature_denoising_score_export(
    config: ExperimentConfig,
    checkpoint: CheckpointSelection = "best",
    include_train: bool = False,
    include_validation: bool = False,
) -> dict[str, object]:
    """Export reconstruction scores for ID and configured OOD datasets."""

    if config.strategy.name != "feature_denoising":
        raise ValueError(
            "export-feature-denoising-scores requires strategy.name: feature_denoising"
        )
    training_defaults = config.training.defaults
    set_seed(training_defaults.seed)
    device = resolve_device(training_defaults.device)
    experiment_dir = Path(config.run_dir)
    output_dir = experiment_dir / "feature_denoising_scores"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_normalization = dataset_normalization(config.dataset)

    loaders = []
    if include_train:
        loaders.append(build_in_distribution_train_loader(config.dataset, training_defaults.seed))
    if include_validation:
        loaders.append(build_in_distribution_validation_loader(config.dataset, training_defaults.seed))
    loaders.extend(
        [
            build_in_distribution_test_loader(config.dataset),
            *build_ood_loaders(config.dataset),
        ]
    )

    teacher = load_teacher(config.teacher, device)
    perturbation_forwarder = build_feature_forwarder(teacher, config.student.feature_layer)
    perturbation_forwarder.to(device)
    perturbation_forwarder.eval()
    pca_projector = None
    if config.strategy.feature_denoising.method == "pca_masked_reconstruction":
        pca_projector = load_pca_projector(pca_projector_path(experiment_dir), device)
    feature_normalizer = None
    normalizer_path = feature_normalizer_path(experiment_dir)
    if (
        config.strategy.feature_denoising.method
        == "spatial_block_residual_reconstruction"
    ):
        if not normalizer_path.exists():
            raise FileNotFoundError(
                f"Missing Feature Denoising normalizer artifact: {normalizer_path}"
            )
        feature_normalizer = load_feature_normalizer(normalizer_path, device)
    class_channel_corruption = None
    corruption_path = class_channel_corruption_path(experiment_dir)
    if (
        config.strategy.feature_denoising.method
        in {
            "confusion_channel_replacement_reconstruction",
            "confusion_channel_replacement_residual_reconstruction",
        }
    ):
        if not corruption_path.exists():
            raise FileNotFoundError(
                "Missing class-channel corruption artifact: "
                f"{corruption_path}"
            )
        class_channel_corruption = load_class_channel_corruption_bank(
            corruption_path,
            expected_prototype_count=(
                config.strategy.feature_denoising.prototype_count
            ),
        )
    student = build_student(config.student)
    checkpoint_path = _student_checkpoint_path(
        experiment_dir,
        config.strategy.feature_denoising.method,
        checkpoint,
    )
    student.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    student.to(device)
    student.eval()
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

    artifacts = []
    for named_loader in loaders:
        set_seed(training_defaults.seed)
        scores = collect_feature_denoising_reconstruction_scores(
            student=student,
            loader=named_loader.loader,
            device=device,
            perturbation_forwarder=perturbation_forwarder,
            feature_denoising_config=config.strategy.feature_denoising,
            pca_projector=pca_projector,
            feature_normalizer=feature_normalizer,
            class_channel_corruption=class_channel_corruption,
            image_normalization=image_normalization,
            pixmix_provider=pixmix_provider,
        )
        path = output_dir / named_loader.name / f"student_{checkpoint}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            **feature_denoising_metadata(config, checkpoint=checkpoint),
            "dataset": named_loader.name,
            "split": named_loader.split,
            "checkpoint_path": str(checkpoint_path),
        }
        if feature_normalizer is not None:
            metadata["feature_normalizer_path"] = str(normalizer_path)
        if class_channel_corruption is not None:
            metadata["class_channel_corruption_path"] = str(corruption_path)
        torch.save(
            {
                **scores,
                "metadata": metadata,
            },
            path,
        )
        artifacts.append(
            {
                "dataset": named_loader.name,
                "split": named_loader.split,
                "model": "student",
                "checkpoint": checkpoint,
                "path": str(path),
            }
        )

    manifest = {
        "experiment_name": config.experiment_name,
        "dataset": asdict(config.dataset),
        "teacher": asdict(config.teacher),
        "strategy": asdict(config.strategy),
        "checkpoint_selection": checkpoint,
        "artifacts": artifacts,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def _student_checkpoint_path(
    experiment_dir: Path,
    method: str,
    checkpoint: CheckpointSelection,
) -> Path:
    metrics_path = experiment_dir / method / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing Feature Denoising metrics artifact: {metrics_path}")
    import json

    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    key = f"{checkpoint}_checkpoint_path"
    raw_path = metrics.get(key)
    if not isinstance(raw_path, str):
        raise ValueError(f"Missing {key} in {metrics_path}")
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise FileNotFoundError(f"Student checkpoint not found: {path}")
    return path
