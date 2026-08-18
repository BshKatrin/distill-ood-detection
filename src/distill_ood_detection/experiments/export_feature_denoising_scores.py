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
    collect_channel_knn_reconstruction_scores,
    collect_feature_denoising_reconstruction_scores,
    feature_denoising_metadata,
    feature_normalizer_path,
    load_channel_knn_index,
    load_channel_groups,
    load_class_channel_corruption_bank,
    load_feature_normalizer,
)
from distill_ood_detection.distillation.perturbation import (
    load_pca_projector,
    pca_projector_path,
)
from distill_ood_detection.distillation.nmf import (
    load_nmf_concept_projector,
    nmf_concept_projector_path,
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
        loaders.append(
            build_in_distribution_train_loader(config.dataset, training_defaults.seed)
        )
    if include_validation:
        loaders.append(
            build_in_distribution_validation_loader(
                config.dataset, training_defaults.seed
            )
        )
    loaders.extend(
        [
            build_in_distribution_test_loader(config.dataset),
            *build_ood_loaders(config.dataset),
        ]
    )

    teacher = load_teacher(config.teacher, device)
    perturbation_forwarder = build_feature_forwarder(
        teacher, config.student.feature_layer
    )
    perturbation_forwarder.to(device)
    perturbation_forwarder.eval()
    is_knn = config.strategy.feature_denoising.method in {
        "channel_masked_knn_reconstruction",
        "channel_group_masked_knn_reconstruction",
    }
    pca_projector = None
    if config.strategy.feature_denoising.method == "pca_masked_reconstruction":
        pca_projector = load_pca_projector(pca_projector_path(experiment_dir), device)
    nmf_projector = None
    nmf_path = nmf_concept_projector_path(experiment_dir)
    if (
        config.strategy.feature_denoising.method
        == "nmf_concept_masked_residual_reconstruction"
    ):
        if not nmf_path.exists():
            raise FileNotFoundError(
                f"Missing Feature Denoising NMF projector artifact: {nmf_path}"
            )
        nmf_projector = load_nmf_concept_projector(nmf_path, device)
        if (
            nmf_projector.concept_count
            != config.strategy.feature_denoising.nmf_components
        ):
            raise ValueError(
                "NMF projector concept count does not match the resolved config"
            )
        if nmf_projector.channel_count != config.student.input_shape[0]:
            raise ValueError(
                "NMF projector channels do not match the student feature shape"
            )
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
    if config.strategy.feature_denoising.method in {
        "confusion_channel_replacement_reconstruction",
        "confusion_channel_replacement_residual_reconstruction",
    }:
        if not corruption_path.exists():
            raise FileNotFoundError(
                f"Missing class-channel corruption artifact: {corruption_path}"
            )
        class_channel_corruption = load_class_channel_corruption_bank(
            corruption_path,
            expected_prototype_count=(
                config.strategy.feature_denoising.prototype_count
            ),
        )
    channel_groups = None
    if (
        config.strategy.feature_denoising.method
        in {
            "channel_group_masked_residual_reconstruction",
            "channel_group_stratified_masked_residual_reconstruction",
            "channel_group_masked_knn_reconstruction",
        }
    ):
        group_config = config.strategy.feature_denoising
        if group_config.channel_group_path is None:
            raise ValueError("Channel-group masking requires channel_group_path")
        if config.student.feature_layer is None:
            raise ValueError("Channel-group masking requires student.feature_layer")
        channel_groups = load_channel_groups(
            Path(group_config.channel_group_path),
            device,
            distance_threshold=group_config.channel_group_distance_threshold,
            expected_dataset=f"{config.dataset.name}_train",
            expected_layer=config.student.feature_layer,
            expected_feature_shape=config.student.input_shape,
        )
    student = None
    checkpoint_path = None
    knn_index = None
    if is_knn:
        activation_path = config.strategy.feature_denoising.activation_path
        if activation_path is None:
            raise ValueError(
                "channel_masked_knn_reconstruction requires activation_path"
            )
        feature_layer = config.student.feature_layer
        if feature_layer is None:
            raise ValueError(
                "channel_masked_knn_reconstruction requires student.feature_layer"
            )
        knn_index = load_channel_knn_index(
            Path(activation_path),
            device,
            expected_dataset=f"{config.dataset.name}_train",
            expected_layer=feature_layer,
            expected_feature_shape=config.student.input_shape,
        )
    else:
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
        if is_knn:
            if knn_index is None:
                raise RuntimeError("k-NN reference index was not initialized")
            scores = collect_channel_knn_reconstruction_scores(
                loader=named_loader.loader,
                device=device,
                perturbation_forwarder=perturbation_forwarder,
                feature_denoising_config=config.strategy.feature_denoising,
                index=knn_index,
                channel_groups=channel_groups,
            )
            path = output_dir / named_loader.name / "knn.pt"
        else:
            if student is None:
                raise RuntimeError("Feature Denoising student was not initialized")
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
                channel_groups=channel_groups,
                nmf_projector=nmf_projector,
            )
            path = output_dir / named_loader.name / f"student_{checkpoint}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            **feature_denoising_metadata(
                config,
                checkpoint=None if is_knn else checkpoint,
            ),
            "dataset": named_loader.name,
            "split": named_loader.split,
            "model": "knn" if is_knn else "student",
        }
        if checkpoint_path is not None:
            metadata["checkpoint_path"] = str(checkpoint_path)
        if knn_index is not None:
            metadata.update(
                {
                    "reference_activation_path": (
                        config.strategy.feature_denoising.activation_path
                    ),
                    "reference_count": knn_index.reference_count,
                    "distance": "masked_squared_l2_pre_gap",
                    "aggregation": "mean",
                }
            )
        if feature_normalizer is not None:
            metadata["feature_normalizer_path"] = str(normalizer_path)
        if class_channel_corruption is not None:
            metadata["class_channel_corruption_path"] = str(corruption_path)
        if channel_groups is not None:
            group_index = config.strategy.feature_denoising.channel_group_index
            group_size = (
                int(channel_groups.group_sizes[group_index].item())
                if group_index is not None
                else None
            )
            is_fractional = (
                config.strategy.feature_denoising.method
                == "channel_group_stratified_masked_residual_reconstruction"
            )
            masked_count = (
                max(
                    1,
                    int(
                        config.strategy.feature_denoising.channel_group_mask_fraction
                        * group_size
                        + 0.5
                    ),
                )
                if is_fractional and group_size is not None
                else group_size
            )
            metadata.update(
                {
                    "channel_group_path": channel_groups.source_path,
                    "channel_group_distance_threshold": (
                        channel_groups.distance_threshold
                    ),
                    "channel_group_count": channel_groups.group_count,
                    "channel_group_sizes": channel_groups.group_sizes.cpu().tolist(),
                    "channel_group_index": group_index,
                    "channel_group_size": group_size,
                    "masked_channel_count": masked_count,
                    "realized_mask_fraction": (
                        masked_count / group_size
                        if masked_count is not None and group_size is not None
                        else None
                    ),
                    "channel_group_sampling": (
                        "within_fixed_group_per_sample_per_draw"
                        if is_fractional and group_index is not None
                        else "stratified_within_all_eligible_groups_per_sample_per_draw"
                        if is_fractional
                        else "fixed_complete_group"
                        if group_index is not None
                        else "uniform_per_sample_per_draw"
                    ),
                }
            )
        if nmf_projector is not None:
            metadata.update(
                {
                    "nmf_concept_projector_path": str(nmf_path),
                    "nmf_concept_count": nmf_projector.concept_count,
                    "nmf_fit_reconstruction_error": (
                        nmf_projector.fit_reconstruction_error
                    ),
                    "nmf_fit_n_iter": nmf_projector.fit_n_iter,
                }
            )
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
                "model": "knn" if is_knn else "student",
                "checkpoint": None if is_knn else checkpoint,
                "path": str(path),
            }
        )

    manifest = {
        "experiment_name": config.experiment_name,
        "dataset": asdict(config.dataset),
        "teacher": asdict(config.teacher),
        "strategy": asdict(config.strategy),
        "checkpoint_selection": None if is_knn else checkpoint,
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
        raise FileNotFoundError(
            f"Missing Feature Denoising metrics artifact: {metrics_path}"
        )
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
