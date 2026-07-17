"""Train PyTorch students from a pretrained teacher."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import mlflow
import torch

from distill_ood_detection.config import DistillationMethod, ExperimentConfig
from distill_ood_detection.datasets.inference import build_id_loaders, dataset_normalization
from distill_ood_detection.datasets.pixmix import build_pixmix_mixing_provider
from distill_ood_detection.distillation.train import train_student
from distill_ood_detection.distillation.activation_subspace import (
    collect_clean_pooled_embeddings,
    component_training_tensors,
    fit_activation_subspaces,
    save_activation_subspaces,
    train_activation_subspace_student,
)
from distill_ood_detection.distillation.perturbation import (
    fit_pca_projector_from_activations,
    pca_projector_path,
    save_pca_projector,
)
from distill_ood_detection.distillation.feature_denoising import (
    reconstruction_loss,
    train_feature_denoising_student,
)
from distill_ood_detection.evaluation.metrics import accuracy, distillation_metrics
from distill_ood_detection.models.student import build_student
from distill_ood_detection.models.teacher import (
    TeacherFeatureExtractor,
    build_feature_forwarder,
    load_teacher,
)
from distill_ood_detection.utils import resolve_device, set_seed, write_json


def run_experiment(
    config: ExperimentConfig,
    method: DistillationMethod | None = None,
) -> list[dict[str, float | int | str]]:
    """Run one or all student distillation methods from an experiment config."""

    training_defaults = config.training.defaults
    set_seed(training_defaults.seed)
    device = resolve_device(training_defaults.device)
    methods = (method,) if method else config.training.enabled_methods()
    experiment_dir = Path(config.run_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    write_json(experiment_dir / "resolved_config.json", asdict(config))
    if config.mlflow.enabled:
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment(config.mlflow.experiment_name)

    loaders = build_id_loaders(config.dataset, seed=training_defaults.seed)
    image_normalization = dataset_normalization(config.dataset)
    perturbation_pixmix_provider = (
        build_pixmix_mixing_provider(
            config.dataset,
            config.strategy.perturbation.pixmix,
            training_defaults.seed,
        )
        if (
            config.strategy.name == "perturbation"
            and config.strategy.perturbation.method == "pixmix"
        )
        else None
    )
    feature_denoising_pixmix_provider = (
        build_pixmix_mixing_provider(
            config.dataset,
            config.strategy.feature_denoising.pixmix,
            training_defaults.seed,
        )
        if (
            config.strategy.name == "feature_denoising"
            and config.strategy.feature_denoising.method
            == "pixel_augmented_embedding_prediction"
            and config.strategy.feature_denoising.pixel_augmentation_method == "pixmix"
        )
        else None
    )
    teacher = load_teacher(config.teacher, device)
    pca_projector = None
    if (
        (
            config.strategy.name == "perturbation"
            and config.strategy.perturbation.method
            in {"pca_projection", "pca_masked_projection"}
        )
        or (
            config.strategy.name == "feature_denoising"
            and config.strategy.feature_denoising.method == "pca_masked_reconstruction"
        )
    ):
        activation_path = (
            config.strategy.feature_denoising.pca_activation_path
            if config.strategy.name == "feature_denoising"
            else config.strategy.perturbation.pca_activation_path
        )
        if activation_path is None:
            raise ValueError(
                "pca_activation_path is required for PCA-based training"
            )
        pca_components = (
            config.strategy.feature_denoising.pca_components
            if config.strategy.name == "feature_denoising"
            else config.strategy.perturbation.pca_components
        )
        pca_projector = fit_pca_projector_from_activations(
            Path(activation_path),
            pca_components,
            expected_dataset=f"{config.dataset.name}_train",
        ).to(device)
        save_pca_projector(
            pca_projector_path(experiment_dir),
            pca_projector,
            {
                "activation_path": activation_path,
                "pca_components": pca_components,
                "pca_mask_probability": (
                    config.strategy.feature_denoising.pca_mask_probability
                    if config.strategy.name == "feature_denoising"
                    else config.strategy.perturbation.pca_mask_probability
                ),
                "feature_layer": config.student.feature_layer,
                "strategy": config.strategy.name,
            },
        )
    if (
        config.strategy.name == "perturbation"
        and config.strategy.perturbation.method == "clipping"
    ):
        perturbation_forwarder = build_feature_forwarder(teacher, "layer4")
    elif config.strategy.name in {
        "perturbation",
        "feature_denoising",
        "activation_subspace",
    }:
        if config.student.feature_layer is None:
            raise ValueError("student.feature_layer is required for this strategy")
        perturbation_forwarder = build_feature_forwarder(
            teacher,
            config.student.feature_layer,
        )
    else:
        perturbation_forwarder = None
    feature_extractor = (
        TeacherFeatureExtractor(teacher, config.student.feature_layer)
        if config.student.feature_layer is not None and perturbation_forwarder is None
        else None
    )
    teacher_metrics = {
        "validation_accuracy": accuracy(teacher, loaders.validation, device),
        "test_accuracy": accuracy(teacher, loaders.test, device),
    }
    write_json(experiment_dir / "teacher_metrics.json", teacher_metrics)

    summaries: list[dict[str, float | int | str]] = []
    with _mlflow_parent_run(config):
        if config.mlflow.enabled:
            mlflow.log_params(_flatten_config(asdict(config)))
            mlflow.log_metrics(
                {
                    "teacher_validation_accuracy": teacher_metrics["validation_accuracy"],
                    "teacher_test_accuracy": teacher_metrics["test_accuracy"],
                }
            )
            mlflow.log_artifact(str(experiment_dir / "resolved_config.json"))

        if config.strategy.name == "activation_subspace":
            if perturbation_forwarder is None:
                raise ValueError(
                    "Activation-subspace training requires a feature forwarder"
                )
            classifier = getattr(teacher, "fc", None)
            if not isinstance(classifier, torch.nn.Linear):
                raise ValueError(
                    "Activation-subspace training requires a ResNet linear fc head"
                )
            subspace_config = config.strategy.activation_subspace
            train_split = collect_clean_pooled_embeddings(
                forwarder=perturbation_forwarder,
                loader=loaders.train,
                device=device,
            )
            validation_split = collect_clean_pooled_embeddings(
                forwarder=perturbation_forwarder,
                loader=loaders.validation,
                device=device,
            )
            test_split = collect_clean_pooled_embeddings(
                forwarder=perturbation_forwarder,
                loader=loaders.test,
                device=device,
            )
            subspaces = fit_activation_subspaces(
                classifier_weight=classifier.weight.detach(),
                train_embeddings=train_split.embeddings,
                device=device,
            )
            component_dimension = subspaces.component_dimension(
                subspace_config.component
            )
            if component_dimension != subspace_config.expected_dimension:
                raise ValueError(
                    "Fitted activation-subspace dimension differs from config: "
                    f"{component_dimension} != {subspace_config.expected_dimension}"
                )
            if config.student.input_shape != (component_dimension,):
                raise ValueError(
                    "student.input_shape must match the fitted component dimension: "
                    f"{config.student.input_shape} != {(component_dimension,)}"
                )
            if (
                subspace_config.target == "projected_logits"
                and config.student.kind != "linear"
            ):
                raise ValueError(
                    "Projected-logit activation-subspace targets require a linear student"
                )
            if (
                subspace_config.target == "coordinates"
                and config.student.kind != "autoencoder"
            ):
                raise ValueError(
                    "Coordinate activation-subspace targets require an autoencoder"
                )
            save_activation_subspaces(
                path=experiment_dir / "activation_subspace.pt",
                subspaces=subspaces,
                config=subspace_config,
                fit_samples=train_split.embeddings.shape[0],
                classifier_weight_shape=tuple(classifier.weight.shape),
            )
            train_data = component_training_tensors(
                split=train_split,
                subspaces=subspaces,
                component=subspace_config.component,
                target=subspace_config.target,
                classifier=classifier,
                device=device,
            )
            validation_data = component_training_tensors(
                split=validation_split,
                subspaces=subspaces,
                component=subspace_config.component,
                target=subspace_config.target,
                classifier=classifier,
                device=device,
            )
            test_data = component_training_tensors(
                split=test_split,
                subspaces=subspaces,
                component=subspace_config.component,
                target=subspace_config.target,
                classifier=classifier,
                device=device,
            )
            student = build_student(config.student)
            output_dir = experiment_dir / subspace_config.component
            if len(methods) != 1:
                raise ValueError(
                    "Activation-subspace training requires exactly one enabled "
                    "distillation method"
                )
            activation_method = methods[0]
            if (
                subspace_config.target == "coordinates"
                and activation_method != "mse_logits"
            ):
                raise ValueError(
                    "Coordinate activation-subspace training requires "
                    "training.methods.mse_logits"
                )
            method_training_config = config.training.for_method(activation_method)
            with _mlflow_method_run(config, subspace_config.component):
                if config.mlflow.enabled:
                    mlflow.log_param(
                        "distillation_method",
                        f"activation_subspace_{subspace_config.component}_"
                        f"{activation_method}",
                    )
                    mlflow.log_param("distillation_strategy", config.strategy.name)
                    mlflow.log_param(
                        "decisive_dimension",
                        subspaces.decisive_dimension,
                    )
                    mlflow.log_param(
                        "insignificant_dimension",
                        subspaces.insignificant_dimension,
                    )
                summary = train_activation_subspace_student(
                    student=student,
                    component=subspace_config.component,
                    target=subspace_config.target,
                    distillation_method=activation_method,
                    train_data=train_data,
                    validation_data=validation_data,
                    test_data=test_data,
                    batch_size=config.dataset.batch_size,
                    device=device,
                    optimizer_config=config.optimizer.activation_subspace,
                    training_config=method_training_config,
                    output_dir=output_dir,
                    mlflow_enabled=config.mlflow.enabled,
                )
                summary.update(
                    {
                        "activation_dimension": subspaces.activation_dimension,
                        "decisive_dimension": subspaces.decisive_dimension,
                        "insignificant_dimension": subspaces.insignificant_dimension,
                        "subspace_path": str(
                            experiment_dir / "activation_subspace.pt"
                        ),
                    }
                )
                write_json(output_dir / "metrics.json", summary)
                if config.mlflow.enabled:
                    mlflow.log_artifact(str(output_dir / "metrics.json"))
                    mlflow.log_artifact(str(output_dir / "history.json"))
                    mlflow.log_artifact(
                        str(experiment_dir / "activation_subspace.pt")
                    )
                summaries.append(summary)
            write_json(experiment_dir / "summary.json", {"methods": summaries})
            return summaries

        if config.strategy.name == "feature_denoising":
            if perturbation_forwarder is None:
                raise ValueError("Feature Denoising training requires a feature forwarder")
            method_training_config = config.training.for_method("mse_logits")
            student = build_student(config.student)
            output_dir = experiment_dir / config.strategy.feature_denoising.method
            with _mlflow_method_run(config, config.strategy.feature_denoising.method):
                if config.mlflow.enabled:
                    mlflow.log_param("distillation_method", config.strategy.feature_denoising.method)
                    mlflow.log_param("distillation_strategy", config.strategy.name)
                summary = train_feature_denoising_student(
                    student=student,
                    train_loader=loaders.train,
                    validation_loader=loaders.validation,
                    device=device,
                    optimizer_config=config.optimizer.pca_masked_reconstruction,
                    training_config=method_training_config,
                    output_dir=output_dir,
                    mlflow_enabled=config.mlflow.enabled,
                    perturbation_forwarder=perturbation_forwarder,
                    feature_denoising_config=config.strategy.feature_denoising,
                    pca_projector=pca_projector,
                    image_normalization=image_normalization,
                    pixmix_provider=feature_denoising_pixmix_provider,
                )
                best_checkpoint_path = Path(str(summary["best_checkpoint_path"]))
                student.load_state_dict(
                    torch.load(
                        best_checkpoint_path,
                        map_location=device,
                        weights_only=True,
                    )
                )
                test_reconstruction_loss = reconstruction_loss(
                    student=student,
                    loader=loaders.test,
                    device=device,
                    perturbation_forwarder=perturbation_forwarder,
                    feature_denoising_config=config.strategy.feature_denoising,
                    pca_projector=pca_projector,
                    image_normalization=image_normalization,
                    pixmix_provider=feature_denoising_pixmix_provider,
                )
                summary["test_reconstruction_loss"] = test_reconstruction_loss
                write_json(output_dir / "metrics.json", summary)
                if config.mlflow.enabled:
                    mlflow.log_metric("test_reconstruction_loss", test_reconstruction_loss)
                    mlflow.log_artifact(str(output_dir / "metrics.json"))
                    mlflow.log_artifact(str(output_dir / "history.json"))
                summaries.append(summary)
            write_json(experiment_dir / "summary.json", {"methods": summaries})
            return summaries

        for current_method in methods:
            method_training_config = config.training.for_method(current_method)
            student = build_student(config.student)
            output_dir = experiment_dir / current_method
            with _mlflow_method_run(config, current_method):
                if config.mlflow.enabled:
                    mlflow.log_param("distillation_method", current_method)
                    mlflow.log_param("distillation_strategy", config.strategy.name)
                summary = train_student(
                    method=current_method,
                    teacher=teacher,
                    student=student,
                    train_loader=loaders.train,
                    validation_loader=loaders.validation,
                    device=device,
                    optimizer_config=config.optimizer.for_method(current_method),
                    training_config=method_training_config,
                    output_dir=output_dir,
                    mlflow_enabled=config.mlflow.enabled,
                    feature_extractor=feature_extractor,
                    perturbation_forwarder=perturbation_forwarder,
                    perturbation_config=(
                        config.strategy.perturbation
                        if config.strategy.name == "perturbation"
                        else None
                    ),
                    pca_projector=pca_projector,
                    image_normalization=image_normalization,
                    pixmix_provider=perturbation_pixmix_provider,
                )
                best_checkpoint_path = Path(str(summary["best_checkpoint_path"]))
                student.load_state_dict(
                    torch.load(
                        best_checkpoint_path,
                        map_location=device,
                        weights_only=True,
                    )
                )
                test_metrics = distillation_metrics(
                    method=current_method,
                    teacher=teacher,
                    student=student,
                    loader=loaders.test,
                    device=device,
                    temperature=method_training_config.temperature,
                    alpha=method_training_config.alpha,
                    feature_extractor=feature_extractor,
                    perturbation_forwarder=perturbation_forwarder,
                    perturbation_config=(
                        config.strategy.perturbation
                        if config.strategy.name == "perturbation"
                        else None
                    ),
                    pca_projector=pca_projector,
                    image_normalization=image_normalization,
                    pixmix_provider=perturbation_pixmix_provider,
                    apply_perturbation=(
                        config.strategy.name == "perturbation"
                        and config.strategy.perturbation.method
                        in {"pixel_augmentation", "pixmix"}
                    ),
                    split="test",
                )
                summary.update(test_metrics)
                write_json(output_dir / "metrics.json", summary)
                if config.mlflow.enabled:
                    mlflow.log_metrics(test_metrics)
                    mlflow.log_artifact(str(output_dir / "metrics.json"))
                    mlflow.log_artifact(str(output_dir / "history.json"))
                summaries.append(summary)

    write_json(experiment_dir / "summary.json", {"methods": summaries})
    return summaries


def _mlflow_parent_run(config: ExperimentConfig):
    if not config.mlflow.enabled:
        return _null_context()
    return mlflow.start_run(run_name=config.experiment_name)


def _mlflow_method_run(config: ExperimentConfig, method: str):
    if not config.mlflow.enabled:
        return _null_context()
    return mlflow.start_run(run_name=f"{config.experiment_name}/{method}", nested=True)


def _flatten_config(config: dict[str, object], prefix: str = "") -> dict[str, object]:
    flattened: dict[str, object] = {}
    for key, value in config.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten_config(value, name))
        elif isinstance(value, (str, int, float, bool)):
            flattened[name] = value
        elif isinstance(value, (list, tuple)):
            flattened[name] = ",".join(str(item) for item in value)
        else:
            flattened[name] = str(value)
    return flattened


class _null_context:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None
