"""Train PyTorch students from a pretrained teacher."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import mlflow
import torch

from distill_ood_detection.config import DistillationMethod, ExperimentConfig
from distill_ood_detection.datasets.inference import build_id_loaders
from distill_ood_detection.distillation.train import train_student
from distill_ood_detection.distillation.perturbation import (
    fit_pca_projector_from_activations,
    pca_projector_path,
    save_pca_projector,
)
from distill_ood_detection.evaluation.metrics import accuracy, distillation_metrics
from distill_ood_detection.models.student import build_student
from distill_ood_detection.models.teacher import (
    ResNetFeatureForwarder,
    TeacherFeatureExtractor,
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
    experiment_dir = Path(config.output_dir) / config.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    write_json(experiment_dir / "resolved_config.json", asdict(config))
    if config.mlflow.enabled:
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment(config.mlflow.experiment_name)

    loaders = build_id_loaders(config.dataset, seed=training_defaults.seed)
    teacher = load_teacher(config.teacher, device)
    pca_projector = None
    if config.strategy.name == "perturbation" and config.strategy.perturbation.method == "pca_projection":
        activation_path = config.strategy.perturbation.pca_activation_path
        if activation_path is None:
            raise ValueError(
                "strategy.perturbation.pca_activation_path is required for PCA projection"
            )
        pca_projector = fit_pca_projector_from_activations(
            Path(activation_path),
            config.strategy.perturbation.pca_components,
            expected_dataset=f"{config.dataset.name}_train",
        ).to(device)
        save_pca_projector(
            pca_projector_path(experiment_dir),
            pca_projector,
            {
                "activation_path": activation_path,
                "pca_components": config.strategy.perturbation.pca_components,
                "feature_layer": config.student.feature_layer,
            },
        )
    perturbation_forwarder = (
        ResNetFeatureForwarder(teacher, config.student.feature_layer)
        if config.strategy.name == "perturbation"
        else None
    )
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


def _mlflow_method_run(config: ExperimentConfig, method: DistillationMethod):
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
