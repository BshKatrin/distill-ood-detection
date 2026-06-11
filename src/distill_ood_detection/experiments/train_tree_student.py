"""Train sklearn random-forest students from teacher predictions."""

from __future__ import annotations

import pickle
import time
from dataclasses import asdict
from pathlib import Path

import mlflow
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from torch import nn
from torch.utils.data import DataLoader

from distill_ood_detection.config import (
    ExperimentConfig,
    TreeDistillationMode,
)
from distill_ood_detection.distillation.perturbation import sample_clipping_perturbation
from distill_ood_detection.datasets.inference import (
    build_in_distribution_test_loader,
    build_in_distribution_train_loader,
    build_in_distribution_validation_loader,
)
from distill_ood_detection.inference import (
    ModelOutputs,
    predict_tree_model_outputs,
    save_model_outputs,
)
from distill_ood_detection.models.teacher import ResNetFeatureForwarder, load_teacher
from distill_ood_detection.utils import resolve_device, set_seed, write_json


def run_tree_experiment(
    config: ExperimentConfig,
    mode: TreeDistillationMode | None = None,
) -> list[dict[str, float | int | str]]:
    """Run one or all random-forest student distillation modes."""

    if config.student.kind != "random_forest":
        raise ValueError(
            "Tree training expects student.kind: random_forest; "
            f"got {config.student.kind!r}"
        )

    training_defaults = config.training.defaults
    set_seed(training_defaults.seed)
    device = resolve_device(training_defaults.device)
    modes = (mode,) if mode else config.tree.enabled_modes()
    experiment_dir = Path(config.output_dir) / config.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    write_json(experiment_dir / "resolved_config.json", asdict(config))
    if config.mlflow.enabled:
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment(config.mlflow.experiment_name)

    train_loader = build_in_distribution_train_loader(
        config.dataset,
        seed=training_defaults.seed,
    )
    validation_loader = build_in_distribution_validation_loader(
        config.dataset,
        seed=training_defaults.seed,
    )
    test_loader = build_in_distribution_test_loader(config.dataset)
    teacher = load_teacher(config.teacher, device)
    perturbation_forwarder = (
        ResNetFeatureForwarder(teacher, config.student.feature_layer)
        if config.strategy.name == "perturbation"
        else None
    )

    train_data = _collect_tree_dataset(
        teacher,
        train_loader.loader,
        device,
        perturbation_forwarder=perturbation_forwarder,
        config=config,
    )
    train_teacher_path = (
        experiment_dir
        / "teacher_inference"
        / train_loader.name
        / "teacher.pt"
    )
    save_model_outputs(
        path=train_teacher_path,
        outputs=train_data.outputs,
        metadata={
            "model": "teacher",
            "dataset": train_loader.name,
            "split": train_loader.split,
            "strategy": config.strategy.name,
        },
    )

    summaries: list[dict[str, float | int | str]] = []
    with _mlflow_parent_run(config):
        if config.mlflow.enabled:
            mlflow.log_params(_flatten_config(asdict(config)))
            mlflow.log_artifact(str(experiment_dir / "resolved_config.json"))
            mlflow.log_artifact(str(train_teacher_path))

        for current_mode in modes:
            output_dir = experiment_dir / current_mode
            output_dir.mkdir(parents=True, exist_ok=True)
            config.tree.for_mode(current_mode)
            with _mlflow_mode_run(config, current_mode):
                if config.mlflow.enabled:
                    mlflow.log_param("tree_distillation_mode", current_mode)
                summary = _train_random_forest_student(
                    mode=current_mode,
                    train_data=train_data,
                    validation_loader=validation_loader.loader,
                    test_loader=test_loader.loader,
                    teacher=teacher,
                    device=device,
                    config=config,
                    output_dir=output_dir,
                    perturbation_forwarder=perturbation_forwarder,
                )
                summary["teacher_train_inference_path"] = str(train_teacher_path)
                write_json(output_dir / "metrics.json", summary)
                if config.mlflow.enabled:
                    mlflow.log_metrics(
                        {
                            key: value
                            for key, value in summary.items()
                            if isinstance(value, (float, int))
                        }
                    )
                    mlflow.log_artifact(str(output_dir / "metrics.json"))
                    mlflow.log_artifact(str(output_dir / "history.json"))
                summaries.append(summary)

    write_json(experiment_dir / "summary.json", {"methods": summaries})
    return summaries


def _train_random_forest_student(
    mode: TreeDistillationMode,
    train_data: "_TreeDataset",
    validation_loader: DataLoader[tuple[torch.Tensor, int]],
    test_loader: DataLoader[tuple[torch.Tensor, int]],
    teacher: nn.Module,
    device: torch.device,
    config: ExperimentConfig,
    output_dir: Path,
    perturbation_forwarder: ResNetFeatureForwarder | None = None,
) -> dict[str, float | int | str]:
    started_at = time.time()
    targets = _target_matrix(
        mode=mode,
        teacher_logits=train_data.outputs.logits.numpy(),
    )
    model = RandomForestRegressor(
        n_estimators=config.tree.random_forest.n_estimators,
        max_depth=config.tree.random_forest.max_depth,
        min_samples_split=config.tree.random_forest.min_samples_split,
        min_samples_leaf=config.tree.random_forest.min_samples_leaf,
        max_features=config.tree.random_forest.max_features,
        bootstrap=config.tree.random_forest.bootstrap,
        n_jobs=config.tree.random_forest.n_jobs,
        random_state=config.training.defaults.seed,
    )
    model.fit(train_data.features, targets)

    checkpoint_path = output_dir / "student_random_forest.pkl"
    with checkpoint_path.open("wb") as handle:
        pickle.dump(
            {
                "model": model,
                "mode": mode,
                "student": asdict(config.student),
                "random_forest": asdict(config.tree.random_forest),
                "strategy": asdict(config.strategy),
            },
            handle,
        )

    train_metrics = _tree_metrics(
        model=model,
        mode=mode,
        features=train_data.features,
        teacher_logits=train_data.outputs.logits.numpy(),
        labels=train_data.outputs.labels.numpy(),
        prefix="train",
    )
    validation_data = _collect_tree_dataset(
        teacher,
        validation_loader,
        device,
        perturbation_forwarder=perturbation_forwarder,
        config=config,
    )
    validation_metrics = _tree_metrics(
        model=model,
        mode=mode,
        features=validation_data.features,
        teacher_logits=validation_data.outputs.logits.numpy(),
        labels=validation_data.outputs.labels.numpy(),
        prefix="validation",
    )
    test_data = _collect_tree_dataset(
        teacher,
        test_loader,
        device,
        perturbation_forwarder=perturbation_forwarder,
        config=config,
    )
    test_metrics = _tree_metrics(
        model=model,
        mode=mode,
        features=test_data.features,
        teacher_logits=test_data.outputs.logits.numpy(),
        labels=test_data.outputs.labels.numpy(),
        prefix="test",
    )

    history = {
        "history": [
            {
                "step": "fit",
                **train_metrics,
                **validation_metrics,
            }
        ]
    }
    write_json(output_dir / "history.json", history)

    return {
        "method": mode,
        "mode": mode,
        "n_estimators": config.tree.random_forest.n_estimators,
        **train_metrics,
        **validation_metrics,
        **test_metrics,
        "latest_checkpoint_path": str(checkpoint_path),
        "best_checkpoint_path": str(checkpoint_path),
        "checkpoint_path": str(checkpoint_path),
        "seconds": round(time.time() - started_at, 3),
    }


def _target_matrix(
    mode: TreeDistillationMode,
    teacher_logits: np.ndarray,
) -> np.ndarray:
    if mode == "logits":
        return _center_logits(teacher_logits).astype(np.float32)
    raise ValueError(f"Unsupported tree distillation mode: {mode}")


def _tree_metrics(
    model: RandomForestRegressor,
    mode: TreeDistillationMode,
    features: np.ndarray,
    teacher_logits: np.ndarray,
    labels: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    student_logits, student_probabilities = predict_tree_model_outputs(model, mode, features)
    teacher_probabilities = _softmax(teacher_logits)
    predictions = student_logits.argmax(axis=1)
    target = _target_matrix(
        mode=mode,
        teacher_logits=teacher_logits,
    )
    prediction_target = _center_logits(student_logits)
    target_mse = np.mean((prediction_target - target) ** 2)
    kl_divergence = _kl_divergence(teacher_probabilities, student_probabilities)
    return {
        f"{prefix}_accuracy": float(np.mean(predictions == labels)),
        f"{prefix}_target_mse": float(target_mse),
        f"{prefix}_kl_divergence": float(kl_divergence),
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _center_logits(logits: np.ndarray) -> np.ndarray:
    return logits - logits.mean(axis=1, keepdims=True)


def _kl_divergence(teacher_probabilities: np.ndarray, student_probabilities: np.ndarray) -> float:
    clipped_teacher = np.clip(teacher_probabilities, 1e-12, 1.0)
    clipped_student = np.clip(student_probabilities, 1e-12, 1.0)
    return float(
        np.mean(
            np.sum(
                clipped_teacher * (np.log(clipped_teacher) - np.log(clipped_student)),
                axis=1,
            )
        )
    )


@torch.no_grad()
def _collect_tree_dataset(
    teacher: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
    perturbation_forwarder: ResNetFeatureForwarder | None = None,
    config: ExperimentConfig | None = None,
) -> "_TreeDataset":
    feature_batches: list[np.ndarray] = []
    logits_batches: list[torch.Tensor] = []
    probability_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    teacher.eval()
    if perturbation_forwarder is not None:
        perturbation_forwarder.eval()
    for images, labels in loader:
        images = images.to(device)
        if perturbation_forwarder is not None:
            if config is None:
                raise ValueError("config is required for perturbation tree collection")
            features = perturbation_forwarder.forward_to_features(images)
            perturbation_batch = sample_clipping_perturbation(
                features,
                config.strategy.perturbation,
            )
            feature_batches.append(
                torch.flatten(perturbation_batch.student_inputs, start_dim=1)
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            logits = perturbation_forwarder.forward_from_features(
                perturbation_batch.perturbed_features
            )
        else:
            feature_batches.append(
                torch.flatten(images.cpu(), start_dim=1).numpy().astype(np.float32)
            )
            logits = teacher(images)
        probabilities = torch.softmax(logits, dim=1)
        logits_batches.append(logits.cpu())
        probability_batches.append(probabilities.cpu())
        label_batches.append(labels.cpu())
    return _TreeDataset(
        features=np.concatenate(feature_batches, axis=0),
        outputs=ModelOutputs(
            logits=torch.cat(logits_batches, dim=0),
            probabilities=torch.cat(probability_batches, dim=0),
            labels=torch.cat(label_batches, dim=0),
        ),
    )


class _TreeDataset:
    def __init__(self, features: np.ndarray, outputs: ModelOutputs) -> None:
        self.features = features
        self.outputs = outputs


def _mlflow_parent_run(config: ExperimentConfig):
    if not config.mlflow.enabled:
        return _null_context()
    return mlflow.start_run(run_name=config.experiment_name)


def _mlflow_mode_run(config: ExperimentConfig, mode: TreeDistillationMode):
    if not config.mlflow.enabled:
        return _null_context()
    return mlflow.start_run(run_name=f"{config.experiment_name}/{mode}", nested=True)


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
