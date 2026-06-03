"""YAML-backed experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


DistillationMethod = Literal["mse", "cross_entropy_probabilities"]
OODDatasetName = Literal["mnist", "svhn"]


@dataclass(frozen=True)
class OODDatasetConfig:
    """Out-of-distribution dataset associated with the ID dataset."""

    name: OODDatasetName
    split: str = "test"


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset settings shared by teacher and student."""

    name: str = "cifar10"
    data_dir: str = "data"
    batch_size: int = 256
    num_workers: int = 2
    validation_fraction: float = 0.1
    ood_datasets: tuple[OODDatasetConfig, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TeacherConfig:
    """Teacher model settings."""

    hf_model_id: str = "edadaltocg/resnet18_cifar10"
    revision: str = "main"


@dataclass(frozen=True)
class StudentConfig:
    """Student model settings."""

    kind: str = "linear"
    input_shape: tuple[int, int, int] = (3, 32, 32)
    num_classes: int = 10


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer settings for the student."""

    name: str = "sgd"
    learning_rate: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 0.0


@dataclass(frozen=True)
class TrainingConfig:
    """Training loop settings."""

    epochs: int = 10
    seed: int = 123
    device: str = "auto"
    log_every_steps: int = 50
    methods: tuple[DistillationMethod, ...] = (
        "mse",
        "cross_entropy_probabilities",
    )


@dataclass(frozen=True)
class MlflowConfig:
    """MLflow tracking settings."""

    enabled: bool = True
    experiment_name: str = "distill-ood-detection"
    tracking_uri: str = "sqlite:///mlflow.db"


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete experiment configuration."""

    experiment_name: str
    output_dir: str = "runs"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    mlflow: MlflowConfig = field(default_factory=MlflowConfig)


def load_config(path: Path) -> ExperimentConfig:
    """Load an experiment configuration from a YAML file."""

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return parse_config(raw)


def parse_config(raw: dict[str, Any]) -> ExperimentConfig:
    """Parse a raw dictionary into typed configuration objects."""

    dataset_raw = raw.get("dataset", {}).copy()
    if "ood_datasets" in dataset_raw:
        dataset_raw["ood_datasets"] = tuple(
            OODDatasetConfig(**item) for item in dataset_raw["ood_datasets"]
        )
    dataset = DatasetConfig(**dataset_raw)
    teacher = TeacherConfig(**raw.get("teacher", {}))
    student_raw = raw.get("student", {}).copy()
    if "input_shape" in student_raw:
        student_raw["input_shape"] = tuple(student_raw["input_shape"])
    student = StudentConfig(**student_raw)
    optimizer = OptimizerConfig(**raw.get("optimizer", {}))
    training_raw = raw.get("training", {}).copy()
    if "methods" in training_raw:
        training_raw["methods"] = tuple(training_raw["methods"])
    training = TrainingConfig(**training_raw)
    mlflow = MlflowConfig(**raw.get("mlflow", {}))

    return ExperimentConfig(
        experiment_name=raw["experiment_name"],
        output_dir=raw.get("output_dir", "runs"),
        dataset=dataset,
        teacher=teacher,
        student=student,
        optimizer=optimizer,
        training=training,
        mlflow=mlflow,
    )
