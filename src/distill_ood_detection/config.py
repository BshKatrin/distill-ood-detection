"""YAML-backed experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


DistillationMethod = Literal[
    "mse_softmax",
    "cross_entropy",
    "mse_logits",
]
OODDatasetName = Literal["mnist", "svhn", "cifar100"]
LEGACY_METHOD_ALIASES: dict[str, DistillationMethod] = {
    "cross_entropy_softmax": "cross_entropy",
}


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
class OptimizerByMethodConfig:
    """Optimizer settings for each distillation method."""

    mse_softmax: OptimizerConfig = field(default_factory=OptimizerConfig)
    cross_entropy: OptimizerConfig = field(default_factory=OptimizerConfig)
    mse_logits: OptimizerConfig = field(default_factory=OptimizerConfig)

    def for_method(self, method: DistillationMethod) -> OptimizerConfig:
        """Return the optimizer settings for one distillation method."""

        return getattr(self, method)


@dataclass(frozen=True)
class TrainingDefaultsConfig:
    """Training settings shared by all enabled distillation methods."""

    epochs: int = 10
    seed: int = 123
    device: str = "auto"
    log_every_steps: int = 50


@dataclass(frozen=True)
class MethodTrainingConfig:
    """Method-specific training settings."""

    temperature: float | None = None
    alpha: float | None = None


@dataclass(frozen=True)
class TrainingByMethodConfig:
    """Enabled distillation methods and their method-specific settings."""

    mse_softmax: MethodTrainingConfig | None = field(default_factory=MethodTrainingConfig)
    cross_entropy: MethodTrainingConfig | None = field(default_factory=MethodTrainingConfig)
    mse_logits: MethodTrainingConfig | None = field(default_factory=MethodTrainingConfig)

    def enabled_methods(self) -> tuple[DistillationMethod, ...]:
        """Return enabled methods in a deterministic order."""

        return tuple(
            method
            for method in ("mse_softmax", "cross_entropy", "mse_logits")
            if getattr(self, method) is not None
        )


@dataclass(frozen=True)
class ResolvedTrainingMethodConfig:
    """Fully resolved training settings for one distillation method."""

    epochs: int
    seed: int
    device: str
    log_every_steps: int
    temperature: float = 1.0
    alpha: float = 0.5


@dataclass(frozen=True)
class TrainingConfig:
    """Training settings and enabled distillation methods."""

    defaults: TrainingDefaultsConfig = field(default_factory=TrainingDefaultsConfig)
    methods: TrainingByMethodConfig = field(default_factory=TrainingByMethodConfig)

    def enabled_methods(self) -> tuple[DistillationMethod, ...]:
        """Return enabled methods in a deterministic order."""

        return self.methods.enabled_methods()

    def for_method(self, method: DistillationMethod) -> ResolvedTrainingMethodConfig:
        """Resolve the training configuration for one method."""

        method_config = getattr(self.methods, method)
        if method_config is None:
            raise ValueError(f"Training configuration does not enable method: {method}")
        return _resolve_method_training_config(
            method=method,
            defaults=self.defaults,
            method_config=method_config,
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
    optimizer: OptimizerByMethodConfig = field(default_factory=OptimizerByMethodConfig)
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
    optimizer = _parse_optimizer_config(raw.get("optimizer", {}))
    training = _parse_training_config(raw.get("training", {}))
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


def _parse_optimizer_config(raw: dict[str, Any]) -> OptimizerByMethodConfig:
    """Parse optimizer settings from a shared or per-method config block."""

    method_names: tuple[DistillationMethod, ...] = (
        "mse_softmax",
        "cross_entropy",
        "mse_logits",
    )
    optimizer_raw = {
        _normalize_method_name(name) if name in LEGACY_METHOD_ALIASES else name: value
        for name, value in raw.copy().items()
    }
    if any(name in optimizer_raw for name in method_names):
        missing_methods = [name for name in method_names if name not in optimizer_raw]
        if missing_methods:
            missing = ", ".join(missing_methods)
            raise ValueError(
                "optimizer must define settings for every distillation method when "
                f"using per-method configuration; missing: {missing}"
            )
        return OptimizerByMethodConfig(
            **{
                name: OptimizerConfig(**optimizer_raw[name])
                for name in method_names
            }
        )
    shared_optimizer = OptimizerConfig(**optimizer_raw)
    return OptimizerByMethodConfig(
        **{name: shared_optimizer for name in method_names}
    )


def _parse_training_config(raw: dict[str, Any]) -> TrainingConfig:
    """Parse shared and method-specific training settings."""

    training_raw = raw.copy()
    defaults_raw = training_raw.pop("defaults", {}).copy()
    for key in ("epochs", "seed", "device", "log_every_steps"):
        if key in training_raw and key not in defaults_raw:
            defaults_raw[key] = training_raw.pop(key)
    defaults = TrainingDefaultsConfig(**defaults_raw)
    methods = _parse_training_methods(
        raw_methods=training_raw.pop("methods", None),
        legacy_temperature=training_raw.pop("temperature", None),
        legacy_alpha=training_raw.pop("alpha", None),
    )
    if training_raw:
        unknown = ", ".join(sorted(training_raw))
        raise ValueError(f"Unsupported training config fields: {unknown}")
    training = TrainingConfig(defaults=defaults, methods=methods)
    if defaults.epochs <= 0:
        raise ValueError("training.defaults.epochs must be positive")
    if defaults.log_every_steps <= 0:
        raise ValueError("training.defaults.log_every_steps must be positive")
    if not training.enabled_methods():
        raise ValueError("training.methods must enable at least one distillation method")
    for method in training.enabled_methods():
        training.for_method(method)
    return training


def _parse_training_methods(
    raw_methods: Any,
    legacy_temperature: Any,
    legacy_alpha: Any,
) -> TrainingByMethodConfig:
    """Parse per-method training settings."""

    method_names: tuple[DistillationMethod, ...] = (
        "mse_softmax",
        "cross_entropy",
        "mse_logits",
    )
    if raw_methods is None:
        methods_raw: dict[str, Any] = {name: {} for name in method_names}
    elif isinstance(raw_methods, list):
        methods_raw = {
            _normalize_method_name(method): {}
            for method in raw_methods
        }
    elif isinstance(raw_methods, dict):
        methods_raw = {
            _normalize_method_name(name) if name in LEGACY_METHOD_ALIASES else name: value
            for name, value in raw_methods.items()
        }
    else:
        raise ValueError("training.methods must be a mapping from method name to config")

    if (legacy_temperature is not None or legacy_alpha is not None) and (
        raw_methods is None or "cross_entropy" in methods_raw
    ):
        cross_entropy_raw = methods_raw.setdefault("cross_entropy", {})
        if not isinstance(cross_entropy_raw, dict):
            raise ValueError("training.methods.cross_entropy must be a mapping")
        if legacy_temperature is not None and "temperature" not in cross_entropy_raw:
            cross_entropy_raw["temperature"] = legacy_temperature
        if legacy_alpha is not None and "alpha" not in cross_entropy_raw:
            cross_entropy_raw["alpha"] = legacy_alpha

    unknown_methods = [name for name in methods_raw if name not in method_names]
    if unknown_methods:
        unknown = ", ".join(sorted(unknown_methods))
        raise ValueError(f"Unsupported training methods: {unknown}")

    return TrainingByMethodConfig(
        **{
            method: _parse_method_training_config(method, methods_raw.get(method))
            for method in method_names
        }
    )


def _parse_method_training_config(
    method: DistillationMethod,
    raw: Any,
) -> MethodTrainingConfig | None:
    """Parse method-specific training settings for one method."""

    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"training.methods.{method} must be a mapping")
    return MethodTrainingConfig(**raw)


def _resolve_method_training_config(
    method: DistillationMethod,
    defaults: TrainingDefaultsConfig,
    method_config: MethodTrainingConfig,
) -> ResolvedTrainingMethodConfig:
    """Resolve one method's training settings and validate method-specific fields."""

    if method == "cross_entropy":
        temperature = method_config.temperature if method_config.temperature is not None else 1.0
        alpha = method_config.alpha if method_config.alpha is not None else 0.5
        if temperature <= 0.0:
            raise ValueError("training.methods.cross_entropy.temperature must be positive")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("training.methods.cross_entropy.alpha must be between 0.0 and 1.0")
        return ResolvedTrainingMethodConfig(
            epochs=defaults.epochs,
            seed=defaults.seed,
            device=defaults.device,
            log_every_steps=defaults.log_every_steps,
            temperature=temperature,
            alpha=alpha,
        )
    if method_config.temperature is not None:
        raise ValueError(f"training.methods.{method}.temperature is not supported")
    if method_config.alpha is not None:
        raise ValueError(f"training.methods.{method}.alpha is not supported")
    return ResolvedTrainingMethodConfig(
        epochs=defaults.epochs,
        seed=defaults.seed,
        device=defaults.device,
        log_every_steps=defaults.log_every_steps,
    )


def _normalize_method_name(method: str) -> DistillationMethod:
    """Map legacy method names to the current vocabulary."""

    return LEGACY_METHOD_ALIASES.get(method, method)  # type: ignore[return-value]
