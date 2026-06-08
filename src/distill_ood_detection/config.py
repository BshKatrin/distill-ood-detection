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
TreeDistillationMode = Literal["logits", "softmax"]
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
    hidden_channels: tuple[int, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RandomForestConfig:
    """Random forest settings for tree-based students."""

    n_estimators: int = 200
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str | int | float | None = "sqrt"
    bootstrap: bool = True
    n_jobs: int | None = -1


@dataclass(frozen=True)
class TreeMethodConfig:
    """Method-specific settings for tree distillation."""

    temperature: float | None = None
    alpha: float | None = None


@dataclass(frozen=True)
class TreeMethodsConfig:
    """Enabled tree distillation modes."""

    logits: TreeMethodConfig | None = field(default_factory=TreeMethodConfig)
    softmax: TreeMethodConfig | None = field(default_factory=TreeMethodConfig)

    def enabled_modes(self) -> tuple[TreeDistillationMode, ...]:
        """Return enabled tree distillation modes in a deterministic order."""

        return tuple(
            mode
            for mode in ("logits", "softmax")
            if getattr(self, mode) is not None
        )


@dataclass(frozen=True)
class ResolvedTreeMethodConfig:
    """Fully resolved settings for one tree distillation mode."""

    temperature: float = 1.0
    alpha: float = 1.0


@dataclass(frozen=True)
class TreeConfig:
    """Tree-based student experiment settings."""

    random_forest: RandomForestConfig = field(default_factory=RandomForestConfig)
    methods: TreeMethodsConfig = field(default_factory=TreeMethodsConfig)

    def enabled_modes(self) -> tuple[TreeDistillationMode, ...]:
        """Return enabled tree distillation modes in a deterministic order."""

        return self.methods.enabled_modes()

    def for_mode(self, mode: TreeDistillationMode) -> ResolvedTreeMethodConfig:
        """Resolve the training configuration for one tree distillation mode."""

        method_config = getattr(self.methods, mode)
        if method_config is None:
            raise ValueError(f"Tree configuration does not enable mode: {mode}")
        return _resolve_tree_method_config(mode, method_config)


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
    tree: TreeConfig = field(default_factory=TreeConfig)
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
    if "hidden_channels" in student_raw:
        student_raw["hidden_channels"] = tuple(student_raw["hidden_channels"])
    student = _parse_student_config(student_raw)
    optimizer = _parse_optimizer_config(raw.get("optimizer", {}))
    training = _parse_training_config(
        raw.get("training", {}),
        require_methods=student.kind != "random_forest",
    )
    tree = _parse_tree_config(raw.get("tree", {}))
    mlflow = MlflowConfig(**raw.get("mlflow", {}))

    return ExperimentConfig(
        experiment_name=raw["experiment_name"],
        output_dir=raw.get("output_dir", "runs"),
        dataset=dataset,
        teacher=teacher,
        student=student,
        optimizer=optimizer,
        training=training,
        tree=tree,
        mlflow=mlflow,
    )


def _parse_student_config(raw: dict[str, Any]) -> StudentConfig:
    """Parse neural-network student settings."""

    student = StudentConfig(**raw)
    if student.kind not in {"linear", "mlp", "random_forest"}:
        raise ValueError(f"Unsupported student kind: {student.kind}")
    if len(student.input_shape) != 3:
        raise ValueError("student.input_shape must contain channel, height, and width")
    if any(dimension <= 0 for dimension in student.input_shape):
        raise ValueError("student.input_shape dimensions must be positive")
    if student.num_classes <= 0:
        raise ValueError("student.num_classes must be positive")
    if student.kind == "mlp" and not student.hidden_channels:
        raise ValueError(
            "student.hidden_channels must define at least one hidden layer "
            "for MLP students"
        )
    if any(hidden_channel <= 0 for hidden_channel in student.hidden_channels):
        raise ValueError("student.hidden_channels values must be positive")
    return student


def _parse_tree_config(raw: dict[str, Any]) -> TreeConfig:
    """Parse tree-based student settings."""

    tree_raw = raw.copy()
    random_forest = RandomForestConfig(**tree_raw.pop("random_forest", {}))
    methods = _parse_tree_methods(tree_raw.pop("methods", None))
    if tree_raw:
        unknown = ", ".join(sorted(tree_raw))
        raise ValueError(f"Unsupported tree config fields: {unknown}")
    tree = TreeConfig(random_forest=random_forest, methods=methods)
    if random_forest.n_estimators <= 0:
        raise ValueError("tree.random_forest.n_estimators must be positive")
    if random_forest.min_samples_split < 2:
        raise ValueError("tree.random_forest.min_samples_split must be at least 2")
    if random_forest.min_samples_leaf <= 0:
        raise ValueError("tree.random_forest.min_samples_leaf must be positive")
    if not tree.enabled_modes():
        raise ValueError("tree.methods must enable at least one mode")
    for mode in tree.enabled_modes():
        tree.for_mode(mode)
    return tree


def _parse_tree_methods(raw_methods: Any) -> TreeMethodsConfig:
    """Parse enabled tree distillation modes."""

    mode_names: tuple[TreeDistillationMode, ...] = ("logits", "softmax")
    if raw_methods is None:
        methods_raw: dict[str, Any] = {name: {} for name in mode_names}
    elif isinstance(raw_methods, list):
        methods_raw = {mode: {} for mode in raw_methods}
    elif isinstance(raw_methods, dict):
        methods_raw = raw_methods.copy()
    else:
        raise ValueError("tree.methods must be a mapping from mode name to config")

    unknown_modes = [name for name in methods_raw if name not in mode_names]
    if unknown_modes:
        unknown = ", ".join(sorted(unknown_modes))
        raise ValueError(f"Unsupported tree methods: {unknown}")

    return TreeMethodsConfig(
        **{
            mode: _parse_tree_method_config(mode, methods_raw.get(mode))
            for mode in mode_names
        }
    )


def _parse_tree_method_config(
    mode: TreeDistillationMode,
    raw: Any,
) -> TreeMethodConfig | None:
    """Parse settings for one tree distillation mode."""

    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"tree.methods.{mode} must be a mapping")
    return TreeMethodConfig(**raw)


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


def _parse_training_config(raw: dict[str, Any], require_methods: bool = True) -> TrainingConfig:
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
    if require_methods and not training.enabled_methods():
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


def _resolve_tree_method_config(
    mode: TreeDistillationMode,
    method_config: TreeMethodConfig,
) -> ResolvedTreeMethodConfig:
    """Resolve and validate settings for one tree distillation mode."""

    if mode == "softmax":
        temperature = method_config.temperature if method_config.temperature is not None else 1.0
        alpha = method_config.alpha if method_config.alpha is not None else 1.0
        if temperature <= 0.0:
            raise ValueError("tree.methods.softmax.temperature must be positive")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("tree.methods.softmax.alpha must be between 0.0 and 1.0")
        return ResolvedTreeMethodConfig(temperature=temperature, alpha=alpha)
    if method_config.temperature is not None:
        raise ValueError("tree.methods.logits.temperature is not supported")
    if method_config.alpha is not None:
        raise ValueError("tree.methods.logits.alpha is not supported")
    return ResolvedTreeMethodConfig()


def _normalize_method_name(method: str) -> DistillationMethod:
    """Map legacy method names to the current vocabulary."""

    return LEGACY_METHOD_ALIASES.get(method, method)  # type: ignore[return-value]
