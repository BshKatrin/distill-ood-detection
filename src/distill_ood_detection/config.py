"""YAML-backed experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from distill_ood_detection.utils import get_data_dir_override


DISTILLATION_METHODS: tuple[str, ...] = (
    "cross_entropy",
    "mse_logits",
    "kl_divergence",
)
DistillationMethod = Literal[
    "cross_entropy",
    "mse_logits",
    "kl_divergence",
]
FeatureDenoisingMethod = Literal[
    "pca_masked_reconstruction",
    "spatial_masked_reconstruction",
    "channel_masked_reconstruction",
    "spatial_token_prediction",
    "pixel_masked_embedding_prediction",
    "pixel_augmented_embedding_prediction",
    "pixel_masked_multilayer_prediction",
    "pixel_masked_multilayer_l234_prediction",
]
EmbeddingPool = Literal["avg", "flatten"]
PixelAugmentationMethod = Literal["affine", "pixmix"]
TreeDistillationMode = Literal["logits"]
OODDatasetName = Literal["cifar10", "cifar100", "mnist", "svhn"]
StrategyName = Literal[
    "baseline",
    "perturbation",
    "feature_denoising",
    "activation_subspace",
]
ActivationSubspaceComponent = Literal["decisive", "insignificant"]
ActivationSubspaceTarget = Literal["projected_logits", "coordinates"]
PerturbationMethod = Literal[
    "clipping",
    "mc_dropout",
    "pca_projection",
    "pca_masked_projection",
    "pixel_augmentation",
    "pixmix",
]
TeacherTarget = Literal["clean", "perturbed"]
ClippingMode = Literal["constant", "spatial_dependent", "channel_dependent"]
DropoutMode = Literal["element", "channel", "spatial"]
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
    pre_size: int | None = None
    image_size: int | None = None
    normalization: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None


@dataclass(frozen=True)
class TeacherConfig:
    """Teacher model settings."""

    hf_model_id: str = "edadaltocg/resnet18_cifar10"
    revision: str = "main"
    num_classes: int = 10


@dataclass(frozen=True)
class StudentConfig:
    """Student model settings."""

    kind: str = "linear"
    input_shape: tuple[int, ...] = (3, 32, 32)
    num_classes: int = 10
    hidden_channels: tuple[int, ...] = field(default_factory=tuple)
    feature_layer: str | None = None


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
class PixMixConfig:
    """PixMix pixel-corruption settings."""

    mixing_set_path: str | None = None
    mixing_iterations: int = 4
    beta: float = 3.0
    augmentation_severity: float = 3.0
    all_ops: bool = True
    working_size: int = 32


@dataclass(frozen=True)
class ClippingLayerConfig:
    """Clipping settings for one ResNet feature layer."""

    clipping_mode: ClippingMode
    u_min: float
    u_max: float


@dataclass(frozen=True)
class PerturbationConfig:
    """Perturbation settings for stochastic distillation."""

    method: PerturbationMethod = "clipping"
    teacher_target: TeacherTarget = "clean"
    clipping_layers: dict[str, ClippingLayerConfig] = field(default_factory=dict)
    dropout_probability: float = 0.5
    dropout_mode: DropoutMode = "element"
    pca_components: int = 128
    pca_mask_probability: float = 0.3
    pca_activation_path: str | None = None
    evaluation_draws: int = 1
    rotation_degrees: float = 10.0
    translate_fraction: float = 0.10
    scale_min: float = 0.90
    scale_max: float = 1.10
    brightness_delta: float = 0.10
    contrast_delta: float = 0.20
    embedding_pool: EmbeddingPool = "flatten"
    pixmix: PixMixConfig = field(default_factory=PixMixConfig)


@dataclass(frozen=True)
class FeatureDenoisingConfig:
    """Feature Denoising representation-prediction settings."""

    method: FeatureDenoisingMethod = "pca_masked_reconstruction"
    activation_path: str | None = None
    mask_probability: float = 0.3
    target_block_count: int = 4
    target_block_scale_min: float = 0.15
    target_block_scale_max: float = 0.20
    target_aspect_ratio_min: float = 0.75
    target_aspect_ratio_max: float = 1.50
    context_scale_min: float = 0.85
    context_scale_max: float = 1.00
    target_token_count: int = 0
    image_mask_block_count: int = 2
    image_mask_scale_min: float = 0.15
    image_mask_scale_max: float = 0.20
    image_mask_aspect_ratio_min: float = 0.75
    image_mask_aspect_ratio_max: float = 1.50
    embedding_pool: str = "avg"
    rotation_degrees: float = 10.0
    translate_fraction: float = 0.10
    scale_min: float = 0.90
    scale_max: float = 1.10
    brightness_delta: float = 0.10
    contrast_delta: float = 0.20
    pca_components: int = 128
    pca_mask_probability: float = 0.3
    pca_activation_path: str | None = None
    evaluation_draws: int = 1
    pixel_augmentation_method: PixelAugmentationMethod = "affine"
    pixmix: PixMixConfig = field(default_factory=PixMixConfig)


@dataclass(frozen=True)
class ActivationSubspaceConfig:
    """Classifier-SVD activation-subspace student settings."""

    component: ActivationSubspaceComponent = "decisive"
    target: ActivationSubspaceTarget = "projected_logits"
    expected_dimension: int = 1
    embedding_pool: str = "avg"


@dataclass(frozen=True)
class StrategyConfig:
    """Distillation strategy settings."""

    name: StrategyName = "baseline"
    perturbation: PerturbationConfig = field(default_factory=PerturbationConfig)
    feature_denoising: FeatureDenoisingConfig = field(default_factory=FeatureDenoisingConfig)
    activation_subspace: ActivationSubspaceConfig = field(
        default_factory=ActivationSubspaceConfig
    )


@dataclass(frozen=True)
class TreeMethodConfig:
    """Method-specific settings for tree distillation."""

    temperature: float | None = None
    alpha: float | None = None


@dataclass(frozen=True)
class TreeMethodsConfig:
    """Enabled tree distillation modes."""

    logits: TreeMethodConfig | None = field(default_factory=TreeMethodConfig)

    def enabled_modes(self) -> tuple[TreeDistillationMode, ...]:
        """Return enabled tree distillation modes in a deterministic order."""

        return tuple(
            mode
            for mode in ("logits",)
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

    cross_entropy: OptimizerConfig = field(default_factory=OptimizerConfig)
    mse_logits: OptimizerConfig = field(default_factory=OptimizerConfig)
    kl_divergence: OptimizerConfig = field(default_factory=OptimizerConfig)
    pca_masked_reconstruction: OptimizerConfig = field(default_factory=OptimizerConfig)
    activation_subspace: OptimizerConfig = field(default_factory=OptimizerConfig)

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

    cross_entropy: MethodTrainingConfig | None = field(default_factory=MethodTrainingConfig)
    mse_logits: MethodTrainingConfig | None = field(default_factory=MethodTrainingConfig)
    kl_divergence: MethodTrainingConfig | None = None

    def enabled_methods(self) -> tuple[DistillationMethod, ...]:
        """Return enabled methods in a deterministic order."""

        return tuple(
            method
            for method in DISTILLATION_METHODS
            if getattr(self, method) is not None
        )  # type: ignore[return-value]


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
    run_dir: str
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    optimizer: OptimizerByMethodConfig = field(default_factory=OptimizerByMethodConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tree: TreeConfig = field(default_factory=TreeConfig)
    mlflow: MlflowConfig = field(default_factory=MlflowConfig)


@dataclass(frozen=True)
class TeacherActivationConfig:
    """Configuration for exporting raw teacher activations."""

    experiment_name: str
    run_dir: str
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    layers: tuple[str, ...] = ("layer3",)
    device: str = "auto"
    seed: int = 123


@dataclass(frozen=True)
class TeacherProbabilityConfig:
    """Configuration for exporting deterministic teacher probabilities."""

    experiment_name: str
    run_dir: str
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    device: str = "auto"
    seed: int = 123


def load_config(path: Path) -> ExperimentConfig:
    """Load an experiment configuration from a YAML file."""

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return parse_config(raw)


def load_teacher_activation_config(path: Path) -> TeacherActivationConfig:
    """Load a teacher activation export configuration from a YAML file."""

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return parse_teacher_activation_config(raw)


def load_teacher_probability_config(path: Path) -> TeacherProbabilityConfig:
    """Load a teacher probability export configuration from a YAML file."""

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return parse_teacher_probability_config(raw)


def parse_teacher_activation_config(raw: dict[str, Any]) -> TeacherActivationConfig:
    """Parse a raw dictionary into a teacher activation export config."""

    dataset = _parse_dataset_config(raw.get("dataset", {}))
    teacher = TeacherConfig(**raw.get("teacher", {}))
    layers = tuple(raw.get("layers", ("layer3",)))
    config = TeacherActivationConfig(
        experiment_name=raw["experiment_name"],
        run_dir=raw["run_dir"],
        dataset=dataset,
        teacher=teacher,
        layers=layers,
        device=raw.get("device", "auto"),
        seed=raw.get("seed", 123),
    )
    if teacher.num_classes <= 0:
        raise ValueError("teacher.num_classes must be positive")
    if not config.layers:
        raise ValueError("layers must contain at least one teacher layer")
    if any(not isinstance(layer, str) or not layer for layer in config.layers):
        raise ValueError("layers must contain non-empty layer names")
    return config


def parse_teacher_probability_config(raw: dict[str, Any]) -> TeacherProbabilityConfig:
    """Parse a raw dictionary into a teacher probability export config."""

    dataset = _parse_dataset_config(raw.get("dataset", {}))
    teacher = TeacherConfig(**raw.get("teacher", {}))
    config = TeacherProbabilityConfig(
        experiment_name=raw["experiment_name"],
        run_dir=raw["run_dir"],
        dataset=dataset,
        teacher=teacher,
        device=raw.get("device", "auto"),
        seed=raw.get("seed", 123),
    )
    if teacher.num_classes <= 0:
        raise ValueError("teacher.num_classes must be positive")
    return config


def parse_config(raw: dict[str, Any]) -> ExperimentConfig:
    """Parse a raw dictionary into typed configuration objects."""

    dataset = _parse_dataset_config(raw.get("dataset", {}))
    teacher = TeacherConfig(**raw.get("teacher", {}))
    if teacher.num_classes <= 0:
        raise ValueError("teacher.num_classes must be positive")
    strategy = _parse_strategy_config(raw.get("strategy", {}))
    student_raw = raw.get("student", {}).copy()
    if "input_shape" in student_raw:
        student_raw["input_shape"] = tuple(student_raw["input_shape"])
    if "hidden_channels" in student_raw:
        student_raw["hidden_channels"] = tuple(student_raw["hidden_channels"])
    student = _parse_student_config(student_raw, strategy=strategy)
    if strategy.name == "activation_subspace":
        activation_subspace = strategy.activation_subspace
        if activation_subspace.target == "projected_logits" and (
            activation_subspace.component != "decisive"
            or student.kind != "linear"
        ):
            raise ValueError(
                "activation-subspace projected_logits targets require a "
                "decisive linear student"
            )
        if (
            activation_subspace.target == "coordinates"
            and student.kind != "autoencoder"
        ):
            raise ValueError(
                "activation-subspace coordinate targets require an autoencoder"
            )
    if (
        strategy.name
        in {"perturbation", "feature_denoising", "activation_subspace"}
        and student.feature_layer is None
        and not (
            strategy.name == "perturbation"
            and strategy.perturbation.method == "clipping"
        )
    ):
        raise ValueError(
            "student.feature_layer is required for perturbation, Feature Denoising, "
            "and activation-subspace strategies"
        )
    if (
        strategy.name == "perturbation"
        and strategy.perturbation.method in {"pixel_augmentation", "pixmix"}
        and student.kind == "random_forest"
    ):
        raise ValueError("pixel-space perturbations support only linear and MLP students")
    optimizer = _parse_optimizer_config(raw.get("optimizer", {}))
    training = _parse_training_config(
        raw.get("training", {}),
        require_methods=(
            student.kind != "random_forest"
            and strategy.name not in {"feature_denoising", "activation_subspace"}
        ),
    )
    tree = _parse_tree_config(raw.get("tree", {}))
    mlflow = MlflowConfig(**raw.get("mlflow", {}))

    return ExperimentConfig(
        experiment_name=raw["experiment_name"],
        run_dir=raw["run_dir"],
        dataset=dataset,
        teacher=teacher,
        student=student,
        strategy=strategy,
        optimizer=optimizer,
        training=training,
        tree=tree,
        mlflow=mlflow,
    )


def _parse_dataset_config(raw: dict[str, Any]) -> DatasetConfig:
    """Parse dataset settings and apply local environment overrides."""

    dataset_raw = raw.copy()
    if "ood_datasets" in dataset_raw:
        dataset_raw["ood_datasets"] = tuple(
            OODDatasetConfig(**item) for item in dataset_raw["ood_datasets"]
        )
    if "normalization" in dataset_raw and dataset_raw["normalization"] is not None:
        mean, std = dataset_raw["normalization"]
        dataset_raw["normalization"] = (tuple(mean), tuple(std))
    data_dir_override = get_data_dir_override()
    if data_dir_override:
        dataset_raw["data_dir"] = data_dir_override
    return DatasetConfig(**dataset_raw)


def _parse_student_config(raw: dict[str, Any], strategy: StrategyConfig) -> StudentConfig:
    """Parse neural-network student settings."""

    student = StudentConfig(**raw)
    if student.kind not in {
        "linear",
        "mlp",
        "random_forest",
        "feature_reconstructor",
        "spatial_token_predictor",
        "autoencoder",
    }:
        raise ValueError(f"Unsupported student kind: {student.kind}")
    if not student.input_shape:
        raise ValueError("student.input_shape must contain at least one dimension")
    if strategy.name == "baseline" and len(student.input_shape) != 3:
        raise ValueError(
            "baseline student.input_shape must contain channel, height, and width"
        )
    if any(dimension <= 0 for dimension in student.input_shape):
        raise ValueError("student.input_shape dimensions must be positive")
    if student.num_classes <= 0:
        raise ValueError("student.num_classes must be positive")
    if student.kind == "mlp" and not student.hidden_channels:
        raise ValueError(
            "student.hidden_channels must define at least one hidden layer "
            "for MLP students"
        )
    if student.kind == "autoencoder" and not student.hidden_channels:
        raise ValueError(
            "student.hidden_channels must define the autoencoder bottleneck path"
        )
    if student.kind == "autoencoder" and len(student.input_shape) != 1:
        raise ValueError("autoencoder student.input_shape must be one-dimensional")
    if student.kind == "autoencoder" and student.num_classes != student.input_shape[0]:
        raise ValueError(
            "autoencoder student.num_classes must match its one-dimensional input"
        )
    if any(hidden_channel <= 0 for hidden_channel in student.hidden_channels):
        raise ValueError("student.hidden_channels values must be positive")
    return student


def _parse_strategy_config(raw: dict[str, Any]) -> StrategyConfig:
    """Parse distillation strategy settings."""

    strategy_raw = raw.copy()
    perturbation_raw = strategy_raw.pop("perturbation", {})
    feature_denoising_raw = strategy_raw.pop("feature_denoising", {})
    activation_subspace_raw = strategy_raw.pop("activation_subspace", {})
    strategy_name = strategy_raw.pop("name", "baseline")
    if (
        perturbation_raw.get("method") == "pixel_augmentation"
        and "teacher_target" not in perturbation_raw
    ):
        perturbation_raw["teacher_target"] = "perturbed"
    if perturbation_raw.get("method") == "pixmix" and "teacher_target" not in perturbation_raw:
        perturbation_raw["teacher_target"] = "perturbed"
    legacy_clipping_fields = {
        field_name
        for field_name in ("u_min", "u_max", "clipping_mode")
        if field_name in perturbation_raw
    }
    if legacy_clipping_fields:
        fields = ", ".join(sorted(legacy_clipping_fields))
        raise ValueError(
            "Clipping requires strategy.perturbation.clipping_layers; "
            f"unsupported legacy fields: {fields}"
        )
    clipping_layers_raw = perturbation_raw.pop("clipping_layers", {})
    if not isinstance(clipping_layers_raw, dict):
        raise ValueError("strategy.perturbation.clipping_layers must be a mapping")
    supported_clipping_layers = ("layer1", "layer2", "layer3", "layer4")
    unknown_clipping_layers = set(clipping_layers_raw) - set(supported_clipping_layers)
    if unknown_clipping_layers:
        unknown = ", ".join(sorted(unknown_clipping_layers))
        raise ValueError(f"Unsupported clipping layers: {unknown}")
    clipping_layers = {
        layer_name: ClippingLayerConfig(**clipping_layers_raw[layer_name])
        for layer_name in supported_clipping_layers
        if layer_name in clipping_layers_raw
    }
    perturbation_pixmix_raw = perturbation_raw.pop("pixmix", {})
    feature_denoising_pixmix_raw = feature_denoising_raw.pop("pixmix", {})
    perturbation_raw["pixmix"] = PixMixConfig(**perturbation_pixmix_raw)
    feature_denoising_raw["pixmix"] = PixMixConfig(**feature_denoising_pixmix_raw)
    perturbation = PerturbationConfig(
        **perturbation_raw,
        clipping_layers=clipping_layers,
    )
    feature_denoising = FeatureDenoisingConfig(**feature_denoising_raw)
    activation_subspace = ActivationSubspaceConfig(**activation_subspace_raw)
    strategy = StrategyConfig(
        name=strategy_name,
        perturbation=perturbation,
        feature_denoising=feature_denoising,
        activation_subspace=activation_subspace,
    )
    if strategy_raw:
        unknown = ", ".join(sorted(strategy_raw))
        raise ValueError(f"Unsupported strategy config fields: {unknown}")
    if strategy.name not in {
        "baseline",
        "perturbation",
        "feature_denoising",
        "activation_subspace",
    }:
        raise ValueError(f"Unsupported strategy: {strategy.name}")
    if perturbation.method not in {
        "clipping",
        "mc_dropout",
        "pca_projection",
        "pca_masked_projection",
        "pixel_augmentation",
        "pixmix",
    }:
        raise ValueError(f"Unsupported perturbation method: {perturbation.method}")
    if (
        strategy.name == "perturbation"
        and perturbation.method == "clipping"
        and not perturbation.clipping_layers
    ):
        raise ValueError(
            "strategy.perturbation.clipping_layers is required for clipping"
        )
    if (
        strategy.name == "perturbation"
        and perturbation.method != "clipping"
        and perturbation.clipping_layers
    ):
        raise ValueError(
            "strategy.perturbation.clipping_layers is supported only for clipping"
        )
    for layer_name, layer_config in perturbation.clipping_layers.items():
        if layer_config.clipping_mode not in {
            "constant",
            "spatial_dependent",
            "channel_dependent",
        }:
            raise ValueError(
                f"Unsupported clipping mode for {layer_name}: "
                f"{layer_config.clipping_mode}"
            )
        if not 0.0 <= layer_config.u_min < layer_config.u_max <= 1.0:
            raise ValueError(
                f"strategy.perturbation.clipping_layers.{layer_name} requires "
                "0 <= u_min < u_max <= 1"
            )
    if not 0.0 <= perturbation.dropout_probability < 1.0:
        raise ValueError(
            "strategy.perturbation.dropout_probability requires 0 <= p < 1"
        )
    if perturbation.dropout_mode not in {"element", "channel", "spatial"}:
        raise ValueError(f"Unsupported dropout mode: {perturbation.dropout_mode}")
    if perturbation.pca_components <= 0:
        raise ValueError("strategy.perturbation.pca_components must be positive")
    if not 0.0 <= perturbation.pca_mask_probability < 1.0:
        raise ValueError(
            "strategy.perturbation.pca_mask_probability requires 0 <= p < 1"
        )
    if perturbation.evaluation_draws <= 0:
        raise ValueError("strategy.perturbation.evaluation_draws must be positive")
    if perturbation.embedding_pool not in {"avg", "flatten"}:
        raise ValueError(
            "strategy.perturbation.embedding_pool must be 'avg' or 'flatten'"
        )
    if perturbation.rotation_degrees < 0.0:
        raise ValueError("strategy.perturbation.rotation_degrees must be non-negative")
    if not 0.0 <= perturbation.translate_fraction <= 1.0:
        raise ValueError(
            "strategy.perturbation.translate_fraction requires 0 <= fraction <= 1"
        )
    if perturbation.scale_min <= 0.0 or perturbation.scale_min > perturbation.scale_max:
        raise ValueError(
            "strategy.perturbation requires 0 < scale_min <= scale_max"
        )
    if not 0.0 <= perturbation.brightness_delta <= 1.0:
        raise ValueError(
            "strategy.perturbation.brightness_delta requires 0 <= delta <= 1"
        )
    if not 0.0 <= perturbation.contrast_delta <= 1.0:
        raise ValueError(
            "strategy.perturbation.contrast_delta requires 0 <= delta <= 1"
        )
    _validate_pixmix_config(perturbation.pixmix, "strategy.perturbation.pixmix")
    if feature_denoising.method not in {
        "pca_masked_reconstruction",
        "spatial_masked_reconstruction",
        "channel_masked_reconstruction",
        "spatial_token_prediction",
        "pixel_masked_embedding_prediction",
        "pixel_augmented_embedding_prediction",
        "pixel_masked_multilayer_prediction",
        "pixel_masked_multilayer_l234_prediction",
    }:
        raise ValueError(f"Unsupported Feature Denoising method: {feature_denoising.method}")
    if not 0.0 < feature_denoising.mask_probability < 1.0:
        raise ValueError("strategy.feature_denoising.mask_probability requires 0 < p < 1")
    if feature_denoising.target_block_count <= 0:
        raise ValueError("strategy.feature_denoising.target_block_count must be positive")
    if not 0.0 < feature_denoising.target_block_scale_min <= feature_denoising.target_block_scale_max <= 1.0:
        raise ValueError(
            "strategy.feature_denoising target block scale requires 0 < min <= max <= 1"
        )
    if (
        feature_denoising.target_aspect_ratio_min <= 0.0
        or feature_denoising.target_aspect_ratio_min > feature_denoising.target_aspect_ratio_max
    ):
        raise ValueError(
            "strategy.feature_denoising target aspect ratio requires 0 < min <= max"
        )
    if not 0.0 < feature_denoising.context_scale_min <= feature_denoising.context_scale_max <= 1.0:
        raise ValueError(
            "strategy.feature_denoising context scale requires 0 < min <= max <= 1"
        )
    if feature_denoising.target_token_count < 0:
        raise ValueError("strategy.feature_denoising.target_token_count must be non-negative")
    if feature_denoising.image_mask_block_count <= 0:
        raise ValueError("strategy.feature_denoising.image_mask_block_count must be positive")
    if not 0.0 < feature_denoising.image_mask_scale_min <= feature_denoising.image_mask_scale_max <= 1.0:
        raise ValueError(
            "strategy.feature_denoising image mask scale requires 0 < min <= max <= 1"
        )
    if (
        feature_denoising.image_mask_aspect_ratio_min <= 0.0
        or feature_denoising.image_mask_aspect_ratio_min > feature_denoising.image_mask_aspect_ratio_max
    ):
        raise ValueError(
            "strategy.feature_denoising image mask aspect ratio requires 0 < min <= max"
        )
    if feature_denoising.embedding_pool not in {"avg", "cls"}:
        raise ValueError("strategy.feature_denoising.embedding_pool must be 'avg' or 'cls'")
    if feature_denoising.rotation_degrees < 0.0:
        raise ValueError("strategy.feature_denoising.rotation_degrees must be non-negative")
    if not 0.0 <= feature_denoising.translate_fraction <= 1.0:
        raise ValueError(
            "strategy.feature_denoising.translate_fraction requires 0 <= fraction <= 1"
        )
    if feature_denoising.scale_min <= 0.0 or feature_denoising.scale_min > feature_denoising.scale_max:
        raise ValueError("strategy.feature_denoising requires 0 < scale_min <= scale_max")
    if not 0.0 <= feature_denoising.brightness_delta <= 1.0:
        raise ValueError("strategy.feature_denoising.brightness_delta requires 0 <= delta <= 1")
    if not 0.0 <= feature_denoising.contrast_delta <= 1.0:
        raise ValueError("strategy.feature_denoising.contrast_delta requires 0 <= delta <= 1")
    if feature_denoising.pca_components <= 0:
        raise ValueError("strategy.feature_denoising.pca_components must be positive")
    if not 0.0 < feature_denoising.pca_mask_probability < 1.0:
        raise ValueError("strategy.feature_denoising.pca_mask_probability requires 0 < p < 1")
    if feature_denoising.evaluation_draws <= 0:
        raise ValueError("strategy.feature_denoising.evaluation_draws must be positive")
    if feature_denoising.pixel_augmentation_method not in {"affine", "pixmix"}:
        raise ValueError(
            "strategy.feature_denoising.pixel_augmentation_method must be "
            "'affine' or 'pixmix'"
        )
    _validate_pixmix_config(
        feature_denoising.pixmix,
        "strategy.feature_denoising.pixmix",
    )
    if activation_subspace.component not in {"decisive", "insignificant"}:
        raise ValueError(
            "strategy.activation_subspace.component must be decisive or insignificant"
        )
    if activation_subspace.target not in {"projected_logits", "coordinates"}:
        raise ValueError(
            "strategy.activation_subspace.target must be projected_logits or coordinates"
        )
    if activation_subspace.expected_dimension <= 0:
        raise ValueError(
            "strategy.activation_subspace.expected_dimension must be positive"
        )
    if activation_subspace.embedding_pool != "avg":
        raise ValueError(
            "strategy.activation_subspace.embedding_pool currently supports only avg"
        )
    return strategy


def _validate_pixmix_config(config: PixMixConfig, prefix: str) -> None:
    """Validate shared PixMix configuration values."""

    if config.mixing_iterations < 0:
        raise ValueError(f"{prefix}.mixing_iterations must be non-negative")
    if config.beta <= 0.0:
        raise ValueError(f"{prefix}.beta must be positive")
    if not 0.1 <= config.augmentation_severity <= 10.0:
        raise ValueError(
            f"{prefix}.augmentation_severity requires 0.1 <= severity <= 10"
        )
    if config.working_size <= 0:
        raise ValueError(f"{prefix}.working_size must be positive")


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

    mode_names: tuple[TreeDistillationMode, ...] = ("logits",)
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

    method_names = (
        *DISTILLATION_METHODS,
        "pca_masked_reconstruction",
        "activation_subspace",
    )
    optimizer_raw = {
        _normalize_method_name(name) if name in LEGACY_METHOD_ALIASES else name: value
        for name, value in raw.copy().items()
    }
    if any(name in optimizer_raw for name in method_names):
        unknown_methods = [name for name in optimizer_raw if name not in method_names]
        if unknown_methods:
            unknown = ", ".join(sorted(unknown_methods))
            raise ValueError(f"Unsupported optimizer methods: {unknown}")
        default_optimizer = OptimizerConfig()
        return OptimizerByMethodConfig(
            **{
                name: (
                    OptimizerConfig(**optimizer_raw[name])
                    if name in optimizer_raw
                    else default_optimizer
                )
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

    method_names = DISTILLATION_METHODS
    if raw_methods is None:
        methods_raw: dict[str, Any] = {
            "cross_entropy": {},
            "mse_logits": {},
        }
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
    if method == "kl_divergence":
        if method_config.alpha is not None:
            raise ValueError("training.methods.kl_divergence.alpha is not supported")
        if method_config.temperature is not None:
            raise ValueError("training.methods.kl_divergence.temperature is not supported")
        return ResolvedTrainingMethodConfig(
            epochs=defaults.epochs,
            seed=defaults.seed,
            device=defaults.device,
            log_every_steps=defaults.log_every_steps,
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

    if method_config.temperature is not None:
        raise ValueError("tree.methods.logits.temperature is not supported")
    if method_config.alpha is not None:
        raise ValueError("tree.methods.logits.alpha is not supported")
    return ResolvedTreeMethodConfig()


def _normalize_method_name(method: str) -> DistillationMethod:
    """Map legacy method names to the current vocabulary."""

    return LEGACY_METHOD_ALIASES.get(method, method)  # type: ignore[return-value]
