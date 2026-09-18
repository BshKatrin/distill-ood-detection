"""Run logit and probability inference for ID and configured OOD datasets."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import torch

from distill_ood_detection.config import (
    DistillationMethod,
    ExperimentConfig,
    TreeDistillationMode,
)
from distill_ood_detection.datasets.inference import (
    build_in_distribution_test_loader,
    build_in_distribution_train_loader,
    build_in_distribution_validation_loader,
    build_ood_loaders,
    dataset_normalization,
)
from distill_ood_detection.datasets.pixmix import build_pixmix_mixing_provider
from distill_ood_detection.distillation.perturbation import (
    PcaProjector,
    load_pca_projector,
    pca_projector_path,
)
from distill_ood_detection.inference import (
    collect_feature_model_outputs,
    collect_feature_tree_model_outputs,
    collect_model_outputs,
    collect_perturbed_teacher_outputs,
    collect_perturbation_model_outputs,
    collect_perturbation_tree_model_outputs,
    collect_tree_model_outputs,
    save_model_outputs,
)
from distill_ood_detection.models.student import build_student
from distill_ood_detection.models.teacher import (
    ResNetFeatureForwarder,
    TeacherFeatureExtractor,
    load_teacher,
)
from distill_ood_detection.utils import resolve_device, set_seed, write_json

CheckpointSelection = Literal["best", "latest", "both"]


def run_probability_inference(
    config: ExperimentConfig,
    checkpoint: CheckpointSelection = "best",
    method: DistillationMethod | None = None,
    tree_mode: TreeDistillationMode | None = None,
    include_train: bool = False,
    include_validation: bool = False,
    apply_perturbation: bool = False,
    output_dir: Path | None = None,
) -> dict[str, object]:
    """Infer teacher and student logits/probabilities for ID and OOD datasets."""

    training_defaults = config.training.defaults
    set_seed(training_defaults.seed)
    device = resolve_device(training_defaults.device)
    experiment_dir = Path(config.run_dir)
    if output_dir is None:
        output_dir = _probability_output_dir(
            experiment_dir, config, apply_perturbation
        )
    output_dir.mkdir(parents=True, exist_ok=True)

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
    image_normalization = dataset_normalization(config.dataset)
    pca_projector = None
    if (
        config.strategy.name == "perturbation"
        and config.strategy.perturbation.method
        in {"pca_projection", "pca_masked_projection"}
    ):
        pca_projector = load_pca_projector(pca_projector_path(experiment_dir), device)
    if config.strategy.name == "perturbation":
        feature_layer = (
            "layer4"
            if config.strategy.perturbation.method == "clipping"
            else config.student.feature_layer
        )
        if feature_layer is None:
            raise ValueError("student.feature_layer is required for this perturbation")
        perturbation_forwarder = ResNetFeatureForwarder(teacher, feature_layer)
    else:
        perturbation_forwarder = None
    feature_extractor = (
        TeacherFeatureExtractor(teacher, config.student.feature_layer)
        if config.student.feature_layer is not None and perturbation_forwarder is None
        else None
    )
    if config.student.kind == "random_forest" and method is not None:
        raise ValueError("--method is only supported for PyTorch students; use --tree-mode instead")
    if config.student.kind != "random_forest" and tree_mode is not None:
        raise ValueError("--tree-mode is only supported for random-forest students")
    if (
        config.student.kind == "random_forest"
        and config.strategy.name == "perturbation"
        and config.strategy.perturbation.method in {"pixel_augmentation", "pixmix"}
    ):
        raise ValueError("pixel-space perturbations do not support random-forest inference")

    artifacts: list[dict[str, object]] = []
    for named_loader in loaders:
        dataset_dir = output_dir / named_loader.name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        teacher_path = dataset_dir / "teacher.pt"
        teacher_metadata = {
            "model": "teacher",
            "dataset": named_loader.name,
            "split": named_loader.split,
        }
        if perturbation_forwarder is not None:
            teacher_metadata["strategy"] = config.strategy.name
            if config.student.feature_layer is not None:
                teacher_metadata["feature_layer"] = config.student.feature_layer
            teacher_metadata["perturbation"] = asdict(config.strategy.perturbation)
            teacher_metadata["apply_perturbation"] = apply_perturbation
            set_seed(training_defaults.seed)
        save_model_outputs(
            path=teacher_path,
            outputs=(
                collect_perturbed_teacher_outputs(
                    perturbation_forwarder,
                    config.strategy.perturbation,
                    named_loader.loader,
                    device,
                    apply_perturbation=apply_perturbation,
                    pca_projector=pca_projector,
                    image_normalization=image_normalization,
                    pixmix_provider=(
                        build_pixmix_mixing_provider(
                            config.dataset,
                            config.strategy.perturbation.pixmix,
                            training_defaults.seed,
                        )
                        if (
                            config.strategy.perturbation.method == "pixmix"
                            and apply_perturbation
                        )
                        else None
                    ),
                )
                if perturbation_forwarder is not None
                else collect_model_outputs(teacher, named_loader.loader, device)
            ),
            metadata=teacher_metadata,
        )
        artifacts.append({"dataset": named_loader.name, "model": "teacher", "path": str(teacher_path)})

        if config.student.kind == "random_forest":
            artifacts.extend(
                _infer_tree_students(
                    config=config,
                    experiment_dir=experiment_dir,
                    dataset_dir=dataset_dir,
                    dataset_name=named_loader.name,
                    split=named_loader.split,
                    loader=named_loader.loader,
                    checkpoint=checkpoint,
                    tree_mode=tree_mode,
                    device=device,
                    feature_extractor=feature_extractor,
                    perturbation_forwarder=perturbation_forwarder,
                    apply_perturbation=apply_perturbation,
                    pca_projector=pca_projector,
                )
            )
        else:
            artifacts.extend(
                _infer_torch_students(
                    config=config,
                    experiment_dir=experiment_dir,
                    dataset_dir=dataset_dir,
                    dataset_name=named_loader.name,
                    split=named_loader.split,
                    loader=named_loader.loader,
                    device=device,
                    checkpoint=checkpoint,
                    method=method,
                    feature_extractor=feature_extractor,
                    perturbation_forwarder=perturbation_forwarder,
                    apply_perturbation=apply_perturbation,
                    pca_projector=pca_projector,
                    image_normalization=image_normalization,
                )
            )

    manifest = {
        "experiment_name": config.experiment_name,
        "dataset": asdict(config.dataset),
        "teacher": asdict(config.teacher),
        "checkpoint_selection": checkpoint,
        "apply_perturbation": apply_perturbation,
        "artifacts": artifacts,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def _infer_torch_students(
    config: ExperimentConfig,
    experiment_dir: Path,
    dataset_dir: Path,
    dataset_name: str,
    split: str,
    loader: torch.utils.data.DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
    checkpoint: CheckpointSelection,
    method: DistillationMethod | None,
    feature_extractor: TeacherFeatureExtractor | None,
    perturbation_forwarder: ResNetFeatureForwarder | None,
    apply_perturbation: bool,
    pca_projector: PcaProjector | None,
    image_normalization: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> list[dict[str, object]]:
    artifacts: list[dict[str, object]] = []
    methods = (method,) if method else config.training.enabled_methods()
    for current_method in methods:
        for checkpoint_name in _selected_checkpoints(checkpoint):
            checkpoint_path = _student_checkpoint_path(
                experiment_dir=experiment_dir,
                method=current_method,
                checkpoint=checkpoint_name,
            )
            student = build_student(config.student)
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
            student.load_state_dict(state_dict)
            student.to(device)
            student.eval()
            probability_path = dataset_dir / f"student_{current_method}_{checkpoint_name}.pt"
            metadata = {
                "model": "student",
                "student_kind": config.student.kind,
                "method": current_method,
                "checkpoint": checkpoint_name,
                "checkpoint_path": str(checkpoint_path),
                "dataset": dataset_name,
                "split": split,
            }
            if config.student.feature_layer is not None:
                metadata["feature_layer"] = config.student.feature_layer
            if perturbation_forwarder is not None:
                metadata["strategy"] = config.strategy.name
                metadata["perturbation"] = asdict(config.strategy.perturbation)
                metadata["apply_perturbation"] = apply_perturbation
                set_seed(config.training.defaults.seed)
            save_model_outputs(
                path=probability_path,
                outputs=(
                    collect_perturbation_model_outputs(
                        student,
                        perturbation_forwarder,
                        config.strategy.perturbation,
                        loader,
                        device,
                        apply_perturbation=apply_perturbation,
                        pca_projector=pca_projector,
                        image_normalization=image_normalization,
                        pixmix_provider=(
                            build_pixmix_mixing_provider(
                                config.dataset,
                                config.strategy.perturbation.pixmix,
                                config.training.defaults.seed,
                            )
                            if (
                                config.strategy.perturbation.method == "pixmix"
                                and apply_perturbation
                            )
                            else None
                        ),
                    )
                    if perturbation_forwarder is not None
                    else collect_model_outputs(student, loader, device)
                    if feature_extractor is None
                    else collect_feature_model_outputs(
                        student,
                        feature_extractor,
                        loader,
                        device,
                    )
                ),
                metadata=metadata,
            )
            artifact = {
                "dataset": dataset_name,
                "model": "student",
                "student_kind": config.student.kind,
                "method": current_method,
                "checkpoint": checkpoint_name,
                "path": str(probability_path),
            }
            if config.student.feature_layer is not None:
                artifact["feature_layer"] = config.student.feature_layer
            if perturbation_forwarder is not None:
                artifact["strategy"] = config.strategy.name
            artifacts.append(artifact)
    return artifacts


def _infer_tree_students(
    config: ExperimentConfig,
    experiment_dir: Path,
    dataset_dir: Path,
    dataset_name: str,
    split: str,
    loader: torch.utils.data.DataLoader[tuple[torch.Tensor, int]],
    checkpoint: CheckpointSelection,
    tree_mode: TreeDistillationMode | None,
    device: torch.device,
    feature_extractor: TeacherFeatureExtractor | None,
    perturbation_forwarder: ResNetFeatureForwarder | None,
    apply_perturbation: bool,
    pca_projector: PcaProjector | None,
) -> list[dict[str, object]]:
    artifacts: list[dict[str, object]] = []
    modes = (tree_mode,) if tree_mode else config.tree.enabled_modes()
    for current_mode in modes:
        for checkpoint_name in _selected_checkpoints(checkpoint):
            checkpoint_path = _tree_student_checkpoint_path(
                experiment_dir=experiment_dir,
                mode=current_mode,
                checkpoint=checkpoint_name,
            )
            model, checkpoint_mode = _load_tree_student(checkpoint_path)
            if checkpoint_mode != current_mode:
                raise ValueError(
                    f"Tree checkpoint {checkpoint_path} was trained with mode "
                    f"{checkpoint_mode!r}, expected {current_mode!r}"
                )
            probability_path = dataset_dir / f"student_{current_mode}_{checkpoint_name}.pt"
            metadata = {
                "model": "student",
                "student_kind": config.student.kind,
                "mode": current_mode,
                "checkpoint": checkpoint_name,
                "checkpoint_path": str(checkpoint_path),
                "dataset": dataset_name,
                "split": split,
            }
            if feature_extractor is not None:
                metadata["feature_layer"] = config.student.feature_layer
            if perturbation_forwarder is not None:
                metadata["strategy"] = config.strategy.name
                metadata["feature_layer"] = config.student.feature_layer
                metadata["perturbation"] = asdict(config.strategy.perturbation)
                metadata["apply_perturbation"] = apply_perturbation
                set_seed(config.training.defaults.seed)
            save_model_outputs(
                path=probability_path,
                outputs=(
                    collect_perturbation_tree_model_outputs(
                        model,
                        current_mode,
                        perturbation_forwarder,
                        config.strategy.perturbation,
                        loader,
                        device,
                        apply_perturbation=apply_perturbation,
                        pca_projector=pca_projector,
                    )
                    if perturbation_forwarder is not None
                    else collect_feature_tree_model_outputs(
                        model,
                        current_mode,
                        feature_extractor,
                        loader,
                        device,
                    )
                    if feature_extractor is not None
                    else collect_tree_model_outputs(model, current_mode, loader)
                ),
                metadata=metadata,
            )
            artifact = {
                "dataset": dataset_name,
                "model": "student",
                "student_kind": config.student.kind,
                "mode": current_mode,
                "checkpoint": checkpoint_name,
                "path": str(probability_path),
            }
            if perturbation_forwarder is not None:
                artifact["strategy"] = config.strategy.name
                artifact["feature_layer"] = config.student.feature_layer
            elif feature_extractor is not None:
                artifact["feature_layer"] = config.student.feature_layer
            artifacts.append(artifact)
    return artifacts


def _selected_checkpoints(checkpoint: CheckpointSelection) -> tuple[Literal["best", "latest"], ...]:
    if checkpoint == "both":
        return ("best", "latest")
    return (checkpoint,)


def _probability_output_dir(
    experiment_dir: Path,
    config: ExperimentConfig,
    apply_perturbation: bool,
) -> Path:
    if config.strategy.name != "perturbation":
        return experiment_dir / "probabilities"
    mode_name = "perturbed" if apply_perturbation else "unperturbed"
    return experiment_dir / "probabilities" / mode_name


def _student_checkpoint_path(
    experiment_dir: Path,
    method: DistillationMethod,
    checkpoint: Literal["best", "latest"],
) -> Path:
    metrics_path = experiment_dir / method / "metrics.json"
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


def _tree_student_checkpoint_path(
    experiment_dir: Path,
    mode: TreeDistillationMode,
    checkpoint: Literal["best", "latest"],
) -> Path:
    metrics_path = experiment_dir / mode / "metrics.json"
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
        raise FileNotFoundError(f"Tree student checkpoint not found: {path}")
    return path


def _load_tree_student(path: Path) -> tuple[object, TreeDistillationMode]:
    with path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Tree checkpoint must contain a dictionary: {path}")
    model = checkpoint.get("model")
    mode = checkpoint.get("mode")
    if mode != "logits":
        raise ValueError(f"Tree checkpoint has invalid mode {mode!r}: {path}")
    if model is None:
        raise ValueError(f"Tree checkpoint is missing model: {path}")
    return model, mode
