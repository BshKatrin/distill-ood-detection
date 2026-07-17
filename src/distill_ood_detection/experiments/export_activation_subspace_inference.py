"""Export ID/OOD teacher embeddings and activation-subspace student outputs."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Literal

import torch
from torch import nn

from distill_ood_detection.config import (
    ActivationSubspaceComponent,
    ActivationSubspaceTarget,
    ExperimentConfig,
)
from distill_ood_detection.datasets.inference import (
    build_in_distribution_test_loader,
    build_ood_loaders,
)
from distill_ood_detection.distillation.activation_subspace import (
    PooledEmbeddingSplit,
    collect_clean_pooled_embeddings,
)
from distill_ood_detection.models.student import build_student
from distill_ood_detection.models.teacher import build_feature_forwarder, load_teacher
from distill_ood_detection.utils import resolve_device, set_seed, write_json

CheckpointSelection = Literal["best", "latest"]


def run_activation_subspace_inference_export(
    configs: list[ExperimentConfig],
    teacher_output_dir: Path,
    checkpoint: CheckpointSelection = "best",
) -> dict[str, object]:
    """Export shared ID/OOD embeddings and outputs for several students."""

    _validate_configs(configs)
    base_config = configs[0]
    training_defaults = base_config.training.defaults
    set_seed(training_defaults.seed)
    device = resolve_device(training_defaults.device)
    teacher_output_dir.mkdir(parents=True, exist_ok=True)

    teacher = load_teacher(base_config.teacher, device)
    forwarder = build_feature_forwarder(
        teacher,
        base_config.student.feature_layer,
    )
    forwarder.to(device)
    forwarder.eval()

    embedding_splits: dict[str, PooledEmbeddingSplit] = {}
    embedding_paths: dict[str, Path] = {}
    teacher_artifacts: list[dict[str, object]] = []
    loaders = [
        build_in_distribution_test_loader(base_config.dataset),
        *build_ood_loaders(base_config.dataset),
    ]
    for named_loader in loaders:
        split = collect_clean_pooled_embeddings(
            forwarder=forwarder,
            loader=named_loader.loader,
            device=device,
        )
        path = teacher_output_dir / named_loader.name / "embeddings.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "embeddings": split.embeddings,
                "labels": split.labels,
                "metadata": {
                    "model": "teacher",
                    "teacher": asdict(base_config.teacher),
                    "id_dataset": base_config.dataset.name,
                    "dataset": named_loader.name,
                    "split": named_loader.split,
                    "feature_layer": base_config.student.feature_layer,
                    "pooling": base_config.strategy.activation_subspace.embedding_pool,
                    "embedding_shape": list(split.embeddings.shape[1:]),
                },
            },
            path,
        )
        embedding_splits[named_loader.name] = split
        embedding_paths[named_loader.name] = path
        teacher_artifacts.append(
            {
                "dataset": named_loader.name,
                "split": named_loader.split,
                "samples": split.embeddings.shape[0],
                "embedding_dimension": split.embeddings.shape[1],
                "path": str(path),
            }
        )

    teacher_manifest = {
        "version": 1,
        "artifact_type": "activation_subspace_teacher_embeddings",
        "id_dataset": base_config.dataset.name,
        "teacher": asdict(base_config.teacher),
        "feature_layer": base_config.student.feature_layer,
        "pooling": base_config.strategy.activation_subspace.embedding_pool,
        "artifacts": teacher_artifacts,
    }
    teacher_manifest_path = teacher_output_dir / "manifest.json"
    write_json(teacher_manifest_path, teacher_manifest)

    student_manifests = []
    for config in configs:
        student_manifests.append(
            _export_student_outputs(
                config=config,
                embedding_splits=embedding_splits,
                embedding_paths=embedding_paths,
                checkpoint=checkpoint,
                device=device,
            )
        )

    group_manifest = {
        "version": 1,
        "artifact_type": "activation_subspace_ood_inference",
        "checkpoint_selection": checkpoint,
        "teacher_manifest_path": str(teacher_manifest_path),
        "student_manifests": student_manifests,
    }
    write_json(teacher_output_dir / "student_inference_manifest.json", group_manifest)
    return group_manifest


def infer_activation_subspace_student(
    *,
    student: nn.Module,
    embeddings: torch.Tensor,
    right_basis: torch.Tensor,
    decisive_dimension: int,
    component: ActivationSubspaceComponent,
    target: ActivationSubspaceTarget,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Infer raw outputs from fixed pooled embeddings in the selected coordinates."""

    if component == "decisive":
        component_basis = right_basis[:decisive_dimension]
    else:
        component_basis = right_basis[decisive_dimension:]
    component_basis = component_basis.to(device)
    outputs = []
    student.to(device)
    student.eval()
    with torch.no_grad():
        for start in range(0, embeddings.shape[0], batch_size):
            batch = embeddings[start : start + batch_size].to(device)
            coordinates = batch @ component_basis.T
            outputs.append(student(coordinates).cpu())
    output_tensor = torch.cat(outputs, dim=0)
    if target == "projected_logits":
        return {
            "logits": output_tensor,
            "probabilities": torch.softmax(output_tensor, dim=1),
        }
    return {"reconstructed_coordinates": output_tensor}


def _export_student_outputs(
    *,
    config: ExperimentConfig,
    embedding_splits: dict[str, PooledEmbeddingSplit],
    embedding_paths: dict[str, Path],
    checkpoint: CheckpointSelection,
    device: torch.device,
) -> dict[str, object]:
    experiment_dir = Path(config.run_dir)
    subspace_config = config.strategy.activation_subspace
    component = subspace_config.component
    target = subspace_config.target
    checkpoint_path = experiment_dir / component / f"{checkpoint}_student.pt"
    subspace_path = experiment_dir / "activation_subspace.pt"
    subspace = torch.load(subspace_path, map_location="cpu", weights_only=True)
    right_basis = subspace["right_basis"]
    decisive_dimension = int(subspace["decisive_dimension"])
    component_dimension = (
        decisive_dimension
        if component == "decisive"
        else right_basis.shape[0] - decisive_dimension
    )
    if component_dimension != config.strategy.activation_subspace.expected_dimension:
        raise ValueError(
            "Saved activation-subspace dimension differs from config: "
            f"{component_dimension} != "
            f"{config.strategy.activation_subspace.expected_dimension}"
        )

    student = build_student(config.student)
    student.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    enabled_methods = config.training.enabled_methods()
    if len(enabled_methods) != 1:
        raise ValueError(
            "Activation-subspace inference requires exactly one enabled method"
        )
    method = enabled_methods[0]
    output_dir = experiment_dir / "activation_subspace_inference"
    artifacts = []
    for dataset_name, split in embedding_splits.items():
        outputs = infer_activation_subspace_student(
            student=student,
            embeddings=split.embeddings,
            right_basis=right_basis,
            decisive_dimension=decisive_dimension,
            component=component,
            target=target,
            batch_size=config.dataset.batch_size,
            device=device,
        )
        path = output_dir / dataset_name / f"student_{checkpoint}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                **outputs,
                "labels": split.labels,
                "metadata": {
                    "model": "student",
                    "experiment_name": config.experiment_name,
                    "id_dataset": config.dataset.name,
                    "dataset": dataset_name,
                    "component": component,
                    "target": target,
                    "distillation_method": method,
                    "checkpoint": checkpoint,
                    "checkpoint_path": str(checkpoint_path),
                    "subspace_path": str(subspace_path),
                    "teacher_embedding_path": str(embedding_paths[dataset_name]),
                    "decisive_dimension": decisive_dimension,
                    "insignificant_dimension": right_basis.shape[0]
                    - decisive_dimension,
                    "output_kind": (
                        "logits_and_probabilities"
                        if target == "projected_logits"
                        else f"reconstructed_{component}_coordinates"
                    ),
                },
            },
            path,
        )
        artifacts.append(
            {
                "dataset": dataset_name,
                "samples": split.embeddings.shape[0],
                "output_shape": list(next(iter(outputs.values())).shape[1:]),
                "path": str(path),
            }
        )

    manifest = {
        "version": 1,
        "artifact_type": "activation_subspace_student_inference",
        "experiment_name": config.experiment_name,
        "run_dir": config.run_dir,
        "component": component,
        "target": target,
        "distillation_method": method,
        "checkpoint_selection": checkpoint,
        "checkpoint_path": str(checkpoint_path),
        "subspace_path": str(subspace_path),
        "artifacts": artifacts,
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    return {
        "experiment_name": config.experiment_name,
        "component": component,
        "target": target,
        "distillation_method": method,
        "manifest_path": str(manifest_path),
    }


def _validate_configs(configs: list[ExperimentConfig]) -> None:
    if not configs:
        raise ValueError("At least one activation-subspace config is required")
    base = configs[0]
    for config in configs:
        if config.strategy.name != "activation_subspace":
            raise ValueError(
                "Activation-subspace inference requires strategy.name: "
                "activation_subspace"
            )
        if config.dataset != base.dataset or config.teacher != base.teacher:
            raise ValueError(
                "All configs in one inference export must share dataset and teacher"
            )
        if config.student.feature_layer != base.student.feature_layer:
            raise ValueError("All configs must use the same teacher feature layer")
        if config.strategy.activation_subspace.embedding_pool != "avg":
            raise ValueError("Activation-subspace inference currently requires avg pooling")
