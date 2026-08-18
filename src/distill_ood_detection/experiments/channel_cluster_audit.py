"""Task generation and execution for the NMF channel-cluster audit."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import torch
import yaml

from distill_ood_detection.config import ExperimentConfig, load_config
from distill_ood_detection.utils import resolve_device, set_seed, write_json

AuditVariant = Literal["whole", "fraction_p025"]


@dataclass(frozen=True)
class AuditLayerConfig:
    """One layer and hierarchy cut used by an audit."""

    template_config: str
    distance_threshold: float


@dataclass(frozen=True)
class ChannelClusterAuditConfig:
    """Dataset-level configuration that expands into specialist tasks."""

    experiment_name: str
    run_dir: str
    dataset: str
    layers: dict[str, AuditLayerConfig]
    variants: tuple[AuditVariant, ...] = ("whole", "fraction_p025")
    seed: int = 42
    epochs: int = 50
    fraction: float = 0.25
    fractional_min_cluster_size: int = 4
    fractional_evaluation_draws: int = 10


@dataclass(frozen=True)
class ChannelClusterAuditTask:
    """One fixed-cluster specialist training and inference task."""

    task_index: int
    audit_config_path: str
    template_config: str
    dataset: str
    layer: str
    distance_threshold: float
    variant: AuditVariant
    cluster_id: str
    cluster_order: int
    channel_group_index: int
    channel_indices: tuple[int, ...]
    cluster_size: int
    masked_channel_count: int
    realized_mask_fraction: float
    evaluation_draws: int
    run_dir: str


def load_channel_cluster_audit_config(path: Path) -> ChannelClusterAuditConfig:
    """Load and validate one dataset-level channel-cluster audit config."""

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Audit config must contain a mapping: {path}")
    layer_raw = raw.pop("layers", None)
    if not isinstance(layer_raw, dict) or set(layer_raw) != {
        "layer2",
        "layer3",
        "layer4",
    }:
        raise ValueError("Audit layers must define exactly layer2, layer3, and layer4")
    layers = {
        name: AuditLayerConfig(**settings)
        for name, settings in layer_raw.items()
    }
    variants_raw = raw.pop("variants", ("whole", "fraction_p025"))
    if not isinstance(variants_raw, (list, tuple)):
        raise ValueError("Audit variants must be a sequence")
    variants = tuple(variants_raw)
    config = ChannelClusterAuditConfig(layers=layers, variants=variants, **raw)
    if config.dataset not in {"cifar10", "cifar100"}:
        raise ValueError("Audit dataset must be cifar10 or cifar100")
    if config.seed != 42 or config.epochs != 50:
        raise ValueError("Channel-cluster audit requires seed 42 and 50 epochs")
    if (
        not config.variants
        or len(set(config.variants)) != len(config.variants)
        or any(variant not in {"whole", "fraction_p025"} for variant in config.variants)
    ):
        raise ValueError("Audit variants must uniquely select whole and/or fraction_p025")
    if not 0.0 < config.fraction <= 1.0:
        raise ValueError("Audit fraction must satisfy 0 < fraction <= 1")
    if config.fractional_min_cluster_size != 4:
        raise ValueError("Fractional audit requires minimum cluster size 4")
    if config.fractional_evaluation_draws != 10:
        raise ValueError("Fractional audit requires ten evaluation draws")
    return config


def build_channel_cluster_audit_tasks(
    config_paths: Sequence[Path],
) -> list[ChannelClusterAuditTask]:
    """Expand audit configs into deterministic hierarchy-ordered tasks."""

    tasks: list[ChannelClusterAuditTask] = []
    from distill_ood_detection.distillation.feature_denoising import (
        load_channel_groups,
    )

    for config_path in config_paths:
        audit = load_channel_cluster_audit_config(config_path)
        for layer_name in ("layer2", "layer3", "layer4"):
            layer = audit.layers[layer_name]
            template = load_config(Path(layer.template_config))
            if template.dataset.name != audit.dataset:
                raise ValueError(
                    f"Template dataset mismatch for {layer_name}: "
                    f"{template.dataset.name} != {audit.dataset}"
                )
            if template.student.feature_layer != layer_name:
                raise ValueError(f"Template feature layer mismatch for {layer_name}")
            group_path_raw = template.strategy.feature_denoising.channel_group_path
            if group_path_raw is None:
                raise ValueError(f"Template has no channel hierarchy: {layer.template_config}")
            groups = load_channel_groups(
                Path(group_path_raw),
                torch.device("cpu"),
                distance_threshold=layer.distance_threshold,
                expected_dataset=f"{audit.dataset}_train",
                expected_layer=layer_name,
                expected_feature_shape=template.student.input_shape,
            )
            hierarchy = torch.load(
                group_path_raw,
                map_location="cpu",
                weights_only=False,
            )
            leaf_order = hierarchy.get("leaf_order")
            if torch.is_tensor(leaf_order):
                leaf_order = leaf_order.tolist()
            if not isinstance(leaf_order, (list, tuple)):
                leaf_order = list(range(groups.channel_count))
            leaf_position = {int(channel): index for index, channel in enumerate(leaf_order)}
            ordered_group_indices = sorted(
                range(groups.group_count),
                key=lambda index: min(
                    leaf_position[int(channel)]
                    for channel in groups.membership[index]
                    .nonzero(as_tuple=False)
                    .flatten()
                    .tolist()
                ),
            )
            for cluster_order, group_index in enumerate(ordered_group_indices):
                channels = tuple(
                    int(value)
                    for value in groups.membership[group_index]
                    .nonzero(as_tuple=False)
                    .flatten()
                    .tolist()
                )
                for variant in audit.variants:
                    if (
                        variant == "fraction_p025"
                        and len(channels) < audit.fractional_min_cluster_size
                    ):
                        continue
                    masked_count = (
                        len(channels)
                        if variant == "whole"
                        else max(1, math.floor(audit.fraction * len(channels) + 0.5))
                    )
                    cluster_id = f"C{cluster_order:03d}"
                    tasks.append(
                        ChannelClusterAuditTask(
                            task_index=len(tasks),
                            audit_config_path=str(config_path),
                            template_config=layer.template_config,
                            dataset=audit.dataset,
                            layer=layer_name,
                            distance_threshold=layer.distance_threshold,
                            variant=variant,
                            cluster_id=cluster_id,
                            cluster_order=cluster_order,
                            channel_group_index=group_index,
                            channel_indices=channels,
                            cluster_size=len(channels),
                            masked_channel_count=masked_count,
                            realized_mask_fraction=masked_count / len(channels),
                            evaluation_draws=(
                                1
                                if variant == "whole"
                                else audit.fractional_evaluation_draws
                            ),
                            run_dir=str(
                                Path(audit.run_dir)
                                / variant
                                / layer_name
                                / cluster_id
                            ),
                        )
                    )
    identities = {
        (task.dataset, task.layer, task.variant, task.cluster_id) for task in tasks
    }
    if len(identities) != len(tasks):
        raise ValueError("Generated audit tasks do not have unique stable identities")
    return tasks


def write_channel_cluster_audit_manifest(
    config_paths: Sequence[Path], output_path: Path
) -> dict[str, Any]:
    """Write the combined deterministic task manifest."""

    tasks = build_channel_cluster_audit_tasks(config_paths)
    payload: dict[str, Any] = {
        "artifact_version": 1,
        "config_paths": [str(path) for path in config_paths],
        "task_count": len(tasks),
        "tasks": [asdict(task) for task in tasks],
    }
    write_json(output_path, payload)
    return payload


def load_channel_cluster_audit_manifest(path: Path) -> list[ChannelClusterAuditTask]:
    """Load tasks from a generated audit manifest."""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    task_raw = payload.get("tasks")
    if not isinstance(task_raw, list):
        raise ValueError(f"Audit manifest is missing tasks: {path}")
    tasks = []
    for item in task_raw:
        item["channel_indices"] = tuple(item["channel_indices"])
        tasks.append(ChannelClusterAuditTask(**item))
    if [task.task_index for task in tasks] != list(range(len(tasks))):
        raise ValueError("Audit manifest task indices are not contiguous")
    return tasks


def resolve_channel_cluster_task_config(
    task: ChannelClusterAuditTask,
) -> ExperimentConfig:
    """Resolve one manifest task into the standard experiment configuration."""

    audit = load_channel_cluster_audit_config(Path(task.audit_config_path))
    template = load_config(Path(task.template_config))
    method = (
        "channel_group_masked_residual_reconstruction"
        if task.variant == "whole"
        else "channel_group_stratified_masked_residual_reconstruction"
    )
    denoising = replace(
        template.strategy.feature_denoising,
        method=method,
        channel_group_distance_threshold=task.distance_threshold,
        channel_group_index=task.channel_group_index,
        channel_group_min_size=(
            1 if task.variant == "whole" else audit.fractional_min_cluster_size
        ),
        channel_group_mask_fraction=audit.fraction,
        evaluation_draws=task.evaluation_draws,
    )
    strategy = replace(template.strategy, feature_denoising=denoising)
    defaults = replace(
        template.training.defaults,
        epochs=audit.epochs,
        seed=audit.seed,
    )
    training = replace(template.training, defaults=defaults)
    return replace(
        template,
        experiment_name=(
            f"{audit.experiment_name}_{task.variant}_{task.layer}_{task.cluster_id}"
        ),
        run_dir=task.run_dir,
        strategy=strategy,
        training=training,
    )


def channel_cluster_task_is_complete(
    task: ChannelClusterAuditTask,
    *,
    deep: bool = False,
) -> bool:
    """Return whether all required training and score artifacts are valid."""

    config = resolve_channel_cluster_task_config(task)
    method_dir = Path(task.run_dir) / config.strategy.feature_denoising.method
    required = [
        method_dir / "best_student.pt",
        method_dir / "latest_student.pt",
        method_dir / "history.json",
        method_dir / "metrics.json",
        Path(task.run_dir) / "feature_denoising_scores" / "manifest.json",
    ]
    loader_names = [
        f"{config.dataset.name}_test",
        *(f"{item.name}_{item.split}" for item in config.dataset.ood_datasets),
    ]
    score_paths = [
        Path(task.run_dir)
        / "feature_denoising_scores"
        / name
        / "student_best.pt"
        for name in loader_names
    ]
    if not all(path.is_file() and path.stat().st_size > 0 for path in required + score_paths):
        return False
    try:
        with (method_dir / "history.json").open("r", encoding="utf-8") as handle:
            history = json.load(handle).get("history", [])
        with (method_dir / "metrics.json").open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
    except (OSError, ValueError, TypeError):
        return False
    if len(history) != config.training.defaults.epochs:
        return False
    if not all(
        np.isfinite(float(metrics.get(key, float("nan"))))
        for key in (
            "best_validation_reconstruction_loss",
            "final_validation_reconstruction_loss",
            "test_reconstruction_loss",
        )
    ):
        return False
    if not deep:
        return True
    for score_path in score_paths:
        try:
            artifact = torch.load(score_path, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, ValueError):
            return False
        labels = artifact.get("labels")
        if not torch.is_tensor(labels):
            return False
        expected = (
            "raw_reconstruction_error",
            "identity_error",
            "improvement",
            "relative_improvement",
        )
        for key in expected:
            values = artifact.get(key)
            if (
                not torch.is_tensor(values)
                or values.shape != labels.shape
                or not bool(torch.isfinite(values).all())
            ):
                return False
    return True


def incomplete_channel_cluster_task_indices(
    manifest_path: Path,
    *,
    deep: bool = False,
) -> list[int]:
    """Return manifest indices whose artifacts are incomplete."""

    return [
        task.task_index
        for task in load_channel_cluster_audit_manifest(manifest_path)
        if not channel_cluster_task_is_complete(task, deep=deep)
    ]


def slurm_array_spec(indices: Sequence[int]) -> str:
    """Compress sorted task indices into a SLURM array range expression."""

    unique = sorted(set(indices))
    if not unique:
        return ""
    ranges: list[str] = []
    start = previous = unique[0]
    for value in unique[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def run_channel_cluster_audit_task(
    manifest_path: Path,
    task_index: int,
    *,
    skip_complete: bool = True,
) -> dict[str, Any]:
    """Train and export scores for one restartable manifest task."""

    from distill_ood_detection.experiments.export_feature_denoising_scores import (
        run_feature_denoising_score_export,
    )
    from distill_ood_detection.experiments.train_student import run_experiment

    tasks = load_channel_cluster_audit_manifest(manifest_path)
    if not 0 <= task_index < len(tasks):
        raise ValueError(f"Task index {task_index} is outside [0, {len(tasks)})")
    task = tasks[task_index]
    if skip_complete and channel_cluster_task_is_complete(task, deep=True):
        return {"status": "already_complete", **asdict(task)}
    run_dir = Path(task.run_dir)
    write_json(run_dir / "audit_task.json", asdict(task))
    config = resolve_channel_cluster_task_config(task)
    run_experiment(config)
    run_feature_denoising_score_export(config, checkpoint="best")
    if not channel_cluster_task_is_complete(task, deep=True):
        raise RuntimeError(f"Audit task failed completeness validation: {task_index}")
    result = {"status": "complete", **asdict(task)}
    write_json(run_dir / "audit_completion.json", result)
    return result


@torch.no_grad()
def export_channel_cluster_audit_sample_metadata(
    config_path: Path,
) -> dict[str, Any]:
    """Export shared labels and teacher assignments once per audit dataset split."""

    from torch.nn import functional as F

    from distill_ood_detection.datasets.inference import (
        build_in_distribution_test_loader,
        build_ood_loaders,
    )
    from distill_ood_detection.models.teacher import load_teacher

    audit = load_channel_cluster_audit_config(config_path)
    template = load_config(Path(audit.layers["layer2"].template_config))
    device = resolve_device(template.training.defaults.device)
    set_seed(audit.seed)
    teacher = load_teacher(template.teacher, device)
    teacher.to(device)
    teacher.eval()
    loaders = [
        build_in_distribution_test_loader(template.dataset),
        *build_ood_loaders(template.dataset),
    ]
    teacher_id_class_names = _loader_class_names(loaders[0])
    output_dir = Path(audit.run_dir) / "sample_metadata"
    artifacts: list[dict[str, Any]] = []
    for named_loader in loaders:
        labels: list[torch.Tensor] = []
        predictions: list[torch.Tensor] = []
        confidences: list[torch.Tensor] = []
        for images, batch_labels in named_loader.loader:
            probabilities = F.softmax(teacher(images.to(device)), dim=1)
            confidence, prediction = probabilities.max(dim=1)
            labels.append(batch_labels.cpu())
            predictions.append(prediction.cpu())
            confidences.append(confidence.cpu())
        labels_tensor = torch.cat(labels)
        path = output_dir / f"{named_loader.name}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "sample_index": torch.arange(labels_tensor.numel()),
                "true_class": labels_tensor,
                "alignment_labels": labels_tensor.clone(),
                "teacher_predicted_id_class": torch.cat(predictions),
                "teacher_confidence": torch.cat(confidences),
                "metadata": {
                    "artifact_version": 1,
                    "dataset": named_loader.name,
                    "split": named_loader.split,
                    "id_dataset": audit.dataset,
                    "sample_count": labels_tensor.numel(),
                    "class_names": _loader_class_names(named_loader),
                    "teacher_id_class_names": teacher_id_class_names,
                    "teacher": asdict(template.teacher),
                    "seed": audit.seed,
                },
            },
            path,
        )
        artifacts.append(
            {
                "dataset": named_loader.name,
                "path": str(path),
                "sample_count": labels_tensor.numel(),
            }
        )
    manifest = {
        "artifact_version": 1,
        "id_dataset": audit.dataset,
        "artifacts": artifacts,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def _loader_class_names(named_loader: Any) -> list[str]:
    dataset = named_loader.loader.dataset
    classes = getattr(dataset, "classes", None)
    if isinstance(classes, (list, tuple)):
        return [str(value) for value in classes]
    dataset_name = named_loader.name.rsplit("_", 1)[0]
    class_count = 100 if dataset_name == "cifar100" else 10
    return _dataset_class_names(dataset_name, class_count)


def _dataset_class_names(dataset_name: str, class_count: int) -> list[str]:
    if dataset_name in {"mnist", "svhn"}:
        return [str(index) for index in range(10)]
    return [f"{dataset_name} class {index}" for index in range(class_count)]
