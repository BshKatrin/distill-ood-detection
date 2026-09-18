"""Export and summarize global-student channel-cluster improvements."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np
import torch

from distill_ood_detection.config import ExperimentConfig, load_config
from distill_ood_detection.evaluation.global_channel_cluster_distributions import (
    boxplot_summary,
    teacher_model_name,
)
from distill_ood_detection.evaluation.channel_cluster_audit import (
    id_ood_point_metrics,
)
from distill_ood_detection.utils import resolve_device, set_seed, write_json

if TYPE_CHECKING:
    from distill_ood_detection.distillation.feature_denoising import ChannelGroups

CheckpointSelection = Literal["best", "latest"]
SCORES = (
    "raw_reconstruction_error",
    "absolute_improvement",
    "relative_improvement",
)
SCORE_SIGNS = {
    "raw_reconstruction_error": -1.0,
    "absolute_improvement": 1.0,
    "relative_improvement": 1.0,
}
DATASET_LABELS = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "mnist": "MNIST",
    "svhn": "SVHN",
}


def run_global_channel_cluster_distribution_export(
    config: ExperimentConfig,
    checkpoint: CheckpointSelection = "best",
) -> dict[str, object]:
    """Export per-image cluster improvements for one global NMF student."""

    denoising = config.strategy.feature_denoising
    if (
        config.strategy.name != "feature_denoising"
        or denoising.method
        != "channel_group_stratified_masked_residual_reconstruction"
        or denoising.channel_group_index is not None
    ):
        raise ValueError("Distribution export requires a global stratified student")
    if denoising.channel_group_path is None or config.student.feature_layer is None:
        raise ValueError("Distribution export requires a channel hierarchy and layer")

    from distill_ood_detection.distillation.feature_denoising import (
        collect_channel_group_cluster_improvements,
        load_channel_groups,
    )
    from distill_ood_detection.datasets.inference import (
        build_in_distribution_test_loader,
        build_ood_loaders,
    )
    from distill_ood_detection.models.student import build_student
    from distill_ood_detection.models.teacher import (
        build_feature_forwarder,
        load_teacher,
    )

    set_seed(config.training.defaults.seed)
    device = resolve_device(config.training.defaults.device)
    teacher = load_teacher(config.teacher, device)
    forwarder = build_feature_forwarder(teacher, config.student.feature_layer)
    forwarder.to(device)
    forwarder.eval()
    groups = load_channel_groups(
        Path(denoising.channel_group_path),
        device,
        distance_threshold=denoising.channel_group_distance_threshold,
        expected_dataset=f"{config.dataset.name}_train",
        expected_layer=config.student.feature_layer,
        expected_feature_shape=config.student.input_shape,
    )
    ordered_groups, cluster_metadata = _ordered_eligible_groups(
        groups,
        Path(denoising.channel_group_path),
        min_group_size=denoising.channel_group_min_size,
        mask_fraction=denoising.channel_group_mask_fraction,
    )
    student = build_student(config.student)
    checkpoint_path = _student_checkpoint_path(
        Path(config.run_dir),
        denoising.method,
        checkpoint,
    )
    student.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    student.to(device)
    student.eval()

    output_dir = Path(config.run_dir) / "global_channel_cluster_distributions"
    loaders = [
        build_in_distribution_test_loader(config.dataset),
        *build_ood_loaders(config.dataset),
    ]
    artifacts: list[dict[str, object]] = []
    for named_loader in loaders:
        set_seed(config.training.defaults.seed)
        scores = collect_channel_group_cluster_improvements(
            student=student,
            loader=named_loader.loader,
            device=device,
            perturbation_forwarder=forwarder,
            feature_denoising_config=denoising,
            channel_groups=ordered_groups,
        )
        path = output_dir / named_loader.name / f"student_{checkpoint}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                **scores,
                "metadata": {
                    "artifact_version": 1,
                    "experiment_name": config.experiment_name,
                    "id_dataset": config.dataset.name,
                    "dataset": named_loader.name,
                    "split": named_loader.split,
                    "teacher_model": teacher_model_name(config.teacher.hf_model_id),
                    "teacher": asdict(config.teacher),
                    "feature_layer": config.student.feature_layer,
                    "checkpoint": checkpoint,
                    "checkpoint_path": str(checkpoint_path),
                    "evaluation_draws": denoising.evaluation_draws,
                    "score_aggregation": "per_image_mean_over_draws",
                    "relative_improvement_epsilon": 1.0e-12,
                    "channel_group_path": denoising.channel_group_path,
                    "channel_group_distance_threshold": (
                        denoising.channel_group_distance_threshold
                    ),
                    "channel_group_min_size": denoising.channel_group_min_size,
                    "channel_group_mask_fraction": (
                        denoising.channel_group_mask_fraction
                    ),
                    "clusters": cluster_metadata,
                },
            },
            path,
        )
        artifacts.append(
            {
                "dataset": named_loader.name,
                "split": named_loader.split,
                "path": str(path),
                "sample_count": int(scores["labels"].numel()),
            }
        )
    manifest = {
        "artifact_version": 1,
        "experiment_name": config.experiment_name,
        "checkpoint": checkpoint,
        "scores": list(SCORES),
        "artifacts": artifacts,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def summarize_global_channel_cluster_distributions(
    config_paths: Sequence[Path],
    output_dir: Path,
    checkpoint: CheckpointSelection = "best",
) -> dict[str, object]:
    """Write compact boxplot summaries for the global-student dashboard."""

    import pandas as pd

    rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for config_path in config_paths:
        config = load_config(config_path)
        artifact_dir = Path(config.run_dir) / "global_channel_cluster_distributions"
        dataset_names = [
            f"{config.dataset.name}_test",
            *(f"{item.name}_{item.split}" for item in config.dataset.ood_datasets),
        ]
        artifacts: dict[str, dict[str, object]] = {}
        for dataset_name in dataset_names:
            path = artifact_dir / dataset_name / f"student_{checkpoint}.pt"
            if not path.is_file():
                raise FileNotFoundError(f"Missing cluster distribution artifact: {path}")
            artifact = torch.load(path, map_location="cpu", weights_only=False)
            artifacts[dataset_name] = artifact
            metadata = artifact.get("metadata")
            if not isinstance(metadata, dict):
                raise ValueError(f"Missing metadata in {path}")
            clusters = metadata.get("clusters")
            if not isinstance(clusters, list) or not clusters:
                raise ValueError(f"Missing cluster metadata in {path}")
            distribution, distribution_order, dataset_label = _distribution_context(
                config.dataset.name,
                dataset_name,
            )
            for score in SCORES:
                values = artifact.get(score)
                expected_shape = (artifact["labels"].numel(), len(clusters))
                if not torch.is_tensor(values) or tuple(values.shape) != expected_shape:
                    raise ValueError(
                        f"Invalid {score} shape in {path}: "
                        f"{getattr(values, 'shape', None)} != {expected_shape}"
                    )
                array = values.numpy().astype(np.float64, copy=False)
                for column, cluster in enumerate(clusters):
                    rows.append(
                        {
                            "teacher_model": metadata["teacher_model"],
                            "id_dataset": config.dataset.name,
                            "layer": metadata["feature_layer"],
                            "score": score,
                            "distribution": distribution,
                            "distribution_order": distribution_order,
                            "dataset_label": dataset_label,
                            **cluster,
                            **boxplot_summary(array[:, column]),
                        }
                    )
        id_artifact_name = f"{config.dataset.name}_test"
        id_artifact = artifacts[id_artifact_name]
        id_metadata = id_artifact["metadata"]
        clusters = id_metadata["clusters"]
        for ood_artifact_name in dataset_names[1:]:
            ood_artifact = artifacts[ood_artifact_name]
            distribution, distribution_order, dataset_label = _distribution_context(
                config.dataset.name,
                ood_artifact_name,
            )
            for score in SCORES:
                id_values = id_artifact[score].numpy().astype(np.float64, copy=False)
                ood_values = ood_artifact[score].numpy().astype(np.float64, copy=False)
                sign = SCORE_SIGNS[score]
                for column, cluster in enumerate(clusters):
                    metrics = id_ood_point_metrics(
                        id_values[:, column] * sign,
                        ood_values[:, column] * sign,
                    )
                    metric_rows.append(
                        {
                            "teacher_model": id_metadata["teacher_model"],
                            "id_dataset": config.dataset.name,
                            "layer": id_metadata["feature_layer"],
                            "score": score,
                            "ood_distribution": distribution,
                            "ood_distribution_order": distribution_order,
                            "ood_dataset": ood_artifact_name.rsplit("_", 1)[0],
                            "ood_dataset_label": dataset_label,
                            **cluster,
                            **metrics,
                        }
                    )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No global cluster distribution rows were generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "boxplot_summaries.parquet"
    frame.to_parquet(summary_path, index=False)
    metric_frame = pd.DataFrame(metric_rows)
    if metric_frame.empty:
        raise ValueError("No global cluster OOD metric rows were generated")
    metric_path = output_dir / "ood_metrics.parquet"
    metric_frame.to_parquet(metric_path, index=False)
    manifest = {
        "artifact_version": 1,
        "config_paths": [str(path) for path in config_paths],
        "checkpoint": checkpoint,
        "row_count": len(rows),
        "summary_path": str(summary_path),
        "ood_metric_row_count": len(metric_rows),
        "ood_metric_path": str(metric_path),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def _ordered_eligible_groups(
    groups: ChannelGroups,
    artifact_path: Path,
    *,
    min_group_size: int,
    mask_fraction: float,
) -> tuple[ChannelGroups, list[dict[str, object]]]:
    from distill_ood_detection.distillation.feature_denoising import ChannelGroups

    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    leaf_order = artifact.get("leaf_order")
    if torch.is_tensor(leaf_order):
        leaf_order = leaf_order.tolist()
    if not isinstance(leaf_order, (list, tuple)):
        leaf_order = list(range(groups.channel_count))
    leaf_position = {int(channel): index for index, channel in enumerate(leaf_order)}
    membership = groups.membership.detach().cpu()
    ordered_indices = sorted(
        range(groups.group_count),
        key=lambda group_index: min(
            leaf_position[int(channel)]
            for channel in membership[group_index]
            .nonzero(as_tuple=False)
            .flatten()
            .tolist()
        ),
    )
    eligible_indices: list[int] = []
    cluster_metadata: list[dict[str, object]] = []
    for cluster_order, group_index in enumerate(ordered_indices):
        cluster_size = int(groups.group_sizes[group_index].item())
        if cluster_size < min_group_size:
            continue
        eligible_indices.append(group_index)
        masked_count = max(1, math.floor(mask_fraction * cluster_size + 0.5))
        cluster_metadata.append(
            {
                "cluster_id": f"C{cluster_order:03d}",
                "cluster_order": cluster_order,
                "channel_group_index": group_index,
                "cluster_size": cluster_size,
                "masked_channel_count": masked_count,
                "realized_mask_fraction": masked_count / cluster_size,
            }
        )
    if not eligible_indices:
        raise ValueError("No channel groups satisfy the configured minimum size")
    ordered_groups = ChannelGroups(
        membership=groups.membership[eligible_indices],
        distance_threshold=groups.distance_threshold,
        source_path=groups.source_path,
    )
    return ordered_groups, cluster_metadata


def _student_checkpoint_path(
    experiment_dir: Path,
    method: str,
    checkpoint: CheckpointSelection,
) -> Path:
    metrics_path = experiment_dir / method / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing student metrics artifact: {metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    raw_path = metrics.get(f"{checkpoint}_checkpoint_path")
    if not isinstance(raw_path, str):
        raise ValueError(f"Missing {checkpoint} checkpoint path in {metrics_path}")
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.is_file():
        raise FileNotFoundError(f"Student checkpoint not found: {path}")
    return path


def _distribution_context(
    id_dataset: str,
    artifact_name: str,
) -> tuple[str, int, str]:
    dataset_name = artifact_name.rsplit("_", 1)[0]
    if dataset_name == id_dataset:
        return "id", 0, DATASET_LABELS[id_dataset]
    if dataset_name == "mnist":
        return "mnist", 1, "MNIST"
    if dataset_name == "svhn":
        return "svhn", 2, "SVHN"
    opposite = "cifar100" if id_dataset == "cifar10" else "cifar10"
    if dataset_name == opposite:
        return "opposite_cifar", 3, DATASET_LABELS[opposite]
    raise ValueError(f"Unsupported dashboard dataset: {artifact_name}")
