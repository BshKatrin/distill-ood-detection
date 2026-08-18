"""Build hierarchical teacher-channel groups from ID training activations."""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from sklearn.decomposition import MiniBatchNMF

from distill_ood_detection.config import ChannelGroupingConfig
from distill_ood_detection.datasets.inference import (
    NamedLoader,
    build_in_distribution_full_train_loader,
)
from distill_ood_detection.evaluation.channel_grouping import (
    correlation_distance,
    flat_channel_groups,
    hierarchical_channel_linkage,
    hierarchy_leaf_order,
    latent_channel_cosine_distance,
    pearson_channel_correlation,
    render_channel_dendrograms,
    standardize_channel_profiles,
    top_activation_profiles,
)
from distill_ood_detection.experiments.export_teacher_activations import (
    TeacherActivationCollector,
)
from distill_ood_detection.models.teacher import load_teacher
from distill_ood_detection.utils import resolve_device, set_seed, write_json


def run_channel_grouping(config: ChannelGroupingConfig) -> dict[str, object]:
    """Build and save channel hierarchies from the full ID training split."""

    if config.method == "nmf_latent_cosine":
        return run_nmf_latent_channel_grouping(config)
    return run_top_activation_channel_grouping(config)


def run_top_activation_channel_grouping(
    config: ChannelGroupingConfig,
) -> dict[str, object]:
    """Build top-activation Pearson channel hierarchies."""

    set_seed(config.seed)
    device = resolve_device(config.device)
    output_dir = Path(config.run_dir) / "channel_groups"
    output_dir.mkdir(parents=True, exist_ok=True)
    named_loader = build_in_distribution_full_train_loader(config.dataset)
    teacher = load_teacher(config.teacher, device)
    profiles_by_layer, feature_shapes = collect_channel_profiles(
        teacher=teacher,
        layers=config.layers,
        named_loader=named_loader,
        device=device,
        top_fraction=config.profile_top_fraction,
    )

    artifacts = []
    for layer in config.layers:
        artifacts.append(
            build_layer_hierarchy(
                profiles=profiles_by_layer[layer],
                layer=layer,
                output_dir=output_dir,
                device=device,
                top_fraction=config.profile_top_fraction,
                epsilon=config.standardization_epsilon,
                linkage_method=config.linkage_method,
                cut_thresholds=config.cut_thresholds,
                dataset_name=named_loader.name,
                dataset_split=named_loader.split,
                feature_shape=feature_shapes[layer],
            )
        )

    manifest = {
        "experiment_name": config.experiment_name,
        "dataset": asdict(config.dataset),
        "teacher": asdict(config.teacher),
        "source_dataset": named_loader.name,
        "source_split": named_loader.split,
        "sample_count": len(named_loader.loader.dataset),
        "layers": list(config.layers),
        "profile": {
            "name": "mean_top_spatial_activations",
            "top_fraction": config.profile_top_fraction,
            "rounding": "ceil",
        },
        "standardization": {
            "axis": "ID training samples, independently per channel",
            "population_standard_deviation": True,
            "epsilon": config.standardization_epsilon,
        },
        "distance": "1 - pearson_correlation",
        "linkage_method": config.linkage_method,
        "cut_thresholds": list(config.cut_thresholds),
        "raw_profiles_saved": False,
        "artifacts": artifacts,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def run_nmf_latent_channel_grouping(
    config: ChannelGroupingConfig,
) -> dict[str, object]:
    """Fit global NMF bases and cluster channel columns by cosine distance."""

    set_seed(config.seed)
    device = resolve_device(config.device)
    output_dir = Path(config.run_dir) / "channel_groups"
    output_dir.mkdir(parents=True, exist_ok=True)
    named_loader = build_in_distribution_full_train_loader(config.dataset)
    teacher = load_teacher(config.teacher, device)
    models, feature_shapes = fit_streaming_nmf_models(
        teacher=teacher,
        layers=config.layers,
        named_loader=named_loader,
        device=device,
        n_components=config.nmf_components,
        update_batch_size=config.nmf_batch_size,
        epochs=config.nmf_epochs,
        random_state=config.seed,
    )

    artifacts = [
        build_nmf_latent_layer_hierarchy(
            model=models[layer],
            layer=layer,
            output_dir=output_dir,
            linkage_method=config.linkage_method,
            dataset_name=named_loader.name,
            dataset_split=named_loader.split,
            sample_count=len(named_loader.loader.dataset),
            feature_shape=feature_shapes[layer],
            nmf_epochs=config.nmf_epochs,
            nmf_batch_size=config.nmf_batch_size,
        )
        for layer in config.layers
    ]
    manifest = {
        "experiment_name": config.experiment_name,
        "dataset": asdict(config.dataset),
        "teacher": asdict(config.teacher),
        "source_dataset": named_loader.name,
        "source_split": named_loader.split,
        "sample_count": len(named_loader.loader.dataset),
        "layers": list(config.layers),
        "nmf": {
            "scope": "global",
            "components": config.nmf_components,
            "batch_size": config.nmf_batch_size,
            "epochs": config.nmf_epochs,
            "objective": "frobenius_residual",
            "regularization": "none",
        },
        "channel_profile": "column of NMF P matrix",
        "distance": "cosine_distance",
        "linkage_method": config.linkage_method,
        "flat_cuts_saved": False,
        "raw_activations_saved": False,
        "artifacts": artifacts,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


@torch.no_grad()
def fit_streaming_nmf_models(
    teacher: nn.Module,
    layers: tuple[str, ...],
    named_loader: NamedLoader,
    device: torch.device,
    n_components: int,
    update_batch_size: int,
    epochs: int,
    random_state: int,
) -> tuple[dict[str, MiniBatchNMF], dict[str, tuple[int, int, int]]]:
    """Fit one global MiniBatchNMF model per layer without saving activations."""

    if n_components <= 0 or update_batch_size <= 0 or epochs <= 0:
        raise ValueError("NMF components, update batch size, and epochs must be positive")
    collector = TeacherActivationCollector(teacher, layers)
    models: dict[str, MiniBatchNMF] = {}
    feature_shapes: dict[str, tuple[int, int, int]] = {}
    try:
        teacher.eval()
        for _epoch in range(epochs):
            for images, _labels in named_loader.loader:
                collector.clear()
                teacher(images.to(device))
                for layer in layers:
                    activations = collector.activation(layer)
                    feature_shape = tuple(activations.shape[1:])
                    if len(feature_shape) != 3:
                        raise ValueError(
                            f"NMF grouping requires a 4D feature map at {layer}; "
                            f"got shape {tuple(activations.shape)}"
                        )
                    if n_components > feature_shape[0]:
                        raise ValueError(
                            f"nmf_components={n_components} exceeds {layer} "
                            f"channel count {feature_shape[0]}"
                        )
                    if layer in feature_shapes and feature_shapes[layer] != feature_shape:
                        raise ValueError(f"Feature shape changed between batches at {layer}")
                    feature_shapes[layer] = feature_shape
                    minimum = float(activations.min())
                    if minimum < -1.0e-6:
                        raise ValueError(
                            "NMF channel grouping requires post-ReLU non-negative "
                            f"activations at {layer}; observed minimum {minimum:.6g}"
                        )
                    values = (
                        activations.clamp_min(0.0)
                        .permute(0, 2, 3, 1)
                        .reshape(-1, feature_shape[0])
                        .float()
                        .cpu()
                        .numpy()
                    )
                    model = models.get(layer)
                    if model is None:
                        model = MiniBatchNMF(
                            n_components=n_components,
                            init="nndsvda",
                            batch_size=update_batch_size,
                            beta_loss="frobenius",
                            alpha_W=0.0,
                            alpha_H=0.0,
                            l1_ratio=0.0,
                            forget_factor=1.0,
                            random_state=random_state,
                        )
                        models[layer] = model
                    for start in range(0, values.shape[0], update_batch_size):
                        model.partial_fit(values[start : start + update_batch_size])
    finally:
        collector.close()
    return models, feature_shapes


def build_nmf_latent_layer_hierarchy(
    model: MiniBatchNMF,
    layer: str,
    output_dir: Path,
    linkage_method: str,
    dataset_name: str,
    dataset_split: str,
    sample_count: int,
    feature_shape: tuple[int, int, int],
    nmf_epochs: int,
    nmf_batch_size: int,
) -> dict[str, object]:
    """Cluster channel columns of one fitted NMF basis and save compact outputs."""

    p_matrix = torch.from_numpy(model.components_.copy()).float()
    if p_matrix.shape[1] != feature_shape[0]:
        raise ValueError("NMF P matrix channel count must match feature_shape")
    distances, active_channels = latent_channel_cosine_distance(p_matrix)
    linkage_matrix = hierarchical_channel_linkage(distances, method=linkage_method)
    leaf_order = hierarchy_leaf_order(linkage_matrix)
    zero_profile_channels = (
        (~active_channels).nonzero(as_tuple=False).flatten().tolist()
    )
    artifact_path = output_dir / f"{layer}.pt"
    torch.save(
        {
            "method": "nmf_latent_cosine",
            "layer": layer,
            "dataset": dataset_name,
            "split": dataset_split,
            "sample_count": sample_count,
            "spatial_vector_count": (
                sample_count * feature_shape[1] * feature_shape[2]
            ),
            "feature_shape": feature_shape,
            "concept_count": p_matrix.shape[0],
            "channel_count": p_matrix.shape[1],
            "p_matrix": p_matrix,
            "zero_profile_channels": zero_profile_channels,
            "distance": "cosine_distance_between_p_matrix_columns",
            "linkage": torch.from_numpy(linkage_matrix),
            "linkage_method": linkage_method,
            "leaf_order": leaf_order,
            "nmf_epochs": nmf_epochs,
            "nmf_batch_size": nmf_batch_size,
            "nmf_steps": int(model.n_steps_),
        },
        artifact_path,
    )
    overview_path, labeled_path = render_channel_dendrograms(
        linkage_matrix=linkage_matrix,
        layer=layer,
        output_dir=output_dir,
        thresholds=(),
        title=f"{layer}: NMF latent-profile channel hierarchy",
        distance_label="Cosine distance between NMF P columns",
    )
    return {
        "layer": layer,
        "sample_count": sample_count,
        "spatial_vector_count": sample_count * feature_shape[1] * feature_shape[2],
        "concept_count": p_matrix.shape[0],
        "channel_count": p_matrix.shape[1],
        "zero_profile_channel_count": len(zero_profile_channels),
        "nmf_steps": int(model.n_steps_),
        "artifact_path": str(artifact_path),
        "overview_dendrogram_path": str(overview_path),
        "labeled_dendrogram_path": str(labeled_path),
    }


@torch.no_grad()
def collect_channel_profiles(
    teacher: nn.Module,
    layers: tuple[str, ...],
    named_loader: NamedLoader,
    device: torch.device,
    top_fraction: float,
) -> tuple[dict[str, torch.Tensor], dict[str, tuple[int, int, int]]]:
    """Stream images through a teacher and retain only compact channel profiles."""

    collector = TeacherActivationCollector(teacher, layers)
    batches_by_layer: dict[str, list[torch.Tensor]] = {layer: [] for layer in layers}
    feature_shapes: dict[str, tuple[int, int, int]] = {}
    try:
        teacher.eval()
        for images, _labels in named_loader.loader:
            collector.clear()
            teacher(images.to(device))
            for layer in layers:
                activations = collector.activation(layer)
                feature_shape = tuple(activations.shape[1:])
                if len(feature_shape) != 3:
                    raise ValueError(
                        f"Channel grouping requires a 4D feature map at {layer}; "
                        f"got shape {tuple(activations.shape)}"
                    )
                if layer in feature_shapes and feature_shapes[layer] != feature_shape:
                    raise ValueError(f"Feature shape changed between batches at {layer}")
                feature_shapes[layer] = feature_shape
                batch_profiles = top_activation_profiles(
                    activations,
                    top_fraction=top_fraction,
                )
                batches_by_layer[layer].append(batch_profiles.cpu())
    finally:
        collector.close()
    profiles = {
        layer: torch.cat(layer_batches, dim=0)
        for layer, layer_batches in batches_by_layer.items()
    }
    return profiles, feature_shapes


def build_layer_hierarchy(
    profiles: torch.Tensor,
    layer: str,
    output_dir: Path,
    device: torch.device,
    top_fraction: float,
    epsilon: float,
    linkage_method: str,
    cut_thresholds: tuple[float, ...],
    dataset_name: str,
    dataset_split: str,
    feature_shape: tuple[int, int, int],
) -> dict[str, object]:
    """Standardize profiles, cluster channels, and write one layer's artifacts."""

    if profiles.ndim != 2:
        raise ValueError("profiles must have shape (samples, channels)")
    if feature_shape[0] != profiles.shape[1]:
        raise ValueError("feature_shape channel count must match profiles")
    profiles_device = profiles.to(device)
    standardized, means, standard_deviations, active_channels = (
        standardize_channel_profiles(profiles_device, epsilon=epsilon)
    )
    correlations = pearson_channel_correlation(standardized, active_channels)
    distances = correlation_distance(correlations)
    linkage_matrix = hierarchical_channel_linkage(distances, method=linkage_method)
    groups = {
        f"{threshold:g}": flat_channel_groups(linkage_matrix, threshold)
        for threshold in cut_thresholds
    }
    leaf_order = hierarchy_leaf_order(linkage_matrix)
    inactive_channels = (~active_channels).nonzero(as_tuple=False).flatten().tolist()
    artifact_path = output_dir / f"{layer}.pt"
    torch.save(
        {
            "layer": layer,
            "dataset": dataset_name,
            "split": dataset_split,
            "sample_count": profiles.shape[0],
            "channel_count": profiles.shape[1],
            "feature_shape": feature_shape,
            "profile_top_fraction": top_fraction,
            "profile_top_count": max(
                1,
                math.ceil(top_fraction * feature_shape[1] * feature_shape[2]),
            ),
            "profile_mean": means.cpu(),
            "profile_std": standard_deviations.cpu(),
            "constant_channels": inactive_channels,
            "correlation": correlations.cpu(),
            "distance": distances.cpu(),
            "linkage": torch.from_numpy(linkage_matrix),
            "linkage_method": linkage_method,
            "leaf_order": leaf_order,
            "cut_groups": groups,
        },
        artifact_path,
    )
    groups_path = output_dir / f"{layer}_groups.json"
    write_json(
        groups_path,
        {
            "layer": layer,
            "distance": "1 - pearson_correlation",
            "linkage_method": linkage_method,
            "constant_channels": inactive_channels,
            "leaf_order": leaf_order,
            "groups_by_distance_threshold": groups,
        },
    )
    overview_path, labeled_path = render_channel_dendrograms(
        linkage_matrix=linkage_matrix,
        layer=layer,
        output_dir=output_dir,
        thresholds=cut_thresholds,
    )
    del profiles_device, standardized, correlations, distances
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "layer": layer,
        "sample_count": profiles.shape[0],
        "channel_count": profiles.shape[1],
        "constant_channel_count": len(inactive_channels),
        "artifact_path": str(artifact_path),
        "groups_path": str(groups_path),
        "overview_dendrogram_path": str(overview_path),
        "labeled_dendrogram_path": str(labeled_path),
        "group_counts": {
            threshold: len(threshold_groups)
            for threshold, threshold_groups in groups.items()
        },
    }
