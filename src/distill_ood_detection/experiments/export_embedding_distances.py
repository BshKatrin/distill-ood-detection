"""Export layerwise k-NN neighbor-output OOD Scores."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from distill_ood_detection.config import EmbeddingDistanceConfig
from distill_ood_detection.datasets.inference import (
    NamedLoader,
    build_in_distribution_full_train_loader,
    build_in_distribution_test_loader,
    build_ood_loaders,
)
from distill_ood_detection.evaluation.ood_scores import (
    absolute_energy_gap,
    absolute_max_probability_difference,
    student_energy,
    student_msp,
    student_teacher_kl_divergence,
)
from distill_ood_detection.models.teacher import load_teacher
from distill_ood_detection.utils import resolve_device, set_seed, write_json


KNN_SCORE_SIGNS = {
    "student_teacher_kl_divergence": -1,
    "absolute_max_probability_difference": -1,
    "absolute_energy_gap": -1,
    "student_msp": +1,
    "student_energy": +1,
    "ensemble_predictive_entropy": -1,
    "ensemble_bald": -1,
}


def run_embedding_distance_export(config: EmbeddingDistanceConfig) -> dict[str, object]:
    """Export pooled embeddings and neighbor-output OOD Scores."""

    set_seed(config.seed)
    device = resolve_device(config.device)
    output_dir = Path(config.run_dir)
    embeddings_dir = output_dir / "embeddings"
    distances_dir = output_dir / "distances"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    distances_dir.mkdir(parents=True, exist_ok=True)

    loaders = [
        build_in_distribution_full_train_loader(config.dataset),
        build_in_distribution_test_loader(config.dataset),
        *build_ood_loaders(config.dataset),
    ]
    teacher = load_teacher(config.teacher, device)

    embedding_artifacts: list[dict[str, object]] = []
    for named_loader in loaders:
        print(f"Extracting pooled embeddings: {named_loader.name}", flush=True)
        embedding_artifacts.extend(
            export_loader_embeddings(
                teacher=teacher,
                layers=config.layers,
                named_loader=named_loader,
                output_dir=embeddings_dir,
                device=device,
                shard_size=config.embedding_shard_size,
                metadata={
                    "experiment_name": config.experiment_name,
                    "model": "teacher",
                    "teacher": asdict(config.teacher),
                    "pooling": "gap",
                    "dtype": "float32",
                },
            )
        )

    embedding_manifest = {
        "version": 3,
        "artifact_type": "knn_teacher_embeddings_and_outputs",
        "experiment_name": config.experiment_name,
        "dataset": asdict(config.dataset),
        "teacher": asdict(config.teacher),
        "layers": list(config.layers),
        "pooling": "gap",
        "dtype": "float32",
        "seed": config.seed,
        "shard_size": config.embedding_shard_size,
        "artifacts": embedding_artifacts,
    }
    write_json(embeddings_dir / "manifest.json", embedding_manifest)

    reference_name = f"{config.dataset.name}_train"
    query_loaders = loaders[1:]
    distance_artifacts: list[dict[str, object]] = []
    for layer in config.layers:
        print(f"Loading ID reference embeddings: {layer}", flush=True)
        reference_embeddings, reference_labels = load_embedding_shards(
            embedding_artifacts,
            dataset=reference_name,
            layer=layer,
        )
        reference_logits, reference_probabilities = load_output_shards(
            embedding_artifacts,
            dataset=reference_name,
            layer=layer,
        )
        statistics = fit_id_statistics(
            reference_embeddings,
            reference_labels,
            epsilon=config.standardization_epsilon,
        )
        for named_loader in query_loaders:
            print(
                f"Computing exact distances: {named_loader.name} {layer}",
                flush=True,
            )
            query_embeddings, query_labels = load_embedding_shards(
                embedding_artifacts,
                dataset=named_loader.name,
                layer=layer,
            )
            query_logits, query_probabilities = load_output_shards(
                embedding_artifacts,
                dataset=named_loader.name,
                layer=layer,
            )
            artifact = export_dataset_distances(
                query_embeddings=query_embeddings,
                query_labels=query_labels,
                query_logits=query_logits,
                query_probabilities=query_probabilities,
                reference_embeddings=reference_embeddings,
                reference_labels=reference_labels,
                reference_logits=reference_logits,
                reference_probabilities=reference_probabilities,
                statistics=statistics,
                output_dir=distances_dir,
                dataset=named_loader.name,
                split=named_loader.split,
                layer=layer,
                device=device,
                k_neighbors=config.k_neighbors,
                query_block_size=config.query_block_size,
                reference_block_size=config.reference_block_size,
                metadata={
                    "experiment_name": config.experiment_name,
                    "teacher": asdict(config.teacher),
                    "id_reference_dataset": reference_name,
                    "pooling": "gap",
                },
            )
            distance_artifacts.append(artifact)

    distance_manifest = {
        "version": 3,
        "artifact_type": "knn_neighbor_output_ood_scores",
        "experiment_name": config.experiment_name,
        "dataset": asdict(config.dataset),
        "teacher": asdict(config.teacher),
        "layers": list(config.layers),
        "id_reference_dataset": reference_name,
        "k_neighbors": config.k_neighbors,
        "query_block_size": config.query_block_size,
        "reference_block_size": config.reference_block_size,
        "standardization_epsilon": config.standardization_epsilon,
        "score_signs": KNN_SCORE_SIGNS,
        "higher_score_is": "more_id_like",
        "seed": config.seed,
        "artifacts": distance_artifacts,
    }
    write_json(distances_dir / "manifest.json", distance_manifest)
    return distance_manifest


@torch.inference_mode()
def export_loader_embeddings(
    teacher: nn.Module,
    layers: tuple[str, ...],
    named_loader: NamedLoader,
    output_dir: Path,
    device: torch.device,
    shard_size: int,
    metadata: dict[str, object],
) -> list[dict[str, object]]:
    """Export GAP embeddings and teacher outputs in bounded-size shards."""

    collector = PooledActivationCollector(teacher, layers)
    pending_embeddings: dict[str, list[torch.Tensor]] = {layer: [] for layer in layers}
    pending_labels: list[torch.Tensor] = []
    pending_logits: list[torch.Tensor] = []
    pending_count = 0
    sample_offset = 0
    shard_index = 0
    artifacts: list[dict[str, object]] = []
    teacher.eval()
    try:
        for images, labels in named_loader.loader:
            collector.clear()
            logits = teacher(images.to(device, non_blocking=True))
            for layer in layers:
                pending_embeddings[layer].append(collector.embedding(layer).cpu())
            pending_labels.append(labels.cpu())
            pending_logits.append(logits.cpu())
            pending_count += labels.shape[0]
            if pending_count >= shard_size:
                artifacts.extend(
                    _write_embedding_shard(
                        pending_embeddings=pending_embeddings,
                        pending_labels=pending_labels,
                        pending_logits=pending_logits,
                        output_dir=output_dir,
                        named_loader=named_loader,
                        layers=layers,
                        shard_index=shard_index,
                        sample_offset=sample_offset,
                        metadata=metadata,
                    )
                )
                sample_offset += pending_count
                shard_index += 1
                pending_embeddings = {layer: [] for layer in layers}
                pending_labels = []
                pending_logits = []
                pending_count = 0
        if pending_count:
            artifacts.extend(
                _write_embedding_shard(
                    pending_embeddings=pending_embeddings,
                    pending_labels=pending_labels,
                    pending_logits=pending_logits,
                    output_dir=output_dir,
                    named_loader=named_loader,
                    layers=layers,
                    shard_index=shard_index,
                    sample_offset=sample_offset,
                    metadata=metadata,
                )
            )
    finally:
        collector.close()
    return artifacts


def _write_embedding_shard(
    pending_embeddings: dict[str, list[torch.Tensor]],
    pending_labels: list[torch.Tensor],
    pending_logits: list[torch.Tensor],
    output_dir: Path,
    named_loader: NamedLoader,
    layers: tuple[str, ...],
    shard_index: int,
    sample_offset: int,
    metadata: dict[str, object],
) -> list[dict[str, object]]:
    """Write one aligned embedding and teacher-output shard per layer."""

    labels = torch.cat(pending_labels, dim=0)
    logits = torch.cat(pending_logits, dim=0).float()
    probabilities = torch.softmax(logits, dim=1)
    sample_count = labels.shape[0]
    artifacts: list[dict[str, object]] = []
    for layer in layers:
        embeddings = torch.cat(pending_embeddings[layer], dim=0).float()
        path = (
            output_dir
            / named_loader.name
            / named_loader.split
            / layer
            / f"part-{shard_index:05d}.pt"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                **metadata,
                "version": 3,
                "artifact_type": "knn_teacher_embedding_and_output_shard",
                "dataset": named_loader.name,
                "split": named_loader.split,
                "layer": layer,
                "sample_offset": sample_offset,
                "sample_count": sample_count,
                "embedding_dimension": embeddings.shape[1],
                "embeddings": embeddings,
                "logits": logits,
                "probabilities": probabilities,
                "labels": labels,
            },
            path,
        )
        artifacts.append(
            {
                "dataset": named_loader.name,
                "split": named_loader.split,
                "layer": layer,
                "sample_offset": sample_offset,
                "sample_count": sample_count,
                "embedding_dimension": embeddings.shape[1],
                "path": str(path),
            }
        )
    return artifacts


class PooledActivationCollector:
    """Capture GAP embeddings from multiple named teacher layers."""

    def __init__(self, teacher: nn.Module, layers: tuple[str, ...]) -> None:
        self._embeddings: dict[str, torch.Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        for layer_name in layers:
            try:
                layer = teacher.get_submodule(layer_name)
            except AttributeError as error:
                raise ValueError(f"Teacher does not have layer: {layer_name}") from error
            self._hooks.append(layer.register_forward_hook(self._capture(layer_name)))

    def clear(self) -> None:
        """Clear embeddings captured during the previous forward pass."""

        self._embeddings.clear()

    def embedding(self, layer: str) -> torch.Tensor:
        """Return one captured pooled embedding batch."""

        try:
            return self._embeddings[layer]
        except KeyError as error:
            raise RuntimeError(f"Feature hook did not run for layer: {layer}") from error

    def close(self) -> None:
        """Remove all registered hooks."""

        for hook in self._hooks:
            hook.remove()

    def _capture(
        self,
        layer_name: str,
    ) -> Callable[[nn.Module, tuple[object, ...], torch.Tensor], None]:
        def hook(
            _module: nn.Module,
            _inputs: tuple[object, ...],
            output: torch.Tensor,
        ) -> None:
            if output.ndim != 4:
                raise ValueError(
                    f"Expected a 4D activation at {layer_name}; got {tuple(output.shape)}"
                )
            self._embeddings[layer_name] = output.detach().mean(dim=(-2, -1))

        return hook


def load_embedding_shards(
    artifacts: list[dict[str, object]],
    dataset: str,
    layer: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load and concatenate the manifest-selected shards for a dataset layer."""

    selected = sorted(
        (
            artifact
            for artifact in artifacts
            if artifact["dataset"] == dataset and artifact["layer"] == layer
        ),
        key=lambda artifact: int(artifact["sample_offset"]),
    )
    if not selected:
        raise ValueError(f"No embedding shards for dataset={dataset!r}, layer={layer!r}")
    embeddings: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for artifact in selected:
        saved = torch.load(
            Path(str(artifact["path"])),
            map_location="cpu",
            weights_only=False,
        )
        embeddings.append(saved["embeddings"].float())
        labels.append(saved["labels"])
    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)


def load_output_shards(
    artifacts: list[dict[str, object]],
    dataset: str,
    layer: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load aligned teacher logits and probabilities from embedding shards."""

    selected = sorted(
        (
            artifact
            for artifact in artifacts
            if artifact["dataset"] == dataset and artifact["layer"] == layer
        ),
        key=lambda artifact: int(artifact["sample_offset"]),
    )
    if not selected:
        raise ValueError(f"No output shards for dataset={dataset!r}, layer={layer!r}")
    logits: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    for artifact in selected:
        saved = torch.load(
            Path(str(artifact["path"])),
            map_location="cpu",
            weights_only=False,
        )
        logits.append(saved["logits"].float())
        probabilities.append(saved["probabilities"].float())
    return torch.cat(logits, dim=0), torch.cat(probabilities, dim=0)


def fit_id_statistics(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
) -> dict[str, torch.Tensor]:
    """Fit normalization and class centroids using ID training embeddings only."""

    mean = embeddings.mean(dim=0)
    std = embeddings.std(dim=0, unbiased=False).clamp_min(epsilon)
    classes = labels.unique(sorted=True)
    raw_class_centroids = torch.stack(
        [embeddings[labels == class_label].mean(dim=0) for class_label in classes]
    )
    standardized = (embeddings - mean) / std
    standardized_class_centroids = torch.stack(
        [standardized[labels == class_label].mean(dim=0) for class_label in classes]
    )
    return {
        "mean": mean,
        "std": std,
        "classes": classes,
        "raw_centroid": mean,
        "raw_class_centroids": raw_class_centroids,
        "standardized_centroid": standardized.mean(dim=0),
        "standardized_class_centroids": standardized_class_centroids,
    }


def export_dataset_distances(
    query_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    query_logits: torch.Tensor,
    query_probabilities: torch.Tensor,
    reference_embeddings: torch.Tensor,
    reference_labels: torch.Tensor,
    reference_logits: torch.Tensor,
    reference_probabilities: torch.Tensor,
    statistics: dict[str, torch.Tensor],
    output_dir: Path,
    dataset: str,
    split: str,
    layer: str,
    device: torch.device,
    k_neighbors: int,
    query_block_size: int,
    reference_block_size: int,
    metadata: dict[str, object],
) -> dict[str, object]:
    """Select neighbors and save their classifier-output OOD Scores."""

    raw = compute_distance_quantities(
        queries=query_embeddings,
        references=reference_embeddings,
        reference_labels=reference_labels,
        query_logits=query_logits,
        query_probabilities=query_probabilities,
        reference_logits=reference_logits,
        reference_probabilities=reference_probabilities,
        centroid=statistics["raw_centroid"],
        class_centroids=statistics["raw_class_centroids"],
        classes=statistics["classes"],
        device=device,
        k_neighbors=k_neighbors,
        query_block_size=query_block_size,
        reference_block_size=reference_block_size,
    )
    standardized_queries = (query_embeddings - statistics["mean"]) / statistics["std"]
    standardized_references = (
        reference_embeddings - statistics["mean"]
    ) / statistics["std"]
    standardized = compute_distance_quantities(
        queries=standardized_queries,
        references=standardized_references,
        reference_labels=reference_labels,
        query_logits=query_logits,
        query_probabilities=query_probabilities,
        reference_logits=reference_logits,
        reference_probabilities=reference_probabilities,
        centroid=statistics["standardized_centroid"],
        class_centroids=statistics["standardized_class_centroids"],
        classes=statistics["classes"],
        device=device,
        k_neighbors=k_neighbors,
        query_block_size=query_block_size,
        reference_block_size=reference_block_size,
    )
    dimension_scale = math.sqrt(query_embeddings.shape[1])
    raw_rms = {
        name: raw[name] / dimension_scale
        for name in ("global_centroid", "class_centroid")
    }

    path = output_dir / dataset / split / f"{layer}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **metadata,
            "version": 3,
            "artifact_type": "knn_neighbor_output_ood_scores",
            "dataset": dataset,
            "split": split,
            "layer": layer,
            "embedding_dimension": query_embeddings.shape[1],
            "query_count": query_embeddings.shape[0],
            "reference_count": reference_embeddings.shape[0],
            "k_neighbors": min(k_neighbors, reference_embeddings.shape[0]),
            "labels": query_labels,
            "score_signs": KNN_SCORE_SIGNS,
            "higher_score_is": "more_id_like",
            "id_statistics": statistics,
            "distances": {
                "raw": raw,
                "raw_rms": raw_rms,
                "id_standardized": standardized,
            },
        },
        path,
    )
    return {
        "dataset": dataset,
        "split": split,
        "layer": layer,
        "embedding_dimension": query_embeddings.shape[1],
        "query_count": query_embeddings.shape[0],
        "path": str(path),
    }


def compute_distance_quantities(
    queries: torch.Tensor,
    references: torch.Tensor,
    reference_labels: torch.Tensor,
    query_logits: torch.Tensor,
    query_probabilities: torch.Tensor,
    reference_logits: torch.Tensor,
    reference_probabilities: torch.Tensor,
    centroid: torch.Tensor,
    class_centroids: torch.Tensor,
    classes: torch.Tensor,
    device: torch.device,
    k_neighbors: int,
    query_block_size: int,
    reference_block_size: int,
) -> dict[str, object]:
    """Select nearest neighbors and compute output-dependent OOD Scores."""

    neighbor_distances, neighbor_indices = blocked_topk_euclidean(
        queries=queries,
        references=references,
        k=k_neighbors,
        device=device,
        query_block_size=query_block_size,
        reference_block_size=reference_block_size,
    )
    centroid_distances = torch.linalg.vector_norm(queries - centroid, dim=1)
    class_distances = torch.cdist(queries, class_centroids)
    nearest_class_distances, nearest_class_indices = class_distances.min(dim=1)
    neighbor_logits = reference_logits[neighbor_indices]
    neighbor_probabilities = reference_probabilities[neighbor_indices]
    neighbor_mean_logits = neighbor_logits.mean(dim=1)
    neighbor_mean_probabilities = neighbor_probabilities.mean(dim=1)
    raw_metrics, ood_scores = knn_score_vectors(
        query_logits=query_logits,
        query_probabilities=query_probabilities,
        neighbor_mean_logits=neighbor_mean_logits,
        neighbor_mean_probabilities=neighbor_mean_probabilities,
        neighbor_probabilities=neighbor_probabilities,
    )
    return {
        "neighbor_distances": neighbor_distances,
        "neighbor_indices": neighbor_indices,
        "neighbor_mean_logits": neighbor_mean_logits,
        "neighbor_mean_probabilities": neighbor_mean_probabilities,
        "raw_metrics": raw_metrics,
        "ood_scores": ood_scores,
        "nearest_neighbor_label": reference_labels[neighbor_indices[:, 0]],
        "global_centroid": centroid_distances,
        "class_centroid": nearest_class_distances,
        "nearest_class_label": classes[nearest_class_indices],
    }


def knn_score_vectors(
    *,
    query_logits: torch.Tensor,
    query_probabilities: torch.Tensor,
    neighbor_mean_logits: torch.Tensor,
    neighbor_mean_probabilities: torch.Tensor,
    neighbor_probabilities: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return raw and ID-oriented scores using neighbor means as predictions."""

    query_logits_array = query_logits.detach().cpu().double().numpy()
    query_probabilities_array = query_probabilities.detach().cpu().double().numpy()
    mean_logits_array = neighbor_mean_logits.detach().cpu().double().numpy()
    mean_probabilities_array = (
        neighbor_mean_probabilities.detach().cpu().double().numpy()
    )
    neighbor_probabilities = neighbor_probabilities.detach().cpu().double()
    predictive_entropy = torch.special.entr(
        neighbor_probabilities.mean(dim=1)
    ).sum(dim=1)
    expected_neighbor_entropy = torch.special.entr(neighbor_probabilities).sum(
        dim=2
    ).mean(dim=1)
    bald = (predictive_entropy - expected_neighbor_entropy).clamp_min(0.0)
    raw_arrays = {
        "student_teacher_kl_divergence": student_teacher_kl_divergence(
            query_probabilities_array,
            mean_probabilities_array,
        ),
        "absolute_max_probability_difference": absolute_max_probability_difference(
            query_probabilities_array,
            mean_probabilities_array,
        ),
        "absolute_energy_gap": absolute_energy_gap(
            query_logits_array,
            mean_logits_array,
        ),
        "student_msp": student_msp(mean_probabilities_array),
        "student_energy": student_energy(mean_logits_array),
        "ensemble_predictive_entropy": predictive_entropy.numpy(),
        "ensemble_bald": bald.numpy(),
    }
    raw_metrics = {
        name: torch.from_numpy(np.asarray(values))
        for name, values in raw_arrays.items()
    }
    ood_scores = {
        name: values * KNN_SCORE_SIGNS[name]
        for name, values in raw_metrics.items()
    }
    return raw_metrics, ood_scores


def blocked_topk_euclidean(
    queries: torch.Tensor,
    references: torch.Tensor,
    k: int,
    device: torch.device,
    query_block_size: int,
    reference_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact top-k Euclidean distances using bounded matrix blocks."""

    if queries.ndim != 2 or references.ndim != 2:
        raise ValueError("queries and references must be two-dimensional")
    if queries.shape[1] != references.shape[1]:
        raise ValueError("queries and references must have the same dimension")
    if references.shape[0] == 0:
        raise ValueError("references must not be empty")
    effective_k = min(k, references.shape[0])
    all_distances: list[torch.Tensor] = []
    all_indices: list[torch.Tensor] = []
    for query_start in range(0, queries.shape[0], query_block_size):
        query = queries[query_start : query_start + query_block_size].to(device)
        query_norm = query.square().sum(dim=1, keepdim=True)
        best_distances = torch.full(
            (query.shape[0], effective_k),
            float("inf"),
            device=device,
        )
        best_indices = torch.full(
            (query.shape[0], effective_k),
            -1,
            dtype=torch.long,
            device=device,
        )
        for reference_start in range(0, references.shape[0], reference_block_size):
            reference = references[
                reference_start : reference_start + reference_block_size
            ].to(device)
            squared_distances = (
                query_norm
                + reference.square().sum(dim=1).unsqueeze(0)
                - 2.0 * query @ reference.T
            ).clamp_min_(0.0)
            reference_indices = torch.arange(
                reference_start,
                reference_start + reference.shape[0],
                device=device,
            ).expand(query.shape[0], -1)
            candidate_distances = torch.cat((best_distances, squared_distances), dim=1)
            candidate_indices = torch.cat((best_indices, reference_indices), dim=1)
            best_distances, positions = candidate_distances.topk(
                effective_k,
                dim=1,
                largest=False,
                sorted=True,
            )
            best_indices = candidate_indices.gather(1, positions)
        all_distances.append(best_distances.sqrt().cpu())
        all_indices.append(best_indices.cpu())
    return torch.cat(all_distances, dim=0), torch.cat(all_indices, dim=0)
