"""Export layer4 FAISS k-NN neighbor-output OOD Scores."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np
import torch
from torch import nn

from distill_ood_detection.config import EmbeddingDistanceConfig, PerturbationConfig
from distill_ood_detection.datasets.inference import (
    NamedLoader,
    build_in_distribution_full_train_loader,
    build_in_distribution_test_loader,
    build_ood_loaders,
    dataset_normalization,
)
from distill_ood_detection.distillation.perturbation import (
    build_sequential_clipping_batch,
    forward_to_clipping_start,
    sample_pixel_augmentation,
)
from distill_ood_detection.evaluation.ood_scores import (
    absolute_energy_gap,
    absolute_max_probability_difference,
    student_energy,
    student_msp,
    student_teacher_kl_divergence,
)
from distill_ood_detection.evaluation.nearest_neighbors import FaissExactL2Index
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

KNN_LAYER = "layer4"


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
    image_normalization = dataset_normalization(config.dataset)

    embedding_artifacts: list[dict[str, object]] = []
    extraction_loaders = loaders[:1] if config.query_perturbed else loaders
    for loader_index, named_loader in enumerate(extraction_loaders):
        is_reference = loader_index == 0
        apply_perturbation = config.perturbation is not None and (
            is_reference or config.query_perturbed
        )
        draw_count = 1
        print(f"Extracting pooled embeddings: {named_loader.name}", flush=True)
        embedding_artifacts.extend(
            export_loader_embeddings(
                teacher=teacher,
                layer=KNN_LAYER,
                named_loader=named_loader,
                output_dir=embeddings_dir,
                device=device,
                shard_size=config.embedding_shard_size,
                perturbation=config.perturbation,
                apply_perturbation=apply_perturbation,
                draw_count=draw_count,
                image_normalization=image_normalization,
                metadata={
                    "experiment_name": config.experiment_name,
                    "model": "teacher",
                    "teacher": asdict(config.teacher),
                    "pooling": "gap",
                    "perturbation": (
                        asdict(config.perturbation)
                        if config.perturbation is not None
                        else None
                    ),
                    "apply_perturbation": apply_perturbation,
                    "draw_count": draw_count,
                    "dtype": "float32",
                },
            )
        )

    embedding_manifest = {
        "version": 4,
        "artifact_type": "knn_teacher_embeddings_and_outputs",
        "experiment_name": config.experiment_name,
        "dataset": asdict(config.dataset),
        "teacher": asdict(config.teacher),
        "layer": KNN_LAYER,
        "pooling": "gap",
        "perturbation": (
            asdict(config.perturbation)
            if config.perturbation is not None
            else None
        ),
        "query_perturbed": config.query_perturbed,
        "dtype": "float32",
        "seed": config.seed,
        "shard_size": config.embedding_shard_size,
        "artifacts": embedding_artifacts,
    }
    write_json(embeddings_dir / "manifest.json", embedding_manifest)

    reference_name = f"{config.dataset.name}_train"
    query_loaders = loaders[1:]
    print(f"Loading ID reference embeddings: {KNN_LAYER}", flush=True)
    reference_embeddings, reference_labels = load_embedding_shards(
        embedding_artifacts,
        dataset=reference_name,
        layer=KNN_LAYER,
    )
    reference_logits, reference_probabilities = load_output_shards(
        embedding_artifacts,
        dataset=reference_name,
        layer=KNN_LAYER,
    )
    gpu_resources = faiss.StandardGpuResources() if device.type == "cuda" else None
    index = FaissExactL2Index(
        reference_embeddings,
        device,
        gpu_resources=gpu_resources,
    )
    search_metadata = {
        "library": "faiss",
        "version": faiss.__version__,
        "index": "IndexFlatL2",
        "device": index.device,
        "exact": True,
        "faiss_distance": "squared_l2",
        "stored_distance": "euclidean",
    }
    distance_artifacts: list[dict[str, object]] = []
    for named_loader in query_loaders:
        print(
            f"Computing exact FAISS distances: {named_loader.name} {KNN_LAYER}",
            flush=True,
        )
        artifact_metadata = {
            "experiment_name": config.experiment_name,
            "teacher": asdict(config.teacher),
            "id_reference_dataset": reference_name,
            "pooling": "gap",
            "neighbor_search": search_metadata,
        }
        if config.query_perturbed:
            if config.perturbation is None:
                raise RuntimeError("query_perturbed requires a perturbation")
            artifact = export_perturbed_loader_distances(
                teacher=teacher,
                named_loader=named_loader,
                device=device,
                perturbation=config.perturbation,
                image_normalization=image_normalization,
                index=index,
                reference_labels=reference_labels,
                reference_logits=reference_logits,
                reference_probabilities=reference_probabilities,
                output_dir=distances_dir,
                layer=KNN_LAYER,
                k_neighbors=config.k_neighbors,
                search_batch_size=config.search_batch_size,
                metadata=artifact_metadata,
            )
        else:
            query_embeddings, query_labels = load_embedding_shards(
                embedding_artifacts,
                dataset=named_loader.name,
                layer=KNN_LAYER,
            )
            query_logits, query_probabilities = load_output_shards(
                embedding_artifacts,
                dataset=named_loader.name,
                layer=KNN_LAYER,
            )
            artifact = export_dataset_distances(
                query_embeddings=query_embeddings,
                query_labels=query_labels,
                query_logits=query_logits,
                query_probabilities=query_probabilities,
                index=index,
                reference_labels=reference_labels,
                reference_logits=reference_logits,
                reference_probabilities=reference_probabilities,
                output_dir=distances_dir,
                dataset=named_loader.name,
                split=named_loader.split,
                layer=KNN_LAYER,
                k_neighbors=config.k_neighbors,
                search_batch_size=config.search_batch_size,
                metadata=artifact_metadata,
            )
        distance_artifacts.append(artifact)

    distance_manifest = {
        "version": 4,
        "artifact_type": "knn_neighbor_output_ood_scores",
        "experiment_name": config.experiment_name,
        "dataset": asdict(config.dataset),
        "teacher": asdict(config.teacher),
        "perturbation": (
            asdict(config.perturbation)
            if config.perturbation is not None
            else None
        ),
        "query_perturbed": config.query_perturbed,
        "layer": KNN_LAYER,
        "id_reference_dataset": reference_name,
        "k_neighbors": config.k_neighbors,
        "search_batch_size": config.search_batch_size,
        "neighbor_search": search_metadata,
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
    layer: str,
    named_loader: NamedLoader,
    output_dir: Path,
    device: torch.device,
    shard_size: int,
    metadata: dict[str, object],
    perturbation: PerturbationConfig | None = None,
    apply_perturbation: bool = False,
    draw_count: int = 1,
    image_normalization: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    | None = None,
) -> list[dict[str, object]]:
    """Export layer4 GAP representations and clean teacher outputs in shards."""

    if draw_count <= 0:
        raise ValueError("draw_count must be positive")
    if apply_perturbation and perturbation is None:
        raise ValueError("apply_perturbation requires a perturbation configuration")
    if (
        perturbation is not None
        and perturbation.method == "pixel_augmentation"
        and apply_perturbation
        and image_normalization is None
    ):
        raise ValueError("pixel augmentation requires image normalization")

    collector = (
        PooledActivationCollector(teacher, layer)
        if perturbation is None or perturbation.method == "pixel_augmentation"
        else None
    )
    pending_embeddings: list[torch.Tensor] = []
    pending_labels: list[torch.Tensor] = []
    pending_logits: list[torch.Tensor] = []
    pending_count = 0
    sample_offset = 0
    shard_index = 0
    artifacts: list[dict[str, object]] = []
    teacher.eval()
    try:
        for images, labels in named_loader.loader:
            images = images.to(device, non_blocking=True)
            embeddings, logits = _knn_batch_representations(
                teacher=teacher,
                images=images,
                collector=collector,
                perturbation=perturbation,
                apply_perturbation=apply_perturbation,
                draw_count=draw_count,
                image_normalization=image_normalization,
            )
            pending_embeddings.append(embeddings.cpu())
            pending_labels.append(labels.cpu())
            pending_logits.append(logits.cpu())
            pending_count += labels.shape[0]
            if pending_count >= shard_size:
                artifacts.append(
                    _write_embedding_shard(
                        pending_embeddings=pending_embeddings,
                        pending_labels=pending_labels,
                        pending_logits=pending_logits,
                        output_dir=output_dir,
                        named_loader=named_loader,
                        layer=layer,
                        shard_index=shard_index,
                        sample_offset=sample_offset,
                        metadata=metadata,
                    )
                )
                sample_offset += pending_count
                shard_index += 1
                pending_embeddings = []
                pending_labels = []
                pending_logits = []
                pending_count = 0
        if pending_count:
            artifacts.append(
                _write_embedding_shard(
                    pending_embeddings=pending_embeddings,
                    pending_labels=pending_labels,
                    pending_logits=pending_logits,
                    output_dir=output_dir,
                    named_loader=named_loader,
                    layer=layer,
                    shard_index=shard_index,
                    sample_offset=sample_offset,
                    metadata=metadata,
                )
            )
    finally:
        if collector is not None:
            collector.close()
    return artifacts


def _knn_batch_representations(
    *,
    teacher: nn.Module,
    images: torch.Tensor,
    collector: PooledActivationCollector | None,
    perturbation: PerturbationConfig | None,
    apply_perturbation: bool,
    draw_count: int,
    image_normalization: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-draw post-GAP representations and clean teacher logits."""

    if perturbation is None:
        if collector is None:
            raise RuntimeError("A layer4 collector is required for clean extraction")
        collector.clear()
        logits = teacher(images)
        return collector.embedding(), logits

    if perturbation.method == "pixel_augmentation":
        if collector is None:
            raise RuntimeError("A layer4 collector is required for pixel augmentation")
        collector.clear()
        clean_logits = teacher(images)
        if not apply_perturbation:
            return collector.embedding(), clean_logits
        if image_normalization is None:
            raise ValueError("pixel augmentation requires image normalization")
        draws: list[torch.Tensor] = []
        for _ in range(draw_count):
            pixel_batch = sample_pixel_augmentation(
                images,
                perturbation,
                image_normalization,
            )
            collector.clear()
            teacher(pixel_batch.perturbed_images)
            draws.append(collector.embedding())
        stacked = torch.stack(draws, dim=1)
        return (stacked[:, 0] if draw_count == 1 else stacked), clean_logits

    if perturbation.method == "clipping":
        start_features = forward_to_clipping_start(teacher, images, perturbation)
        clean_batch = build_sequential_clipping_batch(
            teacher,
            start_features,
            perturbation,
            apply_perturbation=False,
        )
        if not apply_perturbation:
            return (
                clean_batch.final_features.mean(dim=(-2, -1)),
                clean_batch.teacher_logits,
            )
        draw_chunks: list[torch.Tensor] = []
        max_draw_examples = 4096
        draws_per_chunk = max(1, max_draw_examples // images.shape[0])
        remaining = draw_count
        while remaining:
            current_draws = min(remaining, draws_per_chunk)
            clipped_batch = build_sequential_clipping_batch(
                teacher,
                start_features.repeat_interleave(current_draws, dim=0),
                perturbation,
                apply_perturbation=True,
            )
            pooled = clipped_batch.final_features.mean(dim=(-2, -1)).reshape(
                images.shape[0],
                current_draws,
                -1,
            )
            draw_chunks.append(pooled)
            remaining -= current_draws
        stacked = torch.cat(draw_chunks, dim=1)
        return (stacked[:, 0] if draw_count == 1 else stacked), clean_batch.teacher_logits

    raise ValueError(f"Unsupported FAISS k-NN perturbation: {perturbation.method}")


def _write_embedding_shard(
    pending_embeddings: list[torch.Tensor],
    pending_labels: list[torch.Tensor],
    pending_logits: list[torch.Tensor],
    output_dir: Path,
    named_loader: NamedLoader,
    layer: str,
    shard_index: int,
    sample_offset: int,
    metadata: dict[str, object],
) -> dict[str, object]:
    """Write one aligned embedding and teacher-output shard."""

    labels = torch.cat(pending_labels, dim=0)
    logits = torch.cat(pending_logits, dim=0).float()
    probabilities = torch.softmax(logits, dim=1)
    sample_count = labels.shape[0]
    embeddings = torch.cat(pending_embeddings, dim=0).float()
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
            "version": 4,
            "artifact_type": "knn_teacher_embedding_and_output_shard",
            "dataset": named_loader.name,
            "split": named_loader.split,
            "layer": layer,
            "sample_offset": sample_offset,
            "sample_count": sample_count,
            "embedding_dimension": embeddings.shape[-1],
            "draw_count": 1 if embeddings.ndim == 2 else embeddings.shape[1],
            "embeddings": embeddings,
            "logits": logits,
            "probabilities": probabilities,
            "labels": labels,
        },
        path,
    )
    return {
        "dataset": named_loader.name,
        "split": named_loader.split,
        "layer": layer,
        "sample_offset": sample_offset,
        "sample_count": sample_count,
        "embedding_dimension": embeddings.shape[-1],
        "draw_count": 1 if embeddings.ndim == 2 else embeddings.shape[1],
        "path": str(path),
    }


class PooledActivationCollector:
    """Capture GAP embeddings from one named teacher layer."""

    def __init__(self, teacher: nn.Module, layer_name: str) -> None:
        self._embedding: torch.Tensor | None = None
        self._layer_name = layer_name
        try:
            layer = teacher.get_submodule(layer_name)
        except AttributeError as error:
            raise ValueError(f"Teacher does not have layer: {layer_name}") from error
        self._hook = layer.register_forward_hook(self._capture)

    def clear(self) -> None:
        """Clear embeddings captured during the previous forward pass."""

        self._embedding = None

    def embedding(self) -> torch.Tensor:
        """Return one captured pooled embedding batch."""

        if self._embedding is None:
            raise RuntimeError(f"Feature hook did not run for layer: {self._layer_name}")
        return self._embedding

    def close(self) -> None:
        """Remove all registered hooks."""

        self._hook.remove()

    def _capture(
        self,
        _module: nn.Module,
        _inputs: tuple[object, ...],
        output: torch.Tensor,
    ) -> None:
        if output.ndim != 4:
            raise ValueError(
                f"Expected a 4D activation at {self._layer_name}; "
                f"got {tuple(output.shape)}"
            )
        self._embedding = output.detach().mean(dim=(-2, -1))


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


def export_dataset_distances(
    query_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    query_logits: torch.Tensor,
    query_probabilities: torch.Tensor,
    index: FaissExactL2Index,
    reference_labels: torch.Tensor,
    reference_logits: torch.Tensor,
    reference_probabilities: torch.Tensor,
    output_dir: Path,
    dataset: str,
    split: str,
    layer: str,
    k_neighbors: int,
    search_batch_size: int,
    metadata: dict[str, object],
) -> dict[str, object]:
    """Select neighbors and save their classifier-output OOD Scores."""

    raw = compute_distance_quantities(
        queries=query_embeddings,
        index=index,
        reference_labels=reference_labels,
        query_logits=query_logits,
        query_probabilities=query_probabilities,
        reference_logits=reference_logits,
        reference_probabilities=reference_probabilities,
        k_neighbors=k_neighbors,
        search_batch_size=search_batch_size,
    )

    return _write_distance_artifact(
        raw=raw,
        query_labels=query_labels,
        output_dir=output_dir,
        dataset=dataset,
        split=split,
        layer=layer,
        embedding_dimension=query_embeddings.shape[-1],
        query_draw_count=(
            1 if query_embeddings.ndim == 2 else query_embeddings.shape[1]
        ),
        reference_count=index.reference_count,
        k_neighbors=k_neighbors,
        metadata=metadata,
    )


@torch.inference_mode()
def export_perturbed_loader_distances(
    *,
    teacher: nn.Module,
    named_loader: NamedLoader,
    device: torch.device,
    perturbation: PerturbationConfig,
    image_normalization: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    index: FaissExactL2Index,
    reference_labels: torch.Tensor,
    reference_logits: torch.Tensor,
    reference_probabilities: torch.Tensor,
    output_dir: Path,
    layer: str,
    k_neighbors: int,
    search_batch_size: int,
    metadata: dict[str, object],
) -> dict[str, object]:
    """Search perturbed query draws batchwise without saving their embeddings."""

    collector = (
        PooledActivationCollector(teacher, layer)
        if perturbation.method == "pixel_augmentation"
        else None
    )
    labels: list[torch.Tensor] = []
    quantity_batches: list[dict[str, object]] = []
    embedding_dimension: int | None = None
    try:
        for images, batch_labels in named_loader.loader:
            query_embeddings, query_logits = _knn_batch_representations(
                teacher=teacher,
                images=images.to(device, non_blocking=True),
                collector=collector,
                perturbation=perturbation,
                apply_perturbation=True,
                draw_count=perturbation.evaluation_draws,
                image_normalization=image_normalization,
            )
            embedding_dimension = query_embeddings.shape[-1]
            query_probabilities = torch.softmax(query_logits, dim=1)
            quantity_batches.append(
                compute_distance_quantities(
                    queries=query_embeddings,
                    index=index,
                    reference_labels=reference_labels,
                    query_logits=query_logits,
                    query_probabilities=query_probabilities,
                    reference_logits=reference_logits,
                    reference_probabilities=reference_probabilities,
                    k_neighbors=k_neighbors,
                    search_batch_size=search_batch_size,
                )
            )
            labels.append(batch_labels.cpu())
    finally:
        if collector is not None:
            collector.close()
    if embedding_dimension is None:
        raise ValueError(f"Query loader is empty: {named_loader.name}")
    return _write_distance_artifact(
        raw=_concatenate_distance_quantities(quantity_batches),
        query_labels=torch.cat(labels),
        output_dir=output_dir,
        dataset=named_loader.name,
        split=named_loader.split,
        layer=layer,
        embedding_dimension=embedding_dimension,
        query_draw_count=perturbation.evaluation_draws,
        reference_count=index.reference_count,
        k_neighbors=k_neighbors,
        metadata=metadata,
    )


def _concatenate_distance_quantities(
    batches: list[dict[str, object]],
) -> dict[str, object]:
    """Concatenate batchwise neighbor quantities along the sample dimension."""

    if not batches:
        raise ValueError("distance quantity batches must not be empty")
    direct_keys = (
        "neighbor_distances",
        "neighbor_indices",
        "neighbor_mean_logits",
        "neighbor_mean_probabilities",
        "nearest_neighbor_label",
    )
    nested_keys = ("raw_metrics", "ood_scores")
    combined: dict[str, object] = {}
    for key in direct_keys:
        combined[key] = torch.cat([batch[key] for batch in batches])
    for key in nested_keys:
        first = batches[0][key]
        if not isinstance(first, dict):
            raise TypeError(f"Expected a metric mapping for {key}")
        combined[key] = {
            metric_name: torch.cat(
                [batch[key][metric_name] for batch in batches]  # type: ignore[index]
            )
            for metric_name in first
        }
    return combined


def _write_distance_artifact(
    *,
    raw: dict[str, object],
    query_labels: torch.Tensor,
    output_dir: Path,
    dataset: str,
    split: str,
    layer: str,
    embedding_dimension: int,
    query_draw_count: int,
    reference_count: int,
    k_neighbors: int,
    metadata: dict[str, object],
) -> dict[str, object]:
    """Write one complete neighbor-output artifact."""

    path = output_dir / dataset / split / f"{layer}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **metadata,
            "version": 4,
            "artifact_type": "knn_neighbor_output_ood_scores",
            "dataset": dataset,
            "split": split,
            "layer": layer,
            "embedding_dimension": embedding_dimension,
            "query_draw_count": query_draw_count,
            "query_count": query_labels.shape[0],
            "reference_count": reference_count,
            "k_neighbors": min(k_neighbors, reference_count),
            "labels": query_labels,
            "score_signs": KNN_SCORE_SIGNS,
            "higher_score_is": "more_id_like",
            "distances": {"raw": raw},
        },
        path,
    )
    return {
        "dataset": dataset,
        "split": split,
        "layer": layer,
        "embedding_dimension": embedding_dimension,
        "query_draw_count": query_draw_count,
        "query_count": query_labels.shape[0],
        "path": str(path),
    }


def compute_distance_quantities(
    queries: torch.Tensor,
    index: FaissExactL2Index,
    reference_labels: torch.Tensor,
    query_logits: torch.Tensor,
    query_probabilities: torch.Tensor,
    reference_logits: torch.Tensor,
    reference_probabilities: torch.Tensor,
    k_neighbors: int,
    search_batch_size: int,
) -> dict[str, object]:
    """Select nearest neighbors and compute output-dependent OOD Scores."""

    if queries.ndim not in {2, 3}:
        raise ValueError(
            "queries must have shape (samples, dimensions) or "
            "(samples, draws, dimensions)"
        )
    query_count = queries.shape[0]
    draw_count = 1 if queries.ndim == 2 else queries.shape[1]
    flat_queries = queries.reshape(-1, queries.shape[-1])
    flat_neighbor_distances, flat_neighbor_indices = index.search(
        flat_queries,
        k=k_neighbors,
        batch_size=search_batch_size,
    )
    effective_k = flat_neighbor_indices.shape[1]
    neighbor_distances_by_draw = flat_neighbor_distances.reshape(
        query_count,
        draw_count,
        effective_k,
    )
    neighbor_indices_by_draw = flat_neighbor_indices.reshape(
        query_count,
        draw_count,
        effective_k,
    )
    neighbor_logits = reference_logits[neighbor_indices_by_draw].reshape(
        query_count,
        draw_count * effective_k,
        -1,
    )
    neighbor_probabilities = reference_probabilities[
        neighbor_indices_by_draw
    ].reshape(query_count, draw_count * effective_k, -1)
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
        "neighbor_distances": (
            neighbor_distances_by_draw[:, 0]
            if draw_count == 1
            else neighbor_distances_by_draw
        ),
        "neighbor_indices": (
            neighbor_indices_by_draw[:, 0]
            if draw_count == 1
            else neighbor_indices_by_draw
        ),
        "neighbor_mean_logits": neighbor_mean_logits,
        "neighbor_mean_probabilities": neighbor_mean_probabilities,
        "raw_metrics": raw_metrics,
        "ood_scores": ood_scores,
        "nearest_neighbor_label": reference_labels[
            neighbor_indices_by_draw[:, 0, 0]
        ],
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
