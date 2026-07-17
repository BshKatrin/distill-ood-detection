"""Export OOD Scores from activation-subspace inference artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from distill_ood_detection.config import ExperimentConfig
from distill_ood_detection.evaluation.ood_metrics import ood_detection_metrics
from distill_ood_detection.evaluation.ood_scores import (
    absolute_energy_gap,
    absolute_max_probability_difference,
    energy,
    energy_gap,
    logit_l2_distance,
    max_probability_difference,
    student_energy,
    student_msp,
    student_teacher_kl_divergence,
)
from distill_ood_detection.models.teacher import load_teacher
from distill_ood_detection.utils import resolve_device, set_seed, write_json

CheckpointSelection = Literal["best", "latest"]
DECISIVE_SCORE_SIGNS = {
    "msp": +1,
    "energy": +1,
    "student_teacher_kl_divergence": -1,
    "max_probability_difference": +1,
    "absolute_max_probability_difference": -1,
    "logit_l2_distance": -1,
    "energy_gap": +1,
    "absolute_energy_gap": -1,
    "student_msp": +1,
    "student_energy": +1,
}
INSIGNIFICANT_SCORE_SIGNS = {
    "raw_reconstruction_error": -1,
    "relative_reconstruction_error": -1,
    "cosine_similarity": +1,
}


def run_activation_subspace_score_export(
    configs: list[ExperimentConfig],
    teacher_embedding_dir: Path,
    checkpoint: CheckpointSelection = "best",
) -> dict[str, object]:
    """Export raw metrics and ID-oriented OOD Scores for trained students."""

    _validate_configs(configs)
    base_config = configs[0]
    set_seed(base_config.training.defaults.seed)
    device = resolve_device(base_config.training.defaults.device)
    teacher = load_teacher(base_config.teacher, device)
    classifier = getattr(teacher, "fc", None)
    if not isinstance(classifier, nn.Linear):
        raise ValueError("Activation-subspace scores require a ResNet linear fc head")

    teacher_data = _load_teacher_data(
        config=base_config,
        teacher_embedding_dir=teacher_embedding_dir,
        classifier=classifier,
        device=device,
    )
    student_manifests = []
    for config in configs:
        student_manifests.append(
            _export_student_scores(
                config=config,
                teacher_data=teacher_data,
                checkpoint=checkpoint,
                device=device,
            )
        )

    manifest = {
        "version": 1,
        "artifact_type": "activation_subspace_ood_scores",
        "id_dataset": base_config.dataset.name,
        "checkpoint_selection": checkpoint,
        "teacher_embedding_manifest": str(teacher_embedding_dir / "manifest.json"),
        "student_manifests": student_manifests,
    }
    write_json(teacher_embedding_dir / "score_manifest.json", manifest)
    return manifest


def decisive_score_vectors(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return raw and sign-adjusted decisive OOD Score vectors."""

    teacher_logits_array = teacher_logits.detach().cpu().numpy()
    student_logits_array = student_logits.detach().cpu().numpy()
    teacher_probabilities = torch.softmax(teacher_logits.double(), dim=1).cpu().numpy()
    student_probabilities = torch.softmax(student_logits.double(), dim=1).cpu().numpy()
    teacher_msp = teacher_probabilities.max(axis=1)

    raw_arrays = {
        "msp": teacher_msp,
        "energy": energy(teacher_logits_array),
        "student_teacher_kl_divergence": student_teacher_kl_divergence(
            teacher_probabilities,
            student_probabilities,
        ),
        "max_probability_difference": max_probability_difference(
            teacher_probabilities,
            student_probabilities,
        ),
        "absolute_max_probability_difference": absolute_max_probability_difference(
            teacher_probabilities,
            student_probabilities,
        ),
        "logit_l2_distance": logit_l2_distance(
            teacher_logits_array,
            student_logits_array,
        ),
        "energy_gap": energy_gap(teacher_logits_array, student_logits_array),
        "absolute_energy_gap": absolute_energy_gap(
            teacher_logits_array,
            student_logits_array,
        ),
        "student_msp": student_msp(student_probabilities),
        "student_energy": student_energy(student_logits_array),
    }
    raw_metrics = {
        name: torch.from_numpy(np.asarray(values))
        for name, values in raw_arrays.items()
    }
    scores = {
        name: values * DECISIVE_SCORE_SIGNS[name]
        for name, values in raw_metrics.items()
    }
    return raw_metrics, scores


def insignificant_score_vectors(
    targets: torch.Tensor,
    reconstructions: torch.Tensor,
    eps: float = 1.0e-12,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return reconstruction metrics and ID-oriented OOD Score vectors."""

    if targets.shape != reconstructions.shape:
        raise ValueError(
            "targets and reconstructions must have the same shape, got "
            f"{tuple(targets.shape)} and {tuple(reconstructions.shape)}"
        )
    targets = targets.double()
    reconstructions = reconstructions.double()
    residuals = reconstructions - targets
    raw_metrics = {
        "raw_reconstruction_error": residuals.pow(2).mean(dim=1),
        "relative_reconstruction_error": residuals.norm(dim=1)
        / targets.norm(dim=1).clamp_min(eps),
        "cosine_similarity": F.cosine_similarity(
            reconstructions,
            targets,
            dim=1,
            eps=eps,
        ),
    }
    scores = {
        name: values * INSIGNIFICANT_SCORE_SIGNS[name]
        for name, values in raw_metrics.items()
    }
    return raw_metrics, scores


def _load_teacher_data(
    *,
    config: ExperimentConfig,
    teacher_embedding_dir: Path,
    classifier: nn.Linear,
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor | Path]]:
    datasets = {}
    dataset_names = [
        f"{config.dataset.name}_test",
        *[
            f"{ood.name}_{ood.split}"
            for ood in config.dataset.ood_datasets
        ],
    ]
    for dataset_name in dataset_names:
        embedding_path = teacher_embedding_dir / dataset_name / "embeddings.pt"
        artifact = torch.load(embedding_path, map_location="cpu", weights_only=True)
        embeddings = artifact["embeddings"]
        datasets[dataset_name] = {
            "embeddings": embeddings,
            "labels": artifact["labels"],
            "teacher_logits": _classifier_logits(
                classifier=classifier,
                embeddings=embeddings,
                batch_size=config.dataset.batch_size,
                device=device,
            ),
            "embedding_path": embedding_path,
        }
    return datasets


def _classifier_logits(
    *,
    classifier: nn.Linear,
    embeddings: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    outputs = []
    classifier.to(device)
    classifier.eval()
    with torch.no_grad():
        for start in range(0, embeddings.shape[0], batch_size):
            outputs.append(
                classifier(embeddings[start : start + batch_size].to(device)).cpu()
            )
    return torch.cat(outputs, dim=0)


def _project_component_coordinates(
    *,
    embeddings: torch.Tensor,
    right_basis: torch.Tensor,
    decisive_dimension: int,
    component: str,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if component == "decisive":
        basis = right_basis[:decisive_dimension].to(device)
    else:
        basis = right_basis[decisive_dimension:].to(device)
    coordinates = []
    with torch.no_grad():
        for start in range(0, embeddings.shape[0], batch_size):
            batch = embeddings[start : start + batch_size].to(device)
            coordinates.append((batch @ basis.T).cpu())
    return torch.cat(coordinates, dim=0)


def _export_student_scores(
    *,
    config: ExperimentConfig,
    teacher_data: dict[str, dict[str, torch.Tensor | Path]],
    checkpoint: CheckpointSelection,
    device: torch.device,
) -> dict[str, object]:
    experiment_dir = Path(config.run_dir)
    subspace_config = config.strategy.activation_subspace
    component = subspace_config.component
    target = subspace_config.target
    subspace_path = experiment_dir / "activation_subspace.pt"
    subspace = torch.load(subspace_path, map_location="cpu", weights_only=True)
    right_basis = subspace["right_basis"]
    decisive_dimension = int(subspace["decisive_dimension"])
    inference_dir = experiment_dir / "activation_subspace_inference"
    output_dir = experiment_dir / "activation_subspace_scores"
    artifacts = []
    scores_by_dataset: dict[str, dict[str, torch.Tensor]] = {}

    for dataset_name, teacher_artifact in teacher_data.items():
        inference_path = inference_dir / dataset_name / f"student_{checkpoint}.pt"
        inference = torch.load(inference_path, map_location="cpu", weights_only=True)
        labels = teacher_artifact["labels"]
        if not isinstance(labels, torch.Tensor):
            raise TypeError("Teacher embedding labels must be a tensor")
        if not torch.equal(labels, inference["labels"]):
            raise ValueError(f"Teacher and student labels differ for {dataset_name}")

        if target == "projected_logits":
            teacher_logits = teacher_artifact["teacher_logits"]
            if not isinstance(teacher_logits, torch.Tensor):
                raise TypeError("Teacher logits must be a tensor")
            raw_metrics, scores = decisive_score_vectors(
                teacher_logits,
                inference["logits"],
            )
            signs = DECISIVE_SCORE_SIGNS
        else:
            embeddings = teacher_artifact["embeddings"]
            if not isinstance(embeddings, torch.Tensor):
                raise TypeError("Teacher embeddings must be a tensor")
            targets = _project_component_coordinates(
                embeddings=embeddings,
                right_basis=right_basis,
                decisive_dimension=decisive_dimension,
                component=component,
                batch_size=config.dataset.batch_size,
                device=device,
            )
            raw_metrics, scores = insignificant_score_vectors(
                targets,
                inference["reconstructed_coordinates"],
            )
            signs = INSIGNIFICANT_SCORE_SIGNS

        path = output_dir / dataset_name / f"student_{checkpoint}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "raw_metrics": raw_metrics,
                "ood_scores": scores,
                "labels": labels,
                "metadata": {
                    "experiment_name": config.experiment_name,
                    "id_dataset": config.dataset.name,
                    "dataset": dataset_name,
                    "component": component,
                    "target": target,
                    "checkpoint": checkpoint,
                    "score_signs": signs,
                    "higher_score_is": "more_id_like",
                    "inference_path": str(inference_path),
                    "teacher_embedding_path": str(
                        teacher_artifact["embedding_path"]
                    ),
                    "subspace_path": str(subspace_path),
                },
            },
            path,
        )
        scores_by_dataset[dataset_name] = scores
        artifacts.append(
            {
                "dataset": dataset_name,
                "samples": labels.shape[0],
                "score_names": list(scores),
                "path": str(path),
            }
        )

    id_dataset_name = f"{config.dataset.name}_test"
    ood_dataset_names = [
        f"{ood.name}_{ood.split}"
        for ood in config.dataset.ood_datasets
    ]
    metrics = detection_metrics_by_score(
        id_scores=scores_by_dataset[id_dataset_name],
        ood_scores_by_dataset={
            name: scores_by_dataset[name]
            for name in ood_dataset_names
        },
    )
    manifest = {
        "version": 1,
        "artifact_type": "activation_subspace_student_ood_scores",
        "experiment_name": config.experiment_name,
        "component": component,
        "target": target,
        "checkpoint_selection": checkpoint,
        "score_signs": signs,
        "id_dataset": id_dataset_name,
        "metrics": metrics,
        "artifacts": artifacts,
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    return {
        "experiment_name": config.experiment_name,
        "component": component,
        "target": target,
        "manifest_path": str(manifest_path),
        "metrics": metrics,
    }


def detection_metrics_by_score(
    *,
    id_scores: dict[str, torch.Tensor],
    ood_scores_by_dataset: dict[str, dict[str, torch.Tensor]],
) -> dict[str, object]:
    """Compute per-OOD and macro metrics for every exported score vector."""

    results = {}
    for score_name, id_values in id_scores.items():
        per_ood = []
        for dataset_name, dataset_scores in ood_scores_by_dataset.items():
            ood_values = dataset_scores[score_name]
            labels = np.concatenate(
                [
                    np.ones(id_values.shape[0], dtype=int),
                    np.zeros(ood_values.shape[0], dtype=int),
                ]
            )
            values = np.concatenate(
                [
                    id_values.detach().cpu().numpy(),
                    ood_values.detach().cpu().numpy(),
                ]
            )
            per_ood.append(
                {
                    "ood_dataset": dataset_name,
                    "metrics": ood_detection_metrics(labels, values),
                }
            )
        results[score_name] = {
            "ood": per_ood,
            "macro": {
                "roc_auc": float(
                    np.mean([item["metrics"]["roc_auc"] for item in per_ood])
                ),
                "fpr_at_95_tpr": float(
                    np.mean(
                        [
                            item["metrics"]["fpr_at_95_tpr"]
                            for item in per_ood
                        ]
                    )
                ),
            },
        }
    return results


def _validate_configs(configs: list[ExperimentConfig]) -> None:
    if not configs:
        raise ValueError("At least one activation-subspace config is required")
    base = configs[0]
    for config in configs:
        if config.strategy.name != "activation_subspace":
            raise ValueError(
                "Activation-subspace score export requires strategy.name: "
                "activation_subspace"
            )
        if config.dataset != base.dataset or config.teacher != base.teacher:
            raise ValueError(
                "All configs in one score export must share dataset and teacher"
            )
