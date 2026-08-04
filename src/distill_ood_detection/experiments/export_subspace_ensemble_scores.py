"""Export predictive-entropy and BALD OOD Scores for subspace ensembles."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch

from distill_ood_detection.config import ExperimentConfig
from distill_ood_detection.evaluation.ood_metrics import ood_detection_metrics
from distill_ood_detection.evaluation.ood_scores import (
    ENSEMBLE_BALD,
    ENSEMBLE_PREDICTIVE_ENTROPY,
    SIGNS,
    ensemble_bald,
    ensemble_predictive_entropy,
)
from distill_ood_detection.utils import write_json

CheckpointSelection = Literal["best", "latest"]
SUBSPACE_ENSEMBLE_SCORE_SIGNS = {
    ENSEMBLE_PREDICTIVE_ENTROPY: SIGNS[ENSEMBLE_PREDICTIVE_ENTROPY],
    ENSEMBLE_BALD: SIGNS[ENSEMBLE_BALD],
}


def run_subspace_ensemble_score_export(
    config: ExperimentConfig,
    checkpoint: CheckpointSelection = "best",
) -> dict[str, object]:
    """Export raw uncertainty metrics, negative scores, and OOD metrics."""

    if config.strategy.name != "subspace_ensemble":
        raise ValueError(
            "Subspace-ensemble scoring requires strategy.name: subspace_ensemble"
        )
    experiment_dir = Path(config.run_dir)
    inference_dir = experiment_dir / "subspace_ensemble_inference"
    output_dir = experiment_dir / "subspace_ensemble_scores"
    dataset_names = [
        f"{config.dataset.name}_test",
        *[f"{ood.name}_{ood.split}" for ood in config.dataset.ood_datasets],
    ]
    method_manifests = []

    for method in config.training.enabled_methods():
        scores_by_dataset: dict[str, dict[str, torch.Tensor]] = {}
        artifacts = []
        for dataset_name in dataset_names:
            inference_path = (
                inference_dir
                / dataset_name
                / f"student_{method}_{checkpoint}.pt"
            )
            inference = torch.load(
                inference_path,
                map_location="cpu",
                weights_only=True,
            )
            raw_metrics, scores = subspace_ensemble_score_vectors(
                inference["probabilities"]
            )
            labels = inference["labels"]
            path = (
                output_dir
                / dataset_name
                / f"student_{method}_{checkpoint}.pt"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "raw_metrics": raw_metrics,
                    "ood_scores": scores,
                    "labels": labels,
                    "metadata": {
                        "artifact_type": "subspace_ensemble_ood_scores",
                        "experiment_name": config.experiment_name,
                        "id_dataset": config.dataset.name,
                        "dataset": dataset_name,
                        "method": method,
                        "checkpoint": checkpoint,
                        "score_signs": SUBSPACE_ENSEMBLE_SCORE_SIGNS,
                        "higher_score_is": "more_id_like",
                        "inference_path": str(inference_path),
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
        metrics = detection_metrics_by_score(
            id_scores=scores_by_dataset[id_dataset_name],
            ood_scores_by_dataset={
                name: scores_by_dataset[name]
                for name in dataset_names
                if name != id_dataset_name
            },
        )
        method_manifest = {
            "method": method,
            "checkpoint_selection": checkpoint,
            "score_signs": SUBSPACE_ENSEMBLE_SCORE_SIGNS,
            "id_dataset": id_dataset_name,
            "metrics": metrics,
            "artifacts": artifacts,
        }
        method_manifests.append(method_manifest)
        write_json(output_dir / method / "manifest.json", method_manifest)

    manifest = {
        "version": 1,
        "artifact_type": "subspace_ensemble_ood_scores",
        "experiment_name": config.experiment_name,
        "checkpoint_selection": checkpoint,
        "score_signs": SUBSPACE_ENSEMBLE_SCORE_SIGNS,
        "methods": method_manifests,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def subspace_ensemble_score_vectors(
    probabilities: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return raw and sign-adjusted ensemble uncertainty vectors."""

    probabilities_array = probabilities.detach().cpu().double().numpy()
    raw_metrics = {
        ENSEMBLE_PREDICTIVE_ENTROPY: torch.from_numpy(
            ensemble_predictive_entropy(probabilities_array)
        ),
        ENSEMBLE_BALD: torch.from_numpy(ensemble_bald(probabilities_array)),
    }
    scores = {
        name: values * SUBSPACE_ENSEMBLE_SCORE_SIGNS[name]
        for name, values in raw_metrics.items()
    }
    return raw_metrics, scores


def detection_metrics_by_score(
    *,
    id_scores: dict[str, torch.Tensor],
    ood_scores_by_dataset: dict[str, dict[str, torch.Tensor]],
) -> dict[str, object]:
    """Compute per-OOD and arithmetic-macro metrics for each score."""

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
