"""Export compact OpenOOD metrics for a pretrained ViT MSP baseline.

The job keeps only one scalar max-softmax score per image in memory and writes
aggregate metrics.  It intentionally does not persist logits, probabilities,
images, or per-example score vectors.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from distill_ood_detection.config import load_config
from distill_ood_detection.datasets.openood_cifar import (
    OPENOOD_CIFAR_PROTOCOL,
    build_openood_cifar_dataset_config,
    validate_openood_cifar_manifests,
)
from distill_ood_detection.datasets.inference import (
    build_in_distribution_test_loader,
    build_ood_loaders,
)
from distill_ood_detection.evaluation.openood import evaluate_openood_scores
from distill_ood_detection.models.teacher import load_teacher
from distill_ood_detection.utils import resolve_device, set_seed, write_json


@torch.inference_mode()
def run_vit_openood_msp(
    config_path: Path,
    openood_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Evaluate ViT max-softmax confidence on fixed OpenOOD manifests."""

    config = load_config(config_path)
    if config.teacher.architecture not in ("auto", "vit"):
        raise ValueError("ViT MSP export requires a ViT teacher configuration")
    if "vit" not in config.teacher.hf_model_id.lower() and config.teacher.architecture == "auto":
        raise ValueError("The configured teacher does not look like a ViT")

    dataset = build_openood_cifar_dataset_config(config.dataset, openood_root)
    manifest_sizes = validate_openood_cifar_manifests(dataset)
    evaluation_config = replace(config, dataset=dataset)
    device = resolve_device(config.training.defaults.device)
    set_seed(config.training.defaults.seed)
    teacher = load_teacher(evaluation_config.teacher, device)

    named_loaders = [
        build_in_distribution_test_loader(evaluation_config.dataset),
        *build_ood_loaders(evaluation_config.dataset),
    ]
    score_vectors: dict[str, np.ndarray] = {}
    for named_loader in named_loaders:
        batches: list[np.ndarray] = []
        for images, _labels in named_loader.loader:
            logits = teacher(images.to(device, non_blocking=True))
            batches.append(
                torch.softmax(logits, dim=1)
                .amax(dim=1)
                .detach()
                .float()
                .cpu()
                .numpy()
            )
        if not batches:
            raise ValueError(f"OpenOOD loader is empty: {named_loader.name}")
        score_vectors[named_loader.name] = np.concatenate(batches)

    id_name = f"{evaluation_config.dataset.name}_test"
    id_scores = score_vectors[id_name]
    ood_scores = {
        f"{ood.name}_{ood.split}": score_vectors[f"{ood.name}_{ood.split}"]
        for ood in evaluation_config.dataset.ood_datasets
    }
    groups = {
        f"{ood.name}_{ood.split}": str(ood.group)
        for ood in evaluation_config.dataset.ood_datasets
    }
    metrics = evaluate_openood_scores(id_scores, ood_scores, groups)
    payload: dict[str, Any] = {
        "version": 1,
        "protocol": OPENOOD_CIFAR_PROTOCOL,
        "positive_class": "ood",
        "id_label": 0,
        "ood_label": 1,
        "input_score_orientation": "higher_is_id",
        "fpr95_definition": "ID false-positive rate at 95% OOD true-positive rate",
        "score": "teacher_msp",
        "inference_mode": "raw_image_teacher",
        "source_config": str(config_path),
        "experiment_name": config.experiment_name,
        "teacher": asdict(config.teacher),
        "manifest_sizes": manifest_sizes,
        "sample_counts": {name: int(values.size) for name, values in score_vectors.items()},
        "metrics": metrics,
        "storage_policy": "aggregate metrics only; no logits, probabilities, images, or score vectors",
    }
    write_json(output_path, payload)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--openood-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run the compact ViT MSP exporter from the command line."""

    args = _parse_args()
    run_vit_openood_msp(args.config, args.openood_root, args.output)


if __name__ == "__main__":
    main()
