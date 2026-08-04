"""Export raw predictions from trained random-subspace student ensembles."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Literal

import torch

from distill_ood_detection.config import DistillationMethod, ExperimentConfig
from distill_ood_detection.datasets.inference import (
    build_in_distribution_test_loader,
    build_ood_loaders,
)
from distill_ood_detection.distillation.subspace_ensemble import (
    FittedSubspaceEnsemble,
    infer_subspace_ensemble,
    load_fitted_subspace_ensemble,
)
from distill_ood_detection.models.student import build_student
from distill_ood_detection.models.teacher import build_feature_forwarder, load_teacher
from distill_ood_detection.utils import resolve_device, set_seed, write_json

CheckpointSelection = Literal["best", "latest"]


def run_subspace_ensemble_inference_export(
    config: ExperimentConfig,
    checkpoint: CheckpointSelection = "best",
) -> dict[str, object]:
    """Export member logits and probabilities for ID and configured OOD data."""

    _validate_config(config)
    seed = config.training.defaults.seed
    set_seed(seed)
    device = resolve_device(config.training.defaults.device)
    teacher = load_teacher(config.teacher, device)
    if config.student.feature_layer is None:
        raise ValueError("Subspace-ensemble inference requires student.feature_layer")
    forwarder = build_feature_forwarder(teacher, config.student.feature_layer)
    experiment_dir = Path(config.run_dir)
    subspace_path = experiment_dir / "subspace_ensemble.pt"
    fitted = load_fitted_subspace_ensemble(subspace_path, device)
    _validate_fitted_config(config, fitted)
    loaders = [
        build_in_distribution_test_loader(config.dataset),
        *build_ood_loaders(config.dataset),
    ]
    artifacts: list[dict[str, object]] = []
    output_dir = experiment_dir / "subspace_ensemble_inference"

    for method in config.training.enabled_methods():
        students = torch.nn.ModuleList(
            [
                build_student(config.student)
                for _ in range(config.strategy.subspace_ensemble.ensemble_size)
            ]
        )
        checkpoint_path = (
            experiment_dir / method / f"{checkpoint}_ensemble.pt"
        )
        students.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
        )
        for named_loader in loaders:
            logits, probabilities, labels = infer_subspace_ensemble(
                students=students,
                fitted=fitted,
                forwarder=forwarder,
                loader=named_loader.loader,
                device=device,
            )
            path = (
                output_dir
                / named_loader.name
                / f"student_{method}_{checkpoint}.pt"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "logits": logits,
                    "probabilities": probabilities,
                    "labels": labels,
                    "metadata": {
                        "artifact_type": "subspace_ensemble_predictions",
                        "experiment_name": config.experiment_name,
                        "id_dataset": config.dataset.name,
                        "dataset": named_loader.name,
                        "split": named_loader.split,
                        "method": method,
                        "subspace_method": fitted.method,
                        "ensemble_size": fitted.ensemble_size,
                        "subset_size": fitted.subset_size,
                        "checkpoint": checkpoint,
                        "checkpoint_path": str(checkpoint_path),
                        "subspace_path": str(subspace_path),
                        "member_axis": 1,
                    },
                },
                path,
            )
            artifacts.append(
                {
                    "dataset": named_loader.name,
                    "method": method,
                    "samples": labels.shape[0],
                    "logit_shape": list(logits.shape),
                    "path": str(path),
                }
            )

    manifest = {
        "version": 1,
        "artifact_type": "subspace_ensemble_inference",
        "experiment_name": config.experiment_name,
        "run_dir": config.run_dir,
        "dataset": asdict(config.dataset),
        "teacher": asdict(config.teacher),
        "subspace_ensemble": asdict(config.strategy.subspace_ensemble),
        "methods": list(config.training.enabled_methods()),
        "checkpoint_selection": checkpoint,
        "subspace_path": str(subspace_path),
        "artifacts": artifacts,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def _validate_config(config: ExperimentConfig) -> None:
    if config.strategy.name != "subspace_ensemble":
        raise ValueError(
            "Subspace-ensemble inference requires strategy.name: subspace_ensemble"
        )
    if config.student.kind != "linear":
        raise ValueError("Subspace-ensemble inference currently requires linear students")


def _validate_fitted_config(
    config: ExperimentConfig,
    fitted: FittedSubspaceEnsemble,
) -> None:
    ensemble = config.strategy.subspace_ensemble
    if (
        fitted.method != ensemble.method
        or fitted.assignment != ensemble.assignment
        or fitted.ensemble_size != ensemble.ensemble_size
        or fitted.subset_size != ensemble.subset_size
        or fitted.expected_feature_shape != ensemble.expected_feature_shape
        or fitted.master_seed != config.training.defaults.seed
    ):
        raise ValueError("Saved subspace ensemble does not match the resolved config")
