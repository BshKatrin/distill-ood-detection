"""Run logit and probability inference for ID and configured OOD datasets."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from distill_ood_detection.config import DistillationMethod, ExperimentConfig
from distill_ood_detection.datasets.inference import (
    NamedLoader,
    build_in_distribution_validation_loader,
    build_ood_loaders,
)
from distill_ood_detection.models.student import build_student
from distill_ood_detection.models.teacher import load_teacher
from distill_ood_detection.utils import resolve_device, set_seed, write_json

CheckpointSelection = Literal["best", "latest", "both"]


def run_probability_inference(
    config: ExperimentConfig,
    checkpoint: CheckpointSelection = "best",
    method: DistillationMethod | None = None,
) -> dict[str, object]:
    """Infer teacher and student logits/probabilities for ID and OOD datasets."""

    set_seed(config.training.seed)
    device = resolve_device(config.training.device)
    experiment_dir = Path(config.output_dir) / config.experiment_name
    output_dir = experiment_dir / "probabilities"
    output_dir.mkdir(parents=True, exist_ok=True)

    loaders = [
        build_in_distribution_validation_loader(config.dataset, seed=config.training.seed),
        *build_ood_loaders(config.dataset),
    ]
    teacher = load_teacher(config.teacher, device)
    methods = (method,) if method else config.training.methods

    artifacts: list[dict[str, object]] = []
    for named_loader in loaders:
        dataset_dir = output_dir / named_loader.name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        teacher_path = dataset_dir / "teacher.pt"
        _save_inference_artifact(
            path=teacher_path,
            model=teacher,
            loader=named_loader.loader,
            device=device,
            metadata={
                "model": "teacher",
                "dataset": named_loader.name,
                "split": named_loader.split,
            },
        )
        artifacts.append({"dataset": named_loader.name, "model": "teacher", "path": str(teacher_path)})

        for current_method in methods:
            for checkpoint_name in _selected_checkpoints(checkpoint):
                checkpoint_path = _student_checkpoint_path(
                    experiment_dir=experiment_dir,
                    method=current_method,
                    checkpoint=checkpoint_name,
                )
                student = build_student(config.student)
                state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
                student.load_state_dict(state_dict)
                student.to(device)
                student.eval()
                probability_path = dataset_dir / f"student_{current_method}_{checkpoint_name}.pt"
                _save_inference_artifact(
                    path=probability_path,
                    model=student,
                    loader=named_loader.loader,
                    device=device,
                    metadata={
                        "model": "student",
                        "method": current_method,
                        "checkpoint": checkpoint_name,
                        "checkpoint_path": str(checkpoint_path),
                        "dataset": named_loader.name,
                        "split": named_loader.split,
                    },
                )
                artifacts.append(
                    {
                        "dataset": named_loader.name,
                        "model": "student",
                        "method": current_method,
                        "checkpoint": checkpoint_name,
                        "path": str(probability_path),
                    }
                )

    manifest = {
        "experiment_name": config.experiment_name,
        "dataset": asdict(config.dataset),
        "checkpoint_selection": checkpoint,
        "artifacts": artifacts,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


@torch.no_grad()
def _save_inference_artifact(
    path: Path,
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
    metadata: dict[str, object],
) -> None:
    logits_batches: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    model.eval()
    for images, batch_labels in loader:
        images = images.to(device)
        logits = model(images)
        logits_batches.append(logits.cpu())
        probabilities.append(F.softmax(logits, dim=1).cpu())
        labels.append(batch_labels.cpu())

    torch.save(
        {
            **metadata,
            "logits": torch.cat(logits_batches, dim=0),
            "probabilities": torch.cat(probabilities, dim=0),
            "labels": torch.cat(labels, dim=0),
        },
        path,
    )


def _selected_checkpoints(checkpoint: CheckpointSelection) -> tuple[Literal["best", "latest"], ...]:
    if checkpoint == "both":
        return ("best", "latest")
    return (checkpoint,)


def _student_checkpoint_path(
    experiment_dir: Path,
    method: DistillationMethod,
    checkpoint: Literal["best", "latest"],
) -> Path:
    metrics_path = experiment_dir / method / "metrics.json"
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    key = f"{checkpoint}_checkpoint_path"
    raw_path = metrics.get(key)
    if not isinstance(raw_path, str):
        raise ValueError(f"Missing {key} in {metrics_path}")
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise FileNotFoundError(f"Student checkpoint not found: {path}")
    return path
