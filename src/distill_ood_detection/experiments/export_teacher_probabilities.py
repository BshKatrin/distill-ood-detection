"""Export deterministic teacher outputs for ID and configured OOD datasets."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from distill_ood_detection.config import TeacherProbabilityConfig
from distill_ood_detection.datasets.inference import (
    NamedLoader,
    build_in_distribution_test_loader,
    build_ood_loaders,
)
from distill_ood_detection.inference.outputs import collect_model_outputs
from distill_ood_detection.models.teacher import load_teacher
from distill_ood_detection.utils import resolve_device, set_seed, write_json


def run_teacher_probability_export(
    config: TeacherProbabilityConfig,
) -> dict[str, object]:
    """Export deterministic teacher logits and probabilities for configured datasets."""

    set_seed(config.seed)
    device = resolve_device(config.device)
    output_dir = Path(config.run_dir) / "teacher_probabilities"
    output_dir.mkdir(parents=True, exist_ok=True)

    loaders = [
        build_in_distribution_test_loader(config.dataset),
        *build_ood_loaders(config.dataset),
    ]
    teacher = load_teacher(config.teacher, device)

    artifacts: list[dict[str, object]] = []
    for named_loader in loaders:
        artifact = export_loader_probabilities(
            teacher=teacher,
            named_loader=named_loader,
            output_dir=output_dir,
            device=device,
            metadata={
                "model": "teacher",
                "dataset": named_loader.name,
                "split": named_loader.split,
                "teacher": asdict(config.teacher),
            },
        )
        artifacts.append(artifact)

    manifest = {
        "experiment_name": config.experiment_name,
        "dataset": asdict(config.dataset),
        "teacher": asdict(config.teacher),
        "artifacts": artifacts,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


@torch.no_grad()
def export_loader_probabilities(
    teacher: nn.Module,
    named_loader: NamedLoader,
    output_dir: Path,
    device: torch.device,
    metadata: dict[str, object],
) -> dict[str, object]:
    """Collect and save teacher logits and probabilities for one dataset loader."""

    outputs = collect_model_outputs(teacher, named_loader.loader, device)
    path = output_dir / named_loader.name / "probabilities.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **metadata,
            "logits": outputs.logits,
            "probabilities": outputs.probabilities,
            "labels": outputs.labels,
        },
        path,
    )
    return {
        "dataset": named_loader.name,
        "split": named_loader.split,
        "path": str(path),
    }
