"""Export raw teacher activations for ID and configured OOD datasets."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from distill_ood_detection.config import TeacherActivationConfig
from distill_ood_detection.datasets.inference import (
    NamedLoader,
    build_in_distribution_test_loader,
    build_ood_loaders,
)
from distill_ood_detection.models.teacher import load_teacher
from distill_ood_detection.utils import resolve_device, set_seed, write_json


def run_teacher_activation_export(config: TeacherActivationConfig) -> dict[str, object]:
    """Export raw teacher activations for the configured layers and datasets."""

    set_seed(config.seed)
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir) / config.experiment_name / "teacher_activations"
    output_dir.mkdir(parents=True, exist_ok=True)

    loaders = [
        build_in_distribution_test_loader(config.dataset),
        *build_ood_loaders(config.dataset),
    ]
    teacher = load_teacher(config.teacher, device)

    artifacts: list[dict[str, object]] = []
    for named_loader in loaders:
        dataset_artifacts = export_loader_activations(
            teacher=teacher,
            layers=config.layers,
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
        artifacts.extend(dataset_artifacts)

    manifest = {
        "experiment_name": config.experiment_name,
        "dataset": asdict(config.dataset),
        "teacher": asdict(config.teacher),
        "layers": list(config.layers),
        "artifacts": artifacts,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


@torch.no_grad()
def export_loader_activations(
    teacher: nn.Module,
    layers: tuple[str, ...],
    named_loader: NamedLoader,
    output_dir: Path,
    device: torch.device,
    metadata: dict[str, object],
) -> list[dict[str, object]]:
    """Collect and save raw activations for each requested layer."""

    collector = TeacherActivationCollector(teacher, layers)
    try:
        labels: list[torch.Tensor] = []
        activations_by_layer: dict[str, list[torch.Tensor]] = {
            layer: [] for layer in layers
        }
        teacher.eval()
        for images, batch_labels in named_loader.loader:
            images = images.to(device)
            collector.clear()
            teacher(images)
            for layer in layers:
                activations_by_layer[layer].append(collector.activation(layer).cpu())
            labels.append(batch_labels.cpu())
    finally:
        collector.close()

    dataset_dir = output_dir / named_loader.name
    label_tensor = torch.cat(labels, dim=0)
    artifacts: list[dict[str, object]] = []
    for layer, batches in activations_by_layer.items():
        path = dataset_dir / f"{layer}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        activations = torch.cat(batches, dim=0)
        torch.save(
            {
                **metadata,
                "layer": layer,
                "activation_shape": tuple(activations.shape[1:]),
                "activations": activations,
                "labels": label_tensor,
            },
            path,
        )
        artifacts.append(
            {
                "dataset": named_loader.name,
                "split": named_loader.split,
                "layer": layer,
                "path": str(path),
            }
        )
    return artifacts


class TeacherActivationCollector:
    """Capture activations from multiple named teacher layers."""

    def __init__(self, teacher: nn.Module, layers: tuple[str, ...]) -> None:
        self._activations: dict[str, torch.Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        for layer_name in layers:
            try:
                layer = teacher.get_submodule(layer_name)
            except AttributeError as error:
                msg = f"Teacher does not have layer: {layer_name}"
                raise ValueError(msg) from error
            self._hooks.append(layer.register_forward_hook(self._capture(layer_name)))

    def clear(self) -> None:
        """Clear activations from the previous forward pass."""

        self._activations.clear()

    def activation(self, layer: str) -> torch.Tensor:
        """Return the captured activation for one layer."""

        try:
            return self._activations[layer]
        except KeyError as error:
            msg = f"Activation hook did not run for layer: {layer}"
            raise RuntimeError(msg) from error

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
            self._activations[layer_name] = output.detach()

        return hook
