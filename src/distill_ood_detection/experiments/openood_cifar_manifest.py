"""Build and execute restartable OpenOOD CIFAR evaluation task manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from distill_ood_detection.config import load_config
from distill_ood_detection.experiments.evaluate_openood_cifar import (
    run_openood_cifar_evaluation,
)
from distill_ood_detection.utils import write_json


def build_openood_cifar_task_manifest(
    selection_path: Path,
    output_path: Path,
    config_root: Path | None = None,
) -> dict[str, Any]:
    """Resolve selected variants and record ready and missing checkpoints."""

    with selection_path.open("r", encoding="utf-8") as handle:
        selection = yaml.safe_load(handle)
    raw_variants = selection.get("variants")
    if not isinstance(raw_variants, list) or not raw_variants:
        raise ValueError("OpenOOD CIFAR selection must contain a non-empty variants list")

    ready = []
    missing = []
    for raw_variant in raw_variants:
        variant = dict(raw_variant)
        config_path = Path(str(variant["config"]))
        if config_root is not None and not config_path.is_absolute():
            config_path = config_root / config_path
        config = load_config(config_path)
        checkpoint = str(variant.get("checkpoint", "best"))
        method = variant.get("method")
        checkpoint_method = (
            config.strategy.feature_denoising.method
            if config.strategy.name == "feature_denoising"
            else method
        )
        if not isinstance(checkpoint_method, str):
            raise ValueError(f"Variant requires a distillation method: {config_path}")
        metrics_path = Path(config.run_dir) / checkpoint_method / "metrics.json"
        checkpoint_path, reason = _checkpoint_from_metrics(metrics_path, checkpoint)
        task = {
            "name": str(variant["name"]),
            "config": str(config_path),
            "run_dir": config.run_dir,
            "method": method,
            "checkpoint": checkpoint,
            "apply_perturbation": bool(variant.get("apply_perturbation", False)),
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        }
        if reason is None:
            task["task_index"] = len(ready)
            ready.append(task)
        else:
            task["reason"] = reason
            missing.append(task)

    manifest = {
        "version": 1,
        "protocol": selection.get("protocol", "openood_v1_5"),
        "selection_path": str(selection_path),
        "tasks": ready,
        "missing": missing,
    }
    write_json(output_path, manifest)
    return manifest


def run_openood_cifar_manifest_task(
    manifest_path: Path,
    task_index: int,
    openood_root: Path,
    force: bool = False,
) -> dict[str, Any]:
    """Execute one ready task from an OpenOOD CIFAR evaluation manifest."""

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    tasks = manifest.get("tasks", [])
    if task_index < 0 or task_index >= len(tasks):
        raise IndexError(f"OpenOOD CIFAR task index out of range: {task_index}")
    task = tasks[task_index]
    config_path = Path(task["config"])
    config = load_config(config_path)
    return run_openood_cifar_evaluation(
        config,
        source_config_path=config_path,
        openood_root=openood_root,
        checkpoint=task["checkpoint"],
        method=task["method"],
        apply_perturbation=task["apply_perturbation"],
        force=force,
    )


def _checkpoint_from_metrics(
    metrics_path: Path,
    checkpoint: str,
) -> tuple[Path | None, str | None]:
    if not metrics_path.is_file():
        return None, f"missing metrics artifact: {metrics_path}"
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    raw_path = metrics.get(f"{checkpoint}_checkpoint_path")
    if not isinstance(raw_path, str):
        return None, f"metrics artifact has no {checkpoint}_checkpoint_path"
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.is_file():
        return None, f"missing checkpoint artifact: {path}"
    return path, None
