"""Build compact Parquet summaries for the channel-cluster audit dashboard."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from distill_ood_detection.evaluation.channel_cluster_audit import (
    AUDIT_SCORES,
    audit_score_values,
    class_conditioned_metrics,
    distribution_summary,
    id_ood_point_metrics,
    validate_sample_alignment,
)
from distill_ood_detection.experiments.channel_cluster_audit import (
    ChannelClusterAuditTask,
    channel_cluster_task_is_complete,
    load_channel_cluster_audit_config,
    load_channel_cluster_audit_manifest,
    resolve_channel_cluster_task_config,
)
from distill_ood_detection.utils import write_json


def summarize_channel_cluster_audit(
    manifest_path: Path,
    output_dir: Path,
    *,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Write dashboard-ready Parquet summaries from task artifacts."""

    tasks = load_channel_cluster_audit_manifest(manifest_path)
    task_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    complete_tasks = 0
    metadata_cache: dict[tuple[str, str], dict[str, Any]] = {}

    for task in tasks:
        complete = channel_cluster_task_is_complete(task, deep=True)
        status = "complete" if complete else "incomplete"
        task_row = {**asdict(task), "status": status}
        task_rows.append(task_row)
        if not complete:
            continue
        complete_tasks += 1
        config = resolve_channel_cluster_task_config(task)
        method_dir = Path(task.run_dir) / config.strategy.feature_denoising.method
        with (method_dir / "metrics.json").open("r", encoding="utf-8") as handle:
            training_metrics = json.load(handle)
        with (method_dir / "history.json").open("r", encoding="utf-8") as handle:
            history = json.load(handle)["history"]
        task_row.update(
            {
                "best_epoch": int(
                    min(
                        history,
                        key=lambda row: row["validation_reconstruction_loss"],
                    )["epoch"]
                ),
                "best_validation_loss": training_metrics[
                    "best_validation_reconstruction_loss"
                ],
                "final_validation_loss": training_metrics[
                    "final_validation_reconstruction_loss"
                ],
                "test_reconstruction_loss": training_metrics[
                    "test_reconstruction_loss"
                ],
                "runtime_seconds": training_metrics["seconds"],
            }
        )
        for record in history:
            curve_rows.append(
                {
                    **_task_context(task),
                    "epoch": int(record["epoch"]),
                    "training_loss": float(record["reconstruction_loss"]),
                    "validation_loss": float(
                        record["validation_reconstruction_loss"]
                    ),
                }
            )
        score_artifacts = _load_task_score_artifacts(task)
        shared_metadata = {
            name: _shared_metadata(task, name, metadata_cache)
            for name in score_artifacts
        }
        for name, artifact in score_artifacts.items():
            validate_sample_alignment(artifact, shared_metadata[name])
        id_name = f"{task.dataset}_test"
        near_name = "cifar100_test" if task.dataset == "cifar10" else "cifar10_test"
        ood_names = ("mnist_test", "svhn_test", near_name)
        for score_name in AUDIT_SCORES:
            id_scores = audit_score_values(score_artifacts[id_name], score_name)
            id_summary = distribution_summary(id_scores)
            distribution_rows.append(
                {
                    **_task_context(task),
                    "score": score_name,
                    "distribution": "id",
                    "dataset_name": id_name,
                    **id_summary,
                }
            )
            actual_metrics: dict[str, dict[str, Any]] = {}
            for ood_name in ood_names:
                ood_scores = audit_score_values(score_artifacts[ood_name], score_name)
                ood_summary = distribution_summary(ood_scores)
                distribution_rows.append(
                    {
                        **_task_context(task),
                        "score": score_name,
                        "distribution": "ood",
                        "dataset_name": ood_name,
                        **ood_summary,
                    }
                )
                metrics = id_ood_point_metrics(id_scores, ood_scores)
                actual_metrics[ood_name] = metrics
                metric_rows.append(
                    {
                        **_task_context(task),
                        "score": score_name,
                        "ood_dataset": _display_ood_name(ood_name, near_name),
                        "ood_artifact_name": ood_name,
                        "ood_kind": "near" if ood_name == near_name else "far",
                        **metrics,
                    }
                )
            far = [actual_metrics["mnist_test"], actual_metrics["svhn_test"]]
            metric_rows.append(
                {
                    **_task_context(task),
                    "score": score_name,
                    "ood_dataset": "far_macro",
                    "ood_artifact_name": "far_macro",
                    "ood_kind": "far_macro",
                    **{
                        key: float(np.mean([row[key] for row in far]))
                        for key in (
                            "roc_auc",
                            "fpr_at_95_tpr",
                            "signed_median_difference",
                        )
                    },
                    "id_count": int(far[0]["id_count"]),
                    "ood_count": int(sum(row["ood_count"] for row in far)),
                }
            )
            class_rows.extend(
                _class_metric_rows(
                    task,
                    score_name,
                    id_scores,
                    audit_score_values(score_artifacts[near_name], score_name),
                    shared_metadata[id_name],
                    shared_metadata[near_name],
                    near_name,
                )
            )

    if require_complete and complete_tasks != len(tasks):
        raise RuntimeError(
            f"Only {complete_tasks}/{len(tasks)} audit tasks have complete artifacts"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "task_metadata": pd.DataFrame(task_rows),
        "cluster_metrics": pd.DataFrame(metric_rows),
        "distribution_summaries": pd.DataFrame(distribution_rows),
        "class_metrics": pd.DataFrame(class_rows),
        "training_curves": pd.DataFrame(curve_rows),
    }
    for name, frame in frames.items():
        _write_parquet(frame, output_dir / f"{name}.parquet")
    result = {
        "artifact_version": 1,
        "manifest_path": str(manifest_path),
        "task_count": len(tasks),
        "complete_task_count": complete_tasks,
        "parquet_files": {
            name: str(output_dir / f"{name}.parquet") for name in frames
        },
    }
    write_json(output_dir / "manifest.json", result)
    return result


def _task_context(task: ChannelClusterAuditTask) -> dict[str, Any]:
    return {
        "task_index": task.task_index,
        "dataset": task.dataset,
        "layer": task.layer,
        "variant": task.variant,
        "cluster_id": task.cluster_id,
        "cluster_order": task.cluster_order,
        "channel_group_index": task.channel_group_index,
        "cluster_size": task.cluster_size,
        "masked_channel_count": task.masked_channel_count,
        "realized_mask_fraction": task.realized_mask_fraction,
        "run_dir": task.run_dir,
    }


def _load_task_score_artifacts(
    task: ChannelClusterAuditTask,
) -> dict[str, dict[str, Any]]:
    config = resolve_channel_cluster_task_config(task)
    names = [
        f"{task.dataset}_test",
        *(f"{item.name}_{item.split}" for item in config.dataset.ood_datasets),
    ]
    return {
        name: torch.load(
            Path(task.run_dir)
            / "feature_denoising_scores"
            / name
            / "student_best.pt",
            map_location="cpu",
            weights_only=False,
        )
        for name in names
    }


def _shared_metadata(
    task: ChannelClusterAuditTask,
    dataset_name: str,
    cache: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    key = (task.dataset, dataset_name)
    if key not in cache:
        audit = load_channel_cluster_audit_config(Path(task.audit_config_path))
        cache[key] = torch.load(
            Path(audit.run_dir) / "sample_metadata" / f"{dataset_name}.pt",
            map_location="cpu",
            weights_only=False,
        )
    return cache[key]


def _class_metric_rows(
    task: ChannelClusterAuditTask,
    score_name: str,
    id_scores: np.ndarray,
    near_scores: np.ndarray,
    id_metadata: dict[str, Any],
    near_metadata: dict[str, Any],
    near_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    true_classes = near_metadata["true_class"].numpy()
    id_predictions = id_metadata["teacher_predicted_id_class"].numpy()
    near_predictions = near_metadata["teacher_predicted_id_class"].numpy()
    id_confidence = id_metadata["teacher_confidence"].numpy()
    near_confidence = near_metadata["teacher_confidence"].numpy()
    near_class_names = near_metadata["metadata"].get("class_names", [])
    id_class_names = id_metadata["metadata"].get("teacher_id_class_names", [])
    for mode, id_classes, ood_classes, class_names, match_id in (
        ("true_ood_class", id_predictions, true_classes, near_class_names, False),
        (
            "teacher_predicted_class",
            id_predictions,
            near_predictions,
            id_class_names,
            True,
        ),
    ):
        class_ids = range(len(class_names)) if class_names else sorted(set(ood_classes.tolist()))
        for class_id in class_ids:
            metrics = class_conditioned_metrics(
                id_scores,
                near_scores,
                id_classes,
                ood_classes,
                class_id,
                match_id_class=match_id,
            )
            id_mask = id_classes == class_id if match_id else np.ones(id_classes.shape, dtype=bool)
            ood_mask = ood_classes == class_id
            rows.append(
                {
                    **_task_context(task),
                    "score": score_name,
                    "mode": mode,
                    "near_dataset": near_name,
                    "class_id": int(class_id),
                    "class_name": (
                        str(class_names[class_id])
                        if class_id < len(class_names)
                        else str(class_id)
                    ),
                    "id_teacher_confidence_mean": (
                        float(id_confidence[id_mask].mean())
                        if id_mask.any()
                        else float("nan")
                    ),
                    "ood_teacher_confidence_mean": (
                        float(near_confidence[ood_mask].mean())
                        if ood_mask.any()
                        else float("nan")
                    ),
                    **metrics,
                }
            )
    return rows


def _display_ood_name(name: str, near_name: str) -> str:
    if name == "mnist_test":
        return "MNIST"
    if name == "svhn_test":
        return "SVHN"
    if name == near_name:
        return "opposite_cifar"
    return name


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)
