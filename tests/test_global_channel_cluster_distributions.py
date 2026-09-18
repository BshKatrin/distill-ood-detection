"""Tests for global-student cluster distributions and dashboard."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from distill_ood_detection.evaluation.global_channel_cluster_distributions import (
    boxplot_summary,
    teacher_model_name,
)

pytest.importorskip("panel")

import panel as pn

from distill_ood_detection.dashboard.global_channel_cluster_distributions import (
    GlobalClusterDistributionRepository,
    LAYERS,
    _cluster_boxplot,
    build_dashboard,
)
from distill_ood_detection.experiments import global_channel_cluster_distributions as exporter


def test_boxplot_summary_uses_tukey_whiskers() -> None:
    summary = boxplot_summary(np.array([0.0, 1.0, 2.0, 3.0, 100.0]))

    assert summary["count"] == 5
    assert summary["median"] == 2.0
    assert summary["q25"] == 1.0
    assert summary["q75"] == 3.0
    assert summary["lower_whisker"] == 0.0
    assert summary["upper_whisker"] == 3.0
    assert summary["maximum"] == 100.0


def test_teacher_model_name_supports_both_dashboard_models() -> None:
    assert teacher_model_name("org/resnet18_cifar10") == "resnet18"
    assert teacher_model_name("org/resnet50_cifar100") == "resnet50"
    with pytest.raises(ValueError, match="Cannot infer"):
        teacher_model_name("org/vit")


def test_summary_builds_compact_boxplot_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    config = SimpleNamespace(
        run_dir=str(run_dir),
        dataset=SimpleNamespace(
            name="cifar10",
            ood_datasets=(
                SimpleNamespace(name="mnist", split="test"),
                SimpleNamespace(name="svhn", split="test"),
                SimpleNamespace(name="cifar100", split="test"),
            ),
        ),
    )
    monkeypatch.setattr(exporter, "load_config", lambda _: config)
    clusters = [
        {
            "cluster_id": "C000",
            "cluster_order": 0,
            "channel_group_index": 1,
            "cluster_size": 4,
            "masked_channel_count": 1,
            "realized_mask_fraction": 0.25,
        },
        {
            "cluster_id": "C001",
            "cluster_order": 1,
            "channel_group_index": 0,
            "cluster_size": 8,
            "masked_channel_count": 2,
            "realized_mask_fraction": 0.25,
        },
    ]
    for dataset_name in ("cifar10_test", "mnist_test", "svhn_test", "cifar100_test"):
        path = (
            run_dir
            / "global_channel_cluster_distributions"
            / dataset_name
            / "student_best.pt"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "labels": torch.tensor([0, 1, 2]),
                "raw_reconstruction_error": torch.tensor(
                    [[0.8, 0.7], [0.7, 0.6], [0.6, 0.5]]
                ),
                "absolute_improvement": torch.tensor(
                    [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]]
                ),
                "relative_improvement": torch.tensor(
                    [[0.4, 0.5], [0.5, 0.6], [0.6, 0.7]]
                ),
                "metadata": {
                    "teacher_model": "resnet18",
                    "feature_layer": "layer2",
                    "clusters": clusters,
                },
            },
            path,
        )

    output_dir = tmp_path / "summary"
    manifest = exporter.summarize_global_channel_cluster_distributions(
        [Path("synthetic.yaml")],
        output_dir,
    )

    frame = pd.read_parquet(output_dir / "boxplot_summaries.parquet")
    metrics = pd.read_parquet(output_dir / "ood_metrics.parquet")
    assert manifest["row_count"] == 24
    assert len(frame) == 24
    assert set(frame.score) == {
        "raw_reconstruction_error",
        "absolute_improvement",
        "relative_improvement",
    }
    assert set(frame.distribution) == {"id", "mnist", "svhn", "opposite_cifar"}
    assert manifest["ood_metric_row_count"] == 18
    assert len(metrics) == 18
    assert set(metrics.ood_dataset) == {"mnist", "svhn", "cifar100"}
    assert set(metrics.score) == {
        "raw_reconstruction_error",
        "absolute_improvement",
        "relative_improvement",
    }
    assert metrics.roc_auc.between(0.0, 1.0).all()
    assert metrics.fpr_at_95_tpr.between(0.0, 1.0).all()


def test_global_cluster_dashboard_constructs_all_layers(tmp_path: Path) -> None:
    rows = []
    for teacher_model in ("resnet18", "resnet50"):
        for id_dataset in ("cifar10", "cifar100"):
            opposite = "CIFAR-100" if id_dataset == "cifar10" else "CIFAR-10"
            labels = {
                "id": "CIFAR-10" if id_dataset == "cifar10" else "CIFAR-100",
                "mnist": "MNIST",
                "svhn": "SVHN",
                "opposite_cifar": opposite,
            }
            for layer in LAYERS:
                for score in (
                    "raw_reconstruction_error",
                    "absolute_improvement",
                    "relative_improvement",
                ):
                    for cluster_order, cluster_id in enumerate(("C000", "C001")):
                        for distribution_order, distribution in enumerate(labels):
                            rows.append(
                                {
                                    "teacher_model": teacher_model,
                                    "id_dataset": id_dataset,
                                    "layer": layer,
                                    "score": score,
                                    "distribution": distribution,
                                    "distribution_order": distribution_order,
                                    "dataset_label": labels[distribution],
                                    "cluster_id": cluster_id,
                                    "cluster_order": cluster_order,
                                    "cluster_size": 8,
                                    "masked_channel_count": 2,
                                    "count": 100,
                                    "mean": 0.3,
                                    "standard_deviation": 0.1,
                                    "median": 0.3,
                                    "q25": 0.2,
                                    "q75": 0.4,
                                    "iqr": 0.2,
                                    "lower_whisker": 0.0,
                                    "upper_whisker": 0.6,
                                    "minimum": -0.1,
                                    "maximum": 0.8,
                                }
                            )
    pd.DataFrame(rows).to_parquet(tmp_path / "boxplot_summaries.parquet")

    repository = GlobalClusterDistributionRepository(tmp_path)
    plots = [
        _cluster_boxplot(
            repository,
            "resnet18",
            "cifar10",
            "absolute_improvement",
            layer,
        )
        for layer in LAYERS
    ]
    assert all(isinstance(plot, pn.pane.Bokeh) for plot in plots)

    dashboard = build_dashboard(tmp_path)
    assert dashboard.title == "Global NMF Student: Cluster Scores"
    assert [widget.name for widget in dashboard.sidebar] == [
        "Teacher model",
        "ID dataset",
        "Score",
    ]
