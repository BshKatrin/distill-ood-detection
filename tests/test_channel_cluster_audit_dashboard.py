"""Synthetic smoke tests for the minimal channel-cluster audit dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("panel")

import panel as pn

from distill_ood_detection.dashboard.channel_cluster_audit import (
    AuditRepository,
    LAYERS,
    _cluster_score_plot,
    build_dashboard,
)


def test_minimal_dashboard_constructs_all_layer_plots(tmp_path: Path) -> None:
    task_rows = []
    metric_rows = []
    task_index = 0
    for dataset in ("cifar10", "cifar100"):
        for variant in ("whole", "fraction_p025"):
            for layer in LAYERS:
                for cluster_order, cluster_id in enumerate(("C001", "C002"), start=1):
                    task = {
                        "task_index": task_index,
                        "dataset": dataset,
                        "layer": layer,
                        "variant": variant,
                        "cluster_id": cluster_id,
                        "cluster_order": cluster_order,
                        "status": "complete",
                    }
                    task_rows.append(task)
                    for score in ("raw", "absolute_improvement", "relative_improvement"):
                        for ood_dataset in ("MNIST", "SVHN", "opposite_cifar", "far_macro"):
                            metric_rows.append(
                                {
                                    **task,
                                    "score": score,
                                    "ood_dataset": ood_dataset,
                                    "roc_auc": 0.70,
                                    "fpr_at_95_tpr": 0.40,
                                }
                            )
                    task_index += 1

    pd.DataFrame(task_rows).to_parquet(tmp_path / "task_metadata.parquet")
    pd.DataFrame(metric_rows).to_parquet(tmp_path / "cluster_metrics.parquet")

    repository = AuditRepository(tmp_path)
    plots = [
        _cluster_score_plot(repository, "cifar10", "whole", "raw", layer)
        for layer in LAYERS
    ]
    assert all(isinstance(plot, pn.pane.Bokeh) for plot in plots)

    dashboard = build_dashboard(tmp_path)
    assert dashboard.title == "NMF Channel-Cluster Audit"
    assert [widget.name for widget in dashboard.sidebar] == [
        "ID dataset",
        "Masking variant",
        "OOD score input",
    ]
