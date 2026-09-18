"""Tests for NMF channel-cluster audit orchestration and metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from distill_ood_detection.evaluation.channel_cluster_audit import (
    audit_score_values,
    class_conditioned_metrics,
    distribution_summary,
    id_ood_point_metrics,
    validate_sample_alignment,
)
from distill_ood_detection.experiments.channel_cluster_audit import (
    build_channel_cluster_audit_tasks,
    slurm_array_spec,
)


def test_real_audit_manifest_expands_to_269_unique_tasks() -> None:
    paths = [
        Path("configs/channel_cluster_audit/nmf_latent_cosine/cifar_10/resnet18.yaml"),
        Path("configs/channel_cluster_audit/nmf_latent_cosine/cifar_100/resnet18.yaml"),
    ]
    if not Path(
        "runs/channel_grouping/nmf_latent_cosine/cifar_10/resnet18/channel_groups/layer2.pt"
    ).exists():
        pytest.skip("Local fitted NMF hierarchies are not available")
    tasks = build_channel_cluster_audit_tasks(paths)

    assert len(tasks) == 269
    assert sum(task.variant == "whole" for task in tasks) == 148
    assert sum(task.variant == "fraction_p025" for task in tasks) == 121
    assert len(
        {(task.dataset, task.layer, task.variant, task.cluster_id) for task in tasks}
    ) == 269
    assert all(task.cluster_size >= 4 for task in tasks if task.variant == "fraction_p025")
    assert all(task.evaluation_draws == 1 for task in tasks if task.variant == "whole")
    assert all(task.evaluation_draws == 10 for task in tasks if task.variant == "fraction_p025")
    assert {
        task.layer: task.distance_threshold for task in tasks if task.dataset == "cifar10"
    } == {"layer2": 0.7, "layer3": 0.6, "layer4": 0.6}


def test_score_formulas_and_orientation() -> None:
    artifact = {
        "raw_reconstruction_error": torch.tensor([1.0, 3.0]),
        "identity_error": torch.tensor([4.0, 4.0]),
        "improvement": torch.tensor([3.0, 1.0]),
        "relative_improvement": torch.tensor([0.75, 0.25]),
    }
    np.testing.assert_allclose(audit_score_values(artifact, "raw"), [-1.0, -3.0])
    np.testing.assert_allclose(
        audit_score_values(artifact, "absolute_improvement"), [3.0, 1.0]
    )
    np.testing.assert_allclose(
        audit_score_values(artifact, "relative_improvement"), [0.75, 0.25]
    )


def test_point_metrics_distribution_and_signed_gap() -> None:
    id_scores = np.array([0.8, 0.9, 1.0])
    ood_scores = np.array([0.0, 0.1, 0.2])
    metrics = id_ood_point_metrics(id_scores, ood_scores)
    summary = distribution_summary(id_scores)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["fpr_at_95_tpr"] == pytest.approx(0.0)
    assert metrics["signed_median_difference"] == pytest.approx(0.8)
    assert summary["median"] == pytest.approx(0.9)
    assert summary["iqr"] == pytest.approx(0.1)


def test_class_conditioning_handles_empty_and_small_subsets() -> None:
    result = class_conditioned_metrics(
        np.array([0.9, 0.8]),
        np.array([0.1, 0.2]),
        np.array([0, 0]),
        np.array([1, 1]),
        0,
        match_id_class=True,
    )
    assert result["ood_count"] == 0
    assert np.isnan(result["roc_auc"])
    assert result["small_sample"] is True


def test_alignment_detects_label_mismatch() -> None:
    validate_sample_alignment(
        {"labels": torch.tensor([0, 1])},
        {"alignment_labels": torch.tensor([0, 1])},
    )
    with pytest.raises(ValueError, match="misaligned"):
        validate_sample_alignment(
            {"labels": torch.tensor([0, 1])},
            {"alignment_labels": torch.tensor([1, 0])},
        )


def test_slurm_array_spec_supports_targeted_retries() -> None:
    assert slurm_array_spec([0, 1, 2, 5, 7, 8, 8]) == "0-2,5,7-8"
    assert slurm_array_spec([]) == ""
