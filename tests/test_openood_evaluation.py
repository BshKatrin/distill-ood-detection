"""Tests for OpenOOD metric definitions and score adapters."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from distill_ood_detection.evaluation.openood import (
    collect_confidence_scores,
    evaluate_openood_scores,
    openood_metrics,
)


def test_perfect_id_confidence_has_perfect_openood_metrics() -> None:
    metrics = openood_metrics(np.array([0.8, 0.9]), np.array([0.1, 0.2]))

    assert metrics == {
        "fpr95": 0.0,
        "auroc": 1.0,
        "aupr_in": 1.0,
        "aupr_out": 1.0,
    }


def test_fpr95_treats_ood_as_the_positive_class() -> None:
    metrics = openood_metrics(
        np.array([0.9, 0.8, 0.7, 0.1]),
        np.array([0.6, 0.5, 0.4, 0.3]),
    )

    assert metrics["fpr95"] == pytest.approx(0.25)


def test_aggregates_near_and_far_as_dataset_macro_averages() -> None:
    id_scores = np.array([0.8, 0.9])
    ood_scores = {
        "ssb_hard_test": np.array([0.1, 0.2]),
        "ninco_test": np.array([0.4, 0.85]),
        "textures_test": np.array([0.05, 0.15]),
    }
    result = evaluate_openood_scores(
        id_scores,
        ood_scores,
        {
            "ssb_hard_test": "near",
            "ninco_test": "near",
            "textures_test": "far",
        },
    )

    expected_near_auroc = np.mean(
        [
            result["datasets"]["ssb_hard_test"]["auroc"],
            result["datasets"]["ninco_test"]["auroc"],
        ]
    )
    assert result["groups"]["near"]["auroc"] == pytest.approx(expected_near_auroc)
    assert result["groups"]["far"] == result["datasets"]["textures_test"]


def test_collects_one_pluggable_score_per_image() -> None:
    images = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    labels = torch.zeros(3, dtype=torch.long)
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)

    scores = collect_confidence_scores(
        loader,
        lambda batch: batch.sum(dim=1),
        torch.device("cpu"),
    )

    np.testing.assert_array_equal(scores, np.array([3.0, 7.0, 11.0]))
