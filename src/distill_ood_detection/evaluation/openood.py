"""OpenOOD-style score collection and ImageNet-200 benchmark aggregation."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import torch
from numpy.typing import ArrayLike
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

ConfidenceScore = Callable[[torch.Tensor], torch.Tensor]
OPENOOD_METRIC_NAMES = ("fpr95", "auroc", "aupr_in", "aupr_out")


@torch.no_grad()
def collect_confidence_scores(
    loader: DataLoader[tuple[torch.Tensor, int]],
    score: ConfidenceScore,
    device: torch.device,
) -> np.ndarray:
    """Collect one ID-confidence value per image from a pluggable score function."""

    batches: list[np.ndarray] = []
    for images, _labels in loader:
        values = score(images.to(device, non_blocking=True))
        if values.ndim != 1 or values.shape[0] != images.shape[0]:
            raise ValueError(
                "An OpenOOD score function must return one value per input image"
            )
        batches.append(values.detach().float().cpu().numpy())
    if not batches:
        raise ValueError("Cannot evaluate an empty dataloader")
    return np.concatenate(batches)


def openood_metrics(id_scores: ArrayLike, ood_scores: ArrayLike) -> dict[str, float]:
    """Compute OpenOOD's FPR95, AUROC, AUPR-IN, and AUPR-OUT metrics.

    Scores must be ID confidence: larger values indicate that a sample is more
    likely to be in-distribution. Returned values are fractions in ``[0, 1]``.
    """

    id_values = _score_vector(id_scores, "id_scores")
    ood_values = _score_vector(ood_scores, "ood_scores")
    # OpenOOD v1.5 treats OOD as the positive class. Its postprocessors still
    # emit ID confidence, so negate that confidence for ROC/FPR calculations.
    ood_labels = np.concatenate(
        [
            np.zeros(id_values.size, dtype=np.int8),
            np.ones(ood_values.size, dtype=np.int8),
        ]
    )
    id_confidence = np.concatenate([id_values, ood_values])
    ood_confidence = -id_confidence
    fpr, tpr, _ = roc_curve(ood_labels, ood_confidence, pos_label=1)
    fpr95 = float(np.min(fpr[tpr >= 0.95]))
    in_precision, in_recall, _ = precision_recall_curve(
        1 - ood_labels, id_confidence, pos_label=1
    )
    out_precision, out_recall, _ = precision_recall_curve(
        ood_labels,
        ood_confidence,
        pos_label=1,
    )
    return {
        "fpr95": fpr95,
        "auroc": float(roc_auc_score(ood_labels, ood_confidence)),
        "aupr_in": float(auc(in_recall, in_precision)),
        "aupr_out": float(auc(out_recall, out_precision)),
    }


def evaluate_openood_scores(
    id_scores: ArrayLike,
    ood_scores: Mapping[str, ArrayLike],
    groups: Mapping[str, str],
) -> dict[str, dict[str, dict[str, float]]]:
    """Evaluate every OOD dataset and compute per-group macro averages."""

    if not ood_scores:
        raise ValueError("At least one OOD score vector is required")
    missing_groups = sorted(set(ood_scores) - set(groups))
    if missing_groups:
        raise ValueError(f"Missing OpenOOD groups for datasets: {missing_groups}")
    per_dataset = {
        name: openood_metrics(id_scores, scores) for name, scores in ood_scores.items()
    }
    grouped_names: dict[str, list[str]] = {}
    for name in ood_scores:
        grouped_names.setdefault(groups[name], []).append(name)
    group_metrics = {
        group: {
            metric: float(np.mean([per_dataset[name][metric] for name in names]))
            for metric in OPENOOD_METRIC_NAMES
        }
        for group, names in grouped_names.items()
    }
    return {"datasets": per_dataset, "groups": group_metrics}


def _score_vector(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array
