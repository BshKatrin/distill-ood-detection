"""OOD detection metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import roc_auc_score, roc_curve


def ood_detection_metrics(
    true_labels: Sequence[int] | ArrayLike,
    pred_labels: Sequence[float] | ArrayLike,
) -> dict[str, float]:
    """Compute ROC-AUC and FPR@95 for OOD detection scores.

    Args:
        true_labels: Binary labels with 1 for ID samples and 0 for OOD samples.
        pred_labels: ID confidence scores for the same samples.

    Returns:
        A dictionary with ``roc_auc`` and ``fpr_at_95_tpr``.

    Raises:
        ValueError: If labels and scores have different lengths, labels are not
            binary 0/1 values, or either class is missing.
    """

    labels = np.asarray(true_labels)
    scores = np.asarray(pred_labels, dtype=float)

    if labels.ndim != 1 or scores.ndim != 1:
        msg = "true_labels and pred_labels must be one-dimensional arrays."
        raise ValueError(msg)
    if labels.shape[0] != scores.shape[0]:
        msg = "true_labels and pred_labels must contain the same number of samples."
        raise ValueError(msg)

    unique_labels = set(np.unique(labels).tolist())
    if not unique_labels.issubset({0, 1}):
        msg = "true_labels must contain only binary labels: 1 for ID and 0 for OOD."
        raise ValueError(msg)
    if unique_labels != {0, 1}:
        msg = "true_labels must contain at least one ID sample and one OOD sample."
        raise ValueError(msg)

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fpr_at_95_tpr = float(np.min(fpr[tpr >= 0.95]))

    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "fpr_at_95_tpr": fpr_at_95_tpr,
    }
