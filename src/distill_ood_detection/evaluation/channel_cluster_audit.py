"""Metrics and score transformations for channel-cluster audits."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from distill_ood_detection.evaluation.ood_metrics import ood_detection_metrics

AUDIT_SCORES = ("raw", "absolute_improvement", "relative_improvement")


def audit_score_values(
    artifact: Mapping[str, object], score: str
) -> np.ndarray:
    """Return one higher-is-ID audit score from a per-sample artifact."""

    if score == "raw":
        key, sign = "raw_reconstruction_error", -1.0
    elif score == "absolute_improvement":
        key, sign = "improvement", 1.0
    elif score == "relative_improvement":
        key, sign = "relative_improvement", 1.0
    else:
        raise ValueError(f"Unsupported audit score: {score}")
    values = artifact.get(key)
    if not torch.is_tensor(values) or values.ndim != 1:
        raise ValueError(f"Audit artifact is missing one-dimensional tensor {key!r}")
    array = values.detach().cpu().numpy().astype(np.float64, copy=False) * sign
    if not np.isfinite(array).all():
        raise ValueError(f"Audit score {score!r} contains non-finite values")
    return array


def distribution_summary(values: np.ndarray) -> dict[str, float | int]:
    """Return point estimates for one score distribution."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("Distribution values must be one-dimensional")
    if array.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "standard_deviation": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "iqr": float("nan"),
        }
    q25, median, q75 = np.quantile(array, [0.25, 0.5, 0.75])
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(median),
        "standard_deviation": float(array.std(ddof=0)),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
    }


def id_ood_point_metrics(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
) -> dict[str, float | int]:
    """Return OOD metrics and signed ID-minus-OOD median difference."""

    id_values = np.asarray(id_scores, dtype=np.float64)
    ood_values = np.asarray(ood_scores, dtype=np.float64)
    if id_values.size == 0 or ood_values.size == 0:
        return {
            "roc_auc": float("nan"),
            "fpr_at_95_tpr": float("nan"),
            "signed_median_difference": float("nan"),
            "id_count": int(id_values.size),
            "ood_count": int(ood_values.size),
        }
    labels = np.concatenate(
        [np.ones(id_values.size, dtype=np.int8), np.zeros(ood_values.size, dtype=np.int8)]
    )
    scores = np.concatenate([id_values, ood_values])
    metrics = ood_detection_metrics(labels, scores)
    return {
        **metrics,
        "signed_median_difference": float(
            np.median(id_values) - np.median(ood_values)
        ),
        "id_count": int(id_values.size),
        "ood_count": int(ood_values.size),
    }


def validate_sample_alignment(
    score_artifact: Mapping[str, object],
    metadata_artifact: Mapping[str, object],
) -> None:
    """Raise when score labels are not aligned with shared sample metadata."""

    labels = score_artifact.get("labels")
    alignment_labels = metadata_artifact.get("alignment_labels")
    if not torch.is_tensor(labels) or not torch.is_tensor(alignment_labels):
        raise ValueError("Alignment validation requires tensor labels in both artifacts")
    if labels.shape != alignment_labels.shape or not torch.equal(
        labels.cpu(), alignment_labels.cpu()
    ):
        raise ValueError("Specialist scores and shared sample metadata are misaligned")


def class_conditioned_metrics(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    id_classes: np.ndarray,
    ood_classes: np.ndarray,
    class_id: int,
    *,
    match_id_class: bool,
) -> dict[str, float | int | bool]:
    """Return metrics for a true-OOD or matched teacher-predicted class."""

    id_mask = id_classes == class_id if match_id_class else np.ones_like(id_classes, dtype=bool)
    ood_mask = ood_classes == class_id
    metrics = id_ood_point_metrics(id_scores[id_mask], ood_scores[ood_mask])
    return {
        **metrics,
        "small_sample": bool(metrics["id_count"] < 20 or metrics["ood_count"] < 20),
    }
