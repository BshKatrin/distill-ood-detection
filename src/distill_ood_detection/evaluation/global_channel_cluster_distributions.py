"""Distribution summaries for global channel-cluster students."""

from __future__ import annotations

import numpy as np


def boxplot_summary(values: np.ndarray) -> dict[str, float | int]:
    """Return Tukey boxplot statistics for one finite distribution."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("Boxplot values must be a non-empty one-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError("Boxplot values must be finite")
    q25, median, q75 = np.quantile(array, (0.25, 0.5, 0.75))
    iqr = q75 - q25
    lower_candidates = array[array >= q25 - 1.5 * iqr]
    upper_candidates = array[array <= q75 + 1.5 * iqr]
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=0)),
        "median": float(median),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(iqr),
        "lower_whisker": float(lower_candidates.min()),
        "upper_whisker": float(upper_candidates.max()),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def teacher_model_name(hf_model_id: str) -> str:
    """Return the supported ResNet architecture encoded in a teacher ID."""

    for name in ("resnet18", "resnet50"):
        if name in hf_model_id.lower():
            return name
    raise ValueError(f"Cannot infer teacher model from {hf_model_id!r}")
