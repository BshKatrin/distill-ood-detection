"""Standardization and smooth composition of compatible OOD Scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class ScoreStandardization:
    """Population statistics used to standardize one OOD Score."""

    mean: float
    standard_deviation: float


def fit_score_standardization(scores: ArrayLike) -> ScoreStandardization:
    """Fit population mean and standard deviation to finite 1D scores."""

    values = _score_vector(scores, "scores")
    standard_deviation = float(values.std(ddof=0))
    if standard_deviation <= 0.0:
        raise ValueError("scores must have a positive standard deviation")
    return ScoreStandardization(
        mean=float(values.mean()),
        standard_deviation=standard_deviation,
    )


def standardize_score(
    scores: ArrayLike,
    statistics: ScoreStandardization,
) -> NDArray[np.float64]:
    """Standardize scores using precomputed reference statistics."""

    values = _score_vector(scores, "scores")
    if not np.isfinite(statistics.mean):
        raise ValueError("standardization mean must be finite")
    if (
        not np.isfinite(statistics.standard_deviation)
        or statistics.standard_deviation <= 0.0
    ):
        raise ValueError("standardization standard deviation must be positive and finite")
    return (values - statistics.mean) / statistics.standard_deviation


def compose_standardized_scores(
    score_layer3: ArrayLike,
    score_layer4: ArrayLike,
    beta: float,
) -> NDArray[np.float64]:
    r"""Compose two standardized scores with the HEAT smooth operator.

    For nonzero ``beta``, this computes
    ``log(exp(beta * layer3) + exp(beta * layer4)) / beta`` using a stable
    log-sum-exp calculation. At ``beta == 0``, it returns the arithmetic mean.
    """

    layer3 = _score_vector(score_layer3, "score_layer3")
    layer4 = _score_vector(score_layer4, "score_layer4")
    if layer3.shape != layer4.shape:
        raise ValueError(
            "score_layer3 and score_layer4 must have the same shape, "
            f"got {layer3.shape} and {layer4.shape}"
        )
    beta = float(beta)
    if not np.isfinite(beta):
        raise ValueError("beta must be finite")
    if beta == 0.0:
        return (layer3 + layer4) / 2.0

    scaled = beta * np.stack((layer3, layer4), axis=0)
    maxima = scaled.max(axis=0)
    log_sum_exp = maxima + np.log(np.exp(scaled - maxima).sum(axis=0))
    return log_sum_exp / beta


def _score_vector(scores: ArrayLike, name: str) -> NDArray[np.float64]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if values.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values
