"""Tests for standardized OOD Score composition."""

from __future__ import annotations

import numpy as np
import pytest

from distill_ood_detection.evaluation import (
    ScoreStandardization,
    compose_standardized_scores,
    fit_score_standardization,
    standardize_score,
)


def test_fits_population_standardization_and_reuses_it() -> None:
    reference = np.array([1.0, 2.0, 3.0])

    statistics = fit_score_standardization(reference)

    assert statistics.mean == pytest.approx(2.0)
    assert statistics.standard_deviation == pytest.approx(np.sqrt(2.0 / 3.0))
    np.testing.assert_allclose(
        standardize_score(np.array([2.0, 4.0]), statistics),
        np.array([0.0, np.sqrt(6.0)]),
    )


def test_beta_zero_is_arithmetic_mean() -> None:
    layer3 = np.array([-2.0, 1.0, 4.0])
    layer4 = np.array([2.0, 3.0, 0.0])

    np.testing.assert_allclose(
        compose_standardized_scores(layer3, layer4, beta=0.0),
        np.array([0.0, 2.0, 2.0]),
    )


@pytest.mark.parametrize("beta", [-100.0, -10.0, -1.0, -0.1, 0.1, 1.0, 10.0, 100.0])
def test_nonzero_beta_matches_requested_formula(beta: float) -> None:
    layer3 = np.array([-0.2, 0.3, 1.1])
    layer4 = np.array([0.4, -0.7, 0.6])
    expected = np.log(
        np.exp(beta * layer3) + np.exp(beta * layer4)
    ) / beta

    np.testing.assert_allclose(
        compose_standardized_scores(layer3, layer4, beta),
        expected,
    )


def test_extreme_beta_is_stable() -> None:
    layer3 = np.array([100.0, -100.0])
    layer4 = np.array([-100.0, 100.0])

    scores = compose_standardized_scores(layer3, layer4, beta=100.0)

    assert np.isfinite(scores).all()
    np.testing.assert_allclose(scores, np.array([100.0, 100.0]))


def test_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="positive standard deviation"):
        fit_score_standardization([1.0, 1.0])
    with pytest.raises(ValueError, match="same shape"):
        compose_standardized_scores([1.0], [1.0, 2.0], beta=1.0)
    with pytest.raises(ValueError, match="beta must be finite"):
        compose_standardized_scores([1.0], [2.0], beta=np.inf)
    with pytest.raises(ValueError, match="positive and finite"):
        standardize_score([1.0], ScoreStandardization(0.0, 0.0))
