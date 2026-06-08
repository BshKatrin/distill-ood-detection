"""Tests for tree-based student inference outputs."""

from __future__ import annotations

import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from distill_ood_detection.inference import collect_tree_model_outputs


class _FixedPredictionModel:
    def __init__(self, predictions: np.ndarray) -> None:
        self._predictions = predictions

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self._predictions[: features.shape[0]]


class TreeInferenceOutputTests(unittest.TestCase):
    """Validate random-forest output artifact construction."""

    def test_logits_mode_converts_predictions_to_probabilities(self) -> None:
        model = _FixedPredictionModel(
            np.array(
                [
                    [1.0, 2.0],
                    [3.0, 0.0],
                ],
                dtype=np.float32,
            )
        )
        loader = _loader()

        outputs = collect_tree_model_outputs(model, "logits", loader)

        np.testing.assert_allclose(outputs.logits.numpy(), model._predictions)
        np.testing.assert_allclose(outputs.probabilities.sum(dim=1).numpy(), np.ones(2))
        self.assertEqual(outputs.labels.tolist(), [0, 1])

    def test_softmax_mode_normalizes_predictions(self) -> None:
        model = _FixedPredictionModel(
            np.array(
                [
                    [2.0, 2.0],
                    [-1.0, 3.0],
                ],
                dtype=np.float32,
            )
        )
        loader = _loader()

        outputs = collect_tree_model_outputs(model, "softmax", loader)

        np.testing.assert_allclose(outputs.probabilities.sum(dim=1).numpy(), np.ones(2))
        np.testing.assert_allclose(outputs.probabilities[0].numpy(), np.array([0.5, 0.5]))
        self.assertTrue(torch.all(torch.isfinite(outputs.logits)))


def _loader() -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    images = torch.zeros((2, 3, 32, 32), dtype=torch.float32)
    labels = torch.tensor([0, 1])
    return DataLoader(TensorDataset(images, labels), batch_size=1)


if __name__ == "__main__":
    unittest.main()
