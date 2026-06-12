"""Tests for tree-based student inference outputs."""

from __future__ import annotations

import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from distill_ood_detection.config import PerturbationConfig
from distill_ood_detection.inference import collect_tree_model_outputs
from distill_ood_detection.inference.outputs import (
    collect_feature_tree_model_outputs,
    collect_perturbed_teacher_outputs,
    collect_perturbation_model_outputs,
    collect_perturbation_tree_model_outputs,
)


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

    def test_rejects_removed_cross_entropy_mode(self) -> None:
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

        with self.assertRaises(ValueError):
            collect_tree_model_outputs(model, "cross_entropy", loader)

    def test_feature_tree_outputs_use_extracted_features(self) -> None:
        model = _FeatureShapeModel()
        loader = _loader()
        feature_extractor = _FeatureExtractor()

        outputs = collect_feature_tree_model_outputs(
            model,
            "logits",
            feature_extractor,
            loader,
            torch.device("cpu"),
        )

        self.assertEqual(model.feature_shape, (2, 4))
        self.assertEqual(tuple(outputs.logits.shape), (2, 2))
        self.assertEqual(outputs.labels.tolist(), [0, 1])

    def test_perturbation_outputs_keep_draw_dimension(self) -> None:
        loader = _loader()
        forwarder = _PerturbationForwarder()
        config = PerturbationConfig(evaluation_draws=3)
        student = _PerturbationStudent()

        outputs = collect_perturbation_model_outputs(
            student,
            forwarder,
            config,
            loader,
            torch.device("cpu"),
        )
        teacher_outputs = collect_perturbed_teacher_outputs(
            forwarder,
            config,
            loader,
            torch.device("cpu"),
        )

        self.assertEqual(tuple(outputs.logits.shape), (2, 3, 2))
        self.assertEqual(tuple(outputs.probabilities.shape), (2, 3, 2))
        self.assertEqual(tuple(teacher_outputs.logits.shape), (2, 3, 2))
        self.assertEqual(tuple(teacher_outputs.probabilities.shape), (2, 3, 2))
        self.assertEqual(outputs.labels.tolist(), [0, 1])

    def test_perturbation_tree_outputs_keep_draw_dimension(self) -> None:
        loader = _loader()
        forwarder = _PerturbationForwarder()
        config = PerturbationConfig(evaluation_draws=3)
        model = _FeatureShapeModel()

        outputs = collect_perturbation_tree_model_outputs(
            model,
            "logits",
            forwarder,
            config,
            loader,
            torch.device("cpu"),
        )

        self.assertEqual(tuple(outputs.logits.shape), (2, 3, 2))
        self.assertEqual(tuple(outputs.probabilities.shape), (2, 3, 2))
        self.assertEqual(outputs.labels.tolist(), [0, 1])


def _loader() -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    images = torch.zeros((2, 3, 32, 32), dtype=torch.float32)
    labels = torch.tensor([0, 1])
    return DataLoader(TensorDataset(images, labels), batch_size=1)


class _FeatureShapeModel:
    def __init__(self) -> None:
        self.feature_shape: tuple[int, int] | None = None

    def predict(self, features: np.ndarray) -> np.ndarray:
        self.feature_shape = features.shape
        return np.zeros((features.shape[0], 2), dtype=np.float32)


class _FeatureExtractor(torch.nn.Module):
    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = images.shape[0]
        logits = torch.zeros((batch_size, 2), dtype=torch.float32)
        features = torch.ones((batch_size, 1, 2, 2), dtype=torch.float32)
        return logits, features


class _PerturbationForwarder(torch.nn.Module):
    def forward_to_features(self, images: torch.Tensor) -> torch.Tensor:
        return torch.ones((images.shape[0], 1, 2, 2), dtype=torch.float32)

    def forward_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return torch.zeros((features.shape[0], 2), dtype=torch.float32)


class _PerturbationStudent(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros((inputs.shape[0], 2), dtype=torch.float32)


if __name__ == "__main__":
    unittest.main()
