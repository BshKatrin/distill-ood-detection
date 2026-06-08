"""Tests for teacher feature extraction helpers."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from distill_ood_detection.models.teacher import TeacherFeatureExtractor


class ToyTeacher(nn.Module):
    """Small model with a named feature layer for hook tests."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Conv2d(3, 4, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(4, 2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return logits for a batch of images."""

        features = self.features(images)
        pooled = torch.flatten(self.pool(features), start_dim=1)
        return self.classifier(pooled)


class TeacherFeatureExtractorTests(unittest.TestCase):
    """Validate configured teacher feature extraction."""

    def test_extracts_named_layer_features_with_logits(self) -> None:
        teacher = ToyTeacher()
        extractor = TeacherFeatureExtractor(teacher, "features")
        images = torch.zeros(2, 3, 8, 8)

        logits, features = extractor(images)

        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertEqual(tuple(features.shape), (2, 4, 8, 8))

    def test_rejects_unknown_feature_layer(self) -> None:
        teacher = ToyTeacher()

        with self.assertRaises(ValueError):
            TeacherFeatureExtractor(teacher, "missing")


if __name__ == "__main__":
    unittest.main()
