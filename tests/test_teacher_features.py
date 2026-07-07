"""Tests for teacher feature extraction helpers."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from distill_ood_detection.models.teacher import (
    CifarResNet18,
    CifarResNet50,
    HuggingFaceImageClassifier,
    ResNetFeatureForwarder,
    TeacherFeatureExtractor,
    VitClsFeatureForwarder,
    _infer_architecture_from_hf_model_id,
)


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


class ToyHuggingFaceOutputs:
    """Minimal Hugging Face-style output container for tests."""

    def __init__(
        self,
        logits: torch.Tensor,
        hidden_states: tuple[torch.Tensor, ...],
    ) -> None:
        self.logits = logits
        self.hidden_states = hidden_states


class ToyVitTeacher(nn.Module):
    """Small model that mimics ViT hidden-state outputs."""

    def forward(
        self,
        images: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> ToyHuggingFaceOutputs:
        """Return logits and optional hidden states."""

        batch_size = images.shape[0]
        logits = torch.zeros(batch_size, 3)
        hidden_states = tuple(
            torch.full((batch_size, 5, 7), fill_value=float(layer))
            for layer in range(13)
        )
        if not output_hidden_states:
            hidden_states = ()
        return ToyHuggingFaceOutputs(logits=logits, hidden_states=hidden_states)


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

    def test_resnet_forwarder_resumes_from_feature_layer(self) -> None:
        teacher = CifarResNet18(num_classes=3)
        teacher.eval()
        forwarder = ResNetFeatureForwarder(teacher, "layer3")
        images = torch.randn(2, 3, 32, 32)

        features = forwarder.forward_to_features(images)
        resumed_logits = forwarder.forward_from_features(features)
        full_logits = teacher(images)

        self.assertEqual(tuple(features.shape), (2, 256, 8, 8))
        torch.testing.assert_close(resumed_logits, full_logits)

    def test_resnet_forwarder_rejects_non_residual_layer(self) -> None:
        teacher = CifarResNet18(num_classes=3)

        with self.assertRaises(ValueError):
            ResNetFeatureForwarder(teacher, "avgpool")

    def test_resnet50_forwarder_resumes_from_feature_layer(self) -> None:
        teacher = CifarResNet50(num_classes=3)
        teacher.eval()
        forwarder = ResNetFeatureForwarder(teacher, "layer4")
        images = torch.randn(2, 3, 32, 32)

        features = forwarder.forward_to_features(images)
        resumed_logits = forwarder.forward_from_features(features)
        full_logits = teacher(images)

        self.assertEqual(tuple(features.shape), (2, 2048, 4, 4))
        torch.testing.assert_close(resumed_logits, full_logits)

    def test_vit_forwarder_extracts_cls_token_from_layer(self) -> None:
        teacher = HuggingFaceImageClassifier(ToyVitTeacher())
        forwarder = VitClsFeatureForwarder(teacher, "layer6")
        images = torch.randn(2, 3, 224, 224)

        logits, features = forwarder(images)

        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertEqual(tuple(features.shape), (2, 7))
        torch.testing.assert_close(features, torch.full((2, 7), 6.0))

    def test_infers_architecture_from_hf_model_id(self) -> None:
        self.assertEqual(
            _infer_architecture_from_hf_model_id("edadaltocg/resnet18_cifar10"),
            "resnet18",
        )
        self.assertEqual(
            _infer_architecture_from_hf_model_id("edadaltocg/resnet50_cifar100"),
            "resnet50",
        )
        self.assertEqual(
            _infer_architecture_from_hf_model_id("nateraw/vit-base-patch16-224-cifar10"),
            "vit",
        )

    def test_rejects_unknown_teacher_architecture(self) -> None:
        with self.assertRaises(ValueError):
            _infer_architecture_from_hf_model_id("edadaltocg/wide_resnet_cifar10")


if __name__ == "__main__":
    unittest.main()
