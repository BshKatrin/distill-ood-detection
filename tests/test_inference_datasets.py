"""Tests for inference dataset transforms."""

from __future__ import annotations

import unittest

from PIL import Image
from torchvision.transforms import Pad

from distill_ood_detection.datasets.inference import _cifar_like_transform


class InferenceDatasetTransformTests(unittest.TestCase):
    """Validate OOD image transforms for CIFAR-shaped inference."""

    def test_mnist_transform_pads_to_cifar_shape(self) -> None:
        transform = _cifar_like_transform("cifar10", "mnist")
        image = Image.new("L", (28, 28), color=0)

        transformed = transform(image)

        self.assertIsInstance(transform.transforms[0], Pad)
        self.assertEqual(tuple(transformed.shape), (3, 32, 32))


if __name__ == "__main__":
    unittest.main()
