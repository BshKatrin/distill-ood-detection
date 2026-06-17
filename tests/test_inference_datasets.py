"""Tests for inference dataset transforms."""

from __future__ import annotations

import unittest

from PIL import Image
from torchvision.transforms import CenterCrop, Lambda, Resize

from distill_ood_detection.datasets.inference import ID_PREPROCESSING, _inference_transform


class InferenceDatasetTransformTests(unittest.TestCase):
    """Validate OOD image transforms for CIFAR-shaped inference."""

    def test_mnist_transform_uses_standard_inference_pipeline(self) -> None:
        transform = _inference_transform("cifar10")
        image = Image.new("L", (28, 28), color=0)
        preprocessing = ID_PREPROCESSING["cifar10"]

        transformed = transform(image)

        self.assertIsInstance(transform.transforms[0], Lambda)
        self.assertIsInstance(transform.transforms[1], Resize)
        self.assertIsInstance(transform.transforms[2], CenterCrop)
        self.assertEqual(transform.transforms[1].size, preprocessing["pre_size"])
        self.assertEqual(
            transform.transforms[2].size,
            (preprocessing["image_size"], preprocessing["image_size"]),
        )
        self.assertEqual(
            tuple(transformed.shape),
            (3, preprocessing["image_size"], preprocessing["image_size"]),
        )

    def test_imagenet_transform_uses_imagenet_size(self) -> None:
        transform = _inference_transform("imagenet")
        image = Image.new("RGB", (32, 32), color=0)
        preprocessing = ID_PREPROCESSING["imagenet"]

        transformed = transform(image)

        self.assertEqual(transform.transforms[1].size, preprocessing["pre_size"])
        self.assertEqual(
            transform.transforms[2].size,
            (preprocessing["image_size"], preprocessing["image_size"]),
        )
        self.assertEqual(
            tuple(transformed.shape),
            (3, preprocessing["image_size"], preprocessing["image_size"]),
        )


if __name__ == "__main__":
    unittest.main()
