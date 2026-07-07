"""Tests for Feature Denoising PCA reconstruction helpers."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch
from torch import nn

from distill_ood_detection.config import load_config
from distill_ood_detection.distillation.feature_denoising import (
    hidden_component_mse,
    feature_denoising_score_name,
    sample_pixel_block_keep_mask,
    sample_pixel_augmented_embedding_batch,
    sample_pixel_masked_embedding_batch,
    sample_pixel_masked_multilayer_l234_batch,
    sample_pixel_masked_multilayer_batch,
    sample_channel_keep_mask,
    sample_feature_denoising_keep_mask,
    sample_feature_denoising_pca_batch,
    sample_spatial_token_indices,
    sample_spatial_keep_mask,
)
from distill_ood_detection.distillation.perturbation import PcaProjector
from distill_ood_detection.models.student import build_student


class FeatureDenoisingTests(unittest.TestCase):
    """Validate Feature Denoising PCA masking behavior."""

    def test_config_matches_pca_input_and_output_shape(self) -> None:
        config = load_config(
            Path(
                "configs/students/feature_denoising/pca_masking/cifar_10/resnet18/"
                "linear_layer4_pca9_mask_p030.yaml"
            )
        )

        self.assertEqual(config.strategy.name, "feature_denoising")
        self.assertEqual(config.strategy.feature_denoising.pca_components, 9)
        self.assertEqual(config.student.input_shape, (9,))
        self.assertEqual(config.student.num_classes, 9)

    def test_feature_masking_configs_match_feature_shape(self) -> None:
        for path in (
            Path(
                "configs/students/feature_denoising/feature_masking/cifar_10/resnet18/"
                "spatial_layer4_mask_p030.yaml"
            ),
            Path(
                "configs/students/feature_denoising/feature_masking/cifar_10/resnet18/"
                "channel_layer4_mask_p030.yaml"
            ),
        ):
            with self.subTest(path=str(path)):
                config = load_config(path)
                self.assertEqual(config.strategy.name, "feature_denoising")
                self.assertEqual(config.student.kind, "feature_reconstructor")
                self.assertEqual(config.student.input_shape, (512, 4, 4))

    def test_spatial_token_prediction_config_matches_feature_shape(self) -> None:
        config = load_config(
            Path(
                "configs/students/feature_denoising/spatial_token_prediction/cifar_10/resnet18/"
                "token_layer4_blocks2_targets6.yaml"
            )
        )

        self.assertEqual(config.strategy.name, "feature_denoising")
        self.assertEqual(config.strategy.feature_denoising.method, "spatial_token_prediction")
        self.assertEqual(config.student.kind, "spatial_token_predictor")
        self.assertEqual(config.student.input_shape, (512, 4, 4))
        self.assertEqual(config.strategy.feature_denoising.target_block_count, 2)
        self.assertEqual(config.strategy.feature_denoising.target_token_count, 6)

    def test_pixel_masked_embedding_configs_match_pooled_shapes(self) -> None:
        expected = {
            "layer1": 64,
            "layer2": 128,
            "layer3": 256,
            "layer4": 512,
        }

        for layer, channels in expected.items():
            with self.subTest(layer=layer):
                config = load_config(
                    Path(
                        "configs/students/feature_denoising/pixel_masked_embedding/cifar_10/resnet18/"
                        f"mlp_{layer}_blocks2_scale_015_020.yaml"
                    )
                )

                self.assertEqual(config.strategy.name, "feature_denoising")
                self.assertEqual(
                    config.strategy.feature_denoising.method,
                    "pixel_masked_embedding_prediction",
                )
                self.assertEqual(config.student.kind, "mlp")
                self.assertEqual(config.student.feature_layer, layer)
                self.assertEqual(config.student.input_shape, (channels,))
                self.assertEqual(config.student.num_classes, channels)
                self.assertEqual(config.strategy.feature_denoising.image_mask_block_count, 2)

    def test_pixel_augmented_embedding_configs_match_pooled_shapes(self) -> None:
        expected = {
            "layer1": 64,
            "layer2": 128,
            "layer3": 256,
            "layer4": 512,
        }

        config_cases = {
            ("cifar_10", "resnet18"): (10, expected),
            ("cifar_100", "resnet18"): (100, expected),
            ("cifar_10", "resnet50"): (
                10,
                {
                    "layer1": 256,
                    "layer2": 512,
                    "layer3": 1024,
                    "layer4": 2048,
                },
            ),
            ("cifar_10", "vit_base_patch16_224"): (
                10,
                {
                    "layer3": 768,
                    "layer6": 768,
                    "layer10": 768,
                    "layer12": 768,
                },
            ),
            ("cifar_100", "vit_base_patch16_224"): (
                100,
                {
                    "layer3": 768,
                    "layer6": 768,
                    "layer10": 768,
                    "layer12": 768,
                },
            ),
        }

        for (dataset, architecture), (teacher_classes, layer_dims) in config_cases.items():
            for layer, channels in layer_dims.items():
                with self.subTest(dataset=dataset, architecture=architecture, layer=layer):
                    config = load_config(
                        Path(
                            "configs/students/feature_denoising/pixel_augmented_embedding/"
                            f"{dataset}/{architecture}/mlp_{layer}_aug_strong.yaml"
                        )
                    )

                    self.assertEqual(config.strategy.name, "feature_denoising")
                    self.assertEqual(
                        config.strategy.feature_denoising.method,
                        "pixel_augmented_embedding_prediction",
                    )
                    self.assertEqual(config.teacher.num_classes, teacher_classes)
                    self.assertEqual(config.student.kind, "mlp")
                    self.assertEqual(config.student.feature_layer, layer)
                    self.assertEqual(config.student.input_shape, (channels,))
                    self.assertEqual(config.student.num_classes, channels)
                    self.assertEqual(config.strategy.feature_denoising.rotation_degrees, 20.0)
                    if architecture == "vit_base_patch16_224":
                        self.assertEqual(config.dataset.image_size, 224)
                        self.assertEqual(config.strategy.feature_denoising.embedding_pool, "cls")

    def test_pixel_masked_multilayer_config_matches_concat_shape(self) -> None:
        config = load_config(
            Path(
                "configs/students/feature_denoising/pixel_masked_multilayer/cifar_10/resnet18/"
                "mlp_layer3_layer4_logits_blocks2_scale_015_020.yaml"
            )
        )

        self.assertEqual(config.strategy.name, "feature_denoising")
        self.assertEqual(config.strategy.feature_denoising.method, "pixel_masked_multilayer_prediction")
        self.assertEqual(config.student.kind, "mlp")
        self.assertEqual(config.student.input_shape, (768,))
        self.assertEqual(config.student.num_classes, 778)

    def test_pixel_masked_multilayer_l234_configs_match_concat_shape(self) -> None:
        expected = {
            Path(
                "configs/students/feature_denoising/pixel_masked_multilayer_l234/cifar_10/"
                "resnet18/mlp_layer2_layer3_layer4_logits_blocks2_scale_015_020.yaml"
            ): ((896,), 906),
            Path(
                "configs/students/feature_denoising/pixel_masked_multilayer_l234/cifar_100/"
                "resnet18/mlp_layer2_layer3_layer4_logits_blocks2_scale_015_020.yaml"
            ): ((896,), 996),
            Path(
                "configs/students/feature_denoising/pixel_masked_multilayer_l234/cifar_10/"
                "resnet50/mlp_layer2_layer3_layer4_logits_blocks2_scale_015_020.yaml"
            ): ((3584,), 3594),
        }
        for path, (input_shape, output_dim) in expected.items():
            with self.subTest(path=str(path)):
                config = load_config(path)
                self.assertEqual(
                    config.strategy.feature_denoising.method,
                    "pixel_masked_multilayer_l234_prediction",
                )
                self.assertEqual(config.student.input_shape, input_shape)
                self.assertEqual(config.student.num_classes, output_dim)

    def test_keep_mask_forces_at_least_one_hidden_component(self) -> None:
        projected = torch.ones(4, 3)

        torch.manual_seed(123)
        keep_mask = sample_feature_denoising_keep_mask(projected, mask_probability=0.01)

        self.assertEqual(tuple(keep_mask.shape), tuple(projected.shape))
        self.assertTrue(torch.all(keep_mask.sum(dim=1) < projected.shape[1]))

    def test_sample_pca_batch_uses_masked_projection_only_as_input(self) -> None:
        features = torch.tensor(
            [
                [[[-1.0]], [[2.0]], [[3.0]]],
                [[[4.0]], [[5.0]], [[-6.0]]],
            ]
        )
        projector = PcaProjector(
            mean=torch.zeros(3),
            components=torch.eye(3),
            component_stds=torch.tensor([1.0, 2.0, 3.0]),
        )
        config = load_config(
            Path(
                "configs/students/feature_denoising/pca_masking/cifar_10/resnet18/"
                "linear_layer4_pca9_mask_p030.yaml"
            )
        ).strategy.feature_denoising

        torch.manual_seed(123)
        batch = sample_feature_denoising_pca_batch(features, config, projector)

        self.assertEqual(tuple(batch.student_inputs.shape), (2, 3))
        expected_targets = torch.tensor(
            [
                [-1.0, 1.0, 1.0],
                [4.0, 2.5, -2.0],
            ]
        )
        torch.testing.assert_close(batch.targets, expected_targets)
        torch.testing.assert_close(batch.student_inputs, batch.targets * batch.keep_mask)

    def test_hidden_component_mse_scores_hidden_coordinates_only(self) -> None:
        predictions = torch.tensor([[100.0, 2.0, 9.0]])
        targets = torch.tensor([[0.0, 4.0, 3.0]])
        keep_mask = torch.tensor([[1.0, 0.0, 0.0]])

        loss = hidden_component_mse(predictions, targets, keep_mask)

        self.assertAlmostEqual(float(loss), 20.0)

    def test_feature_masks_force_hidden_units(self) -> None:
        features = torch.ones(3, 5, 4, 4)

        torch.manual_seed(123)
        spatial_mask = sample_spatial_keep_mask(features, mask_probability=0.01)
        channel_mask = sample_channel_keep_mask(features, mask_probability=0.01)

        self.assertEqual(tuple(spatial_mask.shape), (3, 1, 4, 4))
        self.assertEqual(tuple(channel_mask.shape), (3, 5, 1, 1))
        self.assertTrue(torch.all(spatial_mask.flatten(start_dim=1).sum(dim=1) < 16))
        self.assertTrue(torch.all(channel_mask.flatten(start_dim=1).sum(dim=1) < 5))

    def test_hidden_component_mse_broadcasts_feature_masks(self) -> None:
        predictions = torch.zeros(1, 2, 2, 2)
        targets = torch.ones(1, 2, 2, 2)
        keep_mask = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])

        loss = hidden_component_mse(predictions, targets, keep_mask)

        self.assertAlmostEqual(float(loss), 1.0)

    def test_spatial_feature_masking_uses_raw_features(self) -> None:
        config = load_config(
            Path(
                "configs/students/feature_denoising/feature_masking/cifar_10/resnet18/"
                "spatial_layer4_mask_p030.yaml"
            )
        ).strategy.feature_denoising
        features = torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 3, 2, 2)

        torch.manual_seed(123)
        batch = sample_feature_denoising_pca_batch(features, config)

        torch.testing.assert_close(batch.targets, features)
        torch.testing.assert_close(batch.student_inputs, features * batch.keep_mask)

    def test_spatial_token_indices_use_fixed_target_count(self) -> None:
        config = load_config(
            Path(
                "configs/students/feature_denoising/spatial_token_prediction/cifar_10/resnet18/"
                "token_layer4_blocks2_targets6.yaml"
            )
        ).strategy.feature_denoising

        torch.manual_seed(123)
        targets, context = sample_spatial_token_indices(
            height=4,
            width=4,
            config=config,
            device=torch.device("cpu"),
        )

        self.assertEqual(tuple(targets.shape), (6,))
        self.assertEqual(targets.numel() + context.numel(), 16)
        self.assertGreater(context.numel(), 0)
        self.assertEqual(
            set(targets.tolist()).intersection(set(context.tolist())),
            set(),
        )

    def test_spatial_token_prediction_batch_hides_target_content(self) -> None:
        config = load_config(
            Path(
                "configs/students/feature_denoising/spatial_token_prediction/cifar_10/resnet18/"
                "token_layer4_blocks2_targets6.yaml"
            )
        ).strategy.feature_denoising
        features = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)

        torch.manual_seed(123)
        batch = sample_feature_denoising_pca_batch(features, config)

        self.assertIsInstance(batch.student_inputs, dict)
        self.assertEqual(tuple(batch.targets.shape), (2, 6, 3))
        self.assertEqual(tuple(batch.keep_mask.shape), (2, 6, 3))
        self.assertTrue(torch.all(batch.keep_mask == 0.0))
        self.assertEqual(tuple(batch.student_inputs["visible_tokens"].shape), (2, 16, 3))
        self.assertEqual(tuple(batch.student_inputs["target_positions"].shape), (2, 6))

    def test_pixel_block_mask_zeros_image_regions(self) -> None:
        config = load_config(
            Path(
                "configs/students/feature_denoising/pixel_masked_embedding/cifar_10/resnet18/"
                "mlp_layer4_blocks2_scale_015_020.yaml"
            )
        ).strategy.feature_denoising
        images = torch.ones(4, 3, 32, 32)

        torch.manual_seed(123)
        keep_mask = sample_pixel_block_keep_mask(images, config)

        self.assertEqual(tuple(keep_mask.shape), (4, 1, 32, 32))
        hidden_fraction = (1.0 - keep_mask).flatten(start_dim=1).mean(dim=1)
        self.assertTrue(torch.all(hidden_fraction > 0.0))
        self.assertTrue(torch.all(hidden_fraction < 1.0))

    def test_pixel_masked_embedding_batch_uses_clean_and_masked_embeddings(self) -> None:
        config = load_config(
            Path(
                "configs/students/feature_denoising/pixel_masked_embedding/cifar_10/resnet18/"
                "mlp_layer4_blocks2_scale_015_020.yaml"
            )
        ).strategy.feature_denoising
        images = torch.ones(2, 3, 4, 4)
        forwarder = _ToyFeatureForwarder()

        torch.manual_seed(123)
        batch = sample_pixel_masked_embedding_batch(images, forwarder, config)

        self.assertEqual(tuple(batch.student_inputs.shape), (2, 3))
        self.assertEqual(tuple(batch.targets.shape), (2, 3))
        self.assertTrue(torch.all(batch.keep_mask == 0.0))
        self.assertTrue(torch.all(batch.targets > batch.student_inputs))
        self.assertEqual(
            feature_denoising_score_name("pixel_masked_embedding_prediction"),
            "feature_denoising_pixel_embedding_prediction_error",
        )

    def test_pixel_augmented_embedding_batch_uses_clean_and_augmented_embeddings(self) -> None:
        config = load_config(
            Path(
                "configs/students/feature_denoising/pixel_augmented_embedding/cifar_10/resnet18/"
                "mlp_layer4_aug_strong.yaml"
            )
        ).strategy.feature_denoising
        images = torch.ones(2, 3, 4, 4)
        forwarder = _ToyFeatureForwarder()

        torch.manual_seed(123)
        batch = sample_pixel_augmented_embedding_batch(
            images,
            forwarder,
            config,
            ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )

        self.assertEqual(tuple(batch.student_inputs.shape), (2, 3))
        self.assertEqual(tuple(batch.targets.shape), (2, 3))
        self.assertTrue(torch.all(batch.keep_mask == 0.0))
        self.assertFalse(torch.equal(batch.targets, batch.student_inputs))
        self.assertEqual(
            feature_denoising_score_name("pixel_augmented_embedding_prediction"),
            "feature_denoising_pixel_augmented_embedding_prediction_error",
        )

    def test_pixel_masked_multilayer_batch_concatenates_context_and_targets(self) -> None:
        config = load_config(
            Path(
                "configs/students/feature_denoising/pixel_masked_multilayer/cifar_10/resnet18/"
                "mlp_layer3_layer4_logits_blocks2_scale_015_020.yaml"
            )
        ).strategy.feature_denoising
        images = torch.ones(2, 3, 8, 8)
        forwarder = _ToyMultilayerForwarder()

        torch.manual_seed(123)
        batch = sample_pixel_masked_multilayer_batch(images, forwarder, config)

        self.assertEqual(tuple(batch.student_inputs.shape), (2, 768))
        self.assertEqual(tuple(batch.targets.shape), (2, 778))
        self.assertEqual(tuple(batch.keep_mask.shape), (2, 778))
        self.assertEqual(
            feature_denoising_score_name("pixel_masked_multilayer_prediction"),
            "feature_denoising_pixel_multilayer_prediction_error",
        )

    def test_pixel_masked_multilayer_l234_batch_concatenates_context_and_targets(self) -> None:
        config = load_config(
            Path(
                "configs/students/feature_denoising/pixel_masked_multilayer_l234/cifar_10/"
                "resnet18/mlp_layer2_layer3_layer4_logits_blocks2_scale_015_020.yaml"
            )
        ).strategy.feature_denoising
        images = torch.ones(2, 3, 8, 8)
        forwarder = _ToyMultilayerForwarder()

        torch.manual_seed(123)
        batch = sample_pixel_masked_multilayer_l234_batch(images, forwarder, config)

        self.assertEqual(tuple(batch.student_inputs.shape), (2, 896))
        self.assertEqual(tuple(batch.targets.shape), (2, 906))
        self.assertEqual(tuple(batch.keep_mask.shape), (2, 906))
        self.assertEqual(
            feature_denoising_score_name("pixel_masked_multilayer_l234_prediction"),
            "feature_denoising_pixel_multilayer_l234_prediction_error",
        )

    def test_feature_reconstructor_preserves_feature_shape(self) -> None:
        config = load_config(
            Path(
                "configs/students/feature_denoising/feature_masking/cifar_10/resnet18/"
                "spatial_layer4_mask_p030.yaml"
            )
        )
        model = build_student(config.student)

        outputs = model(torch.zeros(2, 512, 4, 4))

        self.assertEqual(tuple(outputs.shape), (2, 512, 4, 4))

    def test_spatial_token_predictor_preserves_target_shape(self) -> None:
        config = load_config(
            Path(
                "configs/students/feature_denoising/spatial_token_prediction/cifar_10/resnet18/"
                "token_layer4_blocks2_targets6.yaml"
            )
        )
        model = build_student(config.student)
        batch = sample_feature_denoising_pca_batch(
            torch.zeros(2, 512, 4, 4),
            config.strategy.feature_denoising,
        )

        outputs = model(batch.student_inputs)

        self.assertEqual(tuple(outputs.shape), (2, 6, 512))

class _ToyFeatureForwarder:
    def forward_to_features(self, images: torch.Tensor) -> torch.Tensor:
        pooled = images.mean(dim=(2, 3), keepdim=True)
        return pooled + 0.5


class _ToyMultilayerForwarder:
    def __init__(self) -> None:
        self.teacher = _ToyTeacher()


class _ToyTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = _RepeatLayer(3)
        self.bn1 = nn.Identity()
        self.relu = nn.Identity()
        self.maxpool = nn.Identity()
        self.layer1 = _RepeatLayer(8)
        self.layer2 = _RepeatLayer(128)
        self.layer3 = _RepeatLayer(256)
        self.layer4 = _RepeatLayer(512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10)


class _RepeatLayer(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = inputs.mean(dim=1, keepdim=True)
        return values.repeat(1, self.channels, 1, 1)


if __name__ == "__main__":
    unittest.main()
