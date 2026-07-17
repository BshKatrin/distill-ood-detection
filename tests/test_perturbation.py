"""Tests for perturbation-based distillation helpers."""

from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import torch

from distill_ood_detection.config import (
    ClippingLayerConfig,
    PerturbationConfig,
    load_config,
)
from distill_ood_detection.distillation.perturbation import (
    PcaProjector,
    _build_pca_masked_projection_batch,
    _clip_single_feature_map,
    apply_pixel_augmentation,
    build_pixel_student_inputs,
    build_sequential_clipping_batch,
    build_unperturbed_pixel_params,
    build_unperturbed_perturbation_batch,
    fit_pca_projector_from_activations,
    forward_clean_from_clipping_start,
    forward_to_clipping_start,
    sample_pixel_augmentation,
    sample_pixel_augmentation_params,
    sample_clipping_perturbation,
    sample_mc_dropout_perturbation,
    sample_perturbation,
    teacher_target_features,
    pool_pixel_features,
)
from distill_ood_detection.models.teacher import CifarResNet18


class PerturbationTests(unittest.TestCase):
    """Validate clipping perturbation behavior."""

    def test_perturbation_configs_match_student_input_shape(self) -> None:
        feature_shapes = {
            ("resnet18", "layer1"): (64, 32, 32),
            ("resnet18", "layer2"): (128, 16, 16),
            ("resnet18", "layer3"): (256, 8, 8),
            ("resnet18", "layer4"): (512, 4, 4),
            ("resnet50", "layer1"): (256, 32, 32),
            ("resnet50", "layer2"): (512, 16, 16),
            ("resnet50", "layer3"): (1024, 8, 8),
            ("resnet50", "layer4"): (2048, 4, 4),
        }

        for path in sorted(Path("configs/students/perturbation").rglob("*.yaml")):
            with self.subTest(path=str(path)):
                config = load_config(path)
                architecture = "resnet50" if "resnet50" in path.parts else "resnet18"
                perturbation = config.strategy.perturbation
                if perturbation.method == "clipping":
                    final_channels, final_height, final_width = feature_shapes[
                        (architecture, "layer4")
                    ]
                    embedding_dim = (
                        final_channels
                        if perturbation.embedding_pool == "avg"
                        else final_channels * final_height * final_width
                    )
                    perturbation_dim = 0
                    for layer_name, layer_config in perturbation.clipping_layers.items():
                        channels, height, width = feature_shapes[
                            (architecture, layer_name)
                        ]
                        if layer_config.clipping_mode == "constant":
                            perturbation_dim += 1
                        elif layer_config.clipping_mode == "spatial_dependent":
                            perturbation_dim += height * width
                        elif layer_config.clipping_mode == "channel_dependent":
                            perturbation_dim += channels
                        else:
                            self.fail(
                                f"Unexpected clipping mode in {path}: "
                                f"{layer_config.clipping_mode}"
                            )
                    self.assertEqual(
                        config.student.input_shape,
                        (embedding_dim + perturbation_dim,),
                    )
                    continue

                feature_layer = config.student.feature_layer
                feature_shape = feature_shapes.get((architecture, feature_layer))
                if feature_shape is None:
                    self.fail(f"Unexpected perturbation feature layer in {path}: {feature_layer}")
                channels, height, width = feature_shape
                feature_dim = channels * height * width
                if perturbation.method == "mc_dropout":
                    perturbation_dim = 0
                    expected_shape = (feature_dim + perturbation_dim,)
                elif perturbation.method == "pixel_augmentation":
                    embedding_dim = (
                        channels
                        if perturbation.embedding_pool == "avg"
                        else feature_dim
                    )
                    expected_shape = (embedding_dim + 6,)
                elif perturbation.method == "pixmix":
                    embedding_dim = (
                        channels
                        if perturbation.embedding_pool == "avg"
                        else feature_dim
                    )
                    expected_shape = (embedding_dim,)
                elif perturbation.method == "pca_projection":
                    expected_shape = (perturbation.pca_components,)
                elif perturbation.method == "pca_masked_projection":
                    expected_shape = (2 * perturbation.pca_components,)
                else:
                    self.fail(
                        f"Unexpected perturbation config in {path}: {perturbation}"
                    )

                self.assertEqual(config.student.input_shape, expected_shape)

    def test_clipping_modes_return_perturbation_aware_inputs(self) -> None:
        features = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
        for clipping_mode, u_shape, input_shape in (
            ("constant", (2, 1), (2, 49)),
            ("spatial_dependent", (2, 16), (2, 64)),
            ("channel_dependent", (2, 3), (2, 51)),
        ):
            with self.subTest(clipping_mode=clipping_mode):
                torch.manual_seed(123)
                config = ClippingLayerConfig(
                    u_min=0.25,
                    u_max=0.75,
                    clipping_mode=clipping_mode,
                )

                batch = sample_clipping_perturbation(features, config)

                self.assertEqual(tuple(batch.perturbed_features.shape), tuple(features.shape))
                self.assertEqual(tuple(batch.perturbations.shape), u_shape)
                self.assertEqual(tuple(batch.student_inputs.shape), input_shape)
                self.assertTrue(torch.all(batch.perturbed_features <= features))
                self.assertTrue(torch.all(batch.perturbations >= 0.25))
                self.assertTrue(torch.all(batch.perturbations <= 0.75))
                torch.testing.assert_close(
                    batch.student_inputs[:, :48],
                    torch.flatten(batch.perturbed_features, start_dim=1),
                )
                torch.testing.assert_close(
                    batch.student_inputs[:, 48:],
                    batch.perturbations,
                )

    def test_sequential_clipping_uses_final_layer4_pool_and_all_percentiles(self) -> None:
        teacher = CifarResNet18(num_classes=10).eval()
        images = torch.randn(2, 3, 32, 32)
        config = PerturbationConfig(
            method="clipping",
            embedding_pool="avg",
            clipping_layers={
                "layer3": ClippingLayerConfig(
                    clipping_mode="constant",
                    u_min=0.25,
                    u_max=0.75,
                ),
                "layer4": ClippingLayerConfig(
                    clipping_mode="spatial_dependent",
                    u_min=0.50,
                    u_max=1.00,
                ),
            },
        )

        start_features = forward_to_clipping_start(teacher, images, config)
        clean_logits = forward_clean_from_clipping_start(
            teacher,
            start_features,
            config,
        )
        unperturbed = build_sequential_clipping_batch(
            teacher,
            start_features,
            config,
            apply_perturbation=False,
        )
        torch.manual_seed(123)
        perturbed = build_sequential_clipping_batch(
            teacher,
            start_features,
            config,
            apply_perturbation=True,
        )

        self.assertEqual(tuple(start_features.shape), (2, 128, 16, 16))
        self.assertEqual(tuple(perturbed.final_features.shape), (2, 512, 4, 4))
        self.assertEqual(tuple(perturbed.student_inputs.shape), (2, 529))
        self.assertEqual(tuple(perturbed.perturbations["layer3"].shape), (2, 1))
        self.assertEqual(tuple(perturbed.perturbations["layer4"].shape), (2, 16))
        torch.testing.assert_close(unperturbed.teacher_logits, clean_logits)
        torch.testing.assert_close(
            unperturbed.student_inputs[:, :512],
            unperturbed.final_features.mean(dim=(-2, -1)),
        )
        self.assertFalse(
            torch.equal(perturbed.final_features, unperturbed.final_features)
        )

    def test_pixel_augmentation_samples_normalized_parameters(self) -> None:
        images = torch.zeros(4, 3, 32, 32)
        config = PerturbationConfig(
            method="pixel_augmentation",
            rotation_degrees=10.0,
            translate_fraction=0.10,
            scale_min=0.90,
            scale_max=1.10,
            brightness_delta=0.10,
            contrast_delta=0.20,
        )

        torch.manual_seed(123)
        raw_params, normalized_params = sample_pixel_augmentation_params(images, config)

        self.assertEqual(tuple(raw_params.shape), (4, 6))
        self.assertEqual(tuple(normalized_params.shape), (4, 6))
        self.assertTrue(torch.all(raw_params[:, 0] >= -10.0))
        self.assertTrue(torch.all(raw_params[:, 0] <= 10.0))
        self.assertTrue(torch.all(raw_params[:, 1] >= -3.0))
        self.assertTrue(torch.all(raw_params[:, 1] <= 3.0))
        self.assertTrue(torch.all(raw_params[:, 2] >= -3.0))
        self.assertTrue(torch.all(raw_params[:, 2] <= 3.0))
        self.assertTrue(torch.all(raw_params[:, 3] >= 0.90))
        self.assertTrue(torch.all(raw_params[:, 3] <= 1.10))
        self.assertTrue(torch.all(raw_params[:, 4] >= 0.90))
        self.assertTrue(torch.all(raw_params[:, 4] <= 1.10))
        self.assertTrue(torch.all(raw_params[:, 5] >= 0.80))
        self.assertTrue(torch.all(raw_params[:, 5] <= 1.20))
        self.assertTrue(torch.all(normalized_params >= -1.0))
        self.assertTrue(torch.all(normalized_params <= 1.0))

    def test_pixel_augmentation_applies_transform_and_renormalizes(self) -> None:
        normalization = ((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
        images = torch.zeros(2, 3, 8, 8)
        raw_params = torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [5.0, 1.0, -1.0, 1.0, 1.05, 1.1],
            ],
            dtype=torch.float32,
        )

        perturbed = apply_pixel_augmentation(images, raw_params, normalization)

        self.assertEqual(tuple(perturbed.shape), tuple(images.shape))
        self.assertEqual(perturbed.dtype, images.dtype)
        torch.testing.assert_close(perturbed[0], images[0])
        self.assertTrue(torch.any(perturbed[1] != images[1]))

    def test_pixel_augmentation_batch_and_student_inputs(self) -> None:
        normalization = ((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
        images = torch.zeros(2, 3, 8, 8)
        features = torch.ones(2, 4, 2, 2)
        config = PerturbationConfig(method="pixel_augmentation")

        torch.manual_seed(123)
        batch = sample_pixel_augmentation(images, config, normalization)
        student_inputs = build_pixel_student_inputs(
            features,
            batch.normalized_transform_params,
        )

        self.assertEqual(tuple(batch.perturbed_images.shape), tuple(images.shape))
        self.assertEqual(tuple(student_inputs.shape), (2, 22))
        torch.testing.assert_close(student_inputs[:, :16], torch.ones(2, 16))
        torch.testing.assert_close(
            build_unperturbed_pixel_params(images),
            torch.zeros(2, 6),
        )

    def test_pixel_student_inputs_support_global_average_pooling(self) -> None:
        features = torch.arange(2 * 4 * 2 * 2, dtype=torch.float32).reshape(
            2,
            4,
            2,
            2,
        )
        params = torch.zeros(2, 6)

        student_inputs = build_pixel_student_inputs(
            features,
            params,
            embedding_pool="avg",
        )

        self.assertEqual(tuple(student_inputs.shape), (2, 10))
        torch.testing.assert_close(
            student_inputs[:, :4],
            features.mean(dim=(-2, -1)),
        )
        torch.testing.assert_close(
            pool_pixel_features(features, "flatten"),
            torch.flatten(features, start_dim=1),
        )

    def test_rejects_non_convolutional_features(self) -> None:
        config = ClippingLayerConfig(
            clipping_mode="constant",
            u_min=0.0,
            u_max=1.0,
        )

        with self.assertRaises(ValueError):
            sample_clipping_perturbation(torch.zeros(2, 3), config)

        with self.assertRaises(ValueError):
            sample_mc_dropout_perturbation(
                torch.zeros(2, 3),
                PerturbationConfig(method="mc_dropout"),
            )

    def test_unperturbed_batch_preserves_features_with_neutral_perturbation(self) -> None:
        features = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
        config = PerturbationConfig(method="mc_dropout")

        batch = build_unperturbed_perturbation_batch(features, config)

        torch.testing.assert_close(batch.perturbed_features, features)
        torch.testing.assert_close(batch.percentiles, torch.ones_like(features))
        torch.testing.assert_close(batch.perturbations, torch.ones(2, 48))
        torch.testing.assert_close(
            batch.student_inputs,
            torch.flatten(features, start_dim=1),
        )

    def test_mc_dropout_returns_flattened_dropped_features(self) -> None:
        features = torch.ones(2, 3, 4, 4)
        config = PerturbationConfig(method="mc_dropout", dropout_probability=0.5)

        torch.manual_seed(123)
        batch = sample_mc_dropout_perturbation(features, config)

        self.assertEqual(tuple(batch.perturbed_features.shape), tuple(features.shape))
        self.assertEqual(tuple(batch.student_inputs.shape), (2, 48))
        self.assertEqual(tuple(batch.perturbations.shape), (2, 48))
        torch.testing.assert_close(
            batch.perturbed_features,
            batch.student_inputs.reshape_as(features),
        )
        self.assertTrue(torch.any(batch.student_inputs == 0.0))
        self.assertTrue(torch.any(batch.student_inputs == 2.0))

    def test_mc_dropout_modes_use_expected_mask_shapes(self) -> None:
        features = torch.ones(2, 3, 4, 4)
        for dropout_mode, mask_shape in (
            ("element", (2, 3, 4, 4)),
            ("channel", (2, 3, 1, 1)),
            ("spatial", (2, 1, 4, 4)),
        ):
            with self.subTest(dropout_mode=dropout_mode):
                torch.manual_seed(123)
                config = PerturbationConfig(
                    method="mc_dropout",
                    dropout_probability=0.5,
                    dropout_mode=dropout_mode,
                )

                batch = sample_mc_dropout_perturbation(features, config)

                self.assertEqual(tuple(batch.percentiles.shape), mask_shape)
                self.assertEqual(tuple(batch.student_inputs.shape), (2, 48))
                if dropout_mode == "channel":
                    dropped = batch.student_inputs.reshape_as(features)
                    for channel in range(features.shape[1]):
                        channel_values = dropped[:, channel, :, :]
                        self.assertTrue(torch.all(channel_values == channel_values[:, :1, :1]))
                if dropout_mode == "spatial":
                    dropped = batch.student_inputs.reshape_as(features)
                    for row in range(features.shape[2]):
                        for column in range(features.shape[3]):
                            location_values = dropped[:, :, row, column]
                            self.assertTrue(
                                torch.all(location_values == location_values[:, :1])
                            )

    def test_unperturbed_mc_dropout_batch_preserves_flattened_features(self) -> None:
        features = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
        config = PerturbationConfig(method="mc_dropout")

        batch = build_unperturbed_perturbation_batch(features, config)

        torch.testing.assert_close(batch.perturbed_features, features)
        torch.testing.assert_close(batch.student_inputs, torch.flatten(features, start_dim=1))
        torch.testing.assert_close(batch.perturbations, torch.ones(2, 48))

    def test_pca_projection_uses_fitted_components(self) -> None:
        activations = torch.arange(5 * 2 * 2 * 2, dtype=torch.float32).reshape(5, 2, 2, 2)

        with tempfile.TemporaryDirectory() as directory:
            activation_path = Path(directory) / "layer4.pt"
            torch.save(
                {
                    "dataset": "cifar10_train",
                    "split": "train",
                    "activations": activations,
                },
                activation_path,
            )

            projector = fit_pca_projector_from_activations(
                activation_path,
                n_components=3,
                expected_dataset="cifar10_train",
            )

        self.assertEqual(tuple(projector.mean.shape), (8,))
        self.assertEqual(tuple(projector.components.shape), (3, 8))

        batch = sample_perturbation(
            activations[:2],
            PerturbationConfig(method="pca_projection", pca_components=3),
            pca_projector=projector,
        )

        self.assertEqual(tuple(batch.student_inputs.shape), (2, 3))
        self.assertEqual(tuple(batch.perturbed_features.shape), (2, 2, 2, 2))

    def test_masked_pca_projection_concatenates_projection_and_component_keep_mask(self) -> None:
        activations = torch.arange(5 * 2 * 2 * 2, dtype=torch.float32).reshape(5, 2, 2, 2)

        with tempfile.TemporaryDirectory() as directory:
            activation_path = Path(directory) / "layer4.pt"
            torch.save(
                {
                    "dataset": "cifar10_train",
                    "split": "train",
                    "activations": activations,
                },
                activation_path,
            )
            projector = fit_pca_projector_from_activations(
                activation_path,
                n_components=3,
                expected_dataset="cifar10_train",
            )

        config = PerturbationConfig(
            method="pca_masked_projection",
            pca_components=3,
            pca_mask_probability=0.5,
        )
        torch.manual_seed(123)
        batch = sample_perturbation(
            activations[:2],
            config,
            pca_projector=projector,
        )

        flat = torch.flatten(activations[:2], start_dim=1)
        expected_projection = ((flat - projector.mean) @ projector.components.T) * batch.perturbations
        self.assertEqual(tuple(batch.perturbations.shape), (2, 3))
        self.assertEqual(tuple(batch.student_inputs.shape), (2, 6))
        self.assertTrue(torch.all((batch.perturbations == 0.0) | (batch.perturbations == 1.0)))
        torch.testing.assert_close(batch.student_inputs[:, :3], expected_projection)
        torch.testing.assert_close(batch.student_inputs[:, 3:], batch.perturbations)

    def test_masked_pca_projection_matches_explicit_component_masking(self) -> None:
        features = torch.arange(2 * 2 * 2 * 2, dtype=torch.float32).reshape(2, 2, 2, 2)
        activations = torch.arange(5 * 2 * 2 * 2, dtype=torch.float32).reshape(5, 2, 2, 2)

        with tempfile.TemporaryDirectory() as directory:
            activation_path = Path(directory) / "layer4.pt"
            torch.save(
                {
                    "dataset": "cifar10_train",
                    "split": "train",
                    "activations": activations,
                },
                activation_path,
            )
            projector = fit_pca_projector_from_activations(
                activation_path,
                n_components=3,
                expected_dataset="cifar10_train",
            )

        config = PerturbationConfig(
            method="pca_masked_projection",
            pca_components=3,
            pca_mask_probability=0.5,
        )
        torch.manual_seed(123)
        batch = sample_perturbation(features, config, pca_projector=projector)

        centered = torch.flatten(features, start_dim=1) - projector.mean
        explicit_rows = []
        for row, keep_mask in zip(centered, batch.perturbations, strict=True):
            masked_components = projector.components * keep_mask[:, None]
            explicit_rows.append(row @ masked_components.T)
        explicit_projection = torch.stack(explicit_rows)

        torch.testing.assert_close(batch.student_inputs[:, :3], explicit_projection)

    def test_masked_pca_projection_masks_q_columns_before_projection(self) -> None:
        features = torch.tensor(
            [
                [[[-1.0, 2.0], [3.0, 4.0]]],
                [[[5.0, -6.0], [7.0, 8.0]]],
            ]
        )
        projector = PcaProjector(
            mean=torch.zeros(4),
            components=torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0],
                ]
            ),
        )
        keep_mask = torch.tensor(
            [
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )

        batch = _build_pca_masked_projection_batch(features, projector, keep_mask)

        flat_features = torch.flatten(features, start_dim=1)
        q = projector.components.T
        explicit_projection = torch.stack(
            [
                embedding @ (q * mask.unsqueeze(0))
                for embedding, mask in zip(flat_features, keep_mask, strict=True)
            ]
        )
        self.assertEqual(tuple(q.shape), (4, 3))
        self.assertEqual(tuple(batch.student_inputs.shape), (2, 6))
        torch.testing.assert_close(batch.student_inputs[:, :3], explicit_projection)
        torch.testing.assert_close(batch.student_inputs[:, 3:], keep_mask)

    def test_unperturbed_masked_pca_keeps_all_components(self) -> None:
        features = torch.arange(2 * 2 * 2 * 2, dtype=torch.float32).reshape(2, 2, 2, 2)
        activations = torch.arange(5 * 2 * 2 * 2, dtype=torch.float32).reshape(5, 2, 2, 2)

        with tempfile.TemporaryDirectory() as directory:
            activation_path = Path(directory) / "layer4.pt"
            torch.save(
                {
                    "dataset": "cifar10_train",
                    "split": "train",
                    "activations": activations,
                },
                activation_path,
            )
            projector = fit_pca_projector_from_activations(
                activation_path,
                n_components=3,
                expected_dataset="cifar10_train",
            )

        batch = build_unperturbed_perturbation_batch(
            features,
            PerturbationConfig(method="pca_masked_projection", pca_components=3),
            pca_projector=projector,
        )

        torch.testing.assert_close(batch.perturbations, torch.ones(2, 3))
        torch.testing.assert_close(batch.student_inputs[:, :3], projector.transform(features))
        torch.testing.assert_close(batch.student_inputs[:, 3:], torch.ones(2, 3))

    def test_masked_pca_perturbed_teacher_target_uses_reconstruction(self) -> None:
        features = torch.arange(2 * 2 * 2 * 2, dtype=torch.float32).reshape(2, 2, 2, 2)
        projector = PcaProjector(
            mean=torch.zeros(8),
            components=torch.eye(3, 8),
        )
        keep_mask = torch.tensor(
            [
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        batch = _build_pca_masked_projection_batch(
            features,
            projector,
            keep_mask,
        )

        perturbed = teacher_target_features(
            batch,
            PerturbationConfig(
                method="pca_masked_projection",
                teacher_target="perturbed",
            ),
        )

        torch.testing.assert_close(perturbed, batch.perturbed_features)
        self.assertFalse(torch.equal(perturbed, features))

    def test_teacher_target_selects_clean_or_perturbed_features(self) -> None:
        features = torch.ones(2, 3, 4, 4)
        torch.manual_seed(123)
        batch = sample_mc_dropout_perturbation(
            features,
            PerturbationConfig(method="mc_dropout", dropout_probability=0.5),
        )

        clean = teacher_target_features(
            batch,
            PerturbationConfig(method="mc_dropout", teacher_target="clean"),
        )
        perturbed = teacher_target_features(
            batch,
            PerturbationConfig(method="mc_dropout", teacher_target="perturbed"),
        )

        torch.testing.assert_close(clean, features)
        torch.testing.assert_close(perturbed, batch.perturbed_features)
        self.assertFalse(torch.equal(clean, perturbed))

    def test_pca_fit_rejects_test_split_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            activation_path = Path(directory) / "layer4.pt"
            torch.save(
                {
                    "dataset": "cifar10_test",
                    "split": "test",
                    "activations": torch.zeros(5, 2, 2, 2),
                },
                activation_path,
            )

            with self.assertRaisesRegex(ValueError, "complete ID training-split"):
                fit_pca_projector_from_activations(
                    activation_path,
                    n_components=3,
                    expected_dataset="cifar10_train",
                )

    def test_vectorized_clipping_matches_torch_quantile_reference(self) -> None:
        feature_map = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(3, 4, 4)

        spatial_percentiles = torch.linspace(0.1, 0.9, steps=16).reshape(4, 4)
        spatial_actual = _clip_single_feature_map(
            feature_map,
            spatial_percentiles,
            ClippingLayerConfig(
                clipping_mode="spatial_dependent",
                u_min=0.0,
                u_max=1.0,
            ),
        )
        spatial_thresholds = torch.stack(
            [
                torch.stack(
                    [
                        torch.quantile(feature_map[:, row, column], spatial_percentiles[row, column])
                        for column in range(4)
                    ]
                )
                for row in range(4)
            ]
        )
        spatial_expected = torch.minimum(feature_map, spatial_thresholds.unsqueeze(0))
        torch.testing.assert_close(spatial_actual, spatial_expected)

        channel_percentiles = torch.tensor([0.1, 0.5, 0.9])
        channel_actual = _clip_single_feature_map(
            feature_map,
            channel_percentiles,
            ClippingLayerConfig(
                clipping_mode="channel_dependent",
                u_min=0.0,
                u_max=1.0,
            ),
        )
        channel_thresholds = torch.stack(
            [
                torch.quantile(feature_map[channel].flatten(), channel_percentiles[channel])
                for channel in range(3)
            ]
        )
        channel_expected = torch.minimum(feature_map, channel_thresholds[:, None, None])
        torch.testing.assert_close(channel_actual, channel_expected)


if __name__ == "__main__":
    unittest.main()
