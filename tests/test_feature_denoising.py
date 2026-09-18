"""Tests for Feature Denoising PCA reconstruction helpers."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from distill_ood_detection.config import (
    FeatureDenoisingConfig,
    OptimizerConfig,
    ResolvedTrainingMethodConfig,
    StudentConfig,
    load_config,
    parse_config,
)
from distill_ood_detection.distillation.feature_denoising import (
    ChannelGroups,
    ClassChannelCorruptionBank,
    FeatureNormalizer,
    collect_channel_knn_reconstruction_scores,
    collect_feature_denoising_reconstruction_scores,
    feature_denoising_predictions,
    feature_denoising_score_name,
    fit_class_channel_corruption_bank,
    fit_feature_normalizer_from_loader,
    hidden_component_mse,
    load_channel_knn_index,
    load_channel_groups,
    load_class_channel_corruption_bank,
    load_feature_normalizer,
    sample_channel_keep_mask,
    sample_channel_group_keep_mask,
    sample_channel_group_stratified_keep_mask,
    sample_confusion_channel_replacement_batch,
    sample_feature_denoising_keep_mask,
    sample_feature_denoising_pca_batch,
    sample_pixel_augmented_embedding_batch,
    sample_pixel_block_keep_mask,
    sample_pixel_masked_embedding_batch,
    sample_pixel_masked_multilayer_batch,
    sample_pixel_masked_multilayer_l234_batch,
    sample_spatial_block_hidden_mask,
    sample_spatial_keep_mask,
    sample_spatial_token_indices,
    save_class_channel_corruption_bank,
    save_feature_normalizer,
    train_feature_denoising_student,
)
from distill_ood_detection.distillation.perturbation import PcaProjector
from distill_ood_detection.evaluation.channel_grouping import (
    hierarchical_channel_linkage,
)
from distill_ood_detection.models.student import build_student


class FeatureDenoisingTests(unittest.TestCase):
    """Validate Feature Denoising PCA masking behavior."""

    def test_knn_channel_masking_configs_match_resnet18_shapes(self) -> None:
        expected_shapes = {
            "layer3": (256, 8, 8),
            "layer4": (512, 4, 4),
        }
        for dataset in ("cifar_10", "cifar_100"):
            for layer, expected_shape in expected_shapes.items():
                path = Path(
                    "configs/students/feature_denoising/knn_channel_masking/"
                    f"{dataset}/resnet18/knn_{layer}_k10_mask_p020.yaml"
                )
                with self.subTest(path=str(path)):
                    config = load_config(path)
                    denoising = config.strategy.feature_denoising
                    self.assertEqual(
                        denoising.method,
                        "channel_masked_knn_reconstruction",
                    )
                    self.assertEqual(config.student.kind, "knn")
                    self.assertEqual(config.student.input_shape, expected_shape)
                    self.assertEqual(denoising.k_neighbors, 10)
                    self.assertEqual(denoising.mask_probability, 0.2)

    def test_knn_channel_group_masking_configs_match_resnet18_shapes(self) -> None:
        expected_shapes = {
            "layer3": (256, 8, 8),
            "layer4": (512, 4, 4),
        }
        for dataset in ("cifar_10", "cifar_100"):
            for layer, expected_shape in expected_shapes.items():
                path = Path(
                    "configs/students/feature_denoising/knn_channel_group_masking/"
                    f"{dataset}/resnet18/knn_{layer}_k10_d050.yaml"
                )
                with self.subTest(path=str(path)):
                    config = load_config(path)
                    denoising = config.strategy.feature_denoising
                    self.assertEqual(
                        denoising.method,
                        "channel_group_masked_knn_reconstruction",
                    )
                    self.assertEqual(config.student.kind, "knn")
                    self.assertEqual(config.student.input_shape, expected_shape)
                    self.assertEqual(denoising.k_neighbors, 10)
                    self.assertEqual(denoising.channel_group_distance_threshold, 0.5)
                    self.assertEqual(denoising.evaluation_draws, 10)

    def test_load_channel_knn_index_validates_activation_provenance(self) -> None:
        activations = torch.randn(5, 3, 2, 2)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "layer3.pt"
            torch.save(
                {
                    "dataset": "toy_train",
                    "split": "train",
                    "layer": "layer3",
                    "activations": activations,
                },
                path,
            )
            index = load_channel_knn_index(
                path,
                torch.device("cpu"),
                expected_dataset="toy_train",
                expected_layer="layer3",
                expected_feature_shape=(3, 2, 2),
            )

        self.assertEqual(index.reference_count, 5)
        self.assertEqual(index.feature_shape, (3, 2, 2))

    def test_collect_channel_knn_scores_retains_neighbors_per_draw(self) -> None:
        references = torch.randn(6, 3, 2, 2)
        queries = torch.randn(4, 3, 2, 2)
        labels = torch.tensor([0, 1, 0, 1])
        index_path: Path
        with TemporaryDirectory() as directory:
            index_path = Path(directory) / "layer3.pt"
            torch.save(
                {
                    "dataset": "toy_train",
                    "split": "train",
                    "layer": "layer3",
                    "activations": references,
                },
                index_path,
            )
            index = load_channel_knn_index(
                index_path,
                torch.device("cpu"),
                expected_dataset="toy_train",
                expected_layer="layer3",
                expected_feature_shape=(3, 2, 2),
            )
            torch.manual_seed(7)
            scores = collect_channel_knn_reconstruction_scores(
                loader=DataLoader(
                    TensorDataset(queries, labels),
                    batch_size=2,
                ),
                device=torch.device("cpu"),
                perturbation_forwarder=_IdentityFeatureForwarder(),
                feature_denoising_config=FeatureDenoisingConfig(
                    method="channel_masked_knn_reconstruction",
                    mask_probability=0.2,
                    evaluation_draws=2,
                    k_neighbors=2,
                    knn_query_batch_size=1,
                    knn_reference_chunk_size=3,
                ),
                index=index,
            )

        torch.testing.assert_close(scores["labels"], labels)
        self.assertEqual(tuple(scores["neighbor_indices"].shape), (4, 2, 2))
        self.assertEqual(
            tuple(scores["neighbor_squared_distances"].shape),
            (4, 2, 2),
        )
        self.assertEqual(tuple(scores["visible_channel_counts"].shape), (4, 2))
        self.assertTrue(torch.all(torch.isfinite(scores["scores"])))

    def test_collect_channel_group_knn_scores_uses_complete_groups(self) -> None:
        references = torch.randn(6, 4, 2, 2)
        queries = torch.randn(3, 4, 2, 2)
        labels = torch.tensor([0, 1, 0])
        groups = ChannelGroups(
            membership=torch.tensor(
                [
                    [True, True, False, False],
                    [False, False, True, False],
                    [False, False, False, True],
                ]
            ),
            distance_threshold=0.5,
            source_path="toy.pt",
        )
        index_path: Path
        with TemporaryDirectory() as directory:
            index_path = Path(directory) / "layer3.pt"
            torch.save(
                {
                    "dataset": "toy_train",
                    "split": "train",
                    "layer": "layer3",
                    "activations": references,
                },
                index_path,
            )
            index = load_channel_knn_index(
                index_path,
                torch.device("cpu"),
                expected_dataset="toy_train",
                expected_layer="layer3",
                expected_feature_shape=(4, 2, 2),
            )
            torch.manual_seed(7)
            scores = collect_channel_knn_reconstruction_scores(
                loader=DataLoader(TensorDataset(queries, labels), batch_size=3),
                device=torch.device("cpu"),
                perturbation_forwarder=_IdentityFeatureForwarder(),
                feature_denoising_config=FeatureDenoisingConfig(
                    method="channel_group_masked_knn_reconstruction",
                    channel_group_path="toy.pt",
                    channel_group_distance_threshold=0.5,
                    evaluation_draws=3,
                    k_neighbors=2,
                    knn_query_batch_size=2,
                    knn_reference_chunk_size=3,
                ),
                index=index,
                channel_groups=groups,
            )

        sampled = scores["sampled_channel_group_indices"]
        expected_visible = (4 - groups.group_sizes[sampled]).to(dtype=torch.int32)
        self.assertEqual(tuple(sampled.shape), (3, 3))
        torch.testing.assert_close(scores["visible_channel_counts"], expected_visible)
        self.assertTrue(torch.all(torch.isfinite(scores["scores"])))

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

    def test_spatial_block_residual_configs_match_layer_shapes(self) -> None:
        expected = {
            "layer1": ((65, 32, 32), 64, (3, 5)),
            "layer2": ((129, 16, 16), 128, (3, 5)),
            "layer3": ((257, 8, 8), 256, (1, 3)),
            "layer4": ((513, 4, 4), 512, (1,)),
        }
        for dataset in ("cifar_10", "cifar_100"):
            for layer, (input_shape, output_channels, block_sizes) in expected.items():
                suffix = (
                    "blocks_3_5"
                    if layer in {"layer1", "layer2"}
                    else "blocks_1_3"
                    if layer == "layer3"
                    else "block_1"
                )
                path = Path(
                    "configs/students/feature_denoising/feature_masking/"
                    f"{dataset}/resnet18/residual_{layer}_{suffix}_mean.yaml"
                )
                with self.subTest(path=str(path)):
                    config = load_config(path)
                    self.assertEqual(
                        config.strategy.feature_denoising.method,
                        "spatial_block_residual_reconstruction",
                    )
                    self.assertEqual(config.student.kind, "feature_residual_denoiser")
                    self.assertEqual(config.student.input_shape, input_shape)
                    self.assertEqual(config.student.num_classes, output_channels)
                    self.assertEqual(
                        config.strategy.feature_denoising.spatial_mask_block_sizes,
                        block_sizes,
                    )

    def test_resnet50_spatial_block_residual_configs_use_capped_width(self) -> None:
        expected = {
            "layer1": ((257, 32, 32), 256, (3, 5)),
            "layer2": ((513, 16, 16), 512, (3, 5)),
            "layer3": ((1025, 8, 8), 1024, (1, 3)),
            "layer4": ((2049, 4, 4), 2048, (1,)),
        }
        for dataset in ("cifar_10", "cifar_100"):
            for layer, (input_shape, output_channels, block_sizes) in expected.items():
                suffix = (
                    "blocks_3_5"
                    if layer in {"layer1", "layer2"}
                    else "blocks_1_3"
                    if layer == "layer3"
                    else "block_1"
                )
                path = Path(
                    "configs/students/feature_denoising/feature_masking/"
                    f"{dataset}/resnet50/residual_{layer}_{suffix}_mean.yaml"
                )
                with self.subTest(path=str(path)):
                    config = load_config(path)
                    self.assertEqual(config.dataset.batch_size, 128)
                    self.assertEqual(config.student.input_shape, input_shape)
                    self.assertEqual(config.student.num_classes, output_channels)
                    self.assertEqual(config.student.hidden_channels, (64,))
                    self.assertEqual(
                        config.strategy.feature_denoising.spatial_mask_block_sizes,
                        block_sizes,
                    )

    def test_channel_residual_configs_match_backbone_shapes(self) -> None:
        shapes = {
            "resnet18": {
                "layer1": (64, 32, 32),
                "layer2": (128, 16, 16),
                "layer3": (256, 8, 8),
                "layer4": (512, 4, 4),
            },
            "resnet50": {
                "layer1": (256, 32, 32),
                "layer2": (512, 16, 16),
                "layer3": (1024, 8, 8),
                "layer4": (2048, 4, 4),
            },
        }
        for dataset in ("cifar_10", "cifar_100"):
            for backbone, layer_shapes in shapes.items():
                for layer, input_shape in layer_shapes.items():
                    path = Path(
                        "configs/students/feature_denoising/feature_masking/"
                        f"{dataset}/{backbone}/channel_residual_{layer}_mask_p020.yaml"
                    )
                    with self.subTest(path=str(path)):
                        config = load_config(path)
                        self.assertEqual(
                            config.strategy.feature_denoising.method,
                            "channel_masked_residual_reconstruction",
                        )
                        self.assertEqual(
                            config.strategy.feature_denoising.mask_probability,
                            0.2,
                        )
                        self.assertEqual(config.student.input_shape, input_shape)
                        self.assertEqual(config.student.num_classes, input_shape[0])
                        self.assertEqual(
                            config.dataset.batch_size,
                            256 if backbone == "resnet18" else 128,
                        )
                        self.assertEqual(
                            config.student.hidden_channels,
                            () if backbone == "resnet18" else (64,),
                        )

    def test_confusion_channel_replacement_launch_configs_cover_all_layers(
        self,
    ) -> None:
        layer_shapes = {
            "layer1": (64, 32, 32),
            "layer2": (128, 16, 16),
            "layer3": (256, 8, 8),
            "layer4": (512, 4, 4),
        }
        for dataset, class_count in (("cifar_10", 10), ("cifar_100", 100)):
            for variant, method in (
                (
                    "direct",
                    "confusion_channel_replacement_reconstruction",
                ),
                (
                    "residual",
                    "confusion_channel_replacement_residual_reconstruction",
                ),
            ):
                for layer, input_shape in layer_shapes.items():
                    path = Path(
                        "configs/students/feature_denoising/feature_masking/"
                        f"{dataset}/resnet18/"
                        f"confusion_channel_replacement_{variant}_{layer}_p020.yaml"
                    )
                    with self.subTest(path=str(path)):
                        config = load_config(path)
                        denoising = config.strategy.feature_denoising
                        self.assertEqual(denoising.method, method)
                        self.assertEqual(
                            config.teacher.num_classes,
                            class_count,
                        )
                        self.assertEqual(config.student.feature_layer, layer)
                        self.assertEqual(
                            config.student.input_shape,
                            input_shape,
                        )
                        self.assertEqual(
                            config.student.num_classes,
                            input_shape[0],
                        )
                        self.assertEqual(denoising.mask_probability, 0.2)
                        self.assertEqual(denoising.prototype_count, 50)
                        self.assertEqual(
                            denoising.class_statistics_epsilon,
                            1.0e-6,
                        )
                        self.assertEqual(denoising.confusion_split, "test")

    def test_channel_group_masking_configs_cover_both_datasets_and_all_layers(
        self,
    ) -> None:
        expected_shapes = {
            "layer1": (64, 32, 32),
            "layer2": (128, 16, 16),
            "layer3": (256, 8, 8),
            "layer4": (512, 4, 4),
        }
        for dataset in ("cifar_10", "cifar_100"):
            for layer, expected_shape in expected_shapes.items():
                path = Path(
                    "configs/students/feature_denoising/channel_group_masking/"
                    f"{dataset}/resnet18/cluster_{layer}_d050.yaml"
                )
                with self.subTest(path=str(path)):
                    config = load_config(path)
                    denoising = config.strategy.feature_denoising
                    self.assertEqual(
                        denoising.method,
                        "channel_group_masked_residual_reconstruction",
                    )
                    self.assertEqual(config.student.input_shape, expected_shape)
                    self.assertEqual(denoising.channel_group_distance_threshold, 0.5)
                    self.assertEqual(denoising.evaluation_draws, 10)
                    self.assertIn(f"/{layer}.pt", denoising.channel_group_path or "")

    def test_channel_group_stratified_layer4_config(self) -> None:
        path = Path(
            "configs/students/feature_denoising/channel_group_stratified_masking/"
            "cifar_10/resnet18/cluster_layer4_d050_min4_p025.yaml"
        )
        config = load_config(path)
        denoising = config.strategy.feature_denoising

        self.assertEqual(
            denoising.method,
            "channel_group_stratified_masked_residual_reconstruction",
        )
        self.assertEqual(config.student.feature_layer, "layer4")
        self.assertEqual(config.student.input_shape, (512, 4, 4))
        self.assertEqual(denoising.channel_group_distance_threshold, 0.5)
        self.assertEqual(denoising.channel_group_min_size, 4)
        self.assertEqual(denoising.channel_group_mask_fraction, 0.25)
        self.assertEqual(denoising.evaluation_draws, 10)

    def test_nmf_channel_group_masking_configs_cover_requested_grid(self) -> None:
        layer_settings = {
            "layer2": ((128, 16, 16), "070", 0.7),
            "layer3": ((256, 8, 8), "060", 0.6),
            "layer4": ((512, 4, 4), "060", 0.6),
        }
        variants = {
            "channel_group_masking": (
                "channel_group_masked_residual_reconstruction",
                "",
            ),
            "channel_group_stratified_masking": (
                "channel_group_stratified_masked_residual_reconstruction",
                "_p025",
            ),
        }
        for dataset in ("cifar_10", "cifar_100"):
            for variant, (method, suffix) in variants.items():
                for layer, (shape, distance_code, threshold) in layer_settings.items():
                    path = Path(
                        f"configs/students/feature_denoising/{variant}/"
                        f"nmf_latent_cosine/{dataset}/resnet18/"
                        f"cluster_{layer}_d{distance_code}{suffix}.yaml"
                    )
                    with self.subTest(path=str(path)):
                        config = load_config(path)
                        denoising = config.strategy.feature_denoising
                        self.assertEqual(denoising.method, method)
                        self.assertEqual(config.student.input_shape, shape)
                        self.assertEqual(
                            denoising.channel_group_distance_threshold,
                            threshold,
                        )
                        self.assertEqual(denoising.evaluation_draws, 10)
                        self.assertIn(
                            f"nmf_latent_cosine/{dataset}/resnet18/"
                            f"channel_groups/{layer}.pt",
                            denoising.channel_group_path or "",
                        )
                        if variant == "channel_group_stratified_masking":
                            self.assertEqual(denoising.channel_group_min_size, 1)
                            self.assertEqual(
                                denoising.channel_group_mask_fraction,
                                0.25,
                            )

    def test_resnet50_nmf_stratified_configs_are_generic_students(self) -> None:
        layer_settings = {
            "layer2": ((512, 16, 16), "060", 0.6),
            "layer3": ((1024, 8, 8), "060", 0.6),
            "layer4": ((2048, 4, 4), "055", 0.55),
        }
        for dataset in ("cifar_10", "cifar_100"):
            for layer, (shape, distance_code, threshold) in layer_settings.items():
                path = Path(
                    "configs/students/feature_denoising/"
                    "channel_group_stratified_masking/nmf_latent_cosine/"
                    f"{dataset}/resnet50/"
                    f"cluster_{layer}_d{distance_code}_p025.yaml"
                )
                with self.subTest(path=str(path)):
                    config = load_config(path)
                    denoising = config.strategy.feature_denoising
                    self.assertEqual(config.student.input_shape, shape)
                    self.assertEqual(config.student.hidden_channels, (64,))
                    self.assertIsNone(denoising.channel_group_index)
                    self.assertEqual(denoising.channel_group_min_size, 1)
                    self.assertEqual(denoising.channel_group_mask_fraction, 0.25)
                    self.assertEqual(
                        denoising.channel_group_distance_threshold,
                        threshold,
                    )
                    self.assertEqual(denoising.evaluation_draws, 10)

    def test_load_channel_groups_validates_partition_and_provenance(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "layer3.pt"
            torch.save(
                {
                    "dataset": "toy_train",
                    "split": "train",
                    "layer": "layer3",
                    "feature_shape": (5, 2, 2),
                    "cut_groups": {"0.5": [[0, 2], [1], [3, 4]]},
                },
                path,
            )
            groups = load_channel_groups(
                path,
                torch.device("cpu"),
                distance_threshold=0.5,
                expected_dataset="toy_train",
                expected_layer="layer3",
                expected_feature_shape=(5, 2, 2),
            )

        self.assertEqual(groups.group_count, 3)
        self.assertEqual(groups.channel_count, 5)
        self.assertEqual(groups.group_sizes.tolist(), [2, 1, 2])

    def test_load_channel_groups_cuts_nmf_linkage_at_runtime(self) -> None:
        distances = torch.tensor(
            [
                [0.0, 0.1, 0.9],
                [0.1, 0.0, 0.8],
                [0.9, 0.8, 0.0],
            ]
        )
        linkage_matrix = hierarchical_channel_linkage(distances, method="average")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "layer2.pt"
            torch.save(
                {
                    "method": "nmf_latent_cosine",
                    "dataset": "toy_train",
                    "split": "train",
                    "layer": "layer2",
                    "feature_shape": (3, 2, 2),
                    "linkage": torch.from_numpy(linkage_matrix),
                },
                path,
            )
            groups = load_channel_groups(
                path,
                torch.device("cpu"),
                distance_threshold=0.5,
                expected_dataset="toy_train",
                expected_layer="layer2",
                expected_feature_shape=(3, 2, 2),
            )

        self.assertEqual(groups.group_count, 2)
        self.assertEqual(groups.group_sizes.tolist(), [2, 1])

    def test_spatial_block_residual_config_rejects_shape_mismatches(self) -> None:
        path = Path(
            "configs/students/feature_denoising/feature_masking/cifar_10/"
            "resnet18/residual_layer4_block_1_mean.yaml"
        )
        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        raw["student"]["input_shape"][0] = 512
        with self.assertRaisesRegex(ValueError, "num_classes \\+ 1"):
            parse_config(raw)

        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        raw["strategy"]["feature_denoising"]["spatial_mask_block_sizes"] = [5]
        with self.assertRaisesRegex(ValueError, "must fit"):
            parse_config(raw)

    def test_spatial_token_prediction_config_matches_feature_shape(self) -> None:
        config = load_config(
            Path(
                "configs/students/feature_denoising/spatial_token_prediction/cifar_10/resnet18/"
                "token_layer4_blocks2_targets6.yaml"
            )
        )

        self.assertEqual(config.strategy.name, "feature_denoising")
        self.assertEqual(
            config.strategy.feature_denoising.method, "spatial_token_prediction"
        )
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
                self.assertEqual(
                    config.strategy.feature_denoising.image_mask_block_count, 2
                )

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
                    "layer11": 768,
                    "layer12": 768,
                },
            ),
            ("cifar_100", "vit_base_patch16_224"): (
                100,
                {
                    "layer3": 768,
                    "layer6": 768,
                    "layer10": 768,
                    "layer11": 768,
                    "layer12": 768,
                },
            ),
        }

        for (dataset, architecture), (
            teacher_classes,
            layer_dims,
        ) in config_cases.items():
            for layer, channels in layer_dims.items():
                with self.subTest(
                    dataset=dataset, architecture=architecture, layer=layer
                ):
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
                    self.assertEqual(
                        config.strategy.feature_denoising.rotation_degrees, 20.0
                    )
                    if architecture == "vit_base_patch16_224":
                        self.assertEqual(config.dataset.image_size, 224)
                        self.assertEqual(
                            config.strategy.feature_denoising.embedding_pool, "cls"
                        )

    def test_pixel_masked_multilayer_config_matches_concat_shape(self) -> None:
        config = load_config(
            Path(
                "configs/students/feature_denoising/pixel_masked_multilayer/cifar_10/resnet18/"
                "mlp_layer3_layer4_logits_blocks2_scale_015_020.yaml"
            )
        )

        self.assertEqual(config.strategy.name, "feature_denoising")
        self.assertEqual(
            config.strategy.feature_denoising.method,
            "pixel_masked_multilayer_prediction",
        )
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
        torch.testing.assert_close(
            batch.student_inputs, batch.targets * batch.keep_mask
        )

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

    def test_channel_masking_hides_channels_with_probability_point_two(self) -> None:
        features = torch.ones(1024, 64, 2, 2)

        torch.manual_seed(123)
        keep_mask = sample_channel_keep_mask(features, mask_probability=0.2)

        hidden_fraction = float((1.0 - keep_mask).mean())
        self.assertAlmostEqual(hidden_fraction, 0.2, delta=0.01)
        self.assertTrue(torch.all(keep_mask.flatten(start_dim=1).sum(dim=1) < 64))

    def test_channel_group_masking_hides_one_complete_group_per_sample(self) -> None:
        groups = ChannelGroups(
            membership=torch.tensor(
                [
                    [True, False, True, False, False],
                    [False, True, False, False, False],
                    [False, False, False, True, True],
                ]
            ),
            distance_threshold=0.5,
            source_path="toy.pt",
        )
        features = torch.ones(128, 5, 2, 2)

        torch.manual_seed(123)
        keep_mask, sampled_groups = sample_channel_group_keep_mask(features, groups)

        hidden = (1.0 - keep_mask).squeeze(-1).squeeze(-1).bool()
        torch.testing.assert_close(hidden, groups.membership[sampled_groups])
        self.assertEqual(tuple(sampled_groups.shape), (128,))
        self.assertGreater(torch.unique(sampled_groups).numel(), 1)

    def test_channel_group_masking_can_fix_one_complete_group(self) -> None:
        groups = ChannelGroups(
            membership=torch.tensor(
                [
                    [True, False, True, False, False],
                    [False, True, False, False, False],
                    [False, False, False, True, True],
                ]
            ),
            distance_threshold=0.5,
            source_path="toy.pt",
        )
        features = torch.ones(16, 5, 2, 2)

        first, first_indices = sample_channel_group_keep_mask(
            features, groups, group_index=2
        )
        torch.manual_seed(999)
        second, second_indices = sample_channel_group_keep_mask(
            features, groups, group_index=2
        )

        torch.testing.assert_close(first, second)
        torch.testing.assert_close(first_indices, torch.full((16,), 2))
        torch.testing.assert_close(second_indices, torch.full((16,), 2))
        torch.testing.assert_close(
            (1.0 - first[:, :, 0, 0]).bool(),
            groups.membership[2].expand(16, -1),
        )

    def test_channel_group_stratified_masking_samples_within_each_large_group(
        self,
    ) -> None:
        group_sizes = (3, 4, 5, 8)
        membership = torch.zeros((len(group_sizes), sum(group_sizes)), dtype=torch.bool)
        start = 0
        for group_index, group_size in enumerate(group_sizes):
            membership[group_index, start : start + group_size] = True
            start += group_size
        groups = ChannelGroups(
            membership=membership,
            distance_threshold=0.5,
            source_path="toy.pt",
        )
        features = torch.ones(64, sum(group_sizes), 2, 2)

        torch.manual_seed(123)
        keep_mask = sample_channel_group_stratified_keep_mask(
            features,
            groups,
            min_group_size=4,
            mask_fraction=0.25,
        )

        hidden = (1.0 - keep_mask[:, :, 0, 0]).bool()
        expected_counts = (0, 1, 1, 2)
        for group_index, expected_count in enumerate(expected_counts):
            counts = hidden[:, membership[group_index]].sum(dim=1)
            torch.testing.assert_close(
                counts,
                torch.full_like(counts, expected_count),
            )

    def test_channel_group_stratified_masking_can_target_one_group(self) -> None:
        membership = torch.tensor(
            [
                [True, True, True, False, False, False, False],
                [False, False, False, True, True, True, True],
            ]
        )
        groups = ChannelGroups(membership, 0.5, "toy.pt")
        features = torch.ones(32, 7, 2, 2)

        keep_mask = sample_channel_group_stratified_keep_mask(
            features,
            groups,
            min_group_size=4,
            mask_fraction=0.25,
            group_index=1,
        )

        hidden = (1.0 - keep_mask[:, :, 0, 0]).bool()
        torch.testing.assert_close(hidden[:, :3], torch.zeros((32, 3), dtype=torch.bool))
        torch.testing.assert_close(hidden[:, 3:].sum(dim=1), torch.ones(32, dtype=torch.long))

    def test_channel_group_stratified_rounds_mask_count_to_nearest_integer(self) -> None:
        membership = torch.ones((1, 6), dtype=torch.bool)
        groups = ChannelGroups(membership, 0.5, "toy.pt")
        features = torch.ones(12, 6, 1, 1)

        keep_mask = sample_channel_group_stratified_keep_mask(
            features,
            groups,
            min_group_size=4,
            mask_fraction=0.25,
            group_index=0,
        )

        torch.testing.assert_close(
            (1.0 - keep_mask[:, :, 0, 0]).sum(dim=1),
            torch.full((12,), 2.0),
        )

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

    def test_vit_patch_token_configs_use_layer12_patch_grids(self) -> None:
        for dataset, teacher_classes in (("cifar_10", 10), ("cifar_100", 100)):
            path = Path(
                "configs/students/feature_denoising/patch_token_masking/"
                f"{dataset}/vit_base_patch16_224/residual_layer12_mask_p020.yaml"
            )
            with self.subTest(path=str(path)):
                config = load_config(path)

                self.assertEqual(config.dataset.pre_size, 224)
                self.assertEqual(config.dataset.image_size, 224)
                self.assertEqual(config.dataset.batch_size, 64)
                self.assertEqual(config.teacher.num_classes, teacher_classes)
                self.assertEqual(config.student.kind, "feature_residual_denoiser")
                self.assertEqual(config.student.feature_layer, "layer12")
                self.assertEqual(config.student.input_shape, (768, 14, 14))
                self.assertEqual(config.student.hidden_channels, (64,))
                self.assertEqual(
                    config.strategy.feature_denoising.method,
                    "patch_token_masked_residual_reconstruction",
                )
                self.assertEqual(
                    config.strategy.feature_denoising.mask_probability,
                    0.2,
                )

    def test_patch_token_masking_is_bernoulli_and_shared_across_channels(self) -> None:
        config = FeatureDenoisingConfig(
            method="patch_token_masked_residual_reconstruction",
            mask_probability=0.2,
        )
        features = torch.ones(512, 3, 14, 14)

        torch.manual_seed(123)
        batch = sample_feature_denoising_pca_batch(features, config)

        self.assertEqual(tuple(batch.keep_mask.shape), (512, 1, 14, 14))
        torch.testing.assert_close(batch.student_inputs, features * batch.keep_mask)
        torch.testing.assert_close(batch.corrupted_features, batch.student_inputs)
        realized_fraction = float((1.0 - batch.keep_mask).mean())
        self.assertAlmostEqual(realized_fraction, 0.2, delta=0.01)
        hidden_counts = (1.0 - batch.keep_mask).flatten(start_dim=1).sum(dim=1)
        self.assertGreater(len(torch.unique(hidden_counts)), 1)
        self.assertTrue(torch.all(hidden_counts >= 1))

    def test_patch_token_reconstruction_adds_student_residual(self) -> None:
        config = FeatureDenoisingConfig(
            method="patch_token_masked_residual_reconstruction",
            mask_probability=0.2,
        )
        features = torch.ones(2, 4, 3, 3)
        batch = sample_feature_denoising_pca_batch(features, config)
        student = _SameShapeZeroCorrectionStudent()

        predictions = feature_denoising_predictions(student, batch, config)

        torch.testing.assert_close(predictions, batch.corrupted_features)
        self.assertAlmostEqual(
            float(hidden_component_mse(predictions, batch.targets, batch.keep_mask)),
            1.0,
        )

    def test_spatial_block_mask_is_shared_across_channels(self) -> None:
        features = torch.zeros(512, 7, 8, 8)

        torch.manual_seed(123)
        hidden_mask = sample_spatial_block_hidden_mask(features, (1, 3))

        self.assertEqual(tuple(hidden_mask.shape), (512, 1, 8, 8))
        hidden_areas = hidden_mask.flatten(start_dim=1).sum(dim=1)
        self.assertEqual(set(hidden_areas.tolist()), {1.0, 9.0})
        three_by_three_fraction = float((hidden_areas == 9.0).float().mean())
        self.assertAlmostEqual(three_by_three_fraction, 0.5, delta=0.08)
        for mask in hidden_mask[:20, 0]:
            coordinates = mask.nonzero(as_tuple=False)
            block_height = int(coordinates[:, 0].max() - coordinates[:, 0].min() + 1)
            block_width = int(coordinates[:, 1].max() - coordinates[:, 1].min() + 1)
            self.assertEqual(block_height, block_width)
            self.assertEqual(block_height * block_width, int(mask.sum()))

    def test_spatial_block_batch_uses_mean_fill_and_explicit_mask(self) -> None:
        config = load_config(
            Path(
                "configs/students/feature_denoising/feature_masking/cifar_10/"
                "resnet18/residual_layer3_blocks_1_3_mean.yaml"
            )
        ).strategy.feature_denoising
        features = torch.arange(
            2 * 3 * 8 * 8,
            dtype=torch.float32,
        ).reshape(2, 3, 8, 8)
        channel_mean = torch.tensor([10.0, 20.0, 30.0]).reshape(1, 3, 1, 1)
        normalizer = FeatureNormalizer(
            mean=channel_mean,
            std=torch.ones_like(channel_mean),
        )

        torch.manual_seed(123)
        batch = sample_feature_denoising_pca_batch(
            features,
            config,
            feature_normalizer=normalizer,
        )

        hidden_mask = batch.student_inputs[:, -1:]
        corrupted = batch.student_inputs[:, :-1]
        self.assertTrue(torch.all((hidden_mask == 0.0) | (hidden_mask == 1.0)))
        torch.testing.assert_close(batch.keep_mask, 1.0 - hidden_mask)
        torch.testing.assert_close(batch.targets, features)
        torch.testing.assert_close(
            corrupted,
            features * batch.keep_mask + channel_mean * hidden_mask,
        )
        torch.testing.assert_close(batch.corrupted_features, corrupted)

    def test_channel_residual_batch_zero_fills_without_mask_input(self) -> None:
        config = load_config(
            Path(
                "configs/students/feature_denoising/feature_masking/cifar_10/"
                "resnet18/channel_residual_layer1_mask_p020.yaml"
            )
        ).strategy.feature_denoising
        features = torch.arange(
            2 * 64 * 2 * 2,
            dtype=torch.float32,
        ).reshape(2, 64, 2, 2)

        torch.manual_seed(123)
        batch = sample_feature_denoising_pca_batch(features, config)

        self.assertEqual(tuple(batch.student_inputs.shape), tuple(features.shape))
        self.assertEqual(tuple(batch.keep_mask.shape), (2, 64, 1, 1))
        torch.testing.assert_close(batch.targets, features)
        torch.testing.assert_close(
            batch.student_inputs,
            features * batch.keep_mask,
        )
        torch.testing.assert_close(batch.corrupted_features, batch.student_inputs)

    def test_confusion_channel_replacement_adjusts_same_index_channels(
        self,
    ) -> None:
        channel_count = 5
        features = torch.arange(
            2 * channel_count,
            dtype=torch.float32,
        ).reshape(2, channel_count, 1, 1)
        class_ids = torch.tensor([0, 1])
        class_mean = torch.tensor(
            [[1.0] * channel_count, [10.0] * channel_count]
        ).reshape(2, channel_count, 1, 1)
        class_std = torch.tensor(
            [[2.0] * channel_count, [4.0] * channel_count]
        ).reshape(2, channel_count, 1, 1)
        prototypes = torch.stack(
            (
                torch.arange(channel_count, dtype=torch.float32).reshape(
                    1,
                    channel_count,
                    1,
                    1,
                ),
                (20.0 + torch.arange(channel_count, dtype=torch.float32)).reshape(
                    1, channel_count, 1, 1
                ),
            )
        )
        bank = _class_corruption_bank(
            confusing_class=torch.tensor([1, 0]),
            class_mean=class_mean,
            class_std=class_std,
            prototypes=prototypes,
        )
        config = FeatureDenoisingConfig(
            method="confusion_channel_replacement_residual_reconstruction",
            mask_probability=0.2,
        )

        torch.manual_seed(123)
        batch = sample_confusion_channel_replacement_batch(
            features,
            class_ids,
            config,
            bank,
        )

        hidden_channels = (batch.keep_mask == 0.0).flatten(start_dim=1)
        self.assertEqual(hidden_channels.sum(dim=1).tolist(), [1, 1])
        for row in range(features.shape[0]):
            channel = int(hidden_channels[row].nonzero()[0])
            source_class = int(class_ids[row])
            donor_class = int(bank.confusing_class[source_class])
            donor = bank.prototypes[donor_class, 0, channel]
            expected = (
                bank.class_mean[source_class, channel]
                + bank.class_std[source_class, channel]
                * (donor - bank.class_mean[donor_class, channel])
                / bank.class_std[donor_class, channel]
            )
            torch.testing.assert_close(
                batch.student_inputs[row, channel],
                expected,
            )
            visible = hidden_channels[row].logical_not()
            torch.testing.assert_close(
                batch.student_inputs[row, visible],
                features[row, visible],
            )

    def test_confusion_channel_replacement_rounds_twenty_percent(
        self,
    ) -> None:
        channel_count = 64
        bank = _class_corruption_bank(
            confusing_class=torch.tensor([1, 0]),
            class_mean=torch.zeros(2, channel_count, 1, 1),
            class_std=torch.ones(2, channel_count, 1, 1),
            prototypes=torch.zeros(2, 1, channel_count, 2, 2),
        )

        batch = sample_confusion_channel_replacement_batch(
            features=torch.ones(3, channel_count, 2, 2),
            class_ids=torch.tensor([0, 1, 0]),
            config=FeatureDenoisingConfig(
                method=("confusion_channel_replacement_residual_reconstruction"),
                mask_probability=0.2,
            ),
            bank=bank,
        )

        hidden_count = (batch.keep_mask == 0.0).flatten(start_dim=1).sum(dim=1)
        self.assertEqual(hidden_count.tolist(), [13, 13, 13])

    def test_class_channel_corruption_fit_and_artifact_round_trip(self) -> None:
        validation_features = torch.stack(
            [torch.full((2, 2, 2), value) for value in (1.0, 1.0, 2.0, 2.0, 2.0, 2.0)]
        )
        validation_labels = torch.tensor([0, 0, 1, 1, 2, 2])
        confusion_features = torch.stack(
            [torch.full((2, 2, 2), value) for value in (2.0, 2.0, 0.0, 0.0, 2.0, 2.0)]
        )
        confusion_labels = torch.tensor([0, 0, 1, 1, 2, 2])
        prototype_features = torch.stack(
            [
                torch.full((2, 2, 2), float(class_id * 10 + example))
                for class_id in range(3)
                for example in range(2)
            ]
        )
        prototype_labels = torch.tensor([0, 0, 1, 1, 2, 2])
        validation_loader = DataLoader(
            TensorDataset(validation_features, validation_labels),
            batch_size=3,
        )
        confusion_loader = DataLoader(
            TensorDataset(confusion_features, confusion_labels),
            batch_size=3,
        )
        prototype_loader = DataLoader(
            TensorDataset(prototype_features, prototype_labels),
            batch_size=2,
        )
        forwarder = _EncodedPredictionForwarder(num_classes=3)

        bank = fit_class_channel_corruption_bank(
            validation_loader=validation_loader,
            confusion_loader=confusion_loader,
            prototype_loader=prototype_loader,
            perturbation_forwarder=forwarder,
            device=torch.device("cpu"),
            num_classes=3,
            prototype_count=2,
        )

        torch.testing.assert_close(
            bank.confusion_matrix,
            torch.tensor(
                [
                    [0, 0, 2],
                    [2, 0, 0],
                    [0, 0, 2],
                ]
            ),
        )
        self.assertEqual(bank.confusing_class.tolist(), [2, 0, 0])
        torch.testing.assert_close(
            bank.class_mean[:, 0, 0, 0],
            torch.tensor([1.0, 2.0, 2.0]),
        )
        torch.testing.assert_close(
            bank.class_std,
            torch.zeros_like(bank.class_std),
        )
        self.assertEqual(tuple(bank.prototypes.shape), (3, 2, 2, 2, 2))
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "class_channel_corruption.pt"
            save_class_channel_corruption_bank(
                path,
                bank,
                {
                    "artifact_version": 1,
                    "dataset": "toy_validation",
                },
            )
            loaded = load_class_channel_corruption_bank(
                path,
                expected_prototype_count=2,
            )
        torch.testing.assert_close(loaded.prototypes, bank.prototypes)
        torch.testing.assert_close(
            loaded.confusing_class,
            bank.confusing_class,
        )

    def test_feature_normalizer_fits_and_round_trips_training_features(self) -> None:
        features = torch.tensor(
            [
                [[[1.0, 3.0]], [[2.0, 4.0]]],
                [[[5.0, 7.0]], [[6.0, 8.0]]],
            ]
        )
        labels = torch.zeros(2, dtype=torch.long)
        loader = DataLoader(TensorDataset(features, labels), batch_size=1)
        forwarder = _IdentityFeatureForwarder()

        normalizer = fit_feature_normalizer_from_loader(
            loader,
            forwarder,
            torch.device("cpu"),
        )

        torch.testing.assert_close(
            normalizer.mean,
            torch.tensor([4.0, 5.0]).reshape(1, 2, 1, 1),
        )
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "feature_normalizer.pt"
            save_feature_normalizer(
                path,
                normalizer,
                {"dataset": "toy_train", "split": "train"},
            )
            loaded = load_feature_normalizer(path)
        torch.testing.assert_close(loaded.mean, normalizer.mean)
        torch.testing.assert_close(loaded.std, normalizer.std)

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
        self.assertEqual(
            tuple(batch.student_inputs["visible_tokens"].shape), (2, 16, 3)
        )
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

    def test_pixel_masked_embedding_batch_uses_clean_and_masked_embeddings(
        self,
    ) -> None:
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

    def test_pixel_augmented_embedding_batch_uses_clean_and_augmented_embeddings(
        self,
    ) -> None:
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

    def test_pixel_masked_multilayer_batch_concatenates_context_and_targets(
        self,
    ) -> None:
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

    def test_pixel_masked_multilayer_l234_batch_concatenates_context_and_targets(
        self,
    ) -> None:
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

    def test_feature_residual_denoiser_architecture_and_correction(self) -> None:
        config = load_config(
            Path(
                "configs/students/feature_denoising/feature_masking/cifar_10/"
                "resnet18/residual_layer1_blocks_3_5_mean.yaml"
            )
        )
        model = build_student(config.student)
        layers = list(model.denoiser)

        self.assertEqual(
            [
                (layers[index].in_channels, layers[index].out_channels)
                for index in (0, 2, 4)
            ],
            [(65, 16), (16, 16), (16, 64)],
        )
        self.assertIsInstance(layers[1], nn.ReLU)
        self.assertIsInstance(layers[3], nn.ReLU)
        self.assertEqual(layers[0].kernel_size, (3, 3))
        self.assertEqual(layers[0].padding, (1, 1))

        inputs = torch.zeros(1, 65, 32, 32, requires_grad=True)
        correction = model(inputs)
        self.assertEqual(tuple(correction.shape), (1, 64, 32, 32))
        correction.mean().backward()
        self.assertIsNotNone(inputs.grad)

    def test_feature_residual_denoiser_respects_configured_hidden_width(self) -> None:
        config = load_config(
            Path(
                "configs/students/feature_denoising/feature_masking/cifar_10/"
                "resnet50/residual_layer4_block_1_mean.yaml"
            )
        )
        model = build_student(config.student)
        layers = list(model.denoiser)

        self.assertEqual(
            [
                (layers[index].in_channels, layers[index].out_channels)
                for index in (0, 2, 4)
            ],
            [(2049, 64), (64, 64), (64, 2048)],
        )
        inputs = torch.zeros(1, 2049, 4, 4, requires_grad=True)
        correction = model(inputs)
        self.assertEqual(tuple(correction.shape), (1, 2048, 4, 4))
        correction.mean().backward()
        self.assertIsNotNone(inputs.grad)

    def test_channel_residual_denoiser_uses_no_mask_input_channel(self) -> None:
        config = load_config(
            Path(
                "configs/students/feature_denoising/feature_masking/cifar_10/"
                "resnet50/channel_residual_layer4_mask_p020.yaml"
            )
        )
        model = build_student(config.student)
        layers = list(model.denoiser)

        self.assertEqual(
            [
                (layers[index].in_channels, layers[index].out_channels)
                for index in (0, 2, 4)
            ],
            [(2048, 64), (64, 64), (64, 2048)],
        )
        inputs = torch.zeros(1, 2048, 4, 4, requires_grad=True)
        correction = model(inputs)
        self.assertEqual(tuple(correction.shape), (1, 2048, 4, 4))
        correction.mean().backward()
        self.assertIsNotNone(inputs.grad)

    def test_residual_prediction_adds_correction_before_masked_loss(self) -> None:
        config = load_config(
            Path(
                "configs/students/feature_denoising/feature_masking/cifar_10/"
                "resnet18/residual_layer4_block_1_mean.yaml"
            )
        ).strategy.feature_denoising
        corrupted = torch.zeros(1, 2, 2, 2)
        hidden_mask = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
        batch = sample_feature_denoising_pca_batch(
            torch.ones(1, 2, 2, 2),
            config,
            feature_normalizer=FeatureNormalizer(
                mean=torch.zeros(1, 2, 1, 1),
                std=torch.ones(1, 2, 1, 1),
            ),
        )
        batch.corrupted_features = corrupted
        batch.student_inputs = torch.cat((corrupted, hidden_mask), dim=1)
        batch.keep_mask = 1.0 - hidden_mask
        student = _ConstantCorrectionStudent(0.75)

        reconstructed = feature_denoising_predictions(student, batch, config)
        loss = hidden_component_mse(
            reconstructed,
            batch.targets,
            batch.keep_mask,
        )

        torch.testing.assert_close(reconstructed, torch.full_like(reconstructed, 0.75))
        self.assertAlmostEqual(float(loss), 0.0625)

    def test_confusion_channel_direct_prediction_does_not_add_corrupted_input(
        self,
    ) -> None:
        config = FeatureDenoisingConfig(
            method="confusion_channel_replacement_reconstruction",
            mask_probability=0.5,
        )
        features = torch.ones(1, 2, 2, 2)
        bank = _class_corruption_bank(
            confusing_class=torch.tensor([1, 0]),
            class_mean=torch.zeros(2, 2, 1, 1),
            class_std=torch.ones(2, 2, 1, 1),
            prototypes=torch.full((2, 1, 2, 2, 2), 7.0),
        )
        batch = sample_confusion_channel_replacement_batch(
            features=features,
            class_ids=torch.tensor([0]),
            config=config,
            bank=bank,
        )

        predictions = feature_denoising_predictions(
            _SameShapeZeroCorrectionStudent(),
            batch,
            config,
        )
        loss = hidden_component_mse(
            predictions,
            batch.targets,
            batch.keep_mask,
        )

        torch.testing.assert_close(predictions, torch.zeros_like(features))
        self.assertAlmostEqual(float(loss), 1.0)
        self.assertEqual(
            feature_denoising_score_name(
                "confusion_channel_replacement_reconstruction"
            ),
            ("feature_denoising_confusion_channel_replacement_reconstruction_error"),
        )

    def test_residual_score_export_includes_masked_identity_improvement(self) -> None:
        config = load_config(
            Path(
                "configs/students/feature_denoising/feature_masking/cifar_10/"
                "resnet18/residual_layer4_block_1_mean.yaml"
            )
        ).strategy.feature_denoising
        features = torch.ones(2, 512, 4, 4)
        labels = torch.tensor([0, 1])
        loader = DataLoader(TensorDataset(features, labels), batch_size=2)
        normalizer = FeatureNormalizer(
            mean=torch.zeros(1, 512, 1, 1),
            std=torch.ones(1, 512, 1, 1),
        )

        scores = collect_feature_denoising_reconstruction_scores(
            student=_ZeroCorrectionStudent(),
            loader=loader,
            device=torch.device("cpu"),
            perturbation_forwarder=_IdentityFeatureForwarder(),
            feature_denoising_config=config,
            feature_normalizer=normalizer,
        )

        torch.testing.assert_close(
            scores["raw_reconstruction_error"],
            scores["identity_error"],
        )
        torch.testing.assert_close(
            scores["improvement"],
            torch.zeros_like(scores["improvement"]),
        )
        self.assertNotIn("cosine_similarity", scores)
        self.assertNotIn("z_pred", scores)
        self.assertEqual(
            feature_denoising_score_name("spatial_block_residual_reconstruction"),
            "feature_denoising_spatial_block_residual_reconstruction_error",
        )
        self.assertEqual(
            feature_denoising_score_name(
                "patch_token_masked_residual_reconstruction"
            ),
            "feature_denoising_patch_token_residual_reconstruction_error",
        )

    def test_channel_residual_score_export_includes_identity_improvement(self) -> None:
        config = load_config(
            Path(
                "configs/students/feature_denoising/feature_masking/cifar_10/"
                "resnet18/channel_residual_layer4_mask_p020.yaml"
            )
        ).strategy.feature_denoising
        features = torch.ones(2, 512, 4, 4)
        labels = torch.tensor([0, 1])
        loader = DataLoader(TensorDataset(features, labels), batch_size=2)

        scores = collect_feature_denoising_reconstruction_scores(
            student=_SameShapeZeroCorrectionStudent(),
            loader=loader,
            device=torch.device("cpu"),
            perturbation_forwarder=_IdentityFeatureForwarder(),
            feature_denoising_config=config,
        )

        torch.testing.assert_close(
            scores["raw_reconstruction_error"],
            scores["identity_error"],
        )
        torch.testing.assert_close(
            scores["improvement"],
            torch.zeros_like(scores["improvement"]),
        )
        self.assertNotIn("cosine_similarity", scores)
        self.assertEqual(
            feature_denoising_score_name("channel_masked_residual_reconstruction"),
            "feature_denoising_channel_residual_reconstruction_error",
        )

    def test_channel_group_score_export_resamples_group_for_ten_draws(self) -> None:
        config = FeatureDenoisingConfig(
            method="channel_group_masked_residual_reconstruction",
            channel_group_path="toy.pt",
            channel_group_distance_threshold=0.5,
            evaluation_draws=10,
        )
        groups = ChannelGroups(
            membership=torch.eye(4, dtype=torch.bool),
            distance_threshold=0.5,
            source_path="toy.pt",
        )
        features = torch.ones(3, 4, 2, 2)
        labels = torch.tensor([0, 1, 2])
        loader = DataLoader(TensorDataset(features, labels), batch_size=3)

        torch.manual_seed(123)
        scores = collect_feature_denoising_reconstruction_scores(
            student=_SameShapeZeroCorrectionStudent(),
            loader=loader,
            device=torch.device("cpu"),
            perturbation_forwarder=_IdentityFeatureForwarder(),
            feature_denoising_config=config,
            channel_groups=groups,
        )

        self.assertEqual(tuple(scores["sampled_channel_group_indices"].shape), (3, 10))
        self.assertTrue(
            torch.all(
                (scores["sampled_channel_group_indices"] >= 0)
                & (scores["sampled_channel_group_indices"] < 4)
            )
        )
        torch.testing.assert_close(
            scores["raw_reconstruction_error"],
            scores["identity_error"],
        )
        self.assertEqual(
            feature_denoising_score_name(
                "channel_group_masked_residual_reconstruction"
            ),
            "feature_denoising_channel_group_residual_reconstruction_error",
        )

    def test_confusion_channel_score_export_uses_teacher_predictions(
        self,
    ) -> None:
        features = torch.zeros(2, 1, 1, 1)
        real_labels = torch.zeros(2, dtype=torch.long)
        loader = DataLoader(
            TensorDataset(features, real_labels),
            batch_size=2,
        )
        bank = _class_corruption_bank(
            confusing_class=torch.tensor([1, 0]),
            class_mean=torch.zeros(2, 1, 1, 1),
            class_std=torch.ones(2, 1, 1, 1),
            prototypes=torch.tensor([3.0, 7.0]).reshape(2, 1, 1, 1, 1),
        )

        scores = collect_feature_denoising_reconstruction_scores(
            student=_SameShapeZeroCorrectionStudent(),
            loader=loader,
            device=torch.device("cpu"),
            perturbation_forwarder=_FixedPredictionForwarder(
                predicted_class=1,
                num_classes=2,
            ),
            feature_denoising_config=FeatureDenoisingConfig(
                method=("confusion_channel_replacement_residual_reconstruction"),
                mask_probability=0.2,
                evaluation_draws=1,
            ),
            class_channel_corruption=bank,
        )

        torch.testing.assert_close(
            scores["raw_reconstruction_error"],
            torch.full((2,), 9.0),
        )
        torch.testing.assert_close(
            scores["raw_reconstruction_error"],
            scores["identity_error"],
        )
        self.assertEqual(
            feature_denoising_score_name(
                "confusion_channel_replacement_residual_reconstruction"
            ),
            (
                "feature_denoising_confusion_channel_replacement_residual_"
                "reconstruction_error"
            ),
        )

    def test_residual_denoiser_training_smoke(self) -> None:
        features = torch.randn(6, 3, 4, 4)
        labels = torch.zeros(6, dtype=torch.long)
        train_loader = DataLoader(
            TensorDataset(features[:4], labels[:4]),
            batch_size=2,
        )
        validation_loader = DataLoader(
            TensorDataset(features[4:], labels[4:]),
            batch_size=2,
        )
        student = build_student(
            StudentConfig(
                kind="feature_residual_denoiser",
                input_shape=(4, 4, 4),
                num_classes=3,
            )
        )
        normalizer = FeatureNormalizer(
            mean=features[:4].mean(dim=(0, 2, 3), keepdim=True),
            std=features[:4].std(dim=(0, 2, 3), keepdim=True),
        )

        with TemporaryDirectory() as temporary_directory:
            summary = train_feature_denoising_student(
                student=student,
                train_loader=train_loader,
                validation_loader=validation_loader,
                device=torch.device("cpu"),
                optimizer_config=OptimizerConfig(
                    name="adamw",
                    learning_rate=1.0e-3,
                ),
                training_config=ResolvedTrainingMethodConfig(
                    epochs=1,
                    seed=42,
                    device="cpu",
                    log_every_steps=10,
                ),
                output_dir=Path(temporary_directory),
                perturbation_forwarder=_IdentityFeatureForwarder(),
                feature_denoising_config=FeatureDenoisingConfig(
                    method="spatial_block_residual_reconstruction",
                    spatial_mask_block_sizes=(1, 3),
                ),
                feature_normalizer=normalizer,
                mlflow_enabled=False,
            )

            self.assertTrue(Path(str(summary["best_checkpoint_path"])).exists())

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


class _IdentityFeatureForwarder(nn.Module):
    def forward_to_features(self, images: torch.Tensor) -> torch.Tensor:
        return images


class _EncodedPredictionForwarder(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        predictions = images[:, 0, 0, 0].long().remainder(self.num_classes)
        logits = torch.full(
            (images.shape[0], self.num_classes),
            -10.0,
            device=images.device,
        )
        logits.scatter_(1, predictions[:, None], 10.0)
        return logits, images

    def forward_to_features(self, images: torch.Tensor) -> torch.Tensor:
        return images


class _FixedPredictionForwarder(nn.Module):
    def __init__(self, predicted_class: int, num_classes: int) -> None:
        super().__init__()
        self.predicted_class = predicted_class
        self.num_classes = num_classes

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.full(
            (images.shape[0], self.num_classes),
            -10.0,
            device=images.device,
        )
        logits[:, self.predicted_class] = 10.0
        return logits, images

    def forward_to_features(self, images: torch.Tensor) -> torch.Tensor:
        return images


def _class_corruption_bank(
    confusing_class: torch.Tensor,
    class_mean: torch.Tensor,
    class_std: torch.Tensor,
    prototypes: torch.Tensor,
) -> ClassChannelCorruptionBank:
    num_classes, prototype_count = prototypes.shape[:2]
    return ClassChannelCorruptionBank(
        confusion_matrix=torch.zeros(
            num_classes,
            num_classes,
            dtype=torch.long,
        ),
        confusing_class=confusing_class.long(),
        class_mean=class_mean.float(),
        class_std=class_std.float(),
        prototypes=prototypes.float(),
        prototype_source_indices=torch.arange(
            num_classes * prototype_count,
        ).reshape(num_classes, prototype_count),
        validation_class_counts=torch.ones(num_classes, dtype=torch.long),
        mean_teacher_probabilities=torch.eye(num_classes),
    )


class _ConstantCorrectionStudent(nn.Module):
    def __init__(self, correction: float) -> None:
        super().__init__()
        self.correction = correction

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.full_like(inputs[:, :-1], self.correction)


class _ZeroCorrectionStudent(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(inputs[:, :-1])


class _SameShapeZeroCorrectionStudent(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(inputs)


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
