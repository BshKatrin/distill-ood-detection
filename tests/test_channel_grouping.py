"""Tests for correlation-based teacher channel grouping."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from distill_ood_detection.config import parse_channel_grouping_config
from distill_ood_detection.datasets.inference import NamedLoader
from distill_ood_detection.evaluation.channel_grouping import (
    correlation_distance,
    flat_channel_groups,
    hierarchical_channel_linkage,
    latent_channel_cosine_distance,
    pearson_channel_correlation,
    standardize_channel_profiles,
    top_activation_profiles,
)
from distill_ood_detection.experiments.build_channel_groups import (
    build_layer_hierarchy,
    build_nmf_latent_layer_hierarchy,
    collect_channel_profiles,
    fit_streaming_nmf_models,
)


class ToyTeacher(nn.Module):
    """Teacher with two hookable feature layers."""

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Identity()
        self.layer2 = nn.Conv2d(2, 3, kernel_size=1, bias=False)
        with torch.no_grad():
            self.layer2.weight.copy_(
                torch.tensor(
                    [
                        [[[1.0]], [[0.0]]],
                        [[[0.0]], [[1.0]]],
                        [[[1.0]], [[1.0]]],
                    ]
                )
            )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return pooled toy outputs."""

        return self.layer2(self.layer1(images)).mean(dim=(2, 3))


class ChannelGroupingTests(unittest.TestCase):
    """Validate profiles, correlation distance, clustering, and artifacts."""

    def test_parses_channel_grouping_config(self) -> None:
        config = parse_channel_grouping_config(
            {
                "experiment_name": "groups",
                "run_dir": "runs/tests/groups",
                "layers": ["layer1", "layer4"],
                "profile_top_fraction": 0.1,
                "linkage_method": "average",
                "cut_thresholds": [0.2, 0.5],
            }
        )

        self.assertEqual(config.layers, ("layer1", "layer4"))
        self.assertEqual(config.cut_thresholds, (0.2, 0.5))

    def test_parses_nmf_latent_channel_grouping_config(self) -> None:
        config = parse_channel_grouping_config(
            {
                "experiment_name": "nmf_groups",
                "run_dir": "runs/tests/nmf_groups",
                "method": "nmf_latent_cosine",
                "layers": ["layer1", "layer4"],
                "nmf_components": 32,
                "nmf_batch_size": 65536,
                "nmf_epochs": 2,
                "linkage_method": "average",
                "cut_thresholds": [],
            }
        )

        self.assertEqual(config.method, "nmf_latent_cosine")
        self.assertEqual(config.nmf_components, 32)
        self.assertEqual(config.cut_thresholds, ())

    def test_profile_uses_ceiled_top_spatial_count(self) -> None:
        features = torch.arange(40, dtype=torch.float32).reshape(1, 2, 4, 5)

        profiles = top_activation_profiles(features, top_fraction=0.11)

        # ceil(0.11 * 20) = 3 values per channel.
        torch.testing.assert_close(profiles, torch.tensor([[18.0, 38.0]]))

    def test_standardized_profiles_produce_pearson_distance(self) -> None:
        profiles = torch.tensor(
            [
                [1.0, 2.0, 5.0],
                [2.0, 4.0, 5.0],
                [3.0, 6.0, 5.0],
            ]
        )

        standardized, means, standard_deviations, active = (
            standardize_channel_profiles(profiles, epsilon=1.0e-6)
        )
        correlations = pearson_channel_correlation(standardized, active)
        distances = correlation_distance(correlations)

        torch.testing.assert_close(means, torch.tensor([2.0, 4.0, 5.0]))
        self.assertEqual(active.tolist(), [True, True, False])
        self.assertEqual(standard_deviations[-1].item(), 0.0)
        torch.testing.assert_close(correlations[0, 1], torch.tensor(1.0))
        torch.testing.assert_close(correlations[0, 2], torch.tensor(0.0))
        torch.testing.assert_close(distances[0, 1], torch.tensor(0.0))
        torch.testing.assert_close(distances[0, 2], torch.tensor(1.0))

        linkage_matrix = hierarchical_channel_linkage(distances, method="average")
        self.assertEqual(flat_channel_groups(linkage_matrix, threshold=0.1), [[0, 1], [2]])

    def test_nmf_p_columns_produce_cosine_distance(self) -> None:
        p_matrix = torch.tensor(
            [
                [1.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )

        distances, active = latent_channel_cosine_distance(p_matrix)

        self.assertEqual(active.tolist(), [True, True, True, False])
        torch.testing.assert_close(distances[0, 1], torch.tensor(0.0))
        torch.testing.assert_close(distances[0, 2], torch.tensor(1.0))
        torch.testing.assert_close(distances[0, 3], torch.tensor(1.0))
        torch.testing.assert_close(distances.diag(), torch.zeros(4))

    def test_streams_only_reduced_profiles(self) -> None:
        images = torch.arange(64, dtype=torch.float32).reshape(2, 2, 4, 4)
        labels = torch.tensor([0, 1])
        named_loader = NamedLoader(
            name="toy_train",
            split="train",
            loader=DataLoader(TensorDataset(images, labels), batch_size=1),
        )

        profiles, feature_shapes = collect_channel_profiles(
            teacher=ToyTeacher(),
            layers=("layer1", "layer2"),
            named_loader=named_loader,
            device=torch.device("cpu"),
            top_fraction=0.25,
        )

        self.assertEqual(tuple(profiles["layer1"].shape), (2, 2))
        self.assertEqual(tuple(profiles["layer2"].shape), (2, 3))
        self.assertEqual(feature_shapes["layer1"], (2, 4, 4))

    def test_writes_compact_hierarchy_and_dendrograms(self) -> None:
        profiles = torch.tensor(
            [
                [1.0, 2.0, 4.0],
                [2.0, 4.0, 3.0],
                [3.0, 6.0, 2.0],
                [4.0, 8.0, 1.0],
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            summary = build_layer_hierarchy(
                profiles=profiles,
                layer="layer1",
                output_dir=output_dir,
                device=torch.device("cpu"),
                top_fraction=0.1,
                epsilon=1.0e-6,
                linkage_method="average",
                cut_thresholds=(0.1, 0.5),
                dataset_name="toy_train",
                dataset_split="train",
                feature_shape=(3, 2, 2),
            )
            artifact = torch.load(
                output_dir / "layer1.pt",
                map_location="cpu",
                weights_only=False,
            )

            self.assertTrue((output_dir / "layer1_groups.json").is_file())
            self.assertTrue((output_dir / "layer1_dendrogram_overview.png").is_file())
            self.assertTrue((output_dir / "layer1_dendrogram_labeled.pdf").is_file())
            self.assertNotIn("profiles", artifact)
            self.assertNotIn("standardized_profiles", artifact)
            self.assertEqual(tuple(artifact["correlation"].shape), (3, 3))
            self.assertEqual(summary["group_counts"]["0.1"], 2)

    def test_streams_nmf_and_saves_p_matrix_with_only_dendrograms(self) -> None:
        images = torch.arange(128, dtype=torch.float32).reshape(4, 2, 4, 4)
        labels = torch.tensor([0, 1, 0, 1])
        named_loader = NamedLoader(
            name="toy_train",
            split="train",
            loader=DataLoader(TensorDataset(images, labels), batch_size=2),
        )
        models, feature_shapes = fit_streaming_nmf_models(
            teacher=ToyTeacher(),
            layers=("layer1", "layer2"),
            named_loader=named_loader,
            device=torch.device("cpu"),
            n_components=2,
            update_batch_size=16,
            epochs=1,
            random_state=42,
        )

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            summary = build_nmf_latent_layer_hierarchy(
                model=models["layer2"],
                layer="layer2",
                output_dir=output_dir,
                linkage_method="average",
                dataset_name="toy_train",
                dataset_split="train",
                sample_count=4,
                feature_shape=feature_shapes["layer2"],
                nmf_epochs=1,
                nmf_batch_size=16,
            )
            artifact = torch.load(
                output_dir / "layer2.pt",
                map_location="cpu",
                weights_only=False,
            )

            self.assertEqual(tuple(artifact["p_matrix"].shape), (2, 3))
            self.assertTrue(torch.all(artifact["p_matrix"] >= 0.0))
            self.assertTrue((output_dir / "layer2_dendrogram_overview.png").is_file())
            self.assertTrue((output_dir / "layer2_dendrogram_labeled.pdf").is_file())
            self.assertFalse((output_dir / "layer2_groups.json").exists())
            self.assertEqual(summary["concept_count"], 2)


if __name__ == "__main__":
    unittest.main()
