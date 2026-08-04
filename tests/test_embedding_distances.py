"""Tests for pooled embedding extraction and blocked ID distances."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from distill_ood_detection.config import parse_embedding_distance_config
from distill_ood_detection.datasets.inference import NamedLoader
from distill_ood_detection.experiments.export_embedding_distances import (
    KNN_SCORE_SIGNS,
    blocked_topk_euclidean,
    compute_distance_quantities,
    export_loader_embeddings,
    knn_score_vectors,
    load_embedding_shards,
    load_output_shards,
)


class ToyTeacher(nn.Module):
    """Small convolutional teacher exposing two residual-style stages."""

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Conv2d(3, 4, kernel_size=1, bias=False)
        self.layer2 = nn.Conv2d(4, 5, kernel_size=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(5, 2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return logits for one image batch."""

        features = self.layer2(self.layer1(images))
        return self.classifier(torch.flatten(self.pool(features), start_dim=1))


class EmbeddingDistanceTests(unittest.TestCase):
    """Validate pooled artifacts and exact blocked distances."""

    def test_parses_embedding_distance_config(self) -> None:
        config = parse_embedding_distance_config(
            {
                "experiment_name": "embedding_distances",
                "run_dir": "runs/embedding_distances",
                "layers": ["layer1", "layer2", "layer3", "layer4"],
                "k_neighbors": 5,
                "dataset": {"name": "cifar10"},
            }
        )

        self.assertEqual(config.layers, ("layer1", "layer2", "layer3", "layer4"))
        self.assertEqual(config.k_neighbors, 5)

    def test_rejects_unsupported_layer(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported embedding-distance"):
            parse_embedding_distance_config(
                {
                    "experiment_name": "embedding_distances",
                    "run_dir": "runs/embedding_distances",
                    "layers": ["fc"],
                }
            )

    def test_exports_gap_embeddings_in_shards(self) -> None:
        torch.manual_seed(3)
        teacher = ToyTeacher()
        images = torch.randn(5, 3, 4, 4)
        labels = torch.tensor([0, 1, 0, 1, 0])
        loader = DataLoader(TensorDataset(images, labels), batch_size=2)
        named_loader = NamedLoader(name="toy_test", split="test", loader=loader)

        with tempfile.TemporaryDirectory() as directory:
            artifacts = export_loader_embeddings(
                teacher=teacher,
                layers=("layer1", "layer2"),
                named_loader=named_loader,
                output_dir=Path(directory),
                device=torch.device("cpu"),
                shard_size=3,
                metadata={"pooling": "gap"},
            )
            embeddings, saved_labels = load_embedding_shards(
                artifacts,
                dataset="toy_test",
                layer="layer1",
            )
            logits, probabilities = load_output_shards(
                artifacts,
                dataset="toy_test",
                layer="layer1",
            )

        expected = teacher.layer1(images).mean(dim=(-2, -1))
        expected_logits = teacher(images)
        torch.testing.assert_close(embeddings, expected)
        torch.testing.assert_close(saved_labels, labels)
        torch.testing.assert_close(logits, expected_logits)
        torch.testing.assert_close(probabilities, torch.softmax(expected_logits, dim=1))
        self.assertEqual(len(artifacts), 4)

    def test_knn_scores_use_mean_neighbor_outputs(self) -> None:
        query_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        query_probabilities = torch.softmax(query_logits, dim=1)
        neighbor_mean_logits = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        neighbor_probabilities = torch.tensor(
            [
                [[0.8, 0.2], [0.4, 0.6]],
                [[0.9, 0.1], [0.1, 0.9]],
            ]
        )
        neighbor_mean_probabilities = neighbor_probabilities.mean(dim=1)

        raw_metrics, ood_scores = knn_score_vectors(
            query_logits=query_logits,
            query_probabilities=query_probabilities,
            neighbor_mean_logits=neighbor_mean_logits,
            neighbor_mean_probabilities=neighbor_mean_probabilities,
            neighbor_probabilities=neighbor_probabilities,
        )

        self.assertEqual(set(raw_metrics), set(KNN_SCORE_SIGNS))
        torch.testing.assert_close(
            raw_metrics["student_msp"],
            torch.tensor([0.6, 0.5], dtype=torch.float64),
        )
        self.assertTrue(
            torch.all(torch.isfinite(raw_metrics["student_energy"])).item()
        )
        expected_predictive_entropy = torch.special.entr(
            neighbor_probabilities.double().mean(dim=1)
        ).sum(dim=1)
        expected_bald = expected_predictive_entropy - torch.special.entr(
            neighbor_probabilities.double()
        ).sum(dim=2).mean(dim=1)
        torch.testing.assert_close(
            raw_metrics["ensemble_predictive_entropy"],
            expected_predictive_entropy,
        )
        torch.testing.assert_close(raw_metrics["ensemble_bald"], expected_bald)
        for name, raw_values in raw_metrics.items():
            torch.testing.assert_close(
                ood_scores[name],
                raw_values * KNN_SCORE_SIGNS[name],
            )

    def test_distance_selection_controls_neighbor_output_means(self) -> None:
        queries = torch.tensor([[0.1], [9.0]])
        references = torch.tensor([[0.0], [1.0], [10.0]])
        reference_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [4.0, 0.0]])
        reference_probabilities = torch.softmax(reference_logits, dim=1)
        query_logits = torch.tensor([[1.0, 0.0], [3.0, 0.0]])

        result = compute_distance_quantities(
            queries=queries,
            references=references,
            reference_labels=torch.tensor([0, 1, 0]),
            query_logits=query_logits,
            query_probabilities=torch.softmax(query_logits, dim=1),
            reference_logits=reference_logits,
            reference_probabilities=reference_probabilities,
            centroid=references.mean(dim=0),
            class_centroids=torch.tensor([[5.0], [1.0]]),
            classes=torch.tensor([0, 1]),
            device=torch.device("cpu"),
            k_neighbors=2,
            query_block_size=1,
            reference_block_size=2,
        )

        torch.testing.assert_close(
            result["neighbor_indices"],
            torch.tensor([[0, 1], [2, 1]]),
        )
        torch.testing.assert_close(
            result["neighbor_mean_logits"],
            torch.tensor([[1.0, 1.0], [2.0, 1.0]]),
        )
        torch.testing.assert_close(
            result["neighbor_mean_probabilities"],
            torch.stack(
                (
                    reference_probabilities[[0, 1]].mean(dim=0),
                    reference_probabilities[[2, 1]].mean(dim=0),
                )
            ),
        )

    def test_blocked_topk_matches_full_cdist(self) -> None:
        torch.manual_seed(7)
        queries = torch.randn(7, 6)
        references = torch.randn(11, 6)

        distances, indices = blocked_topk_euclidean(
            queries=queries,
            references=references,
            k=3,
            device=torch.device("cpu"),
            query_block_size=2,
            reference_block_size=4,
        )
        expected_distances, expected_indices = torch.cdist(queries, references).topk(
            3,
            dim=1,
            largest=False,
            sorted=True,
        )

        torch.testing.assert_close(distances, expected_distances, atol=1.0e-5, rtol=1.0e-5)
        torch.testing.assert_close(indices, expected_indices)


if __name__ == "__main__":
    unittest.main()
