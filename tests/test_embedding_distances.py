"""Tests for layer4 embedding extraction and exact FAISS k-NN search."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from distill_ood_detection.config import parse_embedding_distance_config
from distill_ood_detection.datasets.inference import NamedLoader
from distill_ood_detection.evaluation.nearest_neighbors import (
    FaissExactL2Index,
    MaskedChannelExactL2Index,
)
from distill_ood_detection.experiments.export_embedding_distances import (
    KNN_SCORE_SIGNS,
    compute_distance_quantities,
    export_loader_embeddings,
    knn_score_vectors,
    load_embedding_shards,
    load_output_shards,
)


class ToyTeacher(nn.Module):
    """Small convolutional teacher exposing a layer4 stage."""

    def __init__(self) -> None:
        super().__init__()
        self.layer4 = nn.Conv2d(3, 5, kernel_size=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(5, 2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return logits for one image batch."""

        features = self.layer4(images)
        return self.classifier(torch.flatten(self.pool(features), start_dim=1))


class EmbeddingDistanceTests(unittest.TestCase):
    """Validate pooled artifacts and exact FAISS distances."""

    def test_masked_channel_index_matches_brute_force_across_chunks(self) -> None:
        references = torch.tensor(
            [
                [[[0.0, 1.0]], [[2.0, 3.0]], [[10.0, 10.0]]],
                [[[1.0, 1.0]], [[2.0, 4.0]], [[20.0, 20.0]]],
                [[[4.0, 4.0]], [[4.0, 4.0]], [[30.0, 30.0]]],
                [[[8.0, 8.0]], [[8.0, 8.0]], [[40.0, 40.0]]],
            ]
        )
        queries = torch.tensor(
            [
                [[[0.0, 1.0]], [[2.0, 3.0]], [[999.0, 999.0]]],
                [[[4.0, 5.0]], [[100.0, 100.0]], [[30.0, 31.0]]],
            ]
        )
        keep_mask = torch.tensor(
            [
                [[[1.0]], [[1.0]], [[0.0]]],
                [[[1.0]], [[0.0]], [[1.0]]],
            ]
        )
        index = MaskedChannelExactL2Index(references, torch.device("cpu"))

        distances, indices = index.search(
            queries,
            keep_mask,
            k=2,
            query_batch_size=1,
            reference_chunk_size=2,
        )

        brute_force = (
            ((queries[:, None] - references[None]).square() * keep_mask[:, None])
            .flatten(start_dim=2)
            .sum(dim=2)
        )
        expected_distances, expected_indices = brute_force.topk(
            2,
            dim=1,
            largest=False,
            sorted=True,
        )
        torch.testing.assert_close(distances, expected_distances)
        torch.testing.assert_close(indices, expected_indices)

        predictions = index.reconstruct_hidden_channels(
            queries,
            keep_mask,
            indices,
        )
        neighbor_means = references[indices].mean(dim=1)
        expected_predictions = queries * keep_mask + neighbor_means * (1.0 - keep_mask)
        torch.testing.assert_close(predictions, expected_predictions)

    def test_parses_embedding_distance_config(self) -> None:
        config = parse_embedding_distance_config(
            {
                "experiment_name": "embedding_distances",
                "run_dir": "runs/embedding_distances",
                "k_neighbors": 5,
                "search_batch_size": 64,
                "dataset": {"name": "cifar10"},
            }
        )

        self.assertEqual(config.k_neighbors, 5)
        self.assertEqual(config.search_batch_size, 64)

    def test_parses_perturbed_embedding_distance_config(self) -> None:
        config = parse_embedding_distance_config(
            {
                "experiment_name": "perturbed_knn",
                "run_dir": "runs/perturbed_knn",
                "query_perturbed": True,
                "perturbation": {
                    "method": "clipping",
                    "teacher_target": "clean",
                    "evaluation_draws": 3,
                    "clipping_layers": {
                        "layer3": {
                            "clipping_mode": "spatial_dependent",
                            "u_min": 0.5,
                            "u_max": 1.0,
                        },
                        "layer4": {
                            "clipping_mode": "spatial_dependent",
                            "u_min": 0.5,
                            "u_max": 1.0,
                        },
                    },
                },
            }
        )

        self.assertTrue(config.query_perturbed)
        self.assertIsNotNone(config.perturbation)
        assert config.perturbation is not None
        self.assertEqual(config.perturbation.evaluation_draws, 3)
        self.assertEqual(
            config.perturbation.clipping_layers["layer3"].clipping_mode,
            "spatial_dependent",
        )

    def test_rejects_obsolete_search_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "Obsolete embedding-distance"):
            parse_embedding_distance_config(
                {
                    "experiment_name": "embedding_distances",
                    "run_dir": "runs/embedding_distances",
                    "reference_block_size": 1024,
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
                layer="layer4",
                named_loader=named_loader,
                output_dir=Path(directory),
                device=torch.device("cpu"),
                shard_size=3,
                metadata={"pooling": "gap"},
            )
            embeddings, saved_labels = load_embedding_shards(
                artifacts,
                dataset="toy_test",
                layer="layer4",
            )
            logits, probabilities = load_output_shards(
                artifacts,
                dataset="toy_test",
                layer="layer4",
            )

        expected = teacher.layer4(images).mean(dim=(-2, -1))
        expected_logits = teacher(images)
        torch.testing.assert_close(embeddings, expected)
        torch.testing.assert_close(saved_labels, labels)
        torch.testing.assert_close(logits, expected_logits)
        torch.testing.assert_close(probabilities, torch.softmax(expected_logits, dim=1))
        self.assertEqual(len(artifacts), 2)

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
        self.assertTrue(torch.all(torch.isfinite(raw_metrics["student_energy"])).item())
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
        index = FaissExactL2Index(references, torch.device("cpu"))

        result = compute_distance_quantities(
            queries=queries,
            index=index,
            reference_labels=torch.tensor([0, 1, 0]),
            query_logits=query_logits,
            query_probabilities=torch.softmax(query_logits, dim=1),
            reference_logits=reference_logits,
            reference_probabilities=reference_probabilities,
            k_neighbors=2,
            search_batch_size=1,
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

    def test_multi_draw_queries_average_all_neighbor_predictions(self) -> None:
        queries = torch.tensor([[[0.1], [9.9]]])
        references = torch.tensor([[0.0], [10.0]])
        reference_logits = torch.tensor([[4.0, 0.0], [0.0, 2.0]])
        reference_probabilities = torch.softmax(reference_logits, dim=1)
        query_logits = torch.tensor([[1.0, 1.0]])
        index = FaissExactL2Index(references, torch.device("cpu"))

        result = compute_distance_quantities(
            queries=queries,
            index=index,
            reference_labels=torch.tensor([0, 1]),
            query_logits=query_logits,
            query_probabilities=torch.softmax(query_logits, dim=1),
            reference_logits=reference_logits,
            reference_probabilities=reference_probabilities,
            k_neighbors=1,
            search_batch_size=2,
        )

        self.assertEqual(result["neighbor_indices"].shape, (1, 2, 1))
        torch.testing.assert_close(
            result["neighbor_indices"],
            torch.tensor([[[0], [1]]]),
        )
        torch.testing.assert_close(
            result["neighbor_mean_logits"],
            reference_logits.mean(dim=0, keepdim=True),
        )
        torch.testing.assert_close(
            result["neighbor_mean_probabilities"],
            reference_probabilities.mean(dim=0, keepdim=True),
        )

    def test_faiss_topk_matches_full_cdist(self) -> None:
        torch.manual_seed(7)
        queries = torch.randn(7, 6)
        references = torch.randn(11, 6)

        index = FaissExactL2Index(references, torch.device("cpu"))
        distances, indices = index.search(
            queries,
            k=3,
            batch_size=2,
        )
        expected_distances, expected_indices = torch.cdist(queries, references).topk(
            3,
            dim=1,
            largest=False,
            sorted=True,
        )

        torch.testing.assert_close(
            distances, expected_distances, atol=1.0e-5, rtol=1.0e-5
        )
        torch.testing.assert_close(indices, expected_indices)

    def test_faiss_caps_k_and_accepts_empty_queries(self) -> None:
        references = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        index = FaissExactL2Index(references, torch.device("cpu"))

        distances, indices = index.search(
            torch.empty((0, 2)),
            k=5,
            batch_size=4,
        )

        self.assertEqual(distances.shape, (0, 2))
        self.assertEqual(indices.shape, (0, 2))


if __name__ == "__main__":
    unittest.main()
