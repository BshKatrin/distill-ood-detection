"""Tests for random channel and whitened-PCA subspace ensembles."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from distill_ood_detection.config import load_config, parse_config
from distill_ood_detection.distillation.subspace_ensemble import (
    FittedSubspaceEnsemble,
    assign_member_subspaces,
    fit_whitened_pca,
    infer_subspace_ensemble,
    load_fitted_subspace_ensemble,
    member_inputs,
    save_fitted_subspace_ensemble,
)


class _FeatureForwarder(nn.Module):
    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = images.repeat(1, 2, 1, 1)
        logits = features.mean(dim=(2, 3))[:, :2]
        return logits, features


class SubspaceEnsembleTests(unittest.TestCase):
    """Validate selection, whitening, transforms, configs, and inference."""

    def test_ordered_assignment_is_a_complete_disjoint_partition(self) -> None:
        selections = assign_member_subspaces(
            ensemble_size=16,
            total_dimension=512,
            assignment="ordered",
            seed=42,
        )

        self.assertEqual(selections.shape, (16, 32))
        torch.testing.assert_close(selections.flatten(), torch.arange(512))

    def test_partitioned_assignment_is_disjoint_and_reproducible(self) -> None:
        first = assign_member_subspaces(
            ensemble_size=16,
            total_dimension=512,
            assignment="partitioned",
            seed=42,
        )
        second = assign_member_subspaces(
            ensemble_size=16,
            total_dimension=512,
            assignment="partitioned",
            seed=42,
        )

        torch.testing.assert_close(first, second)
        self.assertEqual(first.shape, (16, 32))
        self.assertEqual(first.unique().numel(), 512)
        torch.testing.assert_close(first.flatten().sort().values, torch.arange(512))
        self.assertFalse(torch.equal(first[0], first[1]))
        self.assertEqual(torch.isin(first[0], first[1]).sum().item(), 0)

    def test_assignment_rejects_non_divisible_dimension(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            assign_member_subspaces(
                ensemble_size=16,
                total_dimension=513,
                assignment="partitioned",
                seed=42,
            )

    def test_complete_pca_coordinates_are_whitened(self) -> None:
        generator = torch.Generator().manual_seed(42)
        base = torch.randn(500, 4, generator=generator)
        mixing = torch.tensor(
            [
                [3.0, 0.5, 0.0, 0.0],
                [0.0, 2.0, 0.4, 0.0],
                [0.0, 0.0, 1.0, 0.2],
                [0.1, 0.0, 0.0, 0.5],
            ]
        )
        embeddings = base @ mixing + torch.tensor([2.0, -1.0, 0.5, 3.0])

        mean, components, stds = fit_whitened_pca(
            embeddings,
            epsilon=1.0e-6,
        )
        coordinates = (embeddings - mean) @ components.T / stds
        covariance = coordinates.T @ coordinates / (coordinates.shape[0] - 1)

        torch.testing.assert_close(
            coordinates.mean(dim=0),
            torch.zeros(4),
            atol=1.0e-5,
            rtol=0.0,
        )
        torch.testing.assert_close(
            covariance,
            torch.eye(4),
            atol=1.0e-4,
            rtol=1.0e-4,
        )

    def test_member_inputs_cover_all_three_methods(self) -> None:
        features = torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2)
        selections = torch.tensor([[0, 2], [1, 3]])
        common = {
            "selections": selections,
            "expected_feature_shape": (4, 2, 2),
            "master_seed": 42,
        }

        flattened = member_inputs(
            features,
            FittedSubspaceEnsemble(method="channel_flatten", **common),
        )
        gap = member_inputs(
            features,
            FittedSubspaceEnsemble(method="channel_gap", **common),
        )
        pca = member_inputs(
            features,
            FittedSubspaceEnsemble(
                method="pca_gap",
                pca_mean=torch.zeros(4),
                pca_components=torch.eye(4),
                pca_stds=torch.ones(4),
                **common,
            ),
        )

        self.assertEqual(flattened[0].shape, (1, 8))
        torch.testing.assert_close(gap[0], torch.tensor([[1.5, 9.5]]))
        torch.testing.assert_close(pca[1], torch.tensor([[5.5, 13.5]]))

    def test_fitted_artifact_round_trip(self) -> None:
        fitted = FittedSubspaceEnsemble(
            method="pca_gap",
            assignment="partitioned",
            selections=torch.tensor([[0, 2], [1, 3]]),
            expected_feature_shape=(4, 1, 1),
            master_seed=42,
            pca_mean=torch.arange(4, dtype=torch.float32),
            pca_components=torch.eye(4),
            pca_stds=torch.ones(4),
            pca_fit_samples=20,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "subspace_ensemble.pt"
            save_fitted_subspace_ensemble(path, fitted)
            loaded = load_fitted_subspace_ensemble(path, torch.device("cpu"))

        self.assertEqual(loaded.method, fitted.method)
        self.assertEqual(loaded.pca_fit_samples, 20)
        torch.testing.assert_close(loaded.selections, fitted.selections)
        torch.testing.assert_close(loaded.pca_components, fitted.pca_components)

    def test_inference_preserves_member_axis(self) -> None:
        students = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
        fitted = FittedSubspaceEnsemble(
            method="channel_gap",
            selections=torch.tensor([[0, 1], [1, 0]]),
            expected_feature_shape=(2, 2, 2),
            master_seed=42,
        )
        loader = DataLoader(
            TensorDataset(
                torch.randn(3, 1, 2, 2),
                torch.tensor([0, 1, 0]),
            ),
            batch_size=2,
        )

        logits, probabilities, labels = infer_subspace_ensemble(
            students=students,
            fitted=fitted,
            forwarder=_FeatureForwarder(),
            loader=loader,
            device=torch.device("cpu"),
        )

        self.assertEqual(logits.shape, (3, 2, 2))
        self.assertEqual(probabilities.shape, (3, 2, 2))
        torch.testing.assert_close(probabilities.sum(dim=2), torch.ones(3, 2))
        torch.testing.assert_close(labels, torch.tensor([0, 1, 0]))

    def test_all_assignment_experiment_configs_load(self) -> None:
        config_paths = sorted(
            Path("configs/students/subspace_ensemble").glob("**/*.yaml")
        )

        self.assertEqual(len(config_paths), 12)
        assignments: list[str] = []
        for path in config_paths:
            with self.subTest(path=path):
                config = load_config(path)
                self.assertEqual(config.strategy.name, "subspace_ensemble")
                self.assertEqual(
                    config.training.enabled_methods(),
                    ("mse_logits", "kl_divergence"),
                )
                self.assertEqual(
                    config.strategy.subspace_ensemble.ensemble_size,
                    16,
                )
                self.assertEqual(
                    config.strategy.subspace_ensemble.subset_size,
                    32,
                )
                assignments.append(config.strategy.subspace_ensemble.assignment)
                self.assertEqual(config.training.defaults.seed, 42)
        self.assertEqual(assignments.count("ordered"), 6)
        self.assertEqual(assignments.count("partitioned"), 6)

    def test_config_rejects_non_divisible_feature_dimension(self) -> None:
        raw = {
            "experiment_name": "invalid_subspace_ensemble",
            "run_dir": "runs/tests/invalid_subspace_ensemble",
            "student": {
                "kind": "linear",
                "feature_layer": "layer4",
                "input_shape": [32],
            },
            "strategy": {
                "name": "subspace_ensemble",
                "subspace_ensemble": {
                    "ensemble_size": 16,
                    "expected_feature_shape": [513, 1, 1],
                },
            },
            "training": {"methods": {"mse_logits": {}}},
        }

        with self.assertRaisesRegex(ValueError, "divisible"):
            parse_config(raw)

    def test_rejects_cross_entropy_subspace_ensemble(self) -> None:
        raw = {
            "experiment_name": "invalid_subspace_ensemble",
            "run_dir": "runs/tests/invalid_subspace_ensemble",
            "student": {
                "kind": "linear",
                "feature_layer": "layer4",
                "input_shape": [128],
            },
            "strategy": {
                "name": "subspace_ensemble",
                "subspace_ensemble": {},
            },
            "training": {"methods": {"cross_entropy": {}}},
        }

        with self.assertRaises(ValueError):
            parse_config(raw)


if __name__ == "__main__":
    unittest.main()
