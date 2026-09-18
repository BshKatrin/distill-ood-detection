"""Tests for global NMF concept masking."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from distill_ood_detection.config import FeatureDenoisingConfig, load_config
from distill_ood_detection.distillation.feature_denoising import (
    collect_feature_denoising_reconstruction_scores,
    feature_denoising_predictions,
    feature_denoising_score_name,
    sample_feature_denoising_pca_batch,
    sample_nmf_concept_keep_mask,
)
from distill_ood_detection.distillation.nmf import (
    NmfConceptProjector,
    fit_nmf_concept_projector_from_activations,
    load_nmf_concept_projector,
    save_nmf_concept_projector,
)


def test_nmf_projection_preserves_nchw_spatial_alignment() -> None:
    features = torch.tensor(
        [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[9.0, 10.0], [11.0, 12.0]],
            ]
        ]
    )
    projector = NmfConceptProjector(torch.eye(3), encoding_max_iter=2)

    scores = projector.transform(features)
    reconstructed = projector.inverse_transform(scores)

    torch.testing.assert_close(scores, features.permute(0, 2, 3, 1))
    torch.testing.assert_close(reconstructed, features)


def test_nmf_concept_mask_is_per_image_and_forces_a_hidden_concept() -> None:
    torch.manual_seed(7)
    concept_scores = torch.ones(2_000, 4, 4, 32)

    keep_mask = sample_nmf_concept_keep_mask(concept_scores, 0.2)

    assert keep_mask.shape == (2_000, 1, 1, 32)
    assert torch.all((keep_mask == 0.0) | (keep_mask == 1.0))
    assert torch.all((keep_mask == 0.0).any(dim=-1))
    masked_fraction = float((keep_mask == 0.0).float().mean())
    assert 0.18 < masked_fraction < 0.22


def test_nmf_batch_targets_clean_sp_and_masks_full_concept_directions() -> None:
    torch.manual_seed(2)
    features = torch.arange(1, 25, dtype=torch.float32).reshape(2, 3, 2, 2)
    projector = NmfConceptProjector(torch.eye(3), encoding_max_iter=2)
    config = FeatureDenoisingConfig(
        method="nmf_concept_masked_residual_reconstruction",
        nmf_components=3,
        nmf_mask_probability=0.2,
    )

    batch = sample_feature_denoising_pca_batch(
        features,
        config,
        nmf_projector=projector,
    )
    predictions = feature_denoising_predictions(_ZeroCorrection(), batch, config)

    torch.testing.assert_close(batch.targets, features)
    torch.testing.assert_close(predictions, batch.corrupted_features)
    assert batch.keep_mask.shape == (2, 1, 1, 1)
    assert torch.count_nonzero(batch.keep_mask) == 0
    assert torch.all(
        (batch.student_inputs != batch.targets).flatten(start_dim=1).any(dim=1)
    )


def test_nmf_score_export_omits_baseline_and_keeps_improvements() -> None:
    torch.manual_seed(3)
    features = torch.arange(1, 25, dtype=torch.float32).reshape(2, 3, 2, 2)
    labels = torch.tensor([0, 1])
    loader = DataLoader(TensorDataset(features, labels), batch_size=2)
    projector = NmfConceptProjector(torch.eye(3), encoding_max_iter=2)
    config = FeatureDenoisingConfig(
        method="nmf_concept_masked_residual_reconstruction",
        nmf_components=3,
        nmf_mask_probability=0.2,
        evaluation_draws=3,
    )

    result = collect_feature_denoising_reconstruction_scores(
        student=_ZeroCorrection(),
        loader=loader,
        device=torch.device("cpu"),
        perturbation_forwarder=_IdentityFeatureForwarder(),
        feature_denoising_config=config,
        nmf_projector=projector,
    )

    assert "identity_error" not in result
    assert set(result) == {
        "labels",
        "raw_reconstruction_error",
        "scores",
        "improvement",
        "relative_improvement",
    }
    torch.testing.assert_close(result["scores"], -result["raw_reconstruction_error"])
    torch.testing.assert_close(result["improvement"], torch.zeros(2))
    torch.testing.assert_close(result["relative_improvement"], torch.zeros(2))


def test_minibatch_nmf_fit_and_artifact_round_trip() -> None:
    activations = torch.rand(8, 4, 2, 2)
    with TemporaryDirectory() as directory:
        activation_path = Path(directory) / "activations.pt"
        projector_path = Path(directory) / "projector.pt"
        torch.save(
            {
                "dataset": "toy_train",
                "split": "train",
                "layer": "layer4",
                "activations": activations,
            },
            activation_path,
        )
        projector = fit_nmf_concept_projector_from_activations(
            activation_path,
            n_components=2,
            batch_size=8,
            max_iter=10,
            encoding_max_iter=5,
            random_state=42,
            expected_dataset="toy_train",
            expected_layer="layer4",
        )
        save_nmf_concept_projector(projector_path, projector, {"scope": "global"})
        loaded = load_nmf_concept_projector(projector_path)

    assert loaded.components.shape == (2, 4)
    assert torch.all(loaded.components >= 0.0)
    assert loaded.encoding_max_iter == 5
    assert loaded.fit_reconstruction_error is not None
    assert loaded.fit_n_iter is not None
    torch.testing.assert_close(loaded.components, projector.components)


def test_initial_nmf_configs_use_agreed_parameters() -> None:
    for dataset in ("cifar_10", "cifar_100"):
        config = load_config(
            Path(
                "configs/students/feature_denoising/nmf_concept_masking/"
                f"{dataset}/resnet18/global_layer4_k32_mask_p020.yaml"
            )
        )
        denoising = config.strategy.feature_denoising
        assert denoising.method == "nmf_concept_masked_residual_reconstruction"
        assert denoising.nmf_components == 32
        assert denoising.nmf_mask_probability == 0.2
        assert denoising.evaluation_draws == 10
        assert config.student.input_shape == (512, 4, 4)


def test_nmf_score_name() -> None:
    assert feature_denoising_score_name(
        "nmf_concept_masked_residual_reconstruction"
    ) == "feature_denoising_nmf_concept_residual_reconstruction_error"


class _ZeroCorrection(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(inputs)


class _IdentityFeatureForwarder(nn.Module):
    def forward_to_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs
