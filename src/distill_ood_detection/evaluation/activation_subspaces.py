"""Activation-subspace decomposition for Feature Denoising residuals."""

from __future__ import annotations

import torch


def classifier_svd(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return classifier singular values and a complete right-singular basis.

    The rows of the returned basis are ordered from the most decisive to the
    least decisive classifier directions.
    """

    if weight.ndim != 2:
        raise ValueError(f"Classifier weight must be two-dimensional; got {weight.shape}")
    _left, singular_values, right_basis = torch.linalg.svd(
        weight,
        full_matrices=True,
    )
    return singular_values, right_basis


def select_balanced_subspace_dimension(
    right_basis: torch.Tensor,
    id_train_activations: torch.Tensor,
) -> tuple[int, torch.Tensor]:
    """Select ActSub ``k`` by balancing mean projected activation norms.

    This implements the criterion used by Zongur et al.: for every split in
    the complete SVD basis, compare the mean L2 norms of the decisive and
    insignificant projections and choose the split with the smallest absolute
    difference. Candidate dimensions match the reference implementation,
    ``0 <= k < activation_dimension``.
    """

    _validate_basis_and_embeddings(right_basis, id_train_activations)
    coordinates = id_train_activations @ right_basis.T
    squared = coordinates.square()
    decisive_squared = torch.cat(
        [
            torch.zeros(
                (squared.shape[0], 1),
                dtype=squared.dtype,
                device=squared.device,
            ),
            squared.cumsum(dim=1)[:, :-1],
        ],
        dim=1,
    )
    total_squared = squared.sum(dim=1, keepdim=True)
    insignificant_squared = (total_squared - decisive_squared).clamp_min(0.0)
    norm_gaps = (
        decisive_squared.sqrt().mean(dim=0)
        - insignificant_squared.sqrt().mean(dim=0)
    ).abs()
    return int(norm_gaps.argmin().item()), norm_gaps


def feature_denoising_subspace_errors(
    *,
    right_basis: torch.Tensor,
    decisive_dimension: int,
    context: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1.0e-12,
) -> dict[str, torch.Tensor]:
    """Return per-sample reconstruction diagnostics in both SVD subspaces.

    Reconstruction errors are mean squared errors within each subspace.
    Relative improvement follows the existing project definition separately
    in each subspace: ``(identity_error - reconstruction_error) /
    max(identity_error, eps)``.
    """

    _validate_basis_and_embeddings(right_basis, target)
    if context.shape != target.shape or prediction.shape != target.shape:
        raise ValueError(
            "Context, prediction, and target must have identical shapes; got "
            f"{context.shape}, {prediction.shape}, and {target.shape}"
        )
    dimension = right_basis.shape[0]
    if not 0 < decisive_dimension < dimension:
        raise ValueError(
            "Subspace diagnostics require both subspaces to be non-empty; got "
            f"k={decisive_dimension} for dimension={dimension}"
        )

    prediction_residual = prediction - target
    identity_residual = context - target
    decisive_basis = right_basis[:decisive_dimension]
    decisive_prediction_energy = (
        prediction_residual @ decisive_basis.T
    ).square().sum(dim=1)
    decisive_identity_energy = (
        identity_residual @ decisive_basis.T
    ).square().sum(dim=1)
    prediction_energy = prediction_residual.square().sum(dim=1)
    identity_energy = identity_residual.square().sum(dim=1)
    component_energies = {
        "decisive": (
            decisive_prediction_energy,
            decisive_identity_energy,
            decisive_dimension,
        ),
        "insignificant": (
            (prediction_energy - decisive_prediction_energy).clamp_min(0.0),
            (identity_energy - decisive_identity_energy).clamp_min(0.0),
            dimension - decisive_dimension,
        ),
    }
    result: dict[str, torch.Tensor] = {}
    for name, energies in component_energies.items():
        reconstruction_energy, identity_component_energy, subspace_dimension = energies
        reconstruction_error = reconstruction_energy / subspace_dimension
        identity_error = identity_component_energy / subspace_dimension
        result[f"{name}_reconstruction_error"] = reconstruction_error
        result[f"{name}_identity_error"] = identity_error
        result[f"{name}_improvement"] = identity_error - reconstruction_error
        result[f"{name}_relative_improvement"] = (
            identity_error - reconstruction_error
        ) / identity_error.clamp_min(eps)
    return result


def _validate_basis_and_embeddings(
    right_basis: torch.Tensor,
    embeddings: torch.Tensor,
) -> None:
    if right_basis.ndim != 2 or right_basis.shape[0] != right_basis.shape[1]:
        raise ValueError(f"Right-singular basis must be square; got {right_basis.shape}")
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be two-dimensional; got {embeddings.shape}")
    if embeddings.shape[1] != right_basis.shape[1]:
        raise ValueError(
            "Embedding and basis dimensions differ: "
            f"{embeddings.shape[1]} != {right_basis.shape[1]}"
        )
