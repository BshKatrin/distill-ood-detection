"""Training helpers for ensembles over random teacher-feature subspaces."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import mlflow
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from distill_ood_detection.config import (
    DistillationMethod,
    OptimizerConfig,
    ResolvedTrainingMethodConfig,
    SubspaceAssignment,
    SubspaceEnsembleConfig,
    SubspaceEnsembleMethod,
)
from distill_ood_detection.distillation.losses import distillation_loss
from distill_ood_detection.distillation.train import build_optimizer
from distill_ood_detection.utils import write_json


@dataclass(frozen=True)
class FittedSubspaceEnsemble:
    """Selected member subspaces and optional whitened GAP-PCA transform."""

    method: SubspaceEnsembleMethod
    selections: torch.Tensor
    expected_feature_shape: tuple[int, int, int]
    master_seed: int
    assignment: SubspaceAssignment = "partitioned"
    pca_mean: torch.Tensor | None = None
    pca_components: torch.Tensor | None = None
    pca_stds: torch.Tensor | None = None
    pca_fit_samples: int | None = None
    whitening_epsilon: float = 1.0e-6

    @property
    def ensemble_size(self) -> int:
        """Return the number of independently selected member subspaces."""

        return self.selections.shape[0]

    @property
    def subset_size(self) -> int:
        """Return the number of channels or PCA coordinates per member."""

        return self.selections.shape[1]

    def to(self, device: torch.device) -> FittedSubspaceEnsemble:
        """Move tensor state to a device."""

        return FittedSubspaceEnsemble(
            method=self.method,
            selections=self.selections.to(device),
            expected_feature_shape=self.expected_feature_shape,
            master_seed=self.master_seed,
            assignment=self.assignment,
            pca_mean=None if self.pca_mean is None else self.pca_mean.to(device),
            pca_components=(
                None if self.pca_components is None else self.pca_components.to(device)
            ),
            pca_stds=None if self.pca_stds is None else self.pca_stds.to(device),
            pca_fit_samples=self.pca_fit_samples,
            whitening_epsilon=self.whitening_epsilon,
        )


def assign_member_subspaces(
    *,
    ensemble_size: int,
    total_dimension: int,
    assignment: SubspaceAssignment,
    seed: int,
) -> torch.Tensor:
    """Assign a complete, deterministic, non-overlapping feature partition."""

    if ensemble_size <= 1:
        raise ValueError("ensemble_size must be greater than one")
    if total_dimension <= 0:
        raise ValueError("total_dimension must be positive")
    if total_dimension % ensemble_size != 0:
        raise ValueError("total_dimension must be divisible by ensemble_size")
    if assignment == "ordered":
        indices = torch.arange(total_dimension)
    elif assignment == "partitioned":
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(total_dimension, generator=generator)
    else:
        raise ValueError(f"Unsupported subspace assignment: {assignment}")
    return indices.reshape(ensemble_size, total_dimension // ensemble_size)


def fit_whitened_pca(
    embeddings: torch.Tensor,
    *,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit a complete covariance PCA transform with unit-variance coordinates."""

    if embeddings.ndim != 2:
        raise ValueError("PCA embeddings must have shape (samples, channels)")
    if embeddings.shape[0] < 2:
        raise ValueError("PCA fitting requires at least two samples")
    if epsilon <= 0.0:
        raise ValueError("PCA whitening epsilon must be positive")

    embeddings = embeddings.float()
    mean = embeddings.mean(dim=0)
    centered = embeddings - mean
    covariance = centered.T @ centered / (embeddings.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance.double())
    order = torch.arange(eigenvalues.shape[0] - 1, -1, -1)
    eigenvalues = eigenvalues[order]
    components = eigenvectors[:, order].T
    components = _canonicalize_component_signs(components)
    stds = eigenvalues.clamp_min(0.0).sqrt()
    return mean, components.float(), stds.float()


@torch.no_grad()
def collect_gap_embeddings(
    *,
    forwarder: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
    expected_feature_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Collect GAP embeddings from the deterministic ID student-training split."""

    batches: list[torch.Tensor] = []
    forwarder.eval()
    for images, _labels in loader:
        features = forwarder.forward_to_features(images.to(device))
        _validate_feature_shape(features, expected_feature_shape)
        batches.append(features.mean(dim=(2, 3)).cpu())
    return torch.cat(batches, dim=0)


def fit_subspace_ensemble(
    *,
    config: SubspaceEnsembleConfig,
    forwarder: nn.Module,
    train_loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
    seed: int,
) -> FittedSubspaceEnsemble:
    """Fit the shared transform and reproducibly select all member subspaces."""

    channel_count = config.expected_feature_shape[0]
    selections = assign_member_subspaces(
        ensemble_size=config.ensemble_size,
        total_dimension=channel_count,
        assignment=config.assignment,
        seed=seed,
    )
    if config.method != "pca_gap":
        return FittedSubspaceEnsemble(
            method=config.method,
            selections=selections,
            expected_feature_shape=config.expected_feature_shape,
            master_seed=seed,
            assignment=config.assignment,
            whitening_epsilon=config.whitening_epsilon,
        )

    train_embeddings = collect_gap_embeddings(
        forwarder=forwarder,
        loader=train_loader,
        device=device,
        expected_feature_shape=config.expected_feature_shape,
    )
    mean, components, stds = fit_whitened_pca(
        train_embeddings,
        epsilon=config.whitening_epsilon,
    )
    return FittedSubspaceEnsemble(
        method=config.method,
        selections=selections,
        expected_feature_shape=config.expected_feature_shape,
        master_seed=seed,
        assignment=config.assignment,
        pca_mean=mean,
        pca_components=components,
        pca_stds=stds,
        pca_fit_samples=train_embeddings.shape[0],
        whitening_epsilon=config.whitening_epsilon,
    )


def member_inputs(
    features: torch.Tensor,
    fitted: FittedSubspaceEnsemble,
) -> list[torch.Tensor]:
    """Build one selected feature representation for every ensemble member."""

    _validate_feature_shape(features, fitted.expected_feature_shape)
    if fitted.method == "channel_flatten":
        return [
            features.index_select(1, indices).flatten(start_dim=1)
            for indices in fitted.selections
        ]

    pooled = features.mean(dim=(2, 3))
    if fitted.method == "channel_gap":
        representation = pooled
    elif fitted.method == "pca_gap":
        if (
            fitted.pca_mean is None
            or fitted.pca_components is None
            or fitted.pca_stds is None
        ):
            raise ValueError("pca_gap requires fitted PCA mean, components, and stds")
        mean = fitted.pca_mean.to(device=pooled.device, dtype=pooled.dtype)
        components = fitted.pca_components.to(
            device=pooled.device,
            dtype=pooled.dtype,
        )
        stds = fitted.pca_stds.to(device=pooled.device, dtype=pooled.dtype)
        representation = ((pooled - mean) @ components.T) / stds.clamp_min(
            fitted.whitening_epsilon
        )
    else:
        raise ValueError(f"Unsupported subspace ensemble method: {fitted.method}")
    return [
        representation.index_select(1, indices)
        for indices in fitted.selections
    ]


def save_fitted_subspace_ensemble(
    path: Path,
    fitted: FittedSubspaceEnsemble,
) -> None:
    """Save the shared selection and optional PCA state."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 2,
            "method": fitted.method,
            "assignment": fitted.assignment,
            "selections": fitted.selections.cpu(),
            "expected_feature_shape": fitted.expected_feature_shape,
            "master_seed": fitted.master_seed,
            "pca_mean": (
                None if fitted.pca_mean is None else fitted.pca_mean.cpu()
            ),
            "pca_components": (
                None if fitted.pca_components is None else fitted.pca_components.cpu()
            ),
            "pca_stds": (
                None if fitted.pca_stds is None else fitted.pca_stds.cpu()
            ),
            "pca_fit_samples": fitted.pca_fit_samples,
            "whitening_epsilon": fitted.whitening_epsilon,
        },
        path,
    )


def load_fitted_subspace_ensemble(
    path: Path,
    device: torch.device,
) -> FittedSubspaceEnsemble:
    """Load a saved shared selection and optional PCA state."""

    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact.get("version") != 2:
        raise ValueError(
            "Unsupported subspace ensemble artifact version; retrain with a "
            "deterministic disjoint assignment"
        )
    fitted = FittedSubspaceEnsemble(
        method=artifact["method"],
        assignment=artifact["assignment"],
        selections=artifact["selections"],
        expected_feature_shape=tuple(artifact["expected_feature_shape"]),
        master_seed=int(artifact["master_seed"]),
        pca_mean=artifact["pca_mean"],
        pca_components=artifact["pca_components"],
        pca_stds=artifact["pca_stds"],
        pca_fit_samples=artifact["pca_fit_samples"],
        whitening_epsilon=float(artifact["whitening_epsilon"]),
    )
    return fitted.to(device)


def train_subspace_ensemble(
    *,
    students: nn.ModuleList,
    fitted: FittedSubspaceEnsemble,
    method: DistillationMethod,
    forwarder: nn.Module,
    train_loader: DataLoader[tuple[torch.Tensor, int]],
    validation_loader: DataLoader[tuple[torch.Tensor, int]],
    test_loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
    optimizer_config: OptimizerConfig,
    training_config: ResolvedTrainingMethodConfig,
    output_dir: Path,
    mlflow_enabled: bool = False,
) -> dict[str, object]:
    """Train all members jointly while preserving independent member gradients."""

    if method not in {"mse_logits", "kl_divergence"}:
        raise ValueError("Subspace ensembles support only MSE-logit and KL training")
    if len(students) != fitted.ensemble_size:
        raise ValueError("Student count must match fitted ensemble size")
    students.to(device)
    forwarder.to(device)
    forwarder.eval()
    fitted = fitted.to(device)
    optimizer = build_optimizer(students, optimizer_config)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = output_dir / "best_ensemble.pt"
    latest_checkpoint = output_dir / "latest_ensemble.pt"
    history: list[dict[str, object]] = []
    best_validation_loss = float("inf")
    started_at = time.time()

    for epoch in range(1, training_config.epochs + 1):
        students.train()
        total_member_loss = 0.0
        total_examples = 0
        progress = tqdm(train_loader, desc=f"{method} ensemble epoch {epoch}", leave=False)
        for step, (images, labels) in enumerate(progress, start=1):
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                teacher_logits, features = forwarder(images)
                inputs = member_inputs(features, fitted)
            optimizer.zero_grad(set_to_none=True)
            member_losses = [
                distillation_loss(
                    method,
                    student(member_input),
                    teacher_logits,
                    labels=labels,
                    temperature=training_config.temperature,
                    alpha=training_config.alpha,
                )
                for student, member_input in zip(students, inputs, strict=True)
            ]
            torch.stack(member_losses).sum().backward()
            optimizer.step()
            batch_size = images.shape[0]
            mean_loss = torch.stack([loss.detach() for loss in member_losses]).mean()
            total_member_loss += mean_loss.item() * batch_size
            total_examples += batch_size
            if step % training_config.log_every_steps == 0:
                progress.set_postfix(
                    loss=f"{total_member_loss / total_examples:.4f}"
                )

        validation = evaluate_subspace_ensemble(
            students=students,
            fitted=fitted,
            method=method,
            forwarder=forwarder,
            loader=validation_loader,
            device=device,
            training_config=training_config,
        )
        record: dict[str, object] = {
            "epoch": epoch,
            "training_loss": total_member_loss / total_examples,
            **validation,
        }
        history.append(record)
        write_json(output_dir / "history.json", {"history": history})
        validation_loss = float(validation["validation_distillation_loss"])
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(students.state_dict(), best_checkpoint)
        if mlflow_enabled:
            mlflow.log_metrics(
                {
                    "training_loss": float(record["training_loss"]),
                    "validation_distillation_loss": validation_loss,
                    "validation_ensemble_accuracy": float(
                        validation["validation_ensemble_accuracy"]
                    ),
                },
                step=epoch,
            )

    torch.save(students.state_dict(), latest_checkpoint)
    students.load_state_dict(
        torch.load(best_checkpoint, map_location=device, weights_only=True)
    )
    test = evaluate_subspace_ensemble(
        students=students,
        fitted=fitted,
        method=method,
        forwarder=forwarder,
        loader=test_loader,
        device=device,
        training_config=training_config,
        split="test",
    )
    summary: dict[str, object] = {
        "method": method,
        "ensemble_size": fitted.ensemble_size,
        "subset_size": fitted.subset_size,
        "subspace_method": fitted.method,
        "subspace_assignment": fitted.assignment,
        "epochs": training_config.epochs,
        "best_validation_distillation_loss": best_validation_loss,
        "final_validation_distillation_loss": history[-1][
            "validation_distillation_loss"
        ],
        "final_validation_ensemble_accuracy": history[-1][
            "validation_ensemble_accuracy"
        ],
        **test,
        "latest_checkpoint_path": str(latest_checkpoint),
        "best_checkpoint_path": str(best_checkpoint),
        "seconds": round(time.time() - started_at, 3),
    }
    write_json(output_dir / "metrics.json", summary)
    return summary


@torch.no_grad()
def evaluate_subspace_ensemble(
    *,
    students: nn.ModuleList,
    fitted: FittedSubspaceEnsemble,
    method: DistillationMethod,
    forwarder: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
    training_config: ResolvedTrainingMethodConfig,
    split: str = "validation",
) -> dict[str, object]:
    """Evaluate mean member loss and the mean-probability ensemble classifier."""

    students.eval()
    forwarder.eval()
    total_member_losses = torch.zeros(len(students), dtype=torch.float64)
    member_correct = torch.zeros(len(students), dtype=torch.long)
    ensemble_correct = 0
    total_examples = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        teacher_logits, features = forwarder(images)
        inputs = member_inputs(features, fitted)
        logits = torch.stack(
            [
                student(member_input)
                for student, member_input in zip(students, inputs, strict=True)
            ],
            dim=1,
        )
        for member_index in range(len(students)):
            loss = distillation_loss(
                method,
                logits[:, member_index],
                teacher_logits,
                labels=labels,
                temperature=training_config.temperature,
                alpha=training_config.alpha,
            )
            total_member_losses[member_index] += loss.item() * images.shape[0]
            member_correct[member_index] += (
                logits[:, member_index].argmax(dim=1) == labels
            ).sum().cpu()
        mean_probabilities = F.softmax(logits, dim=2).mean(dim=1)
        ensemble_correct += (
            mean_probabilities.argmax(dim=1) == labels
        ).sum().item()
        total_examples += images.shape[0]

    member_losses = total_member_losses / total_examples
    member_accuracies = member_correct.double() / total_examples
    return {
        f"{split}_distillation_loss": member_losses.mean().item(),
        f"{split}_ensemble_accuracy": ensemble_correct / total_examples,
        f"{split}_member_losses": member_losses.tolist(),
        f"{split}_member_accuracies": member_accuracies.tolist(),
    }


@torch.no_grad()
def infer_subspace_ensemble(
    *,
    students: nn.ModuleList,
    fitted: FittedSubspaceEnsemble,
    forwarder: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return logits/probabilities with shape ``(samples, members, classes)``."""

    students.to(device)
    students.eval()
    forwarder.to(device)
    forwarder.eval()
    fitted = fitted.to(device)
    logits_batches: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for images, batch_labels in loader:
        _teacher_logits, features = forwarder(images.to(device))
        inputs = member_inputs(features, fitted)
        logits_batches.append(
            torch.stack(
                [
                    student(member_input)
                    for student, member_input in zip(students, inputs, strict=True)
                ],
                dim=1,
            ).cpu()
        )
        labels.append(batch_labels.cpu())
    logits = torch.cat(logits_batches, dim=0)
    return logits, F.softmax(logits, dim=2), torch.cat(labels, dim=0)


def _canonicalize_component_signs(components: torch.Tensor) -> torch.Tensor:
    largest_loading_indices = components.abs().argmax(dim=1)
    signs = torch.sign(
        components[
            torch.arange(components.shape[0]),
            largest_loading_indices,
        ]
    )
    signs[signs == 0] = 1
    return components * signs[:, None]


def _validate_feature_shape(
    features: torch.Tensor,
    expected_feature_shape: tuple[int, int, int],
) -> None:
    if features.ndim != 4:
        raise ValueError(
            "Subspace ensembles require feature maps with shape "
            "(batch, channels, height, width)"
        )
    if tuple(features.shape[1:]) != expected_feature_shape:
        raise ValueError(
            "Teacher feature shape differs from config: "
            f"{tuple(features.shape[1:])} != {expected_feature_shape}"
        )
