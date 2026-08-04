"""Training over decisive and insignificant classifier activation subspaces."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import mlflow
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from distill_ood_detection.config import (
    ActivationSubspaceComponent,
    ActivationSubspaceConfig,
    ActivationSubspaceTarget,
    DistillationMethod,
    OptimizerConfig,
    ResolvedTrainingMethodConfig,
)
from distill_ood_detection.distillation.losses import distillation_loss
from distill_ood_detection.distillation.train import build_optimizer
from distill_ood_detection.evaluation.activation_subspaces import (
    classifier_svd,
    select_balanced_subspace_dimension,
)
from distill_ood_detection.utils import write_json


@dataclass(frozen=True)
class PooledEmbeddingSplit:
    """Clean pooled teacher embeddings and dataset labels for one split."""

    embeddings: torch.Tensor
    labels: torch.Tensor


@dataclass(frozen=True)
class FittedActivationSubspaces:
    """Complete classifier SVD basis and its ActSub split dimension."""

    right_basis: torch.Tensor
    singular_values: torch.Tensor
    decisive_dimension: int
    norm_gaps: torch.Tensor

    @property
    def activation_dimension(self) -> int:
        """Return the full pooled activation dimension."""

        return self.right_basis.shape[0]

    @property
    def insignificant_dimension(self) -> int:
        """Return the complement dimension."""

        return self.activation_dimension - self.decisive_dimension

    def component_dimension(self, component: ActivationSubspaceComponent) -> int:
        """Return the selected component's coordinate dimension."""

        if component == "decisive":
            return self.decisive_dimension
        return self.insignificant_dimension


def collect_clean_pooled_embeddings(
    *,
    forwarder: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
) -> PooledEmbeddingSplit:
    """Collect deterministic globally averaged teacher layer features."""

    embeddings = []
    labels = []
    forwarder.eval()
    with torch.no_grad():
        for images, batch_labels in loader:
            features = forwarder.forward_to_features(images.to(device))
            if features.ndim != 4:
                raise ValueError(
                    "Activation-subspace avg pooling requires feature maps; got "
                    f"shape {tuple(features.shape)}"
                )
            embeddings.append(features.mean(dim=(2, 3)).cpu())
            labels.append(batch_labels.cpu())
    return PooledEmbeddingSplit(
        embeddings=torch.cat(embeddings, dim=0),
        labels=torch.cat(labels, dim=0),
    )


def fit_activation_subspaces(
    *,
    classifier_weight: torch.Tensor,
    train_embeddings: torch.Tensor,
    device: torch.device,
) -> FittedActivationSubspaces:
    """Fit the complete classifier SVD basis and ActSub norm-balanced split."""

    singular_values, right_basis = classifier_svd(classifier_weight.to(device))
    decisive_dimension, norm_gaps = select_balanced_subspace_dimension(
        right_basis,
        train_embeddings.to(device),
    )
    if not 0 < decisive_dimension < right_basis.shape[0]:
        raise ValueError(
            "ActSub selected an empty component; got "
            f"k={decisive_dimension}, dimension={right_basis.shape[0]}"
        )
    return FittedActivationSubspaces(
        right_basis=right_basis,
        singular_values=singular_values,
        decisive_dimension=decisive_dimension,
        norm_gaps=norm_gaps,
    )


def component_training_tensors(
    *,
    split: PooledEmbeddingSplit,
    subspaces: FittedActivationSubspaces,
    component: ActivationSubspaceComponent,
    target: ActivationSubspaceTarget,
    classifier: nn.Linear,
    device: torch.device,
) -> TensorDataset:
    """Build fixed student inputs, targets, and labels for one data split."""

    embeddings = split.embeddings.to(device)
    basis = subspaces.right_basis
    if component == "decisive":
        component_basis = basis[: subspaces.decisive_dimension]
        inputs = embeddings @ component_basis.T
        if target == "projected_logits":
            projected_embeddings = inputs @ component_basis
            targets = F.linear(
                projected_embeddings,
                classifier.weight,
                classifier.bias,
            )
        else:
            targets = inputs
    else:
        component_basis = basis[subspaces.decisive_dimension :]
        inputs = embeddings @ component_basis.T
        targets = inputs
    inputs_cpu = inputs.cpu()
    targets_cpu = inputs_cpu if target == "coordinates" else targets.cpu()
    return TensorDataset(inputs_cpu, targets_cpu, split.labels)


def train_activation_subspace_student(
    *,
    student: nn.Module,
    component: ActivationSubspaceComponent,
    target: ActivationSubspaceTarget,
    distillation_method: DistillationMethod,
    train_data: TensorDataset,
    validation_data: TensorDataset,
    test_data: TensorDataset,
    batch_size: int,
    device: torch.device,
    optimizer_config: OptimizerConfig,
    training_config: ResolvedTrainingMethodConfig,
    output_dir: Path,
    mlflow_enabled: bool = False,
) -> dict[str, float | int | str]:
    """Train one fixed-input activation-subspace student and save checkpoints."""

    student.to(device)
    optimizer = build_optimizer(student, optimizer_config)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_checkpoint = output_dir / "latest_student.pt"
    best_checkpoint = output_dir / "best_student.pt"
    history: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")
    best_validation_accuracy = 0.0
    generator = torch.Generator().manual_seed(training_config.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(validation_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    started_at = time.time()

    for epoch in range(1, training_config.epochs + 1):
        student.train()
        total_loss = 0.0
        total_examples = 0
        progress = tqdm(train_loader, desc=f"{component} epoch {epoch}", leave=False)
        for step, (inputs, targets, labels) in enumerate(progress, start=1):
            inputs = inputs.to(device)
            targets = targets.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = student(inputs)
            loss = activation_subspace_loss(
                component,
                predictions,
                targets,
                target=target,
                labels=labels,
                distillation_method=distillation_method,
                temperature=training_config.temperature,
                alpha=training_config.alpha,
            )
            loss.backward()
            optimizer.step()
            batch_examples = inputs.shape[0]
            total_loss += loss.item() * batch_examples
            total_examples += batch_examples
            if step % training_config.log_every_steps == 0:
                progress.set_postfix(loss=f"{total_loss / total_examples:.4f}")

        validation = evaluate_activation_subspace_student(
            student=student,
            component=component,
            target=target,
            distillation_method=distillation_method,
            training_config=training_config,
            loader=validation_loader,
            device=device,
        )
        record: dict[str, float | int] = {
            "epoch": epoch,
            "training_loss": total_loss / total_examples,
            "validation_loss": validation["loss"],
        }
        if "accuracy" in validation:
            record["validation_accuracy"] = validation["accuracy"]
            best_validation_accuracy = max(
                best_validation_accuracy,
                validation["accuracy"],
            )
        history.append(record)
        write_json(output_dir / "history.json", {"history": history})
        if validation["loss"] < best_validation_loss:
            best_validation_loss = validation["loss"]
            torch.save(student.state_dict(), best_checkpoint)
        if mlflow_enabled:
            mlflow.log_metrics(record, step=epoch)

    torch.save(student.state_dict(), latest_checkpoint)
    student.load_state_dict(
        torch.load(best_checkpoint, map_location=device, weights_only=True)
    )
    test = evaluate_activation_subspace_student(
        student=student,
        component=component,
        target=target,
        distillation_method=distillation_method,
        training_config=training_config,
        loader=test_loader,
        device=device,
    )
    summary: dict[str, float | int | str] = {
        "component": component,
        "target": target,
        "distillation_method": distillation_method,
        "epochs": training_config.epochs,
        "best_validation_loss": best_validation_loss,
        "final_validation_loss": history[-1]["validation_loss"],
        "test_loss": test["loss"],
        "latest_checkpoint_path": str(latest_checkpoint),
        "best_checkpoint_path": str(best_checkpoint),
        "seconds": round(time.time() - started_at, 3),
    }
    if target == "projected_logits":
        summary.update(
            {
                "best_validation_accuracy": best_validation_accuracy,
                "final_validation_accuracy": history[-1]["validation_accuracy"],
                "test_accuracy": test["accuracy"],
            }
        )
    write_json(output_dir / "metrics.json", summary)
    return summary


def activation_subspace_loss(
    component: ActivationSubspaceComponent,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    target: ActivationSubspaceTarget = "projected_logits",
    labels: torch.Tensor | None = None,
    distillation_method: DistillationMethod = "mse_logits",
    temperature: float = 1.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Return the decisive distillation or insignificant reconstruction loss."""

    if target == "projected_logits":
        return distillation_loss(
            distillation_method,
            predictions,
            targets,
            labels=labels,
            temperature=temperature,
            alpha=alpha,
        )
    return F.mse_loss(predictions, targets)


def evaluate_activation_subspace_student(
    *,
    student: nn.Module,
    component: ActivationSubspaceComponent,
    target: ActivationSubspaceTarget,
    distillation_method: DistillationMethod,
    training_config: ResolvedTrainingMethodConfig,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, float]:
    """Evaluate ID loss and decisive-student classification accuracy."""

    student.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for inputs, targets, labels in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            labels = labels.to(device)
            predictions = student(inputs)
            loss = activation_subspace_loss(
                component,
                predictions,
                targets,
                target=target,
                labels=labels,
                distillation_method=distillation_method,
                temperature=training_config.temperature,
                alpha=training_config.alpha,
            )
            batch_examples = inputs.shape[0]
            total_loss += loss.item() * batch_examples
            total_examples += batch_examples
            if target == "projected_logits":
                total_correct += (predictions.argmax(dim=1) == labels).sum().item()
    metrics = {"loss": total_loss / total_examples}
    if target == "projected_logits":
        metrics["accuracy"] = total_correct / total_examples
    return metrics


def save_activation_subspaces(
    *,
    path: Path,
    subspaces: FittedActivationSubspaces,
    config: ActivationSubspaceConfig,
    fit_samples: int,
    classifier_weight_shape: tuple[int, ...],
) -> None:
    """Save the fitted basis and trace metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "right_basis": subspaces.right_basis.cpu(),
            "singular_values": subspaces.singular_values.cpu(),
            "decisive_dimension": subspaces.decisive_dimension,
            "insignificant_dimension": subspaces.insignificant_dimension,
            "norm_gaps": subspaces.norm_gaps.cpu(),
            "component": config.component,
            "target": config.target,
            "fit_samples": fit_samples,
            "classifier_weight_shape": classifier_weight_shape,
            "selection": "actsub_mean_l2_norm_balance",
        },
        path,
    )
