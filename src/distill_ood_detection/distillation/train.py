"""Student training loop."""

from __future__ import annotations

import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import SGD, AdamW, Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import mlflow

from distill_ood_detection.config import DistillationMethod, OptimizerConfig, TrainingConfig
from distill_ood_detection.distillation.losses import distillation_loss
from distill_ood_detection.evaluation.metrics import distillation_validation_metrics
from distill_ood_detection.utils import write_json


def train_student(
    method: DistillationMethod,
    teacher: nn.Module,
    student: nn.Module,
    train_loader: DataLoader[tuple[torch.Tensor, int]],
    validation_loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
    optimizer_config: OptimizerConfig,
    training_config: TrainingConfig,
    output_dir: Path,
    mlflow_enabled: bool = True,
) -> dict[str, float | int | str]:
    """Train a student against teacher predictions and save trace artifacts."""
    student.to(device)
    optimizer = build_optimizer(student, optimizer_config)
    history: list[dict[str, float | int]] = []
    best_validation_accuracy = 0.0
    best_validation_distillation_loss = float("inf")
    checkpoint_path = output_dir / "latest_student.pt"
    best_checkpoint_path = output_dir / "best_student.pt"
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()

    for epoch in range(1, training_config.epochs + 1):
        student.train()
        total_loss = 0.0
        total_examples = 0
        progress = tqdm(train_loader, desc=f"{method} epoch {epoch}", leave=False)
        for step, (images, labels) in enumerate(progress, start=1):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_logits = teacher(images)
            student_logits = student(images)
            loss = distillation_loss(
                method,
                student_logits,
                teacher_logits,
                labels=labels,
                temperature=training_config.temperature,
                alpha=training_config.alpha,
            )
            loss.backward()
            optimizer.step()

            batch_size = images.shape[0]
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            if step % training_config.log_every_steps == 0:
                progress.set_postfix(loss=f"{total_loss / total_examples:.4f}")

        train_loss = total_loss / total_examples
        validation_metrics = distillation_validation_metrics(
            method=method,
            teacher=teacher,
            student=student,
            loader=validation_loader,
            device=device,
            temperature=training_config.temperature,
            alpha=training_config.alpha,
        )
        validation_accuracy = validation_metrics["validation_accuracy"]
        validation_distillation_loss = validation_metrics["validation_distillation_loss"]
        best_validation_accuracy = max(best_validation_accuracy, validation_accuracy)
        if validation_distillation_loss < best_validation_distillation_loss:
            best_validation_distillation_loss = validation_distillation_loss
            torch.save(student.state_dict(), best_checkpoint_path)
        epoch_record = {
            "epoch": epoch,
            "distillation_loss": train_loss,
            **validation_metrics,
        }
        history.append(epoch_record)
        write_json(output_dir / "history.json", {"history": history})
        if mlflow_enabled:
            mlflow.log_metrics(
                {
                    "distillation_loss": train_loss,
                    **validation_metrics,
                },
                step=epoch,
            )

    torch.save(student.state_dict(), checkpoint_path)

    summary = {
        "method": method,
        "epochs": training_config.epochs,
        "best_validation_accuracy": best_validation_accuracy,
        "best_validation_distillation_loss": best_validation_distillation_loss,
        "final_validation_accuracy": history[-1]["validation_accuracy"],
        "final_validation_distillation_loss": history[-1]["validation_distillation_loss"],
        "latest_checkpoint_path": str(checkpoint_path),
        "best_checkpoint_path": str(best_checkpoint_path),
        "checkpoint_path": str(checkpoint_path),
        "seconds": round(time.time() - started_at, 3),
    }
    write_json(output_dir / "metrics.json", summary)
    return summary


def build_optimizer(model: nn.Module, config: OptimizerConfig) -> Optimizer:
    """Create the optimizer configured for the student."""

    if config.name == "sgd":
        return SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    if config.name == "adamw":
        return AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {config.name}")
