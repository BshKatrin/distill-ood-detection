"""Tests for combined distillation evaluation metrics."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from distill_ood_detection.evaluation.metrics import distillation_metrics


def test_distillation_metrics_names_results_for_requested_split() -> None:
    """Accuracy and loss should share one split-specific metric namespace."""

    images = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
    labels = torch.tensor([0, 1])
    loader = DataLoader(TensorDataset(images, labels), batch_size=1)
    teacher = nn.Identity()
    student = nn.Identity()

    metrics = distillation_metrics(
        method="kl_divergence",
        teacher=teacher,
        student=student,
        loader=loader,
        device=torch.device("cpu"),
        split="test",
    )

    assert metrics["test_accuracy"] == 1.0
    assert metrics["test_distillation_loss"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["test_kl_divergence"] == pytest.approx(0.0, abs=1e-7)
    assert not any(name.startswith("validation_") for name in metrics)


def test_report_formatter_uses_test_metrics_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report cells must never fall back to validation results."""

    script_path = (
        Path(__file__).parents[1]
        / "reports"
        / "scripts"
        / "export_test_metrics_table.py"
    )
    spec = importlib.util.spec_from_file_location("export_test_metrics_table", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    format_test_metric = module.format_test_metric

    assert format_test_metric(
        {
            "test_accuracy": 0.875,
            "test_distillation_loss": 0.125,
            "best_validation_accuracy": 1.0,
            "best_validation_distillation_loss": 0.001,
        }
    ) == "0.8750/0.1250"
    assert format_test_metric(
        {
            "best_validation_accuracy": 1.0,
            "best_validation_distillation_loss": 0.001,
        }
    ) == ""

    assert module.teacher_test_metric("cifar10") == "0.9498/--"
    assert module.teacher_test_metric("cifar100") == "0.7926/--"

    method_dir = tmp_path / "runs" / "example_run" / "mse_logits"
    method_dir.mkdir(parents=True)
    (method_dir / "metrics.json").write_text(
        json.dumps(
            {
                "test_accuracy": 0.75,
                "test_distillation_loss": 0.25,
                "best_validation_accuracy": 1.0,
                "best_validation_distillation_loss": 0.001,
            }
        )
    )
    monkeypatch.setattr(module, "ROOT", tmp_path)
    output_path = tmp_path / "test_metrics.json"
    module.export_run_metrics(["example_run"], output_path)

    assert json.loads(output_path.read_text()) == {
        "version": 1,
        "runs": [
            {
                "run_name": "example_run",
                "metrics": [
                    {
                        "method": "mse_logits",
                        "test_accuracy": 0.75,
                        "test_distillation_loss": 0.25,
                    }
                ],
            }
        ],
    }
