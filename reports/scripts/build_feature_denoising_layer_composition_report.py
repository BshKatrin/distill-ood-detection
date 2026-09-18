"""Evaluate uniform-channel layer3/layer4 Feature Denoising composition."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from distill_ood_detection.evaluation.ood_metrics import ood_detection_metrics
from distill_ood_detection.evaluation.score_composition import (
    compose_standardized_scores,
    fit_score_standardization,
    standardize_score,
)


ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = ROOT / "runs" / "students"
REPORT_PATH = (
    ROOT
    / "experiments"
    / "feature_denoising"
    / "uniform_channel_layer3_layer4_absolute_improvement_composition.md"
)
SUMMARY_PATH = (
    ROOT
    / "reports"
    / "outputs"
    / "json"
    / "feature_denoising_uniform_channel_layer3_layer4_composition.json"
)
BETAS = (-100.0, -10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0, 100.0)


@dataclass(frozen=True)
class ExperimentSpec:
    """Layer-pair experiment definition for one ID dataset."""

    teacher: str
    id_dataset: str
    id_label: str
    near_dataset: str
    near_label: str
    far_datasets: tuple[str, str]
    far_labels: tuple[str, str]

    @property
    def datasets(self) -> tuple[str, str, str]:
        """Return Near-OOD followed by the two Far-OOD artifact names."""

        return (self.near_dataset, *self.far_datasets)

    @property
    def dataset_labels(self) -> dict[str, str]:
        """Return display labels for all OOD datasets."""

        return {
            self.near_dataset: self.near_label,
            self.far_datasets[0]: self.far_labels[0],
            self.far_datasets[1]: self.far_labels[1],
        }


SPECS = (
    ExperimentSpec(
        teacher="resnet18",
        id_dataset="cifar10_test",
        id_label="CIFAR-10",
        near_dataset="cifar100_test",
        near_label="CIFAR-100",
        far_datasets=("mnist_test", "svhn_test"),
        far_labels=("MNIST", "SVHN"),
    ),
    ExperimentSpec(
        teacher="resnet18",
        id_dataset="cifar100_test",
        id_label="CIFAR-100",
        near_dataset="cifar10_test",
        near_label="CIFAR-10",
        far_datasets=("mnist_test", "svhn_test"),
        far_labels=("MNIST", "SVHN"),
    ),
    ExperimentSpec(
        teacher="resnet50",
        id_dataset="cifar10_test",
        id_label="CIFAR-10",
        near_dataset="cifar100_test",
        near_label="CIFAR-100",
        far_datasets=("mnist_test", "svhn_test"),
        far_labels=("MNIST", "SVHN"),
    ),
    ExperimentSpec(
        teacher="resnet50",
        id_dataset="cifar100_test",
        id_label="CIFAR-100",
        near_dataset="cifar10_test",
        near_label="CIFAR-10",
        far_datasets=("mnist_test", "svhn_test"),
        far_labels=("MNIST", "SVHN"),
    ),
)


def main() -> None:
    """Build JSON and the Markdown experiment report."""

    results = [evaluate_spec(spec) for spec in SPECS]
    write_json(SUMMARY_PATH, {"version": 1, "betas": BETAS, "results": results})
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_markdown(results), encoding="utf-8")


def evaluate_spec(spec: ExperimentSpec) -> dict[str, Any]:
    """Evaluate both single layers and every beta for one ID dataset."""

    layer_scores: dict[str, dict[str, np.ndarray]] = {"layer3": {}, "layer4": {}}
    reference_labels: dict[str, np.ndarray] = {}
    for layer in layer_scores:
        for dataset in (spec.id_dataset, *spec.datasets):
            scores, labels = load_improvement(spec, layer, dataset)
            layer_scores[layer][dataset] = scores
            if dataset in reference_labels:
                if not np.array_equal(reference_labels[dataset], labels):
                    raise ValueError(
                        f"Layer label order differs for {spec.id_label} / {dataset}"
                    )
            else:
                reference_labels[dataset] = labels

    statistics = {
        layer: fit_score_standardization(scores[spec.id_dataset])
        for layer, scores in layer_scores.items()
    }
    standardized = {
        layer: {
            dataset: standardize_score(values, statistics[layer])
            for dataset, values in scores.items()
        }
        for layer, scores in layer_scores.items()
    }

    rows = []
    for layer in ("layer3", "layer4"):
        rows.append(
            metric_row(
                name=layer,
                beta=None,
                id_scores=standardized[layer][spec.id_dataset],
                ood_scores={
                    dataset: standardized[layer][dataset]
                    for dataset in spec.datasets
                },
                spec=spec,
            )
        )
    for beta in BETAS:
        combined = {
            dataset: compose_standardized_scores(
                standardized["layer3"][dataset],
                standardized["layer4"][dataset],
                beta,
            )
            for dataset in (spec.id_dataset, *spec.datasets)
        }
        rows.append(
            metric_row(
                name=f"beta={format_beta(beta)}",
                beta=beta,
                id_scores=combined[spec.id_dataset],
                ood_scores={dataset: combined[dataset] for dataset in spec.datasets},
                spec=spec,
            )
        )

    return {
        "teacher": spec.teacher,
        "id_dataset": spec.id_dataset,
        "id_label": spec.id_label,
        "near_dataset": spec.near_dataset,
        "near_label": spec.near_label,
        "far_datasets": spec.far_datasets,
        "far_labels": spec.far_labels,
        "standardization_split": spec.id_dataset,
        "standardization": {
            layer: asdict(layer_statistics)
            for layer, layer_statistics in statistics.items()
        },
        "artifact_directories": {
            layer: str(score_directory(spec, layer).relative_to(ROOT))
            for layer in ("layer3", "layer4")
        },
        "rows": rows,
    }


def metric_row(
    *,
    name: str,
    beta: float | None,
    id_scores: np.ndarray,
    ood_scores: dict[str, np.ndarray],
    spec: ExperimentSpec,
) -> dict[str, Any]:
    """Compute per-dataset and macro metrics for one score vector family."""

    metrics = {
        dataset: metric_pair(id_scores, scores)
        for dataset, scores in ood_scores.items()
    }
    near = metrics[spec.near_dataset]
    far = average_metrics([metrics[dataset] for dataset in spec.far_datasets])
    macro = average_metrics(list(metrics.values()))
    return {
        "name": name,
        "beta": beta,
        "metrics": metrics,
        "near_macro": near,
        "far_macro": far,
        "macro": macro,
    }


def metric_pair(id_scores: np.ndarray, ood_scores: np.ndarray) -> dict[str, float]:
    """Compute project OOD metrics from ID-oriented score vectors."""

    labels = np.concatenate(
        (np.ones(id_scores.size, dtype=int), np.zeros(ood_scores.size, dtype=int))
    )
    return ood_detection_metrics(labels, np.concatenate((id_scores, ood_scores)))


def average_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    """Average ROC-AUC and FPR@95 independently."""

    return {
        key: float(np.mean([values[key] for values in metrics]))
        for key in ("roc_auc", "fpr_at_95_tpr")
    }


def load_improvement(
    spec: ExperimentSpec,
    layer: str,
    dataset: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one absolute-improvement vector and its labels."""

    path = score_directory(spec, layer) / dataset / "student_best.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing Feature Denoising score artifact: {path}")
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if "improvement" not in artifact:
        raise KeyError(f"Artifact does not contain improvement: {path}")
    scores = np.asarray(artifact["improvement"], dtype=np.float64)
    labels = np.asarray(artifact["labels"])
    if scores.ndim != 1 or labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError(f"Invalid score/label shapes in {path}")
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"Non-finite improvement scores in {path}")
    return scores, labels


def score_directory(spec: ExperimentSpec, layer: str) -> Path:
    """Return one uniform 20% channel-masking score directory."""

    id_path = "cifar_10" if spec.id_dataset == "cifar10_test" else "cifar_100"
    return (
        RUNS_ROOT
        / "feature_denoising"
        / "feature_masking"
        / id_path
        / spec.teacher
        / f"channel_residual_{layer}_mask_p020"
        / "feature_denoising_scores"
    )


def build_markdown(results: list[dict[str, Any]]) -> str:
    """Render the complete experiment report."""

    lines = [
        "# Uniform Channel-Masking Layer3 + Layer4 Composition",
        "",
        "Status: completed.",
        "",
        "## Objective",
        "",
        "Compose the `layer3` and `layer4` absolute-improvement scores from "
        "Feature Denoising residual CNN students trained with uniform 20% "
        "channel masking. ResNet-18 and ResNet-50 teachers are evaluated "
        "independently.",
        "",
        "For each layer, the per-sample absolute-improvement score is:",
        "",
        "```text",
        "improvement = identity_error - raw_reconstruction_error",
        "```",
        "",
        "Each score is standardized using its own ID test-split population mean "
        "and standard deviation. The same statistics are then reused for all OOD "
        "datasets. Standardized layer scores are composed using the HEAT operator "
        "from [Lafon et al. (2023)](https://arxiv.org/pdf/2305.16966):",
        "",
        "```text",
        "combined_beta = log(exp(beta * z_layer3) + exp(beta * z_layer4)) / beta",
        "combined_0    = (z_layer3 + z_layer4) / 2",
        "```",
        "",
        "The nonzero implementation uses a numerically stable log-sum-exp. Higher "
        "scores remain more ID-like.",
        "",
        "## Setup",
        "",
        "- Teachers: ResNet-18 and ResNet-50 trained on the corresponding ID dataset.",
        "- Students: independently trained layer3 and layer4 residual CNN denoisers.",
        "- Corruption: each channel is hidden independently with probability 0.2.",
        "- Score: absolute improvement (not relative improvement and not an "
        "absolute-value transform).",
        "- Betas: `-100, -10, -1, -0.1, 0, 0.1, 1, 10, 100`.",
        "- Metrics: ROC-AUC and FPR@95; macro values average the three OOD datasets.",
        "",
    ]
    for teacher in ("resnet18", "resnet50"):
        lines.extend([f"## {teacher.replace('resnet', 'ResNet-')}", ""])
        for result in results:
            if result["teacher"] == teacher:
                lines.extend(markdown_result(result))
    lines.extend(markdown_conclusions(results))
    return "\n".join(lines) + "\n"


def markdown_result(result: dict[str, Any]) -> list[str]:
    """Render one ID-dataset result section."""

    layer3 = result["standardization"]["layer3"]
    layer4 = result["standardization"]["layer4"]
    dataset_order = (
        result["near_dataset"],
        result["far_datasets"][0],
        result["far_datasets"][1],
    )
    labels = {
        result["near_dataset"]: result["near_label"],
        result["far_datasets"][0]: result["far_labels"][0],
        result["far_datasets"][1]: result["far_labels"][1],
    }
    lines = [
        f"### {result['id_label']} ID",
        "",
        "ID test standardization statistics:",
        "",
        "| Layer | Mean | Population std |",
        "|---|---:|---:|",
        f"| layer3 | {layer3['mean']:.8g} | {layer3['standard_deviation']:.8g} |",
        f"| layer4 | {layer4['mean']:.8g} | {layer4['standard_deviation']:.8g} |",
        "",
        "OOD metrics (`ROC-AUC / FPR@95`):",
        "",
        "| Score | " + " | ".join(labels[dataset] for dataset in dataset_order) + " | Far macro | Overall macro |",
        "|---|" + "---:|" * 5,
    ]
    for row in result["rows"]:
        cells = [format_pair(row["metrics"][dataset]) for dataset in dataset_order]
        lines.append(
            f"| {row['name']} | "
            + " | ".join(cells)
            + f" | {format_pair(row['far_macro'])} | {format_pair(row['macro'])} |"
        )
    lines.append("")
    return lines


def markdown_conclusions(results: list[dict[str, Any]]) -> list[str]:
    """Render data-derived best-beta conclusions."""

    lines = ["## Results Summary", ""]
    for result in results:
        beta_rows = [row for row in result["rows"] if row["beta"] is not None]
        best_auc = max(beta_rows, key=lambda row: row["macro"]["roc_auc"])
        best_fpr = min(beta_rows, key=lambda row: row["macro"]["fpr_at_95_tpr"])
        best_near_auc = max(beta_rows, key=lambda row: row["near_macro"]["roc_auc"])
        best_far_auc = max(beta_rows, key=lambda row: row["far_macro"]["roc_auc"])
        lines.extend(
            [
                f"- **{result['teacher'].replace('resnet', 'ResNet-')}, "
                f"{result['id_label']} ID**: best macro ROC-AUC uses "
                f"`beta={format_beta(best_auc['beta'])}` "
                f"({best_auc['macro']['roc_auc']:.3f}); best macro FPR@95 uses "
                f"`beta={format_beta(best_fpr['beta'])}` "
                f"({best_fpr['macro']['fpr_at_95_tpr']:.3f}). Best Near-OOD "
                f"ROC-AUC is at `beta={format_beta(best_near_auc['beta'])}`, while "
                f"best Far-OOD macro ROC-AUC is at "
                f"`beta={format_beta(best_far_auc['beta'])}`.",
            ]
        )
    lines.extend(
        [
            "",
            "Complete machine-readable metrics and standardization statistics are "
            "stored in "
            "`reports/outputs/json/feature_denoising_uniform_channel_layer3_"
            "layer4_composition.json`.",
        ]
    )
    return lines


def format_pair(metrics: dict[str, float]) -> str:
    """Format one ROC-AUC/FPR@95 pair."""

    return f"{metrics['roc_auc']:.3f} / {metrics['fpr_at_95_tpr']:.3f}"


def format_beta(beta: float) -> str:
    """Format beta without an unnecessary decimal suffix."""

    return f"{beta:g}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a deterministic JSON summary."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
