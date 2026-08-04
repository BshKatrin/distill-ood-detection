"""Build the layerwise k-NN neighbor-output OOD Score report."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from distill_ood_detection.evaluation.ood_metrics import ood_detection_metrics


ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = ROOT / "runs" / "embedding_distances"
REPORT_PATH = (
    ROOT / "experiments" / "near_ood_detection" / "embedding_distances.md"
)
PLOT_PATH = (
    ROOT
    / "experiments"
    / "near_ood_detection"
    / "embedding_distance_distributions.png"
)
SUMMARY_PATH = (
    ROOT / "reports" / "outputs" / "json" / "embedding_distance_summary.json"
)
BEGIN_MARKER = "<!-- BEGIN GENERATED RESULTS -->"
END_MARKER = "<!-- END GENERATED RESULTS -->"
LAYERS = ("layer1", "layer2", "layer3", "layer4")
SCORE_NAMES = (
    "student_teacher_kl_divergence",
    "absolute_max_probability_difference",
    "absolute_energy_gap",
    "student_msp",
    "student_energy",
    "ensemble_predictive_entropy",
    "ensemble_bald",
)
SPACES = ("raw", "id_standardized")
DATASET_LABELS = {
    "cifar10_test": "CIFAR-10",
    "cifar100_test": "CIFAR-100",
    "mnist_test": "MNIST",
    "svhn_test": "SVHN",
}
CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
CIFAR100_CLASSES = (
    "apple", "aquarium fish", "baby", "bear", "beaver", "bed", "bee",
    "beetle", "bicycle", "bottle", "bowl", "boy", "bridge", "bus",
    "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab",
    "crocodile", "cup", "dinosaur", "dolphin", "elephant", "flatfish",
    "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn mower", "leopard", "lion", "lizard", "lobster", "man",
    "maple tree", "motorcycle", "mountain", "mouse", "mushroom", "oak tree",
    "orange", "orchid", "otter", "palm tree", "pear", "pickup truck",
    "pine tree", "plain", "plate", "poppy", "porcupine", "possum", "rabbit",
    "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark",
    "shrew", "skunk", "skyscraper", "snail", "snake", "spider", "squirrel",
    "streetcar", "sunflower", "sweet pepper", "table", "tank", "telephone",
    "television", "tiger", "tractor", "train", "trout", "tulip", "turtle",
    "wardrobe", "whale", "willow tree", "wolf", "woman", "worm",
)


@dataclass(frozen=True)
class RunSpec:
    """One ID dataset and teacher combination."""

    id_dir: str
    id_dataset: str
    near_dataset: str
    model: str
    title: str

    @property
    def run_dir(self) -> Path:
        """Return the local distance-artifact directory."""

        return RUNS_ROOT / self.id_dir / self.model / "distances"

    @property
    def ood_datasets(self) -> tuple[str, str, str]:
        """Return Near-OOD followed by the two Far-OOD datasets."""

        return (self.near_dataset, "mnist_test", "svhn_test")


RUNS = (
    RunSpec("cifar_10", "cifar10_test", "cifar100_test", "resnet18", "CIFAR-10 ID, ResNet-18"),
    RunSpec("cifar_10", "cifar10_test", "cifar100_test", "resnet50", "CIFAR-10 ID, ResNet-50"),
    RunSpec("cifar_100", "cifar100_test", "cifar10_test", "resnet18", "CIFAR-100 ID, ResNet-18"),
    RunSpec("cifar_100", "cifar100_test", "cifar10_test", "resnet50", "CIFAR-100 ID, ResNet-50"),
)


def load_artifact(spec: RunSpec, dataset: str, layer: str) -> dict[str, Any]:
    """Load one synced distance artifact."""

    path = spec.run_dir / dataset / "test" / f"{layer}.pt"
    return torch.load(path, map_location="cpu", weights_only=False)


def values(
    spec: RunSpec,
    dataset: str,
    layer: str,
    space: str = "id_standardized",
    score_name: str = "student_teacher_kl_divergence",
) -> np.ndarray:
    """Load one per-sample sign-adjusted OOD Score vector as NumPy."""

    tensor = load_artifact(spec, dataset, layer)["distances"][space]["ood_scores"][
        score_name
    ]
    return tensor.numpy()


def metric_pair(id_scores: np.ndarray, ood_scores: np.ndarray) -> tuple[float, float]:
    """Return ROC-AUC and FPR@95 from ID-oriented OOD Scores."""

    labels = np.concatenate((np.ones(id_scores.size), np.zeros(ood_scores.size)))
    scores = np.concatenate((id_scores, ood_scores))
    metrics = ood_detection_metrics(labels, scores)
    return metrics["roc_auc"], metrics["fpr_at_95_tpr"]


def format_pair(pair: tuple[float, float]) -> str:
    """Format one ROC-AUC/FPR@95 pair."""

    return f"{pair[0]:.3f} / {pair[1]:.3f}"


def add_table(
    lines: list[str],
    headers: tuple[str, ...],
    rows: list[tuple[str, ...]],
) -> None:
    """Append a compact Markdown table."""

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" if index == 0 else "---:" for index in range(len(headers))) + "|")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    lines.append("")


def descriptive_statistics(data: np.ndarray) -> dict[str, float | int]:
    """Return the report's descriptive statistics for one distance vector."""

    percentiles = np.percentile(data, (5, 25, 50, 75, 95))
    return {
        "count": int(data.size),
        "mean": float(np.mean(data)),
        "standard_deviation": float(np.std(data)),
        "p05": float(percentiles[0]),
        "p25": float(percentiles[1]),
        "median": float(percentiles[2]),
        "p75": float(percentiles[3]),
        "p95": float(percentiles[4]),
    }


def export_numeric_summary() -> None:
    """Export complete descriptive statistics and OOD metrics as JSON."""

    records: list[dict[str, Any]] = []
    for spec in RUNS:
        for layer in LAYERS:
            for space in SPACES:
                for score_name in SCORE_NAMES:
                    id_scores = values(spec, spec.id_dataset, layer, space, score_name)
                    for dataset in (spec.id_dataset, *spec.ood_datasets):
                        scores = values(spec, dataset, layer, space, score_name)
                        record: dict[str, Any] = {
                            "id_dataset": spec.id_dataset,
                            "teacher": spec.model,
                            "dataset": dataset,
                            "layer": layer,
                            "space": space,
                            "score_name": score_name,
                            **descriptive_statistics(scores),
                        }
                        if dataset != spec.id_dataset:
                            roc_auc, fpr = metric_pair(id_scores, scores)
                            record.update({"roc_auc": roc_auc, "fpr_at_95_tpr": fpr})
                        records.append(record)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        json.dumps({"records": records}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def bootstrap_penalty(
    id_distances: np.ndarray,
    near_distances: np.ndarray,
    far_distances: np.ndarray,
    seed: int,
    draws: int = 500,
) -> tuple[float, float, float, float, float, float]:
    """Return Near-OOD penalties and percentile intervals against one Far-OOD set."""

    near_pair = metric_pair(id_distances, near_distances)
    far_pair = metric_pair(id_distances, far_distances)
    auc_penalty = far_pair[0] - near_pair[0]
    fpr_penalty = near_pair[1] - far_pair[1]
    rng = np.random.default_rng(seed)
    auc_samples = np.empty(draws)
    fpr_samples = np.empty(draws)
    for draw in range(draws):
        sampled_id = id_distances[rng.integers(id_distances.size, size=id_distances.size)]
        sampled_near = near_distances[
            rng.integers(near_distances.size, size=near_distances.size)
        ]
        sampled_far = far_distances[
            rng.integers(far_distances.size, size=far_distances.size)
        ]
        sampled_near_pair = metric_pair(sampled_id, sampled_near)
        sampled_far_pair = metric_pair(sampled_id, sampled_far)
        auc_samples[draw] = sampled_far_pair[0] - sampled_near_pair[0]
        fpr_samples[draw] = sampled_near_pair[1] - sampled_far_pair[1]
    auc_interval = np.percentile(auc_samples, (2.5, 97.5))
    fpr_interval = np.percentile(fpr_samples, (2.5, 97.5))
    return (
        auc_penalty,
        float(auc_interval[0]),
        float(auc_interval[1]),
        fpr_penalty,
        float(fpr_interval[0]),
        float(fpr_interval[1]),
    )


def class_name(spec: RunSpec, label: int) -> str:
    """Return a human-readable ID class name."""

    classes = CIFAR10_CLASSES if spec.id_dataset == "cifar10_test" else CIFAR100_CLASSES
    return classes[label]


def top_nearest_classes(spec: RunSpec, count: int = 5) -> str:
    """Return the most frequent standardized layer4 nearest-neighbor ID classes."""

    artifact = load_artifact(spec, spec.near_dataset, "layer4")
    labels = artifact["distances"]["id_standardized"]["nearest_neighbor_label"]
    frequencies = torch.bincount(labels.to(torch.long))
    top_counts, top_labels = frequencies.topk(min(count, frequencies.numel()))
    total = labels.numel()
    return ", ".join(
        f"{class_name(spec, int(label))} ({100.0 * int(value) / total:.1f}%)"
        for value, label in zip(top_counts, top_labels, strict=True)
    )


def build_distribution_plot() -> None:
    """Plot standardized-space KL-divergence OOD Score distributions."""

    sns.set_theme(style="whitegrid", context="notebook")
    figure, axes = plt.subplots(4, 4, figsize=(16, 13), sharey=False)
    rng = np.random.default_rng(42)
    for row, spec in enumerate(RUNS):
        for column, layer in enumerate(LAYERS):
            axis = axes[row, column]
            datasets = (spec.id_dataset, spec.near_dataset, "mnist_test", "svhn_test")
            labels = ("ID", "Near", "MNIST", "SVHN")
            plot_values: list[np.ndarray] = []
            plot_labels: list[np.ndarray] = []
            for dataset, label in zip(datasets, labels, strict=True):
                scores = values(spec, dataset, layer)
                if scores.size > 2000:
                    scores = scores[
                        rng.choice(scores.size, size=2000, replace=False)
                    ]
                plot_values.append(scores)
                plot_labels.append(np.repeat(label, scores.size))
            sns.boxplot(
                x=np.concatenate(plot_labels),
                y=np.concatenate(plot_values),
                order=labels,
                showfliers=False,
                width=0.65,
                ax=axis,
            )
            axis.set_title(layer)
            axis.set_xlabel("")
            axis.set_ylabel("negative KL divergence" if column == 0 else "")
            if column == 0:
                axis.text(
                    -0.34,
                    0.5,
                    spec.title,
                    rotation=90,
                    va="center",
                    ha="center",
                    transform=axis.transAxes,
                    fontsize=10,
                )
    figure.suptitle("k-NN neighbor-mean KL OOD Score", fontsize=15)
    figure.tight_layout(rect=(0.03, 0.0, 1.0, 0.98))
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(PLOT_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)


def generated_markdown() -> str:
    """Return the generated results section."""

    lines = [
        BEGIN_MARKER,
        "## Results",
        "",
        "Each result uses the complete ID test and OOD test sets against the complete",
        "ID training reference. Embedding distance selects the exact top-10 neighbors;",
        "their mean probabilities and logits produce the OOD Scores. Metric cells are",
        "`ROC-AUC / FPR@95`; higher ROC-AUC and lower FPR@95 are better. All score",
        "vectors are sign-adjusted so higher means more ID-like.",
        "",
        "![Layerwise neighbor-output score distributions](embedding_distance_distributions.png)",
        "",
        "The figure shows the sign-adjusted KL score using neighbors",
        "selected in the ID-standardized embedding space.",
        "",
        "### Best Near-OOD Layer by Score",
        "",
    ]
    for spec in RUNS:
        lines.extend((f"#### {spec.title}", ""))
        rows: list[tuple[str, ...]] = []
        for score_name in SCORE_NAMES:
            layer_results = []
            for layer in LAYERS:
                id_scores = values(spec, spec.id_dataset, layer, score_name=score_name)
                near_scores = values(spec, spec.near_dataset, layer, score_name=score_name)
                layer_results.append((layer, metric_pair(id_scores, near_scores)))
            layer, near_pair = max(layer_results, key=lambda result: result[1][0])
            id_scores = values(spec, spec.id_dataset, layer, score_name=score_name)
            rows.append(
                (
                    score_name,
                    layer,
                    format_pair(near_pair),
                    format_pair(
                        metric_pair(
                            id_scores,
                            values(spec, "mnist_test", layer, score_name=score_name),
                        )
                    ),
                    format_pair(
                        metric_pair(
                            id_scores,
                            values(spec, "svhn_test", layer, score_name=score_name),
                        )
                    ),
                )
            )
        add_table(
            lines,
            (
                "OOD Score",
                "Best layer",
                f"Near: {DATASET_LABELS[spec.near_dataset]}",
                "Far: MNIST",
                "Far: SVHN",
            ),
            rows,
        )

    lines.extend(
        (
            "### Raw Versus ID-Standardized Neighbor Selection",
            "",
            "This comparison uses the KL score at layer4. Only neighbor",
            "selection changes; query and reference classifier outputs are unchanged.",
            "",
        )
    )
    rows = []
    for spec in RUNS:
        cells = []
        for space in ("raw", "id_standardized"):
            id_scores = values(spec, spec.id_dataset, "layer4", space)
            near_scores = values(spec, spec.near_dataset, "layer4", space)
            cells.append(format_pair(metric_pair(id_scores, near_scores)))
        rows.append((spec.title, *cells))
    add_table(lines, ("Setting", "Raw", "ID-standardized"), rows)

    lines.extend(
        (
            "### Near-OOD Nearest ID Classes at Layer4",
            "",
            "The five most frequent nearest-neighbor ID training labels provide a",
            "traceability check on the examples used to form the output means.",
            "",
        )
    )
    add_table(
        lines,
        ("Setting", "Most frequent nearest ID classes"),
        [(spec.title, top_nearest_classes(spec)) for spec in RUNS],
    )

    lines.extend(
        (
            "Complete descriptive statistics for every layer, dataset, embedding space,",
            "and classifier-based OOD Score are generated at",
            "`reports/outputs/json/embedding_distance_summary.json`.",
            END_MARKER,
            "",
        )
    )
    return "\n".join(lines)


def update_report() -> None:
    """Insert or replace the generated results block in the experiment report."""

    report = REPORT_PATH.read_text(encoding="utf-8")
    generated = generated_markdown()
    if BEGIN_MARKER in report:
        prefix, remainder = report.split(BEGIN_MARKER, maxsplit=1)
        _, suffix = remainder.split(END_MARKER, maxsplit=1)
        report = prefix + generated + suffix.lstrip("\n")
    else:
        insertion = "## Submitted Runs"
        report = report.replace(insertion, generated + "\n" + insertion)
    report = report.replace(
        "Status: implemented; cluster runs submitted on 2026-07-22.",
        "Status: completed on 2026-07-22.",
    )
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    """Generate numeric, visual, and Markdown embedding-distance reports."""

    export_numeric_summary()
    build_distribution_plot()
    update_report()


if __name__ == "__main__":
    main()
