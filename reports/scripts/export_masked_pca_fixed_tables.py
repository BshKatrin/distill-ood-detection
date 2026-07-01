"""Export corrected masked-PCA metrics tables to LaTeX."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OOD_METRICS_PATH = (
    ROOT / "reports" / "outputs" / "json" / "masked_pca_fixed_ood_metrics.json"
)
DEFAULT_TEST_METRICS_PATH = (
    ROOT / "reports" / "outputs" / "json" / "masked_pca_fixed_test_metrics.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "outputs" / "latex"

ID_DATASETS = ("cifar10_test", "cifar100_test")
ID_DATASET_LABELS = {
    "cifar10_test": "CIFAR-10 (ID)",
    "cifar100_test": "CIFAR-100 (ID)",
}
OOD_DATASETS = {
    "cifar10_test": ("mnist_test", "svhn_test", "cifar100_test"),
    "cifar100_test": ("mnist_test", "svhn_test", "cifar10_test"),
}
OOD_DATASET_LABELS = {
    "mnist_test": "MNIST",
    "svhn_test": "SVHN",
    "cifar100_test": "CIFAR-100",
    "cifar10_test": "CIFAR-10",
}
ARCHITECTURE_ORDER = ("resnet18", "resnet50")
ARCHITECTURE_LABELS = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
}
OBJECTIVE_ORDER = ("cross_entropy", "kl_divergence", "mse_logits")
OBJECTIVE_LABELS = {
    "cross_entropy": "Cross-entropy",
    "kl_divergence": "KL divergence",
    "mse_logits": "Logit MSE",
}
OOD_SCORE_ORDER = (
    "max_probability_difference",
    "absolute_max_probability_difference",
    "student_teacher_kl_divergence",
    "logit_l2_distance",
    "energy_gap",
    "absolute_energy_gap",
    "student_msp",
    "student_energy",
)
OOD_SCORE_LABELS = {
    "max_probability_difference": "Max probability difference",
    "absolute_max_probability_difference": "Absolute max probability difference",
    "student_teacher_kl_divergence": "Student--teacher KL divergence",
    "logit_l2_distance": "Logit L2 distance",
    "energy_gap": "Energy gap",
    "absolute_energy_gap": "Absolute energy gap",
    "student_msp": "Student MSP",
    "student_energy": "Student Energy",
}

RunKey = tuple[str, str]
OodMetricKey = tuple[str, str, str, str, str, str]
TestMetricKey = tuple[str, str, str, str]


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON metrics artifact."""

    with path.open(encoding="utf-8") as file:
        return json.load(file)


def architecture_from_run_name(run_name: str) -> str:
    """Return the ResNet architecture encoded in a run name."""

    for architecture in ARCHITECTURE_ORDER:
        if architecture in run_name:
            return architecture
    raise ValueError(f"Cannot infer architecture from run name: {run_name}")


def id_dataset_from_test_run_name(run_name: str) -> str:
    """Return the ID dataset key encoded in an exported test-metrics run name."""

    if "/cifar_10/" in run_name:
        return "cifar10_test"
    if "/cifar_100/" in run_name:
        return "cifar100_test"
    raise ValueError(f"Cannot infer ID dataset from run name: {run_name}")


def mask_probability_from_run_name(run_name: str) -> str:
    """Return a LaTeX mask-probability label from a masked-PCA run name."""

    match = re.search(r"_mask_p(\d{3})", run_name)
    if match is None:
        return "?"
    digits = match.group(1)
    return f"{int(digits) / 100:.2g}"


def extract_ood_metrics(
    data: dict[str, Any],
) -> tuple[dict[RunKey, str], dict[RunKey, str], dict[OodMetricKey, dict[str, float]]]:
    """Index OOD metrics by architecture, ID dataset, mode, objective, score, and OOD dataset."""

    run_names: dict[RunKey, str] = {}
    mask_probabilities: dict[RunKey, str] = {}
    metrics: dict[OodMetricKey, dict[str, float]] = {}
    for run in data.get("runs", []):
        run_name = str(run["run_name"])
        architecture = architecture_from_run_name(run_name)
        id_dataset = str(run["id_dataset"])
        run_key = (architecture, id_dataset)
        run_names[run_key] = run_name
        mask_probabilities[run_key] = mask_probability_from_run_name(run_name)
        for metric in run.get("metrics", []):
            key = (
                architecture,
                id_dataset,
                str(metric["probability_mode"]),
                str(metric["method"]),
                str(metric["ood_score"]),
                str(metric["ood_dataset"]),
            )
            metrics[key] = {
                "roc_auc": float(metric["roc_auc"]),
                "fpr_at_95_tpr": float(metric["fpr_at_95_tpr"]),
            }
    return run_names, mask_probabilities, metrics


def extract_test_metrics(data: dict[str, Any]) -> dict[TestMetricKey, dict[str, float]]:
    """Index test metrics by architecture, ID dataset, mode, and objective."""

    metrics: dict[TestMetricKey, dict[str, float]] = {}
    for run in data.get("runs", []):
        run_name = str(run["run_name"])
        architecture = architecture_from_run_name(run_name)
        id_dataset = id_dataset_from_test_run_name(run_name)
        for metric in run.get("metrics", []):
            loss = metric.get("test_distillation_loss")
            metrics[
                (
                    architecture,
                    id_dataset,
                    str(metric["probability_mode"]),
                    str(metric["method"]),
                )
            ] = {
                "test_accuracy": float(metric["test_accuracy"]),
                "test_distillation_loss": float(loss) if loss is not None else float("nan"),
            }
    return metrics


def format_architecture_label(architecture: str, mask_probability: str | None) -> str:
    """Format the leading table label for one architecture."""

    suffix = f" ($p={mask_probability}$)" if mask_probability else ""
    return f"{ARCHITECTURE_LABELS[architecture]}{suffix}"


def format_ood_metric(metric: dict[str, float] | None) -> str:
    """Format one OOD metric cell as ROC-AUC/FPR@95."""

    if metric is None:
        return "--"
    return f"{metric['roc_auc']:.2f}/{metric['fpr_at_95_tpr']:.2f}"


def format_test_accuracy(metric: dict[str, float] | None) -> str:
    """Format test accuracy."""

    if metric is None:
        return "--"
    return f"{metric['test_accuracy']:.2f}"


def format_distillation_loss(metric: dict[str, float] | None) -> str:
    """Format test distillation loss."""

    if metric is None:
        return "--"
    return f"{metric['test_distillation_loss']:.3g}"


def result_headers() -> list[str]:
    """Return result headers for both ID datasets."""

    headers: list[str] = []
    for id_dataset in ID_DATASETS:
        headers.extend(["Test acc.", "Distill. loss"])
        headers.extend(OOD_DATASET_LABELS[dataset] for dataset in OOD_DATASETS[id_dataset])
    return headers


def result_cells(
    *,
    ood_metrics: dict[OodMetricKey, dict[str, float]],
    test_metrics: dict[TestMetricKey, dict[str, float]],
    architecture: str,
    mode: str,
    objective: str,
    score: str,
    include_test_metrics: bool,
) -> list[str]:
    """Return ordered metric cells for one table row."""

    cells: list[str] = []
    for id_dataset in ID_DATASETS:
        test_metric = test_metrics.get((architecture, id_dataset, mode, objective))
        if include_test_metrics:
            cells.extend(
                [
                    format_test_accuracy(test_metric),
                    format_distillation_loss(test_metric),
                ]
            )
        else:
            cells.extend(["", ""])
        for ood_dataset in OOD_DATASETS[id_dataset]:
            cells.append(
                format_ood_metric(
                    ood_metrics.get(
                        (
                            architecture,
                            id_dataset,
                            mode,
                            objective,
                            score,
                            ood_dataset,
                        )
                    )
                )
            )
    return cells


def latex_document(title: str, mode: str, rows: list[str]) -> str:
    """Wrap table rows in a standalone LaTeX document."""

    headers = ["Teacher", "Objective", "OOD Score", *result_headers()]
    header = " & ".join(headers) + r" \\"
    body = "\n".join(rows)
    return rf"""\documentclass[border=2pt]{{standalone}}
\usepackage{{booktabs}}

\begin{{document}}

\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{lllcccccccccc}}
\multicolumn{{13}}{{c}}{{\textbf{{{title}}}}}\\[4pt]
\multicolumn{{3}}{{c}}{{Inference: {mode}}} & \multicolumn{{5}}{{c}}{{{ID_DATASET_LABELS["cifar10_test"]}}} & \multicolumn{{5}}{{c}}{{{ID_DATASET_LABELS["cifar100_test"]}}} \\
\cmidrule(lr){{4-8}} \cmidrule(lr){{9-13}}
\toprule
{header}
\midrule
{body}
\bottomrule
\end{{tabular}}

\end{{document}}
"""


def build_table(
    *,
    run_names: dict[RunKey, str],
    mask_probabilities: dict[RunKey, str],
    ood_metrics: dict[OodMetricKey, dict[str, float]],
    test_metrics: dict[TestMetricKey, dict[str, float]],
    mode: str,
) -> str:
    """Build one corrected masked-PCA table for a probability mode."""

    rows: list[str] = []
    for architecture_index, architecture in enumerate(ARCHITECTURE_ORDER):
        if architecture_index:
            rows.append(r"\midrule")
        mask_probability = next(
            (
                mask_probabilities[key]
                for key in run_names
                if key[0] == architecture
            ),
            None,
        )
        for objective_index, objective in enumerate(OBJECTIVE_ORDER):
            if objective_index:
                rows.append(r"\addlinespace")
            for score_index, score in enumerate(OOD_SCORE_ORDER):
                labels = [
                    (
                        format_architecture_label(architecture, mask_probability)
                        if objective_index == 0 and score_index == 0
                        else ""
                    ),
                    OBJECTIVE_LABELS[objective] if score_index == 0 else "",
                    OOD_SCORE_LABELS[score],
                ]
                rows.append(
                    " & ".join(
                        labels
                        + result_cells(
                            ood_metrics=ood_metrics,
                            test_metrics=test_metrics,
                            architecture=architecture,
                            mode=mode,
                            objective=objective,
                            score=score,
                            include_test_metrics=score_index == 0,
                        )
                    )
                    + r" \\"
                )
    return latex_document(
        title="Corrected masked PCA projection --- linear student, layer4",
        mode=mode,
        rows=rows,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ood-metrics", type=Path, default=DEFAULT_OOD_METRICS_PATH)
    parser.add_argument("--test-metrics", type=Path, default=DEFAULT_TEST_METRICS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Write corrected masked-PCA LaTeX tables."""

    args = parse_args()
    run_names, mask_probabilities, ood_metrics = extract_ood_metrics(load_json(args.ood_metrics))
    test_metrics = extract_test_metrics(load_json(args.test_metrics))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for mode in ("unperturbed", "perturbed"):
        output_path = args.output_dir / f"metrics_masked_pca_fixed_{mode}.tex"
        output_path.write_text(
            build_table(
                run_names=run_names,
                mask_probabilities=mask_probabilities,
                ood_metrics=ood_metrics,
                test_metrics=test_metrics,
                mode=mode,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
