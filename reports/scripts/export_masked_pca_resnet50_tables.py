"""Export ResNet-50 masked-PCA metrics tables to LaTeX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OOD_METRICS_PATH = (
    ROOT / "reports" / "outputs" / "json" / "masked_pca_resnet50_ood_metrics.json"
)
DEFAULT_TEST_METRICS_PATH = (
    ROOT / "reports" / "outputs" / "json" / "masked_pca_resnet50_test_metrics.json"
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
RUN_BY_ID_DATASET = {
    "cifar10_test": "perturbation_linear_layer4_student_resnet50_cifar10_pca17_mask_p050",
    "cifar100_test": "perturbation_linear_layer4_student_resnet50_cifar100_pca100_mask_p050",
}

OodMetricKey = tuple[str, str, str, str, str]
TestMetricKey = tuple[str, str, str]


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON metrics file."""

    with path.open(encoding="utf-8") as file:
        return json.load(file)


def extract_ood_metrics(data: dict[str, Any]) -> dict[OodMetricKey, dict[str, float]]:
    """Index OOD metrics by run, mode, objective, score, and OOD dataset."""

    metrics: dict[OodMetricKey, dict[str, float]] = {}
    for run in data.get("runs", []):
        run_name = str(run["run_name"])
        for metric in run.get("metrics", []):
            key = (
                run_name,
                str(metric["probability_mode"]),
                str(metric["method"]),
                str(metric["ood_score"]),
                str(metric["ood_dataset"]),
            )
            metrics[key] = {
                "roc_auc": float(metric["roc_auc"]),
                "fpr_at_95_tpr": float(metric["fpr_at_95_tpr"]),
            }
    return metrics


def extract_test_metrics(data: dict[str, Any]) -> dict[TestMetricKey, dict[str, float]]:
    """Index test metrics by ID dataset, mode, and objective."""

    metrics: dict[TestMetricKey, dict[str, float]] = {}
    for run in data.get("runs", []):
        run_name = str(run["run_name"])
        if "/cifar_10/" in run_name:
            id_dataset = "cifar10_test"
        elif "/cifar_100/" in run_name:
            id_dataset = "cifar100_test"
        else:
            continue
        for metric in run.get("metrics", []):
            loss = metric.get("test_distillation_loss")
            metrics[
                (
                    id_dataset,
                    str(metric["probability_mode"]),
                    str(metric["method"]),
                )
            ] = {
                "test_accuracy": float(metric["test_accuracy"]),
                "test_distillation_loss": float(loss) if loss is not None else float("nan"),
            }
    return metrics


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
    mode: str,
    objective: str,
    score: str,
    include_test_metrics: bool,
) -> list[str]:
    """Return ordered metric cells for one table row."""

    cells: list[str] = []
    for id_dataset in ID_DATASETS:
        run_name = RUN_BY_ID_DATASET[id_dataset]
        test_metric = test_metrics.get((id_dataset, mode, objective))
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
                    ood_metrics.get((run_name, mode, objective, score, ood_dataset))
                )
            )
    return cells


def latex_document(title: str, mode: str, rows: list[str]) -> str:
    """Wrap table rows in a standalone LaTeX document."""

    headers = ["Objective", "OOD Score", *result_headers()]
    header = " & ".join(headers) + r" \\"
    body = "\n".join(rows)
    return rf"""\documentclass[border=2pt]{{standalone}}
\usepackage{{booktabs}}

\begin{{document}}

\scriptsize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{llcccccccccc}}
\multicolumn{{12}}{{c}}{{\textbf{{{title}}}}}\\[4pt]
\multicolumn{{2}}{{c}}{{Inference: {mode}}} & \multicolumn{{5}}{{c}}{{{ID_DATASET_LABELS["cifar10_test"]}}} & \multicolumn{{5}}{{c}}{{{ID_DATASET_LABELS["cifar100_test"]}}} \\
\cmidrule(lr){{3-7}} \cmidrule(lr){{8-12}}
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
    ood_metrics: dict[OodMetricKey, dict[str, float]],
    test_metrics: dict[TestMetricKey, dict[str, float]],
    mode: str,
) -> str:
    """Build one masked-PCA table for a probability mode."""

    rows: list[str] = []
    for objective_index, objective in enumerate(OBJECTIVE_ORDER):
        if objective_index:
            rows.append(r"\addlinespace")
        for score_index, score in enumerate(OOD_SCORE_ORDER):
            labels = [
                OBJECTIVE_LABELS[objective] if score_index == 0 else "",
                OOD_SCORE_LABELS[score],
            ]
            rows.append(
                " & ".join(
                    labels
                    + result_cells(
                        ood_metrics=ood_metrics,
                        test_metrics=test_metrics,
                        mode=mode,
                        objective=objective,
                        score=score,
                        include_test_metrics=score_index == 0,
                    )
                )
                + r" \\"
            )
    return latex_document(
        title="Masked PCA projection $(p=0.5)$ --- linear student, ResNet-50 layer4",
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
    """Write masked-PCA ResNet-50 LaTeX tables."""

    args = parse_args()
    ood_metrics = extract_ood_metrics(load_json(args.ood_metrics))
    test_metrics = extract_test_metrics(load_json(args.test_metrics))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for mode in ("unperturbed", "perturbed"):
        output_path = args.output_dir / f"metrics_masked_pca_resnet50_{mode}.tex"
        output_path.write_text(
            build_table(
                ood_metrics=ood_metrics,
                test_metrics=test_metrics,
                mode=mode,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
