"""Export ResNet-50 pixel-augmentation metrics tables to LaTeX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OOD_METRICS_PATH = (
    ROOT / "reports" / "outputs" / "json" / "pixel_augmentation_resnet50_ood_metrics.json"
)
DEFAULT_TEST_METRICS_PATH = (
    ROOT / "reports" / "outputs" / "json" / "pixel_augmentation_resnet50_test_metrics.json"
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
TARGET_ORDER = ("perturbed", "clean")
TARGET_LABELS = {
    "perturbed": "Perturbed",
    "clean": "Clean",
}
OBJECTIVE_ORDER = ("cross_entropy", "mse_logits")
OBJECTIVE_LABELS = {
    "cross_entropy": "Cross-entropy",
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
EXCLUDED_OOD_SCORES = {
    "max_probability_difference",
    "logit_l2_distance",
    "energy_gap",
    "absolute_energy_gap",
}
DISPLAY_OOD_SCORE_ORDER = tuple(
    score for score in OOD_SCORE_ORDER if score not in EXCLUDED_OOD_SCORES
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
PROMISING_MIN_ROC_AUC = 0.85
PROMISING_MAX_AVG_FPR_AT_95_TPR = 0.60

OodMetricKey = tuple[str, str, str, str, str]
TestMetricKey = tuple[str, str, str, str]
RunKey = tuple[str, str]


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON metrics file."""

    with path.open(encoding="utf-8") as file:
        return json.load(file)


def target_from_run_name(run_name: str) -> str:
    """Return teacher target encoded in a pixel-augmentation run name."""

    return "clean" if "clean_target" in run_name else "perturbed"


def id_dataset_from_test_run_name(run_name: str) -> str:
    """Return the ID dataset key encoded in an exported test-metrics run name."""

    if "/cifar_10/" in run_name:
        return "cifar10_test"
    if "/cifar_100/" in run_name:
        return "cifar100_test"
    raise ValueError(f"Cannot infer ID dataset from run name: {run_name}")


def extract_ood_metrics(
    data: dict[str, Any],
) -> tuple[dict[RunKey, str], dict[OodMetricKey, dict[str, float]]]:
    """Index OOD metrics by run, mode, objective, score, and OOD dataset."""

    run_names: dict[RunKey, str] = {}
    metrics: dict[OodMetricKey, dict[str, float]] = {}
    for run in data.get("runs", []):
        run_name = str(run["run_name"])
        id_dataset = str(run["id_dataset"])
        target = target_from_run_name(run_name)
        run_names[(id_dataset, target)] = run_name
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
    return run_names, metrics


def extract_test_metrics(data: dict[str, Any]) -> dict[TestMetricKey, dict[str, float]]:
    """Index test metrics by ID dataset, target, mode, and objective."""

    metrics: dict[TestMetricKey, dict[str, float]] = {}
    for run in data.get("runs", []):
        run_name = str(run["run_name"])
        id_dataset = id_dataset_from_test_run_name(run_name)
        target = target_from_run_name(run_name)
        for metric in run.get("metrics", []):
            loss = metric.get("test_distillation_loss")
            metrics[
                (
                    id_dataset,
                    target,
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


def bold_latex(value: str) -> str:
    """Return a bold LaTeX value."""

    if value == "--":
        return value
    return rf"\textbf{{{value}}}"


def is_promising_ood_block(
    *,
    run_names: dict[RunKey, str],
    ood_metrics: dict[OodMetricKey, dict[str, float]],
    id_dataset: str,
    target: str,
    mode: str,
    objective: str,
    score: str,
) -> bool:
    """Return whether a row is strong across all OOD datasets for one ID dataset."""

    run_name = run_names[(id_dataset, target)]
    metrics = [
        ood_metrics.get((run_name, mode, objective, score, ood_dataset))
        for ood_dataset in OOD_DATASETS[id_dataset]
    ]
    if any(metric is None for metric in metrics):
        return False

    present_metrics = [metric for metric in metrics if metric is not None]
    min_roc_auc = min(metric["roc_auc"] for metric in present_metrics)
    avg_fpr = sum(metric["fpr_at_95_tpr"] for metric in present_metrics) / len(
        present_metrics
    )
    return (
        min_roc_auc >= PROMISING_MIN_ROC_AUC
        and avg_fpr <= PROMISING_MAX_AVG_FPR_AT_95_TPR
    )


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
    run_names: dict[RunKey, str],
    ood_metrics: dict[OodMetricKey, dict[str, float]],
    test_metrics: dict[TestMetricKey, dict[str, float]],
    target: str,
    mode: str,
    objective: str,
    score: str,
    include_test_metrics: bool,
) -> list[str]:
    """Return ordered metric cells for one table row."""

    cells: list[str] = []
    for id_dataset in ID_DATASETS:
        run_name = run_names[(id_dataset, target)]
        test_metric = test_metrics.get((id_dataset, target, mode, objective))
        bold_ood_cells = is_promising_ood_block(
            run_names=run_names,
            ood_metrics=ood_metrics,
            id_dataset=id_dataset,
            target=target,
            mode=mode,
            objective=objective,
            score=score,
        )
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
            formatted_metric = format_ood_metric(
                ood_metrics.get((run_name, mode, objective, score, ood_dataset))
            )
            cells.append(bold_latex(formatted_metric) if bold_ood_cells else formatted_metric)
    return cells


def latex_document(title: str, mode: str, rows: list[str]) -> str:
    """Wrap table rows in a standalone LaTeX document."""

    headers = ["Teacher target", "Objective", "OOD Score", *result_headers()]
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
    ood_metrics: dict[OodMetricKey, dict[str, float]],
    test_metrics: dict[TestMetricKey, dict[str, float]],
    mode: str,
) -> str:
    """Build one pixel-augmentation table for a probability mode."""

    rows: list[str] = []
    for target_index, target in enumerate(TARGET_ORDER):
        if target_index:
            rows.append(r"\midrule")
        for objective_index, objective in enumerate(OBJECTIVE_ORDER):
            if objective_index:
                rows.append(r"\addlinespace")
            for score_index, score in enumerate(DISPLAY_OOD_SCORE_ORDER):
                labels = [
                    TARGET_LABELS[target] if objective_index == 0 and score_index == 0 else "",
                    OBJECTIVE_LABELS[objective] if score_index == 0 else "",
                    OOD_SCORE_LABELS[score],
                ]
                rows.append(
                    " & ".join(
                        labels
                        + result_cells(
                            run_names=run_names,
                            ood_metrics=ood_metrics,
                            test_metrics=test_metrics,
                            target=target,
                            mode=mode,
                            objective=objective,
                            score=score,
                            include_test_metrics=score_index == 0,
                        )
                    )
                    + r" \\"
                )
    return latex_document(
        title="Pixel augmentation --- linear student, ResNet-50 layer4",
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
    """Write pixel-augmentation ResNet-50 LaTeX tables."""

    args = parse_args()
    run_names, ood_metrics = extract_ood_metrics(load_json(args.ood_metrics))
    test_metrics = extract_test_metrics(load_json(args.test_metrics))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for mode in ("unperturbed", "perturbed"):
        output_path = args.output_dir / f"metrics_pixel_augmentation_resnet50_{mode}.tex"
        output_path.write_text(
            build_table(
                run_names=run_names,
                ood_metrics=ood_metrics,
                test_metrics=test_metrics,
                mode=mode,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
