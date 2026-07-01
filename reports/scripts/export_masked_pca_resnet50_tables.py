"""Export ResNet-50 CIFAR-10 masked-PCA sweep metrics tables to LaTeX."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OOD_METRICS_PATH = (
    ROOT
    / "reports"
    / "outputs"
    / "json"
    / "masked_pca_resnet50_cifar10_ood_metrics.json"
)
DEFAULT_TEST_METRICS_PATH = (
    ROOT
    / "reports"
    / "outputs"
    / "json"
    / "masked_pca_resnet50_cifar10_completed_test_metrics.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "outputs" / "latex"

ID_DATASET = "cifar10_test"
ID_DATASET_LABEL = "CIFAR-10 (ID)"
OOD_DATASETS = ("mnist_test", "svhn_test", "cifar100_test")
OOD_DATASET_LABELS = {
    "mnist_test": "MNIST",
    "svhn_test": "SVHN",
    "cifar100_test": "CIFAR-100",
}
TEACHER_TARGET_ORDER = ("clean", "perturbed")
TEACHER_TARGET_LABELS = {
    "clean": "Clean",
    "perturbed": "Perturbed",
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

OodMetricKey = tuple[str, float, str, str, str, str]
TestMetricKey = tuple[str, float, str]


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON metrics file."""

    with path.open(encoding="utf-8") as file:
        return json.load(file)


def teacher_target_from_run_name(run_name: str) -> str:
    """Return the teacher target encoded in a run name."""

    return "perturbed" if run_name.endswith("_perturbed") else "clean"


def mask_probability_from_run_name(run_name: str) -> float:
    """Return the mask probability encoded in a run name."""

    match = re.search(r"_mask_p(\d{3})", run_name)
    if match is None:
        raise ValueError(f"Cannot infer mask probability from run name: {run_name}")
    return int(match.group(1)) / 100


def extract_ood_metrics(
    data: dict[str, Any],
) -> tuple[list[float], dict[OodMetricKey, dict[str, float]]]:
    """Index OOD metrics by teacher target, mask probability, mode, objective, score, and OOD dataset."""

    metrics: dict[OodMetricKey, dict[str, float]] = {}
    mask_probabilities: set[float] = set()
    for run in data.get("runs", []):
        run_name = str(run["run_name"])
        teacher_target = teacher_target_from_run_name(run_name)
        mask_probability = mask_probability_from_run_name(run_name)
        mask_probabilities.add(mask_probability)
        for metric in run.get("metrics", []):
            key = (
                teacher_target,
                mask_probability,
                str(metric["probability_mode"]),
                str(metric["method"]),
                str(metric["ood_score"]),
                str(metric["ood_dataset"]),
            )
            metrics[key] = {
                "roc_auc": float(metric["roc_auc"]),
                "fpr_at_95_tpr": float(metric["fpr_at_95_tpr"]),
            }
    return sorted(mask_probabilities), metrics


def extract_test_metrics(data: dict[str, Any]) -> dict[TestMetricKey, dict[str, float]]:
    """Index test metrics by teacher target, mask probability, and objective."""

    metrics: dict[TestMetricKey, dict[str, float]] = {}
    for run in data.get("runs", []):
        teacher_target = str(run["teacher_target"])
        mask_probability = float(run["pca_mask_probability"])
        for metric in run.get("metrics", []):
            loss = metric.get("test_distillation_loss")
            metrics[
                (
                    teacher_target,
                    mask_probability,
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
    """Return result headers for the ID dataset."""

    return ["Test acc.", "Distill. loss", *(OOD_DATASET_LABELS[dataset] for dataset in OOD_DATASETS)]


def result_cells(
    *,
    ood_metrics: dict[OodMetricKey, dict[str, float]],
    test_metrics: dict[TestMetricKey, dict[str, float]],
    teacher_target: str,
    mask_probability: float,
    mode: str,
    objective: str,
    score: str,
    include_test_metrics: bool,
) -> list[str]:
    """Return ordered metric cells for one table row."""

    cells: list[str] = []
    test_metric = test_metrics.get((teacher_target, mask_probability, objective))
    if include_test_metrics:
        cells.extend(
            [
                format_test_accuracy(test_metric),
                format_distillation_loss(test_metric),
            ]
        )
    else:
        cells.extend(["", ""])
    for ood_dataset in OOD_DATASETS:
        cells.append(
            format_ood_metric(
                ood_metrics.get(
                    (
                        teacher_target,
                        mask_probability,
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

    headers = ["Teacher", "$p$", "Objective", "OOD Score", *result_headers()]
    header = " & ".join(headers) + r" \\"
    body = "\n".join(rows)
    return rf"""\documentclass[border=2pt]{{standalone}}
\usepackage{{booktabs}}

\begin{{document}}

\scriptsize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{llllccccc}}
\multicolumn{{9}}{{c}}{{\textbf{{{title}}}}}\\[4pt]
\multicolumn{{4}}{{c}}{{Inference: {mode}}} & \multicolumn{{5}}{{c}}{{{ID_DATASET_LABEL}}} \\
\cmidrule(lr){{5-9}}
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
    mask_probabilities: list[float],
    ood_metrics: dict[OodMetricKey, dict[str, float]],
    test_metrics: dict[TestMetricKey, dict[str, float]],
    mode: str,
) -> str:
    """Build one masked-PCA table for a probability mode."""

    rows: list[str] = []
    for teacher_target_index, teacher_target in enumerate(TEACHER_TARGET_ORDER):
        if teacher_target_index:
            rows.append(r"\midrule")
        for mask_index, mask_probability in enumerate(mask_probabilities):
            if mask_index:
                rows.append(r"\addlinespace")
            for objective_index, objective in enumerate(OBJECTIVE_ORDER):
                if objective_index:
                    rows.append(r"\addlinespace")
                for score_index, score in enumerate(OOD_SCORE_ORDER):
                    labels = [
                        (
                            TEACHER_TARGET_LABELS[teacher_target]
                            if mask_index == 0 and objective_index == 0 and score_index == 0
                            else ""
                        ),
                        f"{mask_probability:.2g}" if objective_index == 0 and score_index == 0 else "",
                        OBJECTIVE_LABELS[objective] if score_index == 0 else "",
                        OOD_SCORE_LABELS[score],
                    ]
                    rows.append(
                        " & ".join(
                            labels
                            + result_cells(
                                ood_metrics=ood_metrics,
                                test_metrics=test_metrics,
                                teacher_target=teacher_target,
                                mask_probability=mask_probability,
                                mode=mode,
                                objective=objective,
                                score=score,
                                include_test_metrics=score_index == 0,
                            )
                        )
                        + r" \\"
                    )
    return latex_document(
        title="Masked PCA projection sweep --- linear student, ResNet-50 CIFAR-10 layer4",
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
    mask_probabilities, ood_metrics = extract_ood_metrics(load_json(args.ood_metrics))
    test_metrics = extract_test_metrics(load_json(args.test_metrics))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for mode in ("unperturbed", "perturbed"):
        output_path = args.output_dir / f"metrics_masked_pca_resnet50_cifar10_{mode}.tex"
        output_path.write_text(
            build_table(
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
