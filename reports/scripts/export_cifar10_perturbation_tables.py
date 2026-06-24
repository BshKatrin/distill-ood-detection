"""Export the CIFAR-10 perturbation comparison tables to LaTeX."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS_PATH = (
    ROOT
    / "reports"
    / "outputs"
    / "json"
    / "perturbation_20260623_ood_metrics_cifar10.json"
)
DEFAULT_TEACHERS_PATH = (
    ROOT / "reports" / "outputs" / "json" / "teacher_baselines.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "outputs" / "latex"

ID_DATASETS = ("cifar10_test", "cifar100_test")
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
)
OOD_SCORE_LABELS = {
    "max_probability_difference": "Max probability difference",
    "absolute_max_probability_difference": "Absolute max probability difference",
    "student_teacher_kl_divergence": "Student--teacher KL divergence",
    "msp": "MSP",
    "energy": "Energy",
}
MODE_ORDER = ("unperturbed", "perturbed")
CLIPPING_LEVEL_ORDER = ("constant", "spatial", "channel")
DROPOUT_LEVEL_ORDER = ("element", "spatial", "channel")

@dataclass(frozen=True)
class RunMetadata:
    """Metadata encoded by a perturbation experiment name."""

    family: Literal["clipping", "dropout", "pca"]
    level: str
    aggressive: bool
    teacher_perturbed: bool


MetricKey = tuple[str, str, str, str, str]
RunKey = tuple[RunMetadata, str]
TeacherMetricKey = tuple[str, str, str, str]
BestMetricKey = tuple[str, str]


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON report."""

    with path.open(encoding="utf-8") as file:
        return json.load(file)


def parse_run_name(run_name: str) -> RunMetadata:
    """Extract table metadata from a perturbation run name."""

    if "_clip_" in run_name:
        level = next(
            (value for value in CLIPPING_LEVEL_ORDER if f"_clip_{value}" in run_name),
            None,
        )
        if level is None:
            raise ValueError(f"Unknown clipping level in run name: {run_name}")
        return RunMetadata(
            family="clipping",
            level=level,
            aggressive="_aggressive" in run_name,
            teacher_perturbed=run_name.endswith("_perturbed"),
        )

    if "_mc_dropout" in run_name:
        if "_mc_dropout_spatial" in run_name:
            level = "spatial"
        elif "_mc_dropout_channel" in run_name:
            level = "channel"
        else:
            level = "element"
        return RunMetadata(
            family="dropout",
            level=level,
            aggressive=False,
            teacher_perturbed=run_name.endswith("_perturbed"),
        )

    if "_pca" in run_name:
        return RunMetadata(
            family="pca",
            level="pca",
            aggressive=False,
            teacher_perturbed=run_name.endswith("_perturbed"),
        )

    raise ValueError(f"Unknown perturbation run: {run_name}")


def extract_metrics(
    data: dict[str, Any],
) -> tuple[dict[MetricKey, dict[str, float]], dict[RunKey, str]]:
    """Index student metrics and parsed run metadata."""

    metrics: dict[MetricKey, dict[str, float]] = {}
    runs_by_variant: dict[RunKey, str] = {}
    for run in data.get("runs", []):
        id_dataset = str(run.get("id_dataset"))
        if id_dataset not in ID_DATASETS:
            continue
        run_name = str(run["run_name"])
        run_metadata = parse_run_name(run_name)
        runs_by_variant[(run_metadata, id_dataset)] = run_name
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
    return metrics, runs_by_variant


def extract_teacher_metrics(data: dict[str, Any]) -> dict[TeacherMetricKey, dict[str, float]]:
    """Index CIFAR-10 ResNet-18 and ResNet-50 teacher baselines."""

    metrics: dict[TeacherMetricKey, dict[str, float]] = {}
    for run in data.get("runs", []):
        id_dataset = str(run.get("id_dataset"))
        if id_dataset not in ID_DATASETS:
            continue
        run_name = str(run["run_name"])
        if run_name in {
            "teacher_resnet18_cifar10",
            "teacher_resnet18_cifar100",
        }:
            model = "ResNet-18"
        elif run_name in {
            "teacher_resnet50_cifar10",
            "teacher_resnet50_cifar100",
        }:
            model = "ResNet-50"
        else:
            continue
        for metric in run.get("metrics", []):
            key = (
                model,
                id_dataset,
                str(metric["ood_score"]),
                str(metric["ood_dataset"]),
            )
            metrics[key] = {
                "roc_auc": float(metric["roc_auc"]),
                "fpr_at_95_tpr": float(metric["fpr_at_95_tpr"]),
            }
    return metrics


def format_metric(
    metric: dict[str, float] | None,
    best: dict[str, float] | None = None,
) -> str:
    """Format one cell as ROC-AUC/FPR@95 or a missing-value marker."""

    if metric is None:
        return "--"
    roc_auc = f"{metric['roc_auc']:.2f}"
    fpr = f"{metric['fpr_at_95_tpr']:.2f}"
    value = f"{roc_auc}/{fpr}"
    if best is None:
        return value
    has_best_roc_auc = roc_auc == f"{best['roc_auc']:.2f}"
    has_best_fpr = fpr == f"{best['fpr_at_95_tpr']:.2f}"
    if has_best_roc_auc or has_best_fpr:
        return rf"\textbf{{{value}}}"
    return value


def yes_no(value: bool) -> str:
    """Format a boolean for a table cell."""

    return "Yes" if value else "No"


def teacher_rows(
    teacher_metrics: dict[TeacherMetricKey, dict[str, float]],
    prefix_cells: int,
) -> list[str]:
    """Build the four teacher baseline rows for a table."""

    rows: list[str] = []
    for model in ("ResNet-18", "ResNet-50"):
        for score in ("msp", "energy"):
            label_cells = [
                rf"\multicolumn{{{prefix_cells}}}{{l}}{{{model} {OOD_SCORE_LABELS[score]}}}"
            ]
            values = [
                format_metric(
                    teacher_metrics.get((model, id_dataset, score, dataset))
                )
                for id_dataset in ID_DATASETS
                for dataset in OOD_DATASETS[id_dataset]
            ]
            rows.append(" & ".join(label_cells + values) + r" \\")
    return rows


def latex_document(
    *,
    title: str,
    headers: list[str],
    rows: list[str],
    metadata_columns: int,
) -> str:
    """Wrap rows in a standalone booktabs LaTeX document."""

    total_columns = len(headers)
    metric_start = metadata_columns + 1
    metric_end = total_columns
    metric_columns = sum(len(OOD_DATASETS[value]) for value in ID_DATASETS)
    column_spec = "l" * metadata_columns + "c" * metric_columns
    header = " & ".join(headers) + r" \\"
    body = "\n".join(rows)
    return rf"""\documentclass[border=2pt]{{standalone}}
\usepackage{{booktabs}}

\begin{{document}}

\scriptsize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{{column_spec}}}
\multicolumn{{{total_columns}}}{{c}}{{\textbf{{{title}}}}}\\[4pt]
\multicolumn{{{metadata_columns}}}{{c}}{{}} & \multicolumn{{3}}{{c}}{{CIFAR-10 (ID)}} & \multicolumn{{3}}{{c}}{{CIFAR-100 (ID)}} \\
\multicolumn{{{metadata_columns}}}{{c}}{{}} & \multicolumn{{6}}{{c}}{{ROC-AUC $\uparrow$ / FPR@95 $\downarrow$}} \\
\cmidrule(lr){{{metric_start}-{metric_start + 2}}} \cmidrule(lr){{{metric_start + 3}-{metric_end}}}
\toprule
{header}
\midrule
{body}
\bottomrule
\end{{tabular}}

\end{{document}}
"""


def student_cells(
    metrics: dict[MetricKey, dict[str, float]],
    runs_by_variant: dict[RunKey, str],
    metadata: RunMetadata,
    mode: str,
    objective: str,
    score: str,
    best_values: dict[BestMetricKey, dict[str, float]],
) -> list[str]:
    """Return the ordered OOD dataset cells for one student row."""

    cells: list[str] = []
    for id_dataset in ID_DATASETS:
        run_name = runs_by_variant.get((metadata, id_dataset))
        for dataset in OOD_DATASETS[id_dataset]:
            key = None
            if run_name is not None:
                key = (run_name, mode, objective, score, dataset)
            metric = metrics.get(key) if key is not None else None
            cells.append(format_metric(metric, best_values.get((id_dataset, dataset))))
    return cells


def find_best_values(
    metrics: dict[MetricKey, dict[str, float]],
    runs_by_variant: dict[RunKey, str],
    variants: set[RunMetadata],
    modes: tuple[str, ...] = MODE_ORDER,
) -> dict[BestMetricKey, dict[str, float]]:
    """Find the best student ROC-AUC and FPR for every OOD dataset column."""

    values: dict[BestMetricKey, list[dict[str, float]]] = {}
    for metadata in variants:
        for id_dataset in ID_DATASETS:
            run_name = runs_by_variant.get((metadata, id_dataset))
            if run_name is None:
                continue
            for mode in modes:
                for objective in OBJECTIVE_ORDER:
                    for score in OOD_SCORE_ORDER:
                        for dataset in OOD_DATASETS[id_dataset]:
                            metric = metrics.get(
                                (run_name, mode, objective, score, dataset)
                            )
                            if metric is not None:
                                values.setdefault((id_dataset, dataset), []).append(
                                    metric
                                )
    return {
        key: {
            "roc_auc": max(metric["roc_auc"] for metric in column_values),
            "fpr_at_95_tpr": min(
                metric["fpr_at_95_tpr"] for metric in column_values
            ),
        }
        for key, column_values in values.items()
    }


def build_clipping_table(
    metrics: dict[MetricKey, dict[str, float]],
    runs_by_variant: dict[RunKey, str],
    teacher_metrics: dict[TeacherMetricKey, dict[str, float]],
    aggressive: bool,
) -> str:
    """Build one clipping-strength comparison table."""

    headers = [
        "Clipping level",
        "Teacher perturbed?",
        "Inference perturbed?",
        "Objective",
        "OOD Score",
        *(
            OOD_DATASET_LABELS[value]
            for id_value in ID_DATASETS
            for value in OOD_DATASETS[id_value]
        ),
    ]
    rows = teacher_rows(teacher_metrics, prefix_cells=5) + [r"\midrule"]
    variants = {
        metadata
        for metadata, _ in runs_by_variant
        if metadata.family == "clipping" and metadata.aggressive == aggressive
    }
    best_values = find_best_values(metrics, runs_by_variant, variants)
    sorted_variants = sorted(
        variants,
        key=lambda metadata: (
            CLIPPING_LEVEL_ORDER.index(metadata.level),
            metadata.teacher_perturbed,
        )
    )
    for run_index, meta in enumerate(sorted_variants):
        if run_index:
            rows.append(r"\addlinespace")
        first_run_row = True
        for mode in MODE_ORDER:
            first_mode_row = True
            for objective in OBJECTIVE_ORDER:
                for score_index, score in enumerate(OOD_SCORE_ORDER):
                    labels = [
                        meta.level.title(),
                        yes_no(meta.teacher_perturbed),
                        yes_no(mode == "perturbed"),
                        OBJECTIVE_LABELS[objective] if score_index == 0 else "",
                        OOD_SCORE_LABELS[score],
                    ]
                    if not first_run_row:
                        labels[:2] = ["", ""]
                    if not first_mode_row:
                        labels[2] = ""
                    row = labels + student_cells(
                        metrics,
                        runs_by_variant,
                        meta,
                        mode,
                        objective,
                        score,
                        best_values,
                    )
                    rows.append(" & ".join(row) + r" \\")
                    first_run_row = False
                    first_mode_row = False
    return latex_document(
        title=(
            "Aggressive clipping $(0, 0.5)$ --- linear student, ResNet-18 layer4"
            if aggressive
            else "Non-aggressive clipping $(0.5, 1)$ --- linear student, ResNet-18 layer4"
        ),
        headers=headers,
        rows=rows,
        metadata_columns=5,
    )


def build_dropout_table(
    metrics: dict[MetricKey, dict[str, float]],
    runs_by_variant: dict[RunKey, str],
    teacher_metrics: dict[TeacherMetricKey, dict[str, float]],
) -> str:
    """Build the Monte-Carlo dropout comparison table."""

    headers = [
        "Dropout level",
        "Inference perturbed?",
        "Objective",
        "OOD Score",
        *(
            OOD_DATASET_LABELS[value]
            for id_value in ID_DATASETS
            for value in OOD_DATASETS[id_value]
        ),
    ]
    rows = teacher_rows(teacher_metrics, prefix_cells=4) + [r"\midrule"]
    variants = {
        metadata
        for metadata, _ in runs_by_variant
        if metadata.family == "dropout" and not metadata.teacher_perturbed
    }
    best_values = find_best_values(metrics, runs_by_variant, variants)
    sorted_variants = sorted(
        variants, key=lambda metadata: DROPOUT_LEVEL_ORDER.index(metadata.level)
    )
    for run_index, meta in enumerate(sorted_variants):
        if run_index:
            rows.append(r"\addlinespace")
        first_run_row = True
        for mode in MODE_ORDER:
            first_mode_row = True
            for objective in OBJECTIVE_ORDER:
                for score_index, score in enumerate(OOD_SCORE_ORDER):
                    labels = [
                        meta.level.title(),
                        yes_no(mode == "perturbed"),
                        OBJECTIVE_LABELS[objective] if score_index == 0 else "",
                        OOD_SCORE_LABELS[score],
                    ]
                    if not first_run_row:
                        labels[0] = ""
                    if not first_mode_row:
                        labels[1] = ""
                    row = labels + student_cells(
                        metrics,
                        runs_by_variant,
                        meta,
                        mode,
                        objective,
                        score,
                        best_values,
                    )
                    rows.append(" & ".join(row) + r" \\")
                    first_run_row = False
                    first_mode_row = False
    return latex_document(
        title="Monte-Carlo dropout --- linear student, ResNet-18 layer4",
        headers=headers,
        rows=rows,
        metadata_columns=4,
    )


def build_pca_table(
    metrics: dict[MetricKey, dict[str, float]],
    runs_by_variant: dict[RunKey, str],
    teacher_metrics: dict[TeacherMetricKey, dict[str, float]],
) -> str:
    """Build the PCA comparison table."""

    headers = [
        "Objective",
        "OOD Score",
        *(
            OOD_DATASET_LABELS[value]
            for id_value in ID_DATASETS
            for value in OOD_DATASETS[id_value]
        ),
    ]
    rows = teacher_rows(teacher_metrics, prefix_cells=2) + [r"\midrule"]
    variants = {metadata for metadata, _ in runs_by_variant if metadata.family == "pca"}
    best_values = find_best_values(
        metrics, runs_by_variant, variants, modes=("unperturbed",)
    )
    for run_index, meta in enumerate(sorted(variants, key=lambda value: value.level)):
        if run_index:
            rows.append(r"\addlinespace")
        for objective in OBJECTIVE_ORDER:
            for score_index, score in enumerate(OOD_SCORE_ORDER):
                labels = [
                    OBJECTIVE_LABELS[objective] if score_index == 0 else "",
                    OOD_SCORE_LABELS[score],
                ]
                row = labels + student_cells(
                    metrics,
                    runs_by_variant,
                    meta,
                    "unperturbed",
                    objective,
                    score,
                    best_values,
                )
                rows.append(" & ".join(row) + r" \\")
            if objective != OBJECTIVE_ORDER[-1]:
                rows.append(r"\addlinespace")
    return latex_document(
        title="PCA-9 projection --- linear student, ResNet-18 layer4",
        headers=headers,
        rows=rows,
        metadata_columns=2,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--teachers", type=Path, default=DEFAULT_TEACHERS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Load JSON metrics and write the perturbation tables."""

    args = parse_args()
    metrics, runs_by_variant = extract_metrics(load_json(args.metrics))
    teacher_metrics = extract_teacher_metrics(load_json(args.teachers))
    outputs = {
        "metrics_perturbation_clipping_non_aggressive_cifar10.tex": build_clipping_table(
            metrics, runs_by_variant, teacher_metrics, aggressive=False
        ),
        "metrics_perturbation_clipping_aggressive_cifar10.tex": build_clipping_table(
            metrics, runs_by_variant, teacher_metrics, aggressive=True
        ),
        "metrics_perturbation_dropout_cifar10.tex": build_dropout_table(
            metrics, runs_by_variant, teacher_metrics
        ),
        "metrics_perturbation_pca_cifar10.tex": build_pca_table(
            metrics, runs_by_variant, teacher_metrics
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in outputs.items():
        output_path = args.output_dir / filename
        output_path.write_text(content, encoding="utf-8")
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
