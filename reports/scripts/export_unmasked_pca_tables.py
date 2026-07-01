"""Export unmasked PCA component-sweep metrics tables to LaTeX."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
JSON_DIR = ROOT / "reports" / "outputs" / "json"
DEFAULT_OOD_METRICS_PATHS = (
    JSON_DIR / "pca_reduced_components.json",
    JSON_DIR / "perturbation_20260623_ood_metrics_cifar10.json",
    JSON_DIR / "perturbation_20260623_ood_metrics_cifar100.json",
    JSON_DIR / "unmasked_pca_resnet18_reduced_components_ood_metrics.json",
    JSON_DIR / "unmasked_pca_resnet18_pca100_completed_ood_metrics.json",
)
DEFAULT_TEST_METRICS_PATHS = (
    JSON_DIR / "perturbation_20260623_test_metrics_cifar10.json",
    JSON_DIR / "perturbation_20260623_test_metrics_cifar100.json",
    JSON_DIR / "unmasked_pca_resnet18_reduced_components_test_metrics.json",
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
EXCLUDED_OOD_SCORES = {
    "logit_l2_distance",
    "energy_gap",
    "absolute_energy_gap",
}
DISPLAY_OOD_SCORE_ORDER = tuple(
    score for score in OOD_SCORE_ORDER if score not in EXCLUDED_OOD_SCORES
)
SELECTED_COMPONENTS = {
    "cifar10_test": 9,
    "cifar100_test": 100,
}
PROMISING_MIN_ROC_AUC = 0.87
PROMISING_MAX_AVG_FPR_AT_95_TPR = 0.60
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

VariantKey = tuple[str, int, bool]
OodMetricKey = tuple[str, int, bool, str, str, str, str]
TestMetricKey = tuple[str, int, bool, str, str]


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON metrics artifact."""

    with path.open(encoding="utf-8") as file:
        return json.load(file)


def latex_escape(text: str) -> str:
    """Escape a small string for use in LaTeX table cells."""

    return text.replace("_", r"\_")


def id_dataset_from_run_name(run_name: str) -> str:
    """Return the ID dataset key encoded in a run name."""

    if "cifar_100" in run_name or "cifar100" in run_name:
        return "cifar100_test"
    if "cifar_10" in run_name or "cifar10" in run_name:
        return "cifar10_test"
    raise ValueError(f"Cannot infer ID dataset from run name: {run_name}")


def pca_components_from_run_name(run_name: str) -> int:
    """Return the PCA component count encoded in a run name."""

    match = re.search(r"pca(?P<components>\d+)", run_name)
    if match is None:
        raise ValueError(f"Cannot infer PCA components from run name: {run_name}")
    return int(match.group("components"))


def is_pca_run(run_name: str) -> bool:
    """Return whether a run name encodes a PCA component count."""

    return re.search(r"pca\d+", run_name) is not None


def teacher_perturbed_from_run_name(run_name: str) -> bool:
    """Return whether the training target was the perturbed teacher output."""

    return run_name.endswith("_perturbed")


def extract_ood_metrics(
    paths: tuple[Path, ...],
) -> tuple[set[VariantKey], dict[OodMetricKey, dict[str, float]]]:
    """Index OOD metrics by dataset, PCA components, perturbation flag, and metric identity."""

    variants: set[VariantKey] = set()
    metrics: dict[OodMetricKey, dict[str, float]] = {}
    for path in paths:
        if not path.exists():
            continue
        for run in load_json(path).get("runs", []):
            run_name = str(run["run_name"])
            if not is_pca_run(run_name):
                continue
            id_dataset = str(run.get("id_dataset") or id_dataset_from_run_name(run_name))
            components = pca_components_from_run_name(run_name)
            teacher_perturbed = teacher_perturbed_from_run_name(run_name)
            variants.add((id_dataset, components, teacher_perturbed))
            for metric in run.get("metrics", []):
                key = (
                    id_dataset,
                    components,
                    teacher_perturbed,
                    str(metric["probability_mode"]),
                    str(metric["method"]),
                    str(metric["ood_score"]),
                    str(metric["ood_dataset"]),
                )
                metrics[key] = {
                    "roc_auc": float(metric["roc_auc"]),
                    "fpr_at_95_tpr": float(metric["fpr_at_95_tpr"]),
                }
    return variants, metrics


def extract_test_metrics(paths: tuple[Path, ...]) -> dict[TestMetricKey, dict[str, float]]:
    """Index test metrics by dataset, PCA components, perturbation flag, mode, and objective."""

    metrics: dict[TestMetricKey, dict[str, float]] = {}
    for path in paths:
        if not path.exists():
            continue
        for run in load_json(path).get("runs", []):
            run_name = str(run["run_name"])
            if not is_pca_run(run_name):
                continue
            id_dataset = id_dataset_from_run_name(run_name)
            components = pca_components_from_run_name(run_name)
            teacher_perturbed = teacher_perturbed_from_run_name(run_name)
            for metric in run.get("metrics", []):
                loss = metric.get("test_distillation_loss")
                key = (
                    id_dataset,
                    components,
                    teacher_perturbed,
                    str(metric["probability_mode"]),
                    str(metric["method"]),
                )
                metrics[key] = {
                    "test_accuracy": float(metric["test_accuracy"]),
                    "test_distillation_loss": float(loss) if loss is not None else float("nan"),
                }
    return metrics


def format_metric(metric: dict[str, float] | None) -> str:
    """Format one OOD metric cell as ROC-AUC/FPR@95."""

    if metric is None:
        return "--"
    return f"{metric['roc_auc']:.2f}/{metric['fpr_at_95_tpr']:.2f}"


def bold_latex(value: str) -> str:
    """Return a bold LaTeX value."""

    if value == "--":
        return value
    return rf"\textbf{{{value}}}"


def is_promising_selected_block(
    *,
    ood_metrics: dict[OodMetricKey, dict[str, float]],
    id_dataset: str,
    mode: str,
    objective: str,
    score: str,
) -> bool:
    """Return whether selected-row metrics are strong across all OOD datasets."""

    components = SELECTED_COMPONENTS[id_dataset]
    teacher_perturbed = False
    metrics = [
        ood_metrics.get(
            (
                id_dataset,
                components,
                teacher_perturbed,
                mode,
                objective,
                score,
                ood_dataset,
            )
        )
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


def yes_no(value: bool) -> str:
    """Format a boolean as a compact table label."""

    return "Yes" if value else "No"


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
    id_dataset: str,
    components: int,
    teacher_perturbed: bool,
    mode: str,
    objective: str,
    score: str,
    include_test_metrics: bool,
) -> list[str]:
    """Return metric cells for both ID dataset column groups."""

    cells: list[str] = []
    for column_id_dataset in ID_DATASETS:
        if column_id_dataset != id_dataset:
            cells.extend(["--", "--", "--", "--", "--"])
            continue
        test_metric = test_metrics.get(
            (id_dataset, components, teacher_perturbed, mode, objective)
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
            metric = ood_metrics.get(
                (
                    id_dataset,
                    components,
                    teacher_perturbed,
                    mode,
                    objective,
                    score,
                    ood_dataset,
                )
            )
            cells.append(format_metric(metric))
    return cells


def selected_result_cells(
    *,
    ood_metrics: dict[OodMetricKey, dict[str, float]],
    test_metrics: dict[TestMetricKey, dict[str, float]],
    mode: str,
    objective: str,
    score: str,
    include_test_metrics: bool,
) -> list[str]:
    """Return cells for the selected unmasked-PCA component counts."""

    cells: list[str] = []
    teacher_perturbed = False
    for id_dataset in ID_DATASETS:
        components = SELECTED_COMPONENTS[id_dataset]
        bold_ood_cells = is_promising_selected_block(
            ood_metrics=ood_metrics,
            id_dataset=id_dataset,
            mode=mode,
            objective=objective,
            score=score,
        )
        test_metric = test_metrics.get(
            (id_dataset, components, teacher_perturbed, mode, objective)
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
            cells.append(
                (
                    bold_latex(
                        format_metric(
                            ood_metrics.get(
                                (
                                    id_dataset,
                                    components,
                                    teacher_perturbed,
                                    mode,
                                    objective,
                                    score,
                                    ood_dataset,
                                )
                            )
                        )
                    )
                    if bold_ood_cells
                    else format_metric(
                        ood_metrics.get(
                            (
                                id_dataset,
                                components,
                                teacher_perturbed,
                                mode,
                                objective,
                                score,
                                ood_dataset,
                            )
                        )
                    )
                )
            )
    return cells


def latex_document(title: str, mode: str, rows: list[str]) -> str:
    """Wrap table rows in a standalone LaTeX document."""

    headers = ["ID dataset", "PCA components", "Teacher perturbed?", "Objective", "OOD Score"]
    headers.extend(result_headers())
    header = " & ".join(headers) + r" \\"
    body = "\n".join(rows)
    return rf"""\documentclass[border=2pt]{{standalone}}
\usepackage{{booktabs}}

\begin{{document}}

\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{lllllcccccccccc}}
\multicolumn{{15}}{{c}}{{\textbf{{{title}}}}}\\[4pt]
\multicolumn{{5}}{{c}}{{Inference: {latex_escape(mode)}}} & \multicolumn{{5}}{{c}}{{{ID_DATASET_LABELS["cifar10_test"]}}} & \multicolumn{{5}}{{c}}{{{ID_DATASET_LABELS["cifar100_test"]}}} \\
\cmidrule(lr){{6-10}} \cmidrule(lr){{11-15}}
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
    variants: set[VariantKey],
    ood_metrics: dict[OodMetricKey, dict[str, float]],
    test_metrics: dict[TestMetricKey, dict[str, float]],
    mode: str,
) -> str:
    """Build one unmasked-PCA component-sweep table for a probability mode."""

    rows: list[str] = []
    mode_variants = {
        variant
        for variant in variants
        if any(key[:4] == (*variant, mode) for key in ood_metrics)
    }
    for variant_index, (id_dataset, components, teacher_perturbed) in enumerate(
        sorted(mode_variants, key=lambda item: (ID_DATASETS.index(item[0]), item[1], item[2]))
    ):
        if variant_index:
            rows.append(r"\addlinespace")
        first_variant_row = True
        for objective in OBJECTIVE_ORDER:
            for score_index, score in enumerate(DISPLAY_OOD_SCORE_ORDER):
                labels = [
                    ID_DATASET_LABELS[id_dataset],
                    str(components),
                    yes_no(teacher_perturbed),
                    OBJECTIVE_LABELS[objective] if score_index == 0 else "",
                    OOD_SCORE_LABELS[score],
                ]
                if not first_variant_row:
                    labels[:3] = ["", "", ""]
                cells = result_cells(
                    ood_metrics=ood_metrics,
                    test_metrics=test_metrics,
                    id_dataset=id_dataset,
                    components=components,
                    teacher_perturbed=teacher_perturbed,
                    mode=mode,
                    objective=objective,
                    score=score,
                    include_test_metrics=score_index == 0,
                )
                rows.append(" & ".join(labels + cells) + r" \\")
                first_variant_row = False
    return latex_document(
        title="Unmasked PCA projection component sweep --- linear student, ResNet-18 layer4",
        mode=mode,
        rows=rows,
    )


def selected_latex_document(title: str, mode: str, rows: list[str]) -> str:
    """Wrap selected-component table rows in a standalone LaTeX document."""

    headers = ["Objective", "OOD Score", *result_headers()]
    header = " & ".join(headers) + r" \\"
    body = "\n".join(rows)
    cifar10_label = (
        f'{ID_DATASET_LABELS["cifar10_test"]}, '
        f'{SELECTED_COMPONENTS["cifar10_test"]} PCs'
    )
    cifar100_label = (
        f'{ID_DATASET_LABELS["cifar100_test"]}, '
        f'{SELECTED_COMPONENTS["cifar100_test"]} PCs'
    )
    return rf"""\documentclass[border=2pt]{{standalone}}
\usepackage{{booktabs}}

\begin{{document}}

\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{llcccccccccc}}
\multicolumn{{12}}{{c}}{{\textbf{{{title}}}}}\\[4pt]
\multicolumn{{2}}{{c}}{{Inference: {latex_escape(mode)}}} & \multicolumn{{5}}{{c}}{{{cifar10_label}}} & \multicolumn{{5}}{{c}}{{{cifar100_label}}} \\
\cmidrule(lr){{3-7}} \cmidrule(lr){{8-12}}
\toprule
{header}
\midrule
{body}
\bottomrule
\end{{tabular}}

\end{{document}}
"""


def build_selected_table(
    *,
    ood_metrics: dict[OodMetricKey, dict[str, float]],
    test_metrics: dict[TestMetricKey, dict[str, float]],
    mode: str,
) -> str:
    """Build one table for the selected unmasked-PCA component counts."""

    rows: list[str] = []
    for objective_index, objective in enumerate(OBJECTIVE_ORDER):
        if objective_index:
            rows.append(r"\addlinespace")
        for score_index, score in enumerate(DISPLAY_OOD_SCORE_ORDER):
            labels = [
                OBJECTIVE_LABELS[objective] if score_index == 0 else "",
                OOD_SCORE_LABELS[score],
            ]
            cells = selected_result_cells(
                ood_metrics=ood_metrics,
                test_metrics=test_metrics,
                mode=mode,
                objective=objective,
                score=score,
                include_test_metrics=score_index == 0,
            )
            rows.append(" & ".join(labels + cells) + r" \\")
    return selected_latex_document(
        title="Unmasked PCA projection --- linear student, ResNet-18 layer4",
        mode=mode,
        rows=rows,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ood-metrics",
        type=Path,
        action="append",
        default=None,
        help="OOD metric JSON artifact. May be passed more than once.",
    )
    parser.add_argument(
        "--test-metrics",
        type=Path,
        action="append",
        default=None,
        help="Test metric JSON artifact. May be passed more than once.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Write unmasked-PCA LaTeX tables."""

    args = parse_args()
    ood_paths = tuple(args.ood_metrics or DEFAULT_OOD_METRICS_PATHS)
    test_paths = tuple(args.test_metrics or DEFAULT_TEST_METRICS_PATHS)
    variants, ood_metrics = extract_ood_metrics(ood_paths)
    test_metrics = extract_test_metrics(test_paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for mode in ("unperturbed", "perturbed"):
        output_path = args.output_dir / f"metrics_unmasked_pca_resnet18_{mode}.tex"
        output_path.write_text(
            build_table(
                variants=variants,
                ood_metrics=ood_metrics,
                test_metrics=test_metrics,
                mode=mode,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {output_path}")
    selected_output_path = args.output_dir / "metrics_unmasked_pca_resnet18_selected_unperturbed.tex"
    selected_output_path.write_text(
        build_selected_table(
            ood_metrics=ood_metrics,
            test_metrics=test_metrics,
            mode="unperturbed",
        ),
        encoding="utf-8",
    )
    print(f"Wrote {selected_output_path}")


if __name__ == "__main__":
    main()
