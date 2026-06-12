"""Export the OOD metrics report table from saved run artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from distill_ood_detection.evaluation.ood_metrics import ood_detection_metrics
from distill_ood_detection.evaluation.ood_scores import (
    ABSOLUTE_MAX_PROBABILITY_DIFFERENCE,
    LOGIT_L2_DISTANCE,
    MAX_PROBABILITY_DIFFERENCE,
    STUDENT_TEACHER_KL_DIVERGENCE,
    absolute_max_probability_difference,
    logit_l2_distance,
    max_probability_difference,
    student_teacher_kl_divergence,
)


ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
OUTPUT_DIR = REPORTS_DIR / "output"
DEFAULT_OUTPUT = OUTPUT_DIR / "metrics.tex"

RUN_ORDER = [
    "linear_student_resnet18_cifar10",
    "linear_student_resnet18_cifar100",
    "mlp_student_resnet18_cifar10",
    "mlp_student_resnet18_cifar100",
    "feature_mlp_layer3_student_resnet18_cifar10",
    "feature_mlp_layer3_student_resnet18_cifar100",
    "feature_random_forest_layer3_student_resnet18_cifar10",
]

STUDENT_SECTION_LABELS = {
    "linear_student": "Linear Student",
    "mlp_student": "MLP Student",
    "feature_mlp_layer3_student": "Layer3 Feature Linear Student",
    "feature_random_forest_layer3_student": "Layer3 Feature Random Forest Student",
}

DATASET_LABELS = {
    "cifar10_test": "CIFAR-10",
    "cifar100_test": "CIFAR-100",
    "mnist_test": "MNIST",
    "svhn_test": "SVHN",
}

METHOD_LABELS = {
    "cross_entropy": "CE",
    "mse_logits": "Logit MSE",
    "logits": "Logits",
}

METHOD_ORDER = ["cross_entropy", "mse_logits"]

METHOD_ORDER_BY_FAMILY = {
    "linear_student": ["cross_entropy", "mse_logits"],
    "mlp_student": ["cross_entropy", "mse_logits"],
    "feature_mlp_layer3_student": ["cross_entropy", "mse_logits"],
    "feature_random_forest_layer3_student": ["logits"],
}

OOD_DATASET_ORDER_BY_ID = {
    "cifar10_test": ["mnist_test", "svhn_test", "cifar100_test"],
    "cifar100_test": ["mnist_test", "svhn_test", "cifar10_test"],
}

TEACHER_SCORE_LABEL = "Teacher MSP"

OOD_SCORE_LABELS = {
    MAX_PROBABILITY_DIFFERENCE: "Max Diff",
    ABSOLUTE_MAX_PROBABILITY_DIFFERENCE: "Abs. Max Diff",
    STUDENT_TEACHER_KL_DIVERGENCE: "KL Div.",
    LOGIT_L2_DISTANCE: "Logit L2",
}

PROBABILITY_SCORE_FUNCTIONS: dict[
    str,
    Callable[[np.ndarray, np.ndarray], np.ndarray],
] = {
    MAX_PROBABILITY_DIFFERENCE: lambda teacher, student: max_probability_difference(
        teacher,
        student,
        signed=True,
    ),
    ABSOLUTE_MAX_PROBABILITY_DIFFERENCE: (
        lambda teacher, student: absolute_max_probability_difference(
            teacher,
            student,
            signed=True,
        )
    ),
    STUDENT_TEACHER_KL_DIVERGENCE: lambda teacher, student: student_teacher_kl_divergence(
        teacher,
        student,
        signed=True,
    ),
}

SCORE_ORDER = [
    MAX_PROBABILITY_DIFFERENCE,
    ABSOLUTE_MAX_PROBABILITY_DIFFERENCE,
    STUDENT_TEACHER_KL_DIVERGENCE,
    LOGIT_L2_DISTANCE,
]


def load_artifact(path: Path) -> dict[str, Any]:
    """Load one saved probability artifact on CPU."""

    return torch.load(path, map_location="cpu")


def to_numpy(value: Any) -> np.ndarray:
    """Convert a tensor-like artifact value to a NumPy array."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def teacher_msp_scores(artifact: dict[str, Any]) -> np.ndarray:
    """Compute maximum softmax probability scores from one teacher artifact."""

    return to_numpy(artifact["probabilities"]).max(axis=1)


def run_family(run_name: str) -> str:
    """Return the report section family for a run directory name."""

    for prefix in STUDENT_SECTION_LABELS:
        if run_name.startswith(prefix):
            return prefix
    msg = f"Missing student section label for run: {run_name}"
    raise ValueError(msg)


def latex_escape(text: str) -> str:
    """Escape text for LaTeX table cells."""

    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
    }
    escaped = text
    for old, new in replacements.items():
        escaped = escaped.replace(old, new)
    return escaped


def format_metric(metrics: dict[str, float]) -> str:
    """Format ROC-AUC and FPR@95 as one report cell."""

    return f"{metrics['roc_auc']:.2f}/{metrics['fpr_at_95_tpr']:.2f}"


class RunArtifacts:
    """Loaded artifacts and metadata for one run directory."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_name = run_dir.name
        self.family = run_family(self.run_name)
        self.manifest = self._load_manifest()
        self.id_dataset_key = f"{self.manifest['dataset']['name']}_test"
        self.ood_dataset_keys = [
            f"{dataset['name']}_{dataset['split']}"
            for dataset in self.manifest["dataset"]["ood_datasets"]
        ]
        self.teacher_artifacts = self._load_teacher_artifacts()
        self.student_artifacts = self._load_student_artifacts()

    def _load_manifest(self) -> dict[str, Any]:
        manifest_path = self.run_dir / "probabilities" / "manifest.json"
        with manifest_path.open() as file:
            return json.load(file)

    def _load_teacher_artifacts(self) -> dict[str, dict[str, Any]]:
        artifacts = {}
        for entry in self.manifest["artifacts"]:
            if entry["model"] == "teacher":
                artifacts[entry["dataset"]] = load_artifact(ROOT / entry["path"])
        return artifacts

    def _load_student_artifacts(self) -> dict[tuple[str, str], dict[str, Any]]:
        artifacts = {}
        for entry in self.manifest["artifacts"]:
            if entry["model"] == "student":
                method_key = entry.get("method") or entry.get("mode")
                artifacts[(entry["dataset"], method_key)] = load_artifact(
                    ROOT / entry["path"],
                )
        return artifacts


def metric_for_scores(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
) -> dict[str, float]:
    """Compute OOD detection metrics for one ID/OOD score pair."""

    labels = np.concatenate(
        [
            np.ones(id_scores.shape[0], dtype=int),
            np.zeros(ood_scores.shape[0], dtype=int),
        ],
    )
    scores = np.concatenate([id_scores, ood_scores])
    return ood_detection_metrics(labels, scores)


def teacher_metrics_by_column(
    runs_by_id_dataset: dict[str, RunArtifacts],
    columns: list[tuple[str, str]],
) -> dict[tuple[str, str], str]:
    """Compute teacher MSP metrics for each report column."""

    values = {}
    for id_dataset_key, ood_dataset_key in columns:
        run = runs_by_id_dataset[id_dataset_key]
        id_scores = teacher_msp_scores(run.teacher_artifacts[id_dataset_key])
        ood_scores = teacher_msp_scores(run.teacher_artifacts[ood_dataset_key])
        values[(id_dataset_key, ood_dataset_key)] = format_metric(
            metric_for_scores(id_scores, ood_scores),
        )
    return values


def student_metric(
    run: RunArtifacts,
    method_key: str,
    score_key: str,
    ood_dataset_key: str,
) -> str:
    """Compute one student metric table cell."""

    id_dataset_key = run.id_dataset_key
    id_teacher = run.teacher_artifacts[id_dataset_key]
    ood_teacher = run.teacher_artifacts[ood_dataset_key]
    id_student = run.student_artifacts[(id_dataset_key, method_key)]
    ood_student = run.student_artifacts[(ood_dataset_key, method_key)]

    if score_key == LOGIT_L2_DISTANCE:
        id_scores = logit_l2_distance(
            to_numpy(id_teacher["logits"]),
            to_numpy(id_student["logits"]),
            signed=True,
        )
        ood_scores = logit_l2_distance(
            to_numpy(ood_teacher["logits"]),
            to_numpy(ood_student["logits"]),
            signed=True,
        )
    else:
        score_function = PROBABILITY_SCORE_FUNCTIONS[score_key]
        id_scores = score_function(
            to_numpy(id_teacher["probabilities"]),
            to_numpy(id_student["probabilities"]),
        )
        ood_scores = score_function(
            to_numpy(ood_teacher["probabilities"]),
            to_numpy(ood_student["probabilities"]),
        )

    return format_metric(metric_for_scores(id_scores, ood_scores))


def grouped_runs(runs: Iterable[RunArtifacts]) -> dict[str, dict[str, RunArtifacts]]:
    """Group runs by report section and ID dataset."""

    grouped: dict[str, dict[str, RunArtifacts]] = {}
    for run in runs:
        grouped.setdefault(run.family, {})[run.id_dataset_key] = run
    return grouped


def build_table_rows(
    runs: list[RunArtifacts],
    columns: list[tuple[str, str]],
) -> list[str]:
    """Build LaTeX table body rows."""

    runs_by_id_dataset = {run.id_dataset_key: run for run in runs}
    rows = []
    teacher_values = teacher_metrics_by_column(runs_by_id_dataset, columns)
    rows.append(
        "- & "
        + TEACHER_SCORE_LABEL
        + " & "
        + " & ".join(teacher_values[column] for column in columns)
        + r" \\",
    )

    by_family = grouped_runs(runs)
    for family_key, section_label in STUDENT_SECTION_LABELS.items():
        rows.append(r"\midrule")
        rows.append(
            rf"\multicolumn{{{2 + len(columns)}}}{{l}}{{\textbf{{{section_label}}}}} \\",
        )
        for method_index, method_key in enumerate(METHOD_ORDER_BY_FAMILY[family_key]):
            if method_index > 0:
                rows.append(r"\addlinespace")
            for score_index, score_key in enumerate(SCORE_ORDER):
                method_label = METHOD_LABELS[method_key] if score_index == 0 else ""
                score_label = OOD_SCORE_LABELS[score_key]
                cells = []
                for id_dataset_key, ood_dataset_key in columns:
                    run = by_family[family_key].get(id_dataset_key)
                    if run is None:
                        cells.append("")
                    else:
                        cells.append(
                            student_metric(run, method_key, score_key, ood_dataset_key),
                        )
                rows.append(
                    f"{method_label} & {score_label} & "
                    + " & ".join(cells)
                    + r" \\",
                )
    return rows


def build_latex_document(runs: list[RunArtifacts]) -> str:
    """Render the standalone LaTeX metrics table."""

    id_dataset_keys = ["cifar10_test", "cifar100_test"]
    columns = []
    for id_dataset_key in id_dataset_keys:
        run = next(run for run in runs if run.id_dataset_key == id_dataset_key)
        available_ood_datasets = set(run.ood_dataset_keys)
        columns.extend(
            (id_dataset_key, ood_dataset_key)
            for ood_dataset_key in OOD_DATASET_ORDER_BY_ID[id_dataset_key]
            if ood_dataset_key in available_ood_datasets
        )
    column_count = 2 + len(columns)
    column_spec = "ll" + "c" * len(columns)
    header_labels = " & ".join(
        latex_escape(DATASET_LABELS[ood_dataset_key])
        for _, ood_dataset_key in columns
    )

    cifar10_span = len([column for column in columns if column[0] == "cifar10_test"])
    cifar100_span = len([column for column in columns if column[0] == "cifar100_test"])
    cifar10_start = 3
    cifar10_end = cifar10_start + cifar10_span - 1
    cifar100_start = cifar10_end + 1
    cifar100_end = cifar100_start + cifar100_span - 1

    rows = build_table_rows(runs, columns)
    body = "\n".join(rows)

    return rf"""\documentclass[border=2pt]{{standalone}}
\usepackage{{booktabs}}

\begin{{document}}

\setlength{{\tabcolsep}}{{5pt}}
\begin{{tabular}}{{{column_spec}}}
\toprule
& & \multicolumn{{{cifar10_span}}}{{c}}{{CIFAR-10 (ID)}} & \multicolumn{{{cifar100_span}}}{{c}}{{CIFAR-100 (ID)}} \\
\cmidrule(lr){{{cifar10_start}-{cifar10_end}}} \cmidrule(lr){{{cifar100_start}-{cifar100_end}}}
Training & OOD Score & {header_labels} \\
\midrule
{body}
\bottomrule
\end{{tabular}}

\end{{document}}
"""


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        nargs="+",
        default=[str(ROOT / "runs" / run_name) for run_name in RUN_ORDER],
        help="Run directories to include in the report table.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write the generated LaTeX document.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate the report metrics LaTeX table."""

    args = parse_args()
    runs = [RunArtifacts(Path(run_dir)) for run_dir in args.runs]
    latex = build_latex_document(runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(latex)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
