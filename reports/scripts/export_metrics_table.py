"""Export the OOD metrics report table from saved run artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

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
OUTPUT_DIR = REPORTS_DIR / "outputs" / "latex"
DEFAULT_OUTPUT_PATTERN = "metrics_{strategy}.tex"

DEFAULT_CONFIG_PATHS = [
    ROOT / "configs" / "baseline",
    ROOT / "configs" / "perturbation",
]

DATASET_LABELS = {
    "cifar10_test": "CIFAR-10",
    "cifar100_test": "CIFAR-100",
    "mnist_test": "MNIST",
    "svhn_test": "SVHN",
}

METHOD_LABELS = {
    "cross_entropy": "CE",
    "kl_divergence": "KL",
    "mse_logits": "Logit MSE",
    "logits": "Logits",
}

METHOD_ORDER = ["cross_entropy", "mse_logits", "kl_divergence", "logits"]

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


@dataclass(frozen=True)
class ExperimentConfig:
    """Report metadata loaded from one experiment config file."""

    path: Path
    strategy: str
    experiment_name: str
    student_kind: str
    feature_source: str
    perturbation: str | None
    method_keys: tuple[str, ...]
    run_dir: Path


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

    probabilities = to_numpy(artifact["probabilities"])
    if probabilities.ndim != 2:
        msg = (
            "Teacher MSP must use raw-image teacher probabilities with shape "
            f"(n_samples, n_classes), got {probabilities.shape}."
        )
        raise ValueError(msg)
    return probabilities.max(axis=1)


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML experiment config."""

    with path.open() as file:
        return yaml.safe_load(file)


def feature_source_label(feature_layer: str | None) -> str:
    """Return the report label for a student input feature source."""

    if feature_layer is None:
        return "Raw pixels"
    return feature_layer.capitalize()


def student_kind_label(student_kind: str) -> str:
    """Return the report label for a student model kind."""

    labels = {
        "linear": "Linear",
        "mlp": "MLP",
        "random_forest": "Random Forest",
    }
    return labels.get(student_kind, student_kind.replace("_", " ").title())


def perturbation_label(perturbation: str | None) -> str:
    """Return the report label for a perturbation mode."""

    if perturbation is None:
        return ""
    labels = {
        "channel_dependent": "Channel",
        "constant": "Constant",
        "spatial": "Spatial",
    }
    return labels.get(perturbation, perturbation.replace("_", " ").title())


def method_keys_from_config(config: dict[str, Any]) -> tuple[str, ...]:
    """Return training methods from a config in stable report order."""

    methods = {
        **config.get("training", {}).get("methods", {}),
        **config.get("tree", {}).get("methods", {}),
    }
    return tuple(method for method in METHOD_ORDER if method in methods)


def config_strategy(path: Path, config: dict[str, Any]) -> str:
    """Return the OOD strategy for one config."""

    strategy = config.get("strategy", {}).get("name")
    if isinstance(strategy, str):
        return strategy
    try:
        return path.relative_to(ROOT / "configs").parts[0]
    except ValueError:
        return "baseline"


def experiment_config(path: Path) -> ExperimentConfig:
    """Load report metadata from one experiment config file."""

    config = load_yaml(path)
    student = config["student"]
    strategy = config_strategy(path, config)
    experiment_name = config["experiment_name"]
    output_dir = ROOT / config.get("output_dir", "runs")
    feature_layer = student.get("feature_layer")
    perturbation = (
        config.get("strategy", {})
        .get("perturbation", {})
        .get("clipping_mode")
    )
    return ExperimentConfig(
        path=path,
        strategy=strategy,
        experiment_name=experiment_name,
        student_kind=student_kind_label(student["kind"]),
        feature_source=feature_source_label(feature_layer),
        perturbation=(
            perturbation_label(perturbation) if perturbation is not None else None
        ),
        method_keys=method_keys_from_config(config),
        run_dir=output_dir / experiment_name,
    )


def expand_config_paths(paths: Iterable[Path]) -> list[Path]:
    """Expand config files and directories into a stable list of YAML files."""

    config_paths: list[Path] = []
    for path in paths:
        if path.is_dir():
            config_paths.extend(sorted(path.rglob("*.yaml")))
            config_paths.extend(sorted(path.rglob("*.yml")))
        else:
            config_paths.append(path)
    return sorted(dict.fromkeys(config_paths))


def deduplicate_configs(
    configs: Iterable[ExperimentConfig],
    *,
    warn: bool = True,
) -> list[ExperimentConfig]:
    """Keep one config per experiment name, preferring filename/name matches."""

    selected: dict[str, ExperimentConfig] = {}
    for config in configs:
        existing = selected.get(config.experiment_name)
        if existing is None:
            selected[config.experiment_name] = config
            continue
        current_matches_name = config.path.stem in config.experiment_name
        existing_matches_name = existing.path.stem in existing.experiment_name
        if current_matches_name and not existing_matches_name:
            selected[config.experiment_name] = config
            skipped = existing
        else:
            skipped = config
        if warn:
            print(
                "Skipping "
                f"{skipped.path}: duplicate experiment_name {skipped.experiment_name}",
                file=sys.stderr,
            )
    return list(selected.values())


def run_artifacts_available(config: ExperimentConfig) -> bool:
    """Return whether all probability manifest artifacts exist locally."""

    manifest_path = config.run_dir / "probabilities" / "manifest.json"
    if not manifest_path.exists():
        return False
    with manifest_path.open() as file:
        manifest = json.load(file)
    return all((ROOT / entry["path"]).exists() for entry in manifest["artifacts"])


def available_configs(
    configs: Iterable[ExperimentConfig],
    *,
    warn: bool = True,
) -> list[ExperimentConfig]:
    """Return configs with local probability artifacts, warning for missing runs."""

    configs = deduplicate_configs(configs, warn=warn)
    existing_configs = [
        config
        for config in configs
        if run_artifacts_available(config)
    ]
    missing_configs = [config for config in configs if config not in existing_configs]
    if warn:
        for config in missing_configs:
            print(
                "Skipping "
                f"{config.path}: missing probability artifacts in {config.run_dir}",
                file=sys.stderr,
            )
    return existing_configs


def teacher_reference_runs() -> dict[str, RunArtifacts]:
    """Load baseline raw-image teacher artifacts by ID dataset."""

    configs = [
        experiment_config(path)
        for path in expand_config_paths([ROOT / "configs" / "baseline"])
    ]
    references: dict[str, RunArtifacts] = {}
    for config in available_configs(configs, warn=False):
        if config.feature_source != "Raw pixels":
            continue
        run = RunArtifacts(config.run_dir)
        references.setdefault(run.id_dataset_key, run)
    return references


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
    teacher_runs_by_id_dataset: dict[str, RunArtifacts],
    columns: list[tuple[str, str]],
) -> dict[tuple[str, str], str]:
    """Compute teacher MSP metrics for each report column."""

    values = {}
    for id_dataset_key, ood_dataset_key in columns:
        if id_dataset_key not in teacher_runs_by_id_dataset:
            msg = (
                "Missing baseline raw-image teacher artifacts for "
                f"{DATASET_LABELS.get(id_dataset_key, id_dataset_key)}."
            )
            raise FileNotFoundError(msg)
        run = teacher_runs_by_id_dataset[id_dataset_key]
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


def available_columns(runs: list[RunArtifacts]) -> list[tuple[str, str]]:
    """Return report metric columns available for a set of runs."""

    id_dataset_keys = ["cifar10_test", "cifar100_test"]
    columns = []
    for id_dataset_key in id_dataset_keys:
        run = next((run for run in runs if run.id_dataset_key == id_dataset_key), None)
        if run is None:
            continue
        available_ood_datasets = set(run.ood_dataset_keys)
        columns.extend(
            (id_dataset_key, ood_dataset_key)
            for ood_dataset_key in OOD_DATASET_ORDER_BY_ID[id_dataset_key]
            if ood_dataset_key in available_ood_datasets
        )
    return columns


def build_table_rows(
    configs: list[ExperimentConfig],
    runs: list[RunArtifacts],
    teacher_runs_by_id_dataset: dict[str, RunArtifacts],
    columns: list[tuple[str, str]],
) -> list[str]:
    """Build LaTeX table body rows."""

    rows = []
    teacher_values = teacher_metrics_by_column(teacher_runs_by_id_dataset, columns)
    strategy = configs[0].strategy
    teacher_prefix = "Teacher & - & - & " + TEACHER_SCORE_LABEL
    if strategy == "perturbation":
        teacher_prefix = "Teacher & - & - & - & " + TEACHER_SCORE_LABEL
    rows.append(
        teacher_prefix
        + " & "
        + " & ".join(teacher_values[column] for column in columns)
        + r" \\",
    )

    for config in configs:
        run_by_id_dataset = {
            run.id_dataset_key: run for run in runs if run.run_name == config.experiment_name
        }
        if not run_by_id_dataset:
            continue
        method_keys = config.method_keys
        if not method_keys:
            continue
        rows.append(r"\midrule")
        for method_index, method_key in enumerate(method_keys):
            if method_index > 0:
                rows.append(r"\addlinespace")
            for score_index, score_key in enumerate(SCORE_ORDER):
                show_config_labels = method_index == 0 and score_index == 0
                student_label = config.student_kind if show_config_labels else ""
                feature_label = config.feature_source if show_config_labels else ""
                perturbation = config.perturbation or ""
                perturbation_label_text = (
                    perturbation if method_index == 0 and score_index == 0 else ""
                )
                method_label = METHOD_LABELS[method_key] if score_index == 0 else ""
                score_label = OOD_SCORE_LABELS[score_key]
                cells = []
                for id_dataset_key, ood_dataset_key in columns:
                    run = run_by_id_dataset.get(id_dataset_key)
                    if run is None or (id_dataset_key, method_key) not in run.student_artifacts:
                        cells.append("")
                    else:
                        cells.append(
                            student_metric(run, method_key, score_key, ood_dataset_key),
                        )
                if strategy == "perturbation":
                    row_prefix = (
                        f"{student_label} & {feature_label} & "
                        f"{perturbation_label_text} & {method_label} & {score_label}"
                    )
                else:
                    row_prefix = (
                        f"{student_label} & {feature_label} & {method_label} & {score_label}"
                    )
                rows.append(row_prefix + " & " + " & ".join(cells) + r" \\")
    return rows


def build_strategy_table(
    strategy: str,
    configs: list[ExperimentConfig],
    runs: list[RunArtifacts],
    teacher_runs_by_id_dataset: dict[str, RunArtifacts],
) -> str:
    """Render one LaTeX metrics table for one OOD strategy."""

    columns = available_columns(runs)
    if not columns:
        return ""
    metadata_headers = ["Student", "Features"]
    if strategy == "perturbation":
        metadata_headers.append("Perturbation")
    metadata_headers.extend(["Training", "OOD Score"])

    column_spec = "l" * len(metadata_headers) + "c" * len(columns)
    header_labels = " & ".join(
        latex_escape(DATASET_LABELS[ood_dataset_key])
        for _, ood_dataset_key in columns
    )

    cifar10_span = len([column for column in columns if column[0] == "cifar10_test"])
    cifar100_span = len([column for column in columns if column[0] == "cifar100_test"])
    metric_start = len(metadata_headers) + 1
    cifar10_start = metric_start
    cifar10_end = cifar10_start + cifar10_span - 1
    cifar100_start = cifar10_end + 1
    cifar100_end = cifar100_start + cifar100_span - 1

    rows = build_table_rows(configs, runs, teacher_runs_by_id_dataset, columns)
    body = "\n".join(rows)
    header_prefix = " & ".join(metadata_headers)
    id_headers = []
    cmidrules = []
    if cifar10_span:
        id_headers.append(rf"\multicolumn{{{cifar10_span}}}{{c}}{{CIFAR-10 (ID)}}")
        cmidrules.append(rf"\cmidrule(lr){{{cifar10_start}-{cifar10_end}}}")
    if cifar100_span:
        id_headers.append(rf"\multicolumn{{{cifar100_span}}}{{c}}{{CIFAR-100 (ID)}}")
        cmidrules.append(rf"\cmidrule(lr){{{cifar100_start}-{cifar100_end}}}")
    blank_metadata_headers = " & " * len(metadata_headers)
    strategy_title = strategy.replace("_", " ").title()
    total_columns = len(metadata_headers) + len(columns)

    return rf"""\begin{{tabular}}{{{column_spec}}}
\multicolumn{{{total_columns}}}{{c}}{{\textbf{{{latex_escape(strategy_title)}}}}}\\[4pt]
\toprule
{blank_metadata_headers}{' & '.join(id_headers)} \\
{' '.join(cmidrules)}
{header_prefix} & {header_labels} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
"""


def grouped_available_configs(
    configs: list[ExperimentConfig],
) -> dict[str, list[ExperimentConfig]]:
    """Group configs with local artifacts by OOD strategy."""

    by_strategy: dict[str, list[ExperimentConfig]] = {}
    for config in available_configs(configs):
        by_strategy.setdefault(config.strategy, []).append(config)
    return by_strategy


def build_latex_document(
    strategy: str,
    configs: list[ExperimentConfig],
    teacher_runs_by_id_dataset: dict[str, RunArtifacts],
) -> str:
    """Render one standalone LaTeX metrics table."""

    runs = [RunArtifacts(config.run_dir) for config in configs]
    body = build_strategy_table(strategy, configs, runs, teacher_runs_by_id_dataset)
    return rf"""\documentclass[border=2pt]{{standalone}}
\usepackage{{booktabs}}

\begin{{document}}

\setlength{{\tabcolsep}}{{5pt}}
{body}
\end{{document}}
"""


def output_path_for_strategy(output_dir: Path, strategy: str) -> Path:
    """Return the default LaTeX output path for one strategy."""

    return output_dir / DEFAULT_OUTPUT_PATTERN.format(strategy=strategy)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "configs",
        nargs="*",
        type=Path,
        default=DEFAULT_CONFIG_PATHS,
        help=(
            "Config files or directories to include. Directories are expanded "
            "recursively for .yaml and .yml files."
        ),
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Path to write one generated LaTeX document. Only valid when the "
            "selected configs contain one OOD strategy."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for generated strategy-specific LaTeX documents.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate the report metrics LaTeX table."""

    args = parse_args()
    if args.runs is not None:
        msg = "--runs is deprecated; pass config files or directories instead."
        raise ValueError(msg)
    configs = [experiment_config(path) for path in expand_config_paths(args.configs)]
    configs_by_strategy = grouped_available_configs(configs)
    teacher_runs_by_id_dataset = teacher_reference_runs()

    if args.output is not None and len(configs_by_strategy) != 1:
        msg = "--output can only be used when one OOD strategy is selected."
        raise ValueError(msg)

    for strategy, strategy_configs in configs_by_strategy.items():
        output_path = args.output or output_path_for_strategy(args.output_dir, strategy)
        latex = build_latex_document(
            strategy,
            strategy_configs,
            teacher_runs_by_id_dataset,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(latex)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
