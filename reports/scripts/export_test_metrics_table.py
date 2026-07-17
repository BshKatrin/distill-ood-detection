"""Export test accuracy/loss report tables from saved run metrics."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
import torch

from distill_ood_detection.distillation.losses import distillation_loss


ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
OUTPUT_DIR = REPORTS_DIR / "outputs" / "latex"
DEFAULT_OUTPUT_PATTERN = "metrics_test_{strategy}.tex"
DEFAULT_RUN_METRICS_PATH = REPORTS_DIR / "outputs" / "json" / "test_metrics.json"

DEFAULT_CONFIG_PATHS = [
    ROOT / "configs" / "students" / "baseline",
    ROOT / "configs" / "students" / "perturbation",
]

DATASET_LABELS = {
    "cifar10": "CIFAR-10 (ID)",
    "cifar100": "CIFAR-100 (ID)",
}

ID_DATASET_ORDER = ["cifar10", "cifar100"]

# Published test accuracies from the configured Hugging Face teacher model cards:
# https://huggingface.co/edadaltocg/resnet18_cifar10
# https://huggingface.co/edadaltocg/resnet18_cifar100
TEACHER_TEST_ACCURACIES = {
    "cifar10": 0.9498,
    "cifar100": 0.7926,
}

METHOD_LABELS = {
    "cross_entropy": "CE",
    "kl_divergence": "KL",
    "mse_logits": "Logit MSE",
}

METHOD_ORDER = ["cross_entropy", "kl_divergence", "mse_logits"]
STUDENT_ORDER = {
    "Linear": 0,
    "MLP": 1,
    "Random Forest": 2,
}
FEATURE_ORDER = {
    "Raw Pixels": 0,
    "Layer3": 1,
    "Layer4": 2,
}
PERTURBATION_ORDER = {
    "Constant": 0,
    "Channel": 1,
    "Spatial": 2,
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Report metadata loaded from one experiment config file."""

    path: Path
    strategy: str
    experiment_name: str
    dataset_name: str
    student_kind: str
    feature_source: str
    perturbation: str | None
    method_keys: tuple[str, ...]
    run_dir: Path


@dataclass(frozen=True)
class ReportRowGroup:
    """Configs that share the same rendered report metadata columns."""

    student_kind: str
    feature_source: str
    perturbation: str | None
    configs: tuple[ExperimentConfig, ...]


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML experiment config."""

    with path.open() as file:
        return yaml.safe_load(file)


def feature_source_label(feature_layer: str | None) -> str:
    """Return the report label for a student input feature source."""

    if feature_layer is None:
        return "Raw Pixels"
    return feature_layer.capitalize()


def student_kind_label(student_kind: str) -> str:
    """Return the report label for a student model kind."""

    labels = {
        "linear": "Linear",
        "mlp": "MLP",
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
        "spatial_dependent": "Spatial",
    }
    return labels.get(perturbation, perturbation.replace("_", " ").title())


def clipping_layers_label(perturbation_config: dict[str, Any]) -> str | None:
    """Return a report label for configured clipping layers."""

    clipping_layers = perturbation_config.get("clipping_layers")
    if not isinstance(clipping_layers, dict) or not clipping_layers:
        return None
    labels = []
    for layer_name, layer_config in clipping_layers.items():
        if not isinstance(layer_config, dict):
            continue
        clipping_mode = layer_config.get("clipping_mode")
        if isinstance(clipping_mode, str):
            labels.append(f"{layer_name}: {perturbation_label(clipping_mode)}")
    return "; ".join(labels) if labels else None


def method_keys_from_config(config: dict[str, Any]) -> tuple[str, ...]:
    """Return neural-network training methods from a config in report order."""

    methods = config.get("training", {}).get("methods", {})
    return tuple(method for method in METHOD_ORDER if methods.get(method) is not None)


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
    dataset = config["dataset"]
    strategy = config_strategy(path, config)
    feature_layer = student.get("feature_layer")
    perturbation_config = (
        config.get("strategy", {})
        .get("perturbation", {})
    )
    perturbation = clipping_layers_label(perturbation_config)
    if perturbation is not None:
        feature_layer = "layer4"
    return ExperimentConfig(
        path=path,
        strategy=strategy,
        experiment_name=config["experiment_name"],
        dataset_name=dataset["name"],
        student_kind=student_kind_label(student["kind"]),
        feature_source=feature_source_label(feature_layer),
        perturbation=perturbation,
        method_keys=method_keys_from_config(config),
        run_dir=ROOT / config["run_dir"],
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


def deduplicate_configs(configs: Iterable[ExperimentConfig]) -> list[ExperimentConfig]:
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
        print(
            "Skipping "
            f"{skipped.path}: duplicate experiment_name {skipped.experiment_name}",
            file=sys.stderr,
        )
    return list(selected.values())


def metrics_path(config: ExperimentConfig, method_key: str) -> Path:
    """Return the metrics path for one experiment method."""

    return config.run_dir / method_key / "metrics.json"


def config_has_metrics(config: ExperimentConfig) -> bool:
    """Return whether at least one configured method has saved metrics."""

    return any(metrics_path(config, method_key).exists() for method_key in config.method_keys)


def available_configs(
    configs: Iterable[ExperimentConfig],
    *,
    warn: bool = True,
) -> list[ExperimentConfig]:
    """Return configs with local metrics artifacts, warning for missing runs."""

    configs = deduplicate_configs(configs)
    existing_configs = [config for config in configs if config_has_metrics(config)]
    if warn:
        for config in configs:
            if config not in existing_configs:
                print(
                    f"Skipping {config.path}: missing method metrics in {config.run_dir}",
                    file=sys.stderr,
                )
    return existing_configs


def grouped_available_configs(
    configs: list[ExperimentConfig],
) -> dict[str, list[ExperimentConfig]]:
    """Group configs by OOD strategy when the strategy has local metrics."""

    configs = deduplicate_configs(configs)
    available_experiment_names = {
        config.experiment_name for config in available_configs(configs, warn=False)
    }
    by_strategy: dict[str, list[ExperimentConfig]] = {}
    strategies_with_metrics = {
        config.strategy
        for config in configs
        if config.experiment_name in available_experiment_names
    }
    for config in configs:
        if config.strategy not in strategies_with_metrics:
            continue
        if config.experiment_name not in available_experiment_names:
            print(
                f"Skipping {config.path}: missing method metrics in {config.run_dir}",
                file=sys.stderr,
            )
        by_strategy.setdefault(config.strategy, []).append(config)
    for strategy, strategy_configs in by_strategy.items():
        by_strategy[strategy] = sorted(strategy_configs, key=report_sort_key)
    return by_strategy


def report_sort_key(config: ExperimentConfig) -> tuple[int, int, int, str, str, str]:
    """Return report ordering for experiment rows."""

    return (
        STUDENT_ORDER.get(config.student_kind, len(STUDENT_ORDER)),
        FEATURE_ORDER.get(config.feature_source, len(FEATURE_ORDER)),
        PERTURBATION_ORDER.get(config.perturbation or "", len(PERTURBATION_ORDER)),
        config.dataset_name,
        config.experiment_name,
        config.path.name,
    )


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


def load_method_metrics(config: ExperimentConfig, method_key: str) -> dict[str, Any] | None:
    """Load metrics for one method if available."""

    path = metrics_path(config, method_key)
    if not path.exists():
        return None
    with path.open() as file:
        return json.load(file)


def format_test_metric(metrics: dict[str, Any]) -> str:
    """Format best-checkpoint test accuracy and distillation loss as one cell."""

    accuracy = metrics.get("test_accuracy")
    loss = metrics.get("test_distillation_loss")
    if not isinstance(accuracy, int | float) or not isinstance(loss, int | float):
        return ""
    return f"{accuracy:.4f}/{loss:.4f}"


def export_run_metrics(run_names: Iterable[str], output_path: Path) -> None:
    """Calculate test metrics from every discovered probability mode."""

    exported_runs: list[dict[str, Any]] = []
    for run_name in dict.fromkeys(run_names):
        run_dir = ROOT / "runs" / run_name
        config_path = run_dir / "resolved_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing resolved config for {run_name!r}: {config_path}")
        with config_path.open() as file:
            config = json.load(file)
        dataset_key = f"{config['dataset']['name']}_test"
        method_metrics: list[dict[str, Any]] = []
        probability_modes = probability_mode_dirs(run_dir)
        for probability_mode, probability_dir in probability_modes:
            dataset_dir = probability_dir / dataset_key
            teacher_path = dataset_dir / "teacher.pt"
            if not teacher_path.exists():
                continue
            teacher = torch.load(teacher_path, map_location="cpu")
            for method, student_path in student_artifact_paths(dataset_dir).items():
                student = torch.load(student_path, map_location="cpu")
                accuracy, loss = test_metrics_from_artifacts(method, teacher, student, config)
                method_metrics.append(
                    {
                        "probability_mode": probability_mode,
                        "method": method,
                        "test_accuracy": accuracy,
                        "test_distillation_loss": loss,
                    }
                )

        if not method_metrics:
            raise FileNotFoundError(f"Missing test probability artifacts for {run_name!r}")
        exported_runs.append(
            {
                "run_name": run_name,
                "probability_modes": [mode for mode, _path in probability_modes],
                "metrics": method_metrics,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file:
        json.dump({"version": 1, "runs": exported_runs}, file, indent=2, sort_keys=True)
    print(f"Wrote {output_path}")


def probability_mode_dirs(run_dir: Path) -> list[tuple[str, Path]]:
    """Discover mode-specific directories without reading a manifest."""

    root = run_dir / "probabilities"
    modes = [
        (mode, root / mode)
        for mode in ("unperturbed", "perturbed")
        if (root / mode).is_dir()
    ]
    return modes or ([('unperturbed', root)] if root.is_dir() else [])


def student_artifact_paths(dataset_dir: Path) -> dict[str, Path]:
    """Discover one preferred checkpoint artifact per student method."""

    paths: dict[str, Path] = {}
    for checkpoint in ("latest", "best"):
        for path in sorted(dataset_dir.glob(f"student_*_{checkpoint}.pt")):
            method = path.name[len("student_") : -len(f"_{checkpoint}.pt")]
            paths[method] = path
    return paths


def test_metrics_from_artifacts(
    method: str,
    teacher: dict[str, Any],
    student: dict[str, Any],
    config: dict[str, Any],
) -> tuple[float, float | None]:
    """Calculate test accuracy and distillation loss from output artifacts."""

    student_logits = torch.as_tensor(student["logits"])
    teacher_logits = torch.as_tensor(teacher["logits"])
    labels = torch.as_tensor(student["labels"]).long()
    prediction_logits = student_logits.mean(dim=1) if student_logits.ndim == 3 else student_logits
    accuracy = float((prediction_logits.argmax(dim=-1) == labels).float().mean().item())
    if method not in {"cross_entropy", "kl_divergence", "mse_logits"}:
        return accuracy, None

    if student_logits.ndim == 3:
        draws = student_logits.shape[1]
        student_logits = student_logits.reshape(-1, student_logits.shape[-1])
        teacher_logits = teacher_logits.reshape(-1, teacher_logits.shape[-1])
        labels = labels.repeat_interleave(draws)
    method_config = config.get("training", {}).get("methods", {}).get(method) or {}
    temperature = method_config.get("temperature") or 1.0
    alpha = method_config.get("alpha") or 0.5
    loss = distillation_loss(
        method,
        student_logits,
        teacher_logits,
        labels=labels,
        temperature=temperature,
        alpha=alpha,
    )
    return accuracy, float(loss.item())


def teacher_test_metric(dataset_name: str) -> str:
    """Format the published teacher test accuracy without a distillation loss."""

    accuracy = TEACHER_TEST_ACCURACIES[dataset_name]
    return f"{accuracy:.4f}/--"


def report_row_groups(configs: list[ExperimentConfig]) -> list[ReportRowGroup]:
    """Group configs that share the same rendered report metadata."""

    groups: dict[tuple[str, str, str | None], list[ExperimentConfig]] = {}
    for config in configs:
        key = (config.student_kind, config.feature_source, config.perturbation)
        groups.setdefault(key, []).append(config)
    return [
        ReportRowGroup(
            student_kind=key[0],
            feature_source=key[1],
            perturbation=key[2],
            configs=tuple(sorted(group_configs, key=report_sort_key)),
        )
        for key, group_configs in sorted(
            groups.items(),
            key=lambda item: report_sort_key(item[1][0]),
        )
    ]


def group_method_keys(group: ReportRowGroup) -> tuple[str, ...]:
    """Return all methods used by a row group in stable report order."""

    methods = {
        method_key
        for config in group.configs
        for method_key in config.method_keys
    }
    return tuple(method_key for method_key in METHOD_ORDER if method_key in methods)


def group_configs_by_dataset(
    group: ReportRowGroup,
    method_key: str,
) -> dict[str, ExperimentConfig]:
    """Return one config per ID dataset for a rendered row group and method."""

    selected: dict[str, ExperimentConfig] = {}
    for config in sorted(group.configs, key=report_sort_key):
        if method_key not in config.method_keys:
            continue
        existing = selected.get(config.dataset_name)
        if existing is not None:
            print(
                "Skipping duplicate rendered row source "
                f"{config.path}: already using {existing.path} for "
                f"{DATASET_LABELS.get(config.dataset_name, config.dataset_name)}, "
                f"{group.student_kind}/{group.feature_source}/"
                f"{group.perturbation or '-'}, "
                f"{METHOD_LABELS.get(method_key, method_key)}",
                file=sys.stderr,
            )
            continue
        selected[config.dataset_name] = config
    return selected


def build_table_rows(configs: list[ExperimentConfig], strategy: str) -> list[str]:
    """Build LaTeX table body rows."""

    metadata_columns = 4 if strategy == "perturbation" else 3
    teacher_prefix = " & ".join(["Teacher", *(["--"] * (metadata_columns - 1))])
    teacher_values = [teacher_test_metric(name) for name in ID_DATASET_ORDER]
    rows = [teacher_prefix + " & " + " & ".join(teacher_values) + r" \\", r"\midrule"]
    for group in report_row_groups(configs):
        method_keys = group_method_keys(group)
        if not method_keys:
            continue
        first_group_row = True
        rows_for_group = []
        for method_key in method_keys:
            config_by_dataset = group_configs_by_dataset(group, method_key)
            values = []
            for dataset_name in ID_DATASET_ORDER:
                config = config_by_dataset.get(dataset_name)
                metrics = (
                    load_method_metrics(config, method_key)
                    if config is not None
                    else None
                )
                values.append(
                    format_test_metric(metrics)
                    if metrics is not None
                    else ""
                )
            if not any(values):
                continue
            student_label = group.student_kind if first_group_row else ""
            feature_label = group.feature_source if first_group_row else ""
            method_label = METHOD_LABELS[method_key]
            if strategy == "perturbation":
                perturbation = group.perturbation if first_group_row else ""
                prefix = (
                    f"{student_label} & {feature_label} & {perturbation} & {method_label}"
                )
            else:
                prefix = f"{student_label} & {feature_label} & {method_label}"
            rows_for_group.append(prefix + " & " + " & ".join(values) + r" \\")
            first_group_row = False
        if rows_for_group:
            rows.extend(rows_for_group)
            rows.append(r"\midrule")
    if rows[-1] == r"\midrule":
        rows.pop()
    return rows


def build_strategy_table(strategy: str, configs: list[ExperimentConfig]) -> str:
    """Render one LaTeX test metrics table for one OOD strategy."""

    metadata_headers = ["Student", "Features"]
    if strategy == "perturbation":
        metadata_headers.append("Perturbation")
    metadata_headers.append("Training")

    column_spec = "l" * len(metadata_headers) + "cc"
    header_prefix = " & ".join(metadata_headers)
    id_headers = " & ".join(DATASET_LABELS[name] for name in ID_DATASET_ORDER)
    rows = build_table_rows(configs, strategy)
    body = "\n".join(rows)
    total_columns = len(metadata_headers) + len(ID_DATASET_ORDER)
    strategy_title = strategy.replace("_", " ").title()

    return rf"""\begin{{tabular}}{{{column_spec}}}
\multicolumn{{{total_columns}}}{{c}}{{\textbf{{{latex_escape(strategy_title)} Test Accuracy/Distillation Loss}}}}\\[4pt]
\toprule
{header_prefix} & {id_headers} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
"""


def build_latex_document(strategy: str, configs: list[ExperimentConfig]) -> str:
    """Render one standalone LaTeX test metrics table."""

    body = build_strategy_table(strategy, configs)
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
        "--run-names",
        nargs="+",
        default=None,
        metavar="RUN_NAME",
        help=(
            "Export test accuracy and distillation loss for these runs directly "
            "from their method metrics files, without requiring config files."
        ),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help=(
            "JSON destination for --run-names; defaults to "
            "reports/outputs/json/test_metrics.json."
        ),
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
    """Generate the report test metrics LaTeX tables."""

    args = parse_args()
    if args.json_output is not None and args.run_names is None:
        raise ValueError("--json-output requires --run-names.")
    if args.run_names is not None:
        export_run_metrics(
            args.run_names,
            args.json_output or DEFAULT_RUN_METRICS_PATH,
        )
        return
    configs = [experiment_config(path) for path in expand_config_paths(args.configs)]
    configs_by_strategy = grouped_available_configs(configs)

    if args.output is not None and len(configs_by_strategy) != 1:
        msg = "--output can only be used when one OOD strategy is selected."
        raise ValueError(msg)

    for strategy, strategy_configs in configs_by_strategy.items():
        output_path = args.output or output_path_for_strategy(args.output_dir, strategy)
        latex = build_latex_document(strategy, strategy_configs)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(latex)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
