"""Export the OOD metrics report table from saved run artifacts."""

from __future__ import annotations

import argparse
import gc
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
DEFAULT_CACHE_PATH = REPORTS_DIR / "outputs" / "cache" / "ood_metrics.json"
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

METHOD_ORDER = ["cross_entropy", "kl_divergence", "mse_logits", "logits"]
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

OOD_DATASET_ORDER_BY_ID = {
    "cifar10_test": ["mnist_test", "svhn_test", "cifar100_test"],
    "cifar100_test": ["mnist_test", "svhn_test", "cifar10_test"],
}

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
    dataset_name: str
    ood_dataset_keys: tuple[str, ...]
    teacher_label: str
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


def load_artifact(path: Path) -> dict[str, Any]:
    """Load one saved probability artifact on CPU."""

    return torch.load(path, map_location="cpu")


def load_artifact_array(path: Path, key: str) -> np.ndarray:
    """Load one array from a saved probability artifact on CPU."""

    artifact = load_artifact(path)
    try:
        return to_numpy(artifact[key])
    finally:
        del artifact


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
        return "Raw Pixels"
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
        "spatial_dependent": "Spatial",
    }
    return labels.get(perturbation, perturbation.replace("_", " ").title())


def teacher_name_label(hf_model_id: str) -> str:
    """Return the report label for one teacher checkpoint."""

    model_id = hf_model_id.lower()
    if "resnet18" in model_id:
        return "ResNet-18"
    if "resnet50" in model_id:
        return "ResNet-50"
    model_name = hf_model_id.rsplit("/", maxsplit=1)[-1]
    return model_name.replace("_", " ").title()


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
    dataset = config["dataset"]
    strategy = config_strategy(path, config)
    experiment_name = config["experiment_name"]
    output_dir = ROOT / config.get("output_dir", "runs")
    feature_layer = student.get("feature_layer")
    teacher_hf_model_id = config["teacher"]["hf_model_id"]
    perturbation = (
        config.get("strategy", {})
        .get("perturbation", {})
        .get("clipping_mode")
    )
    return ExperimentConfig(
        path=path,
        strategy=strategy,
        experiment_name=experiment_name,
        dataset_name=dataset["name"],
        ood_dataset_keys=tuple(
            f"{ood_dataset['name']}_{ood_dataset['split']}"
            for ood_dataset in dataset.get("ood_datasets", [])
        ),
        teacher_label=teacher_name_label(teacher_hf_model_id),
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


def teacher_reference_runs(
    configs: Iterable[ExperimentConfig],
) -> dict[tuple[str, str], TeacherProbabilityRunArtifacts]:
    """Load teacher-only probability artifacts keyed by ID dataset and teacher label."""

    required_teacher_labels = {config.teacher_label for config in configs}
    references: dict[tuple[str, str], TeacherProbabilityRunArtifacts] = {}
    for manifest_path in sorted((ROOT / "runs").glob("*/teacher_probabilities/manifest.json")):
        run_dir = manifest_path.parents[1]
        run = TeacherProbabilityRunArtifacts(run_dir)
        if run.teacher_label not in required_teacher_labels:
            continue
        references.setdefault((run.id_dataset_key, run.teacher_label), run)
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


def progress(message: str) -> None:
    """Print one report export progress message."""

    print(f"[metrics] {message}", file=sys.stderr, flush=True)


def artifact_fingerprint(path: Path) -> dict[str, int | str]:
    """Return a compact fingerprint for cache invalidation."""

    stat = path.stat()
    return {
        "path": str(path.relative_to(ROOT)),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def cache_key(parts: dict[str, str]) -> str:
    """Return a stable string key for one cached metric."""

    return json.dumps(parts, sort_keys=True, separators=(",", ":"))


class MetricCache:
    """Persistent cache for computed report metrics."""

    def __init__(self, path: Path, *, save_interval: int = 10) -> None:
        self.path = path
        self.save_interval = save_interval
        self.entries = self._load()
        self.dirty = False
        self.pending_writes = 0

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        with self.path.open() as file:
            payload = json.load(file)
        if not isinstance(payload, dict) or payload.get("version") != 1:
            return {}
        entries = payload.get("entries", {})
        return entries if isinstance(entries, dict) else {}

    def get(
        self,
        key: dict[str, str],
        fingerprints: list[dict[str, int | str]],
    ) -> dict[str, float] | None:
        """Return a cached metric when all artifact fingerprints still match."""

        entry = self.entries.get(cache_key(key))
        if not isinstance(entry, dict):
            return None
        if entry.get("artifacts") != fingerprints:
            return None
        metrics = entry.get("metrics")
        if not isinstance(metrics, dict):
            return None
        roc_auc = metrics.get("roc_auc")
        fpr_at_95_tpr = metrics.get("fpr_at_95_tpr")
        if not isinstance(roc_auc, int | float) or not isinstance(
            fpr_at_95_tpr,
            int | float,
        ):
            return None
        return {
            "roc_auc": float(roc_auc),
            "fpr_at_95_tpr": float(fpr_at_95_tpr),
        }

    def set(
        self,
        key: dict[str, str],
        fingerprints: list[dict[str, int | str]],
        metrics: dict[str, float],
    ) -> None:
        """Store one computed metric."""

        self.entries[cache_key(key)] = {
            "artifacts": fingerprints,
            "metrics": {
                "roc_auc": float(metrics["roc_auc"]),
                "fpr_at_95_tpr": float(metrics["fpr_at_95_tpr"]),
            },
        }
        self.dirty = True
        self.pending_writes += 1
        if self.pending_writes >= self.save_interval:
            self.save()

    def save(self) -> None:
        """Write cache entries to disk if they changed."""

        if not self.dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "entries": self.entries}
        with self.path.open("w") as file:
            json.dump(payload, file, indent=2, sort_keys=True)
        self.dirty = False
        self.pending_writes = 0


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
        self._teacher_artifact_paths = self._teacher_paths()
        self._student_artifact_paths = self._student_paths()

    def _load_manifest(self) -> dict[str, Any]:
        manifest_path = self.run_dir / "probabilities" / "manifest.json"
        with manifest_path.open() as file:
            return json.load(file)

    def _teacher_paths(self) -> dict[str, Path]:
        paths = {}
        for entry in self.manifest["artifacts"]:
            if entry["model"] == "teacher":
                paths[entry["dataset"]] = ROOT / entry["path"]
        return paths

    def _student_paths(self) -> dict[tuple[str, str], Path]:
        paths = {}
        for entry in self.manifest["artifacts"]:
            if entry["model"] == "student":
                method_key = entry.get("method") or entry.get("mode")
                paths[(entry["dataset"], method_key)] = ROOT / entry["path"]
        return paths

    def teacher_artifact_path(self, dataset_key: str) -> Path:
        """Return the path to one teacher probability artifact."""

        return self._teacher_artifact_paths[dataset_key]

    def student_artifact_path(self, dataset_key: str, method_key: str) -> Path:
        """Return the path to one student probability artifact."""

        return self._student_artifact_paths[(dataset_key, method_key)]

    def has_student_artifact(self, dataset_key: str, method_key: str) -> bool:
        """Return whether one student probability artifact exists in the manifest."""

        return (dataset_key, method_key) in self._student_artifact_paths

    @property
    def teacher_score_label(self) -> str:
        """Return the teacher MSP row label for this run."""

        teacher = self.manifest.get("teacher", {})
        hf_model_id = teacher.get("hf_model_id")
        if not isinstance(hf_model_id, str) or not hf_model_id:
            return "Teacher MSP"
        return f"{teacher_name_label(hf_model_id)} MSP"


class TeacherProbabilityRunArtifacts:
    """Loaded teacher-only probability artifacts for one teacher export run."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_name = run_dir.name
        self.manifest = self._load_manifest()
        self.id_dataset_key = f"{self.manifest['dataset']['name']}_test"
        teacher = self.manifest.get("teacher", {})
        hf_model_id = teacher.get("hf_model_id", "")
        self.teacher_label = teacher_name_label(hf_model_id)
        self._artifact_paths = self._paths()

    def _load_manifest(self) -> dict[str, Any]:
        manifest_path = self.run_dir / "teacher_probabilities" / "manifest.json"
        with manifest_path.open() as file:
            return json.load(file)

    def _paths(self) -> dict[str, Path]:
        paths = {}
        for entry in self.manifest["artifacts"]:
            dataset_key = entry["dataset"]
            declared_path = ROOT / entry["path"]
            fallback_path = (
                self.run_dir / "teacher_probabilities" / dataset_key / "probabilities.pt"
            )
            paths[dataset_key] = declared_path if declared_path.exists() else fallback_path
        return paths

    def artifact_path(self, dataset_key: str) -> Path:
        """Return the path to one teacher-only probability artifact."""

        return self._artifact_paths[dataset_key]


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
    teacher_runs: dict[tuple[str, str], TeacherProbabilityRunArtifacts],
    teacher_label: str,
    columns: list[tuple[str, str]],
    metric_cache: MetricCache,
) -> dict[tuple[str, str], str]:
    """Compute teacher MSP metrics for each report column."""

    values = {}
    total_columns = len(columns)
    for column_index, (id_dataset_key, ood_dataset_key) in enumerate(columns, start=1):
        progress(
            "teacher MSP "
            f"{column_index}/{total_columns}: "
            f"{DATASET_LABELS.get(id_dataset_key, id_dataset_key)} (ID) vs "
            f"{DATASET_LABELS.get(ood_dataset_key, ood_dataset_key)}",
        )
        run = teacher_runs.get((id_dataset_key, teacher_label))
        if run is None:
            values[(id_dataset_key, ood_dataset_key)] = ""
            continue
        values[(id_dataset_key, ood_dataset_key)] = teacher_metric(
            run,
            id_dataset_key,
            ood_dataset_key,
            metric_cache,
        )
    return values


def teacher_metric(
    run: TeacherProbabilityRunArtifacts,
    id_dataset_key: str,
    ood_dataset_key: str,
    metric_cache: MetricCache,
) -> str:
    """Compute or load one cached Teacher MSP metric table cell."""

    id_teacher_path = run.artifact_path(id_dataset_key)
    ood_teacher_path = run.artifact_path(ood_dataset_key)
    fingerprints = [
        artifact_fingerprint(id_teacher_path),
        artifact_fingerprint(ood_teacher_path),
    ]
    key = {
        "kind": "teacher_msp",
        "run": run.run_name,
        "id_dataset": id_dataset_key,
        "ood_dataset": ood_dataset_key,
    }
    cached = metric_cache.get(key, fingerprints)
    if cached is not None:
        progress("cache hit Teacher MSP")
        return format_metric(cached)

    id_scores = teacher_msp_scores_for_dataset(run, id_dataset_key)
    ood_scores = teacher_msp_scores_for_dataset(run, ood_dataset_key)
    try:
        metrics = metric_for_scores(id_scores, ood_scores)
    finally:
        del id_scores, ood_scores
        gc.collect()
    metric_cache.set(key, fingerprints, metrics)
    return format_metric(metrics)


def teacher_msp_scores_for_dataset(
    run: TeacherProbabilityRunArtifacts,
    dataset_key: str,
) -> np.ndarray:
    """Compute teacher MSP scores for one dataset artifact."""

    probabilities = load_artifact_array(
        run.artifact_path(dataset_key),
        "probabilities",
    )
    if probabilities.ndim != 2:
        msg = (
            "Teacher MSP must use raw-image teacher probabilities with shape "
            f"(n_samples, n_classes), got {probabilities.shape}."
        )
        raise ValueError(msg)
    try:
        return probabilities.max(axis=1)
    finally:
        del probabilities


def student_scores_for_dataset(
    run: RunArtifacts,
    dataset_key: str,
    method_key: str,
    score_key: str,
) -> np.ndarray:
    """Compute student-teacher OOD scores for one dataset artifact pair."""

    artifact_key = "logits" if score_key == LOGIT_L2_DISTANCE else "probabilities"
    teacher_values = load_artifact_array(
        run.teacher_artifact_path(dataset_key),
        artifact_key,
    )
    student_values = load_artifact_array(
        run.student_artifact_path(dataset_key, method_key),
        artifact_key,
    )
    try:
        if score_key == LOGIT_L2_DISTANCE:
            return logit_l2_distance(teacher_values, student_values, signed=True)
        score_function = PROBABILITY_SCORE_FUNCTIONS[score_key]
        return score_function(teacher_values, student_values)
    finally:
        del teacher_values, student_values


def student_metric(
    run: RunArtifacts,
    method_key: str,
    score_key: str,
    ood_dataset_key: str,
    metric_cache: MetricCache,
) -> str:
    """Compute one student metric table cell."""

    id_dataset_key = run.id_dataset_key
    artifact_key = "logits" if score_key == LOGIT_L2_DISTANCE else "probabilities"
    id_teacher_path = run.teacher_artifact_path(id_dataset_key)
    id_student_path = run.student_artifact_path(id_dataset_key, method_key)
    ood_teacher_path = run.teacher_artifact_path(ood_dataset_key)
    ood_student_path = run.student_artifact_path(ood_dataset_key, method_key)
    fingerprints = [
        artifact_fingerprint(id_teacher_path),
        artifact_fingerprint(id_student_path),
        artifact_fingerprint(ood_teacher_path),
        artifact_fingerprint(ood_student_path),
    ]
    key = {
        "kind": "student",
        "run": run.run_name,
        "method": method_key,
        "score": score_key,
        "artifact_key": artifact_key,
        "id_dataset": id_dataset_key,
        "ood_dataset": ood_dataset_key,
    }
    cached = metric_cache.get(key, fingerprints)
    if cached is not None:
        progress("cache hit student metric")
        return format_metric(cached)

    id_scores = student_scores_for_dataset(
        run,
        id_dataset_key,
        method_key,
        score_key,
    )
    ood_scores = student_scores_for_dataset(
        run,
        ood_dataset_key,
        method_key,
        score_key,
    )
    try:
        metrics = metric_for_scores(id_scores, ood_scores)
    finally:
        del id_scores, ood_scores
        gc.collect()
    metric_cache.set(key, fingerprints, metrics)
    return format_metric(metrics)


def id_dataset_key_from_config(config: ExperimentConfig) -> str:
    """Return the report dataset key for one config's ID dataset."""

    return f"{config.dataset_name}_test"


def available_columns(
    configs: list[ExperimentConfig],
    runs: list[RunArtifacts],
) -> list[tuple[str, str]]:
    """Return report metric columns for selected configs and available runs."""

    id_dataset_keys = ["cifar10_test", "cifar100_test"]
    columns = []
    for id_dataset_key in id_dataset_keys:
        available_ood_datasets = {
            ood_dataset_key
            for config in configs
            if id_dataset_key_from_config(config) == id_dataset_key
            for ood_dataset_key in config.ood_dataset_keys
        }
        available_ood_datasets.update(
            {
                ood_dataset_key
                for run in runs
                if run.id_dataset_key == id_dataset_key
                for ood_dataset_key in run.ood_dataset_keys
            },
        )
        if not available_ood_datasets:
            continue
        columns.extend(
            (id_dataset_key, ood_dataset_key)
            for ood_dataset_key in OOD_DATASET_ORDER_BY_ID[id_dataset_key]
            if ood_dataset_key in available_ood_datasets
        )
    return columns


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


def group_configs_by_id_dataset(
    group: ReportRowGroup,
    method_key: str,
) -> dict[str, ExperimentConfig]:
    """Return one config per ID dataset for a rendered row group and method."""

    selected: dict[str, ExperimentConfig] = {}
    for config in sorted(group.configs, key=report_sort_key):
        if method_key not in config.method_keys:
            continue
        id_dataset_key = id_dataset_key_from_config(config)
        existing = selected.get(id_dataset_key)
        if existing is not None:
            progress(
                "skipping duplicate rendered row source "
                f"{config.path}: already using {existing.path} for "
                f"{DATASET_LABELS.get(id_dataset_key, id_dataset_key)}, "
                f"{group.student_kind}/{group.feature_source}/"
                f"{group.perturbation or '-'}, "
                f"{METHOD_LABELS.get(method_key, method_key)}",
            )
            continue
        selected[id_dataset_key] = config
    return selected


def build_table_rows(
    configs: list[ExperimentConfig],
    runs: list[RunArtifacts],
    teacher_runs: dict[tuple[str, str], TeacherProbabilityRunArtifacts],
    columns: list[tuple[str, str]],
    metric_cache: MetricCache,
) -> list[str]:
    """Build LaTeX table body rows."""

    rows = []
    row_groups = report_row_groups(configs)
    total_metric_cells = sum(
        len(group_method_keys(group)) * len(SCORE_ORDER) * len(columns)
        for group in row_groups
    )
    metric_cell_index = 0
    strategy = configs[0].strategy
    teacher_labels = list(dict.fromkeys(config.teacher_label for config in configs))
    for teacher_label in teacher_labels:
        progress(f"computing {teacher_label} MSP row across {len(columns)} columns")
        teacher_values = teacher_metrics_by_column(
            teacher_runs,
            teacher_label,
            columns,
            metric_cache,
        )
        teacher_prefix = "Teacher & - & - & " + f"{teacher_label} MSP"
        if strategy == "perturbation":
            teacher_prefix = "Teacher & - & - & - & " + f"{teacher_label} MSP"
        rows.append(
            teacher_prefix
            + " & "
            + " & ".join(teacher_values[column] for column in columns)
            + r" \\",
        )

    runs_by_name = {run.run_name: run for run in runs}
    for group in row_groups:
        method_keys = group_method_keys(group)
        if not method_keys:
            progress(
                "skipping "
                f"{group.student_kind}/{group.feature_source}/"
                f"{group.perturbation or '-'}: no configured methods",
            )
            continue
        progress(
            "row group "
            f"{group.student_kind}/{group.feature_source}/{group.perturbation or '-'}: "
            f"{len(method_keys)} methods, {len(columns)} columns",
        )
        rows.append(r"\midrule")
        for method_index, method_key in enumerate(method_keys):
            progress(
                f"method {METHOD_LABELS.get(method_key, method_key)} "
                f"for {group.student_kind}/{group.feature_source}/"
                f"{group.perturbation or '-'}",
            )
            if method_index > 0:
                rows.append(r"\addlinespace")
            config_by_id_dataset = group_configs_by_id_dataset(group, method_key)
            run_by_id_dataset = {
                id_dataset_key: runs_by_name[config.experiment_name]
                for id_dataset_key, config in config_by_id_dataset.items()
                if config.experiment_name in runs_by_name
            }
            for score_index, score_key in enumerate(SCORE_ORDER):
                show_config_labels = method_index == 0 and score_index == 0
                student_label = group.student_kind if show_config_labels else ""
                feature_label = group.feature_source if show_config_labels else ""
                perturbation = group.perturbation or ""
                perturbation_label_text = (
                    perturbation if method_index == 0 and score_index == 0 else ""
                )
                method_label = METHOD_LABELS[method_key] if score_index == 0 else ""
                score_label = OOD_SCORE_LABELS[score_key]
                cells = []
                for id_dataset_key, ood_dataset_key in columns:
                    metric_cell_index += 1
                    config = config_by_id_dataset.get(id_dataset_key)
                    run = run_by_id_dataset.get(id_dataset_key)
                    if config is None or method_key not in config.method_keys:
                        progress(
                            "blank "
                            f"{metric_cell_index}/{total_metric_cells}: "
                            f"{group.student_kind}/{group.feature_source}/"
                            f"{group.perturbation or '-'}, "
                            f"{METHOD_LABELS.get(method_key, method_key)}, "
                            f"{score_label}, "
                            f"{DATASET_LABELS.get(id_dataset_key, id_dataset_key)} "
                            f"vs {DATASET_LABELS.get(ood_dataset_key, ood_dataset_key)}",
                        )
                        cells.append("")
                    elif run is None or not run.has_student_artifact(
                        id_dataset_key,
                        method_key,
                    ) or not run.has_student_artifact(
                        ood_dataset_key,
                        method_key,
                    ):
                        progress(
                            "blank "
                            f"{metric_cell_index}/{total_metric_cells}: "
                            f"{config.experiment_name}, "
                            f"{METHOD_LABELS.get(method_key, method_key)}, "
                            f"{score_label}, "
                            f"{DATASET_LABELS.get(id_dataset_key, id_dataset_key)} "
                            f"vs {DATASET_LABELS.get(ood_dataset_key, ood_dataset_key)}",
                        )
                        cells.append("")
                    else:
                        progress(
                            "cell "
                            f"{metric_cell_index}/{total_metric_cells}: "
                            f"{config.experiment_name}, "
                            f"{METHOD_LABELS.get(method_key, method_key)}, "
                            f"{score_label}, "
                            f"{DATASET_LABELS.get(id_dataset_key, id_dataset_key)} "
                            f"vs {DATASET_LABELS.get(ood_dataset_key, ood_dataset_key)}",
                        )
                        cells.append(
                            student_metric(
                                run,
                                method_key,
                                score_key,
                                ood_dataset_key,
                                metric_cache,
                            ),
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
    teacher_runs: dict[tuple[str, str], TeacherProbabilityRunArtifacts],
    metric_cache: MetricCache,
) -> str:
    """Render one LaTeX metrics table for one OOD strategy."""

    columns = available_columns(configs, runs)
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

    rows = build_table_rows(
        configs,
        runs,
        teacher_runs,
        columns,
        metric_cache,
    )
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
    title_suffix = r" ROCAUC $\uparrow$ / FPR@95 $\downarrow$"
    total_columns = len(metadata_headers) + len(columns)

    return rf"""\begin{{tabular}}{{{column_spec}}}
\multicolumn{{{total_columns}}}{{c}}{{\textbf{{{latex_escape(strategy_title)}{title_suffix}}}}}\\[4pt]
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
    """Group configs by OOD strategy when the strategy has local artifacts."""

    configs = deduplicate_configs(configs)
    available_experiment_names = {
        config.experiment_name for config in available_configs(configs, warn=False)
    }
    by_strategy: dict[str, list[ExperimentConfig]] = {}
    strategies_with_artifacts = {
        config.strategy
        for config in configs
        if config.experiment_name in available_experiment_names
    }
    for config in configs:
        if config.strategy not in strategies_with_artifacts:
            continue
        if config.experiment_name not in available_experiment_names:
            print(
                "Skipping "
                f"{config.path}: missing probability artifacts in {config.run_dir}",
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


def build_latex_document(
    strategy: str,
    configs: list[ExperimentConfig],
    teacher_runs: dict[tuple[str, str], TeacherProbabilityRunArtifacts],
    metric_cache: MetricCache,
) -> str:
    """Render one standalone LaTeX metrics table."""

    runs = [
        RunArtifacts(config.run_dir)
        for config in configs
        if run_artifacts_available(config)
    ]
    body = build_strategy_table(
        strategy,
        configs,
        runs,
        teacher_runs,
        metric_cache,
    )
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
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help=(
            "Path to the persistent OOD metric cache. Cached cells are reused "
            "when source artifact paths, sizes, and mtimes match."
        ),
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
    teacher_runs = teacher_reference_runs(configs)
    metric_cache = MetricCache(args.cache)

    if args.output is not None and len(configs_by_strategy) != 1:
        msg = "--output can only be used when one OOD strategy is selected."
        raise ValueError(msg)

    for strategy, strategy_configs in configs_by_strategy.items():
        output_path = args.output or output_path_for_strategy(args.output_dir, strategy)
        latex = build_latex_document(
            strategy,
            strategy_configs,
            teacher_runs,
            metric_cache,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(latex)
        metric_cache.save()
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
