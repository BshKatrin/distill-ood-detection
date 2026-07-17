"""Export OOD metrics for Feature Denoising reconstruction score artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from distill_ood_detection.evaluation.ood_metrics import ood_detection_metrics

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = ROOT / "reports" / "outputs" / "latex" / "metrics_feature_denoising.tex"
DEFAULT_JSON_OUTPUT = ROOT / "reports" / "outputs" / "json" / "feature_denoising_ood_metrics.json"
SCORES = (
    "default",
    "relative_improvement",
    "improvement",
    "cosine_similarity",
)

DATASET_LABELS = {
    "cifar10_test": "CIFAR-10",
    "cifar100_test": "CIFAR-100",
    "mnist_test": "MNIST",
    "svhn_test": "SVHN",
}


def main() -> None:
    """Run the Feature Denoising OOD metric exporter."""

    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="*", type=Path, default=[ROOT / "configs" / "students" / "feature_denoising"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument(
        "--score",
        choices=SCORES,
        default="default",
        help=(
            "Score to compute from Feature Denoising artifacts. "
            "'default' uses the exported signed reconstruction score."
        ),
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-12,
        help="Numerical floor for relative_improvement denominator.",
    )
    args = parser.parse_args()

    configs = available_configs(expand_config_paths(args.configs))
    rows = [metrics_for_config(config, args.score, args.eps) for config in configs]
    write_json(args.json_output, {"version": 1, "runs": rows})
    write_latex(args.output, rows)


def expand_config_paths(paths: Iterable[Path]) -> list[Path]:
    """Expand config files and directories into YAML config paths."""

    config_paths: list[Path] = []
    for path in paths:
        path = path if path.is_absolute() else ROOT / path
        if path.is_dir():
            config_paths.extend(sorted(path.rglob("*.yaml")))
            config_paths.extend(sorted(path.rglob("*.yml")))
        else:
            config_paths.append(path)
    return sorted(dict.fromkeys(config_paths))


def available_configs(config_paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Return Feature Denoising configs with local score artifacts."""

    configs = []
    for path in config_paths:
        with path.open() as file:
            config = yaml.safe_load(file)
        if config.get("strategy", {}).get("name") not in {"feature_denoising", "jepa"}:
            continue
        if score_artifact_dir(config).is_none:
            print(f"Skipping {path}: missing Feature Denoising score artifacts", file=sys.stderr)
            continue
        config["_path"] = str(path)
        configs.append(config)
    return configs


def metrics_for_config(config: dict[str, Any], score: str, eps: float) -> dict[str, Any]:
    """Compute OOD metrics for one Feature Denoising run."""

    run_dir = ROOT / config["run_dir"]
    score_dir = score_artifact_dir(config).path
    id_key = f"{config['dataset']['name']}_test"
    id_artifact = load_score_artifact(score_dir / id_key / "student_best.pt")
    id_scores, score_name = artifact_scores(id_artifact, score, eps)
    ood_results = []
    for ood_dataset in config["dataset"].get("ood_datasets", []):
        ood_key = f"{ood_dataset['name']}_{ood_dataset['split']}"
        ood_artifact = load_score_artifact(score_dir / ood_key / "student_best.pt")
        ood_scores, _ = artifact_scores(ood_artifact, score, eps)
        labels = np.concatenate(
            [
                np.ones_like(id_scores, dtype=int),
                np.zeros_like(ood_scores, dtype=int),
            ]
        )
        scores = np.concatenate([id_scores, ood_scores])
        ood_results.append(
            {
                "ood_dataset": ood_key,
                "metrics": ood_detection_metrics(labels, scores),
            }
        )
    return {
        "experiment_name": config["experiment_name"],
        "run_dir": config["run_dir"],
        "config_path": config["_path"],
        "id_dataset": id_key,
        "score": score_name,
        "ood": ood_results,
    }


class ScoreArtifactDir:
    """Resolved Feature Denoising score artifact directory."""

    def __init__(self, path: Path | None) -> None:
        self.path = path or Path()
        self.is_none = path is None


def score_artifact_dir(config: dict[str, Any]) -> ScoreArtifactDir:
    """Return the available score artifact directory for new or legacy runs."""

    run_dir = ROOT / config["run_dir"]
    for name in ("feature_denoising_scores", "jepa_scores"):
        path = run_dir / name
        if path.is_dir():
            return ScoreArtifactDir(path)
    return ScoreArtifactDir(None)


def artifact_scores(
    artifact: dict[str, Any],
    score: str,
    eps: float,
) -> tuple[np.ndarray, str]:
    """Return one ID-oriented score vector from a score artifact."""

    if score == "default":
        score_name = artifact.get("metadata", {}).get("score", "feature_denoising_pca_reconstruction_error")
        return to_numpy(artifact["scores"]), str(score_name)
    if score == "relative_improvement":
        identity_error = to_numpy(required_artifact_value(artifact, "identity_error"))
        reconstruction_error = to_numpy(required_artifact_value(artifact, "raw_reconstruction_error"))
        denominator = np.maximum(identity_error, eps)
        return (identity_error - reconstruction_error) / denominator, "relative_improvement"
    if score == "improvement":
        return to_numpy(required_artifact_value(artifact, "improvement")), "improvement"
    if score == "cosine_similarity":
        return to_numpy(required_artifact_value(artifact, "cosine_similarity")), "cosine_similarity"
    raise ValueError(f"Unsupported score: {score}")


def required_artifact_value(artifact: dict[str, Any], key: str) -> Any:
    """Return a required value from a Feature Denoising artifact."""

    if key not in artifact:
        raise KeyError(f"Artifact does not contain required score component: {key}")
    return artifact[key]


def load_score_artifact(path: Path) -> dict[str, Any]:
    """Load one Feature Denoising score artifact."""

    if not path.exists():
        raise FileNotFoundError(f"Missing Feature Denoising score artifact: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def to_numpy(value: Any) -> np.ndarray:
    """Convert a tensor-like value to NumPy."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)


def write_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a compact LaTeX table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Run & OOD Dataset & ROC-AUC / FPR@95 \\",
        r"\midrule",
    ]
    for row in rows:
        run = latex_escape(row["experiment_name"])
        for ood in row["ood"]:
            dataset = latex_escape(DATASET_LABELS.get(ood["ood_dataset"], ood["ood_dataset"]))
            metrics = ood["metrics"]
            value = f"{metrics['roc_auc']:.2f} / {metrics['fpr_at_95_tpr']:.2f}"
            lines.append(f"{run} & {dataset} & {value} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def latex_escape(text: str) -> str:
    """Escape LaTeX table text."""

    return text.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


if __name__ == "__main__":
    main()
