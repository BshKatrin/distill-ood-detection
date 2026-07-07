"""Export final Feature Denoising reconstruction losses from saved metrics artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = ROOT / "reports" / "outputs" / "latex" / "metrics_test_feature_denoising.tex"
DEFAULT_JSON_OUTPUT = ROOT / "reports" / "outputs" / "json" / "feature_denoising_reconstruction_losses.json"
METHOD_DIR = "pca_masked_reconstruction"


def main() -> None:
    """Run the Feature Denoising reconstruction loss exporter."""

    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="*", type=Path, default=[ROOT / "configs" / "students" / "feature_denoising"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    args = parser.parse_args()

    rows = [losses_for_config(config) for config in available_configs(expand_config_paths(args.configs))]
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
    """Return Feature Denoising configs with local reconstruction metrics."""

    configs = []
    for path in config_paths:
        with path.open() as file:
            config = yaml.safe_load(file)
        if config.get("strategy", {}).get("name") != "feature_denoising":
            continue
        metrics_path = ROOT / config["run_dir"] / METHOD_DIR / "metrics.json"
        if not metrics_path.exists():
            print(f"Skipping {path}: missing Feature Denoising reconstruction metrics", file=sys.stderr)
            continue
        config["_path"] = str(path)
        configs.append(config)
    return configs


def losses_for_config(config: dict[str, Any]) -> dict[str, Any]:
    """Load final reconstruction losses for one config."""

    metrics_path = ROOT / config["run_dir"] / METHOD_DIR / "metrics.json"
    with metrics_path.open(encoding="utf-8") as file:
        metrics = json.load(file)
    return {
        "experiment_name": config["experiment_name"],
        "run_dir": config["run_dir"],
        "config_path": config["_path"],
        "best_validation_reconstruction_loss": metrics.get("best_validation_reconstruction_loss"),
        "final_validation_reconstruction_loss": metrics.get("final_validation_reconstruction_loss"),
        "test_reconstruction_loss": metrics.get("test_reconstruction_loss"),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)


def write_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a compact LaTeX table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Run & Best Val. & Final Val. & Test \\",
        r"\midrule",
    ]
    for row in rows:
        run = latex_escape(row["experiment_name"])
        best = format_loss(row["best_validation_reconstruction_loss"])
        final = format_loss(row["final_validation_reconstruction_loss"])
        test = format_loss(row["test_reconstruction_loss"])
        lines.append(f"{run} & {best} & {final} & {test} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_loss(value: Any) -> str:
    """Format one loss value."""

    if isinstance(value, int | float):
        return f"{float(value):.6f}"
    return "--"


def latex_escape(text: str) -> str:
    """Escape LaTeX table text."""

    return text.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


if __name__ == "__main__":
    main()
