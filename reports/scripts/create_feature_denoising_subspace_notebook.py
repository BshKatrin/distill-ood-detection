"""Create a notebook for Feature Denoising activation-subspace errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat
import yaml

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = (
    ROOT
    / "reports"
    / "templates"
    / "notebooks"
    / "feature_denoising_subspace_errors.ipynb"
)
DEFAULT_OUTPUT = (
    ROOT
    / "reports"
    / "outputs"
    / "notebooks"
    / "feature_denoising_layer4_activation_subspaces.ipynb"
)


def main() -> None:
    """Generate the activation-subspace distribution notebook."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--title",
        default="Feature Denoising Activation Subspaces",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output notebook already exists: {args.output}")

    run_dirs = [_run_dir(path) for path in args.configs]
    for run_dir in run_dirs:
        manifest = run_dir / "feature_denoising_subspace_errors" / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(f"Missing activation-subspace manifest: {manifest}")

    notebook = nbformat.read(TEMPLATE, as_version=4)
    replacements = {
        "{{ notebook_title }}": args.title,
        "{{ run_dirs }}": json.dumps(
            [str(path.relative_to(ROOT)) for path in run_dirs],
            indent=4,
        ),
    }
    for cell in notebook.cells:
        for placeholder, value in replacements.items():
            cell.source = cell.source.replace(placeholder, value)
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
    notebook.metadata.setdefault("distill_ood_detection", {})
    notebook.metadata["distill_ood_detection"].update(
        {
            "generated_from": str(TEMPLATE.relative_to(ROOT)),
            "run_dirs": [str(path.relative_to(ROOT)) for path in run_dirs],
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, args.output)
    print(f"Wrote {args.output}")


def _run_dir(config_path: Path) -> Path:
    path = config_path if config_path.is_absolute() else ROOT / config_path
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return ROOT / config["run_dir"]


if __name__ == "__main__":
    main()
