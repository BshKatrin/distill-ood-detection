"""Export PCA explained-variance statistics and a plotting notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ACTIVATION_PATH = Path(
    "runs/teacher_resnet18_cifar10/teacher_activations/cifar10_train/layer4.pt"
)
NOTEBOOK_TEMPLATE_PATH = (
    ROOT / "reports" / "templates" / "notebooks" / "pca_explained_variance.ipynb"
)
NOTEBOOK_OUTPUT_DIR = ROOT / "reports" / "outputs" / "notebooks"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "activation_path",
        nargs="?",
        type=Path,
        default=DEFAULT_ACTIVATION_PATH,
        help="Training-split teacher activation artifact.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="JSON file to write. Defaults beside the activation artifact.",
    )
    parser.add_argument(
        "--max-components",
        type=int,
        default=512,
        help="Number of leading principal components to estimate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the randomized SVD solver.",
    )
    parser.add_argument(
        "--power-iterations",
        type=int,
        default=4,
        help="Power iterations used by randomized SVD.",
    )
    parser.add_argument(
        "--notebook-output-dir",
        type=Path,
        default=NOTEBOOK_OUTPUT_DIR,
        help="Directory where the generated notebook will be written.",
    )
    parser.add_argument(
        "--number",
        type=int,
        help="Numeric notebook prefix. Defaults to the next available number.",
    )
    parser.add_argument(
        "--title",
        help="Notebook title. Defaults to one derived from artifact metadata.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the generated notebook if it already exists.",
    )
    return parser.parse_args()


def load_flattened_activations(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    """Load and validate a complete training-split activation artifact."""

    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if artifact.get("split") != "train":
        raise ValueError(
            f"PCA statistics require a training-split artifact; got "
            f"split={artifact.get('split')!r}: {path}"
        )
    activations = artifact.get("activations")
    if not torch.is_tensor(activations):
        raise ValueError(f"Activation artifact is missing tensor 'activations': {path}")
    if activations.ndim < 2:
        raise ValueError(
            f"Activations must include sample and feature dimensions; got "
            f"shape={tuple(activations.shape)}"
        )

    values = torch.flatten(activations.float(), start_dim=1).numpy()
    metadata = {
        "dataset": artifact.get("dataset"),
        "split": artifact.get("split"),
        "layer": artifact.get("layer"),
        "activation_shape": list(activations.shape[1:]),
    }
    return values, metadata


def explained_variance_statistics(
    values: np.ndarray,
    *,
    max_components: int,
    seed: int,
    power_iterations: int,
) -> dict[str, object]:
    """Estimate leading PCA explained-variance statistics."""

    max_rank = min(values.shape)
    if not 1 <= max_components <= max_rank:
        raise ValueError(
            f"max_components must be between 1 and {max_rank}; got {max_components}"
        )
    if power_iterations < 0:
        raise ValueError("power_iterations must be non-negative")

    pca = PCA(
        n_components=max_components,
        svd_solver="randomized",
        random_state=seed,
        iterated_power=power_iterations,
    )
    pca.fit(values)
    ratios = pca.explained_variance_ratio_
    return {
        "n_samples": int(values.shape[0]),
        "n_features": int(values.shape[1]),
        "max_components": max_components,
        "solver": "randomized",
        "seed": seed,
        "power_iterations": power_iterations,
        "component": list(range(1, max_components + 1)),
        "explained_variance_ratio": ratios.tolist(),
        "cumulative_explained_variance_ratio": np.cumsum(ratios).tolist(),
    }


def default_statistics_path(activation_path: Path) -> Path:
    """Return the default JSON statistics path for an activation artifact."""

    return activation_path.with_name(
        f"{activation_path.stem}_pca_explained_variance.json"
    )


def next_notebook_number(output_dir: Path) -> int:
    """Return the next numeric notebook prefix in an output directory."""

    numbers = []
    for path in output_dir.glob("*.ipynb"):
        prefix, separator, _ = path.name.partition("_")
        if separator and prefix.isdigit():
            numbers.append(int(prefix))
    return max(numbers, default=0) + 1


def notebook_slug(metadata: dict[str, object]) -> str:
    """Return a filename slug derived from activation metadata."""

    dataset = str(metadata.get("dataset") or "dataset").lower().replace("-", "")
    layer = str(metadata.get("layer") or "activations").lower().replace("-", "_")
    return f"{dataset}_{layer}_pca_explained_variance"


def notebook_title(metadata: dict[str, object]) -> str:
    """Return a readable title derived from activation metadata."""

    dataset = str(metadata.get("dataset") or "Dataset").replace("_", " ").upper()
    layer = str(metadata.get("layer") or "activations").replace("_", " ").title()
    return f"{dataset} {layer} PCA Explained Variance"


def render_notebook(
    *,
    statistics_path: Path,
    activation_path: Path,
    output_path: Path,
    title: str,
    template_path: Path = NOTEBOOK_TEMPLATE_PATH,
) -> None:
    """Render a PCA explained-variance notebook from the template."""

    try:
        notebook_statistics_path = statistics_path.resolve().relative_to(ROOT)
    except ValueError:
        notebook_statistics_path = statistics_path.resolve()
    notebook = json.loads(template_path.read_text())
    replacements = {
        "{{ notebook_title }}": title,
        "{{ statistics_path }}": json.dumps(str(notebook_statistics_path)),
        "{{ activation_path }}": str(activation_path),
    }
    for cell in notebook["cells"]:
        source = cell["source"]
        if isinstance(source, list):
            source = [
                _replace_placeholders(line, replacements) for line in source
            ]
        else:
            source = _replace_placeholders(source, replacements)
        cell["source"] = source
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    notebook["metadata"].setdefault("distill_ood_detection", {})
    notebook["metadata"]["distill_ood_detection"].update(
        {
            "generated_from": str(template_path.relative_to(ROOT)),
            "activation_path": str(activation_path),
            "statistics_path": str(statistics_path),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(notebook, indent=1) + "\n")


def _replace_placeholders(source: str, replacements: dict[str, str]) -> str:
    """Replace notebook template placeholders in one source string."""

    for placeholder, value in replacements.items():
        source = source.replace(placeholder, value)
    return source


def main() -> None:
    """Export explained-variance statistics and a plotting notebook."""

    args = parse_args()
    values, artifact_metadata = load_flattened_activations(args.activation_path)
    statistics = explained_variance_statistics(
        values,
        max_components=args.max_components,
        seed=args.seed,
        power_iterations=args.power_iterations,
    )
    output = {
        "activation_path": str(args.activation_path),
        **artifact_metadata,
        **statistics,
    }
    statistics_path = args.output_path or default_statistics_path(args.activation_path)
    statistics_path.parent.mkdir(parents=True, exist_ok=True)
    statistics_path.write_text(json.dumps(output, indent=2) + "\n")

    number = args.number or next_notebook_number(args.notebook_output_dir)
    notebook_output_path = (
        args.notebook_output_dir
        / f"{number:02d}_{notebook_slug(artifact_metadata)}.ipynb"
    )
    if notebook_output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output notebook already exists: {notebook_output_path}")
    render_notebook(
        statistics_path=statistics_path,
        activation_path=args.activation_path,
        output_path=notebook_output_path,
        title=args.title or notebook_title(artifact_metadata),
    )

    final_ratio = output["cumulative_explained_variance_ratio"][-1]
    print(f"Wrote {statistics_path}")
    print(f"Wrote {notebook_output_path}")
    print(
        f"The first {args.max_components} components explain "
        f"{100 * final_ratio:.2f}% of variance."
    )


if __name__ == "__main__":
    main()
