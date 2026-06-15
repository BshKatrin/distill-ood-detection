"""Utilities for generating report notebooks from templates."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
RUNS_DIR = ROOT / "runs"
NOTEBOOK_TEMPLATE = (
    REPORTS_DIR / "templates" / "notebooks" / "probability_artifact_analysis.ipynb"
)
PERTURBATION_NOTEBOOK_TEMPLATE = (
    REPORTS_DIR
    / "templates"
    / "notebooks"
    / "perturbation_probability_artifact_analysis.ipynb"
)
NOTEBOOK_OUTPUT_DIR = REPORTS_DIR / "outputs" / "notebooks"


def experiment_title(experiment_name: str) -> str:
    """Return a readable notebook title for an experiment name."""

    words = experiment_name.replace("_", " ").split()
    dataset = next(
        (
            word.upper().replace("CIFAR", "CIFAR-")
            for word in words
            if word.startswith("cifar")
        ),
        "",
    )
    model = "ResNet-18" if "resnet18" in words else " ".join(words).title()

    if experiment_name.startswith("perturbation_"):
        clip_label = ""
        match = re.search(r"_clip_([a-z0-9_]+)$", experiment_name)
        if match:
            clip_label = " Clip " + match.group(1).replace("_", " ").title()
        return f"{model} {dataset} Perturbation Linear Layer3{clip_label}".strip()

    if experiment_name.startswith("linear_student_"):
        student = "Linear Students"
    elif experiment_name.startswith("mlp_student_"):
        student = "MLP Students"
    elif experiment_name.startswith("feature_mlp_layer3_student_"):
        student = "Layer3 Feature Students"
    elif experiment_name.startswith("feature_random_forest_layer3_student_"):
        student = "Random Forest Students"
    else:
        student = "Students"

    return f"{model} {dataset} Teacher vs {student}".strip()


def filename_slug(experiment_name: str) -> str:
    """Return the report notebook filename slug for an experiment name."""

    words = experiment_name.split("_")
    dataset = next((word for word in words if word.startswith("cifar")), None)
    backbone = next((word for word in words if word.startswith("resnet")), None)

    if backbone is None or dataset is None:
        return f"{experiment_name}_probabilities"

    if experiment_name.startswith("linear_student_"):
        descriptor = "linear"
    elif experiment_name.startswith("mlp_student_"):
        descriptor = "mlp"
    elif experiment_name.startswith("feature_mlp_layer3_student_"):
        descriptor = "feature_mlp_layer3"
    elif experiment_name.startswith("feature_random_forest_layer3_student_"):
        descriptor = "random_forest"
    elif experiment_name.startswith("perturbation_linear_layer3_student_"):
        descriptor = "perturbation_linear_layer3"
        match = re.search(r"_clip_([a-z0-9_]+)$", experiment_name)
        if match:
            descriptor = f"{descriptor}_clip_{match.group(1)}"
    else:
        descriptor = experiment_name

    return f"{backbone}_{dataset}_{descriptor}_probabilities"


def next_notebook_number(output_dir: Path = NOTEBOOK_OUTPUT_DIR) -> int:
    """Return the next numeric report notebook prefix."""

    max_number = 0
    for path in output_dir.glob("*.ipynb"):
        match = re.match(r"(\d+)_", path.name)
        if match:
            max_number = max(max_number, int(match.group(1)))
    return max_number + 1


def render_notebook_template(
    *,
    experiment_name: str,
    output_path: Path,
    notebook_title: str | None = None,
    template_path: Path | None = None,
) -> None:
    """Render a probability-artifact notebook from the notebook template."""

    manifest_path = RUNS_DIR / experiment_name / "probabilities" / "manifest.json"
    if not manifest_path.exists():
        msg = f"Missing probability manifest: {manifest_path}"
        raise FileNotFoundError(msg)

    title = notebook_title or experiment_title(experiment_name)
    selected_template = template_path or template_for_experiment(experiment_name)
    notebook = nbformat.read(selected_template, as_version=4)
    for cell in notebook.cells:
        if isinstance(cell.source, str):
            cell.source = cell.source.replace("{{ experiment_name }}", experiment_name)
            cell.source = cell.source.replace("{{ notebook_title }}", title)
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    notebook.metadata.setdefault("distill_ood_detection", {})
    notebook.metadata["distill_ood_detection"].update(
        {
            "generated_from": str(selected_template.relative_to(ROOT)),
            "experiment_name": experiment_name,
        },
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output_path)


def template_for_experiment(experiment_name: str) -> Path:
    """Return the notebook template path for an experiment name."""

    if experiment_name.startswith("perturbation_"):
        return PERTURBATION_NOTEBOOK_TEMPLATE
    return NOTEBOOK_TEMPLATE
