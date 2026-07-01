"""Plot compact OOD-averaged PCA variant sweeps for report figures."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Literal

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
JSON_DIR = ROOT / "reports" / "outputs" / "json"
PLOTS_DIR = ROOT / "reports" / "outputs" / "plots"

DEFAULT_JSON_PATHS = (
    JSON_DIR / "pca_reduced_components.json",
    JSON_DIR / "perturbation_20260623_ood_metrics_cifar10.json",
    JSON_DIR / "perturbation_20260623_ood_metrics_cifar100.json",
    JSON_DIR / "unmasked_pca_resnet18_reduced_components_ood_metrics.json",
    JSON_DIR / "unmasked_pca_resnet18_pca100_completed_ood_metrics.json",
    JSON_DIR / "masked_pca_fixed_ood_metrics.json",
    JSON_DIR / "masked_pca_resnet50_cifar10_ood_metrics.json",
)

DATASET_LABELS = {
    "cifar10_test": "CIFAR-10",
    "cifar100_test": "CIFAR-100",
    "mnist_test": "MNIST",
    "svhn_test": "SVHN",
}

OBJECTIVE_LABELS = {
    "cross_entropy": "Cross-entropy",
    "kl_divergence": "KL divergence",
    "mse_logits": "Logit MSE",
}

OOD_SCORE_LABELS = {
    "absolute_max_probability_difference": "Abs. Max Diff",
    "student_teacher_kl_divergence": "Student-teacher KL",
    "student_msp": "Student MSP",
    "student_energy": "Student Energy",
}

SELECTED_OOD_SCORES = (
    "absolute_max_probability_difference",
    "student_teacher_kl_divergence",
    "student_msp",
    "student_energy",
)

Variant = Literal["masked", "unmasked"]

sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 14,
    }
)


@dataclass(frozen=True)
class SweepSelection:
    """The internally comparable sweep selected for one ID dataset."""

    id_dataset: str
    architecture: str
    pca_components: int | None
    teacher_perturbed: bool
    method: str
    probability_mode: str
    hyperparameter_values: tuple[float, ...]


def load_reports(paths: tuple[Path, ...]) -> dict[str, Any]:
    """Load and merge report JSON files by run name and metric identity."""

    runs_by_name: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            print(f"Skipping missing JSON report: {path}")
            continue

        with path.open(encoding="utf-8") as file:
            data = json.load(file)

        for run in data.get("runs", []):
            if not isinstance(run, dict):
                continue
            run_name = run.get("run_name")
            if not isinstance(run_name, str):
                continue

            merged_run = runs_by_name.setdefault(
                run_name,
                {key: value for key, value in run.items() if key != "metrics"},
            )
            merged_run.update({key: value for key, value in run.items() if key != "metrics"})

            metrics_by_key = {
                metric_key(metric): metric
                for metric in merged_run.get("metrics", [])
                if isinstance(metric, dict)
            }
            for metric in run.get("metrics", []):
                if isinstance(metric, dict):
                    metrics_by_key[metric_key(metric)] = metric
            merged_run["metrics"] = list(metrics_by_key.values())

    return {"runs": list(runs_by_name.values()), "version": 1}


def metric_key(metric: dict[str, Any]) -> tuple[Any, ...]:
    """Return the merge key for one metric record."""

    return (
        metric.get("id_dataset"),
        metric.get("ood_dataset"),
        metric.get("ood_score"),
        metric.get("method"),
        metric.get("probability_mode"),
    )


def extract_architecture(run_name: str) -> str | None:
    """Extract the architecture token from a run name."""

    match = re.search(r"resnet\d+", run_name)
    return match.group(0) if match else None


def extract_pca_components(run: dict[str, Any]) -> int | None:
    """Extract the PCA component count from metadata or run name."""

    value = run.get("pca_components")
    if value is not None:
        return int(value)
    match = re.search(r"_pca(\d+)(?:_|$)", str(run.get("run_name", "")))
    return int(match.group(1)) if match else None


def extract_mask_probability(run: dict[str, Any]) -> float | None:
    """Extract the PCA mask probability from metadata or run name."""

    value = run.get("pca_mask_probability")
    if value is not None:
        return float(value)
    match = re.search(r"_mask_p(\d{3})(?:_|$)", str(run.get("run_name", "")))
    return int(match.group(1)) / 100 if match else None


def extract_rows(data: dict[str, Any]) -> pd.DataFrame:
    """Extract PCA OOD metrics into a plot-ready DataFrame."""

    rows: list[dict[str, Any]] = []
    for run in data.get("runs", []):
        if not isinstance(run, dict):
            continue

        run_name = str(run.get("run_name", ""))
        if "_pca" not in run_name:
            continue

        is_masked = "_mask_p" in run_name
        variant: Variant = "masked" if is_masked else "unmasked"
        pca_components = extract_pca_components(run)
        mask_probability = extract_mask_probability(run)
        hyperparameter_value = mask_probability if is_masked else pca_components
        architecture = extract_architecture(run_name)
        id_dataset = str(run.get("id_dataset", ""))
        teacher_perturbed = run_name.endswith("_perturbed")

        if architecture is None or pca_components is None or hyperparameter_value is None:
            continue

        for metric in run.get("metrics", []):
            if not isinstance(metric, dict):
                continue
            ood_score = str(metric.get("ood_score"))
            if ood_score not in SELECTED_OOD_SCORES:
                continue
            if metric.get("roc_auc") is None or metric.get("fpr_at_95_tpr") is None:
                continue

            roc_auc = float(metric["roc_auc"])
            fpr_at_95_tpr = float(metric["fpr_at_95_tpr"])
            rows.append(
                {
                    "variant": variant,
                    "run_name": run_name,
                    "id_dataset": id_dataset,
                    "architecture": architecture,
                    "pca_components": pca_components,
                    "mask_probability": mask_probability,
                    "teacher_perturbed": teacher_perturbed,
                    "hyperparameter_value": float(hyperparameter_value),
                    "method": str(metric["method"]),
                    "probability_mode": str(metric["probability_mode"]),
                    "ood_dataset": str(metric["ood_dataset"]),
                    "ood_score": ood_score,
                    "roc_auc": roc_auc,
                    "fpr_at_95_tpr": fpr_at_95_tpr,
                    "fpr_at_95_tpr_complement": 1.0 - fpr_at_95_tpr,
                    "balanced_metric": (roc_auc + (1.0 - fpr_at_95_tpr)) / 2,
                }
            )

    df = pd.DataFrame(rows)
    print(f"Extracted PCA metric rows: {len(df)}")
    return df


def sweep_identity_columns(variant: Variant) -> list[str]:
    """Return columns that define an internally comparable sweep."""

    if variant == "masked":
        return ["variant", "id_dataset", "architecture", "pca_components", "teacher_perturbed"]
    return ["variant", "id_dataset", "architecture", "teacher_perturbed"]


def select_sweeps(df: pd.DataFrame, variant: Variant) -> list[SweepSelection]:
    """Select the widest hyperparameter sweep and best objective for each ID dataset."""

    variant_df = df[df["variant"] == variant]
    if variant_df.empty:
        return []

    identity_columns = sweep_identity_columns(variant)
    coverage = (
        variant_df.groupby(identity_columns, dropna=False)
        .agg(
            hyperparameter_count=("hyperparameter_value", "nunique"),
            row_count=("hyperparameter_value", "size"),
        )
        .reset_index()
    )
    coverage = coverage[coverage["hyperparameter_count"] >= 2]

    selections: list[SweepSelection] = []
    for id_dataset in sorted(coverage["id_dataset"].dropna().unique()):
        id_coverage = coverage[coverage["id_dataset"] == id_dataset].sort_values(
            ["hyperparameter_count", "row_count"],
            ascending=[False, False],
        )
        if id_coverage.empty:
            print(f"No {variant} {id_dataset} sweep has at least two hyperparameter values.")
            continue

        chosen = id_coverage.iloc[0]
        mask = pd.Series(True, index=variant_df.index)
        for column in identity_columns:
            mask &= variant_df[column] == chosen[column]
        sweep_df = variant_df[mask]

        objective_summary = (
            sweep_df.groupby(["method", "probability_mode"], as_index=False)
            .agg(
                mean_balanced_metric=("balanced_metric", "mean"),
                hyperparameter_count=("hyperparameter_value", "nunique"),
                row_count=("hyperparameter_value", "size"),
            )
            .sort_values(
                ["hyperparameter_count", "mean_balanced_metric", "row_count"],
                ascending=[False, False, False],
            )
        )
        best_objective = objective_summary.iloc[0]
        objective_df = sweep_df[
            (sweep_df["method"] == best_objective["method"])
            & (sweep_df["probability_mode"] == best_objective["probability_mode"])
        ]
        hyperparameter_values = tuple(
            float(value) for value in sorted(objective_df["hyperparameter_value"].dropna().unique())
        )

        selections.append(
            SweepSelection(
                id_dataset=str(chosen["id_dataset"]),
                architecture=str(chosen["architecture"]),
                pca_components=(
                    int(chosen["pca_components"]) if variant == "masked" else None
                ),
                teacher_perturbed=bool(chosen["teacher_perturbed"]),
                method=str(best_objective["method"]),
                probability_mode=str(best_objective["probability_mode"]),
                hyperparameter_values=hyperparameter_values,
            )
        )

    return selections


def selection_mask(df: pd.DataFrame, variant: Variant, selection: SweepSelection) -> pd.Series:
    """Return rows belonging to one selected sweep and objective."""

    mask = (
        (df["variant"] == variant)
        & (df["id_dataset"] == selection.id_dataset)
        & (df["architecture"] == selection.architecture)
        & (df["teacher_perturbed"] == selection.teacher_perturbed)
        & (df["method"] == selection.method)
        & (df["probability_mode"] == selection.probability_mode)
    )
    if variant == "masked":
        mask &= df["pca_components"] == selection.pca_components
    return mask


def format_hyperparameter(value: float, variant: Variant) -> str:
    """Format one hyperparameter tick label."""

    if variant == "masked":
        return f"{value:.1f}"
    return str(int(value))


def plot_variant(df: pd.DataFrame, variant: Variant) -> None:
    """Plot one compact PCA variant sweep figure."""

    selections = select_sweeps(df, variant)
    if not selections:
        print(f"No plottable {variant} PCA sweeps found.")
        return

    nrows = len(selections)
    ncols = len(SELECTED_OOD_SCORES)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 3.5 * nrows),
        sharey=True,
        squeeze=False,
    )
    metric_styles = {
        "roc_auc": {
            "label": "ROC AUC",
            "color": "#1f77b4",
            "marker": "o",
            "linestyle": "-",
        },
        "fpr_at_95_tpr_complement": {
            "label": "1 - FPR@95",
            "color": "#d62728",
            "marker": "s",
            "linestyle": "--",
        },
    }

    for row_index, selection in enumerate(selections):
        selected_df = df[selection_mask(df, variant, selection)]
        averaged = (
            selected_df.groupby(["ood_score", "hyperparameter_value"], as_index=False)
            .agg(
                roc_auc=("roc_auc", "mean"),
                fpr_at_95_tpr_complement=("fpr_at_95_tpr_complement", "mean"),
                ood_dataset_count=("ood_dataset", "nunique"),
            )
            .sort_values(["ood_score", "hyperparameter_value"])
        )

        for col_index, ood_score in enumerate(SELECTED_OOD_SCORES):
            ax = axes[row_index, col_index]
            score_df = averaged[averaged["ood_score"] == ood_score]
            for metric_key, style in metric_styles.items():
                ax.plot(
                    score_df["hyperparameter_value"],
                    score_df[metric_key],
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=2,
                    markersize=5,
                    label=style["label"],
                )

            if row_index == 0:
                ax.set_title(OOD_SCORE_LABELS.get(ood_score, ood_score))
            ax.set_ylim(0, 1)
            ax.set_xticks(selection.hyperparameter_values)
            ax.set_xticklabels(
                [format_hyperparameter(value, variant) for value in selection.hyperparameter_values]
            )
            ax.grid(True, which="major", axis="both", alpha=0.35)

            if row_index == nrows - 1:
                ax.set_xlabel("Mask probability" if variant == "masked" else "PCA components")
            if col_index == 0:
                id_label = DATASET_LABELS.get(selection.id_dataset, selection.id_dataset)
                objective_label = OBJECTIVE_LABELS.get(selection.method, selection.method)
                target_label = "perturbed target" if selection.teacher_perturbed else "clean target"
                component_label = (
                    f", k={selection.pca_components}" if variant == "masked" else ""
                )
                row_label = (
                    f"{id_label}\n{selection.architecture}{component_label}, {target_label}\n"
                    f"{objective_label}, {selection.probability_mode}"
                )
                ax.set_ylabel("Mean metric value")
                ax.annotate(
                    row_label,
                    xy=(-0.34, 0.5),
                    xycoords="axes fraction",
                    ha="right",
                    va="center",
                )

    handles = [
        Line2D(
            [0],
            [0],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2,
            markersize=5,
            label=style["label"],
        )
        for style in metric_styles.values()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.02),
        frameon=True,
    )

    variant_label = "Masked PCA" if variant == "masked" else "Unmasked PCA"
    fig.suptitle(f"{variant_label}: OOD-averaged detection metrics", y=0.97)
    fig.tight_layout(rect=[0, 0.1, 1, 0.93])

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_base = PLOTS_DIR / f"pca_{variant}_ood_averaged_metrics"
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved {variant} PCA plots to:\n  - {png_path}\n  - {pdf_path}")


def main() -> None:
    """Generate compact PCA variant sweep plots."""

    data = load_reports(DEFAULT_JSON_PATHS)
    df = extract_rows(data)
    if df.empty:
        raise ValueError("No PCA metric rows were found.")

    for variant in ("unmasked", "masked"):
        plot_variant(df, variant)


if __name__ == "__main__":
    main()
