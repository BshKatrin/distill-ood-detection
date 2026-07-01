"""Plot OOD score ROC-AUC/FPR@95 tradeoffs across a hyperparameter sweep."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from math import ceil
from pathlib import Path
import re
from typing import Any, Literal

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns


Hyperparameter = Literal["n-estimators", "pca-components", "pca-mask-probability"]

DATASET_LABELS = {
    "cifar10_test": "CIFAR-10",
    "cifar100_test": "CIFAR-100",
    "mnist_test": "MNIST",
    "svhn_test": "SVHN",
}

OOD_SCORE_LABELS = {
    "max_probability_difference": "Max probability difference",
    "absolute_max_probability_difference": "Absolute max probability difference",
    "student_teacher_kl_divergence": "Student-teacher KL divergence",
    "logit_l2_distance": "Logit L2 distance",
    "energy_gap": "Energy gap",
    "absolute_energy_gap": "Absolute energy gap",
    "student_msp": "Student MSP",
    "student_energy": "Student Energy",
}

MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<", ">", "h")

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
class RunRecord:
    """Normalized metadata for one exported report run."""

    run_name: str
    id_dataset: str
    hyperparameter_value: float
    hyperparameter_label: str


def load_reports(paths: list[Path]) -> dict[str, Any]:
    """Load and merge report JSON files by run name and metric identity."""

    runs_by_name: dict[str, dict[str, Any]] = {}
    for path in paths:
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
    """Extract a ResNet architecture token from a run name."""

    match = re.search(r"resnet\d+", run_name)
    return match.group(0) if match else None


def extract_feature_layer(run_name: str) -> str | None:
    """Extract a feature layer token from a run name."""

    match = re.search(r"layer\d+", run_name)
    return match.group(0) if match else None


def extract_pca_components(run_name: str) -> int | None:
    """Extract a PCA component count from a run name."""

    match = re.search(r"_pca(\d+)(?:_|$)", run_name)
    return int(match.group(1)) if match else None


def extract_mask_probability(run_name: str) -> float | None:
    """Extract a masked-PCA probability from a run name."""

    match = re.search(r"_mask_p(\d{3})(?:_|$)", run_name)
    return int(match.group(1)) / 100 if match else None


def extract_hyperparameter(
    run: dict[str, Any],
    hyperparameter: Hyperparameter,
) -> tuple[float, str] | None:
    """Extract the selected hyperparameter from run metadata."""

    run_name = str(run["run_name"])
    if hyperparameter == "n-estimators":
        value = run.get("n_estimators")
        if value is None:
            match = re.search(r"_n(\d+)(?:_|$)", run_name)
            value = int(match.group(1)) if match else None
        if value is None:
            return None
        numeric_value = float(value)
        return numeric_value, str(int(numeric_value))

    if hyperparameter == "pca-components":
        value = run.get("pca_components") or extract_pca_components(run_name)
        if value is None:
            return None
        numeric_value = float(value)
        return numeric_value, str(int(numeric_value))

    value = run.get("pca_mask_probability") or extract_mask_probability(run_name)
    if value is None:
        return None
    numeric_value = float(value)
    return numeric_value, f"{numeric_value:.2g}"


def run_matches_filters(run: dict[str, Any], args: argparse.Namespace) -> bool:
    """Return whether a run passes explicit run-level filters."""

    run_name = str(run.get("run_name", ""))
    if any(token not in run_name for token in args.include_run_name):
        return False
    if any(token in run_name for token in args.exclude_run_name):
        return False
    if extract_architecture(run_name) != args.architecture:
        return False
    if extract_feature_layer(run_name) != args.feature_layer:
        return False
    if args.pca_components is not None and extract_pca_components(run_name) != args.pca_components:
        return False
    if (
        args.pca_mask_probability is not None
        and extract_mask_probability(run_name) != args.pca_mask_probability
    ):
        return False
    return str(run.get("id_dataset")) in set(args.id_dataset)


def extract_rows(data: dict[str, Any], args: argparse.Namespace) -> pd.DataFrame:
    """Extract filtered OOD metric rows into a plot-ready DataFrame."""

    id_datasets = set(args.id_dataset)
    ood_datasets = set(args.ood_dataset)
    ood_scores = set(args.ood_score)

    rows: list[dict[str, Any]] = []
    skipped_missing_hyperparameter = 0
    skipped_runs_by_filter = 0

    for run in data.get("runs", []):
        if not isinstance(run, dict):
            continue
        if not run_matches_filters(run, args):
            skipped_runs_by_filter += 1
            continue

        hyperparameter = extract_hyperparameter(run, args.hyperparameter)
        if hyperparameter is None:
            skipped_missing_hyperparameter += 1
            continue
        hyperparameter_value, hyperparameter_label = hyperparameter
        record = RunRecord(
            run_name=str(run["run_name"]),
            id_dataset=str(run["id_dataset"]),
            hyperparameter_value=hyperparameter_value,
            hyperparameter_label=hyperparameter_label,
        )

        for metric in run.get("metrics", []):
            if not isinstance(metric, dict):
                continue
            if str(metric.get("probability_mode")) != args.probability_mode:
                continue
            if str(metric.get("method")) != args.method:
                continue
            if str(metric.get("id_dataset")) not in id_datasets:
                continue
            if str(metric.get("ood_dataset")) not in ood_datasets:
                continue
            if str(metric.get("ood_score")) not in ood_scores:
                continue
            if metric.get("roc_auc") is None or metric.get("fpr_at_95_tpr") is None:
                continue

            rows.append(
                {
                    "run_name": record.run_name,
                    "id_dataset": record.id_dataset,
                    "ood_dataset": str(metric["ood_dataset"]),
                    "ood_score": str(metric["ood_score"]),
                    "method": str(metric["method"]),
                    "hyperparameter_value": record.hyperparameter_value,
                    "hyperparameter_label": record.hyperparameter_label,
                    "roc_auc": float(metric["roc_auc"]),
                    "fpr_at_95_tpr": float(metric["fpr_at_95_tpr"]),
                }
            )

    df = pd.DataFrame(rows)
    print(f"Skipped runs by run-level filters: {skipped_runs_by_filter}")
    print(f"Skipped filtered runs without {args.hyperparameter}: {skipped_missing_hyperparameter}")
    print(f"Extracted metric rows: {len(df)}")
    return df


def validate_rows(df: pd.DataFrame, args: argparse.Namespace) -> None:
    """Validate that the selected rows form a useful sweep."""

    if df.empty:
        raise ValueError("No metric rows matched the explicit filters.")

    if args.hyperparameter == "pca-mask-probability" and args.pca_components is None:
        raise ValueError("--pca-components is required for pca-mask-probability sweeps.")

    values_by_id_dataset = df.groupby("id_dataset")["hyperparameter_value"].nunique()
    invalid_id_datasets = [
        id_dataset
        for id_dataset, value_count in values_by_id_dataset.items()
        if value_count < 2
    ]
    if invalid_id_datasets:
        joined = ", ".join(invalid_id_datasets)
        raise ValueError(
            "Each selected ID dataset must have at least two hyperparameter values; "
            f"invalid ID datasets: {joined}"
        )

    duplicate_columns = [
        "id_dataset",
        "ood_dataset",
        "ood_score",
        "method",
        "hyperparameter_value",
    ]
    duplicate_rows = df[df.duplicated(duplicate_columns, keep=False)]
    if not duplicate_rows.empty:
        examples = duplicate_rows[duplicate_columns + ["run_name"]].head(10)
        raise ValueError(
            "Filters selected duplicate metric rows for the same plotted point. "
            "Add a stricter run-name, method, or dataset filter.\n"
            f"{examples.to_string(index=False)}"
        )


def sorted_unique_values(df: pd.DataFrame, column: str) -> list[str]:
    """Return unique string values sorted by first appearance."""

    return list(dict.fromkeys(df[column].astype(str)))


def hyperparameter_label(hyperparameter: Hyperparameter) -> str:
    """Return a display label for a hyperparameter parser."""

    labels = {
        "n-estimators": "n_estimators",
        "pca-components": "PCA components",
        "pca-mask-probability": "PCA mask probability",
    }
    return labels[hyperparameter]


def plot_id_dataset(df: pd.DataFrame, id_dataset: str, args: argparse.Namespace) -> None:
    """Plot all selected OOD scores for one ID dataset."""

    id_df = df[df["id_dataset"] == id_dataset].copy()
    ood_scores = sorted_unique_values(id_df, "ood_score")
    ood_datasets = sorted_unique_values(id_df, "ood_dataset")
    hyperparameter_rows = (
        id_df[["hyperparameter_value", "hyperparameter_label"]]
        .drop_duplicates()
        .sort_values("hyperparameter_value")
    )
    hyperparameter_labels = hyperparameter_rows["hyperparameter_label"].tolist()

    ncols = min(3, len(ood_scores))
    nrows = ceil(len(ood_scores) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.0 * ncols, 4.2 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    flat_axes = axes.flatten()

    palette = sns.color_palette("viridis", len(hyperparameter_labels))
    color_map = dict(zip(hyperparameter_labels, palette))
    marker_map = {
        ood_dataset: MARKERS[index % len(MARKERS)]
        for index, ood_dataset in enumerate(ood_datasets)
    }

    for ax, ood_score in zip(flat_axes, ood_scores):
        score_df = id_df[id_df["ood_score"] == ood_score]
        for _, row in score_df.iterrows():
            ax.scatter(
                row["fpr_at_95_tpr"],
                row["roc_auc"],
                s=72,
                marker=marker_map[row["ood_dataset"]],
                color=color_map[row["hyperparameter_label"]],
                edgecolor="black",
                linewidth=0.45,
                alpha=0.9,
            )

        ax.set_title(OOD_SCORE_LABELS.get(ood_score, ood_score.replace("_", " ")))
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel(r"FPR@95 ($\downarrow$)")
        ax.set_ylabel(r"ROC-AUC ($\uparrow$)")

    for ax in flat_axes[len(ood_scores) :]:
        ax.axis("off")

    color_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color_map[label],
            markeredgecolor="black",
            markersize=8,
            label=label,
        )
        for label in hyperparameter_labels
    ]
    marker_handles = [
        Line2D(
            [0],
            [0],
            marker=marker_map[dataset],
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=8,
            label=DATASET_LABELS.get(dataset, dataset),
        )
        for dataset in ood_datasets
    ]

    fig.legend(
        handles=color_handles,
        title=hyperparameter_label(args.hyperparameter),
        loc="lower center",
        bbox_to_anchor=(0.34, 0.02),
        ncol=max(1, min(5, len(color_handles))),
        frameon=True,
    )
    fig.legend(
        handles=marker_handles,
        title="OOD dataset",
        loc="lower center",
        bbox_to_anchor=(0.78, 0.02),
        ncol=max(1, min(4, len(marker_handles))),
        frameon=True,
    )

    id_label = DATASET_LABELS.get(id_dataset, id_dataset)
    fig.suptitle(
        f"{id_label}: OOD score tradeoff by {hyperparameter_label(args.hyperparameter)} "
        f"({args.probability_mode}, {args.method})",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0.14, 1, 0.94])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_base = args.output_dir / f"{args.output_stem}_{id_dataset}"
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved {id_dataset} plots to:\n  - {png_path}\n  - {pdf_path}")


def summarize_selection(df: pd.DataFrame) -> None:
    """Print a compact summary of the selected plotting data."""

    value_summary = (
        df[["id_dataset", "hyperparameter_value", "hyperparameter_label"]]
        .drop_duplicates()
        .sort_values(["id_dataset", "hyperparameter_value"])
        .groupby("id_dataset")["hyperparameter_label"]
        .apply(lambda values: ", ".join(values.astype(str)))
        .to_dict()
    )
    score_counts = defaultdict(int)
    for score in df["ood_score"].astype(str):
        score_counts[score] += 1

    print("Selected hyperparameter values by ID dataset:")
    for id_dataset, values in value_summary.items():
        print(f"  - {id_dataset}: {values}")
    print("Selected OOD scores:")
    for score in sorted(score_counts):
        print(f"  - {score}: {score_counts[score]} rows")

    hyperparameter_counts = df.groupby("id_dataset")["hyperparameter_value"].nunique()
    ood_counts = df.groupby(["id_dataset", "ood_score"])["ood_dataset"].nunique()
    expected_counts = {
        key: int(hyperparameter_counts[key[0]]) * int(ood_count)
        for key, ood_count in ood_counts.items()
    }
    actual_counts = df.groupby(["id_dataset", "ood_score"]).size()
    missing_groups = []
    for key, expected_count in expected_counts.items():
        actual_count = int(actual_counts.get(key, 0))
        if actual_count < expected_count:
            missing_groups.append((key[0], key[1], actual_count, expected_count))

    if missing_groups:
        print("Incomplete score coverage:")
        for id_dataset, ood_score, actual_count, expected_count in missing_groups:
            print(f"  - {id_dataset} / {ood_score}: {actual_count}/{expected_count} points")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Plot FPR@95 versus ROC-AUC for selected OOD score metrics across "
            "an explicitly filtered hyperparameter sweep."
        )
    )
    parser.add_argument(
        "--json",
        action="append",
        type=Path,
        required=True,
        help="Input JSON report path. Pass once per file to merge multiple reports.",
    )
    parser.add_argument(
        "--hyperparameter",
        choices=["n-estimators", "pca-components", "pca-mask-probability"],
        required=True,
        help="Hyperparameter parser to apply to selected run names or metadata.",
    )
    parser.add_argument(
        "--include-run-name",
        action="append",
        required=True,
        help="Required substring in run_name. Pass multiple times to require all substrings.",
    )
    parser.add_argument(
        "--exclude-run-name",
        action="append",
        default=[],
        help="Forbidden substring in run_name. Pass multiple times to exclude any match.",
    )
    parser.add_argument("--architecture", required=True, help="Required architecture token, e.g. resnet18.")
    parser.add_argument("--feature-layer", required=True, help="Required feature layer token, e.g. layer4.")
    parser.add_argument(
        "--probability-mode",
        required=True,
        help="Metric probability mode to plot, e.g. unperturbed or perturbed.",
    )
    parser.add_argument(
        "--method",
        required=True,
        help="Metric method/objective to include.",
    )
    parser.add_argument(
        "--id-dataset",
        action="append",
        required=True,
        help="ID dataset key to include. Pass multiple times for multiple output figures.",
    )
    parser.add_argument(
        "--ood-dataset",
        action="append",
        required=True,
        help="OOD dataset key to include. Pass multiple times for multiple markers.",
    )
    parser.add_argument(
        "--ood-score",
        action="append",
        required=True,
        help="OOD score key to include. Pass multiple times for multiple subplots.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        help="Optional run-level filter for a fixed PCA component count.",
    )
    parser.add_argument(
        "--pca-mask-probability",
        type=float,
        help="Optional run-level filter for a fixed masked-PCA probability.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for generated PNG and PDF plots.",
    )
    parser.add_argument(
        "--output-stem",
        required=True,
        help="Output filename stem. The ID dataset key is appended before the suffix.",
    )
    return parser.parse_args()


def main() -> None:
    """Load metrics and write one tradeoff figure per selected ID dataset."""

    args = parse_args()
    data = load_reports(args.json)
    df = extract_rows(data, args)
    validate_rows(df, args)
    summarize_selection(df)

    for id_dataset in args.id_dataset:
        plot_id_dataset(df, id_dataset, args)


if __name__ == "__main__":
    main()
