"""Plot Random Forest hyperparameter sweep overlaying ROC AUC and FPR@95."""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
JSON_DIR = ROOT / "reports" / "outputs" / "json"
JSON_PATHS = [
    JSON_DIR / "random_forest_sweep.json",
    JSON_DIR / "random_forest_sweep_extra.json",
]
PLOTS_DIR = ROOT / "reports" / "outputs" / "plots"

# Styling configuration
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
})

DATASET_LABELS = {
    "cifar10_test": "CIFAR-10",
    "cifar100_test": "CIFAR-100",
    "mnist_test": "MNIST",
    "svhn_test": "SVHN",
}

OOD_SCORE_LABELS = {
    "max_probability_difference": "Max Diff",
    "absolute_max_probability_difference": "Abs. Max Diff",
    "student_teacher_kl_divergence": "KL Div.",
    "logit_l2_distance": "Logit L2",
    "energy_gap": "Energy Gap",
    "absolute_energy_gap": "Abs. Energy Gap",
    "student_msp": "Student MSP",
    "student_energy": "Student Energy",
}

OOD_SCORE_ORDER = [
    "max_probability_difference",
    "absolute_max_probability_difference",
    "student_teacher_kl_divergence",
    "logit_l2_distance",
    "energy_gap",
    "absolute_energy_gap",
    "student_msp",
    "student_energy",
]

# Teacher MSP baselines (from metrics_baseline.tex)
TEACHER_MSP = {
    ("cifar10_test", "mnist_test"): 0.92,  # ROC AUC
    ("cifar10_test", "svhn_test"): 0.90,
    ("cifar10_test", "cifar100_test"): 0.88,
    ("cifar100_test", "mnist_test"): 0.70,
    ("cifar100_test", "svhn_test"): 0.81,
    ("cifar100_test", "cifar10_test"): 0.79,
}

TEACHER_MSP_FPR = {
    ("cifar10_test", "mnist_test"): 0.56,  # FPR@95
    ("cifar10_test", "svhn_test"): 0.60,
    ("cifar10_test", "cifar100_test"): 0.62,
    ("cifar100_test", "mnist_test"): 0.96,
    ("cifar100_test", "svhn_test"): 0.81,
    ("cifar100_test", "cifar10_test"): 0.79,
}

# Define grid structure
GRID_CONFIG = [
    # (row_idx, id_dataset, ood_dataset, col_idx)
    (0, "cifar10_test", "mnist_test", 0),
    (0, "cifar10_test", "svhn_test", 1),
    (0, "cifar10_test", "cifar100_test", 2),
    (1, "cifar100_test", "mnist_test", 0),
    (1, "cifar100_test", "svhn_test", 1),
    (1, "cifar100_test", "cifar10_test", 2),
]


def load_sweep_df(paths: list[Path]) -> pd.DataFrame:
    """Load sweep JSON files and parse merged layer-4 runs into a DataFrame."""

    data = load_sweep_data(paths)
    records = []
    runs = data.get("runs", [])
    for run in runs:
        run_name = run.get("run_name", "")
        if "random_forest_layer4" not in run_name:
            continue

        id_dataset = run.get("id_dataset")
        n_estimators = run.get("n_estimators")
        metrics = run.get("metrics", [])

        for metric in metrics:
            ood_dataset = metric.get("ood_dataset")
            ood_score = metric.get("ood_score")
            roc_auc = metric.get("roc_auc")
            fpr_at_95_tpr = metric.get("fpr_at_95_tpr")

            records.append({
                "n_estimators": n_estimators,
                "id_dataset": id_dataset,
                "ood_dataset": ood_dataset,
                "ood_score": ood_score,
                "roc_auc": roc_auc,
                "fpr_at_95_tpr": fpr_at_95_tpr,
            })

    return pd.DataFrame(records)


def load_sweep_data(paths: list[Path]) -> dict[str, object]:
    """Load and merge random forest sweep JSON data."""

    runs_by_name: dict[str, dict[str, object]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r") as file:
            data = json.load(file)

        for run in data.get("runs", []):
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
                metrics_by_key[metric_key(metric)] = metric
            merged_run["metrics"] = list(metrics_by_key.values())

    return {"runs": list(runs_by_name.values()), "version": 1}


def metric_key(metric: dict[str, object]) -> tuple[object, ...]:
    """Return the merge key for one metric record."""

    return (
        metric.get("id_dataset"),
        metric.get("ood_dataset"),
        metric.get("ood_score"),
        metric.get("method"),
        metric.get("probability_mode"),
    )


def plot_overlay(df: pd.DataFrame) -> None:
    """Generate and save the overlay plot."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True)

    # Color palette for OOD scores
    scores = OOD_SCORE_ORDER
    palette = sns.color_palette("Set2", len(scores))
    color_map = dict(zip(scores, palette))

    legend_handles = []
    legend_labels = []
    style_added = False

    for row_idx, id_dataset, ood_dataset, col_idx in GRID_CONFIG:
        ax = axes[row_idx, col_idx]

        # Filter data for this subplot
        sub_df = df[(df["id_dataset"] == id_dataset) & (df["ood_dataset"] == ood_dataset)]

        # Plot line for each OOD score method
        for score in scores:
            score_data = sub_df[sub_df["ood_score"] == score].sort_values("n_estimators")
            if not score_data.empty:
                # 1. Plot ROC AUC (solid line)
                l_auc, = ax.plot(
                    score_data["n_estimators"],
                    score_data["roc_auc"],
                    marker="o",
                    linestyle="-",
                    linewidth=2,
                    color=color_map[score],
                )
                # 2. Plot FPR@95 (dashed line)
                ax.plot(
                    score_data["n_estimators"],
                    score_data["fpr_at_95_tpr"],
                    marker="x",
                    linestyle="--",
                    linewidth=1.5,
                    color=color_map[score],
                )
                if row_idx == 0 and col_idx == 0:
                    legend_handles.append(l_auc)
                    legend_labels.append(OOD_SCORE_LABELS.get(score, score))

        # Add Teacher MSP Baselines
        t_auc = TEACHER_MSP.get((id_dataset, ood_dataset))
        t_fpr = TEACHER_MSP_FPR.get((id_dataset, ood_dataset))

        if t_auc is not None:
            ax.axhline(t_auc, color="gray", linestyle="-", linewidth=1.5)
        if t_fpr is not None:
            ax.axhline(t_fpr, color="gray", linestyle="--", linewidth=1.5)

        # Add style and teacher indicator items to the legend once
        if row_idx == 0 and col_idx == 0 and not style_added:
            style_auc = Line2D([0], [0], color="black", linestyle="-", linewidth=2)
            style_fpr = Line2D([0], [0], color="black", linestyle="--", linewidth=1.5)
            teacher_legend = Line2D([0], [0], color="gray", linestyle="-", linewidth=1.5)

            legend_handles.extend([style_auc, style_fpr, teacher_legend])
            legend_labels.extend([r"ROC AUC ($\uparrow$)", r"FPR@95 ($\downarrow$)", "Teacher MSP"])
            style_added = True

        # Subplot Title
        id_lbl = DATASET_LABELS.get(id_dataset, id_dataset)
        ood_lbl = DATASET_LABELS.get(ood_dataset, ood_dataset)
        ax.set_title(f"ID: {id_lbl} vs OOD: {ood_lbl}")
        ax.set_ylim(-0.05, 1.05)

        # Labels
        if row_idx == 1:
            ax.set_xlabel("n_estimators")
        if col_idx == 0:
            ax.set_ylabel("Metric Value")

    # Put a unified legend at the bottom
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.02),
        frameon=True,
    )

    # Adjust layout
    plt.suptitle("Random Forest (layer4) Sweep: ROC AUC vs FPR@95 Overlay", y=0.96)
    plt.tight_layout(rect=[0, 0.08, 1, 0.94])

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    png_path = PLOTS_DIR / "random_forest_overlay_sweep.png"
    pdf_path = PLOTS_DIR / "random_forest_overlay_sweep.pdf"

    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"Saved overlay plots to:\n  - {png_path}\n  - {pdf_path}")


def plot_single_metric(
    df: pd.DataFrame,
    metric_key: str,
    title: str,
    ylabel: str,
    output_stem: str,
) -> None:
    """Generate and save one Random Forest sweep plot for a single metric."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
    scores = OOD_SCORE_ORDER
    palette = sns.color_palette("Set2", len(scores))
    color_map = dict(zip(scores, palette))

    legend_handles = []
    legend_labels = []

    for row_idx, id_dataset, ood_dataset, col_idx in GRID_CONFIG:
        ax = axes[row_idx, col_idx]
        sub_df = df[(df["id_dataset"] == id_dataset) & (df["ood_dataset"] == ood_dataset)]

        for score in scores:
            score_data = sub_df[sub_df["ood_score"] == score].sort_values("n_estimators")
            if score_data.empty:
                continue
            line, = ax.plot(
                score_data["n_estimators"],
                score_data[metric_key],
                marker="o",
                linestyle="-",
                linewidth=2,
                color=color_map[score],
            )
            if row_idx == 0 and col_idx == 0:
                legend_handles.append(line)
                legend_labels.append(OOD_SCORE_LABELS.get(score, score))

        teacher_metric = TEACHER_MSP if metric_key == "roc_auc" else TEACHER_MSP_FPR
        teacher_value = teacher_metric.get((id_dataset, ood_dataset))
        if teacher_value is not None:
            ax.axhline(teacher_value, color="gray", linestyle="--", linewidth=1.5)
            if row_idx == 0 and col_idx == 0:
                legend_handles.append(
                    Line2D([0], [0], color="gray", linestyle="--", linewidth=1.5)
                )
                legend_labels.append("Teacher MSP")

        id_lbl = DATASET_LABELS.get(id_dataset, id_dataset)
        ood_lbl = DATASET_LABELS.get(ood_dataset, ood_dataset)
        ax.set_title(f"ID: {id_lbl} vs OOD: {ood_lbl}")
        ax.set_ylim(-0.05, 1.05)
        if row_idx == 1:
            ax.set_xlabel("n_estimators")
        if col_idx == 0:
            ax.set_ylabel(ylabel)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.02),
        frameon=True,
    )
    plt.suptitle(title, y=0.96)
    plt.tight_layout(rect=[0, 0.08, 1, 0.94])

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    png_path = PLOTS_DIR / f"{output_stem}.png"
    pdf_path = PLOTS_DIR / f"{output_stem}.pdf"
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"Saved {metric_key} plots to:\n  - {png_path}\n  - {pdf_path}")


def plot_ood_averaged_metrics(df: pd.DataFrame) -> None:
    """Plot ROC AUC and 1 - FPR@95 averaged across OOD datasets by OOD score."""

    scores = OOD_SCORE_ORDER
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

    grouped = (
        df.groupby(["id_dataset", "ood_score", "n_estimators"], as_index=False)
        .agg(
            roc_auc=("roc_auc", "mean"),
            fpr_at_95_tpr=("fpr_at_95_tpr", "mean"),
            ood_dataset_count=("ood_dataset", "nunique"),
        )
        .sort_values(["id_dataset", "ood_score", "n_estimators"])
    )
    grouped["fpr_at_95_tpr_complement"] = 1.0 - grouped["fpr_at_95_tpr"]

    for id_dataset in sorted(df["id_dataset"].dropna().unique()):
        id_df = grouped[grouped["id_dataset"] == id_dataset]
        if id_df.empty:
            continue

        estimator_values = sorted(id_df["n_estimators"].dropna().unique())
        fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), sharex=True, sharey=True)
        flat_axes = axes.flatten()

        for ax, score in zip(flat_axes, scores):
            score_df = id_df[id_df["ood_score"] == score].sort_values("n_estimators")
            if score_df.empty:
                ax.axis("off")
                continue

            for metric_key, style in metric_styles.items():
                ax.plot(
                    score_df["n_estimators"],
                    score_df[metric_key],
                    label=style["label"],
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=2,
                    markersize=5,
                )

            ax.set_title(OOD_SCORE_LABELS.get(score, score))
            ax.set_xscale("log")
            ax.set_xticks(estimator_values)
            ax.set_xticklabels([str(int(value)) for value in estimator_values])
            ax.set_ylim(0, 1)
            ax.grid(True, which="major", axis="both", alpha=0.35)

        for ax in flat_axes[len(scores) :]:
            ax.axis("off")

        for ax in axes[-1, :]:
            ax.set_xlabel("n_estimators")
        for ax in axes[:, 0]:
            ax.set_ylabel("Mean metric value")

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

        id_label = DATASET_LABELS.get(id_dataset, id_dataset)
        fig.suptitle(
            f"Random Forest (layer4): OOD-averaged detection metrics by n_estimators ({id_label})",
            y=0.96,
        )
        fig.tight_layout(rect=[0, 0.08, 1, 0.93])

        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        output_base = PLOTS_DIR / f"random_forest_ood_averaged_metrics_{id_dataset}"
        png_path = output_base.with_suffix(".png")
        pdf_path = output_base.with_suffix(".pdf")
        fig.savefig(png_path, dpi=300)
        fig.savefig(pdf_path)
        plt.close(fig)
        print(f"Saved OOD-averaged plots for {id_dataset} to:\n  - {png_path}\n  - {pdf_path}")


def main() -> None:
    """Load data and generate overlay plots."""
    print("Loading data from:")
    for path in JSON_PATHS:
        print(f"  - {path}")
    df = load_sweep_df(JSON_PATHS)

    print("Generating overlay plot...")
    plot_overlay(df)
    print("Generating ROC AUC plot...")
    plot_single_metric(
        df,
        "roc_auc",
        r"Random Forest (layer4) Sweep: ROC AUC ($\uparrow$)",
        "ROC AUC",
        "random_forest_roc_auc_sweep",
    )
    print("Generating FPR@95 plot...")
    plot_single_metric(
        df,
        "fpr_at_95_tpr",
        r"Random Forest (layer4) Sweep: FPR@95 ($\downarrow$)",
        "FPR@95",
        "random_forest_fpr_at_95_tpr_sweep",
    )
    print("Generating OOD-averaged metric plots...")
    plot_ood_averaged_metrics(df)


if __name__ == "__main__":
    main()
