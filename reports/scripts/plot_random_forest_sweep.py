"""Plot Random Forest hyperparameter sweep overlaying ROC AUC and FPR@95."""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns

# Define paths relative to the project root
ROOT = Path(__file__).resolve().parents[2]
JSON_PATH = ROOT / "reports" / "outputs" / "json" / "random_forest_sweep.json"
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


def load_sweep_df(path: Path) -> pd.DataFrame:
    """Load sweep JSON and parse it into a pandas DataFrame."""
    with path.open("r") as file:
        data = json.load(file)

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


def plot_overlay(df: pd.DataFrame) -> None:
    """Generate and save the overlay plot."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True)

    # Color palette for OOD scores
    scores = [
        "max_probability_difference",
        "absolute_max_probability_difference",
        "student_teacher_kl_divergence",
        "logit_l2_distance",
        "energy_gap",
        "absolute_energy_gap",
        "student_msp",
        "student_energy",
    ]
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
    scores = [
        "max_probability_difference",
        "absolute_max_probability_difference",
        "student_teacher_kl_divergence",
        "logit_l2_distance",
        "energy_gap",
        "absolute_energy_gap",
        "student_msp",
        "student_energy",
    ]
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


def main() -> None:
    """Load data and generate overlay plots."""
    print(f"Loading data from {JSON_PATH}...")
    df = load_sweep_df(JSON_PATH)

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


if __name__ == "__main__":
    main()
