"""Export Random Forest (layer4) hyperparameter sweep table to LaTeX."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Define paths relative to the project root
ROOT = Path(__file__).resolve().parents[2]
JSON_PATH = ROOT / "reports" / "outputs" / "json" / "random_forest_sweep.json"
OUTPUT_PATH = ROOT / "reports" / "outputs" / "latex" / "metrics_random_forest_sweep.tex"

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
}

ID_DATASET_ORDER = ["cifar10_test", "cifar100_test"]

OOD_DATASET_ORDER = {
    "cifar10_test": ["mnist_test", "svhn_test", "cifar100_test"],
    "cifar100_test": ["mnist_test", "svhn_test", "cifar10_test"],
}

OOD_SCORE_ORDER = [
    "max_probability_difference",
    "absolute_max_probability_difference",
    "student_teacher_kl_divergence",
    "logit_l2_distance",
    "energy_gap",
    "absolute_energy_gap",
]

ESTIMATORS_ORDER = [20, 50, 100, 200, 300]

TEACHER_MSP = {
    ("cifar10_test", "mnist_test"): "0.92/0.56",
    ("cifar10_test", "svhn_test"): "0.90/0.60",
    ("cifar10_test", "cifar100_test"): "0.88/0.62",
    ("cifar100_test", "mnist_test"): "0.70/0.96",
    ("cifar100_test", "svhn_test"): "0.81/0.81",
    ("cifar100_test", "cifar10_test"): "0.79/0.79",
}


def load_sweep_data(path: Path) -> dict[str, Any]:
    """Load the random forest sweep JSON data.

    Args:
        path: Path to the JSON sweep file.

    Returns:
        The parsed JSON dictionary.
    """
    with path.open("r") as file:
        return json.load(file)


def extract_layer4_metrics(data: dict[str, Any]) -> dict[tuple[int, str, str, str], dict[str, float]]:
    """Extract metrics for Random Forest (layer4) runs.

    Args:
        data: The parsed JSON sweep dictionary.

    Returns:
        A dictionary mapping (n_estimators, id_dataset, ood_dataset, ood_score)
        to the metric values dictionary containing 'roc_auc' and 'fpr_at_95_tpr'.
    """
    extracted = {}
    runs = data.get("runs", [])
    for run in runs:
        run_name = run.get("run_name", "")
        # Only process Random Forest (layer4) runs
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

            if (
                id_dataset is not None
                and ood_dataset is not None
                and ood_score is not None
                and n_estimators is not None
                and roc_auc is not None
                and fpr_at_95_tpr is not None
            ):
                key = (n_estimators, id_dataset, ood_dataset, ood_score)
                extracted[key] = {
                    "roc_auc": roc_auc,
                    "fpr_at_95_tpr": fpr_at_95_tpr,
                }
    return extracted


def build_latex_table(
    metrics_map: dict[tuple[int, str, str, str], dict[str, float]]
) -> str:
    """Build the LaTeX table content in the style of metrics_perturbation.tex.

    Args:
        metrics_map: The extracted metrics dictionary.

    Returns:
        The complete LaTeX document content.
    """
    column_spec = "cccccccc"
    total_columns = 8

    # Define columns to fetch data for
    data_columns = [
        ("cifar10_test", "mnist_test"),
        ("cifar10_test", "svhn_test"),
        ("cifar10_test", "cifar100_test"),
        ("cifar100_test", "mnist_test"),
        ("cifar100_test", "svhn_test"),
        ("cifar100_test", "cifar10_test"),
    ]

    # 1. Teacher MSP Row
    teacher_msp_cells = []
    for id_dataset, ood_dataset in data_columns:
        val = TEACHER_MSP.get((id_dataset, ood_dataset), "-")
        teacher_msp_cells.append(val)
    teacher_msp_row = "- & Teacher MSP & " + " & ".join(teacher_msp_cells) + r" \\"

    # 2. Teacher Energy Row (empty cells for data since we don't have teacher logits)
    teacher_energy_row = r"- & Teacher Energy & & & & & & \\"

    rows = [
        teacher_msp_row,
        teacher_energy_row,
        r"\midrule",
    ]

    first_student = True
    for n_estimators in ESTIMATORS_ORDER:
        if not first_student:
            rows.append(r"\addlinespace")
        first_student = False

        first_row_for_est = True

        for ood_score in OOD_SCORE_ORDER:
            score_label = OOD_SCORE_LABELS.get(ood_score, ood_score)

            cells = []
            for id_dataset, ood_dataset in data_columns:
                key = (n_estimators, id_dataset, ood_dataset, ood_score)
                metric = metrics_map.get(key)
                if metric is not None:
                    roc_auc = metric["roc_auc"]
                    fpr = metric["fpr_at_95_tpr"]
                    cells.append(f"{roc_auc:.2f}/{fpr:.2f}")
                else:
                    cells.append("-")

            # Column grouping labels
            est_col = str(n_estimators) if first_row_for_est else ""

            row_str = f"{est_col} & {score_label} & " + " & ".join(cells) + r" \\"
            rows.append(row_str)
            first_row_for_est = False

    body = "\n".join(rows)

    # Compile the final document
    latex = rf"""\documentclass[border=2pt]{{standalone}}
\usepackage{{booktabs}}

\begin{{document}}

\setlength{{\tabcolsep}}{{5pt}}
\begin{{tabular}}{{{column_spec}}}
\multicolumn{{{total_columns}}}{{c}}{{\textbf{{Random forest (layer4) ROCAUC $\uparrow$ / FPR@95 $\downarrow$}}}}\\[4pt]
\toprule
 &  & \multicolumn{{3}}{{c}}{{CIFAR-10 (ID)}} & \multicolumn{{3}}{{c}}{{CIFAR-100 (ID)}} \\
\cmidrule(lr){{3-5}} \cmidrule(lr){{6-8}}
n\_estimators & OOD Score & MNIST & SVHN & CIFAR-100 & MNIST & SVHN & CIFAR-10 \\
\midrule
{body}
\bottomrule
\end{{tabular}}

\end{{document}}
"""
    return latex


def main() -> None:
    """Load sweep data, extract layer4 metrics, format to LaTeX, and save."""
    print(f"Loading data from {JSON_PATH}...")
    data = load_sweep_data(JSON_PATH)

    print("Extracting layer4 metrics...")
    metrics_map = extract_layer4_metrics(data)

    print("Building LaTeX table...")
    latex = build_latex_table(metrics_map)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(latex)
    print(f"Wrote LaTeX table to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
