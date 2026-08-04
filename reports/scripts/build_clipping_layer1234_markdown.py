"""Build the sequential layer1-layer4 clipping experiment report."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STUDENT_METRICS_PATH = (
    ROOT / "reports" / "outputs" / "json" / "clipping_layer1234_metrics.json"
)
LAYER34_METRICS_PATH = (
    ROOT / "reports" / "outputs" / "json" / "clipping_layer34_metrics.json"
)
TEACHER_METRICS_PATH = (
    ROOT / "reports" / "outputs" / "json" / "clipping_layer34_teacher_metrics.json"
)
OUTPUT_PATH = (
    ROOT
    / "experiments"
    / "embedding_clipping"
    / "layer1_layer2_layer3_layer4_clipping.md"
)

DATASET_LABELS = {
    "cifar10_test": "CIFAR-10",
    "cifar100_test": "CIFAR-100",
    "mnist_test": "MNIST",
    "svhn_test": "SVHN",
}
OOD_DATASETS = {
    "cifar10_test": ("mnist_test", "svhn_test", "cifar100_test"),
    "cifar100_test": ("mnist_test", "svhn_test", "cifar10_test"),
}
MODEL_LABELS = {"resnet18": "ResNet-18", "resnet50": "ResNet-50"}
CLIPPING_LABELS = {
    "constant": "Constant",
    "spatial": "Spatial",
    "channel": "Channel",
}
POOL_LABELS = {"avg": "GAP", "flatten": "Flatten"}
METHOD_LABELS = {
    "mse_logits": "Centered-logit MSE",
    "kl_divergence": "KL divergence",
}
SCORE_LABELS = {
    "max_probability_difference": "Max probability difference",
    "absolute_max_probability_difference": "Absolute max probability difference",
    "student_teacher_kl_divergence": "KL (teacher / student)",
    "logit_l2_distance": "Centered-logit L2 distance",
    "energy_gap": "Energy gap",
    "absolute_energy_gap": "Absolute energy gap",
    "student_msp": "Student MSP",
    "student_energy": "Student energy",
}
SCORE_ORDER = tuple(SCORE_LABELS)
TEACHER_SCORE_LABELS = {"msp": "Teacher MSP", "energy": "Teacher energy"}

Variant = tuple[str, str, str, str]
MetricKey = tuple[str, str, str, str, str, str]


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON artifact."""

    with path.open() as file:
        return json.load(file)


def parse_variant(
    run_name: str,
    *,
    run_stem: str,
    experiment_stem: str,
) -> Variant:
    """Parse dataset, teacher, clipping mode, and pooling from a run name."""

    parts = Path(run_name).parts
    if "clipping" in parts:
        clipping_index = parts.index("clipping")
        dataset = parts[clipping_index + 1]
        model = parts[clipping_index + 2]
        clipping_mode, pool = parts[-1].removeprefix(run_stem).rsplit("_", 1)
        return dataset, model, clipping_mode, pool

    suffix = run_name.removeprefix(experiment_stem)
    model, dataset, marker, clipping_mode, pool = suffix.split("_", 4)
    if marker != "clip":
        raise ValueError(f"Unexpected clipping run name: {run_name}")
    dataset = {"cifar10": "cifar_10", "cifar100": "cifar_100"}[dataset]
    return dataset, model, clipping_mode, pool


def id_dataset_key(dataset: str) -> str:
    """Return the probability-artifact dataset key."""

    return f"{dataset.replace('_', '')}_test"


def metric_pair(records: list[dict[str, Any]]) -> tuple[float, float]:
    """Return macro ROC-AUC and FPR@95."""

    return (
        mean(float(record["roc_auc"]) for record in records),
        mean(float(record["fpr_at_95_tpr"]) for record in records),
    )


def format_pair(pair: tuple[float, float]) -> str:
    """Format one ROC-AUC/FPR@95 pair."""

    return f"{pair[0]:.3f} / {pair[1]:.3f}"


def add_table(
    lines: list[str],
    headers: tuple[str, ...],
    rows: list[tuple[str, ...]],
) -> None:
    """Append a Markdown table."""

    lines.extend(
        [
            "| " + " | ".join(headers) + " |",
            "|"
            + "|".join(
                "---" if index == 0 else "---:" for index in range(len(headers))
            )
            + "|",
        ]
    )
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    lines.append("")


def metric_index(
    payload: dict[str, Any],
    *,
    run_stem: str,
    experiment_stem: str,
) -> tuple[set[Variant], dict[MetricKey, dict[str, dict[str, Any]]]]:
    """Index clean-inference OOD metrics."""

    variants: set[Variant] = set()
    records: dict[MetricKey, dict[str, dict[str, Any]]] = {}
    for run in payload["runs"]:
        variant = parse_variant(
            run["run_name"],
            run_stem=run_stem,
            experiment_stem=experiment_stem,
        )
        variants.add(variant)
        for record in run["metrics"]:
            if record["probability_mode"] != "unperturbed":
                continue
            key = (*variant, record["method"], record["ood_score"])
            records.setdefault(key, {})[record["ood_dataset"]] = record
    return variants, records


def macro_pair(
    index: dict[MetricKey, dict[str, dict[str, Any]]],
    key: MetricKey,
) -> tuple[float, float]:
    """Return a clean-inference macro metric pair."""

    dataset_key = id_dataset_key(key[0])
    records = index[key]
    return metric_pair([records[ood] for ood in OOD_DATASETS[dataset_key]])


def run_dir(variant: Variant) -> Path:
    """Return one all-layer run directory."""

    dataset, model, clipping_mode, pool = variant
    return (
        ROOT
        / "runs"
        / "students"
        / "perturbation"
        / "embedding"
        / "clipping"
        / dataset
        / model
        / f"linear_layer1_layer2_layer3_layer4_clip_{clipping_mode}_{pool}"
    )


def training_metrics(variant: Variant) -> dict[str, dict[str, Any]]:
    """Load method-indexed training metrics."""

    payload = load_json(run_dir(variant) / "summary.json")
    return {entry["method"]: entry for entry in payload["methods"]}


def input_dimension(variant: Variant) -> int:
    """Load the student input dimension."""

    payload = load_json(run_dir(variant) / "resolved_config.json")
    return int(payload["student"]["input_shape"][0])


def teacher_metric_index(
    payload: dict[str, Any],
) -> dict[tuple[str, str, str], dict[str, dict[str, Any]]]:
    """Index teacher baseline metrics."""

    result: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for run in payload["runs"]:
        model = next(
            part for part in run["run_name"].split("_") if part in MODEL_LABELS
        )
        dataset_key = run["id_dataset"]
        for score in TEACHER_SCORE_LABELS:
            result[(dataset_key, model, score)] = {
                record["ood_dataset"]: record
                for record in run["metrics"]
                if record["ood_score"] == score
            }
    return result


def describe_key(key: MetricKey) -> str:
    """Describe one metric configuration."""

    return (
        f"{SCORE_LABELS[key[5]]}; {CLIPPING_LABELS[key[2]]}, "
        f"{POOL_LABELS[key[3]]}, {METHOD_LABELS[key[4]]}"
    )


def paired_layer_comparison(
    current: dict[MetricKey, dict[str, dict[str, Any]]],
    previous: dict[MetricKey, dict[str, dict[str, Any]]],
    dataset: str,
    model: str,
) -> tuple[int, int, int, float, float]:
    """Compare all-layer and layer3-layer4 macro metrics."""

    keys = [
        key
        for key in current
        if key in previous and key[0] == dataset and key[1] == model
    ]
    current_pairs = [macro_pair(current, key) for key in keys]
    previous_pairs = [macro_pair(previous, key) for key in keys]
    return (
        sum(left[0] > right[0] for left, right in zip(current_pairs, previous_pairs)),
        sum(left[1] < right[1] for left, right in zip(current_pairs, previous_pairs)),
        len(keys),
        mean(left[0] - right[0] for left, right in zip(current_pairs, previous_pairs)),
        mean(left[1] - right[1] for left, right in zip(current_pairs, previous_pairs)),
    )


def build_report() -> str:
    """Build the complete all-layer clipping report."""

    current_payload = load_json(STUDENT_METRICS_PATH)
    previous_payload = load_json(LAYER34_METRICS_PATH)
    teacher_payload = load_json(TEACHER_METRICS_PATH)
    variants, current = metric_index(
        current_payload,
        run_stem="linear_layer1_layer2_layer3_layer4_clip_",
        experiment_stem="perturbation_linear_layer1_layer2_layer3_layer4_student_",
    )
    _, previous = metric_index(
        previous_payload,
        run_stem="linear_layer3_layer4_clip_",
        experiment_stem="perturbation_linear_layer3_layer4_student_",
    )
    teachers = teacher_metric_index(teacher_payload)

    if len(variants) != 24:
        raise ValueError(f"Expected 24 all-layer runs, found {len(variants)}")
    expected_records = 24 * 2 * 3 * len(SCORE_ORDER)
    if len(current) * 3 != expected_records:
        raise ValueError(
            f"Expected {expected_records} clean metric records, "
            f"found {len(current) * 3}"
        )

    lines = [
        "# Sequential Layer1-Layer4 Clipping Students",
        "",
        "Status: completed for CIFAR-10 and CIFAR-100 ID with ResNet-18 and "
        "ResNet-50 teachers.",
        "",
        "## Setup",
        "",
        "- Clipping sequence: clip `layer1`, propagate, clip `layer2`, propagate, "
        "clip `layer3`, propagate, then clip `layer4`.",
        "- Every layer independently samples from the same percentile range "
        "`[0.5, 1.0]`.",
        "- Clipping modes: constant, spatial-dependent, and channel-dependent; "
        "one mode is shared by all four layers.",
        "- Student input: GAP or flattened clipped `layer4`, concatenated with "
        "`u_layer1`, `u_layer2`, `u_layer3`, and `u_layer4`.",
        "- Student: linear classifier trained against clean teacher logits.",
        "- Objectives: centered-logit MSE and KL divergence.",
        "- Training: 30 epochs, seed 42, batch size 256; best validation-loss "
        "checkpoint.",
        "- OOD datasets: MNIST, SVHN, and the opposite CIFAR test set.",
        "- Inference scope: clean/unperturbed only. The completed jobs did not "
        "export 50-draw clipped inference artifacts.",
        "",
        "Student input dimensions:",
        "",
    ]

    dimension_rows = []
    for model in MODEL_LABELS:
        for clipping_mode in CLIPPING_LABELS:
            for pool in POOL_LABELS:
                variant = ("cifar_10", model, clipping_mode, pool)
                dimension_rows.append(
                    (
                        MODEL_LABELS[model],
                        CLIPPING_LABELS[clipping_mode],
                        POOL_LABELS[pool],
                        str(input_dimension(variant)),
                    )
                )
    add_table(
        lines,
        ("Teacher", "Clipping", "Pooling", "Student input"),
        dimension_rows,
    )

    lines.extend(
        [
            "Run-directory pattern:",
            "",
            "```text",
            "runs/students/perturbation/embedding/clipping/"
            "<cifar_10|cifar_100>/<resnet18|resnet50>/"
            "linear_layer1_layer2_layer3_layer4_clip_"
            "<constant|spatial|channel>_<avg|flatten>",
            "```",
            "",
            "## Training Results",
            "",
            "Losses are objective-specific and should not be compared across "
            "centered-logit MSE and KL divergence.",
            "",
        ]
    )

    for dataset in ("cifar_10", "cifar_100"):
        for model in MODEL_LABELS:
            lines.extend(
                [
                    f"### {DATASET_LABELS[id_dataset_key(dataset)]} ID, "
                    f"{MODEL_LABELS[model]}",
                    "",
                ]
            )
            rows = []
            for clipping_mode in CLIPPING_LABELS:
                for pool in POOL_LABELS:
                    variant = (dataset, model, clipping_mode, pool)
                    metrics = training_metrics(variant)
                    for method in METHOD_LABELS:
                        entry = metrics[method]
                        rows.append(
                            (
                                CLIPPING_LABELS[clipping_mode],
                                POOL_LABELS[pool],
                                METHOD_LABELS[method],
                                f"{entry['test_accuracy']:.4f}",
                                f"{entry['test_distillation_loss']:.6g}",
                            )
                        )
            add_table(
                lines,
                (
                    "Clipping",
                    "Pooling",
                    "Objective",
                    "Test accuracy",
                    "Test distillation loss",
                ),
                rows,
            )

    lines.extend(
        [
            "## OOD Metrics",
            "",
            "Every cell is `ROC-AUC / FPR@95`; higher ROC-AUC and lower FPR@95 "
            "are better. All OOD Scores follow the project convention that higher "
            "values are more ID-like.",
            "",
            "## Teacher Baselines",
            "",
        ]
    )
    for dataset in ("cifar_10", "cifar_100"):
        dataset_key = id_dataset_key(dataset)
        for model in MODEL_LABELS:
            lines.extend(
                [
                    f"### {DATASET_LABELS[dataset_key]} ID, {MODEL_LABELS[model]}",
                    "",
                ]
            )
            rows = []
            for score, label in TEACHER_SCORE_LABELS.items():
                records = teachers[(dataset_key, model, score)]
                ordered = [records[ood] for ood in OOD_DATASETS[dataset_key]]
                rows.append(
                    (
                        label,
                        *(
                            format_pair(
                                (record["roc_auc"], record["fpr_at_95_tpr"])
                            )
                            for record in ordered
                        ),
                        format_pair(metric_pair(ordered)),
                    )
                )
            add_table(
                lines,
                (
                    "Score",
                    *(DATASET_LABELS[ood] for ood in OOD_DATASETS[dataset_key]),
                    "Macro",
                ),
                rows,
            )

    lines.extend(["## Main Findings", ""])
    best_rows = []
    comparison_rows = []
    for dataset in ("cifar_10", "cifar_100"):
        for model in MODEL_LABELS:
            candidates = [
                (key, macro_pair(current, key))
                for key in current
                if key[0] == dataset and key[1] == model
            ]
            best_roc_key, best_roc = max(candidates, key=lambda item: item[1][0])
            best_fpr_key, best_fpr = min(candidates, key=lambda item: item[1][1])
            best_rows.append(
                (
                    DATASET_LABELS[id_dataset_key(dataset)],
                    MODEL_LABELS[model],
                    describe_key(best_roc_key),
                    format_pair(best_roc),
                    describe_key(best_fpr_key),
                    format_pair(best_fpr),
                )
            )
            roc_wins, fpr_wins, total, roc_delta, fpr_delta = (
                paired_layer_comparison(current, previous, dataset, model)
            )
            comparison_rows.append(
                (
                    DATASET_LABELS[id_dataset_key(dataset)],
                    MODEL_LABELS[model],
                    f"{roc_wins}/{total}",
                    f"{roc_delta:+.3f}",
                    f"{fpr_wins}/{total}",
                    f"{fpr_delta:+.3f}",
                )
            )
    add_table(
        lines,
        (
            "ID dataset",
            "Teacher",
            "Best macro ROC-AUC configuration",
            "Macro",
            "Best macro FPR@95 configuration",
            "Macro",
        ),
        best_rows,
    )

    lines.extend(
        [
            "Paired comparison against clean layer3-layer4 inference. A positive "
            "ROC-AUC delta and negative FPR@95 delta favor all-layer clipping.",
            "The full reference results are in the "
            "[layer3-layer4 report](layer3_layer4_clipping.md).",
            "",
        ]
    )
    add_table(
        lines,
        (
            "ID dataset",
            "Teacher",
            "ROC-AUC wins",
            "Mean ROC-AUC delta",
            "FPR@95 wins",
            "Mean FPR@95 delta",
        ),
        comparison_rows,
    )
    lines.extend(
        [
            "Across the paired macro comparisons, extending clipping to layer1 and "
            "layer2 is not a consistent improvement over layer3-layer4 clipping. "
            "The strongest all-layer case is CIFAR-100/ResNet-18 for ROC-AUC, while "
            "the other settings lose most matched comparisons.",
            "",
            "Classification quality also deteriorates sharply for many spatial and "
            "channel variants. In particular, the CIFAR-10/ResNet-50 spatial-GAP KL "
            "student reaches the best macro OOD pair in that group but only 0.5223 "
            "test accuracy, versus 0.9238 for its layer3-layer4 counterpart. The "
            "CIFAR-100/ResNet-50 spatial-flatten KL student similarly reaches the "
            "best macro ROC-AUC with only 0.1142 test accuracy, versus 0.7884 for "
            "layer3-layer4 clipping. These OOD gains therefore reflect degraded "
            "students rather than a better accuracy-preserving detector.",
            "",
            "Constant clipping with flattened layer4 features is the most stable "
            "all-layer family, but its best test accuracies still trail the matching "
            "layer3-layer4 runs. Overall, the clean-inference results do not support "
            "extending sequential clipping into layer1 and layer2.",
            "",
            "## Macro OOD Comparison",
            "",
        ]
    )

    for dataset in ("cifar_10", "cifar_100"):
        for model in MODEL_LABELS:
            lines.extend(
                [
                    f"### {DATASET_LABELS[id_dataset_key(dataset)]} ID, "
                    f"{MODEL_LABELS[model]}",
                    "",
                ]
            )
            for method in METHOD_LABELS:
                lines.extend([f"#### {METHOD_LABELS[method]}", ""])
                rows = []
                for clipping_mode in CLIPPING_LABELS:
                    for pool in POOL_LABELS:
                        prefix = (dataset, model, clipping_mode, pool, method)
                        rows.append(
                            (
                                CLIPPING_LABELS[clipping_mode],
                                POOL_LABELS[pool],
                                *(
                                    format_pair(macro_pair(current, (*prefix, score)))
                                    for score in SCORE_ORDER
                                ),
                            )
                        )
                add_table(
                    lines,
                    (
                        "Clipping",
                        "Pooling",
                        *(SCORE_LABELS[score] for score in SCORE_ORDER),
                    ),
                    rows,
                )

    lines.extend(
        [
            "## Detailed OOD Results",
            "",
            "Each configuration contains both training-objective tables.",
            "",
        ]
    )
    for dataset in ("cifar_10", "cifar_100"):
        dataset_key = id_dataset_key(dataset)
        for model in MODEL_LABELS:
            lines.extend(
                [
                    f"### {DATASET_LABELS[dataset_key]} ID, {MODEL_LABELS[model]}",
                    "",
                ]
            )
            for clipping_mode in CLIPPING_LABELS:
                for pool in POOL_LABELS:
                    lines.extend(
                        [
                            "<details>",
                            f"<summary>{CLIPPING_LABELS[clipping_mode]}, "
                            f"{POOL_LABELS[pool]}</summary>",
                            "",
                        ]
                    )
                    for method in METHOD_LABELS:
                        lines.extend([f"#### {METHOD_LABELS[method]}", ""])
                        rows = []
                        for score in SCORE_ORDER:
                            key = (
                                dataset,
                                model,
                                clipping_mode,
                                pool,
                                method,
                                score,
                            )
                            records = current[key]
                            ordered = [
                                records[ood] for ood in OOD_DATASETS[dataset_key]
                            ]
                            rows.append(
                                (
                                    SCORE_LABELS[score],
                                    *(
                                        format_pair(
                                            (
                                                record["roc_auc"],
                                                record["fpr_at_95_tpr"],
                                            )
                                        )
                                        for record in ordered
                                    ),
                                    format_pair(metric_pair(ordered)),
                                )
                            )
                        add_table(
                            lines,
                            (
                                "Score",
                                *(
                                    DATASET_LABELS[ood]
                                    for ood in OOD_DATASETS[dataset_key]
                                ),
                                "Macro",
                            ),
                            rows,
                        )
                    lines.extend(["</details>", ""])

    lines.extend(
        [
            "## Artifacts",
            "",
            "```text",
            "reports/outputs/json/clipping_layer1234_metrics.json",
            "reports/outputs/json/clipping_layer34_metrics.json",
            "reports/outputs/json/clipping_layer34_teacher_metrics.json",
            "```",
            "",
            "SLURM train-and-clean-inference jobs: `407342`, `407343`, `407344`, "
            "and `407345`.",
            "",
            "A direct stochastic-inference comparison requires a separate "
            "`APPLY_PERTURBATION=1` export for these 24 runs.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Write the all-layer clipping Markdown report."""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(build_report())
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
