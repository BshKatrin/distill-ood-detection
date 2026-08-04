"""Build the sequential layer3/layer4 clipping experiment report."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STUDENT_METRICS_PATH = (
    ROOT / "reports" / "outputs" / "json" / "clipping_layer34_metrics.json"
)
TEACHER_METRICS_PATH = (
    ROOT / "reports" / "outputs" / "json" / "clipping_layer34_teacher_metrics.json"
)
OUTPUT_PATH = (
    ROOT / "experiments" / "embedding_clipping" / "layer3_layer4_clipping.md"
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
MODEL_LABELS = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
}
CLIPPING_LABELS = {
    "constant": "Constant",
    "spatial": "Spatial",
    "channel": "Channel",
}
POOL_LABELS = {
    "avg": "GAP",
    "flatten": "Flatten",
}
METHOD_LABELS = {
    "mse_logits": "Centered-logit MSE",
    "kl_divergence": "KL divergence",
}
PROBABILITY_MODE_LABELS = {
    "unperturbed": "Clean",
    "perturbed": "Clipped, 50-draw mean",
}
SCORE_LABELS = {
    "max_probability_difference": "Max probability difference",
    "absolute_max_probability_difference": "Absolute max probability difference",
    "student_teacher_kl_divergence": "KL (teacher || student)",
    "logit_l2_distance": "Centered-logit L2 distance",
    "energy_gap": "Energy gap",
    "absolute_energy_gap": "Absolute energy gap",
    "student_msp": "Student MSP",
    "student_energy": "Student energy",
}
SCORE_ORDER = tuple(SCORE_LABELS)
TEACHER_SCORE_LABELS = {
    "msp": "Teacher MSP",
    "energy": "Teacher energy",
}


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON report artifact."""

    with path.open() as file:
        return json.load(file)


def parse_variant(run_name: str) -> tuple[str, str, str, str]:
    """Parse dataset, teacher, clipping mode, and pooling from a run path."""

    parts = Path(run_name).parts
    if "clipping" in parts:
        clipping_index = parts.index("clipping")
        dataset = parts[clipping_index + 1]
        model = parts[clipping_index + 2]
        suffix = parts[-1].removeprefix("linear_layer3_layer4_clip_")
        clipping_mode, pool = suffix.rsplit("_", maxsplit=1)
        return dataset, model, clipping_mode, pool

    prefix = "perturbation_linear_layer3_layer4_student_"
    suffix = run_name.removeprefix(prefix)
    model, dataset, clipping_marker, clipping_mode, pool = suffix.split("_", maxsplit=4)
    if clipping_marker != "clip":
        raise ValueError(f"Unexpected sequential clipping run name: {run_name}")
    dataset = {"cifar10": "cifar_10", "cifar100": "cifar_100"}[dataset]
    return dataset, model, clipping_mode, pool


def id_dataset_key(dataset: str) -> str:
    """Return the probability-artifact key for an ID dataset directory name."""

    return f"{dataset.replace('_', '')}_test"


def metric_pair(records: list[dict[str, Any]]) -> tuple[float, float]:
    """Return macro ROC-AUC and FPR@95 for a set of OOD metric records."""

    return (
        mean(float(record["roc_auc"]) for record in records),
        mean(float(record["fpr_at_95_tpr"]) for record in records),
    )


def format_pair(pair: tuple[float, float]) -> str:
    """Format one ROC-AUC/FPR@95 pair."""

    return f"{pair[0]:.3f} / {pair[1]:.3f}"


def student_metric_index(
    payload: dict[str, Any],
) -> tuple[
    dict[tuple[str, str, str, str], dict[str, Any]],
    dict[tuple[str, str, str, str, str, str, str], dict[str, Any]],
]:
    """Index student runs and their metric records."""

    runs: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    records: dict[
        tuple[str, str, str, str, str, str, str],
        dict[str, Any],
    ] = {}
    for run in payload["runs"]:
        dataset, model, clipping_mode, pool = parse_variant(run["run_name"])
        variant = (dataset, model, clipping_mode, pool)
        runs[variant] = run
        for record in run["metrics"]:
            key = (
                dataset,
                model,
                clipping_mode,
                pool,
                record["method"],
                record["probability_mode"],
                record["ood_score"],
            )
            records.setdefault(key, {})[record["ood_dataset"]] = record
    return runs, records


def teacher_metric_index(
    payload: dict[str, Any],
) -> dict[tuple[str, str, str], dict[str, dict[str, Any]]]:
    """Index raw-image teacher metric records."""

    result: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for run in payload["runs"]:
        name_parts = run["run_name"].split("_")
        model = next(part for part in name_parts if part in MODEL_LABELS)
        dataset_key = run["id_dataset"]
        for score in TEACHER_SCORE_LABELS:
            result[(dataset_key, model, score)] = {
                record["ood_dataset"]: record
                for record in run["metrics"]
                if record["ood_score"] == score
            }
    return result


def macro_student_pair(
    index: dict[tuple[str, str, str, str, str, str, str], dict[str, Any]],
    key: tuple[str, str, str, str, str, str, str],
) -> tuple[float, float]:
    """Return one student score's macro pair."""

    dataset = key[0]
    records = index[key]
    return metric_pair([records[ood] for ood in OOD_DATASETS[id_dataset_key(dataset)]])


def run_dir(variant: tuple[str, str, str, str]) -> Path:
    """Return the local run directory for a clipping variant."""

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
        / f"linear_layer3_layer4_clip_{clipping_mode}_{pool}"
    )


def training_metrics(
    variant: tuple[str, str, str, str],
) -> dict[str, dict[str, Any]]:
    """Load the method-indexed training summary for one variant."""

    payload = load_json(run_dir(variant) / "summary.json")
    return {entry["method"]: entry for entry in payload["methods"]}


def input_dimension(variant: tuple[str, str, str, str]) -> int:
    """Read the configured student input dimension for one variant."""

    config = load_json(run_dir(variant) / "resolved_config.json")
    return int(config["student"]["input_shape"][0])


def add_table(
    lines: list[str],
    headers: tuple[str, ...],
    rows: list[tuple[str, ...]],
) -> None:
    """Append a Markdown table."""

    lines.extend(
        [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join("---" if index == 0 else "---:" for index in range(len(headers))) + "|",
        ]
    )
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    lines.append("")


def comparison_counts(
    index: dict[tuple[str, str, str, str, str, str, str], dict[str, Any]],
    fixed_dataset: str,
    fixed_model: str,
    dimension: int,
    left_value: str,
    right_value: str,
) -> tuple[int, int, int]:
    """Count matched macro comparisons won by the left side for both metrics."""

    grouped: dict[tuple[str, ...], dict[str, tuple[float, float]]] = defaultdict(dict)
    for key in index:
        dataset, model, clipping_mode, pool, method, probability_mode, score = key
        if dataset != fixed_dataset or model != fixed_model:
            continue
        values = (clipping_mode, pool, method, probability_mode)
        side = values[dimension]
        if side not in {left_value, right_value}:
            continue
        match = (*values[:dimension], *values[dimension + 1 :], score)
        grouped[match][side] = macro_student_pair(index, key)
    complete = [group for group in grouped.values() if len(group) == 2]
    roc_wins = sum(group[left_value][0] > group[right_value][0] for group in complete)
    fpr_wins = sum(group[left_value][1] < group[right_value][1] for group in complete)
    return roc_wins, fpr_wins, len(complete)


def build_report() -> str:
    """Build and return the complete Markdown experiment report."""

    student_payload = load_json(STUDENT_METRICS_PATH)
    teacher_payload = load_json(TEACHER_METRICS_PATH)
    runs, student_index = student_metric_index(student_payload)
    teacher_index = teacher_metric_index(teacher_payload)

    if len(runs) != 24:
        raise ValueError(f"Expected 24 student runs, found {len(runs)}")
    expected_records = 24 * 2 * 2 * 3 * len(SCORE_ORDER)
    actual_records = sum(len(run["metrics"]) for run in runs.values())
    if actual_records != expected_records:
        raise ValueError(
            f"Expected {expected_records} student metric records, found {actual_records}"
        )

    lines = [
        "# Sequential Layer3/Layer4 Clipping Students",
        "",
        "Status: completed for CIFAR-10 and CIFAR-100 ID with ResNet-18 and "
        "ResNet-50 teachers.",
        "",
        "## Setup",
        "",
        "- Teacher: ResNet-18 or ResNet-50.",
        "- Clipping sequence: clip `layer3`, continue the teacher forward pass, "
        "then clip `layer4`.",
        "- Clipping percentile range at both layers: `[0.5, 1.0]`.",
        "- Layer3 and layer4 percentiles are sampled independently from the same range.",
        "- Clipping modes: constant, spatial-dependent, and channel-dependent; "
        "the same mode is used at both layers.",
        "- Student input: pooled clipped `layer4` plus `u_layer3` and `u_layer4`.",
        "- Embedding pooling: global average pooling (GAP) or flattening.",
        "- Student: linear classifier.",
        "- Teacher target: clean logits.",
        "- Objectives: centered-logit MSE and KL divergence.",
        "- Training: 30 epochs, seed 42, batch size 256.",
        "- Checkpoint selection: best validation-loss checkpoint.",
        "- Stochastic evaluation: mean over 50 independent clipping draws.",
        "- OOD datasets: MNIST, SVHN, and the opposite CIFAR test set.",
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
            "linear_layer3_layer4_clip_<constant|spatial|channel>_<avg|flatten>",
            "```",
            "",
            "## Training Results",
            "",
            "Loss values are objective-dependent and should not be compared directly "
            "between centered-logit MSE and KL divergence.",
            "",
        ]
    )

    for dataset in ("cifar_10", "cifar_100"):
        dataset_label = DATASET_LABELS[id_dataset_key(dataset)]
        for model in MODEL_LABELS:
            lines.extend([f"### {dataset_label} ID, {MODEL_LABELS[model]}", ""])
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
            "## OOD Scores",
            "",
            "All scores follow the project convention that higher values are more "
            "ID-like. Every cell reports `ROC-AUC / FPR@95`; higher ROC-AUC and "
            "lower FPR@95 are better.",
            "",
            "Teacher MSP and energy use separate standard raw-image teacher inference. "
            "The student-related scores are:",
            "",
            "- negative KL divergence from teacher to student;",
            "- teacher minus student maximum probability;",
            "- negative absolute maximum-probability difference;",
            "- negative centered-logit L2 distance;",
            "- teacher minus student energy;",
            "- negative absolute energy gap;",
            "- student MSP;",
            "- student energy.",
            "",
            "Student scores are reported for clean inference and for the mean over 50 "
            "independently clipped inference draws.",
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
            for score, score_label in TEACHER_SCORE_LABELS.items():
                records = teacher_index[(dataset_key, model, score)]
                ordered = [records[ood] for ood in OOD_DATASETS[dataset_key]]
                macro = metric_pair(ordered)
                rows.append(
                    (
                        score_label,
                        *(format_pair((record["roc_auc"], record["fpr_at_95_tpr"])) for record in ordered),
                        format_pair(macro),
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

    lines.extend(
        [
            "## Macro Comparison",
            "",
            "These tables average each score across the three OOD datasets. Detailed "
            "per-dataset results follow in the next section.",
            "",
        ]
    )

    for dataset in ("cifar_10", "cifar_100"):
        dataset_label = DATASET_LABELS[id_dataset_key(dataset)]
        for model in MODEL_LABELS:
            lines.extend([f"### {dataset_label} ID, {MODEL_LABELS[model]}", ""])
            for method in METHOD_LABELS:
                for probability_mode in PROBABILITY_MODE_LABELS:
                    lines.extend(
                        [
                            f"#### {METHOD_LABELS[method]}, "
                            f"{PROBABILITY_MODE_LABELS[probability_mode]}",
                            "",
                        ]
                    )
                    rows = []
                    for clipping_mode in CLIPPING_LABELS:
                        for pool in POOL_LABELS:
                            prefix = (
                                dataset,
                                model,
                                clipping_mode,
                                pool,
                                method,
                                probability_mode,
                            )
                            rows.append(
                                (
                                    CLIPPING_LABELS[clipping_mode],
                                    POOL_LABELS[pool],
                                    *(
                                        format_pair(
                                            macro_student_pair(
                                                student_index,
                                                (*prefix, score),
                                            )
                                        )
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

    lines.extend(["## Aggregate Observations", ""])
    best_rows = []
    for dataset in ("cifar_10", "cifar_100"):
        for model in MODEL_LABELS:
            candidates = []
            for key in student_index:
                if key[0] == dataset and key[1] == model:
                    candidates.append((key, macro_student_pair(student_index, key)))
            best_roc_key, best_roc = max(candidates, key=lambda item: item[1][0])
            best_fpr_key, best_fpr = min(candidates, key=lambda item: item[1][1])

            def describe(key: tuple[str, ...]) -> str:
                return (
                    f"{SCORE_LABELS[key[6]]}; {CLIPPING_LABELS[key[2]]}, "
                    f"{POOL_LABELS[key[3]]}, {METHOD_LABELS[key[4]]}, "
                    f"{PROBABILITY_MODE_LABELS[key[5]]}"
                )

            best_rows.append(
                (
                    DATASET_LABELS[id_dataset_key(dataset)],
                    MODEL_LABELS[model],
                    describe(best_roc_key),
                    format_pair(best_roc),
                    describe(best_fpr_key),
                    format_pair(best_fpr),
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

    for dataset in ("cifar_10", "cifar_100"):
        for model in MODEL_LABELS:
            pool_roc, pool_fpr, pool_total = comparison_counts(
                student_index,
                dataset,
                model,
                1,
                "avg",
                "flatten",
            )
            draw_roc, draw_fpr, draw_total = comparison_counts(
                student_index,
                dataset,
                model,
                3,
                "perturbed",
                "unperturbed",
            )
            method_roc, method_fpr, method_total = comparison_counts(
                student_index,
                dataset,
                model,
                2,
                "mse_logits",
                "kl_divergence",
            )
            lines.extend(
                [
                    f"- {DATASET_LABELS[id_dataset_key(dataset)]}, "
                    f"{MODEL_LABELS[model]}: GAP beats flattening in "
                    f"{pool_roc}/{pool_total} matched macro ROC-AUC comparisons and "
                    f"{pool_fpr}/{pool_total} macro FPR@95 comparisons.",
                    f"- {DATASET_LABELS[id_dataset_key(dataset)]}, "
                    f"{MODEL_LABELS[model]}: 50-draw clipped inference beats clean "
                    f"inference in {draw_roc}/{draw_total} matched macro ROC-AUC "
                    f"comparisons and {draw_fpr}/{draw_total} macro FPR@95 comparisons.",
                    f"- {DATASET_LABELS[id_dataset_key(dataset)]}, "
                    f"{MODEL_LABELS[model]}: centered-logit MSE beats KL training in "
                    f"{method_roc}/{method_total} matched macro ROC-AUC comparisons and "
                    f"{method_fpr}/{method_total} macro FPR@95 comparisons.",
                ]
            )
    lines.append("")

    lines.extend(
        [
            "## Detailed OOD Results",
            "",
            "Each configuration below contains all objective and inference-mode tables.",
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
                        for probability_mode in PROBABILITY_MODE_LABELS:
                            lines.extend(
                                [
                                    f"#### {METHOD_LABELS[method]}, "
                                    f"{PROBABILITY_MODE_LABELS[probability_mode]}",
                                    "",
                                ]
                            )
                            rows = []
                            for score in SCORE_ORDER:
                                key = (
                                    dataset,
                                    model,
                                    clipping_mode,
                                    pool,
                                    method,
                                    probability_mode,
                                    score,
                                )
                                records = student_index[key]
                                ordered = [records[ood] for ood in OOD_DATASETS[dataset_key]]
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
            "Numeric metric exports:",
            "",
            "```text",
            "reports/outputs/json/clipping_layer34_metrics.json",
            "reports/outputs/json/clipping_layer34_teacher_metrics.json",
            "```",
            "",
            "Probability artifacts remain in the corresponding cluster run directories:",
            "",
            "```text",
            "<run_dir>/probabilities/unperturbed/",
            "<run_dir>/probabilities/perturbed/",
            "```",
            "",
            "SLURM jobs:",
            "",
            "- training and clean inference: `406170`, `406172`, `406174`, `406324`;",
            "- 50-draw clipped inference: `406171`, `406173`, `406175`, `406325`;",
            "- student OOD metric export: `406373`;",
            "- teacher baseline metric export: `406372`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Write the sequential clipping Markdown report."""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(build_report())
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
