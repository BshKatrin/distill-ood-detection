"""Build the complete FAISS k-NN OOD Score experiment report."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from distill_ood_detection.evaluation.ood_metrics import ood_detection_metrics


ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = ROOT / "runs" / "embedding_distances"
REPORT_PATH = (
    ROOT
    / "experiments"
    / "near_ood_detection"
    / "knn_faiss_ood_scores.md"
)
SUMMARY_PATH = (
    ROOT / "reports" / "outputs" / "json" / "knn_faiss_ood_scores.json"
)

SCORES = {
    "student_teacher_kl_divergence": "KL(teacher || k-NN)",
    "absolute_max_probability_difference": "Absolute max-probability gap",
    "absolute_energy_gap": "Absolute energy gap",
    "ensemble_predictive_entropy": "Predictive entropy",
    "ensemble_bald": "BALD",
    "student_msp": "k-NN MSP",
    "student_energy": "k-NN energy",
}


@dataclass(frozen=True)
class RunSpec:
    """One completed k-NN experiment setting."""

    key: str
    title: str
    short_title: str
    id_dataset: str
    id_label: str
    near_dataset: str
    near_label: str
    architecture: str
    variant: str
    run_path: str
    config_path: str
    job_id: int
    elapsed: str
    query_draws: int = 1
    baseline_key: str | None = None

    @property
    def distances_dir(self) -> Path:
        """Return the local distance artifact directory."""

        return RUNS_ROOT / self.run_path / "distances"

    @property
    def ood_datasets(self) -> tuple[tuple[str, str], ...]:
        """Return Near-OOD followed by the two Far-OOD datasets."""

        return (
            (self.near_dataset, self.near_label),
            ("mnist_test", "MNIST"),
            ("svhn_test", "SVHN"),
        )


RUNS = (
    RunSpec(
        "raw_c10_r18", "Raw layer4, ResNet-18", "Raw R18", "cifar10_test",
        "CIFAR-10", "cifar100_test", "CIFAR-100", "ResNet-18", "raw",
        "cifar_10/resnet18", "configs/embedding_distances/cifar_10/resnet18.yaml",
        408658, "00:02:22",
    ),
    RunSpec(
        "raw_c10_r50", "Raw layer4, ResNet-50", "Raw R50", "cifar10_test",
        "CIFAR-10", "cifar100_test", "CIFAR-100", "ResNet-50", "raw",
        "cifar_10/resnet50", "configs/embedding_distances/cifar_10/resnet50.yaml",
        408659, "00:02:22",
    ),
    RunSpec(
        "affine_c10_r50", "Affine reference corruption, ResNet-50", "Affine R50",
        "cifar10_test", "CIFAR-10", "cifar100_test", "CIFAR-100", "ResNet-50",
        "affine_clean_inference", "perturbation/pixel/augmentation/cifar_10/resnet50",
        "configs/embedding_distances/perturbation/pixel/augmentation/cifar_10/resnet50.yaml",
        408662, "00:03:30", baseline_key="raw_c10_r50",
    ),
    RunSpec(
        "channel_c10_r18", "Layers3-4 channel clipping, ResNet-18", "Channel clip R18",
        "cifar10_test", "CIFAR-10", "cifar100_test", "CIFAR-100", "ResNet-18",
        "layer34_channel_clean_inference",
        "perturbation/embedding/clipping/cifar_10/resnet18_channel_flatten",
        "configs/embedding_distances/perturbation/embedding/clipping/cifar_10/resnet18_channel_flatten.yaml",
        408664, "00:00:31", baseline_key="raw_c10_r18",
    ),
    RunSpec(
        "spatial_c10_r50", "Layers3-4 spatial clipping, ResNet-50", "Spatial clip R50",
        "cifar10_test", "CIFAR-10", "cifar100_test", "CIFAR-100", "ResNet-50",
        "layer34_spatial_50_draw_inference",
        "perturbation/embedding/clipping/cifar_10/resnet50_spatial_flatten",
        "configs/embedding_distances/perturbation/embedding/clipping/cifar_10/resnet50_spatial_flatten.yaml",
        408666, "00:10:41", query_draws=50, baseline_key="raw_c10_r50",
    ),
    RunSpec(
        "raw_c100_r18", "Raw layer4, ResNet-18", "Raw R18", "cifar100_test",
        "CIFAR-100", "cifar10_test", "CIFAR-10", "ResNet-18", "raw",
        "cifar_100/resnet18", "configs/embedding_distances/cifar_100/resnet18.yaml",
        408660, "00:00:32",
    ),
    RunSpec(
        "raw_c100_r50", "Raw layer4, ResNet-50", "Raw R50", "cifar100_test",
        "CIFAR-100", "cifar10_test", "CIFAR-10", "ResNet-50", "raw",
        "cifar_100/resnet50", "configs/embedding_distances/cifar_100/resnet50.yaml",
        408661, "00:01:00",
    ),
    RunSpec(
        "affine_c100_r50", "Affine reference corruption, ResNet-50", "Affine R50",
        "cifar100_test", "CIFAR-100", "cifar10_test", "CIFAR-10", "ResNet-50",
        "affine_clean_inference", "perturbation/pixel/augmentation/cifar_100/resnet50",
        "configs/embedding_distances/perturbation/pixel/augmentation/cifar_100/resnet50.yaml",
        408663, "00:03:29", baseline_key="raw_c100_r50",
    ),
    RunSpec(
        "channel_c100_r18", "Layers3-4 channel clipping, ResNet-18", "Channel clip R18",
        "cifar100_test", "CIFAR-100", "cifar10_test", "CIFAR-10", "ResNet-18",
        "layer34_channel_clean_inference",
        "perturbation/embedding/clipping/cifar_100/resnet18_channel_flatten",
        "configs/embedding_distances/perturbation/embedding/clipping/cifar_100/resnet18_channel_flatten.yaml",
        408665, "00:00:34", baseline_key="raw_c100_r18",
    ),
    RunSpec(
        "spatial_c100_r50", "Layers3-4 spatial clipping, ResNet-50", "Spatial clip R50",
        "cifar100_test", "CIFAR-100", "cifar10_test", "CIFAR-10", "ResNet-50",
        "layer34_spatial_50_draw_inference",
        "perturbation/embedding/clipping/cifar_100/resnet50_spatial_flatten",
        "configs/embedding_distances/perturbation/embedding/clipping/cifar_100/resnet50_spatial_flatten.yaml",
        408667, "00:10:26", query_draws=50, baseline_key="raw_c100_r50",
    ),
)


def artifact_path(spec: RunSpec, dataset: str) -> Path:
    """Return one local layer4 distance artifact path."""

    return spec.distances_dir / dataset / "test" / "layer4.pt"


def score_values(spec: RunSpec, dataset: str, score: str) -> np.ndarray:
    """Load one sign-adjusted OOD Score vector."""

    artifact = torch.load(
        artifact_path(spec, dataset), map_location="cpu", weights_only=False
    )
    return artifact["distances"]["raw"]["ood_scores"][score].numpy()


def metric_pair(id_scores: np.ndarray, ood_scores: np.ndarray) -> tuple[float, float]:
    """Return ROC-AUC and FPR@95 for ID-oriented scores."""

    labels = np.concatenate((np.ones(id_scores.size), np.zeros(ood_scores.size)))
    scores = np.concatenate((id_scores, ood_scores))
    metrics = ood_detection_metrics(labels, scores)
    return metrics["roc_auc"], metrics["fpr_at_95_tpr"]


def compromise(pair: tuple[float, float]) -> float:
    """Return the recap report's balanced AUROC/FPR@95 objective."""

    return 0.5 * (pair[0] + 1.0 - pair[1])


def evaluate(spec: RunSpec, score: str) -> dict[str, Any]:
    """Evaluate one score against every configured OOD dataset."""

    id_scores = score_values(spec, spec.id_dataset, score)
    pairs = {
        dataset: metric_pair(id_scores, score_values(spec, dataset, score))
        for dataset, _ in spec.ood_datasets
    }
    macro = (
        float(np.mean([pair[0] for pair in pairs.values()])),
        float(np.mean([pair[1] for pair in pairs.values()])),
    )
    return {"pairs": pairs, "macro": macro, "compromise": compromise(macro)}


def format_pair(pair: tuple[float, float]) -> str:
    """Format an AUROC/FPR@95 pair."""

    return f"{pair[0]:.3f} / {pair[1]:.3f}"


def markdown_table(
    headers: tuple[str, ...], rows: list[tuple[str, ...]]
) -> list[str]:
    """Return one Markdown table as lines."""

    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    lines.append("")
    return lines


def best_result(spec: RunSpec) -> tuple[str, dict[str, Any]]:
    """Return the compromise-selected score for one setting."""

    results = {score: evaluate(spec, score) for score in SCORES}
    return max(results.items(), key=lambda item: item[1]["compromise"])


def descriptive_record(
    spec: RunSpec, dataset: str, score: str, values: np.ndarray
) -> dict[str, Any]:
    """Return one complete numeric-summary record."""

    quantiles = np.percentile(values, (5, 25, 50, 75, 95))
    record: dict[str, Any] = {
        "setting": spec.key,
        "variant": spec.variant,
        "architecture": spec.architecture,
        "id_dataset": spec.id_label,
        "dataset": dataset,
        "score": score,
        "count": int(values.size),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std()),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
    }
    if dataset != spec.id_dataset:
        pair = metric_pair(score_values(spec, spec.id_dataset, score), values)
        record.update({"roc_auc": pair[0], "fpr_at_95_tpr": pair[1]})
    return record


def export_summary() -> None:
    """Write exhaustive descriptive and OOD metric records as JSON."""

    records = []
    for spec in RUNS:
        datasets = (spec.id_dataset, *(dataset for dataset, _ in spec.ood_datasets))
        for score in SCORES:
            for dataset in datasets:
                records.append(
                    descriptive_record(
                        spec, dataset, score, score_values(spec, dataset, score)
                    )
                )
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        json.dumps({"records": records}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_report() -> str:
    """Return the complete Markdown report."""

    by_key = {spec.key: spec for spec in RUNS}
    selected = {spec.key: best_result(spec) for spec in RUNS}
    overall = max(
        (
            (spec, score, result)
            for spec in RUNS
            for score, result in ((name, evaluate(spec, name)) for name in SCORES)
        ),
        key=lambda item: item[2]["compromise"],
    )
    score_robustness = []
    for score in SCORES:
        results = [evaluate(spec, score) for spec in RUNS]
        score_robustness.append(
            (
                score,
                float(np.mean([result["compromise"] for result in results])),
                min(result["compromise"] for result in results),
                float(
                    np.mean(
                        [
                            compromise(result["pairs"][spec.near_dataset])
                            for spec, result in zip(RUNS, results, strict=True)
                        ]
                    )
                ),
            )
        )
    score_robustness.sort(key=lambda item: item[1], reverse=True)

    perturbation_deltas = []
    for spec in RUNS:
        if spec.baseline_key is None:
            continue
        baseline = by_key[spec.baseline_key]
        perturb_score, perturb_result = selected[spec.key]
        baseline_score, baseline_result = selected[baseline.key]
        perturb_near = compromise(perturb_result["pairs"][spec.near_dataset])
        baseline_near = compromise(baseline_result["pairs"][baseline.near_dataset])
        perturbation_deltas.append(
            (
                spec,
                baseline,
                perturb_score,
                perturb_result,
                baseline_score,
                baseline_result,
                perturb_result["compromise"] - baseline_result["compromise"],
                perturb_near - baseline_near,
            )
        )

    improved = sum(delta[6] > 0.0 for delta in perturbation_deltas)
    best_score_name, best_score_mean, _, _ = score_robustness[0]
    lines = [
        "# FAISS k-NN Neighbor-Output OOD Scores",
        "",
        "Status: completed on 2026-08-05.",
        "",
        "## Executive summary",
        "",
        f"The best overall setting is **{overall[0].short_title} with "
        f"{SCORES[overall[1]]} on {overall[0].id_label} ID**, reaching a macro "
        f"{format_pair(overall[2]['macro'])} and compromise "
        f"{overall[2]['compromise']:.3f}.",
        "",
        f"Across all ten settings, **{SCORES[best_score_name]}** is the most "
        f"robust score by mean compromise ({best_score_mean:.3f}). Of the six "
        f"perturbation-conditioned settings, {improved} improve on their matched "
        "raw-architecture baseline after selecting each setting's best score.",
        "",
        "The exhaustive tables below report every requested OOD Score. Macro values "
        "average CIFAR near-OOD, MNIST, and SVHN independently for ROC-AUC and "
        "FPR@95.",
        "",
        "## Method",
        "",
        "Each run builds an exact FAISS `IndexFlatL2` index from raw post-GAP "
        "`layer4` ID training representations. For each query, the selected "
        "`k = 10` neighbors act as a non-parametric student:",
        "",
        "- mean neighbor probabilities are used for KL, maximum-probability gap, "
        "and k-NN MSP;",
        "- mean neighbor logits are used for the absolute energy gap and k-NN energy;",
        "- individual neighbor probabilities define predictive entropy and BALD.",
        "",
        "All artifact scores are sign-adjusted so higher means more ID-like. Table "
        "cells are `ROC-AUC / FPR@95`; higher ROC-AUC and lower FPR@95 are better. "
        "The compromise objective is `0.5 * (macro AUROC + 1 - macro FPR@95)`.",
        "",
        "For affine and channel-clipping runs, one corrupted representation per ID "
        "training image is indexed and queries remain clean. Spatial-clipping "
        "ResNet-50 runs search 50 independently clipped representations per query; "
        "all `50 * k` neighbor outputs are aggregated.",
        "",
        "The `flatten` and training-objective labels belong to the replaced parametric "
        "students. The non-parametric replacement follows the requested post-GAP "
        "geometry, and Logit-MSE/KL training objectives have no operational role "
        "because no student is optimized.",
        "",
        "## Completed runs",
        "",
    ]
    run_rows = []
    for spec in RUNS:
        config_link = f"[config](../../{spec.config_path})"
        run_rows.append(
            (
                spec.id_label,
                spec.short_title,
                str(spec.query_draws),
                config_link,
                f"`{spec.job_id}`",
                spec.elapsed,
            )
        )
    lines.extend(
        markdown_table(
            ("ID", "Setting", "Query draws", "Config", "Job", "Elapsed"),
            run_rows,
        )
    )

    lines.extend(("## Compromise-selected result per setting", ""))
    selected_rows = []
    for spec in RUNS:
        score, result = selected[spec.key]
        selected_rows.append(
            (
                spec.id_label,
                spec.short_title,
                SCORES[score],
                format_pair(result["pairs"][spec.near_dataset]),
                format_pair(result["pairs"]["mnist_test"]),
                format_pair(result["pairs"]["svhn_test"]),
                format_pair(result["macro"]),
                f"{result['compromise']:.3f}",
            )
        )
    lines.extend(
        markdown_table(
            ("ID", "Setting", "Selected score", "Near", "MNIST", "SVHN", "Macro", "Compromise"),
            selected_rows,
        )
    )

    lines.extend(("## Perturbation comparison with matched raw k-NN", ""))
    lines.append(
        "Each side selects its own best score by macro compromise. Positive deltas favor the perturbation."
    )
    lines.append("")
    comparison_rows = []
    for delta in perturbation_deltas:
        spec, baseline = delta[0], delta[1]
        comparison_rows.append(
            (
                spec.id_label,
                spec.short_title,
                SCORES[delta[2]],
                format_pair(delta[3]["macro"]),
                SCORES[delta[4]],
                format_pair(delta[5]["macro"]),
                f"{delta[6]:+.3f}",
                f"{delta[7]:+.3f}",
            )
        )
    lines.extend(
        markdown_table(
            (
                "ID", "Perturbation", "Its best score", "Perturbation macro",
                "Raw best score", "Raw macro", "Δ macro compromise", "Δ near compromise",
            ),
            comparison_rows,
        )
    )

    lines.extend(("## Score robustness across all settings", ""))
    robustness_rows = [
        (SCORES[score], f"{mean:.3f}", f"{worst:.3f}", f"{near:.3f}")
        for score, mean, worst, near in score_robustness
    ]
    lines.extend(
        markdown_table(
            ("OOD Score", "Mean macro compromise", "Worst macro compromise", "Mean near-OOD compromise"),
            robustness_rows,
        )
    )

    lines.extend(
        (
            "## Interpretation",
            "",
            "### Raw layer4 k-NN is the strongest choice",
            "",
            "No corruption-conditioned run improves macro compromise over its "
            "matched raw architecture. The strongest result is raw ResNet-18 on "
            "CIFAR-10 with k-NN energy (`0.905 / 0.440`). Raw ResNet-50 is also "
            "better than both of its CIFAR-10 perturbation variants.",
            "",
            "### CIFAR-100 remains the difficult regime",
            "",
            "Every CIFAR-100 setting has high FPR@95. The best is raw ResNet-50 "
            "with the absolute energy gap (`0.794 / 0.714`, compromise `0.540`). "
            "Affine corruption raises macro AUROC to `0.813` with KL, but worsens "
            "macro FPR@95 to `0.772`, so its compromise is lower (`0.520`). This "
            "confirms that AUROC-only selection would overstate the affine gain.",
            "",
            "### Perturbations offer only isolated near-OOD gains",
            "",
            "Affine ResNet-50 improves CIFAR-100 near-OOD compromise by `+0.026`, "
            "and channel clipping improves it by `+0.002`; neither improvement "
            "survives the macro comparison. On CIFAR-10, 50-draw spatial clipping "
            "with BALD essentially ties raw ResNet-50 on near-OOD (`+0.001`) but "
            "loses `0.039` macro compromise because far-OOD FPR@95 is worse.",
            "",
            "### Output scale makes the absolute energy gap fragile",
            "",
            "The absolute energy gap is the least robust score overall. Under "
            "CIFAR-100 spatial clipping it collapses to macro `0.258 / 0.998`, "
            "indicating that averaging logits across many clipped neighborhoods "
            "introduces a scale shift that is almost maximally harmful at the "
            "95% TPR operating point. KL and uncertainty-based scores tolerate "
            "the corruption substantially better.",
            "",
            "### Recommended use",
            "",
            "Use raw post-GAP layer4 k-NN as the default. Prefer k-NN energy for "
            "CIFAR-10. For CIFAR-100, choose scores per architecture: KL for "
            "ResNet-18 and absolute energy gap for raw ResNet-50. The perturbation "
            "variants are useful as negative or robustness ablations, not as "
            "replacements for the raw index.",
            "",
        )
    )

    lines.extend(("## Exhaustive OOD metrics", ""))
    for id_label in ("CIFAR-10", "CIFAR-100"):
        lines.extend((f"### {id_label} as ID", ""))
        rows = []
        for spec in (candidate for candidate in RUNS if candidate.id_label == id_label):
            for score in SCORES:
                result = evaluate(spec, score)
                rows.append(
                    (
                        spec.short_title,
                        SCORES[score],
                        format_pair(result["pairs"][spec.near_dataset]),
                        format_pair(result["pairs"]["mnist_test"]),
                        format_pair(result["pairs"]["svhn_test"]),
                        format_pair(result["macro"]),
                        f"{result['compromise']:.3f}",
                    )
                )
        lines.extend(
            markdown_table(
                ("Setting", "OOD Score", f"Near: {'CIFAR-100' if id_label == 'CIFAR-10' else 'CIFAR-10'}", "Far: MNIST", "Far: SVHN", "Macro", "Compromise"),
                rows,
            )
        )

    lines.extend(
        (
            "## Reproducibility",
            "",
            "The report is generated by "
            "[`reports/scripts/build_knn_ood_score_report.py`](../../reports/scripts/build_knn_ood_score_report.py). "
            "Complete per-dataset descriptive statistics and OOD metrics are stored in "
            "[`reports/outputs/json/knn_faiss_ood_scores.json`](../../reports/outputs/json/knn_faiss_ood_scores.json).",
            "",
            "The raw artifacts remain under the corresponding `runs/embedding_distances/` "
            "directories and contain neighbor indices, distances, mean probabilities, "
            "mean logits, raw metrics, and sign-adjusted OOD Scores.",
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    """Generate the numeric summary and Markdown report."""

    export_summary()
    REPORT_PATH.write_text(build_report(), encoding="utf-8")


if __name__ == "__main__":
    main()
