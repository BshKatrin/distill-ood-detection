"""Evaluate existing CIFAR students on fixed OpenOOD v1.5 manifests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from distill_ood_detection.config import DistillationMethod, ExperimentConfig
from distill_ood_detection.datasets.openood_cifar import (
    OPENOOD_CIFAR_PROTOCOL,
    build_openood_cifar_dataset_config,
    expected_openood_cifar_sizes,
    validate_openood_cifar_manifests,
)
from distill_ood_detection.evaluation.ood_scores import (
    ABSOLUTE_ENERGY_GAP,
    ABSOLUTE_MAX_PROBABILITY_DIFFERENCE,
    ENERGY_GAP,
    LOGIT_L2_DISTANCE,
    MAX_PROBABILITY_DIFFERENCE,
    STUDENT_ENERGY,
    STUDENT_MSP,
    STUDENT_TEACHER_KL_DIVERGENCE,
    absolute_energy_gap,
    absolute_max_probability_difference,
    energy_gap,
    logit_l2_distance,
    max_probability_difference,
    student_energy,
    student_msp,
    student_teacher_kl_divergence_from_logits,
)
from distill_ood_detection.evaluation.openood import evaluate_openood_scores
from distill_ood_detection.experiments.export_feature_denoising_scores import (
    run_feature_denoising_score_export,
)
from distill_ood_detection.experiments.infer_probabilities import (
    run_probability_inference,
)
from distill_ood_detection.utils import write_json

PROBABILITY_SCORE_NAMES = (
    MAX_PROBABILITY_DIFFERENCE,
    ABSOLUTE_MAX_PROBABILITY_DIFFERENCE,
    STUDENT_TEACHER_KL_DIVERGENCE,
    LOGIT_L2_DISTANCE,
    ENERGY_GAP,
    ABSOLUTE_ENERGY_GAP,
    STUDENT_MSP,
    STUDENT_ENERGY,
    "teacher_msp",
    "teacher_energy",
)


def run_openood_cifar_evaluation(
    config: ExperimentConfig,
    source_config_path: Path,
    openood_root: Path,
    checkpoint: str = "best",
    method: DistillationMethod | None = None,
    apply_perturbation: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Run inference and OOD-positive OpenOOD metrics for one existing student."""

    dataset = build_openood_cifar_dataset_config(config.dataset, openood_root)
    manifest_sizes = validate_openood_cifar_manifests(dataset)
    evaluation_config = replace(config, dataset=dataset)
    evaluation_root = (
        Path(config.run_dir) / "evaluations" / OPENOOD_CIFAR_PROTOCOL
    )

    if config.strategy.name == "feature_denoising":
        if method is not None:
            raise ValueError("Feature Denoising OpenOOD evaluation does not use --method")
        if apply_perturbation:
            raise ValueError(
                "Feature Denoising OpenOOD evaluation does not use --apply-perturbation"
            )
        metrics_path = evaluation_root / "metrics" / f"feature_denoising_{checkpoint}.json"
    elif config.strategy.name == "perturbation":
        if method is None:
            raise ValueError("Perturbation OpenOOD evaluation requires --method")
        mode = "perturbed" if apply_perturbation else "unperturbed"
        metrics_path = evaluation_root / "metrics" / f"{method}_{mode}_{checkpoint}.json"
    else:
        raise ValueError(
            "OpenOOD CIFAR variant evaluation supports perturbation and "
            "Feature Denoising strategies"
        )

    if metrics_path.is_file() and not force:
        with metrics_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    if config.strategy.name == "feature_denoising":
        artifact_dir = evaluation_root / "feature_denoising_scores"
        manifest = run_feature_denoising_score_export(
            evaluation_config,
            checkpoint=checkpoint,  # type: ignore[arg-type]
            output_dir=artifact_dir,
        )
        score_vectors = _load_feature_denoising_scores(manifest)
        score_results = {
            name: _evaluate_scores(evaluation_config, dataset_scores)
            for name, dataset_scores in score_vectors.items()
        }
        inference_mode = "10_draw_reconstruction"
    else:
        if method is None:
            raise RuntimeError("Perturbation method was not validated")
        mode = "perturbed" if apply_perturbation else "unperturbed"
        artifact_dir = evaluation_root / "probabilities" / mode
        manifest = run_probability_inference(
            evaluation_config,
            checkpoint=checkpoint,  # type: ignore[arg-type]
            method=method,
            apply_perturbation=apply_perturbation,
            output_dir=artifact_dir,
        )
        score_vectors = _load_probability_scores(manifest, method, checkpoint)
        score_results = {
            name: _evaluate_scores(evaluation_config, dataset_scores)
            for name, dataset_scores in score_vectors.items()
        }
        inference_mode = mode

    expected_sizes = expected_openood_cifar_sizes(dataset.name)
    _validate_result_counts(score_results, expected_sizes)
    payload = {
        "version": 1,
        "protocol": OPENOOD_CIFAR_PROTOCOL,
        "positive_class": "ood",
        "id_label": 0,
        "ood_label": 1,
        "input_score_orientation": "higher_is_id",
        "fpr95_definition": "ID false-positive rate at 95% OOD true-positive rate",
        "source_config": str(source_config_path),
        "experiment_name": config.experiment_name,
        "run_dir": config.run_dir,
        "checkpoint": checkpoint,
        "distillation_method": method,
        "inference_mode": inference_mode,
        "manifest_sizes": manifest_sizes,
        "artifact_manifest": str(artifact_dir / "manifest.json"),
        "scores": score_results,
    }
    write_json(metrics_path, payload)
    return payload


def _load_probability_scores(
    manifest: dict[str, object],
    method: DistillationMethod,
    checkpoint: str,
) -> dict[str, dict[str, np.ndarray]]:
    by_dataset: dict[str, dict[str, Path]] = {}
    for raw_artifact in manifest["artifacts"]:  # type: ignore[index]
        artifact = dict(raw_artifact)  # type: ignore[arg-type]
        dataset = str(artifact["dataset"])
        model = str(artifact["model"])
        if model == "student" and (
            artifact.get("method") != method
            or artifact.get("checkpoint") != checkpoint
        ):
            continue
        by_dataset.setdefault(dataset, {})[model] = Path(str(artifact["path"]))

    results = {name: {} for name in PROBABILITY_SCORE_NAMES}
    for dataset, paths in by_dataset.items():
        if set(paths) != {"teacher", "student"}:
            raise ValueError(f"Incomplete probability artifacts for {dataset}: {paths}")
        teacher = torch.load(paths["teacher"], map_location="cpu", weights_only=False)
        student = torch.load(paths["student"], map_location="cpu", weights_only=False)
        teacher_probabilities = _numpy(teacher["probabilities"])
        student_probabilities = _numpy(student["probabilities"])
        teacher_logits = _numpy(teacher["logits"])
        student_logits = _numpy(student["logits"])
        dataset_scores = {
            MAX_PROBABILITY_DIFFERENCE: max_probability_difference(
                teacher_probabilities, student_probabilities, signed=True
            ),
            ABSOLUTE_MAX_PROBABILITY_DIFFERENCE: absolute_max_probability_difference(
                teacher_probabilities, student_probabilities, signed=True
            ),
            STUDENT_TEACHER_KL_DIVERGENCE: student_teacher_kl_divergence_from_logits(
                teacher_logits, student_logits, signed=True
            ),
            LOGIT_L2_DISTANCE: logit_l2_distance(
                teacher_logits, student_logits, signed=True
            ),
            ENERGY_GAP: energy_gap(teacher_logits, student_logits, signed=True),
            ABSOLUTE_ENERGY_GAP: absolute_energy_gap(
                teacher_logits, student_logits, signed=True
            ),
            STUDENT_MSP: student_msp(student_probabilities, signed=True),
            STUDENT_ENERGY: student_energy(student_logits, signed=True),
            "teacher_msp": student_msp(teacher_probabilities, signed=True),
            "teacher_energy": student_energy(teacher_logits, signed=True),
        }
        for name, values in dataset_scores.items():
            results[name][dataset] = values
    return results


def _load_feature_denoising_scores(
    manifest: dict[str, object],
) -> dict[str, dict[str, np.ndarray]]:
    scores: dict[str, np.ndarray] = {}
    score_name = None
    for raw_artifact in manifest["artifacts"]:  # type: ignore[index]
        artifact = dict(raw_artifact)  # type: ignore[arg-type]
        path = Path(str(artifact["path"]))
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata", {})
        current_name = str(metadata.get("score", "feature_denoising_reconstruction_error"))
        if score_name is None:
            score_name = current_name
        elif current_name != score_name:
            raise ValueError("Feature Denoising artifacts use inconsistent scores")
        scores[str(artifact["dataset"])] = _numpy(payload["scores"])
    if score_name is None:
        raise ValueError("Feature Denoising inference produced no score artifacts")
    return {score_name: scores}


def _evaluate_scores(
    config: ExperimentConfig,
    scores: dict[str, np.ndarray],
) -> dict[str, Any]:
    id_name = f"{config.dataset.name}_test"
    try:
        id_scores = scores[id_name]
    except KeyError as error:
        raise ValueError(f"Missing OpenOOD ID scores: {id_name}") from error
    ood_scores = {
        f"{dataset.name}_{dataset.split}": scores[
            f"{dataset.name}_{dataset.split}"
        ]
        for dataset in config.dataset.ood_datasets
    }
    groups = {
        f"{dataset.name}_{dataset.split}": str(dataset.group)
        for dataset in config.dataset.ood_datasets
    }
    result = evaluate_openood_scores(id_scores, ood_scores, groups)
    result["sample_counts"] = {
        id_name: int(id_scores.size),
        **{name: int(values.size) for name, values in ood_scores.items()},
    }
    return result


def _validate_result_counts(
    score_results: dict[str, Any],
    expected: dict[str, int],
) -> None:
    for result in score_results.values():
        counts = result["sample_counts"]
        if counts != expected:
            raise ValueError(f"OpenOOD score counts do not match manifests: {counts} != {expected}")


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
