"""Utilities for saving model inference artifacts."""

from distill_ood_detection.inference.outputs import (
    ModelOutputs,
    collect_feature_tree_model_outputs,
    collect_feature_model_outputs,
    collect_model_outputs,
    collect_perturbed_teacher_outputs,
    collect_perturbation_tree_model_outputs,
    collect_perturbation_model_outputs,
    collect_tree_model_outputs,
    predict_tree_model_outputs,
    save_model_outputs,
)

__all__ = [
    "ModelOutputs",
    "collect_feature_tree_model_outputs",
    "collect_feature_model_outputs",
    "collect_model_outputs",
    "collect_perturbed_teacher_outputs",
    "collect_perturbation_tree_model_outputs",
    "collect_perturbation_model_outputs",
    "collect_tree_model_outputs",
    "predict_tree_model_outputs",
    "save_model_outputs",
]
