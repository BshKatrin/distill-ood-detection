"""Utilities for saving model inference artifacts."""

from distill_ood_detection.inference.outputs import (
    ModelOutputs,
    collect_model_outputs,
    collect_tree_model_outputs,
    predict_tree_model_outputs,
    save_model_outputs,
)

__all__ = [
    "ModelOutputs",
    "collect_model_outputs",
    "collect_tree_model_outputs",
    "predict_tree_model_outputs",
    "save_model_outputs",
]
