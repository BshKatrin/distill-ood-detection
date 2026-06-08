"""Utilities for saving model inference artifacts."""

from distill_ood_detection.inference.outputs import (
    ModelOutputs,
    collect_model_outputs,
    save_model_outputs,
)

__all__ = ["ModelOutputs", "collect_model_outputs", "save_model_outputs"]
