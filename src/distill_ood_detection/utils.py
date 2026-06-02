"""Small utilities for reproducible experiments."""

from __future__ import annotations

import json
import random
import warnings
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set common random seeds for reproducible training."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(requested: str) -> torch.device:
    """Resolve a requested device string into a PyTorch device."""

    if requested != "auto":
        device = torch.device(requested)
        _validate_device(device)
        return device
    if torch.cuda.is_available():
        cuda_device = torch.device("cuda")
        if _can_execute_on_device(cuda_device):
            return cuda_device
        warnings.warn(
            "CUDA is visible but unusable for this run; falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _can_execute_on_device(device: torch.device) -> bool:
    if device.type != "cuda":
        return True
    try:
        images = torch.zeros((1, 3, 8, 8), device=device)
        weights = torch.zeros((4, 3, 3, 3), device=device)
        torch.nn.functional.conv2d(images, weights)
        torch.cuda.synchronize(device)
    except RuntimeError:
        return False
    return True


def _validate_device(device: torch.device) -> None:
    if device.type == "cuda" and not _can_execute_on_device(device):
        raise RuntimeError(
            "CUDA was requested, but PyTorch could not execute a small CUDA "
            "convolution. Try running inside a GPU allocation, for example with "
            "srun/sbatch, or set training.device to 'cpu'."
        )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
