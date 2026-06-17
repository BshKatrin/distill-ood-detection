"""Small utilities for reproducible experiments."""

from __future__ import annotations

import json
import os
import random
import warnings
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv


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
        can_execute, error = _can_execute_on_device(cuda_device)
        if can_execute:
            return cuda_device
        warnings.warn(
            "CUDA is visible but unusable for this run; falling back to CPU. "
            f"PyTorch error: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _can_execute_on_device(device: torch.device) -> tuple[bool, str | None]:
    if device.type != "cuda":
        return True, None
    try:
        images = torch.zeros((1, 3, 8, 8), device=device)
        weights = torch.zeros((4, 3, 3, 3), device=device)
        torch.nn.functional.conv2d(images, weights)
        torch.cuda.synchronize(device)
    except RuntimeError as error:
        return False, str(error)
    return True, None


def _validate_device(device: torch.device) -> None:
    if device.type != "cuda":
        return
    can_execute, error = _can_execute_on_device(device)
    if not can_execute:
        raise RuntimeError(
            "CUDA was requested, but PyTorch could not execute a small CUDA "
            "convolution. Try running inside a GPU allocation, for example with "
            "srun/sbatch, or set training.device to 'cpu'. "
            f"PyTorch error: {error}"
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


@lru_cache(maxsize=1)
def load_project_dotenv() -> None:
    """Load the repository ``.env`` file once when present."""

    current = Path(__file__).resolve()
    for parent in current.parents:
        dotenv_path = parent / ".env"
        if dotenv_path.is_file():
            load_dotenv(dotenv_path=dotenv_path, override=False)
            return


def get_hf_token() -> str | None:
    """Return the Hugging Face access token from the environment."""

    load_project_dotenv()
    return os.getenv("HF_TOKEN") or os.getenv("HF_token")


def get_data_dir_override() -> str | None:
    """Return the shared dataset directory override from the environment."""

    load_project_dotenv()
    return os.getenv("DISTILL_OOD_DATA_DIR")
