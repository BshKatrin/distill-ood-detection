#!/usr/bin/env bash
#SBATCH --job-name=pixmix_smoke
#SBATCH --partition=electronic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/%x-%j.out

set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_DIR}"

uv sync --project envs/gpu --locked
uv run --project envs/gpu --no-sync python - <<'PY'
from pathlib import Path

import torch

from distill_ood_detection.config import load_config
from distill_ood_detection.datasets.pixmix import build_pixmix_mixing_provider
from distill_ood_detection.distillation.pixmix import sample_pixmix

config = load_config(
    Path(
        "configs/students/perturbation/pixel/pixmix/cifar_10/resnet50/"
        "linear_layer4_pixmix_avg.yaml"
    )
)
device = torch.device("cuda")
provider = build_pixmix_mixing_provider(
    config.dataset,
    config.strategy.perturbation.pixmix,
    config.training.defaults.seed,
)
images = torch.zeros(64, 3, 32, 32, device=device)
mixing_images = provider.sample(64, device, images.dtype)
batch = sample_pixmix(
    images,
    mixing_images,
    config.strategy.perturbation.pixmix,
    ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
)
assert batch.perturbed_images.shape == images.shape
assert torch.isfinite(batch.perturbed_images).all()
print("pixmix cuda smoke passed")
PY
