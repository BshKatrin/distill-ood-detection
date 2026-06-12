#!/usr/bin/env bash
#SBATCH --job-name=cifar10_perturb
#SBATCH --partition=electronic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/%x-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1
export UV_NO_PROGRESS=1
PS4='[$(date --iso-8601=seconds)] + '
set -x

trap 'status=$?; echo "[$(date --iso-8601=seconds)] Failed at line ${LINENO} with exit status ${status}"; exit "${status}"' ERR

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_DIR}"

mkdir -p slurm

echo "[$(date --iso-8601=seconds)] Syncing uv GPU environment"
uv --version
uv sync --project envs/gpu --locked

configs=(
  "configs/perturbation/cifar_10/linear_layer3_clip_constant.yaml"
  "configs/perturbation/cifar_10/linear_layer3_clip_channel.yaml"
  "configs/perturbation/cifar_10/linear_layer3_clip_spatial.yaml"
)

for config in "${configs[@]}"; do
  echo "[$(date --iso-8601=seconds)] Inferring best perturbation probabilities: ${config}"
  uv run --project envs/gpu --no-sync python -m distill_ood_detection.cli infer-probabilities \
    --config "${config}" --checkpoint best
done

echo "[$(date --iso-8601=seconds)] CIFAR-10 perturbation training and inference complete"
