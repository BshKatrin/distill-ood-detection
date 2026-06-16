#!/usr/bin/env bash
#SBATCH --job-name=teacher_acts
#SBATCH --partition=electronic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
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

CONFIG_PATH="${1:-configs/teachers/resnet18_cifar10_layers.yaml}"

echo "[$(date --iso-8601=seconds)] Syncing uv GPU environment"
uv --version
uv sync --project envs/gpu --locked

echo "[$(date --iso-8601=seconds)] Exporting teacher activations: ${CONFIG_PATH}"
uv run --project envs/gpu --no-sync python -m distill_ood_detection.cli \
  export-teacher-activations \
  --config "${CONFIG_PATH}"

echo "[$(date --iso-8601=seconds)] Teacher activation export complete"
