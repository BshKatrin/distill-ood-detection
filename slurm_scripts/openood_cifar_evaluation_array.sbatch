#!/usr/bin/env bash
#SBATCH --job-name=openood_cifar_eval
#SBATCH --partition=electronic,hard
#SBATCH --exclude=daft
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/%x-%A_%a.out

set -euo pipefail
export PYTHONUNBUFFERED=1
export UV_NO_PROGRESS=1
export PATH="${HOME}/.local/bin:${PATH}"
export HF_HUB_OFFLINE=1

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
SOURCE_DIR="${SOURCE_DIR:-${PROJECT_DIR}}"
GPU_ENV_DIR="${GPU_ENV_DIR:-${PROJECT_DIR}/envs/gpu}"
MANIFEST="${MANIFEST:?MANIFEST is required}"
OPENOOD_ROOT="${OPENOOD_ROOT:-/home/bogush/openood}"
export PYTHONPATH="${SOURCE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${PROJECT_DIR}"
mkdir -p slurm

uv run --project "${GPU_ENV_DIR}" --no-sync \
  python -m distill_ood_detection.cli run-openood-cifar-task \
  --manifest "${MANIFEST}" \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --openood-root "${OPENOOD_ROOT}"
