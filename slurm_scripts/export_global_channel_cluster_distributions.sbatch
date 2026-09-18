#!/usr/bin/env bash
#SBATCH --job-name=global_cluster_dist
#SBATCH --partition=electronic
#SBATCH --exclude=daft
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/%x-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1 UV_NO_PROGRESS=1 HF_HUB_OFFLINE=1
export PATH="${HOME}/.local/bin:${PATH}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
SOURCE_DIR="${SOURCE_DIR:-${PROJECT_DIR}}"
GPU_ENV_DIR="${GPU_ENV_DIR:-${SOURCE_DIR}/envs/gpu}"
CHECKPOINT="${CHECKPOINT:-best}"
export PYTHONPATH="${SOURCE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_DIR}"
mkdir -p slurm

if [ "$#" -eq 0 ]; then
  echo "Usage: sbatch $0 <global-student-config> [...]"
  exit 1
fi
if [ "${SKIP_UV_SYNC:-0}" != "1" ]; then
  uv sync --project "${GPU_ENV_DIR}" --locked
fi
for config in "$@"; do
  uv run --project "${GPU_ENV_DIR}" --no-sync python -m distill_ood_detection.cli \
    export-global-channel-cluster-distributions \
    --config "${config}" \
    --checkpoint "${CHECKPOINT}"
done
