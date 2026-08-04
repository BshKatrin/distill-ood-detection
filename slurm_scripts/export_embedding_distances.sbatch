#!/usr/bin/env bash
#SBATCH --job-name=embedding_dist
#SBATCH --partition=electronic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/%x-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1
export UV_NO_PROGRESS=1
export PATH="${HOME}/.local/bin:${PATH}"
export HF_HUB_OFFLINE=1
PS4='[$(date --iso-8601=seconds)] + '
set -x

trap 'status=$?; echo "[$(date --iso-8601=seconds)] Failed at line ${LINENO} with exit status ${status}"; exit "${status}"' ERR

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
SOURCE_DIR="${EMBEDDING_DISTANCE_SOURCE_DIR:-${PROJECT_DIR}}"
export PYTHONPATH="${SOURCE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_DIR}"

mkdir -p slurm

CONFIG_PATH="${1:?Usage: sbatch slurm_scripts/export_embedding_distances.sbatch CONFIG_PATH}"

echo "[$(date --iso-8601=seconds)] Syncing uv GPU environment"
uv --version
uv sync --project envs/gpu --locked

echo "[$(date --iso-8601=seconds)] Exporting embedding distances: ${CONFIG_PATH}"
uv run --project envs/gpu --no-sync python -m distill_ood_detection.cli \
  export-embedding-distances \
  --config "${CONFIG_PATH}"

echo "[$(date --iso-8601=seconds)] Embedding-distance export complete"
