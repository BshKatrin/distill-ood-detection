#!/usr/bin/env bash
#SBATCH --job-name=fd_layer_comp
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/%x-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1
export UV_NO_PROGRESS=1
export PATH="${HOME}/.local/bin:${PATH}"
PS4='[$(date --iso-8601=seconds)] + '
set -x

trap 'status=$?; echo "[$(date --iso-8601=seconds)] Failed at line ${LINENO} with exit status ${status}"; exit "${status}"' ERR

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_DIR}"

mkdir -p slurm

echo "[$(date --iso-8601=seconds)] Syncing uv notebooks environment"
uv --version
uv sync --project envs/notebooks --locked

echo "[$(date --iso-8601=seconds)] Building layer-composition report"
uv run --project envs/notebooks --no-sync python \
  reports/scripts/build_feature_denoising_layer_composition_report.py

echo "[$(date --iso-8601=seconds)] Layer-composition report complete"
