#!/usr/bin/env bash
#SBATCH --job-name=feature_denoising_losses
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
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

OUTPUT_PATH="${OUTPUT_PATH:-}"
JSON_OUTPUT="${JSON_OUTPUT:-}"

args=(reports/scripts/export_feature_denoising_reconstruction_losses.py)

if [ -n "${OUTPUT_PATH}" ]; then
  args+=(--output "${OUTPUT_PATH}")
fi

if [ -n "${JSON_OUTPUT}" ]; then
  args+=(--json-output "${JSON_OUTPUT}")
fi

args+=("$@")

echo "[$(date --iso-8601=seconds)] Syncing uv notebooks environment"
uv --version
uv sync --project envs/notebooks --locked

echo "[$(date --iso-8601=seconds)] Exporting Feature Denoising reconstruction losses on CPU"
uv run --project envs/notebooks --no-sync python "${args[@]}"

echo "[$(date --iso-8601=seconds)] Feature Denoising reconstruction loss export complete"
