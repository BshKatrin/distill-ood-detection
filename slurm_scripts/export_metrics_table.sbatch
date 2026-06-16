#!/usr/bin/env bash
#SBATCH --job-name=ood_metrics
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

CACHE_PATH="${CACHE_PATH:-}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

args=(reports/scripts/export_metrics_table.py)

if [ -n "${CACHE_PATH}" ]; then
  args+=(--cache "${CACHE_PATH}")
fi

if [ -n "${OUTPUT_PATH}" ]; then
  args+=(--output "${OUTPUT_PATH}")
fi

if [ -n "${OUTPUT_DIR}" ]; then
  args+=(--output-dir "${OUTPUT_DIR}")
fi

args+=("$@")

echo "[$(date --iso-8601=seconds)] Syncing uv notebooks environment"
uv --version
uv sync --project envs/notebooks --locked

echo "[$(date --iso-8601=seconds)] Exporting OOD metrics tables on CPU"
uv run --project envs/notebooks --no-sync python "${args[@]}"

echo "[$(date --iso-8601=seconds)] OOD metrics table export complete"
