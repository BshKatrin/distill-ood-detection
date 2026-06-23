#!/usr/bin/env bash
#SBATCH --job-name=test_metrics
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
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

OUTPUT_PATH="${OUTPUT_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
JSON_OUTPUT="${JSON_OUTPUT:-}"

args=(reports/scripts/export_test_metrics_table.py)

if [ -n "${OUTPUT_PATH}" ]; then
  args+=(--output "${OUTPUT_PATH}")
fi

if [ -n "${OUTPUT_DIR}" ]; then
  args+=(--output-dir "${OUTPUT_DIR}")
fi

if [ -n "${JSON_OUTPUT}" ]; then
  args+=(--json-output "${JSON_OUTPUT}")
fi

args+=("$@")

echo "[$(date --iso-8601=seconds)] Syncing uv notebooks environment"
uv --version
uv sync --project envs/notebooks --locked

echo "[$(date --iso-8601=seconds)] Exporting test metrics tables on CPU"
uv run --project envs/notebooks --no-sync python "${args[@]}"

echo "[$(date --iso-8601=seconds)] Test metrics table export complete"
