#!/usr/bin/env bash
#SBATCH --job-name=global_cluster_panel
#SBATCH --partition=electronic
#SBATCH --exclude=daft
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/%x-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1 UV_NO_PROGRESS=1
export PATH="${HOME}/.local/bin:${PATH}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
SOURCE_DIR="${SOURCE_DIR:-${PROJECT_DIR}}"
DASHBOARD_ENV_DIR="${DASHBOARD_ENV_DIR:-${SOURCE_DIR}/envs/dashboard}"
SUMMARY_DIR="${SUMMARY_DIR:-runs/global_channel_cluster_distributions/nmf_latent_cosine/summary}"
PORT="${PORT:-5007}"
ADDRESS="${ADDRESS:-0.0.0.0}"
export GLOBAL_CLUSTER_DISTRIBUTION_SUMMARY_DIR="${SUMMARY_DIR}"
export PYTHONPATH="${SOURCE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_DIR}"
mkdir -p slurm

if [ "${SKIP_UV_SYNC:-0}" != "1" ]; then
  uv sync --project "${DASHBOARD_ENV_DIR}" --locked
fi
echo "Panel node: $(hostname); address: ${ADDRESS}; port: ${PORT}"
uv run --project "${DASHBOARD_ENV_DIR}" --no-sync panel serve \
  "${SOURCE_DIR}/src/distill_ood_detection/dashboard/global_channel_cluster_distributions.py" \
  --address "${ADDRESS}" \
  --port "${PORT}" \
  --allow-websocket-origin "localhost:${PORT}" \
  --num-procs 1
