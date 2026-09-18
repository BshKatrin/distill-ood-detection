#!/usr/bin/env bash
#SBATCH --job-name=nmf_audit_panel
#SBATCH --partition=electronic
#SBATCH --exclude=daft
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/%x-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1 UV_NO_PROGRESS=1
export PATH="${HOME}/.local/bin:${PATH}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
SOURCE_DIR="${SOURCE_DIR:-${PROJECT_DIR}}"
DASHBOARD_ENV_DIR="${DASHBOARD_ENV_DIR:-${SOURCE_DIR}/envs/dashboard}"
PORT="${PORT:-5006}"
ADDRESS="${ADDRESS:-0.0.0.0}"
export CHANNEL_CLUSTER_AUDIT_SUMMARY_DIR="${SUMMARY_DIR:-runs/channel_cluster_audit/nmf_latent_cosine/summary}"
export PYTHONPATH="${SOURCE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_DIR}"
mkdir -p slurm
if [ "${SKIP_UV_SYNC:-0}" != "1" ]; then
  uv sync --project "${DASHBOARD_ENV_DIR}" --locked
fi
echo "Panel node: $(hostname); address: ${ADDRESS}; port: ${PORT}"
uv run --project "${DASHBOARD_ENV_DIR}" --no-sync panel serve \
  "${SOURCE_DIR}/src/distill_ood_detection/dashboard/channel_cluster_audit.py" \
  --address "${ADDRESS}" \
  --port "${PORT}" \
  --allow-websocket-origin "localhost:${PORT}" \
  --num-procs 1
