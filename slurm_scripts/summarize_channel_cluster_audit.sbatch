#!/usr/bin/env bash
#SBATCH --job-name=nmf_audit_summary
#SBATCH --partition=electronic
#SBATCH --exclude=daft
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/%x-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1 UV_NO_PROGRESS=1
export PATH="${HOME}/.local/bin:${PATH}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
SOURCE_DIR="${SOURCE_DIR:-${PROJECT_DIR}}"
DASHBOARD_ENV_DIR="${DASHBOARD_ENV_DIR:-${SOURCE_DIR}/envs/dashboard}"
MANIFEST="${MANIFEST:-runs/channel_cluster_audit/nmf_latent_cosine/manifest.json}"
SUMMARY_DIR="${SUMMARY_DIR:-runs/channel_cluster_audit/nmf_latent_cosine/summary}"
export PYTHONPATH="${SOURCE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_DIR}"
mkdir -p slurm
if [ "${SKIP_UV_SYNC:-0}" != "1" ]; then
  uv sync --project "${DASHBOARD_ENV_DIR}" --locked
fi
uv run --project "${DASHBOARD_ENV_DIR}" --no-sync python -m distill_ood_detection.cli \
  summarize-channel-cluster-audit \
  --manifest "${MANIFEST}" \
  --output-dir "${SUMMARY_DIR}" \
  --require-complete
