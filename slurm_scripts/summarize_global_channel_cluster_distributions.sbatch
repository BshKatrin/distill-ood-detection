#!/usr/bin/env bash
#SBATCH --job-name=global_cluster_summary
#SBATCH --partition=electronic
#SBATCH --exclude=daft
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/%x-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1 UV_NO_PROGRESS=1
export PATH="${HOME}/.local/bin:${PATH}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
SOURCE_DIR="${SOURCE_DIR:-${PROJECT_DIR}}"
DASHBOARD_ENV_DIR="${DASHBOARD_ENV_DIR:-${SOURCE_DIR}/envs/dashboard}"
SUMMARY_DIR="${SUMMARY_DIR:-runs/global_channel_cluster_distributions/nmf_latent_cosine/summary}"
CHECKPOINT="${CHECKPOINT:-best}"
export PYTHONPATH="${SOURCE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_DIR}"
mkdir -p slurm

if [ "$#" -eq 0 ]; then
  echo "Usage: sbatch $0 <global-student-config> [...]"
  exit 1
fi
if [ "${SKIP_UV_SYNC:-0}" != "1" ]; then
  uv sync --project "${DASHBOARD_ENV_DIR}" --locked
fi
config_args=()
for config in "$@"; do
  config_args+=(--config "${config}")
done
uv run --project "${DASHBOARD_ENV_DIR}" --no-sync python -m distill_ood_detection.cli \
  summarize-global-channel-cluster-distributions \
  "${config_args[@]}" \
  --output-dir "${SUMMARY_DIR}" \
  --checkpoint "${CHECKPOINT}"
