#!/usr/bin/env bash
#SBATCH --job-name=channel_groups
#SBATCH --partition=electronic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
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
SOURCE_DIR="${SOURCE_DIR:-${PROJECT_DIR}}"
GPU_ENV_DIR="${GPU_ENV_DIR:-${SOURCE_DIR}/envs/gpu}"
export PYTHONPATH="${SOURCE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_DIR}"

mkdir -p slurm

configs=("$@")
if [ "${#configs[@]}" -eq 0 ]; then
  echo "Usage: sbatch slurm_scripts/build_channel_groups.sbatch configs/channel_grouping/...yaml [configs/channel_grouping/...yaml ...]"
  exit 1
fi

uv --version
if [ "${SKIP_UV_SYNC:-0}" = "1" ]; then
  echo "[$(date --iso-8601=seconds)] Reusing existing uv GPU environment"
else
  echo "[$(date --iso-8601=seconds)] Syncing uv GPU environment"
  uv sync --project "${GPU_ENV_DIR}" --locked
fi

for config in "${configs[@]}"; do
  if [ ! -f "${config}" ]; then
    echo "[$(date --iso-8601=seconds)] Config does not exist: ${config}"
    exit 1
  fi
  echo "[$(date --iso-8601=seconds)] Building channel groups: ${config}"
  uv run --project "${GPU_ENV_DIR}" --no-sync python -m distill_ood_detection.cli \
    build-channel-groups \
    --config "${config}"
done

echo "[$(date --iso-8601=seconds)] Channel grouping complete"
