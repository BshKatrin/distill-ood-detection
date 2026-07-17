#!/usr/bin/env bash
#SBATCH --job-name=fd_subspaces
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
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_DIR}"
mkdir -p slurm

if [ "$#" -eq 0 ]; then
  echo "Usage: sbatch $0 <layer4-feature-denoising-config> [...]"
  exit 1
fi

echo "[$(date --iso-8601=seconds)] Syncing uv GPU environment"
uv --version
uv sync --project envs/gpu --locked

for config in "$@"; do
  echo "[$(date --iso-8601=seconds)] Exporting activation-subspace errors: ${config}"
  uv run --project envs/gpu --no-sync python -m distill_ood_detection.cli \
    export-feature-denoising-subspace-errors \
    --config "${config}" \
    --checkpoint best
done

echo "[$(date --iso-8601=seconds)] Feature Denoising subspace export complete"
