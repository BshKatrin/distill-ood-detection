#!/usr/bin/env bash
#SBATCH --job-name=teacher_probs
#SBATCH --partition=electronic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
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

configs=("$@")

if [ -n "${CONFIG_LIST:-}" ]; then
  if [ ! -f "${CONFIG_LIST}" ]; then
    echo "[$(date --iso-8601=seconds)] CONFIG_LIST does not exist: ${CONFIG_LIST}"
    exit 1
  fi

  while IFS= read -r config || [ -n "${config}" ]; do
    case "${config}" in
      ""|\#*) continue ;;
      *) configs+=("${config}") ;;
    esac
  done < "${CONFIG_LIST}"
fi

if [ "${#configs[@]}" -eq 0 ]; then
  configs=("configs/teachers/resnet50_cifar10_probabilities.yaml")
fi

echo "[$(date --iso-8601=seconds)] Syncing uv GPU environment"
uv --version
uv sync --project envs/gpu --locked

for config in "${configs[@]}"; do
  if [ ! -f "${config}" ]; then
    echo "[$(date --iso-8601=seconds)] Config does not exist: ${config}"
    exit 1
  fi

  echo "[$(date --iso-8601=seconds)] Exporting teacher probabilities: ${config}"
  uv run --project envs/gpu --no-sync python -m distill_ood_detection.cli \
    export-teacher-probabilities \
    --config "${config}"
done

echo "[$(date --iso-8601=seconds)] Teacher probability export complete"
