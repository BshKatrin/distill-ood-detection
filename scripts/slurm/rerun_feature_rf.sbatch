#!/usr/bin/env bash
#SBATCH --job-name=feature_rf
#SBATCH --partition=electronic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
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

echo "[$(date --iso-8601=seconds)] Syncing uv GPU environment"
uv --version
uv sync --project envs/gpu --locked

run_distill_ood() {
  uv run --project envs/gpu --no-sync python -m distill_ood_detection.cli "$@"
}

feature_random_forest_configs=(
  "configs/baseline/cifar_10/feature_random_forest_layer3.yaml"
)

train_and_infer_feature_random_forests() {
  export CUDA_VISIBLE_DEVICES="${FEATURE_RANDOM_FOREST_GPU:-1}"
  for config in "${feature_random_forest_configs[@]}"; do
    echo "[$(date --iso-8601=seconds)] Training layer3 random forest: ${config}"
    run_distill_ood train-tree-student \
      --config "${config}"

    echo "[$(date --iso-8601=seconds)] Inferring layer3 random-forest probabilities: ${config}"
    run_distill_ood infer-probabilities \
      --config "${config}" \
      --checkpoint best
  done
}

train_and_infer_feature_random_forests &
feature_rf_pid=$!

wait "${feature_rf_pid}"

echo "[$(date --iso-8601=seconds)] Layer3 random-forest chores complete"
