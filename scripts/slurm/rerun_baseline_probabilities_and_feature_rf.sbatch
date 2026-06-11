#!/usr/bin/env bash
#SBATCH --job-name=baseline_chore
#SBATCH --partition=electronic
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
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

baseline_probability_configs=(
  "configs/baseline/cifar_10/linear.yaml"
  "configs/baseline/cifar_10/mlp.yaml"
  "configs/baseline/cifar_10/feature_linear_layer3.yaml"
  "configs/baseline/cifar_10/random_forest.yaml"
  "configs/baseline/cifar_100/linear.yaml"
  "configs/baseline/cifar_100/mlp.yaml"
  "configs/baseline/cifar_100/feature_linear_layer3.yaml"
  "configs/baseline/cifar_100/random_forest.yaml"
)

feature_random_forest_configs=(
  "configs/baseline/cifar_10/feature_random_forest_layer3.yaml"
  "configs/baseline/cifar_100/feature_random_forest_layer3.yaml"
)

rerun_baseline_probabilities() {
  export CUDA_VISIBLE_DEVICES="${BASELINE_PROBABILITIES_GPU:-0}"
  for config in "${baseline_probability_configs[@]}"; do
    echo "[$(date --iso-8601=seconds)] Inferring baseline probabilities: ${config}"
    run_distill_ood infer-probabilities \
      --config "${config}" \
      --checkpoint best
  done
}

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

rerun_baseline_probabilities &
baseline_pid=$!

train_and_infer_feature_random_forests &
feature_rf_pid=$!

wait "${baseline_pid}"
wait "${feature_rf_pid}"

echo "[$(date --iso-8601=seconds)] Baseline probability and layer3 random-forest chores complete"
