#!/usr/bin/env bash
#SBATCH --job-name=cifar100_ood
#SBATCH --partition=electronic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
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

echo "[$(date --iso-8601=seconds)] Syncing uv environment"
uv --version
uv sync --project envs/gpu --locked

torch_configs=(
  "configs/baseline/cifar_100/linear.yaml"
  "configs/baseline/cifar_100/mlp.yaml"
  "configs/baseline/cifar_100/feature_linear_layer3.yaml"
)

tree_configs=(
  "configs/baseline/cifar_100/random_forest.yaml"
)

for config in "${torch_configs[@]}"; do
  echo "[$(date --iso-8601=seconds)] Training PyTorch student: ${config}"
  uv run --project envs/gpu --no-sync python -m distill_ood_detection.cli train-student \
    --config "${config}"

  echo "[$(date --iso-8601=seconds)] Inferring best PyTorch student probabilities: ${config}"
  uv run --project envs/gpu --no-sync python -m distill_ood_detection.cli infer-probabilities \
    --config "${config}" --checkpoint best

done

for config in "${tree_configs[@]}"; do
  echo "[$(date --iso-8601=seconds)] Training tree student: ${config}"
  uv run --project envs/gpu --no-sync python -m distill_ood_detection.cli train-tree-student \
    --config "${config}"

  echo "[$(date --iso-8601=seconds)] Inferring best tree student probabilities: ${config}"
  uv run --project envs/gpu --no-sync python -m distill_ood_detection.cli infer-probabilities \
    --config "${config}" --checkpoint best
  
done

echo "[$(date --iso-8601=seconds)] CIFAR-100 training and inference complete"
