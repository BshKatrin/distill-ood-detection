#!/usr/bin/env bash
#SBATCH --job-name=ce_students
#SBATCH --partition=electronic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/%x-%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
cd "${PROJECT_DIR}"

mkdir -p slurm

uv sync

configs=(
  "configs/distill_linear_cifar10.yaml"
  "configs/distill_mlp_cifar10.yaml"
  "configs/distill_feature_linear_layer3_cifar10.yaml"
)

for config in "${configs[@]}"; do
  echo "[$(date --iso-8601=seconds)] Training cross_entropy: ${config}"
  uv run distill-ood train-student \
    --config "${config}" \
    --method cross_entropy

  echo "[$(date --iso-8601=seconds)] Inferring best student probabilities: ${config}"
  uv run distill-ood infer-probabilities \
    --config "${config}" \
    --method cross_entropy \
    --checkpoint best
done

echo "[$(date --iso-8601=seconds)] Done"
