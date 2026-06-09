#!/usr/bin/env bash
#SBATCH --job-name=cifar100_ood
#SBATCH --partition=electronic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/%x-%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
cd "${PROJECT_DIR}"

mkdir -p slurm

echo "[$(date --iso-8601=seconds)] Syncing uv environment"
uv sync

torch_configs=(
  "configs/distill_linear_cifar100.yaml"
  "configs/distill_mlp_cifar100.yaml"
  "configs/distill_feature_linear_layer3_cifar100.yaml"
)

tree_configs=(
  "configs/distill_random_forest_cifar100.yaml"
)

for config in "${torch_configs[@]}"; do
  echo "[$(date --iso-8601=seconds)] Training PyTorch student: ${config}"
  uv run distill-ood train-student \
    --config "${config}"

  echo "[$(date --iso-8601=seconds)] Inferring best PyTorch student probabilities: ${config}"
  uv run distill-ood infer-probabilities \
    --config "${config}" \
    --checkpoint best
done

for config in "${tree_configs[@]}"; do
  echo "[$(date --iso-8601=seconds)] Training tree student: ${config}"
  uv run distill-ood train-tree-student \
    --config "${config}"

  echo "[$(date --iso-8601=seconds)] Inferring best tree student probabilities: ${config}"
  uv run distill-ood infer-probabilities \
    --config "${config}" \
    --checkpoint best
done

experiment_names=(
  "linear_student_resnet18_cifar100"
  "mlp_student_resnet18_cifar100"
  "feature_mlp_layer3_student_resnet18_cifar100"
  "random_forest_student_resnet18_cifar100"
)

for experiment_name in "${experiment_names[@]}"; do
  manifest="runs/${experiment_name}/probabilities/manifest.json"
  echo "[$(date --iso-8601=seconds)] Reporting OOD metrics: ${manifest}"
  uv run distill-ood report-ood \
    --manifest "${manifest}"
done

echo "[$(date --iso-8601=seconds)] CIFAR-100 training, inference, and reporting complete"
