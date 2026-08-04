#!/usr/bin/env bash
#SBATCH --job-name=fd_res_metrics
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/%x-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1
export UV_NO_PROGRESS=1
export PATH="${HOME}/.local/bin:${PATH}"

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_DIR}"

mkdir -p reports/outputs/json reports/outputs/latex slurm

configs=(
  configs/students/feature_denoising/feature_masking/cifar_10/resnet18/residual_layer1_blocks_3_5_mean.yaml
  configs/students/feature_denoising/feature_masking/cifar_10/resnet18/residual_layer2_blocks_3_5_mean.yaml
  configs/students/feature_denoising/feature_masking/cifar_10/resnet18/residual_layer3_blocks_1_3_mean.yaml
  configs/students/feature_denoising/feature_masking/cifar_10/resnet18/residual_layer4_block_1_mean.yaml
  configs/students/feature_denoising/feature_masking/cifar_100/resnet18/residual_layer1_blocks_3_5_mean.yaml
  configs/students/feature_denoising/feature_masking/cifar_100/resnet18/residual_layer2_blocks_3_5_mean.yaml
  configs/students/feature_denoising/feature_masking/cifar_100/resnet18/residual_layer3_blocks_1_3_mean.yaml
  configs/students/feature_denoising/feature_masking/cifar_100/resnet18/residual_layer4_block_1_mean.yaml
  configs/students/feature_denoising/feature_masking/cifar_10/resnet50/residual_layer1_blocks_3_5_mean.yaml
  configs/students/feature_denoising/feature_masking/cifar_10/resnet50/residual_layer2_blocks_3_5_mean.yaml
  configs/students/feature_denoising/feature_masking/cifar_10/resnet50/residual_layer3_blocks_1_3_mean.yaml
  configs/students/feature_denoising/feature_masking/cifar_10/resnet50/residual_layer4_block_1_mean.yaml
  configs/students/feature_denoising/feature_masking/cifar_100/resnet50/residual_layer1_blocks_3_5_mean.yaml
  configs/students/feature_denoising/feature_masking/cifar_100/resnet50/residual_layer2_blocks_3_5_mean.yaml
  configs/students/feature_denoising/feature_masking/cifar_100/resnet50/residual_layer3_blocks_1_3_mean.yaml
  configs/students/feature_denoising/feature_masking/cifar_100/resnet50/residual_layer4_block_1_mean.yaml
)

run_python=(uv run --project envs/gpu --no-sync python)

"${run_python[@]}" reports/scripts/export_feature_denoising_ood_metrics.py \
  --score default \
  --output reports/outputs/latex/spatial_block_residual_default.tex \
  --json-output reports/outputs/json/spatial_block_residual_default.json \
  "${configs[@]}"

"${run_python[@]}" reports/scripts/export_feature_denoising_ood_metrics.py \
  --score improvement \
  --output reports/outputs/latex/spatial_block_residual_improvement.tex \
  --json-output reports/outputs/json/spatial_block_residual_improvement.json \
  "${configs[@]}"

"${run_python[@]}" reports/scripts/export_feature_denoising_ood_metrics.py \
  --score relative_improvement \
  --output reports/outputs/latex/spatial_block_residual_relative_improvement.tex \
  --json-output reports/outputs/json/spatial_block_residual_relative_improvement.json \
  "${configs[@]}"

"${run_python[@]}" reports/scripts/export_feature_denoising_ood_metrics.py \
  --score negative_identity_error \
  --output reports/outputs/latex/spatial_block_residual_negative_identity.tex \
  --json-output reports/outputs/json/spatial_block_residual_negative_identity.json \
  "${configs[@]}"

"${run_python[@]}" reports/scripts/export_feature_denoising_reconstruction_losses.py \
  --output reports/outputs/latex/spatial_block_residual_losses.tex \
  --json-output reports/outputs/json/spatial_block_residual_losses.json \
  "${configs[@]}"
