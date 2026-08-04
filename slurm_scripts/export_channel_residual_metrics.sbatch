#!/usr/bin/env bash
#SBATCH --job-name=fd_channel_metrics
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

config_list="slurm_scripts/feature_denoising_channel_residual_resnet18_resnet50.txt"
configs=()
while IFS= read -r config || [ -n "${config}" ]; do
  case "${config}" in
    ""|\#*) continue ;;
    *) configs+=("${config}") ;;
  esac
done < "${config_list}"

run_python=(uv run --project envs/gpu --no-sync python)

export_ood_metrics() {
  local score="$1"
  local output_name="$2"
  "${run_python[@]}" reports/scripts/export_feature_denoising_ood_metrics.py \
    --score "${score}" \
    --output "reports/outputs/latex/${output_name}.tex" \
    --json-output "reports/outputs/json/${output_name}.json" \
    "${configs[@]}"
}

export_ood_metrics default channel_residual_default
export_ood_metrics improvement channel_residual_improvement
export_ood_metrics relative_improvement channel_residual_relative_improvement
export_ood_metrics negative_identity_error channel_residual_negative_identity

"${run_python[@]}" reports/scripts/export_feature_denoising_reconstruction_losses.py \
  --output reports/outputs/latex/channel_residual_losses.tex \
  --json-output reports/outputs/json/channel_residual_losses.json \
  "${configs[@]}"
