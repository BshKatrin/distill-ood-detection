#!/usr/bin/env bash
#SBATCH --job-name=cifar10_perturb
#SBATCH --partition=electronic
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=09:00:00
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

configs=(
  "configs/perturbation/cifar_10/linear_layer3_clip_constant.yaml"
  "configs/perturbation/cifar_10/linear_layer3_clip_channel.yaml"
  "configs/perturbation/cifar_10/linear_layer3_clip_spatial.yaml"
)

if [ "${#configs[@]}" -eq 0 ]; then
  echo "[$(date --iso-8601=seconds)] No perturbation configs provided"
  exit 0
fi

if [ "${#configs[@]}" -gt "${SLURM_GPUS_ON_NODE:-0}" ]; then
  echo "[$(date --iso-8601=seconds)] Requested ${#configs[@]} config pipelines but only ${SLURM_GPUS_ON_NODE:-0} GPUs are available on this node"
  exit 1
fi

train_and_infer_config() {
  local gpu_id="$1"
  local config="$2"

  export CUDA_VISIBLE_DEVICES="${gpu_id}"

  echo "[$(date --iso-8601=seconds)] [gpu=${gpu_id}] Training perturbation student: ${config}"
  run_distill_ood train-student --config "${config}"

  echo "[$(date --iso-8601=seconds)] [gpu=${gpu_id}] Inferring best perturbation probabilities: ${config}"
  run_distill_ood infer-probabilities \
    --config "${config}" \
    --checkpoint best
}

pids=()

for idx in "${!configs[@]}"; do
  train_and_infer_config "${idx}" "${configs[${idx}]}" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

echo "[$(date --iso-8601=seconds)] CIFAR-10 perturbation training and inference complete"
