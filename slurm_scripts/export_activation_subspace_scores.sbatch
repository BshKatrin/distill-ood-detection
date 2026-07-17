#!/usr/bin/env bash
#SBATCH --job-name=actsub_scores
#SBATCH --partition=electronic
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/%x-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1
export UV_NO_PROGRESS=1
export PATH="${HOME}/.local/bin:${PATH}"
export HF_HUB_OFFLINE=1

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_DIR}"
mkdir -p slurm

configs=("$@")
if [ "${#configs[@]}" -eq 0 ]; then
  echo "No activation-subspace configs provided"
  exit 1
fi

cifar10_configs=()
cifar100_configs=()
for config in "${configs[@]}"; do
  if [ ! -f "${config}" ]; then
    echo "Config does not exist: ${config}"
    exit 1
  fi
  case "${config}" in
    */cifar_10/*) cifar10_configs+=("${config}") ;;
    */cifar_100/*) cifar100_configs+=("${config}") ;;
    *)
      echo "Config path must identify cifar_10 or cifar_100: ${config}"
      exit 1
      ;;
  esac
done

echo "[$(date --iso-8601=seconds)] Syncing uv GPU environment"
uv --version
uv sync --project envs/gpu --locked

run_group() {
  local gpu_id="$1"
  local teacher_embedding_dir="$2"
  shift 2
  local args=(
    export-activation-subspace-scores
    --teacher-embedding-dir "${teacher_embedding_dir}"
    --checkpoint best
  )
  local config
  for config in "$@"; do
    args+=(--config "${config}")
  done
  echo "[$(date --iso-8601=seconds)] [gpu=${gpu_id}] Scoring $# student(s)"
  CUDA_VISIBLE_DEVICES="${gpu_id}" uv run --project envs/gpu --no-sync \
    python -m distill_ood_detection.cli "${args[@]}"
}

pids=()
if [ "${#cifar10_configs[@]}" -gt 0 ]; then
  run_group \
    0 \
    runs/teachers/cifar_10/resnet18/activation_subspace_embeddings \
    "${cifar10_configs[@]}" &
  pids+=("$!")
fi
if [ "${#cifar100_configs[@]}" -gt 0 ]; then
  run_group \
    1 \
    runs/teachers/cifar_100/resnet18/activation_subspace_embeddings \
    "${cifar100_configs[@]}" &
  pids+=("$!")
fi

for pid in "${pids[@]}"; do
  wait "${pid}"
done

echo "[$(date --iso-8601=seconds)] Activation-subspace score export complete"
