#!/usr/bin/env bash
#SBATCH --job-name=cifar10_perturb
#SBATCH --partition=electronic
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
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

get_allocated_gpu_count() {
  if [ -n "${SLURM_STEP_GPUS:-}" ]; then
    echo "${SLURM_STEP_GPUS}" | tr ',' '\n' | sed '/^$/d' | wc -l | tr -d ' '
    return
  fi

  if [ -n "${SLURM_JOB_GPUS:-}" ]; then
    echo "${SLURM_JOB_GPUS}" | tr ',' '\n' | sed '/^$/d' | wc -l | tr -d ' '
    return
  fi

  if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
    echo "${SLURM_GPUS_ON_NODE}"
    return
  fi

  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | sed '/^$/d' | wc -l | tr -d ' '
    return
  fi

  echo 0
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

gpu_count="$(get_allocated_gpu_count)"

if ! [[ "${gpu_count}" =~ ^[0-9]+$ ]] || [ "${gpu_count}" -lt 1 ]; then
  echo "[$(date --iso-8601=seconds)] Could not determine a positive allocated GPU count from Slurm"
  exit 1
fi

parallel_jobs="${gpu_count}"
if [ "${parallel_jobs}" -gt "${#configs[@]}" ]; then
  parallel_jobs="${#configs[@]}"
fi

echo "[$(date --iso-8601=seconds)] Using ${parallel_jobs} parallel config pipeline(s) across ${gpu_count} allocated GPU(s)"

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
gpu_ids=()

for idx in "${!configs[@]}"; do
  while [ "${#pids[@]}" -ge "${parallel_jobs}" ]; do
    finished=false
    for pid_idx in "${!pids[@]}"; do
      pid="${pids[${pid_idx}]}"
      if ! kill -0 "${pid}" 2>/dev/null; then
        wait "${pid}"
        unset 'pids[pid_idx]'
        unset 'gpu_ids[pid_idx]'
        pids=("${pids[@]}")
        gpu_ids=("${gpu_ids[@]}")
        finished=true
        break
      fi
    done

    if [ "${finished}" = false ]; then
      sleep 5
    fi
  done

  gpu_id=0
  while printf '%s\n' "${gpu_ids[@]}" | grep -Fxq "${gpu_id}"; do
    gpu_id=$((gpu_id + 1))
  done

  train_and_infer_config "${gpu_id}" "${configs[${idx}]}" &
  pids+=("$!")
  gpu_ids+=("${gpu_id}")
done

for pid_idx in "${!pids[@]}"; do
  wait "${pids[${pid_idx}]}"
done

echo "[$(date --iso-8601=seconds)] CIFAR-10 perturbation training and inference complete"
