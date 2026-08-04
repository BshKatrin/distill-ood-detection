#!/usr/bin/env bash
#SBATCH --job-name=distill_ood
#SBATCH --partition=electronic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=09:00:00
#SBATCH --output=slurm/%x-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1
export UV_NO_PROGRESS=1
export PATH="${HOME}/.local/bin:${PATH}"
export HF_HUB_OFFLINE=1
PS4='[$(date --iso-8601=seconds)] + '
set -x

trap 'status=$?; echo "[$(date --iso-8601=seconds)] Failed at line ${LINENO} with exit status ${status}"; exit "${status}"' ERR

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_DIR}"

mkdir -p slurm

MODE="${MODE:-train-infer}"
PARALLEL="${PARALLEL:-auto}"
CHECKPOINT="${CHECKPOINT:-best}"
METHOD="${METHOD:-}"
TREE_MODE="${TREE_MODE:-}"
INCLUDE_TRAIN="${INCLUDE_TRAIN:-0}"
INCLUDE_VALIDATION="${INCLUDE_VALIDATION:-0}"
APPLY_PERTURBATION="${APPLY_PERTURBATION:-0}"
SKIP_UV_SYNC="${SKIP_UV_SYNC:-0}"

configs=("$@")

if [ -n "${CONFIG_LIST:-}" ]; then
  if [ ! -f "${CONFIG_LIST}" ]; then
    echo "[$(date --iso-8601=seconds)] CONFIG_LIST does not exist: ${CONFIG_LIST}"
    exit 1
  fi

  while IFS= read -r config || [ -n "${config}" ]; do
    case "${config}" in
      ""|\#*) continue ;;
      *) configs+=("${config}") ;;
    esac
  done < "${CONFIG_LIST}"
fi

if [ "${#configs[@]}" -eq 0 ]; then
  echo "[$(date --iso-8601=seconds)] No configs provided"
  echo "Usage: sbatch slurm_scripts/run_configs.sbatch configs/...yaml [configs/...yaml ...]"
  echo "Or set CONFIG_LIST to a newline-delimited file of config paths."
  exit 1
fi

case "${MODE}" in
  train|infer|train-infer) ;;
  *)
    echo "[$(date --iso-8601=seconds)] MODE must be train, infer, or train-infer; got ${MODE}"
    exit 1
    ;;
esac

uv --version
if [ "${SKIP_UV_SYNC}" = "1" ]; then
  echo "[$(date --iso-8601=seconds)] Reusing existing uv GPU environment"
else
  echo "[$(date --iso-8601=seconds)] Syncing uv GPU environment"
  uv sync --project envs/gpu --locked
fi

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

is_tree_config() {
  local config="$1"
  [[ "${config}" == *random_forest* ]]
}

is_feature_denoising_config() {
  local config="$1"
  [[ "${config}" == configs/students/feature_denoising/* ]]
}

is_subspace_ensemble_config() {
  local config="$1"
  [[ "${config}" == configs/students/subspace_ensemble/* ]]
}

train_config() {
  local config="$1"

  if is_tree_config "${config}"; then
    echo "[$(date --iso-8601=seconds)] Training tree student: ${config}"
    if [ -n "${TREE_MODE}" ]; then
      run_distill_ood train-tree-student --config "${config}" --mode "${TREE_MODE}"
    else
      run_distill_ood train-tree-student --config "${config}"
    fi
  else
    echo "[$(date --iso-8601=seconds)] Training PyTorch student: ${config}"
    if [ -n "${METHOD}" ]; then
      run_distill_ood train-student --config "${config}" --method "${METHOD}"
    else
      run_distill_ood train-student --config "${config}"
    fi
  fi
}

infer_config() {
  local config="$1"
  if is_subspace_ensemble_config "${config}"; then
    echo "[$(date --iso-8601=seconds)] Exporting subspace-ensemble predictions: ${config}"
    run_distill_ood \
      export-subspace-ensemble-inference \
      --config "${config}" \
      --checkpoint "${CHECKPOINT}"
    echo "[$(date --iso-8601=seconds)] Exporting subspace-ensemble OOD Scores: ${config}"
    run_distill_ood \
      export-subspace-ensemble-scores \
      --config "${config}" \
      --checkpoint "${CHECKPOINT}"
    return
  fi

  if is_feature_denoising_config "${config}"; then
    local args=(export-feature-denoising-scores --config "${config}" --checkpoint "${CHECKPOINT}")

    if [ "${INCLUDE_TRAIN}" = "1" ]; then
      args+=(--include-train)
    fi

    if [ "${INCLUDE_VALIDATION}" = "1" ]; then
      args+=(--include-validation)
    fi

    echo "[$(date --iso-8601=seconds)] Exporting Feature Denoising reconstruction scores: ${config}"
    run_distill_ood "${args[@]}"
    return
  fi

  local args=(infer-probabilities --config "${config}" --checkpoint "${CHECKPOINT}")

  if is_tree_config "${config}" && [ -n "${TREE_MODE}" ]; then
    args+=(--tree-mode "${TREE_MODE}")
  fi

  if ! is_tree_config "${config}" && [ -n "${METHOD}" ]; then
    args+=(--method "${METHOD}")
  fi

  if [ "${INCLUDE_TRAIN}" = "1" ]; then
    args+=(--include-train)
  fi

  if [ "${INCLUDE_VALIDATION}" = "1" ]; then
    args+=(--include-validation)
  fi

  if [ "${APPLY_PERTURBATION}" = "1" ]; then
    args+=(--apply-perturbation)
  fi

  echo "[$(date --iso-8601=seconds)] Inferring probabilities: ${config}"
  run_distill_ood "${args[@]}"
}

run_config_pipeline() {
  local gpu_id="$1"
  local config="$2"

  export CUDA_VISIBLE_DEVICES="${gpu_id}"

  if [ ! -f "${config}" ]; then
    echo "[$(date --iso-8601=seconds)] Config does not exist: ${config}"
    exit 1
  fi

  echo "[$(date --iso-8601=seconds)] [gpu=${gpu_id}] Starting ${MODE}: ${config}"

  case "${MODE}" in
    train) train_config "${config}" ;;
    infer) infer_config "${config}" ;;
    train-infer)
      train_config "${config}"
      infer_config "${config}"
      ;;
  esac

  echo "[$(date --iso-8601=seconds)] [gpu=${gpu_id}] Finished ${MODE}: ${config}"
}

gpu_count="$(get_allocated_gpu_count)"

if ! [[ "${gpu_count}" =~ ^[0-9]+$ ]] || [ "${gpu_count}" -lt 1 ]; then
  echo "[$(date --iso-8601=seconds)] Could not determine a positive allocated GPU count from Slurm"
  exit 1
fi

if [ "${PARALLEL}" = "auto" ]; then
  parallel_jobs="${gpu_count}"
elif [[ "${PARALLEL}" =~ ^[0-9]+$ ]] && [ "${PARALLEL}" -ge 1 ]; then
  parallel_jobs="${PARALLEL}"
else
  echo "[$(date --iso-8601=seconds)] PARALLEL must be auto or a positive integer; got ${PARALLEL}"
  exit 1
fi

if [ "${parallel_jobs}" -gt "${gpu_count}" ]; then
  parallel_jobs="${gpu_count}"
fi

if [ "${parallel_jobs}" -gt "${#configs[@]}" ]; then
  parallel_jobs="${#configs[@]}"
fi

echo "[$(date --iso-8601=seconds)] Running ${#configs[@]} config(s), mode=${MODE}, parallel_jobs=${parallel_jobs}, allocated_gpus=${gpu_count}"

for ((start = 0; start < ${#configs[@]}; start += parallel_jobs)); do
  pids=()

  for ((slot = 0; slot < parallel_jobs && start + slot < ${#configs[@]}; slot++)); do
    config="${configs[start + slot]}"
    run_config_pipeline "${slot}" "${config}" &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
done

echo "[$(date --iso-8601=seconds)] All config pipelines complete"
