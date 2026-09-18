#!/usr/bin/env bash

set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
SOURCE_DIR="${SOURCE_DIR:-${PROJECT_DIR}}"
GPU_ENV_DIR="${GPU_ENV_DIR:-${PROJECT_DIR}/envs/gpu}"
OPENOOD_ROOT="${OPENOOD_ROOT:-/home/bogush/openood}"
SELECTION="${SELECTION:-${SOURCE_DIR}/configs/evaluation/openood_cifar_v1_5/selected_variants.yaml}"
MANIFEST="${MANIFEST:-${PROJECT_DIR}/runs/evaluations/openood_v1_5/selected_variants_manifest.json}"
MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
export PYTHONPATH="${SOURCE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${PROJECT_DIR}"
mkdir -p "$(dirname "${MANIFEST}")" slurm

uv run --project "${GPU_ENV_DIR}" --no-sync \
  python -m distill_ood_detection.cli build-openood-cifar-manifest \
  --selection "${SELECTION}" \
  --output "${MANIFEST}" \
  --config-root "${SOURCE_DIR}"

read -r task_count missing_count < <(
  uv run --project "${GPU_ENV_DIR}" --no-sync python -c \
    'import json,sys; x=json.load(open(sys.argv[1])); print(len(x["tasks"]), len(x["missing"]))' \
    "${MANIFEST}"
)
echo "Ready OpenOOD evaluations: ${task_count}; missing checkpoints: ${missing_count}"
if [ "${task_count}" -eq 0 ]; then
  echo "No evaluation jobs can be submitted. See ${MANIFEST}."
  exit 1
fi

data_job="$(sbatch --parsable \
  --export="ALL,OPENOOD_ROOT=${OPENOOD_ROOT}" \
  "${SOURCE_DIR}/slurm_scripts/prepare_openood_cifar_data.sbatch")"
array_job="$(sbatch --parsable \
  --dependency="afterok:${data_job}" \
  --array="0-$((task_count - 1))%${MAX_CONCURRENT}" \
  --export="ALL,PROJECT_DIR=${PROJECT_DIR},SOURCE_DIR=${SOURCE_DIR},GPU_ENV_DIR=${GPU_ENV_DIR},MANIFEST=${MANIFEST},OPENOOD_ROOT=${OPENOOD_ROOT}" \
  "${SOURCE_DIR}/slurm_scripts/openood_cifar_evaluation_array.sbatch")"

echo "Data preparation job: ${data_job}"
echo "Evaluation array: ${array_job} (0-$((task_count - 1))%${MAX_CONCURRENT})"
echo "Task manifest: ${MANIFEST}"
