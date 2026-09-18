#!/usr/bin/env bash

set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
SOURCE_DIR="${SOURCE_DIR:-${PROJECT_DIR}}"
GPU_ENV_DIR="${GPU_ENV_DIR:-${PROJECT_DIR}/envs/gpu}"
OPENOOD_ROOT="${OPENOOD_ROOT:-/home/bogush/openood}"

cd "${PROJECT_DIR}"
OPENOOD_ROOT="${OPENOOD_ROOT}" OPENOOD_DOWNLOAD_ONLY=1 \
  bash "${SOURCE_DIR}/slurm_scripts/prepare_openood_cifar_data.sbatch"

PROJECT_DIR="${PROJECT_DIR}" \
SOURCE_DIR="${SOURCE_DIR}" \
GPU_ENV_DIR="${GPU_ENV_DIR}" \
OPENOOD_ROOT="${OPENOOD_ROOT}" \
  "${SOURCE_DIR}/slurm_scripts/submit_openood_cifar_evaluation.sh"
