#!/usr/bin/env bash
#SBATCH --job-name=pixmix_data
#SBATCH --partition=electronic
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/%x-%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
cd "${PROJECT_DIR}"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

data_dir="${DISTILL_OOD_DATA_DIR:-data}"
archive_root="${data_dir}/pixmix"
target="${archive_root}/fractals_and_fvis"

if [ ! -d "${target}/fractals" ] || [ ! -d "${target}/first_layers_resized256_onevis" ]; then
  mkdir -p "${archive_root}"
  if [ -n "${PIXMIX_ARCHIVE:-}" ]; then
    echo "3619fb7e2c76130749d97913fdd3ab27  ${PIXMIX_ARCHIVE}" | md5sum --check -
    tar -xf "${PIXMIX_ARCHIVE}" -C "${archive_root}"
  else
    curl \
      --location \
      --fail \
      --retry 5 \
      --retry-delay 10 \
      "https://drive.usercontent.google.com/download?id=1qC2gIUx9ARU7zhgI4IwGD3YcFhm8J4cA&export=download&confirm=t" \
      | tar -xf - -C "${archive_root}"
  fi
fi

test -d "${target}/fractals"
test -d "${target}/first_layers_resized256_onevis"
