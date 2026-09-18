#!/usr/bin/env bash
#SBATCH --job-name=openood_cifar_data
#SBATCH --partition=hard
#SBATCH --exclude=daft
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/%x-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1
export PATH="${HOME}/.local/bin:${PATH}"

OPENOOD_ROOT="${OPENOOD_ROOT:-/home/bogush/openood}"
OPENOOD_DOWNLOAD_ONLY="${OPENOOD_DOWNLOAD_ONLY:-0}"
DATA_ROOT="${OPENOOD_ROOT}/data"
IMAGE_ROOT="${DATA_ROOT}/images_classic"
mkdir -p "${IMAGE_ROOT}" slurm

download_google_drive() {
  local file_id="$1"
  local destination="$2"
  local partial="${destination}.part"
  curl --fail --location --retry 5 --retry-delay 10 --continue-at - \
    --output "${partial}" \
    "https://drive.usercontent.google.com/download?id=${file_id}&export=download&confirm=t"
  mv "${partial}" "${destination}"
}

extract_zip() {
  local archive="$1"
  local destination="$2"
  python3 -m zipfile -t "${archive}"
  python3 -m zipfile -e "${archive}" "${destination}"
}

prepare_archive() {
  local dataset="$1"
  local file_id="$2"
  local destination="${IMAGE_ROOT}/${dataset}"
  local archive="${destination}/${dataset}.zip"
  local marker="${destination}/.openood_extracted"
  mkdir -p "${destination}"
  if [ -f "${marker}" ]; then
    echo "OpenOOD ${dataset}: already extracted"
    return
  fi
  if [ ! -f "${archive}" ]; then
    echo "OpenOOD ${dataset}: downloading"
    download_google_drive "${file_id}" "${archive}"
  fi
  if [ "${OPENOOD_DOWNLOAD_ONLY}" = "1" ]; then
    return
  fi
  extract_zip "${archive}" "${destination}"
  touch "${marker}"
}

if [ ! -d "${DATA_ROOT}/benchmark_imglist" ]; then
  benchmark_archive="${DATA_ROOT}/benchmark_imglist.zip"
  mkdir -p "${DATA_ROOT}"
  download_google_drive "1lI1j0_fDDvjIt9JlWAw09X8ks-yrR_H1" "${benchmark_archive}"
  if [ "${OPENOOD_DOWNLOAD_ONLY}" != "1" ]; then
    extract_zip "${benchmark_archive}" "${DATA_ROOT}"
  fi
fi

prepare_archive cifar10 1Co32RiiWe16lTaiOU6JMMnyUYS41IlO1
prepare_archive cifar100 1PGKheHUsf29leJPPGuXqzLBMwl8qMF8_
prepare_archive tin 1PZ-ixyx52U989IKsMA2OT-24fToTrelC
prepare_archive mnist 1CCHAGWqA1KJTFFswuF9cbhmB-j98Y1Sb
prepare_archive svhn 1DQfc11HOtB1nEwqS4pWUFp8vtQ3DczvI
prepare_archive texture 1OSz1m3hHfVWbRdmMwKbUzoU8Hg9UKcam
prepare_archive places365 1Ec-LRSTf6u5vEctKX9vRp9OA6tqnJ0Ay

if [ "${OPENOOD_DOWNLOAD_ONLY}" = "1" ]; then
  echo "OpenOOD CIFAR archives downloaded: ${DATA_ROOT}"
else
  echo "OpenOOD CIFAR data preparation complete: ${DATA_ROOT}"
fi
