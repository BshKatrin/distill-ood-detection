#!/usr/bin/env bash

set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
SOURCE_DIR="${SOURCE_DIR:-${PROJECT_DIR}}"
GPU_ENV_DIR="${GPU_ENV_DIR:-${SOURCE_DIR}/envs/gpu}"
MANIFEST="${MANIFEST:-runs/channel_cluster_audit/nmf_latent_cosine/manifest.json}"
SUMMARY_DIR="${SUMMARY_DIR:-runs/channel_cluster_audit/nmf_latent_cosine/summary}"
MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
CONFIG_C10="configs/channel_cluster_audit/nmf_latent_cosine/cifar_10/resnet18.yaml"
CONFIG_C100="configs/channel_cluster_audit/nmf_latent_cosine/cifar_100/resnet18.yaml"
export PYTHONPATH="${SOURCE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_DIR}"
mkdir -p "$(dirname "${MANIFEST}")" slurm

if [ "${SKIP_UV_SYNC:-0}" != "1" ]; then
  uv sync --project "${GPU_ENV_DIR}" --locked
fi
uv run --project "${GPU_ENV_DIR}" --no-sync python -m distill_ood_detection.cli \
  build-channel-cluster-audit-manifest \
  --config "${CONFIG_C10}" \
  --config "${CONFIG_C100}" \
  --output "${MANIFEST}"

ARRAY_SPEC="$(uv run --project "${GPU_ENV_DIR}" --no-sync python -m distill_ood_detection.cli channel-cluster-audit-status --manifest "${MANIFEST}" --deep)"
if [ -z "${ARRAY_SPEC}" ]; then
  echo "All audit tasks are already complete."
  exit 0
fi

metadata_job="$(sbatch --parsable --export="ALL,SKIP_UV_SYNC=1" slurm_scripts/channel_cluster_audit_metadata.sbatch "${CONFIG_C10}" "${CONFIG_C100}")"
array_job="$(sbatch --parsable --dependency="afterok:${metadata_job}" --array="${ARRAY_SPEC}%${MAX_CONCURRENT}" --export="ALL,MANIFEST=${MANIFEST},SKIP_UV_SYNC=1" slurm_scripts/channel_cluster_audit_array.sbatch)"
summary_job="$(sbatch --parsable --dependency="afterok:${array_job}" --export="ALL,MANIFEST=${MANIFEST},SUMMARY_DIR=${SUMMARY_DIR}" slurm_scripts/summarize_channel_cluster_audit.sbatch)"
dashboard_job="$(sbatch --parsable --dependency="afterok:${summary_job}" --export="ALL,SUMMARY_DIR=${SUMMARY_DIR}" slurm_scripts/serve_channel_cluster_audit.sbatch)"

echo "Metadata job: ${metadata_job}"
echo "Audit array: ${array_job} (${ARRAY_SPEC}%${MAX_CONCURRENT})"
echo "Summary job: ${summary_job}"
echo "Panel job: ${dashboard_job}"
