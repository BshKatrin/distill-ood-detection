#!/usr/bin/env bash
#SBATCH --job-name=nmf_audit_launch
#SBATCH --partition=electronic
#SBATCH --exclude=daft
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/%x-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1 UV_NO_PROGRESS=1
export PATH="${HOME}/.local/bin:${PATH}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
SOURCE_DIR="${SOURCE_DIR:-${PROJECT_DIR}}"
GPU_ENV_DIR="${GPU_ENV_DIR:-${SOURCE_DIR}/envs/gpu}"
MANIFEST="${MANIFEST:-runs/channel_cluster_audit/nmf_latent_cosine/manifest.json}"
MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
MAX_TASKS="${MAX_TASKS:-1000}"
export PYTHONPATH="${SOURCE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_DIR}"
mkdir -p "$(dirname "${MANIFEST}")" slurm

configs=("$@")
if [ "${#configs[@]}" -eq 0 ]; then
  echo "Usage: sbatch launch_channel_cluster_audit_array.sbatch <audit-config> [...]"
  exit 1
fi

if [ "${SKIP_UV_SYNC:-0}" != "1" ]; then
  uv sync --project "${GPU_ENV_DIR}" --locked
fi
manifest_args=()
for config in "${configs[@]}"; do
  manifest_args+=(--config "${config}")
done
uv run --project "${GPU_ENV_DIR}" --no-sync python -m distill_ood_detection.cli \
  build-channel-cluster-audit-manifest \
  "${manifest_args[@]}" \
  --output "${MANIFEST}"

task_count="$(jq -r '.task_count' "${MANIFEST}")"
if [ "${task_count}" -gt "${MAX_TASKS}" ]; then
  echo "Refusing to submit ${task_count} tasks; MAX_TASKS=${MAX_TASKS}"
  exit 1
fi
jq -r '
  .tasks
  | group_by([.dataset, .layer])[]
  | "\(.[0].dataset) \(.[0].layer): \(length) specialists"
' "${MANIFEST}"

array_spec="$(
  uv run --project "${GPU_ENV_DIR}" --no-sync python -m distill_ood_detection.cli \
    channel-cluster-audit-status \
    --manifest "${MANIFEST}" \
    --deep
)"
if [ -z "${array_spec}" ]; then
  echo "All manifest tasks are already complete."
  exit 0
fi

array_job="$(
  sbatch --parsable \
    --exclude=daft \
    --array="${array_spec}%${MAX_CONCURRENT}" \
    --export="ALL,MANIFEST=${MANIFEST},SKIP_UV_SYNC=1" \
    slurm_scripts/channel_cluster_audit_array.sbatch
)"
echo "Manifest: ${MANIFEST}"
echo "Audit array: ${array_job} (${array_spec}%${MAX_CONCURRENT})"
