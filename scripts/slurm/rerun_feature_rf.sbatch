#!/usr/bin/env bash
#SBATCH --job-name=feature_rf
#SBATCH --partition=electronic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
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

echo "[$(date --iso-8601=seconds)] Slurm GPU environment"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "SLURM_GPUS=${SLURM_GPUS:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi

run_distill_ood() {
  srun --ntasks=1 --gres=gpu:1 uv run --project envs/gpu --no-sync \
    python -m distill_ood_detection.cli "$@"
}

echo "[$(date --iso-8601=seconds)] Probing PyTorch CUDA inside the Slurm job step"
srun --ntasks=1 --gres=gpu:1 uv run --project envs/gpu --no-sync python - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"torch.version.cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"device_count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"device_name={torch.cuda.get_device_name(0)}")
    images = torch.zeros((1, 3, 8, 8), device="cuda")
    weights = torch.zeros((4, 3, 3, 3), device="cuda")
    torch.nn.functional.conv2d(images, weights)
    torch.cuda.synchronize()
    print("cuda_conv2d_probe=ok")
PY

config="configs/baseline/cifar_10/feature_random_forest_layer3.yaml"

echo "[$(date --iso-8601=seconds)] Training layer3 random forest: ${config}"
run_distill_ood train-tree-student \
  --config "${config}"

echo "[$(date --iso-8601=seconds)] Inferring layer3 random-forest probabilities: ${config}"
run_distill_ood infer-probabilities \
  --config "${config}" \
  --checkpoint best

echo "[$(date --iso-8601=seconds)] Layer3 random-forest chores complete"
