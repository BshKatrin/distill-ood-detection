#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Sync selected run folders from a remote GPU cluster into local runs/.

Usage:
  slurm_scripts/sync_runs_from_cluster.sh --host SSH_ALIAS --remote-repo REMOTE_REPO_PATH RUN_DIR [RUN_DIR ...]

Example:
  slurm_scripts/sync_runs_from_cluster.sh \
    --host my-cluster \
    --remote-repo /scratch/user/distill-ood-detection \
    linear_student_resnet18_cifar10 mlp_student_resnet18_cifar10

  slurm_scripts/sync_runs_from_cluster.sh \
    --host my-cluster \
    --remote-repo '~/distill-ood-detection' \
    perturbation_linear_layer3_student_resnet18_cifar10

Options:
  --host SSH_ALIAS           SSH alias or user@host for the GPU cluster.
  --remote-repo PATH         Repository path on the remote cluster.
  --local-runs PATH          Local runs directory. Defaults to ./runs.
  --dry-run                  Show what would be transferred.
  --delete                   Delete local files in selected run folders that no longer exist remotely.
  -h, --help                 Show this help text.

Only top-level run directories named as positional arguments are transferred.
EOF
}

host=""
remote_repo=""
local_runs="runs"
dry_run=()
delete=()
run_dirs=()

while (($#)); do
  case "$1" in
    --host)
      if (($# < 2)); then
        echo "error: --host requires a value" >&2
        exit 2
      fi
      host="$2"
      shift 2
      ;;
    --remote-repo)
      if (($# < 2)); then
        echo "error: --remote-repo requires a value" >&2
        exit 2
      fi
      remote_repo="$2"
      shift 2
      ;;
    --local-runs)
      if (($# < 2)); then
        echo "error: --local-runs requires a value" >&2
        exit 2
      fi
      local_runs="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=(--dry-run)
      shift
      ;;
    --delete)
      delete=(--delete)
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while (($#)); do
        run_dirs+=("$1")
        shift
      done
      ;;
    -*)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      run_dirs+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$host" ]]; then
  echo "error: --host is required" >&2
  usage >&2
  exit 2
fi

if [[ -z "$remote_repo" ]]; then
  echo "error: --remote-repo is required" >&2
  usage >&2
  exit 2
fi

local_home="${HOME%/}"
if [[ "$remote_repo" == "$local_home" || "$remote_repo" == "$local_home/"* ]]; then
  cat >&2 <<EOF
error: --remote-repo expanded to your local home directory:
  $remote_repo

If the repository is under the remote home directory, quote the tilde:
  --remote-repo '~/distill-ood-detection'

Or pass the absolute path as it exists on the cluster.
EOF
  exit 2
fi

if ((${#run_dirs[@]} == 0)); then
  echo "error: provide at least one run directory to sync" >&2
  usage >&2
  exit 2
fi

mkdir -p "$local_runs"

rsync_args=(
  -avh
  --progress
  --partial
  "${dry_run[@]}"
  "${delete[@]}"
)

remote_runs="${remote_repo%/}/runs"

for run_dir in "${run_dirs[@]}"; do
  if [[ "$run_dir" == /* || "$run_dir" == *".."* || "$run_dir" == *"/"* ]]; then
    echo "error: run directory must be a top-level folder name under runs/: $run_dir" >&2
    exit 2
  fi

  echo "Syncing ${host}:${remote_runs}/${run_dir}/ -> ${local_runs%/}/${run_dir}/"
  rsync "${rsync_args[@]}" \
    "${host}:${remote_runs}/${run_dir}/" \
    "${local_runs%/}/${run_dir}/"
done
