# Syncing run artifacts from the GPU cluster

Use `slurm_scripts/sync_runs_from_cluster.sh` to copy selected run directories
from remote `runs/` into the matching local hierarchy.

Cluster-specific values such as SSH aliases and remote repository paths belong
in `docs/hpc/hpc.local.md`; do not commit those values.

```bash
slurm_scripts/sync_runs_from_cluster.sh \
  --host <ssh-alias> \
  --remote-repo <cluster-path-to-distill-ood-detection> \
  students/baseline/cifar_10/resnet18/linear \
  students/baseline/cifar_10/resnet18/mlp
```

If the repository is under the remote home directory, quote `~` so your local
shell does not expand it before `rsync` runs:

```bash
slurm_scripts/sync_runs_from_cluster.sh \
  --host <ssh-alias> \
  --remote-repo '~/distill-ood-detection' \
  students/perturbation/embedding/clipping/cifar_10/resnet18/linear_layer3_clip_constant
```

Preview a transfer before downloading files:

```bash
slurm_scripts/sync_runs_from_cluster.sh \
  --host <ssh-alias> \
  --remote-repo <cluster-path-to-distill-ood-detection> \
  --dry-run \
  students/baseline/cifar_10/resnet18/linear
```

Add `--delete` only when the local selected run folder should exactly match the
remote selected run folder.
