# Syncing run artifacts from the GPU cluster

Use `slurm_scripts/sync_runs_from_cluster.sh` to copy selected top-level folders from
remote `runs/` into local `runs/`.

Cluster-specific values such as SSH aliases and remote repository paths belong
in `docs/hpc/hpc.local.md`; do not commit those values.

```bash
slurm_scripts/sync_runs_from_cluster.sh \
  --host <ssh-alias> \
  --remote-repo <cluster-path-to-distill-ood-detection> \
  linear_student_resnet18_cifar10 mlp_student_resnet18_cifar10
```

If the repository is under the remote home directory, quote `~` so your local
shell does not expand it before `rsync` runs:

```bash
slurm_scripts/sync_runs_from_cluster.sh \
  --host <ssh-alias> \
  --remote-repo '~/distill-ood-detection' \
  perturbation_linear_layer3_student_resnet18_cifar10
```

Preview a transfer before downloading files:

```bash
slurm_scripts/sync_runs_from_cluster.sh \
  --host <ssh-alias> \
  --remote-repo <cluster-path-to-distill-ood-detection> \
  --dry-run \
  linear_student_resnet18_cifar10
```

Add `--delete` only when the local selected run folder should exactly match the
remote selected run folder.
