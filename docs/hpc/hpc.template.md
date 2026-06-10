# HPC system access Template

This file is safe to commit. Copy it to `docs/hpc/hpc.local.md` and fill
in the private/local values there.

Do not put passphrases, tokens, passwords, private usernames, or private host
details in committed files.

## Repository

Cluster repository path:

```bash
<cluster-path-to-distill-ood-detection>
```

## SSH

SSH alias:

```bash
ssh <ssh-alias>
```

Before connecting, make sure `ssh-agent` is running and the relevant SSH key is
loaded so the passphrase does not need to be entered repeatedly.

```bash
ssh-add -l
ssh-add ~/.ssh/<key-name>
```

## GPU Access

Preferred partitions:

- `<partition-name>`: `<notes>`

Preferred nodes:

- `<node-name>`: `<desired status or notes>`

## SLURM

Interactive GPU session:

```bash
<srun command>
```

Batch job:

```bash
<sbatch command>
```

## Environment

Use `uv` for Python dependency management.

```bash
uv sync
```

## Safety

- Keep private cluster details in `docs/hpc/hpc.local.md`.
- Do not commit `docs/hpc/hpc.local.md`.
- Do not print secrets or passphrases in responses or logs.
