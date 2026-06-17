# High Performance Computing (HPC) Cluster documentation

This directory contains HPC cluster-related documentation.

- [SLURM jobs](slurm-jobs.md): Submit reusable config-driven sbatch jobs.
- [Syncing run artifacts](sync-runs.md): Copy selected remote `runs/` folders
  into local `runs/`.

## Local configuration

Cluster access instructions are intentionally not stored in version control.

If cluster work is requested, first check whether the following file exists:

docs/hpc/hpc.local.md

That file may contain:

- private SSH aliases
- authentication notes
- GPU partition names
- module-loading commands
- scratch paths
- job-submission examples

If present, treat it as the authoritative source for cluster-specific instructions.

## Rules

- Do not print secrets, passphrases, tokens, private hostnames, or private usernames in responses.
- Do not commit `docs/hpc/hpc.local.md`.
- If `docs/hpc/hpc.local.md` is missing, ask the user for the non-sensitive cluster details needed for the task, or ask them to create it from `docs/hpc/hpc.template.md`.
- Prefer documented cluster commands from `docs/hpc/hpc.local.md` over guessing.

## Environment

Use the focused GPU environment for cluster jobs:

```bash
uv sync --project envs/gpu --locked
```

Submitted jobs should run with `PYTHONPATH` pointing at `src/` and use
`uv run --project envs/gpu --no-sync python -m distill_ood_detection.cli ...`
after the GPU environment has been synced.

If the cluster provides a shared dataset folder, set it with
`DISTILL_OOD_DATA_DIR` in the job environment or repository `.env` file. This
overrides `dataset.data_dir` from experiment YAML files and avoids downloading
datasets into the repository-local `data/` directory.

Example from the repository root:

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" uv run --project envs/gpu --no-sync python -m distill_ood_detection.cli train-student --config configs/perturbation/cifar_10/linear_layer3_clip_constant.yaml
```
