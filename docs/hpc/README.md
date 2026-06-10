# High Performance Computing (HPC) Cluster documentation

This directory contains HPC cluster-related documentation.

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
