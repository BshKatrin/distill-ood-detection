# Project documentation

This folder contains the project knowledge that agents and humans should consult before making changes.

Sections marked "Not implemented" describe planned work. All other sections correspond to implemented functionality.

## Start here

- [Strategies](strategies/README.md): OOD distillation training strategies and their rationale.
- [Configs](configs.md): Experiment configuration layout and naming conventions.
- [Environments](envs.md): Focused `uv` environments for GPU jobs, notebooks, and tests.
- [HPC](hpc/README.md): High-performance computing cluster access. Read this only when HPC or GPU cluster access is requested.
- [Syncing run artifacts from the GPU cluster](hpc/sync-runs.md): Copy selected remote `runs/` folders into local `runs/`.
- [Objectives](objectives/README.md): Distillation objective functions for deep-learning students.
- [OOD Scores](ood_scores/README.md): Per-sample scores that quantify whether a sample is more ID-like or OOD-like.
