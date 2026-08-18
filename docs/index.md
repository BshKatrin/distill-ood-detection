# Project documentation

This folder contains the project knowledge that agents and humans should consult before making changes.

Sections marked "Not implemented" describe planned work. All other sections correspond to implemented functionality.

## Start here

- [Strategies](strategies/README.md): OOD distillation training strategies and their rationale.
- [Activation-Subspace Students](strategies/activation_subspace.md):
  self-contained ActSub paper background, project-specific student variant,
  training/inference/scoring workflow, and artifact contracts.
- [Disjoint Subspace Ensembles](strategies/subspace_ensemble.md): channel and
  whitened GAP-PCA ensembles with predictive-entropy and BALD OOD Scores.
- [Feature Denoising channel grouping](strategies/feature_denoising/channel_grouping/README.md):
  methods for building semantic channel candidates before grouped masking.
- [Feature Denoising NMF concept masking](strategies/feature_denoising/nmf_concept_masking.md):
  global non-negative concept directions, complete-direction masking, and
  residual reconstruction scoring.
- [NMF channel-cluster audit](strategies/feature_denoising/channel_grouping/channel_cluster_audit.md):
  269 fixed-cluster specialists, audit metrics, restartable arrays, and the
  minimal remote Panel score application.
- [Configs](configs.md): Experiment configuration layout and naming conventions.
- [Environments](envs.md): Focused `uv` environments for GPU jobs, notebooks, and tests.
- [HPC](hpc/README.md): High-performance computing cluster access. Read this only when HPC or GPU cluster access is requested.
- [SLURM jobs](hpc/slurm-jobs.md): Submit reusable config-driven sbatch jobs.
- [Panel dashboards on SLURM](hpc/panel-dashboards.md): build summaries, serve
  both cluster dashboards, and connect through SSH tunnels.
- [Syncing run artifacts from the GPU cluster](hpc/sync-runs.md): Copy selected remote `runs/` folders into local `runs/`.
- [Objectives](objectives/README.md): Distillation objective functions for deep-learning students.
- [OOD Scores](ood_scores/README.md): Per-sample scores that quantify whether a sample is more ID-like or OOD-like.
- [Reports](reports/README.md): Export metric reports, render report outputs,
  and generate analysis notebooks.
