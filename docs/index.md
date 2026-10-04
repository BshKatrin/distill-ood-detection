# Project documentation

Use this index before changing an experiment. Method references describe
implemented behaviour; sections marked **Not implemented** describe planned
work. Curated results live in `experiments/`, while report deliverables live
in `reports/`.

## Setup and experiment identity

- [Root README](../README.md): repository layout and a baseline reproduction.
- [Vocabulary](vocabulary.md): project terminology, including ID, OOD, and OOD Score.
- [Environments](envs.md): focused `uv` projects and dependencies.
- [Configs](configs.md): config layout, run directories, and artifact ownership.

## Methods

- [Strategies](strategies/README.md): baseline, perturbation, Feature Denoising,
  activation-subspace students, and subspace ensembles.
- [Objectives](objectives/README.md): output-distillation loss definitions.
- [Feature Denoising](strategies/feature_denoising/README.md): method catalogue,
  target representations, masking, and subspace diagnostics.
- [Channel grouping](strategies/feature_denoising/channel_grouping/README.md):
  correlation and NMF hierarchy construction.
- [Activation-subspace reference](strategies/activation_subspace.md) and
  [workflow](workflows/activation_subspace.md): decomposition, students, and artifacts.
- [Disjoint subspace ensembles](strategies/subspace_ensemble.md): channel and
  whitened GAP-PCA students, predictive entropy, and BALD.

## Evaluation

- [Evaluation overview](evaluation/README.md): scores, metrics, and protocols.
- [OOD Scores](evaluation/ood-scores.md): per-sample definitions and signs.
- [Aggregate metrics](evaluation/metrics.md): ID+/OOD+ FPR@95, units, and averaging.
- [OpenOOD CIFAR](evaluation/openood/cifar.md): fixed manifests and selected students.
- [OpenOOD ImageNet](evaluation/openood/imagenet.md): ImageNet-200/ImageNet-1K evaluation.

## Reporting

- [Deliverable catalogue](../reports/README.md): final report, slides, and recap archive.
- [Reporting workflows](reports/README.md): exports, notebooks, builds, and Git policy.
- [Recaps and final-report builds](reports/recap.md): ISO dates and compiler scratch files.

## Operations

- [Archive and restore](archive.md): final dependency locks, local-only artifacts,
  and fresh-clone verification before removing a checkout.
- [HPC](hpc/README.md): cluster access and local configuration. Read when HPC
  or GPU cluster access is requested.
- [SLURM jobs](hpc/slurm-jobs.md): reusable config-driven jobs.
- [Dashboards](hpc/panel-dashboards.md): summary generation, serving, and SSH tunnels.
- [Sync run artifacts](hpc/sync-runs.md): selected remote run transfers.
- [NMF channel-cluster audit](strategies/feature_denoising/channel_grouping/channel_cluster_audit.md):
  specialist tasks, restartable arrays, metrics, and dashboard contracts.

## Experiment results

- [Experiment catalogue](../experiments/README.md): report sections mapped to
  experiment records and configurations.
- [Final report](../reports/final-report/main.pdf): internship findings and convention audit.
- [Presentation](../reports/presentations/internship/presentation.pdf): final slides.
