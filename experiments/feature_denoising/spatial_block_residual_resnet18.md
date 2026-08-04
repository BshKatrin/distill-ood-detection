# Spatial Block Residual Feature Denoising: ResNet-18

Status: completed for CIFAR-10 and CIFAR-100 ID, layers 1-4.

## Setup

- Teacher: ResNet-18.
- Corruption: one contiguous spatial feature block, shared across channels.
- Fill value: per-channel mean fitted from the deterministic ID
  student-training subset.
- Student input: corrupted feature map plus a binary hidden-position channel.
- Student output: a residual correction added to the corrupted feature map.
- Objective: reconstruction MSE on hidden positions only.
- Evaluation draws: `10`.
- Checkpoint: best validation reconstruction loss.

Block-size distributions:

| Layer | Feature map | Uniform square side lengths |
| --- | --- | --- |
| layer1 | `64x32x32` | `3`, `5` |
| layer2 | `128x16x16` | `3`, `5` |
| layer3 | `256x8x8` | `1`, `3` |
| layer4 | `512x4x4` | `1` |

## Runs

The eight configs are listed in:

```text
slurm_scripts/feature_denoising_spatial_block_residual_resnet18.txt
```

The completed cluster jobs were:

| Job | ID dataset | Configs | State |
| --- | --- | ---: | --- |
| `407330` | CIFAR-10 | 4 | completed (`0:0`) |
| `407331` | CIFAR-100 | 4 | completed (`0:0`) |

Rerun training and score export with:

```bash
sbatch --export=ALL,CONFIG_LIST=slurm_scripts/feature_denoising_spatial_block_residual_resnet18.txt \
  slurm_scripts/run_configs.sbatch
```

Run directories mirror the config paths below
`runs/students/feature_denoising/feature_masking/`.

## Scores

The primary OOD Score is negated masked-region reconstruction MSE. Score
artifacts also contain masked-region `identity_error` for the uncorrected
mean-filled feature map and `improvement`, defined as identity error minus
reconstruction error.

All OOD tables report ROC-AUC / FPR@95. Higher OOD Scores indicate a sample is
more ID-like.

## Reconstruction Losses

Loss is the masked-region reconstruction MSE. Its scale differs by layer, so
the values should not be compared across layers as a direct representation
quality ranking.

| ID dataset | Layer | Best validation | Final validation | Test |
| --- | --- | ---: | ---: | ---: |
| CIFAR-10 | layer1 | 0.009639 | 0.009639 | 0.009630 |
| CIFAR-10 | layer2 | 0.011649 | 0.011723 | 0.011816 |
| CIFAR-10 | layer3 | 0.003375 | 0.003398 | 0.003409 |
| CIFAR-10 | layer4 | 0.002578 | 0.002639 | 0.002682 |
| CIFAR-100 | layer1 | 0.003305 | 0.003318 | 0.003349 |
| CIFAR-100 | layer2 | 0.005276 | 0.005285 | 0.005341 |
| CIFAR-100 | layer3 | 0.003042 | 0.003071 | 0.003039 |
| CIFAR-100 | layer4 | 0.008699 | 0.008699 | 0.007986 |

## Primary Reconstruction Score

Score: `-raw_reconstruction_error`.

### CIFAR-10 ID

| Layer | MNIST ROC-AUC / FPR@95 | SVHN ROC-AUC / FPR@95 | CIFAR-100 ROC-AUC / FPR@95 | Macro ROC-AUC / FPR@95 |
| --- | ---: | ---: | ---: | ---: |
| layer1 | 0.439 / 0.998 | 0.059 / 1.000 | 0.532 / 0.910 | 0.343 / 0.969 |
| layer2 | 0.612 / 0.945 | 0.083 / 0.999 | 0.484 / 0.933 | 0.393 / 0.959 |
| layer3 | 0.493 / 0.974 | 0.067 / 0.999 | 0.336 / 0.986 | 0.299 / 0.986 |
| layer4 | 0.571 / 0.952 | 0.440 / 0.979 | 0.711 / 0.841 | 0.574 / 0.924 |

### CIFAR-100 ID

| Layer | MNIST ROC-AUC / FPR@95 | SVHN ROC-AUC / FPR@95 | CIFAR-10 ROC-AUC / FPR@95 | Macro ROC-AUC / FPR@95 |
| --- | ---: | ---: | ---: | ---: |
| layer1 | 0.412 / 0.999 | 0.043 / 1.000 | 0.478 / 0.971 | 0.311 / 0.990 |
| layer2 | 0.500 / 0.992 | 0.117 / 0.999 | 0.489 / 0.975 | 0.368 / 0.989 |
| layer3 | 0.121 / 1.000 | 0.110 / 0.999 | 0.497 / 0.967 | 0.242 / 0.988 |
| layer4 | 0.594 / 0.922 | 0.370 / 0.981 | 0.481 / 0.967 | 0.482 / 0.957 |

The primary score is weak. Layer4 is best for both ID datasets, but its macro
FPR@95 remains at least `0.924`.

## Improvement Score

Score: `identity_error - raw_reconstruction_error`.

### CIFAR-10 ID

| Layer | MNIST ROC-AUC / FPR@95 | SVHN ROC-AUC / FPR@95 | CIFAR-100 ROC-AUC / FPR@95 | Macro ROC-AUC / FPR@95 |
| --- | ---: | ---: | ---: | ---: |
| layer1 | 0.451 / 0.985 | 0.918 / 0.441 | 0.491 / 0.937 | 0.620 / 0.787 |
| layer2 | 0.215 / 0.992 | 0.923 / 0.283 | 0.570 / 0.889 | 0.569 / 0.722 |
| layer3 | 0.863 / 0.770 | 0.959 / 0.237 | 0.833 / 0.675 | 0.885 / 0.561 |
| layer4 | 0.933 / 0.309 | 0.874 / 0.486 | 0.787 / 0.700 | 0.865 / 0.498 |

### CIFAR-100 ID

| Layer | MNIST ROC-AUC / FPR@95 | SVHN ROC-AUC / FPR@95 | CIFAR-10 ROC-AUC / FPR@95 | Macro ROC-AUC / FPR@95 |
| --- | ---: | ---: | ---: | ---: |
| layer1 | 0.689 / 0.888 | 0.966 / 0.161 | 0.510 / 0.958 | 0.722 / 0.669 |
| layer2 | 0.312 / 0.990 | 0.850 / 0.568 | 0.524 / 0.970 | 0.562 / 0.842 |
| layer3 | 0.916 / 0.332 | 0.881 / 0.436 | 0.506 / 0.960 | 0.768 / 0.576 |
| layer4 | 0.633 / 0.967 | 0.801 / 0.776 | 0.766 / 0.830 | 0.733 / 0.857 |

Improvement is the useful signal in this sweep. Layer3 has the best macro
ROC-AUC for both ID datasets (`0.885` and `0.768`); layer4 has the best
CIFAR-10 macro FPR@95 (`0.498`) and the best CIFAR-100-to-CIFAR-10 near-OOD
separation (`0.766 / 0.830`).

## Mean-Fill Identity Baseline

Score: `-identity_error`. This measures the uncorrected channel-mean-filled
feature map on exactly the same masked positions.

### CIFAR-10 ID

| Layer | MNIST ROC-AUC / FPR@95 | SVHN ROC-AUC / FPR@95 | CIFAR-100 ROC-AUC / FPR@95 | Macro ROC-AUC / FPR@95 |
| --- | ---: | ---: | ---: | ---: |
| layer1 | 0.503 / 0.987 | 0.059 / 0.999 | 0.521 / 0.914 | 0.361 / 0.967 |
| layer2 | 0.712 / 0.851 | 0.069 / 0.998 | 0.459 / 0.950 | 0.413 / 0.933 |
| layer3 | 0.254 / 1.000 | 0.035 / 1.000 | 0.209 / 0.999 | 0.166 / 1.000 |
| layer4 | 0.068 / 1.000 | 0.125 / 1.000 | 0.220 / 0.998 | 0.138 / 0.999 |

### CIFAR-100 ID

| Layer | MNIST ROC-AUC / FPR@95 | SVHN ROC-AUC / FPR@95 | CIFAR-10 ROC-AUC / FPR@95 | Macro ROC-AUC / FPR@95 |
| --- | ---: | ---: | ---: | ---: |
| layer1 | 0.336 / 0.998 | 0.033 / 1.000 | 0.484 / 0.968 | 0.285 / 0.989 |
| layer2 | 0.594 / 0.968 | 0.115 / 0.999 | 0.481 / 0.977 | 0.396 / 0.981 |
| layer3 | 0.091 / 1.000 | 0.099 / 0.999 | 0.494 / 0.970 | 0.228 / 0.990 |
| layer4 | 0.372 / 0.999 | 0.199 / 1.000 | 0.237 / 0.998 | 0.269 / 0.999 |

The identity baseline is consistently weak. The large gain from identity to
improvement shows that most usable separation comes from the learned residual,
not from the magnitude of the mean-fill corruption itself.

## Exported Artifacts

Machine-readable metrics:

```text
reports/outputs/json/spatial_block_residual_losses.json
reports/outputs/json/spatial_block_residual_default.json
reports/outputs/json/spatial_block_residual_improvement.json
reports/outputs/json/spatial_block_residual_negative_identity.json
```

Only masked-region metrics are reported; no full-map metric is included.
