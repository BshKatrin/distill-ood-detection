# Spatial Block Residual Feature Denoising: ResNet-50

Status: completed for CIFAR-10 and CIFAR-100 ID, layers 1-4.

## Setup

- Teacher: ResNet-50.
- Batch size: `128`.
- Student hidden width: `64` for every layer.
- Corruption: one contiguous spatial feature block, shared across channels.
- Fill value: per-channel mean fitted from the deterministic ID
  student-training subset.
- Objective and scoring: masked-region reconstruction MSE only.
- Optimizer: AdamW, learning rate `0.001`, weight decay `0.0001`.
- Training: 50 epochs, seed `42`, best-validation checkpoint.
- Evaluation draws: `10`.
- MLflow: disabled.

| Layer | Feature map | Student input | Block side lengths |
| --- | --- | --- | --- |
| layer1 | `256x32x32` | `257x32x32` | `3`, `5` |
| layer2 | `512x16x16` | `513x16x16` | `3`, `5` |
| layer3 | `1024x8x8` | `1025x8x8` | `1`, `3` |
| layer4 | `2048x4x4` | `2049x4x4` | `1` |

The eight configs are listed in:

```text
slurm_scripts/feature_denoising_spatial_block_residual_resnet50.txt
```

## Cluster Jobs

| Job | ID dataset | Configs | Resources | State at submission |
| --- | --- | ---: | --- | --- |
| `407346` | CIFAR-10 | 4 | 2 GPUs, 8 CPUs, 64 GiB, 18 h | pending: GPU quota |
| `407347` | CIFAR-100 | 4 | 2 GPUs, 8 CPUs, 64 GiB, 18 h | pending: GPU quota |

Run directories mirror the config paths below
`runs/students/feature_denoising/feature_masking/`.

OOD results are consolidated in
[Residual Feature Masking: OOD Score Report](residual_feature_masking_ood_scores.md).
