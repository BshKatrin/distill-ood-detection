# Channel-Masked Residual Feature Denoising

Status: completed for ResNet-18 and ResNet-50, CIFAR-10 and CIFAR-100 ID,
layers 1-4.

## Setup

- Masking: each feature channel is independently hidden with probability
  `0.2`; at least one channel is hidden per example.
- Fill value: zero across the hidden channel's entire spatial extent.
- Student input: corrupted feature map only; no mask channel.
- Student output: a residual correction added to the corrupted feature map.
- Objective and scoring: reconstruction MSE on hidden channels only.
- Optimizer: AdamW, learning rate `0.001`, weight decay `0.0001`.
- Training: 50 epochs, seed `42`, best-validation checkpoint.
- Evaluation draws: `10`.
- MLflow: disabled.

| Backbone | Batch | Hidden widths by layer |
| --- | ---: | --- |
| ResNet-18 | 256 | 16, 32, 64, 128 |
| ResNet-50 | 128 | 64, 64, 64, 64 |

The 16 configs are listed in:

```text
slurm_scripts/feature_denoising_channel_residual_resnet18_resnet50.txt
```

## Cluster Jobs

| Job | Backbone | ID dataset | Configs | Resources | State at submission |
| --- | --- | --- | ---: | --- | --- |
| `407351` | ResNet-18 | CIFAR-10 | 4 | 2 GPUs, 8 CPUs, 64 GiB, 18 h | pending: GPU quota |
| `407352` | ResNet-18 | CIFAR-100 | 4 | 2 GPUs, 8 CPUs, 64 GiB, 18 h | pending: GPU quota |
| `407353` | ResNet-50 | CIFAR-10 | 4 | 2 GPUs, 8 CPUs, 64 GiB, 18 h | pending: GPU quota |
| `407354` | ResNet-50 | CIFAR-100 | 4 | 2 GPUs, 8 CPUs, 64 GiB, 18 h | pending: GPU quota |

Job `407351` completed layer1 before a transient DataLoader shared-memory
error. Retry job `407550` completed layers 2-4 with unchanged configurations.
The other three jobs completed normally.

Run directories mirror the config paths below
`runs/students/feature_denoising/feature_masking/`.

OOD results are consolidated in
[Residual Feature Masking: OOD Score Report](residual_feature_masking_ood_scores.md).
