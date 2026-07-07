# Pixel-Masked Embedding Prediction: CIFAR-10 / ResNet-18

Status: completed for CIFAR-10 ID, ResNet-18 teacher, layers 1-4.

## Setup

- Teacher: ResNet-18 trained on CIFAR-10.
- ID dataset: CIFAR-10 test.
- OOD datasets: MNIST test, SVHN test, CIFAR-100 test.
- Strategy: `pixel_masked_embedding_prediction`.
- Masking: two rectangular image masks, scale range `0.15-0.20`, aspect ratio
  range `0.75-1.50`.
- Student: one MLP per teacher layer.
- Representation: global-average-pooled teacher feature from the selected
  layer.
- Target: clean-image pooled teacher feature from the same layer.
- Context: pixel-masked-image pooled teacher feature from the same layer.
- Default score: `-raw_reconstruction_error`, where higher is more ID-like.

Run directories:

```text
runs/students/feature_denoising/pixel_masked_embedding/cifar_10/resnet18/mlp_layer1_blocks2_scale_015_020
runs/students/feature_denoising/pixel_masked_embedding/cifar_10/resnet18/mlp_layer2_blocks2_scale_015_020
runs/students/feature_denoising/pixel_masked_embedding/cifar_10/resnet18/mlp_layer3_blocks2_scale_015_020
runs/students/feature_denoising/pixel_masked_embedding/cifar_10/resnet18/mlp_layer4_blocks2_scale_015_020
```

## Training Losses

| Layer | Final Train Loss | Final Validation Loss | Best Validation Loss |
|---|---:|---:|---:|
| layer1 | 0.000173 | 0.000178 | 0.000174 |
| layer2 | 0.000197 | 0.000197 | 0.000196 |
| layer3 | 0.000228 | 0.000230 | 0.000229 |
| layer4 | 0.028476 | 0.029250 | 0.028155 |

The loss scale differs strongly by layer, so reconstruction loss values should
not be compared across layers as a direct measure of representation quality.

## Default Reconstruction Score

Score: `-raw_reconstruction_error`.

| Layer | MNIST ROC-AUC / FPR@95 | SVHN ROC-AUC / FPR@95 | CIFAR-100 ROC-AUC / FPR@95 | Macro ROC-AUC / FPR@95 |
|---|---:|---:|---:|---:|
| layer1 | 0.822 / 0.837 | 0.811 / 0.678 | 0.613 / 0.841 | 0.749 / 0.785 |
| layer2 | 0.825 / 0.764 | 0.957 / 0.196 | 0.635 / 0.817 | 0.805 / 0.592 |
| layer3 | 0.304 / 1.000 | 0.696 / 0.921 | 0.428 / 0.989 | 0.476 / 0.970 |
| layer4 | 0.510 / 1.000 | 0.555 / 0.981 | 0.649 / 0.969 | 0.571 / 0.983 |

Takeaway: `layer2` is the best single layer by the default reconstruction
score. Its strongest result is SVHN, with ROC-AUC `0.957` and FPR@95 `0.196`.
CIFAR-100 remains weak for every layer.

## Improvement Score

Score:

```text
improvement = identity_error - reconstruction_error
```

Higher means the student improves more over directly using the masked/context
teacher embedding.

| Layer | MNIST ROC-AUC / FPR@95 | SVHN ROC-AUC / FPR@95 | CIFAR-100 ROC-AUC / FPR@95 | Macro ROC-AUC / FPR@95 |
|---|---:|---:|---:|---:|
| layer1 | 0.648 / 0.999 | 0.724 / 0.793 | 0.481 / 0.917 | 0.618 / 0.903 |
| layer2 | 0.422 / 0.996 | 0.171 / 0.985 | 0.478 / 0.914 | 0.357 / 0.965 |
| layer3 | 0.803 / 0.980 | 0.726 / 0.831 | 0.767 / 0.808 | 0.765 / 0.873 |
| layer4 | 0.905 / 0.902 | 0.518 / 0.971 | 0.648 / 0.872 | 0.690 / 0.915 |

Takeaway: `layer3` is best overall for the improvement score. It is also the
best layer for CIFAR-100, with ROC-AUC `0.767` and FPR@95 `0.808`.

## Component Summary

| Score | Best Layer | Macro ROC-AUC / FPR@95 | Notes |
|---|---:|---:|---|
| `-raw_reconstruction_error` | layer2 | 0.805 / 0.592 | Best default score. |
| `improvement` | layer3 | 0.765 / 0.873 | Best identity-corrected signal. |
| `cosine_similarity` | layer4 | 0.724 / 0.948 | Usable ROC-AUC, poor FPR@95. |
| `-identity_error` | layer2 | 0.718 / 0.743 | Weaker than reconstruction. |

## Interpretation

- `layer2` looks most useful when the score is absolute reconstruction error.
- `layer3` looks most useful when the score is identity-corrected improvement.
- `layer4` alone is not a good default for masked pixel-Feature Denoising in this setup.
- CIFAR-100 is consistently the hardest OOD dataset, which is expected because
  it is near-OOD relative to CIFAR-10.
- The high FPR@95 values mean these scores are not yet production-quality OOD
  scores, but the layer benchmark gives a clear direction: favor layer2/layer3
  over layer4-only.
