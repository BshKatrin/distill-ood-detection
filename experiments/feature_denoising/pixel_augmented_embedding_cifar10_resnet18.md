# Pixel-Augmented Embedding Prediction: CIFAR-10 / ResNet-18

Status: completed for CIFAR-10 ID, ResNet-18 teacher, layers 1-4.

This variant replaces pixel masking with the existing pixel augmentation
strategy while keeping the Feature Denoising clean-target pipeline:

```text
x_aug   -> teacher(layerN) -> pooled z_context
x_clean -> teacher(layerN) -> pooled z_target
student(z_context) -> z_pred
loss = MSE(z_pred, z_target)
```

## Setup

- Teacher: ResNet-18 trained on CIFAR-10.
- ID dataset: CIFAR-10 test.
- OOD datasets: MNIST test, SVHN test, CIFAR-100 test.
- Strategy: `pixel_augmented_embedding_prediction`.
- Student: one MLP per teacher layer.
- Augmentation: rotation `20` degrees, translation fraction `0.15`, scale
  range `0.80-1.20`, brightness delta `0.20`, contrast delta `0.35`.
- Evaluation draws: `10`.
- Default score: `-raw_reconstruction_error`.

Run directories:

```text
runs/students/feature_denoising/pixel_augmented_embedding/cifar_10/resnet18/mlp_layer1_aug_strong
runs/students/feature_denoising/pixel_augmented_embedding/cifar_10/resnet18/mlp_layer2_aug_strong
runs/students/feature_denoising/pixel_augmented_embedding/cifar_10/resnet18/mlp_layer3_aug_strong
runs/students/feature_denoising/pixel_augmented_embedding/cifar_10/resnet18/mlp_layer4_aug_strong
```

## Training Losses

| Layer | Final Train Loss | Final Validation Loss | Best Validation Loss |
|---|---:|---:|---:|
| layer1 | 0.000280 | 0.000281 | 0.000280 |
| layer2 | 0.000247 | 0.000249 | 0.000249 |
| layer3 | 0.000175 | 0.000177 | 0.000177 |
| layer4 | 0.010855 | 0.011189 | 0.010538 |

As with masking, loss scale differs by layer and should not be interpreted as
a direct cross-layer quality measure.

## Default Reconstruction Score

Score: `-raw_reconstruction_error`.

| Layer | MNIST ROC-AUC / FPR@95 | SVHN ROC-AUC / FPR@95 | CIFAR-100 ROC-AUC / FPR@95 | Macro ROC-AUC / FPR@95 |
|---|---:|---:|---:|---:|
| layer1 | 0.837 / 0.936 | 0.950 / 0.266 | 0.620 / 0.848 | 0.802 / 0.683 |
| layer2 | 0.654 / 0.935 | 0.968 / 0.158 | 0.634 / 0.835 | 0.752 / 0.643 |
| layer3 | 0.365 / 0.999 | 0.735 / 0.803 | 0.459 / 0.964 | 0.519 / 0.922 |
| layer4 | 0.742 / 0.979 | 0.708 / 0.963 | 0.832 / 0.793 | 0.761 / 0.912 |

Takeaway: `layer1` has the best macro ROC-AUC for the default reconstruction
score, while `layer2` has the best macro FPR@95. `layer2` is again strongest on
SVHN, with ROC-AUC `0.968` and FPR@95 `0.158`. Unlike the masked variant,
`layer4` becomes useful for CIFAR-100, with ROC-AUC `0.832`.

## Improvement Score

Score:

```text
improvement = identity_error - reconstruction_error
```

| Layer | MNIST ROC-AUC / FPR@95 | SVHN ROC-AUC / FPR@95 | CIFAR-100 ROC-AUC / FPR@95 | Macro ROC-AUC / FPR@95 |
|---|---:|---:|---:|---:|
| layer1 | 0.999 / 0.000 | 0.905 / 0.316 | 0.511 / 0.925 | 0.805 / 0.414 |
| layer2 | 0.999 / 0.000 | 0.847 / 0.422 | 0.554 / 0.897 | 0.800 / 0.440 |
| layer3 | 0.998 / 0.000 | 0.921 / 0.349 | 0.779 / 0.805 | 0.899 / 0.384 |
| layer4 | 0.934 / 0.410 | 0.867 / 0.774 | 0.762 / 0.736 | 0.854 / 0.640 |

Takeaway: `layer3` is clearly best for the improvement score. It has the best
macro ROC-AUC and macro FPR@95, and it is the best layer for CIFAR-100 under
this score.

## Component Summary

| Score | Best Layer | Macro ROC-AUC / FPR@95 | Notes |
|---|---:|---:|---|
| `-raw_reconstruction_error` | layer1 | 0.802 / 0.683 | Best macro ROC-AUC for default score. |
| `-raw_reconstruction_error` | layer2 | 0.752 / 0.643 | Best macro FPR@95 for default score. |
| `improvement` | layer3 | 0.899 / 0.384 | Best overall score. |
| `cosine_similarity` | layer4 | 0.819 / 0.822 | Best cosine score, still high FPR@95. |
| `-identity_error` | layer4 | 0.498 / 0.965 | Identity alone is not useful. |

## Comparison With Pixel Masking

- Pixel augmentation substantially improves the `improvement` score:
  masked `layer3` macro ROC-AUC / FPR@95 was `0.765 / 0.873`; augmented
  `layer3` is `0.899 / 0.384`.
- Pixel augmentation also improves CIFAR-100 for high layers:
  default `layer4` CIFAR-100 ROC-AUC / FPR@95 improves from `0.649 / 0.969`
  with masking to `0.832 / 0.793` with augmentation.
- `layer2` remains strong for SVHN under raw reconstruction in both variants:
  masking gives `0.957 / 0.196`, augmentation gives `0.968 / 0.158`.
- The best overall current single-layer Feature Denoising score is pixel augmentation with
  `layer3` and the `improvement` score.

## Interpretation

- Pixel augmentation is more promising than hard pixel masking for this
  Feature Denoising setup.
- `improvement` is much stronger than raw reconstruction error for the
  augmented variant, especially on MNIST and CIFAR-100.
- `layer3` is the best single layer when identity correction is used.
- `layer2` is still the best raw-reconstruction layer for SVHN.
- `layer4` becomes useful for CIFAR-100 under augmentation, which suggests
  semantic high-level features benefit more from realistic image corruptions
  than from zero-filled masks.
