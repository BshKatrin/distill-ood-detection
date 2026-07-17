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

## Relative Improvement Score

Score:

```text
relative_improvement = (identity_error - reconstruction_error) / max(identity_error, eps)
```

| Layer | MNIST ROC-AUC / FPR@95 | SVHN ROC-AUC / FPR@95 | CIFAR-100 ROC-AUC / FPR@95 | Macro ROC-AUC / FPR@95 |
|---|---:|---:|---:|---:|
| layer1 | 0.999 / 0.000 | 0.965 / 0.184 | 0.594 / 0.885 | 0.853 / 0.356 |
| layer2 | 0.999 / 0.000 | 0.951 / 0.222 | 0.616 / 0.856 | 0.855 / 0.359 |
| layer3 | 0.999 / 0.000 | 0.925 / 0.326 | 0.746 / 0.816 | 0.890 / 0.381 |
| layer4 | 0.954 / 0.231 | 0.885 / 0.621 | 0.819 / 0.774 | 0.886 / 0.542 |

Takeaway: `layer3` has the best macro ROC-AUC, while `layer1` has the best
macro FPR@95 by a narrow margin. Higher layers remain stronger for near-OOD
CIFAR-100.

## Component Summary

| Score | Best Layer | Macro ROC-AUC / FPR@95 | Notes |
|---|---:|---:|---|
| `-raw_reconstruction_error` | layer1 | 0.802 / 0.683 | Best macro ROC-AUC for default score. |
| `-raw_reconstruction_error` | layer2 | 0.752 / 0.643 | Best macro FPR@95 for default score. |
| `relative_improvement` | layer3 | 0.890 / 0.381 | Best macro ROC-AUC. |
| `relative_improvement` | layer1 | 0.853 / 0.356 | Best macro FPR@95. |
| `cosine_similarity` | layer4 | 0.819 / 0.822 | Best cosine score, still high FPR@95. |
| `-identity_error` | layer4 | 0.498 / 0.965 | Identity alone is not useful. |

## Comparison With Pixel Masking

- Pixel augmentation makes the normalized student correction useful:
  `layer3` relative improvement reaches macro ROC-AUC / FPR@95
  `0.890 / 0.381`.
- Pixel augmentation also improves CIFAR-100 for high layers:
  default `layer4` CIFAR-100 ROC-AUC / FPR@95 improves from `0.649 / 0.969`
  with masking to `0.832 / 0.793` with augmentation.
- `layer2` remains strong for SVHN under raw reconstruction in both variants:
  masking gives `0.957 / 0.196`, augmentation gives `0.968 / 0.158`.
- The best current single-layer Feature Denoising score for CIFAR-10 ID is
  pixel augmentation with `layer3` and the relative improvement score by
  macro ROC-AUC.

## Interpretation

- Pixel augmentation is more promising than hard pixel masking for this
  Feature Denoising setup.
- Relative improvement is much stronger than raw reconstruction error for the
  augmented variant, especially on MNIST and CIFAR-100.
- `layer3` is the best single layer by macro ROC-AUC when identity correction
  is normalized by the perturbation size.
- `layer2` is still the best raw-reconstruction layer for SVHN.
- `layer4` becomes useful for CIFAR-100 under augmentation, which suggests
  semantic high-level features benefit more from realistic image corruptions
  than from zero-filled masks.
