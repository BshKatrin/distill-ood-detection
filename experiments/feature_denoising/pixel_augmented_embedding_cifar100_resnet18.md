# Pixel-Augmented Embedding Prediction: CIFAR-100 / ResNet-18

Status: completed for CIFAR-100 ID, ResNet-18 teacher, layers 1-4.

This experiment repeats the CIFAR-10 pixel-augmentation benchmark with a
CIFAR-100 teacher and CIFAR-100 as the ID dataset.

## Setup

- Teacher: ResNet-18 trained on CIFAR-100.
- ID dataset: CIFAR-100 test.
- OOD datasets: MNIST test, SVHN test, CIFAR-10 test.
- Strategy: `pixel_augmented_embedding_prediction`.
- Student: one MLP per teacher layer.
- Augmentation: rotation `20` degrees, translation fraction `0.15`, scale
  range `0.80-1.20`, brightness delta `0.20`, contrast delta `0.35`.
- Evaluation draws: `10`.
- Default score: `-raw_reconstruction_error`.

Run directories:

```text
runs/students/feature_denoising/pixel_augmented_embedding/cifar_100/resnet18/mlp_layer1_aug_strong
runs/students/feature_denoising/pixel_augmented_embedding/cifar_100/resnet18/mlp_layer2_aug_strong
runs/students/feature_denoising/pixel_augmented_embedding/cifar_100/resnet18/mlp_layer3_aug_strong
runs/students/feature_denoising/pixel_augmented_embedding/cifar_100/resnet18/mlp_layer4_aug_strong
```

## Training Losses

| Layer | Final Train Loss | Final Validation Loss | Best Validation Loss |
|---|---:|---:|---:|
| layer1 | 0.000083 | 0.000083 | 0.000083 |
| layer2 | 0.000171 | 0.000173 | 0.000172 |
| layer3 | 0.000159 | 0.000164 | 0.000162 |
| layer4 | 0.041101 | 0.041703 | 0.040927 |

As in the CIFAR-10 runs, loss scale differs strongly by layer.

## Default Reconstruction Score

Score: `-raw_reconstruction_error`.

| Layer | MNIST ROC-AUC / FPR@95 | SVHN ROC-AUC / FPR@95 | CIFAR-10 ROC-AUC / FPR@95 | Macro ROC-AUC / FPR@95 |
|---|---:|---:|---:|---:|
| layer1 | 0.478 / 0.993 | 0.889 / 0.653 | 0.435 / 0.979 | 0.601 / 0.875 |
| layer2 | 0.517 / 0.983 | 0.900 / 0.504 | 0.441 / 0.979 | 0.619 / 0.822 |
| layer3 | 0.069 / 1.000 | 0.800 / 0.683 | 0.471 / 0.982 | 0.447 / 0.888 |
| layer4 | 0.648 / 0.936 | 0.653 / 0.958 | 0.696 / 0.931 | 0.666 / 0.941 |

Takeaway: raw reconstruction is weaker and less consistent than in the
CIFAR-10 ID setting. `layer4` has the best macro ROC-AUC, but its FPR@95 is
still very high. `layer2` has the best macro FPR@95 among raw reconstruction
scores, mostly because of SVHN.

## Relative Improvement Score

Score:

```text
relative_improvement = (identity_error - reconstruction_error) / max(identity_error, eps)
```

| Layer | MNIST ROC-AUC / FPR@95 | SVHN ROC-AUC / FPR@95 | CIFAR-10 ROC-AUC / FPR@95 | Macro ROC-AUC / FPR@95 |
|---|---:|---:|---:|---:|
| layer1 | 0.980 / 0.112 | 0.924 / 0.422 | 0.477 / 0.964 | 0.793 / 0.499 |
| layer2 | 0.994 / 0.013 | 0.935 / 0.338 | 0.475 / 0.972 | 0.802 / 0.441 |
| layer3 | 0.991 / 0.033 | 0.931 / 0.303 | 0.493 / 0.964 | 0.805 / 0.433 |
| layer4 | 0.903 / 0.481 | 0.835 / 0.725 | 0.686 / 0.932 | 0.808 / 0.713 |

Takeaway: `layer4` has the best macro ROC-AUC, mostly due to near-OOD
CIFAR-10, while `layer3` has the best macro FPR@95. CIFAR-10 remains hard as
near-OOD: every layer has FPR@95 above `0.90`.

## Component Summary

| Score | Best Layer | Macro ROC-AUC / FPR@95 | Notes |
|---|---:|---:|---|
| `-raw_reconstruction_error` | layer4 | 0.666 / 0.941 | Best raw ROC-AUC, poor FPR@95. |
| `-raw_reconstruction_error` | layer2 | 0.619 / 0.822 | Best raw FPR@95. |
| `relative_improvement` | layer4 | 0.808 / 0.713 | Best macro ROC-AUC. |
| `relative_improvement` | layer3 | 0.805 / 0.433 | Best macro FPR@95. |
| `cosine_similarity` | layer4 | 0.733 / 0.903 | Best cosine score, poor FPR@95. |
| `-identity_error` | layer4 | 0.395 / 0.994 | Identity alone is not useful. |

## Comparison With CIFAR-10 ID

- The most stable relative-improvement layer remains `layer3` by macro FPR@95.
- CIFAR-100 ID is harder overall because CIFAR-10 is near-OOD and remains
  difficult for all layer-wise scores.
- Far-OOD behavior is still good under relative improvement: MNIST ROC-AUC is
  `0.980-0.994` for layers 1-3.
- SVHN is strongest at `layer2` by ROC-AUC and `layer3` by FPR@95.

## Interpretation

- Pixel augmentation remains promising across ID datasets, but performance is
  not symmetric between CIFAR-10 and CIFAR-100.
- Relative improvement is still the most useful ResNet-18 score family.
- `layer3` is the most stable single layer across both ID settings when FPR@95
  is prioritized.
- Near-OOD remains the main weakness. For CIFAR-100 ID, CIFAR-10 is not
  separated well enough by any single layer.
