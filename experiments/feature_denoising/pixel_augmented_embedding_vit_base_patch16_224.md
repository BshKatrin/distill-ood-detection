# Pixel-Augmented Embedding Prediction: ViT-Base Patch16-224

Status: completed for CIFAR-10 and CIFAR-100 ID, ViT-Base Patch16-224
teachers, CLS embeddings from transformer layers 3, 6, 10, 11, and 12.

This experiment repeats the pixel-augmentation Feature Denoising benchmark with a
teacher architecture that is very different from the ResNet runs:

```text
x_aug   -> teacher(layerN CLS) -> z_context
x_clean -> teacher(layerN CLS) -> z_target
student(z_context) -> z_pred
loss = MSE(z_pred, z_target)
```

## Setup

- CIFAR-10 teacher: `nateraw/vit-base-patch16-224-cifar10`.
- CIFAR-100 teacher: `MatanBT/vit-base-patch16-224-cifar100`.
- Teacher embedding: CLS token from layers `3`, `6`, `10`, `11`, and `12`.
- Student: one MLP per layer, input/output dimension `768`.
- Image preprocessing: resize/crop to `224x224`, ImageNet normalization.
- Augmentation: rotation `20` degrees, translation fraction `0.15`, scale
  range `0.80-1.20`, brightness delta `0.20`, contrast delta `0.35`.
- Evaluation draws: `10`.

Run directories:

```text
runs/students/feature_denoising/pixel_augmented_embedding/cifar_10/vit_base_patch16_224/mlp_layer3_aug_strong
runs/students/feature_denoising/pixel_augmented_embedding/cifar_10/vit_base_patch16_224/mlp_layer6_aug_strong
runs/students/feature_denoising/pixel_augmented_embedding/cifar_10/vit_base_patch16_224/mlp_layer10_aug_strong
runs/students/feature_denoising/pixel_augmented_embedding/cifar_10/vit_base_patch16_224/mlp_layer11_aug_strong
runs/students/feature_denoising/pixel_augmented_embedding/cifar_10/vit_base_patch16_224/mlp_layer12_aug_strong
runs/students/feature_denoising/pixel_augmented_embedding/cifar_100/vit_base_patch16_224/mlp_layer3_aug_strong
runs/students/feature_denoising/pixel_augmented_embedding/cifar_100/vit_base_patch16_224/mlp_layer6_aug_strong
runs/students/feature_denoising/pixel_augmented_embedding/cifar_100/vit_base_patch16_224/mlp_layer10_aug_strong
runs/students/feature_denoising/pixel_augmented_embedding/cifar_100/vit_base_patch16_224/mlp_layer11_aug_strong
runs/students/feature_denoising/pixel_augmented_embedding/cifar_100/vit_base_patch16_224/mlp_layer12_aug_strong
```

## CIFAR-10 ID

Training losses:

| Layer | Final validation | Best validation | Test |
|---|---:|---:|---:|
| layer3 | 0.018895 | 0.018895 | 0.019043 |
| layer6 | 0.163078 | 0.161693 | 0.162237 |
| layer10 | 0.410392 | 0.403709 | 0.403473 |
| layer11 | 1.496612 | 1.492736 | 1.533370 |
| layer12 | 9.512312 | 9.066995 | 9.801400 |

Default reconstruction score: `-raw_reconstruction_error`.

| Layer | MNIST ROC-AUC / FPR@95 | SVHN ROC-AUC / FPR@95 | CIFAR-100 ROC-AUC / FPR@95 | Macro ROC-AUC / FPR@95 |
|---|---:|---:|---:|---:|
| layer3 | 0.949 / 0.450 | 0.954 / 0.281 | 0.614 / 0.865 | 0.839 / 0.532 |
| layer6 | 1.000 / 0.000 | 0.989 / 0.039 | 0.702 / 0.773 | 0.897 / 0.271 |
| layer10 | 1.000 / 0.000 | 0.992 / 0.035 | 0.760 / 0.697 | 0.917 / 0.244 |
| layer11 | 0.991 / 0.000 | 0.958 / 0.290 | 0.835 / 0.754 | 0.928 / 0.348 |
| layer12 | 0.944 / 0.574 | 0.940 / 0.547 | 0.916 / 0.616 | 0.933 / 0.579 |

Improvement score:

```text
improvement = identity_error - reconstruction_error
```

| Layer | MNIST ROC-AUC / FPR@95 | SVHN ROC-AUC / FPR@95 | CIFAR-100 ROC-AUC / FPR@95 | Macro ROC-AUC / FPR@95 |
|---|---:|---:|---:|---:|
| layer3 | 1.000 / 0.000 | 0.205 / 0.971 | 0.517 / 0.911 | 0.574 / 0.627 |
| layer6 | 1.000 / 0.000 | 0.351 / 0.952 | 0.573 / 0.885 | 0.641 / 0.612 |
| layer10 | 1.000 / 0.000 | 0.589 / 0.766 | 0.654 / 0.792 | 0.748 / 0.519 |
| layer11 | 1.000 / 0.000 | 0.959 / 0.181 | 0.904 / 0.375 | 0.954 / 0.185 |
| layer12 | 0.989 / 0.016 | 0.861 / 0.377 | 0.895 / 0.267 | 0.915 / 0.220 |

Relative improvement score:

```text
relative_improvement = (identity_error - reconstruction_error) / max(identity_error, eps)
```

| Layer | MNIST ROC-AUC / FPR@95 | SVHN ROC-AUC / FPR@95 | CIFAR-100 ROC-AUC / FPR@95 | Macro ROC-AUC / FPR@95 |
|---|---:|---:|---:|---:|
| layer3 | 1.000 / 0.000 | 0.754 / 0.860 | 0.602 / 0.887 | 0.785 / 0.582 |
| layer6 | 1.000 / 0.000 | 0.941 / 0.389 | 0.696 / 0.806 | 0.879 / 0.399 |
| layer10 | 1.000 / 0.000 | 0.975 / 0.129 | 0.766 / 0.685 | 0.913 / 0.271 |
| layer11 | 1.000 / 0.000 | 0.977 / 0.119 | 0.922 / 0.386 | 0.966 / 0.168 |
| layer12 | 0.994 / 0.019 | 0.937 / 0.409 | 0.939 / 0.283 | 0.957 / 0.237 |

Cosine similarity score:

| Layer | MNIST ROC-AUC / FPR@95 | SVHN ROC-AUC / FPR@95 | CIFAR-100 ROC-AUC / FPR@95 | Macro ROC-AUC / FPR@95 |
|---|---:|---:|---:|---:|
| layer3 | 0.767 / 0.997 | 0.929 / 0.408 | 0.610 / 0.876 | 0.769 / 0.761 |
| layer6 | 0.990 / 0.008 | 0.989 / 0.037 | 0.694 / 0.781 | 0.891 / 0.275 |
| layer10 | 1.000 / 0.000 | 0.968 / 0.186 | 0.730 / 0.705 | 0.899 / 0.297 |
| layer11 | 0.973 / 0.123 | 0.946 / 0.409 | 0.908 / 0.529 | 0.942 / 0.354 |
| layer12 | 0.973 / 0.091 | 0.986 / 0.064 | 0.960 / 0.210 | 0.973 / 0.121 |

Takeaway: for CIFAR-10 ID, `layer12` is still the best single layer for cosine
similarity, with macro ROC-AUC / FPR@95 `0.973 / 0.121`. `layer11` is strongest
for relative improvement, with macro ROC-AUC / FPR@95 `0.966 / 0.168`, and is
also strong for absolute improvement. Near-OOD CIFAR-100 is best separated by
`layer12` for cosine and relative improvement, while `layer11` is competitive
and much better than earlier layers.

Fused cosine scores:

```text
score_final = zscore(cos_l6) + zscore(cos_l10) + zscore(cos_l12)
score_final_l10_l12 = zscore(cos_l10) + zscore(cos_l12)
score_final_l10_l12_min = min(zscore(cos_l10), zscore(cos_l12))
```

Three-layer fusion:

| MNIST ROC-AUC / FPR@95 | SVHN ROC-AUC / FPR@95 | CIFAR-100 ROC-AUC / FPR@95 | Macro ROC-AUC / FPR@95 |
|---:|---:|---:|---:|
| 1.000 / 0.000 | 0.997 / 0.004 | 0.905 / 0.354 | 0.967 / 0.119 |

Layer10+layer12 fusion:

| MNIST ROC-AUC / FPR@95 | SVHN ROC-AUC / FPR@95 | CIFAR-100 ROC-AUC / FPR@95 | Macro ROC-AUC / FPR@95 |
|---:|---:|---:|---:|
| 1.000 / 0.000 | 0.994 / 0.019 | 0.932 / 0.267 | 0.975 / 0.095 |

Layer10+layer12 min fusion:

| MNIST ROC-AUC / FPR@95 | SVHN ROC-AUC / FPR@95 | CIFAR-100 ROC-AUC / FPR@95 | Macro ROC-AUC / FPR@95 |
|---:|---:|---:|---:|
| 0.999 / 0.000 | 0.986 / 0.060 | 0.934 / 0.268 | 0.973 / 0.109 |

The layer10+layer12 fusion is the best CIFAR-10 ID aggregate score in this
benchmark. It keeps excellent far-OOD detection and improves near-OOD
CIFAR-100 compared with the three-layer fusion.

## CIFAR-100 ID

Training losses:

| Layer | Final validation | Best validation | Test |
|---|---:|---:|---:|
| layer3 | 0.027779 | 0.027192 | 0.026770 |
| layer6 | 0.249927 | 0.248185 | 0.246231 |
| layer10 | 0.905625 | 0.901039 | 0.901682 |
| layer11 | 2.285255 | 2.279761 | 2.284144 |
| layer12 | 18.100122 | 17.745545 | 18.070741 |

Default reconstruction score: `-raw_reconstruction_error`.

| Layer | MNIST ROC-AUC / FPR@95 | SVHN ROC-AUC / FPR@95 | CIFAR-10 ROC-AUC / FPR@95 | Macro ROC-AUC / FPR@95 |
|---|---:|---:|---:|---:|
| layer3 | 0.713 / 1.000 | 0.935 / 0.416 | 0.459 / 0.971 | 0.702 / 0.796 |
| layer6 | 0.998 / 0.000 | 0.982 / 0.087 | 0.430 / 0.974 | 0.803 / 0.354 |
| layer10 | 0.998 / 0.000 | 0.988 / 0.065 | 0.374 / 0.987 | 0.786 / 0.351 |
| layer11 | 0.995 / 0.003 | 0.971 / 0.169 | 0.476 / 0.969 | 0.814 / 0.380 |
| layer12 | 0.971 / 0.190 | 0.914 / 0.568 | 0.820 / 0.719 | 0.901 / 0.492 |

Improvement score:

| Layer | MNIST ROC-AUC / FPR@95 | SVHN ROC-AUC / FPR@95 | CIFAR-10 ROC-AUC / FPR@95 | Macro ROC-AUC / FPR@95 |
|---|---:|---:|---:|---:|
| layer3 | 0.999 / 0.000 | 0.112 / 0.989 | 0.478 / 0.976 | 0.530 / 0.655 |
| layer6 | 1.000 / 0.000 | 0.167 / 0.980 | 0.482 / 0.977 | 0.550 / 0.652 |
| layer10 | 1.000 / 0.000 | 0.325 / 0.934 | 0.529 / 0.972 | 0.618 / 0.635 |
| layer11 | 1.000 / 0.000 | 0.563 / 0.842 | 0.625 / 0.943 | 0.729 / 0.595 |
| layer12 | 0.996 / 0.013 | 0.598 / 0.885 | 0.821 / 0.550 | 0.805 / 0.483 |

Relative improvement score:

| Layer | MNIST ROC-AUC / FPR@95 | SVHN ROC-AUC / FPR@95 | CIFAR-10 ROC-AUC / FPR@95 | Macro ROC-AUC / FPR@95 |
|---|---:|---:|---:|---:|
| layer3 | 0.999 / 0.000 | 0.623 / 0.965 | 0.453 / 0.975 | 0.692 / 0.647 |
| layer6 | 1.000 / 0.000 | 0.831 / 0.854 | 0.434 / 0.978 | 0.755 / 0.611 |
| layer10 | 1.000 / 0.000 | 0.913 / 0.560 | 0.427 / 0.980 | 0.780 / 0.513 |
| layer11 | 1.000 / 0.000 | 0.909 / 0.550 | 0.586 / 0.952 | 0.832 / 0.501 |
| layer12 | 0.995 / 0.014 | 0.759 / 0.894 | 0.862 / 0.559 | 0.872 / 0.489 |

Cosine similarity score:

| Layer | MNIST ROC-AUC / FPR@95 | SVHN ROC-AUC / FPR@95 | CIFAR-10 ROC-AUC / FPR@95 | Macro ROC-AUC / FPR@95 |
|---|---:|---:|---:|---:|
| layer3 | 0.222 / 1.000 | 0.920 / 0.466 | 0.479 / 0.967 | 0.540 / 0.811 |
| layer6 | 0.886 / 0.800 | 0.977 / 0.106 | 0.464 / 0.964 | 0.776 / 0.623 |
| layer10 | 0.978 / 0.115 | 0.921 / 0.481 | 0.437 / 0.981 | 0.779 / 0.526 |
| layer11 | 0.924 / 0.369 | 0.818 / 0.717 | 0.656 / 0.911 | 0.799 / 0.666 |
| layer12 | 0.855 / 0.949 | 0.960 / 0.199 | 0.900 / 0.454 | 0.905 / 0.534 |

Takeaway: for CIFAR-100 ID, `layer12` again has the best macro ROC-AUC for the
most useful non-identity score families. `layer11` improves far-OOD
reconstruction over `layer12`, but it still fails near-OOD CIFAR-10. The best
FPR@95 is still not as clean: default reconstruction at `layer10` gives the
lowest macro FPR@95, `0.351`, because it is very strong on MNIST and SVHN, but
it fails near-OOD CIFAR-10. For near-OOD CIFAR-10, `layer12` is clearly best
across reconstruction, improvement, relative improvement, and cosine.

Fused cosine scores:

```text
score_final = zscore(cos_l6) + zscore(cos_l10) + zscore(cos_l12)
score_final_l10_l12 = zscore(cos_l10) + zscore(cos_l12)
score_final_l10_l12_min = min(zscore(cos_l10), zscore(cos_l12))
```

Three-layer fusion:

| MNIST ROC-AUC / FPR@95 | SVHN ROC-AUC / FPR@95 | CIFAR-10 ROC-AUC / FPR@95 | Macro ROC-AUC / FPR@95 |
|---:|---:|---:|---:|
| 0.975 / 0.135 | 0.989 / 0.040 | 0.717 / 0.895 | 0.894 / 0.357 |

Layer10+layer12 fusion:

| MNIST ROC-AUC / FPR@95 | SVHN ROC-AUC / FPR@95 | CIFAR-10 ROC-AUC / FPR@95 | Macro ROC-AUC / FPR@95 |
|---:|---:|---:|---:|
| 0.979 / 0.113 | 0.979 / 0.106 | 0.796 / 0.791 | 0.918 / 0.337 |

Layer10+layer12 min fusion:

| MNIST ROC-AUC / FPR@95 | SVHN ROC-AUC / FPR@95 | CIFAR-10 ROC-AUC / FPR@95 | Macro ROC-AUC / FPR@95 |
|---:|---:|---:|---:|
| 0.962 / 0.235 | 0.960 / 0.246 | 0.844 / 0.616 | 0.922 / 0.366 |

The layer10+layer12 sum fusion improves macro ROC-AUC and FPR@95 compared with
the three-layer fusion and is less bad on near-OOD CIFAR-10, but near-OOD
FPR@95 is still high. The min fusion improves near-OOD CIFAR-10 further,
from FPR@95 `0.791` to `0.616`, but gives up some MNIST/SVHN performance.

## Component Summary

| ID Dataset | Best Score | Best Layer | Macro ROC-AUC / FPR@95 | Notes |
|---|---|---:|---:|---|
| CIFAR-10 | `cosine_similarity` | layer12 | 0.973 / 0.121 | Best overall ViT result. |
| CIFAR-10 | `z(cos_l10)+z(cos_l12)` | fused | 0.975 / 0.095 | Best aggregate score and best macro FPR@95. |
| CIFAR-10 | `min(z(cos_l10), z(cos_l12))` | fused | 0.973 / 0.109 | Similar near-OOD behavior to sum fusion, slightly worse macro FPR. |
| CIFAR-10 | `z(cos_l6)+z(cos_l10)+z(cos_l12)` | fused | 0.967 / 0.119 | Strong far-OOD score, but weaker on near-OOD CIFAR-100 than layer10+layer12. |
| CIFAR-10 | `relative_improvement` | layer11 | 0.966 / 0.168 | Best non-fused relative-improvement result. |
| CIFAR-10 | `improvement` | layer11 | 0.954 / 0.185 | Better than layer12 for macro score, but less interpretable than relative improvement. |
| CIFAR-10 | `-raw_reconstruction_error` | layer10 | 0.917 / 0.244 | Best raw FPR@95; layer12 has better ROC-AUC but worse FPR. |
| CIFAR-100 | `cosine_similarity` | layer12 | 0.905 / 0.534 | Best macro ROC-AUC, but MNIST FPR is poor. |
| CIFAR-100 | `z(cos_l10)+z(cos_l12)` | fused | 0.918 / 0.337 | Best aggregate score and less weak on near-OOD CIFAR-10. |
| CIFAR-100 | `min(z(cos_l10), z(cos_l12))` | fused | 0.922 / 0.366 | Better near-OOD CIFAR-10 than sum fusion, worse far-OOD FPR. |
| CIFAR-100 | `z(cos_l6)+z(cos_l10)+z(cos_l12)` | fused | 0.894 / 0.357 | Better macro FPR@95, but poor near-OOD CIFAR-10 FPR. |
| CIFAR-100 | `-raw_reconstruction_error` | layer10 | 0.786 / 0.351 | Best macro FPR@95, but near-OOD CIFAR-10 is bad. |
| CIFAR-100 | `relative_improvement` | layer12 | 0.872 / 0.489 | Best relative-improvement layer and strongest on near-OOD CIFAR-10. |

## Interpretation

- ViT behaves differently from ResNet-18: the last transformer layer is the
  strongest layer for near-OOD in both ID settings.
- Relative improvement is much more useful than absolute improvement for ViT.
  Absolute improvement can be actively misleading for earlier layers,
  especially on SVHN.
- Cosine similarity between `z_pred` and `z_target` is surprisingly strong,
  especially for CIFAR-10 ID and `layer12`.
- The CIFAR-100 ID setting remains harder. `layer12` improves near-OOD CIFAR-10
  substantially, but FPR@95 is still high compared with far-OOD MNIST/SVHN.
- Training loss scale grows sharply with layer depth, so loss values should not
  be compared across layers as a quality ranking. The OOD metrics are the
  useful comparison signal here.
