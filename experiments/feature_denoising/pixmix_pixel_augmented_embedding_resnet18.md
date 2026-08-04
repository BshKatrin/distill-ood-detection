# PixMix Pixel-Augmented Embedding Prediction: ResNet-18

Status: completed for CIFAR-10 and CIFAR-100 ID.

This Feature Denoising variant predicts the clean pooled teacher embedding
from a PixMix-corrupted view:

```text
x_pixmix -> teacher(layer4) -> pooled z_context
x_clean  -> teacher(layer4) -> pooled z_target
student(z_context) -> z_pred
loss = MSE(z_pred, z_target)
```

## Setup

- Teacher: ResNet-18.
- Teacher feature: globally averaged `layer4` embedding.
- Student: MLP `512 -> 1024 -> 512 -> 512`.
- Feature Denoising method: `pixel_augmented_embedding_prediction`.
- Pixel augmentation method: PixMix.
- Mixing set: official `fractals_and_fvis` images.
- Mixing iterations: `4`.
- Beta: `3.0`.
- Augmentation severity: `3.0`.
- Operations: all PixMix augmentation operations.
- Working image size: `32`.
- Evaluation draws: `10`.
- Checkpoint selection: best validation reconstruction loss.
- ID datasets: CIFAR-10 and CIFAR-100 test.
- OOD datasets:
  - CIFAR-10 ID: MNIST, SVHN, and CIFAR-100 test;
  - CIFAR-100 ID: MNIST, SVHN, and CIFAR-10 test.

Run directories:

```text
runs/students/feature_denoising/pixel_augmented_embedding/cifar_10/resnet18/mlp_layer4_pixmix_avg
runs/students/feature_denoising/pixel_augmented_embedding/cifar_100/resnet18/mlp_layer4_pixmix_avg
```

## Training Results

| ID dataset | Best validation loss | Final validation loss | Test reconstruction loss |
|---|---:|---:|---:|
| CIFAR-10 | 0.015602 | 0.016330 | 0.016391 |
| CIFAR-100 | 0.075178 | 0.076590 | 0.064555 |

## OOD Scores

All scores follow the convention that higher values are more ID-like.

Raw reconstruction error:

```text
raw_reconstruction_error = mean((z_pred - z_target)^2)
score = -raw_reconstruction_error
```

Relative improvement:

```text
identity_error = mean((z_context - z_target)^2)
relative_improvement =
    (identity_error - raw_reconstruction_error)
    / max(identity_error, eps)
```

Cosine similarity:

```text
cosine_similarity = cosine(z_pred, z_target)
```

Raw reconstruction error is negated before OOD evaluation. Relative
improvement and cosine similarity are already oriented so that higher values
are more ID-like.

## OOD Results

All tables report `ROC-AUC / FPR@95`.

### CIFAR-10 ID

| Score | MNIST | SVHN | CIFAR-100 | Macro |
|---|---:|---:|---:|---:|
| Raw reconstruction error | 0.517 / 0.996 | 0.684 / 0.923 | 0.678 / 0.951 | 0.626 / 0.957 |
| Relative improvement | 0.772 / 0.838 | 0.686 / 0.928 | 0.742 / 0.781 | 0.733 / 0.849 |
| Cosine similarity | 0.705 / 0.944 | 0.853 / 0.737 | 0.759 / 0.875 | 0.772 / 0.852 |

Cosine similarity has the best macro ROC-AUC (`0.772`), while relative
improvement has a marginally better macro FPR@95 (`0.849` versus `0.852`).
Cosine similarity is strongest on SVHN, whereas relative improvement is
stronger on MNIST and has the best CIFAR-100 FPR@95. Raw reconstruction error
is weak, particularly at the 95% TPR operating point.

### CIFAR-100 ID

| Score | CIFAR-10 | MNIST | SVHN | Macro |
|---|---:|---:|---:|---:|
| Raw reconstruction error | 0.445 / 0.993 | 0.458 / 0.992 | 0.596 / 0.965 | 0.500 / 0.983 |
| Relative improvement | 0.570 / 0.919 | 0.758 / 0.858 | 0.664 / 0.918 | 0.664 / 0.898 |
| Cosine similarity | 0.628 / 0.933 | 0.539 / 0.961 | 0.856 / 0.621 | 0.674 / 0.838 |

Cosine similarity is strongest overall at macro `0.674 / 0.838`, driven by
SVHN (`0.856 / 0.621`). Relative improvement is better on MNIST but weaker in
aggregate. Near-OOD CIFAR-10 remains difficult, with FPR@95 above `0.90` for
all three scores.

## Comparison With Affine Pixel Augmentation

- For CIFAR-10 ID, the previously tested affine layer4 student is stronger:
  relative improvement reaches macro `0.886 / 0.542`, compared with
  `0.733 / 0.849` for PixMix.
- For CIFAR-100 ID, affine relative improvement also remains stronger at
  `0.808 / 0.713`, compared with `0.664 / 0.898`.
- PixMix cosine similarity improves CIFAR-100 macro FPR@95 relative to the
  affine layer4 cosine score (`0.838` versus `0.903`), but its ROC-AUC is
  lower (`0.674` versus `0.733`).

## Interpretation

- The raw PixMix reconstruction magnitude is not a useful OOD score.
- Normalizing by corruption difficulty or measuring angular agreement
  recovers a meaningful signal.
- Cosine similarity is the best aggregate PixMix Feature Denoising score,
  although relative improvement is preferable for CIFAR-10 ID when FPR@95 is
  prioritized.
- This layer4 PixMix variant does not outperform the existing affine
  pixel-augmentation Feature Denoising student.
- Near-OOD separation remains the main weakness for CIFAR-100 ID.

## Artifacts

Score artifacts:

```text
<run_dir>/feature_denoising_scores/
  manifest.json
  <dataset>/student_best.pt
```

Training job: `406156`.
Score-export job: `406350`.

Numeric metric exports:

```text
reports/outputs/json/pixmix_feature_denoising_default_ood_metrics.json
reports/outputs/json/pixmix_feature_denoising_relative_improvement_ood_metrics.json
reports/outputs/json/pixmix_feature_denoising_cosine_similarity_ood_metrics.json
```
