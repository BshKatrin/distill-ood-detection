# PixMix Pixel-Augmented Embedding Prediction: ResNet-18

Status: training completed for CIFAR-10 and CIFAR-100 ID; OOD score export in
progress.

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

Pending completion of Feature Denoising score-export job `406350`.

Tables will report `ROC-AUC / FPR@95` for raw reconstruction error, relative
improvement, and cosine similarity on every OOD dataset and as a macro average.

## Artifacts

Score artifacts:

```text
<run_dir>/feature_denoising_scores/
  manifest.json
  <dataset>/student_best.pt
```

Training job: `406156`.

