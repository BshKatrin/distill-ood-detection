# Pixel Augmentation

The `pixel_augmentation` perturbation applies a mild composed image transform
before teacher feature extraction. It is intended to change pixels without
destroying the ID class semantics.

## Transformation

Each training encounter samples one transform per image:

- Rotation: `[-rotation_degrees, rotation_degrees]`
- Translation: integer x/y offsets up to `translate_fraction` of image width
  and height
- Scale: `[scale_min, scale_max]`
- Brightness factor: `[1 - brightness_delta, 1 + brightness_delta]`
- Contrast factor: `[1 - contrast_delta, 1 + contrast_delta]`

Affine transforms use bilinear interpolation and black fill for exposed border
pixels. Images are unnormalized, transformed in pixel space, clamped to
`[0, 1]`, and normalized again before the teacher receives them.

## Student input and target

The teacher extracts an embedding from the perturbed image at
`student.feature_layer`. `strategy.perturbation.embedding_pool` controls the
embedding representation:

```text
pool(teacher_embedding_from_perturbed_image)
+ normalized [angle, tx, ty, scale, brightness, contrast]
```

`flatten` preserves the existing behavior. `avg` globally averages the spatial
dimensions before appending the six parameters.

The six transformation parameters are normalized around identity, usually into
approximately `[-1, 1]`. Raw parameter values are not appended.

`strategy.perturbation.teacher_target` controls the loss target:

- `perturbed`: use teacher logits from the perturbed image.
- `clean`: run the teacher a second time on the clean image and use clean
  logits.

## Training and inference

Training uses one sampled composed transform per image per encounter. Validation,
test, and saved probability inference average stochastic predictions over
`evaluation_draws` when perturbation inference is requested.

The v1 implementation supports neural students (`linear` and `mlp`) and rejects
random-forest students.

## Configuration

Default moderate CIFAR-scale settings:

```yaml
strategy:
  name: perturbation
  perturbation:
    method: pixel_augmentation
    teacher_target: perturbed
    embedding_pool: flatten
    rotation_degrees: 10.0
    translate_fraction: 0.10
    scale_min: 0.90
    scale_max: 1.10
    brightness_delta: 0.10
    contrast_delta: 0.20
    evaluation_draws: 16
```

For a ResNet-50 `layer4` embedding on CIFAR images:

- `embedding_pool: flatten` uses `student.input_shape: [32774]`.
- `embedding_pool: avg` uses `student.input_shape: [2054]`.

Example configs:

- `configs/students/perturbation/pixel/augmentation/cifar_10/resnet50/linear_layer4_pixel_augmentation.yaml`
- `configs/students/perturbation/pixel/augmentation/cifar_100/resnet50/linear_layer4_pixel_augmentation.yaml`

## Implementation

Pixel augmentation sampling is implemented in
`src/distill_ood_detection/distillation/perturbation.py`. Training is wired
through `src/distill_ood_detection/distillation/train.py`, and saved probability
artifacts are collected by `src/distill_ood_detection/inference/outputs.py`.
