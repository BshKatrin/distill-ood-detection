# Pixel-Augmented Embedding Prediction

Pixel-Augmented Embedding Prediction trains a student to map a teacher embedding
from an augmented image back to the teacher embedding from the clean image.

## Method

For a normalized image `x`, select either the affine/photometric augmentation
or PixMix:

```text
x_aug = pixel_augmentation(x)
z_context = avgpool(teacher_feature(x_aug, feature_layer))
z_target = avgpool(teacher_feature(x, feature_layer))
z_pred = student(z_context)
```

The teacher is frozen. The target is `stop_gradient(z_target)`.

With `pixel_augmentation_method: pixmix`, `x_aug` is the PixMix image and the
target view is the paper-style crop/flip view immediately before mixing.

Unlike the perturbation strategy, this Feature Denoising variant does not append the sampled
transform parameters to the student input. The benchmark asks whether the
teacher embedding from an augmented image contains enough ID-consistent
structure to predict the clean teacher embedding.

## Student

The benchmark configs use one MLP student per pooled ResNet-18 layer:

```yaml
student:
  kind: mlp
  feature_layer: layer3
  input_shape: [256]
  hidden_channels: [512, 512]
  num_classes: 256
```

For ResNet-18, the pooled dimensions are `layer1=64`, `layer2=128`,
`layer3=256`, and `layer4=512`.

## Loss

Training minimizes MSE between the predicted and clean pooled embeddings:

```text
loss = mean((z_pred - z_target) ** 2)
```

## Exported Scores

`distill-ood export-feature-denoising-scores` exports the same component scores as the
pixel-masked embedding variant:

- `raw_reconstruction_error`: MSE between `z_pred` and `z_target`;
- `scores`: `-raw_reconstruction_error`;
- `identity_error`: MSE between `z_context` and `z_target`;
- `improvement`: `identity_error - raw_reconstruction_error`;
- `cosine_similarity`: cosine similarity between `z_pred` and `z_target`;
- `z_context`, `z_pred`, and `z_target` for later ID-density analysis.

The default project OOD Score is:

- name: `feature_denoising_pixel_augmented_embedding_prediction_error`
- sign: `-1`

## Configuration

The first benchmark uses a stronger version of the documented pixel
augmentation recipe:

```yaml
strategy:
  name: feature_denoising
  feature_denoising:
    method: pixel_augmented_embedding_prediction
    pixel_augmentation_method: affine
    rotation_degrees: 20.0
    translate_fraction: 0.15
    scale_min: 0.80
    scale_max: 1.20
    brightness_delta: 0.20
    contrast_delta: 0.35
    embedding_pool: avg
    evaluation_draws: 10
```
