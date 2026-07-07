# Pixel-Masked Embedding Prediction

Pixel-Masked Embedding Prediction trains a student to map a teacher embedding
from a masked image back to the teacher embedding from the clean image.

## Method

For a normalized image `x`, sample rectangular image masks and fill
hidden pixels with `0`, which corresponds to the dataset mean in normalized
image space:

```text
x_masked = x * keep_mask
z_context = avgpool(teacher_feature(x_masked, feature_layer))
z_target = avgpool(teacher_feature(x, feature_layer))
z_pred = student(z_context)
```

The teacher is frozen. The target is `stop_gradient(z_target)`.

## Student

The first implementation uses an MLP over pooled teacher embeddings. The
benchmark configs use one student per ResNet layer so layer-specific scores can
be compared without training a single concatenated student:

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

`distill-ood export-feature-denoising-scores` exports:

- `raw_reconstruction_error`: MSE between `z_pred` and `z_target`;
- `scores`: `-raw_reconstruction_error`;
- `identity_error`: MSE between `z_context` and `z_target`;
- `improvement`: `identity_error - raw_reconstruction_error`;
- `cosine_similarity`: cosine similarity between `z_pred` and `z_target`;
- `z_context`, `z_pred`, and `z_target` for later ID-density analysis.

The default project OOD Score is:

- name: `feature_denoising_pixel_embedding_prediction_error`
- sign: `-1`

## Configuration

```yaml
strategy:
  name: feature_denoising
  feature_denoising:
    method: pixel_masked_embedding_prediction
    image_mask_block_count: 2
    image_mask_scale_min: 0.15
    image_mask_scale_max: 0.20
    image_mask_aspect_ratio_min: 0.75
    image_mask_aspect_ratio_max: 1.50
    embedding_pool: avg
    evaluation_draws: 10
```
