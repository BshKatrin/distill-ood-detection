# Feature Masked Reconstruction

Feature Masked Reconstruction is a Feature Denoising strategy that masks a teacher
feature map and trains a residual CNN student to reconstruct the clean feature
map.

## Feature values

Feature Masked Reconstruction uses raw teacher feature-map values:

`z = teacher_feature(x, layer)`

No mean subtraction or standard-deviation scaling is applied. Hidden positions
are zeroed in raw feature space so `0` remains the teacher feature value zero,
not an average normalized feature value.

## Methods

Spatial masking samples a keep mask over feature-map locations:

`m ~ Bernoulli(1 - p), shape: (batch, 1, height, width)`

Channel masking samples a keep mask over feature channels:

`m ~ Bernoulli(1 - p), shape: (batch, channels, 1, 1)`

In both methods, the student receives:

`z * m`

The keep mask is not concatenated to the student input. Hidden positions are
represented by zeros in raw feature space. The target is `stop_gradient(z)`.

The sampler forces at least one hidden location or channel per sample so the
hidden reconstruction loss is always defined.

## Student

Use `student.kind: feature_reconstructor` for v2. The student is a residual CNN:

```text
1x1 conv: C -> hidden
residual 3x3 conv blocks
1x1 conv: hidden -> C
```

Set `student.hidden_channels` to `[hidden_width, block_count]`.

Example for ResNet-18 `layer4` on CIFAR:

```yaml
student:
  kind: feature_reconstructor
  feature_layer: layer4
  input_shape: [512, 4, 4]
  hidden_channels: [256, 2]
  num_classes: 512
```

## OOD Score

For one mask draw:

`raw_error = mean(((z_hat - z) ** 2) * (1 - m))`

For stochastic inference, draw `strategy.feature_denoising.evaluation_draws` independent
masks and average the raw errors. Convert the raw reconstruction error into the
project OOD Score convention with `sign: -1`.

## Configuration

Spatial masking:

```yaml
strategy:
  name: feature_denoising
  feature_denoising:
    method: spatial_masked_reconstruction
    activation_path: runs/teachers/cifar_10/resnet18/teacher_activations/cifar10_train/layer4.pt
    mask_probability: 0.3
    evaluation_draws: 10
```

Channel masking:

```yaml
strategy:
  name: feature_denoising
  feature_denoising:
    method: channel_masked_reconstruction
    activation_path: runs/teachers/cifar_10/resnet18/teacher_activations/cifar10_train/layer4.pt
    mask_probability: 0.3
    evaluation_draws: 10
```

## Implementation

Feature Masked Reconstruction training is implemented by
`distill-ood train-student`. Reconstruction-score export is implemented by
`distill-ood export-feature-denoising-scores`.
