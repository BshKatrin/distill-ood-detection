# Feature Masked Reconstruction

Feature Masked Reconstruction predicts clean teacher feature maps from corrupted
maps. Direct methods predict the clean map; residual methods add the student
correction to the corrupted map. The teacher is frozen.

## Feature values

`z = teacher_feature(x, layer)` uses raw teacher feature-map values. Independent
spatial and channel masking zero hidden values without mean subtraction or
standard-deviation scaling. Spatial block masking fills hidden values with ID
training channel means. Confusion-channel replacement adjusts donor channels
using class-conditioned statistics. PCA whitening is specific to
[PCA Masked Reconstruction](pca_masking.md).

## Method families

| Family | Methods | Details |
| --- | --- | --- |
| Spatial masking | Independent locations; mean-filled residual blocks | [Spatial feature masking](spatial_masking.md) |
| Channel masking | Independent channels; direct or residual prediction | [Channel feature masking](channel_masking.md) |
| Grouped channel masking | Whole hierarchy-cut groups; stratified subsets | [Grouped channel masking](grouped_channel_masking.md) |
| k-NN reconstruction | Independent or grouped hidden channels; inference only | [k-NN reconstruction](knn_reconstruction.md) |
| Confusion-channel replacement | Class-conditioned donors; direct or residual prediction | [Confusion-channel replacement](confusion_channel_replacement.md) |

## Keep masks

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

Use `student.kind: feature_reconstructor` for direct spatial/channel reconstruction.
The student is a residual CNN:

```text
1x1 conv: C -> hidden
residual 3x3 conv blocks
1x1 conv: hidden -> C
```

Set `student.hidden_channels` to `[hidden_width, block_count]`.

Use `student.kind: feature_residual_denoiser` for spatial block residual
reconstruction. Its three padded `3x3` convolutions use widths
`C+1 -> max(16, C/4) -> max(16, C/4) -> C`, with ReLU after the first two
convolutions. `student.input_shape` includes the mask channel and
`student.num_classes` is `C`. Set a single `student.hidden_channels` value to
override the derived width; for example, `[64]` caps wider ResNet-50 stages at
64 channels while preserving the same three-convolution architecture.
For channel-masked residual reconstruction the widths are
`C -> hidden -> hidden -> C`; for spatial block residual reconstruction the
explicit mask makes the first width `C+1`.
Both confusion-channel replacement variants use the same
`C -> hidden -> hidden -> C` architecture as channel-masked residual
reconstruction. This holds architecture constant when comparing direct and
residual reconstruction.

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

For each sample and mask draw, expand `1 - m` to the target shape before counting
hidden scalar elements:

```text
hidden = expand_as(1 - m, z)
raw_error = sum((z_hat - z)^2 * hidden) / max(sum(hidden), 1)
```

This is the mean error over hidden elements only. Averaging the masked error
over every feature element would also scale it by the hidden fraction.
The sampler forces at least one hidden location or channel per sample.
Average per-sample raw errors over
`strategy.feature_denoising.evaluation_draws` independent draws, then use
`sign: -1` so the OOD Score is larger for more ID-like samples.

See [aggregate metrics](../../evaluation/metrics.md) for positive-class and
FPR@95 conventions.

## Implementation

Use `train-student` for learned reconstructors and
`export-feature-denoising-scores` for score export. k-NN reconstruction is
inference-only. Run these through the repository CLI with a synced `uv`
environment and `src/` on `PYTHONPATH`, as described in the
[root README](../../../README.md).
