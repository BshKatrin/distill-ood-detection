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

Confusion-channel replacement is the exception: only donor channels are
class-conditionally location-scale adjusted before being inserted into the raw
target feature map.

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

### Spatial block residual reconstruction

`spatial_block_residual_reconstruction` samples one square block independently
for each example. Its side length is selected uniformly from
`spatial_mask_block_sizes`, then its valid top-left position is sampled
uniformly. The block is shared across all channels.

Hidden feature values are replaced by the channel mean fitted from the
deterministic ID student-training subset. The student input concatenates the
corrupted feature map and a single binary channel where `1` means hidden and
`0` means visible:

```text
student_input = concat(z_corrupted, hidden_mask)
```

The residual CNN predicts `delta_z`; reconstruction and training loss are:

```text
z_reconstructed = z_corrupted + delta_z
loss = MSE(z_reconstructed, z) over hidden positions only
```

The channel statistics and their dataset, split, sample count, and feature
layer provenance are saved in `<run_dir>/feature_normalizer.pt`. Score export
loads this artifact and never refits statistics on inference datasets.

### Channel-masked residual reconstruction

`channel_masked_residual_reconstruction` independently hides every feature
channel with probability `mask_probability` for each example and draw. At
least one channel is forced hidden. A hidden channel is zeroed over its entire
spatial extent.

Unlike spatial block residual reconstruction, the mask is not concatenated to
the student input and no feature-normalizer artifact is fitted:

```text
student_input = z_corrupted = z * keep_mask
z_reconstructed = z_corrupted + delta_z
loss = MSE(z_reconstructed, z) over hidden channels only
```

Corrections on visible channels are not constrained by the loss and are not
included in OOD scoring.

### Confusion-channel replacement

`confusion_channel_replacement_reconstruction` and
`confusion_channel_replacement_residual_reconstruction` replace complete
channels with the same-index channels from an activation-map exemplar of a
confusing class. Both support any four-dimensional ResNet feature layer. The
current launch configs cover ResNet-18 layers 1-4 with CIFAR-10 or CIFAR-100
as ID.

Before training, one deterministic pass over the configured
`confusion_split` computes:

- the hard teacher confusion matrix;
- the most-confused off-diagonal class for every real class.

The current experiment configs set `confusion_split: test`. A separate
deterministic pass over the ID validation split computes:

- each class and channel's mean and population standard deviation over samples
  and spatial positions.

If a class has no hard confusion-split errors, its confusing class is the
off-diagonal class with the highest mean teacher probability. Remaining ties
are resolved by the lowest class index.

Fifty complete activation-map exemplars per real class are collected from the
deterministic ID student-training subset. This avoids requiring 50 examples of
every class in the random validation subset. The confusion matrix, statistics,
exemplars, source indices, and split provenance are saved in
`<run_dir>/class_channel_corruption.pt`.

For a feature map with source class `X`, one exemplar is sampled from
`Y = confusing_class[X]`. Exactly
`max(1, round(mask_probability * channels))` channel indices are sampled
without replacement. Each selected same-index donor channel is adjusted by:

```text
adjusted_Y_i =
    mean_X_i
    + std_X_i * (prototype_Y_i - mean_Y_i)
    / max(std_Y_i, class_statistics_epsilon)
```

The student receives the replaced feature map without an explicit mask.
The residual method reconstructs with
`replaced_features + student_output`; the direct method reconstructs with
`student_output`. Training and OOD scoring use MSE over replaced channels
only. Training and validation use the real class as `X`; score export uses the
teacher-predicted class because inference labels are unavailable. Every
evaluation draw independently resamples both the channel indices and the
class-`Y` exemplar.

## Student

Use `student.kind: feature_reconstructor` for v2. The student is a residual CNN:

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

Spatial block residual reconstruction:

```yaml
student:
  kind: feature_residual_denoiser
  feature_layer: layer3
  input_shape: [257, 8, 8]
  num_classes: 256

strategy:
  name: feature_denoising
  feature_denoising:
    method: spatial_block_residual_reconstruction
    spatial_mask_block_sizes: [1, 3]
    evaluation_draws: 10
```

ResNet-50 example with an explicit hidden width:

```yaml
student:
  kind: feature_residual_denoiser
  feature_layer: layer3
  input_shape: [1025, 8, 8]
  hidden_channels: [64]
  num_classes: 1024
```

Channel-masked residual example:

```yaml
student:
  kind: feature_residual_denoiser
  feature_layer: layer3
  input_shape: [1024, 8, 8]
  hidden_channels: [64]
  num_classes: 1024

strategy:
  name: feature_denoising
  feature_denoising:
    method: channel_masked_residual_reconstruction
    mask_probability: 0.2
    evaluation_draws: 10
```

Confusion-channel replacement residual example:

```yaml
student:
  kind: feature_residual_denoiser
  feature_layer: layer4
  input_shape: [512, 4, 4]
  num_classes: 512

strategy:
  name: feature_denoising
  feature_denoising:
    method: confusion_channel_replacement_residual_reconstruction
    mask_probability: 0.2
    prototype_count: 50
    class_statistics_epsilon: 0.000001
    confusion_split: test
    evaluation_draws: 10
```

## Implementation

Feature Masked Reconstruction training is implemented by
`distill-ood train-student`. Reconstruction-score export is implemented by
`distill-ood export-feature-denoising-scores`.
