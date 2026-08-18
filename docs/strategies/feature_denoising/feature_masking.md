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

### Channel-group-masked residual reconstruction

`channel_group_masked_residual_reconstruction` replaces independent Bernoulli
channel masking with a flat cut of a precomputed channel hierarchy. The current
experiments use the signed Pearson-correlation distance
`1 - Pearson correlation`, average linkage, and a cut distance of `0.5` from
the complete official ID training split. See
[Top-activation profile correlation](channel_grouping/top_activation_correlation.md)
for the grouping definition and artifact provenance.

For every example, one cluster is sampled uniformly from the clusters in the
selected cut. Every channel in that cluster is zeroed across its complete
spatial map. Cluster sampling is not weighted by cluster size, and singleton
clusters remain eligible. `mask_probability` is not used by this method.

The student architecture, residual target, and hidden-only reconstruction loss
are the same as for channel-masked residual reconstruction:

```text
group_id ~ Uniform({0, ..., group_count - 1})
hidden_channels = groups[group_id]
z_corrupted[hidden_channels, :, :] = 0
z_reconstructed = z_corrupted + delta_z
loss = MSE(z_reconstructed, z) over hidden_channels only
```

Training samples a new group independently for every example each time it is
seen. Inference performs `evaluation_draws` independent samples for every
example and averages their reconstruction errors. The current configs use 10
draws. Score artifacts also save `sampled_channel_group_indices` with shape
`(sample_count, evaluation_draws)`, making every random draw auditable.

The group artifact loader verifies that the artifact comes from the matching
complete ID training split and feature layer, matches the configured feature
shape, contains the requested cut, and partitions every channel exactly once.
For NMF latent-profile artifacts, which intentionally retain only the hierarchy
rather than predefined cuts, the loader derives the requested flat cut from the
saved linkage matrix and performs the same partition checks.

At distance `0.5`, the current ResNet-18 artifacts contain:

| ID dataset | Layer | Clusters | Cluster-size range | Singleton clusters |
| --- | --- | ---: | ---: | ---: |
| CIFAR-10 | layer1 | 22 | 1-9 | 11 |
| CIFAR-10 | layer2 | 91 | 1-4 | 63 |
| CIFAR-10 | layer3 | 170 | 1-10 | 126 |
| CIFAR-10 | layer4 | 36 | 4-35 | 0 |
| CIFAR-100 | layer1 | 17 | 1-16 | 3 |
| CIFAR-100 | layer2 | 89 | 1-6 | 65 |
| CIFAR-100 | layer3 | 252 | 1-2 | 248 |
| CIFAR-100 | layer4 | 401 | 1-3 | 297 |

### Stratified within-group channel masking

`channel_group_stratified_masked_residual_reconstruction` considers every
hierarchy-cut group with at least `channel_group_min_size` channels. For each
example and draw, it samples without replacement inside every eligible group
and hides:

```text
max(1, round(channel_group_mask_fraction * group_size))
```

The selected channels from all eligible groups form one combined mask. The
student and hidden-only residual loss are unchanged from channel-group-masked
residual reconstruction. The CIFAR-10 ResNet-18 layer4 configuration uses cut
distance `0.5`, minimum group size `4`, mask fraction `0.25`, and 10 evaluation
draws. All 36 groups qualify; discrete rounding masks 135 of 512 channels per
draw.

### Channel-masked k-NN reconstruction

`channel_masked_knn_reconstruction` and
`channel_group_masked_knn_reconstruction` replace the learned student with
exact k-NN regression over clean ID training activation maps. Each reference
and query remains a pre-GAP map with shape `(C, H, W)`. For every independently
sampled mask, squared L2 distance is calculated only over unmasked channels:

```text
distance(q, x; m) = sum(m * (q - x)^2)
```

Search uses batched GPU matrix multiplication and chunks the reference bank;
it does not use FAISS because each query has a different channel subspace. The
reference channelwise squared norms are precomputed, and each search keeps only
the running global top-k candidates. The hidden-channel prediction is the mean
of the selected neighbors' clean maps. Visible query channels are copied
unchanged. No student is trained or checkpointed.

The channel-group variant samples one complete hierarchy-cut cluster uniformly
per example and draw. It uses `channel_group_path` and
`channel_group_distance_threshold` with the same validation and sampling rules
as the learned channel-group residual method. It retains the sampled group IDs
alongside neighbor indices and distances.

The reference artifact must be the complete clean ID training split exported
by `export-teacher-activations`, with the same feature layer and map shape as
the query. Score artifacts retain neighbor indices, masked squared distances,
and visible-channel counts for every evaluation draw.

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

Channel-group-masked residual example:

```yaml
student:
  kind: feature_residual_denoiser
  feature_layer: layer3
  input_shape: [256, 8, 8]
  num_classes: 256

strategy:
  name: feature_denoising
  feature_denoising:
    method: channel_group_masked_residual_reconstruction
    channel_group_path: runs/channel_grouping/top_activation_correlation/cifar_10/resnet18/channel_groups/layer3.pt
    channel_group_distance_threshold: 0.5
    evaluation_draws: 10
```

Stratified within-group residual example:

```yaml
strategy:
  name: feature_denoising
  feature_denoising:
    method: channel_group_stratified_masked_residual_reconstruction
    channel_group_path: runs/channel_grouping/top_activation_correlation/cifar_10/resnet18/channel_groups/layer4.pt
    channel_group_distance_threshold: 0.5
    channel_group_min_size: 4
    channel_group_mask_fraction: 0.25
    evaluation_draws: 10
```

Channel-masked k-NN example:

```yaml
student:
  kind: knn
  feature_layer: layer3
  input_shape: [256, 8, 8]
  num_classes: 256

strategy:
  name: feature_denoising
  feature_denoising:
    method: channel_masked_knn_reconstruction
    activation_path: runs/teachers/cifar_10/resnet18/teacher_activations/cifar10_train/layer3.pt
    mask_probability: 0.2
    evaluation_draws: 10
    k_neighbors: 10
    knn_query_batch_size: 128
    knn_reference_chunk_size: 50000
```

Run score export directly; the method is inference-only:

```bash
distill-ood export-feature-denoising-scores --config <config>
```

Channel-group-masked k-NN changes the method and adds the hierarchy artifact:

```yaml
strategy:
  name: feature_denoising
  feature_denoising:
    method: channel_group_masked_knn_reconstruction
    activation_path: runs/teachers/cifar_10/resnet18/teacher_activations/cifar10_train/layer3.pt
    channel_group_path: runs/channel_grouping/top_activation_correlation/cifar_10/resnet18/channel_groups/layer3.pt
    channel_group_distance_threshold: 0.5
    evaluation_draws: 10
    k_neighbors: 10
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
