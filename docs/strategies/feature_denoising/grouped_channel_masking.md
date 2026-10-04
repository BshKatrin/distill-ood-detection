# Grouped Channel Masking

See [Feature Masked Reconstruction](feature_masking.md) for shared feature-value,
student, and hidden-element MSE definitions.

## Channel-group-masked residual reconstruction

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

## Stratified within-group channel masking

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

## Configuration

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
