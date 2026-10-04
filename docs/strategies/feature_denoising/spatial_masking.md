# Spatial Feature Masking

See [Feature Masked Reconstruction](feature_masking.md) for shared feature-value,
student, and hidden-element MSE definitions.

## Independent spatial masking

`spatial_masked_reconstruction` samples a Bernoulli keep mask with shape
`(batch, 1, height, width)`, shared across channels. The student receives
`z * m`, without an explicit mask channel, and predicts the clean raw map.
At least one location is hidden per sample.

## Spatial block residual reconstruction

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
