# Channel Feature Masking

See [Feature Masked Reconstruction](feature_masking.md) for shared feature-value,
student, and hidden-element MSE definitions.

## Independent channel masking

`channel_masked_reconstruction` samples a Bernoulli keep mask with shape
`(batch, channels, 1, 1)`. A hidden channel is zeroed across its full spatial
extent. The student predicts the clean raw map without receiving the mask.
At least one channel is hidden per sample.

## Channel-masked residual reconstruction

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

## Configuration

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
