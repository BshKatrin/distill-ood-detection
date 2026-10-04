# Confusion-Channel Replacement

See [Feature Masked Reconstruction](feature_masking.md) for shared feature-value,
student, and hidden-element MSE definitions.

## Confusion-channel replacement

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

## Configuration

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
