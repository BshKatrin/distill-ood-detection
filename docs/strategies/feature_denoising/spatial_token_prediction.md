# Spatial Token Prediction

Spatial Token Prediction is the Feature Denoising method closest to Feature Denoising in this
repository. The student never receives hidden feature values. It receives only
visible teacher feature tokens and target spatial positions, then predicts the
teacher features at those target positions.

## Teacher Features

The teacher is frozen. For an input image `x`, compute a raw teacher feature map:

`z = teacher_feature(x, layer), shape: (channels, height, width)`

For ResNet-18 `layer4` on CIFAR, `z` has shape `(512, 4, 4)`. No feature
normalization is applied.

## Masking

The masking policy follows the multi-block masking idea at feature-map scale:

- sample `target_block_count` target rectangles;
- target rectangle scale defaults to `(0.15, 0.20)`;
- target rectangle aspect ratio defaults to `(0.75, 1.50)`;
- sample one context rectangle with scale `(0.85, 1.00)` and unit aspect ratio;
- remove target positions from the visible context.

Because ResNet-18 `layer4` on CIFAR is only `4x4`, configs can set
`target_token_count` to make the task difficulty fixed across samples. The first
v3 config uses `target_token_count: 6`, so each draw predicts 6 of 16 spatial
tokens from the 10 visible context tokens.

## Student

Use `student.kind: spatial_token_predictor`.

```text
visible tokens: Linear(C -> hidden) + position embedding
context encoder: TransformerEncoder over visible tokens only
target queries: learned target token + target position embedding
cross-attention: target queries attend to visible context
prediction head: Linear(hidden -> C)
```

`student.hidden_channels` is interpreted as:

`[hidden_width, encoder_layer_count, attention_head_count]`

Example:

```yaml
student:
  kind: spatial_token_predictor
  feature_layer: layer4
  input_shape: [512, 4, 4]
  hidden_channels: [256, 2, 4]
  num_classes: 512
```

## Input, Output, Loss

The student input is a dictionary of padded visible context tokens:

```text
visible_tokens: [batch, max_tokens, channels]
visible_positions: [batch, max_tokens]e
visible_padding_mask: [batch, max_tokens]
target_positions: [batch, target_tokens]
```

The output is:

```text
predicted_target_tokens: [batch, target_tokens, channels]
```

The target is the stop-gradient raw teacher feature token at each target
position. The loss is MSE over target tokens only:

`mean((predicted_target_tokens - target_tokens) ** 2)`

## OOD Score

For one mask draw, the raw score is target-token reconstruction error:

`raw_error = mean((predicted_target_tokens - target_tokens) ** 2)`

For stochastic inference, draw `strategy.feature_denoising.evaluation_draws` masks and
average the raw errors. The project OOD Score is:

- name: `feature_denoising_spatial_token_prediction_error`
- sign: `-1`

Higher signed values remain more ID-like.

## Configuration

```yaml
strategy:
  name: feature_denoising
  feature_denoising:
    method: spatial_token_prediction
    activation_path: runs/teachers/cifar_10/resnet18/teacher_activations/cifar10_train/layer4.pt
    target_block_count: 2
    target_block_scale_min: 0.15
    target_block_scale_max: 0.20
    target_aspect_ratio_min: 0.75
    target_aspect_ratio_max: 1.50
    context_scale_min: 1.00
    context_scale_max: 1.00
    target_token_count: 6
    evaluation_draws: 10
```

## Implementation

Spatial Token Prediction training is implemented by `distill-ood train-student`.
Reconstruction-score export is implemented by `distill-ood export-feature-denoising-scores`.
