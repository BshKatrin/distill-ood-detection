# Clipping

Clipping modifies one or more ResNet feature layers by limiting their values to
sampled upper-percentile thresholds. Layers are processed in teacher-forward
order, so clipping an earlier layer changes every downstream activation.

## Algorithm

For each stochastic draw, sample `u` from a uniform distribution:

`u ~ Uniform(u_min, u_max)`

where `0 <= u_min < u_max <= 1`. Depending on the clipping mode, `u` is a
scalar, spatial matrix, or channel vector.

For every sampled percentile, compute the threshold over the corresponding
subset of embedding values and clip each value from above:

`z_tilde = min(z, threshold)`

For example, `u = 0.5` clips at the median and `u = 0.9` clips at the 90th
percentile. Smaller values of `u` produce stronger/aggressive perturbations.

## Clipping modes

Each convolutional embedding has spatial and channel dimensions. Set
`clipping_mode` independently under every configured clipping layer:

- `constant`: `u` is a scalar. One threshold is computed over the complete
  embedding and applied to every value.
- `spatial_dependent`: `u` has one value per spatial location. Each threshold
  is computed over the channels at that location.
- `channel_dependent`: `u` has one value per channel. Each threshold is
  computed over the spatial values in that channel and shared across spatial
  locations.

## Student input

After all configured clipping operations, the teacher continues through
`layer4`. The student receives the pooled final `layer4` activation and every
sampled percentile tensor in `layer1` to `layer4` order:

`concat(pool(z_layer4), flatten(u_layer1), ..., flatten(u_layer4))`

Only configured clipping layers contribute percentile tensors. Set
`strategy.perturbation.embedding_pool` to:

- `flatten`: flatten the final `layer4` feature map;
- `avg`: apply global average pooling to the final `layer4` feature map.

The teacher target is selected as described in the
[embedding-space strategy](README.md#teacher-target).

## Probability inference

With `--apply-perturbation`, inference samples clipping percentiles and exports
the configured number of stochastic draws. Without the flag, the student
receives the unmodified embedding and a neutral all-ones perturbation vector:

`concat(pool(z_layer4), flatten(1_layer1), ..., flatten(1_layer4))`

## Configuration

Clipping requires an explicit `method: clipping` and a non-empty
`clipping_layers` mapping. The former top-level `u_min`, `u_max`, and
`clipping_mode` fields are not supported.

```yaml
strategy:
  name: perturbation
  perturbation:
    method: clipping
    teacher_target: clean
    embedding_pool: avg
    clipping_layers:
      layer3:
        clipping_mode: channel_dependent
        u_min: 0.5
        u_max: 1.0
      layer4:
        clipping_mode: spatial_dependent
        u_min: 0.25
        u_max: 0.75
    evaluation_draws: 50
```

Supported keys are `layer1`, `layer2`, `layer3`, and `layer4`. Any subset is
accepted, and different clipping modes and percentile ranges may be configured
per layer.

See the clipping configs under
[`configs/students/perturbation/embedding/clipping/`](../../../configs/students/perturbation/embedding/clipping/).

## Implementation

Sampling, threshold calculation, and input construction are implemented in
[perturbation.py](../../../src/distill_ood_detection/distillation/perturbation.py).
