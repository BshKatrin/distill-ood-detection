# Clipping

Clipping modifies an intermediate teacher embedding `z` by limiting its values
to sampled upper-percentile thresholds.

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

The convolutional embedding has spatial and channel dimensions. Set
`strategy.perturbation.clipping_mode` to one of:

- `constant`: `u` is a scalar. One threshold is computed over the complete
  embedding and applied to every value.
- `spatial_dependent`: `u` has one value per spatial location. Each threshold
  is computed over the channels at that location.
- `channel_dependent`: `u` has one value per channel. Each threshold is
  computed over the spatial values in that channel and shared across spatial
  locations.

## Student input

The student receives the perturbed embedding and the sampled percentile:

`concat(flatten(z_tilde), flatten(u))`

The teacher target is selected as described in the
[embedding-space strategy](README.md#teacher-target).

## Probability inference

With `--apply-perturbation`, inference samples clipping percentiles and exports
the configured number of stochastic draws. Without the flag, the student
receives the unmodified embedding and a neutral all-ones perturbation vector:

`concat(flatten(z), flatten(1))`

## Configuration

Clipping is the default when `strategy.perturbation.method` is omitted. Relevant
fields are:

```yaml
strategy:
  name: perturbation
  perturbation:
    teacher_target: clean
    u_min: 0.5
    u_max: 1.0
    clipping_mode: constant
    evaluation_draws: 50
```

See the clipping configs under
[`configs/perturbation/`](../../../configs/perturbation/).

## Implementation

Sampling, threshold calculation, and input construction are implemented in
[perturbation.py](../../../src/distill_ood_detection/distillation/perturbation.py).
