# Perturbation Strategy

**Status**: Not implemented

This strategy is a perturbation-based stochastic distillation algorithm. The core idea is to perturb an intermediate image embedding produced by the teacher model, then train the student to predict the teacher output for that perturbed embedding.

The student must receive enough information to be perturbation-aware: it should observe both the original embedding and the perturbation applied to that embedding.

## Teacher Model

Use a relatively strong CNN teacher model: ResNet-18.

Train the teacher on the ID classification task. During distillation, the teacher provides the supervision signal for the student.

## Student Models

The following student architectures are evaluated:

- Linear model (no hidden layers)
- Multi-Layer Perceptron (MLP, 3 hidden layers)
- Random Forest

## Strategy

1. [Teacher] Extract an intermediate image embedding from the teacher model. Denote this embedding as `z`.
2. [Teacher] Sample a perturbation `u` and apply it to `z` to obtain the perturbed embedding `z_tilde`.
3. [Teacher] Continue the teacher forward pass from `z_tilde` through the remaining teacher layers to obtain `y_tilde_teacher`.
4. [Student] Concatenate `z` and `u`, then provide the concatenated vector to the student as input.
5. [Student] Predict `y_tilde_student`.
6. Compute the distillation loss between `y_tilde_teacher` and `y_tilde_student`.

`y_tilde_teacher` and `y_tilde_student` may be represented either as logits or as probability distributions after softmax, depending on the selected distillation objective.

### Perturbation strategies

#### Clipping-based

This perturbation strategy modifies an intermediate teacher embedding `z` by clipping its values to an upper percentile threshold.

For each stochastic draw, sample `u` from a uniform distribution:

`u ~ Uniform(u_min, u_max)`

where `0 <= u_min < u_max <= 1`.

Given the sampled percentile `u`, compute the clipping threshold as the `u`-percentile of the relevant subset of values. Each element is then clipped as `z' = min(z, threshold)`.

Examples:

- `u = 0.5` corresponds to clipping at the 50th percentile (median).
- `u = 0.9` corresponds to clipping at the 90th percentile.
- Smaller values of `u` produce stronger perturbations.

The intermediate embedding is extracted from a convolutional layer and has shape `(i, j, d)`, where `i, j` are spatial coordinates and `d` is the channel dimension.

##### Clipping modes

- `constant`: a single clipping threshold is computed using all values in the embedding. The same threshold is applied to every element.
- `spatial_dependent`: a separate clipping threshold is computed for each spatial location `(i, j)` using all channel values at that location.
- `channel_dependent`: a separate clipping threshold is computed for each channel `d` using all spatial values in that channel. The threshold is shared across spatial locations.

## OOD Score

The perturbation strategy is stochastic. Therefore, the OOD Score for a sample is estimated as the expected score over multiple perturbation draws.

For each test sample:

1. Generate `K > 0` independent perturbations.
2. Compute the selected OOD Score for each perturbation.
3. Average the scores across all perturbations.

## Distillation Objective

Train the student to reproduce the teacher output using one of the following objectives:

- `mse_logits`
- `cross_entropy`

See [Objectives](../objectives/README.md) for details.
