# Perturbation Strategy

This strategy is a perturbation-based stochastic distillation algorithm. The core idea is to perturb an intermediate image embedding produced by the teacher model, then train the student to predict a teacher output from the perturbed embedding.

For clipping-based perturbations, the student receives both the perturbed embedding and the sampled perturbation applied to the original embedding. For Monte-Carlo dropout, the student receives only the dropped embedding.

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
4. [Student] Concatenate `z_tilde` and `u`, then provide the concatenated vector to the student as input.
5. [Student] Predict `y_tilde_student`.
6. Compute the distillation loss between `y_tilde_teacher` and `y_tilde_student`.

`y_tilde_teacher` and `y_tilde_student` may be represented either as logits or as probability distributions after softmax, depending on the selected distillation objective.

### Perturbation methods

Set `strategy.perturbation.method` to choose the perturbation method. Existing configs that omit this field use `clipping`.

#### Clipping-based

This perturbation strategy modifies an intermediate teacher embedding `z` by clipping its values to an upper percentile threshold.

For each stochastic draw, sample `u` from a uniform distribution. Depending on
the clipping mode, `u` may be a scalar, spatial matrix, or channel vector:

`u ~ Uniform(u_min, u_max)`

where `0 <= u_min < u_max <= 1`.

Given each sampled percentile in `u`, compute the clipping threshold as that
percentile of the relevant subset of values. Each element is then clipped as
`z' = min(z, threshold)`.

Examples:

- `u = 0.5` corresponds to clipping at the 50th percentile (median).
- `u = 0.9` corresponds to clipping at the 90th percentile.
- Smaller values of `u` produce stronger perturbations.

The repository keeps the standard configs at `u_min: 0.5, u_max: 1.0` and
adds `_aggressive` variants at `u_min: 0.0, u_max: 0.5` for stronger clipping.

The intermediate embedding is extracted from a convolutional layer and has shape `(i, j, d)`, where `i, j` are spatial coordinates and `d` is the channel dimension.

##### Clipping modes

- `constant`: `u` is a scalar. A single clipping threshold is computed using all values in the embedding. The same threshold is applied to every element.
- `spatial_dependent`: `u` has shape `(i, j)`. A separate clipping threshold is computed for each spatial location `(i, j)` using all channel values at that location.
- `channel_dependent`: `u` has shape `(d)`. A separate clipping threshold is computed for each channel `d` using all spatial values in that channel. The threshold is shared across spatial locations.

#### Monte-Carlo dropout

This perturbation method randomly drops elements of the intermediate teacher embedding before passing it to the student:

`z_drop = dropout(z, p)`

where `p = strategy.perturbation.dropout_probability`.

Set `strategy.perturbation.dropout_mode` to choose which parts of the feature tensor are dropped:

- `element`: Drop individual activation values independently.
- `channel`: Drop whole channels independently. A dropped channel is removed at every spatial location.
- `spatial`: Drop whole spatial locations independently. A dropped location is removed across all channels.

The teacher target remains the unperturbed teacher continuation from `z`. The student is therefore trained to predict the teacher output from incomplete feature evidence, rather than to imitate a teacher that has also seen the same dropped embedding.

At inference time, use `distill-ood infer-probabilities --apply-perturbation` to keep dropout active and export `K = strategy.perturbation.evaluation_draws` stochastic student draws. This is the Monte-Carlo dropout estimate. The implementation samples dropout masks directly for the feature tensor, so dropout remains active without putting the full model into training mode.

The first Monte-Carlo dropout configs use `dropout_probability: 0.5`. The unsuffixed files use `dropout_mode: element`; `_channel` and `_spatial` variants use structured dropout:

- `configs/perturbation/cifar_10/linear_layer4_mc_dropout.yaml`
- `configs/perturbation/cifar_10/linear_layer4_mc_dropout_channel.yaml`
- `configs/perturbation/cifar_10/linear_layer4_mc_dropout_spatial.yaml`
- `configs/perturbation/cifar_100/linear_layer4_mc_dropout.yaml`
- `configs/perturbation/cifar_100/linear_layer4_mc_dropout_channel.yaml`
- `configs/perturbation/cifar_100/linear_layer4_mc_dropout_spatial.yaml`

## OOD Score

The perturbation strategy can be stochastic at probability-inference time. This is controlled by the `distill-ood infer-probabilities --apply-perturbation` command-line flag.

When `--apply-perturbation` is set, test and OOD inference samples are perturbed in the same format used during training. The OOD Score for a sample can be estimated as the expected score over multiple perturbation draws:

1. Generate `K > 0` independent perturbations.
2. Compute the selected OOD Score for each perturbation.
3. Average the scores across all perturbations.

By default, probability inference uses the unmodified teacher embedding `z`. For clipping, the student still receives a perturbation-aware input vector, but the perturbation component is a neutral all-ones vector with the same shape as `u`, so the input is `concat(flatten(z), flatten(1))`. For Monte-Carlo dropout, the default unperturbed input is just `flatten(z)`.

Use `strategy.perturbation.evaluation_draws` to set `K` when running `infer-probabilities --apply-perturbation`. Without that flag, the process is not stochastic and probability inference exports one deterministic draw.

The two inference modes are saved under separate directories to avoid overwriting artifacts:

- default unperturbed inference: `runs/<experiment_name>/probabilities/unperturbed/`
- perturbed inference: `runs/<experiment_name>/probabilities/perturbed/`

## Distillation Objective

Train the student to reproduce the teacher output using one of the following objectives:

- `mse_logits`
- `cross_entropy`
- `kl_divergence`

See [Objectives](../objectives/README.md) for details.

## Implementation

- Perturbation sampling is implemented in [perturbation.py](../../src/distill_ood_detection/distillation/perturbation.py). For clipping perturbations, the student input is `concat(flatten(z_tilde), flatten(u))`. For Monte-Carlo dropout, the student input is `flatten(dropout(z, p))` and the teacher target is the unperturbed continuation from `z`.
- `distill-ood infer-probabilities --apply-perturbation` controls whether probability inference samples are perturbed. Training samples are always perturbed for this strategy.
- ResNet feature continuation is implemented by `ResNetFeatureForwarder` in [teacher.py](../../src/distill_ood_detection/models/teacher.py).
- Raw pre-perturbation teacher activations can be exported with `distill-ood export-teacher-activations --config configs/teachers/resnet18_cifar10_layers.yaml`. Each configured layer is saved as a separate `.pt` file under `runs/<experiment_name>/teacher_activations/<dataset_name>/`.
- PyTorch student training is implemented in [train_student.py](../../src/distill_ood_detection/experiments/train_student.py).
- Random-forest student training is implemented in [train_tree_student.py](../../src/distill_ood_detection/experiments/train_tree_student.py).
