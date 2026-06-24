# Monte-Carlo Dropout

Monte-Carlo dropout randomly drops values or structured regions of an
intermediate teacher embedding before passing it to the student:

`z_tilde = dropout(z, p)`

Set `strategy.perturbation.dropout_probability` to the dropout probability
`p`.

## Dropout modes

Set `strategy.perturbation.dropout_mode` to one of:

- `element`: drop individual activation values independently.
- `channel`: drop complete channels independently across all spatial
  locations.
- `spatial`: drop complete spatial locations independently across all
  channels.

## Student and teacher inputs

The student receives only the flattened dropped embedding:

`flatten(z_tilde)`

The supplied configs use the clean teacher continuation from `z`. Set
`teacher_target: perturbed` to continue the teacher from the dropped embedding
instead.

## Probability inference

Use `distill-ood infer-probabilities --apply-perturbation` to sample dropout
masks and export
`K = strategy.perturbation.evaluation_draws` stochastic student draws. The
implementation samples masks directly for the feature tensor, so dropout stays
active without putting the complete model into training mode.

Without `--apply-perturbation`, the student receives `flatten(z)` and inference
exports one deterministic draw.

## Configuration

```yaml
strategy:
  name: perturbation
  perturbation:
    method: mc_dropout
    teacher_target: clean
    dropout_probability: 0.8
    dropout_mode: element
    evaluation_draws: 50
```

The unsuffixed Monte-Carlo dropout configs use `element`; `_channel` and
`_spatial` variants use structured dropout. See the configs under
[`configs/perturbation/`](../../../configs/perturbation/).

## Implementation

Dropout-mask sampling and student-input construction are implemented in
[perturbation.py](../../../src/distill_ood_detection/distillation/perturbation.py).
