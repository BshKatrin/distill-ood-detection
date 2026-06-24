# Embedding-Space Perturbations

Embedding-space perturbations modify an intermediate image embedding produced
by the teacher model. The student is trained from the modified embedding while
the teacher supplies either a clean or perturbed target.

## Strategy

1. Extract an intermediate teacher embedding `z`.
2. Apply the selected perturbation to obtain `z_tilde`.
3. Continue the teacher forward pass from either `z` or `z_tilde`, according to
   `strategy.perturbation.teacher_target`.
4. Construct the method-specific student input from the perturbed embedding.
5. Predict `y_tilde_student` with the student.
6. Compute the distillation loss between the selected teacher target and
   `y_tilde_student`.

The implemented methods construct the student input differently:

- [Clipping](clipping.md):
  `concat(flatten(z_tilde), flatten(u))`
- [Monte-Carlo dropout](dropout.md): `flatten(z_tilde)`
- [PCA projection](pca.md): the reduced PCA representation of `flatten(z)`

## Teacher target

Set `strategy.perturbation.teacher_target` for every embedding perturbation:

- `clean`: continue the teacher from the original embedding `z`. Only the
  student input is perturbed. This is denoising distillation and is the default
  in the provided configs.
- `perturbed`: continue the teacher from `z_tilde`. This distills the teacher's
  behavior after the same perturbation.

For PCA projection, the perturbed teacher continuation uses a reconstruction in
the original embedding shape obtained with the PCA inverse transform.

## Methods

Set `strategy.perturbation.method` to select a method:

| Method                            | Configuration value   | Student input                              |
| --------------------------------- | --------------------- | ------------------------------------------ |
| [Clipping](clipping.md)           | `clipping` or omitted | Perturbed embedding and sampled percentile |
| [Monte-Carlo dropout](dropout.md) | `mc_dropout`          | Dropped embedding                          |
| [PCA projection](pca.md)          | `pca_projection`      | Reduced PCA representation                 |

Configs that omit `method` use clipping.

## Probability inference and OOD Score

Use `distill-ood infer-probabilities --apply-perturbation` to apply stochastic
perturbations during probability inference. The OOD Score for a sample can then
be estimated over multiple perturbation draws:

1. Generate `K > 0` independent perturbations.
2. Compute the selected OOD Score for every draw.
3. Average the scores across the draws.

Set `strategy.perturbation.evaluation_draws` to choose `K`. Without
`--apply-perturbation`, inference exports one deterministic draw. The exact
unperturbed behavior is method-specific and documented on each method page.

The two inference modes use separate artifact directories:

- `<run_dir>/probabilities/unperturbed/`
- `<run_dir>/probabilities/perturbed/`

## Implementation

Embedding perturbation sampling and student-input construction are implemented
in
[perturbation.py](../../../src/distill_ood_detection/distillation/perturbation.py).
ResNet continuation from an intermediate embedding is implemented by
`ResNetFeatureForwarder` in
[teacher.py](../../../src/distill_ood_detection/models/teacher.py).

Raw, pre-perturbation teacher activations can be exported with
`distill-ood export-teacher-activations`. The export contains the complete
official ID training split, the ID test split, and configured OOD splits. Each
selected layer is saved separately under
`runs/teachers/<id_dataset>/<teacher_architecture>/teacher_activations/<dataset_name>/`.
