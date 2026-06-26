# PCA Projection

PCA projection reduces a flattened teacher embedding with a fixed Principal
Component Analysis (PCA) projection before passing it to the student:

`z_pca = PCA(flatten(z), n_components)`

Set the number of retained components with
`strategy.perturbation.pca_components`.

## Fit the projector

The PCA basis is fitted from a previously exported ID teacher-activation
artifact. Export teacher activations first, for example:

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" uv run --project envs/gpu --no-sync python -m distill_ood_detection.cli export-teacher-activations --config configs/teachers/cifar_10/resnet18.yaml
```

Then reference the ID training artifact in the student config:

```yaml
strategy:
  name: perturbation
  perturbation:
    method: pca_projection
    teacher_target: clean
    pca_components: 9
    pca_activation_path: runs/teachers/cifar_10/resnet18/teacher_activations/cifar10_train/layer4.pt
```

Teacher-activation export includes the complete official ID training split as
well as the ID and OOD test splits. Training fits the projector once from
`pca_activation_path` and saves it as:

```text
<run_dir>/pca_projector.pt
```

## Student input and teacher target

The student receives `z_pca`. With `teacher_target: clean`, the teacher
continues from the original embedding `z`. With `teacher_target: perturbed`, the
PCA inverse transform reconstructs an embedding in the original shape before
the remaining teacher layers run.

Choose `pca_components` from the cumulative explained variance of the complete
ID training-split activation artifact. See the
[PCA explained-variance notebook](../../reports/notebooks.md#pca-explained-variance).

Current examples are:

- [`cifar_10/linear_layer4_pca9.yaml`](../../../configs/students/perturbation/embedding/pca/cifar_10/resnet18/linear_layer4_pca9.yaml)
- [`cifar_100/linear_layer4_pca100.yaml`](../../../configs/students/perturbation/embedding/pca/cifar_100/resnet18/linear_layer4_pca100.yaml)

## Masked PCA projection

Masked PCA projection uses the same fitted top-k PCA components, then samples a
binary keep mask over the columns of the stored component matrix for each
sample. These columns correspond to flattened teacher-embedding dimensions:

`m_j ~ Bernoulli(1 - p)`

where `p = strategy.perturbation.pca_mask_probability` is the probability that
a component-matrix column is masked. The mask uses `1` for a kept column and
`0` for a masked column.

Conceptually, the sampled mask is applied to the PCA component matrix before the
dot product. If `C` is the stored component matrix with shape
`(pca_components, flattened_embedding_dim)` and `x = flatten(z) - mean`, the
masked projection is:

`z_masked_pca = x @ (C * m)^T`

The implementation computes the equivalent operation without materializing one
masked component matrix per sample:

`z_masked_pca = (x * m) @ C^T`

This is equivalent because each mask value zeros or keeps one complete column
of the component matrix before projection. The student receives the masked
projection and the keep mask:

`concat(z_masked_pca, m)`

Masked PCA projection supports only `teacher_target: clean`; the teacher
continues from the original embedding `z`.

Example:

```yaml
strategy:
  name: perturbation
  perturbation:
    method: pca_masked_projection
    teacher_target: clean
    pca_components: 9
    pca_mask_probability: 0.3
    pca_activation_path: runs/teachers/cifar_10/resnet18/teacher_activations/cifar10_train/layer4.pt
```

## Probability inference

Probability inference reloads the saved projector for every ID and OOD test
split. It never fits or updates PCA from evaluation activations. PCA projection
is applied to `flatten(z)` in both perturbed and unperturbed inference modes.
For masked PCA projection, unperturbed inference uses an all-ones keep mask,
while perturbed inference samples `strategy.perturbation.evaluation_draws`
independent masks.

## Implementation

Projector fitting, serialization, reconstruction, and student-input
construction are implemented in
[perturbation.py](../../../src/distill_ood_detection/distillation/perturbation.py).
