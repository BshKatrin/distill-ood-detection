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
binary keep mask over the retained PCA eigenvectors for each sample:

`m_j ~ Bernoulli(1 - p)`

where `p = strategy.perturbation.pca_mask_probability` is the probability that
a PCA eigenvector is hidden. The mask uses `1` for a preserved eigenvector and
`0` for a hidden eigenvector.

Conceptually, the sampled mask is applied to the PCA component matrix before the
dot product. The implementation stores PCA components as `C` with shape
`(pca_components, flattened_embedding_dim)`. Equivalently, write the projection
matrix as `Q = C^T` with shape
`(flattened_embedding_dim, pca_components)`. The unmasked projection is:

`z_pca = (flatten(z) - mean) @ Q`

Masked PCA samples a per-sample keep mask `m` over the configured PCA
components:

`m ~ Bernoulli(1 - p), shape: (pca_components,)`

The mask zeros or keeps columns of `Q` before projection:

`Q_masked = Q * m`

`z_masked_pca = (flatten(z) - mean) @ Q_masked`

Each mask value zeros or keeps one PCA component. The student receives the
masked projection and the component keep mask:

`concat(z_masked_pca, m)`

With `teacher_target: clean`, the teacher continues from the original embedding
`z`. With `teacher_target: perturbed`, the teacher continues from a
reconstruction of the masked PCA projection in the original embedding shape.

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
