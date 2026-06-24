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
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" uv run --project envs/gpu --no-sync python -m distill_ood_detection.cli export-teacher-activations --config configs/teachers/resnet18_cifar10.yaml
```

Then reference the ID training artifact in the student config:

```yaml
strategy:
  name: perturbation
  perturbation:
    method: pca_projection
    teacher_target: clean
    pca_components: 9
    pca_activation_path: runs/teacher_resnet18_cifar10/teacher_activations/cifar10_train/layer4.pt
```

Teacher-activation export includes the complete official ID training split as
well as the ID and OOD test splits. Training fits the projector once from
`pca_activation_path` and saves it as:

```text
runs/<experiment_name>/pca_projector.pt
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

- [`cifar_10/linear_layer4_pca9.yaml`](../../../configs/perturbation/cifar_10/linear_layer4_pca9.yaml)
- [`cifar_100/linear_layer4_pca100.yaml`](../../../configs/perturbation/cifar_100/linear_layer4_pca100.yaml)

## Probability inference

Probability inference reloads the saved projector for every ID and OOD test
split. It never fits or updates PCA from evaluation activations. PCA projection
is applied to `flatten(z)` in both perturbed and unperturbed inference modes.

## Implementation

Projector fitting, serialization, reconstruction, and student-input
construction are implemented in
[perturbation.py](../../../src/distill_ood_detection/distillation/perturbation.py).
