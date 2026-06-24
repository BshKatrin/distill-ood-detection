# Analysis Notebooks

Notebook templates live under `reports/templates/notebooks/`. Generated
notebooks are written to `reports/outputs/notebooks/` and are not tracked in
Git.

Use `--number` with any generator to select a stable numeric filename prefix.
Without it, the generator chooses the next available number. Use `--overwrite`
to replace an existing generated notebook.

## Probability artifacts

Generate a notebook for an experiment with saved probability artifacts:

```bash
uv run --project envs/notebooks --no-sync python reports/scripts/create_probability_notebook.py \
  runs/students/baseline/cifar_10/resnet18/linear
```

The generator chooses the standard or perturbation template from the
experiment name. Use `--title` to override the generated notebook title.

## Teacher activations

Generate a teacher-activation notebook from a config under
`configs/teachers/`:

```bash
uv run --project envs/notebooks --no-sync python reports/scripts/create_teacher_activation_notebook.py \
  configs/teachers/cifar_10/resnet18.yaml
```

The matching activation manifest must already exist below the config's
`<run_dir>/teacher_activations/`. The same command works with other teacher
configs after their activation artifacts have been exported.

## PCA explained variance

Export leading PCA explained-variance statistics from a complete ID
training-split activation artifact and generate its plotting notebook:

```bash
uv run --project envs/notebooks --no-sync python reports/scripts/export_pca_explained_variance.py \
  runs/teachers/cifar_10/resnet18/teacher_activations/cifar10_train/layer4.pt
```

The exporter uses deterministic randomized PCA so it remains practical on a
local CPU. By default, it estimates 512 components. Increase the range with
`--max-components`, for example `--max-components 1024`, when the desired
variance threshold has not been reached.

The statistics JSON is written beside the activation artifact unless
`--output-path` is supplied. The notebook is generated from
`reports/templates/notebooks/pca_explained_variance.ipynb`.
