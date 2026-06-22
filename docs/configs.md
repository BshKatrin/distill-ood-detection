# Configs directory structure

The [configs](../configs/) directory contains `.yaml` configuration files for
experiments.

Configs are organized by the experiment identity, from broad strategy to
specific training dataset:

```text
configs/
  <ood_distillation_strategy>/
    <id_dataset>/
      <experiment_config>.yaml
  teachers/
    <teacher_export_config>.yaml
```

## First-level directory: OOD distillation strategy

The first-level directory names the OOD distillation strategy used by the
experiment.

Each strategy must be documented in [docs/strategies](strategies/). For example:

- [baseline](../configs/baseline/) corresponds to
  [docs/strategies/baseline.md](strategies/baseline.md).
- [perturbation](../configs/perturbation/) corresponds to
  [docs/strategies/perturbation.md](strategies/perturbation.md).

When adding a new first-level directory under `configs/`, also add or update the
matching strategy documentation in `docs/strategies/`.

## Second-level directory: ID training dataset

The second-level directory names the ID dataset used to train the student model.

Use the dataset name as it appears in the codebase and existing configs. Keep
names stable across strategies so experiments are easy to compare.

Examples:

- [configs/baseline/cifar_10](../configs/baseline/cifar_10/) contains baseline
  configs where CIFAR-10 is the ID training dataset.
- [configs/baseline/cifar_100](../configs/baseline/cifar_100/) contains baseline
  configs where CIFAR-100 is the ID training dataset.
- [configs/perturbation/cifar_10](../configs/perturbation/cifar_10/) contains
  perturbation configs where CIFAR-10 is the ID training dataset.

## Config file names

Individual `.yaml` files describe concrete experiment variants within a
strategy and ID dataset.

Name config files after the student model, feature source, or important variant
that distinguishes the experiment. For example:

- `linear.yaml`
- `mlp.yaml`
- `random_forest.yaml`
- `feature_linear_layer3.yaml`
- `linear_layer3.yaml`

When a strategy needs a stronger or alternate sampling regime, keep the
original file and add a sibling with a suffix that names the variant, such as
`_aggressive`, `_mc_dropout`, `_mc_dropout_channel`,
`_mc_dropout_spatial`, `_pca128`, or `_pca512`.

Prefer short, descriptive names that make related configs easy to scan.

## Teacher artifact configs

The [configs/teachers](../configs/teachers/) directory contains teacher-only
artifact export configs. These are not student training experiments; they define
which teacher checkpoint, datasets, and teacher layers should be used for
inference artifacts such as raw activations.

Teacher activation configs use a top-level `layers` list. Each requested layer
is saved to its own `.pt` file under:

```text
runs/<experiment_name>/teacher_activations/<dataset_name>/<layer>.pt
```

The matching manifest is saved at:

```text
runs/<experiment_name>/teacher_activations/manifest.json
```

Teacher probability configs omit `layers` and export deterministic raw-image
teacher logits and probabilities for the ID test split and configured OOD
datasets. Each dataset artifact is saved at:

```text
runs/<experiment_name>/teacher_probabilities/<dataset_name>/probabilities.pt
```

The matching manifest is saved at:

```text
runs/<experiment_name>/teacher_probabilities/manifest.json
```

## Student probability artifacts

`distill-ood infer-probabilities` writes student and teacher probability
artifacts under:

```text
runs/<experiment_name>/probabilities/
```

For perturbation-strategy experiments, inference can be run with or without
stochastic perturbations. These modes are separated to avoid overwriting:

```text
runs/<experiment_name>/probabilities/unperturbed/
runs/<experiment_name>/probabilities/perturbed/
```

Clipping perturbation configs use student input shapes that include both the
flattened feature tensor and the perturbation code. Monte-Carlo dropout configs
use only the flattened feature tensor because the dropout mask is not provided
to the student. PCA projection configs use `student.input_shape` equal to the
configured number of PCA components and require
`strategy.perturbation.pca_activation_path` to point at the exported complete
ID training-split teacher activation artifact. The fitted projector is then
reused unchanged for ID and OOD test inference.
