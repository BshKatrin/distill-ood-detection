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

Prefer short, descriptive names that make related configs easy to scan.
