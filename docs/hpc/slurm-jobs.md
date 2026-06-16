# SLURM jobs

Use [scripts/slurm/run_configs.sbatch](../../scripts/slurm/run_configs.sbatch)
for ordinary config-driven GPU jobs instead of creating one sbatch file per
experiment group.

The script accepts config paths as positional arguments:

```bash
sbatch \
  --job-name=cifar10_perturb \
  --gres=gpu:2 \
  scripts/run_configs.sbatch \
  configs/perturbation/cifar_10/linear_layer3_clip_constant.yaml \
  configs/perturbation/cifar_10/linear_layer3_clip_channel.yaml \
  configs/perturbation/cifar_10/linear_layer3_clip_spatial.yaml
```

By default, each config runs a train-then-infer pipeline. Configs whose path
contains `random_forest` use `train-tree-student`; all other configs use
`train-student`.

## Modes

Set `MODE` to choose the pipeline:

- `MODE=train-infer`: train the student, then infer probabilities. This is the
  default.
- `MODE=train`: train only.
- `MODE=infer`: infer probabilities only.

Examples:

```bash
sbatch --gres=gpu:1 --export=ALL,MODE=infer \
  scripts/slurm/run_configs.sbatch \
  configs/baseline/cifar_100/linear.yaml
```

```bash
sbatch --gres=gpu:1 --export=ALL,MODE=train \
  scripts/slurm/run_configs.sbatch \
  configs/baseline/cifar_10/feature_random_forest_layer3.yaml
```

## Parallelism

`PARALLEL=auto` runs up to one config pipeline per allocated GPU. This is the
default. For a single-command job that should avoid parallelization, request one
GPU or set `PARALLEL=1`:

```bash
sbatch --gres=gpu:1 --export=ALL,MODE=infer,PARALLEL=1 \
  scripts/slurm/run_configs.sbatch \
  configs/perturbation/cifar_10/linear_layer3_clip_constant.yaml
```

## Config list files

For long config lists, put one config path per line in a text file and submit
with `CONFIG_LIST`:

```bash
sbatch --gres=gpu:2 --export=ALL,CONFIG_LIST=configs_to_run.txt \
  scripts/slurm/run_configs.sbatch
```

Blank lines and lines beginning with `#` are ignored.

## Optional arguments

The script maps these environment variables to CLI flags:

- `CHECKPOINT`: `infer-probabilities --checkpoint`; default is `best`.
- `METHOD`: `train-student --method` and `infer-probabilities --method` for
  non-tree configs.
- `TREE_MODE`: `train-tree-student --mode` and
  `infer-probabilities --tree-mode` for tree configs.
- `INCLUDE_TRAIN=1`: add `infer-probabilities --include-train`.
- `INCLUDE_VALIDATION=1`: add `infer-probabilities --include-validation`.
