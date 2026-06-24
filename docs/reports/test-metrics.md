# Test Metric Reports

Test reports contain test accuracy and distillation loss. Their LaTeX tables
use saved method metrics under
`runs/<experiment_name>/<method>/metrics.json` together with experiment
metadata from `configs/`.

Each ID-dataset cell is formatted as
`test_accuracy/test_distillation_loss`. Both values are computed in one pass
over the test split after loading the checkpoint selected by the lowest
validation distillation loss.

## Strategy tables

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" uv run --project envs/notebooks --no-sync python reports/scripts/export_test_metrics_table.py
```

By default, the exporter reads `configs/baseline/` and
`configs/perturbation/`. It writes `metrics_test_baseline.tex` and
`metrics_test_perturbation.tex` under `reports/outputs/latex/` when matching
metrics exist.

Pass config files or directories to restrict the report:

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" uv run --project envs/notebooks --no-sync \
  python reports/scripts/export_test_metrics_table.py \
  configs/baseline/cifar_10 configs/perturbation/cifar_10
```

Baseline tables include `Student`, `Features`, and `Training` columns.
Perturbation tables also include `Perturbation`. Both retain `CIFAR-10 (ID)`
and `CIFAR-100 (ID)` columns and leave cells blank when local metrics are
missing. Matching displayed metadata is merged into one row block across ID
datasets.

The hardcoded Teacher row contains published test accuracies from the Hugging
Face model cards for
[`resnet18_cifar10`](https://huggingface.co/edadaltocg/resnet18_cifar10) and
[`resnet18_cifar100`](https://huggingface.co/edadaltocg/resnet18_cifar100).
Because those model cards do not publish distillation loss, teacher cells use
`accuracy/--`.

## Numeric metrics for explicit runs

Use `--run-names` to export test metrics for selected experiments:

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" uv run --project envs/notebooks --no-sync \
  python reports/scripts/export_test_metrics_table.py \
  --run-names <run-name> [<run-name> ...]
```

This mode computes test accuracy and distillation loss directly from each
probability mode's test artifacts instead of using the probability manifest.
It writes `reports/outputs/json/test_metrics.json` by default; use
`--json-output <path>` to override it.

Every method record contains `probability_mode`, `test_accuracy`, and
`test_distillation_loss`. Unsupported tree objectives use `null` for
distillation loss and do not fall back to a validation metric.

## SLURM

On the GPU cluster, submit the CPU-only exporter with:

```bash
sbatch slurm_scripts/export_test_metrics_table.sbatch
```

The job reads saved artifacts and writes generated report files; it should not
allocate a GPU. Read the [HPC documentation](../hpc/README.md) before using
cluster commands.
