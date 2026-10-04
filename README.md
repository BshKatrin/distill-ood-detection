# distill-ood-detection

Research code for Out-of-Distribution detection using model distillation.
Experiments cover CIFAR ResNet and ViT teachers, classical output distillation,
embedding and pixel perturbations, Feature Denoising, activation-subspace
students, and subspace ensembles. Fixed OpenOOD evaluations cover CIFAR,
ImageNet-200, and ImageNet-1K.

Start with the [documentation index](docs/index.md), the
[experiment catalogue](experiments/README.md), or the
[final report and slides](reports/README.md).

## Repository layout

| Directory | Responsibility |
| --- | --- |
| `src/distill_ood_detection/` | Importable training, inference, and evaluation code |
| `configs/` | Explicit experiment definitions and run directories |
| `envs/` | Focused `uv` environments for GPU jobs, notebooks, tests, and dashboards |
| `experiments/` | Curated experiment records, result JSON, and supporting figures |
| `reports/` | Final deliverables, dated recaps, and reporting tools |
| `docs/` | Method references, evaluation conventions, and operating workflows |
| `runs/` | Local/generated configs, histories, checkpoints, and inference artifacts |
| `slurm_scripts/` | Existing cluster job and submission scripts |
| `tests/` | Automated correctness checks |

## Reproduce a baseline

Dependencies use `uv`. Sync the required environment before using `--no-sync`;
the focused environments do not install the repository package, so put `src/`
on `PYTHONPATH`. Run these commands from the repository root:

```bash
uv sync --project envs/gpu
uv sync --project envs/notebooks
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

uv run --project envs/gpu --no-sync python -m distill_ood_detection.cli train-student \
  --config configs/students/baseline/cifar_100/resnet18/linear.yaml --method mse_logits

uv run --project envs/gpu --no-sync python -m distill_ood_detection.cli infer-probabilities \
  --config configs/students/baseline/cifar_100/resnet18/linear.yaml --checkpoint best --method mse_logits

uv run --project envs/notebooks --no-sync python reports/scripts/export_metrics_table.py \
  configs/students/baseline/cifar_100/resnet18/linear.yaml
```

This checked-in config selects a CUDA device, seed 42, CIFAR-100 ID, and
CIFAR-10/MNIST/SVHN OOD test splits. The teacher is downloaded from Hugging Face;
data and checkpoints must be available to the execution host. The commands
train, save inference artifacts, and export a historical ID-positive metric
table; see [metric conventions](docs/evaluation/metrics.md) before comparing
FPR@95 with OpenOOD results. Training uses the config's explicit `run_dir`;
the report command writes under `reports/outputs/latex/`.

MLflow is disabled and configuration loading rejects `mlflow.enabled: true`.
The resolved config, histories, metrics, and checkpoints in each run directory
are the authoritative experiment record. See [configs](docs/configs.md) for
the run hierarchy and [environments](docs/envs.md) for platform dependencies.
Dependency lock files are currently ignored; keep the local environment lock
and resolved run config when recording an exact reproduction.

## Other workflows

- [Methods](docs/strategies/README.md) and [objectives](docs/objectives/README.md)
  define student inputs, targets, and losses.
- [Evaluation](docs/evaluation/README.md) defines score signs, metrics, and
  fixed benchmark splits.
- [Reporting](docs/reports/README.md) covers metric exports and analysis notebooks.
- [Activation-subspace workflow](docs/workflows/activation_subspace.md) uses
  dedicated inference and score commands.
- [HPC operations](docs/hpc/README.md) apply when cluster access is needed;
  keep host-specific values in the ignored local configuration.

Run CPU correctness tests with:

```bash
uv sync --project envs/tests
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" uv run --project envs/tests --no-sync pytest
```
