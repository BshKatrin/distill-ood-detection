# Activation-subspace workflow

Read the [method reference](../strategies/activation_subspace.md) for the
classifier-SVD decomposition, objectives, score definitions, and invariants.
Results are preserved in the [experiment record](../../experiments/activation_subspace/resnet18_cifar10_cifar100.md).

## Environment and commands

Sync `envs/gpu` with `uv` and put `src/` on `PYTHONPATH`. From the repository root:

```bash
uv sync --project envs/gpu
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
uv run --project envs/gpu --no-sync python -m distill_ood_detection.cli train-student \
  --config configs/students/activation_subspace/decisive/cifar_100/resnet18/linear.yaml
uv run --project envs/gpu --no-sync python -m distill_ood_detection.cli export-activation-subspace-inference \
  --config configs/students/activation_subspace/decisive/cifar_100/resnet18/linear.yaml \
  --teacher-output-dir runs/teachers/cifar_100/resnet18/activation_subspace_embeddings
uv run --project envs/gpu --no-sync python -m distill_ood_detection.cli export-activation-subspace-scores \
  --config configs/students/activation_subspace/decisive/cifar_100/resnet18/linear.yaml \
  --teacher-embedding-dir runs/teachers/cifar_100/resnet18/activation_subspace_embeddings
```

Repeat `--config` for same-teacher variants when exporting inference or scores.

## Shared training settings

Current configs use:

- 50 epochs;
- batch size 256;
- AdamW;
- learning rate `1e-3`;
- weight decay `1e-4`;
- seed 42;
- best checkpoint selected by minimum ID validation loss;
- `mlflow.enabled: false`; config loading rejects enabled MLflow.

Linear-logit validation/test artifacts also record classification accuracy.
Autoencoder metrics record reconstruction loss only.

## Configs

```text
configs/students/activation_subspace/decisive/...
configs/students/activation_subspace/insignificant/...
```

For the GPU cluster, read `docs/hpc/README.md` and the private local HPC
configuration first. `slurm_scripts/run_configs.sbatch` accepts multiple
configs. Use `MODE=train`; generic inference is not valid for this strategy.

## Training artifacts

Every config has a distinct `run_dir`. Its important files are

```text
<run_dir>/
  resolved_config.json
  teacher_metrics.json
  activation_subspace.pt
  summary.json
  <decisive-or-insignificant>/
    best_student.pt
    latest_student.pt
    history.json
    metrics.json
```

`activation_subspace.pt` contains:

- `right_basis`: complete `V^T` basis;
- `singular_values`;
- `decisive_dimension` and `insignificant_dimension`;
- `norm_gaps`: the norm-balance objective for every candidate split;
- component, fit-sample count, classifier-weight shape, and selection metadata.

Do not mix a checkpoint with another run's SVD artifact, even when dimensions
match.

## Raw ID/OOD inference

Activation-subspace students cannot use the generic probability inference
path because they consume SVD coordinates rather than images or flattened raw
features. Use

```text
export-activation-subspace-inference
```

and pass all same-teacher configs by repeating `--config`. The exporter:

1. loads one frozen teacher per ID dataset;
2. exports clean pooled layer4 embeddings once for ID test and all configured
   OOD datasets;
3. loads each run's own SVD basis and best/latest checkpoint;
4. projects embeddings into the configured component;
5. exports raw student outputs and provenance manifests.

Shared teacher artifacts:

```text
runs/teachers/<id_dataset>/resnet18/activation_subspace_embeddings/
  manifest.json
  student_inference_manifest.json
  <dataset>/embeddings.pt
```

Each `embeddings.pt` contains `embeddings`, `labels`, and metadata. Student
artifacts are

```text
<run_dir>/activation_subspace_inference/
  manifest.json
  <dataset>/student_best.pt
```

Projected-logit files contain `logits`, `probabilities`, `labels`, and
metadata. Both decisive and insignificant coordinate-autoencoder files contain
`reconstructed_coordinates`, `labels`, and metadata. Metadata records
`component` and `target` and links the exact checkpoint, SVD file, and teacher
embedding used.

The reusable cluster entrypoint is
`slurm_scripts/export_activation_subspace_inference.sbatch`. It groups configs
by CIFAR ID dataset and evaluates both teachers in parallel on two GPUs.

## Score artifacts

Score artifacts are

```text
<run_dir>/activation_subspace_scores/
  manifest.json
  <dataset>/student_best.pt
```

Each tensor file contains `raw_metrics`, sign-adjusted `ood_scores`, `labels`,
and provenance metadata. The manifest computes ROC-AUC and FPR@95 for every
ID/OOD pair and arithmetic macro averages over the three OOD datasets. ID is
the positive class.

Teacher-level group manifests live beside the shared embeddings as
`score_manifest.json`.

## Implementation map

- SVD and `k` selection:
  [`evaluation/activation_subspaces.py`](../../src/distill_ood_detection/evaluation/activation_subspaces.py)
- Fixed embedding collection, targets, losses, training, and SVD persistence:
  [`distillation/activation_subspace.py`](../../src/distill_ood_detection/distillation/activation_subspace.py)
- Linear and autoencoder student definitions:
  [`models/student.py`](../../src/distill_ood_detection/models/student.py)
- Strategy dispatch and training orchestration:
  [`experiments/train_student.py`](../../src/distill_ood_detection/experiments/train_student.py)
- Shared distillation loss, including cross-entropy alpha/temperature:
  [`distillation/losses.py`](../../src/distill_ood_detection/distillation/losses.py)
- ID/OOD raw inference:
  [`experiments/export_activation_subspace_inference.py`](../../src/distill_ood_detection/experiments/export_activation_subspace_inference.py)
- Score and aggregate metric export:
  [`experiments/export_activation_subspace_scores.py`](../../src/distill_ood_detection/experiments/export_activation_subspace_scores.py)
- Config parsing and validation:
  [`config.py`](../../src/distill_ood_detection/config.py)
- CLI commands: [`cli.py`](../../src/distill_ood_detection/cli.py)
- Tests:
  `tests/test_activation_subspaces.py`,
  `tests/test_activation_subspace_training.py`,
  `tests/test_activation_subspace_inference.py`, and
  `tests/test_activation_subspace_scores.py`.
