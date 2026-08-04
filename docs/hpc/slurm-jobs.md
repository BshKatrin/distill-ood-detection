# SLURM jobs

Use [slurm_scripts/run_configs.sbatch](../../slurm_scripts/run_configs.sbatch)
for ordinary config-driven GPU jobs instead of creating one sbatch file per
experiment group.

The script accepts config paths as positional arguments:

```bash
sbatch \
  --job-name=cifar10_perturb \
  --gres=gpu:2 \
  slurm_scripts/run_configs.sbatch \
  configs/students/perturbation/embedding/clipping/cifar_10/resnet18/linear_layer3_clip_constant.yaml \
  configs/students/perturbation/embedding/clipping/cifar_10/resnet18/linear_layer3_clip_channel.yaml \
  configs/students/perturbation/embedding/clipping/cifar_10/resnet18/linear_layer3_clip_spatial.yaml
```

By default, each config runs a train-then-infer pipeline. Configs whose path
contains `random_forest` use `train-tree-student`; all other configs use
`train-student`.

Configs below `configs/students/subspace_ensemble/` use the specialized
ensemble inference and score exporters after training. Their `train-infer`
pipeline therefore finishes with predictive-entropy and BALD OOD metrics
rather than the generic single-student probability export.

## Modes

Set `MODE` to choose the pipeline:

- `MODE=train-infer`: train the student, then infer probabilities. This is the
  default.
- `MODE=train`: train only.
- `MODE=infer`: infer probabilities only.

Examples:

```bash
sbatch --gres=gpu:1 --export=ALL,MODE=infer \
  slurm_scripts/run_configs.sbatch \
  configs/students/baseline/cifar_100/resnet18/linear.yaml
```

```bash
sbatch --gres=gpu:1 --export=ALL,MODE=train \
  slurm_scripts/run_configs.sbatch \
  configs/students/baseline/cifar_10/resnet18/feature_random_forest_layer3.yaml
```

## Parallelism

`PARALLEL=auto` runs up to one config pipeline per allocated GPU. This is the
default. For a single-command job that should avoid parallelization, request one
GPU or set `PARALLEL=1`:

```bash
sbatch --gres=gpu:1 --export=ALL,MODE=infer,PARALLEL=1 \
  slurm_scripts/run_configs.sbatch \
  configs/students/perturbation/embedding/clipping/cifar_10/resnet18/linear_layer3_clip_constant.yaml
```

## Config list files

For long config lists, put one config path per line in a text file and submit
with `CONFIG_LIST`:

```bash
sbatch --gres=gpu:2 --export=ALL,CONFIG_LIST=configs_to_run.txt \
  slurm_scripts/run_configs.sbatch
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
- `APPLY_PERTURBATION=1`: add `infer-probabilities --apply-perturbation`.
- `SKIP_UV_SYNC=1`: reuse an already-synchronized GPU environment. Use this
  only when the environment has been preflighted for the current checkout.

## Teacher-only probability exports

Use
[slurm_scripts/export_teacher_probabilities.sbatch](../../slurm_scripts/export_teacher_probabilities.sbatch)
to export deterministic teacher logit/probability artifacts from configs under
`configs/teachers/`.

The script accepts one or more config paths as positional arguments and runs
them sequentially in one GPU job:

```bash
sbatch \
  --gres=gpu:1 \
  slurm_scripts/export_teacher_probabilities.sbatch \
  configs/teachers/cifar_10/resnet50.yaml \
  configs/teachers/cifar_100/resnet50.yaml
```

For longer config lists, set `CONFIG_LIST` to a newline-delimited file, using
the same format supported by `run_configs.sbatch`.

Teacher logit/probability artifacts are written under:

```text
<run_dir>/teacher_probabilities/
```

## Layerwise embedding distances

Use `slurm_scripts/export_embedding_distances.sbatch` with one config under
`configs/embedding_distances/`. The job extracts GAP embeddings from all four
ResNet stages, selects exact top-k ID training neighbors in bounded blocks, and
computes classifier-based OOD Scores from their mean probabilities and logits:

```bash
sbatch --exclude=daft \
  slurm_scripts/export_embedding_distances.sbatch \
  configs/embedding_distances/cifar_10/resnet50.yaml
```

Artifacts are written below
`runs/embedding_distances/<id_dataset>/<teacher_architecture>/`.

## Report tables

Use [slurm_scripts/export_metrics_table.sbatch](../../slurm_scripts/export_metrics_table.sbatch)
to compute OOD metrics tables on the cluster from existing
student `run_dir` probability artifacts.

This is a CPU and disk-I/O job. Do not request `--gres=gpu:*` for this script.
Increase `--mem` if the job is killed while loading probability artifacts.

```bash
sbatch \
  --cpus-per-task=4 \
  --mem=32G \
  --time=01:00:00 \
  slurm_scripts/export_metrics_table.sbatch
```

Pass config files or directories after the script to restrict the export:

```bash
sbatch \
  --mem=32G \
  slurm_scripts/export_metrics_table.sbatch \
  configs/students/baseline/cifar_10 \
  configs/students/perturbation/embedding
```

The script maps these environment variables to exporter flags:

- `CACHE_PATH`: `export_metrics_table.py --cache`.
- `OUTPUT_PATH`: `export_metrics_table.py --output`.
- `OUTPUT_DIR`: `export_metrics_table.py --output-dir`.

Use [slurm_scripts/export_test_metrics_table.sbatch](../../slurm_scripts/export_test_metrics_table.sbatch)
to compute test metrics tables on the cluster from existing
`<run_dir>/<method>/metrics.json` artifacts.

This is also a CPU job and should not request a GPU.

```bash
sbatch slurm_scripts/export_test_metrics_table.sbatch
```

Pass config files or directories after the script to restrict the export:

```bash
sbatch \
  slurm_scripts/export_test_metrics_table.sbatch \
  configs/students/baseline/cifar_10 \
  configs/students/perturbation/embedding
```

The script maps these environment variables to exporter flags:

- `OUTPUT_PATH`: `export_test_metrics_table.py --output`.
- `OUTPUT_DIR`: `export_test_metrics_table.py --output-dir`.
- `JSON_OUTPUT`: `export_test_metrics_table.py --json-output`.

## Feature Denoising activation subspaces

Submit the exact per-draw classifier-subspace analysis:

```bash
sbatch --exclude=daft \
  slurm_scripts/export_feature_denoising_subspace_errors.sbatch \
  <layer4-config>
```

The job is inference-only. It makes one clean teacher pass over ID training
data to choose `k`, then writes compact distribution
artifacts below `<run_dir>/feature_denoising_subspace_errors/`.

## PixMix mixing set

Before the first PixMix experiment, if the data is not already present, download and extract the official external mixing set through SLURM:

```bash
sbatch --exclude=daft slurm_scripts/prepare_pixmix_data.sbatch
```

The script respects `DISTILL_OOD_DATA_DIR` and installs the archive under
`<data-dir>/pixmix/fractals_and_fvis/`.
