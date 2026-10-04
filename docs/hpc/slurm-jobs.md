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

Configs below `configs/students/feature_denoising/knn_channel_masking/` are
inference-only. The runner skips training and directly exports exact masked
k-NN reconstruction scores.
The same applies to `knn_channel_group_masking/`, which masks one complete
hierarchy-cut cluster for each example and evaluation draw.

Configs below `configs/students/feature_denoising/channel_group_masking/` run
the normal train-then-score Feature Denoising pipeline. They load the matching
ID training channel hierarchy, sample one complete cluster per training
example, and independently resample one cluster for each inference draw.

## NMF channel-cluster audit

Use `slurm_scripts/submit_channel_cluster_audit.sh` for the specialist audit.
It expands the two dataset-level configs into one 269-task manifest, exports
shared teacher metadata, and submits a restartable one-GPU array with at most
two simultaneous tasks. Successful completion triggers the CPU Parquet summary
job and then the CPU-only Panel server job.

Rerunning the submission script uses deep artifact validation and submits only
missing task indices. To inspect or submit a targeted subset directly:

```bash
PYTHONPATH="$PWD/src" uv run --project envs/gpu --no-sync \
  python -m distill_ood_detection.cli channel-cluster-audit-status \
  --manifest runs/channel_cluster_audit/nmf_latent_cosine/manifest.json \
  --deep
```

When channel hierarchies must be fitted first, submit
`slurm_scripts/launch_channel_cluster_audit_array.sbatch` with an `afterok`
dependency on the grouping jobs. It builds the manifest after those artifacts
exist, enforces `MAX_TASKS`, and submits only incomplete training-and-inference
array elements.

The Panel job defaults to port 5006 and eight hours. It binds to the compute
node's cluster-facing interfaces and restricts WebSocket origins to
`localhost:5006`; use the SSH tunnel shown
in the [audit documentation](../strategies/feature_denoising/channel_grouping/channel_cluster_audit.md#dashboard).

The separate global-student distribution dashboard exports cluster-decomposed
absolute and relative improvements from the existing NMF students, summarizes
Tukey boxplots, and serves on port 5007. Complete commands for launching and
tunneling to both Panel applications are in
[Panel dashboards on SLURM](panel-dashboards.md).

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

## Layer4 embedding distances

Use `slurm_scripts/export_embedding_distances.sbatch` with one config under
`configs/embedding_distances/`. The job extracts post-GAP `layer4` embeddings,
builds one exact raw-embedding GPU FAISS index, and computes
classifier-based OOD Scores from the selected neighbors' mean probabilities
and logits:

```bash
sbatch --exclude=daft \
  slurm_scripts/export_embedding_distances.sbatch \
  configs/embedding_distances/cifar_10/resnet50.yaml
```

Artifacts are written below
`runs/embedding_distances/<id_dataset>/<teacher_architecture>/`.

Configs below `configs/embedding_distances/perturbation/` use the same command.
They preserve post-GAP `layer4` FAISS geometry while applying the configured
affine or sequential clipping corruption to ID references and, when
`query_perturbed: true`, to each query draw. Their artifacts mirror the config
hierarchy below `runs/embedding_distances/perturbation/`.

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

## OpenOOD CIFAR evaluation

Use `slurm_scripts/submit_openood_cifar_evaluation.sh` to evaluate the focused
CIFAR-10/CIFAR-100 variants against the fixed OpenOOD v1.5 manifests. The
submission script first resolves existing checkpoints into a task manifest,
submits a CPU data-preparation job, and then submits a dependent one-GPU array
with at most two simultaneous tasks.

```bash
slurm_scripts/submit_openood_cifar_evaluation.sh
```

The jobs reuse the existing GPU environment and project checkpoints. Set
`SOURCE_DIR` when the evaluation code is staged separately from the project
checkout, and set `OPENOOD_ROOT`, `MANIFEST`, or `MAX_CONCURRENT` to override
their documented defaults. Missing checkpoints remain in the manifest's
`missing` list and are never silently substituted. Protocol details and exact
sample counts are in [OpenOOD CIFAR](../evaluation/openood/cifar.md).

When compute nodes have no outbound access to the OpenOOD archive host, run
`slurm_scripts/launch_openood_cifar_after_transfer.sh` in the background on the
login node. It performs resumable network transfer only, then submits the
normal SLURM extraction and GPU jobs automatically. It does not install a
downloader or perform extraction/model computation on the login node.

## ViT patch-token Feature Denoising

Train the CIFAR-10 and CIFAR-100 `layer12` patch-token residual students in
parallel, then export their standard ID/OOD reconstruction scores:

```bash
sbatch --gres=gpu:2 --job-name=fd_vit_patch_l12 \
  slurm_scripts/run_configs.sbatch \
  configs/students/feature_denoising/patch_token_masking/cifar_10/vit_base_patch16_224/residual_layer12_mask_p020.yaml \
  configs/students/feature_denoising/patch_token_masking/cifar_100/vit_base_patch16_224/residual_layer12_mask_p020.yaml
```

The runner sets Hugging Face offline mode. Both ViT checkpoints must therefore
already be present in the compute environment's Hugging Face cache, as required
by the teacher configs. After training, the normal OpenOOD CIFAR submission
includes both best checkpoints in its selected-variant manifest.

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

## Feature Denoising channel grouping

Build top-10%-activation channel profiles, Pearson correlation matrices, and
average-linkage hierarchies on the GPU cluster:

```bash
sbatch --exclude=daft \
  slurm_scripts/build_channel_groups.sbatch \
  configs/channel_grouping/top_activation_correlation/cifar_10/resnet18.yaml
```

Submit the CIFAR-100 config separately to let SLURM schedule the two ID jobs in
parallel. Each config processes `layer1` through `layer4` in one teacher pass.
The job never writes raw feature maps or per-sample profiles. It writes compact
numeric artifacts, JSON group cuts, an overview PNG, and a labeled vector PDF
below `<run_dir>/channel_groups/`.

The same runner accepts `configs/channel_grouping/nmf_latent_cosine/...`.
Those jobs stream post-ReLU spatial activations through global MiniBatchNMF
models and export each layer's `P` matrix plus overview and labeled
dendrograms. They intentionally omit flat group cuts.

Sync the complete `channel_groups/` directory to the Mac. It is intentionally
small enough to retain the linkage and correlation data for later recutting,
while the remote-rendered dendrograms can be inspected immediately.

## Feature Denoising channel-group masking

After building the grouping artifacts, run the distance-`0.5` ResNet-18
experiments with the ordinary config runner. For example, request two GPUs and
pass the four configs for one ID dataset; the runner executes two pipelines at
a time:

```bash
sbatch --exclude=daft --gres=gpu:2 \
  --job-name=fd_groups_c10 \
  slurm_scripts/run_configs.sbatch \
  configs/students/feature_denoising/channel_group_masking/cifar_10/resnet18/cluster_layer1_d050.yaml \
  configs/students/feature_denoising/channel_group_masking/cifar_10/resnet18/cluster_layer2_d050.yaml \
  configs/students/feature_denoising/channel_group_masking/cifar_10/resnet18/cluster_layer3_d050.yaml \
  configs/students/feature_denoising/channel_group_masking/cifar_10/resnet18/cluster_layer4_d050.yaml
```

Submit CIFAR-100 as a separate job. Each pipeline trains a residual denoiser
and then exports best-checkpoint reconstruction scores for ID and configured
OOD datasets. The configs use 10 independently sampled cluster masks per image
at inference and retain the sampled group IDs in each score artifact.

## Feature Denoising layer composition

After exporting the strong pixel-augmented MLP scores for layer3 and layer4 on
both ID datasets, submit the CPU-only composition report:

```bash
sbatch slurm_scripts/build_feature_denoising_layer_composition_report.sbatch
```

The job standardizes each layer's absolute-improvement score using the matching
ID test split, evaluates the configured beta sweep, and writes the JSON summary,
Markdown experiment report, and ROC-AUC/FPR@95 plots. Do not request a GPU for
this report job.

## PixMix mixing set

Before the first PixMix experiment, if the data is not already present, download and extract the official external mixing set through SLURM:

```bash
sbatch --exclude=daft slurm_scripts/prepare_pixmix_data.sbatch
```

The script respects `DISTILL_OOD_DATA_DIR` and installs the archive under
`<data-dir>/pixmix/fractals_and_fvis/`.
