# OpenOOD CIFAR evaluation

The `openood_v1_5` evaluation reuses this project's trained students and
Hugging Face teacher checkpoints while replacing the evaluation datasets with
OpenOOD's fixed CIFAR-10 and CIFAR-100 image-list manifests. It does not load
OpenOOD model checkpoints. The original CIFAR preprocessing and normalization
are retained because they are part of the inputs expected by these models.

## Fixed splits

OpenOOD's image lists are used verbatim; no runtime random subsampling is
performed.

| ID dataset | ID test | Near OOD | Far OOD |
| --- | ---: | --- | --- |
| CIFAR-10 | 9,000 | CIFAR-100: 9,000; TIN: 7,793 | MNIST: 70,000; SVHN: 26,032; Textures: 5,640; Places365: 35,195 |
| CIFAR-100 | 9,000 | CIFAR-10: 10,000; TIN: 6,526 | MNIST: 70,000; SVHN: 26,032; Textures: 5,640; Places365: 33,773 |

The evaluator validates every manifest count before loading a model and checks
every exported score vector against the same counts before writing metrics.
In particular, MNIST is the complete 70,000-image OpenOOD set, rather than only
the torchvision test split.

## Metric convention

Saved metrics follow the project's OpenOOD comparison convention (OOD+).
See [aggregate metrics](../metrics.md) for its distinction from historical ID+ reports:

- OOD is the positive class (`1`).
- ID is the negative class (`0`).
- score implementations first produce ID confidence (larger means more ID-like)
  and negate it for OOD-positive ROC and FPR calculations.
- `FPR95` is the ID false-positive rate when OOD true-positive rate is at least
  95%.
- `AUPR-IN` uses ID as positive; `AUPR-OUT` uses OOD as positive.

Each metrics JSON records these choices explicitly. Results are written below
the existing run without replacing older evaluation artifacts:

```text
<run_dir>/evaluations/openood_v1_5/
```

## Selected variants and restartability

The requested evaluation set is defined in
`configs/evaluation/openood_cifar_v1_5/selected_variants.yaml`. Before SLURM
submission, the manifest builder resolves each requested best checkpoint from
the existing training metrics. Missing checkpoints are recorded separately and
are not scheduled or replaced with a different training variant.

The selected Feature Denoising variants include the CIFAR-10 and CIFAR-100
ViT-B/16 `layer12` patch-token residual students. Their OpenOOD score averages
masked-patch reconstruction error over 10 Bernoulli mask draws while retaining
the 224x224 ViT preprocessing from training.

Run the complete preflight, data preparation, and evaluation array with:

```bash
slurm_scripts/submit_openood_cifar_evaluation.sh
```

If compute nodes cannot reach the archive host, use the repository's two-phase
transfer workflow instead:

```bash
nohup slurm_scripts/launch_openood_cifar_after_transfer.sh \
  > openood-cifar-transfer.out 2>&1 </dev/null &
```

This performs only resumable `curl` transfers on the login node. After every
archive is present, it automatically submits the extraction job and dependent
evaluation array. No Python download package is required.

The data preparation covers OpenOOD's packaged CIFAR, TIN, MNIST, SVHN,
Textures, and Places365 archives. The GPU array evaluates at most two variants
concurrently. Existing metrics files are treated as completed tasks, so
rerunning the array is safe; pass `--force` to the single-task CLI only when
artifacts must deliberately be regenerated.
