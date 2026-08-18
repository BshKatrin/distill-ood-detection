# OOD Metric Reports

The OOD metric exporter computes ROC-AUC and FPR@95 from saved probability
artifacts under student `run_dir` paths. It uses experiment configs for report
metadata and writes one LaTeX report per strategy.

## Group-balanced macro metrics

When a report aggregates results across Near-OOD and Far-OOD datasets, compute
the mean inside each group first and then combine the two group means:

\[
M_\lambda(m)
= \lambda\frac{1}{|\mathcal N|}\sum_{d\in\mathcal N}m_d
+ (1-\lambda)\frac{1}{|\mathcal F|}\sum_{d\in\mathcal F}m_d,
\qquad \lambda=0.5,
\]

where `m` is either ROC-AUC or FPR@95, `N` is the set of Near-OOD datasets,
and `F` is the set of Far-OOD datasets. Thus Near-OOD and Far-OOD each
contribute half of the macro, regardless of how many datasets are present in
either group. Each Near-OOD dataset has weight `1 / (2 * |N|)` and each
Far-OOD dataset has weight `1 / (2 * |F|)`.

With the current evaluation suite, the opposite CIFAR dataset is the only
Near-OOD dataset and MNIST and SVHN are Far-OOD. The macro therefore assigns
weights `1/2`, `1/4`, and `1/4`, respectively. Apply this aggregation
independently to ROC-AUC and FPR@95. If a report uses the scalar compromise for
selection, define it from the group-balanced metrics:

\[
C = \frac{1}{2}\left(M_{0.5}(\mathrm{ROC\mbox{-}AUC})
+ 1 - M_{0.5}(\mathrm{FPR@95})\right).
\]

Do not average all OOD datasets directly: that would make the relative
importance of Near-OOD and Far-OOD depend on how many datasets happen to be
included in each group.

## Strategy tables

Run the exporter with its default config directories:

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" uv run --project envs/notebooks --no-sync python reports/scripts/export_metrics_table.py
```

By default, it reads `configs/students/baseline/` and
`configs/students/perturbation/` and writes
`metrics_baseline.tex` and `metrics_perturbation.tex` under
`reports/outputs/latex/` when matching run artifacts exist.

Pass config files or directories to restrict the report:

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" uv run --project envs/notebooks --no-sync \
  python reports/scripts/export_metrics_table.py \
  configs/students/baseline/cifar_10 \
  configs/students/perturbation/embedding
```

Baseline tables include a `Features` column derived from
`student.feature_layer`; a missing layer is displayed as `Raw Pixels`.
Perturbation tables also include the configured perturbation. Rows with the
same displayed metadata are merged across ID datasets so matching CIFAR-10 and
CIFAR-100 experiments fill the same row block.

The Teacher MSP row always uses raw-image teacher probability artifacts from
baseline runs, including in perturbation reports. OOD Score display names are
report-specific pretty names.

## Metric cache

Computed metrics are cached at `reports/outputs/cache/ood_metrics.json`. A
cached value is reused when its source artifact paths, sizes, and modification
times match. Use `--cache <path>` to select another cache file.

## Numeric metrics for explicit runs

Use `--run-names` to export numeric metrics for run paths relative to `runs/`:

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" uv run --project envs/notebooks --no-sync \
  python reports/scripts/export_metrics_table.py \
  --run-names students/baseline/cifar_10/resnet18/random_forest_n50 \
              students/baseline/cifar_10/resnet18/random_forest_n200 \
  --json-output reports/outputs/json/random_forest_n_estimators.json
```

Without `--json-output`, this mode writes
`reports/outputs/json/ood_metrics.json`. It discovers artifacts directly under
`probabilities/unperturbed/` and `probabilities/perturbed/`; it does not depend
on a probability manifest that may represent only the latest partial inference
run. Legacy artifacts directly under `probabilities/` are treated as
`unperturbed`.

Dataset metadata comes from `resolved_config.json`. The output contains run
metadata under `probability_modes` and a `probability_mode` field on every
metric record.

Use `--scores` with `--run-names` to limit expensive calculations:

```bash
--scores max_probability_difference absolute_max_probability_difference student_teacher_kl_divergence
```

## Teacher baselines

Use `--teacher-run-names` to export MSP and energy metrics from raw-image
teacher artifacts:

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" uv run --project envs/notebooks --no-sync \
  python reports/scripts/export_metrics_table.py \
  --teacher-run-names teachers/cifar_10/resnet18 teachers/cifar_10/resnet50 \
  --json-output reports/outputs/json/teacher_baselines.json
```

Each record contains the OOD Score, ID and OOD datasets, ROC-AUC, and FPR@95.

## Perturbation comparison tables

After exporting the selected CIFAR-10 and CIFAR-100 perturbation runs, test
metrics, and teacher baselines to JSON, generate separate aggressive-clipping,
non-aggressive-clipping, Monte-Carlo-dropout, and PCA tables:

```bash
uv run --project envs/notebooks --no-sync python \
  reports/scripts/export_cifar10_perturbation_tables.py
```

The tables report OOD cells as `ROC-AUC/FPR@95` and include test accuracy and
test distillation loss once per objective and inference mode. They include
ResNet-18 and ResNet-50 MSP and energy baselines for both ID datasets.

Clipping tables separate clipping level, teacher target, and inference mode.
The dropout table separates dropout level and inference mode. The PCA table
omits those inapplicable columns. Every table retains the training objective
because each OOD Score is available for multiple objectives. Within an OOD
dataset, a complete metric pair is bold when it contains the highest student
ROC-AUC or lowest student FPR@95.

## Random-forest estimator sweep

The random-forest sweep table consumes the fixed numeric export
`reports/outputs/json/random_forest_sweep.json` plus the supplemental
`reports/outputs/json/random_forest_sweep_extra.json` file when present.
Generate its LaTeX tables with:

```bash
uv run --project envs/notebooks --no-sync python reports/scripts/export_random_forest_sweep_table.py
```

Generate the ROC-AUC and FPR@95 sweep plots with:

```bash
uv run --project envs/notebooks --no-sync python reports/scripts/plot_random_forest_sweep.py
```

The script writes the full sweep table and a compact Student MSP/Energy table
under `reports/outputs/latex/`. The compact table consumes
`reports/outputs/json/student_msp_energy_cifar10.json` and
`reports/outputs/json/student_msp_energy_cifar100.json` when present. The plot
script also consumes `reports/outputs/json/random_forest_sweep_extra.json` when
present. Plot files are written under `reports/outputs/plots/`. These scripts
select layer-4 random-forest runs from the JSON and use their built-in estimator
ordering and teacher baselines.

## SLURM

On the GPU cluster, submit the general exporter through its CPU-only SLURM job:

```bash
sbatch --mem=32G slurm_scripts/export_metrics_table.sbatch
```

The job reads saved probability artifacts and writes LaTeX files under
`reports/outputs/latex/`; it should not allocate a GPU. Read the
[HPC documentation](../hpc/README.md) before using cluster commands.
