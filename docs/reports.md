# Reports

Report scripts live in `reports/scripts/`. Shared report-only helpers live in
`reports/lib/`, outside the package source tree.

Generated report files are split by artifact type:

- `reports/outputs/latex/`: generated LaTeX source files. These are tracked.
- `reports/outputs/cache/`: generated metric cache files. These are not tracked.
- `reports/outputs/pdf/`: rendered PDFs. These are not tracked.
- `reports/outputs/notebooks/`: generated analysis notebooks. These are not tracked.
- `reports/build/`: compiler scratch files such as `.aux`, `.log`, `.fls`, and
  `.fdb_latexmk`. These are not tracked.

## OOD metrics table

The OOD metrics table is generated from saved probability artifacts under
`runs/*/probabilities/` and experiment metadata in `configs/`. The export script
accepts config files or config directories, computes ROC-AUC and FPR@95 from the
saved teacher and student artifacts, applies report-specific pretty names for
OOD Scores, and writes one LaTeX file per OOD strategy under
`reports/outputs/latex/`.

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" uv run --project envs/notebooks --no-sync python reports/scripts/export_metrics_table.py
```

By default, the exporter reads `configs/baseline/` and `configs/perturbation/`
and writes `metrics_baseline.tex` and `metrics_perturbation.tex` when matching
run artifacts are available. Pass config paths to restrict the output, for
example:

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" uv run --project envs/notebooks --no-sync python reports/scripts/export_metrics_table.py configs/baseline/cifar_10 configs/perturbation/cifar_10
```

Baseline tables include a `Features` column from `student.feature_layer` (`Raw
Pixels` when absent). Perturbation tables also include a `Perturbation` column
from `strategy.perturbation.clipping_mode`. In every strategy table, the Teacher
MSP row uses raw-image teacher probability artifacts from baseline runs.
Rows with the same displayed metadata are merged across ID datasets, so matching
CIFAR-10 and CIFAR-100 runs fill the same row block instead of repeating the
metadata with blank cells.

Computed OOD metrics are cached in
`reports/outputs/cache/ood_metrics.json` by default. The cache is reused when
the source artifact paths, sizes, and modification times match. Use `--cache` to
choose a different cache file.

To export numeric metrics for an explicit list of run names, use
`--run-names`. This mode reads each run's probability manifest directly, so the
original experiment config does not need to be present. It writes numeric
ROC-AUC and FPR@95 values to JSON and includes `n_estimators` when it is present
in the saved random-forest method metrics:

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" uv run --project envs/notebooks --no-sync \
  python reports/scripts/export_metrics_table.py \
  --run-names random_forest_student_resnet18_cifar10_n50 \
              random_forest_student_resnet18_cifar10_n200 \
  --json-output reports/outputs/json/random_forest_n_estimators.json
```

Without `--json-output`, this mode writes
`reports/outputs/json/ood_metrics.json`.

On the GPU cluster, run the same exporter through SLURM with
`slurm_scripts/export_metrics_table.sbatch`. This job should not allocate a GPU;
it loads saved probability artifacts on CPU and writes the generated LaTeX files
under `reports/outputs/latex/`.

```bash
sbatch --mem=32G slurm_scripts/export_metrics_table.sbatch
```

## Test metrics table

The test metrics table is generated from saved method metrics under
`runs/<experiment_name>/<method>/metrics.json` and experiment metadata in
`configs/`. Each ID-dataset cell is formatted as
`test_accuracy/test_distillation_loss`. Both values are computed in one pass over
the test split after loading the checkpoint selected by the lowest validation
distillation loss. A hardcoded Teacher row reports the published test accuracies
from the Hugging Face model cards for
[`resnet18_cifar10`](https://huggingface.co/edadaltocg/resnet18_cifar10) and
[`resnet18_cifar100`](https://huggingface.co/edadaltocg/resnet18_cifar100).
Its cells use `accuracy/--` because the model cards do not publish a teacher
distillation loss.

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" uv run --project envs/notebooks --no-sync python reports/scripts/export_validation_metrics_table.py
```

By default, the exporter reads `configs/baseline/` and `configs/perturbation/`
and writes `metrics_test_baseline.tex` and
`metrics_test_perturbation.tex` when matching method metrics are
available. Pass config paths to restrict the output, for example:

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" uv run --project envs/notebooks --no-sync python reports/scripts/export_validation_metrics_table.py configs/baseline/cifar_10 configs/perturbation/cifar_10
```

Baseline test tables include `Student`, `Features`, and `Training`
columns. Perturbation test tables also include a `Perturbation` column.
Both strategy tables keep `CIFAR-10 (ID)` and `CIFAR-100 (ID)` columns, leaving
cells blank when local metrics are missing. Rows with the same displayed
metadata are merged across ID datasets, matching the OOD metrics table layout.

On the GPU cluster, run the test metrics exporter through SLURM with
`slurm_scripts/export_validation_metrics_table.sbatch`. This job should not
allocate a GPU; it reads saved method metrics JSON files and writes generated
LaTeX files under `reports/outputs/latex/`.

```bash
sbatch slurm_scripts/export_validation_metrics_table.sbatch
```

Render the generated LaTeX sources to one PDF per strategy under
`reports/outputs/pdf/` with:

```bash
uv run --project envs/notebooks --no-sync python reports/scripts/render_metrics.py
```

Use the `envs/notebooks` environment because it includes the reporting
dependencies used by the export script, including `torch`, `numpy`, and
`scikit-learn`. Set `PYTHONPATH` for the export command so the environment can
import the repository package from `src/` without installing it.

## Probability artifact notebooks

Generated report notebooks are created from
`reports/templates/notebooks/` and written to `reports/outputs/notebooks/`.
The generator selects the standard or perturbation template from the experiment
name.

```bash
uv run --project envs/notebooks --no-sync python reports/scripts/create_probability_notebook.py linear_student_resnet18_cifar10
```

Use `--number` to choose a stable notebook prefix and `--overwrite` to replace an
existing generated notebook.

## Teacher activation notebooks

Teacher activation notebooks are generated from
`reports/templates/notebooks/teacher_activation_bars.ipynb` and populated from a
teacher activation config under `configs/teachers/`. The generator requires the
matching activation manifest under
`runs/<experiment_name>/teacher_activations/manifest.json`.

```bash
uv run --project envs/notebooks --no-sync python reports/scripts/create_teacher_activation_notebook.py configs/teachers/resnet18_cifar10_layers.yaml
```

Use `--number` to choose a stable notebook prefix and `--overwrite` to replace an
existing generated notebook. The same command works for any teacher activation
config, for example `configs/teachers/resnet18_cifar100_layers.yaml`, once its
matching activation artifacts have been exported.

## PCA explained-variance notebook

Export the leading PCA explained-variance statistics from the complete ID
training-split activation artifact before opening the notebook. The export uses
deterministic randomized PCA so it remains practical on a local CPU machine:

```bash
uv run --project envs/notebooks --no-sync python reports/scripts/export_pca_explained_variance.py \
  runs/teacher_resnet18_cifar10/teacher_activations/cifar10_train/layer4.pt
```

The default export estimates 512 components. Increase the range when a target
variance threshold has not been reached, for example with
`--max-components 1024`. The command writes the statistics JSON beside the
activation artifact and generates a numbered notebook under
`reports/outputs/notebooks/` from
`reports/templates/notebooks/pca_explained_variance.ipynb`. Use `--number` to
choose a stable notebook prefix and `--overwrite` to replace an existing
generated notebook.
