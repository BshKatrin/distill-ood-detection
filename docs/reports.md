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

On the GPU cluster, run the same exporter through SLURM with
`slurm_scripts/export_metrics_table.sbatch`. This job should not allocate a GPU;
it loads saved probability artifacts on CPU and writes the generated LaTeX files
under `reports/outputs/latex/`.

```bash
sbatch --mem=32G slurm_scripts/export_metrics_table.sbatch
```

## Validation metrics table

The validation metrics table is generated from saved method metrics under
`runs/<experiment_name>/<method>/metrics.json` and experiment metadata in
`configs/`. Each ID-dataset cell is formatted as
`best_validation_accuracy/best_validation_distillation_loss`, using the same
best-validation-loss checkpoint selection that is used for student probability
inference.

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" uv run --project envs/notebooks --no-sync python reports/scripts/export_validation_metrics_table.py
```

By default, the exporter reads `configs/baseline/` and `configs/perturbation/`
and writes `metrics_validation_baseline.tex` and
`metrics_validation_perturbation.tex` when matching method metrics are
available. Pass config paths to restrict the output, for example:

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" uv run --project envs/notebooks --no-sync python reports/scripts/export_validation_metrics_table.py configs/baseline/cifar_10 configs/perturbation/cifar_10
```

Baseline validation tables include `Student`, `Features`, and `Training`
columns. Perturbation validation tables also include a `Perturbation` column.
Both strategy tables keep `CIFAR-10 (ID)` and `CIFAR-100 (ID)` columns, leaving
cells blank when local metrics are missing. Rows with the same displayed
metadata are merged across ID datasets, matching the OOD metrics table layout.

On the GPU cluster, run the validation metrics exporter through SLURM with
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
