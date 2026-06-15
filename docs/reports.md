# Reports

Report scripts live in `reports/scripts/`. Shared report-only helpers live in
`reports/lib/`, outside the package source tree.

Generated report files are split by artifact type:

- `reports/outputs/latex/`: generated LaTeX source files. These are tracked.
- `reports/outputs/pdf/`: rendered PDFs. These are not tracked.
- `reports/outputs/notebooks/`: generated analysis notebooks. These are not tracked.
- `reports/build/`: compiler scratch files such as `.aux`, `.log`, `.fls`, and
  `.fdb_latexmk`. These are not tracked.

## Metrics table

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
pixels` when absent). Perturbation tables also include a `Perturbation` column
from `strategy.perturbation.clipping_mode`. In every strategy table, the Teacher
MSP row uses raw-image teacher probability artifacts from baseline runs.

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
