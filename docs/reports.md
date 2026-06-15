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
`runs/*/probabilities/`. The export script computes ROC-AUC and FPR@95 from the
saved teacher and student artifacts, applies report-specific pretty names for
OOD Scores, and writes `reports/outputs/latex/metrics.tex`.

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" uv run --project envs/notebooks --no-sync python reports/scripts/export_metrics_table.py
```

Render the generated LaTeX source to `reports/outputs/pdf/metrics.pdf` with:

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
