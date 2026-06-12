# Reports

Report scripts live in `reports/scripts/`. Generated report sources and rendered
outputs live in `reports/output/`.

## Metrics table

The OOD metrics table is generated from saved probability artifacts under
`runs/*/probabilities/`. The export script computes ROC-AUC and FPR@95 from the
saved teacher and student artifacts, applies report-specific pretty names for
OOD Scores, and writes `reports/output/metrics.tex`.

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" uv run --project envs/notebooks --no-sync python reports/scripts/export_metrics_table.py
```

Render the generated LaTeX source to `reports/output/metrics.pdf` with:

```bash
uv run --project envs/notebooks --no-sync python reports/scripts/render_metrics.py
```

Use the `envs/notebooks` environment because it includes the reporting
dependencies used by the export script, including `torch`, `numpy`, and
`scikit-learn`. Set `PYTHONPATH` for the export command so the environment can
import the repository package from `src/` without installing it.
