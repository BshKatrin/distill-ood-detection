# Reporting workflows

The [deliverable catalogue](../../reports/README.md) lists finished reports and
slides. Curated experimental conclusions, result JSON, and supporting figures
live alongside their experiment records under `experiments/`.

Reporting tools remain under `reports/scripts/`, with shared report-only helpers
in `reports/lib/` and notebook templates in `reports/templates/`. Use the synced
`envs/notebooks` environment; commands importing the package put `src/` on
`PYTHONPATH`. See [environments](../envs.md).

## Workflows

- [OOD metric reports](ood-metrics.md): ROC-AUC and FPR@95 exports from saved probabilities.
- [Tradeoff plots](ood-score-tradeoff-plots.md): ROC-AUC versus FPR@95 across sweeps.
- [Test metrics](test-metrics.md): accuracy and distillation-loss exports.
- [Analysis notebooks](notebooks.md): probability, activation, and PCA notebooks.
- [Recaps and final-report builds](recap.md): dated progress reports and deliverable builds.

Read [metric conventions](../evaluation/metrics.md) before comparing FPR@95
across historical reports and OpenOOD evaluations.

## Artifact ownership and Git policy

| Location | Contents | Git policy |
| --- | --- | --- |
| `reports/final-report/` | Final source bundle, logo, and `main.pdf` | Tracked; compiler scratch ignored |
| `reports/presentations/internship/` | Final slides PDF | Tracked |
| `reports/recaps/YYYY-MM-DD/` | Progress sources and delivered `main.pdf` | New recaps trackable; six historical folders remain ignored |
| `reports/final-report/archive/` | Separate legacy PDF export | Local/ignored |
| `experiments/` | Curated narratives, result JSON, and supporting figures | Tracked when selected for the research record |
| `reports/outputs/latex/` | Existing generated LaTeX tables | Tracked |
| `reports/outputs/cache/` | Reusable metric caches | Existing two snapshots retained; new cache files ignored |
| `reports/outputs/json/`, `reports/outputs/plots/` | Working numeric/plot exports | Ignored |
| `reports/outputs/pdf/`, `reports/outputs/notebooks/` | Rendered tables and generated notebooks | Ignored |
| `reports/build/` | Compiler/rendering scratch files | Ignored |

An output becomes a curated result by copying the selected JSON or figure into
its corresponding `experiments/<family>/` record and documenting its generating
command, config, checkpoint, score, metric convention, and units. Keep script
default output paths unchanged; do not commit every working export.

## Render metric tables

Render all `reports/outputs/latex/metrics_*.tex` files into
`reports/outputs/pdf/`:

```bash
uv run --project envs/notebooks --no-sync python reports/scripts/render_metrics.py
```

Pass explicit LaTeX paths to render selected tables. This existing renderer
requires `latexmk` or `pdflatex`; scratch files go under `reports/build/latex/`.
Its outputs are working reports rather than the final internship deliverable.
