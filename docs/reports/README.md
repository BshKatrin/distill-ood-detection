# Reports

Reporting code lives outside the package source tree under `reports/`:

- `reports/scripts/`: executable report commands
- `reports/lib/`: shared report-only Python helpers
- `reports/templates/`: notebook and report templates

Use the `envs/notebooks` environment for report commands. It contains the
reporting dependencies, including PyTorch, NumPy, and scikit-learn. Commands
that import `distill_ood_detection` directly from `src/` also set `PYTHONPATH`.

## Workflows

- [OOD metric reports](ood-metrics.md): export ROC-AUC and FPR@95 tables or
  numeric JSON from saved probability artifacts.
- [OOD score tradeoff plots](ood-score-tradeoff-plots.md): plot ROC-AUC versus
  FPR@95 across explicit hyperparameter sweeps.
- [Test metric reports](test-metrics.md): export test accuracy and distillation
  loss tables or numeric JSON.
- [Analysis notebooks](notebooks.md): generate probability, teacher-activation,
  and PCA explained-variance notebooks.

## Generated artifacts

| Path | Contents | Tracked |
| --- | --- | --- |
| `reports/outputs/latex/` | Generated LaTeX reports | Yes |
| `reports/outputs/json/` | Numeric metric exports | Yes |
| `reports/outputs/cache/` | Reusable metric caches | Yes |
| `reports/outputs/plots/` | Generated plot files | Yes |
| `reports/outputs/pdf/` | Rendered reports | No |
| `reports/outputs/notebooks/` | Generated analysis notebooks | No |
| `reports/build/` | LaTeX compiler scratch files | No |

## Render LaTeX reports

Render all `reports/outputs/latex/metrics_*.tex` files to
`reports/outputs/pdf/`:

```bash
uv run --project envs/notebooks --no-sync python reports/scripts/render_metrics.py
```

Pass one or more LaTeX paths to render only those reports. Rendering requires
either `latexmk` or `pdflatex` on `PATH`; intermediate compiler files are
written under `reports/build/latex/`.
