# Archive and restore

Git preserves the code, configs, curated experiment records, final report source
and PDF, presentation PDF, documentation, and dependency locks. It does not
preserve every file in a working checkout.

## Before removing a checkout

Confirm the branch is pushed and compare its commit with the remote. Preserve
local-only research files in a verified archive outside the checkout:

| Local material | Restoration |
| --- | --- |
| `runs/` | Checkpoints, resolved configs, histories, raw activations/probabilities, and score artifacts require a separate backup or a verified cluster copy. A clone cannot recover them. |
| Historical `reports/recaps/` folders | Sources, PDFs, CSV/Parquet/HTML evidence remain local/ignored. Preserve them separately. |
| `reports/final-report/archive/` | The separate legacy full-report export remains local/ignored. |
| `reports/outputs/json/`, plots, notebooks, and PDFs | Working exports are ignored; preserve any evidence needed beyond the curated Git snapshots. |
| `papers/`, `tmp/`, and legacy `mlflow.db` | Local reference copies, temporary scripts/material, and historical tracking data require a separate archive if retained. |
| `.env`, `docs/hpc/hpc.local.md` | Private local configuration must stay outside Git. Keep its archive private. |
| Untracked notes | Review `git ls-files --others --exclude-standard`; include any retained notes in the archive. |
| `.venv`, `envs/*/.venv`, Python/test caches | Rebuild from tracked `uv.lock` files; these need not be copied. |

An archive of the checkout excluding virtual environments and caches retains
the local research record as well as the pushed source snapshot. Verify every
retained file against the original before removing the original checkout.
Keep an archive manifest and restore instructions beside the archive. A local
archive protects against deleting the checkout, but not loss of the machine;
use separate storage when machine-loss protection is needed.

## Fresh clone

```bash
git clone git@github.com:BshKatrin/distill-ood-detection.git
cd distill-ood-detection
uv sync --project envs/tests --locked
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" uv run --project envs/tests --no-sync pytest -q
```

The tests use synthetic/local fixtures. The NMF audit integration check requires
fitted hierarchy artifacts under `runs/` and skips when they are absent.
Dashboard test modules use the dedicated environment:

```bash
uv sync --project envs/dashboard --locked
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" uv run --project envs/dashboard --no-sync \
  pytest -q tests/test_channel_cluster_audit_dashboard.py tests/test_global_channel_cluster_distributions.py
```

The final deliverables can be opened immediately after cloning; no build is
needed. Recompiling LaTeX requires a compatible compiler, and the final source
uses `\\today`. Preserve the committed PDF as the delivered version.

## Restore research artifacts

Restore only the folders needed for the intended analysis, keeping their run
hierarchy and checkpoint/provenance files together. For an archive containing
the checkout under `repository/`, for example:

```bash
research_archive=/path/to/archive
rsync -a "$research_archive/repository/runs/" runs/
rsync -a "$research_archive/repository/reports/recaps/" reports/recaps/
```

Restore private configuration separately, or recreate it from `.env.example`
and `docs/hpc/hpc.template.md`. The checked-in ImageNet configs retain the
original cluster paths; use local config copies and the
[portable protocol instructions](evaluation/openood/imagenet.md) on another host.
Training and full benchmark reproduction additionally require datasets and
teacher weights; dependency locks do not supply those external artifacts.

Use [the experiment catalogue](../experiments/README.md) to choose the exact
config, score definition, metric convention, and checkpoint for a result.
