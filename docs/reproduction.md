# Regeneration from a fresh clone

The intended recovery workflow is to clone the repository and rerun experiments
from committed source, configs, and dependency locks. Generated datasets,
checkpoints, inference tensors, score exports, caches, plots, and compiler files
stay local and must not be added to Git. An archive of the old checkout is
optional; it is not an input to these workflows.

The final report source bundle and the delivered report/presentation PDFs are
the explicitly requested committed deliverables. Existing historical result
snapshots remain in Git as the numerical research record.

## Set up and verify

```bash
git clone git@github.com:BshKatrin/distill-ood-detection.git
cd distill-ood-detection
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
uv sync --project envs/tests --locked
uv run --project envs/tests --no-sync pytest -q

uv sync --project envs/dashboard --locked
uv run --project envs/dashboard --no-sync pytest -q \
  tests/test_channel_cluster_audit_dashboard.py \
  tests/test_global_channel_cluster_distributions.py
```

Use `envs/gpu` for training/inference and `envs/notebooks` for reporting, syncing
each with `uv sync --project <environment> --locked` before using `--no-sync`.
See [environments](envs.md) for supported platforms. The NMF integration test
requires a fitted hierarchy under `runs/` and skips in a clone until it is built.

Recreate private configuration from `.env.example` and, when using a cluster,
`docs/hpc/hpc.template.md`. Neither private configuration nor the old checkout
is required as a source of experiment code.

## Rebuild experiment artifacts

Choose the experiment in the [catalogue](../experiments/README.md), then follow
its config and workflow in dependency order:

| Experiment family | Regeneration steps and source |
| --- | --- |
| Baseline and perturbation students | Run `train-student`, then `infer-probabilities`, then the metric exporter. The [root README](../README.md#reproduce-a-baseline) contains a complete baseline command sequence. |
| Feature Denoising | Build any configured channel hierarchy first; train the configured student, export inference, and generate its scores/tables. Follow the [method catalogue](strategies/feature_denoising/README.md) and [channel grouping](strategies/feature_denoising/channel_grouping/README.md). |
| Activation-subspace students | Train students and their SVD decomposition, then export shared teacher embeddings, student inference, and scores. Follow the [execution workflow](workflows/activation_subspace.md). |
| k-NN embedding distances | Run `export-embedding-distances --config <config>` for each configuration in `RUNS` in [the report generator](../reports/scripts/build_knn_ood_score_report.py), then run that generator. It computes the local JSON summary from regenerated distance tensors. |
| OpenOOD CIFAR | Retrain the selected student variants, prepare benchmark datasets/manifests, then run the [evaluation workflow](evaluation/openood/cifar.md). The evaluation submission script requires trained checkpoints; it does not train missing variants. |
| OpenOOD ImageNet | Acquire the ID images and teacher checkpoints, prepare the benchmark assets, set local paths in config copies, and run the [ImageNet protocol](evaluation/openood/imagenet.md). |
| Tables, plots, and notebooks | After inference/score inputs exist, run the appropriate [reporting tools](reports/README.md). Rendered outputs and caches are disposable. |

For example, after regenerating all ten k-NN settings:

```bash
uv run --project envs/notebooks --no-sync python reports/scripts/build_knn_ood_score_report.py
```

It writes the ignored `reports/outputs/json/knn_faiss_ood_scores.json` and updates
the experiment Markdown. Review the narrative diff before committing it; keep
the numeric export local.

The standard CIFAR/MNIST/SVHN loaders download missing data. Hugging Face
teacher configs download their weights. Internet access, storage, and suitable
compute are external requirements. ImageNet data and locally referenced
teacher checkpoints must be obtained separately; cloning does not supply them.

## Deliverables and historical limits

Rebuild the final report with `make -C reports final-report`, using a compatible
TeX installation. Its tracked chapter sources, table source, references, and
logo are self-contained. The source uses `\today`, so a later build can differ
from the delivered PDF. See [build instructions](reports/recap.md).

The presentation PDF is the delivered export; its editable Google Slides source
is not committed. The six historical recap folders were already ignored and
their sources are not available in a clone. Their dated Makefile targets require
those sources. Neither the slides' editable source nor those recaps can be
regenerated from the currently committed files.

Rerunning the implemented experiments is different from reproducing every
historical number exactly. Some teacher configs use the mutable Hugging Face
revision `main`, and the catalogue does not identify a complete immutable
resolved config/checkpoint for every historical row. GPU kernels and package
platforms can also affect numerical results. Keep rerun configs, teacher
revisions, seeds, and provenance in the generated run directory; label new
results separately and preserve the historical metric conventions.

Before removing a checkout, confirm that source changes are pushed, review
untracked files for any required source or configuration, and verify a fresh
clone. Deleting generated artifacts means accepting the time and external
inputs needed to rerun them; an artifact backup is not required for regeneration.
