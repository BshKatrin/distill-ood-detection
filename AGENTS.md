# AGENTS.md

Purpose: research codebase for Out-of-Distribution detection using model distillation.

## Project philosophy

This repository prioritizes:

1. Reproducibility.
2. Simplicity.
3. Experiment traceability.
4. Small, composable modules.

Prefer explicit code over abstractions.

Avoid introducing frameworks or patterns unless they reduce duplication across multiple experiments.

## Directory responsibilities

- `src/distill_ood_detection/datasets/`: dataset loading and transforms.
- `src/distill_ood_detection/models/`: neural network definitions only.
- `src/distill_ood_detection/distillation/`: distillation losses and training logic.
- `src/distill_ood_detection/evaluation/`: metrics, OOD Scores, and plots.
- `src/distill_ood_detection/experiments/`: executable entrypoints.
- `configs/`: experiment configuration files. See `docs/configs.md` for the config directory structure.
- `docs/`: project documentation, strategy notes, objectives, and OOD Score definitions.
- `runs/`: training artifacts, including `.json` files containing training histories and `.pt` files containing saved model checkpoints.

### Project documentation (docs)

- Project-specific terminology and vocabulary is defined in `docs/vocabulary.md`. When you encounter an unknown abbreviation (e.g. ID, OOD) consult the relevant vocabulary documentation. Use the project terminology exactly as defined there.
- Before making non-trivial changes, read `docs/index.md` to find the relevant project documentation.
- If HPC or GPU cluster access is requested, read `docs/hpc/README.md` and follow its local-configuration rules.
- If a documented section is marked `Not implemented` and you implement it, remove the `Not implemented` status in the same change.

Do not duplicate long explanations here. Add or update detailed documentation in `docs/`, then link it from `docs/index.md`.

## Coding guidelines

- Prefer typed Python.
- Add docstrings to public functions.
- Avoid global state.
- Use deterministic seeds when feasible.
- Separate training, evaluation and plotting logic.

## Python environment and dependency management

All Python dependencies must use `uv`.

- `uv` is the only supported dependency manager
- Do NOT use pip, venv, or poetry directly

### Install dependencies

```bash
uv add <package>
```

## Notebook guidelines

- Keep text minimal.
- Prefer short section titles over explanatory paragraphs.
- Focus on creating clear plots.
- For plots, prefer Seaborn for static statistical visuals.
- Use Plotly when interactivity is useful or necessary.
- Avoid long textual analysis unless explicitly requested.
