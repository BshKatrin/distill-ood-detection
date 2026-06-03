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

## Project vocabulary
- ID (In-Distribution): Data drawn from the same distribution as the training data, or from the distribution the model is expected to encounter during deployment.

- OOD (Out-of-Distribution): Data that does not follow the training-data distribution and differs significantly from the examples seen during training.

## Directory responsibilities

- datasets/: dataset loading and transforms.
- models/: neural network definitions only.
- distillation/: distillation losses and training logic.
- evaluation/: metrics, OOD scores, plots.
- experiments/: executable entrypoints.
- configs/: experiment configuration files.
- docs/: research notes and experiment logs.
- runs/: training artifacts, including .json files containing training histories
  and .pt files containing saved model checkpoints

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