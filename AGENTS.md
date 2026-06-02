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

- datasets/: dataset loading and transforms.
- models/: neural network definitions only.
- distillation/: distillation losses and training logic.
- evaluation/: metrics, OOD scores, plots.
- experiments/: executable entrypoints.
- configs/: experiment configuration files.
- docs/: research notes and experiment logs.

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
