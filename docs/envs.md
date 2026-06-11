# Environments

Dependency environments live under `envs/`. Each subdirectory is an independent
`uv` project with its own `pyproject.toml` and `.venv`.

The root project keeps package metadata for editable installs and console-script
packaging, but day-to-day work should use one of the focused environments.

## Available environments

- `envs/gpu`: minimal runtime for GPU-cluster jobs.
- `envs/notebooks`: notebook, plotting, and experiment-visualization tools.
- `envs/tests`: test runner and test dependencies.

## Usage

Sync an environment from the repository root:

```bash
uv sync --project envs/gpu
uv sync --project envs/notebooks
uv sync --project envs/tests
```

The environment projects set `tool.uv.package = false`. They do not install the
repository package during sync, which avoids resolving build requirements on the
cluster. Run project code by putting `src/` on `PYTHONPATH`.

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
uv run --project envs/gpu --no-sync python -m distill_ood_detection.cli train-student --config configs/baseline/cifar_100/linear.yaml
uv run --project envs/tests --no-sync pytest
```

Use `--no-sync` inside submitted jobs after the environment has already been
synced. This keeps jobs from spending allocation time resolving dependencies.

## Adding an environment

Create a new subdirectory under `envs/` with a focused `pyproject.toml`.
Prefer explicit dependencies over shared indirection so each experiment context
is reproducible on its own.
