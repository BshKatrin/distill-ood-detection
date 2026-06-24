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
uv run --project envs/gpu --no-sync python -m distill_ood_detection.cli train-student --config configs/students/baseline/cifar_100/resnet18/linear.yaml
uv run --project envs/tests --no-sync pytest
```

Use `--no-sync` only after the environment has already been synced. If the
environment does not exist yet, `uv run --no-sync` can create an empty `.venv`
and then skip dependency installation.

## Environment variables

When the project downloads teacher checkpoints from Hugging Face Hub, it
automatically loads a repository-root `.env` file if one exists.

Use `HF_TOKEN` for private or gated Hub access, and keep the real token only in
the local `.env` file. Commit examples such as `.env.example`, not the secret.

Set `DISTILL_OOD_DATA_DIR` to override every config's `dataset.data_dir`.
This is intended for shared dataset roots on clusters, for example:

```bash
DISTILL_OOD_DATA_DIR=/path/to/shared/datasets
```

When this variable is set, resolved run configs record the expanded dataset
directory for traceability.

For a GPU-cluster training run from the repository root:

```bash
uv sync --project envs/gpu --locked
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" uv run --project envs/gpu --no-sync python -m distill_ood_detection.cli train-student --config configs/students/perturbation/embedding/clipping/cifar_10/resnet18/linear_layer3_clip_constant.yaml
```

For submitted jobs, keep the sync step in a setup phase or at the start of the
job, then use `--no-sync` for the actual training commands. This keeps repeated
commands from spending allocation time resolving dependencies.

## Adding an environment

Create a new subdirectory under `envs/` with a focused `pyproject.toml`.
Prefer explicit dependencies over shared indirection so each experiment context
is reproducible on its own.
