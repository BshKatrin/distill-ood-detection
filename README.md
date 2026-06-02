# distill-ood-detection

Research code for distilling a Hugging Face CIFAR-10 ResNet-18 teacher
(`edadaltocg/resnet18_cifar10`) into simple student models for OOD-detection
experiments.

## Repository layout

- `src/distill_ood_detection/`: importable Python package.
- `configs/`: reproducible YAML experiment definitions.
- `runs/`: generated metrics, checkpoints, and resolved configs.
- `docs/`: research notes and experiment logs.

## Environment

Dependencies are managed with `uv`.

```bash
uv sync
```

The project uses a `src` layout with setuptools metadata, so notebooks can
import package modules directly:

```python
from distill_ood_detection.config import load_config
```

On macOS, PyTorch resolves from PyPI. On Linux x86_64, `pyproject.toml`
configures `uv` to resolve `torch` and `torchvision` from the official
CUDA 12.8 PyTorch wheel index.

## Training

Run both configured student objectives:

```bash
uv run distill-ood train-student --config configs/distill_linear_cifar10.yaml
```

Run just one method:

```bash
uv run distill-ood train-student \
  --config configs/distill_linear_cifar10.yaml \
  --method cross_entropy_probabilities
```

The baseline config trains a linear student with two objectives:

- `mse`: MSE between student and teacher probabilities.
- `cross_entropy_probabilities`: cross-entropy against teacher probabilities.

MLflow logging is enabled in the YAML config. Each run logs one parent
experiment run and one nested run per distillation method. Per epoch, the
student run logs distillation loss, validation accuracy, and validation KL
divergence from teacher probabilities to student probabilities.

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```
