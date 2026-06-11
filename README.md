# distill-ood-detection

Research code for distilling Hugging Face CIFAR ResNet-18 teachers into simple
student models for OOD-detection experiments.

## Repository layout

- `src/distill_ood_detection/`: importable Python package.
- `configs/`: reproducible YAML experiment definitions.
- `runs/`: generated metrics, checkpoints, and resolved configs.
- `docs/`: project documentation, strategy notes, objectives, and OOD Score definitions.
- `AGENTS.md`: instructions for AI agents working in this repository.

For project terminology, research strategy documentation, objective definitions,
and OOD Score conventions, start with [`docs/index.md`](docs/index.md).

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

### PyTorch-based students

```bash
uv run distill-ood train-student --config <CONFIG_PATH> [--method <METHOD>]
```

See [`docs/objectives/README.md`](docs/objectives/README.md) for objective definitions.

MLflow logging is enabled in the YAML config. View logged runs with:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Tree Students

Train sklearn random-forest students from teacher outputs:

```text
uv run distill-ood train-tree-student --config <CONFIG_PATH>
```

The tree config supports one mode:

- `logits`: fit a multi-output random-forest regressor to centered teacher logits.

## Probability Inference

After training, save teacher and student logits/probabilities for the ID and OOD datasets test splits :

```text
uv run distill-ood infer-probabilities --config <CONFIG_PATH> [--checkpoint {latest,checkpoint}] [--include-train] [--include-validation]
```

Artifacts are written under
`runs/<experiment_name>/probabilities/`, with one `.pt` file per dataset/model
and a `manifest.json` for notebook discovery.

<!-- ## CIFAR-100 ID Experiments

The CIFAR-100 experiment configs use `edadaltocg/resnet18_cifar100` as teacher,
train 100-class students, and evaluate CIFAR-10, MNIST, and SVHN as OOD
datasets.

```bash
uv run distill-ood train-student --config configs/distill_linear_cifar100.yaml
uv run distill-ood train-student --config configs/distill_mlp_cifar100.yaml
uv run distill-ood train-student --config configs/distill_feature_linear_layer3_cifar100.yaml
uv run distill-ood train-tree-student --config configs/distill_random_forest_cifar100.yaml
```

After training, save best-checkpoint probabilities:

```bash
uv run distill-ood infer-probabilities --config configs/distill_linear_cifar100.yaml
uv run distill-ood infer-probabilities --config configs/distill_mlp_cifar100.yaml
uv run distill-ood infer-probabilities --config configs/distill_feature_linear_layer3_cifar100.yaml
uv run distill-ood infer-probabilities --config configs/distill_random_forest_cifar100.yaml
``` -->
