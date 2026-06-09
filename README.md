# distill-ood-detection

Research code for distilling Hugging Face CIFAR ResNet-18 teachers into simple
student models for OOD-detection experiments.

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
uv run distill-ood train-student --config configs/distill_linear_cifar10.yaml --method cross_entropy
```

Train the three-hidden-layer MLP student:

```bash
uv run distill-ood train-student --config configs/distill_mlp_cifar10.yaml
```

Train an MLP student on frozen teacher features from a hidden layer instead of raw pixels:

```bash
uv run distill-ood train-student --config configs/distill_feature_linear_layer3_cifar10.yaml
```

The baseline config trains a linear student with two objectives:

- `cross_entropy`: `alpha * temperature^2 * Loss_soft + (1 - alpha) * Loss_hard`,
  where `Loss_soft` is temperature-scaled cross-entropy against teacher
  probabilities and `Loss_hard` is standard cross-entropy against true class
  labels.
- `mse_logits`: MSE between centered student and teacher logits.

The baseline config keeps shared loop settings under `training.defaults` and
method-specific settings under `training.methods`. In practice,
`training.methods.cross_entropy` sets `temperature` and `alpha`, while
`mse_logits` uses an empty method block.

MLflow logging is enabled in the YAML config. Each run logs one parent
experiment run and one nested run per distillation method. Per epoch, the
student run logs distillation loss, validation accuracy, and validation KL
divergence `KL(teacher || student)`.

## Feature Students

PyTorch students can consume frozen teacher features by setting
`student.feature_layer` in the YAML config. In that case `student.input_shape`
must match the selected teacher layer output shape. For the CIFAR ResNet-18
teacher, useful feature layers include:

- `layer2`: `[128, 16, 16]`
- `layer3`: `[256, 8, 8]`
- `layer4`: `[512, 4, 4]`
- `avgpool`: `[512, 1, 1]`

The config `configs/distill_feature_mlp_layer3_cifar10.yaml` uses `layer3`.
Training still distills against the teacher logits, but the student receives
the configured teacher features as input.

## Tree Students

Train sklearn random-forest students from teacher outputs:

```bash
uv run distill-ood train-tree-student --config configs/distill_random_forest_cifar10.yaml
```

The tree config supports one mode:

- `logits`: fit a multi-output random-forest regressor to centered teacher logits.

Teacher inference on the deterministic CIFAR-10 training split is saved under
`runs/<experiment_name>/teacher_inference/cifar10_train/teacher.pt`.

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Probability Inference

After training, save teacher and student logits/probabilities for the CIFAR-10
validation split and configured OOD datasets:

```bash
uv run distill-ood infer-probabilities --config configs/distill_linear_cifar10.yaml
```

For feature students, use the matching feature-student config:

```bash
uv run distill-ood infer-probabilities --config configs/distill_feature_linear_layer3_cifar10.yaml
```

By default this uses each student's best checkpoint. Use `--checkpoint latest`
or `--checkpoint both` to infer from other saved checkpoints. Add
`--include-train` or `--include-validation` to include the deterministic ID
training or validation split. Artifacts are written under
`runs/<experiment_name>/probabilities/`, with one `.pt` file per dataset/model
and a `manifest.json` for notebook discovery.

The baseline config evaluates three OOD datasets during inference:
`MNIST`, `SVHN`, and Hugging Face `uoft-cs/cifar100` on the `test` split.

For random-forest students, use the tree config. This saves teacher outputs and
one student artifact for CIFAR-10 test and configured OOD test splits:

```bash
uv run distill-ood infer-probabilities --config configs/distill_random_forest_cifar10.yaml
```

Use `--tree-mode logits` to run only the enabled random-forest mode explicitly.

## CIFAR-100 ID Experiments

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
```
