# Strategies

This directory contains different distillation (student-training) strategies.

- [Baseline](baseline.md): Baseline strategy.
- [Perturbation](perturbation.md): Perturbation-based strategy, including
  [embedding-space](embedding/README.md) and
  [pixel-space](pixel/README.md) perturbations.

## Implementation Links

- Baseline PyTorch student training is implemented in [train_student.py](../../src/distill_ood_detection/experiments/train_student.py).
- Baseline random-forest student training is implemented in [train_tree_student.py](../../src/distill_ood_detection/experiments/train_tree_student.py).
- Perturbation sampling is implemented in [perturbation.py](../../src/distill_ood_detection/distillation/perturbation.py).
- Perturbation training is implemented in the PyTorch and random-forest training entrypoints.
