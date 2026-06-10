# Strategies

This directory contains different distillation (student-training) strategies.

- [Baseline](baseline.md): Baseline strategy.
- [Perturbation](perturbation.md): Perturbation-based strategy. **Status**: Not implemented.

## Implementation Links

- Baseline PyTorch student training is implemented in [train_student.py](../../src/distill_ood_detection/experiments/train_student.py).
- Baseline random-forest student training is implemented in [train_tree_student.py](../../src/distill_ood_detection/experiments/train_tree_student.py).
- Perturbation training does not currently have a dedicated implementation.
