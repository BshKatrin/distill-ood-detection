# Strategies

This directory contains different distillation (student-training) strategies.

- [Baseline](baseline.md): Baseline strategy.
- [Perturbation](perturbation.md): Perturbation-based strategy, including
  [embedding-space](embedding/README.md) and
  [pixel-space](pixel/README.md) perturbations.
- [Feature Denoising](feature_denoising.md): representation-prediction strategy,
  including [PCA Masked Reconstruction](feature_denoising/pca_masking.md) and
  [Feature Masked Reconstruction](feature_denoising/feature_masking.md).

## Implementation Links

- Baseline PyTorch student training is implemented in [train_student.py](../../src/distill_ood_detection/experiments/train_student.py).
- Baseline random-forest student training is implemented in [train_tree_student.py](../../src/distill_ood_detection/experiments/train_tree_student.py).
- Perturbation sampling is implemented in [perturbation.py](../../src/distill_ood_detection/distillation/perturbation.py).
- Perturbation training is implemented in the PyTorch and random-forest training entrypoints.
- Feature Denoising training is implemented in [train_student.py](../../src/distill_ood_detection/experiments/train_student.py).
