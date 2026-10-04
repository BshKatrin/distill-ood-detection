# Strategies

This directory contains different distillation (student-training) strategies.

- [Baseline](baseline.md): Baseline strategy.
- [Perturbation](perturbation.md): Perturbation-based strategy, including
  [embedding-space](embedding/README.md) and
  [pixel-space](pixel/README.md) perturbations.
- [Feature Denoising](feature_denoising/README.md): representation-prediction strategy,
  including [PCA Masked Reconstruction](feature_denoising/pca_masking.md) and
  [Feature Masked Reconstruction](feature_denoising/feature_masking.md).
- [Activation-Subspace Students](activation_subspace.md): self-contained
  ActSub-inspired variant using decisive logit distillation plus decisive and
  insignificant classical autoencoding over classifier-SVD coordinates. This
  is a project-specific extension, not a reproduction of the paper's combined
  detector.
- [Disjoint Subspace Ensembles](subspace_ensemble.md): 16 linear students
  trained on ordered or seeded-partitioned channel or whitened GAP-PCA
  subspaces, scored with predictive entropy and BALD.

## Implementation Links

- Baseline PyTorch student training is implemented in [train_student.py](../../src/distill_ood_detection/experiments/train_student.py).
- Baseline random-forest student training is implemented in [train_tree_student.py](../../src/distill_ood_detection/experiments/train_tree_student.py).
- Perturbation sampling is implemented in [perturbation.py](../../src/distill_ood_detection/distillation/perturbation.py).
- Perturbation training is implemented in the PyTorch and random-forest training entrypoints.
- Feature Denoising training is implemented in [train_student.py](../../src/distill_ood_detection/experiments/train_student.py).
- Activation-subspace training is implemented in
  [activation_subspace.py](../../src/distill_ood_detection/distillation/activation_subspace.py)
  and the PyTorch training entrypoint.
- Random-subspace selection, PCA, training, and inference are implemented in
  [subspace_ensemble.py](../../src/distill_ood_detection/distillation/subspace_ensemble.py).
