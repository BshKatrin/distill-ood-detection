# OOD Scores

An OOD Score is defined per sample. It quantifies whether the sample is more ID-like or more OOD-like.

In this project, use the following convention:

- Higher OOD Score = more ID-like.
- Lower OOD Score = more OOD-like.

Some raw metrics naturally increase for OOD-like samples. For those metrics, multiply the raw value by `-1` so the final OOD Score follows the project convention.

The `sign` field defines how to convert a raw metric value into the final OOD Score:

- `sign: +1`: Use the raw metric value directly.
- `sign: -1`: Multiply the raw metric value by `-1`.

## Scores

- MSP (Maximum Softmax Probability): Maximum softmax probability of the
  **teacher** model from standard raw-image inference. MSP is the baseline OOD
  Score and does not depend on the student model or OOD distillation strategy.
  In different OOD strategy reports, the Teacher MSP row must still use the raw-pixel
  teacher probability artifacts from the matching baseline run.
  - `sign: +1`

- Energy: Teacher-only energy baseline from
  [Energy-based Out-of-distribution Detection](https://arxiv.org/pdf/2010.03759).
  The paper defines free energy as `-T * logsumexp(logits / T)`, where lower
  values are more ID-like. This project stores the sign-adjusted score
  `T * logsumexp(logits / T)` so that higher values are more ID-like.
  - `sign: +1`

- KL (teacher || student): KL divergence from the teacher probability distribution to the student probability distribution.
  - `sign: -1`

- Max probability difference: Maximum teacher probability minus maximum student probability.
  - `sign: +1`

- Absolute max probability difference: Absolute value of the maximum teacher probability minus the maximum student probability.
  - `sign: -1`

- Centered-logit L2 distance: L2 distance between mean-centered teacher logits and mean-centered student logits.
  - `sign: -1`

- Energy gap: Teacher energy minus student energy, using the same sign-adjusted
  energy definition as above.
  - `sign: +1`

- Absolute energy gap: Absolute value of the teacher-student energy gap. This
  is a mismatch score, so it is negated before computing OOD metrics.
  - `sign: -1`

- Student MSP : Maximum softmax probability of the
  **student** model (similar to teacher's baseline MSP OOD score). For perturbation OOD strategies, the MSP should be averaged.
  - `sign: +1`

- Student energy : Student-only energy. Defined as `T * logsumexp(logits / T)` (similar to teacher's baseline energy OOD score)
  - `sign: +1`

- Feature Denoising PCA reconstruction error: Hidden-component reconstruction error in
  whitened PCA space from PCA Masked Reconstruction. The raw error increases
  when a sample is less predictable from ID teacher-feature structure, so it is
  negated before computing OOD metrics.
  - `sign: -1`

- Feature Denoising spatial reconstruction error: Hidden-location reconstruction error in
  raw teacher feature-map space from spatial Feature Masked Reconstruction.
  - `sign: -1`

- Feature Denoising channel reconstruction error: Hidden-channel reconstruction error in
  raw teacher feature-map space from channel Feature Masked Reconstruction.
  - `sign: -1`

- Feature Denoising spatial token prediction error: Target-token prediction error in raw
  teacher feature-map space from Spatial Token Prediction.
  - `sign: -1`

- Feature Denoising pixel embedding prediction error: Clean pooled teacher-embedding
  prediction error from Pixel-Masked Embedding Prediction.
  - `sign: -1`

- Feature Denoising pixel multilayer prediction error: Clean pooled layer3, pooled layer4,
  and centered-logit prediction error from Pixel-Masked Multilayer Prediction.
  - `sign: -1`

- Feature Denoising pixel multilayer L234 prediction error: Clean pooled layer2, layer3,
  layer4, and centered-logit prediction error from Pixel-Masked Multilayer L234
  Prediction.
  - `sign: -1`

- Activation-subspace insignificant reconstruction error: Per-sample MSE
  between reconstructed and target insignificant SVD coordinates.
  - `sign: -1`

- Activation-subspace insignificant relative reconstruction error: Per-sample
  `L2(reconstruction - target) / max(L2(target), eps)`.
  - `sign: -1`

- Activation-subspace insignificant cosine similarity: Per-sample cosine
  similarity between reconstructed and target insignificant SVD coordinates.
  - `sign: +1`

## Implementation

- OOD Score functions are implemented in [ood_scores.py](../../src/distill_ood_detection/evaluation/ood_scores.py).
- Aggregate OOD detection metrics, such as ROC-AUC and FPR@95 TPR, are implemented in [ood_metrics.py](../../src/distill_ood_detection/evaluation/ood_metrics.py).
