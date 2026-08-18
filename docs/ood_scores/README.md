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

- KL (teacher \ student): KL divergence from the teacher probability distribution to the student probability distribution.
  When logits are available, reports compute the equivalent log-softmax form
  to avoid zeros caused by floating-point softmax underflow.
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

## k-NN output aggregation

The layerwise k-NN variant uses embedding distance only to select the exact
top-k ID training examples. It does not use the mean top-k distance as an OOD
Score. For each query and embedding space (raw or ID-standardized), it averages
the selected neighbors' teacher probabilities and teacher logits separately.
The query teacher output is then compared with these neighbor means as follows:

- Probability mean: KL (teacher || neighbor mean), max probability difference,
  absolute max probability difference, and Student MSP.
- Logit mean: centered-logit L2 distance, energy gap, absolute energy gap, and
  Student energy.
- Neighbor distributions: predictive entropy of the mean neighbor probability
  distribution and BALD,
  `H(mean_j p_j) - mean_j H(p_j)`. Both raw quantities are negated because
  higher uncertainty or disagreement is more OOD-like.
- Query-only baselines: teacher MSP and teacher energy. These are exported with
  every layer for a complete, directly comparable score set.

Here the selected ID neighbors act as a local ensemble. Predictive entropy is
high when their mean prediction is uncertain. BALD is high when individual
neighbors are confident but disagree. This BALD is a local-neighborhood
disagreement score, not posterior uncertainty from independently trained
models. With `k = 1`, BALD is exactly zero.

Every configured post-GAP layer performs its own neighbor selection and writes
its own score artifact, so `layers` can contain any subset of `layer1` through
`layer4`.

Artifacts retain neighbor indices, distances, mean probabilities, mean logits,
raw metrics, and sign-adjusted `ood_scores`. The sign-adjusted values follow the
project convention that higher values are more ID-like.

- Feature Denoising PCA reconstruction error: Hidden-component reconstruction error in
  whitened PCA space from PCA Masked Reconstruction. The raw error increases
  when a sample is less predictable from ID teacher-feature structure, so it is
  negated before computing OOD metrics.
  - `sign: -1`

- Feature Denoising spatial reconstruction error: Hidden-location reconstruction error in
  raw teacher feature-map space from spatial Feature Masked Reconstruction.
  - `sign: -1`

- Feature Denoising spatial block residual reconstruction error: Hidden-block
  reconstruction error after adding the residual CNN correction to the
  mean-filled teacher feature map.
  - `sign: -1`

- Feature Denoising channel reconstruction error: Hidden-channel reconstruction error in
  raw teacher feature-map space from channel Feature Masked Reconstruction.
  - `sign: -1`

- Feature Denoising channel residual reconstruction error: Hidden-channel
  reconstruction error after adding the residual CNN correction to the
  zero-filled teacher feature map.
  - `sign: -1`

- Feature Denoising NMF concept residual reconstruction error: Full-map error
  after the residual CNN reconstructs the clean `S P` projection from a
  concept-masked `S' P` projection. The raw error is negated for the OOD Score.
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

- Ensemble predictive entropy: Shannon entropy of the mean probability
  distribution across the 16 subspace students,
  `H(mean_s p_s)`, using natural logarithms. High raw entropy is OOD-like.
  - `sign: -1`

- Ensemble BALD: Disagreement across the 16 subspace students,
  `H(mean_s p_s) - mean_s H(p_s)`. High raw BALD is OOD-like.
  - `sign: -1`

## Implementation

- OOD Score functions are implemented in [ood_scores.py](../../src/distill_ood_detection/evaluation/ood_scores.py).
- Aggregate OOD detection metrics, such as ROC-AUC and FPR@95 TPR, are implemented in [ood_metrics.py](../../src/distill_ood_detection/evaluation/ood_metrics.py).
