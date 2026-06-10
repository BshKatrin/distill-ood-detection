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

- MSP (Maximum Softmax Probability): Maximum softmax probability of the **teacher** model. MSP is the baseline OOD Score.
  - `sign: +1`

- KL (teacher || student): KL divergence from the teacher probability distribution to the student probability distribution.
  - `sign: -1`

- Max probability difference: Maximum teacher probability minus maximum student probability.
  - `sign: +1`

- Absolute max probability difference: Absolute value of the maximum teacher probability minus the maximum student probability.
  - `sign: -1`

- Centered-logit L2 distance: L2 distance between zero-meaned teacher logits and zero-meaned student logits.
  - `sign: -1`
