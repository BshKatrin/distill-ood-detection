# Distillation Objective Functions

Implemented objective functions are defined in [losses.py](../../src/distill_ood_detection/distillation/losses.py).

## `mse_logits`

Mean Squared Error (MSE) between teacher and student logits after centering each logit vector by subtracting its mean.

For each sample:

- Center teacher logits: `z_teacher <- z_teacher - mean(z_teacher)`
- Center student logits: `z_student <- z_student - mean(z_student)`
- Compute MSE between the centered logits.

## `cross_entropy`

The student is trained using a weighted combination of two cross-entropy objectives:

1. **Soft-target loss**
   - Cross-entropy between student predictions and teacher soft targets.
   - Both teacher and student probabilities are computed using temperature `T`.

2. **Hard-target loss**
   - Standard cross-entropy with ground-truth labels.
   - Student logits are used directly, equivalent to temperature `1`.

The final loss is `L = alpha * T^2 * L_soft + (1 - alpha) * L_hard`

where:

- `L_soft` is the cross-entropy with teacher soft targets.
- `L_hard` is the cross-entropy with ground-truth labels.
- `T` is the distillation temperature.
- The `T²` factor compensates for the `1/T²` scaling of gradients from the soft-target loss.
- `alpha` controls the weight of the teacher signal.

## `kl_divergence`

Kullback–Leibler (KL) divergence between the teacher and student output probability distributions.

Minimize:

`D_KL(P_teacher || P_student)`

where:

- P_teacher is the teacher's output probability distribution (treated as the target distribution).
- P_student is the student's output probability distribution.

## References

- Hinton, Vinyals, Dean (2015), "Distilling the Knowledge in a Neural Network". URL: https://arxiv.org/abs/1503.02531
