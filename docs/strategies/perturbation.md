# Perturbation Strategy

The perturbation strategy trains a student from modified representations of an
input. A perturbation can be applied either to an intermediate embedding
produced by the teacher or directly to the input pixels.

The teacher is trained on the ID classification task and provides the
supervision signal during distillation. The current experiments use a
ResNet-18 teacher and evaluate the following student architectures:

- Linear model with no hidden layers
- Multi-Layer Perceptron (MLP) with three hidden layers
- Random Forest

## Perturbation levels

- [Embedding-space perturbations](embedding/README.md) modify an intermediate
  teacher embedding.
- [Pixel-space perturbations](pixel/README.md) modify the input image before
  feature extraction.

## Distillation objective

The student is trained to reproduce the selected teacher output using one of
the following objectives:

- `mse_logits`
- `cross_entropy`
- `kl_divergence`

Teacher and student outputs may be logits or probability distributions after
softmax, depending on the objective. See [Objectives](../objectives/README.md)
for the definitions.

## Implementation

- Perturbation sampling is implemented in
  [perturbation.py](../../src/distill_ood_detection/distillation/perturbation.py).
- PyTorch student training is implemented in
  [train_student.py](../../src/distill_ood_detection/experiments/train_student.py).
- Random-forest student training is implemented in
  [train_tree_student.py](../../src/distill_ood_detection/experiments/train_tree_student.py).
- Probability inference is implemented in
  [infer_probabilities.py](../../src/distill_ood_detection/experiments/infer_probabilities.py).
