# Baseline Strategy

This strategy follows the standard teacher–student distillation setup without any input perturbation or synthetic data generation.

## Teacher Model

Use a relatively strong CNN teacher model: ResNet-18.

Train the teacher on the ID classification task. During distillation, the teacher provides the supervision signal for the student.

## Student Models

The following student architectures are evaluated:

- Linear model (no hidden layers)
- Multi-Layer Perceptron (MLP, 3 hidden layers)
- Random Forest

## Student Inputs

Two input representations are explored:

### Raw Pixels

The student receives the original image pixels as input.

### Teacher Embeddings

The student receives feature representations extracted from an intermediate teacher layer, for example:

- Penultimate layer embedding
- Antepenultimate layer embedding

## Distillation Objective

The student is trained to reproduce the teacher outputs using one of the objectives:

- `mse_logits`
- `cross_entropy`

See [Objectives](../objectives/README.md) for details.

## Implementation

- PyTorch student training is implemented in [train_student.py](../../src/distill_ood_detection/experiments/train_student.py).
- Random-forest student training is implemented in [train_tree_student.py](../../src/distill_ood_detection/experiments/train_tree_student.py).
