# Baseline Strategy

This strategy follows the standard teacher–student distillation setup without any input perturbation or synthetic data generation.

## Teacher Model

The teacher is a relatively strong CNN model: ResNet-18

The teacher is trained on the ID classification task and provides the supervision signal for the student.

## Student Models

The following student architectures are evaluated:

- Linear model (no hidden layers)
- Multi-Layer Perceptron (MLP, 3 hidden layers)
- Random Forest

## Student Inputs

2 input representations are explored:

### Raw Pixels

The student receives the original image pixels as input.

### Teacher Embeddings

The student receives feature representations extracted from an intermediate teacher layer, for example:

- Penultimate layer embedding
- Antepenultimate layer embedding

## Distillation Objective

The student is trained to reproduce the teacher outputs using one of the objectives:

- `mse_logits`
- `kl_divergence`

See `docs/objectives/` for details.
