# PixMix Pixel Perturbation Students: ResNet-50

Status: training completed for CIFAR-10 and CIFAR-100 ID; OOD inference in
progress.

## Setup

- Teacher: ResNet-50.
- Teacher feature: globally averaged `layer4` embedding.
- Student: linear classifier with a 2,048-dimensional input.
- Pixel perturbation: PixMix.
- Embedding pool: global average pooling.
- Mixing set: official `fractals_and_fvis` images.
- Mixing iterations: `4`.
- Beta: `3.0`.
- Augmentation severity: `3.0`.
- Operations: all PixMix augmentation operations.
- Working image size: `32`.
- Evaluation draws for stochastic PixMix inference: `16`.
- Objectives: cross-entropy distillation and centered-logit MSE.
- Teacher targets: perturbed image and clean image.
- Checkpoint selection: best validation-loss checkpoint.
- ID datasets: CIFAR-10 and CIFAR-100 test.
- OOD datasets:
  - CIFAR-10 ID: MNIST, SVHN, and CIFAR-100 test;
  - CIFAR-100 ID: MNIST, SVHN, and CIFAR-10 test.

Unlike affine pixel augmentation, PixMix does not expose a compact
perturbation vector to the student. The student receives only the globally
averaged teacher embedding of the input view.

Run directories:

```text
runs/students/perturbation/pixel/pixmix/cifar_10/resnet50/linear_layer4_pixmix_avg
runs/students/perturbation/pixel/pixmix/cifar_10/resnet50/linear_layer4_pixmix_avg_clean_target
runs/students/perturbation/pixel/pixmix/cifar_100/resnet50/linear_layer4_pixmix_avg
runs/students/perturbation/pixel/pixmix/cifar_100/resnet50/linear_layer4_pixmix_avg_clean_target
```

## Training Results

| ID dataset | Teacher target | Objective | Test accuracy | Test distillation loss |
|---|---|---|---:|---:|
| CIFAR-10 | Perturbed | Cross-entropy | 0.9394 | 20.2392 |
| CIFAR-10 | Perturbed | Logit MSE | 0.9394 | 0.000410 |
| CIFAR-10 | Clean | Cross-entropy | 0.9438 | 20.6118 |
| CIFAR-10 | Clean | Logit MSE | 0.9455 | 4.13774 |
| CIFAR-100 | Perturbed | Cross-entropy | 0.7972 | 56.7686 |
| CIFAR-100 | Perturbed | Logit MSE | 0.7927 | 0.006324 |
| CIFAR-100 | Clean | Cross-entropy | 0.8020 | 56.9203 |
| CIFAR-100 | Clean | Logit MSE | 0.8007 | 1.01663 |

Loss values are objective-dependent and should not be compared directly
between cross-entropy and logit MSE.

## OOD Scores

All reported scores follow the project convention that higher values are more
ID-like. The experiment evaluates:

- teacher MSP;
- teacher energy;
- negative KL divergence from teacher to student;
- teacher minus student maximum probability;
- negative absolute maximum-probability difference;
- negative centered-logit L2 distance;
- teacher minus student energy;
- negative absolute energy gap;
- student MSP;
- student energy.

Teacher MSP and energy use standard raw-image teacher inference. Student
scores are evaluated in two modes:

- `unperturbed`: one clean input view;
- `perturbed`: average over 16 independently sampled PixMix views.

## OOD Results

Pending completion of clean-image inference job `406351` and stochastic
PixMix inference job `406352`.

Tables will report `ROC-AUC / FPR@95` for every OOD dataset and the macro
average across the three OOD datasets.

## Artifacts

Probability artifacts:

```text
<run_dir>/probabilities/unperturbed/
<run_dir>/probabilities/perturbed/
```

Training job: `406157`.

