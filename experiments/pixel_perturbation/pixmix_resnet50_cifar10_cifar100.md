# PixMix Pixel Perturbation Students: ResNet-50

Status: completed for CIFAR-10 and CIFAR-100 ID with standard, unperturbed
inference.

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

Teacher MSP and energy use standard raw-image teacher inference. The results
below use one clean input view for the teacher and student. The separately
submitted stochastic inference evaluates averages over 16 independently
sampled PixMix views and is not included here.

## OOD Results

All tables report `ROC-AUC / FPR@95`.

### CIFAR-10 ID: Teacher Baselines

| Score | MNIST | SVHN | CIFAR-100 | Macro |
|---|---:|---:|---:|---:|
| Teacher MSP | 0.897 / 0.622 | 0.891 / 0.680 | 0.873 / 0.628 | 0.887 / 0.643 |
| Teacher energy | 0.932 / 0.408 | 0.911 / 0.517 | 0.868 / 0.517 | 0.904 / 0.481 |

### CIFAR-10 ID: Perturbed Target, Cross-Entropy

| Score | MNIST | SVHN | CIFAR-100 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.325 / 0.766 | 0.251 / 0.803 | 0.356 / 0.758 | 0.311 / 0.776 |
| Absolute max probability difference | 0.895 / 0.631 | 0.929 / 0.494 | 0.875 / 0.621 | 0.900 / 0.582 |
| Teacher-student KL | 0.905 / 0.611 | 0.948 / 0.375 | 0.883 / 0.594 | 0.912 / 0.527 |
| Centered-logit L2 distance | 0.721 / 0.953 | 0.926 / 0.409 | 0.748 / 0.765 | 0.798 / 0.709 |
| Energy gap | 0.419 / 0.935 | 0.127 / 0.996 | 0.349 / 0.960 | 0.298 / 0.964 |
| Absolute energy gap | 0.582 / 0.999 | 0.873 / 0.407 | 0.651 / 0.840 | 0.702 / 0.748 |
| Student MSP | 0.898 / 0.619 | 0.898 / 0.659 | 0.874 / 0.619 | 0.890 / 0.632 |
| Student energy | 0.932 / 0.411 | 0.922 / 0.476 | 0.872 / 0.511 | 0.909 / 0.466 |

### CIFAR-10 ID: Perturbed Target, Logit MSE

| Score | MNIST | SVHN | CIFAR-100 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.599 / 0.689 | 0.770 / 0.585 | 0.542 / 0.709 | 0.637 / 0.661 |
| Absolute max probability difference | 0.894 / 0.608 | 0.904 / 0.675 | 0.864 / 0.638 | 0.887 / 0.640 |
| Teacher-student KL | 0.905 / 0.585 | 0.909 / 0.680 | 0.864 / 0.638 | 0.893 / 0.634 |
| Centered-logit L2 distance | 0.503 / 0.996 | 0.523 / 0.993 | 0.538 / 0.949 | 0.521 / 0.980 |
| Energy gap | 0.336 / 0.999 | 0.577 / 0.899 | 0.486 / 0.959 | 0.466 / 0.952 |
| Absolute energy gap | 0.507 / 0.986 | 0.480 / 0.989 | 0.485 / 0.980 | 0.491 / 0.985 |
| Student MSP | 0.897 / 0.623 | 0.890 / 0.683 | 0.873 / 0.628 | 0.886 / 0.645 |
| Student energy | 0.932 / 0.406 | 0.910 / 0.522 | 0.868 / 0.520 | 0.904 / 0.482 |

### CIFAR-10 ID: Clean Target, Cross-Entropy

| Score | MNIST | SVHN | CIFAR-100 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.377 / 0.768 | 0.285 / 0.772 | 0.436 / 0.724 | 0.366 / 0.755 |
| Absolute max probability difference | 0.866 / 0.764 | 0.948 / 0.369 | 0.876 / 0.625 | 0.897 / 0.586 |
| Teacher-student KL | 0.862 / 0.813 | 0.959 / 0.281 | 0.885 / 0.590 | 0.902 / 0.561 |
| Centered-logit L2 distance | 0.455 / 1.000 | 0.927 / 0.407 | 0.779 / 0.723 | 0.720 / 0.710 |
| Energy gap | 0.399 / 0.996 | 0.125 / 0.997 | 0.352 / 0.950 | 0.292 / 0.981 |
| Absolute energy gap | 0.397 / 1.000 | 0.816 / 0.436 | 0.629 / 0.831 | 0.614 / 0.755 |
| Student MSP | 0.897 / 0.635 | 0.939 / 0.458 | 0.880 / 0.615 | 0.905 / 0.569 |
| Student energy | 0.926 / 0.473 | 0.965 / 0.185 | 0.882 / 0.496 | 0.924 / 0.385 |

### CIFAR-10 ID: Clean Target, Logit MSE

| Score | MNIST | SVHN | CIFAR-100 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.360 / 0.777 | 0.243 / 0.801 | 0.369 / 0.751 | 0.324 / 0.776 |
| Absolute max probability difference | 0.865 / 0.769 | 0.946 / 0.382 | 0.880 / 0.615 | 0.897 / 0.589 |
| Teacher-student KL | 0.859 / 0.806 | 0.958 / 0.282 | 0.887 / 0.576 | 0.901 / 0.555 |
| Centered-logit L2 distance | 0.661 / 1.000 | 0.954 / 0.269 | 0.816 / 0.699 | 0.811 / 0.656 |
| Energy gap | 0.393 / 1.000 | 0.104 / 1.000 | 0.317 / 0.983 | 0.271 / 0.994 |
| Absolute energy gap | 0.552 / 1.000 | 0.888 / 0.435 | 0.666 / 0.836 | 0.702 / 0.757 |
| Student MSP | 0.894 / 0.650 | 0.938 / 0.491 | 0.884 / 0.616 | 0.906 / 0.586 |
| Student energy | 0.926 / 0.469 | 0.967 / 0.179 | 0.891 / 0.484 | 0.928 / 0.378 |

### CIFAR-10 Summary

- Clean-target logit MSE with student energy is strongest overall:
  macro `0.928 / 0.378`.
- It improves over teacher energy (`0.904 / 0.481`) in both aggregate
  ROC-AUC and FPR@95.
- Clean-target cross-entropy student energy is close at `0.924 / 0.385`.
- The strongest teacher-student mismatch score is perturbed-target
  cross-entropy KL at `0.912 / 0.527`.
- Signed energy gap and absolute energy gap are not useful aggregate scores.

### CIFAR-100 ID: Teacher Baselines

| Score | MNIST | SVHN | CIFAR-10 | Macro |
|---|---:|---:|---:|---:|
| Teacher MSP | 0.811 / 0.823 | 0.758 / 0.793 | 0.792 / 0.781 | 0.787 / 0.799 |
| Teacher energy | 0.878 / 0.670 | 0.783 / 0.766 | 0.801 / 0.774 | 0.821 / 0.737 |

### CIFAR-100 ID: Perturbed Target, Cross-Entropy

| Score | CIFAR-10 | MNIST | SVHN | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.375 / 0.874 | 0.320 / 0.923 | 0.271 / 0.897 | 0.322 / 0.898 |
| Absolute max probability difference | 0.729 / 0.903 | 0.733 / 0.980 | 0.809 / 0.719 | 0.757 / 0.867 |
| Teacher-student KL | 0.721 / 0.911 | 0.706 / 0.985 | 0.860 / 0.612 | 0.762 / 0.836 |
| Centered-logit L2 distance | 0.320 / 0.991 | 0.107 / 1.000 | 0.713 / 0.712 | 0.380 / 0.901 |
| Energy gap | 0.752 / 0.806 | 0.848 / 0.682 | 0.378 / 0.965 | 0.659 / 0.818 |
| Absolute energy gap | 0.247 / 0.996 | 0.150 / 1.000 | 0.622 / 0.695 | 0.340 / 0.897 |
| Student MSP | 0.789 / 0.786 | 0.808 / 0.832 | 0.778 / 0.783 | 0.792 / 0.800 |
| Student energy | 0.795 / 0.780 | 0.873 / 0.696 | 0.810 / 0.741 | 0.826 / 0.739 |

### CIFAR-100 ID: Perturbed Target, Logit MSE

| Score | CIFAR-10 | MNIST | SVHN | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.231 / 0.976 | 0.167 / 0.997 | 0.305 / 0.975 | 0.234 / 0.983 |
| Absolute max probability difference | 0.779 / 0.817 | 0.832 / 0.813 | 0.703 / 0.905 | 0.772 / 0.845 |
| Teacher-student KL | 0.791 / 0.805 | 0.833 / 0.897 | 0.681 / 0.922 | 0.768 / 0.875 |
| Centered-logit L2 distance | 0.340 / 0.991 | 0.141 / 1.000 | 0.228 / 0.999 | 0.236 / 0.997 |
| Energy gap | 0.759 / 0.800 | 0.834 / 0.755 | 0.852 / 0.720 | 0.815 / 0.758 |
| Absolute energy gap | 0.241 / 0.994 | 0.166 / 1.000 | 0.148 / 1.000 | 0.185 / 0.998 |
| Student MSP | 0.794 / 0.773 | 0.814 / 0.816 | 0.749 / 0.790 | 0.786 / 0.793 |
| Student energy | 0.802 / 0.778 | 0.881 / 0.669 | 0.771 / 0.779 | 0.818 / 0.742 |

### CIFAR-100 ID: Clean Target, Cross-Entropy

| Score | CIFAR-10 | MNIST | SVHN | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.576 / 0.858 | 0.660 / 0.912 | 0.321 / 0.881 | 0.519 / 0.884 |
| Absolute max probability difference | 0.744 / 0.901 | 0.716 / 0.982 | 0.831 / 0.708 | 0.764 / 0.864 |
| Teacher-student KL | 0.753 / 0.898 | 0.720 / 0.983 | 0.855 / 0.632 | 0.776 / 0.838 |
| Centered-logit L2 distance | 0.374 / 0.984 | 0.175 / 0.999 | 0.772 / 0.639 | 0.440 / 0.874 |
| Energy gap | 0.520 / 0.983 | 0.440 / 1.000 | 0.162 / 1.000 | 0.374 / 0.994 |
| Absolute energy gap | 0.410 / 0.988 | 0.242 / 1.000 | 0.654 / 0.589 | 0.435 / 0.859 |
| Student MSP | 0.783 / 0.794 | 0.802 / 0.845 | 0.805 / 0.753 | 0.797 / 0.797 |
| Student energy | 0.787 / 0.788 | 0.871 / 0.712 | 0.859 / 0.637 | 0.839 / 0.712 |

### CIFAR-100 ID: Clean Target, Logit MSE

| Score | CIFAR-10 | MNIST | SVHN | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.346 / 0.920 | 0.246 / 0.977 | 0.188 / 0.955 | 0.260 / 0.951 |
| Absolute max probability difference | 0.685 / 0.928 | 0.755 / 0.985 | 0.832 / 0.668 | 0.757 / 0.860 |
| Teacher-student KL | 0.682 / 0.929 | 0.724 / 0.991 | 0.884 / 0.561 | 0.764 / 0.827 |
| Centered-logit L2 distance | 0.381 / 0.992 | 0.246 / 1.000 | 0.841 / 0.617 | 0.489 / 0.870 |
| Energy gap | 0.805 / 0.724 | 0.836 / 0.678 | 0.502 / 0.889 | 0.714 / 0.764 |
| Absolute energy gap | 0.195 / 0.998 | 0.164 / 1.000 | 0.498 / 0.837 | 0.286 / 0.945 |
| Student MSP | 0.766 / 0.798 | 0.799 / 0.843 | 0.836 / 0.706 | 0.800 / 0.783 |
| Student energy | 0.754 / 0.822 | 0.852 / 0.751 | 0.881 / 0.624 | 0.829 / 0.732 |

### CIFAR-100 Summary

- Clean-target cross-entropy student energy is strongest overall:
  macro `0.839 / 0.712`.
- This is a small improvement over teacher energy at `0.821 / 0.737`.
- The improvement is driven by MNIST and SVHN. Near-OOD CIFAR-10 remains
  difficult: student energy reaches only `0.787 / 0.788`, below teacher
  energy at `0.801 / 0.774`.
- Most teacher-student mismatch scores have poor FPR@95. Student-only energy
  is more reliable than disagreement magnitude.

## Overall Interpretation

- Student energy is the clear default score for these PixMix students.
- Clean teacher targets are stronger than perturbed targets under standard
  clean-image inference.
- The improvement over teacher energy is substantial for CIFAR-10 ID and
  modest for CIFAR-100 ID.
- Cross-entropy and logit MSE are both viable. Logit MSE is best for CIFAR-10,
  while cross-entropy is best for CIFAR-100.
- PixMix does not solve the CIFAR-100-ID versus CIFAR-10 near-OOD problem.

## Artifacts

Probability artifacts used in this report:

```text
<run_dir>/probabilities/unperturbed/
```

Training job: `406157`.
Inference job: `406351`.

Numeric metric exports:

```text
reports/outputs/json/pixmix_pixel_perturbation_unperturbed_ood_metrics.json
reports/outputs/json/pixmix_teacher_resnet50_ood_metrics.json
```
