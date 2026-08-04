# Activation-Subspace Students: ResNet-18

Status: completed for CIFAR-10 and CIFAR-100 ID.

## Setup

- Teacher: ResNet-18.
- Feature: globally averaged layer4 embedding.
- Decisive dimensions: 7 for CIFAR-10 and 38 for CIFAR-100.
- Insignificant dimensions: 505 for CIFAR-10 and 474 for CIFAR-100.
- Checkpoint selection: best validation-loss checkpoint.
- OOD datasets: MNIST, SVHN, and the opposite CIFAR dataset.

Students per ID dataset:

- decisive linear student with centered-logit MSE;
- decisive linear student with cross-entropy, temperature 1, alpha 0.5;
- decisive linear student with cross-entropy, temperature 1, alpha 0.8;
- decisive coordinate autoencoder (`7 -> 4 -> 7` for CIFAR-10 and
  `38 -> 24 -> 12 -> 24 -> 38` for CIFAR-100);
- insignificant `256 -> 64 -> 256` autoencoder.

## Training Results

| ID dataset | Student                         | Best validation loss | Test loss | Test accuracy |
| ---------- | ------------------------------- | -------------------: | --------: | ------------: |
| CIFAR-10   | Decisive MSE                    |             1.49e-11 |  1.56e-11 |         0.931 |
| CIFAR-10   | Decisive CE alpha 0.5           |                0.108 |     0.226 |         0.943 |
| CIFAR-10   | Decisive CE alpha 0.8           |                0.144 |     0.222 |         0.937 |
| CIFAR-10   | Decisive coordinate autoencoder |                1.410 |     1.334 |             - |
| CIFAR-10   | Insignificant autoencoder       |             0.000132 |  0.000152 |             - |
| CIFAR-100  | Decisive MSE                    |             2.11e-13 |  1.62e-13 |         0.783 |
| CIFAR-100  | Decisive CE alpha 0.5           |                0.438 |     1.496 |         0.785 |
| CIFAR-100  | Decisive CE alpha 0.8           |                0.650 |     1.803 |         0.784 |
| CIFAR-100  | Decisive coordinate autoencoder |                0.583 |     0.458 |             - |
| CIFAR-100  | Insignificant autoencoder       |             0.003896 |  0.003494 |             - |

## Scores

All tables report `ROC-AUC / FPR@95`. Scores use the project convention where
higher means more ID-like. Decisive students use all classifier scores from
`docs/ood_scores/README.md`.

Insignificant students use:

```text
raw_reconstruction_error = mean((reconstruction - target)^2)
relative_reconstruction_error = ||reconstruction - target||2 / max(||target||2, eps)
cosine_similarity = cosine(reconstruction, target)
```

Reconstruction error and relative reconstruction error are negated before OOD
evaluation. Cosine similarity is used directly.

## Selected Macro Scores

The table below selects one student-dependent OOD Score per student and ID
dataset. Selection maximizes macro ROC-AUC, with lower macro FPR@95 used as a
tie-breaker. Teacher MSP and teacher energy are excluded from the decisive
student selection because they do not depend on the student. Cross-entropy
uses alpha `0.5`, which is stronger than alpha `0.8` for both ID datasets.

| ID dataset | Student type | Training objective | Score | Macro ROC-AUC / FPR@95 |
|---|---|---|---|---:|
| CIFAR-10 | Decisive | Centered-logit MSE | `student_energy` | 0.792 / 0.872 |
| CIFAR-10 | Decisive | Cross-entropy, alpha 0.5 | `student_energy` | 0.819 / 0.779 |
| CIFAR-10 | Insignificant autoencoder | Reconstruction | `cosine_similarity` | 0.907 / 0.550 |
| CIFAR-10 | Decisive coordinate autoencoder | Reconstruction | `relative_reconstruction_error` | 0.509 / 0.915 |
| CIFAR-100 | Decisive | Centered-logit MSE | `energy_gap` | 0.766 / 0.868 |
| CIFAR-100 | Decisive | Cross-entropy, alpha 0.5 | `student_energy` | 0.762 / 0.854 |
| CIFAR-100 | Insignificant autoencoder | Reconstruction | `cosine_similarity` | 0.752 / 0.781 |
| CIFAR-100 | Decisive coordinate autoencoder | Reconstruction | `relative_reconstruction_error` | 0.723 / 0.791 |

## CIFAR-10 ID

### Decisive Centered-Logit MSE

| Score                                 |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ------------------------------------- | ------------: | ------------: | ------------: | ------------: |
| `msp`                                 | 0.918 / 0.556 | 0.904 / 0.598 | 0.879 / 0.624 | 0.900 / 0.593 |
| `energy`                              | 0.958 / 0.260 | 0.918 / 0.412 | 0.879 / 0.506 | 0.918 / 0.392 |
| `student_teacher_kl_divergence`       | 0.684 / 0.956 | 0.618 / 0.992 | 0.722 / 0.933 | 0.675 / 0.960 |
| `max_probability_difference`          | 0.610 / 0.624 | 0.623 / 0.635 | 0.488 / 0.710 | 0.574 / 0.656 |
| `absolute_max_probability_difference` | 0.671 / 0.985 | 0.654 / 0.994 | 0.725 / 0.978 | 0.683 / 0.986 |
| `logit_l2_distance`                   | 0.288 / 1.000 | 0.231 / 1.000 | 0.396 / 0.998 | 0.305 / 0.999 |
| `energy_gap`                          | 0.820 / 0.918 | 0.779 / 0.958 | 0.659 / 0.923 | 0.753 / 0.933 |
| `absolute_energy_gap`                 | 0.173 / 1.000 | 0.190 / 1.000 | 0.340 / 0.998 | 0.234 / 0.999 |
| `student_msp`                         | 0.772 / 0.846 | 0.760 / 0.852 | 0.779 / 0.851 | 0.770 / 0.850 |
| `student_energy`                      | 0.827 / 0.841 | 0.768 / 0.884 | 0.781 / 0.890 | 0.792 / 0.872 |

### Decisive Cross-Entropy, Alpha 0.5

| Score                                 |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ------------------------------------- | ------------: | ------------: | ------------: | ------------: |
| `msp`                                 | 0.918 / 0.556 | 0.904 / 0.598 | 0.879 / 0.624 | 0.900 / 0.593 |
| `energy`                              | 0.958 / 0.260 | 0.918 / 0.412 | 0.879 / 0.506 | 0.918 / 0.392 |
| `student_teacher_kl_divergence`       | 0.707 / 0.878 | 0.678 / 0.940 | 0.753 / 0.855 | 0.713 / 0.891 |
| `max_probability_difference`          | 0.586 / 0.637 | 0.574 / 0.657 | 0.460 / 0.724 | 0.540 / 0.673 |
| `absolute_max_probability_difference` | 0.701 / 0.922 | 0.698 / 0.934 | 0.752 / 0.889 | 0.717 / 0.915 |
| `logit_l2_distance`                   | 0.178 / 1.000 | 0.190 / 1.000 | 0.350 / 0.997 | 0.239 / 0.999 |
| `energy_gap`                          | 0.791 / 0.910 | 0.757 / 0.951 | 0.641 / 0.900 | 0.730 / 0.920 |
| `absolute_energy_gap`                 | 0.212 / 1.000 | 0.218 / 1.000 | 0.370 / 0.999 | 0.267 / 0.999 |
| `student_msp`                         | 0.802 / 0.712 | 0.791 / 0.727 | 0.806 / 0.723 | 0.800 / 0.721 |
| `student_energy`                      | 0.859 / 0.717 | 0.800 / 0.804 | 0.798 / 0.815 | 0.819 / 0.779 |

### Decisive Cross-Entropy, Alpha 0.8

| Score                                 |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ------------------------------------- | ------------: | ------------: | ------------: | ------------: |
| `msp`                                 | 0.918 / 0.556 | 0.904 / 0.598 | 0.879 / 0.624 | 0.900 / 0.593 |
| `energy`                              | 0.958 / 0.260 | 0.918 / 0.412 | 0.879 / 0.506 | 0.918 / 0.392 |
| `student_teacher_kl_divergence`       | 0.688 / 0.930 | 0.635 / 0.979 | 0.732 / 0.909 | 0.685 / 0.939 |
| `max_probability_difference`          | 0.576 / 0.642 | 0.578 / 0.651 | 0.464 / 0.726 | 0.539 / 0.673 |
| `absolute_max_probability_difference` | 0.681 / 0.970 | 0.671 / 0.978 | 0.732 / 0.950 | 0.695 / 0.966 |
| `logit_l2_distance`                   | 0.234 / 1.000 | 0.214 / 1.000 | 0.376 / 0.998 | 0.275 / 0.999 |
| `energy_gap`                          | 0.828 / 0.849 | 0.779 / 0.939 | 0.662 / 0.894 | 0.756 / 0.894 |
| `absolute_energy_gap`                 | 0.175 / 1.000 | 0.194 / 1.000 | 0.348 / 0.999 | 0.239 / 0.999 |
| `student_msp`                         | 0.786 / 0.781 | 0.774 / 0.794 | 0.790 / 0.789 | 0.783 / 0.788 |
| `student_energy`                      | 0.841 / 0.790 | 0.784 / 0.849 | 0.788 / 0.859 | 0.804 / 0.833 |

### Insignificant Autoencoder

| Score                           |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ------------------------------- | ------------: | ------------: | ------------: | ------------: |
| `raw_reconstruction_error`      | 0.851 / 0.850 | 0.860 / 0.743 | 0.850 / 0.694 | 0.854 / 0.762 |
| `relative_reconstruction_error` | 0.946 / 0.392 | 0.912 / 0.575 | 0.860 / 0.691 | 0.906 / 0.553 |
| `cosine_similarity`             | 0.947 / 0.385 | 0.913 / 0.574 | 0.860 / 0.691 | 0.907 / 0.550 |

### Decisive Coordinate Autoencoder

| Score                           |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ------------------------------- | ------------: | ------------: | ------------: | ------------: |
| `raw_reconstruction_error`      | 0.338 / 1.000 | 0.270 / 1.000 | 0.351 / 0.991 | 0.319 / 0.997 |
| `relative_reconstruction_error` | 0.661 / 0.832 | 0.377 / 0.994 | 0.489 / 0.920 | 0.509 / 0.915 |
| `cosine_similarity`             | 0.661 / 0.835 | 0.377 / 0.994 | 0.488 / 0.920 | 0.509 / 0.916 |

### CIFAR-10 Summary

- Teacher energy is the strongest overall score: macro `0.918 / 0.392`.
- Insignificant cosine similarity is close in ROC-AUC at `0.907` and is much
  stronger than all student-only decisive scores, although its FPR@95 is
  higher at `0.550`.
- Normalizing the autoencoder residual matters: relative error improves macro
  ROC-AUC from `0.854` to `0.906` and FPR@95 from `0.762` to `0.553`.
- Among decisive students, cross-entropy alpha 0.5 gives the best student-only
  energy result (`0.819 / 0.779`) and student MSP result (`0.800 / 0.721`).
- Teacher-student distance scores are generally weak at FPR@95. Centered-logit
  and absolute-energy distances are effectively inverted or non-separating.
- The decisive coordinate autoencoder does not recover a useful aggregate
  signal: relative error and cosine are near chance (`0.509` macro ROC-AUC),
  while raw error is inverted (`0.319`). Its only moderately useful pair is
  Far-OOD MNIST (`0.661` ROC-AUC).

## CIFAR-100 ID

### Decisive Centered-Logit MSE

| Score                                 |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ------------------------------------- | ------------: | ------------: | ------------: | ------------: |
| `msp`                                 | 0.701 / 0.956 | 0.806 / 0.812 | 0.792 / 0.794 | 0.767 / 0.854 |
| `energy`                              | 0.695 / 0.974 | 0.829 / 0.771 | 0.795 / 0.800 | 0.773 / 0.848 |
| `student_teacher_kl_divergence`       | 0.582 / 0.998 | 0.491 / 0.985 | 0.584 / 0.951 | 0.552 / 0.978 |
| `max_probability_difference`          | 0.409 / 0.990 | 0.497 / 0.977 | 0.458 / 0.973 | 0.455 / 0.980 |
| `absolute_max_probability_difference` | 0.591 / 0.998 | 0.503 / 0.989 | 0.543 / 0.949 | 0.545 / 0.979 |
| `logit_l2_distance`                   | 0.334 / 1.000 | 0.158 / 1.000 | 0.256 / 0.999 | 0.249 / 1.000 |
| `energy_gap`                          | 0.693 / 0.976 | 0.829 / 0.765 | 0.774 / 0.865 | 0.766 / 0.868 |
| `absolute_energy_gap`                 | 0.307 / 1.000 | 0.170 / 1.000 | 0.226 / 0.999 | 0.234 / 1.000 |
| `student_msp`                         | 0.667 / 0.970 | 0.782 / 0.836 | 0.798 / 0.796 | 0.749 / 0.867 |
| `student_energy`                      | 0.662 / 0.976 | 0.797 / 0.839 | 0.793 / 0.795 | 0.751 / 0.870 |

### Decisive Cross-Entropy, Alpha 0.5

| Score                                 |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ------------------------------------- | ------------: | ------------: | ------------: | ------------: |
| `msp`                                 | 0.701 / 0.956 | 0.806 / 0.812 | 0.792 / 0.794 | 0.767 / 0.854 |
| `energy`                              | 0.695 / 0.974 | 0.829 / 0.771 | 0.795 / 0.800 | 0.773 / 0.848 |
| `student_teacher_kl_divergence`       | 0.597 / 0.997 | 0.539 / 0.978 | 0.639 / 0.938 | 0.592 / 0.971 |
| `max_probability_difference`          | 0.391 / 0.980 | 0.468 / 0.955 | 0.418 / 0.957 | 0.426 / 0.964 |
| `absolute_max_probability_difference` | 0.611 / 0.993 | 0.535 / 0.984 | 0.585 / 0.933 | 0.577 / 0.970 |
| `logit_l2_distance`                   | 0.331 / 1.000 | 0.175 / 1.000 | 0.238 / 0.999 | 0.248 / 1.000 |
| `energy_gap`                          | 0.680 / 0.981 | 0.815 / 0.819 | 0.760 / 0.887 | 0.752 / 0.896 |
| `absolute_energy_gap`                 | 0.319 / 1.000 | 0.183 / 1.000 | 0.239 / 0.999 | 0.247 / 1.000 |
| `student_msp`                         | 0.674 / 0.966 | 0.785 / 0.836 | 0.799 / 0.799 | 0.752 / 0.867 |
| `student_energy`                      | 0.678 / 0.971 | 0.812 / 0.802 | 0.798 / 0.788 | 0.762 / 0.854 |

### Decisive Cross-Entropy, Alpha 0.8

| Score                                 |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ------------------------------------- | ------------: | ------------: | ------------: | ------------: |
| `msp`                                 | 0.701 / 0.956 | 0.806 / 0.812 | 0.792 / 0.794 | 0.767 / 0.854 |
| `energy`                              | 0.695 / 0.974 | 0.829 / 0.771 | 0.795 / 0.800 | 0.773 / 0.848 |
| `student_teacher_kl_divergence`       | 0.587 / 0.998 | 0.506 / 0.983 | 0.601 / 0.947 | 0.565 / 0.976 |
| `max_probability_difference`          | 0.402 / 0.989 | 0.483 / 0.975 | 0.442 / 0.971 | 0.443 / 0.978 |
| `absolute_max_probability_difference` | 0.598 / 0.996 | 0.517 / 0.988 | 0.558 / 0.945 | 0.558 / 0.976 |
| `logit_l2_distance`                   | 0.332 / 1.000 | 0.161 / 1.000 | 0.251 / 0.999 | 0.248 / 1.000 |
| `energy_gap`                          | 0.685 / 0.980 | 0.819 / 0.806 | 0.771 / 0.868 | 0.758 / 0.885 |
| `absolute_energy_gap`                 | 0.315 / 1.000 | 0.181 / 1.000 | 0.229 / 0.999 | 0.242 / 1.000 |
| `student_msp`                         | 0.669 / 0.969 | 0.783 / 0.835 | 0.798 / 0.796 | 0.750 / 0.866 |
| `student_energy`                      | 0.675 / 0.971 | 0.811 / 0.806 | 0.796 / 0.795 | 0.761 / 0.857 |

### Insignificant Autoencoder

| Score                           |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ------------------------------- | ------------: | ------------: | ------------: | ------------: |
| `raw_reconstruction_error`      | 0.793 / 0.766 | 0.562 / 0.937 | 0.485 / 0.963 | 0.613 / 0.889 |
| `relative_reconstruction_error` | 0.848 / 0.618 | 0.764 / 0.815 | 0.642 / 0.910 | 0.752 / 0.781 |
| `cosine_similarity`             | 0.849 / 0.618 | 0.764 / 0.814 | 0.642 / 0.910 | 0.752 / 0.781 |

### Decisive Coordinate Autoencoder

| Score                           |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ------------------------------- | ------------: | ------------: | ------------: | ------------: |
| `raw_reconstruction_error`      | 0.554 / 0.988 | 0.629 / 0.974 | 0.360 / 0.991 | 0.514 / 0.985 |
| `relative_reconstruction_error` | 0.641 / 0.970 | 0.871 / 0.545 | 0.658 / 0.859 | 0.723 / 0.791 |
| `cosine_similarity`             | 0.654 / 0.937 | 0.862 / 0.568 | 0.645 / 0.873 | 0.721 / 0.792 |

### CIFAR-100 Summary

- Teacher energy is best by macro ROC-AUC (`0.773`), but its macro FPR@95 is
  high (`0.848`).
- Insignificant cosine similarity and relative reconstruction error reach
  macro `0.752 / 0.781`, improving FPR@95 over teacher energy while losing
  some ROC-AUC.
- Relative normalization again helps substantially: raw autoencoder error is
  only `0.613 / 0.889` macro.
- Near-OOD CIFAR-10 remains difficult for the insignificant autoencoder:
  cosine and relative error reach only `0.642 / 0.910`.
- Cross-entropy alpha 0.5 is slightly stronger than alpha 0.8 for student
  energy and teacher-student KL, but neither decisive variant clearly improves
  on teacher energy.
- The decisive coordinate autoencoder is competitive with the insignificant
  one but not better overall: relative error reaches `0.723 / 0.791` macro
  versus `0.752 / 0.781`. It is strongest on SVHN (`0.871 / 0.545`) and gives
  a modest Near-OOD CIFAR-10 improvement (`0.658 / 0.859` versus
  `0.642 / 0.910`).

## Overall Interpretation

- The insignificant subspace contains a strong OOD signal, particularly for
  CIFAR-10 ID. Its normalized residual direction is substantially more useful
  than residual magnitude alone.
- Cosine similarity and relative reconstruction error are nearly equivalent
  in both ID settings, indicating that angular reconstruction quality drives
  most of the normalized autoencoder signal.
- A simple “decisive errors detect Far-OOD, insignificant errors detect
  Near-OOD” split is not supported. The CIFAR-10 decisive autoencoder is weak
  on both SVHN and Near-OOD CIFAR-100. For CIFAR-100 ID it is strong on SVHN
  and slightly improves Near-OOD CIFAR-10, while the insignificant
  autoencoder remains stronger in aggregate.
- The decisive students preserve classification accuracy but their mismatch
  scores are weak OOD detectors. Student-only energy/MSP are more useful than
  KL or centered-logit distance.
- Teacher energy remains the strongest CIFAR-10 aggregate score and the best
  CIFAR-100 score by ROC-AUC. The CIFAR-100 insignificant autoencoder offers a
  better FPR tradeoff but does not solve near-OOD CIFAR-10.

## Artifacts

Raw embeddings and inference outputs:

```text
runs/teachers/<id_dataset>/resnet18/activation_subspace_embeddings/
<student_run_dir>/activation_subspace_inference/
```

Score artifacts:

```text
<student_run_dir>/activation_subspace_scores/
  manifest.json
  <dataset>/student_best.pt
```

Each score tensor contains raw metrics, sign-adjusted OOD Scores, labels, and
provenance metadata. There are 40 score tensors: four datasets, including ID,
for each of ten students.

Local metric summaries:

```text
reports/outputs/json/activation_subspace_cifar10_resnet18_metrics.json
reports/outputs/json/activation_subspace_cifar100_resnet18_metrics.json
```

Export jobs:

- `406342`: decisive coordinate-autoencoder training;
- `406343`: refreshed ID/OOD embedding and all-student inference export;
- `406344`: refreshed score and aggregate metric export.
