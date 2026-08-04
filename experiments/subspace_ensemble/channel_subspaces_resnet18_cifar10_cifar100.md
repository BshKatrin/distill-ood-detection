# Channel-Subspace Ensembles: ResNet-18

Status: completed for overlapping, ordered, and partitioned assignments with
CIFAR-10 and CIFAR-100 ID.

## Setup

- Teacher: ResNet-18.
- Teacher feature: `layer4`, shape `512 x 4 x 4`.
- Ensemble: 16 linear students.
- Subspace size: 128 of 512 channels per student.
- Sampling: without replacement within one student, with overlap allowed
  between students.
- Channel representations:
  - `channel_flatten`: selected feature maps flattened to 2,048 values;
  - `channel_gap`: GAP followed by selection of 128 channels.
- Objectives: centered-logit MSE and `KL(teacher || student)`.
- Training: 50 epochs, AdamW, learning rate `1e-3`, weight decay `1e-4`.
- Seed: 42 for the deterministic train/validation split, subspace stream,
  model initialization stream, and training.
- Checkpoint selection: minimum mean member validation distillation loss.
- OOD datasets:
  - CIFAR-10 ID: MNIST, SVHN, and CIFAR-100 test;
  - CIFAR-100 ID: CIFAR-10, MNIST, and SVHN test.

Run directories:

```text
runs/students/subspace_ensemble/channel_flatten/cifar_10/resnet18/linear
runs/students/subspace_ensemble/channel_flatten/cifar_100/resnet18/linear
runs/students/subspace_ensemble/channel_gap/cifar_10/resnet18/linear
runs/students/subspace_ensemble/channel_gap/cifar_100/resnet18/linear
```

## Training Results

| ID dataset | Representation | Objective | Best validation loss | Test loss | Test ensemble accuracy |
|---|---|---|---:|---:|---:|
| CIFAR-10 | Flattened channels | Logit MSE | 0.003826 | 0.004406 | 0.9495 |
| CIFAR-10 | Flattened channels | KL | 1.02e-05 | 0.000235 | 0.9494 |
| CIFAR-10 | Channel GAP | Logit MSE | 0.007310 | 0.008097 | 0.9495 |
| CIFAR-10 | Channel GAP | KL | 4.22e-05 | 0.000922 | 0.9494 |
| CIFAR-100 | Flattened channels | Logit MSE | 0.056827 | 0.048506 | 0.7911 |
| CIFAR-100 | Flattened channels | KL | 0.000376 | 0.014251 | 0.7921 |
| CIFAR-100 | Channel GAP | Logit MSE | 0.072086 | 0.061115 | 0.7922 |
| CIFAR-100 | Channel GAP | KL | 0.000648 | 0.023771 | 0.7903 |

Loss values are objective-dependent and should not be compared directly
between logit MSE and KL.

## OOD Scores

For member probability vectors `p_s`, the raw metrics are

```text
predictive_entropy = H(mean_s p_s)
BALD = H(mean_s p_s) - mean_s H(p_s)
```

Both raw metrics increase with OOD-likeness. They are negated before OOD
evaluation so that higher scores follow the project convention and indicate
more ID-like samples.

All tables report `ROC-AUC / FPR@95` using the best ensemble checkpoint.

## CIFAR-10 ID

| Representation | Objective | Score | MNIST | SVHN | CIFAR-100 | Macro |
|---|---|---|---:|---:|---:|---:|
| Flattened channels | Logit MSE | Predictive entropy | 0.925 / 0.512 | 0.909 / 0.566 | 0.882 / 0.605 | 0.905 / 0.561 |
| Flattened channels | Logit MSE | BALD | 0.913 / 0.609 | 0.896 / 0.672 | 0.881 / 0.634 | 0.897 / 0.638 |
| Flattened channels | KL | Predictive entropy | 0.925 / 0.512 | 0.909 / 0.563 | 0.882 / 0.604 | 0.905 / 0.560 |
| Flattened channels | KL | BALD | 0.923 / 0.528 | 0.913 / 0.549 | 0.882 / 0.607 | 0.906 / 0.561 |
| Channel GAP | Logit MSE | Predictive entropy | 0.926 / 0.509 | 0.909 / 0.563 | 0.882 / 0.603 | 0.906 / 0.558 |
| Channel GAP | Logit MSE | BALD | 0.912 / 0.606 | 0.899 / 0.669 | 0.880 / 0.643 | 0.897 / 0.639 |
| Channel GAP | KL | Predictive entropy | 0.926 / 0.509 | 0.908 / 0.572 | 0.883 / 0.607 | 0.906 / 0.563 |
| Channel GAP | KL | BALD | 0.918 / 0.542 | 0.905 / 0.582 | 0.880 / 0.600 | 0.901 / 0.574 |

### CIFAR-10 Summary

- Predictive entropy is exceptionally stable across representation and
  objective: every macro ROC-AUC is `0.905-0.906`.
- Channel GAP with logit MSE has the best predictive-entropy operating point,
  macro `0.906 / 0.558`, but the difference from flattening is negligible.
- KL changes BALD much more than it changes predictive entropy. For flattened
  channels, KL improves BALD from `0.897 / 0.638` to `0.906 / 0.561`.
- CIFAR-100 is the most difficult OOD dataset for every channel ensemble,
  with ROC-AUC around `0.88` and FPR@95 around `0.60`.

## CIFAR-100 ID

| Representation | Objective | Score | CIFAR-10 | MNIST | SVHN | Macro |
|---|---|---|---:|---:|---:|---:|
| Flattened channels | Logit MSE | Predictive entropy | 0.796 / 0.791 | 0.706 / 0.967 | 0.827 / 0.778 | 0.776 / 0.845 |
| Flattened channels | Logit MSE | BALD | 0.756 / 0.887 | 0.772 / 0.739 | 0.795 / 0.856 | 0.774 / 0.827 |
| Flattened channels | KL | Predictive entropy | 0.795 / 0.793 | 0.700 / 0.968 | 0.819 / 0.785 | 0.771 / 0.849 |
| Flattened channels | KL | BALD | 0.759 / 0.897 | 0.748 / 0.837 | 0.791 / 0.886 | 0.766 / 0.874 |
| Channel GAP | Logit MSE | Predictive entropy | 0.795 / 0.793 | 0.699 / 0.969 | 0.824 / 0.779 | 0.773 / 0.847 |
| Channel GAP | Logit MSE | BALD | 0.746 / 0.886 | 0.762 / 0.744 | 0.803 / 0.830 | 0.770 / 0.820 |
| Channel GAP | KL | Predictive entropy | 0.795 / 0.794 | 0.696 / 0.971 | 0.815 / 0.794 | 0.768 / 0.853 |
| Channel GAP | KL | BALD | 0.778 / 0.854 | 0.756 / 0.765 | 0.823 / 0.751 | 0.786 / 0.790 |

### CIFAR-100 Summary

- Channel-GAP KL BALD is strongest overall at macro `0.786 / 0.790`.
- Predictive entropy is less sensitive to the representation but reaches only
  `0.768-0.776` macro ROC-AUC with FPR@95 above `0.84`.
- Unlike CIFAR-10 ID, KL is not uniformly beneficial. It improves BALD for
  channel GAP but weakens BALD for flattened channels.
- MNIST is unexpectedly difficult for predictive entropy, with ROC-AUC around
  `0.70` and FPR@95 around `0.97`. BALD substantially improves its FPR@95.
- Near-OOD CIFAR-10 remains difficult, especially at the 95% TPR operating
  point.

## Overall Interpretation

- Spatial flattening contributes almost no predictive-entropy improvement over
  GAP. Selecting channel identities appears to drive the useful uncertainty
  signal.
- Predictive entropy is the most reliable channel score for CIFAR-10 ID.
- BALD is more objective-sensitive. KL produces the strongest BALD result for
  both ID datasets, although its benefit depends on the channel representation.
- The ensembles preserve strong classification accuracy while creating useful
  disagreement. OOD separation is substantially stronger for CIFAR-10 ID than
  for CIFAR-100 ID.

## Disjoint Assignment Experiments

The follow-up experiment partitions all 512 channels across the 16 students.
Every student receives 32 channels and no channel appears in more than one
student subspace. Two deterministic assignments are compared:

- `ordered`: student `s` receives contiguous indices `32s` through
  `32(s + 1) - 1`;
- `partitioned`: seed 42 generates one permutation of all 512 indices, which
  is reshaped into 16 rows of 32 indices.

Flattened students therefore receive `32 x 4 x 4 = 512` inputs; GAP students
receive 32 inputs. All other training and evaluation settings match the
original overlapping experiment.

Run directories append `/ordered` or `/partitioned` to the original run
directory.

### Disjoint Training Results

| ID dataset | Representation | Assignment | Objective | Best validation loss | Test loss | Test ensemble accuracy |
|---|---|---|---|---:|---:|---:|
| CIFAR-10 | Flattened channels | Ordered | Logit MSE | 0.095850 | 0.103756 | 0.9493 |
| CIFAR-10 | Flattened channels | Ordered | KL | 0.000144 | 0.003126 | 0.9494 |
| CIFAR-10 | Flattened channels | Partitioned | Logit MSE | 0.097919 | 0.106181 | 0.9490 |
| CIFAR-10 | Flattened channels | Partitioned | KL | 0.000160 | 0.003522 | 0.9503 |
| CIFAR-10 | Channel GAP | Ordered | Logit MSE | 0.180530 | 0.197413 | 0.9491 |
| CIFAR-10 | Channel GAP | Ordered | KL | 0.001296 | 0.011124 | 0.9496 |
| CIFAR-10 | Channel GAP | Partitioned | Logit MSE | 0.192319 | 0.211522 | 0.9493 |
| CIFAR-10 | Channel GAP | Partitioned | KL | 0.001423 | 0.012922 | 0.9498 |
| CIFAR-100 | Flattened channels | Ordered | Logit MSE | 0.901935 | 0.670177 | 0.7878 |
| CIFAR-100 | Flattened channels | Ordered | KL | 0.019937 | 0.411592 | 0.7891 |
| CIFAR-100 | Flattened channels | Partitioned | Logit MSE | 0.899836 | 0.669591 | 0.7897 |
| CIFAR-100 | Flattened channels | Partitioned | KL | 0.018851 | 0.410118 | 0.7913 |
| CIFAR-100 | Channel GAP | Ordered | Logit MSE | 1.228768 | 0.901834 | 0.7874 |
| CIFAR-100 | Channel GAP | Ordered | KL | 0.071009 | 0.509744 | 0.7886 |
| CIFAR-100 | Channel GAP | Partitioned | Logit MSE | 1.229096 | 0.903606 | 0.7906 |
| CIFAR-100 | Channel GAP | Partitioned | KL | 0.071530 | 0.512200 | 0.7880 |

### Disjoint CIFAR-10 ID Scores

| Representation | Assignment | Objective | Score | MNIST | SVHN | CIFAR-100 | Macro |
|---|---|---|---|---:|---:|---:|---:|
| Flattened channels | Ordered | Logit MSE | Predictive entropy | 0.928 / 0.501 | 0.910 / 0.563 | 0.884 / 0.604 | 0.907 / 0.556 |
| Flattened channels | Ordered | Logit MSE | BALD | 0.915 / 0.588 | 0.894 / 0.698 | 0.877 / 0.663 | 0.895 / 0.650 |
| Flattened channels | Ordered | KL | Predictive entropy | 0.923 / 0.520 | 0.908 / 0.574 | 0.883 / 0.607 | 0.905 / 0.567 |
| Flattened channels | Ordered | KL | BALD | 0.913 / 0.573 | 0.894 / 0.647 | 0.879 / 0.621 | 0.895 / 0.614 |
| Flattened channels | Partitioned | Logit MSE | Predictive entropy | 0.927 / 0.501 | 0.909 / 0.561 | 0.883 / 0.601 | 0.907 / 0.554 |
| Flattened channels | Partitioned | Logit MSE | BALD | 0.906 / 0.649 | 0.897 / 0.637 | 0.877 / 0.659 | 0.893 / 0.648 |
| Flattened channels | Partitioned | KL | Predictive entropy | 0.921 / 0.527 | 0.908 / 0.566 | 0.883 / 0.604 | 0.904 / 0.566 |
| Flattened channels | Partitioned | KL | BALD | 0.908 / 0.626 | 0.903 / 0.603 | 0.878 / 0.615 | 0.896 / 0.615 |
| Channel GAP | Ordered | Logit MSE | Predictive entropy | 0.930 / 0.492 | 0.909 / 0.561 | 0.882 / 0.606 | 0.907 / 0.553 |
| Channel GAP | Ordered | Logit MSE | BALD | 0.921 / 0.586 | 0.897 / 0.714 | 0.869 / 0.715 | 0.895 / 0.671 |
| Channel GAP | Ordered | KL | Predictive entropy | 0.930 / 0.468 | 0.905 / 0.554 | 0.884 / 0.574 | 0.906 / 0.532 |
| Channel GAP | Ordered | KL | BALD | 0.915 / 0.556 | 0.899 / 0.593 | 0.880 / 0.574 | 0.898 / 0.574 |
| Channel GAP | Partitioned | Logit MSE | Predictive entropy | 0.925 / 0.505 | 0.908 / 0.558 | 0.883 / 0.603 | 0.905 / 0.555 |
| Channel GAP | Partitioned | Logit MSE | BALD | 0.904 / 0.659 | 0.893 / 0.695 | 0.871 / 0.697 | 0.890 / 0.684 |
| Channel GAP | Partitioned | KL | Predictive entropy | 0.926 / 0.490 | 0.903 / 0.563 | 0.885 / 0.578 | 0.905 / 0.544 |
| Channel GAP | Partitioned | KL | BALD | 0.903 / 0.651 | 0.891 / 0.640 | 0.881 / 0.581 | 0.892 / 0.624 |

### Disjoint CIFAR-100 ID Scores

| Representation | Assignment | Objective | Score | CIFAR-10 | MNIST | SVHN | Macro |
|---|---|---|---|---:|---:|---:|---:|
| Flattened channels | Ordered | Logit MSE | Predictive entropy | 0.779 / 0.818 | 0.760 / 0.947 | 0.874 / 0.688 | 0.804 / 0.818 |
| Flattened channels | Ordered | Logit MSE | BALD | 0.288 / 0.993 | 0.544 / 0.975 | 0.260 / 0.995 | 0.364 / 0.988 |
| Flattened channels | Ordered | KL | Predictive entropy | 0.789 / 0.811 | 0.733 / 0.953 | 0.828 / 0.766 | 0.783 / 0.843 |
| Flattened channels | Ordered | KL | BALD | 0.771 / 0.844 | 0.767 / 0.806 | 0.803 / 0.849 | 0.780 / 0.833 |
| Flattened channels | Partitioned | Logit MSE | Predictive entropy | 0.786 / 0.818 | 0.755 / 0.950 | 0.866 / 0.694 | 0.802 / 0.820 |
| Flattened channels | Partitioned | Logit MSE | BALD | 0.295 / 0.991 | 0.504 / 0.985 | 0.290 / 0.992 | 0.363 / 0.989 |
| Flattened channels | Partitioned | KL | Predictive entropy | 0.791 / 0.804 | 0.732 / 0.953 | 0.833 / 0.752 | 0.785 / 0.836 |
| Flattened channels | Partitioned | KL | BALD | 0.780 / 0.837 | 0.780 / 0.770 | 0.833 / 0.774 | 0.797 / 0.794 |
| Channel GAP | Ordered | Logit MSE | Predictive entropy | 0.780 / 0.818 | 0.678 / 0.974 | 0.845 / 0.727 | 0.768 / 0.840 |
| Channel GAP | Ordered | Logit MSE | BALD | 0.206 / 0.999 | 0.343 / 1.000 | 0.190 / 1.000 | 0.246 / 1.000 |
| Channel GAP | Ordered | KL | Predictive entropy | 0.791 / 0.808 | 0.700 / 0.969 | 0.824 / 0.737 | 0.772 / 0.838 |
| Channel GAP | Ordered | KL | BALD | 0.774 / 0.850 | 0.743 / 0.807 | 0.820 / 0.738 | 0.779 / 0.798 |
| Channel GAP | Partitioned | Logit MSE | Predictive entropy | 0.788 / 0.817 | 0.670 / 0.973 | 0.831 / 0.722 | 0.763 / 0.837 |
| Channel GAP | Partitioned | Logit MSE | BALD | 0.212 / 0.998 | 0.345 / 1.000 | 0.207 / 1.000 | 0.255 / 0.999 |
| Channel GAP | Partitioned | KL | Predictive entropy | 0.795 / 0.803 | 0.693 / 0.963 | 0.832 / 0.736 | 0.773 / 0.834 |
| Channel GAP | Partitioned | KL | BALD | 0.778 / 0.850 | 0.754 / 0.778 | 0.851 / 0.666 | 0.794 / 0.765 |

### Disjoint Assignment Summary

- CIFAR-10 predictive entropy is almost unchanged by removing overlap:
  macro ROC-AUC remains `0.904-0.907`. Ordered channel-GAP KL has the best
  disjoint FPR@95 at `0.906 / 0.532`.
- CIFAR-10 BALD weakens relative to the best overlapping result. Ordered
  channel-GAP KL is the strongest disjoint BALD configuration at
  `0.898 / 0.574`.
- For CIFAR-100, partitioned flattened KL BALD reaches `0.797 / 0.794`, while
  partitioned channel-GAP KL trades slightly lower ROC-AUC for the best FPR@95
  at `0.794 / 0.765`.
- MSE BALD becomes strongly inverted for CIFAR-100 when every student sees
  only 32 non-overlapping channels. KL is necessary to recover useful
  disagreement.
- Ordered and partitioned assignments preserve essentially the same ensemble
  classification accuracy. Assignment choice affects BALD much more than
  predictive entropy.

## Artifacts

Inference artifacts:

```text
<run_dir>/subspace_ensemble_inference/
  manifest.json
  <dataset>/student_<objective>_best.pt
```

Score artifacts:

```text
<run_dir>/subspace_ensemble_scores/
  manifest.json
  <objective>/manifest.json
  <dataset>/student_<objective>_best.pt
```

Training, inference, and score-export jobs:

- `406532`: CIFAR-10;
- `406533`: CIFAR-100;
- `406648`: ordered disjoint training;
- `406971`: partitioned disjoint training completion;
- `406980`, `406981`, `406985`, `406986`: disjoint inference and score export.
