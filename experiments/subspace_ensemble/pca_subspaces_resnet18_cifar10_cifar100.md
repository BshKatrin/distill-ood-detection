# Whitened PCA-Subspace Ensembles: ResNet-18

Status: completed for overlapping, ordered, and partitioned assignments with
CIFAR-10 and CIFAR-100 ID.

## Setup

- Teacher: ResNet-18.
- Teacher feature: GAP over `layer4`, producing 512 values.
- PCA fit data: deterministic 90% ID student-training subset only.
- PCA transform: complete 512-component PCA with whitening.
- Ensemble: 16 linear students.
- Student input: 128 of 512 whitened PCA coordinates.
- Sampling: without replacement within one student, with overlap allowed
  between students.
- Objectives: centered-logit MSE and `KL(teacher || student)`.
- Training: 50 epochs, AdamW, learning rate `1e-3`, weight decay `1e-4`.
- Seed: 42 for the deterministic split, PCA-subspace stream, model
  initialization stream, and training.
- Checkpoint selection: minimum mean member validation distillation loss.
- OOD datasets:
  - CIFAR-10 ID: MNIST, SVHN, and CIFAR-100 test;
  - CIFAR-100 ID: CIFAR-10, MNIST, and SVHN test.

Run directories:

```text
runs/students/subspace_ensemble/pca_gap/cifar_10/resnet18/linear
runs/students/subspace_ensemble/pca_gap/cifar_100/resnet18/linear
```

For a pooled embedding `a`, PCA mean `mu`, component matrix `V`, and training
coordinate standard deviations `sigma`, the shared representation is

```text
z = ((a - mu) V^T) / max(sigma, 1e-6).
```

Each student selects 128 coordinates from `z`.

## Training Results

| ID dataset | Objective | Best validation loss | Test loss | Test ensemble accuracy |
|---|---|---:|---:|---:|
| CIFAR-10 | Logit MSE | 11.4348 | 11.0613 | 0.9266 |
| CIFAR-10 | KL | 0.687004 | 0.790006 | 0.9440 |
| CIFAR-100 | Logit MSE | 1.73278 | 1.32158 | 0.7884 |
| CIFAR-100 | KL | 0.008590 | 0.274118 | 0.7883 |

Loss values are objective-dependent and should not be compared directly.
The larger PCA-MSE scale also reflects the difficulty of reconstructing full
teacher logits from randomly selected whitened coordinates.

## OOD Scores

For member probability vectors `p_s`, the raw metrics are

```text
predictive_entropy = H(mean_s p_s)
BALD = H(mean_s p_s) - mean_s H(p_s)
```

Both raw metrics are negated before OOD evaluation, so higher values indicate
more ID-like samples. All tables report `ROC-AUC / FPR@95` using the best
ensemble checkpoint.

## CIFAR-10 ID

| Objective | Score | MNIST | SVHN | CIFAR-100 | Macro |
|---|---|---:|---:|---:|---:|
| Logit MSE | Predictive entropy | 0.844 / 0.683 | 0.680 / 0.923 | 0.685 / 0.794 | 0.736 / 0.800 |
| Logit MSE | BALD | 0.065 / 1.000 | 0.158 / 1.000 | 0.235 / 0.994 | 0.153 / 0.998 |
| KL | Predictive entropy | 0.867 / 0.584 | 0.845 / 0.618 | 0.849 / 0.599 | 0.854 / 0.601 |
| KL | BALD | 0.726 / 0.868 | 0.788 / 0.815 | 0.802 / 0.687 | 0.772 / 0.790 |

### CIFAR-10 Summary

- KL predictive entropy is strongest at macro `0.854 / 0.601`.
- KL improves predictive entropy over logit MSE by `0.118` macro ROC-AUC and
  lowers macro FPR@95 by `0.199`.
- PCA-MSE BALD is strongly inverted: macro ROC-AUC is `0.153` and FPR@95 is
  `0.998`. Under the fixed negative sign, this indicates greater member
  disagreement on ID than on OOD samples.
- KL recovers a useful BALD signal, but it remains below KL predictive entropy.
- Even with KL, the PCA ensemble is weaker than the channel ensembles for
  CIFAR-10 ID.

## CIFAR-100 ID

| Objective | Score | CIFAR-10 | MNIST | SVHN | Macro |
|---|---|---:|---:|---:|---:|
| Logit MSE | Predictive entropy | 0.783 / 0.825 | 0.727 / 0.961 | 0.837 / 0.739 | 0.782 / 0.842 |
| Logit MSE | BALD | 0.229 / 0.999 | 0.282 / 1.000 | 0.173 / 1.000 | 0.228 / 1.000 |
| KL | Predictive entropy | 0.794 / 0.795 | 0.726 / 0.958 | 0.818 / 0.794 | 0.779 / 0.849 |
| KL | BALD | 0.780 / 0.840 | 0.777 / 0.710 | 0.809 / 0.804 | 0.789 / 0.785 |

### CIFAR-100 Summary

- KL BALD is strongest overall at macro `0.789 / 0.785`.
- This is marginally stronger than the best channel result, channel-GAP KL
  BALD at `0.786 / 0.790`.
- Predictive entropy is nearly objective-invariant in macro ROC-AUC, but both
  objectives have high macro FPR@95 (`0.842-0.849`).
- PCA-MSE BALD is again strongly inverted on every OOD dataset.
- KL BALD improves MNIST FPR@95 from roughly `0.96` for predictive entropy to
  `0.710`, while maintaining useful results on CIFAR-10 and SVHN.

## Overall Interpretation

- Whitening and random PCA-coordinate selection make the ensemble highly
  sensitive to the distillation objective.
- Logit MSE produces poor BALD behavior for both ID datasets. The members
  preserve reasonable classification accuracy but their disagreement is
  larger on ID samples, reversing the intended OOD ordering.
- KL is the appropriate objective for PCA-subspace disagreement. It improves
  both CIFAR-10 scores and yields the strongest CIFAR-100 macro BALD result
  among all tested subspace ensembles.
- PCA does not improve the reliable predictive-entropy result obtained from
  channel selection. Its only advantage is the small CIFAR-100 KL-BALD gain.

## Disjoint Assignment Experiments

The follow-up experiment partitions all 512 whitened PCA coordinates across
the 16 students. Every student receives 32 coordinates and no coordinate
appears in more than one student subspace. `ordered` uses contiguous PCA
indices in decreasing-eigenvalue order; `partitioned` reshapes one seed-42
permutation of all coordinates into 16 equal rows.

Run directories append `/ordered` or `/partitioned` to the original run
directory. PCA is refitted separately for each run on the same deterministic
ID training subset.

### Disjoint Training Results

| ID dataset | Assignment | Objective | Best validation loss | Test loss | Test ensemble accuracy |
|---|---|---|---:|---:|---:|
| CIFAR-10 | Ordered | Logit MSE | 13.639660 | 13.202963 | 0.7425 |
| CIFAR-10 | Ordered | KL | 2.142037 | 2.096017 | 0.9488 |
| CIFAR-10 | Partitioned | Logit MSE | 13.639344 | 13.199660 | 0.7621 |
| CIFAR-10 | Partitioned | KL | 1.807103 | 1.807699 | 0.9401 |
| CIFAR-100 | Ordered | Logit MSE | 2.152667 | 1.636543 | 0.7838 |
| CIFAR-100 | Ordered | KL | 3.512613 | 2.948293 | 0.7881 |
| CIFAR-100 | Partitioned | Logit MSE | 2.152631 | 1.636822 | 0.7713 |
| CIFAR-100 | Partitioned | KL | 0.799510 | 1.582936 | 0.7746 |

### Disjoint CIFAR-10 ID Scores

| Assignment | Objective | Score | MNIST | SVHN | CIFAR-100 | Macro |
|---|---|---|---:|---:|---:|---:|
| Ordered | Logit MSE | Predictive entropy | 0.392 / 0.995 | 0.561 / 0.999 | 0.475 / 0.969 | 0.476 / 0.988 |
| Ordered | Logit MSE | BALD | 0.114 / 1.000 | 0.265 / 1.000 | 0.195 / 0.998 | 0.191 / 0.999 |
| Ordered | KL | Predictive entropy | 0.596 / 0.640 | 0.557 / 0.692 | 0.638 / 0.653 | 0.597 / 0.662 |
| Ordered | KL | BALD | 0.068 / 0.999 | 0.174 / 0.984 | 0.233 / 0.970 | 0.158 / 0.984 |
| Partitioned | Logit MSE | Predictive entropy | 0.534 / 0.945 | 0.660 / 0.989 | 0.569 / 0.898 | 0.588 / 0.944 |
| Partitioned | Logit MSE | BALD | 0.110 / 1.000 | 0.080 / 0.999 | 0.204 / 0.991 | 0.131 / 0.997 |
| Partitioned | KL | Predictive entropy | 0.869 / 0.563 | 0.892 / 0.620 | 0.860 / 0.617 | 0.874 / 0.600 |
| Partitioned | KL | BALD | 0.178 / 0.999 | 0.126 / 0.997 | 0.352 / 0.963 | 0.218 / 0.986 |

### Disjoint CIFAR-100 ID Scores

| Assignment | Objective | Score | CIFAR-10 | MNIST | SVHN | Macro |
|---|---|---|---:|---:|---:|---:|
| Ordered | Logit MSE | Predictive entropy | 0.790 / 0.812 | 0.696 / 0.979 | 0.844 / 0.695 | 0.777 / 0.829 |
| Ordered | Logit MSE | BALD | 0.216 / 0.999 | 0.296 / 1.000 | 0.160 / 1.000 | 0.224 / 1.000 |
| Ordered | KL | Predictive entropy | 0.792 / 0.789 | 0.721 / 0.966 | 0.803 / 0.818 | 0.772 / 0.858 |
| Ordered | KL | BALD | 0.217 / 0.993 | 0.392 / 0.970 | 0.214 / 0.993 | 0.274 / 0.985 |
| Partitioned | Logit MSE | Predictive entropy | 0.742 / 0.887 | 0.534 / 0.995 | 0.873 / 0.615 | 0.716 / 0.832 |
| Partitioned | Logit MSE | BALD | 0.224 / 0.999 | 0.366 / 1.000 | 0.191 / 1.000 | 0.260 / 0.999 |
| Partitioned | KL | Predictive entropy | 0.793 / 0.804 | 0.707 / 0.975 | 0.805 / 0.840 | 0.768 / 0.873 |
| Partitioned | KL | BALD | 0.701 / 0.927 | 0.800 / 0.745 | 0.737 / 0.869 | 0.746 / 0.847 |

### Disjoint Assignment Summary

- Assignment structure is decisive for PCA. With CIFAR-10 ID, partitioned KL
  predictive entropy reaches `0.874 / 0.600`, slightly exceeding the original
  overlapping result (`0.854 / 0.601`), while ordered KL falls to
  `0.597 / 0.662`.
- BALD remains inverted for every disjoint CIFAR-10 PCA configuration. The
  stronger partitioned predictive-entropy result therefore does not translate
  into useful member disagreement.
- With CIFAR-100 ID, ordered MSE predictive entropy is the strongest disjoint
  PCA entropy result at `0.777 / 0.829`.
- Partitioned KL substantially improves disjoint PCA BALD over ordered KL
  (`0.746 / 0.847` versus `0.274 / 0.985`), but remains below the original
  overlapping KL-BALD result (`0.789 / 0.785`).
- Contiguous PCA blocks are especially fragile because PCA coordinates are
  ordered by explained variance; the students consequently receive strongly
  unequal information content despite equal input dimensionality.

## Artifacts

Shared PCA and selected subspaces:

```text
<run_dir>/subspace_ensemble.pt
```

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
