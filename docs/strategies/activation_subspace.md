# Activation-Subspace Students

This reference defines the project's ActSub-inspired student methods.
Use the [workflow](../workflows/activation_subspace.md) for commands and artifact
contracts, and the [experiment record](../../experiments/activation_subspace/resnet18_cifar10_cifar100.md)
for measured results.

## Provenance

The decomposition is based on **Activation Subspaces for Out-of-Distribution
Detection**, Barış Zöngür, Robin Hesse, and Stefan Roth, ICCV 2025:

- [CVF paper](https://openaccess.thecvf.com/content/ICCV2025/html/Zongur_Activation_Subspaces_for_Out-of-Distribution_Detection_ICCV_2025_paper.html)
- [Official implementation: `visinf/actsub`](https://github.com/visinf/actsub/)

A local Markdown copy may be kept under `papers/md/`; that ignored folder is
not included in a Git clone. Use the public paper link above as the reference.

The official repository provides separate OpenOOD and standard evaluation
setups. This repository does not copy those pipelines; it implements a new
student-training variant using the paper's classifier-SVD decomposition and
norm-balancing rule.

## Paper idea

Let a frozen classifier map an input to a penultimate activation
`a in R^n`, followed by a linear head

```text
l = W a + b,  W in R^(c x n).
```

The paper observes that not every activation direction contributes equally to
the logits. The nullspace of `W` has no effect on the logits, while other
directions may have only a small effect. The paper also identifies a
softmax-invariant direction: adding an equal offset to every logit does not
change the softmax output. These weakly constrained activation directions can
still contain input statistics useful for OOD detection.

ActSub uses a complete SVD of the classifier weight:

```text
W = U diag(s) V^T,
```

with singular values ordered from largest to smallest. The rows of `V^T` form
an orthonormal activation-space basis:

- early rows are **decisive** directions, with the strongest effect on logits;
- later rows are **insignificant** directions, with a weak or zero effect.

The paper argues that the insignificant component is useful for Far-OOD
because it is less shaped by the ID classification objective. Conversely,
insignificant directions can interfere with activation-shaping methods, so the
decisive component is useful for Near-OOD shaping. The original ActSub detector
combines a neighbor-based cosine score in the insignificant subspace with an
energy score after shaping the decisive component.

## What this project changes

This project does **not** reproduce the paper's final combined detector. It
asks whether shallow students trained separately in the two subspaces learn
useful ID structure:

```text
frozen teacher image encoder
  -> layer4 feature map
  -> global average pooling
  -> 512-dimensional embedding a
  -> classifier-SVD coordinates
       |-> decisive linear logit students
       |-> decisive coordinate autoencoder
       `-> insignificant autoencoder
```

The paper contributes the basis and split criterion. The linear student,
autoencoder, distillation objectives, raw inference artifacts, and OOD Scores
below are project-specific extensions.

Project terminology uses **decisive**, matching the paper and code. In
discussion, “significant” refers to the same component. Configs and artifact
metadata always use `decisive` or `insignificant`.

## Decomposition and split selection

For a pooled embedding row vector `a`, define SVD coordinates

```text
z = a V^T.
```

For a candidate split `k`:

```text
V_dec = V^T[:k]
V_ins = V^T[k:]

z_dec = a V_dec^T
z_ins = a V_ins^T

a_dec = z_dec V_dec
a_ins = z_ins V_ins
```

The implementation uses `torch.linalg.svd(W, full_matrices=True)`. Full
matrices are required: for a `c x n` classifier with `c < n`, a reduced SVD
would omit nullspace directions. `right_basis` in artifacts is `V^T`, so its
**rows** are ordered basis directions.

Following Sec. 3.1 of the paper, `k` minimizes the gap between average ID-train
component norms:

```text
k = argmin_j abs(
      mean_i ||z_dec^(i; j)||_2
      - mean_i ||z_ins^(i; j)||_2
    ).
```

The implementation evaluates candidates `0 <= j < n`, matching the reference
implementation, and then rejects a selected endpoint because both project
students require non-empty components. The fit uses only the deterministic ID
student-training subset. No OOD sample participates in SVD or `k` selection.

Current ResNet-18 results are:

| ID dataset | Embedding dimension | Decisive `k` | Insignificant dimension |
| ---------- | ------------------: | -----------: | ----------------------: |
| CIFAR-10   |                 512 |            7 |                     505 |
| CIFAR-100  |                 512 |           38 |                     474 |

Configs contain `expected_dimension` and matching `student.input_shape` as
reproducibility guards. Training fails if a different teacher, revision, data
split, or preprocessing produces another `k`; update the config only after
verifying why the fitted split changed.

The classifier bias is irrelevant to the SVD and is therefore not decomposed.
It is included when producing decisive target logits.

## Data and preprocessing

- Teachers: Hugging Face ResNet-18 classifiers trained on CIFAR-10 or
  CIFAR-100.
- Feature layer: `layer4` only.
- Feature-map requirement: the forwarder must return a 4D `NCHW` tensor.
- Pooling: arithmetic mean over height and width.
- Embedding width: 512.
- Train/validation split: deterministic 90/10 split of the official ID train
  set using seed 42.
- Test data: official ID test split.
- OOD data: MNIST test, SVHN test, and the opposite CIFAR test set.
- OOD images use the preprocessing and normalization of the corresponding ID
  experiment, as defined by the shared dataset loader.

The teacher is frozen. Clean pooled embeddings for ID train, validation, and
test are extracted once into CPU tensors during each training run. They are
not persisted as training caches. OOD data is neither opened nor used during
training.

## Decisive students

### Linear logit variants

The input is the decisive coordinate vector `z_dec in R^k`. The target is not
the full teacher logit vector. It is the frozen teacher head applied to the
decisive reconstruction:

```text
target_logits = W (z_dec V_dec) + b.
```

This isolates the contribution represented by the selected decisive
directions. The student is exactly

```text
Linear(k, number_of_ID_classes).
```

There are no hidden layers or nonlinearities.

### Centered-logit MSE

The first variant removes the per-sample logit offset before MSE:

```text
center(z) = z - mean_classes(z)
loss = MSE(center(student_logits), center(target_logits)).
```

Centering respects softmax invariance: adding the same constant to every logit
does not change class probabilities.

### Cross-entropy distillation

The other variants reuse `distillation_loss("cross_entropy", ...)`:

```text
p_teacher = softmax(target_logits / T)
log_p_student = log_softmax(student_logits / T)

soft_loss = -mean(sum(p_teacher * log_p_student))
hard_loss = cross_entropy(student_logits, ID_labels)

loss = alpha * T^2 * soft_loss + (1 - alpha) * hard_loss.
```

Current runs use `T = 1` and `alpha in {0.5, 0.8}`. Thus alpha controls the
weight of projected-teacher soft targets; `1 - alpha` weights the hard CIFAR
labels.

### Coordinate-autoencoder variant

The feature-distillation variant uses both the decisive input and target as
the same raw coordinate vector `z_dec`. It applies ordinary elementwise MSE
without logit centering, labels, whitening, or coordinate rescaling. The
confirmed genuine bottlenecks are:

```text
CIFAR-10:  7 -> 4 -> 7
CIFAR-100: 38 -> 24 -> 12 -> 24 -> 38
```

Every hidden linear layer is followed by GELU and the output layer is linear.
This student reconstructs decisive features; it does not output class logits.
It therefore exports the same three reconstruction-based OOD Scores as the
insignificant autoencoder.

## Insignificant autoencoder

The input and target are the same insignificant coordinate vector
`z_ins in R^(512-k)`. The classical, non-residual autoencoder is

```text
z_ins
  -> Linear(d_ins, 256) -> GELU
  -> Linear(256, 64)    -> GELU
  -> Linear(64, 256)    -> GELU
  -> Linear(256, d_ins)
  -> reconstruction.
```

The output layer is linear. There is no skip connection, normalization,
dropout, whitening, or coordinate rescaling. The objective is ordinary feature
MSE over all insignificant coordinates.

Because the generic student config calls its output width `num_classes`, the
autoencoder configs set `student.num_classes` to `505` or `474`. This does not
mean the autoencoder predicts classes. Similarly, the insignificant configs
enable `training.methods.mse_logits` to select the shared training schedule;
the activation-subspace branch still uses feature reconstruction MSE.

## OOD Scores and metrics

Score export follows the [inference workflow](../workflows/activation_subspace.md).
See [aggregate metrics](../evaluation/metrics.md) for the historical ID-positive convention.

### Projected-logit scores

Teacher logits are reconstructed as `W a + b` from the clean full embedding;
for this ResNet head this is equivalent to the standard clean-image teacher
forward. Student logits come from the decisive student. The exporter computes
all classifier scores in `docs/evaluation/ood-scores.md`:

- teacher MSP and energy;
- `KL(teacher || student)`;
- max-probability difference and its absolute value;
- centered-logit L2 distance;
- energy gap and its absolute value;
- student MSP and student energy.

### Coordinate-reconstruction scores

Given target coordinates `z` and reconstruction `z_hat`:

```text
raw_reconstruction_error = mean((z_hat - z)^2)
relative_reconstruction_error = ||z_hat - z||_2 / max(||z||_2, 1e-12)
cosine_similarity = cosine(z_hat, z).
```

Reconstruction metrics receive sign `-1`; cosine similarity receives sign
`+1`. Every exported OOD Score therefore follows the project convention:
higher means more ID-like. These definitions apply to coordinate
autoencoders in either the decisive or insignificant subspace.

## Invariants and common mistakes

- Only ResNet-style teachers with a linear `.fc` classifier are supported.
- The current implementation is layer4/global-average-pooling only.
- Use a complete SVD; reduced SVD silently drops nullspace directions.
- `right_basis` stores basis vectors as rows, not columns.
- Fit `k` on the deterministic ID training subset only.
- Projected-logit targets come from the decisive projection, not full teacher
  logits.
- Cross-entropy additionally uses hard ID labels; centered MSE does not.
- Autoencoder coordinates are not whitened or standardized.
- `student.num_classes` is the autoencoder output width for coordinate-target
  configs.
- Training, inference, and scoring are separate stages.
- Inference and scoring require each run's own `activation_subspace.pt`.
- Score metrics require both the ID test distribution and OOD distributions.
- Teacher MSP/energy are identical across decisive variants for one teacher;
  repeated rows in the report are expected.
- The current project scores are not the paper's original top-neighbor cosine,
  activation shaping, or combined ActSub score.

## Extending the variant

When adding another dataset, teacher, layer, or architecture:

1. verify how to obtain the exact linear classifier input embedding;
2. verify that the forwarder and pooling preserve the teacher's normal head
   computation;
3. fit the full SVD and inspect the norm-gap curve;
4. record the selected non-empty dimensions in config guards;
5. add separate run directories with `mlflow.enabled: false`;
6. add ID/OOD inference before score export;
7. test artifact shapes, signs, and provenance;
8. update the experiment report rather than treating incomparable loss scales
   as comparable across components or dimensions.
