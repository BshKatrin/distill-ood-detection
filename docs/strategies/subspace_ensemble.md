# Disjoint Subspace Ensembles

This strategy trains 16 linear students on deterministic, disjoint subspaces of the
same frozen ResNet-18 `layer4` representation. It tests whether disagreement
created by partial feature access is useful for OOD detection.

The disagreement score follows
[Bayesian Active Learning for Classification and Preference Learning](https://arxiv.org/pdf/1112.5745),
with the finite student ensemble standing in for the posterior over models.

## Experiment matrix

The implemented matrix contains CIFAR-10 and CIFAR-100 ID experiments for
three input representations:

| Method | Shared teacher representation | One student input |
| --- | --- | ---: |
| `channel_flatten` | `layer4`, shape 512×4×4 | 32 selected channels, flattened to 512 |
| `channel_gap` | GAP over `layer4` | 32 selected channels |
| `pca_gap` | whitened complete PCA after GAP | 32 selected PCA coordinates |

Every ensemble contains 16 students. For feature dimension `D` and ensemble
size `S`, one student receives `D / S` indices. Config validation requires
`D % S == 0`. The assignments collectively contain every index exactly once,
so different students never overlap.

## Reproducibility and fitting

Two deterministic assignment modes are implemented:

- `ordered`: student `s` receives the contiguous interval from
  `s * (D / S)` through `(s + 1) * (D / S) - 1`;
- `partitioned`: seed 42 produces one permutation of all `D` indices, which is
  reshaped into `S` equal rows.

Before each objective, the global seed is reset once and the 16 models are
constructed sequentially. This gives both objectives the same reproducible
sequence of 16 distinct initializations.

For `pca_gap`, PCA is fitted only on the deterministic 90% ID
student-training subset. Validation, ID test, and OOD samples do not
participate. Let `a` be a GAP embedding, `mu` the training mean, rows of `V`
the covariance eigenvectors in decreasing-eigenvalue order, and `sigma` the
training coordinate standard deviations:

```text
z = ((a - mu) V^T) / max(sigma, 1e-6).
```

All 512 components are retained by the shared transform before each student
receives 32 coordinates. PCA state and assigned indices are persisted; later
inference never refits them.

## Students and objectives

Each member is a single linear layer mapping its selected input to the ID
class logits. Two separate ensembles are trained over the same selections:

- centered-logit MSE (`mse_logits`);
- `KL(teacher || student)` (`kl_divergence`).

Current configs use 50 epochs, batch size 256, AdamW with learning rate
`1e-3` and weight decay `1e-4`. The frozen teacher feature map is computed once
per batch and reused by all 16 members. Member losses are summed before
backpropagation; because parameters are disjoint, each member receives the
same gradient it would receive from its own loss without an unintended
`1/16` scale factor.

The best ensemble checkpoint minimizes mean member validation distillation
loss. Reported ensemble accuracy predicts from the arithmetic mean of member
softmax probabilities.

## Inference and OOD Scores

Run:

```bash
distill-ood export-subspace-ensemble-inference --config <config>
distill-ood export-subspace-ensemble-scores --config <config>
```

Inference stores member logits and probabilities with shape
`(samples, 16, classes)`. For member probabilities `p_s`, the raw metrics are

```text
predictive_entropy = H(mean_s p_s)
BALD = H(mean_s p_s) - mean_s H(p_s).
```

Both raw metrics increase with OOD-likeness. Their exported OOD Scores are
their negatives so higher values remain more ID-like.

## Artifacts

```text
<run_dir>/
  resolved_config.json
  subspace_ensemble.pt
  summary.json
  mse_logits/
    best_ensemble.pt
    latest_ensemble.pt
    history.json
    metrics.json
  kl_divergence/
    best_ensemble.pt
    latest_ensemble.pt
    history.json
    metrics.json
  subspace_ensemble_inference/
    manifest.json
    <dataset>/student_<method>_<checkpoint>.pt
  subspace_ensemble_scores/
    manifest.json
    <method>/manifest.json
    <dataset>/student_<method>_<checkpoint>.pt
```

`subspace_ensemble.pt` is shared by both objectives and contains the 16 index
sets, assignment mode, master seed, expected feature shape, and—only for
`pca_gap`—the mean, complete component matrix, whitening standard deviations,
and fit-sample count.

The generic SLURM config runner recognizes paths below
`configs/students/subspace_ensemble/`. In `train-infer` mode it trains both
objectives, exports ensemble predictions, and then exports OOD Scores and
aggregate ROC-AUC/FPR@95 metrics.

## Implementation map

- Config parsing and validation:
  [`config.py`](../../src/distill_ood_detection/config.py)
- Selection, PCA, training, evaluation, and inference:
  [`subspace_ensemble.py`](../../src/distill_ood_detection/distillation/subspace_ensemble.py)
- Training dispatch:
  [`train_student.py`](../../src/distill_ood_detection/experiments/train_student.py)
- Raw inference:
  [`export_subspace_ensemble_inference.py`](../../src/distill_ood_detection/experiments/export_subspace_ensemble_inference.py)
- Score export:
  [`export_subspace_ensemble_scores.py`](../../src/distill_ood_detection/experiments/export_subspace_ensemble_scores.py)
