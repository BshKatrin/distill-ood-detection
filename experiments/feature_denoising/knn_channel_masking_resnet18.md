# Masked k-NN Feature Denoising: OOD Metrics Report

Status: completed for ResNet-18, CIFAR-10 and CIFAR-100 ID, layers 3 and 4.

All values are `ROC-AUC / FPR@95`. Higher ROC-AUC and lower FPR@95 are
better. Macro values are unweighted averages over MNIST, SVHN, and the
opposite CIFAR dataset.

## Question

Can an exact non-parametric regressor replace the learned channel-denoising
student? Given a corrupted pre-GAP teacher activation, neighbor search uses
only unmasked channels and predicts every hidden channel map by averaging its
values across the selected clean ID neighbors.

## Experiment Matrix

- ID datasets: CIFAR-10 and CIFAR-100.
- Backbone: ResNet-18.
- Layers: `layer3` (`256x8x8`) and `layer4` (`512x4x4`).
- References: 50,000 clean ID training activation maps.
- Channel mask probability: `0.2`.
- Neighbors: `k=10`.
- Prediction: arithmetic mean of clean neighbor maps on hidden channels.
- Evaluation draws: 10 independent masks.
- Distance: exact squared L2 over unmasked pre-GAP channels only.
- Seed: `42`.

For query map `q`, reference map `x`, and channel keep mask `m`, search uses:

```text
d(q, x; m) = sum(m * (q - x)^2)
```

The implementation calculates this exactly with batched GPU matrix
multiplication and chunked global top-k merging. It does not train or load a
student.

## Score Families

Four ID-oriented OOD Scores were evaluated:

- `-reconstruction_error`: negated hidden-channel MSE after k-NN prediction;
- `improvement`: zero-fill identity error minus reconstruction error;
- `relative_improvement`: improvement divided by identity error;
- `-identity_error`: negated hidden-channel error before k-NN prediction.

The reconstruction score is the canonical Feature Denoising score. The other
families diagnose whether results arise from the neighbor prediction or from
the scale of the masked teacher activation itself.

## Main Results

### Canonical reconstruction-error score

| ID | Layer | MNIST | SVHN | Opposite CIFAR | Macro |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | layer3 | .394 / .990 | .114 / .996 | .519 / .932 | .342 / .973 |
| CIFAR-10 | layer4 | .820 / .840 | .801 / .859 | .865 / .615 | .829 / .771 |
| CIFAR-100 | layer3 | .063 / 1.000 | .156 / 1.000 | .615 / .938 | .278 / .979 |
| CIFAR-100 | layer4 | .770 / .813 | .801 / .808 | .752 / .851 | .774 / .824 |

Layer4 supplies a useful ranking signal for every OOD dataset. Layer3 is near
chance for the opposite CIFAR dataset and directionally inverted for MNIST
and SVHN. Consequently, layer3 raw reconstruction error should not be treated
as an OOD Score in its current form.

### Macro comparison across score families

| ID | Layer | `-reconstruction` | Improvement | Relative improvement | `-identity` |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | layer3 | .342 / .973 | .821 / .867 | .733 / .905 | .149 / 1.000 |
| CIFAR-10 | layer4 | .829 / .771 | **.898 / .398** | .888 / .549 | .121 / .999 |
| CIFAR-100 | layer3 | .278 / .979 | .550 / .974 | .431 / .974 | .192 / .989 |
| CIFAR-100 | layer4 | .774 / .824 | .761 / .827 | **.782 / .823** | .292 / .999 |

Improvement is strongest for CIFAR-10, especially at layer4. Relative
improvement has the highest macro ROC-AUC for CIFAR-100, although it is only
slightly above raw reconstruction error and does not materially improve
FPR@95.

The very poor identity score shows that hidden-channel activation magnitude
alone is anti-correlated with the desired OOD ranking. Improvement scores can
remove part of this baseline, but they cannot rescue CIFAR-100 layer3.

## Layer4 Per-Dataset Results

Layer4 is the only consistently viable retrieval layer. These tables expose
all non-identity score families without selecting a score separately for each
OOD dataset.

### CIFAR-10 ID

| Score | MNIST | SVHN | CIFAR-100 | Macro |
| --- | ---: | ---: | ---: | ---: |
| `-reconstruction` | .820 / .840 | .801 / .859 | .865 / .615 | .829 / .771 |
| Improvement | **.975 / .138** | **.900 / .438** | .818 / .618 | **.898 / .398** |
| Relative improvement | .915 / .461 | .870 / .631 | **.881 / .555** | .888 / .549 |

Improvement gives the best macro result and substantially strengthens far-OOD
detection. Relative improvement is best on near-OOD CIFAR-100, but the gain is
small and must not be used as post-hoc OOD-test score selection.

### CIFAR-100 ID

| Score | MNIST | SVHN | CIFAR-10 | Macro |
| --- | ---: | ---: | ---: | ---: |
| `-reconstruction` | **.770 / .813** | .801 / .808 | .752 / .851 | .774 / .824 |
| Improvement | .668 / .958 | .830 / .706 | .784 / .817 | .761 / .827 |
| Relative improvement | .726 / .940 | **.837 / .712** | **.784 / .819** | **.782 / .823** |

No score dominates for CIFAR-100. Raw reconstruction is strongest on MNIST,
whereas improvement-based scores are stronger on SVHN and near-OOD CIFAR-10.
This instability argues for predeclaring a score using ID-only validation
diagnostics before any larger sweep.

## Comparison with the Learned Residual Student

At the same ResNet-18 layer4, exact k-NN improves macro ROC-AUC over the
learned channel-masked residual student for every comparable score family. It
also improves FPR@95 except for CIFAR-10 relative improvement:

| ID | Score | Residual student | Exact k-NN | ROC-AUC change |
| --- | --- | ---: | ---: | ---: |
| CIFAR-10 | `-reconstruction` | .639 / .930 | .829 / .771 | +.190 |
| CIFAR-10 | Improvement | .882 / .438 | .898 / .398 | +.016 |
| CIFAR-10 | Relative improvement | .885 / .525 | .888 / .549 | +.003 |
| CIFAR-100 | `-reconstruction` | .672 / .911 | .774 / .824 | +.102 |
| CIFAR-100 | Improvement | .734 / .855 | .761 / .827 | +.027 |
| CIFAR-100 | Relative improvement | .770 / .861 | .782 / .823 | +.012 |

For CIFAR-10, k-NN layer4 improvement also exceeds the best reported learned
ResNet-18 channel/spatial result in both macro metrics: `.898 / .398` versus
approximately `.885 / .438`–`.561` depending on the selection criterion.

For CIFAR-100, the learned channel student at layer3 remains stronger overall:
`.811 / .424` with improvement, compared with the best k-NN macro result of
`.782 / .823` at layer4. The k-NN layer3 failure is therefore the main obstacle
to replacing the learned student across layers.

## Runtime and Feasibility

Successful GPU job `408707` ran two configurations concurrently on two NVIDIA
TITAN RTX GPUs and completed all four configurations in 5m12s. Each config
includes the ID test set and all three configured OOD datasets with 10 mask
draws.

| Layer | Time per ID configuration |
| --- | ---: |
| layer3 | approximately 2m35s |
| layer4 | approximately 1m36s |

The experiment confirms that exact masked pre-GAP search is practical for
ResNet-18 layers 3 and 4 at CIFAR scale. Computation time is not the reason to
reject layer3; its OOD geometry is.

## Conclusions

1. Exact masked k-NN is a credible replacement for the learned student at
   ResNet-18 layer4.
2. The canonical layer4 reconstruction score reaches macro ROC-AUC `.829` for
   CIFAR-10 and `.774` for CIFAR-100.
3. Improvement scoring raises CIFAR-10 layer4 to `.898 / .398` and is the best
   result in this experiment.
4. Layer3 is not viable without feature scaling, normalization, or a different
   distance geometry.
5. Near-OOD FPR@95 remains high, especially for CIFAR-100 ID, despite useful
   ROC-AUC rankings.

The next controlled experiment should remain at layer4 and sweep `k` and mask
probability. Score selection and hyperparameter selection should use a
predeclared ID-only criterion rather than OOD test metrics.

## Artifacts

- Default metrics:
  `reports/outputs/json/knn_channel_masking_resnet18_ood_metrics.json`
- Improvement metrics:
  `reports/outputs/json/knn_channel_masking_resnet18_improvement.json`
- Relative-improvement metrics:
  `reports/outputs/json/knn_channel_masking_resnet18_relative_improvement.json`
- Identity metrics:
  `reports/outputs/json/knn_channel_masking_resnet18_negative_identity_error.json`
- Per-sample scores and per-draw neighbor IDs/distances:
  `runs/students/feature_denoising/knn_channel_masking/`
