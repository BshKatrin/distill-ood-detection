# Direct Confusing-Class Channel Replacement: OOD Score Report

Status: completed for ResNet-18, CIFAR-10 and CIFAR-100 ID, and layers 1-4.

All values are `ROC-AUC / FPR@95`. Higher ROC-AUC and lower FPR@95 are better.
Macro values are unweighted averages over MNIST, SVHN, and the opposite CIFAR
dataset.

## Experiment Setup

The corruption procedure is identical to the residual confusing-class channel
replacement experiment. The student receives a teacher activation map in which
20% of the pre-GAP channels are replaced by same-index channels from an
activation-map prototype of a confusing class.

| Layer | Activation shape | Replaced channels |
| --- | ---: | ---: |
| `layer1` | `64 x 32 x 32` | 13 / 64 |
| `layer2` | `128 x 16 x 16` | 26 / 128 |
| `layer3` | `256 x 8 x 8` | 51 / 256 |
| `layer4` | `512 x 4 x 4` | 102 / 512 |

For source class \(X\) and donor class \(Y\), every selected channel \(i\) is
normalized to the source-class statistics:

\[
F(X)_i \leftarrow \mu_{X,i} + \sigma_{X,i}
\frac{F(Y)_i - \mu_{Y,i}}{\max(\sigma_{Y,i}, \epsilon)}.
\]

Each layer-specific corruption bank contains 50 complete activation maps per
class. Confusing-class pairs come from the teacher confusion matrix on the
official ID test split, class-channel statistics come from the validation
split, and prototypes come from the training split.

During training, \(X\) is the real class. During inference, \(X\) is the
teacher-predicted class.

Unlike the residual variant, the prediction is the raw student output:

\[
\widehat{F(X)} = S(\widetilde{F(X)}),
\]

not \(\widetilde{F(X)} + S(\widetilde{F(X)})\). MSE is calculated directly
between the student output and the clean teacher activation on the replaced
channels. The student architecture is otherwise unchanged, making this a
controlled comparison of direct versus residual reconstruction.

All runs use AdamW with learning rate `0.001`, weight decay `0.0001`, batch
size 256, 50 epochs, seed 42, 10 evaluation corruption draws, and the
best-validation checkpoint.

The evaluated score families are:

- `-reconstruction`: negated direct reconstruction MSE on replaced channels;
- improvement: identity error minus direct reconstruction error;
- relative improvement: improvement divided by identity error;
- `-identity`: negated error in the corrupted input before applying the
  student.

## Confusion-Matrix Audit

The confusing class is the largest off-diagonal hard-confusion count in each
real-class row. No probability fallback was needed.

| ID dataset | Teacher test accuracy | Hard errors | Rows using hard confusion | Rows using probability fallback | Validation statistics samples | Prototypes per class |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CIFAR-10 | 0.9498 | 502 / 10,000 | 10 / 10 | 0 / 10 | 5,000 | 50 |
| CIFAR-100 | 0.7926 | 2,074 / 10,000 | 100 / 100 | 0 / 100 | 5,000 | 50 |

All eight saved banks record `confusion_split: test`, 10,000 confusion
examples, 5,000 validation examples for statistics, and 45,000 training
examples as the prototype source. The direct and residual runs therefore use
the same data provenance.

## Main Results

The table selects independently across all four layers and the three
non-identity score families.

| ID dataset | Best macro ROC-AUC | Best macro FPR@95 |
| --- | --- | --- |
| CIFAR-10 | `layer4` `-reconstruction`: `0.774 / 0.810` | `layer3` improvement: `0.762 / 0.785` |
| CIFAR-100 | `layer1` improvement: `0.729 / 0.753` | `layer1` improvement: `0.729 / 0.753` |

Direct reconstruction preserves the layer preference seen in the residual
experiment for the best macro ROC-AUC: `layer4` for CIFAR-10 and `layer1`
improvement for CIFAR-100. Its absolute OOD performance is lower.

### Per-OOD results at the best macro ROC-AUC operating points

| ID | Layer and score | MNIST | SVHN | Opposite CIFAR | Macro |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | `layer4` `-reconstruction` | 0.759 / 0.895 | 0.740 / 0.826 | 0.825 / 0.710 | 0.774 / 0.810 |
| CIFAR-100 | `layer1` improvement | 0.764 / 0.878 | 0.915 / 0.409 | 0.506 / 0.971 | 0.729 / 0.753 |

As with the residual model, the macro-optimal CIFAR-100 configuration is
driven by far-OOD performance and is nearly random on near-OOD CIFAR-10.

## Dataset-Specific Results

| ID | OOD dataset | Best ROC-AUC configuration | Best FPR@95 configuration |
| --- | --- | --- | --- |
| CIFAR-10 | MNIST | `layer4` `-reconstruction`: `0.759 / 0.895` | `layer3` relative improvement: `0.730 / 0.789` |
| CIFAR-10 | SVHN | `layer1` improvement: `0.900 / 0.513` | `layer1` improvement: `0.900 / 0.513` |
| CIFAR-10 | CIFAR-100 | `layer4` `-reconstruction`: `0.825 / 0.710` | `layer4` `-reconstruction`: `0.825 / 0.710` |
| CIFAR-100 | MNIST | `layer1` relative improvement: `0.939 / 0.394` | `layer1` relative improvement: `0.939 / 0.394` |
| CIFAR-100 | SVHN | `layer1` improvement: `0.915 / 0.409` | `layer1` improvement: `0.915 / 0.409` |
| CIFAR-100 | CIFAR-10 | `layer4` `-reconstruction`: `0.676 / 0.897` | `layer4` `-reconstruction`: `0.676 / 0.897` |

The optimal layer remains dataset-dependent: early layers are strongest for
SVHN and CIFAR-100-ID/MNIST, while `layer4` is strongest for both opposite
CIFAR evaluations.

## Near-OOD Results

| ID | OOD | Best ROC-AUC | Best FPR@95 |
| --- | --- | --- | --- |
| CIFAR-10 | CIFAR-100 | `layer4` `-reconstruction`: `0.825 / 0.710` | `layer4` `-reconstruction`: `0.825 / 0.710` |
| CIFAR-100 | CIFAR-10 | `layer4` `-reconstruction`: `0.676 / 0.897` | `layer4` `-reconstruction`: `0.676 / 0.897` |

Earlier layers do not improve near-OOD. CIFAR-100-ID `layer1` improvement,
despite being macro-optimal, reaches only `0.506 / 0.971` on CIFAR-10.

## Complete Per-Dataset Results

### ResNet-18, CIFAR-10 ID

| Layer | Score | MNIST ROC-AUC / FPR@95 | SVHN ROC-AUC / FPR@95 | CIFAR-100 ROC-AUC / FPR@95 | Macro ROC-AUC / FPR@95 |
| --- | --- | ---: | ---: | ---: | ---: |
| `layer1` | `-reconstruction` | 0.509 / 1.000 | 0.081 / 0.999 | 0.558 / 0.886 | 0.383 / 0.962 |
| `layer1` | Improvement | 0.554 / 0.995 | 0.900 / 0.513 | 0.475 / 0.961 | 0.643 / 0.823 |
| `layer1` | Relative improvement | 0.561 / 0.982 | 0.210 / 0.995 | 0.563 / 0.869 | 0.444 / 0.948 |
| `layer1` | `-identity` | 0.458 / 0.999 | 0.085 / 1.000 | 0.540 / 0.922 | 0.361 / 0.973 |
| `layer2` | `-reconstruction` | 0.484 / 1.000 | 0.106 / 0.999 | 0.515 / 0.913 | 0.368 / 0.970 |
| `layer2` | Improvement | 0.461 / 0.998 | 0.869 / 0.604 | 0.513 / 0.959 | 0.614 / 0.854 |
| `layer2` | Relative improvement | 0.439 / 0.999 | 0.205 / 0.978 | 0.535 / 0.895 | 0.393 / 0.957 |
| `layer2` | `-identity` | 0.521 / 0.996 | 0.102 / 0.999 | 0.496 / 0.951 | 0.373 / 0.982 |
| `layer3` | `-reconstruction` | 0.470 / 0.991 | 0.067 / 0.999 | 0.346 / 0.980 | 0.294 / 0.990 |
| `layer3` | Improvement | 0.730 / 0.896 | 0.853 / 0.577 | 0.704 / 0.883 | 0.762 / 0.785 |
| `layer3` | Relative improvement | 0.730 / 0.789 | 0.253 / 0.981 | 0.570 / 0.910 | 0.518 / 0.893 |
| `layer3` | `-identity` | 0.304 / 1.000 | 0.090 / 1.000 | 0.291 / 0.998 | 0.229 / 0.999 |
| `layer4` | `-reconstruction` | 0.759 / 0.895 | 0.740 / 0.826 | 0.825 / 0.710 | 0.774 / 0.810 |
| `layer4` | Improvement | 0.303 / 0.996 | 0.316 / 0.990 | 0.277 / 0.991 | 0.298 / 0.992 |
| `layer4` | Relative improvement | 0.560 / 0.977 | 0.570 / 0.921 | 0.652 / 0.913 | 0.594 / 0.937 |
| `layer4` | `-identity` | 0.703 / 0.885 | 0.691 / 0.870 | 0.732 / 0.831 | 0.708 / 0.862 |

### ResNet-18, CIFAR-100 ID

| Layer | Score | MNIST ROC-AUC / FPR@95 | SVHN ROC-AUC / FPR@95 | CIFAR-10 ROC-AUC / FPR@95 | Macro ROC-AUC / FPR@95 |
| --- | --- | ---: | ---: | ---: | ---: |
| `layer1` | `-reconstruction` | 0.664 / 1.000 | 0.049 / 1.000 | 0.468 / 0.981 | 0.394 / 0.993 |
| `layer1` | Improvement | 0.764 / 0.878 | 0.915 / 0.409 | 0.506 / 0.971 | 0.729 / 0.753 |
| `layer1` | Relative improvement | 0.939 / 0.394 | 0.309 / 0.976 | 0.449 / 0.984 | 0.566 / 0.785 |
| `layer1` | `-identity` | 0.295 / 1.000 | 0.072 / 1.000 | 0.488 / 0.978 | 0.285 / 0.993 |
| `layer2` | `-reconstruction` | 0.458 / 1.000 | 0.116 / 1.000 | 0.468 / 0.985 | 0.347 / 0.995 |
| `layer2` | Improvement | 0.631 / 0.920 | 0.848 / 0.504 | 0.504 / 0.974 | 0.661 / 0.800 |
| `layer2` | Relative improvement | 0.592 / 0.968 | 0.321 / 0.943 | 0.442 / 0.980 | 0.452 / 0.963 |
| `layer2` | `-identity` | 0.388 / 1.000 | 0.128 / 0.999 | 0.487 / 0.985 | 0.334 / 0.995 |
| `layer3` | `-reconstruction` | 0.051 / 1.000 | 0.102 / 0.999 | 0.484 / 0.977 | 0.212 / 0.992 |
| `layer3` | Improvement | 0.841 / 0.897 | 0.848 / 0.533 | 0.438 / 0.988 | 0.709 / 0.806 |
| `layer3` | Relative improvement | 0.123 / 1.000 | 0.276 / 0.960 | 0.419 / 0.985 | 0.273 / 0.982 |
| `layer3` | `-identity` | 0.090 / 1.000 | 0.109 / 0.999 | 0.529 / 0.964 | 0.243 / 0.988 |
| `layer4` | `-reconstruction` | 0.739 / 0.856 | 0.680 / 0.918 | 0.676 / 0.897 | 0.698 / 0.891 |
| `layer4` | Improvement | 0.340 / 1.000 | 0.259 / 0.999 | 0.275 / 0.985 | 0.292 / 0.995 |
| `layer4` | Relative improvement | 0.591 / 0.945 | 0.390 / 0.981 | 0.402 / 0.981 | 0.461 / 0.969 |
| `layer4` | `-identity` | 0.682 / 0.933 | 0.740 / 0.871 | 0.725 / 0.854 | 0.716 / 0.886 |

## Comparison with Residual Reconstruction

This comparison selects independently across layers 1-4 and the three
non-identity score families. The corruption, data splits, student architecture,
training schedule, and evaluation procedure are held constant.

| ID | Direct best macro ROC-AUC | Residual best macro ROC-AUC | Direct best macro FPR@95 | Residual best macro FPR@95 |
| --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | 0.774 (`layer4` `-reconstruction`) | 0.839 (`layer4` `-reconstruction`) | 0.785 (`layer3` improvement) | 0.583 (`layer4` relative improvement) |
| CIFAR-100 | 0.729 (`layer1` improvement) | 0.744 (`layer1` improvement) | 0.753 (`layer1` improvement) | 0.728 (`layer1` improvement) |

Residual reconstruction is better under every macro selection criterion. Its
best ROC-AUC exceeds direct reconstruction by `0.065` for CIFAR-10 and `0.015`
for CIFAR-100; its best FPR@95 is lower by `0.202` and `0.025`, respectively.

For near-OOD, both variants select `layer4` `-reconstruction`:

| ID | OOD | Direct | Residual | Direct minus residual ROC-AUC | Direct minus residual FPR@95 |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | CIFAR-100 | 0.825 / 0.710 | 0.845 / 0.601 | -0.020 | +0.109 |
| CIFAR-100 | CIFAR-10 | 0.676 / 0.897 | 0.745 / 0.849 | -0.069 | +0.049 |

The identical `-identity` rows in the direct and residual reports provide an
additional consistency check: the saved corruption baseline is unchanged, and
only the student's reconstruction rule differs.

## Training Summary

| ID dataset | Layer | Best validation loss | Final validation loss | Test loss | Training time |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | `layer1` | 0.008558 | 0.008558 | 0.008424 | 829 s |
| CIFAR-10 | `layer2` | 0.009191 | 0.009226 | 0.009166 | 654 s |
| CIFAR-10 | `layer3` | 0.003055 | 0.003071 | 0.003059 | 652 s |
| CIFAR-10 | `layer4` | 0.002770 | 0.002946 | 0.003121 | 818 s |
| CIFAR-100 | `layer1` | 0.002890 | 0.002890 | 0.002905 | 804 s |
| CIFAR-100 | `layer2` | 0.004224 | 0.004224 | 0.004249 | 659 s |
| CIFAR-100 | `layer3` | 0.002873 | 0.002876 | 0.002872 | 660 s |
| CIFAR-100 | `layer4` | 0.025578 | 0.025578 | 0.029254 | 779 s |

Loss magnitudes are not comparable across layers because channel activation
scales differ. The largest validation-to-test increase occurs for CIFAR-100
`layer4`.

## Interpretation

- Direct reconstruction underperforms residual reconstruction overall and on
  both near-OOD comparisons.
- `layer4` remains the strongest representation for opposite-CIFAR near-OOD
  detection, but removing the residual path weakens it substantially.
- Early-layer improvement remains effective for far-OOD SVHN and MNIST.
- The CIFAR-100 macro aggregate remains misleading for near-OOD: its best
  macro configuration is nearly random on CIFAR-10.
- The residual skip connection is useful here: it preserves unmodified
  activation content and lets the student learn a correction, whereas the
  direct model must reconstruct selected target channels from scratch.

## Run and Export Provenance

- SLURM jobs:
  - `layer1`: CIFAR-10 `407885`; CIFAR-100 `407889`;
  - `layer2`: CIFAR-10 `407886`; CIFAR-100 `407890`;
  - `layer3`: CIFAR-10 `407887`; CIFAR-100 `407891`;
  - `layer4`: CIFAR-10 `407888`; CIFAR-100 `407892`.
- Run-directory pattern:
  `runs/students/feature_denoising/feature_masking/cifar_{10,100}/resnet18/confusion_channel_replacement_direct_layer{1,2,3,4}_p020/`.
- Machine-readable OOD exports:
  - `reports/outputs/json/confusion_channel_replacement_direct_default.json`
  - `reports/outputs/json/confusion_channel_replacement_direct_improvement.json`
  - `reports/outputs/json/confusion_channel_replacement_direct_relative_improvement.json`
  - `reports/outputs/json/confusion_channel_replacement_direct_negative_identity_error.json`
- Training-loss export:
  `reports/outputs/json/confusion_channel_replacement_direct_losses.json`.
