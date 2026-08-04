# Confusing-Class Channel Replacement: OOD Score Report

Status: completed for ResNet-18, CIFAR-10 and CIFAR-100 ID, and layers 1-4.

All values are `ROC-AUC / FPR@95`. Higher ROC-AUC and lower FPR@95 are better.
Macro values are unweighted averages over MNIST, SVHN, and the opposite CIFAR
dataset.

## Experiment Setup

The student reconstructs 20% of the teacher's pre-GAP channels after they are
replaced by same-index channels from an activation-map prototype of a
confusing class.

| Layer | Activation shape | Replaced channels |
| --- | ---: | ---: |
| `layer1` | `64 x 32 x 32` | 13 / 64 |
| `layer2` | `128 x 16 x 16` | 26 / 128 |
| `layer3` | `256 x 8 x 8` | 51 / 256 |
| `layer4` | `512 x 4 x 4` | 102 / 512 |

For source class \(X\) and donor class \(Y\), every selected channel \(i\) is
normalized to the source-class statistics before it is given to the student:

\[
F(X)_i \leftarrow \mu_{X,i} + \sigma_{X,i}
\frac{F(Y)_i - \mu_{Y,i}}{\max(\sigma_{Y,i}, \epsilon)}.
\]

Each layer-specific corruption bank contains 50 complete activation maps per
class. The three bank inputs are:

- confusing-class pairs: teacher confusion matrix on the official ID test
  split;
- class-channel means and standard deviations: student validation split;
- activation-map prototypes: student training split.

During training, \(X\) is the real class. During inference, \(X\) is the
teacher-predicted class. The student receives only the corrupted feature map
and is trained with MSE on replaced channels.

All runs use AdamW with learning rate `0.001`, weight decay `0.0001`, batch
size 256, 50 epochs, seed 42, 10 evaluation corruption draws, and the
best-validation checkpoint.

The evaluated score families are:

- `-reconstruction_error`: negated reconstruction MSE on replaced channels;
- `improvement`: identity error minus reconstruction error;
- `relative_improvement`: improvement divided by identity error;
- `-identity_error`: negated error before applying the student correction.

## Confusion-Matrix Audit

The confusing class is the largest off-diagonal hard-confusion count in each
real-class row. If a row has no hard errors, the implementation can fall back
to the off-diagonal class with the largest mean teacher probability, but no
fallback was needed in these runs.

| ID dataset | Teacher test accuracy | Hard errors | Rows using hard confusion | Rows using probability fallback | Validation statistics samples | Prototypes per class |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CIFAR-10 | 0.9498 | 502 / 10,000 | 10 / 10 | 0 / 10 | 5,000 | 50 |
| CIFAR-100 | 0.7926 | 2,074 / 10,000 | 100 / 100 | 0 / 100 | 5,000 | 50 |

The saved provenance was checked independently for all eight layer/dataset
banks. Every bank records `confusion_split: test` and the expected 10,000 test
examples, while its channel statistics record the 5,000-example validation
split.

## Main Results

The table selects independently across all four layers and the three
non-identity score families.

| ID dataset | Best macro ROC-AUC | Best macro FPR@95 |
| --- | --- | --- |
| CIFAR-10 | `layer4` `-reconstruction`: `0.839 / 0.603` | `layer4` relative improvement: `0.802 / 0.583` |
| CIFAR-100 | `layer1` improvement: `0.744 / 0.728` | `layer1` improvement: `0.744 / 0.728` |

`layer4` is strongest for CIFAR-10 ID. For CIFAR-100 ID, `layer1`
improvement gives the best macro result, driven by far-OOD performance.

### Per-OOD results at the best macro ROC-AUC operating points

| ID | Layer and score | MNIST | SVHN | Opposite CIFAR | Macro |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | `layer4` `-reconstruction` | 0.882 / 0.571 | 0.790 / 0.636 | 0.845 / 0.601 | 0.839 / 0.603 |
| CIFAR-100 | `layer1` improvement | 0.805 / 0.822 | 0.925 / 0.389 | 0.501 / 0.972 | 0.744 / 0.728 |

The macro-optimal CIFAR-100 configuration does not transfer to near-OOD
CIFAR-10. Its macro strength comes primarily from SVHN.

## Dataset-Specific Results

| ID | OOD dataset | Best ROC-AUC configuration | Best FPR@95 configuration |
| --- | --- | --- | --- |
| CIFAR-10 | MNIST | `layer4` `-reconstruction`: `0.882 / 0.571` | `layer4` relative improvement: `0.872 / 0.489` |
| CIFAR-10 | SVHN | `layer1` improvement: `0.890 / 0.544` | `layer1` improvement: `0.890 / 0.544` |
| CIFAR-10 | CIFAR-100 | `layer4` `-reconstruction`: `0.845 / 0.601` | `layer4` `-reconstruction`: `0.845 / 0.601` |
| CIFAR-100 | MNIST | `layer1` relative improvement: `0.943 / 0.370` | `layer1` relative improvement: `0.943 / 0.370` |
| CIFAR-100 | SVHN | `layer1` improvement: `0.925 / 0.389` | `layer1` improvement: `0.925 / 0.389` |
| CIFAR-100 | CIFAR-10 | `layer4` `-reconstruction`: `0.745 / 0.849` | `layer4` `-reconstruction`: `0.745 / 0.849` |

The optimal layer is dataset-dependent. `layer1` is strongest for SVHN and
for CIFAR-100-ID/MNIST, whereas `layer4` is strongest for both opposite-CIFAR
near-OOD evaluations.

## Near-OOD Results

Selecting directly on the opposite CIFAR dataset gives:

| ID | OOD | Best ROC-AUC | Best FPR@95 |
| --- | --- | --- | --- |
| CIFAR-10 | CIFAR-100 | `layer4` `-reconstruction`: `0.845 / 0.601` | `layer4` `-reconstruction`: `0.845 / 0.601` |
| CIFAR-100 | CIFAR-10 | `layer4` `-reconstruction`: `0.745 / 0.849` | `layer4` `-reconstruction`: `0.745 / 0.849` |

Earlier layers do not improve near-OOD. In particular, CIFAR-100-ID
`layer1` improvement reaches only `0.501 / 0.972` on CIFAR-10.

## Complete Per-Dataset Results

### ResNet-18, CIFAR-10 ID

| Layer | Score | MNIST ROC-AUC / FPR@95 | SVHN ROC-AUC / FPR@95 | CIFAR-100 ROC-AUC / FPR@95 | Macro ROC-AUC / FPR@95 |
| --- | --- | ---: | ---: | ---: | ---: |
| `layer1` | `-reconstruction` | 0.475 / 1.000 | 0.077 / 1.000 | 0.549 / 0.900 | 0.367 / 0.966 |
| `layer1` | Improvement | 0.553 / 0.992 | 0.890 / 0.544 | 0.478 / 0.959 | 0.640 / 0.832 |
| `layer1` | Relative improvement | 0.537 / 0.961 | 0.255 / 0.991 | 0.548 / 0.883 | 0.446 / 0.945 |
| `layer1` | `-identity` | 0.458 / 0.999 | 0.085 / 1.000 | 0.540 / 0.922 | 0.361 / 0.973 |
| `layer2` | `-reconstruction` | 0.492 / 0.998 | 0.091 / 0.999 | 0.498 / 0.941 | 0.360 / 0.980 |
| `layer2` | Improvement | 0.418 / 0.998 | 0.827 / 0.731 | 0.508 / 0.960 | 0.584 / 0.896 |
| `layer2` | Relative improvement | 0.381 / 0.998 | 0.144 / 0.998 | 0.510 / 0.920 | 0.345 / 0.972 |
| `layer2` | `-identity` | 0.521 / 0.996 | 0.102 / 0.999 | 0.496 / 0.951 | 0.373 / 0.982 |
| `layer3` | `-reconstruction` | 0.426 / 0.998 | 0.085 / 0.999 | 0.346 / 0.990 | 0.286 / 0.996 |
| `layer3` | Improvement | 0.771 / 0.856 | 0.844 / 0.641 | 0.724 / 0.860 | 0.779 / 0.786 |
| `layer3` | Relative improvement | 0.804 / 0.768 | 0.622 / 0.941 | 0.697 / 0.869 | 0.708 / 0.859 |
| `layer3` | `-identity` | 0.304 / 1.000 | 0.090 / 1.000 | 0.291 / 0.998 | 0.229 / 0.999 |
| `layer4` | `-reconstruction` | 0.882 / 0.571 | 0.790 / 0.636 | 0.845 / 0.601 | 0.839 / 0.603 |
| `layer4` | Improvement | 0.598 / 0.877 | 0.543 / 0.870 | 0.495 / 0.920 | 0.545 / 0.889 |
| `layer4` | Relative improvement | 0.872 / 0.489 | 0.740 / 0.622 | 0.793 / 0.637 | 0.802 / 0.583 |
| `layer4` | `-identity` | 0.703 / 0.885 | 0.691 / 0.870 | 0.732 / 0.831 | 0.708 / 0.862 |

### ResNet-18, CIFAR-100 ID

| Layer | Score | MNIST ROC-AUC / FPR@95 | SVHN ROC-AUC / FPR@95 | CIFAR-10 ROC-AUC / FPR@95 | Macro ROC-AUC / FPR@95 |
| --- | --- | ---: | ---: | ---: | ---: |
| `layer1` | `-reconstruction` | 0.521 / 1.000 | 0.079 / 1.000 | 0.475 / 0.982 | 0.358 / 0.994 |
| `layer1` | Improvement | 0.805 / 0.822 | 0.925 / 0.389 | 0.501 / 0.972 | 0.744 / 0.728 |
| `layer1` | Relative improvement | 0.943 / 0.370 | 0.627 / 0.920 | 0.444 / 0.986 | 0.671 / 0.759 |
| `layer1` | `-identity` | 0.295 / 1.000 | 0.072 / 1.000 | 0.488 / 0.978 | 0.285 / 0.993 |
| `layer2` | `-reconstruction` | 0.380 / 1.000 | 0.116 / 1.000 | 0.489 / 0.984 | 0.328 / 0.994 |
| `layer2` | Improvement | 0.573 / 0.945 | 0.803 / 0.598 | 0.519 / 0.973 | 0.632 / 0.839 |
| `layer2` | Relative improvement | 0.401 / 0.991 | 0.177 / 0.993 | 0.509 / 0.957 | 0.362 / 0.980 |
| `layer2` | `-identity` | 0.388 / 1.000 | 0.128 / 0.999 | 0.487 / 0.985 | 0.334 / 0.995 |
| `layer3` | `-reconstruction` | 0.088 / 1.000 | 0.101 / 0.999 | 0.531 / 0.963 | 0.240 / 0.987 |
| `layer3` | Improvement | 0.822 / 0.793 | 0.782 / 0.675 | 0.476 / 0.987 | 0.693 / 0.818 |
| `layer3` | Relative improvement | 0.326 / 0.997 | 0.197 / 0.995 | 0.517 / 0.966 | 0.347 / 0.986 |
| `layer3` | `-identity` | 0.090 / 1.000 | 0.109 / 0.999 | 0.529 / 0.964 | 0.243 / 0.988 |
| `layer4` | `-reconstruction` | 0.717 / 0.920 | 0.758 / 0.859 | 0.745 / 0.849 | 0.740 / 0.876 |
| `layer4` | Improvement | 0.626 / 0.911 | 0.509 / 0.958 | 0.541 / 0.900 | 0.559 / 0.923 |
| `layer4` | Relative improvement | 0.765 / 0.888 | 0.701 / 0.916 | 0.705 / 0.852 | 0.724 / 0.885 |
| `layer4` | `-identity` | 0.682 / 0.933 | 0.740 / 0.871 | 0.725 / 0.854 | 0.716 / 0.886 |

## Comparison with Zero-Filled Channel Masking

The closest prior baseline zeros each selected channel rather than replacing
it with a normalized confusing-class prototype. This comparison selects
independently across layers 1-4 and score families for each method.

| ID | Replacement best macro ROC-AUC | Zero-fill best macro ROC-AUC | Replacement best macro FPR@95 | Zero-fill best macro FPR@95 |
| --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | 0.839 (`layer4` `-reconstruction`) | 0.885 (`layer4` relative improvement) | 0.583 (`layer4` relative improvement) | 0.438 (`layer4` improvement) |
| CIFAR-100 | 0.744 (`layer1` improvement) | 0.811 (`layer3` improvement) | 0.728 (`layer1` improvement) | 0.424 (`layer3` improvement) |

The replacement variant substantially improves the best raw
`-reconstruction_error` baseline, with both methods selecting `layer4`:

| ID | Replacement `-reconstruction` | Zero-fill `-reconstruction` | ROC-AUC change | FPR@95 change |
| --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | 0.839 / 0.603 | 0.639 / 0.930 | +0.200 | -0.327 |
| CIFAR-100 | 0.740 / 0.876 | 0.672 / 0.911 | +0.068 | -0.035 |

After selecting the best score and layer for each method, zero-filled masking
remains stronger overall. Its best macro ROC-AUC is higher by `0.046` for
CIFAR-10 and `0.067` for CIFAR-100, while its best macro FPR@95 is lower by
`0.145` and `0.303`, respectively.

For near-OOD, both methods select `layer4`:

| ID | OOD | Replacement best | Zero-fill best | ROC-AUC change | FPR@95 change |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | CIFAR-100 | 0.845 / 0.601 | 0.855 / 0.628 | -0.010 | -0.027 |
| CIFAR-100 | CIFAR-10 | 0.745 / 0.849 | 0.785 / 0.806 | -0.040 | +0.043 |

## Training Summary

| ID dataset | Layer | Best validation loss | Final validation loss | Test loss | Training time |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | `layer1` | 0.019950 | 0.020163 | 0.019939 | 801 s |
| CIFAR-10 | `layer2` | 0.024248 | 0.024334 | 0.024176 | 665 s |
| CIFAR-10 | `layer3` | 0.007266 | 0.007289 | 0.007302 | 651 s |
| CIFAR-10 | `layer4` | 0.018586 | 0.018655 | 0.023911 | 825 s |
| CIFAR-100 | `layer1` | 0.007130 | 0.007159 | 0.007195 | 798 s |
| CIFAR-100 | `layer2` | 0.013051 | 0.013056 | 0.013189 | 656 s |
| CIFAR-100 | `layer3` | 0.006626 | 0.006677 | 0.006664 | 657 s |
| CIFAR-100 | `layer4` | 0.079741 | 0.081389 | 0.125105 | 787 s |

Loss magnitudes are not comparable across layers because channel activation
scales differ. Within a layer, the validation-to-test gap is largest for
CIFAR-100 `layer4`.

## Interpretation

- `layer4` is the only consistently useful near-OOD representation. It is best
  for the opposite CIFAR dataset under both ID choices.
- `layer1` exposes strong far-OOD signals: it gives the best SVHN result for
  both ID datasets and the best MNIST result for CIFAR-100 ID.
- CIFAR-100 `layer1` improvement improves the macro aggregate, but this masks a
  severe near-OOD failure (`0.501 / 0.972` on CIFAR-10).
- Replacing channels with normalized in-distribution prototypes makes raw
  reconstruction magnitude informative at `layer4`, unlike zero-filled
  masking.
- The replacement sweep does not improve the best overall zero-fill results.

## Run and Export Provenance

- SLURM jobs:
  - `layer1`: CIFAR-10 `407872`; CIFAR-100 `407876`;
  - `layer2`: CIFAR-10 `407873`; CIFAR-100 `407877`;
  - `layer3`: CIFAR-10 `407874`; CIFAR-100 `407878`;
  - `layer4`: CIFAR-10 `407875`; CIFAR-100 `407879`.
- Run-directory pattern:
  `runs/students/feature_denoising/feature_masking/cifar_{10,100}/resnet18/confusion_channel_replacement_residual_layer{1,2,3,4}_p020/`.
- Machine-readable OOD exports:
  - `reports/outputs/json/confusion_channel_replacement_default.json`
  - `reports/outputs/json/confusion_channel_replacement_improvement.json`
  - `reports/outputs/json/confusion_channel_replacement_relative_improvement.json`
  - `reports/outputs/json/confusion_channel_replacement_negative_identity_error.json`
- Training-loss export:
  `reports/outputs/json/confusion_channel_replacement_losses.json`.
