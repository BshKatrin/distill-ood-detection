# Residual Feature Masking: OOD Score Report

Status: completed for spatial-block and channel masking, ResNet-18 and
ResNet-50, CIFAR-10 and CIFAR-100 ID, layers 1-4.

All values are `ROC-AUC / FPR@95`. Higher ROC-AUC and lower FPR@95 are better.
Macro values are unweighted averages over MNIST, SVHN, and the opposite CIFAR
dataset.

## Experiment Matrix

| Variant | Corruption | Student input | Residual baseline |
| --- | --- | --- | --- |
| Spatial block | One uniformly sampled square block, channel-mean fill | Corrupted feature plus mask channel | Mean-filled feature |
| Channel | Each channel hidden independently with probability `0.2`, zero fill | Corrupted feature only | Zero-filled feature |

Both variants add the CNN correction to the corrupted feature and calculate
loss and OOD diagnostics only on hidden coordinates. ResNet-50 uses a hidden
width of 64 and batch size 128; ResNet-18 uses widths 16, 32, 64, and 128 with
batch size 256. All runs use AdamW, 50 epochs, seed 42, 10 evaluation draws,
and the best-validation checkpoint.

The evaluated score families are:

- `-reconstruction_error`: primary negated masked reconstruction MSE;
- `improvement`: identity error minus reconstruction error;
- `relative_improvement`: improvement divided by identity error;
- `-identity_error`: negated error before applying the student correction.

## Main Results

The table reports the best macro ROC-AUC configuration and the best macro
FPR@95 configuration across both masking variants, layers, and the three
non-identity score families.

| Backbone | ID dataset | Best macro ROC-AUC | Best macro FPR@95 |
| --- | --- | --- | --- |
| ResNet-18 | CIFAR-10 | Spatial improvement, layer3: `0.885 / 0.561` | Channel improvement, layer4: `0.882 / 0.438` |
| ResNet-18 | CIFAR-100 | Channel improvement, layer3: `0.811 / 0.424` | Channel improvement, layer3: `0.811 / 0.424` |
| ResNet-50 | CIFAR-10 | Channel relative improvement, layer4: `0.879 / 0.510` | Channel relative improvement, layer4: `0.879 / 0.510` |
| ResNet-50 | CIFAR-100 | Channel improvement, layer3: `0.807 / 0.463` | Channel improvement, layer3: `0.807 / 0.463` |

Channel masking is the stronger overall variant. It wins both selection
criteria in three of four backbone/ID settings. For ResNet-18 with CIFAR-10
ID, spatial layer3 has only a `0.00012` macro ROC-AUC advantage over channel
relative-improvement layer4, while channel improvement layer4 reduces macro
FPR@95 from `0.561` to `0.438`.

### Per-OOD results at the best macro ROC-AUC operating points

| Backbone | ID | Variant and score | Layer | MNIST | SVHN | Opposite CIFAR | Macro |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ResNet-18 | CIFAR-10 | Spatial improvement | 3 | 0.863 / 0.770 | 0.959 / 0.237 | 0.833 / 0.675 | 0.885 / 0.561 |
| ResNet-18 | CIFAR-100 | Channel improvement | 3 | 0.994 / 0.022 | 0.936 / 0.283 | 0.504 / 0.967 | 0.811 / 0.424 |
| ResNet-50 | CIFAR-10 | Channel relative improvement | 4 | 0.955 / 0.251 | 0.862 / 0.607 | 0.822 / 0.671 | 0.879 / 0.510 |
| ResNet-50 | CIFAR-100 | Channel improvement | 3 | 0.991 / 0.038 | 0.909 / 0.392 | 0.522 / 0.958 | 0.807 / 0.463 |

## Near-OOD Results

This table selects directly on the opposite CIFAR dataset rather than on the
three-dataset macro.

| Backbone | ID | Best near-OOD ROC-AUC | Best near-OOD FPR@95 |
| --- | --- | --- | --- |
| ResNet-18 | CIFAR-10 | Channel relative improvement, layer4: `0.855 / 0.628` | Channel relative improvement, layer4: `0.855 / 0.628` |
| ResNet-18 | CIFAR-100 | Channel relative improvement, layer4: `0.785 / 0.806` | Channel relative improvement, layer4: `0.785 / 0.806` |
| ResNet-50 | CIFAR-10 | Channel relative improvement, layer4: `0.822 / 0.671` | Channel relative improvement, layer4: `0.822 / 0.671` |
| ResNet-50 | CIFAR-100 | Spatial relative improvement, layer4: `0.771 / 0.817` | Spatial improvement, layer4: `0.732 / 0.806` |

Near-OOD remains the limiting case. In particular, the macro-optimal
CIFAR-100 configurations derive most of their strength from MNIST and SVHN:
their CIFAR-10 results are only `0.504 / 0.967` for ResNet-18 and
`0.522 / 0.958` for ResNet-50. Selecting layer4 relative improvement improves
near-OOD ROC-AUC, but FPR@95 remains at least `0.806`.

## Dataset-Specific Bottlenecks

The best configuration is not consistent across OOD datasets. This table
selects independently for every backbone, ID dataset, and OOD dataset across
both masking variants, all layers, and all four score families.

| Backbone | ID | OOD dataset | Best ROC-AUC configuration | Best FPR@95 configuration |
| --- | --- | --- | --- | --- |
| ResNet-18 | CIFAR-10 | MNIST | Channel improvement, layer4: `0.979 / 0.115` | Channel improvement, layer4: `0.979 / 0.115` |
| ResNet-18 | CIFAR-10 | SVHN | Channel improvement, layer3: `0.962 / 0.217` | Channel improvement, layer3: `0.962 / 0.217` |
| ResNet-18 | CIFAR-10 | CIFAR-100 | Channel relative improvement, layer4: `0.855 / 0.628` | Channel relative improvement, layer4: `0.855 / 0.628` |
| ResNet-18 | CIFAR-100 | MNIST | Channel improvement, layer3: `0.994 / 0.022` | Channel improvement, layer3: `0.994 / 0.022` |
| ResNet-18 | CIFAR-100 | SVHN | Spatial improvement, layer1: `0.966 / 0.161` | Spatial improvement, layer1: `0.966 / 0.161` |
| ResNet-18 | CIFAR-100 | CIFAR-10 | Channel relative improvement, layer4: `0.785 / 0.806` | Channel relative improvement, layer4: `0.785 / 0.806` |
| ResNet-50 | CIFAR-10 | MNIST | Channel relative improvement, layer4: `0.955 / 0.251` | Channel relative improvement, layer4: `0.955 / 0.251` |
| ResNet-50 | CIFAR-10 | SVHN | Channel improvement, layer3: `0.911 / 0.404` | Channel improvement, layer3: `0.911 / 0.404` |
| ResNet-50 | CIFAR-10 | CIFAR-100 | Channel relative improvement, layer4: `0.822 / 0.671` | Channel relative improvement, layer4: `0.822 / 0.671` |
| ResNet-50 | CIFAR-100 | MNIST | Channel improvement, layer3: `0.991 / 0.038` | Channel improvement, layer3: `0.991 / 0.038` |
| ResNet-50 | CIFAR-100 | SVHN | Channel improvement, layer1: `0.916 / 0.441` | Channel improvement, layer3: `0.909 / 0.392` |
| ResNet-50 | CIFAR-100 | CIFAR-10 | Spatial relative improvement, layer4: `0.771 / 0.817` | Spatial improvement, layer4: `0.732 / 0.806` |

The bottleneck is specifically the opposite CIFAR dataset, not generic
far-OOD detection. MNIST reaches up to `0.994 / 0.022` and SVHN reaches up to
`0.966 / 0.161`, whereas the best opposite-CIFAR FPR@95 is only `0.628` for
CIFAR-10 ID and `0.806` for CIFAR-100 ID. Layer selection is also
dataset-dependent: layer3 is usually strongest for SVHN and macro improvement,
while layer4 relative improvement is consistently strongest for near-OOD.

## Complete Per-Dataset Results

The following tables expose every OOD dataset for every layer and score. The
macro columns are retained only as a compact cross-dataset reference.

### Spatial Block: ResNet-18, CIFAR-10 ID

| Layer | Score | MNIST ROC-AUC / FPR@95 | SVHN ROC-AUC / FPR@95 | CIFAR-100 ROC-AUC / FPR@95 | Macro ROC-AUC / FPR@95 |
| --- | --- | ---: | ---: | ---: | ---: |
| layer1 | `-reconstruction` | 0.439 / 0.998 | 0.059 / 1.000 | 0.532 / 0.910 | 0.343 / 0.969 |
| layer1 | Improvement | 0.451 / 0.985 | 0.918 / 0.441 | 0.491 / 0.937 | 0.620 / 0.787 |
| layer1 | Relative improvement | 0.324 / 0.994 | 0.183 / 0.998 | 0.538 / 0.911 | 0.348 / 0.968 |
| layer1 | `-identity` | 0.503 / 0.987 | 0.059 / 0.999 | 0.521 / 0.914 | 0.361 / 0.967 |
| layer2 | `-reconstruction` | 0.612 / 0.945 | 0.083 / 0.999 | 0.484 / 0.933 | 0.393 / 0.959 |
| layer2 | Improvement | 0.215 / 0.992 | 0.923 / 0.283 | 0.570 / 0.889 | 0.569 / 0.722 |
| layer2 | Relative improvement | 0.223 / 1.000 | 0.432 / 0.983 | 0.568 / 0.897 | 0.407 / 0.960 |
| layer2 | `-identity` | 0.712 / 0.851 | 0.069 / 0.998 | 0.459 / 0.950 | 0.413 / 0.933 |
| layer3 | `-reconstruction` | 0.493 / 0.974 | 0.067 / 0.999 | 0.336 / 0.986 | 0.299 / 0.986 |
| layer3 | Improvement | 0.863 / 0.770 | 0.959 / 0.237 | 0.833 / 0.675 | 0.885 / 0.561 |
| layer3 | Relative improvement | 0.919 / 0.548 | 0.731 / 0.908 | 0.822 / 0.737 | 0.824 / 0.731 |
| layer3 | `-identity` | 0.254 / 1.000 | 0.035 / 1.000 | 0.209 / 0.999 | 0.166 / 1.000 |
| layer4 | `-reconstruction` | 0.571 / 0.952 | 0.440 / 0.979 | 0.711 / 0.841 | 0.574 / 0.924 |
| layer4 | Improvement | 0.933 / 0.309 | 0.874 / 0.486 | 0.787 / 0.700 | 0.865 / 0.498 |
| layer4 | Relative improvement | 0.913 / 0.438 | 0.801 / 0.706 | 0.837 / 0.630 | 0.850 / 0.592 |
| layer4 | `-identity` | 0.068 / 1.000 | 0.125 / 1.000 | 0.220 / 0.998 | 0.138 / 0.999 |

### Spatial Block: ResNet-18, CIFAR-100 ID

| Layer | Score | MNIST ROC-AUC / FPR@95 | SVHN ROC-AUC / FPR@95 | CIFAR-10 ROC-AUC / FPR@95 | Macro ROC-AUC / FPR@95 |
| --- | --- | ---: | ---: | ---: | ---: |
| layer1 | `-reconstruction` | 0.412 / 0.999 | 0.043 / 1.000 | 0.478 / 0.971 | 0.311 / 0.990 |
| layer1 | Improvement | 0.689 / 0.888 | 0.966 / 0.161 | 0.510 / 0.958 | 0.722 / 0.669 |
| layer1 | Relative improvement | 0.655 / 0.828 | 0.418 / 0.989 | 0.475 / 0.969 | 0.516 / 0.929 |
| layer1 | `-identity` | 0.336 / 0.998 | 0.033 / 1.000 | 0.484 / 0.968 | 0.285 / 0.989 |
| layer2 | `-reconstruction` | 0.500 / 0.992 | 0.117 / 0.999 | 0.489 / 0.975 | 0.368 / 0.989 |
| layer2 | Improvement | 0.312 / 0.990 | 0.850 / 0.568 | 0.524 / 0.970 | 0.562 / 0.842 |
| layer2 | Relative improvement | 0.252 / 0.999 | 0.310 / 0.996 | 0.513 / 0.957 | 0.358 / 0.984 |
| layer2 | `-identity` | 0.594 / 0.968 | 0.115 / 0.999 | 0.481 / 0.977 | 0.396 / 0.981 |
| layer3 | `-reconstruction` | 0.121 / 1.000 | 0.110 / 0.999 | 0.497 / 0.967 | 0.242 / 0.988 |
| layer3 | Improvement | 0.916 / 0.332 | 0.881 / 0.436 | 0.506 / 0.960 | 0.768 / 0.576 |
| layer3 | Relative improvement | 0.722 / 0.825 | 0.432 / 0.975 | 0.514 / 0.962 | 0.556 / 0.921 |
| layer3 | `-identity` | 0.091 / 1.000 | 0.099 / 0.999 | 0.494 / 0.970 | 0.228 / 0.990 |
| layer4 | `-reconstruction` | 0.594 / 0.922 | 0.370 / 0.981 | 0.481 / 0.967 | 0.482 / 0.957 |
| layer4 | Improvement | 0.633 / 0.967 | 0.801 / 0.776 | 0.766 / 0.830 | 0.733 / 0.857 |
| layer4 | Relative improvement | 0.689 / 0.960 | 0.744 / 0.922 | 0.769 / 0.821 | 0.734 / 0.901 |
| layer4 | `-identity` | 0.372 / 0.999 | 0.199 / 1.000 | 0.237 / 0.998 | 0.269 / 0.999 |

### Spatial Block: ResNet-50, CIFAR-10 ID

| Layer | Score | MNIST ROC-AUC / FPR@95 | SVHN ROC-AUC / FPR@95 | CIFAR-100 ROC-AUC / FPR@95 | Macro ROC-AUC / FPR@95 |
| --- | --- | ---: | ---: | ---: | ---: |
| layer1 | `-reconstruction` | 0.402 / 0.997 | 0.049 / 1.000 | 0.538 / 0.902 | 0.330 / 0.966 |
| layer1 | Improvement | 0.115 / 1.000 | 0.879 / 0.570 | 0.455 / 0.945 | 0.483 / 0.838 |
| layer1 | Relative improvement | 0.060 / 1.000 | 0.144 / 0.999 | 0.492 / 0.946 | 0.232 / 0.982 |
| layer1 | `-identity` | 0.815 / 0.954 | 0.082 / 0.999 | 0.545 / 0.899 | 0.481 / 0.951 |
| layer2 | `-reconstruction` | 0.507 / 0.987 | 0.067 / 1.000 | 0.506 / 0.922 | 0.360 / 0.970 |
| layer2 | Improvement | 0.131 / 1.000 | 0.696 / 0.906 | 0.526 / 0.916 | 0.451 / 0.941 |
| layer2 | Relative improvement | 0.118 / 1.000 | 0.165 / 0.999 | 0.537 / 0.920 | 0.274 / 0.973 |
| layer2 | `-identity` | 0.746 / 0.916 | 0.103 / 0.999 | 0.490 / 0.931 | 0.446 / 0.949 |
| layer3 | `-reconstruction` | 0.531 / 0.943 | 0.118 / 0.996 | 0.430 / 0.959 | 0.360 / 0.966 |
| layer3 | Improvement | 0.728 / 0.844 | 0.886 / 0.556 | 0.730 / 0.787 | 0.781 / 0.729 |
| layer3 | Relative improvement | 0.810 / 0.832 | 0.541 / 0.916 | 0.719 / 0.825 | 0.690 / 0.857 |
| layer3 | `-identity` | 0.399 / 0.996 | 0.084 / 0.999 | 0.338 / 0.990 | 0.273 / 0.995 |
| layer4 | `-reconstruction` | 0.457 / 0.993 | 0.242 / 0.996 | 0.628 / 0.910 | 0.442 / 0.966 |
| layer4 | Improvement | 0.835 / 0.567 | 0.867 / 0.542 | 0.724 / 0.780 | 0.809 / 0.630 |
| layer4 | Relative improvement | 0.793 / 0.799 | 0.643 / 0.941 | 0.771 / 0.776 | 0.736 / 0.839 |
| layer4 | `-identity` | 0.169 / 0.996 | 0.125 / 0.999 | 0.297 / 0.993 | 0.197 / 0.996 |

### Spatial Block: ResNet-50, CIFAR-100 ID

| Layer | Score | MNIST ROC-AUC / FPR@95 | SVHN ROC-AUC / FPR@95 | CIFAR-10 ROC-AUC / FPR@95 | Macro ROC-AUC / FPR@95 |
| --- | --- | ---: | ---: | ---: | ---: |
| layer1 | `-reconstruction` | 0.477 / 0.992 | 0.039 / 1.000 | 0.476 / 0.971 | 0.331 / 0.988 |
| layer1 | Improvement | 0.183 / 1.000 | 0.899 / 0.509 | 0.546 / 0.954 | 0.543 / 0.821 |
| layer1 | Relative improvement | 0.126 / 0.999 | 0.146 / 1.000 | 0.523 / 0.947 | 0.265 / 0.982 |
| layer1 | `-identity` | 0.757 / 0.967 | 0.071 / 1.000 | 0.457 / 0.977 | 0.428 / 0.981 |
| layer2 | `-reconstruction` | 0.384 / 1.000 | 0.077 / 1.000 | 0.499 / 0.972 | 0.320 / 0.990 |
| layer2 | Improvement | 0.437 / 0.995 | 0.793 / 0.868 | 0.525 / 0.966 | 0.585 / 0.943 |
| layer2 | Relative improvement | 0.280 / 0.999 | 0.144 / 1.000 | 0.522 / 0.951 | 0.315 / 0.983 |
| layer2 | `-identity` | 0.462 / 0.999 | 0.098 / 1.000 | 0.486 / 0.976 | 0.349 / 0.991 |
| layer3 | `-reconstruction` | 0.132 / 0.999 | 0.133 / 0.999 | 0.478 / 0.974 | 0.248 / 0.991 |
| layer3 | Improvement | 0.939 / 0.244 | 0.879 / 0.478 | 0.516 / 0.956 | 0.778 / 0.560 |
| layer3 | Relative improvement | 0.746 / 0.752 | 0.441 / 0.976 | 0.494 / 0.970 | 0.560 / 0.900 |
| layer3 | `-identity` | 0.098 / 0.999 | 0.117 / 0.999 | 0.476 / 0.968 | 0.231 / 0.989 |
| layer4 | `-reconstruction` | 0.433 / 0.998 | 0.642 / 0.931 | 0.741 / 0.857 | 0.606 / 0.929 |
| layer4 | Improvement | 0.900 / 0.475 | 0.729 / 0.841 | 0.732 / 0.806 | 0.787 / 0.708 |
| layer4 | Relative improvement | 0.748 / 0.965 | 0.710 / 0.891 | 0.771 / 0.817 | 0.743 / 0.891 |
| layer4 | `-identity` | 0.066 / 1.000 | 0.298 / 0.998 | 0.344 / 0.986 | 0.236 / 0.995 |

### Channel Masking: ResNet-18, CIFAR-10 ID

| Layer | Score | MNIST ROC-AUC / FPR@95 | SVHN ROC-AUC / FPR@95 | CIFAR-100 ROC-AUC / FPR@95 | Macro ROC-AUC / FPR@95 |
| --- | --- | ---: | ---: | ---: | ---: |
| layer1 | `-reconstruction` | 0.446 / 1.000 | 0.061 / 0.999 | 0.556 / 0.885 | 0.354 / 0.961 |
| layer1 | Improvement | 0.576 / 0.982 | 0.771 / 0.813 | 0.478 / 0.947 | 0.608 / 0.914 |
| layer1 | Relative improvement | 0.458 / 0.991 | 0.076 / 0.999 | 0.561 / 0.875 | 0.365 / 0.955 |
| layer1 | `-identity` | 0.426 / 1.000 | 0.166 / 0.999 | 0.534 / 0.907 | 0.375 / 0.969 |
| layer2 | `-reconstruction` | 0.592 / 0.997 | 0.104 / 0.999 | 0.509 / 0.913 | 0.402 / 0.969 |
| layer2 | Improvement | 0.597 / 0.918 | 0.937 / 0.271 | 0.542 / 0.913 | 0.692 / 0.701 |
| layer2 | Relative improvement | 0.778 / 0.924 | 0.282 / 0.965 | 0.555 / 0.872 | 0.538 / 0.920 |
| layer2 | `-identity` | 0.476 / 0.998 | 0.067 / 0.999 | 0.479 / 0.936 | 0.341 / 0.977 |
| layer3 | `-reconstruction` | 0.438 / 0.993 | 0.059 / 0.999 | 0.339 / 0.981 | 0.279 / 0.991 |
| layer3 | Improvement | 0.846 / 0.835 | 0.962 / 0.217 | 0.841 / 0.662 | 0.883 / 0.571 |
| layer3 | Relative improvement | 0.884 / 0.673 | 0.621 / 0.917 | 0.815 / 0.735 | 0.773 / 0.775 |
| layer3 | `-identity` | 0.221 / 1.000 | 0.031 / 1.000 | 0.195 / 0.999 | 0.149 / 1.000 |
| layer4 | `-reconstruction` | 0.612 / 0.983 | 0.561 / 0.950 | 0.744 / 0.856 | 0.639 / 0.930 |
| layer4 | Improvement | 0.979 / 0.115 | 0.893 / 0.458 | 0.773 / 0.740 | 0.882 / 0.438 |
| layer4 | Relative improvement | 0.948 / 0.301 | 0.852 / 0.647 | 0.855 / 0.628 | 0.885 / 0.525 |
| layer4 | `-identity` | 0.021 / 1.000 | 0.110 / 1.000 | 0.234 / 0.998 | 0.122 / 0.999 |

### Channel Masking: ResNet-18, CIFAR-100 ID

| Layer | Score | MNIST ROC-AUC / FPR@95 | SVHN ROC-AUC / FPR@95 | CIFAR-10 ROC-AUC / FPR@95 | Macro ROC-AUC / FPR@95 |
| --- | --- | ---: | ---: | ---: | ---: |
| layer1 | `-reconstruction` | 0.685 / 1.000 | 0.044 / 1.000 | 0.455 / 0.983 | 0.395 / 0.994 |
| layer1 | Improvement | 0.934 / 0.472 | 0.961 / 0.202 | 0.524 / 0.961 | 0.806 / 0.545 |
| layer1 | Relative improvement | 0.984 / 0.033 | 0.242 / 0.997 | 0.452 / 0.985 | 0.560 / 0.672 |
| layer1 | `-identity` | 0.114 / 1.000 | 0.036 / 1.000 | 0.471 / 0.975 | 0.207 / 0.992 |
| layer2 | `-reconstruction` | 0.516 / 1.000 | 0.113 / 0.999 | 0.466 / 0.983 | 0.365 / 0.994 |
| layer2 | Improvement | 0.881 / 0.545 | 0.900 / 0.421 | 0.552 / 0.950 | 0.778 / 0.639 |
| layer2 | Relative improvement | 0.880 / 0.671 | 0.199 / 0.998 | 0.489 / 0.973 | 0.523 / 0.881 |
| layer2 | `-identity` | 0.218 / 1.000 | 0.094 / 0.999 | 0.453 / 0.983 | 0.255 / 0.994 |
| layer3 | `-reconstruction` | 0.042 / 1.000 | 0.105 / 0.999 | 0.480 / 0.976 | 0.209 / 0.992 |
| layer3 | Improvement | 0.994 / 0.022 | 0.936 / 0.283 | 0.504 / 0.967 | 0.811 / 0.424 |
| layer3 | Relative improvement | 0.895 / 0.443 | 0.465 / 0.930 | 0.470 / 0.980 | 0.610 / 0.784 |
| layer3 | `-identity` | 0.014 / 1.000 | 0.077 / 0.999 | 0.487 / 0.971 | 0.193 / 0.990 |
| layer4 | `-reconstruction` | 0.745 / 0.879 | 0.622 / 0.939 | 0.649 / 0.915 | 0.672 / 0.911 |
| layer4 | Improvement | 0.621 / 0.981 | 0.806 / 0.760 | 0.774 / 0.825 | 0.734 / 0.855 |
| layer4 | Relative improvement | 0.721 / 0.945 | 0.804 / 0.834 | 0.785 / 0.806 | 0.770 / 0.861 |
| layer4 | `-identity` | 0.415 / 0.999 | 0.211 / 1.000 | 0.247 / 0.998 | 0.291 / 0.999 |

### Channel Masking: ResNet-50, CIFAR-10 ID

| Layer | Score | MNIST ROC-AUC / FPR@95 | SVHN ROC-AUC / FPR@95 | CIFAR-100 ROC-AUC / FPR@95 | Macro ROC-AUC / FPR@95 |
| --- | --- | ---: | ---: | ---: | ---: |
| layer1 | `-reconstruction` | 0.442 / 1.000 | 0.056 / 1.000 | 0.557 / 0.888 | 0.351 / 0.962 |
| layer1 | Improvement | 0.155 / 1.000 | 0.824 / 0.744 | 0.446 / 0.950 | 0.475 / 0.898 |
| layer1 | Relative improvement | 0.107 / 1.000 | 0.062 / 1.000 | 0.540 / 0.892 | 0.236 / 0.964 |
| layer1 | `-identity` | 0.822 / 0.996 | 0.152 / 0.998 | 0.555 / 0.893 | 0.510 / 0.962 |
| layer2 | `-reconstruction` | 0.425 / 1.000 | 0.076 / 1.000 | 0.528 / 0.903 | 0.343 / 0.967 |
| layer2 | Improvement | 0.539 / 0.980 | 0.854 / 0.593 | 0.503 / 0.923 | 0.632 / 0.832 |
| layer2 | Relative improvement | 0.407 / 1.000 | 0.139 / 0.999 | 0.547 / 0.879 | 0.364 / 0.959 |
| layer2 | `-identity` | 0.450 / 1.000 | 0.104 / 0.999 | 0.506 / 0.920 | 0.354 / 0.973 |
| layer3 | `-reconstruction` | 0.443 / 0.997 | 0.093 / 0.998 | 0.431 / 0.950 | 0.322 / 0.982 |
| layer3 | Improvement | 0.843 / 0.742 | 0.911 / 0.404 | 0.717 / 0.788 | 0.824 / 0.645 |
| layer3 | Relative improvement | 0.913 / 0.599 | 0.377 / 0.978 | 0.712 / 0.822 | 0.667 / 0.799 |
| layer3 | `-identity` | 0.256 / 1.000 | 0.076 / 0.999 | 0.334 / 0.988 | 0.222 / 0.996 |
| layer4 | `-reconstruction` | 0.656 / 0.980 | 0.463 / 0.974 | 0.721 / 0.828 | 0.613 / 0.928 |
| layer4 | Improvement | 0.909 / 0.399 | 0.868 / 0.511 | 0.622 / 0.871 | 0.799 / 0.594 |
| layer4 | Relative improvement | 0.955 / 0.251 | 0.862 / 0.607 | 0.822 / 0.671 | 0.879 / 0.510 |
| layer4 | `-identity` | 0.097 / 1.000 | 0.137 / 0.999 | 0.390 / 0.980 | 0.208 / 0.993 |

### Channel Masking: ResNet-50, CIFAR-100 ID

| Layer | Score | MNIST ROC-AUC / FPR@95 | SVHN ROC-AUC / FPR@95 | CIFAR-10 ROC-AUC / FPR@95 | Macro ROC-AUC / FPR@95 |
| --- | --- | ---: | ---: | ---: | ---: |
| layer1 | `-reconstruction` | 0.374 / 1.000 | 0.039 / 1.000 | 0.453 / 0.979 | 0.289 / 0.993 |
| layer1 | Improvement | 0.315 / 1.000 | 0.916 / 0.441 | 0.557 / 0.955 | 0.596 / 0.799 |
| layer1 | Relative improvement | 0.126 / 1.000 | 0.082 / 1.000 | 0.475 / 0.980 | 0.228 / 0.993 |
| layer1 | `-identity` | 0.661 / 1.000 | 0.075 / 1.000 | 0.443 / 0.980 | 0.393 / 0.993 |
| layer2 | `-reconstruction` | 0.395 / 1.000 | 0.070 / 1.000 | 0.481 / 0.978 | 0.315 / 0.992 |
| layer2 | Improvement | 0.976 / 0.147 | 0.865 / 0.612 | 0.529 / 0.961 | 0.790 / 0.573 |
| layer2 | Relative improvement | 0.935 / 0.467 | 0.077 / 1.000 | 0.490 / 0.972 | 0.501 / 0.813 |
| layer2 | `-identity` | 0.066 / 1.000 | 0.103 / 0.999 | 0.473 / 0.978 | 0.214 / 0.992 |
| layer3 | `-reconstruction` | 0.042 / 1.000 | 0.108 / 0.999 | 0.471 / 0.981 | 0.207 / 0.993 |
| layer3 | Improvement | 0.991 / 0.038 | 0.909 / 0.392 | 0.522 / 0.958 | 0.807 / 0.463 |
| layer3 | Relative improvement | 0.768 / 0.735 | 0.360 / 0.986 | 0.471 / 0.983 | 0.533 / 0.901 |
| layer3 | `-identity` | 0.022 / 1.000 | 0.095 / 0.999 | 0.472 / 0.975 | 0.196 / 0.992 |
| layer4 | `-reconstruction` | 0.226 / 0.999 | 0.647 / 0.947 | 0.702 / 0.891 | 0.525 / 0.946 |
| layer4 | Improvement | 0.943 / 0.301 | 0.747 / 0.837 | 0.712 / 0.842 | 0.800 / 0.660 |
| layer4 | Relative improvement | 0.701 / 0.977 | 0.741 / 0.872 | 0.757 / 0.835 | 0.733 / 0.895 |
| layer4 | `-identity` | 0.031 / 1.000 | 0.322 / 0.996 | 0.399 / 0.986 | 0.251 / 0.994 |

## Interpretation

- Raw reconstruction magnitude is not a useful primary OOD Score in these
  sweeps. Its best macro result is channel ResNet-18/CIFAR-100 layer4 at only
  `0.672 / 0.911`.
- The zero-fill or mean-fill identity baseline is also weak. The learned
  correction becomes useful only after subtracting the identity error through
  improvement or relative improvement.
- Layer3 absolute improvement is the most reliable setting for CIFAR-100 ID.
  Layer4 becomes preferable for CIFAR-10 ID, especially with channel relative
  improvement or when optimizing FPR@95.
- The macro average materially hides the near-OOD failure. With CIFAR-100 ID,
  the macro-optimal layer3 improvement configurations score nearly perfectly
  on MNIST but only `0.504 / 0.967` (ResNet-18) and `0.522 / 0.958`
  (ResNet-50) on CIFAR-10.
- The opposite CIFAR dataset is therefore the practical bottleneck. Even after
  selecting specifically for that dataset, its best FPR@95 is `0.628` with
  CIFAR-10 ID and `0.806` with CIFAR-100 ID. By comparison, the best MNIST and
  SVHN FPR@95 values are `0.022` and `0.161`.
- ResNet-50 is not uniformly better than ResNet-18. Its strongest CIFAR-10
  result, channel relative improvement at `0.879 / 0.510`, trades slightly
  lower ROC-AUC for slightly lower FPR@95 versus ResNet-18 at
  `0.885 / 0.525`. ResNet-18 is also slightly stronger for channel improvement
  with CIFAR-100 ID and for spatial improvement with CIFAR-10 ID.
- Channel masking is the recommended next direction. A useful follow-up is to
  tune channel mask probability around `0.2` and evaluate layer3/layer4 score
  fusion, while retaining separate near-OOD selection.

## Run and Export Provenance

The spatial jobs `407330`, `407331`, `407346`, and `407347` completed
successfully. Channel jobs `407352`, `407353`, and `407354` completed
successfully. Job `407351` encountered a transient DataLoader shared-memory
unlink error after completing layer1; retry `407550` completed layers 2-4 with
unchanged configurations. Metric export jobs `407551` and `407552` completed
successfully.

There are 32 completed runs and 128 test score artifacts. Channel masking
correctly produced no feature-normalizer artifacts.

Machine-readable report data:

```text
reports/outputs/json/spatial_block_residual_default.json
reports/outputs/json/spatial_block_residual_improvement.json
reports/outputs/json/spatial_block_residual_relative_improvement.json
reports/outputs/json/spatial_block_residual_negative_identity.json
reports/outputs/json/spatial_block_residual_losses.json
reports/outputs/json/channel_residual_default.json
reports/outputs/json/channel_residual_improvement.json
reports/outputs/json/channel_residual_relative_improvement.json
reports/outputs/json/channel_residual_negative_identity.json
reports/outputs/json/channel_residual_losses.json
```
