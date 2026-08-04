# Sequential Layer1-Layer4 Clipping Students

Status: completed for CIFAR-10 and CIFAR-100 ID with ResNet-18 and ResNet-50 teachers.

## Setup

- Clipping sequence: clip `layer1`, propagate, clip `layer2`, propagate, clip `layer3`, propagate, then clip `layer4`.
- Every layer independently samples from the same percentile range `[0.5, 1.0]`.
- Clipping modes: constant, spatial-dependent, and channel-dependent; one mode is shared by all four layers.
- Student input: GAP or flattened clipped `layer4`, concatenated with `u_layer1`, `u_layer2`, `u_layer3`, and `u_layer4`.
- Student: linear classifier trained against clean teacher logits.
- Objectives: centered-logit MSE and KL divergence.
- Training: 30 epochs, seed 42, batch size 256; best validation-loss checkpoint.
- OOD datasets: MNIST, SVHN, and the opposite CIFAR test set.
- Inference scope: clean/unperturbed only. The completed jobs did not export 50-draw clipped inference artifacts.

Student input dimensions:

| Teacher | Clipping | Pooling | Student input |
|---|---:|---:|---:|
| ResNet-18 | Constant | GAP | 516 |
| ResNet-18 | Constant | Flatten | 8196 |
| ResNet-18 | Spatial | GAP | 1872 |
| ResNet-18 | Spatial | Flatten | 9552 |
| ResNet-18 | Channel | GAP | 1472 |
| ResNet-18 | Channel | Flatten | 9152 |
| ResNet-50 | Constant | GAP | 2052 |
| ResNet-50 | Constant | Flatten | 32772 |
| ResNet-50 | Spatial | GAP | 3408 |
| ResNet-50 | Spatial | Flatten | 34128 |
| ResNet-50 | Channel | GAP | 5888 |
| ResNet-50 | Channel | Flatten | 36608 |

Run-directory pattern:

```text
runs/students/perturbation/embedding/clipping/<cifar_10|cifar_100>/<resnet18|resnet50>/linear_layer1_layer2_layer3_layer4_clip_<constant|spatial|channel>_<avg|flatten>
```

## Training Results

Losses are objective-specific and should not be compared across centered-logit MSE and KL divergence.

### CIFAR-10 ID, ResNet-18

| Clipping | Pooling | Objective | Test accuracy | Test distillation loss |
|---|---:|---:|---:|---:|
| Constant | GAP | Centered-logit MSE | 0.9022 | 4.30756 |
| Constant | GAP | KL divergence | 0.9108 | 0.232206 |
| Constant | Flatten | Centered-logit MSE | 0.8958 | 6.56083 |
| Constant | Flatten | KL divergence | 0.9133 | 0.196523 |
| Spatial | GAP | Centered-logit MSE | 0.7698 | 7.62396 |
| Spatial | GAP | KL divergence | 0.8832 | 0.269178 |
| Spatial | Flatten | Centered-logit MSE | 0.8190 | 71.081 |
| Spatial | Flatten | KL divergence | 0.8902 | 0.378189 |
| Channel | GAP | Centered-logit MSE | 0.8151 | 8.17712 |
| Channel | GAP | KL divergence | 0.8429 | 0.378572 |
| Channel | Flatten | Centered-logit MSE | 0.8818 | 11.2146 |
| Channel | Flatten | KL divergence | 0.8375 | 0.378703 |

### CIFAR-10 ID, ResNet-50

| Clipping | Pooling | Objective | Test accuracy | Test distillation loss |
|---|---:|---:|---:|---:|
| Constant | GAP | Centered-logit MSE | 0.8448 | 7.98155 |
| Constant | GAP | KL divergence | 0.8674 | 0.91195 |
| Constant | Flatten | Centered-logit MSE | 0.8390 | 7.1212 |
| Constant | Flatten | KL divergence | 0.8835 | 0.284629 |
| Spatial | GAP | Centered-logit MSE | 0.4487 | 14.7444 |
| Spatial | GAP | KL divergence | 0.5223 | 1.49704 |
| Spatial | Flatten | Centered-logit MSE | 0.4550 | 15.5338 |
| Spatial | Flatten | KL divergence | 0.5479 | 1.18531 |
| Channel | GAP | Centered-logit MSE | 0.6400 | 12.5129 |
| Channel | GAP | KL divergence | 0.7443 | 0.652839 |
| Channel | Flatten | Centered-logit MSE | 0.6770 | 25.5469 |
| Channel | Flatten | KL divergence | 0.7605 | 0.803411 |

### CIFAR-100 ID, ResNet-18

| Clipping | Pooling | Objective | Test accuracy | Test distillation loss |
|---|---:|---:|---:|---:|
| Constant | GAP | Centered-logit MSE | 0.3092 | 1.50274 |
| Constant | GAP | KL divergence | 0.5340 | 2.65626 |
| Constant | Flatten | Centered-logit MSE | 0.7420 | 0.94553 |
| Constant | Flatten | KL divergence | 0.7396 | 0.351887 |
| Spatial | GAP | Centered-logit MSE | 0.2206 | 1.49421 |
| Spatial | GAP | KL divergence | 0.1778 | 2.8166 |
| Spatial | Flatten | Centered-logit MSE | 0.2152 | 1.493 |
| Spatial | Flatten | KL divergence | 0.2961 | 2.40531 |
| Channel | GAP | Centered-logit MSE | 0.2974 | 1.36719 |
| Channel | GAP | KL divergence | 0.2614 | 2.23561 |
| Channel | Flatten | Centered-logit MSE | 0.4596 | 1.28199 |
| Channel | Flatten | KL divergence | 0.5076 | 1.39685 |

### CIFAR-100 ID, ResNet-50

| Clipping | Pooling | Objective | Test accuracy | Test distillation loss |
|---|---:|---:|---:|---:|
| Constant | GAP | Centered-logit MSE | 0.1579 | 2.47466 |
| Constant | GAP | KL divergence | 0.5476 | 3.66724 |
| Constant | Flatten | Centered-logit MSE | 0.7479 | 1.52177 |
| Constant | Flatten | KL divergence | 0.7354 | 0.567488 |
| Spatial | GAP | Centered-logit MSE | 0.0584 | 2.59942 |
| Spatial | GAP | KL divergence | 0.0564 | 4.02012 |
| Spatial | Flatten | Centered-logit MSE | 0.0773 | 2.56704 |
| Spatial | Flatten | KL divergence | 0.1142 | 3.71684 |
| Channel | GAP | Centered-logit MSE | 0.0771 | 2.66263 |
| Channel | GAP | KL divergence | 0.1337 | 4.11718 |
| Channel | Flatten | Centered-logit MSE | 0.1596 | 2.88094 |
| Channel | Flatten | KL divergence | 0.3086 | 4.33684 |

## OOD Metrics

Every cell is `ROC-AUC / FPR@95`; higher ROC-AUC and lower FPR@95 are better. All OOD Scores follow the project convention that higher values are more ID-like.

## Teacher Baselines

### CIFAR-10 ID, ResNet-18

| Score | MNIST | SVHN | CIFAR-100 | Macro |
|---|---:|---:|---:|---:|
| Teacher MSP | 0.918 / 0.556 | 0.904 / 0.598 | 0.879 / 0.624 | 0.900 / 0.593 |
| Teacher energy | 0.958 / 0.260 | 0.918 / 0.412 | 0.879 / 0.506 | 0.918 / 0.392 |

### CIFAR-10 ID, ResNet-50

| Score | MNIST | SVHN | CIFAR-100 | Macro |
|---|---:|---:|---:|---:|
| Teacher MSP | 0.897 / 0.622 | 0.891 / 0.680 | 0.873 / 0.628 | 0.887 / 0.643 |
| Teacher energy | 0.932 / 0.408 | 0.911 / 0.517 | 0.868 / 0.517 | 0.904 / 0.481 |

### CIFAR-100 ID, ResNet-18

| Score | MNIST | SVHN | CIFAR-10 | Macro |
|---|---:|---:|---:|---:|
| Teacher MSP | 0.701 / 0.956 | 0.806 / 0.812 | 0.792 / 0.794 | 0.767 / 0.854 |
| Teacher energy | 0.695 / 0.974 | 0.829 / 0.771 | 0.795 / 0.800 | 0.773 / 0.848 |

### CIFAR-100 ID, ResNet-50

| Score | MNIST | SVHN | CIFAR-10 | Macro |
|---|---:|---:|---:|---:|
| Teacher MSP | 0.811 / 0.823 | 0.758 / 0.793 | 0.792 / 0.781 | 0.787 / 0.799 |
| Teacher energy | 0.878 / 0.670 | 0.783 / 0.766 | 0.801 / 0.774 | 0.821 / 0.737 |

## Main Findings

| ID dataset | Teacher | Best macro ROC-AUC configuration | Macro | Best macro FPR@95 configuration | Macro |
|---|---:|---:|---:|---:|---:|
| CIFAR-10 | ResNet-18 | Student energy; Spatial, Flatten, KL divergence | 0.871 / 0.485 | Student energy; Spatial, GAP, Centered-logit MSE | 0.868 / 0.476 |
| CIFAR-10 | ResNet-50 | Energy gap; Spatial, GAP, KL divergence | 0.910 / 0.469 | Energy gap; Spatial, GAP, KL divergence | 0.910 / 0.469 |
| CIFAR-100 | ResNet-18 | KL (teacher / student); Constant, Flatten, KL divergence | 0.780 / 0.776 | KL (teacher / student); Constant, Flatten, KL divergence | 0.780 / 0.776 |
| CIFAR-100 | ResNet-50 | Energy gap; Spatial, Flatten, KL divergence | 0.830 / 0.770 | Student energy; Spatial, Flatten, Centered-logit MSE | 0.691 / 0.690 |

Paired comparison against clean layer3-layer4 inference. A positive ROC-AUC delta and negative FPR@95 delta favor all-layer clipping.
The full reference results are in the [layer3-layer4 report](layer3_layer4_clipping.md).

| ID dataset | Teacher | ROC-AUC wins | Mean ROC-AUC delta | FPR@95 wins | Mean FPR@95 delta |
|---|---:|---:|---:|---:|---:|
| CIFAR-10 | ResNet-18 | 27/96 | -0.027 | 35/96 | +0.063 |
| CIFAR-10 | ResNet-50 | 19/96 | -0.088 | 29/96 | +0.063 |
| CIFAR-100 | ResNet-18 | 38/96 | -0.031 | 22/96 | +0.030 |
| CIFAR-100 | ResNet-50 | 25/96 | -0.108 | 26/96 | +0.064 |

Across the paired macro comparisons, extending clipping to layer1 and layer2 is not a consistent improvement over layer3-layer4 clipping. The strongest all-layer case is CIFAR-100/ResNet-18 for ROC-AUC, while the other settings lose most matched comparisons.

Classification quality also deteriorates sharply for many spatial and channel variants. In particular, the CIFAR-10/ResNet-50 spatial-GAP KL student reaches the best macro OOD pair in that group but only 0.5223 test accuracy, versus 0.9238 for its layer3-layer4 counterpart. The CIFAR-100/ResNet-50 spatial-flatten KL student similarly reaches the best macro ROC-AUC with only 0.1142 test accuracy, versus 0.7884 for layer3-layer4 clipping. These OOD gains therefore reflect degraded students rather than a better accuracy-preserving detector.

Constant clipping with flattened layer4 features is the most stable all-layer family, but its best test accuracies still trail the matching layer3-layer4 runs. Overall, the clean-inference results do not support extending sequential clipping into layer1 and layer2.

## Macro OOD Comparison

### CIFAR-10 ID, ResNet-18

#### Centered-logit MSE

| Clipping | Pooling | Max probability difference | Absolute max probability difference | KL (teacher / student) | Centered-logit L2 distance | Energy gap | Absolute energy gap | Student MSP | Student energy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Constant | GAP | 0.417 / 0.787 | 0.706 / 0.848 | 0.755 / 0.866 | 0.306 / 0.995 | 0.685 / 0.932 | 0.315 / 0.998 | 0.788 / 0.691 | 0.824 / 0.584 |
| Constant | Flatten | 0.554 / 0.681 | 0.769 / 0.868 | 0.798 / 0.874 | 0.195 / 0.998 | 0.466 / 0.999 | 0.254 / 0.998 | 0.768 / 0.769 | 0.805 / 0.637 |
| Spatial | GAP | 0.303 / 0.911 | 0.728 / 0.828 | 0.777 / 0.886 | 0.315 / 0.999 | 0.668 / 0.948 | 0.329 / 0.998 | 0.846 / 0.614 | 0.868 / 0.476 |
| Spatial | Flatten | 0.615 / 0.690 | 0.789 / 0.882 | 0.820 / 0.826 | 0.173 / 0.997 | 0.251 / 0.997 | 0.243 / 0.997 | 0.753 / 0.850 | 0.852 / 0.559 |
| Channel | GAP | 0.593 / 0.635 | 0.655 / 0.928 | 0.736 / 0.862 | 0.212 / 0.995 | 0.720 / 0.964 | 0.397 / 0.995 | 0.620 / 0.880 | 0.679 / 0.766 |
| Channel | Flatten | 0.607 / 0.667 | 0.761 / 0.902 | 0.807 / 0.818 | 0.275 / 0.996 | 0.519 / 0.994 | 0.370 / 0.994 | 0.724 / 0.858 | 0.752 / 0.798 |

#### KL divergence

| Clipping | Pooling | Max probability difference | Absolute max probability difference | KL (teacher / student) | Centered-logit L2 distance | Energy gap | Absolute energy gap | Student MSP | Student energy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Constant | GAP | 0.366 / 0.869 | 0.695 / 0.830 | 0.758 / 0.830 | 0.291 / 0.993 | 0.772 / 0.702 | 0.228 / 0.998 | 0.795 / 0.660 | 0.819 / 0.657 |
| Constant | Flatten | 0.539 / 0.699 | 0.775 / 0.849 | 0.818 / 0.820 | 0.167 / 0.998 | 0.459 / 0.999 | 0.249 / 0.999 | 0.771 / 0.744 | 0.804 / 0.678 |
| Spatial | GAP | 0.468 / 0.763 | 0.741 / 0.863 | 0.811 / 0.790 | 0.171 / 0.996 | 0.354 / 0.988 | 0.215 / 0.989 | 0.781 / 0.733 | 0.844 / 0.594 |
| Spatial | Flatten | 0.624 / 0.696 | 0.826 / 0.837 | 0.839 / 0.821 | 0.160 / 0.996 | 0.220 / 0.998 | 0.220 / 0.998 | 0.791 / 0.799 | 0.871 / 0.485 |
| Channel | GAP | 0.546 / 0.705 | 0.599 / 0.950 | 0.711 / 0.829 | 0.139 / 0.996 | 0.725 / 0.892 | 0.284 / 0.997 | 0.681 / 0.883 | 0.746 / 0.777 |
| Channel | Flatten | 0.576 / 0.692 | 0.571 / 0.933 | 0.686 / 0.826 | 0.188 / 0.995 | 0.722 / 0.897 | 0.294 / 0.997 | 0.650 / 0.855 | 0.735 / 0.798 |

### CIFAR-10 ID, ResNet-50

#### Centered-logit MSE

| Clipping | Pooling | Max probability difference | Absolute max probability difference | KL (teacher / student) | Centered-logit L2 distance | Energy gap | Absolute energy gap | Student MSP | Student energy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Constant | GAP | 0.431 / 0.849 | 0.623 / 0.905 | 0.737 / 0.849 | 0.279 / 0.993 | 0.791 / 0.789 | 0.209 / 0.997 | 0.714 / 0.854 | 0.644 / 0.884 |
| Constant | Flatten | 0.444 / 0.778 | 0.667 / 0.896 | 0.753 / 0.860 | 0.311 / 0.996 | 0.706 / 0.916 | 0.289 / 0.997 | 0.715 / 0.832 | 0.668 / 0.900 |
| Spatial | GAP | 0.517 / 0.932 | 0.484 / 0.983 | 0.594 / 0.977 | 0.257 / 0.995 | 0.903 / 0.489 | 0.097 / 0.999 | 0.654 / 0.948 | 0.540 / 0.985 |
| Spatial | Flatten | 0.594 / 0.899 | 0.419 / 0.994 | 0.612 / 0.976 | 0.366 / 0.992 | 0.898 / 0.483 | 0.102 / 0.999 | 0.523 / 0.984 | 0.485 / 0.987 |
| Channel | GAP | 0.430 / 0.812 | 0.660 / 0.905 | 0.625 / 0.993 | 0.241 / 0.996 | 0.777 / 0.721 | 0.230 / 0.999 | 0.723 / 0.832 | 0.657 / 0.907 |
| Channel | Flatten | 0.440 / 0.733 | 0.781 / 0.861 | 0.657 / 0.989 | 0.252 / 0.999 | 0.546 / 0.960 | 0.422 / 0.964 | 0.772 / 0.806 | 0.804 / 0.757 |

#### KL divergence

| Clipping | Pooling | Max probability difference | Absolute max probability difference | KL (teacher / student) | Centered-logit L2 distance | Energy gap | Absolute energy gap | Student MSP | Student energy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Constant | GAP | 0.457 / 0.957 | 0.543 / 0.908 | 0.651 / 0.856 | 0.160 / 0.994 | 0.901 / 0.497 | 0.099 / 0.999 | 0.738 / 0.802 | 0.664 / 0.865 |
| Constant | Flatten | 0.442 / 0.732 | 0.814 / 0.818 | 0.834 / 0.841 | 0.238 / 0.997 | 0.453 / 0.998 | 0.264 / 0.998 | 0.799 / 0.776 | 0.790 / 0.804 |
| Spatial | GAP | 0.601 / 0.859 | 0.399 / 0.984 | 0.552 / 0.963 | 0.177 / 0.997 | 0.910 / 0.469 | 0.090 / 0.999 | 0.582 / 0.927 | 0.479 / 0.973 |
| Spatial | Flatten | 0.456 / 0.950 | 0.545 / 0.997 | 0.664 / 0.965 | 0.297 / 0.994 | 0.902 / 0.485 | 0.098 / 0.999 | 0.700 / 0.984 | 0.544 / 0.996 |
| Channel | GAP | 0.393 / 0.884 | 0.644 / 0.925 | 0.668 / 0.982 | 0.163 / 0.997 | 0.856 / 0.594 | 0.144 / 0.998 | 0.751 / 0.850 | 0.693 / 0.807 |
| Channel | Flatten | 0.566 / 0.686 | 0.634 / 0.946 | 0.712 / 0.896 | 0.316 / 0.999 | 0.705 / 0.972 | 0.418 / 0.982 | 0.616 / 0.920 | 0.633 / 0.892 |

### CIFAR-100 ID, ResNet-18

#### Centered-logit MSE

| Clipping | Pooling | Max probability difference | Absolute max probability difference | KL (teacher / student) | Centered-logit L2 distance | Energy gap | Absolute energy gap | Student MSP | Student energy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Constant | GAP | 0.761 / 0.854 | 0.239 / 0.999 | 0.247 / 0.999 | 0.267 / 0.999 | 0.773 / 0.851 | 0.227 / 1.000 | 0.676 / 0.875 | 0.589 / 0.969 |
| Constant | Flatten | 0.596 / 0.924 | 0.404 / 0.995 | 0.496 / 0.982 | 0.316 / 0.999 | 0.762 / 0.864 | 0.238 / 1.000 | 0.700 / 0.904 | 0.690 / 0.902 |
| Spatial | GAP | 0.758 / 0.854 | 0.242 / 0.997 | 0.252 / 0.999 | 0.277 / 0.999 | 0.772 / 0.851 | 0.228 / 1.000 | 0.568 / 0.935 | 0.612 / 0.940 |
| Spatial | Flatten | 0.744 / 0.854 | 0.256 / 0.999 | 0.268 / 0.998 | 0.275 / 0.999 | 0.769 / 0.856 | 0.231 / 1.000 | 0.616 / 0.919 | 0.627 / 0.912 |
| Channel | GAP | 0.704 / 0.898 | 0.296 / 0.991 | 0.334 / 0.984 | 0.285 / 0.998 | 0.766 / 0.864 | 0.234 / 0.999 | 0.646 / 0.930 | 0.642 / 0.886 |
| Channel | Flatten | 0.630 / 0.969 | 0.367 / 0.994 | 0.417 / 0.987 | 0.285 / 0.999 | 0.735 / 0.935 | 0.250 / 0.999 | 0.628 / 0.927 | 0.653 / 0.891 |

#### KL divergence

| Clipping | Pooling | Max probability difference | Absolute max probability difference | KL (teacher / student) | Centered-logit L2 distance | Energy gap | Absolute energy gap | Student MSP | Student energy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Constant | GAP | 0.752 / 0.853 | 0.248 / 1.000 | 0.283 / 0.999 | 0.272 / 0.999 | 0.772 / 0.851 | 0.228 / 1.000 | 0.611 / 0.921 | 0.686 / 0.921 |
| Constant | Flatten | 0.582 / 0.813 | 0.674 / 0.914 | 0.780 / 0.776 | 0.374 / 0.997 | 0.507 / 0.996 | 0.420 / 0.996 | 0.716 / 0.900 | 0.710 / 0.903 |
| Spatial | GAP | 0.730 / 0.864 | 0.270 / 0.998 | 0.276 / 0.998 | 0.287 / 1.000 | 0.769 / 0.848 | 0.231 / 1.000 | 0.603 / 0.924 | 0.568 / 0.922 |
| Spatial | Flatten | 0.691 / 0.819 | 0.382 / 0.998 | 0.416 / 0.995 | 0.375 / 0.999 | 0.694 / 0.980 | 0.347 / 0.993 | 0.526 / 0.947 | 0.561 / 0.940 |
| Channel | GAP | 0.611 / 0.934 | 0.386 / 0.994 | 0.428 / 0.978 | 0.345 / 0.999 | 0.694 / 0.989 | 0.256 / 0.999 | 0.662 / 0.896 | 0.669 / 0.883 |
| Channel | Flatten | 0.568 / 0.884 | 0.530 / 0.973 | 0.645 / 0.968 | 0.337 / 0.998 | 0.572 / 1.000 | 0.456 / 1.000 | 0.675 / 0.904 | 0.700 / 0.864 |

### CIFAR-100 ID, ResNet-50

#### Centered-logit MSE

| Clipping | Pooling | Max probability difference | Absolute max probability difference | KL (teacher / student) | Centered-logit L2 distance | Energy gap | Absolute energy gap | Student MSP | Student energy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Constant | GAP | 0.767 / 0.800 | 0.233 / 0.979 | 0.256 / 0.994 | 0.208 / 0.997 | 0.820 / 0.736 | 0.180 / 0.999 | 0.668 / 0.769 | 0.645 / 0.958 |
| Constant | Flatten | 0.607 / 0.890 | 0.393 / 0.984 | 0.519 / 0.953 | 0.254 / 0.991 | 0.803 / 0.779 | 0.197 / 0.999 | 0.698 / 0.892 | 0.664 / 0.898 |
| Spatial | GAP | 0.771 / 0.801 | 0.229 / 0.995 | 0.192 / 0.998 | 0.184 / 0.997 | 0.821 / 0.736 | 0.179 / 0.999 | 0.667 / 0.730 | 0.564 / 0.918 |
| Spatial | Flatten | 0.771 / 0.803 | 0.229 / 0.999 | 0.213 / 0.995 | 0.187 / 0.996 | 0.818 / 0.750 | 0.182 / 0.999 | 0.609 / 0.805 | 0.691 / 0.690 |
| Channel | GAP | 0.768 / 0.801 | 0.232 / 0.998 | 0.262 / 0.982 | 0.205 / 0.994 | 0.821 / 0.739 | 0.179 / 0.999 | 0.536 / 0.977 | 0.508 / 0.993 |
| Channel | Flatten | 0.656 / 0.921 | 0.344 / 0.976 | 0.424 / 0.992 | 0.327 / 0.990 | 0.805 / 0.744 | 0.195 / 0.998 | 0.639 / 0.954 | 0.588 / 0.929 |

#### KL divergence

| Clipping | Pooling | Max probability difference | Absolute max probability difference | KL (teacher / student) | Centered-logit L2 distance | Energy gap | Absolute energy gap | Student MSP | Student energy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Constant | GAP | 0.770 / 0.801 | 0.230 / 0.999 | 0.271 / 0.995 | 0.214 / 0.996 | 0.821 / 0.736 | 0.179 / 0.999 | 0.699 / 0.877 | 0.606 / 0.959 |
| Constant | Flatten | 0.490 / 0.855 | 0.552 / 0.962 | 0.704 / 0.889 | 0.425 / 0.985 | 0.773 / 0.908 | 0.218 / 0.998 | 0.688 / 0.882 | 0.640 / 0.866 |
| Spatial | GAP | 0.760 / 0.802 | 0.240 / 0.997 | 0.238 / 0.996 | 0.194 / 0.995 | 0.821 / 0.740 | 0.179 / 0.999 | 0.587 / 0.930 | 0.565 / 0.821 |
| Spatial | Flatten | 0.778 / 0.804 | 0.222 / 0.997 | 0.249 / 0.995 | 0.192 / 0.992 | 0.830 / 0.770 | 0.170 / 0.999 | 0.484 / 0.916 | 0.550 / 0.839 |
| Channel | GAP | 0.678 / 0.890 | 0.322 / 0.995 | 0.446 / 0.994 | 0.331 / 0.987 | 0.800 / 0.746 | 0.201 / 0.998 | 0.529 / 0.965 | 0.603 / 0.944 |
| Channel | Flatten | 0.588 / 0.823 | 0.548 / 0.948 | 0.697 / 0.829 | 0.359 / 0.993 | 0.619 / 0.989 | 0.626 / 0.989 | 0.595 / 0.939 | 0.607 / 0.931 |

## Detailed OOD Results

Each configuration contains both training-objective tables.

### CIFAR-10 ID, ResNet-18

<details>
<summary>Constant, GAP</summary>

#### Centered-logit MSE

| Score | MNIST | SVHN | CIFAR-100 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.330 / 0.814 | 0.515 / 0.765 | 0.407 / 0.780 | 0.417 / 0.787 |
| Absolute max probability difference | 0.786 / 0.801 | 0.607 / 0.891 | 0.726 / 0.851 | 0.706 / 0.848 |
| KL (teacher / student) | 0.812 / 0.891 | 0.675 / 0.919 | 0.779 / 0.789 | 0.755 / 0.866 |
| Centered-logit L2 distance | 0.214 / 1.000 | 0.294 / 1.000 | 0.411 / 0.984 | 0.306 / 0.995 |
| Energy gap | 0.685 / 0.925 | 0.714 / 0.982 | 0.655 / 0.890 | 0.685 / 0.932 |
| Absolute energy gap | 0.314 / 1.000 | 0.284 / 1.000 | 0.348 / 0.995 | 0.315 / 0.998 |
| Student MSP | 0.863 / 0.606 | 0.716 / 0.739 | 0.784 / 0.727 | 0.788 / 0.691 |
| Student energy | 0.912 / 0.402 | 0.783 / 0.634 | 0.776 / 0.717 | 0.824 / 0.584 |

#### KL divergence

| Score | MNIST | SVHN | CIFAR-100 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.273 / 0.898 | 0.451 / 0.866 | 0.374 / 0.844 | 0.366 / 0.869 |
| Absolute max probability difference | 0.777 / 0.758 | 0.602 / 0.879 | 0.705 / 0.852 | 0.695 / 0.830 |
| KL (teacher / student) | 0.814 / 0.888 | 0.684 / 0.849 | 0.775 / 0.753 | 0.758 / 0.830 |
| Centered-logit L2 distance | 0.171 / 1.000 | 0.303 / 1.000 | 0.399 / 0.980 | 0.291 / 0.993 |
| Energy gap | 0.794 / 0.624 | 0.806 / 0.763 | 0.717 / 0.720 | 0.772 / 0.702 |
| Absolute energy gap | 0.206 / 1.000 | 0.194 / 1.000 | 0.283 / 0.995 | 0.228 / 0.998 |
| Student MSP | 0.881 / 0.538 | 0.717 / 0.724 | 0.786 / 0.719 | 0.795 / 0.660 |
| Student energy | 0.906 / 0.501 | 0.775 / 0.696 | 0.775 / 0.774 | 0.819 / 0.657 |

</details>

<details>
<summary>Constant, Flatten</summary>

#### Centered-logit MSE

| Score | MNIST | SVHN | CIFAR-100 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.457 / 0.702 | 0.692 / 0.630 | 0.513 / 0.712 | 0.554 / 0.681 |
| Absolute max probability difference | 0.818 / 0.844 | 0.727 / 0.896 | 0.761 / 0.863 | 0.769 / 0.868 |
| KL (teacher / student) | 0.834 / 0.884 | 0.760 / 0.935 | 0.800 / 0.804 | 0.798 / 0.874 |
| Centered-logit L2 distance | 0.089 / 1.000 | 0.160 / 1.000 | 0.335 / 0.995 | 0.195 / 0.998 |
| Energy gap | 0.428 / 1.000 | 0.506 / 1.000 | 0.465 / 0.996 | 0.466 / 0.999 |
| Absolute energy gap | 0.233 / 1.000 | 0.232 / 1.000 | 0.298 / 0.993 | 0.254 / 0.998 |
| Student MSP | 0.838 / 0.710 | 0.698 / 0.804 | 0.767 / 0.792 | 0.768 / 0.769 |
| Student energy | 0.892 / 0.475 | 0.753 / 0.701 | 0.769 / 0.733 | 0.805 / 0.637 |

#### KL divergence

| Score | MNIST | SVHN | CIFAR-100 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.431 / 0.752 | 0.683 / 0.631 | 0.503 / 0.713 | 0.539 / 0.699 |
| Absolute max probability difference | 0.794 / 0.897 | 0.748 / 0.841 | 0.782 / 0.808 | 0.775 / 0.849 |
| KL (teacher / student) | 0.815 / 0.926 | 0.810 / 0.823 | 0.830 / 0.712 | 0.818 / 0.820 |
| Centered-logit L2 distance | 0.055 / 1.000 | 0.146 / 1.000 | 0.300 / 0.993 | 0.167 / 0.998 |
| Energy gap | 0.446 / 1.000 | 0.482 / 1.000 | 0.449 / 0.998 | 0.459 / 0.999 |
| Absolute energy gap | 0.175 / 1.000 | 0.269 / 1.000 | 0.305 / 0.998 | 0.249 / 0.999 |
| Student MSP | 0.852 / 0.708 | 0.687 / 0.782 | 0.775 / 0.741 | 0.771 / 0.744 |
| Student energy | 0.879 / 0.624 | 0.763 / 0.679 | 0.770 / 0.731 | 0.804 / 0.678 |

</details>

<details>
<summary>Spatial, GAP</summary>

#### Centered-logit MSE

| Score | MNIST | SVHN | CIFAR-100 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.250 / 0.917 | 0.237 / 0.963 | 0.421 / 0.853 | 0.303 / 0.911 |
| Absolute max probability difference | 0.775 / 0.690 | 0.773 / 0.855 | 0.636 / 0.939 | 0.728 / 0.828 |
| KL (teacher / student) | 0.816 / 0.887 | 0.773 / 0.948 | 0.741 / 0.822 | 0.777 / 0.886 |
| Centered-logit L2 distance | 0.240 / 1.000 | 0.309 / 1.000 | 0.396 / 0.996 | 0.315 / 0.999 |
| Energy gap | 0.681 / 0.992 | 0.608 / 0.993 | 0.716 / 0.859 | 0.668 / 0.948 |
| Absolute energy gap | 0.310 / 1.000 | 0.386 / 0.998 | 0.292 / 0.996 | 0.329 / 0.998 |
| Student MSP | 0.915 / 0.458 | 0.872 / 0.536 | 0.750 / 0.848 | 0.846 / 0.614 |
| Student energy | 0.958 / 0.263 | 0.937 / 0.354 | 0.710 / 0.811 | 0.868 / 0.476 |

#### KL divergence

| Score | MNIST | SVHN | CIFAR-100 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.379 / 0.783 | 0.494 / 0.815 | 0.531 / 0.690 | 0.468 / 0.763 |
| Absolute max probability difference | 0.802 / 0.816 | 0.687 / 0.901 | 0.735 / 0.873 | 0.741 / 0.863 |
| KL (teacher / student) | 0.850 / 0.823 | 0.773 / 0.835 | 0.811 / 0.713 | 0.811 / 0.790 |
| Centered-logit L2 distance | 0.061 / 1.000 | 0.103 / 1.000 | 0.347 / 0.987 | 0.171 / 0.996 |
| Energy gap | 0.301 / 1.000 | 0.320 / 1.000 | 0.440 / 0.963 | 0.354 / 0.988 |
| Absolute energy gap | 0.194 / 1.000 | 0.129 / 1.000 | 0.321 / 0.966 | 0.215 / 0.989 |
| Student MSP | 0.849 / 0.627 | 0.761 / 0.749 | 0.732 / 0.824 | 0.781 / 0.733 |
| Student energy | 0.935 / 0.332 | 0.862 / 0.606 | 0.736 / 0.845 | 0.844 / 0.594 |

</details>

<details>
<summary>Spatial, Flatten</summary>

#### Centered-logit MSE

| Score | MNIST | SVHN | CIFAR-100 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.521 / 0.691 | 0.666 / 0.724 | 0.659 / 0.655 | 0.615 / 0.690 |
| Absolute max probability difference | 0.831 / 0.838 | 0.765 / 0.913 | 0.772 / 0.895 | 0.789 / 0.882 |
| KL (teacher / student) | 0.850 / 0.813 | 0.791 / 0.919 | 0.818 / 0.745 | 0.820 / 0.826 |
| Centered-logit L2 distance | 0.085 / 1.000 | 0.100 / 1.000 | 0.333 / 0.990 | 0.173 / 0.997 |
| Energy gap | 0.182 / 1.000 | 0.223 / 1.000 | 0.347 / 0.991 | 0.251 / 0.997 |
| Absolute energy gap | 0.172 / 1.000 | 0.213 / 1.000 | 0.343 / 0.991 | 0.243 / 0.997 |
| Student MSP | 0.813 / 0.784 | 0.730 / 0.855 | 0.714 / 0.912 | 0.753 / 0.850 |
| Student energy | 0.934 / 0.329 | 0.878 / 0.517 | 0.745 / 0.830 | 0.852 / 0.559 |

#### KL divergence

| Score | MNIST | SVHN | CIFAR-100 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.542 / 0.701 | 0.693 / 0.718 | 0.635 / 0.671 | 0.624 / 0.696 |
| Absolute max probability difference | 0.843 / 0.827 | 0.813 / 0.874 | 0.823 / 0.811 | 0.826 / 0.837 |
| KL (teacher / student) | 0.839 / 0.874 | 0.834 / 0.875 | 0.843 / 0.713 | 0.839 / 0.821 |
| Centered-logit L2 distance | 0.075 / 1.000 | 0.101 / 1.000 | 0.304 / 0.989 | 0.160 / 0.996 |
| Energy gap | 0.155 / 1.000 | 0.195 / 1.000 | 0.310 / 0.994 | 0.220 / 0.998 |
| Absolute energy gap | 0.155 / 1.000 | 0.195 / 1.000 | 0.310 / 0.994 | 0.220 / 0.998 |
| Student MSP | 0.846 / 0.751 | 0.762 / 0.802 | 0.766 / 0.844 | 0.791 / 0.799 |
| Student energy | 0.943 / 0.272 | 0.889 / 0.441 | 0.781 / 0.741 | 0.871 / 0.485 |

</details>

<details>
<summary>Channel, GAP</summary>

#### Centered-logit MSE

| Score | MNIST | SVHN | CIFAR-100 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.531 / 0.665 | 0.752 / 0.513 | 0.496 / 0.729 | 0.593 / 0.635 |
| Absolute max probability difference | 0.692 / 0.939 | 0.593 / 0.941 | 0.680 / 0.905 | 0.655 / 0.928 |
| KL (teacher / student) | 0.748 / 0.926 | 0.706 / 0.822 | 0.754 / 0.837 | 0.736 / 0.862 |
| Centered-logit L2 distance | 0.065 / 1.000 | 0.224 / 1.000 | 0.347 / 0.984 | 0.212 / 0.995 |
| Energy gap | 0.689 / 0.990 | 0.818 / 0.923 | 0.654 / 0.978 | 0.720 / 0.964 |
| Absolute energy gap | 0.338 / 1.000 | 0.491 / 0.996 | 0.362 / 0.991 | 0.397 / 0.995 |
| Student MSP | 0.710 / 0.893 | 0.456 / 0.924 | 0.695 / 0.824 | 0.620 / 0.880 |
| Student energy | 0.824 / 0.644 | 0.505 / 0.873 | 0.708 / 0.779 | 0.679 / 0.766 |

#### KL divergence

| Score | MNIST | SVHN | CIFAR-100 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.499 / 0.744 | 0.683 / 0.608 | 0.455 / 0.763 | 0.546 / 0.705 |
| Absolute max probability difference | 0.629 / 0.982 | 0.498 / 0.966 | 0.670 / 0.902 | 0.599 / 0.950 |
| KL (teacher / student) | 0.727 / 0.893 | 0.624 / 0.834 | 0.782 / 0.760 | 0.711 / 0.829 |
| Centered-logit L2 distance | 0.050 / 1.000 | 0.063 / 1.000 | 0.306 / 0.988 | 0.139 / 0.996 |
| Energy gap | 0.735 / 0.897 | 0.783 / 0.890 | 0.656 / 0.888 | 0.725 / 0.892 |
| Absolute energy gap | 0.268 / 1.000 | 0.224 / 1.000 | 0.359 / 0.991 | 0.284 / 0.997 |
| Student MSP | 0.744 / 0.919 | 0.571 / 0.906 | 0.727 / 0.824 | 0.681 / 0.883 |
| Student energy | 0.835 / 0.760 | 0.659 / 0.798 | 0.745 / 0.774 | 0.746 / 0.777 |

</details>

<details>
<summary>Channel, Flatten</summary>

#### Centered-logit MSE

| Score | MNIST | SVHN | CIFAR-100 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.542 / 0.686 | 0.734 / 0.618 | 0.544 / 0.698 | 0.607 / 0.667 |
| Absolute max probability difference | 0.772 / 0.944 | 0.740 / 0.900 | 0.771 / 0.861 | 0.761 / 0.902 |
| KL (teacher / student) | 0.801 / 0.900 | 0.801 / 0.800 | 0.818 / 0.753 | 0.807 / 0.818 |
| Centered-logit L2 distance | 0.201 / 1.000 | 0.263 / 1.000 | 0.361 / 0.988 | 0.275 / 0.996 |
| Energy gap | 0.492 / 1.000 | 0.594 / 0.992 | 0.471 / 0.991 | 0.519 / 0.994 |
| Absolute energy gap | 0.259 / 1.000 | 0.473 / 0.992 | 0.379 / 0.991 | 0.370 / 0.994 |
| Student MSP | 0.800 / 0.864 | 0.627 / 0.889 | 0.744 / 0.822 | 0.724 / 0.858 |
| Student energy | 0.840 / 0.796 | 0.664 / 0.817 | 0.753 / 0.781 | 0.752 / 0.798 |

#### KL divergence

| Score | MNIST | SVHN | CIFAR-100 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.583 / 0.714 | 0.687 / 0.590 | 0.459 / 0.771 | 0.576 / 0.692 |
| Absolute max probability difference | 0.557 / 0.977 | 0.497 / 0.943 | 0.659 / 0.878 | 0.571 / 0.933 |
| KL (teacher / student) | 0.659 / 0.895 | 0.624 / 0.825 | 0.776 / 0.759 | 0.686 / 0.826 |
| Centered-logit L2 distance | 0.076 / 1.000 | 0.146 / 1.000 | 0.343 / 0.984 | 0.188 / 0.995 |
| Energy gap | 0.734 / 0.919 | 0.783 / 0.886 | 0.649 / 0.885 | 0.722 / 0.897 |
| Absolute energy gap | 0.268 / 1.000 | 0.240 / 1.000 | 0.375 / 0.991 | 0.294 / 0.997 |
| Student MSP | 0.668 / 0.907 | 0.562 / 0.874 | 0.719 / 0.783 | 0.650 / 0.855 |
| Student energy | 0.816 / 0.808 | 0.653 / 0.801 | 0.737 / 0.784 | 0.735 / 0.798 |

</details>

### CIFAR-10 ID, ResNet-50

<details>
<summary>Constant, GAP</summary>

#### Centered-logit MSE

| Score | MNIST | SVHN | CIFAR-100 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.352 / 0.859 | 0.474 / 0.859 | 0.466 / 0.831 | 0.431 / 0.849 |
| Absolute max probability difference | 0.697 / 0.783 | 0.573 / 0.988 | 0.599 / 0.942 | 0.623 / 0.905 |
| KL (teacher / student) | 0.807 / 0.821 | 0.669 / 0.933 | 0.735 / 0.794 | 0.737 / 0.849 |
| Centered-logit L2 distance | 0.272 / 0.998 | 0.193 / 1.000 | 0.373 / 0.981 | 0.279 / 0.993 |
| Energy gap | 0.710 / 0.909 | 0.886 / 0.702 | 0.777 / 0.755 | 0.791 / 0.789 |
| Absolute energy gap | 0.290 / 0.998 | 0.113 / 1.000 | 0.223 / 0.993 | 0.209 / 0.997 |
| Student MSP | 0.800 / 0.696 | 0.663 / 0.976 | 0.680 / 0.890 | 0.714 / 0.854 |
| Student energy | 0.858 / 0.725 | 0.464 / 0.991 | 0.609 / 0.937 | 0.644 / 0.884 |

#### KL divergence

| Score | MNIST | SVHN | CIFAR-100 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.352 / 0.978 | 0.520 / 0.949 | 0.501 / 0.944 | 0.457 / 0.957 |
| Absolute max probability difference | 0.648 / 0.772 | 0.480 / 0.996 | 0.499 / 0.955 | 0.543 / 0.908 |
| KL (teacher / student) | 0.751 / 0.689 | 0.587 / 0.987 | 0.617 / 0.891 | 0.651 / 0.856 |
| Centered-logit L2 distance | 0.128 / 1.000 | 0.103 / 0.999 | 0.250 / 0.983 | 0.160 / 0.994 |
| Energy gap | 0.917 / 0.459 | 0.919 / 0.504 | 0.868 / 0.528 | 0.901 / 0.497 |
| Absolute energy gap | 0.083 / 1.000 | 0.081 / 1.000 | 0.132 / 0.996 | 0.099 / 0.999 |
| Student MSP | 0.874 / 0.577 | 0.642 / 0.973 | 0.699 / 0.857 | 0.738 / 0.802 |
| Student energy | 0.883 / 0.683 | 0.482 / 0.988 | 0.628 / 0.925 | 0.664 / 0.865 |

</details>

<details>
<summary>Constant, Flatten</summary>

#### Centered-logit MSE

| Score | MNIST | SVHN | CIFAR-100 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.387 / 0.775 | 0.468 / 0.797 | 0.478 / 0.763 | 0.444 / 0.778 |
| Absolute max probability difference | 0.728 / 0.794 | 0.631 / 0.979 | 0.643 / 0.915 | 0.667 / 0.896 |
| KL (teacher / student) | 0.814 / 0.807 | 0.700 / 0.960 | 0.744 / 0.812 | 0.753 / 0.860 |
| Centered-logit L2 distance | 0.267 / 1.000 | 0.253 / 1.000 | 0.414 / 0.988 | 0.311 / 0.996 |
| Energy gap | 0.615 / 0.983 | 0.803 / 0.888 | 0.701 / 0.879 | 0.706 / 0.916 |
| Absolute energy gap | 0.370 / 0.998 | 0.189 / 1.000 | 0.309 / 0.993 | 0.289 / 0.997 |
| Student MSP | 0.777 / 0.702 | 0.682 / 0.948 | 0.688 / 0.846 | 0.715 / 0.832 |
| Student energy | 0.839 / 0.776 | 0.519 / 0.991 | 0.647 / 0.933 | 0.668 / 0.900 |

#### KL divergence

| Score | MNIST | SVHN | CIFAR-100 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.404 / 0.719 | 0.414 / 0.768 | 0.508 / 0.710 | 0.442 / 0.732 |
| Absolute max probability difference | 0.857 / 0.766 | 0.804 / 0.863 | 0.780 / 0.826 | 0.814 / 0.818 |
| KL (teacher / student) | 0.863 / 0.865 | 0.817 / 0.913 | 0.820 / 0.746 | 0.834 / 0.841 |
| Centered-logit L2 distance | 0.098 / 1.000 | 0.202 / 1.000 | 0.414 / 0.990 | 0.238 / 0.997 |
| Energy gap | 0.385 / 1.000 | 0.480 / 1.000 | 0.495 / 0.994 | 0.453 / 0.998 |
| Absolute energy gap | 0.306 / 1.000 | 0.143 / 1.000 | 0.341 / 0.994 | 0.264 / 0.998 |
| Student MSP | 0.836 / 0.721 | 0.811 / 0.811 | 0.750 / 0.796 | 0.799 / 0.776 |
| Student energy | 0.878 / 0.650 | 0.765 / 0.927 | 0.726 / 0.834 | 0.790 / 0.804 |

</details>

<details>
<summary>Spatial, GAP</summary>

#### Centered-logit MSE

| Score | MNIST | SVHN | CIFAR-100 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.412 / 0.963 | 0.549 / 0.944 | 0.589 / 0.890 | 0.517 / 0.932 |
| Absolute max probability difference | 0.588 / 0.966 | 0.451 / 0.997 | 0.413 / 0.987 | 0.484 / 0.983 |
| KL (teacher / student) | 0.693 / 0.966 | 0.546 / 0.998 | 0.544 / 0.966 | 0.594 / 0.977 |
| Centered-logit L2 distance | 0.304 / 0.998 | 0.178 / 0.999 | 0.289 / 0.989 | 0.257 / 0.995 |
| Energy gap | 0.893 / 0.582 | 0.947 / 0.360 | 0.870 / 0.526 | 0.903 / 0.489 |
| Absolute energy gap | 0.107 / 1.000 | 0.053 / 1.000 | 0.130 / 0.997 | 0.097 / 0.999 |
| Student MSP | 0.792 / 0.894 | 0.595 / 0.990 | 0.575 / 0.961 | 0.654 / 0.948 |
| Student energy | 0.820 / 0.971 | 0.269 / 0.998 | 0.531 / 0.984 | 0.540 / 0.985 |

#### KL divergence

| Score | MNIST | SVHN | CIFAR-100 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.447 / 0.920 | 0.740 / 0.822 | 0.615 / 0.835 | 0.601 / 0.859 |
| Absolute max probability difference | 0.553 / 0.964 | 0.260 / 1.000 | 0.385 / 0.988 | 0.399 / 0.984 |
| KL (teacher / student) | 0.534 / 1.000 | 0.611 / 0.943 | 0.513 / 0.946 | 0.552 / 0.963 |
| Centered-logit L2 distance | 0.152 / 1.000 | 0.125 / 0.999 | 0.254 / 0.991 | 0.177 / 0.997 |
| Energy gap | 0.915 / 0.495 | 0.942 / 0.392 | 0.874 / 0.519 | 0.910 / 0.469 |
| Absolute energy gap | 0.085 / 1.000 | 0.058 / 1.000 | 0.126 / 0.997 | 0.090 / 0.999 |
| Student MSP | 0.797 / 0.824 | 0.375 / 0.998 | 0.575 / 0.958 | 0.582 / 0.927 |
| Student energy | 0.794 / 0.942 | 0.162 / 0.999 | 0.481 / 0.978 | 0.479 / 0.973 |

</details>

<details>
<summary>Spatial, Flatten</summary>

#### Centered-logit MSE

| Score | MNIST | SVHN | CIFAR-100 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.597 / 0.890 | 0.581 / 0.936 | 0.602 / 0.872 | 0.594 / 0.899 |
| Absolute max probability difference | 0.417 / 0.998 | 0.424 / 0.999 | 0.416 / 0.985 | 0.419 / 0.994 |
| KL (teacher / student) | 0.770 / 0.981 | 0.499 / 0.997 | 0.569 / 0.952 | 0.612 / 0.976 |
| Centered-logit L2 distance | 0.392 / 0.995 | 0.343 / 0.997 | 0.364 / 0.984 | 0.366 / 0.992 |
| Energy gap | 0.884 / 0.612 | 0.953 / 0.293 | 0.856 / 0.544 | 0.898 / 0.483 |
| Absolute energy gap | 0.116 / 1.000 | 0.047 / 1.000 | 0.144 / 0.997 | 0.102 / 0.999 |
| Student MSP | 0.523 / 0.986 | 0.523 / 0.998 | 0.524 / 0.967 | 0.523 / 0.984 |
| Student energy | 0.666 / 0.988 | 0.282 / 0.999 | 0.507 / 0.975 | 0.485 / 0.987 |

#### KL divergence

| Score | MNIST | SVHN | CIFAR-100 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.340 / 0.978 | 0.516 / 0.957 | 0.512 / 0.915 | 0.456 / 0.950 |
| Absolute max probability difference | 0.660 / 0.998 | 0.484 / 1.000 | 0.490 / 0.993 | 0.545 / 0.997 |
| KL (teacher / student) | 0.709 / 0.965 | 0.679 / 0.996 | 0.605 / 0.935 | 0.664 / 0.965 |
| Centered-logit L2 distance | 0.297 / 1.000 | 0.267 / 0.997 | 0.329 / 0.987 | 0.297 / 0.994 |
| Energy gap | 0.893 / 0.567 | 0.947 / 0.356 | 0.866 / 0.533 | 0.902 / 0.485 |
| Absolute energy gap | 0.107 / 1.000 | 0.053 / 1.000 | 0.134 / 0.997 | 0.098 / 0.999 |
| Student MSP | 0.832 / 0.973 | 0.621 / 0.999 | 0.648 / 0.981 | 0.700 / 0.984 |
| Student energy | 0.789 / 0.993 | 0.293 / 1.000 | 0.551 / 0.994 | 0.544 / 0.996 |

</details>

<details>
<summary>Channel, GAP</summary>

#### Centered-logit MSE

| Score | MNIST | SVHN | CIFAR-100 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.422 / 0.811 | 0.350 / 0.865 | 0.518 / 0.761 | 0.430 / 0.812 |
| Absolute max probability difference | 0.669 / 0.923 | 0.716 / 0.861 | 0.594 / 0.930 | 0.660 / 0.905 |
| KL (teacher / student) | 0.682 / 0.999 | 0.598 / 0.999 | 0.596 / 0.980 | 0.625 / 0.993 |
| Centered-logit L2 distance | 0.249 / 1.000 | 0.173 / 1.000 | 0.301 / 0.988 | 0.241 / 0.996 |
| Energy gap | 0.769 / 0.704 | 0.789 / 0.757 | 0.772 / 0.701 | 0.777 / 0.721 |
| Absolute energy gap | 0.241 / 1.000 | 0.211 / 1.000 | 0.237 / 0.998 | 0.230 / 0.999 |
| Student MSP | 0.740 / 0.844 | 0.780 / 0.792 | 0.650 / 0.861 | 0.723 / 0.832 |
| Student energy | 0.720 / 0.841 | 0.629 / 0.981 | 0.622 / 0.900 | 0.657 / 0.907 |

#### KL divergence

| Score | MNIST | SVHN | CIFAR-100 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.354 / 0.898 | 0.375 / 0.911 | 0.450 / 0.841 | 0.393 / 0.884 |
| Absolute max probability difference | 0.678 / 0.902 | 0.654 / 0.946 | 0.602 / 0.928 | 0.644 / 0.925 |
| KL (teacher / student) | 0.722 / 1.000 | 0.608 / 0.998 | 0.673 / 0.947 | 0.668 / 0.982 |
| Centered-logit L2 distance | 0.134 / 1.000 | 0.087 / 1.000 | 0.269 / 0.992 | 0.163 / 0.997 |
| Energy gap | 0.879 / 0.538 | 0.862 / 0.615 | 0.829 / 0.630 | 0.856 / 0.594 |
| Absolute energy gap | 0.121 / 1.000 | 0.138 / 1.000 | 0.172 / 0.994 | 0.144 / 0.998 |
| Student MSP | 0.800 / 0.825 | 0.749 / 0.880 | 0.703 / 0.846 | 0.751 / 0.850 |
| Student energy | 0.729 / 0.786 | 0.676 / 0.867 | 0.674 / 0.768 | 0.693 / 0.807 |

</details>

<details>
<summary>Channel, Flatten</summary>

#### Centered-logit MSE

| Score | MNIST | SVHN | CIFAR-100 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.380 / 0.746 | 0.406 / 0.768 | 0.534 / 0.684 | 0.440 / 0.733 |
| Absolute max probability difference | 0.824 / 0.825 | 0.780 / 0.878 | 0.739 / 0.881 | 0.781 / 0.861 |
| KL (teacher / student) | 0.685 / 0.999 | 0.621 / 0.998 | 0.664 / 0.968 | 0.657 / 0.989 |
| Centered-logit L2 distance | 0.272 / 1.000 | 0.106 / 1.000 | 0.379 / 0.996 | 0.252 / 0.999 |
| Energy gap | 0.561 / 0.981 | 0.492 / 0.991 | 0.585 / 0.909 | 0.546 / 0.960 |
| Absolute energy gap | 0.416 / 0.986 | 0.365 / 0.992 | 0.483 / 0.914 | 0.422 / 0.964 |
| Student MSP | 0.828 / 0.736 | 0.793 / 0.830 | 0.695 / 0.852 | 0.772 / 0.806 |
| Student energy | 0.838 / 0.754 | 0.848 / 0.734 | 0.725 / 0.783 | 0.804 / 0.757 |

#### KL divergence

| Score | MNIST | SVHN | CIFAR-100 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.673 / 0.617 | 0.449 / 0.766 | 0.575 / 0.673 | 0.566 / 0.686 |
| Absolute max probability difference | 0.570 / 0.975 | 0.694 / 0.933 | 0.637 / 0.931 | 0.634 / 0.946 |
| KL (teacher / student) | 0.612 / 0.969 | 0.771 / 0.923 | 0.753 / 0.796 | 0.712 / 0.896 |
| Centered-logit L2 distance | 0.356 / 1.000 | 0.183 / 1.000 | 0.409 / 0.996 | 0.316 / 0.999 |
| Energy gap | 0.819 / 0.983 | 0.626 / 0.985 | 0.671 / 0.949 | 0.705 / 0.972 |
| Absolute energy gap | 0.471 / 0.994 | 0.358 / 0.990 | 0.423 / 0.962 | 0.418 / 0.982 |
| Student MSP | 0.533 / 0.952 | 0.709 / 0.897 | 0.606 / 0.910 | 0.616 / 0.920 |
| Student energy | 0.519 / 0.979 | 0.740 / 0.825 | 0.641 / 0.874 | 0.633 / 0.892 |

</details>

### CIFAR-100 ID, ResNet-18

<details>
<summary>Constant, GAP</summary>

#### Centered-logit MSE

| Score | MNIST | SVHN | CIFAR-10 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.687 / 0.955 | 0.809 / 0.813 | 0.786 / 0.794 | 0.761 / 0.854 |
| Absolute max probability difference | 0.313 / 0.999 | 0.191 / 1.000 | 0.214 / 0.999 | 0.239 / 0.999 |
| KL (teacher / student) | 0.349 / 1.000 | 0.170 / 1.000 | 0.223 / 0.997 | 0.247 / 0.999 |
| Centered-logit L2 distance | 0.397 / 1.000 | 0.174 / 1.000 | 0.228 / 0.999 | 0.267 / 0.999 |
| Energy gap | 0.695 / 0.974 | 0.828 / 0.779 | 0.795 / 0.800 | 0.773 / 0.851 |
| Absolute energy gap | 0.305 / 1.000 | 0.172 / 1.000 | 0.205 / 0.999 | 0.227 / 1.000 |
| Student MSP | 0.653 / 0.983 | 0.670 / 0.785 | 0.706 / 0.858 | 0.676 / 0.875 |
| Student energy | 0.470 / 1.000 | 0.672 / 0.932 | 0.624 / 0.974 | 0.589 / 0.969 |

#### KL divergence

| Score | MNIST | SVHN | CIFAR-10 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.669 / 0.955 | 0.811 / 0.808 | 0.775 / 0.795 | 0.752 / 0.853 |
| Absolute max probability difference | 0.331 / 1.000 | 0.189 / 1.000 | 0.225 / 0.999 | 0.248 / 1.000 |
| KL (teacher / student) | 0.421 / 1.000 | 0.170 / 1.000 | 0.257 / 0.996 | 0.283 / 0.999 |
| Centered-logit L2 distance | 0.409 / 0.999 | 0.169 / 1.000 | 0.236 / 0.999 | 0.272 / 0.999 |
| Energy gap | 0.692 / 0.975 | 0.829 / 0.778 | 0.795 / 0.799 | 0.772 / 0.851 |
| Absolute energy gap | 0.308 / 1.000 | 0.171 / 1.000 | 0.205 / 0.999 | 0.228 / 1.000 |
| Student MSP | 0.576 / 0.979 | 0.557 / 0.929 | 0.701 / 0.853 | 0.611 / 0.921 |
| Student energy | 0.697 / 0.968 | 0.682 / 0.867 | 0.680 / 0.927 | 0.686 / 0.921 |

</details>

<details>
<summary>Constant, Flatten</summary>

#### Centered-logit MSE

| Score | MNIST | SVHN | CIFAR-10 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.489 / 0.974 | 0.633 / 0.894 | 0.665 / 0.905 | 0.596 / 0.924 |
| Absolute max probability difference | 0.511 / 0.999 | 0.366 / 0.996 | 0.335 / 0.991 | 0.404 / 0.995 |
| KL (teacher / student) | 0.618 / 0.989 | 0.478 / 0.975 | 0.392 / 0.982 | 0.496 / 0.982 |
| Centered-logit L2 distance | 0.442 / 0.999 | 0.269 / 1.000 | 0.237 / 0.998 | 0.316 / 0.999 |
| Energy gap | 0.677 / 0.980 | 0.809 / 0.818 | 0.800 / 0.796 | 0.762 / 0.864 |
| Absolute energy gap | 0.323 / 1.000 | 0.191 / 1.000 | 0.200 / 1.000 | 0.238 / 1.000 |
| Student MSP | 0.643 / 0.982 | 0.746 / 0.891 | 0.713 / 0.839 | 0.700 / 0.904 |
| Student energy | 0.619 / 0.983 | 0.770 / 0.854 | 0.681 / 0.867 | 0.690 / 0.902 |

#### KL divergence

| Score | MNIST | SVHN | CIFAR-10 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.547 / 0.833 | 0.576 / 0.752 | 0.623 / 0.855 | 0.582 / 0.813 |
| Absolute max probability difference | 0.664 / 0.950 | 0.743 / 0.844 | 0.614 / 0.949 | 0.674 / 0.914 |
| KL (teacher / student) | 0.736 / 0.857 | 0.886 / 0.540 | 0.716 / 0.930 | 0.780 / 0.776 |
| Centered-logit L2 distance | 0.361 / 1.000 | 0.461 / 0.994 | 0.300 / 0.997 | 0.374 / 0.997 |
| Energy gap | 0.520 / 1.000 | 0.487 / 0.999 | 0.514 / 0.989 | 0.507 / 0.996 |
| Absolute energy gap | 0.436 / 1.000 | 0.393 / 0.999 | 0.430 / 0.990 | 0.420 / 0.996 |
| Student MSP | 0.652 / 0.985 | 0.770 / 0.883 | 0.727 / 0.832 | 0.716 / 0.900 |
| Student energy | 0.641 / 0.981 | 0.766 / 0.891 | 0.725 / 0.837 | 0.710 / 0.903 |

</details>

<details>
<summary>Spatial, GAP</summary>

#### Centered-logit MSE

| Score | MNIST | SVHN | CIFAR-10 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.684 / 0.955 | 0.811 / 0.811 | 0.779 / 0.796 | 0.758 / 0.854 |
| Absolute max probability difference | 0.316 / 1.000 | 0.189 / 1.000 | 0.221 / 0.992 | 0.242 / 0.997 |
| KL (teacher / student) | 0.348 / 1.000 | 0.160 / 1.000 | 0.247 / 0.997 | 0.252 / 0.999 |
| Centered-logit L2 distance | 0.428 / 0.999 | 0.155 / 1.000 | 0.248 / 0.999 | 0.277 / 0.999 |
| Energy gap | 0.692 / 0.975 | 0.828 / 0.778 | 0.794 / 0.800 | 0.772 / 0.851 |
| Absolute energy gap | 0.308 / 1.000 | 0.172 / 1.000 | 0.206 / 0.999 | 0.228 / 1.000 |
| Student MSP | 0.523 / 0.998 | 0.499 / 0.936 | 0.682 / 0.871 | 0.568 / 0.935 |
| Student energy | 0.579 / 0.992 | 0.624 / 0.890 | 0.633 / 0.937 | 0.612 / 0.940 |

#### KL divergence

| Score | MNIST | SVHN | CIFAR-10 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.624 / 0.965 | 0.818 / 0.813 | 0.748 / 0.815 | 0.730 / 0.864 |
| Absolute max probability difference | 0.376 / 1.000 | 0.182 / 1.000 | 0.252 / 0.993 | 0.270 / 0.998 |
| KL (teacher / student) | 0.402 / 0.998 | 0.136 / 1.000 | 0.289 / 0.996 | 0.276 / 0.998 |
| Centered-logit L2 distance | 0.441 / 1.000 | 0.170 / 1.000 | 0.249 / 0.999 | 0.287 / 1.000 |
| Energy gap | 0.682 / 0.978 | 0.839 / 0.753 | 0.785 / 0.813 | 0.769 / 0.848 |
| Absolute energy gap | 0.318 / 1.000 | 0.161 / 1.000 | 0.215 / 0.999 | 0.231 / 1.000 |
| Student MSP | 0.687 / 0.960 | 0.431 / 0.954 | 0.692 / 0.857 | 0.603 / 0.924 |
| Student energy | 0.545 / 0.996 | 0.454 / 0.956 | 0.707 / 0.815 | 0.568 / 0.922 |

</details>

<details>
<summary>Spatial, Flatten</summary>

#### Centered-logit MSE

| Score | MNIST | SVHN | CIFAR-10 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.651 / 0.954 | 0.810 / 0.812 | 0.770 / 0.797 | 0.744 / 0.854 |
| Absolute max probability difference | 0.349 / 0.999 | 0.190 / 1.000 | 0.230 / 0.998 | 0.256 / 0.999 |
| KL (teacher / student) | 0.383 / 1.000 | 0.157 / 1.000 | 0.265 / 0.996 | 0.268 / 0.998 |
| Centered-logit L2 distance | 0.430 / 1.000 | 0.141 / 1.000 | 0.253 / 0.999 | 0.275 / 0.999 |
| Energy gap | 0.689 / 0.975 | 0.827 / 0.788 | 0.790 / 0.804 | 0.769 / 0.856 |
| Absolute energy gap | 0.311 / 1.000 | 0.173 / 1.000 | 0.210 / 0.999 | 0.231 / 1.000 |
| Student MSP | 0.611 / 0.993 | 0.548 / 0.908 | 0.689 / 0.858 | 0.616 / 0.919 |
| Student energy | 0.547 / 0.995 | 0.650 / 0.865 | 0.683 / 0.877 | 0.627 / 0.912 |

#### KL divergence

| Score | MNIST | SVHN | CIFAR-10 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.595 / 0.861 | 0.832 / 0.682 | 0.645 / 0.914 | 0.691 / 0.819 |
| Absolute max probability difference | 0.457 / 1.000 | 0.317 / 1.000 | 0.372 / 0.994 | 0.382 / 0.998 |
| KL (teacher / student) | 0.505 / 0.991 | 0.317 / 0.999 | 0.427 / 0.994 | 0.416 / 0.995 |
| Centered-logit L2 distance | 0.538 / 1.000 | 0.316 / 0.998 | 0.271 / 0.998 | 0.375 / 0.999 |
| Energy gap | 0.660 / 1.000 | 0.788 / 0.944 | 0.633 / 0.996 | 0.694 / 0.980 |
| Absolute energy gap | 0.326 / 1.000 | 0.440 / 0.981 | 0.274 / 0.998 | 0.347 / 0.993 |
| Student MSP | 0.491 / 0.999 | 0.433 / 0.967 | 0.656 / 0.877 | 0.526 / 0.947 |
| Student energy | 0.473 / 1.000 | 0.519 / 0.956 | 0.689 / 0.864 | 0.561 / 0.940 |

</details>

<details>
<summary>Channel, GAP</summary>

#### Centered-logit MSE

| Score | MNIST | SVHN | CIFAR-10 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.639 / 0.976 | 0.747 / 0.870 | 0.725 / 0.849 | 0.704 / 0.898 |
| Absolute max probability difference | 0.361 / 1.000 | 0.253 / 1.000 | 0.275 / 0.973 | 0.296 / 0.991 |
| KL (teacher / student) | 0.357 / 1.000 | 0.269 / 1.000 | 0.376 / 0.952 | 0.334 / 0.984 |
| Centered-logit L2 distance | 0.312 / 1.000 | 0.190 / 1.000 | 0.354 / 0.995 | 0.285 / 0.998 |
| Energy gap | 0.695 / 0.970 | 0.818 / 0.800 | 0.783 / 0.822 | 0.766 / 0.864 |
| Absolute energy gap | 0.305 / 1.000 | 0.182 / 1.000 | 0.217 / 0.998 | 0.234 / 0.999 |
| Student MSP | 0.514 / 0.999 | 0.706 / 0.930 | 0.718 / 0.861 | 0.646 / 0.930 |
| Student energy | 0.413 / 1.000 | 0.749 / 0.877 | 0.765 / 0.783 | 0.642 / 0.886 |

#### KL divergence

| Score | MNIST | SVHN | CIFAR-10 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.534 / 0.973 | 0.639 / 0.963 | 0.661 / 0.867 | 0.611 / 0.934 |
| Absolute max probability difference | 0.461 / 1.000 | 0.349 / 0.999 | 0.347 / 0.985 | 0.386 / 0.994 |
| KL (teacher / student) | 0.416 / 1.000 | 0.336 / 0.999 | 0.532 / 0.936 | 0.428 / 0.978 |
| Centered-logit L2 distance | 0.429 / 1.000 | 0.289 / 0.999 | 0.316 / 0.999 | 0.345 / 0.999 |
| Energy gap | 0.646 / 0.999 | 0.733 / 0.996 | 0.703 / 0.971 | 0.694 / 0.989 |
| Absolute energy gap | 0.298 / 1.000 | 0.209 / 1.000 | 0.261 / 0.997 | 0.256 / 0.999 |
| Student MSP | 0.607 / 0.979 | 0.741 / 0.790 | 0.639 / 0.918 | 0.662 / 0.896 |
| Student energy | 0.502 / 0.999 | 0.761 / 0.797 | 0.746 / 0.852 | 0.669 / 0.883 |

</details>

<details>
<summary>Channel, Flatten</summary>

#### Centered-logit MSE

| Score | MNIST | SVHN | CIFAR-10 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.605 / 0.988 | 0.654 / 0.949 | 0.631 / 0.971 | 0.630 / 0.969 |
| Absolute max probability difference | 0.392 / 1.000 | 0.342 / 0.995 | 0.366 / 0.988 | 0.367 / 0.994 |
| KL (teacher / student) | 0.387 / 1.000 | 0.393 / 0.995 | 0.470 / 0.965 | 0.417 / 0.987 |
| Centered-logit L2 distance | 0.304 / 1.000 | 0.194 / 1.000 | 0.358 / 0.996 | 0.285 / 0.999 |
| Energy gap | 0.690 / 0.968 | 0.777 / 0.900 | 0.739 / 0.937 | 0.735 / 0.935 |
| Absolute energy gap | 0.296 / 1.000 | 0.207 / 1.000 | 0.247 / 0.998 | 0.250 / 0.999 |
| Student MSP | 0.450 / 0.999 | 0.684 / 0.930 | 0.750 / 0.851 | 0.628 / 0.927 |
| Student energy | 0.397 / 1.000 | 0.783 / 0.845 | 0.778 / 0.830 | 0.653 / 0.891 |

#### KL divergence

| Score | MNIST | SVHN | CIFAR-10 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.508 / 0.949 | 0.640 / 0.795 | 0.556 / 0.909 | 0.568 / 0.884 |
| Absolute max probability difference | 0.522 / 0.985 | 0.543 / 0.963 | 0.524 / 0.970 | 0.530 / 0.973 |
| KL (teacher / student) | 0.618 / 0.974 | 0.667 / 0.954 | 0.652 / 0.976 | 0.645 / 0.968 |
| Centered-logit L2 distance | 0.563 / 0.997 | 0.188 / 0.999 | 0.260 / 0.999 | 0.337 / 0.998 |
| Energy gap | 0.665 / 1.000 | 0.548 / 1.000 | 0.502 / 1.000 | 0.572 / 1.000 |
| Absolute energy gap | 0.565 / 1.000 | 0.444 / 1.000 | 0.361 / 1.000 | 0.456 / 1.000 |
| Student MSP | 0.620 / 0.969 | 0.659 / 0.904 | 0.747 / 0.839 | 0.675 / 0.904 |
| Student energy | 0.524 / 0.995 | 0.781 / 0.784 | 0.794 / 0.811 | 0.700 / 0.864 |

</details>

### CIFAR-100 ID, ResNet-50

<details>
<summary>Constant, GAP</summary>

#### Centered-logit MSE

| Score | MNIST | SVHN | CIFAR-10 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.805 / 0.825 | 0.715 / 0.793 | 0.782 / 0.782 | 0.767 / 0.800 |
| Absolute max probability difference | 0.195 / 1.000 | 0.285 / 0.940 | 0.218 / 0.997 | 0.233 / 0.979 |
| KL (teacher / student) | 0.173 / 1.000 | 0.322 / 1.000 | 0.272 / 0.984 | 0.256 / 0.994 |
| Centered-logit L2 distance | 0.048 / 1.000 | 0.295 / 0.999 | 0.282 / 0.991 | 0.208 / 0.997 |
| Energy gap | 0.877 / 0.673 | 0.782 / 0.765 | 0.801 / 0.771 | 0.820 / 0.736 |
| Absolute energy gap | 0.123 / 1.000 | 0.218 / 0.999 | 0.199 / 0.996 | 0.180 / 0.999 |
| Student MSP | 0.814 / 0.518 | 0.707 / 0.808 | 0.484 / 0.982 | 0.668 / 0.769 |
| Student energy | 0.826 / 0.912 | 0.691 / 0.977 | 0.418 / 0.985 | 0.645 / 0.958 |

#### KL divergence

| Score | MNIST | SVHN | CIFAR-10 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.797 / 0.826 | 0.746 / 0.794 | 0.768 / 0.783 | 0.770 / 0.801 |
| Absolute max probability difference | 0.203 / 1.000 | 0.254 / 1.000 | 0.232 / 0.998 | 0.230 / 0.999 |
| KL (teacher / student) | 0.254 / 1.000 | 0.273 / 0.998 | 0.285 / 0.987 | 0.271 / 0.995 |
| Centered-logit L2 distance | 0.050 / 1.000 | 0.297 / 0.999 | 0.296 / 0.990 | 0.214 / 0.996 |
| Energy gap | 0.878 / 0.671 | 0.784 / 0.765 | 0.800 / 0.773 | 0.821 / 0.736 |
| Absolute energy gap | 0.122 / 1.000 | 0.216 / 0.999 | 0.200 / 0.996 | 0.179 / 0.999 |
| Student MSP | 0.819 / 0.743 | 0.582 / 0.968 | 0.695 / 0.919 | 0.699 / 0.877 |
| Student energy | 0.687 / 1.000 | 0.476 / 0.999 | 0.654 / 0.876 | 0.606 / 0.959 |

</details>

<details>
<summary>Constant, Flatten</summary>

#### Centered-logit MSE

| Score | MNIST | SVHN | CIFAR-10 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.580 / 0.912 | 0.627 / 0.869 | 0.615 / 0.889 | 0.607 / 0.890 |
| Absolute max probability difference | 0.420 / 0.979 | 0.373 / 0.990 | 0.385 / 0.985 | 0.393 / 0.984 |
| KL (teacher / student) | 0.524 / 0.945 | 0.525 / 0.943 | 0.508 / 0.970 | 0.519 / 0.953 |
| Centered-logit L2 distance | 0.047 / 1.000 | 0.441 / 0.979 | 0.274 / 0.994 | 0.254 / 0.991 |
| Energy gap | 0.849 / 0.794 | 0.775 / 0.747 | 0.784 / 0.797 | 0.803 / 0.779 |
| Absolute energy gap | 0.151 / 1.000 | 0.225 / 1.000 | 0.216 / 0.997 | 0.197 / 0.999 |
| Student MSP | 0.746 / 0.887 | 0.634 / 0.922 | 0.714 / 0.869 | 0.698 / 0.892 |
| Student energy | 0.800 / 0.763 | 0.569 / 0.979 | 0.621 / 0.951 | 0.664 / 0.898 |

#### KL divergence

| Score | MNIST | SVHN | CIFAR-10 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.435 / 0.918 | 0.550 / 0.783 | 0.486 / 0.864 | 0.490 / 0.855 |
| Absolute max probability difference | 0.575 / 0.946 | 0.530 / 0.985 | 0.549 / 0.955 | 0.552 / 0.962 |
| KL (teacher / student) | 0.665 / 0.949 | 0.763 / 0.778 | 0.683 / 0.941 | 0.704 / 0.889 |
| Centered-logit L2 distance | 0.178 / 1.000 | 0.708 / 0.965 | 0.389 / 0.991 | 0.425 / 0.985 |
| Energy gap | 0.792 / 0.982 | 0.805 / 0.773 | 0.723 / 0.968 | 0.773 / 0.908 |
| Absolute energy gap | 0.166 / 1.000 | 0.237 / 0.998 | 0.251 / 0.997 | 0.218 / 0.998 |
| Student MSP | 0.734 / 0.853 | 0.624 / 0.950 | 0.706 / 0.842 | 0.688 / 0.882 |
| Student energy | 0.759 / 0.748 | 0.479 / 0.976 | 0.682 / 0.875 | 0.640 / 0.866 |

</details>

<details>
<summary>Spatial, GAP</summary>

#### Centered-logit MSE

| Score | MNIST | SVHN | CIFAR-10 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.801 / 0.827 | 0.720 / 0.795 | 0.793 / 0.781 | 0.771 / 0.801 |
| Absolute max probability difference | 0.199 / 0.999 | 0.280 / 0.987 | 0.207 / 0.999 | 0.229 / 0.995 |
| KL (teacher / student) | 0.162 / 1.000 | 0.179 / 1.000 | 0.234 / 0.995 | 0.192 / 0.998 |
| Centered-logit L2 distance | 0.039 / 1.000 | 0.227 / 0.999 | 0.286 / 0.992 | 0.184 / 0.997 |
| Energy gap | 0.878 / 0.671 | 0.783 / 0.764 | 0.801 / 0.772 | 0.821 / 0.736 |
| Absolute energy gap | 0.122 / 1.000 | 0.217 / 0.999 | 0.199 / 0.996 | 0.179 / 0.999 |
| Student MSP | 0.810 / 0.467 | 0.717 / 0.744 | 0.474 / 0.978 | 0.667 / 0.730 |
| Student energy | 0.666 / 0.806 | 0.568 / 0.976 | 0.459 / 0.972 | 0.564 / 0.918 |

#### KL divergence

| Score | MNIST | SVHN | CIFAR-10 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.785 / 0.826 | 0.716 / 0.797 | 0.779 / 0.783 | 0.760 / 0.802 |
| Absolute max probability difference | 0.215 / 0.997 | 0.284 / 0.998 | 0.221 / 0.997 | 0.240 / 0.997 |
| KL (teacher / student) | 0.202 / 1.000 | 0.212 / 1.000 | 0.298 / 0.987 | 0.238 / 0.996 |
| Centered-logit L2 distance | 0.040 / 1.000 | 0.226 / 0.999 | 0.317 / 0.985 | 0.194 / 0.995 |
| Energy gap | 0.878 / 0.679 | 0.784 / 0.766 | 0.801 / 0.774 | 0.821 / 0.740 |
| Absolute energy gap | 0.122 / 1.000 | 0.216 / 0.999 | 0.199 / 0.997 | 0.179 / 0.999 |
| Student MSP | 0.614 / 0.932 | 0.649 / 0.933 | 0.498 / 0.926 | 0.587 / 0.930 |
| Student energy | 0.793 / 0.507 | 0.426 / 0.988 | 0.475 / 0.967 | 0.565 / 0.821 |

</details>

<details>
<summary>Spatial, Flatten</summary>

#### Centered-logit MSE

| Score | MNIST | SVHN | CIFAR-10 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.781 / 0.832 | 0.747 / 0.796 | 0.786 / 0.781 | 0.771 / 0.803 |
| Absolute max probability difference | 0.219 / 0.998 | 0.253 / 1.000 | 0.214 / 0.998 | 0.229 / 0.999 |
| KL (teacher / student) | 0.186 / 1.000 | 0.161 / 1.000 | 0.292 / 0.985 | 0.213 / 0.995 |
| Centered-logit L2 distance | 0.043 / 1.000 | 0.189 / 1.000 | 0.329 / 0.989 | 0.187 / 0.996 |
| Energy gap | 0.873 / 0.704 | 0.779 / 0.775 | 0.802 / 0.771 | 0.818 / 0.750 |
| Absolute energy gap | 0.127 / 1.000 | 0.221 / 0.999 | 0.198 / 0.997 | 0.182 / 0.999 |
| Student MSP | 0.823 / 0.510 | 0.535 / 0.948 | 0.468 / 0.957 | 0.609 / 0.805 |
| Student energy | 0.882 / 0.301 | 0.720 / 0.808 | 0.471 / 0.959 | 0.691 / 0.690 |

#### KL divergence

| Score | MNIST | SVHN | CIFAR-10 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.706 / 0.895 | 0.872 / 0.695 | 0.757 / 0.820 | 0.778 / 0.804 |
| Absolute max probability difference | 0.294 / 0.997 | 0.128 / 0.999 | 0.243 / 0.994 | 0.222 / 0.997 |
| KL (teacher / student) | 0.247 / 1.000 | 0.084 / 1.000 | 0.416 / 0.986 | 0.249 / 0.995 |
| Centered-logit L2 distance | 0.048 / 1.000 | 0.103 / 1.000 | 0.426 / 0.977 | 0.192 / 0.992 |
| Energy gap | 0.863 / 0.778 | 0.820 / 0.762 | 0.806 / 0.770 | 0.830 / 0.770 |
| Absolute energy gap | 0.137 / 1.000 | 0.180 / 1.000 | 0.194 / 0.998 | 0.170 / 0.999 |
| Student MSP | 0.685 / 0.847 | 0.277 / 0.966 | 0.490 / 0.935 | 0.484 / 0.916 |
| Student energy | 0.784 / 0.577 | 0.387 / 0.979 | 0.481 / 0.960 | 0.550 / 0.839 |

</details>

<details>
<summary>Channel, GAP</summary>

#### Centered-logit MSE

| Score | MNIST | SVHN | CIFAR-10 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.798 / 0.827 | 0.729 / 0.793 | 0.778 / 0.784 | 0.768 / 0.801 |
| Absolute max probability difference | 0.202 / 0.999 | 0.271 / 0.998 | 0.222 / 0.998 | 0.232 / 0.998 |
| KL (teacher / student) | 0.226 / 1.000 | 0.204 / 1.000 | 0.357 / 0.945 | 0.262 / 0.982 |
| Centered-logit L2 distance | 0.049 / 1.000 | 0.177 / 1.000 | 0.388 / 0.982 | 0.205 / 0.994 |
| Energy gap | 0.878 / 0.676 | 0.784 / 0.763 | 0.800 / 0.777 | 0.821 / 0.739 |
| Absolute energy gap | 0.122 / 1.000 | 0.216 / 0.999 | 0.200 / 0.996 | 0.179 / 0.999 |
| Student MSP | 0.623 / 0.981 | 0.480 / 0.984 | 0.505 / 0.965 | 0.536 / 0.977 |
| Student energy | 0.560 / 1.000 | 0.432 / 0.997 | 0.531 / 0.984 | 0.508 / 0.993 |

#### KL divergence

| Score | MNIST | SVHN | CIFAR-10 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.674 / 0.913 | 0.678 / 0.868 | 0.681 / 0.887 | 0.678 / 0.890 |
| Absolute max probability difference | 0.326 / 1.000 | 0.323 / 0.995 | 0.319 / 0.989 | 0.322 / 0.995 |
| KL (teacher / student) | 0.443 / 1.000 | 0.320 / 1.000 | 0.575 / 0.981 | 0.446 / 0.994 |
| Centered-logit L2 distance | 0.218 / 1.000 | 0.195 / 0.999 | 0.581 / 0.962 | 0.331 / 0.987 |
| Energy gap | 0.863 / 0.701 | 0.762 / 0.733 | 0.775 / 0.803 | 0.800 / 0.746 |
| Absolute energy gap | 0.138 / 1.000 | 0.240 / 1.000 | 0.224 / 0.995 | 0.201 / 0.998 |
| Student MSP | 0.542 / 0.999 | 0.478 / 0.963 | 0.568 / 0.932 | 0.529 / 0.965 |
| Student energy | 0.588 / 0.998 | 0.591 / 0.883 | 0.630 / 0.952 | 0.603 / 0.944 |

</details>

<details>
<summary>Channel, Flatten</summary>

#### Centered-logit MSE

| Score | MNIST | SVHN | CIFAR-10 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.687 / 0.940 | 0.609 / 0.903 | 0.672 / 0.920 | 0.656 / 0.921 |
| Absolute max probability difference | 0.313 / 1.000 | 0.391 / 0.945 | 0.328 / 0.985 | 0.344 / 0.976 |
| KL (teacher / student) | 0.390 / 1.000 | 0.342 / 0.998 | 0.539 / 0.979 | 0.424 / 0.992 |
| Centered-logit L2 distance | 0.169 / 1.000 | 0.251 / 0.999 | 0.562 / 0.970 | 0.327 / 0.990 |
| Energy gap | 0.869 / 0.699 | 0.770 / 0.728 | 0.778 / 0.805 | 0.805 / 0.744 |
| Absolute energy gap | 0.131 / 1.000 | 0.230 / 0.999 | 0.222 / 0.995 | 0.195 / 0.998 |
| Student MSP | 0.607 / 1.000 | 0.658 / 0.926 | 0.651 / 0.936 | 0.639 / 0.954 |
| Student energy | 0.534 / 1.000 | 0.550 / 0.892 | 0.679 / 0.894 | 0.588 / 0.929 |

#### KL divergence

| Score | MNIST | SVHN | CIFAR-10 | Macro |
|---|---:|---:|---:|---:|
| Max probability difference | 0.577 / 0.862 | 0.599 / 0.774 | 0.587 / 0.833 | 0.588 / 0.823 |
| Absolute max probability difference | 0.520 / 0.985 | 0.583 / 0.914 | 0.541 / 0.945 | 0.548 / 0.948 |
| KL (teacher / student) | 0.549 / 0.948 | 0.807 / 0.629 | 0.735 / 0.910 | 0.697 / 0.829 |
| Centered-logit L2 distance | 0.308 / 1.000 | 0.378 / 0.984 | 0.389 / 0.996 | 0.359 / 0.993 |
| Energy gap | 0.719 / 1.000 | 0.573 / 0.969 | 0.563 / 0.998 | 0.619 / 0.989 |
| Absolute energy gap | 0.713 / 1.000 | 0.618 / 0.969 | 0.547 / 0.998 | 0.626 / 0.989 |
| Student MSP | 0.614 / 0.963 | 0.551 / 0.939 | 0.619 / 0.916 | 0.595 / 0.939 |
| Student energy | 0.540 / 1.000 | 0.612 / 0.840 | 0.669 / 0.952 | 0.607 / 0.931 |

</details>

## Artifacts

```text
reports/outputs/json/clipping_layer1234_metrics.json
reports/outputs/json/clipping_layer34_metrics.json
reports/outputs/json/clipping_layer34_teacher_metrics.json
```

SLURM train-and-clean-inference jobs: `407342`, `407343`, `407344`, and `407345`.

A direct stochastic-inference comparison requires a separate `APPLY_PERTURBATION=1` export for these 24 runs.
