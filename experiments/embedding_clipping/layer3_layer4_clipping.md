# Sequential Layer3/Layer4 Clipping Students

Status: completed for CIFAR-10 and CIFAR-100 ID with ResNet-18 and ResNet-50 teachers.

## Setup

- Teacher: ResNet-18 or ResNet-50.
- Clipping sequence: clip `layer3`, continue the teacher forward pass, then clip `layer4`.
- Clipping percentile range at both layers: `[0.5, 1.0]`.
- Layer3 and layer4 percentiles are sampled independently from the same range.
- Clipping modes: constant, spatial-dependent, and channel-dependent; the same mode is used at both layers.
- Student input: pooled clipped `layer4` plus `u_layer3` and `u_layer4`.
- Embedding pooling: global average pooling (GAP) or flattening.
- Student: linear classifier.
- Teacher target: clean logits.
- Objectives: centered-logit MSE and KL divergence.
- Training: 30 epochs, seed 42, batch size 256.
- Checkpoint selection: best validation-loss checkpoint.
- Stochastic evaluation: mean over 50 independent clipping draws.
- OOD datasets: MNIST, SVHN, and the opposite CIFAR test set.

Student input dimensions:

| Teacher   | Clipping | Pooling | Student input |
| --------- | -------: | ------: | ------------: |
| ResNet-18 | Constant |     GAP |           514 |
| ResNet-18 | Constant | Flatten |          8194 |
| ResNet-18 |  Spatial |     GAP |           592 |
| ResNet-18 |  Spatial | Flatten |          8272 |
| ResNet-18 |  Channel |     GAP |          1280 |
| ResNet-18 |  Channel | Flatten |          8960 |
| ResNet-50 | Constant |     GAP |          2050 |
| ResNet-50 | Constant | Flatten |         32770 |
| ResNet-50 |  Spatial |     GAP |          2128 |
| ResNet-50 |  Spatial | Flatten |         32848 |
| ResNet-50 |  Channel |     GAP |          5120 |
| ResNet-50 |  Channel | Flatten |         35840 |

Run-directory pattern:

```text
runs/students/perturbation/embedding/clipping/<cifar_10|cifar_100>/<resnet18|resnet50>/linear_layer3_layer4_clip_<constant|spatial|channel>_<avg|flatten>
```

## Training Results

Loss values are objective-dependent and should not be compared directly between centered-logit MSE and KL divergence.

### CIFAR-10 ID, ResNet-18

| Clipping | Pooling |          Objective | Test accuracy | Test distillation loss |
| -------- | ------: | -----------------: | ------------: | ---------------------: |
| Constant |     GAP | Centered-logit MSE |        0.9283 |                1.71131 |
| Constant |     GAP |      KL divergence |        0.9337 |              0.0983843 |
| Constant | Flatten | Centered-logit MSE |        0.9429 |                6.22433 |
| Constant | Flatten |      KL divergence |        0.9347 |              0.0909539 |
| Spatial  |     GAP | Centered-logit MSE |        0.9333 |                4.81806 |
| Spatial  |     GAP |      KL divergence |        0.9295 |               0.116419 |
| Spatial  | Flatten | Centered-logit MSE |        0.9219 |                30.8525 |
| Spatial  | Flatten |      KL divergence |        0.9277 |               0.214486 |
| Channel  |     GAP | Centered-logit MSE |        0.9443 |                 0.7132 |
| Channel  |     GAP |      KL divergence |        0.9239 |               0.158253 |
| Channel  | Flatten | Centered-logit MSE |        0.9428 |                 1.2444 |
| Channel  | Flatten |      KL divergence |        0.9349 |               0.121275 |

### CIFAR-10 ID, ResNet-50

| Clipping | Pooling |          Objective | Test accuracy | Test distillation loss |
| -------- | ------: | -----------------: | ------------: | ---------------------: |
| Constant |     GAP | Centered-logit MSE |        0.9317 |                4.03145 |
| Constant |     GAP |      KL divergence |        0.9322 |              0.0777212 |
| Constant | Flatten | Centered-logit MSE |        0.9035 |                10.0774 |
| Constant | Flatten |      KL divergence |        0.9334 |               0.088515 |
| Spatial  |     GAP | Centered-logit MSE |        0.8791 |                7.97262 |
| Spatial  |     GAP |      KL divergence |        0.9238 |                0.15052 |
| Spatial  | Flatten | Centered-logit MSE |        0.7863 |                235.937 |
| Spatial  | Flatten |      KL divergence |        0.9273 |               0.166563 |
| Channel  |     GAP | Centered-logit MSE |        0.9427 |                3.20529 |
| Channel  |     GAP |      KL divergence |        0.9377 |               0.043854 |
| Channel  | Flatten | Centered-logit MSE |        0.9423 |                3.78656 |
| Channel  | Flatten |      KL divergence |        0.9321 |               0.121014 |

### CIFAR-100 ID, ResNet-18

| Clipping | Pooling |          Objective | Test accuracy | Test distillation loss |
| -------- | ------: | -----------------: | ------------: | ---------------------: |
| Constant |     GAP | Centered-logit MSE |        0.7737 |               0.814638 |
| Constant |     GAP |      KL divergence |        0.7833 |               0.362047 |
| Constant | Flatten | Centered-logit MSE |        0.7865 |               0.302145 |
| Constant | Flatten |      KL divergence |        0.7876 |              0.0460758 |
| Spatial  |     GAP | Centered-logit MSE |        0.3907 |                1.19882 |
| Spatial  |     GAP |      KL divergence |        0.3409 |                1.85644 |
| Spatial  | Flatten | Centered-logit MSE |        0.5547 |                 1.0726 |
| Spatial  | Flatten |      KL divergence |        0.7649 |                1.99303 |
| Channel  |     GAP | Centered-logit MSE |        0.7851 |               0.178264 |
| Channel  |     GAP |      KL divergence |        0.7698 |               0.183765 |
| Channel  | Flatten | Centered-logit MSE |        0.7837 |               0.212395 |
| Channel  | Flatten |      KL divergence |        0.7694 |               0.251939 |

### CIFAR-100 ID, ResNet-50

| Clipping | Pooling |          Objective | Test accuracy | Test distillation loss |
| -------- | ------: | -----------------: | ------------: | ---------------------: |
| Constant |     GAP | Centered-logit MSE |        0.8006 |                1.07764 |
| Constant |     GAP |      KL divergence |        0.8034 |               0.309868 |
| Constant | Flatten | Centered-logit MSE |        0.8029 |               0.682584 |
| Constant | Flatten |      KL divergence |        0.8032 |               0.086317 |
| Spatial  |     GAP | Centered-logit MSE |        0.7747 |                1.42636 |
| Spatial  |     GAP |      KL divergence |        0.7798 |               0.269693 |
| Spatial  | Flatten | Centered-logit MSE |        0.7775 |                1.39799 |
| Spatial  | Flatten |      KL divergence |        0.7884 |               0.345028 |
| Channel  |     GAP | Centered-logit MSE |        0.7991 |               0.708599 |
| Channel  |     GAP |      KL divergence |        0.7916 |               0.259856 |
| Channel  | Flatten | Centered-logit MSE |        0.7965 |               0.749025 |
| Channel  | Flatten |      KL divergence |        0.7938 |               0.240338 |

## OOD Scores

All scores follow the project convention that higher values are more ID-like. Every cell reports `ROC-AUC / FPR@95`; higher ROC-AUC and lower FPR@95 are better.

Teacher MSP and energy use separate standard raw-image teacher inference. The student-related scores are:

- negative KL divergence from teacher to student;
- teacher minus student maximum probability;
- negative absolute maximum-probability difference;
- negative centered-logit L2 distance;
- teacher minus student energy;
- negative absolute energy gap;
- student MSP;
- student energy.

Student scores are reported for clean inference and for the mean over 50 independently clipped inference draws.

## Teacher Baselines

### CIFAR-10 ID, ResNet-18

| Score          |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| -------------- | ------------: | ------------: | ------------: | ------------: |
| Teacher MSP    | 0.918 / 0.556 | 0.904 / 0.598 | 0.879 / 0.624 | 0.900 / 0.593 |
| Teacher energy | 0.958 / 0.260 | 0.918 / 0.412 | 0.879 / 0.506 | 0.918 / 0.392 |

### CIFAR-10 ID, ResNet-50

| Score          |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| -------------- | ------------: | ------------: | ------------: | ------------: |
| Teacher MSP    | 0.897 / 0.622 | 0.891 / 0.680 | 0.873 / 0.628 | 0.887 / 0.643 |
| Teacher energy | 0.932 / 0.408 | 0.911 / 0.517 | 0.868 / 0.517 | 0.904 / 0.481 |

### CIFAR-100 ID, ResNet-18

| Score          |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| -------------- | ------------: | ------------: | ------------: | ------------: |
| Teacher MSP    | 0.701 / 0.956 | 0.806 / 0.812 | 0.792 / 0.794 | 0.767 / 0.854 |
| Teacher energy | 0.695 / 0.974 | 0.829 / 0.771 | 0.795 / 0.800 | 0.773 / 0.848 |

### CIFAR-100 ID, ResNet-50

| Score          |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| -------------- | ------------: | ------------: | ------------: | ------------: |
| Teacher MSP    | 0.811 / 0.823 | 0.758 / 0.793 | 0.792 / 0.781 | 0.787 / 0.799 |
| Teacher energy | 0.878 / 0.670 | 0.783 / 0.766 | 0.801 / 0.774 | 0.821 / 0.737 |

## Macro Comparison

These tables average each score across the three OOD datasets. Detailed per-dataset results follow in the next section.

### CIFAR-10 ID, ResNet-18

#### Centered-logit MSE, Clean

| Clipping | Pooling | Max probability difference | Absolute max probability difference | KL (teacher / student) | Centered-logit L2 distance |    Energy gap | Absolute energy gap |   Student MSP | Student energy |
| -------- | ------: | -------------------------: | ----------------------------------: | ---------------------: | -------------------------: | ------------: | ------------------: | ------------: | -------------: |
| Constant |     GAP |              0.519 / 0.696 |                       0.835 / 0.863 |          0.849 / 0.789 |              0.445 / 0.992 | 0.285 / 0.991 |       0.294 / 0.992 | 0.862 / 0.722 |  0.892 / 0.589 |
| Constant | Flatten |              0.503 / 0.726 |                       0.888 / 0.662 |          0.876 / 0.679 |              0.229 / 0.998 | 0.212 / 0.999 |       0.202 / 0.999 | 0.902 / 0.560 |  0.913 / 0.433 |
| Spatial  |     GAP |              0.554 / 0.700 |                       0.847 / 0.757 |          0.870 / 0.698 |              0.203 / 0.993 | 0.211 / 0.996 |       0.162 / 0.996 | 0.872 / 0.666 |  0.909 / 0.452 |
| Spatial  | Flatten |              0.619 / 0.709 |                       0.865 / 0.758 |          0.846 / 0.795 |              0.176 / 0.999 | 0.199 / 1.000 |       0.155 / 1.000 | 0.873 / 0.660 |  0.882 / 0.628 |
| Channel  |     GAP |              0.634 / 0.706 |                       0.881 / 0.698 |          0.879 / 0.678 |              0.313 / 0.975 | 0.224 / 0.973 |       0.223 / 0.973 | 0.895 / 0.605 |  0.914 / 0.416 |
| Channel  | Flatten |              0.686 / 0.679 |                       0.887 / 0.681 |          0.886 / 0.658 |              0.306 / 0.987 | 0.235 / 0.990 |       0.237 / 0.990 | 0.889 / 0.614 |  0.907 / 0.445 |

#### Centered-logit MSE, Clipped, 50-draw mean

| Clipping | Pooling | Max probability difference | Absolute max probability difference | KL (teacher / student) | Centered-logit L2 distance |    Energy gap | Absolute energy gap |   Student MSP | Student energy |
| -------- | ------: | -------------------------: | ----------------------------------: | ---------------------: | -------------------------: | ------------: | ------------------: | ------------: | -------------: |
| Constant |     GAP |              0.619 / 0.684 |                       0.382 / 0.937 |          0.256 / 0.984 |              0.097 / 0.999 | 0.915 / 0.407 |       0.084 / 0.999 | 0.786 / 0.675 |  0.839 / 0.581 |
| Constant | Flatten |              0.604 / 0.694 |                       0.420 / 0.909 |          0.379 / 0.973 |              0.123 / 0.999 | 0.901 / 0.429 |       0.089 / 0.999 | 0.763 / 0.667 |  0.860 / 0.508 |
| Spatial  |     GAP |              0.305 / 0.944 |                       0.710 / 0.778 |          0.777 / 0.810 |              0.199 / 0.995 | 0.890 / 0.449 |       0.110 / 0.998 | 0.885 / 0.507 |  0.904 / 0.427 |
| Spatial  | Flatten |              0.402 / 0.707 |                       0.869 / 0.677 |          0.838 / 0.766 |              0.689 / 0.878 | 0.583 / 0.899 |       0.359 / 0.970 | 0.882 / 0.528 |  0.919 / 0.409 |
| Channel  |     GAP |              0.326 / 0.789 |                       0.857 / 0.604 |          0.815 / 0.865 |              0.626 / 0.882 | 0.487 / 0.953 |       0.369 / 0.931 | 0.884 / 0.413 |  0.906 / 0.378 |
| Channel  | Flatten |              0.323 / 0.792 |                       0.862 / 0.597 |          0.823 / 0.840 |              0.654 / 0.870 | 0.464 / 0.967 |       0.371 / 0.928 | 0.887 / 0.405 |  0.911 / 0.368 |

#### KL divergence, Clean

| Clipping | Pooling | Max probability difference | Absolute max probability difference | KL (teacher / student) | Centered-logit L2 distance |    Energy gap | Absolute energy gap |   Student MSP | Student energy |
| -------- | ------: | -------------------------: | ----------------------------------: | ---------------------: | -------------------------: | ------------: | ------------------: | ------------: | -------------: |
| Constant |     GAP |              0.683 / 0.689 |                       0.853 / 0.784 |          0.869 / 0.706 |              0.291 / 0.999 | 0.166 / 0.999 |       0.107 / 0.999 | 0.874 / 0.676 |  0.900 / 0.552 |
| Constant | Flatten |              0.699 / 0.685 |                       0.854 / 0.785 |          0.868 / 0.705 |              0.305 / 0.999 | 0.182 / 0.999 |       0.125 / 0.999 | 0.873 / 0.678 |  0.899 / 0.539 |
| Spatial  |     GAP |              0.493 / 0.718 |                       0.859 / 0.777 |          0.874 / 0.716 |              0.249 / 0.998 | 0.232 / 0.998 |       0.208 / 0.995 | 0.874 / 0.674 |  0.904 / 0.509 |
| Spatial  | Flatten |              0.758 / 0.664 |                       0.873 / 0.719 |          0.880 / 0.718 |              0.122 / 1.000 | 0.150 / 1.000 |       0.149 / 1.000 | 0.863 / 0.705 |  0.895 / 0.582 |
| Channel  |     GAP |              0.491 / 0.727 |                       0.836 / 0.765 |          0.861 / 0.743 |              0.359 / 0.975 | 0.432 / 0.999 |       0.314 / 0.994 | 0.855 / 0.659 |  0.887 / 0.492 |
| Channel  | Flatten |              0.610 / 0.684 |                       0.882 / 0.687 |          0.873 / 0.740 |              0.302 / 0.999 | 0.256 / 0.999 |       0.254 / 0.999 | 0.874 / 0.631 |  0.900 / 0.422 |

#### KL divergence, Clipped, 50-draw mean

| Clipping | Pooling | Max probability difference | Absolute max probability difference | KL (teacher / student) | Centered-logit L2 distance |    Energy gap | Absolute energy gap |   Student MSP | Student energy |
| -------- | ------: | -------------------------: | ----------------------------------: | ---------------------: | -------------------------: | ------------: | ------------------: | ------------: | -------------: |
| Constant |     GAP |              0.574 / 0.712 |                       0.428 / 0.903 |          0.378 / 0.919 |              0.133 / 0.998 | 0.909 / 0.417 |       0.087 / 0.999 | 0.801 / 0.631 |  0.855 / 0.545 |
| Constant | Flatten |              0.584 / 0.705 |                       0.418 / 0.907 |          0.325 / 0.973 |              0.121 / 0.998 | 0.911 / 0.415 |       0.086 / 0.999 | 0.794 / 0.628 |  0.849 / 0.537 |
| Spatial  |     GAP |              0.241 / 0.977 |                       0.764 / 0.713 |          0.824 / 0.755 |              0.227 / 0.993 | 0.889 / 0.438 |       0.111 / 0.998 | 0.906 / 0.433 |  0.923 / 0.364 |
| Spatial  | Flatten |              0.239 / 0.937 |                       0.821 / 0.703 |          0.843 / 0.745 |              0.391 / 0.984 | 0.788 / 0.631 |       0.207 / 0.992 | 0.896 / 0.478 |  0.924 / 0.371 |
| Channel  |     GAP |              0.275 / 0.846 |                       0.859 / 0.590 |          0.849 / 0.704 |              0.605 / 0.860 | 0.527 / 0.996 |       0.359 / 0.957 | 0.897 / 0.392 |  0.913 / 0.362 |
| Channel  | Flatten |              0.307 / 0.794 |                       0.877 / 0.584 |          0.852 / 0.742 |              0.335 / 0.999 | 0.364 / 0.998 |       0.269 / 0.998 | 0.901 / 0.396 |  0.918 / 0.355 |

### CIFAR-10 ID, ResNet-50

#### Centered-logit MSE, Clean

| Clipping | Pooling | Max probability difference | Absolute max probability difference | KL (teacher / student) | Centered-logit L2 distance |    Energy gap | Absolute energy gap |   Student MSP | Student energy |
| -------- | ------: | -------------------------: | ----------------------------------: | ---------------------: | -------------------------: | ------------: | ------------------: | ------------: | -------------: |
| Constant |     GAP |              0.450 / 0.731 |                       0.748 / 0.830 |          0.769 / 0.751 |              0.398 / 0.992 | 0.717 / 0.940 |       0.279 / 0.999 | 0.788 / 0.755 |  0.691 / 0.879 |
| Constant | Flatten |              0.518 / 0.718 |                       0.787 / 0.873 |          0.804 / 0.818 |              0.398 / 0.988 | 0.497 / 0.998 |       0.351 / 0.998 | 0.793 / 0.838 |  0.746 / 0.881 |
| Spatial  |     GAP |              0.534 / 0.827 |                       0.540 / 0.986 |          0.625 / 0.854 |              0.288 / 0.995 | 0.855 / 0.691 |       0.144 / 1.000 | 0.622 / 0.970 |  0.526 / 0.994 |
| Spatial  | Flatten |              0.718 / 0.677 |                       0.774 / 0.886 |          0.745 / 0.915 |              0.261 / 0.998 | 0.349 / 0.999 |       0.335 / 0.999 | 0.683 / 0.926 |  0.709 / 0.947 |
| Channel  |     GAP |              0.626 / 0.677 |                       0.887 / 0.602 |          0.892 / 0.579 |              0.516 / 0.999 | 0.277 / 1.000 |       0.277 / 1.000 | 0.877 / 0.634 |  0.888 / 0.517 |
| Channel  | Flatten |              0.653 / 0.668 |                       0.885 / 0.635 |          0.890 / 0.596 |              0.519 / 0.982 | 0.340 / 0.986 |       0.340 / 0.986 | 0.864 / 0.647 |  0.862 / 0.566 |

#### Centered-logit MSE, Clipped, 50-draw mean

| Clipping | Pooling | Max probability difference | Absolute max probability difference | KL (teacher / student) | Centered-logit L2 distance |    Energy gap | Absolute energy gap |   Student MSP | Student energy |
| -------- | ------: | -------------------------: | ----------------------------------: | ---------------------: | -------------------------: | ------------: | ------------------: | ------------: | -------------: |
| Constant |     GAP |              0.691 / 0.723 |                       0.309 / 0.993 |          0.298 / 0.978 |              0.133 / 0.997 | 0.905 / 0.486 |       0.095 / 0.999 | 0.564 / 0.959 |  0.570 / 0.965 |
| Constant | Flatten |              0.472 / 0.790 |                       0.541 / 0.842 |          0.507 / 0.841 |              0.209 / 0.991 | 0.876 / 0.526 |       0.105 / 0.999 | 0.787 / 0.678 |  0.832 / 0.625 |
| Spatial  |     GAP |              0.719 / 0.793 |                       0.281 / 0.998 |          0.374 / 0.967 |              0.135 / 0.996 | 0.905 / 0.488 |       0.095 / 0.999 | 0.453 / 0.995 |  0.492 / 0.994 |
| Spatial  | Flatten |              0.325 / 0.785 |                       0.875 / 0.664 |          0.842 / 0.795 |              0.693 / 0.915 | 0.517 / 0.939 |       0.348 / 0.968 | 0.889 / 0.526 |  0.911 / 0.476 |
| Channel  |     GAP |              0.367 / 0.745 |                       0.852 / 0.683 |          0.831 / 0.836 |              0.744 / 0.881 | 0.463 / 0.963 |       0.452 / 0.927 | 0.861 / 0.565 |  0.875 / 0.566 |
| Channel  | Flatten |              0.355 / 0.753 |                       0.854 / 0.711 |          0.827 / 0.854 |              0.695 / 0.913 | 0.440 / 0.971 |       0.446 / 0.939 | 0.868 / 0.567 |  0.886 / 0.554 |

#### KL divergence, Clean

| Clipping | Pooling | Max probability difference | Absolute max probability difference | KL (teacher / student) | Centered-logit L2 distance |    Energy gap | Absolute energy gap |   Student MSP | Student energy |
| -------- | ------: | -------------------------: | ----------------------------------: | ---------------------: | -------------------------: | ------------: | ------------------: | ------------: | -------------: |
| Constant |     GAP |              0.532 / 0.691 |                       0.844 / 0.716 |          0.863 / 0.673 |              0.451 / 0.983 | 0.529 / 0.995 |       0.386 / 0.996 | 0.821 / 0.706 |  0.796 / 0.727 |
| Constant | Flatten |              0.618 / 0.677 |                       0.859 / 0.688 |          0.881 / 0.650 |              0.487 / 0.992 | 0.445 / 0.998 |       0.433 / 0.998 | 0.815 / 0.713 |  0.788 / 0.715 |
| Spatial  |     GAP |              0.589 / 0.682 |                       0.844 / 0.752 |          0.863 / 0.733 |              0.464 / 0.986 | 0.450 / 0.998 |       0.374 / 0.998 | 0.809 / 0.757 |  0.755 / 0.858 |
| Spatial  | Flatten |              0.646 / 0.672 |                       0.867 / 0.724 |          0.876 / 0.724 |              0.440 / 0.991 | 0.369 / 0.999 |       0.362 / 0.999 | 0.820 / 0.728 |  0.785 / 0.769 |
| Channel  |     GAP |              0.441 / 0.722 |                       0.836 / 0.698 |          0.849 / 0.680 |              0.380 / 0.990 | 0.645 / 0.971 |       0.325 / 0.999 | 0.858 / 0.658 |  0.851 / 0.618 |
| Channel  | Flatten |              0.639 / 0.682 |                       0.865 / 0.722 |          0.872 / 0.705 |              0.362 / 0.999 | 0.410 / 0.999 |       0.391 / 0.999 | 0.825 / 0.701 |  0.807 / 0.688 |

#### KL divergence, Clipped, 50-draw mean

| Clipping | Pooling | Max probability difference | Absolute max probability difference | KL (teacher / student) | Centered-logit L2 distance |    Energy gap | Absolute energy gap |   Student MSP | Student energy |
| -------- | ------: | -------------------------: | ----------------------------------: | ---------------------: | -------------------------: | ------------: | ------------------: | ------------: | -------------: |
| Constant |     GAP |              0.514 / 0.737 |                       0.486 / 0.857 |          0.513 / 0.812 |              0.172 / 0.996 | 0.901 / 0.487 |       0.098 / 0.999 | 0.826 / 0.626 |  0.760 / 0.698 |
| Constant | Flatten |              0.508 / 0.756 |                       0.492 / 0.871 |          0.484 / 0.862 |              0.171 / 0.996 | 0.900 / 0.493 |       0.099 / 0.999 | 0.810 / 0.669 |  0.763 / 0.720 |
| Spatial  |     GAP |              0.304 / 0.935 |                       0.697 / 0.701 |          0.831 / 0.648 |              0.249 / 0.992 | 0.893 / 0.500 |       0.107 / 0.998 | 0.902 / 0.465 |  0.893 / 0.430 |
| Spatial  | Flatten |              0.270 / 0.959 |                       0.733 / 0.708 |          0.838 / 0.704 |              0.271 / 0.990 | 0.884 / 0.515 |       0.116 / 0.998 | 0.896 / 0.505 |  0.884 / 0.464 |
| Channel  |     GAP |              0.290 / 0.848 |                       0.831 / 0.704 |          0.831 / 0.786 |              0.507 / 0.924 | 0.768 / 0.661 |       0.232 / 0.983 | 0.864 / 0.566 |  0.842 / 0.601 |
| Channel  | Flatten |              0.338 / 0.766 |                       0.856 / 0.695 |          0.841 / 0.776 |              0.575 / 0.905 | 0.494 / 0.999 |       0.320 / 0.970 | 0.871 / 0.564 |  0.870 / 0.594 |

### CIFAR-100 ID, ResNet-18

#### Centered-logit MSE, Clean

| Clipping | Pooling | Max probability difference | Absolute max probability difference | KL (teacher / student) | Centered-logit L2 distance |    Energy gap | Absolute energy gap |   Student MSP | Student energy |
| -------- | ------: | -------------------------: | ----------------------------------: | ---------------------: | -------------------------: | ------------: | ------------------: | ------------: | -------------: |
| Constant |     GAP |              0.717 / 0.851 |                       0.283 / 0.989 |          0.319 / 0.987 |              0.263 / 0.999 | 0.773 / 0.851 |       0.227 / 1.000 | 0.717 / 0.871 |  0.677 / 0.883 |
| Constant | Flatten |              0.478 / 0.885 |                       0.547 / 0.962 |          0.598 / 0.958 |              0.250 / 0.999 | 0.737 / 0.954 |       0.232 / 0.999 | 0.742 / 0.858 |  0.745 / 0.839 |
| Spatial  |     GAP |              0.758 / 0.844 |                       0.242 / 0.995 |          0.248 / 0.985 |              0.246 / 0.998 | 0.778 / 0.842 |       0.222 / 1.000 | 0.463 / 0.965 |  0.480 / 0.948 |
| Spatial  | Flatten |              0.731 / 0.841 |                       0.275 / 0.989 |          0.325 / 0.989 |              0.286 / 0.999 | 0.782 / 0.866 |       0.220 / 0.999 | 0.494 / 0.942 |  0.501 / 0.944 |
| Channel  |     GAP |              0.327 / 0.934 |                       0.727 / 0.827 |          0.783 / 0.822 |              0.375 / 0.998 | 0.526 / 0.962 |       0.444 / 0.997 | 0.790 / 0.810 |  0.799 / 0.774 |
| Channel  | Flatten |              0.342 / 0.944 |                       0.671 / 0.936 |          0.724 / 0.945 |              0.364 / 0.999 | 0.683 / 0.929 |       0.316 / 0.999 | 0.776 / 0.827 |  0.780 / 0.795 |

#### Centered-logit MSE, Clipped, 50-draw mean

| Clipping | Pooling | Max probability difference | Absolute max probability difference | KL (teacher / student) | Centered-logit L2 distance |    Energy gap | Absolute energy gap |   Student MSP | Student energy |
| -------- | ------: | -------------------------: | ----------------------------------: | ---------------------: | -------------------------: | ------------: | ------------------: | ------------: | -------------: |
| Constant |     GAP |              0.763 / 0.854 |                       0.237 / 0.998 |          0.235 / 0.996 |              0.264 / 0.999 | 0.773 / 0.849 |       0.227 / 1.000 | 0.640 / 0.913 |  0.588 / 0.931 |
| Constant | Flatten |              0.748 / 0.854 |                       0.252 / 0.992 |          0.249 / 0.993 |              0.268 / 0.999 | 0.771 / 0.851 |       0.229 / 1.000 | 0.677 / 0.876 |  0.654 / 0.890 |
| Spatial  |     GAP |              0.765 / 0.854 |                       0.235 / 0.999 |          0.237 / 0.999 |              0.263 / 0.999 | 0.773 / 0.848 |       0.227 / 1.000 | 0.645 / 0.855 |  0.572 / 0.954 |
| Spatial  | Flatten |              0.761 / 0.854 |                       0.239 / 0.998 |          0.251 / 0.994 |              0.269 / 0.999 | 0.772 / 0.848 |       0.228 / 1.000 | 0.613 / 0.864 |  0.611 / 0.925 |
| Channel  |     GAP |              0.603 / 0.846 |                       0.418 / 0.932 |          0.492 / 0.925 |              0.419 / 0.985 | 0.733 / 0.868 |       0.261 / 0.998 | 0.688 / 0.775 |  0.707 / 0.774 |
| Channel  | Flatten |              0.620 / 0.834 |                       0.399 / 0.940 |          0.476 / 0.932 |              0.406 / 0.987 | 0.742 / 0.859 |       0.257 / 0.998 | 0.672 / 0.784 |  0.688 / 0.804 |

#### KL divergence, Clean

| Clipping | Pooling | Max probability difference | Absolute max probability difference | KL (teacher / student) | Centered-logit L2 distance |    Energy gap | Absolute energy gap |   Student MSP | Student energy |
| -------- | ------: | -------------------------: | ----------------------------------: | ---------------------: | -------------------------: | ------------: | ------------------: | ------------: | -------------: |
| Constant |     GAP |              0.512 / 0.980 |                       0.488 / 0.987 |          0.513 / 0.983 |              0.263 / 0.999 | 0.782 / 0.849 |       0.218 / 0.999 | 0.730 / 0.870 |  0.721 / 0.867 |
| Constant | Flatten |              0.709 / 0.817 |                       0.719 / 0.860 |          0.760 / 0.833 |              0.349 / 0.998 | 0.532 / 0.967 |       0.470 / 0.970 | 0.734 / 0.870 |  0.742 / 0.857 |
| Spatial  |     GAP |              0.726 / 0.844 |                       0.293 / 0.989 |          0.342 / 0.982 |              0.372 / 0.997 | 0.795 / 0.859 |       0.218 / 0.999 | 0.454 / 0.964 |  0.419 / 0.964 |
| Spatial  | Flatten |              0.765 / 0.817 |                       0.770 / 0.817 |          0.801 / 0.740 |              0.433 / 0.996 | 0.387 / 0.993 |       0.387 / 0.993 | 0.643 / 0.917 |  0.650 / 0.914 |
| Channel  |     GAP |              0.422 / 0.919 |                       0.729 / 0.850 |          0.779 / 0.780 |              0.333 / 0.995 | 0.234 / 0.992 |       0.228 / 0.992 | 0.805 / 0.736 |  0.820 / 0.683 |
| Channel  | Flatten |              0.449 / 0.917 |                       0.738 / 0.856 |          0.775 / 0.796 |              0.347 / 0.994 | 0.189 / 0.998 |       0.176 / 0.998 | 0.815 / 0.731 |  0.829 / 0.687 |

#### KL divergence, Clipped, 50-draw mean

| Clipping | Pooling | Max probability difference | Absolute max probability difference | KL (teacher / student) | Centered-logit L2 distance |    Energy gap | Absolute energy gap |   Student MSP | Student energy |
| -------- | ------: | -------------------------: | ----------------------------------: | ---------------------: | -------------------------: | ------------: | ------------------: | ------------: | -------------: |
| Constant |     GAP |              0.754 / 0.854 |                       0.246 / 0.995 |          0.250 / 0.984 |              0.268 / 0.999 | 0.772 / 0.849 |       0.228 / 1.000 | 0.662 / 0.868 |  0.632 / 0.898 |
| Constant | Flatten |              0.743 / 0.853 |                       0.257 / 0.988 |          0.259 / 0.977 |              0.269 / 0.999 | 0.771 / 0.851 |       0.229 / 1.000 | 0.673 / 0.862 |  0.652 / 0.896 |
| Spatial  |     GAP |              0.760 / 0.853 |                       0.240 / 0.999 |          0.260 / 0.976 |              0.270 / 0.999 | 0.773 / 0.848 |       0.227 / 1.000 | 0.601 / 0.912 |  0.511 / 0.976 |
| Spatial  | Flatten |              0.699 / 0.861 |                       0.299 / 0.990 |          0.371 / 0.923 |              0.310 / 0.998 | 0.755 / 0.879 |       0.233 / 0.999 | 0.652 / 0.850 |  0.627 / 0.864 |
| Channel  |     GAP |              0.551 / 0.907 |                       0.474 / 0.903 |          0.598 / 0.828 |              0.424 / 0.941 | 0.658 / 0.966 |       0.263 / 0.996 | 0.750 / 0.732 |  0.777 / 0.738 |
| Channel  | Flatten |              0.545 / 0.915 |                       0.493 / 0.899 |          0.632 / 0.806 |              0.390 / 0.976 | 0.619 / 0.977 |       0.262 / 0.995 | 0.756 / 0.747 |  0.782 / 0.754 |

### CIFAR-100 ID, ResNet-50

#### Centered-logit MSE, Clean

| Clipping | Pooling | Max probability difference | Absolute max probability difference | KL (teacher / student) | Centered-logit L2 distance |    Energy gap | Absolute energy gap |   Student MSP | Student energy |
| -------- | ------: | -------------------------: | ----------------------------------: | ---------------------: | -------------------------: | ------------: | ------------------: | ------------: | -------------: |
| Constant |     GAP |              0.527 / 0.831 |                       0.473 / 0.820 |          0.622 / 0.785 |              0.286 / 0.993 | 0.809 / 0.777 |       0.191 / 0.998 | 0.843 / 0.661 |  0.810 / 0.595 |
| Constant | Flatten |              0.262 / 0.974 |                       0.741 / 0.796 |          0.799 / 0.780 |              0.305 / 0.990 | 0.749 / 0.872 |       0.251 / 0.999 | 0.813 / 0.740 |  0.835 / 0.630 |
| Spatial  |     GAP |              0.515 / 0.852 |                       0.485 / 0.879 |          0.669 / 0.817 |              0.385 / 0.975 | 0.795 / 0.816 |       0.205 / 0.998 | 0.840 / 0.631 |  0.778 / 0.658 |
| Spatial  | Flatten |              0.574 / 0.829 |                       0.426 / 0.882 |          0.590 / 0.826 |              0.346 / 0.986 | 0.806 / 0.787 |       0.194 / 0.998 | 0.830 / 0.664 |  0.764 / 0.674 |
| Channel  |     GAP |              0.338 / 0.905 |                       0.732 / 0.849 |          0.793 / 0.783 |              0.460 / 0.991 | 0.564 / 0.987 |       0.379 / 0.997 | 0.797 / 0.784 |  0.819 / 0.739 |
| Channel  | Flatten |              0.367 / 0.898 |                       0.704 / 0.868 |          0.771 / 0.790 |              0.467 / 0.989 | 0.596 / 0.988 |       0.351 / 0.998 | 0.782 / 0.794 |  0.798 / 0.767 |

#### Centered-logit MSE, Clipped, 50-draw mean

| Clipping | Pooling | Max probability difference | Absolute max probability difference | KL (teacher / student) | Centered-logit L2 distance |    Energy gap | Absolute energy gap |   Student MSP | Student energy |
| -------- | ------: | -------------------------: | ----------------------------------: | ---------------------: | -------------------------: | ------------: | ------------------: | ------------: | -------------: |
| Constant |     GAP |              0.765 / 0.800 |                       0.235 / 0.992 |          0.256 / 0.993 |              0.213 / 0.997 | 0.820 / 0.740 |       0.180 / 0.999 | 0.713 / 0.873 |  0.624 / 0.917 |
| Constant | Flatten |              0.735 / 0.803 |                       0.265 / 0.987 |          0.274 / 0.986 |              0.216 / 0.997 | 0.819 / 0.742 |       0.181 / 0.999 | 0.738 / 0.810 |  0.665 / 0.893 |
| Spatial  |     GAP |              0.763 / 0.801 |                       0.237 / 0.985 |          0.276 / 0.994 |              0.222 / 0.997 | 0.820 / 0.739 |       0.180 / 0.999 | 0.800 / 0.600 |  0.771 / 0.556 |
| Spatial  | Flatten |              0.763 / 0.800 |                       0.237 / 0.978 |          0.290 / 0.984 |              0.226 / 0.997 | 0.820 / 0.740 |       0.180 / 0.999 | 0.804 / 0.520 |  0.762 / 0.574 |
| Channel  |     GAP |              0.617 / 0.938 |                       0.385 / 0.989 |          0.459 / 0.971 |              0.327 / 0.991 | 0.806 / 0.747 |       0.193 / 0.998 | 0.592 / 0.958 |  0.592 / 0.958 |
| Channel  | Flatten |              0.618 / 0.938 |                       0.385 / 0.990 |          0.462 / 0.965 |              0.333 / 0.991 | 0.806 / 0.746 |       0.193 / 0.998 | 0.588 / 0.960 |  0.583 / 0.962 |

#### KL divergence, Clean

| Clipping | Pooling | Max probability difference | Absolute max probability difference | KL (teacher / student) | Centered-logit L2 distance |    Energy gap | Absolute energy gap |   Student MSP | Student energy |
| -------- | ------: | -------------------------: | ----------------------------------: | ---------------------: | -------------------------: | ------------: | ------------------: | ------------: | -------------: |
| Constant |     GAP |              0.281 / 0.989 |                       0.719 / 0.786 |          0.789 / 0.747 |              0.383 / 0.968 | 0.790 / 0.799 |       0.210 / 1.000 | 0.820 / 0.709 |  0.842 / 0.580 |
| Constant | Flatten |              0.578 / 0.863 |                       0.770 / 0.867 |          0.798 / 0.828 |              0.368 / 0.987 | 0.327 / 0.965 |       0.319 / 0.965 | 0.791 / 0.814 |  0.825 / 0.693 |
| Spatial  |     GAP |              0.327 / 0.914 |                       0.719 / 0.923 |          0.772 / 0.842 |              0.449 / 0.974 | 0.625 / 0.959 |       0.349 / 0.999 | 0.796 / 0.769 |  0.816 / 0.678 |
| Spatial  | Flatten |              0.629 / 0.865 |                       0.751 / 0.875 |          0.784 / 0.832 |              0.344 / 0.986 | 0.253 / 0.995 |       0.253 / 0.995 | 0.783 / 0.824 |  0.821 / 0.693 |
| Channel  |     GAP |              0.518 / 0.869 |                       0.788 / 0.828 |          0.835 / 0.708 |              0.444 / 0.991 | 0.269 / 0.986 |       0.269 / 0.986 | 0.811 / 0.770 |  0.831 / 0.710 |
| Channel  | Flatten |              0.445 / 0.865 |                       0.801 / 0.807 |          0.850 / 0.668 |              0.502 / 0.985 | 0.302 / 0.984 |       0.309 / 0.984 | 0.822 / 0.763 |  0.834 / 0.709 |

#### KL divergence, Clipped, 50-draw mean

| Clipping | Pooling | Max probability difference | Absolute max probability difference | KL (teacher / student) | Centered-logit L2 distance |    Energy gap | Absolute energy gap |   Student MSP | Student energy |
| -------- | ------: | -------------------------: | ----------------------------------: | ---------------------: | -------------------------: | ------------: | ------------------: | ------------: | -------------: |
| Constant |     GAP |              0.744 / 0.801 |                       0.256 / 0.991 |          0.255 / 0.981 |              0.211 / 0.997 | 0.820 / 0.741 |       0.180 / 0.999 | 0.739 / 0.786 |  0.656 / 0.904 |
| Constant | Flatten |              0.709 / 0.803 |                       0.290 / 0.973 |          0.273 / 0.983 |              0.213 / 0.997 | 0.817 / 0.753 |       0.182 / 0.999 | 0.741 / 0.771 |  0.694 / 0.873 |
| Spatial  |     GAP |              0.736 / 0.801 |                       0.264 / 0.990 |          0.319 / 0.977 |              0.239 / 0.996 | 0.819 / 0.742 |       0.181 / 0.999 | 0.771 / 0.714 |  0.785 / 0.574 |
| Spatial  | Flatten |              0.631 / 0.817 |                       0.369 / 0.937 |          0.481 / 0.942 |              0.278 / 0.996 | 0.810 / 0.773 |       0.190 / 0.998 | 0.821 / 0.624 |  0.816 / 0.521 |
| Channel  |     GAP |              0.545 / 0.920 |                       0.468 / 0.984 |          0.608 / 0.938 |              0.476 / 0.966 | 0.748 / 0.892 |       0.217 / 0.997 | 0.646 / 0.947 |  0.614 / 0.955 |
| Channel  | Flatten |              0.534 / 0.935 |                       0.473 / 0.984 |          0.640 / 0.881 |              0.449 / 0.957 | 0.747 / 0.894 |       0.224 / 0.998 | 0.662 / 0.938 |  0.636 / 0.946 |

## Aggregate Observations

| ID dataset |   Teacher |                                            Best macro ROC-AUC configuration |         Macro |                                        Best macro FPR@95 configuration |         Macro |
| ---------- | --------- | -------------------------------------------------------------------------- | ------------: | --------------------------------------------------------------------- | ------------: |
| CIFAR-10   | ResNet-18 |      Student energy; Spatial, Flatten, KL divergence, Clipped, 50-draw mean | 0.924 / 0.371 | Student energy; Channel, Flatten, KL divergence, Clipped, 50-draw mean | 0.918 / 0.355 |
| CIFAR-10   | ResNet-50 | Student energy; Spatial, Flatten, Centered-logit MSE, Clipped, 50-draw mean | 0.911 / 0.476 |     Student energy; Spatial, GAP, KL divergence, Clipped, 50-draw mean | 0.893 / 0.430 |
| CIFAR-100  | ResNet-18 |                      Student energy; Channel, Flatten, KL divergence, Clean | 0.829 / 0.687 |                     Student energy; Channel, GAP, KL divergence, Clean | 0.820 / 0.683 |
| CIFAR-100  | ResNet-50 |             KL (teacher / student); Channel, Flatten, KL divergence, Clean | 0.850 / 0.668 | Student MSP; Spatial, Flatten, Centered-logit MSE, Clipped, 50-draw mean | 0.804 / 0.520 |

- CIFAR-10, ResNet-18: GAP beats flattening in 42/96 matched macro ROC-AUC comparisons and 48/96 macro FPR@95 comparisons.
- CIFAR-10, ResNet-18: 50-draw clipped inference beats clean inference in 40/96 matched macro ROC-AUC comparisons and 64/96 macro FPR@95 comparisons.
- CIFAR-10, ResNet-18: centered-logit MSE beats KL training in 47/96 matched macro ROC-AUC comparisons and 47/96 macro FPR@95 comparisons.
- CIFAR-10, ResNet-50: GAP beats flattening in 36/96 matched macro ROC-AUC comparisons and 51/96 macro FPR@95 comparisons.
- CIFAR-10, ResNet-50: 50-draw clipped inference beats clean inference in 37/96 matched macro ROC-AUC comparisons and 54/96 macro FPR@95 comparisons.
- CIFAR-10, ResNet-50: centered-logit MSE beats KL training in 41/96 matched macro ROC-AUC comparisons and 41/96 macro FPR@95 comparisons.
- CIFAR-100, ResNet-18: GAP beats flattening in 35/96 matched macro ROC-AUC comparisons and 37/96 macro FPR@95 comparisons.
- CIFAR-100, ResNet-18: 50-draw clipped inference beats clean inference in 40/96 matched macro ROC-AUC comparisons and 38/96 macro FPR@95 comparisons.
- CIFAR-100, ResNet-18: centered-logit MSE beats KL training in 34/96 matched macro ROC-AUC comparisons and 22/96 macro FPR@95 comparisons.
- CIFAR-100, ResNet-50: GAP beats flattening in 40/96 matched macro ROC-AUC comparisons and 46/96 macro FPR@95 comparisons.
- CIFAR-100, ResNet-50: 50-draw clipped inference beats clean inference in 26/96 matched macro ROC-AUC comparisons and 36/96 macro FPR@95 comparisons.
- CIFAR-100, ResNet-50: centered-logit MSE beats KL training in 35/96 matched macro ROC-AUC comparisons and 36/96 macro FPR@95 comparisons.

## Detailed OOD Results

Each configuration below contains all objective and inference-mode tables.

### CIFAR-10 ID, ResNet-18

<details>
<summary>Constant, GAP</summary>

#### Centered-logit MSE, Clean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.510 / 0.702 | 0.453 / 0.722 | 0.593 / 0.662 | 0.519 / 0.696 |
| Absolute max probability difference | 0.828 / 0.914 | 0.845 / 0.846 | 0.831 / 0.830 | 0.835 / 0.863 |
| KL (teacher / student) | 0.837 / 0.875 | 0.870 / 0.736 | 0.841 / 0.756 | 0.849 / 0.789 |
| Centered-logit L2 distance          | 0.279 / 1.000 | 0.535 / 0.999 | 0.521 / 0.977 | 0.445 / 0.992 |
| Energy gap                          | 0.211 / 1.000 | 0.222 / 0.999 | 0.421 / 0.972 | 0.285 / 0.991 |
| Absolute energy gap                 | 0.161 / 1.000 | 0.264 / 0.998 | 0.457 / 0.979 | 0.294 / 0.992 |
| Student MSP                         | 0.878 / 0.711 | 0.871 / 0.715 | 0.837 / 0.741 | 0.862 / 0.722 |
| Student energy                      | 0.934 / 0.480 | 0.900 / 0.585 | 0.841 / 0.702 | 0.892 / 0.589 |

#### Centered-logit MSE, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.627 / 0.650 | 0.640 / 0.694 | 0.589 / 0.709 | 0.619 / 0.684 |
| Absolute max probability difference | 0.373 / 0.942 | 0.361 / 0.944 | 0.411 / 0.925 | 0.382 / 0.937 |
| KL (teacher / student) | 0.245 / 0.983 | 0.215 / 0.999 | 0.308 / 0.971 | 0.256 / 0.984 |
| Centered-logit L2 distance          | 0.022 / 1.000 | 0.092 / 1.000 | 0.175 / 0.997 | 0.097 / 0.999 |
| Energy gap                          | 0.954 / 0.282 | 0.917 / 0.424 | 0.874 / 0.513 | 0.915 / 0.407 |
| Absolute energy gap                 | 0.045 / 1.000 | 0.082 / 1.000 | 0.125 / 0.998 | 0.084 / 0.999 |
| Student MSP                         | 0.819 / 0.687 | 0.759 / 0.648 | 0.779 / 0.691 | 0.786 / 0.675 |
| Student energy                      | 0.915 / 0.457 | 0.817 / 0.580 | 0.785 / 0.704 | 0.839 / 0.581 |

#### KL divergence, Clean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.699 / 0.690 | 0.685 / 0.711 | 0.667 / 0.665 | 0.683 / 0.689 |
| Absolute max probability difference | 0.858 / 0.815 | 0.848 / 0.801 | 0.853 / 0.737 | 0.853 / 0.784 |
| KL (teacher / student) | 0.865 / 0.776 | 0.882 / 0.650 | 0.860 / 0.692 | 0.869 / 0.706 |
| Centered-logit L2 distance          | 0.159 / 1.000 | 0.358 / 1.000 | 0.357 / 0.996 | 0.291 / 0.999 |
| Energy gap                          | 0.117 / 1.000 | 0.127 / 1.000 | 0.254 / 0.997 | 0.166 / 0.999 |
| Absolute energy gap                 | 0.041 / 1.000 | 0.061 / 1.000 | 0.220 / 0.997 | 0.107 / 0.999 |
| Student MSP                         | 0.894 / 0.652 | 0.883 / 0.668 | 0.846 / 0.707 | 0.874 / 0.676 |
| Student energy                      | 0.938 / 0.438 | 0.910 / 0.540 | 0.851 / 0.678 | 0.900 / 0.552 |

#### KL divergence, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.542 / 0.696 | 0.623 / 0.709 | 0.558 / 0.732 | 0.574 / 0.712 |
| Absolute max probability difference | 0.460 / 0.884 | 0.380 / 0.918 | 0.445 / 0.906 | 0.428 / 0.903 |
| KL (teacher / student) | 0.422 / 0.891 | 0.321 / 0.935 | 0.391 / 0.930 | 0.378 / 0.919 |
| Centered-logit L2 distance          | 0.039 / 1.000 | 0.150 / 0.999 | 0.210 / 0.994 | 0.133 / 0.998 |
| Energy gap                          | 0.948 / 0.301 | 0.914 / 0.430 | 0.866 / 0.521 | 0.909 / 0.417 |
| Absolute energy gap                 | 0.048 / 1.000 | 0.083 / 1.000 | 0.130 / 0.997 | 0.087 / 0.999 |
| Student MSP                         | 0.876 / 0.564 | 0.742 / 0.650 | 0.787 / 0.679 | 0.801 / 0.631 |
| Student energy                      | 0.934 / 0.383 | 0.822 / 0.561 | 0.808 / 0.691 | 0.855 / 0.545 |

</details>

<details>
<summary>Constant, Flatten</summary>

#### Centered-logit MSE, Clean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.531 / 0.737 | 0.369 / 0.770 | 0.610 / 0.672 | 0.503 / 0.726 |
| Absolute max probability difference | 0.884 / 0.708 | 0.912 / 0.610 | 0.867 / 0.668 | 0.888 / 0.662 |
| KL (teacher / student) | 0.871 / 0.735 | 0.900 / 0.650 | 0.857 / 0.653 | 0.876 / 0.679 |
| Centered-logit L2 distance          | 0.132 / 1.000 | 0.203 / 1.000 | 0.353 / 0.995 | 0.229 / 0.998 |
| Energy gap                          | 0.207 / 1.000 | 0.137 / 1.000 | 0.292 / 0.998 | 0.212 / 0.999 |
| Absolute energy gap                 | 0.147 / 1.000 | 0.169 / 1.000 | 0.291 / 0.998 | 0.202 / 0.999 |
| Student MSP                         | 0.917 / 0.542 | 0.928 / 0.486 | 0.861 / 0.652 | 0.902 / 0.560 |
| Student energy                      | 0.940 / 0.366 | 0.948 / 0.310 | 0.851 / 0.623 | 0.913 / 0.433 |

#### Centered-logit MSE, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.563 / 0.683 | 0.656 / 0.678 | 0.591 / 0.722 | 0.604 / 0.694 |
| Absolute max probability difference | 0.464 / 0.887 | 0.371 / 0.921 | 0.425 / 0.920 | 0.420 / 0.909 |
| KL (teacher / student) | 0.465 / 0.973 | 0.315 / 0.980 | 0.356 / 0.968 | 0.379 / 0.973 |
| Centered-logit L2 distance          | 0.050 / 1.000 | 0.112 / 1.000 | 0.206 / 0.997 | 0.123 / 0.999 |
| Energy gap                          | 0.932 / 0.344 | 0.916 / 0.405 | 0.854 / 0.539 | 0.901 / 0.429 |
| Absolute energy gap                 | 0.054 / 1.000 | 0.085 / 1.000 | 0.130 / 0.998 | 0.089 / 0.999 |
| Student MSP                         | 0.855 / 0.596 | 0.689 / 0.702 | 0.746 / 0.702 | 0.763 / 0.667 |
| Student energy                      | 0.947 / 0.306 | 0.805 / 0.600 | 0.829 / 0.618 | 0.860 / 0.508 |

#### KL divergence, Clean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.711 / 0.695 | 0.730 / 0.693 | 0.657 / 0.667 | 0.699 / 0.685 |
| Absolute max probability difference | 0.857 / 0.820 | 0.854 / 0.802 | 0.853 / 0.734 | 0.854 / 0.785 |
| KL (teacher / student) | 0.861 / 0.778 | 0.884 / 0.646 | 0.859 / 0.690 | 0.868 / 0.705 |
| Centered-logit L2 distance          | 0.171 / 1.000 | 0.385 / 1.000 | 0.359 / 0.996 | 0.305 / 0.999 |
| Energy gap                          | 0.135 / 1.000 | 0.157 / 1.000 | 0.253 / 0.997 | 0.182 / 0.999 |
| Absolute energy gap                 | 0.061 / 1.000 | 0.095 / 1.000 | 0.220 / 0.997 | 0.125 / 0.999 |
| Student MSP                         | 0.894 / 0.653 | 0.878 / 0.678 | 0.848 / 0.705 | 0.873 / 0.678 |
| Student energy                      | 0.938 / 0.423 | 0.904 / 0.541 | 0.855 / 0.653 | 0.899 / 0.539 |

#### KL divergence, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.536 / 0.698 | 0.654 / 0.689 | 0.562 / 0.729 | 0.584 / 0.705 |
| Absolute max probability difference | 0.465 / 0.885 | 0.349 / 0.926 | 0.440 / 0.909 | 0.418 / 0.907 |
| KL (teacher / student) | 0.379 / 0.976 | 0.226 / 0.983 | 0.371 / 0.960 | 0.325 / 0.973 |
| Centered-logit L2 distance          | 0.034 / 1.000 | 0.126 / 1.000 | 0.204 / 0.996 | 0.121 / 0.998 |
| Energy gap                          | 0.948 / 0.302 | 0.917 / 0.422 | 0.867 / 0.522 | 0.911 / 0.415 |
| Absolute energy gap                 | 0.049 / 1.000 | 0.081 / 1.000 | 0.129 / 0.997 | 0.086 / 0.999 |
| Student MSP                         | 0.886 / 0.530 | 0.710 / 0.679 | 0.786 / 0.676 | 0.794 / 0.628 |
| Student energy                      | 0.940 / 0.344 | 0.802 / 0.587 | 0.807 / 0.680 | 0.849 / 0.537 |

</details>

<details>
<summary>Spatial, GAP</summary>

#### Centered-logit MSE, Clean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.524 / 0.701 | 0.526 / 0.732 | 0.613 / 0.667 | 0.554 / 0.700 |
| Absolute max probability difference | 0.855 / 0.763 | 0.835 / 0.774 | 0.850 / 0.735 | 0.847 / 0.757 |
| KL (teacher / student) | 0.874 / 0.727 | 0.875 / 0.681 | 0.862 / 0.686 | 0.870 / 0.698 |
| Centered-logit L2 distance          | 0.119 / 1.000 | 0.128 / 1.000 | 0.363 / 0.979 | 0.203 / 0.993 |
| Energy gap                          | 0.154 / 1.000 | 0.143 / 1.000 | 0.334 / 0.987 | 0.211 / 0.996 |
| Absolute energy gap                 | 0.081 / 1.000 | 0.087 / 1.000 | 0.317 / 0.987 | 0.162 / 0.996 |
| Student MSP                         | 0.895 / 0.643 | 0.881 / 0.654 | 0.841 / 0.701 | 0.872 / 0.666 |
| Student energy                      | 0.954 / 0.313 | 0.928 / 0.423 | 0.846 / 0.620 | 0.909 / 0.452 |

#### Centered-logit MSE, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.268 / 0.946 | 0.350 / 0.944 | 0.297 / 0.942 | 0.305 / 0.944 |
| Absolute max probability difference | 0.747 / 0.730 | 0.662 / 0.837 | 0.720 / 0.767 | 0.710 / 0.778 |
| KL (teacher / student) | 0.830 / 0.822 | 0.697 / 0.857 | 0.806 / 0.751 | 0.777 / 0.810 |
| Centered-logit L2 distance          | 0.094 / 1.000 | 0.163 / 0.996 | 0.341 / 0.988 | 0.199 / 0.995 |
| Energy gap                          | 0.935 / 0.330 | 0.896 / 0.460 | 0.840 / 0.556 | 0.890 / 0.449 |
| Absolute energy gap                 | 0.065 / 1.000 | 0.104 / 0.999 | 0.160 / 0.995 | 0.110 / 0.998 |
| Student MSP                         | 0.939 / 0.409 | 0.839 / 0.559 | 0.878 / 0.553 | 0.885 / 0.507 |
| Student energy                      | 0.959 / 0.263 | 0.883 / 0.481 | 0.870 / 0.536 | 0.904 / 0.427 |

#### KL divergence, Clean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.473 / 0.712 | 0.397 / 0.763 | 0.611 / 0.679 | 0.493 / 0.718 |
| Absolute max probability difference | 0.862 / 0.781 | 0.868 / 0.779 | 0.847 / 0.771 | 0.859 / 0.777 |
| KL (teacher / student) | 0.878 / 0.765 | 0.885 / 0.675 | 0.861 / 0.709 | 0.874 / 0.716 |
| Centered-logit L2 distance          | 0.129 / 1.000 | 0.224 / 1.000 | 0.395 / 0.995 | 0.249 / 0.998 |
| Energy gap                          | 0.204 / 1.000 | 0.161 / 1.000 | 0.331 / 0.995 | 0.232 / 0.998 |
| Absolute energy gap                 | 0.130 / 1.000 | 0.169 / 0.999 | 0.323 / 0.987 | 0.208 / 0.995 |
| Student MSP                         | 0.896 / 0.654 | 0.887 / 0.646 | 0.838 / 0.722 | 0.874 / 0.674 |
| Student energy                      | 0.941 / 0.405 | 0.927 / 0.448 | 0.844 / 0.673 | 0.904 / 0.509 |

#### KL divergence, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.218 / 0.982 | 0.272 / 0.975 | 0.233 / 0.976 | 0.241 / 0.977 |
| Absolute max probability difference | 0.788 / 0.675 | 0.732 / 0.752 | 0.773 / 0.711 | 0.764 / 0.713 |
| KL (teacher / student) | 0.859 / 0.751 | 0.771 / 0.788 | 0.843 / 0.727 | 0.824 / 0.755 |
| Centered-logit L2 distance          | 0.093 / 1.000 | 0.227 / 0.996 | 0.362 / 0.983 | 0.227 / 0.993 |
| Energy gap                          | 0.939 / 0.311 | 0.891 / 0.457 | 0.838 / 0.545 | 0.889 / 0.438 |
| Absolute energy gap                 | 0.061 / 1.000 | 0.109 / 0.999 | 0.162 / 0.994 | 0.111 / 0.998 |
| Student MSP                         | 0.945 / 0.336 | 0.870 / 0.476 | 0.903 / 0.487 | 0.906 / 0.433 |
| Student energy                      | 0.963 / 0.209 | 0.912 / 0.387 | 0.895 / 0.496 | 0.923 / 0.364 |

</details>

<details>
<summary>Spatial, Flatten</summary>

#### Centered-logit MSE, Clean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.666 / 0.705 | 0.519 / 0.753 | 0.672 / 0.670 | 0.619 / 0.709 |
| Absolute max probability difference | 0.872 / 0.774 | 0.867 / 0.768 | 0.856 / 0.733 | 0.865 / 0.758 |
| KL (teacher / student) | 0.855 / 0.864 | 0.829 / 0.806 | 0.854 / 0.717 | 0.846 / 0.795 |
| Centered-logit L2 distance          | 0.114 / 1.000 | 0.094 / 1.000 | 0.321 / 0.997 | 0.176 / 0.999 |
| Energy gap                          | 0.185 / 1.000 | 0.136 / 1.000 | 0.276 / 0.999 | 0.199 / 1.000 |
| Absolute energy gap                 | 0.124 / 1.000 | 0.086 / 1.000 | 0.255 / 0.999 | 0.155 / 1.000 |
| Student MSP                         | 0.894 / 0.637 | 0.903 / 0.612 | 0.821 / 0.731 | 0.873 / 0.660 |
| Student energy                      | 0.909 / 0.617 | 0.934 / 0.465 | 0.803 / 0.804 | 0.882 / 0.628 |

#### Centered-logit MSE, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.436 / 0.649 | 0.407 / 0.727 | 0.362 / 0.745 | 0.402 / 0.707 |
| Absolute max probability difference | 0.899 / 0.639 | 0.835 / 0.722 | 0.874 / 0.669 | 0.869 / 0.677 |
| KL (teacher / student) | 0.878 / 0.670 | 0.783 / 0.878 | 0.852 / 0.749 | 0.838 / 0.766 |
| Centered-logit L2 distance          | 0.775 / 0.858 | 0.487 / 0.950 | 0.805 / 0.827 | 0.689 / 0.878 |
| Energy gap                          | 0.724 / 0.775 | 0.542 / 0.961 | 0.483 / 0.961 | 0.583 / 0.899 |
| Absolute energy gap                 | 0.369 / 0.986 | 0.291 / 0.977 | 0.415 / 0.948 | 0.359 / 0.970 |
| Student MSP                         | 0.906 / 0.548 | 0.855 / 0.507 | 0.886 / 0.528 | 0.882 / 0.528 |
| Student energy                      | 0.942 / 0.364 | 0.913 / 0.397 | 0.903 / 0.467 | 0.919 / 0.409 |

#### KL divergence, Clean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.810 / 0.639 | 0.726 / 0.698 | 0.739 / 0.656 | 0.758 / 0.664 |
| Absolute max probability difference | 0.885 / 0.724 | 0.870 / 0.726 | 0.865 / 0.707 | 0.873 / 0.719 |
| KL (teacher / student) | 0.888 / 0.777 | 0.881 / 0.676 | 0.871 / 0.702 | 0.880 / 0.718 |
| Centered-logit L2 distance          | 0.083 / 1.000 | 0.075 / 1.000 | 0.207 / 0.999 | 0.122 / 1.000 |
| Energy gap                          | 0.128 / 1.000 | 0.119 / 1.000 | 0.204 / 0.999 | 0.150 / 1.000 |
| Absolute energy gap                 | 0.126 / 1.000 | 0.115 / 1.000 | 0.205 / 0.999 | 0.149 / 1.000 |
| Student MSP                         | 0.878 / 0.705 | 0.880 / 0.680 | 0.831 / 0.732 | 0.863 / 0.705 |
| Student energy                      | 0.923 / 0.528 | 0.917 / 0.520 | 0.845 / 0.699 | 0.895 / 0.582 |

#### KL divergence, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.223 / 0.930 | 0.271 / 0.937 | 0.222 / 0.943 | 0.239 / 0.937 |
| Absolute max probability difference | 0.848 / 0.704 | 0.780 / 0.717 | 0.834 / 0.689 | 0.821 / 0.703 |
| KL (teacher / student) | 0.877 / 0.758 | 0.788 / 0.755 | 0.865 / 0.721 | 0.843 / 0.745 |
| Centered-logit L2 distance          | 0.230 / 1.000 | 0.419 / 0.988 | 0.524 / 0.964 | 0.391 / 0.984 |
| Energy gap                          | 0.860 / 0.495 | 0.784 / 0.680 | 0.721 / 0.719 | 0.788 / 0.631 |
| Absolute energy gap                 | 0.136 / 1.000 | 0.210 / 0.995 | 0.275 / 0.982 | 0.207 / 0.992 |
| Student MSP                         | 0.932 / 0.429 | 0.855 / 0.503 | 0.900 / 0.503 | 0.896 / 0.478 |
| Student energy                      | 0.955 / 0.258 | 0.917 / 0.376 | 0.899 / 0.479 | 0.924 / 0.371 |

</details>

<details>
<summary>Channel, GAP</summary>

#### Centered-logit MSE, Clean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.567 / 0.753 | 0.697 / 0.677 | 0.638 / 0.689 | 0.634 / 0.706 |
| Absolute max probability difference | 0.882 / 0.735 | 0.893 / 0.668 | 0.868 / 0.690 | 0.881 / 0.698 |
| KL (teacher / student) | 0.890 / 0.707 | 0.885 / 0.653 | 0.862 / 0.676 | 0.879 / 0.678 |
| Centered-logit L2 distance          | 0.315 / 1.000 | 0.228 / 0.973 | 0.396 / 0.953 | 0.313 / 0.975 |
| Energy gap                          | 0.179 / 0.999 | 0.227 / 0.963 | 0.268 / 0.957 | 0.224 / 0.973 |
| Absolute energy gap                 | 0.166 / 0.999 | 0.230 / 0.963 | 0.274 / 0.957 | 0.223 / 0.973 |
| Student MSP                         | 0.921 / 0.554 | 0.892 / 0.621 | 0.872 / 0.640 | 0.895 / 0.605 |
| Student energy                      | 0.956 / 0.285 | 0.910 / 0.439 | 0.876 / 0.524 | 0.914 / 0.416 |

#### Centered-logit MSE, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.136 / 0.910 | 0.414 / 0.737 | 0.428 / 0.719 | 0.326 / 0.789 |
| Absolute max probability difference | 0.952 / 0.316 | 0.798 / 0.742 | 0.823 / 0.755 | 0.857 / 0.604 |
| KL (teacher / student) | 0.882 / 0.882 | 0.750 / 0.924 | 0.812 / 0.790 | 0.815 / 0.865 |
| Centered-logit L2 distance          | 0.798 / 0.752 | 0.387 / 0.986 | 0.692 / 0.907 | 0.626 / 0.882 |
| Energy gap                          | 0.292 / 1.000 | 0.605 / 0.921 | 0.564 / 0.939 | 0.487 / 0.953 |
| Absolute energy gap                 | 0.547 / 0.828 | 0.229 / 0.990 | 0.330 / 0.976 | 0.369 / 0.931 |
| Student MSP                         | 0.991 / 0.045 | 0.825 / 0.568 | 0.836 / 0.626 | 0.884 / 0.413 |
| Student energy                      | 0.998 / 0.002 | 0.866 / 0.521 | 0.854 / 0.610 | 0.906 / 0.378 |

#### KL divergence, Clean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.364 / 0.743 | 0.583 / 0.734 | 0.525 / 0.703 | 0.491 / 0.727 |
| Absolute max probability difference | 0.896 / 0.702 | 0.779 / 0.831 | 0.834 / 0.763 | 0.836 / 0.765 |
| KL (teacher / student) | 0.898 / 0.704 | 0.832 / 0.777 | 0.852 / 0.750 | 0.861 / 0.743 |
| Centered-logit L2 distance          | 0.352 / 1.000 | 0.225 / 0.992 | 0.501 / 0.934 | 0.359 / 0.975 |
| Energy gap                          | 0.394 / 1.000 | 0.431 / 1.000 | 0.470 / 0.996 | 0.432 / 0.999 |
| Absolute energy gap                 | 0.316 / 1.000 | 0.235 / 0.999 | 0.390 / 0.984 | 0.314 / 0.994 |
| Student MSP                         | 0.907 / 0.577 | 0.837 / 0.701 | 0.820 / 0.699 | 0.855 / 0.659 |
| Student energy                      | 0.943 / 0.333 | 0.889 / 0.514 | 0.828 / 0.628 | 0.887 / 0.492 |

#### KL divergence, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.094 / 0.952 | 0.365 / 0.803 | 0.367 / 0.784 | 0.275 / 0.846 |
| Absolute max probability difference | 0.953 / 0.254 | 0.794 / 0.747 | 0.829 / 0.770 | 0.859 / 0.590 |
| KL (teacher / student) | 0.940 / 0.512 | 0.779 / 0.837 | 0.830 / 0.763 | 0.849 / 0.704 |
| Centered-logit L2 distance          | 0.834 / 0.654 | 0.395 / 0.986 | 0.586 / 0.940 | 0.605 / 0.860 |
| Energy gap                          | 0.410 / 1.000 | 0.597 / 0.998 | 0.574 / 0.991 | 0.527 / 0.996 |
| Absolute energy gap                 | 0.498 / 0.899 | 0.259 / 0.992 | 0.321 / 0.979 | 0.359 / 0.957 |
| Student MSP                         | 0.996 / 0.014 | 0.840 / 0.554 | 0.856 / 0.606 | 0.897 / 0.392 |
| Student energy                      | 0.998 / 0.000 | 0.884 / 0.492 | 0.858 / 0.594 | 0.913 / 0.362 |

</details>

<details>
<summary>Channel, Flatten</summary>

#### Centered-logit MSE, Clean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.664 / 0.699 | 0.733 / 0.652 | 0.660 / 0.687 | 0.686 / 0.679 |
| Absolute max probability difference | 0.894 / 0.705 | 0.897 / 0.659 | 0.871 / 0.680 | 0.887 / 0.681 |
| KL (teacher / student) | 0.895 / 0.699 | 0.898 / 0.611 | 0.866 / 0.664 | 0.886 / 0.658 |
| Centered-logit L2 distance          | 0.269 / 1.000 | 0.266 / 0.982 | 0.381 / 0.979 | 0.306 / 0.987 |
| Energy gap                          | 0.225 / 0.999 | 0.224 / 0.988 | 0.256 / 0.983 | 0.235 / 0.990 |
| Absolute energy gap                 | 0.216 / 0.999 | 0.230 / 0.988 | 0.263 / 0.983 | 0.237 / 0.990 |
| Student MSP                         | 0.912 / 0.572 | 0.884 / 0.631 | 0.872 / 0.638 | 0.889 / 0.614 |
| Student energy                      | 0.941 / 0.349 | 0.905 / 0.454 | 0.875 / 0.533 | 0.907 / 0.445 |

#### Centered-logit MSE, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.139 / 0.907 | 0.418 / 0.737 | 0.413 / 0.732 | 0.323 / 0.792 |
| Absolute max probability difference | 0.956 / 0.303 | 0.797 / 0.740 | 0.831 / 0.748 | 0.862 / 0.597 |
| KL (teacher / student) | 0.894 / 0.835 | 0.758 / 0.904 | 0.819 / 0.782 | 0.823 / 0.840 |
| Centered-logit L2 distance          | 0.812 / 0.722 | 0.442 / 0.985 | 0.709 / 0.903 | 0.654 / 0.870 |
| Energy gap                          | 0.279 / 1.000 | 0.574 / 0.944 | 0.539 / 0.958 | 0.464 / 0.967 |
| Absolute energy gap                 | 0.547 / 0.822 | 0.231 / 0.988 | 0.335 / 0.973 | 0.371 / 0.928 |
| Student MSP                         | 0.991 / 0.042 | 0.825 / 0.567 | 0.847 / 0.606 | 0.887 / 0.405 |
| Student energy                      | 0.998 / 0.003 | 0.874 / 0.508 | 0.861 / 0.594 | 0.911 / 0.368 |

#### KL divergence, Clean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.499 / 0.682 | 0.712 / 0.695 | 0.620 / 0.676 | 0.610 / 0.684 |
| Absolute max probability difference | 0.900 / 0.618 | 0.877 / 0.761 | 0.870 / 0.682 | 0.882 / 0.687 |
| KL (teacher / student) | 0.896 / 0.649 | 0.856 / 0.868 | 0.867 / 0.702 | 0.873 / 0.740 |
| Centered-logit L2 distance          | 0.354 / 1.000 | 0.145 / 1.000 | 0.408 / 0.998 | 0.302 / 0.999 |
| Energy gap                          | 0.216 / 1.000 | 0.233 / 1.000 | 0.320 / 0.997 | 0.256 / 0.999 |
| Absolute energy gap                 | 0.209 / 1.000 | 0.229 / 1.000 | 0.324 / 0.997 | 0.254 / 0.999 |
| Student MSP                         | 0.911 / 0.558 | 0.868 / 0.660 | 0.844 / 0.674 | 0.874 / 0.631 |
| Student energy                      | 0.946 / 0.288 | 0.910 / 0.424 | 0.843 / 0.554 | 0.900 / 0.422 |

#### KL divergence, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.142 / 0.895 | 0.379 / 0.754 | 0.399 / 0.732 | 0.307 / 0.794 |
| Absolute max probability difference | 0.962 / 0.284 | 0.824 / 0.727 | 0.844 / 0.740 | 0.877 / 0.584 |
| KL (teacher / student) | 0.930 / 0.607 | 0.788 / 0.876 | 0.839 / 0.742 | 0.852 / 0.742 |
| Centered-logit L2 distance          | 0.452 / 1.000 | 0.149 / 1.000 | 0.403 / 0.996 | 0.335 / 0.999 |
| Energy gap                          | 0.202 / 1.000 | 0.443 / 1.000 | 0.447 / 0.995 | 0.364 / 0.998 |
| Absolute energy gap                 | 0.412 / 0.999 | 0.138 / 1.000 | 0.256 / 0.995 | 0.269 / 0.998 |
| Student MSP                         | 0.994 / 0.030 | 0.852 / 0.546 | 0.858 / 0.612 | 0.901 / 0.396 |
| Student energy                      | 0.998 / 0.001 | 0.897 / 0.482 | 0.859 / 0.581 | 0.918 / 0.355 |

</details>

### CIFAR-10 ID, ResNet-50

<details>
<summary>Constant, GAP</summary>

#### Centered-logit MSE, Clean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.500 / 0.725 | 0.394 / 0.764 | 0.457 / 0.704 | 0.450 / 0.731 |
| Absolute max probability difference | 0.693 / 0.892 | 0.782 / 0.830 | 0.771 / 0.767 | 0.748 / 0.830 |
| KL (teacher / student) | 0.691 / 0.856 | 0.820 / 0.733 | 0.796 / 0.665 | 0.769 / 0.751 |
| Centered-logit L2 distance          | 0.115 / 1.000 | 0.536 / 0.998 | 0.542 / 0.979 | 0.398 / 0.992 |
| Energy gap                          | 0.668 / 0.999 | 0.795 / 0.938 | 0.688 / 0.882 | 0.717 / 0.940 |
| Absolute energy gap                 | 0.225 / 1.000 | 0.238 / 1.000 | 0.376 / 0.998 | 0.279 / 0.999 |
| Student MSP                         | 0.761 / 0.769 | 0.812 / 0.779 | 0.792 / 0.717 | 0.788 / 0.755 |
| Student energy                      | 0.757 / 0.862 | 0.627 / 0.947 | 0.689 / 0.828 | 0.691 / 0.879 |

#### Centered-logit MSE, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.616 / 0.737 | 0.767 / 0.732 | 0.689 / 0.701 | 0.691 / 0.723 |
| Absolute max probability difference | 0.384 / 0.993 | 0.233 / 1.000 | 0.311 / 0.986 | 0.309 / 0.993 |
| KL (teacher / student) | 0.384 / 0.998 | 0.153 / 0.996 | 0.355 / 0.941 | 0.298 / 0.978 |
| Centered-logit L2 distance          | 0.073 / 1.000 | 0.107 / 1.000 | 0.218 / 0.991 | 0.133 / 0.997 |
| Energy gap                          | 0.927 / 0.434 | 0.917 / 0.506 | 0.870 / 0.518 | 0.905 / 0.486 |
| Absolute energy gap                 | 0.073 / 1.000 | 0.083 / 1.000 | 0.130 / 0.997 | 0.095 / 0.999 |
| Student MSP                         | 0.718 / 0.932 | 0.405 / 0.998 | 0.569 / 0.948 | 0.564 / 0.959 |
| Student energy                      | 0.742 / 0.936 | 0.435 / 0.997 | 0.534 / 0.961 | 0.570 / 0.965 |

#### KL divergence, Clean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.738 / 0.603 | 0.345 / 0.780 | 0.514 / 0.690 | 0.532 / 0.691 |
| Absolute max probability difference | 0.829 / 0.763 | 0.869 / 0.690 | 0.835 / 0.697 | 0.844 / 0.716 |
| KL (teacher / student) | 0.851 / 0.749 | 0.885 / 0.641 | 0.851 / 0.629 | 0.863 / 0.673 |
| Centered-logit L2 distance          | 0.219 / 1.000 | 0.556 / 0.992 | 0.578 / 0.956 | 0.451 / 0.983 |
| Energy gap                          | 0.615 / 1.000 | 0.457 / 1.000 | 0.515 / 0.987 | 0.529 / 0.995 |
| Absolute energy gap                 | 0.480 / 1.000 | 0.262 / 1.000 | 0.415 / 0.987 | 0.386 / 0.996 |
| Student MSP                         | 0.769 / 0.761 | 0.878 / 0.663 | 0.815 / 0.695 | 0.821 / 0.706 |
| Student energy                      | 0.786 / 0.732 | 0.833 / 0.736 | 0.771 / 0.713 | 0.796 / 0.727 |

#### KL divergence, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.436 / 0.747 | 0.528 / 0.754 | 0.578 / 0.711 | 0.514 / 0.737 |
| Absolute max probability difference | 0.564 / 0.741 | 0.472 / 0.908 | 0.422 / 0.923 | 0.486 / 0.857 |
| KL (teacher / student) | 0.602 / 0.691 | 0.505 / 0.838 | 0.430 / 0.907 | 0.513 / 0.812 |
| Centered-logit L2 distance          | 0.094 / 1.000 | 0.177 / 0.999 | 0.246 / 0.988 | 0.172 / 0.996 |
| Energy gap                          | 0.925 / 0.433 | 0.912 / 0.513 | 0.867 / 0.516 | 0.901 / 0.487 |
| Absolute energy gap                 | 0.075 / 1.000 | 0.088 / 1.000 | 0.132 / 0.997 | 0.098 / 0.999 |
| Student MSP                         | 0.938 / 0.347 | 0.768 / 0.799 | 0.770 / 0.731 | 0.826 / 0.626 |
| Student energy                      | 0.910 / 0.422 | 0.685 / 0.853 | 0.683 / 0.820 | 0.760 / 0.698 |

</details>

<details>
<summary>Constant, Flatten</summary>

#### Centered-logit MSE, Clean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.529 / 0.719 | 0.487 / 0.748 | 0.537 / 0.687 | 0.518 / 0.718 |
| Absolute max probability difference | 0.778 / 0.895 | 0.784 / 0.880 | 0.799 / 0.842 | 0.787 / 0.873 |
| KL (teacher / student) | 0.794 / 0.851 | 0.792 / 0.864 | 0.826 / 0.740 | 0.804 / 0.818 |
| Centered-logit L2 distance          | 0.457 / 0.999 | 0.244 / 0.999 | 0.494 / 0.967 | 0.398 / 0.988 |
| Energy gap                          | 0.585 / 1.000 | 0.426 / 1.000 | 0.481 / 0.994 | 0.497 / 0.998 |
| Absolute energy gap                 | 0.439 / 1.000 | 0.246 / 1.000 | 0.368 / 0.994 | 0.351 / 0.998 |
| Student MSP                         | 0.805 / 0.842 | 0.795 / 0.853 | 0.779 / 0.819 | 0.793 / 0.838 |
| Student energy                      | 0.703 / 0.934 | 0.796 / 0.858 | 0.737 / 0.851 | 0.746 / 0.881 |

#### Centered-logit MSE, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.370 / 0.815 | 0.543 / 0.778 | 0.504 / 0.777 | 0.472 / 0.790 |
| Absolute max probability difference | 0.642 / 0.736 | 0.470 / 0.913 | 0.512 / 0.878 | 0.541 / 0.842 |
| KL (teacher / student) | 0.672 / 0.724 | 0.326 / 0.928 | 0.521 / 0.871 | 0.507 / 0.841 |
| Centered-logit L2 distance          | 0.193 / 0.992 | 0.170 / 0.998 | 0.264 / 0.985 | 0.209 / 0.991 |
| Energy gap                          | 0.881 / 0.512 | 0.901 / 0.521 | 0.846 / 0.544 | 0.876 / 0.526 |
| Absolute energy gap                 | 0.095 / 1.000 | 0.081 / 1.000 | 0.139 / 0.996 | 0.105 / 0.999 |
| Student MSP                         | 0.906 / 0.524 | 0.684 / 0.818 | 0.771 / 0.690 | 0.787 / 0.678 |
| Student energy                      | 0.925 / 0.445 | 0.779 / 0.747 | 0.792 / 0.682 | 0.832 / 0.625 |

#### KL divergence, Clean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.767 / 0.593 | 0.495 / 0.760 | 0.592 / 0.679 | 0.618 / 0.677 |
| Absolute max probability difference | 0.875 / 0.693 | 0.852 / 0.705 | 0.849 / 0.665 | 0.859 / 0.688 |
| KL (teacher / student) | 0.900 / 0.648 | 0.881 / 0.678 | 0.861 / 0.625 | 0.881 / 0.650 |
| Centered-logit L2 distance          | 0.439 / 1.000 | 0.497 / 0.999 | 0.524 / 0.976 | 0.487 / 0.992 |
| Energy gap                          | 0.571 / 1.000 | 0.349 / 1.000 | 0.416 / 0.995 | 0.445 / 0.998 |
| Absolute energy gap                 | 0.561 / 1.000 | 0.330 / 1.000 | 0.407 / 0.995 | 0.433 / 0.998 |
| Student MSP                         | 0.750 / 0.774 | 0.875 / 0.673 | 0.820 / 0.692 | 0.815 / 0.713 |
| Student energy                      | 0.742 / 0.759 | 0.842 / 0.698 | 0.782 / 0.689 | 0.788 / 0.715 |

#### KL divergence, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.433 / 0.767 | 0.531 / 0.769 | 0.561 / 0.732 | 0.508 / 0.756 |
| Absolute max probability difference | 0.568 / 0.778 | 0.469 / 0.916 | 0.439 / 0.918 | 0.492 / 0.871 |
| KL (teacher / student) | 0.619 / 0.724 | 0.398 / 0.951 | 0.436 / 0.911 | 0.484 / 0.862 |
| Centered-logit L2 distance          | 0.097 / 1.000 | 0.170 / 0.999 | 0.246 / 0.988 | 0.171 / 0.996 |
| Energy gap                          | 0.922 / 0.442 | 0.911 / 0.518 | 0.866 / 0.520 | 0.900 / 0.493 |
| Absolute energy gap                 | 0.077 / 1.000 | 0.087 / 1.000 | 0.133 / 0.997 | 0.099 / 0.999 |
| Student MSP                         | 0.915 / 0.467 | 0.746 / 0.808 | 0.769 / 0.731 | 0.810 / 0.669 |
| Student energy                      | 0.886 / 0.511 | 0.699 / 0.842 | 0.702 / 0.806 | 0.763 / 0.720 |

</details>

<details>
<summary>Spatial, GAP</summary>

#### Centered-logit MSE, Clean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.551 / 0.850 | 0.542 / 0.828 | 0.508 / 0.803 | 0.534 / 0.827 |
| Absolute max probability difference | 0.507 / 0.991 | 0.530 / 0.997 | 0.583 / 0.971 | 0.540 / 0.986 |
| KL (teacher / student) | 0.553 / 0.925 | 0.630 / 0.862 | 0.693 / 0.775 | 0.625 / 0.854 |
| Centered-logit L2 distance          | 0.105 / 1.000 | 0.312 / 1.000 | 0.448 / 0.985 | 0.288 / 0.995 |
| Energy gap                          | 0.852 / 0.788 | 0.913 / 0.630 | 0.799 / 0.655 | 0.855 / 0.691 |
| Absolute energy gap                 | 0.146 / 1.000 | 0.084 / 1.000 | 0.201 / 0.999 | 0.144 / 1.000 |
| Student MSP                         | 0.605 / 0.977 | 0.605 / 0.993 | 0.655 / 0.939 | 0.622 / 0.970 |
| Student energy                      | 0.573 / 0.997 | 0.459 / 0.999 | 0.545 / 0.984 | 0.526 / 0.994 |

#### Centered-logit MSE, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.685 / 0.805 | 0.777 / 0.802 | 0.694 / 0.773 | 0.719 / 0.793 |
| Absolute max probability difference | 0.315 / 1.000 | 0.223 / 1.000 | 0.306 / 0.993 | 0.281 / 0.998 |
| KL (teacher / student) | 0.422 / 0.991 | 0.262 / 0.989 | 0.439 / 0.922 | 0.374 / 0.967 |
| Centered-logit L2 distance          | 0.068 / 1.000 | 0.112 / 1.000 | 0.226 / 0.989 | 0.135 / 0.996 |
| Energy gap                          | 0.930 / 0.420 | 0.914 / 0.523 | 0.870 / 0.520 | 0.905 / 0.488 |
| Absolute energy gap                 | 0.070 / 1.000 | 0.086 / 1.000 | 0.130 / 0.997 | 0.095 / 0.999 |
| Student MSP                         | 0.528 / 1.000 | 0.341 / 1.000 | 0.491 / 0.985 | 0.453 / 0.995 |
| Student energy                      | 0.532 / 1.000 | 0.453 / 1.000 | 0.491 / 0.982 | 0.492 / 0.994 |

#### KL divergence, Clean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.745 / 0.640 | 0.439 / 0.733 | 0.581 / 0.674 | 0.589 / 0.682 |
| Absolute max probability difference | 0.831 / 0.817 | 0.868 / 0.716 | 0.834 / 0.725 | 0.844 / 0.752 |
| KL (teacher / student) | 0.849 / 0.856 | 0.888 / 0.670 | 0.851 / 0.673 | 0.863 / 0.733 |
| Centered-logit L2 distance          | 0.338 / 1.000 | 0.480 / 0.998 | 0.576 / 0.960 | 0.464 / 0.986 |
| Energy gap                          | 0.496 / 1.000 | 0.397 / 1.000 | 0.456 / 0.994 | 0.450 / 0.998 |
| Absolute energy gap                 | 0.435 / 1.000 | 0.294 / 1.000 | 0.394 / 0.994 | 0.374 / 0.998 |
| Student MSP                         | 0.775 / 0.802 | 0.858 / 0.725 | 0.794 / 0.744 | 0.809 / 0.757 |
| Student energy                      | 0.746 / 0.865 | 0.790 / 0.879 | 0.728 / 0.829 | 0.755 / 0.858 |

#### KL divergence, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.288 / 0.937 | 0.259 / 0.950 | 0.364 / 0.918 | 0.304 / 0.935 |
| Absolute max probability difference | 0.712 / 0.678 | 0.741 / 0.613 | 0.637 / 0.810 | 0.697 / 0.701 |
| KL (teacher / student) | 0.874 / 0.568 | 0.852 / 0.599 | 0.767 / 0.778 | 0.831 / 0.648 |
| Centered-logit L2 distance          | 0.140 / 1.000 | 0.295 / 0.997 | 0.311 / 0.978 | 0.249 / 0.992 |
| Energy gap                          | 0.922 / 0.429 | 0.900 / 0.547 | 0.858 / 0.524 | 0.893 / 0.500 |
| Absolute energy gap                 | 0.078 / 1.000 | 0.100 / 1.000 | 0.142 / 0.995 | 0.107 / 0.998 |
| Student MSP                         | 0.928 / 0.408 | 0.918 / 0.398 | 0.859 / 0.588 | 0.902 / 0.465 |
| Student energy                      | 0.933 / 0.357 | 0.911 / 0.349 | 0.835 / 0.583 | 0.893 / 0.430 |

</details>

<details>
<summary>Spatial, Flatten</summary>

#### Centered-logit MSE, Clean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.791 / 0.657 | 0.632 / 0.726 | 0.732 / 0.648 | 0.718 / 0.677 |
| Absolute max probability difference | 0.774 / 0.891 | 0.777 / 0.893 | 0.771 / 0.874 | 0.774 / 0.886 |
| KL (teacher / student) | 0.736 / 0.973 | 0.717 / 0.950 | 0.781 / 0.821 | 0.745 / 0.915 |
| Centered-logit L2 distance          | 0.295 / 1.000 | 0.125 / 1.000 | 0.363 / 0.993 | 0.261 / 0.998 |
| Energy gap                          | 0.378 / 1.000 | 0.276 / 1.000 | 0.394 / 0.998 | 0.349 / 0.999 |
| Absolute energy gap                 | 0.370 / 1.000 | 0.250 / 1.000 | 0.384 / 0.998 | 0.335 / 0.999 |
| Student MSP                         | 0.656 / 0.941 | 0.756 / 0.904 | 0.639 / 0.932 | 0.683 / 0.926 |
| Student energy                      | 0.675 / 0.965 | 0.797 / 0.923 | 0.653 / 0.952 | 0.709 / 0.947 |

#### Centered-logit MSE, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.308 / 0.781 | 0.307 / 0.810 | 0.361 / 0.765 | 0.325 / 0.785 |
| Absolute max probability difference | 0.902 / 0.627 | 0.863 / 0.667 | 0.860 / 0.698 | 0.875 / 0.664 |
| KL (teacher / student) | 0.874 / 0.756 | 0.808 / 0.875 | 0.845 / 0.753 | 0.842 / 0.795 |
| Centered-logit L2 distance          | 0.674 / 0.938 | 0.666 / 0.927 | 0.739 / 0.879 | 0.693 / 0.915 |
| Energy gap                          | 0.562 / 0.915 | 0.463 / 0.965 | 0.526 / 0.935 | 0.517 / 0.939 |
| Absolute energy gap                 | 0.311 / 0.981 | 0.378 / 0.967 | 0.356 / 0.956 | 0.348 / 0.968 |
| Student MSP                         | 0.915 / 0.481 | 0.879 / 0.537 | 0.873 / 0.560 | 0.889 / 0.526 |
| Student energy                      | 0.935 / 0.445 | 0.918 / 0.448 | 0.882 / 0.536 | 0.911 / 0.476 |

#### KL divergence, Clean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.802 / 0.615 | 0.493 / 0.731 | 0.642 / 0.669 | 0.646 / 0.672 |
| Absolute max probability difference | 0.869 / 0.772 | 0.879 / 0.693 | 0.852 / 0.707 | 0.867 / 0.724 |
| KL (teacher / student) | 0.880 / 0.801 | 0.889 / 0.689 | 0.859 / 0.682 | 0.876 / 0.724 |
| Centered-logit L2 distance          | 0.483 / 0.995 | 0.370 / 1.000 | 0.468 / 0.978 | 0.440 / 0.991 |
| Energy gap                          | 0.486 / 1.000 | 0.260 / 1.000 | 0.361 / 0.997 | 0.369 / 0.999 |
| Absolute energy gap                 | 0.482 / 1.000 | 0.250 / 1.000 | 0.355 / 0.997 | 0.362 / 0.999 |
| Student MSP                         | 0.771 / 0.786 | 0.879 / 0.674 | 0.809 / 0.723 | 0.820 / 0.728 |
| Student energy                      | 0.727 / 0.844 | 0.855 / 0.720 | 0.772 / 0.745 | 0.785 / 0.769 |

#### KL divergence, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.282 / 0.955 | 0.218 / 0.972 | 0.311 / 0.951 | 0.270 / 0.959 |
| Absolute max probability difference | 0.723 / 0.743 | 0.783 / 0.597 | 0.692 / 0.783 | 0.733 / 0.708 |
| KL (teacher / student) | 0.855 / 0.677 | 0.854 / 0.664 | 0.806 / 0.772 | 0.838 / 0.704 |
| Centered-logit L2 distance          | 0.132 / 1.000 | 0.339 / 0.995 | 0.341 / 0.974 | 0.271 / 0.990 |
| Energy gap                          | 0.919 / 0.430 | 0.887 / 0.577 | 0.848 / 0.537 | 0.884 / 0.515 |
| Absolute energy gap                 | 0.081 / 1.000 | 0.113 / 1.000 | 0.152 / 0.995 | 0.116 / 0.998 |
| Student MSP                         | 0.897 / 0.541 | 0.919 / 0.409 | 0.871 / 0.565 | 0.896 / 0.505 |
| Student energy                      | 0.883 / 0.507 | 0.921 / 0.328 | 0.847 / 0.556 | 0.884 / 0.464 |

</details>

<details>
<summary>Channel, GAP</summary>

#### Centered-logit MSE, Clean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.756 / 0.600 | 0.475 / 0.752 | 0.647 / 0.680 | 0.626 / 0.677 |
| Absolute max probability difference | 0.899 / 0.615 | 0.894 / 0.560 | 0.866 / 0.631 | 0.887 / 0.602 |
| KL (teacher / student) | 0.906 / 0.594 | 0.906 / 0.516 | 0.862 / 0.627 | 0.892 / 0.579 |
| Centered-logit L2 distance          | 0.564 / 1.000 | 0.521 / 1.000 | 0.461 / 0.996 | 0.516 / 0.999 |
| Energy gap                          | 0.405 / 1.000 | 0.152 / 1.000 | 0.275 / 0.999 | 0.277 / 1.000 |
| Absolute energy gap                 | 0.405 / 1.000 | 0.152 / 1.000 | 0.275 / 0.999 | 0.277 / 1.000 |
| Student MSP                         | 0.863 / 0.686 | 0.907 / 0.579 | 0.861 / 0.637 | 0.877 / 0.634 |
| Student energy                      | 0.888 / 0.579 | 0.924 / 0.424 | 0.853 / 0.549 | 0.888 / 0.517 |

#### Centered-logit MSE, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.229 / 0.815 | 0.437 / 0.724 | 0.435 / 0.694 | 0.367 / 0.745 |
| Absolute max probability difference | 0.930 / 0.467 | 0.794 / 0.814 | 0.833 / 0.769 | 0.852 / 0.683 |
| KL (teacher / student) | 0.889 / 0.818 | 0.776 / 0.909 | 0.829 / 0.782 | 0.831 / 0.836 |
| Centered-logit L2 distance          | 0.846 / 0.788 | 0.622 / 0.979 | 0.764 / 0.876 | 0.744 / 0.881 |
| Energy gap                          | 0.269 / 0.998 | 0.625 / 0.944 | 0.495 / 0.946 | 0.463 / 0.963 |
| Absolute energy gap                 | 0.627 / 0.830 | 0.264 / 0.994 | 0.463 / 0.958 | 0.452 / 0.927 |
| Student MSP                         | 0.950 / 0.296 | 0.800 / 0.720 | 0.832 / 0.679 | 0.861 / 0.565 |
| Student energy                      | 0.975 / 0.160 | 0.810 / 0.845 | 0.839 / 0.693 | 0.875 / 0.566 |

#### KL divergence, Clean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.565 / 0.655 | 0.312 / 0.791 | 0.446 / 0.720 | 0.441 / 0.722 |
| Absolute max probability difference | 0.799 / 0.742 | 0.878 / 0.662 | 0.832 / 0.690 | 0.836 / 0.698 |
| KL (teacher / student) | 0.815 / 0.758 | 0.890 / 0.615 | 0.842 / 0.665 | 0.849 / 0.680 |
| Centered-logit L2 distance          | 0.215 / 1.000 | 0.470 / 0.999 | 0.456 / 0.971 | 0.380 / 0.990 |
| Energy gap                          | 0.755 / 0.951 | 0.567 / 0.997 | 0.612 / 0.965 | 0.645 / 0.971 |
| Absolute energy gap                 | 0.197 / 1.000 | 0.410 / 1.000 | 0.367 / 0.997 | 0.325 / 0.999 |
| Student MSP                         | 0.840 / 0.696 | 0.890 / 0.634 | 0.845 / 0.645 | 0.858 / 0.658 |
| Student energy                      | 0.845 / 0.639 | 0.883 / 0.601 | 0.825 / 0.613 | 0.851 / 0.618 |

#### KL divergence, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.160 / 0.899 | 0.371 / 0.826 | 0.340 / 0.820 | 0.290 / 0.848 |
| Absolute max probability difference | 0.925 / 0.406 | 0.758 / 0.887 | 0.809 / 0.818 | 0.831 / 0.704 |
| KL (teacher / student) | 0.918 / 0.662 | 0.752 / 0.913 | 0.822 / 0.784 | 0.831 / 0.786 |
| Centered-logit L2 distance          | 0.671 / 0.854 | 0.277 / 0.990 | 0.574 / 0.926 | 0.507 / 0.924 |
| Energy gap                          | 0.660 / 0.777 | 0.895 / 0.515 | 0.749 / 0.690 | 0.768 / 0.661 |
| Absolute energy gap                 | 0.339 / 0.968 | 0.105 / 0.998 | 0.251 / 0.982 | 0.232 / 0.983 |
| Student MSP                         | 0.961 / 0.239 | 0.793 / 0.782 | 0.839 / 0.676 | 0.864 / 0.566 |
| Student energy                      | 0.971 / 0.206 | 0.733 / 0.892 | 0.824 / 0.705 | 0.842 / 0.601 |

</details>

<details>
<summary>Channel, Flatten</summary>

#### Centered-logit MSE, Clean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.801 / 0.571 | 0.490 / 0.758 | 0.668 / 0.675 | 0.653 / 0.668 |
| Absolute max probability difference | 0.898 / 0.648 | 0.892 / 0.601 | 0.867 / 0.656 | 0.885 / 0.635 |
| KL (teacher / student) | 0.912 / 0.593 | 0.897 / 0.553 | 0.862 / 0.641 | 0.890 / 0.596 |
| Centered-logit L2 distance          | 0.744 / 0.968 | 0.378 / 0.991 | 0.435 / 0.986 | 0.519 / 0.982 |
| Energy gap                          | 0.574 / 0.974 | 0.157 / 0.994 | 0.289 / 0.989 | 0.340 / 0.986 |
| Absolute energy gap                 | 0.574 / 0.974 | 0.157 / 0.994 | 0.289 / 0.989 | 0.340 / 0.986 |
| Student MSP                         | 0.826 / 0.723 | 0.906 / 0.573 | 0.858 / 0.644 | 0.864 / 0.647 |
| Student energy                      | 0.813 / 0.721 | 0.924 / 0.414 | 0.849 / 0.563 | 0.862 / 0.566 |

#### Centered-logit MSE, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.235 / 0.820 | 0.405 / 0.744 | 0.425 / 0.697 | 0.355 / 0.753 |
| Absolute max probability difference | 0.919 / 0.569 | 0.804 / 0.811 | 0.839 / 0.753 | 0.854 / 0.711 |
| KL (teacher / student) | 0.871 / 0.859 | 0.776 / 0.924 | 0.833 / 0.778 | 0.827 / 0.854 |
| Centered-logit L2 distance          | 0.793 / 0.877 | 0.525 / 0.985 | 0.767 / 0.878 | 0.695 / 0.913 |
| Energy gap                          | 0.277 / 0.999 | 0.560 / 0.967 | 0.484 / 0.948 | 0.440 / 0.971 |
| Absolute energy gap                 | 0.598 / 0.867 | 0.270 / 0.994 | 0.471 / 0.957 | 0.446 / 0.939 |
| Student MSP                         | 0.947 / 0.339 | 0.817 / 0.694 | 0.840 / 0.667 | 0.868 / 0.567 |
| Student energy                      | 0.970 / 0.198 | 0.841 / 0.796 | 0.846 / 0.667 | 0.886 / 0.554 |

#### KL divergence, Clean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.789 / 0.605 | 0.535 / 0.759 | 0.592 / 0.683 | 0.639 / 0.682 |
| Absolute max probability difference | 0.876 / 0.735 | 0.860 / 0.744 | 0.859 / 0.686 | 0.865 / 0.722 |
| KL (teacher / student) | 0.888 / 0.735 | 0.866 / 0.729 | 0.861 / 0.650 | 0.872 / 0.705 |
| Centered-logit L2 distance          | 0.393 / 1.000 | 0.260 / 1.000 | 0.435 / 0.997 | 0.362 / 0.999 |
| Energy gap                          | 0.572 / 1.000 | 0.274 / 1.000 | 0.383 / 0.998 | 0.410 / 0.999 |
| Absolute energy gap                 | 0.564 / 1.000 | 0.239 / 1.000 | 0.371 / 0.998 | 0.391 / 0.999 |
| Student MSP                         | 0.759 / 0.771 | 0.887 / 0.660 | 0.829 / 0.670 | 0.825 / 0.701 |
| Student energy                      | 0.730 / 0.803 | 0.890 / 0.596 | 0.800 / 0.663 | 0.807 / 0.688 |

#### KL divergence, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |     CIFAR-100 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.238 / 0.811 | 0.396 / 0.753 | 0.380 / 0.735 | 0.338 / 0.766 |
| Absolute max probability difference | 0.928 / 0.490 | 0.794 / 0.836 | 0.845 / 0.758 | 0.856 / 0.695 |
| KL (teacher / student) | 0.905 / 0.674 | 0.778 / 0.901 | 0.841 / 0.755 | 0.841 / 0.776 |
| Centered-logit L2 distance          | 0.677 / 0.813 | 0.487 / 0.980 | 0.562 / 0.923 | 0.575 / 0.905 |
| Energy gap                          | 0.388 / 1.000 | 0.607 / 1.000 | 0.488 / 0.997 | 0.494 / 0.999 |
| Absolute energy gap                 | 0.451 / 0.941 | 0.156 / 0.996 | 0.352 / 0.974 | 0.320 / 0.970 |
| Student MSP                         | 0.944 / 0.338 | 0.814 / 0.716 | 0.854 / 0.638 | 0.871 / 0.564 |
| Student energy                      | 0.954 / 0.324 | 0.810 / 0.818 | 0.847 / 0.640 | 0.870 / 0.594 |

</details>

### CIFAR-100 ID, ResNet-18

<details>
<summary>Constant, GAP</summary>

#### Centered-logit MSE, Clean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.627 / 0.953 | 0.782 / 0.808 | 0.741 / 0.790 | 0.717 / 0.851 |
| Absolute max probability difference | 0.373 / 1.000 | 0.218 / 1.000 | 0.259 / 0.969 | 0.283 / 0.989 |
| KL (teacher / student) | 0.428 / 1.000 | 0.229 / 0.999 | 0.300 / 0.961 | 0.319 / 0.987 |
| Centered-logit L2 distance          | 0.382 / 1.000 | 0.188 / 1.000 | 0.218 / 0.998 | 0.263 / 0.999 |
| Energy gap                          | 0.696 / 0.974 | 0.827 / 0.789 | 0.796 / 0.792 | 0.773 / 0.851 |
| Absolute energy gap                 | 0.304 / 1.000 | 0.173 / 1.000 | 0.204 / 0.999 | 0.227 / 1.000 |
| Student MSP                         | 0.649 / 0.971 | 0.734 / 0.817 | 0.769 / 0.826 | 0.717 / 0.871 |
| Student energy                      | 0.571 / 0.993 | 0.768 / 0.762 | 0.691 / 0.893 | 0.677 / 0.883 |

#### Centered-logit MSE, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.693 / 0.956 | 0.806 / 0.812 | 0.790 / 0.794 | 0.763 / 0.854 |
| Absolute max probability difference | 0.307 / 0.996 | 0.194 / 1.000 | 0.210 / 0.997 | 0.237 / 0.998 |
| KL (teacher / student) | 0.310 / 1.000 | 0.181 / 1.000 | 0.213 / 0.988 | 0.235 / 0.996 |
| Centered-logit L2 distance          | 0.387 / 1.000 | 0.176 / 1.000 | 0.228 / 0.998 | 0.264 / 0.999 |
| Energy gap                          | 0.694 / 0.975 | 0.829 / 0.772 | 0.795 / 0.799 | 0.773 / 0.849 |
| Absolute energy gap                 | 0.306 / 1.000 | 0.171 / 1.000 | 0.205 / 0.999 | 0.227 / 1.000 |
| Student MSP                         | 0.785 / 0.814 | 0.560 / 0.964 | 0.575 / 0.962 | 0.640 / 0.913 |
| Student energy                      | 0.750 / 0.837 | 0.527 / 0.984 | 0.487 / 0.973 | 0.588 / 0.931 |

#### KL divergence, Clean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.414 / 0.999 | 0.610 / 0.962 | 0.514 / 0.979 | 0.512 / 0.980 |
| Absolute max probability difference | 0.586 / 0.992 | 0.390 / 0.987 | 0.486 / 0.981 | 0.488 / 0.987 |
| KL (teacher / student) | 0.625 / 0.988 | 0.396 / 0.979 | 0.519 / 0.982 | 0.513 / 0.983 |
| Centered-logit L2 distance          | 0.399 / 1.000 | 0.163 / 1.000 | 0.227 / 0.998 | 0.263 / 0.999 |
| Energy gap                          | 0.700 / 0.977 | 0.850 / 0.773 | 0.794 / 0.797 | 0.782 / 0.849 |
| Absolute energy gap                 | 0.300 / 1.000 | 0.150 / 1.000 | 0.206 / 0.998 | 0.218 / 0.999 |
| Student MSP                         | 0.671 / 0.957 | 0.735 / 0.843 | 0.783 / 0.811 | 0.730 / 0.870 |
| Student energy                      | 0.645 / 0.978 | 0.752 / 0.784 | 0.766 / 0.840 | 0.721 / 0.867 |

#### KL divergence, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.672 / 0.957 | 0.807 / 0.812 | 0.783 / 0.795 | 0.754 / 0.854 |
| Absolute max probability difference | 0.328 / 0.988 | 0.193 / 1.000 | 0.217 / 0.997 | 0.246 / 0.995 |
| KL (teacher / student) | 0.357 / 0.958 | 0.177 / 1.000 | 0.214 / 0.995 | 0.250 / 0.984 |
| Centered-logit L2 distance          | 0.407 / 0.999 | 0.168 / 1.000 | 0.230 / 0.998 | 0.268 / 0.999 |
| Energy gap                          | 0.693 / 0.975 | 0.829 / 0.772 | 0.795 / 0.799 | 0.772 / 0.849 |
| Absolute energy gap                 | 0.307 / 1.000 | 0.171 / 1.000 | 0.205 / 0.999 | 0.228 / 1.000 |
| Student MSP                         | 0.810 / 0.695 | 0.540 / 0.967 | 0.635 / 0.943 | 0.662 / 0.868 |
| Student energy                      | 0.786 / 0.763 | 0.554 / 0.967 | 0.556 / 0.962 | 0.632 / 0.898 |

</details>

<details>
<summary>Constant, Flatten</summary>

#### Centered-logit MSE, Clean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.461 / 0.904 | 0.526 / 0.853 | 0.447 / 0.897 | 0.478 / 0.885 |
| Absolute max probability difference | 0.563 / 0.980 | 0.506 / 0.960 | 0.572 / 0.946 | 0.547 / 0.962 |
| KL (teacher / student) | 0.587 / 0.970 | 0.568 / 0.952 | 0.639 / 0.952 | 0.598 / 0.958 |
| Centered-logit L2 distance          | 0.303 / 1.000 | 0.192 / 1.000 | 0.253 / 0.996 | 0.250 / 0.999 |
| Energy gap                          | 0.687 / 0.996 | 0.777 / 0.943 | 0.746 / 0.923 | 0.737 / 0.954 |
| Absolute energy gap                 | 0.272 / 1.000 | 0.186 / 1.000 | 0.237 / 0.998 | 0.232 / 0.999 |
| Student MSP                         | 0.662 / 0.966 | 0.775 / 0.808 | 0.787 / 0.800 | 0.742 / 0.858 |
| Student energy                      | 0.655 / 0.970 | 0.805 / 0.722 | 0.776 / 0.824 | 0.745 / 0.839 |

#### Centered-logit MSE, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.660 / 0.958 | 0.803 / 0.810 | 0.781 / 0.794 | 0.748 / 0.854 |
| Absolute max probability difference | 0.340 / 0.983 | 0.197 / 1.000 | 0.219 / 0.995 | 0.252 / 0.992 |
| KL (teacher / student) | 0.350 / 0.987 | 0.180 / 1.000 | 0.217 / 0.991 | 0.249 / 0.993 |
| Centered-logit L2 distance          | 0.406 / 1.000 | 0.171 / 1.000 | 0.228 / 0.998 | 0.268 / 0.999 |
| Energy gap                          | 0.690 / 0.978 | 0.828 / 0.776 | 0.795 / 0.797 | 0.771 / 0.851 |
| Absolute energy gap                 | 0.310 / 1.000 | 0.172 / 1.000 | 0.205 / 0.999 | 0.229 / 1.000 |
| Student MSP                         | 0.795 / 0.730 | 0.597 / 0.956 | 0.639 / 0.942 | 0.677 / 0.876 |
| Student energy                      | 0.799 / 0.737 | 0.601 / 0.967 | 0.561 / 0.966 | 0.654 / 0.890 |

#### KL divergence, Clean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.742 / 0.909 | 0.750 / 0.646 | 0.635 / 0.896 | 0.709 / 0.817 |
| Absolute max probability difference | 0.670 / 0.958 | 0.804 / 0.698 | 0.683 / 0.925 | 0.719 / 0.860 |
| KL (teacher / student) | 0.683 / 0.962 | 0.852 / 0.631 | 0.747 / 0.906 | 0.760 / 0.833 |
| Centered-logit L2 distance          | 0.419 / 1.000 | 0.310 / 0.996 | 0.318 / 0.997 | 0.349 / 0.998 |
| Energy gap                          | 0.587 / 0.999 | 0.558 / 0.902 | 0.450 / 0.999 | 0.532 / 0.967 |
| Absolute energy gap                 | 0.535 / 0.999 | 0.499 / 0.911 | 0.377 / 1.000 | 0.470 / 0.970 |
| Student MSP                         | 0.668 / 0.958 | 0.751 / 0.842 | 0.783 / 0.808 | 0.734 / 0.870 |
| Student energy                      | 0.663 / 0.970 | 0.780 / 0.783 | 0.782 / 0.817 | 0.742 / 0.857 |

#### KL divergence, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.646 / 0.960 | 0.807 / 0.804 | 0.778 / 0.794 | 0.743 / 0.853 |
| Absolute max probability difference | 0.354 / 0.972 | 0.194 / 1.000 | 0.222 / 0.994 | 0.257 / 0.988 |
| KL (teacher / student) | 0.383 / 0.939 | 0.175 / 0.999 | 0.221 / 0.991 | 0.259 / 0.977 |
| Centered-logit L2 distance          | 0.410 / 0.999 | 0.163 / 1.000 | 0.234 / 0.998 | 0.269 / 0.999 |
| Energy gap                          | 0.688 / 0.980 | 0.830 / 0.774 | 0.795 / 0.800 | 0.771 / 0.851 |
| Absolute energy gap                 | 0.310 / 1.000 | 0.171 / 1.000 | 0.205 / 0.999 | 0.229 / 1.000 |
| Student MSP                         | 0.808 / 0.700 | 0.553 / 0.956 | 0.657 / 0.930 | 0.673 / 0.862 |
| Student energy                      | 0.795 / 0.752 | 0.568 / 0.973 | 0.592 / 0.964 | 0.652 / 0.896 |

</details>

<details>
<summary>Spatial, GAP</summary>

#### Centered-logit MSE, Clean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.714 / 0.951 | 0.808 / 0.777 | 0.751 / 0.804 | 0.758 / 0.844 |
| Absolute max probability difference | 0.286 / 1.000 | 0.192 / 1.000 | 0.249 / 0.985 | 0.242 / 0.995 |
| KL (teacher / student) | 0.250 / 1.000 | 0.174 / 1.000 | 0.321 / 0.955 | 0.248 / 0.985 |
| Centered-logit L2 distance          | 0.317 / 1.000 | 0.135 / 1.000 | 0.285 / 0.995 | 0.246 / 0.998 |
| Energy gap                          | 0.704 / 0.972 | 0.839 / 0.745 | 0.790 / 0.810 | 0.778 / 0.842 |
| Absolute energy gap                 | 0.296 / 1.000 | 0.161 / 1.000 | 0.210 / 0.999 | 0.222 / 1.000 |
| Student MSP                         | 0.318 / 1.000 | 0.392 / 0.987 | 0.680 / 0.909 | 0.463 / 0.965 |
| Student energy                      | 0.322 / 1.000 | 0.427 / 0.988 | 0.691 / 0.858 | 0.480 / 0.948 |

#### Centered-logit MSE, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.698 / 0.956 | 0.806 / 0.812 | 0.790 / 0.794 | 0.765 / 0.854 |
| Absolute max probability difference | 0.302 / 0.999 | 0.194 / 1.000 | 0.210 / 0.997 | 0.235 / 0.999 |
| KL (teacher / student) | 0.317 / 1.000 | 0.179 / 1.000 | 0.214 / 0.996 | 0.237 / 0.999 |
| Centered-logit L2 distance          | 0.388 / 1.000 | 0.172 / 1.000 | 0.230 / 0.998 | 0.263 / 0.999 |
| Energy gap                          | 0.694 / 0.974 | 0.829 / 0.772 | 0.795 / 0.799 | 0.773 / 0.848 |
| Absolute energy gap                 | 0.306 / 1.000 | 0.171 / 1.000 | 0.205 / 0.999 | 0.227 / 1.000 |
| Student MSP                         | 0.723 / 0.815 | 0.594 / 0.825 | 0.617 / 0.925 | 0.645 / 0.855 |
| Student energy                      | 0.689 / 0.955 | 0.440 / 0.989 | 0.587 / 0.918 | 0.572 / 0.954 |

#### KL divergence, Clean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.724 / 0.939 | 0.807 / 0.676 | 0.648 / 0.916 | 0.726 / 0.844 |
| Absolute max probability difference | 0.266 / 1.000 | 0.263 / 1.000 | 0.352 / 0.968 | 0.293 / 0.989 |
| KL (teacher / student) | 0.254 / 1.000 | 0.296 / 0.997 | 0.475 / 0.949 | 0.342 / 0.982 |
| Centered-logit L2 distance          | 0.520 / 0.999 | 0.235 / 1.000 | 0.359 / 0.993 | 0.372 / 0.997 |
| Energy gap                          | 0.785 / 0.958 | 0.875 / 0.660 | 0.726 / 0.959 | 0.795 / 0.859 |
| Absolute energy gap                 | 0.193 / 1.000 | 0.214 / 1.000 | 0.247 / 0.997 | 0.218 / 0.999 |
| Student MSP                         | 0.326 / 1.000 | 0.361 / 0.996 | 0.676 / 0.896 | 0.454 / 0.964 |
| Student energy                      | 0.243 / 1.000 | 0.331 / 0.999 | 0.682 / 0.893 | 0.419 / 0.964 |

#### KL divergence, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.690 / 0.956 | 0.806 / 0.809 | 0.786 / 0.793 | 0.760 / 0.853 |
| Absolute max probability difference | 0.310 / 0.998 | 0.194 / 1.000 | 0.214 / 0.998 | 0.240 / 0.999 |
| KL (teacher / student) | 0.379 / 0.932 | 0.179 / 0.999 | 0.222 / 0.998 | 0.260 / 0.976 |
| Centered-logit L2 distance          | 0.412 / 0.999 | 0.164 / 1.000 | 0.233 / 0.998 | 0.270 / 0.999 |
| Energy gap                          | 0.694 / 0.974 | 0.829 / 0.771 | 0.795 / 0.799 | 0.773 / 0.848 |
| Absolute energy gap                 | 0.306 / 1.000 | 0.171 / 1.000 | 0.205 / 0.999 | 0.227 / 1.000 |
| Student MSP                         | 0.747 / 0.897 | 0.439 / 0.915 | 0.617 / 0.925 | 0.601 / 0.912 |
| Student energy                      | 0.521 / 0.998 | 0.446 / 0.985 | 0.566 / 0.943 | 0.511 / 0.976 |

</details>

<details>
<summary>Spatial, Flatten</summary>

#### Centered-logit MSE, Clean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.749 / 0.887 | 0.800 / 0.707 | 0.644 / 0.928 | 0.731 / 0.841 |
| Absolute max probability difference | 0.250 / 1.000 | 0.222 / 1.000 | 0.354 / 0.967 | 0.275 / 0.989 |
| KL (teacher / student) | 0.252 / 1.000 | 0.279 / 0.999 | 0.446 / 0.969 | 0.325 / 0.989 |
| Centered-logit L2 distance          | 0.315 / 1.000 | 0.268 / 1.000 | 0.275 / 0.996 | 0.286 / 0.999 |
| Energy gap                          | 0.737 / 0.987 | 0.868 / 0.668 | 0.742 / 0.942 | 0.782 / 0.866 |
| Absolute energy gap                 | 0.252 / 1.000 | 0.161 / 1.000 | 0.246 / 0.998 | 0.220 / 0.999 |
| Student MSP                         | 0.323 / 0.999 | 0.429 / 0.979 | 0.731 / 0.847 | 0.494 / 0.942 |
| Student energy                      | 0.344 / 0.999 | 0.448 / 0.968 | 0.712 / 0.866 | 0.501 / 0.944 |

#### Centered-logit MSE, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.691 / 0.956 | 0.806 / 0.811 | 0.787 / 0.794 | 0.761 / 0.854 |
| Absolute max probability difference | 0.309 / 0.998 | 0.194 / 1.000 | 0.213 / 0.998 | 0.239 / 0.998 |
| KL (teacher / student) | 0.353 / 0.988 | 0.178 / 1.000 | 0.224 / 0.994 | 0.251 / 0.994 |
| Centered-logit L2 distance          | 0.408 / 1.000 | 0.167 / 1.000 | 0.233 / 0.998 | 0.269 / 0.999 |
| Energy gap                          | 0.694 / 0.974 | 0.829 / 0.772 | 0.795 / 0.798 | 0.772 / 0.848 |
| Absolute energy gap                 | 0.306 / 1.000 | 0.171 / 1.000 | 0.205 / 0.999 | 0.228 / 1.000 |
| Student MSP                         | 0.758 / 0.705 | 0.476 / 0.935 | 0.606 / 0.953 | 0.613 / 0.864 |
| Student energy                      | 0.728 / 0.909 | 0.510 / 0.911 | 0.596 / 0.956 | 0.611 / 0.925 |

#### KL divergence, Clean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.717 / 0.907 | 0.822 / 0.723 | 0.757 / 0.820 | 0.765 / 0.817 |
| Absolute max probability difference | 0.711 / 0.907 | 0.818 / 0.723 | 0.781 / 0.820 | 0.770 / 0.817 |
| KL (teacher / student) | 0.746 / 0.856 | 0.881 / 0.493 | 0.775 / 0.870 | 0.801 / 0.740 |
| Centered-logit L2 distance          | 0.594 / 0.998 | 0.426 / 0.990 | 0.278 / 1.000 | 0.433 / 0.996 |
| Energy gap                          | 0.492 / 0.998 | 0.438 / 0.982 | 0.230 / 1.000 | 0.387 / 0.993 |
| Absolute energy gap                 | 0.492 / 0.998 | 0.438 / 0.982 | 0.230 / 1.000 | 0.387 / 0.993 |
| Student MSP                         | 0.568 / 0.955 | 0.613 / 0.935 | 0.749 / 0.862 | 0.643 / 0.917 |
| Student energy                      | 0.547 / 0.992 | 0.614 / 0.935 | 0.789 / 0.816 | 0.650 / 0.914 |

#### KL divergence, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.571 / 0.972 | 0.796 / 0.797 | 0.731 / 0.813 | 0.699 / 0.861 |
| Absolute max probability difference | 0.427 / 0.975 | 0.204 / 1.000 | 0.268 / 0.994 | 0.299 / 0.990 |
| KL (teacher / student) | 0.595 / 0.781 | 0.213 / 0.995 | 0.305 / 0.992 | 0.371 / 0.923 |
| Centered-logit L2 distance          | 0.497 / 0.997 | 0.199 / 1.000 | 0.234 / 0.997 | 0.310 / 0.998 |
| Energy gap                          | 0.658 / 0.992 | 0.834 / 0.787 | 0.775 / 0.856 | 0.755 / 0.879 |
| Absolute energy gap                 | 0.326 / 1.000 | 0.159 / 1.000 | 0.213 / 0.999 | 0.233 / 0.999 |
| Student MSP                         | 0.798 / 0.697 | 0.494 / 0.948 | 0.664 / 0.905 | 0.652 / 0.850 |
| Student energy                      | 0.770 / 0.713 | 0.480 / 0.949 | 0.631 / 0.929 | 0.627 / 0.864 |

</details>

<details>
<summary>Channel, GAP</summary>

#### Centered-logit MSE, Clean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.227 / 0.964 | 0.230 / 0.973 | 0.524 / 0.865 | 0.327 / 0.934 |
| Absolute max probability difference | 0.793 / 0.795 | 0.782 / 0.737 | 0.607 / 0.948 | 0.727 / 0.827 |
| KL (teacher / student) | 0.808 / 0.838 | 0.855 / 0.688 | 0.688 / 0.939 | 0.783 / 0.822 |
| Centered-logit L2 distance          | 0.495 / 1.000 | 0.252 / 0.999 | 0.379 / 0.995 | 0.375 / 0.998 |
| Energy gap                          | 0.396 / 0.999 | 0.447 / 1.000 | 0.736 / 0.887 | 0.526 / 0.962 |
| Absolute energy gap                 | 0.542 / 0.995 | 0.460 / 0.997 | 0.329 / 0.998 | 0.444 / 0.997 |
| Student MSP                         | 0.744 / 0.929 | 0.854 / 0.691 | 0.772 / 0.809 | 0.790 / 0.810 |
| Student energy                      | 0.741 / 0.953 | 0.900 / 0.533 | 0.756 / 0.837 | 0.799 / 0.774 |

#### Centered-logit MSE, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.352 / 0.999 | 0.808 / 0.678 | 0.648 / 0.861 | 0.603 / 0.846 |
| Absolute max probability difference | 0.641 / 0.806 | 0.244 / 0.999 | 0.368 / 0.991 | 0.418 / 0.932 |
| KL (teacher / student) | 0.760 / 0.796 | 0.244 / 0.996 | 0.473 / 0.984 | 0.492 / 0.925 |
| Centered-logit L2 distance          | 0.717 / 0.957 | 0.137 / 1.000 | 0.405 / 0.998 | 0.419 / 0.985 |
| Energy gap                          | 0.586 / 1.000 | 0.845 / 0.743 | 0.767 / 0.862 | 0.733 / 0.868 |
| Absolute energy gap                 | 0.405 / 0.996 | 0.152 / 1.000 | 0.227 / 0.999 | 0.261 / 0.998 |
| Student MSP                         | 0.922 / 0.426 | 0.463 / 0.994 | 0.679 / 0.905 | 0.688 / 0.775 |
| Student energy                      | 0.928 / 0.410 | 0.536 / 0.995 | 0.657 / 0.916 | 0.707 / 0.774 |

#### KL divergence, Clean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.301 / 0.962 | 0.291 / 0.944 | 0.675 / 0.852 | 0.422 / 0.919 |
| Absolute max probability difference | 0.677 / 0.941 | 0.796 / 0.698 | 0.715 / 0.913 | 0.729 / 0.850 |
| KL (teacher / student) | 0.667 / 0.972 | 0.900 / 0.467 | 0.770 / 0.902 | 0.779 / 0.780 |
| Centered-logit L2 distance          | 0.433 / 0.996 | 0.134 / 0.998 | 0.431 / 0.989 | 0.333 / 0.995 |
| Energy gap                          | 0.159 / 1.000 | 0.103 / 1.000 | 0.439 / 0.977 | 0.234 / 0.992 |
| Absolute energy gap                 | 0.116 / 1.000 | 0.155 / 1.000 | 0.414 / 0.977 | 0.228 / 0.992 |
| Student MSP                         | 0.755 / 0.887 | 0.901 / 0.504 | 0.759 / 0.818 | 0.805 / 0.736 |
| Student energy                      | 0.769 / 0.903 | 0.943 / 0.308 | 0.748 / 0.838 | 0.820 / 0.683 |

#### KL divergence, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.292 / 1.000 | 0.721 / 0.853 | 0.641 / 0.869 | 0.551 / 0.907 |
| Absolute max probability difference | 0.691 / 0.725 | 0.315 / 0.999 | 0.415 / 0.986 | 0.474 / 0.903 |
| KL (teacher / student) | 0.875 / 0.519 | 0.360 / 0.992 | 0.559 / 0.973 | 0.598 / 0.828 |
| Centered-logit L2 distance          | 0.795 / 0.832 | 0.095 / 0.999 | 0.383 / 0.993 | 0.424 / 0.941 |
| Energy gap                          | 0.500 / 1.000 | 0.742 / 0.979 | 0.732 / 0.920 | 0.658 / 0.966 |
| Absolute energy gap                 | 0.395 / 0.990 | 0.141 / 1.000 | 0.252 / 0.999 | 0.263 / 0.996 |
| Student MSP                         | 0.940 / 0.338 | 0.619 / 0.964 | 0.693 / 0.893 | 0.750 / 0.732 |
| Student energy                      | 0.935 / 0.373 | 0.724 / 0.940 | 0.672 / 0.901 | 0.777 / 0.738 |

</details>

<details>
<summary>Channel, Flatten</summary>

#### Centered-logit MSE, Clean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.275 / 0.966 | 0.295 / 0.979 | 0.456 / 0.886 | 0.342 / 0.944 |
| Absolute max probability difference | 0.734 / 0.937 | 0.706 / 0.921 | 0.573 / 0.950 | 0.671 / 0.936 |
| KL (teacher / student) | 0.762 / 0.976 | 0.776 / 0.910 | 0.633 / 0.950 | 0.724 / 0.945 |
| Centered-logit L2 distance          | 0.501 / 1.000 | 0.252 / 0.999 | 0.340 / 0.997 | 0.364 / 0.999 |
| Energy gap                          | 0.587 / 0.996 | 0.670 / 0.998 | 0.793 / 0.793 | 0.683 / 0.929 |
| Absolute energy gap                 | 0.411 / 1.000 | 0.325 / 1.000 | 0.213 / 0.999 | 0.316 / 0.999 |
| Student MSP                         | 0.720 / 0.946 | 0.832 / 0.728 | 0.775 / 0.806 | 0.776 / 0.827 |
| Student energy                      | 0.705 / 0.968 | 0.878 / 0.584 | 0.758 / 0.834 | 0.780 / 0.795 |

#### Centered-logit MSE, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.376 / 0.999 | 0.827 / 0.644 | 0.656 / 0.858 | 0.620 / 0.834 |
| Absolute max probability difference | 0.618 / 0.831 | 0.224 / 1.000 | 0.354 / 0.991 | 0.399 / 0.940 |
| KL (teacher / student) | 0.744 / 0.812 | 0.228 / 0.997 | 0.456 / 0.986 | 0.476 / 0.932 |
| Centered-logit L2 distance          | 0.700 / 0.965 | 0.129 / 1.000 | 0.389 / 0.998 | 0.406 / 0.987 |
| Energy gap                          | 0.602 / 1.000 | 0.853 / 0.718 | 0.769 / 0.861 | 0.742 / 0.859 |
| Absolute energy gap                 | 0.393 / 0.997 | 0.150 / 1.000 | 0.226 / 0.999 | 0.257 / 0.998 |
| Student MSP                         | 0.918 / 0.449 | 0.424 / 0.995 | 0.673 / 0.908 | 0.672 / 0.784 |
| Student energy                      | 0.915 / 0.497 | 0.498 / 0.996 | 0.652 / 0.920 | 0.688 / 0.804 |

#### KL divergence, Clean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.290 / 0.957 | 0.349 / 0.932 | 0.707 / 0.863 | 0.449 / 0.917 |
| Absolute max probability difference | 0.700 / 0.931 | 0.783 / 0.743 | 0.732 / 0.893 | 0.738 / 0.856 |
| KL (teacher / student) | 0.668 / 0.968 | 0.885 / 0.522 | 0.773 / 0.897 | 0.775 / 0.796 |
| Centered-logit L2 distance          | 0.480 / 0.993 | 0.163 / 0.997 | 0.397 / 0.993 | 0.347 / 0.994 |
| Energy gap                          | 0.119 / 1.000 | 0.081 / 1.000 | 0.368 / 0.995 | 0.189 / 0.998 |
| Absolute energy gap                 | 0.093 / 1.000 | 0.077 / 1.000 | 0.359 / 0.995 | 0.176 / 0.998 |
| Student MSP                         | 0.789 / 0.851 | 0.898 / 0.519 | 0.759 / 0.822 | 0.815 / 0.731 |
| Student energy                      | 0.797 / 0.879 | 0.940 / 0.340 | 0.751 / 0.842 | 0.829 / 0.687 |

#### KL divergence, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.293 / 0.996 | 0.701 / 0.877 | 0.639 / 0.873 | 0.545 / 0.915 |
| Absolute max probability difference | 0.695 / 0.715 | 0.341 / 0.998 | 0.442 / 0.982 | 0.493 / 0.899 |
| KL (teacher / student) | 0.885 / 0.463 | 0.423 / 0.985 | 0.590 / 0.970 | 0.632 / 0.806 |
| Centered-logit L2 distance          | 0.629 / 0.957 | 0.165 / 0.996 | 0.376 / 0.974 | 0.390 / 0.976 |
| Energy gap                          | 0.448 / 1.000 | 0.704 / 0.991 | 0.705 / 0.941 | 0.619 / 0.977 |
| Absolute energy gap                 | 0.374 / 0.986 | 0.134 / 1.000 | 0.278 / 0.999 | 0.262 / 0.995 |
| Student MSP                         | 0.924 / 0.401 | 0.648 / 0.948 | 0.696 / 0.891 | 0.756 / 0.747 |
| Student energy                      | 0.930 / 0.401 | 0.739 / 0.961 | 0.676 / 0.901 | 0.782 / 0.754 |

</details>

### CIFAR-100 ID, ResNet-50

<details>
<summary>Constant, GAP</summary>

#### Centered-logit MSE, Clean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.476 / 0.864 | 0.472 / 0.830 | 0.633 / 0.799 | 0.527 / 0.831 |
| Absolute max probability difference | 0.524 / 0.737 | 0.528 / 0.756 | 0.367 / 0.968 | 0.473 / 0.820 |
| KL (teacher / student) | 0.708 / 0.706 | 0.706 / 0.689 | 0.453 / 0.961 | 0.622 / 0.785 |
| Centered-logit L2 distance          | 0.134 / 1.000 | 0.497 / 0.983 | 0.226 / 0.996 | 0.286 / 0.993 |
| Energy gap                          | 0.860 / 0.750 | 0.763 / 0.806 | 0.802 / 0.773 | 0.809 / 0.777 |
| Absolute energy gap                 | 0.140 / 1.000 | 0.237 / 0.999 | 0.198 / 0.996 | 0.191 / 0.998 |
| Student MSP                         | 0.910 / 0.574 | 0.866 / 0.597 | 0.754 / 0.812 | 0.843 / 0.661 |
| Student energy                      | 0.954 / 0.291 | 0.871 / 0.599 | 0.605 / 0.896 | 0.810 / 0.595 |

#### Centered-logit MSE, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.792 / 0.825 | 0.723 / 0.794 | 0.781 / 0.781 | 0.765 / 0.800 |
| Absolute max probability difference | 0.208 / 0.999 | 0.277 / 0.983 | 0.219 / 0.995 | 0.235 / 0.992 |
| KL (teacher / student) | 0.204 / 1.000 | 0.340 / 0.981 | 0.224 / 0.997 | 0.256 / 0.993 |
| Centered-logit L2 distance          | 0.058 / 1.000 | 0.305 / 0.999 | 0.276 / 0.993 | 0.213 / 0.997 |
| Energy gap                          | 0.878 / 0.677 | 0.783 / 0.769 | 0.801 / 0.775 | 0.820 / 0.740 |
| Absolute energy gap                 | 0.122 / 1.000 | 0.217 / 0.999 | 0.199 / 0.996 | 0.180 / 0.999 |
| Student MSP                         | 0.819 / 0.799 | 0.764 / 0.870 | 0.555 / 0.952 | 0.713 / 0.873 |
| Student energy                      | 0.802 / 0.807 | 0.654 / 0.969 | 0.417 / 0.977 | 0.624 / 0.917 |

#### KL divergence, Clean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.169 / 0.999 | 0.272 / 0.992 | 0.401 / 0.977 | 0.281 / 0.989 |
| Absolute max probability difference | 0.831 / 0.651 | 0.728 / 0.771 | 0.599 / 0.937 | 0.719 / 0.786 |
| KL (teacher / student) | 0.896 / 0.645 | 0.820 / 0.673 | 0.652 / 0.922 | 0.789 / 0.747 |
| Centered-logit L2 distance          | 0.196 / 1.000 | 0.702 / 0.912 | 0.252 / 0.993 | 0.383 / 0.968 |
| Energy gap                          | 0.828 / 0.806 | 0.731 / 0.835 | 0.812 / 0.757 | 0.790 / 0.799 |
| Absolute energy gap                 | 0.172 / 1.000 | 0.269 / 1.000 | 0.188 / 0.999 | 0.210 / 1.000 |
| Student MSP                         | 0.888 / 0.645 | 0.814 / 0.671 | 0.757 / 0.811 | 0.820 / 0.709 |
| Student energy                      | 0.952 / 0.312 | 0.853 / 0.590 | 0.721 / 0.837 | 0.842 / 0.580 |

#### KL divergence, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.750 / 0.826 | 0.719 / 0.795 | 0.764 / 0.781 | 0.744 / 0.801 |
| Absolute max probability difference | 0.250 / 0.989 | 0.281 / 0.992 | 0.236 / 0.991 | 0.256 / 0.991 |
| KL (teacher / student) | 0.272 / 0.961 | 0.259 / 0.993 | 0.234 / 0.990 | 0.255 / 0.981 |
| Centered-logit L2 distance          | 0.059 / 1.000 | 0.292 / 0.999 | 0.283 / 0.992 | 0.211 / 0.997 |
| Energy gap                          | 0.877 / 0.678 | 0.782 / 0.769 | 0.801 / 0.775 | 0.820 / 0.741 |
| Absolute energy gap                 | 0.123 / 1.000 | 0.218 / 0.999 | 0.199 / 0.996 | 0.180 / 0.999 |
| Student MSP                         | 0.900 / 0.540 | 0.685 / 0.892 | 0.632 / 0.925 | 0.739 / 0.786 |
| Student energy                      | 0.821 / 0.791 | 0.652 / 0.953 | 0.495 / 0.967 | 0.656 / 0.904 |

</details>

<details>
<summary>Constant, Flatten</summary>

#### Centered-logit MSE, Clean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.162 / 0.993 | 0.257 / 0.979 | 0.369 / 0.949 | 0.262 / 0.974 |
| Absolute max probability difference | 0.838 / 0.698 | 0.745 / 0.771 | 0.639 / 0.918 | 0.741 / 0.796 |
| KL (teacher / student) | 0.884 / 0.746 | 0.824 / 0.684 | 0.689 / 0.911 | 0.799 / 0.780 |
| Centered-logit L2 distance          | 0.174 / 1.000 | 0.455 / 0.978 | 0.287 / 0.993 | 0.305 / 0.990 |
| Energy gap                          | 0.769 / 0.920 | 0.670 / 0.919 | 0.807 / 0.778 | 0.749 / 0.872 |
| Absolute energy gap                 | 0.230 / 1.000 | 0.330 / 0.998 | 0.193 / 0.999 | 0.251 / 0.999 |
| Student MSP                         | 0.867 / 0.705 | 0.804 / 0.705 | 0.768 / 0.810 | 0.813 / 0.740 |
| Student energy                      | 0.936 / 0.419 | 0.855 / 0.642 | 0.715 / 0.829 | 0.835 / 0.630 |

#### Centered-logit MSE, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.742 / 0.829 | 0.700 / 0.797 | 0.762 / 0.782 | 0.735 / 0.803 |
| Absolute max probability difference | 0.258 / 0.988 | 0.300 / 0.982 | 0.238 / 0.992 | 0.265 / 0.987 |
| KL (teacher / student) | 0.251 / 1.000 | 0.339 / 0.965 | 0.234 / 0.994 | 0.274 / 0.986 |
| Centered-logit L2 distance          | 0.068 / 1.000 | 0.307 / 0.999 | 0.274 / 0.993 | 0.216 / 0.997 |
| Energy gap                          | 0.876 / 0.682 | 0.781 / 0.771 | 0.801 / 0.773 | 0.819 / 0.742 |
| Absolute energy gap                 | 0.124 / 1.000 | 0.219 / 0.999 | 0.199 / 0.997 | 0.181 / 0.999 |
| Student MSP                         | 0.860 / 0.657 | 0.739 / 0.840 | 0.616 / 0.934 | 0.738 / 0.810 |
| Student energy                      | 0.810 / 0.790 | 0.709 / 0.918 | 0.476 / 0.971 | 0.665 / 0.893 |

#### KL divergence, Clean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.668 / 0.908 | 0.419 / 0.864 | 0.645 / 0.818 | 0.578 / 0.863 |
| Absolute max probability difference | 0.753 / 0.954 | 0.787 / 0.799 | 0.770 / 0.847 | 0.770 / 0.867 |
| KL (teacher / student) | 0.779 / 0.980 | 0.814 / 0.700 | 0.802 / 0.804 | 0.798 / 0.828 |
| Centered-logit L2 distance          | 0.212 / 1.000 | 0.447 / 0.993 | 0.445 / 0.969 | 0.368 / 0.987 |
| Energy gap                          | 0.241 / 1.000 | 0.178 / 0.999 | 0.563 / 0.897 | 0.327 / 0.965 |
| Absolute energy gap                 | 0.231 / 1.000 | 0.167 / 0.999 | 0.558 / 0.897 | 0.319 / 0.965 |
| Student MSP                         | 0.804 / 0.841 | 0.812 / 0.772 | 0.756 / 0.830 | 0.791 / 0.814 |
| Student energy                      | 0.889 / 0.609 | 0.859 / 0.647 | 0.727 / 0.823 | 0.825 / 0.693 |

#### KL divergence, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.684 / 0.833 | 0.702 / 0.796 | 0.743 / 0.779 | 0.709 / 0.803 |
| Absolute max probability difference | 0.316 / 0.948 | 0.298 / 0.985 | 0.257 / 0.986 | 0.290 / 0.973 |
| KL (teacher / student) | 0.319 / 0.967 | 0.247 / 0.994 | 0.253 / 0.988 | 0.273 / 0.983 |
| Centered-logit L2 distance          | 0.071 / 1.000 | 0.284 / 0.998 | 0.284 / 0.992 | 0.213 / 0.997 |
| Energy gap                          | 0.872 / 0.703 | 0.779 / 0.781 | 0.801 / 0.775 | 0.817 / 0.753 |
| Absolute energy gap                 | 0.127 / 1.000 | 0.220 / 0.999 | 0.199 / 0.997 | 0.182 / 0.999 |
| Student MSP                         | 0.886 / 0.555 | 0.676 / 0.855 | 0.661 / 0.901 | 0.741 / 0.771 |
| Student energy                      | 0.839 / 0.744 | 0.690 / 0.920 | 0.555 / 0.956 | 0.694 / 0.873 |

</details>

<details>
<summary>Spatial, GAP</summary>

#### Centered-logit MSE, Clean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.474 / 0.891 | 0.453 / 0.854 | 0.619 / 0.810 | 0.515 / 0.852 |
| Absolute max probability difference | 0.526 / 0.895 | 0.547 / 0.767 | 0.381 / 0.976 | 0.485 / 0.879 |
| KL (teacher / student) | 0.711 / 0.880 | 0.797 / 0.614 | 0.499 / 0.956 | 0.669 / 0.817 |
| Centered-logit L2 distance          | 0.146 / 1.000 | 0.714 / 0.935 | 0.294 / 0.991 | 0.385 / 0.975 |
| Energy gap                          | 0.843 / 0.826 | 0.741 / 0.844 | 0.801 / 0.777 | 0.795 / 0.816 |
| Absolute energy gap                 | 0.157 / 1.000 | 0.259 / 0.997 | 0.199 / 0.997 | 0.205 / 0.998 |
| Student MSP                         | 0.902 / 0.550 | 0.887 / 0.509 | 0.731 / 0.834 | 0.840 / 0.631 |
| Student energy                      | 0.925 / 0.465 | 0.850 / 0.589 | 0.559 / 0.919 | 0.778 / 0.658 |

#### Centered-logit MSE, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.790 / 0.825 | 0.713 / 0.795 | 0.785 / 0.782 | 0.763 / 0.801 |
| Absolute max probability difference | 0.210 / 0.996 | 0.287 / 0.963 | 0.215 / 0.996 | 0.237 / 0.985 |
| KL (teacher / student) | 0.253 / 0.999 | 0.328 / 0.991 | 0.247 / 0.991 | 0.276 / 0.994 |
| Centered-logit L2 distance          | 0.070 / 1.000 | 0.317 / 0.998 | 0.277 / 0.992 | 0.222 / 0.997 |
| Energy gap                          | 0.877 / 0.676 | 0.782 / 0.769 | 0.801 / 0.773 | 0.820 / 0.739 |
| Absolute energy gap                 | 0.123 / 1.000 | 0.218 / 0.999 | 0.199 / 0.996 | 0.180 / 0.999 |
| Student MSP                         | 0.944 / 0.322 | 0.896 / 0.546 | 0.560 / 0.934 | 0.800 / 0.600 |
| Student energy                      | 0.976 / 0.133 | 0.896 / 0.574 | 0.442 / 0.962 | 0.771 / 0.556 |

#### KL divergence, Clean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.221 / 0.989 | 0.314 / 0.902 | 0.445 / 0.851 | 0.327 / 0.914 |
| Absolute max probability difference | 0.776 / 0.974 | 0.742 / 0.848 | 0.638 / 0.946 | 0.719 / 0.923 |
| KL (teacher / student) | 0.762 / 0.986 | 0.839 / 0.635 | 0.716 / 0.904 | 0.772 / 0.842 |
| Centered-logit L2 distance          | 0.179 / 1.000 | 0.706 / 0.953 | 0.462 / 0.969 | 0.449 / 0.974 |
| Energy gap                          | 0.618 / 1.000 | 0.534 / 0.993 | 0.724 / 0.884 | 0.625 / 0.959 |
| Absolute energy gap                 | 0.308 / 1.000 | 0.428 / 1.000 | 0.310 / 0.998 | 0.349 / 0.999 |
| Student MSP                         | 0.834 / 0.717 | 0.806 / 0.754 | 0.747 / 0.837 | 0.796 / 0.769 |
| Student energy                      | 0.909 / 0.482 | 0.846 / 0.684 | 0.692 / 0.869 | 0.816 / 0.678 |

#### KL divergence, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.744 / 0.827 | 0.705 / 0.796 | 0.757 / 0.782 | 0.736 / 0.801 |
| Absolute max probability difference | 0.256 / 0.983 | 0.295 / 0.993 | 0.243 / 0.993 | 0.264 / 0.990 |
| KL (teacher / student) | 0.341 / 0.964 | 0.310 / 0.991 | 0.306 / 0.976 | 0.319 / 0.977 |
| Centered-logit L2 distance          | 0.089 / 1.000 | 0.335 / 0.998 | 0.292 / 0.991 | 0.239 / 0.996 |
| Energy gap                          | 0.876 / 0.682 | 0.781 / 0.772 | 0.801 / 0.773 | 0.819 / 0.742 |
| Absolute energy gap                 | 0.124 / 1.000 | 0.219 / 0.999 | 0.199 / 0.996 | 0.181 / 0.999 |
| Student MSP                         | 0.906 / 0.465 | 0.761 / 0.767 | 0.647 / 0.910 | 0.771 / 0.714 |
| Student energy                      | 0.975 / 0.144 | 0.848 / 0.642 | 0.531 / 0.936 | 0.785 / 0.574 |

</details>

<details>
<summary>Spatial, Flatten</summary>

#### Centered-logit MSE, Clean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.553 / 0.865 | 0.499 / 0.832 | 0.670 / 0.791 | 0.574 / 0.829 |
| Absolute max probability difference | 0.447 / 0.887 | 0.501 / 0.774 | 0.330 / 0.984 | 0.426 / 0.882 |
| KL (teacher / student) | 0.621 / 0.852 | 0.727 / 0.660 | 0.423 / 0.966 | 0.590 / 0.826 |
| Centered-logit L2 distance          | 0.144 / 1.000 | 0.642 / 0.963 | 0.253 / 0.996 | 0.346 / 0.986 |
| Energy gap                          | 0.857 / 0.769 | 0.754 / 0.825 | 0.806 / 0.766 | 0.806 / 0.787 |
| Absolute energy gap                 | 0.143 / 1.000 | 0.246 / 0.998 | 0.194 / 0.997 | 0.194 / 0.998 |
| Student MSP                         | 0.887 / 0.638 | 0.891 / 0.503 | 0.712 / 0.850 | 0.830 / 0.664 |
| Student energy                      | 0.910 / 0.576 | 0.864 / 0.513 | 0.517 / 0.934 | 0.764 / 0.674 |

#### Centered-logit MSE, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.792 / 0.825 | 0.711 / 0.794 | 0.786 / 0.781 | 0.763 / 0.800 |
| Absolute max probability difference | 0.208 / 0.995 | 0.289 / 0.941 | 0.214 / 0.997 | 0.237 / 0.978 |
| KL (teacher / student) | 0.257 / 0.995 | 0.372 / 0.966 | 0.242 / 0.991 | 0.290 / 0.984 |
| Centered-logit L2 distance          | 0.072 / 1.000 | 0.332 / 0.998 | 0.275 / 0.993 | 0.226 / 0.997 |
| Energy gap                          | 0.877 / 0.677 | 0.782 / 0.769 | 0.801 / 0.773 | 0.820 / 0.740 |
| Absolute energy gap                 | 0.123 / 1.000 | 0.218 / 0.999 | 0.199 / 0.996 | 0.180 / 0.999 |
| Student MSP                         | 0.960 / 0.194 | 0.920 / 0.429 | 0.531 / 0.935 | 0.804 / 0.520 |
| Student energy                      | 0.974 / 0.156 | 0.896 / 0.592 | 0.416 / 0.975 | 0.762 / 0.574 |

#### KL divergence, Clean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.679 / 0.920 | 0.524 / 0.843 | 0.683 / 0.831 | 0.629 / 0.865 |
| Absolute max probability difference | 0.748 / 0.938 | 0.738 / 0.834 | 0.767 / 0.853 | 0.751 / 0.875 |
| KL (teacher / student) | 0.786 / 0.970 | 0.782 / 0.686 | 0.784 / 0.839 | 0.784 / 0.832 |
| Centered-logit L2 distance          | 0.194 / 1.000 | 0.356 / 0.997 | 0.482 / 0.960 | 0.344 / 0.986 |
| Energy gap                          | 0.189 / 1.000 | 0.179 / 1.000 | 0.390 / 0.985 | 0.253 / 0.995 |
| Absolute energy gap                 | 0.189 / 1.000 | 0.179 / 1.000 | 0.390 / 0.985 | 0.253 / 0.995 |
| Student MSP                         | 0.796 / 0.815 | 0.794 / 0.819 | 0.758 / 0.838 | 0.783 / 0.824 |
| Student energy                      | 0.883 / 0.583 | 0.853 / 0.668 | 0.726 / 0.828 | 0.821 / 0.693 |

#### KL divergence, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.609 / 0.852 | 0.596 / 0.812 | 0.689 / 0.787 | 0.631 / 0.817 |
| Absolute max probability difference | 0.391 / 0.888 | 0.404 / 0.941 | 0.311 / 0.983 | 0.369 / 0.937 |
| KL (teacher / student) | 0.541 / 0.911 | 0.483 / 0.957 | 0.418 / 0.957 | 0.481 / 0.942 |
| Centered-logit L2 distance          | 0.148 / 1.000 | 0.374 / 0.998 | 0.313 / 0.990 | 0.278 / 0.996 |
| Energy gap                          | 0.862 / 0.743 | 0.767 / 0.804 | 0.800 / 0.773 | 0.810 / 0.773 |
| Absolute energy gap                 | 0.138 / 1.000 | 0.233 / 0.999 | 0.200 / 0.996 | 0.190 / 0.998 |
| Student MSP                         | 0.943 / 0.284 | 0.827 / 0.702 | 0.693 / 0.887 | 0.821 / 0.624 |
| Student energy                      | 0.983 / 0.088 | 0.885 / 0.554 | 0.579 / 0.922 | 0.816 / 0.521 |

</details>

<details>
<summary>Channel, GAP</summary>

#### Centered-logit MSE, Clean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.318 / 0.936 | 0.329 / 0.888 | 0.365 / 0.890 | 0.338 / 0.905 |
| Absolute max probability difference | 0.722 / 0.872 | 0.759 / 0.785 | 0.717 / 0.889 | 0.732 / 0.849 |
| KL (teacher / student) | 0.787 / 0.842 | 0.834 / 0.627 | 0.759 / 0.880 | 0.793 / 0.783 |
| Centered-logit L2 distance          | 0.435 / 1.000 | 0.475 / 0.995 | 0.470 / 0.978 | 0.460 / 0.991 |
| Energy gap                          | 0.639 / 0.999 | 0.448 / 0.995 | 0.605 / 0.966 | 0.564 / 0.987 |
| Absolute energy gap                 | 0.276 / 1.000 | 0.504 / 0.996 | 0.356 / 0.995 | 0.379 / 0.997 |
| Student MSP                         | 0.801 / 0.807 | 0.800 / 0.757 | 0.788 / 0.786 | 0.797 / 0.784 |
| Student energy                      | 0.846 / 0.733 | 0.850 / 0.675 | 0.761 / 0.809 | 0.819 / 0.739 |

#### Centered-logit MSE, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.516 / 0.994 | 0.791 / 0.851 | 0.545 / 0.968 | 0.617 / 0.938 |
| Absolute max probability difference | 0.483 / 0.995 | 0.216 / 1.000 | 0.456 / 0.972 | 0.385 / 0.989 |
| KL (teacher / student) | 0.510 / 0.979 | 0.281 / 0.987 | 0.587 / 0.945 | 0.459 / 0.971 |
| Centered-logit L2 distance          | 0.253 / 1.000 | 0.207 / 0.999 | 0.521 / 0.974 | 0.327 / 0.991 |
| Energy gap                          | 0.841 / 0.748 | 0.823 / 0.684 | 0.755 / 0.809 | 0.806 / 0.747 |
| Absolute energy gap                 | 0.159 / 1.000 | 0.177 / 1.000 | 0.245 / 0.994 | 0.193 / 0.998 |
| Student MSP                         | 0.735 / 0.975 | 0.331 / 0.999 | 0.709 / 0.900 | 0.592 / 0.958 |
| Student energy                      | 0.731 / 0.971 | 0.350 / 0.999 | 0.694 / 0.905 | 0.592 / 0.958 |

#### KL divergence, Clean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.491 / 0.922 | 0.416 / 0.858 | 0.648 / 0.827 | 0.518 / 0.869 |
| Absolute max probability difference | 0.753 / 0.915 | 0.854 / 0.707 | 0.756 / 0.862 | 0.788 / 0.828 |
| KL (teacher / student) | 0.808 / 0.866 | 0.911 / 0.422 | 0.787 / 0.837 | 0.835 / 0.708 |
| Centered-logit L2 distance          | 0.548 / 1.000 | 0.344 / 0.986 | 0.439 / 0.988 | 0.444 / 0.991 |
| Energy gap                          | 0.237 / 1.000 | 0.148 / 1.000 | 0.421 / 0.959 | 0.269 / 0.986 |
| Absolute energy gap                 | 0.233 / 1.000 | 0.156 / 1.000 | 0.419 / 0.959 | 0.269 / 0.986 |
| Student MSP                         | 0.828 / 0.777 | 0.863 / 0.696 | 0.740 / 0.837 | 0.811 / 0.770 |
| Student energy                      | 0.865 / 0.780 | 0.912 / 0.500 | 0.716 / 0.850 | 0.831 / 0.710 |

#### KL divergence, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.487 / 0.948 | 0.648 / 0.875 | 0.498 / 0.937 | 0.545 / 0.920 |
| Absolute max probability difference | 0.520 / 0.990 | 0.370 / 0.999 | 0.514 / 0.963 | 0.468 / 0.984 |
| KL (teacher / student) | 0.618 / 0.961 | 0.551 / 0.929 | 0.655 / 0.923 | 0.608 / 0.938 |
| Centered-logit L2 distance          | 0.406 / 0.999 | 0.656 / 0.914 | 0.364 / 0.983 | 0.476 / 0.966 |
| Energy gap                          | 0.803 / 0.866 | 0.760 / 0.865 | 0.681 / 0.943 | 0.748 / 0.892 |
| Absolute energy gap                 | 0.161 / 1.000 | 0.205 / 1.000 | 0.286 / 0.993 | 0.217 / 0.997 |
| Student MSP                         | 0.699 / 0.978 | 0.515 / 0.993 | 0.723 / 0.872 | 0.646 / 0.947 |
| Student energy                      | 0.635 / 0.993 | 0.496 / 0.995 | 0.712 / 0.878 | 0.614 / 0.955 |

</details>

<details>
<summary>Channel, Flatten</summary>

#### Centered-logit MSE, Clean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.377 / 0.923 | 0.364 / 0.876 | 0.361 / 0.897 | 0.367 / 0.898 |
| Absolute max probability difference | 0.669 / 0.897 | 0.731 / 0.814 | 0.713 / 0.893 | 0.704 / 0.868 |
| KL (teacher / student) | 0.749 / 0.825 | 0.811 / 0.661 | 0.754 / 0.883 | 0.771 / 0.790 |
| Centered-logit L2 distance          | 0.447 / 1.000 | 0.479 / 0.994 | 0.476 / 0.975 | 0.467 / 0.989 |
| Energy gap                          | 0.697 / 1.000 | 0.483 / 0.997 | 0.608 / 0.968 | 0.596 / 0.988 |
| Absolute energy gap                 | 0.238 / 1.000 | 0.466 / 0.998 | 0.349 / 0.995 | 0.351 / 0.998 |
| Student MSP                         | 0.777 / 0.820 | 0.782 / 0.779 | 0.788 / 0.782 | 0.782 / 0.794 |
| Student energy                      | 0.812 / 0.776 | 0.827 / 0.713 | 0.756 / 0.810 | 0.798 / 0.767 |

#### Centered-logit MSE, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.518 / 0.994 | 0.792 / 0.850 | 0.544 / 0.970 | 0.618 / 0.938 |
| Absolute max probability difference | 0.481 / 0.996 | 0.216 / 1.000 | 0.457 / 0.975 | 0.385 / 0.990 |
| KL (teacher / student) | 0.513 / 0.962 | 0.285 / 0.986 | 0.586 / 0.948 | 0.462 / 0.965 |
| Centered-logit L2 distance          | 0.260 / 1.000 | 0.224 / 0.999 | 0.516 / 0.976 | 0.333 / 0.991 |
| Energy gap                          | 0.842 / 0.741 | 0.824 / 0.682 | 0.754 / 0.816 | 0.806 / 0.746 |
| Absolute energy gap                 | 0.158 / 1.000 | 0.176 / 1.000 | 0.246 / 0.994 | 0.193 / 0.998 |
| Student MSP                         | 0.726 / 0.984 | 0.331 / 0.999 | 0.708 / 0.898 | 0.588 / 0.960 |
| Student energy                      | 0.704 / 0.985 | 0.353 / 0.998 | 0.692 / 0.904 | 0.583 / 0.962 |

#### KL divergence, Clean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.315 / 0.918 | 0.399 / 0.843 | 0.620 / 0.836 | 0.445 / 0.865 |
| Absolute max probability difference | 0.803 / 0.853 | 0.856 / 0.686 | 0.744 / 0.882 | 0.801 / 0.807 |
| KL (teacher / student) | 0.853 / 0.753 | 0.922 / 0.379 | 0.776 / 0.870 | 0.850 / 0.668 |
| Centered-logit L2 distance          | 0.547 / 0.999 | 0.518 / 0.966 | 0.442 / 0.990 | 0.502 / 0.985 |
| Energy gap                          | 0.260 / 1.000 | 0.197 / 0.999 | 0.447 / 0.954 | 0.302 / 0.984 |
| Absolute energy gap                 | 0.245 / 1.000 | 0.240 / 0.999 | 0.441 / 0.954 | 0.309 / 0.984 |
| Student MSP                         | 0.864 / 0.725 | 0.857 / 0.732 | 0.744 / 0.831 | 0.822 / 0.763 |
| Student energy                      | 0.882 / 0.700 | 0.898 / 0.591 | 0.722 / 0.836 | 0.834 / 0.709 |

#### KL divergence, Clipped, 50-draw mean

| Score                               |         MNIST |          SVHN |      CIFAR-10 |         Macro |
| ----------------------------------- | ------------: | ------------: | ------------: | ------------: |
| Max probability difference          | 0.449 / 0.973 | 0.644 / 0.891 | 0.510 / 0.943 | 0.534 / 0.935 |
| Absolute max probability difference | 0.552 / 0.983 | 0.369 / 0.998 | 0.499 / 0.970 | 0.473 / 0.984 |
| KL (teacher / student) | 0.701 / 0.797 | 0.585 / 0.907 | 0.634 / 0.940 | 0.640 / 0.881 |
| Centered-logit L2 distance          | 0.349 / 0.999 | 0.651 / 0.885 | 0.346 / 0.986 | 0.449 / 0.957 |
| Energy gap                          | 0.789 / 0.891 | 0.755 / 0.860 | 0.696 / 0.931 | 0.747 / 0.894 |
| Absolute energy gap                 | 0.177 / 1.000 | 0.216 / 1.000 | 0.278 / 0.993 | 0.224 / 0.998 |
| Student MSP                         | 0.757 / 0.948 | 0.517 / 0.989 | 0.713 / 0.876 | 0.662 / 0.938 |
| Student energy                      | 0.698 / 0.961 | 0.514 / 0.993 | 0.697 / 0.885 | 0.636 / 0.946 |

</details>

## Artifacts

Numeric metric exports:

```text
reports/outputs/json/clipping_layer34_metrics.json
reports/outputs/json/clipping_layer34_teacher_metrics.json
```

Probability artifacts remain in the corresponding cluster run directories:

```text
<run_dir>/probabilities/unperturbed/
<run_dir>/probabilities/perturbed/
```

SLURM jobs:

- training and clean inference: `406170`, `406172`, `406174`, `406324`;
- 50-draw clipped inference: `406171`, `406173`, `406175`, `406325`;
- student OOD metric export: `406373`;
- teacher baseline metric export: `406372`.
