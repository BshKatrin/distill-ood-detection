# Uniform Channel-Masking Layer3 + Layer4 Composition

Status: completed.

## Objective

Compose the `layer3` and `layer4` absolute-improvement scores from Feature Denoising residual CNN students trained with uniform 20% channel masking. ResNet-18 and ResNet-50 teachers are evaluated independently.

For each layer, the per-sample absolute-improvement score is:

```text
improvement = identity_error - raw_reconstruction_error
```

Each score is standardized using its own ID test-split population mean and standard deviation. The same statistics are then reused for all OOD datasets. Standardized layer scores are composed using the HEAT operator from [Lafon et al. (2023)](https://arxiv.org/pdf/2305.16966):

```text
combined_beta = log(exp(beta * z_layer3) + exp(beta * z_layer4)) / beta
combined_0    = (z_layer3 + z_layer4) / 2
```

The nonzero implementation uses a numerically stable log-sum-exp. Higher scores remain more ID-like.

## Setup

- Teachers: ResNet-18 and ResNet-50 trained on the corresponding ID dataset.
- Students: independently trained layer3 and layer4 residual CNN denoisers.
- Corruption: each channel is hidden independently with probability 0.2.
- Score: absolute improvement (not relative improvement and not an absolute-value transform).
- Betas: `-100, -10, -1, -0.1, 0, 0.1, 1, 10, 100`.
- Metrics: ROC-AUC and FPR@95; macro values average the three OOD datasets.

## ResNet-18

### CIFAR-10 ID

ID test standardization statistics:

| Layer | Mean | Population std |
|---|---:|---:|
| layer3 | 0.0049044755 | 0.0026866352 |
| layer4 | 0.13126341 | 0.028887875 |

OOD metrics (`ROC-AUC / FPR@95`):

| Score | CIFAR-100 | MNIST | SVHN | Far macro | Overall macro |
|---|---:|---:|---:|---:|---:|
| layer3 | 0.841 / 0.662 | 0.846 / 0.835 | 0.962 / 0.217 | 0.904 / 0.526 | 0.883 / 0.571 |
| layer4 | 0.773 / 0.740 | 0.979 / 0.115 | 0.893 / 0.458 | 0.936 / 0.287 | 0.882 / 0.438 |
| beta=-100 | 0.830 / 0.740 | 0.978 / 0.115 | 0.934 / 0.457 | 0.956 / 0.286 | 0.914 / 0.437 |
| beta=-10 | 0.833 / 0.739 | 0.978 / 0.115 | 0.936 / 0.451 | 0.957 / 0.283 | 0.916 / 0.435 |
| beta=-1 | 0.861 / 0.624 | 0.986 / 0.075 | 0.955 / 0.278 | 0.971 / 0.176 | 0.934 / 0.325 |
| beta=-0.1 | 0.867 / 0.574 | 0.988 / 0.061 | 0.960 / 0.228 | 0.974 / 0.144 | 0.938 / 0.288 |
| beta=0 | 0.866 / 0.574 | 0.988 / 0.061 | 0.960 / 0.226 | 0.974 / 0.144 | 0.938 / 0.287 |
| beta=0.1 | 0.866 / 0.573 | 0.987 / 0.061 | 0.960 / 0.225 | 0.974 / 0.143 | 0.938 / 0.286 |
| beta=1 | 0.862 / 0.568 | 0.984 / 0.074 | 0.960 / 0.213 | 0.972 / 0.144 | 0.935 / 0.285 |
| beta=10 | 0.849 / 0.576 | 0.956 / 0.336 | 0.955 / 0.209 | 0.955 / 0.272 | 0.920 / 0.374 |
| beta=100 | 0.848 / 0.585 | 0.953 / 0.373 | 0.954 / 0.215 | 0.953 / 0.294 | 0.918 / 0.391 |

### CIFAR-100 ID

ID test standardization statistics:

| Layer | Mean | Population std |
|---|---:|---:|
| layer3 | 0.0026622967 | 0.00052865263 |
| layer4 | 0.18989354 | 0.088189017 |

OOD metrics (`ROC-AUC / FPR@95`):

| Score | CIFAR-10 | MNIST | SVHN | Far macro | Overall macro |
|---|---:|---:|---:|---:|---:|
| layer3 | 0.504 / 0.967 | 0.994 / 0.022 | 0.936 / 0.283 | 0.965 / 0.153 | 0.811 / 0.424 |
| layer4 | 0.774 / 0.825 | 0.621 / 0.981 | 0.806 / 0.760 | 0.713 / 0.871 | 0.734 / 0.855 |
| beta=-100 | 0.690 / 0.945 | 0.993 / 0.025 | 0.946 / 0.273 | 0.969 / 0.149 | 0.876 / 0.414 |
| beta=-10 | 0.690 / 0.945 | 0.993 / 0.026 | 0.947 / 0.269 | 0.970 / 0.147 | 0.877 / 0.413 |
| beta=-1 | 0.693 / 0.926 | 0.993 / 0.025 | 0.965 / 0.171 | 0.979 / 0.098 | 0.884 / 0.374 |
| beta=-0.1 | 0.684 / 0.896 | 0.987 / 0.064 | 0.969 / 0.152 | 0.978 / 0.108 | 0.880 / 0.371 |
| beta=0 | 0.682 / 0.893 | 0.984 / 0.081 | 0.969 / 0.153 | 0.976 / 0.117 | 0.878 / 0.375 |
| beta=0.1 | 0.680 / 0.890 | 0.979 / 0.101 | 0.968 / 0.157 | 0.974 / 0.129 | 0.876 / 0.383 |
| beta=1 | 0.666 / 0.879 | 0.921 / 0.347 | 0.954 / 0.218 | 0.937 / 0.283 | 0.847 / 0.482 |
| beta=10 | 0.644 / 0.873 | 0.836 / 0.688 | 0.923 / 0.352 | 0.880 / 0.520 | 0.801 / 0.638 |
| beta=100 | 0.643 / 0.875 | 0.833 / 0.704 | 0.922 / 0.363 | 0.877 / 0.534 | 0.799 / 0.647 |

## ResNet-50

### CIFAR-10 ID

ID test standardization statistics:

| Layer | Mean | Population std |
|---|---:|---:|
| layer3 | 0.0023666512 | 0.00055675735 |
| layer4 | 0.043133581 | 0.0079227999 |

OOD metrics (`ROC-AUC / FPR@95`):

| Score | CIFAR-100 | MNIST | SVHN | Far macro | Overall macro |
|---|---:|---:|---:|---:|---:|
| layer3 | 0.717 / 0.788 | 0.843 / 0.742 | 0.911 / 0.404 | 0.877 / 0.573 | 0.824 / 0.645 |
| layer4 | 0.622 / 0.871 | 0.909 / 0.399 | 0.868 / 0.511 | 0.888 / 0.455 | 0.799 / 0.594 |
| beta=-100 | 0.684 / 0.840 | 0.915 / 0.433 | 0.901 / 0.445 | 0.908 / 0.439 | 0.833 / 0.573 |
| beta=-10 | 0.685 / 0.839 | 0.916 / 0.431 | 0.903 / 0.440 | 0.909 / 0.435 | 0.835 / 0.570 |
| beta=-1 | 0.702 / 0.809 | 0.938 / 0.352 | 0.925 / 0.347 | 0.931 / 0.350 | 0.855 / 0.503 |
| beta=-0.1 | 0.709 / 0.792 | 0.944 / 0.318 | 0.933 / 0.313 | 0.939 / 0.315 | 0.862 / 0.474 |
| beta=0 | 0.709 / 0.790 | 0.945 / 0.316 | 0.933 / 0.310 | 0.939 / 0.313 | 0.862 / 0.472 |
| beta=0.1 | 0.709 / 0.790 | 0.945 / 0.314 | 0.933 / 0.307 | 0.939 / 0.310 | 0.862 / 0.470 |
| beta=1 | 0.705 / 0.789 | 0.940 / 0.339 | 0.933 / 0.302 | 0.937 / 0.320 | 0.859 / 0.476 |
| beta=10 | 0.697 / 0.790 | 0.924 / 0.430 | 0.927 / 0.319 | 0.926 / 0.374 | 0.849 / 0.513 |
| beta=100 | 0.696 / 0.791 | 0.923 / 0.434 | 0.926 / 0.321 | 0.925 / 0.377 | 0.848 / 0.515 |

### CIFAR-100 ID

ID test standardization statistics:

| Layer | Mean | Population std |
|---|---:|---:|
| layer3 | 0.0010497776 | 0.00015987931 |
| layer4 | 0.068250219 | 0.016494391 |

OOD metrics (`ROC-AUC / FPR@95`):

| Score | CIFAR-10 | MNIST | SVHN | Far macro | Overall macro |
|---|---:|---:|---:|---:|---:|
| layer3 | 0.522 / 0.958 | 0.991 / 0.038 | 0.909 / 0.392 | 0.950 / 0.215 | 0.807 / 0.463 |
| layer4 | 0.712 / 0.842 | 0.943 / 0.301 | 0.747 / 0.837 | 0.845 / 0.569 | 0.800 / 0.660 |
| beta=-100 | 0.652 / 0.911 | 0.993 / 0.016 | 0.912 / 0.421 | 0.953 / 0.218 | 0.852 / 0.449 |
| beta=-10 | 0.652 / 0.910 | 0.993 / 0.013 | 0.913 / 0.419 | 0.953 / 0.216 | 0.853 / 0.448 |
| beta=-1 | 0.659 / 0.908 | 0.997 / 0.003 | 0.929 / 0.360 | 0.963 / 0.181 | 0.862 / 0.424 |
| beta=-0.1 | 0.660 / 0.895 | 0.999 / 0.003 | 0.932 / 0.345 | 0.965 / 0.174 | 0.863 / 0.414 |
| beta=0 | 0.659 / 0.892 | 0.999 / 0.004 | 0.930 / 0.351 | 0.964 / 0.177 | 0.863 / 0.415 |
| beta=0.1 | 0.659 / 0.892 | 0.998 / 0.004 | 0.927 / 0.365 | 0.963 / 0.184 | 0.861 / 0.420 |
| beta=1 | 0.648 / 0.888 | 0.995 / 0.022 | 0.895 / 0.451 | 0.945 / 0.237 | 0.846 / 0.454 |
| beta=10 | 0.631 / 0.890 | 0.986 / 0.073 | 0.860 / 0.553 | 0.923 / 0.313 | 0.826 / 0.506 |
| beta=100 | 0.631 / 0.889 | 0.986 / 0.074 | 0.859 / 0.553 | 0.922 / 0.314 | 0.825 / 0.506 |

## Results Summary

- **ResNet-18, CIFAR-10 ID**: best macro ROC-AUC uses `beta=-0.1` (0.938); best macro FPR@95 uses `beta=1` (0.285). Best Near-OOD ROC-AUC is at `beta=-0.1`, while best Far-OOD macro ROC-AUC is at `beta=0`.
- **ResNet-18, CIFAR-100 ID**: best macro ROC-AUC uses `beta=-1` (0.884); best macro FPR@95 uses `beta=-0.1` (0.371). Best Near-OOD ROC-AUC is at `beta=-1`, while best Far-OOD macro ROC-AUC is at `beta=-1`.
- **ResNet-50, CIFAR-10 ID**: best macro ROC-AUC uses `beta=0.1` (0.862); best macro FPR@95 uses `beta=0.1` (0.470). Best Near-OOD ROC-AUC is at `beta=-0.1`, while best Far-OOD macro ROC-AUC is at `beta=0.1`.
- **ResNet-50, CIFAR-100 ID**: best macro ROC-AUC uses `beta=-0.1` (0.863); best macro FPR@95 uses `beta=-0.1` (0.414). Best Near-OOD ROC-AUC is at `beta=-0.1`, while best Far-OOD macro ROC-AUC is at `beta=-0.1`.

Complete machine-readable metrics and standardization statistics are stored in `reports/outputs/json/feature_denoising_uniform_channel_layer3_layer4_composition.json`.
