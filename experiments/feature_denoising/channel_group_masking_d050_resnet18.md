# Channel-Group Masking at Distance 0.5

ResNet-18, 10 masking draws. Values are `ROC-AUC / FPR@95`; macro is the
unweighted mean over the three OOD datasets.

## Trained student: residual CNN

Student type: learned `feature_residual_denoiser`, trained for 50 epochs.

### Raw reconstruction score

| ID | Layer | MNIST | SVHN | Opposite CIFAR | Macro |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | layer1 | 0.549 / 0.860 | 0.090 / 1.000 | 0.554 / 0.896 | 0.397 / 0.919 |
| CIFAR-10 | layer2 | 0.415 / 0.986 | 0.131 / 0.992 | 0.501 / 0.933 | 0.349 / 0.970 |
| CIFAR-10 | layer3 | 0.488 / 0.941 | 0.175 / 0.978 | 0.404 / 0.977 | 0.355 / 0.965 |
| CIFAR-10 | layer4 | 0.550 / 0.968 | 0.509 / 0.945 | 0.728 / 0.843 | 0.596 / 0.919 |
| CIFAR-100 | layer1 | 0.630 / 0.987 | 0.105 / 1.000 | 0.465 / 0.973 | 0.400 / 0.987 |
| CIFAR-100 | layer2 | 0.547 / 0.904 | 0.164 / 0.991 | 0.482 / 0.970 | 0.398 / 0.955 |
| CIFAR-100 | layer3 | 0.181 / 0.998 | 0.221 / 0.972 | 0.500 / 0.972 | 0.301 / 0.981 |
| CIFAR-100 | layer4 | 0.616 / 0.929 | 0.646 / 0.911 | 0.615 / 0.932 | 0.626 / 0.924 |

### Absolute improvement

| ID | Layer | MNIST | SVHN | Opposite CIFAR | Macro |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | layer1 | 0.427 / 0.970 | 0.254 / 0.980 | 0.464 / 0.952 | 0.382 / 0.967 |
| CIFAR-10 | layer2 | 0.604 / 0.870 | 0.725 / 0.663 | 0.529 / 0.922 | 0.619 / 0.818 |
| CIFAR-10 | layer3 | 0.619 / 0.853 | 0.705 / 0.717 | 0.641 / 0.905 | 0.655 / 0.825 |
| CIFAR-10 | layer4 | 0.770 / 0.912 | 0.721 / 0.946 | 0.606 / 0.969 | 0.699 / 0.942 |
| CIFAR-100 | layer1 | 0.719 / 0.792 | 0.572 / 0.823 | 0.516 / 0.955 | 0.602 / 0.856 |
| CIFAR-100 | layer2 | 0.715 / 0.703 | 0.617 / 0.795 | 0.508 / 0.955 | 0.613 / 0.818 |
| CIFAR-100 | layer3 | 0.732 / 0.767 | 0.679 / 0.804 | 0.485 / 0.969 | 0.632 / 0.847 |
| CIFAR-100 | layer4 | 0.502 / 0.947 | 0.603 / 0.897 | 0.562 / 0.931 | 0.555 / 0.925 |

### Relative improvement

| ID | Layer | MNIST | SVHN | Opposite CIFAR | Macro |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | layer1 | 0.490 / 0.862 | 0.079 / 1.000 | 0.525 / 0.922 | 0.364 / 0.928 |
| CIFAR-10 | layer2 | 0.546 / 0.922 | 0.360 / 0.920 | 0.537 / 0.916 | 0.481 / 0.919 |
| CIFAR-10 | layer3 | 0.648 / 0.833 | 0.566 / 0.818 | 0.621 / 0.912 | 0.612 / 0.854 |
| CIFAR-10 | layer4 | 0.798 / 0.874 | 0.727 / 0.925 | 0.729 / 0.910 | 0.752 / 0.903 |
| CIFAR-100 | layer1 | 0.774 / 0.689 | 0.180 / 0.992 | 0.470 / 0.971 | 0.475 / 0.884 |
| CIFAR-100 | layer2 | 0.753 / 0.661 | 0.296 / 0.970 | 0.490 / 0.967 | 0.513 / 0.866 |
| CIFAR-100 | layer3 | 0.597 / 0.835 | 0.523 / 0.870 | 0.473 / 0.969 | 0.531 / 0.891 |
| CIFAR-100 | layer4 | 0.533 / 0.974 | 0.637 / 0.953 | 0.590 / 0.968 | 0.587 / 0.965 |

## k-NN student: exact nearest neighbors — CIFAR-10 ID

Student type: inference-only exact k-NN (`k=10`), without learned parameters.

### Raw reconstruction score

| Layer | MNIST | SVHN | CIFAR-100 | Macro |
| --- | ---: | ---: | ---: | ---: |
| layer3 | 0.488 / 0.956 | 0.242 / 0.961 | 0.534 / 0.932 | 0.421 / 0.950 |
| layer4 | 0.786 / 0.804 | 0.754 / 0.847 | 0.831 / 0.683 | 0.790 / 0.778 |

### Absolute improvement

| Layer | MNIST | SVHN | CIFAR-100 | Macro |
| --- | ---: | ---: | ---: | ---: |
| layer3 | 0.623 / 0.889 | 0.711 / 0.753 | 0.745 / 0.783 | 0.693 / 0.808 |
| layer4 | 0.801 / 0.829 | 0.748 / 0.904 | 0.649 / 0.938 | 0.732 / 0.890 |

### Relative improvement

| Layer | MNIST | SVHN | CIFAR-100 | Macro |
| --- | ---: | ---: | ---: | ---: |
| layer3 | 0.638 / 0.909 | 0.619 / 0.814 | 0.766 / 0.784 | 0.674 / 0.836 |
| layer4 | 0.877 / 0.613 | 0.833 / 0.730 | 0.836 / 0.706 | 0.849 / 0.683 |

## Trained student: stratified within-cluster residual CNN — CIFAR-10 layer4

Student type: learned `feature_residual_denoiser`, trained for 50 epochs. The
mask samples 25% of channels from every distance-0.5 cluster containing at
least four channels.

| Score | MNIST | SVHN | CIFAR-100 | Macro |
| --- | ---: | ---: | ---: | ---: |
| Raw reconstruction | 0.585 / 0.995 | 0.581 / 0.956 | 0.778 / 0.821 | 0.648 / 0.924 |
| Absolute improvement | 0.984 / 0.088 | 0.907 / 0.413 | 0.772 / 0.716 | 0.888 / 0.405 |
| Relative improvement | 0.940 / 0.360 | 0.863 / 0.639 | 0.866 / 0.608 | 0.890 / 0.535 |
