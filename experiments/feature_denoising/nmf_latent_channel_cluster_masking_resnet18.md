# NMF-Latent Channel-Cluster Masking

ResNet-18, 50 epochs, 10 masking draws. Channels are clustered by average
linkage over cosine distances between NMF `P`-matrix columns. Cuts are `0.7`
for layer2 and `0.6` for layers 3–4. Values are `ROC-AUC / FPR@95`; macro is
the unweighted mean over the three OOD datasets.

## Raw reconstruction score

| Mask | ID | Layer | MNIST | SVHN | Opposite CIFAR | Macro |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| One cluster | CIFAR-10 | layer2 | 0.349 / 0.996 | 0.076 / 0.999 | 0.505 / 0.923 | 0.310 / 0.973 |
| One cluster | CIFAR-10 | layer3 | 0.429 / 0.983 | 0.120 / 0.987 | 0.374 / 0.981 | 0.308 / 0.983 |
| One cluster | CIFAR-10 | layer4 | 0.650 / 0.947 | 0.577 / 0.930 | 0.763 / 0.797 | 0.664 / 0.891 |
| One cluster | CIFAR-100 | layer2 | 0.269 / 1.000 | 0.104 / 0.999 | 0.481 / 0.976 | 0.285 / 0.992 |
| One cluster | CIFAR-100 | layer3 | 0.119 / 1.000 | 0.149 / 0.995 | 0.489 / 0.969 | 0.253 / 0.988 |
| One cluster | CIFAR-100 | layer4 | 0.682 / 0.914 | 0.638 / 0.951 | 0.646 / 0.927 | 0.656 / 0.931 |
| 25% per cluster | CIFAR-10 | layer2 | 0.468 / 1.000 | 0.093 / 0.999 | 0.513 / 0.907 | 0.358 / 0.969 |
| 25% per cluster | CIFAR-10 | layer3 | 0.424 / 0.995 | 0.061 / 0.999 | 0.336 / 0.981 | 0.274 / 0.992 |
| 25% per cluster | CIFAR-10 | layer4 | 0.591 / 0.993 | 0.624 / 0.928 | 0.765 / 0.844 | 0.660 / 0.921 |
| 25% per cluster | CIFAR-100 | layer2 | 0.421 / 1.000 | 0.106 / 1.000 | 0.464 / 0.981 | 0.330 / 0.994 |
| 25% per cluster | CIFAR-100 | layer3 | 0.037 / 1.000 | 0.115 / 0.999 | 0.480 / 0.979 | 0.211 / 0.993 |
| 25% per cluster | CIFAR-100 | layer4 | 0.759 / 0.866 | 0.646 / 0.927 | 0.634 / 0.919 | 0.680 / 0.904 |

## Absolute improvement

| Mask | ID | Layer | MNIST | SVHN | Opposite CIFAR | Macro |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| One cluster | CIFAR-10 | layer2 | 0.656 / 0.833 | 0.895 / 0.356 | 0.529 / 0.915 | 0.693 / 0.702 |
| One cluster | CIFAR-10 | layer3 | 0.767 / 0.752 | 0.865 / 0.495 | 0.745 / 0.820 | 0.793 / 0.689 |
| One cluster | CIFAR-10 | layer4 | 0.829 / 0.882 | 0.746 / 0.912 | 0.668 / 0.937 | 0.748 / 0.910 |
| One cluster | CIFAR-100 | layer2 | 0.809 / 0.678 | 0.819 / 0.607 | 0.533 / 0.951 | 0.720 / 0.745 |
| One cluster | CIFAR-100 | layer3 | 0.756 / 0.558 | 0.803 / 0.578 | 0.519 / 0.960 | 0.693 / 0.699 |
| One cluster | CIFAR-100 | layer4 | 0.583 / 0.977 | 0.790 / 0.756 | 0.652 / 0.925 | 0.675 / 0.886 |
| 25% per cluster | CIFAR-10 | layer2 | 0.702 / 0.832 | 0.961 / 0.159 | 0.547 / 0.905 | 0.736 / 0.632 |
| 25% per cluster | CIFAR-10 | layer3 | 0.863 / 0.826 | 0.978 / 0.109 | 0.842 / 0.655 | 0.895 / 0.530 |
| 25% per cluster | CIFAR-10 | layer4 | 0.982 / 0.102 | 0.900 / 0.437 | 0.777 / 0.718 | 0.886 / 0.419 |
| 25% per cluster | CIFAR-100 | layer2 | 0.898 / 0.584 | 0.925 / 0.366 | 0.554 / 0.953 | 0.792 / 0.634 |
| 25% per cluster | CIFAR-100 | layer3 | 0.994 / 0.020 | 0.936 / 0.263 | 0.519 / 0.962 | 0.816 / 0.415 |
| 25% per cluster | CIFAR-100 | layer4 | 0.625 / 0.979 | 0.806 / 0.748 | 0.769 / 0.828 | 0.733 / 0.852 |

## Relative improvement

| Mask | ID | Layer | MNIST | SVHN | Opposite CIFAR | Macro |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| One cluster | CIFAR-10 | layer2 | 0.557 / 0.917 | 0.493 / 0.886 | 0.538 / 0.918 | 0.529 / 0.907 |
| One cluster | CIFAR-10 | layer3 | 0.792 / 0.703 | 0.674 / 0.765 | 0.729 / 0.839 | 0.732 / 0.769 |
| One cluster | CIFAR-10 | layer4 | 0.852 / 0.772 | 0.756 / 0.864 | 0.786 / 0.864 | 0.798 / 0.833 |
| One cluster | CIFAR-100 | layer2 | 0.670 / 0.859 | 0.281 / 0.985 | 0.517 / 0.945 | 0.489 / 0.930 |
| One cluster | CIFAR-100 | layer3 | 0.573 / 0.754 | 0.562 / 0.821 | 0.516 / 0.965 | 0.550 / 0.847 |
| One cluster | CIFAR-100 | layer4 | 0.676 / 0.948 | 0.843 / 0.698 | 0.715 / 0.878 | 0.745 / 0.841 |
| 25% per cluster | CIFAR-10 | layer2 | 0.757 / 0.957 | 0.349 / 0.960 | 0.569 / 0.879 | 0.558 / 0.932 |
| 25% per cluster | CIFAR-10 | layer3 | 0.894 / 0.653 | 0.687 / 0.856 | 0.817 / 0.739 | 0.799 / 0.749 |
| 25% per cluster | CIFAR-10 | layer4 | 0.933 / 0.389 | 0.872 / 0.591 | 0.860 / 0.623 | 0.889 / 0.534 |
| 25% per cluster | CIFAR-100 | layer2 | 0.844 / 0.802 | 0.210 / 0.997 | 0.490 / 0.959 | 0.515 / 0.919 |
| 25% per cluster | CIFAR-100 | layer3 | 0.771 / 0.736 | 0.493 / 0.924 | 0.495 / 0.975 | 0.587 / 0.878 |
| 25% per cluster | CIFAR-100 | layer4 | 0.736 / 0.917 | 0.815 / 0.799 | 0.772 / 0.818 | 0.774 / 0.845 |

## Takeaway

The 25%-per-cluster variant is clearly stronger on improvement scores. Its
best macro absolute-improvement results are CIFAR-10 layer3 (`0.895 / 0.530`)
and CIFAR-100 layer3 (`0.816 / 0.415`); CIFAR-10 layer4 gives the lowest macro
FPR@95 (`0.419`) at nearly the same ROC-AUC (`0.886`). Raw reconstruction is
weak except at layer4, and opposite-CIFAR detection remains the main weakness,
especially for CIFAR-100 ID.
