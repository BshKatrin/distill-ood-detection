# NMF-Latent Channel-Group Stratified Masking: ResNet-50

ResNet-50 teacher, residual CNN student with hidden width 64, 50 epochs, and
10 masking draws. Channels are clustered by average linkage over cosine
distances between NMF `P`-matrix columns. Both ID datasets use cuts of `0.6`
for layers 2–3 and `0.55` for layer 4. Each draw masks 25% of the channels in
every eligible cluster; clusters smaller than four channels are excluded.

Training and inference jobs: `410542` (CIFAR-10) and `410543` (CIFAR-100).
Metric export job: `410551`.

Values are `ROC-AUC / FPR@95`; macro is the unweighted mean over MNIST, SVHN,
and the opposite CIFAR dataset.

## Raw reconstruction score

| ID | Layer | MNIST | SVHN | Opposite CIFAR | Macro |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | layer2 | 0.471 / 1.000 | 0.073 / 0.999 | 0.528 / 0.902 | 0.357 / 0.967 |
| CIFAR-10 | layer3 | 0.451 / 0.998 | 0.092 / 0.998 | 0.433 / 0.948 | 0.325 / 0.981 |
| CIFAR-10 | layer4 | 0.661 / 0.975 | 0.494 / 0.982 | 0.729 / 0.823 | 0.628 / 0.927 |
| CIFAR-100 | layer2 | 0.355 / 1.000 | 0.064 / 1.000 | 0.481 / 0.977 | 0.300 / 0.992 |
| CIFAR-100 | layer3 | 0.042 / 1.000 | 0.113 / 0.999 | 0.468 / 0.986 | 0.208 / 0.995 |
| CIFAR-100 | layer4 | 0.207 / 1.000 | 0.641 / 0.911 | 0.693 / 0.895 | 0.514 / 0.935 |

## Absolute improvement

| ID | Layer | MNIST | SVHN | Opposite CIFAR | Macro |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | layer2 | 0.560 / 0.979 | 0.876 / 0.572 | 0.499 / 0.926 | 0.645 / 0.826 |
| CIFAR-10 | layer3 | 0.852 / 0.723 | 0.907 / 0.422 | 0.719 / 0.790 | 0.826 / 0.645 |
| CIFAR-10 | layer4 | 0.919 / 0.360 | 0.879 / 0.468 | 0.628 / 0.856 | 0.809 / 0.561 |
| CIFAR-100 | layer2 | 0.988 / 0.045 | 0.901 / 0.486 | 0.536 / 0.960 | 0.808 / 0.497 |
| CIFAR-100 | layer3 | 0.990 / 0.036 | 0.918 / 0.349 | 0.520 / 0.953 | 0.810 / 0.446 |
| CIFAR-100 | layer4 | 0.949 / 0.259 | 0.741 / 0.828 | 0.697 / 0.853 | 0.796 / 0.647 |

## Relative improvement

| ID | Layer | MNIST | SVHN | Opposite CIFAR | Macro |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | layer2 | 0.506 / 0.998 | 0.134 / 0.999 | 0.543 / 0.888 | 0.394 / 0.962 |
| CIFAR-10 | layer3 | 0.925 / 0.553 | 0.358 / 0.985 | 0.711 / 0.821 | 0.665 / 0.787 |
| CIFAR-10 | layer4 | 0.961 / 0.220 | 0.895 / 0.544 | 0.832 / 0.639 | 0.896 / 0.468 |
| CIFAR-100 | layer2 | 0.959 / 0.263 | 0.075 / 1.000 | 0.499 / 0.970 | 0.511 / 0.744 |
| CIFAR-100 | layer3 | 0.694 / 0.796 | 0.415 / 0.988 | 0.462 / 0.983 | 0.524 / 0.922 |
| CIFAR-100 | layer4 | 0.705 / 0.982 | 0.733 / 0.805 | 0.754 / 0.832 | 0.731 / 0.873 |

## Takeaway

CIFAR-10 layer 4 is strongest overall with relative improvement
(`0.896 / 0.468` macro) and gives the best opposite-CIFAR result
(`0.832 / 0.639`). For CIFAR-100, absolute improvement is strongest: layer 3
has the best macro result (`0.810 / 0.446`), while layer 4 is better on the
opposite-CIFAR near-OOD set (`0.697 / 0.853`). Raw reconstruction remains weak,
and near-OOD detection is the limiting case for CIFAR-100.
