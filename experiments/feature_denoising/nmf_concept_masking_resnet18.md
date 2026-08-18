# NMF Concept Masking

ResNet-18 layer4, one global 32-concept NMF basis, 20% Bernoulli concept
masking, residual CNN trained for 50 epochs, and 10 inference draws. Values are
`ROC-AUC / FPR@95`; macro is the unweighted mean over the three OOD datasets.

| ID | Score | MNIST | SVHN | Opposite CIFAR | Macro |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | Raw reconstruction | 0.709 / 0.959 | 0.743 / 0.914 | 0.773 / 0.837 | 0.741 / 0.903 |
| CIFAR-10 | Absolute improvement | 0.846 / 0.781 | 0.845 / 0.842 | 0.748 / 0.908 | 0.813 / 0.844 |
| CIFAR-10 | Relative improvement | 0.866 / 0.709 | 0.873 / 0.735 | 0.823 / 0.784 | **0.854 / 0.743** |
| CIFAR-100 | Raw reconstruction | 0.466 / 0.978 | 0.333 / 0.994 | 0.395 / 0.988 | 0.398 / 0.986 |
| CIFAR-100 | Absolute improvement | 0.468 / 0.984 | 0.647 / 0.925 | 0.606 / 0.925 | **0.574 / 0.945** |
| CIFAR-100 | Relative improvement | 0.420 / 0.987 | 0.524 / 0.927 | 0.518 / 0.926 | 0.487 / 0.947 |

Relative improvement is strongest for CIFAR-10, but the same global NMF setup
does not transfer reliably to CIFAR-100; only absolute improvement remains
above chance on average.
