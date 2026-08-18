# FAISS k-NN Neighbor-Output OOD Scores

Status: completed on 2026-08-05.

## Executive summary

The best overall setting is **Raw R18 with k-NN energy on CIFAR-10 ID**, reaching a macro 0.905 / 0.440 and compromise 0.733.

Across all ten settings, **k-NN energy** is the most robust score by mean compromise (0.565). Of the six perturbation-conditioned settings, 0 improve on their matched raw-architecture baseline after selecting each setting's best score.

The exhaustive tables below report every requested OOD Score. Macro values average CIFAR near-OOD, MNIST, and SVHN independently for ROC-AUC and FPR@95.

## Method

Each run builds an exact FAISS `IndexFlatL2` index from raw post-GAP `layer4` ID training representations. For each query, the selected `k = 10` neighbors act as a non-parametric student:

- mean neighbor probabilities are used for KL, maximum-probability gap, and k-NN MSP;
- mean neighbor logits are used for the absolute energy gap and k-NN energy;
- individual neighbor probabilities define predictive entropy and BALD.

All artifact scores are sign-adjusted so higher means more ID-like. Table cells are `ROC-AUC / FPR@95`; higher ROC-AUC and lower FPR@95 are better. The compromise objective is `0.5 * (macro AUROC + 1 - macro FPR@95)`.

For affine and channel-clipping runs, one corrupted representation per ID training image is indexed and queries remain clean. Spatial-clipping ResNet-50 runs search 50 independently clipped representations per query; all `50 * k` neighbor outputs are aggregated.

The `flatten` and training-objective labels belong to the replaced parametric students. The non-parametric replacement follows the requested post-GAP geometry, and Logit-MSE/KL training objectives have no operational role because no student is optimized.

## Completed runs

| ID | Setting | Query draws | Config | Job | Elapsed |
|---|---|---|---|---|---|
| CIFAR-10 | Raw R18 | 1 | [config](../../configs/embedding_distances/cifar_10/resnet18.yaml) | `408658` | 00:02:22 |
| CIFAR-10 | Raw R50 | 1 | [config](../../configs/embedding_distances/cifar_10/resnet50.yaml) | `408659` | 00:02:22 |
| CIFAR-10 | Affine R50 | 1 | [config](../../configs/embedding_distances/perturbation/pixel/augmentation/cifar_10/resnet50.yaml) | `408662` | 00:03:30 |
| CIFAR-10 | Channel clip R18 | 1 | [config](../../configs/embedding_distances/perturbation/embedding/clipping/cifar_10/resnet18_channel_flatten.yaml) | `408664` | 00:00:31 |
| CIFAR-10 | Spatial clip R50 | 50 | [config](../../configs/embedding_distances/perturbation/embedding/clipping/cifar_10/resnet50_spatial_flatten.yaml) | `408666` | 00:10:41 |
| CIFAR-100 | Raw R18 | 1 | [config](../../configs/embedding_distances/cifar_100/resnet18.yaml) | `408660` | 00:00:32 |
| CIFAR-100 | Raw R50 | 1 | [config](../../configs/embedding_distances/cifar_100/resnet50.yaml) | `408661` | 00:01:00 |
| CIFAR-100 | Affine R50 | 1 | [config](../../configs/embedding_distances/perturbation/pixel/augmentation/cifar_100/resnet50.yaml) | `408663` | 00:03:29 |
| CIFAR-100 | Channel clip R18 | 1 | [config](../../configs/embedding_distances/perturbation/embedding/clipping/cifar_100/resnet18_channel_flatten.yaml) | `408665` | 00:00:34 |
| CIFAR-100 | Spatial clip R50 | 50 | [config](../../configs/embedding_distances/perturbation/embedding/clipping/cifar_100/resnet50_spatial_flatten.yaml) | `408667` | 00:10:26 |

## Compromise-selected result per setting

| ID | Setting | Selected score | Near | MNIST | SVHN | Macro | Compromise |
|---|---|---|---|---|---|---|---|
| CIFAR-10 | Raw R18 | k-NN energy | 0.863 / 0.576 | 0.949 / 0.294 | 0.902 / 0.448 | 0.905 / 0.440 | 0.733 |
| CIFAR-10 | Raw R50 | k-NN energy | 0.854 / 0.559 | 0.914 / 0.509 | 0.906 / 0.498 | 0.891 / 0.522 | 0.685 |
| CIFAR-10 | Affine R50 | k-NN energy | 0.838 / 0.640 | 0.873 / 0.723 | 0.918 / 0.402 | 0.876 / 0.588 | 0.644 |
| CIFAR-10 | Channel clip R18 | k-NN energy | 0.824 / 0.671 | 0.898 / 0.623 | 0.855 / 0.694 | 0.859 / 0.662 | 0.598 |
| CIFAR-10 | Spatial clip R50 | BALD | 0.875 / 0.579 | 0.906 / 0.545 | 0.881 / 0.667 | 0.888 / 0.597 | 0.645 |
| CIFAR-100 | Raw R18 | KL(teacher || k-NN) | 0.789 / 0.814 | 0.716 / 0.923 | 0.826 / 0.725 | 0.777 / 0.820 | 0.478 |
| CIFAR-100 | Raw R50 | Absolute energy gap | 0.742 / 0.799 | 0.863 / 0.614 | 0.778 / 0.730 | 0.794 / 0.714 | 0.540 |
| CIFAR-100 | Affine R50 | KL(teacher || k-NN) | 0.794 / 0.798 | 0.818 / 0.812 | 0.826 / 0.706 | 0.813 / 0.772 | 0.520 |
| CIFAR-100 | Channel clip R18 | KL(teacher || k-NN) | 0.790 / 0.811 | 0.695 / 0.963 | 0.826 / 0.685 | 0.770 / 0.820 | 0.475 |
| CIFAR-100 | Spatial clip R50 | k-NN energy | 0.732 / 0.865 | 0.861 / 0.642 | 0.703 / 0.815 | 0.765 / 0.774 | 0.496 |

## Perturbation comparison with matched raw k-NN

Each side selects its own best score by macro compromise. Positive deltas favor the perturbation.

| ID | Perturbation | Its best score | Perturbation macro | Raw best score | Raw macro | Δ macro compromise | Δ near compromise |
|---|---|---|---|---|---|---|---|
| CIFAR-10 | Affine R50 | k-NN energy | 0.876 / 0.588 | k-NN energy | 0.891 / 0.522 | -0.041 | -0.048 |
| CIFAR-10 | Channel clip R18 | k-NN energy | 0.859 / 0.662 | k-NN energy | 0.905 / 0.440 | -0.134 | -0.067 |
| CIFAR-10 | Spatial clip R50 | BALD | 0.888 / 0.597 | k-NN energy | 0.891 / 0.522 | -0.039 | +0.001 |
| CIFAR-100 | Affine R50 | KL(teacher || k-NN) | 0.813 / 0.772 | Absolute energy gap | 0.794 / 0.714 | -0.020 | +0.026 |
| CIFAR-100 | Channel clip R18 | KL(teacher || k-NN) | 0.770 / 0.820 | KL(teacher || k-NN) | 0.777 / 0.820 | -0.003 | +0.002 |
| CIFAR-100 | Spatial clip R50 | k-NN energy | 0.765 / 0.774 | Absolute energy gap | 0.794 / 0.714 | -0.045 | -0.038 |

## Score robustness across all settings

| OOD Score | Mean macro compromise | Worst macro compromise | Mean near-OOD compromise |
|---|---|---|---|
| k-NN energy | 0.565 | 0.435 | 0.540 |
| KL(teacher || k-NN) | 0.551 | 0.383 | 0.538 |
| Predictive entropy | 0.540 | 0.429 | 0.530 |
| BALD | 0.540 | 0.425 | 0.529 |
| k-NN MSP | 0.529 | 0.421 | 0.523 |
| Absolute max-probability gap | 0.508 | 0.253 | 0.500 |
| Absolute energy gap | 0.460 | 0.130 | 0.440 |

## Interpretation

### Raw layer4 k-NN is the strongest choice

No corruption-conditioned run improves macro compromise over its matched raw architecture. The strongest result is raw ResNet-18 on CIFAR-10 with k-NN energy (`0.905 / 0.440`). Raw ResNet-50 is also better than both of its CIFAR-10 perturbation variants.

### CIFAR-100 remains the difficult regime

Every CIFAR-100 setting has high FPR@95. The best is raw ResNet-50 with the absolute energy gap (`0.794 / 0.714`, compromise `0.540`). Affine corruption raises macro AUROC to `0.813` with KL, but worsens macro FPR@95 to `0.772`, so its compromise is lower (`0.520`). This confirms that AUROC-only selection would overstate the affine gain.

### Perturbations offer only isolated near-OOD gains

Affine ResNet-50 improves CIFAR-100 near-OOD compromise by `+0.026`, and channel clipping improves it by `+0.002`; neither improvement survives the macro comparison. On CIFAR-10, 50-draw spatial clipping with BALD essentially ties raw ResNet-50 on near-OOD (`+0.001`) but loses `0.039` macro compromise because far-OOD FPR@95 is worse.

### Output scale makes the absolute energy gap fragile

The absolute energy gap is the least robust score overall. Under CIFAR-100 spatial clipping it collapses to macro `0.258 / 0.998`, indicating that averaging logits across many clipped neighborhoods introduces a scale shift that is almost maximally harmful at the 95% TPR operating point. KL and uncertainty-based scores tolerate the corruption substantially better.

### Recommended use

Use raw post-GAP layer4 k-NN as the default. Prefer k-NN energy for CIFAR-10. For CIFAR-100, choose scores per architecture: KL for ResNet-18 and absolute energy gap for raw ResNet-50. The perturbation variants are useful as negative or robustness ablations, not as replacements for the raw index.

## Exhaustive OOD metrics

### CIFAR-10 as ID

| Setting | OOD Score | Near: CIFAR-100 | Far: MNIST | Far: SVHN | Macro | Compromise |
|---|---|---|---|---|---|---|
| Raw R18 | KL(teacher || k-NN) | 0.881 / 0.600 | 0.922 / 0.526 | 0.905 / 0.567 | 0.902 / 0.564 | 0.669 |
| Raw R18 | Absolute max-probability gap | 0.875 / 0.634 | 0.915 / 0.575 | 0.900 / 0.606 | 0.897 / 0.605 | 0.646 |
| Raw R18 | Absolute energy gap | 0.845 / 0.561 | 0.930 / 0.398 | 0.890 / 0.531 | 0.888 / 0.497 | 0.696 |
| Raw R18 | Predictive entropy | 0.872 / 0.673 | 0.913 / 0.597 | 0.902 / 0.568 | 0.896 / 0.613 | 0.642 |
| Raw R18 | BALD | 0.876 / 0.659 | 0.912 / 0.523 | 0.905 / 0.551 | 0.898 / 0.578 | 0.660 |
| Raw R18 | k-NN MSP | 0.871 / 0.679 | 0.909 / 0.618 | 0.900 / 0.589 | 0.893 / 0.629 | 0.632 |
| Raw R18 | k-NN energy | 0.863 / 0.576 | 0.949 / 0.294 | 0.902 / 0.448 | 0.905 / 0.440 | 0.733 |
| Raw R50 | KL(teacher || k-NN) | 0.876 / 0.589 | 0.906 / 0.557 | 0.890 / 0.663 | 0.890 / 0.603 | 0.644 |
| Raw R50 | Absolute max-probability gap | 0.869 / 0.637 | 0.897 / 0.618 | 0.882 / 0.685 | 0.882 / 0.647 | 0.618 |
| Raw R50 | Absolute energy gap | 0.828 / 0.575 | 0.903 / 0.430 | 0.859 / 0.627 | 0.863 / 0.544 | 0.660 |
| Raw R50 | Predictive entropy | 0.865 / 0.672 | 0.873 / 0.706 | 0.896 / 0.646 | 0.878 / 0.675 | 0.602 |
| Raw R50 | BALD | 0.868 / 0.664 | 0.874 / 0.699 | 0.898 / 0.647 | 0.880 / 0.670 | 0.605 |
| Raw R50 | k-NN MSP | 0.863 / 0.677 | 0.870 / 0.709 | 0.894 / 0.648 | 0.876 / 0.678 | 0.599 |
| Raw R50 | k-NN energy | 0.854 / 0.559 | 0.914 / 0.509 | 0.906 / 0.498 | 0.891 / 0.522 | 0.685 |
| Affine R50 | KL(teacher || k-NN) | 0.869 / 0.642 | 0.860 / 0.799 | 0.934 / 0.373 | 0.888 / 0.604 | 0.642 |
| Affine R50 | Absolute max-probability gap | 0.852 / 0.761 | 0.845 / 0.896 | 0.924 / 0.466 | 0.873 / 0.708 | 0.583 |
| Affine R50 | Absolute energy gap | 0.716 / 0.723 | 0.840 / 0.543 | 0.672 / 0.819 | 0.743 / 0.695 | 0.524 |
| Affine R50 | Predictive entropy | 0.849 / 0.643 | 0.837 / 0.745 | 0.929 / 0.385 | 0.872 / 0.591 | 0.640 |
| Affine R50 | BALD | 0.845 / 0.643 | 0.823 / 0.752 | 0.928 / 0.373 | 0.865 / 0.589 | 0.638 |
| Affine R50 | k-NN MSP | 0.846 / 0.684 | 0.834 / 0.772 | 0.924 / 0.459 | 0.868 / 0.638 | 0.615 |
| Affine R50 | k-NN energy | 0.838 / 0.640 | 0.873 / 0.723 | 0.918 / 0.402 | 0.876 / 0.588 | 0.644 |
| Channel clip R18 | KL(teacher || k-NN) | 0.870 / 0.684 | 0.896 / 0.690 | 0.882 / 0.698 | 0.883 / 0.691 | 0.596 |
| Channel clip R18 | Absolute max-probability gap | 0.862 / 0.736 | 0.889 / 0.736 | 0.874 / 0.758 | 0.875 / 0.743 | 0.566 |
| Channel clip R18 | Absolute energy gap | 0.747 / 0.692 | 0.834 / 0.571 | 0.789 / 0.636 | 0.790 / 0.633 | 0.579 |
| Channel clip R18 | Predictive entropy | 0.836 / 0.709 | 0.866 / 0.682 | 0.851 / 0.727 | 0.851 / 0.706 | 0.572 |
| Channel clip R18 | BALD | 0.826 / 0.708 | 0.843 / 0.686 | 0.831 / 0.733 | 0.834 / 0.709 | 0.562 |
| Channel clip R18 | k-NN MSP | 0.834 / 0.721 | 0.861 / 0.707 | 0.850 / 0.736 | 0.848 / 0.721 | 0.563 |
| Channel clip R18 | k-NN energy | 0.824 / 0.671 | 0.898 / 0.623 | 0.855 / 0.694 | 0.859 / 0.662 | 0.598 |
| Spatial clip R50 | KL(teacher || k-NN) | 0.839 / 0.724 | 0.870 / 0.682 | 0.874 / 0.621 | 0.861 / 0.676 | 0.592 |
| Spatial clip R50 | Absolute max-probability gap | 0.808 / 0.766 | 0.832 / 0.717 | 0.833 / 0.721 | 0.825 / 0.735 | 0.545 |
| Spatial clip R50 | Absolute energy gap | 0.590 / 0.911 | 0.564 / 0.940 | 0.561 / 0.939 | 0.572 / 0.930 | 0.321 |
| Spatial clip R50 | Predictive entropy | 0.875 / 0.580 | 0.906 / 0.547 | 0.881 / 0.666 | 0.888 / 0.598 | 0.645 |
| Spatial clip R50 | BALD | 0.875 / 0.579 | 0.906 / 0.545 | 0.881 / 0.667 | 0.888 / 0.597 | 0.645 |
| Spatial clip R50 | k-NN MSP | 0.866 / 0.633 | 0.892 / 0.609 | 0.881 / 0.685 | 0.880 / 0.642 | 0.619 |
| Spatial clip R50 | k-NN energy | 0.866 / 0.608 | 0.902 / 0.574 | 0.868 / 0.690 | 0.879 / 0.624 | 0.628 |

### CIFAR-100 as ID

| Setting | OOD Score | Near: CIFAR-10 | Far: MNIST | Far: SVHN | Macro | Compromise |
|---|---|---|---|---|---|---|
| Raw R18 | KL(teacher || k-NN) | 0.789 / 0.814 | 0.716 / 0.923 | 0.826 / 0.725 | 0.777 / 0.820 | 0.478 |
| Raw R18 | Absolute max-probability gap | 0.782 / 0.825 | 0.706 / 0.937 | 0.812 / 0.776 | 0.767 / 0.846 | 0.460 |
| Raw R18 | Absolute energy gap | 0.691 / 0.853 | 0.695 / 0.891 | 0.773 / 0.714 | 0.720 / 0.819 | 0.450 |
| Raw R18 | Predictive entropy | 0.768 / 0.810 | 0.670 / 0.923 | 0.734 / 0.863 | 0.724 / 0.866 | 0.429 |
| Raw R18 | BALD | 0.754 / 0.823 | 0.717 / 0.907 | 0.730 / 0.857 | 0.734 / 0.862 | 0.436 |
| Raw R18 | k-NN MSP | 0.767 / 0.834 | 0.669 / 0.916 | 0.732 / 0.872 | 0.723 / 0.874 | 0.424 |
| Raw R18 | k-NN energy | 0.774 / 0.813 | 0.656 / 0.913 | 0.748 / 0.843 | 0.726 / 0.856 | 0.435 |
| Raw R50 | KL(teacher || k-NN) | 0.798 / 0.791 | 0.823 / 0.812 | 0.779 / 0.763 | 0.800 / 0.789 | 0.506 |
| Raw R50 | Absolute max-probability gap | 0.791 / 0.788 | 0.823 / 0.792 | 0.764 / 0.797 | 0.793 / 0.792 | 0.500 |
| Raw R50 | Absolute energy gap | 0.742 / 0.799 | 0.863 / 0.614 | 0.778 / 0.730 | 0.794 / 0.714 | 0.540 |
| Raw R50 | Predictive entropy | 0.763 / 0.842 | 0.744 / 0.890 | 0.671 / 0.871 | 0.726 / 0.867 | 0.429 |
| Raw R50 | BALD | 0.757 / 0.837 | 0.700 / 0.911 | 0.698 / 0.860 | 0.718 / 0.869 | 0.425 |
| Raw R50 | k-NN MSP | 0.759 / 0.852 | 0.732 / 0.909 | 0.670 / 0.873 | 0.721 / 0.878 | 0.421 |
| Raw R50 | k-NN energy | 0.770 / 0.830 | 0.810 / 0.882 | 0.671 / 0.860 | 0.750 / 0.857 | 0.446 |
| Affine R50 | KL(teacher || k-NN) | 0.794 / 0.798 | 0.818 / 0.812 | 0.826 / 0.706 | 0.813 / 0.772 | 0.520 |
| Affine R50 | Absolute max-probability gap | 0.768 / 0.849 | 0.808 / 0.805 | 0.806 / 0.719 | 0.794 / 0.791 | 0.501 |
| Affine R50 | Absolute energy gap | 0.654 / 0.859 | 0.790 / 0.733 | 0.558 / 0.939 | 0.667 / 0.844 | 0.412 |
| Affine R50 | Predictive entropy | 0.745 / 0.804 | 0.748 / 0.881 | 0.805 / 0.532 | 0.766 / 0.739 | 0.513 |
| Affine R50 | BALD | 0.740 / 0.803 | 0.690 / 0.882 | 0.821 / 0.522 | 0.750 / 0.736 | 0.507 |
| Affine R50 | k-NN MSP | 0.741 / 0.819 | 0.739 / 0.888 | 0.796 / 0.604 | 0.759 / 0.770 | 0.494 |
| Affine R50 | k-NN energy | 0.737 / 0.803 | 0.800 / 0.859 | 0.786 / 0.574 | 0.774 / 0.745 | 0.515 |
| Channel clip R18 | KL(teacher || k-NN) | 0.790 / 0.811 | 0.695 / 0.963 | 0.826 / 0.685 | 0.770 / 0.820 | 0.475 |
| Channel clip R18 | Absolute max-probability gap | 0.750 / 0.859 | 0.644 / 0.968 | 0.745 / 0.843 | 0.713 / 0.890 | 0.412 |
| Channel clip R18 | Absolute energy gap | 0.558 / 0.892 | 0.562 / 0.968 | 0.453 / 0.971 | 0.524 / 0.944 | 0.290 |
| Channel clip R18 | Predictive entropy | 0.722 / 0.820 | 0.680 / 0.905 | 0.815 / 0.728 | 0.739 / 0.818 | 0.461 |
| Channel clip R18 | BALD | 0.717 / 0.818 | 0.668 / 0.903 | 0.794 / 0.731 | 0.726 / 0.817 | 0.454 |
| Channel clip R18 | k-NN MSP | 0.719 / 0.831 | 0.677 / 0.917 | 0.809 / 0.744 | 0.735 / 0.831 | 0.452 |
| Channel clip R18 | k-NN energy | 0.722 / 0.822 | 0.692 / 0.909 | 0.839 / 0.709 | 0.751 / 0.813 | 0.469 |
| Spatial clip R50 | KL(teacher || k-NN) | 0.631 / 0.916 | 0.741 / 0.819 | 0.589 / 0.929 | 0.654 / 0.888 | 0.383 |
| Spatial clip R50 | Absolute max-probability gap | 0.448 / 0.955 | 0.512 / 0.917 | 0.409 / 0.979 | 0.456 / 0.950 | 0.253 |
| Spatial clip R50 | Absolute energy gap | 0.288 / 0.994 | 0.218 / 1.000 | 0.266 / 0.999 | 0.258 / 0.998 | 0.130 |
| Spatial clip R50 | Predictive entropy | 0.716 / 0.856 | 0.849 / 0.697 | 0.687 / 0.903 | 0.750 / 0.819 | 0.466 |
| Spatial clip R50 | BALD | 0.716 / 0.856 | 0.849 / 0.697 | 0.687 / 0.903 | 0.750 / 0.819 | 0.466 |
| Spatial clip R50 | k-NN MSP | 0.749 / 0.830 | 0.811 / 0.699 | 0.668 / 0.867 | 0.743 / 0.799 | 0.472 |
| Spatial clip R50 | k-NN energy | 0.732 / 0.865 | 0.861 / 0.642 | 0.703 / 0.815 | 0.765 / 0.774 | 0.496 |

## Reproducibility

The report is generated by [`reports/scripts/build_knn_ood_score_report.py`](../../reports/scripts/build_knn_ood_score_report.py). Complete per-dataset descriptive statistics and OOD metrics are stored in [`reports/outputs/json/knn_faiss_ood_scores.json`](../../reports/outputs/json/knn_faiss_ood_scores.json).

The raw artifacts remain under the corresponding `runs/embedding_distances/` directories and contain neighbor indices, distances, mean probabilities, mean logits, raw metrics, and sign-adjusted OOD Scores.
