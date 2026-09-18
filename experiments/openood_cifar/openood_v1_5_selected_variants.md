# OpenOOD CIFAR-10/CIFAR-100 evaluation

This report contains every metric produced for the completed selected variants under the fixed OpenOOD v1.5 CIFAR protocol.

## Evaluation convention

- OOD is the positive class and ID is the negative class.
- Input scores are ID confidence; they are negated for OOD-positive ROC/FPR calculations.
- FPR95 is the ID false-positive rate at 95% OOD true-positive rate.
- All metrics are fractions in `[0, 1]`.
- Near/Far rows are arithmetic macro averages over their datasets.

## Evaluated variants

| # | ID | Variant | Distillation | Inference | Scores |
| ---: | --- | --- | --- | --- | ---: |
| 1 | CIFAR-10 | Cifar10 affine resnet50 linear clean target mse logits clean inference | `mse_logits` | unperturbed | 10 |
| 2 | CIFAR-100 | Cifar100 affine resnet50 linear clean target mse logits clean inference | `mse_logits` | unperturbed | 10 |
| 3 | CIFAR-10 | Cifar10 channel clip resnet18 layers34 flatten kl clean inference | `kl_divergence` | unperturbed | 10 |
| 4 | CIFAR-100 | Cifar100 channel clip resnet18 layers34 flatten kl clean inference | `kl_divergence` | unperturbed | 10 |
| 5 | CIFAR-10 | Cifar10 feature denoising resnet18 layer2 mask p020 10 draw | `feature denoising` | 10_draw_reconstruction | 3 |
| 6 | CIFAR-10 | Cifar10 feature denoising resnet18 layer3 mask p020 10 draw | `feature denoising` | 10_draw_reconstruction | 3 |
| 7 | CIFAR-10 | Cifar10 feature denoising resnet18 layer4 mask p020 10 draw | `feature denoising` | 10_draw_reconstruction | 3 |
| 8 | CIFAR-100 | Cifar100 feature denoising resnet18 layer2 mask p020 10 draw | `feature denoising` | 10_draw_reconstruction | 3 |
| 9 | CIFAR-100 | Cifar100 feature denoising resnet18 layer3 mask p020 10 draw | `feature denoising` | 10_draw_reconstruction | 3 |
| 10 | CIFAR-100 | Cifar100 feature denoising resnet18 layer4 mask p020 10 draw | `feature denoising` | 10_draw_reconstruction | 3 |
| 11 | CIFAR-10 | Cifar10 feature denoising vit patch tokens layer12 mask p020 10 draw | `feature denoising` | 10_draw_reconstruction | 3 |
| 12 | CIFAR-100 | Cifar100 feature denoising vit patch tokens layer12 mask p020 10 draw | `feature denoising` | 10_draw_reconstruction | 3 |

## Dataset sample counts

### CIFAR-10 ID

| Dataset | Samples |
| --- | ---: |
| CIFAR-100 | 9,000 |
| CIFAR-10 | 9,000 |
| MNIST | 70,000 |
| Places365 | 35,195 |
| SVHN | 26,032 |
| Textures | 5,640 |
| TIN | 7,793 |

### CIFAR-100 ID

| Dataset | Samples |
| --- | ---: |
| CIFAR-100 | 9,000 |
| CIFAR-10 | 10,000 |
| MNIST | 70,000 |
| Places365 | 33,773 |
| SVHN | 26,032 |
| Textures | 5,640 |
| TIN | 6,526 |

## Complete metrics

### 1. Cifar10 affine resnet50 linear clean target mse logits clean inference

- ID dataset: CIFAR-10
- Experiment: `linear_layer4_pixel_augmentation_clean_target_student_resnet50_cifar10`
- Checkpoint: `best`
- Distillation method: `mse_logits`
- Inference mode: `unperturbed`

#### Absolute energy gap

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-100 | Near | 0.9817 | 0.3770 | 0.4107 | 0.4391 |
| MNIST | Far | 0.9873 | 0.2821 | 0.0736 | 0.8073 |
| Places365 | Far | 0.9854 | 0.2995 | 0.1388 | 0.7009 |
| SVHN | Far | 0.9624 | 0.6320 | 0.3046 | 0.8445 |
| Textures | Far | 0.9736 | 0.5370 | 0.6046 | 0.4919 |
| TIN | Near | 0.9810 | 0.3776 | 0.4460 | 0.4032 |
| Near macro | Near | 0.9813 | 0.3773 | 0.4283 | 0.4212 |
| Far macro | Far | 0.9772 | 0.4376 | 0.2804 | 0.7112 |

#### Absolute max probability difference

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-100 | Near | 0.4679 | 0.8687 | 0.8686 | 0.8322 |
| MNIST | Far | 0.2443 | 0.9029 | 0.7906 | 0.9738 |
| Places365 | Far | 0.3831 | 0.8801 | 0.7242 | 0.9504 |
| SVHN | Far | 0.2257 | 0.9235 | 0.8683 | 0.9600 |
| Textures | Far | 0.3648 | 0.8895 | 0.9265 | 0.7979 |
| TIN | Near | 0.3842 | 0.8839 | 0.9025 | 0.8268 |
| Near macro | Near | 0.4261 | 0.8763 | 0.8856 | 0.8295 |
| Far macro | Far | 0.3045 | 0.8990 | 0.8274 | 0.9205 |

#### Energy gap

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-100 | Near | 0.9802 | 0.3098 | 0.3866 | 0.3938 |
| MNIST | Far | 0.8992 | 0.3466 | 0.1404 | 0.8219 |
| Places365 | Far | 0.9560 | 0.3047 | 0.1504 | 0.6929 |
| SVHN | Far | 0.9986 | 0.0765 | 0.1449 | 0.5442 |
| Textures | Far | 0.9917 | 0.4172 | 0.5287 | 0.4133 |
| TIN | Near | 0.9786 | 0.3194 | 0.4257 | 0.3658 |
| Near macro | Near | 0.9794 | 0.3146 | 0.4062 | 0.3798 |
| Far macro | Far | 0.9614 | 0.2863 | 0.2411 | 0.6181 |

#### Centered-logit L2 distance

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-100 | Near | 0.9020 | 0.6466 | 0.6114 | 0.6380 |
| MNIST | Far | 0.9894 | 0.2727 | 0.0730 | 0.7940 |
| Places365 | Far | 0.9290 | 0.5534 | 0.2341 | 0.8157 |
| SVHN | Far | 0.5088 | 0.8472 | 0.7108 | 0.9210 |
| Textures | Far | 0.6509 | 0.8322 | 0.8749 | 0.7633 |
| TIN | Near | 0.8949 | 0.6490 | 0.6524 | 0.6022 |
| Near macro | Near | 0.8984 | 0.6478 | 0.6319 | 0.6201 |
| Far macro | Far | 0.7695 | 0.6264 | 0.4732 | 0.8235 |

#### Max probability difference

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-100 | Near | 0.9913 | 0.6013 | 0.5027 | 0.6967 |
| MNIST | Far | 0.9851 | 0.7368 | 0.1903 | 0.9567 |
| Places365 | Far | 0.9898 | 0.6513 | 0.2466 | 0.8985 |
| SVHN | Far | 0.9976 | 0.3001 | 0.1829 | 0.7237 |
| Textures | Far | 0.9956 | 0.5691 | 0.5858 | 0.6042 |
| TIN | Near | 0.9932 | 0.6138 | 0.5419 | 0.6846 |
| Near macro | Near | 0.9923 | 0.6075 | 0.5223 | 0.6906 |
| Far macro | Far | 0.9920 | 0.5643 | 0.3014 | 0.7958 |

#### Student energy

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-100 | Near | 0.5654 | 0.8707 | 0.8563 | 0.8617 |
| MNIST | Far | 0.2552 | 0.9307 | 0.7596 | 0.9879 |
| Places365 | Far | 0.4003 | 0.9046 | 0.7450 | 0.9686 |
| SVHN | Far | 0.1456 | 0.9612 | 0.9229 | 0.9827 |
| Textures | Far | 0.6191 | 0.8467 | 0.8760 | 0.7688 |
| TIN | Near | 0.4680 | 0.8906 | 0.8953 | 0.8670 |
| Near macro | Near | 0.5167 | 0.8806 | 0.8758 | 0.8643 |
| Far macro | Far | 0.3551 | 0.9108 | 0.8259 | 0.9270 |

#### Student MSP

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-100 | Near | 0.4733 | 0.8698 | 0.8735 | 0.8415 |
| MNIST | Far | 0.3011 | 0.9018 | 0.7078 | 0.9802 |
| Places365 | Far | 0.3773 | 0.8888 | 0.7501 | 0.9584 |
| SVHN | Far | 0.1959 | 0.9302 | 0.8905 | 0.9629 |
| Textures | Far | 0.4423 | 0.8766 | 0.9179 | 0.7830 |
| TIN | Near | 0.4013 | 0.8845 | 0.9024 | 0.8383 |
| Near macro | Near | 0.4373 | 0.8772 | 0.8880 | 0.8399 |
| Far macro | Far | 0.3292 | 0.8993 | 0.8166 | 0.9211 |

#### Teacher-student KL divergence

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-100 | Near | 0.4898 | 0.8669 | 0.8646 | 0.8351 |
| MNIST | Far | 0.2457 | 0.8973 | 0.8005 | 0.9683 |
| Places365 | Far | 0.4031 | 0.8763 | 0.7149 | 0.9477 |
| SVHN | Far | 0.2362 | 0.9226 | 0.8687 | 0.9574 |
| Textures | Far | 0.3573 | 0.8973 | 0.9309 | 0.8251 |
| TIN | Near | 0.4022 | 0.8836 | 0.9013 | 0.8299 |
| Near macro | Near | 0.4460 | 0.8753 | 0.8829 | 0.8325 |
| Far macro | Far | 0.3106 | 0.8984 | 0.8288 | 0.9246 |

#### Teacher energy

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-100 | Near | 0.5980 | 0.8679 | 0.8499 | 0.8650 |
| MNIST | Far | 0.1763 | 0.9552 | 0.8557 | 0.9923 |
| Places365 | Far | 0.3919 | 0.9135 | 0.7509 | 0.9729 |
| SVHN | Far | 0.3248 | 0.9084 | 0.8292 | 0.9560 |
| Textures | Far | 0.5388 | 0.8682 | 0.8963 | 0.7976 |
| TIN | Near | 0.4748 | 0.8958 | 0.8967 | 0.8781 |
| Near macro | Near | 0.5364 | 0.8818 | 0.8733 | 0.8716 |
| Far macro | Far | 0.3579 | 0.9113 | 0.8330 | 0.9297 |

#### Teacher MSP

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-100 | Near | 0.4716 | 0.8726 | 0.8759 | 0.8456 |
| MNIST | Far | 0.2370 | 0.9203 | 0.7994 | 0.9838 |
| Places365 | Far | 0.3524 | 0.8950 | 0.7592 | 0.9612 |
| SVHN | Far | 0.3041 | 0.8891 | 0.8303 | 0.9382 |
| Textures | Far | 0.4173 | 0.8764 | 0.9155 | 0.7809 |
| TIN | Near | 0.3819 | 0.8886 | 0.9060 | 0.8435 |
| Near macro | Near | 0.4267 | 0.8806 | 0.8909 | 0.8445 |
| Far macro | Far | 0.3277 | 0.8952 | 0.8261 | 0.9160 |

### 2. Cifar100 affine resnet50 linear clean target mse logits clean inference

- ID dataset: CIFAR-100
- Experiment: `linear_layer4_pixel_augmentation_clean_target_student_resnet50_cifar100`
- Checkpoint: `best`
- Distillation method: `mse_logits`
- Inference mode: `unperturbed`

#### Absolute energy gap

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | Near | 0.9656 | 0.4250 | 0.4310 | 0.4566 |
| MNIST | Far | 0.9823 | 0.2237 | 0.0701 | 0.7727 |
| Places365 | Far | 0.9643 | 0.4055 | 0.1779 | 0.7249 |
| SVHN | Far | 0.8594 | 0.7051 | 0.4072 | 0.8895 |
| Textures | Far | 0.9631 | 0.4067 | 0.5621 | 0.3437 |
| TIN | Near | 0.9711 | 0.3832 | 0.5099 | 0.3393 |
| Near macro | Near | 0.9683 | 0.4041 | 0.4704 | 0.3980 |
| Far macro | Far | 0.9423 | 0.4353 | 0.3043 | 0.6827 |

#### Absolute max probability difference

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | Near | 0.6188 | 0.7571 | 0.7698 | 0.7106 |
| MNIST | Far | 0.6139 | 0.7439 | 0.3940 | 0.9283 |
| Places365 | Far | 0.6891 | 0.7564 | 0.4923 | 0.8963 |
| SVHN | Far | 0.3573 | 0.8752 | 0.7938 | 0.9452 |
| Textures | Far | 0.7274 | 0.7523 | 0.8190 | 0.6158 |
| TIN | Near | 0.5850 | 0.7816 | 0.8426 | 0.6518 |
| Near macro | Near | 0.6019 | 0.7693 | 0.8062 | 0.6812 |
| Far macro | Far | 0.5969 | 0.7820 | 0.6248 | 0.8464 |

#### Energy gap

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | Near | 0.9209 | 0.4570 | 0.4777 | 0.4714 |
| MNIST | Far | 0.9024 | 0.2625 | 0.1403 | 0.7801 |
| Places365 | Far | 0.9357 | 0.4327 | 0.2032 | 0.7322 |
| SVHN | Far | 0.9998 | 0.0833 | 0.1457 | 0.5459 |
| Textures | Far | 0.9887 | 0.3364 | 0.5157 | 0.2888 |
| TIN | Near | 0.9389 | 0.4063 | 0.5485 | 0.3473 |
| Near macro | Near | 0.9299 | 0.4316 | 0.5131 | 0.4093 |
| Far macro | Far | 0.9566 | 0.2787 | 0.2512 | 0.5868 |

#### Centered-logit L2 distance

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | Near | 0.7723 | 0.6476 | 0.6671 | 0.6164 |
| MNIST | Far | 0.9587 | 0.1935 | 0.0775 | 0.7623 |
| Places365 | Far | 0.7443 | 0.6794 | 0.4504 | 0.8517 |
| SVHN | Far | 0.2083 | 0.9621 | 0.9104 | 0.9857 |
| Textures | Far | 0.8494 | 0.6998 | 0.7675 | 0.5883 |
| TIN | Near | 0.7330 | 0.7039 | 0.7816 | 0.5646 |
| Near macro | Near | 0.7527 | 0.6758 | 0.7243 | 0.5905 |
| Far macro | Far | 0.6902 | 0.6337 | 0.5514 | 0.7970 |

#### Max probability difference

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | Near | 0.9741 | 0.6029 | 0.4967 | 0.6573 |
| MNIST | Far | 0.9654 | 0.4175 | 0.1027 | 0.8781 |
| Places365 | Far | 0.9794 | 0.5747 | 0.2186 | 0.8541 |
| SVHN | Far | 0.9999 | 0.2269 | 0.1672 | 0.6601 |
| Textures | Far | 0.9929 | 0.4976 | 0.5613 | 0.4637 |
| TIN | Near | 0.9808 | 0.5716 | 0.5708 | 0.5610 |
| Near macro | Near | 0.9774 | 0.5872 | 0.5338 | 0.6091 |
| Far macro | Far | 0.9844 | 0.4292 | 0.2624 | 0.7140 |

#### Student energy

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | Near | 0.7154 | 0.7771 | 0.7560 | 0.7617 |
| MNIST | Far | 0.3342 | 0.8865 | 0.7299 | 0.9771 |
| Places365 | Far | 0.7158 | 0.7699 | 0.4890 | 0.9060 |
| SVHN | Far | 0.2483 | 0.9291 | 0.8719 | 0.9696 |
| Textures | Far | 0.6227 | 0.7973 | 0.8613 | 0.6676 |
| TIN | Near | 0.5958 | 0.8166 | 0.8552 | 0.7241 |
| Near macro | Near | 0.6556 | 0.7968 | 0.8056 | 0.7429 |
| Far macro | Far | 0.4802 | 0.8457 | 0.7380 | 0.8801 |

#### Student MSP

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | Near | 0.6729 | 0.7766 | 0.7677 | 0.7560 |
| MNIST | Far | 0.4159 | 0.8304 | 0.6560 | 0.9632 |
| Places365 | Far | 0.6448 | 0.7805 | 0.5318 | 0.9108 |
| SVHN | Far | 0.3372 | 0.8725 | 0.8070 | 0.9398 |
| Textures | Far | 0.6124 | 0.7824 | 0.8574 | 0.6411 |
| TIN | Near | 0.5603 | 0.8120 | 0.8587 | 0.7131 |
| Near macro | Near | 0.6166 | 0.7943 | 0.8132 | 0.7346 |
| Far macro | Far | 0.5026 | 0.8165 | 0.7131 | 0.8637 |

#### Teacher-student KL divergence

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | Near | 0.6018 | 0.7677 | 0.7850 | 0.7157 |
| MNIST | Far | 0.6417 | 0.7164 | 0.4156 | 0.9123 |
| Places365 | Far | 0.6654 | 0.7705 | 0.5277 | 0.9008 |
| SVHN | Far | 0.2966 | 0.9132 | 0.8478 | 0.9625 |
| Textures | Far | 0.7486 | 0.7632 | 0.8214 | 0.6346 |
| TIN | Near | 0.5766 | 0.7989 | 0.8560 | 0.6652 |
| Near macro | Near | 0.5892 | 0.7833 | 0.8205 | 0.6904 |
| Far macro | Far | 0.5881 | 0.7908 | 0.6531 | 0.8525 |

#### Teacher energy

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | Near | 0.6448 | 0.7988 | 0.7861 | 0.7789 |
| MNIST | Far | 0.3661 | 0.8779 | 0.7078 | 0.9761 |
| Places365 | Far | 0.7047 | 0.7811 | 0.4972 | 0.9129 |
| SVHN | Far | 0.7116 | 0.7730 | 0.5776 | 0.8922 |
| Textures | Far | 0.6862 | 0.7787 | 0.8427 | 0.6431 |
| TIN | Near | 0.5802 | 0.8279 | 0.8644 | 0.7406 |
| Near macro | Near | 0.6125 | 0.8133 | 0.8252 | 0.7598 |
| Far macro | Far | 0.6171 | 0.8027 | 0.6563 | 0.8561 |

#### Teacher MSP

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | Near | 0.6182 | 0.7895 | 0.7890 | 0.7657 |
| MNIST | Far | 0.4367 | 0.8196 | 0.6361 | 0.9606 |
| Places365 | Far | 0.6323 | 0.7863 | 0.5386 | 0.9136 |
| SVHN | Far | 0.7289 | 0.7405 | 0.5476 | 0.8748 |
| Textures | Far | 0.6746 | 0.7659 | 0.8401 | 0.6193 |
| TIN | Near | 0.5412 | 0.8188 | 0.8657 | 0.7213 |
| Near macro | Near | 0.5797 | 0.8042 | 0.8274 | 0.7435 |
| Far macro | Far | 0.6181 | 0.7781 | 0.6406 | 0.8421 |

### 3. Cifar10 channel clip resnet18 layers34 flatten kl clean inference

- ID dataset: CIFAR-10
- Experiment: `perturbation_linear_layer3_layer4_student_resnet18_cifar10_clip_channel_flatten`
- Checkpoint: `best`
- Distillation method: `kl_divergence`
- Inference mode: `unperturbed`

#### Absolute energy gap

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-100 | Near | 0.9811 | 0.3239 | 0.3945 | 0.3831 |
| MNIST | Far | 0.9896 | 0.2382 | 0.0697 | 0.7843 |
| Places365 | Far | 0.9844 | 0.2879 | 0.1378 | 0.6758 |
| SVHN | Far | 0.9852 | 0.2525 | 0.1698 | 0.5985 |
| Textures | Far | 0.9857 | 0.2445 | 0.4707 | 0.2610 |
| TIN | Near | 0.9842 | 0.2896 | 0.4138 | 0.3385 |
| Near macro | Near | 0.9827 | 0.3067 | 0.4042 | 0.3608 |
| Far macro | Far | 0.9862 | 0.2558 | 0.2120 | 0.5799 |

#### Absolute max probability difference

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-100 | Near | 0.4372 | 0.8706 | 0.8775 | 0.8328 |
| MNIST | Far | 0.2739 | 0.9064 | 0.7285 | 0.9807 |
| Places365 | Far | 0.4087 | 0.8745 | 0.7162 | 0.9500 |
| SVHN | Far | 0.2910 | 0.8965 | 0.8376 | 0.9402 |
| Textures | Far | 0.3553 | 0.8886 | 0.9327 | 0.7815 |
| TIN | Near | 0.3918 | 0.8791 | 0.9017 | 0.8200 |
| Near macro | Near | 0.4145 | 0.8748 | 0.8896 | 0.8264 |
| Far macro | Far | 0.3322 | 0.8915 | 0.8037 | 0.9131 |

#### Energy gap

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-100 | Near | 0.9888 | 0.3193 | 0.3890 | 0.3819 |
| MNIST | Far | 0.9771 | 0.2440 | 0.0796 | 0.7850 |
| Places365 | Far | 0.9859 | 0.2859 | 0.1372 | 0.6754 |
| SVHN | Far | 0.9800 | 0.2547 | 0.1741 | 0.5990 |
| Textures | Far | 0.9947 | 0.2279 | 0.4576 | 0.2571 |
| TIN | Near | 0.9908 | 0.2826 | 0.4065 | 0.3366 |
| Near macro | Near | 0.9898 | 0.3010 | 0.3978 | 0.3592 |
| Far macro | Far | 0.9844 | 0.2531 | 0.2121 | 0.5791 |

#### Centered-logit L2 distance

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-100 | Near | 0.9808 | 0.4079 | 0.4413 | 0.4176 |
| MNIST | Far | 0.9856 | 0.3596 | 0.0846 | 0.8153 |
| Places365 | Far | 0.9884 | 0.3399 | 0.1487 | 0.6955 |
| SVHN | Far | 0.9958 | 0.2343 | 0.1639 | 0.5964 |
| Textures | Far | 0.9930 | 0.2986 | 0.4944 | 0.2755 |
| TIN | Near | 0.9876 | 0.3636 | 0.4494 | 0.3654 |
| Near macro | Near | 0.9842 | 0.3857 | 0.4453 | 0.3915 |
| Far macro | Far | 0.9907 | 0.3081 | 0.2229 | 0.5957 |

#### Max probability difference

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-100 | Near | 0.9906 | 0.6208 | 0.5144 | 0.7127 |
| MNIST | Far | 0.9917 | 0.5480 | 0.1155 | 0.9286 |
| Places365 | Far | 0.9891 | 0.6318 | 0.2358 | 0.8941 |
| SVHN | Far | 0.9892 | 0.6806 | 0.3226 | 0.8828 |
| Textures | Far | 0.9934 | 0.5598 | 0.5813 | 0.5896 |
| TIN | Near | 0.9906 | 0.6284 | 0.5510 | 0.6968 |
| Near macro | Near | 0.9906 | 0.6246 | 0.5327 | 0.7047 |
| Far macro | Far | 0.9909 | 0.6050 | 0.3138 | 0.8238 |

#### Student energy

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-100 | Near | 0.6390 | 0.8439 | 0.8327 | 0.8439 |
| MNIST | Far | 0.3018 | 0.9370 | 0.7519 | 0.9901 |
| Places365 | Far | 0.5474 | 0.8798 | 0.6875 | 0.9634 |
| SVHN | Far | 0.4796 | 0.9073 | 0.7783 | 0.9639 |
| Textures | Far | 0.5082 | 0.8933 | 0.9261 | 0.8467 |
| TIN | Near | 0.5444 | 0.8778 | 0.8896 | 0.8608 |
| Near macro | Near | 0.5917 | 0.8608 | 0.8611 | 0.8524 |
| Far macro | Far | 0.4592 | 0.9043 | 0.7860 | 0.9410 |

#### Student MSP

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-100 | Near | 0.5894 | 0.8442 | 0.8450 | 0.8219 |
| MNIST | Far | 0.3190 | 0.9059 | 0.7267 | 0.9821 |
| Places365 | Far | 0.5019 | 0.8656 | 0.7024 | 0.9521 |
| SVHN | Far | 0.5080 | 0.8738 | 0.7371 | 0.9432 |
| Textures | Far | 0.5168 | 0.8742 | 0.9164 | 0.7948 |
| TIN | Near | 0.5092 | 0.8638 | 0.8857 | 0.8185 |
| Near macro | Near | 0.5493 | 0.8540 | 0.8654 | 0.8202 |
| Far macro | Far | 0.4614 | 0.8799 | 0.7706 | 0.9181 |

#### Teacher-student KL divergence

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-100 | Near | 0.4737 | 0.8674 | 0.8760 | 0.8226 |
| MNIST | Far | 0.2943 | 0.9067 | 0.7479 | 0.9776 |
| Places365 | Far | 0.4484 | 0.8697 | 0.7105 | 0.9424 |
| SVHN | Far | 0.3137 | 0.8909 | 0.8358 | 0.9283 |
| Textures | Far | 0.3951 | 0.8818 | 0.9271 | 0.7571 |
| TIN | Near | 0.4413 | 0.8742 | 0.8963 | 0.8069 |
| Near macro | Near | 0.4575 | 0.8708 | 0.8862 | 0.8147 |
| Far macro | Far | 0.3629 | 0.8873 | 0.8054 | 0.9014 |

#### Teacher energy

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-100 | Near | 0.5414 | 0.8790 | 0.8675 | 0.8734 |
| MNIST | Far | 0.1589 | 0.9630 | 0.8795 | 0.9939 |
| Places365 | Far | 0.4176 | 0.9091 | 0.7437 | 0.9716 |
| SVHN | Far | 0.2979 | 0.9346 | 0.8483 | 0.9726 |
| Textures | Far | 0.5494 | 0.8790 | 0.9041 | 0.8252 |
| TIN | Near | 0.4611 | 0.8971 | 0.9021 | 0.8781 |
| Near macro | Near | 0.5013 | 0.8881 | 0.8848 | 0.8757 |
| Far macro | Far | 0.3559 | 0.9214 | 0.8439 | 0.9408 |

#### Teacher MSP

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-100 | Near | 0.4374 | 0.8796 | 0.8855 | 0.8536 |
| MNIST | Far | 0.2303 | 0.9253 | 0.8198 | 0.9854 |
| Places365 | Far | 0.3734 | 0.8938 | 0.7555 | 0.9616 |
| SVHN | Far | 0.2661 | 0.9184 | 0.8565 | 0.9596 |
| Textures | Far | 0.4122 | 0.8852 | 0.9208 | 0.8005 |
| TIN | Near | 0.3781 | 0.8912 | 0.9089 | 0.8493 |
| Near macro | Near | 0.4078 | 0.8854 | 0.8972 | 0.8515 |
| Far macro | Far | 0.3205 | 0.9057 | 0.8381 | 0.9268 |

### 4. Cifar100 channel clip resnet18 layers34 flatten kl clean inference

- ID dataset: CIFAR-100
- Experiment: `perturbation_linear_layer3_layer4_student_resnet18_cifar100_clip_channel_flatten`
- Checkpoint: `best`
- Distillation method: `kl_divergence`
- Inference mode: `unperturbed`

#### Absolute energy gap

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | Near | 0.9554 | 0.3600 | 0.4020 | 0.4250 |
| MNIST | Far | 0.9909 | 0.0792 | 0.0609 | 0.7368 |
| Places365 | Far | 0.9812 | 0.2581 | 0.1378 | 0.6610 |
| SVHN | Far | 0.9967 | 0.0795 | 0.1444 | 0.5457 |
| Textures | Far | 0.9942 | 0.1559 | 0.4339 | 0.2416 |
| TIN | Near | 0.9726 | 0.2364 | 0.4410 | 0.2862 |
| Near macro | Near | 0.9640 | 0.2982 | 0.4215 | 0.3556 |
| Far macro | Far | 0.9908 | 0.1432 | 0.1942 | 0.5463 |

#### Absolute max probability difference

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | Near | 0.6339 | 0.7308 | 0.7543 | 0.6871 |
| MNIST | Far | 0.6348 | 0.7354 | 0.4162 | 0.9344 |
| Places365 | Far | 0.5911 | 0.7554 | 0.5739 | 0.9004 |
| SVHN | Far | 0.5507 | 0.7822 | 0.6631 | 0.8981 |
| Textures | Far | 0.6442 | 0.7266 | 0.8287 | 0.5863 |
| TIN | Near | 0.5918 | 0.7379 | 0.8261 | 0.5966 |
| Near macro | Near | 0.6128 | 0.7343 | 0.7902 | 0.6419 |
| Far macro | Far | 0.6052 | 0.7499 | 0.6205 | 0.8298 |

#### Energy gap

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | Near | 0.9273 | 0.3693 | 0.4342 | 0.4277 |
| MNIST | Far | 0.9539 | 0.1067 | 0.1016 | 0.7414 |
| Places365 | Far | 0.9642 | 0.2669 | 0.1453 | 0.6628 |
| SVHN | Far | 0.9804 | 0.0799 | 0.1530 | 0.5455 |
| Textures | Far | 0.9757 | 0.1647 | 0.4496 | 0.2436 |
| TIN | Near | 0.9458 | 0.2499 | 0.4708 | 0.2897 |
| Near macro | Near | 0.9366 | 0.3096 | 0.4525 | 0.3587 |
| Far macro | Far | 0.9686 | 0.1545 | 0.2124 | 0.5483 |

#### Centered-logit L2 distance

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | Near | 0.9551 | 0.3986 | 0.4224 | 0.4435 |
| MNIST | Far | 0.8482 | 0.4475 | 0.1899 | 0.8377 |
| Places365 | Far | 0.9782 | 0.2824 | 0.1431 | 0.6676 |
| SVHN | Far | 0.9990 | 0.1795 | 0.1558 | 0.5797 |
| Textures | Far | 0.9981 | 0.2387 | 0.4590 | 0.2608 |
| TIN | Near | 0.9700 | 0.3662 | 0.4978 | 0.3313 |
| Near macro | Near | 0.9626 | 0.3824 | 0.4601 | 0.3874 |
| Far macro | Far | 0.9559 | 0.2870 | 0.2369 | 0.5864 |

#### Max probability difference

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | Near | 0.9380 | 0.7060 | 0.6202 | 0.6999 |
| MNIST | Far | 0.9846 | 0.3867 | 0.0913 | 0.8732 |
| Places365 | Far | 0.9863 | 0.6470 | 0.2625 | 0.8740 |
| SVHN | Far | 0.9980 | 0.3371 | 0.1868 | 0.7005 |
| Textures | Far | 0.9949 | 0.5026 | 0.5666 | 0.4348 |
| TIN | Near | 0.9583 | 0.6651 | 0.6588 | 0.5829 |
| Near macro | Near | 0.9482 | 0.6855 | 0.6395 | 0.6414 |
| Far macro | Far | 0.9909 | 0.4684 | 0.2768 | 0.7206 |

#### Student energy

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | Near | 0.6992 | 0.7487 | 0.7450 | 0.7230 |
| MNIST | Far | 0.3918 | 0.8653 | 0.6851 | 0.9714 |
| Places365 | Far | 0.5746 | 0.8070 | 0.6123 | 0.9241 |
| SVHN | Far | 0.2557 | 0.9370 | 0.8855 | 0.9727 |
| Textures | Far | 0.5474 | 0.8576 | 0.9012 | 0.7954 |
| TIN | Near | 0.5079 | 0.8287 | 0.8813 | 0.7345 |
| Near macro | Near | 0.6036 | 0.7887 | 0.8132 | 0.7288 |
| Far macro | Far | 0.4424 | 0.8667 | 0.7710 | 0.9159 |

#### Student MSP

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | Near | 0.6602 | 0.7569 | 0.7624 | 0.7332 |
| MNIST | Far | 0.4582 | 0.8341 | 0.6253 | 0.9666 |
| Places365 | Far | 0.5711 | 0.7920 | 0.6034 | 0.9165 |
| SVHN | Far | 0.3599 | 0.8943 | 0.8238 | 0.9507 |
| Textures | Far | 0.5924 | 0.8143 | 0.8759 | 0.7233 |
| TIN | Near | 0.5078 | 0.8189 | 0.8766 | 0.7228 |
| Near macro | Near | 0.5840 | 0.7879 | 0.8195 | 0.7280 |
| Far macro | Far | 0.4954 | 0.8337 | 0.7321 | 0.8893 |

#### Teacher-student KL divergence

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | Near | 0.5911 | 0.7702 | 0.7888 | 0.7155 |
| MNIST | Far | 0.6038 | 0.7426 | 0.4741 | 0.9258 |
| Places365 | Far | 0.5182 | 0.8141 | 0.6508 | 0.9259 |
| SVHN | Far | 0.4282 | 0.8823 | 0.7812 | 0.9433 |
| Textures | Far | 0.5964 | 0.8016 | 0.8702 | 0.6798 |
| TIN | Near | 0.4914 | 0.8040 | 0.8725 | 0.6668 |
| Near macro | Near | 0.5413 | 0.7871 | 0.8307 | 0.6912 |
| Far macro | Far | 0.5367 | 0.8102 | 0.6941 | 0.8687 |

#### Teacher energy

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | Near | 0.5967 | 0.7932 | 0.7971 | 0.7666 |
| MNIST | Far | 0.5534 | 0.7800 | 0.5278 | 0.9469 |
| Places365 | Far | 0.5538 | 0.8015 | 0.6224 | 0.9168 |
| SVHN | Far | 0.5211 | 0.8155 | 0.7011 | 0.9029 |
| Textures | Far | 0.6049 | 0.7890 | 0.8637 | 0.6573 |
| TIN | Near | 0.4834 | 0.8350 | 0.8863 | 0.7351 |
| Near macro | Near | 0.5401 | 0.8141 | 0.8417 | 0.7508 |
| Far macro | Far | 0.5583 | 0.7965 | 0.6787 | 0.8560 |

#### Teacher MSP

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | Near | 0.5933 | 0.7901 | 0.7962 | 0.7646 |
| MNIST | Far | 0.5559 | 0.7763 | 0.5213 | 0.9482 |
| Places365 | Far | 0.5446 | 0.8018 | 0.6261 | 0.9181 |
| SVHN | Far | 0.5446 | 0.7948 | 0.6758 | 0.8902 |
| Textures | Far | 0.6031 | 0.7808 | 0.8598 | 0.6410 |
| TIN | Near | 0.4837 | 0.8325 | 0.8852 | 0.7351 |
| Near macro | Near | 0.5385 | 0.8113 | 0.8407 | 0.7498 |
| Far macro | Far | 0.5620 | 0.7884 | 0.6707 | 0.8494 |

### 5. Cifar10 feature denoising resnet18 layer2 mask p020 10 draw

- ID dataset: CIFAR-10
- Experiment: `feature_denoising_channel_residual_layer2_resnet18_cifar10_mask_p020`
- Checkpoint: `best`
- Distillation method: `feature_denoising`
- Inference mode: `10_draw_reconstruction`

#### Feature-reconstruction absolute improvement

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-100 | Near | 0.9529 | 0.5406 | 0.5185 | 0.5512 |
| MNIST | Far | 0.6519 | 0.7530 | 0.4274 | 0.9510 |
| Places365 | Far | 0.9009 | 0.5899 | 0.2649 | 0.8431 |
| SVHN | Far | 0.2112 | 0.9576 | 0.8967 | 0.9837 |
| Textures | Far | 0.9930 | 0.6465 | 0.6609 | 0.6169 |
| TIN | Near | 0.8957 | 0.5986 | 0.6222 | 0.5527 |
| Near macro | Near | 0.9243 | 0.5696 | 0.5703 | 0.5519 |
| Far macro | Far | 0.6892 | 0.7368 | 0.5625 | 0.8486 |

#### Negative feature-reconstruction error

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-100 | Near | 0.9597 | 0.5100 | 0.4938 | 0.5359 |
| MNIST | Far | 0.8520 | 0.4842 | 0.1964 | 0.8467 |
| Places365 | Far | 0.9579 | 0.5453 | 0.2152 | 0.8293 |
| SVHN | Far | 0.9993 | 0.0870 | 0.1458 | 0.5468 |
| Textures | Far | 0.9971 | 0.4430 | 0.5361 | 0.4711 |
| TIN | Near | 0.9591 | 0.4800 | 0.5184 | 0.4573 |
| Near macro | Near | 0.9594 | 0.4950 | 0.5061 | 0.4966 |
| Far macro | Far | 0.9516 | 0.3899 | 0.2734 | 0.6735 |

#### Feature-reconstruction relative improvement

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-100 | Near | 0.9492 | 0.5594 | 0.5298 | 0.5862 |
| MNIST | Far | 0.4491 | 0.8098 | 0.6243 | 0.9490 |
| Places365 | Far | 0.9312 | 0.6628 | 0.2800 | 0.8886 |
| SVHN | Far | 0.9979 | 0.3061 | 0.1773 | 0.6591 |
| Textures | Far | 0.9940 | 0.5816 | 0.6089 | 0.5992 |
| TIN | Near | 0.9382 | 0.5798 | 0.5845 | 0.5608 |
| Near macro | Near | 0.9437 | 0.5696 | 0.5572 | 0.5735 |
| Far macro | Far | 0.8431 | 0.5900 | 0.4227 | 0.7740 |

### 6. Cifar10 feature denoising resnet18 layer3 mask p020 10 draw

- ID dataset: CIFAR-10
- Experiment: `feature_denoising_channel_residual_layer3_resnet18_cifar10_mask_p020`
- Checkpoint: `best`
- Distillation method: `feature_denoising`
- Inference mode: `10_draw_reconstruction`

#### Feature-reconstruction absolute improvement

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-100 | Near | 0.5477 | 0.8401 | 0.8497 | 0.8210 |
| MNIST | Far | 0.3092 | 0.8804 | 0.7531 | 0.9728 |
| Places365 | Far | 0.3298 | 0.9050 | 0.8045 | 0.9667 |
| SVHN | Far | 0.1229 | 0.9745 | 0.9451 | 0.9900 |
| Textures | Far | 0.3461 | 0.9298 | 0.9486 | 0.9031 |
| TIN | Near | 0.3737 | 0.8954 | 0.9175 | 0.8612 |
| Near macro | Near | 0.4607 | 0.8678 | 0.8836 | 0.8411 |
| Far macro | Far | 0.2770 | 0.9224 | 0.8628 | 0.9581 |

#### Negative feature-reconstruction error

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-100 | Near | 0.9899 | 0.3383 | 0.3959 | 0.3987 |
| MNIST | Far | 0.9589 | 0.3205 | 0.0892 | 0.8095 |
| Places365 | Far | 0.9919 | 0.2518 | 0.1310 | 0.6641 |
| SVHN | Far | 1.0000 | 0.0393 | 0.1427 | 0.5358 |
| Textures | Far | 1.0000 | 0.2294 | 0.4532 | 0.3060 |
| TIN | Near | 0.9911 | 0.2693 | 0.4022 | 0.3324 |
| Near macro | Near | 0.9905 | 0.3038 | 0.3991 | 0.3656 |
| Far macro | Far | 0.9877 | 0.2103 | 0.2040 | 0.5788 |

#### Feature-reconstruction relative improvement

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-100 | Near | 0.5706 | 0.8162 | 0.8277 | 0.7866 |
| MNIST | Far | 0.3562 | 0.8752 | 0.7151 | 0.9731 |
| Places365 | Far | 0.4927 | 0.8483 | 0.6785 | 0.9444 |
| SVHN | Far | 0.7780 | 0.6726 | 0.4688 | 0.8298 |
| Textures | Far | 0.6271 | 0.8231 | 0.8690 | 0.7681 |
| TIN | Near | 0.4932 | 0.8466 | 0.8783 | 0.7941 |
| Near macro | Near | 0.5319 | 0.8314 | 0.8530 | 0.7903 |
| Far macro | Far | 0.5635 | 0.8048 | 0.6828 | 0.8789 |

### 7. Cifar10 feature denoising resnet18 layer4 mask p020 10 draw

- ID dataset: CIFAR-10
- Experiment: `feature_denoising_channel_residual_layer4_resnet18_cifar10_mask_p020`
- Checkpoint: `best`
- Distillation method: `feature_denoising`
- Inference mode: `10_draw_reconstruction`

#### Feature-reconstruction absolute improvement

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-100 | Near | 0.7210 | 0.7724 | 0.7684 | 0.7637 |
| MNIST | Far | 0.1003 | 0.9797 | 0.9152 | 0.9970 |
| Places365 | Far | 0.6476 | 0.8033 | 0.5658 | 0.9332 |
| SVHN | Far | 0.4128 | 0.9121 | 0.8032 | 0.9660 |
| Textures | Far | 0.7933 | 0.7513 | 0.8024 | 0.6794 |
| TIN | Near | 0.7041 | 0.7780 | 0.8004 | 0.7383 |
| Near macro | Near | 0.7126 | 0.7752 | 0.7844 | 0.7510 |
| Far macro | Far | 0.4885 | 0.8616 | 0.7716 | 0.8939 |

#### Negative feature-reconstruction error

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-100 | Near | 0.6857 | 0.7459 | 0.7647 | 0.6981 |
| MNIST | Far | 0.7833 | 0.5756 | 0.2517 | 0.8837 |
| Places365 | Far | 0.6978 | 0.7309 | 0.4793 | 0.8884 |
| SVHN | Far | 0.8277 | 0.5705 | 0.3877 | 0.7569 |
| Textures | Far | 0.6920 | 0.7586 | 0.8321 | 0.6536 |
| TIN | Near | 0.6624 | 0.7569 | 0.7979 | 0.6734 |
| Near macro | Near | 0.6741 | 0.7514 | 0.7813 | 0.6858 |
| Far macro | Far | 0.7502 | 0.6589 | 0.4877 | 0.7956 |

#### Feature-reconstruction relative improvement

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-100 | Near | 0.5211 | 0.8543 | 0.8578 | 0.8307 |
| MNIST | Far | 0.2529 | 0.9373 | 0.7946 | 0.9888 |
| Places365 | Far | 0.4749 | 0.8639 | 0.6788 | 0.9489 |
| SVHN | Far | 0.4410 | 0.8705 | 0.7618 | 0.9351 |
| Textures | Far | 0.4811 | 0.8722 | 0.9120 | 0.7961 |
| TIN | Near | 0.4919 | 0.8624 | 0.8815 | 0.8135 |
| Near macro | Near | 0.5065 | 0.8583 | 0.8697 | 0.8221 |
| Far macro | Far | 0.4125 | 0.8860 | 0.7868 | 0.9172 |

### 8. Cifar100 feature denoising resnet18 layer2 mask p020 10 draw

- ID dataset: CIFAR-100
- Experiment: `feature_denoising_channel_residual_layer2_resnet18_cifar100_mask_p020`
- Checkpoint: `best`
- Distillation method: `feature_denoising`
- Inference mode: `10_draw_reconstruction`

#### Feature-reconstruction absolute improvement

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | Near | 0.8941 | 0.5531 | 0.5438 | 0.5553 |
| MNIST | Far | 0.3224 | 0.9163 | 0.7337 | 0.9857 |
| Places365 | Far | 0.8792 | 0.5709 | 0.2842 | 0.8157 |
| SVHN | Far | 0.3720 | 0.9136 | 0.8098 | 0.9637 |
| Textures | Far | 0.9986 | 0.4084 | 0.5262 | 0.3701 |
| TIN | Near | 0.8979 | 0.5360 | 0.6325 | 0.4351 |
| Near macro | Near | 0.8960 | 0.5446 | 0.5881 | 0.4952 |
| Far macro | Far | 0.6431 | 0.7023 | 0.5885 | 0.7838 |

#### Negative feature-reconstruction error

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | Near | 0.9473 | 0.4670 | 0.4630 | 0.4845 |
| MNIST | Far | 0.8800 | 0.4736 | 0.1714 | 0.8411 |
| Places365 | Far | 0.9370 | 0.5358 | 0.2352 | 0.7941 |
| SVHN | Far | 0.9984 | 0.0959 | 0.1464 | 0.5491 |
| Textures | Far | 0.9956 | 0.4886 | 0.5609 | 0.4881 |
| TIN | Near | 0.9462 | 0.4689 | 0.5707 | 0.3829 |
| Near macro | Near | 0.9468 | 0.4679 | 0.5168 | 0.4337 |
| Far macro | Far | 0.9527 | 0.3985 | 0.2785 | 0.6681 |

#### Feature-reconstruction relative improvement

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | Near | 0.9392 | 0.4895 | 0.4812 | 0.5019 |
| MNIST | Far | 0.3052 | 0.9005 | 0.7480 | 0.9773 |
| Places365 | Far | 0.9177 | 0.6029 | 0.2701 | 0.8392 |
| SVHN | Far | 0.9963 | 0.1884 | 0.1567 | 0.5796 |
| Textures | Far | 0.9992 | 0.4481 | 0.5356 | 0.4736 |
| TIN | Near | 0.9546 | 0.4866 | 0.5694 | 0.4055 |
| Near macro | Near | 0.9469 | 0.4881 | 0.5253 | 0.4537 |
| Far macro | Far | 0.8046 | 0.5350 | 0.4276 | 0.7174 |

### 9. Cifar100 feature denoising resnet18 layer3 mask p020 10 draw

- ID dataset: CIFAR-100
- Experiment: `feature_denoising_channel_residual_layer3_resnet18_cifar100_mask_p020`
- Checkpoint: `best`
- Distillation method: `feature_denoising`
- Inference mode: `10_draw_reconstruction`

#### Feature-reconstruction absolute improvement

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | Near | 0.9300 | 0.5052 | 0.4970 | 0.5163 |
| MNIST | Far | 0.0531 | 0.9881 | 0.9483 | 0.9981 |
| Places365 | Far | 0.8358 | 0.6163 | 0.3435 | 0.8372 |
| SVHN | Far | 0.3528 | 0.9278 | 0.8293 | 0.9715 |
| Textures | Far | 0.9892 | 0.6126 | 0.6436 | 0.5622 |
| TIN | Near | 0.8924 | 0.5562 | 0.6500 | 0.4582 |
| Near macro | Near | 0.9112 | 0.5307 | 0.5735 | 0.4873 |
| Far macro | Far | 0.5577 | 0.7862 | 0.6912 | 0.8423 |

#### Negative feature-reconstruction error

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | Near | 0.9436 | 0.4798 | 0.4736 | 0.4934 |
| MNIST | Far | 0.9916 | 0.0664 | 0.0615 | 0.7337 |
| Places365 | Far | 0.9513 | 0.4390 | 0.1943 | 0.7391 |
| SVHN | Far | 0.9987 | 0.1190 | 0.1488 | 0.5564 |
| Textures | Far | 0.9953 | 0.4476 | 0.5396 | 0.4629 |
| TIN | Near | 0.9490 | 0.4571 | 0.5613 | 0.3757 |
| Near macro | Near | 0.9463 | 0.4685 | 0.5175 | 0.4345 |
| Far macro | Far | 0.9842 | 0.2680 | 0.2360 | 0.6230 |

#### Feature-reconstruction relative improvement

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | Near | 0.9318 | 0.4663 | 0.4712 | 0.4812 |
| MNIST | Far | 0.3246 | 0.9216 | 0.7275 | 0.9863 |
| Places365 | Far | 0.9120 | 0.5723 | 0.2603 | 0.8213 |
| SVHN | Far | 0.9792 | 0.5113 | 0.2451 | 0.7662 |
| Textures | Far | 0.9861 | 0.5945 | 0.6272 | 0.5816 |
| TIN | Near | 0.9222 | 0.5165 | 0.6047 | 0.4220 |
| Near macro | Near | 0.9270 | 0.4914 | 0.5379 | 0.4516 |
| Far macro | Far | 0.8005 | 0.6499 | 0.4650 | 0.7889 |

### 10. Cifar100 feature denoising resnet18 layer4 mask p020 10 draw

- ID dataset: CIFAR-100
- Experiment: `feature_denoising_channel_residual_layer4_resnet18_cifar100_mask_p020`
- Checkpoint: `best`
- Distillation method: `feature_denoising`
- Inference mode: `10_draw_reconstruction`

#### Feature-reconstruction absolute improvement

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | Near | 0.6306 | 0.7694 | 0.7734 | 0.7455 |
| MNIST | Far | 0.6459 | 0.7159 | 0.4188 | 0.9303 |
| Places365 | Far | 0.6308 | 0.7578 | 0.5518 | 0.8986 |
| SVHN | Far | 0.5789 | 0.7844 | 0.6485 | 0.8921 |
| Textures | Far | 0.6956 | 0.7428 | 0.8253 | 0.6167 |
| TIN | Near | 0.5683 | 0.7801 | 0.8482 | 0.6546 |
| Near macro | Near | 0.5994 | 0.7747 | 0.8108 | 0.7001 |
| Far macro | Far | 0.6378 | 0.7502 | 0.6111 | 0.8344 |

#### Negative feature-reconstruction error

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | Near | 0.8066 | 0.6504 | 0.6563 | 0.6299 |
| MNIST | Far | 0.6076 | 0.7228 | 0.4758 | 0.9342 |
| Places365 | Far | 0.7321 | 0.7158 | 0.4651 | 0.8859 |
| SVHN | Far | 0.7579 | 0.6516 | 0.4843 | 0.8121 |
| Textures | Far | 0.7588 | 0.7383 | 0.8095 | 0.6331 |
| TIN | Near | 0.6682 | 0.7357 | 0.8125 | 0.6184 |
| Near macro | Near | 0.7374 | 0.6930 | 0.7344 | 0.6242 |
| Far macro | Far | 0.7141 | 0.7071 | 0.5587 | 0.8163 |

#### Feature-reconstruction relative improvement

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | Near | 0.6218 | 0.7808 | 0.7865 | 0.7550 |
| MNIST | Far | 0.5254 | 0.7716 | 0.5414 | 0.9450 |
| Places365 | Far | 0.5763 | 0.8071 | 0.6153 | 0.9257 |
| SVHN | Far | 0.5364 | 0.7974 | 0.6786 | 0.8878 |
| Textures | Far | 0.5611 | 0.8234 | 0.8854 | 0.7169 |
| TIN | Near | 0.4792 | 0.8320 | 0.8873 | 0.7348 |
| Near macro | Near | 0.5505 | 0.8064 | 0.8369 | 0.7449 |
| Far macro | Far | 0.5498 | 0.7999 | 0.6802 | 0.8689 |

### 11. Cifar10 feature denoising vit patch tokens layer12 mask p020 10 draw

- ID dataset: CIFAR-10
- Experiment: `feature_denoising_patch_token_residual_layer12_vit_base_patch16_224_cifar10_mask_p020`
- Checkpoint: `best`
- Distillation method: `feature_denoising`
- Inference mode: `10_draw_reconstruction`

#### Feature-reconstruction absolute improvement

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-100 | Near | 0.6727 | 0.8628 | 0.8310 | 0.8849 |
| MNIST | Far | 0.9201 | 0.4289 | 0.1156 | 0.8526 |
| Places365 | Far | 0.2152 | 0.9644 | 0.8702 | 0.9908 |
| SVHN | Far | 0.4296 | 0.9186 | 0.8011 | 0.9694 |
| Textures | Far | 0.0122 | 0.9956 | 0.9968 | 0.9940 |
| TIN | Near | 0.3861 | 0.9424 | 0.9379 | 0.9472 |
| Near macro | Near | 0.5294 | 0.9026 | 0.8844 | 0.9160 |
| Far macro | Far | 0.3943 | 0.8269 | 0.6959 | 0.9517 |

#### Feature denoising patch token residual reconstruction error

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-100 | Near | 0.8378 | 0.6803 | 0.6705 | 0.6791 |
| MNIST | Far | 0.7102 | 0.6730 | 0.3556 | 0.9218 |
| Places365 | Far | 0.7747 | 0.7282 | 0.4306 | 0.9072 |
| SVHN | Far | 0.6309 | 0.7943 | 0.6328 | 0.9038 |
| Textures | Far | 0.5320 | 0.8744 | 0.9107 | 0.8288 |
| TIN | Near | 0.7627 | 0.7404 | 0.7612 | 0.7153 |
| Near macro | Near | 0.8002 | 0.7104 | 0.7159 | 0.6972 |
| Far macro | Far | 0.6619 | 0.7675 | 0.5824 | 0.8904 |

#### Feature-reconstruction relative improvement

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-100 | Near | 0.5911 | 0.8958 | 0.8693 | 0.9127 |
| MNIST | Far | 0.7430 | 0.6920 | 0.3253 | 0.9376 |
| Places365 | Far | 0.1216 | 0.9774 | 0.9108 | 0.9942 |
| SVHN | Far | 0.2121 | 0.9597 | 0.9045 | 0.9841 |
| Textures | Far | 0.0052 | 0.9974 | 0.9982 | 0.9954 |
| TIN | Near | 0.2299 | 0.9629 | 0.9612 | 0.9644 |
| Near macro | Near | 0.4105 | 0.9293 | 0.9153 | 0.9385 |
| Far macro | Far | 0.2705 | 0.9066 | 0.7847 | 0.9779 |

### 12. Cifar100 feature denoising vit patch tokens layer12 mask p020 10 draw

- ID dataset: CIFAR-100
- Experiment: `feature_denoising_patch_token_residual_layer12_vit_base_patch16_224_cifar100_mask_p020`
- Checkpoint: `best`
- Distillation method: `feature_denoising`
- Inference mode: `10_draw_reconstruction`

#### Feature-reconstruction absolute improvement

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | Near | 0.8889 | 0.6289 | 0.5893 | 0.6557 |
| MNIST | Far | 0.9792 | 0.1482 | 0.0662 | 0.7534 |
| Places365 | Far | 0.7172 | 0.8224 | 0.5412 | 0.9461 |
| SVHN | Far | 0.8118 | 0.6445 | 0.4329 | 0.8257 |
| Textures | Far | 0.3404 | 0.9353 | 0.9542 | 0.9089 |
| TIN | Near | 0.8058 | 0.7721 | 0.7921 | 0.7434 |
| Near macro | Near | 0.8473 | 0.7005 | 0.6907 | 0.6995 |
| Far macro | Far | 0.7122 | 0.6376 | 0.4987 | 0.8585 |

#### Feature denoising patch token residual reconstruction error

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | Near | 0.9027 | 0.5747 | 0.5503 | 0.5941 |
| MNIST | Far | 0.8682 | 0.4521 | 0.1663 | 0.8461 |
| Places365 | Far | 0.9076 | 0.5381 | 0.2496 | 0.8110 |
| SVHN | Far | 0.7149 | 0.7169 | 0.5386 | 0.8587 |
| Textures | Far | 0.6452 | 0.8128 | 0.8701 | 0.7401 |
| TIN | Near | 0.8824 | 0.5862 | 0.6676 | 0.5036 |
| Near macro | Near | 0.8926 | 0.5804 | 0.6090 | 0.5488 |
| Far macro | Far | 0.7840 | 0.6300 | 0.4562 | 0.8140 |

#### Feature-reconstruction relative improvement

| Scope | Group | FPR95 ↓ | AUROC ↑ | AUPR-IN ↑ | AUPR-OUT ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | Near | 0.8644 | 0.6883 | 0.6410 | 0.7028 |
| MNIST | Far | 0.9432 | 0.2348 | 0.0880 | 0.7814 |
| Places365 | Far | 0.7260 | 0.8432 | 0.5574 | 0.9518 |
| SVHN | Far | 0.7380 | 0.7314 | 0.5370 | 0.8698 |
| Textures | Far | 0.3047 | 0.9312 | 0.9548 | 0.8825 |
| TIN | Near | 0.7877 | 0.7985 | 0.8171 | 0.7542 |
| Near macro | Near | 0.8261 | 0.7434 | 0.7291 | 0.7285 |
| Far macro | Far | 0.6780 | 0.6851 | 0.5343 | 0.8714 |

## Selection entries not evaluated

These entries were absent from the task array because their checkpoints were unavailable when the manifest was built.

| Variant | Reason |
| --- | --- |
| Cifar10 aggressive spatial clip resnet18 layer4 flatten kl 50 draw | missing metrics artifact: runs/students/perturbation/embedding/clipping/cifar_10/resnet18/linear_layer4_clip_spatial_aggressive/kl_divergence/metrics.json |
| Cifar100 aggressive spatial clip resnet18 layer4 flatten kl 50 draw | missing metrics artifact: runs/students/perturbation/embedding/clipping/cifar_100/resnet18/linear_layer4_clip_spatial_aggressive/kl_divergence/metrics.json |
