# Stratified Channel-Group Masking: Layer4

- ID dataset: CIFAR-10.
- Teacher: ResNet-18, layer4.
- Hierarchy cut distance: `0.5`.
- Eligible cluster size: at least 4 channels.
- Per-cluster mask fraction: `0.25`, rounded to the nearest channel.
- Eligible clusters: 36 of 36.
- Hidden channels per draw: 135 of 512.
- Student: trained residual CNN.
- Inference draws: 10.

Config:
`configs/students/feature_denoising/channel_group_stratified_masking/cifar_10/resnet18/cluster_layer4_d050_min4_p025.yaml`.

Cluster job: `409389` (submitted 2026-08-11).
