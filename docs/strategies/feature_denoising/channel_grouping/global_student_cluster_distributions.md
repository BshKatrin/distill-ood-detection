# Global-Student Cluster Distributions

This audit decomposes the existing global NMF student's reconstruction behavior
by channel cluster. It does not train specialists and does not mask one cluster
in isolation. Every inference draw reproduces the student's training
corruption: the configured fraction is sampled independently in every eligible
cluster for every image.

For each masked cluster, sample, and draw, the exporter computes reconstruction
and identity MSE over that cluster's masked channel maps. It retains:

```text
raw reconstruction error = reconstruction MSE on masked channels
absolute improvement = identity error - reconstruction error
relative improvement = absolute improvement / max(identity error, 1e-12)
```

Ten draw-level values are averaged per image. Identity error is not included in
the dashboard summary. The summary stores Tukey boxplot statistics
for ID, MNIST, SVHN, and the opposite CIFAR dataset in stable hierarchy order.
It also stores per-cluster ROC-AUC and FPR@95 for each OOD dataset. Absolute
and relative improvement are treated as higher-is-ID; raw reconstruction error
is negated before its OOD metrics are calculated.

The Panel application has one page with three controls: teacher model
(ResNet-18 or ResNet-50), ID dataset, and score. Layer 2–4 plots are
stacked vertically. Each cluster has four dataset-colored boxes, a zero
reference line, and hover metadata for distribution and cluster statistics.

Cluster export, summary, server, and SSH-tunnel commands are documented in
[Panel dashboards on SLURM](../../../hpc/panel-dashboards.md).
