# NMF Channel-Cluster Audit

This audit trains one residual reconstruction student for every eligible NMF
channel cluster at layers 2–4 for CIFAR-10 and CIFAR-100.

## Task definition

Two dataset-level configs under `configs/channel_cluster_audit/` expand into a
stable manifest. Cluster IDs follow dendrogram leaf order and are never sorted
by OOD performance.

- Whole-cluster specialists mask their fixed cluster in every training and
  inference example. Inference uses one deterministic draw.
- Fractional specialists sample, without replacement, the nearest integer to
  25% of their fixed cluster on every example. Clusters smaller than four
  channels are excluded. Inference stores the mean over ten independent draws.

The current hierarchy cuts are 0.7 for layer 2 and 0.6 for layers 3 and 4.
Together they produce 148 whole-cluster and 121 fractional specialists: 269
tasks in total. Every student trains for 50 epochs with seed 42.

ResNet-50 is not part of the specialist audit. Its NMF experiment uses one
generic student per layer and ID dataset, with hierarchy cuts of 0.6 for layers
2 and 3 and 0.55 for layer 4. Each student masks the nearest integer to 25% of
the channels in every cluster per example and averages ten inference draws.
Every layer uses the prior ResNet-50 residual CNN width of 64 and batch size
128.

## Artifacts and metrics

Each specialist writes its standard checkpoint, history, and Feature Denoising
score artifacts below:

```text
runs/channel_cluster_audit/nmf_latent_cosine/<id_dataset>/resnet18/
  <variant>/<layer>/<cluster_id>/
```

The score artifacts contain raw reconstruction error, identity error, absolute
improvement, and relative improvement. Raw error is oriented as
`-raw_reconstruction_error`; both improvements are already higher-is-ID.

Shared sample metadata is exported once per dataset split. It records true
classes, teacher-predicted ID classes, teacher confidence, and duplicate labels
used to validate specialist-score alignment.

The summary job validates all artifacts and writes compact Parquet files for
cluster metrics, distribution summaries, class-conditioned metrics, task
metadata, and training curves. Point estimates include ROC-AUC, FPR@95,
distribution mean/median/standard deviation/IQR, and signed
`median(ID) - median(OOD)`. MNIST and SVHN also have an unweighted far-OOD
macro; the opposite CIFAR dataset is near-OOD.

## Dashboard

The CPU-only Panel application currently has one minimal tab, **Scores per
cluster**. Its only controls select the ID dataset, masking variant, and score
input. Layer 2–4 plots are stacked vertically in hierarchy order. Each OOD
dataset has a stable color; a wide translucent bar shows ROC-AUC and a narrower
hatched bar shows FPR@95. The application reads only the compact cluster-metric
and task-metadata Parquet summaries.

Submit the full dependency chain with:

```bash
bash slurm_scripts/submit_channel_cluster_audit.sh
```

The script regenerates the manifest, retries only incomplete tasks, summarizes
after the array succeeds, and then starts an eight-hour Panel job on port 5006.
The dashboard binds to the compute node's cluster-facing interfaces while its
WebSocket origin remains restricted to the local forwarded port. After finding
the node with `squeue`, connect through the cluster login host:

```bash
ssh -N -L 5006:<compute-node>:5006 <cluster-login>
```

Then open `http://localhost:5006/channel_cluster_audit` locally. The server
accepts only `localhost:5006` as its WebSocket origin.

Exact commands for this dashboard and the global-student distribution dashboard
are collected in [Panel dashboards on SLURM](../../../hpc/panel-dashboards.md).
