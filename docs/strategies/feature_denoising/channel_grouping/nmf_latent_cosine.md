# NMF Latent-Profile Cosine

This method builds channel hierarchies for ResNet-18 and ResNet-50 feature
layers from the complete official ID training split.

## NMF channel profiles

For each layer, post-ReLU activations are aligned as:

```text
(N, C, H, W) -> (N, H, W, C) -> V with shape (N * H * W, C)
```

A global 32-concept `MiniBatchNMF` model minimizes the unregularized Frobenius
residual `||V - S P||_F`. Fitting streams every spatial activation vector for
two deterministic passes with 65,536-row updates; raw activations are never
saved.

The fitted matrix `P` has shape `(32, C)`. Channel `c` is represented by column
`P[:, c]`, its non-negative profile across the learned concepts.

## Hierarchy

For channels `c1` and `c2`, the clustering distance is:

```text
d(c1, c2) = 1 - cosine(P[:, c1], P[:, c2])
```

Hierarchical clustering uses average linkage. The grouping job exports no flat
cuts; the dendrogram is inspected before selecting masking distances. Masking
jobs cut the saved linkage matrix at their configured distance, so the same
compact hierarchy artifact can support multiple experiments without rerunning
NMF.

## Artifacts

For every layer, `<run_dir>/channel_groups/` contains:

- `<layer>.pt`: fitted `p_matrix`, linkage matrix, leaf order, shape and fit
  provenance;
- `<layer>_dendrogram_overview.png`: fixed-size overview;
- `<layer>_dendrogram_labeled.pdf`: zoomable vector dendrogram labeled by
  channel index.

`manifest.json` records the shared NMF and clustering settings. No raw
activations, distance matrix, or flat-group JSON is saved.

Configs are under:

```text
configs/channel_grouping/nmf_latent_cosine/<id_dataset>/<backbone>.yaml
```

The first masking grid uses layer-specific cuts (`layer2: 0.7`, `layer3: 0.6`,
`layer4: 0.6`) and compares two corruptions: one uniformly sampled complete
cluster, or 25% of the channels sampled independently inside every cluster.
The latter includes all clusters, including singletons. Its configs set
`channel_group_min_size: 1` and therefore hide at least one channel per cluster
because mask counts are rounded to the nearest channel with a minimum of one.

ResNet-50 hierarchies are available for layers 2–4. Their cut distances are
selected only after inspecting the exported dendrograms. The later fractional
audit will exclude clusters smaller than four channels and retain the earlier
ResNet-50 residual-denoiser hidden width of 64.
