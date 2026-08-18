# Top-Activation Profile Correlation

This method builds a channel hierarchy independently for every teacher feature
layer from the complete official ID training split.

## Channel profile

For sample `i`, channel `c`, and layer `l`, let `F[i,c,l]` be the spatial
activation map. Flatten its `H * W` spatial values and retain

```text
k = ceil(0.10 * H * W)
```

values with the largest activation. The scalar profile value is their mean:

```text
g[i,c,l] = mean(top-k(F[i,c,l]))
```

The profile for channel `c` is the vector of `g[i,c,l]` over all ID training
samples. For the CIFAR ResNet-18 layers, `k` is 103, 26, 7, and 2 for
`layer1` through `layer4`, respectively.

## Standardization and correlation

For each channel independently, compute the profile mean and population
standard deviation over the complete official ID training split. Standardize
with those training statistics:

```text
z[i,c,l] = (g[i,c,l] - mean[c,l]) / std[c,l]
```

The channel similarity matrix is the Pearson correlation between standardized
profiles from the same layer. The clustering distance is:

```text
d(c1, c2) = 1 - Pearson(z[:,c1,l], z[:,c2,l])
```

The explicit standardization makes the fitted training statistics reusable
and makes the correlation a normalized matrix product. A channel with profile
standard deviation at or below `standardization_epsilon` is recorded as a
constant channel. Its off-diagonal correlation is conservatively set to zero,
so it attaches to the hierarchy at distance 1 instead of creating NaNs.

## Hierarchy and exploratory cuts

Hierarchical clustering uses average linkage. Average linkage is compatible
with the precomputed correlation distance; Ward linkage is not used because it
optimizes Euclidean within-cluster variance.

The hierarchy is the primary grouping artifact because it represents groups
at every possible cut. Initial configs also export exploratory flat cuts at
distances `0.1`, `0.2`, `0.3`, and `0.5`. These correspond to merge-level
correlations of `0.9`, `0.8`, `0.7`, and `0.5`, but average-linkage clusters do
not require every pair of channels inside a cluster to meet that correlation.
Choose a final cut only after inspecting cluster stability and downstream
masking behavior.

## Artifacts

For each layer, `<run_dir>/channel_groups/` contains:

- `<layer>.pt`: profile mean/std, constant-channel indices, correlation and
  distance matrices, linkage matrix, leaf order, and flat-cut groups;
- `<layer>_groups.json`: lightweight flat groups and leaf order for direct use
  by later masking code;
- `<layer>_dendrogram_overview.png`: fixed-size overview without crowded leaf
  labels;
- `<layer>_dendrogram_labeled.pdf`: vector dendrogram with channel indices for
  zoomed inspection.

`manifest.json` records the dataset, teacher, split, profile definition,
standardization, clustering settings, and artifact paths. Raw activations and
per-sample profiles are deliberately not saved. For all four ResNet-18 layers,
the saved numeric matrices occupy only a few megabytes, so the complete
`channel_groups/` folder is suitable for cluster-to-Mac synchronization.

## Running

Configs are under:

```text
configs/channel_grouping/top_activation_correlation/<id_dataset>/resnet18.yaml
```

Run one config directly:

```bash
distill-ood build-channel-groups --config configs/channel_grouping/top_activation_correlation/cifar_10/resnet18.yaml
```

The recommended cluster workflow is documented in
[SLURM jobs](../../../hpc/slurm-jobs.md#feature-denoising-channel-grouping).
