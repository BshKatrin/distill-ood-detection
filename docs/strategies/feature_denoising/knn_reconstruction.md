# Channel-Masked k-NN Reconstruction

See [Feature Masked Reconstruction](feature_masking.md) for shared feature-value,
student, and hidden-element MSE definitions.

## Channel-masked k-NN reconstruction

`channel_masked_knn_reconstruction` and
`channel_group_masked_knn_reconstruction` replace the learned student with
exact k-NN regression over clean ID training activation maps. Each reference
and query remains a pre-GAP map with shape `(C, H, W)`. For every independently
sampled mask, squared L2 distance is calculated only over unmasked channels:

```text
distance(q, x; m) = sum(m * (q - x)^2)
```

Search uses batched GPU matrix multiplication and chunks the reference bank;
it does not use FAISS because each query has a different channel subspace. The
reference channelwise squared norms are precomputed, and each search keeps only
the running global top-k candidates. The hidden-channel prediction is the mean
of the selected neighbors' clean maps. Visible query channels are copied
unchanged. No student is trained or checkpointed.

The channel-group variant samples one complete hierarchy-cut cluster uniformly
per example and draw. It uses `channel_group_path` and
`channel_group_distance_threshold` with the same validation and sampling rules
as the learned channel-group residual method. It retains the sampled group IDs
alongside neighbor indices and distances.

The reference artifact must be the complete clean ID training split exported
by `export-teacher-activations`, with the same feature layer and map shape as
the query. Score artifacts retain neighbor indices, masked squared distances,
and visible-channel counts for every evaluation draw.

## Configuration

Channel-masked k-NN example:

```yaml
student:
  kind: knn
  feature_layer: layer3
  input_shape: [256, 8, 8]
  num_classes: 256

strategy:
  name: feature_denoising
  feature_denoising:
    method: channel_masked_knn_reconstruction
    activation_path: runs/teachers/cifar_10/resnet18/teacher_activations/cifar10_train/layer3.pt
    mask_probability: 0.2
    evaluation_draws: 10
    k_neighbors: 10
    knn_query_batch_size: 128
    knn_reference_chunk_size: 50000
```

Run score export directly; the method is inference-only:

```bash
distill-ood export-feature-denoising-scores --config <config>
```

Channel-group-masked k-NN changes the method and adds the hierarchy artifact:

```yaml
strategy:
  name: feature_denoising
  feature_denoising:
    method: channel_group_masked_knn_reconstruction
    activation_path: runs/teachers/cifar_10/resnet18/teacher_activations/cifar10_train/layer3.pt
    channel_group_path: runs/channel_grouping/top_activation_correlation/cifar_10/resnet18/channel_groups/layer3.pt
    channel_group_distance_threshold: 0.5
    evaluation_draws: 10
    k_neighbors: 10
```
