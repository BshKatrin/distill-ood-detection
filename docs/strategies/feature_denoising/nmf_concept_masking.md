# NMF Concept-Masked Reconstruction

NMF Concept-Masked Reconstruction removes complete learned directions from a
non-negative teacher feature space and trains the residual CNN student to
restore the clean low-rank projection.

## Global concept basis

Fit one global NMF reducer from the complete ID training-split activation
artifact. Activations must be post-ReLU maps with shape `(N, C, H, W)`. Spatial
vectors are aligned explicitly before fitting:

```text
(N, C, H, W) -> (N, H, W, C) -> V with shape (N * H * W, C)
```

`MiniBatchNMF` learns non-negative matrices `S` and `P` by minimizing the
paper's residual objective without regularization:

```text
U = V - S P
minimize ||U||_F, subject to S >= 0 and P >= 0
```

The fitted concept directions `P` have shape `(K, C)` and are saved in
`<run_dir>/nmf_concept_projector.pt`. For new activation maps, `P` remains
fixed and non-negative multiplicative updates solve for `S`. The reconstruction
reverses the exact spatial ordering above.

## Concept masking and student target

For every image, sample a Bernoulli keep mask over the `K` concept columns. The
same mask is applied at all `H * W` positions of that image, and at least one
concept is forced hidden:

```text
m ~ Bernoulli(1 - nmf_mask_probability), shape (K,)
S' = S * m
clean_projection = S P
corrupted_projection = S' P
```

The residual CNN receives the reshaped corrupted projection and predicts a
full feature-map correction:

```text
prediction = corrupted_projection + CNN(corrupted_projection)
loss = MSE(prediction, clean_projection) over all C * H * W values
```

The residual `U` is excluded from both input and target. Initial ResNet-18
layer4 configs use 32 global concepts, masking probability `0.2`, and 10
evaluation draws for CIFAR-10 and CIFAR-100.

## OOD outputs

For each sample, scoring averages the following values over evaluation draws:

```text
reconstruction_error = MSE(prediction, S P)
baseline_error = MSE(S' P, S P)
improvement = baseline_error - reconstruction_error
relative_improvement = improvement / max(baseline_error, 1e-12)
```

The exported artifact contains `raw_reconstruction_error`, its sign-adjusted
`scores`, `improvement`, and `relative_improvement`. The intermediate
`baseline_error` is used for the improvement calculations but is not exported.

## Configuration

```yaml
strategy:
  name: feature_denoising
  feature_denoising:
    method: nmf_concept_masked_residual_reconstruction
    nmf_components: 32
    nmf_mask_probability: 0.2
    nmf_activation_path: runs/teachers/cifar_10/resnet18/teacher_activations/cifar10_train/layer4.pt
    nmf_batch_size: 1024
    nmf_max_iter: 200
    nmf_encoding_max_iter: 200
    evaluation_draws: 10
student:
  kind: feature_residual_denoiser
  feature_layer: layer4
  input_shape: [512, 4, 4]
  num_classes: 512
```
