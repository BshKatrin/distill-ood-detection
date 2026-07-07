# PCA Masked Reconstruction

PCA Masked Reconstruction is the first implemented Feature Denoising strategy. It
trains a student to reconstruct a clean whitened teacher PCA embedding from a
masked whitened teacher PCA embedding.

## Fit the projector

Fit PCA only from complete ID training-split teacher activations. Do not fit or
update the PCA basis from ID test or OOD activations.

Export teacher activations for the selected `student.feature_layer`, then set
`strategy.feature_denoising.pca_activation_path` to the complete ID training activation
artifact:

```yaml
strategy:
  name: feature_denoising
  feature_denoising:
    method: pca_masked_reconstruction
    pca_components: 9
    pca_mask_probability: 0.3
    pca_activation_path: runs/teachers/cifar_10/resnet18/teacher_activations/cifar10_train/layer4.pt
    evaluation_draws: 10
```

The raw projected teacher embedding is:

`z_pca = PCA(flatten(z), n_components)`

where `z` is the clean teacher embedding at `student.feature_layer`.

The Feature Denoising target uses whitened PCA coordinates:

`z_white_j = z_pca_j / std_train(z_pca_j)`

The component standard deviations are computed from the same complete ID
training activation artifact used to fit PCA. Whitening keeps high-variance PCA
components from dominating the reconstruction loss.

## Student input and target

For each sample, draw a binary keep mask over PCA components:

`m ~ Bernoulli(1 - p), shape: (pca_components,)`

where `p = strategy.feature_denoising.pca_mask_probability`.

The sampler forces at least one hidden component per sample so the
hidden-component reconstruction loss is always defined.

The student input is the masked whitened PCA embedding:

`z_white * m`

The keep mask is not concatenated to the student input. Hidden PCA components
are represented by zeros in the masked projection.

The target is the clean full whitened PCA embedding:

`stop_gradient(z_white)`

For v1, `student.input_shape` must be `pca_components` because the student
receives only the masked PCA values.
Set `student.num_classes` to `pca_components` so the student outputs one
reconstruction value per PCA component.

Start with a linear student as the simplest reconstruction baseline.

## Training loss

Compute reconstruction loss on hidden whitened PCA components only:

`loss = mse((z_hat - z_white) * (1 - m))`

Scoring only hidden components prevents the student from being rewarded for
copying visible PCA values.

## OOD Score

The OOD Score is `feature_denoising_pca_reconstruction_error`.

For one mask draw:

`raw_error = mean(((z_hat - z_white) ** 2) * (1 - m))`

For stochastic inference, draw `strategy.feature_denoising.evaluation_draws` independent
masks and average the raw errors. Convert the raw reconstruction error into the
project OOD Score convention with `sign: -1`:

`ood_score = -mean(raw_error_draw_1, ..., raw_error_draw_K)`

Higher scores are therefore more ID-like and lower scores are more OOD-like.

## Configuration

Example:

```yaml
strategy:
  name: feature_denoising
  feature_denoising:
    method: pca_masked_reconstruction
    pca_components: 9
    pca_mask_probability: 0.3
    pca_activation_path: runs/teachers/cifar_10/resnet18/teacher_activations/cifar10_train/layer4.pt
    evaluation_draws: 10
student:
  kind: linear
  feature_layer: layer4
  input_shape: [9]
  num_classes: 9
```

## Implementation

PCA Masked Reconstruction training is implemented by
`distill-ood train-student`. Reconstruction-score export is implemented by
`distill-ood export-feature-denoising-scores`.
