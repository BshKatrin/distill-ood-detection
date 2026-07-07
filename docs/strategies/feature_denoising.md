# Feature Denoising Strategy

The Feature Denoising strategy trains a student to predict teacher
representations instead of teacher logits, teacher probabilities, or image
pixels. The teacher is frozen. Student targets are clean teacher embeddings or
clean projections of teacher embeddings, treated as stop-gradient targets.

The first implemented method is
[PCA Masked Reconstruction](feature_denoising/pca_masking.md). It masks part of a whitened
teacher PCA embedding and trains the student to reconstruct the clean full
whitened PCA embedding. The reconstruction error becomes the mismatch signal
for OOD detection.

The v2 methods are [Feature Masked Reconstruction](feature_denoising/feature_masking.md):
spatial masking and channel masking over normalized teacher feature maps.

## Configuration

Set `strategy.name: feature_denoising` and select the method with
`strategy.feature_denoising.method: pca_masked_reconstruction`.

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

For PCA Masked Reconstruction, `student.input_shape` is
`pca_components` because the student receives only the masked PCA values.
Set `student.num_classes` to the same value so the student outputs one
reconstruction value per PCA component.

## OOD Score

Reconstruction error is larger for samples that are less predictable from ID
teacher-feature structure. Since this project uses higher OOD Score values for
more ID-like samples, this score uses `sign: -1`.

## Implementation

Feature Denoising training is implemented by `distill-ood train-student`.
Reconstruction-score export is implemented by `distill-ood export-feature-denoising-scores`.
