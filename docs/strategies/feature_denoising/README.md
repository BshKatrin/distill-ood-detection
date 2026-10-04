# Feature Denoising Methods

Feature Denoising methods train students to predict clean teacher representations
from partial or corrupted teacher representations. These methods are separate
from logits and probability distillation objectives.

## Feature values and scores

ResNet feature-map reconstruction uses raw teacher values. PCA reconstruction
uses whitened PCA coordinates; pixel embedding prediction uses the clean
teacher embedding defined by its method. The method-specific target and score
contract is authoritative. Reconstruction error normally receives `sign: -1`
so higher OOD Scores remain more ID-like.

## Methods

Channel groups used by masking strategies are documented separately under
[Channel Grouping Methods](channel_grouping/README.md).

| Method | Configuration value | Student input | Target |
| --- | --- | --- | --- |
| [PCA Masked Reconstruction](pca_masking.md) | `pca_masked_reconstruction` | Masked whitened PCA embedding | Clean whitened PCA embedding |
| [Feature Masked Reconstruction](spatial_masking.md) | `spatial_masked_reconstruction` | Spatially masked raw feature map | Clean raw feature map |
| [Feature Masked Reconstruction](spatial_masking.md) | `spatial_block_residual_reconstruction` | Mean-filled feature map + hidden-position mask | Residual correction for the clean raw feature map |
| [Feature Masked Reconstruction](channel_masking.md) | `channel_masked_reconstruction` | Channel-masked raw feature map | Clean raw feature map |
| [Feature Masked Reconstruction](channel_masking.md) | `channel_masked_residual_reconstruction` | Channel-masked raw feature map | Residual correction for the clean raw feature map |
| [Feature Masked Reconstruction](grouped_channel_masking.md) | `channel_group_masked_residual_reconstruction` | One hierarchy-cut channel group zeroed in the raw feature map | Residual correction for the clean raw feature map |
| [Feature Masked Reconstruction](grouped_channel_masking.md) | `channel_group_stratified_masked_residual_reconstruction` | A sampled channel subset from every sufficiently large hierarchy-cut group | Residual correction for the clean raw feature map |
| [NMF Concept-Masked Reconstruction](nmf_concept_masking.md) | `nmf_concept_masked_residual_reconstruction` | Low-rank feature map with complete NMF concept directions removed | Residual correction for the clean low-rank feature map |
| [Feature Masked Reconstruction](knn_reconstruction.md) | `channel_masked_knn_reconstruction` | Unmasked raw feature-map channels | Mean hidden channels from exact ID neighbors |
| [Feature Masked Reconstruction](knn_reconstruction.md) | `channel_group_masked_knn_reconstruction` | Raw feature map with one hierarchy-cut channel group hidden | Mean hidden group channels from exact ID neighbors |
| [Feature Masked Reconstruction](confusion_channel_replacement.md) | `confusion_channel_replacement_residual_reconstruction` | Confusing-class channel-replaced raw feature map | Residual correction for the clean raw feature map |
| [Confusion-Channel Replacement](confusion_channel_replacement.md) | `confusion_channel_replacement_reconstruction` | Confusing-class channel-replaced raw feature map | Clean raw feature map |
| [Spatial Token Prediction](spatial_token_prediction.md) | `spatial_token_prediction` | Visible raw feature tokens and target positions | Target raw feature tokens |
| [ViT Patch-Token Masked Reconstruction](patch_token_masking.md) | `patch_token_masked_residual_reconstruction` | Zero-masked ViT patch-embedding grid | Residual correction for the original patch grid |
| [Pixel-Masked Embedding Prediction](pixel_masked_embedding.md) | `pixel_masked_embedding_prediction` | Pooled embedding from pixel-masked image | Pooled embedding from clean image |
| [Pixel-Augmented Embedding Prediction](pixel_augmented_embedding.md) | `pixel_augmented_embedding_prediction` | Pooled embedding from pixel-augmented image | Pooled embedding from clean image |
| Pixel-Masked Multilayer Prediction | `pixel_masked_multilayer_prediction` | Pooled layer3 + layer4 from pixel-masked image | Clean pooled layer3 + layer4 + logits |
| Pixel-Masked Multilayer L234 Prediction | `pixel_masked_multilayer_l234_prediction` | Pooled layer2 + layer3 + layer4 from pixel-masked image | Clean pooled layer2 + layer3 + layer4 + logits |

PCA Masked Reconstruction predicts clean whitened coordinates. Spatial and
channel masked reconstruction predict clean feature maps. Spatial block residual
reconstruction adds contiguous mean-filled blocks, an explicit mask channel, and
a residual CNN. Spatial Token Prediction avoids passing hidden feature values
to the student. Pixel-Masked Embedding Prediction tests whether
corrupted image views can predict clean teacher embeddings.

## Implementation

Feature Denoising training is implemented by `distill-ood train-student`.
Reconstruction-score export is implemented by `distill-ood export-feature-denoising-scores`.
The k-NN methods are inference-only and do not train or load a student.

## PCA configuration example

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

## Classifier activation subspaces

For pooled `layer4` embedding-prediction variants, reconstruction diagnostics
can be decomposed using the SVD of the teacher's linear classification head.
The complete right-singular basis is split into decisive and insignificant
components. Following ActSub, the decisive dimension `k` minimizes the absolute
difference between the mean component L2 norms over clean ID training targets.

Calculate exact per-draw component errors from an existing student checkpoint:

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" uv run --project envs/gpu --no-sync \
  python -m distill_ood_detection.cli export-feature-denoising-subspace-errors --config <config>
```

The command makes one clean teacher pass over the ID training split to select
`k`, then exports per-sample reconstruction error, identity error,
absolute improvement, and relative improvement for both subspaces under
`<run_dir>/feature_denoising_subspace_errors/`. It performs inference only and
does not train or update the student.

It does not save the temporary training embeddings. This classifier-weight
decomposition is not applied to intermediate ResNet layers because their
mapping to the classifier is nonlinear.
