# Feature Denoising Methods

Feature Denoising methods train students to predict clean teacher representations
from partial or corrupted teacher representations. These methods are separate
from logits and probability distillation objectives.

## Methods

Channel groups used by masking strategies are documented separately under
[Channel Grouping Methods](channel_grouping/README.md).

| Method | Configuration value | Student input | Target |
| --- | --- | --- | --- |
| [PCA Masked Reconstruction](pca_masking.md) | `pca_masked_reconstruction` | Masked whitened PCA embedding | Clean whitened PCA embedding |
| [Feature Masked Reconstruction](feature_masking.md) | `spatial_masked_reconstruction` | Spatially masked raw feature map | Clean raw feature map |
| [Feature Masked Reconstruction](feature_masking.md) | `spatial_block_residual_reconstruction` | Mean-filled feature map + hidden-position mask | Residual correction for the clean raw feature map |
| [Feature Masked Reconstruction](feature_masking.md) | `channel_masked_reconstruction` | Channel-masked raw feature map | Clean raw feature map |
| [Feature Masked Reconstruction](feature_masking.md) | `channel_masked_residual_reconstruction` | Channel-masked raw feature map | Residual correction for the clean raw feature map |
| [Feature Masked Reconstruction](feature_masking.md) | `channel_group_masked_residual_reconstruction` | One hierarchy-cut channel group zeroed in the raw feature map | Residual correction for the clean raw feature map |
| [Feature Masked Reconstruction](feature_masking.md) | `channel_group_stratified_masked_residual_reconstruction` | A sampled channel subset from every sufficiently large hierarchy-cut group | Residual correction for the clean raw feature map |
| [NMF Concept-Masked Reconstruction](nmf_concept_masking.md) | `nmf_concept_masked_residual_reconstruction` | Low-rank feature map with complete NMF concept directions removed | Residual correction for the clean low-rank feature map |
| [Feature Masked Reconstruction](feature_masking.md) | `channel_masked_knn_reconstruction` | Unmasked raw feature-map channels | Mean hidden channels from exact ID neighbors |
| [Feature Masked Reconstruction](feature_masking.md) | `channel_group_masked_knn_reconstruction` | Raw feature map with one hierarchy-cut channel group hidden | Mean hidden group channels from exact ID neighbors |
| [Feature Masked Reconstruction](feature_masking.md) | `confusion_channel_replacement_residual_reconstruction` | Confusing-class channel-replaced raw feature map | Residual correction for the clean raw feature map |
| [Spatial Token Prediction](spatial_token_prediction.md) | `spatial_token_prediction` | Visible raw feature tokens and target positions | Target raw feature tokens |
| [Pixel-Masked Embedding Prediction](pixel_masked_embedding.md) | `pixel_masked_embedding_prediction` | Pooled embedding from pixel-masked image | Pooled embedding from clean image |
| [Pixel-Augmented Embedding Prediction](pixel_augmented_embedding.md) | `pixel_augmented_embedding_prediction` | Pooled embedding from pixel-augmented image | Pooled embedding from clean image |
| Pixel-Masked Multilayer Prediction | `pixel_masked_multilayer_prediction` | Pooled layer3 + layer4 from pixel-masked image | Clean pooled layer3 + layer4 + logits |
| Pixel-Masked Multilayer L234 Prediction | `pixel_masked_multilayer_l234_prediction` | Pooled layer2 + layer3 + layer4 from pixel-masked image | Clean pooled layer2 + layer3 + layer4 + logits |

PCA Masked Reconstruction is the v1 method. Spatial and channel masked
feature reconstruction are v2 methods. Spatial block residual reconstruction
adds contiguous mean-filled blocks, an explicit mask channel, and a residual
CNN. Spatial Token Prediction is the v3 method that avoids passing hidden
feature values to the student. Pixel-Masked Embedding Prediction tests whether
corrupted image views can predict clean teacher embeddings.

## Implementation

Feature Denoising training is implemented by `distill-ood train-student`.
Reconstruction-score export is implemented by `distill-ood export-feature-denoising-scores`.
The k-NN methods are inference-only and do not train or load a student.
