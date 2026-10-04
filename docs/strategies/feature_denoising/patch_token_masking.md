# ViT Patch-Token Masked Reconstruction

This experiment reconstructs randomly hidden ViT patch embeddings with a small
residual CNN. The frozen teacher is a CIFAR-finetuned ViT-B/16 operating on
224x224 resized images.

## Teacher features

Only the patch tokens from `layer12` are used. The CLS token is removed before
the feature tensor enters the reconstruction pipeline. For a 224x224 image and
16x16 patches, the 196 patch tokens are reshaped from `[B, 196, 768]` to
`[B, 768, 14, 14]`.

The source CIFAR images remain 32x32 on disk. The dataset transform resizes and
center-crops them to 224x224 to match the downloaded ViT checkpoints.

## Masking and student

Every spatial patch location is hidden independently with Bernoulli probability
`0.2`. At least one location per sample is forced hidden. A hidden token has all
768 coordinates replaced by zero; no mask channel is appended.

The `feature_residual_denoiser` student uses:

```text
Conv3x3(768 -> 64) + ReLU
Conv3x3(64 -> 64) + ReLU
Conv3x3(64 -> 768)
prediction = corrupted patch grid + predicted residual
```

The student has 922,496 trainable parameters. Reconstruction MSE is averaged
only over hidden patch coordinates.

## OOD score

Inference averages masked-patch reconstruction error over 10 independent mask
draws. The OOD Score is named
`feature_denoising_patch_token_residual_reconstruction_error` and has sign
`-1`, so larger signed values remain more ID-like.

The CIFAR-10 and CIFAR-100 variants are included in the fixed OpenOOD CIFAR
evaluation selection described in [OpenOOD CIFAR](../../evaluation/openood/cifar.md).
