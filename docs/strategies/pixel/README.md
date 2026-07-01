# Pixel-Space Perturbations

Pixel-space perturbations modify the input image before the teacher extracts an
embedding. This page will define the shared training and inference behavior
for raw-pixel augmentation experiments.

## Methods

- [Pixel augmentation](pixel_augmentation.md): Mild affine and photometric
  jitter applied to normalized image tensors by unnormalizing, transforming in
  pixel space, and re-normalizing before teacher inference.

## Shared behavior

Pixel-space methods run the teacher on perturbed images up to the configured
student feature layer. The student receives the flattened teacher embedding
concatenated with method-specific transformation parameters. Probability
inference can draw multiple perturbations per image and stores one logit and
probability vector per draw.
