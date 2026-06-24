# Pixel-Space Perturbations

**Not implemented.**

Pixel-space perturbations modify the input image before the teacher extracts an
embedding. This page will define the shared training and inference behavior
once the first pixel-space method is specified.

Add one page per perturbation method to this directory. Each method page should
document:

- The transformation and its parameters
- Whether the teacher receives the clean or perturbed image
- The student input and teacher target
- Training and probability-inference behavior
- Configuration fields and example configs
- Implementation and artifact locations
