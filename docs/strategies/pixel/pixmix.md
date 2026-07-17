# PixMix

The `pixmix` perturbation implements the CIFAR pipeline from *PixMix:
Dreamlike Pictures Comprehensively Improve Safety Measures*.

## Transformation

Each normalized input is unnormalized and resized to the configured PixMix
working size, which is 32 pixels for the CIFAR experiments. The clean target
view receives random horizontal flipping and a four-pixel zero-padded random
crop.

PixMix then:

1. Initializes from the clean target view or one random augmentation.
2. Samples a mixing-round count uniformly from zero through
   `mixing_iterations`.
3. In each round, mixes with either another augmentation of the clean view or
   one image from the external mixing set.
4. Selects additive or multiplicative mixing and samples the reference
   beta-distributed coefficients.
5. Clamps pixels to `[0, 1]` after each round.

The full operation set contains autocontrast, equalize, posterize, rotate,
solarize, shear, translation, color, contrast, brightness, and sharpness.

## Student and target

The student receives the teacher embedding from the PixMix image. It does not
receive the variable-length PixMix operation trace.

```text
student_input = pool(teacher_feature(x_pixmix))
```

`embedding_pool` supports:

- `avg`: global average pooling;
- `flatten`: flatten every feature-map value.

`teacher_target: clean` uses teacher logits from the crop/flip view before
PixMix. `teacher_target: perturbed` uses teacher logits from the PixMix image.

## Mixing set

The official Fractals + Feature Visualizations archive is an external data
asset and is not stored in this repository. Relative `mixing_set_path` values
are resolved under `dataset.data_dir`.

```yaml
strategy:
  name: perturbation
  perturbation:
    method: pixmix
    teacher_target: perturbed
    embedding_pool: avg
    evaluation_draws: 16
    pixmix:
      mixing_set_path: pixmix/fractals_and_fvis
      mixing_iterations: 4
      beta: 3.0
      augmentation_severity: 3.0
      all_ops: true
      working_size: 32
```

For ResNet-50 `layer4`, global average pooling uses
`student.input_shape: [2048]`; flattening uses `[32768]`.

## Feature Denoising

The existing `pixel_augmented_embedding_prediction` method selects PixMix with:

```yaml
feature_denoising:
  method: pixel_augmented_embedding_prediction
  pixel_augmentation_method: pixmix
  embedding_pool: avg
  pixmix:
    mixing_set_path: pixmix/fractals_and_fvis
```

The context is pooled from the PixMix image and the target is pooled from its
corresponding pre-PixMix crop/flip view.
