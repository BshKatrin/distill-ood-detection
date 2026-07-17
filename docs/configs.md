# Experiment Configs and Run Directories

Experiment configs are organized by artifact owner and experimental identity.
Every config declares an explicit `run_dir`; code must not derive storage from
`experiment_name`.

## Directory structure

```text
configs/
  teachers/
    <id_dataset>/
      <teacher_architecture>.yaml
  students/
    baseline/
      <id_dataset>/
        <teacher_architecture>/
          <student_variant>.yaml
    perturbation/
      <perturbation_level>/
        <perturbation_method>/
          <id_dataset>/
            <teacher_architecture>/
              <student_variant>.yaml
    feature_denoising/
      <feature_denoising_method>/
        <id_dataset>/
          <teacher_architecture>/
            <student_variant>.yaml
    activation_subspace/
      <component>/
        <id_dataset>/
          <teacher_architecture>/
            <student_variant>.yaml
```

Current perturbation levels are `embedding` and `pixel`. Embedding
methods currently include `clipping`, `dropout`, and `pca`.
Pixel methods include affine `pixel_augmentation` and `pixmix`.
Current Feature Denoising methods include `pca_masking`, `feature_masking`,
`spatial_token_prediction`, and `pixel_masked_embedding`.
Activation-subspace components are `decisive` and `insignificant`.
Their explicit targets are `projected_logits` or `coordinates`;
coordinate-target students use the `autoencoder` kind.

Examples:

- `configs/teachers/cifar_10/resnet18.yaml`
- `configs/students/baseline/cifar_10/resnet18/linear.yaml`
- `configs/students/perturbation/embedding/clipping/cifar_10/resnet18/linear_layer4_clip_channel.yaml`
- `configs/students/feature_denoising/pca_masking/cifar_10/resnet18/linear_layer4_pca9_mask_p030.yaml`
- `configs/students/activation_subspace/decisive/cifar_10/resnet18/linear.yaml`

When adding a new strategy, perturbation level, or method, update the matching
[strategy documentation](strategies/README.md).

## Run directory mapping

The `run_dir` value mirrors the config path under `runs/`, without the `.yaml`
suffix:

```yaml
experiment_name: perturbation_linear_layer4_student_resnet18_cifar10_clip_channel
run_dir: runs/students/perturbation/embedding/clipping/cifar_10/resnet18/linear_layer4_clip_channel
```

`experiment_name` is a globally unique display and tracking label used by
MLflow and manifests. `run_dir` is the authoritative artifact location. Do not
construct paths by combining `runs/` with `experiment_name`.

## Teacher artifacts

Teacher configs can export activations and probabilities into the same teacher
run directory:

```text
runs/teachers/<id_dataset>/<teacher_architecture>/
  teacher_activations/
    manifest.json
    <dataset_name>/<layer>.pt
  teacher_probabilities/
    manifest.json
    <dataset_name>/probabilities.pt
```

Activation configs include a top-level `layers` list. Probability export uses
the same config and ignores `layers`.

## Student artifacts

Student training and inference share the configured run directory:

```text
<run_dir>/
  resolved_config.json
  summary.json
  <distillation_method>/
  probabilities/
```

Perturbation inference separates deterministic and stochastic modes:

```text
<run_dir>/probabilities/unperturbed/
<run_dir>/probabilities/perturbed/
```

Clipping configs explicitly set `method: clipping`, select `embedding_pool` as
`avg` or `flatten`, and define a non-empty `clipping_layers` mapping. Each
configured ResNet layer has its own `clipping_mode`, `u_min`, and `u_max`.
Clipping configs do not use `student.feature_layer`; the student representation
always comes from the final propagated `layer4` activation.

PCA configs set `student.input_shape` to the component count and point
`strategy.perturbation.pca_activation_path` to a complete ID training-split
teacher activation under `runs/teachers/`.
Masked PCA configs set `student.input_shape` to the PCA component count plus
the PCA component count because the student receives both the masked PCA
projection and the binary PCA-component keep mask.
Feature Denoising PCA masked-reconstruction configs set `student.input_shape` and
`student.num_classes` to the PCA component count because the student receives
only the masked PCA projection and reconstructs the clean PCA projection.
Feature Denoising spatial token prediction configs set `student.input_shape` to the teacher
feature-map shape. The student receives padded visible tokens plus spatial
position indices, not a zero-masked dense feature map.
