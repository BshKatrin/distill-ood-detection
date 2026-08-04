# Experiment Configs and Run Directories

Experiment configs are organized by artifact owner and experimental identity.
Every config declares an explicit `run_dir`; code must not derive storage from
`experiment_name`.

## MLflow policy

MLflow tracking is disabled for this project. All experiment configs must set
`mlflow.enabled: false`; config loading rejects `true`. Treat the explicit
`run_dir` artifacts (`resolved_config.json`, histories, metrics, checkpoints,
and inference outputs) as the authoritative experiment record.

## Directory structure

```text
configs/
  embedding_distances/
    <id_dataset>/
      <teacher_architecture>.yaml
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
    subspace_ensemble/
      <subspace_method>/
        <id_dataset>/
          <teacher_architecture>/
            <student_variant>.yaml
```

Embedding-distance configs select any subset of the four post-GAP ResNet stage
outputs (`layer1` through `layer4`), exact k-NN selection block sizes, and the
pooled-embedding shard size. Each configured layer selects neighbors
independently. The selected ID
neighbors' mean probabilities and logits produce the classifier-based OOD
Scores documented under [OOD Scores](ood_scores/README.md#k-nn-output-aggregation).
They write to
`runs/embedding_distances/<id_dataset>/<teacher_architecture>/`.

Current perturbation levels are `embedding` and `pixel`. Embedding
methods currently include `clipping`, `dropout`, and `pca`.
Pixel methods include affine `pixel_augmentation` and `pixmix`.
Current Feature Denoising methods include `pca_masking`, `feature_masking`,
`spatial_token_prediction`, and `pixel_masked_embedding`.
Activation-subspace components are `decisive` and `insignificant`.
Their explicit targets are `projected_logits` or `coordinates`;
coordinate-target students use the `autoencoder` kind.
Subspace-ensemble methods are `channel_flatten`, `channel_gap`, and `pca_gap`.

Examples:

- `configs/teachers/cifar_10/resnet18.yaml`
- `configs/students/baseline/cifar_10/resnet18/linear.yaml`
- `configs/students/perturbation/embedding/clipping/cifar_10/resnet18/linear_layer4_clip_channel.yaml`
- `configs/students/feature_denoising/pca_masking/cifar_10/resnet18/linear_layer4_pca9_mask_p030.yaml`
- `configs/students/activation_subspace/decisive/cifar_10/resnet18/linear.yaml`
- `configs/students/subspace_ensemble/channel_gap/cifar_10/resnet18/linear.yaml`

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
Feature Denoising spatial block residual configs set `student.input_shape` to
the teacher feature-map shape with one additional mask channel and set
`student.num_classes` to the teacher channel count. The configured
`spatial_mask_block_sizes` are square side lengths sampled uniformly per
example.
Feature Denoising confusion-channel replacement configs use an unmodified
four-dimensional teacher feature-map shape. They save validation-derived
class statistics and confusion pairs plus training-subset activation exemplars
in `<run_dir>/class_channel_corruption.pt`. The first launch configs cover
CIFAR-10 and CIFAR-100 ID datasets with a ResNet-18 teacher at `layer4`.

Subspace-ensemble configs describe one 16-member ensemble and may enable both
`mse_logits` and `kl_divergence`. Both objectives reuse the exact selections
stored in `<run_dir>/subspace_ensemble.pt`. The `ordered` and `partitioned`
assignment modes are deterministic and disjoint. The per-member dimension is
derived as the teacher output dimension divided by the ensemble size, and
config loading rejects a non-zero remainder. For the 512-dimensional,
16-student configs, `student.input_shape` is 512 for 32 flattened `layer4`
channels and 32 for channel-GAP or whitened GAP-PCA inputs.
