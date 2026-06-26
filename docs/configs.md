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
```

Current perturbation levels are `embedding` and the planned `pixel`. Embedding
methods currently include `clipping`, `dropout`, and `pca`.

Examples:

- `configs/teachers/cifar_10/resnet18.yaml`
- `configs/students/baseline/cifar_10/resnet18/linear.yaml`
- `configs/students/perturbation/embedding/clipping/cifar_10/resnet18/linear_layer4_clip_channel.yaml`

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

PCA configs set `student.input_shape` to the component count and point
`strategy.perturbation.pca_activation_path` to a complete ID training-split
teacher activation under `runs/teachers/`.
Masked PCA configs set `student.input_shape` to the PCA component count plus
the flattened embedding dimension because the student receives both the masked
PCA projection and the binary component-matrix-column keep mask.
