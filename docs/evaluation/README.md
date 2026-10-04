# Evaluation

Keep per-sample OOD Scores, aggregate metrics, and benchmark protocols separate.
Project scores always increase for more ID-like samples. Evaluators choose the
positive class and convert score direction explicitly.

- [OOD Scores](ood-scores.md): scalar definitions, signs, and student comparisons.
- [Aggregate metrics](metrics.md): ID-positive and OOD-positive FPR@95, units,
  AUPR, and averaging.
- [OpenOOD CIFAR](openood/cifar.md): fixed manifests and existing-student evaluation.
- [OpenOOD ImageNet](openood/imagenet.md): ImageNet-200 and ImageNet-1K protocols.
- [Metric report commands](../reports/ood-metrics.md): exports from saved artifacts.
- [Experiment records](../../experiments/README.md): results and their provenance.
