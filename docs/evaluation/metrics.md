# Aggregate OOD detection metrics

The [OOD Score](ood-scores.md) is a per-sample ID-confidence value: larger means
more ID-like. A metric additionally specifies labels, operating point, benchmark
splits, and aggregation. Score sign alone does not determine FPR@95.

## Positive-class conventions

| Convention | Positive class | ROC input | Meaning of FPR@95 |
| --- | --- | --- | --- |
| Historical project evaluation (ID+) | ID = 1, OOD = 0 | ID confidence | Fraction of OOD accepted as ID at at least 95% ID true-positive rate |
| OpenOOD comparison evaluation (OOD+) | OOD = 1, ID = 0 | Negated ID confidence | Fraction of ID flagged as OOD at at least 95% OOD true-positive rate |

The historical implementation is
[`ood_detection_metrics`](../../src/distill_ood_detection/evaluation/ood_metrics.py)
and returns `roc_auc` and `fpr_at_95_tpr`. The OpenOOD comparison implementation is
[`openood_metrics`](../../src/distill_ood_detection/evaluation/openood.py)
and returns `auroc`, `fpr95`, `aupr_in`, and `aupr_out`. Both select the minimum
FPR among empirical ROC points with TPR at least `0.95`.

Reversing labels and score direction together preserves AUROC, but fixes the
95% recall operating point for a different population. The two FPR@95 values
cannot be substituted for one another. Label tables **ID+** or **OOD+** and keep
their numerical values unchanged when comparing historical results.

Here OOD+ names the project's OpenOOD comparison evaluator; it is not a blanket
claim about every OpenOOD release or publication. The
[final-report convention audit](../../reports/final-report/06_openood_evaluation.tex)
records the code/version/spreadsheet distinctions behind the comparison.

## Units and precision-recall

Stored ROC-AUC/AUROC, FPR@95/FPR95, and AUPR values are fractions in `[0, 1]`.
Multiply by 100 only when a table explicitly labels percentages. Higher AUROC
and AUPR are better; lower FPR@95 is better.

`AUPR-IN` treats ID as positive and uses ID confidence. `AUPR-OUT` treats OOD
as positive and negates ID confidence. The OpenOOD evaluator computes the area
under the precision-recall curve using trapezoidal integration, rather than
average precision. Record the dataset populations because AUPR depends on
class prevalence.

## Aggregation

- Dataset-level metrics are authoritative. Compute each ID/OOD comparison first.
- OpenOOD evaluation averages dataset metrics arithmetically within each Near-OOD
  or Far-OOD group; it does not pool score vectors across datasets.
- The current probability-report exporter uses a group-balanced macro:
  `0.5 * mean(Near-OOD) + 0.5 * mean(Far-OOD)`. See the
  [report workflow](../reports/ood-metrics.md#group-balanced-macro-metrics).
- Some historical exports, including activation-subspace score manifests, use
  an arithmetic mean across all configured OOD datasets. Preserve those values
  and identify their aggregation explicitly; they are not group-balanced macros.

Record the config, checkpoint, inference mode/draw count, benchmark manifests,
score definition/sign, positive class, units, and aggregation with every curated
comparison. Recompute only from matching saved scores and label a new export
explicitly; do not silently replace historical results.
