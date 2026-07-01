# OOD Score Tradeoff Plots

Use an OOD score tradeoff plot to compare one hyperparameter sweep for the same
student, strategy, ID dataset, OOD datasets, probability mode, and objective.

The plot uses:

- x-axis: FPR@95, lower is better.
- y-axis: ROC-AUC, higher is better.
- color: selected hyperparameter value.
- marker: OOD dataset.
- subplot: OOD score.
- figure: ID dataset.

The report command intentionally has no input, filter, or output defaults. Every
selection must be explicit so the comparison is reproducible and the plotted
runs are comparable.

## Command

```bash
uv run --project envs/notebooks --no-sync python reports/scripts/plot_ood_score_tradeoff.py \
  --json <report-json> \
  --hyperparameter <n-estimators|pca-components|pca-mask-probability> \
  --include-run-name <required-run-name-substring> \
  --architecture <architecture> \
  --feature-layer <feature-layer> \
  --probability-mode <probability-mode> \
  --method <objective-or-method> \
  --id-dataset <id-dataset> \
  --ood-dataset <ood-dataset> \
  --ood-score <ood-score> \
  --output-dir reports/outputs/plots \
  --output-stem <output-stem>
```

Pass repeated arguments multiple times:

- `--json` merges several JSON reports by `run_name` and metric identity.
- `--include-run-name` requires every passed substring to be present.
- `--exclude-run-name` excludes runs containing any passed substring.
- `--id-dataset` creates one figure per selected ID dataset.
- `--ood-dataset` creates one marker style per selected OOD dataset.
- `--ood-score` creates one subplot per selected OOD Score.

The command errors if an ID dataset has fewer than two selected hyperparameter
values or if the filters select duplicate points.

## Hyperparameter parsers

`n-estimators` reads `run["n_estimators"]` when present, then falls back to
`_n<integer>` in `run_name`.

`pca-components` reads `run["pca_components"]` when present, then falls back to
`_pca<integer>` in `run_name`.

`pca-mask-probability` reads `run["pca_mask_probability"]` when present, then
falls back to `_mask_p<ddd>` in `run_name`, where `p030` is plotted as `0.3`.
For this sweep, pass `--pca-components` to keep the component count fixed.

## Example: random forest `n_estimators`

```bash
uv run --project envs/notebooks --no-sync python reports/scripts/plot_ood_score_tradeoff.py \
  --json reports/outputs/json/random_forest_sweep.json \
  --json reports/outputs/json/random_forest_sweep_extra.json \
  --hyperparameter n-estimators \
  --include-run-name random_forest_layer4 \
  --architecture resnet18 \
  --feature-layer layer4 \
  --probability-mode unperturbed \
  --method logits \
  --id-dataset cifar10_test \
  --id-dataset cifar100_test \
  --ood-dataset mnist_test \
  --ood-dataset svhn_test \
  --ood-dataset cifar10_test \
  --ood-dataset cifar100_test \
  --ood-score max_probability_difference \
  --ood-score absolute_max_probability_difference \
  --ood-score student_teacher_kl_divergence \
  --ood-score logit_l2_distance \
  --ood-score energy_gap \
  --ood-score absolute_energy_gap \
  --ood-score student_msp \
  --ood-score student_energy \
  --output-dir reports/outputs/plots \
  --output-stem random_forest_layer4_n_estimators_tradeoff
```

The command writes:

- `reports/outputs/plots/random_forest_layer4_n_estimators_tradeoff_cifar10_test.png`
- `reports/outputs/plots/random_forest_layer4_n_estimators_tradeoff_cifar10_test.pdf`
- `reports/outputs/plots/random_forest_layer4_n_estimators_tradeoff_cifar100_test.png`
- `reports/outputs/plots/random_forest_layer4_n_estimators_tradeoff_cifar100_test.pdf`

## Example: PCA component count

PCA component metrics are currently split across several JSON reports. Merge
them explicitly and exclude masked PCA runs when plotting the unmasked PCA
component sweep.

```bash
uv run --project envs/notebooks --no-sync python reports/scripts/plot_ood_score_tradeoff.py \
  --json reports/outputs/json/pca_reduced_components.json \
  --json reports/outputs/json/perturbation_20260623_ood_metrics_cifar10.json \
  --json reports/outputs/json/perturbation_20260623_ood_metrics_cifar100.json \
  --hyperparameter pca-components \
  --include-run-name _pca \
  --exclude-run-name _mask_p \
  --architecture resnet18 \
  --feature-layer layer4 \
  --probability-mode unperturbed \
  --method cross_entropy \
  --id-dataset cifar10_test \
  --id-dataset cifar100_test \
  --ood-dataset mnist_test \
  --ood-dataset svhn_test \
  --ood-dataset cifar10_test \
  --ood-dataset cifar100_test \
  --ood-score max_probability_difference \
  --ood-score absolute_max_probability_difference \
  --ood-score student_teacher_kl_divergence \
  --ood-score logit_l2_distance \
  --ood-score energy_gap \
  --ood-score absolute_energy_gap \
  --ood-score student_msp \
  --ood-score student_energy \
  --output-dir reports/outputs/plots \
  --output-stem pca_components_cross_entropy_tradeoff
```

Run the same command with `--method kl_divergence` or `--method mse_logits` to
plot a different objective.

## Example: masked PCA probability

Use this parser only when the selected artifacts contain at least two mask
probabilities for the same architecture, feature layer, ID dataset, objective,
and PCA component count.

```bash
uv run --project envs/notebooks --no-sync python reports/scripts/plot_ood_score_tradeoff.py \
  --json reports/outputs/json/masked_pca_fixed_ood_metrics.json \
  --hyperparameter pca-mask-probability \
  --include-run-name _mask_p \
  --architecture resnet18 \
  --feature-layer layer4 \
  --pca-components 9 \
  --probability-mode perturbed \
  --method cross_entropy \
  --id-dataset cifar10_test \
  --ood-dataset mnist_test \
  --ood-dataset svhn_test \
  --ood-dataset cifar100_test \
  --ood-score max_probability_difference \
  --ood-score absolute_max_probability_difference \
  --ood-score student_teacher_kl_divergence \
  --ood-score logit_l2_distance \
  --ood-score energy_gap \
  --ood-score absolute_energy_gap \
  --ood-score student_msp \
  --ood-score student_energy \
  --output-dir reports/outputs/plots \
  --output-stem masked_pca_probability_cross_entropy_tradeoff
```

If the selected JSON only contains one mask probability, the command exits
instead of producing a misleading single-point sweep.
