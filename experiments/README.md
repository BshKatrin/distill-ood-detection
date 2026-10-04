# Experiment catalogue

These folders contain curated experiment narratives, result JSON, and supporting
figures. The [final report](../reports/final-report/main.pdf) summarises their
progression; [method references](../docs/strategies/README.md) define reusable
behaviour and [evaluation documentation](../docs/evaluation/README.md) defines
scores, metric conventions, and benchmark splits.

## Final-report evidence

Configuration links below are entry points to each experiment family, not a
claim that every row used the current version of a config. Match the record's
variant and checkpoint with its run's `resolved_config.json` for exact
reproduction. Raw run artifacts remain under the explicit `run_dir`.

| Report topic | Experiment record or numeric table | Configuration entry point |
| --- | --- | --- |
| Baseline output distillation | [Baseline table](../reports/outputs/latex/metrics_baseline.tex), [random-forest sweep](../reports/outputs/latex/metrics_random_forest_sweep.tex) | [Baseline students](../configs/students/baseline/) |
| Embedding clipping | [Layers 3/4](embedding_clipping/layer3_layer4_clipping.md), [layers 1–4](embedding_clipping/layer1_layer2_layer3_layer4_clipping.md) | [Clipping](../configs/students/perturbation/embedding/clipping/) |
| Dropout and PCA perturbations | [Dropout](../reports/outputs/latex/metrics_perturbation_dropout_cifar10.tex), [unmasked PCA](../reports/outputs/latex/metrics_unmasked_pca_resnet18_unperturbed.tex), [masked PCA](../reports/outputs/latex/metrics_masked_pca_fixed_unperturbed.tex) | [Embedding perturbations](../configs/students/perturbation/embedding/) |
| Affine pixel perturbation | [Clean inference](../reports/outputs/latex/metrics_pixel_augmentation_resnet50_unperturbed.tex), [perturbed inference](../reports/outputs/latex/metrics_pixel_augmentation_resnet50_perturbed.tex) | [Pixel augmentation](../configs/students/perturbation/pixel/augmentation/) |
| PixMix | [ResNet-50 CIFAR experiments](pixel_perturbation/pixmix_resnet50_cifar10_cifar100.md) | [PixMix](../configs/students/perturbation/pixel/pixmix/) |
| Channel and PCA subspace ensembles | [Channel ensembles](subspace_ensemble/channel_subspaces_resnet18_cifar10_cifar100.md), [PCA ensembles](subspace_ensemble/pca_subspaces_resnet18_cifar10_cifar100.md) | [Subspace ensembles](../configs/students/subspace_ensemble/) |
| Activation-subspace students | [ResNet-18 results](activation_subspace/resnet18_cifar10_cifar100.md) | [Activation subspaces](../configs/students/activation_subspace/) |
| Feature-map reconstruction | [Channel residual](feature_denoising/channel_residual_resnet18_resnet50.md), [spatial blocks](feature_denoising/spatial_block_residual_resnet18.md) | [Feature Denoising](../configs/students/feature_denoising/) |
| Channel grouping and NMF concepts | [Group masking](feature_denoising/channel_group_masking_d050_resnet18.md), [NMF concepts](feature_denoising/nmf_concept_masking_resnet18.md) | [Channel grouping](../configs/channel_grouping/), [Feature Denoising](../configs/students/feature_denoising/) |
| Confusion-channel replacement | [Direct and residual results](feature_denoising/confusion_channel_replacement_ood_scores.md) | [Feature Denoising](../configs/students/feature_denoising/) |
| k-NN output prediction and reconstruction | [Output scores](near_ood_detection/knn_faiss_ood_scores.md), [hidden-channel prediction](feature_denoising/knn_channel_masking_resnet18.md) | [Embedding distances](../configs/embedding_distances/), [Feature Denoising](../configs/students/feature_denoising/) |
| Pixel embedding prediction and layer fusion | [Pixel-masked embeddings](feature_denoising/pixel_masked_embedding_cifar10_resnet18.md), [layer-3/4 fusion](feature_denoising/layer3_layer4_absolute_improvement_composition.md) | [Feature Denoising](../configs/students/feature_denoising/) |
| Fixed OpenOOD CIFAR evaluation, including ViT | [Complete metrics](openood_cifar/openood_v1_5_selected_variants.md), [JSON](openood_cifar/openood_v1_5_selected_variants.json), [ViT CIFAR-10 MSP](openood_cifar/vit_cifar10_msp.json), [ViT CIFAR-100 MSP](openood_cifar/vit_cifar100_msp.json) | [Selected variants](../configs/evaluation/openood_cifar_v1_5/selected_variants.yaml) |
| ImageNet transfer | [Clipping](openood_imagenet200/clipping.md), [affine perturbation](openood_imagenet200/affine_pixel_perturbation.md), [Feature Denoising](openood_imagenet200/feature_denoising.md) | [ImageNet-200 teacher](../configs/teachers/image_net_200/resnet18_openood_seed0.yaml), [ImageNet-1K teacher](../configs/teachers/image_net_1k/resnet50_torchvision_v1.yaml) |

## Interpret historical comparisons

Historical CIFAR tables generally use ID-positive FPR@95. The OpenOOD selected
variant record explicitly uses OOD-positive FPR95; the final report presents
both conventions. Keep score direction, positive class, units, inference draws,
dataset populations, and averaging explicit. A historical all-dataset macro
and a Near/Far-balanced macro are different aggregations.

Do not replace historical tables or infer a missing experiment from another
method. Preserve the numerical record and create a separately labelled export
when changing benchmark or metric conventions.

## Add a curated result

Keep a short narrative beside its selected JSON and figures. Record the config,
resolved run directory, checkpoint, seed, inference mode, generating command,
benchmark manifests, score definition, metric convention, units, and aggregation.
Link reusable method explanations from `docs/` instead of duplicating them.
Working exports continue to use `reports/outputs/`; only the selected evidence
belongs here. Add a catalogue row when a result contributes to a deliverable.
