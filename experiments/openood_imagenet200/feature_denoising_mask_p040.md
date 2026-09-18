# Feature denoising with 40% channel masking on ImageNet-200

ResNet-18 teacher, residual CNN student, independent channel masking with
probability `0.4`, 10 masking draws, and 50 training epochs. Values are
`ROC-AUC / FPR@95`. The macro gives equal weight to the Near and Far group
means.

Student training used deterministic `Resize(256)` and `CenterCrop(224)`
preprocessing. Evaluation preprocessing matches OpenOOD.

| Layer | OOD score | SSB-hard | NINCO | iNaturalist | Textures | OpenImage-O | Near | Far | Macro |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| - | Teacher MSP | 0.804 / 0.726 | 0.862 / 0.648 | 0.931 / 0.398 | 0.883 / 0.496 | 0.892 / 0.564 | 0.833 / 0.687 | 0.902 / 0.486 | 0.867 / 0.586 |
| - | Teacher energy | 0.798 / 0.730 | 0.853 / 0.661 | 0.932 / 0.414 | 0.906 / 0.413 | 0.896 / 0.558 | 0.826 / 0.695 | 0.911 / 0.462 | 0.868 / 0.579 |
| layer2 | Raw reconstruction | 0.376 / 0.984 | 0.388 / 0.975 | 0.603 / 0.969 | 0.472 / 0.833 | 0.547 / 0.906 | 0.382 / 0.979 | 0.541 / 0.903 | 0.461 / 0.941 |
| layer2 | Absolute improvement | 0.633 / 0.895 | 0.620 / 0.841 | 0.451 / 0.983 | 0.616 / 0.765 | 0.494 / 0.913 | 0.626 / 0.868 | 0.521 / 0.887 | 0.574 / 0.877 |
| layer2 | Relative improvement | 0.476 / 0.966 | 0.469 / 0.955 | 0.661 / 0.943 | 0.633 / 0.716 | 0.589 / 0.867 | 0.473 / 0.961 | 0.628 / 0.842 | 0.550 / 0.901 |
| layer3 | Raw reconstruction | 0.352 / 0.984 | 0.325 / 0.984 | 0.336 / 0.992 | 0.417 / 0.838 | 0.461 / 0.942 | 0.338 / 0.984 | 0.405 / 0.924 | 0.372 / 0.954 |
| layer3 | Absolute improvement | 0.654 / 0.888 | 0.704 / 0.791 | 0.766 / 0.706 | 0.726 / 0.645 | 0.637 / 0.841 | 0.679 / 0.839 | 0.710 / 0.731 | 0.694 / 0.785 |
| layer3 | Relative improvement | 0.474 / 0.951 | 0.505 / 0.921 | 0.745 / 0.793 | 0.694 / 0.609 | 0.666 / 0.778 | 0.490 / 0.936 | 0.702 / 0.726 | 0.596 / 0.831 |
| layer4 | Raw reconstruction | 0.399 / 0.987 | 0.430 / 0.987 | 0.384 / 0.993 | 0.310 / 0.987 | 0.382 / 0.992 | 0.415 / 0.987 | 0.359 / 0.991 | 0.387 / 0.989 |
| layer4 | Absolute improvement | 0.730 / 0.788 | 0.767 / 0.757 | 0.870 / 0.571 | 0.906 / 0.377 | 0.850 / 0.608 | 0.749 / 0.772 | 0.875 / 0.519 | 0.812 / 0.646 |
| layer4 | Relative improvement | 0.652 / 0.930 | 0.742 / 0.896 | 0.845 / 0.788 | 0.878 / 0.551 | 0.815 / 0.810 | 0.697 / 0.913 | 0.846 / 0.716 | 0.771 / 0.815 |
