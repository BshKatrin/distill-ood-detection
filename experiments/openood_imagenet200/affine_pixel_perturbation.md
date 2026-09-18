# Affine pixel perturbation on ImageNet-200

ResNet-18 teacher, trained linear student, layer 4 flattened features, clean
teacher target, logit MSE, and clean inference. Values are `ROC-AUC / FPR@95`.
The macro gives equal weight to the Near and Far group means.

Student training used `Resize(256)` and `CenterCrop(224)`, not OpenOOD's random
resized crop and horizontal flip. Evaluation preprocessing matches OpenOOD.

| OOD score | SSB-hard | NINCO | iNaturalist | Textures | OpenImage-O | Near | Far | Macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Teacher MSP | 0.804 / 0.726 | 0.862 / 0.648 | 0.931 / 0.398 | 0.883 / 0.496 | 0.892 / 0.564 | 0.833 / 0.687 | 0.902 / 0.486 | 0.867 / 0.586 |
| Teacher energy | 0.798 / 0.730 | 0.853 / 0.661 | 0.932 / 0.414 | 0.906 / 0.413 | 0.896 / 0.558 | 0.826 / 0.695 | 0.911 / 0.462 | 0.868 / 0.579 |
| Max probability difference | 0.569 / 0.858 | 0.608 / 0.820 | 0.690 / 0.794 | 0.531 / 0.882 | 0.613 / 0.830 | 0.589 / 0.839 | 0.611 / 0.836 | 0.600 / 0.837 |
| Absolute max probability difference | 0.735 / 0.888 | 0.781 / 0.865 | 0.789 / 0.877 | 0.753 / 0.889 | 0.780 / 0.877 | 0.758 / 0.876 | 0.774 / 0.881 | 0.766 / 0.879 |
| KL divergence | 0.767 / 0.879 | 0.830 / 0.824 | 0.884 / 0.770 | 0.831 / 0.848 | 0.839 / 0.855 | 0.799 / 0.851 | 0.851 / 0.824 | 0.825 / 0.838 |
| Centered-logit L2 | 0.407 / 0.983 | 0.492 / 0.968 | 0.596 / 0.970 | 0.413 / 0.988 | 0.422 / 0.992 | 0.450 / 0.975 | 0.477 / 0.983 | 0.463 / 0.979 |
| Energy gap | 0.444 / 0.987 | 0.453 / 0.995 | 0.455 / 1.000 | 0.351 / 1.000 | 0.422 / 0.999 | 0.449 / 0.991 | 0.410 / 1.000 | 0.429 / 0.995 |
| Absolute energy gap | 0.345 / 0.987 | 0.339 / 0.996 | 0.302 / 1.000 | 0.230 / 0.998 | 0.283 / 0.999 | 0.342 / 0.992 | 0.272 / 0.999 | 0.307 / 0.995 |
| Student MSP | 0.798 / 0.728 | 0.855 / 0.665 | 0.922 / 0.443 | 0.883 / 0.491 | 0.886 / 0.577 | 0.827 / 0.697 | 0.897 / 0.504 | 0.862 / 0.600 |
| Student energy | 0.791 / 0.736 | 0.844 / 0.685 | 0.921 / 0.473 | 0.907 / 0.410 | 0.889 / 0.578 | 0.817 / 0.711 | 0.906 / 0.487 | 0.862 / 0.599 |
