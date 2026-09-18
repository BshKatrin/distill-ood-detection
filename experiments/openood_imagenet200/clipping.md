# Multi-layer clipping on ImageNet-200

ResNet-18 teacher, trained linear student, layers 3 and 4, channel-dependent
clipping, flattened layer 4 features, clean teacher target, KL training, and
clean inference. Values are `ROC-AUC / FPR@95`. The macro gives equal weight
to the Near and Far group means.

Student training used `Resize(256)` and `CenterCrop(224)`, not OpenOOD's random
resized crop and horizontal flip. Evaluation preprocessing matches OpenOOD.

| OOD score | SSB-hard | NINCO | iNaturalist | Textures | OpenImage-O | Near | Far | Macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Teacher MSP | 0.804 / 0.726 | 0.862 / 0.648 | 0.931 / 0.398 | 0.883 / 0.496 | 0.892 / 0.564 | 0.833 / 0.687 | 0.902 / 0.486 | 0.867 / 0.586 |
| Teacher energy | 0.798 / 0.730 | 0.853 / 0.661 | 0.932 / 0.414 | 0.906 / 0.413 | 0.896 / 0.558 | 0.826 / 0.695 | 0.911 / 0.462 | 0.868 / 0.579 |
| Max probability difference | 0.724 / 0.800 | 0.739 / 0.767 | 0.800 / 0.728 | 0.710 / 0.813 | 0.764 / 0.766 | 0.731 / 0.784 | 0.758 / 0.769 | 0.744 / 0.776 |
| Absolute max probability difference | 0.772 / 0.809 | 0.815 / 0.775 | 0.846 / 0.742 | 0.793 / 0.818 | 0.824 / 0.778 | 0.794 / 0.792 | 0.821 / 0.779 | 0.807 / 0.786 |
| KL divergence | 0.783 / 0.854 | 0.841 / 0.786 | 0.879 / 0.814 | 0.819 / 0.894 | 0.842 / 0.862 | 0.812 / 0.820 | 0.847 / 0.857 | 0.829 / 0.838 |
| Centered-logit L2 | 0.310 / 0.992 | 0.303 / 0.992 | 0.200 / 1.000 | 0.155 / 0.999 | 0.245 / 0.998 | 0.307 / 0.992 | 0.200 / 0.999 | 0.253 / 0.995 |
| Energy gap | 0.288 / 0.992 | 0.233 / 0.997 | 0.136 / 1.000 | 0.118 / 1.000 | 0.184 / 1.000 | 0.261 / 0.995 | 0.146 / 1.000 | 0.203 / 0.997 |
| Absolute energy gap | 0.288 / 0.992 | 0.233 / 0.997 | 0.136 / 1.000 | 0.118 / 1.000 | 0.184 / 1.000 | 0.261 / 0.995 | 0.146 / 1.000 | 0.203 / 0.997 |
| Student MSP | 0.752 / 0.812 | 0.816 / 0.745 | 0.885 / 0.598 | 0.864 / 0.583 | 0.850 / 0.683 | 0.784 / 0.778 | 0.866 / 0.622 | 0.825 / 0.700 |
| Student energy | 0.770 / 0.760 | 0.830 / 0.695 | 0.919 / 0.450 | 0.913 / 0.389 | 0.877 / 0.605 | 0.800 / 0.728 | 0.903 / 0.481 | 0.852 / 0.604 |
