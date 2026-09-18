# Aggressive spatial layer-4 clipping on ImageNet-200

ResNet-18 teacher, trained linear student, layer 4, spatial-dependent clipping,
clipping percentiles sampled from `Uniform(0.0, 0.5)`, flattened features,
clean teacher target, KL training, and 50 clipped inference draws. Student
training used deterministic `Resize(256)` and `CenterCrop(224)` preprocessing.
Values are `ROC-AUC / FPR@95`. The macro gives equal weight to the Near and Far
group means.

| OOD score | SSB-hard | NINCO | iNaturalist | Textures | OpenImage-O | Near | Far | Macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Teacher MSP | 0.804 / 0.726 | 0.862 / 0.648 | 0.931 / 0.398 | 0.883 / 0.496 | 0.892 / 0.564 | 0.833 / 0.687 | 0.902 / 0.486 | 0.867 / 0.586 |
| Teacher energy | 0.798 / 0.730 | 0.853 / 0.661 | 0.932 / 0.414 | 0.906 / 0.413 | 0.896 / 0.558 | 0.826 / 0.695 | 0.911 / 0.462 | 0.868 / 0.579 |
| Max probability difference | 0.802 / 0.728 | 0.861 / 0.650 | 0.930 / 0.403 | 0.881 / 0.497 | 0.891 / 0.568 | 0.831 / 0.689 | 0.901 / 0.489 | 0.866 / 0.589 |
| Absolute max probability difference | 0.198 / 0.994 | 0.139 / 0.999 | 0.070 / 1.000 | 0.119 / 0.998 | 0.109 / 1.000 | 0.169 / 0.997 | 0.099 / 0.999 | 0.134 / 0.998 |
| KL divergence | 0.209 / 0.992 | 0.148 / 0.996 | 0.069 / 0.999 | 0.117 / 0.992 | 0.108 / 0.999 | 0.179 / 0.994 | 0.098 / 0.997 | 0.138 / 0.995 |
| Centered-logit L2 | 0.321 / 0.989 | 0.293 / 0.993 | 0.237 / 0.998 | 0.162 / 0.997 | 0.236 / 0.997 | 0.307 / 0.991 | 0.212 / 0.998 | 0.259 / 0.994 |
| Energy gap | 0.797 / 0.734 | 0.852 / 0.664 | 0.931 / 0.423 | 0.905 / 0.420 | 0.895 / 0.563 | 0.825 / 0.699 | 0.910 / 0.469 | 0.868 / 0.584 |
| Absolute energy gap | 0.203 / 0.995 | 0.148 / 1.000 | 0.069 / 1.000 | 0.095 / 0.999 | 0.105 / 1.000 | 0.175 / 0.998 | 0.090 / 1.000 | 0.133 / 0.999 |
| Student MSP | 0.421 / 0.953 | 0.391 / 0.962 | 0.478 / 0.930 | 0.500 / 0.925 | 0.421 / 0.945 | 0.406 / 0.958 | 0.466 / 0.933 | 0.436 / 0.945 |
| Student energy | 0.429 / 0.953 | 0.404 / 0.954 | 0.484 / 0.930 | 0.487 / 0.929 | 0.424 / 0.943 | 0.417 / 0.953 | 0.465 / 0.934 | 0.441 / 0.944 |
