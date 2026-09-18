# Feature denoising on ImageNet-200

ResNet-18 teacher, residual CNN student, 20% uniform channel masking, 10
masking draws, and 50 training epochs. Values are `ROC-AUC / FPR@95`. The
macro gives equal weight to the Near and Far group means.

Student training used `Resize(256)` and `CenterCrop(224)`, not OpenOOD's random
resized crop and horizontal flip. Evaluation preprocessing matches OpenOOD.

| Layer | OOD score | SSB-hard | NINCO | iNaturalist | Textures | OpenImage-O | Near | Far | Macro |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| - | Teacher MSP | 0.804 / 0.726 | 0.862 / 0.648 | 0.931 / 0.398 | 0.883 / 0.496 | 0.892 / 0.564 | 0.833 / 0.687 | 0.902 / 0.486 | 0.867 / 0.586 |
| - | Teacher energy | 0.798 / 0.730 | 0.853 / 0.661 | 0.932 / 0.414 | 0.906 / 0.413 | 0.896 / 0.558 | 0.826 / 0.695 | 0.911 / 0.462 | 0.868 / 0.579 |
| layer2 | Raw reconstruction | 0.375 / 0.985 | 0.387 / 0.976 | 0.598 / 0.970 | 0.465 / 0.835 | 0.544 / 0.914 | 0.381 / 0.981 | 0.536 / 0.906 | 0.458 / 0.944 |
| layer2 | Absolute improvement | 0.628 / 0.896 | 0.617 / 0.843 | 0.446 / 0.984 | 0.608 / 0.782 | 0.489 / 0.915 | 0.623 / 0.869 | 0.515 / 0.894 | 0.569 / 0.882 |
| layer2 | Relative improvement | 0.472 / 0.968 | 0.466 / 0.958 | 0.629 / 0.946 | 0.604 / 0.737 | 0.570 / 0.891 | 0.469 / 0.963 | 0.601 / 0.858 | 0.535 / 0.910 |
| layer3 | Raw reconstruction | 0.346 / 0.985 | 0.322 / 0.985 | 0.320 / 0.993 | 0.401 / 0.854 | 0.448 / 0.945 | 0.334 / 0.985 | 0.390 / 0.931 | 0.362 / 0.958 |
| layer3 | Absolute improvement | 0.642 / 0.894 | 0.695 / 0.797 | 0.745 / 0.733 | 0.707 / 0.673 | 0.621 / 0.849 | 0.668 / 0.845 | 0.691 / 0.752 | 0.680 / 0.799 |
| layer3 | Relative improvement | 0.431 / 0.966 | 0.472 / 0.939 | 0.638 / 0.883 | 0.614 / 0.688 | 0.608 / 0.838 | 0.451 / 0.953 | 0.620 / 0.803 | 0.536 / 0.878 |
| layer4 | Raw reconstruction | 0.396 / 0.987 | 0.432 / 0.985 | 0.402 / 0.991 | 0.326 / 0.983 | 0.391 / 0.991 | 0.414 / 0.986 | 0.373 / 0.988 | 0.393 / 0.987 |
| layer4 | Absolute improvement | 0.722 / 0.797 | 0.761 / 0.755 | 0.869 / 0.556 | 0.908 / 0.367 | 0.847 / 0.607 | 0.742 / 0.776 | 0.875 / 0.510 | 0.808 / 0.643 |
| layer4 | Relative improvement | 0.632 / 0.927 | 0.731 / 0.883 | 0.853 / 0.724 | 0.887 / 0.517 | 0.816 / 0.790 | 0.682 / 0.905 | 0.852 / 0.677 | 0.767 / 0.791 |
