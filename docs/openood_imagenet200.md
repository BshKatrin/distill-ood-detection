# OpenOOD ImageNet evaluation

The repository supports the ImageNet-200 benchmark layout from OpenOOD v1.5.
Its manifests define the 200 ImageNet classes and all evaluation subsets; the
loader does not infer classes from directory names or evaluate entire source
datasets.

It also supports the corresponding ImageNet-1K protocol with torchvision's
ResNet-50 ImageNet-1K V1 checkpoint. The ImageNet-1K configs use
`benchmark_imglist/imagenet/{train,val,test}_imagenet.txt` and a 1,000-way
classifier. OpenOOD's five semantic OOD manifests are byte-identical between
its ImageNet-200 and ImageNet-1K protocols, so both use the same curated OOD
images and Near/Far grouping.

## Data layout

The cluster setup uses these read-only shared datasets:

- `/data/common/ImageNet` for the ImageNet-1K images selected by the ID manifests.
- `/data/common/dataset_suite/dtddataset/dtd/images` for the 5,160 DTD images
  selected by `test_texture.txt`.

The remaining assets live under `/home/bogush/openood`:

```text
openood/
  archives/
  data/
    benchmark_imglist/imagenet200/
    images_largescale/{ssb_hard,ninco,inaturalist,openimage_o}/
  results/
    imagenet200_resnet18_224x224_base_e90_lr0.1_default/s0/best.ckpt
```

Submit the restartable download job from the repository root:

```bash
mkdir -p /home/bogush/openood/logs
sbatch slurm_scripts/download_openood_imagenet200.sbatch /home/bogush/openood
```

The job downloads 6.71 GB of archives and validates every byte count and ZIP
archive before extraction. On clusters where compute nodes cannot reach Google
Drive, run it once with `OPENOOD_DOWNLOAD_ONLY=1` on the transfer/login side,
then submit the same script normally to validate and extract. It does not write
to `/data/common` and deliberately does not download a second copy of DTD.

## Fixed benchmark splits

The reference config is
`configs/teachers/image_net_200/resnet18_openood_seed0.yaml`. It uses manifest
rows as the source of truth:

- ID: `train_imagenet200.txt`, `val_imagenet200.txt`, and
  `test_imagenet200.txt`.
- Near-OOD: `test_ssb_hard.txt` and `test_ninco.txt`.
- Far-OOD: `test_inaturalist.txt`, `test_textures.txt`, and
  `test_openimage_o.txt`.

This distinction matters most for iNaturalist: the benchmark evaluates the
curated 10,000-image ImageNet-OOD subset named in the manifest, not the full
iNaturalist release. The same manifest-only rule applies to every OOD dataset.

`ImageListDataset` resolves OpenOOD relative paths against the configured image
root. Its optional basename lookup handles the cluster's class-directory
ImageNet validation layout without copying or relinking images.

## Models and variant scores

The teacher config loads OpenOOD's standard 224-pixel ResNet-18 architecture
with a 200-way classifier from the local seed-0 `best.ckpt`. Existing teacher,
student, activation, and probability exporters can use the same dataset config.
The ImageNet-1K reference config is
`configs/teachers/image_net_1k/resnet50_torchvision_v1.yaml`; it loads the
standard torchvision ResNet-50 architecture and the local V1 state dictionary.

New variants only need to supply a callable that maps a batch of images to one
ID-confidence value per image:

```python
id_scores = collect_confidence_scores(id_loader, score_variant, device)
ood_scores = {
    item.name: collect_confidence_scores(item.loader, score_variant, device)
    for item in ood_loaders
}
groups = {item.name: item.group for item in ood_loaders if item.group is not None}
results = evaluate_openood_scores(id_scores, ood_scores, groups)
```

Higher scores must mean more ID-like. The evaluator returns FPR95, AUROC,
AUPR-IN, and AUPR-OUT as fractions, plus arithmetic macro averages for the Near
and Far groups. Dataset-level results remain available, so a macro average never
hides a failed or missing benchmark subset.
