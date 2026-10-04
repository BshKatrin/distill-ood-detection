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

Use locally configured paths for the ImageNet image root, the shared DTD image
root, and the OpenOOD asset root. On a cluster, keep machine-specific values in
`docs/hpc/hpc.local.md`, following the existing local-configuration policy.
The checked-in teacher configs record the original cluster paths; create a local
config copy and set `dataset.data_dir`, each split/OOD `imglist_path`, OOD
`data_dir`, and `teacher.checkpoint_path` to your actual locations.
`DISTILL_OOD_DATA_DIR` does not relocate manifest or checkpoint paths.

```text
<openood-root>/
  archives/
  data/
    benchmark_imglist/imagenet200/
    images_largescale/{ssb_hard,ninco,inaturalist,openimage_o}/
  results/
    imagenet200_resnet18_224x224_base_e90_lr0.1_default/s0/best.ckpt
```

The ImageNet image root supplies ID images selected by the manifests. The DTD
image root supplies the 5,160 images selected by `test_texture.txt`; no extra
copy of the shared DTD dataset is downloaded.

When cluster access is needed, read the [HPC guide](../../hpc/README.md) and
configure local paths before submitting the restartable download job:

```bash
openood_asset_root=/path/to/openood
mkdir -p "$openood_asset_root/logs"
sbatch slurm_scripts/download_openood_imagenet200.sbatch "$openood_asset_root"
```

The explicit argument overrides the script's original cluster default. The job
downloads 6.71 GB of archives and validates byte counts and ZIP archives before
extraction. If compute nodes cannot reach Google Drive, run the script once with
`OPENOOD_DOWNLOAD_ONLY=1` on the transfer/login side, then submit it normally to
validate and extract. Downloads and extraction stay under the asset root.

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

Higher scores must mean more ID-like. Evaluation uses the OOD-positive
[comparison convention](../metrics.md). The evaluator returns FPR95, AUROC,
AUPR-IN, and AUPR-OUT as fractions, plus arithmetic macro averages for the Near
and Far groups. Dataset-level results remain available, so a macro average never
hides a failed or missing benchmark subset.
