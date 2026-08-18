# Panel dashboards on SLURM

Both dashboards run as CPU-only SLURM jobs and are accessed through an SSH
tunnel via the cluster login host. Compute nodes do not need to accept direct
SSH connections. Replace `<cluster-login-alias>` with the private alias from
`docs/hpc/hpc.local.md` and `<compute-node>` with the node reported by `squeue`.

Before first use, synchronize the locked environments from the repository root:

```bash
uv sync --project envs/gpu --locked
uv sync --project envs/dashboard --locked
```

## Specialist channel-cluster audit

This is the earlier dashboard that compares ROC-AUC and FPR@95 for one student
per cluster. If its summaries already exist, start only the Panel server:

```bash
panel_job=$(sbatch --parsable \
  --exclude=daft \
  --time=1-00:00:00 \
  --export=ALL,SKIP_UV_SYNC=1,PORT=5006 \
  slurm_scripts/serve_channel_cluster_audit.sbatch)
echo "${panel_job}"
squeue -j "${panel_job}" -o "%.18i %.9T %.20N"
```

To regenerate missing specialist artifacts, summaries, and the server as one
dependency chain instead, use:

```bash
bash slurm_scripts/submit_channel_cluster_audit.sh
```

After the Panel job is running, open the tunnel from the local machine:

```bash
ssh -N -L 5006:<compute-node>:5006 <cluster-login-alias>
```

Open `http://localhost:5006/channel_cluster_audit`.

## Global-student cluster distributions

This dashboard compares the per-image distribution of absolute or relative
improvement for the global NMF students. It supports ResNet-18, ResNet-50,
CIFAR-10 ID, and CIFAR-100 ID. Each value is averaged over ten independent
masks before its boxplot statistics are computed.

On the cluster, collect the twelve global-student artifacts, summarize them,
and start Panel with explicit dependencies:

```bash
configs=(
  configs/students/feature_denoising/channel_group_stratified_masking/nmf_latent_cosine/cifar_{10,100}/resnet{18,50}/cluster_layer*.yaml
)

export_job=$(sbatch --parsable \
  --exclude=daft \
  --export=ALL,SKIP_UV_SYNC=1 \
  slurm_scripts/export_global_channel_cluster_distributions.sbatch \
  "${configs[@]}")

summary_job=$(sbatch --parsable \
  --exclude=daft \
  --dependency="afterok:${export_job}" \
  --export=ALL,SKIP_UV_SYNC=1 \
  slurm_scripts/summarize_global_channel_cluster_distributions.sbatch \
  "${configs[@]}")

panel_job=$(sbatch --parsable \
  --exclude=daft \
  --dependency="afterok:${summary_job}" \
  --time=1-00:00:00 \
  --export=ALL,SKIP_UV_SYNC=1,PORT=5007 \
  slurm_scripts/serve_global_channel_cluster_distributions.sbatch)

printf 'export=%s summary=%s panel=%s\n' \
  "${export_job}" "${summary_job}" "${panel_job}"
squeue -j "${panel_job}" -o "%.18i %.9T %.20N"
```

If the exported tensors and Parquet summary are already complete, submit only
the final `panel_job` command without its dependency.

After the Panel job is running, open the tunnel locally:

```bash
ssh -N -L 5007:<compute-node>:5007 <cluster-login-alias>
```

Open `http://localhost:5007/global_channel_cluster_distributions`.

## Run both dashboards together

The two Panel jobs may run on different compute nodes. One local SSH command can
forward both ports:

```bash
ssh -N \
  -L 5006:<specialist-panel-node>:5006 \
  -L 5007:<global-panel-node>:5007 \
  <cluster-login-alias>
```

Keep that terminal open while using either dashboard. If a page cannot connect,
first confirm that its Panel job is still `RUNNING` and that the node in the
tunnel matches the current `squeue` output.
