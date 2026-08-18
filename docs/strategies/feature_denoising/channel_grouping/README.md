# Channel Grouping Methods

Channel grouping methods identify sets of teacher feature channels that may
represent a shared concept. The resulting groups are inputs to later Feature
Denoising masking experiments; they are not OOD Scores by themselves.

## Methods

- [Top-activation profile correlation](top_activation_correlation.md): group
  channels with similar per-image high-activation profiles using Pearson
  correlation and hierarchical clustering.
- [NMF latent-profile cosine](nmf_latent_cosine.md): fit a global non-negative
  basis and group channels by cosine distance between columns of its `P` matrix.
- [NMF channel-cluster audit](channel_cluster_audit.md): train fixed-cluster
  specialists and inspect cross-cluster OOD behavior in a Panel app.
- [Global-student cluster distributions](global_student_cluster_distributions.md):
  decompose draw-averaged absolute and relative improvements by NMF cluster for
  the existing global students.

Each method has its own Markdown file in this folder. Add future alternatives
here rather than extending one method document with unrelated grouping rules.
