"""Channel profiles and correlation-based hierarchical grouping."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform


def top_activation_profiles(
    features: torch.Tensor,
    top_fraction: float,
) -> torch.Tensor:
    """Return one top-spatial-activation mean per sample and channel."""

    if features.ndim != 4:
        raise ValueError("features must have shape (samples, channels, height, width)")
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must satisfy 0 < fraction <= 1")
    spatial_values = features.flatten(start_dim=2)
    top_count = max(1, math.ceil(top_fraction * spatial_values.shape[-1]))
    return spatial_values.topk(top_count, dim=-1, sorted=False).values.mean(dim=-1)


def standardize_channel_profiles(
    profiles: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standardize each channel across ID training samples.

    Channels whose population standard deviation is at most ``epsilon`` receive
    an all-zero standardized profile and are marked as constant.
    """

    if profiles.ndim != 2:
        raise ValueError("profiles must have shape (samples, channels)")
    if profiles.shape[0] < 2:
        raise ValueError("at least two samples are required")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    means = profiles.mean(dim=0)
    standard_deviations = profiles.std(dim=0, correction=0)
    active_channels = standard_deviations > epsilon
    safe_standard_deviations = torch.where(
        active_channels,
        standard_deviations,
        torch.ones_like(standard_deviations),
    )
    standardized = (profiles - means) / safe_standard_deviations
    standardized[:, ~active_channels] = 0.0
    return standardized, means, standard_deviations, active_channels


def pearson_channel_correlation(
    standardized_profiles: torch.Tensor,
    active_channels: torch.Tensor,
) -> torch.Tensor:
    """Return the channelwise Pearson correlation matrix."""

    if standardized_profiles.ndim != 2:
        raise ValueError("standardized_profiles must have shape (samples, channels)")
    if active_channels.shape != (standardized_profiles.shape[1],):
        raise ValueError("active_channels must match the channel dimension")
    sample_count = standardized_profiles.shape[0]
    correlations = standardized_profiles.T @ standardized_profiles / sample_count
    correlations = (correlations + correlations.T) * 0.5
    correlations = correlations.clamp(min=-1.0, max=1.0)
    inactive = ~active_channels
    correlations[inactive, :] = 0.0
    correlations[:, inactive] = 0.0
    correlations.fill_diagonal_(1.0)
    return correlations


def correlation_distance(correlations: torch.Tensor) -> torch.Tensor:
    """Convert a Pearson correlation matrix to ``1 - correlation`` distance."""

    if correlations.ndim != 2 or correlations.shape[0] != correlations.shape[1]:
        raise ValueError("correlations must be a square matrix")
    distances = (1.0 - correlations).clamp(min=0.0, max=2.0)
    distances = (distances + distances.T) * 0.5
    distances.fill_diagonal_(0.0)
    return distances


def latent_channel_cosine_distance(
    p_matrix: torch.Tensor,
    epsilon: float = 1.0e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cosine distances between channel columns of an NMF ``P`` matrix.

    A channel whose latent profile has norm at most ``epsilon`` is marked as
    zero and assigned distance one from every other channel. Its self-distance
    remains zero.
    """

    if p_matrix.ndim != 2:
        raise ValueError("p_matrix must have shape (concepts, channels)")
    if p_matrix.shape[0] < 1 or p_matrix.shape[1] < 2:
        raise ValueError("p_matrix must contain concepts and at least two channels")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if float(p_matrix.min()) < -1.0e-7:
        raise ValueError("NMF P matrix must be non-negative")

    channel_profiles = p_matrix.T
    norms = channel_profiles.norm(dim=1)
    active_channels = norms > epsilon
    normalized = channel_profiles / norms.clamp_min(epsilon).unsqueeze(1)
    similarities = normalized @ normalized.T
    similarities = (similarities + similarities.T) * 0.5
    similarities = similarities.clamp(min=-1.0, max=1.0)
    inactive = ~active_channels
    similarities[inactive, :] = 0.0
    similarities[:, inactive] = 0.0
    similarities.fill_diagonal_(1.0)
    distances = (1.0 - similarities).clamp(min=0.0, max=2.0)
    distances = (distances + distances.T) * 0.5
    distances.fill_diagonal_(0.0)
    return distances, active_channels


def hierarchical_channel_linkage(
    distances: torch.Tensor,
    method: str,
) -> np.ndarray:
    """Build a SciPy hierarchical linkage matrix from channel distances."""

    if method not in {"single", "complete", "average"}:
        raise ValueError(f"Unsupported linkage method: {method}")
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distances must be a square matrix")
    if distances.shape[0] < 2:
        raise ValueError("at least two channels are required")
    condensed = squareform(distances.detach().cpu().numpy(), checks=True)
    return linkage(condensed, method=method, optimal_ordering=True)


def flat_channel_groups(
    linkage_matrix: np.ndarray,
    threshold: float,
) -> list[list[int]]:
    """Cut a hierarchy at one correlation-distance threshold."""

    if not 0.0 < threshold < 2.0:
        raise ValueError("threshold must be strictly between 0 and 2")
    labels = fcluster(linkage_matrix, t=threshold, criterion="distance")
    groups_by_label: dict[int, list[int]] = {}
    for channel, label in enumerate(labels.tolist()):
        groups_by_label.setdefault(label, []).append(channel)
    return sorted(groups_by_label.values(), key=lambda channels: channels[0])


def render_channel_dendrograms(
    linkage_matrix: np.ndarray,
    layer: str,
    output_dir: Path,
    thresholds: tuple[float, ...],
    title: str | None = None,
    distance_label: str = "Correlation distance (1 - Pearson r)",
) -> tuple[Path, Path]:
    """Write an overview PNG and a zoomable, channel-labeled PDF dendrogram."""

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    overview_path = output_dir / f"{layer}_dendrogram_overview.png"
    labeled_path = output_dir / f"{layer}_dendrogram_labeled.pdf"
    channel_count = linkage_matrix.shape[0] + 1
    color_threshold = max(thresholds) if thresholds else None

    figure, axis = plt.subplots(figsize=(16, 8), constrained_layout=True)
    dendrogram(
        linkage_matrix,
        no_labels=True,
        color_threshold=color_threshold,
        above_threshold_color="black",
        ax=axis,
    )
    _decorate_dendrogram(axis, layer, thresholds, title, distance_label)
    figure.savefig(overview_path, dpi=180)
    plt.close(figure)

    labeled_width = max(16.0, min(64.0, channel_count * 0.08))
    figure, axis = plt.subplots(
        figsize=(labeled_width, 9),
        constrained_layout=True,
    )
    dendrogram(
        linkage_matrix,
        labels=[str(channel) for channel in range(channel_count)],
        leaf_font_size=4.0,
        color_threshold=color_threshold,
        above_threshold_color="black",
        ax=axis,
    )
    _decorate_dendrogram(axis, layer, thresholds, title, distance_label)
    axis.set_xlabel("Channel index")
    figure.savefig(labeled_path)
    plt.close(figure)
    return overview_path, labeled_path


def _decorate_dendrogram(
    axis: object,
    layer: str,
    thresholds: tuple[float, ...],
    title: str | None,
    distance_label: str,
) -> None:
    axis.set_title(title or f"{layer}: top-activation Pearson channel hierarchy")
    axis.set_ylabel(distance_label)
    colors = ("#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd")
    for index, threshold in enumerate(thresholds):
        axis.axhline(
            threshold,
            color=colors[index % len(colors)],
            linewidth=0.8,
            linestyle="--",
            label=f"cut={threshold:g}",
        )
    if thresholds:
        axis.legend(loc="upper right", fontsize="small")


def hierarchy_leaf_order(linkage_matrix: np.ndarray) -> list[int]:
    """Return channel indices in dendrogram leaf order."""

    return leaves_list(linkage_matrix).astype(int).tolist()
