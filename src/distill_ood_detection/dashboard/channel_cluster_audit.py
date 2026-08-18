"""Minimal Panel dashboard for NMF channel-cluster OOD scores."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import panel as pn
from bokeh.models import ColumnDataSource, FixedTicker, HoverTool, Range1d
from bokeh.plotting import figure

LAYERS = ("layer2", "layer3", "layer4")
SCORE_LABELS = {
    "Raw reconstruction score": "raw",
    "Absolute improvement": "absolute_improvement",
    "Relative improvement": "relative_improvement",
}
VARIANT_LABELS = {
    "Whole cluster": "whole",
    "25% within cluster": "fraction_p025",
}
OOD_DATASETS = ("MNIST", "SVHN", "opposite_cifar")
OOD_COLORS = {
    "MNIST": "#4C78A8",
    "SVHN": "#F58518",
    "opposite_cifar": "#54A24B",
}


class AuditRepository:
    """Read the compact task and OOD-metric summaries."""

    def __init__(self, summary_dir: Path) -> None:
        self.tasks = pd.read_parquet(summary_dir / "task_metadata.parquet")
        self.metrics = pd.read_parquet(summary_dir / "cluster_metrics.parquet")


def build_dashboard(summary_dir: Path) -> pn.template.FastListTemplate:
    """Construct the minimal channel-cluster score dashboard."""

    pn.extension(sizing_mode="stretch_width")
    repository = AuditRepository(summary_dir)
    complete = repository.tasks[repository.tasks.status == "complete"]
    if complete.empty:
        raise ValueError(f"No complete audit tasks in {summary_dir}")

    datasets = sorted(complete.dataset.unique().tolist())
    variants = [value for value in VARIANT_LABELS.values() if value in complete.variant.unique()]
    dataset = pn.widgets.Select(name="ID dataset", options=datasets)
    variant = pn.widgets.Select(
        name="Masking variant",
        options={label: value for label, value in VARIANT_LABELS.items() if value in variants},
    )
    score = pn.widgets.Select(name="OOD score input", options=SCORE_LABELS)

    @pn.depends(dataset.param.value, variant.param.value, score.param.value)
    def scores_per_cluster(
        dataset_value: str,
        variant_value: str,
        score_value: str,
    ) -> pn.Column:
        plots = [
            _cluster_score_plot(
                repository,
                dataset_value,
                variant_value,
                score_value,
                layer,
            )
            for layer in LAYERS
        ]
        return pn.Column(
            pn.pane.Markdown(
                "Wide/light bars show **ROC-AUC**; narrow/hatched bars show "
                "**FPR@95**. Higher ROC-AUC and lower FPR@95 are better."
            ),
            *plots,
            sizing_mode="stretch_width",
        )

    tabs = pn.Tabs(
        ("Scores per cluster", scores_per_cluster),
        dynamic=True,
        sizing_mode="stretch_width",
    )
    return pn.template.FastListTemplate(
        title="NMF Channel-Cluster Audit",
        sidebar=[dataset, variant, score],
        main=[tabs],
        sidebar_width=270,
        accent_base_color="#355C7D",
    )


def _cluster_score_plot(
    repository: AuditRepository,
    dataset: str,
    variant: str,
    score: str,
    layer: str,
) -> pn.viewable.Viewable:
    rows = repository.metrics[
        (repository.metrics.dataset == dataset)
        & (repository.metrics.layer == layer)
        & (repository.metrics.variant == variant)
        & (repository.metrics.score == score)
        & (repository.metrics.ood_dataset.isin(OOD_DATASETS))
    ].copy()
    if rows.empty:
        return pn.pane.Alert(f"No complete scores for {layer}.", alert_type="warning")

    cluster_order = (
        rows[["cluster_id", "cluster_order"]]
        .drop_duplicates()
        .sort_values("cluster_order")
    )
    clusters = cluster_order.cluster_id.tolist()
    cluster_positions = {cluster: index for index, cluster in enumerate(clusters)}
    dataset_offsets = {"MNIST": -0.25, "SVHN": 0.0, "opposite_cifar": 0.25}
    opposite_cifar = "CIFAR-100" if dataset == "cifar10" else "CIFAR-10"
    display_names = {
        "MNIST": "MNIST",
        "SVHN": "SVHN",
        "opposite_cifar": opposite_cifar,
    }

    plot = figure(
        title=layer.replace("layer", "Layer "),
        x_axis_label="Cluster (hierarchy order)",
        y_axis_label="OOD score",
        x_range=Range1d(-0.55, max(len(clusters) - 0.45, 0.55)),
        y_range=Range1d(0.0, 1.0),
        height=330,
        sizing_mode="stretch_width",
        tools="xpan,xwheel_zoom,reset",
        active_scroll="xwheel_zoom",
        toolbar_location="above",
    )
    hover_renderers = []
    for ood_dataset in OOD_DATASETS:
        dataset_rows = rows[rows.ood_dataset == ood_dataset].copy()
        dataset_rows["x"] = dataset_rows.cluster_id.map(cluster_positions) + dataset_offsets[ood_dataset]
        dataset_rows["ood_label"] = display_names[ood_dataset]
        source = ColumnDataSource(dataset_rows)
        color = OOD_COLORS[ood_dataset]
        roc_renderer = plot.vbar(
            x="x",
            top="roc_auc",
            width=0.22,
            source=source,
            color=color,
            fill_alpha=0.35,
            line_alpha=0.7,
            legend_label=display_names[ood_dataset],
        )
        fpr_renderer = plot.vbar(
            x="x",
            top="fpr_at_95_tpr",
            width=0.10,
            source=source,
            color=color,
            fill_alpha=0.72,
            line_alpha=0.9,
            hatch_pattern="/",
            hatch_color=color,
            hatch_alpha=0.9,
        )
        hover_renderers.extend((roc_renderer, fpr_renderer))

    plot.add_tools(
        HoverTool(
            renderers=hover_renderers,
            tooltips=[
                ("Cluster", "@cluster_id"),
                ("OOD dataset", "@ood_label"),
                ("ROC-AUC", "@roc_auc{0.0000}"),
                ("FPR@95", "@fpr_at_95_tpr{0.0000}"),
            ],
        )
    )
    plot.xaxis.ticker = FixedTicker(ticks=list(range(len(clusters))))
    plot.xaxis.major_label_overrides = {
        index: cluster for index, cluster in enumerate(clusters)
    }
    plot.xaxis.major_label_orientation = 1.0
    plot.xaxis.major_label_text_font_size = "7pt"
    plot.yaxis.ticker.desired_num_ticks = 6
    plot.legend.orientation = "horizontal"
    plot.legend.location = "top_center"
    plot.legend.click_policy = "hide"
    plot.grid.grid_line_alpha = 0.2
    return pn.pane.Bokeh(plot, sizing_mode="stretch_width")


def serve_from_environment() -> pn.template.FastListTemplate:
    """Build the app using ``CHANNEL_CLUSTER_AUDIT_SUMMARY_DIR``."""

    summary_dir = Path(
        os.environ.get(
            "CHANNEL_CLUSTER_AUDIT_SUMMARY_DIR",
            "runs/channel_cluster_audit/nmf_latent_cosine/summary",
        )
    )
    return build_dashboard(summary_dir)


if pn.state.served:
    serve_from_environment().servable()
