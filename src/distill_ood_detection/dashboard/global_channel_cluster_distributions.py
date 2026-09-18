"""Panel dashboard for global-student channel-cluster improvements."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import panel as pn
from bokeh.models import ColumnDataSource, FixedTicker, HoverTool, Range1d, Span
from bokeh.plotting import figure

LAYERS = ("layer2", "layer3", "layer4")
MODEL_LABELS = {"ResNet-18": "resnet18", "ResNet-50": "resnet50"}
ID_DATASET_LABELS = {"CIFAR-10": "cifar10", "CIFAR-100": "cifar100"}
SCORE_LABELS = {
    "Raw reconstruction error": "raw_reconstruction_error",
    "Absolute improvement": "absolute_improvement",
    "Relative improvement": "relative_improvement",
}
DISTRIBUTIONS = ("id", "mnist", "svhn", "opposite_cifar")
DISTRIBUTION_COLORS = {
    "id": "#4C78A8",
    "mnist": "#F58518",
    "svhn": "#E45756",
    "opposite_cifar": "#54A24B",
}


class GlobalClusterDistributionRepository:
    """Read compact boxplot summaries for global NMF students."""

    def __init__(self, summary_dir: Path) -> None:
        self.boxplots = pd.read_parquet(summary_dir / "boxplot_summaries.parquet")


def build_dashboard(summary_dir: Path) -> pn.template.FastListTemplate:
    """Construct the global-student cluster-distribution dashboard."""

    pn.extension(sizing_mode="stretch_width")
    repository = GlobalClusterDistributionRepository(summary_dir)
    if repository.boxplots.empty:
        raise ValueError(f"No boxplot summaries in {summary_dir}")
    available_models = repository.boxplots.teacher_model.unique().tolist()
    available_datasets = repository.boxplots.id_dataset.unique().tolist()
    model = pn.widgets.Select(
        name="Teacher model",
        options={
            label: value
            for label, value in MODEL_LABELS.items()
            if value in available_models
        },
    )
    id_dataset = pn.widgets.Select(
        name="ID dataset",
        options={
            label: value
            for label, value in ID_DATASET_LABELS.items()
            if value in available_datasets
        },
    )
    score = pn.widgets.Select(name="Score", options=SCORE_LABELS)

    @pn.depends(model.param.value, id_dataset.param.value, score.param.value)
    def cluster_distributions(
        model_value: str,
        dataset_value: str,
        score_value: str,
    ) -> pn.Column:
        return pn.Column(
            pn.pane.Markdown(
                "Each box summarizes one value per image, averaged over ten "
                "independent masking draws. Raw reconstruction error is the "
                "masked-channel MSE; positive improvement means the student "
                "reconstructs masked channels better than identity."
            ),
            *(
                _cluster_boxplot(
                    repository,
                    model_value,
                    dataset_value,
                    score_value,
                    layer,
                )
                for layer in LAYERS
            ),
            sizing_mode="stretch_width",
        )

    return pn.template.FastListTemplate(
        title="Global NMF Student: Cluster Scores",
        sidebar=[model, id_dataset, score],
        main=[cluster_distributions],
        sidebar_width=270,
        accent_base_color="#355C7D",
    )


def _cluster_boxplot(
    repository: GlobalClusterDistributionRepository,
    teacher_model: str,
    id_dataset: str,
    score: str,
    layer: str,
) -> pn.viewable.Viewable:
    rows = repository.boxplots[
        (repository.boxplots.teacher_model == teacher_model)
        & (repository.boxplots.id_dataset == id_dataset)
        & (repository.boxplots.score == score)
        & (repository.boxplots.layer == layer)
    ].copy()
    if rows.empty:
        return pn.pane.Alert(f"No distribution summaries for {layer}.", alert_type="warning")
    cluster_rows = (
        rows[["cluster_id", "cluster_order"]]
        .drop_duplicates()
        .sort_values("cluster_order")
    )
    clusters = cluster_rows.cluster_id.tolist()
    positions = {cluster_id: index for index, cluster_id in enumerate(clusters)}
    offsets = {"id": -0.27, "mnist": -0.09, "svhn": 0.09, "opposite_cifar": 0.27}
    score_label = next(label for label, value in SCORE_LABELS.items() if value == score)
    y_start = min(float(rows.lower_whisker.min()), 0.0)
    y_end = max(float(rows.upper_whisker.max()), 0.0)
    y_padding = max((y_end - y_start) * 0.08, 1.0e-6)
    plot = figure(
        title=layer.replace("layer", "Layer "),
        x_axis_label="Cluster (hierarchy order)",
        y_axis_label=score_label,
        x_range=Range1d(-0.6, max(len(clusters) - 0.4, 0.6)),
        y_range=Range1d(y_start - y_padding, y_end + y_padding),
        height=360,
        sizing_mode="stretch_width",
        tools="xpan,xwheel_zoom,reset",
        active_scroll="xwheel_zoom",
        toolbar_location="above",
    )
    hover_renderers = []
    for distribution in DISTRIBUTIONS:
        distribution_rows = rows[rows.distribution == distribution].copy()
        if distribution_rows.empty:
            continue
        distribution_rows["x"] = (
            distribution_rows.cluster_id.map(positions) + offsets[distribution]
        )
        distribution_rows["legend_label"] = distribution_rows.apply(
            lambda row: (
                f"ID ({row.dataset_label})"
                if distribution == "id"
                else str(row.dataset_label)
            ),
            axis=1,
        )
        source = ColumnDataSource(distribution_rows)
        color = DISTRIBUTION_COLORS[distribution]
        plot.segment(
            x0="x",
            y0="lower_whisker",
            x1="x",
            y1="q25",
            source=source,
            color=color,
        )
        plot.segment(
            x0="x",
            y0="q75",
            x1="x",
            y1="upper_whisker",
            source=source,
            color=color,
        )
        box = plot.vbar(
            x="x",
            width=0.14,
            top="q75",
            bottom="q25",
            source=source,
            fill_color=color,
            fill_alpha=0.55,
            line_color=color,
            legend_label=str(distribution_rows.legend_label.iloc[0]),
        )
        plot.scatter(
            x="x",
            y="median",
            source=source,
            marker="dash",
            size=11,
            line_width=2,
            color=color,
        )
        plot.scatter(
            x="x",
            y="lower_whisker",
            source=source,
            marker="dash",
            size=7,
            line_width=1,
            color=color,
        )
        plot.scatter(
            x="x",
            y="upper_whisker",
            source=source,
            marker="dash",
            size=7,
            line_width=1,
            color=color,
        )
        hover_renderers.append(box)
    plot.add_layout(Span(location=0.0, dimension="width", line_dash="dashed", line_alpha=0.6))
    plot.add_tools(
        HoverTool(
            renderers=hover_renderers,
            tooltips=[
                ("Cluster", "@cluster_id"),
                ("Dataset", "@legend_label"),
                ("Count", "@count{0,0}"),
                ("Median", "@median{0.000000}"),
                ("Q1 / Q3", "@q25{0.000000} / @q75{0.000000}"),
                ("Whiskers", "@lower_whisker{0.000000} / @upper_whisker{0.000000}"),
                ("Mean", "@mean{0.000000}"),
                ("Std", "@standard_deviation{0.000000}"),
                ("Cluster size", "@cluster_size"),
                ("Masked channels", "@masked_channel_count"),
            ],
        )
    )
    plot.xaxis.ticker = FixedTicker(ticks=list(range(len(clusters))))
    plot.xaxis.major_label_overrides = {
        index: cluster_id for index, cluster_id in enumerate(clusters)
    }
    plot.xaxis.major_label_orientation = 1.0
    plot.xaxis.major_label_text_font_size = "7pt"
    plot.legend.orientation = "horizontal"
    plot.legend.location = "top_center"
    plot.legend.click_policy = "hide"
    plot.grid.grid_line_alpha = 0.2
    return pn.pane.Bokeh(plot, sizing_mode="stretch_width")


def serve_from_environment() -> pn.template.FastListTemplate:
    """Build the app from ``GLOBAL_CLUSTER_DISTRIBUTION_SUMMARY_DIR``."""

    summary_dir = Path(
        os.environ.get(
            "GLOBAL_CLUSTER_DISTRIBUTION_SUMMARY_DIR",
            "runs/global_channel_cluster_distributions/nmf_latent_cosine/summary",
        )
    )
    return build_dashboard(summary_dir)


if pn.state.served:
    serve_from_environment().servable()
