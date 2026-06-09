"""Build repeatable OOD metric reports from probability manifests."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from numpy.typing import NDArray

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from distill_ood_detection.evaluation.ood_metrics import ood_detection_metrics
from distill_ood_detection.evaluation.ood_scores import (
    absolute_max_probability_difference,
    logit_l2_distance,
    max_probability_difference,
    student_teacher_kl_divergence,
)
from distill_ood_detection.utils import write_json


def build_ood_report(
    manifest_path: Path,
    output_dir: Path | None = None,
    write_html: bool = True,
    write_plots: bool = True,
) -> dict[str, object]:
    """Compute OOD metrics from an inference manifest.

    Args:
        manifest_path: Path to a probability inference ``manifest.json``.
        output_dir: Directory for report artifacts. Defaults to the manifest's
            parent directory.
        write_html: Whether to write ``ood_report.html`` in addition to JSON.
        write_plots: Whether to write histogram plots for each score row.

    Returns:
        A dictionary containing report metadata and metric rows.
    """

    manifest = _load_json(manifest_path)
    report_dir = output_dir or manifest_path.parent
    report_dir.mkdir(parents=True, exist_ok=True)
    score_rows = _score_rows(manifest)
    rows = [_metric_row(row) for row in score_rows]
    if write_plots:
        _write_histogram_plots(score_rows, rows, report_dir / "plots")
    report = {
        "experiment_name": manifest["experiment_name"],
        "id_dataset": manifest["dataset"]["name"],
        "checkpoint_selection": manifest["checkpoint_selection"],
        "plots_written": write_plots,
        "rows": rows,
    }
    write_json(report_dir / "ood_metrics.json", report)
    if write_html:
        (report_dir / "ood_report.html").write_text(
            _render_html(report),
            encoding="utf-8",
        )
    return report


def build_ood_reports(
    run_dirs: list[Path],
    output_dir: Path,
    write_html: bool = True,
    write_plots: bool = True,
) -> dict[str, object]:
    """Build an aggregate OOD report from one or more experiment run folders.

    Args:
        run_dirs: Experiment directories such as ``runs/linear_student_resnet18_cifar100``.
        output_dir: Directory for aggregate JSON, plots, and HTML.
        write_html: Whether to write ``ood_report.html``.
        write_plots: Whether to write histogram plots.

    Returns:
        A dictionary containing all per-experiment reports and metric rows.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    experiments = []
    for run_dir in run_dirs:
        manifest_path = _manifest_from_run_dir(run_dir)
        experiment_output_dir = output_dir / _safe_name(run_dir.name)
        experiment_report = build_ood_report(
            manifest_path=manifest_path,
            output_dir=experiment_output_dir,
            write_html=False,
            write_plots=write_plots,
        )
        experiments.append(experiment_report)

    rows = [
        {**row, "experiment_name": experiment["experiment_name"]}
        for experiment in experiments
        for row in experiment["rows"]
    ]
    report = {
        "experiments": experiments,
        "plots_written": write_plots,
        "rows": rows,
    }
    write_json(output_dir / "ood_metrics.json", report)
    if write_html:
        (output_dir / "ood_report.html").write_text(
            _render_aggregate_html(report),
            encoding="utf-8",
        )
    return report


def _manifest_from_run_dir(run_dir: Path) -> Path:
    manifest_path = run_dir / "probabilities" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Probability manifest not found: {manifest_path}")
    return manifest_path


def _score_rows(manifest: dict[str, Any]) -> list[dict[str, object]]:
    id_dataset = f"{manifest['dataset']['name']}_test"
    ood_datasets = [
        f"{item['name']}_{item.get('split', 'test')}"
        for item in manifest["dataset"].get("ood_datasets", [])
    ]
    artifacts = _load_artifacts(manifest)
    rows: list[dict[str, object]] = []

    id_teacher = artifacts[("teacher", id_dataset, "teacher")]
    for ood_dataset in ood_datasets:
        ood_teacher = artifacts[("teacher", ood_dataset, "teacher")]
        rows.append(
            _score_row(
                model="teacher",
                score="msp",
                dataset=ood_dataset,
                id_scores=_msp(id_teacher["probabilities"]),
                ood_scores=_msp(ood_teacher["probabilities"]),
            )
        )

    for key, id_student in artifacts.items():
        model, dataset, model_name = key
        if model != "student" or dataset != id_dataset:
            continue
        for ood_dataset in ood_datasets:
            ood_student = artifacts.get(("student", ood_dataset, model_name))
            if ood_student is None:
                continue
            ood_teacher = artifacts[("teacher", ood_dataset, "teacher")]
            rows.extend(
                _student_rows(
                    model_name=model_name,
                    ood_dataset=ood_dataset,
                    id_teacher=id_teacher,
                    id_student=id_student,
                    ood_teacher=ood_teacher,
                    ood_student=ood_student,
                )
            )
    return rows


def _student_rows(
    model_name: str,
    ood_dataset: str,
    id_teacher: dict[str, object],
    id_student: dict[str, object],
    ood_teacher: dict[str, object],
    ood_student: dict[str, object],
) -> list[dict[str, object]]:
    id_teacher_probabilities = _array(id_teacher["probabilities"])
    id_student_probabilities = _array(id_student["probabilities"])
    ood_teacher_probabilities = _array(ood_teacher["probabilities"])
    ood_student_probabilities = _array(ood_student["probabilities"])
    id_teacher_logits = _array(id_teacher["logits"])
    id_student_logits = _array(id_student["logits"])
    ood_teacher_logits = _array(ood_teacher["logits"])
    ood_student_logits = _array(ood_student["logits"])

    scores = {
        "student_msp": (
            _msp(id_student_probabilities),
            _msp(ood_student_probabilities),
        ),
        "max_probability_difference": (
            max_probability_difference(
                id_teacher_probabilities,
                id_student_probabilities,
                signed=True,
            ),
            max_probability_difference(
                ood_teacher_probabilities,
                ood_student_probabilities,
                signed=True,
            ),
        ),
        "absolute_max_probability_difference": (
            absolute_max_probability_difference(
                id_teacher_probabilities,
                id_student_probabilities,
                signed=True,
            ),
            absolute_max_probability_difference(
                ood_teacher_probabilities,
                ood_student_probabilities,
                signed=True,
            ),
        ),
        "student_teacher_kl_divergence": (
            student_teacher_kl_divergence(
                id_teacher_probabilities,
                id_student_probabilities,
                signed=True,
            ),
            student_teacher_kl_divergence(
                ood_teacher_probabilities,
                ood_student_probabilities,
                signed=True,
            ),
        ),
        "logit_l2_distance": (
            logit_l2_distance(id_teacher_logits, id_student_logits, signed=True),
            logit_l2_distance(ood_teacher_logits, ood_student_logits, signed=True),
        ),
    }
    return [
        _score_row(
            model=model_name,
            score=score,
            dataset=ood_dataset,
            id_scores=id_scores,
            ood_scores=ood_scores,
        )
        for score, (id_scores, ood_scores) in scores.items()
    ]


def _score_row(
    model: str,
    score: str,
    dataset: str,
    id_scores: NDArray[np.float64],
    ood_scores: NDArray[np.float64],
) -> dict[str, object]:
    return {
        "model": model,
        "score": score,
        "ood_dataset": dataset,
        "id_scores": id_scores,
        "ood_scores": ood_scores,
    }


def _metric_row(score_row: dict[str, object]) -> dict[str, object]:
    id_scores = score_row["id_scores"]
    ood_scores = score_row["ood_scores"]
    if not isinstance(id_scores, np.ndarray) or not isinstance(ood_scores, np.ndarray):
        raise TypeError("score row must contain numpy score arrays")
    row = _row_metrics(
        model=str(score_row["model"]),
        score=str(score_row["score"]),
        dataset=str(score_row["ood_dataset"]),
        id_scores=id_scores,
        ood_scores=ood_scores,
    )
    row["plot_path"] = _plot_filename(row)
    return row


def _row_metrics(
    model: str,
    score: str,
    dataset: str,
    id_scores: NDArray[np.float64],
    ood_scores: NDArray[np.float64],
) -> dict[str, object]:
    labels = np.concatenate(
        [
            np.ones(id_scores.shape[0], dtype=int),
            np.zeros(ood_scores.shape[0], dtype=int),
        ]
    )
    scores = np.concatenate([id_scores, ood_scores])
    metrics = ood_detection_metrics(labels, scores)
    return {
        "model": model,
        "score": score,
        "ood_dataset": dataset,
        "n_id": int(id_scores.shape[0]),
        "n_ood": int(ood_scores.shape[0]),
        **metrics,
    }


def _load_artifacts(manifest: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, object]]:
    artifacts = {}
    for entry in manifest["artifacts"]:
        artifact = torch.load(_artifact_path(entry["path"]), map_location="cpu", weights_only=True)
        model = entry["model"]
        dataset = entry["dataset"]
        if model == "teacher":
            model_name = "teacher"
        else:
            mode_or_method = entry.get("method") or entry.get("mode")
            checkpoint = entry["checkpoint"]
            model_name = f"{entry['student_kind']}:{mode_or_method}:{checkpoint}"
        artifacts[(model, dataset, model_name)] = artifact
    return artifacts


def _artifact_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _msp(probabilities: object) -> NDArray[np.float64]:
    return _array(probabilities).max(axis=1)


def _array(value: object) -> NDArray[np.float64]:
    if isinstance(value, torch.Tensor):
        return value.cpu().numpy().astype(np.float64)
    return np.asarray(value, dtype=np.float64)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _render_html(report: dict[str, object]) -> str:
    rows = sorted(
        report["rows"],
        key=lambda item: (item["ood_dataset"], item["model"], item["score"]),
    )
    table_rows = "\n".join(_render_row(row) for row in rows)
    plot_cards = "\n".join(_render_plot_card(row) for row in rows)
    plot_section = (
        f"""  <section class="plots">
{plot_cards}
  </section>"""
        if report.get("plots_written", True)
        else ""
    )
    title = html.escape(str(report["experiment_name"]))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title} OOD Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #172026; }}
    h1 {{ font-size: 24px; margin: 0 0 8px; }}
    .meta {{ color: #5c6870; margin-bottom: 24px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #dde3e8; padding: 8px 10px; text-align: left; }}
    th {{ background: #f5f7f9; font-weight: 650; }}
    td.metric {{ font-variant-numeric: tabular-nums; text-align: right; }}
    .plots {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 18px; margin-top: 28px; }}
    figure {{ margin: 0; border: 1px solid #dde3e8; border-radius: 6px; padding: 10px; }}
    img {{ max-width: 100%; height: auto; display: block; }}
    figcaption {{ color: #5c6870; font-size: 13px; margin-top: 8px; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="meta">ID: {html.escape(str(report["id_dataset"]))} | checkpoint: {html.escape(str(report["checkpoint_selection"]))}</div>
  <table>
    <thead>
      <tr>
        <th>OOD dataset</th>
        <th>Model</th>
        <th>Score</th>
        <th>ROC-AUC</th>
        <th>FPR@95</th>
        <th>N ID</th>
        <th>N OOD</th>
      </tr>
    </thead>
    <tbody>
{table_rows}
    </tbody>
  </table>
{plot_section}
</body>
</html>
"""


def _render_aggregate_html(report: dict[str, object]) -> str:
    rows = sorted(
        report["rows"],
        key=lambda item: (
            item["experiment_name"],
            item["ood_dataset"],
            item["model"],
            item["score"],
        ),
    )
    table_rows = "\n".join(_render_aggregate_row(row) for row in rows)
    experiment_sections = (
        "\n".join(
            _render_experiment_section(experiment)
            for experiment in report["experiments"]
        )
        if report.get("plots_written", True)
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>OOD Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #172026; }}
    h1 {{ font-size: 26px; margin: 0 0 8px; }}
    h2 {{ font-size: 20px; margin: 32px 0 10px; }}
    .meta {{ color: #5c6870; margin-bottom: 24px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #dde3e8; padding: 8px 10px; text-align: left; }}
    th {{ background: #f5f7f9; font-weight: 650; position: sticky; top: 0; }}
    td.metric {{ font-variant-numeric: tabular-nums; text-align: right; }}
    .plots {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 18px; }}
    figure {{ margin: 0; border: 1px solid #dde3e8; border-radius: 6px; padding: 10px; }}
    img {{ max-width: 100%; height: auto; display: block; }}
    figcaption {{ color: #5c6870; font-size: 13px; margin-top: 8px; }}
  </style>
</head>
<body>
  <h1>OOD Report</h1>
  <div class="meta">{len(report["experiments"])} experiment(s), {len(rows)} metric rows</div>
  <table>
    <thead>
      <tr>
        <th>Experiment</th>
        <th>OOD dataset</th>
        <th>Model</th>
        <th>Score</th>
        <th>ROC-AUC</th>
        <th>FPR@95</th>
        <th>N ID</th>
        <th>N OOD</th>
      </tr>
    </thead>
    <tbody>
{table_rows}
    </tbody>
  </table>
{experiment_sections}
</body>
</html>
"""


def _render_row(row: dict[str, object]) -> str:
    return (
        "      <tr>"
        f"<td>{html.escape(str(row['ood_dataset']))}</td>"
        f"<td>{html.escape(str(row['model']))}</td>"
        f"<td>{html.escape(str(row['score']))}</td>"
        f"<td class=\"metric\">{float(row['roc_auc']):.4f}</td>"
        f"<td class=\"metric\">{float(row['fpr_at_95_tpr']):.4f}</td>"
        f"<td class=\"metric\">{int(row['n_id'])}</td>"
        f"<td class=\"metric\">{int(row['n_ood'])}</td>"
        "</tr>"
    )


def _render_aggregate_row(row: dict[str, object]) -> str:
    return (
        "      <tr>"
        f"<td>{html.escape(str(row['experiment_name']))}</td>"
        f"<td>{html.escape(str(row['ood_dataset']))}</td>"
        f"<td>{html.escape(str(row['model']))}</td>"
        f"<td>{html.escape(str(row['score']))}</td>"
        f"<td class=\"metric\">{float(row['roc_auc']):.4f}</td>"
        f"<td class=\"metric\">{float(row['fpr_at_95_tpr']):.4f}</td>"
        f"<td class=\"metric\">{int(row['n_id'])}</td>"
        f"<td class=\"metric\">{int(row['n_ood'])}</td>"
        "</tr>"
    )


def _render_experiment_section(experiment: dict[str, object]) -> str:
    rows = sorted(
        experiment["rows"],
        key=lambda item: (item["ood_dataset"], item["model"], item["score"]),
    )
    plot_cards = "\n".join(
        _render_plot_card(row, prefix=f"{_safe_name(str(experiment['experiment_name']))}/")
        for row in rows
    )
    return f"""  <section>
    <h2>{html.escape(str(experiment["experiment_name"]))}</h2>
    <div class="plots">
{plot_cards}
    </div>
  </section>
"""


def _render_plot_card(row: dict[str, object], prefix: str = "") -> str:
    plot_path = prefix + str(row["plot_path"])
    caption = (
        f"{row['ood_dataset']} | {row['model']} | {row['score']} | "
        f"ROC-AUC {float(row['roc_auc']):.4f} | FPR@95 {float(row['fpr_at_95_tpr']):.4f}"
    )
    return f"""    <figure>
      <img src="{html.escape(plot_path)}" alt="{html.escape(caption)}">
      <figcaption>{html.escape(caption)}</figcaption>
    </figure>"""


def _write_histogram_plots(
    score_rows: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
    plot_dir: Path,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    for score_row, metric_row in zip(score_rows, metric_rows, strict=True):
        id_scores = score_row["id_scores"]
        ood_scores = score_row["ood_scores"]
        if not isinstance(id_scores, np.ndarray) or not isinstance(ood_scores, np.ndarray):
            raise TypeError("score row must contain numpy score arrays")
        _write_histogram_plot(
            path=plot_dir / str(metric_row["plot_path"]).removeprefix("plots/"),
            id_scores=id_scores,
            ood_scores=ood_scores,
            metric_row=metric_row,
        )


def _write_histogram_plot(
    path: Path,
    id_scores: NDArray[np.float64],
    ood_scores: NDArray[np.float64],
    metric_row: dict[str, object],
) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 4.2), dpi=140)
    axis.hist(id_scores, bins=40, alpha=0.62, density=True, label="ID")
    axis.hist(ood_scores, bins=40, alpha=0.62, density=True, label="OOD")
    axis.set_title(
        f"{metric_row['ood_dataset']} | {metric_row['model']} | {metric_row['score']}",
        fontsize=10,
    )
    axis.set_xlabel("ID-likeness score")
    axis.set_ylabel("Density")
    axis.legend(frameon=False)
    axis.text(
        0.99,
        0.98,
        f"ROC-AUC {float(metric_row['roc_auc']):.4f}\nFPR@95 {float(metric_row['fpr_at_95_tpr']):.4f}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#dde3e8"},
    )
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _plot_filename(row: dict[str, object]) -> str:
    return (
        "plots/"
        f"{_safe_name(str(row['ood_dataset']))}__"
        f"{_safe_name(str(row['model']))}__"
        f"{_safe_name(str(row['score']))}.png"
    )


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
