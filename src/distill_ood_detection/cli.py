"""Command-line interface for distillation experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from distill_ood_detection.config import DistillationMethod, load_config
from distill_ood_detection.evaluation.report import build_ood_report, build_ood_reports
from distill_ood_detection.experiments.infer_probabilities import run_probability_inference
from distill_ood_detection.experiments.train_tree_student import run_tree_experiment
from distill_ood_detection.experiments.train_student import run_experiment


def build_parser() -> argparse.ArgumentParser:
    """Build the project command-line parser."""

    parser = argparse.ArgumentParser(prog="distill-ood")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train-student")
    train_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/distill_linear_cifar10.yaml"),
        help="Path to a YAML experiment config.",
    )
    train_parser.add_argument(
        "--method",
        choices=("cross_entropy", "mse_logits"),
        default=None,
        help="Run only one distillation method. Defaults to all methods in config.",
    )
    tree_parser = subparsers.add_parser("train-tree-student")
    tree_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/distill_random_forest_cifar10.yaml"),
        help="Path to a YAML tree experiment config.",
    )
    tree_parser.add_argument(
        "--mode",
        choices=("logits",),
        default=None,
        help="Run only one tree distillation mode. Defaults to all modes in config.",
    )
    infer_parser = subparsers.add_parser("infer-probabilities")
    infer_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/distill_linear_cifar10.yaml"),
        help="Path to a YAML experiment config.",
    )
    infer_parser.add_argument(
        "--checkpoint",
        choices=("best", "latest", "both"),
        default="best",
        help="Student checkpoint to use. Defaults to best.",
    )
    infer_parser.add_argument(
        "--method",
        choices=("cross_entropy", "mse_logits"),
        default=None,
        help="Run only one distillation method. Defaults to all methods in config.",
    )
    infer_parser.add_argument(
        "--tree-mode",
        choices=("logits",),
        default=None,
        help="Run only one tree distillation mode. Defaults to all modes in config.",
    )
    infer_parser.add_argument(
        "--include-train",
        action="store_true",
        help="Also infer the deterministic ID training split.",
    )
    infer_parser.add_argument(
        "--include-validation",
        action="store_true",
        help="Also infer the deterministic ID validation split.",
    )
    report_parser = subparsers.add_parser("report-ood")
    report_inputs = report_parser.add_mutually_exclusive_group(required=True)
    report_inputs.add_argument(
        "--manifest",
        type=Path,
        help="Path to a probability inference manifest.json.",
    )
    report_inputs.add_argument(
        "--run-dirs",
        type=Path,
        nargs="+",
        help="Experiment run directories containing probabilities/manifest.json.",
    )
    report_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for ood_metrics.json, plots, and ood_report.html. "
            "Defaults to manifest parent for --manifest; required for multiple --run-dirs."
        ),
    )
    report_parser.add_argument(
        "--no-html",
        action="store_true",
        help="Only write ood_metrics.json.",
    )
    report_parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip histogram PNG generation.",
    )
    return parser


def main() -> None:
    """Run the command-line interface."""

    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train-student":
        config = load_config(args.config)
        run_experiment(config, method=args.method)
    if args.command == "train-tree-student":
        config = load_config(args.config)
        run_tree_experiment(config, mode=args.mode)
    if args.command == "infer-probabilities":
        config = load_config(args.config)
        run_probability_inference(
            config,
            checkpoint=args.checkpoint,
            method=args.method,
            tree_mode=args.tree_mode,
            include_train=args.include_train,
            include_validation=args.include_validation,
        )
    if args.command == "report-ood":
        if args.manifest is not None:
            build_ood_report(
                manifest_path=args.manifest,
                output_dir=args.output_dir,
                write_html=not args.no_html,
                write_plots=not args.no_plots,
            )
        else:
            if args.output_dir is None:
                parser.error("--output-dir is required when using --run-dirs")
            build_ood_reports(
                run_dirs=args.run_dirs,
                output_dir=args.output_dir,
                write_html=not args.no_html,
                write_plots=not args.no_plots,
            )


if __name__ == "__main__":
    main()
