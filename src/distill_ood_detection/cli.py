"""Command-line interface for distillation experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from distill_ood_detection.config import DistillationMethod, load_config
from distill_ood_detection.experiments.infer_probabilities import run_probability_inference
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
        choices=("mse_softmax", "cross_entropy_softmax", "mse_logits"),
        default=None,
        help="Run only one distillation method. Defaults to all methods in config.",
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
        choices=("mse_softmax", "cross_entropy_softmax", "mse_logits"),
        default=None,
        help="Run only one distillation method. Defaults to all methods in config.",
    )
    return parser


def main() -> None:
    """Run the command-line interface."""

    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train-student":
        config = load_config(args.config)
        run_experiment(config, method=args.method)
    if args.command == "infer-probabilities":
        config = load_config(args.config)
        run_probability_inference(
            config,
            checkpoint=args.checkpoint,
            method=args.method,
        )


if __name__ == "__main__":
    main()
