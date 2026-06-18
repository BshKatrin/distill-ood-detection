"""Command-line interface for distillation experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from distill_ood_detection.config import (
    load_config,
    load_teacher_activation_config,
    load_teacher_probability_config,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the project command-line parser."""

    parser = argparse.ArgumentParser(prog="distill-ood")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train-student")
    train_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline/cifar_10/linear.yaml"),
        help="Path to a YAML experiment config.",
    )
    train_parser.add_argument(
        "--method",
        choices=("cross_entropy", "mse_logits", "kl_divergence"),
        default=None,
        help="Run only one distillation method. Defaults to all methods in config.",
    )
    tree_parser = subparsers.add_parser("train-tree-student")
    tree_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline/cifar_10/random_forest.yaml"),
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
        default=Path("configs/baseline/cifar_10/linear.yaml"),
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
        choices=("cross_entropy", "mse_logits", "kl_divergence"),
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
    infer_parser.add_argument(
        "--apply-perturbation",
        action="store_true",
        help=(
            "Apply stochastic perturbations during perturbation-strategy inference. "
            "Defaults to deterministic unperturbed inference."
        ),
    )
    activations_parser = subparsers.add_parser("export-teacher-activations")
    activations_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/teachers/resnet18_cifar10_layers.yaml"),
        help="Path to a YAML teacher activation export config.",
    )
    probabilities_parser = subparsers.add_parser("export-teacher-probabilities")
    probabilities_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/teachers/resnet50_cifar10_probabilities.yaml"),
        help="Path to a YAML teacher probability export config.",
    )
    return parser


def main() -> None:
    """Run the command-line interface."""

    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train-student":
        from distill_ood_detection.experiments.train_student import run_experiment

        config = load_config(args.config)
        run_experiment(config, method=args.method)
    if args.command == "train-tree-student":
        from distill_ood_detection.experiments.train_tree_student import run_tree_experiment

        config = load_config(args.config)
        run_tree_experiment(config, mode=args.mode)
    if args.command == "infer-probabilities":
        from distill_ood_detection.experiments.infer_probabilities import (
            run_probability_inference,
        )

        config = load_config(args.config)
        run_probability_inference(
            config,
            checkpoint=args.checkpoint,
            method=args.method,
            tree_mode=args.tree_mode,
            include_train=args.include_train,
            include_validation=args.include_validation,
            apply_perturbation=args.apply_perturbation,
        )
    if args.command == "export-teacher-activations":
        from distill_ood_detection.experiments.export_teacher_activations import (
            run_teacher_activation_export,
        )

        config = load_teacher_activation_config(args.config)
        run_teacher_activation_export(config)
    if args.command == "export-teacher-probabilities":
        from distill_ood_detection.experiments.export_teacher_probabilities import (
            run_teacher_probability_export,
        )

        config = load_teacher_probability_config(args.config)
        run_teacher_probability_export(config)


if __name__ == "__main__":
    main()
