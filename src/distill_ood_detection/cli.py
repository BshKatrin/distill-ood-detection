"""Command-line interface for distillation experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from distill_ood_detection.config import (
    load_embedding_distance_config,
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
        default=Path("configs/students/baseline/cifar_10/resnet18/linear.yaml"),
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
        default=Path("configs/students/baseline/cifar_10/resnet18/random_forest.yaml"),
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
        default=Path("configs/students/baseline/cifar_10/resnet18/linear.yaml"),
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
        default=Path("configs/teachers/cifar_10/resnet18.yaml"),
        help="Path to a YAML teacher activation export config.",
    )
    embedding_distances_parser = subparsers.add_parser(
        "export-embedding-distances",
        help="Export pooled teacher embeddings and exact ID-reference distances.",
    )
    embedding_distances_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML embedding-distance config.",
    )
    probabilities_parser = subparsers.add_parser("export-teacher-probabilities")
    probabilities_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/teachers/cifar_10/resnet50.yaml"),
        help="Path to a YAML teacher probability export config.",
    )
    feature_denoising_scores_parser = subparsers.add_parser(
        "export-feature-denoising-scores",
    )
    feature_denoising_scores_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/students/feature_denoising/pca_masking/cifar_10/resnet18/linear_layer4_pca9_mask_p030.yaml"),
        help="Path to a YAML Feature Denoising experiment config.",
    )
    feature_denoising_scores_parser.add_argument(
        "--checkpoint",
        choices=("best", "latest"),
        default="best",
        help="Student checkpoint to use. Defaults to best.",
    )
    feature_denoising_scores_parser.add_argument(
        "--include-train",
        action="store_true",
        help="Also export the deterministic ID training split.",
    )
    feature_denoising_scores_parser.add_argument(
        "--include-validation",
        action="store_true",
        help="Also export the deterministic ID validation split.",
    )
    feature_denoising_subspaces_parser = subparsers.add_parser(
        "export-feature-denoising-subspace-errors",
        help="Export layer4 reconstruction errors in ActSub classifier subspaces.",
    )
    feature_denoising_subspaces_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a layer4 Feature Denoising experiment config.",
    )
    feature_denoising_subspaces_parser.add_argument(
        "--checkpoint",
        choices=("best", "latest"),
        default="best",
        help="Student checkpoint to use. Defaults to best.",
    )
    activation_subspace_inference_parser = subparsers.add_parser(
        "export-activation-subspace-inference",
        help="Export OOD pooled teacher embeddings and raw student outputs.",
    )
    activation_subspace_inference_parser.add_argument(
        "--config",
        type=Path,
        action="append",
        required=True,
        help="Activation-subspace student config. Repeat for every student.",
    )
    activation_subspace_inference_parser.add_argument(
        "--teacher-output-dir",
        type=Path,
        required=True,
        help="Shared output directory for teacher embedding artifacts.",
    )
    activation_subspace_inference_parser.add_argument(
        "--checkpoint",
        choices=("best", "latest"),
        default="best",
        help="Student checkpoint to use. Defaults to best.",
    )
    activation_subspace_scores_parser = subparsers.add_parser(
        "export-activation-subspace-scores",
        help="Export OOD Scores from activation-subspace inference artifacts.",
    )
    activation_subspace_scores_parser.add_argument(
        "--config",
        type=Path,
        action="append",
        required=True,
        help="Activation-subspace student config. Repeat for every student.",
    )
    activation_subspace_scores_parser.add_argument(
        "--teacher-embedding-dir",
        type=Path,
        required=True,
        help="Directory containing shared teacher embedding artifacts.",
    )
    activation_subspace_scores_parser.add_argument(
        "--checkpoint",
        choices=("best", "latest"),
        default="best",
        help="Student inference checkpoint to score. Defaults to best.",
    )
    subspace_ensemble_inference_parser = subparsers.add_parser(
        "export-subspace-ensemble-inference",
        help="Export per-member logits and probabilities for a subspace ensemble.",
    )
    subspace_ensemble_inference_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a subspace-ensemble experiment config.",
    )
    subspace_ensemble_inference_parser.add_argument(
        "--checkpoint",
        choices=("best", "latest"),
        default="best",
        help="Ensemble checkpoint to use. Defaults to best.",
    )
    subspace_ensemble_scores_parser = subparsers.add_parser(
        "export-subspace-ensemble-scores",
        help="Export predictive-entropy and BALD OOD Scores.",
    )
    subspace_ensemble_scores_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a subspace-ensemble experiment config.",
    )
    subspace_ensemble_scores_parser.add_argument(
        "--checkpoint",
        choices=("best", "latest"),
        default="best",
        help="Ensemble inference checkpoint to score. Defaults to best.",
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
    if args.command == "export-embedding-distances":
        from distill_ood_detection.experiments.export_embedding_distances import (
            run_embedding_distance_export,
        )

        config = load_embedding_distance_config(args.config)
        run_embedding_distance_export(config)
    if args.command == "export-feature-denoising-scores":
        from distill_ood_detection.experiments.export_feature_denoising_scores import (
            run_feature_denoising_score_export,
        )

        config = load_config(args.config)
        run_feature_denoising_score_export(
            config,
            checkpoint=args.checkpoint,
            include_train=args.include_train,
            include_validation=args.include_validation,
        )
    if args.command == "export-feature-denoising-subspace-errors":
        from distill_ood_detection.experiments.export_feature_denoising_subspace_errors import (
            run_feature_denoising_subspace_error_export,
        )

        config = load_config(args.config)
        run_feature_denoising_subspace_error_export(
            config,
            checkpoint=args.checkpoint,
        )
    if args.command == "export-activation-subspace-inference":
        from distill_ood_detection.experiments.export_activation_subspace_inference import (
            run_activation_subspace_inference_export,
        )

        configs = [load_config(path) for path in args.config]
        run_activation_subspace_inference_export(
            configs,
            teacher_output_dir=args.teacher_output_dir,
            checkpoint=args.checkpoint,
        )
    if args.command == "export-activation-subspace-scores":
        from distill_ood_detection.experiments.export_activation_subspace_scores import (
            run_activation_subspace_score_export,
        )

        configs = [load_config(path) for path in args.config]
        run_activation_subspace_score_export(
            configs,
            teacher_embedding_dir=args.teacher_embedding_dir,
            checkpoint=args.checkpoint,
        )
    if args.command == "export-subspace-ensemble-inference":
        from distill_ood_detection.experiments.export_subspace_ensemble_inference import (
            run_subspace_ensemble_inference_export,
        )

        config = load_config(args.config)
        run_subspace_ensemble_inference_export(
            config,
            checkpoint=args.checkpoint,
        )
    if args.command == "export-subspace-ensemble-scores":
        from distill_ood_detection.experiments.export_subspace_ensemble_scores import (
            run_subspace_ensemble_score_export,
        )

        config = load_config(args.config)
        run_subspace_ensemble_score_export(
            config,
            checkpoint=args.checkpoint,
        )


if __name__ == "__main__":
    main()
