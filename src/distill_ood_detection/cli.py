"""Command-line interface for distillation experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from distill_ood_detection.config import (
    load_channel_grouping_config,
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
    openood_cifar_parser = subparsers.add_parser(
        "evaluate-openood-cifar",
        help="Evaluate one existing CIFAR student on fixed OpenOOD v1.5 manifests.",
    )
    openood_cifar_parser.add_argument("--config", type=Path, required=True)
    openood_cifar_parser.add_argument(
        "--openood-root",
        type=Path,
        default=Path("/home/bogush/openood"),
    )
    openood_cifar_parser.add_argument(
        "--checkpoint", choices=("best", "latest"), default="best"
    )
    openood_cifar_parser.add_argument(
        "--method",
        choices=("cross_entropy", "mse_logits", "kl_divergence"),
        default=None,
    )
    openood_cifar_parser.add_argument("--apply-perturbation", action="store_true")
    openood_cifar_parser.add_argument("--force", action="store_true")
    openood_manifest_parser = subparsers.add_parser(
        "build-openood-cifar-manifest",
        help="Resolve selected OpenOOD CIFAR variants against existing checkpoints.",
    )
    openood_manifest_parser.add_argument("--selection", type=Path, required=True)
    openood_manifest_parser.add_argument("--output", type=Path, required=True)
    openood_manifest_parser.add_argument("--config-root", type=Path, default=None)
    openood_task_parser = subparsers.add_parser(
        "run-openood-cifar-task",
        help="Run one ready task from an OpenOOD CIFAR evaluation manifest.",
    )
    openood_task_parser.add_argument("--manifest", type=Path, required=True)
    openood_task_parser.add_argument("--task-index", type=int, required=True)
    openood_task_parser.add_argument(
        "--openood-root", type=Path, default=Path("/home/bogush/openood")
    )
    openood_task_parser.add_argument("--force", action="store_true")
    activations_parser = subparsers.add_parser("export-teacher-activations")
    activations_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/teachers/cifar_10/resnet18.yaml"),
        help="Path to a YAML teacher activation export config.",
    )
    channel_groups_parser = subparsers.add_parser(
        "build-channel-groups",
        help="Build correlation-based hierarchical teacher-channel groups.",
    )
    channel_groups_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a channel-grouping YAML config.",
    )
    embedding_distances_parser = subparsers.add_parser(
        "export-embedding-distances",
        help="Export layer4 embeddings and exact FAISS ID-reference scores.",
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
    audit_manifest_parser = subparsers.add_parser(
        "build-channel-cluster-audit-manifest",
        help="Expand dataset-level audit configs into stable specialist tasks.",
    )
    audit_manifest_parser.add_argument(
        "--config",
        type=Path,
        action="append",
        required=True,
        help="Dataset-level audit config; repeat for both CIFAR datasets.",
    )
    audit_manifest_parser.add_argument("--output", type=Path, required=True)
    audit_task_parser = subparsers.add_parser(
        "run-channel-cluster-audit-task",
        help="Run one restartable specialist task from an audit manifest.",
    )
    audit_task_parser.add_argument("--manifest", type=Path, required=True)
    audit_task_parser.add_argument("--task-index", type=int, required=True)
    audit_task_parser.add_argument(
        "--force",
        action="store_true",
        help="Run even if the task already has complete artifacts.",
    )
    audit_metadata_parser = subparsers.add_parser(
        "export-channel-cluster-audit-metadata",
        help="Export shared class and teacher metadata for one audit dataset.",
    )
    audit_metadata_parser.add_argument("--config", type=Path, required=True)
    audit_summary_parser = subparsers.add_parser(
        "summarize-channel-cluster-audit",
        help="Build compact Parquet summaries for the audit dashboard.",
    )
    audit_summary_parser.add_argument("--manifest", type=Path, required=True)
    audit_summary_parser.add_argument("--output-dir", type=Path, required=True)
    audit_summary_parser.add_argument("--require-complete", action="store_true")
    audit_status_parser = subparsers.add_parser(
        "channel-cluster-audit-status",
        help="Print missing task indices as a SLURM array expression.",
    )
    audit_status_parser.add_argument("--manifest", type=Path, required=True)
    audit_status_parser.add_argument("--deep", action="store_true")
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
    global_cluster_distributions_parser = subparsers.add_parser(
        "export-global-channel-cluster-distributions",
        help="Export per-image cluster improvements for one global student.",
    )
    global_cluster_distributions_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a global stratified channel-group student config.",
    )
    global_cluster_distributions_parser.add_argument(
        "--checkpoint",
        choices=("best", "latest"),
        default="best",
    )
    global_cluster_summary_parser = subparsers.add_parser(
        "summarize-global-channel-cluster-distributions",
        help="Build boxplot summaries for the global-student dashboard.",
    )
    global_cluster_summary_parser.add_argument(
        "--config",
        type=Path,
        action="append",
        required=True,
        help="Global student config; repeat for every model, dataset, and layer.",
    )
    global_cluster_summary_parser.add_argument("--output-dir", type=Path, required=True)
    global_cluster_summary_parser.add_argument(
        "--checkpoint",
        choices=("best", "latest"),
        default="best",
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
    if args.command == "evaluate-openood-cifar":
        from distill_ood_detection.experiments.evaluate_openood_cifar import (
            run_openood_cifar_evaluation,
        )

        config = load_config(args.config)
        run_openood_cifar_evaluation(
            config,
            source_config_path=args.config,
            openood_root=args.openood_root,
            checkpoint=args.checkpoint,
            method=args.method,
            apply_perturbation=args.apply_perturbation,
            force=args.force,
        )
    if args.command == "build-openood-cifar-manifest":
        from distill_ood_detection.experiments.openood_cifar_manifest import (
            build_openood_cifar_task_manifest,
        )

        build_openood_cifar_task_manifest(
            args.selection,
            args.output,
            config_root=args.config_root,
        )
    if args.command == "run-openood-cifar-task":
        from distill_ood_detection.experiments.openood_cifar_manifest import (
            run_openood_cifar_manifest_task,
        )

        run_openood_cifar_manifest_task(
            args.manifest,
            args.task_index,
            args.openood_root,
            force=args.force,
        )
    if args.command == "export-teacher-activations":
        from distill_ood_detection.experiments.export_teacher_activations import (
            run_teacher_activation_export,
        )

        config = load_teacher_activation_config(args.config)
        run_teacher_activation_export(config)
    if args.command == "build-channel-groups":
        from distill_ood_detection.experiments.build_channel_groups import (
            run_channel_grouping,
        )

        config = load_channel_grouping_config(args.config)
        run_channel_grouping(config)
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
    if args.command == "build-channel-cluster-audit-manifest":
        from distill_ood_detection.experiments.channel_cluster_audit import (
            write_channel_cluster_audit_manifest,
        )

        write_channel_cluster_audit_manifest(args.config, args.output)
    if args.command == "run-channel-cluster-audit-task":
        from distill_ood_detection.experiments.channel_cluster_audit import (
            run_channel_cluster_audit_task,
        )

        run_channel_cluster_audit_task(
            args.manifest,
            args.task_index,
            skip_complete=not args.force,
        )
    if args.command == "export-channel-cluster-audit-metadata":
        from distill_ood_detection.experiments.channel_cluster_audit import (
            export_channel_cluster_audit_sample_metadata,
        )

        export_channel_cluster_audit_sample_metadata(args.config)
    if args.command == "summarize-channel-cluster-audit":
        from distill_ood_detection.experiments.summarize_channel_cluster_audit import (
            summarize_channel_cluster_audit,
        )

        summarize_channel_cluster_audit(
            args.manifest,
            args.output_dir,
            require_complete=args.require_complete,
        )
    if args.command == "channel-cluster-audit-status":
        from distill_ood_detection.experiments.channel_cluster_audit import (
            incomplete_channel_cluster_task_indices,
            slurm_array_spec,
        )

        print(
            slurm_array_spec(
                incomplete_channel_cluster_task_indices(
                    args.manifest,
                    deep=args.deep,
                )
            )
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
    if args.command == "export-global-channel-cluster-distributions":
        from distill_ood_detection.experiments.global_channel_cluster_distributions import (
            run_global_channel_cluster_distribution_export,
        )

        config = load_config(args.config)
        run_global_channel_cluster_distribution_export(
            config,
            checkpoint=args.checkpoint,
        )
    if args.command == "summarize-global-channel-cluster-distributions":
        from distill_ood_detection.experiments.global_channel_cluster_distributions import (
            summarize_global_channel_cluster_distributions,
        )

        summarize_global_channel_cluster_distributions(
            args.config,
            args.output_dir,
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
