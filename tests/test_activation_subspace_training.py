"""Tests for classifier activation-subspace student training."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import TensorDataset

from distill_ood_detection.config import (
    OptimizerConfig,
    ResolvedTrainingMethodConfig,
    load_config,
)
from distill_ood_detection.distillation.activation_subspace import (
    FittedActivationSubspaces,
    PooledEmbeddingSplit,
    activation_subspace_loss,
    component_training_tensors,
    train_activation_subspace_student,
)
from distill_ood_detection.models.student import build_student


class ActivationSubspaceTrainingTests(unittest.TestCase):
    """Validate configs, targets, architectures, and checkpoint traces."""

    def test_configs_define_expected_resnet18_component_dimensions(self) -> None:
        cases = {
            Path(
                "configs/students/activation_subspace/decisive/cifar_10/"
                "resnet18/linear.yaml"
            ): ("decisive", "projected_logits", "linear", (7,)),
            Path(
                "configs/students/activation_subspace/decisive/cifar_10/"
                "resnet18/linear_cross_entropy.yaml"
            ): ("decisive", "projected_logits", "linear", (7,)),
            Path(
                "configs/students/activation_subspace/decisive/cifar_10/"
                "resnet18/linear_cross_entropy_alpha_0_8.yaml"
            ): ("decisive", "projected_logits", "linear", (7,)),
            Path(
                "configs/students/activation_subspace/decisive/cifar_10/"
                "resnet18/autoencoder.yaml"
            ): ("decisive", "coordinates", "autoencoder", (7,)),
            Path(
                "configs/students/activation_subspace/insignificant/cifar_10/"
                "resnet18/autoencoder.yaml"
            ): ("insignificant", "coordinates", "autoencoder", (505,)),
            Path(
                "configs/students/activation_subspace/decisive/cifar_100/"
                "resnet18/linear.yaml"
            ): ("decisive", "projected_logits", "linear", (38,)),
            Path(
                "configs/students/activation_subspace/decisive/cifar_100/"
                "resnet18/linear_cross_entropy.yaml"
            ): ("decisive", "projected_logits", "linear", (38,)),
            Path(
                "configs/students/activation_subspace/decisive/cifar_100/"
                "resnet18/linear_cross_entropy_alpha_0_8.yaml"
            ): ("decisive", "projected_logits", "linear", (38,)),
            Path(
                "configs/students/activation_subspace/decisive/cifar_100/"
                "resnet18/autoencoder.yaml"
            ): ("decisive", "coordinates", "autoencoder", (38,)),
            Path(
                "configs/students/activation_subspace/insignificant/cifar_100/"
                "resnet18/autoencoder.yaml"
            ): ("insignificant", "coordinates", "autoencoder", (474,)),
        }
        for path, expected in cases.items():
            with self.subTest(path=str(path)):
                config = load_config(path)
                self.assertEqual(config.strategy.name, "activation_subspace")
                self.assertEqual(
                    (
                        config.strategy.activation_subspace.component,
                        config.strategy.activation_subspace.target,
                        config.student.kind,
                        config.student.input_shape,
                    ),
                    expected,
                )

    def test_cross_entropy_configs_use_temperature_one(self) -> None:
        variants = {
            "linear_cross_entropy.yaml": 0.5,
            "linear_cross_entropy_alpha_0_8.yaml": 0.8,
        }
        for dataset in ("cifar_10", "cifar_100"):
            for filename, expected_alpha in variants.items():
                with self.subTest(dataset=dataset, filename=filename):
                    config = load_config(
                        Path(
                            "configs/students/activation_subspace/decisive/"
                            f"{dataset}/resnet18/{filename}"
                        )
                    )
                    self.assertEqual(
                        config.training.enabled_methods(),
                        ("cross_entropy",),
                    )
                    training = config.training.for_method("cross_entropy")
                    self.assertEqual(training.temperature, 1.0)
                    self.assertEqual(training.alpha, expected_alpha)

    def test_autoencoder_is_bottlenecked_and_preserves_coordinate_shape(self) -> None:
        config = load_config(
            Path(
                "configs/students/activation_subspace/insignificant/cifar_10/"
                "resnet18/autoencoder.yaml"
            )
        )
        student = build_student(config.student)
        inputs = torch.randn(3, 505)

        outputs = student(inputs)

        self.assertEqual(tuple(outputs.shape), tuple(inputs.shape))
        linear_layers = [
            module for module in student.modules() if isinstance(module, nn.Linear)
        ]
        self.assertEqual(
            [(layer.in_features, layer.out_features) for layer in linear_layers],
            [(505, 256), (256, 64), (64, 256), (256, 505)],
        )

    def test_decisive_autoencoders_use_confirmed_bottlenecks(self) -> None:
        cases = {
            "cifar_10": [(7, 4), (4, 7)],
            "cifar_100": [
                (38, 24),
                (24, 12),
                (12, 24),
                (24, 38),
            ],
        }
        for dataset, expected_layers in cases.items():
            with self.subTest(dataset=dataset):
                config = load_config(
                    Path(
                        "configs/students/activation_subspace/decisive/"
                        f"{dataset}/resnet18/autoencoder.yaml"
                    )
                )
                student = build_student(config.student)
                linear_layers = [
                    module
                    for module in student.modules()
                    if isinstance(module, nn.Linear)
                ]
                self.assertEqual(
                    [
                        (layer.in_features, layer.out_features)
                        for layer in linear_layers
                    ],
                    expected_layers,
                )

    def test_component_tensors_use_coordinates_and_decisive_projected_logits(self) -> None:
        basis = torch.eye(3)
        subspaces = FittedActivationSubspaces(
            right_basis=basis,
            singular_values=torch.tensor([2.0, 1.0]),
            decisive_dimension=1,
            norm_gaps=torch.zeros(3),
        )
        split = PooledEmbeddingSplit(
            embeddings=torch.tensor([[2.0, 3.0, 4.0]]),
            labels=torch.tensor([1]),
        )
        classifier = nn.Linear(3, 2)
        with torch.no_grad():
            classifier.weight.copy_(
                torch.tensor([[1.0, 10.0, 100.0], [-1.0, -10.0, -100.0]])
            )
            classifier.bias.copy_(torch.tensor([0.5, -0.5]))

        decisive = component_training_tensors(
            split=split,
            subspaces=subspaces,
            component="decisive",
            target="projected_logits",
            classifier=classifier,
            device=torch.device("cpu"),
        )
        insignificant = component_training_tensors(
            split=split,
            subspaces=subspaces,
            component="insignificant",
            target="coordinates",
            classifier=classifier,
            device=torch.device("cpu"),
        )

        torch.testing.assert_close(decisive.tensors[0], torch.tensor([[2.0]]))
        torch.testing.assert_close(
            decisive.tensors[1], torch.tensor([[2.5, -2.5]])
        )
        torch.testing.assert_close(
            insignificant.tensors[0], torch.tensor([[3.0, 4.0]])
        )
        torch.testing.assert_close(insignificant.tensors[1], insignificant.tensors[0])

        decisive_coordinates = component_training_tensors(
            split=split,
            subspaces=subspaces,
            component="decisive",
            target="coordinates",
            classifier=classifier,
            device=torch.device("cpu"),
        )
        torch.testing.assert_close(
            decisive_coordinates.tensors[1],
            decisive_coordinates.tensors[0],
        )

    def test_decisive_loss_removes_per_sample_logit_offset(self) -> None:
        targets = torch.tensor([[1.0, 2.0, 3.0]])
        predictions = targets + 17.0

        loss = activation_subspace_loss("decisive", predictions, targets)

        torch.testing.assert_close(loss, torch.tensor(0.0))

    def test_coordinate_loss_does_not_remove_feature_offset(self) -> None:
        targets = torch.tensor([[1.0, 2.0, 3.0]])
        predictions = targets + 2.0

        loss = activation_subspace_loss(
            "decisive",
            predictions,
            targets,
            target="coordinates",
        )

        torch.testing.assert_close(loss, torch.tensor(4.0))

    def test_decisive_cross_entropy_reuses_project_distillation_loss(self) -> None:
        targets = torch.tensor([[2.0, -1.0], [-0.5, 1.5]])
        predictions = torch.tensor([[1.0, 0.0], [0.5, -0.25]])
        labels = torch.tensor([0, 1])

        actual = activation_subspace_loss(
            "decisive",
            predictions,
            targets,
            labels=labels,
            distillation_method="cross_entropy",
            temperature=1.0,
            alpha=0.5,
        )
        expected = torch.nn.functional.cross_entropy(predictions, labels) * 0.5
        expected += 0.5 * (
            -(
                torch.softmax(targets, dim=1)
                * torch.log_softmax(predictions, dim=1)
            ).sum(dim=1).mean()
        )

        torch.testing.assert_close(actual, expected)

    def test_training_writes_best_latest_history_and_metrics(self) -> None:
        generator = torch.Generator().manual_seed(3)
        inputs = torch.randn(12, 2, generator=generator)
        targets = inputs.clone()
        labels = torch.zeros(12, dtype=torch.long)
        dataset = TensorDataset(inputs, targets, labels)
        student = nn.Sequential(nn.Linear(2, 1), nn.GELU(), nn.Linear(1, 2))
        training = ResolvedTrainingMethodConfig(
            epochs=2,
            seed=7,
            device="cpu",
            log_every_steps=100,
        )

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            summary = train_activation_subspace_student(
                student=student,
                component="insignificant",
                target="coordinates",
                distillation_method="mse_logits",
                train_data=dataset,
                validation_data=dataset,
                test_data=dataset,
                batch_size=4,
                device=torch.device("cpu"),
                optimizer_config=OptimizerConfig(
                    name="adamw",
                    learning_rate=0.01,
                ),
                training_config=training,
                output_dir=output_dir,
                mlflow_enabled=False,
            )

            self.assertTrue((output_dir / "best_student.pt").exists())
            self.assertTrue((output_dir / "latest_student.pt").exists())
            self.assertTrue((output_dir / "history.json").exists())
            self.assertTrue((output_dir / "metrics.json").exists())
            self.assertEqual(summary["component"], "insignificant")
            self.assertIn("test_loss", summary)


if __name__ == "__main__":
    unittest.main()
