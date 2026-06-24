"""Tests for run-centric OOD metric JSON export."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest

import torch


MODULE_PATH = Path(__file__).parents[1] / "reports" / "scripts" / "export_metrics_table.py"
SPEC = importlib.util.spec_from_file_location("export_metrics_table", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ExportRunMetricsTests(unittest.TestCase):
    """Validate run-centric OOD metric exports."""

    def test_exports_numeric_metrics_for_selected_run(self) -> None:
        """Export numeric metrics without requiring an experiment config."""

        with self.subTest(), self._temporary_directory() as tmp_path:
            run_name = "random_forest_n50"
            run_dir = tmp_path / "runs" / run_name
            probability_roots = [
                run_dir / "probabilities" / "unperturbed",
                run_dir / "probabilities" / "perturbed",
            ]
            datasets = ["cifar10_test", "mnist_test"]
            for probability_dir in probability_roots:
                for dataset_index, dataset in enumerate(datasets):
                    dataset_dir = probability_dir / dataset
                    dataset_dir.mkdir(parents=True)
                    teacher_path = dataset_dir / "teacher.pt"
                    student_path = dataset_dir / "student_logits_best.pt"
                    teacher_logits = torch.tensor([[4.0, 0.0], [3.0, 1.0]])
                    if dataset_index:
                        teacher_logits = torch.tensor([[1.0, 1.0], [0.5, 0.5]])
                    teacher_probabilities = torch.softmax(teacher_logits, dim=1)
                    student_logits = teacher_logits + torch.tensor(
                        [[0.2, -0.2], [-0.1, 0.1]]
                    )
                    torch.save(
                        {"logits": teacher_logits, "probabilities": teacher_probabilities},
                        teacher_path,
                    )
                    torch.save(
                        {
                            "logits": student_logits,
                            "probabilities": torch.softmax(student_logits, dim=1),
                        },
                        student_path,
                    )
            resolved_config = {
                "experiment_name": run_name,
                "run_dir": str(run_dir),
                "dataset": {
                    "name": "cifar10",
                    "ood_datasets": [{"name": "mnist", "split": "test"}],
                },
                "teacher": {"hf_model_id": "edadaltocg/resnet18_cifar10"},
            }
            (run_dir / "resolved_config.json").write_text(json.dumps(resolved_config))
            for probability_dir in probability_roots:
                (probability_dir / "manifest.json").write_text(
                    json.dumps({"artifacts": []})
                )
            metrics_dir = run_dir / "logits"
            metrics_dir.mkdir()
            (metrics_dir / "metrics.json").write_text(
                json.dumps({"n_estimators": 50})
            )

            original_root = MODULE.ROOT
            MODULE.ROOT = tmp_path
            try:
                output_path = tmp_path / "metrics.json"
                cache = MODULE.MetricCache(tmp_path / "cache.json")
                MODULE.export_run_metrics([run_name], output_path, cache)
            finally:
                MODULE.ROOT = original_root

            payload = json.loads(output_path.read_text())
            exported_run = payload["runs"][0]
            self.assertEqual(exported_run["run_name"], run_name)
            self.assertEqual(exported_run["n_estimators"], 50)
            self.assertEqual(exported_run["probability_modes"], ["unperturbed", "perturbed"])
            self.assertEqual(len(exported_run["metrics"]), 2 * len(MODULE.SCORE_ORDER))
            self.assertEqual(exported_run["metrics"][0]["probability_mode"], "unperturbed")
            self.assertIsInstance(exported_run["metrics"][0]["roc_auc"], float)
            self.assertIsInstance(
                exported_run["metrics"][0]["fpr_at_95_tpr"], float
            )

    @staticmethod
    def _temporary_directory():
        """Return a temporary directory as a Path context manager."""

        import tempfile

        class TemporaryPath:
            def __enter__(self) -> Path:
                self.directory = tempfile.TemporaryDirectory()
                return Path(self.directory.__enter__())

            def __exit__(self, *args: object) -> None:
                self.directory.__exit__(*args)

        return TemporaryPath()
