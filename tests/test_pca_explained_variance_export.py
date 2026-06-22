"""Tests for the PCA explained-variance statistics export."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch


SCRIPT_PATH = Path("reports/scripts/export_pca_explained_variance.py")
SPEC = importlib.util.spec_from_file_location("export_pca_explained_variance", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PcaExplainedVarianceExportTests(unittest.TestCase):
    """Validate activation loading and PCA statistics."""

    def test_statistics_are_reproducible_and_cumulative(self) -> None:
        rng = np.random.default_rng(4)
        values = rng.normal(size=(40, 8)).astype(np.float32)

        first = MODULE.explained_variance_statistics(
            values, max_components=4, seed=7, power_iterations=2
        )
        second = MODULE.explained_variance_statistics(
            values, max_components=4, seed=7, power_iterations=2
        )

        self.assertEqual(first, second)
        cumulative = first["cumulative_explained_variance_ratio"]
        self.assertEqual(first["component"], [1, 2, 3, 4])
        self.assertTrue(all(left <= right for left, right in zip(cumulative, cumulative[1:])))
        self.assertLessEqual(cumulative[-1], 1.0)

    def test_loader_rejects_non_training_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "layer4.pt"
            torch.save(
                {"split": "test", "activations": torch.zeros(3, 2, 2)},
                path,
            )

            with self.assertRaisesRegex(ValueError, "training-split"):
                MODULE.load_flattened_activations(path)

    def test_default_statistics_path_is_beside_activation_artifact(self) -> None:
        path = Path("runs/example/teacher_activations/train/layer3.pt")

        self.assertEqual(
            MODULE.default_statistics_path(path),
            path.with_name("layer3_pca_explained_variance.json"),
        )

    def test_renders_notebook_with_statistics_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "analysis.ipynb"
            statistics_path = Path(directory) / "statistics.json"

            MODULE.render_notebook(
                statistics_path=statistics_path,
                activation_path=Path("runs/example/layer4.pt"),
                output_path=output_path,
                title="Example PCA",
            )

            notebook = json.loads(output_path.read_text())
            sources = "\n".join(
                "".join(cell["source"]) for cell in notebook["cells"]
            )
            self.assertIn("# Example PCA", sources)
            self.assertIn(str(statistics_path.resolve()), sources)
            self.assertNotIn("{{ statistics_path }}", sources)
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    self.assertEqual(cell["outputs"], [])
                    self.assertIsNone(cell["execution_count"])


if __name__ == "__main__":
    unittest.main()
