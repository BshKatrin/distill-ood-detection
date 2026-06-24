"""Tests for config-to-run-directory mapping."""

from pathlib import Path
import unittest

import yaml


class RunDirectoryConfigTests(unittest.TestCase):
    """Ensure every config has one predictable artifact directory."""

    def test_run_directories_mirror_config_paths(self) -> None:
        """Map configs/<identity>.yaml to runs/<identity>."""

        run_dirs: set[str] = set()
        for config_path in sorted(Path("configs").rglob("*.yaml")):
            with config_path.open() as file:
                config = yaml.safe_load(file)
            expected = Path("runs") / config_path.relative_to("configs").with_suffix("")
            self.assertEqual(Path(config["run_dir"]), expected, config_path)
            self.assertNotIn(config["run_dir"], run_dirs, config_path)
            run_dirs.add(config["run_dir"])


if __name__ == "__main__":
    unittest.main()
