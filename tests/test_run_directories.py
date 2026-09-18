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
            if "run_dir" not in config:
                continue
            config_identity = config_path.relative_to("configs").with_suffix("")
            if config.get("strategy", {}).get("name") == "subspace_ensemble":
                config_identity = config_identity.with_name(
                    config_identity.name.removesuffix("_partitioned")
                ) / config["strategy"]["subspace_ensemble"]["assignment"]
            expected = Path("runs") / config_identity
            self.assertEqual(Path(config["run_dir"]), expected, config_path)
            self.assertNotIn(config["run_dir"], run_dirs, config_path)
            run_dirs.add(config["run_dir"])


if __name__ == "__main__":
    unittest.main()
