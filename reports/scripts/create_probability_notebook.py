"""Create a generated probability-artifact analysis notebook."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reports.lib.notebook_generation import (
    NOTEBOOK_OUTPUT_DIR,
    filename_slug,
    next_notebook_number,
    render_notebook_template,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Experiment run directory containing probabilities/.",
    )
    parser.add_argument(
        "--title",
        help="Notebook title. Defaults to a title derived from the experiment name.",
    )
    parser.add_argument(
        "--number",
        type=int,
        help="Numeric notebook prefix. Defaults to the next available number.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=NOTEBOOK_OUTPUT_DIR,
        help="Directory where the generated notebook will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output notebook if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate a notebook from the probability-artifact template."""

    args = parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    with (run_dir / "resolved_config.json").open() as file:
        experiment_name = json.load(file)["experiment_name"]
    notebook_number = args.number or next_notebook_number(args.output_dir)
    output_path = (
        args.output_dir
        / f"{notebook_number:02d}_{filename_slug(experiment_name)}.ipynb"
    )
    if output_path.exists() and not args.overwrite:
        msg = f"Output notebook already exists: {output_path}"
        raise FileExistsError(msg)

    render_notebook_template(
        run_dir=run_dir,
        notebook_title=args.title,
        output_path=output_path,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
