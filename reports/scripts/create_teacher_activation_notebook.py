"""Create a generated teacher-activation analysis notebook."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from distill_ood_detection.config import load_teacher_activation_config
from reports.lib.notebook_generation import (
    NOTEBOOK_OUTPUT_DIR,
    next_notebook_number,
    render_teacher_activation_notebook_template,
    teacher_activation_filename_slug,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        type=Path,
        help="Teacher activation YAML config under configs/teachers/.",
    )
    parser.add_argument(
        "--title",
        help="Notebook title. Defaults to a title derived from the teacher config.",
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
    """Generate a notebook from the teacher activation template."""

    args = parse_args()
    config = load_teacher_activation_config(args.config)
    notebook_number = args.number or next_notebook_number(args.output_dir)
    output_path = (
        args.output_dir
        / f"{notebook_number:02d}_{teacher_activation_filename_slug(config)}.ipynb"
    )
    if output_path.exists() and not args.overwrite:
        msg = f"Output notebook already exists: {output_path}"
        raise FileExistsError(msg)

    render_teacher_activation_notebook_template(
        config=config,
        notebook_title=args.title,
        output_path=output_path,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
