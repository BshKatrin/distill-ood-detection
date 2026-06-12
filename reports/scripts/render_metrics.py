"""Render the metrics report PDF from its LaTeX source."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


REPORTS_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPORTS_DIR / "output"
TEX_FILE = OUTPUT_DIR / "metrics.tex"


def main() -> None:
    """Compile ``reports/output/metrics.tex`` into ``metrics.pdf``."""

    if not TEX_FILE.exists():
        msg = f"Missing LaTeX source: {TEX_FILE}"
        raise FileNotFoundError(msg)

    if shutil.which("latexmk") is not None:
        command = ["latexmk", "-pdf", "-interaction=nonstopmode", TEX_FILE.name]
    elif shutil.which("pdflatex") is not None:
        command = ["pdflatex", "-interaction=nonstopmode", TEX_FILE.name]
    else:
        msg = "Neither latexmk nor pdflatex is available on PATH."
        raise RuntimeError(msg)

    subprocess.run(command, cwd=OUTPUT_DIR, check=True)


if __name__ == "__main__":
    main()
