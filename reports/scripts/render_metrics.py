"""Render the metrics report PDF from its LaTeX source."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


REPORTS_DIR = Path(__file__).resolve().parents[1]
LATEX_OUTPUT_DIR = REPORTS_DIR / "outputs" / "latex"
PDF_OUTPUT_DIR = REPORTS_DIR / "outputs" / "pdf"
LATEX_BUILD_DIR = REPORTS_DIR / "build" / "latex"
TEX_FILE = LATEX_OUTPUT_DIR / "metrics.tex"


def main() -> None:
    """Compile ``reports/outputs/latex/metrics.tex`` into ``metrics.pdf``."""

    if not TEX_FILE.exists():
        msg = f"Missing LaTeX source: {TEX_FILE}"
        raise FileNotFoundError(msg)

    LATEX_BUILD_DIR.mkdir(parents=True, exist_ok=True)
    PDF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if shutil.which("latexmk") is not None:
        command = [
            "latexmk",
            "-g",
            "-pdf",
            "-interaction=nonstopmode",
            f"-outdir={LATEX_BUILD_DIR}",
            TEX_FILE,
        ]
    elif shutil.which("pdflatex") is not None:
        command = [
            "pdflatex",
            "-interaction=nonstopmode",
            f"-output-directory={LATEX_BUILD_DIR}",
            TEX_FILE,
        ]
    else:
        msg = "Neither latexmk nor pdflatex is available on PATH."
        raise RuntimeError(msg)

    subprocess.run(command, cwd=REPORTS_DIR, check=True)

    built_pdf = LATEX_BUILD_DIR / f"{TEX_FILE.stem}.pdf"
    if not built_pdf.exists():
        msg = f"Expected rendered PDF was not created: {built_pdf}"
        raise FileNotFoundError(msg)
    shutil.copy2(built_pdf, PDF_OUTPUT_DIR / built_pdf.name)


if __name__ == "__main__":
    main()
