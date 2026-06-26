"""Render the metrics report PDF from its LaTeX source."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


REPORTS_DIR = Path(__file__).resolve().parents[1]
LATEX_OUTPUT_DIR = REPORTS_DIR / "outputs" / "latex"
PDF_OUTPUT_DIR = REPORTS_DIR / "outputs" / "pdf"
LATEX_BUILD_DIR = REPORTS_DIR / "build" / "latex"
DEFAULT_TEX_GLOB = "metrics_*.tex"


def compile_latex(tex_file: Path) -> Path:
    """Compile one LaTeX metrics source into a PDF report."""

    tex_file = tex_file.resolve()
    if not tex_file.exists():
        msg = f"Missing LaTeX source: {tex_file}"
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
            tex_file,
        ]
    elif shutil.which("pdflatex") is not None:
        command = [
            "pdflatex",
            "-interaction=nonstopmode",
            f"-output-directory={LATEX_BUILD_DIR}",
            tex_file,
        ]
    else:
        msg = "Neither latexmk nor pdflatex is available on PATH."
        raise RuntimeError(msg)

    subprocess.run(command, cwd=REPORTS_DIR, check=True)

    built_pdf = LATEX_BUILD_DIR / f"{tex_file.stem}.pdf"
    if not built_pdf.exists():
        msg = f"Expected rendered PDF was not created: {built_pdf}"
        raise FileNotFoundError(msg)
    output_pdf = PDF_OUTPUT_DIR / built_pdf.name
    shutil.copy2(built_pdf, output_pdf)
    return output_pdf


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tex_files",
        nargs="*",
        type=Path,
        help=(
            "LaTeX files to compile. Defaults to all strategy-specific "
            "metrics_*.tex files."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Compile strategy-specific metrics LaTeX sources into PDFs."""

    args = parse_args()
    tex_files = args.tex_files or sorted(LATEX_OUTPUT_DIR.glob(DEFAULT_TEX_GLOB))
    if not tex_files:
        msg = f"Missing LaTeX sources matching {LATEX_OUTPUT_DIR / DEFAULT_TEX_GLOB}"
        raise FileNotFoundError(msg)

    for tex_file in tex_files:
        output_pdf = compile_latex(tex_file)
        print(f"Wrote {output_pdf}")


if __name__ == "__main__":
    main()
