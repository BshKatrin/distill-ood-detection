# Recap Reports

Weekly or ad-hoc recap reports live under `reports/recap/`. These are short
LaTeX reports meant for human progress updates, for example a report to a
professor or supervisor.

## Folder Layout

Each recap should have its own dated subfolder:

```text
reports/recap/
  Makefile
  09072026/
    main.tex
    schema.tex
    main.pdf
```

Use the date folder as the report identifier. The existing folder name
`09072026` follows the current project convention. Keep all source files needed
for that recap inside the dated folder unless they are shared report outputs
from `reports/outputs/`.

The usual files are:

| Path | Purpose | Tracked |
| --- | --- | --- |
| `reports/recap/<date>/main.tex` | Main report source | Yes |
| `reports/recap/<date>/schema.tex` | Small standalone diagram or supporting figure source | Yes |
| `reports/recap/<date>/main.pdf` | Compiled recap deliverable | Yes |
| `reports/build/recap/<date>/` | LaTeX build artifacts and intermediate PDFs | No |

## Build With Make

Compile recaps from the `reports/recap/` directory:

```bash
make -C reports/recap 09072026
```

The Makefile first builds `schema.tex`, then builds `main.tex`, and finally
copies the compiled PDF back to the dated recap folder:

```text
reports/build/recap/09072026/main.pdf
reports/recap/09072026/main.pdf
```

The first path is the compiler output under the scratch build tree. The second
path is the report deliverable that should be kept with the recap sources.

## Build Artifacts

LaTeX scratch files should stay under:

```text
reports/build/recap/<date>/
```

This includes files such as `.aux`, `.fdb_latexmk`, `.fls`, `.log`, `.out`, and
intermediate PDFs. Do not keep these files inside `reports/recap/<date>/`.

The only generated file expected in the dated recap folder is `main.pdf`,
because it is the easy-to-open final report.

Clean recap build artifacts with:

```bash
make -C reports/recap clean
```

This removes `reports/build/recap/` and the copied recap PDF for the currently
listed Makefile target.

## Adding A New Recap

To add another recap:

1. Create a new dated folder under `reports/recap/`.
2. Add at least `main.tex`; add `schema.tex` if the report needs a compiled
   diagram.
3. Add a Makefile target for the new date.
4. Make the target write compiler artifacts to `reports/build/recap/<date>/`.
5. Copy the final `main.pdf` back into `reports/recap/<date>/`.

Keep the source simple and explicit. Recaps are meant to be quick progress
summaries, not reusable report-generation infrastructure.
