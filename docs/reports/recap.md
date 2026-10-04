# Recaps and final-report builds

The final internship report lives in `reports/final-report/`; progress reports
live in `reports/recaps/YYYY-MM-DD/`. Use ISO dates for new recaps. Keep the
main source and report-specific supporting sources/data together, and reference
shared tables or experiment figures through their existing locations.

## Build commands

From the repository root, with `latexmk` and a compatible TeX installation:

```bash
make -C reports final-report
make -C reports recap DATE=2026-07-09
# Equivalent direct recap command:
make -C reports/recaps 2026-07-09
```

Builds are explicit requests to regenerate the delivered PDF. The final report
uses `\today`, so a later build can change its title-page date; preserve the
committed delivery PDF when only organising files or consulting findings.

| Report | Compiler output | Delivered PDF |
| --- | --- | --- |
| Final | `reports/build/final-report/main.pdf` | `reports/final-report/main.pdf` |
| Recap | `reports/build/recaps/<date>/main.pdf` | `reports/recaps/<date>/main.pdf` |

The recap Makefile runs from `reports/recaps/`. Diagram sources are built before
the main document and found through its build-directory graphic path. The
2026-07-28 recap also builds its table appendices. The 2026-08-17 recap requires
the experiment plots and local `runs/` dendrogram images referenced by its source.

## Tracking and scratch files

Track new recap sources, necessary supporting assets, and the delivered
`main.pdf`. The six historical recap folders retain their previous ignored
status; their Makefile targets require those local archives to be present.
See the [catalogue](../../reports/README.md) for the date mapping and
[tracking policy](README.md#artifact-ownership-and-git-policy).

Compiler files such as `.aux`, `.out`, `.toc`, `.log`, `.fls`, and
`.fdb_latexmk` belong under `reports/build/`. Existing scratch files were moved
into `legacy/` subfolders there without changing delivered PDFs. Earlier build
directories are preserved in `previous-build/` subfolders under the new paths.

```bash
make -C reports clean
# Only recap scratch files:
make -C reports/recaps clean
```

Cleanup removes build scratch directories and preserves deliverable PDFs.

## Add a recap

1. Create `reports/recaps/YYYY-MM-DD/main.tex` and supporting files.
2. Add an explicit dated target to `reports/recaps/Makefile`.
3. Compile from the recap root, writing scratch files below
   `reports/build/recaps/YYYY-MM-DD/`; build required diagrams/appendices first.
4. Copy only the final `main.pdf` back to the dated source folder.
5. Record the datasets, configs, checkpoints, OOD Scores, metric convention,
   units, and source experiment records alongside the narrative.
