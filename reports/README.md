# Reports and deliverables

| Deliverable | Location |
| --- | --- |
| Final internship report, 18 pages | [PDF](final-report/main.pdf), [LaTeX source](final-report/main.tex) |
| Final presentation, 14 slides | [PDF](presentations/internship/presentation.pdf) |
| Dated progress recaps | `recaps/YYYY-MM-DD/` (historical folders remain local/ignored) |

The final-report folder contains its chapter sources, tables, references, and
logo. The presentation PDF was exported from Google Slides; editable slide
sources are not included in this repository. `final-report/archive/` preserves
the separate local `report-final-full.pdf` export without making it the
canonical deliverable.

## Supporting material

- [Experiment catalogue](../experiments/README.md): curated results, configurations,
  and supporting figures referenced by the report.
- `scripts/`, `lib/`, and `templates/`: existing reporting commands, helpers,
  and notebook templates.
- `outputs/`: generated tables and working exports.
- `build/`: ignored compiler scratch files, including preserved legacy scratch files.

## Progress archive

The six existing recaps were renamed from `DDMMYYYY` to sortable ISO dates:

| Previous folder | Current folder |
| --- | --- |
| `09072026` | `recaps/2026-07-09/` |
| `17072026` | `recaps/2026-07-17/` |
| `28072026` | `recaps/2026-07-28/` |
| `17082026` | `recaps/2026-08-17/` |
| `26082026` | `recaps/2026-08-26/` |
| `31082026` | `recaps/2026-08-31/` |

Their sources, compiled PDFs, and supporting data are preserved locally.
Existing ignore rules explicitly retain their previous untracked status.
New recap sources and deliverable PDFs can be tracked normally.

See [reporting documentation](../docs/reports/README.md) for commands and the
tracking policy, and [build instructions](../docs/reports/recap.md) for report builds.
