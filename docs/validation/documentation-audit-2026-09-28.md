# Documentation audit — 2026-09-28

This audit checks the documentation against the `refactor/scientific-readability`
branch after the results/diagnostics implementation at `2b126fc`.
It does not claim to rerun historical real-data experiments.

## Coverage

- Read the project README, API reference, user/developer guides, both migration
  guides, GLMsingle comparison, and NSD README against source signatures,
  result fields, method implementations, example helpers, and existing tests.
- Read Markdown cells in all three notebooks and checked their explanations
  against the corresponding helper modules. Stop-signal publication wording
  and the NSD description of OLS versus ridge fits were corrected without
  changing code cells, outputs, or metadata.
- Inspected package and example docstrings. Corrected stale shared-alpha-only
  descriptions and claims of atomic publication of a complete artifact set.
- Inventoried all Markdown plans, specs, reviews, and validation reports.
  Marked dated records explicitly and linked them to the current documentation
  index. Historical scientific measurements and test counts are preserved.
- Checked local Markdown/notebook links and current Python documentation
  snippets. Verification results are recorded below.

## Corrections

| Area | Correction |
| --- | --- |
| Result API | Candidate scores use `grid`/`regularization`; trial results use `design.matrices`, with assignment/provenance under selected designs |
| Fractional prediction | OLS targets stay fixed across fractions within a split/HRF model; shared-alpha targets remain candidate-regularized |
| Examples | Encoding snippets explicitly require multiple runs and numeric category coding |
| HRF workflows | Distinguished standalone fixed-HRF commands, selected-HRF notebook GLMs, dedicated saved-map reuse, and absence of a general selection-object loader |
| Libraries | Distinguished the 649-entry expanded grid from the notebooks' default 513-entry Sobol library |
| Reliability | Identified which notebooks compute curve correlations/parameter variability; none claims ICC inference |
| Logging | Removed the blanket sanitization guarantee: conventional/grouped error logs can contain raw exception text; prepared fits sanitize separately |
| Publication | Clarified writer locking, individual file replacements, rollback, concurrent-reader limits, and preflight exception types |
| Historical evidence | Marked the NSD fractional audit's older normalized basis/candidate targets and the missing historical simulation-design file |
| Navigation | Added a documentation index and repaired a notebook relative link |

## External method references

The GLMsingle comparison's HRF-selection, repeated-condition tuning, kernel
normalization, and output descriptions were checked against its
[official FAQ](https://glmsingle.readthedocs.io/en/latest/wiki.html) and
[Python output reference](https://glmsingle.readthedocs.io/en/latest/python.html#returns).
The local GLMsingle checkout still reports the documented revision
`1de98a92e80754ff549c87b5c5b815b1aab8148c`; its selector uses peak normalization.
The eLife article URL returned a browser challenge during this audit; the
historical paper citation was retained, without claiming a fresh full-paper review.

## Verification

- All relative file/directory link targets passed across **61 Markdown/notebook
  documents**, including contributor instructions; local heading anchors passed.
- All **17 current Python Markdown snippets** compiled.
- Both README examples executed successfully. The API publication example
  ran against that fit in a temporary directory and saved all its artifacts.
- Python AST comparison after removing docstrings found no executable change.
  Notebook comparison confirmed code cells, outputs, cell identities, and
  metadata were unchanged.
- Black passed all 113 Python files; `git diff --check` passed.
- The underlying results/diagnostics code passed all **801 tests** in 165.64
  seconds, wheel build, and isolated installed-package smoke verification.
  One independent reviewer found no actionable code findings.

The snippets needing dataset-specific variables were syntax/import/API checked
against the implementations and tests, not all executed as standalone analyses.
Both NSD notebooks and the stop-signal notebook ran with synthetic fixtures in
the full suite before these Markdown-only edits.

## Limits and maintenance

Historical reports refer to their recorded revisions and settings, sometimes
including broader working trees. Their code examples and test counts are
historical evidence, not executable current-API guidance. Current user guidance
lives in the documents linked from [the index](../README.md).

No new numerical experiment, remote CI run, or real-data benchmark was performed
for this documentation audit. Fit lifecycle/logging unification, provenance,
publication simplification, and packaged imaging remain roadmap work. Future
increments must update the current guides and migration notes when APIs or
scientific definitions change.
