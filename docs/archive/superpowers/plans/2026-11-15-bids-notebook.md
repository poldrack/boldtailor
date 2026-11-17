# BIDS notebook refresh implementation plan

> **For agentic workers:** Execute inline using the requested isolated worktree and RED-GREEN-Refactor.

**Goal:** Demonstrate the current full workflow on /Volumes/extdata1/NSD/BIDS and show how to consume its outputs.

**Architecture:** Keep the tracked Jupytext notebook as the entry point, using public workflow APIs. Validate inputs before fitting and read saved CIFTI/table/metadata outputs after every stage. Keep generated results in the worktree by default.

**Tech Stack:** uv, pytest, Jupytext, nbclient, pandas, nibabel.

**Spec:** User request in this session: update examples/bids_workflow for the latest code, use the supplied NSD data, and isolate changes in a worktree.

## Constraints and review focus
- Empty __init__.py files; short notebook cells; uv for Python commands.
- Commit failing tests before notebook changes.
- Environment and NSD_CONFIG path overrides must remain usable.
- A spatial smoke run exercises every stage but is not a full-brain scientific result.
- Missing inputs and existing outputs must stop before fitting.
- Read CIFTI names and axes, retaining NaNs outside selected grayordinates.
- Pair per-run trial rows with beta rows by saved trial IDs.

## Task 1: Full executable walkthrough
- [x] Add pytest execution of the notebook against synthetic six-run BIDS data, checking preflight, full-stage outputs, map summaries, and aligned beta readback; test invalid inputs.
- [x] Run and commit failing tests.
- [x] Update nsd_workflow.py and its README with local paths, preflight, current model interpretation, saved output examples, and execution instructions.
- [x] Run focused tests and the full suite.
- [x] Execute a generated notebook on all real session runs with a small grayordinate subset; preserve the executed notebook locally and document validation limits.
- [x] Review and commit notebook, documentation, and validation record.

## Decisions
Execute in this session under the user's authorization to update the notebook; no additional approval checkpoint is needed. Use the current checkout commit as the worktree base. Results default to a local ignored results directory so dataset derivatives remain untouched.

Validation evidence: `docs/validation/nsd-notebook-2026-11-15.md`.

## Follow-up request: clarity through explicit API calls

- [x] Commit a failing test that disables the two workflow orchestrators and requires inspectable stage results.
- [x] Replace the runner cell with explicit loading, selection, GLM, reliability, beta/CV, diagnostic, and publication sections.
- [x] Explain intermediate data structures, selection versus independent test scores, and the retained blockwise/CV API boundaries.
- [x] Execute synthetic tests (including two spatial blocks), full suite, and real NSD notebook.
- [x] Compare all real-data CIFTI and TSV outputs with the original run; record numerical parity in the validation document.
