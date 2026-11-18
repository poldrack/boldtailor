# NSD full-workflow notebook validation — 2026-11-15

The current notebook uses explicit analysis calls; see the follow-up validation
at the end of this record. The initial validation below describes the earlier
orchestrated version, retained as the numerical reference.

Updated `examples/bids_workflow/nsd_workflow.py` against checkout base
`84cf7148365cc769f132ad326c0b1c9c40d60990` in the isolated worktree on
`docs/nsd-entire-workflow`.

## Automated checks

- Baseline: 29 workflow/notebook safeguard tests passed.
- Committed two failing notebook tests before implementation (`02c9e05`).
  They execute the actual notebook against six synthetic BIDS runs and require
  preflight validation, saved CIFTI summaries, trial-aligned beta readback, and
  rejection of missing-session inputs before output publication.
- Updated notebook tests passed.
- Full suite: **2463 passed, 9 skipped, 2 xfailed** (213.75 seconds).

## Real-data execution

Executed all 13 code cells through nbclient using the worktree's uv environment.
The generated `examples/bids_workflow/nsd_workflow.ipynb` is intentionally ignored
by Git and retains all figures and tables locally. Jupytext synchronization after
formatting retained all execution counts and outputs; nbformat validation passed
and there are no error outputs.

- BIDS: `/Volumes/extdata1/NSD/BIDS`.
- fMRIPrep: `derivatives/fmriprep-25.2.5`.
- Subject/session/task: `sub-07`, `ses-nsd10`, `nsdcore`.
- All 12 runs, 750 trials, original 91,282-grayordinate output axis.
- Default library: 533 HRFs; default fractional ridge CV, all four stages enabled.
- Smoke-run spatial subset: 128 grayordinates, block size 128, four requested workers.
- Published 286 files, 13 figures, and a self-contained HTML report; no skipped stages.
- All three optimized-GLM R² maps have 123 finite grayordinates. Undefined and
  unprocessed locations remain NaN.
- Read back CIFTI map names, finite-value summaries, first-run optimized OLS beta
  rows aligned to the saved trial IDs, and resolved settings metadata.

Results occupy about 2.4 GB in this worktree's ignored `results/nsd_workflow/`:
CIFTI outputs retain the full spatial axis even for a subset. The report is
`sub-07_ses-nsd10_task-nsdcore_report.html` in that directory. No source BIDS or
fMRIPrep files were modified.

This validates end-to-end execution and output consumption on a spatial subset;
it is not a full-brain result or an independent scientific validation of the
models. For the full spatial analysis, use `max_grayordinates=None`,
`block_size=4096`, and a new output directory. Re-execution into the same output
directory deliberately stops at preflight unless overwrite is explicitly enabled.

## Follow-up: explicit analysis walkthrough

Replaced the orchestrated fit with explicit session loading, task model and block
construction, all/odd/even HRF selection, core `ModelSpec` / `fit` /
`task_delta_r2` calls, reliability calculations, separate canonical/optimized OLS
and ridge-CV calls, diagnostics, and publication. The notebook now exposes the
intermediate objects and explains the inner/outer/final CV results separately.
It calls neither `run_workflow` nor `fit_beta_models`. Publication helpers only
serialize the already computed objects.

The updated regression test failed before implementation and was committed as
`5ca3e39`. It disables both orchestration functions while executing the notebook,
checks the exposed stage results, and verifies saved maps against in-memory fits.
The synthetic fixture was strengthened to use two spatial blocks: all three
nonconstant grayordinates must appear in the exported maps. Both notebook tests
passed after this strengthening. The full suite passed before that test-only
strengthening: **2463 passed, 9 skipped, 2 xfailed** (211.93 seconds). Black and
Jupytext round-trip checks passed.

Executed all **20 code cells** on the same real-data inputs and default settings
(12 runs, 750 trials, 533 HRFs, 128 requested grayordinates, fractional ridge CV).
All stages completed; the resulting notebook retains the figures, tables, and
execution counts, passes nbformat validation, and contains no error outputs.
Results are in `results/nsd_workflow_explicit/`, leaving the earlier run intact.

Compared the explicit run with the original orchestrated run:

- Identical relative file sets: **286 files**, including **13 figures**.
- All **186 CIFTI files** have matching scalar and brain-model axes and matching
  numeric values (`rtol=1e-6`, `atol=1e-7`, equal NaNs).
- All **38 TSV tables** match using pandas frame equality.
- No skipped stages; the same spatial-subset interpretation limits apply.

The current HTML report is
`results/nsd_workflow_explicit/sub-07_ses-nsd10_task-nsdcore_report.html`.
JSON execution identities and output-root settings naturally differ; they were
not required to be byte-identical. Numerical and table parity is the check that
the expanded teaching implementation preserves the existing analysis.
