# NSD full-workflow notebook validation — 2026-11-15

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
