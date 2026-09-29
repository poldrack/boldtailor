# NSD multi-session notebook implementation plan

**Goal:** Compare full HRF shapes and canonical versus optimized beta-series
results at each grayordinate across sub-07 sessions 10–19.

**Design:** Reuse the single-session workflow as the producer. Automatically run
it for sessions without complete requested outputs; completed sessions are
reused without fitting. Read compact exported summary maps rather than holding
all trial betas across sessions in memory. Require matching CIFTI axes and exact
HRF libraries. Reuse the existing full-curve correlation implementation.

**User decisions:** Include both mean trial beta and task-added R², plus RT
associations. Default to sessions 10–19 and automatic fitting of missing sessions.

**Comparisons:** All unordered session pairs get full-HRF Pearson r over the
stored time grid, without temporal shifts; include canonical shape similarity
as a baseline. For OLS and fractional-CV separately, compute within-session
optimized-minus-canonical mean beta, descriptive task t, task-added R², signed
RT r, and absolute RT r. Aggregate matched finite pairs with equal session
weights. Export contributing counts and SD of changes. Require two valid
sessions for across-session summaries. No inference treats overlapping HRF
pairs or trial betas as independent observations. RT and fractional-CV summaries
are descriptive because RT participates in tuning. Raw beta differences retain
native signal units; differences of task t statistics are not t tests.

**Files:** `multisession_inputs.py` validates compact exports;
`multisession_workflow.py` prepares missing sessions;
`multisession_analysis.py` computes paired summaries;
`multisession_plots.py` provides figures;
`multisession_outputs.py` publishes CIFTI/TSV/JSON/figures;
`nsd_multisession.ipynb` orchestrates the analysis. Tests live beside these helpers.

**Constraints:** uv; pytest functions/fixtures; failing tests committed before
implementation; empty `__init__.py`; short modular functions. Preserve existing
user notebook outputs and local configuration.

## Tasks

- [ ] 1. Write, run RED, and commit tests for saved-map loading, matching axes and
  libraries, missing sessions, and paired summaries with asymmetric missingness.
  Implement loaders and analysis; run GREEN checks.
- [ ] 2. Test automatic fitting on two small synthetic sessions, then forbid
  fitting on rerun and verify reuse. Implement sequential notebook execution
  with explicit configuration and no fitting of completed sessions.
- [ ] 3. Test the new notebook end to end and verify exported numerical maps.
  Build the notebook, plots, and publication helper; document defaults and
  interpretation. Run focused tests and full pytest, inspect rendered figures,
  review changes, and commit implementation.

## Review focus

- Spatial axes, scalar names, library fingerprints and full curves must agree.
- Settings conflicts or incomplete outputs must never silently mix analyses.
- Match canonical/optimized finite values before averaging; retain NaNs/counts.
- Reuse must not fit or modify completed per-session numerical results.
- Changing plotting settings must not require refitting; missing sessions have
  explicit status and excluded data are never represented as measured zeros.
