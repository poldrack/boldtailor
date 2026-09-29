# Notebook simplification and final architecture review

Date: 2026-09-28. Branch: `refactor/scientific-readability`.
Notebook increment baseline: `5269e72`. Full branch baseline: `57acff5`.

## Revised boundary

The user rejected promoting imaging workflows into the installed library.
NSD/CIFTI adapters remain in `examples/NSD/`; NIfTI operations remain in
`examples/stop_signal_demo.py`. This replaces the original review's packaging
recommendation. This increment changes no core source, runtime dependencies,
scientific estimators, or public modeling APIs.

Both NSD notebooks now delegate presentation details to small example-local
functions. Model definitions, HRF construction, selection, CV settings, and
fitting calls remain visible. Figure keys and exported scientific artifacts
are preserved. Direct tests check plotted arrays, acquisition times, peak-time
colors, paired masks, summary values, undefined selections, and session means.

NSD notebooks require `NSD_BIDS_ROOT` or an explicit `bids_root` configuration.
Optional preprocessing/output paths use explicit overrides, environment values,
then derivative defaults. No `.env` is read and path resolution writes no files.
The stop-signal notebook now requires nonempty `BOLDTAILOR_BIDS_ROOT`.
Setup guidance is in each notebook and the [NSD README](../../examples/NSD/README.md).

## Test-first evidence

| Change | RED | GREEN |
| --- | --- | --- |
| Example path and plot helpers | 13 failures for absent helpers; tests committed `f152dbd` | 45 helper/workflow/fractional/session tests passed in 29.00 s |
| Preserve expanded notebook path overrides | 1 failure: raw override replaced expanded path; test committed `de2e7c9` | 14 helper/path tests passed in 1.99 s |
| Require stop-signal data root | 3 failures for unset/empty/blank root, 6 existing configuration tests passed; tests committed `01e5380` | All 50 stop-signal tests passed in 10.84 s |

The initial constant-parameter test incorrectly assumed equal HRF durations:
canonical SPM uses 32 seconds and the custom fixture uses 36. Commit `4656fc9`
corrected the test to use their shared zero onset. No scientific requirement
was weakened. The preview fixture now explicitly supplies a data location.

Full suite before the final review fix: **885 passed in 162.82 s**.
Final reviewed full suite: **888 passed in 160.39 s**, with
`uv run pytest -q -W error`, after the stop-signal fix in `2cba131`.
Black checked 119 Python files; wheel build and isolated installed-package
smoke test passed. All three updated notebooks validate, compile, and have empty
outputs/execution counts. All package initializers remain empty; all 35 local
links in the initially changed guides/plans resolve. Black's notebook extra
is not installed, so notebook cells were checked by compilation and execution.

## Independent review

One read-only review examined `5269e72..36a7ece` closely and audited integration
across `57acff5..36a7ece`, using prior increment validation records. No extra
reviewers or second review pass were used. The reviewer confirmed the
example/core boundary, plot masks and anchors, lifecycle integration, composed
results, internal prepared-table access, and publication recovery contracts.

One important completion gap remained: a personal data-root default in the
stop-signal notebook. The executor accepted the finding, added and committed
failing tests, and fixed it in `2cba131`. The reviewer did not reinspect this
fix or independently rerun the suite; the executor verified it. No other
blocking findings were reported in the inspected boundaries.

Optional follow-ups are not prerequisites for this revised refactor:

- Replace older stop-signal AST/source-spelling assertions and notebook
  instrumentation with helper behavior tests when changing that workflow.
  Existing scientific/integration coverage is retained rather than deleted.
- Revisit private validation-helper imports between prepared modules if their
  ownership becomes confusing during future changes; no new framework now.
- Replace standalone NSD script personal CLI defaults in a separate CLI change.
  The notebook configuration change does not alter those commands; the NSD
  README explicitly distinguishes them.

## Architecture audit against the original review

| Area | Final disposition |
| --- | --- |
| Numerical preparation and CV generator coordination | Named preparation states and direct candidate evaluation; fixed OLS target reuses preparation. Raw fractional and normalized ridge bases remain distinct. |
| Array ownership and prepared-table copying | Ordinary owned read-only arrays, explicit editable copies, direct internal reads of owned prepared tables. |
| Result container duplication | Composed trial results and a shared candidate-score schema; migration guidance provided. |
| Fit lifecycle, logging and provenance | Shared lifecycle context, categorical export errors, direct provenance extension; redundant records removed. |
| Publication complexity | Ordinary-path staging, per-file replacement, rollback; retained backups and caller-visible recovery location on failed restoration. No atomic-set claim. |
| Imaging packaging and notebooks | User-revised scope: keep adapters in examples; extract presentation, retain visible science; default pytest includes NSD. |
| Distribution, CI and documentation | Lazy version resolution, kernel in dev dependencies, CI configuration, installed-wheel smoke check, current API/method guides and migration notes. Remote CI execution is not claimed. |

The numerical, ownership, result, lifecycle, and publication validation records
provide detailed evidence for earlier stages. The historical review examined
a broader working tree than the committed baseline; this audit does not claim
to restore independent scientific work from the preserved stash.

## Review scope and execution decisions

The executor accepted the reviewer's following scope limits:

- Ridge estimand, fixed-effects calibration, HRF transfer assumptions and
  constant-feature conventions remain separate scientific decisions. Changing
  them here would confound the readability refactor with scientific changes.
- Stashed scientific remediation remains preserved and unapplied. This means
  some historical findings about that broader tree require separate work.
- Delta-R² identity checks compare documented metadata, not signal contents.
  They cannot establish equality of two arbitrary BOLD arrays.
- Deliberate mutation of read-only arrays, hostile filesystem races, crash
  recovery and atomic visibility of an entire artifact set are excluded
  contracts; caller-facing documentation describes these limits.
- Broader provenance annotation-schema changes remain deferred; there is no
  blanket privacy guarantee for arbitrary caller metadata or artifact contents.
- Real-data fitting, remote CI, the known integer-image dependency behavior,
  and visual aesthetics were not independently exercised by the reviewer.
  Verification uses synthetic data, scientific regressions and installed-package
  checks; plot extraction tests assert data rather than aesthetic judgments.

Execution continued inline in the existing refactor checkout under the user's
approval; this gives less filesystem isolation than another worktree. Existing
notebook execution tests were retained because they exercise different supported
workflows. Some RT/ridge presentation loops remain where they expose analysis
choices. These decisions favor readable science and preserved coverage over a
line-count target. Main and the saved user changes remain untouched.

Final main revision remains `57acff54ebd04e9b86244755db2f149c8ef17c7c`;
stash `c4cfbec5fccd5397ea650c550f5b485ea588c2cd` remains intact. The unrelated
untracked NSD environment file and preview image were neither read nor changed.
No merge or push was performed.
