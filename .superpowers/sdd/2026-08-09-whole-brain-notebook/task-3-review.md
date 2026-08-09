# Task 3 Senior Re-review

## Scope

- Review range: `1065890294a7d1520dffc6928177fa5613b7f729..5dac551`
- Review mode: fresh read-only review except for this requested uncommitted review artifact
- Sources reviewed in full: Task 3 brief, rewritten Task 3 report, frozen review diff, previous Task 3 review, changed helper/test/README content, and notebook changes
- History audited: test-only `073fa538dd23b095ebd325bf4ecafbfbfa605bea` precedes production `3e5d418d589332b44a4cd72b7a84e07aaa6b3d92`

## Findings

### Critical

None.

### Important

None. Both Important findings from the previous review are fully addressed.

### Minor

None.

## Resolution of previous Important findings

1. **Exact model and plotting behavior is now regression-protected.** The test-only commit instruments the executed notebook around the real `plotting.plot_stat_map`, records its bound arguments, and compares each plotted image's values inside the actual common mask with the fitted result arrays. Both supported working-directory executions require exactly four calls: one z-score map for each of the three exact contrasts, followed by the aggregate `result.r2` map. The tests require `threshold=None` and `colorbar=True` for all four calls, plus `cmap="viridis"` and `symmetric_cbar=False` for aggregate R-squared. They also require the exact rendered multiple-comparison disclosure. The notebook-published fit provenance is checked for exactly these expressions and no extras:

   - `successful_inhibition`: `stop_success - stop_failure`
   - `stop_vs_go`: `(stop_success + stop_failure) - go_success`
   - `go_success_vs_baseline`: `go_success`
   - noise model: `ar1`

2. **Actual notebook-produced configuration and provenance are now regression-protected.** A behavior test executes the notebook, opens its published `desc-example_config.json` and canonical provenance JSON, and asserts exact mask metadata, exact non-preprocessing masker settings, exact memory/chunking assumptions, exact transformed signal shapes and `float64` dtypes, exact contrasts, session/task identity, and whole-brain example label. It checks the resolved fixture BIDS root is absent from rendered output and all shareable, non-binary published text. Exact equality of the configuration and normalization metadata also prevents `BIDS_ROOT` or any other unapproved key from entering those structures.

## Requirement audit

- Strict TDD history: **Pass.** `073fa53` changes only `tests/test_stop_signal_demo.py`; `3e5d418` follows it with production files. The only test changes in the production commit are Black line wrapping. Inspection of the base notebook confirms the strengthened tests genuinely distinguish RED from GREEN: the base has one statistical-map call, the old contrast expression, ROI metadata, and no whole-brain published metadata. The supplied detached replay at `073fa53` produced the expected `3 failed` before production.
- Both notebook working directories: **Pass.** The strengthened parameterized execution test covers repository-root and notebook-directory kernels.
- Shared whole-brain masking and arrays: **Pass.** One fitted intersection-mask `NiftiMasker` is reused for both runs with all preprocessing options disabled. Tests require owned, write-protected `float64` arrays and identical cross-run feature ordering for the same spatial pattern.
- Model and presentation: **Pass.** Exact three-expression/AR(1) provenance and exact three-z-plus-aggregate-R-squared plotting behavior are now tested, including plot arguments and disclosure.
- Derivatives: **Pass.** Ten deterministic NIfTI images and the image-manifest TSV retain round-trip geometry/value, ordering, media type, size, and SHA-256 coverage.
- Provenance/configuration/privacy: **Pass.** Tests inspect actual notebook publications rather than caller-supplied unit-test metadata, and reject absolute dataset-root leakage from rendered and shareable published text.
- ROI/legacy removal: **Pass.** Production notebook/helper/README contain no case-insensitive ROI term. The ROI dispatch, `common_roi_voxels`, `roi_image`, ROI extraction helpers, legacy `LoadedRun` spatial fields, and all-omitted artifact-context branch are absent.
- Publication safety: **Pass.** Temporary output remains the default; persistent output is restricted to the exact derivative destination with symlink, protected-source, overwrite, and transactional safeguards preserved.
- README, modularity, and initialization: **Pass.** README accurately labels the two-session common-mask whole-brain notebook, helper functions remain short and focused, and package initialization remains empty.
- Formatting, lock, and diff hygiene: **Pass.** No dependency or lock change is present; supplied fresh Black/lock checks and independent diff checks are clean.

## Verification evidence

- Fresh current-head strengthened nodes run during this review:
  `PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/test_stop_signal_demo.py -k "notebook_executes or publishes_complete_private_metadata" -q -W error -p no:cacheprovider`
  -> `3 passed, 29 deselected in 11.14s`.
- Supplied fresh detached RED replay at final test-only commit `073fa53` -> `3 failed` as expected: both working-directory cases observed one plot call instead of four, and the metadata node found the old ROI/incomplete publication.
- Supplied fresh focused suite -> `32 passed`.
- Supplied fresh full warning-strict suite -> `219 passed`.
- Supplied fresh formatting/lock verification -> Black clean (`24 files would be left unchanged`) and `uv lock --check` clean.
- Independent history audit -> `073fa53` contains only the test file; production follows in `3e5d418`; its test-file delta is formatting only.
- Independent production audit -> no case-insensitive ROI or removed legacy bridge symbol in `examples/stop_signal_demo.py`, `examples/stop_signal_demo.ipynb`, or `README.md`.
- Independent `git diff --check 1065890..5dac551` and worktree `git diff --check` -> exit 0.

## Verdict

- TDD: **Pass**
- Functional specification: **Pass**
- Test quality: **Pass**
- Code quality: **Pass**
- Ready: **Yes.** Task 3 is ready for integration; the two prior Important coverage gaps are fully resolved by behavior tests that precede production.
