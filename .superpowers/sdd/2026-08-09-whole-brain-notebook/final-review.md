# Final Whole-Branch Senior Review

## Scope

- Reviewed range: `3f44d7e5c7cb82f92103d1486ce8a046efc794c4..b6c10b871d1af825cb06bdfe2621bc3708319cef`
- Review mode: read-only except for this requested uncommitted review artifact
- Read in full: approved design, implementation plan, Task 1-4 reports, Task 1-3 review files, and the frozen whole-branch review diff
- Audited the final notebook, helper module, focused tests, package publication/provenance paths, and test/production commit ordering

## Strengths

- The history preserves literal test-before-production ordering. `9e2bf11`/`37eb1f8` precede `e55663d`, `fbcb961`/`720b851` precede `7166422`, and rewritten Task 3 test commit `073fa53` precedes `3e5d418`. Test-only commits contain no production files; the Task 3 production commit changes the test file only through Black formatting. The masker dimensionality correction is justified by Nilearn's actual 3D transform behavior and does not weaken the requirement.
- The final workflow computes a nonempty logical mask intersection, requires matching mask and BOLD geometry, fits one shared `NiftiMasker`, and explicitly disables standardization, detrending, smoothing, temporal filtering, and reports. Both runs retain the same feature order and become owned, immutable `float64` arrays without chunking, truncation, downsampling, or fallback resampling.
- The notebook discloses the 32 GiB assumption and no-chunking behavior, displays scan/voxel/shape/memory summaries, fits run-specific designs with AR(1), and uses exactly the three approved contrast expressions. It plots exactly three descriptive unthresholded z maps and one aggregate R-squared map with the required disclosure.
- Publication emits exactly ten deterministic gzip-compressed NIfTI images in stable order: the common mask, three effect maps, three z maps, two run R-squared maps, and aggregate R-squared. Focused tests independently verify paths, values, geometry, gzip round-trip, manifest ordering, `application/gzip`, byte sizes, and SHA-256 digests.
- Both source masks are retained as dataset-relative signal annotations and therefore participate in the data fingerprint. The canonical/projected provenance and shareable configuration contain the mask strategy/shape/affine/voxel count, exact masker settings, resource assumptions, transformed signal shapes/dtypes, sessions, and contrasts without exposing the absolute dataset root.
- Temporary publication is the default. Persistent output is restricted to exactly `<dataset>/derivatives/boldtailor`, while transactional publication retains traversal, overlap, symlink, collision, overwrite, rollback, and privacy-safe failure handling.
- Final production state contains no ROI terminology or migration bridge. README wording is accurate, helpers are modular, the package initializer is empty, and the lock/package contracts are unchanged.
- Synthetic notebook execution is covered from both repository-root and notebook-directory kernels. Task 4 records successful real `ses-06`/`ses-08` execution in 23.87 seconds with all required rendered fields and persistent destination absent before and after.

## Issues

### Critical

None.

### Important

None.

### Minor

1. `.superpowers/sdd/2026-08-09-whole-brain-notebook/progress.md:14` still marks Task 4 as incomplete even though `task-4-report.md` is committed at the reviewed head. This is a stale project-status record, not a functional defect. Fix: mark Task 4 complete and reference the Task 4 report/head.

2. `examples/stop_signal_demo.py:187` and `examples/stop_signal_demo.py:219` validate only that the four spatial-context arguments are non-`None`; they do not verify that `common_mask` matches `masker.mask_img_` in shape, affine, and voxel membership. The notebook constructs the masker from that exact mask, so every reviewed execution and published artifact is correct. A direct caller can nevertheless supply inconsistent objects and receive a mixed-geometry ten-image set. Fix: compare the supplied mask with the fitted mask before serialization and add a RED test that rejects shape, affine, and voxel-content disagreement.

## Verification

Fresh current-head checks performed during this review:

- `PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/test_stop_signal_demo.py -q -W error -p no:cacheprovider` -> `32 passed in 11.81s`
- `PYTHONDONTWRITEBYTECODE=1 uv run pytest -q -W error -p no:cacheprovider` -> `219 passed in 13.61s`
- `uv run black --check src tests examples/stop_signal_demo.py` -> `24 files would be left unchanged`
- `uv lock --check` -> `Resolved 76 packages`
- Empty-initializer check -> `checked: 1, nonempty: []`

An additional isolated repository-contract invocation encountered a sandbox-only uv cache permission error and its escalation was interrupted. This does not leave the contract unreviewed: the fresh full 219-test run above includes `tests/test_repository_contracts.py`, and Task 4 separately records `2 passed` for that file. Task 4 also supplies the successful build and real-data evidence; those expensive checks were not repeated during this final read-only review.

## Recommendations

- Treat both Minor items as follow-up cleanup; neither changes the notebook's verified analysis or publication behavior.
- If the spatial-context validation is addressed before merge, preserve strict RED-GREEN history with a committed failing test before the helper change.
- Do not rerun real data merely for the stale ledger correction.

## Verdict

**READY TO MERGE.** The branch satisfies the approved whole-brain notebook specification and has no Critical or Important findings. The two Minor findings do not affect the verified notebook path, published derivatives, privacy guarantees, or persistent-output safety.
