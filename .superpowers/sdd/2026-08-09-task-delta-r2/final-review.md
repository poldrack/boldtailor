# Final senior code review

**Range:** `6fc4cc789719f41616e17c23ecf778dcd84382ae..d8a0480d8d07c460931cbea2394d40ca686f66c1`

### Strengths

- The final implementation matches the statistical design: nuisance designs contain the exact selected confounds, drift terms, and constant but no event-derived regressors; both models use the requested noise model; and nuisance residual/centered-total sums are pooled across runs before R-squared is calculated (`src/boldtailor/design.py:30`, `src/boldtailor/design.py:98`, `src/boldtailor/fit.py:194`). Parent analysis fingerprint and run/feature dimensions are checked before comparison (`src/boldtailor/fit.py:214`, `src/boldtailor/fit.py:226`).
- Raw and clipped delta invariants, negative diagnostics, defensive nuisance-design access, and success/failure lifecycle provenance are cleanly separated and well tested (`src/boldtailor/results.py:158`, `src/boldtailor/fit.py:108`, `tests/test_logging.py:276`). Provenance records the parent identity, exact nuisance settings and columns, clipping definition, and diagnostics without design values or absolute paths.
- The notebook and publication story is unusually thorough: the fifth map has the exact required data/options/title, the compact variance-partition metadata is live-derived, and the deterministic eleventh NIfTI participates in the exact ordered manifest with geometry/value/hash checks (`examples/stop_signal_demo.py:323`, `examples/stop_signal_demo.py:408`, `tests/test_stop_signal_demo.py:763`). Temporary-default and persistent path, symlink, overwrite, protected-source, and transaction behavior remain routed through the existing safety layer.
- The commit history contains committed test-only RED states before each production behavior and review fix. In particular, the raw-serializer and raw-display false positives received genuine regression tests before their fixes (`a1a9a58..0a40135` and `77712ec..a3c4928`). No test was weakened.
- Supplied fresh controller evidence is strong: real `ses-06`/`ses-08` execution completed with 11 images and no persistent derivative, 38 focused and 246 full warning-strict tests passed, Black/lock/build/repository-contract/initializer gates passed, and the standalone status/diff retry was clean. I did not rerun tests during this read-only review.

### Issues

#### Critical (Must Fix)

None.

#### Important (Should Fix)

1. **The four returned delta-result arrays do not satisfy the literal owned-storage contract.**
   - File: `src/boldtailor/results.py:165-174`; backing helper: `src/boldtailor/_arrays.py:6-9`; incomplete assertion: `tests/test_fit.py:589-603`
   - `make_task_delta_r2_result()` routes `full_r2`, `nuisance_r2`, `raw_delta_r2`, and `delta_r2` through `immutable_float_array()`. That helper creates an array with `np.frombuffer(contiguous.tobytes())` and reshapes it, so the ndarray is backed by a `bytes` base and reports `flags.owndata == False`. The new test proves non-aliasing and strict read-only behavior but never asserts owned storage, despite both the design and implementation plan requiring all four properties to be owned immutable float64 arrays.
   - Why it matters: this is a direct public-result invariant, not a cosmetic flag or merely missing test coverage. The branch currently claims and tests only three of the four promised array properties.
   - Fix: first commit a failing assertion that every returned array has `flags.owndata` while retaining the existing dtype, shape, no-alias, and `setflags(write=True)` rejection checks. Then use an owning strictly immutable ndarray representation (for example, a narrowly scoped owning ndarray subclass that refuses write re-enablement) and rerun the focused and full warning-strict gates.

#### Minor (Nice to Have)

1. **Direct comparison-level coverage for multi-run pooled R-squared and zero-SST remains absent.**
   - File: `tests/test_fit.py:96-144`, `tests/test_fit.py:571-603`; implementation: `src/boldtailor/fit.py:194-211`
   - The task-specific AR(1) fixture is single-run and checks comparison self-consistency/inequality rather than independently reconstructing nuisance predictions and pooled sums. Zero-SST is exercised through `fit()` but not through `task_delta_r2()` (`tests/test_fit.py:214`).
   - Why it remains Minor: the comparison uses the same `_fit_glm`, `_sums_of_squares`, and `_r2_from_sums` helpers whose multi-run AR(1), original-signal-space, pooled-sum, and zero-SST behavior already has direct lower-level coverage (`tests/test_multirun.py:178-215`). The task-specific pooling code is short and presently correct, so the residual risk is regression protection rather than an observed defect.
   - Fix: add a two-run AR(1) oracle with deliberately unequal run SSTs that independently reconstructs nuisance RSS/TSS and distinguishes pooling from averaging, plus a task-level zero-SST case asserting the intended error/log lifecycle.

2. **The raw-signal display regression audit can be bypassed by a copied, indirectly named slice.**
   - File: `tests/test_stop_signal_demo.py:89-121`, `tests/test_stop_signal_demo.py:276-325`
   - Runtime detection relies on `np.shares_memory`, while the static check only rejects direct `.signals`/`signals` references passed to `display()` in `design-fit`. A regression such as assigning `loaded_runs[0].signals[:, :1].copy()` to an innocuous intermediate name and displaying it would share no memory, remain below the 256-element ceiling, and evade the direct-argument AST check.
   - Why it matters: the current notebook is clean on direct inspection, so this is a test limitation only; however, the prior false-positive exercise shows that this privacy guard is intended to prevent exactly this class of regression.
   - Fix: add that concrete copied-slice mutation as a failing regression and strengthen the audit with taint/dataflow tracking or an exact allowlist of permitted notebook display expressions across the relevant cells.

### Recommendations

- Route the owned-storage fix through the repository's required committed RED-GREEN sequence, then repeat the focused comparison tests, full warning-strict suite, formatting, build, and repository-contract gates.
- Keep the deferred pooled/zero-SST item at Minor severity, but close it while touching the comparison tests so future refactors cannot silently replace pooled sums with run-score averaging.
- Treat the copied-signal audit as defense-in-depth; it should not block the owned-storage fix, but the existing concrete false-positive methodology is well suited to closing it.

### Assessment

**Ready to merge? With fixes**

**Reasoning:** The statistical comparison, provenance, notebook display, deterministic publication, and safety integration are otherwise merge-quality and supported by strong fresh evidence. The explicit owned-array invariant is currently false for every new result array and must be corrected before merge; the two Minor items are test-hardening follow-ups.

---

## Important fix-round re-review

**Range:** `d8a0480d8d07c460931cbea2394d40ca686f66c1..b46147caa5b00530c0694c97c972cc41e4ed28e1`

### Verdict

The owned-storage Important is **addressed**. No new Critical or Important issue was found.

### Evidence

- TDD order is literal and scoped. Test-only commit `8ec4cda` has parent `d8a0480`, changes only `tests/test_fit.py`, and adds the missing `flags.owndata is True` assertion while retaining the existing float64, one-dimensional, no-alias, and `setflags(write=True)` rejection checks. The supplied RED failed diagnostically with `OWNDATA : False`. Production commit `b46147c` has parent `8ec4cda` and changes only `src/boldtailor/_arrays.py`.
- `_ImmutableFloatArray` is allocated directly through `np.ndarray.__new__` with the source shape, explicit `np.float64`, and `order="C"`; it therefore owns its storage and is C-contiguous. Values are copied into that allocation before the base ndarray API freezes it (`src/boldtailor/_arrays.py:13-23`), so the returned array does not alias its input.
- The subclass's public `setflags(write=True)` path raises before delegating, while false/unspecified flag operations retain normal ndarray behavior (`src/boldtailor/_arrays.py:6-10`). This preserves the pre-existing strict public immutability assertion as well as the new literal ownership assertion.
- The helper remains shape-generic and is still used consistently for immutable signals, frame times, complete-result arrays, contrasts, and all four delta-result arrays. The allocation/copy/freeze sequence preserves scalar, empty, one-dimensional, and multidimensional C-order float64 behavior. Supplied verification records the focused RED, focused GREEN, 30 passing `test_fit.py` tests, a 246-test full warning-strict pass after the production commit, and clean Black checks; this re-review did not rerun tests.

### Remaining issues

- The direct multi-run pooled comparison/zero-SST regression gap remains **Minor**.
- The copied raw-signal display audit bypass remains **Minor**; the current notebook remains clean.

### Updated assessment

**Ready to merge? Yes**

**Reasoning:** The only merge-blocking invariant now has a genuine committed RED followed by a narrow production fix that satisfies literal ownership, dtype/order, no-alias, and public write-protection requirements without changing callers. The two previously documented Minor test-hardening items remain deferred and do not block merge.
