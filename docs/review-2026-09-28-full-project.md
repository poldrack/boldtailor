# Full project review: scientific accuracy, engineering, architecture

> **Historical review:** Describes the reviewed working tree, including work
> preserved separately from this branch. Findings may already be addressed.
> See the [documentation index](README.md) and active refactor roadmap.

**Date:** 2026-09-28
**Scope:** all of `src/boldtailor/` (7,735 lines), `tests/` (11,249 lines),
`examples/` (≈8,800 lines plus notebooks), packaging, and documentation.
**Method:** full read of every source module, the prior review/audit documents
(`docs/architecture-review-2026-08-12.md`, `docs/red-team-audit-2026-09-27.md`,
`docs/red-team-audit-response-2026-09-27.md`,
`docs/scientific-follow-ups-2026-09-28.md`), and spot verification. Both test
suites were run: `uv run pytest` (721 passed) and
`uv run pytest examples/NSD` (140 passed). No files were changed by this review.

This review builds on two earlier rounds (the 2026-08-12 architecture review
and the 2026-09-27 red-team audit). Where their findings persist in the
current tree, that is stated explicitly; new findings are marked **NEW**.

---

## 1. Verdict

The **scientific core is sound and unusually well validated for a research
package**. The numerical kernels (`_conventional`, `_single_trial_fit`,
`_fractional_ridge`, `_hrf_cv`, `_hrf_design`) are compact, correct against
independent oracles, and wrapped in careful cross-validation leakage
discipline. The documentation is exceptionally honest about limitations and
distinguishes verified behavior from open questions.

The **engineering and architecture remain out of proportion to the mission**.
This was the central conclusion of the 2026-08-12 review, and it is still
true: most of its specific findings are present in the current tree, and the
codebase has since grown by ~3,100 source lines and ~7,300 test lines in the
same style. Roughly a quarter of the source is provenance/publication/logging
plumbing; the fit entry points are dominated by identity bookkeeping rather
than statistics; and a 754-line transactional filesystem writer defends
against a threat model no document names. The dominant risk is not incorrect
science — it is that the surrounding machinery is expensive to maintain and
deters the scientific follow-ups the project itself has identified as most
important.

---

## 2. Scientific accuracy

### 2.1 What is solid

- **Conventional GLM**: nilearn's OLS/AR(1) with t contrasts; rank-deficient
  designs are refit in a full-rank estimable basis with `n - rank` degrees of
  freedom (red-team finding #10 is remediated); exactly constant features are
  masked to NaN before inference (finding #1 remediated); estimability of
  every contrast is verified per run before fitting.
- **HRF selection**: leave-one-run-out prediction of a shared mean-stimulus
  amplitude, pooled before argmax; nuisance coefficients profiled per run; no
  held-out BOLD leakage; lazy winner eligibility now concerns the mean-stimulus
  model only (finding #2 remediated). Sufficient-statistics batching is exact.
- **Single-trial and ridge**: FWL-correct unpenalized nuisance; normalized
  ridge and raw-basis fractional ridge both match augmented-lstsq/root-finding
  oracles to ~1e-12; the fraction→alpha map is found by bracketed bisection in
  log-space (60 iterations — sound and cheap).
- **Encoding-guided CV**: nested HRF selection uses only inner-training runs;
  fractional CV uses fixed OLS validation targets; within-run centering uses
  run-specific training intercepts and held-out offsets only for scoring.
  Fold bookkeeping (`_ridge_cv._score_fold`) is leakage-safe by inspection.
- **Timing**: automatic design paths now reject irregular frame times instead
  of silently stretching kernels (finding #6 remediated); the prepared-OLS
  escape hatch for censored data is documented with correct reasoning.
- **Honest reporting**: boundary selections, the matched-filter large-alpha
  limit, the nilearn-SPM anchor's resolution dependence, TR-dependent impulse
  beta units, and the metadata-identity (not content-identity) basis of ΔR²
  gating are all documented in the user guide.

### 2.2 Remaining scientific concerns

These are mostly already tracked in `docs/scientific-follow-ups-2026-09-28.md`;
they are restated here with their importance to *this* review's priorities.

1. **The ridge-tuning estimand is unresolved (highest scientific priority).**
   The current default objective predicts within-run variation of
   candidate-regularized (shared-alpha) or OLS-target (fractional) beta series.
   Neither measures recovery of true trial amplitudes; the ablation report
   shows strongest-endpoint selection in ~68%/23% of simulated cases depending
   on centering. This is a research decision, not a bug, but every other
   validation claim about the single-trial pipeline is downstream of it.
2. **Equal-weight fixed effects with summed degrees of freedom.**
   Anticonservative under heteroscedastic runs (the audit measured t = 2.91
   vs 3.66 for inverse-variance weighting in one scenario). Documented, but
   for a package whose headline output is contrast statistics, calibration
   evidence (Welch–Satterthwaite or a named inverse-variance option) matters
   more than any remaining code hygiene item.
3. **ΔR² identity checks are metadata-only.** Replacing BOLD content while
   keeping source descriptors passes every `task_delta_r2*` parent check.
   Documented, and tracked as a follow-up (content fingerprints), but worth
   restating because the error messages ("full result does not match data and
   model") imply a stronger guarantee than is delivered.
4. **HRF selection assumes cross-run transfer of the mean stimulus response.**
   A changing mean response across runs silently degrades the selection
   objective. Documented; no in-package diagnostic (e.g., per-fold amplitude
   variability) is offered. A cheap, useful addition.
5. **AR(1) limitations inherited from nilearn** (OLS-residual Yule–Walker,
   0.01-bin truncation, unscaled first sample) are accurately documented; the
   AR(1) full-model R² (GLS betas on unwhitened data) differs from the
   task-ΔR² full R² by design. Fine, but users will trip on this; consider
   surfacing it in `AnalysisResult` docstrings, not only the user guide.
6. **Canonical-vs-custom peak-time mismatch** in exported HRF parameter maps
   (nilearn anchor at 5.1 s vs 5.0 s for the parameter-matched double-gamma)
   persists by design (compatibility anchor). The documentation is good; the
   residual risk is that parameter-map consumers compare across the
   canonical/custom boundary without reading it.

No new numerical errors were found in this pass.

---

## 3. Software engineering and architecture

### 3.1 Persistence check on the 2026-08-12 architecture review

That review's findings were re-verified against the current tree:

| Finding | Status in current tree |
| --- | --- |
| A1 imaging code lives in `examples/` | **Persists, and grew**: `examples/NSD/` is now ~5,800 lines of library-grade workflow code plus ~2,900 lines of tests; `stop_signal_demo.py` is 694 lines |
| A2 provenance/publication ≈ 43% of source | Improved to ≈26% only because the scientific core grew; the plumbing itself was not reduced |
| A3 fd-anchored publication without a threat model | **Persists verbatim** (`publication.py`, 754 lines) |
| A4 `fit.py`/`prepared_fit.py` parallel lifecycles | **Persists**: identical `_rank_warnings`, `_validate_nested_ols_delta`, constants `_DIAGNOSTIC_NOISE_MODEL`/`_NESTED_OLS_TOLERANCE`/`_TASK_DELTA_R2_DEFINITION`, near-identical dimension validators and contrast serializers |
| A5 private imports across modules | **Persists**: `prepared.py` imports five `_`-privates from `data.py`; `prepared_fit.py` imports `_prepare_contrasts`/`_validate_noise_model` from `model.py` and privates from `provenance.py` |
| B1 quadratic deep copies | **Persists**: `prepared_fit._run_diagnostic` indexes `prepared.design_matrices[run]` (a full deep copy of *all* runs) inside a per-run loop; same pattern at lines 216, 306, 405, 428 |
| B2 `_ImmutableFloatArray` bypassable via `np.ndarray.setflags` | **Persists** |
| B3 two error-privacy policies (`fit` logs raw, `fit_prepared` sanitizes) | **Persists** |
| B4 substring env-redaction per error | **Persists** |
| B5 three `ProvenanceRecord` constructions per `from_arrays` | **Persists** (`data.py` discards one; `prepared.py` the same pattern) |
| B6 no-op `{**activity, "drift_order": ...}` | **Persists** (`fit.py`, `_model_provenance`) |
| B7 ΔR² gated on provenance fingerprints | **Persists** (now documented) |
| C1 notebook driven by string-injection tests | **Persists and grew**: `test_stop_signal_demo.py` is now 1,729 lines, still monkeypatching `display`/`plot_stat_map` inside a live kernel and pinning cosmetic details |
| C2 machine-specific "vacuous" assertions | **Persists**: `tests/test_provenance.py:318` asserts the absence of `/Users/poldrack/Dropbox/code/boldtailor` |
| C3 `test_repository_contracts.py` pins packaging strings | **Persists** |
| D personal default path in notebook | **Persists**: `stop_signal_demo.ipynb` defaults to `/Users/poldrack/data_unsynced/network/bids`; executed notebooks also embed machine-specific `.venv` tracebacks in committed outputs |
| D hardcoded `BOLDTAILOR_VERSION` | **Persists** (`bids_provenance.py:13` vs `pyproject.toml`) |
| D import-time `package_version()` at module scope | **Persists** (`prepared.py:35`, `prepared_fit.py:45`) — `import boldtailor.prepared` fails in an uninstalled checkout |
| D stale branches, `.gitignore` gaps | **Persists**: 7 `safety/*` + 5 merged-looking `feature/*` branches; `.gitignore` lacks `__pycache__/`, `*.egg-info/`, `.coverage` |

The remediation commits after that review addressed the red-team *scientific*
findings thoroughly. The *architectural* backlog was essentially untouched.

### 3.2 New findings in this review

**NEW-1. `_fractional_ridge._prepare` is a denormalized clone with vestigial
machinery.** It calls `_project_design(x, n)` (which internally SVDs the
normalized projected design), then SVDs the *unnormalized* projected design a
second time, sets `scale = np.ones(...)` so the later `/ scale` divisions in
`_solve` are no-ops, and overwrites the `diagnostics` dict it just unpacked.
The raw-vs-normalized basis difference is scientifically important and recently
changed; the code structure obscures exactly the quantity that matters.
Proposal: one `_prepare` that returns the raw-basis SVD explicitly, with the
normalized ridge path sharing it.

**NEW-2. `freeze_fraction_result` mutates frozen dataclasses from another
module.** `SingleTrialResult.__post_init__` and `HrfSingleTrialResult.__post_init__`
both call `object.__setattr__` through a helper in `_fractional_ridge`,
conditionally on `ridge_fraction is not None`. Cross-module mutation of frozen
instances is fragile and surprising; fold the freezing into each class.

**NEW-3. Near-duplicate result containers.** `SingleTrialResult` /
`HrfSingleTrialResult` and `RidgeCandidateScores` / `FractionCandidateScores`
differ by one or two fields; `_SelectionBoundaries` distinguishes them with
`hasattr(self, "fractions")` duck-typing. A shared base with a generic
grid-parameter name would remove ~100 lines and a category of drift bugs.

**NEW-4. The `_ridge_cv._run_beta_path` generator protocol is the least
readable code in the package.** Correctness depends on every consumer pulling
`next()` in exactly the candidate order, across runs, per fold; `_score_fold`
additionally runs a *second* beta path with `[1.0]` to obtain the fixed OLS
target, recomputing the design decomposition. The memory motivation is real
but undocumented at the call site; an explicit small class ("BetaPath") with a
`betas_at(alpha)` method and a shared decomposition would be clearer and allow
the OLS target to come from the same decomposition.

**NEW-5. Dead calls.** `prepared_fit._log_validation_failure` and
`_log_comparison_validation_failure` call `append_event_history(...)` and
discard the result — only the logging side effect survives, which is
misleading in functions named "log". Same family as B5/B6.

**NEW-6. Packaging issues.**
- `ipykernel>=7.3.0` is a **runtime** dependency of the library; it belongs in
  the dev group only (it is already duplicated there).
- No CI configuration exists at all (no `.github/`, no CI YAML). For a project
  with this much process discipline (TDD mandates, fixture pins, black checks
  documented in `docs/development.md`), the absence of any automated check on
  a clean machine is the single largest process gap — it is also why
  machine-specific assertions (C2) and uninstalled-import fragility (D)
  survive.
- `pythonpath = ["."]` in pytest config lets tests import each other
  (`from test_fractional_ridge import oracle`) — convenient, but it silently
  couples test modules; a `tests/oracles.py` imported normally would be
  cleaner.

**NEW-7. Docs reference private state.** `docs/glmsingle-comparison.md` pins a
personal absolute path (`/Users/poldrack/Dropbox/code/GLMsingle`) and a local
checkout hash as part of the scientific comparison. Keep the hash (it is good
provenance) but move the path to an environment note.

### 3.3 Where the complexity is, and what it buys

Measured on the current tree:

| Layer | Lines | Assessment |
| --- | --- | --- |
| Numerical kernels (`_conventional`, `_single_trial_fit`, `_fractional_ridge`, `_hrf_cv`, `_hrf_design`, `_design_basis`, `_timing`, `_arrays`, `_contrast_expression`) | ≈900 | Proportionate. The best part of the codebase. |
| Public API + validation (`data`, `model`, `design`, `fit`, `single_trial`, `hrf_selection`, `prepared`, `prepared_fit`, result classes, selectors) | ≈3,500 | Roughly half of this is defensive validation and provenance plumbing. Validation is thorough to the point of validating things the caller cannot plausibly get wrong (e.g., boolean-exclusion checks on every numeric parameter), while a few genuinely reachable gaps remain (B3). |
| Provenance/logging/BIDS/publication (`provenance`, `logging`, `bids_provenance`, `publication`) | 1,983 | The over-complexity hotspot. `provenance.py`'s freeze/thaw/path-safety machinery and `publication.py`'s fd-anchored transactional writer each defend against adversaries no document names; `bids_provenance.py` implements a pinned *draft* BEP subset whose scientific value is real but which adds a third record format to maintain. |
| Examples (`examples/`) | ≈8,800 + notebooks | Library-grade imaging I/O and workflows living outside the package, with their own 2,900-line test suite not collected by default `pytest` (documented in the developer guide, but easy to miss). |

The pattern the 2026-08-12 review named — review cycles that always answered
"more hardening, more pinning" without asking "proportionate to what?" — is
still visible in the newest code: `_ridge_cv` carries six different
fingerprints per CV run; `Artifact` validates TSV column counts and rejects
`NaN` JSON constants; provenance rejects string values that merely *look like*
paths. Each is defensible alone; together they are why a first-level fMRI
GLM needs 7,700 lines.

---

## 4. Proposed changes, by importance

### Tier 1 — scientific priorities (do these before further hardening)

1. **Resolve the ridge-tuning estimand** (follow-up #1 in
   `docs/scientific-follow-ups-2026-09-28.md`). Write the estimand spec, run
   the promised common-benchmark comparison, and only then consider default
   changes. This is the highest-value remaining work in the project.
2. **Fixed-effects calibration report** (follow-up #3). Measure coverage under
   heteroscedastic runs; decide whether Welch–Satterthwaite dof or a named
   inverse-variance option is warranted.
3. **Content identity for ΔR² comparisons** (bounded follow-up). Hash signal
   content (or an explicit caller-supplied data fingerprint) rather than only
   source descriptors, or rename the guarantee so error messages stop implying
   content identity.
4. **Add a cross-run stability diagnostic to HRF selection** — e.g., expose
   per-fold amplitudes or their spread — so the documented transfer assumption
   is checkable in practice.

### Tier 2 — architectural simplification (the "overly complicated" backlog)

5. **Decide the publication threat model, then shrink `publication.py`.**
   If no adversary is named, stage-to-tempdir + `os.replace` + fsync (~80
   lines) delivers the same user-visible guarantee (atomic sets, rollback).
   Delete fd-anchoring, `F_GETPATH`//proc probing, casefold-collision walks,
   and the recovery-copy subsystem, or write down why they exist.
6. **Extract one shared fit lifecycle** (bind context → emit started → run →
   emit completed/failed → extend provenance) used by `fit`, `fit_prepared`,
   `fit_single_trials`, `fit_selected_glm`, and both delta-R² paths. This
   deletes the A4 duplication and makes the next change to logging semantics a
   one-file edit.
7. **Promote the imaging workflows into the package** (`boldtailor.nsd`,
   `boldtailor.nifti`), leaving notebooks as narrative. Move the NSD test
   suite into `tests/` (or add `examples/NSD` to `testpaths`) so `uv run
   pytest` is the whole suite. Then replace the 1,729-line notebook
   instrumentation tests with direct tests of the library module plus one
   notebook smoke test.
8. **Collapse the immutability boilerplate.** Pick one: (a) plain arrays with
   `writeable=False` (accept that `np.ndarray.setflags` can bypass it — it is
   a lint, not a lock), or (b) documented owned copies. Either way delete
   `_ImmutableFloatArray`, `freeze_fraction_result`'s cross-module mutation,
   and the ~120 lines of underscore-field + trivial-property boilerplate.
   Fix B1 (quadratic deep copies) at the same time — internal callers should
   not go through the defensive-copy accessors.
9. **Unify the error/privacy policy** (B3/B4): one sanitization rule applied
   at the logging boundary for all entry points, implemented as an allow-list
   of what may appear in messages rather than env-substring redaction. Delete
   the dead `append_event_history` calls (NEW-5), the discarded
   `ProvenanceRecord` constructions (B5), and the no-op fingerprint line (B6).

### Tier 3 — targeted cleanups

10. **Rewrite `_fractional_ridge._prepare`** around a single SVD in the raw
    basis; remove the vestigial `scale` (NEW-1).
11. **Replace the `_run_beta_path` generator protocol** with an explicit
    object sharing one decomposition per (run, candidate), including the OLS
    target (NEW-4).
12. **Merge the near-duplicate result containers** and replace the
    `hasattr`-based `_SelectionBoundaries` mixin with an explicit shared
    interface (NEW-3).
13. **Add minimal CI** (uv sync, `pytest tests examples/NSD`, `black
    --check`) on a clean runner (NEW-6). This will catch the import-time
    `package_version()` fragility and the machine-specific assertions.
14. **Packaging hygiene**: move `ipykernel` to the dev group; read
    `BOLDTAILOR_VERSION` from `importlib.metadata` (lazily); defer module-scope
    version lookups into the functions that need them; fix the stale "in
    Phase 1" error string in `model.py`; update `.gitignore`
    (`__pycache__/`, `*.egg-info/`, `.coverage`); delete merged/stale
    branches.
15. **Test-suite hygiene**: delete or fix the machine-specific assertion in
    `test_provenance.py:318`; decide whether `test_repository_contracts.py`
    earns its keep; move shared oracles into `tests/oracles.py`; strip
    machine-specific tracebacks from committed notebook outputs (or stop
    committing outputs).
16. **Docs**: remove the personal path from `glmsingle-comparison.md`; make
    the notebook `BIDS_ROOT` default environment-only (fail with instructions
    rather than a personal path); add a scope note to
    `docs/validation/nsd-notebook.md` if not already done (it describes the
    superseded fixed-ridge/649-candidate defaults); consider moving
    `docs/superpowers/` process artifacts out of `main` or adding a one-line
    orientation note at its root.

### Tier 4 — leave as is (explicit non-changes)

- The **nilearn SPM anchor** for candidate 0: compatibility is a deliberate,
  documented contract.
- The **AR(1) full-R² vs task-ΔR² distinction**: two diagnostics measuring
  different things, both documented.
- The **`encoding_mode="absolute"` legacy path**: it is documented, pinned,
  and cheap to keep; removing it would orphan the September validation
  records.
- The **BEP028 draft projection**: keep if a real consumer uses it; otherwise
  this is the next candidate for the "no named requirement" test after
  `publication.py`.

---

## 5. Bottom line

Ship the science agenda in Tier 1: the numerical engine deserves confidence,
and its open questions are research questions, not code defects. Then do Tier
2 exactly once: the codebase's complexity budget is currently spent on
filesystem adversaries and bookkeeping identity checks, while the places where
complexity would pay scientific rent — the ridge estimand, cross-run
inference, and imaging I/O living in the package — remain unfunded. The
single most instructive metric: `publication.py` (754 lines) is larger than
the entire single-trial estimation path (`_single_trial_design` +
`_single_trial_fit` + `_fractional_ridge` + `trial_encoding`, 600 lines
combined) that the package exists to provide.
