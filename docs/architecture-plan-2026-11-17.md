# Architecture review and remediation plan

Reviewed 2026-11-17 against `73689bf` on `main` (76 modules, 19.5k lines in
`src/boldtailor`). No production code was changed and no tests were run.

**Evidence status.** Items marked *traced* were confirmed by reading the code
paths end to end during the review. All other items come from full-file reads
of each partition with the line references given; re-confirm each reference
before acting on it, since line numbers drift.

**Overall.** The numerics are in good shape. The complexity is concentrated in
orchestration layers (`workflow/beta_series.py`, `workflow/run.py` → outputs,
the entry half of `denoising.py`, the fit/selection half of `reml.py`,
`_ridge_cv.py`) and in a few shared objects that do too much (`RunDesign`,
provenance records, `WorkflowSettings`).

## Decisions already made

- **Inner-fold selection model (phase 1, item 1) is a bug**, not intended
  behaviour. Fix it.
- **Top-level `__init__.py` keeps its public exports.** The general
  "no code or imports in `__init__.py`" rule does not apply to this package.
  The docs that say initializers are empty are stale and are corrected in
  phase 3.

## Working rules for every phase

- RED-GREEN-Refactor: write the failing test, commit it, then implement.
- Never weaken an existing test to make a refactor pass.
- Each phase is independently mergeable. Phases 1–2 are small; 4–6 are the
  larger ones.
- During edits run the affected module tests and their direct consumers;
  before committing add `uv run pytest -m integration -q -W error`; before
  merging a phase run `uv run pytest -q -W error --durations=20`.
- Pure refactor phases (3–7) must leave numerical outputs bit-identical unless
  the step says otherwise.

---

## Phase 1 — Behaviour fixes

### 1.1 Inner ridge-tuning folds ignore the HRF selection model (*traced*)

- `workflow/beta_series.py:443-445` adds `task_model` to `selection_options`
  only when `hrf_method == "reml"`.
- `_score_block` (`beta_series.py:136-149`) splats those options into
  `score_ridge_candidates` / `score_fraction_candidates`, so under
  `hrf_method="cv"` the scorer uses its default `task_model=TaskModel()`
  (`_ridge_cv.py:258`, passed to `select_hrfs` at `:98-106`).
- `_outer_block` (`:246-255`) and `_final_selection` (`:376-385`) pass
  `ctx.selection_model` explicitly.
- Effect: when the task model has modulators (e.g. RT), inner-fold HRF
  selection uses a different model from outer evaluation and the final fit.

Steps:
1. Test: with a task model that has a modulator and `hrf_method="cv"`, assert
   the inner-fold `select_hrfs` receives the selection model (and that the
   tuning provenance records it). Must fail on `main`.
2. Pass `task_model=ctx.selection_model` explicitly in `_score_block`; remove
   the REML-only branch at `:443-445`.
3. Delete `_block_selection_options(include_model=)` and the
   `method` → `hrf_method` rename at `:138`; the options object then carries
   only method and REML options.

### 1.2 CV beta path builds its own blocks

- `inputs.make_blocks` (`workflow/inputs.py:231-254`) drops constant
  grayordinates. `beta_series._settings` (`:95-99`) rebuilds contiguous
  unfiltered blocks, and `_cv_model` (`:741-768`) never receives the session's
  blocks (`fit_beta_models` passes them only to `_fixed_models`).
- The numeric effect on constant features was not traced. Step 1 establishes it.

Steps:
1. Test: a synthetic session with constant grayordinates; assert the CV path
   uses the same feature set as the fixed path and restores excluded
   positions as NaN.
2. `fit_cv_beta_series` takes `blocks`; `_cv_model` passes the session blocks.
3. Remove block construction and the duplicated `block_size` /
   `max_grayordinates` / `n_jobs` / percentile validation from `_settings`.
4. Replace the four restatements of the minimum-runs rule
   (`inputs.py:169`, `run.py:200`, `run.py:211`, `beta_series.py:92`) with one
   `min_runs_per_half(method, optimized)` helper.

### 1.3 Non-converged mixture threshold is used silently

- `_mixture_threshold.py:131-132` suppresses `ConvergenceWarning` and records
  `converged`; `denoising.py:280-287` never checks it. `except Exception` at
  `_mixture_threshold.py:134` is broad.

Steps:
1. Test: a forced non-converging fit surfaces a warning in the denoising
   result's provenance and to the caller.
2. Warn when `converged` is false; narrow the `except`.

---

## Phase 2 — Seal caches and shared mutable state

### 2.1 `RunDesign` (`_hrf_cv.py:55-161`) (*traced*)

- `_cached_design` (`lru_cache(maxsize=32)`) returns the same instance to
  every caller. It exposes mutable dicts (`eligibility`,
  `trial_eligibility`, `_blocks`) and writeable arrays (`block().x`, `.a`,
  `.raw`, `q`).
- `@lru_cache(maxsize=256)` on the method `trial_matrix` (`:111`) is one
  class-level cache keyed on `self`; it keeps instances alive after the outer
  cache evicts them and can thrash when runs × selected HRFs exceeds 256.
- `SingleTrialResult.design` and denoising's `_Terms` (`denoising.py:561`)
  hold references into the cached object.

Steps:
1. Tests: cached block arrays are not writeable; mutating one result's design
   cannot change a second analysis in the same process; evicted designs are
   garbage-collected.
2. Set `write=False` on arrays where `_Block` is built; make the eligibility
   dicts private behind `eligible()` / `trial_eligible()`.
3. Replace the method-level `lru_cache` with a per-instance dict.
4. Later (phase 7): split the trial-matrix provider role from the task-model
   statistics role.

### 2.2 Other cache and aliasing sites

- `_boxcar_sampling` (`_hrf_design.py:176`) hands out writeable cached arrays;
  2-argument and 4-argument calls are separate cache keys.
- `default_hrf_library` (`hrf_library.py:805`): `()`, `(512)` and
  `(n_samples=512)` are distinct keys for a build of several seconds.
- `HrfLibrary.origin` is only shallowly frozen; a nested write mutates the
  cached default, and that dict feeds provenance.

Steps: read-only flags on cached arrays; normalize arguments in an uncached
wrapper; deep-freeze `origin` (test: nested assignment raises).

### 2.3 Denoising identity hashing

- `run_identities(data)` is computed at `denoising.py:1348`, `1480`, `1515`
  and again in `with_denoising`; `array_digest` (`:99-101`) uses
  `array.tobytes()`, copying each run.

Steps: compute once in `select_denoising` and pass down; hash via
`digest.update(memoryview(...))`. Digest values must not change (pin with a
test before the edit).

### 2.4 Process-wide stdout redirection

- `design.py:153`, `_task_design.py:129`, `workflow/analysis.py:129` use
  `redirect_stdout`, which swallows all process stdout during the call.

Step: confine to the narrowest Nilearn call and document the caveat; no
behaviour change expected.

---

## Phase 3 — Untangle imports and module ownership

Static facts: 57 cross-module imports of `_private` names; two import cycles;
12 in-function imports.

1. **Cycle `_single_trial_design` ↔ `_hrf_design`.** Move `_validate_events`,
   `_nuisance_matrix` and a single `validate_onsets` (currently copied at
   `_hrf_cv.py:164-167`, `_hrf_design.py:151`, `:206`) into a leaf module with
   public names. Remove the lazy imports at `_hrf_design.py:169`, `:202`.
2. **Cycle `results` ↔ `_conventional` ↔ `_fit_diagnostics` ↔ `model`.** Move
   `nested_ols_delta` (`_fit_diagnostics.py:44-56`) next to `fit_r2_designs`
   in `_conventional.py`; the lazy import at `:47` goes away.
3. **Input normalization.** Give the six `data._*` helpers imported by
   `prepared.py:15-22` a shared home with public names; make `owned_table`
   public in `_arrays.py` (used by four result modules).
4. **Logging helpers.** `_make_event`, `_emit_record`, `_error_code` are
   imported privately by `data`, `prepared`, `_fit_lifecycle`, `publication`:
   expose them or wrap them (see 7.4).
5. **Workflow output modules.** Make `_tuned`, `_scope_selection`,
   `_predictor_table` public in `tables`; make the four metadata builders
   that `export_cifti` imports public in `manifest`; delete the
   `export_cifti._all_selection_maps` pass-through; move
   `analysis.selection_maps` to `hrf_maps`; have `run.py` call owners directly
   instead of through the 20-name `outputs.__all__` facade.
6. **Hoist needless lazy imports** (`ridge_selection.py:102`,
   `fractional_ridge.py:100`, `workflow/analysis.py:255`): none dodges a cycle.
   Keep the lazy imports in `cli.py` (they keep `--help` light).
7. **Naming/placement.** `prepare_runs`, `subset_runs`, `RunDesign` are used by
   REML, ridge and single-trial but live in `_hrf_cv.py`;
   `TrialEncodingResult` lives in `ridge_results.py`; `_hrf_design.py:23`
   re-exports constants defined in `_constants`.
8. **Docs.** Correct the statements that package initializers are empty
   (`docs/code-review.md:17`, `docs/development.md:62`, `docs/README.md:4`)
   to describe the top-level public exports.

Add one test that asserts the package import graph has no cycles so the
result is protected.

---

## Phase 4 — Named records instead of dicts and tuples

### 4.1 Workflow records (`workflow/_records.py`)

- `BetaFit`, `GlmBlock`, `OuterBlock` (17 fields), `OuterEvaluation` (23) are
  `TypedDict`s mutated after construction (`analysis.py:273`, `289`, `291`;
  `beta_series.py:327`, `359-363`, `411`, `570`). Frozen `BetaModel.fit` is
  one of these mutable dicts.
- Block-invariant fields ride in per-block payloads and the last block wins
  (`beta_series.py:202`, `359-363`; `analysis.py:289`).
- "Fractional" is detected four ways: `ctx.fractional`, key presence
  (`beta_series.py:710`, `export_cifti.py:392`),
  `isinstance(FractionSelection)` (`tables.py:73`), `settings.ridge_mode`.

Steps: frozen dataclasses for assembled results with an explicit
`regularization` field; workers return per-block payloads and the parent
assembles invariants once; replace the four detections with the field.

### 4.2 Denoising entry path (`denoising.py:1189-1579`)

- `gate` and `grid` are packed as tuples (`:1559-1560`), unpacked in
  `_settings`, stored as a 4-tuple (`:1206`), unpacked again (`:1292`,
  `:1132`); `gate` is later rebound to a `SignificanceGate` (`:1567`).
- `pool` is a 4-tuple (`:1275`) unpacked at four sites.
- `_arrays` (`:1093`) is a string-keyed dict mutated across functions and
  renamed into `SignificanceGate` (`f` → `f_statistic`, `rho` →
  `ar_coefficients`).

Steps: named gate fields on `_Settings`; a small `_PoolStage` dataclass; a
private `_GateArrays` with `.freeze()` to the public type.

### 4.3 REML fit/selection (`reml.py:594-970`)

- `RatioOptimum` → dict (`:721-728`) → `fit_buffers` (`:731-748`) → `RemlFit`
  (`:750-762`); `selection_state` (`:842-849`) is mutated at `:852-873`.

Steps: private `_FitBuffers` / `_SelectionBuffers` with `.freeze()`.

### 4.4 Equality on array-holding dataclasses

- `eq=False` is applied to seven denoising classes but not to `PoolMasks`,
  `RunComponents` or any `reml.py` / `ar1.py` dataclass, where `==` raises or
  misleads. Apply it uniformly.

---

## Phase 5 — Workflow signatures and settings

1. **Pass state objects through.** `_Session` / `_Analysis` (`run.py:52-84`)
   are unpacked at `run.py:426-470` into `workflow_artifacts` (13 args),
   `manifest.metadata` (11, different positional order) and `render_report`
   (22, which only copies kwargs into `ReportInputs`). Promote the two
   dataclasses to `_records.py` and pass them; `render_report(settings, data,
   figures)`.
2. **Remove `save_workflow`** (`outputs.py:193`): no `src` caller, duplicates
   `_publish`. Confirm test/example usage first.
3. **Type `_stage_artifacts(**options)`** (`outputs.py:98`).
4. **Provenance links.** `run.py:395-410` regex-parses artifact paths to
   recover links; carry the link on the artifact instead.
5. **`WorkflowSettings`** restates its 28 fields in `DEFAULTS`,
   `_FIELD_NAMES`, the `__init__` signature and properties
   (`settings.py:170-433`), plus CLI defaults and `report._VALUE_FLAGS`.
   Keep the hand-written class; derive `DEFAULTS` / `_FIELD_NAMES` from the
   signature, add a drift test, make validators pure (`_validate_reml` calls
   `build_library()` on a half-built object at `:499`), and return an
   immutable mapping from `surface_meshes`.
6. **Preflight.** `describe_inputs` (`run.py:110-144`) re-implements `_load`'s
   checks and every session is loaded twice; share the checks.
   `cli.py:285` labels any `FileNotFoundError` an input error.
7. **Figures.** Summary tables are by-products of plotting (`plots.py:63`,
   `99`, `133`); compute them in `summaries`. Use `Figure` objects everywhere
   so a failing plot cannot leak past `_close`.

---

## Phase 6 — One regularizer strategy

- `fractional` drives about 15 branches in `beta_series.py` and is threaded
  through four functions in `_ridge_cv.py` (`:34-54`, `:84-95`, `:111-175`,
  `:187-243`, `:246-319`). Variables named `alphas` hold fractions
  (`beta_series.py:143`, `504`; `_ridge_cv.py:232`).
- `_score_fold` has three identical try/except relabelers.
- Encoding arrays, masks, design and rank are recomputed per fold × alpha
  although none depend on alpha; `run_tables(data)` is called per fold.

Steps:
1. A two-instance regularizer strategy (scorer, selector, grid keyword, fit
   option, provenance descriptors) used by both `_ridge_cv` and `beta_series`.
2. A public `EncodingDesign` in `trial_encoding` prepared once per fold with
   `.score(betas)`; this also removes the private imports of
   `_predictor_arrays` / `_training_design`.
3. Move the ~200 lines of provenance builders (`beta_series.py:489-692`) into
   their own module and remove the method prose duplicated in
   `manifest.py:66/80`.
4. `PreparedFractionBetas` (`_fractional_ridge.py:65-77`) retains three
   signal-sized arrays per run; keep only what `fit_fraction_run` needs.
5. Check whether `score_candidates` bypassing `fit_operation`
   (`_ridge_cv.py:234-243`) is intentional; align if not.
6. Check `manifest.py:52`, which states "within-run validation SST"
   regardless of `encoding_mode`.

---

## Phase 7 — Consolidate duplicates and repeated work

1. **Single-trial assembly.** `_shared_hrf_fit.py:99-122` and
   `_selected_hrf_fit.py:187-205` duplicate result assembly; implied alphas
   travel in `TrialRunFit.diagnostics["ridge_alphas"]`; a `sha256` object is
   mutated across functions (`_selected_hrf_fit.py:83`, `:104-108`). One
   `assemble_result`, a `ridge_alphas` field, per-group digests combined at
   the end.
2. **HRF-assignment fingerprint** is written four times
   (`hrf_selection.py:107`, `hrf_results.py:149`, `_hrf_reml.py:74`,
   `_ridge_cv.py:215`). One function.
3. **HRF evaluation.** `evaluate_hrf_split` computes training statistics twice
   (`hrf_selection.py:83`, `:362`); `_predict` (`:193-203`) duplicates
   `loro_scores`; the provenance epilogue is pasted in five places. One
   `finish_fit(operation, data, activity)` helper.
4. **Normalization lifecycle.** `data.from_arrays` (`data.py:85-148`) and
   `prepared._prepare_analysis` (`prepared.py:146-221`) hand-roll the same
   bind/start/try/emit skeleton. Add a context manager beside `fit_operation`.
   Also reconcile the two completion-event protocols
   (`_fit_lifecycle.py:48-52` fabricates a second completion event).
5. **Provenance as state.** `prepared_fit.py:149-158` rebuilds the model from
   `activities[-1]`; the same convention is at `provenance.py:385-389`,
   `bids_provenance.py:172`, `205`, `hrf_results.py:148`, `_hrf_glm.py:61`.
   Add a named accessor; make `analysis_fingerprint` a real field instead of
   `_extra`; pick one name each for `metadata_fingerprint` / `data_id` /
   `data_fingerprint` and `analysis_fingerprint` / `analysis_id`.
6. **Table accessors.** `AnalysisData.events` / `.confounds` deep-copy on every
   access (`data.py:54-60`). Package code reads the private tables under the
   no-mutate contract; public accessors keep returning copies.
7. **Design rank.** Computed repeatedly (`_conventional.py:127`, `:260`,
   `:266`, `:283`, `_standard_glm.py:189`, `prepared_fit.py:327`) and reported
   as a warning, an activity string and sometimes an exception. Compute once
   per design and pass down; `standard_task_delta_r2` reuses the parent's
   designs instead of recompiling (`_standard_glm.py:73`).
8. **Delta-R² paths.** `_standard_glm.py:81-95` and `prepared_fit.py:105-119`
   are near-parallel, and parent validation diverges (skip vs raise when an id
   is None). Share the diagnostics builder and choose one policy.
9. **Denoising projections.** `y - q @ (q.T @ y)` is inline at seven sites
   although `_numerics.residualise` exists; the numerical-zero rule is
   repeated three times. One helper each.
10. **JSON and BIDS helpers.** Five canonical-JSON dumps with inconsistent
    `ensure_ascii`; `dataset_description.json` built in both
    `bids_provenance.py:91-96` and `workflow/artifacts.py:40-50`; two
    `_validate_uri` functions with opposite rules. One of each.
11. **`choose_eligible`** (`_hrf_cv.py:287-335`): vectorize and split into
    argmax / exclude-and-retry / table.
12. **`DenoisingResult`** stores `n_components` and `pcstop_count` twice and
    a flattened `PoolMasks`; `components` vs `run_components` is confusing.
    API-visible, so do last and with deprecation aliases.
13. **Schedule removal** of the one-release deprecation shims and of test-only
    code in `src` (`stimulus_regressor`, `convolve_events`,
    `prepare_run_beta_path`, `RunDesign.trial_eligible`, `criterion_slope`,
    `_log_grid`).

---

## Leave alone

- The single transactional publish boundary (`publication.publish_artifact_set`);
  analysis code writes nothing.
- The read-only owned-array contract and `_arrays.py`'s `object.__setattr__`
  idiom.
- The hand-written immutable classes introduced in `f919e96`.
- `reml.py`'s search section (`:356-591`) and `_numerics.py`.
- Seeded, local RNG use; the two intentional AR(1) conventions.
- `denoising.py` and `reml.py` as single files: both have real, headed
  sections. The problems are in their orchestration halves, not the split.
- `prepared_fit.py` and `bids_provenance.py`: public, documented entry points,
  not dead code.
- `hrf_maps.py`, `files.py`, `artifacts.py`, `diagnostics.py`,
  `reliability.py`, `task_detection.py`, `cifti.py`.

## Already fixed before this review

Findings from `docs/archive/review-2026-11-16-opaque-contracts.md` that are
resolved on `main`: deprecated-keyword sentinel, shared workflow `_State`,
in-place provenance linking, `locals()` report inputs.
