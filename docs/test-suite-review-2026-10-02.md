# Test-suite review: which tests earn their keep

**Date:** 2026-10-02
**Scope:** every test function in `tests/` (39 files, 453 functions, 758
collected) and `examples/NSD/` (21 files, 148 functions, 263 collected);
`conftest.py`, `oracles.py`, CI configuration. Full suite: 1,021 collected,
244 s wall time. **No files were changed.**
**Method:** AST classification of all 601 functions (assert counts,
`pytest.raises`/`match=`, oracle calls, provenance/ownership/notebook
markers), JUnit timing of the full run, and a line-by-line read of every test
by four reviewers using one shared rubric. Per-test verdict tables are in the
Appendix.

**Standing decision (user, 2026-10-02):** the default `pytest` run covers the
package only. Notebook-execution and `examples/NSD` tests become opt-in.

---

## 1. Headline

Of 601 test functions, **291 should stay as they are, 254 should be merged
into roughly 80 parametrized tests, and 56 should be deleted.** Nothing in
the CUT or MERGE columns protects a numerical guarantee that a KEEP test does
not also protect. After pruning and moving example tests out of the default
run, the default suite would be about **270 functions in `tests/` running in
roughly 60 s** instead of 1,021 in 244 s, and the irreplaceable oracle and
leakage tests would be easier to find.

| group | files | functions | KEEP | MERGE | CUT | wall time |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| GLM / API (`test_fit`, `test_prepared*`, `test_multirun`, `test_hrf_glm`, `test_data`, `test_design`) | 7 | 145 | 38 | 91 | 16 | ~8 s |
| Numerics core (single-trial, ridge, encoding, HRF, task design, simulations) | 18 | 161 | 110 | 40 | 11 | ~45 s |
| Infrastructure (stop-signal notebook, publication, provenance, logging, BIDS, model, misc) | 14 | 147 | 58 | 74 | 15 | ~22 s |
| `examples/NSD` | 21 | 148 | 85 | 49 | 14 | ~170 s |
| **total** | **60** | **601** | **291** | **254** | **56** | **244 s** |

Time is dominated by notebook execution: 14 kernel launches plus 7 in-process
whole-notebook executions in `examples/NSD`, three executions of the
stop-signal notebook in `tests/`, together roughly 120 s. The five slowest
files are all example notebooks (`test_multisession_workflow` 30 s,
`test_ridge_outputs` 25 s, `test_nsd_workflow` 18 s, `test_ridge_workflow`
15 s, `test_within_run_encoding` 14 s); `test_fractional_cv` (23 s) is the
only package file in the top ten.

## 2. What is genuinely valuable

The suite's strength is concentrated in about 70 tests, listed in the
Appendix as "irreplaceable". They fall into four kinds:

- **Independent numerical oracles.** nilearn `run_glm`/`compute_contrast`
  and `FirstLevelModel` for the conventional GLM (`test_fit:241`,
  `test_multirun:132`, `test_hrf_glm:142`, `test_prepared_fit:120`);
  `make_first_level_design_matrix` and `compute_regressor` for designs and
  the fast convolution path (`test_design:30,86,199`, `test_hrf_design:130`);
  augmented-lstsq and `brentq` for ridge and fractional ridge
  (`tests/oracles.py`, `test_fractional_ridge:35,52`, `test_single_trial:109`);
  dummy-intercept OLS and closed forms for the encoding model
  (`test_trial_encoding:188,241,309`); stacked-OLS and LORO oracles for HRF
  selection (`test_hrf_cv:146`, `test_hrf_selection:92`).
- **Leakage guards with detectable perturbations.** Held-out BOLD replaced by
  noise or scaled by −3 with training coefficients, fold HRF indices, and
  predictions asserted unchanged (`test_within_run_cv_isolation:13,35`,
  `test_hrf_selection:125,153`, `test_fractional_cv:173`, `test_ridge_cv:145`).
- **Behavioral contracts users rely on.** Zero-SST features give NaN R²,
  canonical selection reproduces the plain fit, spatial-signature mismatch is
  rejected, GLM designs equal the selection's scored designs, task-model
  subset rules, min_onset exclusion, pooled R² definition, atomic publication
  with rollback at every replace boundary, concurrent writers serialize.
- **Golden fixtures with a real external contract.** The BEP028 projection
  fixture (`test_bids_provenance:125`) and the empty-`__init__` rule
  (`test_distribution:10`).

Example tests worth keeping follow the same pattern: one-sample t and RT
correlation oracles (`test_beta_activation`, `test_rt_diagnostics`), odd/even
isolation (`test_nsd_hrf_selection:83,188`), CIFTI axis round-trips
(`test_nsd_cifti:69,181`, `test_workflow_surfaces:68`), serial-vs-parallel
invariance, and reuse-validation rejections.

## 3. Where the volume comes from

**3.1 Validation permutations (≈190 functions, 189 "error-only").** About 45 %
of infrastructure tests and a third of GLM/API tests assert a `ValueError`
for one bad input each, with `match=` on exact text (≈73 % of all
`pytest.raises`). `test_model.py` alone collects 40 items for a frozen
dataclass. Each validator deserves one parametrized table of distinct failure
modes, not one function per permutation. Two tests pin nilearn's internal
warning text ("Matrix is singular at working precision", `test_multirun:218,278`)
and will break on upgrade.

**3.2 Provenance literal pinning (≈100 assertions, 21 exact strings).**
Activity names, score formulas (`test_hrf_selection:443` pins
`"64 * eps * (C + sum_k abs(2*beta_k*B_k) + abs(beta' A beta))"`), key sets,
single-key tests (`test_fit:548` pins `drift_order`). Keep one serialization
test per entry point that asserts key presence and privacy, plus the
fingerprint-sensitivity tests that the ΔR² parent checks depend on.

**3.3 Lifecycle/logging templates (≈25 functions).** The
success / late-failure / early-failure triple is pasted six times
(`test_fit:856,900`, `test_prepared_fit:1372,1425`, `test_hrf_glm:465,509`),
`test_normalization_builds_one_record` is byte-identical in `test_data` and
`test_prepared`, `test_logging:345` duplicates `test_fit_lifecycle:80`, and
ten more `test_prepared_fit` tests replay the same context-reset check. One
parametrized `test_lifecycle.py` over (entry point, late-patch target, early
bad call) replaces them.

**3.4 Ownership/immutability (91 functions touch it).** All rest on
`_arrays.readonly_array` (52 call sites in `src`). Keep `test_arrays:10` and
one ownership test per public result class; fold the rest. Several tests pin
implementation performance details instead (call counts on `np.linalg.svd`,
weakref liveness, "forbid internal use of public property" monkeypatches:
`test_single_trial:146,181`, `test_fractional_cv:216`, `test_fractional_ridge:178`,
`test_prepared_fit:1326`).

**3.5 Notebook execution and cosmetics (45 functions).** The stop-signal
notebook runs three times and pins `cmap`, `cut_coords`, titles,
`display_count == 16`, and a memory estimate to 17 digits. In `examples/NSD`,
14 functions `exec()` notebook cells looked up by hardcoded cell id or
substring, and `test_workflow_surfaces` has 3 behavioral tests out of 14, the
rest asserting matplotlib state. One smoke execution per notebook on fixture
data is worth keeping, opt-in.

**3.6 Tests of scripts rather than the package (19 functions).**
`test_fractional_ridge_ablation`, `test_fractional_ablation_simulation`, and
`test_ridge_objective_simulation` exercise `examples/validation/*.py`; eight
pin row counts or CSV columns. Two genuine package oracles hide among them
(`test_ridge_objective_simulation:68,94`) and should move to
`test_trial_encoding`/`test_single_trial`.

## 4. Tests that are wrong, vacuous, or pin a bug

| location | problem |
| --- | --- |
| `tests/test_fit.py:279` `test_fit_accepts_zero_sst_features_without_inference_warnings` | Asserts contrasts only for the varying feature and NaN only for R² of the constant features. The constant features' `stat`/`z_score` are never checked, which is how the spurious-statistic bug (review S1) survived; the test's "without warnings" clause is satisfied by the `RuntimeWarning` filter that hides the symptom. |
| `tests/test_stop_signal_demo.py:52,57,887,656,731` | Pins the mis-weighted contrast `"(stop_success + stop_failure) - go_success"` (weights 1, 1, −1) into config and provenance. |
| `tests/test_stop_signal_demo.py:1492` | Pins a five-session `DEFAULT_SESSIONS` that the notebook cannot execute (`subplots(1, 2)` + strict zip; validator demands two). Every executing test injects a two-session override, so the pinned default is never run. |
| `tests/test_hrf_glm.py:709` `test_task_delta_r2_uses_the_same_task_model_designs` | Asserts only finiteness and `delta_r2 >= 0`, which clipping guarantees; never inspects designs. The task-model selected-GLM path therefore has **no** numerical oracle (its fixture uses pure-noise signals). |
| `tests/test_fit.py:690` | Pins that `make_task_delta_r2_result` silently clips a raw delta of −0.25 while `task_delta_r2` raises below −1e-12: an inconsistent contract, frozen by the test. |
| `tests/test_fractional_ridge.py:211` | Monkeypatches `_fractional_ridge.freeze_fraction_result` with `raising=False`; the symbol no longer exists, so the guard can never fire. |
| `tests/test_prepared_fit.py:764` | Sets an environment variable no source module reads; asserts only an error code. |
| `tests/test_publication.py:482,545` | Patches `os.supports_dir_fd` (never read) and asserts `TypeError` for a nonexistent keyword (Python default behaviour). |
| `tests/test_provenance.py:102,311`, `tests/test_logging.py:137-140` | Assert absence of the literal `/Users/poldrack/Dropbox/code/boldtailor`, `Path.home()`, hostname, cwd; machine-dependent and pass trivially. Use injected sentinels as `test_publication:307` does. |
| `tests/test_fractional_ablation_simulation.py:61,122` | `selected_fraction == 1` with grid `(1.0,)`; paired RMSE against itself equals 0. |
| `tests/test_hrf_cv.py:285` | Asserts `isfinite | isnan`, which only excludes infinities. |
| `tests/test_ridge_selection.py:19` | Data cannot distinguish percentile interpolation methods, so the method is untested. |
| `tests/test_single_trial.py:337` | "AR noise" at sd 0.01 on unit amplitudes (SNR ≈ 100) is effectively noiseless. |
| `tests/test_multirun.py:178` | Expected value derived from `result.run_r2`, half-circular. |
| `tests/test_sobol_hrf_library.py:21` | Pins seven floats of scipy's scrambled Sobol output with scipy unpinned. |
| `tests/test_repository_contracts.py:7` | Pins dependency version strings from `pyproject.toml`. |

## 5. Non-independent oracles

- Every alpha-ridge lstsq oracle (`test_single_trial:109,194`,
  `test_selected_hrf_fit:15`, `test_ridge_cv` `augmented_beta`) hardcodes the
  implementation's penalty scaling `sqrt(alpha)·‖X_perp‖`; the solve is
  independent but the normalization convention is pinned only by the
  hand-computed `test_single_trial:95`. Acceptable, but say so in a comment.
- `test_hrf_design:44` rebuilds nilearn's `_sample_condition` grid formula;
  `:130` (direct `compute_regressor` across six HRFs on irregular times) is
  the real oracle.
- `test_hrf_glm:170` re-derives multi-run fixed effects with the same
  `(1/3)·(c0+c1+c2)` as the implementation; `test_multirun:132` is OLS only,
  so **multi-run AR(1) contrast combination has no independent oracle**.

## 6. Missing tests (ordered by scientific importance)

1. **Constant-feature contrasts are NaN** for OLS and AR(1), single and
   multi-run, in `fit`, `fit_prepared`, and `fit(..., hrf_selection=)`.
2. **Task-model selected-GLM numerical oracle** (RT + trial_type modulators)
   against nilearn on a signal with known amplitudes.
3. **Multi-run AR(1) contrast oracle** via `FirstLevelModel(noise_model="ar1")`.
4. **Fraction recovery under noise**: the truth-optimal fraction < 1 is
   selected; 1.0 is selected when noiseless.
5. **HRF recovery under realistic noise** (AR(1) ρ 0.3–0.5, SNR ≈ 1–3):
   true HRF wins, `delta_cv_r2 > 0`; canonical wins when truth is canonical.
6. **Alpha recovery end-to-end** through `score_ridge_candidates` →
   `select_ridge_penalty`.
7. **Rank-deficient but estimable design**: variance and t agree with nilearn.
8. **Convolution oracle at a second TR** (0.8, 2.0 s) and oversampling ≠ 50.
9. **`fit_selected_hrfs` on new runs** where a `-1` feature is non-constant
   (must stay NaN, not refit).
10. **`fit` end-to-end with `excluded_event_count > 0`** and with irregular
    frame times.
11. **Percentile method** case where linear and nearest differ.

## 7. Structural recommendations

- **Default run = package only.** Set `testpaths = ["tests"]`; mark notebook
  tests `@pytest.mark.notebook` and example tests via a separate command
  (`uv run pytest examples/NSD`). CI runs the package suite on every push and
  the example/notebook suite on a schedule or label.
- **Consolidate fixtures.** `_complete_sources()` ×5, the two-row HRF library
  ×3, four copies of the nilearn R² oracle, scattered `oracle`/`reference`
  helpers in six files, and `replace_data` duplicating `oracles.subset_runs`
  all move to `conftest.py`/`oracles.py`. Remove `pythonpath=["."]` and the
  renewed cross-test imports (`test_hrf_cv` ↔ `test_hrf_selection`).
- **One parametrized table per validator** for `ModelSpec`,
  `Modulator`/`TaskModel`, `SourceRef`, publish preflight,
  `project_bids_provenance`, run labels, contrast preflight, and the
  "rejects X before GLM" cluster (currently 7 near-identical tests).
- **One lifecycle test module** parametrized over entry points; delete the six
  pasted triples.
- **One smoke execution per notebook**, carrying any assertions worth keeping,
  with cosmetics removed. Extract notebook settings resolution into a
  `resolve_settings()` function so the 10 cell-by-id `exec()` tests become
  unit tests.
- **`examples/NSD/conftest.py`** is a prerequisite for any relocation: nine
  files import fixtures from `test_nsd_cifti`, five from `test_ridge_workflow`.
- **Drop implementation-detail tests** (svd call counts, weakrefs, forbidden
  property use, filesystem-write guards ×3) and the script-output pins in the
  three simulation test files; keep the two package oracles they contain.

## 8. Expected shape after pruning

| | now | after |
| --- | ---: | ---: |
| `tests/` functions | 453 | ≈270 (206 keep + ≈55 merged + ≈11 new) |
| `tests/` wall time | ≈75 s | ≈60 s (notebook smoke opt-in) |
| `examples/NSD` functions | 148 | ≈100, opt-in |
| default `pytest` collected | 1,021 | ≈450 |
| default wall time | 244 s | ≈60 s |

---

## Appendix: per-test verdicts

Categories: A oracle · B leakage/isolation · C contract · D validation ·
E provenance pinning · F ownership/immutability · G lifecycle/logging ·
N notebook · P plotting · S simulation · H vacuous/redundant.
Verdicts: KEEP · MERGE (fold into named siblings) · CUT.


### Test audit, group 1

Files: tests/test_fit.py, tests/test_prepared_fit.py, tests/test_prepared.py, tests/test_multirun.py,
tests/test_hrf_glm.py, tests/test_data.py, tests/test_design.py (plus conftest.py / oracles.py fixtures).

Categories: A oracle, B leakage/isolation, C contract, D validation, E provenance pinning,
F ownership/immutability, G lifecycle/logging, H vacuous/redundant.

---

## tests/test_fit.py

| line | test | cat | verdict | reason (<=15 words) |
|---|---|---|---|---|
| 241 | test_fit_matches_nilearn_ols_contrast | A | KEEP | Full nilearn contrast oracle (effect/var/stat/z/p), OLS. |
| 260 | test_fit_matches_nilearn_ar1_contrast | A | MERGE | Parametrize noise_model with 241 (as test_prepared_fit 120 does). |
| 279 | test_fit_accepts_zero_sst_features_without_inference_warnings | A | KEEP | Zero-variance voxels: NaN R2, no warnings, oracle for varying voxel. |
| 321 | test_fit_rejects_run_without_positive_residual_degrees_of_freedom | D | KEEP | Saturated design must fail; real scientific guard, one per entrypoint. |
| 365 | test_fit_returns_readonly_arrays | A | KEEP | Misnamed: only lstsq R2 oracle for single-run fit. Trim readonly loop. |
| 395 | test_fit_owns_designs_and_exposes_immutable_provenance | F | MERGE | Fold design-copy check into 365; provenance keys are E noise. |
| 409 | test_fit_does_not_mutate_inputs | F | CUT | from_arrays already copies (test_data 118); fit never sees caller arrays. |
| 420 | test_fit_extends_input_provenance_and_exposes_result_provenance | E | MERGE | Activity-name pin; fold into 439. |
| 439 | test_fit_analysis_fingerprint_depends_on_data_and_model | C | KEEP | Fingerprint sensitivity underpins task_delta_r2 parent checks. |
| 476 | test_fit_serializes_model_spec_and_run_diagnostics | E | MERGE | One provenance-serialization test; absorb 548, 565, multirun 244. Keep privacy asserts. |
| 548 | test_fit_serializes_nondefault_drift_order | E | CUT | One-key pin; add drift_order=3 to 476 if desired. |
| 565 | test_fit_marks_local_callable_hrf_as_partially_reproducible | E | MERGE | Local-callable reproducibility + no memory addresses; fold into 476. |
| 597 | test_fit_does_not_write_to_filesystem | H | CUT | Monkeypatched open; triplicated (prepared_fit 822, 1302). Keep at most one suite-wide. |
| 625 | test_fit_rejects_unknown_semantic_contrast | D | MERGE | Fold into one parametrized contrast-preflight test with multirun 278/341/363. |
| 637 | test_task_delta_r2_compares_complete_and_nuisance_models | A | KEEP | Nested-OLS delta R2 vs nilearn oracle, clip semantics. Trim F loop. |
| 683 | test_task_delta_r2_rejects_broken_nested_ols_monotonicity | H | CUT | Calls private 2-line fn with literal tolerance; prepared_fit 1129 covers. |
| 690 | test_make_task_delta_r2_result_clips_and_owns_values | C | MERGE | Constructor clipping/copy; combine with 725/741/769 into one constructor test. |
| 725 | test_make_task_delta_r2_result_rejects_invalid_inputs | D | MERGE | Already parametrized; absorb 741 and 769 as extra params. |
| 741 | test_make_task_delta_r2_result_rejects_overflow_without_warning | D | MERGE | One more param of 725. |
| 769 | test_make_task_delta_r2_result_rejects_nonfinite_or_nonnumeric_nuisance_designs | D | MERGE | Two more params of 725. |
| 787 | test_task_delta_r2_rejects_full_result_for_changed_model | C | MERGE | Parametrize changed in {model,data,anonymous} with 801, 815 (cf. prepared_fit 1043). |
| 801 | test_task_delta_r2_rejects_full_result_for_changed_data | C | MERGE | See 787. |
| 815 | test_task_delta_r2_requires_fingerprintable_provenance | C | MERGE | See 787; hrf_glm 397 duplicates for selected path. |
| 829 | test_task_delta_r2_records_parent_model_and_diagnostics | E | MERGE | Keep parent_analysis_id + diagnostic_noise_model asserts inside 637; drop rest. |
| 856 | test_ordinary_complete_lifecycle | G | MERGE | Copy-paste template x6 across files; one parametrized test_lifecycle.py. |
| 900 | test_comparison_complete_lifecycle | G | MERGE | Same template. |
| 953 | test_model_provenance_adds_task_model_only_when_set | E | CUT | Private fn; hrf_glm 611/675 assert task_model provenance publicly. |
| 965 | test_model_provenance_without_task_model_keeps_legacy_keys | E | CUT | Key-set pin duplicating 476. |

Summary: The scientific core is 241/279/321/365/439/637 (nilearn oracles for contrasts, zero-SST policy, dof guard, R2, nested delta R2). Roughly half the file is provenance-dict pinning, lifecycle logging templates, and ownership boilerplate that could collapse into two or three parametrized tests without losing a single regression signal. The two lifecycle tests and the filesystem guard are verbatim copies of tests in other files.

---

## tests/test_prepared_fit.py

| line | test | cat | verdict | reason (<=15 words) |
|---|---|---|---|---|
| 120 | test_fit_prepared_matches_nilearn | A | KEEP | Parametrized OLS/AR1 nilearn contrast oracle for prepared path. |
| 142 | test_fit_prepared_combines_run_specific_designs_and_pools_r_squared | A | KEEP | Column order differs per run; fixed-effects average and pooled R2 vs oracle. |
| 246 | test_fit_prepared_rejects_invalid_semantic_contrasts | D | KEEP | Canonical parametrized contrast validator (8 cases); absorb 338, 348, 356. |
| 255 | test_fit_prepared_rejects_contrast_term_missing_from_one_run_before_glm | C | MERGE | Parametrize preflight (missing/non-estimable/all-zero) with 289, 316. |
| 289 | test_fit_prepared_rejects_non_estimable_contrast_before_glm | C | MERGE | Estimability check is real; fold into 255. |
| 316 | test_fit_prepared_rejects_all_zero_semantic_contrast_before_glm | C | MERGE | Fold into 255. |
| 338 | test_fit_prepared_rejects_nonpositive_residual_degrees_of_freedom | D | MERGE | One param in 246. |
| 348 | test_fit_prepared_rejects_unsupported_noise_model | D | MERGE | One param in 246. |
| 356 | test_fit_prepared_rejects_unsafe_model_metadata | D | MERGE | One param in 246; also dedupe with 636/683 path-like tests. |
| 368 | test_fit_prepared_owns_results_and_reports_prepared_design_provenance | F | MERGE | One ownership test with 967. |
| 404 | test_fit_prepared_records_stable_analysis_identity_and_complete_activity | E | MERGE | Full activity-dict pin; keep stability assert inside 471. |
| 471 | test_fit_prepared_analysis_fingerprint_tracks_all_fit_inputs | C | KEEP | Fingerprint changes for design/contrasts/noise/metadata; parent checks rely on it. |
| 526 | test_fit_prepared_leaves_analysis_identity_unavailable_for_anonymous_sources | C | MERGE | Anonymous branch; fold into 471. |
| 540 | test_fit_prepared_logs_lifecycle_and_resets_context | G | MERGE | Duplicates 1372 success branch; lifecycle suite. |
| 583 | test_fit_prepared_preserves_privacy_in_failure_logs_and_provenance | C | KEEP | Single privacy test: signal values never in logs/provenance. Absorb 636/683/782. |
| 636 | test_fit_prepared_rejects_path_like_model_keys_without_logging_them | D | MERGE | Fold into 583. |
| 683 | test_fit_prepared_rejects_path_like_mapping_keys_inside_metadata_sequences | D | MERGE | Fold into 583 or 246. |
| 730 | test_fit_prepared_logs_invalid_specification_failures_and_resets_context | G | CUT | Subset of 1372 early_failure plus 540. |
| 764 | test_fit_prepared_ignores_empty_sensitive_environment_values | H | CUT | Library never reads env vars; asserts only error_code. Vacuous. |
| 782 | test_fit_prepared_sanitizes_injected_traceback_and_object_repr | G | MERGE | Injected failure re-asserts sanitizer; fold one case into 583. |
| 822 | test_fit_prepared_never_writes_to_the_filesystem | H | CUT | Duplicate of test_fit 597. |
| 934 | test_task_delta_r2_prepared_uses_nested_ols_and_role_selected_designs | A | KEEP | Role-selected nuisance designs + nested-OLS oracle; irreplaceable. |
| 967 | test_task_delta_r2_prepared_returns_owned_readonly_arrays_and_copied_designs | F | MERGE | Fold into 368 (asserts not-writeable twice in same loop). |
| 1015 | test_task_delta_r2_prepared_rejects_incomplete_role_partitions | D | KEEP | Each param is a distinct scientific precondition for nested model. |
| 1043 | test_task_delta_r2_prepared_rejects_changed_parent_identity | B | KEEP | Parametrized 5-way parent identity guard; model for test_fit 787/801/815. |
| 1100 | test_task_delta_r2_prepared_rejects_invalid_parent_dimensions | D | MERGE | Add as param of 1043 (uses replace on private _r2). |
| 1116 | test_task_delta_r2_prepared_requires_reliable_parent_identity | C | MERGE | Add as param of 1043. |
| 1129 | test_task_delta_r2_prepared_rejects_materially_negative_nested_ols_difference | H | KEEP | Only public-path exercise of monotonicity guard; keep instead of test_fit 683. |
| 1150 | test_task_delta_r2_prepared_records_lifecycle_and_diagnostic_provenance | E | MERGE | nuisance_columns pin already in 934; rest to lifecycle suite. |
| 1209 | test_task_delta_r2_prepared_logs_downstream_failure_with_comparison_id | G | MERGE | Lifecycle late_failure variant; one extra assert (analysis_id present). |
| 1266 | test_task_delta_r2_prepared_logs_preidentity_failure_without_analysis_id | G | MERGE | Lifecycle early_failure variant. |
| 1302 | test_task_delta_r2_prepared_never_writes_to_the_filesystem | H | CUT | Triplicate filesystem guard. |
| 1326 | test_prepared_fitting_uses_owned_designs_without_bulk_copy | H | CUT | Forbids internal use of public properties; breaks on harmless refactor; oracle duplicates 934. |
| 1372 | test_prepared_complete_lifecycle | G | MERGE | Lifecycle template. |
| 1425 | test_prepared_comparison_complete_lifecycle | G | MERGE | Lifecycle template. |

Summary: 120/142/934 are strong nilearn oracles and 1043/1015/471 define the identity semantics that make delta R2 safe; those nine KEEPs are the file. The other 26 tests are overwhelmingly logging/privacy/path-like-key permutations and ownership checks; ten of them replay the same `emit_event(...); records[-1].get("execution_id") is None` context-reset dance. One privacy test and one lifecycle parametrization would preserve every behavioral guarantee.

---

## tests/test_prepared.py

| line | test | cat | verdict | reason (<=15 words) |
|---|---|---|---|---|
| 101 | test_prepared_analysis_owns_inputs_and_returns_defensive_state | F | MERGE | One ownership test with 129, 144. |
| 129 | test_prepared_analysis_accessors_return_defensive_copies | F | MERGE | See 101. |
| 144 | test_prepared_analysis_arrays_are_readonly | F | MERGE | See 101; duplicate not-writeable assert. |
| 188 | test_prepared_analysis_rejects_invalid_design_inputs | D | KEEP | Canonical parametrized validator; absorb 198-270, 446. |
| 198 | test_prepared_analysis_rejects_absolute_path_design_column | D | MERGE | Param of 188. |
| 214 | test_prepared_analysis_rejects_empty_design_dimensions | D | MERGE | Params of 188. |
| 224 | test_prepared_analysis_rejects_design_signal_row_mismatch | D | MERGE | Param of 188 (important case, keep as param). |
| 232 | test_prepared_analysis_rejects_inconsistent_signal_feature_counts | D | MERGE | Param of 188. |
| 248 | test_prepared_analysis_rejects_mismatched_run_counts | D | MERGE | Params of 188. |
| 258 | test_prepared_analysis_rejects_nonmapping_run_metadata | D | MERGE | Param of 188. |
| 270 | test_prepared_analysis_requires_exactly_one_timing_source | D | MERGE | Param of 188; mirrors test_data 91. |
| 277 | test_prepared_analysis_accepts_explicit_other_column_role | C | CUT | Three-line positive case; 188 already has invalid-role negative. |
| 286 | test_prepared_design_fingerprint_is_stable_and_value_sensitive | C | KEEP | Design fingerprint semantics drive parent-identity checks. |
| 305 | test_prepared_design_fingerprint_includes_structure_and_run_order | C | MERGE | Fold into 286 (column order, roles, times, run order). |
| 337 | test_prepared_design_normalization_records_lifecycle_and_provenance | E | MERGE | Full activity-dict pin incl. software versions; fold with 399. |
| 399 | test_prepared_design_provenance_retains_canonical_metadata_and_versions | E | MERGE | Keep "fingerprint ignores run_metadata / key order" asserts only. |
| 446 | test_prepared_design_provenance_rejects_nested_path_like_metadata_keys | D | MERGE | Param of 188. |
| 462 | test_prepared_design_provenance_is_private_and_warns_for_anonymous_sources | C | KEEP | Provenance never carries signal/design values; anonymous warning. |
| 485 | test_normalization_builds_one_record | H | MERGE | Monkeypatch counts ProvenanceRecord calls; identical to test_data 310; parametrize over constructors. |
| 526 | test_normalization_catches_final_failures_and_isolates_context | G | MERGE | Identical body to test_data 353; lifecycle suite. |

Summary: Only 188 (validator), 286 (fingerprint) and 462 (privacy) carry distinct guarantees. The ten validation tests are all `deepcopy(prepared_inputs); mutate; raises` and belong in one parametrized table. Both normalization tests are byte-for-byte duplicates of test_data.py.

---

## tests/test_multirun.py

| line | test | cat | verdict | reason (<=15 words) |
|---|---|---|---|---|
| 132 | test_fit_allows_run_specific_designs_and_matches_first_level_model | A | KEEP | FirstLevelModel oracle, two runs with different columns; irreplaceable. |
| 178 | test_fit_aggregates_r2_from_sums_not_run_means | A | MERGE | Partly circular (uses result.run_r2); 193 has the independent oracle. |
| 193 | test_fit_ar1_r2_uses_original_signal_space_across_runs | A | KEEP | AR1 R2 in original (unwhitened) space, pooled by sums; irreplaceable. |
| 218 | test_fit_warns_for_rank_deficient_design | C | MERGE | Pins nilearn's "Matrix is singular..." text; merge with 313, drop that string. |
| 244 | test_fit_records_run_diagnostics_without_serializing_design_values | E | MERGE | Same asserts as test_fit 476 with timing_source=frame_times. |
| 278 | test_fit_rejects_non_estimable_contrast_before_glm | C | MERGE | Preflight parametrization with 341, 363, test_fit 625; drop nilearn string. |
| 313 | test_fit_records_rank_deficiency_warning_in_provenance | C | MERGE | Fold into 218. |
| 341 | test_fit_rejects_contrast_term_missing_from_one_run_before_glm | C | MERGE | Good run-1 case; keep as param of preflight test. |
| 363 | test_fit_rejects_all_zero_semantic_contrast_before_glm | C | MERGE | Param of preflight test. |

Summary: 132 and 193 are two of the most valuable tests in the suite (the only FirstLevelModel multi-run oracle and the only AR1 original-space R2 check). The remaining seven are preflight/rank-warning permutations that duplicate test_prepared_fit 255-316 for the other entrypoint and should become a single parametrized test over entrypoint x case.

---

## tests/test_hrf_glm.py

| line | test | cat | verdict | reason (<=15 words) |
|---|---|---|---|---|
| 142 | test_voxelwise_contrasts_match_independent_run_glms | A | KEEP | Per-voxel HRF GLM vs run_glm oracle, OLS/AR1, fixed-effects average. Irreplaceable. |
| 193 | test_canonical_assignment_matches_existing_api | A | KEEP | Canonical-only selection equals plain fit to 1e-12; guards selected path. |
| 217 | test_delta_r2_uses_nested_ols_with_selected_hrfs | A | KEEP | Nested lstsq oracle with per-voxel HRFs, AR1 parent. Irreplaceable. |
| 252 | test_rejects_spatial_signature_mismatch | B | KEEP | Feature-order identity guard; absorb 364. |
| 258 | test_rejects_bad_assignment_and_library_identity | D | KEEP | Bad index / wrong library / feature count; loose regex, tighten. |
| 274 | test_selection_can_transfer_to_one_new_run | C | MERGE | Shape/finite only; fold with 410 into one transfer test. |
| 288 | test_result_owns_designs_and_records_effective_model | F | MERGE | Keep only "ignored hrf_model does not change fingerprint" assert. |
| 320 | test_delta_rejects_a_different_analysis | B | KEEP | Events/confounds/sources/model change invalidates parent; parametrized. |
| 345 | test_all_undefined_hrfs_return_nan_maps | C | KEEP | NaN policy for undefined voxels; JSON-safe provenance. Absorb 410. |
| 364 | test_signature_without_selection_is_not_silently_ignored | D | MERGE | Param of 252. |
| 370 | test_custom_hrf_keeps_condition_and_confound_names_distinct | A | KEEP | Confound named like kernel column; oracle designs. Guards name collision. |
| 391 | test_grouped_fit_rejects_missing_contrast_column | D | MERGE | Parametrize with 432. |
| 397 | test_selected_delta_requires_complete_sources | C | MERGE | Dup of test_fit 815 for selected path; param of 320. |
| 410 | test_selected_delta_handles_constant_target_after_hrf_transfer | C | MERGE | Per-voxel NaN propagation; fold into 345. |
| 432 | test_undefined_assignment_still_validates_contrasts | D | MERGE | Parametrize with 391. |
| 445 | test_completed_operations_log_their_analysis_identity | G | CUT | Strict subset of 465/509 success branch. |
| 465 | test_selected_glm_complete_lifecycle | G | MERGE | Lifecycle template. |
| 509 | test_selected_comparison_complete_lifecycle | G | MERGE | Lifecycle template. |
| 611 | test_selected_glm_group_designs_equal_scored_task_columns | B | KEEP | GLM designs identical to selection's scored designs; missing-RT indicator run 1 only. |
| 642 | test_selected_glm_spm_group_also_uses_shared_task_columns | C | MERGE | Canonical-group case; fold into 611. |
| 658 | test_selected_glm_requires_matching_task_model | B | KEEP | Selection under different task model (incl. centering) rejected. |
| 675 | test_selected_glm_accepts_selection_on_a_subset_task_model | C | KEEP | Recent feature: subset selection model accepted, both fingerprints recorded. |
| 701 | test_selected_glm_requires_matching_convolution_settings | D | MERGE | Params of 658. |
| 709 | test_task_delta_r2_uses_the_same_task_model_designs | H | CUT | delta_r2 >= 0 is guaranteed by clipping; checks no designs. Vacuous. |
| 717 | test_legacy_selected_glm_without_task_model_is_unchanged | H | CUT | Duplicates 142 line 160 assert_frame_equal. |
| 724 | test_selected_glm_reports_design_errors_with_the_run_once | C | MERGE | Same regex as test_design 238; param of 658 group. |

Summary: This is the strongest file: 142/193/217/370 are genuine oracles and 252/320/611/658 are real isolation guards for the selected-HRF path. Note that task_model_problem signals are pure noise, so the RT/trial_type-modulated selected GLM has no numerical oracle at all; 709 pretends to be one but is vacuous.

---

## tests/test_data.py

| line | test | cat | verdict | reason (<=15 words) |
|---|---|---|---|---|
| 48 | test_from_arrays_normalizes_one_run_from_tr | C | KEEP | Core constructor contract; absorb 72, 164, 205 as positive params. |
| 72 | test_from_arrays_accepts_run_wise_frame_times | C | MERGE | Param of 48. |
| 91 | test_from_arrays_requires_exactly_one_timing_source | D | MERGE | One parametrized validator with 109-299. |
| 109 | test_from_arrays_rejects_invalid_frame_times | D | MERGE | Params of validator (strictly increasing is real). |
| 118 | test_from_arrays_owns_readonly_copies | F | KEEP | Single ownership test for AnalysisData; lets test_fit 409 go. |
| 145 | test_from_arrays_owns_nested_tabular_payloads | F | CUT | Deep-copy of list-valued cells; no scientific regression possible. |
| 164 | test_from_arrays_accepts_negative_onsets_without_trial_type | C | MERGE | Param of 48 (negative onset, zero duration, no trial_type). |
| 180 | test_from_arrays_rejects_invalid_signals | D | MERGE | Params of validator. |
| 199 | test_from_arrays_rejects_nonreal_tr_values | D | MERGE | Nine params -> three (bool, str, array) in validator. |
| 205 | test_from_arrays_accepts_real_scalar_tr_values | C | MERGE | Param of 48 (np.int64 TR). |
| 211 | test_from_arrays_rejects_incompatible_runs | D | MERGE | Param of validator. |
| 218 | test_from_arrays_rejects_wrong_confound_length | D | MERGE | Param of validator. |
| 228 | test_from_arrays_rejects_invalid_event_timing | D | MERGE | Param of validator. |
| 235 | test_from_arrays_accepts_run_wise_sources_and_stable_fingerprint | C | KEEP | Metadata fingerprint stable across executions; trim MappingProxy asserts. |
| 285 | test_from_arrays_rejects_source_count_mismatch | D | MERGE | Param of validator. |
| 299 | test_from_arrays_rejects_path_like_provenance_metadata | D | MERGE | Param of validator. |
| 310 | test_normalization_builds_one_record | H | MERGE | Identical to test_prepared 485; parametrize over constructors. |
| 353 | test_normalization_catches_final_failures_and_isolates_context | G | MERGE | Identical to test_prepared 526; lifecycle suite. |

Summary: Eleven of eighteen tests are `pytest.raises(ValueError, match=...)` on from_arrays and belong in one table. 48/118/235 cover the constructor contract, copy ownership and fingerprint stability; nothing else here would catch a scientific regression.

---

## tests/test_design.py

| line | test | cat | verdict | reason (<=15 words) |
|---|---|---|---|---|
| 30 | test_compile_designs_matches_nilearn | A | KEEP | Full design vs make_first_level_design_matrix; confound selection. |
| 57 | test_compile_nuisance_designs_excludes_events_and_retains_nuisance | C | MERGE | Column-name checks; fold into 86 using cosine drift. |
| 86 | test_compile_nuisance_designs_matches_nilearn_without_events | A | KEEP | Nuisance design oracle. |
| 120 | test_compile_designs_rejects_missing_selected_confound | D | MERGE | One parametrized (compiler x case) with 130, 172, 238. |
| 130 | test_compile_designs_rejects_nonfinite_selected_confound | D | MERGE | NaN confound (fMRIPrep FD row 0) is a real hazard; keep as param. |
| 140 | test_compile_designs_warns_and_records_early_event_exclusion | C | KEEP | min_onset event exclusion warns and is recorded. |
| 172 | test_compile_designs_contextualizes_invalid_nilearn_options | D | MERGE | Params of 120. |
| 199 | test_compile_designs_with_task_model_uses_nilearn_task_columns_and_nuisance | A | KEEP | Parametric modulators + missing indicator vs nilearn; irreplaceable. |
| 238 | test_compile_designs_with_task_model_reports_data_errors_with_run | C | MERGE | Same regex as hrf_glm 724; param of 120. |

Summary: Compact and mostly oracle-driven; 30/86/199 are the only design oracles outside the HRF tests and 140 is the only min_onset exclusion test. The four validation tests should be one parametrized table.

---

## Totals

| file | n tests | KEEP | MERGE | CUT |
|---|---|---|---|---|
| test_fit.py | 28 | 6 | 16 | 6 |
| test_prepared_fit.py | 35 | 9 | 21 | 5 |
| test_prepared.py | 20 | 3 | 16 | 1 |
| test_multirun.py | 9 | 2 | 7 | 0 |
| test_hrf_glm.py | 26 | 11 | 12 | 3 |
| test_data.py | 18 | 3 | 14 | 1 |
| test_design.py | 9 | 4 | 5 | 0 |
| **total** | **145** | **38** | **91** | **16** |

## Irreplaceable (would catch a real scientific regression)

- test_fit.py:241 test_fit_matches_nilearn_ols_contrast (+260 AR1 variant)
- test_fit.py:279 test_fit_accepts_zero_sst_features_without_inference_warnings
- test_fit.py:365 test_fit_returns_readonly_arrays (lstsq R2 oracle, misnamed)
- test_fit.py:637 test_task_delta_r2_compares_complete_and_nuisance_models
- test_prepared_fit.py:120 test_fit_prepared_matches_nilearn
- test_prepared_fit.py:142 test_fit_prepared_combines_run_specific_designs_and_pools_r_squared
- test_prepared_fit.py:934 test_task_delta_r2_prepared_uses_nested_ols_and_role_selected_designs
- test_prepared_fit.py:1043 test_task_delta_r2_prepared_rejects_changed_parent_identity
- test_multirun.py:132 test_fit_allows_run_specific_designs_and_matches_first_level_model
- test_multirun.py:193 test_fit_ar1_r2_uses_original_signal_space_across_runs
- test_hrf_glm.py:142 test_voxelwise_contrasts_match_independent_run_glms
- test_hrf_glm.py:193 test_canonical_assignment_matches_existing_api
- test_hrf_glm.py:217 test_delta_r2_uses_nested_ols_with_selected_hrfs
- test_hrf_glm.py:252 test_rejects_spatial_signature_mismatch
- test_hrf_glm.py:320 test_delta_rejects_a_different_analysis
- test_hrf_glm.py:370 test_custom_hrf_keeps_condition_and_confound_names_distinct
- test_hrf_glm.py:611 test_selected_glm_group_designs_equal_scored_task_columns
- test_hrf_glm.py:658 test_selected_glm_requires_matching_task_model
- test_design.py:30 test_compile_designs_matches_nilearn
- test_design.py:86 test_compile_nuisance_designs_matches_nilearn_without_events
- test_design.py:140 test_compile_designs_warns_and_records_early_event_exclusion
- test_design.py:199 test_compile_designs_with_task_model_uses_nilearn_task_columns_and_nuisance

## Wrong / vacuous / bug-pinning

- test_hrf_glm.py:709 — asserts `delta_r2 >= 0` (guaranteed by clipping) and finiteness only; name promises design equality it never checks.
- test_prepared_fit.py:764 — sets BOLDTAILOR_EMPTY_SECRET but no source module reads os.environ; asserts only error_code.
- test_multirun.py:218, 278 — pin nilearn's internal warning text "Matrix is singular at working precision, regularizing..."; breaks on nilearn upgrade.
- test_prepared_fit.py:1326 — forbids internal access to public properties; fails on harmless refactor, not on a bug.
- test_fit.py:690 vs 637/1129 — make_task_delta_r2_result silently clips raw_delta = -0.25 while task_delta_r2 raises below -1e-12; constructor accepts scientifically impossible input without error.
- test_multirun.py:178 — expected value derived from result.run_r2 (half-circular).
- test_hrf_glm.py:258 — `match="identity|fingerprint|HRF|hrf"` matches nearly any message.

## Cross-file observations

1. `_complete_sources()` is copy-pasted in test_fit, test_prepared_fit, test_prepared, test_multirun, test_data (plus `_sources` in test_hrf_glm); move one parametrizable helper to conftest.py.
2. Four copies of the same nilearn R2 oracle (`_nilearn_original_space_r2`, `_nilearn_pooled_ols_r2`, `_ols_r2_oracle`, `_original_space_ar1_diagnostics`); belongs in tests/oracles.py.
3. The success/late_failure/early_failure lifecycle template is pasted 6x (test_fit 856/900, test_prepared_fit 1372/1425, test_hrf_glm 465/509) and `test_normalization_*` pairs are duplicated in test_data/test_prepared; all carry unused `from dataclasses import replace`. One parametrized test_lifecycle.py over (operation, late-patch target, early-bad-call) replaces 8 tests / 22 cases.
4. "Rejects X before GLM" with a fail_glm monkeypatch appears 6x (test_multirun 278/341/363, test_prepared_fit 255/289/316) plus test_fit 625; filesystem-write guard 3x.
5. Coverage gaps: no numerical oracle for the task_model (RT/trial_type) selected GLM — task_model_problem uses pure-noise signals, so contrasts are never checked; no AR1 multi-run FirstLevelModel contrast oracle (test_multirun 132 is OLS only); no test that `fit` records excluded_event_count > 0 end-to-end (only compile_designs is checked); no test of irregular (non-uniform) frame_times through fit.

### Test-suite audit, group 2 (14 files, 147 test functions, 263 collected items)

Categories: A oracle, B leakage/isolation, C contract, D validation, E provenance pinning,
F ownership/immutability, G lifecycle/logging, N notebook, H vacuous/redundant.
Verdicts: KEEP, MERGE (fold into named sibling), CUT.

## tests/test_stop_signal_demo.py (41 functions, 50 collected)

| line | test | cat | verdict | reason (<=15 words) |
|---|---|---|---|---|
| 292 | test_prepared_column_roles_follow_nuisance_compiler | C | KEEP | role map drives prepared fit; checks order, no "other" |
| 325 | test_prepared_column_roles_rejects_missing_nuisance_column | D | MERGE | parametrize with 341, 350 into one roles-validator table |
| 341 | test_prepared_column_roles_requires_task_column | D | MERGE | into 325 table |
| 350 | test_prepared_column_roles_requires_matching_run_counts | D | MERGE | into 325 table |
| 536 | test_notebook_uses_prepared_design_estimation_boundary | H | CUT | pins notebook prose/import strings; runtime activity_names (1547) already proves boundary |
| 559 | test_notebook_source_boundary_ignores_stored_outputs | H | CUT | tests the test helper `_notebook_calls`, not the package |
| 821 | test_discover_run_inputs_matches_all_entities | C | KEEP | BIDS discovery happy path for the demo loader |
| 832 | test_discover_run_inputs_rejects_missing_file | D | MERGE | parametrize missing/duplicate with 839 |
| 839 | test_discover_run_inputs_rejects_duplicate_file | D | MERGE | into 832 |
| 897 | test_common_brain_mask_intersects_runs | A | KEEP | hand-computed 342-voxel intersection, affine preserved |
| 916 | test_common_brain_mask_rejects_mismatched_affine | D | MERGE | with 983 into one geometry-mismatch table |
| 932 | test_masker_preserves_whole_brain_values | C | KEEP | FD NaN->0, float64 read-only signals; drop masker attribute pins |
| 955 | test_shared_masker_preserves_feature_order_across_runs | B | KEEP | voxel identity/order preserved across runs via spatial pattern |
| 983 | test_load_run_rejects_bold_mask_geometry_mismatch | D | MERGE | into 916 |
| 1005 | test_whole_brain_image_round_trips_mask_values | A | KEEP | round trip through nilearn inverse_transform/transform |
| 1023 | test_signal_memory_estimate_uses_float64_storage | H | CUT | restates the one-line formula; notebook smoke renders the value |
| 1029 | test_load_run_rejects_confound_length_mismatch | D | MERGE | with 1039, 1063 into one load_run-validator table |
| 1039 | test_load_run_rejects_unexpected_nonfinite_confound | D | MERGE | into 1029 |
| 1050 | test_run_sources_records_dataset_relative_inputs | C | KEEP | dataset-relative URIs are the no-absolute-path privacy contract |
| 1063 | test_load_run_rejects_event_beyond_acquisition | D | MERGE | into 1029 |
| 1074 | test_result_artifacts_follow_result_contrast_names | C | KEEP | artifact names derive from result, camelCase BIDS label |
| 1104 | test_result_artifacts_reject_casefolded_contrast_label_collisions | D | MERGE | with 1129 into label-validator table |
| 1129 | test_result_artifacts_reject_non_ascii_contrast_labels | D | MERGE | into 1104 |
| 1151 | test_result_artifacts_are_deterministic_valid_metadata | C | KEEP | irreplaceable: image voxels == result arrays, manifest sha, design TSV, determinism |
| 1321 | test_result_artifacts_serializes_clipped_task_delta_values | C | MERGE | fold negative-raw-delta variant into 1151 |
| 1361 | test_result_artifacts_rejects_common_mask_mismatch[x3] | D | KEEP | anchor validator table; fold 1389, 1415 in |
| 1389 | test_result_artifacts_rejects_task_delta_length_mismatch | D | MERGE | into 1361 |
| 1415 | test_result_artifacts_rejects_incomplete_spatial_context | D | MERGE | into 1361 |
| 1433 | test_publication_destination_defaults_to_temp | C | MERGE | into 1446 destination-policy test |
| 1446 | test_persistent_destination_is_restricted | C | KEEP | persistent writes confined to derivatives/boldtailor (data safety) |
| 1467 | test_persistent_destination_rejects_symlink_components[x2] | D | MERGE | into 1446 |
| 1482 | test_notebook_configuration_requires_data_root[x3] | D | MERGE | one configuration-cell test with 1515, 1528 |
| 1492 | test_notebook_configuration_defaults_to_complete_real_sessions | E | CUT | pins 5 private-dataset session labels; default is unrunnable (see bugs) |
| 1515 | test_notebook_configuration_normalizes_session_override | C | MERGE | into 1482 |
| 1528 | test_notebook_configuration_rejects_invalid_session_override[x4] | D | MERGE | into 1482 |
| 1538 | test_notebook_design_fit_displays_compact_variance_summary | N | CUT | AST-inspects display() calls; runtime display audit covers raw-signal leak |
| 1547 | test_variance_partition_prepared_runtime_executes_against_fixture[x2] | N | KEEP | the one notebook smoke; keep repository-root only; strip cmap/cut_coords/title/display_count pins |
| 1589 | test_variance_partition_notebook_publishes_complete_private_metadata | N | MERGE | third full notebook run; fold `_assert_published_metadata` into 1547 |
| 1610 | test_readme_links_real_data_notebook | H | CUT | pins README prose ("whole-brain" on a link line) |
| 1621 | test_protected_source_paths_include_all_inputs | H | MERGE | restates a 4-line comprehension; 1636 exercises it |
| 1636 | test_publication_refuses_overlap_with_protected_source | C | KEEP | demo-level proof that raw events can't be overwritten |

Summary: Three full notebook executions (two cwd params plus the metadata test) dominate runtime;
one execution carrying all four assertion helpers (minus cosmetic pins) loses nothing. Roughly half
the file is "rejects X" permutations that collapse into four parametrized tables. The genuinely
protective tests are the artifact-content test (1151), the feature-order test (955), the round-trip
oracle (1005), and the destination/source-protection tests (1446, 1636).

## tests/test_publication.py (26 functions, 55 collected)

| line | test | cat | verdict | reason (<=15 words) |
|---|---|---|---|---|
| 90 | test_artifact_owns_payload_bytes_and_validates_relative_posix_path | F | MERGE | fold bytes-copy assert into 117 |
| 117 | test_artifact_rejects_paths_that_are_not_relative_normalized_posix[x11] | D | KEEP | path-traversal/NUL/backslash rejection is security; prune to ~6 params |
| 122 | test_publish_rejects_empty_artifact_set | D | MERGE | preflight-rejects table with 134, 449, 473 |
| 134 | test_preflight_rejects_exact_and_casefolded_duplicate_paths[x2] | D | MERGE | into 122 table (casefold matters on macOS; keep that row) |
| 151 | test_preflight_rejects_invalid_metadata_before_writing[x5] | C | KEEP | JSON/TSV validated before any write, no debris |
| 161 | test_publication_writes_real_files_and_returns_only_artifact_paths | C | KEEP | happy path: bytes on disk equal payloads, no control dirs returned |
| 179 | test_existing_collision_is_refused_before_any_artifact_is_replaced | C | KEEP | no partial writes on collision |
| 198 | test_preflight_never_follows_destination_symlinks[x3] | C | KEEP | symlink escape at destination/parent/artifact; security |
| 223 | test_preflight_rejects_source_output_overlap_without_modifying_source | C | MERGE | source-protection table with 349, 458, 656 |
| 245 | test_failure_at_every_replace_boundary_restores_original_set[x5] | C | KEEP | irreplaceable: rollback correct at every promotion boundary |
| 285 | test_successful_overwrite_replaces_the_complete_set_and_cleans_debris | C | KEEP | overwrite=True path, no .backup debris |
| 307 | test_failure_log_is_single_jsonl_stream_and_redacts_sensitive_context | G | KEEP | ledger redaction with injected sentinels (the right way to test leaks) |
| 349 | test_source_files_are_read_only_inputs_on_success_and_failure | C | MERGE | into 223 (mode/mtime untouched) |
| 387 | test_concurrent_writers_serialize_complete_transactions | C | KEEP | irreplaceable: lock makes two writers produce a consistent set |
| 419 | test_lock_timeout_is_contextual_and_leaves_no_partial_artifact | C | KEEP | lock timeout path, no partial artifact |
| 449 | test_publication_rejects_nonfinite_timeout_before_writing[x3] | D | MERGE | into 122 table (one row) |
| 458 | test_source_overlap_through_destination_parent_alias_is_rejected | C | MERGE | into 223 (".." alias row) |
| 473 | test_control_directory_names_are_reserved_case_insensitively[x3] | D | MERGE | into 122 table (one row) |
| 482 | test_publication_works_without_descriptor_relative_operations | H | CUT | patches os.supports_dir_fd, which publication.py never reads; duplicates 161 |
| 496 | test_failed_restore_retains_original_and_reports_recovery[x2] | C | KEEP | irreplaceable: rollback failure keeps backups, reports recovery dir |
| 545 | test_removed_retention_option_is_rejected_before_writing | H | CUT | asserts Python's own TypeError for an unknown kwarg; no such option in source |
| 555 | test_staging_failure_preserves_original_and_cause[x2] | C | KEEP | pre-promotion failure chains cause, leaves original |
| 580 | test_cleanup_failure_does_not_change_publication_outcome[x2] | C | MERGE | into 555 (secondary failures don't alter outcome) |
| 607 | test_existing_transaction_is_never_removed_on_identifier_collision | C | KEEP | never rmtree a foreign transaction dir (backup safety) |
| 626 | test_publication_failure_ledger_omits_exception_names_and_text | G | MERGE | into 307 (add unprintable-exception row) |
| 656 | test_control_files_cannot_overwrite_protected_sources[x2] | C | MERGE | into 223 |

Summary: This is the strongest file in the group; the transactional/rollback/concurrency tests are
real and not reproducible elsewhere. Two tests are vacuous (482, 545: they exercise nothing in the
source). The preflight validators and the four source-protection tests collapse into two tables.

## tests/test_provenance.py (14 functions, 25 collected)

| line | test | cat | verdict | reason (<=15 words) |
|---|---|---|---|---|
| 44 | test_source_ref_owns_immutable_annotations | F | MERGE | fold into 157 (one ownership test for the module) |
| 75 | test_source_ref_accepts_dataset_relative_and_bids_uris[x2] | C | MERGE | into 105 as accept rows |
| 105 | test_source_ref_rejects_invalid_values[x11] | D | KEEP | relative/traversal/scheme URI rejection is privacy-critical; replace Path.home() param with literal |
| 120 | test_run_sources_requires_exactly_one_signal_and_events_source | D | MERGE | with 139 into RunSources-validator test |
| 139 | test_run_sources_rejects_role_mismatches_for_named_slots | D | MERGE | into 120 |
| 157 | test_provenance_record_owns_nested_inputs_and_preserves_order | F | KEEP | single ownership + event-order test for ProvenanceRecord |
| 196 | test_provenance_record_round_trips_without_sharing_state | C | KEEP | to_dict/from_dict round trip |
| 209 | test_provenance_record_validates_execution_ids_and_schema_versions | D | MERGE | schema table with 217, 225 |
| 217 | test_provenance_record_accepts_additive_version_one_fields | C | MERGE | into 209 (forward-compat row) |
| 225 | test_provenance_record_rejects_top_level_digest_field | D | MERGE | into 209 |
| 230 | test_canonical_json_and_metadata_fingerprint_are_stable_across_mapping_order | C | KEEP | canonicalization is what makes fingerprints comparable |
| 294 | test_incomplete_sources_clear_metadata_fingerprint_and_add_quality_warning | C | KEEP | fingerprint None when metadata incomplete; downstream analysis_id depends |
| 311 | test_serialized_record_excludes_digest_and_sensitive_runtime_strings | H | CUT | machine-specific: asserts "/Users/poldrack/...", Path.home().name, hostname absent from a record that never contained them |
| 323 | test_extension_preserves_fields_without_parent_serialization | C | KEEP | extend_provenance preserves extra fields; drop the to_dict monkeypatch (implementation pin) |

Summary: Six tests carry the module (URI validation, ownership/order, round trip, canonical
fingerprint, incomplete-source policy, extension). Test 311 is the clearest vacuous test in the group.
Validators collapse into two tables.

## tests/test_logging.py (11 functions, 20 collected)

| line | test | cat | verdict | reason (<=15 words) |
|---|---|---|---|---|
| 49 | test_bind_context_nests_optional_ids_and_resets_after_success | G | KEEP | core context nesting/reset semantics |
| 88 | test_from_arrays_logs_structured_records_without_mutating_loggers | G | KEEP | no raw label/value leak, no logger mutation; drop hostname/home/cwd asserts |
| 143 | test_from_arrays_logs_failure_and_resets_context_after_exception | G | MERGE | parametrize failure-logging over from_arrays/fit/task_delta_r2 with 216, 312 |
| 176 | test_fit_logs_structured_records_with_correlated_ids | G | MERGE | parametrize success-logging over fit/task_delta_r2 with 277 |
| 216 | test_fit_logs_failure_and_resets_context_after_exception | G | MERGE | into 143 |
| 277 | test_task_delta_r2_logs_structured_records_with_comparison_id | G | MERGE | into 176 |
| 312 | test_task_delta_r2_logs_failure_and_resets_context | G | MERGE | into 143 |
| 345 | test_task_delta_r2_logs_provenance_failure_before_completion | G | CUT | duplicates test_fit_lifecycle:80[construction], same patched symbol |
| 390 | test_failure_events_export_only_fixed_categories[x7] | G | KEEP | error-code mapping table; proves no message leak |
| 398 | test_failure_logging_does_not_stringify_exceptions | G | MERGE | into 390 as a row |
| 409 | test_emit_event_preserves_custom_numeric_levels[x4] | H | CUT | custom levels 15/25/35/45 unused anywhere in package |

Summary: Three copies of "success logs started/completed with correlated ids" and three copies of
"failure logs X_failed and resets context" differ only in the entry point; two parametrized tests
replace six. Machine-specific leak asserts (hostname, Path.home().name) should become injected
sentinels as in test_publication:307.

## tests/test_bids_provenance.py (12 functions, 21 collected)

| line | test | cat | verdict | reason (<=15 words) |
|---|---|---|---|---|
| 98 | test_projection_uses_installed_version | C | MERGE | into 125 as a monkeypatched variant |
| 125 | test_projection_matches_hand_authored_pinned_draft_fixture | E | KEEP | golden BEP028 snapshot; field names are the contract |
| 136 | test_projection_is_immutable_bytes_and_performs_no_io | C | MERGE | into 125 (no-I/O patch + MappingProxy type) |
| 160 | test_stable_dataset_description_normalizes_optional_metadata | E | CUT | same inputs as 125; dataset_description.json already in golden fixture |
| 194 | test_draft_records_form_semantic_activity_source_graph | C | KEEP | semantic graph check survives fixture regeneration |
| 218 | test_opt_out_omits_only_pinned_draft_projection | C | KEEP | export_bids_prov=False yields exactly the 3 stable files |
| 235 | test_logs_are_deterministic_newline_terminated_and_exclude_sensitive_runtime | C | MERGE | determinism into 125; forbidden-key list mirrors source `_SENSITIVE_KEYS` (tautological) |
| 264 | test_projection_rejects_invalid_bids_provenance_labels[x5] | D | KEEP | anchor rejects-table; fold 279, 287, 301, 312 in |
| 279 | test_projection_rejects_unsafe_derivative_sidecar_paths[x5] | D | MERGE | into 264 |
| 287 | test_projection_rejects_case_folded_output_collisions | D | MERGE | into 264 |
| 301 | test_projection_rejects_sidecars_that_overwrite_reserved_artifacts[x2] | D | MERGE | into 264 |
| 312 | test_projection_rejects_relationship_sources_absent_from_record | D | MERGE | into 264 (privacy-relevant row, keep it) |

Summary: The golden fixture (125) plus the semantic graph test (194) are the real protection; 160 and
most of 235 re-assert what the snapshot already fixes. Five rejects tests collapse to one table.

## tests/test_fit_lifecycle.py (5 functions, 8 collected)

| line | test | cat | verdict | reason (<=15 words) |
|---|---|---|---|---|
| 15 | test_late_constructor_failure_never_logs_completion | G | KEEP | completion must not be logged if body raises after provenance() |
| 31 | test_completion_matches_returned_provenance | G | KEEP | log record equals provenance event; id correlation |
| 47 | test_anonymous_child_restores_outer_context[x2] | G | MERGE | into 104 (context restoration) |
| 80 | test_invalid_finalization_logs_failure[x3] | G | KEEP | missing/duplicate/construction failures logged; supersedes test_logging:345 |
| 104 | test_nested_operations_restore_context_and_bound_history | G | KEEP | bounded history (8) and nested restore |

Summary: Tight, non-redundant file; only the anonymous-child test folds into the nested test.

## tests/test_software.py (3 functions, 3 collected)

| line | test | cat | verdict | reason (<=15 words) |
|---|---|---|---|---|
| 10 | test_source_checkout_version_is_explicitly_unknown | C | KEEP | "unknown" fallback only for own distribution |
| 22 | test_source_import_does_not_require_own_distribution_metadata | C | KEEP | importable from a source checkout (user-facing) |
| 41 | test_imports_do_not_query_versions | C | MERGE | implementation pin that makes 22 pass; fold into same subprocess |

Summary: Two subprocess spawns for one contract; combine.

## tests/test_distribution.py (2 functions, 2 collected)

| line | test | cat | verdict | reason (<=15 words) |
|---|---|---|---|---|
| 5 | test_notebook_kernel_is_not_a_runtime_requirement | C | KEEP | cheap packaging contract; ipykernel stays in dev group |
| 10 | test_all_package_initializers_are_empty | C | KEEP | enforces the project's empty-__init__ rule |

Summary: Both cheap and meaningful; keep.

## tests/test_repository_contracts.py (1 function, 1 collected)

| line | test | cat | verdict | reason (<=15 words) |
|---|---|---|---|---|
| 7 | test_project_uses_uv_managed_src_layout | E | CUT | pins dependency strings; every dep bump edits a test; uv/build verify this |

Summary: A test that re-reads pyproject.toml is the wrong tool; CI `uv sync` and the build backend enforce layout.

## tests/test_arrays.py (2 functions, 4 collected)

| line | test | cat | verdict | reason (<=15 words) |
|---|---|---|---|---|
| 10 | test_readonly_array_owns_values_and_keeps_dtype[x3] | F | KEEP | the shared primitive used 52x in src; owndata/contiguous/no-share is the contract |
| 29 | test_readonly_array_removes_subclass_behavior | F | MERGE | add subclass source as a param of 10 |

Summary: This is where ownership belongs; most F tests elsewhere are downstream of this primitive.

## tests/test_result_schemas.py (5 functions, 9 collected)

| line | test | cat | verdict | reason (<=15 words) |
|---|---|---|---|---|
| 12 | test_candidate_scores_name_the_grid_and_scientific_basis[x2] | C | KEEP | grid sorting, cv_r2 == 1 - SSE/SST identity, dtypes |
| 37 | test_trial_results_share_science_and_name_their_design[x4] | C | MERGE | collapse fractional axis (schema identical); keep selected/shared |
| 71 | test_shared_trial_design_owns_its_input_table | F | MERGE | into 81 |
| 81 | test_selected_trial_design_owns_arrays_and_mapping | F | KEEP | single design-ownership test, no fit needed |
| 99 | test_custom_hrf_has_a_shared_trial_design | C | MERGE | into 37 as hrf=custom row |

Summary: Four full single-trial fits to check result types is expensive for the information gained; two suffice.

## tests/test_fit_diagnostics.py (3 functions, 3 collected)

| line | test | cat | verdict | reason (<=15 words) |
|---|---|---|---|---|
| 5 | test_nested_ols_tolerance_keeps_roundoff_but_rejects_material_loss | C | KEEP | 1e-12 clip boundary guards the delta-R2 policy |
| 13 | test_rank_warning_names_the_affected_run | E | CUT | pins message text of 3-line helper; rank warnings belong in fit tests |
| 20 | test_contrast_metadata_preserves_expression_weights_and_ownership | C | MERGE | into test_model:7 (ModelSpec ownership); wrong file |

Summary: One real boundary test; the other two are string pins or misplaced.

## tests/test_model.py (12 functions, 40 collected)

| line | test | cat | verdict | reason (<=15 words) |
|---|---|---|---|---|
| 7 | test_model_spec_owns_contrasts_and_confounds | F | KEEP | single ModelSpec ownership test; absorb fit_diagnostics:20 |
| 26 | test_model_spec_accepts_symbolic_contrast | C | MERGE | into 7 (defaults ar1 / -24 as one assert) |
| 35 | test_model_spec_rejects_unknown_noise_model[x5] | D | MERGE | into 54 table; 5 bad strings -> 1 |
| 40 | test_model_spec_rejects_positional_contrast_vector | D | MERGE | into 54 table |
| 54 | test_model_spec_rejects_invalid_weight_mapping[x4] | D | KEEP | anchor table; all-zero and NaN weights are real science bugs |
| 60 | test_model_spec_rejects_boolean_weight[x4] | D | MERGE | into 54; True/False/np.bool_ x2 -> 1 row |
| 69 | test_model_spec_rejects_boolean_design_option[x16] | D | MERGE | 16 items -> 1 row in 54 |
| 74 | test_model_spec_rejects_empty_contrasts | D | MERGE | into 54 |
| 79 | test_model_spec_rejects_nonfinite_min_onset | D | MERGE | into 54 |
| 83 | test_model_spec_accepts_task_model_with_spm_or_glover | C | KEEP | task_model x hrf_model compatibility anchor; fold 95, 101 in |
| 95 | test_task_model_requires_single_column_string_hrf[x4] | D | MERGE | into 83 |
| 101 | test_task_model_must_be_a_task_model | D | MERGE | into 83 |

Summary: 40 collected items for a dataclass validator; one accept test, one ownership test, and one
rejects table (~10 rows) retain every distinct failure mode.

## tests/test_task_model.py (10 functions, 22 collected)

| line | test | cat | verdict | reason (<=15 words) |
|---|---|---|---|---|
| 10 | test_default_task_model_is_task_only | C | MERGE | into 19 |
| 19 | test_nsd_task_model_names_and_dict | C | KEEP | regressor/profiled name order drives design columns and provenance |
| 37 | test_fingerprint_is_sha256_of_canonical_dict_and_changes_with_settings | C | KEEP | fingerprint sensitivity to center/missing gates selection reuse |
| 57 | test_reserved_or_invalid_modulator_columns_rejected[x8] | D | KEEP | reserved names collide with design columns; anchor table |
| 62 | test_trial_type_is_an_ordinary_modulator_column | H | CUT | asserts constructor stores its argument; 19 already builds it |
| 66 | test_invalid_missing_policy_rejected[x4] | D | MERGE | into 57 |
| 73 | test_center_must_be_boolean[x3] | D | MERGE | into 57 |
| 78 | test_duplicate_or_non_modulator_entries_rejected | D | MERGE | into 57 |
| 85 | test_task_model_is_immutable_and_owns_its_tuple | F | MERGE | one-line assert into 19 |
| 94 | test_subset_requires_identical_shared_modulators | B | KEEP | is_subset_of gates selection->fit compatibility (commit afb9bc3) |

Summary: Four tests carry the module (names/dict, fingerprint, reserved-column table, subset rule).

## Totals

| file | n tests | collected | KEEP | MERGE | CUT |
|---|---|---|---|---|---|
| test_stop_signal_demo.py | 41 | 50 | 13 | 22 | 6 |
| test_publication.py | 26 | 55 | 13 | 11 | 2 |
| test_provenance.py | 14 | 25 | 6 | 7 | 1 |
| test_logging.py | 11 | 20 | 3 | 6 | 2 |
| test_bids_provenance.py | 12 | 21 | 4 | 7 | 1 |
| test_fit_lifecycle.py | 5 | 8 | 4 | 1 | 0 |
| test_software.py | 3 | 3 | 2 | 1 | 0 |
| test_distribution.py | 2 | 2 | 2 | 0 | 0 |
| test_repository_contracts.py | 1 | 1 | 0 | 0 | 1 |
| test_arrays.py | 2 | 4 | 1 | 1 | 0 |
| test_result_schemas.py | 5 | 9 | 2 | 3 | 0 |
| test_fit_diagnostics.py | 3 | 3 | 1 | 1 | 1 |
| test_model.py | 12 | 40 | 3 | 9 | 0 |
| test_task_model.py | 10 | 22 | 4 | 5 | 1 |
| **Total** | **147** | **263** | **58** | **74** | **15** |

## Irreplaceable tests

- test_stop_signal_demo.py:1151 test_result_artifacts_are_deterministic_valid_metadata
- test_stop_signal_demo.py:1547 test_variance_partition_prepared_runtime_executes_against_fixture[repository-root]
- test_stop_signal_demo.py:955 test_shared_masker_preserves_feature_order_across_runs
- test_stop_signal_demo.py:1005 test_whole_brain_image_round_trips_mask_values
- test_stop_signal_demo.py:1446 test_persistent_destination_is_restricted
- test_publication.py:245 test_failure_at_every_replace_boundary_restores_original_set
- test_publication.py:387 test_concurrent_writers_serialize_complete_transactions
- test_publication.py:496 test_failed_restore_retains_original_and_reports_recovery
- test_publication.py:198 test_preflight_never_follows_destination_symlinks
- test_publication.py:117 test_artifact_rejects_paths_that_are_not_relative_normalized_posix
- test_provenance.py:230 test_canonical_json_and_metadata_fingerprint_are_stable_across_mapping_order
- test_provenance.py:294 test_incomplete_sources_clear_metadata_fingerprint_and_add_quality_warning
- test_bids_provenance.py:125 test_projection_matches_hand_authored_pinned_draft_fixture
- test_bids_provenance.py:194 test_draft_records_form_semantic_activity_source_graph
- test_fit_lifecycle.py:15 test_late_constructor_failure_never_logs_completion
- test_logging.py:390 test_failure_events_export_only_fixed_categories
- test_task_model.py:94 test_subset_requires_identical_shared_modulators
- test_task_model.py:37 test_fingerprint_is_sha256_of_canonical_dict_and_changes_with_settings
- test_fit_diagnostics.py:5 test_nested_ols_tolerance_keeps_roundoff_but_rejects_material_loss
- test_arrays.py:10 test_readonly_array_owns_values_and_keeps_dtype
- test_distribution.py:10 test_all_package_initializers_are_empty

## Wrong or bug-pinning tests

1. Mis-weighted contrast pinned. `stop_vs_go = "(stop_success + stop_failure) - go_success"` has
   weights (1, 1, -1); a stop-vs-go difference needs (0.5, 0.5, -1). Pinned at
   test_stop_signal_demo.py:52, :57 (NOTEBOOK_CONTRAST_EXPRESSIONS), :887 (example_result fixture),
   and asserted into published config (:656) and fit_prepared provenance (:731). Source of the bug:
   examples/stop_signal_demo.ipynb cells `ingestion` and `design-fit`. Fix notebook and tests together.
2. Unrunnable default pinned. test_stop_signal_demo.py:1492 pins DEFAULT_SESSIONS = 5 sessions, but
   the notebook's `design-fit` cell does `plt.subplots(1, 2)` and `zip(SESSIONS, ..., design_axes,
   strict=True)`, and `_configured_sessions()` demands exactly two when overridden. With the default,
   the notebook raises in the design-fit cell. Every executing test injects a 2-session override, so the
   suite never runs the default it pins. The notebook title says "Five-session" while
   provenance_metadata says "two-session".
3. Vacuous machine-specific asserts: test_provenance.py:311 (asserts
   "/Users/poldrack/Dropbox/code/boldtailor", Path.home().name, hostname absent); test_logging.py:88
   and :137-140 (same pattern); test_provenance.py:102 uses Path.home() as a parametrize value.
4. No-op monkeypatch / nonexistent option: test_publication.py:482 patches `os.supports_dir_fd`
   (never consulted in publication.py); test_publication.py:545 asserts TypeError for
   `retain_incomplete`, a kwarg that does not exist, i.e. Python's default behaviour.
5. test_fit_diagnostics.py:13 pins the exact rank-warning message text.
6. test_repository_contracts.py:7 pins "nilearn>=0.14.0,<0.15" and "requires-python >=3.12" as
   literal strings; a dependency bump breaks a test without any behaviour change.

### Test audit — Group 3: numerical / scientific core

Scope: 18 files, 161 test functions. Source: `/Users/poldrack/Dropbox/code/boldtailor/src/boldtailor`.
Categories: A oracle, B leakage/isolation, C contract, D validation, E provenance pinning, F ownership/immutability, S simulation, H vacuous/redundant.

Shared helpers: `tests/conftest.py` (`ridge_problem`, `selected_fixture`), `tests/oracles.py` (`fractional_beta_oracle` = augmented lstsq + brentq in the *raw* basis, independent of the production SVD/bisection; `subset_runs`).

---

## tests/test_single_trial.py (19)

| line | test | cat | verdict | reason (<=15 words) |
|---|---|---|---|---|
| 95 | test_normalized_ridge_leaves_nuisance_unpenalized | A | KEEP | Hand-computed closed form; proves normalization convention beta/(1+alpha) and unpenalized intercept |
| 109 | test_beta_path_matches_augmented_ols | A | KEEP | lstsq oracle for path + prepared solver, ill-conditioned branch, NaN constant feature |
| 146 | test_beta_path_factors_design_once | H | CUT | Monkeypatch call-count of private `_project_design`; performance pin, not behavior |
| 162 | test_beta_path_retains_solver_rejections | D | MERGE | Fold into 271/278 (same rank/dof/alpha validators via path API) |
| 181 | test_prepared_trial_betas_do_not_retain_outputs | F | CUT | weakref/gc memory check; nothing behavioral; brittle under refcount changes |
| 194 | test_betas_and_pooled_r2_match_independent_augmented_ols | A | KEEP | Public API vs lstsq; pooled (not mean) R2; nuisance/delta R2 |
| 228 | test_rt_and_stimulus_ids_never_change_fit | C | KEEP | Metadata columns cannot leak into design; cheap, important |
| 243 | test_constant_features_keep_nan_positions_and_zero_sums | C | KEEP | NaN policy for constant features with zero SSE/SST |
| 254 | test_redundant_nuisance_has_same_fit | C | KEEP | Rank-deficient nuisance tolerated and gives identical fit |
| 265 | test_invalid_alpha_rejected | D | KEEP | One validator sweep, 5 params; adequate |
| 271 | test_unidentifiable_trials_rejected | D | KEEP | Rank/support rejection (absorb 162 here) |
| 278 | test_no_residual_degrees_of_freedom_rejected | D | MERGE | Into 271; same validator family |
| 283 | test_run_labels_are_unique_and_checked | D | KEEP | Label uniqueness/count/path-leak contract |
| 289 | test_results_are_readonly_and_metadata_is_copied | F | KEEP | One ownership test per result type is fine; this is it |
| 307 | test_provenance_tracks_design_and_penalty_without_behavior_values | E | KEEP | Fingerprint sensitivity (alpha, onset) + no PII in logs; behavioral |
| 337 | test_known_rt_variability_with_ar_noise_and_nuisance_only_feature | S | KEEP | Recovery property, but noise sd 0.01 vs amplitude ~1 -> essentially noiseless; raise noise |
| 372 | test_nested_event_metadata_is_owned_on_every_result_access | F | MERGE | Into 289 (same deep-copy ownership theme) |
| 388 | test_ols_and_ridge_preserve_supported_trials_with_small_hrf_support | A | KEEP | Closed form 2/(1+alpha) for a one-sample onset near run end; real edge case |
| 405 | test_shared_trials_complete_lifecycle | E | MERGE | Identical pattern in test_fit/test_hrf_glm/test_prepared_fit/test_selected_hrf_fit; one parametrized lifecycle test |

Summary: The core here is strong: three independent lstsq/closed-form oracles (95, 109, 194, 388) plus the NaN/constant/redundant-nuisance contracts. Note the lstsq oracle re-derives the implementation's penalty scaling (`sqrt(alpha)*||x_perp||`) — the *solve* is independent but the *normalization convention* is only pinned by 95. The monkeypatch call-count, weakref, and lifecycle tests are not numerical and belong elsewhere or nowhere.

## tests/test_single_trial_design.py (9)

| line | test | cat | verdict | reason |
|---|---|---|---|---|
| 35 | test_subtr_timing_repeat_identity_and_row_order | A | KEEP | compute_regressor oracle per trial, column order, trial_id format, nuisance passthrough |
| 51 | test_rt_and_image_identity_have_no_design_effect | C | MERGE | Into test_single_trial:228 (same property one layer down) |
| 59 | test_zero_duration_has_nilearn_impulse_convention | A | KEEP | Nilearn impulse convention for duration 0; subtle and load-bearing |
| 71 | test_invalid_timing_is_rejected | D | KEEP | One timing validator sweep |
| 78 | test_events_without_supported_response_are_rejected | D | MERGE | Into test_hrf_design:84 (same onset-support rule, better coverage there) |
| 89 | test_reserved_event_names_are_rejected | D | KEEP | Reserved-name collision is a real user hazard |
| 96 | test_invalid_nuisance_is_rejected | D | KEEP | Four nuisance failures in one test |
| 108 | test_input_tables_are_not_mutated | F | KEEP | Cheap, one per compiler |
| 117 | test_nuisance_cannot_reuse_a_generated_trial_name | D | MERGE | Into 89 (name collision family) |

Summary: Small, mostly fine. The design compiler is oracle-checked against `compute_regressor` including the duration-0 impulse case. Two validators can be merged into siblings.

## tests/test_selected_hrf_fit.py (5)

| line | test | cat | verdict | reason |
|---|---|---|---|---|
| 15 | test_grouped_betas_and_r_squared_match_augmented_ols | A | KEEP | Per-feature HRF groups vs compute_regressor+lstsq; design matrices pinned; R2 pooled |
| 87 | test_identity_checks_and_apply_to_new_runs | D/C | KEEP | Signature/library/index mismatch + apply to a new run; dense and useful |
| 128 | test_canonical_only_matches_legacy_including_per_run_constant | A | KEEP | Equivalence to fit_single_trials when library is canonical-only |
| 147 | test_grouped_results_own_nested_metadata_and_record_designs | F/E | KEEP | Ownership + fingerprint presence; one per result type |
| 176 | test_selected_trials_complete_lifecycle | E | MERGE | Fifth copy of lifecycle pattern; parametrize across entry points |

Summary: Tight file. The oracle in line 15 is fully independent (nilearn convolution + numpy lstsq). Only the lifecycle duplicate should go.

## tests/test_fractional_ridge.py (9)

| line | test | cat | verdict | reason |
|---|---|---|---|---|
| 35 | test_fraction_path_matches_requested_norm_and_oracle | A | KEEP | brentq/lstsq oracle for betas and implied alpha; norm ratio exact; per-feature alphas differ |
| 52 | test_fraction_mapping_preserves_target_scaling_and_unpenalized_confounds | A/C | KEEP | Scale/offset equivariance, alpha invariance, nuisance via oracle, SSE closed form |
| 79 | test_fraction_solver_handles_ill_conditioning_and_undefined_features | C | KEEP | NaN fraction map, constant/nuisance-span features NaN, cond ~1e8 still hits fraction |
| 118 | test_invalid_fraction_grids_fail | D | KEEP | One grid validator sweep |
| 125 | test_public_fraction_fits_keep_trial_units_and_hrf_groups | A/E | KEEP | Public API vs oracle incl. per-feature HRF; mutual exclusion with alpha; strings pinned (E part could go) |
| 169 | test_fraction_one_matches_existing_ols | A | MERGE | Into 125 (fraction 1 == OLS is one extra assert) |
| 178 | test_prepared_fraction_betas_reuse_state_in_any_order | A/H | MERGE | Oracle in any order is useful; `np.linalg.svd` monkeypatch and weakref are implementation pins; fold oracle part into 35 |
| 204 | test_prepared_fraction_rejects_invalid_values | D | MERGE | Into 118 (same `fraction_map` validator) |
| 211 | test_fraction_results_own_arrays_without_solver_mutation | F/H | CUT | Monkeypatches nonexistent `freeze_fraction_result` with `raising=False` -> vacuous; rest is dataclasses.replace semantics |

Summary: The fractional solver is the best-oracled component in the package: `tests/oracles.py` solves the augmented problem directly and root-finds with brentq, so the production log-space bisection is independently checked for betas *and* alpha. Test 211 is partly vacuous (patches a symbol that does not exist in `src/`), and two tests pin memory/factorization behavior rather than numerics.

## tests/test_fractional_cv.py (7)

| line | test | cat | verdict | reason |
|---|---|---|---|---|
| 21 | test_selector_chooses_different_fractions_and_preserves_undefined_locations | C | KEEP | Hand-built scores: argmax, 1e-12 tie -> less shrinkage, grid sorted desc, NaN/mask |
| 45 | test_selector_rejects_misaligned_inputs | D | KEEP | One validator sweep |
| 122 | test_fraction_cv_matches_nested_fixed_ols_oracle | A/B | KEEP | Full nested LORO reference with inner select_hrf, fixed-OLS targets, both encoding modes |
| 149 | test_inner_validation_data_does_not_change_hrf_or_training_transform | B | MERGE | Same perturbation as test_within_run_cv_isolation:35[fractional=True]; keep one |
| 173 | test_fixed_targets_do_not_depend_on_fraction_grid | C | KEEP | SST grid-invariant and SSE aligns across grids: proves fixed targets |
| 189 | test_prepared_fraction_run_betas_match_oracle_in_any_order | A | KEEP | `_ridge_cv.prepare_run_beta_path(fractional=True)` vs oracle incl. HRF ids and -1 |
| 216 | test_fraction_cv_prepares_each_run_once_per_fold | H | CUT | Call-count monkeypatch; oracle half duplicates 122[optimized=False] |

Summary: The nested-CV reference (`reference`) is a genuinely independent re-implementation (per-run oracle betas, explicit dummy-intercept lstsq, within-run residual centering). One isolation test and one call-count test are redundant with siblings.

## tests/test_fractional_ridge_ablation.py (6) — targets `examples/validation/fractional_ridge_ablation.py`

| line | test | cat | verdict | reason |
|---|---|---|---|---|
| 91 | test_split_matches_augmented_oracle | A/S | KEEP | Block-diagonal augmented oracle for per_run/pooled, normalized/raw; 8 params |
| 125 | test_raw_per_run_ablation_agrees_with_production | A | KEEP | Experiment code equals production path; guards drift of the ablation baseline |
| 152 | test_held_out_data_never_changes_training_state | B | KEEP | Strong perturbation (x*-3+55, x*7, design*2); checks alphas/coefs/calibration |
| 182 | test_fixed_targets_do_not_shrink_with_candidate | C | MERGE | Into 91 or test_fractional_cv:173 (same fixed-target property) |
| 194 | test_training_only_calibration_and_identity_fallback | A | KEEP | Hand-computed affine calibration mapping |
| 204 | test_basis_and_pooling_are_distinct_models | C | CUT | Only asserts three configs differ; would pass with any wrong-but-different code |

Summary: These test an *example script*, not the package. The oracle test is good and the isolation test is strong, but the whole file runs example code that could drift without affecting users; consider moving under `examples/validation/tests` or marking slow.

## tests/test_fractional_ablation_simulation.py (5) — targets `examples/validation/fractional_ridge_ablation_simulation.py`

| line | test | cat | verdict | reason |
|---|---|---|---|---|
| 33 | test_generator_is_paired_and_stresses_the_design | S | MERGE | Shapes/determinism of generator; fold into 61 |
| 61 | test_noiseless_ols_recovers_truth | S | KEEP | beta_rmse<1e-9 is meaningful; `selected_fraction==1` trivially true (grid has only 1.0); row counts 51/24 are pinning |
| 82 | test_outer_outcomes_and_latent_truth_do_not_select_candidates | B | KEEP | Inner selection invariant to held-out outcomes and truth; strong perturbation |
| 110 | test_metrics_keep_offsets_and_shape_errors_distinct | A | KEEP | Hand-computed within-run RMSE/ratio vs raw RMSE |
| 122 | test_export_is_complete_and_uncertainty_uses_seeds | S | CUT | Row counts 102/96/34, column sets, file names; `paired base == 0` is self-comparison; slow and brittle |

Summary: Half of this is export-format pinning for a simulation script. Nothing here demonstrates fraction *recovery under noise* (the scientific point of the experiment); the only recovery claim is the noiseless case.

## tests/test_ridge_cv.py (9)

| line | test | cat | verdict | reason |
|---|---|---|---|---|
| 108 | test_cv_matches_independent_augmented_ols_and_encoding | A/B | KEEP | Nested LORO with inner HRF selection vs lstsq reference; both modes; readonly |
| 145 | test_validation_targets_use_candidate_penalty | C | KEEP | Distinguishes candidate-regularized vs OLS targets; would catch target swap |
| 156 | test_inner_validation_cannot_train_hrf_or_encoding | B | MERGE | Duplicate of test_within_run_cv_isolation:35[fractional=False]; keep the parametrized one |
| 212 | test_behavior_is_in_scoring_provenance | E | CUT | Fingerprint differs + two literal strings; low information |
| 225 | test_too_few_runs_rejected | D | KEEP | Two-run minimum is a real contract |
| 236 | test_validation_design_failure_names_fold_and_run | D | KEEP | Error message names the run; useful for users |
| 250 | test_modes_identify_objective_and_default | C/E | KEEP | Default == within_run, modes differ, train predictor means pinned numerically; trim string pins |
| 309 | test_encoding_preflight_precedes_hrf_fitting | C | MERGE | 'rank' case subsumed by test_within_run_cv_isolation:13 (stronger); keep 'mode' case there |
| 333 | test_prepared_run_betas_preserve_feature_order_and_candidate_independence | A | KEEP | Alpha-path solver vs augmented lstsq with mixed HRF ids and -1 |

Summary: Two strong oracle tests (108, 333) and a target-semantics test (145) carry this file. The leakage tests are triplicated across test_ridge_cv, test_fractional_cv and test_within_run_cv_isolation; consolidate in the latter.

## tests/test_ridge_selection.py (7)

| line | test | cat | verdict | reason |
|---|---|---|---|---|
| 19 | test_percentile_uses_all_grayordinates | C | KEEP | Percentile over population, but data do not discriminate `linear` from `lower` method |
| 26 | test_common_finite_mask_and_negative_scores | C | KEEP | Common finite mask across alphas; hand-computed percentile |
| 33 | test_anatomical_mask_and_sort_preserve_candidate_identity | C/F | KEEP | Grid sorting keeps alpha identity; mask; readonly |
| 47 | test_near_ties_prefer_smaller_penalty | C | KEEP | Tie rule within 1e-12 |
| 64 | test_invalid_grid_rejected | D | KEEP | One grid validator sweep (7 params is generous; fine) |
| 70 | test_invalid_percentile_rejected | D | MERGE | Into 64 as a second parametrize axis |
| 84 | test_empty_or_malformed_population_rejected | D | MERGE | Into 64 |

Summary: Small and correct; selector semantics (mask, tie, sort) are well pinned by hand computation. Missing a percentile case where interpolation method matters (e.g. 6 distinct values at p=90).

## tests/test_ridge_objective_simulation.py (8) — targets `examples/validation/ridge_objective_simulation.py` (+2 package tests)

| line | test | cat | verdict | reason |
|---|---|---|---|---|
| 8 | test_centering_removes_only_held_out_offset | A | KEEP | Hand-computed R2 (1-400/45 vs 1) for current vs centered objective |
| 25 | test_pool_sums_before_division_and_exclude_missing_predictions | A | KEEP | Hand-computed pooled R2 with NaN exclusion |
| 35 | test_fixed_ols_targets_are_candidate_independent | C | MERGE | Same property as test_fractional_ridge_ablation:182 and test_fractional_cv:173 |
| 48 | test_simulation_records_both_regularizers_and_four_objectives | S | CUT | Row counts and label sets; finiteness only |
| 68 | test_current_alpha_score_matches_matched_filter_limit | A | KEEP | Package test: alpha->inf ridge equals matched filter, closed form; misplaced in this file |
| 94 | test_historical_objectives_keep_shared_intercept_fit | A | KEEP | lstsq oracle for shared-intercept objectives vs within_run slope 3 |
| 117 | test_recovery_metrics_center_each_run_before_pooling | A | MERGE | Near-duplicate of test_fractional_ablation_simulation:110 (same sqrt(8/3), ratio 2) |
| 131 | test_confounding_stress_experiment_recovers_known_slopes | S | KEEP | Scientifically meaningful: within_run recovers slope 3, offset-confounded objectives are biased |

Summary: Mixed bag: two package-level closed-form oracles (68, 94) hide in a simulation file and should move to test_trial_encoding / test_single_trial. The simulation half has one meaningful recovery test (131) and one count-pinning test (48).

## tests/test_trial_encoding.py (13)

| line | test | cat | verdict | reason |
|---|---|---|---|---|
| 36 | test_no_validation_intercept_refit | A | KEEP | Hand-computed 1-400/45; proves intercept not refit on test |
| 47 | test_predictions_and_pooled_loss_match_independent_ols | A/F | KEEP | lstsq oracle, NaN masks, pooled vs mean R2, input immutability, readonly |
| 98 | test_validation_covariates_do_not_change_training_transform | B | KEEP | Test-run predictors cannot alter coefficients/means; cheap |
| 108 | test_constant_and_incomplete_features_are_undefined | C | MERGE | Into 293 (same NaN/constant policy, 293 is more complete) |
| 131 | test_bad_splits_rejected | D | KEEP | One split validator sweep (8 params; could trim to 4) |
| 149 | test_invalid_predictors_rejected | D | KEEP | One predictor validator sweep |
| 188 | test_run_intercepts_and_scores_match_dummy_ols | A | KEEP | Dummy-intercept lstsq oracle for within_run and absolute; scoring offsets |
| 241 | test_default_centers_offsets_but_retains_slope_error | A | KEEP | Closed-form R2 (1, 15/16, -24) for slope errors under centering; excellent |
| 256 | test_run_shifts_leave_slopes_and_scores_invariant | C/B | KEEP | Shift equivariance of intercepts, invariance of slopes/scores, affine predictor shift |
| 293 | test_feature_invalidity_and_zero_sst_are_preserved | C | KEEP | NaN propagation and zero-SST policy |
| 309 | test_within_run_design_rejects_unidentifiable_slopes | D | KEEP | Within-run rank failures incl. between-run-only variance; important |
| 326 | test_single_training_run_or_single_row_run_can_fit | C | KEEP | Edge: one training run / one-row run still identifiable |
| 339 | test_unknown_encoding_mode_rejected | D | MERGE | Into 131 |

Summary: The best contract file in the group: three independent lstsq/closed-form oracles plus equivariance and identifiability contracts. Only two small merges.

## tests/test_within_run_cv_isolation.py (2)

| line | test | cat | verdict | reason |
|---|---|---|---|---|
| 13 | test_only_informative_run_cannot_supply_training_rank | B/C | KEEP | Subtle: rank only via held-out run must fail before any HRF fit; both regularizers |
| 35 | test_candidate_predictions_ignore_held_out_outcomes | B | KEEP | Canonical home for the triplicated isolation check; strong perturbation (random signals) |

Summary: Keep both and make this the single home for inner-fold isolation, deleting the copies in test_ridge_cv:156 and test_fractional_cv:149.

## tests/test_hrf_cv.py (13)

| line | test | cat | verdict | reason |
|---|---|---|---|---|
| 118 | test_statistics_shapes_and_profiled_columns | C | KEEP | Shapes plus real invariants: A = X'X, Q'X = 0, profiled lowers C only where present |
| 146 | test_task_model_loro_matches_stacked_nilearn_ols | A | KEEP | nilearn run_glm stacked OLS oracle, three batch sizes; the core HRF-CV oracle |
| 156 | test_task_only_model_reproduces_mean_stimulus_statistics | A | KEEP | Task-only path equals test_hrf_selection oracle_cv; cross-checks two oracles |
| 170 | test_noise_free_task_model_scores_one_only_with_the_full_model | S | KEEP | Noiseless: full model 1, others <1, task-only <1; weak but discriminating |
| 185 | test_singular_pooled_training_matrix_marks_candidate_ineligible | C | KEEP | Hand-built singular A -> -inf; pooled_amplitude semantics |
| 205 | test_run_level_rank_failures_are_candidate_specific | C | KEEP | Candidate-specific ineligibility with zeroed blocks |
| 226 | test_profiled_column_inside_nuisance_span_is_ineligible | C | MERGE | Into 205 (eligibility reasons family) |
| 242 | test_trial_eligibility_is_separate_from_task_model_eligibility | C | MERGE | Into 205; shapes only |
| 252 | test_cache_key_includes_task_model_and_modulator_values | C | KEEP | Cache identity depends on task model and RT values; a real cache-poisoning hazard |
| 272 | test_onsets_outside_supported_window_are_rejected | D | MERGE | Into test_hrf_design:84 (same -24s window rule) |
| 285 | test_signal_inside_profiled_span_does_not_raise | H | CUT | Assertion `isfinite | isnan` only excludes inf; no-raise guard duplicated by 324 |
| 306 | test_run_ineligible_candidate_scores_minus_inf | C | MERGE | Into 205 (same altered fixture, adds -inf score) |
| 324 | test_prediction_loss_tolerance_sums_absolute_regressor_cross_terms | C | KEEP | Roundoff tolerance semantics with cancelling cross terms; precise |

Summary: The LORO statistics engine is checked against nilearn `run_glm` on a stacked design with per-run nuisance blocks — a truly independent oracle. Four eligibility tests reuse the same fixture modification and should be one test. Test 285 is near-vacuous.

## tests/test_hrf_selection.py (18)

| line | test | cat | verdict | reason |
|---|---|---|---|---|
| 92 | test_pooled_loro_matches_stacked_training_ols | A | KEEP | lstsq stacked oracle; indices/cv_r2/canonical/delta; -1/NaN for constant features |
| 108 | test_noise_free_mean_response_and_rank_redundant_nuisance | C | KEEP | Rank-redundant nuisance (x1e4 duplicate) tolerated; noiseless recovery trivially 1 |
| 125 | test_outer_test_changes_cannot_select_the_hrf | B/A | KEEP | Test-run signals and RT perturbed; selection/amplitudes fixed; oracle on test_r2 |
| 153 | test_amplitude_shift_is_not_refitted_on_test_runs | A | KEEP | Closed form 1-16/9 proves no amplitude refit on test |
| 189 | test_invalid_folds_rejected | D | KEEP | One fold validator sweep (7 params; trim) |
| 197 | test_selection_requires_two_runs_and_valid_batch | D | KEEP | Two contracts in one test |
| 214 | test_canonical_only_negative_scores_and_stable_ties | C | KEEP | Negative R2 allowed; exact ties -> lowest index (kernel-length-only difference) |
| 239 | test_structurally_invalid_candidate_excluded_and_canonical_nan | C | KEEP | Canonical absorbed by nuisance -> NaN canonical/delta, never chosen; canonical-only raises |
| 258 | test_results_owned_metadata_independent_and_fingerprinted | F/E | KEEP | Fingerprint ignores RT/image, tracks onsets; readonly; one per result type |
| 321 | test_select_hrf_with_task_model_matches_task_model_oracle | A/E | KEEP | Task-model path vs nilearn oracle; drop the 7 literal provenance pins |
| 342 | test_default_selection_carries_task_only_model | E | CUT | Default TaskModel() and two literal lists |
| 354 | test_task_model_changes_selection_identity_and_rt_now_matters | C/E | KEEP | RT changes fingerprint only when modeled; behavioral |
| 372 | test_evaluate_split_with_task_model_returns_named_amplitude_rows | A | KEEP | stacked_oracle for multi-row amplitudes and test_r2 |
| 393 | test_default_evaluation_amplitudes_are_one_row_named_task | E | MERGE | Into 153 (one assert on shape/name) |
| 402 | test_task_model_argument_is_validated | D | KEEP | Type validation once |
| 414 | test_task_model_selection_feeds_single_trial_fits | C | MERGE | Into test_selected_hrf_fit:87; shape/finite only |
| 426 | test_evaluation_result_rejects_mismatched_amplitude_rows | D | MERGE | Into 402 |
| 438 | test_selection_provenance_states_per_regressor_roundoff_tolerance | E | CUT | Pins a formula string in provenance |

Summary: Strong oracle and isolation core (92, 125, 153, 321, 372). The task-model additions brought several provenance-string and shape tests that add little; four can be cut or merged.

## tests/test_hrf_design.py (6)

| line | test | cat | verdict | reason |
|---|---|---|---|---|
| 34 | test_default_single_trial_design_is_unchanged | C | KEEP | Default hrf == 'spm' == canonical candidate; cheap regression guard |
| 44 | test_custom_convolution_matches_independent_sampled_boxcars | A | MERGE | Re-derives nilearn's grid (semi-independent); 130 is the true nilearn oracle; keep duration-0 and sum identity asserts |
| 84 | test_custom_boundary_and_run_isolation | D/C | KEEP | -24s window boundaries exact; canonical home for onset-support rule |
| 101 | test_identified_hrf_fit_and_provenance | C/E | KEEP | Custom HRF changes betas; default == explicit canonical |
| 122 | test_unidentified_hrfs_are_rejected | D | KEEP | 'glover' and lambdas rejected |
| 130 | test_batched_trial_convolution_matches_nilearn_across_library | A | KEEP | Prefix-sum convolution vs compute_regressor across 6 library candidates, irregular times, edge onsets |

Summary: Test 130 is the irreplaceable one: it proves the custom prefix-sum/interp convolution reproduces nilearn to 1e-11 across the library, including onset at `times[0]-24`, negative onset, and irregular sampling. Test 44's oracle shares nilearn's `_sample_condition` grid formula so is only semi-independent.

## tests/test_hrf_library.py (7)

| line | test | cat | verdict | reason |
|---|---|---|---|---|
| 17 | test_expanded_library_includes_legacy_anchor | C/A | KEEP | 649 candidates, spm_hrf anchor, unit-sum, uniqueness, endpoints |
| 62 | test_custom_kernel_matches_notebook_and_independent_density | A | KEEP | Closed-form double-gamma density oracle plus notebook literals; fully independent |
| 84 | test_library_owns_parameters_tables_and_curves | F | KEEP | One ownership test for the library |
| 120 | test_invalid_or_duplicate_parameters_are_rejected | D | KEEP | One parameter validator sweep (9 params; adequate) |
| 126 | test_parameter_order_is_stable_and_empty_grid_is_canonical | C | KEEP | Fingerprint order-invariance; empty grid -> canonical only |
| 139 | test_invalid_sampling_rejected | D | MERGE | Into 120 as second axis |
| 148 | test_direct_spm_candidate_cannot_mislabel_fixed_kernel | D | MERGE | Into 120 |

Summary: Kernel math is independently verified by a closed-form gamma density; the rest are reasonable contracts. Minor validator consolidation only.

## tests/test_sobol_hrf_library.py (6)

| line | test | cat | verdict | reason |
|---|---|---|---|---|
| 21 | test_default_matches_approved_preview_and_canonical_anchor | C/E | KEEP | 513 rows, unit-sum, spm anchor; the 7 literal floats pin scipy Sobol scrambling (fragile across scipy) |
| 42 | test_sampling_spans_continuous_parameter_box | C | KEEP | Marginal stratification proves true Sobol base-2 design, not random |
| 56 | test_seed_controls_library_identity_without_affecting_global_rng | C | KEEP | Seed identity and no global RNG side effect |
| 67 | test_power_of_two_counts_and_numpy_integer_seed | C | MERGE | Into 56 |
| 74 | test_invalid_sample_count_is_rejected | D | KEEP | One validator sweep (8 params; trim) |
| 80 | test_invalid_seed_is_rejected | D | MERGE | Into 74 |

Summary: Fine as a factory test. The literal Sobol parameter vector in line 21 will break on any scipy QMC scrambling change without any boldtailor bug; consider relaxing to the stratification property.

## tests/test_task_design.py (12)

| line | test | cat | verdict | reason |
|---|---|---|---|---|
| 32 | test_expansion_orders_regressors_and_computes_amplitudes | A | KEEP | Hand-computed centered RT (-2..2), indicator, order, immutability |
| 55 | test_indicator_absent_when_nothing_is_missing | C | KEEP | Indicator column only when needed; design-shape semantics |
| 65 | test_task_only_model_expands_to_unit_task_rows | C | MERGE | Into 32 |
| 73 | test_uncentered_modulator_keeps_raw_values | C | MERGE | Into 32 (already asserted there via trial_type) |
| 82 | test_error_policy_rejects_missing_values | D | KEEP | Missing='error' policy names run and column |
| 89 | test_text_in_modulator_column_is_an_error_not_missing | D | KEEP | Strings are errors, not NaNs; real data hazard |
| 98 | test_all_missing_indicator_modulator_is_rejected | D | MERGE | Into 105 (degenerate modulator family) |
| 105 | test_regressor_without_nonzero_amplitude_is_rejected | D | KEEP | Zero regressor rejected with run/column in message |
| 114 | test_missing_modulator_column_or_timing_is_rejected | D | MERGE | Into 105 |
| 124 | test_task_columns_match_compute_regressor_per_condition | A | KEEP | compute_regressor oracle per condition with modulation; no stdout noise |
| 153 | test_task_columns_accept_string_hrfs_and_honor_settings | A | KEEP | Honors min_onset/oversampling against nilearn; glover differs |
| 171 | test_string_hrf_keeps_kernel_suffix_in_regressor_names | C | KEEP | Guards nilearn's name-mangling of `_kernel` suffix; obscure but real |

Summary: Expansion arithmetic is hand-verified and the convolution step is oracle-checked against nilearn with explicit settings. Four small validator/contract tests can be merged.

---

## Totals

| file | n | KEEP | MERGE | CUT |
|---|---|---|---|---|
| test_single_trial.py | 19 | 13 | 4 | 2 |
| test_single_trial_design.py | 9 | 6 | 3 | 0 |
| test_selected_hrf_fit.py | 5 | 4 | 1 | 0 |
| test_fractional_ridge.py | 9 | 5 | 3 | 1 |
| test_fractional_cv.py | 7 | 5 | 1 | 1 |
| test_fractional_ridge_ablation.py | 6 | 4 | 1 | 1 |
| test_fractional_ablation_simulation.py | 5 | 3 | 1 | 1 |
| test_ridge_cv.py | 9 | 6 | 2 | 1 |
| test_ridge_selection.py | 7 | 5 | 2 | 0 |
| test_ridge_objective_simulation.py | 8 | 5 | 2 | 1 |
| test_trial_encoding.py | 13 | 11 | 2 | 0 |
| test_within_run_cv_isolation.py | 2 | 2 | 0 | 0 |
| test_hrf_cv.py | 13 | 7 | 5 | 1 |
| test_hrf_selection.py | 18 | 12 | 4 | 2 |
| test_hrf_design.py | 6 | 5 | 1 | 0 |
| test_hrf_library.py | 7 | 5 | 2 | 0 |
| test_sobol_hrf_library.py | 6 | 4 | 2 | 0 |
| test_task_design.py | 12 | 8 | 4 | 0 |
| **Total** | **161** | **110** | **40** | **11** |

## Irreplaceable tests

- test_single_trial.py:109 `test_beta_path_matches_augmented_ols`, :194 `..._match_independent_augmented_ols`, :388 `..._small_hrf_support`
- test_selected_hrf_fit.py:15 `test_grouped_betas_and_r_squared_match_augmented_ols`
- test_fractional_ridge.py:35 `test_fraction_path_matches_requested_norm_and_oracle`, :52 `..._target_scaling_and_unpenalized_confounds`
- test_fractional_cv.py:122 `test_fraction_cv_matches_nested_fixed_ols_oracle`, :173 `test_fixed_targets_do_not_depend_on_fraction_grid`
- test_ridge_cv.py:108 `test_cv_matches_independent_augmented_ols_and_encoding`, :145 `test_validation_targets_use_candidate_penalty`
- test_within_run_cv_isolation.py:13, :35 (both)
- test_trial_encoding.py:188 `..._match_dummy_ols`, :241 `test_default_centers_offsets_but_retains_slope_error`, :309 `..._rejects_unidentifiable_slopes`
- test_hrf_cv.py:146 `test_task_model_loro_matches_stacked_nilearn_ols`, :324 `test_prediction_loss_tolerance...`
- test_hrf_selection.py:92 `test_pooled_loro_matches_stacked_training_ols`, :125 `test_outer_test_changes_cannot_select_the_hrf`, :153 `test_amplitude_shift_is_not_refitted_on_test_runs`
- test_hrf_design.py:130 `test_batched_trial_convolution_matches_nilearn_across_library`
- test_hrf_library.py:62 `test_custom_kernel_matches_notebook_and_independent_density`
- test_task_design.py:32, :124
- test_ridge_objective_simulation.py:68 `test_current_alpha_score_matches_matched_filter_limit` (package test; relocate)

## Non-independent or semi-independent oracles

- All augmented-lstsq oracles for alpha-ridge (test_single_trial:109/194, test_selected_hrf_fit:15, test_ridge_cv `augmented_beta`) hard-code the implementation's penalty scaling `sqrt(alpha)*||X_perp||`; the *solve* is independent, the *normalization convention* is pinned only by the hand-computed test_single_trial:95.
- test_hrf_design:44 rebuilds nilearn's oversampled grid with the same formula the implementation inherits from `_sample_condition`; test_hrf_design:130 (direct `compute_regressor`) is the independent one.
- test_hrf_cv:156 cross-checks two in-suite oracles (`oracle_cv` vs `loro_scores`) — useful but not external.
- test_sobol_hrf_library:21 pins scipy's scrambled Sobol output literally (external-library pin, not a boldtailor property).

## Wrong / vacuous tests

- test_fractional_ridge.py:211 monkeypatches `_fractional_ridge.freeze_fraction_result` with `raising=False`; the symbol does not exist in `src/`, so the "numerical module must not mutate" guard can never fire.
- test_hrf_cv.py:285 asserts `isfinite | isnan`, which only excludes +/-inf; the intended "does not raise" property is implicit.
- test_fractional_ablation_simulation.py:61 asserts `selected_fraction == 1` with a one-element grid `(1.0,)`; :122 asserts the paired baseline's RMSE difference is 0 against itself.
- test_ridge_selection.py:19 cannot distinguish percentile interpolation methods (90th pct of [0,0,0,0,1,1] is 1 under linear/lower/higher).
- test_single_trial.py:337 claims AR noise but uses sd 0.01 against unit amplitudes (SNR ~100); the correlation > 0.98 bar is nearly noiseless.

## Missing tests

1. Fraction recovery under noise: a simulated problem where the truth-optimal fraction is clearly < 1 and `score_fraction_candidates` + `select_ridge_fractions` pick it (and pick 1.0 when noise is absent). Currently only noiseless/oracle-matching.
2. Alpha recovery under noise for `select_ridge_penalty` end-to-end on `score_ridge_candidates` output.
3. HRF recovery under realistic noise: cv_fixture noise sd 0.015 vs amplitude ~3. Add a test with noise comparable to signal and AR(1) rho ~0.3-0.5 asserting the true candidate still wins for most features and `delta_cv_r2 > 0`.
4. Convolution oracle at a second TR (e.g. 0.8 s and 2.0 s) and with oversampling != 50; everything in this group uses TR 1.6.
5. Percentile semantics: a population with 6+ distinct values where `linear` vs `lower`/`nearest` give different objectives.
6. Fractional ridge with duplicated/collinear trials: fraction path when two trial columns are near-identical (cond ~1e8 is tested for 2 columns with a synthetic signal but not through `fit_single_trials`).
7. Multi-run AR(1) oracle: no AR noise model exists for single-trial/HRF paths; if `_conventional.py`'s `noise_model='ar1'` is user-facing, a nilearn `run_glm(noise_model='ar1')` oracle across runs is absent in this group (likely belongs to the conventional-GLM group).
8. Within-run encoding with unequal trial counts per run where a run has exactly one complete trial (intercept consumes it) — check the slope fit ignores that run's contribution correctly.
9. `fit_selected_hrfs` on a subset of runs with a feature whose selected HRF is `-1` but is non-constant in the new run (should remain NaN, not refit).

### Audit: examples/NSD/test_*.py (21 files, 148 test functions, 4,879 lines)

No `conftest.py` exists in `examples/` or `examples/NSD/`; fixtures (`confounds`, `events`, `dataset`,
`six_run_dataset`, `cv_library`, `hrf_nsd`, `mini_nsd`, `four_runs`, `saved_sessions`) are re-exported
by importing sibling test modules (`from examples.NSD.test_nsd_cifti import ...`), so 9 test files
import other test files.

Categories: A oracle, B leakage/isolation, C contract, D validation, E provenance pin, N notebook,
P plotting, H vacuous/redundant. Verdicts: KEEP / MERGE / CUT.

---

## test_beta_activation.py (5 tests, 92 lines)

| line | test | cat | verdict | reason |
|---|---|---|---|---|
| 17 | test_trial_weighted_t_test_matches_scipy_and_preserves_inputs | A | KEEP | ttest_1samp oracle, pooled across unequal runs, inputs unmutated |
| 35 | test_missing_values_and_degenerate_columns_are_explicit | A | KEEP | NaN/inf/zero-variance policy with scipy oracle on finite column |
| 55 | test_large_offsets_use_centered_sample_variance | A | KEEP | catches E[x^2]-E[x]^2 cancellation; cheap and real |
| 63 | test_invalid_beta_dimensions_are_rejected | D | KEEP | one parametrized validator, three shapes |
| 68 | test_activation_exports_preserve_map_names_axis_and_values | C | KEEP | CIFTI axis + map names round trip; uses private `_activation_artifacts` |

Summary: Small, fast, oracle-grade file. Everything is worth keeping. Only concern is reaching into
`workflow_outputs._activation_artifacts`; a package move should expose a public artifact builder.

## test_fractional_notebook.py (2 tests, 87 lines)

| line | test | cat | verdict | reason |
|---|---|---|---|---|
| 16 | test_explicit_fractional_mode_uses_fraction_grid | N | MERGE | cell-exec of settings cell; add as a row to ridge_outputs:40 param table |
| 24 | test_notebook_default_executes_fractional_cv | N | MERGE | 4th full nsd_workflow kernel run; fold into the single smoke run (nsd_workflow:286) |

Summary: Both tests duplicate coverage. The fingerprint linkage and metadata strings asserted here
are already checked without a kernel in test_fractional_workflow:33. The fractional_cv default should
be the config of the one retained end-to-end smoke test, not a separate 8 s execution.

## test_fractional_workflow.py (4 tests / 7 cases, 234 lines)

| line | test | cat | verdict | reason |
|---|---|---|---|---|
| 33 | test_fraction_workflow_matches_whole_array_and_exports[2] | A | KEEP | per-scope oracle vs package selectors, outer OLS targets, export round trip; trim xlabel/metadata asserts |
| 168 | test_fraction_blocks_workers_and_missing_rt_preserve_results | C | KEEP | block/worker invariance plus missing-RT mask; merge with ridge_workflow:156 as one parametrized test |
| 196 | test_outer_fraction_choices_and_training_coefficients_are_isolated[2] | B | KEEP | test-half BOLD and RT scrambled; training choices unchanged |
| 231 | test_fraction_and_alpha_grids_are_mutually_exclusive | D | MERGE | add as a case to ridge_workflow:225 preflight parametrization |

Summary: The oracle test is long (130 lines) but irreplaceable: it is the only place the fractional
tuning, outer evaluation, final refit and exports are tied to independent package calls. Lines
159-165 (metadata strings, tuning_figure xlabel) are E/P noise inside an A test.

## test_hrf_reliability.py (3 tests / 8 cases, 63 lines)

| line | test | cat | verdict | reason |
|---|---|---|---|---|
| 26 | test_curve_correlations_match_pearson_over_full_common_time_grid | A | KEEP | np.corrcoef oracle incl. NaN/negative ids and padded grid |
| 57 | test_invalid_selections_are_not_silently_reinterpreted[6] | D | KEEP | one validator; could trim 6 params to 3 |
| 62 | test_empty_selections_preserve_three_map_shape | C | MERGE | one-line shape check; append to line 26 |

Summary: Clean oracle file for a function that is a package-grade candidate. Nothing lost by folding
the empty-input shape check into the oracle test.

## test_session_hrf_reliability.py (3 tests / 9 cases, 94 lines)

| line | test | cat | verdict | reason |
|---|---|---|---|---|
| 26 | test_all_pairs_and_canonical_baselines_match_direct_curve_correlations | A | KEEP | pairwise/baseline/delta/summary vs corrcoef; row-count summary hand-checked |
| 61 | test_parameter_variability_uses_values_not_categorical_ids | A | KEEP | pandas std(ddof=1) oracle; catches id-as-value bug |
| 92 | test_invalid_session_selections_are_rejected[7] | D | KEEP | one validator; 7 params is generous, 4 suffice |

Summary: All oracle/validation; keep intact. `compare_hrfs` is library-grade and these tests would
move with it unchanged except the import path.

## test_rt_diagnostics.py (6 tests, 90 lines)

| line | test | cat | verdict | reason |
|---|---|---|---|---|
| 14 | test_correlation_removes_run_mean_confounding | A | KEEP | hand-built data where pooled r=+1 but within-run r=-1 |
| 24 | test_matched_feature_masks_exclude_only_invalid_rt_or_beta | C | KEEP | mask semantics (NaN beta, NaN/0/negative RT) and counts |
| 43 | test_constant_rt_and_constant_beta_are_undefined | C | MERGE | same undefined-value policy; add rows to line 24 |
| 51 | test_select_cortex_stable_ties_and_no_even_run_leakage | B | KEEP | odd-run vertex choice unchanged when even data scrambled |
| 66 | test_scatter_points_are_even_only_and_centered_with_matching_mask | C | KEEP | even_run_points hand values; drop the PNG-magic-bytes asserts (P) |
| 83 | test_mismatched_trials_and_ambiguous_run_numbers_fail | D | KEEP | one validator, three bad inputs |

Summary: Good unit file for a library-grade module. Only consolidation is 43 into 24; the PNG header
assertions in 66 test matplotlib, not the example.

## test_multisession_analysis.py (9 tests / 10 cases, 284 lines)

| line | test | cat | verdict | reason |
|---|---|---|---|---|
| 116 | test_loading_and_hrf_correlations_use_full_curves | A | KEEP | loader + analysis vs corrcoef and hand-set fixture differences |
| 139 | test_paired_summary_uses_same_sessions_for_both_models | A | KEEP | hand-computed paired means/sd/fractions with NaN pairing |
| 151 | test_mean_peak_time_averages_session_peaks_and_exports_counts | A | KEEP | peak means hand-computed (12.7/3), export axis and counts |
| 176 | test_glm_effect_means_match_sessions_and_export_units | A | KEEP | hand-computed GLM means with injected NaN; drop glm_units string pin |
| 212 | test_loader_rejects_misaligned_grayordinates | D | KEEP | anchor for one parametrized loader-validation test |
| 233 | test_loader_rejects_wrong_scalar_names[2] | D | MERGE | into 212 |
| 250 | test_loader_rejects_incompatible_libraries_and_missing_results | D | MERGE | into 212 |
| 264 | test_at_least_two_distinct_sessions_are_required | D | MERGE | into 212 |
| 271 | test_loader_rejects_different_global_alpha_selection_percentiles | D | MERGE | into 212; settings-compat check, one case |

Summary: Four strong oracle tests on a synthetic saved-session fixture, then five separate functions
for one loader's validation branches. Collapse the five into a single parametrized test that mutates
the fixture and asserts the match string; no coverage is lost.

## test_multisession_workflow.py (7 tests, 233 lines)

| line | test | cat | verdict | reason |
|---|---|---|---|---|
| 33 | test_missing_sessions_fit_then_reuse_without_fitting | N/B | KEEP | runs nsd_workflow kernel twice (~11 s); reuse-without-refit + hrf_seed conflict is irreplaceable |
| 69 | test_read_only_mode_reports_missing_sessions_without_fitting | D | KEEP | cheap, fit_missing=False contract |
| 79 | test_missing_rt_sessions_fit_glms_and_retain_all_trial_betas | N | MERGE | two more kernel runs for a policy unit-tested 4x elsewhere; put NaN RT in fixture of line 33 |
| 129 | test_multisession_notebook_executes_and_exports_paired_maps | N | KEEP | the one smoke run of nsd_multisession.ipynb on saved fixtures |
| 172 | test_multisession_export_preserves_input_files | C | KEEP | idempotent re-publication, inputs untouched |
| 184 | test_incompatible_estimators_fail_before_fitting | D | KEEP | fails before `_execute_workflow` |
| 205 | test_complete_sessions_are_validated_before_missing_session_fits | C | KEEP | ordering contract: validate completed before expensive fits |

Summary: Hidden cost: `ensure_session_outputs` executes the full single-session notebook via nbclient
per missing session, so lines 33 and 79 are four kernel runs that do not look like notebook tests.
Keep 33 (reuse semantics) and 129 (one smoke per notebook); fold 79 into 33.

## test_notebook_helpers.py (10 tests / 13 cases, 181 lines)

| line | test | cat | verdict | reason |
|---|---|---|---|---|
| 41 | test_paths_require_explicit_data_location | D | KEEP | anchor for notebook_paths validation |
| 47 | test_paths_derive_from_root_without_creating_directories | C | KEEP | derived defaults, no side effects |
| 61 | test_explicit_paths_override_environment | C | MERGE | into 47 |
| 77 | test_empty_root_gets_actionable_error | D | MERGE | into 41 |
| 82 | test_library_plot_preserves_curves_and_peak_colors | P | CUT | asserts plotted arrays equal inputs; matplotlib internals |
| 96 | test_design_plot_retains_original_acquisition_times | P | CUT | same; passes for any plot of the data |
| 105 | test_glm_comparison_uses_paired_values_and_signed_difference | C | KEEP | paired n/median table is real; drop the collections/patches/xlim asserts |
| 126 | test_parameter_agreement_excludes_unpaired_and_constant_values[2] | A | KEEP | Pearson over paired, NaN for constant parameter; drop `len(fig.axes)` |
| 146 | test_curve_agreement_uses_same_grayordinates_for_every_comparison[2] | A | KEEP | hand-computed median/q25, inputs unmutated; drop the "No paired HRFs" text pin |
| 163 | test_session_figures_preserve_pair_means_and_undefined_sessions[2] | P | CUT | compare_hrfs is oracle-tested; imshow array/patch heights add nothing |

Summary: Path helpers and the three table-producing plot helpers carry value through their returned
DataFrames. Pure figure-property tests (82, 96, 163) and the figure asserts inside 105/126 are
matplotlib tests and should go.

## test_nsd_cifti.py (11 tests / 13 cases, 252 lines) -- legacy CLI stack

| line | test | cat | verdict | reason |
|---|---|---|---|---|
| 69 | test_confounds_use_24_motion_top_six_combined_components_and_cosines | C | KEEP | confound selection rule (combined mask, top-6, fill only leading NaN) |
| 79 | test_only_initial_motion_derivative_nans_are_filled[2] | D | KEEP | anchor for select_confounds validation |
| 86 | test_fewer_than_six_retained_combined_components_is_an_error | D | MERGE | into 79 |
| 94 | test_rt_modulation_is_centered_and_preserves_stimulus_timing | A | KEEP | compute_regressor oracle; legacy two-column design (stimulus, RT) |
| 112 | test_missing_response_time_is_rejected_explicitly | D | MERGE | legacy policy (reject) differs from workflow (indicator); one param in 79 |
| 181 | test_complete_example_publishes_correct_pooled_maps_and_designs | A | KEEP | lstsq oracle for pooled R2 through full CLI; legacy |
| 213 | test_discovery_requires_every_run_to_have_cifti | D | KEEP | anchor for discover_runs/run preflight |
| 220 | test_inconsistent_grayordinate_order_is_rejected | D | MERGE | into 213 |
| 231 | test_confounds_length_must_match_cifti | D | MERGE | into 213 |
| 240 | test_relative_fmriprep_override_works | C | MERGE | weak (counts 3 files); fold into 249 as fmriprep-root resolution test |
| 249 | test_external_fmriprep_root_has_explicit_provenance_error | D | MERGE | with 240 |

Summary: `nsd_cifti.run_analysis`/`task_regressors` are used only by the legacy CLI; the review already
flags two parallel stacks. The `dataset`/`confounds`/`events` fixtures here are imported by 9 other
files, so this module cannot be deleted without a conftest. If the legacy stack is retired, 94 and
181 go with it; `select_confounds` and `discover_runs` tests (69, 79, 213) survive.

## test_nsd_hrf_selection.py (9 tests / 10 cases, 461 lines)

| line | test | cat | verdict | reason |
|---|---|---|---|---|
| 83 | test_expanded_artifacts_reconstruct_grouped_fits | A | KEEP | OLS+ridge lstsq oracle from saved NPZ designs; trim E pins (TrainingRuns, "descriptive") |
| 188 | test_blocks_and_even_run_edits_preserve_training_decisions | B | KEEP | block invariance and even-run edits leave odd selection/vertices unchanged |
| 227 | test_hrf_comparison_uses_pooled_matching_canonical_trial_fit | A | KEEP | canonical pooled-R2 oracle for hrfdeltarsquared |
| 294 | test_too_few_odd_runs_records_unavailable_evaluation | C | KEEP | graceful NaN + reason with two runs |
| 309 | test_collision_and_publication_rollback | C | MERGE | collision: into single run_single_trial_analysis collision test (single_trial:147); rollback duplicates single_trial:185 |
| 345 | test_undefined_odd_hrf_at_rt_selected_vertex_does_not_abort | C | KEEP | robustness; note the conditional assert weakens it |
| 366 | test_canonical_ineligible_diagnostic_does_not_abort_expanded_outputs | C | KEEP | canonical-ineligible path; uses private `_load_runs`, `compile_trial_run` |
| 422 | test_comparison_artifact_preserves_signed_difference_and_undefined | H | MERGE | 4-value subtraction already covered by 227's sign/NaN checks |
| 446 | test_comparison_collisions_precede_fitting[2] | C | MERGE | same precheck as 309; one parametrized collision test |

Summary: Two of the best oracle tests in the suite (83, 227) live here. Collision checks for
`run_single_trial_analysis` appear in four files (this, split_hrf:126, single_trial:147, and here
twice); they exercise one preflight and should be one parametrized test.

## test_nsd_parallel.py (5 tests / 8 cases, 175 lines)

| line | test | cat | verdict | reason |
|---|---|---|---|---|
| 38 | test_process_batches_are_bounded_ordered_and_thread_limited | C | KEEP | map_blocks: ordered yield, bounded concurrency, OMP=1 in workers |
| 79 | test_parallel_canonical_cli_matches_serial | C | KEEP | only test of the argparse CLI path; n_jobs equivalence |
| 124 | test_parallel_expanded_matches_serial_artifacts | C | MERGE | expanded invariance already in hrf_selection:188; vary n_jobs there |
| 148 | test_parallel_worker_failure_publishes_nothing | C | KEEP | worker exception propagates, output dir untouched |
| 173 | test_invalid_worker_count_rejected_before_input_discovery[4] | D | KEEP | one validator; trim to 2 params |

Summary: `parallel_blocks.map_blocks` is library-grade and 38 is its only direct test. The
serial-vs-parallel artifact comparisons are duplicated across parallel/hrf_selection/single_trial
files; one per entrypoint suffices.

## test_nsd_single_trial.py (5 tests / 6 cases, 226 lines)

| line | test | cat | verdict | reason |
|---|---|---|---|---|
| 50 | test_cifti_betas_and_pooled_maps_match_independent_fits | A | KEEP | OLS/ridge lstsq oracle, rtcount axis, existing files preserved |
| 147 | test_collision_stops_before_fitting | C | KEEP | anchor: one parametrized collision test for run_single_trial_analysis |
| 168 | test_incomplete_or_misaligned_runs_rejected[2] | H | MERGE | identical to nsd_cifti:213/220 (same discover_runs) |
| 185 | test_failed_publication_leaves_no_partial_single_trial_set | C | KEEP | one example-level rollback test (drop the copy in hrf_selection:309) |
| 208 | test_block_size_does_not_change_maps_or_trial_identity | C | MERGE | into parallel:79 (block 1 serial vs parallel already compares) |

Summary: The oracle test is irreplaceable; the rest duplicate preflight/collision/invariance
checks found in sibling files.

## test_nsd_split_hrf.py (4 tests / 8 cases, 143 lines)

| line | test | cat | verdict | reason |
|---|---|---|---|---|
| 42 | test_split_parameters_match_each_halfs_library_winners | A/B | KEEP | planted HRFs recovered per half; fold train/test disjoint within half |
| 88 | test_split_selection_cannot_use_other_halfs_signals[2] | B | KEEP | editing one half leaves the other half's maps bit-identical |
| 107 | test_insufficient_half_has_nan_maps_and_explicit_reason | C | KEEP | three-run session: even half NaN with reason |
| 126 | test_new_map_collision_is_checked_before_fitting[3] | C | MERGE | three params of the same preflight; into single_trial:147 |

Summary: Best leakage coverage in the suite (42, 88). Only the collision parametrization is
redundant.

## test_nsd_workflow.py (21 tests / 31 cases, 599 lines)

| line | test | cat | verdict | reason |
|---|---|---|---|---|
| 55 | test_nsd_task_model_centers_rt_and_keeps_trial_type_uncentered | C | KEEP | defines NSD_TASK_MODEL; expand_events amplitudes checked |
| 82 | test_invalid_glm_covariates_fail_explicitly[2] | D | MERGE | one validate_glm_events param test with 107, 113 |
| 89 | test_nonpositive_rt_becomes_missing_with_indicator[5] | C | KEEP | RT missing policy (NaN/inf/0/negative -> indicator); trim to 3 params |
| 107 | test_glm_requires_observed_rt_to_estimate_rt_effect | D | MERGE | into 82 |
| 113 | test_glm_does_not_treat_malformed_rt_text_as_missing | D | MERGE | into 82 |
| 120 | test_trimming_keeps_acquisition_times_and_matches_confounds | C | KEEP | nonsteady trimming keeps frame times/onsets aligned; irreplaceable |
| 140 | test_interior_nonsteady_flag_is_rejected | D | KEEP | contiguity rule |
| 194 | test_both_glms_match_independent_ols_and_keep_spatial_order[4] | A | KEEP | independent OLS oracle for canonical+optimized GLM; drop stdout asserts; 2 params suffice |
| 238 | test_glm_model_and_selection_share_the_nsd_task_model | C | MERGE | compares example design to the package function it calls; numeric check already in 194 |
| 264 | test_quiet_glm_still_emits_design_warnings | H | CUT | stdout hygiene; warning passthrough is Nilearn's behavior |
| 286 | test_notebook_executes_full_workflow_and_exports_reusable_artifacts[2] | N | KEEP | the one nsd_workflow smoke run; keep 1 param, move string pins (386-427) to a `_metadata` unit test |
| 466 | test_notebook_default_preview_uses_approved_sobol_library | N | MERGE | cell exec by id `de5dc917`/`f3d49ae3`; extract settings resolution to a function and test that |
| 476 | test_notebook_preserves_expanded_configured_paths | N | CUT | expanduser via cell exec; notebook_paths tests cover path handling |
| 481 | test_notebook_preview_honors_sobol_settings | N | MERGE | into 466 |
| 489 | test_notebook_preview_can_reproduce_grid_or_use_custom_rows | N | MERGE | into 466 |
| 499 | test_notebook_preview_rejects_unknown_library | N | MERGE | into 466 |
| 504 | test_rerunning_beta_cell_uses_current_settings | N | CUT | exec of a cell found by substring; "rerun a cell" is not a library behavior |
| 536 | test_beta_series_progress_is_brief_across_blocks | H | CUT | counts printed progress lines |
| 551 | test_selection_task_model_switch_drops_only_rt | C | KEEP | selection_task_model(False) is a subset; bool validation |
| 564 | test_select_hrfs_without_rt_still_feeds_the_full_glm | C | KEEP | RT-free selection still yields RT column in GLM designs |
| 582 | test_metadata_and_reuse_follow_the_rt_selection_switch | E | MERGE | metadata string pins + validate_saved_settings; join workflow_reuse:119 and a `_metadata` test |

Summary: The core contracts (55, 89, 120, 194, 551, 564) are strong. Eleven of 21 functions touch the
notebook: one real smoke run plus ten `exec()`-by-id/substring tests whose logic (library/ridge
settings resolution) lives in the notebook settings cell and should be a tested function. The
smoke test carries 40 lines of exact metadata strings that belong in a cheap `_metadata` unit test.

## test_ridge_outputs.py (4 tests / 11 cases, 180 lines)

| line | test | cat | verdict | reason |
|---|---|---|---|---|
| 40 | test_notebook_resolves_new_defaults_and_legacy_options[6] | N | MERGE | cell exec; keep the 6-row table but run it against an extracted `resolve_settings` |
| 46 | test_notebook_rejects_unknown_ridge_mode | N | MERGE | into 40 |
| 51 | test_ridge_artifacts_match_numeric_results | C | KEEP | cv_r2/scores/folds/provenance export round trip; republish collision |
| 126 | test_notebook_executes_ridge_modes[3] | N | MERGE | three more kernel runs; mode-specific asserts are metadata pins testable on fit_cv_beta_series |

Summary: One genuine export-contract test. The other three re-execute the notebook (or its settings
cell) to read back configuration; all would be unit tests once settings resolution leaves the cell.

## test_ridge_workflow.py (7 tests / 13 cases, 313 lines)

| line | test | cat | verdict | reason |
|---|---|---|---|---|
| 76 | test_workflow_matches_whole_array_reference[2] | A | KEEP | per-scope scores, final fit, outer evaluation vs package calls |
| 156 | test_block_size_and_parallelism_preserve_global_choice | C | MERGE | parametrize with fractional_workflow:168 (same function, different grid) |
| 173 | test_outer_data_cannot_change_training_choices[2] | B | KEEP | scrambled test-half leaves training alpha/coefficients unchanged |
| 202 | test_missing_rt_keeps_beta_rows_and_requested_axis | C | KEEP | trial masks and NaN beyond max_grayordinates |
| 225 | test_workflow_preflight_rejects_invalid_inputs[6] | D | KEEP | anchor; absorb fractional:231 and within_run:143 |
| 253 | test_final_provenance_identifies_the_tuning_decision | C | KEEP | fingerprint stable across reruns, changes with settings/predictors |
| 301 | test_rt_switch_reaches_every_ridge_selection | C | KEEP | narrow task model propagates to final and outer selections |

Summary: Strong file; only the block/parallel invariance duplicates the fractional version.

## test_session_hrf.py (13 tests / 16 cases, 406 lines)

| line | test | cat | verdict | reason |
|---|---|---|---|---|
| 61 | test_session_selection_matches_public_api_and_reuses_without_fitting | A/B | KEEP | select_hrf oracle + cache reuse forbids fit_session |
| 93 | test_source_library_and_coverage_changes_require_new_estimates | B | KEEP | cache invalidation on events, coverage, library |
| 114 | test_damaged_cache_is_recomputed_instead_of_silently_reused | C | KEEP | anchor for damaged/malformed cache |
| 130 | test_malformed_cache_manifest_is_recomputed[2] | C | MERGE | into 114 |
| 174 | test_matching_full_workflow_estimates_are_reused | B | KEEP | import from full-workflow export without fitting |
| 202 | test_partial_full_workflow_exports_are_refitted[2] | C | KEEP | anchor for incomplete/damaged export -> refit |
| 231 | test_damaged_full_workflow_exports_are_refitted[2] | C | MERGE | into 202 |
| 257 | test_hrf_only_analysis_retains_trials_with_missing_reaction_times | C | KEEP | hrf_only loader keeps NaN-RT trials, same events as GLM loader |
| 278 | test_hrf_only_analysis_requires_two_runs_without_odd_even_splits | H | MERGE | asserts only finiteness; make two-run `dataset` a param of 257 |
| 288 | test_incompatible_axes_or_missing_sessions_fail_before_fitting | D | KEEP | fails before fit_session |
| 319 | test_notebook_fits_three_sessions_exports_comparisons_and_resumes | N | KEEP | one smoke run per notebook; drop the second execute() (resume covered by 61) |
| 376 | test_rt_switch_changes_request_identity_and_cache_metadata | B/C | KEEP | request id and task_model regressors change with include_rt |
| 392 | test_full_workflow_export_with_rt_is_not_imported_when_rt_is_off | B | KEEP | RT-based export refused for RT-off request |

Summary: Cache/reuse semantics are well covered and mostly non-redundant. Four refit-on-damage
tests can be two parametrized ones. The notebook test executes the kernel twice; the rerun half
duplicates line 61.

## test_within_run_encoding.py (2 tests / 5 cases, 152 lines)

| line | test | cat | verdict | reason |
|---|---|---|---|---|
| 20 | test_outer_oracle_blocks_and_export[4] | A | KEEP | lstsq oracle for coefficients/intercepts/offsets/SSE in both modes; loss NPZ round trip |
| 143 | test_workflow_rejects_unknown_encoding_mode | D | MERGE | into ridge_workflow:225 |

Summary: The oracle is independent of `evaluate_trial_encoding` (raw lstsq), so it complements
ridge_workflow:76 rather than duplicating it.

## test_workflow_reuse.py (4 tests / 6 cases, 127 lines)

| line | test | cat | verdict | reason |
|---|---|---|---|---|
| 35 | test_notebook_reuses_saved_results_without_fitting[3] | N/B | MERGE | 7 in-process full-notebook execs; keep one mode (fractional_cv) incl. the overwrite check |
| 93 | test_output_policy_allows_overwrite_and_detects_reuse | D | KEEP | anchor for check_output |
| 109 | test_reuse_rejects_partial_outputs | D | MERGE | into 93 |
| 119 | test_reuse_rejects_changed_analysis_settings | D | KEEP | validate_saved_settings; absorb nsd_workflow:582's reuse half |

Summary: The reuse-without-refit contract matters, but three modes x 2-3 executions each is the
most expensive in-process test in the suite for one code path (`reused_results`).

## test_workflow_surfaces.py (14 tests / 18 cases, 487 lines)

| line | test | cat | verdict | reason |
|---|---|---|---|---|
| 68 | test_surface_values_use_vertex_ids_and_exclude_volume | A | KEEP | hand-computed vertex mapping; volume excluded; length validated |
| 81 | test_multisession_hrf_surface_shows_matched_delta | N/P | CUT | substring cell exec; ylim/label/bbox asserts |
| 138 | test_discovery_uses_fslr_meshes_and_supports_explicit_paths | C | KEEP | mesh discovery, explicit override, ambiguity error |
| 155 | test_multisession_glm_cell_renders_absolute_and_delta_maps | N/P | CUT | cell-id exec; facecolor-vs-colorbar bin matching |
| 219 | test_surface_figure_keeps_signed_shared_scale_and_missing_data | P | MERGE | extract shared-limit computation to a function; test limits, not figures |
| 239 | test_empty_cortex_renders_without_warnings_and_preserves_negative_r2 | P | MERGE | into the same limits function test |
| 258 | test_wrong_surface_density_is_rejected | D | KEEP | vertex-count validation |
| 273 | test_peak_time_surface_uses_sequential_seconds_scale[3] | N/P | CUT | cell-id exec; cmap name "viridis" |
| 311 | test_delta_r2_surface_labels_task_contribution | P | CUT | ylabel text |
| 324 | test_delta_r2_heat_colors_emphasize_small_values | P | CUT | colormap/norm internals |
| 344 | test_beta_surface_cell_plots_full_minus_confound_r2 | N/H | CUT | cell exec with a capture mock; re-asserts the mock's inputs |
| 377 | test_activation_surface_uses_fixed_t_scale[2] | P | MERGE | into limits function test |
| 392 | test_notebook_surface_cells_render_existing_results_and_register_exports | N | CUT | three cell ids exec; PNG size > 1000 |
| 438 | test_inline_notebook_displays_each_surface_figure_once | N | MERGE | one legitimate concern (double display); check image/png count in the smoke run instead |

Summary: 487 lines, of which only 68, 138 and 258 test behavior. Seven tests exec notebook cells by
id or substring and assert figure properties; four more assert matplotlib state directly. The real
logic worth testing (signed shared limits, volume exclusion, fixed t range) is a pure function away.

---

## Totals

| file | n | KEEP | MERGE | CUT |
|---|---|---|---|---|
| test_beta_activation.py | 5 | 5 | 0 | 0 |
| test_fractional_notebook.py | 2 | 0 | 2 | 0 |
| test_fractional_workflow.py | 4 | 3 | 1 | 0 |
| test_hrf_reliability.py | 3 | 2 | 1 | 0 |
| test_session_hrf_reliability.py | 3 | 3 | 0 | 0 |
| test_rt_diagnostics.py | 6 | 5 | 1 | 0 |
| test_multisession_analysis.py | 9 | 5 | 4 | 0 |
| test_multisession_workflow.py | 7 | 6 | 1 | 0 |
| test_notebook_helpers.py | 10 | 5 | 2 | 3 |
| test_nsd_cifti.py | 11 | 5 | 6 | 0 |
| test_nsd_hrf_selection.py | 9 | 6 | 3 | 0 |
| test_nsd_parallel.py | 5 | 4 | 1 | 0 |
| test_nsd_single_trial.py | 5 | 3 | 2 | 0 |
| test_nsd_split_hrf.py | 4 | 3 | 1 | 0 |
| test_nsd_workflow.py | 21 | 8 | 9 | 4 |
| test_ridge_outputs.py | 4 | 1 | 3 | 0 |
| test_ridge_workflow.py | 7 | 6 | 1 | 0 |
| test_session_hrf.py | 13 | 9 | 4 | 0 |
| test_within_run_encoding.py | 2 | 1 | 1 | 0 |
| test_workflow_reuse.py | 4 | 2 | 2 | 0 |
| test_workflow_surfaces.py | 14 | 3 | 4 | 7 |
| **total** | **148** | **85** | **49** | **14** |

MERGE collapses ~49 functions into roughly 15 parametrized tests, so the post-audit suite is about
100 functions and loses no numerical, isolation, or I/O contract.

## Irreplaceable tests

- test_beta_activation.py:17, 35, 55 (ttest_1samp oracles incl. cancellation)
- test_fractional_workflow.py:33 (fractional tuning/outer/final/export oracle), 196 (outer isolation)
- test_hrf_reliability.py:26; test_session_hrf_reliability.py:26, 61 (corrcoef/std oracles)
- test_rt_diagnostics.py:14 (run-mean confounding), 51 (odd-only vertex selection)
- test_multisession_analysis.py:116, 139, 151, 176 (hand-computed cross-session summaries)
- test_multisession_workflow.py:33 (fit once, reuse without refit), 205 (validate before fitting)
- test_nsd_cifti.py:69 (confound selection rule), 181 (pooled R2 lstsq oracle)
- test_nsd_hrf_selection.py:83 (grouped-design OLS/ridge oracle), 188, 227 (canonical oracle)
- test_nsd_parallel.py:38 (map_blocks ordering/concurrency/threads), 148 (worker failure publishes nothing)
- test_nsd_single_trial.py:50 (single-trial lstsq oracle)
- test_nsd_split_hrf.py:42, 88 (planted per-half HRFs; cross-half isolation)
- test_nsd_workflow.py:55 (NSD_TASK_MODEL), 89 (RT missing policy), 120 (trimming alignment), 194 (independent OLS oracle), 564 (RT-free selection feeds full GLM)
- test_ridge_outputs.py:51 (ridge export round trip)
- test_ridge_workflow.py:76, 173, 253 (oracle, isolation, provenance identity)
- test_session_hrf.py:61, 93, 174, 376, 392 (cache/import semantics incl. RT switch)
- test_within_run_encoding.py:20 (raw lstsq encoding oracle, both modes)
- test_workflow_surfaces.py:68 (vertex mapping)
- One smoke run per notebook: test_nsd_workflow.py:286 (1 param), test_multisession_workflow.py:129, test_session_hrf.py:319 (single execute)

## Notebook execution inventory

Kernel (nbclient) executions, 8 test functions / 14 kernel launches:
test_fractional_notebook:24 (1), test_multisession_workflow:33 (2, hidden via `_execute_workflow`),
:79 (2, hidden), :129 (1, multisession nb), test_nsd_workflow:286 (2 params), test_ridge_outputs:126
(3 params), test_session_hrf:319 (executes twice), test_workflow_surfaces:438 (1 mini-notebook).
In-process `exec()` of every cell: test_workflow_reuse:35 (3 params, 7 full executions).
`exec()` of cells by hardcoded id/substring (no fitting): test_nsd_workflow:466, 476, 481, 489, 499
(ids de5dc917/f3d49ae3), 504 (substring "beta_models ="); test_ridge_outputs:40, 46;
test_fractional_notebook:16; test_workflow_surfaces:81 (substring), 155, 273, 344, 392 (ids).
Total: 23 of 148 functions touch a notebook.
Measured: one full kernel run 5.5-8.4 s, in-process full exec ~2 s, mini-notebook 3 s. Estimated
~100-120 s of the suite's 246 s total; the recommended set (3 smoke runs, 1 reuse exec) is ~25 s.

## Stale or fragile text pins

- No test currently pins trial_type as centered or "mean stimulus response" selection wording; the
  tests were updated (commit 506fe52). The stale wording survives untested in modules:
  `nsd_hrf.py:387` SelectionScore "mean-stimulus leave-one-run-out CV R2" and `hrf_artifacts.py:228`
  Method "mean-stimulus prediction within each half" (both now use the task model), and
  `README.md:369-370` (session reliability "RT and trial type do not enter" vs include_rt=True default).
- Exact multi-sentence pins that will churn on any wording edit: test_nsd_workflow:410-427
  (trial_type, task, hrf_selection, hrf_curve_correlations method), :587-595 (hrf_selection with RT off,
  "never used to select HRFs"); test_nsd_hrf_selection:176 ("descriptive" in RT), :289-290 ("in-sample",
  Formula); test_multisession_analysis:203-206 (glm_units); test_fractional_notebook:69-71 and
  test_fractional_workflow:159-161 (selection_rule/validation_target/percentile_role, pinned twice);
  test_ridge_outputs:106-107, 177-180; test_within_run_encoding:115-119 (prediction_reference);
  test_ridge_workflow:284, 287 (validation_target, activity name); test_notebook_helpers:155
  ("No paired HRFs"); test_workflow_surfaces:124-126, 195-197, 301, 317-318, 373 (axis labels).
- test_nsd_workflow:470-471 pins the default library as 513 Sobol candidates with seed 0 via cell exec.

## Tests that would change if library-grade code moved into the package

- Import-path-only changes (importlib.import_module("examples.NSD.x") or direct imports):
  test_beta_activation, test_hrf_reliability, test_session_hrf_reliability, test_rt_diagnostics,
  test_within_run_encoding, test_nsd_parallel:38 (parallel_blocks), test_ridge_workflow,
  test_fractional_workflow, test_multisession_analysis.
- Private-name coupling that breaks on a move: test_beta_activation:82-84 (`_activation_artifacts`),
  test_fractional_workflow:20 (`_beta_artifacts`), test_session_hrf:148-166 (`_hrf_artifacts`,
  `_input_artifacts`, `_metadata`, `_stem`), test_nsd_workflow:586-593 (`_metadata`),
  test_nsd_hrf_selection:370-376 (`_load_runs`, `boldtailor._single_trial_design.compile_trial_run`),
  test_nsd_workflow:56, 100, 239-240 (`boldtailor._task_design`, `_hrf_design`).
- Monkeypatch targets tied to module attribute lookup: test_session_hrf:84, 186, 306 (`fit_session`),
  test_multisession_workflow:57, 193, 227 (`_execute_workflow`), test_nsd_single_trial:161
  (`fit_single_trials` on the example module), test_workflow_reuse:70-71 (`workflow_analysis.*`,
  `ridge_workflow.fit_cv_beta_series`), test_nsd_hrf_selection:26 (`expanded_hrf_library`), :325, :458
  and test_nsd_split_hrf:140 (`boldtailor.hrf_selection.select_hrf`).
- Cross-test fixture imports (no conftest): 9 files import `confounds/dataset/events` from
  test_nsd_cifti; test_ridge_workflow's `six_run_dataset`/`cv_library` feed 5 files; `hrf_nsd`, `mini_nsd`,
  `four_runs`, `saved_sessions`, `find`, `run`, `api`, `preview_library` are imported across files. A move
  needs an `examples/NSD/conftest.py` (or package test fixtures) first.
- Notebook-coupled: all 23 notebook tests reference `Path(__file__).with_name(*.ipynb)` and
  `path.parents[2]` as the kernel cwd; test_nsd_parallel:79 runs `nsd_single_trial.py` as a script.
