# SDD ledger — plan: docs/superpowers/plans/2026-10-02-review-remediation.md
Spec: docs/review-2026-10-01-full-project.md + docs/test-suite-review-2026-10-02.md (both reachable).
Worktree: .worktrees/review-remediation, branch fix/review-remediation-2026-10, base f0936ee.
Task numbering: plan uses "Task P.N"; task-brief is invoked with N="P.N".

## Pre-flight conflict scan
| pair / task | produces vs consumes | finding → ruling |
| --- | --- | --- |
| 0.1 ↔ 0.3 (pyproject) | 0.1 keeps pythonpath, 0.3 removes it | consistent |
| 0.2 ↔ 0.3 (conftest fixtures) | 0.2 Step 3 sentinel test uses `complete_sources`, defined only in 0.3 | Ruling: 0.2 uses the module's existing local `_complete_sources()` helper; 0.3 swaps to the fixture — avoids forward dependency; cost if wrong: one extra edit in 0.3 |
| 0.2 ↔ 1.1, 3.1, 4.1 (line refs) | later tasks cite original line numbers; 0.2 deletions shift lines | Ruling: line numbers are anchors only; implementers locate by test/function name (stated in every dispatch) |
| 0.2 ↔ 4.1 (`test_fit.py:690`) | 0.2 deletes the clipping pin; 4.1 adds the raising-constructor test | consistent |
| 0.5 ↔ 2.4 (`ENTRY_POINTS`) | 0.5 creates list; 2.4 appends hrf_selection names | consistent |
| 1.1 ↔ 4.7 (`results.py`) | 1.1 adds `mask_contrast` using kw construction; 4.7 sets kw_only | consistent |
| 1.3 ↔ 1.5 (`_compatible`, reuse keys) | 1.3 adds task_model_fingerprint; 1.5 adds hrf_normalization beside it | consistent; order 1.3 → 1.5 |
| 1.3 Step 1 (git stash notebooks) | worktree has clean notebooks; executed copies live in main's working tree | Ruling: skip the stash in the worktree; `.gitignore` `.env` entry still added; nothing to `rm`; report to user at finish that main's tree still holds executed notebooks, .env, png — cost if wrong: none (user state untouched) |
| 1.5 ↔ 3.1 (`fit(..., hrf_model="spm", task_model=...)` oracle) | 1.5 makes "spm" a peak-normalized callable; 3.1 oracle reads designs from the result | consistent (oracle uses fitted designs, not nilearn string) |
| 1.5 ↔ single_trial provenance tests | 1.5 changes `hrf_metadata("spm")` from string to dict | requirement change named in 1.5 brief; tests pinning `hrf == "spm"` update |
| 3.1 ↔ 4.4 (`group_designs`) | 3.1's test reads `result.group_designs[(0, cid)]`; 4.4 replaces with `group_design(run, hrf_id)` | Ruling: 4.4 dispatch must update 3.1's oracle test to the method form — cost if wrong: one failing test caught by 4.4's run |
| 2.1 ↔ 2.4 (software key) | 2.1 adds via FitOperation; 2.4 routes selection through FitOperation | consistent; 2.3's `activity["library"]` test unaffected |
| 2.2 ↔ 4.5 (digest handling) | 2.2 removes the ban; 4.5 removes `_safe_json` filtering | consistent |
| 3.2 ↔ 4.7 (result fields) | 3.2 adds defaulted fields; 4.7 kw_only | consistent |
| 0.1 ↔ 4.6 (`examples/NSD/conftest.py`) | 0.1 creates with marker hooks; 4.6 adds fixtures | additive |
| 4.6 ↔ README | 4.6 deletes CLI modules referenced by README command table | 4.6 brief includes README update |
| 3.3 (xfail on failure) | rubric: a strict xfail is not a vacuous test; failure is a recorded finding | acceptable per plan text |
| per-task self-consistency | each task's tests reference the code it specifies; files created precede files touched | no contradictions found |

## Progress
Task 0.1: dispatched (BASE f0936ee, implementer sonnet)
Task 0.1: minor (deferred): self-referential comment in tests/conftest.py addoption guard; broad `except ValueError: pass` in both conftests (consider root conftest or message check); two multisession tests marked `notebook` reach kernels only via ensure_session_outputs (naming nit); docs/development.md still says "CI runs the full suite" after the edited sentence.
Task 0.1: complete (commits f0936ee..e8a60d8, review clean)
Task 0.2: dispatched (BASE e8a60d8, implementer sonnet)
Task 0.2: Ruling: 14 appendix CUT rows live in examples/NSD/test_*.py (mis-filed under tests/test_task_design.py in the appendix); deferred to Task 4.6, which restructures the example tests — cost if wrong: 14 low-value example tests linger until Phase 4.
Task 0.2: Ruling: provenance sentinel test (plan template) was vacuous — sentinel must be injected into the environment (HOME + cwd) before building the record, as the logging sentinel test does — cost if wrong: a low-value test survives until the final review.
Task 0.2: Ruling: `test_make_task_delta_r2_result_clips_and_owns_values` (test_fit.py:690) is deleted per the brief despite its Appendix MERGE verdict — the review §4 names its pinned contract as inconsistent and Task 4.1 replaces it with a raising-constructor test; the nested-OLS oracle at test_fit.py:637 keeps clip semantics covered meanwhile — cost if wrong: ownership asserts for TaskDeltaR2Result are uncovered until 4.1/4.7.
Task 0.2: fix round 1/5 (2 addressed, 0 open — vacuous provenance sentinel; test_fit.py:690 deletion; commits 116843c..ad440b2)
Task 0.2: minor (deferred): none beyond the examples/NSD CUT rows carried to Task 4.6.
Task 0.2: complete (commits e8a60d8..ad440b2, review clean after 1 fix round)
Task 0.3: dispatched (BASE ad440b2, implementer sonnet)
Task 0.3: minor (deferred): docstrings on moved oracles (columns_for, task_model_oracle, oracle_cv); three conftests duplicate the --run-notebooks hooks (shared plugin later); unused `replace` import at tests/test_selected_hrf_fit.py (pre-existing); reviewer's shared-mutable-curves concern is moot (HrfLibrary already wraps curves/times in readonly_array).
Task 0.3: complete (commits ad440b2..1fb23d0, review clean)
Task 0.4: dispatched (BASE 1fb23d0, implementer sonnet)
Task 0.4: fix round 1/5 (2 addressed, 0 open — generic match fragments in test_model.py / test_task_model.py; commits fbe598e..f83743c)
Task 0.4: minor (deferred): no `ids=` on large tables (test_task_model, test_prepared); mutator return-dict side channel in test_prepared.py table; shared module-level `_BAD_DURATION` DataFrame in test_data.py; six near-duplicate `_fit_*` helpers in test_multirun.py; stale name test_run_sources_requires_exactly_one_signal_and_events_source (accept-path only); KEEP test renamed test_selected_glm_requires_matching_task_model → test_selected_glm_rejects_mismatched_selection_settings (assertions intact).
Task 0.4: complete (commits 1fb23d0..f83743c, review clean after 1 fix round)
Task 0.5: dispatched (BASE f83743c, implementer sonnet)
Task 0.5: fix round 1/5 (2 addressed, 0 open — privacy assertion, exception identity; commits 7ef8d8c..6da3c18)
Task 0.5: minor (deferred): failed-event data_id compared only between started/failed (no result to compare against); weak `len({data_id}) == 1` assertion; positional (valid, bad) call pairs in `_*_calls` helpers lack a docstring; early-failure privacy check follows the original weak pattern.
Task 0.5: complete (commits f83743c..6da3c18, review clean after 1 fix round)
Phase 0 complete.
Task 1.1: dispatched (BASE 6da3c18, implementer sonnet)
Task 1.1: Ruling: constancy is decided from the signal (`np.ptp(signal, axis=0) == 0` per run), not from an exactly-zero sum of squares — rounding leaves SST ~1e-31 for constants like 0.1, so the plan-text rule missed real constant features (reviewer probe: AR(1) z=3.75, r2=1.0). Constant runs contribute exactly 0 to SSE/SST so per-run R² is NaN; pooled R² keeps the existing guard; contrasts masked if constant in ANY run — cost if wrong: none for varying features (unchanged); a near-constant feature with ptp>0 is still fitted, which is the documented behaviour.
Task 1.1: fix round 1/5 (2 addressed + minor, 0 open — ptp-based constancy; doc wording; selected-HRF pin; commits fd7d484..ef93416)
Task 1.1: minor (deferred): `mask_contrast` does not validate `undefined` shape; prepared-fit constant test covers OLS single-run only.
Task 1.1: complete (commits 6da3c18..ef93416, review clean after 1 fix round)
Task 1.2: dispatched (BASE ef93416, implementer sonnet)
Task 1.2: minor (deferred): `_sidecar_start_time` duplicates its error message across two ifs and would raise AttributeError for a non-object JSON root; five-session default validated but not executed end-to-end; notebook-directory cwd execution case dropped; final review should confirm `_assert_published_metadata` still covers manifest `is_file` checks.
Task 1.2: complete (commits ef93416..d9eddf4, review clean)
Task 1.3: dispatched (BASE d9eddf4, implementer sonnet)
Task 1.3: Ruling: `uv run pytest examples/NSD` fails collection (15 import errors: `examples` not importable) since Task 0.3 dropped `pythonpath`; fix by inserting the repo root into sys.path in examples/NSD/conftest.py exactly as examples/validation/conftest.py does — load-bearing for the CI examples job and the documented command; folded into Task 1.3 as an extra commit — cost if wrong: none for the default suite (NSD conftest is not loaded there).
Task 1.3: minor (deferred): cosmetic import order in test_workflow_reuse.py; awkward line wraps in README.md and docs/glmsingle-comparison.md replacements; docs/user-guide.md:125,442 still say "mean stimulus response" for the general API default (accurate for TaskModel(); leave).
Task 1.3: complete (commits d9eddf4..712bbf6, review clean; includes ruled addendum 712bbf6)
Task 1.4: dispatched (BASE 712bbf6, implementer haiku)
Task 1.4: Ruling: reviewer's "Critical" (commit trailer should name Claude Haiku 4.5) is rejected — the plan's Global Constraints and the session attribution rule fix the trailer as `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; the reviewer applied its own model's harness reminder — cost if wrong: none (attribution text only). AST-equivalence verified for all 11 files by controller.
Task 1.4: complete (commits 712bbf6..74deb26, formatting only)
Task 1.5: dispatched (BASE 74deb26, implementer opus)
Task 1.5: Ruling (scientific, needs user visibility): peak-one KERNELS inside nilearn's dt-free convolution make betas depend on TR, oversampling and event duration (reviewer probe: 3 s SPM event regressor peak 34.5 at os=20, 85.2 at os=50, TR 1.6; 70.7 vs 86.2 across candidates), so the user's goal (comparable betas across HRFs) is not met by kernel scaling alone. Decision: normalize each EVENT RESPONSE to unit peak — scale every event's amplitude by 1/peak_k(d), where peak_k(d) is the maximum sliding-window sum of n = max(1, round(d/dt)) samples of kernel k at dt = TR/oversampling (nominal sampled boxcar); applied via nilearn's modulation column in task_columns/_make_design_matrix/convolve_events and as a per-column factor on the trial_regressors fast path. Kernels stay peak-one (literal request, now cosmetic for betas). Result: a beta is the peak BOLD response to that presentation in signal units, invariant to TR/oversampling, comparable across kernels and durations (GLMsingle convention). Cost if wrong: beta units change once more; reverting is deleting one scaling function; all selection/ridge/R² statistics are scale-invariant.
Task 1.5: fix round 1/5 (2 minors addressed, CRITICAL partially — nominal n=round(d/dt) leaves onset-alignment error up to 1/n for short events, e.g. 0.1 s event peak 1.50; NSD 3 s events 0.9%; commits 13d7bcf..47ba409)
Task 1.5: Ruling (amends previous): the boxcar sample count per event must be the REALIZED count on nilearn's high-resolution grid for that run (the `starts/stops` searchsorted logic `_boxcar_sampling` already implements, at least 1 sample), not round(d/dt); scales therefore depend on (run grid, onset, duration, kernel) and make every event's realized high-resolution response peak exactly 1 — cost if wrong: none beyond compute (per-event scale evaluation), and test oracles must derive n independently via nilearn's `_sample_condition`.
Task 1.5: fix round 2/5 (CRITICAL addressed — realized grid counts; commits 47ba409..22fa6e7)
Task 1.5: minor (deferred): `task_columns` does not validate onsets at/after the last grid point (count 0 → clamped to 1; pre-existing edge case); README exception wording for derivative/FIR bases was added in round 1 — final review to confirm; error-message text for plain spm/glover drift failures now says "nuisance design compilation failed".
Task 1.5: complete (commits 74deb26..22fa6e7, review clean after 2 fix rounds)
Phase 1 complete.
Task 2.1: dispatched (BASE 22fa6e7, implementer sonnet)
Task 2.1: Ruling: the `from_arrays`/`PreparedDesignAnalysis` normalization activities carry no software record (prepared lost `software_versions` per the brief); add `software=software_environment()` to both normalization activities in Task 2.4 alongside routing HRF selection through fit_operation — cost if wrong: duplicate key in two records.
Task 2.1: minor (deferred): import ordering in _fit_lifecycle.py and test_lifecycle.py; function-local `import platform` in _software.py (plan-mandated snippet).
Task 2.1: complete (commits 22fa6e7..35deffe, review clean)
Task 2.2: dispatched (BASE 35deffe, implementer sonnet)
Task 2.2: minor (deferred): parent-mismatch message literal repeated in three modules (shared constant); regex compiled per call in `_validate_sha256`; anonymous sources with sha256 never reach `Digest` (undocumented); duplicate-URI sources: first digest wins silently.
Task 2.2: complete (commits 35deffe..3ca11e1, review clean)
Tasks 2.3+2.4: dispatched as one batch (BASE 3ca11e1, implementer sonnet) — includes ruled addendum: `software` key on from_arrays/prepared normalization activities.
Tasks 2.3+2.4: Ruling: activities that embed another record's last activity (hrf_selection evaluate_hrf_split `training_selection`, _selected_hrf_fit `selection`, _hrf_glm `selection`, _ridge_cv fold `hrf_selection`) must embed it WITHOUT the `software` key so analysis ids stay version-independent; one shared helper `identity_activity(record)` in provenance.py; fixed in this batch rather than deferred because the defect is structural and the implementer is live — cost if wrong: embedded selection records lose the software key (still available on the selection's own record).
Tasks 2.3+2.4: fix round 1/5 (Important + minor addressed — identity_activity helper at four embedding sites; commits 7f5702f..4d31f27)
Tasks 2.3+2.4: minor (deferred): shallow freeze of `origin`; lifecycle test does not pin outer execution_id across nested selection; ridge `_fold_selection` identity covered only via the shared helper.
Tasks 2.3+2.4: complete (commits 3ca11e1..4d31f27, review clean after 1 fix round)
Phase 2 complete.
Task 3.1: dispatched (BASE 4d31f27, implementer sonnet)
Task 3.1: minor (deferred): selected-GLM oracle combines runs with nilearn Contrast arithmetic (same mechanism as production) and does not check design task columns or amplitudes in SE units; multi-run test does not assert designs equal the fixture's.
Task 3.1: complete (commits 4d31f27..c27f892, review clean; strict xfail records the nilearn rank-deficient variance defect)
Task 3.1: Ruling: SCIENTIFIC FINDING — nilearn 0.14.0 `OLSModel.fit` (regression.py:196-198) divides dispersion by n − n_columns while df_residuals = n − rank, so on rank-deficient designs contrast variance is inflated by (n−rank)/(n−cols) (25/24 in the probe) and t/p are conservative; boldtailor inherits it. Remedy: new Task 3.5 — fit rank-deficient designs on a full-rank basis (SVD row-space basis V_r: X_r = X V_r, c_r = V_rᵀ c for estimable c) through nilearn and map contrasts back, turning the strict xfail into a passing assertion — cost if wrong: full-rank designs are untouched (r = p path identical); rank-deficient fits change only variance/stat/p by the (n−rank)/(n−cols) factor.
Task 3.5: dispatched (BASE 3e0c156, implementer sonnet)
Task 3.5: minor (deferred): rank computed in three places in _conventional.py (shared helper would prevent tolerance drift); extra SVD per candidate design in fit_r2_designs; development.md paragraph not line-wrapped.
Task 3.5: complete (commits 3e0c156..d531040, review clean)
Task 3.2: dispatched (BASE d531040, implementer sonnet)
Task 3.2: minor (deferred): redundant `kind != "spm"` mask in `_at_parameter_bound`; no shape check on user-supplied `at_boundary`; missing edge-case tests (unscored False, one-point grid, zero-width bounds, exact 2 % threshold); tie behaviour (flat profile → endpoint → True) undocumented; cosmetic doc wrapping.
Task 3.2: complete (commits d531040..da42326, review clean)
Task 3.3: dispatched (BASE da42326, implementer opus)
Task 3.3: complete (commits da42326..1833930, review clean; all five recovery tests pass)
Task 3.3: Ruling: the brief's HRF recovery library row id 3 (5,14,1.0,1.5,6,1.0) is cosine 0.995 to canonical, so recovery passed at exactly the 0.75 threshold; this is a test-design error (not a method finding) — Task 3.4 replaces id 3 with a well-separated kernel, adds a near-tie row to the canonical control, keeps thresholds, and writes docs/validation/recovery-2026-10.md recording: shrinkage selected (0.4, centered RMSE 0.81 vs OLS 2.21), alpha 10 at the grid top with the objective still rising (at_boundary), OLS control margin 0.752 vs 0.751, HRF confusion details — cost if wrong: a stronger test that could fail honestly; that would itself be a finding.
Task 3.3: minor (deferred): `delta_cv_r2` median assertion nearly implied by argmax selection; module-level `HRF_TIMES` shared array; implementer report misnamed the confusable candidate (corrected in the ledger above).
Task 3.4: dispatched (BASE 1833930, implementer sonnet) — includes the ruled test-design fix and validation note.
Task 3.4: minor (deferred): long unwrapped line in user-guide fixed-effects paragraph; 2026-09-28 review bullet "(finding #1 remediated)" not struck through (note sits above it); recovery note's "not by a selection bias" overclaims (soften to "consistent with"); >88-char string in hrf_selection.py.
Task 3.4: complete (commits 1833930..fa071ab, review clean)
Phase 3 complete.
Task 4.1: dispatched (BASE fa071ab, implementer sonnet)
Task 4.1: Ruling: the ΔR² identity hashed into analysis ids is derived from the shared builder — `delta_r2_identity(...)` = `delta_r2_activity(...)` minus `stage`, `parent_analysis_id`, `undefined_features` — used by all three entry points; analysis ids for ΔR² results change once on this branch (identities already changed for the peak-normalization and software work; none are pinned) — cost if wrong: one-time id change for saved comparisons, which reuse checks already reject for other reasons.
Task 4.1: fix round 1/5 (2 Important + minors addressed — delta_r2_identity shared; HRF id pinned; commits 419c0e9..7ab4f2d)
Task 4.1: minor (deferred): dead `data_id` parameter in prepared_fit._prepared_comparison_id; function-local private import of fit._nuisance_model_settings in _hrf_glm (Task 4.3 moves it); no direct constructor tests for within-tolerance clip and allow_undefined NaN; `_PREPARED_NUISANCE_MODEL` value schema differs from conventional nuisance_model; adjacent duplicate `from tests.oracles import` lines in two test files.
Task 4.1: complete (commits fa071ab..7ab4f2d, review clean after 1 fix round)
Tasks 4.2+4.3: dispatched as one batch (BASE 7ab4f2d, implementer sonnet)
Tasks 4.2+4.3: minor (deferred): direct `compile_trial_run(..., "a b")` no longer rejected (unpinned); two `from boldtailor.data import` lines in _ridge_cv.py; `"keep"` string sentinel in model_identity; label-validator test omits evaluate_hrf_split and fractional scoring; weak `not hasattr` test; blank-line spacing in _single_trial_design.py.
Tasks 4.2+4.3: complete (commits 7ab4f2d..1b51f5f, review clean)
Task 4.4: dispatched (BASE 1b51f5f, implementer opus) — carries pre-flight ruling: update Task 3.1's selected-GLM oracle test from `group_designs[(run, cid)]` to `group_design(run, cid)`.
Task 4.4: fix round 1/5 (Important + minors addressed — run tables read once per fold; pickling documented; float run → KeyError; commits f6aabd2..9f67012)
Task 4.4: minor (deferred): library-path fold reads events twice (subset_runs) so the "≤1 access" bound is untested/false there (constant, not quadratic); memory test excludes the rebuild closure (which keeps RunDesign caches alive); `group_design` deep-copies a freshly compiled frame; bool run accepted as int.
Task 4.4: complete (commits 1b51f5f..9f67012, review clean after 1 fix round)
Task 4.5: dispatched (BASE 9f67012, implementer opus)
Task 4.5: fix round 1/5 (Important + 5 minors addressed — symlinked destination resolved; Command mappings; recovery_directory; filters; docs; commits 9ad63b9..fba0673)
Task 4.5: minor (deferred): three modules total 1,343 lines (plan estimate was ≤1,000; remaining code serves kept guarantees); case-only duplicate artifact names can overwrite on macOS now that casefold scanning is gone (documented limitation); only the last activity carries BEP028 timestamps; one overlong line in development.md.
Task 4.5: complete (commits 9f67012..fba0673, review clean after 1 fix round)
Task 4.6: dispatched (BASE fba0673, implementer opus) — includes the 14 examples/NSD CUT rows deferred from Task 0.2.
Task 4.6: Ruling: the CLI stack deletion (ff0ec1b) removed outputs with no notebook equivalent (stimulus+RT-only GLM, per-run RT/rtcount maps, beta-series hrfΔR² maps, selected-vertex RT table/scatter, HRF-curve plots, fmriprep-inside-bids error) against the dispatch's stop-and-report instruction; revert ff0ec1b so the CLI stack stays (now on public helpers) and surface its retirement as a user decision — cost if wrong: two stacks persist a little longer (E3 partially open).
Task 4.6: Ruling: 17 commit trailers name "Claude Opus 5.5"; rewrite them to the plan's `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` on this unpushed branch (msg-filter over fba0673..HEAD) — cost if wrong: none (message text only).
Task 4.6: deferred to Task 5.1: examples/validation still imports 7 private solver primitives (out of 4.6 scope); `test_generator_is_paired_and_stresses_the_design` in examples/validation fails at base fba0673 (pre-existing) — investigate.
Task 4.6: Ruling (revised): the Opus-authored commits keep their `Claude Opus 5.5` trailers — the implementer argued, reasonably, that rewriting them would credit a model that did not write them; the branch now carries mixed trailers (Fable 5.1 on sonnet-authored commits that followed the brief, Opus 5.5 on this task). Parked for the user; cost if wrong: attribution text only.
Task 4.6: fix round 1/5 (CLI stack restored per ruling; BIDS_ROOT default removed; trailers parked; commits 41abdcd..34b15da)
Task 4.6: fix round 2/5 (Important addressed — restored CLI de-duplicated onto shared helpers; minors; commits 34b15da..8888469)
Task 4.6: parked — six CLI-only outputs (stimulus+RT-only GLM, per-run RT/rtcount maps, beta-series hrfΔR² maps, selected-vertex RT table/scatter, HRF-curve plots, fmriprep-inside-bids check) — Ruling: CLI stack retained pending the user's decision on retiring it.
Task 4.6: minor (deferred): no value-equivalence test for nsd_cifti R² outputs written via scalar_artifact (passing suite + code inspection only); `workflow_plots.py` has its own corrcoef (plot-only); examples/validation still imports internal solver primitives (`_alphas`, `_project_design`, `fraction_beta_path`, `trial_beta_path`); long CLI runner functions (85/99/47 lines) pre-existing; nibabel>=5.2 now a runtime dependency (nilearn already requires it).
Task 4.6: complete (commits fba0673..8888469, review clean after 2 fix rounds)
Task 4.7: dispatched (BASE 8888469, implementer opus)
Task 4.7: finding carried to 5.1: tests/test_stop_signal_demo.py notebook smoke test fails intermittently under --run-notebooks with FileNotFoundError in publication staging; reproduces at 8888469 (so introduced by Task 4.5 publication rewrite or earlier) — investigate and fix in Task 5.1.
Task 4.7: minor (deferred): sentinel-typed annotations (`HrfSelectionResult = UNSET`) in single_trial.py/prepared_fit.py; BIDS `Command` for selection changed to `boldtailor.select_hrfs` without a migration note; `tests/test_shared_constants.py` identity test on int 50 is weak; `hrf_glm_problem` fixture imported across test modules; partial old-keyword calls to task_delta_r2_prepared fill missing keywords with defaults rather than stored values (documented).
Task 4.7: complete (commits 8888469..29d809e, review clean)
Phase 4 complete.
Tasks 5.1+5.2: dispatched as one batch (BASE 29d809e, implementer sonnet) — includes: intermittent stop-signal notebook staging FileNotFoundError; examples/validation private imports; pre-existing failing validation test.
Tasks 5.1+5.2: minor (deferred): thin margin (0.114 vs 0.1) in the restated validation generator test; `is_control_directory` is suffix-based (a destination named `*.boldtailor` would be misclassified); overlong import comment in single_trial.py; development.md branch inventory does not record the four deleted (merged) branches.
Tasks 5.1+5.2: complete (commits 29d809e..6afd45f, review clean)
Phase 5 complete. All plan tasks complete at 6afd45f. Final whole-branch review next (merge base f0936ee).
Final review (opus, f0936ee..6afd45f): no Critical; 5 Important (boundary flags uninformative; no beta-scale migration note; NSD ridge_provenance identity depends on software; stdout regression from kernel_task_columns; cross-test imports remain) + must-fix minors (dev.md CI line; callables/FIR normalization wording; "consistent with"; nsd-session factor; publication-migration body; nsd_cifti constant; truncation doc).
Final: Ruling: HRF boundary diagnostics become a per-parameter table (`HrfSelectionResult.parameter_bound_flags`, features × 6 parameters × {low, high}) and the scalar `at_parameter_bound` is the any() over parameters with ≥3 distinct library values and nonzero width (so constant or two-level grid parameters do not flag every pick); ridge `at_boundary` stays factual at both ends but the docs distinguish the shrinkage end ("extend the grid") from f=1.0/α=0 ("no regularization preferred"); the NSD workflow writes a per-parameter flagged-fraction summary into its metadata — cost if wrong: diagnostics only; no numerics change.
Final: Ruling: one fix wave covering all five Important items and the reviewer's must-fix minors; residuals after one scoped re-review are adjudicated, not re-dispatched.
Final: fix wave dispatched (BASE 6afd45f, implementer opus)
Final fix wave: 4 of 5 Important + all must-fix minors ADDRESSED (commits 6afd45f..1a590f0); residual: Important #1 partially — per-parameter table and informative-parameter scalar exist, but the scalar `at_parameter_bound` still flags ~99 % of `expanded_hrf_library()` candidates (five 3-level factors) while the user guide implies it is meaningful there.
Final: parked — scalar `at_parameter_bound` saturates on coarse full-factorial grids — Ruling: real but documentation-only; the per-parameter table (`parameter_bound_table()`) is the informative output and the scalar is meaningful for continuous (Sobol) libraries; one sentence in docs/user-guide.md and docs/api.md should say the scalar saturates on coarse grids and point to the table — surfaced to the user (no second fix wave by process); cost if wrong: a misleading diagnostic sentence until fixed.
Final: parked — test name mangled by search-and-replace (`test_selected_delta_requires_completeglm_run_sources`, tests/test_hrf_glm.py:281) — cosmetic.
All work complete at 1a590f0.
