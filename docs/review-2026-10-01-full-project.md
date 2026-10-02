# Full project review: scientific accuracy and software architecture

**Date:** 2026-10-01
**Scope:** `src/boldtailor/` (7,294 lines, 38 modules), `tests/` (13,122 lines,
39 files, 453 test functions), `examples/` (NSD workflow 6,132 non-test lines in
29 modules plus 4,879 test lines; `stop_signal_demo.py` 622 lines; four
notebooks; `examples/validation/` 874 lines), packaging, CI, and all
documentation under `docs/`.
**Method:** full read of every numerical module by the lead reviewer
(`_conventional`, `_single_trial_design`, `_single_trial_fit`,
`_fractional_ridge`, `fractional_ridge`, `_ridge_cv`, `ridge_selection`,
`trial_encoding`, `_hrf_design`, `hrf_library`, `_hrf_cv`, `hrf_selection`,
`_hrf_assignment`, `_task_design`, `design`, `_hrf_glm_design`, `_hrf_glm`,
`_selected_hrf_fit`, `single_trial`, `model`), plus four parallel
reviews of the API/lifecycle layer, the provenance/publication layer, the
examples, and the tests/docs/CI. Claims below were verified by reading code
and, where marked, by running small probes. The prior review
(`docs/review-2026-09-28-full-project.md`) was used as the baseline for
persistence checks. **No files were changed by this review.**

**Verification state of the tree:** `uv run pytest -q` passes (1,021 tests,
246 s). `uv run black --check src tests examples/NSD examples/stop_signal_demo.py`
**fails on 17 files**, so the CI workflow as written would fail on `main`.
Two notebooks are modified but uncommitted, and `examples/NSD/.env` plus a
733 KB PNG are untracked (see §4.4).

---

## 1. Verdict

The numerical core remains correct and well guarded against leakage. The
fractional-ridge root finding, the normalized-ridge SVD path, the FWL nuisance
handling, the HRF sufficient-statistics machinery, and the within-run encoding
objective are all implemented as documented, and the independent oracles in
the test suite are real.

Three things changed the picture relative to the 2026-09-28 review:

1. **One genuine scientific bug was found in the conventional GLM.** A feature
   whose signal is exactly constant (zero-padded grayordinates, masked-out
   voxels, dead channels) receives a finite, often large t and z statistic
   instead of NaN. The prior review recorded this as remediated; it is not
   in the current tree and no test covers it (§2.2, S1).
2. **The architectural backlog was substantially worked down** on 2026-09-28
   (publication shrank 754→480 lines, fd-anchoring and the sanitizer are gone,
   `fit_operation` exists, CI exists, `ipykernel` is a dev dependency). What
   remains is a second tier: three delta-R² lifecycles, 25 cross-module
   private imports, 40 `object.__setattr__` ownership hacks, three run-label
   validators, and 6,100 lines of library-grade code still living in
   `examples/NSD`.
3. **The examples drifted from the package.** The task model changed to an
   uncentered `trial_type` on 2026-09-30, but notebook prose, README text, the
   GLMsingle comparison, and the saved-result compatibility checks still
   describe or assume the old model. The stop-signal demo has a mis-weighted
   contrast and crashes with its own default session list.

The highest-value work, in order: fix S1; finish the provenance record so it
actually identifies inputs and software; reconcile examples with the current
task model; then resolve the ridge-tuning estimand, which remains the open
research question everything downstream depends on.

---

## 2. Scientific accuracy

### 2.1 What is solid (verified)

- **Conventional GLM** (`_conventional.py`): nilearn `run_glm` OLS/AR(1);
  per-run estimability check via `c·pinv(X)·X = c`; residual-dof guard;
  rank-deficiency warnings; R² pooled from per-run SSE/SST with per-run means.
  Equal-weight fixed effects via nilearn `Contrast.__add__` (effect mean,
  variance Σvar/n², dof summed) match the documented contract.
- **Single-trial OLS/ridge** (`_single_trial_fit.py`): nuisance basis by SVD
  with rank tolerance; trial columns residualized and unit-normalized;
  ridge via `s/(s²+α)` on the SVD, then de-scaled. FWL-correct; nuisance
  coefficients recovered by lstsq on the remainder. Features with zero range
  get NaN.
- **Fractional ridge** (`_fractional_ridge.py`): fraction defined as
  ‖β_f‖/‖β_OLS‖ in the raw coefficient basis after nuisance projection.
  Bracketing with `α = s²(1−f)/f` at the smallest/largest singular value is
  exactly right; 60 log-space bisections give far more than double precision.
  Features with no OLS signal are masked before the norm ratio.
- **Encoding-guided CV** (`_ridge_cv.py`, `trial_encoding.py`): per-fold HRF
  selection uses training runs only; within-run mode fits slopes on
  within-run-centered training betas and scores held-out residuals after
  removing their per-feature mean (one held-out intercept per feature, used
  for scoring only). Fractional mode uses fixed OLS targets. Fold bookkeeping is
  leakage-safe by inspection, and `test_within_run_cv_isolation.py` would
  detect a regression.
- **HRF selection** (`_hrf_cv.py`): A = XᵣᵀXᵣ, B = XᵣᵀYᵣ, C = ‖M Y‖² per
  (run, candidate) with confounds and profiled columns projected per run;
  pooled training amplitude β = (ΣA)⁻¹ΣB; held-out loss C − 2βᵀB + βᵀAβ with
  a roundoff guard; score = 1 − Σloss/Σenergy with a candidate-independent
  denominator. Batched products are exact. The fast single-trial convolution
  path (`_hrf_design.trial_regressors`) is tested against `compute_regressor`.
- **HRF library** (`hrf_library.py`): candidate 0 is bit-exact nilearn
  `spm_hrf`; custom kernels sum-normalized; Sobol sampling deterministic and
  sorted so IDs are stable.
- **Honest documentation** of the AR(1) unwhitened R², sum-normalized kernel
  units, one-sided p-values, nilearn/double-gamma peak-time offset, and the
  cross-run transfer assumption.

### 2.2 New scientific findings

**S1 (high, NEW). Exactly constant features receive finite, spurious contrast
statistics.** `_conventional._fit_run` computes contrasts for every column of
`signals`; for a constant column, nilearn returns effect ≈ 1e-15 and variance ≈
1e-30, so `effect/sqrt(variance)` is an O(1)–O(10) number. Verified with a
4-feature probe where feature 2 was the constant 50.0:

| noise model | effect | t | z | R² |
| --- | --- | --- | --- | --- |
| ols | 0.000 | 7.662 | 6.861 | NaN |
| ar1 | −0.000 | −2.141 | −2.115 | NaN |

`r2` is correctly NaN (the `total_sum > 0` guard exists), but contrasts are
not masked, and the divide-by-zero `RuntimeWarning` from nilearn is explicitly
silenced (`_conventional.py:179-185`), which hides the symptom. No test in
`tests/` mentions constant or zero-variance features for the conventional
path. Affects `fit`, `fit_prepared`, and `fit(..., hrf_selection=)` equally
(all share `fit_designs`). Realistic triggers: zero-filled medial-wall
grayordinates in some CIFTI pipelines, voxels at mask edges after
resampling, runs where a sensor channel is flat. Fix: in `_fit_run`, mask
`total_sum == 0` features to NaN in every `_ContrastResult` field (the
single-trial path already does the equivalent with `varying`), remove the
warning filter, and add the missing test. The 2026-09-28 review's statement
that this was "remediated" should be struck.

**S2 (medium, NEW). HRF-selection scores contain a candidate-dependent
in-sample component through profiled indicator columns.** When a `Modulator`
uses `missing="indicator"`, the `missing_<column>` regressor is convolved with
the *candidate* HRF (`_hrf_cv.RunDesign._build` → `task_columns` with
`hrf_model(candidate)`) and then projected out of the held-out run in-sample
(`qp` in `_Block`, and `c[cid]` in `signal_statistics`). A candidate whose
kernel happens to fit the missing-trial response well lowers the held-out C
without any out-of-sample prediction. With few missing trials the effect is
small, and the user guide does say the coefficient is "fit freely within each
run", but the scientific consequence (the score is no longer a pure
leave-one-run-out prediction) is not stated. Options: convolve indicators
with a fixed canonical kernel so they are candidate-independent; or include
them in the predicted set with pooled amplitudes; or at minimum document the
bias and report the fraction of trials affected.

**S3 (medium, NEW). No boundary/edge-of-grid diagnostics remain in the API.**
The ridge ablation found endpoint selection in 19–44 % of cases, and the prior
review relied on `_SelectionBoundaries` for reporting. That class is gone
(`grep -i boundar src/` is empty) and nothing replaced it: `FractionSelection`,
`RidgeSelection`, and `HrfSelectionResult` carry no "winner is at a grid
endpoint" or "winner is at the edge of the Sobol box" flag. For the HRF
library this matters because the parameter box (response delay 3–6 s,
dispersion 0.5–1.5, onset 0–2 s) implies peak times of roughly 1.5–7.5 s;
GLMsingle's empirical library spans later peaks. A grayordinate whose true
HRF peaks at 8–9 s will pick the slowest available kernel with no indication.
Add a boolean `at_boundary` per feature for ridge grids and a per-parameter
"at box edge" table for HRF selections, and summarize them in the NSD outputs.

**S4 (medium, NEW). Examples describe a task model the code no longer fits.**
`workflow_inputs.py:31` uses `Modulator("trial_type", center=False)` (since
718daed, 2026-09-30). `nsd_workflow.ipynb` cell 5 still says "Binary code
minus its run mean"; `nsd_multisession.ipynb` cell 9 says the task coefficient
is "evaluated at mean trial type"; README:124 and `glmsingle-comparison.md:68`
still say HRF selection uses "the mean stimulus response" although the NSD
selection now uses the full task model. The uncommitted working copy of
`nsd_workflow.ipynb` goes further and introduces the wrong statement "Both
modulators are centered within run". Separately, `workflow_reuse.ANALYSIS_SETTINGS`
and `multisession_inputs._compatible` do **not** include a task-model
fingerprint, so results fitted under the old centered model are reported as
"Reused" and can be pooled with new ones. The session-HRF cache
(`session_hrf_cache.py:45`) does this correctly; copy that pattern.

**S5 (high, NEW, examples). Stop-signal demo contrast and session handling.**
`stop_signal_demo.ipynb` cells 7 and 9 define
`stop_vs_go = "(stop_success + stop_failure) - go_success"`, weights (1, 1, −1),
which equals 2·mean(stop) − go and is nonzero under the null of equal means.
Intended is `0.5*stop_success + 0.5*stop_failure - go_success`. The test
(`tests/test_stop_signal_demo.py:52,57,887`) pins the current string, so this
is a legitimate requirement change to the test. In the same notebook, cell 1
defaults to five sessions while the validator demands "exactly two", the
title says "Five-session", provenance says "two-session", and cell 9 does
`plt.subplots(1, 2)` then a `strict=True` zip over sessions, which raises
with the default list. Also `stop_signal_demo.py:588` builds frame times as
`arange(n)·TR`, ignoring the BOLD sidecar `StartTime` that the NSD loader
honours (`nsd_cifti.py:174-180`), so onsets are misaligned by up to TR/2
relative to the NSD example.

**S6 (low-medium, NEW). Provenance cannot identify inputs or environment.**
The record hashes `(uri, byte_size, modified_at, media_type, annotations)` and
*forbids* a `digest` key (`provenance.py:14,503-509`; stripped again in
`bids_provenance.py:29-39`). Software versions are recorded inconsistently:
none for `fit()` or `select_hrf()`, numpy/nilearn for single-trial via
`__version__`, boldtailor/numpy/pandas for prepared fits; scipy (Sobol),
Python, and platform never. The HRF library is recorded only as a hash, not
as `{"kind": "sobol", "n_samples": 512, "seed": 0}`. A reader of the JSON
alone cannot reproduce the candidate set or confirm which data were fitted.
This is a scientific-reproducibility gap more than an engineering one.

### 2.3 Persisting scientific concerns (restated briefly)

- **P1. Ridge-tuning estimand (highest research priority).** The 2026-09-28
  adoption of raw-basis fractions with fixed OLS targets was a defensible
  default, and the ablation is honest that it does not dominate on every
  metric. The objective still rewards predictability from the encoding model,
  not recovery of true trial amplitudes; the shared-alpha path still scores
  candidate-regularized targets. No test shows the selector picks shrinkage
  when shrinkage is truly optimal (see T1).
- **P2. Equal-weight fixed effects with summed dof.** Correct for the stated
  contract; anticonservative under heteroscedastic runs. Still no calibration
  evidence, and `docs/api.md` does not say it is not precision-weighted.
- **P3. ΔR² parent checks are metadata-only** and the error message "full
  result does not match data, model, or HRF identity" implies content
  identity. Now compounded by S6.
- **P4. Cross-run transfer assumption** in HRF selection: still no in-package
  per-fold amplitude spread diagnostic.
- **P5. `_fractional_ridge.prepare_fraction_betas` still does two SVDs**
  (normalized for rank validation, raw for the fit); the vestigial `scale`
  is gone, so this is now a cost/readability item rather than a correctness
  risk.
- **P6. nilearn pin.** `uv.lock` pins nilearn 0.14.0, which the project's own
  validation record notes uv reports as yanked. `_hrf_design.py:7` imports the
  private `_sample_condition`; the pin is what keeps that import safe.

Small items: `model._validate_noise_model` message still says "in Phase 1";
`model._is_integer` rejects numpy integers for `drift_order`/`oversampling`
while every other validator uses `numbers.Integral`; `design._select_confounds`
coerces non-numeric confounds to NaN and then reports "must be finite";
`fit(..., hrf_selection=)` silently requires `ModelSpec` defaults for
`min_onset` and `oversampling` because `_hrf_cv` hardcodes −24/50, and the
error does not say so; `_ridge_cv._score_fold:164` reads `fit` from the last
loop iteration.

---

## 3. Software architecture

### 3.1 Persistence check against the 2026-09-28 review

| Finding (2026-09-28) | Status now |
| --- | --- |
| A1 imaging code in `examples/` | **Persists, grew**: 6,132 non-test lines in 29 NSD modules; two parallel stacks (CLI `nsd_cifti`/`nsd_hrf`/`nsd_single_trial` vs notebook `workflow_*`/`ridge_*`/`session_hrf*`) whose R² "need not agree" per README |
| A2 provenance/publication share of source | 1,820/7,294 = 25 %; absolute size roughly unchanged |
| A3 fd-anchored publication | **Largely resolved**: no `fcntl`/`F_GETPATH`/`/proc`; 480 lines; negative threat model written in `development.md:310-315` |
| A4 `fit`/`prepared_fit` parallel lifecycles | **Partially resolved**: `_fit_diagnostics`, `fit_operation` exist; three delta-R² lifecycles with three activity schemas remain |
| A5 cross-module private imports | **Persists**: 25 `from boldtailor.x import _y` statements; `prepared.py` imports 10 privates |
| B1 quadratic deep copies | **Fixed in `prepared_fit`**; persists in `design.py:35` and `_ridge_cv.py:32-34,83,90` |
| B2 `_ImmutableFloatArray` | **Replaced** by `_arrays.readonly_array` (documented convention) |
| B3/B4 two privacy policies, env redaction | **Resolved**: single `logging._error_code` |
| B5 triple `ProvenanceRecord` construction | **Resolved**; but `from_arrays` (74 lines) and `_prepare_analysis` (88 lines) still hand-roll the lifecycle `fit_operation` abstracts |
| B6 no-op fingerprint line | **Resolved** |
| B7 ΔR² gated on metadata fingerprints | **Persists** (documented) |
| C1 notebook string-injection tests | **Persists**: `test_stop_signal_demo.py` 1,651 lines, pins `cmap`, `cut_coords`, titles, `display_count == 16`, a memory estimate to 17 digits |
| C2 `/Users/poldrack/...` assertion | **Persists** at `tests/test_provenance.py:318` plus `Path.home()`/`gethostname()`/`Path.cwd()` asserts |
| C3 `test_repository_contracts.py` pins packaging strings | **Persists** |
| D hardcoded `BOLDTAILOR_VERSION` / import-time `package_version()` | **Resolved** (`_software.py`, `test_software.py`) |
| D no CI | **Resolved** (`.github/workflows/tests.yml`), but Black step fails on current tree and the wheel name is hardcoded |
| D `.gitignore` gaps | **Resolved**, new gap: no `.env` |
| D stale branches | **Persists**: 7 `safety/*`, 2 `task*-review-safety`, 7 `feature/*`/`fix/*` |
| NEW-1 `_prepare` double SVD / vestigial scale | **Half resolved** (scale gone, double SVD remains) |
| NEW-2 cross-module frozen mutation | **Resolved** |
| NEW-3 duplicate result containers | **Resolved** (`CandidateScores`, `SharedTrialDesign`/`SelectedTrialDesign`) |
| NEW-4 `_run_beta_path` generator protocol | **Resolved** (`RunBetaPath.betas_at`) |
| NEW-5 dead `append_event_history` | **Resolved** |
| NEW-6 packaging (`ipykernel`, CI, `pythonpath`) | `ipykernel` fixed; `pythonpath=["."]` persists and cross-test imports returned (`test_hrf_cv` ↔ `test_hrf_selection`) |
| NEW-7 personal path in `glmsingle-comparison.md` | **Persists** (line 98) |

### 3.2 Measured facts (current tree)

| Layer | Lines | Share |
| --- | --- | --- |
| Numerics and estimation (14 modules) | 2,389 | 33 % |
| API, results, validation (18 modules) | 3,085 | 42 % |
| Provenance, logging, publication, BIDS, lifecycle (6 modules) | 1,820 | 25 % |

- Functions over 40 lines: 11 of 448 (`prepared._prepare_analysis` 88,
  `trial_encoding.evaluate_trial_encoding` 75, `data.from_arrays` 74,
  `_ridge_cv._score_fold` 64, `single_trial.fit_single_trials` 58,
  `_ridge_cv.score_candidates` 57, `_ridge_cv._provenance` 57,
  `hrf_selection.evaluate_hrf_split` 53, `_hrf_cv.choose_eligible` 49,
  `_selected_hrf_fit.fit_groups` 43, `hrf_selection._select` 42).
- `object.__setattr__` calls: 61 package-wide (ridge_results 15,
  single_trial_results 9, hrf_results 7, hrf_glm_results 5, model 4).
- Cross-module private-name imports: 25 statements in 16 modules; examples
  add 14 more against `boldtailor._*`.
- Lazy in-function imports forming cycles: `fit` ↔ `_hrf_glm`,
  `_hrf_design` ↔ `_single_trial_design`.
- `pytest.raises` with `match=`: ≈171 of 235 (73 %); 21 exact-string
  provenance assertions.
- Modules with no dedicated test file: 9 of 38 (`_conventional`,
  `_hrf_assignment`, `_hrf_glm_design`, `_single_trial_fit`, and five result
  containers).
- `__init__.py`: empty (rule satisfied).

### 3.3 Core numerics (lead reviewer)

**N1 (medium). Result objects retain every (run, HRF) design matrix.**
`_selected_hrf_fit._fit_run` stores `np.column_stack([x, nuisance])` per
`(run, hrf_id)` into `SelectedTrialDesign`; `HrfAnalysisResult` keeps
`group_designs` and deep-copies all of them on every property access. With a
513-candidate library most IDs are selected somewhere, so a 12-run NSD session
retains on the order of 500 × 12 matrices of (T × (trials + confounds)) floats,
roughly a gigabyte, per block result. The NSD examples consume these only to
write provenance (`workflow_analysis.py:89-142`, `nsd_single_trial.py:146`).
Store a design fingerprint plus a lazily rebuildable `(events, hrf, nuisance)`
reference instead, or make retention opt-in.

**N2 (medium). Global caches with unbounded retention.** `_hrf_cv._cached_design`
is a module-level `lru_cache(32)` of `RunDesign` objects keyed on the raw
event/time/nuisance bytes, and `RunDesign.trial_matrix` is an `lru_cache` on an
instance method (256 entries, holding `self` alive). Both are correct but mean
that memory from earlier datasets persists for the process lifetime and that
joblib workers each build their own. Replace with an explicit `DesignCache`
passed through `prepare_runs`, or a `WeakValueDictionary`.

**N3 (low-medium). `_fractional_ridge.prepare_fraction_betas` factorizes twice**
(P5). One SVD of the raw residualized design suffices; derive the normalized
rank check from the raw singular values and column norms.

**N4 (low). `_ridge_cv._score_fold` is 64 lines with four try/except wrappers
that only re-label errors**, and `_provenance` is 57 lines building a dict
with six fingerprints. Move error labelling into `prepare_run_beta_path` and
the activity dict into a small dataclass with a `to_dict`.

**N5 (low). Private nilearn API.** `_hrf_design.py:7` imports
`nilearn.glm.first_level.hemodynamic_models._sample_condition`. It is pinned
by the `<0.15` range, but the comment should say which behaviour is relied on
(grid construction and impulse convention) so an upgrade can be checked.

**N6 (low). Dead/test-only code in production modules.**
`_single_trial_design._trial_column` has zero references;
`_single_trial_fit.trial_beta_path` and `_fractional_ridge.fraction_beta_path`
are used only by tests. Move the latter to `tests/oracles.py`.

### 3.4 API and lifecycle layer

**L1 (medium, persists A4). Three delta-R² lifecycles, three activity schemas.**
`fit.task_delta_r2`, `prepared_fit.task_delta_r2_prepared`, and
`_hrf_glm.selected_task_delta_r2` each fingerprint the parent, validate it,
run two OLS R² fits, apply the nested guard, and `replace(comparison,
_provenance=...)`. `fit._fit_r2_analysis` and `prepared_fit._fit_prepared_r2`
have identical bodies. The activity dicts disagree on keys
(`clip_below_zero` vs `roundoff_tolerance`/`nuisance_rule` vs
`undefined_features`), and `_hrf_glm` and `hrf_glm_results` repeat the
literals `"full_r2 - nuisance_r2"`, `"ols"`, and `-1e-12` instead of the
constants in `_fit_diagnostics`. One `nested_ols_delta(...)` helper and one
schema would remove ~120 lines and give the BIDS projection a single shape.

**L2 (medium, NEW). HRF selection bypasses the lifecycle.** `select_hrf` and
`evaluate_hrf_split` call `extend_provenance` directly with
`events=data.provenance.events`; they emit no `*_started/_completed/_failed`
records and inherit only the parent's event history, unlike every fit entry
point. Wrap them in `fit_operation("hrf_selection", ...)`.

**L3 (medium, persists B1). Per-run loops over copying accessors.**
`AnalysisData.events`/`.confounds` copy every run (including a Python-level
deepcopy of object columns) on each access; `design.compile_nuisance_designs`,
`_ridge_cv.subset_runs`, and `prepare_run_beta_path` index them inside
per-run loops, giving O(n_runs²) copies per fold. Hoist once, or add a
per-run read-only accessor.

**L4 (medium, NEW). Three ownership idioms for frozen results.** `results.py`
uses private fields plus validating factories; five other result modules
rebind public fields through 40 `object.__setattr__` calls in `__post_init__`;
`hrf_glm_results._masked_delta_result` and `hrf_selection._select` construct
results positionally with 8–10 arguments, bypassing the validating factory.
Pick one idiom and set `kw_only=True` on every result dataclass.

**L5 (medium, persists A5). Coupling via private imports and lazy cycles.**
`_hrf_glm` imports `fit._model_provenance` inside functions while `fit`
lazily imports `_hrf_glm`; `_model_provenance` is a pure function of
`ModelSpec` and belongs in `model.py`. `prepared.py` imports 10 privates from
`data`, `logging`, and `provenance` and still duplicates
`_validate_run_count` verbatim. `_fit_lifecycle` and `data` import
`logging._emit_record`/`_make_event`; `publication` imports
`logging._error_code`. Promote the handful of shared helpers to public
names in one module.

**L6 (medium, NEW). Three run-label validators with three rules.**
`single_trial.fit_single_trials` (unique + str), `_single_trial_design._validate_events`
(regex), and `hrf_selection.run_labels_for` (regex + unique + count) yield
different errors for the same bad input depending on entry point. One
`run_labels_for` in `data.py`.

**L7 (medium, NEW). API coherence.** `fit(..., hrf_selection=)` vs
`fit_selected_hrfs(..., selection=)`; `ModelSpec.hrf_model` vs
`fit_single_trials(hrf=)`; `select_hrf` (singular) vs `select_ridge_fractions`
(plural); `task_delta_r2(data, model, full_result)` vs
`task_delta_r2_prepared(prepared, full_result, *, contrasts, noise_model,
model_metadata)` which forces the caller to re-supply the model solely to
recompute a fingerprint already stored in `full_result.provenance`;
`evaluate_hrf_split` hardcodes `candidate_batch_size=32` three times;
`select_ridge_penalty(candidate_r2, alphas)` takes arrays that
`CandidateScores` already pairs. Type annotations present in half the
modules, absent in `hrf_selection`, `_hrf_glm`, `_selected_hrf_fit`,
`ridge_selection`, `fractional_ridge`.

**L8 (low-medium, NEW). `_selected_hrf_fit` passes a 7-tuple by index** and
re-implements trial-table assembly a third time (`_tables`,
`compile_trial_run`, `single_trial._assemble_result`). Return a dataclass;
share one `trial_table()` helper.

**L9 (low). Duplicated scientific constants.** `oversampling=50` is a literal
in four modules while `_hrf_cv.OVERSAMPLING` exists; tie tolerance `1e-12`
appears in three places beside `NESTED_OLS_TOLERANCE`; the bool-exclusion
idiom `isinstance(x, (bool, np.bool_))` appears 12 times while `model.py`
defines private `_is_boolean/_is_integer` helpers nobody else uses.

### 3.5 Provenance, logging, publication, BIDS

Function-length discipline here is good (nothing over 40 lines); the cost is
count: 125 functions, 79 `raise` sites, 44 `isinstance` checks for 1,729
lines. The requirements are written down in
`docs/superpowers/specs/2026-08-07-...md`, but that spec is stale on
publication (still describes fd anchoring and recovery copies) and its
privacy rationale concedes that free-form annotations defeat the heuristic.

**V1 (high, scientific; see S6).** Metadata-only identity with content digests
forbidden; inconsistent/missing software versions; HRF library recorded as a
hash only. Allow an optional caller-supplied `sha256` on `SourceRef`, emit it
as BEP028 `Digest`, record one `software_environment()` (python, platform,
boldtailor, numpy, scipy, pandas, nilearn) from `fit_operation`, and record
library constructor arguments in the selection activity. Rename
`metadata_fingerprint` to something that does not promise content identity.

**V2 (medium, persists). Path-like *value* rejection** (`provenance.py:439-446`)
rejects strings beginning with `~`, `/`, `./`, `../`, `bids:` or containing
`\` anywhere in annotations, so `{"note": "~5 mm smoothing"}` raises, and a
weaker duplicate lives in `prepared_fit.py:250-277`. Validate URIs only;
delete both (~110 lines).

**V3 (medium, NEW). Two incompatible `_validate_relative_path` rules.**
`provenance` accepts any non-traversing POSIX text; `bids_provenance`
additionally requires `[A-Za-z0-9+_.-]+` per component, so a `SourceRef`
accepted at construction fails at projection. One rule in `provenance.py`.

**V4 (medium, NEW). Publication rejects destinations under symlinked ancestors
before resolving**, so `/tmp/...` and `/var/...` fail on macOS (verified:
`_reject_symlink_components(Path('/tmp/x'))` raises). pytest's `tmp_path`
lives under `/private/var`, which is why the suite never sees it. Resolve
first, then reject symlinks only inside the destination.

**V5 (medium, persists). The publisher parses artifact payloads**
(`_validate_metadata` JSON-loads every `.json`, line-parses `.jsonl`, and
width-checks every `.tsv`), runs the destination preflight three times, and
fsyncs directories six times while the docs disclaim power-loss recovery.
Delete payload parsing; keep the under-lock preflight; fsync files only.

**V6 (medium, NEW). BEP028 projection details.** `Command` is filled with a
JSON dump of the activity rather than the entry point; `StartedAtTime`/
`EndedAtTime` are omitted although lifecycle events carry timestamps;
`Environments` is an empty shell; `_safe_json` silently drops keys from the
projection that remain in `record.to_dict()`, so the two "projections of one
record" diverge. `.boldtailor/` control data is written inside the BIDS
derivative root without a `.bidsignore` note, and `publication_failures.jsonl`
grows forever by design.

**V7 (low). Small items.** `_FrozenSequence` is unused; frozen dataclasses are
round-tripped through `to_dict()/from_dict()` three times on construction;
`_validate_schema` says "major version" but does string equality;
`_validate_modified_at` rejects the valid `+00:00` suffix;
`logs/boldtailor_events.jsonl` is an 8-event tail named like a log; the
sequence counter is process-global so joblib workers collide.

A proportionate version of this layer would be roughly 700–800 lines against
1,729 today; the scientific gain comes from V1 alone.

### 3.6 Examples

**E1 (high). Uncommitted notebook state must not be committed as-is.**
`nsd_workflow.ipynb` and `nsd_multisession.ipynb` working copies (6,457
inserted lines) contain executed outputs with 16 distinct
`/Volumes/extdata1/NSD/BIDS/...` strings, replace the env-var contract with
a hardcoded `NSD_CONFIG` default, duplicate paragraphs, and introduce the
incorrect "Both modulators are centered" sentence (S4). Revert the source
edits, clear outputs, keep `notebook_paths` as the only configuration path.
Add `.env` to `.gitignore` and delete `examples/NSD/.env` and
`sobol_hrfs_preview.png`.

**E2 (medium). Committed personal path.** `nsd_cifti.py:26`
`BIDS_ROOT = Path("/Volumes/extdata1/NSD/BIDS")` is the CLI default and is
advertised in README:441. Require `--bids-root` or the env var.

**E3 (medium). Library-grade code and duplication in `examples/NSD`.**
Clearly reusable: `rt_diagnostics.correlate_rt`, `beta_activation`,
`hrf_reliability.hrf_curve_correlations` and
`session_hrf_reliability.compare_hrfs` (same unit-norm curve-correlation lookup
implemented twice), `parallel_blocks.map_blocks`, `nsd_hrf.spatial_signature`,
`single_trial_artifacts.scalar_artifact` (generic dscalar writer),
`workflow_surfaces` (generic CIFTI→fsLR plotting). Duplicated: odd/even run
parity in five modules, RT extraction in four, `dataset_description.json`
block in three, BIDS-label regex in three, three `_map` wrappers, two
artifact-helper sets, `if __package__:` dual-import boilerplate in three.
Promote `expand_events`, `prepare_runs`, `subset_runs`, `fraction_grid`,
`regularization`, `NORM_BASIS`, `r_squared` to public names (14 private
imports today), move numerics/CIFTI I/O into `boldtailor.cifti`,
`boldtailor.reliability`, `boldtailor.parallel`, and retire or thin the older
CLI stack so one stack remains.

**E4 (medium). README:369-370** says the session-reliability notebook's HRF
selection ignores RT and trial type; `session_hrf.estimate_sessions` defaults
`include_rt=True` and selects with the full task model. Also `nsd_hrf._metadata`
and `hrf_artifacts.py:228` still say "mean-stimulus".

**E5 (low). Curve correlations on the 0–36 s grid include the canonical
kernel's zero-padded 32–36 s tail**, slightly inflating all canonical
comparisons; metadata notes it, README does not. `load_block` always records
the full task model even when selection used the reduced one
(`workflow_inputs.py:147-150`).

**E6 (low-medium). Notebook tests `exec()` cells by hardcoded cell id**
(`test_nsd_workflow.py:448,456`, `test_workflow_surfaces.py:180,288`) or by
substring, which is brittle coupling; move cell logic into tested functions.

### 3.7 Tests, CI, documentation

**T1 (medium, NEW). Recovery under realistic noise is untested.**
`test_fractional_ablation_simulation.py:57-74` asserts `selected_fraction == 1`
in a noiseless problem (true for any monotone scorer);
`test_ridge_objective_simulation.py:48-63` asserts only shapes and finiteness.
HRF selection is tested noise-free or near noise-free (AR 0.3, sd 0.015 on
amplitude ≈3). Add seeded tests where the known-optimal fraction is < 1 and
the generating HRF is recovered at plausible SNR, and where canonical wins
when truth is canonical under noise.

**T2 (medium, persists). Missing nilearn oracles.** Multi-run AR(1) contrast
combination has no `FirstLevelModel` oracle (`test_multirun.py:132` is OLS
only); `test_hrf_glm.py:170` re-derives the combination with the same
`(1/3)*(c0+c1+c2)` formula as the implementation. Rank-deficient-but-estimable
designs are checked for warnings only, not for variance/t agreement. Add the
S1 constant-feature test here as well.

**T3 (medium, persists). Error-text pinning.** ≈73 % of `pytest.raises` carry
`match=`; 21 provenance assertions pin literal strings, including a formula
(`test_hrf_selection.py:443`). Keep matches that distinguish behaviours;
assert keys/types otherwise.

**T4 (low-medium). Test hygiene.** `pythonpath=["."]` plus renewed cross-test
imports (`test_hrf_cv` ↔ `test_hrf_selection`; the numerics validation doc's
"no cross-imports remain" is now false); `_complete_sources()` defined five
times; the two-row library built in three places; oracles scattered across
six files despite `development.md:170`; `tests/test_provenance.py:318` personal
path and `Path.home()/gethostname()/cwd()` asserts; a monkeypatch of a
function that no longer exists (`test_fractional_ridge.py:227`); an env-var
test for a variable nothing reads (`test_prepared_fit.py:764`); seven literal
Sobol floats pinned with scipy unpinned above; over-mocking of private names
(`np.linalg.svd` globally, `_conventional.run_glm` ×5). No `slow` marker
although seven files execute notebooks.

**T5 (medium). CI gate is red and partial.** Black fails on 17 files; the
wheel filename is hardcoded; `git diff --check` from `development.md` is not
run; no coverage report despite `pytest-cov`; one OS/Python. Validation docs
state remote CI has never actually executed.

**T6 (docs).** `glmsingle-comparison.md:98` personal path (persists);
README:124 and `glmsingle-comparison.md:68` "mean stimulus response" (S4);
`api.md` should state fixed effects are not precision-weighted (P2);
`documentation-audit-2026-09-28.md` says prepared fits "sanitize separately"
(contradicts `development.md:266`) and lacks the dated banner;
`nsd-fractional-ridge.md:86-88` points to temp audit scripts that no longer
exist. Everything else in `api.md` (signatures, defaults, result fields,
run minimums, R² formula, kernel normalization) was checked against the code
and is accurate.

---

## 4. Proposed changes, by priority

### Tier 1: correctness and reproducibility (do first)

1. **S1** Mask zero-variance features to NaN in all contrast fields; remove
   the warning filter; add tests for OLS and AR(1), single and multi-run,
   conventional, prepared, and selected-HRF paths.
2. **S5** Fix `stop_vs_go` weights, session count/validator/title/subplots,
   and `StartTime` handling in the stop-signal demo; update the pinned test
   strings as a requirement change.
3. **S4 / E1 / E4** Reconcile every description of the task model with
   `workflow_inputs.py`; add the task-model fingerprint to
   `workflow_reuse.ANALYSIS_SETTINGS` and `multisession_inputs._compatible`;
   revert and clear the uncommitted notebooks; gitignore `.env`.
4. **V1 / S6** Optional content digest on `SourceRef`; one
   `software_environment()` recorded by `fit_operation`; library constructor
   args in selection provenance; rename `metadata_fingerprint`.
5. **T5** Run Black, fix the wheel glob, add `git diff --check`, and confirm CI
   actually executes on a push.

### Tier 2: scientific follow-through

6. **P1 / T1** Write the ridge estimand spec; add recovery tests at realistic
   SNR for both fraction selection and HRF selection; only then revisit
   defaults.
7. **S3** Add boundary flags for ridge grids and HRF parameter-box edges; widen
   or document the Sobol box (peak ≤ ~7.5 s).
8. **S2** Make missing-indicator columns candidate-independent or document the
   in-sample component.
9. **P2** Fixed-effects calibration under heteroscedastic runs; state
   non-precision-weighting in `api.md`.
10. **T2** nilearn oracles for multi-run AR(1) and rank-deficient inference.

### Tier 3: architecture

11. **L1** One nested-OLS delta helper and one activity schema.
12. **L2** Route HRF selection through `fit_operation`.
13. **L4** One result-ownership idiom; `kw_only=True`.
14. **L5 / L6 / L7** Move `_model_provenance` to `model.py`; extract shared
    input normalization; one run-label validator; rename `selection=`→
    `hrf_selection=`, `hrf=`→`hrf_model=`; simplify
    `task_delta_r2_prepared(prepared, full_result)`.
15. **N1 / N2 / L3** Stop retaining per-(run, HRF) design matrices in results;
    replace global caches with an explicit cache object; hoist copying
    accessors out of per-run loops.
16. **V2–V6** Delete path-like value heuristics and the `prepared_fit`
    duplicate; unify relative-path rules; resolve before symlink checks; drop
    payload parsing, extra preflights, and directory fsyncs; fix BEP028
    `Command`/timestamps/`Environments`; move control data out of the dataset
    root. Target ~800 lines for the layer.
17. **E3** Promote shared helpers to public names; move numerics and CIFTI I/O
    into the package; collapse the two NSD stacks; de-duplicate parity, RT
    extraction, artifact helpers.

### Tier 4: hygiene

18. **N3 / N6 / L8 / L9** Single SVD in `prepare_fraction_betas`; delete
    `_trial_column`; move `*_beta_path` to `tests/oracles.py`; dataclass
    instead of the 7-tuple; centralize `OVERSAMPLING`/tie tolerance/bool
    checks; fix "Phase 1" text and `_is_integer`.
19. **T3 / T4** Trim error-text matches; centralize fixtures and oracles;
    remove machine-specific asserts and vacuous guards; add a `slow` marker.
20. **T6** Fix the documentation items; add dated banners; delete stale
    branches; note the nilearn yank and the private `_sample_condition`
    dependency in `development.md`.

### Explicit non-changes

- Keep the **nilearn SPM anchor** at candidate 0 and the sum-one kernel
  normalization: deliberate, documented contracts with tests.
- Keep the **AR(1) unwhitened R²** and the OLS-based ΔR²: two diagnostics
  measuring different things, both documented.
- Keep **`encoding_mode="absolute"`**: cheap, pinned, and needed to reproduce
  the September validation records.
- Keep the **BEP028 projection** as long as the stop-signal demo consumes it,
  but fix V6 before adding consumers.
- Keep the **leave-one-run-out HRF objective**: the pooled-amplitude design is
  sound; S2 and S3 are refinements, not replacements.

---

## 5. Bottom line

The science is implemented carefully and, with one exception, correctly.
That exception (S1) is a textbook failure mode that any whole-brain run will
hit at mask edges, and it is hidden by a deliberately silenced warning, so fix
it first and add the test that was believed to exist. The second-largest
risk is drift between the package and its own examples and documentation: the
task model changed on 2026-09-30 and five places still describe or assume the
old one, including the saved-result compatibility check. The provenance layer
is large but still cannot tell a reader which data or which software produced
a result; that is the piece of engineering most worth finishing. After those,
the architecture backlog is real but second-order: a few hundred lines of
duplicated lifecycle code, three ownership idioms, and 6,000 lines of good
code in the wrong directory. Each item above names a file and a change so it
can be planned directly.
