# Voxelwise fractional ridge implementation plan

> **Development history:** This dated plan/design may describe proposed or
> superseded behavior. Use the [documentation index](../../README.md) for current guidance.

> **For agentic workers:** Use superpowers:executing-plans, with tests committed before implementation and one final independent review.

**Goal:** Choose a fractional ridge setting independently at each grayordinate using held-out trial encoding R², retaining nested HRF selection and outer-run evaluation.

**Architecture:** Extend the existing normalized single-trial solver, candidate scorer, and NSD adapter. Fixed-alpha behavior remains available. Fractional selection uses each feature's pooled inner-CV R²; spatial percentiles are descriptive summaries only.

**Tech stack:** Existing NumPy/SciPy SVD and root solving, pandas, CIFTI publication, pytest, uv. No new dependency.

**Spec:** User-approved design in this conversation, reproduced below. Implementation is authorized; this plan records the work without another approval checkpoint.

## Agreed design and constraints

- Evaluate fractions 0.1 through 1.0, select independently per grayordinate.
- Use the existing task intercept + trial_type + response_time encoding objective. Validation targets use the same candidate fraction; OLS is only the reference defining the fraction.
- Define the fraction as the coefficient-norm ratio in the nuisance-projected, unit-L2 trial-design basis already used for ridge. Return native-unit betas. This preserves existing column normalization and nuisance semantics.
- Nuisance parameters remain unpenalized. One selected fraction applies across runs; implied alpha is recomputed per run/feature from its design and response.
- Fractions must satisfy 0 < f <= 1. Fraction 1 reproduces OLS. A zero OLS task norm has undefined fractional shrinkage and is excluded. Public fraction maps may contain NaN to mark excluded features.
- Require finite scores at every candidate for a feature to be eligible. Retain negative R². Ties within 1e-12 favor the largest fraction (least shrinkage). Entirely invalid spatial blocks must return NaNs without aborting valid blocks.
- Inner HRFs and encoding coefficients depend only on training runs. Outer-test BOLD/behavior cannot change training decisions. Alpha conversion for an outer-test run is part of constructing its regularized target, not tuning the selected fraction.
- Maintain canonical/optimized independent tuning; odd→even/even→odd evaluation; separate all-run tuning and refit.
- Preserve the existing alpha-CV, fixed-alpha and OLS APIs. Notebook defaults to fractional_cv; old explicit mode/config overrides remain valid.
- Export selected fractions and per-run implied alphas, candidate maps, outer scores/predictions/targets, and provenance linked to the selection decision. Use distinct FractionalCV descriptors.
- Conventional GLMs and the across-session HRF reliability notebook remain unchanged. Preserve user edits. All __init__.py files remain empty. Use uv and pytest RED→commit tests→GREEN.

## Review focus

1. Response scaling and heterogeneous HRFs/designs must not turn one selected fraction into a shared alpha. Test with analytically distinct signal directions and unequal design-column norms.
2. Undefined task response or incomplete candidates must not silently select a default fraction or abort a whole-brain block. Test mixed valid/invalid and entirely invalid blocks.
3. Held-out target construction depends on that run's BOLD, but selected fractions, training HRFs, and encoding coefficients must not. Perturb each outer half.
4. Final and outer provenance must identify fraction maps, candidate grids, predictor/fold identity, and per-run alpha maps. Test distinct settings with the same selected fractions.
5. Notebook/API compatibility: CV fraction defaults must not reinterpret fixed-alpha overrides or old alpha-CV calls. Execute fractional, alpha, fixed and off modes and preserve user outputs unrelated to this analysis.

## Task 1: Fractional solver and single-trial fitting

**Files:** new `_fractional_ridge.py`; extend `single_trial.py`, `_selected_hrf_fit.py`, result dataclasses; new `tests/test_fractional_ridge.py`.

**Interfaces:** `fraction_beta_path(x,nuisance,y,*,fractions)` yields `(fraction,betas,alphas)`; `fit_single_trials(...,ridge_fraction=None)` and `fit_selected_hrfs(...,ridge_fraction=None)` accept scalar or feature map. Results expose immutable `ridge_fraction` and `run_ridge_alphas`; fractional fits have `ridge_alpha=None`.

- [ ] Write numerical tests using independently nuisance-projected designs and augmented least-squares solutions/root finding. Assert norm ratios, OLS equivalence, response-scale invariance, different alpha per target/run, unpenalized confounds, beta units, invalid values and undefined features. Exercise selected HRFs and source identity. Run to RED and commit.
- [ ] Reuse the projected-design SVD. Solve monotonic coefficient-norm ratios by vectorized bisection with guaranteed spectral bounds. Do not penalize confounds or rescale betas post hoc. Propagate maps/provenance and leave scalar-alpha fits unchanged.
- [ ] Run new solver tests and existing single-trial/selected-HRF regressions to GREEN, format and commit.

## Task 2: Per-grayordinate fractional CV

**Files:** `fractional_ridge.py`, `ridge_results.py`, `_ridge_cv.py`, new `tests/test_fractional_cv.py`.

**Interfaces:** `score_fraction_candidates(data,predictors,*,fractions,library=None,run_labels=None,feature_signature=None)` returns fraction-grid scores; `select_ridge_fractions(candidate_r2,fractions,*,feature_mask=None)` returns fraction map, selected R², indices and common eligible mask.

- [ ] Write tests with different optimum fractions across features, negative/tied/missing scores, descending grid, immutable maps, same-fraction held-out targets, inner HRF isolation and pooled losses. An independent oracle must reproduce the complete nested scorer. Run RED and commit.
- [ ] Reuse the existing run-wise scorer with an explicit fractional fitting path and correctly identified provenance. Select each eligible feature independently; use NaN/-1 for undefined selections.
- [ ] Run fractional/alpha CV and encoding regressions to GREEN, format and commit.

## Task 3: NSD workflow and artifacts

**Files:** NSD ridge workflow/provenance/output helpers, workflow_analysis.py, new `test_fractional_workflow.py` and output tests.

**Interface:** `fit_cv_beta_series(...,alphas=None,fractions=None)` takes exactly one grid. Fractional results retain the final/tuning/evaluation layout, with fraction maps and per-run alpha maps.

- [ ] Add six-run CIFTI tests for whole-array equivalence, block/parallel invariance, outer perturbation isolation, missing RT alignment, all-invalid blocks, map export axes/NaNs, and durable final tuning links. Run RED and commit.
- [ ] Extend the adapter and publication paths using per-block slices of the selected map. Keep inner scores distinct from outer evaluation; export maps/metadata with explicit fraction meaning. Generalize plot/table helpers to fractional summaries without claiming a global selected fraction.
- [ ] Run old/new NSD adapter tests to GREEN, format and commit.

## Task 4: Notebook, docs and verification

**Files:** `nsd_workflow.ipynb`, notebook regression tests, README/user guide/API/comparison docs and validation report.

- [ ] Add executed notebook tests for fractional default and explicit fractional mode, with a short grid/small library. Retain old fixed/off/alpha behavior. Run RED and commit.
- [ ] Update notebook cells/settings, labels, fraction distributions and exports. Rewrite the ridge documentation to distinguish voxelwise fractions from legacy global-alpha selection. Clear only affected analysis outputs and preserve unrelated edits.
- [ ] Run the full suite with warnings as errors; Black touched Python files, check empty __init__ and git diff whitespace. Include preserved existing stop-signal fixes when validating the combined tree.
- [ ] Run all 12 NSD ses10 runs, the full 513-HRF library, and eight grayordinates with three fractions. Independently verify fraction norms, different implied alphas, candidate scores, outer predictions, saved fraction/alpha maps and final betas. Keep production defaults independent of this smoke's scores.
- [ ] One fresh whole-branch reviewer; fix substantive findings with RED→GREEN. Integrate into the user's checkout while preserving notebook outputs unrelated to ridge. Record limits: no whole-brain performance claim.
