# Code-review guide

[API reference](api.md) · [User guide](user-guide.md) · [Developer guide](development.md) · [Testing guide](testing.md) · [Archived function inventory](archive/api-review/README.md)

This guide maps Boldtailor's workflow from input arrays to statistical estimates
and published files. Use it to trace each subsystem through `src/boldtailor/`
and its test suite.

## Core objects across module boundaries

| Object | Module | Role and contract |
| --- | --- | --- |
| `AnalysisData` | `boldtailor.data` | Aligned runs (`scans × features`), event DataFrames (`onset`, `duration` in seconds), confounds, frame times, and provenance |
| `ModelSpec` | `boldtailor.model` | Conventional GLM settings (contrasts, confounds, drift model, HRF model, noise model, optional `TaskModel`) |
| `TaskModel`, `Modulator` | `boldtailor.model` | Shared task-regressor specification for HRF selection and task-model GLMs |
| `PreparedDesignAnalysis` | `boldtailor.prepared` | Signals paired with caller-supplied design matrices and column roles (`task`, `nuisance`, `intercept`, `other`) |
| `HrfLibrary`, `HrfCandidate` | `boldtailor.hrf_library` | Candidate HRF kernels (ID 0 is canonical SPM; all kernels peak at 1) |
| `HrfSelectionResult`, `HrfEvaluationResult` | `boldtailor.hrf_results` | Per-feature HRF choices (`method` `"cv"` or `"reml"`), CV or REML diagnostics, and split-test evaluations |
| `AnalysisResult`, `HrfAnalysisResult` | `boldtailor.results`, `boldtailor.hrf_glm_results` | Conventional contrast maps and $R^2$; selected-HRF designs rebuild via `rebuild_group_design(run, hrf_id)` |
| `SingleTrialResult` | `boldtailor.single_trial_results` | Per-trial betas (`trials × features` per run), `trial_table`, and `SharedTrialDesign` or `SelectedTrialDesign` |
| `CandidateScores`, `RidgeSelection`, `FractionSelection` | `boldtailor.ridge_results` | Cross-validated encoding scores (`candidates × features`) and chosen penalties |
| `EncodingDesign`, `TrialEncodingResult` | `boldtailor.trial_encoding` | Within-run or absolute trial-encoding design and held-out prediction results |
| `RemlGroup`, `RemlFit`, `RemlSelection` | `boldtailor.reml` | Per-group fixed/random designs and per-feature restricted-likelihood variance estimates |
| `DenoisingResult`, `SignificanceGate`, `PoolMasks` | `boldtailor.denoising_results` | Noise-pool masks, run PCs, `pcstop` count, and F-test significance gate |
| `ProvenanceRecord`, `Artifact` | `boldtailor.provenance`, `boldtailor.publication` | Content/analysis fingerprints and in-memory output payloads |

All returned numeric arrays are read-only NumPy arrays; public table accessors
return copies; and dataclasses holding arrays or tables compare by object
identity (`eq=False`).

## Review routes

### 1. Conventional GLMs and prepared designs

| Modules | Key functions and classes |
| --- | --- |
| [`data.py`](../src/boldtailor/data.py), [`_run_inputs.py`](../src/boldtailor/_run_inputs.py), [`model.py`](../src/boldtailor/model.py) | `from_arrays`, `AnalysisData`, `ModelSpec`, `TaskModel`, `Modulator` |
| [`design.py`](../src/boldtailor/design.py), [`_task_design.py`](../src/boldtailor/_task_design.py) | `compile_designs`, `compile_nuisance_designs`, `kernel_task_columns` |
| [`fit.py`](../src/boldtailor/fit.py), [`_standard_glm.py`](../src/boldtailor/_standard_glm.py), [`_conventional.py`](../src/boldtailor/_conventional.py) | `fit`, `task_delta_r2`, `fit_designs`, `nested_ols_delta` |
| [`_fit_diagnostics.py`](../src/boldtailor/_fit_diagnostics.py), [`results.py`](../src/boldtailor/results.py) | `AnalysisResult`, `TaskDeltaR2Result`, parent-analysis validation |
| [`prepared.py`](../src/boldtailor/prepared.py), [`prepared_fit.py`](../src/boldtailor/prepared_fit.py) | `PreparedDesignAnalysis.from_arrays`, `fit_prepared`, `task_delta_r2_prepared` |

**What to check:**
- Multi-run contrasts use equal-weight fixed effects, not precision weighting.
- Rank and estimability checks normalize nonzero design columns to unit L2 norm
  before SVD thresholding (`_conventional._full_rank_basis`).
- `task_delta_r2` refits nested OLS models even when the parent fit used AR(1),
  and verifies parent `data_fingerprint` and `analysis_fingerprint`.
- Tests: `tests/test_fit.py`, `tests/test_multirun.py`,
  `tests/test_prepared_fit.py`, `tests/test_fit_diagnostics.py`,
  `tests/test_conventional_properties.py`.

### 2. HRF libraries, CV/REML selection, and selected-HRF GLMs

| Modules | Key functions and classes |
| --- | --- |
| [`hrf_library.py`](../src/boldtailor/hrf_library.py) | `default_hrf_library`, `timing_hrf_library`, `sobol_hrf_library`, `expanded_hrf_library`, `glmsingle_hrf_library`, `spm_parameters_from_realized` |
| [`hrf_selection.py`](../src/boldtailor/hrf_selection.py), [`hrf_options.py`](../src/boldtailor/hrf_options.py), [`_hrf_design.py`](../src/boldtailor/_hrf_design.py), [`_run_design.py`](../src/boldtailor/_run_design.py) | `select_hrfs`, `evaluate_hrf_split`, `RemlHrfOptions`, `RunDesign`, `prepare_runs` |
| [`_hrf_cv.py`](../src/boldtailor/_hrf_cv.py), [`_hrf_reml.py`](../src/boldtailor/_hrf_reml.py), [`hrf_results.py`](../src/boldtailor/hrf_results.py) | `loro_scores`, `choose_eligible`, `select_hrfs_reml`, `HrfSelectionResult`, `HrfEvaluationResult` |
| [`_hrf_glm_design.py`](../src/boldtailor/_hrf_glm_design.py), [`_hrf_glm.py`](../src/boldtailor/_hrf_glm.py), [`hrf_glm_results.py`](../src/boldtailor/hrf_glm_results.py) | `fit_selected_glm`, `selected_task_delta_r2`, `HrfAnalysisResult.rebuild_group_design` |
| [`reliability.py`](../src/boldtailor/reliability.py) | `curve_correlations`, `compare_hrfs` |

**What to check:**
- Every event's predicted response on the oversampled grid is scaled to unit peak.
- `evaluate_hrf_split` learns HRFs and task amplitudes strictly on `train_runs`.
- Selected-HRF `fit()` validates `feature_signature`, `oversampling`,
  `min_onset`, and `TaskModel.is_subset_of`.
- Tests: `tests/test_hrf_cv.py`, `tests/test_hrf_cv_properties.py`,
  `tests/test_hrf_selection.py`, `tests/test_hrf_reml*.py`,
  `tests/test_hrf_glm.py`, `tests/test_hrf_timing.py`.

### 3. Single-trial estimation and encoding-guided regularization

| Modules | Key functions and classes |
| --- | --- |
| [`single_trial.py`](../src/boldtailor/single_trial.py), [`_shared_hrf_fit.py`](../src/boldtailor/_shared_hrf_fit.py), [`_selected_hrf_fit.py`](../src/boldtailor/_selected_hrf_fit.py) | `fit_single_trials`, `fit_selected_hrfs`, `assemble_trial_result` |
| [`_design_inputs.py`](../src/boldtailor/_design_inputs.py), [`_single_trial_design.py`](../src/boldtailor/_single_trial_design.py), [`_single_trial_fit.py`](../src/boldtailor/_single_trial_fit.py), [`_fractional_ridge.py`](../src/boldtailor/_fractional_ridge.py) | `ProjectedTrialDesign`, `PreparedTrialBetas`, `PreparedFractionBetas`, `fraction_alphas` |
| [`single_trial_results.py`](../src/boldtailor/single_trial_results.py), [`ridge_results.py`](../src/boldtailor/ridge_results.py) | `SingleTrialResult`, `SharedTrialDesign`, `SelectedTrialDesign`, `CandidateScores` |
| [`trial_encoding.py`](../src/boldtailor/trial_encoding.py) | `trial_predictors`, `EncodingDesign`, `evaluate_trial_encoding` |
| [`ridge_selection.py`](../src/boldtailor/ridge_selection.py), [`fractional_ridge.py`](../src/boldtailor/fractional_ridge.py), [`_regularizers.py`](../src/boldtailor/_regularizers.py), [`_ridge_cv.py`](../src/boldtailor/_ridge_cv.py) | `score_ridge_candidates`, `select_ridge_penalty`, `score_fraction_candidates`, `select_ridge_fractions` |
| [`diagnostics.py`](../src/boldtailor/diagnostics.py) | `one_sample_t`, `correlate_rt`, `even_run_points` |

**What to check:**
- Nuisance columns and the intercept are unpenalized.
- `ridge_alpha` acts on unit-norm projected trial columns; `ridge_fraction`
  constrains coefficient norm in the raw projected trial basis and scores
  against fixed OLS validation betas.
- Within-run encoding centers test residuals for scoring only; test betas never
  influence fitted slopes or predictions.
- Tests: `tests/test_single_trial.py`, `tests/test_fractional_ridge.py`,
  `tests/test_fractional_cv.py`, `tests/test_ridge_cv.py`,
  `tests/test_ridge_properties.py`, `tests/test_trial_encoding*.py`,
  `tests/test_within_run_cv_isolation.py`.

### 4. Scalar REML and AR(1)

| Modules | Key functions and classes |
| --- | --- |
| [`ar1.py`](../src/boldtailor/ar1.py), [`_ar1_moments.py`](../src/boldtailor/_ar1_moments.py) | `whiten`, `prepare_reference`, `estimate_rho` |
| [`reml.py`](../src/boldtailor/reml.py) | `RemlGroup`, `fit_reml`, `select_random_effects_design`, `RemlFit`, `RemlSelection` |
| [`_numerics.py`](../src/boldtailor/_numerics.py) | `conditioned_basis`, `residual_floors`, `rank_tolerance` |

**What to check:**
- `rho` is supplied to `fit_reml` / `select_random_effects_design`, not
  estimated inside REML.
- Interpret `ratio`, `noise_variance`, and `effect_variance` together with
  `status`, `at_upper`, `unrefined`, `any_at_upper`, and `any_unrefined`.
- Tests: `tests/test_reml*.py`, `tests/test_ar1.py`, `tests/test_numerics*.py`.

### 5. Optional task-guided denoising

| Modules | Key functions and classes |
| --- | --- |
| [`denoising.py`](../src/boldtailor/denoising.py), [`_mixture_threshold.py`](../src/boldtailor/_mixture_threshold.py) | `select_denoising`, `with_denoising`, `onoff_r2`, `pool_masks`, `apply_gate`, `mixture_threshold` |
| [`denoising_results.py`](../src/boldtailor/denoising_results.py) | `DenoisingResult`, `SignificanceGate`, `PoolMasks`, `PcaDiagnostics`, `DenoisingFold` |

**What to check:**
- Held-out targets are projected off baseline confounds (and indicators) only,
  never off candidate PCs.
- The significance gate (`gate_noise_model="ar1"` by default) prewhitens each
  run and feature before computing nested F-tests across all runs.
- `with_denoising` enforces exact run, timing, signal, confound, and event
  identity before appending `denoise_pc_000`, ...
- Tests: `tests/test_denoising*.py`, `tests/test_mixture_threshold.py`.

### 6. Ownership, provenance, lifecycle, and publication

| Modules | Key functions and classes |
| --- | --- |
| [`_arrays.py`](../src/boldtailor/_arrays.py), [`_scalars.py`](../src/boldtailor/_scalars.py) | `readonly_array`, `owned_table`, scalar validators |
| [`provenance.py`](../src/boldtailor/provenance.py), [`bids_provenance.py`](../src/boldtailor/bids_provenance.py) | `SourceRef`, `RunSources`, `ProvenanceRecord`, `analysis_fingerprint`, `project_bids_provenance` |
| [`logging.py`](../src/boldtailor/logging.py), [`_fit_lifecycle.py`](../src/boldtailor/_fit_lifecycle.py) | `bind_context`, `emit_event`, `normalization`, `fit_operation`, `build_fit_provenance`, `finish_fit` |
| [`publication.py`](../src/boldtailor/publication.py) | `Artifact`, `publish_artifact_set`, `PublicationError` |

**What to check:**
- `data_fingerprint` and `analysis_fingerprint` are used consistently across
  attributes, log records, and serialized JSON.
- Failure events record `error_code` without leaking exception text.
- Tests: `tests/test_arrays.py`, `tests/test_dataclass_equality.py`,
  `tests/test_provenance.py`, `tests/test_identity_names.py`,
  `tests/test_fit_lifecycle.py`, `tests/test_publication*.py`.

### 7. BIDS workflow, CIFTI outputs, and CLI

| Modules | Key functions and classes |
| --- | --- |
| [`cli.py`](../src/boldtailor/cli.py), [`workflow/settings.py`](../src/boldtailor/workflow/settings.py), [`workflow/run.py`](../src/boldtailor/workflow/run.py) | `main`, `WorkflowSettings`, `describe_inputs`, `run_workflow` |
| [`workflow/files.py`](../src/boldtailor/workflow/files.py), [`workflow/inputs.py`](../src/boldtailor/workflow/inputs.py), [`task_detection.py`](../src/boldtailor/task_detection.py) | `discover_runs`, `load_session`, `select_confounds`, `make_blocks`, `detect_task_model` |
| [`workflow/analysis.py`](../src/boldtailor/workflow/analysis.py), [`workflow/beta_series.py`](../src/boldtailor/workflow/beta_series.py), [`workflow/records.py`](../src/boldtailor/workflow/records.py), [`workflow/ridge_provenance.py`](../src/boldtailor/workflow/ridge_provenance.py) | `select_hrfs`, `fit_glms`, `fit_beta_models`, `GlmFit`, `BetaFit`, `OuterEvaluation`, `TuningResult`, `CvBetaResult` |
| [`cifti.py`](../src/boldtailor/cifti.py), [`workflow/hrf_maps.py`](../src/boldtailor/workflow/hrf_maps.py), [`workflow/artifacts.py`](../src/boldtailor/workflow/artifacts.py), [`workflow/export_cifti.py`](../src/boldtailor/workflow/export_cifti.py), [`workflow/outputs.py`](../src/boldtailor/workflow/outputs.py), [`workflow/manifest.py`](../src/boldtailor/workflow/manifest.py) | `scalar_artifact`, `selection_maps`, `workflow_artifacts`, `save_workflow`, `metadata` |
| [`workflow/summaries.py`](../src/boldtailor/workflow/summaries.py), [`workflow/tables.py`](../src/boldtailor/workflow/tables.py), [`workflow/figures.py`](../src/boldtailor/workflow/figures.py), [`workflow/plots.py`](../src/boldtailor/workflow/plots.py), [`workflow/surfaces.py`](../src/boldtailor/workflow/surfaces.py), [`workflow/report.py`](../src/boldtailor/workflow/report.py) | Summary tables, cortical/diagnostic figures, and `render_report` |

**What to check:**
- Inner ridge-tuning folds, outer evaluations, and final fits share the session's
  non-constant grayordinate blocks and HRF selection `TaskModel`.
- Excluded constant grayordinates are restored as NaN in exported maps.
- Tests: `tests/workflow/test_synthetic_bids.py`,
  `tests/workflow/test_analysis.py`, `tests/workflow/test_beta_series.py`,
  `tests/workflow/test_outputs.py`, `tests/workflow/test_reml_workflow.py`.

## Runnable examples

Run these snippets in a Python session (`uv run python`):

### 1. Conventional GLM contrast

```python
import numpy as np
import pandas as pd

from boldtailor import ModelSpec, fit, from_arrays

rng = np.random.default_rng(7)
events = pd.DataFrame({
    "onset": [10., 30., 50., 70., 90., 110.],
    "duration": [1.] * 6,
    "trial_type": ["face", "house"] * 3,
})
signals = [100 + rng.normal(size=(80, 3)) for _ in range(3)]
data = from_arrays(signals, [events.copy() for _ in signals], tr=2.0)
model = ModelSpec(
    contrasts={"face_gt_house": {"face": 1., "house": -1.}},
    hrf_model="spm", noise_model="ols",
)
result = fit(data, model)
print(result.effect("face_gt_house").shape)  # (3,)
print(result.design_matrices[0].columns)
```

### 2. HRF selection and single-trial fitting

```python
from boldtailor.hrf_library import HrfLibrary
from boldtailor.hrf_selection import select_hrfs
from boldtailor.single_trial import fit_selected_hrfs

library = HrfLibrary.from_parameters([], origin={"kind": "review_example"})
selection = select_hrfs(data, library=library, feature_signature="example-order")
trials = fit_selected_hrfs(
    data, hrf_selection=selection, feature_signature="example-order",
    ridge_fraction=0.5,
)
print(trials.run_betas[0].shape)  # (6, 3)
print(trials.trial_table[["trial_id", "run_index", "event_index"]])
```

### 3. Fractional ridge selection

```python
from boldtailor.fractional_ridge import (
    score_fraction_candidates, select_ridge_fractions,
)

predictors = [pd.DataFrame({"face": [1., 0.] * 3}) for _ in signals]
scores = score_fraction_candidates(data, predictors, fractions=[1., 0.5])
chosen = select_ridge_fractions(scores)
print(scores.cv_r2.shape)  # (2, 3)
print(chosen.ridge_fraction)
```

### 4. Scalar REML and AR(1) estimation

```python
from boldtailor.ar1 import estimate_rho
from boldtailor.reml import RemlGroup, fit_reml, select_random_effects_design

fixed = [np.ones((y.shape[0], 1)) for y in signals]
random = [rng.normal(size=(y.shape[0], 4)) for y in signals]
groups = [RemlGroup(y, c) for y, c in zip(signals, fixed, strict=True)]
rho = estimate_rho(signals, fixed)
variance_fit = fit_reml(groups, random, rho=rho.pooled)
print(variance_fit.status)
print(variance_fit.ratio.shape)  # (3,)
comparison = select_random_effects_design(groups, [random], rho=rho.pooled)
print(comparison.winner)
```
